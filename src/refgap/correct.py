"""Reference-logit correction from Wang et al., arXiv:2609.35708, sections 3.2-3.4.

The caller passes pre-softmax attention logits and region masks for one layer.
This module does not run a diffusion model and does not read images.

Logit shapes:
  (queries, keys)
  (heads, queries, keys), heads on axis 0

There is no batch axis. Masks are hard partitions: reference_mask over keys,
edit_mask and keep_mask over queries. Queries in neither mask are left
unchanged. A query marked in both masks takes the edit offset. Reference-token
queries, if they share the query axis, should be False in both query masks;
this module does not infer that from the key mask (cross-attention and
self-attention index spaces need not match).

Layer selection is separate. apply_correction always corrects the logits it is
given. The caller applies it only on layers returned by select_layers.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

# Deployed coefficients, section 4.2. Fixed, not swept.
GAMMA_EDIT = 0.6
GAMMA_KEEP = 0.8

# Layer-selection thresholds and the ratio stabilizer in eq. (6).
TAU_B = 2.7
TAU_B_BAR = 3.7
EPS_B = 1e-6

# Appendix A floors both region masses at 1e-6 before the bias logarithm.
# Softmax masses cannot exceed 1, so there is no upper clamp.
RHO_FLOOR = 1e-6
# Clip only the edit-side bias. The keep-side bias is not clipped.
EDIT_BIAS_CLIP_MIN = 0.0
EDIT_BIAS_CLIP_MAX = 8.0


class ReferenceMass(NamedTuple):
    """Per-query reference mass and the region averages in eq. (2).

    rho: shape (queries,) or (heads, queries,). Fraction of stable-softmax
        mass on reference keys, eq. (1).
    rho_bar_edit: mean of rho over edit queries and over heads. nan if the
        edit mask is missing or empty.
    rho_bar_keep: mean of rho over keep queries and over heads. nan if the
        keep mask is missing or empty.
    gap: rho_bar_edit - rho_bar_keep. nan if either region statistic is nan.
        This is an unnormalized edit-keep difference, not the token-count
        normalized diagnostic u_g in eq. (3). Eq. (3) is not used for the
        correction.
    """

    rho: jnp.ndarray
    rho_bar_edit: jnp.ndarray
    rho_bar_keep: jnp.ndarray
    gap: jnp.ndarray


def reference_mass(logits, reference_mask, edit_mask=None, keep_mask=None):
    """Reference attention mass for one layer, eq. (1) and eq. (2).

    logits: pre-softmax scores, (queries, keys) or (heads, queries, keys).
    reference_mask: boolean mask over keys, shape (keys,).
    edit_mask, keep_mask: optional boolean masks over queries, shape (queries,).
        Omit a mask, or pass an empty one, to get nan for that region mean.
        An empty region does not invent a finite statistic.
    """

    logits = _as_logits(logits)
    n_queries, n_keys = _query_key_sizes(logits)
    reference_mask = _as_mask(reference_mask, n_keys, "reference_mask")
    rho = _per_query_mass(logits, reference_mask)
    rho_bar_edit = _region_mean(rho, _optional_mask(edit_mask, n_queries, "edit_mask"))
    rho_bar_keep = _region_mean(rho, _optional_mask(keep_mask, n_queries, "keep_mask"))
    return ReferenceMass(
        rho=rho,
        rho_bar_edit=rho_bar_edit,
        rho_bar_keep=rho_bar_keep,
        gap=rho_bar_edit - rho_bar_keep,
    )


def apply_correction(logits, edit_mask, keep_mask, reference_mask):
    """Add the eq. (5) offset to reference-key logits.

    Returns corrected pre-softmax logits, same shape as logits. Rest-key
    logits are unchanged. Every reference key of a query receives the same
    offset, so relative weights inside the reference group and inside the
    rest group stay the same.

    The offset uses the deployed coefficients gamma_edit = 0.6 and
    gamma_keep = 0.8. It is recomputed from these logits (the online rule).
    It does not depend on the layer-selection thresholds. Appendix A floors
    both region masses at 1e-6 before the logarithm and clips the edit-side
    bias to [0, 8]. The keep-side bias is not clipped.

    An empty edit or keep region yields nan for that region's mass and does
    not write an offset for it. The other region is still corrected from its
    own mass.
    """

    logits = _as_logits(logits)
    n_queries, n_keys = _query_key_sizes(logits)
    edit_mask = _as_mask(edit_mask, n_queries, "edit_mask")
    keep_mask = _as_mask(keep_mask, n_queries, "keep_mask")
    reference_mask = _as_mask(reference_mask, n_keys, "reference_mask")

    mass = reference_mass(logits, reference_mask, edit_mask, keep_mask)
    b_edit = _region_bias(mass.rho_bar_edit, GAMMA_EDIT, sign=1.0)
    b_keep = _region_bias(mass.rho_bar_keep, GAMMA_KEEP, sign=-1.0)
    b_edit = jnp.clip(
        b_edit,
        jnp.asarray(EDIT_BIAS_CLIP_MIN, dtype=b_edit.dtype),
        jnp.asarray(EDIT_BIAS_CLIP_MAX, dtype=b_edit.dtype),
    )
    query_offset = _query_offsets(edit_mask, keep_mask, b_edit, b_keep, logits.dtype)
    return _add_reference_offset(logits, query_offset, reference_mask)


def select_layers(rho_edit, rho_keep):
    """Layer indices to correct, eq. (6)-(8).

    rho_edit and rho_keep are per-layer region masses already averaged over
    denoising steps and heads. Length L, one scalar per layer. They are not
    recomputed here, and gamma is not used.

    The paper's set is 1-indexed. Returned indices are 0-indexed:
      second half if B >= 2.7 or b_bar_edit >= 3.7: floor(L/2) .. L-1
      otherwise: 0 .. L-1

    For L = 25 the second half is 12 .. 24, matching the appendix deployment
    on FLUX.2-klein. This function runs eagerly on concrete profiles; the
    selected length depends on the statistics.
    """

    rho_edit = _as_profile(rho_edit, "rho_edit")
    rho_keep = _as_profile(rho_keep, "rho_keep")
    if rho_edit.shape != rho_keep.shape:
        raise ValueError(
            "rho_edit and rho_keep must have the same shape, "
            f"got {tuple(rho_edit.shape)} and {tuple(rho_keep.shape)}"
        )

    length = int(rho_edit.shape[0])
    if length == 0:
        return ()

    # eq. (6). eps_B stabilizes only this ratio, not the attention biases.
    ratio = rho_edit / (rho_keep + jnp.asarray(EPS_B, dtype=rho_edit.dtype))
    selectivity = float(jnp.max(ratio))
    # eq. (7). Natural log. No rho clamp: layer profiles are the paper's inputs.
    log_inverse = float(jnp.mean(jnp.log(1.0 / rho_edit)))
    second_half = (selectivity >= TAU_B) or (log_inverse >= TAU_B_BAR)
    start = length // 2 if second_half else 0
    return tuple(range(start, length))


def _as_logits(logits):
    logits = jnp.asarray(logits)
    if logits.ndim not in (2, 3):
        raise ValueError(
            "logits must have shape (queries, keys) or (heads, queries, keys), "
            f"got {tuple(logits.shape)}"
        )
    if not jnp.issubdtype(logits.dtype, jnp.floating):
        logits = logits.astype(jnp.float32)
    return logits


def _query_key_sizes(logits):
    return int(logits.shape[-2]), int(logits.shape[-1])


def _as_mask(mask, size, name):
    mask = jnp.asarray(mask)
    if mask.ndim != 1 or int(mask.shape[0]) != size:
        raise ValueError(f"{name} must have shape ({size},), got {tuple(mask.shape)}")
    return mask.astype(jnp.bool_)


def _optional_mask(mask, size, name):
    if mask is None:
        return None
    return _as_mask(mask, size, name)


def _as_profile(values, name):
    values = jnp.asarray(values)
    if values.ndim != 1:
        raise ValueError(f"{name} must be 1D of length L, got {tuple(values.shape)}")
    if not jnp.issubdtype(values.dtype, jnp.floating):
        values = values.astype(jnp.float32)
    return values


def _per_query_mass(logits, reference_mask):
    """Stable softmax, then sum probabilities on reference keys.

    Equivalent to the exp ratio in eq. (1), but the max is subtracted first so
    large logits do not overflow. The key axis is last.
    """

    peak = jnp.max(logits, axis=-1, keepdims=True)
    weights = jnp.exp(logits - peak)
    probs = weights / jnp.sum(weights, axis=-1, keepdims=True)
    ref = reference_mask.astype(probs.dtype)
    return jnp.sum(probs * ref, axis=-1)


def _region_mean(rho, query_mask):
    """Mean of rho over the masked queries and over heads. nan if empty."""

    if query_mask is None:
        return jnp.array(jnp.nan, dtype=rho.dtype)
    weights = query_mask.astype(rho.dtype)
    if rho.ndim == 1:
        numer = jnp.sum(rho * weights)
        denom = jnp.sum(weights)
    else:
        numer = jnp.sum(rho * weights[None, :])
        denom = jnp.asarray(rho.shape[0], dtype=rho.dtype) * jnp.sum(weights)
    safe_denom = jnp.where(denom > 0, denom, jnp.ones((), dtype=rho.dtype))
    mean = numer / safe_denom
    return jnp.where(denom > 0, mean, jnp.array(jnp.nan, dtype=rho.dtype))


def _region_bias(rho_bar, gamma, sign):
    """Eq. (5) for one region. Floor mass at 1e-6. nan mass stays nan."""

    dtype = rho_bar.dtype
    floored = jnp.maximum(rho_bar, jnp.asarray(RHO_FLOOR, dtype=dtype))
    bias = (
        jnp.asarray(sign, dtype=dtype)
        * jnp.asarray(gamma, dtype=dtype)
        * jnp.log(1.0 / floored)
    )
    return jnp.where(jnp.isfinite(rho_bar), bias, jnp.array(jnp.nan, dtype=dtype))


def _query_offsets(edit_mask, keep_mask, b_edit, b_keep, dtype):
    """Per-query offset. Edit wins if a query is in both masks.

    Non-finite region biases are replaced with 0 before the multiply. A 0 mask
    times a nan bias would otherwise become nan and leak into other queries.
    """

    edit = edit_mask.astype(dtype)
    keep = keep_mask.astype(dtype) * (jnp.asarray(1.0, dtype=dtype) - edit)
    b_edit = jnp.where(jnp.isfinite(b_edit), b_edit, jnp.array(0.0, dtype=dtype))
    b_keep = jnp.where(jnp.isfinite(b_keep), b_keep, jnp.array(0.0, dtype=dtype))
    return edit * b_edit.astype(dtype) + keep * b_keep.astype(dtype)


def _add_reference_offset(logits, query_offset, reference_mask):
    ref = reference_mask.astype(logits.dtype)
    offset = query_offset.astype(logits.dtype)
    if logits.ndim == 2:
        return logits + offset[:, None] * ref[None, :]
    return logits + offset[None, :, None] * ref[None, None, :]
