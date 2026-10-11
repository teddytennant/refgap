"""Hand-computed checks for the RefGAP logit correction.

Expected numbers are literals from the formulas, not a second call into the
implementation. Logarithms below are natural logs.
"""

import jax.numpy as jnp
import pytest

from refgap import apply_correction, reference_mass, select_layers

# 0.6 * ln(3/2) and -0.8 * ln(3/2), for uniform logits with rho = 2/3.
EDIT_OFFSET = 0.24327906486489861
KEEP_OFFSET = -0.32437208648653154

# exp(-1), exp(-0.75): within-group softmax ratios for the non-uniform case.
REF_SOFTMAX_RATIO = 0.36787944117144233
REST_SOFTMAX_RATIO = 0.4723665527410147

# 0.6 * ln(1e12) = 0.6 * 12 * ln(10). Offset after clamping rho to 1e-12.
CLAMPED_EDIT_OFFSET = 16.578612669557128


def _uniform_masks():
    edit_mask = jnp.array([True, False])
    keep_mask = jnp.array([False, True])
    reference_mask = jnp.array([True, True, False])
    return edit_mask, keep_mask, reference_mask


def test_uniform_logits_reference_mass_is_two_thirds():
    # Two reference keys and one rest key, equal logits. Each key has softmax
    # mass 1/3, so the reference mass is 2/3.
    logits = jnp.zeros((1, 3))
    reference_mask = jnp.array([True, True, False])
    mass = reference_mass(logits, reference_mask)
    assert mass.rho.shape == (1,)
    assert float(mass.rho[0]) == pytest.approx(2.0 / 3.0)
    assert bool(jnp.isnan(mass.rho_bar_edit))
    assert bool(jnp.isnan(mass.rho_bar_keep))


def test_region_mean_over_one_edit_query_is_two_thirds():
    logits = jnp.zeros((2, 3))
    edit_mask, keep_mask, reference_mask = _uniform_masks()
    mass = reference_mass(logits, reference_mask, edit_mask, keep_mask)
    assert float(mass.rho_bar_edit) == pytest.approx(2.0 / 3.0)
    assert float(mass.rho_bar_keep) == pytest.approx(2.0 / 3.0)
    assert float(mass.gap) == pytest.approx(0.0)


def test_edit_and_keep_offsets_on_uniform_logits():
    logits = jnp.zeros((2, 3))
    edit_mask, keep_mask, reference_mask = _uniform_masks()
    corrected = apply_correction(logits, edit_mask, keep_mask, reference_mask)

    # Edit query: +0.6 * ln(3/2) on both reference keys, rest key unchanged.
    assert float(corrected[0, 0]) == pytest.approx(EDIT_OFFSET)
    assert float(corrected[0, 1]) == pytest.approx(EDIT_OFFSET)
    assert float(corrected[0, 2]) == pytest.approx(0.0)
    # Keep query: -0.8 * ln(3/2) on both reference keys.
    assert float(corrected[1, 0]) == pytest.approx(KEEP_OFFSET)
    assert float(corrected[1, 1]) == pytest.approx(KEEP_OFFSET)
    assert float(corrected[1, 2]) == pytest.approx(0.0)
    # Input is not modified.
    assert float(logits[0, 0]) == pytest.approx(0.0)


def test_same_offset_preserves_within_group_softmax_ratios():
    # Reference keys differ by 1, rest keys differ by 0.75. A shared additive
    # offset on reference keys leaves both within-group ratios unchanged.
    logits = jnp.array(
        [
            [0.0, 1.0, -0.5, 0.25],
            [0.0, 1.0, -0.5, 0.25],
        ]
    )
    edit_mask = jnp.array([True, False])
    keep_mask = jnp.array([False, True])
    reference_mask = jnp.array([True, True, False, False])
    corrected = apply_correction(logits, edit_mask, keep_mask, reference_mask)

    edit_delta_0 = float(corrected[0, 0] - logits[0, 0])
    edit_delta_1 = float(corrected[0, 1] - logits[0, 1])
    keep_delta_0 = float(corrected[1, 0] - logits[1, 0])
    keep_delta_1 = float(corrected[1, 1] - logits[1, 1])
    assert edit_delta_0 == pytest.approx(edit_delta_1)
    assert keep_delta_0 == pytest.approx(keep_delta_1)
    assert edit_delta_0 > 0.0
    assert keep_delta_0 < 0.0

    # Rest keys are copied, not shifted.
    assert float(corrected[0, 2]) == pytest.approx(-0.5)
    assert float(corrected[0, 3]) == pytest.approx(0.25)
    assert float(corrected[1, 2]) == pytest.approx(-0.5)
    assert float(corrected[1, 3]) == pytest.approx(0.25)

    ref_ratio, rest_ratio = _group_ratios(corrected[0])
    assert float(ref_ratio) == pytest.approx(REF_SOFTMAX_RATIO)
    assert float(rest_ratio) == pytest.approx(REST_SOFTMAX_RATIO)


def test_large_logits_use_stable_softmax():
    # Raw exp(1000) overflows. Stable softmax of equal logits still gives 2/3.
    logits = jnp.full((1, 3), 1000.0)
    mass = reference_mass(logits, jnp.array([True, True, False]))
    assert bool(jnp.isfinite(mass.rho[0]))
    assert float(mass.rho[0]) == pytest.approx(2.0 / 3.0)


def test_second_half_when_selectivity_exceeds_threshold():
    # L = 4. B = 0.5 / (0.1 + 1e-6) = 4.99995 >= 2.7, so only the second half.
    # 0-indexed that is layers 2 and 3 (paper 1-indexed 3 and 4).
    rho_edit = [0.1, 0.1, 0.5, 0.5]
    rho_keep = [0.1, 0.1, 0.1, 0.1]
    assert select_layers(rho_edit, rho_keep) == (2, 3)


def test_low_profile_selects_every_layer():
    # L = 4, rho_edit = rho_keep = 0.5.
    # B = 0.5 / (0.5 + 1e-6) = 0.999998 < 2.7.
    # b_bar_edit = ln(2) = 0.693147 < 3.7.
    # Both statistics are under their thresholds, so every layer is selected.
    rho = [0.5, 0.5, 0.5, 0.5]
    assert select_layers(rho, rho) == (0, 1, 2, 3)


def test_log_inverse_alone_selects_second_half():
    # B = 0.01 / (0.01 + 1e-6) = 0.9999 < 2.7.
    # b_bar_edit = ln(100) = 4.605170 >= 3.7, so the OR still selects the half.
    rho = [0.01, 0.01, 0.01, 0.01]
    assert select_layers(rho, rho) == (2, 3)


def test_selectivity_uses_the_max_not_the_mean():
    # Per-layer ratios are about 4, 1, 1, 1. The max is 4 >= 2.7.
    # The mean ratio is about 1.75, which would not cross 2.7.
    # b_bar_edit = (ln(2.5) + 3 * ln(10)) / 4 = 1.956 < 3.7.
    # Only the max in eq. (6) selects the second half.
    rho_edit = [0.4, 0.1, 0.1, 0.1]
    rho_keep = [0.1, 0.1, 0.1, 0.1]
    assert select_layers(rho_edit, rho_keep) == (2, 3)


def test_odd_length_second_half_starts_at_floor_half():
    # L = 5, B = 0.5 / (0.1 + 1e-6) >= 2.7.
    # 0-indexed start is floor(5/2) = 2, so indices 2, 3, 4.
    rho_edit = [0.5, 0.5, 0.5, 0.5, 0.5]
    rho_keep = [0.1, 0.1, 0.1, 0.1, 0.1]
    assert select_layers(rho_edit, rho_keep) == (2, 3, 4)


def test_twenty_five_layers_second_half_is_twelve_through_twenty_four():
    # Appendix: FLUX.2-klein has 25 layers and corrects 0-indexed 12..24
    # when the rule picks the second half.
    rho_edit = [0.5] * 25
    rho_keep = [0.1] * 25
    assert select_layers(rho_edit, rho_keep) == tuple(range(12, 25))


def test_empty_edit_mask_does_not_write_edit_offset():
    logits = jnp.zeros((2, 3))
    edit_mask = jnp.array([False, False])
    keep_mask = jnp.array([True, False])
    reference_mask = jnp.array([True, True, False])

    mass = reference_mass(logits, reference_mask, edit_mask, keep_mask)
    assert bool(jnp.isnan(mass.rho_bar_edit))
    assert float(mass.rho_bar_keep) == pytest.approx(2.0 / 3.0)
    assert bool(jnp.isnan(mass.gap))

    corrected = apply_correction(logits, edit_mask, keep_mask, reference_mask)
    # Keep query gets the keep offset, not the edit offset and not nan.
    assert float(corrected[0, 0]) == pytest.approx(KEEP_OFFSET)
    assert float(corrected[0, 1]) == pytest.approx(KEEP_OFFSET)
    assert float(corrected[0, 2]) == pytest.approx(0.0)
    assert float(corrected[0, 0]) != pytest.approx(EDIT_OFFSET)
    # Query in neither region is unchanged.
    assert float(corrected[1, 0]) == pytest.approx(0.0)
    assert float(corrected[1, 1]) == pytest.approx(0.0)
    assert float(corrected[1, 2]) == pytest.approx(0.0)
    assert bool(jnp.all(jnp.isfinite(corrected)))


def test_empty_keep_mask_does_not_write_keep_offset():
    logits = jnp.zeros((1, 3))
    corrected = apply_correction(
        logits,
        jnp.array([True]),
        jnp.array([False]),
        jnp.array([True, True, False]),
    )
    assert float(corrected[0, 0]) == pytest.approx(EDIT_OFFSET)
    assert float(corrected[0, 1]) == pytest.approx(EDIT_OFFSET)
    assert float(corrected[0, 2]) == pytest.approx(0.0)
    assert bool(jnp.all(jnp.isfinite(corrected)))


def test_both_regions_empty_leaves_logits_unchanged():
    logits = jnp.array([[1.0, 2.0, 3.0]])
    corrected = apply_correction(
        logits,
        jnp.array([False]),
        jnp.array([False]),
        jnp.array([True, False, True]),
    )
    assert float(corrected[0, 0]) == pytest.approx(1.0)
    assert float(corrected[0, 1]) == pytest.approx(2.0)
    assert float(corrected[0, 2]) == pytest.approx(3.0)


def test_rho_below_clamp_uses_one_e_minus_twelve():
    # One reference key at 0 and one rest key at 40.
    # rho = 1 / (1 + exp(40)) is far below 1e-12, so the guard clamps it.
    # Edit offset is then 0.6 * ln(1e12), not 0.6 * ln(1/rho) and not the
    # appendix clip of 8.
    logits = jnp.array([[0.0, 40.0]])
    corrected = apply_correction(
        logits,
        jnp.array([True]),
        jnp.array([False]),
        jnp.array([True, False]),
    )
    assert float(corrected[0, 0]) == pytest.approx(CLAMPED_EDIT_OFFSET, abs=1e-4)
    assert float(corrected[0, 1]) == pytest.approx(40.0)


def test_heads_axis_shares_one_region_offset():
    logits = jnp.zeros((2, 2, 3))
    edit_mask, keep_mask, reference_mask = _uniform_masks()
    corrected = apply_correction(logits, edit_mask, keep_mask, reference_mask)
    assert corrected.shape == (2, 2, 3)
    assert float(corrected[0, 0, 0]) == pytest.approx(EDIT_OFFSET)
    assert float(corrected[1, 0, 1]) == pytest.approx(EDIT_OFFSET)
    assert float(corrected[0, 1, 0]) == pytest.approx(KEEP_OFFSET)
    assert float(corrected[1, 1, 2]) == pytest.approx(0.0)


def test_region_mean_averages_heads_and_queries():
    # Head 0 is uniform, so rho = 2/3.
    # Head 1 has rest-key logit ln(2): exp scores 1, 1, 2, so rho = 1/2.
    # One edit query: rho_bar_edit = (2/3 + 1/2) / 2 = 7/12.
    logits = jnp.zeros((2, 1, 3))
    logits = logits.at[1, 0, 2].set(jnp.log(jnp.asarray(2.0)))
    mass = reference_mass(
        logits,
        jnp.array([True, True, False]),
        edit_mask=jnp.array([True]),
        keep_mask=jnp.array([False]),
    )
    assert mass.rho.shape == (2, 1)
    assert float(mass.rho[0, 0]) == pytest.approx(2.0 / 3.0)
    assert float(mass.rho[1, 0]) == pytest.approx(0.5, abs=1e-5)
    assert float(mass.rho_bar_edit) == pytest.approx(7.0 / 12.0, abs=1e-5)
    assert bool(jnp.isnan(mass.rho_bar_keep))


def test_two_edit_queries_are_averaged():
    # Query 0: uniform, rho = 2/3. Query 1: rest logit ln(2), rho = 1/2.
    # Mean over the two edit queries is 7/12. Keep is empty.
    logits = jnp.zeros((2, 3))
    logits = logits.at[1, 2].set(jnp.log(jnp.asarray(2.0)))
    mass = reference_mass(
        logits,
        jnp.array([True, True, False]),
        edit_mask=jnp.array([True, True]),
        keep_mask=jnp.array([False, False]),
    )
    assert float(mass.rho[0]) == pytest.approx(2.0 / 3.0)
    assert float(mass.rho[1]) == pytest.approx(0.5, abs=1e-5)
    assert float(mass.rho_bar_edit) == pytest.approx(7.0 / 12.0, abs=1e-5)
    assert bool(jnp.isnan(mass.rho_bar_keep))


def test_overlap_takes_the_edit_offset():
    logits = jnp.zeros((1, 3))
    corrected = apply_correction(
        logits,
        jnp.array([True]),
        jnp.array([True]),
        jnp.array([True, True, False]),
    )
    assert float(corrected[0, 0]) == pytest.approx(EDIT_OFFSET)
    assert float(corrected[0, 1]) == pytest.approx(EDIT_OFFSET)
    assert float(corrected[0, 2]) == pytest.approx(0.0)


def _group_ratios(row):
    shifted = row - jnp.max(row)
    probs = jnp.exp(shifted)
    probs = probs / jnp.sum(probs)
    return probs[0] / probs[1], probs[2] / probs[3]
