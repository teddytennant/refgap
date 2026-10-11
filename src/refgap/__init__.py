"""RefGAP reference-logit correction."""

from refgap.correct import (
    EDIT_BIAS_CLIP_MAX,
    EDIT_BIAS_CLIP_MIN,
    EPS_B,
    GAMMA_EDIT,
    GAMMA_KEEP,
    RHO_FLOOR,
    TAU_B,
    TAU_B_BAR,
    ReferenceMass,
    apply_correction,
    reference_mass,
    select_layers,
)

__all__ = [
    "EDIT_BIAS_CLIP_MAX",
    "EDIT_BIAS_CLIP_MIN",
    "EPS_B",
    "GAMMA_EDIT",
    "GAMMA_KEEP",
    "RHO_FLOOR",
    "ReferenceMass",
    "TAU_B",
    "TAU_B_BAR",
    "apply_correction",
    "reference_mass",
    "select_layers",
]

__version__ = "0.1.0"
