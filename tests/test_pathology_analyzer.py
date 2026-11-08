"""Unit tests for clinical gait pathology simulator and assessment."""

from biomech_gait.clinical.pathology_analyzer import (
    GaitPathology,
    GaitPathologySimulator,
)


def test_pathology_differentiation() -> None:
    sim = GaitPathologySimulator()
    p_norm = sim.generate_profile(GaitPathology.NORMAL)
    p_crouch = sim.generate_profile(GaitPathology.CROUCH_GAIT)
    p_stiff = sim.generate_profile(GaitPathology.STIFF_KNEE)

    e_norm = sim.evaluate_gait(p_norm)
    e_crouch = sim.evaluate_gait(p_crouch)
    e_stiff = sim.evaluate_gait(p_stiff)

    assert e_norm.pathology == GaitPathology.NORMAL
    assert e_crouch.pathology == GaitPathology.CROUCH_GAIT
    assert e_stiff.pathology == GaitPathology.STIFF_KNEE

    # Crouch severity should be high for crouch, low for normal
    assert e_crouch.crouch_severity_index > 0.5
    assert e_norm.crouch_severity_index < 0.2

    # Stiff knee index should be high for stiff knee
    assert e_stiff.stiff_knee_index > 0.4
