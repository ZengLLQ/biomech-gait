"""Unit tests for Hill muscle actuator curves."""

from biomech_gait.muscles.hill_model import MuscleParameters, HillMuscleActuator


def test_hill_curves() -> None:
    p = MuscleParameters(
        name="TestSoleus",
        max_isometric_force=3000.0,
        optimal_fiber_length=0.05,
        tendon_slack_length=0.25,
        pennation_angle_deg=20.0,
        max_contraction_velocity=10.0,
    )
    actuator = HillMuscleActuator(p)

    # Active force length peaks at norm_length = 1.0
    f_l_peak = actuator.active_force_length(1.0)
    assert f_l_peak == 1.0
    assert actuator.active_force_length(0.7) < 1.0
    assert actuator.active_force_length(1.3) < 1.0

    # Passive force is zero below resting length
    assert actuator.passive_force_length(0.9) == 0.0
    assert actuator.passive_force_length(1.2) > 0.0

    # Total force evaluation
    f_total = actuator.evaluate_force(1.0, 0.30, 0.0)
    assert f_total > 500.0
