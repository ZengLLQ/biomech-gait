"""Unit tests for lower limb sagittal kinematics and anthropometrics."""

import numpy as np
from biomech_gait.kinematics.coordinates import (
    DEFAULT_ANTHROPOMETRY,
    SagittalKinematicModel,
)


def test_anthropometry_parameters() -> None:
    assert "thigh" in DEFAULT_ANTHROPOMETRY
    assert "shank" in DEFAULT_ANTHROPOMETRY
    assert "foot" in DEFAULT_ANTHROPOMETRY

    model = SagittalKinematicModel(body_mass_kg=70.0, body_height_m=1.75)
    assert model.thigh_mass > model.shank_mass > model.foot_mass
    assert model.thigh_len > 0.40
    assert model.shank_len > 0.40


def test_forward_kinematics_standing() -> None:
    model = SagittalKinematicModel(body_mass_kg=70.0, body_height_m=1.75)
    pelvis = np.array([0.0, 1.0])
    # Neutral standing: 0 angles
    pos = model.forward_kinematics(pelvis, hip_angle_rad=0.0, knee_angle_rad=0.0, ankle_angle_rad=0.0)

    # In neutral standing, knee is directly below pelvis
    assert np.isclose(pos.knee[0], pelvis[0], atol=1e-3)
    assert pos.knee[1] < pelvis[1]
    # Ankle is directly below knee
    assert np.isclose(pos.ankle[0], pos.knee[0], atol=1e-3)
    assert pos.ankle[1] < pos.knee[1]
    # Toe is forward of ankle (x_toe > x_ankle)
    assert pos.toe[0] > pos.ankle[0]
