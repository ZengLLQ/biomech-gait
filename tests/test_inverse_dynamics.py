"""Unit tests for Newton-Euler inverse dynamics."""

import numpy as np
from biomech_gait.kinematics.coordinates import SagittalKinematicModel
from biomech_gait.dynamics.inverse_dynamics import NewtonEulerInverseDynamics


def test_static_standing_equilibrium() -> None:
    mass = 70.0
    model = SagittalKinematicModel(body_mass_kg=mass, body_height_m=1.75)
    pos = model.forward_kinematics(np.array([0.0, 1.0]), 0.0, 0.0, 0.0)

    # In static standing on one leg, vertical GRF = total body weight
    bw_n = mass * 9.81
    grf = np.array([0.0, bw_n])
    cop = pos.foot_com.copy()

    dynamics = NewtonEulerInverseDynamics(model)
    kinetics = dynamics.solve_instant(
        positions=pos,
        foot_acc_com=np.array([0.0, 0.0]),
        shank_acc_com=np.array([0.0, 0.0]),
        thigh_acc_com=np.array([0.0, 0.0]),
        foot_ang_acc=0.0,
        shank_ang_acc=0.0,
        thigh_ang_acc=0.0,
        ground_reaction_force=grf,
        center_of_pressure=cop,
    )

    # The force exerted downward by the tibia onto the foot has magnitude ~ (total body - foot) weight
    assert kinetics.ankle_force[1] < 0.0  # Downward compressive action on foot
    assert abs(kinetics.ankle_force[1]) > 500.0
    assert abs(kinetics.knee_force[1]) > 500.0
    assert abs(kinetics.hip_force[1]) > 500.0
