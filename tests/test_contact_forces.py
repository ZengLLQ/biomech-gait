"""Unit tests for articular contact force estimations."""

import numpy as np
from biomech_gait.dynamics.inverse_dynamics import JointKinetics
from biomech_gait.optimization.muscle_distribution import MuscleDistributionResult
from biomech_gait.clinical.contact_forces import JointContactCalculator


def test_contact_forces() -> None:
    calc = JointContactCalculator(body_mass_kg=70.0)
    kinetics = JointKinetics(
        ankle_force=np.array([0.0, 700.0]),
        knee_force=np.array([0.0, 700.0]),
        hip_force=np.array([0.0, 700.0]),
        ankle_moment=50.0,
        knee_moment=20.0,
        hip_moment=30.0,
    )
    muscles = MuscleDistributionResult(
        activations={"vastus_group": 0.3},
        forces={"vastus_group": 1200.0, "soleus": 1500.0},
        achieved_moments=np.array([50.0, 20.0, 30.0]),
        target_moments=np.array([50.0, 20.0, 30.0]),
        equilibrium_residual=0.0,
        cost=0.1,
    )

    loads = calc.compute_contact_forces(kinetics, muscles)
    assert loads.knee_contact_n > 1500.0
    assert loads.knee_contact_bw > 2.0
    assert "Normal" in loads.tibiofemoral_oa_risk or "Moderate" in loads.tibiofemoral_oa_risk
