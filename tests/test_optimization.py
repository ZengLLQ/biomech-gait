"""Unit tests for static optimization muscle force distribution."""

import numpy as np
from biomech_gait.optimization.muscle_distribution import StaticOptimizationSolver


def test_static_optimization_plantarflexion() -> None:
    solver = StaticOptimizationSolver()
    # Request pure ankle plantarflexion moment (50 N*m)
    target_m = np.array([50.0, 0.0, 0.0])
    res = solver.solve(target_m, 0.0, 0.0, 0.0)

    # Soleus or gastrocnemius should be strongly activated
    sol_act = res.activations["soleus"]
    gas_act = res.activations["gastrocnemius"]
    assert sol_act > 0.05 or gas_act > 0.05
    # Antagonist tibialis anterior should be minimal
    assert res.activations["tibialis_anterior"] < 0.05
    # Residual error should be small
    assert res.equilibrium_residual < 2.0
