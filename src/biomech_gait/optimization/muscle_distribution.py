"""Static optimization resolving musculoskeletal redundancy via physiological cost minimization."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

from biomech_gait.muscles.musculoskeletal_geometry import MusculoskeletalGeometry


@dataclass
class MuscleDistributionResult:
    """Estimated muscle activations and forces for a time instant."""

    activations: dict[str, float]
    forces: dict[str, float]
    achieved_moments: np.ndarray  # [Ankle, Knee, Hip] achieved
    target_moments: np.ndarray    # [Ankle, Knee, Hip] requested
    equilibrium_residual: float   # L2 norm of moment error
    cost: float                   # Sum of squared activations


class StaticOptimizationSolver:
    """Nonlinear constrained optimizer finding minimum fatigue muscle force distribution.

    Objective:
        min sum_{m=1}^M a_m^2 + w_res * ||reserve_moments||^2
    Subject to:
        R * F(a) + M_reserve = M_target
        0 <= a_m <= 1
    """

    def __init__(self, geometry: MusculoskeletalGeometry | None = None) -> None:
        self.geom = geometry or MusculoskeletalGeometry()
        self.num_muscles = self.geom.num_muscles

    def solve(
        self,
        target_moments: np.ndarray,  # (3,) [Ankle, Knee, Hip]
        hip_rad: float,
        knee_rad: float,
        ankle_rad: float,
        reserve_weight: float = 1000.0,
    ) -> MuscleDistributionResult:
        """Solve muscle activation distribution for a single time frame.

        Parameters
        ----------
        target_moments : np.ndarray
            [Ankle, Knee, Hip] required net internal joint moments in N*m.
        hip_rad, knee_rad, ankle_rad : float
            Joint angles in radians.
        reserve_weight : float
            Penalty weight on reserve actuator moments (prevents infeasibility).
        """
        R = self.geom.get_moment_arms(hip_rad, knee_rad, ankle_rad)
        lengths = self.geom.get_muscle_lengths(hip_rad, knee_rad, ankle_rad)

        # Precompute maximum isometric muscle capacities F0_cos_alpha
        f_max = np.zeros(self.num_muscles, dtype=np.float64)
        for m_idx, name in enumerate(self.geom.muscle_names):
            actuator = self.geom.muscles[name]
            f_max[m_idx] = actuator.evaluate_force(1.0, lengths[m_idx], 0.0)

        # Decision variables x: [a_1, ..., a_M, res_ankle, res_knee, res_hip]
        # Total variables: M + 3
        dim = self.num_muscles + 3
        bounds = [(0.0, 1.0) for _ in range(self.num_muscles)] + [(-500.0, 500.0) for _ in range(3)]

        def objective(x: np.ndarray) -> float:
            activations = x[: self.num_muscles]
            reserves = x[self.num_muscles :]
            # Muscle fatigue minimization: sum a_m^2 + reserve penalties
            return float(np.sum(activations**2) + reserve_weight * np.sum(reserves**2))

        def moment_constraints(x: np.ndarray) -> np.ndarray:
            activations = x[: self.num_muscles]
            reserves = x[self.num_muscles :]
            # R is (3, M), forces are activations * f_max
            forces = activations * f_max
            m_muscles = R @ forces
            # Constraint: m_muscles + reserves - target = 0
            return m_muscles + reserves - target_moments

        x0 = np.zeros(dim, dtype=np.float64)
        x0[: self.num_muscles] = 0.05  # small initial activation

        opt_res = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints={"type": "eq", "fun": moment_constraints},
            options={"maxiter": 150, "ftol": 1e-5},
        )

        best_x = opt_res.x
        opt_a = np.clip(best_x[: self.num_muscles], 0.0, 1.0)
        opt_forces = opt_a * f_max
        achieved_m = R @ opt_forces

        act_dict = {name: float(opt_a[i]) for i, name in enumerate(self.geom.muscle_names)}
        force_dict = {name: float(opt_forces[i]) for i, name in enumerate(self.geom.muscle_names)}
        residual = float(np.linalg.norm(achieved_m - target_moments))
        cost = float(np.sum(opt_a**2))

        return MuscleDistributionResult(
            activations=act_dict,
            forces=force_dict,
            achieved_moments=achieved_m,
            target_moments=target_moments,
            equilibrium_residual=residual,
            cost=cost,
        )
