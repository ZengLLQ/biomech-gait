"""Hill-type 3-element musculoskeletal muscle actuator model."""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MuscleParameters:
    """Biomechanical architectural parameters for a muscle unit."""

    name: str
    max_isometric_force: float  # F0 (Newtons)
    optimal_fiber_length: float # l0_ce (meters)
    tendon_slack_length: float  # l_slack (meters)
    pennation_angle_deg: float  # alpha0 at resting length (degrees)
    max_contraction_velocity: float # v_max in fiber lengths/sec (typically 10.0)


class HillMuscleActuator:
    """Computes instantaneous muscle fiber forces, tendon forces, and passive elastic responses."""

    def __init__(self, params: MuscleParameters) -> None:
        self.p = params
        self.alpha0 = math.radians(params.pennation_angle_deg)

    def active_force_length(self, norm_length: float) -> float:
        """Active force-length curve f_L(l_tilde) Gaussian bell curve."""
        gamma = 0.45
        return float(np.exp(-((norm_length - 1.0) / gamma) ** 2))

    def passive_force_length(self, norm_length: float) -> float:
        """Passive parallel elastic element curve f_PE(l_tilde)."""
        if norm_length <= 1.0:
            return 0.0
        k_pe = 4.0
        strain = norm_length - 1.0
        return float((np.exp(k_pe * strain / 0.6) - 1.0) / (np.exp(k_pe) - 1.0))

    def force_velocity(self, norm_velocity: float) -> float:
        """Force-velocity multiplier f_V(v_tilde).

        Negative velocity = concentric shortening; Positive = eccentric lengthening.
        """
        # Clamped normalized velocity in [-1, 1]
        v_tilde = np.clip(norm_velocity, -1.0, 1.0)
        if v_tilde <= 0.0:
            # Concentric (Hill hyperbolic relationship)
            a_curv = 0.25
            return float((1.0 + v_tilde) / (1.0 - v_tilde / a_curv))
        else:
            # Eccentric (asymptote ~ 1.5)
            f_len = 1.4
            return float(1.0 + (f_len - 1.0) * (v_tilde / (v_tilde + 0.15)))

    def evaluate_force(
        self,
        activation: float,
        muscle_tendon_length: float,
        muscle_tendon_velocity: float = 0.0,
    ) -> float:
        """Compute muscle-tendon force (N) given activation a in [0, 1] and kinematics.

        Parameters
        ----------
        activation : float
            Neural excitation/activation level clamped to [0.01, 1.0].
        muscle_tendon_length : float
            Total distance from anatomical origin to insertion (meters).
        muscle_tendon_velocity : float
            Rate of change of muscle-tendon length (m/s).

        Returns
        -------
        tendon_force : float
            Tensile force transmitted to bone insertion site (N).
        """
        a = float(np.clip(activation, 0.001, 1.0))

        # Rigid tendon assumption for static / quasi-static equilibrium:
        # l_mt = l_slack + l_ce * cos(alpha)
        # Approximate fiber length:
        l_slack = self.p.tendon_slack_length
        cos_alpha = math.cos(self.alpha0)
        l_ce = max(0.001, (muscle_tendon_length - l_slack) / max(0.1, cos_alpha))
        norm_l_ce = l_ce / self.p.optimal_fiber_length

        norm_v_ce = muscle_tendon_velocity / (self.p.optimal_fiber_length * self.p.max_contraction_velocity)

        f_l = self.active_force_length(norm_l_ce)
        f_pe = self.passive_force_length(norm_l_ce)
        f_v = self.force_velocity(norm_v_ce)

        # Fiber force: F_ce = (a * f_L * f_V + f_PE) * F0
        f_fiber = (a * f_l * f_v + f_pe) * self.p.max_isometric_force
        # Tendon force along insertion direction
        f_tendon = max(0.0, f_fiber * cos_alpha)
        return float(f_tendon)
