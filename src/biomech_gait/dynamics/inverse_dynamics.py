"""Newton-Euler multi-segment inverse dynamics formulation for sagittal gait."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from biomech_gait.kinematics.coordinates import SagittalKinematicModel, JointPositions


@dataclass
class JointKinetics:
    """Net internal joint reaction forces (N) and moments (N*m)."""

    ankle_force: np.ndarray    # (2,) [Fx, Fy]
    knee_force: np.ndarray
    hip_force: np.ndarray
    ankle_moment: float        # Positive = Plantarflexor moment
    knee_moment: float         # Positive = Extensor moment
    hip_moment: float          # Positive = Extensor moment


class NewtonEulerInverseDynamics:
    """Bottom-up recursive Newton-Euler inverse dynamics solver."""

    GRAVITY: np.ndarray = np.array([0.0, -9.81])

    def __init__(self, model: SagittalKinematicModel) -> None:
        self.model = model

    def solve_instant(
        self,
        positions: JointPositions,
        foot_acc_com: np.ndarray,
        shank_acc_com: np.ndarray,
        thigh_acc_com: np.ndarray,
        foot_ang_acc: float,
        shank_ang_acc: float,
        thigh_ang_acc: float,
        ground_reaction_force: np.ndarray,  # (2,) [GRFx, GRFy]
        center_of_pressure: np.ndarray,     # (2,) [COPx, COPy]
    ) -> JointKinetics:
        """Calculate internal joint forces and moments for a single time frame.

        Parameters
        ----------
        positions : JointPositions
            Cartesian coordinates of landmarks and CoMs.
        foot_acc_com, shank_acc_com, thigh_acc_com : np.ndarray
            Linear accelerations of segment CoMs (m/s^2).
        foot_ang_acc, shank_ang_acc, thigh_ang_acc : float
            Segment angular accelerations (rad/s^2).
        ground_reaction_force : np.ndarray
            GRF vector [Fx, Fy] in Newtons.
        center_of_pressure : np.ndarray
            Application point of GRF.

        Returns
        -------
        kinetics : JointKinetics
            Internal joint kinetics.
        """
        # 1. FOOT SEGMENT (Distal: GRF at COP; Proximal: Ankle joint)
        # Sum(F) = m * a -> F_ankle + F_grf + m_foot * g = m_foot * a_foot
        # F_ankle = m_foot * (a_foot - g) - F_grf
        f_ankle = self.model.foot_mass * (foot_acc_com - self.GRAVITY) - ground_reaction_force

        # Moment equation about foot CoM:
        # I * alpha = M_ankle + (r_cop - r_com) x F_grf + (r_ankle - r_com) x F_ankle
        # 2D cross product: r x F = rx * Fy - ry * Fx
        r_cop_rel = center_of_pressure - positions.foot_com
        r_ankle_rel = positions.ankle - positions.foot_com

        tau_grf = r_cop_rel[0] * ground_reaction_force[1] - r_cop_rel[1] * ground_reaction_force[0]
        tau_ankle_f = r_ankle_rel[0] * f_ankle[1] - r_ankle_rel[1] * f_ankle[0]

        # M_ankle = I * alpha - tau_grf - tau_ankle_f
        m_ankle = self.model.foot_inertia * foot_ang_acc - tau_grf - tau_ankle_f

        # 2. SHANK SEGMENT (Distal: -F_ankle, -M_ankle; Proximal: Knee joint)
        # Sum(F) = m * a -> F_knee - F_ankle + m_shank * g = m_shank * a_shank
        f_knee = self.model.shank_mass * (shank_acc_com - self.GRAVITY) + f_ankle

        # Moment about shank CoM:
        r_ankle_distal = positions.ankle - positions.shank_com
        r_knee_proximal = positions.knee - positions.shank_com

        tau_ankle_distal = r_ankle_distal[0] * (-f_ankle[1]) - r_ankle_distal[1] * (-f_ankle[0])
        tau_knee_f = r_knee_proximal[0] * f_knee[1] - r_knee_proximal[1] * f_knee[0]

        # I * alpha = M_knee - M_ankle + tau_ankle_distal + tau_knee_f
        # M_knee = I * alpha + M_ankle - tau_ankle_distal - tau_knee_f
        m_knee = self.model.shank_inertia * shank_ang_acc + m_ankle - tau_ankle_distal - tau_knee_f

        # 3. THIGH SEGMENT (Distal: -F_knee, -M_knee; Proximal: Hip joint)
        # Sum(F) = m * a -> F_hip - F_knee + m_thigh * g = m_thigh * a_thigh
        f_hip = self.model.thigh_mass * (thigh_acc_com - self.GRAVITY) + f_knee

        # Moment about thigh CoM:
        r_knee_distal = positions.knee - positions.thigh_com
        r_hip_proximal = positions.hip - positions.thigh_com

        tau_knee_distal = r_knee_distal[0] * (-f_knee[1]) - r_knee_distal[1] * (-f_knee[0])
        tau_hip_f = r_hip_proximal[0] * f_hip[1] - r_hip_proximal[1] * f_hip[0]

        m_hip = self.model.thigh_inertia * thigh_ang_acc + m_knee - tau_knee_distal - tau_hip_f

        return JointKinetics(
            ankle_force=f_ankle,
            knee_force=f_knee,
            hip_force=f_hip,
            ankle_moment=float(m_ankle),
            knee_moment=float(m_knee),
            hip_moment=float(m_hip),
        )
