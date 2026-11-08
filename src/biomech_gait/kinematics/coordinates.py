"""Lower limb sagittal kinematic transformations and anthropometric segment model."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SegmentParameters:
    """De Leva / Winter anthropometric body segment parameters."""

    name: str
    mass_fraction: float        # Fraction of total body mass M
    length_fraction: float      # Fraction of body height H
    com_proximal_fraction: float  # Center of mass distance from proximal joint as fraction of segment length
    radius_of_gyration_com: float # Radius of gyration about transverse axis through CoM as fraction of length


# Anthropometric standards for adult human (Winter / De Leva)
DEFAULT_ANTHROPOMETRY: dict[str, SegmentParameters] = {
    "thigh": SegmentParameters(
        name="Thigh",
        mass_fraction=0.100,
        length_fraction=0.245,
        com_proximal_fraction=0.433,
        radius_of_gyration_com=0.323,
    ),
    "shank": SegmentParameters(
        name="Shank",
        mass_fraction=0.0465,
        length_fraction=0.246,
        com_proximal_fraction=0.433,
        radius_of_gyration_com=0.302,
    ),
    "foot": SegmentParameters(
        name="Foot",
        mass_fraction=0.0145,
        length_fraction=0.152,
        com_proximal_fraction=0.500,
        radius_of_gyration_com=0.475,
    ),
}


@dataclass
class JointPositions:
    """2D sagittal Cartesian coordinates of lower extremity landmarks (x, y)."""

    hip: np.ndarray        # Shape (2,) or (N, 2)
    knee: np.ndarray
    ankle: np.ndarray
    toe: np.ndarray
    thigh_com: np.ndarray
    shank_com: np.ndarray
    foot_com: np.ndarray


class SagittalKinematicModel:
    """Computes spatial joint centers and segment CoMs from sagittal joint angles."""

    def __init__(
        self,
        body_mass_kg: float = 70.0,
        body_height_m: float = 1.75,
        params: dict[str, SegmentParameters] | None = None,
    ) -> None:
        self.mass = float(body_mass_kg)
        self.height = float(body_height_m)
        self.params = params or DEFAULT_ANTHROPOMETRY

        # Calculate absolute segment metrics
        self.thigh_len = self.params["thigh"].length_fraction * self.height
        self.shank_len = self.params["shank"].length_fraction * self.height
        self.foot_len = self.params["foot"].length_fraction * self.height

        self.thigh_mass = self.params["thigh"].mass_fraction * self.mass
        self.shank_mass = self.params["shank"].mass_fraction * self.mass
        self.foot_mass = self.params["foot"].mass_fraction * self.mass

        # Moments of inertia about CoM: I = m * (r_gyr * L)^2
        self.thigh_inertia = self.thigh_mass * (self.params["thigh"].radius_of_gyration_com * self.thigh_len) ** 2
        self.shank_inertia = self.shank_mass * (self.params["shank"].radius_of_gyration_com * self.shank_len) ** 2
        self.foot_inertia = self.foot_mass * (self.params["foot"].radius_of_gyration_com * self.foot_len) ** 2

    def forward_kinematics(
        self,
        pelvis_pos: np.ndarray,
        hip_angle_rad: float,
        knee_angle_rad: float,
        ankle_angle_rad: float,
    ) -> JointPositions:
        """Compute landmarks for a single time instant.

        Parameters
        ----------
        pelvis_pos : np.ndarray
            (x, y) coordinates of hip joint center.
        hip_angle_rad : float
            Hip flexion angle (rad, >0 is forward flexion).
        knee_angle_rad : float
            Knee flexion angle (rad, >0 is knee bent backward).
        ankle_angle_rad : float
            Ankle dorsiflexion angle (rad, >0 is dorsiflexion, <0 is plantarflexion).
        """
        # Segment absolute orientations relative to vertical downward axis
        # Vertical downward vector is (0, -1)
        phi_thigh = -hip_angle_rad  # negative rotates thigh forward/anticlockwise
        # Thigh vector: (sin(hip), -cos(hip))
        thigh_vec = np.array([np.sin(hip_angle_rad), -np.cos(hip_angle_rad)])
        knee_pos = pelvis_pos + self.thigh_len * thigh_vec

        # Shank orientation: hip angle minus knee flexion
        shank_angle = hip_angle_rad - knee_angle_rad
        shank_vec = np.array([np.sin(shank_angle), -np.cos(shank_angle)])
        ankle_pos = knee_pos + self.shank_len * shank_vec

        # Foot orientation: shank angle plus (pi/2 + ankle dorsiflexion)
        # In neutral standing, foot is perpendicular to shank pointing forward
        foot_angle = shank_angle + (np.pi / 2.0 - ankle_angle_rad)
        foot_vec = np.array([np.cos(foot_angle), np.sin(foot_angle)])
        toe_pos = ankle_pos + self.foot_len * foot_vec

        # Center of mass positions
        thigh_com = pelvis_pos + (self.params["thigh"].com_proximal_fraction * self.thigh_len) * thigh_vec
        shank_com = knee_pos + (self.params["shank"].com_proximal_fraction * self.shank_len) * shank_vec
        foot_com = ankle_pos + (self.params["foot"].com_proximal_fraction * self.foot_len) * foot_vec

        return JointPositions(
            hip=pelvis_pos,
            knee=knee_pos,
            ankle=ankle_pos,
            toe=toe_pos,
            thigh_com=thigh_com,
            shank_com=shank_com,
            foot_com=foot_com,
        )
