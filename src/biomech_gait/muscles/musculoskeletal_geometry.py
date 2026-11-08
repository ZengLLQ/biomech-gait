"""Musculoskeletal lower-extremity geometry, moment arms, and tendon pathways."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from biomech_gait.muscles.hill_model import MuscleParameters, HillMuscleActuator


# Standard OpenSim / Delp lower-limb muscle architecture parameters
LOWER_LIMB_MUSCLES: dict[str, MuscleParameters] = {
    "soleus": MuscleParameters(
        name="Soleus",
        max_isometric_force=3500.0,
        optimal_fiber_length=0.050,
        tendon_slack_length=0.250,
        pennation_angle_deg=25.0,
        max_contraction_velocity=10.0,
    ),
    "gastrocnemius": MuscleParameters(
        name="Gastrocnemius",
        max_isometric_force=1800.0,
        optimal_fiber_length=0.055,
        tendon_slack_length=0.380,
        pennation_angle_deg=17.0,
        max_contraction_velocity=10.0,
    ),
    "tibialis_anterior": MuscleParameters(
        name="Tibialis Anterior",
        max_isometric_force=900.0,
        optimal_fiber_length=0.080,
        tendon_slack_length=0.220,
        pennation_angle_deg=9.0,
        max_contraction_velocity=10.0,
    ),
    "vastus_group": MuscleParameters(
        name="Vastus Group",
        max_isometric_force=4500.0,
        optimal_fiber_length=0.085,
        tendon_slack_length=0.160,
        pennation_angle_deg=10.0,
        max_contraction_velocity=10.0,
    ),
    "rectus_femoris": MuscleParameters(
        name="Rectus Femoris",
        max_isometric_force=1200.0,
        optimal_fiber_length=0.080,
        tendon_slack_length=0.340,
        pennation_angle_deg=5.0,
        max_contraction_velocity=10.0,
    ),
    "hamstrings": MuscleParameters(
        name="Hamstrings",
        max_isometric_force=2600.0,
        optimal_fiber_length=0.090,
        tendon_slack_length=0.330,
        pennation_angle_deg=8.0,
        max_contraction_velocity=10.0,
    ),
    "gluteus_maximus": MuscleParameters(
        name="Gluteus Maximus",
        max_isometric_force=3000.0,
        optimal_fiber_length=0.140,
        tendon_slack_length=0.120,
        pennation_angle_deg=15.0,
        max_contraction_velocity=10.0,
    ),
    "iliopsoas": MuscleParameters(
        name="Iliopsoas",
        max_isometric_force=1500.0,
        optimal_fiber_length=0.100,
        tendon_slack_length=0.100,
        pennation_angle_deg=7.0,
        max_contraction_velocity=10.0,
    ),
}


class MusculoskeletalGeometry:
    """Computes moment arms matrix and muscle-tendon lengths across ankle, knee, and hip joints."""

    def __init__(self) -> None:
        self.muscles = {name: HillMuscleActuator(p) for name, p in LOWER_LIMB_MUSCLES.items()}
        self.muscle_names = list(LOWER_LIMB_MUSCLES.keys())
        self.num_muscles = len(self.muscle_names)

    def get_moment_arms(
        self,
        hip_rad: float,
        knee_rad: float,
        ankle_rad: float,
    ) -> np.ndarray:
        """Construct (3, M) matrix R of muscle moment arms across [Ankle, Knee, Hip].

        Sign convention:
          - Ankle: Plantarflexion = + (soleus, gastrocnemius), Dorsiflexion = - (tibialis anterior).
          - Knee: Extension = + (vastus, rectus femoris), Flexion = - (hamstrings, gastrocnemius).
          - Hip: Extension = + (gluteus, hamstrings), Flexion = - (iliopsoas, rectus femoris).

        Returns
        -------
        R : np.ndarray
            (3, num_muscles) matrix where element (j, m) is the moment arm of muscle m at joint j in meters.
        """
        R = np.zeros((3, self.num_muscles), dtype=np.float64)

        # 0: soleus -> Ankle plantarflexor
        R[0, 0] = 0.050

        # 1: gastrocnemius -> Ankle plantarflexor, Knee flexor
        R[0, 1] = 0.050
        R[1, 1] = -0.025

        # 2: tibialis anterior -> Ankle dorsiflexor
        R[0, 2] = -0.035

        # 3: vastus group -> Knee extensor (patella lever arm)
        R[1, 3] = 0.045

        # 4: rectus femoris -> Knee extensor, Hip flexor
        R[1, 4] = 0.042
        R[2, 4] = -0.035

        # 5: hamstrings -> Knee flexor, Hip extensor
        R[1, 5] = -0.038
        R[2, 5] = 0.055

        # 6: gluteus maximus -> Hip extensor
        R[2, 6] = 0.065

        # 7: iliopsoas -> Hip flexor
        R[2, 7] = -0.040

        return R

    def get_muscle_lengths(
        self,
        hip_rad: float,
        knee_rad: float,
        ankle_rad: float,
    ) -> np.ndarray:
        """Estimate muscle-tendon lengths based on nominal joint excursion."""
        R = self.get_moment_arms(hip_rad, knee_rad, ankle_rad)
        lengths = np.zeros(self.num_muscles, dtype=np.float64)
        angles = np.array([ankle_rad, knee_rad, hip_rad])

        for m_idx, (name, actuator) in enumerate(self.muscles.items()):
            nominal_l = actuator.p.tendon_slack_length + actuator.p.optimal_fiber_length
            # Length change: delta_L = - sum_j (R_j * delta_theta_j)
            delta_l = -float(np.dot(R[:, m_idx], angles))
            lengths[m_idx] = max(0.05, nominal_l + delta_l)

        return lengths
