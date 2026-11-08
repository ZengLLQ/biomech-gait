"""Clinical joint contact force estimations and osteoarthritis biomechanical risk indices."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from biomech_gait.dynamics.inverse_dynamics import JointKinetics
from biomech_gait.optimization.muscle_distribution import MuscleDistributionResult


@dataclass
class JointContactLoads:
    """Estimated internal joint contact compressive forces in Newtons and Body Weight (BW)."""

    knee_contact_n: float
    knee_contact_bw: float
    ankle_contact_n: float
    ankle_contact_bw: float
    hip_contact_n: float
    hip_contact_bw: float
    tibiofemoral_oa_risk: str


class JointContactCalculator:
    """Computes articular joint contact forces combining bone reaction and muscle tension."""

    def __init__(self, body_mass_kg: float = 70.0) -> None:
        self.mass = float(body_mass_kg)
        self.body_weight_n = self.mass * 9.81

    def compute_contact_forces(
        self,
        kinetics: JointKinetics,
        muscles: MuscleDistributionResult,
    ) -> JointContactLoads:
        """Calculate articular contact forces across ankle, knee, and hip joints.

        Parameters
        ----------
        kinetics : JointKinetics
            Internal joint reaction forces and moments from inverse dynamics.
        muscles : MuscleDistributionResult
            Muscle forces from static optimization.
        """
        # Knee contact: Reaction magnitude + knee muscle forces (vastus, rectus femoris, hamstrings, gastrocnemius)
        knee_reaction_mag = float(np.linalg.norm(kinetics.knee_force))
        knee_muscle_sum = (
            muscles.forces.get("vastus_group", 0.0)
            + muscles.forces.get("rectus_femoris", 0.0)
            + muscles.forces.get("hamstrings", 0.0)
            + muscles.forces.get("gastrocnemius", 0.0)
        )
        f_knee_contact = knee_reaction_mag + knee_muscle_sum
        f_knee_bw = f_knee_contact / self.body_weight_n

        # Ankle contact: Reaction magnitude + soleus + gastrocnemius + tibialis anterior
        ankle_reaction_mag = float(np.linalg.norm(kinetics.ankle_force))
        ankle_muscle_sum = (
            muscles.forces.get("soleus", 0.0)
            + muscles.forces.get("gastrocnemius", 0.0)
            + muscles.forces.get("tibialis_anterior", 0.0)
        )
        f_ankle_contact = ankle_reaction_mag + ankle_muscle_sum
        f_ankle_bw = f_ankle_contact / self.body_weight_n

        # Hip contact: Reaction magnitude + gluteus maximus + hamstrings + iliopsoas + rectus femoris
        hip_reaction_mag = float(np.linalg.norm(kinetics.hip_force))
        hip_muscle_sum = (
            muscles.forces.get("gluteus_maximus", 0.0)
            + muscles.forces.get("hamstrings", 0.0)
            + muscles.forces.get("iliopsoas", 0.0)
            + muscles.forces.get("rectus_femoris", 0.0)
        )
        f_hip_contact = hip_reaction_mag + hip_muscle_sum
        f_hip_bw = f_hip_contact / self.body_weight_n

        # Clinical OA Risk Stratification
        if f_knee_bw > 4.5:
            oa_risk = "Severe (Excessive compressive overload > 4.5 BW)"
        elif f_knee_bw > 3.5:
            oa_risk = "Elevated (Moderate compressive loading 3.5 - 4.5 BW)"
        else:
            oa_risk = "Normal Physiological Loading (< 3.5 BW)"

        return JointContactLoads(
            knee_contact_n=f_knee_contact,
            knee_contact_bw=f_knee_bw,
            ankle_contact_n=f_ankle_contact,
            ankle_contact_bw=f_ankle_bw,
            hip_contact_n=f_hip_contact,
            hip_contact_bw=f_hip_bw,
            tibiofemoral_oa_risk=oa_risk,
        )
