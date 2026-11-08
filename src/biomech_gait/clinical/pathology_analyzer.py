"""Clinical gait impairment classification, crouch gait, and stroke hemiparesis analyzer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import numpy as np


class GaitPathology(str, Enum):
    """Clinical gait disorder classifications."""

    NORMAL = "Normal Adult Gait"
    CROUCH_GAIT = "Cerebral Palsy Crouch Gait"
    STIFF_KNEE = "Post-Stroke Stiff-Knee Gait"
    PARKINSONIAN = "Parkinsonian Festinating Gait"


@dataclass
class GaitKinematicsProfile:
    """Full gait cycle (0% to 100%) time series for sagittal joints."""

    percent_gait_cycle: np.ndarray  # (N,) array 0.0 to 100.0%
    hip_flexion_deg: np.ndarray
    knee_flexion_deg: np.ndarray
    ankle_dorsiflexion_deg: np.ndarray
    grf_vertical_bw: np.ndarray     # Vertical ground reaction force normalized to body weight
    grf_anterior_posterior_bw: np.ndarray


@dataclass
class ClinicalGaitEvaluation:
    """Diagnostic metrics and rehabilitation assessment report."""

    pathology: GaitPathology
    mean_stance_knee_flexion_deg: float
    peak_swing_knee_flexion_deg: float
    crouch_severity_index: float     # 0.0 (normal) to 1.0 (severe crouch)
    stiff_knee_index: float          # 0.0 (normal) to 1.0 (severe stiff knee)
    rehabilitation_recommendations: list[str]


class GaitPathologySimulator:
    """Generates standard clinical kinematic profiles and evaluates pathological indices."""

    @staticmethod
    def generate_profile(
        pathology: GaitPathology = GaitPathology.NORMAL,
        num_points: int = 101,
    ) -> GaitKinematicsProfile:
        """Synthesize normalized gait cycle kinematics for a given pathology."""
        pct = np.linspace(0.0, 100.0, num_points)
        t = pct / 100.0

        if pathology == GaitPathology.NORMAL:
            # Normal: Hip flexion peaks ~30 deg at 0%, extends to -10 deg at 50%
            hip = 30.0 * np.cos(2 * np.pi * t) + 5.0
            # Normal: Knee flexion initial 5 deg, loading response 18 deg, extension 5 deg, swing peak 60 deg at 75%
            knee = 15.0 + 10.0 * np.sin(2 * np.pi * t) + 40.0 * np.exp(-((t - 0.75) / 0.12) ** 2)
            # Normal Ankle: neutral 0 deg at 0%, dorsiflexion 10 deg in midstance, plantarflexion -15 deg at toe-off (60%)
            ankle = 8.0 * np.sin(2 * np.pi * t - 0.5) - 15.0 * np.exp(-((t - 0.60) / 0.08) ** 2)
            # Normal GRF: dual peak "M-shape" in stance (0-60%), 0 in swing (60-100%)
            grf_v = np.zeros(num_points)
            stance_mask = pct <= 60.0
            s_t = pct[stance_mask] / 60.0
            grf_v[stance_mask] = 1.15 * np.sin(np.pi * s_t) + 0.25 * np.sin(3 * np.pi * s_t)
            grf_ap = np.zeros(num_points)
            grf_ap[stance_mask] = -0.20 * np.cos(np.pi * s_t)

        elif pathology == GaitPathology.CROUCH_GAIT:
            # Cerebral Palsy Crouch Gait: persistent knee flexion > 35 deg, hip flexion elevated
            hip = 40.0 + 15.0 * np.cos(2 * np.pi * t)
            knee = 42.0 + 8.0 * np.sin(2 * np.pi * t) + 20.0 * np.exp(-((t - 0.75) / 0.15) ** 2)
            ankle = 12.0 * np.ones(num_points)  # excessive dorsiflexion
            grf_v = np.zeros(num_points)
            stance_mask = pct <= 60.0
            s_t = pct[stance_mask] / 60.0
            grf_v[stance_mask] = 1.05 * np.sin(np.pi * s_t)
            grf_ap = np.zeros(num_points)

        elif pathology == GaitPathology.STIFF_KNEE:
            # Stroke hemiparesis: failure to flex knee during swing (< 35 deg), prolonged stance
            hip = 20.0 + 10.0 * np.cos(2 * np.pi * t)
            knee = 12.0 + 8.0 * np.sin(2 * np.pi * t) + 12.0 * np.exp(-((t - 0.75) / 0.15) ** 2)  # Peak swing ~ 28 deg
            ankle = -10.0 + 5.0 * np.sin(2 * np.pi * t)  # equinus / foot drop
            grf_v = np.zeros(num_points)
            stance_mask = pct <= 65.0
            s_t = pct[stance_mask] / 65.0
            grf_v[stance_mask] = 0.95 * np.sin(np.pi * s_t)
            grf_ap = np.zeros(num_points)

        else:
            # Default fallback
            return GaitPathologySimulator.generate_profile(GaitPathology.NORMAL, num_points)

        return GaitKinematicsProfile(
            percent_gait_cycle=pct,
            hip_flexion_deg=hip,
            knee_flexion_deg=knee,
            ankle_dorsiflexion_deg=ankle,
            grf_vertical_bw=grf_v,
            grf_anterior_posterior_bw=grf_ap,
        )

    def evaluate_gait(self, profile: GaitKinematicsProfile) -> ClinicalGaitEvaluation:
        """Classify and assess clinical impairment metrics from a kinematics profile."""
        stance_mask = profile.percent_gait_cycle <= 60.0
        swing_mask = profile.percent_gait_cycle > 60.0

        mean_stance_knee = float(np.mean(profile.knee_flexion_deg[stance_mask]))
        peak_swing_knee = float(np.max(profile.knee_flexion_deg[swing_mask]))

        # Crouch index: Normal stance knee is ~10 deg. >30 deg is clinical crouch.
        crouch_index = float(np.clip((mean_stance_knee - 15.0) / 30.0, 0.0, 1.0))

        # Stiff knee index: Normal peak swing knee is ~60 deg. <40 deg is stiff knee.
        stiff_knee_index = float(np.clip((55.0 - peak_swing_knee) / 30.0, 0.0, 1.0))

        # Classification decision
        if crouch_index > 0.45:
            pathology = GaitPathology.CROUCH_GAIT
            recs = [
                "Consider hamstring lengthening / recession surgery to reduce knee flexion contracture.",
                "Psoas tenotomy to reduce excessive hip flexion.",
                "Grounded-reaction ankle-foot orthosis (GRAFO) to promote knee extension in stance.",
            ]
        elif stiff_knee_index > 0.40:
            pathology = GaitPathology.STIFF_KNEE
            recs = [
                "Evaluate rectus femoris intramuscular botulinum toxin injection (spasticity mitigation).",
                "Rectus femoris transfer (RFT) to gracilis or semitendinosus.",
                "Ankle-foot orthosis (AFO) to prevent toe drag and compensatory hip circumduction.",
            ]
        else:
            pathology = GaitPathology.NORMAL
            recs = ["Gait parameters within standard adult normative corridors.", "Maintain routine conditioning."]

        return ClinicalGaitEvaluation(
            pathology=pathology,
            mean_stance_knee_flexion_deg=mean_stance_knee,
            peak_swing_knee_flexion_deg=peak_swing_knee,
            crouch_severity_index=crouch_index,
            stiff_knee_index=stiff_knee_index,
            rehabilitation_recommendations=recs,
        )
