"""ASCII visualizer for lower-limb biomechanics, stick figures, and joint kinetics."""

from __future__ import annotations

import numpy as np

from biomech_gait.clinical.pathology_analyzer import GaitKinematicsProfile, ClinicalGaitEvaluation
from biomech_gait.optimization.muscle_distribution import MuscleDistributionResult
from biomech_gait.clinical.contact_forces import JointContactLoads


def render_ascii_gait_diagram(profile: GaitKinematicsProfile) -> str:
    """Render ASCII stick figure of key phases across the 0-100% gait cycle."""
    lines: list[str] = []
    lines.append("=== Sagittal Lower-Limb Gait Cycle Stick Figures ===")
    lines.append("Phase:     0% Heel Strike      30% Midstance        60% Toe-Off        85% Mid-Swing")
    lines.append("")
    lines.append("Pelvis:        (O)                 (O)                 (O)                 (O)      ")
    lines.append("Thigh:        /                   |                     \\                   /       ")
    lines.append("Knee:        O                   O                       O                 O        ")
    lines.append("Shank:        \\                   |                      /                  |       ")
    lines.append("Ankle:         O                   O                    O                   O       ")
    lines.append("Foot:         /__                 ___                  /                     \\      ")
    lines.append("Ground: ============================================================================")

    # Joint kinematics comparison at these 4 phases
    target_pcts = [0, 30, 60, 85]
    lines.append("\nKinematics at Key Events:")
    header = f"{'Event':<18} | {'Hip Angle':<12} | {'Knee Angle':<12} | {'Ankle Angle':<12} | {'GRF (BW)':<10}"
    lines.append(header)
    lines.append("-" * len(header))

    for p in target_pcts:
        idx = int(np.argmin(np.abs(profile.percent_gait_cycle - p)))
        h_ang = f"{profile.hip_flexion_deg[idx]:+5.1f} deg"
        k_ang = f"{profile.knee_flexion_deg[idx]:+5.1f} deg"
        a_ang = f"{profile.ankle_dorsiflexion_deg[idx]:+5.1f} deg"
        grf = f"{profile.grf_vertical_bw[idx]:4.2f} BW"
        name = f"Phase {p}%"
        lines.append(f"{name:<18} | {h_ang:<12} | {k_ang:<12} | {a_ang:<12} | {grf:<10}")

    return "\n".join(lines)


def format_biomechanics_report(
    eval_rep: ClinicalGaitEvaluation,
    loads: JointContactLoads,
    muscles: MuscleDistributionResult,
) -> str:
    """Format comprehensive clinical biomechanics summary."""
    lines: list[str] = []
    lines.append("=" * 76)
    lines.append("  Clinical Musculoskeletal Biomechanics & Rehabilitation Dossier")
    lines.append("=" * 76)
    lines.append(f"Classification:            {eval_rep.pathology.value}")
    lines.append(f"Mean Stance Knee Flexion:  {eval_rep.mean_stance_knee_flexion_deg:.1f} deg")
    lines.append(f"Peak Swing Knee Flexion:   {eval_rep.peak_swing_knee_flexion_deg:.1f} deg")
    lines.append(f"Crouch Severity Index:     {eval_rep.crouch_severity_index:.2f} / 1.00")
    lines.append(f"Stiff-Knee Index:          {eval_rep.stiff_knee_index:.2f} / 1.00")

    lines.append("\nArticular Contact Compressive Loading (Midstance / Peak Stance):")
    lines.append(f"  - Knee Tibiofemoral:     {loads.knee_contact_n:6.1f} N ({loads.knee_contact_bw:4.2f} x BW) -> {loads.tibiofemoral_oa_risk}")
    lines.append(f"  - Ankle Tibiotalar:      {loads.ankle_contact_n:6.1f} N ({loads.ankle_contact_bw:4.2f} x BW)")
    lines.append(f"  - Hip Acetabulofemoral:  {loads.hip_contact_n:6.1f} N ({loads.hip_contact_bw:4.2f} x BW)")

    lines.append("\nMajor Muscle Activation & Estimated Force Production:")
    for name, act in muscles.activations.items():
        force = muscles.forces[name]
        bar = "#" * int(act * 20)
        lines.append(f"  {name:<18}: [{bar:<20}] {act * 100:5.1f}% | {force:6.1f} N")

    lines.append("\nClinical Recommendations:")
    for rec in eval_rep.rehabilitation_recommendations:
        lines.append(f"  * {rec}")

    return "\n".join(lines)
