"""Command line interface for BiomechGait musculoskeletal rehabilitation toolkit."""

from __future__ import annotations

import argparse
import sys
import numpy as np

from biomech_gait.kinematics.coordinates import SagittalKinematicModel
from biomech_gait.dynamics.inverse_dynamics import NewtonEulerInverseDynamics
from biomech_gait.optimization.muscle_distribution import StaticOptimizationSolver
from biomech_gait.clinical.contact_forces import JointContactCalculator
from biomech_gait.clinical.pathology_analyzer import (
    GaitPathology,
    GaitPathologySimulator,
)
from biomech_gait.visualizer import (
    render_ascii_gait_diagram,
    format_biomechanics_report,
)


def handle_simulate(args: argparse.Namespace) -> int:
    """Run full inverse dynamics and muscle distribution simulation."""
    path_map = {
        "normal": GaitPathology.NORMAL,
        "crouch": GaitPathology.CROUCH_GAIT,
        "stiff_knee": GaitPathology.STIFF_KNEE,
    }
    target_pathology = path_map.get(args.pathology.lower(), GaitPathology.NORMAL)

    print("=" * 76)
    print(f"  BiomechGait Simulation: {target_pathology.value}")
    print("=" * 76)

    simulator = GaitPathologySimulator()
    profile = simulator.generate_profile(target_pathology)

    print("\n" + render_ascii_gait_diagram(profile))

    # Evaluate clinical classification
    eval_rep = simulator.evaluate_gait(profile)

    # Solve dynamics and muscle force distribution at midstance (30% gait cycle)
    mid_idx = int(np.argmin(np.abs(profile.percent_gait_cycle - 30.0)))
    h_rad = float(np.radians(profile.hip_flexion_deg[mid_idx]))
    k_rad = float(np.radians(profile.knee_flexion_deg[mid_idx]))
    a_rad = float(np.radians(profile.ankle_dorsiflexion_deg[mid_idx]))

    kin_model = SagittalKinematicModel(body_mass_kg=args.mass, body_height_m=args.height)
    pos = kin_model.forward_kinematics(np.array([0.0, 0.95]), h_rad, k_rad, a_rad)

    # GRF at midstance: mostly vertical support
    grf_v_n = profile.grf_vertical_bw[mid_idx] * (args.mass * 9.81)
    grf_ap_n = profile.grf_anterior_posterior_bw[mid_idx] * (args.mass * 9.81)
    grf = np.array([grf_ap_n, grf_v_n])
    cop = pos.foot_com.copy()

    dynamics = NewtonEulerInverseDynamics(kin_model)
    # Midstance acceleration is modest quasi-static
    kinetics = dynamics.solve_instant(
        positions=pos,
        foot_acc_com=np.array([0.0, 0.0]),
        shank_acc_com=np.array([0.0, 0.0]),
        thigh_acc_com=np.array([0.0, 0.0]),
        foot_ang_acc=0.0,
        shank_ang_acc=0.0,
        thigh_ang_acc=0.0,
        ground_reaction_force=grf,
        center_of_pressure=cop,
    )

    opt = StaticOptimizationSolver()
    target_moments = np.array([kinetics.ankle_moment, kinetics.knee_moment, kinetics.hip_moment])
    muscles = opt.solve(target_moments, h_rad, k_rad, a_rad)

    contact_calc = JointContactCalculator(body_mass_kg=args.mass)
    loads = contact_calc.compute_contact_forces(kinetics, muscles)

    print("\n" + format_biomechanics_report(eval_rep, loads, muscles))
    return 0


def handle_compare(args: argparse.Namespace) -> int:
    """Compare normal adult gait vs CP crouch gait."""
    print("=" * 76)
    print("  BiomechGait Clinical Comparison: Normal vs Cerebral Palsy Crouch Gait")
    print("=" * 76)

    simulator = GaitPathologySimulator()
    p_norm = simulator.generate_profile(GaitPathology.NORMAL)
    p_crouch = simulator.generate_profile(GaitPathology.CROUCH_GAIT)

    e_norm = simulator.evaluate_gait(p_norm)
    e_crouch = simulator.evaluate_gait(p_crouch)

    header = f"{'Biomechanical Metric':<32} | {'Normal Adult':<18} | {'CP Crouch Gait':<18}"
    print(header)
    print("-" * len(header))
    print(f"{'Mean Stance Knee Flexion':<32} | {e_norm.mean_stance_knee_flexion_deg:6.1f} deg        | {e_crouch.mean_stance_knee_flexion_deg:6.1f} deg")
    print(f"{'Peak Swing Knee Flexion':<32} | {e_norm.peak_swing_knee_flexion_deg:6.1f} deg        | {e_crouch.peak_swing_knee_flexion_deg:6.1f} deg")
    print(f"{'Crouch Severity Index':<32} | {e_norm.crouch_severity_index:6.2f} / 1.00      | {e_crouch.crouch_severity_index:6.2f} / 1.00")
    print(f"{'Primary Clinical Indication':<32} | {'Normative corridor':<18} | {'Hamstring Lengthening':<18}")

    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI main entry point."""
    parser = argparse.ArgumentParser(
        prog="biomech-gait",
        description="Musculoskeletal Inverse Dynamics & Hill-Type Muscle Force Distribution",
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    sim_p = subparsers.add_parser("simulate", help="Run gait simulation and muscle force distribution")
    sim_p.add_argument(
        "--pathology",
        choices=["normal", "crouch", "stiff_knee"],
        default="normal",
        help="Gait condition (default: normal)",
    )
    sim_p.add_argument("--mass", type=float, default=70.0, help="Body mass in kg")
    sim_p.add_argument("--height", type=float, default=1.75, help="Body height in meters")

    comp_p = subparsers.add_parser("compare", help="Compare normal vs pathological gait kinematics")

    args = parser.parse_args(argv)
    if args.command == "simulate":
        return handle_simulate(args)
    elif args.command == "compare":
        return handle_compare(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
