"""Example: Inverse dynamics and muscle force distribution in normal adult gait."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from biomech_gait import (
    SagittalKinematicModel,
    NewtonEulerInverseDynamics,
    StaticOptimizationSolver,
    JointContactCalculator,
    GaitPathology,
    GaitPathologySimulator,
    render_ascii_gait_diagram,
    format_biomechanics_report,
)


def main() -> None:
    print("Simulating normal adult human gait cycle...")
    sim = GaitPathologySimulator()
    profile = sim.generate_profile(GaitPathology.NORMAL)
    eval_rep = sim.evaluate_gait(profile)

    print("\n" + render_ascii_gait_diagram(profile))

    # Evaluate at midstance (30% gait cycle)
    idx_30 = int(np.argmin(np.abs(profile.percent_gait_cycle - 30.0)))
    h_rad = float(np.radians(profile.hip_flexion_deg[idx_30]))
    k_rad = float(np.radians(profile.knee_flexion_deg[idx_30]))
    a_rad = float(np.radians(profile.ankle_dorsiflexion_deg[idx_30]))

    mass = 75.0
    kin_model = SagittalKinematicModel(body_mass_kg=mass, body_height_m=1.78)
    pos = kin_model.forward_kinematics(np.array([0.0, 0.98]), h_rad, k_rad, a_rad)

    grf = np.array([0.0, profile.grf_vertical_bw[idx_30] * (mass * 9.81)])
    dynamics = NewtonEulerInverseDynamics(kin_model)
    kinetics = dynamics.solve_instant(
        positions=pos,
        foot_acc_com=np.array([0.0, 0.0]),
        shank_acc_com=np.array([0.0, 0.0]),
        thigh_acc_com=np.array([0.0, 0.0]),
        foot_ang_acc=0.0,
        shank_ang_acc=0.0,
        thigh_ang_acc=0.0,
        ground_reaction_force=grf,
        center_of_pressure=pos.foot_com.copy(),
    )

    opt = StaticOptimizationSolver()
    target_m = np.array([kinetics.ankle_moment, kinetics.knee_moment, kinetics.hip_moment])
    muscles = opt.solve(target_m, h_rad, k_rad, a_rad)

    contact_calc = JointContactCalculator(body_mass_kg=mass)
    loads = contact_calc.compute_contact_forces(kinetics, muscles)

    print("\n" + format_biomechanics_report(eval_rep, loads, muscles))


if __name__ == "__main__":
    main()
