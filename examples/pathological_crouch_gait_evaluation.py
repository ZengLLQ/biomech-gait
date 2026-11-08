"""Example: Evaluation of Cerebral Palsy crouch gait and quadriceps overload."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from biomech_gait import (
    GaitPathology,
    GaitPathologySimulator,
    render_ascii_gait_diagram,
)


def main() -> None:
    print("Simulating Cerebral Palsy crouch gait kinematics...")
    sim = GaitPathologySimulator()
    profile = sim.generate_profile(GaitPathology.CROUCH_GAIT)
    eval_rep = sim.evaluate_gait(profile)

    print("\n" + render_ascii_gait_diagram(profile))
    print("\nClinical Impairment Findings:")
    print(f"Diagnosis:                  {eval_rep.pathology.value}")
    print(f"Mean Stance Knee Flexion:   {eval_rep.mean_stance_knee_flexion_deg:.1f} deg (Normal < 15 deg)")
    print(f"Crouch Severity Index:      {eval_rep.crouch_severity_index:.2f} / 1.00")
    print("\nRehabilitation Guidance:")
    for rec in eval_rep.rehabilitation_recommendations:
        print(f"  * {rec}")


if __name__ == "__main__":
    main()
