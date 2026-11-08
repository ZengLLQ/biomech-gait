"""BiomechGait: Musculoskeletal Inverse Dynamics & Hill-Type Muscle Force Distribution for Clinical Gait Rehabilitation."""

from biomech_gait.kinematics.coordinates import (
    SegmentParameters,
    DEFAULT_ANTHROPOMETRY,
    JointPositions,
    SagittalKinematicModel,
)
from biomech_gait.kinematics.differentiation import KinematicFilter
from biomech_gait.dynamics.inverse_dynamics import (
    JointKinetics,
    NewtonEulerInverseDynamics,
)
from biomech_gait.muscles.hill_model import (
    MuscleParameters,
    HillMuscleActuator,
)
from biomech_gait.muscles.musculoskeletal_geometry import (
    LOWER_LIMB_MUSCLES,
    MusculoskeletalGeometry,
)
from biomech_gait.optimization.muscle_distribution import (
    MuscleDistributionResult,
    StaticOptimizationSolver,
)
from biomech_gait.clinical.contact_forces import (
    JointContactLoads,
    JointContactCalculator,
)
from biomech_gait.clinical.pathology_analyzer import (
    GaitPathology,
    GaitKinematicsProfile,
    ClinicalGaitEvaluation,
    GaitPathologySimulator,
)
from biomech_gait.visualizer import (
    render_ascii_gait_diagram,
    format_biomechanics_report,
)

__version__ = "0.1.0"

__all__ = [
    "SegmentParameters",
    "DEFAULT_ANTHROPOMETRY",
    "JointPositions",
    "SagittalKinematicModel",
    "KinematicFilter",
    "JointKinetics",
    "NewtonEulerInverseDynamics",
    "MuscleParameters",
    "HillMuscleActuator",
    "LOWER_LIMB_MUSCLES",
    "MusculoskeletalGeometry",
    "MuscleDistributionResult",
    "StaticOptimizationSolver",
    "JointContactLoads",
    "JointContactCalculator",
    "GaitPathology",
    "GaitKinematicsProfile",
    "ClinicalGaitEvaluation",
    "GaitPathologySimulator",
    "render_ascii_gait_diagram",
    "format_biomechanics_report",
]
