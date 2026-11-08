"""Unit tests for kinematic filtering and numerical differentiation."""

import numpy as np
from biomech_gait.kinematics.differentiation import KinematicFilter


def test_differentiation_sine_wave() -> None:
    fs = 100.0
    t = np.linspace(0.0, 1.0, int(fs), endpoint=False)
    freq = 1.0  # 1 Hz oscillation (well below 6 Hz cutoff)
    # y = sin(2*pi*f*t), dy/dt = 2*pi*f * cos(2*pi*f*t)
    y = np.sin(2 * np.pi * freq * t)

    kf = KinematicFilter(sampling_rate_hz=fs, cutoff_freq_hz=6.0)
    vel = kf.differentiate(y, filter_first=False)
    expected_vel = 2 * np.pi * freq * np.cos(2 * np.pi * freq * t)

    # Check central region (exclude edge boundary effects)
    assert np.allclose(vel[10:-10], expected_vel[10:-10], atol=0.1)
