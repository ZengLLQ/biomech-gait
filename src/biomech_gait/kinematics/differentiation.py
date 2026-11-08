"""Biomechanical filtering and numerical differentiation for kinematic time series."""

from __future__ import annotations

import numpy as np
from scipy import signal


class KinematicFilter:
    """Zero-lag digital Butterworth filter and numerical differentiator.

    Parameters
    ----------
    sampling_rate_hz : float
        MoCap acquisition frequency (default 100.0 Hz).
    cutoff_freq_hz : float
        Low-pass filter cut-off frequency (default 6.0 Hz for human gait).
    order : int
        Filter order (default 4).
    """

    def __init__(
        self,
        sampling_rate_hz: float = 100.0,
        cutoff_freq_hz: float = 6.0,
        order: int = 4,
    ) -> None:
        self.fs = float(sampling_rate_hz)
        self.fc = float(cutoff_freq_hz)
        self.order = int(order)
        # Normalized Nyquist frequency
        nyq = 0.5 * self.fs
        normal_cutoff = self.fc / nyq
        self.b, self.a = signal.butter(self.order, normal_cutoff, btype="low", analog=False)

    def filter_signal(self, data: np.ndarray) -> np.ndarray:
        """Apply bidirectional zero-lag low-pass filter (filtfilt)."""
        if len(data) <= 15:
            # Short array fallback
            return data.copy()
        return signal.filtfilt(self.b, self.a, data, axis=0)

    def differentiate(self, data: np.ndarray, filter_first: bool = True) -> np.ndarray:
        """Compute first-order time derivative (velocity) via central difference."""
        filtered = self.filter_signal(data) if filter_first else data
        dt = 1.0 / self.fs
        return np.gradient(filtered, dt, axis=0)

    def double_differentiate(self, data: np.ndarray, filter_first: bool = True) -> np.ndarray:
        """Compute second-order time derivative (acceleration)."""
        vel = self.differentiate(data, filter_first=filter_first)
        dt = 1.0 / self.fs
        return np.gradient(vel, dt, axis=0)
