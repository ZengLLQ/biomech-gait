"""Unit tests for musculoskeletal geometry and moment arms."""

from biomech_gait.muscles.musculoskeletal_geometry import MusculoskeletalGeometry


def test_moment_arms_signs() -> None:
    geom = MusculoskeletalGeometry()
    R = geom.get_moment_arms(0.0, 0.0, 0.0)
    assert R.shape == (3, 8)

    # Soleus (index 0) has positive ankle moment arm (plantarflexor)
    assert R[0, 0] > 0.0
    # Tibialis anterior (index 2) has negative ankle moment arm (dorsiflexor)
    assert R[0, 2] < 0.0

    # Vastus group (index 3) has positive knee moment arm (extensor)
    assert R[1, 3] > 0.0
    # Hamstrings (index 5) has negative knee moment arm (flexor)
    assert R[1, 5] < 0.0
