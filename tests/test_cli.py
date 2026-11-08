"""Unit tests for BiomechGait CLI subcommands."""

from biomech_gait.cli import main


def test_cli_simulate_normal() -> None:
    code = main(["simulate", "--pathology", "normal"])
    assert code == 0


def test_cli_compare() -> None:
    code = main(["compare"])
    assert code == 0
