"""Run the shipped demos as users do: isolated CLI processes and saved outputs."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "script", ["embedded_demo.py", "regression_demo.py", "cli_demo.py"]
)
def test_demo_completes_and_its_saved_metrics_match_stdout(tmp_path, script):
    directory = tmp_path / script.removesuffix(".py")
    command = [
        sys.executable,
        str(ROOT / "examples" / script),
        "--epochs",
        "3",
        "--samples",
        "40",
        "--seed",
        "17",
        "--output-dir",
        str(directory),
    ]
    if shutil.which("cc"):
        command.append("--verify-c")
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    metrics = json.loads(result.stdout)
    assert metrics == json.loads((directory / "metrics.json").read_text())
    assert metrics["test_samples"] == 8
    assert metrics["embedded_weight_bytes"] <= 1024
    assert metrics["scratch_bytes"] <= 256
    assert (directory / "model.c").exists()
    assert (directory / "model.h").exists()
    assert (directory / "history.json").exists()
    if "--verify-c" in command:
        assert metrics["c_verification"]["samples"] == 8
        assert metrics["c_verification"]["max_absolute_error"] < 2e-5


def test_demo_rejects_unusable_split_and_resource_budget(tmp_path):
    for options in (["--samples", "3"], ["--samples", "16", "--max-weight-bytes", "1"]):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "examples" / "embedded_demo.py"),
                "--epochs",
                "1",
                "--output-dir",
                str(tmp_path / "output"),
                *options,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0
        assert "Traceback" not in result.stderr
        assert "Error" in result.stderr or "error" in result.stderr
