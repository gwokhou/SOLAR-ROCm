# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the unsupported Linux aarch64 build target."""

from pathlib import Path
import runpy
import sys
import tomllib
import types
from unittest import mock

from packaging.markers import Marker
from packaging.requirements import Requirement
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("machine", ["aarch64", "arm64"])
def test_setup_rejects_linux_aarch64(machine):
    fake_setuptools = types.SimpleNamespace(
        setup=mock.Mock(), find_packages=mock.Mock()
    )

    with (
        mock.patch.dict(sys.modules, {"setuptools": fake_setuptools}),
        mock.patch("sys.platform", "linux"),
        mock.patch("platform.machine", return_value=machine),
            pytest.raises(RuntimeError, match="supports Linux x86_64 only"),
    ):
        runpy.run_path(str(ROOT / "setup.py"), run_name="__main__")


def test_dependency_markers_exclude_linux_aarch64():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        pyproject = tomllib.load(stream)
    requirements = [
        line
        for line in (ROOT / "requirements.txt").read_text().splitlines()
        if line.startswith("torch")
    ]
    environment = {"sys_platform": "linux", "platform_machine": "aarch64"}

    project_torch = [
        requirement
        for value in pyproject["project"]["dependencies"]
        if (requirement := Requirement(value)).name == "torch"
    ]
    requirements_torch = [
        requirement
        for value in requirements
        if (requirement := Requirement(value)).name == "torch"
    ]

    assert all(
        not requirement.marker.evaluate(environment)
        for requirement in project_torch
    )
    assert all(
        not requirement.marker.evaluate(environment)
        for requirement in requirements_torch
    )
    assert all(
        not Marker(value).evaluate(environment)
        for value in pyproject["tool"]["uv"]["environments"]
    )
