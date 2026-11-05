# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""GPU-free tests for ATT resource-mode selection and environment forwarding."""

import json

import pytest


@pytest.fixture
def launch(rocprofv3, monkeypatch):
    environments = []
    monkeypatch.delenv("ROCPROF_ATT_PARAM_RESOURCE_MODE", raising=False)
    monkeypatch.setattr(rocprofv3, "resolve_library_path", lambda path, args: path)
    monkeypatch.setattr(rocprofv3, "check_att_capability", lambda args: True)
    monkeypatch.setattr(
        rocprofv3.os,
        "execvpe",
        lambda executable, argv, env: environments.append(env),
    )
    monkeypatch.setattr(
        rocprofv3.subprocess,
        "check_call",
        lambda argv, env: environments.append(env) or 0,
    )

    def run(*args):
        rocprofv3.main(["--att", *args, "--", "/bin/true"])
        assert len(environments) == 1
        return environments[0]

    return run


@pytest.mark.parametrize("mode", ["default", "hsa", "code-object"])
def test_resource_mode_cli(launch, monkeypatch, mode):
    monkeypatch.setenv("ROCPROF_ATT_PARAM_RESOURCE_MODE", "hsa")
    env = launch("--att-resource-mode", mode)
    assert env["ROCPROF_ATT_PARAM_RESOURCE_MODE"] == mode


def test_resource_mode_omitted_uses_tool_default(launch):
    assert "ROCPROF_ATT_PARAM_RESOURCE_MODE" not in launch()


@pytest.mark.parametrize("mode", ["default", "hsa", "code-object"])
def test_resource_mode_omitted_preserves_environment(launch, monkeypatch, mode):
    monkeypatch.setenv("ROCPROF_ATT_PARAM_RESOURCE_MODE", mode)
    assert launch()["ROCPROF_ATT_PARAM_RESOURCE_MODE"] == mode


@pytest.mark.parametrize("mode", ["hip", "all", "dispatch", "invalid", "4"])
def test_resource_mode_invalid_cli(rocprofv3, mode):
    with pytest.raises(SystemExit) as exc:
        rocprofv3.parse_arguments(["--att", "--att-resource-mode", mode])
    assert exc.value.code == 2


@pytest.mark.parametrize("mode", ["default", "hsa", "code-object"])
def test_resource_mode_json(launch, tmp_path, monkeypatch, mode):
    monkeypatch.setenv("ROCPROF_ATT_PARAM_RESOURCE_MODE", "hsa")
    path = tmp_path / "input.json"
    path.write_text(json.dumps({"jobs": [{"att_resource_mode": mode}]}))
    assert launch("-i", str(path))["ROCPROF_ATT_PARAM_RESOURCE_MODE"] == mode


def test_resource_mode_yaml(launch, tmp_path):
    pytest.importorskip("yaml")
    path = tmp_path / "input.yaml"
    path.write_text("jobs:\n  - att_resource_mode: code-object\n")
    assert launch("-i", str(path))["ROCPROF_ATT_PARAM_RESOURCE_MODE"] == "code-object"


def test_resource_mode_invalid_json(launch, tmp_path):
    path = tmp_path / "input.json"
    path.write_text(json.dumps({"jobs": [{"att_resource_mode": "invalid"}]}))
    with pytest.raises(SystemExit) as exc:
        launch("-i", str(path))
    assert exc.value.code != 0
