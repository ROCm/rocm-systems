#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from resolve_ci_config import GPU_FAMILIES, TRIGGER_TYPES, load_gpu_configs, main

SAMPLE_RUNNER_CONFIG = {
    "version": "1",
    "build_runners": {
        "linux": {
            "default": [
                {"label": "disabled-linux-runner", "weight": 0.0},
                {"label": "configured-linux-runner", "weight": 1.0},
            ]
        }
    },
    "gpu_families": {
        "presubmit": {
            "gfx94x": {
                "linux": {
                    "test-runs-on": "linux-gfx942-1gpu-ccs-csp-ossci-rocm",
                    "test-runs-on-sandbox": "linux-mi325-gpu-rocm-cpu-sandbox",
                    "family": "gfx94X-dcgpu",
                    "fetch-gfx-targets": ["gfx942"],
                }
            },
            "gfx110x": {
                "linux": {
                    "test-runs-on": "linux-gfx110X-gpu-rocm",
                    "family": "gfx110X-all",
                    "fetch-gfx-targets": [],
                }
            },
            "gfx120x": {
                "linux": {
                    "test-runs-on": "linux-gfx120X-gpu-rocm",
                    "family": "gfx120X-all",
                    "fetch-gfx-targets": ["gfx1200", "gfx1201"],
                }
            },
            "gfx1151": {
                "linux": {
                    "test-runs-on": "linux-gfx1151-gpu-rocm",
                    "family": "gfx1151",
                    "fetch-gfx-targets": ["gfx1151"],
                }
            },
        },
        "postsubmit": {
            "gfx950": {
                "linux": {
                    "test-runs-on": "linux-gfx950-1gpu-ccs-ossci-rocm",
                    "family": "gfx950-dcgpu",
                    "fetch-gfx-targets": ["gfx950"],
                }
            }
        },
        "nightly": {
            "gfx90a": {
                "linux": {
                    "test-runs-on": "linux-gfx90a-gpu-rocm",
                    "family": "gfx90a",
                    "fetch-gfx-targets": ["gfx90a"],
                }
            },
            "gfx103x": {
                "linux": {
                    "test-runs-on": "linux-gfx1030-gpu-rocm",
                    "family": "gfx103X-all",
                    "fetch-gfx-targets": ["gfx1030"],
                }
            },
        },
    },
}


def _write_config(tmpdir: Path, config: dict) -> None:
    (tmpdir / "runner-config.json").write_text(json.dumps(config))
    ci_api = textwrap.dedent("""\
        import json
        from pathlib import Path
        from dataclasses import dataclass
        from typing import Any

        @dataclass
        class ConfigV1:
            build_runners: dict
            gpu_families: dict
            _raw: dict
            def get_gpu_families(self, trigger_types):
                result = {}
                for tt in trigger_types:
                    if tt in self.gpu_families:
                        for name, cfg in self.gpu_families[tt].items():
                            result[name] = cfg
                return result

        def load_config_v1(config_path=None):
            if config_path is None:
                config_path = Path(__file__).parent
            with open(config_path / "runner-config.json") as f:
                raw = json.load(f)
            return ConfigV1(
                build_runners=raw["build_runners"],
                gpu_families=raw["gpu_families"],
                _raw=raw,
            )
    """)
    (tmpdir / "ci_config_api.py").write_text(ci_api)


class TestLoadGpuConfigs(unittest.TestCase):
    def test_returns_empty_when_path_missing(self):
        result = load_gpu_configs(Path("/nonexistent/path"))
        self.assertEqual(result, {})

    def test_loads_all_families_from_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            _write_config(tmpdir, SAMPLE_RUNNER_CONFIG)
            result = load_gpu_configs(tmpdir)

            self.assertEqual(
                result["gfx94x"]["test-runs-on"],
                "linux-gfx942-1gpu-ccs-csp-ossci-rocm",
            )
            self.assertEqual(
                result["gfx950"]["test-runs-on"],
                "linux-gfx950-1gpu-ccs-ossci-rocm",
            )
            self.assertEqual(
                result["gfx90a"]["test-runs-on"],
                "linux-gfx90a-gpu-rocm",
            )
            self.assertEqual(
                result["gfx103x"]["test-runs-on"],
                "linux-gfx1030-gpu-rocm",
            )
            self.assertEqual(
                result["gfx110x"]["test-runs-on"],
                "linux-gfx110X-gpu-rocm",
            )
            self.assertEqual(
                result["gfx120x"]["test-runs-on"],
                "linux-gfx120X-gpu-rocm",
            )
            self.assertEqual(
                result["gfx1151"]["test-runs-on"],
                "linux-gfx1151-gpu-rocm",
            )

    def test_missing_family_returns_empty_dict(self):
        config = {
            "version": "1",
            "build_runners": {},
            "gpu_families": {
                "presubmit": {
                    "gfx94x": {
                        "linux": {
                            "test-runs-on": "some-runner",
                            "family": "f",
                            "fetch-gfx-targets": [],
                        }
                    }
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            _write_config(tmpdir, config)
            result = load_gpu_configs(tmpdir)

            self.assertEqual(result["gfx94x"]["test-runs-on"], "some-runner")
            self.assertEqual(result["gfx950"], {})
            self.assertEqual(result["gfx90a"], {})
            self.assertEqual(result["gfx103x"], {})
            self.assertEqual(result["gfx110x"], {})
            self.assertEqual(result["gfx120x"], {})
            self.assertEqual(result["gfx1151"], {})

    def test_returns_empty_on_import_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "ci_config_api.py").write_text("raise ImportError('broken')")
            (tmpdir / "runner-config.json").write_text("{}")
            result = load_gpu_configs(tmpdir)
            self.assertEqual(result, {})

    def test_sandbox_runner_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            _write_config(tmpdir, SAMPLE_RUNNER_CONFIG)
            result = load_gpu_configs(tmpdir)
            self.assertEqual(
                result["gfx94x"]["test-runs-on-sandbox"],
                "linux-mi325-gpu-rocm-cpu-sandbox",
            )


class TestMain(unittest.TestCase):
    def _run_main(self, config=None, env_overrides=None):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            output_file = tmpdir / "github_output"
            output_file.write_text("")

            if config is not None:
                config_dir = tmpdir / "ci-config"
                config_dir.mkdir()
                _write_config(config_dir, config)
            else:
                config_dir = tmpdir / "no-config"

            env = {
                "CI_CONFIG_PATH": str(config_dir),
                "GITHUB_OUTPUT": str(output_file),
                "FALLBACK_GFX94X_RUNNER": "fallback-gfx94x",
                "FALLBACK_GFX94X_SANDBOX_RUNNER": "fallback-gfx94x-sandbox",
                "FALLBACK_GFX950_RUNNER": "fallback-gfx950",
                "FALLBACK_GFX90A_RUNNER": "fallback-gfx90a",
                "FALLBACK_GFX103X_RUNNER": "fallback-gfx103x",
                "FALLBACK_GFX110X_RUNNER": "fallback-gfx110x",
                "FALLBACK_GFX120X_RUNNER": "fallback-gfx120x",
                "FALLBACK_GFX1151_RUNNER": "fallback-gfx1151",
                "FALLBACK_LINUX_BUILD_RUNNER": "fallback-linux-runner",
            }
            if env_overrides:
                env.update(env_overrides)

            with patch.dict(os.environ, env, clear=False):
                main()

            outputs = {}
            for line in output_file.read_text().strip().splitlines():
                key, _, value = line.partition("=")
                outputs[key] = value
            return outputs

    def test_uses_config_values_when_available(self):
        outputs = self._run_main(config=SAMPLE_RUNNER_CONFIG)

        self.assertEqual(outputs["linux_build_runner"], "configured-linux-runner")
        self.assertEqual(
            outputs["gfx94x_runner"], "linux-gfx942-1gpu-ccs-csp-ossci-rocm"
        )
        self.assertEqual(outputs["gfx950_runner"], "linux-gfx950-1gpu-ccs-ossci-rocm")
        self.assertEqual(outputs["gfx90a_runner"], "linux-gfx90a-gpu-rocm")
        self.assertEqual(outputs["gfx103x_runner"], "linux-gfx1030-gpu-rocm")
        self.assertEqual(outputs["gfx110x_runner"], "linux-gfx110X-gpu-rocm")
        self.assertEqual(outputs["gfx120x_runner"], "linux-gfx120X-gpu-rocm")
        self.assertEqual(outputs["gfx1151_runner"], "linux-gfx1151-gpu-rocm")
        self.assertEqual(
            outputs["gfx94x_sandbox_runner"], "linux-mi325-gpu-rocm-cpu-sandbox"
        )

    def test_falls_back_to_env_when_config_missing(self):
        outputs = self._run_main(config=None)

        self.assertEqual(outputs["linux_build_runner"], "fallback-linux-runner")
        self.assertEqual(outputs["gfx94x_runner"], "fallback-gfx94x")
        self.assertEqual(outputs["gfx950_runner"], "fallback-gfx950")
        self.assertEqual(outputs["gfx90a_runner"], "fallback-gfx90a")
        self.assertEqual(outputs["gfx103x_runner"], "fallback-gfx103x")
        self.assertEqual(outputs["gfx110x_runner"], "fallback-gfx110x")
        self.assertEqual(outputs["gfx120x_runner"], "fallback-gfx120x")
        self.assertEqual(outputs["gfx1151_runner"], "fallback-gfx1151")
        self.assertEqual(outputs["gfx94x_sandbox_runner"], "fallback-gfx94x-sandbox")

    def test_partial_config_uses_fallback_for_missing_families(self):
        partial_config = {
            "version": "1",
            "build_runners": {},
            "gpu_families": {
                "presubmit": {
                    "gfx94x": {
                        "linux": {
                            "test-runs-on": "configured-gfx94x",
                            "test-runs-on-sandbox": "configured-sandbox",
                            "family": "f",
                            "fetch-gfx-targets": [],
                        }
                    }
                }
            },
        }
        outputs = self._run_main(config=partial_config)

        self.assertEqual(outputs["linux_build_runner"], "fallback-linux-runner")
        self.assertEqual(outputs["gfx94x_runner"], "configured-gfx94x")
        self.assertEqual(outputs["gfx94x_sandbox_runner"], "configured-sandbox")
        self.assertEqual(outputs["gfx950_runner"], "fallback-gfx950")
        self.assertEqual(outputs["gfx90a_runner"], "fallback-gfx90a")
        self.assertEqual(outputs["gfx103x_runner"], "fallback-gfx103x")
        self.assertEqual(outputs["gfx110x_runner"], "fallback-gfx110x")
        self.assertEqual(outputs["gfx120x_runner"], "fallback-gfx120x")
        self.assertEqual(outputs["gfx1151_runner"], "fallback-gfx1151")

    def test_all_gpu_families_have_output(self):
        outputs = self._run_main(config=SAMPLE_RUNNER_CONFIG)
        for family in GPU_FAMILIES:
            self.assertIn(f"{family}_runner", outputs, f"Missing output for {family}")
            self.assertTrue(outputs[f"{family}_runner"], f"Empty output for {family}")
        self.assertIn("gfx94x_sandbox_runner", outputs)


class TestConstants(unittest.TestCase):
    def test_gpu_families_not_empty(self):
        self.assertTrue(len(GPU_FAMILIES) > 0)

    def test_trigger_types_cover_all_scopes(self):
        self.assertIn("presubmit", TRIGGER_TYPES)
        self.assertIn("postsubmit", TRIGGER_TYPES)
        self.assertIn("nightly", TRIGGER_TYPES)


if __name__ == "__main__":
    unittest.main()
