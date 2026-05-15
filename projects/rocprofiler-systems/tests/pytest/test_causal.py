# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
causal tests
"""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest

pytestmark = [pytest.mark.causal]

# ====================================================================================== #
# Causal fixtures
# ====================================================================================== #


@pytest.fixture
def causal_env() -> dict[str, str]:
    return {
        "ROCPROFSYS_TIME_OUTPUT": "OFF",
        "ROCPROFSYS_FILE_OUTPUT": "ON",
        "ROCPROFSYS_CAUSAL_RANDOM_SEED": "1342342",
    }


@pytest.fixture
def causal_e2e_env(causal_env) -> dict[str, str]:
    """Return e2e environment settings based on build number."""
    import os

    env = causal_env.copy()
    build_number = int(os.environ.get("ROCPROFSYS_BUILD_NUMBER", "0"))
    if build_number > 1:
        env["ROCPROFSYS_VERBOSE"] = "0"
        env["ROCPROFSYS_LOG_LEVEL"] = "trace"
    return env


# ====================================================================================== #
# Causal tests
# ====================================================================================== #


class TestCausal(RocprofsysTest):
    PASS_REGEX = [
        r"Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
    ]

    @pytest.mark.parametrize(
        "target",
        [
            pytest.param("causal-cpu-rocprofsys", id="cpu-rocprofsys"),
            pytest.param("causal-both-rocprofsys", id="both-rocprofsys"),
            pytest.param("lulesh-rocprofsys", marks=pytest.mark.lulesh),
        ],
    )
    def test_baseline(self, target):
        result = self.run_test("baseline", target)
        self.assert_regex(result)

    @pytest.mark.parametrize(
        "target, causal_mode",
        [
            pytest.param(
                "causal-cpu-rocprofsys", "function", id="cpu-rocprofsys-function"
            ),
            pytest.param("causal-cpu-rocprofsys", "line", id="cpu-rocprofsys-line"),
            pytest.param(
                "causal-cpu-rocprofsys-ndebug",
                "function",
                id="cpu-rocprofsys-ndebug-function",
            ),
            pytest.param("lulesh-rocprofsys", "function", marks=pytest.mark.lulesh),
            pytest.param(
                "lulesh-rocprofsys-ndebug", "function", marks=pytest.mark.lulesh
            ),
            pytest.param("lulesh-rocprofsys", "line", marks=pytest.mark.lulesh),
        ],
    )
    def test(self, target, causal_mode, causal_env, create_config_file):
        # Not using a config file causes a timeout for lulesh-rocprofsys-function
        env = causal_env.copy()
        config_file = create_config_file(env, "causal.cfg")
        env["ROCPROFSYS_CONFIG_FILE"] = str(config_file)

        if "lulesh" in target:
            run_args = ["-i", "35", "-s", "50", "-p"]
            causal_args = ["-s", "0,10,25,50,75"]
            if causal_mode == "line":
                causal_args.append("-S")
                causal_args.append("lulesh.cc")
        else:
            run_args = ["70", "10", "432525", "1000000000"]
            causal_args = []

        result = self.run_test(
            "causal",
            target,
            env=env,
            causal_mode=causal_mode,
            run_args=run_args,
            causal_args=causal_args,
            timeout=600,
        )
        self.assert_regex(result, pass_regex=self.PASS_REGEX)

    @pytest.mark.parametrize(
        "target, causal_mode",
        [
            pytest.param(
                "causal-both-rocprofsys", "function", id="both-rocprofsys-function"
            ),
        ],
    )
    def test_config(
        self, target, causal_mode, causal_env, create_config_file, test_output_dir
    ):
        # A base config file disabling the strict config needs to be written first
        env = causal_env.copy()
        env["ROCPROFSYS_STRICT_CONFIG"] = "OFF"
        base_config_file = create_config_file(env, "base_config.cfg")
        print(f"Base config file: {base_config_file}")
        env["ROCPROFSYS_CONFIG_FILE"] = str(base_config_file)

        causal_args = [
            "-n",
            "2",
            "-w",
            "1",
            "-d",
            "3",
            "--monochrome",
            "-g",
            str(test_output_dir),
            "-l",
            "causal-both-rocprofsys",
            "-v",
            "3",
            "-b",
            "timer",
        ]
        run_args = ["70", "10", "432525", "400000000"]

        result = self.run_test(
            "causal",
            target,
            env=env,
            causal_mode=causal_mode,
            run_args=run_args,
            causal_args=causal_args,
            timeout=600,
        )
        self.assert_regex(result, pass_regex=self.PASS_REGEX)


@pytest.mark.causal_e2e
@pytest.mark.slow  # Upwards of 120 seconds
@pytest.mark.parametrize(
    "target, causal_mode, test_type",
    [
        pytest.param(
            "causal-cpu-rocprofsys", "func", "slow", id="cpu-rocprofsys-func-slow"
        ),
        pytest.param(
            "causal-cpu-rocprofsys", "func", "fast", id="cpu-rocprofsys-func-fast"
        ),
        pytest.param(
            "causal-cpu-rocprofsys", "line", "103", id="cpu-rocprofsys-line-103"
        ),
        pytest.param(
            "causal-cpu-rocprofsys", "line", "113", id="cpu-rocprofsys-line-113"
        ),
    ],
)
class TestCausalE2E(RocprofsysTest):
    PASS_REGEX = [
        r"Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
    ]
    RUN_ARGS = ["80", "50", "432525", "100000000"]

    def get_causal_args(self, causal_mode, test_type) -> list[str]:
        ret = [
            "-n",
            "5",
            "-e",
            "-s",
            "0",
            "10",
            "20",
            "30",
            "-B",
            "causal-cpu-rocprofsys",
        ]
        if causal_mode == "func":
            ret.append("-F")
            if test_type == "slow":
                ret.append("cpu_slow_func")
            elif test_type == "fast":
                ret.append("cpu_fast_func")
        elif causal_mode == "line":
            ret.append("-S")
            if test_type == "103":
                ret.append("causal.cpp:103")
            elif test_type == "113":
                ret.append("causal.cpp:113")
        return ret

    def get_validation_args(self, causal_mode, test_type) -> list[str]:
        ret = ["-n", "0", "-v"]
        if causal_mode == "func":
            if test_type == "slow":
                ret.extend(["cpu_slow_func", "causal-cpu-rocprofsys", "10", "10", "5"])
                ret.extend(["cpu_slow_func", "causal-cpu-rocprofsys", "20", "20", "5"])
                ret.extend(["cpu_slow_func", "causal-cpu-rocprofsys", "30", "20", "5"])
            if test_type == "fast":
                ret.extend(["cpu_fast_func", "causal-cpu-rocprofsys", "10", "0", "5"])
                ret.extend(["cpu_fast_func", "causal-cpu-rocprofsys", "20", "0", "5"])
                ret.extend(["cpu_fast_func", "causal-cpu-rocprofsys", "30", "0", "5"])
        elif causal_mode == "line":
            if test_type == "103":
                ret.extend(["causal.cpp:103", "causal-cpu-rocprofsys", "10", "10", "5"])
                ret.extend(["causal.cpp:103", "causal-cpu-rocprofsys", "20", "20", "5"])
                ret.extend(["causal.cpp:103", "causal-cpu-rocprofsys", "30", "20", "5"])
            elif test_type == "113":
                ret.extend(["causal.cpp:113", "causal-cpu-rocprofsys", "10", "0", "5"])
                ret.extend(["causal.cpp:113", "causal-cpu-rocprofsys", "20", "0", "5"])
                ret.extend(["causal.cpp:113", "causal-cpu-rocprofsys", "30", "0", "5"])
        return ret

    def test(self, target, causal_mode, test_type, causal_e2e_env):
        causal_args = self.get_causal_args(causal_mode, test_type)
        validation_args = self.get_validation_args(causal_mode, test_type)

        result = self.run_test(
            "causal",
            target,
            env=causal_e2e_env,
            run_args=self.RUN_ARGS,
            causal_mode=causal_mode,
            causal_args=causal_args,
            timeout=600,
        )
        self.assert_regex(result, pass_regex=self.PASS_REGEX)
        self.assert_causal_json(
            result,
            file_name="causal/experiments.json",
            additional_args=validation_args,
        )
