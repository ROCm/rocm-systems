# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
General configuration file tests.
"""

from __future__ import annotations
import pytest
from pathlib import Path
import shutil
from conftest import RocprofsysTest

pytestmark = [pytest.mark.rocprof_config]

MINIMAL_RUNTIME_ENV = {
    "ROCPROFSYS_TRACE": "OFF",
    "ROCPROFSYS_PROFILE": "OFF",
    "ROCPROFSYS_USE_SAMPLING": "OFF",
    "ROCPROFSYS_USE_PROCESS_SAMPLING": "OFF",
    "ROCPROFSYS_USE_AMD_SMI": "OFF",
    "ROCPROFSYS_MONOCHROME": "ON",
}

INVALID_TYPED_VALUE_CASES = [
    pytest.param(
        {"ROCPROFSYS_MODE": "banana"},
        r"[Ii]nvalid value.*ROCPROFSYS_MODE",
        id="mode",
    ),
    pytest.param(
        {"ROCPROFSYS_PERFETTO_BACKEND": "banana"},
        r"[Ii]nvalid value.*ROCPROFSYS_PERFETTO_BACKEND",
        id="perfetto-backend",
    ),
    pytest.param(
        {"ROCPROFSYS_TRACE": "maybe"},
        r"[Ii]nvalid value.*ROCPROFSYS_TRACE",
        id="boolean",
    ),
    pytest.param(
        {"ROCPROFSYS_TRACE": ""},
        r"[Ii]nvalid value.*ROCPROFSYS_TRACE",
        id="boolean-empty",
    ),
    pytest.param(
        {"ROCPROFSYS_TRACE": "   "},
        r"[Ii]nvalid value.*ROCPROFSYS_TRACE",
        id="boolean-whitespace",
    ),
    pytest.param(
        {"ROCPROFSYS_TRACE_DURATION": "abc"},
        r"[Ii]nvalid value.*ROCPROFSYS_TRACE_DURATION",
        id="numeric-parse",
    ),
    pytest.param(
        {"ROCPROFSYS_USE_SAMPLING": "ON", "ROCPROFSYS_SAMPLING_FREQ": "-1"},
        r"[Ii]nvalid value.*ROCPROFSYS_SAMPLING_FREQ",
        id="numeric-range",
    ),
]

VALID_BOOLEAN_VALUES = [
    pytest.param("on", id="on"),
    pytest.param("1", id="one"),
    pytest.param("+1", id="plus-one"),
    pytest.param("-1", id="minus-one"),
    pytest.param("true", id="true"),
    pytest.param("Y", id="Y"),
]

CONFIG_READ_ERROR_REGEX = r"Exception reading|Error reading configuration"
CONFIG_SIGABRT_REGEX = r"terminate called"

BAD_XML_CASES = [
    pytest.param(
        "malformed.xml",
        "<rocprofiler-systems><settings><ROCPROFSYS_TRACE>false</settings></rocprofiler-systems>",
        [CONFIG_READ_ERROR_REGEX],
        id="malformed",
    ),
    pytest.param(
        "wrong_root.xml",
        "<wrong-root><settings/></wrong-root>",
        [CONFIG_READ_ERROR_REGEX],
        id="wrong-root",
    ),
    pytest.param(
        "empty.xml",
        "",
        [CONFIG_READ_ERROR_REGEX],
        id="empty",
    ),
    pytest.param(
        "random.xml",
        "<not-timemory-xml/>",
        [CONFIG_READ_ERROR_REGEX],
        id="random",
    ),
]

VALID_NON_BOOLEAN_TYPED_VALUE_CASES = [
    pytest.param(
        {"ROCPROFSYS_MODE": "trace"},
        r"[Ii]nvalid value.*ROCPROFSYS_MODE",
        id="mode",
    ),
    pytest.param(
        {"ROCPROFSYS_PERFETTO_BACKEND": "inprocess"},
        r"[Ii]nvalid value.*ROCPROFSYS_PERFETTO_BACKEND",
        id="perfetto-backend",
    ),
    pytest.param(
        {"ROCPROFSYS_TRACE_DURATION": "1.25"},
        r"[Ii]nvalid value.*ROCPROFSYS_TRACE_DURATION",
        id="numeric-parse",
    ),
    pytest.param(
        {"ROCPROFSYS_USE_SAMPLING": "ON", "ROCPROFSYS_SAMPLING_FREQ": "50"},
        r"[Ii]nvalid value.*ROCPROFSYS_SAMPLING_FREQ",
        id="numeric-range",
    ),
]


# =============================================================================
# Config fixtures
# =============================================================================


def _write_text(test_output_dir: Path, name: str, content: str) -> Path:
    path = test_output_dir / name
    path.write_text(content)
    return path


def _true_cmd() -> str:
    cmd = shutil.which("true")
    if cmd is None:
        pytest.skip("true not found")
    return cmd


# `ls` cannot be used as the config-invalid target as it has no instrumentable
# functions in the executable itself, so the instrumented process never
# initializes the runtime far enough to validate the config and abort
# on the unknown setting
@pytest.fixture
def config_target(rocprof_config) -> str:
    """Get the target executable for config tests."""
    target_name = "parallel-overhead"
    try:
        rocprof_config.get_target_executable(target_name)
    except FileNotFoundError:
        # Fall back to system ls command
        target_name = shutil.which("ls") or "ls"
    return target_name


# =============================================================================
# Configuration file tests
# =============================================================================


def _run_invalid_config_via_c_flag(
    test: RocprofsysTest, config_target: str, config_file: Path, pass_regex: list[str]
) -> None:
    result = test.run_test(
        "sys_run",
        target=config_target,
        env=MINIMAL_RUNTIME_ENV,
        sys_run_args=["-c", str(config_file)],
        fail_on_pass=True,
    )
    test.assert_regex(
        result,
        pass_regex=pass_regex,
        fail_regex=[CONFIG_SIGABRT_REGEX],
        use_abort_fail_regex=False,
    )


class TestConfig(RocprofsysTest):
    """Tests for configuration file tests."""

    def run_invalid_config_test(self, config_target, env):
        """Run the target through the runtime path expected to reject bad config."""
        return self.run_test(
            "sys_run",
            target=config_target,
            env=env,
            fail_on_pass=True,
        )

    def test_invalid(self, config_target, create_config_file):
        """Test that invalid config file causes failure."""
        # Write invalid configuration file to test output directory
        config_env = {
            "ROCPROFSYS_CONFIG_FILE": "",
            "FOOBAR": "ON",
        }
        config_file = create_config_file(config_env, "invalid.cfg", skip_filter=True)

        env = MINIMAL_RUNTIME_ENV.copy()
        env["ROCPROFSYS_CONFIG_FILE"] = str(config_file)

        result = self.run_invalid_config_test(config_target, env)

        self.assert_regex(
            result,
            pass_regex=[r"Unknown setting 'FOOBAR' \(value = 'ON'\)"],
            use_abort_fail_regex=False,
        )

    @pytest.mark.parametrize("source", ["environment", "config"])
    @pytest.mark.parametrize("values, pass_regex", INVALID_TYPED_VALUE_CASES)
    def test_invalid_typed_values(
        self, config_target, create_config_file, source, values, pass_regex
    ):
        """Test invalid strongly typed values fail clearly via env and config file."""
        env = MINIMAL_RUNTIME_ENV.copy()
        if source == "environment":
            env.update(values)
        else:
            config_name = f"invalid_{next(iter(values)).lower()}.cfg"
            config_file = create_config_file(values, config_name, skip_filter=True)
            env["ROCPROFSYS_CONFIG_FILE"] = str(config_file)

        result = self.run_invalid_config_test(config_target, env)

        self.assert_regex(
            result,
            pass_regex=[pass_regex],
            use_abort_fail_regex=False,
        )

    @pytest.mark.parametrize("value", VALID_BOOLEAN_VALUES)
    def test_valid_boolean_environment_values_are_not_rejected(
        self, config_target, create_config_file, value
    ):
        """Test accepted boolean spellings are not rejected by strict validation."""
        # Force a later MODE error so this checks TRACE parsing without running
        # a traced workload, which is not stable in this debug build.
        config_file = create_config_file(
            {"ROCPROFSYS_MODE": "banana"},
            f"valid_bool_trace_{value}.cfg",
            skip_filter=True,
        )
        env = MINIMAL_RUNTIME_ENV.copy()
        env["ROCPROFSYS_CONFIG_FILE"] = str(config_file)
        env["ROCPROFSYS_TRACE"] = value

        result = self.run_invalid_config_test(config_target, env)

        self.assert_regex(
            result,
            pass_regex=[r"[Ii]nvalid value.*ROCPROFSYS_MODE"],
            fail_regex=[r"[Ii]nvalid value.*ROCPROFSYS_TRACE"],
            use_abort_fail_regex=False,
        )

    @pytest.mark.parametrize("source", ["environment", "config"])
    @pytest.mark.parametrize("values, absent_regex", VALID_NON_BOOLEAN_TYPED_VALUE_CASES)
    def test_valid_non_boolean_values_are_not_rejected(
        self, config_target, create_config_file, source, values, absent_regex
    ):
        """Test valid non-boolean values pass strict validation via env and config."""
        config_values = {"ROCPROFSYS_TRACE": "maybe"}
        env = MINIMAL_RUNTIME_ENV.copy()
        if source == "environment":
            env.update(values)
        else:
            config_values.update(values)

        config_file = create_config_file(
            config_values,
            f"valid_{source}_{next(iter(values)).lower()}.cfg",
            skip_filter=True,
        )
        env["ROCPROFSYS_CONFIG_FILE"] = str(config_file)

        result = self.run_invalid_config_test(config_target, env)

        self.assert_regex(
            result,
            pass_regex=[r"[Ii]nvalid value.*ROCPROFSYS_TRACE"],
            fail_regex=[absent_regex],
            use_abort_fail_regex=False,
        )

    def test_variable_boolean_value_is_not_rejected(
        self, config_target, create_config_file
    ):
        """Test config variable references survive strict raw pre-validation."""
        config_file = create_config_file(
            {
                "$ENABLE": "ON",
                "ROCPROFSYS_TRACE": "$ENABLE",
                "ROCPROFSYS_MODE": "banana",
            },
            "valid_bool_trace_variable.cfg",
            skip_filter=True,
        )
        env = MINIMAL_RUNTIME_ENV.copy()
        env["ROCPROFSYS_CONFIG_FILE"] = str(config_file)

        result = self.run_invalid_config_test(config_target, env)

        self.assert_regex(
            result,
            pass_regex=[r"[Ii]nvalid value.*ROCPROFSYS_MODE"],
            fail_regex=[r"[Ii]nvalid value.*ROCPROFSYS_TRACE"],
            use_abort_fail_regex=False,
        )

    def test_cli_flag_rejects_invalid(self, config_target, create_config_file):
        """A bad config passed with -c should be rejected, just like the env var.

        The tests above already cover the env var, so here we just pass the same
        bad file through -c to make sure that path is validated too. One binary is
        enough since both share the same arg parsing.
        """
        config_file = create_config_file(
            {"ROCPROFSYS_TRACE_DURATION": "not-a-number"},
            "cli_flag_invalid_trace_duration.cfg",
            skip_filter=True,
        )
        result = self.run_test(
            "sys_run",
            target=config_target,
            env=MINIMAL_RUNTIME_ENV,
            sys_run_args=["-c", str(config_file)],
            fail_on_pass=True,
        )
        self.assert_regex(
            result,
            pass_regex=[r"[Ii]nvalid value.*ROCPROFSYS_TRACE_DURATION"],
            use_abort_fail_regex=False,
        )

    @pytest.mark.timeout(120)
    def test_missing(self, test_output_dir: Path, config_target: str):
        """Test that missing config file causes failure."""
        # Use a path to a config file that doesn't exist
        missing_config = test_output_dir / "missing.cfg"

        env = MINIMAL_RUNTIME_ENV.copy()
        env["ROCPROFSYS_CONFIG_FILE"] = str(missing_config)

        result = self.run_invalid_config_test(config_target, env)

        self.assert_regex(
            result,
            pass_regex=[r"Error reading configuration file"],
            use_abort_fail_regex=False,
        )

    @pytest.mark.timeout(120)
    def test_trace_category_enabled_in_runtime(self, config_target: str):
        """Perfetto settings must appear in the runtime config print when tracing is on.

        Regression for the phantom ``ROCPROFSYS_USE_TRACE`` key: the perfetto
        setting category was gated on an unregistered key, so it was always
        disabled at runtime and every perfetto setting (including
        ``ROCPROFSYS_TRACE``) was silently dropped from the printed
        configuration. The canonical switch is ``ROCPROFSYS_TRACE``.
        """
        env = {
            "ROCPROFSYS_TRACE": "ON",
            "ROCPROFSYS_VERBOSE": "2",
        }

        result = self.run_test("sampling", target=config_target, env=env)

        self.assert_regex(
            result,
            pass_regex=[
                r"ROCPROFSYS_TRACE\s+=\s+(true|false)",
                r"ROCPROFSYS_PERFETTO_\w+\s+=",
            ],
        )


@pytest.mark.sys_run
@pytest.mark.timeout(120)
@pytest.mark.class_name("config-xml")
class TestConfigXml(RocprofsysTest):
    """Load and reject XML configs through rocprof-sys-run -c."""

    def _generate_avail_xml(
        self,
        test_output_dir: Path,
        *,
        basename: str = "rocprof-sys-test",
        env: dict[str, str] | None = None,
    ) -> Path:
        config_base = test_output_dir / basename
        self.run_test(
            "baseline",
            target="rocprof-sys-avail",
            run_args=["-G", str(config_base) + ".cfg", "-F", "xml", "--force"],
            env=env,
            fail_on_not_found=True,
        )
        return test_output_dir / f"{basename}.xml"

    @pytest.mark.parametrize(
        "basename, avail_env",
        [
            pytest.param("rocprof-sys-test", None, id="avail-default"),
            pytest.param(
                "rocprof-sys-tweak",
                {
                    "ROCPROFSYS_TRACE": "OFF",
                    "ROCPROFSYS_PROFILE": "ON",
                    "ROCPROFSYS_USE_SAMPLING": "OFF",
                },
                id="avail-env-tweak",
            ),
        ],
    )
    def test_good_via_c_flag(self, config_target, test_output_dir, basename, avail_env):
        xml_path = self._generate_avail_xml(
            test_output_dir, basename=basename, env=avail_env
        )
        result = self.run_test(
            "sys_run",
            target=config_target,
            env=MINIMAL_RUNTIME_ENV,
            sys_run_args=["-c", str(xml_path), "-v", "2"],
        )
        self.assert_regex(
            result,
            pass_regex=[rf"ROCPROFSYS_CONFIG_FILE\s*=\s*.*{xml_path.name}"],
        )

    @pytest.mark.parametrize("filename, content, pass_regex", BAD_XML_CASES)
    def test_invalid_via_c_flag(
        self, config_target, test_output_dir, filename, content, pass_regex
    ):
        xml_path = _write_text(test_output_dir, filename, content)
        _run_invalid_config_via_c_flag(self, config_target, xml_path, pass_regex)

    def test_missing_via_c_flag(self, test_output_dir, config_target):
        missing_xml = test_output_dir / "missing.xml"
        _run_invalid_config_via_c_flag(
            self, config_target, missing_xml, [CONFIG_READ_ERROR_REGEX]
        )


@pytest.mark.sys_run
@pytest.mark.timeout(120)
@pytest.mark.class_name("export-config-reload")
class TestExportConfigReload(RocprofsysTest):
    """Reload --export-config JSON with -c (must fail) vs --preset (must load)."""

    def _export_balanced(self, test_output_dir: Path) -> Path:
        exported = test_output_dir / "cfg.json"
        self.run_test(
            "baseline",
            target="rocprof-sys-run",
            run_args=[
                f"--export-config={exported}",
                "--preset=balanced",
                "--",
                _true_cmd(),
            ],
            fail_on_not_found=True,
        )
        return exported

    def test_rejected_via_c_flag(self, config_target, test_output_dir):
        exported = self._export_balanced(test_output_dir)
        _run_invalid_config_via_c_flag(
            self,
            config_target,
            exported,
            [
                rf"Exception reading.*{exported.name}"
                r"|missing the expected '.*' root object",
            ],
        )

    def test_loaded_via_preset_flag(self, config_target, test_output_dir):
        exported = self._export_balanced(test_output_dir)
        self.run_test(
            "sys_run",
            target=config_target,
            env=MINIMAL_RUNTIME_ENV,
            sys_run_args=[f"--preset={exported}"],
        )
