# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import test_api_contract as contract

HEADER = """
/* typedef enum { AMDSMI_COMMENT } amdsmi_test_t; */
typedef enum {
    AMDSMI_TEST_FIRST,
    AMDSMI_TEST_ALIAS = AMDSMI_TEST_FIRST, // AMDSMI_COMMENT
    AMDSMI_TEST_LAST /* { , AMDSMI_COMMENT } */,
} amdsmi_test_t;
"""
MEMBERS = ["AMDSMI_TEST_FIRST", "AMDSMI_TEST_ALIAS", "AMDSMI_TEST_LAST"]
MODULE = "module github.com/ROCm/rocm-systems/projects/amdsmi/go\n\ngo 1.20\n"
ENUM_OWNERS = {
    "amdsmi_interface.go": (
        ("amdsmi_status_t", "StatusCode"),
        ("amdsmi_vram_type_t", "VRAMType"),
        ("amdsmi_fw_block_t", "FirmwareBlock"),
        ("amdsmi_temperature_type_t", "TemperatureType"),
        ("amdsmi_temperature_metric_t", "TemperatureMetric"),
        ("amdsmi_clk_type_t", "ClockType"),
        ("amdsmi_memory_type_t", "MemoryType"),
        ("amdsmi_memory_partition_type_t", "MemoryPartitionType"),
        ("amdsmi_accelerator_partition_type_t", "AcceleratorPartitionType"),
        ("amdsmi_gpu_block_t", "GPUBlock"),
        ("amdsmi_ras_err_state_t", "RASState"),
    )
}
ALLOWED_CALLS = set(
    """
amdsmi_init
amdsmi_shut_down
amdsmi_status_code_to_string
amdsmi_get_lib_version
amdsmi_get_socket_handles
amdsmi_get_processor_handles
amdsmi_get_processor_type
amdsmi_get_processor_handle_from_bdf
amdsmi_get_gpu_device_bdf
amdsmi_get_gpu_device_uuid
amdsmi_get_gpu_asic_info
amdsmi_get_gpu_driver_info
amdsmi_get_gpu_board_info
amdsmi_get_fw_info
amdsmi_get_gpu_vbios_info
amdsmi_get_temp_metric
amdsmi_get_power_info
amdsmi_get_power_cap_info
amdsmi_get_clock_info
amdsmi_get_clk_freq
amdsmi_get_gpu_activity
amdsmi_get_gpu_memory_total
amdsmi_get_gpu_memory_usage
amdsmi_get_gpu_vram_info
amdsmi_get_gpu_kfd_info
amdsmi_get_gpu_memory_partition_config
amdsmi_get_gpu_accelerator_partition_profile
amdsmi_get_gpu_ecc_enabled
amdsmi_get_gpu_ecc_count
amdsmi_get_gpu_total_ecc_count
amdsmi_get_gpu_ras_block_features_enabled
amdsmi_get_gpu_ras_feature_info
""".split()
)


def bindings(members: list, go_type: str) -> str:
    return (
        "const (\n"
        + "\n".join("{} {} = C.{}".format(name, go_type, name) for name in members)
        + "\n)\n"
    )


class EnumTests(unittest.TestCase):
    def test_extracts_members_and_aliases_without_comments(self) -> None:
        self.assertTrue(
            callable(getattr(contract, "enum_members", None)), "header enum parser is missing"
        )
        self.assertEqual(contract.enum_members(HEADER, "amdsmi_test_t"), MEMBERS)

    def test_requires_complete_direct_typed_constants(self) -> None:
        self.assertTrue(
            callable(getattr(contract, "check_constants", None)), "constant validator is missing"
        )
        source = bindings(MEMBERS, "TestCode")
        contract.check_constants(
            header=HEADER, source=source, native_type="amdsmi_test_t", go_type="TestCode"
        )
        invalid = [
            bindings(MEMBERS[:-1], "TestCode"),
            source.replace("TestCode", "uint32"),
            source.replace("C.AMDSMI_TEST_FIRST", "0"),
            source.replace("C.AMDSMI_TEST_ALIAS", "C.AMDSMI_TEST_FIRST"),
            source.replace("C.AMDSMI_TEST_LAST", "C.AMDSMI_TEST_LAST + 1"),
            source.replace("const", "var"),
            "/*\n" + source + "*/",
            "// " + source.replace("\n", "\n// "),
            "var example = `" + source + "`",
        ]
        for text in invalid:
            with self.subTest(source=text), self.assertRaisesRegex(ValueError, "constant binding"):
                contract.check_constants(
                    header=HEADER, source=text, native_type="amdsmi_test_t", go_type="TestCode"
                )


class ModuleTests(unittest.TestCase):
    def test_rejects_dependency_and_toolchain_directives(self) -> None:
        self.assertTrue(
            callable(getattr(contract, "check_module", None)), "module validator is missing"
        )
        contract.check_module(MODULE + "// require, replace, exclude, toolchain\n")
        for directive in (
            "require example.com/dependency v1.0.0",
            "require (\nexample.com/dependency v1.0.0\n)",
            "replace example.com/dependency => ../dependency",
            "exclude example.com/dependency v1.0.0",
            "toolchain go1.24.1",
        ):
            with self.subTest(directive=directive):
                with self.assertRaisesRegex(ValueError, "dependencies|toolchain"):
                    contract.check_module(MODULE + directive + "\n")

    def test_requires_expected_module_and_go_1_20(self) -> None:
        contract.check_module("// SPDX-License-Identifier: MIT\n" + MODULE)
        for source in (
            "",
            MODULE.replace("/amdsmi/go", "/other/go"),
            MODULE.replace("1.20", "1.19"),
            MODULE.replace("1.20", "1.24"),
            MODULE.replace("go 1.20", ""),
            MODULE + "go 1.20\n",
        ):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "go 1.20"):
                contract.check_module(source)


class NativeCallTests(unittest.TestCase):
    def test_rejects_every_call_outside_the_32_function_allowlist(self) -> None:
        self.assertTrue(
            callable(getattr(contract, "check_native_calls", None)),
            "native call validator is missing",
        )
        self.assertEqual(len(ALLOWED_CALLS), 32)
        contract.check_native_calls(ALLOWED_CALLS, available_only=True)
        for name in (
            "amdsmi_set_power_cap",
            "amdsmi_reset_gpu",
            "amdsmi_get_gpu_accelerator_partition_profile_config",
            "amdsmi_arbitrary_call",
        ):
            for available_only in (False, True):
                with self.subTest(name=name, available_only=available_only):
                    with self.assertRaisesRegex(ValueError, "prohibited native calls:.*" + name):
                        contract.check_native_calls(
                            ALLOWED_CALLS | {name}, available_only=available_only
                        )

    def test_final_requires_all_calls_but_available_only_accepts_subsets(self) -> None:
        contract.check_native_calls(ALLOWED_CALLS, available_only=False)
        contract.check_native_calls(set(), available_only=True)
        for name in sorted(ALLOWED_CALLS):
            calls = ALLOWED_CALLS - {name}
            with self.subTest(missing=name):
                contract.check_native_calls(calls, available_only=True)
                with self.assertRaisesRegex(ValueError, "missing native calls: " + name + "$"):
                    contract.check_native_calls(calls, available_only=False)

    def test_calls_include_preambles_but_not_comments_literals_or_casts(self) -> None:
        self.assertTrue(
            callable(getattr(contract, "native_calls", None)), "native call parser is missing"
        )
        source = """
// C.amdsmi_set_gpu_power_cap()
/* C.amdsmi_reset_gpu() */
package amdsmi
/*
#include <amd_smi/amdsmi.h>
static amdsmi_status_t go_amdsmi_bdf(amdsmi_processor_handle p) {
    // amdsmi_set_gpu_power_cap();
    const char *note = "amdsmi_reset_gpu()";
    return amdsmi_get_gpu_device_bdf(p, 0);
}
*/
import "C"
var example = `C.amdsmi_reset_gpu(); /* amdsmi_set_gpu_power_cap(); */ import "C"`
var op = "C.amdsmi_reset_gpu()"
type callback func(C.amdsmi_status_t)
func query() {
    C.amdsmi_status_code_to_string(C.amdsmi_status_t(code), &text)
    C.amdsmi_get_processor_type(C.amdsmi_processor_handle(p), &kind)
    C.amdsmi_socket_handle(p)
    C.amdsmi_temperature_metric_t(metric)
    C.amdsmi_gpu_block_t(block)
    C.go_amdsmi_bdf(p)
}
"""
        self.assertEqual(
            contract.native_calls(source),
            {
                "amdsmi_get_gpu_device_bdf",
                "amdsmi_status_code_to_string",
                "amdsmi_get_processor_type",
            },
        )


class ProjectTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="amdsmi-agent-contract-test-")
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name)
        self.package = self.project / "go" / "amdsmi"
        self.package.mkdir(parents=True)
        header_path = self.project / "include" / "amd_smi" / "amdsmi.h"
        header_path.parent.mkdir(parents=True)
        header = []
        for filename, enums in ENUM_OWNERS.items():
            source = []
            for native_type, go_type in enums:
                member = native_type.upper() + "_VALUE"
                header.append("typedef enum { " + member + " } " + native_type + ";")
                source.append(bindings([member], go_type))
            (self.package / filename).write_text("\n".join(source), encoding="utf-8")
        header_path.write_text("\n".join(header), encoding="utf-8")
        (self.project / "go" / "go.mod").write_text(MODULE, encoding="utf-8")
        self.calls_source = "\n".join("C." + name + "()" for name in sorted(ALLOWED_CALLS))
        (self.package / "native_linux.go").write_text(self.calls_source, encoding="utf-8")

    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess:
        script = self.project / "tests" / "go" / "test_api_contract.py"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(Path(contract.__file__).read_text(encoding="utf-8"), encoding="utf-8")
        return subprocess.run(
            [sys.executable, "-B", str(script), *arguments],
            cwd=self.project.parent,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"PATH": ""},
            check=False,
        )

    def test_cli_checks_project_without_go_or_cwd_dependency(self) -> None:
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--available-only", result.stdout)
        for arguments in ((), ("--available-only",)):
            with self.subTest(arguments=arguments):
                result = self.run_cli(*arguments)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Go API contract checks passed", result.stdout)

    def test_cli_reports_failures_without_tracebacks(self) -> None:
        owner = self.package / "amdsmi_interface.go"
        owner.unlink()
        result = self.run_cli("--available-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.run_cli()
        self.assertEqual(result.returncode, 1)
        self.assertIn("missing enum owner files: amdsmi_interface.go", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertNotIn("passed", result.stdout)
        (self.project / "go" / "go.mod").unlink()
        result = self.run_cli("--available-only")
        self.assertEqual(result.returncode, 1)
        self.assertIn("go.mod", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_checks_every_enum_in_its_owner_file(self) -> None:
        self.assertTrue(
            callable(getattr(contract, "check_project", None)),
            "project contract validator is missing",
        )
        contract.check_project(project=self.project, available_only=False)
        for filename, enums in ENUM_OWNERS.items():
            path = self.package / filename
            source = path.read_text(encoding="utf-8")
            for native_type, _ in enums:
                member = native_type.upper() + "_VALUE"
                path.write_text(source.replace("C." + member, "0"), encoding="utf-8")
                with self.subTest(owner=filename, native_type=native_type):
                    with self.assertRaisesRegex(ValueError, "constant binding: " + member):
                        contract.check_project(project=self.project, available_only=False)
                path.write_text(source, encoding="utf-8")

    def test_available_only_skips_missing_owners_but_final_requires_them(self) -> None:
        for filename in ENUM_OWNERS:
            path = self.package / filename
            source = path.read_text(encoding="utf-8")
            path.unlink()
            with self.subTest(owner=filename):
                try:
                    contract.check_project(project=self.project, available_only=True)
                except (FileNotFoundError, ValueError) as error:
                    self.fail("available-only rejected absent owner: " + str(error))
                with self.assertRaisesRegex(ValueError, "missing enum owner files:.*" + filename):
                    contract.check_project(project=self.project, available_only=False)
            path.write_text(source, encoding="utf-8")

    def test_project_checks_module_in_both_modes(self) -> None:
        (self.project / "go" / "go.mod").write_text(
            MODULE + "toolchain go1.24.1\n", encoding="utf-8"
        )
        for available_only in (False, True):
            with self.subTest(available_only=available_only):
                with self.assertRaisesRegex(ValueError, "toolchain"):
                    contract.check_project(project=self.project, available_only=available_only)

    def test_rejects_prohibited_calls_in_production_sources_and_preambles(self) -> None:
        forbidden = "amdsmi_arbitrary_call"
        templates = (
            'package amdsmi\nimport "C"\nfunc query() { C.%s() }\n',
            'package amdsmi\n/*\nstatic void helper(void) { %s(); }\n*/\nimport "C"\n',
        )
        for relative in ("amdsmi/query_linux.go", "examples/telemetry/main.go"):
            path = self.project / "go" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            for template in templates:
                path.write_text(template % forbidden, encoding="utf-8")
                for available_only in (False, True):
                    with self.subTest(path=relative, template=template, mode=available_only):
                        with self.assertRaisesRegex(
                            ValueError, "prohibited native calls:.*" + forbidden
                        ):
                            contract.check_project(
                                project=self.project, available_only=available_only
                            )
                path.unlink()

    def test_test_sources_do_not_count_toward_production_calls(self) -> None:
        (self.package / "native_linux.go").write_text("package amdsmi\n", encoding="utf-8")
        for relative in (
            "query_test.go",
            "mock_bridge_linux.go",
            "testdata/fixture.go",
            "testdata/mock_core.c",
        ):
            path = self.package / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.calls_source + "\nC.amdsmi_arbitrary_call()\n", encoding="utf-8")
        try:
            contract.check_project(project=self.project, available_only=True)
        except ValueError as error:
            self.fail("test sources affected production scope: " + str(error))
        with self.assertRaisesRegex(ValueError, "missing native calls:"):
            contract.check_project(project=self.project, available_only=False)


if __name__ == "__main__":
    unittest.main()
