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
ENUM_TYPES = {
    "amdsmi_status_t": "Status",
    "amdsmi_vram_type_t": "VramType",
    "amdsmi_fw_block_t": "FwBlock",
    "amdsmi_temperature_type_t": "TemperatureType",
    "amdsmi_temperature_metric_t": "TemperatureMetric",
    "amdsmi_clk_type_t": "ClkType",
    "amdsmi_memory_type_t": "MemoryType",
    "amdsmi_memory_partition_type_t": "MemoryPartitionType",
    "amdsmi_accelerator_partition_type_t": "AcceleratorPartitionType",
    "amdsmi_gpu_block_t": "GpuBlock",
    "amdsmi_ras_err_state_t": "RASState",
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
        source = []
        for native_type, go_type in ENUM_TYPES.items():
            member = native_type.upper() + "_VALUE"
            header.append("typedef enum { " + member + " } " + native_type + ";")
            source.append(bindings([member], go_type))
        self.constants = self.package / "constants.go"
        self.constants.write_text("\n".join(source), encoding="utf-8")
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
        self.constants.unlink()
        result = self.run_cli("--available-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.run_cli()
        self.assertEqual(result.returncode, 1)
        self.assertIn("missing direct constant binding: AMDSMI_STATUS_T_VALUE", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertNotIn("passed", result.stdout)
        (self.project / "go" / "go.mod").unlink()
        result = self.run_cli("--available-only")
        self.assertEqual(result.returncode, 1)
        self.assertIn("go.mod", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_checks_every_required_enum(self) -> None:
        self.assertTrue(
            callable(getattr(contract, "check_project", None)),
            "project contract validator is missing",
        )
        contract.check_project(project=self.project, available_only=False)
        source = self.constants.read_text(encoding="utf-8")
        for native_type in ENUM_TYPES:
            member = native_type.upper() + "_VALUE"
            self.constants.write_text(source.replace("C." + member, "0"), encoding="utf-8")
            for available_only in (False, True):
                with self.subTest(native_type=native_type, available_only=available_only):
                    with self.assertRaisesRegex(ValueError, "constant binding: " + member):
                        contract.check_project(project=self.project, available_only=available_only)
            self.constants.write_text(source, encoding="utf-8")

    def test_enum_declarations_can_move_between_package_files(self) -> None:
        source = self.constants.read_text(encoding="utf-8")
        self.constants.unlink()
        blocks = source.split("\n\n")
        for index, block in enumerate(blocks):
            (self.package / ("constants_" + str(index) + ".go")).write_text(block, encoding="utf-8")
        for available_only in (False, True):
            with self.subTest(available_only=available_only):
                contract.check_project(project=self.project, available_only=available_only)

    def test_new_partial_enum_is_rejected_in_each_mode(self) -> None:
        header = self.project / "include" / "amd_smi" / "amdsmi.h"
        header.write_text(header.read_text(encoding="utf-8") + HEADER, encoding="utf-8")
        (self.package / "extra_linux.go").write_text(
            bindings(MEMBERS[:-1], "TestCode"), encoding="utf-8"
        )
        for available_only in (False, True):
            with self.subTest(available_only=available_only):
                with self.assertRaisesRegex(ValueError, "constant binding: AMDSMI_TEST_LAST"):
                    contract.check_project(project=self.project, available_only=available_only)

    def test_new_enum_requires_direct_constants_not_variables(self) -> None:
        header = self.project / "include" / "amd_smi" / "amdsmi.h"
        header.write_text(header.read_text(encoding="utf-8") + HEADER, encoding="utf-8")
        (self.package / "extra.go").write_text(
            bindings(MEMBERS, "TestCode").replace("const", "var"), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "constant binding: AMDSMI_TEST_FIRST"):
            contract.check_project(project=self.project, available_only=False)

    def test_new_enum_rejects_missing_wrong_type_and_indirect_bindings(self) -> None:
        header = self.project / "include" / "amd_smi" / "amdsmi.h"
        header.write_text(header.read_text(encoding="utf-8") + HEADER, encoding="utf-8")
        source = bindings(MEMBERS, "TestCode")
        path = self.package / "extra.go"
        path.write_text(source, encoding="utf-8")
        contract.check_project(project=self.project, available_only=False)
        for invalid in (
            bindings(MEMBERS[1:], "TestCode"),
            source.replace("AMDSMI_TEST_LAST TestCode", "AMDSMI_TEST_LAST uint32"),
            source.replace("C.AMDSMI_TEST_FIRST", "0"),
            source.replace("C.AMDSMI_TEST_ALIAS", "C.AMDSMI_TEST_FIRST"),
            source.replace("C.AMDSMI_TEST_LAST", "C.AMDSMI_TEST_LAST + 1"),
            source.replace("C.AMDSMI_TEST_FIRST", "TestCode(C.AMDSMI_TEST_FIRST)"),
        ):
            path.write_text(invalid, encoding="utf-8")
            for available_only in (False, True):
                with self.subTest(source=invalid, available_only=available_only):
                    with self.assertRaisesRegex(ValueError, "constant binding"):
                        contract.check_project(project=self.project, available_only=available_only)

    def test_copied_constants_are_checked_after_original_declarations_removed(self) -> None:
        copied = self.package / "copied.go"
        source = self.constants.read_text(encoding="utf-8")
        copied.write_text(source, encoding="utf-8")
        self.constants.write_text("package amdsmi\n", encoding="utf-8")
        contract.check_project(project=self.project, available_only=False)
        for invalid in (
            source.replace("AMDSMI_STATUS_T_VALUE Status = C.AMDSMI_STATUS_T_VALUE", ""),
            source.replace("AMDSMI_STATUS_T_VALUE Status", "AMDSMI_STATUS_T_VALUE uint32"),
            source.replace("C.AMDSMI_STATUS_T_VALUE", "0"),
        ):
            copied.write_text(invalid, encoding="utf-8")
            with self.subTest(source=invalid):
                with self.assertRaisesRegex(ValueError, "constant binding: AMDSMI_STATUS_T_VALUE"):
                    contract.check_project(project=self.project, available_only=False)

    def test_test_only_constants_cannot_supply_required_enum(self) -> None:
        source = self.constants.read_text(encoding="utf-8")
        self.constants.unlink()
        for relative in ("constants_test.go", "mock_bridge_linux.go", "testdata/constants.go"):
            path = self.package / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "constant binding: AMDSMI_STATUS_T_VALUE"):
            contract.check_project(project=self.project, available_only=False)

    def test_unmirrored_and_test_only_new_enums_are_ignored(self) -> None:
        header = self.project / "include" / "amd_smi" / "amdsmi.h"
        header.write_text(header.read_text(encoding="utf-8") + HEADER, encoding="utf-8")
        source = bindings(MEMBERS[:1], "TestCode")
        for relative in (
            "go/amdsmi/constants_test.go",
            "go/amdsmi/mock_bridge_linux.go",
            "go/amdsmi/testdata/constants.go",
            "go/examples/constants.go",
        ):
            path = self.project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
        (self.package / "notes.go").write_text(
            "/*\n" + source + "*/\nvar note = `" + source + "`\n", encoding="utf-8"
        )
        for available_only in (False, True):
            contract.check_project(project=self.project, available_only=available_only)

    def test_only_named_init_enum_allows_gpu_subset(self) -> None:
        header = self.project / "include" / "amd_smi" / "amdsmi.h"
        original = header.read_text(encoding="utf-8")
        init_header = (
            "typedef enum { AMDSMI_INIT_AMD_GPUS, AMDSMI_INIT_AMD_CPUS } amdsmi_init_flags_t;"
        )
        header.write_text(original + init_header, encoding="utf-8")
        path = self.package / "init.go"
        path.write_text(bindings(["AMDSMI_INIT_AMD_GPUS"], "InitFlags"), encoding="utf-8")
        contract.check_project(project=self.project, available_only=False)
        for invalid in (
            bindings(["AMDSMI_INIT_AMD_GPUS", "AMDSMI_INIT_AMD_CPUS"], "InitFlags"),
            bindings(["AMDSMI_INIT_AMD_GPUS"], "uint64"),
            bindings(["AMDSMI_INIT_AMD_GPUS"], "InitFlags").replace("C.AMDSMI_INIT_AMD_GPUS", "2"),
        ):
            path.write_text(invalid, encoding="utf-8")
            with self.subTest(source=invalid), self.assertRaises(ValueError):
                contract.check_project(project=self.project, available_only=False)
        path.write_text(bindings(["AMDSMI_INIT_AMD_GPUS"], "InitFlags"), encoding="utf-8")
        header.write_text(
            original + init_header.replace("amdsmi_init_flags_t", "amdsmi_other_flags_t"),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "constant binding: AMDSMI_INIT_AMD_CPUS"):
            contract.check_project(project=self.project, available_only=False)

    def test_available_only_skips_missing_enums_but_final_requires_them(self) -> None:
        source = self.constants.read_text(encoding="utf-8")
        for native_type, go_type in ENUM_TYPES.items():
            member = native_type.upper() + "_VALUE"
            self.constants.write_text(
                source.replace(bindings([member], go_type), ""), encoding="utf-8"
            )
            with self.subTest(native_type=native_type):
                contract.check_project(project=self.project, available_only=True)
                with self.assertRaisesRegex(
                    ValueError, "missing direct constant binding: " + member
                ):
                    contract.check_project(project=self.project, available_only=False)
            self.constants.write_text(source, encoding="utf-8")

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
