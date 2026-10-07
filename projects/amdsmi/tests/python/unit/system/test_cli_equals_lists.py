# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI equals-form list parsing without device access."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from common.common import cli_search_order, find_cli_dir, generated_version_stub, load_cli_module


class TestCliEqualsLists(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cli_dir = find_cli_dir(*cli_search_order(str(Path(__file__).parent)))
        if cli_dir is None:
            raise unittest.SkipTest("amd-smi CLI not found")
        version_stub = {} if "_version" in sys.modules else generated_version_stub()
        with patch.dict("sys.modules", version_stub):
            cls.parser_module = load_cli_module(
                "amdsmi_parser_equals_test",
                str(Path(cli_dir) / "amdsmi_parser.py"),
                sys_path_dir=cli_dir,
            )
            cls.helpers_type = cls.parser_module.AMDSMIHelpers

    def setUp(self) -> None:
        helpers = SimpleNamespace(get_output_format=lambda: "human_readable")
        self.helpers_patch = patch.object(self.parser_module, "AMDSMIHelpers", return_value=helpers)
        self.helpers_patch.start()
        self.addCleanup(self.helpers_patch.stop)
        self.parser = self.parser_module.AMDSMISubParser(prog="amd-smi metric")
        self.parser.add_argument("-g", "--gpu", nargs="+")

    def test_equals_comma_list(self) -> None:
        self.assertEqual(self.parser.parse_args(["--gpu=0,1"]).gpu, ["0", "1"])

    def test_short_equals_comma_list(self) -> None:
        self.assertEqual(self.parser.parse_args(["-g=0,1"]).gpu, ["0", "1"])

    def test_space_lists_unchanged(self) -> None:
        for values in (["0", "1"], ["0,1"]):
            with self.subTest(values=values):
                self.assertEqual(self.parser.parse_args(["--gpu", *values]).gpu, values)

    def test_single_equals_value_unchanged(self) -> None:
        self.assertEqual(self.parser.parse_args(["-g=0"]).gpu, ["0"])
        with self.assertRaises(
            self.parser_module.amdsmi_cli_exceptions.AmdSmiInvalidParameterException
        ):
            self.parser.parse_args(["-g=0", "1"])

    def test_repeated_option_keeps_last_list(self) -> None:
        args = self.parser.parse_args(["-g=0,1", "-g=2,3"])
        self.assertEqual(args.gpu, ["2", "3"])

    def test_optional_list(self) -> None:
        self.parser.add_argument("--clock", nargs="*")
        self.assertEqual(self.parser.parse_args(["--clock=sclk,mclk"]).clock, ["sclk", "mclk"])
        self.assertEqual(self.parser.parse_args(["--clock"]).clock, [])

    def test_fixed_count_with_type_and_append(self) -> None:
        self.parser.add_argument("--pair", nargs=2, type=int, action="append")
        args = self.parser.parse_args(["--pair=0,1", "--pair", "2", "3"])
        self.assertEqual(args.pair, [[0, 1], [2, 3]])

    def test_equals_list_does_not_consume_following_value(self) -> None:
        for nargs in ("+", "*", 3):
            with self.subTest(nargs=nargs):
                parser = self.parser_module.AMDSMISubParser()
                parser.add_argument("--values", nargs=nargs)
                with self.assertRaises(self.parser_module.amdsmi_cli_exceptions.AmdSmiException):
                    parser.parse_args(["--values=0,1", "2"])

    def test_invalid_fixed_count(self) -> None:
        self.parser.add_argument("--triple", nargs=3)
        for value in ("0,1", "0,1,2,3"):
            with self.subTest(value=value):
                with self.assertRaises(
                    self.parser_module.amdsmi_cli_exceptions.AmdSmiInvalidParameterValueException
                ):
                    self.parser.parse_args([f"--triple={value}"])

    def test_fixed_counts_two_through_five(self) -> None:
        for count in range(2, 6):
            with self.subTest(count=count):
                parser = self.parser_module.AMDSMISubParser()
                parser.add_argument("--values", nargs=count, type=int)
                values = list(range(count))
                text = ",".join(str(value) for value in values)
                self.assertEqual(parser.parse_args([f"--values={text}"]).values, values)

    def test_type_and_choice_validation(self) -> None:
        self.parser.add_argument("--level", nargs="+", type=int, choices=[0, 1])
        self.assertEqual(self.parser.parse_args(["--level=0,1"]).level, [0, 1])
        for value in ("0,bad", "0,2"):
            with self.subTest(value=value):
                with self.assertRaises(self.parser_module.amdsmi_cli_exceptions.AmdSmiException):
                    self.parser.parse_args([f"--level={value}"])

    def test_scalar_commas_unchanged(self) -> None:
        for nargs in (None, "?", 1):
            with self.subTest(nargs=nargs):
                parser = self.parser_module.AMDSMISubParser()
                parser.add_argument("--file", nargs=nargs)
                value = "dir/report,a=b.json"
                expected = [value] if nargs == 1 else value
                self.assertEqual(parser.parse_args([f"--file={value}"]).file, expected)

    def test_empty_and_option_components_rejected(self) -> None:
        self.parser.add_argument("--json", action="store_true")
        for value in (",0", "0,", "0,,1", "0, ", "0,--json", "0,-g=1", "0,--", "0,--unknown"):
            with self.subTest(value=value):
                with self.assertRaises(
                    self.parser_module.amdsmi_cli_exceptions.AmdSmiInvalidParameterValueException
                ):
                    self.parser.parse_args([f"--gpu={value}"])

    def test_negative_numbers_use_existing_type_validation(self) -> None:
        self.parser.add_argument("--pair", nargs=2, type=int)
        self.assertEqual(self.parser.parse_args(["--pair=-1,-2"]).pair, [-1, -2])

    def test_whitespace_around_list_values_rejected(self) -> None:
        self.parser.add_argument("--pair", nargs=2, type=int)
        for option in ("--gpu", "--pair"):
            for value in ("0, 1", " 0,1", "0,1\t"):
                with self.subTest(option=option, value=value):
                    with self.assertRaises(
                        self.parser_module.amdsmi_cli_exceptions.AmdSmiInvalidParameterValueException
                    ):
                        self.parser.parse_args([f"{option}={value}"])

    def test_unknown_option_unchanged(self) -> None:
        args, unknown = self.parser.parse_known_args(["--unknown=0,1"])
        self.assertIsNone(args.gpu)
        self.assertEqual(unknown, ["--unknown=0,1"])

    def test_terminator_leaves_tokens_unchanged(self) -> None:
        self.parser.add_argument("rest", nargs="*")
        args = self.parser.parse_args(["--", "--gpu=0,1"])
        self.assertIsNone(args.gpu)
        self.assertEqual(args.rest, ["--gpu=0,1"])

    def test_terminator_after_list_does_not_expand_selection(self) -> None:
        self.parser.add_argument("rest", nargs="*")
        args = self.parser.parse_args(["--gpu=0,1", "--", "all"])
        self.assertEqual(args.gpu, ["0", "1"])
        self.assertEqual(args.rest, ["all"])

    def test_abbreviated_equals_list(self) -> None:
        self.assertEqual(self.parser.parse_args(["--gp=0,1"]).gpu, ["0", "1"])

    def test_no_equals_attached_value_unchanged(self) -> None:
        self.assertEqual(self.parser.parse_args(["-g0,1"]).gpu, ["0,1"])

    def test_disabled_and_ambiguous_abbreviations(self) -> None:
        self.parser.allow_abbrev = False
        args, unknown = self.parser.parse_known_args(["--gp=0,1"])
        self.assertIsNone(args.gpu)
        self.assertEqual(unknown, ["--gp=0,1"])
        self.parser.allow_abbrev = True
        self.parser.add_argument("--gpu-mode")
        with self.assertRaises(
            self.parser_module.amdsmi_cli_exceptions.AmdSmiInvalidParameterException
        ):
            self.parser.parse_args(["--gp=0,1"])

    def test_input_argv_not_modified(self) -> None:
        argv = ["--gpu=0,1"]
        self.parser.parse_args(argv)
        self.assertEqual(argv, ["--gpu=0,1"])
        with patch.object(sys, "argv", ["amd-smi", *argv]):
            self.assertEqual(self.parser.parse_args().gpu, ["0", "1"])
            self.assertEqual(sys.argv, ["amd-smi", *argv])

    def _command_parser(self, command: str):
        helpers_type = self.helpers_type
        helpers = Mock(spec=helpers_type)
        for method in (
            "is_amdgpu_initialized",
            "is_amd_hsmp_initialized",
            "is_ainic_initialized",
            "is_brcm_nic_initialized",
            "is_brcm_switch_initialized",
            "is_linux",
            "is_baremetal",
        ):
            getattr(helpers, method).return_value = True
        helpers.is_hypervisor.return_value = False
        helpers.is_virtual_os.return_value = False
        helpers.get_output_format.return_value = "human_readable"
        choices = {
            str(index): {
                "Device Handle": f"handle-{index}",
                "bdf": f"0000:0{index}:00.0",
                "UUID": f"00000000-0000-0000-0000-00000000000{index}",
                "CUID": f"cuid-{index}",
            }
            for index in range(2)
        }
        for device in ("gpu", "cpu", "core", "nic", "switch"):
            getattr(helpers, f"get_{device}_choices").return_value = (dict(choices), "0, 1")
            name = f"get_device_handles_from_{device}_selections"
            getattr(helpers, name).side_effect = getattr(helpers_type, name).__get__(helpers)
        helpers.is_UUID.side_effect = helpers_type.is_UUID
        callbacks = {
            name: Mock()
            for name in (
                "version",
                "list",
                "static",
                "firmware",
                "bad_pages",
                "metric",
                "process",
                "profile",
                "event",
                "topology",
                "set_value",
                "reset",
                "monitor",
                "xgmi",
                "partition",
                "ras",
                "node",
                "fabric",
                "_rocm_smi",
                "default",
            )
        }
        return self.parser_module.AMDSMIParser(
            **callbacks, helpers=helpers, sys_argv=["amd-smi", command]
        )

    def test_registered_device_options_resolve_both_handles(self) -> None:
        parser = self._command_parser("metric")
        for option, dest in (
            ("-g", "gpu"),
            ("--gpu", "gpu"),
            ("-U", "cpu"),
            ("--cpu", "cpu"),
            ("-O", "core"),
            ("--core", "core"),
            ("--switch", "switch"),
        ):
            with self.subTest(option=option):
                args = parser.parse_args(["metric", f"{option}=0,1"])
                self.assertEqual(getattr(args, dest), ["handle-0", "handle-1"])

    def test_registered_nic_bdfs(self) -> None:
        parser = self._command_parser("metric")
        for option in ("-N", "--nic"):
            with self.subTest(option=option):
                args = parser.parse_args(["metric", f"{option}=0000:00:00.0,0000:01:00.0"])
                self.assertEqual(args.nic, ["handle-0", "handle-1"])

    def test_registered_gpu_identifiers(self) -> None:
        parser = self._command_parser("metric")
        for value in (
            "0000:00:00.0,0000:01:00.0",
            "00000000-0000-0000-0000-000000000000,00000000-0000-0000-0000-000000000001",
        ):
            with self.subTest(value=value):
                self.assertEqual(
                    parser.parse_args(["metric", f"--gpu={value}"]).gpu, ["handle-0", "handle-1"]
                )

    def test_registered_device_conflict_still_rejected(self) -> None:
        parser = self._command_parser("metric")
        with self.assertRaises(
            self.parser_module.amdsmi_cli_exceptions.AmdSmiInvalidParameterException
        ):
            parser.parse_args(["metric", "--gpu=0,1", "--cpu=0,1"])

    def test_registered_gpu_space_comma_still_rejected(self) -> None:
        parser = self._command_parser("metric")
        with self.assertRaises(
            self.parser_module.amdsmi_cli_exceptions.AmdSmiInvalidParameterValueException
        ):
            parser.parse_args(["metric", "--gpu", "0,1"])

    def test_registered_gpu_all(self) -> None:
        parser = self._command_parser("metric")
        with patch.object(
            self.parser_module.amdsmi_interface,
            "amdsmi_get_processor_handles",
            return_value=["handle-0", "handle-1", "handle-2"],
        ):
            self.assertEqual(
                parser.parse_args(["metric", "--gpu=0,all"]).gpu,
                ["handle-0", "handle-1", "handle-2"],
            )
            with self.assertRaises(
                self.parser_module.amdsmi_cli_exceptions.AmdSmiInvalidParameterException
            ):
                parser.parse_args(["metric", "--gpu=0,1", "all"])

    def test_registered_vf_choices(self) -> None:
        parser = self._command_parser("metric")
        parser.helpers.is_hypervisor.return_value = True
        vf_parser = self.parser_module.AMDSMISubParser()
        parser._add_device_arguments(vf_parser)
        self.assertEqual(vf_parser.parse_args(["--vf=1,2"]).vf, ["1", "2"])
        with self.assertRaises(
            self.parser_module.amdsmi_cli_exceptions.AmdSmiInvalidParameterValueException
        ):
            vf_parser.parse_args(["--vf=1,4"])

    def test_registered_scalar_ptl_format(self) -> None:
        parser = self._command_parser("metric")
        ptl_parser = self.parser_module.AMDSMISubParser()
        ptl_parser.add_argument("--ptl-format", action=parser._validate_ptl_format())
        self.assertEqual(
            ptl_parser.parse_args(["--ptl-format=I8,F32"]).ptl_format,
            ptl_parser.parse_args(["--ptl-format", "I8,F32"]).ptl_format,
        )

    def test_registered_cpu_fixed_count(self) -> None:
        parser = self._command_parser("metric")
        args = parser.parse_args(["metric", "--cpu=0,1", "--cpu-io-bandwidth=1,P2"])
        self.assertEqual(args.cpu_io_bandwidth, [["1", "P2"]])

    def test_registered_watch_forms(self) -> None:
        for command in ("metric", "monitor", "process"):
            parser = self._command_parser(command)
            for flags in (
                ("-w=1", "-W=2", "-i=2"),
                ("--watch=1", "--watch_time=2", "--iterations=2"),
                ("-w", "1", "-W", "2", "-i", "2"),
            ):
                with self.subTest(command=command, flags=flags):
                    args = parser.parse_args([command, *flags])
                    self.assertEqual((args.watch, args.watch_time, args.iterations), (1, 2, 2))

    def test_registered_help_documents_equals_lists(self) -> None:
        for command in ("metric", "monitor", "process", "firmware"):
            with self.subTest(command=command):
                parser = self._command_parser(command)
                help_text = parser.subparsers.choices[command].format_help()
                self.assertIn("--option=VALUE1,VALUE2", help_text)
                self.assertIn("--option VALUE1 VALUE2", help_text)
