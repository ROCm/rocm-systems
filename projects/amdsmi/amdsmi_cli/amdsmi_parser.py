#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import errno
import os
import sys
import time
import collections
from amdsmi import amdsmi_interface
from typing import Optional
from typing import Union

from pathlib import Path

from _version import __version__
from amdsmi_helpers import AMDSMIHelpers
import amdsmi_cli_exceptions

try:
    import amdsmi_i18n  # noqa: F401
except ImportError:  # loaded by file path (e.g. unit tests): make the CLI dir importable
    import os as _os
    import sys as _sys

    _sys.path.append(_os.path.dirname(_os.path.abspath(__file__)))
from amdsmi_i18n import gettext as _T, lazy as _L


# Valid value ranges for CPU `set` arguments, defined by the HSMP protocol and
# not queryable from the library. Update these if the supported range changes.
CPU_XGMI_LINK_WIDTH_RANGE = range(0, 2)
CPU_GMI3_LINK_WIDTH_RANGE = range(0, 3)
CPU_LCLK_DPM_LEVEL_RANGE = range(0, 4)
CPU_DISABLE_APB_RANGE = range(0, 4)


def _cpu_set_range_label(value_range):
    # Render a range like range(0, 2) as "0-1" for help text.
    return f"{value_range.start}-{value_range[-1]}"


# Custom Help Formatter for increasing the action max length
class AMDSMIParserHelpFormatter(argparse.HelpFormatter):
    def __init__(self, prog):
        super().__init__(prog=prog, indent_increment=2, max_help_position=24, width=90)
        self._action_max_length = 20


# Custom Help Formatter for not duplicating the metavar in the subparsers
class AMDSMISubparserHelpFormatter(argparse.RawTextHelpFormatter):
    def __init__(self, prog):
        super().__init__(prog, indent_increment=2, max_help_position=80, width=90)
        self._action_max_length = 20

    def _format_action_invocation(self, action):
        if not action.option_strings or action.nargs == 0:
            return super()._format_action_invocation(action)
        default = self._get_default_metavar_for_optional(action)
        args_string = self._format_args(action, default)
        return ", ".join(action.option_strings) + " " + args_string


class AMDSMISubParser(argparse.ArgumentParser):
    """Subcommand parser with custom error handling for AMDSMI CLI.
    Used as parser_class for subcommands so that argparse errors on
    subcommand arguments (e.g., --loglevel INVALID) are routed through
    the same AmdSmiException framework instead of calling sys.exit(2).
    """

    def error(self, message):
        helpers = AMDSMIHelpers()
        outputformat = helpers.get_output_format()
        command = sys.argv[1] if len(sys.argv) > 1 else ""

        if ": invalid choice: " in message:
            # e.g. "argument --loglevel: invalid choice: 'DDEBUG' (choose from ...)"
            # or   "argument --process-isolation: invalid choice: 2 (choose from 0, 1)"
            _after = message.split(": invalid choice: ")[1]
            value = _after.split("'")[1] if _after.startswith("'") else _after.split(" ")[0]
            hint = _after[_after.find(" (") :].rstrip() if " (" in _after else None
            raise amdsmi_cli_exceptions.AmdSmiInvalidParameterValueException(
                command, value, outputformat, hint=hint
            )
        elif ": expected" in message:
            # e.g. "argument --gpu: expected one argument"
            arg = (
                message[len("argument ") :].split(":")[0]
                if message.startswith("argument ")
                else message
            )
            raise amdsmi_cli_exceptions.AmdSmiMissingParameterValueException(arg, outputformat)
        else:
            raise amdsmi_cli_exceptions.AmdSmiInvalidParameterException(
                command, message, outputformat
            )


class AMDSMIParser(argparse.ArgumentParser):
    """Unified Parser for AMDSMI CLI.
        This parser doesn't access amdsmi's lib directly,but via AMDSMIHelpers,
        this allows for us to use this parser with future OS & Platform integration.

    Args:
        argparse (ArgumentParser): argparse.ArgumentParser
    """

    def __init__(
        self,
        version,
        list,
        static,
        firmware,
        bad_pages,
        metric,
        process,
        profile,
        event,
        topology,
        set_value,
        reset,
        monitor,
        xgmi,
        partition,
        ras,
        node,
        fabric,
        _rocm_smi,
        default,
        sys_argv=None,
        helpers=None,
    ):

        # Helper variables
        if helpers is None:
            # If helpers is not provided, create a new instance
            self.helpers = AMDSMIHelpers()
        else:
            self.helpers = helpers

        # Get choices based on driver initialized
        if self.helpers.is_amdgpu_initialized():
            self.gpu_choices, self.gpu_choices_str = self.helpers.get_gpu_choices()
        else:
            self.gpu_choices = {}
            self.gpu_choices_str = ""

        if self.helpers.is_brcm_switch_initialized():
            self.switch_choices, self.switch_choices_str = self.helpers.get_switch_choices()
        else:
            self.switch_choices = {}
            self.switch_choices_str = ""

        if self.helpers.is_ainic_initialized() or self.helpers.is_brcm_nic_initialized():
            self.nic_choices, self.nic_choices_str = self.helpers.get_nic_choices()
        else:
            self.nic_choices = {}
            self.nic_choices_str = ""

        if self.helpers.is_amd_hsmp_initialized():
            self.cpu_choices, self.cpu_choices_str = self.helpers.get_cpu_choices()
            self.core_choices, self.core_choices_str = self.helpers.get_core_choices()
        else:
            self.cpu_choices = {}
            self.cpu_choices_str = ""
            self.core_choices = {}
            self.core_choices_str = ""

        self.vf_choices = ["3", "2", "1"]

        self.version_string = f"Version: {__version__}"
        self.platform_string = f"Platform: {self.helpers.os_info()}"
        self.rocm_version = self.helpers.get_rocm_version()
        self.rocm_version_string = f"ROCm version: {self.rocm_version}"
        self.program_name = "amd-smi"
        self.description = _T(
            "AMD System Management Interface | {version_string} | {rocm_version_string} | {platform_string}"
        ).format(
            version_string=self.version_string,
            rocm_version_string=self.rocm_version_string,
            platform_string=self.platform_string,
        )

        # Adjust argument parser options
        super().__init__(
            formatter_class=lambda prog: AMDSMIParserHelpFormatter(prog),
            description=self.description,
            epilog=_T("For detailed help on specific commands: amd-smi [command] -h"),
            add_help=True,
            prog=self.program_name,
        )

        # Add top-level --rocm-smi flag
        self.add_argument(
            "--rocm-smi",
            action="store_true",
            help=_T("Display GPU information in ROCm-SMI compatible format"),
        )

        # Add top-level command modifiers (for --rocm-smi and other top-level flags)
        loglevel_choices = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        loglevel_help = _T("Set the logging level from the possible choices: {join}").format(
            join=", ".join(loglevel_choices)
        )
        self.add_argument(
            "--loglevel",
            action="store",
            type=str.upper,
            required=False,
            help=loglevel_help,
            default="ERROR",
            metavar="LEVEL",
            choices=loglevel_choices,
        )

        # Setup subparsers
        self.subparsers = self.add_subparsers(
            title=_T("AMD-SMI Commands"),
            parser_class=AMDSMISubParser,
            help=_T("Descriptions:"),
            metavar="",
        )

        # Store possible subcommands & aliases for later errors
        self.possible_commands = [
            "version",
            "list",
            "static",
            "firmware",
            "ucode",
            "bad-pages",
            "metric",
            "process",
            "profile",
            "event",
            "topology",
            "set",
            "reset",
            "monitor",
            "dmon",
            "xgmi",
            "partition",
            "ras",
            "node",
            "fabric",
            "default",
        ]

        # Add all subparsers
        if sys_argv is not None:
            if os.getenv("_ARGCOMPLETE") or any(arg in sys_argv for arg in ["--help", "-h"]):
                self._add_version_parser(self.subparsers, version)
                self._add_list_parser(self.subparsers, list)
                self._add_static_parser(self.subparsers, static)
                self._add_firmware_parser(self.subparsers, firmware)
                self._add_bad_pages_parser(self.subparsers, bad_pages)
                self._add_metric_parser(self.subparsers, metric)
                self._add_process_parser(self.subparsers, process)
                self._add_profile_parser(self.subparsers, profile)
                self._add_event_parser(self.subparsers, event)
                self._add_topology_parser(self.subparsers, topology)
                self._add_set_value_parser(self.subparsers, set_value)
                self._add_reset_parser(self.subparsers, reset)
                self._add_monitor_parser(self.subparsers, monitor)
                self._add_xgmi_parser(self.subparsers, xgmi)
                self._add_partition_parser(self.subparsers, partition)
                self._add_ras_parser(self.subparsers, ras)
                self._add_node_parser(self.subparsers, node)
                self._add_fabric_parser(self.subparsers, fabric)
            elif any(arg in sys_argv for arg in ["version"]):
                self._add_version_parser(self.subparsers, version)
            elif any(arg in sys_argv for arg in ["list"]):
                self._add_list_parser(self.subparsers, list)
            elif any(arg in sys_argv for arg in ["static"]):
                self._add_static_parser(self.subparsers, static)
            elif any(arg in sys_argv for arg in ["firmware", "ucode"]):
                self._add_firmware_parser(self.subparsers, firmware)
            elif any(arg in sys_argv for arg in ["bad-pages"]):
                self._add_bad_pages_parser(self.subparsers, bad_pages)
            elif any(arg in sys_argv for arg in ["metric"]):
                self._add_metric_parser(self.subparsers, metric)
            elif any(arg in sys_argv for arg in ["process"]):
                self._add_process_parser(self.subparsers, process)
            elif any(arg in sys_argv for arg in ["profile"]):
                self._add_profile_parser(self.subparsers, profile)
            elif any(arg in sys_argv for arg in ["event"]):
                self._add_event_parser(self.subparsers, event)
            elif any(arg in sys_argv for arg in ["topology"]):
                self._add_topology_parser(self.subparsers, topology)
            elif any(arg in sys_argv for arg in ["set"]):
                self._add_set_value_parser(self.subparsers, set_value)
            elif any(arg in sys_argv for arg in ["reset"]):
                self._add_reset_parser(self.subparsers, reset)
            elif any(arg in sys_argv for arg in ["monitor", "dmon"]):
                self._add_monitor_parser(self.subparsers, monitor)
            elif any(arg in sys_argv for arg in ["xgmi"]):
                self._add_xgmi_parser(self.subparsers, xgmi)
            elif any(arg in sys_argv for arg in ["partition"]):
                self._add_partition_parser(self.subparsers, partition)
            elif any(arg in sys_argv for arg in ["ras"]):
                self._add_ras_parser(self.subparsers, ras)
            elif any(arg in sys_argv for arg in ["node"]):
                self._add_node_parser(self.subparsers, node)
            elif any(arg in sys_argv for arg in ["fabric"]):
                self._add_fabric_parser(self.subparsers, fabric)
            else:
                # If no subcommand is given, add the default parser
                self._add_default_parser(self.subparsers, default)

    def _not_negative_int(self, int_value, sub_arg=None):
        # Argument type validator
        if int_value.isdigit():  # Is digit doesn't work on negative numbers
            return int(int_value)

        outputformat = self.helpers.get_output_format()
        if int_value == "":
            raise amdsmi_cli_exceptions.AmdSmiMissingParameterValueException(sub_arg, outputformat)
        else:
            raise amdsmi_cli_exceptions.AmdSmiInvalidParameterValueException(
                sys.argv[1], int_value, outputformat
            )

    def _positive_int(self, int_value, sub_arg=None):
        # Argument type validator
        if int_value.isdigit():  # Is digit doesn't work on negative numbers
            if int(int_value) > 0:
                return int(int_value)

        outputformat = self.helpers.get_output_format()
        if int_value == "":
            raise amdsmi_cli_exceptions.AmdSmiMissingParameterValueException(sub_arg, outputformat)
        else:
            raise amdsmi_cli_exceptions.AmdSmiInvalidParameterValueException(
                sys.argv[1], int_value, outputformat
            )

    def _positive_float(self, float_value, sub_arg=None):
        # Argument type validator for positive float values
        try:
            value = float(float_value)
            if value > 0:
                return value
        except ValueError:
            # Non-numeric values are handled via custom CLI exceptions below
            pass

        outputformat = self.helpers.get_output_format()
        if float_value == "":
            raise amdsmi_cli_exceptions.AmdSmiMissingParameterValueException(sub_arg, outputformat)
        else:
            raise amdsmi_cli_exceptions.AmdSmiInvalidParameterValueException(
                sys.argv[1], float_value, outputformat
            )

    def _is_valid_string(self, string_value, sub_arg=None):
        # Argument type validator
        # This is for triggering a cli exception if an empty string is detected
        if string_value:
            return string_value

        outputformat = self.helpers.get_output_format()
        if string_value == "":
            raise amdsmi_cli_exceptions.AmdSmiMissingParameterValueException(sub_arg, outputformat)
        else:
            raise amdsmi_cli_exceptions.AmdSmiInvalidParameterValueException(
                sys.argv[1], string_value, outputformat
            )

    def _valid_ampp_profile_name(self, profile_name):
        # Argument type validator. Same constraints as an AMPP field name:
        # amdsmi_activate_ampp_profile() encodes the name to UTF-8, which
        # raises for a lone surrogate (reachable through argv's
        # surrogateescape) unless it is rejected here.
        if AMDSMIHelpers.is_valid_ampp_field_name(profile_name):
            return profile_name

        raise amdsmi_cli_exceptions.AmdSmiInvalidParameterValueException(
            sys.argv[1],
            ascii(profile_name),
            self.helpers.get_output_format(),
            hint=_L(
                "Profile name cannot be empty or exceed {AMDSMI_MAX_STRING_LENGTH} bytes.",
                AMDSMI_MAX_STRING_LENGTH=amdsmi_interface.AMDSMI_MAX_STRING_LENGTH,
            ),
        )

    def _is_command_supported(self, user_input, acceptable_values, hint=None):
        if str(user_input).upper() not in acceptable_values:
            raise amdsmi_cli_exceptions.AmdSmiInvalidParameterValueException(
                sys.argv[1], str(user_input).upper(), self.helpers.get_output_format(), hint=hint
            )
        else:
            return str(user_input).upper()

    def _limit_select(self):
        """Custom action for setting clock limits"""
        output_format = self.helpers.get_output_format()

        class AMDSMILimitArgs(argparse.Action):
            def __call__(
                self,
                parser: AMDSMIParser,
                namespace: argparse.Namespace,
                values: Union[str, list, None],
                option_string: Optional[str] = None,
            ) -> None:
                # valid values
                valid_clk_types = ("sclk", "mclk", "fclk")
                valid_lim_types = ("min", "max")
                clk_type, lim_type, val = values

                # Check if the sclk and mclk parameters are valid
                if clk_type not in valid_clk_types:
                    raise amdsmi_cli_exceptions.AmdSmiInvalidParameterException(
                        sys.argv[1],
                        clk_type,
                        output_format,
                        hint=_L("Valid options are: {join}.", join=", ".join(valid_clk_types)),
                    )
                if lim_type not in valid_lim_types:
                    raise amdsmi_cli_exceptions.AmdSmiInvalidParameterException(
                        sys.argv[1],
                        lim_type,
                        output_format,
                        hint=_L("Valid options are: {join}.", join=", ".join(valid_lim_types)),
                    )

                # Check if the val is a valid integer value
                if not val.isdigit():
                    raise amdsmi_cli_exceptions.AmdSmiInvalidParameterValueException(
                        sys.argv[1], val, output_format
                    )
                val = int(val)
                if val < 0:
                    raise amdsmi_cli_exceptions.AmdSmiInvalidParameterValueException(
                        sys.argv[1], val, output_format
                    )
                clk_limit_args = collections.namedtuple(
                    "clk_limit_args", ["clk_type", "lim_type", "val"]
                )
                setattr(namespace, self.dest, clk_limit_args(clk_type, lim_type, val))

        return AMDSMILimitArgs

    def _level_select(self):
        """Custom action for setting clock frequencies to particular performance level"""
        output_format = self.helpers.get_output_format()

        class AMDSMIFreqArgs(argparse.Action):
            def __call__(
                self,
                parser: AMDSMIParser,
                namespace: argparse.Namespace,
                values: list,
                option_string: Optional[str] = None,
            ) -> None:
                # valid values
                valid_clk_types = ("sclk", "mclk", "pcie", "fclk", "socclk")
                clk_type = values[0]
                perf_levels_str = values[1:]

                # Check if the sclk and mclk parameters are valid
                if clk_type not in valid_clk_types:
                    raise amdsmi_cli_exceptions.AmdSmiInvalidParameterException(
                        sys.argv[1],
                        clk_type,
                        output_format,
                        hint=_L("Valid options are: {join}.", join=", ".join(valid_clk_types)),
                    )

                if not perf_levels_str:
                    raise amdsmi_cli_exceptions.AmdSmiInvalidParameterValueException(
                        sys.argv[1],
                        clk_type,
                        output_format,
                        hint=_L(
                            "\n--clk-level requires at least one performance level index (e.g. --clk-level {clk_type} 0 1)",
                            clk_type=clk_type,
                        ),
                    )

                perf_levels = []
                # Check if every item in perf level is valid
                for level in perf_levels_str:
                    if not level.isdigit():
                        raise amdsmi_cli_exceptions.AmdSmiInvalidParameterValueException(
                            sys.argv[1], level, output_format
                        )
                    level = int(level)
                    if level < 0:
                        raise amdsmi_cli_exceptions.AmdSmiInvalidParameterValueException(
                            sys.argv[1], level, output_format
                        )
                    perf_levels.append(level)

                clk_level_args = collections.namedtuple(
                    "clk_level_args", ["clk_type", "perf_levels"]
                )
                setattr(namespace, self.dest, clk_level_args(clk_type, perf_levels))

        return AMDSMIFreqArgs

    def _power_cap_options(self):
        """Custom action for setting power cap options"""
        output_format = self.helpers.get_output_format()

        class AMDSMIPowerCapArgs(argparse.Action):
            def __call__(
                self,
                parser: AMDSMIParser,
                namespace: argparse.Namespace,
                values: list,
                option_string: Optional[str] = None,
            ) -> None:
                if len(values) == 1:
                    # Only wattage provided - set all available power cap types
                    power_cap_value = values[0]
                    power_cap_type = None  # None means all available sensors
                elif len(values) == 2:
                    # Both power type and wattage provided
                    power_cap_value = values[0]
                    power_cap_type = values[1]
                    if power_cap_type not in ["ppt0", "ppt1"]:
                        raise amdsmi_cli_exceptions.AmdSmiInvalidParameterException(
                            sys.argv[1], power_cap_type, output_format
                        )
                else:
                    raise amdsmi_cli_exceptions.AmdSmiInvalidParameterException(
                        sys.argv[1], values, output_format
                    )

                if not power_cap_value.isdigit():
                    raise amdsmi_cli_exceptions.AmdSmiInvalidParameterValueException(
                        sys.argv[1], power_cap_value, output_format
                    )

                power_cap_args = collections.namedtuple("power_cap_args", ["pwr_type", "watts"])
                setattr(namespace, self.dest, power_cap_args(power_cap_type, int(power_cap_value)))

        return AMDSMIPowerCapArgs

    def _cpu_power_eff_mode_options(self):
        """Custom action for CPU power efficiency mode validation"""
        output_format = self.helpers.get_output_format()

        class CPUPowerEffModeArgs(argparse.Action):
            def __call__(self, parser, namespace, values, option_string=None):
                if len(values) < 1:
                    raise amdsmi_cli_exceptions.AmdSmiInvalidParameterException(
                        sys.argv[1], _L("Missing MODE argument"), output_format
                    )

                mode = values[0]
                if mode < 0 or mode > 5:
                    raise amdsmi_cli_exceptions.AmdSmiInvalidParameterException(
                        sys.argv[1],
                        _L("Invalid MODE {mode}, must be 0-5", mode=mode),
                        output_format,
                    )
                # Get CPU family and model for validation
                try:
                    cpu_family = amdsmi_interface.amdsmi_get_cpu_family()
                    cpu_model = amdsmi_interface.amdsmi_get_cpu_model()
                except Exception as e:
                    # If we can't get CPU info, skip family/model validation
                    cpu_family = None
                    cpu_model = None

                # Check if this is the specific family/model that requires UTIL and PPT_LIMIT for modes 4,5
                requires_util_ppt = (
                    cpu_family == 0x1A and cpu_model is not None and 0x50 <= cpu_model <= 0x5F
                )

                # Mode-specific validation
                if mode in [4, 5]:  # BalancedCore, BalancedCoreMemory
                    if requires_util_ppt:
                        # Family 0x1A, models 0x50-0x5F: REQUIRE UTIL and PPT_LIMIT
                        if len(values) != 3:
                            raise amdsmi_cli_exceptions.AmdSmiInvalidParameterException(
                                sys.argv[1],
                                _L(
                                    "Mode {mode} requires MODE UTIL PPT_LIMIT for CPU family 0x{cpu_family:X} model 0x{cpu_model:X} (got {num_values} args)",
                                    mode=mode,
                                    cpu_family=cpu_family,
                                    cpu_model=cpu_model,
                                    num_values=len(values),
                                ),
                                output_format,
                            )

                        # Validate UTIL range
                        util = values[1]
                        if util < 0 or util > amdsmi_interface.AMDSMI_MAX_POWER_EFFICIENCY_UTIL:
                            raise amdsmi_cli_exceptions.AmdSmiInvalidParameterException(
                                sys.argv[1],
                                _L("UTIL must be 0-100, got {util}", util=util),
                                output_format,
                            )

                        # Validate PPT_LIMIT range
                        ppt_limit = values[2]
                        if (
                            ppt_limit < 0
                            or ppt_limit > amdsmi_interface.AMDSMI_MAX_POWER_EFFICIENCY_PPTLIMIT
                        ):
                            raise amdsmi_cli_exceptions.AmdSmiInvalidParameterException(
                                sys.argv[1], _L("PPT_LIMIT Out of range"), output_format
                            )
                    else:
                        # Other family/models: modes 4,5 valid but should NOT accept UTIL and PPT_LIMIT
                        if len(values) != 1:
                            raise amdsmi_cli_exceptions.AmdSmiInvalidParameterException(
                                sys.argv[1],
                                _L(
                                    "Mode {mode} requires only MODE argument for this CPU (got {num_values} args). UTIL and PPT_LIMIT not supported.",
                                    mode=mode,
                                    num_values=len(values),
                                ),
                                output_format,
                            )

                else:  # HighPerformance, PowerEfficiency, IOPerformance, BalancedMemory
                    if len(values) != 1:
                        raise amdsmi_cli_exceptions.AmdSmiInvalidParameterException(
                            sys.argv[1],
                            _L(
                                "Mode {mode} requires only MODE argument (got {num_values} args)",
                                mode=mode,
                                num_values=len(values),
                            ),
                            output_format,
                        )

                # Store values using 'append' behavior
                if not hasattr(namespace, self.dest) or getattr(namespace, self.dest) is None:
                    setattr(namespace, self.dest, [])
                getattr(namespace, self.dest).append(values)

        return CPUPowerEffModeArgs

    def _cpu_svi3_vr_controller_temp_options(self):
        """Custom action for CPU SVI3 VR controller temperature validation"""
        output_format = self.helpers.get_output_format()

        class CPUSvi3VrControllerTempArgs(argparse.Action):
            def __call__(self, parser, namespace, values, option_string=None):
                if len(values) < 1:
                    raise amdsmi_cli_exceptions.AmdSmiInvalidParameterException(
                        sys.argv[1], _L("Missing TYPE argument"), output_format
                    )

                rail_type = values[0]

                # Validate TYPE range
                if rail_type not in [0, 1]:
                    raise amdsmi_cli_exceptions.AmdSmiInvalidParameterException(
                        sys.argv[1],
                        _L(
                            "Invalid TYPE {rail_type}, must be 0 (HottestRail) or 1 (IndividualRail)",
                            rail_type=rail_type,
                        ),
                        output_format,
                    )

                # Type-specific validation
                if rail_type == 0:  # HottestRail
                    if len(values) != 1:
                        raise amdsmi_cli_exceptions.AmdSmiInvalidParameterException(
                            sys.argv[1],
                            _L(
                                "TYPE 0 (HottestRail) requires exactly 1 argument (TYPE only), got {num_values} arguments: {values}",
                                num_values=len(values),
                                values=values,
                            ),
                            output_format,
                        )

                elif rail_type == 1:  # IndividualRail
                    if len(values) != 2:
                        raise amdsmi_cli_exceptions.AmdSmiInvalidParameterException(
                            sys.argv[1],
                            _L(
                                "TYPE 1 (IndividualRail) requires exactly 2 arguments (TYPE RAIL_INDEX), got {num_values} arguments: {values}",
                                num_values=len(values),
                                values=values,
                            ),
                            output_format,
                        )

                    # Validate RAIL_INDEX range (0-4 based on help text)
                    rail_index = values[1]
                    if rail_index < 0 or rail_index > 4:
                        raise amdsmi_cli_exceptions.AmdSmiInvalidParameterException(
                            sys.argv[1],
                            _L(
                                "Invalid RAIL_INDEX {rail_index}, must be 0-4 (0=VDDCR_CPU0, 1=VDDCR_CPU1, 2=VDDCR_SOC, 3=VDDIO, 4=VDDIO_MEM_S3)",
                                rail_index=rail_index,
                            ),
                            output_format,
                        )

                # Store values using 'append' behavior
                if not hasattr(namespace, self.dest) or getattr(namespace, self.dest) is None:
                    setattr(namespace, self.dest, [])
                getattr(namespace, self.dest).append(values)

        return CPUSvi3VrControllerTempArgs

    def _check_output_file_path(self):
        """Argument action validator:
        Returns a path to a file from the output file path provided.
        If the path is a directory then create a file within it and return that file path
        If the path is a file and it exists return the file path
        If the path is a file and it doesn't exist create and return the file path
        """

        class CheckOutputFilePath(argparse.Action):
            outputformat = self.helpers.get_output_format()

            # Checks the values
            def __call__(self, parser, args, values, option_string=None):
                path = Path(values)
                if not path.exists():
                    if path.parent.is_dir():
                        path.touch()
                        setattr(args, self.dest, path)
                        return
                    else:
                        raise amdsmi_cli_exceptions.AmdSmiInvalidFilePathException(
                            path, CheckOutputFilePath.outputformat
                        )

                if path.is_dir():
                    file_name = f"{int(time.time())}-amdsmi-output"
                    if args.json:
                        file_name += ".json"
                    elif args.csv:
                        file_name += ".csv"
                    else:
                        file_name += ".txt"
                    path = path / file_name
                    path.touch()
                    setattr(args, self.dest, path)
                elif path.is_file():
                    # Check if --append or --overwrite flags are present in command line
                    has_append = "--append" in sys.argv
                    has_overwrite = "--overwrite" in sys.argv

                    if has_append or getattr(args, "append", False):
                        setattr(args, self.dest, path)
                        return
                    if has_overwrite or getattr(args, "overwrite", False):
                        path.open("w").close()
                        path.touch()
                        setattr(args, self.dest, path)
                        return
                    # Fail fast instead of blocking on the interactive overwrite/
                    # append prompt when stdin is not an interactive terminal
                    # (e.g. launched by an automation framework with stdin piped
                    # or closed). Reading from a held-open pipe can block input()
                    # forever; automation should pass --overwrite or --append to
                    # choose a behavior explicitly.
                    stdin_is_tty = False
                    try:
                        stdin_is_tty = sys.stdin is not None and sys.stdin.isatty()
                    except (ValueError, OSError):
                        stdin_is_tty = False
                    if not stdin_is_tty:
                        raise amdsmi_cli_exceptions.AmdSmiInvalidFilePathException(
                            path,
                            CheckOutputFilePath.outputformat,
                            _L(
                                "File '{path}' exists and stdin is not a TTY; cannot prompt to overwrite or append. Re-run with --overwrite or --append.",
                                path=path,
                            ),
                        )
                    # Prompt if neither --append nor --overwrite are specified
                    try:
                        resp = (
                            input(
                                f"File '{path}' exists. Overwrite (o) / Append (a) / Cancel (N) ? [o/a/N]: "
                            )
                            .strip()
                            .lower()
                        )
                    except Exception:
                        print(
                            _T("Confirmation not given. Exiting without setting value"),
                            file=sys.stderr,
                        )
                        sys.exit(int(amdsmi_cli_exceptions.AmdSmiExitCode.USER_ABORTED))
                    if resp in ("a", "append"):
                        setattr(args, self.dest, path)
                        return
                    elif resp in ("o", "yes"):
                        path.open("w").close()
                        setattr(args, self.dest, path)
                        return
                    else:
                        # Declining is not a bad path: the file was fine, the user said no.
                        print(
                            _T("User declined to overwrite or append existing file."),
                            file=sys.stderr,
                        )
                        sys.exit(int(amdsmi_cli_exceptions.AmdSmiExitCode.USER_ABORTED))
                else:
                    raise amdsmi_cli_exceptions.AmdSmiInvalidFilePathException(
                        path, CheckOutputFilePath.outputformat
                    )

        return CheckOutputFilePath

    def _check_cper_file_path(self):
        """Argument action validator:
        Returns a path to a file from the input file path provided.
        If the file doesn't exist, is empty, or is invalid, raise an error.
        """

        class _CheckInputFilePath(argparse.Action):
            # Checks the values
            outputformat = self.helpers.get_output_format()

            def __call__(self, parser, args, values, option_string=None):
                path = Path(values)
                try:
                    if not path.exists():
                        raise FileNotFoundError(
                            _T(
                                "CPER file could not be read. Make sure the path '{path}' is correct."
                            ).format(path=path)
                        )
                    if path.is_dir():
                        raise IsADirectoryError(
                            _T(
                                "Invalid Path: {path} is a directory when it needs to be a specific file."
                            ).format(path=path)
                        )
                    if path.is_file():
                        if os.stat(values).st_size == 0:
                            raise ValueError(
                                _T("Invalid Path: {path} Input file is empty.").format(path=path)
                            )
                        setattr(args, self.dest, path)
                    else:
                        raise FileNotFoundError(
                            _T(
                                "Invalid Path: {path} Could not determine if the value given is a valid path."
                            ).format(path=path)
                        )
                except Exception as root_cause:
                    raise amdsmi_cli_exceptions.AmdSmiInvalidFilePathException(
                        path, _CheckInputFilePath.outputformat
                    ) from root_cause

        return _CheckInputFilePath

    def _check_watch_selected(self):
        """Validate that the -w/--watch argument was selected
        This is because -W/--watch_time and -i/--iterations are dependent on watch
        """

        class WatchSelectedAction(argparse.Action):
            # Checks the values
            def __call__(self, parser, args, values, option_string=None):
                if args.watch is None:
                    raise argparse.ArgumentError(
                        self,
                        f"invalid argument: '{self.dest}' needs to be paired with -w/--watch. Error code: -2",
                    )
                else:
                    setattr(args, self.dest, values)

        return WatchSelectedAction

    def _gpu_select(self, gpu_choices):
        """Custom argparse action to return the device handle(s) for the gpu(s) selected
        This will set the destination (args.gpu) to a list of 1 or more device handles
        If 1 or more device handles are not found then raise an ArgumentError for the first invalid gpu seen
        """

        amdsmi_helpers = self.helpers

        class _GPUSelectAction(argparse.Action):
            outputformat = self.helpers.get_output_format()

            # Checks the values
            def __call__(self, parser, args, values, option_string=None):
                if "all" in gpu_choices:
                    del gpu_choices["all"]
                status, gpu_format, selected_device_handles = (
                    amdsmi_helpers.get_device_handles_from_gpu_selections(
                        gpu_selections=values, gpu_choices=gpu_choices
                    )
                )
                if status:
                    setattr(args, self.dest, selected_device_handles)
                else:
                    if selected_device_handles == "":
                        raise amdsmi_cli_exceptions.AmdSmiMissingParameterValueException(
                            "--gpu", _GPUSelectAction.outputformat
                        )
                    elif not gpu_format:
                        raise amdsmi_cli_exceptions.AmdSmiInvalidParameterValueException(
                            sys.argv[1], selected_device_handles, _GPUSelectAction.outputformat
                        )
                    else:
                        raise amdsmi_cli_exceptions.AmdSmiDeviceNotFoundException(
                            selected_device_handles,
                            _GPUSelectAction.outputformat,
                            amdsmi_cli_exceptions.AmdSmiDeviceKind.GPU,
                        )

        return _GPUSelectAction

    def _nic_select(self, nic_choices):
        """Custom argparse action to return the device handle(s) for the nics(s) selected
        This will set the destination (args.nic) to a list of 1 or more device handles
        If 1 or more device handles are not found then raise an ArgumentError for the first invalid nic seen
        """

        amdsmi_helpers = self.helpers

        class _NICSelectAction(argparse.Action):
            output_format = self.helpers.get_output_format()

            # Checks the values
            def __call__(self, parser, args, values, option_string=None):
                if "all" in nic_choices:
                    del nic_choices["all"]
                status, selected_device_handles = (
                    amdsmi_helpers.get_device_handles_from_nic_selections(
                        nic_selections=values, nic_choices=nic_choices
                    )
                )
                if status:
                    setattr(args, self.dest, selected_device_handles)
                else:
                    if selected_device_handles == "":
                        raise amdsmi_cli_exceptions.AmdSmiMissingParameterValueException(
                            "--nic", _NICSelectAction.output_format
                        )
                    else:
                        raise amdsmi_cli_exceptions.AmdSmiDeviceNotFoundException(
                            selected_device_handles,
                            _NICSelectAction.output_format,
                            amdsmi_cli_exceptions.AmdSmiDeviceKind.NIC,
                        )

        return _NICSelectAction

    def _switch_select(self, switch_choices):
        """Custom argparse action to return the device handle(s) for the switches(s) selected
        This will set the destination (args.switch) to a list of 1 or more device handles
        If 1 or more device handles are not found then raise an ArgumentError for the first invalid switch seen
        """

        amdsmi_helpers = self.helpers

        class _SwitchSelectAction(argparse.Action):
            output_format = self.helpers.get_output_format()

            # Checks the values
            def __call__(self, parser, args, values, option_string=None):
                if "all" in switch_choices:
                    del switch_choices["all"]
                status, selected_device_handles = (
                    amdsmi_helpers.get_device_handles_from_switch_selections(
                        switch_selections=values, switch_choices=switch_choices
                    )
                )
                if status:
                    setattr(args, self.dest, selected_device_handles)
                else:
                    if selected_device_handles == "":
                        raise amdsmi_cli_exceptions.AmdSmiMissingParameterValueException(
                            "--switch", _SwitchSelectAction.output_format
                        )
                    else:
                        raise amdsmi_cli_exceptions.AmdSmiDeviceNotFoundException(
                            selected_device_handles,
                            _SwitchSelectAction.output_format,
                            amdsmi_cli_exceptions.AmdSmiDeviceKind.SWITCH,
                        )

        return _SwitchSelectAction

    def _cpu_select(self, cpu_choices):
        """Custom argparse action to return the device handle(s) for the cpu(s) selected
        This will set the destination (args.cpu) to a list of 1 or more device handles
        If 1 or more device handles are not found then raise an ArgumentError for the first invalid cpu seen
        """
        amdsmi_helpers = self.helpers

        class _CPUSelectAction(argparse.Action):
            outputformat = self.helpers.get_output_format()

            # Checks the values
            def __call__(self, parser, args, values, option_string=None):
                if "all" in cpu_choices:
                    del cpu_choices["all"]
                status, cpu_format, selected_device_handles = (
                    amdsmi_helpers.get_device_handles_from_cpu_selections(
                        cpu_selections=values, cpu_choices=cpu_choices
                    )
                )
                if status:
                    setattr(args, self.dest, selected_device_handles)
                else:
                    if selected_device_handles == "":
                        raise amdsmi_cli_exceptions.AmdSmiMissingParameterValueException(
                            "--cpu", _CPUSelectAction.outputformat
                        )
                    elif not cpu_format:
                        raise amdsmi_cli_exceptions.AmdSmiInvalidParameterValueException(
                            sys.argv[1], selected_device_handles, _CPUSelectAction.outputformat
                        )
                    else:
                        raise amdsmi_cli_exceptions.AmdSmiDeviceNotFoundException(
                            selected_device_handles,
                            _CPUSelectAction.outputformat,
                            amdsmi_cli_exceptions.AmdSmiDeviceKind.CPU,
                        )

        return _CPUSelectAction

    def _core_select(self, core_choices):
        """Custom argparse action to return the device handle(s) for the core(s) selected
        This will set the destination (args.core) to a list of 1 or more device handles
        If 1 or more device handles are not found then raise an ArgumentError for the first invalid core seen
        """
        amdsmi_helpers = self.helpers

        class _CoreSelectAction(argparse.Action):
            outputformat = self.helpers.get_output_format()

            # Checks the values
            def __call__(self, parser, args, values, option_string=None):
                if "all" in core_choices:
                    del core_choices["all"]
                status, core_format, selected_device_handles = (
                    amdsmi_helpers.get_device_handles_from_core_selections(
                        core_selections=values, core_choices=core_choices
                    )
                )
                if status:
                    setattr(args, self.dest, selected_device_handles)
                else:
                    if selected_device_handles == "":
                        raise amdsmi_cli_exceptions.AmdSmiMissingParameterValueException(
                            "--core", _CoreSelectAction.outputformat
                        )
                    elif not core_format:
                        raise amdsmi_cli_exceptions.AmdSmiInvalidParameterValueException(
                            sys.argv[1], selected_device_handles, _CoreSelectAction.outputformat
                        )
                    else:
                        raise amdsmi_cli_exceptions.AmdSmiDeviceNotFoundException(
                            selected_device_handles,
                            _CoreSelectAction.outputformat,
                            amdsmi_cli_exceptions.AmdSmiDeviceKind.CPU_CORE,
                        )

        return _CoreSelectAction

    def _add_watch_arguments(self, subcommand_parser):
        # Device arguments help text
        watch_help = _T("Reprint the command in a loop of INTERVAL seconds")
        watch_time_help = _T("The total TIME to watch the given command")
        iterations_help = _T("Total number of ITERATIONS to loop on the given command")

        # Mutually Exclusive Args within the subparser
        subcommand_parser.add_argument(
            "-w",
            "--watch",
            action="store",
            metavar="INTERVAL",
            type=lambda value: self._positive_int(value, "--watch"),
            required=False,
            help=watch_help,
        )
        subcommand_parser.add_argument(
            "-W",
            "--watch_time",
            action=self._check_watch_selected(),
            metavar="TIME",
            type=lambda value: self._positive_int(value, "--watch_time"),
            required=False,
            help=watch_time_help,
        )
        subcommand_parser.add_argument(
            "-i",
            "--iterations",
            action=self._check_watch_selected(),
            metavar="ITERATIONS",
            type=lambda value: self._positive_int(value, "--iterations"),
            required=False,
            help=iterations_help,
        )

    def _add_watch_arguments(self, subcommand_parser: argparse.ArgumentParser):
        # Device arguments help text
        watch_help = _T("Reprint the command in a loop of INTERVAL seconds")
        watch_time_help = _T("The total duration of TIME to watch the command")
        iterations_help = _T("The total number of ITERATIONS to repeat the command")

        watch_arguments_group = subcommand_parser.add_argument_group(_T("Watch Arguments"))

        # Mutually Exclusive Args within the subparser
        watch_arguments_group.add_argument(
            "-w",
            "--watch",
            action="store",
            metavar="INTERVAL",
            type=lambda value: self._positive_int(value, "--watch"),
            required=False,
            help=watch_help,
        )
        watch_arguments_group.add_argument(
            "-W",
            "--watch_time",
            action=self._check_watch_selected(),
            metavar="TIME",
            type=lambda value: self._positive_int(value, "--watch_time"),
            required=False,
            help=watch_time_help,
        )
        watch_arguments_group.add_argument(
            "-i",
            "--iterations",
            action=self._check_watch_selected(),
            metavar="ITERATIONS",
            type=lambda value: self._positive_int(value, "--iterations"),
            required=False,
            help=iterations_help,
        )

        return watch_arguments_group

    def _validate_cpu_core(self, value):
        if value == "":
            outputformat = self.helpers.get_output_format()
            raise amdsmi_cli_exceptions.AmdSmiMissingParameterValueException(value, outputformat)
        if isinstance(value, str):
            if value.lower() == "all":
                return value
            if value.isdigit():
                if int(value) < 0:
                    outputformat = self.helpers.get_output_format()
                    raise amdsmi_cli_exceptions.AmdSmiInvalidParameterValueException(
                        sys.argv[1], value, outputformat
                    )
            else:
                outputformat = self.helpers.get_output_format()
                raise amdsmi_cli_exceptions.AmdSmiInvalidParameterValueException(
                    sys.argv[1], value, outputformat
                )

        if isinstance(value, int):
            if int(value) < 0:
                outputformat = self.helpers.get_output_format()
                raise amdsmi_cli_exceptions.AmdSmiInvalidParameterValueException(
                    sys.argv[1], value, outputformat
                )

        return value

    def _validate_set_clock(self, validate_clock_type=True):
        """Validate Clock input"""
        amdsmi_helpers = self.helpers

        class _ValidateClockType(argparse.Action):
            # Checks the clock type and clock values
            def __call__(self, parser, args, values, option_string=None):
                if validate_clock_type:
                    clock_type = values[0]
                    clock_types = amdsmi_helpers.get_clock_types()[0]
                    valid_clock_type, amdsmi_clock_type = amdsmi_helpers.validate_clock_type(
                        input_clock_type=clock_type
                    )
                    if not valid_clock_type:
                        raise argparse.ArgumentError(
                            self,
                            f"Invalid argument: '{clock_type}' needs to be a valid clock type:{clock_types}",
                        )

                    clock_levels = values[1:]
                else:
                    clock_levels = values

                freq_bitmask = 0
                for level in clock_levels:
                    if level > 63:
                        raise argparse.ArgumentError(
                            self,
                            f"Invalid argument: '{level}' needs to be a valid clock level 0-63",
                        )
                    freq_bitmask |= 1 << level

                if validate_clock_type:
                    setattr(args, self.dest, (amdsmi_clock_type, freq_bitmask))
                else:
                    setattr(args, self.dest, freq_bitmask)

        return _ValidateClockType

    def _prompt_spec_warning(self):
        """Prompt out of spec warning"""
        amdsmi_helpers = self.helpers

        class _PromptSpecWarning(argparse.Action):
            # Checks the values
            def __call__(self, parser, args, values, option_string=None):
                amdsmi_helpers.confirm_out_of_spec_warning()
                setattr(args, self.dest, values)

        return _PromptSpecWarning

    def _validate_fan_speed(self):
        """Validate fan speed input

        Accepts:
        - Percentage: 0-100% (converted at command layer based on GPU interface)
        - Direct value: 0-255 (legacy hwmon) or OD_RANGE min-max (gpu_od interface)

        Note: For direct values, we accept 0-255 here for backward compatibility.
        The command layer will validate against the correct range (OD_RANGE for gpu_od,
        0-255 for legacy) after detecting the GPU interface type.
        """
        amdsmi_helpers = self.helpers

        class _ValidateFanSpeed(argparse.Action):
            # Checks the values
            def __call__(self, parser, args, values, option_string=None):
                if isinstance(values, str):
                    if "%" in values:
                        try:
                            # Store percentage value with is_percentage flag
                            # CLI will convert based on gpu_od availability
                            percentage_value = int(values[:-1])
                            if 0 <= percentage_value <= 100:
                                # Making behavior consistent with recent non-percentage path changes
                                # 1. Check input validity (0-100%)
                                # 2. If valid, prompt out-of-spec warning. If not valid, raise argparse error without prompt.
                                # This ensures users are only prompted for valid inputs that may be out-of-spec, not for invalid inputs.
                                amdsmi_helpers.confirm_out_of_spec_warning()
                                # Store as tuple: (percentage_value, is_percentage=True)
                                setattr(args, self.dest, (percentage_value, True))
                            else:
                                raise argparse.ArgumentError(
                                    self, f"Invalid argument: '{values}' needs to be 0-100%"
                                )
                        except ValueError as e:
                            raise argparse.ArgumentError(
                                self, f"Invalid argument: '{values}' needs to be 0-100%"
                            )
                    else:  # Store the direct hardware value
                        values = int(values)
                        if 0 <= values <= 255:
                            amdsmi_helpers.confirm_out_of_spec_warning()
                            # Store as tuple: (direct_value, is_percentage=False)
                            # Note: Accepts 0-255 for legacy compatibility.
                            # Command layer validates against interface-specific range.
                            setattr(args, self.dest, (values, False))
                        else:
                            raise argparse.ArgumentError(
                                self, f"Invalid argument: '{values}' needs to be 0-255"
                            )
                else:
                    raise argparse.ArgumentError(
                        self, f"Invalid argument: '{values}' needs to be 0-255 or 0-100%"
                    )

        return _ValidateFanSpeed

    def _validate_overdrive_percent(self):
        """Validate overdrive percentage input"""
        amdsmi_helpers = self.helpers

        class _ValidateOverdrivePercent(argparse.Action):
            # Checks the values
            def __call__(self, parser, args, values, option_string=None):
                if isinstance(values, str):
                    try:
                        if values[-1] == "%":
                            over_drive_percent = int(values[:-1])
                        else:
                            over_drive_percent = int(values)

                        if 0 <= over_drive_percent <= 20:
                            amdsmi_helpers.confirm_out_of_spec_warning()
                            setattr(args, self.dest, over_drive_percent)
                        else:
                            raise argparse.ArgumentError(
                                self,
                                f"Invalid argument: '{values}' needs to be within range 0-20 or 0-20%",
                            )
                    except ValueError:
                        raise argparse.ArgumentError(
                            self, f"Invalid argument: '{values}' needs to be 0-20 or 0-20%"
                        )
                else:
                    raise argparse.ArgumentError(
                        self, f"Invalid argument: '{values}' needs to be 0-20 or 0-20%"
                    )

        return _ValidateOverdrivePercent

    def _validate_ptl_format(self):
        """Validate and normalize --ptl-format FRMT1,FRMT2."""
        valid_names = [
            name for name in amdsmi_interface.AmdSmiPtlData.__members__ if name != "INVALID"
        ]

        class _ValidatePtlFormat(argparse.Action):
            def __call__(self, parser, args, values, option_string=None):
                if not isinstance(values, str):
                    raise argparse.ArgumentError(
                        self,
                        f"Invalid argument for {option_string}: '{values}' "
                        f"(expected string like I8,F32)",
                    )

                parts = values.split(",")
                if len(parts) != 2:
                    raise argparse.ArgumentError(
                        self,
                        f"{option_string} expects exactly two comma-separated formats, "
                        f"e.g. I8,F32 (got '{values}')",
                    )

                enums = []
                for raw in parts:
                    token = raw.strip().upper()
                    if token == "":
                        raise argparse.ArgumentError(
                            self, f"Empty PTL format in '{values}'. Expected formats like I8,F32."
                        )
                    try:
                        enum_val = amdsmi_interface.AmdSmiPtlData[token]
                    except KeyError:
                        raise argparse.ArgumentError(
                            self,
                            f"Invalid PTL format '{raw}'. Valid formats are: "
                            + ", ".join(valid_names),
                        )
                    if enum_val == amdsmi_interface.AmdSmiPtlData.INVALID:
                        raise argparse.ArgumentError(self, f"INVALID is not a usable PTL format.")
                    enums.append(enum_val)

                if enums[0] == enums[1]:
                    raise argparse.ArgumentError(
                        self, f"PTL formats must be different (got '{values}')."
                    )

                setattr(args, self.dest, (enums[0], enums[1]))

        return _ValidatePtlFormat

    def _ampp_configure_options(self):
        """Custom action for --ampp-configure PROFILE_NAME KEY=VALUE [KEY=VALUE ...]"""
        output_format = self.helpers.get_output_format()

        class AMDSMIAmppConfigureArgs(argparse.Action):
            def __call__(
                self,
                parser: AMDSMIParser,
                namespace: argparse.Namespace,
                values: list,
                option_string: Optional[str] = None,
            ) -> None:
                if not values:
                    raise amdsmi_cli_exceptions.AmdSmiInvalidParameterException(
                        sys.argv[1], _L("Missing PROFILE_NAME argument"), output_format
                    )

                ampp_configure_args = collections.namedtuple(
                    "ampp_configure_args", ["profile_name", "fields", "file_path"]
                )

                # `@<path>` form: restore every writable profile with staged
                # fields found in a JSON file (as produced by
                # `amd-smi static --ampp --json`) in one call.
                if values[0].startswith("@"):
                    if len(values) > 1:
                        raise amdsmi_cli_exceptions.AmdSmiInvalidParameterValueException(
                            sys.argv[1],
                            " ".join(values),
                            output_format,
                            hint=_L(
                                "'@<path>' must be the only argument when loading profiles from a file."
                            ),
                        )
                    file_path = values[0][1:]
                    if not file_path:
                        raise amdsmi_cli_exceptions.AmdSmiInvalidParameterValueException(
                            sys.argv[1],
                            values[0],
                            output_format,
                            hint=_L("File path cannot be empty after '@'."),
                        )
                    setattr(namespace, self.dest, ampp_configure_args(None, None, file_path))
                    return

                profile_name = values[0]
                field_tokens = values[1:]

                if not AMDSMIHelpers.is_valid_ampp_field_name(profile_name):
                    raise amdsmi_cli_exceptions.AmdSmiInvalidParameterValueException(
                        sys.argv[1],
                        ascii(profile_name),
                        output_format,
                        hint=_L(
                            "Profile name cannot be empty or exceed {AMDSMI_MAX_STRING_LENGTH} bytes.",
                            AMDSMI_MAX_STRING_LENGTH=amdsmi_interface.AMDSMI_MAX_STRING_LENGTH,
                        ),
                    )

                if not field_tokens:
                    raise amdsmi_cli_exceptions.AmdSmiInvalidParameterException(
                        sys.argv[1],
                        _L("Missing at least one KEY=VALUE field argument for --ampp-configure"),
                        output_format,
                    )

                fields = []
                for token in field_tokens:
                    if "=" not in token:
                        raise amdsmi_cli_exceptions.AmdSmiInvalidParameterValueException(
                            sys.argv[1],
                            token,
                            output_format,
                            hint=_L("Fields must be given as KEY=VALUE (e.g. PPT0_Limit=300)."),
                        )
                    key, _, raw_value = token.partition("=")
                    key = key.strip()
                    raw_value = raw_value.strip()
                    # Shares its name/value checks with the @<file> JSON
                    # restore path (set_value.py) via AMDSMIHelpers so the
                    # two can't drift apart.
                    if not AMDSMIHelpers.is_valid_ampp_field_name(key):
                        raise amdsmi_cli_exceptions.AmdSmiInvalidParameterValueException(
                            sys.argv[1],
                            token,
                            output_format,
                            hint=_L(
                                "Field name cannot be empty or exceed {AMDSMI_MAX_STRING_LENGTH} bytes in KEY=VALUE.",
                                AMDSMI_MAX_STRING_LENGTH=amdsmi_interface.AMDSMI_MAX_STRING_LENGTH,
                            ),
                        )
                    value = AMDSMIHelpers.parse_ampp_field_value(raw_value)
                    if value is None:
                        raise amdsmi_cli_exceptions.AmdSmiInvalidParameterValueException(
                            sys.argv[1],
                            token,
                            output_format,
                            hint=_L(
                                "Field value must be an integer within int64 range (got '{raw_value}').",
                                raw_value=raw_value,
                            ),
                        )
                    fields.append({"name": key, "value": value})

                setattr(namespace, self.dest, ampp_configure_args(profile_name, fields, None))

        return AMDSMIAmppConfigureArgs

    ### Building parsers ###
    @staticmethod
    def _guard_gtt_gpu_conflict(
        parser, gtt_flags=("--gtt", "-G"), reason="--gtt is a system-wide setting, not per-GPU"
    ):
        """Override *parser*.error() so that combining any of *gtt_flags* with
        --gpu / -g produces a clear mutual-exclusion message instead of
        the confusing "expected at least one argument" from --gpu. Only
        argparse's missing-value diagnostic is replaced."""
        _original_error = parser.error

        def _intercept(message):
            error_prefix = message.split(":", 1)[0]
            if (
                set(gtt_flags).intersection(sys.argv)
                and {"--gpu", "-g"}.intersection(sys.argv)
                and error_prefix in {"argument -g/--gpu", "argument --gpu/-g"}
                and "expected at least one argument" in message
            ):
                flag_str = "/".join(gtt_flags)
                _original_error(
                    f"argument {flag_str}: not allowed with argument --gpu/-g ({reason})"
                )
            _original_error(message)

        parser.error = _intercept

    def _add_device_arguments(self, subcommand_parser: argparse.ArgumentParser, required=False):
        # Device arguments help text
        gpu_help = _T(
            "Select a GPU ID, BDF, or UUID from the possible choices:\n{gpu_choices_str}"
        ).format(gpu_choices_str=self.gpu_choices_str)
        vf_help = _T(
            "Gets general information about the specified VF (timeslice, fb info, …).\
                    \nAvailable only on virtualization OSs"
        )
        cpu_help = _T("Select a CPU ID from the possible choices:\n{cpu_choices_str}").format(
            cpu_choices_str=self.cpu_choices_str
        )
        nic_help = _T("Select a NIC ID from the possible choices:\n{nic_choices_str}").format(
            nic_choices_str=self.nic_choices_str
        )
        core_help = _T("Select a Core ID from the possible choices:\n{core_choices_str}").format(
            core_choices_str=self.core_choices_str
        )

        # Create argument group for all the devices
        device_group = subcommand_parser.add_argument_group(_T("Device Arguments"))

        # Mutually Exclusive Args within the subparser
        device_args = device_group.add_mutually_exclusive_group(required=required)

        if self.helpers.is_amdgpu_initialized():
            device_args.add_argument(
                "-g", "--gpu", action=self._gpu_select(self.gpu_choices), nargs="+", help=gpu_help
            )

        if self.helpers.is_amd_hsmp_initialized():
            device_args.add_argument(
                "-U",
                "--cpu",
                type=self._validate_cpu_core,
                action=self._cpu_select(self.cpu_choices),
                nargs="+",
                help=cpu_help,
            )
            if subcommand_parser._optionals.title != _T("Static Arguments"):
                device_args.add_argument(
                    "-O",
                    "--core",
                    type=self._validate_cpu_core,
                    action=self._core_select(self.core_choices),
                    nargs="+",
                    help=core_help,
                )

        if self.helpers.is_hypervisor():
            device_args.add_argument(
                "-v", "--vf", action="store", nargs="+", help=vf_help, choices=self.vf_choices
            )

        if self.helpers.is_ainic_initialized() or self.helpers.is_brcm_nic_initialized():
            nic_help = _T(
                "Select a NIC ID, BDF, or UUID from the possible choices:\n{nic_choices_str}"
            ).format(nic_choices_str=self.nic_choices_str)
            device_args.add_argument(
                "-N", "--nic", action=self._nic_select(self.nic_choices), nargs="+", help=nic_help
            )
        if self.helpers.is_brcm_switch_initialized():
            switch_help = _T(
                "Select a SWITCH ID, BDF, or UUID from the possible choices:\n{switch_choices_str}"
            ).format(switch_choices_str=self.switch_choices_str)
            device_args.add_argument(
                "--switch",
                action=self._switch_select(self.switch_choices),
                nargs="+",
                help=switch_help,
            )

    def _add_command_modifiers(self, subcommand_parser: argparse.ArgumentParser):
        json_help = _T("Displays output in JSON format")
        csv_help = _T("Displays output in CSV format")
        file_help = _T("Saves output into a file on the provided path")
        loglevel_choices = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        loglevel_choices_str = ", ".join(loglevel_choices)
        loglevel_help = _T(
            "Set the logging level from the possible choices:\n    {loglevel_choices_str}"
        ).format(loglevel_choices_str=loglevel_choices_str)

        command_modifier_group = subcommand_parser.add_argument_group(_T("Command Modifiers"))

        # Output Format options
        logging_args = command_modifier_group.add_mutually_exclusive_group()
        logging_args.add_argument("--json", action="store_true", required=False, help=json_help)
        logging_args.add_argument("--csv", action="store_true", required=False, help=csv_help)

        command_modifier_group.add_argument(
            "--file",
            action=self._check_output_file_path(),
            type=str,
            required=False,
            help=file_help,
        )
        command_modifier_group.add_argument(
            "--overwrite", action="store_true", required=False, help=_T("Overwrite the file")
        )
        command_modifier_group.add_argument(
            "--append", action="store_true", required=False, help=_T("Append to the file")
        )
        # Placing loglevel outside the subcommands so it can be used with any subcommand
        command_modifier_group.add_argument(
            "--loglevel",
            action="store",
            type=str.upper,
            required=False,
            help=loglevel_help,
            default="ERROR",
            metavar="LEVEL",
            choices=loglevel_choices,
        )

        return command_modifier_group

    def _add_default_parser(self, subparsers: argparse._SubParsersAction, func):
        # there should be no args to parse here so let this be a dummy function to preserve later logic
        default_parser = subparsers.add_parser("default", description=None)
        default_parser._optionals.title = None
        default_parser.formatter_class = lambda prog: AMDSMISubparserHelpFormatter(prog)
        default_parser.set_defaults(func=func)

        # Add Universal Arguments
        self._add_command_modifiers(default_parser)

    def _add_version_parser(self, subparsers: argparse._SubParsersAction, func):
        # Subparser help text
        version_help = _T("Display version information")
        description = self.description

        # Create version subparser
        version_parser = subparsers.add_parser(
            "version", help=version_help, description=description
        )
        version_parser._optionals.title = None
        version_parser.formatter_class = lambda prog: AMDSMISubparserHelpFormatter(prog)
        version_parser.set_defaults(func=func)

        # Add Universal Arguments
        self._add_command_modifiers(version_parser)

        # help info
        gpu_version_help = _T("Display the current amdgpu driver version")
        cpu_version_help = _T("Display the current amd_hsmp or hsmp_acpi driver version")
        nic_version_help = _T("Display the current nic driver version")

        # Add GPU and CPU version Arguments
        version_parser.add_argument(
            "-g",
            "--gpu_version",
            action="store_true",
            required=False,
            help=gpu_version_help,
            default=None,
        )
        version_parser.add_argument(
            "-c",
            "--cpu_version",
            action="store_true",
            required=False,
            help=cpu_version_help,
            default=None,
        )
        version_parser.add_argument(
            "-n",
            "--nic_version",
            action="store_true",
            required=False,
            help=nic_version_help,
            default=None,
        )

    def _add_list_parser(self, subparsers: argparse._SubParsersAction, func):
        if not self.helpers.is_amdgpu_initialized():
            # The list subcommand is only applicable to systems with amdgpu initialized
            return

        # Subparser help text
        list_help = _T("List GPU information")
        list_optionals_title = _T("List Arguments")
        list_subcommand_help = _T(
            "{description}\n\nLists all detected devices on the system.                            \nLists the BDF, UUID, KFD_ID, NODE_ID, and Partition ID for each GPU and/or CPUs.                            \nIn virtualization environments, it can also list VFs associated to each                            \nGPU with some basic information for each VF."
        ).format(description=self.description)
        enumeration_help = _T(
            "Enumeration mapping to other features.\
                            \n    Includes CARD, RENDER, HSA_ID, HIP_ID, HIP_UUID, OAM_ID, and PHYSICAL_ACC_ID"
        )

        # Create list subparser
        list_parser = subparsers.add_parser(
            "list", help=list_help, description=list_subcommand_help
        )
        list_parser._optionals.title = list_optionals_title
        list_parser.formatter_class = lambda prog: AMDSMISubparserHelpFormatter(prog)
        list_parser.set_defaults(func=func)

        # Create -e subparser
        list_parser.add_argument("-e", "--enumeration", action="store_true", help=enumeration_help)

        # Add Universal Arguments
        self._add_device_arguments(list_parser, required=False)
        self._add_command_modifiers(list_parser)

    def _add_static_parser(self, subparsers: argparse._SubParsersAction, func):
        # Subparser help text
        static_help = _T("Gets static information about the specified GPU")
        static_subcommand_help = _T(
            "{description}\n\nIf no GPU is specified, returns static information for all GPUs on the system.                                \nIf no static argument is provided, all static information will be displayed."
        ).format(description=self.description)
        static_optionals_title = _T("Static Arguments")

        # Optional arguments help text
        asic_help = _T("All asic information")
        bus_help = _T("All bus information")
        vbios_help = _T("All video bios/IFWI information (if available)")
        limit_help = _T("All limit metric values (i.e. power and thermal limits)")
        driver_help = _T("Displays driver version")
        vram_help = _T("All vram information")
        cache_help = _T("All cache information")
        board_help = _T("All board information")
        soc_pstate_help = _T("The available soc pstate policy")
        xgmi_plpd_help = _T("The available XGMI per-link power down policy")
        process_isolation_help = _T("The process isolation status")
        cuid_help = _T(
            "Component Unified ID: the derived CUID, its component type, whether it is a\n"
            "temporary (auxiliary) identifier, and which layer answered.\n"
            "Not part of the default `amd-smi static` output; ask for it.\n"
            "The primary CUID is not shown by default either: its payload embeds the raw\n"
            "serial number. Add --cuid-primary, as root, to include it."
        )
        cuid_primary_help = _T(
            "Include the primary CUID in the CUID output, implying --cuid. Requires\n"
            "privilege, and prints a value that embeds the device serial number. Do not\n"
            "paste it into a bug report."
        )
        profile_help = _T("Display current and available power profiles")
        clk_options = self.helpers.get_clock_types()[0]
        clk_options.remove("PCIE")
        clk_option_str = ", ".join(clk_options) + ", ALL"
        clock_help = _T(
            "Show one or more valid clock frequency levels. Available options:\n\t{clk_option_str}"
        ).format(clk_option_str=clk_option_str)

        # Options arguments help text for Hypervisors and Baremetal
        # Might be able to remove Sudo requirement in ROCm 7.0
        ras_help = _T(
            "Displays RAS features information;\n\tSudo may be required for some features"
        )
        numa_help = _T("All numa node information")  # Linux only
        partition_help = _T(
            "Partition information:\n\t"
            "No longer available in default output.\n\tArgument is required to display."
            "\n\tEx. `amd-smi static -p` or use"
            "\n\t`amd-smi partition -c -m`/`sudo amd-smi partition -a`"
        )

        # Options arguments help text for Hypervisors
        dfc_help = _T("All DFC FW table information")
        fb_help = _T("Displays Frame Buffer information")
        num_vf_help = _T("Displays number of supported and enabled VFs")

        # Options arguments help text for CPUs
        smu_help = _T("All SMU FW information")
        interface_help = _T("Displays hsmp interface version")

        # Create static subparser
        static_parser = subparsers.add_parser(
            "static", help=static_help, description=static_subcommand_help
        )
        static_parser._optionals.title = static_optionals_title
        static_parser.formatter_class = lambda prog: AMDSMISubparserHelpFormatter(prog)
        static_parser.set_defaults(func=func)

        # Handle GPU Options
        if self.helpers.is_amdgpu_initialized():
            static_parser.add_argument(
                "-a", "--asic", action="store_true", required=False, help=asic_help
            )
            static_parser.add_argument(
                "-b", "--bus", action="store_true", required=False, help=bus_help
            )
            # Accept vbios args without displaying them
            static_parser.add_argument(
                "-V",
                "--vbios",
                dest="vbios",
                action="store_true",
                required=False,
                help=argparse.SUPPRESS,
            )
            static_parser.add_argument(
                "-I", "--ifwi", dest="vbios", action="store_true", required=False, help=vbios_help
            )
            static_parser.add_argument(
                "-d", "--driver", action="store_true", required=False, help=driver_help
            )
            static_parser.add_argument(
                "-v", "--vram", action="store_true", required=False, help=vram_help
            )
            static_parser.add_argument(
                "-c", "--cache", action="store_true", required=False, help=cache_help
            )
            static_parser.add_argument(
                "-B", "--board", action="store_true", required=False, help=board_help
            )
            static_parser.add_argument(
                "-R",
                "--process-isolation",
                action="store_true",
                required=False,
                help=process_isolation_help,
            )
            static_parser.add_argument(
                "-r", "--ras", action="store_true", required=False, help=ras_help
            )
            static_parser.add_argument(
                "-Y", "--cuid", action="store_true", required=False, help=cuid_help
            )
            static_parser.add_argument(
                "-y", "--cuid-primary", action="store_true", required=False, help=cuid_primary_help
            )
            static_parser.add_argument(
                "-C",
                "--clock",
                action="store",
                default=False,
                nargs="*",
                type=str,
                required=False,
                help=clock_help,
            )
            static_parser.add_argument(
                "-p", "--partition", action="store_true", required=False, help=partition_help
            )

            mem_carveout_help = _T(
                "Display VRAM carveout memory options and current setting."
                "\n\tOnly supported on some APUs."
            )
            static_parser.add_argument(
                "-m", "--mem-carveout", action="store_true", required=False, help=mem_carveout_help
            )

            ampp_help = _T(
                "Display AMPP (amdsmi power profile) recipe table: per-profile"
                " version/active/writable/configured state and per-field"
                " name/unit/value/min/max."
            )
            static_parser.add_argument(
                "-A", "--ampp", action="store_true", required=False, help=ampp_help
            )

            # Options to display on Hypervisors and Baremetal
            if self.helpers.is_hypervisor() or self.helpers.is_baremetal():
                static_parser.add_argument(
                    "-l", "--limit", action="store_true", required=False, help=limit_help
                )
                static_parser.add_argument(
                    "-P", "--soc-pstate", action="store_true", required=False, help=soc_pstate_help
                )
                static_parser.add_argument(
                    "-x", "--xgmi-plpd", action="store_true", required=False, help=xgmi_plpd_help
                )
                static_parser.add_argument(
                    "-o", "--profile", action="store_true", required=False, help=profile_help
                )

            if self.helpers.is_linux():
                static_parser.add_argument(
                    "-u", "--numa", action="store_true", required=False, help=numa_help
                )

            # Options to only display on a Hypervisor TODO: Add hypervisor driver check
            if self.helpers.is_hypervisor():
                static_parser.add_argument(
                    "-d", "--dfc-ucode", action="store_true", required=False, help=dfc_help
                )
                static_parser.add_argument(
                    "-f", "--fb-info", action="store_true", required=False, help=fb_help
                )
                static_parser.add_argument(
                    "-n", "--num-vf", action="store_true", required=False, help=num_vf_help
                )

        # Handle CPU Options
        if self.helpers.is_amd_hsmp_initialized():
            cpu_group = static_parser.add_argument_group(_T("CPU Arguments"))
            cpu_group.add_argument(
                "-s", "--smu", action="store_true", required=False, help=smu_help
            )
            cpu_group.add_argument(
                "-i", "--interface-ver", action="store_true", required=False, help=interface_help
            )

        # Add Universal Arguments
        self._add_device_arguments(static_parser, required=False)
        self._add_command_modifiers(static_parser)

    def _add_firmware_parser(self, subparsers: argparse._SubParsersAction, func):
        if not self.helpers.is_amdgpu_initialized():
            # The firmware subcommand is only applicable to systems with amdgpu initialized
            return

        # Subparser help text
        firmware_help = _T("Gets firmware information about the specified GPU")
        firmware_subcommand_help = _T(
            "{description}\n\nIf no GPU is specified, return firmware information for all GPUs on the system."
        ).format(description=self.description)
        firmware_optionals_title = _T("Firmware Arguments")

        # Optional arguments help text
        fw_list_help = _T("All FW list information")
        nic_firmware_help = _T("Broadcom NIC firmware attributes")
        err_records_help = _T("All error records information")

        # Create firmware subparser
        firmware_parser = subparsers.add_parser(
            "firmware", help=firmware_help, description=firmware_subcommand_help, aliases=["ucode"]
        )
        firmware_parser._optionals.title = firmware_optionals_title
        firmware_parser.formatter_class = lambda prog: AMDSMISubparserHelpFormatter(prog)
        firmware_parser.set_defaults(func=func)

        # Optional Args
        firmware_parser.add_argument(
            "-f",
            "--ucode-list",
            "--fw-list",
            dest="fw_list",
            action="store_true",
            required=False,
            help=fw_list_help,
            default=True,
        )
        if self.helpers.is_brcm_nic_initialized():
            firmware_parser.add_argument(
                "--brcm_nic", action="store_true", required=False, help=nic_firmware_help
            )

        # Options to only display on a Hypervisor
        if self.helpers.is_hypervisor():
            firmware_parser.add_argument(
                "-e", "--error-records", action="store_true", required=False, help=err_records_help
            )

        # Add Universal Arguments
        self._add_device_arguments(firmware_parser, required=False)
        self._add_command_modifiers(firmware_parser)

    def _add_bad_pages_parser(self, subparsers: argparse._SubParsersAction, func):
        if not (self.helpers.is_baremetal() and self.helpers.is_linux()):
            # The bad_pages subcommand is only applicable to Linux Baremetal systems
            return

        if not self.helpers.is_amdgpu_initialized():
            # The bad_pages subcommand is only applicable to systems with amdgpu initialized
            return

        # Subparser help text
        bad_pages_help = _T("Gets bad page information about the specified GPU")
        bad_pages_subcommand_help = _T(
            "{description}\n\nIf no GPU is specified, return bad page information for all GPUs on the system."
        ).format(description=self.description)
        bad_pages_optionals_title = _T("Bad Pages Arguments")

        # Optional arguments help text
        pending_help = _T("Displays all pending retired pages")
        retired_help = _T("Displays retired pages")
        un_res_help = _T("Displays unreservable pages")
        hex_help = _T("Displays page addresses and sizes in hexadecimal format")

        # Create bad_pages subparser
        bad_pages_parser = subparsers.add_parser(
            "bad-pages", help=bad_pages_help, description=bad_pages_subcommand_help
        )
        bad_pages_parser._optionals.title = bad_pages_optionals_title
        bad_pages_parser.formatter_class = lambda prog: AMDSMISubparserHelpFormatter(prog)
        bad_pages_parser.set_defaults(func=func)

        # Optional Args
        bad_pages_parser.add_argument(
            "-p", "--pending", action="store_true", required=False, help=pending_help
        )
        bad_pages_parser.add_argument(
            "-r", "--retired", action="store_true", required=False, help=retired_help
        )
        bad_pages_parser.add_argument(
            "-u", "--un-res", action="store_true", required=False, help=un_res_help
        )
        bad_pages_parser.add_argument(
            "-x", "--hex", action="store_true", required=False, help=hex_help
        )

        # Add Universal Arguments
        self._add_device_arguments(bad_pages_parser, required=False)
        self._add_command_modifiers(bad_pages_parser)

    def _add_metric_parser(self, subparsers: argparse._SubParsersAction, func):
        # Subparser help text
        metric_help = _T(
            "Gets metric/performance information about the specified GPU, NIC, or switch"
        )
        metric_subcommand_help = _T(
            "{description}\n\nIf no GPU is specified, returns metric information for all GPUs on the system.                                \nIf no metric argument is provided, all metric information will be displayed."
        ).format(description=self.description)
        metric_optionals_title = _T("Metric arguments")

        # Optional arguments help text
        usage_help = _T("Displays engine usage information")

        # Help text for Arguments only Available on Linux Virtual OS and Baremetal platforms
        mem_usage_help = _T("Memory usage per block")
        nic_metric_help = _T("Broadcom NIC metric attributes")
        switch_metric_help = _T("Broadcom switch metric attributes")

        # Help text for Arguments only on Hypervisor and Baremetal platforms
        power_help = _T("Current power usage")
        clock_help = _T("Average, max, and current clock frequencies")
        temperature_help = _T("Current temperatures")
        ecc_help = _T("Total number of ECC errors")
        ecc_blocks_help = _T("Number of ECC errors per block")
        pcie_help = _T("Current PCIe speed, width, and replay count")
        voltage_help = _T("GPU voltage")
        base_board_help = _T("base_board temperatures")
        gpu_board_help = _T("gpu_board temperatures")

        # Help text for Arguments only on Linux Baremetal platforms
        fan_help = _T("Current fan speed")
        vc_help = _T("Display voltage curve")
        overdrive_help = _T("Current GFX and MEM clock overdrive level")
        perf_level_help = _T("Current DPM performance level")
        xgmi_err_help = _T("XGMI error information since last read")
        energy_help = _T("Amount of energy consumed")
        throttle_help = _T(
            "Displays throttle accumulators;\n    Only available for MI300 or newer ASICs and APUs"
        )
        partition_help = _T(
            "Switch temperature, clock, and usage to partition-scoped\n"
            "    (XCP/AID/MID) data sources; combine with those flags to scope it;"
            "\n    Only available for MI300 or newer ASICs"
        )
        show_unsupported_help = _T(
            "Print every field, including the ones the GPU's gpu_metrics\n"
            "    table version cannot carry and which are omitted by default;\n"
            "    affects human-readable output only, since --json and --csv\n"
            "    always print every field"
        )

        # Help text for Arguments only on Hypervisors
        schedule_help = _T("All scheduling information")
        guard_help = _T("All guard information")
        guest_data_help = _T("All guest data information")
        fb_usage_help = _T("Displays total and used Frame Buffer usage information")
        xgmi_help = _T("Table of current XGMI metrics information")

        # Help text for cpu options
        cpu_power_metrics_help = _T("CPU power metrics")
        cpu_proc_help = _T("Displays prochot status")
        cpu_freq_help = _T("Displays currentFclkMemclk frequencies and cclk frequency limit")
        cpu_c0_res_help = _T("Displays C0 residency")
        cpu_lclk_dpm_help = _T(
            "Displays lclk dpm level range. Requires socket ID and NBOID as inputs"
        )
        cpu_pwr_eff_mode_help = _T(
            "Displays current power efficiency mode.\
        \n For Family 1Ah Models 50h-57h onwards and MODE= 4 or 5, displays utilization percentage and PPT limit in Watts."
        )
        cpu_pwr_svi_telemetry_rails_help = _T("Displays svi based telemetry for all rails")
        cpu_io_bandwidth_help = _T(
            "Displays current IO bandwidth for the selected CPU.\
        \n input parameters are bandwidth type(1) and link ID encodings\
        \n i.e. P2, P3, G0 - G7"
        )
        cpu_xgmi_bandwidth_help = _T(
            "Displays current XGMI bandwidth for the selected CPU\
        \n input parameters are bandwidth type(1,2,4) and link ID encodings\
        \n i.e. P2, P3, G0 - G7"
        )
        cpu_metrics_ver_help = _T("Displays metrics table version")
        cpu_metrics_table_help = _T("Displays metric table")
        cpu_socket_energy_help = _T("Displays socket energy for the selected CPU socket")
        cpu_ddr_bandwidth_help = _T(
            "Displays per socket max ddr bw, current utilized bw,\
        \n and current utilized ddr bw in percentage"
        )
        cpu_temp_help = _T("Displays cpu socket temperature")
        cpu_dimm_temp_range_rate_help = _T("Displays dimm temperature range and refresh rate")
        cpu_dimm_pow_consumption_help = _T("Displays dimm power consumption")
        cpu_dimm_thermal_sensor_help = _T("Displays dimm thermal sensor")
        cpu_xgmi_pstate_range_help = _T(
            "Displays XGMI pstate range, min and max values for the selected CPU"
        )
        cpu_railisofreq_policy_help = _T("Displays CPU ISO frequency policy")
        cpu_dfcstate_ctrl_help = _T("Displays DFCState control status")
        cpu_pc6_enable_help = _T("Displays PC6 enable control")
        cpu_cc6_enable_help = _T("Displays CC6 enable control")
        cpu_dimm_sb_reg_help = _T(
            "Read DIMM sideband register.Requires DIMM_ADDR, LID(0x2->TS0,0x6->TS1,0x9->PMIC0,0xA->SPDHub)\
        \n REG_OFFSET (hex), REG_SPACE (REGSPACE:0->Volatile,1->NVM)"
        )
        cpu_tdelta_help = _T(
            "Displays CPU thermal delta (TDELTA) value for the selected CPU socket"
        )
        cpu_svi3_vr_controller_temp_help = _T(
            "Get SVI3 VR controller temperature. TYPE: 0=HottestRail, 1=IndividualRail, RAIL_INDEX:\
        \n (RAIL_INDEX:0->VDDCR_CPU0,1->VDDCR_CPU1,2->VDDCR_SOC,3->VDDIO,4->VDDIO_MEM_S3) must be, specified if TYPE is 1"
        )
        cpu_enabled_commands_help = _T(
            "Displays HSMP enabled commands bit masks (Read/Write EnabledCommandsBitMask0-2)"
        )
        cpu_sdps_limit_help = _T("Displays CPU SDPS limit for the selected CPU socket in Watts")

        # Help text for core options
        core_energy_help = _T("Displays core energy for the selected core")
        core_boost_limit_help = _T("Get boost limit for the selected cores")
        core_curr_active_freq_core_limit_help = _T("Get Current CCLK limit set per Core")
        core_ccd_power_help = _T("Displays ccd power consumption for the selected core")
        core_floor_limit_help = _T("Displays floor limit frequency for the selected core in MHz")
        core_eff_floor_limit_help = _T(
            "Displays effective floor limit frequency for the selected core in MHz"
        )

        # Create metric subparser
        metric_parser = subparsers.add_parser(
            "metric", help=metric_help, description=metric_subcommand_help
        )
        metric_parser._optionals.title = metric_optionals_title
        metric_parser.formatter_class = lambda prog: AMDSMISubparserHelpFormatter(prog)
        metric_parser.set_defaults(func=func)

        # Optional Args for Linux Virtual OS and Baremetal systems
        if not self.helpers.is_hypervisor() and not self.helpers.is_windows():
            metric_parser.add_argument(
                "-m", "--mem-usage", action="store_true", required=False, help=mem_usage_help
            )

        if self.helpers.is_amdgpu_initialized():
            # Optional Args for Hypervisors and Baremetal systems
            if (
                self.helpers.is_hypervisor()
                or self.helpers.is_baremetal()
                or self.helpers.is_linux()
            ):
                metric_parser.add_argument(
                    "-u", "--usage", action="store_true", required=False, help=usage_help
                )
                metric_parser.add_argument(
                    "-p", "--power", action="store_true", required=False, help=power_help
                )
                metric_parser.add_argument(
                    "-c", "--clock", action="store_true", required=False, help=clock_help
                )
                metric_parser.add_argument(
                    "-t",
                    "--temperature",
                    action="store_true",
                    required=False,
                    help=temperature_help,
                )
                metric_parser.add_argument(
                    "-P", "--pcie", action="store_true", required=False, help=pcie_help
                )
                metric_parser.add_argument(
                    "-e", "--ecc", action="store_true", required=False, help=ecc_help
                )
                metric_parser.add_argument(
                    "-k", "--ecc-blocks", action="store_true", required=False, help=ecc_blocks_help
                )
                metric_parser.add_argument(
                    "-V", "--voltage", action="store_true", required=False, help=voltage_help
                )
                metric_parser.add_argument(
                    "-b",
                    "--base-board",
                    action="store_true",
                    required=False,
                    help=base_board_help,
                    default=False,
                )
                metric_parser.add_argument(
                    "-G",
                    "--gpu-board",
                    action="store_true",
                    required=False,
                    help=gpu_board_help,
                    default=False,
                )

            # Options that only apply to Hypervisors and Baremetal Linux
            if self.helpers.is_hypervisor() or (
                self.helpers.is_baremetal() and self.helpers.is_linux()
            ):
                pass

            # Optional Args for Linux Baremetal Systems
            if self.helpers.is_baremetal() and self.helpers.is_linux():
                metric_parser.add_argument(
                    "-f", "--fan", action="store_true", required=False, help=fan_help
                )
                metric_parser.add_argument(
                    "-C", "--voltage-curve", action="store_true", required=False, help=vc_help
                )
                metric_parser.add_argument(
                    "-o", "--overdrive", action="store_true", required=False, help=overdrive_help
                )
                metric_parser.add_argument(
                    "-l", "--perf-level", action="store_true", required=False, help=perf_level_help
                )
                metric_parser.add_argument(
                    "-x", "--xgmi-err", action="store_true", required=False, help=xgmi_err_help
                )
                metric_parser.add_argument(
                    "-E", "--energy", action="store_true", required=False, help=energy_help
                )
                metric_parser.add_argument(
                    "-v",
                    "--violation",
                    dest="throttle",
                    action="store_true",
                    required=False,
                    help=throttle_help,
                )
                metric_parser.add_argument(
                    "-T",
                    "--throttle",
                    dest="throttle",
                    action="store_true",
                    required=False,
                    help=argparse.SUPPRESS,
                )
                metric_parser.add_argument(
                    "-X", "--partition", action="store_true", required=False, help=partition_help
                )

            # Options to only display to Hypervisors
            # Need to resolve the -G for guard, but technically should never intersect since it's VF only
            if self.helpers.is_hypervisor():
                metric_parser.add_argument(
                    "-s", "--schedule", action="store_true", required=False, help=schedule_help
                )
                metric_parser.add_argument(
                    "-G", "--guard", action="store_true", required=False, help=guard_help
                )
                metric_parser.add_argument(
                    "-u", "--guest-data", action="store_true", required=False, help=guest_data_help
                )
                metric_parser.add_argument(
                    "-f", "--fb-usage", action="store_true", required=False, help=fb_usage_help
                )
                metric_parser.add_argument(
                    "-m", "--xgmi", action="store_true", required=False, help=xgmi_help
                )

        if self.helpers.is_amd_hsmp_initialized():
            # Optional Args for CPUs
            cpu_group = metric_parser.add_argument_group(_T("CPU Arguments"))
            cpu_group.add_argument(
                "--cpu-power-metrics",
                action="store_true",
                required=False,
                help=cpu_power_metrics_help,
            )
            cpu_group.add_argument(
                "--cpu-prochot", action="store_true", required=False, help=cpu_proc_help
            )
            cpu_group.add_argument(
                "--cpu-freq-metrics", action="store_true", required=False, help=cpu_freq_help
            )
            cpu_group.add_argument(
                "--cpu-c0-res", action="store_true", required=False, help=cpu_c0_res_help
            )
            cpu_group.add_argument(
                "--cpu-lclk-dpm-level",
                action="append",
                required=False,
                type=self._not_negative_int,
                nargs=1,
                metavar=("NBIOID"),
                help=cpu_lclk_dpm_help,
            )
            cpu_group.add_argument(
                "--cpu-pwr-svi-telemetry-rails",
                action="store_true",
                required=False,
                help=cpu_pwr_svi_telemetry_rails_help,
            )
            cpu_group.add_argument(
                "--cpu-io-bandwidth",
                action="append",
                required=False,
                nargs=2,
                metavar=("IO_BW", "LINKID_NAME"),
                help=cpu_io_bandwidth_help,
            )
            cpu_group.add_argument(
                "--cpu-xgmi-bandwidth",
                action="append",
                required=False,
                nargs=2,
                metavar=("XGMI_BW", "LINKID_NAME"),
                help=cpu_xgmi_bandwidth_help,
            )
            cpu_group.add_argument(
                "--cpu-pwr-eff-mode",
                action="store_true",
                required=False,
                help=cpu_pwr_eff_mode_help,
            )
            cpu_group.add_argument(
                "--cpu-metrics-ver", action="store_true", required=False, help=cpu_metrics_ver_help
            )
            cpu_group.add_argument(
                "--cpu-metrics-table",
                action="store_true",
                required=False,
                help=cpu_metrics_table_help,
            )
            cpu_group.add_argument(
                "--cpu-socket-energy",
                action="store_true",
                required=False,
                help=cpu_socket_energy_help,
            )
            cpu_group.add_argument(
                "--cpu-ddr-bandwidth",
                action="store_true",
                required=False,
                help=cpu_ddr_bandwidth_help,
            )
            cpu_group.add_argument(
                "--cpu-temp", action="store_true", required=False, help=cpu_temp_help
            )
            cpu_group.add_argument(
                "--cpu-dimm-temp-range-rate",
                action="append",
                required=False,
                type=lambda x: int(x, 0),
                nargs=1,
                metavar=("DIMM_ADDR"),
                help=cpu_dimm_temp_range_rate_help,
            )
            cpu_group.add_argument(
                "--cpu-dimm-pow-consumption",
                action="append",
                required=False,
                type=lambda x: int(x, 0),
                nargs=1,
                metavar=("DIMM_ADDR"),
                help=cpu_dimm_pow_consumption_help,
            )
            cpu_group.add_argument(
                "--cpu-dimm-thermal-sensor",
                action="append",
                required=False,
                type=lambda x: int(x, 0),
                nargs=1,
                metavar=("DIMM_ADDR"),
                help=cpu_dimm_thermal_sensor_help,
            )
            cpu_group.add_argument(
                "--cpu-xgmi-pstate-range",
                action="store_true",
                required=False,
                help=cpu_xgmi_pstate_range_help,
            )
            cpu_group.add_argument(
                "--cpu-railisofreq-policy",
                action="store_true",
                required=False,
                help=cpu_railisofreq_policy_help,
            )
            cpu_group.add_argument(
                "--cpu-dfcstate-ctrl",
                action="store_true",
                required=False,
                help=cpu_dfcstate_ctrl_help,
            )
            cpu_group.add_argument(
                "--cpu-pc6-enable", action="store_true", required=False, help=cpu_pc6_enable_help
            )
            cpu_group.add_argument(
                "--cpu-cc6-enable", action="store_true", required=False, help=cpu_cc6_enable_help
            )
            cpu_group.add_argument(
                "--cpu-dimm-sb-reg",
                action="append",
                required=False,
                type=lambda x: int(x, 0),
                nargs=4,
                metavar=("DIMM_ADDR", "LID", "REG_OFFSET", "REG_SPACE"),
                help=cpu_dimm_sb_reg_help,
            )
            cpu_group.add_argument(
                "--cpu-tdelta", action="store_true", required=False, help=cpu_tdelta_help
            )
            cpu_group.add_argument(
                "--cpu-svi3-vr-controller-temp",
                action=self._cpu_svi3_vr_controller_temp_options(),
                required=False,
                type=int,
                nargs="+",
                metavar=("TYPE [RAIL_INDEX]"),
                help=cpu_svi3_vr_controller_temp_help,
            )
            cpu_group.add_argument(
                "--cpu-enabled-commands",
                action="store_true",
                required=False,
                help=cpu_enabled_commands_help,
            )
            cpu_group.add_argument(
                "--cpu-sdps-limit", action="store_true", required=False, help=cpu_sdps_limit_help
            )

            # Optional Args for CPU cores
            core_group = metric_parser.add_argument_group(_T("CPU Core Arguments"))
            core_group.add_argument(
                "--core-boost-limit",
                action="store_true",
                required=False,
                help=core_boost_limit_help,
            )
            core_group.add_argument(
                "--core-curr-active-freq-core-limit",
                action="store_true",
                required=False,
                help=core_curr_active_freq_core_limit_help,
            )
            core_group.add_argument(
                "--core-energy", action="store_true", required=False, help=core_energy_help
            )
            core_group.add_argument(
                "--core-ccd-power", action="store_true", required=False, help=core_ccd_power_help
            )
            core_group.add_argument(
                "--core-floor-limit",
                action="store_true",
                required=False,
                help=core_floor_limit_help,
            )
            core_group.add_argument(
                "--core-eff-floor-limit",
                action="store_true",
                required=False,
                help=core_eff_floor_limit_help,
            )

        # Add BRCM NIC/Switch Arguments
        if self.helpers.is_brcm_nic_initialized():
            metric_parser.add_argument(
                "--brcm_nic", action="store_true", required=False, help=nic_metric_help
            )
        if self.helpers.is_brcm_switch_initialized():
            metric_parser.add_argument(
                "--brcm_switch", action="store_true", required=False, help=switch_metric_help
            )

        # Add Universal Arguments & watch Args
        self._add_watch_arguments(metric_parser)
        self._add_device_arguments(metric_parser, required=False)
        command_modifier_group = self._add_command_modifiers(metric_parser)
        command_modifier_group.add_argument(
            "--show-unsupported", action="store_true", required=False, help=show_unsupported_help
        )

    def _add_process_parser(self, subparsers: argparse._SubParsersAction, func):
        if self.helpers.is_hypervisor():
            # Don't add this subparser on Hypervisors
            # This subparser is only available to Guest and Baremetal systems
            return

        if not self.helpers.is_amdgpu_initialized():
            # The process subcommand is currently only applicable to systems with amdgpu initialized
            return

        # Subparser help text
        process_help = _T("Lists compute process information running on the specified GPU")
        process_subcommand_help = _T(
            "{description}\n\nIf no GPU is specified, returns information for all GPUs on the system.                                \nIf no process argument is provided, all process information will be displayed."
        ).format(description=self.description)
        process_optionals_title = _T("Process arguments")

        # Optional Arguments help text
        general_help = _T("pid, process name, memory usage")
        engine_help = _T("All engine usages")
        pid_help = _T(
            "Gets compute process GPU information about the specified process based on Process ID"
        )
        name_help = _T(
            "Gets compute process GPU information about the specified process based on Process Name.\
                    \nIf multiple processes have the same name, information is returned for all of them.\
                    \nProcess Name may require elevated permissions."
        )

        # Create process subparser
        process_parser = subparsers.add_parser(
            "process", help=process_help, description=process_subcommand_help
        )
        process_parser._optionals.title = process_optionals_title
        process_parser.formatter_class = lambda prog: AMDSMISubparserHelpFormatter(prog)
        process_parser.set_defaults(func=func)

        # Optional Args
        process_parser.add_argument(
            "-G", "--general", action="store_true", required=False, help=general_help
        )
        process_parser.add_argument(
            "-e", "--engine", action="store_true", required=False, help=engine_help
        )
        process_parser.add_argument(
            "-p",
            "--pid",
            action="store",
            type=lambda value: self._not_negative_int(value, "--pid"),
            required=False,
            help=pid_help,
        )
        process_parser.add_argument(
            "-n",
            "--name",
            action="store",
            type=lambda value: self._is_valid_string(value, "--name"),
            required=False,
            help=name_help,
        )

        process_parser.add_argument(
            "--sort-by-pid",
            action="store_true",
            default=False,
            help=_T("Group process output by PID instead of GPU."),
        )

        # Add Universal Arguments & watch Args
        self._add_watch_arguments(process_parser)
        self._add_device_arguments(process_parser, required=False)
        self._add_command_modifiers(process_parser)

    def _add_profile_parser(self, subparsers: argparse._SubParsersAction, func):
        if not (self.helpers.is_windows() and self.helpers.is_hypervisor()):
            # This subparser only applies to Hypervisors
            return

        # Subparser help text
        profile_help = _T("Displays information about all profiles and current profile")
        profile_subcommand_help = _T(
            "{description}\n\nIf no GPU is specified, returns information for all GPUs on the system."
        ).format(description=self.description)
        profile_optionals_title = _T("Profile Arguments")

        # Create profile subparser
        profile_parser = subparsers.add_parser(
            "profile", help=profile_help, description=profile_subcommand_help
        )
        profile_parser._optionals.title = profile_optionals_title
        profile_parser.formatter_class = lambda prog: AMDSMISubparserHelpFormatter(prog)
        profile_parser.set_defaults(func=func)

        # Add Universal Arguments
        self._add_device_arguments(profile_parser, required=False)
        self._add_command_modifiers(profile_parser)

    def _add_event_parser(self, subparsers: argparse._SubParsersAction, func):
        if not self.helpers.is_amdgpu_initialized():
            # The event subcommand is only applicable to systems with amdgpu initialized
            return

        # Subparser help text
        event_help = _T("Displays event information for the given GPU")
        event_subcommand_help = _T(
            "{description}\n\nIf no GPU is specified, returns event information for all GPUs on the system."
        ).format(description=self.description)
        event_optionals_title = _T("Event Arguments")

        # Create event subparser
        event_parser = subparsers.add_parser(
            "event", help=event_help, description=event_subcommand_help
        )
        event_parser._optionals.title = event_optionals_title
        event_parser.formatter_class = lambda prog: AMDSMISubparserHelpFormatter(prog)
        event_parser.set_defaults(func=func)

        # Add Universal Arguments
        self._add_device_arguments(event_parser, required=False)
        self._add_command_modifiers(event_parser)

    def _add_topology_parser(self, subparsers: argparse._SubParsersAction, func):
        if not self.helpers.is_amdgpu_initialized():
            # The topology subcommand is only applicable to systems with amdgpu initialized
            return

        # Subparser help text
        topology_help = _T("Displays topology information of the devices")
        topology_subcommand_help = _T(
            "{description}\n\nIf no GPU is specified, returns information for all GPUs on the system.                                \nIf no topology argument is provided, all topology information will be displayed."
        ).format(description=self.description)
        topology_optionals_title = _T("Topology arguments")

        # Help text for Arguments only on Guest and BM platforms
        access_help = _T("Displays link accessibility between GPUs")
        weight_help = _T("Displays relative weight between GPUs")
        hops_help = _T("Displays the number of hops between GPUs")
        link_type_help = _T("Displays the link type between GPUs")
        numa_bw_help = _T("Display max and min bandwidth between nodes")
        coherent_help = _T("Display cache coherent (or non-coherent) link capability between nodes")
        atomics_help = _T("Display 32 and 64-bit atomic io link capability between nodes")
        dma_help = _T("Display P2P direct memory access (DMA) link capability between nodes")
        bi_dir_help = _T("Display P2P bi-directional link capability between nodes")
        nic_topo_help = _T("Display nic and gpu connectivity")
        nic_shownuma_help = _T("Display nic,gpu's numa and cpu affinity")

        # Create topology subparser
        topology_parser = subparsers.add_parser(
            "topology", help=topology_help, description=topology_subcommand_help
        )
        topology_parser._optionals.title = topology_optionals_title
        topology_parser.formatter_class = lambda prog: AMDSMISubparserHelpFormatter(prog)
        topology_parser.set_defaults(func=func)

        # Add Universal Arguments
        self._add_command_modifiers(topology_parser)
        self._add_device_arguments(topology_parser, required=False)

        # Optional Args
        topology_parser.add_argument(
            "-a", "--access", action="store_true", required=False, help=access_help
        )
        topology_parser.add_argument(
            "-w", "--weight", action="store_true", required=False, help=weight_help
        )
        topology_parser.add_argument(
            "-o", "--hops", action="store_true", required=False, help=hops_help
        )
        topology_parser.add_argument(
            "-t", "--link-type", action="store_true", required=False, help=link_type_help
        )
        topology_parser.add_argument(
            "-b", "--numa-bw", action="store_true", required=False, help=numa_bw_help
        )
        topology_parser.add_argument(
            "-c", "--coherent", action="store_true", required=False, help=coherent_help
        )
        topology_parser.add_argument(
            "-n", "--atomics", action="store_true", required=False, help=atomics_help
        )
        topology_parser.add_argument(
            "-d", "--dma", action="store_true", required=False, help=dma_help
        )
        topology_parser.add_argument(
            "-z", "--bi-dir", action="store_true", required=False, help=bi_dir_help
        )
        if self.helpers.is_brcm_nic_initialized():
            topology_parser.add_argument(
                "--nic_topo", action="store_true", required=False, help=nic_topo_help
            )
        if self.helpers.is_brcm_switch_initialized():
            topology_parser.add_argument(
                "--nic_switch", action="store_true", required=False, help=nic_shownuma_help
            )

    def _add_set_value_parser(self, subparsers: argparse._SubParsersAction, func):
        if not self.helpers.is_linux():
            # This subparser is only applicable to Linux
            return

        # Subparser help text
        set_value_help = _T("Set options for devices")
        set_value_subcommand_help = _T(
            "{description}\n\nIf no GPU is specified, will select all GPUs on the system.                                    \nA set argument must be provided; Multiple set arguments are accepted.                                    \nRequires 'sudo' privileges."
        ).format(description=self.description)
        set_value_optionals_title = _T("Set Arguments")

        # Help text for Arguments only on BM platforms
        if self.helpers.is_amdgpu_initialized():
            if self.helpers.is_baremetal():
                fan_support = self.helpers.get_fan_support()
                # Check if fan_support contains multi-line format (starts with newline)
                if fan_support.startswith("\n"):
                    # Multi-line format - don't wrap in parentheses
                    set_fan_help = _T("Set GPU fan speed :{fan_support}").format(
                        fan_support=fan_support
                    )
                else:
                    # Single line format - wrap in parentheses
                    set_fan_help = _T("Set GPU fan speed ({fan_support})").format(
                        fan_support=fan_support
                    )
                perf_level_help_choices_str = ", ".join(self.helpers.get_perf_levels()[0][0:-1])
                set_perf_level_help = _T(
                    "Set one of the following performance levels:\n\t{perf_level_help_choices_str}"
                ).format(perf_level_help_choices_str=perf_level_help_choices_str)
                power_profile_choices_str = ", ".join(self.helpers.get_power_profiles())
                set_profile_help = _T(
                    "Set power profile level (#) or choose one of available profiles:\n\t{power_profile_choices_str}"
                ).format(power_profile_choices_str=power_profile_choices_str)
                set_perf_det_help = _T(
                    "Enable performance determinism mode and set GFXCLK softmax limit (in MHz)"
                )
                (accelerator_set_choices, _) = self.helpers.get_accelerator_choices_types_indices()
                memory_partition_choices_str = ", ".join(self.helpers.get_memory_partition_types())
                accelerator_set_choices_str = ", ".join(accelerator_set_choices)
                set_compute_partition_help = _T(
                    "Set one of the following accelerator TYPE or profile INDEX:\n\t{accelerator_set_choices_str}.\n\tUse `sudo amd-smi partition --accelerator` to find acceptable values."
                ).format(accelerator_set_choices_str=accelerator_set_choices_str)
                set_compute_partition_hint = f"\nValid inputs are: {accelerator_set_choices_str}.\nUse `sudo amd-smi partition --accelerator` to find acceptable values."
                set_memory_partition_help = _T(
                    "Set one of the following the memory partition modes:\n\t{memory_partition_choices_str}"
                ).format(memory_partition_choices_str=memory_partition_choices_str)
                soc_pstate_help_info = ", ".join(self.helpers.get_soc_pstates())
                set_soc_pstate_help = _T(
                    "Set the GPU soc pstate policy using policy id, an integer. Valid id's include:\n\t{soc_pstate_help_info}"
                ).format(soc_pstate_help_info=soc_pstate_help_info)
                xgmi_plpd_help_info = ", ".join(self.helpers.get_xgmi_plpd_policies())
                set_xgmi_plpd_help = _T(
                    "Set the GPU XGMI per-link power down policy using policy id, an integer. Valid id's include:\n\t{xgmi_plpd_help_info}"
                ).format(xgmi_plpd_help_info=xgmi_plpd_help_info)
                set_clock_freq_help = _T(
                    "Set one or more sclk (aka gfxclk), mclk, fclk, pcie, or socclk frequency levels.\n\tUse `amd-smi static --clock` to find acceptable levels.\n\tUse `amd-smi static --bus` to find acceptable pcie levels."
                )
                set_ptl_status_help = _T(
                    "Enable or disable the PTL on a GPU processor:\n    0 for disable and 1 for enable."
                )
                ptl_format_help_choices_str = ", ".join(self.helpers.get_ptl_values()[0][0:-1])
                set_ptl_format_help = _T(
                    "Set the PTL format on a GPU processor. For example, --ptl-format I8,F32\n\tSet to one of the following PTL formats: {ptl_format_help_choices_str}"
                ).format(ptl_format_help_choices_str=ptl_format_help_choices_str)
            ppt0_power_cap_min, ppt0_power_cap_max, ppt1_power_cap_min, ppt1_power_cap_max = (
                self.helpers.get_power_caps()
            )
            set_power_cap_help = _T(
                "Set either PPT0 or PPT1 power capacity limit:\n\tEx: `amd-smi set -o 1300 ppt0`\n\tPPT0 min cap: {ppt0_power_cap_min}, PPT0 max cap: {ppt0_power_cap_max}\n\tPPT1 min cap: {ppt1_power_cap_min}, PPT1 max cap: {ppt1_power_cap_max}"
            ).format(
                ppt0_power_cap_min=ppt0_power_cap_min,
                ppt0_power_cap_max=ppt0_power_cap_max,
                ppt1_power_cap_min=ppt1_power_cap_min,
                ppt1_power_cap_max=ppt1_power_cap_max,
            )
            set_clk_limit_help = _T(
                "Sets the sclk (aka gfxclk), mclk, or fclk minimum and maximum frequencies. \n\tex: amd-smi set -L (sclk | mclk | fclk) (min | max) value\n\tFor mclk and fclk ONLY, a max value is rounded down to the nearest selectable DPM level; sclk is honored exactly."
            )
            set_process_isolation_help = _T(
                "Enable or disable the GPU process isolation on a per partition basis:\n    0 for disable and 1 for enable.\n"
            )

        # Help text for CPU set options
        set_cpu_pwr_limit_help = _T(
            "Set power limit for the given socket. Input parameter is power limit value."
        )
        set_cpu_xgmi_link_width_help = _T(
            "Set min and max linkwidth. Input parameters are min and max link width values (MAX >= MIN, values {cpu_set_range_label})"
        ).format(cpu_set_range_label=_cpu_set_range_label(CPU_XGMI_LINK_WIDTH_RANGE))
        set_cpu_lclk_dpm_level_help = _T(
            "Sets the min and max dpm level on a given NBIO.        \n Input parameters are die_index, min dpm, max dpm (MAX >= MIN, values {cpu_set_range_label})."
        ).format(cpu_set_range_label=_cpu_set_range_label(CPU_LCLK_DPM_LEVEL_RANGE))
        set_cpu_pwr_eff_mode_help = _T(
            "Sets the power efficiency mode policy. Input parameters,\
        \n MODE(0=HighPerformance, 1=PowerEfficiency, 2=IOPerformance, 3=BalancedMemory, 4=BalancedCore, 5=BalancedCoreMemory),\n For Family 1Ah Models 50h-57h onwards, UTIL(%%)(0-100) and PPT_limit (in mW) required if MODE= 4 or 5"
        )
        set_cpu_gmi3_link_width_help = _T(
            "Sets min and max gmi3 link width range (MAX >= MIN, values {cpu_set_range_label})"
        ).format(cpu_set_range_label=_cpu_set_range_label(CPU_GMI3_LINK_WIDTH_RANGE))
        set_cpu_pcie_link_rate_help = _T("Sets pcie link rate")
        set_cpu_df_pstate_range_help = _T("Sets min and max df-pstates (MAX <= MIN)")
        set_cpu_enable_apb_help = _T("Enables the DF p-state performance boost algorithm")
        set_cpu_disable_apb_help = _T(
            "Disables the DF p-state performance boost algorithm. Input parameter is DFPstate ({cpu_set_range_label})"
        ).format(cpu_set_range_label=_cpu_set_range_label(CPU_DISABLE_APB_RANGE))
        set_soc_boost_limit_help = _T(
            "Sets the boost limit for the given socket. Input parameter is socket BOOST_LIMIT value"
        )
        set_cpu_xgmi_pstate_range_help = _T(
            "Sets xgmi pstate range. Input parameters are min (0-1) and max (0-1), (MAX <= MIN)"
        )
        set_cpu_railisofreq_policy_help = _T(
            "Sets the CPU ISO frequency policy. Input parameter is value (0-1)"
        )
        set_cpu_dfcstate_ctrl_help = _T("Sets the DFCState control. Input parameter is value (0-1)")
        set_cpu_pc6_enable_help = _T("Sets PC6 enable control. Input parameter is value (0-1)")
        set_cpu_cc6_enable_help = _T("Sets CC6 enable control. Input parameter is value (0-1)")
        set_cpu_floor_limit_help = _T(
            "Sets the floor limit for the given CPU socket. Input parameter is CPU FLOOR_LIMIT value in MHz"
        )
        cpu_msr_floor_limit_help = _T(
            "Sets the CPU MSR floor limit frequency for the given socket. Input parameter is MSR_FLOOR_LIMIT value in MHz"
        )
        cpu_dimm_sb_reg_write_help = _T(
            "Write data to DIMM sideband register. Requires DIMM_ADDR, LID(0x2->TS0,0x6->TS1,0x9->PMIC0,0xA->SPDHub)\
        \n REG_OFFSET (hex), REG_SPACE (REGSPACE:0->Volatile,1->NVM), WRITE_DATA (hex)"
        )
        set_cpu_sdps_limit_help = _T(
            "Set CPU SDPS limit for the given socket. Input parameter is SDPS limit value in milliwatts (mW)."
        )

        # Help text for CPU Core set options
        set_core_boost_limit_help = _T(
            "Sets the boost limit for the given core. Input parameter is core BOOST_LIMIT value"
        )
        set_core_floor_limit_help = _T(
            "Sets the floor limit for the given core. Input parameter is core FLOOR_LIMIT value in MHz"
        )
        set_core_msr_floor_limit_help = _T(
            "Sets the MSR floor limit for the given core. Input parameter is core MSR_FLOOR_LIMIT value in MHz"
        )

        # Create set_value subparser
        set_value_parser = subparsers.add_parser(
            "set", help=set_value_help, description=set_value_subcommand_help
        )
        set_value_parser._optionals.title = set_value_optionals_title
        set_value_parser.formatter_class = lambda prog: AMDSMISubparserHelpFormatter(prog)
        set_value_parser.set_defaults(func=func)

        if self.helpers.is_amdgpu_initialized():
            # set value should only take one of these at a time so args below will be mutually exclusive
            set_value_exclusive_group = set_value_parser.add_mutually_exclusive_group()
            if self.helpers.is_baremetal():
                # Optional GPU Args
                set_value_exclusive_group.add_argument(
                    "-f",
                    "--fan",
                    action=self._validate_fan_speed(),
                    required=False,
                    help=set_fan_help,
                    metavar="%",
                )
                set_value_exclusive_group.add_argument(
                    "-l",
                    "--perf-level",
                    action="store",
                    choices=self.helpers.get_perf_levels()[0],
                    type=str.upper,
                    required=False,
                    help=set_perf_level_help,
                    metavar="LEVEL",
                )
                set_value_exclusive_group.add_argument(
                    "-P",
                    "--profile",
                    action="store",
                    choices=self.helpers.get_power_profiles(),
                    type=str.upper,
                    required=False,
                    help=set_profile_help,
                    metavar="PROFILE_LEVEL",
                )
                set_value_exclusive_group.add_argument(
                    "-d",
                    "--perf-determinism",
                    action="store",
                    type=lambda value: self._not_negative_int(value, "--perf-determinism"),
                    required=False,
                    help=set_perf_det_help,
                    metavar="SCLKMAX",
                )
                set_value_exclusive_group.add_argument(
                    "-C",
                    "--compute-partition",
                    "--accelerator-partition",
                    action="store",
                    choices=accelerator_set_choices,
                    type=lambda value: self._is_command_supported(
                        value, accelerator_set_choices, hint=set_compute_partition_hint
                    ),
                    required=False,
                    help=set_compute_partition_help,
                    metavar=("TYPE/INDEX"),
                )
                set_value_exclusive_group.add_argument(
                    "-M",
                    "--memory-partition",
                    action="store",
                    choices=self.helpers.get_memory_partition_types(),
                    type=str.upper,
                    required=False,
                    help=set_memory_partition_help,
                    metavar="PARTITION",
                )
                set_value_exclusive_group.add_argument(
                    "-a",
                    "--compute-partition-mem-alloc-mode",
                    action="store",
                    choices=["CAPPING", "ALL"],
                    type=str.upper,
                    required=False,
                    help=_T(
                        "Set compute partition memory allocation mode (CAPPING or ALL). Requires sudo."
                    ),
                    metavar="MODE",
                )
            # Power cap is enabled on guest, maintain order
            set_value_exclusive_group.add_argument(
                "-o",
                "--power-cap",
                action=self._power_cap_options(),
                nargs="+",
                required=False,
                help=set_power_cap_help,
                metavar=("WATTS", "[PWR_TYPE]"),
            )
            if self.helpers.is_baremetal():
                set_value_exclusive_group.add_argument(
                    "-p",
                    "--soc-pstate",
                    action="store",
                    required=False,
                    type=lambda value: self._not_negative_int(value, "--soc-pstate"),
                    help=set_soc_pstate_help,
                    metavar="POLICY_ID",
                )
                set_value_exclusive_group.add_argument(
                    "-x",
                    "--xgmi-plpd",
                    action="store",
                    required=False,
                    type=lambda value: self._not_negative_int(value, "--xgmi-plpd"),
                    help=set_xgmi_plpd_help,
                    metavar="POLICY_ID",
                )
                set_value_exclusive_group.add_argument(
                    "-c",
                    "--clk-level",
                    action=self._level_select(),
                    nargs="+",
                    required=False,
                    help=set_clock_freq_help,
                    metavar=("CLK_TYPE", "PERF_LEVELS"),
                )
                set_value_exclusive_group.add_argument(
                    "-S",
                    "--ptl-status",
                    action="store",
                    choices=[0, 1],
                    type=lambda value: self._not_negative_int(value, "--ptl-status"),
                    required=False,
                    help=set_ptl_status_help,
                    metavar=("STATUS"),
                )
                set_value_exclusive_group.add_argument(
                    "-F",
                    "--ptl-format",
                    action=self._validate_ptl_format(),
                    required=False,
                    help=set_ptl_format_help,
                    metavar=("FRMT1,FRMT2"),
                )

            set_value_exclusive_group.add_argument(
                "-L",
                "--clk-limit",
                action=self._limit_select(),
                nargs=3,
                required=False,
                help=set_clk_limit_help,
                metavar=("CLK_TYPE", "LIM_TYPE", "VALUE"),
            )
            set_value_exclusive_group.add_argument(
                "-R",
                "--process-isolation",
                action="store",
                choices=[0, 1],
                type=lambda value: self._not_negative_int(value, "--process-isolation"),
                required=False,
                help=set_process_isolation_help,
                metavar="STATUS",
            )

            if self.helpers.is_baremetal():
                set_mem_carveout_help = _T(
                    "Set VRAM carveout size by option index."
                    "\n\tUse `amd-smi static --mem-carveout` to see available options."
                    "\n\tOnly supported on some APUs; a reboot is required after setting."
                )
                set_value_exclusive_group.add_argument(
                    "-m",
                    "--mem-carveout",
                    action="store",
                    type=lambda value: self._not_negative_int(value, "--mem-carveout"),
                    required=False,
                    help=set_mem_carveout_help,
                    metavar="INDEX",
                )

                set_gtt_help = _T(
                    "Set GTT (shared GPU memory) size in GB.\n\tThis is a system-wide setting, not per-GPU."
                )
                set_value_exclusive_group.add_argument(
                    "-G",
                    "--gtt",
                    action="store",
                    type=lambda value: self._positive_float(value, "--gtt"),
                    required=False,
                    help=set_gtt_help,
                    metavar="GB",
                )

                set_ampp_activate_help = _T(
                    "Activate an AMPP (amdsmi power profile) by name."
                    "\n\tUse `amd-smi static --ampp` to see available profiles."
                )
                set_value_exclusive_group.add_argument(
                    "-A",
                    "--ampp-activate",
                    action="store",
                    type=self._valid_ampp_profile_name,
                    required=False,
                    help=set_ampp_activate_help,
                    metavar="PROFILE_NAME",
                )

                set_ampp_configure_help = _T(
                    "Configure an AMPP (amdsmi power profile) custom slot by staging one or "
                    "more KEY=VALUE fields, then committing them."
                    "\n\tUse `amd-smi static --ampp` to see writable profiles and field names."
                    "\n\tExample: --ampp-configure profile_2 PPT0_Limit=300"
                    "\n\tAlternatively, pass @<path> to restore every writable profile found "
                    "in a JSON file (as produced by `amd-smi static --ampp --json`)."
                    "\n\tExample: --ampp-configure @profiles.json"
                )
                set_value_exclusive_group.add_argument(
                    "--ampp-configure",
                    action=self._ampp_configure_options(),
                    nargs="+",
                    required=False,
                    help=set_ampp_configure_help,
                    metavar=("PROFILE_NAME", "KEY=VALUE"),
                )

                set_node_balancing_mode_help = _T(
                    "Set NPM balancing mode: POWER_BALANCING or FREQUENCY_BALANCING.\n\tThis is a system-wide setting, not per-GPU."
                )
                set_value_exclusive_group.add_argument(
                    "--node-balancing-mode",
                    action="store",
                    choices=["POWER_BALANCING", "FREQUENCY_BALANCING"],
                    type=str.upper,
                    required=False,
                    help=set_node_balancing_mode_help,
                    metavar="{POWER_BALANCING,FREQUENCY_BALANCING}",
                )

            # Node power limit is enabled on guest (1VF), maintain order
            max_node_power_limit = self.helpers.get_max_node_power_limit()
            set_node_power_limit_help = _T(
                "Set the node-level (NPM) power limit in watts.\n\tThis is a node-wide setting, not per-GPU.\n\tMax node power limit: {max_node_power_limit}"
            ).format(max_node_power_limit=max_node_power_limit)
            set_value_exclusive_group.add_argument(
                "-n",
                "--node-power-limit",
                action="store",
                type=lambda value: self._positive_int(value, "--node-power-limit"),
                required=False,
                help=set_node_power_limit_help,
                metavar="WATTS",
            )

        if self.helpers.is_amd_hsmp_initialized():
            if self.helpers.is_baremetal():
                # Optional CPU Args
                cpu_group = set_value_parser.add_argument_group(_T("CPU Arguments"))
                cpu_group.add_argument(
                    "--cpu-pwr-limit",
                    action="append",
                    required=False,
                    type=self._positive_int,
                    nargs=1,
                    metavar=("PWR_LIMIT"),
                    help=set_cpu_pwr_limit_help,
                )
                cpu_group.add_argument(
                    "--cpu-xgmi-link-width",
                    action="append",
                    required=False,
                    type=self._not_negative_int,
                    nargs=2,
                    choices=CPU_XGMI_LINK_WIDTH_RANGE,
                    metavar=("MIN_WIDTH", "MAX_WIDTH"),
                    help=set_cpu_xgmi_link_width_help,
                )
                cpu_group.add_argument(
                    "--cpu-lclk-dpm-level",
                    action="append",
                    required=False,
                    type=self._not_negative_int,
                    nargs=3,
                    choices=CPU_LCLK_DPM_LEVEL_RANGE,
                    metavar=("NBIOID", "MIN_DPM", "MAX_DPM"),
                    help=set_cpu_lclk_dpm_level_help,
                )
                cpu_group.add_argument(
                    "--cpu-pwr-eff-mode",
                    action=self._cpu_power_eff_mode_options(),
                    required=False,
                    type=self._not_negative_int,
                    nargs="+",
                    metavar="MODE [UTIL PPT_LIMIT]",
                    help=set_cpu_pwr_eff_mode_help,
                )
                cpu_group.add_argument(
                    "--cpu-gmi3-link-width",
                    action="append",
                    required=False,
                    type=self._not_negative_int,
                    nargs=2,
                    choices=CPU_GMI3_LINK_WIDTH_RANGE,
                    metavar=("MIN_LW", "MAX_LW"),
                    help=set_cpu_gmi3_link_width_help,
                )
                cpu_group.add_argument(
                    "--cpu-pcie-link-rate",
                    action="append",
                    required=False,
                    type=self._not_negative_int,
                    nargs=1,
                    metavar=("LINK_RATE"),
                    help=set_cpu_pcie_link_rate_help,
                )
                cpu_group.add_argument(
                    "--cpu-df-pstate-range",
                    action="append",
                    required=False,
                    type=self._not_negative_int,
                    nargs=2,
                    metavar=("MIN_PSTATE", "MAX_PSTATE"),
                    help=set_cpu_df_pstate_range_help,
                )
                cpu_group.add_argument(
                    "--cpu-enable-apb",
                    action="store_true",
                    required=False,
                    help=set_cpu_enable_apb_help,
                )
                cpu_group.add_argument(
                    "--cpu-disable-apb",
                    action="append",
                    required=False,
                    type=self._not_negative_int,
                    nargs=1,
                    choices=CPU_DISABLE_APB_RANGE,
                    metavar=("DF_PSTATE"),
                    help=set_cpu_disable_apb_help,
                )
                cpu_group.add_argument(
                    "--soc-boost-limit",
                    action="append",
                    required=False,
                    type=self._positive_int,
                    nargs=1,
                    metavar=("BOOST_LIMIT"),
                    help=set_soc_boost_limit_help,
                )
                cpu_group.add_argument(
                    "--cpu-xgmi-pstate-range",
                    action="append",
                    required=False,
                    type=self._not_negative_int,
                    nargs=2,
                    metavar=("MIN_PSTATE", "MAX_PSTATE"),
                    help=set_cpu_xgmi_pstate_range_help,
                )
                cpu_group.add_argument(
                    "--cpu-railisofreq-policy",
                    action="append",
                    required=False,
                    type=self._not_negative_int,
                    nargs=1,
                    metavar=("VALUE"),
                    help=set_cpu_railisofreq_policy_help,
                )
                cpu_group.add_argument(
                    "--cpu-dfcstate-ctrl",
                    action="append",
                    required=False,
                    type=self._not_negative_int,
                    nargs=1,
                    metavar=("VALUE"),
                    help=set_cpu_dfcstate_ctrl_help,
                )
                cpu_group.add_argument(
                    "--cpu-pc6-enable",
                    action="append",
                    required=False,
                    type=self._not_negative_int,
                    nargs=1,
                    metavar=("VALUE"),
                    help=set_cpu_pc6_enable_help,
                )
                cpu_group.add_argument(
                    "--cpu-cc6-enable",
                    action="append",
                    required=False,
                    type=self._not_negative_int,
                    nargs=1,
                    metavar=("VALUE"),
                    help=set_cpu_cc6_enable_help,
                )
                cpu_group.add_argument(
                    "--cpu-floor-limit",
                    action="append",
                    required=False,
                    type=self._positive_int,
                    nargs=1,
                    metavar=("FLOOR_LIMIT"),
                    help=set_cpu_floor_limit_help,
                )
                cpu_group.add_argument(
                    "--cpu-msr-floor-limit",
                    action="append",
                    required=False,
                    type=self._positive_int,
                    nargs=1,
                    metavar=("MSR_FLOOR_LIMIT"),
                    help=cpu_msr_floor_limit_help,
                )
                cpu_group.add_argument(
                    "--cpu-dimm-sb-reg",
                    action="append",
                    required=False,
                    type=lambda x: int(x, 0),
                    nargs=5,
                    metavar=("DIMM_ADDR", "LID", "REG_OFFSET", "REG_SPACE", "WRITE_DATA"),
                    help=cpu_dimm_sb_reg_write_help,
                )
                cpu_group.add_argument(
                    "--cpu-sdps-limit",
                    action="append",
                    required=False,
                    type=self._positive_int,
                    nargs=1,
                    metavar=("SDPS_LIMIT"),
                    help=set_cpu_sdps_limit_help,
                )
                # Optional CPU Core Args
                core_group = set_value_parser.add_argument_group(_T("CPU Core Arguments"))
                core_group.add_argument(
                    "--core-boost-limit",
                    action="append",
                    required=False,
                    type=self._positive_int,
                    nargs=1,
                    metavar=("BOOST_LIMIT"),
                    help=set_core_boost_limit_help,
                )
                core_group.add_argument(
                    "--core-floor-limit",
                    action="append",
                    required=False,
                    type=self._positive_int,
                    nargs=1,
                    metavar=("FLOOR_LIMIT"),
                    help=set_core_floor_limit_help,
                )
                core_group.add_argument(
                    "--core-msr-floor-limit",
                    action="append",
                    required=False,
                    type=self._positive_int,
                    nargs=1,
                    metavar=("MSR_FLOOR_LIMIT"),
                    help=set_core_msr_floor_limit_help,
                )

        # Reject --gtt combined with --gpu at the argparse level
        self._guard_gtt_gpu_conflict(set_value_parser, gtt_flags=("--gtt", "-G"))
        # Improve the --gpu error message if combined with --node-balancing-mode;
        # actual rejection happens at runtime in set_value.py, since these two
        # flags sit in separate argparse groups and argparse itself never
        # raises for this combination.
        self._guard_gtt_gpu_conflict(
            set_value_parser,
            gtt_flags=("--node-balancing-mode",),
            reason="--node-balancing-mode is a system-wide setting, not per-GPU",
        )
        # Improve the --gpu error message if combined with --node-power-limit;
        # actual rejection happens at runtime in set_value.py, since these two
        # flags sit in separate argparse groups and argparse itself never
        # raises for this combination.
        self._guard_gtt_gpu_conflict(
            set_value_parser,
            gtt_flags=("--node-power-limit", "-n"),
            reason="--node-power-limit is a node-wide setting, not per-GPU",
        )

        # Set accepts default devices of all
        self._add_device_arguments(set_value_parser, required=False)
        # Add Universal Arguments
        self._add_command_modifiers(set_value_parser)

    def _add_reset_parser(self, subparsers: argparse._SubParsersAction, func):
        if not self.helpers.is_linux():
            # This subparser is only applicable to Linux
            return

        if not self.helpers.is_amdgpu_initialized():
            # The reset subcommand is only applicable to systems with amdgpu initialized
            return

        # Subparser help text
        reset_help = _T("Reset options for devices")
        reset_subcommand_help = _T(
            "{description}\n\nIf no GPU is specified, will select all GPUs on the system.                                \nA reset argument must be provided; Multiple reset arguments are accepted.                                \nRequires 'sudo' privileges."
        ).format(description=self.description)
        reset_optionals_title = _T("Reset Arguments")

        # Help text for Arguments only on Guest and BM platforms
        gpureset_help = _T("Reset the specified GPU")
        reset_clocks_help = _T("Reset clocks and overdrive to default")
        reset_fans_help = _T("Reset fans to automatic (driver) control")
        reset_profile_help = _T("Reset power profile back to default")
        reset_xgmierr_help = _T("Reset XGMI error counts")
        reset_perf_det_help = _T("Disable performance determinism")
        reset_power_cap_help = _T("Reset the PPT0 and PPT1 power capacity limit to max capable")
        reset_gpu_clean_local_data_help = _T(
            "Clean up local data in LDS/GPRs on a per partition basis"
        )

        # Create reset subparser
        reset_parser = subparsers.add_parser(
            "reset", help=reset_help, description=reset_subcommand_help
        )
        reset_parser._optionals.title = reset_optionals_title
        reset_parser.formatter_class = lambda prog: AMDSMISubparserHelpFormatter(prog)
        reset_parser.set_defaults(func=func)

        # make reset args mutually exclusive
        reset_exclusive_group = reset_parser.add_mutually_exclusive_group()

        if self.helpers.is_baremetal():
            # Add Baremetal reset arguments
            reset_exclusive_group.add_argument(
                "-G", "--gpureset", action="store_true", required=False, help=gpureset_help
            )
            reset_exclusive_group.add_argument(
                "-c", "--clocks", action="store_true", required=False, help=reset_clocks_help
            )
            reset_exclusive_group.add_argument(
                "-f", "--fans", action="store_true", required=False, help=reset_fans_help
            )
            reset_exclusive_group.add_argument(
                "-p", "--profile", action="store_true", required=False, help=reset_profile_help
            )
            reset_exclusive_group.add_argument(
                "-x", "--xgmierr", action="store_true", required=False, help=reset_xgmierr_help
            )
            reset_exclusive_group.add_argument(
                "-d",
                "--perf-determinism",
                action="store_true",
                required=False,
                help=reset_perf_det_help,
            )
            reset_exclusive_group.add_argument(
                "-o", "--power-cap", action="store_true", required=False, help=reset_power_cap_help
            )

            reset_gtt_help = _T("Reset GTT (shared GPU memory) to system default")
            reset_exclusive_group.add_argument(
                "--gtt", action="store_true", required=False, help=reset_gtt_help
            )

        # Add Baremetal and Virtual OS reset arguments
        reset_exclusive_group.add_argument(
            "-l",
            "--clean-local-data",
            action="store_true",
            required=False,
            help=reset_gpu_clean_local_data_help,
        )

        # Reject --gtt combined with --gpu at the argparse level
        self._guard_gtt_gpu_conflict(reset_parser, gtt_flags=("--gtt",))

        # Reset accepts default devices of all
        self._add_device_arguments(reset_parser, required=False)
        # Add Universal Arguments
        self._add_command_modifiers(reset_parser)

    def _add_monitor_parser(self, subparsers: argparse._SubParsersAction, func):
        if not self.helpers.is_linux():
            # This subparser is only applicable to Linux
            return

        if not self.helpers.is_amdgpu_initialized():
            # The monitor subcommand is only applicable to systems with amdgpu initialized
            return

        # Subparser help text
        monitor_help = _T("Monitor metrics for target devices")
        monitor_subcommand_help = _T(
            "{description}\n\nMonitor a target device for the specified arguments.                                  \nIf no arguments are provided, all arguments will be enabled.                                  \nUse the watch arguments to run continuously."
        ).format(description=self.description)
        monitor_optionals_title = _T("Monitor Arguments")

        # Help text for Arguments only on Guest and BM platforms
        power_usage_help = _T("Monitor power usage and power cap in Watts")
        temperature_help = _T("Monitor temperature in Celsius")
        base_board_temps_help = _T("Monitor base board temperatures in Celsius")
        gpu_board_temps_help = _T("Monitor GPU board temperatures in Celsius")
        gfx_util_help = _T("Monitor graphics utilization (%%) and clock (MHz)")
        mem_util_help = _T("Monitor memory utilization (%%) and clock (MHz)")
        encoder_util_help = _T("Monitor encoder utilization (%%) and clock (MHz)")
        decoder_util_help = _T("Monitor decoder utilization (%%) and clock (MHz)")
        ecc_help = _T("Monitor ECC single bit, ECC double bit, and PCIe replay error counts")
        mem_usage_help = _T("Monitor memory usage in MB")
        pcie_bandwidth_help = _T("Monitor PCIe bandwidth in Mb/s")
        nic_monitor_help = _T("Broadcom NIC monitor attributes")
        switch_monitor_help = _T("Broadcom switch monitor attributes")
        process_help = _T(
            "Enable Process information table below monitor output;\n    Process Name may require elevated permissions"
        )
        violation_help = _T(
            "Monitor power and thermal violation status (%%);\n    Only available for MI300 or newer ASICs"
        )

        # Create monitor subparser
        monitor_parser = subparsers.add_parser(
            "monitor", help=monitor_help, description=monitor_subcommand_help, aliases=["dmon"]
        )
        monitor_parser._optionals.title = monitor_optionals_title
        monitor_parser.formatter_class = lambda prog: AMDSMISubparserHelpFormatter(prog)
        monitor_parser.set_defaults(func=func)

        # Add monitor arguments
        monitor_parser.add_argument(
            "-p", "--power-usage", action="store_true", required=False, help=power_usage_help
        )
        monitor_parser.add_argument(
            "-t", "--temperature", action="store_true", required=False, help=temperature_help
        )
        monitor_parser.add_argument(
            "-b",
            "--base-board-temps",
            action="store_true",
            required=False,
            help=base_board_temps_help,
        )
        monitor_parser.add_argument(
            "-o",
            "--gpu-board-temps",
            action="store_true",
            required=False,
            help=gpu_board_temps_help,
        )
        monitor_parser.add_argument(
            "-u", "--gfx", action="store_true", required=False, help=gfx_util_help
        )
        monitor_parser.add_argument(
            "-m", "--mem", action="store_true", required=False, help=mem_util_help
        )
        monitor_parser.add_argument(
            "-n", "--encoder", action="store_true", required=False, help=encoder_util_help
        )
        monitor_parser.add_argument(
            "-d", "--decoder", action="store_true", required=False, help=decoder_util_help
        )
        monitor_parser.add_argument(
            "-e", "--ecc", action="store_true", required=False, help=ecc_help
        )
        monitor_parser.add_argument(
            "-v", "--vram-usage", action="store_true", required=False, help=mem_usage_help
        )
        monitor_parser.add_argument(
            "-r", "--pcie", action="store_true", required=False, help=pcie_bandwidth_help
        )
        monitor_parser.add_argument(
            "-q", "--process", action="store_true", required=False, help=process_help
        )
        if self.helpers.is_brcm_nic_initialized():
            monitor_parser.add_argument(
                "--brcm_nic", action="store_true", required=False, help=nic_monitor_help
            )
        if self.helpers.is_brcm_switch_initialized():
            monitor_parser.add_argument(
                "--brcm_switch", action="store_true", required=False, help=switch_monitor_help
            )
        if not self.helpers.is_virtual_os():
            monitor_parser.add_argument(
                "-V", "--violation", action="store_true", required=False, help=violation_help
            )

        monitor_parser.add_argument(
            "--sort-by-pid",
            action="store_true",
            default=False,
            help=_T(
                "Group process output by PID instead of GPU. Only applies when --process is used."
            ),
        )

        # Add Universal Arguments & Watch Args
        self._add_watch_arguments(monitor_parser)
        self._add_device_arguments(monitor_parser, required=False)
        self._add_command_modifiers(monitor_parser)

    def _add_xgmi_parser(self, subparsers: argparse._SubParsersAction, func):
        if not self.helpers.is_amdgpu_initialized():
            # The xgmi subcommand is only applicable to systems with amdgpu initialized
            return

        # Subparser help text
        xgmi_help = _T("Displays xgmi information of the devices")
        xgmi_subcommand_help = _T(
            "{description}\n\nIf no GPU is specified, returns information for all GPUs on the system.                                \nIf no xgmi argument is provided, all xgmi information will be displayed."
        ).format(description=self.description)
        xgmi_optionals_title = _T("XGMI arguments")

        # Help text for Arguments only on Guest and BM platforms
        metrics_help = _T("Metric XGMI information")
        xgmi_source_status_help = _T("Source GPU XGMI Link information")
        xgmi_link_status_help = _T("XGMI Link Status information")

        # Create xgmi subparser
        xgmi_parser = subparsers.add_parser(
            "xgmi", help=xgmi_help, description=xgmi_subcommand_help
        )
        xgmi_parser._optionals.title = xgmi_optionals_title
        xgmi_parser.formatter_class = lambda prog: AMDSMISubparserHelpFormatter(prog)
        xgmi_parser.set_defaults(func=func)

        # Optional Args
        xgmi_parser.add_argument(
            "-m", "--metric", action="store_true", required=False, help=metrics_help
        )
        xgmi_parser.add_argument(
            "-s",
            "--source-status",
            action="store_true",
            required=False,
            help=xgmi_source_status_help,
        )
        xgmi_parser.add_argument(
            "-l", "--link-status", action="store_true", required=False, help=xgmi_link_status_help
        )

        # Add Universal Arguments
        self._add_device_arguments(xgmi_parser, required=False)
        self._add_command_modifiers(xgmi_parser)

    def _add_partition_parser(self, subparsers: argparse._SubParsersAction, func):
        if not self.helpers.is_amdgpu_initialized():
            # The partition subcommand is only applicable to systems with amdgpu initialized
            return

        # Subparser help text
        partition_help = _T("Displays partition information of the devices")
        partition_subcommand_help = _T(
            "{description}\n\nIf no GPU is specified, returns information for all GPUs on the system.                                \nIf no partition argument is provided, all partition information will be displayed."
        ).format(description=self.description)
        partition_optionals_title = _T("Partition arguments")

        # Options help text
        current_help = _T("display the current partition information")
        memory_help = _T("display the current memory partition mode and capabilities")
        accelerator_help = _T("display accelerator partition information")

        # Create partition subparser
        partition_parser = subparsers.add_parser(
            "partition", help=partition_help, description=partition_subcommand_help
        )
        partition_parser._optionals.title = partition_optionals_title
        partition_parser.formatter_class = lambda prog: AMDSMISubparserHelpFormatter(prog)
        partition_parser.set_defaults(func=func)

        # Handle GPU Options
        partition_parser.add_argument(
            "-c", "--current", action="store_true", required=False, help=current_help
        )
        partition_parser.add_argument(
            "-m", "--memory", action="store_true", required=False, help=memory_help
        )
        partition_parser.add_argument(
            "-a", "--accelerator", action="store_true", required=False, help=accelerator_help
        )

        # Add Universal Arguments
        self._add_device_arguments(partition_parser, required=False)
        self._add_command_modifiers(partition_parser)

    def _add_ras_parser(self, subparsers: argparse._SubParsersAction, func):
        """
        Adds the 'ras' subcommand.

        Expected command:
            amd-smi ras --cper --severity=nonfatal-uncorrected,fatal --folder <folder_name> --file-limit=1000 --follow

        All parameters are provided via options; no positional arguments or optional --file/--gpu are used.
        """
        # Subparser help text
        ras_help = _T("Retrieve RAS (CPER) entries from the driver")
        ras_description = _T(
            "{description}\n\nRetrieve and decode RAS (CPER) entries from the kernel driver.\nSupports filtering by severity, exporting to different formats, and continuous monitoring.\nThis command accepts options only; no positional arguments are required."
        ).format(description=self.description)
        ras_optionals_title = _T("RAS arguments")

        # Help text for RAS arguments
        cper_help = _T("Trigger current CPER data retrieval")
        afid_help = _T("Generate an AFID (AMD Field ID) given a CPER record file or folder")
        severity_choices = ["nonfatal-uncorrected", "fatal", "nonfatal-corrected", "all"]
        severity_choices_str = ", ".join(severity_choices)
        severity_help = _T(
            "Set the SEVERITY filters from the following:\n    {severity_choices_str}"
        ).format(severity_choices_str=severity_choices_str)
        folder_help = _T(
            "With --cper: folder to dump current CPER report files (created if missing)."
            "\n    With --afid: existing folder of CPER records to decode."
        )
        file_limit_help = _T(
            "Maximum number of current CPER files in target folder\n    Older files beyond limit will be deleted"
        )
        cper_file_help = _T("Full path of a retrieved CPER record file to generate the AFID")
        follow_help = _T("Continuously monitor for new CPER entries")

        ras_parser = subparsers.add_parser("ras", help=ras_help, description=ras_description)
        ras_parser._optionals.title = ras_optionals_title
        ras_parser.formatter_class = lambda prog: AMDSMISubparserHelpFormatter(prog)
        ras_parser.set_defaults(func=func)

        # Create mutually exclusive command ras group (--cper or --afid)
        ras_exclusive_group = ras_parser.add_mutually_exclusive_group(required=True)
        ras_exclusive_group.add_argument("--cper", action="store_true", help=cper_help)
        ras_exclusive_group.add_argument("--afid", action="store_true", help=afid_help)

        # CPER Arguments remove defaults
        cper_group = ras_parser.add_argument_group(_T("CPER Arguments"))
        cper_group.add_argument(
            "--severity",
            type=str.lower,
            nargs="+",
            default=["all"],
            help=severity_help,
            choices=severity_choices,
            metavar="SEVERITY",
        )
        cper_group.add_argument("--folder", type=Path, help=folder_help)
        cper_group.add_argument(
            "--file-limit", type=self._positive_int, action="store", help=file_limit_help
        )
        cper_group.add_argument("--follow", action="store_true", help=follow_help)

        # AFID Arguments
        afid_group = ras_parser.add_argument_group(_T("AFID Arguments"))
        afid_group.add_argument(
            "--cper-file",
            action=self._check_cper_file_path(),
            metavar="CPER_FILE",
            help=cper_file_help,
        )

        # Add common modifiers and device selection arguments.
        self._add_device_arguments(ras_parser, required=False)
        self._add_command_modifiers(ras_parser)

    def _add_node_parser(self, subparsers: argparse._SubParsersAction, func):
        if self.helpers.is_virtual_os():
            # This subparser is not available to Guest and Hypervisor systems
            return

        # Subparser help text
        node_help = _T("Gets power and baseboard information for the node")
        node_subcommand_help = _T(
            "{description}\n\nReturns information for node 0 (OAM_ID 0) on the system.                                \nIf no node argument is provided, all node information except CUIDs will be displayed."
        ).format(description=self.description)
        node_optionals_title = _T("Node arguments")

        # Help text for Node arguments
        power_management_help = _T("Displays power management information")
        base_board_temps_help = _T("Displays baseboard temperatures")
        gtt_help = _T("Displays GTT (shared GPU memory) size")
        tray_help = _T("Displays compute tray type and accelerator count")
        node_cuid_help = _T(
            "Lists the CUID of every component on the node: the platform, CPU\n"
            "packages, GPUs, NPUs and NICs. Every CUID is temporary."
        )
        node_cuid_primary_help = _T(
            "Include each component's primary CUID, implying --cuid. Requires root."
        )

        node_parser = subparsers.add_parser(
            "node", help=node_help, description=node_subcommand_help
        )
        node_parser._optionals.title = node_optionals_title
        node_parser.formatter_class = lambda prog: AMDSMISubparserHelpFormatter(prog)
        node_parser.set_defaults(func=func)

        # Optional Args
        node_parser.add_argument(
            "-p",
            "--power-management",
            action="store_true",
            required=False,
            help=power_management_help,
        )
        node_parser.add_argument(
            "-b",
            "--base-board-temps",
            action="store_true",
            required=False,
            help=base_board_temps_help,
        )
        node_parser.add_argument("-G", "--gtt", action="store_true", required=False, help=gtt_help)
        node_parser.add_argument(
            "-T", "--tray", action="store_true", required=False, help=tray_help
        )
        node_parser.add_argument(
            "-Y", "--cuid", action="store_true", required=False, help=node_cuid_help
        )
        node_parser.add_argument(
            "-y", "--cuid-primary", action="store_true", required=False, help=node_cuid_primary_help
        )

        # Add Universal Arguments
        self._add_command_modifiers(node_parser)

    def _add_fabric_parser(self, subparsers: argparse._SubParsersAction, func):
        if not self.helpers.is_amdgpu_initialized():
            return

        # Subparser help text
        fabric_help = _T("Displays fabric (UALoE/UALink over Ethernet) information of the devices")
        fabric_subcommand_help = _T(
            "{description}\n\nIf no GPU is specified, returns information for all GPUs on the system.                                \nIf no fabric argument is provided, all fabric information will be displayed."
        ).format(description=self.description)
        fabric_optionals_title = _T("Fabric arguments")

        # Help text for arguments
        topology_help = _T(
            "Display fabric device configuration (BDF, bandwidth, latency, vPoD/pPoD, accelerator state)"
        )
        telemetry_help = _T(
            "Display fabric telemetry data (counters per category, instance, and item)"
        )

        # Create fabric subparser
        fabric_parser = subparsers.add_parser(
            "fabric", help=fabric_help, description=fabric_subcommand_help
        )
        fabric_parser._optionals.title = fabric_optionals_title
        fabric_parser.formatter_class = lambda prog: AMDSMISubparserHelpFormatter(prog)
        fabric_parser.set_defaults(func=func)

        # --topology changed meaning without failing to parse, so warn on explicit use.
        # The no-flag default that turns both flags on is not a user asking for it.
        class _WarnTopologySemanticsChange(argparse.Action):
            def __init__(self, option_strings, dest, **kwargs):
                super().__init__(option_strings, dest, nargs=0, **kwargs)

            def __call__(self, parser, args, values, option_string=None):
                print(
                    _T(
                        "amd-smi fabric: warning: --topology now reports fabric device "
                        "configuration; the counters it previously reported moved to --telemetry"
                    ),
                    file=sys.stderr,
                )
                setattr(args, self.dest, True)

        # Optional Args
        fabric_parser.add_argument(
            "-t",
            "--topology",
            action=_WarnTopologySemanticsChange,
            default=False,
            required=False,
            help=topology_help,
        )
        fabric_parser.add_argument(
            "-T", "--telemetry", action="store_true", required=False, help=telemetry_help
        )

        # Add Universal Arguments
        self._add_device_arguments(fabric_parser, required=False)
        self._add_command_modifiers(fabric_parser)

    def error(self, message):
        outputformat = self.helpers.get_output_format()

        if "argument : invalid choice: " in message:
            l = len("argument : invalid choice: ") + 1
            message = message[l:]
            message = message.split("'")[0]
            # Check if the command is possible in other system configurations and error accordingly
            if message in self.possible_commands:
                raise amdsmi_cli_exceptions.AmdSmiCommandNotSupportedException(
                    message, outputformat
                )
            raise amdsmi_cli_exceptions.AmdSmiInvalidCommandException(message, outputformat)
        elif "unrecognized arguments: " in message:
            l = len("unrecognized arguments: ")
            message = message[l:]
            raise amdsmi_cli_exceptions.AmdSmiInvalidParameterException(
                sys.argv[1], message, outputformat
            )
        elif ": invalid choice: " in message:
            # Named argument with invalid choice (e.g., argument --loglevel: invalid choice: ...)
            # or unquoted integer (e.g., argument --process-isolation: invalid choice: 2 ...)
            _after = message.split(": invalid choice: ")[1]
            value = _after.split("'")[1] if _after.startswith("'") else _after.split(" ")[0]
            hint = _after[_after.find(" (") :].rstrip() if " (" in _after else None
            command = sys.argv[1] if len(sys.argv) > 1 else ""
            raise amdsmi_cli_exceptions.AmdSmiInvalidParameterValueException(
                command, value, outputformat, hint=hint
            )
        elif ": expected" in message:
            # Named argument missing its value (e.g., argument --gpu: expected one argument)
            arg = (
                message[len("argument ") :].split(":")[0]
                if message.startswith("argument ")
                else message
            )
            raise amdsmi_cli_exceptions.AmdSmiMissingParameterValueException(arg, outputformat)
        else:
            command = sys.argv[1] if len(sys.argv) > 1 else ""
            raise amdsmi_cli_exceptions.AmdSmiInvalidParameterException(
                command, message, outputformat
            )
