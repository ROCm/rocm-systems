#!/usr/bin/env python3
#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import json
import logging
import math
import sys

from amdsmi_cli_exceptions import AmdSmiRequiredCommandException

from amdsmi import amdsmi_exception, amdsmi_interface
from amdsmi.amdsmi_interface import AMDSMI_MAX_PPT_LIMIT, AMDSMI_MAX_UTIL


class SetValueCommands:
    def set_core(
        self,
        args,
        multiple_devices=False,
        core=None,
        core_boost_limit=None,
        core_floor_limit=None,
        core_msr_floor_limit=None,
    ):
        """Issue set commands to target core(s)

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            core (device_handle, optional): device_handle for target device. Defaults to None.
            core_boost_limit (list, optional): Value override for args.core_boost_limit. Defaults to None.
            core_floor_limit (list, optional): Value override for args.core_floor_limit. Defaults to None.
            core_msr_floor_limit (list, optional): Value override for args.core_msr_floor_limit. Defaults to None.

        Raises:
            ValueError: Value error if no core value is provided
            IndexError: Index error if core list is empty

        Return:
            Nothing
        """
        if core:
            args.core = core
        if core_boost_limit:
            args.core_boost_limit = core_boost_limit
        if core_floor_limit:
            args.core_floor_limit = core_floor_limit
        if core_msr_floor_limit:
            args.core_msr_floor_limit = core_msr_floor_limit

        if args.core == None:
            raise ValueError("No Core provided, specific Core targets(S) are needed")

        # Handle multiple cores
        handled_multiple_cores, device_handle = self.helpers.handle_cores(
            args, self.logger, self.set_core
        )
        if handled_multiple_cores:
            return  # This function is recursive

        # Error if no subcommand args are passed
        if not any([args.core_boost_limit, args.core_floor_limit, args.core_msr_floor_limit]):
            command = " ".join(sys.argv[1:])
            raise AmdSmiRequiredCommandException(command, self.logger.format)

        args.core = device_handle
        # build core string for errors
        try:
            core_id = self.helpers.get_core_id_from_device_handle(args.core)
        except IndexError:
            core_id = f"ID Unavailable for {args.core}"

        static_dict = {}
        if args.core_boost_limit:
            static_dict["set_core_boost_limit"] = {}
            try:
                amdsmi_interface.amdsmi_set_cpu_core_boostlimit(
                    args.core, args.core_boost_limit[0][0]
                )
                # Verify the core boost limit is set
                boost_limit = amdsmi_interface.amdsmi_get_cpu_core_boostlimit(args.core)
                # Extract numeric value from response (remove units if present)
                if isinstance(boost_limit, str):
                    # Extract just the number part (assumes format like "5000 MHz" or "5000")
                    boost_limit = int(boost_limit.split()[0])
                else:
                    boost_limit = int(boost_limit)

                if boost_limit < args.core_boost_limit[0][0]:
                    static_dict["set_core_boost_limit"]["Response"] = (
                        f"Max allowed boostlimit is {boost_limit} MHz"
                    )
                elif boost_limit > args.core_boost_limit[0][0]:
                    static_dict["set_core_boost_limit"]["Response"] = (
                        f"Min allowed boostlimit is {boost_limit} MHz"
                    )
                else:
                    static_dict["set_core_boost_limit"]["Response"] = f"{boost_limit} MHz"
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["set_core_boost_limit"]["Response"] = (
                    f"Error occurred for Core {core_id} - {e.get_error_info()}"
                )
                logging.debug(
                    "Failed to set core boost limit for core %s | %s", core_id, e.get_error_info()
                )

        # Set core floor limit
        if args.core_floor_limit:
            static_dict["floor_limit"] = {}
            try:
                core_floor_limit = args.core_floor_limit[0][0]
                amdsmi_interface.amdsmi_set_cpu_core_floor_freq_limit(args.core, core_floor_limit)

                # Verify the core boost limit is set
                flimit = amdsmi_interface.amdsmi_get_cpu_core_floor_freq_limit(args.core)
                freq_range = amdsmi_interface.amdsmi_get_cpu_freq_range()
                fmax = freq_range["fmax"]
                fmin = freq_range["fmin"]

                if flimit < core_floor_limit:
                    if fmax and flimit != fmax:
                        static_dict["floor_limit"]["Response"] = (
                            f"Set, VALUE: {flimit} MHz, successful"
                        )
                    else:
                        static_dict["floor_limit"]["Response"] = (
                            f"Set, VALUE: {flimit} MHz, successful. Max allowed cpu floor limit is {flimit} MHz"
                        )
                elif flimit > core_floor_limit:
                    if fmin and flimit != fmin:
                        static_dict["floor_limit"]["Response"] = (
                            f"Set, VALUE: {flimit} MHz, successful"
                        )
                    else:
                        static_dict["floor_limit"]["Response"] = (
                            f"Set, VALUE: {flimit} MHz, successful. Min allowed cpu floor limit is {flimit} MHz"
                        )
                else:
                    static_dict["floor_limit"]["Response"] = f"Set, VALUE: {flimit} MHz, successful"
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["floor_limit"]["Response"] = (
                    f"Error occurred for Core {core_id} - {e.get_error_info()}"
                )
                logging.debug(
                    "Failed to set core floor limit for core %s | %s", core_id, e.get_error_info()
                )

        # Set core MSR floor limit
        if args.core_msr_floor_limit:
            static_dict["msr_floor_limit"] = {}
            try:
                msr_floor_limit = args.core_msr_floor_limit[0][0]
                amdsmi_interface.amdsmi_set_cpu_core_msr_floor_freq_limit(
                    args.core, msr_floor_limit
                )
                # Verify the core msr floor limit is set
                effflimit = amdsmi_interface.amdsmi_get_cpu_core_eff_floor_freq_limit(args.core)
                if effflimit == 0:
                    effflimit = msr_floor_limit
                freq_range = amdsmi_interface.amdsmi_get_cpu_freq_range()
                fmax = freq_range["fmax"]
                fmin = freq_range["fmin"]

                if effflimit < msr_floor_limit:
                    if fmax and effflimit != fmax:
                        static_dict["msr_floor_limit"]["Response"] = (
                            f"Set, VALUE: {effflimit} MHz, successful"
                        )
                    else:
                        static_dict["msr_floor_limit"]["Response"] = (
                            f"Set, VALUE: {effflimit} MHz, successful. Max allowed msr cpu floor limit is {effflimit} MHz"
                        )
                elif effflimit > msr_floor_limit:
                    if fmin and effflimit != fmin:
                        static_dict["msr_floor_limit"]["Response"] = (
                            f"Set, VALUE: {effflimit} MHz, successful"
                        )
                    else:
                        static_dict["msr_floor_limit"]["Response"] = (
                            f"Set, VALUE: {effflimit} MHz, successful. Min allowed msr cpu floor limit is {effflimit} MHz"
                        )
                else:
                    static_dict["msr_floor_limit"]["Response"] = (
                        f"Set, VALUE: {effflimit} MHz, successful"
                    )
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["msr_floor_limit"]["Response"] = (
                    f"Error occurred for Core {core_id} - {e.get_error_info()}"
                )
                logging.debug(
                    "Failed to set core MSR floor limit for core %s | %s",
                    core_id,
                    e.get_error_info(),
                )

        multiple_devices_csv_override = False
        self.logger.store_core_output(args.core, "values", static_dict)
        if multiple_devices:
            self.logger.store_multiple_device_output()
            return  # Skip printing when there are multiple devices
        self.logger.print_output(multiple_device_enabled=multiple_devices_csv_override)

    def set_cpu(
        self,
        args,
        multiple_devices=False,
        cpu=None,
        cpu_pwr_limit=None,
        cpu_xgmi_link_width=None,
        cpu_lclk_dpm_level=None,
        cpu_pwr_eff_mode=None,
        cpu_gmi3_link_width=None,
        cpu_pcie_link_rate=None,
        cpu_df_pstate_range=None,
        cpu_enable_apb=None,
        cpu_disable_apb=None,
        soc_boost_limit=None,
        cpu_xgmi_pstate_range=None,
        cpu_railisofreq_policy=None,
        cpu_dfcstate_ctrl=None,
        cpu_pc6_enable=None,
        cpu_cc6_enable=None,
        cpu_floor_limit=None,
        cpu_msr_floor_limit=None,
        cpu_dimm_sb_reg=None,
        cpu_sdps_limit=None,
    ):
        """Issue set commands to target cpu(s)

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            cpu (cpu_handle, optional): device_handle for target device. Defaults to None.
            cpu_pwr_limit (int, optional): Value override for args.cpu_pwr_limit. Defaults to None.
            cpu_xgmi_link_width (List[int], optional): Value override for args.cpu_xgmi_link_width. Defaults to None.
            cpu_lclk_dpm_level (List[int], optional): Value override for args.cpu_lclk_dpm_level. Defaults to None.
            cpu_pwr_eff_mode (List[int], optional): Value override for args.cpu_pwr_eff_mode [mode, util, ppt_limit]. Defaults to None.
            cpu_gmi3_link_width (List[int], optional): Value override for args.cpu_gmi3_link_width. Defaults to None.
            cpu_pcie_link_rate (int, optional): Value override for args.cpu_pcie_link_rate. Defaults to None.
            cpu_df_pstate_range (List[int], optional): Value override for args.cpu_df_pstate_range. Defaults to None.
            cpu_enable_apb (bool, optional): Value override for args.cpu_enable_apb. Defaults to None.
            cpu_disable_apb (int, optional): Value override for args.cpu_disable_apb. Defaults to None.
            soc_boost_limit (int, optional): Value override for args.soc_boost_limit. Defaults to None.
            cpu_xgmi_pstate_range (List[int], optional): Value override for args.cpu_xgmi_pstate_range. Defaults to None.
            cpu_railisofreq_policy (int, optional): Value override for args.cpu_railisofreq_policy. Defaults to None.
            cpu_dfcstate_ctrl (int, optional): Value override for args.cpu_dfcstate_ctrl. Defaults to None.
            cpu_pc6_enable (int, optional): Value override for args.cpu_pc6_enable. Defaults to None.
            cpu_cc6_enable (int, optional): Value override for args.cpu_cc6_enable. Defaults to None.
            cpu_floor_limit (int, optional): Value override for args.cpu_floor_limit. Defaults to None.
            cpu_msr_floor_limit (int, optional): Value override for args.cpu_msr_floor_limit. Defaults to None.
            cpu_dimm_sb_reg (list, optional): DIMM sideband register write parameters [dimm_addr, lid, reg_offset, reg_space, write_data] for write operation. Value override for args.cpu_dimm_sb_reg. Defaults to None.
            cpu_sdps_limit (int, optional): Value override for args.cpu_sdps_limit. Defaults to None.

        Raises:
            ValueError: Value error if no cpu value is provided
            IndexError: Index error if cpu list is empty

        Return:
            Nothing
        """
        if cpu:
            args.cpu = cpu
        if cpu_pwr_limit:
            args.cpu_pwr_limit = cpu_pwr_limit
        if cpu_xgmi_link_width:
            args.cpu_xgmi_link_width = cpu_xgmi_link_width
        if cpu_lclk_dpm_level:
            args.cpu_lclk_dpm_level = cpu_lclk_dpm_level
        if cpu_pwr_eff_mode:
            args.cpu_pwr_eff_mode = cpu_pwr_eff_mode
        if cpu_gmi3_link_width:
            args.cpu_gmi3_link_width = cpu_gmi3_link_width
        if cpu_pcie_link_rate:
            args.cpu_pcie_link_rate = cpu_pcie_link_rate
        if cpu_df_pstate_range:
            args.cpu_df_pstate_range = cpu_df_pstate_range
        if cpu_enable_apb:
            args.cpu_enable_apb = cpu_enable_apb
        if cpu_disable_apb:
            args.cpu_disable_apb = cpu_disable_apb
        if soc_boost_limit:
            args.soc_boost_limit = soc_boost_limit
        if cpu_xgmi_pstate_range:
            args.cpu_xgmi_pstate_range = cpu_xgmi_pstate_range
        if cpu_railisofreq_policy:
            args.cpu_railisofreq_policy = cpu_railisofreq_policy
        if cpu_dfcstate_ctrl:
            args.cpu_dfcstate_ctrl = cpu_dfcstate_ctrl
        if cpu_pc6_enable:
            args.cpu_pc6_enable = cpu_pc6_enable
        if cpu_cc6_enable:
            args.cpu_cc6_enable = cpu_cc6_enable
        if cpu_floor_limit:
            args.cpu_floor_limit = cpu_floor_limit
        if cpu_msr_floor_limit:
            args.cpu_msr_floor_limit = cpu_msr_floor_limit
        if cpu_dimm_sb_reg:
            args.cpu_dimm_sb_reg = cpu_dimm_sb_reg
        if cpu_sdps_limit:
            args.cpu_sdps_limit = cpu_sdps_limit

        if args.cpu == None:
            raise ValueError("No CPU provided, specific CPU targets(S) are needed")

        # Handle multiple CPU's
        handled_multiple_cpus, device_handle = self.helpers.handle_cpus(
            args, self.logger, self.set_cpu
        )
        if handled_multiple_cpus:
            return  # This function is recursive

        args.cpu = device_handle
        # Error if no subcommand args are passed
        if not any(
            [
                args.cpu_pwr_limit,
                args.cpu_xgmi_link_width,
                args.cpu_lclk_dpm_level,
                args.cpu_pwr_eff_mode,
                args.cpu_gmi3_link_width,
                args.cpu_pcie_link_rate,
                args.cpu_df_pstate_range,
                args.cpu_enable_apb,
                args.cpu_disable_apb,
                args.soc_boost_limit,
                args.cpu_xgmi_pstate_range,
                args.cpu_railisofreq_policy,
                args.cpu_dfcstate_ctrl,
                args.cpu_pc6_enable,
                args.cpu_cc6_enable,
                args.cpu_floor_limit,
                args.cpu_msr_floor_limit,
                args.cpu_dimm_sb_reg,
                args.cpu_sdps_limit,
            ]
        ):
            command = " ".join(sys.argv[1:])
            raise AmdSmiRequiredCommandException(command, self.logger.format)

        # Build CPU string for errors
        try:
            cpu_id = self.helpers.get_cpu_id_from_device_handle(args.cpu)
        except IndexError:
            cpu_id = f"ID Unavailable for {args.cpu}"

        static_dict = {}

        if args.cpu_pwr_limit:
            static_dict["set_pwr_limit"] = {}
            try:
                soc_max_pwr_limit = amdsmi_interface.amdsmi_get_cpu_socket_power_cap_max(args.cpu)
                soc_max_pwr_limit = self.helpers.convert_SI_unit(
                    float(soc_max_pwr_limit), self.helpers.SI_Unit.MILLI
                )
                max_power = int(soc_max_pwr_limit)
                if args.cpu_pwr_limit[0][0] > max_power:
                    args.cpu_pwr_limit[0][0] = max_power

                amdsmi_interface.amdsmi_set_cpu_socket_power_cap(args.cpu, args.cpu_pwr_limit[0][0])
                static_dict["set_pwr_limit"]["Response"] = (
                    f"{args.cpu_pwr_limit[0][0] / 1000:.3f} W"
                )
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["set_pwr_limit"]["Response"] = (
                    f"Error occurred for CPU {cpu_id} - {e.get_error_info()}"
                )
                logging.debug(
                    "Failed to set power limit for cpu %s | %s", cpu_id, e.get_error_info()
                )

        if args.cpu_xgmi_link_width:
            static_dict["set_xgmi_link_width"] = {}
            try:
                amdsmi_interface.amdsmi_set_cpu_xgmi_width(
                    args.cpu, args.cpu_xgmi_link_width[0][0], args.cpu_xgmi_link_width[0][1]
                )
                static_dict["set_xgmi_link_width"]["Response"] = (
                    f"{args.cpu_xgmi_link_width[0][0]} - {args.cpu_xgmi_link_width[0][1]}"
                )
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["set_xgmi_link_width"]["Response"] = (
                    f"Error occurred for CPU {cpu_id} - {e.get_error_info()}"
                )
                logging.debug(
                    "Failed to set xgmi link width for cpu %s | %s", cpu_id, e.get_error_info()
                )

        if args.cpu_lclk_dpm_level:
            static_dict["set_lclk_dpm_level"] = {}
            try:
                amdsmi_interface.amdsmi_set_cpu_socket_lclk_dpm_level(
                    args.cpu,
                    args.cpu_lclk_dpm_level[0][0],
                    args.cpu_lclk_dpm_level[0][1],
                    args.cpu_lclk_dpm_level[0][2],
                )
                static_dict["set_lclk_dpm_level"]["Response"] = (
                    f"NBIO[{args.cpu_lclk_dpm_level[0][0]}]"
                )
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["set_lclk_dpm_level"]["Response"] = (
                    f"Error occurred for CPU {cpu_id} - {e.get_error_info()}"
                )
                logging.debug(
                    "Failed to set lclk dpm level for cpu %s | %s", cpu_id, e.get_error_info()
                )

        if args.cpu_pwr_eff_mode:
            static_dict["pwr_eff_mode"] = {}
            try:
                mode = args.cpu_pwr_eff_mode[0][0]
                util = (
                    args.cpu_pwr_eff_mode[0][1]
                    if len(args.cpu_pwr_eff_mode[0]) > 1
                    else AMDSMI_MAX_UTIL
                )
                ppt_limit = (
                    args.cpu_pwr_eff_mode[0][2]
                    if len(args.cpu_pwr_eff_mode[0]) > 2
                    else AMDSMI_MAX_PPT_LIMIT
                )
                updated_util, updated_ppt_limit = (
                    amdsmi_interface.amdsmi_set_cpu_pwr_efficiency_mode(
                        args.cpu, mode, util, ppt_limit
                    )
                )

                # Always show mode
                static_dict["pwr_eff_mode"]["mode"] = f"{mode}"

                # Only show util and ppt_limit for modes 4 and 5
                if mode in [4, 5]:
                    ppt_limit_watts = updated_ppt_limit / 1000.0  # Convert milliwatts to watts
                    static_dict["pwr_eff_mode"]["util"] = f"{updated_util}%"
                    static_dict["pwr_eff_mode"]["ppt_limit"] = f"{ppt_limit_watts} W"
                else:
                    # For modes 0-3, util and ppt_limit are not displayed
                    pass
                static_dict["pwr_eff_mode"]["response"] = (
                    "Set power efficiency mode operation successful"
                )
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["pwr_eff_mode"]["response"] = (
                    f"Error occurred for CPU {cpu_id} - {e.get_error_info()}"
                )
                logging.debug(
                    "Failed to set power efficiency mode for cpu %s | %s",
                    cpu_id,
                    e.get_error_info(),
                )

        if args.cpu_gmi3_link_width:
            static_dict["set_gmi3_link_width"] = {}
            try:
                amdsmi_interface.amdsmi_set_cpu_gmi3_link_width_range(
                    args.cpu, args.cpu_gmi3_link_width[0][0], args.cpu_gmi3_link_width[0][1]
                )
                static_dict["set_gmi3_link_width"]["response"] = (
                    f"{args.cpu_gmi3_link_width[0][0]} - {args.cpu_gmi3_link_width[0][1]}"
                )
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["set_gmi3_link_width"]["response"] = (
                    f"Error occurred for CPU {cpu_id} - {e.get_error_info()}"
                )
                logging.debug(
                    "Failed to set gmi3 link width for cpu %s | %s", cpu_id, e.get_error_info()
                )

        if args.cpu_pcie_link_rate:
            static_dict["set_pcie_link_rate"] = {}
            try:
                resp = amdsmi_interface.amdsmi_set_cpu_pcie_link_rate(
                    args.cpu, args.cpu_pcie_link_rate[0][0]
                )
                static_dict["set_pcie_link_rate"]["prev_mode"] = resp
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["set_pcie_link_rate"]["prev_mode"] = (
                    f"Error occurred for CPU {cpu_id} - {e.get_error_info()}"
                )
                logging.debug(
                    "Failed to set pcie link rate for cpu %s | %s", cpu_id, e.get_error_info()
                )

        if args.cpu_df_pstate_range:
            static_dict["set_df_pstate_range"] = {}
            try:
                amdsmi_interface.amdsmi_set_cpu_df_pstate_range(
                    args.cpu, args.cpu_df_pstate_range[0][0], args.cpu_df_pstate_range[0][1]
                )
                static_dict["set_df_pstate_range"]["response"] = "Set Operation successful"
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["set_df_pstate_range"]["response"] = (
                    f"Error occurred for CPU {cpu_id} - {e.get_error_info()}"
                )
                logging.debug(
                    "Failed to set df pstate range for cpu %s | %s", cpu_id, e.get_error_info()
                )

        if args.cpu_enable_apb:
            static_dict["apbenable"] = {}
            try:
                amdsmi_interface.amdsmi_cpu_apb_enable(args.cpu)
                static_dict["apbenable"]["state"] = (
                    "Enabled DF - Pstate performance boost algorithm"
                )
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["apbenable"]["state"] = (
                    f"Error occurred for CPU {cpu_id} - {e.get_error_info()}"
                )
                logging.debug("Failed to enable APB for cpu %s | %s", cpu_id, e.get_error_info())

        if args.cpu_disable_apb:
            static_dict["apbdisable"] = {}
            try:
                amdsmi_interface.amdsmi_cpu_apb_disable(args.cpu, args.cpu_disable_apb[0][0])
                static_dict["apbdisable"]["state"] = (
                    "Disabled DF - Pstate performance boost algorithm"
                )
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["apbdisable"]["state"] = (
                    f"Error occurred for CPU {cpu_id} - {e.get_error_info()}"
                )
                logging.debug("Failed to enable APB for cpu %s | %s", cpu_id, e.get_error_info())

        if args.soc_boost_limit:
            static_dict["set_soc_boost_limit"] = {}
            try:
                amdsmi_interface.amdsmi_set_cpu_socket_boostlimit(
                    args.cpu, args.soc_boost_limit[0][0]
                )
                static_dict["set_soc_boost_limit"]["Response"] = "Set Operation successful"
            except amdsmi_exception.AmdSmiLibraryException as e:
                # static_dict["set_soc_boost_limit"]["Response"] = "N/A"
                static_dict["set_soc_boost_limit"]["Response"] = (
                    f"Error occurred for CPU {cpu_id} - {e.get_error_info()}"
                )
                logging.debug(
                    "Failed to set socket boost limit for cpu %s | %s", cpu_id, e.get_error_info()
                )

        if args.cpu_xgmi_pstate_range:
            static_dict["xgmi_pstate_range"] = {}
            try:
                amdsmi_interface.amdsmi_set_cpu_xgmi_pstate_range(
                    args.cpu, args.cpu_xgmi_pstate_range[0][0], args.cpu_xgmi_pstate_range[0][1]
                )
                static_dict["xgmi_pstate_range"]["response"] = (
                    f"Set, MIN_PSTATE: {args.cpu_xgmi_pstate_range[0][0]}, MAX_PSTATE: {args.cpu_xgmi_pstate_range[0][1]}, successful"
                )
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["xgmi_pstate_range"]["response"] = (
                    f"Error occurred for CPU {cpu_id} - {e.get_error_info()}"
                )
                logging.debug(
                    "Failed to set xgmi pstate range for cpu %s | %s", cpu_id, e.get_error_info()
                )

        if args.cpu_railisofreq_policy:
            static_dict["railisofreq_policy"] = {}
            try:
                resp = amdsmi_interface.amdsmi_set_cpu_rail_isofreq_policy(
                    args.cpu, args.cpu_railisofreq_policy[0][0]
                )
                static_dict["railisofreq_policy"]["response"] = f"Set, VALUE: {resp}, successful"
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["railisofreq_policy"]["response"] = (
                    f"Error occurred for CPU {cpu_id} - {e.get_error_info()}"
                )
                logging.debug(
                    "Failed to set ISO frequency policy for cpu %s | %s", cpu_id, e.get_error_info()
                )

        if args.cpu_dfcstate_ctrl:
            static_dict["dfcstate_ctrl"] = {}
            try:
                resp = amdsmi_interface.amdsmi_set_cpu_dfc_ctrl(
                    args.cpu, args.cpu_dfcstate_ctrl[0][0]
                )
                static_dict["dfcstate_ctrl"]["response"] = f"Set, VALUE: {resp}, successful"
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["dfcstate_ctrl"]["response"] = (
                    f"Error occurred for CPU {cpu_id} - {e.get_error_info()}"
                )
                logging.debug(
                    "Failed to set dfcstate control for cpu %s | %s", cpu_id, e.get_error_info()
                )

        if args.cpu_pc6_enable:
            static_dict["pc6_enable"] = {}
            try:
                amdsmi_interface.amdsmi_set_cpu_pc6_enable(args.cpu, args.cpu_pc6_enable[0][0])
                static_dict["pc6_enable"]["response"] = (
                    f"Set, VALUE: {args.cpu_pc6_enable[0][0]}, successful"
                )
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["pc6_enable"]["response"] = (
                    f"Error occurred for CPU {cpu_id} - {e.get_error_info()}"
                )
                logging.debug(
                    "Failed to set PC6 enable for cpu %s | %s", cpu_id, e.get_error_info()
                )

        if args.cpu_cc6_enable:
            static_dict["cc6_enable"] = {}
            try:
                amdsmi_interface.amdsmi_set_cpu_cc6_enable(args.cpu, args.cpu_cc6_enable[0][0])
                static_dict["cc6_enable"]["response"] = (
                    f"Set, VALUE: {args.cpu_cc6_enable[0][0]}, successful"
                )
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["cc6_enable"]["response"] = (
                    f"Error occurred for CPU {cpu_id} - {e.get_error_info()}"
                )
                logging.debug(
                    "Failed to set CC6 enable for cpu %s | %s", cpu_id, e.get_error_info()
                )

        if args.cpu_floor_limit:
            static_dict["floor_limit"] = {}
            try:
                cpu_floor_limit = args.cpu_floor_limit[0][0]
                amdsmi_interface.amdsmi_set_cpu_floor_freq_limit(args.cpu, cpu_floor_limit)

                # Verify the cpu floor limit is set
                flimit = amdsmi_interface.amdsmi_get_cpu_floor_freq_limit(args.cpu)
                freq_range = amdsmi_interface.amdsmi_get_cpu_freq_range()
                fmax = freq_range["fmax"]
                fmin = freq_range["fmin"]

                if flimit < cpu_floor_limit:
                    if fmax and flimit != fmax:
                        static_dict["floor_limit"]["Response"] = (
                            f"Set, VALUE: {flimit} MHz, successful"
                        )
                    else:
                        static_dict["floor_limit"]["Response"] = (
                            f"Set, VALUE: {flimit} MHz, successful. Max allowed cpu floor limit is {flimit} MHz"
                        )
                elif flimit > cpu_floor_limit:
                    if fmin and flimit != fmin:
                        static_dict["floor_limit"]["Response"] = (
                            f"Set, VALUE: {flimit} MHz, successful"
                        )
                    else:
                        static_dict["floor_limit"]["Response"] = (
                            f"Set, VALUE: {flimit} MHz, successful. Min allowed cpu floor limit is {flimit} MHz"
                        )
                else:
                    static_dict["floor_limit"]["Response"] = f"Set, VALUE: {flimit} MHz, successful"
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["floor_limit"]["Response"] = (
                    f"Error occurred for CPU {cpu_id} - {e.get_error_info()}"
                )
                logging.debug(
                    "Failed to set socket floor limit for cpu %s | %s", cpu_id, e.get_error_info()
                )

        if args.cpu_msr_floor_limit:
            static_dict["msr_floor_limit"] = {}
            try:
                msr_floor_limit = args.cpu_msr_floor_limit[0][0]
                amdsmi_interface.amdsmi_set_cpu_msr_floor_freq_limit(args.cpu, msr_floor_limit)

                # Verify the cpu msr floor limit is set
                effflimit = amdsmi_interface.amdsmi_get_cpu_eff_floor_freq_limit(args.cpu)
                if effflimit == 0:
                    effflimit = msr_floor_limit
                freq_range = amdsmi_interface.amdsmi_get_cpu_freq_range()
                fmax = freq_range["fmax"]
                fmin = freq_range["fmin"]

                if effflimit < msr_floor_limit:
                    if fmax and effflimit != fmax:
                        static_dict["msr_floor_limit"]["Response"] = (
                            f"Set, VALUE: {effflimit} MHz, successful"
                        )
                    else:
                        static_dict["msr_floor_limit"]["Response"] = (
                            f"Set, VALUE: {effflimit} MHz, successful. Max allowed msr cpu floor limit is {effflimit} MHz"
                        )
                elif effflimit > msr_floor_limit:
                    if fmin and effflimit != fmin:
                        static_dict["msr_floor_limit"]["Response"] = (
                            f"Set, VALUE: {effflimit} MHz, successful"
                        )
                    else:
                        static_dict["msr_floor_limit"]["Response"] = (
                            f"Set, VALUE: {effflimit} MHz, successful. Min allowed msr cpu floor limit is {effflimit} MHz"
                        )
                else:
                    static_dict["msr_floor_limit"]["Response"] = (
                        f"Set, VALUE: {effflimit} MHz, successful"
                    )
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["msr_floor_limit"]["Response"] = (
                    f"Error occurred for CPU {cpu_id} - {e.get_error_info()}"
                )
                logging.debug(
                    "Failed to set CPU MSR floor limit for cpu %s | %s", cpu_id, e.get_error_info()
                )

        if args.cpu_dimm_sb_reg:
            static_dict["dimm_sb_reg"] = {}
            try:
                dimm_addr = args.cpu_dimm_sb_reg[0][0]
                lid = args.cpu_dimm_sb_reg[0][1]
                reg_offset = args.cpu_dimm_sb_reg[0][2]
                reg_space = args.cpu_dimm_sb_reg[0][3]
                write_data = args.cpu_dimm_sb_reg[0][4]
                amdsmi_interface.amdsmi_set_cpu_dimm_sb_reg(
                    args.cpu, dimm_addr, lid, reg_offset, reg_space, write_data
                )
                static_dict["dimm_sb_reg"]["DimmAddress"] = f"0x{dimm_addr:02X}"
                static_dict["dimm_sb_reg"]["Lid"] = f"0x{lid:02X}"
                static_dict["dimm_sb_reg"]["Offset"] = f"0x{reg_offset:04X}"
                static_dict["dimm_sb_reg"]["RegSpace"] = reg_space
                static_dict["dimm_sb_reg"]["Data"] = f"0x{write_data:08X}"
                static_dict["dimm_sb_reg"]["Response"] = (
                    "Set DIMM sideband register write operation successful"
                )
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["dimm_sb_reg"]["Response"] = (
                    f"Error occurred for CPU {cpu_id} - {e.get_error_info()}"
                )
                logging.debug(
                    "Failed to write DIMM sideband register for cpu %s | %s",
                    cpu_id,
                    e.get_error_info(),
                )

        if args.cpu_sdps_limit:
            static_dict["sdps_limit"] = {}
            try:
                amdsmi_interface.amdsmi_set_cpu_sdps_limit(args.cpu, args.cpu_sdps_limit[0][0])
                sdps_limit_watts = float(args.cpu_sdps_limit[0][0]) / 1000
                static_dict["sdps_limit"]["Response"] = (
                    f"Set, VALUE: {sdps_limit_watts:.3f} W, successful"
                )
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["sdps_limit"]["Response"] = (
                    f"Error occurred for CPU {cpu_id} - {e.get_error_info()}"
                )
                logging.debug(
                    "Failed to set socket SDPS limit for cpu %s | %s", cpu_id, e.get_error_info()
                )

        multiple_devices_csv_override = False
        self.logger.store_cpu_output(args.cpu, "values", static_dict)
        if multiple_devices:
            self.logger.store_multiple_device_output()
            return  # Skip printing when there are multiple devices
        self.logger.print_output(multiple_device_enabled=multiple_devices_csv_override)

    def set_gpu(
        self,
        args,
        multiple_devices=False,
        gpu=None,
        fan=None,
        perf_level=None,
        profile=None,
        perf_determinism=None,
        compute_partition=None,
        memory_partition=None,
        power_cap=None,
        soc_pstate=None,
        xgmi_plpd=None,
        process_isolation=None,
        clk_limit=None,
        clk_level=None,
        ptl_status=None,
        ptl_format=None,
        mem_carveout=None,
    ):
        """Issue reset commands to target gpu(s)

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            gpu (device_handle, optional): device_handle for target device. Defaults to None.
            fan (tuple, optional): Value override for args.fan as (value, is_percentage). Defaults to None.
            perf_level (amdsmi_interface.AmdSmiDevPerfLevel, optional): Value override for args.perf_level. Defaults to None.
            profile (bool, optional): Value override for args.profile. Defaults to None.
            perf_determinism (int, optional): Value override for args.perf_determinism. Defaults to None.
            compute_partition (amdsmi_interface.AmdSmiComputePartitionType, optional): Value override for args.compute_partition. Defaults to None.
            memory_partition (amdsmi_interface.AmdSmiMemoryPartitionType, optional): Value override for args.memory_partition. Defaults to None.
            power_cap (int, optional): Value override for args.power_cap. Defaults to None.
            soc_pstate (int, optional): Value override for args.soc_pstate. Defaults to None.
            xgmi_plpd (int, optional): Value override for args.xgmi_plpd. Defaults to None.
            process_isolation (int, optional): Value override for args.process_isolation. Defaults to None.
            ptl_status (int, optional): Value override for args.ptl_status. Defaults to None.
            ptl_format(string, optional): Value override for args.ptl_format. Defaults to None.
        Raises:
            ValueError: Value error if no gpu value is provided
            IndexError: Index error if gpu list is empty

        Return:
            Nothing
        """
        # Set args.* to passed in arguments
        if gpu:
            args.gpu = gpu
        if fan is not None:
            args.fan = fan
        if perf_level:
            args.perf_level = perf_level
        if profile:
            args.profile = profile
        if perf_determinism is not None:
            args.perf_determinism = perf_determinism
        if compute_partition:
            args.compute_partition = compute_partition
        if memory_partition:
            args.memory_partition = memory_partition
        if power_cap:
            args.power_cap = power_cap
        if soc_pstate:
            args.soc_pstate = soc_pstate
        if xgmi_plpd:
            args.xgmi_plpd = xgmi_plpd
        if process_isolation:
            args.process_isolation = process_isolation
        if clk_limit:
            args.clk_limit = clk_limit
        if clk_level:
            args.clk_level = clk_level
        if ptl_status:
            args.ptl_status = ptl_status
        if ptl_format:
            args.ptl_format = ptl_format
        if mem_carveout is not None:
            args.mem_carveout = mem_carveout

        # Handle No GPU passed
        if args.gpu == None:
            args.gpu = self.device_handles

        if not self.group_check_printed:
            self.helpers.check_required_groups()
            self.group_check_printed = True

        # Handle multiple GPUs
        handled_multiple_gpus, device_handle = self.helpers.handle_gpus(
            args, self.logger, self.set_gpu
        )
        if handled_multiple_gpus:
            return  # This function is recursive

        args.gpu = device_handle

        # Error if no subcommand args are passed
        if self.helpers.is_baremetal():
            if not any(
                [
                    getattr(args, "fan", None) is not None,
                    getattr(args, "perf_level", None) is not None,
                    getattr(args, "profile", None) is not None,
                    getattr(args, "compute_partition", None) is not None,
                    getattr(args, "memory_partition", None) is not None,
                    getattr(args, "perf_determinism", None) is not None,
                    getattr(args, "power_cap", None) is not None,
                    getattr(args, "soc_pstate", None) is not None,
                    getattr(args, "xgmi_plpd", None) is not None,
                    getattr(args, "clk_level", None) is not None,
                    getattr(args, "clk_limit", None) is not None,
                    getattr(args, "ptl_status", None) is not None,
                    getattr(args, "ptl_format", None) is not None,
                    getattr(args, "process_isolation", None) is not None,
                    getattr(args, "mem_carveout", None) is not None,
                ]
            ):
                command = " ".join(sys.argv[1:])
                raise AmdSmiRequiredCommandException(command, self.logger.format)
        else:
            if not any(
                [
                    getattr(args, "power_cap", None) is not None,
                    getattr(args, "clk_limit", None) is not None,
                    getattr(args, "process_isolation", None) is not None,
                ]
            ):
                command = " ".join(sys.argv[1:])
                raise AmdSmiRequiredCommandException(command, self.logger.format)

        # Build GPU string for errors
        try:
            gpu_bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(args.gpu)
        except amdsmi_exception.AmdSmiLibraryException:
            gpu_bdf = f"BDF Unavailable for {args.gpu}"
        try:
            gpu_id = self.helpers.get_gpu_id_from_device_handle(args.gpu)
        except IndexError:
            gpu_id = f"ID Unavailable for {args.gpu}"
        gpu_string = f"GPU ID: {gpu_id} BDF:{gpu_bdf}"

        # Handle args
        if self.helpers.is_baremetal():
            if isinstance(args.fan, tuple):
                # Parse input: args.fan is now (value, is_percentage) tuple from parser
                input_value, is_percentage = args.fan

                # Check if gpu_od interface is available
                has_gpu_od, gpu_od_path = self.helpers.detect_gpu_od(gpu_bdf)

                # Helper function for consistent error formatting
                def format_fan_error(message, include_driver_note=False):
                    error_msg = message
                    if include_driver_note:
                        error_msg += (
                            "\nNote: For Navi3x+ GPUs, ensure the amdgpu driver is loaded with"
                            " gpu_od enabled:\n  sudo modprobe amdgpu ppfeaturemask=0xfff7ffff"
                            "\nIf fan operations return 'Device or resource busy', disable"
                            " runtime PM:\n  echo on | sudo tee"
                            " /sys/class/drm/card<N>/device/power/control"
                        )
                    return error_msg

                # Convert based on interface type and input format
                if has_gpu_od:
                    # For gpu_od interface: read OD_RANGE dynamically using shared helper
                    od_min, od_max = self.helpers.parse_gpu_od_fan_range(gpu_od_path)
                    if od_min is None:
                        # Parsing failed - cannot proceed without valid range
                        result = format_fan_error(
                            f"Unable to read gpu_od OD_RANGE from {gpu_od_path}. Cannot set fan speed.",
                            include_driver_note=True,
                        )
                        self.logger.store_output(args.gpu, "fan", result)
                        self.logger.print_output()
                        self.logger.clear_multiple_devices_output()
                        return

                    od_range = od_max - od_min
                    if is_percentage:
                        # Convert percentage (0-100%) to hardware range (od_min-od_max)
                        hw_value = od_min + int((input_value / 100) * od_range)
                        fan_percentage = input_value
                    else:
                        # Direct hardware value
                        if od_min <= input_value <= od_max:
                            hw_value = input_value
                            fan_percentage = (
                                int(((input_value - od_min) / od_range) * 100)
                                if od_range > 0
                                else 0
                            )
                        else:
                            result = format_fan_error(
                                f"Invalid fan speed value {input_value} for gpu_od interface. Valid range: {od_min}-{od_max} or use percentage (0-100%)",
                                include_driver_note=True,
                            )
                            self.logger.store_output(args.gpu, "fan", result)
                            self.logger.print_output()
                            self.logger.clear_multiple_devices_output()
                            return
                else:
                    # For legacy hwmon: range 0-255
                    if is_percentage:
                        # Convert percentage (0-100%) to PWM (0-255) using ceiling rounding
                        hw_value = math.ceil((input_value / 100) * 255)
                        fan_percentage = input_value
                    else:
                        # Direct PWM value
                        if 0 <= input_value <= 255:
                            hw_value = input_value
                            fan_percentage = int(
                                (input_value / 255) * 100 // 1
                            )  # round down (aka floor) to nearest whole number
                        else:
                            result = f"Invalid fan speed value {input_value}. Valid range: 0-255 or use percentage (0-100%)"
                            self.logger.store_output(args.gpu, "fan", result)
                            self.logger.print_output()
                            self.logger.clear_multiple_devices_output()
                            return

                try:
                    amdsmi_interface.amdsmi_set_gpu_fan_speed(args.gpu, 0, hw_value)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        raise PermissionError("Command requires elevation") from e
                    result = format_fan_error(
                        f"[{e.get_error_info(detailed=False)}] Unable to set fan speed to {hw_value} RPM/PWM ({fan_percentage}%)",
                        include_driver_note=has_gpu_od,
                    )
                    self.logger.store_output(args.gpu, "fan", result)
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    return

                self.logger.store_output(
                    args.gpu,
                    "fan",
                    f"Successfully set fan speed to {hw_value} RPM/PWM ({fan_percentage}%)",
                )
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return
            if args.perf_level:
                perf_level = amdsmi_interface.AmdSmiDevPerfLevel[args.perf_level]
                try:
                    amdsmi_interface.amdsmi_set_gpu_perf_level(args.gpu, perf_level)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        raise PermissionError("Command requires elevation") from e
                    self.logger.store_output(
                        args.gpu,
                        "perflevel",
                        f"[{e.get_error_info(detailed=False)}] Unable to set performance level to {args.perf_level}",
                    )
                    perf_options = (
                        str(self.helpers.get_perf_levels()[0][0:-1])
                        .replace("[", "")
                        .replace("]", "")
                        .replace("'", "")
                        .replace(" ", "")
                    )
                    print(f"\nPerformance Level Options:\n\t{perf_options}\n")
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    return

                self.logger.store_output(
                    args.gpu, "perflevel", f"Successfully set performance level {args.perf_level}"
                )
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return
            if args.profile:
                try:
                    # Parse profile input (name or number)
                    profile_input = args.profile.upper()
                    name_mapping = self.helpers.get_power_profile_name_mapping()

                    if profile_input in name_mapping:
                        profile_mask = name_mapping[profile_input]
                    else:
                        # Invalid profile - show available ones
                        try:
                            profile_status = amdsmi_interface.amdsmi_get_gpu_power_profile_presets(
                                args.gpu, 0
                            )
                            available = self.helpers.parse_available_profiles(
                                profile_status["available_profiles"]
                            )
                            available_str = ", ".join(available)
                        except amdsmi_exception.AmdSmiLibraryException as e:
                            available_str = "Unable to fetch available profiles"
                            logging.debug(
                                f"Failed to fetch available profiles: {e.get_error_info()}"
                            )

                        self.logger.store_output(
                            args.gpu,
                            "profile",
                            f"Invalid profile: {args.profile}\n\nAvailable profiles: {available_str}",
                        )
                        self.logger.print_output()
                        self.logger.clear_multiple_devices_output()
                        return

                    # Set the profile
                    amdsmi_interface.amdsmi_set_gpu_power_profile(args.gpu, 0, profile_mask)

                    self.logger.store_output(
                        args.gpu, "profile", f"Successfully set power profile to {profile_input}"
                    )
                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        raise PermissionError("Command requires elevation") from e

                    # Get available profiles for error message
                    try:
                        profile_status = amdsmi_interface.amdsmi_get_gpu_power_profile_presets(
                            args.gpu, 0
                        )
                        available = self.helpers.parse_available_profiles(
                            profile_status["available_profiles"]
                        )
                        available_str = ", ".join(available)
                    except amdsmi_exception.AmdSmiLibraryException as get_error:
                        available_str = "Unable to fetch available profiles"
                        logging.debug(
                            f"Failed to fetch available profiles: {get_error.get_error_info()}"
                        )

                    error_msg = f"[{e.get_error_info(detailed=False)}] Unable to set power profile to {args.profile}"
                    self.logger.store_output(args.gpu, "profile", error_msg)
                    print(f"\nAvailable Power Profiles:\n\t{available_str}\n")
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    return

                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return
            if isinstance(args.perf_determinism, int):
                try:
                    amdsmi_interface.amdsmi_set_gpu_perf_determinism_mode(
                        args.gpu, args.perf_determinism
                    )
                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        raise PermissionError("Command requires elevation") from e
                    self.logger.store_output(
                        args.gpu,
                        "perfdeterminism",
                        f"[{e.get_error_info(detailed=False)}] Unable to enable performance determinism and set GFX clock frequency to {args.perf_determinism} MHz",
                    )
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    return

                self.logger.store_output(
                    args.gpu,
                    "perfdeterminism",
                    f"Successfully enabled performance determinism and set GFX clock frequency to {args.perf_determinism} MHz",
                )
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return
            if args.compute_partition:
                current_set_count = self.helpers.get_set_count()
                future_set_count = 0
                attempted_to_set = "N/A"
                user_requested_partition_args = "N/A"
                try:
                    (accelerator_set_choices, accelerator_profiles) = (
                        self.helpers.get_accelerator_choices_types_indices()
                    )
                    logging.debug(
                        "args.compute_partition: %s; Accelerator_set_choices: %s",
                        str(args.compute_partition),
                        str(json.dumps(accelerator_set_choices, indent=4)),
                    )
                    if args.compute_partition in accelerator_profiles["profile_types"]:
                        compute_partition = amdsmi_interface.AmdSmiComputePartitionType[
                            args.compute_partition
                        ]
                        index = accelerator_profiles["profile_types"].index(args.compute_partition)
                        attempted_to_set = f"Attempted to set accelerator partition to {args.compute_partition} (profile #{accelerator_profiles['profile_indices'][int(index)]}) on {gpu_string}"
                        user_requested_partition_args = f"{args.compute_partition} (profile #{accelerator_profiles['profile_indices'][int(index)]})"
                        amdsmi_interface.amdsmi_set_gpu_compute_partition(
                            args.gpu, compute_partition
                        )
                    elif args.compute_partition in accelerator_profiles["profile_indices"]:
                        compute_partition = int(args.compute_partition)
                        index = accelerator_profiles["profile_indices"].index(
                            args.compute_partition
                        )
                        attempted_to_set = f"Attempted to set accelerator partition to {accelerator_profiles['profile_types'][int(index)]} (profile #{args.compute_partition}) on {gpu_string}"
                        user_requested_partition_args = f"{accelerator_profiles['profile_types'][int(index)]} (profile #{args.compute_partition})"
                        amdsmi_interface.amdsmi_set_gpu_accelerator_partition_profile(
                            args.gpu, compute_partition
                        )
                    else:
                        raise ValueError(
                            f"Invalid accelerator configuration {args.compute_partition} on {gpu_string}"
                        )
                    self.helpers.increment_set_count()
                    future_set_count = self.helpers.get_set_count()
                    if current_set_count == future_set_count - 1:
                        self.logger.store_output(
                            args.gpu,
                            "accelerator_partition",
                            f"Successfully set accelerator partition to {user_requested_partition_args}",
                        )
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    return

                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        raise PermissionError("Command requires elevation") from e
                    elif (
                        e.get_error_code()
                        == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED
                    ):
                        self.helpers.increment_set_count()
                        future_set_count = self.helpers.get_set_count()
                        if current_set_count == future_set_count - 1:
                            out = f"[AMDSMI_STATUS_NOT_SUPPORTED] Unable to set compute partition to {user_requested_partition_args}"
                            self.logger.store_output(args.gpu, "accelerator_partition", out)
                    elif (
                        e.get_error_code()
                        == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_SETTING_UNAVAILABLE
                    ):
                        print(
                            f"\n{attempted_to_set}\n"
                            f"\n[AMDSMI_STATUS_SETTING_UNAVAILABLE] Please check amd-smi partition --memory --accelerator for available profiles.\n"
                            "Users may need to switch memory partition to another mode in order to enable the desired accelerator partition.\n"
                        )
                        raise ValueError(
                            f"[AMDSMI_STATUS_SETTING_UNAVAILABLE] Unable to set accelerator partition to {args.compute_partition} on {gpu_string}"
                        ) from e
                    else:
                        raise ValueError(
                            f"Unable to set accelerator partition to {args.compute_partition} on {gpu_string}"
                        ) from e
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    return
            if args.memory_partition:
                ####################################################################
                # Get current and available memory partition modes                 #
                # Info used if AMDSMI_STATUS_INVAL is caught & to set progress bar #
                ####################################################################
                self.helpers.increment_set_count()
                set_count = self.helpers.get_set_count()
                if set_count == 1:  # only show reload warning on 1st set
                    self.helpers.confirm_changing_memory_partition_gpu_reload_warning()
                try:
                    memory_dict = {"caps": "N/A", "current": "N/A"}
                    memory_partition_config = (
                        amdsmi_interface.amdsmi_get_gpu_memory_partition_config(args.gpu)
                    )
                    memory_dict["caps"] = (
                        str(memory_partition_config["partition_caps"])
                        .replace("]", "")
                        .replace("[", "")
                        .replace("'", "")
                        .replace(" ", "")
                    )
                    memory_dict["current"] = memory_partition_config["mp_mode"]
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug(
                        "Failed to get current memory partition for GPU %s | %s",
                        gpu_id,
                        e.get_error_info(),
                    )
                try:
                    memory_partition = amdsmi_interface.AmdSmiMemoryPartitionType[
                        args.memory_partition
                    ]
                    amdsmi_interface.amdsmi_set_gpu_memory_partition(args.gpu, memory_partition)
                    out = f"Successfully set memory partition to {args.memory_partition}, use `sudo modprobe -r amdgpu && sudo modprobe amdgpu` to reload driver"
                except amdsmi_exception.AmdSmiLibraryException as e:
                    out = f"[{e.get_error_info(detailed=False)}] Unable to set memory partition to {args.memory_partition}"
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        out = "[AMDSMI_STATUS_NO_PERM] Command requires elevation"
                        self.logger.store_output(args.gpu, "memory_partition", out)
                        self.logger.print_output()
                        self.logger.clear_multiple_devices_output()
                        raise PermissionError("Command requires elevation") from e
                    elif e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_INVAL:
                        print(f"Valid Memory partition Modes: {memory_dict['caps']}\n")
                        self.logger.store_output(args.gpu, "memory_partition", out)
                        self.logger.print_output()
                        self.logger.clear_multiple_devices_output()
                        return
                    else:
                        self.logger.store_output(args.gpu, "memory_partition", out)
                        self.logger.print_output()
                        self.logger.clear_multiple_devices_output()
                        return
                self.logger.store_output(args.gpu, "memory_partition", out)
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return
            if isinstance(args.soc_pstate, int):
                try:
                    amdsmi_interface.amdsmi_set_soc_pstate(args.gpu, args.soc_pstate)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        raise PermissionError("Command requires elevation") from e
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_INVAL:
                        soc_pstate_info = amdsmi_interface.amdsmi_get_soc_pstate(args.gpu)
                        policy_string = "N/A"
                        # Check if 'policies' key exists before accessing it
                        if "policies" in soc_pstate_info and soc_pstate_info["policies"]:
                            policy_string = ""
                            for policy in soc_pstate_info["policies"]:
                                policy_string += (
                                    f"{policy['policy_id']}: {policy['policy_description']}, "
                                )
                            policy_string = policy_string.rstrip(
                                ", "
                            )  # Remove trailing comma and space
                        print(f"Valid SOC P-State Policies: [{policy_string}]\n")
                    self.logger.store_output(
                        args.gpu,
                        "socpstate",
                        f"[{e.get_error_info(detailed=False)}] Unable to set soc pstate dpm policy to {args.soc_pstate}",
                    )
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    return
                self.logger.store_output(
                    args.gpu,
                    "socpstate",
                    f"Successfully set soc pstate dpm policy to {args.soc_pstate}",
                )
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return
            if isinstance(args.xgmi_plpd, int):
                try:
                    amdsmi_interface.amdsmi_set_xgmi_plpd(args.gpu, args.xgmi_plpd)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        raise PermissionError("Command requires elevation") from e
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_INVAL:
                        xgmi_plpd_info = amdsmi_interface.amdsmi_get_xgmi_plpd(args.gpu)
                        policy_string = "N/A"
                        # Check if 'policies' key exists before accessing it
                        if "policies" in xgmi_plpd_info and xgmi_plpd_info["policies"]:
                            policy_string = ""
                            for policy in xgmi_plpd_info["policies"]:
                                policy_string += (
                                    f"{policy['policy_id']}: {policy['policy_description']}, "
                                )
                            policy_string = policy_string.rstrip(
                                ", "
                            )  # Remove trailing comma and space
                        print(f"Valid XGMI PLPD Policies: [{policy_string}]\n")
                    self.logger.store_output(
                        args.gpu,
                        "xgmiplpd",
                        f"[{e.get_error_info(detailed=False)}] Unable to set XGMI per-link power down policy to {args.xgmi_plpd}",
                    )
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    return
                self.logger.store_output(
                    args.gpu,
                    "xgmiplpd",
                    f"Successfully set XGMI per-link power down policy to {args.xgmi_plpd}",
                )
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return
            if isinstance(args.clk_level, tuple):
                clk_type = args.clk_level.clk_type
                perf_levels = args.clk_level.perf_levels
                perf_levels_str = str(perf_levels).strip("[]").replace(" ", "")
                smi_clk_type_mapping = {
                    "sclk": amdsmi_interface.AmdSmiClkType.SYS,
                    "mclk": amdsmi_interface.AmdSmiClkType.MEM,
                    "pcie": amdsmi_interface.AmdSmiClkType.PCIE,
                    "fclk": amdsmi_interface.AmdSmiClkType.DF,
                    "socclk": amdsmi_interface.AmdSmiClkType.SOC,
                }
                results_clk_lvl = {
                    "perf_level": "Unable to set performance level to MANUAL",
                    "get_clock_freq": f"Unable to retrieve {clk_type} frequency levels",
                    "set_clock": f"Unable to set {clk_type} perf level(s) to {perf_levels_str}",
                }
                if clk_type not in smi_clk_type_mapping:
                    raise ValueError(
                        f"Invalid clock type {clk_type}. Valid options are: {', '.join(smi_clk_type_mapping.keys())}"
                    )

                # Set perf level to manual if not already set
                try:
                    amdsmi_interface.amdsmi_set_gpu_perf_level(
                        args.gpu, amdsmi_interface.AmdSmiDevPerfLevel.MANUAL
                    )
                    results_clk_lvl["perf_level"] = "Successfully set performance level to MANUAL"
                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        raise PermissionError("Command requires elevation") from e
                    results_clk_lvl["perf_level"] = (
                        f"[{e.get_error_info(detailed=False)}] Unable to set performance level to MANUAL"
                    )
                    self.logger.store_output(args.gpu, "clk_level", results_clk_lvl)
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    return

                if clk_type.lower() == "pcie":
                    # Get PCIe bandwidth levels
                    try:
                        pcie_bandwidth_levels = amdsmi_interface.amdsmi_get_gpu_pci_bandwidth(
                            args.gpu
                        )
                        num_supported = pcie_bandwidth_levels["transfer_rate"]["num_supported"]
                        results_clk_lvl["get_clock_freq"] = (
                            f"Successfully retrieved {clk_type} frequency levels"
                        )
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        results_clk_lvl["get_clock_freq"] = (
                            f"[{e.get_error_info(detailed=False)}] Unable to retrieve {clk_type} frequency levels"
                        )
                        self.logger.store_output(args.gpu, "clk_level", results_clk_lvl)
                        self.logger.print_output()
                        self.logger.clear_multiple_devices_output()
                        return
                else:
                    # Get clock frequency levels
                    try:
                        frequencies = amdsmi_interface.amdsmi_get_clk_freq(
                            args.gpu, smi_clk_type_mapping[clk_type]
                        )
                        num_supported = frequencies["num_supported"]
                        results_clk_lvl["get_clock_freq"] = (
                            f"Successfully retrieved {clk_type} frequency levels"
                        )
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        results_clk_lvl["get_clock_freq"] = (
                            f"[{e.get_error_info(detailed=False)}] Unable to retrieve {clk_type} frequency levels"
                        )
                        self.logger.store_output(args.gpu, "clk_level", results_clk_lvl)
                        self.logger.print_output()
                        self.logger.clear_multiple_devices_output()
                        return

                # Validate bandwidth bitmask
                freq_bitmask = 0
                invalid_levels = []
                for level in perf_levels:
                    if level < num_supported:
                        freq_bitmask |= 1 << level
                    else:
                        invalid_levels.append(level)

                if invalid_levels:
                    # Handle/report invalid levels
                    invalid_levels_str = str(invalid_levels).strip("[]").replace(" ", "")
                    valid_levels_str = f"Valid levels for {clk_type}: 0"
                    if num_supported > 1:
                        valid_levels_str = f"Valid levels for {clk_type}: 0-{num_supported - 1}"
                    print(f"\n{valid_levels_str}\n")
                    results_clk_lvl["set_clock"] = (
                        f"Invalid level(s) {invalid_levels_str} are not within the range of supported levels for {clk_type}"
                    )
                    self.logger.store_output(args.gpu, "clk_level", results_clk_lvl)
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    return
                else:
                    # Proceed with freq_bitmask
                    pass

                if clk_type.lower() == "pcie":
                    try:
                        amdsmi_interface.amdsmi_set_gpu_pci_bandwidth(args.gpu, freq_bitmask)
                        results_clk_lvl["set_clock"] = (
                            f"Successfully set {clk_type} perf level(s) to {perf_levels_str}"
                        )
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        if (
                            e.get_error_code()
                            == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM
                        ):
                            raise PermissionError("Command requires elevation") from e

                        results_clk_lvl["set_clock"] = (
                            f"[{e.get_error_info(detailed=False)}] Unable to set {clk_type} perf level(s) to {perf_levels_str}"
                        )
                        self.logger.store_output(args.gpu, "clk_level", results_clk_lvl)
                        self.logger.print_output()
                        self.logger.clear_multiple_devices_output()
                        return
                else:
                    # For non-pcie clocks
                    if clk_type in self.helpers.convert_clock_type:
                        clk_type_conversion = self.helpers.convert_clock_type[clk_type]
                    else:
                        clk_type_conversion = "N/A"

                    try:
                        amdsmi_interface.amdsmi_set_clk_freq(args.gpu, clk_type, freq_bitmask)
                        results_clk_lvl["set_clock"] = (
                            f"Successfully set {clk_type} perf level(s) to {perf_levels_str}"
                        )
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        if (
                            e.get_error_code()
                            == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM
                        ):
                            raise PermissionError("Command requires elevation") from e
                        results_clk_lvl["set_clock"] = (
                            f"[{e.get_error_info(detailed=False)}] Unable to set {clk_type} perf level(s) to {perf_levels_str}"
                        )
                        self.logger.store_output(args.gpu, "clk_level", results_clk_lvl)
                        self.logger.print_output()
                        self.logger.clear_multiple_devices_output()
                        return
                self.logger.store_output(args.gpu, "clk_level", results_clk_lvl)
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return
            if isinstance(args.ptl_status, int):
                status_string = "Enabled" if args.ptl_status else "Disabled"
                result = f"Requested PTL status to {status_string}"  # This should not print out
                try:  # Due to driver requirements, do NOT check current state. Set state regardless of current state.
                    amdsmi_interface.amdsmi_set_gpu_ptl_state(args.gpu, args.ptl_status)
                    result = f"Successfully set PTL state to {status_string}"
                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        raise PermissionError("Command requires elevation") from e
                    self.logger.store_output(
                        args.gpu,
                        "ptlstatus",
                        f"[{e.get_error_info(detailed=False)}] Unable to set ptl status to {args.ptl_status}",
                    )
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    return
                self.logger.store_output(args.gpu, "ptlstatus", result)
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return
            if isinstance(args.ptl_format, tuple):
                requested_fmt1_enum, requested_fmt2_enum = args.ptl_format
                requested_str = f"{requested_fmt1_enum.name},{requested_fmt2_enum.name}"

                result = f"Requested PTL status to {requested_str}"  # This should not print out
                try:
                    # Get current formats as ints
                    cur1_code, cur2_code = amdsmi_interface.amdsmi_get_gpu_ptl_formats(args.gpu)
                    cur1_enum = amdsmi_interface.AmdSmiPtlData(cur1_code)
                    cur2_enum = amdsmi_interface.AmdSmiPtlData(cur2_code)
                    current_str = f"{cur1_enum.name},{cur2_enum.name}"
                    if (cur1_enum, cur2_enum) == (requested_fmt1_enum, requested_fmt2_enum):
                        result = f"PTL format is already {current_str}"
                    else:
                        amdsmi_interface.amdsmi_set_gpu_ptl_formats(
                            args.gpu, requested_fmt1_enum, requested_fmt2_enum
                        )
                        result = f"Successfully set PTL format to {requested_str}"
                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        raise PermissionError("Command requires elevation") from e
                    self.logger.store_output(
                        args.gpu,
                        "ptlformat",
                        f"[{e.get_error_info(detailed=False)}] Unable to set PTL format to {requested_str}",
                    )
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    return
                self.logger.store_output(args.gpu, "ptlformat", result)
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return

        # Universal args
        if isinstance(args.power_cap, tuple):
            pwr_type = args.power_cap.pwr_type
            requested_power_cap = args.power_cap.watts

            # If pwr_type is None, default to ppt0 (legacy behavior)
            if pwr_type is None:
                pwr_type = "ppt0"
                pwr_type_as_int = 0
            else:
                pwr_type_as_int = 0 if pwr_type == "ppt0" else 1

            # Set the power cap for the specified sensor
            pwr_type_upper = pwr_type.upper()
            result = self.helpers.validate_and_set_power_cap(
                args.gpu, pwr_type_as_int, pwr_type_upper, requested_power_cap, self.logger
            )
            self.logger.store_output(args.gpu, "powercap", result)
            if multiple_devices:
                self.logger.store_multiple_device_output()
                return  # Skip printing when there are multiple devices
            self.logger.print_output()
            self.logger.clear_multiple_devices_output()
            return
        if isinstance(args.clk_limit, tuple):
            clk_type = args.clk_limit.clk_type
            lim_type = args.clk_limit.lim_type
            val = args.clk_limit.val
            val_changed = True  # Assume Clock limit value is changed

            # Validate the value against the extremum
            try:
                # Parser only allows three options sclk, mclk or fclk
                if clk_type == "sclk":
                    amdsmi_clk_type = amdsmi_interface.AmdSmiClkType.GFX
                elif clk_type == "mclk":
                    amdsmi_clk_type = amdsmi_interface.AmdSmiClkType.MEM
                elif clk_type == "fclk":
                    amdsmi_clk_type = amdsmi_interface.AmdSmiClkType.DF
                else:
                    print("Valid clock types are: sclk, mclk, fclk\n")
                    self.logger.store_output(
                        args.gpu, "clk_limit", f"Invalid clock type {args.clk_limit.clk_type}"
                    )
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    return
                clk_tuple = amdsmi_interface.amdsmi_get_clock_info(args.gpu, amdsmi_clk_type)

                if lim_type == "min":
                    amdsmi_lim_type = amdsmi_interface.AmdSmiClkLimitType.MIN
                    if val > clk_tuple["max_clk"]:
                        self.logger.store_output(
                            args.gpu,
                            "clk_limit",
                            f"Cannot set {args.clk_limit.clk_type} min value greater than max ({clk_tuple['max_clk']}MHz)",
                        )
                        self.logger.print_output()
                        self.logger.clear_multiple_devices_output()
                        return

                    if val == clk_tuple["min_clk"]:
                        val_changed = False  # Clock limit value did not changed
                elif lim_type == "max":
                    amdsmi_lim_type = amdsmi_interface.AmdSmiClkLimitType.MAX
                    if val < clk_tuple["min_clk"]:
                        self.logger.store_output(
                            args.gpu,
                            "clk_limit",
                            f"Cannot set {args.clk_limit.clk_type} max value less than min ({clk_tuple['min_clk']}MHz)",
                        )
                        self.logger.print_output()
                        self.logger.clear_multiple_devices_output()
                        return
                    if val == clk_tuple["max_clk"]:
                        val_changed = False  # Clock limit value did not changed
            except amdsmi_exception.AmdSmiLibraryException as e:
                if (
                    e.get_error_code()
                    == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED
                    and lim_type == "min"
                    and clk_type == "mclk"
                ):
                    logging.debug("Setting mclk min is not supported")
                    self.logger.store_output(
                        args.gpu, "clk_limit", "Setting mclk min is not supported"
                    )
                else:
                    logging.debug(
                        "Failed to get clock extremum info for gpu %s | %s",
                        gpu_id,
                        e.get_error_info(),
                    )
                    self.logger.store_output(
                        args.gpu,
                        "clk_limit",
                        f"[{e.get_error_info(detailed=False)}] Unable to change {args.clk_limit.lim_type} of {args.clk_limit.clk_type} to {args.clk_limit.val}MHz",
                    )
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return

            # Set the value
            try:
                if val_changed:
                    amdsmi_interface.amdsmi_set_gpu_clk_limit(args.gpu, clk_type, lim_type, val)
            except amdsmi_exception.AmdSmiLibraryException as e:
                if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                    raise PermissionError("Command requires elevation") from e
                elif (
                    e.get_error_code()
                    == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED
                    and lim_type == "min"
                    and clk_type == "mclk"
                ):
                    logging.debug("Setting mclk min is not supported")
                    self.logger.store_output(
                        args.gpu, "clk_limit", "Setting mclk min is not supported"
                    )
                else:
                    self.logger.store_output(
                        args.gpu,
                        "clk_limit",
                        f"[{e.get_error_info(detailed=False)}] Unable to set {args.clk_limit.lim_type} of {args.clk_limit.clk_type} to {args.clk_limit.val}MHz",
                    )
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return

            if val_changed:
                self.logger.store_output(
                    args.gpu,
                    "clk_limit",
                    f"Successfully changed {args.clk_limit.lim_type} of {args.clk_limit.clk_type} to {args.clk_limit.val}MHz",
                )
            else:
                self.logger.store_output(
                    args.gpu, "clk_limit", f"Clock limit is already set to {args.clk_limit.val}MHz"
                )
            self.logger.print_output()
            self.logger.clear_multiple_devices_output()
            return
        if isinstance(args.process_isolation, int):
            status_string = "Enabled" if args.process_isolation else "Disabled"
            result = f"Requested process isolation to {status_string}"  # This should not print out
            try:
                current_status = amdsmi_interface.amdsmi_get_gpu_process_isolation(args.gpu)
                if current_status == args.process_isolation:
                    result = f"Process isolation is already {status_string}"
                else:
                    amdsmi_interface.amdsmi_set_gpu_process_isolation(
                        args.gpu, args.process_isolation
                    )
                    result = f"Successfully set process isolation to {status_string}"
            except amdsmi_exception.AmdSmiLibraryException as e:
                if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                    raise PermissionError("Command requires elevation") from e
                self.logger.store_output(
                    args.gpu,
                    "process_isolation",
                    f"[{e.get_error_info(detailed=False)}] Unable to set process isolation to {status_string}",
                )
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return

            self.logger.store_output(args.gpu, "process_isolation", result)
            self.logger.print_output()
            self.logger.clear_multiple_devices_output()
            return

        if args.mem_carveout is not None:
            # Validate single GPU (VRAM is per-GPU)
            if isinstance(args.gpu, list) and len(args.gpu) > 1:
                raise ValueError(
                    "VRAM carveout can only be set for a single GPU. Please specify --gpu <id>"
                )

            try:
                uma_info = amdsmi_interface.amdsmi_get_gpu_uma_carveout_info(args.gpu)
                options = uma_info.get("options", [])
                current_index = uma_info.get("current_index", -1)

                # Validate index
                if args.mem_carveout >= len(options):
                    self.logger.store_output(
                        args.gpu,
                        "mem_carveout",
                        f"Invalid index {args.mem_carveout}. Valid range: 0-{len(options) - 1}",
                    )
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    return

                # Check if already set
                if args.mem_carveout == current_index:
                    description = options[args.mem_carveout].get("description", "N/A")
                    self.logger.store_output(
                        args.gpu,
                        "mem_carveout",
                        f"VRAM carveout is already set to [{args.mem_carveout}] {description}",
                    )
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    return

                # Set the value
                amdsmi_interface.amdsmi_set_gpu_uma_carveout(args.gpu, args.mem_carveout)
                description = options[args.mem_carveout].get("description", "N/A")
                self.logger.store_output(
                    args.gpu,
                    "mem_carveout",
                    f"Successfully set VRAM carveout to [{args.mem_carveout}] {description}. Reboot required for changes to take effect.",
                )
                self.logger.print_output()
                self.helpers.prompt_reboot()

            except amdsmi_exception.AmdSmiLibraryException as e:
                if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                    raise PermissionError("Command requires elevation") from e
                if (
                    e.get_error_code()
                    == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED
                ):
                    # Surface an actionable message instead of a raw error code.
                    # Avoid naming specific products here so the message does not
                    # age as new ASICs add or drop UMA carveout support.
                    self.logger.store_output(
                        args.gpu,
                        "mem_carveout",
                        "Not supported: UMA carveout is only available on APUs whose"
                        ' VBIOS exposes the ATCS "Set UMA Allocation Size" function.',
                    )
                else:
                    self.logger.store_output(
                        args.gpu,
                        "mem_carveout",
                        f"[{e.get_error_info(detailed=False)}] Unable to set VRAM carveout to index {args.mem_carveout}",
                    )
                self.logger.print_output()

            self.logger.clear_multiple_devices_output()
            return

    def set_value(
        self,
        args,
        multiple_devices=False,
        gpu=None,
        fan=None,
        perf_level=None,
        profile=None,
        perf_determinism=None,
        compute_partition=None,
        memory_partition=None,
        power_cap=None,
        cpu=None,
        cpu_pwr_limit=None,
        cpu_xgmi_link_width=None,
        cpu_lclk_dpm_level=None,
        cpu_pwr_eff_mode=None,
        cpu_gmi3_link_width=None,
        cpu_pcie_link_rate=None,
        cpu_df_pstate_range=None,
        cpu_enable_apb=None,
        cpu_disable_apb=None,
        soc_boost_limit=None,
        core=None,
        core_boost_limit=None,
        soc_pstate=None,
        xgmi_plpd=None,
        process_isolation=None,
        clk_limit=None,
        clk_level=None,
        ptl_status=None,
        ptl_format=None,
        cpu_xgmi_pstate_range=None,
        cpu_railisofreq_policy=None,
        cpu_dfcstate_ctrl=None,
        cpu_pc6_enable=None,
        cpu_cc6_enable=None,
        cpu_floor_limit=None,
        cpu_msr_floor_limit=None,
        cpu_dimm_sb_reg=None,
        cpu_sdps_limit=None,
        core_floor_limit=None,
        core_msr_floor_limit=None,
        mem_carveout=None,
        gtt=None,
    ):
        """Issue reset commands to target gpu(s)

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            gpu (device_handle, optional): device_handle for target device. Defaults to None.
            fan (tuple, optional): Value override for args.fan as (value, is_percentage). Defaults to None.
            perf_level (amdsmi_interface.AmdSmiDevPerfLevel, optional): Value override for args.perf_level. Defaults to None.
            profile (bool, optional): Value override for args.profile. Defaults to None.
            perf_determinism (int, optional): Value override for args.perf_determinism. Defaults to None.
            compute_partition (amdsmi_interface.AmdSmiComputePartitionType, optional): Value override for args.compute_partition. Defaults to None.
            memory_partition (amdsmi_interface.AmdSmiMemoryPartitionType, optional): Value override for args.memory_partition. Defaults to None.
            power_cap (int, optional): Value override for args.power_cap. Defaults to None.

            cpu (cpu_handle, optional): device_handle for target device. Defaults to None.
            cpu_pwr_limit (int, optional): Value override for args.cpu_pwr_limit. Defaults to None.
            cpu_xgmi_link_width (List[int], optional): Value override for args.cpu_xgmi_link_width. Defaults to None.
            cpu_lclk_dpm_level (List[int], optional): Value override for args.cpu_lclk_dpm_level. Defaults to None.
            cpu_pwr_eff_mode (List[int], optional): Value override for args.cpu_pwr_eff_mode [mode, util, ppt_limit]. Defaults to None.
            cpu_gmi3_link_width (List[int], optional): Value override for args.cpu_gmi3_link_width. Defaults to None.
            cpu_pcie_link_rate (int, optional): Value override for args.cpu_pcie_link_rate. Defaults to None.
            cpu_df_pstate_range (List[int], optional): Value override for args.cpu_df_pstate_range. Defaults to None.
            cpu_enable_apb (bool, optional): Value override for args.cpu_enable_apb. Defaults to None.
            cpu_disable_apb (int, optional): Value override for args.cpu_disable_apb. Defaults to None.
            soc_boost_limit (int, optional): Value override for args.soc_boost_limit. Defaults to None.
            cpu_dfcstate_ctrl (int, optional): Value override for args.cpu_dfcstate_ctrl. Defaults to None.
            cpu_railisofreq_policy (int, optional): Value override for args.cpu_railisofreq_policy. Defaults to None.
            cpu_xgmi_pstate_range (List[int], optional): Value override for args.cpu_xgmi_pstate_range. Defaults to None.
            cpu_pc6_enable (int, optional): Value override for args.cpu_pc6_enable. Defaults to None.
            cpu_cc6_enable (int, optional): Value override for args.cpu_cc6_enable. Defaults to None.
            cpu_dimm_sb_reg (list, optional): DIMM sideband register write parameters [dimm_addr, lid, reg_offset, reg_space, write_data] for write operation. Value override for args.cpu_dimm_sb_reg. Defaults to None.
            cpu_sdps_limit (int, optional): Value override for args.cpu_sdps_limit. Defaults to None.
            cpu_floor_limit (int, optional): Value override for args.cpu_floor_limit. Defaults to None.
            cpu_msr_floor_limit (int, optional): Value override for args.cpu_msr_floor_limit. Defaults to None.

            core (device_handle, optional): device_handle for target core. Defaults to None.
            core_boost_limit (int, optional): Value override for args.core_boost_limit. Defaults to None
            core_floor_limit (int, optional): Value override for args.core_floor_limit. Defaults to None.
            core_msr_floor_limit (int, optional): Value override for args.core_msr_floor_limit. Defaults to None.
            soc_pstate (int, optional): Value override for args.soc_pstate. Defaults to None.
            xgmi_plpd (int, optional): Value override for args.xgmi_plpd. Defaults to None.
            process_isolation (int, optional): Value override for args.process_isolation. Defaults to None.
        Raises:
            ValueError: Value error if no gpu value is provided
            IndexError: Index error if gpu list is empty

        Return:
            Nothing
        """
        # These are the only args checked at this point, the other args will be passed
        #   in through the applicable function set_gpu, set_cpu, or set_core function
        if gpu:
            args.gpu = gpu
        if cpu:
            args.cpu = cpu
        if core:
            args.core = core

        # Special GTT handling (system-wide, not per-GPU) — handle before device dispatch
        if hasattr(args, "gtt") and args.gtt is not None:
            if hasattr(args, "gpu") and args.gpu is not None:
                print(
                    "amd-smi set: error: argument --gtt/-G: not allowed with argument --gpu/-g "
                    "(--gtt is a system-wide setting, not per-GPU)",
                    file=sys.stderr,
                )
                sys.exit(2)
            gb_value = args.gtt
            pages = self.helpers.gb_to_pages(gb_value)
            try:
                amdsmi_interface.amdsmi_set_ttm_pages_limit(pages)
                self.logger.output["set_gtt"] = (
                    f"Successfully set GTT to {gb_value:.2f} GB ({pages} pages). Reboot required for changes to take effect."
                )
                self.logger.print_output()
                self.helpers.prompt_reboot()
                return
            except amdsmi_exception.AmdSmiLibraryException as e:
                if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                    raise PermissionError("Command requires elevation") from e
                self.logger.output["set_gtt"] = (
                    f"[{e.get_error_info(detailed=False)}] Unable to set GTT to {gb_value:.2f} GB"
                )
                self.logger.print_output()
                return

        # Check if a GPU argument has been set
        gpu_args_enabled = False
        gpu_attributes = [
            "fan",
            "perf_level",
            "profile",
            "perf_determinism",
            "compute_partition",
            "memory_partition",
            "power_cap",
            "soc_pstate",
            "xgmi_plpd",
            "process_isolation",
            "clk_limit",
            "clk_level",
            "ptl_status",
            "ptl_format",
            "mem_carveout",
        ]
        for attr in gpu_attributes:
            if hasattr(args, attr):
                if getattr(args, attr) is not None:
                    gpu_args_enabled = True
                    break
        # Check if a CPU argument has been set
        cpu_args_enabled = False
        cpu_attributes = [
            "cpu_pwr_limit",
            "cpu_xgmi_link_width",
            "cpu_lclk_dpm_level",
            "cpu_pwr_eff_mode",
            "cpu_gmi3_link_width",
            "cpu_pcie_link_rate",
            "cpu_df_pstate_range",
            "cpu_enable_apb",
            "cpu_disable_apb",
            "soc_boost_limit",
            "cpu_xgmi_pstate_range",
            "cpu_railisofreq_policy",
            "cpu_dfcstate_ctrl",
            "cpu_pc6_enable",
            "cpu_cc6_enable",
            "cpu_floor_limit",
            "cpu_msr_floor_limit",
            "cpu_dimm_sb_reg",
            "cpu_sdps_limit",
        ]
        for attr in cpu_attributes:
            if hasattr(args, attr):
                if getattr(args, attr) not in [None, False]:
                    cpu_args_enabled = True
                    break

        # Check if a Core argument has been set
        core_args_enabled = False
        core_attributes = ["core_boost_limit", "core_floor_limit", "core_msr_floor_limit"]
        for attr in core_attributes:
            if hasattr(args, attr):
                if getattr(args, attr) is not None:
                    core_args_enabled = True
                    break

        # Error if no subcommand args are passed
        if self.helpers.is_baremetal():
            is_gpu_set = False
            is_cpu_set = False
            is_core_set = False
            try:
                is_gpu_set = any(
                    [
                        args.gpu is not None,
                        args.fan is not None,
                        args.perf_level is not None,
                        args.profile is not None,
                        args.perf_determinism is not None,
                        args.compute_partition is not None,
                        args.memory_partition is not None,
                        args.power_cap is not None,
                        args.soc_pstate is not None,
                        args.xgmi_plpd is not None,
                        args.clk_limit is not None,
                        args.clk_level is not None,
                        args.ptl_status is not None,
                        args.ptl_format is not None,
                        args.process_isolation is not None,
                        args.mem_carveout is not None,
                    ]
                )
            except AttributeError:
                # If attribute error for gpu, then we could be another subcommand
                pass

            try:
                is_cpu_set = any(
                    [
                        args.cpu is not None,
                        args.cpu_pwr_limit is not None,
                        args.cpu_xgmi_link_width is not None,
                        args.cpu_lclk_dpm_level is not None,
                        args.cpu_pwr_eff_mode is not None,
                        args.cpu_gmi3_link_width is not None,
                        args.cpu_pcie_link_rate is not None,
                        args.cpu_df_pstate_range is not None,
                        args.cpu_enable_apb,
                        args.cpu_disable_apb is not None,
                        args.soc_boost_limit is not None,
                        args.cpu_xgmi_pstate_range is not None,
                        args.cpu_railisofreq_policy is not None,
                        args.cpu_dfcstate_ctrl is not None,
                        args.cpu_pc6_enable is not None,
                        args.cpu_cc6_enable is not None,
                        args.cpu_floor_limit is not None,
                        args.cpu_msr_floor_limit is not None,
                        args.cpu_dimm_sb_reg is not None,
                        args.cpu_sdps_limit is not None,
                    ]
                )
            except AttributeError:
                # If attribute error for cpu, then we could be another subcommand
                pass
            try:
                if args.core_boost_limit:
                    is_core_set = True
                if args.core_floor_limit:
                    is_core_set = True
                if args.core_msr_floor_limit:
                    is_core_set = True
            except AttributeError:
                # If attribute error for core, then we could be another subcommand
                pass

            if not (is_gpu_set or is_cpu_set or is_core_set):
                # if neither GPU / CPU / or Core args are provided, then raise error message
                command = " ".join(sys.argv[1:])
                raise AmdSmiRequiredCommandException(command, self.logger.format)
        else:
            if not any(
                [
                    args.process_isolation is not None,
                    args.clk_limit is not None,
                    args.power_cap is not None,
                ]
            ):
                command = " ".join(sys.argv[1:])
                raise AmdSmiRequiredCommandException(command, self.logger.format)

        # Only allow one device's arguments to be set at a time
        if not any([gpu_args_enabled, cpu_args_enabled, core_args_enabled]):
            raise ValueError(
                "No GPU, CPU, or CORE arguments provided, specific arguments are needed"
            )
        elif all([gpu_args_enabled, cpu_args_enabled, core_args_enabled]):
            raise ValueError("Cannot set GPU, CPU, and CORE arguments at the same time")
        elif not (gpu_args_enabled ^ cpu_args_enabled ^ core_args_enabled):
            raise ValueError("Cannot set GPU, CPU, or CORE arguments at the same time")

        if self.helpers.is_amdgpu_initialized() and gpu_args_enabled:
            if args.gpu == None:
                args.gpu = self.device_handles

        if self.helpers.is_amd_hsmp_initialized() and cpu_args_enabled:
            if args.cpu == None:
                args.cpu = self.cpu_handles

        if self.helpers.is_amd_hsmp_initialized() and core_args_enabled:
            if args.core == None:
                args.core = self.core_handles

        # Handle CPU and GPU initialization cases
        if self.helpers.is_amd_hsmp_initialized() and self.helpers.is_amdgpu_initialized():
            # Print out all CPU and all GPU static info only if no device was specified.
            # If a GPU or CPU argument is provided only print out the specified device.
            if args.cpu == None and args.gpu == None and args.core == None:
                raise ValueError("No GPU, CPU, or CORE provided, specific target(s) are needed")

            if args.cpu:
                self.set_cpu(
                    args,
                    multiple_devices,
                    cpu,
                    cpu_pwr_limit,
                    cpu_xgmi_link_width,
                    cpu_lclk_dpm_level,
                    cpu_pwr_eff_mode,
                    cpu_gmi3_link_width,
                    cpu_pcie_link_rate,
                    cpu_df_pstate_range,
                    cpu_enable_apb,
                    cpu_disable_apb,
                    soc_boost_limit,
                    cpu_xgmi_pstate_range,
                    cpu_railisofreq_policy,
                    cpu_dfcstate_ctrl,
                    cpu_pc6_enable,
                    cpu_cc6_enable,
                    cpu_floor_limit,
                    cpu_msr_floor_limit,
                    cpu_dimm_sb_reg,
                    cpu_sdps_limit,
                )
            if args.core:
                self.logger.output = {}
                self.logger.clear_multiple_devices_output()
                self.set_core(
                    args,
                    multiple_devices,
                    core,
                    core_boost_limit,
                    core_floor_limit,
                    core_msr_floor_limit,
                )
            if args.gpu:
                self.logger.output = {}
                self.logger.clear_multiple_devices_output()
                self.set_gpu(
                    args,
                    multiple_devices,
                    gpu,
                    fan,
                    perf_level,
                    profile,
                    perf_determinism,
                    compute_partition,
                    memory_partition,
                    power_cap,
                    soc_pstate,
                    xgmi_plpd,
                    process_isolation,
                    clk_limit,
                    clk_level,
                    ptl_status,
                    ptl_format,
                    mem_carveout,
                )
        elif self.helpers.is_amd_hsmp_initialized():  # Only CPU is initialized
            if args.cpu == None and args.core == None:
                raise ValueError("No CPU or CORE provided, specific target(s) are needed")
            if args.cpu:
                self.set_cpu(
                    args,
                    multiple_devices,
                    cpu,
                    cpu_pwr_limit,
                    cpu_xgmi_link_width,
                    cpu_lclk_dpm_level,
                    cpu_pwr_eff_mode,
                    cpu_gmi3_link_width,
                    cpu_pcie_link_rate,
                    cpu_df_pstate_range,
                    cpu_enable_apb,
                    cpu_disable_apb,
                    soc_boost_limit,
                    cpu_xgmi_pstate_range,
                    cpu_railisofreq_policy,
                    cpu_dfcstate_ctrl,
                    cpu_pc6_enable,
                    cpu_cc6_enable,
                    cpu_floor_limit,
                    cpu_msr_floor_limit,
                    cpu_dimm_sb_reg,
                    cpu_sdps_limit,
                )
            if args.core:
                self.logger.output = {}
                self.logger.clear_multiple_devices_output()
                self.set_core(
                    args,
                    multiple_devices,
                    core,
                    core_boost_limit,
                    core_floor_limit,
                    core_msr_floor_limit,
                )
        elif self.helpers.is_amdgpu_initialized():  # Only GPU is initialized
            if args.gpu == None:
                args.gpu = self.device_handles
            self.logger.clear_multiple_devices_output()
            self.set_gpu(
                args,
                multiple_devices,
                gpu,
                fan,
                perf_level,
                profile,
                perf_determinism,
                compute_partition,
                memory_partition,
                power_cap,
                soc_pstate,
                xgmi_plpd,
                process_isolation,
                clk_limit,
                clk_level,
                ptl_status,
                ptl_format,
                mem_carveout,
            )
