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

import functools
import logging
import os

from amdsmi_cli_exceptions import AmdSmiInvalidParameterException
from amdsmi_helpers import AMDSMIHelpers

from amdsmi import amdsmi_exception, amdsmi_interface


class StaticCommands:
    def static_cpu(self, args, multiple_devices=False, cpu=None, interface_ver=None):
        """Get Static information for target cpu

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            cpu (device_handle, optional): device_handle for target device. Defaults to None.

        Returns:
            None: Print output via AMDSMILogger to destination
        """

        if cpu:
            args.cpu = cpu
        if interface_ver:
            args.interface_ver = interface_ver

        # Store cpu args that are applicable to the current platform
        curr_platform_cpu_args = ["smu", "interface_ver"]
        curr_platform_cpu_values = [args.smu, args.interface_ver]

        # If no cpu options are passed, return all available args
        if not any(curr_platform_cpu_values):
            for arg in curr_platform_cpu_args:
                setattr(args, arg, True)

        # Handle multiple CPUs
        handled_multiple_cpus, device_handle = self.helpers.handle_cpus(
            args, self.logger, self.static_cpu
        )
        if handled_multiple_cpus:
            return  # This function is recursive
        args.cpu = device_handle

        # Get cpu id for logging
        cpu_id = self.helpers.get_cpu_id_from_device_handle(args.cpu)
        logging.debug(f"Static Arg information for CPU {cpu_id} on {self.helpers.os_info()}")

        static_dict = {}
        if self.logger.is_json_format():
            static_dict["cpu"] = int(cpu_id)

        if args.smu:
            try:
                smu = amdsmi_interface.amdsmi_get_cpu_smu_fw_version(args.cpu)
                static_dict["smu"] = {
                    "FW_VERSION": f"{smu['smu_fw_major_ver_num']}."
                    f"{smu['smu_fw_minor_ver_num']}.{smu['smu_fw_debug_ver_num']}"
                }
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["smu"] = "N/A"
                logging.debug("Failed to get SMU FW for cpu %s | %s", cpu_id, e.get_error_info())

        if args.interface_ver:
            static_dict["interface_version"] = {}
            try:
                intf_ver = amdsmi_interface.amdsmi_get_cpu_hsmp_proto_ver(args.cpu)
                static_dict["interface_version"]["proto version"] = intf_ver
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["interface_version"]["proto version"] = "N/A"
                logging.debug(
                    "Failed to get proto version for cpu %s | %s", cpu_id, e.get_error_info()
                )

        multiple_devices_csv_override = False
        if not self.logger.is_json_format():
            self.logger.store_cpu_output(args.cpu, "values", static_dict)
        else:
            self.logger.store_cpu_json_output.append(static_dict)
        if multiple_devices:
            self.logger.store_multiple_device_output()
            return  # Skip printing when there are multiple devices
        if not self.logger.is_json_format():
            self.logger.print_output(multiple_device_enabled=multiple_devices_csv_override)

    def static_gpu(
        self,
        args,
        multiple_devices=False,
        gpu=None,
        asic=None,
        bus=None,
        vbios=None,
        limit=None,
        driver=None,
        ras=None,
        board=None,
        numa=None,
        vram=None,
        cache=None,
        partition=None,
        dfc_ucode=None,
        fb_info=None,
        num_vf=None,
        soc_pstate=None,
        xgmi_plpd=None,
        process_isolation=None,
        clock=None,
        profile=None,
        mem_carveout=None,
    ):
        """Get Static information for target gpu

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            current_platform_args (list): gpu supported platform arguments
            current_platform_values (list): gpu supported platform values for each argument
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            gpu (device_handle, optional): device_handle for target device. Defaults to None.
            asic (bool, optional): Value override for args.asic. Defaults to None.
            bus (bool, optional): Value override for args.bus. Defaults to None.
            vbios (bool, optional): Value override for args.vbios. Defaults to None.
            limit (bool, optional): Value override for args.limit. Defaults to None.
            driver (bool, optional): Value override for args.driver. Defaults to None.
            ras (bool, optional): Value override for args.ras. Defaults to None.
            board (bool, optional): Value override for args.board. Defaults to None.
            numa (bool, optional): Value override for args.numa. Defaults to None.
            vram (bool, optional): Value override for args.vram. Defaults to None.
            cache (bool, optional): Value override for args.cache. Defaults to None.
            partition (bool, optional): Value override for args.partition. Defaults to None.
            dfc_ucode (bool, optional): Value override for args.dfc_ucode. Defaults to None.
            fb_info (bool, optional): Value override for args.fb_info. Defaults to None.
            num_vf (bool, optional): Value override for args.num_vf. Defaults to None.
            soc_pstate (bool, optional): Value override for args.soc_pstate. Defaults to None.
            xgmi_plpd (bool, optional): Value override for args.xgmi_plpd. Defaults to None.
            process_isolation (bool, optional): Value override for args.process_isolation. Defaults to None.
        Returns:
            None: Print output via AMDSMILogger to destination
        """

        if gpu:
            args.gpu = gpu
        if asic:
            args.asic = asic
        if bus:
            args.bus = bus
        if vbios:
            args.vbios = vbios
        if board:
            args.board = board
        if driver:
            args.driver = driver
        if ras:
            args.ras = ras
        if vram:
            args.vram = vram
        if cache:
            args.cache = cache
        if process_isolation:
            args.process_isolation = process_isolation
        if partition:
            args.partition = partition
        if clock:
            args.clock = clock
        if mem_carveout:
            args.mem_carveout = mem_carveout

        # args.clock defaults to False so if it was overwritten to empty list, that indicates that it was given as an arguments but with an empty list
        if args.clock == []:
            args.clock = True

        # Store args that are applicable to the current platform (default arguments)
        current_platform_args = [
            "asic",
            "bus",
            "vbios",
            "driver",
            "ras",
            "vram",
            "cache",
            "board",
            "process_isolation",
            "clock",
            "mem_carveout",
        ]
        current_platform_values = [
            args.asic,
            args.bus,
            args.vbios,
            args.driver,
            args.ras,
            args.vram,
            args.cache,
            args.board,
            args.process_isolation,
            args.clock,
            args.mem_carveout,
        ]

        # amd-smi static default arguments:
        # Exclude args that are not applicable to the current platform,
        # but allow output if argument is passed.
        #
        # Note: Partition is a special case, it is no longer an amd-smi static
        # default argument.
        # Reason: Reading current_compute_partition may momentarily wake the
        #         GPU up. This is due to reading XCD registers, which is expected
        #         behavior. Changing partitions is not a trivial operation,
        #         current_compute_partition SYSFS controls this action.
        if args.partition:
            current_platform_args += ["partition"]
            current_platform_values += [args.partition]

        if not self.group_check_printed:
            self.helpers.check_required_groups()
            self.group_check_printed = True

        if self.helpers.is_linux() and self.helpers.is_baremetal():
            if limit:
                args.limit = limit
            if soc_pstate:
                args.soc_pstate = soc_pstate
            if xgmi_plpd:
                args.xgmi_plpd = xgmi_plpd
            if profile:
                args.profile = profile
            current_platform_args += ["ras", "limit", "soc_pstate", "xgmi_plpd", "profile"]
            current_platform_values += [
                args.ras,
                args.limit,
                args.soc_pstate,
                args.xgmi_plpd,
                args.profile,
            ]

        if self.helpers.is_linux() and not self.helpers.is_virtual_os():
            if numa:
                args.numa = numa
            current_platform_args += ["numa"]
            current_platform_values += [args.numa]

        if self.helpers.is_hypervisor():
            if dfc_ucode:
                args.dfc_ucode = dfc_ucode
            if fb_info:
                args.fb_info = fb_info
            if num_vf:
                args.num_vf = num_vf
            current_platform_args += ["dfc_ucode", "fb_info", "num_vf"]
            current_platform_values += [args.dfc_ucode, args.fb_info, args.num_vf]

        if not any(current_platform_values):
            for arg in current_platform_args:
                setattr(args, arg, True)

        handled_multiple_gpus, device_handle = self.helpers.handle_gpus(
            args, self.logger, self.static_gpu
        )
        if handled_multiple_gpus:
            return  # This function is recursive
        args.gpu = device_handle
        # Get gpu_id for logging
        gpu_id = self.helpers.get_gpu_id_from_device_handle(args.gpu)

        logging.debug("=====================================================================")
        logging.debug(f"Static Arg information for GPU {gpu_id} on {self.helpers.os_info()}")
        logging.debug(f"Function args:           {args}")
        logging.debug(f"Current platform args:   {current_platform_args}")
        logging.debug(f"Current platform values: {current_platform_values}")
        logging.debug("=====================================================================")

        # Populate static dictionary for each enabled argument
        static_dict = {}
        if self.logger.is_json_format():
            static_dict["gpu"] = int(gpu_id)
        if args.asic:
            asic_dict = {
                "market_name": "N/A",
                "vendor_id": "N/A",
                "vendor_name": "N/A",
                "subvendor_id": "N/A",
                "device_id": "N/A",
                "subsystem_id": "N/A",
                "rev_id": "N/A",
                "asic_serial": "N/A",
                "oam_id": "N/A",
                "num_compute_units": "N/A",
                "target_graphics_version": "N/A",
            }

            try:
                asic_info = amdsmi_interface.amdsmi_get_gpu_asic_info(args.gpu)
                for key, value in asic_info.items():
                    asic_dict[key] = value
            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug("Failed to get asic info for gpu %s | %s", gpu_id, e.get_error_info())

            static_dict["asic"] = asic_dict
        if args.bus:
            bus_info = {
                "bdf": "N/A",
                "max_pcie_width": "N/A",
                "max_pcie_speed": "N/A",
                "pcie_levels": "N/A",
                "pcie_interface_version": "N/A",
                "slot_type": "N/A",
            }

            try:
                bus_info["bdf"] = amdsmi_interface.amdsmi_get_gpu_device_bdf(args.gpu)
            except amdsmi_exception.AmdSmiLibraryException as e:
                bus_info["bdf"] = "N/A"
                logging.debug("Failed to get bdf for gpu %s | %s", gpu_id, e.get_error_info())

            try:
                pcie_static = amdsmi_interface.amdsmi_get_pcie_info(args.gpu)["pcie_static"]
                bus_info["max_pcie_width"] = pcie_static["max_pcie_width"]
                bus_info["max_pcie_speed"] = pcie_static["max_pcie_speed"]
                bus_info["pcie_interface_version"] = pcie_static["pcie_interface_version"]
                bus_info["slot_type"] = pcie_static["slot_type"]
                if bus_info["max_pcie_speed"] % 1000 != 0:
                    pcie_speed_GTs_value = round(bus_info["max_pcie_speed"] / 1000, 1)
                else:
                    pcie_speed_GTs_value = round(bus_info["max_pcie_speed"] / 1000)

                bus_info["max_pcie_speed"] = pcie_speed_GTs_value

                if bus_info["pcie_interface_version"] > 0:
                    bus_info["pcie_interface_version"] = f"Gen {bus_info['pcie_interface_version']}"

                # Set the unit for pcie_speed
                pcie_speed_unit = "GT/s"
                if self.logger.is_human_readable_format():
                    bus_info["max_pcie_speed"] = f"{bus_info['max_pcie_speed']} {pcie_speed_unit}"

                if self.logger.is_json_format():
                    bus_info["max_pcie_speed"] = {
                        "value": bus_info["max_pcie_speed"],
                        "unit": pcie_speed_unit,
                    }

            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug("Failed to get bus info for gpu %s | %s", gpu_id, e.get_error_info())

            try:
                pcie_info = amdsmi_interface.amdsmi_get_gpu_pci_bandwidth(args.gpu)
                num_supported = pcie_info["transfer_rate"]["num_supported"]
                if num_supported != 0:
                    bus_info["pcie_levels"] = {}
                    for level in range(0, num_supported):
                        speed = (
                            str(
                                self.helpers.convert_SI_unit(
                                    float(pcie_info["transfer_rate"]["frequency"][level]),
                                    AMDSMIHelpers.SI_Unit.NANO,
                                )
                            )
                            + " GT/s"
                        )
                        width = str(pcie_info["lanes"][level])
                        level_values = (speed, width)
                        bus_info["pcie_levels"].update({str(level): level_values})
                else:
                    bus_info["pcie_levels"] = "N/A"
            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug(
                    "Failed to get pci bandwidth info for gpu %s | %s", gpu_id, e.get_error_info()
                )

            static_dict["bus"] = bus_info
        if args.vbios:
            try:
                vbios_info = amdsmi_interface.amdsmi_get_gpu_vbios_info(args.gpu)
                for key, value in vbios_info.items():
                    if isinstance(value, str):
                        if value.strip() == "":
                            vbios_info[key] = "N/A"
                static_dict["ifwi"] = vbios_info
                # Remove boot_firmware since it's not used
                del static_dict["ifwi"]["boot_firmware"]
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["ifwi"] = "N/A"
                logging.debug(
                    "Failed to get vbios/ifwi info for gpu %s | %s", gpu_id, e.get_error_info()
                )
        if "limit" in current_platform_args:
            if args.limit:
                # Power limits

                power_limit_types = {}
                for power_type in amdsmi_interface.AmdSmiPowerCapType:
                    # Strip 'AMDSMI_POWER_CAP_TYPE_' prefix and convert to lowercase
                    key = power_type.name.replace("AMDSMI_POWER_CAP_TYPE_", "").lower()
                    power_limit_types[key] = {
                        "max_power_limit": "N/A",
                        "min_power_limit": "N/A",
                        "socket_power_limit": "N/A",
                    }

                try:
                    power_cap_types = amdsmi_interface.amdsmi_get_supported_power_cap(args.gpu)
                    for sensor in power_cap_types["sensor_inds"]:
                        power_cap_info = amdsmi_interface.amdsmi_get_power_cap_info(
                            args.gpu, sensor
                        )
                        max_power_limit = power_cap_info["max_power_cap"]
                        max_power_limit = self.helpers.convert_SI_unit(
                            max_power_limit, AMDSMIHelpers.SI_Unit.MICRO
                        )
                        min_power_limit = power_cap_info["min_power_cap"]
                        min_power_limit = self.helpers.convert_SI_unit(
                            min_power_limit, AMDSMIHelpers.SI_Unit.MICRO
                        )
                        socket_power_limit = power_cap_info["power_cap"]
                        socket_power_limit = self.helpers.convert_SI_unit(
                            socket_power_limit, AMDSMIHelpers.SI_Unit.MICRO
                        )
                        ppt = {
                            "max_power_limit": self.helpers.unit_format(
                                self.logger, max_power_limit, "W"
                            ),
                            "min_power_limit": self.helpers.unit_format(
                                self.logger, min_power_limit, "W"
                            ),
                            "socket_power_limit": self.helpers.unit_format(
                                self.logger, socket_power_limit, "W"
                            ),
                        }

                        sensor_name = power_cap_types["sensor_types"][sensor]
                        # Strip 'AMDSMI_POWER_CAP_TYPE_' prefix and convert to lowercase
                        sensor_key = sensor_name.name.replace("AMDSMI_POWER_CAP_TYPE_", "").lower()
                        power_limit_types[sensor_key] = ppt
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug(
                        "Failed to get power cap info for gpu %s | %s", gpu_id, e.get_error_info()
                    )

                # Edge temperature limits
                try:
                    slowdown_temp_edge_limit_error = False
                    slowdown_temp_edge_limit = amdsmi_interface.amdsmi_get_temp_metric(
                        args.gpu,
                        amdsmi_interface.AmdSmiTemperatureType.EDGE,
                        amdsmi_interface.AmdSmiTemperatureMetric.CRITICAL,
                    )
                except amdsmi_exception.AmdSmiLibraryException as e:
                    slowdown_temp_edge_limit_error = True
                    slowdown_temp_edge_limit = "N/A"
                    logging.debug(
                        "Failed to get edge temperature slowdown metric for gpu %s | %s",
                        gpu_id,
                        e.get_error_info(),
                    )

                if slowdown_temp_edge_limit == 0:
                    slowdown_temp_edge_limit_error = True
                    slowdown_temp_edge_limit = "N/A"

                try:
                    shutdown_temp_edge_limit_error = False
                    shutdown_temp_edge_limit = amdsmi_interface.amdsmi_get_temp_metric(
                        args.gpu,
                        amdsmi_interface.AmdSmiTemperatureType.EDGE,
                        amdsmi_interface.AmdSmiTemperatureMetric.EMERGENCY,
                    )
                except amdsmi_exception.AmdSmiLibraryException as e:
                    shutdown_temp_edge_limit_error = True
                    shutdown_temp_edge_limit = "N/A"
                    logging.debug(
                        "Failed to get edge temperature shutdown metrics for gpu %s | %s",
                        gpu_id,
                        e.get_error_info(),
                    )

                if shutdown_temp_edge_limit == 0:
                    shutdown_temp_edge_limit_error = True
                    shutdown_temp_edge_limit = "N/A"

                # Hotspot/Junction temperature limits
                try:
                    slowdown_temp_hotspot_limit_error = False
                    slowdown_temp_hotspot_limit = amdsmi_interface.amdsmi_get_temp_metric(
                        args.gpu,
                        amdsmi_interface.AmdSmiTemperatureType.HOTSPOT,
                        amdsmi_interface.AmdSmiTemperatureMetric.CRITICAL,
                    )
                except amdsmi_exception.AmdSmiLibraryException as e:
                    slowdown_temp_hotspot_limit_error = True
                    slowdown_temp_hotspot_limit = "N/A"
                    logging.debug(
                        "Failed to get hotspot temperature slowdown metrics for gpu %s | %s",
                        gpu_id,
                        e.get_error_info(),
                    )

                try:
                    shutdown_temp_hotspot_limit_error = False
                    shutdown_temp_hotspot_limit = amdsmi_interface.amdsmi_get_temp_metric(
                        args.gpu,
                        amdsmi_interface.AmdSmiTemperatureType.HOTSPOT,
                        amdsmi_interface.AmdSmiTemperatureMetric.EMERGENCY,
                    )
                except amdsmi_exception.AmdSmiLibraryException as e:
                    shutdown_temp_hotspot_limit_error = True
                    shutdown_temp_hotspot_limit = "N/A"
                    logging.debug(
                        "Failed to get hotspot temperature shutdown metrics for gpu %s | %s",
                        gpu_id,
                        e.get_error_info(),
                    )

                # VRAM temperature limits
                try:
                    slowdown_temp_vram_limit_error = False
                    slowdown_temp_vram_limit = amdsmi_interface.amdsmi_get_temp_metric(
                        args.gpu,
                        amdsmi_interface.AmdSmiTemperatureType.VRAM,
                        amdsmi_interface.AmdSmiTemperatureMetric.CRITICAL,
                    )
                except amdsmi_exception.AmdSmiLibraryException as e:
                    slowdown_temp_vram_limit_error = True
                    slowdown_temp_vram_limit = "N/A"
                    logging.debug(
                        "Failed to get vram temperature slowdown metrics for gpu %s | %s",
                        gpu_id,
                        e.get_error_info(),
                    )

                try:
                    shutdown_temp_vram_limit_error = False
                    shutdown_temp_vram_limit = amdsmi_interface.amdsmi_get_temp_metric(
                        args.gpu,
                        amdsmi_interface.AmdSmiTemperatureType.VRAM,
                        amdsmi_interface.AmdSmiTemperatureMetric.EMERGENCY,
                    )
                except amdsmi_exception.AmdSmiLibraryException as e:
                    shutdown_temp_vram_limit_error = True
                    shutdown_temp_vram_limit = "N/A"
                    logging.debug(
                        "Failed to get vram temperature shutdown metrics for gpu %s | %s",
                        gpu_id,
                        e.get_error_info(),
                    )

                # PTL
                try:
                    ptl_state = amdsmi_interface.amdsmi_get_gpu_ptl_state(args.gpu)
                    ptl_state = "Enabled" if ptl_state else "Disabled"
                except amdsmi_exception.AmdSmiLibraryException as e:
                    ptl_state = "N/A"
                    logging.debug(
                        "Failed to get PTL state for gpu %s | %s", gpu_id, e.get_error_info()
                    )

                try:
                    ptl_format1, ptl_format2 = amdsmi_interface.amdsmi_get_gpu_ptl_formats(args.gpu)
                    fmt1_name = (
                        amdsmi_interface.amdsmi_wrapper.amdsmi_ptl_data_format_t__enumvalues.get(
                            ptl_format1
                        )
                    )
                    fmt2_name = (
                        amdsmi_interface.amdsmi_wrapper.amdsmi_ptl_data_format_t__enumvalues.get(
                            ptl_format2
                        )
                    )

                    fmt1_short = (
                        fmt1_name.replace("AMDSMI_PTL_DATA_FORMAT_", "") if fmt1_name else "UNKNOWN"
                    )
                    fmt2_short = (
                        fmt2_name.replace("AMDSMI_PTL_DATA_FORMAT_", "") if fmt2_name else "UNKNOWN"
                    )

                    ptl_format = f"{fmt1_short},{fmt2_short}"
                except amdsmi_exception.AmdSmiLibraryException as e:
                    ptl_format = "N/A"
                    logging.debug(
                        "Failed to get PTL state for gpu %s | %s", gpu_id, e.get_error_info()
                    )

                # Assign units
                power_unit = "W"
                temp_unit_human_readable = "\N{DEGREE SIGN}C"
                temp_unit_json = "C"

                if self.logger.is_human_readable_format():
                    if not slowdown_temp_edge_limit_error:
                        slowdown_temp_edge_limit = (
                            f"{slowdown_temp_edge_limit} {temp_unit_human_readable}"
                        )
                    if not slowdown_temp_hotspot_limit_error:
                        slowdown_temp_hotspot_limit = (
                            f"{slowdown_temp_hotspot_limit} {temp_unit_human_readable}"
                        )
                    if not slowdown_temp_vram_limit_error:
                        slowdown_temp_vram_limit = (
                            f"{slowdown_temp_vram_limit} {temp_unit_human_readable}"
                        )
                    if not shutdown_temp_edge_limit_error:
                        shutdown_temp_edge_limit = (
                            f"{shutdown_temp_edge_limit} {temp_unit_human_readable}"
                        )
                    if not shutdown_temp_hotspot_limit_error:
                        shutdown_temp_hotspot_limit = (
                            f"{shutdown_temp_hotspot_limit} {temp_unit_human_readable}"
                        )
                    if not shutdown_temp_vram_limit_error:
                        shutdown_temp_vram_limit = (
                            f"{shutdown_temp_vram_limit} {temp_unit_human_readable}"
                        )

                if self.logger.is_json_format():
                    if not slowdown_temp_edge_limit_error:
                        slowdown_temp_edge_limit = {
                            "value": slowdown_temp_edge_limit,
                            "unit": temp_unit_json,
                        }
                    if not slowdown_temp_hotspot_limit_error:
                        slowdown_temp_hotspot_limit = {
                            "value": slowdown_temp_hotspot_limit,
                            "unit": temp_unit_json,
                        }
                    if not slowdown_temp_vram_limit_error:
                        slowdown_temp_vram_limit = {
                            "value": slowdown_temp_vram_limit,
                            "unit": temp_unit_json,
                        }
                    if not shutdown_temp_edge_limit_error:
                        shutdown_temp_edge_limit = {
                            "value": shutdown_temp_edge_limit,
                            "unit": temp_unit_json,
                        }
                    if not shutdown_temp_hotspot_limit_error:
                        shutdown_temp_hotspot_limit = {
                            "value": shutdown_temp_hotspot_limit,
                            "unit": temp_unit_json,
                        }
                    if not shutdown_temp_vram_limit_error:
                        shutdown_temp_vram_limit = {
                            "value": shutdown_temp_vram_limit,
                            "unit": temp_unit_json,
                        }

                limit_info = {}
                # Power limits
                limit_info["ppt0"] = power_limit_types["ppt0"]
                limit_info["ppt1"] = power_limit_types["ppt1"]

                # Shutdown limits
                limit_info["slowdown_edge_temperature"] = slowdown_temp_edge_limit
                limit_info["slowdown_hotspot_temperature"] = slowdown_temp_hotspot_limit
                limit_info["slowdown_vram_temperature"] = slowdown_temp_vram_limit
                limit_info["shutdown_edge_temperature"] = shutdown_temp_edge_limit
                limit_info["shutdown_hotspot_temperature"] = shutdown_temp_hotspot_limit
                limit_info["shutdown_vram_temperature"] = shutdown_temp_vram_limit

                # PTL
                limit_info["ptl_state"] = ptl_state
                limit_info["ptl_format"] = ptl_format

                static_dict["limit"] = limit_info
        if args.driver:
            driver_info_dict = {"name": "N/A", "version": "N/A", "os_kernel_version": "N/A"}

            try:
                driver_info = amdsmi_interface.amdsmi_get_gpu_driver_info(args.gpu)
                driver_info_dict["name"] = driver_info["driver_name"]
                driver_info_dict["version"] = driver_info["driver_version"]
            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug(
                    "Failed to get driver info for gpu %s | %s", gpu_id, e.get_error_info()
                )

            try:
                driver_info_dict["os_kernel_version"] = os.uname().release
            except (AttributeError, OSError) as e:
                logging.debug("Failed to get os kernel version for gpu %s | %s", gpu_id, e)

            static_dict["driver"] = driver_info_dict
        if args.board:
            static_dict["board"] = {
                "model_number": "N/A",
                "product_serial": "N/A",
                "fru_id": "N/A",
                "product_name": "N/A",
                "manufacturer_name": "N/A",
            }
            try:
                board_info = amdsmi_interface.amdsmi_get_gpu_board_info(args.gpu)
                for key, value in board_info.items():
                    if isinstance(value, str):
                        if value.strip() == "":
                            board_info[key] = "N/A"
                static_dict["board"] = board_info
            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug(
                    "Failed to get board info for gpu %s | %s", gpu_id, e.get_error_info()
                )
        if "ras" in current_platform_args:
            if args.ras:
                ras_dict = {
                    "eeprom_version": "N/A",
                    "bad_page_threshold": "N/A",
                    "bad_page_threshold_exceeded": "N/A",
                    "parity_schema": "N/A",
                    "single_bit_schema": "N/A",
                    "double_bit_schema": "N/A",
                    "poison_schema": "N/A",
                    "ecc_block_state": "N/A",
                }

                try:
                    ras_info = amdsmi_interface.amdsmi_get_gpu_ras_feature_info(args.gpu)
                    for key, value in ras_info.items():
                        if isinstance(value, int):
                            if value == 65535:
                                logging.debug(f"Failed to get ras {key} for gpu {gpu_id}")
                                ras_info[key] = "N/A"
                                continue
                        if key != "eeprom_version":
                            if value:
                                ras_info[key] = "ENABLED"
                            else:
                                ras_info[key] = "DISABLED"

                    ras_dict.update(ras_info)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug(
                        "Failed to get ras info for gpu %s | %s", gpu_id, e.get_error_info()
                    )
                try:
                    ras_dict["bad_page_threshold"] = (
                        amdsmi_interface.amdsmi_get_gpu_bad_page_threshold(args.gpu)
                    )
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug(
                        "Failed to get bad page threshold count for gpu %s | %s",
                        gpu_id,
                        e.get_error_info(),
                    )
                try:
                    bad_page_info = amdsmi_interface.amdsmi_get_gpu_bad_page_info(args.gpu)
                    retired_pages = 0
                    if bad_page_info:
                        for bad_page in bad_page_info:
                            if (
                                bad_page["status"]
                                == amdsmi_interface.AmdSmiMemoryPageStatus.RESERVED
                            ):
                                retired_pages += 1
                    # default to N/A
                    ras_dict["bad_page_threshold_exceeded"] = "N/A"
                    # If this is an int, then default to False
                    if isinstance(ras_dict["bad_page_threshold"], int):
                        ras_dict["bad_page_threshold_exceeded"] = "False"
                        if retired_pages > ras_dict["bad_page_threshold"]:
                            # If there are more retired pages then set to True
                            ras_dict["bad_page_threshold_exceeded"] = "True"
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug(
                        "Failed to get retired pages count for gpu %s | %s",
                        gpu_id,
                        e.get_error_info(),
                    )

                try:
                    ras_states = amdsmi_interface.amdsmi_get_gpu_ras_block_features_enabled(
                        args.gpu
                    )
                    ecc_block_state_dict = {}
                    for state in ras_states:
                        ecc_block_state_dict[state["block"]] = state["status"]
                    ras_dict["ecc_block_state"] = ecc_block_state_dict
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug(
                        "Failed to get ras block features for gpu %s | %s",
                        gpu_id,
                        e.get_error_info(),
                    )

                static_dict["ras"] = ras_dict
        if args.partition:
            try:
                compute_partition = amdsmi_interface.amdsmi_get_gpu_compute_partition(args.gpu)
            except amdsmi_exception.AmdSmiLibraryException as e:
                compute_partition = "N/A"
                logging.debug(
                    "Failed to get compute partition info for gpu %s | %s",
                    gpu_id,
                    e.get_error_info(),
                )
            try:
                memory_partition = amdsmi_interface.amdsmi_get_gpu_memory_partition(args.gpu)
            except amdsmi_exception.AmdSmiLibraryException as e:
                memory_partition = "N/A"
                logging.debug(
                    "Failed to get memory partition info for gpu %s | %s",
                    gpu_id,
                    e.get_error_info(),
                )
            try:
                kfd_info = amdsmi_interface.amdsmi_get_gpu_kfd_info(args.gpu)
                partition_id = kfd_info["current_partition_id"]
            except amdsmi_exception.AmdSmiLibraryException as e:
                partition_id = "N/A"
                logging.debug(
                    "Failed to get partition ID for gpu %s | %s", gpu_id, e.get_error_info()
                )
            static_dict["partition"] = {
                "accelerator_partition": compute_partition,
                "memory_partition": memory_partition,
                "partition_id": partition_id,
            }
        if "soc_pstate" in current_platform_args:
            if args.soc_pstate:
                try:
                    policy_info = amdsmi_interface.amdsmi_get_soc_pstate(args.gpu)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    policy_info = "N/A"
                    logging.debug(
                        "Failed to get soc pstate policy info for gpu %s | %s",
                        gpu_id,
                        e.get_error_info(),
                    )

                # Format for CSV output - flatten completely to avoid extra columns
                if self.logger.is_csv_format() and isinstance(policy_info, dict):
                    policies_str = (
                        ", ".join(
                            f"{p['policy_id']}:{p['policy_description']}"
                            for p in policy_info.get("policies", [])
                        )
                        or "N/A"
                    )

                    static_dict["num_supported"] = policy_info.get("num_supported", "N/A")
                    static_dict["current_id"] = policy_info.get("current_id", "N/A")
                    static_dict["policies"] = policies_str
                else:
                    static_dict["soc_pstate"] = policy_info
        if "xgmi_plpd" in current_platform_args:
            if args.xgmi_plpd:
                try:
                    policy_info = amdsmi_interface.amdsmi_get_xgmi_plpd(args.gpu)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    policy_info = "N/A"
                    logging.debug(
                        "Failed to get xgmi_plpd info for gpu %s | %s", gpu_id, e.get_error_info()
                    )

                # Format for CSV output - flatten completely to avoid extra columns
                if self.logger.is_csv_format() and isinstance(policy_info, dict):
                    policies_str = (
                        ", ".join(
                            f"{p['policy_id']}:{p['policy_description']}"
                            for p in policy_info.get("policies", [])
                        )
                        or "N/A"
                    )

                    static_dict["num_supported"] = policy_info.get("num_supported", "N/A")
                    static_dict["current_id"] = policy_info.get("current_id", "N/A")
                    static_dict["policies"] = policies_str
                else:
                    static_dict["xgmi_plpd"] = policy_info
        if "profile" in current_platform_args:
            if args.profile:
                try:
                    profile_status = amdsmi_interface.amdsmi_get_gpu_power_profile_presets(
                        args.gpu, 0
                    )

                    # Parse available profiles from bitfield
                    available_profiles = self.helpers.parse_available_profiles(
                        profile_status["available_profiles"]
                    )

                    # Get current profile name
                    current_profile = self.helpers.get_profile_name_from_mask(
                        profile_status["current"]
                    )

                    # Store output
                    static_dict["profile"] = {
                        "available_profiles": available_profiles,
                        "current": current_profile,
                        "num_profiles": profile_status["num_profiles"],
                    }
                except amdsmi_exception.AmdSmiLibraryException as e:
                    static_dict["profile"] = e.get_error_info()
                    logging.debug(
                        "Failed to get power profile info for gpu %s | %s",
                        gpu_id,
                        e.get_error_info(),
                    )
        if "process_isolation" in current_platform_args:
            if args.process_isolation:
                try:
                    status = amdsmi_interface.amdsmi_get_gpu_process_isolation(args.gpu)
                    status = "Enabled" if status else "Disabled"
                except amdsmi_exception.AmdSmiLibraryException as e:
                    status = "N/A"
                    logging.debug(
                        "Failed to process isolation for gpu %s | %s", gpu_id, e.get_error_info()
                    )

                static_dict["process_isolation"] = status
        if "numa" in current_platform_args:
            if args.numa:
                try:
                    numa_node_number = amdsmi_interface.amdsmi_topo_get_numa_node_number(args.gpu)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    numa_node_number = "N/A"
                    logging.debug(
                        "Failed to get numa node number for gpu %s | %s", gpu_id, e.get_error_info()
                    )

                try:
                    numa_affinity = amdsmi_interface.amdsmi_get_gpu_topo_numa_affinity(args.gpu)
                    # -1 means No numa node is assigned to the GPU, so there is no numa affinity
                    if self.logger.is_human_readable_format() and numa_affinity == -1:
                        numa_affinity = "NONE"
                except amdsmi_exception.AmdSmiLibraryException as e:
                    numa_affinity = "N/A"
                    logging.debug(
                        "Failed to get numa affinity for gpu %s | %s", gpu_id, e.get_error_info()
                    )

                try:
                    cpu_set = amdsmi_interface.amdsmi_get_cpu_affinity_with_scope(
                        args.gpu, amdsmi_interface.AmdSmiAffinityScope.NUMA_SCOPE
                    )
                    cpu_set = [f"{cpus:016X}" for cpus in cpu_set]
                    cpu_set = {f"cpu_list_{i}": f"{cpus}" for i, cpus in enumerate(cpu_set)}
                    bitmask_ranges = self.helpers.get_bitmask_ranges(cpu_set)
                    cpu_affinity = {}

                    for key in cpu_set:
                        cpu_affinity[key] = {
                            "bitmask": cpu_set[key],
                            "cpu_cores_affinity": bitmask_ranges[key],
                        }

                except amdsmi_exception.AmdSmiLibraryException as e:
                    cpu_affinity = "N/A"
                    logging.debug(
                        "Failed to get cpu affinity for gpu %s | %s", gpu_id, e.get_error_info()
                    )

                try:
                    socket_set = amdsmi_interface.amdsmi_get_cpu_affinity_with_scope(
                        args.gpu, amdsmi_interface.AmdSmiAffinityScope.SOCKET_SCOPE
                    )
                    socket_set = [f"{cpus:016X}" for cpus in socket_set]
                    socket_set = {f"cpu_list_{i}": f"{cpus}" for i, cpus in enumerate(socket_set)}
                    socket_bitmask_ranges = self.helpers.get_bitmask_ranges(socket_set)
                    socket_affinity = {}
                    for key in socket_set:
                        socket_affinity[key] = {
                            "bitmask": socket_set[key],
                            "cpu_cores_affinity": socket_bitmask_ranges.get(key, "N/A"),
                        }
                except amdsmi_exception.AmdSmiLibraryException as e:
                    socket_affinity = "N/A"
                    logging.debug(
                        "Failed to get socket affinity for gpu %s | %s", gpu_id, e.get_error_info()
                    )

                static_dict["numa"] = {
                    "node": numa_node_number,
                    "affinity": numa_affinity,
                    "cpu_affinity": cpu_affinity,
                    "socket_affinity": socket_affinity,
                }
        if args.vram:
            vram_info_dict = {
                "type": "N/A",
                "vendor": "N/A",
                "size": "N/A",
                "bit_width": "N/A",
                "max_bandwidth": "N/A",
            }
            try:
                vram_info = amdsmi_interface.amdsmi_get_gpu_vram_info(args.gpu)

                # Get vram type string
                vram_type_enum = vram_info["vram_type"]
                if vram_type_enum == amdsmi_interface.amdsmi_wrapper.AMDSMI_VRAM_TYPE__MAX:
                    vram_type = "GDDR7"
                else:
                    vram_type = amdsmi_interface.amdsmi_wrapper.amdsmi_vram_type_t__enumvalues[
                        vram_type_enum
                    ]
                    # Remove amdsmi enum prefix
                    vram_type = vram_type.replace("AMDSMI_VRAM_TYPE_", "").replace("_", "")

                # Get vram vendor string
                vram_vendor = vram_info["vram_vendor"]
                if "PLACEHOLDER" in vram_vendor:
                    vram_vendor = "N/A"

                # Assign cleaned values to vram_info_dict
                vram_info_dict["type"] = vram_type
                vram_info_dict["vendor"] = vram_vendor

                # Populate vram size with unit
                vram_info_dict["size"] = vram_info["vram_size"]
                vram_size_unit = "MB"
                if self.logger.is_human_readable_format():
                    vram_info_dict["size"] = f"{vram_info['vram_size']} {vram_size_unit}"

                if self.logger.is_json_format():
                    vram_info_dict["size"] = {
                        "value": vram_info["vram_size"],
                        "unit": vram_size_unit,
                    }

                # Populate bit width
                vram_info_dict["bit_width"] = vram_info["vram_bit_width"]

                # Populate vram_max_bandwidth
                vram_max_bw = vram_info["vram_max_bandwidth"]
                vram_max_bw_unit = "GB/s"
                if self.logger.is_human_readable_format():
                    vram_info_dict["max_bandwidth"] = (
                        f"{vram_max_bw} {vram_max_bw_unit if vram_max_bw != 'N/A' else ''}"
                    )
                if self.logger.is_json_format():
                    vram_info_dict["max_bandwidth"] = {
                        "value": vram_max_bw,
                        "unit": vram_max_bw_unit,
                    }

            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug("Failed to get vram info for gpu %s | %s", gpu_id, e.get_error_info())

            static_dict["vram"] = vram_info_dict
        if args.cache:
            try:
                cache_info_list = amdsmi_interface.amdsmi_get_gpu_cache_info(args.gpu)["cache"]
                logging.debug(f"cache_info dictionary = {cache_info_list}")

                for index, cache_info in enumerate(cache_info_list):
                    new_cache_info = {"cache": index}
                    new_cache_info.update(cache_info)
                    cache_info_list[index] = new_cache_info

                logging.debug(f"[after update] cache_info_list = {cache_info_list}")

                cache_size_unit = "KB"
                if self.logger.is_human_readable_format():
                    cache_info_dict_format = {}
                    for cache_dict in cache_info_list:
                        cache_index = "cache_" + str(cache_dict["cache"])
                        cache_info_dict_format[cache_index] = cache_dict

                        # Remove cache index from new dictionary
                        cache_info_dict_format[cache_index].pop("cache")

                        # Add cache_size unit
                        cache_size = (
                            f"{cache_info_dict_format[cache_index]['cache_size']} {cache_size_unit}"
                        )
                        cache_info_dict_format[cache_index]["cache_size"] = cache_size

                        # take cache_properties out of list -> display as string, removing brackets
                        cache_info_dict_format[cache_index]["cache_properties"] = ", ".join(
                            cache_info_dict_format[cache_index]["cache_properties"]
                        )

                    cache_info_list = cache_info_dict_format
                    logging.debug(f"[human readable] cache_info_list = {cache_info_list}")

                # Add cache_size_unit to json output
                if self.logger.is_json_format():
                    for cache_dict in cache_info_list:
                        cache_dict["cache_size"] = {
                            "value": cache_dict["cache_size"],
                            "unit": cache_size_unit,
                        }
            except amdsmi_exception.AmdSmiLibraryException as e:
                cache_info_list = "N/A"
                logging.debug(
                    "Failed to get cache info for gpu %s | %s", gpu_id, e.get_error_info()
                )

            static_dict["cache_info"] = cache_info_list

        if args.mem_carveout:
            try:
                uma_info = amdsmi_interface.amdsmi_get_gpu_uma_carveout_info(args.gpu)
                logging.debug(f"UMA carveout info: {uma_info}")

                if self.logger.is_json_format():
                    # JSON: show all options with current index
                    carveout_dict = {
                        "options": uma_info.get("options", []),
                        "current_index": uma_info.get("current_index", -1),
                    }
                    static_dict["mem_carveout"] = carveout_dict
                elif self.logger.is_csv_format():
                    # CSV: show only current index
                    static_dict["mem_carveout_index"] = uma_info.get("current_index", -1)
                else:
                    # Human readable: show all options with current marked
                    options = uma_info.get("options", [])
                    current_index = uma_info.get("current_index", -1)
                    if options:
                        formatted_options = []
                        for idx, option in enumerate(options):
                            marker = "*" if idx == current_index else " "
                            description = option.get("description", "N/A")
                            # Align indices: *[0] vs  [1]
                            formatted_options.append(f"    {marker}[{idx}] {description}")
                        static_dict["mem_carveout"] = "\n" + "\n".join(formatted_options)
                    else:
                        static_dict["mem_carveout"] = "N/A"
            except amdsmi_exception.AmdSmiLibraryException as e:
                # UMA carveout is only exposed by APU VBIOSes that support
                # ATCS function 0xA. On dGPUs and Instinct parts (including
                # MI300A) the sysfs attribute does not exist and the library
                # returns NOT_SUPPORTED; surface a clearer reason than bare N/A.
                not_supported = (
                    e.get_error_code()
                    == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED
                )
                if not_supported and not (
                    self.logger.is_json_format() or self.logger.is_csv_format()
                ):
                    # Human-readable: give a descriptive reason. JSON/CSV
                    # consumers keep the legacy bare "N/A" for back-compat.
                    static_dict["mem_carveout"] = (
                        "N/A (UMA carveout is not supported on this ASIC/VBIOS)"
                    )
                elif self.logger.is_csv_format():
                    static_dict["mem_carveout_index"] = "N/A"
                else:
                    static_dict["mem_carveout"] = "N/A"
                logging.debug(
                    "Failed to get mem carveout info for gpu %s | %s", gpu_id, e.get_error_info()
                )

        # default to printing all clocks, if in current_platform_args; otherwise print specific clocks
        if "clock" in current_platform_args and (
            args.clock == True or isinstance(args.clock, list)
        ):
            original_clock_args = (
                args.clock
            )  # save original args.clock value, so we can reset for multiple devices
            if isinstance(args.clock, bool):
                args.clock = ["sys", "mem", "df", "soc", "dcef", "vclk0", "vclk1", "dclk0", "dclk1"]

            if isinstance(args.clock, list):
                # remove potential duplicates from list
                args.clock = list(set(args.clock))
                # check that clock is valid option
                if "all" in args.clock or len(args.clock) == 0:
                    args.clock = [
                        "sys",
                        "mem",
                        "df",
                        "soc",
                        "dcef",
                        "vclk0",
                        "vclk1",
                        "dclk0",
                        "dclk1",
                    ]

                clk_dict = {
                    "sys": "N/A",
                    "mem": "N/A",
                    "df": "N/A",
                    "soc": "N/A",
                    "dcef": "N/A",
                    "vclk0": "N/A",
                    "vclk1": "N/A",
                    "dclk0": "N/A",
                    "dclk1": "N/A",
                }
                for clk in list(clk_dict.keys()):
                    if clk not in args.clock:
                        del clk_dict[clk]
                for clk in args.clock:
                    if clk in self.helpers.convert_clock_type:
                        clk_type_conversion = self.helpers.convert_clock_type[clk]
                    else:
                        clk_type_conversion = "N/A"
                        output_format = self.helpers.get_output_format()
                        raise AmdSmiInvalidParameterException(
                            "static", clk_type, output_format
                        )  # clk type given is bad

                    try:
                        frequencies = amdsmi_interface.amdsmi_get_clk_freq(
                            args.gpu, clk_type_conversion
                        )
                        # some clocks may have a sysfs file but no frequencies for whatever reason.
                        if len(frequencies["frequency"]) == 0:
                            freq_dict = "N/A"
                            continue
                        freq_dict = {}
                        current_level = frequencies["current"]
                        # The C library reports current = (uint32_t)-1 when the
                        # kernel exposes the pp_dpm_* table without a '*'
                        # current-level marker (e.g. SMU power-gated domains
                        # on gfx1151 APUs at idle). Surface that as N/A so
                        # the CLI doesn't show 4294967295 to users.
                        if current_level == 0xFFFFFFFF:
                            current_level = "N/A"
                        # Add current_level first for proper output ordering
                        freq_dict.update({"current_level": current_level})
                        # Add frequency_levels second
                        freq_dict.update({"frequency_levels": {}})
                        if frequencies["num_supported"] != 0:
                            for level in range(len(frequencies["frequency"])):
                                if frequencies["frequency"][level] != "N/A":
                                    freq = str(
                                        self.helpers.convert_SI_unit(
                                            frequencies["frequency"][level],
                                            AMDSMIHelpers.SI_Unit.MICRO,
                                        )
                                    )
                                    freq_dict["frequency_levels"].update(
                                        {f"Level {level}": {"value": freq, "unit": "MHz"}}
                                    )
                                else:
                                    freq_dict["frequency_levels"].update({f"Level {level}": "N/A"})
                        else:
                            freq_dict = "N/A"
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        freq_dict = "N/A"
                        logging.debug(
                            "Failed to get clock info for gpu %s | %s", gpu_id, e.get_error_info()
                        )
                    clk_dict[clk] = freq_dict

                static_dict["clock"] = clk_dict
            else:
                raise amdsmi_exception.AmdSmiParameterException(args.clock, "list[str]")
            # if original_clock_args is a boolean, set it back to the original value
            if isinstance(original_clock_args, bool):
                args.clock = original_clock_args

        # Convert and store output by pid for csv format
        multiple_devices_csv_override = False
        if self.logger.is_csv_format():
            # For NUMA data - flatten CPU affinity lists
            if "numa" in static_dict and isinstance(static_dict["numa"], dict):
                numa_data = static_dict.pop("numa")
                multiple_devices_csv_override = True

                # Get data
                node = numa_data.get("node", "N/A")
                affinity = numa_data.get("affinity", "N/A")
                cpu_affinity = numa_data.get("cpu_affinity", {})
                socket_affinity = numa_data.get("socket_affinity", {})
                # Create a flattened row for list entry
                row_dict = static_dict.copy()

                if cpu_affinity and isinstance(cpu_affinity, dict):
                    for cpu_list_key in cpu_affinity.keys():
                        cpu_entry = cpu_affinity[cpu_list_key]
                        socket_entry = socket_affinity.get(
                            cpu_list_key, {"bitmask": "N/A", "cpu_cores_affinity": "N/A"}
                        )
                        row_dict.update(
                            {
                                "node": node,
                                "affinity": affinity,
                                "cpu_list": cpu_list_key,
                                "bitmask": cpu_entry.get("bitmask"),
                                "cpu_cores_affinity": cpu_entry.get("cpu_cores_affinity"),
                                "socket_bitmask": socket_entry.get("bitmask"),
                                "socket_cpu_cores_affinity": socket_entry.get("cpu_cores_affinity"),
                            }
                        )
                        self.logger.store_output(args.gpu, "values", row_dict)
                        self.logger.store_gpu_json_output.append(row_dict)
                        self.logger.store_multiple_device_output()
                else:
                    row_dict.update(
                        {
                            "node": node,
                            "affinity": affinity,
                            "cpu_list": "N/A",
                            "bitmask": "N/A",
                            "cpu_cores_affinity": "N/A",
                            "socket_bitmask": "N/A",
                            "socket_cpu_cores_affinity": "N/A",
                        }
                    )
                    self.logger.store_output(args.gpu, "values", row_dict)
                    self.logger.store_gpu_json_output.append(row_dict)
                    self.logger.store_multiple_device_output()
            # expand if ras blocks are populated
            elif self.helpers.is_linux() and self.helpers.is_baremetal() and args.ras:
                if isinstance(static_dict["ras"]["ecc_block_state"], list):
                    ecc_block_dicts = static_dict["ras"].pop("ecc_block_state")
                    multiple_devices_csv_override = True
                    for ecc_block_dict in ecc_block_dicts:
                        for key, value in ecc_block_dict.items():
                            self.logger.store_output(args.gpu, key, value)
                        self.logger.store_output(args.gpu, "values", static_dict)
                        self.logger.store_gpu_json_output.append(static_dict)
                        self.logger.store_multiple_device_output()
                else:
                    # Store values if ras has an error
                    self.logger.store_output(args.gpu, "values", static_dict)
                    self.logger.store_gpu_json_output.append(static_dict)
            else:
                self.logger.store_output(args.gpu, "values", static_dict)
                self.logger.store_gpu_json_output.append(static_dict)
        elif self.logger.is_json_format():
            self.logger.store_gpu_json_output.append(static_dict)
        else:
            # Store values in logger.output
            self.logger.store_output(args.gpu, "values", static_dict)

        if multiple_devices:
            self.logger.store_multiple_device_output()
            return  # Skip printing when there are multiple devices
        if not self.logger.is_json_format():
            self.logger.print_output(multiple_device_enabled=multiple_devices_csv_override)

    def _filter_nics_from_args(subcommand):
        @functools.wraps(subcommand)
        def wrapper(self, *args, **kwargs):
            original_nic = None
            if len(args) > 0:
                original_nic = args[0].nic
                nics, ainics = self._get_nics_from_args(args[0])
                if len(nics) == 0:
                    args[0].nic = None
                else:
                    args[0].nic = nics
            result = subcommand(self, *args, **kwargs)
            if len(args) > 0:
                args[0].nic = original_nic
            return result

        return wrapper

    @_filter_nics_from_args
    def _static_brcm_nic(self, args, multiple_devices=False, nic=None):
        """Get Static information for target nic

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            nic (device_handle, optional): device_handle for target device. Defaults to None.

        Returns:
            None: Print output via AMDSMILogger to destination
        """

        if nic:
            args.nic = nic

        # Handle multiple NICs
        handled_multiple_nics, device_handle = self.helpers.handle_brcm_nics(
            args, self.logger, self._static_brcm_nic
        )
        if handled_multiple_nics:
            return  # This function is recursive
        args.nic = device_handle
        if not args.nic:
            return

        # Get nic id for logging
        nic_id = self.helpers.get_nic_id_from_device_handle(args.nic)
        logging.debug(f"Static Arg information for NIC {nic_id} on {self.helpers.os_info()}")

        static_dict = {}
        if self.logger.is_json_format():
            static_dict["ai_nic"] = int(nic_id)

        if args.nic:
            try:
                nic_info = amdsmi_interface.amdsmi_get_nic_info(args.nic)
                if nic_info:
                    static_dict["nic"] = {
                        "bdf": f"{nic_info['bdf']}",
                        "UUID": f"{nic_info['UUID']}",
                        "Device Name": f"{nic_info['Device Name']}",
                        "Part Number": f"{nic_info['Part Number']}",
                        "Firmware_Version": f"{nic_info['Firmware_Version']}",
                    }
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["nic"] = "N/A"
                logging.debug("Failed to get NIC %s | %s", nic_id, e.get_error_info())

        multiple_devices_csv_override = False
        if not self.logger.is_json_format():
            self.logger.store_nic_output(args.nic, "values", static_dict)
        else:
            self.logger.store_nic_json_output.append(static_dict)
        if multiple_devices:
            self.logger.store_multiple_device_output()
            return  # Skip printing when there are multiple devices
        if not self.logger.is_json_format():
            self.logger.print_output(multiple_device_enabled=multiple_devices_csv_override)

    def _filter_ainics_from_args(subcommand):
        @functools.wraps(subcommand)
        def wrapper(self, *args, **kwargs):
            original_nic = None
            if len(args) > 0:
                original_nic = args[0].nic
                nics, ainics = self._get_nics_from_args(args[0])
                if len(ainics) == 0:
                    args[0].nic = None
                else:
                    args[0].nic = ainics
            result = subcommand(self, *args, **kwargs)
            if len(args) > 0:
                args[0].nic = original_nic
            return result

        return wrapper

    @_filter_ainics_from_args
    def _static_ainic(self, args, multiple_devices=False, nic=None):
        """Get Static information for target ainic

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            nic (device_handle, optional): device_handle for target device. Defaults to None.

        Returns:
            None: Print output via AMDSMILogger to destination
        """

        if nic:
            args.nic = nic

        # Handle multiple NICs
        handled_multiple_nics, device_handle = self.helpers.handle_ainics(
            args, self.logger, self._static_ainic
        )
        if handled_multiple_nics:
            return  # This function is recursive
        args.nic = device_handle
        if not args.nic:
            return

        # Get nic id for logging
        nic_id = self.helpers.get_ainic_id_from_device_handle(args.nic)
        logging.debug(f"Static Arg information for NIC {nic_id} on {self.helpers.os_info()}")

        static_dict = {}
        if self.logger.is_json_format():
            static_dict["ai_nic"] = int(nic_id)

        if args.nic:
            try:
                nic_info = amdsmi_interface.amdsmi_get_ainic_info(args.nic, True)
                info_filter = []
                if hasattr(args, "asic") and getattr(args, "asic"):
                    info_filter.append("asic")
                if hasattr(args, "bus") and getattr(args, "bus"):
                    info_filter.append("bus")
                if hasattr(args, "driver") and getattr(args, "driver"):
                    info_filter.append("driver")
                if hasattr(args, "numa") and getattr(args, "numa"):
                    info_filter.append("numa")
                if len(info_filter) == 0 or len(info_filter) == 4:
                    static_dict["nic"] = nic_info
                else:
                    nic_info_filtered = {}
                    for attr in info_filter:  # remove all attributes except the one in info_filter:
                        nic_info_filtered = nic_info_filtered | {
                            key: value for key, value in nic_info.items() if key.lower() == attr
                        }
                    static_dict["nic"] = nic_info_filtered
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["nic"] = "N/A"
                logging.debug("Failed to get NIC %s | %s", nic_id, e.get_error_info())

        multiple_devices_csv_override = False
        if not self.logger.is_json_format():
            self.logger.store_ainic_output(args.nic, "values", static_dict)
        else:
            self.logger.store_nic_json_output.append(static_dict)
        if multiple_devices:
            self.logger.store_multiple_device_output()
            return  # Skip printing when there are multiple devices
        if not self.logger.is_json_format():
            self.logger.print_output(multiple_device_enabled=multiple_devices_csv_override)

    def _static_nics(self, args, multiple_devices, nic):
        if hasattr(args, "nic") and args.nic == None:
            nic = None
            if self.helpers.is_ainic_initialized():
                nic = self.device_handles_ainics
            if self.helpers.is_brcm_nic_initialized():
                nic = self.device_handles_brcm_nics
            args.nic = nic
            return False
        else:
            if (
                not self.helpers.is_ainic_initialized()
                and not self.helpers.is_brcm_nic_initialized()
            ):
                return False
            self.logger.output = {}
            self.logger.clear_multiple_devices_output()
            if self.helpers.is_ainic_initialized():
                self._static_ainic(args, multiple_devices, nic)
            if self.helpers.is_brcm_nic_initialized():
                self._static_brcm_nic(args, multiple_devices, nic)
            return True

    def static(
        self,
        args,
        multiple_devices=False,
        gpu=None,
        asic=None,
        bus=None,
        vbios=None,
        limit=None,
        driver=None,
        ras=None,
        board=None,
        numa=None,
        vram=None,
        cache=None,
        partition=None,
        dfc_ucode=None,
        fb_info=None,
        num_vf=None,
        cpu=None,
        nic=None,
        interface_ver=None,
        soc_pstate=None,
        xgmi_plpd=None,
        process_isolation=None,
        clock=None,
        profile=None,
        mem_carveout=None,
    ):
        """Get Static information for target gpu and cpu

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            gpu (device_handle, optional): device_handle for target device. Defaults to None.
            asic (bool, optional): Value override for args.asic. Defaults to None.
            bus (bool, optional): Value override for args.bus. Defaults to None.
            vbios (bool, optional): Value override for args.vbios. Defaults to None.
            limit (bool, optional): Value override for args.limit. Defaults to None.
            driver (bool, optional): Value override for args.driver. Defaults to None.
            ras (bool, optional): Value override for args.ras. Defaults to None.
            board (bool, optional): Value override for args.board. Defaults to None.
            numa (bool, optional): Value override for args.numa. Defaults to None.
            vram (bool, optional): Value override for args.vram. Defaults to None.
            cache (bool, optional): Value override for args.cache. Defaults to None.
            partition (bool, optional): Value override for args.partition. Defaults to None.
            dfc_ucode (bool, optional): Value override for args.dfc_ucode. Defaults to None.
            fb_info (bool, optional): Value override for args.fb_info. Defaults to None.
            num_vf (bool, optional): Value override for args.num_vf. Defaults to None.
            cpu (cpu_handle, optional): cpu_handle for target device. Defaults to None.
            nic (nic_handle, optional): nic_handle for target device. Defaults to None.
            interface_ver (bool, optional): Value override for args.interface_ver. Defaults to None
            soc_pstate (bool, optional): Value override for args.soc_pstate. Defaults to None.
            xgmi_plpd (bool, optional): Value override for args.xgmi_plpd. Defaults to None.
            process_isolation (bool, optional): Value override for args.process_isolation. Defaults to None.
        Raises:
            IndexError: Index error if gpu list is empty

        Returns:
            None: Print output via AMDSMILogger to destination
        """
        # Mutually exclusive arguments
        if cpu:
            args.cpu = cpu
        if gpu:
            args.gpu = gpu
        if nic:
            args.nic = nic

        if self._static_nics(args, multiple_devices, nic):
            return True  # we do not want to print cpu or gpu if user only wanted nic

        if (hasattr(args, "cpu") and args.cpu) or (hasattr(args, "gpu") and args.gpu):
            args.nic = None  # we do not want to output nic at the end if user wants only cpu or gpu

        # Check if a CPU argument has been set
        cpu_args_enabled = False
        cpu_attributes = ["smu", "interface_ver"]
        for attr in cpu_attributes:
            if hasattr(args, attr):
                if getattr(args, attr):
                    cpu_args_enabled = True
                    break

        # Check if a GPU argument has been set
        gpu_args_enabled = False
        gpu_attributes = [
            "asic",
            "bus",
            "vbios",
            "limit",
            "driver",
            "ras",
            "board",
            "numa",
            "vram",
            "cache",
            "partition",
            "dfc_ucode",
            "fb_info",
            "num_vf",
            "soc_pstate",
            "xgmi_plpd",
            "process_isolation",
            "clock",
            "profile",
            "mem_carveout",
        ]
        for attr in gpu_attributes:
            if hasattr(args, attr):
                if getattr(args, attr):
                    gpu_args_enabled = True
                    break

        # Handle CPU and GPU initialization cases
        if self.helpers.is_amd_hsmp_initialized() and self.helpers.is_amdgpu_initialized():
            # Print out all CPU and all GPU static info only if no device was specified.
            # If a GPU or CPU argument is provided only print out the specified device.
            if args.cpu == None and args.gpu == None:
                if not cpu_args_enabled and not gpu_args_enabled:
                    args.cpu = self.cpu_handles
                    args.gpu = self.device_handles

            # Handle cases where the user has only specified an argument and no specific device
            if args.gpu == None and gpu_args_enabled:
                args.gpu = self.device_handles
            if args.cpu == None and cpu_args_enabled:
                args.cpu = self.cpu_handles

            if args.cpu:
                self.static_cpu(args, multiple_devices, cpu, interface_ver)
            if args.gpu:
                self.logger.output = {}
                self.logger.clear_multiple_devices_output()
                self.static_gpu(
                    args,
                    multiple_devices,
                    gpu,
                    asic,
                    bus,
                    vbios,
                    limit,
                    driver,
                    ras,
                    board,
                    numa,
                    vram,
                    cache,
                    partition,
                    dfc_ucode,
                    fb_info,
                    num_vf,
                    soc_pstate,
                    xgmi_plpd,
                    process_isolation,
                    clock,
                    profile,
                    mem_carveout,
                )
        elif self.helpers.is_amd_hsmp_initialized():  # Only CPU is initialized
            if args.cpu == None:
                args.cpu = self.cpu_handles

            self.static_cpu(args, multiple_devices, cpu, interface_ver)
        elif self.helpers.is_amdgpu_initialized():  # Only GPU is initialized
            if args.gpu == None:
                args.gpu = self.device_handles

            self.logger.clear_multiple_devices_output()
            self.static_gpu(
                args,
                multiple_devices,
                gpu,
                asic,
                bus,
                vbios,
                limit,
                driver,
                ras,
                board,
                numa,
                vram,
                cache,
                partition,
                dfc_ucode,
                fb_info,
                num_vf,
                soc_pstate,
                xgmi_plpd,
                process_isolation,
                clock,
                profile,
                mem_carveout,
            )

        if hasattr(args, "nic") and args.nic:
            self.logger.output = {}
            self.logger.clear_multiple_devices_output()
            self._static_ainic(args, multiple_devices, nic)
            self._static_brcm_nic(args, multiple_devices, nic)

        if self.logger.is_json_format():
            self.logger.combine_arrays_to_json()
