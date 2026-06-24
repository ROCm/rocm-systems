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
import time

from amdsmi import amdsmi_exception, amdsmi_interface
from amdsmi.amdsmi_interface import AMDSMI_MAX_RAIL_INDEX


class MetricCommands:
    def metric_gpu(
        self,
        args,
        multiple_devices=False,
        watching_output=False,
        gpu=None,
        usage=None,
        watch=None,
        watch_time=None,
        iterations=None,
        power=None,
        clock=None,
        temperature=None,
        ecc=None,
        ecc_blocks=None,
        pcie=None,
        fan=None,
        voltage_curve=None,
        overdrive=None,
        perf_level=None,
        xgmi_err=None,
        energy=None,
        mem_usage=None,
        voltage=None,
        schedule=None,
        guard=None,
        guest_data=None,
        fb_usage=None,
        xgmi=None,
        throttle=None,
        base_board=None,
        gpu_board=None,
        partition=None,
    ):
        """Get Metric information for target gpu

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            watching_output (bool, optional): True if watch argument has been set. Defaults to False.
            gpu (device_handle, optional): device_handle for target device. Defaults to None.
            usage (bool, optional): Value override for args.usage. Defaults to None.
            watch (Positive int, optional): Value override for args.watch. Defaults to None.
            watch_time (Positive int, optional): Value override for args.watch_time. Defaults to None.
            iterations (Positive int, optional): Value override for args.iterations. Defaults to None.
            power (bool, optional): Value override for args.power. Defaults to None.
            clock (bool, optional): Value override for args.clock. Defaults to None.
            temperature (bool, optional): Value override for args.temperature. Defaults to None.
            ecc (bool, optional): Value override for args.ecc. Defaults to None.
            ecc_blocks (bool, optional): Value override for args.ecc. Defaults to None.
            pcie (bool, optional): Value override for args.pcie. Defaults to None.
            fan (bool, optional): Value override for args.fan. Defaults to None.
            voltage_curve (bool, optional): Value override for args.voltage_curve. Defaults to None.
            overdrive (bool, optional): Value override for args.overdrive. Defaults to None.
            perf_level (bool, optional): Value override for args.perf_level. Defaults to None.
            xgmi_err (bool, optional): Value override for args.xgmi_err. Defaults to None.
            energy (bool, optional): Value override for args.energy. Defaults to None.
            mem_usage (bool, optional): Value override for args.mem_usage. Defaults to None.
            voltage (bool, optional): Value override for args.voltage. Defaults to None.
            schedule (bool, optional): Value override for args.schedule. Defaults to None.
            guard (bool, optional): Value override for args.guard. Defaults to None.
            guest_data (bool, optional): Value override for args.guest_data. Defaults to None.
            fb_usage (bool, optional): Value override for args.fb_usage. Defaults to None.
            xgmi (bool, optional): Value override for args.xgmi. Defaults to None.
            throttle (bool, optional): Value override for args.throttle. Defaults to None.
            partition (bool, optional): Value override for args.partition. Defaults to None.

        Raises:
            IndexError: Index error if gpu list is empty

        Returns:
            None: Print output via AMDSMILogger to destination
        """
        # Set args.* to passed in arguments
        if gpu:
            args.gpu = gpu
        if watch:
            args.watch = watch
        if watch_time:
            args.watch_time = watch_time
        if iterations:
            args.iterations = iterations

        # Store args that are applicable to the current platform
        current_platform_args = []
        current_platform_values = []

        if not self.helpers.is_hypervisor() and not self.helpers.is_windows():
            if mem_usage:
                args.mem_usage = mem_usage
            current_platform_args += ["mem_usage"]
            current_platform_values += [args.mem_usage]

        if self.helpers.is_hypervisor() or self.helpers.is_baremetal() or self.helpers.is_linux():
            if usage:
                args.usage = usage
            if base_board:
                args.base_board = base_board
            if gpu_board:
                args.gpu_board = gpu_board
            if partition:
                args.partition = partition
            if power:
                args.power = power
            if clock:
                args.clock = clock
            if temperature:
                args.temperature = temperature
            if voltage:
                args.voltage = voltage
            if pcie:
                args.pcie = pcie
            if ecc:
                args.ecc = ecc
            if ecc_blocks:
                args.ecc_blocks = ecc_blocks
            current_platform_args += [
                "usage",
                "power",
                "clock",
                "temperature",
                "voltage",
                "pcie",
                "ecc",
                "ecc_blocks",
                "base_board",
                "gpu_board",
            ]
            current_platform_values += [
                args.usage,
                args.power,
                args.clock,
                args.temperature,
                args.voltage,
                args.pcie,
            ]
            current_platform_values += [args.ecc, args.ecc_blocks, args.base_board, args.gpu_board]

        if self.helpers.is_baremetal() and self.helpers.is_linux():
            if fan:
                args.fan = fan
            if voltage_curve:
                args.voltage_curve = voltage_curve
            if overdrive:
                args.overdrive = overdrive
            if perf_level:
                args.perf_level = perf_level
            if xgmi_err:
                args.xgmi_err = xgmi_err
            if energy:
                args.energy = energy
            if throttle:
                args.violation = throttle
                args.throttle = throttle
            current_platform_args += [
                "fan",
                "voltage_curve",
                "overdrive",
                "perf_level",
                "xgmi_err",
                "energy",
                "throttle",
            ]
            current_platform_values += [
                args.fan,
                args.voltage_curve,
                args.overdrive,
                args.perf_level,
                args.xgmi_err,
                args.energy,
                args.throttle,
            ]

        if self.helpers.is_hypervisor():
            if schedule:
                args.schedule = schedule
            if guard:
                args.guard = guard
            if guest_data:
                args.guest_data = guest_data
            if fb_usage:
                args.fb_usage = fb_usage
            if xgmi:
                args.xgmi = xgmi
            current_platform_args += ["schedule", "guard", "guest_data", "fb_usage", "xgmi"]
            current_platform_values += [
                args.schedule,
                args.guard,
                args.guest_data,
                args.fb_usage,
                args.xgmi,
            ]

        # Handle No GPU passed
        if args.gpu == None:
            args.gpu = self.device_handles

        if not self.group_check_printed:
            self.helpers.check_required_groups()
            self.group_check_printed = True

        # Handle watch logic, will only enter this block once
        if args.watch:
            self.helpers.handle_watch(args=args, subcommand=self.metric_gpu, logger=self.logger)
            return

        # Handle multiple GPUs
        if isinstance(args.gpu, list):
            if len(args.gpu) > 1:
                # Deepcopy gpus as recursion will destroy the gpu list
                stored_gpus = []
                for gpu in args.gpu:
                    stored_gpus.append(gpu)

                # Store output from multiple devices
                for device_handle in args.gpu:
                    self.metric_gpu(
                        args,
                        multiple_devices=True,
                        watching_output=watching_output,
                        gpu=device_handle,
                    )

                # Reload original gpus
                args.gpu = stored_gpus

                # Print multiple device output
                if not self.logger.is_json_format() or watching_output:
                    self.logger.print_output(
                        multiple_device_enabled=True, watching_output=watching_output
                    )

                # Add output to total watch output and clear multiple device output
                if watching_output:
                    self.logger.store_watch_output(multiple_device_enabled=True)

                return
            elif len(args.gpu) == 1:
                args.gpu = args.gpu[0]
            else:
                raise IndexError("args.gpu should not be an empty list")

        # Get gpu_id for logging
        gpu_id = self.helpers.get_gpu_id_from_device_handle(args.gpu)

        if args.loglevel == "DEBUG":
            try:
                # Get GPU Metrics table version
                gpu_metric_version_info = amdsmi_interface.amdsmi_get_gpu_metrics_header_info(
                    args.gpu
                )
                gpu_metric_version_str = json.dumps(gpu_metric_version_info, indent=4)
                logging.debug(
                    "GPU Metrics table Version for GPU %s | %s", gpu_id, gpu_metric_version_str
                )
            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug(
                    "#1 - Unable to load GPU Metrics table version for %s | %s",
                    gpu_id,
                    e.get_error_info(),
                )

            try:
                # Get GPU Metrics table
                gpu_metric_debug_info = amdsmi_interface.amdsmi_get_gpu_metrics_info(args.gpu)
                gpu_metric_str = json.dumps(gpu_metric_debug_info, indent=4)
                logging.debug("GPU Metrics table for GPU %s | %s", gpu_id, str(gpu_metric_str))
            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug(
                    "#2 - Unable to load GPU Metrics table for %s | %s", gpu_id, e.get_error_info()
                )

        logging.debug(f"Metric Arg information for GPU {gpu_id} on {self.helpers.os_info()}")
        logging.debug(f"Args:   {current_platform_args}")
        logging.debug(f"Values: {current_platform_values}")

        # Set the platform applicable args to True if no args are set
        if not any(current_platform_values):
            for arg in current_platform_args:
                setattr(args, arg, True)

        # Add timestamp and store values for specified arguments
        values_dict = {}

        is_partition_metrics = False  # True if we get the metrics from xcp_metrics file (amdsmi_get_gpu_partition_metrics_info)
        # get metric info only once per gpu, this will speed up data output
        try:
            # Get GPU Metrics table
            gpu_metric = amdsmi_interface.amdsmi_get_gpu_metrics_info(args.gpu)
        except amdsmi_exception.AmdSmiLibraryException as e:
            logging.debug(
                "#3 - Unable to load GPU Metrics table for %s | %s", gpu_id, e.get_error_info()
            )
            gpu_metric = amdsmi_interface._NA_amdsmi_get_gpu_metrics_info()

        # Workaround for XCP (partition) metrics not providing num_partition in v1.9+/v1.1+
        # Provides original formatting for earlier metric versions
        partition_metric_info = self.helpers._get_metric_version_and_partition_info(
            gpu_metric, is_partition_metrics, gpu_id, args.gpu
        )
        num_partition = partition_metric_info["num_partition"]

        # Fetch partition metrics once per GPU; the sections below reuse this result
        gpu_partition_metrics = None
        if args.partition:
            try:
                gpu_partition_metrics = amdsmi_interface.amdsmi_get_gpu_partition_metrics_info(
                    args.gpu
                )
            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug(
                    "Failed to get partition metrics for gpu %s | %s", gpu_id, e.get_error_info()
                )

        if self.logger.is_json_format():
            values_dict["gpu"] = int(gpu_id)
        # Populate the pcie_dict first due to multiple gpu metrics calls incorrectly increasing bandwidth
        if "pcie" in current_platform_args:
            if args.pcie:
                pcie_dict = {
                    "width": "N/A",
                    "speed": "N/A",
                    "bandwidth": "N/A",
                    "replay_count": "N/A",
                    "l0_to_recovery_count": "N/A",
                    "replay_roll_over_count": "N/A",
                    "nak_sent_count": "N/A",
                    "nak_received_count": "N/A",
                    "current_bandwidth_sent": "N/A",
                    "current_bandwidth_received": "N/A",
                    "max_packet_size": "N/A",
                    "lc_perf_other_end_recovery_count": "N/A",
                }

                try:
                    pcie_metric = amdsmi_interface.amdsmi_get_pcie_info(args.gpu)["pcie_metric"]
                    logging.debug("PCIE Metric for %s | %s", gpu_id, pcie_metric)

                    pcie_dict["width"] = pcie_metric["pcie_width"]

                    if pcie_metric["pcie_speed"] != "N/A":
                        if pcie_metric["pcie_speed"] % 1000 != 0:
                            pcie_speed_GTs_value = round(pcie_metric["pcie_speed"] / 1000, 1)
                        else:
                            pcie_speed_GTs_value = round(pcie_metric["pcie_speed"] / 1000)
                        pcie_dict["speed"] = pcie_speed_GTs_value

                    pcie_dict["bandwidth"] = pcie_metric["pcie_bandwidth"]

                    pcie_dict["replay_count"] = pcie_metric["pcie_replay_count"]
                    if pcie_dict["replay_count"] == "N/A":
                        try:
                            pcie_replay = amdsmi_interface.amdsmi_get_gpu_pci_replay_counter(
                                args.gpu
                            )
                            pcie_dict["replay_count"] = pcie_replay
                        except amdsmi_exception.AmdSmiLibraryException as e:
                            logging.debug(
                                "Failed to get sysfs pcie replay counter on gpu %s | %s",
                                gpu_id,
                                e.get_error_info(),
                            )

                    pcie_dict["l0_to_recovery_count"] = pcie_metric["pcie_l0_to_recovery_count"]
                    pcie_dict["replay_roll_over_count"] = pcie_metric["pcie_replay_roll_over_count"]
                    pcie_dict["nak_received_count"] = pcie_metric["pcie_nak_received_count"]
                    pcie_dict["nak_sent_count"] = pcie_metric["pcie_nak_sent_count"]
                    pcie_dict["lc_perf_other_end_recovery_count"] = pcie_metric[
                        "pcie_lc_perf_other_end_recovery_count"
                    ]

                    pcie_speed_unit = "GT/s"
                    pcie_bw_unit = "Mb/s"
                    if self.logger.is_human_readable_format():
                        if pcie_dict["speed"] != "N/A":
                            pcie_dict["speed"] = f"{pcie_dict['speed']} {pcie_speed_unit}"
                        if pcie_dict["bandwidth"] != "N/A":
                            pcie_dict["bandwidth"] = f"{pcie_dict['bandwidth']} {pcie_bw_unit}"
                    if self.logger.is_json_format():
                        if pcie_dict["speed"] != "N/A":
                            pcie_dict["speed"] = {
                                "value": pcie_dict["speed"],
                                "unit": pcie_speed_unit,
                            }
                        if pcie_dict["bandwidth"] != "N/A":
                            pcie_dict["bandwidth"] = {
                                "value": pcie_dict["bandwidth"],
                                "unit": pcie_bw_unit,
                            }
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug(
                        "Failed to get pcie link status for gpu %s | %s", gpu_id, e.get_error_info()
                    )

                try:
                    pcie_bw = amdsmi_interface.amdsmi_get_gpu_pci_throughput(args.gpu)
                    sent = pcie_bw["sent"] * pcie_bw["max_pkt_sz"]
                    received = pcie_bw["received"] * pcie_bw["max_pkt_sz"]

                    bw_unit = "Mb/s"
                    packet_size_unit = "B"
                    if sent > 0:
                        sent = sent // 1024 // 1024
                    if received > 0:
                        received = received // 1024 // 1024

                    if self.logger.is_human_readable_format():
                        sent = f"{sent} {bw_unit}"
                        received = f"{received} {bw_unit}"
                        pcie_bw["max_pkt_sz"] = f"{pcie_bw['max_pkt_sz']} {packet_size_unit}"
                    if self.logger.is_json_format():
                        sent = {"value": sent, "unit": bw_unit}
                        received = {"value": received, "unit": bw_unit}
                        pcie_bw["max_pkt_sz"] = {
                            "value": pcie_bw["max_pkt_sz"],
                            "unit": packet_size_unit,
                        }

                    pcie_dict["current_bandwidth_sent"] = sent
                    pcie_dict["current_bandwidth_received"] = received
                    pcie_dict["max_packet_size"] = pcie_bw["max_pkt_sz"]
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug(
                        "Failed to get pcie bandwidth for gpu %s | %s", gpu_id, e.get_error_info()
                    )

        if "usage" in current_platform_args:
            if args.usage:
                try:
                    engine_usage = amdsmi_interface.amdsmi_get_gpu_activity(args.gpu)
                    logging.debug(f"engine_usage dictionary = {engine_usage}")

                    # TODO: move vcn_activity and jpeg_activity into amdsmi_get_gpu_activity
                    engine_usage["vcn_activity"] = gpu_metric["vcn_activity"]
                    engine_usage["jpeg_activity"] = gpu_metric["jpeg_activity"]
                    engine_usage["gfx_busy_inst"] = "N/A"
                    engine_usage["jpeg_busy"] = "N/A"
                    engine_usage["vcn_busy"] = "N/A"

                    # When partition flag is set, use partition-scoped data source
                    if args.partition and gpu_partition_metrics is not None:
                        xcp_gfx_busy = gpu_partition_metrics.get("xcp_stats.gfx_busy_inst", [])
                        xcp_jpeg_busy = gpu_partition_metrics.get("xcp_stats.jpeg_busy", [])
                        xcp_vcn_busy = gpu_partition_metrics.get("xcp_stats.vcn_busy", [])

                        if xcp_gfx_busy != "N/A" and isinstance(xcp_gfx_busy, list):
                            new_xcp_dict = {}
                            for xcp_idx, gfx_busy in enumerate(xcp_gfx_busy):
                                new_xcp_dict[f"xcp_{xcp_idx}"] = (
                                    list(gfx_busy) if isinstance(gfx_busy, list) else gfx_busy
                                )
                            engine_usage["gfx_busy_inst"] = new_xcp_dict

                        if xcp_jpeg_busy != "N/A" and isinstance(xcp_jpeg_busy, list):
                            new_xcp_dict = {}
                            for xcp_idx, jpeg_busy in enumerate(xcp_jpeg_busy):
                                new_xcp_dict[f"xcp_{xcp_idx}"] = (
                                    list(jpeg_busy) if isinstance(jpeg_busy, list) else jpeg_busy
                                )
                            engine_usage["jpeg_busy"] = new_xcp_dict

                        if xcp_vcn_busy != "N/A" and isinstance(xcp_vcn_busy, list):
                            new_xcp_dict = {}
                            for xcp_idx, vcn_busy in enumerate(xcp_vcn_busy):
                                new_xcp_dict[f"xcp_{xcp_idx}"] = (
                                    list(vcn_busy) if isinstance(vcn_busy, list) else vcn_busy
                                )
                            engine_usage["vcn_busy"] = new_xcp_dict
                    elif num_partition != "N/A":
                        # Socket-level metrics with partition support (existing behavior)
                        new_xcp_dict = {}
                        for current_xcp in range(num_partition):
                            new_xcp_dict[f"xcp_{current_xcp}"] = gpu_metric[
                                "xcp_stats.gfx_busy_inst"
                            ][current_xcp]
                        engine_usage["gfx_busy_inst"] = new_xcp_dict

                        new_xcp_dict = {}
                        for current_xcp in range(num_partition):
                            new_xcp_dict[f"xcp_{current_xcp}"] = gpu_metric["xcp_stats.jpeg_busy"][
                                current_xcp
                            ]
                        engine_usage["jpeg_busy"] = new_xcp_dict

                        new_xcp_dict = {}
                        for current_xcp in range(num_partition):
                            new_xcp_dict[f"xcp_{current_xcp}"] = gpu_metric["xcp_stats.vcn_busy"][
                                current_xcp
                            ]
                        engine_usage["vcn_busy"] = new_xcp_dict

                    logging.debug(f"After updates to engine_usage dictionary = {engine_usage}")

                    for key, value in engine_usage.items():
                        activity_unit = "%"
                        if self.logger.is_human_readable_format():
                            if isinstance(value, list):
                                for index, activity in enumerate(value):
                                    if activity != "N/A":
                                        engine_usage[key][index] = f"{activity} {activity_unit}"
                                # Convert list to a string for human readable format
                                engine_usage[key] = (
                                    "[" + ", ".join(str(x) for x in engine_usage[key]) + "]"
                                )
                            elif isinstance(value, dict):
                                for k, v in value.items():
                                    if isinstance(v, list):
                                        for index, activity in enumerate(v):
                                            if activity != "N/A" and not isinstance(activity, str):
                                                value[k][index] = f"{activity} {activity_unit}"
                                        # Convert list to a string for human readable format
                                        value[k] = "[" + ", ".join(str(x) for x in value[k]) + "]"
                                    elif v != "N/A" and not isinstance(v, str):
                                        value[k] = f"{v} {activity_unit}"
                            elif value != "N/A":
                                engine_usage[key] = f"{value} {activity_unit}"
                        if self.logger.is_json_format():
                            if isinstance(value, list):
                                for index, activity in enumerate(value):
                                    if activity != "N/A":
                                        engine_usage[key][index] = {
                                            "value": activity,
                                            "unit": activity_unit,
                                        }
                            elif isinstance(value, dict):
                                for k, v in value.items():
                                    if isinstance(v, list):
                                        for index, activity in enumerate(v):
                                            if activity != "N/A":
                                                value[k][index] = {
                                                    "value": activity,
                                                    "unit": activity_unit,
                                                }
                                    elif v != "N/A":
                                        value[k] = {"value": v, "unit": activity_unit}
                            elif value != "N/A":
                                engine_usage[key] = {"value": value, "unit": activity_unit}

                    values_dict["usage"] = engine_usage
                except Exception as e:
                    values_dict["usage"] = "N/A"
                    logging.debug("Failed to get gpu activity for gpu %s | %s", gpu_id, e)
        if "power" in current_platform_args:
            if args.power:
                power_dict = {
                    "socket_power": "N/A",
                    "gfx_voltage": "N/A",
                    "soc_voltage": "N/A",
                    "mem_voltage": "N/A",
                    "throttle_status": "N/A",
                    "power_management": "N/A",
                    "ubb_power": "N/A",
                }

                try:
                    voltage_unit = "mV"
                    power_unit = "W"
                    power_info = amdsmi_interface.amdsmi_get_power_info(args.gpu)
                    for key, value in power_info.items():
                        if "voltage" in key:
                            power_info[key] = self.helpers.unit_format(
                                self.logger, value, voltage_unit
                            )
                        elif "power" in key:
                            power_info[key] = self.helpers.unit_format(
                                self.logger, value, power_unit
                            )

                    power_dict["socket_power"] = power_info["socket_power"]
                    power_dict["gfx_voltage"] = power_info["gfx_voltage"]
                    power_dict["soc_voltage"] = power_info["soc_voltage"]
                    power_dict["mem_voltage"] = power_info["mem_voltage"]
                    power_dict["ubb_power"] = power_info.get("ubb_power", "N/A")

                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug(
                        "Failed to get power info for gpu %s | %s", gpu_id, e.get_error_info()
                    )

                try:
                    is_power_management_enabled = (
                        amdsmi_interface.amdsmi_is_gpu_power_management_enabled(args.gpu)
                    )
                    if is_power_management_enabled:
                        power_dict["power_management"] = "ENABLED"
                    else:
                        power_dict["power_management"] = "DISABLED"
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug(
                        "Failed to get power management status for gpu %s | %s",
                        gpu_id,
                        e.get_error_info(),
                    )

                try:
                    power_dict["throttle_status"] = "N/A"
                    throttle_status = gpu_metric["throttle_status"]
                    if throttle_status != "N/A":
                        if throttle_status:
                            power_dict["throttle_status"] = "THROTTLED"
                        else:
                            power_dict["throttle_status"] = "UNTHROTTLED"
                except Exception as e:
                    logging.debug("Failed to get throttle status for gpu %s | %s", gpu_id, e)

                values_dict["power"] = power_dict
        if "clock" in current_platform_args:
            if args.clock:
                # Populate Skeleton output with N/A
                clocks = {}

                for clock_index in range(amdsmi_interface.AMDSMI_MAX_NUM_GFX_CLKS):
                    gfx_index = f"gfx_{clock_index}"
                    clocks[gfx_index] = {
                        "clk": "N/A",
                        "min_clk": "N/A",
                        "max_clk": "N/A",
                        "clk_locked": "N/A",
                        "deep_sleep": "N/A",
                    }

                clocks["mem_0"] = {
                    "clk": "N/A",
                    "min_clk": "N/A",
                    "max_clk": "N/A",
                    "clk_locked": "N/A",
                    "deep_sleep": "N/A",
                }

                for clock_index in range(amdsmi_interface.AMDSMI_MAX_NUM_CLKS):
                    vclk_index = f"vclk_{clock_index}"
                    clocks[vclk_index] = {
                        "clk": "N/A",
                        "min_clk": "N/A",
                        "max_clk": "N/A",
                        "clk_locked": "N/A",
                        "deep_sleep": "N/A",
                    }

                for clock_index in range(amdsmi_interface.AMDSMI_MAX_NUM_CLKS):
                    dclk_index = f"dclk_{clock_index}"
                    clocks[dclk_index] = {
                        "clk": "N/A",
                        "min_clk": "N/A",
                        "max_clk": "N/A",
                        "clk_locked": "N/A",
                        "deep_sleep": "N/A",
                    }

                clocks["fclk_0"] = {
                    "clk": "N/A",
                    "min_clk": "N/A",
                    "max_clk": "N/A",
                    "clk_locked": "N/A",
                    "deep_sleep": "N/A",
                }

                clocks["socclk_0"] = {
                    "clk": "N/A",
                    "min_clk": "N/A",
                    "max_clk": "N/A",
                    "clk_locked": "N/A",
                    "deep_sleep": "N/A",
                }

                clocks["uclk_aid"] = "N/A"
                clocks["socclks_mid"] = "N/A"

                clock_unit = "MHz"

                # When partition flag is set, use partition-scoped data source
                partition_metrics_used = False
                if args.partition and gpu_partition_metrics is not None:
                    try:
                        partition_metrics_used = True

                        # Use partition metrics for GFX/VCLK/DCLK/SOCCLK instead of socket metrics
                        # Populate GFX clocks from partition metrics
                        current_gfx_clocks = gpu_partition_metrics.get("current_gfxclks", "N/A")
                        if current_gfx_clocks != "N/A" and isinstance(current_gfx_clocks, list):
                            for clk_idx, clk in enumerate(current_gfx_clocks):
                                if clk != "N/A":
                                    gfx_index = f"gfx_{clk_idx}"
                                    if gfx_index in clocks:
                                        clocks[gfx_index]["clk"] = self.helpers.unit_format(
                                            self.logger, clk, clock_unit
                                        )

                        # GFX clock lock status from partition metrics; annotate only
                        # slots that carry a clock value so an empty skeleton entry never
                        # gets a lock state. A missing key is unknown ("N/A"); 0 is a valid
                        # "all unlocked" reading.
                        gfxclk_lock_status = gpu_partition_metrics.get("gfxclk_lock_status", "N/A")
                        if gfxclk_lock_status != "N/A":
                            for idx in range(amdsmi_interface.AMDSMI_MAX_NUM_GFX_CLKS):
                                gfx_index = f"gfx_{idx}"
                                if gfx_index in clocks and clocks[gfx_index]["clk"] != "N/A":
                                    if (gfxclk_lock_status >> idx) & 1:
                                        clocks[gfx_index]["clk_locked"] = "ENABLED"
                                    else:
                                        clocks[gfx_index]["clk_locked"] = "DISABLED"

                        # Populate VCLK from partition metrics
                        current_vclk_clocks = gpu_partition_metrics.get("current_vclk0s", "N/A")
                        if current_vclk_clocks != "N/A" and isinstance(current_vclk_clocks, list):
                            for clock_index, current_vclk_clock in enumerate(current_vclk_clocks):
                                if current_vclk_clock != "N/A":
                                    vclk_index = f"vclk_{clock_index}"
                                    if vclk_index in clocks:
                                        clocks[vclk_index]["clk"] = self.helpers.unit_format(
                                            self.logger, current_vclk_clock, clock_unit
                                        )

                        # Populate DCLK from partition metrics
                        current_dclk_clocks = gpu_partition_metrics.get("current_dclk0s", "N/A")
                        if current_dclk_clocks != "N/A" and isinstance(current_dclk_clocks, list):
                            for clock_index, current_dclk_clock in enumerate(current_dclk_clocks):
                                if current_dclk_clock != "N/A":
                                    dclk_index = f"dclk_{clock_index}"
                                    if dclk_index in clocks:
                                        clocks[dclk_index]["clk"] = self.helpers.unit_format(
                                            self.logger, current_dclk_clock, clock_unit
                                        )

                        # Populate SOCCLK from partition metrics (AID level)
                        current_socclks = gpu_partition_metrics.get("current_socclks", "N/A")
                        if current_socclks != "N/A" and isinstance(current_socclks, list):
                            for idx, socclk in enumerate(current_socclks):
                                if socclk != "N/A":
                                    clocks["socclk_0"]["clk"] = self.helpers.unit_format(
                                        self.logger, socclk, clock_unit
                                    )
                                    break  # Use first valid value

                        # Populate MID SOC clocks from partition metrics
                        current_socclks_mid = gpu_partition_metrics.get(
                            "current_socclks_mid", "N/A"
                        )
                        if current_socclks_mid != "N/A" and isinstance(current_socclks_mid, list):
                            clocks["socclks_mid"] = {
                                f"mid_{index}": self.helpers.unit_format(
                                    self.logger, clk, clock_unit
                                )
                                for index, clk in enumerate(current_socclks_mid)
                                if clk != "N/A"
                            }

                        # Add AID-level clock data (VCLK, DCLK, SOCCLK per AID)
                        # Iterate every VCLK position; per-field "N/A" guards skip holes,
                        # so counting only non-"N/A" entries would drop trailing valid AIDs
                        num_aids = 0
                        if isinstance(current_vclk_clocks, list) and current_vclk_clocks != "N/A":
                            num_aids = len(current_vclk_clocks)

                        # Get clock limits for AID clocks
                        try:
                            vclk0_limits = amdsmi_interface.amdsmi_get_clock_info(
                                args.gpu, amdsmi_interface.AmdSmiClkType.VCLK0
                            )
                            dclk0_limits = amdsmi_interface.amdsmi_get_clock_info(
                                args.gpu, amdsmi_interface.AmdSmiClkType.DCLK0
                            )
                            soc_limits = amdsmi_interface.amdsmi_get_clock_info(
                                args.gpu, amdsmi_interface.AmdSmiClkType.SOC
                            )
                        except amdsmi_exception.AmdSmiLibraryException:
                            vclk0_limits = None
                            dclk0_limits = None
                            soc_limits = None

                        for aid_idx in range(num_aids):
                            aid_key = f"aid_{aid_idx}"
                            aid_clocks = {}

                            # VCLK for this AID
                            if (
                                aid_idx < len(current_vclk_clocks)
                                and current_vclk_clocks[aid_idx] != "N/A"
                            ):
                                aid_clocks["vclk"] = self.helpers.unit_format(
                                    self.logger, current_vclk_clocks[aid_idx], clock_unit
                                )
                                if vclk0_limits:
                                    aid_clocks["vclk_min_limit"] = self.helpers.unit_format(
                                        self.logger, vclk0_limits["min_clk"], clock_unit
                                    )
                                    aid_clocks["vclk_max_limit"] = self.helpers.unit_format(
                                        self.logger, vclk0_limits["max_clk"], clock_unit
                                    )

                            # DCLK for this AID
                            if isinstance(current_dclk_clocks, list) and aid_idx < len(
                                current_dclk_clocks
                            ):
                                if current_dclk_clocks[aid_idx] != "N/A":
                                    aid_clocks["dclk"] = self.helpers.unit_format(
                                        self.logger, current_dclk_clocks[aid_idx], clock_unit
                                    )
                                    if dclk0_limits:
                                        aid_clocks["dclk_min_limit"] = self.helpers.unit_format(
                                            self.logger, dclk0_limits["min_clk"], clock_unit
                                        )
                                        aid_clocks["dclk_max_limit"] = self.helpers.unit_format(
                                            self.logger, dclk0_limits["max_clk"], clock_unit
                                        )

                            # SOCCLK for this AID
                            if isinstance(current_socclks, list) and aid_idx < len(current_socclks):
                                if current_socclks[aid_idx] != "N/A":
                                    aid_clocks["socclk"] = self.helpers.unit_format(
                                        self.logger, current_socclks[aid_idx], clock_unit
                                    )
                                    if soc_limits:
                                        aid_clocks["socclk_min_limit"] = self.helpers.unit_format(
                                            self.logger, soc_limits["min_clk"], clock_unit
                                        )
                                        aid_clocks["socclk_max_limit"] = self.helpers.unit_format(
                                            self.logger, soc_limits["max_clk"], clock_unit
                                        )

                            if aid_clocks:
                                clocks[aid_key] = aid_clocks

                        # Add XCP-level GFX clock data (gfx_clk per XCP with limits and lock status)
                        # Determine number of XCPs
                        num_xcps = 0
                        if isinstance(current_gfx_clocks, list) and current_gfx_clocks != "N/A":
                            num_xcps = len(current_gfx_clocks)

                        # Get GFX clock limits
                        try:
                            gfx_limits = amdsmi_interface.amdsmi_get_clock_info(
                                args.gpu, amdsmi_interface.AmdSmiClkType.GFX
                            )
                        except amdsmi_exception.AmdSmiLibraryException:
                            gfx_limits = None

                        for xcp_idx in range(num_xcps):
                            xcp_key = f"xcp_{xcp_idx}"
                            xcp_clocks = {}

                            # GFX clock for this XCP (could be single value or array of engine clocks)
                            if xcp_idx < len(current_gfx_clocks):
                                gfx_clk_data = current_gfx_clocks[xcp_idx]
                                if gfx_clk_data != "N/A":
                                    if isinstance(gfx_clk_data, list):
                                        # Array of engine clocks - take first valid value
                                        for clk in gfx_clk_data:
                                            if clk != "N/A":
                                                xcp_clocks["gfx_clk"] = self.helpers.unit_format(
                                                    self.logger, clk, clock_unit
                                                )
                                                break
                                    else:
                                        # Single clock value
                                        xcp_clocks["gfx_clk"] = self.helpers.unit_format(
                                            self.logger, gfx_clk_data, clock_unit
                                        )

                            # Only annotate limits/lock for an XCP that reports a real gfx
                            # clock; otherwise the entry would carry limits with no value
                            if "gfx_clk" in xcp_clocks:
                                # GFX min/max limits (same for all XCPs)
                                if gfx_limits:
                                    xcp_clocks["gfx_min_clk"] = self.helpers.unit_format(
                                        self.logger, gfx_limits["min_clk"], clock_unit
                                    )
                                    xcp_clocks["gfx_max_clk"] = self.helpers.unit_format(
                                        self.logger, gfx_limits["max_clk"], clock_unit
                                    )

                                # GFX clock lock status for this XCP's gfx domain; missing
                                # status is unknown ("N/A"), 0 is a valid "unlocked" reading
                                if gfxclk_lock_status != "N/A":
                                    is_locked = (gfxclk_lock_status >> xcp_idx) & 1
                                    xcp_clocks["gfx_clk_locked"] = (
                                        "ENABLED" if is_locked else "DISABLED"
                                    )
                                else:
                                    xcp_clocks["gfx_clk_locked"] = "N/A"

                            if xcp_clocks:
                                clocks[xcp_key] = xcp_clocks

                    except amdsmi_exception.AmdSmiLibraryException as e:
                        logging.debug(
                            "Failed to get partition clock metrics for gpu %s | %s",
                            gpu_id,
                            e.get_error_info(),
                        )
                        partition_metrics_used = False

                # Populate clock values from gpu_metrics_info (socket-level or fallback)
                if not partition_metrics_used:
                    # Populate GFX clock values
                    try:
                        current_gfx_clocks = gpu_metric["current_gfxclks"]
                        if current_gfx_clocks != "N/A":
                            for clock_index, current_gfx_clock in enumerate(current_gfx_clocks):
                                # If the current clock is N/A then nothing else applies
                                if current_gfx_clock == "N/A":
                                    continue
                                gfx_index = f"gfx_{clock_index}"
                                clocks[gfx_index]["clk"] = self.helpers.unit_format(
                                    self.logger, current_gfx_clock, clock_unit
                                )
                                # Populate clock locked status
                                if gpu_metric["gfxclk_lock_status"] != "N/A":
                                    gfx_clock_lock_flag = (
                                        1 << clock_index
                                    )  # This is the position of the clock lock flag
                                    if gpu_metric["gfxclk_lock_status"] & gfx_clock_lock_flag:
                                        clocks[gfx_index]["clk_locked"] = "ENABLED"
                                    else:
                                        clocks[gfx_index]["clk_locked"] = "DISABLED"
                    except Exception as e:
                        logging.debug("Failed to get current_gfxclks for gpu %s | %s", gpu_id, e)

                    # Populate MEM clock value
                    try:
                        current_mem_clock = gpu_metric["current_uclk"]  # single value
                        if current_mem_clock != "N/A":
                            clocks["mem_0"]["clk"] = self.helpers.unit_format(
                                self.logger, current_mem_clock, clock_unit
                            )
                    except Exception as e:
                        logging.debug("Failed to get current_uclk for gpu %s | %s", gpu_id, e)

                    # Populate VCLK clock values
                    try:
                        current_vclk_clocks = gpu_metric["current_vclk0s"]
                        # If the current vclk clocks are not available, we cannot proceed further
                        if current_vclk_clocks != "N/A":
                            for clock_index, current_vclk_clock in enumerate(current_vclk_clocks):
                                # If the current clock is N/A then nothing else applies
                                if current_vclk_clock == "N/A":
                                    continue
                                vclk_index = f"vclk_{clock_index}"
                                clocks[vclk_index]["clk"] = self.helpers.unit_format(
                                    self.logger, current_vclk_clock, clock_unit
                                )
                    except Exception as e:
                        logging.debug("Failed to get current_vclk0s for gpu %s | %s", gpu_id, e)

                    # Populate DCLK clock values
                    try:
                        current_dclk_clocks = gpu_metric["current_dclk0s"]
                        # If the current dclk clocks are not available, we cannot proceed further
                        if current_dclk_clocks != "N/A":
                            for clock_index, current_dclk_clock in enumerate(current_dclk_clocks):
                                # If the current clock is N/A then nothing else applies
                                if current_dclk_clock == "N/A":
                                    continue
                                dclk_index = f"dclk_{clock_index}"
                                clocks[dclk_index]["clk"] = self.helpers.unit_format(
                                    self.logger, current_dclk_clock, clock_unit
                                )
                    except Exception as e:
                        logging.debug("Failed to get current_dclk0s for gpu %s | %s", gpu_id, e)

                    # Populate FCLK clock value; fclk not present in gpu_metrics so use amdsmi_get_clk_freq
                    try:
                        frequency_dict = amdsmi_interface.amdsmi_get_clk_freq(
                            args.gpu, amdsmi_interface.AmdSmiClkType.DF
                        )
                        # The C library reports current = (uint32_t)-1 when the
                        # kernel exposes pp_dpm_fclk without a '*' current-level
                        # marker (e.g. SMU power-gated DF on gfx1151 APUs at idle).
                        # Treat that as "no current frequency" so we don't index
                        # out of range.
                        current_fclk_index = frequency_dict["current"]
                        if current_fclk_index == 0xFFFFFFFF or current_fclk_index >= len(
                            frequency_dict["frequency"]
                        ):
                            raise IndexError("no current fclk level reported by kernel")
                        current_fclk_clock = frequency_dict["frequency"][current_fclk_index]
                        current_fclk_clock = self.helpers.convert_SI_unit(
                            current_fclk_clock, self.helpers.SI_Unit.MICRO
                        )
                        clocks["fclk_0"]["clk"] = self.helpers.unit_format(
                            self.logger, current_fclk_clock, clock_unit
                        )
                    except (KeyError, IndexError, amdsmi_exception.AmdSmiLibraryException) as e:
                        logging.debug("Failed to get fclk info for gpu %s | %s", gpu_id, e)

                    # Populate SOCCLK clock value
                    try:
                        current_socclk_clock = gpu_metric["current_socclk"]
                        # If the current socclk clocks are not available, we cannot proceed further
                        if current_socclk_clock != "N/A":
                            clocks["socclk_0"]["clk"] = self.helpers.unit_format(
                                self.logger, current_socclk_clock, clock_unit
                            )
                    except KeyError as e:
                        logging.debug("Failed to get current_socclk for gpu %s | %s", gpu_id, e)

                    try:
                        current_uclk_aid = gpu_metric.get("current_uclk_aid", "N/A")
                        if current_uclk_aid != "N/A":
                            clocks["uclk_aid"] = {
                                f"aid_{index}": self.helpers.unit_format(
                                    self.logger, clk, clock_unit
                                )
                                if clk != "N/A"
                                else "N/A"
                                for index, clk in enumerate(current_uclk_aid)
                            }
                    except Exception as e:
                        logging.debug("Failed to get current_uclk_aid for gpu %s | %s", gpu_id, e)

                    try:
                        current_socclks_mid = gpu_metric.get("current_socclks_mid", "N/A")
                        if current_socclks_mid != "N/A":
                            clocks["socclks_mid"] = {
                                f"mid_{index}": self.helpers.unit_format(
                                    self.logger, clk, clock_unit
                                )
                                if clk != "N/A"
                                else "N/A"
                                for index, clk in enumerate(current_socclks_mid)
                            }
                    except Exception as e:
                        logging.debug(
                            "Failed to get current_socclks_mid for gpu %s | %s", gpu_id, e
                        )

                # Populate the max and min clock values from sysfs.
                # Min and Max values are per clock type, not per clock engine.
                # Populate the deep sleep value from amdsmi_get_clock_info

                # GFX min and max clocks
                try:
                    gfx_clock_info_dict = amdsmi_interface.amdsmi_get_clock_info(
                        args.gpu, amdsmi_interface.AmdSmiClkType.GFX
                    )
                    for clock_index in range(amdsmi_interface.AMDSMI_MAX_NUM_GFX_CLKS):
                        gfx_index = f"gfx_{clock_index}"

                        if clocks[gfx_index]["clk"] == "N/A":
                            # if the current clock is N/A then we shouldn't populate the max and min values
                            continue
                        clocks[gfx_index]["min_clk"] = self.helpers.unit_format(
                            self.logger, gfx_clock_info_dict["min_clk"], clock_unit
                        )
                        clocks[gfx_index]["max_clk"] = self.helpers.unit_format(
                            self.logger, gfx_clock_info_dict["max_clk"], clock_unit
                        )
                        # Add the clk_deep_sleep
                        clocks[gfx_index]["deep_sleep"] = gfx_clock_info_dict["clk_deep_sleep"]
                except (KeyError, amdsmi_exception.AmdSmiLibraryException) as e:
                    logging.debug("Failed to get gfx clock info for gpu %s | %s", gpu_id, e)

                # MEM min and max clocks
                try:
                    mem_clock_info_dict = amdsmi_interface.amdsmi_get_clock_info(
                        args.gpu, amdsmi_interface.AmdSmiClkType.MEM
                    )
                    # if the current clock is N/A then we shouldn't populate the max and min values
                    if clocks["mem_0"]["clk"] != "N/A":
                        clocks["mem_0"]["min_clk"] = self.helpers.unit_format(
                            self.logger, mem_clock_info_dict["min_clk"], clock_unit
                        )
                        clocks["mem_0"]["max_clk"] = self.helpers.unit_format(
                            self.logger, mem_clock_info_dict["max_clk"], clock_unit
                        )
                        # Add the clk_deep_sleep
                        clocks["mem_0"]["deep_sleep"] = mem_clock_info_dict["clk_deep_sleep"]
                except (KeyError, amdsmi_exception.AmdSmiLibraryException) as e:
                    logging.debug("Failed to get mem clock info for gpu %s | %s", gpu_id, e)

                # VCLK min and max clocks
                try:
                    # Retrieve clock information for VCLK0 (Video Clock 0)
                    vclk_clock_info_dict = amdsmi_interface.amdsmi_get_clock_info(
                        args.gpu, amdsmi_interface.AmdSmiClkType.VCLK0
                    )

                    # Iterate through the maximum number of VCLK clocks supported
                    for index in range(amdsmi_interface.AMDSMI_MAX_NUM_CLKS):
                        vclk_index = f"vclk_{index}"  # Construct the index key for the clock

                        # Check if the current clock value is not "N/A"
                        if clocks[vclk_index]["clk"] != "N/A":
                            # Format and assign the minimum clock value for the current VCLK
                            clocks[vclk_index]["min_clk"] = self.helpers.unit_format(
                                self.logger, vclk_clock_info_dict["min_clk"], clock_unit
                            )
                            # Format and assign the maximum clock value for the current VCLK
                            clocks[vclk_index]["max_clk"] = self.helpers.unit_format(
                                self.logger, vclk_clock_info_dict["max_clk"], clock_unit
                            )
                            # Add the clk_deep_sleep
                            clocks[vclk_index]["deep_sleep"] = vclk_clock_info_dict[
                                "clk_deep_sleep"
                            ]
                except (KeyError, amdsmi_exception.AmdSmiLibraryException) as e:
                    # Log a debug message if retrieving VCLK clock information fails
                    logging.debug("Failed to get vclk clock info for gpu %s | %s", gpu_id, e)

                # DCLK min and max clocks
                try:
                    # Retrieve clock information for DCLK0 (Display Clock 0)
                    dclk_clock_info_dict = amdsmi_interface.amdsmi_get_clock_info(
                        args.gpu, amdsmi_interface.AmdSmiClkType.DCLK0
                    )

                    # Iterate through the maximum number of DCLK clocks supported
                    for index in range(amdsmi_interface.AMDSMI_MAX_NUM_CLKS):
                        dclk_index = f"dclk_{index}"  # Construct the index key for the clock

                        # Check if the current clock value is not "N/A"
                        if clocks[dclk_index]["clk"] != "N/A":
                            # Format and assign the minimum clock value for the current DCLK
                            clocks[dclk_index]["min_clk"] = self.helpers.unit_format(
                                self.logger, dclk_clock_info_dict["min_clk"], clock_unit
                            )
                            # Format and assign the maximum clock value for the current DCLK
                            clocks[dclk_index]["max_clk"] = self.helpers.unit_format(
                                self.logger, dclk_clock_info_dict["max_clk"], clock_unit
                            )
                            # Add the clk_deep_sleep
                            clocks[dclk_index]["deep_sleep"] = dclk_clock_info_dict[
                                "clk_deep_sleep"
                            ]
                except (KeyError, amdsmi_exception.AmdSmiLibraryException) as e:
                    logging.debug("Failed to get dclk clock info for gpu %s | %s", gpu_id, e)

                # FCLK min and max clocks
                try:
                    fclk_clk_info_dict = amdsmi_interface.amdsmi_get_clock_info(
                        args.gpu, amdsmi_interface.AmdSmiClkType.DF
                    )
                    # if the current clock is N/A then we shouldn't populate the max and min values
                    if clocks["fclk_0"]["clk"] != "N/A":
                        clocks["fclk_0"]["min_clk"] = self.helpers.unit_format(
                            self.logger, fclk_clk_info_dict["min_clk"], clock_unit
                        )
                        clocks["fclk_0"]["max_clk"] = self.helpers.unit_format(
                            self.logger, fclk_clk_info_dict["max_clk"], clock_unit
                        )
                        # Add the clk_deep_sleep
                        clocks["fclk_0"]["deep_sleep"] = fclk_clk_info_dict["clk_deep_sleep"]
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug(
                        "Failed to get fclk info for gpu %s | %s", gpu_id, e.get_error_info()
                    )

                # SOCCLK min and max clocks
                try:
                    socclk_clk_info_dict = amdsmi_interface.amdsmi_get_clock_info(
                        args.gpu, amdsmi_interface.AmdSmiClkType.SOC
                    )
                    # if the current clock is N/A then we shouldn't populate the max and min values
                    if clocks["socclk_0"]["clk"] != "N/A":
                        clocks["socclk_0"]["min_clk"] = self.helpers.unit_format(
                            self.logger, socclk_clk_info_dict["min_clk"], clock_unit
                        )
                        clocks["socclk_0"]["max_clk"] = self.helpers.unit_format(
                            self.logger, socclk_clk_info_dict["max_clk"], clock_unit
                        )
                        # Add the clk_deep_sleep
                        clocks["socclk_0"]["deep_sleep"] = socclk_clk_info_dict["clk_deep_sleep"]
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug(
                        "Failed to get socclk info for gpu %s | %s", gpu_id, e.get_error_info()
                    )

                # Iterate over each clock and its data to determine if deep sleep is enabled
                # based on the comparison between the current clock value and the minimum clock value.
                for clock, clock_data in clocks.items():
                    clk_value = 0
                    min_clk_value = 0
                    try:
                        clk = clock_data["clk"]
                        min_clk = clock_data["min_clk"]
                        if clk == "N/A" or min_clk == "N/A":
                            continue
                        # Extract numeric value if clk/min_clk is a dict, else use as is
                        if isinstance(clk, dict):
                            clk_value = int(clk.get("value", 0))
                        else:
                            if isinstance(clk, str):
                                clk_value = int(str(clk).split()[0])
                            else:
                                clk_value = int(clk)
                        if isinstance(min_clk, dict):
                            min_clk_value = int(min_clk.get("value", 0))
                        else:
                            if isinstance(min_clk, str):
                                min_clk_value = int(str(min_clk).split()[0])
                            else:
                                min_clk_value = int(min_clk)
                        # If the clk value is less than the min_clk value, then deep sleep is enabled
                        if clk_value < min_clk_value:
                            clock_data["deep_sleep"] = "ENABLED"
                        else:
                            clock_data["deep_sleep"] = "DISABLED"
                    except Exception as e:
                        logging.debug("Failed to get deep sleep status for gpu %s | %s", gpu_id, e)

                values_dict["clock"] = clocks
        if "temperature" in current_platform_args:
            if args.temperature:
                try:
                    temperature_edge_current = amdsmi_interface.amdsmi_get_temp_metric(
                        args.gpu,
                        amdsmi_interface.AmdSmiTemperatureType.EDGE,
                        amdsmi_interface.AmdSmiTemperatureMetric.CURRENT,
                    )
                except amdsmi_exception.AmdSmiLibraryException as e:
                    temperature_edge_current = "N/A"
                    logging.debug(
                        "Failed to get current edge temperature for gpu %s | %s",
                        gpu_id,
                        e.get_error_info(),
                    )

                try:
                    temperature_edge_limit = amdsmi_interface.amdsmi_get_temp_metric(
                        args.gpu,
                        amdsmi_interface.AmdSmiTemperatureType.EDGE,
                        amdsmi_interface.AmdSmiTemperatureMetric.CRITICAL,
                    )
                except amdsmi_exception.AmdSmiLibraryException as e:
                    temperature_edge_limit = "N/A"
                    logging.debug(
                        "Failed to get edge temperature limit for gpu %s | %s",
                        gpu_id,
                        e.get_error_info(),
                    )

                # If edge limit is reporting 0 then set the current edge temp to N/A
                if temperature_edge_limit == 0:
                    temperature_edge_current = "N/A"

                try:
                    temperature_hotspot_current = amdsmi_interface.amdsmi_get_temp_metric(
                        args.gpu,
                        amdsmi_interface.AmdSmiTemperatureType.HOTSPOT,
                        amdsmi_interface.AmdSmiTemperatureMetric.CURRENT,
                    )
                except amdsmi_exception.AmdSmiLibraryException as e:
                    temperature_hotspot_current = "N/A"
                    logging.debug(
                        "Failed to get current hotspot temperature for gpu %s | %s",
                        gpu_id,
                        e.get_error_info(),
                    )

                try:
                    temperature_vram_current = amdsmi_interface.amdsmi_get_temp_metric(
                        args.gpu,
                        amdsmi_interface.AmdSmiTemperatureType.VRAM,
                        amdsmi_interface.AmdSmiTemperatureMetric.CURRENT,
                    )
                except amdsmi_exception.AmdSmiLibraryException as e:
                    temperature_vram_current = "N/A"
                    logging.debug(
                        "Failed to get current vram temperature for gpu %s | %s",
                        gpu_id,
                        e.get_error_info(),
                    )

                # When partition flag is set, use partition-scoped data source
                if args.partition and gpu_partition_metrics is not None:
                    temperatures = {
                        "edge": temperature_edge_current,
                        "hotspot": temperature_hotspot_current,
                        "mem": temperature_vram_current,
                        "hbm_stacks": "N/A",
                        "mid": gpu_partition_metrics.get("temperature_mid", "N/A"),
                        "aid": "N/A",  # AID temps not in partition metrics
                        "xcd": "N/A",
                    }

                    # MID temperatures from partition metrics
                    if temperatures["mid"] != "N/A":
                        temperatures["mid"] = list(temperatures["mid"])

                    # XCP/XCD temperatures from partition metrics
                    xcp_temp_xcd = gpu_partition_metrics.get("xcp_stats.temperature_xcd", "N/A")
                    if xcp_temp_xcd != "N/A" and isinstance(xcp_temp_xcd, list):
                        temperatures["xcd"] = {
                            f"xcp_{idx}": xcp_temp_xcd[idx] for idx in range(len(xcp_temp_xcd))
                        }
                else:
                    # Socket-level metrics (existing behavior)
                    temperatures = {
                        "edge": temperature_edge_current,
                        "hotspot": temperature_hotspot_current,
                        "mem": temperature_vram_current,
                        "hbm_stacks": gpu_metric.get("temperature_hbm_stacks", "N/A"),
                        "mid": gpu_metric.get("temperature_mid", "N/A"),
                        "aid": gpu_metric.get("temperature_aid", "N/A"),
                        "xcd": "N/A",
                    }

                    if temperatures["hbm_stacks"] != "N/A":
                        temperatures["hbm_stacks"] = list(temperatures["hbm_stacks"])
                    if temperatures["mid"] != "N/A":
                        temperatures["mid"] = list(temperatures["mid"])
                    if temperatures["aid"] != "N/A":
                        temperatures["aid"] = list(temperatures["aid"])

                    if num_partition != "N/A":
                        xcp_temp_xcd = gpu_metric.get("xcp_stats.temperature_xcd", "N/A")
                        if xcp_temp_xcd != "N/A":
                            available_partition = min(num_partition, len(xcp_temp_xcd))
                            temperatures["xcd"] = {
                                f"xcp_{current_xcp}": xcp_temp_xcd[current_xcp]
                                for current_xcp in range(available_partition)
                            }

                temp_unit_human_readable = "\N{DEGREE SIGN}C"
                temp_unit_json = "C"
                for temperature_key, temperature_value in temperatures.items():
                    if isinstance(temperature_value, list):
                        if self.logger.is_human_readable_format():
                            formatted_values = [
                                f"{value} {temp_unit_human_readable}" if value != "N/A" else "N/A"
                                for value in temperature_value
                            ]
                            temperatures[temperature_key] = "[" + ", ".join(formatted_values) + "]"
                        if self.logger.is_json_format():
                            temperatures[temperature_key] = [
                                {"value": value, "unit": temp_unit_json}
                                if value != "N/A"
                                else "N/A"
                                for value in temperature_value
                            ]
                    elif isinstance(temperature_value, dict):
                        for key, value in temperature_value.items():
                            if isinstance(value, list):
                                if self.logger.is_human_readable_format():
                                    formatted_values = [
                                        f"{item} {temp_unit_human_readable}"
                                        if item != "N/A"
                                        else "N/A"
                                        for item in value
                                    ]
                                    temperature_value[key] = "[" + ", ".join(formatted_values) + "]"
                                if self.logger.is_json_format():
                                    temperature_value[key] = [
                                        {"value": item, "unit": temp_unit_json}
                                        if item != "N/A"
                                        else "N/A"
                                        for item in value
                                    ]
                            elif value != "N/A":
                                if self.logger.is_human_readable_format():
                                    temperature_value[key] = f"{value} {temp_unit_human_readable}"
                                if self.logger.is_json_format():
                                    temperature_value[key] = {
                                        "value": value,
                                        "unit": temp_unit_json,
                                    }
                    else:
                        if temperature_value == "N/A":
                            continue
                        if self.logger.is_human_readable_format():
                            temperatures[temperature_key] = (
                                f"{temperature_value} {temp_unit_human_readable}"
                            )
                        if self.logger.is_json_format():
                            temperatures[temperature_key] = {
                                "value": temperature_value,
                                "unit": temp_unit_json,
                            }

                values_dict["temperature"] = temperatures

        # Since pcie bw may increase based on frequent metrics calls, we add it to the output here, but the populate the values first
        if "pcie" in current_platform_args:
            if args.pcie:
                values_dict["pcie"] = pcie_dict

        if "gpu_board" in current_platform_args:
            if args.gpu_board:
                gpu_board_temp_dict = self.helpers.get_gpu_board_temperatures(
                    args.gpu, gpu_id, self.logger
                )
                # if every value is N/A, then we don't want to display the values unless explicitly told to
                # all args_list being True indicates that this gpu_board is not explicitly called itself
                args_list = [getattr(args, arg) for arg in current_platform_args]
                if all(value == "N/A" for value in gpu_board_temp_dict.values()) and all(
                    arg == True for arg in args_list
                ):
                    gpu_board_temp_dict = {}
                if gpu_board_temp_dict:
                    values_dict["gpu_board"] = {"temperature": gpu_board_temp_dict}
        if "base_board" in current_platform_args:
            if args.base_board:
                base_board_temp_dict = self.helpers.get_base_board_temperatures(
                    args.gpu, gpu_id, self.logger
                )
                # if every value is N/A, then we don't want to display the values unless explicitly told to
                # all args_list being True indicates that this base_board is not explicitly called itself
                args_list = [getattr(args, arg) for arg in current_platform_args]
                if all(value == "N/A" for value in base_board_temp_dict.values()) and all(
                    arg == True for arg in args_list
                ):
                    base_board_temp_dict = {}
                if base_board_temp_dict:
                    values_dict["base_board"] = {"temperature": base_board_temp_dict}
        if "ecc" in current_platform_args:
            if args.ecc:
                ecc_count = {}
                try:
                    ecc_count = amdsmi_interface.amdsmi_get_gpu_total_ecc_count(args.gpu)
                    ecc_count["total_correctable_count"] = ecc_count.pop("correctable_count")
                    ecc_count["total_uncorrectable_count"] = ecc_count.pop("uncorrectable_count")
                    ecc_count["total_deferred_count"] = ecc_count.pop("deferred_count")
                except amdsmi_exception.AmdSmiLibraryException as e:
                    ecc_count["total_correctable_count"] = "N/A"
                    ecc_count["total_uncorrectable_count"] = "N/A"
                    ecc_count["cache_correctable_count"] = "N/A"
                    ecc_count["cache_uncorrectable_count"] = "N/A"
                    logging.debug(
                        "Failed to get total ecc count for gpu %s | %s", gpu_id, e.get_error_info()
                    )

                if ecc_count["total_correctable_count"] != "N/A":
                    # Get the UMC error count for getting total cache correctable errors
                    umc_block = amdsmi_interface.AmdSmiGpuBlock["UMC"]
                    try:
                        umc_count = amdsmi_interface.amdsmi_get_gpu_ecc_count(args.gpu, umc_block)
                        ecc_count["cache_correctable_count"] = (
                            ecc_count["total_correctable_count"] - umc_count["correctable_count"]
                        )
                        ecc_count["cache_uncorrectable_count"] = (
                            ecc_count["total_uncorrectable_count"]
                            - umc_count["uncorrectable_count"]
                        )
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        ecc_count["cache_correctable_count"] = "N/A"
                        ecc_count["cache_uncorrectable_count"] = "N/A"
                        logging.debug(
                            "Failed to get cache ecc count for gpu %s at block %s | %s",
                            gpu_id,
                            umc_block,
                            e.get_error_info(),
                        )

                values_dict["ecc"] = ecc_count
        if "ecc_blocks" in current_platform_args:
            if args.ecc_blocks:
                ecc_dict = {}
                sysfs_blocks = ["UMC", "SDMA", "GFX", "MMHUB", "PCIE_BIF", "HDP", "XGMI_WAFL"]
                try:
                    ras_states = amdsmi_interface.amdsmi_get_gpu_ras_block_features_enabled(
                        args.gpu
                    )
                    for state in ras_states:
                        # Only add enabled blocks that are also in sysfs
                        if state["status"] == amdsmi_interface.AmdSmiRasErrState.ENABLED.name:
                            gpu_block = amdsmi_interface.AmdSmiGpuBlock[state["block"]]
                            # if the blocks are uncountable do not add them at all.
                            if gpu_block.name in sysfs_blocks:
                                try:
                                    ecc_count = amdsmi_interface.amdsmi_get_gpu_ecc_count(
                                        args.gpu, gpu_block
                                    )
                                    ecc_dict[state["block"]] = {
                                        "correctable_count": ecc_count["correctable_count"],
                                        "uncorrectable_count": ecc_count["uncorrectable_count"],
                                        "deferred_count": ecc_count["deferred_count"],
                                    }
                                except amdsmi_exception.AmdSmiLibraryException as e:
                                    ecc_dict[state["block"]] = {
                                        "correctable_count": "N/A",
                                        "uncorrectable_count": "N/A",
                                        "deferred_count": "N/A",
                                    }
                                    logging.debug(
                                        "Failed to get ecc count for gpu %s at block %s | %s",
                                        gpu_id,
                                        gpu_block,
                                        e.get_error_info(),
                                    )

                    values_dict["ecc_blocks"] = ecc_dict
                except amdsmi_exception.AmdSmiLibraryException as e:
                    values_dict["ecc_blocks"] = "N/A"
                    logging.debug(
                        "Failed to get ecc block features for gpu %s | %s",
                        gpu_id,
                        e.get_error_info(),
                    )
        if "fan" in current_platform_args:
            if args.fan:
                fan_dict = {"speed": "N/A", "max": "N/A", "rpm": "N/A", "usage": "N/A"}

                try:
                    fan_speed = amdsmi_interface.amdsmi_get_gpu_fan_speed(args.gpu, 0)
                    fan_dict["speed"] = fan_speed
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug(
                        "Failed to get fan speed for gpu %s | %s", args.gpu, e.get_error_info()
                    )

                try:
                    fan_max = amdsmi_interface.amdsmi_get_gpu_fan_speed_max(args.gpu, 0)
                    fan_usage = "N/A"
                    if fan_max > 0 and fan_dict["speed"] != "N/A":
                        fan_usage = round((float(fan_speed) / float(fan_max)) * 100, 2)
                        fan_usage_unit = "%"
                        if self.logger.is_human_readable_format():
                            fan_usage = f"{fan_usage} {fan_usage_unit}"
                        if self.logger.is_json_format():
                            fan_usage = {"value": fan_usage, "unit": fan_usage_unit}
                    fan_dict["max"] = fan_max
                    fan_dict["usage"] = fan_usage
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug(
                        "Failed to get fan max speed for gpu %s | %s", args.gpu, e.get_error_info()
                    )

                try:
                    fan_rpm = amdsmi_interface.amdsmi_get_gpu_fan_rpms(args.gpu, 0)
                    fan_dict["rpm"] = fan_rpm
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug(
                        "Failed to get fan rpms for gpu %s | %s", args.gpu, e.get_error_info()
                    )

                values_dict["fan"] = fan_dict
        if "voltage_curve" in current_platform_args:
            if args.voltage_curve:
                # Populate N/A values per voltage point
                voltage_point_dict = {}
                for point in range(amdsmi_interface.AMDSMI_NUM_VOLTAGE_CURVE_POINTS):
                    voltage_point_dict[f"point_{point}_frequency"] = "N/A"
                    voltage_point_dict[f"point_{point}_voltage"] = "N/A"

                try:
                    od_volt = amdsmi_interface.amdsmi_get_gpu_od_volt_info(args.gpu)
                    logging.debug(f"OD Voltage info: {od_volt}")
                except amdsmi_exception.AmdSmiLibraryException as e:
                    od_volt = "N/A"  # Value not used, but needs to not be a dict
                    logging.debug(
                        "Failed to get voltage curve for gpu %s | %s", gpu_id, e.get_error_info()
                    )

                # Populate voltage point values
                for point in range(amdsmi_interface.AMDSMI_NUM_VOLTAGE_CURVE_POINTS):
                    if isinstance(od_volt, dict):
                        logging.debug(
                            f"point_{point} frequency: {od_volt['curve.vc_points'][point]['frequency']}"
                        )
                        logging.debug(
                            f"point_{point} voltage:   {od_volt['curve.vc_points'][point]['voltage']}"
                        )
                        frequency = int(od_volt["curve.vc_points"][point]["frequency"] / 1000000)
                        voltage = int(od_volt["curve.vc_points"][point]["voltage"])
                    else:
                        frequency = "N/A"
                        voltage = "N/A"

                    if frequency == 0:
                        frequency = "N/A"

                    if voltage == 0:
                        voltage = "N/A"

                    if frequency != "N/A":
                        frequency = self.helpers.unit_format(self.logger, frequency, "Mhz")

                    if voltage != "N/A":
                        voltage = self.helpers.unit_format(self.logger, voltage, "mV")

                    voltage_point_dict[f"point_{point}_frequency"] = frequency
                    voltage_point_dict[f"point_{point}_voltage"] = voltage

                values_dict["voltage_curve"] = voltage_point_dict
        if "overdrive" in current_platform_args:
            if args.overdrive:
                try:
                    overdrive_level = amdsmi_interface.amdsmi_get_gpu_overdrive_level(args.gpu)
                    od_unit = "%"
                    values_dict["overdrive"] = self.helpers.unit_format(
                        self.logger, overdrive_level, od_unit
                    )
                except amdsmi_exception.AmdSmiLibraryException as e:
                    values_dict["overdrive"] = "N/A"
                    logging.debug(
                        "Failed to get gpu overdrive level for gpu %s | %s",
                        gpu_id,
                        e.get_error_info(),
                    )

                try:
                    mem_overdrive_level = amdsmi_interface.amdsmi_get_gpu_mem_overdrive_level(
                        args.gpu
                    )
                    od_unit = "%"
                    values_dict["mem_overdrive"] = self.helpers.unit_format(
                        self.logger, mem_overdrive_level, od_unit
                    )
                except amdsmi_exception.AmdSmiLibraryException as e:
                    values_dict["mem_overdrive"] = "N/A"
                    logging.debug(
                        "Failed to get mem overdrive level for gpu %s | %s",
                        gpu_id,
                        e.get_error_info(),
                    )
        if "perf_level" in current_platform_args:
            if args.perf_level:
                try:
                    perf_level = amdsmi_interface.amdsmi_get_gpu_perf_level(args.gpu)
                    values_dict["perf_level"] = perf_level
                except amdsmi_exception.AmdSmiLibraryException as e:
                    values_dict["perf_level"] = "N/A"
                    logging.debug(
                        "Failed to get perf level for gpu %s | %s", gpu_id, e.get_error_info()
                    )
        if "xgmi_err" in current_platform_args:
            if args.xgmi_err:
                try:
                    xgmi_err_status = amdsmi_interface.amdsmi_gpu_xgmi_error_status(args.gpu)
                    values_dict["xgmi_err"] = (
                        amdsmi_interface.amdsmi_wrapper.amdsmi_xgmi_status_t__enumvalues[
                            xgmi_err_status
                        ]
                    )
                except amdsmi_exception.AmdSmiLibraryException as e:
                    values_dict["xgmi_err"] = "N/A"
                    logging.debug(
                        "Failed to get xgmi error status for gpu %s | %s",
                        gpu_id,
                        e.get_error_info(),
                    )
        if "voltage" in current_platform_args:
            if args.voltage:
                voltage_dict = {}
                all_voltage = {"vddboard": amdsmi_interface.AmdSmiVoltageType.VDDBOARD}
                for volt_type, volt_metric in all_voltage.items():
                    try:
                        voltage = amdsmi_interface.amdsmi_get_gpu_volt_metric(
                            args.gpu, volt_metric, amdsmi_interface.AmdSmiVoltageMetric.CURRENT
                        )
                        if voltage == 0:
                            voltage = "N/A"
                        voltage_dict[volt_type] = self.helpers.unit_format(
                            self.logger, voltage, "mV"
                        )
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        voltage_dict[volt_type] = "N/A"
                        logging.debug(
                            "Failed to get voltage for gpu %s | %s", gpu_id, e.get_error_info()
                        )
                values_dict["voltage"] = voltage_dict
        if "energy" in current_platform_args:
            if args.energy:
                try:
                    energy_dict = amdsmi_interface.amdsmi_get_energy_count(args.gpu)

                    energy = round(
                        energy_dict["energy_accumulator"] * energy_dict["counter_resolution"], 3
                    )
                    energy /= 1000000
                    energy = round(energy, 3)

                    energy_unit = "J"
                    if self.logger.is_human_readable_format():
                        energy = f"{energy} {energy_unit}"
                    if self.logger.is_json_format():
                        energy = {"value": energy, "unit": energy_unit}

                    values_dict["energy"] = {"total_energy_consumption": energy}
                except amdsmi_interface.AmdSmiLibraryException as e:
                    values_dict["energy"] = "N/A"
                    logging.debug(
                        "Failed to get energy usage for gpu %s | %s", args.gpu, e.get_error_info()
                    )
        if "mem_usage" in current_platform_args:
            if args.mem_usage:
                memory_usage = {
                    "total_vram": "N/A",
                    "used_vram": "N/A",
                    "free_vram": "N/A",
                    "total_visible_vram": "N/A",
                    "used_visible_vram": "N/A",
                    "free_visible_vram": "N/A",
                    "total_gtt": "N/A",
                    "used_gtt": "N/A",
                    "free_gtt": "N/A",
                }

                # Total VRAM
                try:
                    total_vram = amdsmi_interface.amdsmi_get_gpu_memory_total(
                        args.gpu, amdsmi_interface.AmdSmiMemoryType.VRAM
                    )
                    memory_usage["total_vram"] = total_vram // (1024 * 1024)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug(
                        "Failed to get total VRAM memory for gpu %s | %s",
                        gpu_id,
                        e.get_error_info(),
                    )

                try:
                    total_visible_vram = amdsmi_interface.amdsmi_get_gpu_memory_total(
                        args.gpu, amdsmi_interface.AmdSmiMemoryType.VIS_VRAM
                    )
                    memory_usage["total_visible_vram"] = total_visible_vram // (1024 * 1024)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug(
                        "Failed to get total VIS VRAM memory for gpu %s | %s",
                        gpu_id,
                        e.get_error_info(),
                    )

                try:
                    total_gtt = amdsmi_interface.amdsmi_get_gpu_memory_total(
                        args.gpu, amdsmi_interface.AmdSmiMemoryType.GTT
                    )
                    memory_usage["total_gtt"] = total_gtt // (1024 * 1024)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug(
                        "Failed to get total GTT memory for gpu %s | %s", gpu_id, e.get_error_info()
                    )

                # Used VRAM
                try:
                    used_vram = amdsmi_interface.amdsmi_get_gpu_memory_usage(
                        args.gpu, amdsmi_interface.AmdSmiMemoryType.VRAM
                    )
                    memory_usage["used_vram"] = used_vram // (1024 * 1024)

                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug(
                        "Failed to get used VRAM memory for gpu %s | %s", gpu_id, e.get_error_info()
                    )

                try:
                    used_visible_vram = amdsmi_interface.amdsmi_get_gpu_memory_usage(
                        args.gpu, amdsmi_interface.AmdSmiMemoryType.VIS_VRAM
                    )
                    memory_usage["used_visible_vram"] = used_visible_vram // (1024 * 1024)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug(
                        "Failed to get used VIS VRAM memory for gpu %s | %s",
                        gpu_id,
                        e.get_error_info(),
                    )

                try:
                    used_gtt = amdsmi_interface.amdsmi_get_gpu_memory_usage(
                        args.gpu, amdsmi_interface.AmdSmiMemoryType.GTT
                    )
                    memory_usage["used_gtt"] = used_gtt // (1024 * 1024)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug(
                        "Failed to get used GTT memory for gpu %s | %s", gpu_id, e.get_error_info()
                    )

                # Free VRAM
                if memory_usage["total_vram"] != "N/A" and memory_usage["used_vram"] != "N/A":
                    memory_usage["free_vram"] = (
                        memory_usage["total_vram"] - memory_usage["used_vram"]
                    )

                if (
                    memory_usage["total_visible_vram"] != "N/A"
                    and memory_usage["used_visible_vram"] != "N/A"
                ):
                    memory_usage["free_visible_vram"] = (
                        memory_usage["total_visible_vram"] - memory_usage["used_visible_vram"]
                    )

                if memory_usage["total_gtt"] != "N/A" and memory_usage["used_gtt"] != "N/A":
                    memory_usage["free_gtt"] = memory_usage["total_gtt"] - memory_usage["used_gtt"]

                memory_unit = "MB"
                for key, value in memory_usage.items():
                    if value != "N/A":
                        if self.logger.is_human_readable_format():
                            memory_usage[key] = f"{value} {memory_unit}"
                        if self.logger.is_json_format():
                            memory_usage[key] = {"value": value, "unit": memory_unit}

                values_dict["mem_usage"] = memory_usage
        if "throttle" in current_platform_args:
            if args.throttle:
                throttle_status = {
                    # Current values - counter/accumulated
                    "accumulation_counter": "N/A",
                    "prochot_accumulated": "N/A",
                    "ppt_accumulated": "N/A",
                    "socket_thermal_accumulated": "N/A",
                    "vr_thermal_accumulated": "N/A",
                    "hbm_thermal_accumulated": "N/A",
                    "gfx_clk_below_host_limit_accumulated": "N/A",  # deprecated
                    "gfx_clk_below_host_limit_power_accumulated": "N/A",
                    "gfx_clk_below_host_limit_thermal_accumulated": "N/A",
                    "total_gfx_clk_below_host_limit_accumulated": "N/A",
                    "low_utilization_accumulated": "N/A",
                    # violation status values - active/not active
                    "prochot_violation_status": "N/A",
                    "ppt_violation_status": "N/A",
                    "socket_thermal_violation_status": "N/A",
                    "vr_thermal_violation_status": "N/A",
                    "hbm_thermal_violation_status": "N/A",
                    "gfx_clk_below_host_limit_violation_status": "N/A",  # deprecated
                    "gfx_clk_below_host_limit_power_violation_status": "N/A",
                    "gfx_clk_below_host_limit_thermal_violation_status": "N/A",
                    "total_gfx_clk_below_host_limit_violation_status": "N/A",
                    "low_utilization_violation_status": "N/A",
                    # violation activity values - percent
                    "prochot_violation_activity": "N/A",
                    "ppt_violation_activity": "N/A",
                    "socket_thermal_violation_activity": "N/A",
                    "vr_thermal_violation_activity": "N/A",
                    "hbm_thermal_violation_activity": "N/A",
                    "gfx_clk_below_host_limit_violation_activity": "N/A",  # deprecated
                    "gfx_clk_below_host_limit_power_violation_activity": "N/A",
                    "gfx_clk_below_host_limit_thermal_violation_activity": "N/A",
                    "total_gfx_clk_below_host_limit_violation_activity": "N/A",
                    "low_utilization_violation_activity": "N/A",
                }

                try:
                    violation_status = amdsmi_interface.amdsmi_get_violation_status(args.gpu)
                    throttle_status["accumulation_counter"] = violation_status["acc_counter"]
                    throttle_status["prochot_accumulated"] = violation_status["acc_prochot_thrm"]
                    throttle_status["ppt_accumulated"] = violation_status["acc_ppt_pwr"]
                    throttle_status["socket_thermal_accumulated"] = violation_status[
                        "acc_socket_thrm"
                    ]
                    throttle_status["vr_thermal_accumulated"] = violation_status["acc_vr_thrm"]
                    throttle_status["hbm_thermal_accumulated"] = violation_status["acc_hbm_thrm"]
                    throttle_status["gfx_clk_below_host_limit_accumulated"] = violation_status[
                        "acc_gfx_clk_below_host_limit"
                    ]  # deprecated
                    throttle_status["gfx_clk_below_host_limit_power_accumulated"] = (
                        self.helpers.build_xcp_dict(
                            "acc_gfx_clk_below_host_limit_pwr", violation_status, num_partition
                        )
                    )
                    throttle_status["gfx_clk_below_host_limit_thermal_accumulated"] = (
                        self.helpers.build_xcp_dict(
                            "acc_gfx_clk_below_host_limit_thm", violation_status, num_partition
                        )
                    )
                    throttle_status["total_gfx_clk_below_host_limit_accumulated"] = (
                        self.helpers.build_xcp_dict(
                            "acc_gfx_clk_below_host_limit_total", violation_status, num_partition
                        )
                    )
                    throttle_status["low_utilization_accumulated"] = self.helpers.build_xcp_dict(
                        "acc_low_utilization", violation_status, num_partition
                    )
                    throttle_status["prochot_violation_status"] = self.helpers.build_xcp_dict(
                        "active_prochot_thrm", violation_status, num_partition
                    )
                    throttle_status["ppt_violation_status"] = self.helpers.build_xcp_dict(
                        "active_ppt_pwr", violation_status, num_partition
                    )
                    throttle_status["socket_thermal_violation_status"] = (
                        self.helpers.build_xcp_dict(
                            "active_socket_thrm", violation_status, num_partition
                        )
                    )
                    throttle_status["vr_thermal_violation_status"] = self.helpers.build_xcp_dict(
                        "active_vr_thrm", violation_status, num_partition
                    )
                    throttle_status["hbm_thermal_violation_status"] = self.helpers.build_xcp_dict(
                        "active_hbm_thrm", violation_status, num_partition
                    )
                    throttle_status["gfx_clk_below_host_limit_violation_status"] = (
                        self.helpers.build_xcp_dict(
                            "active_gfx_clk_below_host_limit", violation_status, num_partition
                        )
                    )  # deprecated
                    throttle_status["gfx_clk_below_host_limit_power_violation_status"] = (
                        self.helpers.build_xcp_dict(
                            "active_gfx_clk_below_host_limit_pwr", violation_status, num_partition
                        )
                    )
                    throttle_status["gfx_clk_below_host_limit_thermal_violation_status"] = (
                        self.helpers.build_xcp_dict(
                            "active_gfx_clk_below_host_limit_thm", violation_status, num_partition
                        )
                    )
                    throttle_status["total_gfx_clk_below_host_limit_violation_status"] = (
                        self.helpers.build_xcp_dict(
                            "active_gfx_clk_below_host_limit_total", violation_status, num_partition
                        )
                    )
                    throttle_status["low_utilization_violation_status"] = (
                        self.helpers.build_xcp_dict(
                            "active_low_utilization", violation_status, num_partition
                        )
                    )
                    throttle_status["prochot_violation_activity"] = violation_status[
                        "per_prochot_thrm"
                    ]
                    throttle_status["ppt_violation_activity"] = violation_status["per_ppt_pwr"]
                    throttle_status["socket_thermal_violation_activity"] = violation_status[
                        "per_socket_thrm"
                    ]
                    throttle_status["vr_thermal_violation_activity"] = violation_status[
                        "per_vr_thrm"
                    ]
                    throttle_status["hbm_thermal_violation_activity"] = violation_status[
                        "per_hbm_thrm"
                    ]
                    throttle_status["gfx_clk_below_host_limit_violation_activity"] = (
                        violation_status["per_gfx_clk_below_host_limit"]
                    )  # deprecated
                    throttle_status["gfx_clk_below_host_limit_power_violation_activity"] = (
                        self.helpers.build_xcp_dict(
                            "per_gfx_clk_below_host_limit_pwr", violation_status, num_partition
                        )
                    )
                    throttle_status["gfx_clk_below_host_limit_thermal_violation_activity"] = (
                        self.helpers.build_xcp_dict(
                            "per_gfx_clk_below_host_limit_thm", violation_status, num_partition
                        )
                    )
                    throttle_status["total_gfx_clk_below_host_limit_violation_activity"] = (
                        self.helpers.build_xcp_dict(
                            "per_gfx_clk_below_host_limit_total", violation_status, num_partition
                        )
                    )
                    throttle_status["low_utilization_violation_activity"] = (
                        self.helpers.build_xcp_dict(
                            "per_low_utilization", violation_status, num_partition
                        )
                    )

                except amdsmi_exception.AmdSmiLibraryException as e:
                    values_dict["throttle"] = throttle_status
                    logging.debug(
                        "Failed to get violation status' for gpu %s | %s",
                        gpu_id,
                        e.get_error_info(),
                    )

                for key, value in throttle_status.items():
                    activity_unit = ""
                    if "_activity" in key:
                        activity_unit = "%"

                    if self.logger.is_human_readable_format():
                        if isinstance(value, (list, dict)):
                            for k, v in value.items():
                                for index, activity in enumerate(v):
                                    value[k][index] = self.helpers.unit_format(
                                        self.logger, activity, activity_unit
                                    )
                                value[k] = "[" + ", ".join(value[k]) + "]"
                        elif value != "N/A":
                            throttle_status[key] = self.helpers.unit_format(
                                self.logger, value, activity_unit
                            )
                    if self.logger.is_json_format():
                        if isinstance(value, (list, dict)):
                            for k, v in value.items():
                                for index, activity in enumerate(v):
                                    value[k][index] = self.helpers.unit_format(
                                        self.logger, activity, activity_unit
                                    )
                        elif value != "N/A":
                            throttle_status[key] = self.helpers.unit_format(
                                self.logger, value, activity_unit
                            )
                values_dict["throttle"] = throttle_status

        # Store timestamp first if watching_output is enabled
        if watching_output:
            self.logger.store_output(args.gpu, "timestamp", int(time.time()))
        self.logger.store_output(args.gpu, "values", values_dict)
        self.logger.store_gpu_json_output.append(values_dict)

        if multiple_devices:
            self.logger.store_multiple_device_output()
            return  # Skip printing when there are multiple devices

        if not self.logger.is_json_format() or watching_output:
            self.logger.print_output(watching_output=watching_output)

        if watching_output:  # End of single gpu add to watch_output
            self.logger.store_watch_output(multiple_device_enabled=False)

    def metric_cpu(
        self,
        args,
        multiple_devices=False,
        cpu=None,
        cpu_power_metrics=None,
        cpu_prochot=None,
        cpu_freq_metrics=None,
        cpu_c0_res=None,
        cpu_lclk_dpm_level=None,
        cpu_pwr_svi_telemetry_rails=None,
        cpu_io_bandwidth=None,
        cpu_xgmi_bandwidth=None,
        cpu_pwr_eff_mode=None,
        cpu_metrics_ver=None,
        cpu_metrics_table=None,
        cpu_socket_energy=None,
        cpu_ddr_bandwidth=None,
        cpu_temp=None,
        cpu_dimm_temp_range_rate=None,
        cpu_dimm_pow_consumption=None,
        cpu_dimm_thermal_sensor=None,
        cpu_xgmi_pstate_range=None,
        cpu_railisofreq_policy=None,
        cpu_dfcstate_ctrl=None,
        cpu_pc6_enable=None,
        cpu_cc6_enable=None,
        cpu_dimm_sb_reg=None,
        cpu_tdelta=None,
        cpu_svi3_vr_controller_temp=None,
        cpu_enabled_commands=None,
        cpu_sdps_limit=None,
    ):
        """Get Metric information for target cpu

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            cpu (cpu_handle, optional): device_handle for target device. Defaults to None.
            cpu_power_metrics (bool, optional): Value override for args.cpu_power_metrics. Defaults to None
            cpu_prochot (bool, optional): Value override for args.cpu_prochot. Defaults to None.
            cpu_freq_metrics (bool, optional): Value override for args.cpu_freq_metrics. Defaults to None.
            cpu_c0_res (bool, optional): Value override for args.cpu_c0_res. Defaults to None
            cpu_lclk_dpm_level (list, optional): Value override for args.cpu_lclk_dpm_level. Defaults to None
            cpu_pwr_svi_telemetry_rails (list, optional): value override for args.cpu_pwr_svi_telemetry_rails. Defaults to None
            cpu_io_bandwidth (list, optional): value override for args.cpu_io_bandwidth. Defaults to None
            cpu_xgmi_bandwidth (list, optional): value override for args.cpu_xgmi_bandwidth. Defaults to None
            cpu_pwr_eff_mode (bool, optional): Value override for args.cpu_pwr_eff_mode. Defaults to None
            cpu_metrics_ver (bool, optional): Value override for args.cpu_metrics_ver. Defaults to None
            cpu_metrics_table (bool, optional): Value override for args.cpu_metrics_table. Defaults to None
            cpu_socket_energy (bool, optional): Value override for args.cpu_socket_energy. Defaults to None
            cpu_ddr_bandwidth (bool, optional): Value override for args.cpu_ddr_bandwidth. Defaults to None
            cpu_temp (bool, optional): Value override for args.cpu_temp. Defaults to None
            cpu_dimm_temp_range_rate (list, optional): Dimm address. Value override for args.cpu_dimm_temp_range_rate. Defaults to None
            cpu_dimm_pow_consumption (list, optional): Dimm address. Value override for args.cpu_dimm_pow_consumption. Defaults to None
            cpu_dimm_thermal_sensor (list, optional): Dimm address. Value override for args.cpu_dimm_thermal_sensor. Defaults to None
            cpu_xgmi_pstate_range (bool, optional): Value override for args.cpu_xgmi_pstate_range. Defaults to None
            cpu_railisofreq_policy (bool, optional): Value override for args.cpu_railisofreq_policy. Defaults to None
            cpu_dfcstate_ctrl (bool, optional): Value override for args.cpu_dfcstate_ctrl. Defaults to None
            cpu_pc6_enable (bool, optional): Value override for args.cpu_pc6_enable. Defaults to None
            cpu_cc6_enable (bool, optional): Value override for args.cpu_cc6_enable. Defaults to None
            cpu_dimm_sb_reg (list, optional): DIMM sideband register parameters [dimm_addr, lid, reg_offset, reg_space] for read. Value override for args.cpu_dimm_sb_reg. Defaults to None
            cpu_tdelta (bool, optional): Value override for args.cpu_tdelta. Defaults to None
            cpu_svi3_vr_controller_temp (list, optional): TYPE and optional RAIL_INDEX. Value override for args.cpu_svi3_vr_controller_temp. Defaults to None
            cpu_enabled_commands (bool, optional): Value override for args.cpu_enabled_commands. Defaults to None
            cpu_sdps_limit (bool, optional): Value override for args.cpu_sdps_limit. Defaults to None

        Returns:
            None: Print output via AMDSMILogger to destination
        """

        if cpu:
            args.cpu = cpu
        if cpu_power_metrics:
            args.cpu_power_metrics = cpu_power_metrics
        if cpu_prochot:
            args.cpu_prochot = cpu_prochot
        if cpu_freq_metrics:
            args.cpu_freq_metrics = cpu_freq_metrics
        if cpu_c0_res:
            args.cpu_c0_res = cpu_c0_res
        if cpu_lclk_dpm_level:
            args.cpu_lclk_dpm_level = cpu_lclk_dpm_level
        if cpu_pwr_svi_telemetry_rails:
            args.cpu_pwr_svi_telemetry_rails = cpu_pwr_svi_telemetry_rails
        if cpu_io_bandwidth:
            args.cpu_io_bandwidth = cpu_io_bandwidth
        if cpu_xgmi_bandwidth:
            args.cpu_xgmi_bandwidth = cpu_xgmi_bandwidth
        if cpu_pwr_eff_mode:
            args.cpu_pwr_eff_mode = cpu_pwr_eff_mode
        if cpu_metrics_ver:
            args.cpu_metrics_ver = cpu_metrics_ver
        if cpu_metrics_table:
            args.cpu_metrics_table = cpu_metrics_table
        if cpu_socket_energy:
            args.cpu_socket_energy = cpu_socket_energy
        if cpu_ddr_bandwidth:
            args.cpu_ddr_bandwidth = cpu_ddr_bandwidth
        if cpu_temp:
            args.cpu_temp = cpu_temp
        if cpu_dimm_temp_range_rate:
            args.cpu_dimm_temp_range_rate = cpu_dimm_temp_range_rate
        if cpu_dimm_pow_consumption:
            args.cpu_dimm_pow_consumption = cpu_dimm_pow_consumption
        if cpu_dimm_thermal_sensor:
            args.cpu_dimm_thermal_sensor = cpu_dimm_thermal_sensor
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
        if cpu_dimm_sb_reg:
            args.cpu_dimm_sb_reg = cpu_dimm_sb_reg
        if cpu_tdelta:
            args.cpu_tdelta = cpu_tdelta
        if cpu_svi3_vr_controller_temp:
            args.cpu_svi3_vr_controller_temp = cpu_svi3_vr_controller_temp
        if cpu_enabled_commands:
            args.cpu_enabled_commands = cpu_enabled_commands
        if cpu_sdps_limit:
            args.cpu_sdps_limit = cpu_sdps_limit

        # store cpu args that are applicable to the current platform
        curr_platform_cpu_args = [
            "cpu_power_metrics",
            "cpu_prochot",
            "cpu_freq_metrics",
            "cpu_c0_res",
            "cpu_lclk_dpm_level",
            "cpu_pwr_svi_telemetry_rails",
            "cpu_io_bandwidth",
            "cpu_xgmi_bandwidth",
            "cpu_pwr_eff_mode",
            "cpu_metrics_ver",
            "cpu_metrics_table",
            "cpu_socket_energy",
            "cpu_ddr_bandwidth",
            "cpu_temp",
            "cpu_dimm_temp_range_rate",
            "cpu_dimm_pow_consumption",
            "cpu_dimm_thermal_sensor",
            "cpu_xgmi_pstate_range",
            "cpu_railisofreq_policy",
            "cpu_dfcstate_ctrl",
            "cpu_pc6_enable",
            "cpu_cc6_enable",
            "cpu_dimm_sb_reg",
            "cpu_tdelta",
            "cpu_svi3_vr_controller_temp",
            "cpu_enabled_commands",
            "cpu_sdps_limit",
        ]
        curr_platform_cpu_values = [
            args.cpu_power_metrics,
            args.cpu_prochot,
            args.cpu_freq_metrics,
            args.cpu_c0_res,
            args.cpu_lclk_dpm_level,
            args.cpu_pwr_svi_telemetry_rails,
            args.cpu_io_bandwidth,
            args.cpu_xgmi_bandwidth,
            args.cpu_pwr_eff_mode,
            args.cpu_metrics_ver,
            args.cpu_metrics_table,
            args.cpu_socket_energy,
            args.cpu_ddr_bandwidth,
            args.cpu_temp,
            args.cpu_dimm_temp_range_rate,
            args.cpu_dimm_pow_consumption,
            args.cpu_dimm_thermal_sensor,
            args.cpu_xgmi_pstate_range,
            args.cpu_railisofreq_policy,
            args.cpu_dfcstate_ctrl,
            args.cpu_pc6_enable,
            args.cpu_cc6_enable,
            args.cpu_dimm_sb_reg,
            args.cpu_tdelta,
            args.cpu_svi3_vr_controller_temp,
            args.cpu_enabled_commands,
            args.cpu_sdps_limit,
        ]

        # Handle No CPU passed (fall back as this should be defined in metric())
        if args.cpu == None:
            args.cpu = self.cpu_handles

        if not any(curr_platform_cpu_values):
            for arg in curr_platform_cpu_args:
                if arg not in (
                    "cpu_lclk_dpm_level",
                    "cpu_io_bandwidth",
                    "cpu_xgmi_bandwidth",
                    "cpu_dimm_temp_range_rate",
                    "cpu_dimm_pow_consumption",
                    "cpu_dimm_thermal_sensor",
                    "cpu_dimm_sb_reg",
                    "cpu_svi3_vr_controller_temp",
                ):
                    setattr(args, arg, True)

        handled_multiple_cpus, device_handle = self.helpers.handle_cpus(
            args, self.logger, self.metric_cpu
        )
        if handled_multiple_cpus:
            return  # This function is recursive
        args.cpu = device_handle
        # get cpu id for logging
        cpu_id = self.helpers.get_cpu_id_from_device_handle(args.cpu)
        logging.debug(f"Metric Arg information for CPU {cpu_id} on {self.helpers.os_info()}")

        static_dict = {}
        if self.logger.is_json_format():
            static_dict["cpu"] = int(cpu_id)
        if args.cpu_power_metrics:
            static_dict["power_metrics"] = {}
            try:
                soc_pow = amdsmi_interface.amdsmi_get_cpu_socket_power(args.cpu)
                soc_pow = self.helpers.convert_SI_unit(float(soc_pow), self.helpers.SI_Unit.MILLI)
                static_dict["power_metrics"]["socket power"] = f"{soc_pow:.3f} W"
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["power_metrics"]["socket power"] = "N/A"
                logging.debug(
                    "Failed to get socket power for cpu %s | %s", cpu_id, e.get_error_info()
                )

            try:
                soc_pwr_limit = amdsmi_interface.amdsmi_get_cpu_socket_power_cap(args.cpu)
                soc_pwr_limit = self.helpers.convert_SI_unit(
                    float(soc_pwr_limit), self.helpers.SI_Unit.MILLI
                )
                static_dict["power_metrics"]["socket power limit"] = f"{soc_pwr_limit:.3f} W"
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["power_metrics"]["socket power limit"] = "N/A"
                logging.debug(
                    "Failed to get socket power limit for cpu %s | %s", cpu_id, e.get_error_info()
                )

            try:
                soc_max_pwr_limit = amdsmi_interface.amdsmi_get_cpu_socket_power_cap_max(args.cpu)
                soc_max_pwr_limit = self.helpers.convert_SI_unit(
                    float(soc_max_pwr_limit), self.helpers.SI_Unit.MILLI
                )
                static_dict["power_metrics"]["socket max power limit"] = (
                    f"{soc_max_pwr_limit:.3f} W"
                )
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["power_metrics"]["socket max power limit"] = "N/A"
                logging.debug(
                    "Failed to get max socket power limit for cpu %s | %s",
                    cpu_id,
                    e.get_error_info(),
                )
        if args.cpu_prochot:
            static_dict["prochot"] = {}
            try:
                proc_status = amdsmi_interface.amdsmi_get_cpu_prochot_status(args.cpu)
                static_dict["prochot"]["prochot_status"] = proc_status
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["prochot"]["prochot_status"] = "N/A"
                logging.debug(
                    "Failed to get prochot status for cpu %s | %s", cpu_id, e.get_error_info()
                )
        if args.cpu_freq_metrics:
            static_dict["freq_metrics"] = {}
            try:
                fclk_mclk = amdsmi_interface.amdsmi_get_cpu_fclk_mclk(args.cpu)
                static_dict["freq_metrics"]["fclkmemclk"] = fclk_mclk
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["freq_metrics"]["fclkmemclk"] = "N/A"
                logging.debug(
                    "Failed to get current fclkmemclk freq for cpu %s | %s",
                    cpu_id,
                    e.get_error_info(),
                )

            try:
                cclk_freq = amdsmi_interface.amdsmi_get_cpu_cclk_limit(args.cpu)
                static_dict["freq_metrics"]["cclkfreqlimit"] = cclk_freq
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["freq_metrics"]["cclkfreqlimit"] = "N/A"
                logging.debug(
                    "Failed to get current cclk freq for cpu %s | %s", cpu_id, e.get_error_info()
                )

            try:
                soc_cur_freq_limit = (
                    amdsmi_interface.amdsmi_get_cpu_socket_current_active_freq_limit(args.cpu)
                )
                static_dict["freq_metrics"]["soc_current_active_freq_limit"] = soc_cur_freq_limit
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["freq_metrics"]["soc_current_active_freq_limit"] = "N/A"
                logging.debug(
                    "Failed to get socket current freq limit for cpu %s | %s",
                    cpu_id,
                    e.get_error_info(),
                )

            try:
                soc_freq_range = amdsmi_interface.amdsmi_get_cpu_socket_freq_range(args.cpu)
                static_dict["freq_metrics"]["soc_freq_range"] = soc_freq_range
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["freq_metrics"]["soc_freq_range"] = "N/A"
                logging.debug(
                    "Failed to get socket freq range for cpu %s | %s", cpu_id, e.get_error_info()
                )
        if args.cpu_c0_res:
            static_dict["c0_residency"] = {}
            try:
                residency = amdsmi_interface.amdsmi_get_cpu_socket_c0_residency(args.cpu)
                static_dict["c0_residency"]["residency"] = residency
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["c0_residency"]["residency"] = "N/A"
                logging.debug(
                    "Failed to get C0 residency for cpu %s | %s", cpu_id, e.get_error_info()
                )
        if isinstance(args.cpu_lclk_dpm_level, list) and args.cpu_lclk_dpm_level:
            static_dict["socket_dpm"] = {}
            try:
                dpm_val = amdsmi_interface.amdsmi_get_cpu_socket_lclk_dpm_level(
                    args.cpu, args.cpu_lclk_dpm_level[0][0]
                )
                static_dict["socket_dpm"]["dpml_level_range"] = dpm_val
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["socket_dpm"]["dpml_level_range"] = "N/A"
                logging.debug(
                    "Failed to get socket dpm level range for cpu %s | %s",
                    cpu_id,
                    e.get_error_info(),
                )
        if args.cpu_pwr_svi_telemetry_rails:
            static_dict["svi_telemetry_all_rails"] = {}
            try:
                power = amdsmi_interface.amdsmi_get_cpu_pwr_svi_telemetry_all_rails(args.cpu)
                static_dict["svi_telemetry_all_rails"]["power"] = power
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["c0_residency"]["residency"] = "N/A"
                logging.debug(
                    "Failed to get svi telemetry all rails for cpu %s | %s",
                    cpu_id,
                    e.get_error_info(),
                )
        if isinstance(args.cpu_io_bandwidth, list) and args.cpu_io_bandwidth:
            static_dict["io_bandwidth"] = {}
            try:
                bandwidth = amdsmi_interface.amdsmi_get_cpu_current_io_bandwidth(
                    args.cpu, int(args.cpu_io_bandwidth[0][0]), args.cpu_io_bandwidth[0][1].upper()
                )
                static_dict["io_bandwidth"]["band_width"] = bandwidth
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["io_bandwidth"]["band_width"] = "N/A"
                logging.debug(
                    "Failed to get io bandwidth for cpu %s | %s", cpu_id, e.get_error_info()
                )
        if isinstance(args.cpu_xgmi_bandwidth, list) and args.cpu_xgmi_bandwidth:
            static_dict["xgmi_bandwidth"] = {}
            try:
                bandwidth = amdsmi_interface.amdsmi_get_cpu_current_xgmi_bw(
                    args.cpu,
                    int(args.cpu_xgmi_bandwidth[0][0]),
                    args.cpu_xgmi_bandwidth[0][1].upper(),
                )
                static_dict["xgmi_bandwidth"]["band_width"] = bandwidth
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["xgmi_bandwidth"]["band_width"] = "N/A"
                logging.debug(
                    "Failed to get xgmi bandwidth for cpu %s | %s", cpu_id, e.get_error_info()
                )
        if args.cpu_pwr_eff_mode:
            static_dict["pwr_eff_mode"] = {}
            try:
                mode, util, ppt_limit = amdsmi_interface.amdsmi_get_cpu_pwr_efficiency_mode(
                    args.cpu
                )
                ppt_limit = self.helpers.convert_SI_unit(
                    float(ppt_limit), self.helpers.SI_Unit.MILLI
                )

                # Always show mode
                static_dict["pwr_eff_mode"]["mode"] = f"{mode}"

                # Only show util and ppt_limit for modes 4 and 5
                if mode in [4, 5]:
                    static_dict["pwr_eff_mode"]["util"] = f"{util}%"
                    static_dict["pwr_eff_mode"]["ppt_limit"] = f"{ppt_limit:.3f} W"
                else:
                    # For modes 0-3, util and ppt_limit are not displayed
                    pass

            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["pwr_eff_mode"]["mode"] = "N/A"
                static_dict["pwr_eff_mode"]["util"] = "N/A"
                static_dict["pwr_eff_mode"]["ppt_limit"] = "N/A"
                logging.debug(
                    "Failed to get power efficiency mode for cpu %s | %s",
                    cpu_id,
                    e.get_error_info(),
                )
        if args.cpu_metrics_ver:
            static_dict["metric_version"] = {}
            try:
                version = amdsmi_interface.amdsmi_get_hsmp_metrics_table_version(args.cpu)
                static_dict["metric_version"]["version"] = version
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["metric_version"]["version"] = "N/A"
                logging.debug(
                    "Failed to get metrics table version for cpu %s | %s",
                    cpu_id,
                    e.get_error_info(),
                )
        if args.cpu_metrics_table:
            static_dict["metrics_table"] = {}
            try:
                cpu_fam = amdsmi_interface.amdsmi_get_cpu_family()
                static_dict["metrics_table"]["cpu_family"] = cpu_fam
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["metrics_table"]["cpu_family"] = "N/A"
                logging.debug("Failed to get cpu family | %s", e.get_error_info())
            try:
                cpu_mod = amdsmi_interface.amdsmi_get_cpu_model()
                static_dict["metrics_table"]["cpu_model"] = cpu_mod
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["metrics_table"]["cpu_model"] = "N/A"
                logging.debug("Failed to get cpu model | %s", e.get_error_info())
            try:
                cpu_metrics_table = amdsmi_interface.amdsmi_get_hsmp_metrics_table(args.cpu)
                static_dict["metrics_table"]["response"] = cpu_metrics_table
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["metrics_table"]["response"] = "N/A"
                logging.debug(
                    "Failed to get metrics table for cpu %s | %s", cpu_id, e.get_error_info()
                )
        if args.cpu_socket_energy:
            static_dict["socket_energy"] = {}
            try:
                energy = amdsmi_interface.amdsmi_get_cpu_socket_energy(args.cpu)
                static_dict["socket_energy"]["response"] = energy
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["socket_energy"]["response"] = "N/A"
                logging.debug(
                    "Failed to get socket energy for cpu %s | %s", cpu_id, e.get_error_info()
                )
        if args.cpu_ddr_bandwidth:
            static_dict["ddr_bandwidth"] = {}
            try:
                resp = amdsmi_interface.amdsmi_get_cpu_ddr_bw(args.cpu)
                static_dict["ddr_bandwidth"]["response"] = resp
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["ddr_bandwidth"]["response"] = "N/A"
                logging.debug(
                    "Failed to get ddr bandwidth for cpu %s | %s", cpu_id, e.get_error_info()
                )
        if args.cpu_temp:
            static_dict["cpu_temp"] = {}
            try:
                resp = amdsmi_interface.amdsmi_get_cpu_socket_temperature(args.cpu)
                static_dict["cpu_temp"]["response"] = resp
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["cpu_temp"]["response"] = "N/A"
                logging.debug(
                    "Failed to get cpu temperature for cpu %s | %s", cpu_id, e.get_error_info()
                )
        if isinstance(args.cpu_dimm_temp_range_rate, list) and args.cpu_dimm_temp_range_rate:
            static_dict["dimm_temp_range_rate"] = {}
            try:
                resp = amdsmi_interface.amdsmi_get_cpu_dimm_temp_range_and_refresh_rate(
                    args.cpu, args.cpu_dimm_temp_range_rate[0][0]
                )
                static_dict["dimm_temp_range_rate"]["response"] = resp
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["dimm_temp_range_rate"]["response"] = "N/A"
                logging.debug(
                    "Failed to get dimm temperature range and refresh rate for cpu %s | %s",
                    cpu_id,
                    e.get_error_info(),
                )
        if isinstance(args.cpu_dimm_pow_consumption, list) and args.cpu_dimm_pow_consumption:
            static_dict["dimm_pow_consumption"] = {}
            try:
                resp = amdsmi_interface.amdsmi_get_cpu_dimm_power_consumption(
                    args.cpu, args.cpu_dimm_pow_consumption[0][0]
                )
                static_dict["dimm_pow_consumption"]["response"] = resp
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["dimm_pow_consumption"]["response"] = "N/A"
                logging.debug(
                    "Failed to get dimm temperature range and refresh rate for cpu %s | %s",
                    cpu_id,
                    e.get_error_info(),
                )
        if isinstance(args.cpu_dimm_thermal_sensor, list) and args.cpu_dimm_thermal_sensor:
            static_dict["dimm_thermal_sensor"] = {}
            try:
                resp = amdsmi_interface.amdsmi_get_cpu_dimm_thermal_sensor(
                    args.cpu, args.cpu_dimm_thermal_sensor[0][0]
                )
                static_dict["dimm_thermal_sensor"]["response"] = resp
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["dimm_thermal_sensor"]["response"] = "N/A"
                logging.debug(
                    "Failed to get dimm temperature range and refresh rate for cpu %s | %s",
                    cpu_id,
                    e.get_error_info(),
                )
        if args.cpu_xgmi_pstate_range:
            static_dict["xgmi_pstate_range"] = {}
            try:
                pstate_range = amdsmi_interface.amdsmi_get_cpu_xgmi_pstate_range(args.cpu)
                static_dict["xgmi_pstate_range"]["min_pstate"] = pstate_range["min_pstate"]
                static_dict["xgmi_pstate_range"]["max_pstate"] = pstate_range["max_pstate"]
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["xgmi_pstate_range"]["min_pstate"] = "N/A"
                static_dict["xgmi_pstate_range"]["max_pstate"] = "N/A"
                logging.debug(
                    "Failed to get xgmi pstate range for cpu %s | %s", cpu_id, e.get_error_info()
                )
        if args.cpu_railisofreq_policy:
            static_dict["railisofreq_policy"] = {}
            try:
                cpurailisofreq_policy = amdsmi_interface.amdsmi_get_cpu_rail_isofreq_policy(
                    args.cpu
                )
                static_dict["railisofreq_policy"]["value"] = cpurailisofreq_policy
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["railisofreq_policy"]["value"] = "N/A"
                logging.debug(
                    "Failed to get cpurailiso frequency policy for cpu %s | %s",
                    cpu_id,
                    e.get_error_info(),
                )
        if args.cpu_dfcstate_ctrl:
            static_dict["dfcstate_ctrl"] = {}
            try:
                dfcstatectrl_status = amdsmi_interface.amdsmi_get_cpu_dfc_ctrl(args.cpu)
                static_dict["dfcstate_ctrl"]["value"] = dfcstatectrl_status
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["dfcstate_ctrl"]["value"] = "N/A"
                logging.debug(
                    "Failed to get dfcstate control status for cpu %s | %s",
                    cpu_id,
                    e.get_error_info(),
                )
        if args.cpu_pc6_enable:
            static_dict["pc6_enable"] = {}
            try:
                pc6_enable_status = amdsmi_interface.amdsmi_get_cpu_pc6_enable(args.cpu)
                static_dict["pc6_enable"]["value"] = pc6_enable_status
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["pc6_enable"]["value"] = "N/A"
                logging.debug(
                    "Failed to get PC6 enable status for cpu %s | %s", cpu_id, e.get_error_info()
                )
        if args.cpu_cc6_enable:
            static_dict["cc6_enable"] = {}
            try:
                cc6_enable_status = amdsmi_interface.amdsmi_get_cpu_cc6_enable(args.cpu)
                static_dict["cc6_enable"]["value"] = cc6_enable_status
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["cc6_enable"]["value"] = "N/A"
                logging.debug(
                    "Failed to get CC6 enable status for cpu %s | %s", cpu_id, e.get_error_info()
                )
        if isinstance(args.cpu_dimm_sb_reg, list) and args.cpu_dimm_sb_reg:
            static_dict["dimm_sb_reg"] = {}
            try:
                dimm_addr = args.cpu_dimm_sb_reg[0][0]
                lid = args.cpu_dimm_sb_reg[0][1]
                reg_offset = args.cpu_dimm_sb_reg[0][2]
                reg_space = args.cpu_dimm_sb_reg[0][3]
                dimm_sb_data = amdsmi_interface.amdsmi_get_cpu_dimm_sb_reg(
                    args.cpu, dimm_addr, lid, reg_offset, reg_space
                )
                static_dict["dimm_sb_reg"]["DimmAddress"] = f"0x{dimm_addr:02X}"
                static_dict["dimm_sb_reg"]["Lid"] = f"0x{lid:02X}"
                static_dict["dimm_sb_reg"]["Offset"] = f"0x{reg_offset:04X}"
                static_dict["dimm_sb_reg"]["RegSpace"] = reg_space
                static_dict["dimm_sb_reg"]["Data"] = f"0x{dimm_sb_data:08X}"
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["dimm_sb_reg"]["DimmAddress"] = f"0x{args.cpu_dimm_sb_reg[0][0]:02X}"
                static_dict["dimm_sb_reg"]["Lid"] = f"0x{args.cpu_dimm_sb_reg[0][1]:02X}"
                static_dict["dimm_sb_reg"]["Offset"] = f"0x{args.cpu_dimm_sb_reg[0][2]:04X}"
                static_dict["dimm_sb_reg"]["RegSpace"] = args.cpu_dimm_sb_reg[0][3]
                static_dict["dimm_sb_reg"]["Data"] = "N/A"
                logging.debug(
                    "Failed to read DIMM sideband register for cpu %s | %s",
                    cpu_id,
                    e.get_error_info(),
                )
        if args.cpu_tdelta:
            static_dict["tdelta"] = {}
            try:
                tdelta_value = amdsmi_interface.amdsmi_get_cpu_tdelta(args.cpu)
                static_dict["tdelta"]["value"] = tdelta_value
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["tdelta"]["value"] = "N/A"
                logging.debug(
                    "Failed to get thermal delta (TDELTA) for cpu %s | %s",
                    cpu_id,
                    e.get_error_info(),
                )
        if isinstance(args.cpu_svi3_vr_controller_temp, list) and args.cpu_svi3_vr_controller_temp:
            static_dict["svi3_vr_controller_temp"] = {}
            try:
                rail_type = args.cpu_svi3_vr_controller_temp[0][0]
                rail_index = (
                    args.cpu_svi3_vr_controller_temp[0][1]
                    if len(args.cpu_svi3_vr_controller_temp[0]) > 1
                    else AMDSMI_MAX_RAIL_INDEX
                )
                resp = amdsmi_interface.amdsmi_get_cpu_svi3_vr_controller_temp(
                    args.cpu, rail_type, rail_index
                )
                static_dict["svi3_vr_controller_temp"]["RAIL_SELECTION"] = resp["rail_selection"]
                static_dict["svi3_vr_controller_temp"]["RAIL_INDEX"] = resp["rail_index"]
                temp_celsius = resp["temperature"]
                static_dict["svi3_vr_controller_temp"]["TEMPERATURE"] = (
                    f"{temp_celsius:.1f} Degree C"
                )
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["svi3_vr_controller_temp"]["RAIL_SELECTION"] = "N/A"
                static_dict["svi3_vr_controller_temp"]["RAIL_INDEX"] = "N/A"
                static_dict["svi3_vr_controller_temp"]["TEMPERATURE"] = "N/A"
                logging.debug(
                    "Failed to get SVI3 VR controller temperature for cpu %s | %s",
                    cpu_id,
                    e.get_error_info(),
                )
        if args.cpu_enabled_commands:
            static_dict["enabled_commands"] = {}
            try:
                enabled_cmds = amdsmi_interface.amdsmi_get_cpu_enabled_commands(args.cpu)
                static_dict["enabled_commands"]["READ_ENABLED_COMMANDS_BITMASK0"] = (
                    f"0x{enabled_cmds['ReadEnabledCommandsBitMask0']:08X}"
                )
                static_dict["enabled_commands"]["READ_ENABLED_COMMANDS_BITMASK1"] = (
                    f"0x{enabled_cmds['ReadEnabledCommandsBitMask1']:08X}"
                )
                static_dict["enabled_commands"]["READ_ENABLED_COMMANDS_BITMASK2"] = (
                    f"0x{enabled_cmds['ReadEnabledCommandsBitMask2']:08X}"
                )
                static_dict["enabled_commands"]["WRITE_ENABLED_COMMANDS_BITMASK0"] = (
                    f"0x{enabled_cmds['WriteEnabledCommandsBitMask0']:08X}"
                )
                static_dict["enabled_commands"]["WRITE_ENABLED_COMMANDS_BITMASK1"] = (
                    f"0x{enabled_cmds['WriteEnabledCommandsBitMask1']:08X}"
                )
                static_dict["enabled_commands"]["WRITE_ENABLED_COMMANDS_BITMASK2"] = (
                    f"0x{enabled_cmds['WriteEnabledCommandsBitMask2']:08X}"
                )
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["enabled_commands"]["READ_ENABLED_COMMANDS_BITMASK0"] = "N/A"
                static_dict["enabled_commands"]["READ_ENABLED_COMMANDS_BITMASK1"] = "N/A"
                static_dict["enabled_commands"]["READ_ENABLED_COMMANDS_BITMASK2"] = "N/A"
                static_dict["enabled_commands"]["WRITE_ENABLED_COMMANDS_BITMASK0"] = "N/A"
                static_dict["enabled_commands"]["WRITE_ENABLED_COMMANDS_BITMASK1"] = "N/A"
                static_dict["enabled_commands"]["WRITE_ENABLED_COMMANDS_BITMASK2"] = "N/A"
                logging.debug(
                    "Failed to get enabled commands for cpu %s | %s", cpu_id, e.get_error_info()
                )
        if args.cpu_sdps_limit:
            static_dict["sdps_limit"] = {}
            try:
                sdps_limit = amdsmi_interface.amdsmi_get_cpu_sdps_limit(args.cpu)
                sdps_limit = self.helpers.convert_SI_unit(
                    float(sdps_limit), self.helpers.SI_Unit.MILLI
                )
                static_dict["sdps_limit"]["value"] = f"{sdps_limit:.3f} W"
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["sdps_limit"]["value"] = "N/A"
                logging.debug(
                    "Failed to get socket SDPS limit for cpu %s | %s", cpu_id, e.get_error_info()
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

    def metric_core(
        self,
        args,
        multiple_devices=False,
        core=None,
        core_boost_limit=None,
        core_curr_active_freq_core_limit=None,
        core_energy=None,
        core_ccd_power=None,
        core_floor_limit=None,
        core_eff_floor_limit=None,
    ):
        """Get Static information for target core

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            core (device_handle, optional): device_handle for target core. Defaults to None.
            core_boost_limit (bool, optional): Value override for args.core_boost_limit. Defaults to None
            core_curr_active_freq_core_limit (bool, optional): Value override for args.core_curr_active_freq_core_limit. Defaults to None
            core_energy (bool, optional): Value override for args.core_energy. Defaults to None
            core_ccd_power (bool, optional): Value override for args.core_ccd_power. Defaults to None
            core_floor_limit (bool, optional): Value override for args.core_floor_limit. Defaults to None
            core_eff_floor_limit (bool, optional): Value override for args.core_eff_floor_limit. Defaults to None
        Returns:
            None: Print output via AMDSMILogger to destination
        """
        if core:
            args.core = core
        if core_boost_limit:
            args.core_boost_limit = core_boost_limit
        if core_curr_active_freq_core_limit:
            args.core_curr_active_freq_core_limit = core_curr_active_freq_core_limit
        if core_energy:
            args.core_energy = core_energy
        if core_ccd_power:
            args.core_ccd_power = core_ccd_power
        if core_floor_limit:
            args.core_floor_limit = core_floor_limit
        if core_eff_floor_limit:
            args.core_eff_floor_limit = core_eff_floor_limit

        # store core args that are applicable to the current platform
        curr_platform_core_args = [
            "core_boost_limit",
            "core_curr_active_freq_core_limit",
            "core_energy",
            "core_ccd_power",
            "core_floor_limit",
            "core_eff_floor_limit",
        ]
        curr_platform_core_values = [
            args.core_boost_limit,
            args.core_curr_active_freq_core_limit,
            args.core_energy,
            args.core_ccd_power,
            args.core_floor_limit,
            args.core_eff_floor_limit,
        ]

        # Handle No cores passed
        if args.core == None:
            args.core = self.core_handles

        if not any(curr_platform_core_values):
            for arg in curr_platform_core_args:
                setattr(args, arg, True)

        handled_multiple_cores, device_handle = self.helpers.handle_cores(
            args, self.logger, self.metric_core
        )
        if handled_multiple_cores:
            return  # This function is recursive
        args.core = device_handle
        # get core id for logging
        core_id = self.helpers.get_core_id_from_device_handle(args.core)
        logging.debug(f"Static Arg information for Core {core_id} on {self.helpers.os_info()}")

        static_dict = {}
        if self.logger.is_json_format():
            static_dict["core"] = int(core_id)
        if args.core_boost_limit:
            static_dict["boost_limit"] = {}

            try:
                core_boost_limit = amdsmi_interface.amdsmi_get_cpu_core_boostlimit(args.core)
                static_dict["boost_limit"]["value"] = core_boost_limit
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["boost_limit"]["value"] = "N/A"
                logging.debug(
                    "Failed to get core boost limit for core %s | %s", core_id, e.get_error_info()
                )
        if args.core_curr_active_freq_core_limit:
            static_dict["curr_active_freq_core_limit"] = {}

            try:
                freq = amdsmi_interface.amdsmi_get_cpu_core_current_freq_limit(args.core)
                static_dict["curr_active_freq_core_limit"]["value"] = freq
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["curr_active_freq_core_limit"]["value"] = "N/A"
                logging.debug(
                    "Failed to get current active frequency core for core %s | %s",
                    core_id,
                    e.get_error_info(),
                )
        if args.core_energy:
            static_dict["core_energy"] = {}
            try:
                energy = amdsmi_interface.amdsmi_get_cpu_core_energy(args.core)
                static_dict["core_energy"]["value"] = energy
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["core_energy"]["value"] = "N/A"
                logging.debug(
                    "Failed to get core energy for core %s | %s", core_id, e.get_error_info()
                )

        if args.core_ccd_power:
            static_dict["ccd_power"] = {}
            try:
                power = amdsmi_interface.amdsmi_get_cpu_core_ccd_power(args.core)
                power = self.helpers.convert_SI_unit(float(power), self.helpers.SI_Unit.MILLI)
                static_dict["ccd_power"]["value"] = f"{power:.3f} W"
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["ccd_power"]["value"] = "N/A"
                logging.debug(
                    "Failed to get CCD power for core %s | %s", core_id, e.get_error_info()
                )

        if args.core_floor_limit:
            static_dict["floor_limit"] = {}
            try:
                core_floor_limit = amdsmi_interface.amdsmi_get_cpu_core_floor_freq_limit(args.core)
                static_dict["floor_limit"]["value"] = f"{core_floor_limit} MHz"
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["floor_limit"]["value"] = "N/A"
                logging.debug(
                    "Failed to get core floor limit for core %s | %s", core_id, e.get_error_info()
                )

        if args.core_eff_floor_limit:
            static_dict["eff_floor_limit"] = {}
            try:
                core_eff_floor_limit = amdsmi_interface.amdsmi_get_cpu_core_eff_floor_freq_limit(
                    args.core
                )
                static_dict["eff_floor_limit"]["value"] = f"{core_eff_floor_limit} MHz"
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["eff_floor_limit"]["value"] = "N/A"
                logging.debug(
                    "Failed to get core effective floor limit for core %s | %s",
                    core_id,
                    e.get_error_info(),
                )

        multiple_devices_csv_override = False
        if not self.logger.is_json_format():
            self.logger.store_core_output(args.core, "values", static_dict)
        else:
            self.logger.store_core_json_output.append(static_dict)
        if multiple_devices:
            self.logger.store_multiple_device_output()
            return  # Skip printing when there are multiple devices
        if not self.logger.is_json_format():
            self.logger.print_output(multiple_device_enabled=multiple_devices_csv_override)

    def metric_nic(
        self,
        args,
        multiple_devices=False,
        watching_output=False,
        watch=None,
        watch_time=None,
        iterations=None,
        nic=None,
        nic_power=None,
        nic_temperature=None,
        nic_errors=None,
    ):
        """Get Metric information for target nic

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            watching_output (bool, optional): True if watch argument has been set. Defaults to False.
            nic_power (bool, optional): Value override for args.nic_power. Defaults to None.
            nic_temperature (bool, optional): Value override for args.nic_temperature. Defaults to None.
            nic_errors (bool, optional): Value override for args.nic_errors. Defaults to None.

        Raises:
            IndexError: Index error if nic list is empty

        Returns:
            None: Print output via AMDSMILogger to destination
        """
        # Set args.* to passed in arguments
        if nic:
            args.nic = nic
        if watch:
            args.watch = watch
        if watch_time:
            args.watch_time = watch_time
        if iterations:
            args.iterations = iterations

        # TODO: Need to add OS wise condition for the parameters

        if nic_power:
            args.nic_power = nic_power
        if nic_temperature:
            args.nic_temperature = nic_temperature
        if nic_errors:
            args.nic_errors = nic_errors

        # Maintaining format as per other metric functions so above TODO can be resolved easily
        current_platform_args = ["nic_power", "nic_temperature", "nic_errors"]
        current_platform_values = [args.nic_power, args.nic_temperature, args.nic_errors]

        # Handle No NIC passed
        if args.nic == None:
            args.nic = self.device_handles_brcm_nics

        # Handle watch logic, will only enter this block once
        if args.watch:
            self.helpers.handle_watch(args=args, subcommand=self.metric_nic, logger=self.logger)
            return

        # Handle multiple NICs
        if isinstance(args.nic, list):
            if len(args.nic) > 1:
                # Deepcopy nics as recursion will destroy the nic list
                stored_nics = []
                for nic in args.nic:
                    stored_nics.append(nic)

                # Store output from multiple devices
                for device_handle in args.nic:
                    self.metric_nic(
                        args,
                        multiple_devices=True,
                        watching_output=watching_output,
                        nic=device_handle,
                    )

                # Reload original nics
                args.nic = stored_nics

                # Print multiple device output
                self.logger.print_output(
                    multiple_device_enabled=True, watching_output=watching_output
                )

                # Add output to total watch output and clear multiple device output
                if watching_output:
                    self.logger.store_watch_output(multiple_device_enabled=True)

                    # Flush the watching output
                    self.logger.print_output(
                        multiple_device_enabled=True, watching_output=watching_output
                    )

                return
            elif len(args.nic) == 1:
                args.nic = args.nic[0]
            else:
                raise IndexError("args.nic should not be an empty list")

        # Get nic_id for logging
        nic_id = self.helpers.get_nic_id_from_device_handle(args.nic)

        # Put the metrics table in the debug logs
        nic_metric_info = {}

        try:
            nic_metric_info = amdsmi_interface.amdsmi_get_nic_metrics_info(args.nic)
            nic_metric_str = json.dumps(nic_metric_info, indent=4)
            logging.debug("NIC Metrics table for %s | %s", nic_id, nic_metric_str)
        except amdsmi_exception.AmdSmiLibraryException as e:
            logging.debug("Unable to load NIC Metrics table for %s | %s", nic_id, e.err_info)

        logging.debug(f"Metric Arg information for NIC {nic_id} on {self.helpers.os_info()}")
        logging.debug(f"Args:   {current_platform_args}")
        logging.debug(f"Values: {current_platform_values}")

        # Set the platform applicable args to True if no args are set
        if not any(current_platform_values):
            for arg in current_platform_args:
                setattr(args, arg, True)

        # Add timestamp and store values for specified arguments
        values_dict = {}

        if "nic_power" in current_platform_args:
            if args.nic_power:
                power_dict = {}
                sysfs_blocks = {
                    "nic_power_async": "",
                    "nic_power_control": "",
                    "nic_power_runtime_active_time": "",
                    "nic_power_runtime_status": "",
                    "nic_power_runtime_usage": "",
                    "nic_power_runtime_active_kids": "",
                    "nic_power_runtime_enabled": "",
                    "nic_power_runtime_suspended_time": "",
                }

                for key in nic_metric_info.keys():
                    if key in sysfs_blocks.keys():
                        if isinstance(nic_metric_info[key], int):
                            value = nic_metric_info[key]
                        else:
                            value = (nic_metric_info[key].split("\n")[0]).upper()

                        if value == "" or value == 65535:
                            value = "N/A"
                        power_dict[key] = self.helpers.unit_format(
                            self.logger, value, sysfs_blocks[key]
                        )

                values_dict["nic_power"] = power_dict

        if "nic_temperature" in current_platform_args:
            if args.nic_temperature:
                temp_dict = {}
                sysfs_blocks = {
                    "nic_temp_crit_alarm": "",
                    "nic_temp_emergency_alarm": "",
                    "nic_temp_shutdown_alarm": "",
                    "nic_temp_max_alarm": "",
                    "nic_temp_crit": "\N{DEGREE SIGN}C",
                    "nic_temp_emergency": "\N{DEGREE SIGN}C",
                    "nic_temp_input": "\N{DEGREE SIGN}C",
                    "nic_temp_max": "\N{DEGREE SIGN}C",
                    "nic_temp_shutdown": "\N{DEGREE SIGN}C",
                }

                for key in nic_metric_info.keys():
                    if key in sysfs_blocks.keys():
                        if isinstance(nic_metric_info[key], int):
                            value = nic_metric_info[key]
                        else:
                            value = (nic_metric_info[key].split("\n")[0]).upper()

                        if value == "" or value == 65535:
                            value = "N/A"
                        temp_dict[key] = self.helpers.unit_format(
                            self.logger, value, sysfs_blocks[key]
                        )

                values_dict["nic_temperature"] = temp_dict

        if "nic_errors" in current_platform_args:
            if args.nic_errors:
                err_dict = {}
                sysfs_blocks = ["nic_dev_correctable", "nic_dev_fatal", "nic_dev_nonfatal"]

                for key in nic_metric_info.keys():
                    if key in sysfs_blocks:
                        err_dict[key] = {}
                        content_list = nic_metric_info[key].split("\n")
                        for content in content_list:
                            if content != "" and content.lower() != "n/a":
                                err_dict[key][content.split(" ")[0]] = content.split(" ")[1]

                values_dict["nic_errors"] = err_dict

        # Store timestamp first if watching_output is enabled
        if watching_output:
            self.logger.store_nic_output(args.nic, "timestamp", int(time.time()))
        self.logger.store_nic_output(args.nic, "values", values_dict)

        if multiple_devices:
            self.logger.store_multiple_device_output()
            return  # Skip printing when there are multiple devices

        self.logger.print_output(watching_output=watching_output)

        if watching_output:  # End of single gpu add to watch_output
            self.logger.store_watch_output(multiple_device_enabled=False)

    def metric_switch(
        self,
        args,
        multiple_devices=False,
        watching_output=False,
        watch=None,
        watch_time=None,
        iterations=None,
        switch=None,
        switch_power=None,
        switch_errors=None,
    ):
        """Get Metric information for target switch

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            watching_output (bool, optional): True if watch argument has been set. Defaults to False.
            switch_power (bool, optional): Value override for args.switch_power. Defaults to None.
            switch_errors (bool, optional): Value override for args.switch_errors. Defaults to None.

        Raises:
            IndexError: Index error if switch list is empty

        Returns:
            None: Print output via AMDSMILogger to destination
        """
        # Set args.* to passed in arguments
        if switch:
            args.switch = switch
        if watch:
            args.watch = watch
        if watch_time:
            args.watch_time = watch_time
        if iterations:
            args.iterations = iterations

        # TODO: Need to add OS wise condition for the parameters

        if switch_power:
            args.switch_power = switch_power
        if switch_errors:
            args.switch_errors = switch_errors

        # Maintaining format as per other metric functions so above TODO can be resolved easily
        current_platform_args = ["switch_power", "switch_errors"]
        current_platform_values = [args.switch_power, args.switch_errors]

        # Handle No SWITCH passed
        if args.switch == None:
            args.switch = self.device_handles_switchs

        # Handle watch logic, will only enter this block once
        if args.watch:
            self.helpers.handle_watch(args=args, subcommand=self.metric_switch, logger=self.logger)
            return

        # Handle multiple Switches
        if isinstance(args.switch, list):
            if len(args.switch) > 1:
                # Deepcopy switches as recursion will destroy the switch list
                stored_switches = []
                for switch in args.switch:
                    stored_switches.append(switch)

                # Store output from multiple devices
                for device_handle in args.switch:
                    self.metric_switch(
                        args,
                        multiple_devices=True,
                        watching_output=watching_output,
                        switch=device_handle,
                    )

                # Reload original switches
                args.switch = stored_switches

                # Print multiple device output
                self.logger.print_output(
                    multiple_device_enabled=True, watching_output=watching_output
                )

                # Add output to total watch output and clear multiple device output
                if watching_output:
                    self.logger.store_watch_output(multiple_device_enabled=True)

                    # Flush the watching output
                    self.logger.print_output(
                        multiple_device_enabled=True, watching_output=watching_output
                    )

                return
            elif len(args.switch) == 1:
                args.switch = args.switch[0]
            else:
                return  # intermittent issue with args.switch being an empty list. raise IndexError("args.switch should not be an empty list")

        # Get switch_id for logging
        switch_id = self.helpers.get_switch_id_from_device_handle(args.switch)

        # Put the metrics table in the debug logs
        switch_metric_info = {}
        try:
            switch_metric_info = amdsmi_interface.amdsmi_get_switch_metrics_info(args.switch)
            switch_metric_str = json.dumps(switch_metric_info, indent=4)
            logging.debug("SWITCH Metrics table for %s | %s", switch_id, switch_metric_str)
        except amdsmi_exception.AmdSmiLibraryException as e:
            logging.debug("Unable to load SWITCH Metrics table for %s | %s", switch_id, e.err_info)

        logging.debug(f"Metric Arg information for SWITCH {switch_id} on {self.helpers.os_info()}")
        logging.debug(f"Args:   {current_platform_args}")
        logging.debug(f"Values: {current_platform_values}")

        # Set the platform applicable args to True if no args are set
        if not any(current_platform_values):
            for arg in current_platform_args:
                setattr(args, arg, True)

        # Add timestamp and store values for specified arguments
        values_dict = {}

        if "switch_power" in current_platform_args:
            if args.switch_power:
                power_dict = {}
                sysfs_blocks = {
                    "brcm_power_async": "",
                    "brcm_power_control": "",
                    "brcm_power_runtime_active_kids": "",
                    "brcm_power_runtime_active_time": "",
                    "brcm_power_runtime_enabled": "",
                    "brcm_power_runtime_status": "",
                    "brcm_power_runtime_suspended_time": "",
                    "brcm_power_runtime_usage": "",
                    "brcm_power_wakeup": "",
                    "brcm_power_wakeup_abort_count": "",
                    "brcm_power_wakeup_active": "",
                    "brcm_power_wakeup_active_count": "",
                    "brcm_power_wakeup_count": "",
                    "brcm_power_wakeup_last_time_ms": "",
                    "brcm_power_wakeup_max_time_ms": "",
                    "brcm_power_wakeup_total_time_ms": "",
                }

                for key in switch_metric_info.keys():
                    if key in sysfs_blocks.keys():
                        if isinstance(switch_metric_info[key], int):
                            value = switch_metric_info[key]
                        else:
                            value = (switch_metric_info[key].split("\n")[0]).upper()

                        if value == "":
                            value = "N/A"
                        power_dict[key] = self.helpers.unit_format(
                            self.logger, value, sysfs_blocks[key]
                        )

                values_dict["switch_power"] = power_dict

        if "switch_errors" in current_platform_args:
            if args.switch_errors:
                err_dict = {}
                sysfs_blocks = [
                    "brcm_device_aer_dev_correctable",
                    "brcm_device_aer_dev_fatal",
                    "brcm_device_aer_dev_nonfatal",
                ]

                for key in switch_metric_info.keys():
                    if key in sysfs_blocks:
                        err_dict[key] = {}

                        if switch_metric_info[key] == "N/A":
                            continue

                        content_list = switch_metric_info[key].split("\n")
                        for content in content_list:
                            if content != "":
                                err_dict[key][content.split(" ")[0]] = content.split(" ")[1]

                values_dict["switch_errors"] = err_dict

        # TODO: ADD "NA" conditions in interface file
        # Store timestamp first if watching_output is enabled
        if watching_output:
            self.logger.store_switch_output(args.switch, "timestamp", int(time.time()))

        self.logger.store_switch_output(args.switch, "values", values_dict)

        if multiple_devices:
            self.logger.store_multiple_device_output()
            return  # Skip printing when there are multiple devices

        self.logger.print_output(watching_output=watching_output)

        if watching_output:  # End of single gpu add to watch_output
            self.logger.store_watch_output(multiple_device_enabled=False)

    def metric(
        self,
        args,
        multiple_devices=False,
        watching_output=False,
        gpu=None,
        nic=None,
        nic_power=None,
        nic_temperature=None,
        nic_errors=None,
        brcm_nic=None,
        switch=None,
        switch_power=None,
        switch_errors=None,
        brcm_switch=None,
        usage=None,
        watch=None,
        watch_time=None,
        iterations=None,
        power=None,
        clock=None,
        temperature=None,
        ecc=None,
        ecc_blocks=None,
        pcie=None,
        fan=None,
        voltage_curve=None,
        overdrive=None,
        perf_level=None,
        xgmi_err=None,
        energy=None,
        mem_usage=None,
        voltage=None,
        schedule=None,
        guard=None,
        guest_data=None,
        fb_usage=None,
        xgmi=None,
        cpu=None,
        cpu_power_metrics=None,
        cpu_prochot=None,
        cpu_freq_metrics=None,
        cpu_c0_res=None,
        cpu_lclk_dpm_level=None,
        cpu_pwr_svi_telemetry_rails=None,
        cpu_io_bandwidth=None,
        cpu_xgmi_bandwidth=None,
        cpu_pwr_eff_mode=None,
        cpu_metrics_ver=None,
        cpu_metrics_table=None,
        cpu_socket_energy=None,
        cpu_ddr_bandwidth=None,
        cpu_temp=None,
        cpu_dimm_temp_range_rate=None,
        cpu_dimm_pow_consumption=None,
        cpu_dimm_thermal_sensor=None,
        cpu_xgmi_pstate_range=None,
        cpu_railisofreq_policy=None,
        cpu_dfcstate_ctrl=None,
        cpu_pc6_enable=None,
        cpu_cc6_enable=None,
        cpu_dimm_sb_reg=None,
        cpu_tdelta=None,
        cpu_svi3_vr_controller_temp=None,
        cpu_enabled_commands=None,
        cpu_sdps_limit=None,
        core=None,
        core_boost_limit=None,
        core_curr_active_freq_core_limit=None,
        core_energy=None,
        core_ccd_power=None,
        core_floor_limit=None,
        core_eff_floor_limit=None,
        throttle=None,
        base_board=None,
        gpu_board=None,
    ):
        """Get Metric information for target gpu

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            watching_output (bool, optional): True if watch argument has been set. Defaults to False.
            gpu (device_handle, optional): device_handle for target device. Defaults to None.
            usage (bool, optional): Value override for args.usage. Defaults to None.
            watch (Positive int, optional): Value override for args.watch. Defaults to None.
            watch_time (Positive int, optional): Value override for args.watch_time. Defaults to None.
            iterations (Positive int, optional): Value override for args.iterations. Defaults to None.
            power (bool, optional): Value override for args.power. Defaults to None.
            clock (bool, optional): Value override for args.clock. Defaults to None.
            temperature (bool, optional): Value override for args.temperature. Defaults to None.
            ecc (bool, optional): Value override for args.ecc. Defaults to None.
            ecc_blocks (bool, optional): Value override for args.ecc. Defaults to None.
            pcie (bool, optional): Value override for args.pcie. Defaults to None.
            fan (bool, optional): Value override for args.fan. Defaults to None.
            voltage_curve (bool, optional): Value override for args.voltage_curve. Defaults to None.
            overdrive (bool, optional): Value override for args.overdrive. Defaults to None.
            perf_level (bool, optional): Value override for args.perf_level. Defaults to None.
            xgmi_err (bool, optional): Value override for args.xgmi_err. Defaults to None.
            energy (bool, optional): Value override for args.energy. Defaults to None.
            mem_usage (bool, optional): Value override for args.mem_usage. Defaults to None.
            voltage (bool, optional): Value override for args.voltage. Defaults to None.
            schedule (bool, optional): Value override for args.schedule. Defaults to None.
            guard (bool, optional): Value override for args.guard. Defaults to None.
            guest_data (bool, optional): Value override for args.guest_data. Defaults to None.
            fb_usage (bool, optional): Value override for args.fb_usage. Defaults to None.
            xgmi (bool, optional): Value override for args.xgmi. Defaults to None.

            cpu (cpu_handle, optional): device_handle for target device. Defaults to None.
            cpu_power_metrics (bool, optional): Value override for args.cpu_power_metrics. Defaults to None
            cpu_prochot (bool, optional): Value override for args.cpu_prochot. Defaults to None.
            cpu_freq_metrics (bool, optional): Value override for args.cpu_freq_metrics. Defaults to None.
            cpu_c0_res (bool, optional): Value override for args.cpu_c0_res. Defaults to None
            cpu_lclk_dpm_level (list, optional): Value override for args.cpu_lclk_dpm_level. Defaults to None
            cpu_pwr_svi_telemetry_rails (list, optional): value override for args.cpu_pwr_svi_telemetry_rails. Defaults to None
            cpu_io_bandwidth (list, optional): value override for args.cpu_io_bandwidth. Defaults to None
            cpu_xgmi_bandwidth (list, optional): value override for args.cpu_xgmi_bandwidth. Defaults to None
            cpu_pwr_eff_mode (bool, optional): Value override for args.cpu_pwr_eff_mode. Defaults to None
            cpu_metrics_ver (bool, optional): Value override for args.cpu_metrics_ver. Defaults to None
            cpu_metrics_table (bool, optional): Value override for args.cpu_metrics_table. Defaults to None
            cpu_socket_energy (bool, optional): Value override for args.cpu_socket_energy. Defaults to None
            cpu_ddr_bandwidth (bool, optional): Value override for args.cpu_ddr_bandwidth. Defaults to None
            cpu_temp (bool, optional): Value override for args.cpu_temp. Defaults to None
            cpu_dimm_temp_range_rate (list, optional): Dimm address. Value override for args.cpu_dimm_temp_range_rate. Defaults to None
            cpu_dimm_pow_consumption (list, optional): Dimm address. Value override for args.cpu_dimm_pow_consumption. Defaults to None
            cpu_dimm_thermal_sensor (list, optional): Dimm address. Value override for args.cpu_dimm_thermal_sensor. Defaults to None
            cpu_xgmi_pstate_range (bool, optional): Value override for args.cpu_xgmi_pstate_range. Defaults to None
            cpu_railisofreq_policy (bool, optional): Value override for args.cpu_railisofreq_policy. Defaults to None
            cpu_dfcstate_ctrl (bool, optional): Value override for args.cpu_dfcstate_ctrl. Defaults to None
            cpu_pc6_enable (bool, optional): Value override for args.cpu_pc6_enable. Defaults to None
            cpu_cc6_enable (bool, optional): Value override for args.cpu_cc6_enable. Defaults to None
            cpu_dimm_sb_reg (list, optional): DIMM sideband register parameters [dimm_addr, lid, reg_offset, reg_space] for read. Value override for args.cpu_dimm_sb_reg. Defaults to None
            cpu_tdelta (bool, optional): Value override for args.cpu_tdelta. Defaults to None
            cpu_svi3_vr_controller_temp (list, optional): TYPE and optional RAIL_INDEX. Value override for args.cpu_svi3_vr_controller_temp. Defaults to None
            cpu_enabled_commands (bool, optional): Value override for args.cpu_enabled_commands. Defaults to None
            cpu_sdps_limit (bool, optional): Value override for args.cpu_sdps_limit. Defaults to None

            core (device_handle, optional): device_handle for target core. Defaults to None.
            core_boost_limit (bool, optional): Value override for args.core_boost_limit. Defaults to None
            core_curr_active_freq_core_limit (bool, optional): Value override for args.core_curr_active_freq_core_limit. Defaults to None
            core_energy (bool, optional): Value override for args.core_energy. Defaults to None
            core_ccd_power (bool, optional): Value override for args.core_ccd_power. Defaults to None
            core_floor_limit (bool, optional): Value override for args.core_floor_limit. Defaults to None
            core_eff_floor_limit (bool, optional): Value override for args.core_eff_floor_limit. Defaults to None

            nic (nic_handle, optional): device_handle for target device. Defaults to None.
            nic_power (bool, optional): Value override for args.nic_power. Defaults to None.
            nic_temperature (bool, optional): Value override for args.nic_temperature. Defaults to None.
            nic_errors (bool, optional): Value override for args.nic_errors. Defaults to None.
            brcm_nic (bool, optional): Value override for args.brcm_nic. Defaults to None.
                        switch (cpu_handle, optional): device_handle for target device. Defaults to None.
            switch_power (bool, optional): Value override for args.switch_power. Defaults to None.
            switch_errors (bool, optional): Value override for args.switch_errors. Defaults to None.
                        brcm_switch (bool, optional): Value override for args.brcm_switch. Defaults to None.

        Raises:
            IndexError: Index error if gpu list is empty

        Returns:
            None: Print output via AMDSMILogger to destination
        """
        # TODO Move watch logic into here and make it driver agnostic or enable it for CPU arguments
        # Mutually exclusive args
        if gpu:
            args.gpu = gpu
        if cpu:
            args.cpu = cpu
        if core:
            args.core = core
        if self.helpers.is_brcm_nic_initialized() and (
            getattr(args, "brcm_nic", False) or brcm_nic
        ):
            args.nic_power = args.power
            args.nic_temperature = args.temperature
            args.nic_errors = args.ecc
            self.logger.output = {}
            self.logger.clear_multiple_devices_output()
            self.metric_nic(
                args,
                multiple_devices,
                watching_output,
                watch,
                watch_time,
                iterations,
                nic,
                nic_power,
                nic_temperature,
                nic_errors,
            )
            return

        if self.helpers.is_brcm_switch_initialized() and (
            getattr(args, "brcm_switch", False) or brcm_switch
        ):
            args.switch_power = args.power
            args.switch_errors = args.ecc
            self.logger.output = {}
            self.logger.clear_multiple_devices_output()
            self.metric_switch(
                args,
                multiple_devices,
                watching_output,
                watch,
                watch_time,
                iterations,
                switch,
                switch_power,
                switch_errors,
            )
            return

        # Check if a GPU argument has been set
        gpu_args_enabled = False
        gpu_attributes = [
            "usage",
            "watch",
            "watch_time",
            "iterations",
            "power",
            "clock",
            "temperature",
            "ecc",
            "ecc_blocks",
            "pcie",
            "fan",
            "voltage_curve",
            "overdrive",
            "perf_level",
            "xgmi_err",
            "energy",
            "mem_usage",
            "voltage",
            "schedule",
            "guard",
            "guest_data",
            "fb_usage",
            "xgmi",
            "throttle",
            "base_board",
            "gpu_board",
        ]
        for attr in gpu_attributes:
            if hasattr(args, attr):
                if getattr(args, attr):
                    gpu_args_enabled = True
                    break

        # Check if a CPU argument has been set
        cpu_args_enabled = False
        cpu_attributes = [
            "cpu_power_metrics",
            "cpu_prochot",
            "cpu_freq_metrics",
            "cpu_c0_res",
            "cpu_lclk_dpm_level",
            "cpu_pwr_svi_telemetry_rails",
            "cpu_io_bandwidth",
            "cpu_xgmi_bandwidth",
            "cpu_pwr_eff_mode",
            "cpu_metrics_ver",
            "cpu_metrics_table",
            "cpu_socket_energy",
            "cpu_ddr_bandwidth",
            "cpu_temp",
            "cpu_dimm_temp_range_rate",
            "cpu_dimm_pow_consumption",
            "cpu_dimm_thermal_sensor",
            "cpu_xgmi_pstate_range",
            "cpu_railisofreq_policy",
            "cpu_dfcstate_ctrl",
            "cpu_pc6_enable",
            "cpu_cc6_enable",
            "cpu_dimm_sb_reg",
            "cpu_tdelta",
            "cpu_svi3_vr_controller_temp",
            "cpu_enabled_commands",
            "cpu_sdps_limit",
        ]
        for attr in cpu_attributes:
            if hasattr(args, attr):
                if getattr(args, attr):
                    cpu_args_enabled = True
                    break

        # Check if a Core argument has been set
        core_args_enabled = False
        core_attributes = [
            "core_boost_limit",
            "core_curr_active_freq_core_limit",
            "core_energy",
            "core_ccd_power",
            "core_floor_limit",
            "core_eff_floor_limit",
        ]
        for attr in core_attributes:
            if hasattr(args, attr):
                if getattr(args, attr):
                    core_args_enabled = True
                    break

        # Handle CPU and GPU driver initialization cases
        if self.helpers.is_amd_hsmp_initialized() and self.helpers.is_amdgpu_initialized():
            logging.debug(
                "gpu_args_enabled: %s, cpu_args_enabled: %s, core_args_enabled: %s",
                gpu_args_enabled,
                cpu_args_enabled,
                core_args_enabled,
            )
            logging.debug(
                "args.gpu: %s, args.cpu: %s, args.core: %s", args.gpu, args.cpu, args.core
            )

            # If a GPU or CPU argument is provided only print out the specified device.
            if args.cpu == None and args.gpu == None and args.core == None:
                # If no args are set, print out all CPU, GPU, and Core metrics info
                if not gpu_args_enabled and not cpu_args_enabled and not core_args_enabled:
                    args.cpu = self.cpu_handles
                    args.gpu = self.device_handles
                    args.core = self.core_handles

            # Handle cases where the user has only specified an argument and no specific device
            if args.gpu == None and gpu_args_enabled:
                args.gpu = self.device_handles
            if args.cpu == None and cpu_args_enabled:
                args.cpu = self.cpu_handles
            if args.core == None and core_args_enabled:
                args.core = self.core_handles

            # Print out CPU first
            if args.cpu:
                self.metric_cpu(
                    args,
                    multiple_devices,
                    cpu,
                    cpu_power_metrics,
                    cpu_prochot,
                    cpu_freq_metrics,
                    cpu_c0_res,
                    cpu_lclk_dpm_level,
                    cpu_pwr_svi_telemetry_rails,
                    cpu_io_bandwidth,
                    cpu_xgmi_bandwidth,
                    cpu_pwr_eff_mode,
                    cpu_metrics_ver,
                    cpu_metrics_table,
                    cpu_socket_energy,
                    cpu_ddr_bandwidth,
                    cpu_temp,
                    cpu_dimm_temp_range_rate,
                    cpu_dimm_pow_consumption,
                    cpu_dimm_thermal_sensor,
                    cpu_xgmi_pstate_range,
                    cpu_railisofreq_policy,
                    cpu_dfcstate_ctrl,
                    cpu_pc6_enable,
                    cpu_cc6_enable,
                    cpu_dimm_sb_reg,
                    cpu_tdelta,
                    cpu_svi3_vr_controller_temp,
                    cpu_enabled_commands,
                    cpu_sdps_limit,
                )
            if args.core:
                self.logger.output = {}
                self.logger.clear_multiple_devices_output()
                self.metric_core(
                    args,
                    multiple_devices,
                    core,
                    core_boost_limit,
                    core_curr_active_freq_core_limit,
                    core_energy,
                    core_ccd_power,
                    core_floor_limit,
                    core_eff_floor_limit,
                )
            if args.gpu:
                self.logger.output = {}
                self.logger.clear_multiple_devices_output()
                self.metric_gpu(
                    args,
                    multiple_devices,
                    watching_output,
                    gpu,
                    usage,
                    watch,
                    watch_time,
                    iterations,
                    power,
                    clock,
                    temperature,
                    ecc,
                    ecc_blocks,
                    pcie,
                    fan,
                    voltage_curve,
                    overdrive,
                    perf_level,
                    xgmi_err,
                    energy,
                    mem_usage,
                    voltage,
                    schedule,
                    guard,
                    guest_data,
                    fb_usage,
                    xgmi,
                    throttle,
                    base_board,
                    gpu_board,
                )
        elif self.helpers.is_amd_hsmp_initialized():  # Only CPU is initialized
            if args.cpu == None and args.core == None:
                # If no args are set, print out all CPU and Core metrics info
                if not cpu_args_enabled and not core_args_enabled:
                    args.cpu = self.cpu_handles
                    args.core = self.core_handles

            if args.cpu == None and cpu_args_enabled:
                args.cpu = self.cpu_handles
            if args.core == None and core_args_enabled:
                args.core = self.core_handles

            if args.cpu:
                self.metric_cpu(
                    args,
                    multiple_devices,
                    cpu,
                    cpu_power_metrics,
                    cpu_prochot,
                    cpu_freq_metrics,
                    cpu_c0_res,
                    cpu_lclk_dpm_level,
                    cpu_pwr_svi_telemetry_rails,
                    cpu_io_bandwidth,
                    cpu_xgmi_bandwidth,
                    cpu_pwr_eff_mode,
                    cpu_metrics_ver,
                    cpu_metrics_table,
                    cpu_socket_energy,
                    cpu_ddr_bandwidth,
                    cpu_temp,
                    cpu_dimm_temp_range_rate,
                    cpu_dimm_pow_consumption,
                    cpu_dimm_thermal_sensor,
                    cpu_xgmi_pstate_range,
                    cpu_railisofreq_policy,
                    cpu_dfcstate_ctrl,
                    cpu_pc6_enable,
                    cpu_cc6_enable,
                    cpu_dimm_sb_reg,
                    cpu_tdelta,
                    cpu_svi3_vr_controller_temp,
                    cpu_enabled_commands,
                    cpu_sdps_limit,
                )
            if args.core:
                self.logger.output = {}
                self.logger.clear_multiple_devices_output()
                self.metric_core(
                    args,
                    multiple_devices,
                    core,
                    core_boost_limit,
                    core_curr_active_freq_core_limit,
                    core_energy,
                    core_ccd_power,
                    core_floor_limit,
                    core_eff_floor_limit,
                )
        elif self.helpers.is_amdgpu_initialized():  # Only GPU is initialized
            if args.gpu == None:
                args.gpu = self.device_handles

            self.logger.clear_multiple_devices_output()
            self.metric_gpu(
                args,
                multiple_devices,
                watching_output,
                gpu,
                usage,
                watch,
                watch_time,
                iterations,
                power,
                clock,
                temperature,
                ecc,
                ecc_blocks,
                pcie,
                fan,
                voltage_curve,
                overdrive,
                perf_level,
                xgmi_err,
                energy,
                mem_usage,
                voltage,
                schedule,
                throttle,
                base_board,
                gpu_board,
            )
        if self.logger.is_json_format():
            self.logger.combine_arrays_to_json()
