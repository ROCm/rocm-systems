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

import copy
import json
import logging
import time

from amdsmi_helpers import AMDSMIHelpers

from amdsmi import amdsmi_exception, amdsmi_interface


class MonitorCommands:
    def monitor(
        self,
        args,
        multiple_devices=False,
        watching_output=False,
        gpu=None,
        watch=None,
        watch_time=None,
        iterations=None,
        power_usage=None,
        temperature=None,
        base_board_temps=None,
        gpu_board_temps=None,
        gfx_util=None,
        mem_util=None,
        encoder=None,
        decoder=None,
        ecc=None,
        vram_usage=None,
        pcie=None,
        process=None,
        violation=None,
        nic=None,
        switch=None,
        brcm_nic=None,
        brcm_switch=None,
    ):
        """Populate a table with each GPU as an index to rows of targeted data

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            gpu (device_handle, optional): device_handle for target device. Defaults to None.
            nic (device_handle, optional): device_handle for target nic device. Defaults to None.
            switch (device_handle, optional): device_handle for target switch device. Defaults to None.
            watch (bool, optional): Value override for args.watch. Defaults to None.
            watch_time (int, optional): Value override for args.watch_time. Defaults to None.
            iterations (int, optional): Value override for args.iterations. Defaults to None.
            power_usage (bool, optional): Value override for args.power_usage. Defaults to None.
            temperature (bool, optional): Value override for args.temperature. Defaults to None.
            base_board_temps (bool, optional): Value override for args.base_board_temps. Defaults to None.
            gpu_board_temps (bool, optional): Value override for args.gpu_board_temps. Defaults to None.
            gfx (bool, optional): Value override for args.gfx. Defaults to None.
            mem_util (bool, optional): Value override for args.mem. Defaults to None.
            encoder (bool, optional): Value override for args.encoder. Defaults to None.
            decoder (bool, optional): Value override for args.decoder. Defaults to None.
            ecc (bool, optional): Value override for args.ecc. Defaults to None.
            vram_usage (bool, optional): Value override for args.vram_usage. Defaults to None.
            pcie (bool, optional): Value override for args.pcie. Defaults to None.
            process (bool, optional): Value override for args.process. Defaults to None.
            violation (bool, optional): Value override for args.violation. Defaults to None.
            brcm_nic (bool, optional): Value override for args.brcm_nic. Defaults to None.
            brcm_switch (bool, optional): Value override for args.brcm_switch. Defaults to None.

        Raises:
            ValueError: Value error if no gpu value is provided
            IndexError: Index error if gpu list is empty

        Return:
            Nothing
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
        if nic:
            args.nic = nic
        if switch:
            args.switch = switch

        # monitor args
        if power_usage:
            args.power_usage = power_usage
        if temperature:
            args.temperature = temperature
        if base_board_temps:
            args.base_board_temps = base_board_temps
        if gpu_board_temps:
            args.gpu_board_temps = gpu_board_temps
        if gfx_util:
            args.gfx = gfx_util
        if mem_util:
            args.mem = mem_util
        if encoder:
            args.encoder = encoder
        if decoder:
            args.decoder = decoder
        if ecc:
            args.ecc = ecc
        if vram_usage:
            args.vram_usage = vram_usage
        if pcie:
            args.pcie = pcie
        if process:
            args.process = process
        if self.helpers.is_brcm_nic_initialized() and (
            brcm_nic or getattr(args, "brcm_nic", False)
        ):
            self.metric_nic(
                args,
                multiple_devices,
                watching_output,
                watch,
                watch_time,
                iterations,
                nic=args.nic,
                nic_temperature=args.temperature,
            )
            return
        if self.helpers.is_brcm_switch_initialized() and (
            brcm_switch or getattr(args, "brcm_switch", False)
        ):
            self.metric_switch(
                args,
                multiple_devices,
                watching_output,
                watch,
                watch_time,
                iterations,
                switch=args.switch,
            )
            return
        if not self.helpers.is_virtual_os():
            if violation:
                args.violation = violation
        else:
            args.violation = False  # Disable violation for virtual OS

        # Handle No GPU passed
        if args.gpu == None:
            args.gpu = self.device_handles

        if not self.group_check_printed:
            self.helpers.check_required_groups()
            self.group_check_printed = True

        # If all arguments are False, the print all values
        # Don't include process in this logic as it's an optional edge case
        if not any(
            [
                args.power_usage,
                args.temperature,
                args.base_board_temps,
                args.gpu_board_temps,
                args.gfx,
                args.mem,
                args.encoder,
                args.decoder,
                args.ecc,
                args.vram_usage,
                args.pcie,
                args.violation,
            ]
        ):
            args.power_usage = args.temperature = args.gfx = args.mem = args.encoder = (
                args.decoder
            ) = args.vram_usage = True
            # set extra args for default output filtering
            args.default_output = True
        else:
            if not hasattr(args, "default_output"):
                args.default_output = False

        # Handle watch logic, will only enter this block once
        if args.watch:
            self.helpers.handle_watch(args=args, subcommand=self.monitor, logger=self.logger)
            return

        # Handle multiple GPUs
        if isinstance(args.gpu, list):
            if len(args.gpu) > 1:
                # Deepcopy gpus as recursion will destroy the gpu list
                stored_gpus = []
                for gpu in args.gpu:
                    stored_gpus.append(gpu)

                # When --sort-by-pid, suppress per-GPU process collection in the
                # recursive monitor() calls. Use an instance attribute so args is
                # not mutated (exception-safe with try/finally).
                sort_by_pid = getattr(args, "sort_by_pid", False) and args.process
                self._monitor_suppress_process_collection = sort_by_pid

                try:
                    # Store output from multiple devices without printing to console
                    for device_handle in args.gpu:
                        self.monitor(
                            args,
                            multiple_devices=True,
                            watching_output=watching_output,
                            gpu=device_handle,
                        )
                finally:
                    self._monitor_suppress_process_collection = False

                # Reload original gpus
                args.gpu = stored_gpus

                dual_csv_output = False
                if args.process:
                    if self.logger.is_csv_format():
                        dual_csv_output = True

                # When --sort-by-pid, route PID-grouped data through the logger:
                #   - JSON/CSV: rows appended to multiple_device_output
                #   - human-readable: ASCII table stored in _sort_by_pid_secondary_table
                if sort_by_pid:
                    self._sort_by_pid_secondary_table = None
                    self._monitor_inject_sort_by_pid(args, watching_output)

                # Flush the output
                self.logger.print_output(
                    multiple_device_enabled=True,
                    watching_output=watching_output,
                    tabular=True,
                    dual_csv_output=dual_csv_output,
                )

                # Human-readable PID-grouped table: route via logger destination
                # so --file redirection is honored (only populated in HR mode).
                if sort_by_pid and self._sort_by_pid_secondary_table:
                    self._write_via_logger_destination(self._sort_by_pid_secondary_table)

                # Add output to total watch output and clear multiple device output
                if watching_output:
                    self.logger.store_watch_output(multiple_device_enabled=True)

                return
            elif len(args.gpu) == 1:
                args.gpu = args.gpu[0]
            else:
                raise IndexError("args.gpu should not be an empty list")

        monitor_values = {}

        # Get gpu_id for logging
        gpu_id = self.helpers.get_gpu_id_from_device_handle(args.gpu)

        # Reset the table header and store the timestamp if watch output is enabled
        self.logger.table_header = "GPU"
        if watching_output:
            self.logger.store_output(args.gpu, "timestamp", int(time.time()))
            self.logger.table_header = "TIMESTAMP".rjust(10) + "  " + self.logger.table_header

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
                    "#4 - Unable to load GPU Metrics table version for %s | %s",
                    gpu_id,
                    e.get_error_info(),
                )

            try:
                # Get GPU Metrics table
                gpu_metric_debug_info = amdsmi_interface.amdsmi_get_gpu_metrics_info(args.gpu)

            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug(
                    "#5 - Unable to load GPU Metrics table for %s | %s", gpu_id, e.get_error_info()
                )

        is_partition_metrics = False  # True if we get the metrics from xcp_metrics file (amdsmi_get_gpu_partition_metrics_info)
        # get metric info only once per gpu, this will speed up data output
        try:
            # Get GPU Metrics table
            gpu_metrics_info = amdsmi_interface.amdsmi_get_gpu_metrics_info(args.gpu)
            if args.loglevel == "DEBUG":
                gpu_metric_debug_info = json.dumps(gpu_metrics_info, indent=4)
                logging.debug("GPU Metrics table for GPU %s | %s", gpu_id, gpu_metric_debug_info)
        except amdsmi_exception.AmdSmiLibraryException as e:
            gpu_metrics_info = amdsmi_interface._NA_amdsmi_get_gpu_metrics_info()
            logging.debug(
                "Unable to load GPU Metrics table for %s | %s", gpu_id, e.get_error_info()
            )

        # Workaround for XCP (partition) metrics not providing num_partition in v1.9+/v1.1+
        # Provides original formatting for earlier metric versions
        partition_metric_info = self.helpers._get_metric_version_and_partition_info(
            gpu_metrics_info, is_partition_metrics, gpu_id, args.gpu
        )
        partition_id = partition_metric_info["partition_id"]
        num_partition = partition_metric_info["num_partition"]

        # Update logger for XCP display (only if applicable)
        self.logger.table_header += "XCP".rjust(5, " ")
        self.logger.store_output(
            args.gpu, "xcp", partition_id
        )  # Store partition_id initially; can be updated via num_xcp

        # Store the pcie_bw values due to possible increase in bandwidth due to repeated gpu_metrics calls
        if args.pcie:
            try:
                pcie_info = amdsmi_interface.amdsmi_get_pcie_info(args.gpu)["pcie_metric"]
            except amdsmi_exception.AmdSmiLibraryException as e:
                pcie_info = "N/A"
                logging.debug(
                    "Failed to get pci bandwidth on gpu %s | %s", gpu_id, e.get_error_info()
                )

        power_unit = "W"

        # Resume regular ordering of values
        if args.power_usage:
            try:
                if gpu_metrics_info["current_socket_power"] != "N/A":
                    monitor_values["power_usage"] = gpu_metrics_info["current_socket_power"]
                else:  # Fallback to average_socket_power for older gpu_metrics versions
                    monitor_values["power_usage"] = gpu_metrics_info["average_socket_power"]

                if (
                    self.logger.is_human_readable_format()
                    and monitor_values["power_usage"] != "N/A"
                ):
                    monitor_values["power_usage"] = f"{monitor_values['power_usage']} {power_unit}"
                if self.logger.is_json_format() and monitor_values["power_usage"] != "N/A":
                    monitor_values["power_usage"] = {
                        "value": monitor_values["power_usage"],
                        "unit": power_unit,
                    }

            except (KeyError, amdsmi_exception.AmdSmiLibraryException) as e:
                monitor_values["power_usage"] = "N/A"
                logging.debug("Failed to get power usage on gpu %s | %s", gpu_id, e)

            self.logger.table_header += "POWER".rjust(7)

        if args.power_usage and not args.default_output:
            # Get Current Power Cap
            try:
                # assume that we're always asking for ppt0 for quick checks like this
                power_cap_info = amdsmi_interface.amdsmi_get_power_cap_info(args.gpu, 0)
                monitor_values["max_power"] = power_cap_info[
                    "power_cap"
                ]  # Get current power cap (`power_cap`) socket is set to
                # `max_power_cap`, is the maximum value it can be set to
                monitor_values["max_power"] = self.helpers.convert_SI_unit(
                    monitor_values["max_power"], AMDSMIHelpers.SI_Unit.MICRO
                )

                if self.logger.is_human_readable_format() and monitor_values["max_power"] != "N/A":
                    monitor_values["max_power"] = f"{monitor_values['max_power']} {power_unit}"
                if self.logger.is_json_format() and monitor_values["max_power"] != "N/A":
                    monitor_values["max_power"] = {
                        "value": monitor_values["max_power"],
                        "unit": power_unit,
                    }
            except amdsmi_exception.AmdSmiLibraryException as e:
                monitor_values["max_power"] = "N/A"
                logging.debug(
                    "Failed to get power cap info for gpu %s | %s", gpu_id, e.get_error_info()
                )

            self.logger.table_header += "PWR_CAP".rjust(9)

        if args.temperature:
            try:
                temperature = gpu_metrics_info["temperature_hotspot"]
                # Fallback to APU GFX temperature if hotspot is N/A
                if temperature == "N/A":
                    temperature = gpu_metrics_info.get("apu_metrics.temperature_gfx", "N/A")
                    if temperature != "N/A":
                        temperature = round(temperature)
                monitor_values["hotspot_temperature"] = temperature
            except (KeyError, amdsmi_exception.AmdSmiLibraryException) as e:
                monitor_values["hotspot_temperature"] = "N/A"
                logging.debug("Failed to get hotspot temperature on gpu %s | %s", gpu_id, e)

            try:
                temperature = gpu_metrics_info["temperature_mem"]
                monitor_values["memory_temperature"] = temperature
            except (KeyError, amdsmi_exception.AmdSmiLibraryException) as e:
                monitor_values["memory_temperature"] = "N/A"
                logging.debug("Failed to get memory temperature on gpu %s | %s", gpu_id, e)

            temp_unit_human_readable = "\N{DEGREE SIGN}C"
            temp_unit_json = "C"
            if monitor_values["hotspot_temperature"] != "N/A":
                if self.logger.is_human_readable_format():
                    monitor_values["hotspot_temperature"] = (
                        f"{monitor_values['hotspot_temperature']} {temp_unit_human_readable}"
                    )
                if self.logger.is_json_format():
                    monitor_values["hotspot_temperature"] = {
                        "value": monitor_values["hotspot_temperature"],
                        "unit": temp_unit_json,
                    }
            if monitor_values["memory_temperature"] != "N/A":
                if self.logger.is_human_readable_format():
                    monitor_values["memory_temperature"] = (
                        f"{monitor_values['memory_temperature']} {temp_unit_human_readable}"
                    )
                if self.logger.is_json_format():
                    monitor_values["memory_temperature"] = {
                        "value": monitor_values["memory_temperature"],
                        "unit": temp_unit_json,
                    }

            self.logger.table_header += "GPU_T".rjust(8)
            self.logger.table_header += "MEM_T".rjust(8)

        if args.gpu_board_temps:
            try:
                gpu_board_temp_dict = self.helpers.get_gpu_board_temperatures(
                    args.gpu, gpu_id, self.logger
                )

                temp_unit_json = "C"
                # Add GPU board sensor headers
                if gpu_board_temp_dict:
                    for temp_sensor in sorted(gpu_board_temp_dict.keys()):
                        self.logger.table_header += f"{temp_sensor}".rjust(
                            max(len(temp_sensor) + 2, 7)
                        )
                    for temp_type, temp_value in gpu_board_temp_dict.items():
                        if self.logger.is_json_format() and isinstance(temp_value, dict):
                            temp_value["unit"] = temp_unit_json
                        monitor_values[temp_type] = temp_value
            except Exception as e:
                logging.debug("Failed to get GPU board temperatures on gpu %s | %s", gpu_id, e)

        if args.base_board_temps:
            try:
                base_board_temp_dict = self.helpers.get_base_board_temperatures(
                    args.gpu, gpu_id, self.logger
                )

                temp_unit_json = "C"
                # Add base board sensor headers
                if base_board_temp_dict:
                    for temp_sensor in sorted(base_board_temp_dict.keys()):
                        self.logger.table_header += f"{temp_sensor}".rjust(
                            max(len(temp_sensor) + 2, 7)
                        )
                    for temp_type, temp_value in base_board_temp_dict.items():
                        if self.logger.is_json_format() and isinstance(temp_value, dict):
                            temp_value["unit"] = temp_unit_json
                        monitor_values[temp_type] = temp_value
            except Exception as e:
                logging.debug("Failed to get base board temperatures on gpu %s | %s", gpu_id, e)

        if args.gfx:
            try:
                gfx_clk = gpu_metrics_info["current_gfxclk"]
                # Fallback to APU current GFX clock, then average
                if gfx_clk == "N/A":
                    gfx_clk = gpu_metrics_info.get("apu_metrics.current_gfxclk", "N/A")
                if gfx_clk == "N/A":
                    gfx_clk = gpu_metrics_info.get("apu_metrics.average_gfxclk_frequency", "N/A")
                monitor_values["gfx_clk"] = gfx_clk
                freq_unit = "MHz"
                if gfx_clk != "N/A":
                    if self.logger.is_human_readable_format():
                        monitor_values["gfx_clk"] = f"{monitor_values['gfx_clk']} {freq_unit}"
                    if self.logger.is_json_format():
                        monitor_values["gfx_clk"] = {
                            "value": monitor_values["gfx_clk"],
                            "unit": freq_unit,
                        }

            except (KeyError, amdsmi_exception.AmdSmiLibraryException) as e:
                monitor_values["gfx_clk"] = "N/A"
                logging.debug("Failed to get gfx clock on gpu %s | %s", gpu_id, e)

            self.logger.table_header += "GFX_CLK".rjust(10)

            try:
                gfx_util = gpu_metrics_info["average_gfx_activity"]
                activity_unit = "%"
                if gfx_util != "N/A":
                    monitor_values["gfx"] = gfx_util
                if self.logger.is_human_readable_format():
                    monitor_values["gfx"] = f"{monitor_values['gfx']} {activity_unit}"
                if self.logger.is_json_format():
                    monitor_values["gfx"] = {"value": monitor_values["gfx"], "unit": activity_unit}
            except (KeyError, amdsmi_exception.AmdSmiLibraryException) as e:
                monitor_values["gfx"] = "N/A"
                logging.debug("Failed to get gfx utilization on gpu %s | %s", gpu_id, e)

            self.logger.table_header += "GFX%".rjust(7)

        if args.mem:
            try:
                mem_util = gpu_metrics_info["average_umc_activity"]
                activity_unit = "%"
                if mem_util != "N/A":
                    monitor_values["mem"] = mem_util
                if self.logger.is_human_readable_format():
                    monitor_values["mem"] = f"{monitor_values['mem']} {activity_unit}"
                if self.logger.is_json_format():
                    monitor_values["mem"] = {"value": monitor_values["mem"], "unit": activity_unit}
            except (KeyError, amdsmi_exception.AmdSmiLibraryException) as e:
                monitor_values["mem"] = "N/A"
                logging.debug("Failed to get mem utilization on gpu %s | %s", gpu_id, e)

            self.logger.table_header += "MEM%".rjust(7)

            # don't populate mem clock on default output
            if not args.default_output:
                try:
                    mem_clock = gpu_metrics_info["current_uclk"]
                    # Fallback to APU current UCLK, then average
                    if mem_clock == "N/A":
                        mem_clock = gpu_metrics_info.get("apu_metrics.current_uclk", "N/A")
                    if mem_clock == "N/A":
                        mem_clock = gpu_metrics_info.get(
                            "apu_metrics.average_uclk_frequency", "N/A"
                        )
                    monitor_values["mem_clock"] = mem_clock
                    freq_unit = "MHz"
                    if mem_clock != "N/A":
                        if self.logger.is_human_readable_format():
                            monitor_values["mem_clock"] = (
                                f"{monitor_values['mem_clock']} {freq_unit}"
                            )
                        if self.logger.is_json_format():
                            monitor_values["mem_clock"] = {
                                "value": monitor_values["mem_clock"],
                                "unit": freq_unit,
                            }
                except (KeyError, amdsmi_exception.AmdSmiLibraryException) as e:
                    monitor_values["mem_clock"] = "N/A"
                    logging.debug("Failed to get mem clock on gpu %s | %s", gpu_id, e)

                self.logger.table_header += "MEM_CLOCK".rjust(11)

        if args.encoder:
            # TODO: The encoding utilization is in progress for Navi. Note: MI3x ASICs only support decoding.
            try:
                # Get List of vcn activity values
                encoder_util = "N/A"  # Not yet implemented
                encoding_activity_avg = []
                for value in encoder_util:
                    if isinstance(value, int):
                        encoding_activity_avg.append(value)

                # Averaging the possible encoding activity values
                if encoding_activity_avg:
                    encoding_activity_avg = round(
                        sum(encoding_activity_avg) / len(encoding_activity_avg)
                    )
                else:
                    encoding_activity_avg = "N/A"

                monitor_values["encoder"] = encoding_activity_avg

                activity_unit = "%"
                if monitor_values["encoder"] != "N/A":
                    if self.logger.is_human_readable_format():
                        monitor_values["encoder"] = f"{monitor_values['encoder']} {activity_unit}"
                    if self.logger.is_json_format():
                        monitor_values["encoder"] = {
                            "value": monitor_values["encoder"],
                            "unit": activity_unit,
                        }
            except amdsmi_exception.AmdSmiLibraryException as e:
                monitor_values["encoder"] = "N/A"
                logging.debug(
                    "Failed to get encoder utilization on gpu %s | %s", gpu_id, e.get_error_info()
                )

            self.logger.table_header += "ENC%".rjust(7)

        if args.decoder:
            try:
                # Get List of vcn activity values
                # Note: MI3x ASICs only support decoding, so the vcn_activity/vcn_busy
                #       is used for decoding activity.
                decoder_util = gpu_metrics_info["vcn_activity"]
                if (
                    gpu_metrics_info["vcn_activity"][0] == "N/A"
                    and gpu_metrics_info["xcp_stats.vcn_busy"][partition_id][0] != "N/A"
                ):
                    decoder_util = gpu_metrics_info["xcp_stats.vcn_busy"][partition_id]
                decoding_activity_avg = self.helpers.average_flattened_ints(
                    decoder_util, context="decoder_util"
                )
                monitor_values["decoder"] = decoding_activity_avg

                activity_unit = "%"
                if monitor_values["decoder"] != "N/A":
                    if self.logger.is_human_readable_format():
                        monitor_values["decoder"] = f"{monitor_values['decoder']} {activity_unit}"
                    if self.logger.is_json_format():
                        monitor_values["decoder"] = {
                            "value": monitor_values["decoder"],
                            "unit": activity_unit,
                        }
            except (KeyError, amdsmi_exception.AmdSmiLibraryException) as e:
                monitor_values["decoder"] = "N/A"
                logging.debug("Failed to get decoder utilization on gpu %s | %s", gpu_id, e)

            self.logger.table_header += "DEC%".rjust(7)

        if (args.encoder or args.decoder) and not args.default_output:
            try:
                vclock = gpu_metrics_info["current_vclk0"]
                # Fallback to APU average VCLK if current is N/A
                if vclock == "N/A":
                    vclock = gpu_metrics_info.get("apu_metrics.average_vclk_frequency", "N/A")
                monitor_values["vclock"] = vclock

                freq_unit = "MHz"
                if vclock != "N/A":
                    if self.logger.is_human_readable_format():
                        monitor_values["vclock"] = f"{monitor_values['vclock']} {freq_unit}"
                    if self.logger.is_json_format():
                        monitor_values["vclock"] = {
                            "value": monitor_values["vclock"],
                            "unit": freq_unit,
                        }
            except (KeyError, amdsmi_exception.AmdSmiLibraryException) as e:
                monitor_values["vclock"] = "N/A"
                logging.debug("Failed to get dclock on gpu %s | %s", gpu_id, e)

            self.logger.table_header += "VCLOCK".rjust(10)

            try:
                dclock = gpu_metrics_info["current_dclk0"]
                monitor_values["dclock"] = dclock

                freq_unit = "MHz"
                if dclock != "N/A":
                    if self.logger.is_human_readable_format():
                        monitor_values["dclock"] = f"{monitor_values['dclock']} {freq_unit}"
                    if self.logger.is_json_format():
                        monitor_values["dclock"] = {
                            "value": monitor_values["dclock"],
                            "unit": freq_unit,
                        }
            except (KeyError, amdsmi_exception.AmdSmiLibraryException) as e:
                monitor_values["dclock"] = "N/A"
                logging.debug("Failed to get dclock on gpu %s | %s", gpu_id, e)

            self.logger.table_header += "DCLOCK".rjust(10)

        if args.ecc:
            try:
                ecc = amdsmi_interface.amdsmi_get_gpu_total_ecc_count(args.gpu)
                monitor_values["single_bit_ecc"] = ecc["correctable_count"]
                monitor_values["double_bit_ecc"] = ecc["uncorrectable_count"]
            except amdsmi_exception.AmdSmiLibraryException as e:
                monitor_values["ecc"] = "N/A"
                logging.debug("Failed to get ecc on gpu %s | %s", gpu_id, e.get_error_info())

            self.logger.table_header += "SINGLE_ECC".rjust(12)
            self.logger.table_header += "DOUBLE_ECC".rjust(12)

            try:
                pcie_metric = amdsmi_interface.amdsmi_get_pcie_info(args.gpu)["pcie_metric"]
                logging.debug("PCIE Metric for %s | %s", gpu_id, pcie_metric)
                monitor_values["pcie_replay"] = pcie_metric["pcie_replay_count"]
            except amdsmi_exception.AmdSmiLibraryException as e:
                monitor_values["pcie_replay"] = "N/A"
                logging.debug(
                    "Failed to get gpu_metrics pcie replay counter on gpu %s | %s",
                    gpu_id,
                    e.get_error_info(),
                )

            if monitor_values["pcie_replay"] == "N/A":
                try:
                    pcie_replay = amdsmi_interface.amdsmi_get_gpu_pci_replay_counter(args.gpu)
                    monitor_values["pcie_replay"] = pcie_replay
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug(
                        "Failed to get sysfs pcie replay counter on gpu %s | %s",
                        gpu_id,
                        e.get_error_info(),
                    )

            self.logger.table_header += "PCIE_REPLAY".rjust(13)

        if args.vram_usage and not args.default_output:
            mem_type, mem_type_name = self.helpers.get_apu_memory_type_and_name(args.gpu, gpu_id)

            try:
                mem_used = amdsmi_interface.amdsmi_get_gpu_memory_usage(args.gpu, mem_type) // (
                    1024 * 1024
                )
                mem_total = amdsmi_interface.amdsmi_get_gpu_memory_total(args.gpu, mem_type) // (
                    1024 * 1024
                )
                monitor_values["vram_used"] = mem_used
                monitor_values["vram_free"] = mem_total - mem_used
                monitor_values["vram_total"] = mem_total
                if mem_total != 0:
                    monitor_values["vram_percent"] = round((mem_used / mem_total) * 100, 2)
                else:
                    monitor_values["vram_percent"] = "N/A"

                mem_usage_unit = "MB"
                mem_percent_unit = "%"
                if self.logger.is_human_readable_format():
                    monitor_values["vram_used"] = f"{monitor_values['vram_used']} {mem_usage_unit}"
                    monitor_values["vram_free"] = f"{monitor_values['vram_free']} {mem_usage_unit}"
                    monitor_values["vram_total"] = (
                        f"{monitor_values['vram_total']} {mem_usage_unit}"
                    )
                    monitor_values["vram_percent"] = (
                        f"{monitor_values['vram_percent']} {mem_percent_unit}"
                    )
                if self.logger.is_json_format():
                    monitor_values["vram_used"] = {
                        "value": monitor_values["vram_used"],
                        "unit": mem_usage_unit,
                    }
                    monitor_values["vram_free"] = {
                        "value": monitor_values["vram_free"],
                        "unit": mem_usage_unit,
                    }
                    monitor_values["vram_total"] = {
                        "value": monitor_values["vram_total"],
                        "unit": mem_usage_unit,
                    }
                    monitor_values["vram_percent"] = {
                        "value": monitor_values["vram_percent"],
                        "unit": mem_percent_unit,
                    }
            except amdsmi_exception.AmdSmiLibraryException as e:
                monitor_values["vram_used"] = "N/A"
                monitor_values["vram_free"] = "N/A"
                monitor_values["vram_total"] = "N/A"
                monitor_values["vram_percent"] = "N/A"
                logging.debug(
                    "Failed to get %s memory usage on gpu %s | %s",
                    mem_type_name.lower(),
                    gpu_id,
                    e.get_error_info(),
                )

            # Use appropriate headers based on memory type
            self.logger.table_header += f"{mem_type_name}_USED".rjust(11)
            self.logger.table_header += f"{mem_type_name}_FREE".rjust(12)
            self.logger.table_header += f"{mem_type_name}_TOTAL".rjust(12)
            self.logger.table_header += f"{mem_type_name}%".rjust(9)

        if args.vram_usage and args.default_output:
            mem_type, mem_type_name = self.helpers.get_apu_memory_type_and_name(args.gpu, gpu_id)

            try:
                mem_used = amdsmi_interface.amdsmi_get_gpu_memory_usage(args.gpu, mem_type) // (
                    1024 * 1024
                )
                mem_total = amdsmi_interface.amdsmi_get_gpu_memory_total(args.gpu, mem_type) // (
                    1024 * 1024
                )
                mem_usage_unit = "GB"
                if self.logger.is_json_format():
                    monitor_values["vram_used"] = {
                        "value": round(mem_used / 1024, 1),
                        "unit": mem_usage_unit,
                    }
                    monitor_values["vram_total"] = {
                        "value": round(mem_total / 1024, 1),
                        "unit": mem_usage_unit,
                    }
                elif self.logger.is_csv_format():
                    monitor_values["vram_used"] = round(mem_used / 1024, 1)
                    monitor_values["vram_total"] = round(mem_total / 1024, 1)
                else:
                    monitor_values["vram_usage"] = (
                        f"{mem_used / 1024:5.1f}/{mem_total / 1024:5.1f} {mem_usage_unit}".rjust(
                            16, " "
                        )
                    )
            except amdsmi_exception.AmdSmiLibraryException as e:
                if self.logger.is_json_format():
                    monitor_values["vram_used"] = "N/A"
                    monitor_values["vram_total"] = "N/A"
                else:
                    monitor_values["vram_usage"] = "N/A"
                logging.debug(
                    "Failed to get %s memory usage on gpu %s | %s",
                    mem_type_name.lower(),
                    gpu_id,
                    e.get_error_info(),
                )

            # Use appropriate header based on memory type
            header_name = f"{mem_type_name}_USAGE"
            self.logger.table_header += header_name.rjust(16)

        if args.pcie:
            if pcie_info != "N/A":
                pcie_bw_unit = "Mb/s"
                monitor_values["pcie_bw"] = self.helpers.unit_format(
                    self.logger, pcie_info["pcie_bandwidth"], pcie_bw_unit
                )
            else:
                monitor_values["pcie_bw"] = pcie_info

            self.logger.table_header += "PCIE_BW".rjust(12)

        # initialize dual_csv_format; applicable to process only
        dual_csv_output = False

        # Store process list separately
        # _monitor_suppress_process_collection is set by the multi-GPU dispatcher
        # when --sort-by-pid is active; per-GPU collection is replaced by the
        # PID-grouped aggregation rendered after the loop.
        if args.process and not getattr(self, "_monitor_suppress_process_collection", False):
            # Populate initial processes
            try:
                process_list = amdsmi_interface.amdsmi_get_gpu_process_list(args.gpu)
            except amdsmi_exception.AmdSmiLibraryException as e:
                if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                    raise PermissionError("Command requires elevation") from e
                logging.debug(
                    "Failed to get process list for gpu %s | %s", gpu_id, e.get_error_info()
                )
                raise e

            try:
                num_compute_units = amdsmi_interface.amdsmi_get_gpu_asic_info(args.gpu)[
                    "num_compute_units"
                ]
            except (KeyError, amdsmi_exception.AmdSmiLibraryException) as e:
                num_compute_units = "N/A"
                logging.debug(
                    "Failed to get num compute units for gpu %s | %s", gpu_id, e.get_error_info()
                )

            # Clean processes dictionary
            filtered_process_values = []
            for process_info in process_list:
                process_info.pop("engine_usage")  # Remove 'engine_usage' value
                process_info["mem_usage"] = process_info.pop("mem")
                process_info["cu_occupancy"] = process_info.pop("cu_occupancy")
                process_info["sdma_usage"] = process_info.pop("sdma_usage")
                process_info["evicted_time"] = process_info.pop("evicted_time")

                memory_usage_unit = "B"
                evicted_time_unit = "ms"
                sdma_usage_unit = "us"

                if self.logger.is_human_readable_format():
                    process_info["mem_usage"] = self.helpers.convert_bytes_to_readable(
                        process_info["mem_usage"]
                    )
                    for usage_metric in process_info["memory_usage"]:
                        process_info["memory_usage"][usage_metric] = (
                            self.helpers.convert_bytes_to_readable(
                                process_info["memory_usage"][usage_metric]
                            )
                        )
                    memory_usage_unit = ""

                process_info["mem_usage"] = self.helpers.unit_format(
                    self.logger, process_info["mem_usage"], memory_usage_unit
                )

                if self.logger.is_human_readable_format():
                    process_info["evicted_time"] = self.helpers.convert_time_to_readable(
                        process_info["evicted_time"], "ms"
                    )
                else:
                    process_info["evicted_time"] = self.helpers.unit_format(
                        self.logger, process_info["evicted_time"], evicted_time_unit
                    )

                if self.logger.is_human_readable_format():
                    process_info["sdma_usage"] = self.helpers.convert_time_to_readable(
                        process_info["sdma_usage"], "us"
                    )
                else:
                    process_info["sdma_usage"] = self.helpers.unit_format(
                        self.logger, process_info["sdma_usage"], sdma_usage_unit
                    )

                for usage_metric in process_info["memory_usage"]:
                    process_info["memory_usage"][usage_metric] = self.helpers.unit_format(
                        self.logger, process_info["memory_usage"][usage_metric], memory_usage_unit
                    )

                if "cu_occupancy" in process_info:
                    try:
                        cu_occupancy = process_info["cu_occupancy"]
                        if (
                            num_compute_units != "N/A"
                            and num_compute_units > 0
                            and cu_occupancy != "N/A"
                        ):
                            cu_percentage = round((cu_occupancy / num_compute_units) * 100, 1)
                            process_info["cu_occupancy"] = self.helpers.unit_format(
                                self.logger, cu_percentage, "%"
                            )
                        else:
                            process_info["cu_occupancy"] = "N/A"
                    except Exception as e:
                        process_info["cu_occupancy"] = "N/A"
                        logging.debug(
                            "Failed to calculate cu_occupancy percentage for GPU %s | %s",
                            gpu_id,
                            str(e),
                        )

                filtered_process_values.append({"process_info": process_info})

            # If no processes are populated then we populate an N/A placeholder
            if not filtered_process_values:
                logging.debug("Monitor - Failed to detect any process on gpu %s", gpu_id)
                filtered_process_values.append({"process_info": "N/A"})

            for index, process in enumerate(filtered_process_values):
                if process["process_info"] == "N/A":
                    filtered_process_values[index]["process_info"] = "No running processes detected"

            # Build the process table's title and header
            self.logger.secondary_table_title = "PROCESS INFO"
            self.logger.secondary_table_header = (
                "GPU".rjust(3)
                + "NAME".rjust(19)
                + "PID".rjust(9)
                + "GTT_MEM".rjust(10)
                + "CPU_MEM".rjust(10)
                + "VRAM_MEM".rjust(10)
                + "MEM_USG".rjust(10)
                + "CU%".rjust(9)
                + "SDMA".rjust(8)
                + "EVICT".rjust(8)
            )

            if watching_output:
                self.logger.secondary_table_header = (
                    "TIMESTAMP".rjust(10) + "  " + self.logger.secondary_table_header
                )

            logging.debug(f"Monitor - Process Info for GPU {gpu_id} | {filtered_process_values}")

            if self.logger.is_json_format():
                self.logger.store_output(args.gpu, "process_list", filtered_process_values)

            if self.logger.is_human_readable_format():
                # Print out process in flattened format
                # The logger detects if process list is present and pulls it out and prints
                #  that table with timestamp, gpu, and prints headers separately
                self.logger.store_output(args.gpu, "process_list", filtered_process_values)

            if self.logger.is_csv_format():
                dual_csv_output = True
                # The logger detects if process list is present and pulls it out and prints
                #  that table with timestamp, gpu, and prints headers separately
                self.logger.store_output(args.gpu, "process_list", filtered_process_values)

        ###################
        ### XCP Metrics ###
        ###################
        # Must come after process list - XCP detail is a multi-dimensional array, which is displayed
        # in tabular format with XCP values for same gpu shown on multiple lines.
        if args.violation:
            violation_status = {
                "pviol": "N/A",
                "tviol": "N/A",
                "tviol_active": "N/A",
                "phot_tviol": "N/A",
                "vr_tviol": "N/A",
                "hbm_tviol": "N/A",
                "gfx_clkviol": "N/A",
                "gfxclk_pviol": "N/A",
                "gfxclk_tviol": "N/A",
                "gfxclk_totalviol": "N/A",
                "low_utilviol": "N/A",
            }
            try:
                violations = amdsmi_interface.amdsmi_get_violation_status(args.gpu)
                violation_status["pviol"] = violations["per_ppt_pwr"]
                violation_status["tviol"] = violations["per_socket_thrm"]
                violation_status["tviol_active"] = violations["active_socket_thrm"]
                violation_status["phot_tviol"] = violations["per_prochot_thrm"]
                violation_status["vr_tviol"] = violations["per_vr_thrm"]
                violation_status["hbm_tviol"] = violations["per_hbm_thrm"]
                violation_status["gfx_clkviol"] = violations["per_gfx_clk_below_host_limit"]
                violation_status["gfxclk_pviol"] = violations["per_gfx_clk_below_host_limit_pwr"]
                violation_status["gfxclk_tviol"] = violations["per_gfx_clk_below_host_limit_thm"]
                violation_status["gfxclk_totalviol"] = violations[
                    "per_gfx_clk_below_host_limit_total"
                ]
                violation_status["low_utilviol"] = violations["per_low_utilization"]
            except amdsmi_exception.AmdSmiLibraryException as e:
                monitor_values["pviol"] = violation_status["pviol"]
                monitor_values["tviol"] = violation_status["tviol"]
                monitor_values["tviol_active"] = violation_status["tviol_active"]
                monitor_values["phot_tviol"] = violation_status["phot_tviol"]
                monitor_values["vr_tviol"] = violation_status["vr_tviol"]
                monitor_values["hbm_tviol"] = violation_status["hbm_tviol"]
                monitor_values["gfx_clkviol"] = violation_status["gfx_clkviol"]
                monitor_values["gfxclk_pviol"] = violation_status["gfxclk_pviol"]
                monitor_values["gfxclk_tviol"] = violation_status["gfxclk_tviol"]
                monitor_values["gfxclk_totalviol"] = violation_status["gfxclk_totalviol"]
                monitor_values["low_utilviol"] = violation_status["low_utilviol"]
                logging.debug(
                    "Failed to get violation status on gpu %s | %s", gpu_id, e.get_error_info()
                )
            violation_status_unit = "%"
            kPVIOL_MAX_WIDTH = 7
            kTVIOL_MAX_WIDTH = 7
            kTVIOL_ACTIVE_MAX_WIDTH = 14
            kPHOT_MAX_WIDTH = 12
            kVR_MAX_WIDTH = 10
            kHBM_MAX_WIDTH = 11
            kGFXC_MAX_WIDTH = 13
            kGFXC_PVIOL_MAX_WIDTH = 58
            kGFXC_TVIOL_MAX_WIDTH = kGFXC_PVIOL_MAX_WIDTH
            kGFXC_TOTALVIOL_MAX_WIDTH = kGFXC_PVIOL_MAX_WIDTH
            kLOW_UTILVIOL_MAX_WIDTH = kGFXC_PVIOL_MAX_WIDTH

            for key, value in violation_status.items():
                if not isinstance(value, list):
                    if value != "N/A":
                        if key == "tviol_active" or key == "xcp":
                            monitor_values[key] = value
                        else:
                            monitor_values[key] = self.helpers.unit_format(
                                self.logger, violation_status[key], violation_status_unit
                            )
                    else:
                        monitor_values[key] = violation_status[key]
                else:
                    if num_partition != "N/A":
                        # these are one after another, in order to display each in sub-sections
                        new_xcp_dict = {}
                        for current_xcp in range(num_partition):
                            new_xcp_dict[f"xcp_{current_xcp}"] = self.helpers.unit_format(
                                self.logger, value[current_xcp], "%"
                            )
                        monitor_values[key] = new_xcp_dict
                    else:
                        monitor_values[key] = value[0] if value else "N/A"
            # save deep copy of monitor values, used later to grab xcp specific values
            monitor_values_deepcopy = copy.deepcopy(monitor_values)

            self.logger.table_header += "PVIOL".rjust(kPVIOL_MAX_WIDTH, " ")
            self.logger.table_header += "TVIOL".rjust(kTVIOL_MAX_WIDTH, " ")
            self.logger.table_header += "TVIOL_ACTIVE".rjust(kTVIOL_ACTIVE_MAX_WIDTH, " ")
            self.logger.table_header += "PHOT_TVIOL".rjust(kPHOT_MAX_WIDTH, " ")
            self.logger.table_header += "VR_TVIOL".rjust(kVR_MAX_WIDTH, " ")
            self.logger.table_header += "HBM_TVIOL".rjust(kHBM_MAX_WIDTH, " ")
            self.logger.table_header += "GFX_CLKVIOL".rjust(kGFXC_MAX_WIDTH, " ")
            self.logger.table_header += "GFXCLK_PVIOL".rjust(kGFXC_PVIOL_MAX_WIDTH, " ")
            self.logger.table_header += "GFXCLK_TVIOL".rjust(kGFXC_TVIOL_MAX_WIDTH, " ")
            self.logger.table_header += "GFXCLK_TOTALVIOL".rjust(kGFXC_TOTALVIOL_MAX_WIDTH, " ")
            self.logger.table_header += "LOW_UTILVIOL".rjust(kLOW_UTILVIOL_MAX_WIDTH, " ")

            # Print/capture by XCPs
            if num_partition != "N/A" and partition_id == 0:
                current_xcp = 0
                while current_xcp in range(num_partition) or current_xcp == 0:
                    if not multiple_devices and watching_output and current_xcp == 0:
                        # Need to clear output for single device, otherwise while watching output
                        # XCP detail will continue stacking on top of each other
                        self.logger.clear_multiple_devices_output()

                    if watching_output:
                        self.logger.store_output(args.gpu, "timestamp", int(time.time()))

                    if current_xcp != 0:  # set all other values without XCP stats to N/A
                        self.logger.store_output(args.gpu, "xcp", current_xcp)
                        monitor_values["pviol"] = "N/A"
                        monitor_values["tviol"] = "N/A"
                        monitor_values["tviol_active"] = "N/A"
                        monitor_values["phot_tviol"] = "N/A"
                        monitor_values["vr_tviol"] = "N/A"
                        monitor_values["hbm_tviol"] = "N/A"
                        monitor_values["gfx_clkviol"] = "N/A"
                        for (
                            k,
                            _,
                        ) in monitor_values.items():  # change other keys to "N/A" since we should have all applicable XCP stats
                            # eg. amd-smi monitor -p -t -V should only show XCP info for violations
                            # below primary device
                            if k != "xcp" and k not in [
                                "gfxclk_pviol",
                                "gfxclk_tviol",
                                "gfxclk_totalviol",
                                "low_utilviol",
                            ]:
                                monitor_values[k] = "N/A"

                    if isinstance(monitor_values_deepcopy["gfxclk_pviol"], dict):
                        monitor_values["gfxclk_pviol"] = monitor_values_deepcopy["gfxclk_pviol"][
                            f"xcp_{current_xcp}"
                        ]
                    if isinstance(monitor_values_deepcopy["gfxclk_tviol"], dict):
                        monitor_values["gfxclk_tviol"] = monitor_values_deepcopy["gfxclk_tviol"][
                            f"xcp_{current_xcp}"
                        ]
                    if isinstance(monitor_values_deepcopy["gfxclk_totalviol"], dict):
                        monitor_values["gfxclk_totalviol"] = monitor_values_deepcopy[
                            "gfxclk_totalviol"
                        ][f"xcp_{current_xcp}"]
                    if isinstance(monitor_values_deepcopy["low_utilviol"], dict):
                        monitor_values["low_utilviol"] = monitor_values_deepcopy["low_utilviol"][
                            f"xcp_{current_xcp}"
                        ]

                    if self.logger.is_human_readable_format():
                        monitor_values["pviol"] = monitor_values["pviol"]
                        monitor_values["tviol"] = monitor_values["tviol"]
                        monitor_values["phot_tviol"] = monitor_values["phot_tviol"]
                        monitor_values["vr_tviol"] = monitor_values["vr_tviol"]
                        monitor_values["hbm_tviol"] = monitor_values["hbm_tviol"]
                        monitor_values["gfx_clkviol"] = monitor_values["gfx_clkviol"]
                        monitor_values["gfxclk_pviol"] = str(
                            monitor_values["gfxclk_pviol"]
                        ).replace("'", "")
                        monitor_values["gfxclk_tviol"] = str(
                            monitor_values["gfxclk_tviol"]
                        ).replace("'", "")
                        monitor_values["gfxclk_totalviol"] = str(
                            monitor_values["gfxclk_totalviol"]
                        ).replace("'", "")
                        monitor_values["low_utilviol"] = str(
                            monitor_values["low_utilviol"]
                        ).replace("'", "")
                    self.logger.store_output(args.gpu, "values", monitor_values)
                    self.logger.store_multiple_device_output()
                    current_xcp += 1
            else:
                self.logger.store_output(args.gpu, "xcp", partition_id)
                self.logger.store_output(args.gpu, "values", monitor_values)

        # Store typical output for all commands (XCP data will be handled separately, eg. violation status)
        if not args.violation:
            self.logger.store_output(args.gpu, "values", monitor_values)

        # Now handling the single gpu case only
        if multiple_devices:
            self.logger.store_multiple_device_output()
            return

        if (
            watching_output and not self.logger.destination == "stdout"
        ):  # End of single gpu add to watch_output
            self.logger.store_watch_output(multiple_device_enabled=False)

        if args.violation:
            # Print violation status for single gpu, which have different xcp information per 1 gpu
            self.logger.print_output(
                multiple_device_enabled=True,
                watching_output=watching_output,
                tabular=True,
                dual_csv_output=dual_csv_output,
            )
        else:
            # Print the output for single gpu, which currently does not have multiple xcp information
            self.logger.print_output(
                multiple_device_enabled=False,
                watching_output=watching_output,
                tabular=True,
                dual_csv_output=dual_csv_output,
            )

    def _write_via_logger_destination(self, text):
        """Write text to the logger's destination (stdout or output file).

        Used for content that does not fit the logger's structured output path
        but must still honor --file redirection.
        """
        if self.logger.destination == "stdout":
            print(text)
        else:
            with self.logger.destination.open("a", encoding="utf-8") as f:
                f.write(text + "\n")

    def _monitor_inject_sort_by_pid(self, args, watching_output):
        """Render PID-grouped process info for monitor --process --sort-by-pid.

        For JSON/CSV the rows are pushed into the logger's multiple_device_output
        so they flow through print_output() like every other monitor row. For
        human-readable output an ASCII table is stored in
        self._sort_by_pid_secondary_table; the caller is responsible for writing
        it via _write_via_logger_destination after the primary table.
        """
        handles = args.gpu if isinstance(args.gpu, list) else [args.gpu]
        try:
            pid_list = amdsmi_interface.amdsmi_get_gpu_process_list_by_pid(handles)
        except amdsmi_exception.AmdSmiLibraryException as e:
            logging.debug("Failed to get process list by pid | %s", e.get_error_info())
            return

        # Unit string constants — hoisted out of the loop (style cleanup)
        memory_usage_unit_default = "B"
        evicted_time_unit = "ms"
        sdma_usage_unit = "us"

        # Build one entry per PID+GPU combination, grouped so all GPUs for a PID
        # appear together.
        filtered = []
        for proc in pid_list:
            for gpu_entry in proc["gpus"]:
                memory_usage_unit = memory_usage_unit_default

                mem_usage = gpu_entry["mem"]
                mem_dict = {
                    "gtt_mem": gpu_entry["memory_usage"]["gtt_mem"],
                    "cpu_mem": gpu_entry["memory_usage"]["cpu_mem"],
                    "vram_mem": gpu_entry["memory_usage"]["vram_mem"],
                }

                if self.logger.is_human_readable_format():
                    mem_usage = self.helpers.convert_bytes_to_readable(mem_usage)
                    for k in mem_dict:
                        mem_dict[k] = self.helpers.convert_bytes_to_readable(mem_dict[k])
                    memory_usage_unit = ""

                mem_usage = self.helpers.unit_format(self.logger, mem_usage, memory_usage_unit)
                for k in mem_dict:
                    mem_dict[k] = self.helpers.unit_format(
                        self.logger, mem_dict[k], memory_usage_unit
                    )

                cu = gpu_entry["cu_occupancy"]
                if cu == "N/A":
                    cu_str = "N/A"
                else:
                    cu_str = self.helpers.unit_format(self.logger, cu, "")

                if self.logger.is_human_readable_format():
                    sdma = self.helpers.convert_time_to_readable(gpu_entry["sdma_usage"], "us")
                    evict = self.helpers.convert_time_to_readable(gpu_entry["evicted_time"], "ms")
                else:
                    sdma = self.helpers.unit_format(
                        self.logger, gpu_entry["sdma_usage"], sdma_usage_unit
                    )
                    evict = self.helpers.unit_format(
                        self.logger, gpu_entry["evicted_time"], evicted_time_unit
                    )

                info = {
                    "name": proc["name"],
                    "pid": proc["pid"],
                    "memory_usage": mem_dict,
                    "mem_usage": mem_usage,
                    "cu_occupancy": cu_str,
                    "sdma_usage": sdma,
                    "evicted_time": evict,
                }
                filtered.append({"gpu_index": gpu_entry["gpu_index"], "process_info": info})

        if not filtered:
            filtered.append({"gpu_index": "N/A", "process_info": "No running processes detected"})

        # JSON / CSV: push rows into the logger's multiple_device_output so they
        # flow through print_output() — never write a raw ASCII tail that would
        # corrupt structured output.
        if self.logger.is_json_format() or self.logger.is_csv_format():
            for item in filtered:
                info = item["process_info"]
                if isinstance(info, str):
                    row = {"gpu_index": item["gpu_index"], "message": info}
                else:
                    row = {
                        "gpu_index": item["gpu_index"],
                        "pid": info["pid"],
                        "name": info["name"],
                        "mem_usage": info.get("mem_usage", "N/A"),
                        "cu_occupancy": info.get("cu_occupancy", "N/A"),
                        "sdma_usage": info.get("sdma_usage", "N/A"),
                        "evicted_time": info.get("evicted_time", "N/A"),
                    }
                    mu = info.get("memory_usage", {})
                    if isinstance(mu, dict):
                        if self.logger.is_csv_format():
                            row.update(self.logger.flatten_dict(mu))
                        else:
                            row["memory_usage"] = mu
                self.logger.output = row
                self.logger.store_multiple_device_output()
            return

        # Human-readable: build the ASCII secondary table; the caller writes it
        # via _write_via_logger_destination after print_output().
        header = (
            "\nPROCESS INFO (grouped by PID):\n"
            + "GPU".rjust(3)
            + "NAME".rjust(19)
            + "PID".rjust(9)
            + "GTT_MEM".rjust(10)
            + "CPU_MEM".rjust(10)
            + "VRAM_MEM".rjust(10)
            + "MEM_USG".rjust(10)
            + "CU%".rjust(9)
            + "SDMA".rjust(8)
            + "EVICT".rjust(8)
        )
        rows = []
        for item in filtered:
            info = item["process_info"]
            if isinstance(info, str):
                rows.append("  " + info)
                continue
            name = str(info.get("name", "N/A")).split("/")[-1][:17]
            row = str(item["gpu_index"]).rjust(3)
            row += name.rjust(19)
            row += str(info["pid"]).rjust(9)
            mu = info.get("memory_usage", {})
            row += str(mu.get("gtt_mem", "N/A")).rjust(10)
            row += str(mu.get("cpu_mem", "N/A")).rjust(10)
            row += str(mu.get("vram_mem", "N/A")).rjust(10)
            row += str(info.get("mem_usage", "N/A")).rjust(10)
            row += str(info.get("cu_occupancy", "N/A")).rjust(9)
            row += str(info.get("sdma_usage", "N/A")).rjust(8)
            row += str(info.get("evicted_time", "N/A")).rjust(8)
            rows.append(row)

        self._sort_by_pid_secondary_table = header + "\n" + "\n".join(rows)
