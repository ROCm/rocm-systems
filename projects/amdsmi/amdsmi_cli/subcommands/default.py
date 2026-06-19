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

import logging
import os

from _version import __version__
from amdsmi_helpers import AMDSMIHelpers

from amdsmi import amdsmi_exception, amdsmi_interface


class DefaultCommands:
    def default(self, args):
        """Display the default amdsmi view when no args are given."""

        # check groups first
        if not self.group_check_printed:
            self.helpers.check_required_groups()
            self.group_check_printed = True

        version_info = {
            "amd-smi": "N/A",
            "amdgpu version": "N/A",
            "kernel version": "N/A",
            "fw pldm version": "N/A",
            "vbios version": "N/A",
            "rocm version": (False, "N/A"),
        }
        version_info["rocm version"] = amdsmi_interface.amdsmi_get_rocm_version()
        version_info["kernel version"] = os.uname().release

        if not self.helpers.is_amdgpu_initialized():
            # everything below requires amdgpu to be initialized, so if it's not, skip the rest of the default info and just return what we have so far
            return

        processors = amdsmi_interface.amdsmi_get_processor_handles()
        try:
            version_info["amdgpu version"] = amdsmi_interface.amdsmi_get_gpu_driver_info(
                processors[0]
            )
        except amdsmi_exception.AmdSmiLibraryException as e:
            version_info["amdgpu version"] = "N/A"
            logging.debug("Failed to get driver info for gpu: %s", e.get_error_info())
        try:
            fw_info = amdsmi_interface.amdsmi_get_fw_info(processors[0])
            for fw in fw_info["fw_list"]:
                if fw.get("fw_name") == amdsmi_interface.AmdSmiFwBlock.AMDSMI_FW_ID_PLDM_BUNDLE:
                    version_info["fw pldm version"] = fw["fw_version"]
                    # we only need to find one of them
                    break
        except amdsmi_exception.AmdSmiLibraryException as e:
            version_info["fw pldm version"] = "N/A"
            logging.debug("Failed to get fw pldm info for gpu: %s", e.get_error_info())
        try:
            version_info["vbios version"] = amdsmi_interface.amdsmi_get_gpu_vbios_info(
                processors[0]
            )["version"]
            if version_info["vbios version"] == "":
                version_info["vbios version"] = "N/A"
        except amdsmi_exception.AmdSmiLibraryException as e:
            version_info["vbios version"] = "N/A"
            logging.debug("Failed to get vbios info for gpu: %s", e.get_error_info())

        version_info["amd-smi"] = f"{__version__}"

        default_table_info_dict = {}
        default_table_info_dict.update({"version_info": version_info})

        gpu_info_list = []
        all_process_list = []

        # get info for each processor to display in default output
        for processor in processors:
            gpu_info_dict = {}

            gpu_id = self.helpers.get_gpu_id_from_device_handle(processor)
            gpu_info_dict.update({"gpu_id": gpu_id})
            # get common gpu_metrics first
            try:
                gpu_metrics = amdsmi_interface.amdsmi_get_gpu_metrics_info(processor)
            except amdsmi_exception.AmdSmiLibraryException:
                gpu_metrics = amdsmi_interface._NA_amdsmi_get_gpu_metrics_info()

            # partition info
            try:
                current_mem = amdsmi_interface.amdsmi_get_gpu_memory_partition(processor)
            except amdsmi_exception.AmdSmiLibraryException:
                current_mem = "N/A"
            try:
                current_comp = amdsmi_interface.amdsmi_get_gpu_compute_partition(processor)
            except amdsmi_exception.AmdSmiLibraryException:
                current_comp = "N/A"
            if current_comp == "N/A" or current_mem == "N/A":
                partition_mode = "N/A"
            else:
                partition_mode = f"{current_comp}/{current_mem}"
            gpu_info_dict.update({"partition_mode": partition_mode})

            # GPU name market name and OAM ID
            try:
                asic_info = amdsmi_interface.amdsmi_get_gpu_asic_info(processor)
                market_name = asic_info["market_name"]
                oam_id = asic_info["oam_id"]
                # get num_cu now for use later
                total_num_cu = float(asic_info["num_compute_units"])
            except amdsmi_exception.AmdSmiLibraryException:
                market_name = "N/A"
                oam_id = "N/A"
                total_num_cu = "N/A"
            gpu_info_dict.update({"market_name": market_name})
            gpu_info_dict.update({"oam_id": oam_id})

            # bdf
            try:
                bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(processor)
                # if the len of the bdf is not 12, then invalid values are being populated.
                if len(bdf) != 12:
                    bdf = "N/A"
            except amdsmi_exception.AmdSmiLibraryException:
                bdf = "N/A"
            gpu_info_dict.update({"bdf": bdf})

            # HIP ID
            try:
                enum_info = amdsmi_interface.amdsmi_get_gpu_enumeration_info(processor)
                hip_id = enum_info["hip_id"]
            except amdsmi_exception.AmdSmiLibraryException:
                hip_id = "N/A"
            gpu_info_dict.update({"hip_id": hip_id})

            # mem utilization, GPU utilization, power usage, and temperature from gpu_metrics
            if gpu_metrics != "N/A":
                mem_util = gpu_metrics["average_umc_activity"]
                gfx_util = gpu_metrics["average_gfx_activity"]
                if gpu_metrics["current_socket_power"] != "N/A":
                    current_power = gpu_metrics["current_socket_power"]
                else:
                    current_power = gpu_metrics["average_socket_power"]
                # If the hotspot temperature is not available use the edge temp (applicable to APUs)
                if gpu_metrics["temperature_hotspot"] != "N/A":
                    temperature = gpu_metrics["temperature_hotspot"]
                elif gpu_metrics["temperature_edge"] != "N/A":
                    temperature = gpu_metrics["temperature_edge"]
                else:
                    temperature = "N/A"
            else:
                mem_util = "N/A"
                gfx_util = "N/A"
                current_power = "N/A"
                temperature = "N/A"
            gpu_info_dict.update({"mem_util": mem_util})
            gpu_info_dict.update({"gfx_util": gfx_util})
            gpu_info_dict.update({"temp": temperature})

            # rest of power usage info; Will assume we're always trying to get PPT0 for now
            try:
                power_cap_info = amdsmi_interface.amdsmi_get_power_cap_info(processor, 0)
                socket_power_limit = power_cap_info["power_cap"]
                socket_power_limit = self.helpers.convert_SI_unit(
                    socket_power_limit, AMDSMIHelpers.SI_Unit.MICRO
                )
                power_usage = {"current_power": current_power, "power_limit": socket_power_limit}
            except amdsmi_exception.AmdSmiLibraryException:
                power_usage = "N/A"
            gpu_info_dict.update({"power_usage": power_usage})

            # memory usage - Use APU-aware memory selection
            try:
                # Use helper method to determine appropriate memory type
                mem_type, mem_type_name = self.helpers.get_apu_memory_type_and_name(
                    processor, gpu_id
                )

                # Get memory usage and total using the determined memory type
                used_mem = amdsmi_interface.amdsmi_get_gpu_memory_usage(processor, mem_type) // (
                    1024 * 1024
                )
                total_mem = amdsmi_interface.amdsmi_get_gpu_memory_total(processor, mem_type) // (
                    1024 * 1024
                )

                # Create appropriate dictionary keys based on memory type
                if mem_type_name == "GTT":
                    mem_usage = {"used_gtt": used_mem, "total_gtt": total_mem}
                else:
                    mem_usage = {"used_vram": used_mem, "total_vram": total_mem}
            except amdsmi_exception.AmdSmiLibraryException:
                mem_usage = "N/A"
            gpu_info_dict.update({"mem_usage": mem_usage})

            # uncorrectable ECC errors
            try:
                ecc_count = amdsmi_interface.amdsmi_get_gpu_total_ecc_count(processor)
                uncorrectable = ecc_count.pop("uncorrectable_count")
            except amdsmi_exception.AmdSmiLibraryException:
                uncorrectable = "N/A"
            gpu_info_dict.update({"uncorr_ecc": uncorrectable})

            # Fan usage
            try:
                fan_speed = amdsmi_interface.amdsmi_get_gpu_fan_speed(processor, 0)
            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug(
                    "Failed to get fan speed for gpu %s | %s", processor, e.get_error_info()
                )
                fan_speed = "N/A"
            try:
                fan_max = amdsmi_interface.amdsmi_get_gpu_fan_speed_max(processor, 0)
                fan_usage = "N/A"
                if fan_max > 0 and fan_speed != "N/A":
                    fan_usage = round((float(fan_speed) / float(fan_max)) * 100, 2)
            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug(
                    "Failed to get max fan speed for gpu %s | %s", processor, e.get_error_info()
                )
                fan_usage = "N/A"
            gpu_info_dict.update({"fan": fan_usage})

            gpu_info_list.append(gpu_info_dict)

            # Running Processes
            try:
                raw_process_list = amdsmi_interface.amdsmi_get_gpu_process_list(processor)
                for proc in raw_process_list:
                    proc_info_dict = {
                        "gpu": "N/A",
                        "pid": "N/A",
                        "name": "N/A",
                        "gtt": "N/A",
                        "vram": "N/A",
                        "mem_usage": "N/A",
                        "cu_occupancy": "N/A",
                        "sdma_usage": "N/A",
                    }
                    proc_info_dict["gpu"] = gpu_id
                    proc_info_dict["pid"] = proc["pid"]
                    proc_info_dict["name"] = proc["name"]
                    proc_info_dict["gtt"] = self.helpers.convert_bytes_to_readable(
                        proc["memory_usage"]["gtt_mem"]
                    )
                    proc_info_dict["vram"] = self.helpers.convert_bytes_to_readable(
                        proc["memory_usage"]["vram_mem"]
                    )
                    proc_info_dict["sdma_usage"] = self.helpers.unit_format(
                        self.logger, proc["sdma_usage"], "us"
                    )
                    proc_info_dict["mem_usage"] = self.helpers.convert_bytes_to_readable(
                        proc["mem"]
                    )
                    # Handle cu_occupancy conversion safely
                    try:
                        if proc["cu_occupancy"] != "N/A" and total_num_cu != "N/A":
                            num_cu = float(proc["cu_occupancy"])
                            proc_info_dict["cu_occupancy"] = {
                                "current_cu": num_cu,
                                "total_num_cu": total_num_cu,
                            }
                        else:
                            proc_info_dict["cu_occupancy"] = {
                                "current_cu": "N/A",
                                "total_num_cu": total_num_cu,
                            }
                    except (ValueError, TypeError):
                        proc_info_dict["cu_occupancy"] = {
                            "current_cu": "N/A",
                            "total_num_cu": total_num_cu,
                        }

                    all_process_list.append(proc_info_dict)
            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug(
                    "Failed to get process list for gpu %s | %s", gpu_id, e.get_error_info()
                )

        default_table_info_dict.update({"gpu_info_list": gpu_info_list})
        default_table_info_dict.update({"processes": all_process_list})

        if self.logger.is_json_format():
            self.logger.output = default_table_info_dict
            self.logger.print_output()
        elif self.logger.is_csv_format():
            self.logger.multiple_device_output = default_table_info_dict
            self.logger.print_output(multiple_device_enabled=True, tabular=True, dynamic=True)
        else:
            self.logger.print_default_output(default_table_info_dict)
