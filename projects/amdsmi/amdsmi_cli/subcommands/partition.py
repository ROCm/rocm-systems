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
import os

from amdsmi import amdsmi_exception, amdsmi_interface


class PartitionCommands:
    def partition(
        self, args, multiple_devices=False, gpu=None, current=None, memory=None, accelerator=None
    ):
        """Display partition information for the target GPU
        param:
            args - argparser args to pass to subcommand
            multiple_devices (bool) - True if checking for multiple devices
            gpu (device_handle) - device_handle for target device
            current - boolean which dictates whether the current partition information is shown
            memory - boolean which dictates whether the memory partition information is shown
            accelerator - boolean which dictates whether the accelerator partition information is shown
        returns:
            nothing
        """

        if gpu:
            args.gpu = gpu
        if args.gpu == None:
            args.gpu = self.device_handles
        if not isinstance(args.gpu, list):
            args.gpu = [args.gpu]
        if current:
            args.current = current
        if memory:
            args.memory = memory
        if accelerator:
            args.accelerator = accelerator

        if not self.group_check_printed:
            self.helpers.check_required_groups()
            self.group_check_printed = True

        ###########################################
        # amd-smi partition (no args)             #
        ###########################################
        # if no args are present, then everything should be displayed
        if not args.current and not args.memory and not args.accelerator:
            args.current = True
            args.memory = True
            args.accelerator = True

        ###########################################
        # amd-smi partition --current             #
        ###########################################
        if args.current:
            self.logger.table_header = "".rjust(7)
            current_header = (
                "GPU_ID".ljust(8)
                + "MEMORY".ljust(8)
                + "ACCELERATOR_TYPE".ljust(18)
                + "ACCELERATOR_PROFILE_INDEX".ljust(27)
                + "PARTITION_ID".ljust(14)
            )
            self.logger.table_header = current_header + self.logger.table_header.strip()

            tabular_output = []
            for gpu in args.gpu:
                gpu_id = self.helpers.get_gpu_id_from_device_handle(gpu)
                try:
                    partition_dict = amdsmi_interface.amdsmi_get_gpu_accelerator_partition_profile(
                        gpu
                    )
                    partition_id = (
                        str(partition_dict["partition_id"])
                        .replace("[", "")
                        .replace("]", "")
                        .replace(" ", "")
                    )
                    profile_type = partition_dict["partition_profile"]["profile_type"]
                    profile_index = partition_dict["partition_profile"]["profile_index"]
                except amdsmi_exception.AmdSmiLibraryException as e:
                    profile_type = "N/A"
                    profile_index = "N/A"
                    partition_id = (
                        str(partition_dict["partition_id"])
                        .replace("[", "")
                        .replace("]", "")
                        .replace(" ", "")
                    )
                    logging.debug(
                        "Failed to get accelerator partition profile for GPU %s | %s",
                        gpu_id,
                        e.get_error_info(),
                    )
                try:
                    current_mem_cap = amdsmi_interface.amdsmi_get_gpu_memory_partition(gpu)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    current_mem_cap = "N/A"
                    logging.debug(
                        "Failed to get current memory partition capabilities for GPU %s | %s",
                        gpu_id,
                        e.get_error_info(),
                    )

                if profile_type == 0:
                    profile_type = "N/A"

                tabular_output_dict = {
                    "gpu_id": gpu_id,
                    "memory": current_mem_cap,
                    "accelerator_type": profile_type,
                    "accelerator_profile_index": profile_index,
                    "partition_id": partition_id,
                }
                tabular_output.append(tabular_output_dict)

            self.logger.multiple_device_output = tabular_output
            self.logger.table_title = "CURRENT_PARTITION"
            if self.logger.is_json_format():
                self.logger.store_current_partition_json_output.extend(tabular_output)
                if not (args.memory or args.accelerator):
                    self.logger.combine_arrays_to_json()
            else:
                self.logger.print_output(multiple_device_enabled=True, tabular=True, dynamic=True)
            self.logger.clear_multiple_devices_output()

        ###########################################
        # amd-smi partition --memory              #
        ###########################################
        if args.memory:
            tabular_output = []
            self.logger.table_header = "".rjust(7)
            current_header = (
                "GPU_ID".ljust(8)
                + "MEMORY_PARTITION_CAPS".ljust(23)
                + "CURRENT_MEMORY_PARTITION".ljust(26)
            )
            self.logger.table_header = current_header + self.logger.table_header.strip()

            for gpu in args.gpu:
                gpu_id = self.helpers.get_gpu_id_from_device_handle(gpu)
                mem_caps_str = "N/A"
                current_memory_partition = "N/A"
                try:
                    memory_partition_config = (
                        amdsmi_interface.amdsmi_get_gpu_memory_partition_config(gpu)
                    )
                    mem_caps_str = (
                        str(memory_partition_config["partition_caps"])
                        .replace("]", "")
                        .replace("[", "")
                        .replace("'", "")
                        .replace(" ", "")
                    )
                    current_memory_partition = memory_partition_config["mp_mode"]
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug(
                        "Failed to get current memory partition for GPU %s | %s",
                        gpu_id,
                        e.get_error_info(),
                    )

                tabular_output_dict = {
                    "gpu_id": gpu_id,
                    "memory_partition_caps": mem_caps_str,
                    "current_memory_partition": current_memory_partition,
                }
                tabular_output.append(tabular_output_dict)

            self.logger.multiple_device_output = tabular_output
            self.logger.table_title = "\nMEMORY_PARTITION"
            if self.logger.is_json_format():
                self.logger.store_memory_partition_json_output.extend(tabular_output)
                if not args.accelerator:
                    self.logger.combine_arrays_to_json()
            else:
                self.logger.print_output(multiple_device_enabled=True, tabular=True, dynamic=True)
            self.logger.clear_multiple_devices_output()

        ###########################################
        # amd-smi partition --accelerator         #
        ###########################################
        if args.accelerator:
            self.logger.table_header = "".rjust(7)
            current_header = (
                "GPU_ID".ljust(8)
                + "PROFILE_INDEX".ljust(15)
                + "MEMORY_PARTITION_CAPS".ljust(23)
                + "ACCELERATOR_TYPE".ljust(18)
                + "PARTITION_ID".ljust(17)
                + "NUM_PARTITIONS".ljust(16)
                + "NUM_RESOURCES".ljust(15)
                + "RESOURCE_INDEX".ljust(16)
                + "RESOURCE_TYPE".ljust(15)
                + "RESOURCE_INSTANCES".ljust(20)
                + "RESOURCES_SHARED".ljust(18)
            )
            self.logger.table_header = current_header + self.logger.table_header.strip()

            tabular_output = []
            prev_gpu_id = "N/A"
            for gpu in args.gpu:
                gpu_id = self.helpers.get_gpu_id_from_device_handle(gpu)
                tabular_output_dict = {
                    "gpu_id": gpu_id,
                    "profile_index": "N/A",
                    "memory_partition_caps": "N/A",
                    "accelerator_type": "N/A",
                    "partition_id": "0",
                    "num_partitions": "N/A",
                    "num_resources": "N/A",
                    "resource_index": "N/A",
                    "resource_type": "N/A",
                    "resource_instances": "N/A",
                    "resources_shared": "N/A",
                }
                try:
                    partition_dict = amdsmi_interface.amdsmi_get_gpu_accelerator_partition_profile(
                        gpu
                    )
                    partition_id = (
                        str(partition_dict["partition_id"])
                        .replace("[", "")
                        .replace("]", "")
                        .replace(" ", "")
                    )
                    current_accelerator_type = partition_dict["partition_profile"]["profile_type"]
                    tabular_output_dict["partition_id"] = partition_id

                    # save only the primary GPU node's partition_id (the 1st listed device; non N/A one)
                    # else keep current_partition_id unchanged for displaying in accelerator resource's output
                    if partition_id != "N/A":
                        current_partition_id = partition_id

                except amdsmi_exception.AmdSmiLibraryException as e:
                    profile_type = "N/A"
                    profile_index = "N/A"
                    partition_id = "0"
                    mem_caps_str = "N/A"
                    num_partitions = 0
                    current_accelerator_type = "N/A"
                    logging.debug(
                        "Failed to get accelerator partition profile for GPU %s | %s",
                        gpu_id,
                        e.get_error_info(),
                    )

                try:
                    partition_config_dict = (
                        amdsmi_interface.amdsmi_get_gpu_accelerator_partition_profile_config(gpu)
                    )
                    logging.debug(
                        "amdsmi_commands.py | partition_config_dict: "
                        + str(json.dumps(partition_config_dict, indent=4))
                    )
                    num_profiles = partition_config_dict["num_profiles"]
                    num_resource_profiles = partition_config_dict["num_resource_profiles"]

                    resource_index = 0
                    prev_accelerator_type = "N/A"
                    for p in range(0, num_profiles):
                        accelerator_type = partition_config_dict["profiles"][p]["profile_type"]
                        profile_index = partition_config_dict["profiles"][p]["profile_index"]
                        num_partitions = partition_config_dict["profiles"][p]["num_partitions"]
                        mem_caps_str = (
                            str(partition_config_dict["profiles"][p]["memory_caps"])
                            .replace("]", "")
                            .replace("[", "")
                            .replace("'", "")
                            .replace(" ", "")
                        )
                        # 2 modifications based on the current accelerator type:
                        # 1) display a * for the current accelerator type, otherwise display as normal
                        # 2) display partition id only for the current accelerator profile (the *'d one)
                        if current_accelerator_type == accelerator_type:
                            accelerator_type = accelerator_type + "*"
                            partition_id = current_partition_id
                        else:
                            partition_id = "N/A"
                        # only display the first instance of the gpu_id, rest are empty strings
                        if prev_gpu_id != gpu_id:
                            tabular_gpu_id = gpu_id
                            prev_gpu_id = gpu_id
                        else:
                            tabular_gpu_id = ""
                        logging.debug("amdsmi_commands.py | tabular_gpu_id: " + str(tabular_gpu_id))

                        if num_resource_profiles == 0:
                            if (
                                prev_accelerator_type != accelerator_type
                            ):  # only print the first instance of the resources
                                tabular_output_dict = {
                                    "gpu_id": tabular_gpu_id,
                                    "profile_index": profile_index,
                                    "memory_partition_caps": mem_caps_str,
                                    "accelerator_type": accelerator_type,
                                    "partition_id": partition_id,
                                    "num_partitions": num_partitions,
                                    "num_resources": num_resource_profiles,
                                    "resource_index": "N/A",
                                    "resource_type": "N/A",
                                    "resource_instances": "N/A",
                                    "resources_shared": "N/A",
                                }
                                prev_accelerator_type = accelerator_type
                                tabular_output.append(tabular_output_dict)
                            continue

                        for r in range(0, num_resource_profiles):
                            logging.debug(
                                "amdsmi_commands.py | p: "
                                + str(p)
                                + "; r: "
                                + str(r)
                                + "; accelerator_type: "
                                + str(accelerator_type)
                            )
                            resource_type = partition_config_dict["profiles"][p]["resources"][r][
                                "resource_type"
                            ]
                            resource_instances = partition_config_dict["profiles"][p]["resources"][
                                r
                            ]["partition_resource"]
                            resources_shared = partition_config_dict["profiles"][p]["resources"][r][
                                "num_partitions_share_resource"
                            ]
                            if (
                                prev_accelerator_type != accelerator_type
                            ):  # only print the first instance of the resources
                                tabular_output_dict = {
                                    "gpu_id": tabular_gpu_id,
                                    "profile_index": profile_index,
                                    "memory_partition_caps": mem_caps_str,
                                    "accelerator_type": accelerator_type,
                                    "partition_id": partition_id,
                                    "num_partitions": num_partitions,
                                    "num_resources": num_resource_profiles,
                                    "resource_index": resource_index,
                                    "resource_type": resource_type,
                                    "resource_instances": resource_instances,
                                    "resources_shared": resources_shared,
                                }
                                prev_accelerator_type = accelerator_type
                            else:
                                tabular_output_dict = {
                                    "gpu_id": "",
                                    "profile_index": "",
                                    "memory_partition_caps": "",
                                    "accelerator_type": "",
                                    "partition_id": "",
                                    "num_partitions": "",
                                    "num_resources": "",
                                    "resource_index": resource_index,
                                    "resource_type": resource_type,
                                    "resource_instances": resource_instances,
                                    "resources_shared": resources_shared,
                                }
                            resource_index += 1
                            tabular_output.append(tabular_output_dict)
                except amdsmi_exception.AmdSmiLibraryException:
                    tabular_output.append(tabular_output_dict)

            self.logger.multiple_device_output = tabular_output
            self.logger.table_title = "\nACCELERATOR_PARTITION_PROFILES"
            # only display warning message if not running as root or with sudo
            if os.geteuid() != 0:
                self.logger.warning_message = """
***************************************************************************
** WARNING:                                                              **
** ACCELERATOR_PARTITION_PROFILES requires sudo/root permissions to run. **
** Please run the command with sudo permissions to get accurate results. **
***************************************************************************
"""
            if self.logger.is_json_format():
                self.logger.store_partition_profiles_json_output.extend(tabular_output)
            else:
                self.logger.print_output(multiple_device_enabled=True, tabular=True, dynamic=True)
            self.logger.clear_multiple_devices_output()
            self.logger.warning_message = ""  # clear the warning message

            #########################################
            # print accelerator partition resources #
            #########################################
            self.logger.table_header = "".rjust(7)
            current_header = (
                "RESOURCE_INDEX".ljust(16)
                + "RESOURCE_TYPE".ljust(15)
                + "RESOURCE_INSTANCES".ljust(20)
                + "RESOURCES_SHARED".ljust(18)
            )
            self.logger.table_header = current_header + self.logger.table_header.strip()

            tabular_output = []
            for gpu in args.gpu:
                gpu_id = self.helpers.get_gpu_id_from_device_handle(gpu)
                tabular_output_dict = {
                    "resource_index": "N/A",
                    "resource_type": "N/A",
                    "resource_instances": "N/A",
                    "resources_shared": "N/A",
                }
                try:
                    partition_config_dict = (
                        amdsmi_interface.amdsmi_get_gpu_accelerator_partition_profile_config(gpu)
                    )
                    logging.debug(
                        "amdsmi_commands.py | partition_config_dict: "
                        + str(json.dumps(partition_config_dict, indent=4))
                    )
                    num_profiles = partition_config_dict["num_profiles"]
                    num_resource_profiles = partition_config_dict["num_resource_profiles"]

                    if num_resource_profiles == 0:
                        tabular_output.append(tabular_output_dict)
                        continue

                    resource_index = 0
                    for p in range(0, num_profiles):
                        for r in range(0, num_resource_profiles):
                            resource_type = partition_config_dict["profiles"][p]["resources"][r][
                                "resource_type"
                            ]
                            resource_instances = partition_config_dict["profiles"][p]["resources"][
                                r
                            ]["partition_resource"]
                            resources_shared = partition_config_dict["profiles"][p]["resources"][r][
                                "num_partitions_share_resource"
                            ]
                            tabular_output_dict = {
                                "resource_index": resource_index,
                                "resource_type": resource_type,
                                "resource_instances": resource_instances,
                                "resources_shared": resources_shared,
                            }
                            resource_index += 1
                            tabular_output.append(tabular_output_dict)
                except amdsmi_exception.AmdSmiLibraryException:
                    tabular_output.append(tabular_output_dict)

            self.logger.multiple_device_output = tabular_output
            self.logger.table_title = "\nACCELERATOR_PARTITION_RESOURCES"
            if self.logger.is_json_format():
                self.logger.store_partition_resources_json_output.extend(tabular_output)
            else:
                self.logger.print_output(multiple_device_enabled=True, tabular=True, dynamic=True)
            if self.logger.is_json_format():
                self.logger.combine_arrays_to_json()
            self.logger.clear_multiple_devices_output()

            if self.logger.is_human_readable_format():
                # print legend
                legend_parts = ["\n\nLegend:", "  * = Current mode"]
                legend_output = "\n".join(legend_parts)
                if self.logger.destination == "stdout":
                    print(legend_output)
                else:
                    with self.logger.destination.open("a", encoding="utf-8") as output_file:
                        output_file.write(legend_output + "\n")
