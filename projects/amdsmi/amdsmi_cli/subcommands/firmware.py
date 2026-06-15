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

from amdsmi import amdsmi_exception, amdsmi_interface


class FirmwareCommands:
    def firmware_nic(self, args, multiple_devices=False, nic=None, fw_list=True):
        """Get Firmware information for target nic

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            nic (device_handle, optional): device_handle for target device. Defaults to None.
            fw_list (bool, optional): True to get list of all firmware information
        Raises:
            IndexError: Index error if nic list is empty

        Returns:
            None: Print output via AMDSMILogger to destination
        """
        if fw_list:
            args.fw_list = fw_list
        if nic:
            args.nic = nic

        # Handle No NIC passed
        if args.nic == None:
            args.nic = self.device_handles_brcm_nics

        # Handle multiple NICs

        if args.nic != None:
            handled_multiple_nics, device_handle = self.helpers.handle_brcm_nics(
                args, self.logger, self.firmware_nic
            )
            if handled_multiple_nics:
                return  # This function is recursive

        args.nic = device_handle
        nic_id = self.helpers.get_nic_id_from_device_handle(args.nic)
        if args.fw_list:
            try:
                fw_info = amdsmi_interface.amdsmi_get_nic_fw_info(args.nic)
            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug(
                    "Failed to get firmware info for nic %s | %s", nic_id, e.get_error_info()
                )

        self.logger.store_nic_output(args.nic, "values", fw_info)

        if multiple_devices:
            self.logger.store_multiple_device_output()
            return  # Skip printing when there are multiple devices

        self.logger.print_output()

    def firmware(
        self, args, multiple_devices=False, gpu=None, nic=None, fw_list=True, brcm_nic=None
    ):
        """Get Firmware information for target gpu

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            gpu (device_handle, optional): device_handle for target device. Defaults to None.
            fw_list (bool, optional): True to get list of all firmware information
            brcm_nic (bool, optional): Value override for args.brcm_nic. Defaults to None.
        Raises:
            IndexError: Index error if gpu list is empty

        Returns:
            None: Print output via AMDSMILogger to destination
        """
        if gpu:
            args.gpu = gpu
        if fw_list:
            args.fw_list = fw_list

        # Handle No GPU passed
        if args.gpu == None:
            args.gpu = self.device_handles

        if self.helpers.is_brcm_nic_initialized() and (
            getattr(args, "brcm_nic", False) or brcm_nic
        ):
            self.logger.output = {}
            self.logger.clear_multiple_devices_output()
            self.firmware_nic(args, multiple_devices, nic, fw_list)
            return
        # Handle multiple GPUs
        handled_multiple_gpus, device_handle = self.helpers.handle_gpus(
            args, self.logger, self.firmware
        )
        if handled_multiple_gpus:
            return  # This function is recursive

        args.gpu = device_handle

        fw_list = {}

        # Get gpu_id for logging
        gpu_id = self.helpers.get_gpu_id_from_device_handle(args.gpu)

        if args.fw_list:
            try:
                fw_info = amdsmi_interface.amdsmi_get_fw_info(args.gpu)

                for fw_index, fw_entry in enumerate(fw_info["fw_list"]):
                    # Change fw_name to fw_id
                    fw_entry["fw_id"] = fw_entry.pop("fw_name").name.replace("AMDSMI_FW_ID_", "")
                    fw_entry["fw_version"] = fw_entry.pop("fw_version")  # popping to ensure order

                    # Add custom human readable formatting
                    if self.logger.is_human_readable_format():
                        fw_info["fw_list"][fw_index] = {f"FW {fw_index}": fw_entry}
                    else:
                        fw_info["fw_list"][fw_index] = fw_entry

                fw_list.update(fw_info)
            except amdsmi_exception.AmdSmiLibraryException as e:
                fw_list["fw_list"] = "N/A"
                logging.debug(
                    "Failed to get firmware info for gpu %s | %s", gpu_id, e.get_error_info()
                )

        multiple_devices_csv_override = False
        # Convert and store output by pid for csv format
        if self.logger.is_csv_format():
            fw_key = "fw_list"
            for fw_info_dict in fw_list[fw_key]:
                for key, value in fw_info_dict.items():
                    multiple_devices_csv_override = True
                    self.logger.store_output(args.gpu, key, value)
                self.logger.store_multiple_device_output()
        else:
            # Store values in logger.output
            self.logger.store_output(args.gpu, "values", fw_list)

        if multiple_devices:
            self.logger.store_multiple_device_output()
            return  # Skip printing when there are multiple devices

        self.logger.print_output(multiple_device_enabled=multiple_devices_csv_override)
