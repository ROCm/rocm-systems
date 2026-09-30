#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import concurrent.futures
import logging

from amdsmi import amdsmi_exception, amdsmi_interface


class FirmwareCommands:
    def _get_nic_fw_info(self, device_handle):
        """Query firmware info for one NIC; {} on failure (logged)."""
        try:
            return amdsmi_interface.amdsmi_get_nic_fw_info(device_handle)
        except amdsmi_exception.AmdSmiLibraryException as e:
            nic_id = self.helpers.get_ainic_id_from_device_handle(device_handle)
            logging.debug("Failed to get firmware info for nic %s | %s", nic_id, e.get_error_info())
            return {}

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
            args.nic = self.device_handles_ainics

        # Handle multiple NICs

        if args.nic != None:
            # Each firmware query is a live devlink round trip (~1s); fetch them
            # concurrently instead of once per recursive handle_ainics call below.
            if isinstance(args.nic, list) and len(args.nic) > 1 and args.fw_list:
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.nic)) as pool:
                    fw_infos = pool.map(self._get_nic_fw_info, args.nic)
                args._nic_fw_info_prefetch = dict(zip((h.value for h in args.nic), fw_infos))

            handled_multiple_nics, device_handle = self.helpers.handle_ainics(
                args, self.logger, self.firmware_nic
            )
            if handled_multiple_nics:
                return  # This function is recursive

        args.nic = device_handle
        fw_info = {}
        if args.fw_list:
            prefetch = getattr(args, "_nic_fw_info_prefetch", None)
            if prefetch is not None and args.nic.value in prefetch:
                fw_info = prefetch[args.nic.value]
            else:
                fw_info = self._get_nic_fw_info(args.nic)

        self.logger.store_ainic_output(args.nic, "values", fw_info)

        if multiple_devices:
            self.logger.store_multiple_device_output()
            return  # Skip printing when there are multiple devices

        self.logger.print_output()

    def firmware(self, args, multiple_devices=False, gpu=None, nic=None, fw_list=True):
        """Get Firmware information for target gpu

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            gpu (device_handle, optional): device_handle for target device. Defaults to None.
            fw_list (bool, optional): True to get list of all firmware information
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

        if self.helpers.is_ainic_initialized() and (nic or getattr(args, "nic", None)):
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
