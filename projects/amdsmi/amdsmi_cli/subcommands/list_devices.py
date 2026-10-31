#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging

from amdsmi_cli_exceptions import AmdSmiCommandNotSupportedException

from amdsmi import amdsmi_exception, amdsmi_interface


class ListDevicesCommands:
    def list_gpu(self, args, multiple_devices=False, gpu=None):
        """List information for target gpu

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            gpu (device_handle, optional): device_handle for target device. Defaults to None.

        Raises:
            IndexError: Index error if gpu list is empty

        Returns:
            None: Print output via AMDSMILogger to destination
        """
        # Set args.* to passed in arguments
        if gpu:
            args.gpu = gpu

        # Handle No GPU passed
        if args.gpu == None:
            args.gpu = self.device_handles

        if not self.group_check_printed:
            self.helpers.check_required_groups()
            self.group_check_printed = True

        # Handle multiple GPUs
        handled_multiple_gpus, device_handle = self.helpers.handle_gpus(
            args, self.logger, self.list_gpu
        )
        if handled_multiple_gpus:
            return  # This function is recursive

        args.gpu = device_handle

        # Get gpu_id for logging
        gpu_id = self.helpers.get_gpu_id_from_device_handle(args.gpu)

        # Always try to get BDF regardless of group check
        try:
            bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(args.gpu)
        except amdsmi_exception.AmdSmiLibraryException:
            bdf = "N/A"

        cuid_info = self.helpers.get_gpu_cuid_info(args.gpu)
        uuid = cuid_info["derived_cuid"]
        if cuid_info["identifier_kind"] != "cuid":
            uuid = self.helpers._query_or_na(amdsmi_interface.amdsmi_get_gpu_device_uuid, args.gpu)
            if uuid and uuid != "N/A":
                cuid_info["identifier_kind"] = "legacy_uuid"

        try:
            kfd_info = amdsmi_interface.amdsmi_get_gpu_kfd_info(args.gpu)
            kfd_id = kfd_info["kfd_id"]
            node_id = kfd_info["node_id"]
            partition_id = kfd_info["current_partition_id"]
        except amdsmi_exception.AmdSmiLibraryException as e:
            kfd_id = node_id = partition_id = "N/A"
            logging.debug("Failed to get kfd info for gpu %s | %s", gpu_id, e.get_error_info())

        # CSV format is intentionally aligned with Host
        if self.logger.is_csv_format():
            self.logger.store_output(args.gpu, "gpu_bdf", bdf)
            self.logger.store_output(args.gpu, "gpu_uuid", uuid)
        else:
            self.logger.store_output(args.gpu, "bdf", bdf)
            self.logger.store_output(args.gpu, "uuid", uuid)

        self.logger.store_output(args.gpu, "kfd_id", kfd_id)
        self.logger.store_output(args.gpu, "node_id", node_id)
        self.logger.store_output(args.gpu, "partition_id", partition_id)

        if args.enumeration:
            try:
                enumeration_info = amdsmi_interface.amdsmi_get_gpu_enumeration_info(args.gpu)
            except amdsmi_exception.AmdSmiLibraryException:
                enumeration_info = {
                    "drm_render": "N/A",
                    "drm_card": "N/A",
                    "hsa_id": "N/A",
                    "hip_id": "N/A",
                    "hip_uuid": "N/A",
                    "oam_id": "N/A",
                    "physical_acc_id": "N/A",
                }

            # now store all the fields exactly once:
            if enumeration_info["drm_render"] == "N/A":
                self.logger.store_output(args.gpu, "render", enumeration_info["drm_render"])
            else:
                self.logger.store_output(
                    args.gpu, "render", f"renderD{enumeration_info['drm_render']}"
                )
            if enumeration_info["drm_card"] == "N/A":
                self.logger.store_output(args.gpu, "card", enumeration_info["drm_card"])
            else:
                self.logger.store_output(args.gpu, "card", f"card{enumeration_info['drm_card']}")
            self.logger.store_output(args.gpu, "hsa_id", enumeration_info["hsa_id"])
            self.logger.store_output(args.gpu, "hip_id", enumeration_info["hip_id"])
            self.logger.store_output(args.gpu, "hip_uuid", enumeration_info["hip_uuid"])
            self.logger.store_output(args.gpu, "oam_id", enumeration_info["oam_id"])
            self.logger.store_output(
                args.gpu, "physical_acc_id", enumeration_info["physical_acc_id"]
            )

        self.logger.store_output(args.gpu, "cuid", cuid_info["derived_cuid"])
        for field in ("identifier_kind", "source", "auxiliary", "cuid_metadata_status"):
            self.logger.store_output(args.gpu, field, cuid_info[field])

        if multiple_devices:
            self.logger.store_multiple_device_output()
            return  # Skip printing when there are multiple devices

        self.logger.print_output()

    def list_brcm_nic(self, args, multiple_devices=False, nic=None):
        """List information for target nic

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            nic (device_handle, optional): device_handle for target device. Defaults to None.

        Raises:
            IndexError: Index error if nic list is empty

        Returns:
            None: Print output via AMDSMILogger to destination
        """
        # Set args.* to passed in arguments
        if nic:
            args.nic = nic

        if not self.group_check_printed:
            self.helpers.check_required_groups()
            self.group_check_printed = True

        # Handle multiple NICs
        handled_multiple_nics, device_handle = self.helpers.handle_brcm_nics(
            args, self.logger, self.list_brcm_nic
        )
        if handled_multiple_nics:
            return  # This function is recursive

        args.nic = device_handle

        # Get nic_id for logging
        nic_id = self.helpers.get_nic_id_from_device_handle(args.nic)

        # Get nic info for logging
        try:
            nic_info = amdsmi_interface.amdsmi_get_nic_info(args.nic)
            if nic_info:
                bdf = nic_info["bdf"]
                uuid = nic_info["UUID"]
                device_name = nic_info["Device Name"]
                part_number = nic_info["Part Number"]
                firmware_version = nic_info["Firmware_Version"]
            else:
                bdf = uuid = device_name = part_number = firmware_version = "N/A"

        except amdsmi_exception.AmdSmiLibraryException as e:
            bdf = uuid = device_name = part_number = firmware_version = "N/A"
            logging.debug("Failed to get info for nic %s | %s", nic_id, e.get_error_info())

        # CSV format is intentionally aligned with Host
        if self.logger.is_csv_format():
            self.logger.store_nic_output(args.nic, "nic_bdf", bdf)
            self.logger.store_nic_output(args.nic, "permanent_address", uuid)
            self.logger.store_nic_output(args.nic, "device_name", device_name)
            self.logger.store_nic_output(args.nic, "part_number", part_number)
            self.logger.store_nic_output(args.nic, "firmware_version", firmware_version)
        else:
            self.logger.store_nic_output(args.nic, "bdf", bdf)
            self.logger.store_nic_output(args.nic, "permanent_address", uuid)
            self.logger.store_nic_output(args.nic, "device_name", device_name)
            self.logger.store_nic_output(args.nic, "part_number", part_number)
            self.logger.store_nic_output(args.nic, "firmware_version", firmware_version)

        if multiple_devices:
            self.logger.store_multiple_device_output()
            return  # Skip printing when there are multiple devices

        self.logger.print_output()

    def list_ainic(self, args, multiple_devices=False, nic=None):
        """List information for target ainic

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            nic (device_handle, optional): device_handle for target device. Defaults to None.

        Raises:
            IndexError: Index error if nic list is empty

        Returns:
            None: Print output via AMDSMILogger to destination
        """
        # Set args.* to passed in arguments
        if nic:
            args.nic = nic

        if not self.group_check_printed:
            self.helpers.check_required_groups()
            self.group_check_printed = True

        # Handle multiple NICs
        handled_multiple_nics, device_handle = self.helpers.handle_ainics(
            args, self.logger, self.list_ainic
        )
        if handled_multiple_nics:
            return  # This function is recursive

        args.nic = device_handle

        # Get nic_id for logging
        nic_id = self.helpers.get_ainic_id_from_device_handle(args.nic)

        # Get nic info for logging
        try:
            ainic_info = amdsmi_interface.amdsmi_get_ainic_info(args.nic)
        except amdsmi_exception.AmdSmiLibraryException as e:
            bdf = uuid = device_name = part_number = firmware_version = "N/A"
            logging.debug("Failed to get info for nic %s | %s", nic_id, e.get_error_info())

        # CSV format is intentionally aligned with Host
        if self.logger.is_csv_format():
            self.logger.store_ainic_output(args.nic, "nic_bdf", ainic_info["bdf"])
            self.logger.store_ainic_output(
                args.nic, "permanent_address", ainic_info["Permanent Address"]
            )
            self.logger.store_ainic_output(args.nic, "product_name", ainic_info["Product Name"])
            self.logger.store_ainic_output(args.nic, "part_number", ainic_info["Part Number"])
            self.logger.store_ainic_output(args.nic, "serial_number", ainic_info["Serial Number"])
            self.logger.store_ainic_output(args.nic, "vendor_name", ainic_info["Vendor Name"])
        else:
            self.logger.store_ainic_output(args.nic, "bdf", ainic_info["bdf"])
            self.logger.store_ainic_output(
                args.nic, "permanent_address", ainic_info["Permanent Address"]
            )
            self.logger.store_ainic_output(args.nic, "product_name", ainic_info["Product Name"])
            self.logger.store_ainic_output(args.nic, "part_number", ainic_info["Part Number"])
            self.logger.store_ainic_output(args.nic, "serial_number", ainic_info["Serial Number"])
            self.logger.store_ainic_output(args.nic, "vendor_name", ainic_info["Vendor Name"])

        if multiple_devices:
            self.logger.store_multiple_device_output()
            return  # Skip printing when there are multiple devices

        self.logger.print_output()

    def list_nics(self, args):
        if not self.helpers.is_ainic_initialized() and not self.helpers.is_brcm_nic_initialized():
            return False
        if args.nic == None:
            args.nic = self.device_handles_ainics
            args.nic.extend(self.device_handles_brcm_nics)
            return False
        if not isinstance(args.nic, list):
            return False
        nicCount = len(args.nic)
        self.logger.output = {}
        self.logger.clear_multiple_devices_output()
        if nicCount <= 0:
            return False
        nics, ainics = self._get_nics_from_args(args)
        if len(nics) > 0:
            self.list_brcm_nic(args, False, nic=nics)
        if len(ainics) > 0:
            self.list_ainic(args, False, nic=ainics)
            return True
        return False

    def list_switch(self, args, multiple_devices=False, switch=None):
        """List information for target switch

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            switch (device_handle, optional): device_handle for target device. Defaults to None.

        Raises:
            IndexError: Index error if switch list is empty

        Returns:
            None: Print output via AMDSMILogger to destination
        """
        # Set args.* to passed in arguments
        if switch:
            args.switch = switch

        if not self.group_check_printed:
            self.helpers.check_required_groups()
            self.group_check_printed = True

        # Handle multiple Switches
        handled_multiple_switchs, device_handle = self.helpers.handle_switchs(
            args, self.logger, self.list_switch
        )
        if handled_multiple_switchs:
            return  # This function is recursive

        args.switch = device_handle

        try:
            bdf = amdsmi_interface.amdsmi_get_switch_device_bdf(args.switch)
        except amdsmi_exception.AmdSmiLibraryException as e:
            bdf = e.get_error_info()

        try:
            uuid = amdsmi_interface.amdsmi_get_switch_device_uuid(args.switch)
        except amdsmi_exception.AmdSmiLibraryException as e:
            uuid = e.get_error_info()

        # CSV format is intentionally aligned with Host
        if self.logger.is_csv_format():
            self.logger.store_switch_output(args.switch, "switch_bdf", bdf)
            self.logger.store_switch_output(args.switch, "switch_uuid", uuid)
        else:
            self.logger.store_switch_output(args.switch, "bdf", bdf)
            self.logger.store_switch_output(args.switch, "uuid", uuid)

        if multiple_devices:
            self.logger.store_multiple_device_output()
            return  # Skip printing when there are multiple devices

        self.logger.print_output()

    def list_switchs(self, args):
        if not self.helpers.is_brcm_switch_initialized():
            return False
        if args.switch == None:
            args.switch = self.device_handles_switchs
            return False
        if isinstance(args.switch, list):
            switchCount = len(args.switch)
            self.logger.output = {}
            self.logger.clear_multiple_devices_output()
            if switchCount > 0:
                self.list_switch(args, False, switch=args.switch)
                return True
        return False

    def _get_nics_from_args(self, args):
        nics = []
        ainics = []
        for nic in args.nic:
            for nic_ptr in self.device_handles_brcm_nics:
                if nic_ptr.value == nic.value:
                    nics.append(nic)
            for nic_ptr in self.device_handles_ainics:
                if nic_ptr.value == nic.value:
                    ainics.append(nic)
        return nics, ainics

    def list_cpu(self, args, multiple_devices=False, cpu=None):
        """List CPU identification (family, model, stepping) for target cpu.

        `list` only exposes the socket-level CPU identity. family and model come
        from the amd-smi APIs; stepping is not exposed by amd-smi so it is read
        from /proc/cpuinfo via the helpers.

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            cpu (device_handle, optional): device_handle for target device. Defaults to None.

        Returns:
            None: Print output via AMDSMILogger to destination
        """
        if cpu:
            args.cpu = cpu

        if getattr(args, "cpu", None) is None:
            args.cpu = self.cpu_handles

        if not self.group_check_printed:
            self.helpers.check_required_groups()
            self.group_check_printed = True

        # Handle multiple CPUs
        handled_multiple_cpus, device_handle = self.helpers.handle_cpus(
            args, self.logger, self.list_cpu
        )
        if handled_multiple_cpus:
            return  # This function is recursive

        args.cpu = device_handle

        cpu_id = self.helpers.get_cpu_id_from_device_handle(args.cpu)

        try:
            family = amdsmi_interface.amdsmi_get_cpu_family()
        except amdsmi_exception.AmdSmiLibraryException as e:
            family = "N/A"
            logging.debug("Failed to get cpu family for cpu %s | %s", cpu_id, e.get_error_info())

        try:
            model = amdsmi_interface.amdsmi_get_cpu_model()
        except amdsmi_exception.AmdSmiLibraryException as e:
            model = "N/A"
            logging.debug("Failed to get cpu model for cpu %s | %s", cpu_id, e.get_error_info())

        stepping = self.helpers.get_cpu_stepping()

        # Report the identification values as hex (N/A stays as-is).
        def _as_hex(value):
            return f"0x{value:X}" if isinstance(value, int) else value

        self.logger.store_cpu_output(args.cpu, "family", _as_hex(family))
        self.logger.store_cpu_output(args.cpu, "model", _as_hex(model))
        self.logger.store_cpu_output(args.cpu, "stepping", _as_hex(stepping))

        if multiple_devices:
            self.logger.store_multiple_device_output()
            return  # Skip printing when there are multiple devices

        self.logger.print_output()

    def list_devices(self, args, multiple_devices=False, gpu=None, nic=None, switch=None):

        # `list` reports GPU information. --cpu now reports CPU identification
        # (family/model/stepping); --core is still not serviced and is rejected.
        if getattr(args, "core", None):
            raise AmdSmiCommandNotSupportedException(
                "list",
                self.helpers.get_output_format(),
                message="list reports GPU information. --core is not supported for this command.",
            )
        if getattr(args, "cpu", None):
            self.list_cpu(args)
            return

        if gpu:
            args.gpu = gpu
        if nic:
            args.nic = nic
        if switch:
            args.switch = switch

        gpuCount = 0

        # Handle No GPU passed
        if args.gpu == None:
            args.gpu = self.device_handles_gpus
            if isinstance(args.gpu, list):
                gpuCount = len(args.gpu)
        else:
            if isinstance(args.gpu, list):
                gpuCount = len(args.gpu)
                self.logger.output = {}
                self.logger.clear_multiple_devices_output()

                if gpuCount > 0:
                    self.list_gpu(args, False, gpu=args.gpu)
                    return

        if self.list_nics(args):
            return
        if self.list_switchs(args):
            return

        self.logger.output = {}
        self.logger.clear_multiple_devices_output()

        if gpuCount > 0:
            self.list_gpu(args, False, gpu=args.gpu)

        self.logger.output = {}
        self.logger.clear_multiple_devices_output()

        if self.helpers.is_ainic_initialized() or self.helpers.is_brcm_nic_initialized():
            nics, ainics = self._get_nics_from_args(args)
            if len(nics) > 0:
                self.list_brcm_nic(args, False, nic=nics)
            if len(ainics) > 0:
                self.list_ainic(args, False, nic=ainics)

        self.logger.output = {}
        self.logger.clear_multiple_devices_output()

        if self.helpers.is_brcm_switch_initialized() and args.switch:
            self.list_switch(args, False, switch=args.switch)

        self.logger.output = {}
        self.logger.clear_multiple_devices_output()
