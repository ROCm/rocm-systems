#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import sys

from amdsmi import amdsmi_exception, amdsmi_interface

# PCI VPD is mode 0600, so an unprivileged run cannot tell a card with no VPD
# from one it may not read. Identity fields go "N/A" either way.
_VPD_IDENTITY_FIELDS = ("Product Name", "Part Number", "Serial Number")
_VPD_ROOT_HINT = (
    "Note: PRODUCT_NAME, PART_NUMBER and SERIAL_NUMBER are read from PCI VPD, "
    "which only root can read. An N/A field is either suppressed by permissions "
    "or absent from the device; re-run with sudo to tell the two apart."
)


class ListDevicesCommands:
    # Set by a row whose identity was suppressed by the euid, consumed once
    # after the NIC output prints.
    is_vpd_root_hint_pending = False

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

        cpu_attributes = ["cpu"]
        for attr in cpu_attributes:
            if hasattr(args, "cpu") and getattr(args, "cpu"):
                print("N/A")
                return

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

        # Use CUID for UUID if available, fall back to the standard UUID if not
        uuid = self.helpers.get_gpu_cuid_or_uuid(args.gpu)

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
            self._print_vpd_root_hint()
            return  # This function is recursive

        args.nic = device_handle

        # Get nic_id for logging
        nic_id = self.helpers.get_ainic_id_from_device_handle(args.nic)

        # Get nic info for logging
        is_ainic_info_read = True
        try:
            ainic_info = amdsmi_interface.amdsmi_get_ainic_info(args.nic)
        except amdsmi_exception.AmdSmiLibraryException as e:
            is_ainic_info_read = False
            ainic_info = {
                "bdf": "N/A",
                "Permanent Address": "N/A",
                "Product Name": "N/A",
                "Part Number": "N/A",
                "Serial Number": "N/A",
                "Vendor Name": "N/A",
                "Capability": [],
            }
            logging.debug("Failed to get info for nic %s | %s", nic_id, e.get_error_info())

        # Only a successful read says anything about VPD; the failure path above
        # fills the same fields with "N/A" for an unrelated reason.
        if (
            is_ainic_info_read
            and (os.geteuid() != 0)
            and any(ainic_info[field] == "N/A" for field in _VPD_IDENTITY_FIELDS)
        ):
            self.is_vpd_root_hint_pending = True

        # A fwctl-only card has no host netdev, so its permanent address is
        # meaningless. Synthesize a mode from the capability bits.
        capability = ainic_info["Capability"]
        is_netdev = "NETDEV" in capability
        mode = "netdev" if is_netdev else ("fwctl-only" if "FWCTL" in capability else "unknown")

        # CSV format is intentionally aligned with Host
        if self.logger.is_csv_format():
            self.logger.store_ainic_output(args.nic, "nic_bdf", ainic_info["bdf"])
            self.logger.store_ainic_output(args.nic, "mode", mode)
            # Keep the column for alignment; blank it when there is no netdev.
            self.logger.store_ainic_output(
                args.nic, "permanent_address", ainic_info["Permanent Address"] if is_netdev else ""
            )
            self.logger.store_ainic_output(args.nic, "product_name", ainic_info["Product Name"])
            self.logger.store_ainic_output(args.nic, "part_number", ainic_info["Part Number"])
            self.logger.store_ainic_output(args.nic, "serial_number", ainic_info["Serial Number"])
            self.logger.store_ainic_output(args.nic, "vendor_name", ainic_info["Vendor Name"])
            self.logger.store_ainic_output(args.nic, "capability", ";".join(capability))
        else:
            self.logger.store_ainic_output(args.nic, "bdf", ainic_info["bdf"])
            self.logger.store_ainic_output(args.nic, "mode", mode)
            if is_netdev:
                self.logger.store_ainic_output(
                    args.nic, "permanent_address", ainic_info["Permanent Address"]
                )
            self.logger.store_ainic_output(args.nic, "product_name", ainic_info["Product Name"])
            self.logger.store_ainic_output(args.nic, "part_number", ainic_info["Part Number"])
            self.logger.store_ainic_output(args.nic, "serial_number", ainic_info["Serial Number"])
            self.logger.store_ainic_output(args.nic, "vendor_name", ainic_info["Vendor Name"])
            self.logger.store_ainic_output(args.nic, "capability", capability)

        if multiple_devices:
            self.logger.store_multiple_device_output()
            return  # Skip printing when there are multiple devices

        self.logger.print_output()
        self._print_vpd_root_hint()

    # Goes to stderr so machine-readable output on stdout stays parseable, and
    # only once per invocation however many rows were suppressed.
    def _print_vpd_root_hint(self):
        if not self.is_vpd_root_hint_pending:
            return
        self.is_vpd_root_hint_pending = False
        if self.logger.is_human_readable_format():
            print(_VPD_ROOT_HINT, file=sys.stderr)

    def list_nics(self, args):
        if not self.helpers.is_ainic_initialized():
            return False
        if args.nic == None:
            args.nic = self.device_handles_ainics
            return False
        if not isinstance(args.nic, list):
            return False
        nicCount = len(args.nic)
        self.logger.output = {}
        self.logger.clear_multiple_devices_output()
        if nicCount <= 0:
            return False
        ainics = self._get_ainics_from_args(args)
        if len(ainics) > 0:
            self.list_ainic(args, False, nic=ainics)
            return True
        return False

    def _get_ainics_from_args(self, args):
        ainics = []
        for nic in args.nic:
            for nic_ptr in self.device_handles_ainics:
                if nic_ptr.value == nic.value:
                    ainics.append(nic)
        return ainics

    def list_devices(self, args, multiple_devices=False, gpu=None, nic=None):

        if gpu:
            args.gpu = gpu
        if nic:
            args.nic = nic

        # Capture explicit NIC intent before list_nics() rewrites a None args.nic
        # into the full handle list; a --nic query that matches nothing must not
        # silently fall through to listing GPUs.
        nic_requested = args.nic is not None

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

        if nic_requested:
            # Explicit --nic matched no NICs: report that, don't list GPUs.
            self.logger.output = {}
            self.logger.clear_multiple_devices_output()
            if self.logger.is_human_readable_format():
                print("No AI NICs found")
            else:
                self.logger.print_output(emit_empty=True)
            return

        self.logger.output = {}
        self.logger.clear_multiple_devices_output()

        if gpuCount > 0:
            self.list_gpu(args, False, gpu=args.gpu)

        self.logger.output = {}
        self.logger.clear_multiple_devices_output()

        if self.helpers.is_ainic_initialized():
            ainics = self._get_ainics_from_args(args)
            if len(ainics) > 0:
                self.list_ainic(args, False, nic=ainics)

        self.logger.output = {}
        self.logger.clear_multiple_devices_output()
