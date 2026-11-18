#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging

from amdsmi import amdsmi_exception, amdsmi_interface


def _fabric_ppod_id_to_uuid_string(ppod_id):
    """Format 16-byte pPoD id as 8-4-4-4-12 hex (canonical UUID-style), one line.

    If the value is missing or not 16 bytes, returns the all-9s sentinel used when
    no pPoD id is available from the library.
    """
    sentinel = "99999999-9999-9999-9999-999999999999"
    if not isinstance(ppod_id, (list, tuple)) or len(ppod_id) != 16:
        return sentinel
    try:
        octets = bytes(int(x) & 0xFF for x in ppod_id)
    except (TypeError, ValueError):
        return sentinel
    h = octets.hex()
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


# Unset slots in local_accelerators[] use UINT32_MAX (see amd_smi_gpu_device.cc).
_FABRIC_LOCAL_ACCEL_UNSET = (1 << 32) - 1


def _fabric_local_accelerators_to_line(local_accelerators):
    """Format local accelerator ids as one comma-separated line; skip unset."""
    if not isinstance(local_accelerators, (list, tuple)):
        return "N/A"
    try:
        vals = [int(x) for x in local_accelerators]
    except (TypeError, ValueError):
        return "N/A"
    active = [str(v) for v in vals if v != _FABRIC_LOCAL_ACCEL_UNSET]
    return ", ".join(active) if active else "N/A"


_FABRIC_VPOD_ACTIVE_WORD_BITS = 32
_FABRIC_VPOD_ACTIVE_ROW_LEN = 8


def _fabric_local_active_accelerators_rows(vpod_active_accelerators):
    """Decode the vPoD-active bitmap (bit N % 32 of word N / 32 = accelerator id N) into rows of
    up to 8 comma-separated ids. Returns None if invalid or no accelerator is active."""
    if not isinstance(vpod_active_accelerators, (list, tuple)):
        return None
    try:
        words = [int(x) for x in vpod_active_accelerators]
    except (TypeError, ValueError):
        return None
    ids = [
        word_idx * _FABRIC_VPOD_ACTIVE_WORD_BITS + bit
        for word_idx, word in enumerate(words)
        for bit in range(_FABRIC_VPOD_ACTIVE_WORD_BITS)
        if (word >> bit) & 1
    ]
    if not ids:
        return None
    return [
        ", ".join(str(i) for i in ids[start : start + _FABRIC_VPOD_ACTIVE_ROW_LEN])
        for start in range(0, len(ids), _FABRIC_VPOD_ACTIVE_ROW_LEN)
    ]


class FabricCommands:
    def fabric(self, args, multiple_devices=False, gpu=None, topology=None, telemetry=None):
        """Get fabric (UALoE) information for target GPUs.

        params:
            args             - argparser args to pass to subcommand
            multiple_devices - True if checking for multiple devices
            gpu              - device_handle override for args.gpu
            topology         - Value override for args.topology
            telemetry        - Value override for args.telemetry
        return:
            Nothing
        """
        # Set args.* from overrides
        if gpu:
            args.gpu = gpu
        if topology is not None:
            args.topology = topology
        if telemetry is not None:
            args.telemetry = telemetry

        # Default to all GPUs if none specified
        if args.gpu is None:
            args.gpu = self.device_handles

        handled_multiple_gpus, device_handle = self.helpers.handle_gpus(
            args, self.logger, self.fabric
        )
        if handled_multiple_gpus:
            return

        args.gpu = device_handle

        # Default: show everything if no specific arg given
        if not any([args.topology, args.telemetry]):
            args.topology = True
            args.telemetry = True

        gpu_handle = args.gpu
        gpu_id = self.helpers.get_gpu_id_from_device_handle(gpu_handle)
        gpu_bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(gpu_handle)
        values = {"gpu": gpu_id, "bdf": gpu_bdf}

        if args.topology:
            fabric_info = "N/A"
            try:
                raw = amdsmi_interface.amdsmi_get_gpu_fabric_info(gpu_handle)
                ppod_uuid = _fabric_ppod_id_to_uuid_string(raw["ppod_id"])
                local_accels_line = _fabric_local_accelerators_to_line(raw["local_accelerators"])
                local_active_rows = _fabric_local_active_accelerators_rows(
                    raw["vpod_active_accelerators"]
                )
                if local_active_rows is None:
                    local_active_human = "N/A"
                    local_active_json = "N/A"
                    local_active_csv = "N/A"
                else:
                    local_active_human = "\n".join(local_active_rows)
                    local_active_json = local_active_rows
                    local_active_csv = "; ".join(local_active_rows)
                if self.logger.is_human_readable_format():
                    fabric_info = {
                        "bdf": raw["bdf"],
                        "version": raw["version"],
                        "accelerator_id": raw["accelerator_id"],
                        "fabric_type": raw["fabric_type"],
                        "bandwidth": f"{raw['bandwidth']} Mb/s",
                        "latency": f"{raw['latency']} ns",
                        "ppod_id": ppod_uuid,
                        "ppod_size": raw["ppod_size"],
                        "vpod_id": raw["vpod_id"],
                        "vpod_size": raw["vpod_size"],
                        "local_accelerators": local_accels_line,
                        "local_active_accelerators": local_active_human,
                        "addr_mode": raw["addr_mode"],
                        "accel_state": raw["accel_state"],
                    }
                elif self.logger.is_json_format():
                    fabric_info = {
                        "bdf": raw["bdf"],
                        "version": raw["version"],
                        "accelerator_id": raw["accelerator_id"],
                        "fabric_type": raw["fabric_type"],
                        "bandwidth": {"value": raw["bandwidth"], "unit": "Mb/s"},
                        "latency": {"value": raw["latency"], "unit": "ns"},
                        "ppod_id": ppod_uuid,
                        "ppod_size": raw["ppod_size"],
                        "vpod_id": raw["vpod_id"],
                        "vpod_size": raw["vpod_size"],
                        "local_accelerators": local_accels_line,
                        "local_active_accelerators": local_active_json,
                        "addr_mode": raw["addr_mode"],
                        "accel_state": raw["accel_state"],
                    }
                elif self.logger.is_csv_format():
                    fabric_info = {
                        "bdf": raw["bdf"],
                        "version": raw["version"],
                        "accelerator_id": raw["accelerator_id"],
                        "fabric_type": raw["fabric_type"],
                        "bandwidth_mb_s": raw["bandwidth"],
                        "latency_ns": raw["latency"],
                        "ppod_id": ppod_uuid,
                        "ppod_size": raw["ppod_size"],
                        "vpod_id": raw["vpod_id"],
                        "vpod_size": raw["vpod_size"],
                        "local_accelerators": local_accels_line,
                        "local_active_accelerators": local_active_csv,
                        "addr_mode": raw["addr_mode"],
                        "accel_state": raw["accel_state"],
                    }
            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug(
                    "Failed to get fabric info for GPU %s | %s", gpu_id, e.get_error_info()
                )
            values["fabric_info"] = fabric_info

        if args.telemetry:
            fabric_telemetry = "N/A"
            try:
                category_mask = (
                    amdsmi_interface.amdsmi_wrapper.AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_UALOE
                    | amdsmi_interface.amdsmi_wrapper.AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_SWITCH
                    | amdsmi_interface.amdsmi_wrapper.AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_CRYPTO
                    | amdsmi_interface.amdsmi_wrapper.AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_PFC
                    | amdsmi_interface.amdsmi_wrapper.AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_NETPORT
                    | amdsmi_interface.amdsmi_wrapper.AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_DERIVED_UALOE
                    | amdsmi_interface.amdsmi_wrapper.AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_DERIVED_NETPORT
                )
                fabric_telemetry = amdsmi_interface.amdsmi_get_fabric_telemetry_data(
                    gpu_handle, category_mask
                )
            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug(
                    "Failed to get fabric telemetry for GPU %s | %s", gpu_id, e.get_error_info()
                )
            values["fabric_telemetry"] = fabric_telemetry

        self.logger.store_output(gpu_handle, "fabric", values)
        if multiple_devices:
            self.logger.store_multiple_device_output()
            return

        self.logger.print_output(multiple_device_enabled=False)
