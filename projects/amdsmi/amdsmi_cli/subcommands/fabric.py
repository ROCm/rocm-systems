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


# Unset slots in local_accelerators[] and vpod_active_accelerators[] use UINT32_MAX
# (see amd_smi_gpu_device.cc).
_FABRIC_LOCAL_ACCEL_UNSET = (1 << 32) - 1


def _fabric_local_accelerators_to_line(local_accelerators):
    """Format accelerator id list (local or vPoD-active) as one comma-separated line; skip unset."""
    if not isinstance(local_accelerators, (list, tuple)):
        return "N/A"
    try:
        vals = [int(x) for x in local_accelerators]
    except (TypeError, ValueError):
        return "N/A"
    active = [str(v) for v in vals if v != _FABRIC_LOCAL_ACCEL_UNSET]
    return ", ".join(active) if active else "N/A"


_FABRIC_VPOD_ACTIVE_SLOTS = 32
_FABRIC_VPOD_ACTIVE_ROW_LEN = 8


def _fabric_local_active_accelerators_rows(vpod_active_accelerators):
    """32 vPoD-active slots as 4×8 grid; each row is comma-separated ('-' if unset). Returns None if invalid."""
    unset = _FABRIC_LOCAL_ACCEL_UNSET
    if not isinstance(vpod_active_accelerators, (list, tuple)):
        return None
    try:
        vals = [int(x) for x in vpod_active_accelerators]
    except (TypeError, ValueError):
        return None
    if len(vals) != _FABRIC_VPOD_ACTIVE_SLOTS:
        if len(vals) > _FABRIC_VPOD_ACTIVE_SLOTS:
            vals = vals[:_FABRIC_VPOD_ACTIVE_SLOTS]
        else:
            vals = vals + [unset] * (_FABRIC_VPOD_ACTIVE_SLOTS - len(vals))
    rows = []
    for r in range(_FABRIC_VPOD_ACTIVE_SLOTS // _FABRIC_VPOD_ACTIVE_ROW_LEN):
        start = r * _FABRIC_VPOD_ACTIVE_ROW_LEN
        chunk = vals[start : start + _FABRIC_VPOD_ACTIVE_ROW_LEN]
        rows.append(", ".join(str(v) if v != unset else "-" for v in chunk))
    return rows


class FabricCommands:
    def fabric(self, args, multiple_devices=False, gpu=None, topology=None, info=None):
        """Get fabric (UALoE) information for target GPUs.

        params:
            args             - argparser args to pass to subcommand
            multiple_devices - True if checking for multiple devices
            gpu              - device_handle override for args.gpu
            topology         - Value override for args.topology
            info             - Value override for args.info
        return:
            Nothing
        """
        # Set args.* from overrides
        if gpu:
            args.gpu = gpu
        if topology is not None:
            args.topology = topology
        if info is not None:
            args.info = info

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
        if not any([args.topology, args.info]):
            args.topology = True
            args.info = True

        gpu_handle = args.gpu
        gpu_id = self.helpers.get_gpu_id_from_device_handle(gpu_handle)
        gpu_bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(gpu_handle)
        values = {"gpu": gpu_id, "bdf": gpu_bdf}

        # --info ──────────────────────────────────────────────────────────────
        if args.info:
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

        # --topology ─────────────────────────────────────────────────────────
        if args.topology:
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
