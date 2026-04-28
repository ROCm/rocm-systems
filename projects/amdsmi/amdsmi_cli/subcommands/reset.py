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
import sys

from amdsmi_cli_exceptions import AmdSmiRequiredCommandException
from amdsmi_helpers import AMDSMIHelpers

from amdsmi import amdsmi_exception, amdsmi_interface


class ResetCommands:
    def reset(
        self,
        args,
        multiple_devices=False,
        gpu=None,
        gpureset=None,
        clocks=None,
        fans=None,
        profile=None,
        xgmierr=None,
        perf_determinism=None,
        power_cap=None,
        clean_local_data=None,
        mem_carveout=None,
        gtt=None,
    ):
        """Issue reset commands to target gpu(s)

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            gpu (device_handle, optional): device_handle for target device. Defaults to None.
            gpureset (bool, optional): Value override for args.gpureset. Defaults to None.
            clocks (bool, optional): Value override for args.clocks. Defaults to None.
            fans (bool, optional): Value override for args.fans. Defaults to None.
            profile (bool, optional): Value override for args.profile. Defaults to None.
            xgmierr (bool, optional): Value override for args.xgmierr. Defaults to None.
            perf_determinism (bool, optional): Value override for args.perf_determinism. Defaults to None.
            power_cap (bool, optional): Value override for args.power_cap. Defaults to None.
            clean_local_data (bool, optional): Value override for args.run_cleaner_shader. Defaults to None.

        Raises:
            ValueError: Value error if no gpu value is provided
            IndexError: Index error if gpu list is empty

        Return:
            Nothing
        """
        # Set args.* to passed in arguments
        if gpu:
            args.gpu = gpu
        if gpureset:
            args.gpureset = gpureset
        if clocks:
            args.clocks = clocks
        if fans:
            args.fans = fans
        if profile:
            args.profile = profile
        if xgmierr:
            args.xgmierr = xgmierr
        if perf_determinism:
            args.perf_determinism = perf_determinism
        if power_cap:
            args.power_cap = power_cap
        if clean_local_data:
            args.clean_local_data = clean_local_data
        # Normalize gpureset: not available on VMs
        if not self.helpers.is_baremetal():
            args.gpureset = False

        # Special GTT handling (system-wide, not per-GPU) — handle before device dispatch
        if hasattr(args, "gtt") and args.gtt:
            if hasattr(args, "gpu") and args.gpu is not None:
                print(
                    "amd-smi reset: error: argument --gtt: not allowed with argument --gpu/-g "
                    "(--gtt is a system-wide setting, not per-GPU)",
                    file=sys.stderr,
                )
                sys.exit(2)
            try:
                amdsmi_interface.amdsmi_reset_ttm_pages_limit()
                self.logger.output["reset_gtt"] = (
                    "Successfully reset GTT to system default. Reboot required for changes to take effect."
                )
                self.logger.print_output()
                self.helpers.prompt_reboot()
                return
            except amdsmi_exception.AmdSmiLibraryException as e:
                if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                    raise PermissionError("Command requires elevation") from e
                self.logger.output["reset_gtt"] = (
                    f"[{e.get_error_info(detailed=False)}] Unable to reset GTT"
                )
                self.logger.print_output()
                return

        # Handle No GPU passed
        if args.gpu == None:
            args.gpu = self.device_handles

        if not self.group_check_printed:
            self.helpers.check_required_groups()
            self.group_check_printed = True

        # Mode-1 gpureset is hive-wide.
        # Group GPUs by hive and reset each hive only once.
        gpus_to_reset = []

        if args.gpureset and isinstance(args.gpu, list) and len(args.gpu) > 1:
            # Group GPUs by their XGMI hive ID.
            # If GPU not in a hive or no hive info, reset individually.
            hive_to_gpus = {}
            gpus_without_hive = []

            for gpu in args.gpu:
                try:
                    xgmi_info = amdsmi_interface.amdsmi_get_xgmi_info(gpu)
                    if isinstance(xgmi_info, dict):
                        hive_id = xgmi_info.get("xgmi_hive_id", None)
                        if hive_id is not None and hive_id != 0:
                            if hive_id not in hive_to_gpus:
                                hive_to_gpus[hive_id] = []
                            hive_to_gpus[hive_id].append(gpu)
                        else:
                            gpus_without_hive.append(gpu)
                    else:
                        gpus_without_hive.append(gpu)
                except:
                    gpus_without_hive.append(gpu)

            # For each hive, reset using the first GPU (resets entire hive)
            for hive_id, gpu_list in hive_to_gpus.items():
                gpus_to_reset.append(gpu_list[0])

            # Add all non-hive GPUs to reset individually
            gpus_to_reset.extend(gpus_without_hive)

            # Update args.gpu to only the GPUs to reset
            if gpus_to_reset:
                args.gpu = gpus_to_reset

        # Handle multiple GPUs
        handled_multiple_gpus, device_handle = self.helpers.handle_gpus(
            args, self.logger, self.reset
        )
        if handled_multiple_gpus:
            return  # This function is recursive

        args.gpu = device_handle

        # Get gpu_id for logging
        gpu_id = self.helpers.get_gpu_id_from_device_handle(args.gpu)

        # Error if no subcommand args are passed
        if self.helpers.is_baremetal():
            if not any(
                [
                    args.gpureset,
                    args.clocks,
                    args.fans,
                    args.profile,
                    args.xgmierr,
                    args.perf_determinism,
                    args.power_cap,
                    args.clean_local_data,
                ]
            ):
                command = " ".join(sys.argv[1:])
                raise AmdSmiRequiredCommandException(command, self.logger.format)
        else:
            if not any([args.clean_local_data]):
                command = " ".join(sys.argv[1:])
                raise AmdSmiRequiredCommandException(command, self.logger.format)

        #######################
        # BM commands - START #
        #######################

        if self.helpers.is_baremetal():
            if args.gpureset:
                if self.helpers.is_amd_device(args.gpu):
                    try:
                        amdsmi_interface.amdsmi_reset_gpu(args.gpu)
                        result = "Successfully reset GPU"
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        if (
                            e.get_error_code()
                            == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM
                        ):
                            raise PermissionError("Command requires elevation") from e
                        result = f"[{e.get_error_info(detailed=False)}] Unable to reset GPU"
                        self.logger.store_output(args.gpu, "gpu_reset", result)
                        self.logger.print_output()
                        self.logger.clear_multiple_devices_output()
                        return
                else:
                    result = "Unable to reset non-amd GPU"
                self.logger.store_output(args.gpu, "gpu_reset", result)
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return
            if args.clocks:
                reset_clocks_results = {"overdrive": "", "clocks": "", "performance": ""}
                try:
                    amdsmi_interface.amdsmi_set_gpu_overdrive_level(args.gpu, 0)
                    reset_clocks_results["overdrive"] = "Overdrive set to 0"
                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        raise PermissionError("Command requires elevation") from e
                    logging.debug(
                        "Failed to reset overdrive on gpu %s | %s", gpu_id, e.get_error_info()
                    )
                    reset_clocks_results["overdrive"] = (
                        f"[{e.get_error_info(detailed=False)}] Unable to reset overdrive to 0"
                    )
                    # continue to reset clocks and performance level
                try:
                    level_auto = amdsmi_interface.AmdSmiDevPerfLevel.AUTO
                    amdsmi_interface.amdsmi_set_gpu_perf_level(args.gpu, level_auto)
                    reset_clocks_results["clocks"] = "Successfully reset performance level to auto"
                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        raise PermissionError("Command requires elevation") from e
                    reset_clocks_results["clocks"] = (
                        f"[{e.get_error_info(detailed=False)}] Unable to reset performance level to auto"
                    )
                    logging.debug(
                        "Failed to reset perf level on gpu %s | %s", gpu_id, e.get_error_info()
                    )

                try:
                    # TODO: Check why this is called twice?
                    level_auto = amdsmi_interface.AmdSmiDevPerfLevel.AUTO
                    amdsmi_interface.amdsmi_set_gpu_perf_level(args.gpu, level_auto)
                    reset_clocks_results["performance"] = (
                        "Successfully reset performance level to auto"
                    )
                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        raise PermissionError("Command requires elevation") from e
                    reset_clocks_results["performance"] = (
                        f"[{e.get_error_info(detailed=False)}] Unable to reset performance level to auto"
                    )
                    logging.debug(
                        "Failed to reset perf level on gpu %s | %s", gpu_id, e.get_error_info()
                    )

                self.logger.store_output(args.gpu, "reset_clocks", reset_clocks_results)
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return
            if args.fans:
                try:
                    amdsmi_interface.amdsmi_reset_gpu_fan(args.gpu, 0)
                    result = "Successfully reset fan speed to driver control"
                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        raise PermissionError("Command requires elevation") from e
                    result = f"[{e.get_error_info(detailed=False)}] Unable to reset fan speed to driver control"
                    logging.debug("Failed to reset fans on gpu %s | %s", gpu_id, e.get_error_info())
                    self.logger.store_output(args.gpu, "reset_fans", result)
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    return
                self.logger.store_output(args.gpu, "reset_fans", result)
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return
            if args.profile:
                reset_profile_results = {"power_profile": "N/A"}
                try:
                    power_profile_mask = (
                        amdsmi_interface.AmdSmiPowerProfilePresetMasks.BOOTUP_DEFAULT
                    )
                    amdsmi_interface.amdsmi_set_gpu_power_profile(args.gpu, 0, power_profile_mask)
                    reset_profile_results["power_profile"] = (
                        "Successfully reset Power Profile to default (bootup default)"
                    )
                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        raise PermissionError("Command requires elevation") from e
                    reset_profile_results["power_profile"] = (
                        f"[{e.get_error_info(detailed=False)}] Unable to reset Power Profile to default (bootup default)"
                    )
                    logging.debug(
                        "Failed to reset power profile on gpu %s | %s", gpu_id, e.get_error_info()
                    )

                self.logger.store_output(args.gpu, "reset_profile", reset_profile_results)
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return
            if args.xgmierr:
                try:
                    amdsmi_interface.amdsmi_reset_gpu_xgmi_error(args.gpu)
                    result = "Successfully reset XGMI Error count"
                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        raise PermissionError("Command requires elevation") from e
                    logging.debug(
                        "Failed to reset xgmi error count on gpu %s | %s",
                        gpu_id,
                        e.get_error_info(),
                    )
                    result = (
                        f"[{e.get_error_info(detailed=False)}] Unable to reset XGMI Error count"
                    )
                    self.logger.store_output(args.gpu, "reset_xgmi_err", result)
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    return
                self.logger.store_output(args.gpu, "reset_xgmi_err", result)
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return
            if args.perf_determinism:
                try:
                    level_auto = amdsmi_interface.AmdSmiDevPerfLevel.AUTO
                    amdsmi_interface.amdsmi_set_gpu_perf_level(args.gpu, level_auto)
                    result = "Successfully reset Performance Level to default (auto)"
                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        raise PermissionError("Command requires elevation") from e
                    logging.debug(
                        "Failed to set perf level on gpu %s | %s", gpu_id, e.get_error_info()
                    )
                    result = f"[{e.get_error_info(detailed=False)}] Unable to reset Performance Level to default (auto)"
                    self.logger.store_output(args.gpu, "reset_perf_determinism", result)
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    return
                self.logger.store_output(args.gpu, "reset_perf_determinism", result)
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return
            if args.power_cap:
                final_output = {"ppt0": "N/A", "ppt1": "N/A"}
                power_limit_types = {}
                for power_type in amdsmi_interface.AmdSmiPowerCapType:
                    # Strip 'AMDSMI_POWER_CAP_TYPE_' prefix and convert to lowercase
                    key = power_type.name.replace("AMDSMI_POWER_CAP_TYPE_", "").lower()
                    power_limit_types[key] = "N/A"
                current_sensor_num = 0

                try:
                    power_cap_types = amdsmi_interface.amdsmi_get_supported_power_cap(args.gpu)
                    for sensor in power_cap_types["sensor_inds"]:
                        current_sensor_num = sensor
                        power_cap_info = amdsmi_interface.amdsmi_get_power_cap_info(
                            args.gpu, sensor
                        )
                        logging.debug(
                            f"Power cap info for gpu {gpu_id} ppt{sensor} | {power_cap_info}"
                        )
                        default_power_cap_in_mw = power_cap_info["default_power_cap"]
                        default_power_cap_in_w = self.helpers.convert_SI_unit(
                            default_power_cap_in_mw, AMDSMIHelpers.SI_Unit.MICRO
                        )
                        current_power_cap_in_mw = power_cap_info["power_cap"]
                        current_power_cap_in_w = self.helpers.convert_SI_unit(
                            current_power_cap_in_mw, AMDSMIHelpers.SI_Unit.MICRO
                        )
                        sensor_name = power_cap_types["sensor_types"][sensor]
                        # Strip 'AMDSMI_POWER_CAP_TYPE_' prefix and convert to lowercase
                        sensor_key = sensor_name.name.replace("AMDSMI_POWER_CAP_TYPE_", "").lower()
                        power_limit_types[sensor_key] = (
                            default_power_cap_in_w,
                            current_power_cap_in_w,
                        )
                        amdsmi_interface.amdsmi_set_power_cap(
                            args.gpu, sensor, default_power_cap_in_mw
                        )
                        final_output[f"ppt{current_sensor_num}"] = (
                            f"Successfully reset power cap to {default_power_cap_in_w}W"
                        )
                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        raise PermissionError("Command requires elevation") from e
                    final_output[f"ppt{current_sensor_num}"] = (
                        f"[{e.get_error_info(detailed=False)}] Unable to reset cap to default power cap"
                    )
                self.logger.store_output(args.gpu, "powercap", final_output)
                if multiple_devices:
                    self.logger.store_multiple_device_output()
                    return
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()

        #######################
        # BM commands - END   #
        #######################

        if args.clean_local_data:
            try:
                amdsmi_interface.amdsmi_clean_gpu_local_data(args.gpu)
                result = "Successfully clean GPU local data"
            except amdsmi_exception.AmdSmiLibraryException as e:
                if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                    raise PermissionError("Command requires elevation") from e
                result = f"[{e.get_error_info(detailed=False)}] Unable to clean local data"
                self.logger.store_output(args.gpu, "clean_local_data", result)
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return
            self.logger.store_output(args.gpu, "clean_local_data", result)
            self.logger.print_output()
            self.logger.clear_multiple_devices_output()
            return
