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

import argparse
import json
import logging
import multiprocessing
import os
import sys
import threading
import time
import copy

from _version import __version__
from amdsmi_cli_exceptions import AmdSmiInvalidParameterException, AmdSmiRequiredCommandException, AmdSmiInvalidCommandException
from amdsmi_helpers import AMDSMIHelpers
from amdsmi_logger import AMDSMILogger
from amdsmi import amdsmi_exception, amdsmi_interface
from pathlib import Path

class AMDSMICommands():
    """This class contains all the commands corresponding to AMDSMIParser
    Each command function will interact with AMDSMILogger to handle
    displaying the output to the specified format and destination.
    """

    def __init__(self, format='human_readable', destination='stdout', helpers=None) -> None:
        if helpers is None:
            # If helpers is not provided, create a new instance
            self.helpers = AMDSMIHelpers()
        else:
            self.helpers = helpers
        self.logger = AMDSMILogger(format=format, destination=destination, helpers=self.helpers)
        self.device_handles = []
        self.cpu_handles = []
        self.core_handles = []
        self.stop = ''
        self.group_check_printed = False

        amdsmi_init_flag = self.helpers.get_amdsmi_init_flag()
        logging.debug(f"AMDSMI Init Flag: {amdsmi_init_flag}")
        exit_flag = False

        if self.helpers.is_amdgpu_initialized():
            try:
                self.device_handles = amdsmi_interface.amdsmi_get_processor_handles()
            except amdsmi_exception.AmdSmiLibraryException as e:
                if e.err_code in (amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NOT_INIT,
                                amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_DRIVER_NOT_LOADED):
                    logging.error('Unable to get devices, driver not initialized (amdgpu not found in modules)')
                else:
                    raise e

            if len(self.device_handles) == 0:
                # No GPU's found post amdgpu driver initialization
                logging.error('Unable to detect any GPU devices, check amdgpu version and module status (sudo modprobe amdgpu)')
                exit_flag = True

        if self.helpers.is_amd_hsmp_initialized():
            try:
                self.cpu_handles = amdsmi_interface.amdsmi_get_cpusocket_handles()
            except amdsmi_exception.AmdSmiLibraryException as e:
                if e.err_code in (amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NOT_INIT,
                                amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_DRV):
                    logging.info('Unable to detect any CPU devices, check amd_hsmp version and module status (sudo modprobe amd_hsmp)')
                else:
                    raise e

            # core handles
            try:
                self.core_handles = amdsmi_interface.amdsmi_get_cpucore_handles()
            except amdsmi_exception.AmdSmiLibraryException as e:
                if e.err_code in (amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NOT_INIT,
                                amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_DRV):
                    logging.info('Unable to get CORE devices, amd_hsmp driver not loaded (sudo modprobe amd_hsmp)')
                else:
                    raise e

            if len(self.cpu_handles) == 0 and len(self.core_handles) == 0:
                # No CPU's found post amd_hsmp driver initialization
                logging.error('Unable to detect any CPU devices, check amd_hsmp version and module status (sudo modprobe amd_hsmp)')
                exit_flag = True

        if exit_flag:
            version_args = argparse.Namespace()
            version_args.gpu_version = False
            version_args.cpu_version = False
            self.version(version_args)
            sys.exit(-1)


    def version(self, args, gpu_version=None, cpu_version=None):
        """Print Version String

        Args:
            args (Namespace): Namespace containing the parsed CLI args
        """

        if gpu_version:
            args.gpu_version = gpu_version
        if cpu_version:
            args.cpu_version = cpu_version
        # if no args are given, display everything
        if args.gpu_version is None and args.cpu_version is None:
            args.gpu_version = True
            args.cpu_version = True

        try:
            amdsmi_lib_version = amdsmi_interface.amdsmi_get_lib_version()
            amdsmi_lib_version_str = f"{amdsmi_lib_version['major']}.{amdsmi_lib_version['minor']}.{amdsmi_lib_version['release']}"
        except amdsmi_exception.AmdSmiLibraryException as e:
            amdsmi_lib_version_str = e.get_error_info()

        try:
            rocm_lib_status, rocm_version_str = amdsmi_interface.amdsmi_get_rocm_version()
            if rocm_lib_status is not True:
                rocm_version_str = "N/A"
        except amdsmi_exception.AmdSmiLibraryException as e:
            rocm_version_str = e.get_error_info()

        self.logger.output['tool'] = 'AMDSMI Tool'
        self.logger.output['version'] = f'{__version__}'
        self.logger.output['amdsmi_library_version'] = f'{amdsmi_lib_version_str}'
        self.logger.output['rocm_version'] = f'{rocm_version_str}'

        if args.gpu_version:
            try:
                gpus = amdsmi_interface.amdsmi_get_processor_handles()
                if isinstance(gpus, list) and len(gpus) > 0:
                    gpu_version_info = amdsmi_interface.amdsmi_get_gpu_driver_info(gpus[0])
                    gpu_version_str = gpu_version_info['driver_version']
                else:
                    gpu_version_str = "N/A"
            except amdsmi_exception.AmdSmiLibraryException as e:
                gpu_version_str = e.get_error_info()
            self.logger.output['amdgpu_version'] = gpu_version_str
        if args.cpu_version:
            try:
                cpus = amdsmi_interface.amdsmi_get_cpusocket_handles()
                if isinstance(cpus, list) and len(cpus) > 0:
                    cpu_version_info = amdsmi_interface.amdsmi_get_cpu_hsmp_driver_version(cpus[0])
                    cpu_version_str = str(cpu_version_info['hsmp_driver_major_ver_num']) + "." + str(cpu_version_info['hsmp_driver_minor_ver_num'])
                else:
                    cpu_version_str = "N/A"
            except amdsmi_exception.AmdSmiLibraryException as e:
                cpu_version_str = e.get_error_info()
            self.logger.output['amd_hsmp_driver_version'] = cpu_version_str

        if self.logger.is_human_readable_format():
            human_readable_output = f"AMDSMI Tool: {__version__} | " \
                                    f"AMDSMI Library version: {amdsmi_lib_version_str} | " \
                                    f"ROCm version: {rocm_version_str}"
            if args.gpu_version:
                human_readable_output = human_readable_output + f" | amdgpu version: {gpu_version_str}"
            if args.cpu_version:
                human_readable_output = human_readable_output + f" | amd_hsmp version: {cpu_version_str}"
            # Custom human readable handling for version
            if self.logger.destination == 'stdout':
                print(human_readable_output)
            else:
                with self.logger.destination.open('a', encoding="utf-8") as output_file:
                    output_file.write(human_readable_output + '\n')
        elif self.logger.is_json_format() or self.logger.is_csv_format():
            self.logger.print_output()


    def list(self, args, multiple_devices=False, gpu=None):
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

        # Perform one-time group check. If it fails, record that fact
        # but do NOT abort—just mark that UUID should be "N/A" later.
        _group_check_done = False
        _group_in_groups = False
        if not _group_check_done:
           try:
               self.helpers.check_required_groups()
               _group_in_groups = True
           except Exception as e:
               _group_in_groups = False
               # print the helper's error message exactly once:
               print(f"{e}")
           _group_check_done = True

        # Handle multiple GPUs
        handled_multiple_gpus, device_handle = self.helpers.handle_gpus(args, self.logger, self.list)
        if handled_multiple_gpus:
            return # This function is recursive

        args.gpu = device_handle

        # Get gpu_id for logging
        gpu_id = self.helpers.get_gpu_id_from_device_handle(args.gpu)

        # Only fetch data if group check passed; otherwise force "N/A"
        if  _group_in_groups:
            try:
                bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(args.gpu)
            except amdsmi_exception.AmdSmiLibraryException as e:
                bdf = "N/A"
            try:
                uuid = amdsmi_interface.amdsmi_get_gpu_device_uuid(args.gpu)
            except amdsmi_exception.AmdSmiLibraryException:
                uuid = "N/A"
        else:
            # user not in render/video → everything is N/A
            bdf = "N/A"
            uuid = "N/A"

        try:
            kfd_info = amdsmi_interface.amdsmi_get_gpu_kfd_info(args.gpu)
            kfd_id = kfd_info['kfd_id']
            node_id = kfd_info['node_id']
            partition_id = kfd_info['current_partition_id']
        except amdsmi_exception.AmdSmiLibraryException as e:
            kfd_id = node_id = "N/A"
            logging.debug("Failed to get kfd info for gpu %s | %s", gpu_id, e.get_error_info())

        # CSV format is intentionally aligned with Host
        if self.logger.is_csv_format():
            self.logger.store_output(args.gpu, 'gpu_bdf', bdf)
            self.logger.store_output(args.gpu, 'gpu_uuid', uuid)
        else:
            self.logger.store_output(args.gpu, 'bdf', bdf)
            self.logger.store_output(args.gpu, 'uuid', uuid)

        self.logger.store_output(args.gpu, 'kfd_id', kfd_id)
        self.logger.store_output(args.gpu, 'node_id', node_id)
        self.logger.store_output(args.gpu, 'partition_id', partition_id)

        if args.e:
            try:
                enumeration_info = amdsmi_interface.amdsmi_get_gpu_enumeration_info(args.gpu)
            except amdsmi_exception.AmdSmiLibraryException:
                enumeration_info = {
                    "drm_render": "N/A",
                    "drm_card":   "N/A",
                    "hsa_id":     "N/A",
                    "hip_id":     "N/A",
                    "hip_uuid":   "N/A",
                }

            # __Override__ hip_uuid if the group check failed
            if not _group_in_groups:
               enumeration_info["hip_uuid"] = "N/A"
            # now store all the fields exactly once:
            if enumeration_info['drm_render'] == "N/A":
                self.logger.store_output(args.gpu, 'render', enumeration_info['drm_render'])
            else:
                self.logger.store_output(args.gpu, 'render',
                                         f"renderD{enumeration_info['drm_render']}")
            if enumeration_info['drm_card'] == "N/A":
                self.logger.store_output(args.gpu, 'card', enumeration_info['drm_card'])
            else:
                self.logger.store_output(args.gpu, 'card',
                                         f"card{enumeration_info['drm_card']}")
            self.logger.store_output(args.gpu, 'hsa_id', enumeration_info['hsa_id'])
            self.logger.store_output(args.gpu, 'hip_id', enumeration_info['hip_id'])
            self.logger.store_output(args.gpu, 'hip_uuid', enumeration_info['hip_uuid'])


        if multiple_devices:
            self.logger.store_multiple_device_output()
            return # Skip printing when there are multiple devices

        self.logger.print_output()


    def static_cpu(self, args, multiple_devices=False, cpu=None, interface_ver=None):
        """Get Static information for target cpu

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            cpu (device_handle, optional): device_handle for target device. Defaults to None.

        Returns:
            None: Print output via AMDSMILogger to destination
        """

        if cpu:
            args.cpu = cpu
        if interface_ver:
            args.interface_ver = interface_ver

        # Store cpu args that are applicable to the current platform
        curr_platform_cpu_args = ["smu", "interface_ver"]
        curr_platform_cpu_values = [args.smu, args.interface_ver]

        # If no cpu options are passed, return all available args
        if not any(curr_platform_cpu_values):
            for arg in curr_platform_cpu_args:
                setattr(args, arg, True)

        # Handle multiple CPUs
        handled_multiple_cpus, device_handle = self.helpers.handle_cpus(args,
                                                                        self.logger,
                                                                        self.static_cpu)
        if handled_multiple_cpus:
            return # This function is recursive
        args.cpu = device_handle

        # Get cpu id for logging
        cpu_id = self.helpers.get_cpu_id_from_device_handle(args.cpu)
        logging.debug(f"Static Arg information for CPU {cpu_id} on {self.helpers.os_info()}")

        static_dict = {}
        if self.logger.is_json_format():
            static_dict['cpu'] = int(cpu_id)

        if args.smu:
            try:
                smu = amdsmi_interface.amdsmi_get_cpu_smu_fw_version(args.cpu)
                static_dict["smu"] = {"FW_VERSION" : f"{smu['smu_fw_major_ver_num']}."
                                      f"{smu['smu_fw_minor_ver_num']}.{smu['smu_fw_debug_ver_num']}"}
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["smu"] = "N/A"
                logging.debug("Failed to get SMU FW for cpu %s | %s", cpu_id, e.get_error_info())

        if args.interface_ver:
            static_dict["interface_version"] = {}
            try:
                intf_ver = amdsmi_interface.amdsmi_get_cpu_hsmp_proto_ver(args.cpu)
                static_dict["interface_version"]["proto version"] = intf_ver
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["interface_version"]["proto version"] = "N/A"
                logging.debug("Failed to get proto version for cpu %s | %s", cpu_id, e.get_error_info())

        multiple_devices_csv_override = False
        if not self.logger.is_json_format():
            self.logger.store_cpu_output(args.cpu, 'values', static_dict)
        else:
            self.logger.store_cpu_json_output.append(static_dict)
        if multiple_devices:
            self.logger.store_multiple_device_output()
            return # Skip printing when there are multiple devices
        if not self.logger.is_json_format():
            self.logger.print_output(multiple_device_enabled=multiple_devices_csv_override)


    def static_gpu(self, args, multiple_devices=False, gpu=None, asic=None, bus=None, vbios=None,
                        limit=None, driver=None, ras=None, board=None, numa=None, vram=None,
                        cache=None, partition=None, dfc_ucode=None, fb_info=None, num_vf=None,
                        soc_pstate=None, xgmi_plpd=None, process_isolation=None, clock=None):
        """Get Static information for target gpu

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            current_platform_args (list): gpu supported platform arguments
            current_platform_values (list): gpu supported platform values for each argument
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            gpu (device_handle, optional): device_handle for target device. Defaults to None.
            asic (bool, optional): Value override for args.asic. Defaults to None.
            bus (bool, optional): Value override for args.bus. Defaults to None.
            vbios (bool, optional): Value override for args.vbios. Defaults to None.
            limit (bool, optional): Value override for args.limit. Defaults to None.
            driver (bool, optional): Value override for args.driver. Defaults to None.
            ras (bool, optional): Value override for args.ras. Defaults to None.
            board (bool, optional): Value override for args.board. Defaults to None.
            numa (bool, optional): Value override for args.numa. Defaults to None.
            vram (bool, optional): Value override for args.vram. Defaults to None.
            cache (bool, optional): Value override for args.cache. Defaults to None.
            partition (bool, optional): Value override for args.partition. Defaults to None.
            dfc_ucode (bool, optional): Value override for args.dfc_ucode. Defaults to None.
            fb_info (bool, optional): Value override for args.fb_info. Defaults to None.
            num_vf (bool, optional): Value override for args.num_vf. Defaults to None.
            soc_pstate (bool, optional): Value override for args.soc_pstate. Defaults to None.
            xgmi_plpd (bool, optional): Value override for args.xgmi_plpd. Defaults to None.
            process_isolation (bool, optional): Value override for args.process_isolation. Defaults to None.
        Returns:
            None: Print output via AMDSMILogger to destination
        """

        if gpu:
            args.gpu = gpu
        if asic:
            args.asic = asic
        if bus:
            args.bus = bus
        if vbios:
            args.vbios = vbios
        if board:
            args.board = board
        if driver:
            args.driver = driver
        if ras:
            args.ras = ras
        if vram:
            args.vram = vram
        if cache:
            args.cache = cache
        if process_isolation:
            args.process_isolation = process_isolation
        if partition:
            args.partition = partition
        if clock:
            args.clock = clock
        # args.clock defaults to False so if it was overwritten to empty list, that indicates that it was given as an arguments but with an empty list
        if args.clock == []:
            args.clock = True

        # Store args that are applicable to the current platform (default arguments)
        current_platform_args = ["asic", "bus", "vbios", "driver", "ras",
                                 "vram", "cache", "board", "process_isolation",
                                 "clock"]
        current_platform_values = [args.asic, args.bus, args.vbios, args.driver, args.ras,
                                   args.vram, args.cache, args.board, args.process_isolation,
                                   args.clock]

        # amd-smi static default arguments:
        # Exclude args that are not applicable to the current platform,
        # but allow output if argument is passed.
        #
        # Note: Partition is a special case, it is no longer an amd-smi static
        # default argument.
        # Reason: Reading current_compute_partition may momentarily wake the
        #         GPU up. This is due to reading XCD registers, which is expected
        #         behavior. Changing partitions is not a trivial operation,
        #         current_compute_partition SYSFS controls this action.
        if args.partition:
            current_platform_args += ["partition"]
            current_platform_values += [args.partition]

        if not self.group_check_printed:
            self.helpers.check_required_groups()
            self.group_check_printed = True

        if self.helpers.is_linux() and self.helpers.is_baremetal():
            if limit:
                args.limit = limit
            if soc_pstate:
                args.soc_pstate = soc_pstate
            if xgmi_plpd:
                args.xgmi_plpd = xgmi_plpd
            current_platform_args += ["ras", "limit", "soc_pstate", "xgmi_plpd"]
            current_platform_values += [args.ras, args.limit, args.soc_pstate, args.xgmi_plpd]

        if self.helpers.is_linux() and not self.helpers.is_virtual_os():
            if numa:
                args.numa = numa
            current_platform_args += ["numa"]
            current_platform_values += [args.numa]

        if self.helpers.is_hypervisor():
            if dfc_ucode:
                args.dfc_ucode = dfc_ucode
            if fb_info:
                args.fb_info = fb_info
            if num_vf:
                args.num_vf = num_vf
            current_platform_args += ["dfc_ucode", "fb_info", "num_vf"]
            current_platform_values += [args.dfc_ucode, args.fb_info, args.num_vf]

        if not any(current_platform_values):
            for arg in current_platform_args:
                setattr(args, arg, True)

        handled_multiple_gpus, device_handle = self.helpers.handle_gpus(args, self.logger, self.static_gpu)
        if handled_multiple_gpus:
            return # This function is recursive
        args.gpu = device_handle
        # Get gpu_id for logging
        gpu_id = self.helpers.get_gpu_id_from_device_handle(args.gpu)

        logging.debug("=====================================================================")
        logging.debug(f"Static Arg information for GPU {gpu_id} on {self.helpers.os_info()}")
        logging.debug(f"Function args:           {args}")
        logging.debug(f"Current platform args:   {current_platform_args}")
        logging.debug(f"Current platform values: {current_platform_values}")
        logging.debug("=====================================================================")

        # Populate static dictionary for each enabled argument
        static_dict = {}
        if self.logger.is_json_format():
            static_dict['gpu'] = int(gpu_id)
        if args.asic:
            asic_dict = {
                "market_name" : "N/A",
                "vendor_id" : "N/A",
                "vendor_name" : "N/A",
                "subvendor_id" : "N/A",
                "device_id" : "N/A",
                "subsystem_id" : "N/A",
                "rev_id" : "N/A",
                "asic_serial" : "N/A",
                "oam_id" : "N/A",
                "num_compute_units" : "N/A",
                "target_graphics_version" : "N/A"
            }

            try:
                asic_info = amdsmi_interface.amdsmi_get_gpu_asic_info(args.gpu)
                for key, value in asic_info.items():
                    asic_dict[key] = value
            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug("Failed to get asic info for gpu %s | %s", gpu_id, e.get_error_info())

            static_dict['asic'] = asic_dict
        if args.bus:
            bus_info = {
                'bdf': "N/A",
                'max_pcie_width': "N/A",
                'max_pcie_speed': "N/A",
                'pcie_interface_version': "N/A",
                'slot_type': "N/A"
            }

            try:
                bus_info['bdf'] = amdsmi_interface.amdsmi_get_gpu_device_bdf(args.gpu)
            except amdsmi_exception.AmdSmiLibraryException as e:
                bus_info['bdf'] = "N/A"
                logging.debug("Failed to get bdf for gpu %s | %s", gpu_id, e.get_error_info())

            try:
                pcie_static = amdsmi_interface.amdsmi_get_pcie_info(args.gpu)['pcie_static']
                bus_info['max_pcie_width'] = pcie_static['max_pcie_width']
                bus_info['max_pcie_speed'] = pcie_static['max_pcie_speed']
                bus_info['pcie_interface_version'] = pcie_static['pcie_interface_version']
                bus_info['slot_type'] = pcie_static['slot_type']
                if bus_info['max_pcie_speed'] % 1000 != 0:
                    pcie_speed_GTs_value = round(bus_info['max_pcie_speed'] / 1000, 1)
                else:
                    pcie_speed_GTs_value = round(bus_info['max_pcie_speed'] / 1000)

                bus_info['max_pcie_speed'] = pcie_speed_GTs_value

                if bus_info['pcie_interface_version'] > 0:
                    bus_info['pcie_interface_version'] = f"Gen {bus_info['pcie_interface_version']}"

                # Set the unit for pcie_speed
                pcie_speed_unit ='GT/s'
                if self.logger.is_human_readable_format():
                    bus_info['max_pcie_speed'] = f"{bus_info['max_pcie_speed']} {pcie_speed_unit}"

                if self.logger.is_json_format():
                    bus_info['max_pcie_speed'] = {"value" : bus_info['max_pcie_speed'],
                                                  "unit" : pcie_speed_unit}

            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug("Failed to get bus info for gpu %s | %s", gpu_id, e.get_error_info())

            static_dict['bus'] = bus_info
        if args.vbios:
            try:
                vbios_info = amdsmi_interface.amdsmi_get_gpu_vbios_info(args.gpu)
                for key, value in vbios_info.items():
                    if isinstance(value, str):
                        if value.strip() == '':
                            vbios_info[key] = "N/A"
                static_dict['ifwi'] = vbios_info
                # Remove boot_firmware since it's not used
                del static_dict['ifwi']['boot_firmware']
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict['ifwi'] = "N/A"
                logging.debug("Failed to get vbios/ifwi info for gpu %s | %s", gpu_id, e.get_error_info())
        if 'limit' in current_platform_args:
            if args.limit:
                # Power limits
                try:
                    power_limit_error = False
                    power_cap_info = amdsmi_interface.amdsmi_get_power_cap_info(args.gpu)
                    max_power_limit = power_cap_info['max_power_cap']
                    max_power_limit = self.helpers.convert_SI_unit(max_power_limit, AMDSMIHelpers.SI_Unit.MICRO)
                    min_power_limit = power_cap_info['min_power_cap']
                    min_power_limit = self.helpers.convert_SI_unit(min_power_limit, AMDSMIHelpers.SI_Unit.MICRO)
                    socket_power_limit = power_cap_info['power_cap']
                    socket_power_limit = self.helpers.convert_SI_unit(socket_power_limit, AMDSMIHelpers.SI_Unit.MICRO)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    power_limit_error = True
                    max_power_limit = "N/A"
                    min_power_limit = "N/A"
                    socket_power_limit = "N/A"
                    logging.debug("Failed to get power cap info for gpu %s | %s", gpu_id, e.get_error_info())

                # Edge temperature limits
                try:
                    slowdown_temp_edge_limit_error = False
                    slowdown_temp_edge_limit = amdsmi_interface.amdsmi_get_temp_metric(args.gpu,
                        amdsmi_interface.AmdSmiTemperatureType.EDGE, amdsmi_interface.AmdSmiTemperatureMetric.CRITICAL)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    slowdown_temp_edge_limit_error = True
                    slowdown_temp_edge_limit = "N/A"
                    logging.debug("Failed to get edge temperature slowdown metric for gpu %s | %s", gpu_id, e.get_error_info())

                if slowdown_temp_edge_limit == 0:
                    slowdown_temp_edge_limit_error = True
                    slowdown_temp_edge_limit = "N/A"

                try:
                    shutdown_temp_edge_limit_error = False
                    shutdown_temp_edge_limit = amdsmi_interface.amdsmi_get_temp_metric(args.gpu,
                        amdsmi_interface.AmdSmiTemperatureType.EDGE, amdsmi_interface.AmdSmiTemperatureMetric.EMERGENCY)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    shutdown_temp_edge_limit_error = True
                    shutdown_temp_edge_limit = "N/A"
                    logging.debug("Failed to get edge temperature shutdown metrics for gpu %s | %s", gpu_id, e.get_error_info())

                if shutdown_temp_edge_limit == 0:
                    shutdown_temp_edge_limit_error = True
                    shutdown_temp_edge_limit = "N/A"

                # Hotspot/Junction temperature limits
                try:
                    slowdown_temp_hotspot_limit_error = False
                    slowdown_temp_hotspot_limit = amdsmi_interface.amdsmi_get_temp_metric(args.gpu,
                        amdsmi_interface.AmdSmiTemperatureType.HOTSPOT, amdsmi_interface.AmdSmiTemperatureMetric.CRITICAL)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    slowdown_temp_hotspot_limit_error = True
                    slowdown_temp_hotspot_limit = "N/A"
                    logging.debug("Failed to get hotspot temperature slowdown metrics for gpu %s | %s", gpu_id, e.get_error_info())

                try:
                    shutdown_temp_hotspot_limit_error = False
                    shutdown_temp_hotspot_limit = amdsmi_interface.amdsmi_get_temp_metric(args.gpu,
                        amdsmi_interface.AmdSmiTemperatureType.HOTSPOT, amdsmi_interface.AmdSmiTemperatureMetric.EMERGENCY)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    shutdown_temp_hotspot_limit_error = True
                    shutdown_temp_hotspot_limit = "N/A"
                    logging.debug("Failed to get hotspot temperature shutdown metrics for gpu %s | %s", gpu_id, e.get_error_info())


                # VRAM temperature limits
                try:
                    slowdown_temp_vram_limit_error = False
                    slowdown_temp_vram_limit = amdsmi_interface.amdsmi_get_temp_metric(args.gpu,
                        amdsmi_interface.AmdSmiTemperatureType.VRAM, amdsmi_interface.AmdSmiTemperatureMetric.CRITICAL)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    slowdown_temp_vram_limit_error = True
                    slowdown_temp_vram_limit = "N/A"
                    logging.debug("Failed to get vram temperature slowdown metrics for gpu %s | %s", gpu_id, e.get_error_info())

                try:
                    shutdown_temp_vram_limit_error = False
                    shutdown_temp_vram_limit = amdsmi_interface.amdsmi_get_temp_metric(args.gpu,
                        amdsmi_interface.AmdSmiTemperatureType.VRAM, amdsmi_interface.AmdSmiTemperatureMetric.EMERGENCY)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    shutdown_temp_vram_limit_error = True
                    shutdown_temp_vram_limit = "N/A"
                    logging.debug("Failed to get vram temperature shutdown metrics for gpu %s | %s", gpu_id, e.get_error_info())


                # Assign units
                power_unit = 'W'
                temp_unit_human_readable = '\N{DEGREE SIGN}C'
                temp_unit_json = 'C'
                if not power_limit_error:
                    max_power_limit = self.helpers.unit_format(self.logger,
                                                               max_power_limit,
                                                               power_unit)
                    min_power_limit = self.helpers.unit_format(self.logger,
                                                               min_power_limit,
                                                               power_unit)
                    socket_power_limit = self.helpers.unit_format(self.logger,
                                                                  socket_power_limit,
                                                                  power_unit)

                if self.logger.is_human_readable_format():
                    if not slowdown_temp_edge_limit_error:
                        slowdown_temp_edge_limit = f"{slowdown_temp_edge_limit} {temp_unit_human_readable}"
                    if not slowdown_temp_hotspot_limit_error:
                        slowdown_temp_hotspot_limit = f"{slowdown_temp_hotspot_limit} {temp_unit_human_readable}"
                    if not slowdown_temp_vram_limit_error:
                        slowdown_temp_vram_limit = f"{slowdown_temp_vram_limit} {temp_unit_human_readable}"
                    if not shutdown_temp_edge_limit_error:
                        shutdown_temp_edge_limit = f"{shutdown_temp_edge_limit} {temp_unit_human_readable}"
                    if not shutdown_temp_hotspot_limit_error:
                        shutdown_temp_hotspot_limit = f"{shutdown_temp_hotspot_limit} {temp_unit_human_readable}"
                    if not shutdown_temp_vram_limit_error:
                        shutdown_temp_vram_limit = f"{shutdown_temp_vram_limit} {temp_unit_human_readable}"

                if self.logger.is_json_format():
                    if not slowdown_temp_edge_limit_error:
                        slowdown_temp_edge_limit = {"value" : slowdown_temp_edge_limit,
                                                    "unit" : temp_unit_json}
                    if not slowdown_temp_hotspot_limit_error:
                        slowdown_temp_hotspot_limit = {"value" : slowdown_temp_hotspot_limit,
                                                       "unit" : temp_unit_json}
                    if not slowdown_temp_vram_limit_error:
                        slowdown_temp_vram_limit = {"value" : slowdown_temp_vram_limit,
                                                    "unit" : temp_unit_json}
                    if not shutdown_temp_edge_limit_error:
                        shutdown_temp_edge_limit = {"value" : shutdown_temp_edge_limit,
                                                    "unit" : temp_unit_json}
                    if not shutdown_temp_hotspot_limit_error:
                        shutdown_temp_hotspot_limit = {"value" : shutdown_temp_hotspot_limit,
                                                       "unit" : temp_unit_json}
                    if not shutdown_temp_vram_limit_error:
                        shutdown_temp_vram_limit = {"value" : shutdown_temp_vram_limit,
                                                    "unit" : temp_unit_json}

                limit_info = {}
                # Power limits
                limit_info['max_power'] = max_power_limit
                limit_info['min_power'] = min_power_limit
                limit_info['socket_power'] = socket_power_limit

                # Shutdown limits
                limit_info['slowdown_edge_temperature'] = slowdown_temp_edge_limit
                limit_info['slowdown_hotspot_temperature'] = slowdown_temp_hotspot_limit
                limit_info['slowdown_vram_temperature'] = slowdown_temp_vram_limit
                limit_info['shutdown_edge_temperature'] = shutdown_temp_edge_limit
                limit_info['shutdown_hotspot_temperature'] = shutdown_temp_hotspot_limit
                limit_info['shutdown_vram_temperature'] = shutdown_temp_vram_limit
                static_dict['limit'] = limit_info
        if args.driver:
            driver_info_dict = {"name" : "N/A",
                                "version" : "N/A"}

            try:
                driver_info = amdsmi_interface.amdsmi_get_gpu_driver_info(args.gpu)
                driver_info_dict["name"] = driver_info["driver_name"]
                driver_info_dict["version"] = driver_info["driver_version"]
            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug("Failed to get driver info for gpu %s | %s", gpu_id, e.get_error_info())

            static_dict['driver'] = driver_info_dict
        if args.board:
            static_dict['board'] = {"model_number": "N/A",
                                    "product_serial": "N/A",
                                    "fru_id": "N/A",
                                    "product_name": "N/A",
                                    "manufacturer_name": "N/A"}
            try:
                board_info = amdsmi_interface.amdsmi_get_gpu_board_info(args.gpu)
                for key, value in board_info.items():
                    if isinstance(value, str):
                        if value.strip() == '':
                            board_info[key] = "N/A"
                static_dict['board'] = board_info
            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug("Failed to get board info for gpu %s | %s", gpu_id, e.get_error_info())
        if 'ras' in current_platform_args:
            if args.ras:
                ras_dict = {"eeprom_version": "N/A",
                            "bad_page_threshold": "N/A",
                            "bad_page_threshold_exceeded": "N/A",
                            "parity_schema" : "N/A",
                            "single_bit_schema" : "N/A",
                            "double_bit_schema" : "N/A",
                            "poison_schema" : "N/A",
                            "ecc_block_state": "N/A"}

                try:
                    ras_info = amdsmi_interface.amdsmi_get_gpu_ras_feature_info(args.gpu)
                    for key, value in ras_info.items():
                        if isinstance(value, int):
                            if value == 65535:
                                logging.debug(f"Failed to get ras {key} for gpu {gpu_id}")
                                ras_info[key] = "N/A"
                                continue
                        if key != "eeprom_version":
                            if value:
                                ras_info[key] = "ENABLED"
                            else:
                                ras_info[key] = "DISABLED"

                    ras_dict.update(ras_info)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug("Failed to get ras info for gpu %s | %s", gpu_id, e.get_error_info())
                try:
                    ras_dict["bad_page_threshold"] = amdsmi_interface.amdsmi_get_gpu_bad_page_threshold(args.gpu)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug("Failed to get bad page threshold count for gpu %s | %s", gpu_id, e.get_error_info())
                try:
                    bad_page_info = amdsmi_interface.amdsmi_get_gpu_bad_page_info(args.gpu)
                    retired_pages = 0
                    if bad_page_info:
                        for bad_page in bad_page_info:
                            if bad_page["status"] == amdsmi_interface.AmdSmiMemoryPageStatus.RESERVED:
                                retired_pages += 1
                    # default to N/A
                    ras_dict["bad_page_threshold_exceeded"] = "N/A"
                    # If this is an int, then default to False
                    if isinstance(ras_dict["bad_page_threshold"], int):
                        ras_dict["bad_page_threshold_exceeded"] = "False"
                        if retired_pages > ras_dict["bad_page_threshold"]:
                            # If there are more retired pages then set to True
                            ras_dict["bad_page_threshold_exceeded"] = "True"
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug("Failed to get retired pages count for gpu %s | %s", gpu_id, e.get_error_info())

                try:
                    ras_states = amdsmi_interface.amdsmi_get_gpu_ras_block_features_enabled(args.gpu)
                    ecc_block_state_dict = {}
                    for state in ras_states:
                        ecc_block_state_dict[state["block"]] = state["status"]
                    ras_dict["ecc_block_state"] = ecc_block_state_dict
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug("Failed to get ras block features for gpu %s | %s", gpu_id, e.get_error_info())

                static_dict["ras"] = ras_dict
        if args.partition:
            try:
                compute_partition = amdsmi_interface.amdsmi_get_gpu_compute_partition(args.gpu)
            except amdsmi_exception.AmdSmiLibraryException as e:
                compute_partition = "N/A"
                logging.debug("Failed to get compute partition info for gpu %s | %s", gpu_id, e.get_error_info())
            try:
                memory_partition = amdsmi_interface.amdsmi_get_gpu_memory_partition(args.gpu)
            except amdsmi_exception.AmdSmiLibraryException as e:
                memory_partition = "N/A"
                logging.debug("Failed to get memory partition info for gpu %s | %s", gpu_id, e.get_error_info())
            try:
                kfd_info = amdsmi_interface.amdsmi_get_gpu_kfd_info(args.gpu)
                partition_id = kfd_info['current_partition_id']
            except amdsmi_exception.AmdSmiLibraryException as e:
                partition_id = "N/A"
                logging.debug("Failed to get partition ID for gpu %s | %s", gpu_id, e.get_error_info())
            static_dict['partition'] = {"accelerator_partition": compute_partition,
                                        "memory_partition": memory_partition,
                                        "partition_id": partition_id}
        if 'soc_pstate' in current_platform_args:
            if args.soc_pstate:
                try:
                    policy_info = amdsmi_interface.amdsmi_get_soc_pstate(args.gpu)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    policy_info = "N/A"
                    logging.debug("Failed to get soc pstate policy info for gpu %s | %s", gpu_id, e.get_error_info())

                static_dict['soc_pstate'] = policy_info
        if 'xgmi_plpd' in current_platform_args:
            if args.xgmi_plpd:
                try:
                    policy_info = amdsmi_interface.amdsmi_get_xgmi_plpd(args.gpu)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    policy_info = "N/A"
                    logging.debug("Failed to get xgmi_plpd info for gpu %s | %s", gpu_id, e.get_error_info())

                static_dict['xgmi_plpd'] = policy_info
        if 'process_isolation' in current_platform_args:
            if args.process_isolation:
                try:
                    status = amdsmi_interface.amdsmi_get_gpu_process_isolation(args.gpu)
                    status = "Enabled" if status else "Disabled"
                except amdsmi_exception.AmdSmiLibraryException as e:
                    status = "N/A"
                    logging.debug("Failed to process isolation for gpu %s | %s", gpu_id, e.get_error_info())

                static_dict['process_isolation'] = status
        if 'numa' in current_platform_args:
            if args.numa:
                try:
                    numa_node_number = amdsmi_interface.amdsmi_topo_get_numa_node_number(args.gpu)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    numa_node_number = "N/A"
                    logging.debug("Failed to get numa node number for gpu %s | %s", gpu_id, e.get_error_info())

                try:
                    numa_affinity = amdsmi_interface.amdsmi_get_gpu_topo_numa_affinity(args.gpu)
                    # -1 means No numa node is assigned to the GPU, so there is no numa affinity
                    if self.logger.is_human_readable_format() and numa_affinity == -1:
                        numa_affinity = "NONE"
                except amdsmi_exception.AmdSmiLibraryException as e:
                    numa_affinity = "N/A"
                    logging.debug("Failed to get numa affinity for gpu %s | %s", gpu_id, e.get_error_info())

                try:
                    cpu_set = amdsmi_interface.amdsmi_get_cpu_affinity_with_scope(args.gpu, amdsmi_interface.AmdSmiAffinityScope.NUMA_SCOPE)
                    cpu_set = [f"{cpus:016X}" for cpus in cpu_set]
                    cpu_set = {f'cpu_list_{i}': f"{cpus}" for i, cpus in enumerate(cpu_set)}
                    bitmask_ranges = self.helpers.get_bitmask_ranges(cpu_set)
                    cpu_affinity = {}

                    for key in cpu_set:
                        cpu_affinity[key] = {
                            "bitmask": cpu_set[key],
                            "cpu_cores_affinity" : bitmask_ranges[key]
                        }

                except amdsmi_exception.AmdSmiLibraryException as e:
                    cpu_affinity = "N/A"
                    logging.debug("Failed to get cpu affinity for gpu %s | %s", gpu_id, e.get_error_info())

                try:
                    cpusockets = amdsmi_interface.amdsmi_get_cpu_affinity_with_scope(args.gpu, amdsmi_interface.AmdSmiAffinityScope.SOCKET_SCOPE)
                    cpusockets = {f'socket_{i}': socket for i, socket in enumerate(set(cpusockets))}
                except amdsmi_exception.AmdSmiLibraryException as e:
                    cpusockets = {}
                    logging.debug("Failed to get socket affinity for gpu %s | %s", gpu_id, e.get_error_info())

                static_dict['numa'] = { 'node' : numa_node_number,
                                        'affinity' : numa_affinity,
                                        'cpu_affinity' : cpu_affinity,
                                        'socket_affinity' : cpusockets if cpusockets else "N/A"}
        if args.vram:
            vram_info_dict = {"type" : "N/A",
                              "vendor" : "N/A",
                              "size" : "N/A",
                              "bit_width" : "N/A",
                              "max_bandwidth" : "N/A"}
            try:
                vram_info = amdsmi_interface.amdsmi_get_gpu_vram_info(args.gpu)

                # Get vram type string
                vram_type_enum = vram_info['vram_type']
                if vram_type_enum == amdsmi_interface.amdsmi_wrapper.AMDSMI_VRAM_TYPE__MAX:
                    vram_type = "GDDR7"
                else:
                    vram_type = amdsmi_interface.amdsmi_wrapper.amdsmi_vram_type_t__enumvalues[vram_type_enum]
                    # Remove amdsmi enum prefix
                    vram_type = vram_type.replace('AMDSMI_VRAM_TYPE_', '').replace('_', '')

                # Get vram vendor string
                vram_vendor = vram_info['vram_vendor']
                if "PLACEHOLDER" in vram_vendor:
                    vram_vendor = "N/A"

                # Assign cleaned values to vram_info_dict
                vram_info_dict['type'] = vram_type
                vram_info_dict['vendor'] = vram_vendor

                # Populate vram size with unit
                vram_info_dict['size'] = vram_info['vram_size']
                vram_size_unit = "MB"
                if self.logger.is_human_readable_format():
                    vram_info_dict['size'] = f"{vram_info['vram_size']} {vram_size_unit}"

                if self.logger.is_json_format():
                    vram_info_dict['size'] = {"value" : vram_info['vram_size'],
                                              "unit" : vram_size_unit}

                # Populate bit width
                vram_info_dict['bit_width'] = vram_info['vram_bit_width']

                # Populate vram_max_bandwidth
                vram_max_bw = vram_info['vram_max_bandwidth']
                vram_max_bw_unit = 'GB/s'
                if self.logger.is_human_readable_format():
                    vram_info_dict["max_bandwidth"] = f"{vram_max_bw} {vram_max_bw_unit if vram_max_bw != 'N/A' else ''}"
                if self.logger.is_json_format():
                    vram_info_dict["max_bandwidth"] = {"value" : vram_max_bw,
                                                     "unit" : vram_max_bw_unit}

            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug("Failed to get vram info for gpu %s | %s", gpu_id, e.get_error_info())

            static_dict['vram'] = vram_info_dict
        if args.cache:
            try:
                cache_info_list = amdsmi_interface.amdsmi_get_gpu_cache_info(args.gpu)['cache']
                logging.debug(f"cache_info dictionary = {cache_info_list}")

                for index, cache_info in enumerate(cache_info_list):
                    new_cache_info = {"cache" : index}
                    new_cache_info.update(cache_info)
                    cache_info_list[index] = new_cache_info

                logging.debug(f"[after update] cache_info_list = {cache_info_list}")

                cache_size_unit = "KB"
                if self.logger.is_human_readable_format():
                    cache_info_dict_format = {}
                    for cache_dict in cache_info_list:
                        cache_index = "cache_" + str(cache_dict["cache"])
                        cache_info_dict_format[cache_index] = cache_dict

                        # Remove cache index from new dictionary
                        cache_info_dict_format[cache_index].pop("cache")

                        # Add cache_size unit
                        cache_size = f"{cache_info_dict_format[cache_index]['cache_size']} {cache_size_unit}"
                        cache_info_dict_format[cache_index]["cache_size"] = cache_size

                        # take cache_properties out of list -> display as string, removing brackets
                        cache_info_dict_format[cache_index]["cache_properties"] = ", ".join(cache_info_dict_format[cache_index]["cache_properties"])

                    cache_info_list = cache_info_dict_format
                    logging.debug(f"[human readable] cache_info_list = {cache_info_list}")

                # Add cache_size_unit to json output
                if self.logger.is_json_format():
                    for cache_dict in cache_info_list:
                        cache_dict["cache_size"] = {"value" : cache_dict["cache_size"],
                                                    "unit" : cache_size_unit}
            except amdsmi_exception.AmdSmiLibraryException as e:
                cache_info_list = "N/A"
                logging.debug("Failed to get cache info for gpu %s | %s", gpu_id, e.get_error_info())

            static_dict['cache_info'] = cache_info_list

        # default to printing all clocks, if in current_platform_args; otherwise print specific clocks
        if 'clock' in current_platform_args and (args.clock == True or isinstance(args.clock, list)):
            original_clock_args = args.clock  #save original args.clock value, so we can reset for multiple devices
            if isinstance(args.clock, bool):
                args.clock = ['sys', 'mem', 'df', 'soc', 'dcef', 'vclk0', 'vclk1', 'dclk0', 'dclk1']

            if isinstance(args.clock, list):
                # remove potential duplicates from list
                args.clock = list(set(args.clock))
                # check that clock is valid option
                if "all" in args.clock or len(args.clock) == 0:
                    args.clock = ['sys', 'mem', 'df', 'soc', 'dcef', 'vclk0', 'vclk1', 'dclk0', 'dclk1']

                clk_dict = {
                    'sys': "N/A",
                    'mem': "N/A",
                    'df': "N/A",
                    'soc': "N/A",
                    'dcef': "N/A",
                    'vclk0': "N/A",
                    'vclk1': "N/A",
                    'dclk0': "N/A",
                    'dclk1': "N/A",
                }
                for clk in list(clk_dict.keys()):
                    if clk not in args.clock:
                        del clk_dict[clk]

                for clk in args.clock:
                    clk_type = clk.lower()
                    if clk_type == "sys":
                        clk_type_conversion = amdsmi_interface.AmdSmiClkType.SYS
                    elif clk_type == "mem":
                        clk_type_conversion = amdsmi_interface.AmdSmiClkType.MEM
                    elif clk_type == "df":
                        clk_type_conversion = amdsmi_interface.AmdSmiClkType.DF
                    elif clk_type == "soc":
                        clk_type_conversion = amdsmi_interface.AmdSmiClkType.SOC
                    elif clk_type == "dcef":
                        clk_type_conversion = amdsmi_interface.AmdSmiClkType.DCEF
                    # vclk and dclk currently do not support levels so average clk is given for frequency levels
                    elif clk_type == "vclk0":
                        clk_type_conversion = amdsmi_interface.AmdSmiClkType.VCLK0
                    elif clk_type == "vclk1":
                        clk_type_conversion = amdsmi_interface.AmdSmiClkType.VCLK1
                    elif clk_type == "dclk0":
                        clk_type_conversion = amdsmi_interface.AmdSmiClkType.DCLK0
                    elif clk_type == "dclk1":
                        clk_type_conversion = amdsmi_interface.AmdSmiClkType.DCLK1
                    else:
                        clk_type_conversion = "N/A"
                        output_format = self.helpers.get_output_format()
                        raise AmdSmiInvalidParameterException('static', clk_type, output_format) # clk type given is bad

                    try:
                        frequencies = amdsmi_interface.amdsmi_get_clk_freq(args.gpu, clk_type_conversion)
                        freq_dict = {}
                        freq_dict.update({'current level':frequencies['current']})
                        freq_dict.update({'frequency_levels':{}})
                        if frequencies["num_supported"] != 0:
                            for level in range(len(frequencies['frequency'])):
                                if frequencies['frequency'][level] != "N/A":
                                    freq = str(self.helpers.convert_SI_unit(frequencies['frequency'][level], AMDSMIHelpers.SI_Unit.MICRO)) + " MHz"
                                    freq_dict['frequency_levels'].update({f"Level {level}":freq})
                                else:
                                    freq_dict['frequency_levels'].update({f"Level {level}":"N/A"})
                        else:
                            freq_dict = "N/A"
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        freq_dict = "N/A"
                    clk_dict[clk] = freq_dict

                static_dict['clock'] = clk_dict
            else:
                raise amdsmi_exception.AmdSmiParameterException(args.clock, 'list[str]')
            # if original_clock_args is a boolean, set it back to the original value
            if isinstance(original_clock_args, bool):
                args.clock = original_clock_args

        # Convert and store output by pid for csv format
        multiple_devices_csv_override = False
        if self.logger.is_csv_format():
            # expand if ras blocks are populated
            if self.helpers.is_linux() and self.helpers.is_baremetal() and args.ras:
                if isinstance(static_dict['ras']['ecc_block_state'], list):
                    ecc_block_dicts = static_dict['ras'].pop('ecc_block_state')
                    multiple_devices_csv_override = True
                    for ecc_block_dict in ecc_block_dicts:
                        for key, value in ecc_block_dict.items():
                            self.logger.store_output(args.gpu, key, value)
                        self.logger.store_output(args.gpu, 'values', static_dict)
                        self.logger.store_gpu_json_output.append(static_dict)
                        self.logger.store_multiple_device_output()
                else:
                    # Store values if ras has an error
                    self.logger.store_output(args.gpu, 'values', static_dict)
                    self.logger.store_gpu_json_output.append(static_dict)
                if self.helpers.is_linux() and self.helpers.is_virtual_os():
                    self.logger.store_output(args.gpu, 'values', static_dict)
                    self.logger.store_gpu_json_output.append(static_dict)
            else:
                self.logger.store_output(args.gpu, 'values', static_dict)
                self.logger.store_gpu_json_output.append(static_dict)
        elif self.logger.is_json_format():
            self.logger.store_gpu_json_output.append(static_dict)
        else:
            # Store values in logger.output
            self.logger.store_output(args.gpu, 'values', static_dict)

        if multiple_devices:
            self.logger.store_multiple_device_output()
            return # Skip printing when there are multiple devices
        if not self.logger.is_json_format():
            self.logger.print_output(multiple_device_enabled=multiple_devices_csv_override)


    def static(self, args, multiple_devices=False, gpu=None, asic=None,
                bus=None, vbios=None, limit=None, driver=None, ras=None,
                board=None, numa=None, vram=None, cache=None, partition=None,
                dfc_ucode=None, fb_info=None, num_vf=None, cpu=None,
                interface_ver=None, soc_pstate=None, xgmi_plpd = None, process_isolation=None,
                clock=None):
        """Get Static information for target gpu and cpu

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            gpu (device_handle, optional): device_handle for target device. Defaults to None.
            asic (bool, optional): Value override for args.asic. Defaults to None.
            bus (bool, optional): Value override for args.bus. Defaults to None.
            vbios (bool, optional): Value override for args.vbios. Defaults to None.
            limit (bool, optional): Value override for args.limit. Defaults to None.
            driver (bool, optional): Value override for args.driver. Defaults to None.
            ras (bool, optional): Value override for args.ras. Defaults to None.
            board (bool, optional): Value override for args.board. Defaults to None.
            numa (bool, optional): Value override for args.numa. Defaults to None.
            vram (bool, optional): Value override for args.vram. Defaults to None.
            cache (bool, optional): Value override for args.cache. Defaults to None.
            partition (bool, optional): Value override for args.partition. Defaults to None.
            dfc_ucode (bool, optional): Value override for args.dfc_ucode. Defaults to None.
            fb_info (bool, optional): Value override for args.fb_info. Defaults to None.
            num_vf (bool, optional): Value override for args.num_vf. Defaults to None.
            cpu (cpu_handle, optional): cpu_handle for target device. Defaults to None.
            interface_ver (bool, optional): Value override for args.interface_ver. Defaults to None
            soc_pstate (bool, optional): Value override for args.soc_pstate. Defaults to None.
            xgmi_plpd (bool, optional): Value override for args.xgmi_plpd. Defaults to None.
            process_isolation (bool, optional): Value override for args.process_isolation. Defaults to None.
        Raises:
            IndexError: Index error if gpu list is empty

        Returns:
            None: Print output via AMDSMILogger to destination
        """
        # Mutually exclusive arguments
        if cpu:
            args.cpu = cpu
        if gpu:
            args.gpu = gpu

        # Check if a CPU argument has been set
        cpu_args_enabled = False
        cpu_attributes = ["smu", "interface_ver"]
        for attr in cpu_attributes:
            if hasattr(args, attr):
                if getattr(args, attr):
                    cpu_args_enabled = True
                    break

        # Check if a GPU argument has been set
        gpu_args_enabled = False
        gpu_attributes = ["asic", "bus", "vbios", "limit", "driver", "ras",
                          "board", "numa", "vram", "cache", "partition",
                          "dfc_ucode", "fb_info", "num_vf", "soc_pstate", "xgmi_plpd",
                          "process_isolation", "clock"]
        for attr in gpu_attributes:
            if hasattr(args, attr):
                if getattr(args, attr):
                    gpu_args_enabled = True
                    break

        # Handle CPU and GPU intialization cases
        if self.helpers.is_amd_hsmp_initialized() and self.helpers.is_amdgpu_initialized():
            # Print out all CPU and all GPU static info only if no device was specified.
            # If a GPU or CPU argument is provided only print out the specified device.
            if args.cpu == None and args.gpu == None:
                if not cpu_args_enabled and not gpu_args_enabled:
                    args.cpu = self.cpu_handles
                    args.gpu = self.device_handles

            # Handle cases where the user has only specified an argument and no specific device
            if args.gpu == None and gpu_args_enabled:
                args.gpu = self.device_handles
            if args.cpu == None and cpu_args_enabled:
                args.cpu = self.cpu_handles

            if args.cpu:
                self.static_cpu(args, multiple_devices, cpu, interface_ver)
            if args.gpu:
                self.logger.output = {}
                self.logger.clear_multiple_devices_output()
                self.static_gpu(args, multiple_devices, gpu, asic,
                                    bus, vbios, limit, driver, ras,
                                    board, numa, vram, cache, partition,
                                    dfc_ucode, fb_info, num_vf, soc_pstate,
                                    process_isolation, clock)
        elif self.helpers.is_amd_hsmp_initialized(): # Only CPU is initialized
            if args.cpu == None:
                args.cpu = self.cpu_handles

            self.static_cpu(args, multiple_devices, cpu, interface_ver)
        elif self.helpers.is_amdgpu_initialized(): # Only GPU is initialized
            if args.gpu == None:
                args.gpu = self.device_handles

            self.logger.clear_multiple_devices_output()
            self.static_gpu(args, multiple_devices, gpu, asic,
                                bus, vbios, limit, driver, ras,
                                board, numa, vram, cache, partition,
                                dfc_ucode, fb_info, num_vf, soc_pstate, xgmi_plpd,
                                process_isolation, clock)
        if self.logger.is_json_format():
            self.logger.combine_arrays_to_json()


    def firmware(self, args, multiple_devices=False, gpu=None, fw_list=True):
        """ Get Firmware information for target gpu

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

        # Handle multiple GPUs
        handled_multiple_gpus, device_handle = self.helpers.handle_gpus(args, self.logger, self.firmware)
        if handled_multiple_gpus:
            return # This function is recursive

        args.gpu = device_handle

        fw_list = {}

        # Get gpu_id for logging
        gpu_id = self.helpers.get_gpu_id_from_device_handle(args.gpu)

        if args.fw_list:
            try:
                fw_info = amdsmi_interface.amdsmi_get_fw_info(args.gpu)

                for fw_index, fw_entry in enumerate(fw_info['fw_list']):
                    # Change fw_name to fw_id
                    fw_entry['fw_id'] = fw_entry.pop('fw_name').name.replace("AMDSMI_FW_ID_", "")
                    fw_entry['fw_version'] = fw_entry.pop('fw_version') # popping to ensure order

                    # Add custom human readable formatting
                    if self.logger.is_human_readable_format():
                        fw_info['fw_list'][fw_index] = {f'FW {fw_index}': fw_entry}
                    else:
                        fw_info['fw_list'][fw_index] = fw_entry

                fw_list.update(fw_info)
            except amdsmi_exception.AmdSmiLibraryException as e:
                fw_list['fw_list'] = "N/A"
                logging.debug("Failed to get firmware info for gpu %s | %s", gpu_id, e.get_error_info())

        multiple_devices_csv_override = False
        # Convert and store output by pid for csv format
        if self.logger.is_csv_format():
            fw_key = 'fw_list'
            for fw_info_dict in fw_list[fw_key]:
                for key, value in fw_info_dict.items():
                    multiple_devices_csv_override = True
                    self.logger.store_output(args.gpu, key, value)
                self.logger.store_multiple_device_output()
        else:
            # Store values in logger.output
            self.logger.store_output(args.gpu, 'values', fw_list)

        if multiple_devices:
            self.logger.store_multiple_device_output()
            return # Skip printing when there are multiple devices

        self.logger.print_output(multiple_device_enabled=multiple_devices_csv_override)


    def bad_pages(self, args, multiple_devices=False, gpu=None, retired=None, pending=None, un_res=None):
        """ Get bad pages information for target gpu

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            gpu (device_handle, optional): device_handle for target device. Defaults to None.
            retired (bool, optional) - Value override for args.retired
            pending (bool, optional) - Value override for args.pending/
            un_res (bool, optional) - Value override for args.un_res

        Raises:
            IndexError: Index error if gpu list is empty

        Returns:
            None: Print output via AMDSMILogger to destination
        """
        # Set args.* to passed in arguments
        if gpu:
            args.gpu = gpu
        if retired:
            args.retired = retired
        if pending:
            args.pending = pending
        if un_res:
            args.un_res = un_res

        # Handle No GPU passed
        if args.gpu == None:
            args.gpu = self.device_handles

        # Handle multiple GPUs
        handled_multiple_gpus, device_handle = self.helpers.handle_gpus(args, self.logger, self.bad_pages)
        if handled_multiple_gpus:
            return # This function is recursive

        args.gpu = device_handle

        # If all arguments are False, the print all bad_page information
        if not any([args.retired, args.pending, args.un_res]):
            args.retired = args.pending = args.un_res = True

        values_dict = {}

        # Get gpu_id for logging
        gpu_id = self.helpers.get_gpu_id_from_device_handle(args.gpu)

        bad_pages_not_found = "No bad pages found."
        try:
            bad_page_info = amdsmi_interface.amdsmi_get_gpu_bad_page_info(args.gpu)
            # If bad_page_info is an empty list overwrite with not found error statement
            if bad_page_info == []:
                bad_page_info = bad_pages_not_found
                bad_page_error = True
            else:
                bad_page_error = False
        except amdsmi_exception.AmdSmiLibraryException as e:
            bad_page_info = "N/A"
            bad_page_error = True
            logging.debug("Failed to get bad page info for gpu %s | %s", gpu_id, e.get_error_info())

        if args.retired:
            if bad_page_error:
                values_dict['retired'] = bad_page_info
            else:
                bad_page_info_output = []
                for bad_page in bad_page_info:
                    if bad_page["status"] == amdsmi_interface.AmdSmiMemoryPageStatus.RESERVED:
                        bad_page_info_entry = {}
                        bad_page_info_entry["page_address"] = bad_page["page_address"]
                        bad_page_info_entry["page_size"] = bad_page["page_size"]
                        status_string = amdsmi_interface.amdsmi_wrapper.amdsmi_memory_page_status_t__enumvalues[bad_page["status"]]
                        bad_page_info_entry["status"] = status_string.replace("AMDSMI_MEM_PAGE_STATUS_", "")
                        bad_page_info_output.append(bad_page_info_entry)
                # Remove brackets if there is only one value
                if len(bad_page_info_output) == 1:
                    bad_page_info_output = bad_page_info_output[0]

                if bad_page_info_output == []:
                    values_dict['retired'] = bad_pages_not_found
                else:
                    values_dict['retired'] = bad_page_info_output

        if args.pending:
            if bad_page_error:
                values_dict['pending'] = bad_page_info
            else:
                bad_page_info_output = []
                for bad_page in bad_page_info:
                    if bad_page["status"] == amdsmi_interface.AmdSmiMemoryPageStatus.PENDING:
                        bad_page_info_entry = {}
                        bad_page_info_entry["page_address"] = bad_page["page_address"]
                        bad_page_info_entry["page_size"] = bad_page["page_size"]
                        status_string = amdsmi_interface.amdsmi_wrapper.amdsmi_memory_page_status_t__enumvalues[bad_page["status"]]
                        bad_page_info_entry["status"] = status_string.replace("AMDSMI_MEM_PAGE_STATUS_", "")
                        bad_page_info_output.append(bad_page_info_entry)
                # Remove brackets if there is only one value
                if len(bad_page_info_output) == 1:
                    bad_page_info_output = bad_page_info_output[0]

                if bad_page_info_output == []:
                    values_dict['pending'] = bad_pages_not_found
                else:
                    values_dict['pending'] = bad_page_info_output

        if args.un_res:
            if bad_page_error:
                values_dict['un_res'] = bad_page_info
            else:
                bad_page_info_output = []
                for bad_page in bad_page_info:
                    if bad_page["status"] == amdsmi_interface.AmdSmiMemoryPageStatus.UNRESERVABLE:
                        bad_page_info_entry = {}
                        bad_page_info_entry["page_address"] = bad_page["page_address"]
                        bad_page_info_entry["page_size"] = bad_page["page_size"]
                        status_string = amdsmi_interface.amdsmi_wrapper.amdsmi_memory_page_status_t__enumvalues[bad_page["status"]]
                        bad_page_info_entry["status"] = status_string.replace("AMDSMI_MEM_PAGE_STATUS_", "")
                        bad_page_info_output.append(bad_page_info_entry)
                # Remove brackets if there is only one value
                if len(bad_page_info_output) == 1:
                    bad_page_info_output = bad_page_info_output[0]

                if bad_page_info_output == []:
                    values_dict['un_res'] = bad_pages_not_found
                else:
                    values_dict['un_res'] = bad_page_info_output

        # Store values in logger.output
        self.logger.store_output(args.gpu, 'values', values_dict)

        if multiple_devices:
            self.logger.store_multiple_device_output()
            return # Skip printing when there are multiple devices

        self.logger.print_output()


    def build_xcp_dict(self, key, violation_status, num_partition):
        if not isinstance(violation_status[key], list):
            if "active_" in key:
               if violation_status[key] != "N/A":
                   if violation_status[key] is True:
                       violation_status[key] = "ACTIVE"
                   elif violation_status[key] is False:
                       violation_status[key] = "NOT ACTIVE"
            ret = violation_status[key]
        elif isinstance(violation_status[key], list):
            for row in violation_status[key]:
                for element in row:
                    if element != "N/A":
                        if "active_" in key:
                            if element is True:
                                row[row.index(element)] = "ACTIVE"
                            elif element is False:
                                row[row.index(element)] = "NOT ACTIVE"
                        elif ("per_" or "acc_") in key:
                            row[row.index(element)] = element
                    else:
                        continue
            ret = {f"xcp_{i}": violation_status[key][i] for i in range(num_partition)}
        return ret

    def metric_gpu(self, args, multiple_devices=False, watching_output=False, gpu=None,
                usage=None, watch=None, watch_time=None, iterations=None, power=None,
                clock=None, temperature=None, ecc=None, ecc_blocks=None, pcie=None,
                fan=None, voltage_curve=None, overdrive=None, perf_level=None,
                xgmi_err=None, energy=None, mem_usage=None, voltage=None, schedule=None,
                guard=None, guest_data=None, fb_usage=None, xgmi=None, throttle=None,
                base_board=None, gpu_board=None):
        """Get Metric information for target gpu

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            watching_output (bool, optional): True if watch argument has been set. Defaults to False.
            gpu (device_handle, optional): device_handle for target device. Defaults to None.
            usage (bool, optional): Value override for args.usage. Defaults to None.
            watch (Positive int, optional): Value override for args.watch. Defaults to None.
            watch_time (Positive int, optional): Value override for args.watch_time. Defaults to None.
            iterations (Positive int, optional): Value override for args.iterations. Defaults to None.
            power (bool, optional): Value override for args.power. Defaults to None.
            clock (bool, optional): Value override for args.clock. Defaults to None.
            temperature (bool, optional): Value override for args.temperature. Defaults to None.
            ecc (bool, optional): Value override for args.ecc. Defaults to None.
            ecc_blocks (bool, optional): Value override for args.ecc. Defaults to None.
            pcie (bool, optional): Value override for args.pcie. Defaults to None.
            fan (bool, optional): Value override for args.fan. Defaults to None.
            voltage_curve (bool, optional): Value override for args.voltage_curve. Defaults to None.
            overdrive (bool, optional): Value override for args.overdrive. Defaults to None.
            perf_level (bool, optional): Value override for args.perf_level. Defaults to None.
            xgmi_err (bool, optional): Value override for args.xgmi_err. Defaults to None.
            energy (bool, optional): Value override for args.energy. Defaults to None.
            mem_usage (bool, optional): Value override for args.mem_usage. Defaults to None.
            voltage (bool, optional): Value override for args.voltage. Defaults to None.
            schedule (bool, optional): Value override for args.schedule. Defaults to None.
            guard (bool, optional): Value override for args.guard. Defaults to None.
            guest_data (bool, optional): Value override for args.guest_data. Defaults to None.
            fb_usage (bool, optional): Value override for args.fb_usage. Defaults to None.
            xgmi (bool, optional): Value override for args.xgmi. Defaults to None.
            throttle (bool, optional): Value override for args.throttle. Defaults to None.

        Raises:
            IndexError: Index error if gpu list is empty

        Returns:
            None: Print output via AMDSMILogger to destination
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

        # Store args that are applicable to the current platform
        current_platform_args = []
        current_platform_values = []

        if not self.helpers.is_hypervisor() and not self.helpers.is_windows():
            if mem_usage:
                args.mem_usage = mem_usage
            current_platform_args += ["mem_usage"]
            current_platform_values += [args.mem_usage]

        if self.helpers.is_hypervisor() or self.helpers.is_baremetal() or self.helpers.is_linux():
            if usage:
                args.usage = usage
            if base_board:
                args.base_board = base_board
            if gpu_board:
                args.gpu_board = gpu_board
            if power:
                args.power = power
            if clock:
                args.clock = clock
            if temperature:
                args.temperature = temperature
            if voltage:
                args.voltage = voltage
            if pcie:
                args.pcie = pcie
            if ecc:
                args.ecc = ecc
            if ecc_blocks:
                args.ecc_blocks = ecc_blocks
            current_platform_args += ["usage", "power", "clock", "temperature", "voltage", "pcie", "ecc", "ecc_blocks", "base_board","gpu_board"]
            current_platform_values += [args.usage, args.power, args.clock,
                                        args.temperature, args.voltage, args.pcie]
            current_platform_values += [args.ecc, args.ecc_blocks, args.base_board, args.gpu_board]

        if self.helpers.is_baremetal() and self.helpers.is_linux():
            if fan:
                args.fan = fan
            if voltage_curve:
                args.voltage_curve = voltage_curve
            if overdrive:
                args.overdrive = overdrive
            if perf_level:
                args.perf_level = perf_level
            if xgmi_err:
                args.xgmi_err = xgmi_err
            if energy:
                args.energy = energy
            if throttle:
                args.violation = throttle
                args.throttle = throttle
            current_platform_args += ["fan", "voltage_curve", "overdrive", "perf_level",
                                      "xgmi_err", "energy", "throttle"]
            current_platform_values += [args.fan, args.voltage_curve, args.overdrive,
                                        args.perf_level, args.xgmi_err, args.energy, args.throttle,
                                        ]

        if self.helpers.is_hypervisor():
            if schedule:
                args.schedule = schedule
            if guard:
                args.guard = guard
            if guest_data:
                args.guest_data = guest_data
            if fb_usage:
                args.fb_usage = fb_usage
            if xgmi:
                args.xgmi = xgmi
            current_platform_args += ["schedule", "guard", "guest_data", "fb_usage", "xgmi"]
            current_platform_values += [args.schedule, args.guard, args.guest_data,
                                        args.fb_usage, args.xgmi]

        # Handle No GPU passed
        if args.gpu == None:
            args.gpu = self.device_handles

        if not self.group_check_printed:
            self.helpers.check_required_groups()
            self.group_check_printed = True

        # Handle watch logic, will only enter this block once
        if args.watch:
            self.helpers.handle_watch(args=args, subcommand=self.metric_gpu, logger=self.logger)
            return

        # Handle multiple GPUs
        if isinstance(args.gpu, list):
            if len(args.gpu) > 1:
                # Deepcopy gpus as recursion will destroy the gpu list
                stored_gpus = []
                for gpu in args.gpu:
                    stored_gpus.append(gpu)

                # Store output from multiple devices
                for device_handle in args.gpu:
                    self.metric_gpu(args, multiple_devices=True, watching_output=watching_output, gpu=device_handle)

                # Reload original gpus
                args.gpu = stored_gpus

                # Print multiple device output
                if not self.logger.is_json_format():
                    self.logger.print_output(multiple_device_enabled=True, watching_output=watching_output)

                # Add output to total watch output and clear multiple device output
                if watching_output:
                    self.logger.store_watch_output(multiple_device_enabled=True)

                    # Flush the watching output
                    self.logger.print_output(multiple_device_enabled=True, watching_output=watching_output)

                return
            elif len(args.gpu) == 1:
                args.gpu = args.gpu[0]
            else:
                raise IndexError("args.gpu should not be an empty list")

        # Get gpu_id for logging
        gpu_id = self.helpers.get_gpu_id_from_device_handle(args.gpu)

        if args.loglevel == "DEBUG":
            try:
                # Get GPU Metrics table version
                gpu_metric_version_info = amdsmi_interface.amdsmi_get_gpu_metrics_header_info(args.gpu)
                gpu_metric_version_str = json.dumps(gpu_metric_version_info, indent=4)
                logging.debug("GPU Metrics table Version for GPU %s | %s", gpu_id, gpu_metric_version_str)
            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug("#1 - Unable to load GPU Metrics table version for %s | %s", gpu_id, e.get_error_info())

            try:
                # Get GPU Metrics table
                gpu_metric_debug_info = amdsmi_interface.amdsmi_get_gpu_metrics_info(args.gpu)
                gpu_metric_str = json.dumps(gpu_metric_debug_info, indent=4)
                logging.debug("GPU Metrics table for GPU %s | %s", gpu_id, str(gpu_metric_str))
            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug("#2 - Unable to load GPU Metrics table for %s | %s", gpu_id, e.get_error_info())

        logging.debug(f"Metric Arg information for GPU {gpu_id} on {self.helpers.os_info()}")
        logging.debug(f"Args:   {current_platform_args}")
        logging.debug(f"Values: {current_platform_values}")

        # Set the platform applicable args to True if no args are set
        if not any(current_platform_values):
            for arg in current_platform_args:
                setattr(args, arg, True)

        # Add timestamp and store values for specified arguments
        values_dict = {}

        #get metric info only once per gpu, this will speed up data output
        try:
            # Get GPU Metrics table
            gpu_metric = amdsmi_interface.amdsmi_get_gpu_metrics_info(args.gpu)
        except amdsmi_exception.AmdSmiLibraryException as e:
            logging.debug("#3 - Unable to load GPU Metrics table for %s | %s", gpu_id, e.get_error_info())
            gpu_metric = amdsmi_interface._NA_amdsmi_get_gpu_metrics_info()

        # Workaround for XCP (partition) metrics not providing num_partition in v1.0
        # Confirmed with driver team that we can default to 1 if num_partition is not defined.
        # Pending partitions exist, ie. partition_id > 0. See logic below.
        try:
            partition_id = amdsmi_interface.amdsmi_get_gpu_kfd_info(args.gpu)['current_partition_id']
        except amdsmi_exception.AmdSmiLibraryException as e:
            logging.debug("Failed to get current partition id for gpu %s | %s", gpu_id, e.get_error_info())
            partition_id = "N/A"

        num_partition = gpu_metric['num_partition']
        if num_partition == "N/A" and isinstance(partition_id, int) and partition_id > 0:
            num_partition = 1  # Workaround for XCP metrics not providing num_partition in v1.0
            logging.debug(f"num_partition is N/A and partition_id: {partition_id} (greater > 0).\nModified num_partition: {num_partition} to adjust for XCP metrics.")

        if self.logger.is_json_format():
            values_dict['gpu'] = int(gpu_id)
        # Populate the pcie_dict first due to multiple gpu metrics calls incorrectly increasing bandwidth
        if "pcie" in current_platform_args:
            if args.pcie:
                pcie_dict = {"width": "N/A",
                             "speed": "N/A",
                             "bandwidth": "N/A",
                             "replay_count" : "N/A",
                             "l0_to_recovery_count" : "N/A",
                             "replay_roll_over_count" : "N/A",
                             "nak_sent_count" : "N/A",
                             "nak_received_count" : "N/A",
                             "current_bandwidth_sent": "N/A",
                             "current_bandwidth_received": "N/A",
                             "max_packet_size": "N/A",
                             "lc_perf_other_end_recovery": "N/A"}

                try:
                    pcie_metric = amdsmi_interface.amdsmi_get_pcie_info(args.gpu)['pcie_metric']
                    logging.debug("PCIE Metric for %s | %s", gpu_id, pcie_metric)

                    pcie_dict['width'] = pcie_metric['pcie_width']

                    if pcie_metric['pcie_speed'] != "N/A":
                        if pcie_metric['pcie_speed'] % 1000 != 0:
                            pcie_speed_GTs_value = round(pcie_metric['pcie_speed'] / 1000, 1)
                        else:
                            pcie_speed_GTs_value = round(pcie_metric['pcie_speed'] / 1000)
                        pcie_dict['speed'] = pcie_speed_GTs_value

                    pcie_dict['bandwidth'] = pcie_metric['pcie_bandwidth']

                    pcie_dict['replay_count'] = pcie_metric['pcie_replay_count']
                    if pcie_dict['replay_count'] == "N/A":
                        try:
                            pcie_replay = amdsmi_interface.amdsmi_get_gpu_pci_replay_counter(args.gpu)
                            pcie_dict['replay_count'] = pcie_replay
                        except amdsmi_exception.AmdSmiLibraryException as e:
                            logging.debug("Failed to get sysfs pcie replay counter on gpu %s | %s", gpu_id, e.get_error_info())

                    pcie_dict['l0_to_recovery_count'] = pcie_metric['pcie_l0_to_recovery_count']
                    pcie_dict['replay_roll_over_count'] = pcie_metric['pcie_replay_roll_over_count']
                    pcie_dict['nak_received_count'] = pcie_metric['pcie_nak_received_count']
                    pcie_dict['nak_sent_count'] = pcie_metric['pcie_nak_sent_count']
                    pcie_dict['lc_perf_other_end_recovery'] = pcie_metric['pcie_lc_perf_other_end_recovery_count']

                    pcie_speed_unit = 'GT/s'
                    pcie_bw_unit = 'Mb/s'
                    if self.logger.is_human_readable_format():
                        if pcie_dict['speed'] != "N/A":
                            pcie_dict['speed'] = f"{pcie_dict['speed']} {pcie_speed_unit}"
                        if pcie_dict['bandwidth'] != "N/A":
                            pcie_dict['bandwidth'] = f"{pcie_dict['bandwidth']} {pcie_bw_unit}"
                    if self.logger.is_json_format():
                        if pcie_dict['speed'] != "N/A":
                            pcie_dict['speed'] = {"value" : pcie_dict['speed'],
                                                  "unit" : pcie_speed_unit}
                        if pcie_dict['bandwidth'] != "N/A":
                            pcie_dict['bandwidth'] = {"value" : pcie_dict['bandwidth'],
                                                      "unit" : pcie_bw_unit}
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug("Failed to get pcie link status for gpu %s | %s", gpu_id, e.get_error_info())

                try:
                    pcie_bw = amdsmi_interface.amdsmi_get_gpu_pci_throughput(args.gpu)
                    sent = pcie_bw['sent'] * pcie_bw['max_pkt_sz']
                    received = pcie_bw['received'] * pcie_bw['max_pkt_sz']

                    bw_unit = "Mb/s"
                    packet_size_unit = "B"
                    if sent > 0:
                        sent = sent // 1024 // 1024
                    if received > 0:
                        received = received // 1024 // 1024

                    if self.logger.is_human_readable_format():
                        sent = f"{sent} {bw_unit}"
                        received = f"{received} {bw_unit}"
                        pcie_bw['max_pkt_sz'] = f"{pcie_bw['max_pkt_sz']} {packet_size_unit}"
                    if self.logger.is_json_format():
                        sent = {"value" : sent,
                                "unit" : bw_unit}
                        received = {"value" : received,
                                    "unit" : bw_unit}
                        pcie_bw['max_pkt_sz'] = {"value" : pcie_bw['max_pkt_sz'],
                                                 "unit" : packet_size_unit}

                    pcie_dict['current_bandwidth_sent'] = sent
                    pcie_dict['current_bandwidth_received'] = received
                    pcie_dict['max_packet_size'] = pcie_bw['max_pkt_sz']
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug("Failed to get pcie bandwidth for gpu %s | %s", gpu_id, e.get_error_info())

        if "usage" in current_platform_args:
            if args.usage:
                try:
                    engine_usage = amdsmi_interface.amdsmi_get_gpu_activity(args.gpu)
                    logging.debug(f"engine_usage dictionary = {engine_usage}")

                    # TODO: move vcn_activity and jpeg_activity into amdsmi_get_gpu_activity
                    engine_usage['vcn_activity'] = gpu_metric['vcn_activity']
                    engine_usage['jpeg_activity'] = gpu_metric['jpeg_activity']
                    engine_usage['gfx_busy_inst'] = "N/A"
                    engine_usage['jpeg_busy'] = "N/A"
                    engine_usage['vcn_busy'] = "N/A"

                    if num_partition != "N/A":
                        # these are one after another, in order to display each in sub-sections
                        new_xcp_dict = {}
                        for current_xcp in range(num_partition):
                            new_xcp_dict[f"xcp_{current_xcp}"] = gpu_metric['xcp_stats.gfx_busy_inst'][current_xcp]
                        engine_usage['gfx_busy_inst'] = new_xcp_dict

                        new_xcp_dict = {}
                        for current_xcp in range(num_partition):
                            new_xcp_dict[f"xcp_{current_xcp}"] = gpu_metric['xcp_stats.jpeg_busy'][current_xcp]
                        engine_usage['jpeg_busy'] = new_xcp_dict

                        new_xcp_dict = {}
                        for current_xcp in range(num_partition):
                            new_xcp_dict[f"xcp_{current_xcp}"] = gpu_metric['xcp_stats.vcn_busy'][current_xcp]
                        engine_usage['vcn_busy'] = new_xcp_dict

                    logging.debug(f"After updates to engine_usage dictionary = {engine_usage}")

                    for key, value in engine_usage.items():
                        activity_unit = '%'
                        if self.logger.is_human_readable_format():
                            if isinstance(value, list):
                                for index, activity in enumerate(value):
                                    if activity != "N/A":
                                        engine_usage[key][index] = f"{activity} {activity_unit}"
                                # Convert list to a string for human readable format
                                engine_usage[key] = '[' + ", ".join(engine_usage[key]) + ']'
                            elif isinstance(value, dict):
                                for k, v in value.items():
                                        for index, activity in enumerate(v):
                                            if activity != "N/A":
                                                value[k][index] = f"{activity} {activity_unit}"
                                        # Convert list to a string for human readable format
                                        value[k] = '[' + ", ".join(value[k]) + ']'
                            elif value != "N/A":
                                engine_usage[key] = f"{value} {activity_unit}"
                        if self.logger.is_json_format():
                            if isinstance(value, list):
                                for index, activity in enumerate(value):
                                    if activity != "N/A":
                                        engine_usage[key][index] = {"value" : activity,
                                                                    "unit" : activity_unit}
                            elif isinstance(value, dict):
                                for k, v in value.items():
                                    for index, activity in enumerate(v):
                                        if activity != "N/A":
                                            value[k][index] = {"value" : activity,
                                                                "unit" : activity_unit}
                            elif value != "N/A":
                                engine_usage[key] = {"value" : value,
                                                     "unit" : activity_unit}

                    values_dict['usage'] = engine_usage
                except Exception as e:
                    values_dict['usage'] = "N/A"
                    logging.debug("Failed to get gpu activity for gpu %s | %s", gpu_id, e)
        if "power" in current_platform_args:
            if args.power:
                power_dict = {'socket_power': "N/A",
                              'gfx_voltage': "N/A",
                              'soc_voltage': "N/A",
                              'mem_voltage': "N/A",
                              'throttle_status': "N/A",
                              'power_management': "N/A"}

                try:
                    voltage_unit = "mV"
                    power_unit = "W"
                    power_info = amdsmi_interface.amdsmi_get_power_info(args.gpu)
                    for key, value in power_info.items():
                        if "voltage" in key:
                            power_info[key] = self.helpers.unit_format(self.logger,
                                                                        value,
                                                                        voltage_unit)
                        elif key == "socket_power":
                            power_info[key] = self.helpers.unit_format(self.logger,
                                                                        value,
                                                                        power_unit)

                    power_dict['socket_power'] = power_info['socket_power']
                    power_dict['gfx_voltage'] = power_info['gfx_voltage']
                    power_dict['soc_voltage'] = power_info['soc_voltage']
                    power_dict['mem_voltage'] = power_info['mem_voltage']

                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug("Failed to get power info for gpu %s | %s", gpu_id, e.get_error_info())

                try:
                    is_power_management_enabled = amdsmi_interface.amdsmi_is_gpu_power_management_enabled(args.gpu)
                    if is_power_management_enabled:
                        power_dict['power_management'] = "ENABLED"
                    else:
                        power_dict['power_management'] = "DISABLED"
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug("Failed to get power management status for gpu %s | %s", gpu_id, e.get_error_info())

                try:
                    power_dict['throttle_status'] = "N/A"
                    throttle_status = gpu_metric['throttle_status']
                    if throttle_status != "N/A":
                        if throttle_status:
                            power_dict['throttle_status'] = "THROTTLED"
                        else:
                            power_dict['throttle_status'] = "UNTHROTTLED"
                except Exception as e:
                    logging.debug("Failed to get throttle status for gpu %s | %s", gpu_id, e)

                values_dict['power'] = power_dict
        if "clock" in current_platform_args:
            if args.clock:
                # Populate Skeleton output with N/A
                clocks = {}

                for clock_index in range(amdsmi_interface.AMDSMI_MAX_NUM_GFX_CLKS):
                    gfx_index = f"gfx_{clock_index}"
                    clocks[gfx_index] = {"clk" : "N/A",
                                         "min_clk" : "N/A",
                                         "max_clk" : "N/A",
                                         "clk_locked" : "N/A",
                                         "deep_sleep" : "N/A"}

                clocks["mem_0"] = {"clk" : "N/A",
                                   "min_clk" : "N/A",
                                   "max_clk" : "N/A",
                                   "clk_locked" : "N/A",
                                   "deep_sleep" : "N/A"}

                for clock_index in range(amdsmi_interface.AMDSMI_MAX_NUM_CLKS):
                    vclk_index = f"vclk_{clock_index}"
                    clocks[vclk_index] = {"clk" : "N/A",
                                          "min_clk" : "N/A",
                                          "max_clk" : "N/A",
                                          "clk_locked" : "N/A",
                                          "deep_sleep" : "N/A"}

                for clock_index in range(amdsmi_interface.AMDSMI_MAX_NUM_CLKS):
                    dclk_index = f"dclk_{clock_index}"
                    clocks[dclk_index] = {"clk" : "N/A",
                                          "min_clk" : "N/A",
                                          "max_clk" : "N/A",
                                          "clk_locked" : "N/A",
                                          "deep_sleep" : "N/A"}

                clocks["fclk_0"] = {"clk" : "N/A",
                                    "min_clk" : "N/A",
                                    "max_clk" : "N/A",
                                    "clk_locked" : "N/A",
                                    "deep_sleep" : "N/A"}

                clocks["socclk_0"] = {"clk" : "N/A",
                                      "min_clk" : "N/A",
                                      "max_clk" : "N/A",
                                      "clk_locked" : "N/A",
                                      "deep_sleep" : "N/A"}

                clock_unit = "MHz"

                # Populate clock values from gpu_metrics_info
                # Populate GFX clock values
                try:
                    current_gfx_clocks = gpu_metric["current_gfxclks"]
                    if current_gfx_clocks != "N/A":
                        for clock_index, current_gfx_clock in enumerate(current_gfx_clocks):
                            # If the current clock is N/A then nothing else applies
                            if current_gfx_clock == "N/A":
                                continue
                            gfx_index = f"gfx_{clock_index}"
                            clocks[gfx_index]["clk"] = self.helpers.unit_format(self.logger,
                                                                                current_gfx_clock,
                                                                                clock_unit)
                            # Populate clock locked status
                            if gpu_metric["gfxclk_lock_status"] != "N/A":
                                gfx_clock_lock_flag = 1 << clock_index # This is the position of the clock lock flag
                                if gpu_metric["gfxclk_lock_status"] & gfx_clock_lock_flag:
                                    clocks[gfx_index]["clk_locked"] = "ENABLED"
                                else:
                                    clocks[gfx_index]["clk_locked"] = "DISABLED"
                except Exception as e:
                    logging.debug("Failed to get current_gfxclks for gpu %s | %s", gpu_id, e)

                # Populate MEM clock value
                try:
                    current_mem_clock = gpu_metric["current_uclk"] # single value
                    if current_mem_clock != "N/A":
                        clocks["mem_0"]["clk"] = self.helpers.unit_format(self.logger,
                                                                          current_mem_clock,
                                                                          clock_unit)
                except Exception as e:
                    logging.debug("Failed to get current_uclk for gpu %s | %s", gpu_id, e)

                # Populate VCLK clock values
                try:
                    current_vclk_clocks = gpu_metric["current_vclk0s"]
                    # If the current vclk clocks are not available, we cannot proceed further
                    if current_vclk_clocks != "N/A":
                        for clock_index, current_vclk_clock in enumerate(current_vclk_clocks):
                            # If the current clock is N/A then nothing else applies
                            if current_vclk_clock == "N/A":
                                continue
                            vclk_index = f"vclk_{clock_index}"
                            clocks[vclk_index]["clk"] = self.helpers.unit_format(self.logger,
                                                                                 current_vclk_clock,
                                                                                 clock_unit)
                except Exception as e:
                    logging.debug("Failed to get current_vclk0s for gpu %s | %s", gpu_id, e)

                # Populate DCLK clock values
                try:
                    current_dclk_clocks = gpu_metric["current_dclk0s"]
                    # If the current dclk clocks are not available, we cannot proceed further
                    if current_dclk_clocks != "N/A":
                        for clock_index, current_dclk_clock in enumerate(current_dclk_clocks):
                            # If the current clock is N/A then nothing else applies
                            if current_dclk_clock == "N/A":
                                continue
                            dclk_index = f"dclk_{clock_index}"
                            clocks[dclk_index]["clk"] = self.helpers.unit_format(self.logger,
                                                                                 current_dclk_clock,
                                                                                 clock_unit)
                except Exception as e:
                    logging.debug("Failed to get current_dclk0s for gpu %s | %s", gpu_id, e)

                # Populate FCLK clock value; fclk not present in gpu_metrics so use amdsmi_get_clk_freq
                try:
                    frequency_dict = amdsmi_interface.amdsmi_get_clk_freq(args.gpu, amdsmi_interface.AmdSmiClkType.DF)
                    current_fclk_clock = frequency_dict['frequency'][frequency_dict['current']]
                    current_fclk_clock = self.helpers.convert_SI_unit(current_fclk_clock, self.helpers.SI_Unit.MICRO)
                    clocks["fclk_0"]["clk"] = self.helpers.unit_format(self.logger,
                                                                       current_fclk_clock,
                                                                       clock_unit)
                except (KeyError, amdsmi_exception.AmdSmiLibraryException) as e:
                    logging.debug("Failed to get fclk info for gpu %s | %s", gpu_id, e)

                # Populate SOCCLK clock value
                try:
                    current_socclk_clock = gpu_metric["current_socclk"]
                    # If the current socclk clocks are not available, we cannot proceed further
                    if current_socclk_clock != "N/A":
                        clocks["socclk_0"]["clk"] = self.helpers.unit_format(self.logger,
                                                                             current_socclk_clock,
                                                                             clock_unit)
                except KeyError as e:
                    logging.debug("Failed to get current_socclk for gpu %s | %s", gpu_id, e)


                # Populate the max and min clock values from sysfs.
                # Min and Max values are per clock type, not per clock engine.
                # Populate the deep sleep value from amdsmi_get_clock_info

                # GFX min and max clocks
                try:
                    gfx_clock_info_dict = amdsmi_interface.amdsmi_get_clock_info(args.gpu,
                                                                                 amdsmi_interface.AmdSmiClkType.GFX)
                    for clock_index in range(amdsmi_interface.AMDSMI_MAX_NUM_GFX_CLKS):
                        gfx_index = f"gfx_{clock_index}"

                        if clocks[gfx_index]["clk"] == "N/A":
                            # if the current clock is N/A then we shouldn't populate the max and min values
                            continue
                        clocks[gfx_index]["min_clk"] = self.helpers.unit_format(self.logger,
                                                                                gfx_clock_info_dict["min_clk"],
                                                                                clock_unit)
                        clocks[gfx_index]["max_clk"] = self.helpers.unit_format(self.logger,
                                                                                gfx_clock_info_dict["max_clk"],
                                                                                clock_unit)
                        # Add the clk_deep_sleep
                        clocks[gfx_index]["deep_sleep"] = gfx_clock_info_dict["clk_deep_sleep"]
                except (KeyError, amdsmi_exception.AmdSmiLibraryException) as e:
                    logging.debug("Failed to get gfx clock info for gpu %s | %s", gpu_id, e)

                # MEM min and max clocks
                try:
                    mem_clock_info_dict = amdsmi_interface.amdsmi_get_clock_info(args.gpu,
                                                                                 amdsmi_interface.AmdSmiClkType.MEM)
                    # if the current clock is N/A then we shouldn't populate the max and min values
                    if clocks["mem_0"]["clk"] != "N/A":
                        clocks["mem_0"]["min_clk"] = self.helpers.unit_format(self.logger,
                                                                                mem_clock_info_dict["min_clk"],
                                                                                clock_unit)
                        clocks["mem_0"]["max_clk"] = self.helpers.unit_format(self.logger,
                                                                                mem_clock_info_dict["max_clk"],
                                                                                clock_unit)
                        # Add the clk_deep_sleep
                        clocks["mem_0"]["deep_sleep"] = mem_clock_info_dict["clk_deep_sleep"]
                except (KeyError, amdsmi_exception.AmdSmiLibraryException) as e:
                    logging.debug("Failed to get mem clock info for gpu %s | %s", gpu_id, e)

                # VCLK min and max clocks
                try:
                    # Retrieve clock information for VCLK0 (Video Clock 0)
                    vclk_clock_info_dict = amdsmi_interface.amdsmi_get_clock_info(args.gpu, amdsmi_interface.AmdSmiClkType.VCLK0)

                    # Iterate through the maximum number of VCLK clocks supported
                    for index in range(amdsmi_interface.AMDSMI_MAX_NUM_CLKS):
                        vclk_index = f"vclk_{index}"  # Construct the index key for the clock

                        # Check if the current clock value is not "N/A"
                        if clocks[vclk_index]["clk"] != "N/A":
                            # Format and assign the minimum clock value for the current VCLK
                            clocks[vclk_index]["min_clk"] = self.helpers.unit_format(self.logger,
                                                                                    vclk_clock_info_dict["min_clk"],
                                                                                    clock_unit)
                            # Format and assign the maximum clock value for the current VCLK
                            clocks[vclk_index]["max_clk"] = self.helpers.unit_format(self.logger,
                                                                                    vclk_clock_info_dict["max_clk"],
                                                                                    clock_unit)
                            # Add the clk_deep_sleep
                            clocks[vclk_index]["deep_sleep"] = vclk_clock_info_dict["clk_deep_sleep"]
                except (KeyError, amdsmi_exception.AmdSmiLibraryException) as e:
                    # Log a debug message if retrieving VCLK clock information fails
                    logging.debug("Failed to get vclk clock info for gpu %s | %s", gpu_id, e)

                # DCLK min and max clocks
                try:
                    # Retrieve clock information for DCLK0 (Display Clock 0)
                    dclk_clock_info_dict = amdsmi_interface.amdsmi_get_clock_info(args.gpu, amdsmi_interface.AmdSmiClkType.DCLK0)

                    # Iterate through the maximum number of DCLK clocks supported
                    for index in range(amdsmi_interface.AMDSMI_MAX_NUM_CLKS):
                        dclk_index = f"dclk_{index}" # Construct the index key for the clock

                        # Check if the current clock value is not "N/A"
                        if clocks[dclk_index]["clk"] != "N/A":
                            # Format and assign the minimum clock value for the current DCLK
                            clocks[dclk_index]["min_clk"] = self.helpers.unit_format(self.logger,
                                                                                    dclk_clock_info_dict["min_clk"],
                                                                                    clock_unit)
                            # Format and assign the maximum clock value for the current DCLK
                            clocks[dclk_index]["max_clk"] = self.helpers.unit_format(self.logger,
                                                                                    dclk_clock_info_dict["max_clk"],
                                                                                    clock_unit)
                            # Add the clk_deep_sleep
                            clocks[dclk_index]["deep_sleep"] = dclk_clock_info_dict["clk_deep_sleep"]
                except (KeyError, amdsmi_exception.AmdSmiLibraryException) as e:
                    logging.debug("Failed to get dclk clock info for gpu %s | %s", gpu_id, e)

                # FCLK min and max clocks
                try:
                    fclk_clk_info_dict = amdsmi_interface.amdsmi_get_clock_info(args.gpu,
                                                                                amdsmi_interface.AmdSmiClkType.DF)
                    # if the current clock is N/A then we shouldn't populate the max and min values
                    if clocks["fclk_0"]["clk"] != "N/A":
                        clocks["fclk_0"]["min_clk"] = self.helpers.unit_format(self.logger,
                                                                                fclk_clk_info_dict["min_clk"],
                                                                                clock_unit)
                        clocks["fclk_0"]["max_clk"] = self.helpers.unit_format(self.logger,
                                                                                fclk_clk_info_dict["max_clk"],
                                                                                clock_unit)
                        # Add the clk_deep_sleep
                        clocks["fclk_0"]["deep_sleep"] = fclk_clk_info_dict["clk_deep_sleep"]
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug("Failed to get fclk info for gpu %s | %s", gpu_id, e.get_error_info())

                # SOCCLK min and max clocks
                try:
                    socclk_clk_info_dict = amdsmi_interface.amdsmi_get_clock_info(args.gpu,
                                                                                amdsmi_interface.AmdSmiClkType.SOC)
                    # if the current clock is N/A then we shouldn't populate the max and min values
                    if clocks["socclk_0"]["clk"] != "N/A":
                        clocks["socclk_0"]["min_clk"] = self.helpers.unit_format(self.logger,
                                                                                socclk_clk_info_dict["min_clk"],
                                                                                clock_unit)
                        clocks["socclk_0"]["max_clk"] = self.helpers.unit_format(self.logger,
                                                                                socclk_clk_info_dict["max_clk"],
                                                                                clock_unit)
                        # Add the clk_deep_sleep
                        clocks["socclk_0"]["deep_sleep"] = socclk_clk_info_dict["clk_deep_sleep"]
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug("Failed to get socclk info for gpu %s | %s", gpu_id, e.get_error_info())

                # Iterate over each clock and its data to determine if deep sleep is enabled
                # based on the comparison between the current clock value and the minimum clock value.
                for clock, clock_data in clocks.items():
                    clk_value = 0
                    min_clk_value = 0
                    try:
                        clk = clock_data["clk"]
                        min_clk = clock_data["min_clk"]
                        if clk == "N/A" or min_clk == "N/A":
                            continue
                        # Extract numeric value if clk/min_clk is a dict, else use as is
                        if isinstance(clk, dict):
                            clk_value = int(clk.get("value", 0))
                        else:
                            if isinstance(clk, str):
                                clk_value = int(str(clk).split()[0])
                            else:
                                clk_value = int(clk)
                        if isinstance(min_clk, dict):
                            min_clk_value = int(min_clk.get("value", 0))
                        else:
                            if isinstance(min_clk, str):
                                min_clk_value = int(str(min_clk).split()[0])
                            else:
                                min_clk_value = int(min_clk)
                        # If the clk value is less than the min_clk value, then deep sleep is enabled
                        if clk_value < min_clk_value:
                            clock_data["deep_sleep"] = "ENABLED"
                        else:
                            clock_data["deep_sleep"] = "DISABLED"
                    except Exception as e:
                        logging.debug("Failed to get deep sleep status for gpu %s | %s", gpu_id, e)

                values_dict['clock'] = clocks
        if "temperature" in current_platform_args:
            if args.temperature:
                try:
                    temperature_edge_current = amdsmi_interface.amdsmi_get_temp_metric(
                        args.gpu, amdsmi_interface.AmdSmiTemperatureType.EDGE, amdsmi_interface.AmdSmiTemperatureMetric.CURRENT)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    temperature_edge_current = "N/A"
                    logging.debug("Failed to get current edge temperature for gpu %s | %s", gpu_id, e.get_error_info())

                try:
                    temperature_edge_limit = amdsmi_interface.amdsmi_get_temp_metric(
                        args.gpu, amdsmi_interface.AmdSmiTemperatureType.EDGE, amdsmi_interface.AmdSmiTemperatureMetric.CRITICAL)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    temperature_edge_limit = "N/A"
                    logging.debug("Failed to get edge temperature limit for gpu %s | %s", gpu_id, e.get_error_info())

                # If edge limit is reporting 0 then set the current edge temp to N/A
                if temperature_edge_limit == 0:
                    temperature_edge_current = "N/A"

                try:
                    temperature_hotspot_current = amdsmi_interface.amdsmi_get_temp_metric(
                        args.gpu, amdsmi_interface.AmdSmiTemperatureType.HOTSPOT, amdsmi_interface.AmdSmiTemperatureMetric.CURRENT)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    temperature_hotspot_current = "N/A"
                    logging.debug("Failed to get current hotspot temperature for gpu %s | %s", gpu_id, e.get_error_info())

                try:
                    temperature_vram_current = amdsmi_interface.amdsmi_get_temp_metric(
                        args.gpu, amdsmi_interface.AmdSmiTemperatureType.VRAM, amdsmi_interface.AmdSmiTemperatureMetric.CURRENT)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    temperature_vram_current = "N/A"
                    logging.debug("Failed to get current vram temperature for gpu %s | %s", gpu_id, e.get_error_info())

                temperatures = {'edge': temperature_edge_current,
                                'hotspot': temperature_hotspot_current,
                                'mem': temperature_vram_current}

                temp_unit_human_readable = '\N{DEGREE SIGN}C'
                temp_unit_json = 'C'
                for temperature_key, temperature_value in temperatures.items():
                    if 'N/A' not in str(temperature_value):
                        if self.logger.is_human_readable_format():
                            temperatures[temperature_key] = f"{temperature_value} {temp_unit_human_readable}"
                        if self.logger.is_json_format():
                            temperatures[temperature_key] = {"value" : temperature_value,
                                                            "unit" : temp_unit_json}

                values_dict['temperature'] = temperatures

        # Since pcie bw may increase based on frequent metrics calls, we add it to the output here, but the populate the values first
        if "pcie" in current_platform_args:
            if args.pcie:
                values_dict['pcie'] = pcie_dict

        if "gpu_board" in current_platform_args:
            if args.gpu_board:
                gpu_board_temp_dict = {}
                gpu_board_temp_types = [
                    amdsmi_interface.AmdSmiTemperatureType.GPUBOARD_NODE_RETIMER_X,
                    amdsmi_interface.AmdSmiTemperatureType.GPUBOARD_NODE_OAM_X_IBC,
                    amdsmi_interface.AmdSmiTemperatureType.GPUBOARD_NODE_OAM_X_IBC_2,
                    amdsmi_interface.AmdSmiTemperatureType.GPUBOARD_NODE_OAM_X_VDD18_VR,
                    amdsmi_interface.AmdSmiTemperatureType.GPUBOARD_NODE_OAM_X_04_HBM_B_VR,
                    amdsmi_interface.AmdSmiTemperatureType.GPUBOARD_NODE_OAM_X_04_HBM_D_VR,
                    amdsmi_interface.AmdSmiTemperatureType.GPUBOARD_VDDCR_VDD0,
                    amdsmi_interface.AmdSmiTemperatureType.GPUBOARD_VDDCR_VDD1,
                    amdsmi_interface.AmdSmiTemperatureType.GPUBOARD_VDDCR_VDD2,
                    amdsmi_interface.AmdSmiTemperatureType.GPUBOARD_VDDCR_VDD3,
                    amdsmi_interface.AmdSmiTemperatureType.GPUBOARD_VDDCR_SOC_A,
                    amdsmi_interface.AmdSmiTemperatureType.GPUBOARD_VDDCR_SOC_C,
                    amdsmi_interface.AmdSmiTemperatureType.GPUBOARD_VDDCR_SOCIO_A,
                    amdsmi_interface.AmdSmiTemperatureType.GPUBOARD_VDDCR_SOCIO_C,
                    amdsmi_interface.AmdSmiTemperatureType.GPUBOARD_VDD_085_HBM,
                    amdsmi_interface.AmdSmiTemperatureType.GPUBOARD_VDDCR_11_HBM_B,
                    amdsmi_interface.AmdSmiTemperatureType.GPUBOARD_VDDCR_11_HBM_D,
                    amdsmi_interface.AmdSmiTemperatureType.GPUBOARD_VDD_USR,
                    amdsmi_interface.AmdSmiTemperatureType.GPUBOARD_VDDIO_11_E32
                ]
                for type in gpu_board_temp_types:
                    type_name = type.name.replace("GPUBOARD_", "")
                    try:
                        gpu_board_temp_holder = amdsmi_interface.amdsmi_get_temp_metric(args.gpu, type, amdsmi_interface.AmdSmiTemperatureMetric.CURRENT)
                        if gpu_board_temp_holder != "N/A":
                            gpu_board_temp_dict[f'{type_name}'] = self.helpers.unit_format(self.logger,
                                                                                 gpu_board_temp_holder,
                                                                                 '\N{DEGREE SIGN}C')
                        else:
                            gpu_board_temp_dict[f'{type_name}'] = "N/A"
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        gpu_board_temp_dict[f'{type_name}'] = "N/A"
                        logging.debug("Failed to get gpu_board %s for gpu %s | %s", type_name, gpu_id, e.get_error_info())
                # if every value is N/A, then we don't want to display the values unless explicitly told to
                # all args_list being True indicates that this gpu_board is not explicitly called itself
                args_list = [getattr(args, arg) for arg in current_platform_args]
                if all(value == "N/A" for value in gpu_board_temp_dict.values()) and all(arg == True for arg in args_list):
                    gpu_board_temp_dict = {}
                if gpu_board_temp_dict:
                    values_dict['gpu_board'] = {'temperature':gpu_board_temp_dict}
        if "base_board" in current_platform_args:
            if args.base_board:
                base_board_temp_dict = {}
                base_board_temp_types = [
                    amdsmi_interface.AmdSmiTemperatureType.BASEBOARD_UBB_FPGA,
                    amdsmi_interface.AmdSmiTemperatureType.BASEBOARD_UBB_FRONT,
                    amdsmi_interface.AmdSmiTemperatureType.BASEBOARD_UBB_BACK,
                    amdsmi_interface.AmdSmiTemperatureType.BASEBOARD_UBB_OAM7,
                    amdsmi_interface.AmdSmiTemperatureType.BASEBOARD_UBB_IBC,
                    amdsmi_interface.AmdSmiTemperatureType.BASEBOARD_UBB_UFPGA,
                    amdsmi_interface.AmdSmiTemperatureType.BASEBOARD_UBB_OAM1,
                    amdsmi_interface.AmdSmiTemperatureType.BASEBOARD_OAM_0_1_HSC,
                    amdsmi_interface.AmdSmiTemperatureType.BASEBOARD_OAM_2_3_HSC,
                    amdsmi_interface.AmdSmiTemperatureType.BASEBOARD_OAM_4_5_HSC,
                    amdsmi_interface.AmdSmiTemperatureType.BASEBOARD_OAM_6_7_HSC,
                    amdsmi_interface.AmdSmiTemperatureType.BASEBOARD_UBB_FPGA_0V72_VR,
                    amdsmi_interface.AmdSmiTemperatureType.BASEBOARD_UBB_FPGA_3V3_VR,
                    amdsmi_interface.AmdSmiTemperatureType.BASEBOARD_RETIMER_0_1_2_3_1V2_VR,
                    amdsmi_interface.AmdSmiTemperatureType.BASEBOARD_RETIMER_4_5_6_7_1V2_VR,
                    amdsmi_interface.AmdSmiTemperatureType.BASEBOARD_RETIMER_0_1_0V9_VR,
                    amdsmi_interface.AmdSmiTemperatureType.BASEBOARD_RETIMER_4_5_0V9_VR,
                    amdsmi_interface.AmdSmiTemperatureType.BASEBOARD_RETIMER_2_3_0V9_VR,
                    amdsmi_interface.AmdSmiTemperatureType.BASEBOARD_RETIMER_6_7_0V9_VR,
                    amdsmi_interface.AmdSmiTemperatureType.BASEBOARD_OAM_0_1_2_3_3V3_VR,
                    amdsmi_interface.AmdSmiTemperatureType.BASEBOARD_OAM_4_5_6_7_3V3_VR,
                    amdsmi_interface.AmdSmiTemperatureType.BASEBOARD_IBC_HSC,
                    amdsmi_interface.AmdSmiTemperatureType.BASEBOARD_IBC
                ]
                for type in base_board_temp_types:
                    type_name = type.name.replace("BASEBOARD_", "")
                    try:
                        base_board_temp_holder = amdsmi_interface.amdsmi_get_temp_metric(args.gpu, type, amdsmi_interface.AmdSmiTemperatureMetric.CURRENT)
                        if base_board_temp_holder != "N/A":

                            base_board_temp_dict[f'{type_name}'] = self.helpers.unit_format(self.logger,
                                                                                     base_board_temp_holder,
                                                                                     '\N{DEGREE SIGN}C')
                        else:
                            base_board_temp_dict[f'{type_name}'] = "N/A"
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        base_board_temp_dict[f'{type_name}'] = "N/A"
                        logging.debug("Failed to get base_board %s for gpu %s | %s", type_name, gpu_id, e.get_error_info())
                # if every value is N/A, then we don't want to display the values unless explicitly told to
                # all args_list being True indicates that this base_board is not explicitly called itself
                args_list = [getattr(args, arg) for arg in current_platform_args]
                if all(value == "N/A" for value in base_board_temp_dict.values()) and all(arg == True for arg in args_list):
                    base_board_temp_dict = {}
                if base_board_temp_dict:
                    values_dict['base_board'] = {'temperature':base_board_temp_dict}
        if "ecc" in current_platform_args:
            if args.ecc:
                ecc_count = {}
                try:
                    ecc_count = amdsmi_interface.amdsmi_get_gpu_total_ecc_count(args.gpu)
                    ecc_count['total_correctable_count'] = ecc_count.pop('correctable_count')
                    ecc_count['total_uncorrectable_count'] = ecc_count.pop('uncorrectable_count')
                    ecc_count['total_deferred_count'] = ecc_count.pop('deferred_count')
                except amdsmi_exception.AmdSmiLibraryException as e:
                    ecc_count['total_correctable_count'] = "N/A"
                    ecc_count['total_uncorrectable_count'] = "N/A"
                    ecc_count['cache_correctable_count'] = "N/A"
                    ecc_count['cache_uncorrectable_count'] = "N/A"
                    logging.debug("Failed to get total ecc count for gpu %s | %s", gpu_id, e.get_error_info())

                if ecc_count['total_correctable_count'] != "N/A":
                    # Get the UMC error count for getting total cache correctable errors
                    umc_block = amdsmi_interface.AmdSmiGpuBlock['UMC']
                    try:
                        umc_count = amdsmi_interface.amdsmi_get_gpu_ecc_count(args.gpu, umc_block)
                        ecc_count['cache_correctable_count'] = ecc_count['total_correctable_count'] - umc_count['correctable_count']
                        ecc_count['cache_uncorrectable_count'] = ecc_count['total_uncorrectable_count'] - umc_count['uncorrectable_count']
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        ecc_count['cache_correctable_count'] = "N/A"
                        ecc_count['cache_uncorrectable_count'] = "N/A"
                        logging.debug("Failed to get cache ecc count for gpu %s at block %s | %s", gpu_id, umc_block, e.get_error_info())

                values_dict['ecc'] = ecc_count
        if "ecc_blocks" in current_platform_args:
            if args.ecc_blocks:
                ecc_dict = {}
                sysfs_blocks = ["UMC", "SDMA", "GFX", "MMHUB", "PCIE_BIF", "HDP", "XGMI_WAFL"]
                try:
                    ras_states = amdsmi_interface.amdsmi_get_gpu_ras_block_features_enabled(args.gpu)
                    for state in ras_states:
                        # Only add enabled blocks that are also in sysfs
                        if state['status'] == amdsmi_interface.AmdSmiRasErrState.ENABLED.name:
                            gpu_block = amdsmi_interface.AmdSmiGpuBlock[state['block']]
                            # if the blocks are uncountable do not add them at all.
                            if gpu_block.name in sysfs_blocks:
                                try:
                                    ecc_count = amdsmi_interface.amdsmi_get_gpu_ecc_count(args.gpu, gpu_block)
                                    ecc_dict[state['block']] = {'correctable_count' : ecc_count['correctable_count'],
                                                                'uncorrectable_count' : ecc_count['uncorrectable_count'],
                                                                'deferred_count' : ecc_count['deferred_count']}
                                except amdsmi_exception.AmdSmiLibraryException as e:
                                    ecc_dict[state['block']] = {'correctable_count' : "N/A",
                                                                'uncorrectable_count' : "N/A",
                                                                'deferred_count' : "N/A"}
                                    logging.debug("Failed to get ecc count for gpu %s at block %s | %s", gpu_id, gpu_block, e.get_error_info())

                    values_dict['ecc_blocks'] = ecc_dict
                except amdsmi_exception.AmdSmiLibraryException as e:
                    values_dict['ecc_blocks'] = "N/A"
                    logging.debug("Failed to get ecc block features for gpu %s | %s", gpu_id, e.get_error_info())
        if "fan" in current_platform_args:
            if args.fan:
                fan_dict = {"speed" : "N/A",
                            "max" : "N/A",
                            "rpm" : "N/A",
                            "usage" : "N/A"}

                try:
                    fan_speed = amdsmi_interface.amdsmi_get_gpu_fan_speed(args.gpu, 0)
                    fan_dict["speed"] = fan_speed
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug("Failed to get fan speed for gpu %s | %s", args.gpu, e.get_error_info())

                try:
                    fan_max = amdsmi_interface.amdsmi_get_gpu_fan_speed_max(args.gpu, 0)
                    fan_usage = "N/A"
                    if fan_max > 0 and fan_dict["speed"] != "N/A":
                        fan_usage = round((float(fan_speed) / float(fan_max)) * 100, 2)
                        fan_usage_unit = '%'
                        if self.logger.is_human_readable_format():
                            fan_usage = f"{fan_usage} {fan_usage_unit}"
                        if self.logger.is_json_format():
                            fan_usage = {"value" : fan_usage,
                                         "unit" : fan_usage_unit}
                    fan_dict["max"] = fan_max
                    fan_dict["usage"] = fan_usage
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug("Failed to get fan max speed for gpu %s | %s", args.gpu, e.get_error_info())

                try:
                    fan_rpm = amdsmi_interface.amdsmi_get_gpu_fan_rpms(args.gpu, 0)
                    fan_dict["rpm"] = fan_rpm
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug("Failed to get fan rpms for gpu %s | %s", args.gpu, e.get_error_info())

                values_dict["fan"] = fan_dict
        if "voltage_curve" in current_platform_args:
            if args.voltage_curve:
                # Populate N/A values per voltage point
                voltage_point_dict = {}
                for point in range(amdsmi_interface.AMDSMI_NUM_VOLTAGE_CURVE_POINTS):
                    voltage_point_dict[f'point_{point}_frequency'] = "N/A"
                    voltage_point_dict[f'point_{point}_voltage'] = "N/A"

                try:
                    od_volt = amdsmi_interface.amdsmi_get_gpu_od_volt_info(args.gpu)
                    logging.debug(f"OD Voltage info: {od_volt}")
                except amdsmi_exception.AmdSmiLibraryException as e:
                    od_volt = "N/A" # Value not used, but needs to not be a dict
                    logging.debug("Failed to get voltage curve for gpu %s | %s", gpu_id, e.get_error_info())

                # Populate voltage point values
                for point in range(amdsmi_interface.AMDSMI_NUM_VOLTAGE_CURVE_POINTS):
                    if isinstance(od_volt, dict):
                        logging.debug(f"point_{point} frequency: {od_volt['curve.vc_points'][point].frequency}")
                        logging.debug(f"point_{point} voltage:   {od_volt['curve.vc_points'][point].voltage}")
                        frequency = int(od_volt["curve.vc_points"][point].frequency / 1000000)
                        voltage = int(od_volt["curve.vc_points"][point].voltage)
                    else:
                        frequency = "N/A"
                        voltage = "N/A"

                    if frequency == 0:
                        frequency = "N/A"

                    if voltage == 0:
                        voltage = "N/A"

                    if frequency != "N/A":
                        frequency = self.helpers.unit_format(self.logger, frequency, "Mhz")

                    if voltage != "N/A":
                        voltage = self.helpers.unit_format(self.logger, voltage, "mV")

                    voltage_point_dict[f'point_{point}_frequency'] = frequency
                    voltage_point_dict[f'point_{point}_voltage'] = voltage

                values_dict['voltage_curve'] = voltage_point_dict
        if "overdrive" in current_platform_args:
            if args.overdrive:
                try:
                    overdrive_level = amdsmi_interface.amdsmi_get_gpu_overdrive_level(args.gpu)
                    od_unit = '%'
                    values_dict['overdrive'] = self.helpers.unit_format(self.logger, overdrive_level, od_unit)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    values_dict['overdrive'] = "N/A"
                    logging.debug("Failed to get gpu overdrive level for gpu %s | %s", gpu_id, e.get_error_info())

                try:
                    mem_overdrive_level = amdsmi_interface.amdsmi_get_gpu_mem_overdrive_level(args.gpu)
                    od_unit = '%'
                    values_dict['mem_overdrive'] = self.helpers.unit_format(self.logger, mem_overdrive_level, od_unit)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    values_dict['mem_overdrive'] = "N/A"
                    logging.debug("Failed to get mem overdrive level for gpu %s | %s", gpu_id, e.get_error_info())
        if "perf_level" in current_platform_args:
            if args.perf_level:
                try:
                    perf_level = amdsmi_interface.amdsmi_get_gpu_perf_level(args.gpu)
                    values_dict['perf_level'] = perf_level
                except amdsmi_exception.AmdSmiLibraryException as e:
                    values_dict['perf_level'] = "N/A"
                    logging.debug("Failed to get perf level for gpu %s | %s", gpu_id, e.get_error_info())
        if "xgmi_err" in current_platform_args:
            if args.xgmi_err:
                try:
                    xgmi_err_status = amdsmi_interface.amdsmi_gpu_xgmi_error_status(args.gpu)
                    values_dict['xgmi_err'] = amdsmi_interface.amdsmi_wrapper.amdsmi_xgmi_status_t__enumvalues[xgmi_err_status]
                except amdsmi_exception.AmdSmiLibraryException as e:
                    values_dict['xgmi_err'] = "N/A"
                    logging.debug("Failed to get xgmi error status for gpu %s | %s", gpu_id, e.get_error_info())
        if "voltage" in current_platform_args:
            if args.voltage:
                voltage_dict = {}
                all_voltage = {
                    "vddboard": amdsmi_interface.AmdSmiVoltageType.VDDBOARD
                }
                for volt_type, volt_metric in all_voltage.items():
                    try:
                        voltage = amdsmi_interface.amdsmi_get_gpu_volt_metric(args.gpu, volt_metric, amdsmi_interface.AmdSmiVoltageMetric.CURRENT)
                        if voltage == 0:
                            voltage = "N/A"
                        voltage_dict[volt_type] = self.helpers.unit_format(self.logger, voltage, "mV")
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        voltage_dict[volt_type] = "N/A"
                        logging.debug("Failed to get voltage for gpu %s | %s", gpu_id, e.get_error_info())
                values_dict['voltage'] = voltage_dict
        if "energy" in current_platform_args:
            if args.energy:
                try:
                    energy_dict = amdsmi_interface.amdsmi_get_energy_count(args.gpu)

                    energy = round(energy_dict["energy_accumulator"] * energy_dict["counter_resolution"], 3)
                    energy /= 1000000
                    energy = round(energy, 3)

                    energy_unit = 'J'
                    if self.logger.is_human_readable_format():
                        energy = f"{energy} {energy_unit}"
                    if self.logger.is_json_format():
                        energy = {"value" : energy,
                                  "unit" : energy_unit}

                    values_dict['energy'] = {"total_energy_consumption" : energy}
                except amdsmi_interface.AmdSmiLibraryException as e:
                    values_dict['energy'] = "N/A"
                    logging.debug("Failed to get energy usage for gpu %s | %s", args.gpu, e.get_error_info())
        if "mem_usage" in current_platform_args:
            if args.mem_usage:
                memory_usage = {'total_vram': "N/A",
                                'used_vram': "N/A",
                                'free_vram': "N/A",
                                'total_visible_vram': "N/A",
                                'used_visible_vram': "N/A",
                                'free_visible_vram': "N/A",
                                'total_gtt': "N/A",
                                'used_gtt': "N/A",
                                'free_gtt': "N/A"}

                # Total VRAM
                try:
                    total_vram = amdsmi_interface.amdsmi_get_gpu_memory_total(args.gpu, amdsmi_interface.AmdSmiMemoryType.VRAM)
                    memory_usage['total_vram'] = total_vram // (1024*1024)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug("Failed to get total VRAM memory for gpu %s | %s", gpu_id, e.get_error_info())

                try:
                    total_visible_vram = amdsmi_interface.amdsmi_get_gpu_memory_total(args.gpu, amdsmi_interface.AmdSmiMemoryType.VIS_VRAM)
                    memory_usage['total_visible_vram'] = total_visible_vram // (1024*1024)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug("Failed to get total VIS VRAM memory for gpu %s | %s", gpu_id, e.get_error_info())

                try:
                    total_gtt = amdsmi_interface.amdsmi_get_gpu_memory_total(args.gpu, amdsmi_interface.AmdSmiMemoryType.GTT)
                    memory_usage['total_gtt'] = total_gtt // (1024*1024)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug("Failed to get total GTT memory for gpu %s | %s", gpu_id, e.get_error_info())

                # Used VRAM
                try:
                    used_vram = amdsmi_interface.amdsmi_get_gpu_memory_usage(args.gpu, amdsmi_interface.AmdSmiMemoryType.VRAM)
                    memory_usage['used_vram'] = used_vram // (1024*1024)

                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug("Failed to get used VRAM memory for gpu %s | %s", gpu_id, e.get_error_info())

                try:
                    used_visible_vram = amdsmi_interface.amdsmi_get_gpu_memory_usage(args.gpu, amdsmi_interface.AmdSmiMemoryType.VIS_VRAM)
                    memory_usage['used_visible_vram'] = used_visible_vram // (1024*1024)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug("Failed to get used VIS VRAM memory for gpu %s | %s", gpu_id, e.get_error_info())

                try:
                    used_gtt = amdsmi_interface.amdsmi_get_gpu_memory_usage(args.gpu, amdsmi_interface.AmdSmiMemoryType.GTT)
                    memory_usage['used_gtt'] = used_gtt // (1024*1024)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug("Failed to get used GTT memory for gpu %s | %s", gpu_id, e.get_error_info())

                # Free VRAM
                if memory_usage['total_vram'] != "N/A" and memory_usage['used_vram'] != "N/A":
                    memory_usage['free_vram'] = memory_usage['total_vram'] - memory_usage['used_vram']

                if memory_usage['total_visible_vram'] != "N/A" and memory_usage['used_visible_vram'] != "N/A":
                    memory_usage['free_visible_vram'] = memory_usage['total_visible_vram'] - memory_usage['used_visible_vram']

                if memory_usage['total_gtt'] != "N/A" and memory_usage['used_gtt'] != "N/A":
                    memory_usage['free_gtt'] = memory_usage['total_gtt'] - memory_usage['used_gtt']

                memory_unit = 'MB'
                for key, value in memory_usage.items():
                    if value != "N/A":
                        if self.logger.is_human_readable_format():
                            memory_usage[key] = f"{value} {memory_unit}"
                        if self.logger.is_json_format():
                            memory_usage[key] = {"value" : value,
                                                "unit" : memory_unit}

                values_dict['mem_usage'] = memory_usage
        if "throttle" in current_platform_args:
            if args.throttle:
                throttle_status = {
                    # Current values - counter/accumulated
                    'accumulation_counter': "N/A",
                    'prochot_accumulated': "N/A",
                    'ppt_accumulated': "N/A",
                    'socket_thermal_accumulated': "N/A",
                    'vr_thermal_accumulated': "N/A",
                    'hbm_thermal_accumulated': "N/A",
                    'gfx_clk_below_host_limit_accumulated': "N/A", # deprecated
                    'gfx_clk_below_host_limit_power_accumulated': "N/A",
                    'gfx_clk_below_host_limit_thermal_accumulated': "N/A",
                    'total_gfx_clk_below_host_limit_accumulated': "N/A",
                    'low_utilization_accumulated': "N/A",

                    # violation status values - active/not active
                    'prochot_violation_status': "N/A",
                    'ppt_violation_status': "N/A",
                    'socket_thermal_violation_status': "N/A",
                    'vr_thermal_violation_status': "N/A",
                    'hbm_thermal_violation_status': "N/A",
                    'gfx_clk_below_host_limit_violation_status': "N/A", # deprecated
                    'gfx_clk_below_host_limit_power_violation_status': "N/A",
                    'gfx_clk_below_host_limit_thermal_violation_status': "N/A",
                    'total_gfx_clk_below_host_limit_violation_status': "N/A",
                    'low_utilization_violation_status': "N/A",

                    # violation activity values - percent
                    'prochot_violation_activity': "N/A",
                    'ppt_violation_activity': "N/A",
                    'socket_thermal_violation_activity': "N/A",
                    'vr_thermal_violation_activity': "N/A",
                    'hbm_thermal_violation_activity': "N/A",
                    'gfx_clk_below_host_limit_violation_activity': "N/A", # deprecated
                    'gfx_clk_below_host_limit_power_violation_activity': "N/A",
                    'gfx_clk_below_host_limit_thermal_violation_activity': "N/A",
                    'total_gfx_clk_below_host_limit_violation_activity': "N/A",
                    'low_utilization_violation_activity': "N/A",
                }

                try:
                    violation_status = amdsmi_interface.amdsmi_get_violation_status(args.gpu)
                    throttle_status['accumulation_counter'] = violation_status['acc_counter']
                    throttle_status['prochot_accumulated'] = violation_status['acc_prochot_thrm']
                    throttle_status['ppt_accumulated'] = violation_status['acc_ppt_pwr']
                    throttle_status['socket_thermal_accumulated'] = violation_status['acc_socket_thrm']
                    throttle_status['vr_thermal_accumulated'] = violation_status['acc_vr_thrm']
                    throttle_status['hbm_thermal_accumulated'] = violation_status['acc_hbm_thrm']
                    throttle_status['gfx_clk_below_host_limit_accumulated'] = violation_status['acc_gfx_clk_below_host_limit'] #deprecated
                    throttle_status['gfx_clk_below_host_limit_power_accumulated'] = self.build_xcp_dict('acc_gfx_clk_below_host_limit_pwr', violation_status, num_partition)
                    throttle_status['gfx_clk_below_host_limit_thermal_accumulated'] = self.build_xcp_dict('acc_gfx_clk_below_host_limit_thm', violation_status, num_partition)
                    throttle_status['total_gfx_clk_below_host_limit_accumulated'] = self.build_xcp_dict('acc_gfx_clk_below_host_limit_total', violation_status, num_partition)
                    throttle_status['low_utilization_accumulated'] = self.build_xcp_dict('acc_low_utilization', violation_status, num_partition)
                    throttle_status['prochot_violation_status'] = self.build_xcp_dict('active_prochot_thrm', violation_status, num_partition)
                    throttle_status['ppt_violation_status'] = self.build_xcp_dict('active_ppt_pwr', violation_status, num_partition)
                    throttle_status['socket_thermal_violation_status'] = self.build_xcp_dict('active_socket_thrm', violation_status, num_partition)
                    throttle_status['vr_thermal_violation_status'] = self.build_xcp_dict('active_vr_thrm', violation_status, num_partition)
                    throttle_status['hbm_thermal_violation_status'] = self.build_xcp_dict('active_hbm_thrm', violation_status, num_partition)
                    throttle_status['gfx_clk_below_host_limit_violation_status'] = self.build_xcp_dict('active_gfx_clk_below_host_limit', violation_status, num_partition) # deprecated
                    throttle_status['gfx_clk_below_host_limit_power_violation_status'] = self.build_xcp_dict('active_gfx_clk_below_host_limit_pwr', violation_status, num_partition)
                    throttle_status['gfx_clk_below_host_limit_thermal_violation_status'] = self.build_xcp_dict('active_gfx_clk_below_host_limit_thm', violation_status, num_partition)
                    throttle_status['total_gfx_clk_below_host_limit_violation_status'] = self.build_xcp_dict('active_gfx_clk_below_host_limit_total', violation_status, num_partition)
                    throttle_status['low_utilization_violation_status'] = self.build_xcp_dict('active_low_utilization', violation_status, num_partition)
                    throttle_status['prochot_violation_activity'] = violation_status['per_prochot_thrm']
                    throttle_status['ppt_violation_activity'] = violation_status['per_ppt_pwr']
                    throttle_status['socket_thermal_violation_activity'] = violation_status['per_socket_thrm']
                    throttle_status['vr_thermal_violation_activity'] = violation_status['per_vr_thrm']
                    throttle_status['hbm_thermal_violation_activity'] = violation_status['per_hbm_thrm']
                    throttle_status['gfx_clk_below_host_limit_violation_activity'] = violation_status['per_gfx_clk_below_host_limit'] # deprecated
                    throttle_status['gfx_clk_below_host_limit_power_violation_activity'] = self.build_xcp_dict('per_gfx_clk_below_host_limit_pwr', violation_status, num_partition)
                    throttle_status['gfx_clk_below_host_limit_thermal_violation_activity'] = self.build_xcp_dict('per_gfx_clk_below_host_limit_thm', violation_status, num_partition)
                    throttle_status['total_gfx_clk_below_host_limit_violation_activity'] = self.build_xcp_dict('per_gfx_clk_below_host_limit_total', violation_status, num_partition)
                    throttle_status['low_utilization_violation_activity'] = self.build_xcp_dict('per_low_utilization', violation_status, num_partition)

                except amdsmi_exception.AmdSmiLibraryException as e:
                    values_dict['throttle'] = throttle_status
                    logging.debug("Failed to get violation status' for gpu %s | %s", gpu_id, e.get_error_info())

                for key, value in throttle_status.items():

                    activity_unit = ''
                    if "_activity" in key:
                        activity_unit = '%'

                    if self.logger.is_human_readable_format():
                        if isinstance(value, (list, dict)):
                            for k, v in value.items():
                                for index, activity in enumerate(v):
                                    value[k][index] = self.helpers.unit_format(self.logger, activity, activity_unit)
                                value[k] = '[' + ", ".join(value[k]) + ']'
                        elif value != "N/A":
                            value = self.helpers.unit_format(self.logger, value, activity_unit)
                    if self.logger.is_json_format():
                        if isinstance(value, (list, dict)):
                            for k, v in value.items():
                                for index, activity in enumerate(v):
                                    value[k][index] = self.helpers.unit_format(self.logger, activity, activity_unit)
                        elif value != "N/A":
                            throttle_status[key] = self.helpers.unit_format(self.logger, value, activity_unit)
                values_dict['throttle'] = throttle_status

        # Store timestamp first if watching_output is enabled
        if watching_output:
            self.logger.store_output(args.gpu, 'timestamp', int(time.time()))
        self.logger.store_output(args.gpu, 'values', values_dict)
        self.logger.store_gpu_json_output.append(values_dict)

        if multiple_devices:
            self.logger.store_multiple_device_output()
            return # Skip printing when there are multiple devices

        if not self.logger.is_json_format():
            self.logger.print_output(watching_output=watching_output)

        if watching_output: # End of single gpu add to watch_output
            self.logger.store_watch_output(multiple_device_enabled=False)


    def metric_cpu(self, args, multiple_devices=False, cpu=None, cpu_power_metrics=None, cpu_prochot=None,
                   cpu_freq_metrics=None, cpu_c0_res=None, cpu_lclk_dpm_level=None,
                  cpu_pwr_svi_telemetry_rails=None, cpu_io_bandwidth=None, cpu_xgmi_bandwidth=None,
                   cpu_metrics_ver=None, cpu_metrics_table=None, cpu_socket_energy=None,
                   cpu_ddr_bandwidth=None, cpu_temp=None, cpu_dimm_temp_range_rate=None,
                   cpu_dimm_pow_consumption=None, cpu_dimm_thermal_sensor=None):
        """Get Metric information for target cpu

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            cpu (cpu_handle, optional): device_handle for target device. Defaults to None.
            cpu_power_metrics (bool, optional): Value override for args.cpu_power_metrics. Defaults to None
            cpu_prochot (bool, optional): Value override for args.cpu_prochot. Defaults to None.
            cpu_freq_metrics (bool, optional): Value override for args.cpu_freq_metrics. Defaults to None.
            cpu_c0_res (bool, optional): Value override for args.cpu_c0_res. Defaults to None
            cpu_lclk_dpm_level (list, optional): Value override for args.cpu_lclk_dpm_level. Defaults to None
            cpu_pwr_svi_telemetry_rails (list, optional): value override for args.cpu_pwr_svi_telemetry_rails. Defaults to None
            cpu_io_bandwidth (list, optional): value override for args.cpu_io_bandwidth. Defaults to None
            cpu_xgmi_bandwidth (list, optional): value override for args.cpu_xgmi_bandwidth. Defaults to None
            cpu_metrics_ver (bool, optional): Value override for args.cpu_metrics_ver. Defaults to None
            cpu_metrics_table (bool, optional): Value override for args.cpu_metrics_table. Defaults to None
            cpu_socket_energy (bool, optional): Value override for args.cpu_socket_energy. Defaults to None
            cpu_ddr_bandwidth (bool, optional): Value override for args.cpu_ddr_bandwidth. Defaults to None
            cpu_temp (bool, optional): Value override for args.cpu_temp. Defaults to None
            cpu_dimm_temp_range_rate (list, optional): Dimm address. Value override for args.cpu_dimm_temp_range_rate. Defaults to None
            cpu_dimm_pow_consumption (list, optional): Dimm address. Value override for args.cpu_dimm_pow_consumption. Defaults to None
            cpu_dimm_thermal_sensor (list, optional): Dimm address. Value override for args.cpu_dimm_thermal_sensor. Defaults to None

        Returns:
            None: Print output via AMDSMILogger to destination
        """

        if cpu:
            args.cpu = cpu
        if cpu_power_metrics:
            args.cpu_power_metrics = cpu_power_metrics
        if cpu_prochot:
            args.cpu_prochot = cpu_prochot
        if cpu_freq_metrics:
            args.cpu_freq_metrics = cpu_freq_metrics
        if cpu_c0_res:
            args.cpu_c0_res = cpu_c0_res
        if cpu_lclk_dpm_level:
            args.cpu_lclk_dpm_level = cpu_lclk_dpm_level
        if cpu_pwr_svi_telemetry_rails:
            args.cpu_pwr_svi_telemetry_rails = cpu_pwr_svi_telemetry_rails
        if cpu_io_bandwidth:
            args.cpu_io_bandwidth = cpu_io_bandwidth
        if cpu_xgmi_bandwidth:
            args.cpu_xgmi_bandwidth = cpu_xgmi_bandwidth
        if cpu_metrics_ver:
            args.cpu_metrics_ver = cpu_metrics_ver
        if cpu_metrics_table:
            args.cpu_metrics_table = cpu_metrics_table
        if cpu_socket_energy:
            args.cpu_socket_energy = cpu_socket_energy
        if cpu_ddr_bandwidth:
            args.cpu_ddr_bandwidth = cpu_ddr_bandwidth
        if cpu_temp:
            args.cpu_temp = cpu_temp
        if cpu_dimm_temp_range_rate:
            args.cpu_dimm_temp_range_rate = cpu_dimm_temp_range_rate
        if cpu_dimm_pow_consumption:
            args.cpu_dimm_pow_consumption = cpu_dimm_pow_consumption
        if cpu_dimm_thermal_sensor:
            args.cpu_dimm_thermal_sensor = cpu_dimm_thermal_sensor

        #store cpu args that are applicable to the current platform
        curr_platform_cpu_args = ["cpu_power_metrics", "cpu_prochot", "cpu_freq_metrics",
                                  "cpu_c0_res", "cpu_lclk_dpm_level", "cpu_pwr_svi_telemetry_rails",
                                  "cpu_io_bandwidth", "cpu_xgmi_bandwidth", "cpu_metrics_ver",
                                  "cpu_metrics_table", "cpu_socket_energy", "cpu_ddr_bandwidth",
                                  "cpu_temp", "cpu_dimm_temp_range_rate", "cpu_dimm_pow_consumption",
                                  "cpu_dimm_thermal_sensor"]
        curr_platform_cpu_values = [args.cpu_power_metrics, args.cpu_prochot, args.cpu_freq_metrics,
                                    args.cpu_c0_res, args.cpu_lclk_dpm_level, args.cpu_pwr_svi_telemetry_rails,
                                    args.cpu_io_bandwidth, args.cpu_xgmi_bandwidth, args.cpu_metrics_ver,
                                    args.cpu_metrics_table, args.cpu_socket_energy, args.cpu_ddr_bandwidth,
                                    args.cpu_temp, args.cpu_dimm_temp_range_rate, args.cpu_dimm_pow_consumption,
                                    args.cpu_dimm_thermal_sensor]

        # Handle No CPU passed (fall back as this should be defined in metric())
        if args.cpu == None:
            args.cpu = self.cpu_handles

        if not any(curr_platform_cpu_values):
            for arg in curr_platform_cpu_args:
                if arg not in("cpu_lclk_dpm_level", "cpu_io_bandwidth", "cpu_xgmi_bandwidth",
                              "cpu_dimm_temp_range_rate", "cpu_dimm_pow_consumption", "cpu_dimm_thermal_sensor"):
                    setattr(args, arg, True)

        handled_multiple_cpus, device_handle = self.helpers.handle_cpus(args,
                                                                        self.logger,
                                                                        self.metric_cpu)
        if handled_multiple_cpus:
            return # This function is recursive
        args.cpu = device_handle
        # get cpu id for logging
        cpu_id = self.helpers.get_cpu_id_from_device_handle(args.cpu)
        logging.debug(f"Metric Arg information for CPU {cpu_id} on {self.helpers.os_info()}")

        static_dict = {}
        if self.logger.is_json_format():
            static_dict['cpu'] = int(cpu_id)
        if args.cpu_power_metrics:
            static_dict["power_metrics"] = {}
            try:
                soc_pow = amdsmi_interface.amdsmi_get_cpu_socket_power(args.cpu)
                static_dict["power_metrics"]["socket power"] = soc_pow
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["power_metrics"]["socket power"] = "N/A"
                logging.debug("Failed to get socket power for cpu %s | %s", cpu_id, e.get_error_info())

            try:
                soc_pwr_limit = amdsmi_interface.amdsmi_get_cpu_socket_power_cap(args.cpu)
                static_dict["power_metrics"]["socket power limit"] = soc_pwr_limit
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["power_metrics"]["socket power limit"] = "N/A"
                logging.debug("Failed to get socket power limit for cpu %s | %s", cpu_id, e.get_error_info())

            try:
                soc_max_pwr_limit = amdsmi_interface.amdsmi_get_cpu_socket_power_cap_max(args.cpu)
                static_dict["power_metrics"]["socket max power limit"] = soc_max_pwr_limit
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["power_metrics"]["socket max power limit"] = "N/A"
                logging.debug("Failed to get max socket power limit for cpu %s | %s", cpu_id, e.get_error_info())
        if args.cpu_prochot:
            static_dict["prochot"] = {}
            try:
                proc_status = amdsmi_interface.amdsmi_get_cpu_prochot_status(args.cpu)
                static_dict["prochot"]["prochot_status"] = proc_status
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["prochot"]["prochot_status"] = "N/A"
                logging.debug("Failed to get prochot status for cpu %s | %s", cpu_id, e.get_error_info())
        if args.cpu_freq_metrics:
            static_dict["freq_metrics"] = {}
            try:
                fclk_mclk = amdsmi_interface.amdsmi_get_cpu_fclk_mclk(args.cpu)
                static_dict["freq_metrics"]["fclkmemclk"] = fclk_mclk
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["freq_metrics"]["fclkmemclk"] = "N/A"
                logging.debug("Failed to get current fclkmemclk freq for cpu %s | %s", cpu_id, e.get_error_info())

            try:
                cclk_freq = amdsmi_interface.amdsmi_get_cpu_cclk_limit(args.cpu)
                static_dict["freq_metrics"]["cclkfreqlimit"] = cclk_freq
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["freq_metrics"]["cclkfreqlimit"] = "N/A"
                logging.debug("Failed to get current cclk freq for cpu %s | %s", cpu_id, e.get_error_info())

            try:
                soc_cur_freq_limit = amdsmi_interface.amdsmi_get_cpu_socket_current_active_freq_limit(args.cpu)
                static_dict["freq_metrics"]["soc_current_active_freq_limit"] = soc_cur_freq_limit
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["freq_metrics"]["soc_current_active_freq_limit"] = "N/A"
                logging.debug("Failed to get socket current freq limit for cpu %s | %s", cpu_id, e.get_error_info())

            try:
                soc_freq_range = amdsmi_interface.amdsmi_get_cpu_socket_freq_range(args.cpu)
                static_dict["freq_metrics"]["soc_freq_range"] = soc_freq_range
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["freq_metrics"]["soc_freq_range"] = "N/A"
                logging.debug("Failed to get socket freq range for cpu %s | %s", cpu_id, e.get_error_info())
        if args.cpu_c0_res:
            static_dict["c0_residency"] = {}
            try:
                residency = amdsmi_interface.amdsmi_get_cpu_socket_c0_residency(args.cpu)
                static_dict["c0_residency"]["residency"] = residency
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["c0_residency"]["residency"] = "N/A"
                logging.debug("Failed to get C0 residency for cpu %s | %s", cpu_id, e.get_error_info())
        if args.cpu_lclk_dpm_level:
            static_dict["socket_dpm"] = {}
            try:
                dpm_val = amdsmi_interface.amdsmi_get_cpu_socket_lclk_dpm_level(args.cpu,
                                                                                args.cpu_lclk_dpm_level[0][0])
                static_dict["socket_dpm"]["dpml_level_range"] = dpm_val
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["socket_dpm"]["dpml_level_range"] = "N/A"
                logging.debug("Failed to get socket dpm level range for cpu %s | %s", cpu_id, e.get_error_info())
        if args.cpu_pwr_svi_telemetry_rails:
            static_dict["svi_telemetry_all_rails"] = {}
            try:
                power = amdsmi_interface.amdsmi_get_cpu_pwr_svi_telemetry_all_rails(args.cpu)
                static_dict["svi_telemetry_all_rails"]["power"] = power
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["c0_residency"]["residency"] = "N/A"
                logging.debug("Failed to get svi telemetry all rails for cpu %s | %s", cpu_id, e.get_error_info())
        if args.cpu_io_bandwidth:
            static_dict["io_bandwidth"] = {}
            try:
                bandwidth = amdsmi_interface.amdsmi_get_cpu_current_io_bandwidth(args.cpu,
                                                                                    int(args.cpu_io_bandwidth[0][0]),
                                                                                    args.cpu_io_bandwidth[0][1])
                static_dict["io_bandwidth"]["band_width"] = bandwidth
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["io_bandwidth"]["band_width"] = "N/A"
                logging.debug("Failed to get io bandwidth for cpu %s | %s", cpu_id, e.get_error_info())
        if args.cpu_xgmi_bandwidth:
            static_dict["xgmi_bandwidth"] = {}
            try:
                bandwidth = amdsmi_interface.amdsmi_get_cpu_current_xgmi_bw(args.cpu,
                                                                            int(args.cpu_xgmi_bandwidth[0][0]),
                                                                            args.cpu_xgmi_bandwidth[0][1])
                static_dict["xgmi_bandwidth"]["band_width"] = bandwidth
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["xgmi_bandwidth"]["band_width"] = "N/A"
                logging.debug("Failed to get xgmi bandwidth for cpu %s | %s", cpu_id, e.get_error_info())
        if args.cpu_metrics_ver:
            static_dict["metric_version"] = {}
            try:
                version = amdsmi_interface.amdsmi_get_hsmp_metrics_table_version(args.cpu)
                static_dict["metric_version"]["version"] = version
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["metric_version"]["version"] = "N/A"
                logging.debug("Failed to get metrics table version for cpu %s | %s", cpu_id, e.get_error_info())
        if args.cpu_metrics_table:
            static_dict["metrics_table"] = {}
            try:
                cpu_fam = amdsmi_interface.amdsmi_get_cpu_family()
                static_dict["metrics_table"]["cpu_family"] = cpu_fam
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["metrics_table"]["cpu_family"] = "N/A"
                logging.debug("Failed to get cpu family | %s", e.get_error_info())
            try:
                cpu_mod = amdsmi_interface.amdsmi_get_cpu_model()
                static_dict["metrics_table"]["cpu_model"] = cpu_mod
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["metrics_table"]["cpu_model"] = "N/A"
                logging.debug("Failed to get cpu model | %s", e.get_error_info())
            try:
                cpu_metrics_table = amdsmi_interface.amdsmi_get_hsmp_metrics_table(args.cpu)
                static_dict["metrics_table"]["response"] = cpu_metrics_table
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["metrics_table"]["response"] = "N/A"
                logging.debug("Failed to get metrics table for cpu %s | %s", cpu_id, e.get_error_info())
        if args.cpu_socket_energy:
            static_dict["socket_energy"] = {}
            try:
                energy = amdsmi_interface.amdsmi_get_cpu_socket_energy(args.cpu)
                static_dict["socket_energy"]["response"] = energy
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["socket_energy"]["response"] = "N/A"
                logging.debug("Failed to get socket energy for cpu %s | %s", cpu_id, e.get_error_info())
        if args.cpu_ddr_bandwidth:
            static_dict["ddr_bandwidth"] = {}
            try:
                resp = amdsmi_interface.amdsmi_get_cpu_ddr_bw(args.cpu)
                static_dict["ddr_bandwidth"]["response"] = resp
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["ddr_bandwidth"]["response"] = "N/A"
                logging.debug("Failed to get ddr bandwdith for cpu %s | %s", cpu_id, e.get_error_info())
        if args.cpu_temp:
            static_dict["cpu_temp"] = {}
            try:
                resp = amdsmi_interface.amdsmi_get_cpu_socket_temperature(args.cpu)
                static_dict["cpu_temp"]["response"] = resp
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["cpu_temp"]["response"] = "N/A"
                logging.debug("Failed to get cpu temperature for cpu %s | %s", cpu_id, e.get_error_info())
        if args.cpu_dimm_temp_range_rate:
            static_dict["dimm_temp_range_rate"] = {}
            try:
                resp = amdsmi_interface.amdsmi_get_cpu_dimm_temp_range_and_refresh_rate(args.cpu, args.cpu_dimm_temp_range_rate[0][0])
                static_dict["dimm_temp_range_rate"]["response"] = resp
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["dimm_temp_range_rate"]["response"] = "N/A"
                logging.debug("Failed to get dimm temperature range and refresh rate for cpu %s | %s", cpu_id, e.get_error_info())
        if args.cpu_dimm_pow_consumption:
            static_dict["dimm_pow_consumption"] = {}
            try:
                resp = amdsmi_interface.amdsmi_get_cpu_dimm_power_consumption(args.cpu, args.cpu_dimm_pow_consumption[0][0])
                static_dict["dimm_pow_consumption"]["response"] = resp
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["dimm_pow_consumption"]["response"] = "N/A"
                logging.debug("Failed to get dimm temperature range and refresh rate for cpu %s | %s", cpu_id, e.get_error_info())
        if args.cpu_dimm_thermal_sensor:
            static_dict["dimm_thermal_sensor"] = {}
            try:
                resp = amdsmi_interface.amdsmi_get_cpu_dimm_thermal_sensor(args.cpu, args.cpu_dimm_thermal_sensor[0][0])
                static_dict["dimm_thermal_sensor"]["response"] = resp
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["dimm_thermal_sensor"]["response"] = "N/A"
                logging.debug("Failed to get dimm temperature range and refresh rate for cpu %s | %s", cpu_id, e.get_error_info())

        multiple_devices_csv_override = False
        if not self.logger.is_json_format():
            self.logger.store_cpu_output(args.cpu, 'values', static_dict)
        else:
            self.logger.store_cpu_json_output.append(static_dict)
        if multiple_devices:
            self.logger.store_multiple_device_output()
            return # Skip printing when there are multiple devices
        if not self.logger.is_json_format():
            self.logger.print_output(multiple_device_enabled=multiple_devices_csv_override)


    def metric_core(self, args, multiple_devices=False, core=None, core_boost_limit=None,
                    core_curr_active_freq_core_limit=None, core_energy=None):
        """Get Static information for target core

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            core (device_handle, optional): device_handle for target core. Defaults to None.
            core_boost_limit (bool, optional): Value override for args.core_boost_limit. Defaults to None
            core_curr_active_freq_core_limit (bool, optional): Value override for args.core_curr_active_freq_core_limit. Defaults to None
            core_energy (bool, optional): Value override for args.core_energy. Defaults to None
        Returns:
            None: Print output via AMDSMILogger to destination
        """
        if core:
            args.core = core
        if core_boost_limit:
            args.core_boost_limit = core_boost_limit
        if core_curr_active_freq_core_limit:
            args.core_curr_active_freq_core_limit = core_curr_active_freq_core_limit
        if core_energy:
            args.core_energy = core_energy

        #store core args that are applicable to the current platform
        curr_platform_core_args = ["core_boost_limit", "core_curr_active_freq_core_limit", "core_energy"]
        curr_platform_core_values = [args.core_boost_limit, args.core_curr_active_freq_core_limit, args.core_energy]

        # Handle No cores passed
        if args.core == None:
            args.core = self.core_handles

        if not any(curr_platform_core_values):
            for arg in curr_platform_core_args:
                setattr(args, arg, True)

        handled_multiple_cores, device_handle = self.helpers.handle_cores(args,
                                                                        self.logger,
                                                                        self.metric_core)
        if handled_multiple_cores:
            return # This function is recursive
        args.core = device_handle
        # get core id for logging
        core_id = self.helpers.get_core_id_from_device_handle(args.core)
        logging.debug(f"Static Arg information for Core {core_id} on {self.helpers.os_info()}")

        static_dict = {}
        if self.logger.is_json_format():
            static_dict['core'] = int(core_id)
        if args.core_boost_limit:
            static_dict["boost_limit"] ={}

            try:
                core_boost_limit = amdsmi_interface.amdsmi_get_cpu_core_boostlimit(args.core)
                static_dict["boost_limit"]["value"] = core_boost_limit
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["boost_limit"]["value"] = "N/A"
                logging.debug("Failed to get core boost limit for core %s | %s", core_id, e.get_error_info())
        if args.core_curr_active_freq_core_limit:
            static_dict["curr_active_freq_core_limit"] = {}

            try:
                freq = amdsmi_interface.amdsmi_get_cpu_core_current_freq_limit(args.core)
                static_dict["curr_active_freq_core_limit"]["value"] = freq
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["curr_active_freq_core_limit"]["value"] = "N/A"
                logging.debug("Failed to get current active frequency core for core %s | %s", core_id, e.get_error_info())
        if args.core_energy:
            static_dict["core_energy"] ={}
            try:
                energy = amdsmi_interface.amdsmi_get_cpu_core_energy(args.core)
                static_dict["core_energy"]["value"] = energy
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["core_energy"]["value"] = "N/A"
                logging.debug("Failed to get core energy for core %s | %s", core_id, e.get_error_info())

        multiple_devices_csv_override = False
        if not self.logger.is_json_format():
            self.logger.store_core_output(args.core, 'values', static_dict)
        else:
            self.logger.store_core_json_output.append(static_dict)
        if multiple_devices:
            self.logger.store_multiple_device_output()
            return # Skip printing when there are multiple devices
        if not self.logger.is_json_format():
            self.logger.print_output(multiple_device_enabled=multiple_devices_csv_override)


    def metric(self, args, multiple_devices=False, watching_output=False, gpu=None,
                usage=None, watch=None, watch_time=None, iterations=None, power=None,
                clock=None, temperature=None, ecc=None, ecc_blocks=None, pcie=None,
                fan=None, voltage_curve=None, overdrive=None, perf_level=None,
                xgmi_err=None, energy=None, mem_usage=None, voltage=None, schedule=None,
                guard=None, guest_data=None, fb_usage=None, xgmi=None,
                cpu=None, cpu_power_metrics=None, cpu_prochot=None, cpu_freq_metrics=None,
                cpu_c0_res=None, cpu_lclk_dpm_level=None, cpu_pwr_svi_telemetry_rails=None,
                cpu_io_bandwidth=None, cpu_xgmi_bandwidth=None, cpu_metrics_ver=None,
                cpu_metrics_table=None, cpu_socket_energy=None, cpu_ddr_bandwidth=None,
                cpu_temp=None, cpu_dimm_temp_range_rate=None, cpu_dimm_pow_consumption=None,
                cpu_dimm_thermal_sensor=None,
                core=None, core_boost_limit=None, core_curr_active_freq_core_limit=None,
                core_energy=None, throttle=None, base_board=None, gpu_board=None):
        """Get Metric information for target gpu

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            watching_output (bool, optional): True if watch argument has been set. Defaults to False.
            gpu (device_handle, optional): device_handle for target device. Defaults to None.
            usage (bool, optional): Value override for args.usage. Defaults to None.
            watch (Positive int, optional): Value override for args.watch. Defaults to None.
            watch_time (Positive int, optional): Value override for args.watch_time. Defaults to None.
            iterations (Positive int, optional): Value override for args.iterations. Defaults to None.
            power (bool, optional): Value override for args.power. Defaults to None.
            clock (bool, optional): Value override for args.clock. Defaults to None.
            temperature (bool, optional): Value override for args.temperature. Defaults to None.
            ecc (bool, optional): Value override for args.ecc. Defaults to None.
            ecc_blocks (bool, optional): Value override for args.ecc. Defaults to None.
            pcie (bool, optional): Value override for args.pcie. Defaults to None.
            fan (bool, optional): Value override for args.fan. Defaults to None.
            voltage_curve (bool, optional): Value override for args.voltage_curve. Defaults to None.
            overdrive (bool, optional): Value override for args.overdrive. Defaults to None.
            perf_level (bool, optional): Value override for args.perf_level. Defaults to None.
            xgmi_err (bool, optional): Value override for args.xgmi_err. Defaults to None.
            energy (bool, optional): Value override for args.energy. Defaults to None.
            mem_usage (bool, optional): Value override for args.mem_usage. Defaults to None.
            voltage (bool, optional): Value override for args.voltage. Defaults to None.
            schedule (bool, optional): Value override for args.schedule. Defaults to None.
            guard (bool, optional): Value override for args.guard. Defaults to None.
            guest_data (bool, optional): Value override for args.guest_data. Defaults to None.
            fb_usage (bool, optional): Value override for args.fb_usage. Defaults to None.
            xgmi (bool, optional): Value override for args.xgmi. Defaults to None.

            cpu (cpu_handle, optional): device_handle for target device. Defaults to None.
            cpu_power_metrics (bool, optional): Value override for args.cpu_power_metrics. Defaults to None
            cpu_prochot (bool, optional): Value override for args.cpu_prochot. Defaults to None.
            cpu_freq_metrics (bool, optional): Value override for args.cpu_freq_metrics. Defaults to None.
            cpu_c0_res (bool, optional): Value override for args.cpu_c0_res. Defaults to None
            cpu_lclk_dpm_level (list, optional): Value override for args.cpu_lclk_dpm_level. Defaults to None
            cpu_pwr_svi_telemetry_rails (list, optional): value override for args.cpu_pwr_svi_telemetry_rails. Defaults to None
            cpu_io_bandwidth (list, optional): value override for args.cpu_io_bandwidth. Defaults to None
            cpu_xgmi_bandwidth (list, optional): value override for args.cpu_xgmi_bandwidth. Defaults to None
            cpu_metrics_ver (bool, optional): Value override for args.cpu_metrics_ver. Defaults to None
            cpu_metrics_table (bool, optional): Value override for args.cpu_metrics_table. Defaults to None
            cpu_socket_energy (bool, optional): Value override for args.cpu_socket_energy. Defaults to None
            cpu_ddr_bandwidth (bool, optional): Value override for args.cpu_ddr_bandwidth. Defaults to None
            cpu_temp (bool, optional): Value override for args.cpu_temp. Defaults to None
            cpu_dimm_temp_range_rate (list, optional): Dimm address. Value override for args.cpu_dimm_temp_range_rate. Defaults to None
            cpu_dimm_pow_consumption (list, optional): Dimm address. Value override for args.cpu_dimm_pow_consumption. Defaults to None
            cpu_dimm_thermal_sensor (list, optional): Dimm address. Value override for args.cpu_dimm_thermal_sensor. Defaults to None

            core (device_handle, optional): device_handle for target core. Defaults to None.
            core_boost_limit (bool, optional): Value override for args.core_boost_limit. Defaults to None
            core_curr_active_freq_core_limit (bool, optional): Value override for args.core_curr_active_freq_core_limit. Defaults to None
            core_energy (bool, optional): Value override for args.core_energy. Defaults to None

        Raises:
            IndexError: Index error if gpu list is empty

        Returns:
            None: Print output via AMDSMILogger to destination
        """
        # TODO Move watch logic into here and make it driver agnostic or enable it for CPU arguments
        # Mutually exclusive args
        if gpu:
            args.gpu = gpu
        if cpu:
            args.cpu = cpu
        if core:
            args.core = core

        # Check if a GPU argument has been set
        gpu_args_enabled = False
        gpu_attributes = ["usage", "watch", "watch_time", "iterations", "power", "clock",
                          "temperature", "ecc", "ecc_blocks", "pcie", "fan", "voltage_curve",
                          "overdrive", "perf_level", "xgmi_err", "energy", "mem_usage", "voltage", "schedule",
                          "guard", "guest_data", "fb_usage", "xgmi", "throttle", "base_board", "gpu_board"]
        for attr in gpu_attributes:
            if hasattr(args, attr):
                if getattr(args, attr):
                    gpu_args_enabled = True
                    break

        # Check if a CPU argument has been set
        cpu_args_enabled = False
        cpu_attributes = ["cpu_power_metrics", "cpu_prochot", "cpu_freq_metrics", "cpu_c0_res",
                          "cpu_lclk_dpm_level", "cpu_pwr_svi_telemetry_rails", "cpu_io_bandwidth",
                          "cpu_xgmi_bandwidth", "cpu_metrics_ver", "cpu_metrics_table",
                          "cpu_socket_energy", "cpu_ddr_bandwidth", "cpu_temp", "cpu_dimm_temp_range_rate",
                          "cpu_dimm_pow_consumption", "cpu_dimm_thermal_sensor"]
        for attr in cpu_attributes:
            if hasattr(args, attr):
                if getattr(args, attr):
                    cpu_args_enabled = True
                    break

        # Check if a Core argument has been set
        core_args_enabled = False
        core_attributes = ["core_boost_limit", "core_curr_active_freq_core_limit", "core_energy"]
        for attr in core_attributes:
            if hasattr(args, attr):
                if getattr(args, attr):
                    core_args_enabled = True
                    break

        # Handle CPU and GPU driver intialization cases
        if self.helpers.is_amd_hsmp_initialized() and self.helpers.is_amdgpu_initialized():

            logging.debug("gpu_args_enabled: %s, cpu_args_enabled: %s, core_args_enabled: %s",
                            gpu_args_enabled, cpu_args_enabled, core_args_enabled)
            logging.debug("args.gpu: %s, args.cpu: %s, args.core: %s", args.gpu, args.cpu, args.core)

            # If a GPU or CPU argument is provided only print out the specified device.
            if args.cpu == None and args.gpu == None and args.core == None:
                # If no args are set, print out all CPU, GPU, and Core metrics info
                if not gpu_args_enabled and not cpu_args_enabled and not core_args_enabled:
                    args.cpu = self.cpu_handles
                    args.gpu = self.device_handles
                    args.core = self.core_handles

            # Handle cases where the user has only specified an argument and no specific device
            if args.gpu == None and gpu_args_enabled:
                args.gpu = self.device_handles
            if args.cpu == None and cpu_args_enabled:
                args.cpu = self.cpu_handles
            if args.core == None and core_args_enabled:
                args.core = self.core_handles

            # Print out CPU first
            if args.cpu:
                self.metric_cpu(args, multiple_devices, cpu, cpu_power_metrics, cpu_prochot,
                                cpu_freq_metrics, cpu_c0_res, cpu_lclk_dpm_level,
                                cpu_pwr_svi_telemetry_rails, cpu_io_bandwidth, cpu_xgmi_bandwidth,
                                cpu_metrics_ver, cpu_metrics_table, cpu_socket_energy,
                                cpu_ddr_bandwidth, cpu_temp, cpu_dimm_temp_range_rate,
                                cpu_dimm_pow_consumption, cpu_dimm_thermal_sensor)
            if args.core:
                self.logger.output = {}
                self.logger.clear_multiple_devices_output()
                self.metric_core(args, multiple_devices, core, core_boost_limit,
                                     core_curr_active_freq_core_limit, core_energy)
            if args.gpu:
                self.logger.output = {}
                self.logger.clear_multiple_devices_output()
                self.metric_gpu(args, multiple_devices, watching_output, gpu,
                                usage, watch, watch_time, iterations, power,
                                clock, temperature, ecc, ecc_blocks, pcie,
                                fan, voltage_curve, overdrive, perf_level,
                                xgmi_err, energy, mem_usage, voltage, schedule,
                                guard, guest_data, fb_usage, xgmi, throttle,
                                base_board, gpu_board)
        elif self.helpers.is_amd_hsmp_initialized(): # Only CPU is initialized
            if args.cpu == None and args.core == None:
                # If no args are set, print out all CPU and Core metrics info
                if not cpu_args_enabled and not core_args_enabled:
                    args.cpu = self.cpu_handles
                    args.core = self.core_handles

            if args.cpu == None and cpu_args_enabled:
                args.cpu = self.cpu_handles
            if args.core == None and core_args_enabled:
                args.core = self.core_handles

            if args.cpu:
                self.metric_cpu(args, multiple_devices, cpu, cpu_power_metrics, cpu_prochot,
                                cpu_freq_metrics, cpu_c0_res, cpu_lclk_dpm_level,
                                cpu_pwr_svi_telemetry_rails, cpu_io_bandwidth, cpu_xgmi_bandwidth,
                                cpu_metrics_ver, cpu_metrics_table, cpu_socket_energy,
                                cpu_ddr_bandwidth, cpu_temp, cpu_dimm_temp_range_rate,
                                cpu_dimm_pow_consumption, cpu_dimm_thermal_sensor)
            if args.core:
                self.logger.output = {}
                self.logger.clear_multiple_devices_output()
                self.metric_core(args, multiple_devices, core, core_boost_limit,
                                     core_curr_active_freq_core_limit, core_energy)
        elif self.helpers.is_amdgpu_initialized(): # Only GPU is initialized
            if args.gpu == None:
                args.gpu = self.device_handles

            self.logger.clear_multiple_devices_output()
            self.metric_gpu(args, multiple_devices, watching_output, gpu,
                                usage, watch, watch_time, iterations, power,
                                clock, temperature, ecc, ecc_blocks, pcie,
                                fan, voltage_curve, overdrive, perf_level,
                                xgmi_err, energy, mem_usage, voltage, schedule, throttle,
                                base_board, gpu_board)
        if self.logger.is_json_format():
            self.logger.combine_arrays_to_json()


    def process(self, args, multiple_devices=False, watching_output=False,
                gpu=None, general=None, engine=None, pid=None, name=None,
                watch=None, watch_time=None, iterations=None):
        """Get Process Information from the target GPU

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            watching_output (bool, optional): True if watch argument has been set. Defaults to False.
            gpu (device_handle, optional): device_handle for target device. Defaults to None.
            general (bool, optional): Value override for args.general. Defaults to None.
            engine (bool, optional): Value override for args.engine. Defaults to None.
            pid (Positive int, optional): Value override for args.pid. Defaults to None.
            name (str, optional): Value override for args.name. Defaults to None.
            watch (Positive int, optional): Value override for args.watch. Defaults to None.
            watch_time (Positive int, optional): Value override for args.watch_time. Defaults to None.
            iterations (Positive int, optional): Value override for args.iterations. Defaults to None.

        Raises:
            IndexError: Index error if gpu list is empty

        Returns:
            None: Print output via AMDSMILogger to destination
        """
        # Set args.* to passed in arguments
        if gpu:
            args.gpu = gpu
        if general:
            args.general = general
        if engine:
            args.engine = engine
        if pid:
            args.pid = pid
        if name:
            args.name = name
        if watch:
            args.watch = watch
        if watch_time:
            args.watch_time = watch_time
        if iterations:
            args.iterations = iterations

        # Handle No GPU passed
        if args.gpu == None:
            args.gpu = self.device_handles

        # Handle watch logic, will only enter this block once
        if args.watch:
            self.helpers.handle_watch(args=args, subcommand=self.process, logger=self.logger)
            return

        # Handle multiple GPUs
        if isinstance(args.gpu, list):
            if len(args.gpu) > 1:
                # Deepcopy gpus as recursion will destroy the gpu list
                stored_gpus = []
                for gpu in args.gpu:
                    stored_gpus.append(gpu)

                # Store output from multiple devices
                for device_handle in args.gpu:
                    self.process(args, multiple_devices=True, watching_output=watching_output, gpu=device_handle)

                # Reload original gpus
                args.gpu = stored_gpus

                # Print multiple device output
                self.logger.print_output(multiple_device_enabled=True, watching_output=watching_output)

                # Add output to total watch output and clear multiple device output
                if watching_output:
                    self.logger.store_watch_output(multiple_device_enabled=True)

                    # Flush the watching output
                    self.logger.print_output(multiple_device_enabled=True, watching_output=watching_output)

                return
            elif len(args.gpu) == 1:
                args.gpu = args.gpu[0]
            else:
                raise IndexError("args.gpu should not be an empty list")

        # Get gpu_id for logging
        gpu_id = self.helpers.get_gpu_id_from_device_handle(args.gpu)

        # Populate initial processes
        try:
            process_list = amdsmi_interface.amdsmi_get_gpu_process_list(args.gpu)
        except amdsmi_exception.AmdSmiLibraryException as e:
            logging.debug("Failed to get process list for gpu %s | %s", gpu_id, e.get_error_info())
            raise e

        filtered_process_values = []
        for process_info in process_list:
            process_info = {
                "name": process_info["name"],
                "pid": process_info["pid"],
                "memory_usage": {
                    "gtt_mem": process_info["memory_usage"]["gtt_mem"],
                    "cpu_mem": process_info["memory_usage"]["cpu_mem"],
                    "vram_mem": process_info["memory_usage"]["vram_mem"],
                },
                "mem_usage": process_info["mem"],
                "usage": {
                    "gfx": process_info["engine_usage"]["gfx"],
                    "enc": process_info["engine_usage"]["enc"],
                },
                "cu_occupancy": process_info["cu_occupancy"]
            }

            engine_usage_unit = "ns"
            memory_usage_unit = "B"

            if self.logger.is_human_readable_format():
                process_info['mem_usage'] = self.helpers.convert_bytes_to_readable(process_info['mem_usage'])
                for usage_metric in process_info['memory_usage']:
                    process_info["memory_usage"][usage_metric] = self.helpers.convert_bytes_to_readable(process_info["memory_usage"][usage_metric])
                memory_usage_unit = ""

            process_info['mem_usage'] = self.helpers.unit_format(self.logger,
                                                                 process_info['mem_usage'],
                                                                 memory_usage_unit)

            for usage_metric in process_info['usage']:
                process_info['usage'][usage_metric] = self.helpers.unit_format(self.logger,
                                                                               process_info['usage'][usage_metric],
                                                                               engine_usage_unit)

            for usage_metric in process_info['memory_usage']:
                process_info['memory_usage'][usage_metric] = self.helpers.unit_format(self.logger,
                                                                                      process_info['memory_usage'][usage_metric],
                                                                                      memory_usage_unit)

            filtered_process_values.append({'process_info': process_info})

        if not filtered_process_values:
            process_info = "N/A"
            logging.debug("Failed to detect any process on gpu %s", gpu_id)
            filtered_process_values.append({'process_info': process_info})

        # Arguments will filter the populated processes
        # General and Engine to expose process_info values
        if args.general or args.engine:
            for process_info in filtered_process_values:
                if not process_info['process_info'] == "N/A":
                    if args.general and args.engine:
                        del process_info['process_info']['memory_usage']
                    elif args.general:
                        del process_info['process_info']['memory_usage']
                        del process_info['process_info']['usage'] # Used in engine
                    elif args.engine:
                        del process_info['process_info']['memory_usage']
                        del process_info['process_info']['mem_usage'] # Used in general

        # Filter out non specified pids
        if args.pid:
            process_pids = []
            for process_info in filtered_process_values:
                if process_info['process_info'] == "N/A":
                    continue
                pid = str(process_info['process_info']['pid'])
                if str(args.pid) == pid:
                    process_pids.append(process_info)
            filtered_process_values = process_pids

        # Filter out non specified process names
        if args.name:
            process_names = []
            for process_info in filtered_process_values:
                if process_info['process_info'] == "N/A":
                    continue
                process_name = str(process_info['process_info']['name']).lower()
                if str(args.name).lower() == process_name:
                    process_names.append(process_info)
            filtered_process_values = process_names

        # If the name or pid args filter processes out then insert an N/A placeholder
        if not filtered_process_values:
            filtered_process_values.append({'process_info': "N/A"})

        logging.debug(f"Process Info for GPU {gpu_id} | {filtered_process_values}")

        for index, process in enumerate(filtered_process_values):
            if process['process_info'] == "N/A":
                filtered_process_values[index]['process_info'] = "No running processes detected"

        if self.logger.is_json_format():
            if watching_output:
                self.logger.store_output(args.gpu, 'timestamp', int(time.time()))
            self.logger.store_output(args.gpu, 'process_list', filtered_process_values)

        if self.logger.is_human_readable_format():
            if watching_output:
                self.logger.store_output(args.gpu, 'timestamp', int(time.time()))
            # When we print out process_info we remove the index
            # The removal is needed only for human readable process format to align with Host
            for index, process in enumerate(filtered_process_values):
                self.logger.store_output(args.gpu, f'process_info_{index}', process['process_info'])

        multiple_devices_csv_override = False
        if self.logger.is_csv_format():
            multiple_devices_csv_override = True
            for process in filtered_process_values:
                if watching_output:
                    self.logger.store_output(args.gpu, 'timestamp', int(time.time()))
                self.logger.store_output(args.gpu, 'process_info', process['process_info'])
                self.logger.store_multiple_device_output()

        if multiple_devices:
            self.logger.store_multiple_device_output()
            return # Skip printing when there are multiple devices

        multiple_devices = multiple_devices or multiple_devices_csv_override
        self.logger.print_output(multiple_device_enabled=multiple_devices, watching_output=watching_output)

        if watching_output: # End of single gpu add to watch_output
            self.logger.store_watch_output(multiple_device_enabled=multiple_devices)


    def profile(self, args):
        """Not applicable to linux baremetal"""
        print('Not applicable to linux baremetal')


    def event(self, args, gpu=None):
        """ Get event information for target gpus

        Args:
            args (Namespace): argparser args to pass to subcommand
            gpu (device_handle, optional): device_handle for target device. Defaults to None.

        Return:
            stdout event information for target gpus
        """
        if args.gpu:
            gpu = args.gpu

        if gpu == None:
            args.gpu = self.device_handles

        if not isinstance(args.gpu, list):
            args.gpu = [args.gpu]

        print('EVENT LISTENING:\n')
        print('Press q and hit ENTER when you want to stop.')
        self.stop = False
        threads = []
        for device_handle in range(len(args.gpu)):
            x = threading.Thread(target=self._event_thread, args=(self, device_handle))
            threads.append(x)
            x.start()

        while True:
            user_input = input()
            if user_input == 'q':
                print("Escape Sequence Detected; Exiting")
                self.stop = True
                break

        for thread in threads:
            thread.join()


    def topology(self, args, multiple_devices=False, gpu=None, access=None,
                weight=None, hops=None, link_type=None, numa_bw=None,
                coherent=None, atomics=None, dma=None, bi_dir=None):
        """ Get topology information for target gpus
            params:
                args - argparser args to pass to subcommand
                multiple_devices (bool) - True if checking for multiple devices
                gpu (device_handle) - device_handle for target device
                access (bool) - Value override for args.access
                weight (bool) - Value override for args.weight
                hops (bool) - Value override for args.hops
                type (bool) - Value override for args.type
                numa_bw (bool) - Value override for args.numa_bw
                coherent (bool) - Value override for args.coherent
                atomics (bool) - Value override for args.atomics
                dma (bool) - Value override for args.dma
                bi_dir (bool) - Value override for args.bi_dir
            return:
                Nothing
        """
        # Set args.* to passed in arguments
        if gpu:
            args.gpu = gpu
        if access:
            args.access = access
        if weight:
            args.weight = weight
        if hops:
            args.hops = hops
        if link_type:
            args.link_type = link_type
        if numa_bw:
            args.numa_bw = numa_bw
        if coherent:
            args.coherent = coherent
        if atomics:
            args.atomics = atomics
        if dma:
            args.dma = dma
        if bi_dir:
            args.bi_dir = bi_dir

        # Handle No GPU passed
        if args.gpu == None:
            args.gpu = self.device_handles

        if not isinstance(args.gpu, list):
            args.gpu = [args.gpu]

        # Handle all args being false
        if not any([args.access, args.weight, args.hops, args.link_type, args.numa_bw,
                    args.coherent, args.atomics, args.dma, args.bi_dir]):
            args.access = args.weight = args.hops = args.link_type= args.numa_bw = \
            args.coherent = args.atomics = args.dma = args.bi_dir = True

        # Clear the table header
        self.logger.table_header = ''.rjust(12)

        if not self.group_check_printed:
            self.helpers.check_required_groups()
            self.group_check_printed = True

        p2p_status_cache = {}

        def get_cached_p2p_status(src_gpu, dest_gpu):
            #Get P2P status with caching to avoid duplicate calls
            src_gpu_id = self.helpers.get_gpu_id_from_device_handle(src_gpu)
            dest_gpu_id = self.helpers.get_gpu_id_from_device_handle(dest_gpu)
            key = (src_gpu_id, dest_gpu_id)

            if key not in p2p_status_cache:
                try:
                    if src_gpu == dest_gpu:
                        p2p_status_cache[key] = {"cap": {
                            "is_iolink_coherent": -1,
                            "is_iolink_atomics_32bit": -1,
                            "is_iolink_atomics_64bit": -1,
                            "is_iolink_dma": -1,
                            "is_iolink_bi_directional": -1
                        }}
                    else:
                        p2p_status_cache[key] = amdsmi_interface.amdsmi_topo_get_p2p_status(src_gpu, dest_gpu)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug("Failed to get link status for %s to %s | %s",
                                src_gpu_id,
                                dest_gpu_id,
                                e.get_error_info())
                    p2p_status_cache[key] ={
                        "cap":
                        {
                            "is_iolink_coherent": -1,
                            "is_iolink_atomics_32bit": -1,
                            "is_iolink_atomics_64bit": -1,
                            "is_iolink_dma": -1,
                            "is_iolink_bi_directional": -1
                        }
                    }

            return p2p_status_cache[key]

        # Populate the possible gpus
        topo_values = []
        for src_gpu_index, src_gpu in enumerate(args.gpu):
            src_gpu_id = self.helpers.get_gpu_id_from_device_handle(src_gpu)
            topo_values.append({"gpu" : src_gpu_id})
            src_gpu_bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(src_gpu)
            topo_values[src_gpu_index]['bdf'] = src_gpu_bdf
            self.logger.table_header += src_gpu_bdf.rjust(13)

            if not self.logger.is_json_format():
                continue  # below is for JSON format only

            ##########################
            # JSON formatting start  #
            ##########################
            links = []
            # create json obj for data alignment
            #  dest_gpu_links = {
            #         "gpu": GPU #
            #         "bdf": BDF identification
            #         "weight": 0 - self (current node); weight >= 0 correlated with hops (GPU-CPU, GPU-GPU, GPU-CPU-CPU-GPU, etc..)
            #         "link_status": "ENABLED" - devices linked; "DISABLED" - devices not linked; Correlated to access
            #         "link_type": "SELF" - current node, "PCIE", "XGMI", "N/A" - no link,"UNKNOWN" - unidentified link type
            #         "num_hops": num_hops - # of hops between devices
            #         "bandwidth": numa_bw - The NUMA "minimum bandwidth-maximum bandwidth" beween src and dest nodes
            #                      "N/A" - self node or not connected devices
            #         "coherent": coherent - Coherant / Non-Coherant io links
            #         "atomics": atomics - 32 and 64-bit atomic io link capability between nodes
            #         "dma": dma - P2P direct memory access (DMA) link capability between nodes
            #         "bi_dir": bi_dir - P2P bi-directional link capability between nodes
            #     }

            for dest_gpu_index, dest_gpu in enumerate(args.gpu):
                link_type = "SELF"
                if src_gpu != dest_gpu:
                    link_type = amdsmi_interface.amdsmi_topo_get_link_type(src_gpu, dest_gpu)['type']
                if isinstance(link_type, int):
                    if link_type == amdsmi_interface.amdsmi_wrapper.AMDSMI_LINK_TYPE_INTERNAL:
                        link_type = "UNKNOWN"
                    elif link_type == amdsmi_interface.amdsmi_wrapper.AMDSMI_LINK_TYPE_PCIE:
                        link_type = "PCIE"
                    elif link_type == amdsmi_interface.amdsmi_wrapper.AMDSMI_LINK_TYPE_XGMI:
                        link_type = "XGMI"
                    else:
                        link_type = "N/A"

                numa_bw = "N/A"
                if src_gpu != dest_gpu:
                    try:
                        bw_dict = amdsmi_interface.amdsmi_get_minmax_bandwidth_between_processors(src_gpu, dest_gpu)
                        numa_bw = f"{bw_dict['min_bandwidth']}-{bw_dict['max_bandwidth']}"
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        logging.debug("Failed to get min max bandwidth for %s to %s | %s",
                                    self.helpers.get_gpu_id_from_device_handle(src_gpu),
                                    self.helpers.get_gpu_id_from_device_handle(dest_gpu),
                                    e.get_error_info())

                weight = 0
                num_hops = 0
                if src_gpu != dest_gpu:
                    weight = amdsmi_interface.amdsmi_topo_get_link_weight(src_gpu, dest_gpu)
                    num_hops = amdsmi_interface.amdsmi_topo_get_link_type(src_gpu, dest_gpu)['hops']
                link_status = amdsmi_interface.amdsmi_is_P2P_accessible(src_gpu, dest_gpu)
                if link_status:
                    link_status = "ENABLED"
                else:
                    link_status = "DISABLED"

                link_coherent = "SELF"
                link_atomics = "SELF"
                link_dma = "SELF"
                link_bi_dir = "SELF"

                if src_gpu != dest_gpu:
                    try:
                        cap = get_cached_p2p_status(src_gpu, dest_gpu)['cap']
                        link_coherent = (
                            "C" if cap['is_iolink_coherent'] == 1 else
                            "NC" if cap['is_iolink_coherent'] == 0 else
                            "N/A"
                        )
                        link_atomics = (
                            "64,32" if cap['is_iolink_atomics_32bit'] == 1 and cap['is_iolink_atomics_64bit'] == 1 else
                            "32" if cap['is_iolink_atomics_32bit'] == 1 else
                            "64" if cap['is_iolink_atomics_64bit'] == 1 else
                            "N/A"
                        )
                        link_dma = (
                            "T" if cap['is_iolink_dma'] == 1 else
                            "F" if cap['is_iolink_dma'] == 0 else
                            "N/A"
                        )
                        link_bi_dir = (
                            "T" if cap['is_iolink_bi_directional'] == 1 else
                            "F" if cap['is_iolink_bi_directional'] == 0 else
                            "N/A"
                        )
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        logging.debug("Failed to get link status for %s to %s | %s",
                                    self.helpers.get_gpu_id_from_device_handle(src_gpu),
                                    self.helpers.get_gpu_id_from_device_handle(dest_gpu),
                                    e.get_error_info())


                # link_status = amdsmi_is_P2P_accessible(src,dest)
                dest_gpu_links = {
                    "gpu": self.helpers.get_gpu_id_from_device_handle(dest_gpu),
                    "bdf": amdsmi_interface.amdsmi_get_gpu_device_bdf(dest_gpu),
                    "weight": weight,
                    "link_status": link_status,
                    "link_type": link_type,
                    "num_hops": num_hops,
                    "bandwidth": numa_bw,
                    "coherent": link_coherent,
                    "atomics": link_atomics,
                    "dma": link_dma,
                    "bi_dir": link_bi_dir
                }
                if not args.access:
                    del dest_gpu_links['link_status']
                if not args.weight:
                    del dest_gpu_links['weight']
                if not args.link_type:
                    del dest_gpu_links['link_type']
                if not args.hops:
                    del dest_gpu_links['num_hops']
                if not args.numa_bw:
                    del dest_gpu_links['bandwidth']
                if not args.coherent:
                    del dest_gpu_links['coherent']
                if not args.atomics:
                    del dest_gpu_links['atomics']
                if not args.dma:
                    del dest_gpu_links['dma']
                if not args.bi_dir:
                    del dest_gpu_links['bi_dir']
                links.append(dest_gpu_links)
                dest_end = dest_gpu_index+1 == len(args.gpu)
                isEndOfSrc = src_gpu_index+1 == len(args.gpu)
                if dest_end:
                    topo_values[src_gpu_index]['links'] = links
                    continue
            if isEndOfSrc:
                self.logger.multiple_device_output = topo_values
                self.logger.print_output(multiple_device_enabled=True, tabular=True)
                return
            ##########################
            # JSON formatting end    #
            ##########################

        if args.access:
            tabular_output = []
            for src_gpu_index, src_gpu in enumerate(args.gpu):
                src_gpu_bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(src_gpu)
                if self.logger.is_human_readable_format():
                    tabular_output_dict = {'gpu' : f"{src_gpu_bdf} "}
                else:
                    tabular_output_dict = {'gpu' : src_gpu_bdf}
                src_gpu_links = {}
                for dest_gpu in args.gpu:
                    dest_gpu_id = self.helpers.get_gpu_id_from_device_handle(dest_gpu)
                    dest_gpu_key = f'gpu_{dest_gpu_id}'

                    try:
                        dest_gpu_link_status = amdsmi_interface.amdsmi_is_P2P_accessible(src_gpu, dest_gpu)
                        if dest_gpu_link_status:
                            src_gpu_links[dest_gpu_key] = "ENABLED"
                        else:
                            src_gpu_links[dest_gpu_key] = "DISABLED"
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        src_gpu_links[dest_gpu_key] = "N/A"
                        logging.debug("Failed to get link status for %s to %s | %s",
                                        self.helpers.get_gpu_id_from_device_handle(src_gpu),
                                        self.helpers.get_gpu_id_from_device_handle(dest_gpu),
                                        e.get_error_info())

                topo_values[src_gpu_index]['link_accessibility'] = src_gpu_links

                tabular_output_dict.update(src_gpu_links)
                tabular_output.append(tabular_output_dict)

            if self.logger.is_human_readable_format():
                self.logger.multiple_device_output = tabular_output
                self.logger.table_title = "ACCESS TABLE"
                self.logger.print_output(multiple_device_enabled=True, tabular=True)

        if args.weight:
            tabular_output = []
            for src_gpu_index, src_gpu in enumerate(args.gpu):
                src_gpu_bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(src_gpu)
                if self.logger.is_human_readable_format():
                    tabular_output_dict = {'gpu' : f"{src_gpu_bdf} "}
                else:
                    tabular_output_dict = {'gpu' : src_gpu_bdf}
                src_gpu_weight = {}
                for dest_gpu in args.gpu:
                    dest_gpu_id = self.helpers.get_gpu_id_from_device_handle(dest_gpu)
                    dest_gpu_key = f'gpu_{dest_gpu_id}'

                    if src_gpu == dest_gpu:
                        src_gpu_weight[dest_gpu_key] = 0
                        continue

                    try:
                        dest_gpu_link_weight = amdsmi_interface.amdsmi_topo_get_link_weight(src_gpu, dest_gpu)
                        src_gpu_weight[dest_gpu_key] = dest_gpu_link_weight
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        src_gpu_weight[dest_gpu_key] = "N/A"
                        logging.debug("Failed to get link weight for %s to %s | %s",
                                        self.helpers.get_gpu_id_from_device_handle(src_gpu),
                                        self.helpers.get_gpu_id_from_device_handle(dest_gpu),
                                        e.get_error_info())

                topo_values[src_gpu_index]['weight'] = src_gpu_weight

                tabular_output_dict.update(src_gpu_weight)
                tabular_output.append(tabular_output_dict)

            if self.logger.is_human_readable_format():
                self.logger.multiple_device_output = tabular_output
                self.logger.table_title = "WEIGHT TABLE"
                self.logger.print_output(multiple_device_enabled=True, tabular=True)

        if args.hops:
            tabular_output = []
            for src_gpu_index, src_gpu in enumerate(args.gpu):
                src_gpu_bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(src_gpu)
                if self.logger.is_human_readable_format():
                    tabular_output_dict = {'gpu' : f"{src_gpu_bdf} "}
                else:
                    tabular_output_dict = {'gpu' : src_gpu_bdf}
                src_gpu_hops = {}
                for dest_gpu in args.gpu:
                    dest_gpu_id = self.helpers.get_gpu_id_from_device_handle(dest_gpu)
                    dest_gpu_key = f'gpu_{dest_gpu_id}'

                    if src_gpu == dest_gpu:
                        src_gpu_hops[dest_gpu_key] = 0
                        continue

                    try:
                        dest_gpu_hops = amdsmi_interface.amdsmi_topo_get_link_type(src_gpu, dest_gpu)['hops']
                        src_gpu_hops[dest_gpu_key] = dest_gpu_hops
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        src_gpu_hops[dest_gpu_key] = "N/A"
                        logging.debug("Failed to get link hops for %s to %s | %s",
                                        self.helpers.get_gpu_id_from_device_handle(src_gpu),
                                        self.helpers.get_gpu_id_from_device_handle(dest_gpu),
                                        e.get_error_info())

                topo_values[src_gpu_index]['hops'] = src_gpu_hops

                tabular_output_dict.update(src_gpu_hops)
                tabular_output.append(tabular_output_dict)

            if self.logger.is_human_readable_format():
                self.logger.multiple_device_output = tabular_output
                self.logger.table_title = "HOPS TABLE"
                self.logger.print_output(multiple_device_enabled=True, tabular=True)

        if args.link_type:
            tabular_output = []
            for src_gpu_index, src_gpu in enumerate(args.gpu):
                src_gpu_bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(src_gpu)
                if self.logger.is_human_readable_format():
                    tabular_output_dict = {'gpu' : f"{src_gpu_bdf} "}
                else:
                    tabular_output_dict = {'gpu' : src_gpu_bdf}
                src_gpu_link_type = {}
                for dest_gpu in args.gpu:
                    dest_gpu_id = self.helpers.get_gpu_id_from_device_handle(dest_gpu)
                    dest_gpu_key = f'gpu_{dest_gpu_id}'

                    if src_gpu == dest_gpu:
                        src_gpu_link_type[dest_gpu_key] = "SELF"
                        continue
                    try:
                        link_type = amdsmi_interface.amdsmi_topo_get_link_type(src_gpu, dest_gpu)['type']
                        if isinstance(link_type, int):
                            if link_type == amdsmi_interface.amdsmi_wrapper.AMDSMI_LINK_TYPE_INTERNAL:
                                src_gpu_link_type[dest_gpu_key] = "UNKNOWN"
                            elif link_type == amdsmi_interface.amdsmi_wrapper.AMDSMI_LINK_TYPE_PCIE:
                                src_gpu_link_type[dest_gpu_key] = "PCIE"
                            elif link_type == amdsmi_interface.amdsmi_wrapper.AMDSMI_LINK_TYPE_XGMI:
                                src_gpu_link_type[dest_gpu_key] = "XGMI"
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        src_gpu_link_type[dest_gpu_key] = "N/A"
                        logging.debug("Failed to get link type for %s to %s | %s",
                                        self.helpers.get_gpu_id_from_device_handle(src_gpu),
                                        self.helpers.get_gpu_id_from_device_handle(dest_gpu),
                                        e.get_error_info())

                topo_values[src_gpu_index]['link_type'] = src_gpu_link_type

                tabular_output_dict.update(src_gpu_link_type)
                tabular_output.append(tabular_output_dict)

            if self.logger.is_human_readable_format():
                self.logger.multiple_device_output = tabular_output
                self.logger.table_title = "LINK TYPE TABLE"
                self.logger.print_output(multiple_device_enabled=True, tabular=True)

        if args.numa_bw:
            tabular_output = []
            for src_gpu_index, src_gpu in enumerate(args.gpu):
                src_gpu_bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(src_gpu)
                if self.logger.is_human_readable_format():
                    tabular_output_dict = {'gpu' : f"{src_gpu_bdf} "}
                else:
                    tabular_output_dict = {'gpu' : src_gpu_bdf}
                src_gpu_link_type = {}
                for dest_gpu in args.gpu:
                    dest_gpu_id = self.helpers.get_gpu_id_from_device_handle(dest_gpu)
                    dest_gpu_key = f'gpu_{dest_gpu_id}'

                    if src_gpu == dest_gpu:
                        src_gpu_link_type[dest_gpu_key] = "N/A"
                        continue

                    try:
                        link_type = amdsmi_interface.amdsmi_topo_get_link_type(src_gpu, dest_gpu)['type']
                        if isinstance(link_type, int):
                            if link_type != amdsmi_interface.amdsmi_wrapper.AMDSMI_LINK_TYPE_XGMI:
                                # non_xgmi = True
                                src_gpu_link_type[dest_gpu_key] = "N/A"
                                continue
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        src_gpu_link_type[dest_gpu_key] = "N/A"
                        logging.debug("Failed to get link type for %s to %s | %s",
                                        self.helpers.get_gpu_id_from_device_handle(src_gpu),
                                        self.helpers.get_gpu_id_from_device_handle(dest_gpu),
                                        e.get_error_info())

                    try:
                        bw_dict = amdsmi_interface.amdsmi_get_minmax_bandwidth_between_processors(src_gpu, dest_gpu)
                        src_gpu_link_type[dest_gpu_key] = f"{bw_dict['min_bandwidth']}-{bw_dict['max_bandwidth']}"
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        src_gpu_link_type[dest_gpu_key] =  e.get_error_info()
                        logging.debug("Failed to get min max bandwidth for %s to %s | %s",
                                        self.helpers.get_gpu_id_from_device_handle(src_gpu),
                                        self.helpers.get_gpu_id_from_device_handle(dest_gpu),
                                        e.get_error_info())

                topo_values[src_gpu_index]['numa_bandwidth'] = src_gpu_link_type

                tabular_output_dict.update(src_gpu_link_type)
                tabular_output.append(tabular_output_dict)

            if self.logger.is_human_readable_format():
                self.logger.multiple_device_output = tabular_output
                self.logger.table_title = "NUMA BW TABLE"
                self.logger.print_output(multiple_device_enabled=True, tabular=True)

        if args.coherent:
            tabular_output = []
            for src_gpu_index, src_gpu in enumerate(args.gpu):
                src_gpu_bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(src_gpu)
                if self.logger.is_human_readable_format():
                    tabular_output_dict = {'gpu' : f"{src_gpu_bdf} "}
                else:
                    tabular_output_dict = {'gpu' : src_gpu_bdf}
                src_gpu_coherent = {}
                for dest_gpu in args.gpu:
                    dest_gpu_id = self.helpers.get_gpu_id_from_device_handle(dest_gpu)
                    dest_gpu_key = f'gpu_{dest_gpu_id}'

                    if src_gpu == dest_gpu:
                        src_gpu_coherent[dest_gpu_key] = "SELF"
                        continue
                    try:
                        iolink_coherent = get_cached_p2p_status(src_gpu, dest_gpu)['cap']['is_iolink_coherent']
                        src_gpu_coherent[dest_gpu_key] = "C" if iolink_coherent == 1 else "NC" if iolink_coherent == 0 else "N/A"
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        src_gpu_coherent[dest_gpu_key] = "N/A"
                        logging.debug("Failed to get link coherent for %s to %s | %s",
                                        self.helpers.get_gpu_id_from_device_handle(src_gpu),
                                        self.helpers.get_gpu_id_from_device_handle(dest_gpu),
                                        e.get_error_info())

                topo_values[src_gpu_index]['coherent'] = src_gpu_coherent

                tabular_output_dict.update(src_gpu_coherent)
                tabular_output.append(tabular_output_dict)

            if self.logger.is_human_readable_format():
                self.logger.multiple_device_output = tabular_output
                self.logger.table_title = "CACHE COHERANCY TABLE"
                self.logger.print_output(multiple_device_enabled=True, tabular=True)

        if args.atomics:
            tabular_output = []
            for src_gpu_index, src_gpu in enumerate(args.gpu):
                src_gpu_bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(src_gpu)
                if self.logger.is_human_readable_format():
                    tabular_output_dict = {'gpu' : f"{src_gpu_bdf} "}
                else:
                    tabular_output_dict = {'gpu' : src_gpu_bdf}
                src_gpu_atomics = {}
                for dest_gpu in args.gpu:
                    dest_gpu_id = self.helpers.get_gpu_id_from_device_handle(dest_gpu)
                    dest_gpu_key = f'gpu_{dest_gpu_id}'

                    if src_gpu == dest_gpu:
                        src_gpu_atomics[dest_gpu_key] = "SELF"
                        continue
                    try:
                        cap = get_cached_p2p_status(src_gpu, dest_gpu)['cap']
                        src_gpu_atomics[dest_gpu_key] = (
                            "64,32" if cap['is_iolink_atomics_32bit'] == 1 and cap['is_iolink_atomics_64bit'] == 1 else
                            "32" if cap['is_iolink_atomics_32bit'] == 1 else
                            "64" if cap['is_iolink_atomics_64bit'] == 1 else
                            "N/A"
                        )
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        src_gpu_atomics[dest_gpu_key] = "N/A"
                        logging.debug("Failed to get link atomics for %s to %s | %s",
                                        self.helpers.get_gpu_id_from_device_handle(src_gpu),
                                        self.helpers.get_gpu_id_from_device_handle(dest_gpu),
                                        e.get_error_info())

                topo_values[src_gpu_index]['atomics'] = src_gpu_atomics

                tabular_output_dict.update(src_gpu_atomics)
                tabular_output.append(tabular_output_dict)

            if self.logger.is_human_readable_format():
                self.logger.multiple_device_output = tabular_output
                self.logger.table_title = "ATOMICS TABLE"
                self.logger.print_output(multiple_device_enabled=True, tabular=True)

        if args.dma:
            tabular_output = []
            for src_gpu_index, src_gpu in enumerate(args.gpu):
                src_gpu_bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(src_gpu)
                if self.logger.is_human_readable_format():
                    tabular_output_dict = {'gpu' : f"{src_gpu_bdf} "}
                else:
                    tabular_output_dict = {'gpu' : src_gpu_bdf}
                src_gpu_dma = {}
                for dest_gpu in args.gpu:
                    dest_gpu_id = self.helpers.get_gpu_id_from_device_handle(dest_gpu)
                    dest_gpu_key = f'gpu_{dest_gpu_id}'

                    if src_gpu == dest_gpu:
                        src_gpu_dma[dest_gpu_key] = "SELF"
                        continue
                    try:
                        iolink_dma = get_cached_p2p_status(src_gpu, dest_gpu)['cap']['is_iolink_dma']
                        src_gpu_dma[dest_gpu_key] = "T" if iolink_dma == 1 else "F" if iolink_dma == 0 else "N/A"
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        src_gpu_dma[dest_gpu_key] = "N/A"
                        logging.debug("Failed to get link dma for %s to %s | %s",
                                        self.helpers.get_gpu_id_from_device_handle(src_gpu),
                                        self.helpers.get_gpu_id_from_device_handle(dest_gpu),
                                        e.get_error_info())

                topo_values[src_gpu_index]['dma'] = src_gpu_dma

                tabular_output_dict.update(src_gpu_dma)
                tabular_output.append(tabular_output_dict)

            if self.logger.is_human_readable_format():
                self.logger.multiple_device_output = tabular_output
                self.logger.table_title = "DMA TABLE"
                self.logger.print_output(multiple_device_enabled=True, tabular=True)

        if args.bi_dir:
            tabular_output = []
            for src_gpu_index, src_gpu in enumerate(args.gpu):
                src_gpu_bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(src_gpu)
                if self.logger.is_human_readable_format():
                    tabular_output_dict = {'gpu' : f"{src_gpu_bdf} "}
                else:
                    tabular_output_dict = {'gpu' : src_gpu_bdf}
                src_gpu_bi_dir = {}
                for dest_gpu in args.gpu:
                    dest_gpu_id = self.helpers.get_gpu_id_from_device_handle(dest_gpu)
                    dest_gpu_key = f'gpu_{dest_gpu_id}'

                    if src_gpu == dest_gpu:
                        src_gpu_bi_dir[dest_gpu_key] = "SELF"
                        continue
                    try:
                        iolink_bi_dir = get_cached_p2p_status(src_gpu, dest_gpu)['cap']['is_iolink_bi_directional']
                        src_gpu_bi_dir[dest_gpu_key] = "T" if iolink_bi_dir == 1 else "F" if iolink_bi_dir == 0 else "N/A"
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        src_gpu_bi_dir[dest_gpu_key] = "N/A"
                        logging.debug("Failed to get link bi-directional for %s to %s | %s",
                                        self.helpers.get_gpu_id_from_device_handle(src_gpu),
                                        self.helpers.get_gpu_id_from_device_handle(dest_gpu),
                                        e.get_error_info())

                topo_values[src_gpu_index]['bi_dir'] = src_gpu_bi_dir

                tabular_output_dict.update(src_gpu_bi_dir)
                tabular_output.append(tabular_output_dict)

            if self.logger.is_human_readable_format():
                self.logger.multiple_device_output = tabular_output
                self.logger.table_title = "BI-DIRECTIONAL TABLE"
                self.logger.print_output(multiple_device_enabled=True, tabular=True)

        if self.logger.is_human_readable_format():
            # Populate the legend output
            legend_parts = [
                "\n\nLegend:",
                "  SELF = Current GPU",
                "  ENABLED / DISABLED = Link is enabled or disabled",
                "  N/A = Not supported",
                "  T/F = True / False",
                "  C/NC = Coherant / Non-Coherant io links",
                "  64,32 = 64 bit and 32 bit atomic support",
                "  <BW from>-<BW to>"
            ]
            legend_output = "\n".join(legend_parts)

            if self.logger.destination == 'stdout':
                print(legend_output)
            else:
                with self.logger.destination.open('a', encoding="utf-8") as output_file:
                    output_file.write(legend_output + '\n')

        self.logger.multiple_device_output = topo_values

        if self.logger.is_csv_format():
            new_output = []
            for elem in self.logger.multiple_device_output:
                new_output.append(self.logger.flatten_dict(elem, topology_override=True))
            self.logger.multiple_device_output = new_output

        if not self.logger.is_human_readable_format():
            self.logger.print_output(multiple_device_enabled=True)


    def set_core(self, args, multiple_devices=False, core=None, core_boost_limit=None):
        """Issue set commands to target core(s)

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            core (device_handle, optional): device_handle for target device. Defaults to None.
            core_boost_limit (list, optional): Value override for args.core_boost_limit. Defaults to None. Defaults to None.

        Raises:
            ValueError: Value error if no core value is provided
            IndexError: Index error if core list is empty

        Return:
            Nothing
        """
        if core:
            args.core = core
        if core_boost_limit:
            args.core_boost_limit = core_boost_limit

        if args.core == None:
            raise ValueError('No Core provided, specific Core targets(S) are needed')

        # Handle multiple cores
        handled_multiple_cores, device_handle = self.helpers.handle_cores(args, self.logger, self.set_core)
        if handled_multiple_cores:
            return # This function is recursive

        # Error if no subcommand args are passed
        if not any([args.core_boost_limit]):
            command = " ".join(sys.argv[1:])
            raise AmdSmiRequiredCommandException(command, self.logger.format)

        args.core = device_handle
        # build core string for errors
        try:
            core_id = self.helpers.get_core_id_from_device_handle(args.core)
        except IndexError:
            core_id = f'ID Unavailable for {args.core}'

        static_dict = {}
        if args.core_boost_limit:
            static_dict["set_core_boost_limit"] = {}
            try:
                amdsmi_interface.amdsmi_set_cpu_core_boostlimit(args.core, args.core_boost_limit[0][0])
                #Verify the core boost limit is set
                boost_limit = amdsmi_interface.amdsmi_get_cpu_core_boostlimit(args.core)
                # Extract numeric value from response (remove units if present)
                if isinstance(boost_limit, str):
                    # Extract just the number part (assumes format like "5000 MHz" or "5000")
                    boost_limit = int(boost_limit.split()[0])
                else:
                    boost_limit = int(boost_limit)

                if boost_limit < args.core_boost_limit[0][0]:
                    static_dict["set_core_boost_limit"]["Response"] = f"Max allowed boostlimit is {boost_limit} MHz"
                elif boost_limit > args.core_boost_limit[0][0]:
                    static_dict["set_core_boost_limit"]["Response"] = f"Min allowed boostlimit is {boost_limit} MHz"
                else:
                    static_dict["set_core_boost_limit"]["Response"] = f"{boost_limit} MHz"
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["set_core_boost_limit"]["Response"] = f"Error occurred for Core {core_id} - {e.get_error_info()}"
                logging.debug("Failed to set core boost limit for core %s | %s", core_id, e.get_error_info())

        multiple_devices_csv_override = False
        self.logger.store_core_output(args.core, 'values', static_dict)
        if multiple_devices:
            self.logger.store_multiple_device_output()
            return # Skip printing when there are multiple devices
        self.logger.print_output(multiple_device_enabled=multiple_devices_csv_override)


    def set_cpu(self, args, multiple_devices=False, cpu=None, cpu_pwr_limit=None,
                cpu_xgmi_link_width=None, cpu_lclk_dpm_level=None, cpu_pwr_eff_mode=None,
                cpu_gmi3_link_width=None, cpu_pcie_link_rate=None, cpu_df_pstate_range=None,
                cpu_enable_apb=None, cpu_disable_apb=None, soc_boost_limit=None):
        """Issue set commands to target cpu(s)

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            cpu (cpu_handle, optional): device_handle for target device. Defaults to None.
            cpu_pwr_limit (int, optional): Value override for args.cpu_pwr_limit. Defaults to None.
            cpu_xgmi_link_width (List[int], optional): Value override for args.cpu_xgmi_link_width. Defaults to None.
            cpu_lclk_dpm_level (List[int], optional): Value override for args.cpu_lclk_dpm_level. Defaults to None.
            cpu_pwr_eff_mode (int, optional): Value override for args.cpu_pwr_eff_mode. Defaults to None.
            cpu_gmi3_link_width (List[int], optional): Value override for args.cpu_gmi3_link_width. Defaults to None.
            cpu_pcie_link_rate (int, optional): Value override for args.cpu_pcie_link_rate. Defaults to None.
            cpu_df_pstate_range (List[int], optional): Value override for args.cpu_df_pstate_range. Defaults to None.
            cpu_enable_apb (bool, optional): Value override for args.cpu_enable_apb. Defaults to None.
            cpu_disable_apb (int, optional): Value override for args.cpu_disable_apb. Defaults to None.
            soc_boost_limit (int, optional): Value override for args.soc_boost_limit. Defaults to None.

        Raises:
            ValueError: Value error if no cpu value is provided
            IndexError: Index error if cpu list is empty

        Return:
            Nothing
        """
        if cpu:
            args.cpu = cpu
        if cpu_pwr_limit:
            args.cpu_pwr_limit = cpu_pwr_limit
        if cpu_xgmi_link_width:
            args.cpu_xgmi_link_width = cpu_xgmi_link_width
        if cpu_lclk_dpm_level:
            args.cpu_lclk_dpm_level = cpu_lclk_dpm_level
        if cpu_pwr_eff_mode:
            args.cpu_pwr_eff_mode = cpu_pwr_eff_mode
        if cpu_gmi3_link_width:
            args.cpu_gmi3_link_width = cpu_gmi3_link_width
        if cpu_pcie_link_rate:
            args.cpu_pcie_link_rate = cpu_pcie_link_rate
        if cpu_df_pstate_range:
            args.cpu_df_pstate_range = cpu_df_pstate_range
        if cpu_enable_apb:
            args.cpu_enable_apb = cpu_enable_apb
        if cpu_disable_apb:
            args.cpu_disable_apb = cpu_disable_apb
        if soc_boost_limit:
            args.soc_boost_limit = soc_boost_limit

        if args.cpu == None:
            raise ValueError('No CPU provided, specific CPU targets(S) are needed')

        #Handle multiple CPU's
        handled_multiple_cpus, device_handle = self.helpers.handle_cpus(args, self.logger, self.set_cpu)
        if handled_multiple_cpus:
            return # This function is recursive

        args.cpu = device_handle
        #Error if no subcommand args are passed
        if not any([args.cpu_pwr_limit, args.cpu_xgmi_link_width, args.cpu_lclk_dpm_level,
                    args.cpu_pwr_eff_mode, args.cpu_gmi3_link_width, args.cpu_pcie_link_rate,
                    args.cpu_df_pstate_range, args.cpu_enable_apb, args.cpu_disable_apb,
                    args.soc_boost_limit]):
            command = " ".join(sys.argv[1:])
            raise AmdSmiRequiredCommandException(command, self.logger.format)

        # Build CPU string for errors
        try:
            cpu_id = self.helpers.get_cpu_id_from_device_handle(args.cpu)
        except IndexError:
            cpu_id = f'ID Unavailable for {args.cpu}'

        static_dict = {}

        if args.cpu_pwr_limit:
            static_dict["set_pwr_limit"] = {}
            try:
                soc_max_pwr_limit = amdsmi_interface.amdsmi_get_cpu_socket_power_cap_max(args.cpu)
                extract_numeric = soc_max_pwr_limit.split()[0]
                max_power = int(extract_numeric)

                amdsmi_interface.amdsmi_set_cpu_socket_power_cap(args.cpu, args.cpu_pwr_limit[0][0])
                if args.cpu_pwr_limit[0][0] > max_power:
                    args.cpu_pwr_limit[0][0] = max_power
                static_dict["set_pwr_limit"]["Response"] = f"{args.cpu_pwr_limit[0][0] / 1000:.3f} mW"
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["set_pwr_limit"]["Response"] = f"Error occurred for CPU {cpu_id} - {e.get_error_info()}"
                logging.debug("Failed to set power limit for cpu %s | %s", cpu_id, e.get_error_info())

        if args.cpu_xgmi_link_width:
            static_dict["set_xgmi_link_width"] = {}
            try:
                amdsmi_interface.amdsmi_set_cpu_xgmi_width(args.cpu, args.cpu_xgmi_link_width[0][0],
                                                           args.cpu_xgmi_link_width[0][1])
                static_dict["set_xgmi_link_width"]["Response"] = f"{args.cpu_xgmi_link_width[0][0]} - {args.cpu_xgmi_link_width[0][1]}"
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["set_xgmi_link_width"]["Response"] = f"Error occurred for CPU {cpu_id} - {e.get_error_info()}"
                logging.debug("Failed to set xgmi link width for cpu %s | %s", cpu_id, e.get_error_info())

        if args.cpu_lclk_dpm_level:
            static_dict["set_lclk_dpm_level"] = {}
            try:
                amdsmi_interface.amdsmi_set_cpu_socket_lclk_dpm_level(args.cpu, args.cpu_lclk_dpm_level[0][0],
                                                                      args.cpu_lclk_dpm_level[0][1],
                                                                      args.cpu_lclk_dpm_level[0][2])
                static_dict["set_lclk_dpm_level"]["Response"] = f"NBIO[{args.cpu_lclk_dpm_level[0][0]}]"
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["set_lclk_dpm_level"]["Response"] = f"Error occurred for CPU {cpu_id} - {e.get_error_info()}"
                logging.debug("Failed to set lclk dpm level for cpu %s | %s", cpu_id, e.get_error_info())

        if args.cpu_pwr_eff_mode:
            static_dict["set_pwr_eff_mode"] = {}
            try:
                amdsmi_interface.amdsmi_set_cpu_pwr_efficiency_mode(args.cpu, args.cpu_pwr_eff_mode[0][0])
                static_dict["set_pwr_eff_mode"]["Response"] = f"{args.cpu_pwr_eff_mode[0][0]}"
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["set_pwr_eff_mode"]["Response"] = f"Error occurred for CPU {cpu_id} - {e.get_error_info()}"
                logging.debug("Failed to set power efficiency mode for cpu %s | %s", cpu_id, e.get_error_info())

        if args.cpu_gmi3_link_width:
            static_dict["set_gmi3_link_width"] = {}
            try:
                amdsmi_interface.amdsmi_set_cpu_gmi3_link_width_range(args.cpu, args.cpu_gmi3_link_width[0][0],
                args.cpu_gmi3_link_width[0][1])
                static_dict["set_gmi3_link_width"]["response"] = f"{args.cpu_gmi3_link_width[0][0]} - {args.cpu_gmi3_link_width[0][1]}"
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["set_gmi3_link_width"]["response"] = f"Error occurred for CPU {cpu_id} - {e.get_error_info()}"
                logging.debug("Failed to set gmi3 link width for cpu %s | %s", cpu_id, e.get_error_info())

        if args.cpu_pcie_link_rate:
            static_dict["set_pcie_link_rate"] = {}
            try:
                resp = amdsmi_interface.amdsmi_set_cpu_pcie_link_rate(args.cpu, args.cpu_pcie_link_rate[0][0])
                static_dict["set_pcie_link_rate"]["prev_mode"] = resp
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["set_pcie_link_rate"]["prev_mode"] = f"Error occurred for CPU {cpu_id} - {e.get_error_info()}"
                logging.debug("Failed to set pcie link rate for cpu %s | %s", cpu_id, e.get_error_info())

        if args.cpu_df_pstate_range:
            static_dict["set_df_pstate_range"] = {}
            try:
                amdsmi_interface.amdsmi_set_cpu_df_pstate_range(args.cpu, args.cpu_df_pstate_range[0][0],
                args.cpu_df_pstate_range[0][1])
                static_dict["set_df_pstate_range"]["response"] = "Set Operation successful"
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["set_df_pstate_range"]["response"] = f"Error occurred for CPU {cpu_id} - {e.get_error_info()}"
                logging.debug("Failed to set df pstate range for cpu %s | %s", cpu_id, e.get_error_info())

        if args.cpu_enable_apb:
            static_dict["apbenable"] = {}
            try:
                amdsmi_interface.amdsmi_cpu_apb_enable(args.cpu)
                static_dict["apbenable"]["state"] = "Enabled DF - Pstate performance boost algorithm"
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["apbenable"]["state"] = f"Error occurred for CPU {cpu_id} - {e.get_error_info()}"
                logging.debug("Failed to enable APB for cpu %s | %s", cpu_id, e.get_error_info())

        if args.cpu_disable_apb:
            static_dict["apbdisable"] = {}
            try:
                amdsmi_interface.amdsmi_cpu_apb_disable(args.cpu, args.cpu_disable_apb[0][0])
                static_dict["apbdisable"]["state"] = "Disabled DF - Pstate performance boost algorithm"
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["apbdisable"]["state"] = f"Error occurred for CPU {cpu_id} - {e.get_error_info()}"
                logging.debug("Failed to enable APB for cpu %s | %s", cpu_id, e.get_error_info())

        if args.soc_boost_limit:
            static_dict["set_soc_boost_limit"] = {}
            try:
                amdsmi_interface.amdsmi_set_cpu_socket_boostlimit(args.cpu, args.soc_boost_limit[0][0])
                static_dict["set_soc_boost_limit"]["Response"] = "Set Operation successful"
            except amdsmi_exception.AmdSmiLibraryException as e:
                #static_dict["set_soc_boost_limit"]["Response"] = "N/A"
                static_dict["set_soc_boost_limit"]["Response"] = f"Error occurred for CPU {cpu_id} - {e.get_error_info()}"
                logging.debug("Failed to set socket boost limit for cpu %s | %s", cpu_id, e.get_error_info())

        multiple_devices_csv_override = False
        self.logger.store_cpu_output(args.cpu, 'values', static_dict)
        if multiple_devices:
            self.logger.store_multiple_device_output()
            return # Skip printing when there are multiple devices
        self.logger.print_output(multiple_device_enabled=multiple_devices_csv_override)


    def set_gpu(self, args, multiple_devices=False, gpu=None, fan=None, perf_level=None,
                  profile=None, perf_determinism=None, compute_partition=None,
                  memory_partition=None, power_cap=None, soc_pstate=None, xgmi_plpd = None,
                  process_isolation=None, clk_limit=None, clk_level=None):
        """Issue reset commands to target gpu(s)

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            gpu (device_handle, optional): device_handle for target device. Defaults to None.
            fan (int, optional): Value override for args.fan. Defaults to None.
            perf_level (amdsmi_interface.AmdSmiDevPerfLevel, optional): Value override for args.perf_level. Defaults to None.
            profile (bool, optional): Value override for args.profile. Defaults to None.
            perf_determinism (int, optional): Value override for args.perf_determinism. Defaults to None.
            compute_partition (amdsmi_interface.AmdSmiComputePartitionType, optional): Value override for args.compute_partition. Defaults to None.
            memory_partition (amdsmi_interface.AmdSmiMemoryPartitionType, optional): Value override for args.memory_partition. Defaults to None.
            power_cap (int, optional): Value override for args.power_cap. Defaults to None.
            soc_pstate (int, optional): Value override for args.soc_pstate. Defaults to None.
            xgmi_plpd (int, optional): Value override for args.xgmi_plpd. Defaults to None.
            process_isolation (int, optional): Value override for args.process_isolation. Defaults to None.
        Raises:
            ValueError: Value error if no gpu value is provided
            IndexError: Index error if gpu list is empty

        Return:
            Nothing
        """
        # Set args.* to passed in arguments
        if gpu:
            args.gpu = gpu
        if fan is not None:
            args.fan = fan
        if perf_level:
            args.perf_level = perf_level
        if profile:
            args.profile = profile
        if perf_determinism is not None:
            args.perf_determinism = perf_determinism
        if compute_partition:
            args.compute_partition = compute_partition
        if memory_partition:
            args.memory_partition = memory_partition
        if power_cap:
            args.power_cap = power_cap
        if soc_pstate:
            args.soc_pstate = soc_pstate
        if xgmi_plpd:
            args.xgmi_plpd = xgmi_plpd
        if process_isolation:
            args.process_isolation = process_isolation
        if clk_limit:
            args.clk_limit = clk_limit
        if clk_level:
            args.clk_level = clk_level

        # Handle No GPU passed
        if args.gpu == None:
            args.gpu = self.device_handles

        # Handle multiple GPUs
        handled_multiple_gpus, device_handle = self.helpers.handle_gpus(args, self.logger, self.set_gpu)
        if handled_multiple_gpus:
            return # This function is recursive

        args.gpu = device_handle

        # Error if no subcommand args are passed
        if self.helpers.is_baremetal():
            if not any([args.fan is not None,
                        args.perf_level,
                        args.profile,
                        args.compute_partition,
                        args.memory_partition,
                        args.perf_determinism is not None,
                        args.power_cap is not None,
                        args.soc_pstate is not None,
                        args.xgmi_plpd is not None,
                        args.clk_level is not None,
                        args.clk_limit is not None,
                        args.process_isolation is not None]):
                command = " ".join(sys.argv[1:])
                raise AmdSmiRequiredCommandException(command, self.logger.format)
        else:
            if not any([args.power_cap is not None,
                        args.clk_limit is not None,
                        args.process_isolation is not None]):
                command = " ".join(sys.argv[1:])
                raise AmdSmiRequiredCommandException(command, self.logger.format)

        # Build GPU string for errors
        try:
            gpu_bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(args.gpu)
        except amdsmi_exception.AmdSmiLibraryException:
            gpu_bdf = f'BDF Unavailable for {args.gpu}'
        try:
            gpu_id = self.helpers.get_gpu_id_from_device_handle(args.gpu)
        except IndexError:
            gpu_id = f'ID Unavailable for {args.gpu}'
        gpu_string = f"GPU ID: {gpu_id} BDF:{gpu_bdf}"

        # Handle args
        if self.helpers.is_baremetal():
            if isinstance(args.fan, int):
                # Convert fan speed to percentage
                # Note: amdsmi_set_gpu_fan_speed expects fan speed in RPM, so
                # we convert the value to a percentage based on the maximum fan speed of 255 RPM.
                # We need to round down the user's passed fan speed % to the nearest whole number.
                # This allows us to match the float -> int conversion when converting from percentage to RPM (as previously passed by the parser).
                fan_percentage = int((int(args.fan) / 255) * 100 // 1) # round down (aka floor) to nearest whole number
                try:
                    amdsmi_interface.amdsmi_set_gpu_fan_speed(args.gpu, 0, args.fan)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        raise PermissionError('Command requires elevation') from e
                    result = f"[{e.get_error_info(detailed=False)}] Unable to set fan speed to {args.fan} RPM ({fan_percentage}%)"
                    self.logger.store_output(args.gpu, 'fan', result)
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    return

                self.logger.store_output(args.gpu, 'fan', f"Successfully set fan speed to {args.fan} RPM ({fan_percentage}%)")
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return
            if args.perf_level:
                perf_level = amdsmi_interface.AmdSmiDevPerfLevel[args.perf_level]
                try:
                    amdsmi_interface.amdsmi_set_gpu_perf_level(args.gpu, perf_level)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        raise PermissionError('Command requires elevation') from e
                    self.logger.store_output(args.gpu, 'perflevel', f"[{e.get_error_info(detailed=False)}] Unable to set performance level to {args.perf_level}")
                    perf_options = str(self.helpers.get_perf_levels()[0][0:-1]).replace("[", "").replace("]", "").replace("'", "").replace(" ", "")
                    print(f"\nPerformance Level Options:\n\t{perf_options}\n")
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    return

                self.logger.store_output(args.gpu, 'perflevel', f"Successfully set performance level {args.perf_level}")
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return
            if args.profile:
                self.logger.store_output(args.gpu, 'profile', "Not Yet Implemented")
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return
            if isinstance(args.perf_determinism, int):
                try:
                    amdsmi_interface.amdsmi_set_gpu_perf_determinism_mode(args.gpu, args.perf_determinism)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        raise PermissionError('Command requires elevation') from e
                    self.logger.store_output(args.gpu, 'perfdeterminism', f"[{e.get_error_info(detailed=False)}] Unable to enable performance determinism and set GFX clock frequency to {args.perf_determinism} MHz")
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    return

                self.logger.store_output(args.gpu, 'perfdeterminism', f"Successfully enabled performance determinism and set GFX clock frequency to {args.perf_determinism} MHz")
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return
            if args.compute_partition:
                current_set_count = self.helpers.get_set_count()
                future_set_count = 0
                attempted_to_set = "N/A"
                user_requested_partition_args = "N/A"
                try:
                    (accelerator_set_choices, accelerator_profiles) = self.helpers.get_accelerator_choices_types_indices()
                    logging.debug("args.compute_partition: %s; Accelerator_set_choices: %s", str(args.compute_partition), str(json.dumps(accelerator_set_choices, indent=4)))
                    if args.compute_partition in accelerator_profiles['profile_types']:
                        compute_partition = amdsmi_interface.AmdSmiComputePartitionType[args.compute_partition]
                        index = accelerator_profiles['profile_types'].index(args.compute_partition)
                        attempted_to_set = f"Attempted to set accelerator partition to {args.compute_partition} (profile #{accelerator_profiles['profile_indices'][int(index)]}) on {gpu_string}"
                        user_requested_partition_args = f"{args.compute_partition} (profile #{accelerator_profiles['profile_indices'][int(index)]})"
                        amdsmi_interface.amdsmi_set_gpu_compute_partition(args.gpu, compute_partition)
                    elif args.compute_partition in accelerator_profiles['profile_indices']:
                        compute_partition = int(args.compute_partition)
                        index = accelerator_profiles['profile_indices'].index(args.compute_partition)
                        attempted_to_set = f"Attempted to set accelerator partition to {accelerator_profiles['profile_types'][int(index)]} (profile #{args.compute_partition}) on {gpu_string}"
                        user_requested_partition_args = f"{accelerator_profiles['profile_types'][int(index)]} (profile #{args.compute_partition})"
                        amdsmi_interface.amdsmi_set_gpu_accelerator_partition_profile(args.gpu, compute_partition)
                    else:
                        raise ValueError(f"Invalid accelerator configuration {args.compute_partition} on {gpu_string}")
                    self.helpers.increment_set_count()
                    future_set_count = self.helpers.get_set_count()
                    if current_set_count == future_set_count-1:
                        self.logger.store_output(args.gpu, 'accelerator_partition', f"Successfully set accelerator partition to {user_requested_partition_args}")
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    return

                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        raise PermissionError('Command requires elevation') from e
                    elif e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED:
                            self.helpers.increment_set_count()
                            future_set_count = self.helpers.get_set_count()
                            if current_set_count == future_set_count-1:
                                out = f"[AMDSMI_STATUS_NOT_SUPPORTED] Unable to set compute partition to {user_requested_partition_args}"
                                self.logger.store_output(args.gpu, 'accelerator_partition', out)
                    elif e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_SETTING_UNAVAILABLE:
                        print(f"\n{attempted_to_set}\n"
                              f"\n[AMDSMI_STATUS_SETTING_UNAVAILABLE] Please check amd-smi partition --memory --accelerator for available profiles.\n"
                               "Users may need to switch memory partition to another mode in order to enable the desired accelerator partition.\n")
                        raise ValueError(f"[AMDSMI_STATUS_SETTING_UNAVAILABLE] Unable to set accelerator partition to {args.compute_partition} on {gpu_string}") from e
                    else:
                        raise ValueError(f"Unable to set accelerator partition to {args.compute_partition} on {gpu_string}") from e
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    return
            if args.memory_partition:
                ####################################################################
                # Get current and available memory partition modes                 #
                # Info used if AMDSMI_STATUS_INVAL is caught & to set progress bar #
                ####################################################################
                self.helpers.increment_set_count()
                set_count = self.helpers.get_set_count()
                if set_count == 1: # only show reload warning on 1st set
                    self.helpers.confirm_changing_memory_partition_gpu_reload_warning()
                try:
                    memory_dict = {'caps': "N/A", 'current': "N/A"}
                    memory_partition_config = amdsmi_interface.amdsmi_get_gpu_memory_partition_config(args.gpu)
                    memory_dict['caps'] = str(memory_partition_config['partition_caps']).replace("]", "").replace("[", "").replace("\'", "").replace(" ", "")
                    memory_dict['current'] = memory_partition_config['mp_mode']
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug("Failed to get current memory partition for GPU %s | %s", gpu_id, e.get_error_info())
                try:
                    memory_partition = amdsmi_interface.AmdSmiMemoryPartitionType[args.memory_partition]
                    amdsmi_interface.amdsmi_set_gpu_memory_partition(args.gpu, memory_partition)
                    out = f"Successfully set memory partition to {args.memory_partition}, reload driver when ready"
                except amdsmi_exception.AmdSmiLibraryException as e:
                    out = f"[{e.get_error_info(detailed=False)}] Unable to set memory partition to {args.memory_partition}"
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        out = f"[AMDSMI_STATUS_NO_PERM] Command requires elevation"
                        self.logger.store_output(args.gpu, 'memory_partition', out)
                        self.logger.print_output()
                        self.logger.clear_multiple_devices_output()
                        raise PermissionError('Command requires elevation') from e
                    elif e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_INVAL:
                        print(f"Valid Memory partition Modes: {memory_dict['caps']}\n")
                        self.logger.store_output(args.gpu, 'memory_partition', out)
                        self.logger.print_output()
                        self.logger.clear_multiple_devices_output()
                        return
                    else:
                        self.logger.store_output(args.gpu, 'memory_partition', out)
                        self.logger.print_output()
                        self.logger.clear_multiple_devices_output()
                        return
                self.logger.store_output(args.gpu, 'memory_partition', out)
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return
            if isinstance(args.soc_pstate, int):
                try:
                    amdsmi_interface.amdsmi_set_soc_pstate(args.gpu, args.soc_pstate)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        raise PermissionError('Command requires elevation') from e
                    self.logger.store_output(args.gpu, 'socpstate', f"[{e.get_error_info(detailed=False)}] Unable to set soc pstate dpm policy to {args.soc_pstate}")
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    return
                self.logger.store_output(args.gpu, 'socpstate', f"Successfully set soc pstate dpm policy to {args.soc_pstate}")
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return
            if isinstance(args.xgmi_plpd, int):
                try:
                    amdsmi_interface.amdsmi_set_xgmi_plpd(args.gpu, args.xgmi_plpd)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        raise PermissionError('Command requires elevation') from e
                    self.logger.store_output(args.gpu, 'xgmiplpd', f"[{e.get_error_info(detailed=False)}] Unable to set XGMI per-link power down policy to {args.xgmi_plpd}")
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    return
                self.logger.store_output(args.gpu, 'xgmiplpd', f"Successfully set XGMI per-link power down policy to {args.xgmi_plpd}")
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return
            if isinstance(args.clk_level, tuple):

                clk_type = args.clk_level.clk_type
                perf_levels = args.clk_level.perf_levels
                perf_levels_str = str(perf_levels).strip('[]').replace(" ", "")
                smi_clk_type_mapping = {
                    "sclk": amdsmi_interface.AmdSmiClkType.SYS,
                    "mclk": amdsmi_interface.AmdSmiClkType.MEM,
                    "pcie": amdsmi_interface.AmdSmiClkType.PCIE,
                    "fclk": amdsmi_interface.AmdSmiClkType.DF,
                    "socclk": amdsmi_interface.AmdSmiClkType.SOC
                }
                results_clk_lvl = {'perf_level': f"Unable to set performance level to MANUAL",
                                    'get_clock_freq': f"Unable to retrieve {clk_type} frequency levels",
                                    'set_clock': f"Unable to set {clk_type} perf level(s) to {perf_levels_str}"}
                if clk_type not in smi_clk_type_mapping:
                    raise ValueError(f"Invalid clock type {clk_type}. Valid options are: {', '.join(smi_clk_type_mapping.keys())}")

                # Set perf level to manual if not already set
                try:
                    amdsmi_interface.amdsmi_set_gpu_perf_level(args.gpu, amdsmi_interface.AmdSmiDevPerfLevel.MANUAL)
                    results_clk_lvl['perf_level'] = f"Successfully set performance level to MANUAL"
                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        raise PermissionError('Command requires elevation') from e
                    results_clk_lvl['perf_level'] = f"[{e.get_error_info(detailed=False)}] Unable to set performance level to MANUAL"
                    self.logger.store_output(args.gpu, 'clk_level', results_clk_lvl)
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    return

                if clk_type.lower() == "pcie":
                    # Get PCIe bandwidth levels
                    try:
                        pcie_bandwidth_levels = amdsmi_interface.amdsmi_get_gpu_pci_bandwidth(args.gpu)
                        num_supported = pcie_bandwidth_levels['transfer_rate']['num_supported']
                        results_clk_lvl['get_clock_freq'] = f"Successfully retrieved {clk_type} frequency levels"
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        results_clk_lvl['get_clock_freq'] = f"[{e.get_error_info(detailed=False)}] Unable to retrieve {clk_type} frequency levels"
                        self.logger.store_output(args.gpu, 'clk_level', results_clk_lvl)
                        self.logger.print_output()
                        self.logger.clear_multiple_devices_output()
                        return
                else:
                    # Get clock frequency levels
                    try:
                        frequencies = amdsmi_interface.amdsmi_get_clk_freq(args.gpu, smi_clk_type_mapping[clk_type])
                        num_supported = frequencies['num_supported']
                        results_clk_lvl['get_clock_freq'] = f"Successfully retrieved {clk_type} frequency levels"
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        results_clk_lvl['get_clock_freq'] = f"[{e.get_error_info(detailed=False)}] Unable to retrieve {clk_type} frequency levels"
                        self.logger.store_output(args.gpu, 'clk_level', results_clk_lvl)
                        self.logger.print_output()
                        self.logger.clear_multiple_devices_output()
                        return

                # Validate bandwidth bitmask
                freq_bitmask = 0
                invalid_levels = []
                for level in perf_levels:
                    if level < num_supported:
                        freq_bitmask |= (1 << level)
                    else:
                        invalid_levels.append(level)

                if invalid_levels:
                    # Handle/report invalid levels
                    invalid_levels_str = str(invalid_levels).strip('[]').replace(" ", "")
                    valid_levels_str = f"Valid levels for {clk_type}: 0"
                    if num_supported > 1:
                        valid_levels_str = f"Valid levels for {clk_type}: 0-{num_supported-1}"
                    print(f"\n{valid_levels_str}\n")
                    results_clk_lvl['set_clock'] = f"Invalid level(s) {invalid_levels_str} are not within the range of supported levels for {clk_type}"
                    self.logger.store_output(args.gpu, 'clk_level', results_clk_lvl)
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    return
                else:
                    # Proceed with freq_bitmask
                    pass

                if clk_type.lower() == "pcie":
                    try:
                        amdsmi_interface.amdsmi_set_gpu_pci_bandwidth(args.gpu, freq_bitmask)
                        results_clk_lvl['set_clock'] = f"Successfully set {clk_type} perf level(s) to {perf_levels_str}"
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                            raise PermissionError('Command requires elevation') from e

                        results_clk_lvl['set_clock'] = f"[{e.get_error_info(detailed=False)}] Unable to set {clk_type} perf level(s) to {perf_levels_str}"
                        self.logger.store_output(args.gpu, 'clk_level', results_clk_lvl)
                        self.logger.print_output()
                        self.logger.clear_multiple_devices_output()
                        return
                else:
                    # For non-pcie clocks
                    try:
                        amdsmi_interface.amdsmi_set_clk_freq(args.gpu, clk_type, freq_bitmask)
                        results_clk_lvl['set_clock'] = f"Successfully set {clk_type} perf level(s) to {perf_levels_str}"
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                            raise PermissionError('Command requires elevation') from e
                        results_clk_lvl['set_clock'] = f"[{e.get_error_info(detailed=False)}] Unable to set {clk_type} perf level(s) to {perf_levels_str}"
                        self.logger.store_output(args.gpu, 'clk_level', results_clk_lvl)
                        self.logger.print_output()
                        self.logger.clear_multiple_devices_output()
                        return
                self.logger.store_output(args.gpu, 'clk_level', results_clk_lvl)
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return
        # Universal args
        if isinstance(args.power_cap, int):
            try:
                power_cap_info = amdsmi_interface.amdsmi_get_power_cap_info(args.gpu)
                logging.debug(f"Power cap info for gpu {gpu_id} | {power_cap_info}")
                min_power_cap = power_cap_info["min_power_cap"]
                min_power_cap = self.helpers.convert_SI_unit(min_power_cap, AMDSMIHelpers.SI_Unit.MICRO)
                max_power_cap = power_cap_info["max_power_cap"]
                max_power_cap = self.helpers.convert_SI_unit(max_power_cap, AMDSMIHelpers.SI_Unit.MICRO)
                current_power_cap = power_cap_info["power_cap"]
                current_power_cap = self.helpers.convert_SI_unit(current_power_cap, AMDSMIHelpers.SI_Unit.MICRO)
            except amdsmi_exception.AmdSmiLibraryException as e:
                min_power_cap = "N/A"
                max_power_cap = "N/A"
                current_power_cap = "N/A"
                self.logger.store_output(args.gpu, 'powercap', f"[{e.get_error_info(detailed=False)}] Unable to set power cap to {args.power_cap}W")
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return

            if args.power_cap == current_power_cap:
                self.logger.store_output(args.gpu, 'powercap', f"Power cap is already set to {args.power_cap}W")
            elif current_power_cap == 0:
                self.logger.store_output(args.gpu, 'powercap', f"Unable to set power cap to {args.power_cap}W, current value is {current_power_cap}W")
            elif args.power_cap >= min_power_cap and args.power_cap <= max_power_cap:
                try:
                    new_power_cap = self.helpers.convert_SI_unit(args.power_cap, AMDSMIHelpers.SI_Unit.BASE,
                                                                    AMDSMIHelpers.SI_Unit.MICRO)
                    amdsmi_interface.amdsmi_set_power_cap(args.gpu, 0, new_power_cap)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        raise PermissionError('Command requires elevation') from e
                    self.logger.store_output(args.gpu, 'powercap', f"[{e.get_error_info(detailed=False)}] Unable to set power cap to {args.power_cap}W")
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    return

                self.logger.store_output(args.gpu, 'powercap', f"Successfully set power cap to {args.power_cap}W")
            else:
                # setting power cap to 0 will return the current power cap so the technical minimum value is 1
                if min_power_cap == 0:
                    min_power_cap = 1
                self.logger.store_output(args.gpu, 'powercap', f"Power cap must be between {min_power_cap}W and {max_power_cap}W")
            self.logger.print_output()
            self.logger.clear_multiple_devices_output()
            return
        if isinstance(args.clk_limit, tuple):
            clk_type = args.clk_limit.clk_type
            lim_type = args.clk_limit.lim_type
            val = args.clk_limit.val
            val_changed = True # Assume Clock limit value is changed

            # Validate the value against the extremum
            try:
                # Parser only allows two options sclk or mclk
                if clk_type == "sclk":
                    amdsmi_clk_type =  amdsmi_interface.AmdSmiClkType.GFX
                elif clk_type == "mclk":
                    amdsmi_clk_type =  amdsmi_interface.AmdSmiClkType.MEM
                else:
                    print(f"Valid clock types are: sclk, mclk\n")
                    self.logger.store_output(args.gpu, 'clk_limit', f"Invalid clock type {args.clk_limit.clk_type}")
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    return

                clk_tuple = amdsmi_interface.amdsmi_get_clock_info(args.gpu, amdsmi_clk_type)

                if lim_type == "min":
                    if val > clk_tuple['max_clk']:
                        self.logger.store_output(args.gpu, 'clk_limit', f"Cannot set {args.clk_limit.clk_type} min value greater than max ({clk_tuple['max_clk']}MHz)")
                        self.logger.print_output()
                        self.logger.clear_multiple_devices_output()
                        return

                    if val == clk_tuple['min_clk']:
                        val_changed = False # Clock limit value did not changed

                if lim_type == "max":
                    if val < clk_tuple['min_clk']:
                        self.logger.store_output(args.gpu, 'clk_limit', f"Cannot set {args.clk_limit.clk_type} max value less than min ({clk_tuple['min_clk']}MHz)")
                        self.logger.print_output()
                        self.logger.clear_multiple_devices_output()
                        return
                    if val == clk_tuple['max_clk']:
                        val_changed = False # Clock limit value did not changed
            except amdsmi_exception.AmdSmiLibraryException as e:
                if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED and lim_type == "min" and clk_type == "mclk":
                    logging.debug("Setting mclk min is not supported")
                    self.logger.store_output(args.gpu, 'clk_limit', f"Setting mclk min is not supported")
                else:
                    logging.debug("Failed to get clock extremum info for gpu %s | %s", gpu_id, e.get_error_info())
                    self.logger.store_output(args.gpu, 'clk_limit', f"[{e.get_error_info(detailed=False)}] Unable to change {args.clk_limit.lim_type} of {args.clk_limit.clk_type} to {args.clk_limit.val}MHz")
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return

            # Set the value
            try:
                if val_changed:
                    amdsmi_interface.amdsmi_set_gpu_clk_limit(args.gpu, clk_type, lim_type, val)
            except amdsmi_exception.AmdSmiLibraryException as e:
                if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                    raise PermissionError('Command requires elevation') from e
                elif e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED and lim_type == "min" and clk_type == "mclk":
                    logging.debug("Setting mclk min is not supported")
                    self.logger.store_output(args.gpu, 'clk_limit', f"Setting mclk min is not supported")
                else:
                    self.logger.store_output(args.gpu, 'clk_limit', f"[{e.get_error_info(detailed=False)}] Unable to set {args.clk_limit.lim_type} of {args.clk_limit.clk_type} to {args.clk_limit.val}MHz")
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return

            if val_changed:
                self.logger.store_output(args.gpu, 'clk_limit', f"Successfully changed {args.clk_limit.lim_type} of {args.clk_limit.clk_type} to {args.clk_limit.val}MHz")
            else:
                self.logger.store_output(args.gpu, 'clk_limit', f"Clock limit is already set to {args.clk_limit.val}MHz")
            self.logger.print_output()
            self.logger.clear_multiple_devices_output()
            return
        if isinstance(args.process_isolation, int):
            status_string = "Enabled" if args.process_isolation else "Disabled"
            result = f"Requested process isolation to {status_string}" # This should not print out
            try:
                current_status = amdsmi_interface.amdsmi_get_gpu_process_isolation(args.gpu)
                if current_status == args.process_isolation:
                    result = f"Process isolation is already {status_string}"
                else:
                    amdsmi_interface.amdsmi_set_gpu_process_isolation(args.gpu, args.process_isolation)
                    result = f"Successfully set process isolation to {status_string}"
            except amdsmi_exception.AmdSmiLibraryException as e:
                if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                    raise PermissionError('Command requires elevation') from e
                self.logger.store_output(args.gpu, 'process_isolation', f"[{e.get_error_info(detailed=False)}] Unable to set process isolation to {status_string}")
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return

            self.logger.store_output(args.gpu, 'process_isolation', result)
            self.logger.print_output()
            self.logger.clear_multiple_devices_output()
            return

    def set_value(self, args, multiple_devices=False, gpu=None, fan=None, perf_level=None,
                  profile=None, perf_determinism=None, compute_partition=None,
                  memory_partition=None, power_cap=None,
                  cpu=None, cpu_pwr_limit=None, cpu_xgmi_link_width=None, cpu_lclk_dpm_level=None,
                  cpu_pwr_eff_mode=None, cpu_gmi3_link_width=None, cpu_pcie_link_rate=None,
                  cpu_df_pstate_range=None, cpu_enable_apb=None, cpu_disable_apb=None,
                  soc_boost_limit=None, core=None, core_boost_limit=None, soc_pstate=None, xgmi_plpd=None,
                  process_isolation=None, clk_limit=None, clk_level=None):
        """Issue reset commands to target gpu(s)

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            gpu (device_handle, optional): device_handle for target device. Defaults to None.
            fan (int, optional): Value override for args.fan. Defaults to None.
            perf_level (amdsmi_interface.AmdSmiDevPerfLevel, optional): Value override for args.perf_level. Defaults to None.
            profile (bool, optional): Value override for args.profile. Defaults to None.
            perf_determinism (int, optional): Value override for args.perf_determinism. Defaults to None.
            compute_partition (amdsmi_interface.AmdSmiComputePartitionType, optional): Value override for args.compute_partition. Defaults to None.
            memory_partition (amdsmi_interface.AmdSmiMemoryPartitionType, optional): Value override for args.memory_partition. Defaults to None.
            power_cap (int, optional): Value override for args.power_cap. Defaults to None.

            cpu (cpu_handle, optional): device_handle for target device. Defaults to None.
            cpu_pwr_limit (int, optional): Value override for args.cpu_pwr_limit. Defaults to None.
            cpu_xgmi_link_width (List[int], optional): Value override for args.cpu_xgmi_link_width. Defaults to None.
            cpu_lclk_dpm_level (List[int], optional): Value override for args.cpu_lclk_dpm_level. Defaults to None.
            cpu_pwr_eff_mode (int, optional): Value override for args.cpu_pwr_eff_mode. Defaults to None.
            cpu_gmi3_link_width (List[int], optional): Value override for args.cpu_gmi3_link_width. Defaults to None.
            cpu_pcie_link_rate (int, optional): Value override for args.cpu_pcie_link_rate. Defaults to None.
            cpu_df_pstate_range (List[int], optional): Value override for args.cpu_df_pstate_range. Defaults to None.
            cpu_enable_apb (bool, optional): Value override for args.cpu_enable_apb. Defaults to None.
            cpu_disable_apb (int, optional): Value override for args.cpu_disable_apb. Defaults to None.
            soc_boost_limit (int, optional): Value override for args.soc_boost_limit. Defaults to None.

            core (device_handle, optional): device_handle for target core. Defaults to None.
            core_boost_limit (int, optional): Value override for args.core_boost_limit. Defaults to None
            soc_pstate (int, optional): Value override for args.soc_pstate. Defaults to None.
            xgmi_plpd (int, optional): Value override for args.xgmi_plpd. Defaults to None.
            process_isolation (int, optional): Value override for args.process_isolation. Defaults to None.
        Raises:
            ValueError: Value error if no gpu value is provided
            IndexError: Index error if gpu list is empty

        Return:
            Nothing
        """
        # These are the only args checked at this point, the other args will be passed
        #   in through the applicable function set_gpu, set_cpu, or set_core function
        if gpu:
            args.gpu = gpu
        if cpu:
            args.cpu = cpu
        if core:
            args.core = core

        if not self.group_check_printed:
            self.helpers.check_required_groups()
            self.group_check_printed = True

        # Check if a GPU argument has been set
        gpu_args_enabled = False
        gpu_attributes = ["fan", "perf_level", "profile", "perf_determinism", "compute_partition",
                          "memory_partition", "power_cap", "soc_pstate", "xgmi_plpd",
                          "process_isolation", "clk_limit", "clk_level"]
        for attr in gpu_attributes:
            if hasattr(args, attr):
                if getattr(args, attr) is not None:
                    gpu_args_enabled = True
                    break
        # Check if a CPU argument has been set
        cpu_args_enabled = False
        cpu_attributes = ["cpu_pwr_limit", "cpu_xgmi_link_width", "cpu_lclk_dpm_level", "cpu_pwr_eff_mode",
                          "cpu_gmi3_link_width", "cpu_pcie_link_rate", "cpu_df_pstate_range",
                          "cpu_enable_apb", "cpu_disable_apb", "soc_boost_limit"]
        for attr in cpu_attributes:
            if hasattr(args, attr):
                if getattr(args, attr) not in [None, False]:
                    cpu_args_enabled = True
                    break

        # Check if a Core argument has been set
        core_args_enabled = False
        core_attributes = ["core_boost_limit"]
        for attr in core_attributes:
            if hasattr(args, attr):
                if getattr(args, attr) is not None:
                    core_args_enabled = True
                    break

        # Error if no subcommand args are passed
        if self.helpers.is_baremetal():
            is_gpu_set = False
            is_cpu_set = False
            is_core_set = False
            try:
                is_gpu_set = any([
                            args.gpu is not None,
                            args.fan is not None,
                            args.perf_level is not None,
                            args.profile is not None,
                            args.perf_determinism is not None,
                            args.compute_partition is not None,
                            args.memory_partition is not None,
                            args.power_cap is not None,
                            args.soc_pstate is not None,
                            args.xgmi_plpd is not None,
                            args.clk_limit is not None,
                            args.clk_level is not None,
                            args.process_isolation is not None
                            ])
            except AttributeError:
                # If attribute error for gpu, then we could be another subcommand
                pass

            try:
                is_cpu_set = any([
                            args.cpu is not None,
                            args.cpu_pwr_limit is not None,
                            args.cpu_xgmi_link_width is not None,
                            args.cpu_lclk_dpm_level is not None,
                            args.cpu_pwr_eff_mode is not None,
                            args.cpu_gmi3_link_width is not None,
                            args.cpu_pcie_link_rate is not None,
                            args.cpu_df_pstate_range is not None,
                            args.cpu_enable_apb,
                            args.cpu_disable_apb is not None,
                            args.soc_boost_limit is not None
                            ])
            except AttributeError:
                # If attribute error for cpu, then we could be another subcommand
                pass
            try:
                if args.core_boost_limit:
                    is_core_set = True
            except AttributeError:
                # If attribute error for core, then we could be another subcommand
                pass

            if not (is_gpu_set or is_cpu_set or is_core_set):
                # if neither GPU / CPU / or Core args are provided, then raise error message
                command = " ".join(sys.argv[1:])
                raise AmdSmiRequiredCommandException(command, self.logger.format)
        else:
            if not any([args.process_isolation is not None, args.clk_limit is not None, args.power_cap is not None]):
                command = " ".join(sys.argv[1:])
                raise AmdSmiRequiredCommandException(command, self.logger.format)

        # Only allow one device's arguments to be set at a time
        if not any([gpu_args_enabled, cpu_args_enabled, core_args_enabled]):
            raise ValueError('No GPU, CPU, or CORE arguments provided, specific arguments are needed')
        elif all([gpu_args_enabled, cpu_args_enabled, core_args_enabled]):
            raise ValueError('Cannot set GPU, CPU, and CORE arguments at the same time')
        elif not (gpu_args_enabled ^ cpu_args_enabled ^ core_args_enabled):
            raise ValueError('Cannot set GPU, CPU, or CORE arguments at the same time')

        if self.helpers.is_amdgpu_initialized() and gpu_args_enabled:
            if args.gpu == None:
                args.gpu = self.device_handles

        if self.helpers.is_amd_hsmp_initialized() and cpu_args_enabled:
            if args.cpu == None:
                args.cpu = self.cpu_handles

        if self.helpers.is_amd_hsmp_initialized() and core_args_enabled:
            if args.core == None:
                args.core = self.core_handles


        # Handle CPU and GPU intialization cases
        if self.helpers.is_amd_hsmp_initialized() and self.helpers.is_amdgpu_initialized():
            # Print out all CPU and all GPU static info only if no device was specified.
            # If a GPU or CPU argument is provided only print out the specified device.
            if args.cpu == None and args.gpu == None and args.core == None:
                raise ValueError('No GPU, CPU, or CORE provided, specific target(s) are needed')

            if args.cpu:
                self.set_cpu(args, multiple_devices, cpu, cpu_pwr_limit,
                                cpu_xgmi_link_width, cpu_lclk_dpm_level, cpu_pwr_eff_mode,
                                cpu_gmi3_link_width, cpu_pcie_link_rate, cpu_df_pstate_range,
                                cpu_enable_apb, cpu_disable_apb, soc_boost_limit)
            if args.core:
                self.logger.output = {}
                self.logger.clear_multiple_devices_output()
                self.set_core(args, multiple_devices, core, core_boost_limit)
            if args.gpu:
                self.logger.output = {}
                self.logger.clear_multiple_devices_output()
                self.set_gpu(args, multiple_devices, gpu, fan, perf_level,
                                profile, perf_determinism, compute_partition,
                                memory_partition, power_cap, soc_pstate, xgmi_plpd,
                                process_isolation, clk_limit, clk_level)
        elif self.helpers.is_amd_hsmp_initialized(): # Only CPU is initialized
            if args.cpu == None and args.core == None:
                raise ValueError('No CPU or CORE provided, specific target(s) are needed')
            if args.cpu:
                self.set_cpu(args, multiple_devices, cpu, cpu_pwr_limit,
                                cpu_xgmi_link_width, cpu_lclk_dpm_level, cpu_pwr_eff_mode,
                                cpu_gmi3_link_width, cpu_pcie_link_rate, cpu_df_pstate_range,
                                cpu_enable_apb, cpu_disable_apb, soc_boost_limit)
            if args.core:
                self.logger.output = {}
                self.logger.clear_multiple_devices_output()
                self.set_core(args, multiple_devices, core, core_boost_limit)
        elif self.helpers.is_amdgpu_initialized(): # Only GPU is initialized
            if args.gpu == None:
                args.gpu = self.device_handles
            self.logger.clear_multiple_devices_output()
            self.set_gpu(args, multiple_devices, gpu, fan, perf_level,
                            profile, perf_determinism, compute_partition,
                            memory_partition, power_cap, soc_pstate, xgmi_plpd,
                            process_isolation, clk_limit, clk_level)


    def reset(self, args, multiple_devices=False, gpu=None, gpureset=None,
                clocks=None, fans=None, profile=None, xgmierr=None, perf_determinism=None,
                power_cap=None, reload_driver=None, clean_local_data=None):
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
        if reload_driver:
            args.reload_driver = reload_driver
        if clean_local_data:
            args.clean_local_data = clean_local_data

        # Handle No GPU passed
        if args.gpu == None:
            args.gpu = self.device_handles

        if not self.group_check_printed:
            self.helpers.check_required_groups()
            self.group_check_printed = True

        # Handle multiple GPUs
        handled_multiple_gpus, device_handle = self.helpers.handle_gpus(args, self.logger, self.reset)
        if handled_multiple_gpus:
            return # This function is recursive

        args.gpu = device_handle

        # Get gpu_id for logging
        gpu_id = self.helpers.get_gpu_id_from_device_handle(args.gpu)

        # Error if no subcommand args are passed
        if self.helpers.is_baremetal():
            if not any([args.gpureset, args.clocks, args.fans, args.profile, args.xgmierr, \
                        args.perf_determinism, args.power_cap, args.reload_driver, \
                        args.clean_local_data]):
                command = " ".join(sys.argv[1:])
                raise AmdSmiRequiredCommandException(command, self.logger.format)
        else:
            if not any([args.clean_local_data, args.reload_driver]):
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
                        result = 'Successfully reset GPU'
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                            raise PermissionError('Command requires elevation') from e
                        result = f"[{e.get_error_info(detailed=False)}] Unable to reset GPU"
                        self.logger.store_output(args.gpu, 'gpu_reset', result)
                        self.logger.print_output()
                        self.logger.clear_multiple_devices_output()
                        return
                else:
                    result = 'Unable to reset non-amd GPU'
                self.logger.store_output(args.gpu, 'gpu_reset', result)
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return
            if args.clocks:
                reset_clocks_results = {'overdrive': '',
                                        'clocks': '',
                                        'performance': ''}
                try:
                    amdsmi_interface.amdsmi_set_gpu_overdrive_level(args.gpu, 0)
                    reset_clocks_results['overdrive'] = 'Overdrive set to 0'
                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        raise PermissionError('Command requires elevation') from e
                    logging.debug("Failed to reset overdrive on gpu %s | %s", gpu_id, e.get_error_info())
                    reset_clocks_results['overdrive'] = f"[{e.get_error_info(detailed=False)}] Unable to reset overdrive to 0"
                    # continue to reset clocks and performance level
                try:
                    level_auto = amdsmi_interface.AmdSmiDevPerfLevel.AUTO
                    amdsmi_interface.amdsmi_set_gpu_perf_level(args.gpu, level_auto)
                    reset_clocks_results['clocks'] = 'Successfully reset performance level to auto'
                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        raise PermissionError('Command requires elevation') from e
                    reset_clocks_results['clocks'] = f"[{e.get_error_info(detailed=False)}] Unable to reset performance level to auto"
                    logging.debug("Failed to reset perf level on gpu %s | %s", gpu_id, e.get_error_info())

                try:
                    #TODO: Check why this is called twice?
                    level_auto = amdsmi_interface.AmdSmiDevPerfLevel.AUTO
                    amdsmi_interface.amdsmi_set_gpu_perf_level(args.gpu, level_auto)
                    reset_clocks_results['performance'] = 'Successfully reset performance level to auto'
                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        raise PermissionError('Command requires elevation') from e
                    reset_clocks_results['performance'] = f"[{e.get_error_info(detailed=False)}] Unable to reset performance level to auto"
                    logging.debug("Failed to reset perf level on gpu %s | %s", gpu_id, e.get_error_info())

                self.logger.store_output(args.gpu, 'reset_clocks', reset_clocks_results)
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return
            if args.fans:
                try:
                    amdsmi_interface.amdsmi_reset_gpu_fan(args.gpu, 0)
                    result = 'Successfully reset fan speed to driver control'
                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        raise PermissionError('Command requires elevation') from e
                    result = f"[{e.get_error_info(detailed=False)}] Unable to reset fan speed to driver control"
                    logging.debug("Failed to reset fans on gpu %s | %s", gpu_id, e.get_error_info())
                    self.logger.store_output(args.gpu, 'reset_fans', result)
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    return
                self.logger.store_output(args.gpu, 'reset_fans', result)
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return
            if args.profile:
                reset_profile_results = {'power_profile' : 'N/A'}
                try:
                    power_profile_mask = amdsmi_interface.AmdSmiPowerProfilePresetMasks.BOOTUP_DEFAULT
                    amdsmi_interface.amdsmi_set_gpu_power_profile(args.gpu, 0, power_profile_mask)
                    reset_profile_results['power_profile'] = 'Successfully reset Power Profile to default (bootup default)'
                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        raise PermissionError('Command requires elevation') from e
                    reset_profile_results['power_profile'] = f"[{e.get_error_info(detailed=False)}] Unable to reset Power Profile to default (bootup default)"
                    logging.debug("Failed to reset power profile on gpu %s | %s", gpu_id, e.get_error_info())

                self.logger.store_output(args.gpu, 'reset_profile', reset_profile_results)
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return
            if args.xgmierr:
                try:
                    amdsmi_interface.amdsmi_reset_gpu_xgmi_error(args.gpu)
                    result = 'Successfully reset XGMI Error count'
                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        raise PermissionError('Command requires elevation') from e
                    logging.debug("Failed to reset xgmi error count on gpu %s | %s", gpu_id, e.get_error_info())
                    result = f"[{e.get_error_info(detailed=False)}] Unable to reset XGMI Error count"
                    self.logger.store_output(args.gpu, 'reset_xgmi_err', result)
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    return
                self.logger.store_output(args.gpu, 'reset_xgmi_err', result)
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return
            if args.perf_determinism:
                try:
                    level_auto = amdsmi_interface.AmdSmiDevPerfLevel.AUTO
                    amdsmi_interface.amdsmi_set_gpu_perf_level(args.gpu, level_auto)
                    result = 'Successfully reset Performance Level to default (auto)'
                except amdsmi_exception.AmdSmiLibraryException as e:
                    if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        raise PermissionError('Command requires elevation') from e
                    logging.debug("Failed to set perf level on gpu %s | %s", gpu_id, e.get_error_info())
                    result = f"[{e.get_error_info(detailed=False)}] Unable to reset Performance Level to default (auto)"
                    self.logger.store_output(args.gpu, 'reset_perf_determinism', result)
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    return
                self.logger.store_output(args.gpu, 'reset_perf_determinism', result)
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return
            if args.power_cap:
                try:
                    power_cap_info = amdsmi_interface.amdsmi_get_power_cap_info(args.gpu)
                    logging.debug(f"Power cap info for gpu {gpu_id} | {power_cap_info}")
                    default_power_cap_in_w = power_cap_info["default_power_cap"]
                    default_power_cap_in_w = self.helpers.convert_SI_unit(default_power_cap_in_w, AMDSMIHelpers.SI_Unit.MICRO)
                    current_power_cap_in_w = power_cap_info["power_cap"]
                    current_power_cap_in_w = self.helpers.convert_SI_unit(current_power_cap_in_w, AMDSMIHelpers.SI_Unit.MICRO)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    self.logger.store_output(args.gpu, 'powercap', f"[{e.get_error_info(detailed=False)}] Unable to reset power cap to default")
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    return

                if current_power_cap_in_w == default_power_cap_in_w:
                    self.logger.store_output(args.gpu, 'powercap', f"Power cap is already set to {default_power_cap_in_w}W")
                else:
                    try:
                        default_power_cap_in_uw = self.helpers.convert_SI_unit(default_power_cap_in_w,
                                                                               AMDSMIHelpers.SI_Unit.BASE,
                                                                               AMDSMIHelpers.SI_Unit.MICRO)
                        amdsmi_interface.amdsmi_set_power_cap(args.gpu, 0, default_power_cap_in_uw)
                    except amdsmi_exception.AmdSmiLibraryException as e:
                        if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                            raise PermissionError('Command requires elevation') from e
                        raise ValueError(f"Unable to reset power cap to {default_power_cap_in_w} on GPU {gpu_id}") from e
                    self.logger.store_output(args.gpu, 'powercap', f"Successfully set power cap to {default_power_cap_in_w}W")
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return

        #######################
        # BM commands - END   #
        #######################

        if args.clean_local_data:
            try:
                amdsmi_interface.amdsmi_clean_gpu_local_data(args.gpu)
                result = 'Successfully clean GPU local data'
            except amdsmi_exception.AmdSmiLibraryException as e:
                if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                    raise PermissionError('Command requires elevation') from e
                result = f"[{e.get_error_info(detailed=False)}] Unable to clean local data"
                self.logger.store_output(args.gpu, 'clean_local_data', result)
                self.logger.print_output()
                self.logger.clear_multiple_devices_output()
                return
            self.logger.store_output(args.gpu, 'clean_local_data', result)
            self.logger.print_output()
            self.logger.clear_multiple_devices_output()
            return

        # Adding to VMs since, they should also support same reload as baremetal
        if args.reload_driver:
            # Check permissions BEFORE starting any processes
            # Required to avoid permission errors when starting the progress bar
            try:
                if os.geteuid() != 0:
                    result = "[AMDSMI_STATUS_NO_PERM] Command requires elevation"
                    self.logger.store_output(args.gpu, 'reload_driver', result)
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    raise PermissionError('Command requires elevation')
            except AttributeError:
                pass # os.geteuid() not available on Windows
            lock = multiprocessing.Lock()
            lock.acquire()
            is_lock_released = False
            progress_process = None
            try:
                self.helpers.increment_set_count()
                set_count = self.helpers.get_set_count()
                if set_count == 1:
                    self.helpers.confirm_gpu_driver_reload_warning()
                    # Start progress bar in separate process
                    string_out = f"Reloading driver for all AMD GPUs:"
                    progress_process = multiprocessing.Process(
                        target=self.helpers.showProgressbar,
                        args=(string_out, 140, True)
                    )
                    progress_process.start()
                    # Perform the actual driver reload (this is where permission error occurs)
                    amdsmi_interface.amdsmi_gpu_driver_reload()
                    # If we get here, operation was successful
                    self.helpers.assign_previous_set_success_check(amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_SUCCESS)
                    result = "Successfully reloaded driver"
                else:
                    if self.helpers.get_previous_set_success_check() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_SUCCESS:
                        result = "Successfully reloaded driver"
                    elif self.helpers.get_previous_set_success_check() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                        result = "[AMDSMI_STATUS_NO_PERM] Command requires elevation"
                        raise PermissionError('Command requires elevation')
                    else:
                        previous_check = self.helpers.get_previous_set_success_check()
                        temp_exception = amdsmi_exception.AmdSmiLibraryException(previous_check)
                        str_out = temp_exception.get_error_info(detailed=False)
                        result = f"[{str_out}] Unable to successfully restart driver"
            except amdsmi_exception.AmdSmiLibraryException as e:
                # Handle permission error FIRST, before any cleanup
                self.helpers.assign_previous_set_success_check(e.get_error_code())
                if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                    self.helpers.assign_previous_set_success_check(amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM)
                    result = f"[{e.get_error_info(detailed=False)}] Command requires elevation"
                    # Clean termination of progress bar
                    if progress_process and progress_process.is_alive():
                        progress_process.terminate()
                        progress_process.join(timeout=0.1)  # Wait max 0.1 second
                        if progress_process.is_alive():
                            progress_process.kill()  # Force kill if needed
                        print("\n")  # Clean up progress bar line
                    # Store result and exit early
                    self.logger.store_output(args.gpu, 'reload_driver', result)
                    self.logger.print_output()
                    self.logger.clear_multiple_devices_output()
                    if not is_lock_released:
                        lock.release()
                        is_lock_released = True
                    raise PermissionError('Command requires elevation') from e
                else:
                    # Handle other errors
                    self.helpers.assign_previous_set_success_check(e.get_error_code())
                    result = f"[{e.get_error_info(detailed=False)}] Unable to successfully restart driver"
            finally:
                # Always clean up progress bar process
                if progress_process and progress_process.is_alive():
                    progress_process.terminate()
                    progress_process.join(timeout=0.1)
                    if progress_process.is_alive():
                        progress_process.kill()
                    print("\n")  # Clean up progress bar line
                # Always release lock
                if not is_lock_released:
                    lock.release()
                    is_lock_released = True
            # Store and print result
            self.logger.store_output(args.gpu, 'reload_driver', result)
            self.logger.print_output()
            self.logger.clear_multiple_devices_output()
            return

    def monitor(self, args, multiple_devices=False, watching_output=False, gpu=None,
                    watch=None, watch_time=None, iterations=None, power_usage=None,
                    temperature=None, gfx_util=None, mem_util=None, encoder=None,
                    decoder=None, ecc=None, vram_usage=None, pcie=None, process=None,
                    violation=None):
        """ Populate a table with each GPU as an index to rows of targeted data

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            gpu (device_handle, optional): device_handle for target device. Defaults to None.
            watch (bool, optional): Value override for args.watch. Defaults to None.
            watch_time (int, optional): Value override for args.watch_time. Defaults to None.
            iterations (int, optional): Value override for args.iterations. Defaults to None.
            power_usage (bool, optional): Value override for args.power_usage. Defaults to None.
            temperature (bool, optional): Value override for args.temperature. Defaults to None.
            gfx (bool, optional): Value override for args.gfx. Defaults to None.
            mem_util (bool, optional): Value override for args.mem. Defaults to None.
            encoder (bool, optional): Value override for args.encoder. Defaults to None.
            decoder (bool, optional): Value override for args.decoder. Defaults to None.
            ecc (bool, optional): Value override for args.ecc. Defaults to None.
            vram_usage (bool, optional): Value override for args.vram_usage. Defaults to None.
            pcie (bool, optional): Value override for args.pcie. Defaults to None.
            process (bool, optional): Value override for args.process. Defaults to None.
            violation (bool, optional): Value override for args.violation. Defaults to None.

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

        # monitor args
        if power_usage:
            args.power_usage = power_usage
        if temperature:
            args.temperature = temperature
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
        if not any([args.power_usage, args.temperature, args.gfx, args.mem,
                    args.encoder, args.decoder, args.ecc, args.vram_usage,
                    args.pcie, args.violation]):
            args.power_usage = args.temperature = args.gfx = args.mem = \
                args.encoder = args.decoder = args.vram_usage = True
            # set extra args for default output filtering
            args.default_output = True
        else:
            if not hasattr(args, 'default_output'):
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

                # Store output from multiple devices without printing to console
                for device_handle in args.gpu:
                    self.monitor(args, multiple_devices=True, watching_output=watching_output, gpu=device_handle)

                # Reload original gpus
                args.gpu = stored_gpus

                dual_csv_output = False
                if args.process:
                    if self.logger.is_csv_format():
                        dual_csv_output = True

                # Flush the output
                self.logger.print_output(multiple_device_enabled=True,
                                          watching_output=watching_output,
                                          tabular=True,
                                          dual_csv_output=dual_csv_output)

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
        self.logger.table_header = 'GPU'
        if watching_output:
            self.logger.store_output(args.gpu, 'timestamp', int(time.time()))
            self.logger.table_header = 'TIMESTAMP'.rjust(10) + '  ' + self.logger.table_header

        if args.loglevel == "DEBUG":
            try:
                # Get GPU Metrics table version
                gpu_metric_version_info = amdsmi_interface.amdsmi_get_gpu_metrics_header_info(args.gpu)
                gpu_metric_version_str = json.dumps(gpu_metric_version_info, indent=4)
                logging.debug("GPU Metrics table Version for GPU %s | %s", gpu_id, gpu_metric_version_str)
            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug("#4 - Unable to load GPU Metrics table version for %s | %s", gpu_id, e.get_error_info())

            try:
                # Get GPU Metrics table
                gpu_metric_debug_info = amdsmi_interface.amdsmi_get_gpu_metrics_info(args.gpu)

            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug("#5 - Unable to load GPU Metrics table for %s | %s", gpu_id, e.get_error_info())

        #get metric info only once per gpu, this will speed up data output
        try:
            # Get GPU Metrics table
            gpu_metrics_info = amdsmi_interface.amdsmi_get_gpu_metrics_info(args.gpu)
            if args.loglevel == "DEBUG":
                gpu_metric_debug_info = json.dumps(gpu_metrics_info, indent=4)
                logging.debug("GPU Metrics table for GPU %s | %s", gpu_id, gpu_metric_debug_info)
        except amdsmi_exception.AmdSmiLibraryException as e:
            gpu_metrics_info = amdsmi_interface._NA_amdsmi_get_gpu_metrics_info()
            logging.debug("Unable to load GPU Metrics table for %s | %s", gpu_id, e.get_error_info())

        # Workaround for XCP (partition) metrics not providing num_partition in v1.0
        # Confirmed with driver team that we can default to 1 if num_partition is not defined.
        # Pending partitions exist, ie. partition_id > 0. See logic below.
        try:
            partition_id = amdsmi_interface.amdsmi_get_gpu_kfd_info(args.gpu)['current_partition_id']
        except amdsmi_exception.AmdSmiLibraryException as e:
            logging.debug("Failed to get current partition id for gpu %s | %s", gpu_id, e.get_error_info())
            partition_id = "N/A"

        num_partition = gpu_metrics_info['num_partition']
        if num_partition == "N/A":
            num_partition = partition_id

        num_xcp = num_partition  # used later for XCP metrics
        self.logger.table_header += 'XCP'.rjust(5, ' ')
        self.logger.store_output(args.gpu, 'xcp', partition_id)  # Starting with partition_id.
                                                                 # Outputs which have xcp details
                                                                 # will update this value via num_xcp.
                                                                 # This value will help map to primary device.

        # Store the pcie_bw values due to possible increase in bandwidth due to repeated gpu_metrics calls
        if args.pcie:
            try:
                pcie_info = amdsmi_interface.amdsmi_get_pcie_info(args.gpu)['pcie_metric']
            except amdsmi_exception.AmdSmiLibraryException as e:
                pcie_info = "N/A"
                logging.debug("Failed to get pci bandwidth on gpu %s | %s", gpu_id, e.get_error_info())

        power_unit = 'W'

        # Resume regular ordering of values
        if args.power_usage:
            try:
                if gpu_metrics_info['current_socket_power'] != "N/A":
                    monitor_values['power_usage'] = gpu_metrics_info['current_socket_power']
                else: # Fallback to average_socket_power for older gpu_metrics versions
                    monitor_values['power_usage'] = gpu_metrics_info['average_socket_power']

                if self.logger.is_human_readable_format() and monitor_values['power_usage'] != "N/A":
                    monitor_values['power_usage'] = f"{monitor_values['power_usage']} {power_unit}"
                if self.logger.is_json_format() and monitor_values['power_usage'] != "N/A":
                    monitor_values['power_usage'] = {"value" : monitor_values['power_usage'],
                                                     "unit" : power_unit}

            except (KeyError, amdsmi_exception.AmdSmiLibraryException) as e:
                monitor_values['power_usage'] = "N/A"
                logging.debug("Failed to get power usage on gpu %s | %s", gpu_id, e)

            self.logger.table_header += 'POWER'.rjust(7)

        if args.power_usage and not args.default_output:
            # Get Current Power Cap
            try:
                power_cap_info = amdsmi_interface.amdsmi_get_power_cap_info(args.gpu)
                monitor_values['max_power'] = power_cap_info['power_cap']  # Get current power cap (`power_cap`) socket is set to
                                                                           # `max_power_cap`, is the maximum value it can be set to
                monitor_values['max_power'] = self.helpers.convert_SI_unit(monitor_values['max_power'], AMDSMIHelpers.SI_Unit.MICRO)

                if self.logger.is_human_readable_format() and monitor_values['max_power'] != "N/A":
                    monitor_values['max_power'] = f"{monitor_values['max_power']} {power_unit}"
                if self.logger.is_json_format() and monitor_values['max_power'] != "N/A":
                    monitor_values['max_power'] = {"value" : monitor_values['max_power'],
                                                     "unit" : power_unit}
            except amdsmi_exception.AmdSmiLibraryException as e:
                monitor_values['max_power'] = "N/A"
                logging.debug("Failed to get power cap info for gpu %s | %s", gpu_id, e.get_error_info())

            self.logger.table_header += 'PWR_CAP'.rjust(9)

        if args.temperature:
            try:
                temperature = gpu_metrics_info['temperature_hotspot']
                monitor_values['hotspot_temperature'] = temperature
            except (KeyError, amdsmi_exception.AmdSmiLibraryException) as e:
                monitor_values['hotspot_temperature'] = "N/A"
                logging.debug("Failed to get hotspot temperature on gpu %s | %s", gpu_id, e)

            try:
                temperature = gpu_metrics_info['temperature_mem']
                monitor_values['memory_temperature'] = temperature
            except (KeyError, amdsmi_exception.AmdSmiLibraryException) as e:
                monitor_values['memory_temperature'] = "N/A"
                logging.debug("Failed to get memory temperature on gpu %s | %s", gpu_id, e)

            temp_unit_human_readable = '\N{DEGREE SIGN}C'
            temp_unit_json = 'C'
            if monitor_values['hotspot_temperature'] != "N/A":
                if self.logger.is_human_readable_format():
                    monitor_values['hotspot_temperature'] = f"{monitor_values['hotspot_temperature']} {temp_unit_human_readable}"
                if self.logger.is_json_format():
                    monitor_values['hotspot_temperature'] = {"value" : monitor_values['hotspot_temperature'],
                                                            "unit" : temp_unit_json}
            if monitor_values['memory_temperature'] != "N/A":
                if self.logger.is_human_readable_format():
                    monitor_values['memory_temperature'] = f"{monitor_values['memory_temperature']} {temp_unit_human_readable}"
                if self.logger.is_json_format():
                    monitor_values['memory_temperature'] = {"value" : monitor_values['memory_temperature'],
                                                            "unit" : temp_unit_json}

            self.logger.table_header += 'GPU_T'.rjust(8)
            self.logger.table_header += 'MEM_T'.rjust(8)

        if args.gfx:
            try:
                gfx_clk = gpu_metrics_info['current_gfxclk']
                monitor_values['gfx_clk'] = gfx_clk
                freq_unit = 'MHz'
                if gfx_clk != "N/A":
                    if self.logger.is_human_readable_format():
                        monitor_values['gfx_clk'] = f"{monitor_values['gfx_clk']} {freq_unit}"
                    if self.logger.is_json_format():
                        monitor_values['gfx_clk'] = {"value" : monitor_values['gfx_clk'],
                                                     "unit" : freq_unit}

            except (KeyError, amdsmi_exception.AmdSmiLibraryException) as e:
                monitor_values['gfx_clk'] = "N/A"
                logging.debug("Failed to get gfx clock on gpu %s | %s", gpu_id, e)

            self.logger.table_header += 'GFX_CLK'.rjust(10)

            try:
                gfx_util = gpu_metrics_info['average_gfx_activity']
                activity_unit = '%'
                if gfx_util != "N/A":
                    monitor_values['gfx'] = gfx_util
                if self.logger.is_human_readable_format():
                    monitor_values['gfx'] = f"{monitor_values['gfx']} {activity_unit}"
                if self.logger.is_json_format():
                    monitor_values['gfx'] = {"value" : monitor_values['gfx'],
                                                 "unit" : activity_unit}
            except (KeyError, amdsmi_exception.AmdSmiLibraryException) as e:
                monitor_values['gfx'] = "N/A"
                logging.debug("Failed to get gfx utilization on gpu %s | %s", gpu_id, e)

            self.logger.table_header += 'GFX%'.rjust(7)

        if args.mem:
            try:
                mem_util = gpu_metrics_info['average_umc_activity']
                activity_unit = '%'
                if mem_util != "N/A":
                    monitor_values['mem'] = mem_util
                if self.logger.is_human_readable_format():
                    monitor_values['mem'] = f"{monitor_values['mem']} {activity_unit}"
                if self.logger.is_json_format():
                    monitor_values['mem'] = {"value" : monitor_values['mem'],
                                             "unit" : activity_unit}
            except (KeyError, amdsmi_exception.AmdSmiLibraryException) as e:
                monitor_values['mem'] = "N/A"
                logging.debug("Failed to get mem utilization on gpu %s | %s", gpu_id, e)

            self.logger.table_header += 'MEM%'.rjust(7)

            # don't populate mem clock on default output
            if not args.default_output:
                try:
                    mem_clock = gpu_metrics_info['current_uclk']
                    monitor_values['mem_clock'] = mem_clock
                    freq_unit = 'MHz'
                    if mem_clock != "N/A":
                        if self.logger.is_human_readable_format():
                            monitor_values['mem_clock'] = f"{monitor_values['mem_clock']} {freq_unit}"
                        if self.logger.is_json_format():
                            monitor_values['mem_clock'] = {"value" : monitor_values['mem_clock'],
                                                        "unit" : freq_unit}
                except (KeyError, amdsmi_exception.AmdSmiLibraryException) as e:
                    monitor_values['mem_clock'] = "N/A"
                    logging.debug("Failed to get mem clock on gpu %s | %s", gpu_id, e)

                self.logger.table_header += 'MEM_CLOCK'.rjust(11)

        if args.encoder:
            # TODO: The encoding utilization is in progress for Navi. Note: MI3x ASICs only support decoding.
            try:
                # Get List of vcn activity values
                encoder_util = "N/A" # Not yet implemented
                encoding_activity_avg = []
                for value in encoder_util:
                    if isinstance(value, int):
                        encoding_activity_avg.append(value)

                # Averaging the possible encoding activity values
                if encoding_activity_avg:
                    encoding_activity_avg = round(sum(encoding_activity_avg) / len(encoding_activity_avg))
                else:
                    encoding_activity_avg = "N/A"

                monitor_values['encoder'] = encoding_activity_avg

                activity_unit = '%'
                if monitor_values['encoder'] != "N/A":
                    if self.logger.is_human_readable_format():
                        monitor_values['encoder'] = f"{monitor_values['encoder']} {activity_unit}"
                    if self.logger.is_json_format():
                        monitor_values['encoder'] = {"value" : monitor_values['encoder'],
                                                    "unit" : activity_unit}
            except amdsmi_exception.AmdSmiLibraryException as e:
                monitor_values['encoder'] = "N/A"
                logging.debug("Failed to get encoder utilization on gpu %s | %s", gpu_id, e.get_error_info())

            self.logger.table_header += 'ENC%'.rjust(7)

        if args.decoder:
            try:
                # Get List of vcn activity values
                # Note: MI3x ASICs only support decoding, so the vcn_activity/vcn_busy
                #       is used for decoding activity.
                decoder_util = gpu_metrics_info['vcn_activity']
                if (gpu_metrics_info['vcn_activity'][0] == "N/A" and
                    gpu_metrics_info['xcp_stats.vcn_busy'][partition_id][0] != "N/A"):
                    decoder_util = gpu_metrics_info['xcp_stats.vcn_busy'][partition_id]
                decoding_activity_avg = self.helpers.average_flattened_ints(decoder_util, context="decoder_util")
                monitor_values['decoder'] = decoding_activity_avg

                activity_unit = '%'
                if monitor_values['decoder'] != "N/A":
                    if self.logger.is_human_readable_format():
                        monitor_values['decoder'] = f"{monitor_values['decoder']} {activity_unit}"
                    if self.logger.is_json_format():
                        monitor_values['decoder'] = {"value" : monitor_values['decoder'],
                                                    "unit" : activity_unit}
            except (KeyError, amdsmi_exception.AmdSmiLibraryException) as e:
                monitor_values['decoder'] = "N/A"
                logging.debug("Failed to get decoder utilization on gpu %s | %s", gpu_id, e)

            self.logger.table_header += 'DEC%'.rjust(7)

        if (args.encoder or args.decoder) and not args.default_output:
            try:
                vclock = gpu_metrics_info['current_vclk0']
                monitor_values['vclock'] = vclock

                freq_unit = 'MHz'
                if vclock != "N/A":
                    if self.logger.is_human_readable_format():
                        monitor_values['vclock'] = f"{monitor_values['vclock']} {freq_unit}"
                    if self.logger.is_json_format():
                        monitor_values['vclock'] = {"value" : monitor_values['vclock'],
                                                           "unit" : freq_unit}
            except (KeyError, amdsmi_exception.AmdSmiLibraryException) as e:
                monitor_values['vclock'] = "N/A"
                logging.debug("Failed to get dclock on gpu %s | %s", gpu_id, e)

            self.logger.table_header += 'VCLOCK'.rjust(10)

            try:
                dclock = gpu_metrics_info['current_dclk0']
                monitor_values['dclock'] = dclock

                freq_unit = 'MHz'
                if dclock != "N/A":
                    if self.logger.is_human_readable_format():
                        monitor_values['dclock'] = f"{monitor_values['dclock']} {freq_unit}"
                    if self.logger.is_json_format():
                        monitor_values['dclock'] = {"value" : monitor_values['dclock'],
                                                           "unit" : freq_unit}
            except (KeyError, amdsmi_exception.AmdSmiLibraryException) as e:
                monitor_values['dclock'] = "N/A"
                logging.debug("Failed to get vclock on gpu %s | %s", gpu_id, e)

            self.logger.table_header += 'DCLOCK'.rjust(10)

        if args.ecc:
            try:
                ecc = amdsmi_interface.amdsmi_get_gpu_total_ecc_count(args.gpu)
                monitor_values['single_bit_ecc'] = ecc['correctable_count']
                monitor_values['double_bit_ecc'] = ecc['uncorrectable_count']
            except amdsmi_exception.AmdSmiLibraryException as e:
                monitor_values['ecc'] = "N/A"
                logging.debug("Failed to get ecc on gpu %s | %s", gpu_id, e.get_error_info())

            self.logger.table_header += 'SINGLE_ECC'.rjust(12)
            self.logger.table_header += 'DOUBLE_ECC'.rjust(12)

            try:
                pcie_metric = amdsmi_interface.amdsmi_get_pcie_info(args.gpu)['pcie_metric']
                logging.debug("PCIE Metric for %s | %s", gpu_id, pcie_metric)
                monitor_values['pcie_replay'] = pcie_metric['pcie_replay_count']
            except amdsmi_exception.AmdSmiLibraryException as e:
                monitor_values['pcie_replay'] = "N/A"
                logging.debug("Failed to get gpu_metrics pcie replay counter on gpu %s | %s", gpu_id, e.get_error_info())

            if monitor_values['pcie_replay'] == "N/A":
                try:
                    pcie_replay = amdsmi_interface.amdsmi_get_gpu_pci_replay_counter(args.gpu)
                    monitor_values['pcie_replay'] = pcie_replay
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug("Failed to get sysfs pcie replay counter on gpu %s | %s", gpu_id, e.get_error_info())

            self.logger.table_header += 'PCIE_REPLAY'.rjust(13)

        if args.vram_usage and not args.default_output:
            try:
                vram_used = amdsmi_interface.amdsmi_get_gpu_memory_usage(args.gpu, amdsmi_interface.AmdSmiMemoryType.VRAM) // (1024*1024)
                vram_total = amdsmi_interface.amdsmi_get_gpu_memory_total(args.gpu, amdsmi_interface.AmdSmiMemoryType.VRAM) // (1024*1024)
                monitor_values['vram_used'] = vram_used
                monitor_values['vram_free'] = vram_total - vram_used
                monitor_values['vram_total'] = vram_total
                if vram_total != 0:
                    monitor_values['vram_percent'] = round ((vram_used / vram_total) * 100, 2)
                else:
                    monitor_values['vram_percent'] = "N/A"

                vram_usage_unit = "MB"
                vram_percent_unit = "%"
                if self.logger.is_human_readable_format():
                    monitor_values['vram_used'] = f"{monitor_values['vram_used']} {vram_usage_unit}"
                    monitor_values['vram_free'] = f"{monitor_values['vram_free']} {vram_usage_unit}"
                    monitor_values['vram_total'] = f"{monitor_values['vram_total']} {vram_usage_unit}"
                    monitor_values['vram_percent'] = f"{monitor_values['vram_percent']} {vram_percent_unit}"
                if self.logger.is_json_format():
                    monitor_values['vram_used'] = {"value" : monitor_values['vram_used'],
                                                   "unit" : vram_usage_unit}
                    monitor_values['vram_free'] = {"value" : monitor_values['vram_free'],
                                                   "unit" : vram_usage_unit}
                    monitor_values['vram_total'] = {"value" : monitor_values['vram_total'],
                                                    "unit" : vram_usage_unit}
                    monitor_values['vram_percent'] = {"value" : monitor_values['vram_percent'],
                                                      "unit" : vram_percent_unit}
            except amdsmi_exception.AmdSmiLibraryException as e:
                monitor_values['vram_used'] = "N/A"
                monitor_values['vram_free'] = "N/A"
                monitor_values['vram_total'] = "N/A"
                monitor_values['vram_percent'] = "N/A"
                logging.debug("Failed to get vram memory usage on gpu %s | %s", gpu_id, e.get_error_info())

            self.logger.table_header += 'VRAM_USED'.rjust(11)
            self.logger.table_header += 'VRAM_FREE'.rjust(12)
            self.logger.table_header += 'VRAM_TOTAL'.rjust(12)
            self.logger.table_header += 'VRAM%'.rjust(9)

        if args.vram_usage and args.default_output:
            try:
                vram_used = amdsmi_interface.amdsmi_get_gpu_memory_usage(args.gpu, amdsmi_interface.AmdSmiMemoryType.VRAM) // (1024*1024)
                vram_total = amdsmi_interface.amdsmi_get_gpu_memory_total(args.gpu, amdsmi_interface.AmdSmiMemoryType.VRAM) // (1024*1024)
                vram_usage_unit = "GB"
                if self.logger.is_json_format():
                    monitor_values['vram_used'] = {"value" : round(vram_used/1024,1),
                                                   "unit" : vram_usage_unit}
                    monitor_values['vram_total'] = {"value" : round(vram_total/1024,1),
                                                    "unit" : vram_usage_unit}
                elif self.logger.is_csv_format():
                    monitor_values['vram_used'] = round(vram_used/1024,1)
                    monitor_values['vram_total'] = round(vram_total/1024,1)
                else:
                    monitor_values['vram_usage'] = f"{vram_used/1024:5.1f}/{vram_total/1024:5.1f} {vram_usage_unit}".rjust(16,' ')
            except amdsmi_exception.AmdSmiLibraryException as e:
                if self.logger.is_json_format():
                    monitor_values['vram_used'] = "N/A"
                    monitor_values['vram_total'] = "N/A"
                else:
                    monitor_values['vram_usage'] = "N/A"
                logging.debug("Failed to get vram memory usage on gpu %s | %s", gpu_id, e.get_error_info())

            self.logger.table_header += 'VRAM_USAGE'.rjust(16)

        if args.pcie:
            if pcie_info != "N/A":
                pcie_bw_unit = 'Mb/s'
                monitor_values['pcie_bw'] = self.helpers.unit_format(self.logger, pcie_info['pcie_bandwidth'], pcie_bw_unit)
            else:
                monitor_values['pcie_bw'] = pcie_info

            self.logger.table_header += 'PCIE_BW'.rjust(12)

        # initialize dual_csv_format; applicable to process only
        dual_csv_output = False

        # Store process list separately
        if args.process:
            # Populate initial processes
            try:
                process_list = amdsmi_interface.amdsmi_get_gpu_process_list(args.gpu)
            except amdsmi_exception.AmdSmiLibraryException as e:
                if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                    raise PermissionError('Command requires elevation') from e
                logging.debug("Failed to get process list for gpu %s | %s", gpu_id, e.get_error_info())
                raise e

            try:
                num_compute_units = amdsmi_interface.amdsmi_get_gpu_asic_info(args.gpu)['num_compute_units']
            except (KeyError, amdsmi_exception.AmdSmiLibraryException) as e:
                num_compute_units = "N/A"
                logging.debug("Failed to get num compute units for gpu %s | %s", gpu_id, e.get_error_info())

            # Clean processes dictionary
            filtered_process_values = []
            for process_info in process_list:
                process_info.pop('engine_usage')  # Remove 'engine_usage' value
                process_info['mem_usage'] = process_info.pop('mem')
                process_info['cu_occupancy'] = process_info.pop('cu_occupancy')

                memory_usage_unit = "B"

                if self.logger.is_human_readable_format():
                    process_info['mem_usage'] = self.helpers.convert_bytes_to_readable(process_info['mem_usage'])
                    for usage_metric in process_info['memory_usage']:
                        process_info["memory_usage"][usage_metric] = self.helpers.convert_bytes_to_readable(process_info["memory_usage"][usage_metric])
                    memory_usage_unit = ""

                process_info['mem_usage'] = self.helpers.unit_format(self.logger,
                                                                     process_info['mem_usage'],
                                                                     memory_usage_unit)

                for usage_metric in process_info['memory_usage']:
                    process_info['memory_usage'][usage_metric] = self.helpers.unit_format(self.logger,
                                                                                          process_info['memory_usage'][usage_metric],
                                                                                          memory_usage_unit)

                if 'cu_occupancy' in process_info:
                    try:
                        cu_occupancy = process_info['cu_occupancy']
                        if num_compute_units != "N/A" and num_compute_units > 0 and cu_occupancy != "N/A":
                            cu_percentage = round((cu_occupancy / num_compute_units) * 100, 1)
                            process_info['cu_occupancy'] = self.helpers.unit_format(self.logger,
                                                                                    cu_percentage,
                                                                                    '%')
                        else:
                            process_info['cu_occupancy'] = "N/A"
                    except Exception as e:
                        process_info['cu_occupancy'] = "N/A"
                        logging.debug("Failed to calculate cu_occupancy percentage for GPU %s | %s", gpu_id, str(e))

                filtered_process_values.append({'process_info': process_info})

            # If no processes are populated then we populate an N/A placeholder
            if not filtered_process_values:
                logging.debug("Monitor - Failed to detect any process on gpu %s", gpu_id)
                filtered_process_values.append({'process_info': "N/A"})

            for index, process in enumerate(filtered_process_values):
                if process['process_info'] == "N/A":
                    filtered_process_values[index]['process_info'] = "No running processes detected"

            # Build the process table's title and header
            self.logger.secondary_table_title = "PROCESS INFO"
            self.logger.secondary_table_header = 'GPU'.rjust(3) + "NAME".rjust(19) + "PID".rjust(9) + "GTT_MEM".rjust(10) + \
                                                "CPU_MEM".rjust(10) + "VRAM_MEM".rjust(10) + "MEM_USG".rjust(10) + "CU%".rjust(9)

            if watching_output:
                self.logger.secondary_table_header = 'TIMESTAMP'.rjust(10) + '  ' + self.logger.secondary_table_header

            logging.debug(f"Monitor - Process Info for GPU {gpu_id} | {filtered_process_values}")

            if self.logger.is_json_format():
                self.logger.store_output(args.gpu, 'process_list', filtered_process_values)

            if self.logger.is_human_readable_format():
                # Print out process in flattened format
                # The logger detects if process list is present and pulls it out and prints
                #  that table with timestamp, gpu, and prints headers separately
                self.logger.store_output(args.gpu, 'process_list', filtered_process_values)

            if self.logger.is_csv_format():
                dual_csv_output = True
                # The logger detects if process list is present and pulls it out and prints
                #  that table with timestamp, gpu, and prints headers separately
                self.logger.store_output(args.gpu, 'process_list', filtered_process_values)

        ###################
        ### XCP Metrics ###
        ###################
        # Must come after process list - XCP detail is a multi-dimensional array, which is displayed
        # in tabular format with XCP values for same gpu shown on muliple lines.
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
                "low_utilviol": "N/A"
            }
            try:
                violations = amdsmi_interface.amdsmi_get_violation_status(args.gpu)
                violation_status['pviol'] = violations['per_ppt_pwr']
                violation_status['tviol'] = violations['per_socket_thrm']
                violation_status['tviol_active'] = violations['active_socket_thrm']
                violation_status['phot_tviol'] = violations['per_prochot_thrm']
                violation_status['vr_tviol'] = violations['per_vr_thrm']
                violation_status['hbm_tviol'] = violations['per_hbm_thrm']
                violation_status['gfx_clkviol'] = violations['per_gfx_clk_below_host_limit']
                violation_status['gfxclk_pviol'] = violations['per_gfx_clk_below_host_limit_pwr']
                violation_status['gfxclk_tviol'] = violations['per_gfx_clk_below_host_limit_thm']
                violation_status['gfxclk_totalviol'] = violations['per_gfx_clk_below_host_limit_total']
                violation_status['low_utilviol'] = violations['per_low_utilization']
            except amdsmi_exception.AmdSmiLibraryException as e:
                monitor_values['pviol'] = violation_status['pviol']
                monitor_values['tviol'] = violation_status['tviol']
                monitor_values['tviol_active'] = violation_status['tviol_active']
                monitor_values['phot_tviol'] = violation_status['phot_tviol']
                monitor_values['vr_tviol'] = violation_status['vr_tviol']
                monitor_values['hbm_tviol'] = violation_status['hbm_tviol']
                monitor_values['gfx_clkviol'] = violation_status['gfx_clkviol']
                monitor_values['gfxclk_pviol'] = violation_status['gfxclk_pviol']
                monitor_values['gfxclk_tviol'] = violation_status['gfxclk_tviol']
                monitor_values['gfxclk_totalviol'] = violation_status['gfxclk_totalviol']
                monitor_values['low_utilviol'] = violation_status['low_utilviol']
                logging.debug("Failed to get violation status on gpu %s | %s", gpu_id, e.get_error_info())
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
                        if key == 'tviol_active' or key == 'xcp':
                            monitor_values[key] = value
                        else:
                            monitor_values[key] = self.helpers.unit_format(self.logger, violation_status[key], violation_status_unit)
                    else:
                        monitor_values[key] = violation_status[key]
                else:
                    if num_partition != "N/A":
                        # these are one after another, in order to display each in sub-sections
                        new_xcp_dict = {}
                        for current_xcp in range(num_partition):
                            new_xcp_dict[f"xcp_{current_xcp}"] = self.helpers.unit_format(self.logger, value[current_xcp], "%")
                        monitor_values[key] = new_xcp_dict
                    else:
                        monitor_values[key] = value[0] if value else "N/A"
            # save deep copy of monitor values, used later to grab xcp specific values
            monitor_values_deepcopy = copy.deepcopy(monitor_values)

            self.logger.table_header += 'PVIOL'.rjust(kPVIOL_MAX_WIDTH, ' ')
            self.logger.table_header += 'TVIOL'.rjust(kTVIOL_MAX_WIDTH, ' ')
            self.logger.table_header += 'TVIOL_ACTIVE'.rjust(kTVIOL_ACTIVE_MAX_WIDTH, ' ')
            self.logger.table_header += 'PHOT_TVIOL'.rjust(kPHOT_MAX_WIDTH, ' ')
            self.logger.table_header += 'VR_TVIOL'.rjust(kVR_MAX_WIDTH, ' ')
            self.logger.table_header += 'HBM_TVIOL'.rjust(kHBM_MAX_WIDTH, ' ')
            self.logger.table_header += 'GFX_CLKVIOL'.rjust(kGFXC_MAX_WIDTH, ' ')
            self.logger.table_header += 'GFXCLK_PVIOL'.rjust(kGFXC_PVIOL_MAX_WIDTH, ' ')
            self.logger.table_header += 'GFXCLK_TVIOL'.rjust(kGFXC_TVIOL_MAX_WIDTH, ' ')
            self.logger.table_header += 'GFXCLK_TOTALVIOL'.rjust(kGFXC_TOTALVIOL_MAX_WIDTH, ' ')
            self.logger.table_header += 'LOW_UTILVIOL'.rjust(kLOW_UTILVIOL_MAX_WIDTH, ' ')

            # Print/capture by XCPs
            if num_partition != "N/A" and partition_id == 0:
                current_xcp = 0
                while (current_xcp in range(num_partition) or current_xcp == 0):
                    if not multiple_devices and watching_output and current_xcp == 0:
                        # Need to clear output for single device, otherwise while watching output
                        # XCP detail will continue stacking on top of each other
                        self.logger.clear_multiple_devices_output()

                    if watching_output:
                        self.logger.store_output(args.gpu, 'timestamp', int(time.time()))

                    if current_xcp != 0:  # set all other values without XCP stats to N/A
                        self.logger.store_output(args.gpu, 'xcp', current_xcp)
                        monitor_values['pviol'] = "N/A"
                        monitor_values['tviol'] = "N/A"
                        monitor_values['tviol_active'] = "N/A"
                        monitor_values['phot_tviol'] = "N/A"
                        monitor_values['vr_tviol'] = "N/A"
                        monitor_values['hbm_tviol'] = "N/A"
                        monitor_values['gfx_clkviol'] = "N/A"
                        for k, _ in monitor_values.items():  # change other keys to "N/A" since we should have all applicable XCP stats
                                                             # eg. amd-smi monitor -p -t -V should only show XCP info for violations
                                                             # below primary device
                            if k != 'xcp' and k not in ['gfxclk_pviol', 'gfxclk_tviol', 'gfxclk_totalviol', 'low_utilviol']:
                                monitor_values[k] = "N/A"

                    if isinstance(monitor_values_deepcopy['gfxclk_pviol'], dict):
                        monitor_values['gfxclk_pviol'] = monitor_values_deepcopy['gfxclk_pviol'][f"xcp_{current_xcp}"]
                    if isinstance(monitor_values_deepcopy['gfxclk_tviol'], dict):
                        monitor_values['gfxclk_tviol'] = monitor_values_deepcopy['gfxclk_tviol'][f"xcp_{current_xcp}"]
                    if isinstance(monitor_values_deepcopy['gfxclk_totalviol'], dict):
                        monitor_values['gfxclk_totalviol'] = monitor_values_deepcopy['gfxclk_totalviol'][f"xcp_{current_xcp}"]
                    if isinstance(monitor_values_deepcopy['low_utilviol'], dict):
                        monitor_values['low_utilviol'] = monitor_values_deepcopy['low_utilviol'][f"xcp_{current_xcp}"]

                    if self.logger.is_human_readable_format():
                        monitor_values['pviol'] = monitor_values['pviol']
                        monitor_values['tviol'] = monitor_values['tviol']
                        monitor_values['phot_tviol'] = monitor_values['phot_tviol']
                        monitor_values['vr_tviol'] = monitor_values['vr_tviol']
                        monitor_values['hbm_tviol'] = monitor_values['hbm_tviol']
                        monitor_values['gfx_clkviol'] = monitor_values['gfx_clkviol']
                        monitor_values['gfxclk_pviol'] = str(monitor_values['gfxclk_pviol']).replace('\'', '')
                        monitor_values['gfxclk_tviol'] = str(monitor_values['gfxclk_tviol']).replace('\'', '')
                        monitor_values['gfxclk_totalviol'] = str(monitor_values['gfxclk_totalviol']).replace('\'', '')
                        monitor_values['low_utilviol'] = str(monitor_values['low_utilviol']).replace('\'', '')
                    self.logger.store_output(args.gpu, 'values', monitor_values)
                    self.logger.store_multiple_device_output()
                    current_xcp += 1
            else:
                self.logger.store_output(args.gpu, 'xcp', num_xcp)
                self.logger.store_output(args.gpu, 'values', monitor_values)

        # Store typical output for all commands (XCP data will be handled separately, eg. violation status)
        if not args.violation:
            self.logger.store_output(args.gpu, 'values', monitor_values)

        # Now handling the single gpu case only
        if multiple_devices:
            self.logger.store_multiple_device_output()
            return

        if watching_output and not self.logger.destination == "stdout": # End of single gpu add to watch_output
            self.logger.store_watch_output(multiple_device_enabled=False)


        if args.violation:
            # Print violation status for single gpu, which have different xcp information per 1 gpu
            self.logger.print_output(multiple_device_enabled=True, watching_output=watching_output, tabular=True, dual_csv_output=dual_csv_output)
        else:
            # Print the output for single gpu, which currently does not have multiple xcp information
            self.logger.print_output(multiple_device_enabled=False, watching_output=watching_output, tabular=True, dual_csv_output=dual_csv_output)


    def xgmi(self, args, multiple_devices=False, gpu=None, metric=None, xgmi_link_status=None):
        """ Get topology information for target gpus
            params:
                args - argparser args to pass to subcommand
                multiple_devices (bool) - True if checking for multiple devices
                gpu (device_handle) - device_handle for target device
                metric (bool) - Value override for args.metric
                xgmi_link_status (bool) - Value override for args.xgmi_link_status

            return:
                Nothing
        """
        # Not supported with partitions

        # Set args.* to passed in arguments
        if gpu:
            args.gpu = gpu
        if metric:
            args.metric = metric
        if xgmi_link_status:
            args.link_status = xgmi_link_status

        # Handle No GPU passed
        if args.gpu == None:
            args.gpu = self.device_handles

        if not isinstance(args.gpu, list):
            args.gpu = [args.gpu]

        # Handle all args being false
        if not any([args.metric, args.link_status]):
            args.metric = True
            args.link_status = True

        # Clear the table header
        self.logger.table_header = ''.rjust(7)

        if not self.group_check_printed:
            self.helpers.check_required_groups()
            self.group_check_printed = True

        # Populate the possible gpus and their bdfs
        xgmi_values = []
        for gpu in args.gpu:
            primary_partition = self.helpers.is_primary_partition(gpu)
            if not primary_partition:
                logging.debug(f"Skipping xgmi command due to non zero partition {gpu}")
                continue

            logging.debug("check1 device_handle: %s", gpu)
            gpu_id = self.helpers.get_gpu_id_from_device_handle(gpu)
            gpu_bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(gpu)
            xgmi_values.append({"gpu" : gpu_id,
                                "bdf" : gpu_bdf})
            # Populate header with just bdfs
            self.logger.table_header += gpu_bdf.rjust(13)

        if args.metric:
            # prepend link metrics header to the table header
            link_metrics_header = "       " + "bdf".ljust(14) + \
                  "bit_rate".ljust(10) + "max_bandwidth".ljust(15) + \
                  "link_type".ljust(11)
            self.logger.table_header = link_metrics_header + self.logger.table_header.strip()

            # Populate dictionary according to format
            for xgmi_dict in xgmi_values:
                src_gpu_id = xgmi_dict['gpu']
                src_gpu_bdf = xgmi_dict['bdf']
                src_gpu = amdsmi_interface.amdsmi_get_processor_handle_from_bdf(src_gpu_bdf)
                logging.debug("check2 device_handle: %s", src_gpu)
                # This should be the same order as the check1

                xgmi_dict['link_metrics'] = {
                    "bit_rate" : "N/A",
                    "max_bandwidth" : "N/A",
                    "link_type" : "N/A",
                    "links" : []
                }
                xgmi_metrics_info = {"links": []}

                try:
                    xgmi_metrics_info = amdsmi_interface.amdsmi_get_link_metrics(src_gpu)
                    bitrate = xgmi_metrics_info['links'][0]['bit_rate']
                    max_bandwidth = xgmi_metrics_info['links'][0]['max_bandwidth']
                except amdsmi_exception.AmdSmiLibraryException as e:
                    bitrate = "N/A"
                    max_bandwidth = "N/A"
                    logging.debug("Failed to get bitrate and bandwidth for GPU %s | %s", src_gpu_id,
                                    e.get_error_info())

                # Populate bitrate and max_bandwidth with units logic
                bw_unit = 'Gb/s'
                if self.logger.is_human_readable_format():
                    xgmi_dict['link_metrics']['bit_rate'] = f"{bitrate} {bw_unit}"
                    xgmi_dict['link_metrics']['max_bandwidth'] = f"{max_bandwidth} {bw_unit}"
                elif self.logger.is_json_format():
                    xgmi_dict['link_metrics']['bit_rate'] = {"value" : bitrate,
                                                             "unit" : bw_unit}
                    xgmi_dict['link_metrics']['max_bandwidth'] = {"value" : max_bandwidth,
                                                                  "unit" : bw_unit}
                elif self.logger.is_csv_format():
                    xgmi_dict['link_metrics']['bit_rate'] = bitrate
                    xgmi_dict['link_metrics']['max_bandwidth'] = max_bandwidth

                # Populate link metrics
                for dest_gpu in args.gpu:
                    primary_partition = self.helpers.is_primary_partition(dest_gpu)
                    if not primary_partition:
                        continue

                    dest_gpu_id = self.helpers.get_gpu_id_from_device_handle(dest_gpu)
                    dest_gpu_bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(dest_gpu)
                    dest_link_dict = {
                        "gpu" : dest_gpu_id,
                        "bdf" : dest_gpu_bdf,
                        "read" : 0,
                        "write" : 0,
                    }

                    found = False
                    for link in xgmi_metrics_info['links']:
                        if link['bdf'] == dest_gpu_bdf:
                            # Accumulate read/write if multiple links have the same bdf
                            dest_link_dict['read'] += link['read']
                            dest_link_dict['write'] += link['write']
                            found = True
                    if not found:
                        dest_link_dict['read'] = "N/A"
                        dest_link_dict['write'] = "N/A"
                    else:
                        data_unit = 'KB'
                        if self.logger.is_human_readable_format():
                            dest_link_dict['read'] = self.helpers.convert_bytes_to_readable(dest_link_dict['read'] * 1024, True)
                            dest_link_dict['write'] = self.helpers.convert_bytes_to_readable(dest_link_dict['write'] * 1024, True)
                        elif self.logger.is_json_format():
                            dest_link_dict['read'] = {"value" : dest_link_dict['read'],
                                                    "unit" : data_unit}
                            dest_link_dict['write'] = {"value" : dest_link_dict['write'],
                                                    "unit" : data_unit}

                        try:
                            link_type = amdsmi_interface.amdsmi_topo_get_link_type(src_gpu, dest_gpu)['type']
                            if xgmi_dict['link_metrics']['link_type'] != "XGMI" and isinstance(link_type, int):
                                if link_type == amdsmi_interface.amdsmi_wrapper.AMDSMI_LINK_TYPE_INTERNAL:
                                    xgmi_dict['link_metrics']['link_type'] = "UNKNOWN"
                                elif link_type == amdsmi_interface.amdsmi_wrapper.AMDSMI_LINK_TYPE_PCIE:
                                    xgmi_dict['link_metrics']['link_type'] = "PCIE"
                                elif link_type == amdsmi_interface.amdsmi_wrapper.AMDSMI_LINK_TYPE_XGMI:
                                    xgmi_dict['link_metrics']['link_type'] = "XGMI"
                        except amdsmi_exception.AmdSmiLibraryException as e:
                            logging.debug("Failed to get link type for %s to %s | %s",
                                            self.helpers.get_gpu_id_from_device_handle(src_gpu),
                                            self.helpers.get_gpu_id_from_device_handle(dest_gpu),
                                            e.get_error_info())

                    xgmi_dict['link_metrics']['links'].append(dest_link_dict)

            # Handle printing for tabular format
            if self.logger.is_human_readable_format():
                # Populate tabular output
                tabular_output = []
                for xgmi_dict in xgmi_values:
                    tabular_output_dict = {}

                    # Create GPU row and add to tabular_output
                    for key, value in xgmi_dict.items():
                        if key == "gpu":
                            tabular_output_dict["gpu#"] = f"GPU{value}"
                        if key == "bdf":
                            tabular_output_dict["bdf"] = value
                        if key == "link_metrics":
                            for link_key, link_value in value.items():
                                if link_key == "bit_rate":
                                    tabular_output_dict["bit_rate"] = link_value
                                if link_key == "max_bandwidth":
                                    tabular_output_dict["max_bandwidth"] = link_value
                                if link_key == "link_type":
                                    tabular_output_dict["link_type"] = link_value
                    tabular_output.append(tabular_output_dict)

                    # Create Read and Write rows and add to tabular_output
                    read_output_dict = {"RW" : " Read"}
                    write_output_dict = {"RW" : " Write"}
                    for key, value in xgmi_dict.items():
                        if key == "link_metrics":
                            for link_key, link_value in value.items():
                                if link_key == "links":
                                    for link in link_value:
                                        read_output_dict[f"bdf_{link['gpu']}"] = link["read"]
                                        write_output_dict[f"bdf_{link['gpu']}"] = link["write"]
                    tabular_output.append(read_output_dict)
                    tabular_output.append(write_output_dict)

                # Print out the tabular output
                self.logger.multiple_device_output = tabular_output
                self.logger.table_title = "\nLINK METRIC TABLE"
                self.logger.print_output(multiple_device_enabled=True, tabular=True)

        self.logger.multiple_device_output = xgmi_values

        if self.logger.is_csv_format():
            new_output = []
            for elem in self.logger.multiple_device_output:
                new_output.append(self.logger.flatten_dict(elem, topology_override=True))
            self.logger.multiple_device_output = new_output

        if self.logger.is_json_format():
            self.logger.store_xgmi_metric_json_output.append(xgmi_values)
            if not args.link_status:
                self.logger.combine_arrays_to_json()
        elif not self.logger.is_human_readable_format():
            self.logger.print_output(multiple_device_enabled=True)

        if args.link_status:
            # Header modification
            self.logger.table_header = ''.rjust(7)
            current_header = "     ".ljust(7) + \
                             "bdf".ljust(14) + \
                             "link_status".ljust(20)
            self.logger.table_header = current_header + self.logger.table_header.strip()
            # Process each GPU
            tabular_output = []
            for xgmi_dict in xgmi_values:
                src_gpu_id = xgmi_dict['gpu']
                src_gpu_bdf = xgmi_dict['bdf']
                src_gpu = amdsmi_interface.amdsmi_get_processor_handle_from_bdf(src_gpu_bdf)

                # Populate link statuses
                status_row = []
                tabular_output_dict = {"gpu#": f"GPU{src_gpu_id}",
                                       "gpu": src_gpu_id,
                                       "bdf": src_gpu_bdf,
                                       "link_status": "N/A"}
                try:
                    link_status = amdsmi_interface.amdsmi_get_gpu_xgmi_link_status(src_gpu)
                    tabular_output_dict['link_status'] = link_status['status']
                    if self.logger.is_human_readable_format():
                        del tabular_output_dict['gpu']
                    else:
                        del tabular_output_dict['gpu#']
                    tabular_output.append(tabular_output_dict)
                    if self.logger.is_json_format():
                        self.logger.store_xgmi_link_status_json_output.append(tabular_output_dict)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    xgmi_dict['link_metrics']['link_status']={"status": "failed"}
                    logging.debug("Failed to get XGMI link status for GPU %s | %s", src_gpu_id, e.get_error_info())

                #populate link status data for output
                if self.logger.is_human_readable_format():
                    xgmi_dict['link_status'] = tabular_output
            self.logger.multiple_device_output= tabular_output
            self.logger.table_title = "\nXGMI LINK STATUS"
            if not self.logger.is_json_format():
                self.logger.print_output(multiple_device_enabled=True, tabular=True)
            self.logger.clear_multiple_devices_output()
            if self.logger.is_json_format():
                self.logger.combine_arrays_to_json()
            if self.logger.is_human_readable_format():
            # Populate the legend output
                legend_parts = [
                    "\n\nLegend:",
                    "  SELF = Current GPU",
                    "  N/A = Not supported",
                    "  U / D / X = Link is Up / Down / Disabled",
                    "  Read / Write = GPU Metric Accumulated Read / Write"
                ]
                legend_output = "\n".join(legend_parts)

                if self.logger.destination == 'stdout':
                    print(legend_output)
                else:
                    with self.logger.destination.open('a', encoding="utf-8") as output_file:
                        output_file.write(legend_output + '\n')


    def partition(self, args, multiple_devices=False, gpu=None, current=None, memory=None, accelerator=None):
        """ Display parition information for the target GPU
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
            self.logger.table_header = ''.rjust(7)
            current_header = "GPU_ID".ljust(8) + \
                             "MEMORY".ljust(8) + \
                             "ACCELERATOR_TYPE".ljust(18) + \
                             "ACCELERATOR_PROFILE_INDEX".ljust(27) + \
                             "PARTITION_ID".ljust(14)
            self.logger.table_header = current_header + self.logger.table_header.strip()

            tabular_output = []
            for gpu in args.gpu:
                gpu_id = self.helpers.get_gpu_id_from_device_handle(gpu)
                try:
                    partition_dict = amdsmi_interface.amdsmi_get_gpu_accelerator_partition_profile(gpu)
                    partition_id = str(partition_dict['partition_id']).replace("[", "").replace("]", "").replace(" ", "")
                    profile_type = partition_dict['partition_profile']['profile_type']
                    profile_index = partition_dict['partition_profile']['profile_index']
                except amdsmi_exception.AmdSmiLibraryException as e:
                    profile_type = "N/A"
                    profile_index = "N/A"
                    partition_id = str(partition_dict['partition_id']).replace("[", "").replace("]", "").replace(" ", "")
                    logging.debug("Failed to get accelerator partition profile for GPU %s | %s", gpu_id, e.get_error_info())
                try:
                    current_mem_cap = amdsmi_interface.amdsmi_get_gpu_memory_partition(gpu)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    current_mem_cap = "N/A"
                    logging.debug("Failed to get current memory partition capabilties for GPU %s | %s", gpu_id, e.get_error_info())

                if profile_type == 0:
                    profile_type = "N/A"

                tabular_output_dict = {"gpu_id": gpu_id,
                                       "memory": current_mem_cap,
                                       "accelerator_type": profile_type,
                                       "accelerator_profile_index": profile_index,
                                       "partition_id": partition_id}
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
            self.logger.table_header = ''.rjust(7)
            current_header = "GPU_ID".ljust(8) + \
                             "MEMORY_PARTITION_CAPS".ljust(23) + \
                             "CURRENT_MEMORY_PARTITION".ljust(26)
            self.logger.table_header = current_header + self.logger.table_header.strip()

            for gpu in args.gpu:
                gpu_id = self.helpers.get_gpu_id_from_device_handle(gpu)
                mem_caps_str = "N/A"
                current_memory_partition = "N/A"
                try:
                    memory_partition_config = amdsmi_interface.amdsmi_get_gpu_memory_partition_config(gpu)
                    mem_caps_str = str(memory_partition_config['partition_caps']).replace("]", "").replace("[", "").replace("\'", "").replace(" ", "")
                    current_memory_partition = memory_partition_config['mp_mode']
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug("Failed to get current memory partition for GPU %s | %s", gpu_id, e.get_error_info())

                tabular_output_dict = {"gpu_id": gpu_id,
                                       "memory_partition_caps": mem_caps_str,
                                       "current_memory_partition": current_memory_partition}
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
            self.logger.table_header = ''.rjust(7)
            current_header = "GPU_ID".ljust(8) + \
                             "PROFILE_INDEX".ljust(15) + \
                             "MEMORY_PARTITION_CAPS".ljust(23) + \
                             "ACCELERATOR_TYPE".ljust(18) + \
                             "PARTITION_ID".ljust(17) + \
                             "NUM_PARTITIONS".ljust(16) + \
                             "NUM_RESOURCES".ljust(15) + \
                             "RESOURCE_INDEX".ljust(16) + \
                             "RESOURCE_TYPE".ljust(15) + \
                             "RESOURCE_INSTANCES".ljust(20) + \
                             "RESOURCES_SHARED".ljust(18)
            self.logger.table_header = current_header + self.logger.table_header.strip()

            tabular_output = []
            prev_gpu_id = "N/A"
            for gpu in args.gpu:
                gpu_id = self.helpers.get_gpu_id_from_device_handle(gpu)
                tabular_output_dict = {"gpu_id": gpu_id,
                                       "profile_index": "N/A",
                                       "memory_partition_caps": "N/A",
                                       "accelerator_type": "N/A",
                                       "partition_id": "0",
                                       "num_partitions": "N/A",
                                       "num_resources": "N/A",
                                       "resource_index": "N/A",
                                       "resource_type": "N/A",
                                       "resource_instances": "N/A",
                                       "resources_shared": "N/A"}
                try:
                    partition_dict = amdsmi_interface.amdsmi_get_gpu_accelerator_partition_profile(gpu)
                    partition_id = str(partition_dict['partition_id']).replace("[", "").replace("]", "").replace(" ", "")
                    current_accelerator_type = partition_dict['partition_profile']['profile_type']
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
                    logging.debug("Failed to get accelerator partition profile for GPU %s | %s", gpu_id, e.get_error_info())

                try:
                    partition_config_dict = amdsmi_interface.amdsmi_get_gpu_accelerator_partition_profile_config(gpu)
                    logging.debug("amdsmi_commands.py | partition_config_dict: " + str(json.dumps(partition_config_dict, indent=4)))
                    num_profiles = partition_config_dict['num_profiles']
                    num_resource_profiles = partition_config_dict['num_resource_profiles']

                    resource_index = 0
                    prev_accelerator_type = "N/A"
                    for p in range(0, num_profiles):
                        accelerator_type = partition_config_dict['profiles'][p]['profile_type']
                        profile_index = partition_config_dict['profiles'][p]['profile_index']
                        num_partitions = partition_config_dict['profiles'][p]['num_partitions']
                        mem_caps_str = str(partition_config_dict['profiles'][p]['memory_caps']).replace("]", "").replace("[", "").replace("\'", "").replace(" ", "")
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
                            if prev_accelerator_type != accelerator_type: # only print the first instance of the resources
                                tabular_output_dict = {"gpu_id": tabular_gpu_id,
                                       "profile_index": profile_index,
                                       "memory_partition_caps": mem_caps_str,
                                       "accelerator_type": accelerator_type,
                                       "partition_id": partition_id,
                                       "num_partitions": num_partitions,
                                       "num_resources": num_resource_profiles,
                                       "resource_index": "N/A",
                                       "resource_type": "N/A",
                                       "resource_instances": "N/A",
                                       "resources_shared": "N/A"}
                                prev_accelerator_type = accelerator_type
                                tabular_output.append(tabular_output_dict)
                            continue

                        for r in range(0, num_resource_profiles):
                            logging.debug("amdsmi_commands.py | p: " + str(p) + "; r: " + str(r)
                                  + "; accelerator_type: " + str(accelerator_type))
                            resource_type = partition_config_dict['profiles'][p]['resources'][r]['resource_type']
                            resource_instances = partition_config_dict['profiles'][p]['resources'][r]['partition_resource']
                            resources_shared = partition_config_dict['profiles'][p]['resources'][r]['num_partitions_share_resource']
                            if prev_accelerator_type != accelerator_type: # only print the first instance of the resources
                                tabular_output_dict = {"gpu_id": tabular_gpu_id,
                                       "profile_index": profile_index,
                                       "memory_partition_caps": mem_caps_str,
                                       "accelerator_type": accelerator_type,
                                       "partition_id": partition_id,
                                       "num_partitions": num_partitions,
                                       "num_resources": num_resource_profiles,
                                       "resource_index": resource_index,
                                       "resource_type": resource_type,
                                       "resource_instances": resource_instances,
                                       "resources_shared": resources_shared}
                                prev_accelerator_type = accelerator_type
                            else:
                                tabular_output_dict = {"gpu_id": "",
                                       "profile_index": "",
                                       "memory_partition_caps": "",
                                       "accelerator_type": "",
                                       "partition_id": "",
                                       "num_partitions": "",
                                       "num_resources": "",
                                       "resource_index": resource_index,
                                       "resource_type": resource_type,
                                       "resource_instances": resource_instances,
                                       "resources_shared": resources_shared}
                            resource_index += 1
                            tabular_output.append(tabular_output_dict)
                except amdsmi_exception.AmdSmiLibraryException as e:
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
            self.logger.warning_message = "" # clear the warning message

            #########################################
            # print accelerator partition resources #
            #########################################
            self.logger.table_header = ''.rjust(7)
            current_header = "RESOURCE_INDEX".ljust(16) + \
                             "RESOURCE_TYPE".ljust(15) + \
                             "RESOURCE_INSTANCES".ljust(20) + \
                             "RESOURCES_SHARED".ljust(18)
            self.logger.table_header = current_header + self.logger.table_header.strip()

            tabular_output = []
            for gpu in args.gpu:
                gpu_id = self.helpers.get_gpu_id_from_device_handle(gpu)
                tabular_output_dict = {"resource_index": "N/A",
                                       "resource_type": "N/A",
                                       "resource_instances": "N/A",
                                       "resources_shared": "N/A"}
                try:
                    partition_config_dict = amdsmi_interface.amdsmi_get_gpu_accelerator_partition_profile_config(gpu)
                    logging.debug("amdsmi_commands.py | partition_config_dict: " + str(json.dumps(partition_config_dict, indent=4)))
                    num_profiles = partition_config_dict['num_profiles']
                    num_resource_profiles = partition_config_dict['num_resource_profiles']

                    if num_resource_profiles == 0:
                        tabular_output.append(tabular_output_dict)
                        continue

                    resource_index = 0
                    for p in range(0, num_profiles):
                        for r in range(0, num_resource_profiles):
                            resource_type = partition_config_dict['profiles'][p]['resources'][r]['resource_type']
                            resource_instances = partition_config_dict['profiles'][p]['resources'][r]['partition_resource']
                            resources_shared = partition_config_dict['profiles'][p]['resources'][r]['num_partitions_share_resource']
                            tabular_output_dict = {
                                   "resource_index": resource_index,
                                   "resource_type": resource_type,
                                   "resource_instances": resource_instances,
                                   "resources_shared": resources_shared}
                            resource_index += 1
                            tabular_output.append(tabular_output_dict)
                except amdsmi_exception.AmdSmiLibraryException as e:
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
                legend_parts = [
                    "\n\nLegend:",
                    "  * = Current mode"]
                legend_output = "\n".join(legend_parts)
                if self.logger.destination == 'stdout':
                    print(legend_output)
                else:
                    with self.logger.destination.open('a', encoding="utf-8") as output_file:
                        output_file.write(legend_output + '\n')


    def ras(self, args, multiple_devices=False, gpu=None, cper=None, afid=None,
            severity=None, folder=None, file_limit=None, cper_file=None, follow=None):
        """
        Retrieve and process CPER (RAS) entries for a target GPU.

        Expected command (all options only):
        amd-smi ras --cper --severity=nonfatal-uncorrected,fatal --folder <folder_name> --file-limit=1000 --follow

        Since no timestamp is provided on the command line, the function starts from a default cursor of 0.
        The output file name is auto-generated using the timestamp from the CPER header data (converted from
        the header’s "YYYY/MM/DD HH:MM:SS" format), along with the GPU/platform ID and error severity.
        """

        # GPU handle logic.
        if gpu:
            args.gpu = gpu
        if cper:
            args.cper = cper
        if afid:
            args.afid = afid
        if severity:
            args.severity = severity
        if folder:
            args.folder = folder
        if file_limit:
            args.file_limit = file_limit
        if cper_file:
            args.cper_file = cper_file
        if follow:
            args.follow = follow
        if args.gpu == None:
            args.gpu = self.device_handles

        if args.afid:
            if args.cper_file:
                afids = self.helpers.pvtDumpAfids(args.cper_file)
                print(' '.join(map(str, afids)))
                return
            else:
                command = " ".join(sys.argv[1:])
                message = f"Command '{command}' requires '--cper-file'. Run '--help' for more info."
                raise AmdSmiInvalidCommandException(command,
                                                    self.logger.format,
                                                    message)

        if not self.group_check_printed:
            self.helpers.check_required_groups()
            self.group_check_printed = True

        if not args.cper:
            return

        if not args.gpu:
            return

        if not isinstance(args.gpu, list):
            args.gpu = [args.gpu]

        args.cursor = [0] * len(args.gpu)

        # Using all the devices given in args.gpu
        # Populate a list of all the primary partition GPU ids (GPU 0, GPU 1, etc)
        partition_warning_flag = True
        primary_partition_gpu_ids = set() # set of all primary partition GPU ids from arg.gpu
        for device_handle in args.gpu:
            # First get the partition
            partition_id = self.helpers.get_partition_id(device_handle)
            # If there is a single primary partition within args.gpu then we don't need to print the warning
            if partition_id == 0:
                partition_warning_flag = False
                break
            # Then attempt to get the primary GPU id for that partition
            primary_partition_gpu_id = self.helpers.get_primary_partition_gpu_id(device_handle)
            # Add to the set if it's a non-primary partition and we found a valid primary GPU id
            if partition_id != 0 and primary_partition_gpu_id is not None:
                primary_partition_gpu_ids.add(primary_partition_gpu_id)

        if partition_warning_flag:
            # Create a list of the primary partitions
            primary_partitions_str = " ".join(f"GPU{gpu_id}" for gpu_id in primary_partition_gpu_ids)

            print("WARNING: CPER files are only available on primary partitions")
            if len(primary_partition_gpu_ids) > 1:
                print(f"Try with primary partitions {primary_partitions_str}",end="")
            else:
                print(f"Try with primary partition {primary_partitions_str}",end="")

            print()

        while True:
            for idx, device_handle in enumerate(args.gpu):
                self.helpers.ras_cper(args, device_handle, self.logger, idx)
            if not args.follow:
                break
            time.sleep(1)


    def default(self, args):
        """Display the default amdsmi view when no args are given."""

        # check groups first
        if not self.group_check_printed:
            self.helpers.check_required_groups()
            self.group_check_printed = True

        processors = amdsmi_interface.amdsmi_get_processor_handles()
        version_info = {"amd-smi": "N/A",
                        "amdgpu version": "N/A",
                        "fw pldm version": "N/A",
                        "vbios version": "N/A",
                        "rocm version": (False, "N/A")}
        version_info['rocm version'] = amdsmi_interface.amdsmi_get_rocm_version()
        try:
            version_info["amdgpu version"] = amdsmi_interface.amdsmi_get_gpu_driver_info(processors[0])
        except amdsmi_exception.AmdSmiLibraryException as e:
            version_info["amdgpu version"] = "N/A"
            logging.debug("Failed to get driver info for gpu: %s", e.get_error_info())
        try:
            fw_info = amdsmi_interface.amdsmi_get_fw_info(processors[0])
            for fw in fw_info['fw_list']:
                if "pldm" in fw.keys():
                    version_info['fw pldm version'] = fw['pldm']
                    # we only need to find one of them
                    break
        except amdsmi_exception.AmdSmiLibraryException as e:
            version_info['fw pldm version'] = "N/A"
            logging.debug("Failed to get fw pldm info for gpu: %s", e.get_error_info())
        try:
            version_info['vbios version'] = amdsmi_interface.amdsmi_get_gpu_vbios_info(processors[0])["version"]
            if version_info['vbios version'] == "":
                version_info['vbios version'] = "N/A"
        except amdsmi_exception.AmdSmiLibraryException as e:
            version_info['vbios version'] = "N/A"
            logging.debug("Failed to get vbios info for gpu: %s", e.get_error_info())

        version_info["amd-smi"] = f'{__version__}'

        default_table_info_dict = {}
        default_table_info_dict.update({"version_info": version_info})

        gpu_info_list = []
        all_process_list = []
        all_process_list = []

        # get info for each processor to display in default output
        for processor in processors:
            gpu_info_dict = {}

            gpu_id = self.helpers.get_gpu_id_from_device_handle(processor)
            gpu_info_dict.update({"gpu_id": gpu_id})
            # get common gpu_metrics first
            try:
                gpu_metrics = amdsmi_interface.amdsmi_get_gpu_metrics_info(processor)
            except amdsmi_exception.AmdSmiLibraryException as e:
                gpu_metrics = amdsmi_interface._NA_amdsmi_get_gpu_metrics_info()

            # partition info
            try:
                current_mem = amdsmi_interface.amdsmi_get_gpu_memory_partition(processor)
            except amdsmi_exception.AmdSmiLibraryException as e:
                current_mem = "N/A"
            try:
                current_comp = amdsmi_interface.amdsmi_get_gpu_compute_partition(processor)
            except amdsmi_exception.AmdSmiLibraryException as e:
                current_comp = "N/A"
            if current_comp == "N/A" or current_mem == "N/A":
                partition_mode = "N/A"
            else:
                partition_mode = f"{current_comp}/{current_mem}"
            gpu_info_dict.update({"partition_mode": partition_mode})

            # GPU name market name and OAM ID
            try:
                asic_info = amdsmi_interface.amdsmi_get_gpu_asic_info(processor)
                market_name = asic_info['market_name']
                oam_id = asic_info['oam_id']
                # get num_cu now for use later
                total_num_cu = float(asic_info['num_compute_units'])
            except amdsmi_exception.AmdSmiLibraryException as e:
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
            except amdsmi_exception.AmdSmiLibraryException as e:
                bdf = "N/A"
            gpu_info_dict.update({"bdf": bdf})

            # HIP ID
            try:
                enum_info = amdsmi_interface.amdsmi_get_gpu_enumeration_info(processor)
                hip_id = enum_info['hip_id']
            except amdsmi_exception.AmdSmiLibraryException as e:
                hip_id = "N/A"
            gpu_info_dict.update({"hip_id": hip_id})

            # mem utilization, GPU utilization, power usage, and temperature from gpu_metrics
            if gpu_metrics != "N/A":
                mem_util = gpu_metrics['average_umc_activity']
                gfx_util = gpu_metrics['average_gfx_activity']
                if gpu_metrics['current_socket_power'] != "N/A":
                    current_power = gpu_metrics['current_socket_power']
                else:
                    current_power = gpu_metrics['average_socket_power']
                temperature = gpu_metrics['temperature_hotspot']
            else:
                mem_util = "N/A"
                gfx_util = "N/A"
                current_power = "N/A"
                temperature = "N/A"
            gpu_info_dict.update({"mem_util": mem_util})
            gpu_info_dict.update({"gfx_util": gfx_util})
            gpu_info_dict.update({"temp": temperature})


            # rest of power usage info
            try:
                power_cap_info = amdsmi_interface.amdsmi_get_power_cap_info(processor)
                socket_power_limit = self.helpers.convert_SI_unit(power_cap_info['power_cap'], AMDSMIHelpers.SI_Unit.MICRO)
                power_usage = {"current_power": current_power, "power_limit": socket_power_limit}
            except amdsmi_exception.AmdSmiLibraryException as e:
                power_usage = "N/A"
            gpu_info_dict.update({"power_usage": power_usage})

            # memory usage
            try:
                total_vram = amdsmi_interface.amdsmi_get_gpu_memory_total(processor, amdsmi_interface.AmdSmiMemoryType.VRAM) // (1024*1024)
                used_vram = amdsmi_interface.amdsmi_get_gpu_memory_usage(processor, amdsmi_interface.AmdSmiMemoryType.VRAM) // (1024*1024)
                mem_usage = {"used_vram": used_vram, "total_vram": total_vram}
            except amdsmi_exception.AmdSmiLibraryException as e:
                mem_usage = "N/A"
            gpu_info_dict.update({"mem_usage": mem_usage})

            # uncorrectable ECC errors
            try:
                ecc_count = amdsmi_interface.amdsmi_get_gpu_total_ecc_count(processor)
                uncorrectable = ecc_count.pop('uncorrectable_count')
            except amdsmi_exception.AmdSmiLibraryException as e:
                uncorrectable = "N/A"
            gpu_info_dict.update({"uncorr_ecc": uncorrectable})

            # Fan usage
            try:
                fan_speed = amdsmi_interface.amdsmi_get_gpu_fan_speed(processor, 0)
            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug("Failed to get fan speed for gpu %s | %s", processor, e.get_error_info())
                fan_speed = "N/A"
            try:
                fan_max = amdsmi_interface.amdsmi_get_gpu_fan_speed_max(processor, 0)
                fan_usage = "N/A"
                if fan_max > 0 and fan_speed != "N/A":
                    fan_usage = round((float(fan_speed) / float(fan_max)) * 100, 2)
            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug("Failed to get max fan speed for gpu %s | %s", processor, e.get_error_info())
                fan_usage = "N/A"
            gpu_info_dict.update({"fan": fan_usage})

            gpu_info_list.append(gpu_info_dict)

            # Running Processes
            try:
                raw_process_list = amdsmi_interface.amdsmi_get_gpu_process_list(processor)
                for proc in raw_process_list:
                    proc_info_dict = {"gpu": "N/A", "pid": "N/A", "name": "N/A","gtt": "N/A", "vram": "N/A", "mem_usage": "N/A", "cu_occupancy": "N/A"}
                    proc_info_dict['gpu'] = gpu_id
                    proc_info_dict['pid'] = proc['pid']
                    proc_info_dict['name'] = proc['name']
                    proc_info_dict['gtt'] = self.helpers.convert_bytes_to_readable(proc['memory_usage']['gtt_mem'])
                    proc_info_dict['vram'] = self.helpers.convert_bytes_to_readable(proc['memory_usage']['vram_mem'])
                    proc_info_dict['mem_usage'] = self.helpers.convert_bytes_to_readable(proc['mem'])
                    # Handle cu_occupancy conversion safely
                    try:
                        if proc['cu_occupancy'] != "N/A" and total_num_cu != "N/A":
                            num_cu = float(proc['cu_occupancy'])
                            proc_info_dict['cu_occupancy'] = {"current_cu": num_cu, "total_num_cu": total_num_cu}
                        else:
                            proc_info_dict['cu_occupancy'] = {"current_cu": "N/A", "total_num_cu": total_num_cu}
                    except (ValueError, TypeError):
                        proc_info_dict['cu_occupancy'] = {"current_cu": "N/A", "total_num_cu": total_num_cu}

                    all_process_list.append(proc_info_dict)
            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug("Failed to get process list for gpu %s | %s", gpu_id, e.get_error_info())

        default_table_info_dict.update({f"gpu_info_list": gpu_info_list})
        default_table_info_dict.update({"processes": all_process_list})

        if self.logger.is_json_format():
            self.logger.output = default_table_info_dict
            self.logger.print_output()
        elif self.logger.is_csv_format():
            self.logger.multiple_device_output = default_table_info_dict
            self.logger.print_output(multiple_device_enabled=True, tabular=True, dynamic=True)
        else:
            self.logger.print_default_output(default_table_info_dict)


    def _event_thread(self, commands, i):
        devices = commands.device_handles
        if len(devices) == 0:
            print("No GPUs on machine")
            return

        device = devices[i]
        listener = amdsmi_interface.AmdSmiEventReader(device,
                                        amdsmi_interface.AmdSmiEvtNotificationType)
        values_dict = {}

        while not self.stop:
            try:
                events = listener.read(2000)
                for event in events:
                    values_dict["event"] = event["event"]
                    # parse message as it's own dictionary
                    message_list = event["message"].split("  ")
                    message_dict = {}
                    for item in message_list:
                        if not item == "":
                            item_list = item.split(": ")
                            message_dict.update({item_list[0]: item_list[1]})
                    values_dict["message"] = message_dict
                    commands.logger.store_output(event['processor_handle'], 'values', values_dict)
                    commands.logger.print_output()
            except amdsmi_exception.AmdSmiLibraryException as e:
                if e.err_code != amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_DATA:
                    print(e)
            except Exception as e:
                print(e)

        listener.stop()
