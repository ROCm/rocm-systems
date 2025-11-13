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
import signal
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
        self.node_handle = None
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

            # Resolve the node handle.
            for dev in self.device_handles:
                try:
                    nh = amdsmi_interface.amdsmi_get_node_handle(dev)
                    if nh is not None:
                        self.node_handle = nh
                        # Only need one handle, break after first success
                        break
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug("Unable to get node handle: %s", e.get_error_info())
                    # Node handle functionality is optional, so don't raise an error

        if self.helpers.is_amd_hsmp_initialized():
            try:
                self.cpu_handles = amdsmi_interface.amdsmi_get_cpusocket_handles()
            except amdsmi_exception.AmdSmiLibraryException as e:
                if e.err_code in (amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NOT_INIT,
                                amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_DRV):
                    logging.info('Unable to detect any CPU devices, check amd_hsmp (or) hsmp_acpi version and module status (sudo modprobe amd_hsmp (or) sudo modprobe hsmp_acpi)')
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
                logging.error('Unable to detect any CPU devices, check amd_hsmp (or) hsmp_acpi version and module status (sudo modprobe amd_hsmp (or) sudo modprobe hsmp_acpi)')
                exit_flag = True

        self.convert_clock_type = {
            "sys": amdsmi_interface.AmdSmiClkType.SYS,
            "mem": amdsmi_interface.AmdSmiClkType.MEM,
            "df": amdsmi_interface.AmdSmiClkType.DF,
            "soc": amdsmi_interface.AmdSmiClkType.SOC,
            "dcef": amdsmi_interface.AmdSmiClkType.DCEF,
            # vclk and dclk currently do not support levels so average clk is given for frequency levels
            "vclk0": amdsmi_interface.AmdSmiClkType.VCLK0,
            "vclk1": amdsmi_interface.AmdSmiClkType.VCLK1,
            "dclk0": amdsmi_interface.AmdSmiClkType.DCLK0,
            "dclk1": amdsmi_interface.AmdSmiClkType.DCLK1
        }

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
                human_readable_output = human_readable_output + f" | hsmp version: {cpu_version_str}"
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

        if not self.group_check_printed:
            self.helpers.check_required_groups(check_render=True, check_video=False)
            self.group_check_printed = True

        # Handle multiple GPUs
        handled_multiple_gpus, device_handle = self.helpers.handle_gpus(args, self.logger, self.list)
        if handled_multiple_gpus:
            return # This function is recursive

        args.gpu = device_handle

        # Get gpu_id for logging
        gpu_id = self.helpers.get_gpu_id_from_device_handle(args.gpu)

        # Always try to get BDF regardless of group check
        try:
            bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(args.gpu)
        except amdsmi_exception.AmdSmiLibraryException as e:
            bdf = "N/A"

        try:
            uuid = amdsmi_interface.amdsmi_get_gpu_device_uuid(args.gpu)
        except amdsmi_exception.AmdSmiLibraryException:
            uuid = "N/A"

        try:
            kfd_info = amdsmi_interface.amdsmi_get_gpu_kfd_info(args.gpu)
            kfd_id = kfd_info['kfd_id']
            node_id = kfd_info['node_id']
            partition_id = kfd_info['current_partition_id']
        except amdsmi_exception.AmdSmiLibraryException as e:
            kfd_id = node_id = partition_id = "N/A"
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
                        soc_pstate=None, xgmi_plpd=None, process_isolation=None, clock=None, profile=None):
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
            self.helpers.check_required_groups(check_render=True, check_video=False)
            self.group_check_printed = True

        if self.helpers.is_linux() and self.helpers.is_baremetal():
            if limit:
                args.limit = limit
            if soc_pstate:
                args.soc_pstate = soc_pstate
            if xgmi_plpd:
                args.xgmi_plpd = xgmi_plpd
            if profile:
                args.profile = profile
            current_platform_args += ["ras", "limit", "soc_pstate", "xgmi_plpd", "profile"]
            current_platform_values += [args.ras, args.limit, args.soc_pstate, args.xgmi_plpd, args.profile]

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
                'pcie_levels': "N/A",
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

            try:
                pcie_info = amdsmi_interface.amdsmi_get_gpu_pci_bandwidth(args.gpu)
                num_supported = pcie_info['transfer_rate']['num_supported']
                if num_supported != 0:
                    bus_info['pcie_levels'] = {}
                    for level in range(0, num_supported):
                        speed = str(self.helpers.convert_SI_unit(float(pcie_info['transfer_rate']['frequency'][level]), AMDSMIHelpers.SI_Unit.NANO)) + " GT/s"
                        width = str(pcie_info['lanes'][level])
                        level_values = (speed, width)
                        bus_info['pcie_levels'].update({str(level): level_values})
                else:
                    bus_info['pcie_levels'] = "N/A"
            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug("Failed to get pci bandwidth info for gpu %s | %s", gpu_id, e.get_error_info())

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

                power_limit_types = {}
                for power_type in amdsmi_interface.AmdSmiPowerCapType:
                    # Strip 'AMDSMI_POWER_CAP_TYPE_' prefix and convert to lowercase
                    key = power_type.name.replace('AMDSMI_POWER_CAP_TYPE_', '').lower()
                    power_limit_types[key] = {
                        "max_power_limit" : "N/A",
                        "min_power_limit" : "N/A",
                        "socket_power_limit" : "N/A"
                    }

                try:
                    power_cap_types = amdsmi_interface.amdsmi_get_supported_power_cap(args.gpu)
                    for sensor in power_cap_types['sensor_inds']:
                        power_cap_info = amdsmi_interface.amdsmi_get_power_cap_info(args.gpu, sensor)
                        max_power_limit = power_cap_info['max_power_cap']
                        max_power_limit = self.helpers.convert_SI_unit(max_power_limit, AMDSMIHelpers.SI_Unit.MICRO)
                        min_power_limit = power_cap_info['min_power_cap']
                        min_power_limit = self.helpers.convert_SI_unit(min_power_limit, AMDSMIHelpers.SI_Unit.MICRO)
                        socket_power_limit = power_cap_info['power_cap']
                        socket_power_limit = self.helpers.convert_SI_unit(socket_power_limit, AMDSMIHelpers.SI_Unit.MICRO)
                        ppt = {
                            "max_power_limit" : self.helpers.unit_format(self.logger, max_power_limit, 'W'),
                            "min_power_limit" : self.helpers.unit_format(self.logger, min_power_limit, 'W'),
                            "socket_power_limit" : self.helpers.unit_format(self.logger, socket_power_limit, 'W')
                        }

                        sensor_name = power_cap_types['sensor_types'][sensor]
                        # Strip 'AMDSMI_POWER_CAP_TYPE_' prefix and convert to lowercase
                        sensor_key = sensor_name.name.replace('AMDSMI_POWER_CAP_TYPE_', '').lower()
                        power_limit_types[sensor_key] = ppt
                except amdsmi_exception.AmdSmiLibraryException as e:
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

                # PTL
                try:
                    ptl_state = amdsmi_interface.amdsmi_get_gpu_ptl_state(args.gpu)
                    ptl_state = "Enabled" if ptl_state else "Disabled"
                except amdsmi_exception.AmdSmiLibraryException as e:
                    ptl_state = "N/A"
                    logging.debug("Failed to get PTL state for gpu %s | %s", gpu_id, e.get_error_info())

                try:
                    ptl_format1, ptl_format2 = amdsmi_interface.amdsmi_get_gpu_ptl_formats(args.gpu)
                    fmt1_name = amdsmi_interface.amdsmi_wrapper.amdsmi_ptl_data_format_t__enumvalues.get(ptl_format1)
                    fmt2_name = amdsmi_interface.amdsmi_wrapper.amdsmi_ptl_data_format_t__enumvalues.get(ptl_format2)

                    fmt1_short = fmt1_name.replace("AMDSMI_PTL_DATA_FORMAT_", "") if fmt1_name else "UNKNOWN"
                    fmt2_short = fmt2_name.replace("AMDSMI_PTL_DATA_FORMAT_", "") if fmt2_name else "UNKNOWN"

                    ptl_format = f"{fmt1_short},{fmt2_short}"
                except amdsmi_exception.AmdSmiLibraryException as e:
                    ptl_format = "N/A"
                    logging.debug("Failed to get PTL state for gpu %s | %s", gpu_id, e.get_error_info())

                # Assign units
                power_unit = 'W'
                temp_unit_human_readable = '\N{DEGREE SIGN}C'
                temp_unit_json = 'C'

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
                limit_info['ppt0'] = power_limit_types['ppt0']
                limit_info['ppt1'] = power_limit_types['ppt1']

                # Shutdown limits
                limit_info['slowdown_edge_temperature'] = slowdown_temp_edge_limit
                limit_info['slowdown_hotspot_temperature'] = slowdown_temp_hotspot_limit
                limit_info['slowdown_vram_temperature'] = slowdown_temp_vram_limit
                limit_info['shutdown_edge_temperature'] = shutdown_temp_edge_limit
                limit_info['shutdown_hotspot_temperature'] = shutdown_temp_hotspot_limit
                limit_info['shutdown_vram_temperature'] = shutdown_temp_vram_limit

                # PTL
                limit_info['ptl_state'] = ptl_state
                limit_info['ptl_format'] = ptl_format

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

                # Format for CSV output - flatten completely to avoid extra columns
                if self.logger.is_csv_format() and isinstance(policy_info, dict):
                    policies_str = ', '.join(
                        f"{p['policy_id']}:{p['policy_description']}"
                        for p in policy_info.get('policies', [])
                    ) or 'N/A'
                    
                    static_dict['num_supported'] = policy_info.get('num_supported', 'N/A')
                    static_dict['current_id'] = policy_info.get('current_id', 'N/A')
                    static_dict['policies'] = policies_str
                else:
                    static_dict['soc_pstate'] = policy_info
        if 'xgmi_plpd' in current_platform_args:
            if args.xgmi_plpd:
                try:
                    policy_info = amdsmi_interface.amdsmi_get_xgmi_plpd(args.gpu)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    policy_info = "N/A"
                    logging.debug("Failed to get xgmi_plpd info for gpu %s | %s", gpu_id, e.get_error_info())

                # Format for CSV output - flatten completely to avoid extra columns
                if self.logger.is_csv_format() and isinstance(policy_info, dict):
                    policies_str = ', '.join(
                        f"{p['policy_id']}:{p['policy_description']}"
                        for p in policy_info.get('policies', [])
                    ) or 'N/A'
                    
                    static_dict['num_supported'] = policy_info.get('num_supported', 'N/A')
                    static_dict['current_id'] = policy_info.get('current_id', 'N/A')
                    static_dict['policies'] = policies_str
                else:
                    static_dict['xgmi_plpd'] = policy_info
                static_dict['xgmi_plpd'] = policy_info

        if 'profile' in current_platform_args:
            if args.profile:
                try:
                    profile_status = amdsmi_interface.amdsmi_get_gpu_power_profile_presets(args.gpu, 0)
                    
                    # Parse available profiles from bitfield
                    available_profiles = self.helpers.parse_available_profiles(
                        profile_status['available_profiles']
                    )
                    
                    # Get current profile name
                    current_profile = self.helpers.get_profile_name_from_mask(
                        profile_status['current']
                    )
                    
                    # Store output
                    static_dict['profile'] = {
                        'available_profiles': available_profiles,
                        'current': current_profile,
                        'num_profiles': profile_status['num_profiles']
                    }
                except amdsmi_exception.AmdSmiLibraryException as e:
                    static_dict['profile'] = e.get_error_info()
                    logging.debug("Failed to get power profile info for gpu %s | %s", gpu_id, e.get_error_info())
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
                    socket_set = amdsmi_interface.amdsmi_get_cpu_affinity_with_scope(args.gpu, amdsmi_interface.AmdSmiAffinityScope.SOCKET_SCOPE)
                    socket_set = [f"{cpus:016X}" for cpus in socket_set]
                    socket_set = {f'cpu_list_{i}': f"{cpus}" for i, cpus in enumerate(socket_set)}
                    socket_bitmask_ranges = self.helpers.get_bitmask_ranges(socket_set)
                    socket_affinity = {}
                    for key in socket_set:
                        socket_affinity[key] = {
                            "bitmask": socket_set[key],
                            "cpu_cores_affinity": socket_bitmask_ranges.get(key, "N/A")
                        }
                except amdsmi_exception.AmdSmiLibraryException as e:
                    socket_affinity = "N/A"
                    logging.debug("Failed to get socket affinity for gpu %s | %s", gpu_id, e.get_error_info())

                static_dict['numa'] = { 'node' : numa_node_number,
                                        'affinity' : numa_affinity,
                                        'cpu_affinity' : cpu_affinity,
                                        'socket_affinity' : socket_affinity}
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
                    if clk in self.convert_clock_type:
                        clk_type_conversion = self.convert_clock_type[clk]
                    else:
                        clk_type_conversion = "N/A"
                        output_format = self.helpers.get_output_format()
                        raise AmdSmiInvalidParameterException('static', clk_type, output_format) # clk type given is bad

                    try:
                        frequencies = amdsmi_interface.amdsmi_get_clk_freq(args.gpu, clk_type_conversion)
                        # some clocks may have a sysfs file but no frequencies for whatever reason.
                        if len(frequencies['frequency']) == 0:
                            freq_dict = "N/A"
                            continue
                        freq_dict = {}
                        current_level = frequencies['current']
                        freq_dict.update({'current_level':current_level})
                        current_frequency = str(self.helpers.convert_SI_unit(frequencies['frequency'][current_level], AMDSMIHelpers.SI_Unit.MICRO)) + "MHz"
                        freq_dict.update({'current_frequency':current_frequency})
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
                        logging.debug("Failed to get clock info for gpu %s | %s", gpu_id, e.get_error_info())
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
            # For NUMA data - flatten CPU affinity lists
            if 'numa' in static_dict and isinstance(static_dict['numa'], dict):
                numa_data = static_dict.pop('numa')
                multiple_devices_csv_override = True

                # Get data
                node = numa_data.get('node', 'N/A')
                affinity = numa_data.get('affinity', 'N/A')
                cpu_affinity = numa_data.get('cpu_affinity', {})
                socket_affinity = numa_data.get('socket_affinity', {})
                # Create a flattened row for list entry
                row_dict = static_dict.copy()

                if cpu_affinity and isinstance(cpu_affinity, dict):
                    for cpu_list_key in cpu_affinity.keys():
                        cpu_entry = cpu_affinity[cpu_list_key]
                        socket_entry = socket_affinity.get(cpu_list_key, {"bitmask": "N/A", "cpu_cores_affinity": "N/A"})
                        row_dict.update({
                            'node': node,
                            'affinity': affinity,
                            'cpu_list': cpu_list_key,
                            'bitmask': cpu_entry.get('bitmask'),
                            'cpu_cores_affinity': cpu_entry.get('cpu_cores_affinity'),
                            'socket_bitmask': socket_entry.get('bitmask'),
                            'socket_cpu_cores_affinity': socket_entry.get('cpu_cores_affinity')
                        })
                        self.logger.store_output(args.gpu, 'values', row_dict)
                        self.logger.store_gpu_json_output.append(row_dict)
                        self.logger.store_multiple_device_output()
                else:
                    row_dict.update({
                        'node': node,
                        'affinity': affinity,
                        'cpu_list': 'N/A',
                        'bitmask': 'N/A',
                        'cpu_cores_affinity': 'N/A',
                        'socket_bitmask': 'N/A',
                        'socket_cpu_cores_affinity': 'N/A'
                    })
                    self.logger.store_output(args.gpu, 'values', row_dict)
                    self.logger.store_gpu_json_output.append(row_dict)
            # expand if ras blocks are populated
            elif self.helpers.is_linux() and self.helpers.is_baremetal() and args.ras:
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
                clock=None, profile=None):
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
                          "process_isolation", "clock", "profile"]
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
                                    dfc_ucode, fb_info, num_vf, soc_pstate, xgmi_plpd,
                                    process_isolation, clock, profile)
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
                                process_isolation, clock, profile)
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


    def bad_pages(self, args, multiple_devices=False, gpu=None, retired=None, pending=None, un_res=None, hex_format=None):
        """ Get bad pages information for target gpu

        Args:
            args (Namespace): Namespace containing the parsed CLI args
            multiple_devices (bool, optional): True if checking for multiple devices. Defaults to False.
            gpu (device_handle, optional): device_handle for target device. Defaults to None.
            retired (bool, optional) - Value override for args.retired
            pending (bool, optional) - Value override for args.pending/
            un_res (bool, optional) - Value override for args.un_res
            hex_format (bool, optional) - Value override for args.hex

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
        if hex_format is not None:
            args.hex = hex_format

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
                        # Format page address and size based on --hex flag
                        if args.hex:
                            bad_page_info_entry["page_address"] = f"0x{bad_page['page_address']:x}"
                            bad_page_info_entry["page_size"] = f"0x{bad_page['page_size']:x}"
                        else:
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
                        # Format page address and size based on --hex flag
                        if args.hex:
                            bad_page_info_entry["page_address"] = f"0x{bad_page['page_address']:x}"
                            bad_page_info_entry["page_size"] = f"0x{bad_page['page_size']:x}"
                        else:
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
                        # Format page address and size based on --hex flag
                        if hasattr(args, 'hex') and args.hex:
                            bad_page_info_entry["page_address"] = f"0x{bad_page['page_address']:x}"
                            bad_page_info_entry["page_size"] = f"0x{bad_page['page_size']:x}"
                        else:
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
            self.helpers.check_required_groups(check_render=True, check_video=False)
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

        is_partition_metrics = False  # True if we get the metrics from xcp_metrics file (amdsmi_get_gpu_partition_metrics_info)
        #get metric info only once per gpu, this will speed up data output
        try:
            # Get GPU Metrics table
            gpu_metric = amdsmi_interface.amdsmi_get_gpu_metrics_info(args.gpu)
        except amdsmi_exception.AmdSmiLibraryException as e:
            logging.debug("#3 - Unable to load GPU Metrics table for %s | %s", gpu_id, e.get_error_info())
            gpu_metric = amdsmi_interface._NA_amdsmi_get_gpu_metrics_info()

        # Workaround for XCP (partition) metrics not providing num_partition in v1.9+/v1.1+
        # Provides original formatting for earlier metric versions
        partition_metric_info = self.helpers._get_metric_version_and_partition_info(gpu_metric, is_partition_metrics, gpu_id, args.gpu)
        num_partition = partition_metric_info['num_partition']

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
                        elif 'power' in key:
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
                gpu_board_temp_dict = self.helpers.get_gpu_board_temperatures(args.gpu, gpu_id, self.logger)
                # if every value is N/A, then we don't want to display the values unless explicitly told to
                # all args_list being True indicates that this gpu_board is not explicitly called itself
                args_list = [getattr(args, arg) for arg in current_platform_args]
                if all(value == "N/A" for value in gpu_board_temp_dict.values()) and all(arg == True for arg in args_list):
                    gpu_board_temp_dict = {}
                if gpu_board_temp_dict:
                    values_dict['gpu_board'] = {'temperature':gpu_board_temp_dict}
        if "base_board" in current_platform_args:
            if args.base_board:
                base_board_temp_dict = self.helpers.get_base_board_temperatures(args.gpu, gpu_id, self.logger)
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
                        logging.debug(f"point_{point} frequency: {od_volt['curve.vc_points'][point]['frequency']}")
                        logging.debug(f"point_{point} voltage:   {od_volt['curve.vc_points'][point]['voltage']}")
                        frequency = int(od_volt["curve.vc_points"][point]['frequency'] / 1000000)
                        voltage = int(od_volt["curve.vc_points"][point]['voltage'])
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
                    throttle_status['gfx_clk_below_host_limit_power_accumulated'] = self.helpers.build_xcp_dict('acc_gfx_clk_below_host_limit_pwr', violation_status, num_partition)
                    throttle_status['gfx_clk_below_host_limit_thermal_accumulated'] = self.helpers.build_xcp_dict('acc_gfx_clk_below_host_limit_thm', violation_status, num_partition)
                    throttle_status['total_gfx_clk_below_host_limit_accumulated'] = self.helpers.build_xcp_dict('acc_gfx_clk_below_host_limit_total', violation_status, num_partition)
                    throttle_status['low_utilization_accumulated'] = self.helpers.build_xcp_dict('acc_low_utilization', violation_status, num_partition)
                    throttle_status['prochot_violation_status'] = self.helpers.build_xcp_dict('active_prochot_thrm', violation_status, num_partition)
                    throttle_status['ppt_violation_status'] = self.helpers.build_xcp_dict('active_ppt_pwr', violation_status, num_partition)
                    throttle_status['socket_thermal_violation_status'] = self.helpers.build_xcp_dict('active_socket_thrm', violation_status, num_partition)
                    throttle_status['vr_thermal_violation_status'] = self.helpers.build_xcp_dict('active_vr_thrm', violation_status, num_partition)
                    throttle_status['hbm_thermal_violation_status'] = self.helpers.build_xcp_dict('active_hbm_thrm', violation_status, num_partition)
                    throttle_status['gfx_clk_below_host_limit_violation_status'] = self.helpers.build_xcp_dict('active_gfx_clk_below_host_limit', violation_status, num_partition) # deprecated
                    throttle_status['gfx_clk_below_host_limit_power_violation_status'] = self.helpers.build_xcp_dict('active_gfx_clk_below_host_limit_pwr', violation_status, num_partition)
                    throttle_status['gfx_clk_below_host_limit_thermal_violation_status'] = self.helpers.build_xcp_dict('active_gfx_clk_below_host_limit_thm', violation_status, num_partition)
                    throttle_status['total_gfx_clk_below_host_limit_violation_status'] = self.helpers.build_xcp_dict('active_gfx_clk_below_host_limit_total', violation_status, num_partition)
                    throttle_status['low_utilization_violation_status'] = self.helpers.build_xcp_dict('active_low_utilization', violation_status, num_partition)
                    throttle_status['prochot_violation_activity'] = violation_status['per_prochot_thrm']
                    throttle_status['ppt_violation_activity'] = violation_status['per_ppt_pwr']
                    throttle_status['socket_thermal_violation_activity'] = violation_status['per_socket_thrm']
                    throttle_status['vr_thermal_violation_activity'] = violation_status['per_vr_thrm']
                    throttle_status['hbm_thermal_violation_activity'] = violation_status['per_hbm_thrm']
                    throttle_status['gfx_clk_below_host_limit_violation_activity'] = violation_status['per_gfx_clk_below_host_limit'] # deprecated
                    throttle_status['gfx_clk_below_host_limit_power_violation_activity'] = self.helpers.build_xcp_dict('per_gfx_clk_below_host_limit_pwr', violation_status, num_partition)
                    throttle_status['gfx_clk_below_host_limit_thermal_violation_activity'] = self.helpers.build_xcp_dict('per_gfx_clk_below_host_limit_thm', violation_status, num_partition)
                    throttle_status['total_gfx_clk_below_host_limit_violation_activity'] = self.helpers.build_xcp_dict('per_gfx_clk_below_host_limit_total', violation_status, num_partition)
                    throttle_status['low_utilization_violation_activity'] = self.helpers.build_xcp_dict('per_low_utilization', violation_status, num_partition)

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
                            throttle_status[key] = self.helpers.unit_format(self.logger, value, activity_unit)
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
                   cpu_dimm_pow_consumption=None, cpu_dimm_thermal_sensor=None,
                   cpu_dfcstate_ctrl=None, cpu_railisofreq_policy=None):
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
            cpu_dfcstate_ctrl (bool, optional): Value override for args.cpu_dfcstate_ctrl. Defaults to None
            cpu_railisofreq_policy (bool, optional): Value override for args.cpu_railisofreq_policy. Defaults to None

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
        if cpu_dfcstate_ctrl:
            args.cpu_dfcstate_ctrl = cpu_dfcstate_ctrl
        if cpu_railisofreq_policy:
            args.cpu_railisofreq_policy = cpu_railisofreq_policy

        #store cpu args that are applicable to the current platform
        curr_platform_cpu_args = ["cpu_power_metrics", "cpu_prochot", "cpu_freq_metrics",
                                  "cpu_c0_res", "cpu_lclk_dpm_level", "cpu_pwr_svi_telemetry_rails",
                                  "cpu_io_bandwidth", "cpu_xgmi_bandwidth", "cpu_metrics_ver",
                                  "cpu_metrics_table", "cpu_socket_energy", "cpu_ddr_bandwidth",
                                  "cpu_temp", "cpu_dimm_temp_range_rate", "cpu_dimm_pow_consumption",
                                  "cpu_dimm_thermal_sensor", "cpu_dfcstate_ctrl", "cpu_railisofreq_policy"]
        curr_platform_cpu_values = [args.cpu_power_metrics, args.cpu_prochot, args.cpu_freq_metrics,
                                    args.cpu_c0_res, args.cpu_lclk_dpm_level, args.cpu_pwr_svi_telemetry_rails,
                                    args.cpu_io_bandwidth, args.cpu_xgmi_bandwidth, args.cpu_metrics_ver,
                                    args.cpu_metrics_table, args.cpu_socket_energy, args.cpu_ddr_bandwidth,
                                    args.cpu_temp, args.cpu_dimm_temp_range_rate, args.cpu_dimm_pow_consumption,
                                    args.cpu_dimm_thermal_sensor, args.cpu_dfcstate_ctrl, args.cpu_railisofreq_policy]

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
                                                                                    args.cpu_io_bandwidth[0][1].upper())
                static_dict["io_bandwidth"]["band_width"] = bandwidth
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["io_bandwidth"]["band_width"] = "N/A"
                logging.debug("Failed to get io bandwidth for cpu %s | %s", cpu_id, e.get_error_info())
        if args.cpu_xgmi_bandwidth:
            static_dict["xgmi_bandwidth"] = {}
            try:
                bandwidth = amdsmi_interface.amdsmi_get_cpu_current_xgmi_bw(args.cpu,
                                                                            int(args.cpu_xgmi_bandwidth[0][0]),
                                                                            args.cpu_xgmi_bandwidth[0][1].upper())
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
        if args.cpu_dfcstate_ctrl:
            static_dict["dfcstate"] = {}
            try:
                dfcstatectrl_status = amdsmi_interface.amdsmi_get_dfc_ctrl(args.cpu)
                static_dict["dfcstate"]["dfcstatectrl_status"] = dfcstatectrl_status
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["dfcstate"]["dfcstatectrl_status"] = "N/A"
                logging.debug("Failed to get dfcstate control status for cpu %s | %s", cpu_id, e.get_error_info())
        if args.cpu_railisofreq_policy:
            static_dict["cpurailiso"] = {}
            try:
                cpurailisofreq_policy = amdsmi_interface.amdsmi_get_cpu_rail_isofreq_policy(args.cpu)
                static_dict["cpurailiso"]["cpurailisofreq_policy"] = cpurailisofreq_policy
            except amdsmi_exception.AmdSmiLibraryException as e:
                static_dict["cpurailiso"]["cpurailisofreq_policy"] = "N/A"
                logging.debug("Failed to get cpurailiso frequency policy for cpu %s | %s", cpu_id, e.get_error_info())

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
                cpu_dimm_thermal_sensor=None, cpu_dfcstate_ctrl=None, cpu_railisofreq_policy=None,
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
            cpu_dfcstate_ctrl (bool, optional): Value override for args.cpu_dfcstate_ctrl. Defaults to None
            cpu_railisofreq_policy (bool, optional): Value override for args.cpu_railisofreq_policy. Defaults to None

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
                          "cpu_dimm_pow_consumption", "cpu_dimm_thermal_sensor",
                          "cpu_dfcstate_ctrl", "cpu_railisofreq_policy"]
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
                                cpu_dimm_pow_consumption, cpu_dimm_thermal_sensor,
                                cpu_dfcstate_ctrl, cpu_railisofreq_policy)
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
                                cpu_dimm_pow_consumption, cpu_dimm_thermal_sensor,
                                cpu_dfcstate_ctrl, cpu_railisofreq_policy)
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
                "cu_occupancy": process_info["cu_occupancy"],
                "evicted_time": process_info["evicted_time"]
            }

            engine_usage_unit = "ns"
            memory_usage_unit = "B"
            evicted_time_unit = "ms"

            if self.logger.is_human_readable_format():
                process_info['mem_usage'] = self.helpers.convert_bytes_to_readable(process_info['mem_usage'])
                for usage_metric in process_info['memory_usage']:
                    process_info["memory_usage"][usage_metric] = self.helpers.convert_bytes_to_readable(process_info["memory_usage"][usage_metric])
                memory_usage_unit = ""

            process_info['mem_usage'] = self.helpers.unit_format(self.logger,
                                                                 process_info['mem_usage'],
                                                                 memory_usage_unit)

            process_info['evicted_time'] = self.helpers.unit_format(self.logger,
                                                                 process_info['evicted_time'],
                                                                 evicted_time_unit)

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

        previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
        system_exit_exc = None
        signal.signal(signal.SIGTERM, self._event_sigterm_handler)
        try:
            while True:
                try:
                    user_input = input()
                except EOFError:
                    self.stop = True
                    break
                except KeyboardInterrupt:
                    self.stop = True
                    break

                if self.stop:
                    break

                if user_input == 'q':
                    print("Escape Sequence Detected; Exiting")
                    self.stop = True
                    break
        except SystemExit as exc:
            system_exit_exc = exc
        finally:
            self.stop = True
            for thread in threads:
                thread.join()
            signal.signal(signal.SIGTERM, previous_sigterm_handler)

        if system_exit_exc is not None:
            raise system_exit_exc


    def _event_sigterm_handler(self, signum, frame):
        self.stop = True
        raise SystemExit(128 + signum)


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
            self.helpers.check_required_groups(check_render=True, check_video=False)
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
            #         "bdf": BDF identific