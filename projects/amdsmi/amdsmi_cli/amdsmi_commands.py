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
import logging
import sys

from amdsmi_helpers import AMDSMIHelpers
from amdsmi_logger import AMDSMILogger
from amdsmi import amdsmi_exception, amdsmi_interface

from subcommands import (
    BadPagesCommands,
    DefaultCommands,
    EventCommands,
    FabricCommands,
    FirmwareCommands,
    ListDevicesCommands,
    MetricCommands,
    MonitorCommands,
    NodeCommands,
    PartitionCommands,
    ProcessCommands,
    RasCommands,
    ResetCommands,
    SetValueCommands,
    StaticCommands,
    TopologyCommands,
    VersionCommands,
    XgmiCommands,
)


class AMDSMICommands(
    BadPagesCommands,
    DefaultCommands,
    EventCommands,
    FabricCommands,
    FirmwareCommands,
    ListDevicesCommands,
    MetricCommands,
    MonitorCommands,
    NodeCommands,
    PartitionCommands,
    ProcessCommands,
    RasCommands,
    ResetCommands,
    SetValueCommands,
    StaticCommands,
    TopologyCommands,
    VersionCommands,
    XgmiCommands,
):
    """This class contains all the commands corresponding to AMDSMIParser.
    Each command function will interact with AMDSMILogger to handle
    displaying the output to the specified format and destination.

    Subcommand implementations are organized in the subcommands/ package.
    Each subcommand module defines a base class (e.g., ListDevicesCommands) that
    provides the methods for that CLI subcommand. This class inherits from
    all of them and adds __init__, profile, and rocm_smi.
    """

    def __init__(self, format="human_readable", destination="stdout", helpers=None) -> None:
        if helpers is None:
            # If helpers is not provided, create a new instance
            self.helpers = AMDSMIHelpers()
        else:
            self.helpers = helpers
        self.logger = AMDSMILogger(format=format, destination=destination, helpers=self.helpers)
        self.device_handles = []
        self.device_handles_gpus = []
        self.device_handles_brcm_nics = []
        self.device_handles_ainics = []
        self.device_handles_switchs = []
        self.cpu_handles = []
        self.core_handles = []
        self.node_handle = None
        self.stop = ""
        self.group_check_printed = False

        amdsmi_init_flag = self.helpers.get_amdsmi_init_flag()
        logging.debug(f"AMDSMI Init Flag: {amdsmi_init_flag}")
        exit_flag = False

        if self.helpers.is_amdgpu_initialized():
            try:
                self.device_handles = amdsmi_interface.amdsmi_get_processor_handles()
                self.device_handles_gpus = self.helpers.get_gpu_handles()
            except amdsmi_exception.AmdSmiLibraryException as e:
                if e.err_code in (
                    amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NOT_INIT,
                    amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_DRIVER_NOT_LOADED,
                ):
                    logging.error(
                        "Unable to get devices, driver not initialized (amdgpu not found in modules)"
                    )
                else:
                    raise e

            if len(self.device_handles) == 0:
                # No GPU's found post amdgpu driver initialization
                logging.error(
                    "Unable to detect any GPU devices, check amdgpu version and module status (sudo modprobe amdgpu)"
                )
                exit_flag = True

        if (
            self.helpers.is_ainic_initialized()
            or self.helpers.is_brcm_nic_initialized()
            or self.helpers.is_brcm_switch_initialized()
        ):
            try:
                self.device_handles_brcm_nics = self.helpers.get_nic_handles()
                self.device_handles_ainics = self.helpers.get_ainic_handles()
                if len(self.device_handles_gpus) == 0:
                    self.device_handles_gpus = self.helpers.get_gpu_handles()
                self.device_handles_switchs = self.helpers.get_switch_handles()
            except amdsmi_exception.AmdSmiLibraryException as e:
                if e.err_code in (
                    amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NOT_INIT,
                    amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_DRIVER_NOT_LOADED,
                ):
                    logging.error(
                        "Unable to get devices, driver not initialized (BRCMNIC not found in modules)"
                    )
                else:
                    raise e

        # Resolve the node handle (independent of AINIC init; needed for amd-smi node).
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
                ret = amdsmi_interface.amdsmi_get_cpu_handles()
                self.cpu_handles = ret["processor_handles"]
            except amdsmi_exception.AmdSmiLibraryException as e:
                if e.err_code in (
                    amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NOT_INIT,
                    amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_DRV,
                ):
                    logging.info(
                        "Unable to detect any CPU devices, check amd_hsmp (or) hsmp_acpi version and module status (sudo modprobe amd_hsmp (or) sudo modprobe hsmp_acpi)"
                    )
                else:
                    raise e

            # core handles
            try:
                self.core_handles = amdsmi_interface.amdsmi_get_cpucore_handles()
            except amdsmi_exception.AmdSmiLibraryException as e:
                if e.err_code in (
                    amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NOT_INIT,
                    amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_DRV,
                ):
                    logging.info(
                        "Unable to get CORE devices, amd_hsmp driver not loaded (sudo modprobe amd_hsmp)"
                    )
                else:
                    raise e

            if len(self.cpu_handles) == 0 and len(self.core_handles) == 0:
                # No CPU's found post amd_hsmp driver initialization
                logging.error(
                    "Unable to detect any CPU devices, check amd_hsmp (or) hsmp_acpi version and module status (sudo modprobe amd_hsmp (or) sudo modprobe hsmp_acpi)"
                )
                exit_flag = True

        if exit_flag:
            version_args = argparse.Namespace()
            version_args.gpu_version = False
            version_args.cpu_version = False
            version_args.nic_version = False
            self.version(version_args)
            sys.exit(-1)

    def profile(self, args):
        """Not applicable to linux baremetal"""
        print("Not applicable to linux baremetal")

    def rocm_smi(self, args):
        """
        Display GPU information in ROCm-SMI compatible format (showAllConcise).
        This provides a drop-in replacement for rocm-smi --showallconcise using amdsmi backend.

        Args:
            args: Parsed arguments (unused for this command)
        """
        try:
            # Import the ROCm-SMI compatible functions from the compatibility module
            import sys
            import os

            # Add the current directory to path if needed
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)

            import amdsmi_rocm_smi_compat

            showAllConcise = amdsmi_rocm_smi_compat.showAllConcise
            listDevices = amdsmi_rocm_smi_compat.listDevices
            initializeRsmi = amdsmi_rocm_smi_compat.initializeRsmi
            check_runtime_status = amdsmi_rocm_smi_compat.check_runtime_status

            # Initialize AMD SMI
            if not initializeRsmi():
                logging.error("Failed to initialize AMD SMI")
                return

            try:
                # Get processor handles
                deviceList = listDevices()

                if not deviceList:
                    logging.error("No AMD GPU devices found")
                    return

                # Check runtime status (low power state warning)
                if not check_runtime_status():
                    print(
                        "\nWARNING: AMD GPU device(s) is/are in a low-power state. Check power control/runtime_status\n"
                    )

                # Display ROCm-SMI compatible output
                showAllConcise(deviceList)

            finally:
                # Shutdown AMD SMI
                try:
                    amdsmi_interface.amdsmi_shut_down()
                except:
                    pass

        except ImportError as e:
            logging.error(f"Could not import ROCm-SMI compatibility module: {e}")
            logging.error("Make sure amdsmi_rocm_smi_compat.py is in the amdsmi_cli directory")
            print("ERROR: ROCm-SMI compatibility mode not available")
        except Exception as e:
            logging.error(f"Error in ROCm-SMI compatibility mode: {e}")
            print(f"ERROR: {e}")
