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

from _version import __version__

from amdsmi import amdsmi_exception, amdsmi_interface


class VersionCommands:
    def version(self, args, gpu_version=None, cpu_version=None, nic_version=None):
        """Print Version String

        Args:
            args (Namespace): Namespace containing the parsed CLI args
        """

        if gpu_version:
            args.gpu_version = gpu_version
        if cpu_version:
            args.cpu_version = cpu_version
        if nic_version:
            args.nic_version = nic_version
        # if no args are given, display everything available on this build
        if args.gpu_version is None and args.cpu_version is None and args.nic_version is None:
            args.gpu_version = True
            args.cpu_version = self.helpers.is_amd_hsmp_initialized()
            args.nic_version = True

        if not self.group_check_printed:
            self.helpers.check_required_groups()
            self.group_check_printed = True

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
            logging.debug("Failed to get ROCm version | %s", e.get_error_info())
            rocm_version_str = "N/A"

        self.logger.output["tool"] = "AMDSMI Tool"
        self.logger.output["version"] = f"{__version__}"
        self.logger.output["amdsmi_library_version"] = f"{amdsmi_lib_version_str}"
        self.logger.output["rocm_version"] = f"{rocm_version_str}"

        if args.gpu_version:
            try:
                gpus = amdsmi_interface.amdsmi_get_processor_handles()
                if isinstance(gpus, list) and len(gpus) > 0:
                    gpu_version_info = amdsmi_interface.amdsmi_get_gpu_driver_info(gpus[0])
                    gpu_version_str = gpu_version_info["driver_version"]
                else:
                    gpu_version_str = "N/A"
            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug("Failed to get amdgpu version | %s", e.get_error_info())
                gpu_version_str = "N/A"

            self.logger.output["amdgpu_version"] = gpu_version_str
        if args.cpu_version:
            try:
                ret = amdsmi_interface.amdsmi_get_cpu_handles()
                cpus = ret["processor_handles"]
                if isinstance(cpus, list) and len(cpus) > 0:
                    cpu_version_info = amdsmi_interface.amdsmi_get_cpu_hsmp_driver_version(cpus[0])
                    cpu_version_str = (
                        str(cpu_version_info["hsmp_driver_major_ver_num"])
                        + "."
                        + str(cpu_version_info["hsmp_driver_minor_ver_num"])
                    )
                else:
                    cpu_version_str = "N/A"
            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug("Failed to get CPU version | %s", e.get_error_info())
                cpu_version_str = "N/A"
            self.logger.output["amd_hsmp_driver_version"] = cpu_version_str

        nic_version_str = "N/A"
        if args.nic_version:
            try:
                ainic_device_handles = amdsmi_interface.get_ainic_handles()
                for nic_id, device_handle in enumerate(ainic_device_handles):
                    nic_info = amdsmi_interface.amdsmi_get_ainic_info(device_handle, True)
                    if nic_version_str != "":
                        nic_version_str += ", "
                    nic_version_str += (
                        nic_info["DRIVER"]["NAME"] + "." + nic_info["DRIVER"]["VERSION"]
                    )
            except amdsmi_exception.AmdSmiLibraryException as e:
                nic_version_str = e.get_error_info()
            self.logger.output["nic_driver_version"] = nic_version_str

        if self.logger.is_human_readable_format():
            human_readable_output = (
                f"AMDSMI Tool: {__version__} | "
                f"AMDSMI Library version: {amdsmi_lib_version_str} | "
                f"ROCm version: {rocm_version_str}"
            )
            if args.gpu_version:
                human_readable_output = (
                    human_readable_output + f" | amdgpu version: {gpu_version_str}"
                )
            if args.cpu_version:
                human_readable_output = (
                    human_readable_output + f" | hsmp version: {cpu_version_str}"
                )
            if args.nic_version:
                human_readable_output = (
                    human_readable_output + f" | ionic version: {nic_version_str}"
                )
            # Custom human readable handling for version
            if self.logger.destination == "stdout":
                print(human_readable_output)
            else:
                with self.logger.destination.open("a", encoding="utf-8") as output_file:
                    output_file.write(human_readable_output + "\n")
        elif self.logger.is_json_format() or self.logger.is_csv_format():
            self.logger.print_output()
