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

import grp
import json
import logging
import math
import multiprocessing
import os
import platform
import re
import sys
import time
import glob
import errno
import pwd
import stat
from typing import Tuple, Optional, Union
import tempfile

from enum import Enum
from pathlib import Path
from typing import List, Set, Union
from functools import lru_cache

# Import amdsmi library
from amdsmi_init import *
from BDF import BDF


class AMDSMIHelpers():
    """Helper functions that aren't apart of the AMDSMI API
    Useful for determining platform and device identifiers

    Functions:
        os_info: tuple ()
    """

    def __init__(self) -> None:
        self.operating_system = platform.system()

        self._is_hypervisor = False
        self._is_virtual_os = False
        self._is_baremetal = False
        self._is_passthrough = False

        self._is_linux = False
        self._is_windows = False

        # Counts and Tracking variables
        self._count_of_sets_called = 0
        self._count_of_cper_files = 0
        self._previous_set_success_check = amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_UNKNOWN_ERROR


        # Check if the system is a virtual OS
        if self.operating_system.startswith("Linux"):
            self._is_linux = True
            logging.debug(f"AMDSMIHelpers: Platform is linux:{self._is_linux}")

            try:
                with open('/proc/cpuinfo', 'r') as f:
                    if 'hypervisor' in f.read():
                        self._is_virtual_os = True
            except IOError:
                pass

            self._is_baremetal = not self._is_virtual_os

        if self._is_virtual_os:
            #If hard coded passthrough device ids exist on Virtual OS,
            #   then it is a passthrough system
            output = self.get_pci_device_ids()
            passthrough_device_ids = ["7460", "73c8", "74a0", "74a1", "74a2"]
            if any(('0x' + device_id) in output for device_id in passthrough_device_ids):
                self._is_baremetal = True
                self._is_virtual_os = False
                self._is_passthrough = True

            # Check for passthrough system dynamically via drm querying id_flags
            try:
                if self.is_amdgpu_initialized() and not self._is_passthrough:
                    device_handles = amdsmi_interface.amdsmi_get_processor_handles()
                    for dev in device_handles:
                        virtualization_info = amdsmi_interface.amdsmi_get_gpu_virtualization_mode(dev)
                        if virtualization_info['mode'] == amdsmi_interface.AmdSmiVirtualizationMode.PASSTHROUGH:
                            self._is_baremetal = True
                            self._is_virtual_os = False
                            self._is_passthrough = True
                            break # Once passthrough is determined, we can immediately break
            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug("Unable to determine virtualization status: " + str(e.get_error_code()))


    def increment_set_count(self):
        self._count_of_sets_called += 1


    def get_set_count(self):
        return self._count_of_sets_called

    def assign_previous_set_success_check(self, status):
        """Assigns the previous set success check to the status provided.
        This is used to determine if the last set was successful or not.
        """
        self._previous_set_success_check = status

    def get_previous_set_success_check(self):
        """Returns the previous set success check.
        This is used to determine if the last set was successful or not.
        """
        return self._previous_set_success_check

    def increment_cper_count(self):
        self._count_of_cper_files += 1


    def get_cper_count(self):
        return self._count_of_cper_files


    def is_virtual_os(self):
        return self._is_virtual_os


    def is_hypervisor(self):
        # Returns True if hypervisor is enabled on the system
        return self._is_hypervisor


    def is_baremetal(self):
        # Returns True if system is baremetal, if system is hypervisor this should return False
        return self._is_baremetal


    def is_passthrough(self):
        return self._is_passthrough


    def is_linux(self):
        return self._is_linux


    def is_windows(self):
        return self._is_windows


    def os_info(self, string_format=True):
        """Return operating_system and type information ex. (Linux, Baremetal)
        params:
            string_format (bool) True to return in string format, False to return Tuple
        returns:
            str or (str, str)
        """
        operating_system = ""
        if self.is_linux():
            operating_system = "Linux"
        elif self.is_windows():
            operating_system = "Windows"
        else:
            operating_system = "Unknown"

        operating_system_type = ""
        if self.is_baremetal():
            operating_system_type = "Baremetal"
        elif self.is_virtual_os():
            operating_system_type = "Guest"
        elif self.is_hypervisor():
            operating_system_type = "Hypervisor"
        else:
            operating_system_type = "Unknown"

        # Passthrough Override
        if self.is_passthrough():
            operating_system_type = "Guest (Passthrough)"

        if string_format:
            return f"{operating_system} {operating_system_type}"

        return (operating_system, operating_system_type)


    def get_amdsmi_init_flag(self):
        return AMDSMI_INIT_FLAG


    def is_amdgpu_initialized(self):
        return AMDSMI_INIT_FLAG & amdsmi_interface.amdsmi_wrapper.AMDSMI_INIT_AMD_GPUS


    def is_amd_hsmp_initialized(self):
        return AMDSMI_INIT_FLAG & amdsmi_interface.amdsmi_wrapper.AMDSMI_INIT_AMD_CPUS


    def is_ainic_initialized(self):
        return AMDSMI_INIT_FLAG & amdsmi_interface.amdsmi_wrapper.AMDSMI_INIT_AMD_NICS

    def is_brcm_nic_initialized(self):
        return False

    def is_brcm_switch_initialized(self):
        return False

    def get_rocm_version(self):
        try:
            rocm_lib_status, rocm_version = amdsmi_interface.amdsmi_get_rocm_version()
            if rocm_lib_status is not True:
                return "N/A"
            return rocm_version
        except amdsmi_interface.AmdSmiLibraryException as e:
            return "N/A"


    def get_cpu_choices(self):
        """Return dictionary of possible CPU choices and string of the output:
            Dictionary will be in format: cpus[ID]: Device Handle)
            String output will be in format:
                "ID: 0 "
        params:
            None
        return:
            (dict, str) : (cpu_choices, cpu_choices_str)
        """
        cpu_choices = {}
        cpu_choices_str = ""

        try:
            cpu_handles = []
            # amdsmi_get_cpusocket_handles() returns the cpu socket handles stored for cpu_id
            cpu_handles = amdsmi_interface.amdsmi_get_cpusocket_handles()
        except amdsmi_interface.AmdSmiLibraryException as e:
            if e.err_code in (amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NOT_INIT,
                              amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_DRIVER_NOT_LOADED):
                logging.info('Unable to get device choices, driver not initialized (amd_hsmp  or hsmp_acpi not found in modules)')
            else:
                raise e
        if len(cpu_handles) == 0:
            logging.info('Unable to find any devices, check if driver is initialized (amd_hsmp or hsmp_acpi not found in modules)')
        else:
            # Handle spacing for the gpu_choices_str
            max_padding = int(math.log10(len(cpu_handles))) + 1

            for cpu_id, device_handle in enumerate(cpu_handles):
                cpu_choices[str(cpu_id)] = {
                        "Device Handle": device_handle
                }
                if cpu_id == 0:
                    id_padding = max_padding
                else:
                    id_padding = max_padding - int(math.log10(cpu_id))
                cpu_choices_str += f"ID: {cpu_id}\n"

            # Add the all option to the gpu_choices
            cpu_choices["all"] = "all"
            cpu_choices_str += f"  all{' ' * max_padding}| Selects all devices\n"

        return (cpu_choices, cpu_choices_str)


    def get_core_choices(self):
        """Return dictionary of possible Core choices and string of the output:
            Dictionary will be in format: coress[ID]: Device Handle)
            String output will be in format:
                "ID: 0 "
        params:
            None
        return:
            (dict, str) : (core_choices, core_choices_str)
        """
        core_choices = {}
        core_choices_str = ""

        try:
            core_handles = []
            # amdsmi_get_cpucore_handles() returns the core handles stored for core_id
            core_handles = amdsmi_interface.amdsmi_get_cpucore_handles()
        except amdsmi_interface.AmdSmiLibraryException as e:
            if e.err_code in (amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NOT_INIT,
                              amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_DRIVER_NOT_LOADED):
                logging.info('Unable to get device choices, driver not initialized (amd_hsmp  or hsmp_acpi not found in modules)')
            else:
                raise e
        if len(core_handles) == 0:
            logging.info('Unable to find any devices, check if driver is initialized (amd_hsmp or hsmp_acpi  not found in modules)')
        else:
            # Handle spacing for the gpu_choices_str
            max_padding = int(math.log10(len(core_handles))) + 1

            for core_id, device_handle in enumerate(core_handles):
                core_choices[str(core_id)] = {
                        "Device Handle": device_handle
                }
                if core_id == 0:
                    id_padding = max_padding
                else:
                    id_padding = max_padding - int(math.log10(core_id))
            core_choices_str += f"ID: 0 - {len(core_handles) - 1}\n"

            # Add the all option to the core_choices
            core_choices["all"] = "all"
            core_choices_str += f"  all{' ' * max_padding}| Selects all devices\n"

        return (core_choices, core_choices_str)


    def get_output_format(self):
        """Returns the output format read from sys.argv
        Returns:
            str: outputformat
        """
        args = sys.argv[1:]
        outputformat = "human"
        if "--json" in args or "--j" in args:
            outputformat = "json"
        elif "--csv" in args or "--c" in args:
            outputformat = "csv"
        return outputformat


    def get_gpu_choices(self):
        """Return dictionary of possible GPU choices and string of the output:
            Dictionary will be in format: gpus[ID] : (BDF, UUID, Device Handle)
            String output will be in format:
                "ID: 0 | BDF: 0000:23:00.0 | UUID: ffffffff-0000-1000-0000-000000000000"
        params:
            None
        return:
            (dict, str) : (gpu_choices, gpu_choices_str)
        """
        gpu_choices = {}
        gpu_choices_str = ""
        device_handles = []

        try:
            # amdsmi_get_processor_handles returns the device_handles storted for gpu_id
            device_handles = amdsmi_interface.amdsmi_get_processor_handles()
        except amdsmi_interface.AmdSmiLibraryException as e:
            if e.err_code in (amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NOT_INIT,
                              amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_DRIVER_NOT_LOADED):
                logging.info('Unable to get device choices, driver not initialized (amdgpu not found in modules)')
            else:
                raise e

        if len(device_handles) == 0:
            logging.info('Unable to find any devices, check if driver is initialized (amdgpu not found in modules)')
        else:
            # Handle spacing for the gpu_choices_str
            max_padding = int(math.log10(len(device_handles))) + 1

            for gpu_id, device_handle in enumerate(device_handles):
                bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(device_handle)
                uuid = amdsmi_interface.amdsmi_get_gpu_device_uuid(device_handle)
                gpu_choices[str(gpu_id)] = {
                    "bdf": bdf,
                    "UUID": uuid,
                    "Device Handle": device_handle,
                }

                if gpu_id == 0:
                    id_padding = max_padding
                else:
                    id_padding = max_padding - int(math.log10(gpu_id))
                gpu_choices_str += f"ID: {gpu_id}{' ' * id_padding}| BDF: {bdf} | UUID: {uuid}\n"

            # Add the all option to the gpu_choices
            gpu_choices["all"] = "all"
            gpu_choices_str += f"  all{' ' * max_padding}| Selects all devices\n"

        return (gpu_choices, gpu_choices_str)

    def nic_choices_from_nic_info(self, nic_info, nic_id, device_handle, max_padding, nic_choices, nic_choices_str):
        bdf = nic_info['bdf']
    
        #uuid="abc"
        uuid = nic_info['UUID']
    
        nic_choices[str(nic_id)] = {
            "bdf": bdf,
            "UUID": uuid,
            "Device Handle": device_handle,
        }

        if nic_id == 0:
            id_padding = max_padding
        else:
            id_padding = max_padding - int(math.log10(nic_id))
        nic_choices_str += f"ID: {nic_id}{' ' * id_padding}| BDF: {bdf} | UUID: {uuid}\n"
        return nic_choices, nic_choices_str
    
    def get_nic_choices(self):
        nic_choices = {}
        nic_choices_str = ""
        nic_device_handles = []
        ainic_device_handles = []

        try:
            # get_nic_handles returns the device_handles storted for nic_id
            nic_device_handles = amdsmi_interface.get_nic_handles()
            ainic_device_handles = amdsmi_interface.get_ainic_handles()
         
        except amdsmi_interface.AmdSmiLibraryException as e:
            if e.err_code in (amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NOT_INIT,
                              amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_DRIVER_NOT_LOADED):
                logging.info('Unable to get device choices, driver not initialized (BRCM_NIC, IONIC_NIC, RDMA_NIC not found in modules)')
            else:
                raise e

        if len(nic_device_handles) == 0 and len(ainic_device_handles) == 0:
            logging.info('Unable to find any devices, check if driver is initialized (BRCM_NIC, IONIC_NIC, RDMA_NIC not found in modules)')
        else:
            # Handle spacing for the gpu_choices_str
            max_padding = int(math.log10(len(nic_device_handles) + len(ainic_device_handles))) + 1
          
            for nic_id, device_handle in enumerate(nic_device_handles):
                nic_info = amdsmi_interface.amdsmi_get_nic_info(device_handle)
                if nic_info:
                    nic_choices, nic_choices_str = self.nic_choices_from_nic_info(nic_info, nic_id, device_handle, max_padding, nic_choices, nic_choices_str)
           
            for nic_id, device_handle in enumerate(ainic_device_handles):
                nic_info = amdsmi_interface.amdsmi_get_ainic_info(device_handle)
                nic_id = nic_id + len(nic_device_handles)
                nic_choices, nic_choices_str = self.nic_choices_from_nic_info(nic_info, nic_id, device_handle, max_padding, nic_choices, nic_choices_str)

            # Add the all option to the gpu_choices
            nic_choices["all"] = "all"
            nic_choices_str += f"  all{' ' * max_padding}| Selects all devices\n"

        return (nic_choices, nic_choices_str)

     #BRCM POC to get switch choices 
    def get_switch_choices(self):
        switch_choices = {}
        switch_choices_str = ""
        device_handles = []

        try:
            # get_switch_handles returns the device_handles storted for switch_id
            device_handles = amdsmi_interface.get_switch_handles()
     
        except amdsmi_interface.AmdSmiLibraryException as e:
      
            if e.err_code in (amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NOT_INIT,
                              amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_DRIVER_NOT_LOADED):
                logging.info('Unable to get device choices, driver not initialized (BRCM_switch not found in modules)')
          
            else:
                raise e

        if len(device_handles) == 0:
            logging.info('Unable to find any devices, check if driver is initialized (BRCM_switch not found in modules)')
        else:
            # Handle spacing for the gpu_choices_str
            max_padding = int(math.log10(len(device_handles))) + 1
      
            for switch_id, device_handle in enumerate(device_handles):
                bdf = amdsmi_interface.amdsmi_get_switch_device_bdf(device_handle)
          
                #uuid="abc"
                uuid = amdsmi_interface.amdsmi_get_switch_device_uuid(device_handle)
           
                switch_choices[str(switch_id)] = {
                    "bdf": bdf,
                    "UUID": uuid,
                    "Device Handle": device_handle,
                }

                if switch_id == 0:
                    id_padding = max_padding
                else:
                    id_padding = max_padding - int(math.log10(switch_id))
                switch_choices_str += f"ID: {switch_id}{' ' * id_padding}| BDF: {bdf} | UUID: {uuid}\n"

       
            # Add the all option to the gpu_choices
            switch_choices["all"] = "all"
            switch_choices_str += f"  all{' ' * max_padding}| Selects all devices\n"

        return (switch_choices, switch_choices_str)

    @staticmethod
    def is_UUID(uuid_question: str) -> bool:
        """Determine if given string is of valid UUID format
        Args:
            uuid_question (str): the given string to be evaluated.
        Returns:
            True or False: wether the UUID given matches the UUID format.
        """
        UUID_pattern = re.compile("^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$", flags=re.IGNORECASE)
        if re.match(UUID_pattern, uuid_question) is None:
            return False
        return True


    def get_device_handles_from_gpu_selections(self, gpu_selections: List[str], gpu_choices=None) -> tuple:
        """Convert provided gpu_selections to device_handles

        Args:
            gpu_selections (list[str]): Selected GPU ID(s), BDF(s), or UUID(s):
                    ex: ID:0  | BDF:0000:23:00.0 | UUID:ffffffff-0000-1000-0000-000000000000
            gpu_choices (dict{gpu_choices}): This is a dictionary of the possible gpu_choices
        Returns:
            (True, True, list[device_handles]): Returns a list of all the gpu_selections converted to
                amdsmi device_handles
            (False, valid_gpu_format, str): Return False, whether the format of the GPU input is valid, and the first input that failed to be converted
        """
        if 'all' in gpu_selections:
            return True, True, amdsmi_interface.amdsmi_get_processor_handles()

        if isinstance(gpu_selections, str):
            gpu_selections = [gpu_selections]

        if gpu_choices is None:
            # obtains dictionary of possible gpu choices
            gpu_choices = self.get_gpu_choices()[0]

        selected_device_handles = []
        for gpu_selection in gpu_selections:
            valid_gpu_choice = False

            for gpu_id, gpu_info in gpu_choices.items():
                bdf = gpu_info['bdf']
                is_bdf = True
                uuid = gpu_info['UUID']
                device_handle = gpu_info['Device Handle']

                # Check if passed gpu is a gpu ID or UUID
                if gpu_selection == gpu_id or gpu_selection.lower() == uuid:
                    selected_device_handles.append(device_handle)
                    valid_gpu_choice = True
                    break
                else:  # Check if gpu passed is a BDF object
                    try:
                        if BDF(gpu_selection) == BDF(bdf):
                            selected_device_handles.append(device_handle)
                            valid_gpu_choice = True
                            break
                    except Exception:
                        is_bdf = False
                        pass

            if not valid_gpu_choice:
                logging.debug(f"AMDSMIHelpers.get_device_handles_from_gpu_selections - Unable to convert {gpu_selection}")
                valid_gpu_format = True
                if not self.is_UUID(gpu_selection) and not gpu_selection.isdigit() and not is_bdf:
                    valid_gpu_format = False
                return False, valid_gpu_format, gpu_selection
        return True, True, selected_device_handles


    def get_device_handles_from_nic_selections(self, nic_selections: List[str], nic_choices=None):
    
        """Convert provided nic_selections to device_handles

        Args:
            nic_selections (list[str]): Selected NIC ID(s), BDF(s), or UUID(s):
                    ex: ID:0  | BDF:0000:23:00.0 | UUID:ffffffff-0000-1000-0000-000000000000
            nic_choices (dict{nic_choices}): This is a dictionary of the possible gpu_choices
        Returns:
            (True, list[device_handles]): Returns a list of all the nic_selections converted to
                amdsmi device_handles
            (False, str): Return False, and the first input that failed to be converted
        """
        if 'all' in nic_selections:
            return (True, amdsmi_interface.get_nic_handles() + amdsmi_interface.get_ainic_handles())

        if isinstance(nic_selections, str):
            nic_selections = [nic_selections]
    
        if nic_choices is None:
            nic_choices = self.get_nic_choices()[0]

        selected_device_handles = []
        for nic_selection in nic_selections:
            valid_nic_choice = False

            for nic_id, nic_info in nic_choices.items():
                bdf = nic_info['bdf']
                uuid = nic_info['UUID']
                device_handle = nic_info['Device Handle']
            

                # Check if passed nic is a nic ID or UUID
                if nic_selection == nic_id or nic_selection.lower() == uuid:
                
                    device_type=amdsmi_interface.amdsmi_get_processor_type(device_handle)
                
                    selected_device_handles.append(device_handle)
                    valid_nic_choice = True
                    break
                else:  # Check if nic passed is a BDF object
                    if BDF(nic_selection) == BDF(bdf):
                        selected_device_handles.append(device_handle)
                        valid_nic_choice = True
                        break

            if not valid_nic_choice:
                logging.debug(f"AMDSMIHelpers.get_device_handles_from_gpu_selections - Unable to convert {nic_selection}")
            
                return False, nic_selection
            
        
        return True, selected_device_handles

    #BRCM POC to get device handles from switch selections
    def get_device_handles_from_switch_selections(self, switch_selections: List[str], switch_choices=None):

            """Convert provided switch_selections to device_handles

            Args:
                switch_selections (list[str]): Selected switch ID(s), BDF(s), or UUID(s):
                        ex: ID:0  | BDF:0000:23:00.0 | UUID:ffffffff-0000-1000-0000-000000000000
                switch_choices (dict{switch_choices}): This is a dictionary of the possible gpu_choices
            Returns:
                (True, list[device_handles]): Returns a list of all the switch_selections converted to
                    amdsmi device_handles
                (False, str): Return False, and the first input that failed to be converted
            """
            if 'all' in switch_selections:
                return (True, amdsmi_interface.get_switch_handles())

            if isinstance(switch_selections, str):
                switch_selections = [switch_selections]

            if switch_choices is None:
                switch_choices = self.get_switch_choices()[0]

            selected_device_handles = []
            for switch_selection in switch_selections:
                valid_switch_choice = False

                for switch_id, switch_info in switch_choices.items():
                    bdf = switch_info['bdf']
                    uuid = switch_info['UUID']
                    device_handle = switch_info['Device Handle']
        

                    # Check if passed switch is a switch ID or UUID
                    if switch_selection == switch_id or switch_selection.lower() == uuid:
            
                        device_type=amdsmi_interface.amdsmi_get_processor_type(device_handle)
            
                        selected_device_handles.append(device_handle)
                        valid_switch_choice = True
                        break
                    else:  # Check if switch passed is a BDF object
                        try:
                            if BDF(switch_selection) == BDF(bdf):
                                selected_device_handles.append(device_handle)
                                valid_switch_choice = True
                                break
                        except Exception:
                            # Ignore exception when checking if the gpu_choice is a BDF
                            pass

                if not valid_switch_choice:
                    logging.debug(f"AMDSMIHelpers.get_device_handles_from_gpu_selections - Unable to convert {switch_selection}")
        
                    return False, switch_selection
                
    
            return True, selected_device_handles    
    

    def get_device_handles_from_cpu_selections(self, cpu_selections: List[str], cpu_choices=None):
        """Convert provided cpu_selections to device_handles

        Args:
            cpu_selections (list[str]): Selected CPU ID(s):
                    ex: ID:0
            cpu_choices (dict{cpu_choices}): This is a dictionary of the possible cpu_choices
        Returns:
            (True, list[device_handles]): Returns a list of all the cpu_selections converted to
                amdsmi device_handles
            (False, str): Return False, and the first input that failed to be converted
        """
        if 'all' in cpu_selections:
            return True, True, amdsmi_interface.amdsmi_get_cpusocket_handles()

        if isinstance(cpu_selections, str):
            cpu_selections = [cpu_selections]

        if cpu_choices is None:
            cpu_choices = self.get_cpu_choices()[0]

        selected_device_handles = []
        for cpu_selection in cpu_selections:
            valid_cpu_choice = False
            for cpu_id, cpu_info in cpu_choices.items():
                device_handle = cpu_info['Device Handle']

                # Check if passed gpu is a gpu ID
                if cpu_selection == cpu_id:
                    selected_device_handles.append(device_handle)
                    valid_cpu_choice = True
                    break
            if not valid_cpu_choice:
                logging.debug(f"AMDSMIHelpers.get_device_handles_from_cpu_selections - Unable to convert {cpu_selection}")
                valid_cpu_format = True
                if not cpu_selection.isdigit():
                    valid_cpu_format = False
                return False, valid_cpu_format, cpu_selection
        return True, True, selected_device_handles


    def get_device_handles_from_core_selections(self, core_selections: List[str], core_choices=None):
        """Convert provided core_selections to device_handles

        Args:
            core_selections (list[str]): Selected CORE ID(s):
                    ex: ID:0
            core_choices (dict{core_choices}): This is a dictionary of the possible core_choices
        Returns:
            (True, list[device_handles]): Returns a list of all the core_selections converted to
                amdsmi device_handles
            (False, str): Return False, and the first input that failed to be converted
        """
        if 'all' in core_selections:
            return True, True, amdsmi_interface.amdsmi_get_cpucore_handles()

        if isinstance(core_selections, str):
            core_selections = [core_selections]

        if core_choices is None:
            core_choices = self.get_core_choices()[0]

        selected_device_handles = []
        for core_selection in core_selections:
            valid_core_choice = False
            for core_id, core_info in core_choices.items():
                device_handle = core_info['Device Handle']

                # Check if passed core is a core ID
                if core_selection == core_id:
                    selected_device_handles.append(device_handle)
                    valid_core_choice = True
                    break
            if not valid_core_choice:
                logging.debug(f"AMDSMIHelpers.get_device_handles_from_core_selections - Unable to convert {core_selection}")
                valid_core_format = True
                if not core_selection.isdigit():
                    valid_core_format = False
                return False, valid_core_format, core_selection
        return True, True, selected_device_handles


    def handle_gpus(self, args, logger, subcommand):
        """This function will run execute the subcommands based on the number
            of gpus passed in via args.
        params:
            args - argparser args to pass to subcommand
            current_platform_args (list) - GPU supported platform arguments
            current_platform_values (list) - GPU supported values for the arguments
            logger (AMDSMILogger) - Logger to print out output
            subcommand (AMDSMICommands) - Function that can handle multiple gpus

        return:
            tuple(bool, device_handle) :
                bool - True if executed subcommand for multiple devices
                device_handle - Return the device_handle if the list of devices is a length of 1
            (handled_multiple_gpus, device_handle)

        """
        if isinstance(args.gpu, list):
            if len(args.gpu) > 1:
                for device_handle in args.gpu:
                    # Handle multiple_devices to print all output at once
                    subcommand(args, multiple_devices=True, gpu=device_handle)
                logger.print_output(multiple_device_enabled=True)
                return True, args.gpu
            elif len(args.gpu) == 1:
                args.gpu = args.gpu[0]
                return False, args.gpu
            else:
                logging.debug("args.gpu has an empty list")
        else:
            return False, args.gpu


    def handle_switchs(self, args, logger, subcommand):
    
        """This function will run execute the subcommands based on the number
            of gpus passed in via args.
        params:
            args - argparser args to pass to subcommand
            current_platform_args (list) - GPU supported platform arguments
            current_platform_values (list) - GPU supported values for the arguments
            logger (AMDSMILogger) - Logger to print out output
            subcommand (AMDSMICommands) - Function that can handle multiple gpus

        return:
            tuple(bool, device_handle) :
                bool - True if executed subcommand for multiple devices
                device_handle - Return the device_handle if the list of devices is a length of 1
            (handled_multiple_gpus, device_handle)

        """
    
        if isinstance(args.switch, list):
        
            if len(args.switch) > 1:
                for device_handle in args.switch:
                    device_type=amdsmi_interface.amdsmi_get_processor_type(device_handle)
                    if device_type["processor_type"]==amdsmi_interface.AmdSmiProcessorType(amdsmi_interface.amdsmi_wrapper.AMDSMI_PROCESSOR_TYPE_BRCM_SWITCH).name:
                        subcommand(args, multiple_devices=True, switch=device_handle)
                
                logger.print_output(multiple_device_enabled=True)
                return True, args.switch
            elif len(args.switch) == 1:
                args.switch = args.switch[0]
                return False, args.switch
            else:
                logging.debug("args.switch has an empty list")
        else:
            return False, args.switch

    def handle_brcm_nics(self, args, logger, subcommand):
    
        """This function will run execute the subcommands based on the number
            of nics passed in via args.
        params:
            args - argparser args to pass to subcommand
            current_platform_args (list) - nic supported platform arguments
            current_platform_values (list) - nic supported values for the arguments
            logger (AMDSMILogger) - Logger to print out output
            subcommand (AMDSMICommands) - Function that can handle multiple nics

        return:
            tuple(bool, device_handle) :
                bool - True if executed subcommand for multiple devices
                device_handle - Return the device_handle if the list of devices is a length of 1
            (handled_multiple_gpus, device_handle)

        """
    
        if isinstance(args.nic, list):
            
            if len(args.nic) > 1:
        
                for device_handle in args.nic:
                
                    device_type=amdsmi_interface.amdsmi_get_processor_type(device_handle)
                    if device_type["processor_type"]==amdsmi_interface.AmdSmiProcessorType(amdsmi_interface.amdsmi_wrapper.AMDSMI_PROCESSOR_TYPE_BRCM_NIC).name:
                        subcommand(args, multiple_devices=True, nic=device_handle)

                logger.print_output(multiple_device_enabled=True)
                return True, args.nic
            elif len(args.nic) == 1:
                args.nic = args.nic[0]
                return False, args.nic
            else:
                logging.debug("args.nic has an empty list")
        else:
            return False, args.nic

    def handle_ainics(self, args, logger, subcommand):
    
        """This function will run execute the subcommands based on the number
            of nics passed in via args.
        params:
            args - argparser args to pass to subcommand
            current_platform_args (list) - nic supported platform arguments
            current_platform_values (list) - nic supported values for the arguments
            logger (AMDSMILogger) - Logger to print out output
            subcommand (AMDSMICommands) - Function that can handle multiple nics

        return:
            tuple(bool, device_handle) :
                bool - True if executed subcommand for multiple devices
                device_handle - Return the device_handle if the list of devices is a length of 1
            (handled_multiple_gpus, device_handle)

        """
    
        if isinstance(args.nic, list):
            
            if len(args.nic) > 1:
        
                for device_handle in args.nic:
                
                    device_type=amdsmi_interface.amdsmi_get_processor_type(device_handle)
                    if device_type["processor_type"]==amdsmi_interface.AmdSmiProcessorType(amdsmi_interface.amdsmi_wrapper.AMDSMI_PROCESSOR_TYPE_AMD_NIC).name:
                        subcommand(args, multiple_devices=True, nic=device_handle)

                logger.print_output(multiple_device_enabled=True)
                return True, args.nic
            elif len(args.nic) == 1:
                args.nic = args.nic[0]
                return False, args.nic
            else:
                logging.debug("args.nic has an empty list")
        else:
            return False, args.nic

    def handle_cpus(self, args, logger, subcommand):
        """This function will run execute the subcommands based on the number
            of cpus passed in via args.
        params:
            args - argparser args to pass to subcommand
            logger (AMDSMILogger) - Logger to print out output
            subcommand (AMDSMICommands) - Function that can handle multiple gpus

        return:
            tuple(bool, device_handle) :
                bool - True if executed subcommand for multiple devices
                device_handle - Return the device_handle if the list of devices is a length of 1
            (handled_multiple_gpus, device_handle)

        """
        if isinstance(args.cpu, list):
            if len(args.cpu) > 1:
                for device_handle in args.cpu:
                    # Handle multiple_devices to print all output at once
                    subcommand(args, multiple_devices=True, cpu=device_handle)
                logger.print_output(multiple_device_enabled=True)
                return True, args.cpu
            elif len(args.cpu) == 1:
                args.cpu = args.cpu[0]
                return False, args.cpu
            else:
                logging.debug("args.cpu has empty list")
        else:
            return False, args.cpu


    def handle_cores(self, args, logger, subcommand):
        """This function will run execute the subcommands based on the number
            of cores passed in via args.
        params:
            args - argparser args to pass to subcommand
            logger (AMDSMILogger) - Logger to print out output
            subcommand (AMDSMICommands) - Function that can handle multiple gpus

        return:
            tuple(bool, device_handle) :
                bool - True if executed subcommand for multiple devices
                device_handle - Return the device_handle if the list of devices is a length of 1
            (handled_multiple_gpus, device_handle)

        """
        if isinstance(args.core, list):
            if len(args.core) > 1:
                for device_handle in args.core:
                    # Handle multiple_devices to print all output at once
                    subcommand(args, multiple_devices=True, core=device_handle)
                logger.print_output(multiple_device_enabled=True)
                return True, args.core
            elif len(args.core) == 1:
                args.core = args.core[0]
                return False, args.core
            else:
                logging.debug("args.core has empty list")
        else:
            return False, args.core


    # The below handle_nodes function is currently unused as only node 0 is supported.
    # Marked as a private function until it is needed in the future.
    def _handle_nodes(self, args, logger, subcommand):
        """This function will run execute the subcommands based on the number
            of nodes passed in via args.
        params:
            args - argparser args to pass to subcommand
            current_platform_args (list) - GPU supported platform arguments
            current_platform_values (list) - GPU supported values for the arguments
            logger (AMDSMILogger) - Logger to print out output
            subcommand (AMDSMICommands) - Function that can handle multiple gpus

        return:
            tuple(bool, device_handle) :
                bool - True if executed subcommand for multiple devices
                device_handle - Return the device_handle if the list of devices is a length of 1
            (handled_multiple_nodes, device_handle)

        """
        if isinstance(args.node, list):
            if len(args.node) > 1:
                for node_handle in args.node:
                    # Handle multiple_devices to print all output at once
                    subcommand(args, multiple_devices=True, node=node_handle)
                logger.print_output(multiple_device_enabled=True)
                return True, args.node
            elif len(args.node) == 1:
                args.node = args.node[0]
                return False, args.node
            else:
                logging.debug("args.node has an empty list")
        else:
            return False, args.node


    def handle_watch(self, args, subcommand, logger):
        """This function will run the subcommand multiple times based
            on the passed watch, watch_time, and iterations passed in.
        params:
            args - argparser args to pass to subcommand
            subcommand (AMDSMICommands) - Function that can handle
                watching output (Currently: metric & process)
            logger (AMDSMILogger) - Logger for accessing config values
        return:
            Nothing
        """
        # Set the values for watching as the args will cleared
        watch = args.watch
        watch_time = args.watch_time
        iterations = args.iterations

        # Set the args values to None so we don't loop recursively
        args.watch = None
        args.watch_time = None
        args.iterations = None

        # Set the signal handler to flush a delmiter to file if the format is json
        print("'CTRL' + 'C' to stop watching output:")
        if watch_time:  # Run for set amount of time
            iterations_ran = 0
            end_time = time.time() + watch_time
            while time.time() <= end_time:
                subcommand(args, watching_output=True)
                # Handle iterations limit
                iterations_ran += 1
                if iterations is not None:
                    if iterations <= iterations_ran:
                        break
                time.sleep(watch)
        elif iterations is not None:  # Run for a set amount of iterations
            for iteration in range(iterations):
                subcommand(args, watching_output=True)
                if iteration == iterations - 1:  # Break on iteration completion
                    break
                time.sleep(watch)
        else:  # Run indefinitely as watch_time and iterations are not set
            while True:
                subcommand(args, watching_output=True)
                time.sleep(watch)

        return 1


    def get_gpu_id_from_device_handle(self, input_device_handle):
        """Get the gpu index from the device_handle.
        amdsmi_get_processor_handles() returns the list of device_handles in order of gpu_index
        """
        device_handles = amdsmi_interface.amdsmi_get_processor_handles()
        for gpu_index, device_handle in enumerate(device_handles):
            if input_device_handle.value == device_handle.value:
                return gpu_index
        raise amdsmi_exception.AmdSmiParameterException(input_device_handle,
                                                        amdsmi_interface.amdsmi_wrapper.amdsmi_processor_handle,
                                                        "Unable to find gpu ID from device_handle")
    def get_nic_id_from_device_handle(self, input_device_handle):
        """Get the nic index from the device_handle.
        get_nic_handles() returns the list of device_handles in order of nic_index
        """
        device_handles = amdsmi_interface.get_nic_handles()
        if len(device_handles) == 0:
            return -1
        for nic_index, device_handle in enumerate(device_handles):
            if input_device_handle.value == device_handle.value:
                return nic_index
        raise amdsmi_exception.AmdSmiParameterException(input_device_handle,
                                                        amdsmi_interface.amdsmi_wrapper.amdsmi_processor_handle,
                                                        "Unable to find nic ID from device_handle")

    def get_ainic_id_from_device_handle(self, input_device_handle):
        """Get the ainic index from the device_handle.
        get_ainic_handles() returns the list of device_handles in order of ainic_index
        """
        device_handles = amdsmi_interface.get_ainic_handles()
        if len(device_handles) == 0:
            return -1
        for nic_index, device_handle in enumerate(device_handles):
            if input_device_handle.value == device_handle.value:
                return nic_index
        raise amdsmi_exception.AmdSmiParameterException(input_device_handle,
                                                        amdsmi_interface.amdsmi_wrapper.amdsmi_processor_handle,
                                                        "Unable to find nic ID from device_handle")

    def get_switch_id_from_device_handle(self, input_device_handle):
        """Get the nic index from the device_handle.
        get_switch_handles() returns the list of device_handles in order of nic_index
        """
        device_handles = amdsmi_interface.get_switch_handles()
        for switch_index, device_handle in enumerate(device_handles):
            if input_device_handle.value == device_handle.value:
                return switch_index
        raise amdsmi_exception.AmdSmiParameterException(input_device_handle,
                                                        amdsmi_interface.amdsmi_wrapper.amdsmi_processor_handle,
                                                        "Unable to find switch ID from device_handle")

    def get_cpu_id_from_device_handle(self, input_device_handle):
        """Get the cpu index from the device_handle.
        amdsmi_interface.amdsmi_get_cpusocket_handles() returns the list of device_handles in order of cpu_index
        """
        device_handles = amdsmi_interface.amdsmi_get_cpusocket_handles()
        for cpu_index, device_handle in enumerate(device_handles):
            if input_device_handle.value == device_handle.value:
                return cpu_index
        raise amdsmi_exception.AmdSmiParameterException(input_device_handle,
                                                        amdsmi_interface.amdsmi_wrapper.amdsmi_processor_handle,
                                                        "Unable to find cpu ID from device_handle")


    def get_core_id_from_device_handle(self, input_device_handle):
        """Get the core index from the device_handle.
        amdsmi_interface.amdsmi_get_cpusocket_handles() returns the list of device_handles in order of cpu_index
        """
        device_handles = amdsmi_interface.amdsmi_get_cpucore_handles()
        for core_index, device_handle in enumerate(device_handles):
            if input_device_handle.value == device_handle.value:
                return core_index
        raise amdsmi_exception.AmdSmiParameterException(input_device_handle,
                                                        amdsmi_interface.amdsmi_wrapper.amdsmi_processor_handle,
                                                        "Unable to find core ID from device_handle")


    def get_amd_gpu_bdfs(self):
        """Return a list of GPU BDFs visibile to amdsmi

        Returns:
            list[BDF]: List of GPU BDFs
        """
        gpu_bdfs = []
        device_handles = amdsmi_interface.amdsmi_get_processor_handles()

        for device_handle in device_handles:
            bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(device_handle)
            gpu_bdfs.append(bdf)

        return gpu_bdfs


    def get_apu_memory_type_and_name(self, device_handle, gpu_id=None):
        """Determine the appropriate memory type for APU devices

        For APU devices, compare VRAM and GTT totals and return the larger one.
        For discrete GPUs, return VRAM.

        Args:
            device_handle: GPU device handle
            gpu_id: Optional GPU ID for logging purposes

        Returns:
            tuple: (memory_type, memory_type_name) where memory_type is AmdSmiMemoryType enum
                   and memory_type_name is string ("VRAM" or "GTT")
        """
        # Default to VRAM
        mem_type = amdsmi_interface.AmdSmiMemoryType.VRAM
        mem_type_name = "VRAM"

        if gpu_id is None:
            try:
                gpu_id = self.get_gpu_id_from_device_handle(device_handle)
            except:
                gpu_id = "unknown"

        try:
            # Check ASIC info flags to see if it's an APU (AMDGPU_IDS_FLAGS_FUSION = 0x1)
            asic_info = amdsmi_interface.amdsmi_get_gpu_asic_info(device_handle)
            if 'flags' in asic_info and (asic_info['flags'] & 0x1):
                # For APUs, compare VRAM and GTT totals and use the larger one
                try:
                    vram_total_check = amdsmi_interface.amdsmi_get_gpu_memory_total(device_handle, amdsmi_interface.AmdSmiMemoryType.VRAM) // (1024*1024)
                    gtt_total_check = amdsmi_interface.amdsmi_get_gpu_memory_total(device_handle, amdsmi_interface.AmdSmiMemoryType.GTT) // (1024*1024)

                    if gtt_total_check > vram_total_check:
                        mem_type = amdsmi_interface.AmdSmiMemoryType.GTT
                        mem_type_name = "GTT"
                    logging.debug("APU detected for gpu %s, using %s (VRAM: %d MB, GTT: %d MB)", gpu_id, mem_type_name, vram_total_check, gtt_total_check)
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug("Failed to compare memory types for APU gpu %s, defaulting to VRAM | %s", gpu_id, e.get_error_info())
        except amdsmi_exception.AmdSmiLibraryException as e:
            logging.debug("Failed to get ASIC info for gpu %s, defaulting to VRAM | %s", gpu_id, e.get_error_info())

        return mem_type, mem_type_name


    def is_amd_device(self, device_handle):
        """ Return whether the specified device is an AMD device or not

        param device: DRM device identifier
        """
        # Get card vendor id
        asic_info = amdsmi_interface.amdsmi_get_gpu_asic_info(device_handle)
        try:
            vendor_value = int(asic_info['vendor_id'], 16)
            return vendor_value == AMD_VENDOR_ID
        except:
            return False


    def get_perf_levels(self):
        perf_levels_str = [clock.name for clock in amdsmi_interface.AmdSmiDevPerfLevel]
        perf_levels_int = list(set(clock.value for clock in amdsmi_interface.AmdSmiDevPerfLevel))
        return perf_levels_str, perf_levels_int

    def get_ptl_values(self):
        ptl_values_str = [ptl.name for ptl in amdsmi_interface.AmdSmiPtlData]
        ptl_values_int = list(set(ptl.name for ptl in amdsmi_interface.AmdSmiPtlData))
        return ptl_values_str,ptl_values_int

    def get_accelerator_partition_profile_config(self):
        device_handles = amdsmi_interface.amdsmi_get_processor_handles()
        accelerator_partition_profiles = {'profile_indices':[], 'profile_types':[], 'memory_caps': []}
        for dev in device_handles:
            try:
                profile = amdsmi_interface.amdsmi_get_gpu_accelerator_partition_profile_config(dev)
                num_profiles = profile['num_profiles']
                for p in range(num_profiles):
                    accelerator_partition_profiles['profile_indices'].append(str(profile['profiles'][p]['profile_index']))
                    accelerator_partition_profiles['profile_types'].append(profile['profiles'][p]['profile_type'])
                    accelerator_partition_profiles['memory_caps'].append(profile['profiles'][p]['memory_caps'])
                break # Only need to get the profiles for one device
            except amdsmi_interface.AmdSmiLibraryException as e:
                logging.debug(f"AMDSMIHelpers.get_accelerator_partition_profile_config - Unable to get accelerator partition profile config for device {dev}: {str(e)}")
                if e.err_code == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED:
                    logging.debug(f"AMDSMIHelpers.get_accelerator_partition_profile_config - Device {dev} does not support accelerator partition profiles")
                    return accelerator_partition_profiles
                break
            except Exception as e:
                logging.debug(f"AMDSMIHelpers.get_accelerator_partition_profile_config - Unexpected error occurred --> Unable to get accelerator partition profile config for device {dev}: {str(e)}")
                break
        return accelerator_partition_profiles


    def get_accelerator_choices_types_indices(self):
        return_val = ("N/A", {'profile_indices':[], 'profile_types':[]})
        if os.geteuid() != 0:
            logging.debug("AMDSMIHelpers.get_accelerator_choices_types_indices - Not root, unable to get accelerator partition profiles")
            # If not root, we can't get the accelerator partition profiles
            return return_val
        else:
            logging.debug("AMDSMIHelpers.get_accelerator_choices_types_indices - Root, getting accelerator partition profiles")
        accelerator_partition_profiles = self.get_accelerator_partition_profile_config()
        if len(accelerator_partition_profiles['profile_types']) != 0:
            compute_partitions_list = accelerator_partition_profiles['profile_types'] + accelerator_partition_profiles['profile_indices']
            return_val = (compute_partitions_list, accelerator_partition_profiles)
        return return_val


    def get_memory_partition_types(self):
        memory_partitions_str = [partition.name for partition in amdsmi_interface.AmdSmiMemoryPartitionType]
        if 'UNKNOWN' in memory_partitions_str:
            memory_partitions_str.remove('UNKNOWN')
        return memory_partitions_str


    def get_clock_types(self):
        clock_types_str = [clock.name for clock in amdsmi_interface.AmdSmiClkType]
        clock_types_int = list(set(clock.value for clock in amdsmi_interface.AmdSmiClkType))
        return clock_types_str, clock_types_int


    def get_power_profiles(self):
        power_profiles_str = [profile.name for profile in amdsmi_interface.AmdSmiPowerProfilePresetMasks]
        if 'UNKNOWN' in power_profiles_str:
            power_profiles_str.remove('UNKNOWN')
        return power_profiles_str


    def get_power_profile_name_mapping(self):
        """Returns dict mapping friendly names to enum values"""
        return {
            'CUSTOM': amdsmi_interface.AmdSmiPowerProfilePresetMasks.CUSTOM_MASK,
            'VIDEO': amdsmi_interface.AmdSmiPowerProfilePresetMasks.VIDEO_MASK,
            'POWER_SAVING': amdsmi_interface.AmdSmiPowerProfilePresetMasks.POWER_SAVING_MASK,
            'COMPUTE': amdsmi_interface.AmdSmiPowerProfilePresetMasks.COMPUTE_MASK,
            'VR': amdsmi_interface.AmdSmiPowerProfilePresetMasks.VR_MASK,
            '3D_FULL_SCREEN': amdsmi_interface.AmdSmiPowerProfilePresetMasks.THREE_D_FULL_SCR_MASK,
            'BOOTUP_DEFAULT': amdsmi_interface.AmdSmiPowerProfilePresetMasks.BOOTUP_DEFAULT,
        }


    def get_profile_name_from_mask(self, mask):
        """Convert mask value to friendly name"""
        reverse_mapping = {v: k for k, v in self.get_power_profile_name_mapping().items()}
        return reverse_mapping.get(mask, 'UNKNOWN')


    def parse_available_profiles(self, available_profiles_bitfield):
        """Extract list of profile names from bitfield"""
        profiles = []
        for name, mask in self.get_power_profile_name_mapping().items():
            if available_profiles_bitfield & mask:
                profiles.append(name)
        return profiles


    def get_perf_det_levels(self):
        perf_det_level_str = [level.name for level in amdsmi_interface.AmdSmiDevPerfLevel]
        if 'UNKNOWN' in perf_det_level_str:
            perf_det_level_str.remove('UNKNOWN')
        return perf_det_level_str


    def get_power_caps(self):
        device_handles = amdsmi_interface.amdsmi_get_processor_handles()
        power_limit_types = {
            'ppt0': {
                'power_cap_min': amdsmi_interface.MaxUIntegerTypes.UINT64_T,
                'power_cap_max': 0
            },
            'ppt1': {
                'power_cap_min': amdsmi_interface.MaxUIntegerTypes.UINT64_T,
                'power_cap_max': 0
            }
        }

        for dev in device_handles:
            try:
                power_cap_types = amdsmi_interface.amdsmi_get_supported_power_cap(dev)
                for sensor in power_cap_types['sensor_inds']:
                    power_cap_info = amdsmi_interface.amdsmi_get_power_cap_info(dev, sensor)
                    if power_cap_info['max_power_cap'] > power_limit_types[f'ppt{sensor}']['power_cap_max']:
                        power_limit_types[f'ppt{sensor}']['power_cap_max'] = power_cap_info['max_power_cap']
                    if power_cap_info['min_power_cap'] < power_limit_types[f'ppt{sensor}']['power_cap_min']:
                        power_limit_types[f'ppt{sensor}']['power_cap_min'] = power_cap_info['min_power_cap']
            except (amdsmi_interface.AmdSmiLibraryException, KeyError) as e:
                logging.debug(f"AMDSMIHelpers.get_power_caps - Unable to get power cap info for device {dev}: {str(e)}")
                continue

        # If we never found a real min or max, set them to N/A
        for ppt_key in ['ppt0', 'ppt1']:
            if power_limit_types[ppt_key]['power_cap_min'] == amdsmi_interface.MaxUIntegerTypes.UINT64_T:
                power_limit_types[ppt_key]['power_cap_min'] = "N/A"
            if power_limit_types[ppt_key]['power_cap_max'] == 0:
                power_limit_types[ppt_key]['power_cap_max'] = "N/A"

        ppt0_power_cap_max = self.format_power_cap(power_limit_types['ppt0']['power_cap_max'])
        ppt0_power_cap_min = self.format_power_cap(power_limit_types['ppt0']['power_cap_min'])
        ppt1_power_cap_max = self.format_power_cap(power_limit_types['ppt1']['power_cap_max'])
        ppt1_power_cap_min = self.format_power_cap(power_limit_types['ppt1']['power_cap_min'])

        return (ppt0_power_cap_min, ppt0_power_cap_max, ppt1_power_cap_min, ppt1_power_cap_max)


    def format_power_cap(self, value):
        if value != "N/A":
            converted = self.convert_SI_unit(value, AMDSMIHelpers.SI_Unit.MICRO)
            return f"{converted} W"
        return value


    def get_fan_support(self):
        """Check if fan control is supported on the first device.

        Returns:
            str: "0-255 or 0-100%%" if fan control is supported, "N/A" otherwise
        """
        device_handles = amdsmi_interface.amdsmi_get_processor_handles()
        for dev in device_handles:
            try:
                # Try to get both fan speed and max fan speed
                # If both succeed, fan control is supported
                _ = amdsmi_interface.amdsmi_get_gpu_fan_speed(dev, 0)
                _ = amdsmi_interface.amdsmi_get_gpu_fan_speed_max(dev, 0)
                # Fan control is supported on this device
                return "0-255 or 0-100%%"
            except amdsmi_interface.AmdSmiLibraryException as e:
                logging.debug(f"AMDSMIHelpers.get_fan_support - Unable to get fan info for device {dev}: {str(e)}")
                if e.err_code == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED:
                    logging.debug(f"AMDSMIHelpers.get_fan_support - Device {dev} does not support fan control")
                    return "N/A"
                return "N/A"
            except Exception as e:
                logging.debug(f"AMDSMIHelpers.get_fan_support - Unexpected error occurred --> Unable to get fan info for device {dev}: {str(e)}")
                return "N/A"
            # Only check the first device (socket device, never partition)
            break
        return "N/A"


    def get_soc_pstates(self):
        device_handles = amdsmi_interface.amdsmi_get_processor_handles()
        soc_pstate_profile_list = []
        for dev in device_handles:
            try:
                soc_pstate_info = amdsmi_interface.amdsmi_get_soc_pstate(dev)
                # Check if 'policies' key exists before accessing it
                if 'policies' in soc_pstate_info and soc_pstate_info['policies']:
                    for policy in soc_pstate_info['policies']:
                        policy_string = f"{policy['policy_id']}: {policy['policy_description']}"
                        if not policy_string in soc_pstate_profile_list:
                            soc_pstate_profile_list.append(policy_string)
            except amdsmi_interface.AmdSmiLibraryException as e:
                continue
            except KeyError as e:
                logging.debug(f"AMDSMIHelpers.get_soc_pstates - Missing key in soc_pstate_info: {e}")
                continue
        if len(soc_pstate_profile_list) == 0:
            soc_pstate_profile_list.append("N/A")
        return soc_pstate_profile_list


    def get_xgmi_plpd_policies(self):
        device_handles = amdsmi_interface.amdsmi_get_processor_handles()
        xgmi_plpd_profile_list = []
        for dev in device_handles:
            try:
                xgmi_plpd_info = amdsmi_interface.amdsmi_get_xgmi_plpd(dev)
                # Check if 'policies' key exists before accessing it
                if 'policies' in xgmi_plpd_info and xgmi_plpd_info['policies']:
                    for policy in xgmi_plpd_info['policies']:
                        policy_string = f"{policy['policy_id']}: {policy['policy_description']}"
                        if not policy_string in xgmi_plpd_profile_list:
                            xgmi_plpd_profile_list.append(policy_string)
            except amdsmi_interface.AmdSmiLibraryException as e:
                continue
            except KeyError as e:
                logging.debug(f"AMDSMIHelpers.get_xgmi_plpd_policies - Missing key in xgmi_plpd_info: {e}")
                continue
        if len(xgmi_plpd_profile_list) == 0:
            xgmi_plpd_profile_list.append("N/A")
        return xgmi_plpd_profile_list


    def validate_clock_type(self, input_clock_type):
        valid_clock_types_str, valid_clock_types_int = self.get_clock_types()

        valid_clock_input = False
        if isinstance(input_clock_type, str):
            for clock_type in valid_clock_types_str:
                if input_clock_type.lower() == clock_type.lower():
                    input_clock_type = clock_type # Set input_clock_type to enum value in AmdSmiClkType
                    valid_clock_input = True
                    break
        elif isinstance(input_clock_type, int):
            if input_clock_type in valid_clock_types_int:
                input_clock_type = amdsmi_interface.AmdSmiClkType(input_clock_type)
                valid_clock_input = True

        return valid_clock_input, input_clock_type


    def confirm_out_of_spec_warning(self, auto_respond=False):
        """ Print the warning for running outside of specification and prompt user to accept the terms.

        @param auto_respond: Response to automatically provide for all prompts
        """
        print('''
            ******WARNING******\n
            Operating your AMD GPU outside of official AMD specifications or outside of
            factory settings, including but not limited to the conducting of overclocking,
            over-volting or under-volting (including use of this interface software,
            even if such software has been directly or indirectly provided by AMD or otherwise
            affiliated in any way with AMD), may cause damage to your AMD GPU, system components
            and/or result in system failure, as well as cause other problems.
            DAMAGES CAUSED BY USE OF YOUR AMD GPU OUTSIDE OF OFFICIAL AMD SPECIFICATIONS OR
            OUTSIDE OF FACTORY SETTINGS ARE NOT COVERED UNDER ANY AMD PRODUCT WARRANTY AND
            MAY NOT BE COVERED BY YOUR BOARD OR SYSTEM MANUFACTURER'S WARRANTY.
            Please use this utility with caution.
            ''')
        if not auto_respond:
            user_input = input('Do you accept these terms? [y/n] ')
        else:
            user_input = auto_respond
        if user_input in ['y', 'Y', 'yes', 'Yes', 'YES']:
            return
        else:
            sys.exit('Confirmation not given. Exiting without setting value')


    def confirm_changing_memory_partition_gpu_reload_warning(self, auto_respond=False):
        """ Print the warning for running outside of specification and prompt user to accept the terms.

        :param autoRespond: Response to automatically provide for all prompts
        """

        print('''
            ******WARNING******\n
            After changing memory (NPS) partition modes, users MUST restart
            (reload) the AMD GPU driver. This command NO LONGER AUTOMATICALLY
            reloads the driver, see `amd-smi reset -h` and
            `sudo amd-smi reset -r` for more information.

            This change is intended to allow users the ability to control when is
            the best time to restart the AMD GPU driver, as it may not be desired
            to restart the AMD GPU driver immediately after changing the
            memory (NPS) partition mode.

            Please use `sudo amd-smi reset -r` AFTER successfully
            changing the memory (NPS) partition mode. A successful driver reload
            is REQUIRED in order to complete updating ALL GPUs in the hive to
            the requested partition mode.

            ******REMINDER******
            In order to reload the AMD GPU driver, users MUST quit all GPU
            workloads across all devices.
            ''')

        if not auto_respond:
            user_input = input('Do you accept these terms? [Y/N] ')
        else:
            user_input = auto_respond
        if user_input in ['Yes', 'yes', 'y', 'Y', 'YES']:
            print('')
            return
        else:
            print('Confirmation not given. Exiting without setting value')
            sys.exit(1)

    def confirm_gpu_driver_reload_warning(self, auto_respond=False):
        """ Print the warning for running outside of specification and prompt user to accept the terms.

        :param autoRespond: Response to automatically provide for all prompts
        """
        print('''
          ****** WARNING ******\n
          AMD SMI is about to initiate an AMD GPU driver restart (module reload).

          Reloading the AMD GPU driver REQUIRES users to quit all GPU activity across all
          devices.

          If user is initiating a driver reload AFTER changing memory (NPS) partition
          modes (`sudo amd-smi set -M <NPS_MODE>`), a AMD GPU driver reload is REQUIRED
          to complete updating the partition mode. This change will effect ALL GPUs in
          the hive. Advise using `amd-smi list -e` and `amd-smi partition -c -m`
          afterwards to ensure changes were applied as expected.

          Please use this utility with caution.
          ''')
        if not auto_respond:
            user_input = input('Do you accept these terms? [Y/N] ')
        else:
            user_input = auto_respond
        if user_input in ['Yes', 'yes', 'y', 'Y', 'YES']:
            print('')
            return
        else:
            print('Confirmation not given. Exiting without setting value')
            sys.exit(1)

    def is_valid_profile(self, profile):
        profile_presets = amdsmi_interface.amdsmi_wrapper.amdsmi_power_profile_preset_masks_t__enumvalues
        if profile in profile_presets:
            return True, profile_presets[profile]
        else:
            return False, profile_presets.values()


    def convert_bytes_to_readable(self, bytes_input, format_length=None):
        if isinstance(bytes_input, str):
            return "N/A"
        for unit in ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB"]:
            if abs(bytes_input) < 1024:
                if format_length is not None:
                    if bytes_input < 10:
                        return f"{bytes_input:4.3f} {unit}"
                    elif bytes_input < 100:
                        return f"{bytes_input:4.2f} {unit}"
                    elif bytes_input < 1000:
                        return f"{bytes_input:4.1f} {unit}"
                    else:
                        return f"{bytes_input:4.0f} {unit}"
                else:
                    return f"{bytes_input:3.1f} {unit}"
            bytes_input /= 1024
        return f"{bytes_input:.1f} YB"

    def convert_time_to_readable(self, time_value, base_unit="us"):
        """Convert time values to readable format, auto-scaling units to keep value under 1000.
        
        Keeps the numeric value to max 3 digits by bumping units:
        us (microseconds) -> ms (milliseconds) -> s (seconds)
        
        params:
            time_value - the time value to convert
            base_unit - the base unit of the input ("us" or "ms")
        return:
            str: formatted time string (e.g., "123 us", "45 ms", "2 s")
        """
        if isinstance(time_value, str) or time_value == "N/A":
            return "N/A"
        
        # Define unit progression
        if base_unit == "us":
            units = [("us", 1), ("ms", 1000), ("s", 1000000)]
        elif base_unit == "ms":
            units = [("ms", 1), ("s", 1000)]
        else:
            return f"{time_value} {base_unit}"
        
        # Find appropriate unit (keep rounded value under 1000)
        for unit_name, divisor in units:
            scaled_value = time_value / divisor
            rounded_value = round(scaled_value)
            if rounded_value < 1000:
                return f"{rounded_value} {unit_name}"
        
        # If value is huge, use the largest unit
        unit_name, divisor = units[-1]
        return f"{int(time_value / divisor)} {unit_name}"




    def unit_format(self, logger, value, unit):
        """This function will format output with unit based on the logger output format

        params:
            logger (AMDSMILogger) - Logger to print out output
            value - the value to be formatted
            unit - the unit to be formatted with the value
        return:
            str or dict : formatted output
        """
        if isinstance(value, list):
            formatted_values = []
            for val in value:
                if isinstance(val, str) and val == "N/A":
                    formatted_values.append("N/A")
                else:
                    formatted_values.append(self.unit_format(logger, val, unit))
            return formatted_values
        else:
            if value == "N/A":
                return "N/A"
            if logger.is_json_format():
                if unit:
                    return {"value": value, "unit": unit}
                else:
                    return value
            if logger.is_csv_format():
                # For CSV, return the raw value (number or "N/A"), not a string
                return value
            if logger.is_human_readable_format():
                if unit:
                    return f"{value} {unit}".rstrip()
                else:
                    return f"{value}".rstrip()
            return f"{value}"


    class SI_Unit(float, Enum):
        GIGA = 1000000000  # 10^9
        MEGA = 1000000     # 10^6
        KILO = 1000        # 10^3
        HECTO = 100        # 10^2
        DEKA = 10          # 10^1
        BASE = 1           # 10^0
        DECI = 0.1         # 10^-1
        CENTI = 0.01       # 10^-2
        MILLI = 0.001      # 10^-3
        MICRO = 0.000001   # 10^-6
        NANO = 0.000000001 # 10^-9


    def convert_SI_unit(self, val: Union[int, float], unit_in: SI_Unit, unit_out = SI_Unit.BASE) -> Union[int, float]:
        """This function will convert a value into another
         scientific (SI) unit. Defaults unit_out to SI_Unit.BASE

        params:
            val: int or float unit to convert
            unit_in: Requires using SI_Unit to set current value's SI unit (eg. SI_Unit.MICRO)
            unit_out - Requires using SI_Unit to set current value's SI unit
             default value is SI_Unit.BASE (eg. SI_Unit.MICRO)
        return:
            int or float : converted SI unit of value requested
        """
        if isinstance(val, float):
            return val * unit_in / unit_out
        elif isinstance(val, int):
            return int(float(val) * unit_in / unit_out)
        else:
            raise TypeError("val must be an int or float")


    def get_pci_device_ids(self) -> Set[str]:
        pci_devices_path = "/sys/bus/pci/devices"
        pci_devices: set[str] = set()
        for device in os.listdir(pci_devices_path):
            device_path = os.path.join(pci_devices_path, device, "device")
            try:
                with open(device_path, 'r') as f:
                    device = f.read().strip()
                    pci_devices.add(device)
            except Exception as _:
                continue
        return pci_devices


    def progressbar(self, it, prefix="", size=60, out=sys.stdout, add_newline=False):
        count = len(it)
        if (add_newline):
            print("{}\n".format(prefix),end='\r', file=out, flush=False)
        else:
            print("{}".format(prefix),end='\r', file=out, flush=False)
        def show(j):
            x = int(size*j/count)
            print("[{}{}] {}/{} secs remain".format(u"█"*x, "."*(size-x), j, count),
                    end='\r', file=out, flush=True)
        show(0)
        for i, item in enumerate(it):
            yield item
            show(i+1)
        print("\n\n", end='\r', flush=True, file=out)


    def showProgressbar(self, title="", timeInSeconds=13, add_newline=False):
        if title != "":
            title += " "
        for i in self.progressbar(range(timeInSeconds), title, 40, add_newline=add_newline):
            time.sleep(1)

    @lru_cache(maxsize=128)
    def _cached_group_name(self, gid: int) -> str:
        try:
            return grp.getgrgid(gid).gr_name
        except Exception:
            # In containers, the UID may not resolve to a name
            return str(gid)

    @lru_cache(maxsize=128)
    def _cached_user_name(self, uid: int) -> str:
        try:
            return pwd.getpwuid(uid).pw_name
        except Exception:
            # In containers, the GID may not resolve to a name
            return str(uid)

    # Attempt to grab file info
    def _stat_info(self, path: str) -> dict:
        try:
            st = os.stat(path)
            return {
                "uid": st.st_uid,
                "gid": st.st_gid,
                "user": self._cached_user_name(st.st_uid),
                "group": self._cached_group_name(st.st_gid),
            }
        except Exception as e:
            return {"error": str(e)}

    def _has_read_access(self, path: str) -> Tuple[bool, Optional[int], Optional[str]]:
        """
        Check whether the current (real/effective) user can read the given path
        without opening it. Returns (ok:bool, errno_or_None, message_or_None)
        """
        try:
            st = os.stat(path)
        except OSError as e:
            return False, e.errno, e.strerror

        # root can always read
        if os.geteuid() == 0:
            return True, None, None

        # Use os.access to check read permission (including ACLs), so that
        # permissions granted via mechanisms like udev/uaccess are respected.
        if os.access(path, os.R_OK):
            return True, None, None

        mode = st.st_mode
        uid = st.st_uid
        gid = st.st_gid

        euid = os.geteuid()
        egid = os.getegid()
        groups = os.getgroups()

        # owner
        if euid == uid:
            if mode & stat.S_IRUSR:
                return True, None, None
            return False, errno.EACCES, "Permission denied (owner)"

        # group
        if gid == egid or gid in groups:
            if mode & stat.S_IRGRP:
                return True, None, None
            return False, errno.EACCES, "Permission denied (group)"

        # other
        if mode & stat.S_IROTH:
            return True, None, None

        return False, errno.EACCES, "Permission denied (other)"

    def check_required_groups(self, check_render=True, check_video=True):
        """
        Check if the current user can access kfd and dri
        Specifically, only care for EACCES/EPERM

        Args:
            check_render (bool): Whether to check  /dev/kfd &  /dev/dri/renderD* devices. Defaults to True.
            check_video (bool): Whether to check /dev/dri/card* devices. Defaults to True.

        Returns:
            bool: True if all checked devices are accessible, False if any permission errors found
        """

        # Skip check if running as root.
        if os.geteuid() == 0:
            return True

        paths_to_check = []

        # Only add paths for device types that are flagged for checking
        if check_render and os.path.exists("/dev/kfd"):
            paths_to_check.append("/dev/kfd")
            paths_to_check += [p for p in sorted(glob.glob("/dev/dri/renderD*"))]

        # Video group corresponds to /dev/dri/card*
        if check_video:
            paths_to_check += [p for p in sorted(glob.glob("/dev/dri/card*"))]

        if not paths_to_check:
            return True

        denied = []

        for path in paths_to_check:
            # Do not try to open all paths, may cause driver issues.
            # Read access is sufficient to check permissions.
            #
            # Reason: GPUs which support partitioning (memory/compute),
            # logical devices will not be valid until configured.
            # See `sudo amd-smi set -h` or applicable APIs
            # to configure on supported hardware.
            #
            # Example error dmesg output:
            # [965358.883112] amdgpu 0000:15:00.0: amdgpu: renderD153 partition 1 not valid!
            # [965358.883283] amdgpu 0000:15:00.0: amdgpu: renderD154 partition 2 not valid!
            # [965358.883438] amdgpu 0000:15:00.0: amdgpu: renderD155 partition 3 not valid!
            # [965358.883594] amdgpu 0000:15:00.0: amdgpu: renderD156 partition 4 not valid!
            # [965358.883749] amdgpu 0000:15:00.0: amdgpu: renderD157 partition 5 not valid!
            # [965358.883904] amdgpu 0000:15:00.0: amdgpu: renderD158 partition 6 not valid!
            # [965358.884060] amdgpu 0000:15:00.0: amdgpu: renderD159 partition 7 not valid!
            ok, err, msg = self._has_read_access(path)
            if ok:
                continue
            # if permission denied or operation not permitted
            if err in (errno.EACCES, errno.EPERM):
                denied.append((path, err, msg, self._stat_info(path)))

        if denied:
            # Collect unique group info from denied devices
            required_groups = {"kfd": [], "renderD": [], "card": []}
            device_types = {"kfd": [], "renderD": [], "card": []}

            for path, err, msg, si in denied:
                if "error" not in si:
                    # Categorize devices and collect unique group info
                    if "/dev/kfd" in path:
                        device_types["kfd"].append(path)
                        required_groups["kfd"].append(si)
                    elif "/dev/dri/renderD" in path:
                        device_types["renderD"].append(path)
                        required_groups["renderD"].append(si)
                    elif "/dev/dri/card" in path:
                        device_types["card"].append(path)
                        required_groups["card"].append(si)

            # Deduplicate group info by converting to tuple for hashing
            for device_type in required_groups:
                unique_groups = list(dict.fromkeys(
                    tuple(sorted(d.items())) for d in required_groups[device_type]
                ))
                required_groups[device_type] = [dict(item) for item in unique_groups]

            lines = []
            lines.append("Permission needed to access required GPU device node(s):")

            # Collect all unique groups for usermod command
            all_groups = set()

            # Show summary of denied devices by type with ownership info
            if device_types["kfd"]:
                lines.append("  • /dev/kfd: Permission denied")
                if len(required_groups["kfd"]) > 1:
                    lines.append("    - Required group(s):")
                else:
                    lines.append("    - Required group:")
                for group_info in required_groups["kfd"]:
                    lines.append(
                        "      - User: {user} (UID={uid}) | Group: {group} (GID={gid})".format(
                            user=group_info["user"],
                            uid=group_info["uid"],
                            group=group_info["group"],
                            gid=group_info["gid"],
                        )
                    )
                    all_groups.add(group_info["group"])

            if device_types["renderD"]:
                lines.append(f"  • /dev/dri/renderD*: {len(device_types['renderD'])} device(s) denied")
                if len(required_groups["renderD"]) > 1:
                    lines.append("    - Required group(s):")
                else:
                    lines.append("    - Required group:")
                for group_info in required_groups["renderD"]:
                    lines.append(
                        "      - User: {user} (UID={uid}) | Group: {group} (GID={gid})".format(
                            user=group_info["user"],
                            uid=group_info["uid"],
                            group=group_info["group"],
                            gid=group_info["gid"],
                        )
                    )
                    all_groups.add(group_info["group"])

            if device_types["card"]:
                lines.append(f"  • /dev/dri/card*: {len(device_types['card'])} device(s) denied")
                if len(required_groups["card"]) > 1:
                    lines.append("    - Required group(s):")
                else:
                    lines.append("    - Required group:")
                for group_info in required_groups["card"]:
                    lines.append(
                        "      - User: {user} (UID={uid}) | Group: {group} (GID={gid})".format(
                            user=group_info["user"],
                            uid=group_info["uid"],
                            group=group_info["group"],
                            gid=group_info["gid"],
                        )
                    )
                    all_groups.add(group_info["group"])

            # Generate usermod command with all unique groups
            groups_for_usermod = ",".join(sorted(all_groups))

            lines.extend([
                "",
                "To resolve this issue, try the following:",
                "  • Add your user to the required group(s):",
            f"      sudo usermod -aG {groups_for_usermod} \"$USER\"",
                "  • Log out and log back in for the group changes to take effect",
                "  • Alternatively, run this command with sudo/admin privileges",
                ""
            ])
            print("\n".join(lines))
            return False

        return True

    def _severity_as_string(self, error_severity, notify_type, for_filename):
        if error_severity == "non_fatal_uncorrected":
            if(for_filename):
                return "uncorrected"
            return "NONFATAL-UNCORRECTED"
        elif error_severity == "non_fatal_corrected":
            if(for_filename):
                return "corrected"
            return "NONFATAL-CORRECTED"
        elif error_severity == "fatal":
            if notify_type == "BOOT":
                if(for_filename):
                    return "boot"
                return "BOOT"
            if(for_filename):
                return "fatal"
            return "FATAL"
        if(for_filename):
            return "unknown"
        return "UNKNOWN"

    def display_cper_files_generated(self, entries, device_handle, folder, logger=None):
        """
        Display CPER summary lines. If a logger is provided and its destination is
        not stdout, append the output to that file instead of printing to stdout.
        """
        use_file = (
            logger is not None
            and logger.is_human_readable_format()
            and logger.destination != 'stdout'
        )

        # One‐time initialization: warning & header only once
        if not getattr(self, "_cper_display_initialized", False):
            # Warning if no folder was specified elsewhere
            if not getattr(self, "_cper_warning_printed", False):
                warning = (
                    "WARNING: No CPER files will be dumped unless "
                    "--folder=<folder_name> is specified and cper entries exist."
                )
                if use_file:
                    with logger.destination.open('a', encoding="utf-8") as output_file:
                        output_file.write(warning + '\n')
                else:
                    print(warning)
                self._cper_warning_printed = True

            # Print or log the header
            self._print_header(folder, logger if use_file else None)
            self._cper_display_initialized = True

        # Loop through all entries in the dictionary.
        for entry_index, entry in enumerate(entries.values()):
            # Assume 'entry' is a dictionary with keys: "error_severity" and "notify_type".
            timestamp = entry.get("timestamp", "unknown")
            gpu_id = '-'
            if not isinstance(device_handle, Path):
                gpu_id = self.get_gpu_id_from_device_handle(device_handle)
                prefix = self._severity_as_string(
                    entry.get("error_severity", "Unknown"),
                    entry.get("notify_type", "Unknown"),
                    False
                )
                output = f"{timestamp:<20} {gpu_id:<7} {prefix:<20}"

            if folder:
                prefix_for_filename = self._severity_as_string(
                    entry.get("error_severity", "Unknown"),
                    entry.get("notify_type", "Unknown"),
                    True
                )
                cper_data_file = f"{prefix_for_filename}_{self.get_cper_count() + 1}.cper"
                afids = self.cper_dump_afids(cper_data_file)
                afids_str = ' '.join(map(str, afids))
                output += f" {cper_data_file:<17} {afids_str}"

            if use_file:
                with logger.destination.open('a', encoding="utf-8") as output_file:
                    output_file.write(output + '\n')
            else:
                print(output)

            self.increment_cper_count()

    def _print_header(self, folder, logger=None):
        header = f"{'timestamp':<20} {'gpu_id':<7} {'severity':<20}"
        if folder:
            header += f" {'file_name':<17} {'list of afids'}"
        header += ""
        use_file = (
            logger is not None
            and logger.is_human_readable_format()
            and logger.destination != 'stdout'
        )

        if use_file:
            with logger.destination.open('a', encoding="utf-8") as output_file:
                output_file.write(header + '\n')
        else:
            print(header)


    def dump_cper_entries(self, folder, entries, cper_data, device_handle, file_limit=None, cper_file=None):
        """
        Dump CPER entries to files in the specified folder. Handles batch deletion if file limit is exceeded.

        Parameters:
        folder (str): Path to the folder where CPER files will be dumped.
        entries (dict): Dictionary containing CPER entry metadata.
        cper_data (list): List of CPER data objects with 'bytes' and 'size' keys.
        device_handle: Device handle for GPU identification.
        file_limit (int, optional): Maximum number of files to retain in the folder.
        cper_file (str, optional): cper file name to use when saving to folder
        """
        # Initialize header display
        if not getattr(self, "_cper_display_initialized", False):
            self._print_header(folder)
            self._cper_display_initialized = True

        if folder:
            folder = Path(folder)
            folder.mkdir(parents=True, exist_ok=True)

            output_rows = {}

            for entry_index, entry in enumerate(entries.values()):
                # Determine prefix/severity
                error_severity = entry.get("error_severity", "").lower()
                notify_type = entry.get("notify_type", "")
                prefix = self._severity_as_string(error_severity, notify_type, True)

                # Generate filenames
                count = self.get_cper_count() + 1
                if cper_file:
                    cper_name = cper_file
                else:
                   cper_name = f"{prefix}-{count}.cper"
                json_name = f"{prefix}-{count}.json"
                cper_path = folder / cper_name
                json_path = folder / json_name

                # Write CPER binary file
                try:
                    self.write_binary(
                        cper_data[entry_index]["bytes"],
                        cper_data[entry_index]["size"],
                        cper_path
                    )
                except Exception as e:
                    logging.debug(f"Failed to write CPER file {cper_path}: {e}")

                # Write JSON metadata file
                try:
                    with json_path.open("w") as cper_json_file:
                        json.dump(
                            obj=entry,
                            fp=cper_json_file,
                            indent=2,
                            default=lambda o: o.decode('utf-8') if isinstance(o, bytes) else o
                        )
                except Exception as e:
                    logging.debug(f"Failed to write JSON file {json_path}: {e}")

                # Collect data for printing
                timestamp = entry.get("timestamp", "unknown")
                gpu_id = '-'
                if not isinstance(device_handle, Path):
                    gpu_id = self.get_gpu_id_from_device_handle(device_handle)
                severity = self._severity_as_string(error_severity, notify_type, False)
                output_rows[cper_path] = [timestamp, gpu_id, severity, cper_name]
                self.increment_cper_count()

            # Batch deletion if file limit is exceeded (AFTER writing ALL new files)
            if file_limit:
                folder_files = list(sorted(folder.glob("*.cper"), key=lambda p: p.stat().st_mtime))
                if len(folder_files) > file_limit:
                    files_to_delete = len(folder_files) - file_limit
                    for old_file in folder_files[:files_to_delete]:
                        try:
                            old_file.unlink()
                            json_file = old_file.with_suffix('.json')
                            if json_file.exists():
                                json_file.unlink()
                        except OSError as e:
                            logging.debug(f"Failed to delete file {old_file}: {e}")

            # Print collected rows
            for cper_path, row in output_rows.items():
                timestamp, gpu_id, severity, fname = row
                try:
                    afids = self.cper_dump_afids(cper_path)
                    afids_str = ' '.join(map(str, afids))
                except Exception as e:
                    afids_str = "Error fetching AFIDs"
                    logging.debug(f"Failed to fetch AFIDs for {cper_path}: {e}")
                print(f"{timestamp:<20} {gpu_id:<7} {severity:<20} {fname:<17} {afids_str}")

        else:
            # Print entries as JSON if no folder is specified
            try:
                print(json.dumps(
                    entries,
                    indent=2,
                    default=lambda o: o.decode('utf-8') if isinstance(o, bytes) else o
                ))
            except Exception as e:
                logging.debug(f"Failed to dump entries as JSON: {e}")
    
    def dump_cper_entries_as_json(self, entries, _cper_data, _device_handle):
        """
        Return the CPER entries as a formatted JSON string and print it.
        Parameters largely mirror dump_cper_entries so that callers can reuse the same argument list.
        Unused arguments (_cper_data, _device_handle) are retained for API symmetry.
        Returns:
        str: The JSON representation of the CPER entries, or an empty string on failure.
        """
        try:
            entries_json = json.dumps(
                entries,
                indent=2,
                default=lambda o: o.decode("utf-8") if isinstance(o, bytes) else o,
            )
            print(entries_json)
            return entries_json
        except Exception as e:
            logging.debug(f"Failed to serialize CPER entries as JSON: {e}")
            return ""

    def write_binary(self, data, size, filepath):
        """
        Writes binary data directly to a file.

        Parameters:
        data: Either a bytes object or a list of integers representing binary data.
        size (int): The number of bytes to write.
        filepath: The path to the output file.
        """
        with open(filepath, 'wb') as f:
             if isinstance(data, list):
                try:
                    # Attempt to convert the list to a bytes object.
                    data_bytes = bytes(data[:size])
                except ValueError:
                    # If any value is out of range, force them into 0-255.
                    data_bytes = bytes(x % 256 for x in data[:size])
             else:
                 data_bytes = data[:size]
             f.write(data_bytes)

    def binary_to_hexdump_string(self, data: Union[bytes, List[int]]) -> str:
        """
        Convert binary data to a hexdump string.

        Args:
            data: bytes object or list of integer byte values (0–255).

        Returns:
           A multiline string, each line showing:
           offset (in hex), hex bytes (16 per line), and printable ASCII.
        """
        if isinstance(data, bytes):
            data_ints = list(data)
        else:
            # Allow list of ints or single-character strings
            data_ints = []
            for b in data:
                if isinstance(b, int):
                    data_ints.append(b)
                elif isinstance(b, str) and len(b) == 1:
                    data_ints.append(ord(b))
                else:
                    raise ValueError(f"Invalid type in data: {type(b)}")

        lines: List[str] = []
        size = len(data_ints)

        for offset in range(0, size, 16):
            chunk = data_ints[offset : offset + 16]
            hex_values = " ".join(f"{b:02x}" for b in chunk)
            # Pad hex_values to 16*3-1 = 47 chars (two hex digits + space)
            hex_values = hex_values.ljust(16 * 3 - 1)
            ascii_values = "".join(chr(b) if 32 <= b <= 126 else "." for b in chunk)
            lines.append(f"{offset:08x}  {hex_values}  |{ascii_values}|")

        return "\n".join(lines)

    def cper_dump_afids(self, cper_file):
        # 1) Fetch the CPER “file” and ensure we have raw bytes
        raw_data = cper_file
        if hasattr(raw_data, "read"):
            # fetch_cper_file returned a file‐object
            raw = raw_data.read()
        elif isinstance(raw_data, Path):
            # Path: read the bytes directly
            raw = raw_data.read_bytes()
        elif isinstance(raw_data, str):
            # fetch_cper_file returned a filename
            with open(raw_data, "rb") as f:
                    raw = f.read()
        else:
            # assume it's already bytes
            raw = raw_data
        self.binary_to_hexdump_string(raw)
        try:
            afids, num_afids = amdsmi_interface.amdsmi_get_afids_from_cper(raw)
            return afids
        except amdsmi_exception.AmdSmiLibraryException as e:
            if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_INVAL:
                raise ValueError("Invalid CPER file inputs") from e
            elif e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_UNEXPECTED_SIZE:
                raise ValueError("Invalid CPER file data size") from e
            elif e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_UNEXPECTED_DATA:
                raise ValueError("Unexpected data in CPER file") from e
            elif e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED:
                raise NotImplementedError("AFID decoding not supported") from e
            else:
                raise ValueError("Unexpected Error getting afids from CPER file") from e

    def get_partition_id(self, device_handle, gpu_id = None) -> int:
        partition_id = -1
        try:
            kfd_info = amdsmi_interface.amdsmi_get_gpu_kfd_info(device_handle)
            partition_id = kfd_info['current_partition_id']
        except amdsmi_exception.AmdSmiLibraryException as e:
            logging.debug("Failed to get kfd info for gpu %s | %s", gpu_id, e.get_error_info())
        return partition_id

    def get_primary_partition_gpu_id(self, device_handle) -> Union[int, None]:
        try:
            bdf = amdsmi_interface.amdsmi_get_gpu_device_bdf(device_handle)
            if bdf is None:
                logging.debug("Failed to get device BDF: BDF is None")
                return None
            # Construct primary partition BDF (base + ".0" for function 0)
            primary_bdf = bdf[:10] + ".0"
            try:
                primary_device_handle = amdsmi_interface.amdsmi_get_processor_handle_from_bdf(primary_bdf)
                partition_id = self.get_partition_id(primary_device_handle)
                if partition_id == 0:
                    return self.get_gpu_id_from_device_handle(primary_device_handle)
                return None
            except amdsmi_exception.AmdSmiLibraryException as e:
                logging.debug("Failed to get primary partition device handle with BDF %s: %s", primary_bdf, e.get_error_info())
                return None
        except amdsmi_exception.AmdSmiLibraryException as e:
            logging.debug("Failed to get partition device BDF: %s", e.get_error_info())
            return None

    def is_primary_partition(self, device_handle, gpu_id = None) -> bool:
        partition_id = self.get_partition_id(device_handle, gpu_id)
        if partition_id != 0:
            logging.debug(f"Skipping gpu {gpu_id} on non zero partition {partition_id}")
            return False
        return True

    def ras_cper(self, args, device_handle, logger, gpu_idx):
        # Parse severity mask dynamically from the --severity option.
        severity_mask = 0
        # drop duplicates of args
        logging.debug(args)

        for sev in list(set(args.severity)):
            if sev == "all":
                # Set bits for NON_FATAL_UNCORRECTED (0), FATAL (1), and NON_FATAL_CORRECTED (2)
                severity_mask |= ((1 << 0) | (1 << 1) | (1 << 2))
            elif sev == "fatal":
                # Set bit corresponding to AMDSMI_CPER_SEV_FATAL (which is 1)
                severity_mask |= (1 << 1)
            elif sev in ("nonfatal", "nonfatal-uncorrected"):
                # Set bit corresponding to AMDSMI_CPER_SEV_NON_FATAL_UNCORRECTED (which is 0)
                severity_mask |= (1 << 0)
            elif sev in ("nonfatal-corrected", "corrected"):
                # Set bit corresponding to AMDSMI_CPER_SEV_NON_FATAL_CORRECTED (which is 2)
                severity_mask |= (1 << 2)

        buffer_size = 1048576

        # Decide where to send human-readable output
        dest = getattr(logger, "destination", "stdout") if logger is not None else "stdout"
        log_to_file = dest != 'stdout'
        if log_to_file:
            # destination is usually a Path; fall back to Path(string) if needed
            log_path = dest if isinstance(dest, Path) else Path(dest)
        else:
            log_path = None

        if args.decode and args.cper_file:
            device_handle = args.cper_file
        else:
            gpu_id = self.get_gpu_id_from_device_handle(device_handle)
            if args.follow and not getattr(self, "_cper_follow_prompted", False):
                print("Press CTRL + C to stop.")
                self._cper_follow_prompted = True

            primary_partition = self.is_primary_partition(device_handle, gpu_id)
            if not primary_partition:
                return

        if args.folder and not getattr(self, "_cper_folder_prompted", False):
            self._cper_folder_prompted = True

        logger.set_cper_exit_message(False)
        self.stop = False

        num_entries = 0
        entries = {}
        while True:
            try:
                entries, new_cursor, cper_data, status_code = amdsmi_interface.amdsmi_get_gpu_cper_entries(
                    device_handle, severity_mask, buffer_size, args.cursor[gpu_idx])
                logging.debug(f"cper_entries | entries: {entries}")
                num_entries = num_entries + len(entries)
            except amdsmi_exception.AmdSmiLibraryException as e:
                if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                    raise PermissionError('Error opening CPER file. This command requires elevation') from e
                if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED or \
                        e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_FILE_NOT_FOUND:
                    raise FileNotFoundError('Error accessing CPER files. This command requires CPER to be enabled.') from e
                if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_FILE_ERROR:
                    raise FileExistsError('Error opening CPER file. Unable to read CPER File') from e
                else:
                    logging.debug(f"Cannot retrieve CPER entries: {e}")
                    break

            args.cursor[gpu_idx] = new_cursor
            if len(entries) == 0:
                break
            if args.decode and args.cper_file:
                if args.json:
                    self.dump_cper_entries_as_json(entries, cper_data, device_handle)
                elif args.folder:
                    self.dump_cper_entries(args.folder, entries, cper_data, device_handle, args.file_limit)
                else:
                     with tempfile.TemporaryDirectory() as tmp_dir:
                        self.dump_cper_entries(tmp_dir, entries, cper_data, device_handle, args.file_limit, os.path.basename(args.cper_file))
            
            # When a file destination is set, temporarily redirect stdout
            # so that helper print() calls go into that file.
            if log_to_file and log_path is not None:
                orig_stdout = sys.stdout
                try:
                    try:
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                    except Exception:
                        pass
                    with log_path.open('a', encoding='utf-8') as f:
                        sys.stdout = f
                        if args.folder:
                            self.dump_cper_entries(
                                args.folder, entries, cper_data, device_handle, args.file_limit
                            )
                        else:
                            self.display_cper_files_generated(
                                entries, device_handle, args.folder
                            )
                finally:
                    sys.stdout = orig_stdout
            else:
                if args.folder:
                    self.dump_cper_entries(
                        args.folder, entries, cper_data, device_handle, args.file_limit
                    )
                else:
                    self.display_cper_files_generated(
                        entries, device_handle, args.folder
                    )

        if num_entries == 0 and not args.follow:
            # If nothing was found, still emit the warning/header logic
            # using the same redirection logic.
            if log_to_file and log_path is not None:
                orig_stdout = sys.stdout
                try:
                    try:
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                    except Exception:
                        pass
                    with log_path.open('a', encoding='utf-8') as f:
                        sys.stdout = f
                        if args.folder:
                            self.dump_cper_entries(
                                args.folder, entries, cper_data, device_handle, args.file_limit
                            )
                        else:
                            self.display_cper_files_generated(
                                entries, device_handle, args.folder
                            )
                finally:
                    sys.stdout = orig_stdout
            else:
                if args.folder:
                    self.dump_cper_entries(
                        args.folder, entries, cper_data, device_handle, args.file_limit
                    )
                else:
                    self.display_cper_files_generated(
                        entries, device_handle, args.folder
                    )



    def get_bitmask_ranges(self, bitmask_dict):
        ranges = {}
        #start index of the first bitmask
        current_start = 0

        for cpu, bitmask in bitmask_dict.items():
            # Convert the bitmask to a binary string
            binary_str = bin(int(bitmask, 16))[2:].zfill(64)

            binary_str = binary_str[::-1]
            start = 0
            end = len(binary_str) - 1
            # Find the range of set bits
            start_b = binary_str.find('1')
            end_b = binary_str.rfind('1')

            start_setbit = start_b + current_start
            end_setbit = end_b + current_start

            # Calculate the actual bit positions
            end_bit = current_start + end

            # Update the start index for the next bitmask
            current_start = end_bit + 1

            # Store the range in the dictionary
            if start_b == -1 and end_b == -1:
                ranges[cpu] = "N/A"
            else:
                ranges[cpu] = f"{start_setbit}-{end_setbit}"

        return ranges

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
                        elif ("per_" in key) or ("acc_" in key):
                            row[row.index(element)] = element
                    else:
                        continue
            ret = {f"xcp_{i}": violation_status[key][i] for i in range(num_partition)}
        return ret

    @lru_cache(maxsize=1)
    def _get_socket_counts(self):
        """Discover and cache basic topology counts for sockets.

        This helper queries AMDSMI for all socket handles and categorizes them:
            - total_sockets: total number of sockets (CPU + GPU) reported
            - gpu_sockets: number of GPU sockets (identified by BDF-style strings, e.g. '0000:08:00')
            - cpu_sockets: number of CPU sockets (non-BDF style, e.g. '0', '1', ...)

        The result is cached (LRU maxsize=1). If system topology changes
        (e.g. GPUs added/removed), callers must explicitly clear the cache
        via `self._get_socket_counts.cache_clear()`.

        Returns:
            tuple[int, int, int]:
                (total_sockets, gpu_sockets, cpu_sockets)
        """
        gpu_sockets = 0
        cpu_sockets = 0

        try:
            sockets = amdsmi_interface.amdsmi_get_socket_handles()
            for socket in sockets:
                try:
                    info = str(amdsmi_interface.amdsmi_get_socket_info(socket))
                    logging.debug(f"Socket info: {info}")
                    # Check if it contains BDF format: 0000:08:00 -> GPU socket
                    # CPU socket: 0, 1, etc. (does not contain ':')
                    if info.count(":") == 2:
                        gpu_sockets += 1
                    else:
                        cpu_sockets += 1
                except amdsmi_exception.AmdSmiLibraryException as e:
                    logging.debug(f"Failed to get socket info: {e}")
        except amdsmi_exception.AmdSmiLibraryException as e:
            logging.debug(f"Failed to get socket handles: {e}")
            sockets = []

        return (len(sockets), gpu_sockets, cpu_sockets)

    @staticmethod
    def average_flattened_ints(data, context="data"):
        """Calculate the average of flattened integers from a list or tuple
        Args:
            data (list or tuple): Data to calculate the average from
            context (str, optional): Context for logging. Defaults to "data".
        Returns:
            float or str: Average of integers if available, otherwise "N/A"
        """
        # Type validation - ensure data is list or tuple
        # Note: Data can be nested list of lists and will filter out N/A values
        if not isinstance(data, (list, tuple)):
            logging.debug(f"Invalid data type for {context}: expected list/tuple, got {type(data)}")
            return "N/A"

        # Flatten nested lists and filter integers
        flat = [v for value in data for v in (value if isinstance(value, list) else [value]) if isinstance(v, int)]
        return round(sum(flat) / len(flat)) if flat else "N/A"

    def _get_metric_version_and_partition_info(self, gpu_metrics_info, is_partition_metrics, gpu_id, gpu_handle):
        """
        Helper method to compute metric version, partition ID, and num_partition for dynamic metrics.
        Handles logging updates internally for reusability.

        Args:
            gpu_metrics_info (dict): GPU metrics info from amdsmi_get_gpu_metrics_info.
            is_partition_metrics (bool): Whether this is for partition metrics.
            gpu_id (int): GPU ID for logging.
            gpu_handle: GPU device handle for KFD info retrieval.

        Returns:
            dict: {
                'metric_version': float or "N/A",
                'partition_id': int or "N/A",
                'num_partition': int or "N/A",
                'num_xcp': int or "N/A"  # Alias for num_partition
            }
        """
        # Compute metric version from header revisions
        metric_version = "N/A"
        format_rev = gpu_metrics_info.get('common_header.format_revision', "N/A")
        content_rev = gpu_metrics_info.get('common_header.content_revision', "N/A")
        if format_rev != "N/A" and content_rev != "N/A":
            try:
                metric_version = float(f"{format_rev}.{content_rev}")
            except ValueError:
                metric_version = "N/A"  # Fallback if conversion fails

        # Retrieve partition ID from KFD info
        partition_id = "N/A"
        try:
            kfd_info = amdsmi_interface.amdsmi_get_gpu_kfd_info(gpu_handle)
            partition_id = kfd_info.get('current_partition_id', "N/A")
        except amdsmi_exception.AmdSmiLibraryException as e:
            logging.debug("Failed to get current partition ID for GPU %s | %s", gpu_id, e.get_error_info())

        # Determine num_partition with fallback logic for dynamic metrics
        num_partition = gpu_metrics_info.get('num_partition', "N/A")
        if metric_version != "N/A" and num_partition == "N/A":
            # Workaround: Default to 1 for newer metric versions if num_partition is missing
            # (Confirmed with driver team; applies to GPU and partition metrics)
            if not is_partition_metrics and metric_version >= 1.9:
                num_partition = 1
            elif is_partition_metrics and metric_version >= 1.1:
                num_partition = 1
            elif partition_id != "N/A" and partition_id > 0:
                # Fallback to partition_id if partitions exist but num_partition is unavailable
                num_partition = partition_id
            # Else: Remains "N/A" if no conditions match

        # Alias num_xcp for XCP metrics usage
        num_xcp = num_partition

        # Debug logging
        logging.debug(
            "GPU %s | Metric version: %s, num_partition: %s, partition_id: %s, num_xcp: %s",
            gpu_id, metric_version, num_partition, partition_id, num_xcp
        )

        return {
            'metric_version': metric_version,
            'partition_id': partition_id,
            'num_partition': num_partition,
            'num_xcp': num_xcp
        }

    def get_gpu_board_temperatures(self, device_handle, gpu_id, logger):
        """Get GPU board temperature readings

        Args:
            device_handle: GPU device handle
            gpu_id: GPU identifier for logging
            logger: AMDSMILogger instance

        Returns:
            dict: GPU board temperature data or empty dict if all values are N/A
        """
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

        for temp_type in gpu_board_temp_types:
            type_name = temp_type.name.replace("GPUBOARD_", "")
            try:
                gpu_board_temp_holder = amdsmi_interface.amdsmi_get_temp_metric(
                    device_handle, temp_type, amdsmi_interface.AmdSmiTemperatureMetric.CURRENT)
                if gpu_board_temp_holder != "N/A":
                    gpu_board_temp_dict[f'{type_name}'] = self.unit_format(
                        logger, gpu_board_temp_holder, '\N{DEGREE SIGN}C')
                else:
                    gpu_board_temp_dict[f'{type_name}'] = "N/A"
            except amdsmi_exception.AmdSmiLibraryException as e:
                gpu_board_temp_dict[f'{type_name}'] = "N/A"
                logging.debug("Failed to get gpu_board %s for gpu %s | %s",
                            type_name, gpu_id, e.get_error_info())

        return gpu_board_temp_dict

    def get_base_board_temperatures(self, device_handle, gpu_id, logger):
        """Get base board temperature readings

        Args:
            device_handle: GPU device handle
            gpu_id: GPU identifier for logging
            logger: AMDSMILogger instance

        Returns:
            dict: Base board temperature data or empty dict if all values are N/A
        """
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

        for temp_type in base_board_temp_types:
            type_name = temp_type.name.replace("BASEBOARD_", "")
            try:
                base_board_temp_holder = amdsmi_interface.amdsmi_get_temp_metric(
                    device_handle, temp_type, amdsmi_interface.AmdSmiTemperatureMetric.CURRENT)
                if base_board_temp_holder != "N/A":
                    base_board_temp_dict[f'{type_name}'] = self.unit_format(
                        logger, base_board_temp_holder, '\N{DEGREE SIGN}C')
                else:
                    base_board_temp_dict[f'{type_name}'] = "N/A"
            except amdsmi_exception.AmdSmiLibraryException as e:
                base_board_temp_dict[f'{type_name}'] = "N/A"
                logging.debug("Failed to get base_board %s for gpu %s | %s",
                            type_name, gpu_id, e.get_error_info())

        return base_board_temp_dict

    def validate_and_set_power_cap(self, device_handle, power_type, power_type_key, requested_power_cap, logger):
        """Validate and set power cap for a specific sensor.

        Args:
            device_handle: GPU device handle
            power_type: Sensor ID (0 for ppt0, 1 for ppt1)
            power_type_key: Display name for the sensor (e.g., "PPT0")
            requested_power_cap: Requested power cap value in watts
            logger: AMDSMILogger instance for format-aware output

        Returns:
            dict or str: Structured data for JSON/CSV or formatted string for human-readable output
        """
        try:
            power_cap_info = amdsmi_interface.amdsmi_get_power_cap_info(device_handle, power_type)
            gpu_id = self.get_gpu_id_from_device_handle(device_handle)
            logging.debug(f"Power cap info for gpu {gpu_id} {power_type_key} | {power_cap_info}")

            min_power_cap = self.convert_SI_unit(power_cap_info["min_power_cap"], AMDSMIHelpers.SI_Unit.MICRO)
            max_power_cap = self.convert_SI_unit(power_cap_info["max_power_cap"], AMDSMIHelpers.SI_Unit.MICRO)
            current_power_cap = self.convert_SI_unit(power_cap_info["power_cap"], AMDSMIHelpers.SI_Unit.MICRO)

            # Return structured data for JSON/CSV or formatted string for human-readable
            if requested_power_cap == current_power_cap:
                if logger.is_json_format() or logger.is_csv_format():
                    return {
                        "status": "already_set",
                        "sensor": power_type_key,
                        "requested_power_cap": self.unit_format(logger, requested_power_cap, "W"),
                        "current_power_cap": self.unit_format(logger, current_power_cap, "W"),
                        "message": f"{power_type_key} power cap is already set to {requested_power_cap}W"
                    }
                return f"{power_type_key} power cap is already set to {requested_power_cap}W"
            elif current_power_cap == 0:
                if logger.is_json_format() or logger.is_csv_format():
                    return {
                        "status": "error",
                        "sensor": power_type_key,
                        "requested_power_cap": self.unit_format(logger, requested_power_cap, "W"),
                        "current_power_cap": self.unit_format(logger, current_power_cap, "W"),
                        "message": f"Unable to set {power_type_key} power cap to {requested_power_cap}W, current value is {current_power_cap}W"
                    }
                return f"Unable to set {power_type_key} power cap to {requested_power_cap}W, current value is {current_power_cap}W"
            elif not (min_power_cap < requested_power_cap <= max_power_cap and requested_power_cap > 0):
                # setting power cap to 0 will return the current power cap so the technical minimum value is 1
                min_cap_display = 1 if min_power_cap == 0 else min_power_cap
                if logger.is_json_format() or logger.is_csv_format():
                    return {
                        "status": "error",
                        "sensor": power_type_key,
                        "requested_power_cap": self.unit_format(logger, requested_power_cap, "W"),
                        "min_power_cap": self.unit_format(logger, min_cap_display, "W"),
                        "max_power_cap": self.unit_format(logger, max_power_cap, "W"),
                        "message": f"Power cap must be between {min_cap_display}W and {max_power_cap}W"
                    }
                return f"Power cap must be between {min_cap_display}W and {max_power_cap}W"
            # Set the power cap
            new_power_cap = self.convert_SI_unit(requested_power_cap, AMDSMIHelpers.SI_Unit.BASE, AMDSMIHelpers.SI_Unit.MICRO)
            amdsmi_interface.amdsmi_set_power_cap(device_handle, power_type, new_power_cap)
            if logger.is_json_format() or logger.is_csv_format():
                return {
                    "status": "success",
                    "sensor": power_type_key,
                    "power_cap": self.unit_format(logger, requested_power_cap, "W"),
                    "message": f"Successfully set {power_type_key} power cap to {requested_power_cap}W"
                }
            return f"Successfully set {power_type_key} power cap to {requested_power_cap}W"
        except amdsmi_exception.AmdSmiLibraryException as e:
            if e.get_error_code() == amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                raise PermissionError('Command requires elevation') from e
            error_msg = f"[{e.get_error_info(detailed=False)}] Unable to set {power_type_key} power cap to {requested_power_cap}W"
            if logger.is_json_format() or logger.is_csv_format():
                return {
                    "status": "error",
                    "sensor": power_type_key,
                    "requested_power_cap": self.unit_format(logger, requested_power_cap, "W"),
                    "error": e.get_error_info(detailed=False),
                    "message": error_msg
                }
            return error_msg
