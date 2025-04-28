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

import csv
import json
import re
import time
from typing import Dict
from enum import Enum
import inspect
from amdsmi_helpers import AMDSMIHelpers
import amdsmi_cli_exceptions

class AMDSMILogger():
    def __init__(self, format='human_readable', destination='stdout') -> None:
        self.output = {}
        self.multiple_device_output = []
        self.watch_output = []
        self.format = format # csv, json, or human_readable
        self.destination = destination # stdout, path to a file (append)
        self.table_title = ""
        self.table_header = ""
        self.secondary_table_title = ""
        self.secondary_table_header = ""
        self.warning_message = ""
        self.helpers = AMDSMIHelpers()
        self._cper_exit_message = True
        self.store_cpu_json_output = []
        self.store_core_json_output = []
        self.store_gpu_json_output = []


    class LoggerFormat(Enum):
        """Enum for logger formats"""
        json = 'json'
        csv = 'csv'
        human_readable = 'human_readable'


    class CsvStdoutBuilder(object):
        def __init__(self):
            self.csv_string = []

        def write(self, row):
            self.csv_string.append(row)

        def __str__(self):
            return ''.join(self.csv_string)


    def is_json_format(self):
        return self.format == self.LoggerFormat.json.value


    def is_csv_format(self):
        return self.format == self.LoggerFormat.csv.value


    def is_human_readable_format(self):
        return self.format == self.LoggerFormat.human_readable.value


    def clear_multiple_devices_output(self):
        self.multiple_device_output.clear()


    def cper_exit_message(self):
        """ Store the cper exit message
            params:
                message (str) - message to store
            return:
                cper_exit_message (bool) - True if cper exit message is set
        """
        return self._cper_exit_message


    def set_cper_exit_message(self, flag:bool):
        """ Set the cper exit message
            params:
                flag (bool) - True if cper exit message is set
            return:
                Nothing
        """
        self._cper_exit_message = flag


    def _capitalize_keys(self, input_dict):
        output_dict = {}
        for key in input_dict.keys():
            # Capitalize key if it is a string
            if isinstance(key, str):
                cap_key = key.upper()
            else:
                cap_key = key

            if isinstance(input_dict[key], dict):
                output_dict[cap_key] = self._capitalize_keys(input_dict[key])
            elif isinstance(input_dict[key], list):
                cap_key_list = []
                for data in input_dict[key]:
                    if isinstance(data, dict):
                        cap_key_list.append(self._capitalize_keys(data))
                    else:
                        cap_key_list.append(data)
                output_dict[cap_key] = cap_key_list
            else:
                output_dict[cap_key] = input_dict[key]

        return output_dict


    def _convert_json_to_tabular(self, json_object: Dict[str, any], dynamic=False):
        # TODO make dynamic - convert other python CLI outputs to use (as needed)
        # Update: using dynamic=true provides dynamic re-sizing based on key name length

        table_values = ''
        stored_gpu = ''
        stored_timestamp = ''
        for key, value in json_object.items():
            string_value = str(value)
            if key == 'partition_id':
                # Special case for partition_id: 8 partitions + 7 comma + 2 spaces = 17
                table_values += string_value.ljust(17)
                continue
            key_length = len(key) + 2
            if dynamic and len(key) > 0:
                stored_gpu = string_value
                table_values += string_value.ljust(key_length)
            elif key == 'gpu':
                stored_gpu = string_value
                table_values += string_value.rjust(3)
            elif key == 'timestamp':
                stored_timestamp = string_value
                table_values += string_value.rjust(10) + '  '
            elif key == 'power_usage':
                table_values += string_value.rjust(7)
            elif key == 'max_power':
                table_values += string_value.rjust(9)
            elif key in ('hotspot_temperature', 'memory_temperature'):
                table_values += string_value.rjust(8)
            elif key in ('gfx', 'mem'):
                table_values += string_value.rjust(7)
            elif key in ('gfx_clk'):
                table_values += string_value.rjust(10)
            elif key in ('mem_clock', 'vram_used'):
                table_values += string_value.rjust(11)
            elif key in ('vram_total', 'vram_free'):
                table_values += string_value.rjust(12)
            elif key == 'vram_percent':
                table_values += string_value.rjust(9)
            elif key in ('encoder', 'decoder'):
                table_values += string_value.rjust(7)
            elif key in ('vclock', 'dclock'):
                table_values += string_value.rjust(10)
            elif key in ('single_bit_ecc', 'double_bit_ecc', 'pcie_bw'):
                table_values += string_value.rjust(12)
            elif key in ('pcie_replay'):
                table_values += string_value.rjust(13)
            # Only for handling topology tables
            elif 'gpu_' in key:
                table_values += string_value.ljust(13)
            # Only for handling xgmi tables
            elif key == "gpu#":
                table_values += string_value.ljust(7)
            elif key == "bdf":
                table_values += string_value.ljust(14)
            elif "bdf_" in key:
                table_values += string_value.ljust(13)
            elif key == "bit_rate":
                table_values += string_value.ljust(10)
            elif key == "max_bandwidth":
                table_values += string_value.ljust(15)
            elif key == "link_type":
                table_values += string_value.ljust(11)
            elif key == "link_status":
                for i in value:
                    table_values += str(i).ljust(3)
            elif key == "RW":
                table_values += string_value.ljust(57)
            elif key in ('pviol', 'tviol'):
                table_values += string_value.rjust(7)
            elif key == "tviol_active":
                table_values += string_value.rjust(14)
            elif key == "phot_tviol":
                table_values += string_value.rjust(12)
            elif key == "vr_tviol":
                table_values += string_value.rjust(10)
            elif key == "hbm_tviol":
                table_values += string_value.rjust(11)
            elif key == "gfx_clkviol":
                table_values += string_value.rjust(13)
            elif key == "process_list":
                #Add an additional padding between the first instance of GPU and NAME
                table_values += '  '
                for process_dict in value:
                    if process_dict['process_info'] == "No running processes detected":
                        # Add N/A for empty process_info
                        table_values += "N/A".rjust(20) + "N/A".rjust(9) + "N/A".rjust(10) + \
                                        "N/A".rjust(10) + "N/A".rjust(10) + "N/A".rjust(11) + \
                                        "N/A".rjust(8) + "N/A".rjust(8) + '\n'
                    else:
                        for process_key, process_value in process_dict['process_info'].items():
                            string_process_value = str(process_value)
                            if process_key == "name":
                                # Truncate name if too long
                                process_name = string_process_value[:20]
                                if process_name == "":
                                    process_name = "N/A"
                                table_values += process_name.rjust(20)
                            elif process_key == "pid":
                                table_values += string_process_value.rjust(9)
                            elif process_key == "memory_usage":
                                for memory_key, memory_value in process_value.items():
                                    table_values += str(memory_value).rjust(10)
                            elif process_key == "mem_usage":
                                table_values += string_process_value.rjust(11)
                            elif process_key == "usage":
                                for usage_key, usage_value in process_value.items():
                                    table_values += str(usage_value).rjust(8)
                                # Add the stored gpu and stored timestamp to the next line
                                table_values += '\n'
                                if stored_timestamp:
                                    table_values += stored_timestamp.ljust(10) + '  '
                                table_values += stored_gpu.rjust(3) + '  '

                # Remove excess two values after a new line in table_values
                table_values = table_values[:table_values.rfind('\n')]
                table_values += '\n'
            # Default spacing
            else:
                table_values += string_value.rjust(10)
        return table_values.rstrip()


    def _convert_json_to_human_readable(self, json_object: Dict[str, any]):
        # First Capitalize all keys in the json object
        capitalized_json = self._capitalize_keys(json_object)

        # Increase tabbing for device arguments by pulling them out of the main dictionary and assiging them to an empty string
        tabbed_dictionary = {}
        for key, value in capitalized_json.items():
            if key not in ["GPU", "CPU", "CORE"]:
                tabbed_dictionary[key] = value
            # Filter out N/A values under clock
            if key == "CLOCK":
                valid_clock_data = {}
                if isinstance(value, dict):  # Ensure value is a dictionary
                    for clock_key, clock_data in value.items():
                        if isinstance(clock_data, dict):  # Ensure clock_data is a dictionary
                            non_na = {
                                clock_key: clock_value
                                for clock_key, clock_value in clock_data.items()
                                if clock_value != "N/A"
                            }
                            if non_na:
                                valid_clock_data[clock_key] = non_na
                        elif clock_data != "N/A":  # Handle single-tier clock_data
                            valid_clock_data[clock_key] = clock_data
                elif value != "N/A":  # Handle non-dictionary clock data
                    valid_clock_data = value
                tabbed_dictionary[key] = valid_clock_data

        for key, value in tabbed_dictionary.items():
            del capitalized_json[key]

        capitalized_json["AMDSMI_SPACING_REMOVAL"] = tabbed_dictionary

        # Convert the capitalized JSON to a YAML-like string
        yaml_output = self.custom_dump(capitalized_json)

        # Remove a key line if it is a spacer
        yaml_output = yaml_output.replace("AMDSMI_SPACING_REMOVAL:\n", "")
        yaml_output = yaml_output.replace("'", "") # Remove ''

        # Remove process_info indicies for Host parity:
        yaml_output = re.sub(r'PROCESS_INFO_[0-9]+:', 'PROCESS_INFO:', yaml_output)

        clean_yaml_output = ''
        for line in yaml_output.splitlines():
            line = line.split(':')

            # Remove dashes and increase tabbing split key
            line[0] = line[0].replace("-", " ", 1)
            line[0] = line[0].replace("  ", "    ")

            # Join cleaned output
            line = ':'.join(line) + '\n'
            clean_yaml_output += line

        return clean_yaml_output

    def custom_dump(self, data, indent=0):
        """Converts a Python dictionary to a YAML-like string."""
        yaml_string = ""
        for key, value in data.items():
            if isinstance(value, dict):
                yaml_string += "  " * indent + f"{key}:\n" + self.custom_dump(value, indent + 1)
            elif isinstance(value, list):
                yaml_string += "  " * indent + f"{key}:\n"
                for item in value:
                    if isinstance(item, dict):
                        yaml_string += self.custom_dump(item, indent + 1)
                    else: # If the list is not a dictionary, print it as a string
                        yaml_string += "  " * (indent + 1) + f"- {item}\n"
            else:
                yaml_string += "  " * indent + f"{key}: {value}\n"
        return yaml_string

    def flatten_dict(self, target_dict, topology_override=False):
        """This will flatten a dictionary out to a single level of key value stores
            removing key's with dictionaries and wrapping each value to in a list
            ex:
                {
                    'usage': {
                        'gfx_usage': 0,
                        'mem_usage': 0,
                        'mm_usage_list': [22,0,0]
                    }
                }
            to:
                {
                    'gfx_usage': 0,
                    'mem_usage': 0,
                    'mm_usage_list': [22,0,0]}
                }

        Args:
            target_dict (dict): Dictionary to flatten
        """
        output_dict = {}
        # First flatten out values

        # separetly handle ras and process and firmware

        # If there are multi values, and the values are all dicts
        # Then flatten the sub values with parent key
        for key, value in target_dict.items():
            if isinstance(value, dict):
                # Check number of items in the dict
                if len(value.values()) > 1 or topology_override:
                    value_with_parent_key = {}
                    for parent_key, child_dict in value.items():
                        if isinstance(child_dict, dict):
                            if parent_key in ('gfx'):
                                for child_key, value1 in child_dict.items():
                                    value_with_parent_key[child_key] = value1
                            else:
                                for child_key, value1 in child_dict.items():
                                    value_with_parent_key[parent_key + '_' + child_key] = value1
                        else:
                            if topology_override:
                                value_with_parent_key[key + '_' + parent_key] = child_dict
                            else:
                                value_with_parent_key[parent_key] = child_dict
                    value = value_with_parent_key

                output_dict.update(self.flatten_dict(value).items())
            else:
                output_dict[key] = value
        return output_dict


    def store_output(self, device_handle, argument, data):
        """ Convert device handle to gpu id and store output
            params:
                device_handle - device handle object to the target device output
                argument (str) - key to store data
                data (dict | list) - Data store against argument
            return:
                Nothing
        """
        gpu_id = self.helpers.get_gpu_id_from_device_handle(device_handle)
        self._store_output_amdsmi(gpu_id=gpu_id, argument=argument, data=data)


    def store_cpu_output(self, device_handle, argument, data):
        """ Convert device handle to cpu id and store output
            params:
                device_handle - device handle object to the target device output
                argument (str) - key to store data
                data (dict | list) - Data store against argument
            return:
                Nothing
        """
        cpu_id = self.helpers.get_cpu_id_from_device_handle(device_handle)
        self._store_cpu_output_amdsmi(cpu_id=cpu_id, argument=argument, data=data)


    def store_core_output(self, device_handle, argument, data):
        """ Convert device handle to core id and store output
            params:
                device_handle - device handle object to the target device output
                argument (str) - key to store data
                data (dict | list) - Data store against argument
            return:
                Nothing
        """
        core_id = self.helpers.get_core_id_from_device_handle(device_handle)
        self._store_core_output_amdsmi(core_id=core_id, argument=argument, data=data)


    def _store_core_output_amdsmi(self, core_id, argument, data):
        if argument == 'timestamp': # Make sure timestamp is the first element in the output
            self.output['timestamp'] = int(time.time())

        if self.is_json_format() or self.is_human_readable_format():
            self.output['core'] = int(core_id)
            if argument == 'values' and isinstance(data, dict):
                self.output.update(data)
            else:
                self.output[argument] = data
        elif self.is_csv_format():
            self.output['core'] = int(core_id)

            if argument == 'values' or isinstance(data, dict):
                flat_dict = self.flatten_dict(data)
                self.output.update(flat_dict)
            else:
                self.output[argument] = data
        else:
            raise amdsmi_cli_exceptions(self, "Invalid output format given, only json, csv, and human_readable supported")


    def _store_cpu_output_amdsmi(self, cpu_id, argument, data):
        if argument == 'timestamp': # Make sure timestamp is the first element in the output
            self.output['timestamp'] = int(time.time())

        if self.is_json_format() or self.is_human_readable_format():
            self.output['cpu'] = int(cpu_id)
            if argument == 'values' and isinstance(data, dict):
                self.output.update(data)
            else:
                self.output[argument] = data
        elif self.is_csv_format():
            self.output['cpu'] = int(cpu_id)

            if argument == 'values' or isinstance(data, dict):
                flat_dict = self.flatten_dict(data)
                self.output.update(flat_dict)
            else:
                self.output[argument] = data
        else:
            raise amdsmi_cli_exceptions(self, "Invalid output format given, only json, csv, and human_readable supported")


    def _store_output_amdsmi(self, gpu_id, argument, data):
        if argument == 'timestamp': # Make sure timestamp is the first element in the output
            self.output['timestamp'] = int(time.time())

        if self.is_json_format() or self.is_human_readable_format():
            self.output['gpu'] = int(gpu_id)
            if argument == 'values' and isinstance(data, dict):
                self.output.update(data)
            else:
                self.output[argument] = data
        elif self.is_csv_format():
            self.output['gpu'] = int(gpu_id)

            if argument == 'values' or isinstance(data, dict):
                flat_dict = self.flatten_dict(data)
                self.output.update(flat_dict)
            else:
                self.output[argument] = data
        else:
            raise amdsmi_cli_exceptions(self, "Invalid output format given, only json, csv, and human_readable supported")


    def _store_output_rocmsmi(self, gpu_id, argument, data):
        if self.is_json_format():
            # put output into self.json_output
            pass
        elif self.is_csv_format():
            # put output into self.csv_output
            pass
        elif self.is_human_readable_format():
            # put output into self.human_readable_output
            pass
        else:
            raise amdsmi_cli_exceptions(self, "Invalid output format given, only json, csv, and human_readable supported")


    def store_multiple_device_output(self):
        """ Store the current output into the multiple_device_output
                then clear the current output
            params:
                None
            return:
                Nothing
        """
        if not self.output:
            return
        output = {}
        for key, value in self.output.items():
            output[key] = value

        self.multiple_device_output.append(output)
        self.output = {}


    def store_watch_output(self, multiple_device_enabled=False):
        """ Add the current output or multiple_devices_output
            params:
                multiple_device_enabled (bool) - True if watching multiple devices
            return:
                Nothing
        """
        if multiple_device_enabled:
            for output in self.multiple_device_output:
                self.watch_output.append(output)

            self.multiple_device_output = []
        else:
            output = {}

            for key, value in self.output.items():
                output[key] = value
            self.watch_output.append(output)

            self.output = {}


    def print_output(self, multiple_device_enabled=False, watching_output=False, tabular=False, dual_csv_output=False, dynamic=False):
        """ Print current output acording to format and then destination
            params:
                multiple_device_enabled (bool) - True if printing output from
                    multiple devices
                watching_output (bool) - True if printing watch output
                dynamic (bool) - Defaults to False. True turns on dynamic resizing for
                    left justified table output
            return:
                Nothing
        """
        if self.is_json_format():
            self._print_json_output(multiple_device_enabled=multiple_device_enabled,
                                    watching_output=watching_output)
        elif self.is_csv_format():
            if dual_csv_output:
                self._print_dual_csv_output(multiple_device_enabled=multiple_device_enabled,
                                             watching_output=watching_output)
            else:
                self._print_csv_output(multiple_device_enabled=multiple_device_enabled,
                                        watching_output=watching_output)
        elif self.is_human_readable_format():
            # If tabular output is enabled, redirect to _print_tabular_output
            if tabular:
                self._print_tabular_output(multiple_device_enabled=multiple_device_enabled, watching_output=watching_output, dynamic=dynamic)
            else:
                self._print_human_readable_output(multiple_device_enabled=multiple_device_enabled,
                                                   watching_output=watching_output)


    def _print_json_output(self, multiple_device_enabled=False, watching_output=False):
        if multiple_device_enabled:
            json_output = self.multiple_device_output
        else:
            json_output = [self.output]

        if self.destination == 'stdout':
            if json_output:
                json_std_output = json.dumps(json_output, indent=4)
                print(json_std_output)
        else: # Write output to file
            if watching_output: # Flush the full JSON output to the file on watch command completion
                with self.destination.open('w', encoding="utf-8") as output_file:
                    json.dump(self.watch_output, output_file, indent=4)
            else:
                with self.destination.open('a', encoding="utf-8") as output_file:
                    json.dump(json_output, output_file, indent=4)


    def combine_arrays_to_json(self):
        combined_json = {
            "cpu_data": self.store_cpu_json_output,
            "core_data": self.store_core_json_output,
            "gpu_data": self.store_gpu_json_output
        }
        self.destination == 'stdout'
        json_std_output = json.dumps(combined_json, indent=4)
        print(json_std_output)


    def _print_csv_output(self, multiple_device_enabled=False, watching_output=False):
        if multiple_device_enabled:
            stored_csv_output = self.multiple_device_output
        else:
            if not isinstance(self.output, list):
                stored_csv_output = [self.output]

        if stored_csv_output:
            csv_keys = set()
            for output in stored_csv_output:
                for key in output:
                    csv_keys.add(key)

            for index, output_dict in enumerate(stored_csv_output):
                remaining_keys = csv_keys - set(output_dict.keys())
                for key in remaining_keys:
                    stored_csv_output[index][key] = "N/A"

        if self.destination == 'stdout':
            if stored_csv_output:
                # Get the header as a list of the first element to maintain order
                csv_header = stored_csv_output[0].keys()
                csv_stdout_output = self.CsvStdoutBuilder()
                writer = csv.DictWriter(csv_stdout_output, csv_header)
                writer.writeheader()
                writer.writerows(stored_csv_output)
                print(str(csv_stdout_output))
        else:
            if watching_output:
                with self.destination.open('w', newline = '', encoding="utf-8") as output_file:
                    if self.watch_output:
                        csv_keys = set()
                        for output in self.watch_output:
                            for key in output:
                                csv_keys.add(key)

                        for index, output_dict in enumerate(self.watch_output):
                            remaining_keys = csv_keys - set(output_dict.keys())
                            for key in remaining_keys:
                                self.watch_output[index][key] = "N/A"

                        # Get the header as a list of the first element to maintain order
                        csv_header = self.watch_output[0].keys()
                        writer = csv.DictWriter(output_file, csv_header)
                        writer.writeheader()
                        writer.writerows(self.watch_output)
            else:
                with self.destination.open('a', newline = '', encoding="utf-8") as output_file:
                    # Get the header as a list of the first element to maintain order
                    csv_header = stored_csv_output[0].keys()
                    writer = csv.DictWriter(output_file, csv_header)
                    writer.writeheader()
                    writer.writerows(stored_csv_output)


    def _print_dual_csv_output(self, multiple_device_enabled=False, watching_output=False):
        if multiple_device_enabled:
            stored_csv_output = self.multiple_device_output
        else:
            if not isinstance(self.output, list):
                stored_csv_output = [self.output]

        primary_csv_output = []
        secondary_csv_output = []

        if stored_csv_output:
            # Split stored_csv_output into primary_csv and secondary_csv
            for output_dict in stored_csv_output:
                if 'process_list' in output_dict:
                    # Add a new entry for each process_info
                    for process_info_dict in output_dict['process_list']:
                        secondary_output_dict = {}
                        if watching_output:
                            secondary_output_dict['timestamp'] = output_dict['timestamp']
                        secondary_output_dict['gpu'] = output_dict['gpu']
                        if isinstance(process_info_dict["process_info"], dict):
                            for process_field, process_value in process_info_dict["process_info"].items():
                                if isinstance(process_value, dict):
                                    for key, value in process_value.items():
                                        secondary_output_dict[key] = value
                                else:
                                    secondary_output_dict[process_field] = process_value
                        else:
                            # Handle no process found case
                            secondary_output_dict["process_info"] = process_info_dict["process_info"]
                        secondary_csv_output.append(secondary_output_dict)
                primary_output_dict = {}
                for key, value in output_dict.items():
                    if key != 'process_list':
                        primary_output_dict[key] = value
                primary_csv_output.append(primary_output_dict)

        # Ensure uniform data within primary and secondary csv outputs
        if primary_csv_output:
            primary_keys = set()
            for output in primary_csv_output:
                for key in output:
                    primary_keys.add(key)
            # insert empty data to align with keys that may not exist
            for index, output_dict in enumerate(primary_csv_output):
                remaining_keys = primary_keys - set(output_dict.keys())
                for key in remaining_keys:
                    primary_csv_output[index][key] = "N/A"
        if secondary_csv_output:
            secondary_keys = set()
            for output in secondary_csv_output:
                for key in output:
                    secondary_keys.add(key)
            # insert empty data to align with keys that may not exist
            for index, output_dict in enumerate(secondary_csv_output):
                remaining_keys = secondary_keys - set(output_dict.keys())
                for key in remaining_keys:
                    secondary_csv_output[index][key] = "N/A"

        if self.destination == 'stdout':
            if primary_csv_output:
                # Get the header as a list of the first element to maintain order
                csv_header = primary_csv_output[0].keys()
                csv_stdout_output = self.CsvStdoutBuilder()
                writer = csv.DictWriter(csv_stdout_output, csv_header)
                writer.writeheader()
                writer.writerows(primary_csv_output)
                print(str(csv_stdout_output))
            if secondary_csv_output:
                # Get the header as a list of the first element to maintain order
                csv_header = secondary_csv_output[0].keys()
                csv_stdout_output = self.CsvStdoutBuilder()
                writer = csv.DictWriter(csv_stdout_output, csv_header)
                writer.writeheader()
                writer.writerows(secondary_csv_output)
                print(str(csv_stdout_output))
                if watching_output:
                    print()
        else:
            if watching_output:
                with self.destination.open('w', newline = '', encoding="utf-8") as output_file:
                    primary_csv_output = []
                    secondary_csv_output = []
                    if self.watch_output:
                        # Split watch_output into primary_csv and secondary_csv
                        for output_dict in self.watch_output:
                            if 'process_list' in output_dict:
                                # Add a new entry for each process_info
                                for process_info_dict in output_dict['process_list']:
                                    secondary_output_dict = {}
                                    if watching_output:
                                        secondary_output_dict['timestamp'] = output_dict['timestamp']
                                    secondary_output_dict['gpu'] = output_dict['gpu']
                                    if isinstance(process_info_dict["process_info"], dict):
                                        for process_field, process_value in process_info_dict["process_info"].items():
                                            if isinstance(process_value, dict):
                                                for key, value in process_value.items():
                                                    secondary_output_dict[key] = value
                                            else:
                                                secondary_output_dict[process_field] = process_value
                                    else:
                                        # Handle no process found case
                                        secondary_output_dict["process_info"] = process_info_dict["process_info"]
                                    secondary_csv_output.append(secondary_output_dict)
                            primary_output_dict = {}
                            for key, value in output_dict.items():
                                if key != 'process_list':
                                    primary_output_dict[key] = value
                            primary_csv_output.append(primary_output_dict)

                        # Ensure uniform data within primary and secondary csv outputs
                        if primary_csv_output:
                            primary_keys = set()
                            for output in primary_csv_output:
                                for key in output:
                                    primary_keys.add(key)
                            # insert empty data to align with keys that may not exist
                            for index, output_dict in enumerate(primary_csv_output):
                                remaining_keys = primary_keys - set(output_dict.keys())
                                for key in remaining_keys:
                                    primary_csv_output[index][key] = "N/A"
                        if secondary_csv_output:
                            secondary_keys = set()
                            for output in secondary_csv_output:
                                for key in output:
                                    secondary_keys.add(key)
                            # insert empty data to align with keys that may not exist
                            for index, output_dict in enumerate(secondary_csv_output):
                                remaining_keys = secondary_keys - set(output_dict.keys())
                                for key in remaining_keys:
                                    secondary_csv_output[index][key] = "N/A"

                        if primary_csv_output:
                            # Get the header as a list of the first element to maintain order
                            csv_header = primary_csv_output[0].keys()
                            writer = csv.DictWriter(output_file, csv_header)
                            writer.writeheader()
                            writer.writerows(primary_csv_output)
                        if secondary_csv_output:
                            output_file.write("\n")
                            csv_header = secondary_csv_output[0].keys()
                            writer = csv.DictWriter(output_file, csv_header)
                            writer.writeheader()
                            writer.writerows(secondary_csv_output)
            else:
                with self.destination.open('a', newline = '', encoding="utf-8") as output_file:
                    if primary_csv_output:
                        # Get the header as a list of the first element to maintain order
                        csv_header = primary_csv_output[0].keys()
                        writer = csv.DictWriter(output_file, csv_header)
                        writer.writeheader()
                        writer.writerows(primary_csv_output)
                    if secondary_csv_output:
                        output_file.write("\n")
                        csv_header = secondary_csv_output[0].keys()
                        writer = csv.DictWriter(output_file, csv_header)
                        writer.writeheader()
                        writer.writerows(secondary_csv_output)

    def _print_human_readable_output(self, multiple_device_enabled=False, watching_output=False, tabular=False):
        # If tabular output is enabled, redirect to _print_tabular_output
        if tabular:
            self._print_tabular_output(multiple_device_enabled=multiple_device_enabled, watching_output=watching_output)
            return

        human_readable_output = ''

        if multiple_device_enabled:
            for device_output in self.multiple_device_output:
                human_readable_output += self._convert_json_to_human_readable(device_output) + '\n'
        else:
            human_readable_output += self._convert_json_to_human_readable(self.output)

        if self.destination == 'stdout':
            try:
                # printing as unicode may fail if locale is not set properly
                print(human_readable_output)
            except UnicodeEncodeError:
                # print as ascii, ignore incompatible characters
                print(human_readable_output.encode('ascii', 'ignore').decode('ascii'))
        else:
            if watching_output:
                with self.destination.open('w', encoding="utf-8") as output_file:
                    human_readable_output = ''
                    for output in self.watch_output:
                        human_readable_output += self._convert_json_to_human_readable(output)
                    output_file.write(human_readable_output + '\n')
            else:
                with self.destination.open('a', encoding="utf-8") as output_file:
                    output_file.write(human_readable_output + '\n')


    def _print_tabular_output(self, multiple_device_enabled=False, watching_output=False, dynamic=False):
        primary_table = ''
        secondary_table = ''

        # Populate primary table without process_list
        # Populate secondary table with process_list if exists
        if multiple_device_enabled and self.multiple_device_output:
            for device_output in self.multiple_device_output:
                if 'process_list' in device_output:
                    process_table_dict = {}
                    if watching_output:
                        process_table_dict['timestamp'] = device_output['timestamp']
                    process_table_dict['gpu'] = device_output['gpu']
                    process_table_dict['process_list'] = device_output['process_list']
                    secondary_table += self._convert_json_to_tabular(process_table_dict) + '\n'
                # Add primary table keys without process_list
                primary_table_output = {}
                for key, value in device_output.items():
                    if key != 'process_list':
                        primary_table_output[key] = value
                primary_table += self._convert_json_to_tabular(primary_table_output, dynamic=dynamic) + '\n'
        else: # Single device output
            if 'process_list' in self.output:
                process_table_dict = {}
                if watching_output:
                    process_table_dict['timestamp'] = self.output['timestamp']
                process_table_dict['gpu'] = self.output['gpu']
                process_table_dict['process_list'] = self.output['process_list']
                secondary_table += self._convert_json_to_tabular(process_table_dict) + '\n'
            # Add primary table keys without process_list
            primary_table_output = {}
            for key, value in self.output.items():
                if key != 'process_list':
                    primary_table_output[key] = value
            primary_table += self._convert_json_to_tabular(primary_table_output, dynamic=dynamic) + '\n'
        primary_table = primary_table.rstrip()
        secondary_table = secondary_table.rstrip()

        # Add primary table title and header to primary_table
        if primary_table:
            primary_table_heading = ''
            if self.table_title:
                primary_table_heading = self.table_title + ':\n'
            if self.warning_message:  # Add warning message below the table title
                primary_table_heading += self.warning_message + '\n'
            primary_table_heading += self.table_header + '\n'
            primary_table = primary_table_heading + primary_table

        # Add secondary table title and header to secondary_table
        # Currently just process_info uses this logic
        if secondary_table:
            secondary_table_heading = ''
            if self.secondary_table_title:
                secondary_table_heading = '\n' + self.secondary_table_title + ':\n'
            secondary_table_heading += self.secondary_table_header + '\n'
            secondary_table = secondary_table_heading + secondary_table

        if self.destination == 'stdout':
            try:
                # printing as unicode may fail if locale is not set properly
                print(primary_table)
                if secondary_table:
                    print(secondary_table)
                if watching_output:
                    print("\n")
            except UnicodeEncodeError:
                # print as ascii, ignore incompatible characters
                print(primary_table.encode('ascii', 'ignore').decode('ascii'))
                if secondary_table:
                    print(secondary_table.encode('ascii', 'ignore').decode('ascii'))
                if watching_output:
                    print("\n")
        else:
            if watching_output: # Write all stored watched output to a file
                with self.destination.open('w', encoding="utf-8") as output_file:
                    primary_table = ''
                    secondary_table = ''
                    # Add process_list to the secondary_table
                    # Add remaining watch_output to the primary_table
                    for device_output in self.watch_output:
                        # if process_list is detected in device_output store in secondary_table
                        if 'process_list' in device_output:
                            process_table_dict = {
                                'timestamp': device_output['timestamp'],
                                'gpu': device_output['gpu'],
                                'process_list': device_output['process_list']
                            }
                            secondary_table += self._convert_json_to_tabular(process_table_dict) + '\n'
                        # Add primary table keys without process_list
                        primary_table_output = {}
                        for key, value in device_output.items():
                            if key != 'process_list':
                                primary_table_output[key] = value
                        primary_table += self._convert_json_to_tabular(primary_table_output, dynamic=dynamic) + '\n'
                    primary_table = primary_table.rstrip() # Remove trailing new line
                    secondary_table = secondary_table.rstrip()

                    # Add primary table title and header to primary_table
                    if primary_table:
                        primary_table_heading = ''
                        if self.table_title:
                            primary_table_heading = self.table_title + ':\n'
                        if self.warning_message: # Add warning message below the table title
                            primary_table_heading += self.warning_message + '\n'
                        primary_table_heading += self.table_header + '\n'
                        primary_table = primary_table_heading + primary_table

                    # Add secondary table title and header to secondary_table
                    # Currently just process_info uses this logic
                    if secondary_table:
                        secondary_table_heading = ''
                        if self.secondary_table_title:
                            secondary_table_heading = '\n' + self.secondary_table_title + ':\n'
                        secondary_table_heading += self.secondary_table_header + '\n'
                        secondary_table = secondary_table_heading + secondary_table

                    # Write both full tables to the file
                    output_file.write(primary_table)
                    if secondary_table:
                        output_file.write("\n" + secondary_table)
            else: # Write all singular output to a file
                with self.destination.open('a', encoding="utf-8") as output_file:
                    output_file.write(primary_table + '\n')
                    output_file.write(secondary_table)
