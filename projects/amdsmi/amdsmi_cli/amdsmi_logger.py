#!/usr/bin/env python3
#
# Copyright (C) 2023 Advanced Micro Devices. All rights reserved.
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
#

import csv
import json
import re
import time
from typing import Dict
from enum import Enum
import yaml

from amdsmi_helpers import AMDSMIHelpers
import amdsmi_cli_exceptions

class AMDSMILogger():
    def __init__(self, format='human_readable', destination='stdout') -> None:
        self.output = {}
        self.multiple_device_output = []
        self.watch_output = []
        self.format = format # csv, json, or human_readable
        self.destination = destination # stdout, path to a file (append)
        self.table_header = ""
        self.table_title = ""
        self.helpers = AMDSMIHelpers()


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

    def clear_multiple_devices_ouput(self):
        self.multiple_device_output.clear()

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


    def _convert_json_to_human_readable(self, json_object: Dict[str, any], tabular=False):
        # TODO make dynamic
        if tabular:
            table_values = ''
            for key, value in json_object.items():
                value = str(value)
                if key == 'gpu':
                    table_values += value.rjust(3)
                elif key == 'timestamp':
                    table_values += value.rjust(10) + '  '
                elif key == 'power_usage':
                    table_values += value.rjust(7)
                elif key in ('gfx_clock', 'mem_clock', 'encoder_clock', 'decoder_clock', 'vram_used'):
                    table_values += value.rjust(11)
                elif key == 'vram_total' or 'ecc' in key:
                    table_values += value.rjust(12)
                elif key in ('throttle_status', 'pcie_replay'):
                    table_values += value.rjust(13)
                # Only for handling topology tables
                elif 'gpu_' in key:
                    table_values += value.ljust(13)
                # Only for handling xgmi tables
                elif key == "gpu#":
                    table_values += value.ljust(7)
                elif key == "bdf":
                    table_values += value.ljust(13)
                elif "bdf_" in key:
                    table_values += value.ljust(13)
                elif key == "bit_rate":
                    table_values += value.ljust(9)
                elif key == "max_bandwidth":
                    table_values += value.ljust(14)
                elif key == "link_type":
                    table_values += value.ljust(10)
                elif key == "RW":
                    table_values += " " + value.ljust(52)
                # Default spacing
                else:
                    table_values += value.rjust(10)
            return table_values.rstrip()

        # First Capitalize all keys in the json object
        capitalized_json = self._capitalize_keys(json_object)

        # Increase tabbing for device arguments by pulling them out of the main dictionary and assiging them to an empty string
        tabbed_dictionary = {}
        for key, value in capitalized_json.items():
            if key not in ["GPU", "CPU", "CORE"]:
                tabbed_dictionary[key] = value

        for key, value in tabbed_dictionary.items():
            del capitalized_json[key]

        capitalized_json["AMDSMI_SPACING_REMOVAL"] = tabbed_dictionary

        json_string = json.dumps(capitalized_json, indent=4)
        yaml_data = yaml.safe_load(json_string)
        yaml_output = yaml.dump(yaml_data, sort_keys=False, allow_unicode=True)

        # Remove a key line if it is a spacer
        yaml_output = yaml_output.replace("AMDSMI_SPACING_REMOVAL:\n", "")
        yaml_output = yaml_output.replace("'", "") # Remove ''

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
            parent_key (str):
        """
        # print(target_dict)
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


    def print_output(self, multiple_device_enabled=False, watching_output=False, tabular=False):
        """ Print current output acording to format and then destination
            params:
                multiple_device_enabled (bool) - True if printing output from
                    multiple devices
                watching_output (bool) - True if printing watch output
            return:
                Nothing
        """
        if self.is_json_format():
            self._print_json_output(multiple_device_enabled=multiple_device_enabled,
                                    watching_output=watching_output)
        elif self.is_csv_format():
            self._print_csv_output(multiple_device_enabled=multiple_device_enabled,
                                   watching_output=watching_output)
        elif self.is_human_readable_format():
            self._print_human_readable_output(multiple_device_enabled=multiple_device_enabled,
                                              watching_output=watching_output,
                                              tabular=tabular)


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
                with self.destination.open('w') as output_file:
                    json.dump(self.watch_output, output_file, indent=4)
            else:
                with self.destination.open('a') as output_file:
                    json.dump(json_output, output_file, indent=4)


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
                with self.destination.open('w', newline = '') as output_file:
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
                with self.destination.open('a', newline = '') as output_file:
                    # Get the header as a list of the first element to maintain order
                    csv_header = stored_csv_output[0].keys()
                    writer = csv.DictWriter(output_file, csv_header)
                    writer.writeheader()
                    writer.writerows(stored_csv_output)


    def _print_human_readable_output(self, multiple_device_enabled=False, watching_output=False, tabular=False):
        human_readable_output = ''
        if tabular:
            if self.table_title:
                human_readable_output += self.table_title + ':\n'
            human_readable_output += self.table_header + '\n'

        if multiple_device_enabled:
            for output in self.multiple_device_output:
                human_readable_output += self._convert_json_to_human_readable(output, tabular=tabular)
                human_readable_output += '\n'
        else:
            human_readable_output += self._convert_json_to_human_readable(self.output, tabular=tabular)

        if self.destination == 'stdout':
            try:
                # printing as unicode may fail if locale is not set properly
                print(human_readable_output)
            except UnicodeEncodeError:
                # print as ascii, ignore incompatible characters
                print(human_readable_output.encode('ascii', 'ignore').decode('ascii'))
        else:
            if watching_output:
                with self.destination.open('w') as output_file:
                    human_readable_output = ''
                    if tabular:
                        if self.table_title:
                            human_readable_output += self.table_title + '\n'
                        human_readable_output += self.table_header + '\n'
                    for output in self.watch_output:
                        human_readable_output += self._convert_json_to_human_readable(output, tabular=tabular)
                        if tabular:
                            human_readable_output += '\n'
                    output_file.write(human_readable_output + '\n')
            else:
                with self.destination.open('a') as output_file:
                    output_file.write(human_readable_output + '\n')
