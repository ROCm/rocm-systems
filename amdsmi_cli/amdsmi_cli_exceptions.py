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

import json


AMDSMI_ERROR_MESSAGES = {
    0: "Sucess",
    1: "Invalid parameters",
    2: "Command not supported",
    3: "Command not yet implemented",
    4: "Failed load module",
    5: "Failed load symbole",
    6: "Drm error",
    7: "API call failed",
    8: "Timeout in API call",
    9: "Retry operation",
    10: "Permission Denied",
    11: "Interrupt ocurred during execution",
    12: "I/O Error",
    13: "Address fault",
    14: "Error opening file",
    15: "Not enough memory",
    16: "Internal error",
    17: "Out of bounds",
    18: "Initialization error",
    19: "Internal reference counter exceeded",
    # Reserved for future error messages
    30: "Device busy",
    31: "Device Not found",
    32: "Device not initialized",
    33: "No more free slot",
    34: "Driver not loaded",
    # Reserved for future error messages
    40: "No data was found for given input",
    41: "Insufficient size for operation",
    42: "Unexpected size of data was read",
    43: "The data read or provided was unexpected",
    44: "System has different cpu than AMD",
    45: "Energy driver not found",
    46: "MSR driver not found",
    47: "HSMP driver not found",
    48: "HSMP not supported",
    49: "HSMP message/feature not supported",
    50: "HSMP message timed out",
    51: "No Energy and HSMP driver present",
    52: "File or directory not found",
    53: "Parsed argument is invalid",
    54: "AMDGPU restart error",
    55: "Setting is not available",
    0xFFFFFFFE: "AMD-SMI Library error did not map to a status code",
    0xFFFFFFFF: "Unknown error"
}

def _get_error_message(error_code):
    if abs(error_code) in AMDSMI_ERROR_MESSAGES:
        return AMDSMI_ERROR_MESSAGES[abs(error_code)]
    return "Generic error"


class AmdSmiException(Exception):
    def __init__(self):
        self.json_message = {}
        self.csv_message = ''
        self.stdout_message = ''
        self.message = ''
        self.output_format = ''
        self.device_type = ''

    def __str__(self):
        # Return message according to the current output format
        if self.output_format == 'json':
            self.message = json.dumps(self.json_message)
        elif self.output_format == 'csv':
            self.message = self.csv_message
        else:
            self.message = self.stdout_message

        return self.message


class AmdSmiInvalidCommandException(AmdSmiException):
    def __init__(self, command, outputformat: str, message=None):
        super().__init__()
        self.value = -1
        self.command = command
        self.output_format = outputformat

        common_message = f"Command '{self.command}' is invalid. Run 'amd-smi -h' for more info."

        if message:
            common_message = message

        self.json_message["error"] = common_message
        self.json_message["code"] = self.value
        self.csv_message = f"error,code\n{common_message}, {self.value}"
        self.stdout_message = f"{common_message} Error code: {self.value}"


class AmdSmiInvalidParameterException(AmdSmiException):
    def __init__(self, command, arg, outputformat: str):
        super().__init__()
        self.value = -2
        self.command = command
        self.arg = arg
        self.output_format = outputformat

        common_message = f"Parameter '{self.arg}' is invalid. Run 'amd-smi {self.command} -h' for more info."

        self.json_message["error"] = common_message
        self.json_message["code"] = self.value
        self.csv_message = f"error,code\n{common_message}, {self.value}"
        self.stdout_message = f"{common_message} Error code: {self.value}"


class AmdSmiDeviceNotFoundException(AmdSmiException):
    def __init__(self, command, outputformat: str, gpu: bool, cpu: bool, core: bool):
        super().__init__()
        self.value = -3
        self.command = command
        self.output_format = outputformat

        # Handle different devices
        self.device_type = ""
        if gpu:
            self.device_type = "GPU"
        elif cpu:
            self.device_type = "CPU"
        elif core:
            self.device_type = "CPU CORE"

        common_message = f"Can not find a device: {self.device_type} '{self.command}'"

        self.json_message["error"] = common_message
        self.json_message["code"] = self.value
        self.csv_message = f"error,code\n{common_message}, {self.value}"
        self.stdout_message = f"{common_message} Error code: {self.value}"


class AmdSmiInvalidFilePathException(AmdSmiException):
    def __init__(self, command, outputformat: str, message=None):
        super().__init__()
        self.value = -4
        self.command = command
        self.output_format = outputformat

        common_message = f"Path '{self.command}' cannot be found."

        if message:
            common_message = message

        self.json_message["error"] = common_message
        self.json_message["code"] = self.value
        self.csv_message = f"error,code\n{common_message}, {self.value}"
        self.stdout_message = f"{common_message} Error code: {self.value}"


class AmdSmiInvalidParameterValueException(AmdSmiException):
    def __init__(self, command, arg, outputformat: str):
        super().__init__()
        self.value = -5
        self.command = command
        self.arg = arg
        self.output_format = outputformat

        common_message = f"Value '{self.arg}' is not of valid type or format. Run 'amd-smi {self.command} -h' for more info."

        self.json_message["error"] = common_message
        self.json_message["code"] = self.value
        self.csv_message = f"error,code\n{common_message}, {self.value}"
        self.stdout_message = f"{common_message} Error code: {self.value}"


class AmdSmiMissingParameterValueException(AmdSmiException):
    def __init__(self, command, outputformat: str):
        super().__init__()
        self.value = -6
        self.command = command
        self.output_format = outputformat

        common_message = f"Parameter '{self.command}' requires a value. Run '--help' for more info."

        self.json_message["error"] = common_message
        self.json_message["code"] = self.value
        self.csv_message = f"error,code\n{common_message}, {self.value}"
        self.stdout_message = f"{common_message} Error code: {self.value}"


class AmdSmiCommandNotSupportedException(AmdSmiException):
    def __init__(self, command, outputformat: str):
        super().__init__()
        self.value = -7
        self.command = command
        self.output_format = outputformat

        common_message = f"Command '{self.command}' is not supported on the system. Run '--help' for more info."

        self.json_message["error"] = common_message
        self.json_message["code"] = self.value
        self.csv_message = f"error,code\n{common_message}, {self.value}"
        self.stdout_message = f"{common_message} Error code: {self.value}"


class AmdSmiParameterNotSupportedException(AmdSmiException):
    def __init__(self, command, outputformat: str):
        super().__init__()
        self.value = -8
        self.command = command
        self.output_format = outputformat

        common_message = f"Parameter '{self.command}' is not supported on the system. Run '--help' for more info."

        self.json_message["error"] = common_message
        self.json_message["code"] = self.value
        self.csv_message = f"error,code\n{common_message}, {self.value}"
        self.stdout_message = f"{common_message} Error code: {self.value}"


class AmdSmiRequiredCommandException(AmdSmiException):
    def __init__(self, command, outputformat: str):
        super().__init__()
        self.value = -9
        self.command = command
        self.output_format = outputformat

        common_message = f"Command '{self.command}' requires a target argument. Run 'amd-smi {self.command} -h' for more info."

        self.json_message["error"] = common_message
        self.json_message["code"] = self.value
        self.csv_message = f"error,code\n{common_message}, {self.value}"
        self.stdout_message = f"{common_message} Error code: {self.value}"


class AmdSmiInvalidSubcommandException(AmdSmiException):
    def __init__(self, command, outputformat: str):
        super().__init__()
        self.value = -10
        self.command = command
        self.output_format = outputformat

        common_message = f"AMD-SMI Command '{self.command}' is invalid. Must receive valid AMD-SMI Command first. Run 'amd-smi -h' for more info."

        self.json_message["error"] = common_message
        self.json_message["code"] = self.value
        self.csv_message = f"error,code\n{common_message}, {self.value}"
        self.stdout_message = f"{common_message} Error code: {self.value}"


class AmdSmiPermissionDeniedException(AmdSmiException):
    def __init__(self, command, outputformat: str):
        super().__init__()
        self.value = -11
        self.command = command
        self.output_format = outputformat

        common_message = f"AMD-SMI Command '{self.command}' requires elevation (sudo privileges required)"

        self.json_message["error"] = common_message
        self.json_message["code"] = self.value
        self.csv_message = f"error,code\n{common_message}, {self.value}"
        self.stdout_message = f"{common_message} Error code: {self.value}"


class AmdSmiUnknownErrorException(AmdSmiException):
    def __init__(self, command, outputformat: str):
        super().__init__()
        self.value = -100
        self.command = command
        self.output_format = outputformat

        common_message = "An unknown error has occurred. Run 'help' for more info."

        self.json_message["error"] = common_message
        self.json_message["code"] = self.value
        self.csv_message = f"error,code\n{common_message}, {self.value}"
        self.stdout_message = f"{common_message} Error code: {self.value}"


class AmdSmiLibraryErrorException(AmdSmiException):
    def __init__(self, outputformat: str, error_code):
        super().__init__()
        self.value = -1000 - abs(error_code)
        self.smilibcode = error_code
        self.output_format = outputformat

        common_message = f"AMDSMI has returned error '{self.value}' - '{AMDSMI_ERROR_MESSAGES[abs(self.smilibcode)]}'"

        self.json_message["error"] = common_message
        self.json_message["code"] = self.value
        self.csv_message = f"error,code\n{common_message}, {self.value}"
        self.stdout_message = f"{common_message} Error code: {self.value}"
