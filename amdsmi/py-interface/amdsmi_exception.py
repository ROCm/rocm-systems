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

from enum import IntEnum
from . import amdsmi_wrapper


class AmdSmiException(Exception):
    """Base smi exception class"""
    pass


class AmdSmiLibraryException(AmdSmiException):
    def __init__(self, err_code):
        err_code = abs(err_code)
        super().__init__(err_code)
        self.err_code = err_code
        self.set_err_info()

    def __str__(self):
        return "Error code:\n\t{err_code} | {err_info}".format(
            err_code=self.err_code, err_info=self.err_info
        )

    def get_error_info(self):
        return self.err_info

    def get_error_code(self):
        return self.err_code

    # Translate error codes to error strings
    def set_err_info(self):
        switch = {
            amdsmi_wrapper.AMDSMI_STATUS_INVAL : "AMDSMI_STATUS_INVAL - Invalid parameters",
            amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED : "AMDSMI_STATUS_NOT_SUPPORTED - Feature not supported",
            amdsmi_wrapper.AMDSMI_STATUS_NOT_YET_IMPLEMENTED : "AMDSMI_STATUS_NOT_YET_IMPLEMENTED - Feature not yet implemented",
            amdsmi_wrapper.AMDSMI_STATUS_FAIL_LOAD_MODULE : "AMDSMI_STATUS_FAIL_LOAD_MODULE - Fail to load lib",
            amdsmi_wrapper.AMDSMI_STATUS_FAIL_LOAD_SYMBOL : "AMDSMI_STATUS_FAIL_LOAD_SYMBOL - Fail to load symbol",
            amdsmi_wrapper.AMDSMI_STATUS_DRM_ERROR : "AMDSMI_STATUS_DRM_ERROR - Error when called libdrm",
            amdsmi_wrapper.AMDSMI_STATUS_API_FAILED : "AMDSMI_STATUS_API_FAILED - API call failed",
            amdsmi_wrapper.AMDSMI_STATUS_TIMEOUT : "AMDSMI_STATUS_TIMEOUT - Timeout in API call",
            amdsmi_wrapper.AMDSMI_STATUS_RETRY : "AMDSMI_STATUS_RETRY - Retry operation",
            amdsmi_wrapper.AMDSMI_STATUS_NO_PERM : "AMDSMI_STATUS_NO_PERM - Permission Denied",
            amdsmi_wrapper.AMDSMI_STATUS_INTERRUPT : "AMDSMI_STATUS_INTERRUPT - Interrupt ocurred during execution",
            amdsmi_wrapper.AMDSMI_STATUS_IO : "AMDSMI_STATUS_IO - I/O Error",
            amdsmi_wrapper.AMDSMI_STATUS_ADDRESS_FAULT : "AMDSMI_STATUS_ADDRESS_FAULT - Bad address",
            amdsmi_wrapper.AMDSMI_STATUS_FILE_ERROR : "AMDSMI_STATUS_FILE_ERROR - Error opening file",
            amdsmi_wrapper.AMDSMI_STATUS_OUT_OF_RESOURCES : "AMDSMI_STATUS_OUT_OF_RESOURCES - Not enough memory",
            amdsmi_wrapper.AMDSMI_STATUS_INTERNAL_EXCEPTION : "AMDSMI_STATUS_INTERNAL_EXCEPTION -  Internal error",
            amdsmi_wrapper.AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS : "AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS - Out of bounds",
            amdsmi_wrapper.AMDSMI_STATUS_INIT_ERROR : "AMDSMI_STATUS_INIT_ERROR - Initialization error",
            amdsmi_wrapper.AMDSMI_STATUS_REFCOUNT_OVERFLOW : "AMDSMI_STATUS_REFCOUNT_OVERFLOW - Internal reference counter exceeded INT32_MAX",
            amdsmi_wrapper.AMDSMI_STATUS_BUSY : "AMDSMI_STATUS_BUSY - Device busy",
            amdsmi_wrapper.AMDSMI_STATUS_NOT_FOUND : "AMDSMI_STATUS_NOT_FOUND - Device Not found",
            amdsmi_wrapper.AMDSMI_STATUS_NOT_INIT : "AMDSMI_STATUS_NOT_INIT - Device not initialized",
            amdsmi_wrapper.AMDSMI_STATUS_NO_SLOT : "AMDSMI_STATUS_NO_SLOT - No more free slot",
            amdsmi_wrapper.AMDSMI_STATUS_DRIVER_NOT_LOADED : "AMDSMI_STATUS_DRIVER_NOT_LOADED - Driver not loaded",
            amdsmi_wrapper.AMDSMI_STATUS_NO_DATA : "AMDSMI_STATUS_NO_DATA - No data was found for given input",
            amdsmi_wrapper.AMDSMI_STATUS_INSUFFICIENT_SIZE : "AMDSMI_STATUS_INSUFFICIENT_SIZE - Insufficient size for operation",
            amdsmi_wrapper.AMDSMI_STATUS_UNEXPECTED_SIZE : "AMDSMI_STATUS_UNEXPECTED_SIZE - unexpected size of data was read",
            amdsmi_wrapper.AMDSMI_STATUS_UNEXPECTED_DATA : "AMDSMI_STATUS_UNEXPECTED_DATA - The data read or provided was unexpected",
            amdsmi_wrapper.AMDSMI_STATUS_NON_AMD_CPU : "AMDSMI_STATUS_NON_AMD_CPU - System has non-AMD CPU",
            amdsmi_wrapper.AMDSMI_STATUS_NO_ENERGY_DRV : "AMD_SMI_NO_ENERGY_DRV - Energy driver not found",
            amdsmi_wrapper.AMDSMI_STATUS_NO_MSR_DRV : "AMDSMI_STATUS_NO_MSR_DRV - MSR driver not found",
            amdsmi_wrapper.AMDSMI_STATUS_NO_HSMP_DRV : "AMD_SMI_NO_HSMP_DRV - HSMP driver not found",
            amdsmi_wrapper.AMDSMI_STATUS_NO_HSMP_SUP : "AMD_SMI_NO_HSMP_SUP - HSMP not supported",
            amdsmi_wrapper.AMDSMI_STATUS_NO_HSMP_MSG_SUP : "AMD_SMI_NO_HSMP_MSG_SUP - HSMP message/feature not supported",
            amdsmi_wrapper.AMDSMI_STATUS_HSMP_TIMEOUT : "AMD_SMI_HSMP_TIMEOUT - HSMP message timeout",
            amdsmi_wrapper.AMDSMI_STATUS_NO_DRV : "AMDSMI_STATUS_NO_DRV - No Energy and HSMP driver present",
            amdsmi_wrapper.AMDSMI_STATUS_FILE_NOT_FOUND : "AMDSMI_STATUS_FILE_NOT_FOUND - File or directory not found",
            amdsmi_wrapper.AMDSMI_STATUS_ARG_PTR_NULL : "AMDSMI_STATUS_ARG_PTR_NULL - Parsed argument is invalid",
            amdsmi_wrapper.AMDSMI_STATUS_MAP_ERROR : "AMDSMI_STATUS_MAP_ERROR - The internal library error did not map to a status code",
            amdsmi_wrapper.AMDSMI_STATUS_AMDGPU_RESTART_ERR:  "AMDSMI_STATUS_AMDGPU_RESTART_ERR - AMDGPU restart failed, please check dmsg for errors",
            amdsmi_wrapper.AMDSMI_STATUS_SETTING_UNAVAILABLE:  "AMDSMI_STATUS_SETTING_UNAVAILABLE - Setting is not available",
            amdsmi_wrapper.AMDSMI_STATUS_CORRUPTED_EEPROM:  "AMDSMI_STATUS_CORRUPTED_EEPROM - Setting is not available",
            amdsmi_wrapper.AMDSMI_STATUS_UNKNOWN_ERROR : "AMDSMI_STATUS_UNKNOWN_ERROR - An unknown error occurred"
        }

        self.err_info = switch.get(self.err_code, "AMDSMI_STATUS_UNKNOWN_ERROR - An unknown error occurred")


class AmdSmiRetryException(AmdSmiLibraryException):
    def __init__(self):
        super().__init__(amdsmi_wrapper.AMDSMI_STATUS_RETRY)


class AmdSmiTimeoutException(AmdSmiLibraryException):
    def __init__(self):
        super().__init__(amdsmi_wrapper.AMDSMI_STATUS_TIMEOUT)


class AmdSmiParameterException(AmdSmiException):
    def __init__(self, receivedValue, expectedType, msg=None):
        super().__init__(msg)
        self.err_code = None
        self.actualType = type(receivedValue)
        self.expectedType = expectedType
        self.set_err_msg()
        if msg is not None:
            self.err_msg = msg

    def set_err_msg(self):
        self.err_msg = (
            "Invalid parameter:\n"
            + "Actual type: {actualType}\n".format(actualType=self.actualType)
            + "Expected type: {expectedType}".format(expectedType=self.expectedType)
        )

    def __str__(self):
        return self.err_msg


class AmdSmiKeyException(AmdSmiException):
    def __init__(self, key):
        super().__init__()
        self.key = key
        self.set_err_msg()

    def set_err_msg(self):
        self.err_msg = "Key " + self.key + " is missing from dictionary"

    def __str__(self):
        return self.err_msg


class AmdSmiBdfFormatException(AmdSmiException):
    def __init__(self, bdf):
        super().__init__()
        self.bdf = bdf

    def __str__(self):
        return (
            "Wrong BDF format: {}. \n"
            + "BDF string should be: <domain>:<bus>:<device>.<function>\n"
            + " or <bus>:<device>.<function> in hexcode format.\n"
            + "Where:\n\t<domain> is 4 hex digits long from 0000-FFFF interval\n"
            + "\t<bus> is 2 hex digits long from 00-FF interval\n"
            + "\t<device> is 2 hex digits long from 00-1F interval\n"
            + "\t<function> is 1 hex digit long from 0-7 interval"
        ).format(self.bdf)
