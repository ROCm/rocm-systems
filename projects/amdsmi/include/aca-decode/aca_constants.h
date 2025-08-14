/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/**
 * @file aca_constants.h
 * @brief Shared constants for ACA error decoding
 *
 * This file contains string constants and numerical constants that are used
 * across multiple source files to improve maintainability and prevent typos.
 */

#ifndef ACA_CONSTANTS_H
#define ACA_CONSTANTS_H

/* Error severity constants */
#define ACA_SEVERITY_UNKNOWN "UNKNOWN"
#define ACA_SEVERITY_FATAL "Fatal"
#define ACA_SEVERITY_CORRECTED "Corrected"
#define ACA_SEVERITY_UNCORRECTED_NON_FATAL "Uncorrected, Non-fatal"
#define ACA_SEVERITY_FAIL_TO_INIT "Fail-to-init"
#define ACA_SEVERITY_ALL_CAPS "ALL"

/* Error category constants */
#define ACA_CATEGORY_HBM_ERRORS "HBM Errors"
#define ACA_CATEGORY_DEVICE_INTERNAL_ERRORS "Device Internal Errors"
#define ACA_CATEGORY_OFF_PACKAGE_LINK_ERRORS "Off-Package Link Errors"
#define ACA_CATEGORY_BOOT_TIME_ERRORS "Boot-Time Errors"
#define ACA_CATEGORY_CPER_FORMAT "CPER Format"
#define ACA_CATEGORY_UNIDENTIFIED_ERRORS "Unidentified Errors"

/* Common error type constants */
#define ACA_ERROR_TYPE_ALL_OTHERS "All Others"
#define ACA_ERROR_TYPE_ALL "All"
#define ACA_ERROR_TYPE_DECODE_INAPPLICABLE "Decode Inapplicable"
#define ACA_ERROR_TYPE_BAD_PAGE_RETIREMENT_THRESHOLD "Bad Page Retirement Threshold"
#define ACA_ERROR_TYPE_HARDWARE_ASSERTION "Hardware Assertion (HWA)"
#define ACA_ERROR_TYPE_WATCHDOG_TIMEOUT "Watchdog Timeout (WDT)"
#define ACA_ERROR_TYPE_ON_DIE_ECC "On-die ECC"
#define ACA_ERROR_TYPE_END_TO_END_CRC "End-to-end CRC"
#define ACA_ERROR_TYPE_WAFL "WAFL"
#define ACA_ERROR_TYPE_XGMI "XGMI"

/* Boot-time error type constants */
#define ACA_ERROR_TYPE_FW_LOAD "FW Load"
#define ACA_ERROR_TYPE_HBM_BIST_TEST "HBM BIST Test"
#define ACA_ERROR_TYPE_HBM_MEMORY_TEST "HBM Memory Test"
#define ACA_ERROR_TYPE_HBM_TRAINING "HBM Training"
#define ACA_ERROR_TYPE_UNHANDLED "Unhandled"
#define ACA_ERROR_TYPE_UNKNOWN_ERROR "Unknown"
#define ACA_ERROR_TYPE_USR_CP_LINK_TRAINING "USR CP Link Training"
#define ACA_ERROR_TYPE_USR_DP_LINK_TRAINING "USR DP Link Training"
#define ACA_ERROR_TYPE_WAFL_LINK_TRAINING "WAFL Link Training"
#define ACA_ERROR_TYPE_XGMI_LINK_TRAINING "XGMI Link Training"
#define ACA_ERROR_TYPE_BOOT_CONTROLLER_DATA_ABORT "Boot Controller Data Abort"
#define ACA_ERROR_TYPE_BOOT_CONTROLLER_GENERIC "Boot Controller Generic"

/* Link error type constants */
#define ACA_ERROR_TYPE_PCIE_AER "PCIe AER"

/* CPER format error type constants */
#define ACA_ERROR_TYPE_MALFORMED_CPER "Malformed CPER"
#define ACA_ERROR_TYPE_INCOMPLETE_ACA_DATA "Incomplete ACA Data"
#define ACA_ERROR_TYPE_INVALID_ACA_DATA "Invalid ACA Data"
#define ACA_ERROR_TYPE_UNIDENTIFIED_ERROR "Unidentified Error"

/* Protocol constants */
#define ACA_PROTOCOL_CPER "CPER"
#define ACA_PROTOCOL_CPER_WITH_SPACE "CPER "

/* Bank name strings */
#define ACA_BANK_UMC "umc"
#define ACA_BANK_PSP "psp"
#define ACA_BANK_CS "cs"
#define ACA_BANK_PIE "pie"
#define ACA_BANK_PCS_XGMI "pcs_xgmi"
#define ACA_BANK_KPX_SERDES "kpx_serdes"
#define ACA_BANK_KPX_WAFL "kpx_wafl"

/* Numerical constants */
#define ACA_FLAG_THRESHOLD_EXCEEDED 0x8
#define ACA_REGISTER_ARRAY_SIZE_32_BYTES 4
#define ACA_REGISTER_ARRAY_SIZE_128_BYTES 16

/* Error code ranges */
#define ACA_ERROR_CODE_EXT_MIN 0x3A
#define ACA_ERROR_CODE_EXT_MAX 0x3E

/* Instance ID values for XCD and AID error decoding */
#define ACA_INSTANCE_ID_XCD0_400 0x36430400
#define ACA_INSTANCE_ID_XCD1_400 0x38430400
#define ACA_INSTANCE_ID_XCD0_401 0x36430401
#define ACA_INSTANCE_ID_XCD1_401 0x38430401
#define ACA_INSTANCE_ID_AID_400 0x3B30400
#define ACA_INSTANCE_ID_AID_401 0x3B30401

/* Error return codes */
#define ACA_ERROR_INVALID_ACA_DATA_ID 33
#define ACA_ERROR_UNIDENTIFIED_ERROR_ID 34

#endif /* ACA_CONSTANTS_H */
