// SPDX-License-Identifier: MIT
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

#include "error_map.h"
#include <string.h>

#define AFID_VERSION "0.7"

static const error_map_entry_t error_map[] = {
    {1, "Boot-Time Errors", "FW Load", "CPER", "Fail-to-init"},
    {2, "Boot-Time Errors", "HBM BIST Test", "CPER", "Fail-to-init"},
    {3, "Boot-Time Errors", "HBM Memory Test", "CPER", "Fail-to-init"},
    {4, "Boot-Time Errors", "HBM Training", "CPER", "Fail-to-init"},
    {5, "Boot-Time Errors", "Unhandled", "CPER", "Fail-to-init"},
    {6, "Boot-Time Errors", "Unknown", "CPER", "Fail-to-init"},
    {7, "Boot-Time Errors", "USR CP Link Training", "CPER", "Fail-to-init"},
    {8, "Boot-Time Errors", "USR DP Link Training", "CPER", "Fail-to-init"},
    {9, "Boot-Time Errors", "WAFL Link Training", "CPER", "Fail-to-init"},
    {10, "Boot-Time Errors", "XGMI Link Training", "CPER", "Fail-to-init"},
    {11, "Boot-Time Errors", "Boot Controller Data Abort", "CPER", "Fail-to-init"},
    {12, "Boot-Time Errors", "Boot Controller Generic", "CPER ", "Fail-to-init"},
    {13, "Off-Package Link Errors", "PCIe AER", "CPER", "Corrected"},
    {14, "Off-Package Link Errors", "PCIe AER", "CPER", "Fatal"},
    {15, "Off-Package Link Errors", "WAFL", "CPER", "Corrected"},
    {16, "Off-Package Link Errors", "WAFL", "CPER", "Fatal"},
    {17, "Off-Package Link Errors", "XGMI", "CPER", "Corrected"},
    {18, "Off-Package Link Errors", "XGMI", "CPER", "Fatal"},
    {19, "HBM Errors", "Bad Page Retirement Threshold", "CPER", "Fatal"},
    {20, "HBM Errors", "On-die ECC", "CPER", "Fatal"},
    {21, "HBM Errors", "End-to-end CRC", "CPER", "Fatal"},
    {22, "HBM Errors", "On-die ECC", "CPER", "Uncorrected, Non-fatal"},
    {23, "HBM Errors", "End-to-end CRC", "CPER", "Uncorrected, Non-fatal"},
    {24, "HBM Errors", "All", "CPER", "Corrected"},
    {25, "HBM Errors", "All Others", "CPER", "Fatal"},
    {26, "Device Internal Errors", "Hardware Assertion (HWA)", "CPER", "Fatal"},
    {27, "Device Internal Errors", "Watchdog Timeout (WDT)", "CPER", "Fatal"},
    {28, "Device Internal Errors", "All Others", "CPER", "Uncorrected, Non-fatal"},
    {29, "Device Internal Errors", "All Others", "CPER", "Corrected"},
    {30, "Device Internal Errors", "All Others", "CPER", "Fatal"},
    {31, "CPER Format", "Malformed CPER", "CPER", "ALL"},
    {32, "CPER Format", "Incomplete ACA Data", "CPER", "ALL"},
    {33, "CPER Format", "Invalid ACA Data", "CPER", "ALL"},
    {34, "Unidentified Errors", "Unidentified Error", "CPER", "ALL"}};

static const size_t NUM_ERROR_ENTRIES = sizeof(error_map) / sizeof(error_map[0]);

int get_error_id(const char *error_category, const char *error_type, const char *error_severity)
{
    if (!error_category || !error_type || !error_severity ||
        strcmp(error_category, "UNKNOWN") == 0 ||
        strcmp(error_type, "UNKNOWN") == 0 ||
        strcmp(error_severity, "UNKNOWN") == 0)
    {
        return 33; // Return ID for "Invalid Error" if any input is "UNKNOWN" or NULL
    }

    for (size_t i = 0; i < NUM_ERROR_ENTRIES; i++)
    {
        if (strcmp(error_map[i].error_category, error_category) == 0 &&
            strcmp(error_map[i].error_type, error_type) == 0 &&
            strcmp(error_map[i].error_severity, error_severity) == 0)
        {
            return (int)error_map[i].id;
        }
    }

    return 34; // Return ID for "Unidentified Errors" if no match found
}
