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

#ifndef AMD_SMI_INCLUDE_AMD_SMI_COMMON_H_
#define AMD_SMI_INCLUDE_AMD_SMI_COMMON_H_

#include <map>
#include "amd_smi/amdsmi.h"

#ifdef ENABLE_ESMI_LIB
extern "C" {
    #include <cstdint>
    #include <e_smi/e_smi.h>
}
#endif

namespace amd::smi {

const std::map<unsigned, amdsmi_vram_type_t> vram_type_map = {
    {0, AMDSMI_VRAM_TYPE_UNKNOWN},
    {1, AMDSMI_VRAM_TYPE_GDDR1},
    {2, AMDSMI_VRAM_TYPE_DDR2},
    {3, AMDSMI_VRAM_TYPE_GDDR3},
    {4, AMDSMI_VRAM_TYPE_GDDR4},
    {5, AMDSMI_VRAM_TYPE_GDDR5},
    {6, AMDSMI_VRAM_TYPE_HBM},
    {7, AMDSMI_VRAM_TYPE_DDR3},
    {8, AMDSMI_VRAM_TYPE_DDR4},
    {9, AMDSMI_VRAM_TYPE_GDDR6},
    {10, AMDSMI_VRAM_TYPE_DDR5},
    {11, AMDSMI_VRAM_TYPE_LPDDR4},
    {12, AMDSMI_VRAM_TYPE_LPDDR5},
    {13, AMDSMI_VRAM_TYPE_HBM3E},
};

amdsmi_vram_type_t vram_type_value(unsigned type);

#ifdef ENABLE_ESMI_LIB
// Define a map of esmi status codes to amdsmi status codes
const std::map<esmi_status_t, amdsmi_status_t> esmi_status_map = {
    {ESMI_SUCCESS, AMDSMI_STATUS_SUCCESS},
    {ESMI_INITIALIZED, AMDSMI_STATUS_SUCCESS},
    {ESMI_INVALID_INPUT, AMDSMI_STATUS_INVAL},
    {ESMI_NOT_SUPPORTED, AMDSMI_STATUS_NOT_SUPPORTED},
    {ESMI_PERMISSION, AMDSMI_STATUS_NO_PERM},
    {ESMI_INTERRUPTED, AMDSMI_STATUS_INTERRUPT},
    {ESMI_IO_ERROR, AMDSMI_STATUS_IO},
    {ESMI_FILE_ERROR, AMDSMI_STATUS_FILE_ERROR},
    {ESMI_NO_MEMORY, AMDSMI_STATUS_OUT_OF_RESOURCES},
    {ESMI_DEV_BUSY, AMDSMI_STATUS_BUSY},
    {ESMI_NOT_INITIALIZED, AMDSMI_STATUS_NOT_INIT},
    {ESMI_UNEXPECTED_SIZE, AMDSMI_STATUS_UNEXPECTED_SIZE},
    {ESMI_UNKNOWN_ERROR, AMDSMI_STATUS_UNKNOWN_ERROR},
    {ESMI_NO_ENERGY_DRV, AMDSMI_STATUS_NO_ENERGY_DRV},
    {ESMI_NO_MSR_DRV, AMDSMI_STATUS_NO_MSR_DRV},
    {ESMI_NO_HSMP_DRV, AMDSMI_STATUS_NO_HSMP_DRV},
    {ESMI_NO_HSMP_SUP, AMDSMI_STATUS_NO_HSMP_SUP},
    {ESMI_NO_DRV, AMDSMI_STATUS_NO_DRV},
    {ESMI_FILE_NOT_FOUND, AMDSMI_STATUS_FILE_NOT_FOUND},
    {ESMI_ARG_PTR_NULL, AMDSMI_STATUS_ARG_PTR_NULL},
    {ESMI_HSMP_TIMEOUT, AMDSMI_STATUS_HSMP_TIMEOUT},
    {ESMI_NO_HSMP_MSG_SUP, AMDSMI_STATUS_NO_HSMP_MSG_SUP},
};

amdsmi_status_t esmi_to_amdsmi_status(esmi_status_t status);
#endif

} // namespace amd::smi

#endif  // AMD_SMI_INCLUDE_AMD_SMI_COMMON_H_
