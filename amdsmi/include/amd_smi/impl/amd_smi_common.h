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


} // namespace amd::smi

#endif  // AMD_SMI_INCLUDE_AMD_SMI_COMMON_H_
