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

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_common.h"



namespace amd::smi {

amdsmi_status_t rsmi_to_amdsmi_status(rsmi_status_t status) {
    amdsmi_status_t amdsmi_status = AMDSMI_STATUS_MAP_ERROR;

    // Look for it in the map
    // If found: use the mapped value
    // If not found: return the map error established above
    auto search = amd::smi::rsmi_status_map.find(status);
    if (search != amd::smi::rsmi_status_map.end()) {
        amdsmi_status = search->second;
    }

    return amdsmi_status;
}

amdsmi_vram_type_t vram_type_value(unsigned type) {
    amdsmi_vram_type_t value = AMDSMI_VRAM_TYPE_UNKNOWN;

    auto search = amd::smi::vram_type_map.find(type);
    if (search != amd::smi::vram_type_map.end()) {
        value = search->second;
    }

    return value;
}


#ifdef ENABLE_ESMI_LIB
amdsmi_status_t esmi_to_amdsmi_status(esmi_status_t status) {
    amdsmi_status_t amdsmi_status = AMDSMI_STATUS_MAP_ERROR;

    // Look for it in the map
    // If found: use the mapped value
    // If not found: return the map error established above
    auto search = amd::smi::esmi_status_map.find(status);
    if (search != amd::smi::esmi_status_map.end()) {
        amdsmi_status = search->second;
    }

    return amdsmi_status;
}
#endif

} // namespace amd::smi

