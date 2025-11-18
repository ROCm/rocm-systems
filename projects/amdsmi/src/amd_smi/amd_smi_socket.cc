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

#include "amd_smi/impl/amd_smi_socket.h"


namespace amd::smi {

AMDSmiSocket::~AMDSmiSocket() {
    for (uint32_t i = 0; i < processors_.size(); i++) {
        delete processors_[i];
    }
    processors_.clear();
    for (uint32_t i = 0; i < cpu_processors_.size(); i++) {
        delete cpu_processors_[i];
    }
    cpu_processors_.clear();
    for (uint32_t i = 0; i < cpu_core_processors_.size(); i++) {
        delete cpu_core_processors_[i];
    }
    cpu_core_processors_.clear();
}

amdsmi_status_t AMDSmiSocket::get_processor_count(uint32_t* processor_count) const {
    *processor_count = static_cast<uint32_t>(processors_.size());
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiSocket::get_processor_count(processor_type_t type, uint32_t* processor_count) const {
    amdsmi_status_t ret = AMDSMI_STATUS_SUCCESS;
    switch (type) {
    case AMDSMI_PROCESSOR_TYPE_AMD_GPU:
        *processor_count = static_cast<uint32_t>(processors_.size());
        break;
    case AMDSMI_PROCESSOR_TYPE_AMD_CPU:
        *processor_count = static_cast<uint32_t>(cpu_processors_.size());
        break;
    case AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE:
        *processor_count = static_cast<uint32_t>(cpu_core_processors_.size());
        break;
    default:
        *processor_count = 0;
        ret = AMDSMI_STATUS_INVAL;
        break;
    }
    return ret;
}

} // namespace amd::smi


