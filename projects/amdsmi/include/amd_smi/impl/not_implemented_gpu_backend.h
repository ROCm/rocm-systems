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

#ifndef AMD_SMI_INCLUDE_IMPL_NOT_IMPLEMENTED_GPU_BACKEND_H_
#define AMD_SMI_INCLUDE_IMPL_NOT_IMPLEMENTED_GPU_BACKEND_H_

#include "amd_smi/impl/i_gpu_backend.h"

namespace amd::smi {

// Default-NOT_SUPPORTED implementation of IGPUBackend.
//
// Partial backends (e.g. MockGPUBackend) inherit from this class and override
// only the methods they actually implement. Methods left unoverridden return
// AMDSMI_STATUS_NOT_SUPPORTED, mirroring how real hardware reports unsupported
// features per SKU. Adding a new method to IGPUBackend requires a default impl
// here so existing partial backends keep compiling.
class NotImplementedGPUBackend : public IGPUBackend {
 public:
  amdsmi_status_t get_gpu_metrics_info(amdsmi_gpu_metrics_t* /*pgpu_metrics*/) override {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
};

}  // namespace amd::smi

#endif  // AMD_SMI_INCLUDE_IMPL_NOT_IMPLEMENTED_GPU_BACKEND_H_
