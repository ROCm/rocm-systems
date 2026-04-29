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

#ifndef AMD_SMI_INCLUDE_IMPL_MOCK_GPU_BACKEND_H_
#define AMD_SMI_INCLUDE_IMPL_MOCK_GPU_BACKEND_H_

#include <string>

#include "amd_smi/impl/mock_data_yaml.h"
#include "amd_smi/impl/not_implemented_gpu_backend.h"

namespace amd::smi {

// Per-backend mock that returns canned values loaded from a YAML file.
// Selected at AMDSmiGPUDevice construction when AMDSMI_MOCK_DATA_FILE points
// at a readable file. Inherits NotImplementedGPUBackend so methods absent
// from the YAML automatically return AMDSMI_STATUS_NOT_SUPPORTED, mirroring
// how real hardware reports unsupported features per SKU.
class MockGPUBackend : public NotImplementedGPUBackend {
 public:
  explicit MockGPUBackend(const std::string& yaml_path);

  amdsmi_status_t get_gpu_metrics_info(amdsmi_gpu_metrics_t* pgpu_metrics) override;

 private:
  MockData data_;
  bool data_valid_;
};

}  // namespace amd::smi

#endif  // AMD_SMI_INCLUDE_IMPL_MOCK_GPU_BACKEND_H_
