// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "defines.hpp"
#include <hip/hip_runtime.h>
#include <iostream>

// Enhanced HIP error checking with diagnostics
#define HIP_API_CALL_DIAG(CALL, category, diagnostic)                                              \
    {                                                                                              \
        hipError_t error_ = (CALL);                                                                \
        if(error_ != hipSuccess)                                                                   \
        {                                                                                          \
            std::cerr << category << __FILE__ << ":" << __LINE__                                   \
                      << " :: HIP error in " #CALL ": "                                            \
                      << hipGetErrorString(error_) << " (code " << error_ << ")" << std::endl;     \
            if(is_verbose_logging()) {                                                             \
                std::cerr << "Diagnostic: " << diagnostic << std::endl;                            \
            }                                                                                      \
            throw std::runtime_error(std::string(category) + "HIP API call failed");              \
        }                                                                                          \
    }

// Check GPU availability with diagnostic output
inline bool check_hip_device_available(std::ostream& log = std::cerr) {
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);

    if(err != hipSuccess) {
        log << ERROR_TAG_INFRASTRUCTURE << "Failed to query HIP devices: "
            << hipGetErrorString(err) << std::endl;
        log << "Diagnostic steps:" << std::endl;
        log << "  1. Check if ROCm is installed: ls /opt/rocm" << std::endl;
        log << "  2. Verify GPU devices: rocminfo" << std::endl;
        log << "  3. Check amdgpu driver: lsmod | grep amdgpu" << std::endl;
        log << "  4. Verify user permissions: groups | grep render" << std::endl;
        return false;
    }

    if(device_count == 0) {
        log << ERROR_TAG_INFRASTRUCTURE << "No HIP devices found" << std::endl;
        log << "Diagnostic steps:" << std::endl;
        log << "  1. Check GPU visibility: rocminfo" << std::endl;
        log << "  2. Verify ROCR_VISIBLE_DEVICES is not set to hide devices" << std::endl;
        log << "  3. Check if GPU is in use by another process" << std::endl;
        return false;
    }

    return true;
}

// Validate HIP runtime initialization
inline void validate_hip_runtime(const char* test_name) {
    if(!check_hip_device_available()) {
        std::stringstream msg;
        msg << ERROR_TAG_INFRASTRUCTURE << "Test '" << test_name
            << "' requires GPU but none available";
        throw std::runtime_error(msg.str());
    }
}
