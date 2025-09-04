/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "hip/hip_runtime.h"
#include "hip_platform.hpp"

struct ihipModuleSymbol_t;
using hipKernel_t = ihipModuleSymbol_t*;

namespace hip {
hipError_t hipKernelGetAttribute(int* pi, hipFunction_attribute attrib, hipKernel_t kernel,
                                 hipDevice_t dev) {
  HIP_INIT_API(hipKernelGetAttribute, pi, attrib, kernel, dev);
  if (pi == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  const auto* const d_function = hip::DeviceFunc::asFunction(kernel);
  if (d_function == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }
  const auto* const d_kernel = d_function->kernel();
  if (d_kernel == nullptr) {
    HIP_RETURN(hipErrorInvalidDeviceFunction);
  }

  const auto* const device = hip::getCurrentDevice()->devices()[dev];
  const auto* const wrkGrpInfo = d_kernel->getDeviceKernel(*device)->workGroupInfo();
  if (wrkGrpInfo == nullptr) {
    HIP_RETURN(hipErrorMissingConfiguration);
  }

  switch (attrib) {
    case HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
      *pi = static_cast<int>(wrkGrpInfo->size_);
      break;
    case HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES:
      *pi = static_cast<int>(wrkGrpInfo->localMemSize_);
      break;
    case HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES:
      *pi = static_cast<int>(wrkGrpInfo->constMemSize_ - 1);
      break;
    case HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES:
      *pi = static_cast<int>(wrkGrpInfo->privateMemSize_);
      break;
    case HIP_FUNC_ATTRIBUTE_NUM_REGS:
      *pi = static_cast<int>(wrkGrpInfo->usedVGPRs_);
      break;
    case HIP_FUNC_ATTRIBUTE_PTX_VERSION:
    case HIP_FUNC_ATTRIBUTE_BINARY_VERSION:
      *pi = device->isa().versionMajor() * 10 + device->isa().versionMinor();
      break;
    case HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA:
      *pi = 0;
      break;
    case HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES:
      *pi = static_cast<int>(wrkGrpInfo->availableLDSSize_ - wrkGrpInfo->localMemSize_);
      break;
    case HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT:
      *pi = 0;
      break;
    default:
      HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipSuccess);
}
}  // namespace hip
