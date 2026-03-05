#pragma once

#include <hip/hip_runtime.h>

inline hipError_t GetSmResourceDesc(hipDevResourceDesc_t* desc) {
  hipDevice_t device;
  hipDevResource resource{};

  hipError_t ret = hipDeviceGet(&device, 0);
  if (ret != hipSuccess) {
    return ret;
  }

  ret = hipDeviceGetDevResource(device, &resource, hipDevResourceTypeSm);
  if (ret != hipSuccess) {
    return ret;
  }

  return hipDevResourceGenerateDesc(desc, &resource, 1);
}
