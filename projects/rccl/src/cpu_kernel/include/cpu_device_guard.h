/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 ************************************************************************/

#ifndef RCCL_CPU_DEVICE_GUARD_H_
#define RCCL_CPU_DEVICE_GUARD_H_

#include <hip/hip_runtime.h>

struct rcclCpuDeviceGuard {
  int device;
  bool active;
  rcclCpuDeviceGuard(int cudaDev) : device(-1), active(false) {
    if (cudaDev < 0) return;
    if (hipGetDevice(&device) != hipSuccess) device = -1;
    if (device != cudaDev) hipSetDevice(cudaDev);
    active = true;
  }
  ~rcclCpuDeviceGuard() {
    if (active && device >= 0) hipSetDevice(device);
  }
};

#endif
