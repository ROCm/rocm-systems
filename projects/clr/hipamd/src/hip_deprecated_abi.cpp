/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// These symbols are no longer provided as APIs, but they are supported as part
// of the ABI. If the application insists on using them, they need to call the
// R0000 version

#include <cstdint>

// We name this file .cpp, so that we can use CXXFLAGS.
extern "C" {

typedef int hipDevice_t;
typedef struct hipDeviceProp_tR0000 hipDeviceProp_tR0000;
typedef uint32_t hipError_t;

// This is a duplicate of hip_internal.h. We cannot include anything in this
// file so that we don't pollute the ABI with API versioning macros.

#if defined(__GNUC__) || defined(__clang__)
#define HIP_PUBLIC_API __attribute__((visibility("default")))
#else
#define HIP_PUBLIC_API
#endif

HIP_PUBLIC_API hipError_t hipGetDevicePropertiesR0000(hipDeviceProp_tR0000* props, hipDevice_t device);
HIP_PUBLIC_API hipError_t hipChooseDeviceR0000(int* device, const hipDeviceProp_tR0000* properties);

HIP_PUBLIC_API hipError_t hipGetDeviceProperties(hipDeviceProp_tR0000* props, hipDevice_t device) {
  return hipGetDevicePropertiesR0000(props, device);
}

HIP_PUBLIC_API hipError_t hipChooseDevice(int* device, const hipDeviceProp_tR0000* properties) {
  return hipChooseDeviceR0000(device, properties);
}

}
