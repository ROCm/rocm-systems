// Copyright Advanced Micro Devices, Inc. SPDX-License-Identifier: MIT

// Stand-in for the hip_version.h that the CLR build generates, so the tests can
// parse the in-tree HIP headers without configuring CLR.

#pragma once

#define HIP_VERSION_MAJOR 0
#define HIP_VERSION_MINOR 0
#define HIP_VERSION_PATCH 0
#define HIP_VERSION_GITHASH ""
#define HIP_VERSION_BUILD_ID 0
#define HIP_VERSION_BUILD_NAME ""
#define HIP_VERSION                                                            \
  (HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000 +                 \
   HIP_VERSION_PATCH)
