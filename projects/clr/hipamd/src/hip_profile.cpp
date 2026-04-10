/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>

#include "hip_internal.hpp"
#include "profiler/hip_clr_profiler.hpp"

namespace hip {
hipError_t hipProfilerStart() {
  HIP_INIT_API(hipProfilerStart);

  HipProfilerEnableExt();

  HIP_RETURN(hipSuccess);
}


hipError_t hipProfilerStop() {
  HIP_INIT_API(hipProfilerStop);

  HipProfilerDisableExt();

  HIP_RETURN(hipSuccess);
}
}  // namespace hip
