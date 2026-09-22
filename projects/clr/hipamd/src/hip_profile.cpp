/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>

#include "hip_internal.hpp"

namespace hip {
hipError_t hipProfilerStart() {
  HIP_INIT_API(hipProfilerStart);
  HIP_RETURN(hipErrorNotSupported);
}


hipError_t hipProfilerStop() {
  HIP_INIT_API(hipProfilerStop);
  HIP_RETURN(hipErrorNotSupported);
}
}  // namespace hip
