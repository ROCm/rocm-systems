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

  assert(0 && "Unimplemented");

  HIP_RETURN(hipErrorNotSupported);
}


hipError_t hipProfilerStop() {
  HIP_INIT_API(hipProfilerStop);

  assert(0 && "Unimplemented");

  HIP_RETURN(hipErrorNotSupported);
}
}  // namespace hip
