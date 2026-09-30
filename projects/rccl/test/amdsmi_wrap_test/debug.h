#ifndef DEBUG_H
#define DEBUG_H
#include "nccl.h"
#include "core.h"

static inline ncclResult_t rcclCudaErrorHandler(hipError_t err) {

  // Print the cuda error
  ERROR("HIP failure: '%s'", hipGetErrorString(err));

  // Special error message here:
  switch (err) {
    case hipErrorStreamCaptureInvalidated:
      ERROR("Application is trying to use an invalidated stream to launch RCCL kernel. "
      "This operation is invalid. RCCL is exiting.");
      break;
    default:
      break;
    }
    return ncclUnhandledCudaError;
  }

  #endif