// Stub cuda.h — redirects to our HIP stub for CPU-only RCCL unit tests.
// In the real RCCL build, hipify converts cuda.h includes to hip/hip_runtime.h.
#pragma once
#include <hip/hip_runtime.h>
