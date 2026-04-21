// kernels.hip — minimal kernel for benchmark dispatch overhead.
#include <hip/hip_runtime.h>

extern "C" __global__ void noop() { }
