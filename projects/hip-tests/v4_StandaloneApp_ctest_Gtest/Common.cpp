#include "Common.h"

void ExecuteKernel(hipKernel_t function, float *d_x, float *d_y, float *d_z,
                   int N, hipStream_t stream) {
  ::std::cout << "ExecuteKernel ..." << std::endl;

  void *args1[] = {&d_x, &d_y, &d_z};
  CHECK_HIP_ERROR(hipLaunchKernel(function, 1, N, args1, 0, stream));
  CHECK_HIP_ERROR(hipStreamSynchronize(stream));
}
