#include <hip/hip_runtime.h>
#include <iostream>

#define CHECK_HIP_ERROR(call)                                                  \
  {                                                                            \
    hipError_t err = call;                                                     \
    if (err != hipSuccess) {                                                   \
      std::cerr << " In file " << __FILE__ << " At Line : " << __LINE__        \
                << " HIP error: " << hipGetErrorString(err) << std::endl;      \
      exit(0);                                                                 \
    }                                                                          \
  }

#define REQUIRE(x)                                                             \
  {                                                                            \
    if (x) {                                                                   \
      printf("At line %d : TRUE\n", __LINE__);                                 \
    } else {                                                                   \
      printf("At line %d : FALSE\n", __LINE__);                                \
      return -1;                                                               \
    }                                                                          \
  }

void ExecuteKernel(hipKernel_t function, float *d_x, float *d_y, float *d_z,
                   int N, hipStream_t stream);
