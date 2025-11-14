#include "Addition.h"

__global__ void VecAddKernel(float *x, float *y, float *z) {
  size_t i = threadIdx.x;
  z[i] = x[i] + y[i];
}

Addition::Addition() {
  CHECK_HIP_ERROR(hipStreamCreate(&stream));

  hipStreamAttrID attr = hipStreamAttributeSynchronizationPolicy;
  hipStreamAttrValue valueToSet;
  hipSynchronizationPolicy syncPolicy =
      hipSynchronizationPolicy::hipSyncPolicyYield;
  valueToSet.syncPolicy = syncPolicy;
  CHECK_HIP_ERROR(hipStreamSetAttribute(stream, attr, &valueToSet));

  hipStreamAttrValue valueOut;
  CHECK_HIP_ERROR(hipStreamGetAttribute(stream, attr, &valueOut));
  std::cout << __FUNCTION__ << " : Stream Sync Policy: " << valueOut.syncPolicy
            << std::endl;
}

Addition::~Addition() { CHECK_HIP_ERROR(hipStreamDestroy(stream)); }

void Addition::LaunchKernelForAddition(float *x, float *y, float *z, int N) {
  float *d_x, *d_y, *d_z;

  CHECK_HIP_ERROR(hipMalloc((void **)&d_x, N * sizeof(float)));
  CHECK_HIP_ERROR(hipMalloc((void **)&d_y, N * sizeof(float)));
  CHECK_HIP_ERROR(hipMalloc((void **)&d_z, N * sizeof(float)));

  CHECK_HIP_ERROR(hipMemcpy(d_x, x, N * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(hipMemcpy(d_y, y, N * sizeof(float), hipMemcpyHostToDevice));

  hipFunction_t function;
  CHECK_HIP_ERROR(
      hipGetFuncBySymbol(&function, reinterpret_cast<void *>(VecAddKernel)));

  ExecuteKernel(reinterpret_cast<hipKernel_t>(function), d_x, d_y, d_z, N,
                stream);

  CHECK_HIP_ERROR(hipMemcpy(z, d_z, N * sizeof(float), hipMemcpyDeviceToHost));

  CHECK_HIP_ERROR(hipFree(d_x));
  CHECK_HIP_ERROR(hipFree(d_y));
  CHECK_HIP_ERROR(hipFree(d_z));
}

void Addition::getStream(hipStream_t &strm) { strm = stream; }

void Addition::customizeStreamAttributes(hipStream_t &strm) {
  CHECK_HIP_ERROR(hipStreamCopyAttributes(stream, strm));

  hipStreamAttrID attr = hipStreamAttributeSynchronizationPolicy;
  hipStreamAttrValue valueOut;
  CHECK_HIP_ERROR(hipStreamGetAttribute(stream, attr, &valueOut));
  std::cout << __FUNCTION__ << " : Stream Sync Policy: " << valueOut.syncPolicy
            << std::endl;
}
