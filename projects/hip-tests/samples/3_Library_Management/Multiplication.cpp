#include "Multiplication.h"

Multiplication::Multiplication() {
  CHECK_HIP_ERROR(hipStreamCreate(&stream));

  hipStreamAttrID attr = hipStreamAttributeSynchronizationPolicy;
  hipStreamAttrValue valueToSet;
  hipSynchronizationPolicy syncPolicy =
      hipSynchronizationPolicy::hipSyncPolicySpin;
  valueToSet.syncPolicy = syncPolicy;
  CHECK_HIP_ERROR(hipStreamSetAttribute(stream, attr, &valueToSet));

  hipStreamAttrValue valueOut;
  CHECK_HIP_ERROR(hipStreamGetAttribute(stream, attr, &valueOut));
  std::cout << __FUNCTION__ << " : Stream Sync Policy: " << valueOut.syncPolicy
            << std::endl;
}

Multiplication::~Multiplication() { CHECK_HIP_ERROR(hipStreamDestroy(stream)); }

void Multiplication::LaunchKernelForMultiplication(float *x, float *y, float *z,
                                                   int N) {
  float *d_x, *d_y, *d_z;

  CHECK_HIP_ERROR(hipMalloc((void **)&d_x, N * sizeof(float)));
  CHECK_HIP_ERROR(hipMalloc((void **)&d_y, N * sizeof(float)));
  CHECK_HIP_ERROR(hipMalloc((void **)&d_z, N * sizeof(float)));

  CHECK_HIP_ERROR(hipMemcpy(d_x, x, N * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(hipMemcpy(d_y, y, N * sizeof(float), hipMemcpyHostToDevice));

  hipLibrary_t library;
  std::string libFile = "MultiplyKernel.code";
  CHECK_HIP_ERROR(hipLibraryLoadFromFile(&library, libFile.data(), nullptr,
                                         nullptr, 0, nullptr, nullptr, 0));

  hipKernel_t function;
  CHECK_HIP_ERROR(hipLibraryGetKernel(&function, library, "MultiplyKernel"));

  ExecuteKernel(function, d_x, d_y, d_z, N, stream);

  CHECK_HIP_ERROR(hipMemcpy(z, d_z, N * sizeof(float), hipMemcpyDeviceToHost));

  CHECK_HIP_ERROR(hipFree(d_x));
  CHECK_HIP_ERROR(hipFree(d_y));
  CHECK_HIP_ERROR(hipFree(d_z));
  CHECK_HIP_ERROR(hipLibraryUnload(library));
}

void Multiplication::getStream(hipStream_t &strm) { strm = stream; }

void Multiplication::customizeStreamAttributes(hipStream_t &strm) {
  //CHECK_HIP_ERROR(hipStreamCopyAttributes(stream, strm));

  hipStreamAttrID attr = hipStreamAttributeSynchronizationPolicy;
  hipStreamAttrValue valueOut;
  CHECK_HIP_ERROR(hipStreamGetAttribute(stream, attr, &valueOut));
  std::cout << __FUNCTION__ << " : Stream Sync Policy: " << valueOut.syncPolicy
            << std::endl;
}
