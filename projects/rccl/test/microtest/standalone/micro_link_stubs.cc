// Link-time stubs for HIP symbols referenced by RCCL headers but not defined
// in fakes/hip_fakes.cc. Filled in as undefined-reference link errors appear.
#include <hip/hip_runtime.h>

hipError_t hipSetDevice(int) { return hipErrorInvalidValue; }
hipError_t hipMalloc(void** p, size_t) { if (p) *p = nullptr; return hipErrorInvalidValue; }
hipError_t hipMemcpy(void*, const void*, size_t, hipMemcpyKind) { return hipErrorInvalidValue; }
hipError_t hipMemset(void*, int, size_t) { return hipErrorInvalidValue; }
hipError_t hipDeviceSynchronize(void) { return hipErrorInvalidValue; }
hipError_t hipGetDeviceProperties(hipDeviceProp_t*, int) { return hipErrorInvalidValue; }
hipError_t hipDriverGetVersion(int* v) { if (v) *v = 70002000; return hipSuccess; }
hipError_t hipStreamWaitEvent(hipStream_t, hipEvent_t, unsigned int) { return hipErrorInvalidValue; }
hipError_t hipStreamCreate(hipStream_t*) { return hipErrorInvalidValue; }
hipError_t hipPointerGetAttributes(hipPointerAttribute_t*, const void*) { return hipErrorInvalidValue; }
hipError_t hipHostGetDevicePointer(void**, void*, unsigned int) { return hipErrorInvalidValue; }
hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t*, hipEvent_t) { return hipErrorInvalidValue; }
hipError_t hipEventSynchronize(hipEvent_t) { return hipErrorInvalidValue; }
