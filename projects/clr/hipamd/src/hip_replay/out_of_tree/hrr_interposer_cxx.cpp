/* Copyright (c) 2026 Advanced Micro Devices, Inc. - MIT License */

/* C++ extension for HRR LD_PRELOAD interposer.
 *
 * Intercepts C++ linkage HIP APIs that cannot be safely wrapped from plain C.
 * Currently: hipExtModuleLaunchKernel (mangled C++ function).
 *
 * Compiled alongside hrr_interposer_linux.c with g++. */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <cstddef>
#include <cstdint>

/* Minimal HIP opaque struct forward-declarations.
 * These match the real HIP types and produce the correct Itanium C++ ABI
 * mangling so our function overrides the right symbol. */
struct ihipModuleSymbol_t;
struct ihipStream_t;
struct ihipEvent_t;

typedef int hipError_t;
typedef ihipModuleSymbol_t* hipFunction_t;
typedef ihipStream_t*       hipStream_t;
typedef ihipEvent_t*        hipEvent_t;

/* C recording functions from hrr_trace_writer.c */
extern "C" {
  int  hrr_writer_enabled(void);
  void hrr_record_kernel_launch(const char* kernel_name,
                                 const void* code_object_image,
                                 size_t code_object_size,
                                 uint32_t gx, uint32_t gy, uint32_t gz,
                                 uint32_t bx, uint32_t by, uint32_t bz,
                                 uint32_t shared_mem,
                                 const void* stream,
                                 void** kernel_args);
  void hrr_record_kernel_launch_packed(const char* kernel_name,
                                        uint32_t gx, uint32_t gy, uint32_t gz,
                                        uint32_t bx, uint32_t by, uint32_t bz,
                                        uint32_t shared_mem,
                                        const void* stream,
                                        const void* packed_buf,
                                        size_t packed_size);
  const char* hrr_lookup_function_name(const void* func_handle);
}

/* Real function pointer — looked up via dlsym by mangled name to avoid
 * any ambiguity. */
static hipError_t (*real_hipExtModuleLaunchKernel)(
    hipFunction_t f,
    unsigned int globalWorkSizeX, unsigned int globalWorkSizeY,
    unsigned int globalWorkSizeZ,
    unsigned int localWorkSizeX, unsigned int localWorkSizeY,
    unsigned int localWorkSizeZ,
    unsigned long sharedMemBytes,
    hipStream_t hStream,
    void** kernelParams, void** extra,
    hipEvent_t startEvent, hipEvent_t stopEvent,
    unsigned int flags) = nullptr;

static const char kExtLaunchSym[] =
    "_Z24hipExtModuleLaunchKernelP18ihipModuleSymbol_t"
    "jjjjjjmP12ihipStream_tPPvS4_P11ihipEvent_tS6_j";

/* Override of hipExtModuleLaunchKernel.
 * The function signature here produces exactly the mangled name
 * that MIGraphX and other workloads compiled with HIP expect. */
hipError_t hipExtModuleLaunchKernel(
    hipFunction_t f,
    unsigned int globalWorkSizeX, unsigned int globalWorkSizeY,
    unsigned int globalWorkSizeZ,
    unsigned int localWorkSizeX, unsigned int localWorkSizeY,
    unsigned int localWorkSizeZ,
    unsigned long sharedMemBytes,
    hipStream_t hStream,
    void** kernelParams, void** extra,
    hipEvent_t startEvent = nullptr, hipEvent_t stopEvent = nullptr,
    unsigned int flags = 0) {

  if (!real_hipExtModuleLaunchKernel) {
    real_hipExtModuleLaunchKernel =
        reinterpret_cast<decltype(real_hipExtModuleLaunchKernel)>(
            dlsym(RTLD_NEXT, kExtLaunchSym));

    if (!real_hipExtModuleLaunchKernel) {
      /* RTLD_NEXT failed (early init order issue); try by name in default scope */
      real_hipExtModuleLaunchKernel =
          reinterpret_cast<decltype(real_hipExtModuleLaunchKernel)>(
              dlsym(RTLD_DEFAULT, kExtLaunchSym));
    }
  }

  if (hrr_writer_enabled()) {
    const char* kname = hrr_lookup_function_name(static_cast<const void*>(f));
    if (kernelParams) {
      /* Standard kernelParams path: array of pointers to arg values */
      hrr_record_kernel_launch(kname, nullptr, 0,
                                globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ,
                                localWorkSizeX, localWorkSizeY, localWorkSizeZ,
                                static_cast<uint32_t>(sharedMemBytes),
                                static_cast<const void*>(hStream), kernelParams);
    } else if (extra) {
      /* Packed kernarg buffer path (MIGraphX / hipExtModuleLaunchKernel).
       * extra = { HIP_LAUNCH_PARAM_BUFFER_POINTER, buf,
       *           HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
       *           HIP_LAUNCH_PARAM_END } */
      const void* packed_buf = nullptr;
      size_t packed_size = 0;
      for (int ei = 0; ; ei += 2) {
        if (extra[ei] == reinterpret_cast<void*>(0x01)) {
          packed_buf = extra[ei + 1];
        } else if (extra[ei] == reinterpret_cast<void*>(0x02)) {
          packed_size = *reinterpret_cast<const size_t*>(extra[ei + 1]);
        } else {
          break;  // HIP_LAUNCH_PARAM_END or unknown tag
        }
      }
      if (packed_buf) {
        hrr_record_kernel_launch_packed(kname,
                                        globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ,
                                        localWorkSizeX, localWorkSizeY, localWorkSizeZ,
                                        static_cast<uint32_t>(sharedMemBytes),
                                        static_cast<const void*>(hStream),
                                        packed_buf, packed_size);
      }
    }
  }

  if (!real_hipExtModuleLaunchKernel) return -1;
  return real_hipExtModuleLaunchKernel(
      f,
      globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ,
      localWorkSizeX, localWorkSizeY, localWorkSizeZ,
      sharedMemBytes, hStream,
      kernelParams, extra,
      startEvent, stopEvent, flags);
}
