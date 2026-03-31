/* Copyright (c) 2026 Advanced Micro Devices, Inc. - MIT License */

/* HRR LD_PRELOAD interposer for Linux.
 *
 * Build: gcc -shared -fPIC -o libhrr_record.so \
 *          hrr_interposer_linux.c hrr_trace_writer.c hrr_code_object.c \
 *          -ldl -lpthread
 *
 * Usage: HRR_RECORD=1 LD_PRELOAD=./libhrr_record.so <hip-application>
 *
 * Intercepts HIP API calls via dlsym(RTLD_NEXT) forwarding. */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hrr_trace_writer.h"

/* HIP types (avoid pulling in hip headers) */
typedef int hipError_t;
typedef void* hipModule_t;
typedef void* hipFunction_t;
typedef void* hipStream_t;
typedef void* hipEvent_t;
typedef unsigned int hipMemcpyKind;

/* Real function pointers */
static hipError_t (*real_hipMalloc)(void**, size_t) = NULL;
static hipError_t (*real_hipFree)(void*) = NULL;
static hipError_t (*real_hipMemcpy)(void*, const void*, size_t, hipMemcpyKind) = NULL;
static hipError_t (*real_hipMemset)(void*, int, size_t) = NULL;
static hipError_t (*real_hipModuleLoadData)(hipModule_t*, const void*) = NULL;
static hipError_t (*real_hipModuleUnload)(hipModule_t) = NULL;
static hipError_t (*real_hipModuleLaunchKernel)(hipFunction_t, unsigned, unsigned,
    unsigned, unsigned, unsigned, unsigned, unsigned, hipStream_t,
    void**, void**) = NULL;
static hipError_t (*real_hipModuleGetFunction)(hipFunction_t*, hipModule_t,
    const char*) = NULL;
static hipError_t (*real_hipDeviceSynchronize)(void) = NULL;
static hipError_t (*real_hipStreamSynchronize)(hipStream_t) = NULL;
static hipError_t (*real_hipInit)(unsigned int) = NULL;

#define LOAD_SYM(name) \
  if (!real_##name) real_##name = dlsym(RTLD_NEXT, #name)

static int g_initialized = 0;

static void ensure_init(void) {
  if (g_initialized) return;
  g_initialized = 1;
  hrr_writer_init();
  if (hrr_writer_enabled()) {
    atexit(hrr_writer_shutdown);
  }
}

/* ---- Interposed functions ---- */

hipError_t hipInit(unsigned int flags) {
  LOAD_SYM(hipInit);
  hipError_t ret = real_hipInit(flags);
  ensure_init();
  return ret;
}

hipError_t hipMalloc(void** ptr, size_t size) {
  LOAD_SYM(hipMalloc);
  hipError_t ret = real_hipMalloc(ptr, size);
  if (ret == 0 && hrr_writer_enabled()) {
    hrr_record_malloc(*ptr, size, 0);
  }
  return ret;
}

hipError_t hipFree(void* ptr) {
  LOAD_SYM(hipFree);
  if (hrr_writer_enabled()) {
    hrr_record_free(ptr);
  }
  return real_hipFree(ptr);
}

hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes,
                     hipMemcpyKind kind) {
  LOAD_SYM(hipMemcpy);
  if (hrr_writer_enabled()) {
    hrr_record_memcpy(dst, src, sizeBytes, (unsigned int)kind, NULL);
  }
  return real_hipMemcpy(dst, src, sizeBytes, kind);
}

hipError_t hipMemset(void* dst, int value, size_t count) {
  LOAD_SYM(hipMemset);
  if (hrr_writer_enabled()) {
    hrr_record_memset(dst, value, count, NULL);
  }
  return real_hipMemset(dst, value, count);
}

hipError_t hipModuleLoadData(hipModule_t* module, const void* image) {
  LOAD_SYM(hipModuleLoadData);
  hipError_t ret = real_hipModuleLoadData(module, image);
  if (ret == 0 && hrr_writer_enabled() && module && *module && image) {
    /* Estimate code object size from ELF */
    size_t image_size = 0;
    const unsigned char* p = (const unsigned char*)image;
    if (p[0] == 0x7f && p[1] == 'E' && p[2] == 'L' && p[3] == 'F') {
      uint64_t e_shoff;
      uint16_t e_shentsize, e_shnum;
      memcpy(&e_shoff, p + 40, 8);
      memcpy(&e_shentsize, p + 58, 2);
      memcpy(&e_shnum, p + 60, 2);
      image_size = (size_t)(e_shoff + (size_t)e_shentsize * e_shnum);
    }
    if (image_size > 0) {
      hrr_record_module_load(*module, image, image_size);
    }
  }
  return ret;
}

hipError_t hipModuleUnload(hipModule_t module) {
  LOAD_SYM(hipModuleUnload);
  if (hrr_writer_enabled()) {
    hrr_record_module_unload(module);
  }
  return real_hipModuleUnload(module);
}

hipError_t hipModuleLaunchKernel(hipFunction_t f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, hipStream_t hStream,
    void** kernelParams, void** extra) {
  LOAD_SYM(hipModuleLaunchKernel);

  if (hrr_writer_enabled()) {
    /* We don't have the kernel name from hipFunction_t in the out-of-tree
     * path without accessing CLR internals. Record as "<unknown>" for now.
     * A future enhancement could use hipFuncGetName if available. */
    hrr_record_kernel_launch(NULL, NULL, 0,
                             gridDimX, gridDimY, gridDimZ,
                             blockDimX, blockDimY, blockDimZ,
                             sharedMemBytes, hStream, kernelParams);
  }

  return real_hipModuleLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
      blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream,
      kernelParams, extra);
}

hipError_t hipDeviceSynchronize(void) {
  LOAD_SYM(hipDeviceSynchronize);
  hipError_t ret = real_hipDeviceSynchronize();
  if (hrr_writer_enabled()) {
    hrr_record_device_sync();
  }
  return ret;
}

hipError_t hipStreamSynchronize(hipStream_t stream) {
  LOAD_SYM(hipStreamSynchronize);
  hipError_t ret = real_hipStreamSynchronize(stream);
  if (hrr_writer_enabled()) {
    hrr_record_stream_sync(stream);
  }
  return ret;
}

/* Constructor/destructor for shared library lifecycle */
__attribute__((constructor))
static void hrr_lib_init(void) {
  /* Delay actual init until hipInit is called, to avoid
   * issues with library load ordering */
}

__attribute__((destructor))
static void hrr_lib_fini(void) {
  hrr_writer_shutdown();
}
