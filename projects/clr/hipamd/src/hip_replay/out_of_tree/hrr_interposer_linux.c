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
static hipError_t (*real_hipExtMallocWithFlags)(void**, size_t, unsigned int) = NULL;
static hipError_t (*real_hipMallocManaged)(void**, size_t, unsigned int) = NULL;
static hipError_t (*real_hipMallocAsync)(void**, size_t, hipStream_t) = NULL;
static hipError_t (*real_hipFreeAsync)(void*, hipStream_t) = NULL;
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

/* Explicit handle to the real HIP library.
 * RTLD_NEXT fails during early init if libamdhip64 isn't in the chain yet
 * (e.g. called from another library's constructor). We fall back to
 * dlopen("libamdhip64.so.6") to get the real functions. */
static void* s_hip_lib = NULL;

static void* hrr_load_hip_sym(const char* name) {
  void* sym = dlsym(RTLD_NEXT, name);
  if (sym) return sym;

  /* RTLD_NEXT returned NULL — libamdhip64 not yet in chain. Try explicit load. */
  if (!s_hip_lib) {
    s_hip_lib = dlopen("libamdhip64.so.6", RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
    if (!s_hip_lib)
      s_hip_lib = dlopen("libamdhip64.so", RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
    /* RTLD_NOLOAD: only find it if already mapped; don't pull in a new copy. */
  }
  if (s_hip_lib) return dlsym(s_hip_lib, name);
  return NULL;
}

#define LOAD_SYM(name) \
  if (!real_##name) real_##name = hrr_load_hip_sym(#name)

/* Forward a call to a real HIP function, returning hipErrorUnknown if the
 * symbol could not be resolved (prevents null-pointer dereference). */
#define FORWARD_OR_ERROR(name, args) \
  do { if (!real_##name) return -1; return real_##name args; } while(0)

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
  ensure_init();
  FORWARD_OR_ERROR(hipInit, (flags));
}

hipError_t hipMalloc(void** ptr, size_t size) {
  LOAD_SYM(hipMalloc);
  ensure_init();
  if (!real_hipMalloc) return -1;
  hipError_t ret = real_hipMalloc(ptr, size);
  if (ret == 0 && hrr_writer_enabled()) {
    hrr_record_malloc(*ptr, size, 0);
  }
  return ret;
}

hipError_t hipExtMallocWithFlags(void** ptr, size_t sizeBytes, unsigned int flags) {
  LOAD_SYM(hipExtMallocWithFlags);
  ensure_init();
  if (!real_hipExtMallocWithFlags) return -1;
  hipError_t ret = real_hipExtMallocWithFlags(ptr, sizeBytes, flags);
  if (ret == 0 && hrr_writer_enabled()) {
    hrr_record_malloc(*ptr, sizeBytes, flags);
  }
  return ret;
}

hipError_t hipMallocManaged(void** ptr, size_t size, unsigned int flags) {
  LOAD_SYM(hipMallocManaged);
  ensure_init();
  if (!real_hipMallocManaged) return -1;
  hipError_t ret = real_hipMallocManaged(ptr, size, flags);
  if (ret == 0 && hrr_writer_enabled()) {
    hrr_record_malloc(*ptr, size, flags);
  }
  return ret;
}

hipError_t hipMallocAsync(void** ptr, size_t size, hipStream_t stream) {
  LOAD_SYM(hipMallocAsync);
  ensure_init();
  if (!real_hipMallocAsync) return -1;
  hipError_t ret = real_hipMallocAsync(ptr, size, stream);
  if (ret == 0 && hrr_writer_enabled()) {
    hrr_record_malloc(*ptr, size, 0);
  }
  return ret;
}

hipError_t hipFreeAsync(void* ptr, hipStream_t stream) {
  LOAD_SYM(hipFreeAsync);
  if (hrr_writer_enabled()) {
    hrr_record_free(ptr);
  }
  FORWARD_OR_ERROR(hipFreeAsync, (ptr, stream));
}

hipError_t hipFree(void* ptr) {
  LOAD_SYM(hipFree);
  if (hrr_writer_enabled()) {
    hrr_record_free(ptr);
  }
  FORWARD_OR_ERROR(hipFree, (ptr));
}

hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes,
                     hipMemcpyKind kind) {
  LOAD_SYM(hipMemcpy);
  if (hrr_writer_enabled()) {
    hrr_record_memcpy(dst, src, sizeBytes, (unsigned int)kind, NULL);
  }
  FORWARD_OR_ERROR(hipMemcpy, (dst, src, sizeBytes, kind));
}

hipError_t hipMemset(void* dst, int value, size_t count) {
  LOAD_SYM(hipMemset);
  if (hrr_writer_enabled()) {
    hrr_record_memset(dst, value, count, NULL);
  }
  FORWARD_OR_ERROR(hipMemset, (dst, value, count));
}

hipError_t hipModuleLoadData(hipModule_t* module, const void* image) {
  LOAD_SYM(hipModuleLoadData);
  if (!real_hipModuleLoadData) return -1;
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
  FORWARD_OR_ERROR(hipModuleUnload, (module));
}

hipError_t hipModuleGetFunction(hipFunction_t* hfunc, hipModule_t hmod,
                                const char* name) {
  LOAD_SYM(hipModuleGetFunction);
  if (!real_hipModuleGetFunction) return -1;
  hipError_t ret = real_hipModuleGetFunction(hfunc, hmod, name);
  /* Always register the handle→name mapping regardless of recording state.
   * Used by hipExtModuleLaunchKernel to recover the kernel name. */
  if (ret == 0 && hfunc && *hfunc && name) {
    hrr_register_function(*hfunc, name);
  }
  return ret;
}

hipError_t hipModuleLaunchKernel(hipFunction_t f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, hipStream_t hStream,
    void** kernelParams, void** extra) {
  LOAD_SYM(hipModuleLaunchKernel);

  if (hrr_writer_enabled()) {
    /* Kernel name not available without CLR internals; recorded as anonymous. */
    hrr_record_kernel_launch(NULL, NULL, 0,
                             gridDimX, gridDimY, gridDimZ,
                             blockDimX, blockDimY, blockDimZ,
                             sharedMemBytes, hStream, kernelParams);
  }

  if (!real_hipModuleLaunchKernel) return -1;
  return real_hipModuleLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
      blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream,
      kernelParams, extra);
}

/* Note: hipLaunchKernel (<<<>>> / hipLaunchKernelGGL path) is intentionally
 * NOT intercepted here. Its ABI uses dim3 (a C++ struct with user-provided
 * constructor) which cannot be safely forwarded from a plain-C interposer.
 * MIGraphX and other ONNX-compiled workloads use hipModuleLaunchKernel for
 * pre-compiled code objects, which is captured above. */

hipError_t hipDeviceSynchronize(void) {
  LOAD_SYM(hipDeviceSynchronize);
  if (!real_hipDeviceSynchronize) return -1;
  hipError_t ret = real_hipDeviceSynchronize();
  if (hrr_writer_enabled()) {
    hrr_record_device_sync();
  }
  return ret;
}

hipError_t hipStreamSynchronize(hipStream_t stream) {
  LOAD_SYM(hipStreamSynchronize);
  if (!real_hipStreamSynchronize) return -1;
  hipError_t ret = real_hipStreamSynchronize(stream);
  if (hrr_writer_enabled()) {
    hrr_record_stream_sync(stream);
  }
  return ret;
}

/* Constructor/destructor for shared library lifecycle */
__attribute__((constructor))
static void hrr_lib_init(void) {
  /* Initialize early so recording is active before any HIP calls */
  ensure_init();
}

__attribute__((destructor))
static void hrr_lib_fini(void) {
  hrr_writer_shutdown();
}
