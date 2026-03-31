/* Copyright (c) 2026 Advanced Micro Devices, Inc. - MIT License */

/* HRR Windows proxy DLL.
 *
 * Build: cl /LD /Fe:amdhip64_7.dll hrr_proxy_win.c hrr_trace_writer.c
 *           hrr_code_object.c /link /DEF:amdhip64_7.def
 *
 * Usage: Place amdhip64_7.dll in the application directory alongside
 *        HRR_RECORD=1 environment variable set.
 *
 * The proxy DLL loads the real amdhip64_7.dll via LoadLibraryA using
 * the full path from the HIP installation, then forwards all calls
 * while recording a .hrr trace. */

#ifdef _WIN32

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hrr_trace_writer.h"

/* Real DLL handle */
static HMODULE g_real_dll = NULL;

/* Real function pointers */
typedef int (__cdecl *pf_hipInit)(unsigned int);
typedef int (__cdecl *pf_hipMalloc)(void**, size_t);
typedef int (__cdecl *pf_hipFree)(void*);
typedef int (__cdecl *pf_hipMemcpy)(void*, const void*, size_t, unsigned int);
typedef int (__cdecl *pf_hipMemset)(void*, int, size_t);
typedef int (__cdecl *pf_hipModuleLoadData)(void**, const void*);
typedef int (__cdecl *pf_hipModuleUnload)(void*);
typedef int (__cdecl *pf_hipModuleLaunchKernel)(void*, unsigned, unsigned,
    unsigned, unsigned, unsigned, unsigned, unsigned, void*,
    void**, void**);
typedef int (__cdecl *pf_hipDeviceSynchronize)(void);
typedef int (__cdecl *pf_hipStreamSynchronize)(void*);

static pf_hipInit real_hipInit;
static pf_hipMalloc real_hipMalloc;
static pf_hipFree real_hipFree;
static pf_hipMemcpy real_hipMemcpy;
static pf_hipMemset real_hipMemset;
static pf_hipModuleLoadData real_hipModuleLoadData;
static pf_hipModuleUnload real_hipModuleUnload;
static pf_hipModuleLaunchKernel real_hipModuleLaunchKernel;
static pf_hipDeviceSynchronize real_hipDeviceSynchronize;
static pf_hipStreamSynchronize real_hipStreamSynchronize;

static int g_loaded = 0;

static void load_real_dll(void) {
  if (g_loaded) return;
  g_loaded = 1;

  /* Try to find the real DLL:
   * 1. HRR_REAL_HIP_PATH env var (explicit)
   * 2. ROCM_PATH/bin/amdhip64_7.dll
   * 3. System PATH (last resort) */
  const char* explicit_path = getenv("HRR_REAL_HIP_PATH");
  if (explicit_path && explicit_path[0]) {
    g_real_dll = LoadLibraryA(explicit_path);
  }

  if (!g_real_dll) {
    const char* rocm_path = getenv("ROCM_PATH");
    if (rocm_path) {
      char buf[MAX_PATH];
      snprintf(buf, sizeof(buf), "%s\\bin\\amdhip64_7.dll", rocm_path);
      g_real_dll = LoadLibraryA(buf);
    }
  }

  if (!g_real_dll) {
    /* Try loading from system directory to avoid loading ourselves */
    char sys_dir[MAX_PATH];
    GetSystemDirectoryA(sys_dir, sizeof(sys_dir));
    char buf[MAX_PATH];
    snprintf(buf, sizeof(buf), "%s\\amdhip64_7.dll", sys_dir);
    g_real_dll = LoadLibraryA(buf);
  }

  if (!g_real_dll) {
    fprintf(stderr, "[HRR] FATAL: Cannot find real amdhip64_7.dll\n"
                    "[HRR] Set HRR_REAL_HIP_PATH or ROCM_PATH\n");
    ExitProcess(1);
  }

  /* Load function pointers */
#define LOAD(name) real_##name = (pf_##name)GetProcAddress(g_real_dll, #name)
  LOAD(hipInit);
  LOAD(hipMalloc);
  LOAD(hipFree);
  LOAD(hipMemcpy);
  LOAD(hipMemset);
  LOAD(hipModuleLoadData);
  LOAD(hipModuleUnload);
  LOAD(hipModuleLaunchKernel);
  LOAD(hipDeviceSynchronize);
  LOAD(hipStreamSynchronize);
#undef LOAD
}

/* ---- Exported proxy functions ---- */

__declspec(dllexport) int __cdecl hipInit(unsigned int flags) {
  load_real_dll();
  int ret = real_hipInit(flags);
  hrr_writer_init();
  return ret;
}

__declspec(dllexport) int __cdecl hipMalloc(void** ptr, size_t size) {
  load_real_dll();
  int ret = real_hipMalloc(ptr, size);
  if (ret == 0 && hrr_writer_enabled()) {
    hrr_record_malloc(*ptr, size, 0);
  }
  return ret;
}

__declspec(dllexport) int __cdecl hipFree(void* ptr) {
  load_real_dll();
  if (hrr_writer_enabled()) hrr_record_free(ptr);
  return real_hipFree(ptr);
}

__declspec(dllexport) int __cdecl hipMemcpy(void* dst, const void* src,
                                            size_t size, unsigned int kind) {
  load_real_dll();
  if (hrr_writer_enabled()) hrr_record_memcpy(dst, src, size, kind, NULL);
  return real_hipMemcpy(dst, src, size, kind);
}

__declspec(dllexport) int __cdecl hipMemset(void* dst, int value, size_t count) {
  load_real_dll();
  if (hrr_writer_enabled()) hrr_record_memset(dst, value, count, NULL);
  return real_hipMemset(dst, value, count);
}

__declspec(dllexport) int __cdecl hipModuleLoadData(void** module,
                                                    const void* image) {
  load_real_dll();
  int ret = real_hipModuleLoadData(module, image);
  if (ret == 0 && hrr_writer_enabled() && module && *module && image) {
    const unsigned char* p = (const unsigned char*)image;
    size_t sz = 0;
    if (p[0] == 0x7f && p[1] == 'E' && p[2] == 'L' && p[3] == 'F') {
      uint64_t e_shoff; uint16_t e_shentsz, e_shnum;
      memcpy(&e_shoff, p+40, 8);
      memcpy(&e_shentsz, p+58, 2);
      memcpy(&e_shnum, p+60, 2);
      sz = (size_t)(e_shoff + (size_t)e_shentsz * e_shnum);
    }
    if (sz > 0) hrr_record_module_load(*module, image, sz);
  }
  return ret;
}

__declspec(dllexport) int __cdecl hipModuleUnload(void* module) {
  load_real_dll();
  if (hrr_writer_enabled()) hrr_record_module_unload(module);
  return real_hipModuleUnload(module);
}

__declspec(dllexport) int __cdecl hipModuleLaunchKernel(
    void* f, unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz,
    unsigned shared, void* stream, void** params, void** extra) {
  load_real_dll();
  if (hrr_writer_enabled()) {
    hrr_record_kernel_launch(NULL, NULL, 0,
                             gx, gy, gz, bx, by, bz,
                             shared, stream, params);
  }
  return real_hipModuleLaunchKernel(f, gx, gy, gz, bx, by, bz,
                                    shared, stream, params, extra);
}

__declspec(dllexport) int __cdecl hipDeviceSynchronize(void) {
  load_real_dll();
  int ret = real_hipDeviceSynchronize();
  if (hrr_writer_enabled()) hrr_record_device_sync();
  return ret;
}

__declspec(dllexport) int __cdecl hipStreamSynchronize(void* stream) {
  load_real_dll();
  int ret = real_hipStreamSynchronize(stream);
  if (hrr_writer_enabled()) hrr_record_stream_sync(stream);
  return ret;
}

/* DLL entry point */
BOOL APIENTRY DllMain(HMODULE hModule, DWORD reason, LPVOID lpReserved) {
  switch (reason) {
    case DLL_PROCESS_ATTACH:
      break;
    case DLL_PROCESS_DETACH:
      hrr_writer_shutdown();
      if (g_real_dll) FreeLibrary(g_real_dll);
      break;
  }
  return TRUE;
}

#endif /* _WIN32 */
