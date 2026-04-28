// Wraps a curated subset of HIP runtime dispatch table entries with timing
// shims that record into the in-memory trace buffer.
//
// The HipDispatchTable defined in clr/hipamd/include/hip/amd_detail/hip_api_trace.hpp
// has hundreds of entries -- generating wrappers for every single one would
// require knowing every function's signature. We wrap the most common ones
// used by typical HIP applications. Functions that are not wrapped are left
// pointing at their original implementations and simply will not appear in
// the trace.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

// hip_api_trace.hpp pulls in deprecated and GL types -- include them up front.
#include <hip/hip_runtime.h>
#include <hip/hip_deprecated.h>
#include <hip/hip_gl_interop.h>
#include <hip/amd_detail/hip_api_trace.hpp>

#include "trace_buffer.h"

#include <cstring>

namespace rocprofv3_min {

extern bool g_hip_trace_enabled;

static LARGE_INTEGER g_qpc_freq{};
inline uint64_t now_ns() {
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    if (g_qpc_freq.QuadPart == 0) QueryPerformanceFrequency(&g_qpc_freq);
    long double sec = (long double)c.QuadPart / (long double)g_qpc_freq.QuadPart;
    return (uint64_t)(sec * 1e9L);
}

// We snapshot the entire dispatch table before mutating the broker-supplied
// one, so wrappers can forward via the snapshot (the original pointers).
namespace hip_orig {
    static HipDispatchTable         table_snap{};
    static HipCompilerDispatchTable ctable_snap{};
    static HipDispatchTable*         table   = &table_snap;
    static HipCompilerDispatchTable* ctable  = &ctable_snap;
}

#define EMIT_HIP(name)                                                                   \
    do {                                                                                 \
        HipApiRow row{};                                                                 \
        row.domain = "HIP_API";                                                          \
        row.function_name = #name;                                                       \
        row.process_id = GetCurrentProcessId();                                          \
        row.thread_id = GetCurrentThreadId();                                            \
        row.correlation_id = TraceBuffers::instance().next_correlation_id();             \
        row.start_ns = _start;                                                           \
        row.end_ns = now_ns();                                                           \
        TraceBuffers::instance().push_hip(std::move(row));                               \
    } while (0)

#define EMIT_HIP_COMPILER(name)                                                          \
    do {                                                                                 \
        HipApiRow row{};                                                                 \
        row.domain = "HIP_COMPILER_API";                                                 \
        row.function_name = #name;                                                       \
        row.process_id = GetCurrentProcessId();                                          \
        row.thread_id = GetCurrentThreadId();                                            \
        row.correlation_id = TraceBuffers::instance().next_correlation_id();             \
        row.start_ns = _start;                                                           \
        row.end_ns = now_ns();                                                           \
        TraceBuffers::instance().push_hip(std::move(row));                               \
    } while (0)

// ----- HIP runtime wrappers -----

static hipError_t W_hipGetDeviceCount(int* count) {
    uint64_t _start = now_ns();
    auto r = hip_orig::table->hipGetDeviceCount_fn(count);
    EMIT_HIP(hipGetDeviceCount);
    return r;
}
static hipError_t W_hipGetDevice(int* d) {
    uint64_t _start = now_ns();
    auto r = hip_orig::table->hipGetDevice_fn(d);
    EMIT_HIP(hipGetDevice);
    return r;
}
static hipError_t W_hipSetDevice(int d) {
    uint64_t _start = now_ns();
    auto r = hip_orig::table->hipSetDevice_fn(d);
    EMIT_HIP(hipSetDevice);
    return r;
}
static hipError_t W_hipGetDevicePropertiesR0600(hipDeviceProp_tR0600* p, int d) {
    uint64_t _start = now_ns();
    auto r = hip_orig::table->hipGetDevicePropertiesR0600_fn(p, d);
    EMIT_HIP(hipGetDevicePropertiesR0600);
    return r;
}
static const char* W_hipGetErrorString(hipError_t e) {
    uint64_t _start = now_ns();
    auto r = hip_orig::table->hipGetErrorString_fn(e);
    EMIT_HIP(hipGetErrorString);
    return r;
}
static const char* W_hipGetErrorName(hipError_t e) {
    uint64_t _start = now_ns();
    auto r = hip_orig::table->hipGetErrorName_fn(e);
    EMIT_HIP(hipGetErrorName);
    return r;
}
static hipError_t W_hipDeviceGet(hipDevice_t* d, int o) {
    uint64_t _start = now_ns();
    auto r = hip_orig::table->hipDeviceGet_fn(d, o);
    EMIT_HIP(hipDeviceGet);
    return r;
}
static hipError_t W_hipDeviceGetAttribute(int* v, hipDeviceAttribute_t a, int d) {
    uint64_t _start = now_ns();
    auto r = hip_orig::table->hipDeviceGetAttribute_fn(v, a, d);
    EMIT_HIP(hipDeviceGetAttribute);
    return r;
}
static hipError_t W_hipDeviceGetName(char* name, int len, hipDevice_t d) {
    uint64_t _start = now_ns();
    auto r = hip_orig::table->hipDeviceGetName_fn(name, len, d);
    EMIT_HIP(hipDeviceGetName);
    return r;
}
static hipError_t W_hipDeviceSynchronize(void) {
    uint64_t _start = now_ns();
    auto r = hip_orig::table->hipDeviceSynchronize_fn();
    EMIT_HIP(hipDeviceSynchronize);
    return r;
}
static hipError_t W_hipMalloc(void** ptr, size_t s) {
    uint64_t _start = now_ns();
    auto r = hip_orig::table->hipMalloc_fn(ptr, s);
    EMIT_HIP(hipMalloc);
    return r;
}
static hipError_t W_hipFree(void* p) {
    uint64_t _start = now_ns();
    auto r = hip_orig::table->hipFree_fn(p);
    EMIT_HIP(hipFree);
    return r;
}
static hipError_t W_hipMemcpy(void* d, const void* s, size_t n, hipMemcpyKind k) {
    uint64_t _start = now_ns();
    auto r = hip_orig::table->hipMemcpy_fn(d, s, n, k);
    EMIT_HIP(hipMemcpy);
    return r;
}
static hipError_t W_hipMemcpyAsync(void* d, const void* s, size_t n, hipMemcpyKind k, hipStream_t st) {
    uint64_t _start = now_ns();
    auto r = hip_orig::table->hipMemcpyAsync_fn(d, s, n, k, st);
    EMIT_HIP(hipMemcpyAsync);
    return r;
}
static hipError_t W_hipMemset(void* d, int v, size_t n) {
    uint64_t _start = now_ns();
    auto r = hip_orig::table->hipMemset_fn(d, v, n);
    EMIT_HIP(hipMemset);
    return r;
}
static hipError_t W_hipMemsetAsync(void* d, int v, size_t n, hipStream_t st) {
    uint64_t _start = now_ns();
    auto r = hip_orig::table->hipMemsetAsync_fn(d, v, n, st);
    EMIT_HIP(hipMemsetAsync);
    return r;
}
static hipError_t W_hipStreamCreate(hipStream_t* s) {
    uint64_t _start = now_ns();
    auto r = hip_orig::table->hipStreamCreate_fn(s);
    EMIT_HIP(hipStreamCreate);
    return r;
}
static hipError_t W_hipStreamDestroy(hipStream_t s) {
    uint64_t _start = now_ns();
    auto r = hip_orig::table->hipStreamDestroy_fn(s);
    EMIT_HIP(hipStreamDestroy);
    return r;
}
static hipError_t W_hipStreamSynchronize(hipStream_t s) {
    uint64_t _start = now_ns();
    auto r = hip_orig::table->hipStreamSynchronize_fn(s);
    EMIT_HIP(hipStreamSynchronize);
    return r;
}
static hipError_t W_hipLaunchKernel(const void* f, dim3 g, dim3 b, void** a, size_t sm, hipStream_t st) {
    uint64_t _start = now_ns();
    auto r = hip_orig::table->hipLaunchKernel_fn(f, g, b, a, sm, st);
    EMIT_HIP(hipLaunchKernel);
    return r;
}
static hipError_t W_hipModuleLaunchKernel(hipFunction_t f, unsigned int gx, unsigned int gy,
                                          unsigned int gz, unsigned int bx, unsigned int by,
                                          unsigned int bz, unsigned int sm, hipStream_t st,
                                          void** kp, void** ex) {
    uint64_t _start = now_ns();
    auto r = hip_orig::table->hipModuleLaunchKernel_fn(f, gx, gy, gz, bx, by, bz, sm, st, kp, ex);
    EMIT_HIP(hipModuleLaunchKernel);
    return r;
}
static hipError_t W_hipEventCreate(hipEvent_t* e) {
    uint64_t _start = now_ns();
    auto r = hip_orig::table->hipEventCreate_fn(e);
    EMIT_HIP(hipEventCreate);
    return r;
}
static hipError_t W_hipEventDestroy(hipEvent_t e) {
    uint64_t _start = now_ns();
    auto r = hip_orig::table->hipEventDestroy_fn(e);
    EMIT_HIP(hipEventDestroy);
    return r;
}
static hipError_t W_hipEventRecord(hipEvent_t e, hipStream_t s) {
    uint64_t _start = now_ns();
    auto r = hip_orig::table->hipEventRecord_fn(e, s);
    EMIT_HIP(hipEventRecord);
    return r;
}
static hipError_t W_hipEventSynchronize(hipEvent_t e) {
    uint64_t _start = now_ns();
    auto r = hip_orig::table->hipEventSynchronize_fn(e);
    EMIT_HIP(hipEventSynchronize);
    return r;
}
static hipError_t W_hipModuleLoad(hipModule_t* m, const char* fname) {
    uint64_t _start = now_ns();
    auto r = hip_orig::table->hipModuleLoad_fn(m, fname);
    EMIT_HIP(hipModuleLoad);
    return r;
}
static hipError_t W_hipModuleUnload(hipModule_t m) {
    uint64_t _start = now_ns();
    auto r = hip_orig::table->hipModuleUnload_fn(m);
    EMIT_HIP(hipModuleUnload);
    return r;
}
static hipError_t W_hipModuleGetFunction(hipFunction_t* f, hipModule_t m, const char* n) {
    uint64_t _start = now_ns();
    auto r = hip_orig::table->hipModuleGetFunction_fn(f, m, n);
    EMIT_HIP(hipModuleGetFunction);
    return r;
}
static hipError_t W_hipInit(unsigned int flags) {
    uint64_t _start = now_ns();
    auto r = hip_orig::table->hipInit_fn(flags);
    EMIT_HIP(hipInit);
    return r;
}

// ----- HIP compiler wrappers -----

static hipError_t W___hipPopCallConfiguration(dim3* g, dim3* b, size_t* sm, hipStream_t* s) {
    uint64_t _start = now_ns();
    auto r = hip_orig::ctable->__hipPopCallConfiguration_fn(g, b, sm, s);
    EMIT_HIP_COMPILER(__hipPopCallConfiguration);
    return r;
}
static hipError_t W___hipPushCallConfiguration(dim3 g, dim3 b, size_t sm, hipStream_t s) {
    uint64_t _start = now_ns();
    auto r = hip_orig::ctable->__hipPushCallConfiguration_fn(g, b, sm, s);
    EMIT_HIP_COMPILER(__hipPushCallConfiguration);
    return r;
}
static void** W___hipRegisterFatBinary(const void* d) {
    uint64_t _start = now_ns();
    auto r = hip_orig::ctable->__hipRegisterFatBinary_fn(d);
    EMIT_HIP_COMPILER(__hipRegisterFatBinary);
    return r;
}
static void W___hipRegisterFunction(void** mod, const void* hf, char* df, const char* dn,
                                    unsigned int tl, uint3* tid, uint3* bid, dim3* bd, dim3* gd, int* ws) {
    uint64_t _start = now_ns();
    hip_orig::ctable->__hipRegisterFunction_fn(mod, hf, df, dn, tl, tid, bid, bd, gd, ws);
    EMIT_HIP_COMPILER(__hipRegisterFunction);
}
static void W___hipRegisterVar(void** mod, void* var, char* hv, char* dv, int e, size_t s, int c, int g) {
    uint64_t _start = now_ns();
    hip_orig::ctable->__hipRegisterVar_fn(mod, var, hv, dv, e, s, c, g);
    EMIT_HIP_COMPILER(__hipRegisterVar);
}
static void W___hipUnregisterFatBinary(void** mod) {
    uint64_t _start = now_ns();
    hip_orig::ctable->__hipUnregisterFatBinary_fn(mod);
    EMIT_HIP_COMPILER(__hipUnregisterFatBinary);
}

// ----- Public install hooks -----

void install_hip_wrappers(HipDispatchTable* t) {
    if (!t || !g_hip_trace_enabled) return;
    // Snapshot originals first; size field copies too.
    hip_orig::table_snap = *t;

#define WRAP(field, w)                                                                   \
    if (t->field) t->field = (decltype(t->field))(w)
    WRAP(hipGetDeviceCount_fn,        &W_hipGetDeviceCount);
    WRAP(hipGetDevice_fn,             &W_hipGetDevice);
    WRAP(hipSetDevice_fn,             &W_hipSetDevice);
    WRAP(hipGetDevicePropertiesR0600_fn, &W_hipGetDevicePropertiesR0600);
    WRAP(hipGetErrorString_fn,        &W_hipGetErrorString);
    WRAP(hipGetErrorName_fn,          &W_hipGetErrorName);
    WRAP(hipDeviceGet_fn,             &W_hipDeviceGet);
    WRAP(hipDeviceGetAttribute_fn,    &W_hipDeviceGetAttribute);
    WRAP(hipDeviceGetName_fn,         &W_hipDeviceGetName);
    WRAP(hipDeviceSynchronize_fn,     &W_hipDeviceSynchronize);
    WRAP(hipMalloc_fn,                &W_hipMalloc);
    WRAP(hipFree_fn,                  &W_hipFree);
    WRAP(hipMemcpy_fn,                &W_hipMemcpy);
    WRAP(hipMemcpyAsync_fn,           &W_hipMemcpyAsync);
    WRAP(hipMemset_fn,                &W_hipMemset);
    WRAP(hipMemsetAsync_fn,           &W_hipMemsetAsync);
    WRAP(hipStreamCreate_fn,          &W_hipStreamCreate);
    WRAP(hipStreamDestroy_fn,         &W_hipStreamDestroy);
    WRAP(hipStreamSynchronize_fn,     &W_hipStreamSynchronize);
    WRAP(hipLaunchKernel_fn,          &W_hipLaunchKernel);
    WRAP(hipModuleLaunchKernel_fn,    &W_hipModuleLaunchKernel);
    WRAP(hipEventCreate_fn,           &W_hipEventCreate);
    WRAP(hipEventDestroy_fn,          &W_hipEventDestroy);
    WRAP(hipEventRecord_fn,           &W_hipEventRecord);
    WRAP(hipEventSynchronize_fn,      &W_hipEventSynchronize);
    WRAP(hipModuleLoad_fn,            &W_hipModuleLoad);
    WRAP(hipModuleUnload_fn,          &W_hipModuleUnload);
    WRAP(hipModuleGetFunction_fn,     &W_hipModuleGetFunction);
    WRAP(hipInit_fn,                  &W_hipInit);
#undef WRAP
}

void install_hip_compiler_wrappers(HipCompilerDispatchTable* t) {
    if (!t || !g_hip_trace_enabled) return;
    hip_orig::ctable_snap = *t;

#define WRAP(field, w)                                                                   \
    if (t->field) t->field = (decltype(t->field))(w)
    WRAP(__hipPopCallConfiguration_fn,   &W___hipPopCallConfiguration);
    WRAP(__hipPushCallConfiguration_fn,  &W___hipPushCallConfiguration);
    WRAP(__hipRegisterFatBinary_fn,      &W___hipRegisterFatBinary);
    WRAP(__hipRegisterFunction_fn,       &W___hipRegisterFunction);
    WRAP(__hipRegisterVar_fn,            &W___hipRegisterVar);
    WRAP(__hipUnregisterFatBinary_fn,    &W___hipUnregisterFatBinary);
#undef WRAP
}

} // namespace rocprofv3_min
