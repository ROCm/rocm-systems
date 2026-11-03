// Local POC (not for a PR): one-event-per-call HIP API probes and a kernel_dispatch probe,
// written with TraceLogging (LTTng-UST backend). Follows api-tracing-plan.md section 3.
//
// All TraceLoggingWrite calls live in this TU: on LTTng the provider symbol is a linker-section
// token, so the provider definition and every write must be in the same DSO.

#include <tracelogging/TraceLoggingProvider.h>

#include <hip/hip_runtime_api.h>
#include <hip/amd_detail/hip_api_trace.hpp>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <ctime>

TRACELOGGING_DEFINE_PROVIDER(rocm_hip, "rocm_hip",
                             (0x07416523, 0xfe27, 0x5716, 0xe8, 0x52, 0xd1, 0x87, 0x74, 0xa0, 0x37,
                              0x43));

namespace {
constexpr uint64_t kKwApi = 0x01;
constexpr uint64_t kKwDispatch = 0x02;

inline uint64_t now_ns() {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);  // LTTng's trace clock
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

inline uint64_t u64(const void* p) { return reinterpret_cast<uintptr_t>(p); }

#define TL_LEVEL TraceLoggingLevel(TRACE_LEVEL_INFORMATION)

// ---------------------------------------------------------------- saved table entries
decltype(HipDispatchTable::hipLaunchKernel_fn) next_hipLaunchKernel;
decltype(HipDispatchTable::hipModuleLaunchKernel_fn) next_hipModuleLaunchKernel;
decltype(HipDispatchTable::hipMalloc_fn) next_hipMalloc;
decltype(HipDispatchTable::hipFree_fn) next_hipFree;
decltype(HipDispatchTable::hipMemcpy_fn) next_hipMemcpy;
decltype(HipDispatchTable::hipMemcpyAsync_fn) next_hipMemcpyAsync;
decltype(HipDispatchTable::hipDeviceSynchronize_fn) next_hipDeviceSynchronize;
decltype(HipDispatchTable::hipStreamCreate_fn) next_hipStreamCreate;
decltype(HipDispatchTable::hipStreamDestroy_fn) next_hipStreamDestroy;
decltype(HipDispatchTable::hipStreamSynchronize_fn) next_hipStreamSynchronize;
decltype(HipDispatchTable::hipSetDevice_fn) next_hipSetDevice;
decltype(HipDispatchTable::hipGetDevice_fn) next_hipGetDevice;
decltype(HipDispatchTable::hipEventRecord_fn) next_hipEventRecord;
decltype(HipDispatchTable::hipEventSynchronize_fn) next_hipEventSynchronize;

// ---------------------------------------------------------------- probes: one event per call
hipError_t probe_hipLaunchKernel(const void* function_address, dim3 numBlocks, dim3 dimBlocks,
                                 void** args, size_t sharedMemBytes, hipStream_t stream) {
  if (!TraceLoggingEventEnabled(rocm_hip, "hipLaunchKernel"))
    return next_hipLaunchKernel(function_address, numBlocks, dimBlocks, args, sharedMemBytes,
                                stream);
  const uint64_t start = now_ns();
  const hipError_t r =
      next_hipLaunchKernel(function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream);
  TraceLoggingWrite(rocm_hip, "hipLaunchKernel", TL_LEVEL, TraceLoggingKeyword(kKwApi),
                    TraceLoggingUInt64(start, "start_timestamp"),
                    TraceLoggingHexUInt64(u64(function_address), "function_address"),
                    TraceLoggingUInt32(numBlocks.x, "numBlocks_x"),
                    TraceLoggingUInt32(numBlocks.y, "numBlocks_y"),
                    TraceLoggingUInt32(numBlocks.z, "numBlocks_z"),
                    TraceLoggingUInt32(dimBlocks.x, "dimBlocks_x"),
                    TraceLoggingUInt32(dimBlocks.y, "dimBlocks_y"),
                    TraceLoggingUInt32(dimBlocks.z, "dimBlocks_z"),
                    TraceLoggingHexUInt64(u64(args), "args"),
                    TraceLoggingUInt64(sharedMemBytes, "sharedMemBytes"),
                    TraceLoggingHexUInt64(u64(stream), "stream"), TraceLoggingInt32(r, "result"));
  return r;
}

hipError_t probe_hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX,
                                       unsigned int gridDimY, unsigned int gridDimZ,
                                       unsigned int blockDimX, unsigned int blockDimY,
                                       unsigned int blockDimZ, unsigned int sharedMemBytes,
                                       hipStream_t stream, void** kernelParams, void** extra) {
  if (!TraceLoggingEventEnabled(rocm_hip, "hipModuleLaunchKernel"))
    return next_hipModuleLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                                      blockDimZ, sharedMemBytes, stream, kernelParams, extra);
  const uint64_t start = now_ns();
  const hipError_t r = next_hipModuleLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX,
                                                  blockDimY, blockDimZ, sharedMemBytes, stream,
                                                  kernelParams, extra);
  TraceLoggingWrite(rocm_hip, "hipModuleLaunchKernel", TL_LEVEL, TraceLoggingKeyword(kKwApi),
                    TraceLoggingUInt64(start, "start_timestamp"),
                    TraceLoggingHexUInt64(u64(f), "f"), TraceLoggingUInt32(gridDimX, "gridDimX"),
                    TraceLoggingUInt32(gridDimY, "gridDimY"),
                    TraceLoggingUInt32(gridDimZ, "gridDimZ"),
                    TraceLoggingUInt32(blockDimX, "blockDimX"),
                    TraceLoggingUInt32(blockDimY, "blockDimY"),
                    TraceLoggingUInt32(blockDimZ, "blockDimZ"),
                    TraceLoggingUInt32(sharedMemBytes, "sharedMemBytes"),
                    TraceLoggingHexUInt64(u64(stream), "stream"), TraceLoggingInt32(r, "result"));
  return r;
}

hipError_t probe_hipMalloc(void** ptr, size_t size) {
  if (!TraceLoggingEventEnabled(rocm_hip, "hipMalloc")) return next_hipMalloc(ptr, size);
  const uint64_t start = now_ns();
  const hipError_t r = next_hipMalloc(ptr, size);
  TraceLoggingWrite(rocm_hip, "hipMalloc", TL_LEVEL, TraceLoggingKeyword(kKwApi),
                    TraceLoggingUInt64(start, "start_timestamp"), TraceLoggingUInt64(size, "size"),
                    TraceLoggingHexUInt64(r == hipSuccess && ptr ? u64(*ptr) : 0, "ptr"),
                    TraceLoggingInt32(r, "result"));
  return r;
}

hipError_t probe_hipFree(void* ptr) {
  if (!TraceLoggingEventEnabled(rocm_hip, "hipFree")) return next_hipFree(ptr);
  const uint64_t start = now_ns();
  const hipError_t r = next_hipFree(ptr);
  TraceLoggingWrite(rocm_hip, "hipFree", TL_LEVEL, TraceLoggingKeyword(kKwApi),
                    TraceLoggingUInt64(start, "start_timestamp"),
                    TraceLoggingHexUInt64(u64(ptr), "ptr"), TraceLoggingInt32(r, "result"));
  return r;
}

hipError_t probe_hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind) {
  if (!TraceLoggingEventEnabled(rocm_hip, "hipMemcpy"))
    return next_hipMemcpy(dst, src, sizeBytes, kind);
  const uint64_t start = now_ns();
  const hipError_t r = next_hipMemcpy(dst, src, sizeBytes, kind);
  TraceLoggingWrite(rocm_hip, "hipMemcpy", TL_LEVEL, TraceLoggingKeyword(kKwApi),
                    TraceLoggingUInt64(start, "start_timestamp"),
                    TraceLoggingHexUInt64(u64(dst), "dst"), TraceLoggingHexUInt64(u64(src), "src"),
                    TraceLoggingUInt64(sizeBytes, "sizeBytes"),
                    TraceLoggingInt32(static_cast<int32_t>(kind), "kind"),
                    TraceLoggingInt32(r, "result"));
  return r;
}

hipError_t probe_hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                                hipStream_t stream) {
  if (!TraceLoggingEventEnabled(rocm_hip, "hipMemcpyAsync"))
    return next_hipMemcpyAsync(dst, src, sizeBytes, kind, stream);
  const uint64_t start = now_ns();
  const hipError_t r = next_hipMemcpyAsync(dst, src, sizeBytes, kind, stream);
  TraceLoggingWrite(rocm_hip, "hipMemcpyAsync", TL_LEVEL, TraceLoggingKeyword(kKwApi),
                    TraceLoggingUInt64(start, "start_timestamp"),
                    TraceLoggingHexUInt64(u64(dst), "dst"), TraceLoggingHexUInt64(u64(src), "src"),
                    TraceLoggingUInt64(sizeBytes, "sizeBytes"),
                    TraceLoggingInt32(static_cast<int32_t>(kind), "kind"),
                    TraceLoggingHexUInt64(u64(stream), "stream"), TraceLoggingInt32(r, "result"));
  return r;
}

hipError_t probe_hipDeviceSynchronize() {
  if (!TraceLoggingEventEnabled(rocm_hip, "hipDeviceSynchronize"))
    return next_hipDeviceSynchronize();
  const uint64_t start = now_ns();
  const hipError_t r = next_hipDeviceSynchronize();
  TraceLoggingWrite(rocm_hip, "hipDeviceSynchronize", TL_LEVEL, TraceLoggingKeyword(kKwApi),
                    TraceLoggingUInt64(start, "start_timestamp"), TraceLoggingInt32(r, "result"));
  return r;
}

hipError_t probe_hipStreamCreate(hipStream_t* stream) {
  if (!TraceLoggingEventEnabled(rocm_hip, "hipStreamCreate")) return next_hipStreamCreate(stream);
  const uint64_t start = now_ns();
  const hipError_t r = next_hipStreamCreate(stream);
  TraceLoggingWrite(rocm_hip, "hipStreamCreate", TL_LEVEL, TraceLoggingKeyword(kKwApi),
                    TraceLoggingUInt64(start, "start_timestamp"),
                    TraceLoggingHexUInt64(r == hipSuccess && stream ? u64(*stream) : 0, "stream"),
                    TraceLoggingInt32(r, "result"));
  return r;
}

hipError_t probe_hipStreamDestroy(hipStream_t stream) {
  if (!TraceLoggingEventEnabled(rocm_hip, "hipStreamDestroy")) return next_hipStreamDestroy(stream);
  const uint64_t start = now_ns();
  const hipError_t r = next_hipStreamDestroy(stream);
  TraceLoggingWrite(rocm_hip, "hipStreamDestroy", TL_LEVEL, TraceLoggingKeyword(kKwApi),
                    TraceLoggingUInt64(start, "start_timestamp"),
                    TraceLoggingHexUInt64(u64(stream), "stream"), TraceLoggingInt32(r, "result"));
  return r;
}

hipError_t probe_hipStreamSynchronize(hipStream_t stream) {
  if (!TraceLoggingEventEnabled(rocm_hip, "hipStreamSynchronize"))
    return next_hipStreamSynchronize(stream);
  const uint64_t start = now_ns();
  const hipError_t r = next_hipStreamSynchronize(stream);
  TraceLoggingWrite(rocm_hip, "hipStreamSynchronize", TL_LEVEL, TraceLoggingKeyword(kKwApi),
                    TraceLoggingUInt64(start, "start_timestamp"),
                    TraceLoggingHexUInt64(u64(stream), "stream"), TraceLoggingInt32(r, "result"));
  return r;
}

hipError_t probe_hipSetDevice(int deviceId) {
  if (!TraceLoggingEventEnabled(rocm_hip, "hipSetDevice")) return next_hipSetDevice(deviceId);
  const uint64_t start = now_ns();
  const hipError_t r = next_hipSetDevice(deviceId);
  TraceLoggingWrite(rocm_hip, "hipSetDevice", TL_LEVEL, TraceLoggingKeyword(kKwApi),
                    TraceLoggingUInt64(start, "start_timestamp"),
                    TraceLoggingInt32(deviceId, "deviceId"), TraceLoggingInt32(r, "result"));
  return r;
}

hipError_t probe_hipGetDevice(int* deviceId) {
  if (!TraceLoggingEventEnabled(rocm_hip, "hipGetDevice")) return next_hipGetDevice(deviceId);
  const uint64_t start = now_ns();
  const hipError_t r = next_hipGetDevice(deviceId);
  TraceLoggingWrite(rocm_hip, "hipGetDevice", TL_LEVEL, TraceLoggingKeyword(kKwApi),
                    TraceLoggingUInt64(start, "start_timestamp"),
                    TraceLoggingInt32(r == hipSuccess && deviceId ? *deviceId : 0, "deviceId"),
                    TraceLoggingInt32(r, "result"));
  return r;
}

hipError_t probe_hipEventRecord(hipEvent_t event, hipStream_t stream) {
  if (!TraceLoggingEventEnabled(rocm_hip, "hipEventRecord"))
    return next_hipEventRecord(event, stream);
  const uint64_t start = now_ns();
  const hipError_t r = next_hipEventRecord(event, stream);
  TraceLoggingWrite(rocm_hip, "hipEventRecord", TL_LEVEL, TraceLoggingKeyword(kKwApi),
                    TraceLoggingUInt64(start, "start_timestamp"),
                    TraceLoggingHexUInt64(u64(event), "event"),
                    TraceLoggingHexUInt64(u64(stream), "stream"), TraceLoggingInt32(r, "result"));
  return r;
}

hipError_t probe_hipEventSynchronize(hipEvent_t event) {
  if (!TraceLoggingEventEnabled(rocm_hip, "hipEventSynchronize"))
    return next_hipEventSynchronize(event);
  const uint64_t start = now_ns();
  const hipError_t r = next_hipEventSynchronize(event);
  TraceLoggingWrite(rocm_hip, "hipEventSynchronize", TL_LEVEL, TraceLoggingKeyword(kKwApi),
                    TraceLoggingUInt64(start, "start_timestamp"),
                    TraceLoggingHexUInt64(u64(event), "event"), TraceLoggingInt32(r, "result"));
  return r;
}

#define INSTALL(name)                                                                              \
  next_##name = t->name##_fn;                                                                      \
  t->name##_fn = probe_##name

std::atomic<bool> g_registered{false};
}  // namespace

namespace hip_tltrace {

// End of UpdateDispatchTable (HLD 6.1): wrap the entries, so tools that wrap the table later
// keep this probe innermost. ROCM_TRACE_DISABLE=1 installs nothing.
void InstallApiWrappers(HipDispatchTable* t) {
  const char* off = std::getenv("ROCM_TRACE_DISABLE");
  if (off != nullptr && off[0] == '1') return;
  if (!g_registered.exchange(true)) TraceLoggingRegister(rocm_hip);
  INSTALL(hipLaunchKernel);
  INSTALL(hipModuleLaunchKernel);
  INSTALL(hipMalloc);
  INSTALL(hipFree);
  INSTALL(hipMemcpy);
  INSTALL(hipMemcpyAsync);
  INSTALL(hipDeviceSynchronize);
  INSTALL(hipStreamCreate);
  INSTALL(hipStreamDestroy);
  INSTALL(hipStreamSynchronize);
  INSTALL(hipSetDevice);
  INSTALL(hipGetDevice);
  INSTALL(hipEventRecord);
  INSTALL(hipEventSynchronize);
}

bool DispatchEnabled() {
  return g_registered.load(std::memory_order_relaxed) &&
         TraceLoggingEventEnabled(rocm_hip, "kernel_dispatch");
}

// Emitted on the dispatching thread after the packet is written. The POC carries the kernel name
// here instead of a separate kernel_symbol event.
void KernelDispatch(uint64_t queue_id, uint64_t ring_va, uint32_t doorbell_slot,
                    uint64_t packet_index, uint64_t kernel_object, const char* kernel_name,
                    const uint32_t grid[3], const uint32_t workgroup[3]) {
  TraceLoggingWrite(rocm_hip, "kernel_dispatch", TL_LEVEL, TraceLoggingKeyword(kKwDispatch),
                    TraceLoggingUInt64(queue_id, "queue_id"),
                    TraceLoggingHexUInt64(ring_va, "ring_va"),
                    TraceLoggingUInt32(doorbell_slot, "doorbell_slot"),
                    TraceLoggingUInt64(packet_index, "packet_index"),
                    TraceLoggingHexUInt64(kernel_object, "kernel_object"),
                    TraceLoggingString(kernel_name != nullptr ? kernel_name : "", "kernel_name"),
                    TraceLoggingUInt32(grid[0], "grid_x"), TraceLoggingUInt32(grid[1], "grid_y"),
                    TraceLoggingUInt32(grid[2], "grid_z"),
                    TraceLoggingUInt32(workgroup[0], "workgroup_x"),
                    TraceLoggingUInt32(workgroup[1], "workgroup_y"),
                    TraceLoggingUInt32(workgroup[2], "workgroup_z"));
}

}  // namespace hip_tltrace
