/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR API-matrix workloads (direct GPU implementations)
 * @{
 * @ingroup HRRTest
 * Each workload calls a family of HIP APIs that hrr_api_matrix_test.cc captures and replays.
 * Workloads start with hipSetDevice(0) so the first hipMalloc is not the first HIP call, which capture misses.
 */

#include "hrr_test_common.hh"
#include <hip/hiprtc.h>
#include <hip/hip_ext.h>  // hipExtModuleLaunchKernel, hipExtLaunchKernel

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#define HIPRTC_CHECK(expr)                                                    \
  do {                                                                        \
    hiprtcResult _r = (expr);                                                 \
    REQUIRE(_r == HIPRTC_SUCCESS);                                            \
  } while (0)

namespace {

constexpr int    kN  = 1024;
constexpr size_t kSZ = kN * sizeof(int);

// Names differ from hrr_workload_test.cc, whose kernels link into the same HrrTest binary.
__global__ void hrr_mtx_fill(int* out, int val, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = val;
}

__global__ void hrr_mtx_add(const int* a, const int* b, int* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

// HIP_SYMBOL needs a host shadow, which only statically registered variables have, so this cannot be an RTC symbol.
__device__ int hrr_mtx_symbol[4] = {0, 0, 0, 0};

__global__ void hrr_mtx_noargs() {}

__global__ void hrr_mtx_coop(int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = i;
}

const char* kRtcSource = R"(
extern "C" __global__ void mtx_rtc_fill(int* out, int val, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = val;
}
extern "C" __global__ void mtx_rtc_coop(int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = i * 2;
}
__device__ int mtx_rtc_global[64];
)";

std::vector<char> compile_rtc() {
  hiprtcProgram prog = nullptr;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, kRtcSource, "mtx_rtc.hip",
                                   0, nullptr, nullptr));
  hiprtcResult crc = hiprtcCompileProgram(prog, 0, nullptr);
  if (crc != HIPRTC_SUCCESS) {
    size_t log_sz = 0;
    (void)hiprtcGetProgramLogSize(prog, &log_sz);
    std::string log(log_sz, '\0');
    (void)hiprtcGetProgramLog(prog, log.data());
    (void)hiprtcDestroyProgram(&prog);
    FAIL("hiprtcCompileProgram failed: " + log);
  }
  size_t co_size = 0;
  HIPRTC_CHECK(hiprtcGetCodeSize(prog, &co_size));
  std::vector<char> co(co_size);
  HIPRTC_CHECK(hiprtcGetCode(prog, co.data()));
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
  return co;
}

// Payload for host-function and callback APIs. Only the recorded call matters, not the result.
void mtx_host_fn(void* user_data) {
  if (user_data) *static_cast<int*>(user_data) += 1;
}

void mtx_stream_callback(hipStream_t, hipError_t, void* user_data) {
  if (user_data) *static_cast<int*>(user_data) += 1;
}

int device_attr(hipDeviceAttribute_t attr, int device = 0) {
  int value = 0;
  if (hipDeviceGetAttribute(&value, attr, device) != hipSuccess) return 0;
  return value;
}

}  // namespace

TEST_CASE("Unit_HRR_ApiMatrix_UC1Dense_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  {
    int count = 0, dev = 0, rt = 0, drv = 0;
    HRR_HIP_CHECK(hipGetDeviceCount(&count));
    HRR_HIP_CHECK(hipGetDevice(&dev));
    (void)hipRuntimeGetVersion(&rt);
    (void)hipDriverGetVersion(&drv);
    size_t free_bytes = 0, total_bytes = 0;
    HRR_HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
    REQUIRE(total_bytes > 0);
  }

  int *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d_a, kSZ));
  HRR_HIP_CHECK(hipMalloc(&d_b, kSZ));
  HRR_HIP_CHECK(hipMalloc(&d_out, kSZ));

  int* h_pinned = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h_pinned), kSZ,
                          hipHostMallocDefault));
  for (int i = 0; i < kN; ++i) h_pinned[i] = 7;

  std::vector<int> h_reg(kN, 7);
  if (hipHostRegister(h_reg.data(), kSZ, hipHostRegisterDefault) == hipSuccess) {
    void* dev_ptr = nullptr;
    (void)hipHostGetDevicePointer(&dev_ptr, h_reg.data(), 0);
    (void)hipHostUnregister(h_reg.data());
  }

  {
    hipPointerAttribute_t attrs{};
    (void)hipPointerGetAttributes(&attrs, d_a);
  }

  hipStream_t s_default = nullptr, s_prio = nullptr;
  HRR_HIP_CHECK(hipStreamCreate(&s_default));
  hipStream_t s_nb = nullptr;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s_nb, hipStreamNonBlocking));
  {
    int lo = 0, hi = 0;
    (void)hipDeviceGetStreamPriorityRange(&lo, &hi);
    HRR_HIP_CHECK(hipStreamCreateWithPriority(&s_prio, hipStreamDefault, hi));
    int prio = 0;
    unsigned int flags = 0;
    (void)hipStreamGetPriority(s_prio, &prio);
    (void)hipStreamGetFlags(s_prio, &flags);
  }

  hipEvent_t ev_start = nullptr, ev_stop = nullptr, ev_flag = nullptr;
  HRR_HIP_CHECK(hipEventCreate(&ev_start));
  HRR_HIP_CHECK(hipEventCreate(&ev_stop));
  HRR_HIP_CHECK(hipEventCreateWithFlags(&ev_flag, hipEventDisableTiming));

  HRR_HIP_CHECK(hipMemcpy(d_a, h_pinned, kSZ, hipMemcpyHostToDevice));
  HRR_HIP_CHECK(hipMemcpyAsync(d_b, h_pinned, kSZ, hipMemcpyHostToDevice, s_default));
  HRR_HIP_CHECK(hipMemcpyWithStream(d_out, h_pinned, kSZ, hipMemcpyHostToDevice, s_nb));
  HRR_HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(d_a), h_pinned, kSZ));
  HRR_HIP_CHECK(hipMemcpyHtoDAsync(reinterpret_cast<hipDeviceptr_t>(d_b), h_pinned,
                               kSZ, s_default));
  HRR_HIP_CHECK(hipMemcpyDtoD(reinterpret_cast<hipDeviceptr_t>(d_out),
                          reinterpret_cast<hipDeviceptr_t>(d_a), kSZ));
  HRR_HIP_CHECK(hipMemset(d_out, 0, kSZ));
  HRR_HIP_CHECK(hipMemsetAsync(d_out, 0, kSZ, s_default));
  HRR_HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d_out), 0, kN));
  HRR_HIP_CHECK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(d_out), 0, kN,
                              s_default));
  HRR_HIP_CHECK(hipStreamSynchronize(s_default));

  HRR_HIP_CHECK(hipEventRecord(ev_start, s_default));
  hipLaunchKernelGGL(hrr_mtx_add, dim3((kN + 255) / 256), dim3(256), 0, s_default,
                     d_a, d_b, d_out, kN);
  HRR_HIP_CHECK(hipEventRecord(ev_stop, s_default));
  HRR_HIP_CHECK(hipEventSynchronize(ev_stop));
  {
    float ms = 0.f;
    (void)hipEventElapsedTime(&ms, ev_start, ev_stop);
    (void)hipEventQuery(ev_stop);
    (void)hipStreamQuery(s_default);
  }
  HRR_HIP_CHECK(hipStreamWaitEvent(s_nb, ev_stop, 0));

  {
    hipFuncAttributes fattr{};
    (void)hipFuncGetAttributes(&fattr, reinterpret_cast<const void*>(hrr_mtx_add));
  }

  std::vector<char> co = compile_rtc();
  hipModule_t mod = nullptr;
  HRR_HIP_CHECK(hipModuleLoadData(&mod, co.data()));
  hipFunction_t fn = nullptr;
  HRR_HIP_CHECK(hipModuleGetFunction(&fn, mod, "mtx_rtc_fill"));
  {
    int attr_value = 0;
    (void)hipFuncGetAttribute(&attr_value, HIP_FUNC_ATTRIBUTE_NUM_REGS, fn);
  }

  const int blocks = (kN + 255) / 256;
  int  fill_val = 7;
  int  n_arg    = kN;
  void* kargs[] = {&d_out, &fill_val, &n_arg};

  HRR_HIP_CHECK(hipModuleLaunchKernel(fn, blocks, 1, 1, 256, 1, 1, 0, s_default,
                                  kargs, nullptr));

  HRR_HIP_CHECK(hipExtModuleLaunchKernel(fn, kN, 1, 1, 256, 1, 1, 0, s_default,
                                     kargs, nullptr, nullptr, nullptr, 0));

  // Triton's launch path. The config is a const-struct pointer, so a hand-written shim records the descriptor.
  {
    HIP_LAUNCH_CONFIG cfg{};
    cfg.gridDimX = blocks;
    cfg.gridDimY = 1;
    cfg.gridDimZ = 1;
    cfg.blockDimX = 256;
    cfg.blockDimY = 1;
    cfg.blockDimZ = 1;
    cfg.sharedMemBytes = 0;
    cfg.hStream = s_default;
    cfg.attrs = nullptr;
    cfg.numAttrs = 0;
    (void)hipDrvLaunchKernelEx(&cfg, fn, kargs, nullptr);
  }

  // Runtime-API spelling used by CK cluster launches. It loses the const-struct pointer the same way.
  {
    hipLaunchConfig_t cfg{};
    cfg.gridDim = dim3(blocks, 1, 1);
    cfg.blockDim = dim3(256, 1, 1);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = s_default;
    cfg.attrs = nullptr;
    cfg.numAttrs = 0;
    int* out_arg = d_out;
    int  val_arg = 7;
    void* c_args[] = {&out_arg, &val_arg, &n_arg};
    (void)hipLaunchKernelExC(&cfg, reinterpret_cast<const void*>(hrr_mtx_fill),
                             c_args);
  }
  HRR_HIP_CHECK(hipStreamSynchronize(s_default));

  // Replay does not reproduce module lifetime.
  HRR_HIP_CHECK(hipModuleUnload(mod));

  {
    void* sym = nullptr;
    (void)hipGetProcAddress("hipMalloc", &sym, 0, 0, nullptr);
    hipFunction_t by_symbol = nullptr;
    (void)hipGetFuncBySymbol(&by_symbol,
                             reinterpret_cast<const void*>(hrr_mtx_add));
  }

  {
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    hipGraph_t graph = nullptr;
    hipGraphExec_t exec = nullptr;
    int host_fn_hits = 0;

    hipStreamCaptureMode mode = hipStreamCaptureModeThreadLocal;
    (void)hipThreadExchangeStreamCaptureMode(&mode);

    HRR_HIP_CHECK(hipStreamBeginCapture(s_nb, hipStreamCaptureModeThreadLocal));
    hipLaunchKernelGGL(hrr_mtx_add, dim3(blocks), dim3(256), 0, s_nb,
                       d_a, d_b, d_out, kN);
    // CK's FMHA-backward launcher records a host callback into the captured graph.
    // A recorded function pointer cannot be re-executed at replay.
    (void)hipLaunchHostFunc(s_nb, mtx_host_fn, &host_fn_hits);
    (void)hipStreamIsCapturing(s_nb, &status);
    {
      unsigned long long capture_id = 0;
      (void)hipStreamGetCaptureInfo(s_nb, &status, &capture_id);
    }
    HRR_HIP_CHECK(hipStreamEndCapture(s_nb, &graph));
    (void)hipThreadExchangeStreamCaptureMode(&mode);

    REQUIRE(graph != nullptr);
    HRR_HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    HRR_HIP_CHECK(hipGraphLaunch(exec, s_nb));
    HRR_HIP_CHECK(hipStreamSynchronize(s_nb));
    HRR_HIP_CHECK(hipGraphExecDestroy(exec));
    HRR_HIP_CHECK(hipGraphDestroy(graph));
  }

  HRR_HIP_CHECK(hipDeviceSynchronize());
  std::vector<int> h_out(kN, 0);
  HRR_HIP_CHECK(hipMemcpy(h_out.data(), d_out, kSZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < kN; ++i) REQUIRE(h_out[i] == 14);
  HRR_HIP_CHECK(hipMemcpyDtoH(h_out.data(), reinterpret_cast<hipDeviceptr_t>(d_out),
                          kSZ));
  HRR_HIP_CHECK(hipMemcpyDtoHAsync(h_out.data(),
                               reinterpret_cast<hipDeviceptr_t>(d_out), kSZ,
                               s_default));
  HRR_HIP_CHECK(hipStreamSynchronize(s_default));

  HRR_HIP_CHECK(hipEventDestroy(ev_flag));
  HRR_HIP_CHECK(hipEventDestroy(ev_stop));
  HRR_HIP_CHECK(hipEventDestroy(ev_start));
  HRR_HIP_CHECK(hipStreamDestroy(s_prio));
  HRR_HIP_CHECK(hipStreamDestroy(s_nb));
  HRR_HIP_CHECK(hipStreamDestroy(s_default));
  HRR_HIP_CHECK(hipHostFree(h_pinned));
  HRR_HIP_CHECK(hipFree(d_out));
  HRR_HIP_CHECK(hipFree(d_b));
  HRR_HIP_CHECK(hipFree(d_a));
}

TEST_CASE("Unit_HRR_ApiMatrix_PayloadLoss_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, kSZ));
  hipStream_t s = nullptr;
  HRR_HIP_CHECK(hipStreamCreate(&s));
  hipLaunchKernelGGL(hrr_mtx_fill, dim3((kN + 255) / 256), dim3(256), 0, s,
                     d, 3, kN);
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  // hipIpcMemHandle_t is HIP_IPC_HANDLE_SIZE (64) bytes passed by value, and
  // capture does not record it. HIP rejects an open in the owning process, and
  // capture records hipIpcOpenMemHandle only on hipSuccess, so the matrix
  // expects no event for it.
  {
    hipIpcMemHandle_t mem_handle{};
    std::memset(&mem_handle, 0xA5, sizeof(mem_handle));
    if (hipIpcGetMemHandle(&mem_handle, d) == hipSuccess) {
      void* opened = nullptr;
      hipError_t open_rc = hipIpcOpenMemHandle(&opened, mem_handle,
                                               hipIpcMemLazyEnablePeerAccess);
      if (open_rc == hipSuccess && opened != nullptr)
        (void)hipIpcCloseMemHandle(opened);
    }
  }
  {
    hipEvent_t ipc_event = nullptr;
    if (hipEventCreateWithFlags(&ipc_event, hipEventInterprocess |
                                            hipEventDisableTiming) == hipSuccess) {
      hipIpcEventHandle_t event_handle{};
      if (hipIpcGetEventHandle(&event_handle, ipc_event) == hipSuccess) {
        hipEvent_t reopened = nullptr;
        // A same-process open is usually rejected. If it succeeds, release the event as with the IPC mem handle above.
        if (hipIpcOpenEventHandle(&reopened, event_handle) == hipSuccess &&
            reopened != nullptr)
          (void)hipEventDestroy(reopened);
      }
      (void)hipEventDestroy(ipc_event);
    }
  }

  // hipMemAllocationProp is carried inline, and replay allocates with the recorded type and location.
  if (device_attr(hipDeviceAttributeVirtualMemoryManagementSupported)) {
    hipMemAllocationProp prop{};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = 0;

    size_t granularity = 0;
    if (hipMemGetAllocationGranularity(&granularity, &prop,
                                       hipMemAllocationGranularityRecommended)
            == hipSuccess && granularity > 0) {
      hipMemGenericAllocationHandle_t handle{};
      if (hipMemCreate(&handle, granularity, &prop, 0) == hipSuccess) {
        hipMemAllocationProp read_back{};
        (void)hipMemGetAllocationPropertiesFromHandle(&read_back, handle);
        (void)hipMemRelease(handle);
      }
    }
  }

  // The op array is a non-const pointer, so derive_manifest.py does not flag it; api_matrix.yaml marks it by hand.
  // The ops are carried inline and replay translates their device addresses.
  {
    uint32_t* flag = nullptr;
    if (hipHostMalloc(reinterpret_cast<void**>(&flag), sizeof(uint32_t),
                      hipHostMallocDefault) == hipSuccess) {
      *flag = 0;
      void* flag_dev = nullptr;
      (void)hipHostGetDevicePointer(&flag_dev, flag, 0);

      hipStreamBatchMemOpParams ops[1]{};
      ops[0].operation = hipStreamMemOpWriteValue32;
      ops[0].writeValue.operation = hipStreamMemOpWriteValue32;
      ops[0].writeValue.address =
          reinterpret_cast<hipDeviceptr_t>(flag_dev ? flag_dev : flag);
      ops[0].writeValue.value = 0xC0FFEEu;
      ops[0].writeValue.flags = 0;
      (void)hipStreamBatchMemOp(s, 1, ops, 0);
      (void)hipStreamSynchronize(s);
      (void)hipHostFree(flag);
    }
  }

  // No portable handle exists to import, so both calls fail. Capture records only hipSuccess, so no event is expected.
  {
    hipExternalMemoryHandleDesc mem_desc{};
    mem_desc.type = hipExternalMemoryHandleTypeOpaqueFd;
    mem_desc.handle.fd = -1;
    mem_desc.size = kSZ;
    hipExternalMemory_t ext_mem{};
    if (hipImportExternalMemory(&ext_mem, &mem_desc) == hipSuccess) {
      hipExternalMemoryBufferDesc buf_desc{};
      buf_desc.offset = 0;
      buf_desc.size = kSZ;
      void* mapped = nullptr;
      (void)hipExternalMemoryGetMappedBuffer(&mapped, ext_mem, &buf_desc);
      (void)hipDestroyExternalMemory(ext_mem);
    }

    hipExternalSemaphoreHandleDesc sem_desc{};
    sem_desc.type = hipExternalSemaphoreHandleTypeOpaqueFd;
    sem_desc.handle.fd = -1;
    hipExternalSemaphore_t ext_sem{};
    if (hipImportExternalSemaphore(&ext_sem, &sem_desc) == hipSuccess) {
      hipExternalSemaphoreSignalParams sig{};
      hipExternalSemaphoreWaitParams wait{};
      (void)hipSignalExternalSemaphoresAsync(&ext_sem, &sig, 1, s);
      (void)hipWaitExternalSemaphoresAsync(&ext_sem, &wait, 1, s);
      (void)hipDestroyExternalSemaphore(ext_sem);
    }
  }

  // hipLaunchParams carries the kernel as a host function address: the payload-loss shape, in the launch family least likely to survive it.
  {
    int fill_value = 3;
    int count = kN;
    void* args[] = {&d, &fill_value, &count};
    hipLaunchParams launch_params{};
    launch_params.func = reinterpret_cast<void*>(hrr_mtx_fill);
    launch_params.gridDim = dim3(1);
    launch_params.blockDim = dim3(64);
    launch_params.args = args;
    launch_params.sharedMem = 0;
    launch_params.stream = s;
    (void)hipLaunchCooperativeKernelMultiDevice(&launch_params, 1, 0);
    (void)hipStreamSynchronize(s);
  }

  HRR_HIP_CHECK(hipDeviceSynchronize());
  std::vector<int> h(kN, 0);
  HRR_HIP_CHECK(hipMemcpy(h.data(), d, kSZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < kN; ++i) REQUIRE(h[i] == 3);

  HRR_HIP_CHECK(hipStreamDestroy(s));
  HRR_HIP_CHECK(hipFree(d));
}

TEST_CASE("Unit_HRR_ApiMatrix_SilentFailure_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, kSZ));
  hipStream_t s = nullptr;
  HRR_HIP_CHECK(hipStreamCreate(&s));

  {
    void* p = nullptr;
    if (hipHostAlloc(&p, kSZ, hipHostMallocDefault) == hipSuccess && p) {
      unsigned int flags = 0;
      (void)hipHostGetFlags(&flags, p);
      (void)hipFreeHost(p);
    }
    void* mh = nullptr;
    if (hipMallocHost(&mh, kSZ) == hipSuccess && mh) (void)hipFreeHost(mh);
    void* ah = nullptr;
    if (hipMemAllocHost(&ah, kSZ) == hipSuccess && ah) (void)hipFreeHost(ah);
  }

  {
    std::vector<char> co = compile_rtc();
    hipModule_t mod = nullptr;
    if (hipModuleLoadData(&mod, co.data()) == hipSuccess) {
      hipDeviceptr_t sym = 0;
      size_t sym_bytes = 0;
      // No-op at replay: the replayed module never yields the symbol address, so writes through it land nowhere.
      (void)hipModuleGetGlobal(&sym, &sym_bytes, mod, "mtx_rtc_global");
      (void)hipModuleUnload(mod);
    }
  }

  // hipMemcpyToSymbol initialises a __device__ global. If replay makes it a no-op, kernels read zeros and give plausible wrong results.
  {
    const int host_seed[4] = {11, 22, 33, 44};
    int read_back[4] = {0, 0, 0, 0};

    void* sym_addr = nullptr;
    size_t sym_size = 0;
    (void)hipGetSymbolAddress(&sym_addr, HIP_SYMBOL(hrr_mtx_symbol));
    (void)hipGetSymbolSize(&sym_size, HIP_SYMBOL(hrr_mtx_symbol));

    (void)hipMemcpyToSymbol(HIP_SYMBOL(hrr_mtx_symbol), host_seed,
                            sizeof(host_seed), 0, hipMemcpyHostToDevice);
    (void)hipMemcpyFromSymbol(read_back, HIP_SYMBOL(hrr_mtx_symbol),
                              sizeof(read_back), 0, hipMemcpyDeviceToHost);
    (void)hipMemcpyToSymbolAsync(HIP_SYMBOL(hrr_mtx_symbol), host_seed,
                                 sizeof(host_seed), 0, hipMemcpyHostToDevice, s);
    (void)hipMemcpyFromSymbolAsync(read_back, HIP_SYMBOL(hrr_mtx_symbol),
                                   sizeof(read_back), 0, hipMemcpyDeviceToHost,
                                   s);
    (void)hipStreamSynchronize(s);
  }

  // If replay made the write a no-op, a real waiter would never be satisfied.
  // A pinned host flag keeps the capture run from hanging.
  {
    uint32_t* flag32 = nullptr;
    if (hipHostMalloc(reinterpret_cast<void**>(&flag32), sizeof(uint64_t),
                      hipHostMallocDefault) == hipSuccess) {
      *flag32 = 0;
      void* flag_dev = nullptr;
      (void)hipHostGetDevicePointer(&flag_dev, flag32, 0);
      hipDeviceptr_t addr =
          reinterpret_cast<hipDeviceptr_t>(flag_dev ? flag_dev : flag32);

      (void)hipStreamWriteValue32(s, addr, 1u, 0);
      (void)hipStreamWaitValue32(s, addr, 1u, hipStreamWaitValueGte, 0xFFFFFFFFu);
      (void)hipStreamWriteValue64(s, addr, 1ull, 0);
      (void)hipStreamWaitValue64(s, addr, 1ull, hipStreamWaitValueGte,
                                 0xFFFFFFFFFFFFFFFFull);
      (void)hipStreamSynchronize(s);
      (void)hipHostFree(flag32);
    }
  }

  // MIOpen's Winograd Fury RxS solver uses this on Instinct.
  if (device_attr(hipDeviceAttributeCooperativeLaunch)) {
    std::vector<char> co = compile_rtc();
    hipModule_t mod = nullptr;
    if (hipModuleLoadData(&mod, co.data()) == hipSuccess) {
      hipFunction_t coop_fn = nullptr;
      if (hipModuleGetFunction(&coop_fn, mod, "mtx_rtc_coop") == hipSuccess) {
        int n_arg = kN;
        void* coop_args[] = {&d, &n_arg};
        (void)hipModuleLaunchCooperativeKernel(coop_fn, 1, 1, 1, 64, 1, 1, 0, s,
                                               coop_args);
        (void)hipStreamSynchronize(s);
      }
      (void)hipModuleUnload(mod);
    }
  }

  // Mutate the instantiated graph as llama.cpp does per token. Not every spelling is a no-op at replay.
  {
    hipGraph_t graph = nullptr;
    hipGraphExec_t exec = nullptr;
    int* d_copy = nullptr;
    HRR_HIP_CHECK(hipMalloc(&d_copy, kSZ));
    int host_hits = 0;

    HRR_HIP_CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));
    HRR_HIP_CHECK(hipMemsetAsync(d, 0, kSZ, s));
    hipLaunchKernelGGL(hrr_mtx_fill, dim3((kN + 255) / 256), dim3(256), 0, s,
                       d, 5, kN);
    // A memcpy node and a host node give the mutation loop one node of each kind to set params on.
    HRR_HIP_CHECK(hipMemcpyAsync(d_copy, d, kSZ, hipMemcpyDeviceToDevice, s));
    (void)hipLaunchHostFunc(s, mtx_host_fn, &host_hits);
    HRR_HIP_CHECK(hipStreamEndCapture(s, &graph));
    REQUIRE(graph != nullptr);
    HRR_HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    size_t node_count = 0;
    (void)hipGraphGetNodes(graph, nullptr, &node_count);
    std::vector<hipGraphNode_t> nodes(node_count ? node_count : 1, nullptr);
    if (node_count) (void)hipGraphGetNodes(graph, nodes.data(), &node_count);

    for (size_t i = 0; i < node_count; ++i) {
      hipGraphNodeType type{};
      if (hipGraphNodeGetType(nodes[i], &type) != hipSuccess) continue;

      if (type == hipGraphNodeTypeKernel) {
        hipKernelNodeParams kp{};
        if (hipGraphKernelNodeGetParams(nodes[i], &kp) == hipSuccess) {
          (void)hipGraphKernelNodeSetParams(nodes[i], &kp);
          (void)hipGraphExecKernelNodeSetParams(exec, nodes[i], &kp);

          // The generic spelling has its own NOOP handler, so it needs its own call site.
          // No getter exists for the union, so build it from kp. It must be valid: shims record only hipSuccess.
          hipGraphNodeParams np{};
          np.type = hipGraphNodeTypeKernel;
          np.kernel = kp;
          (void)hipGraphNodeSetParams(nodes[i], &np);
          (void)hipGraphExecNodeSetParams(exec, nodes[i], &np);
        }
      } else if (type == hipGraphNodeTypeMemset) {
        hipMemsetParams mp{};
        if (hipGraphMemsetNodeGetParams(nodes[i], &mp) == hipSuccess) {
          (void)hipGraphMemsetNodeSetParams(nodes[i], &mp);
          (void)hipGraphExecMemsetNodeSetParams(exec, nodes[i], &mp);
        }
      } else if (type == hipGraphNodeTypeMemcpy) {
        hipMemcpy3DParms cp{};
        if (hipGraphMemcpyNodeGetParams(nodes[i], &cp) == hipSuccess) {
          (void)hipGraphMemcpyNodeSetParams(nodes[i], &cp);
          (void)hipGraphExecMemcpyNodeSetParams(exec, nodes[i], &cp);
        }
        // The 1D spelling of the same runtime memcpy node. hipDrvGraphMemcpyNodeSetParams is not used:
        // HIP rejects it unless the node came from hipDrvGraphAddMemcpyNode, which stream capture never does.
        (void)hipGraphMemcpyNodeSetParams1D(nodes[i], d_copy, d, kSZ,
                                            hipMemcpyDeviceToDevice);
        (void)hipGraphExecMemcpyNodeSetParams1D(exec, nodes[i], d_copy, d, kSZ,
                                                hipMemcpyDeviceToDevice);
      } else if (type == hipGraphNodeTypeHost) {
        hipHostNodeParams hp{};
        if (hipGraphHostNodeGetParams(nodes[i], &hp) == hipSuccess) {
          (void)hipGraphHostNodeSetParams(nodes[i], &hp);
          (void)hipGraphExecHostNodeSetParams(exec, nodes[i], &hp);
        }
      }
    }

    {
      hipGraphExecUpdateResult update_result{};
      hipGraphNode_t error_node = nullptr;
      (void)hipGraphExecUpdate(exec, graph, &error_node, &update_result);
    }

    // Stream capture never produces batch-memop, driver-memset or driver-memcpy nodes, so build them in a second graph.
    // Instantiate it so the exec-level spellings have an exec to mutate.
    // The constructors reject a null hipCtx_t, hence the context.
    {
      hipDevice_t mutation_device = 0;
      (void)hipDeviceGet(&mutation_device, 0);
      hipCtx_t mutation_ctx = nullptr;
      (void)hipCtxCreate(&mutation_ctx, 0, mutation_device);

      hipGraph_t built = nullptr;
      if (hipGraphCreate(&built, 0) == hipSuccess) {
        hipStreamBatchMemOpParams memop{};
        memop.waitValue.operation = hipStreamMemOpWaitValue32;
        memop.waitValue.address = reinterpret_cast<hipDeviceptr_t>(d);
        memop.waitValue.value = 0;
        memop.waitValue.flags = hipStreamWaitValueGte;
        hipBatchMemOpNodeParams batch_params{};
        batch_params.ctx = mutation_ctx;
        batch_params.count = 1;
        batch_params.paramArray = &memop;
        hipGraphNode_t batch_node = nullptr;
        const bool have_batch =
            hipGraphAddBatchMemOpNode(&batch_node, built, nullptr, 0,
                                      &batch_params) == hipSuccess;
        if (have_batch)
          (void)hipGraphBatchMemOpNodeSetParams(batch_node, &batch_params);

        hipMemsetParams memset_params{};
        memset_params.dst = d;
        memset_params.elementSize = sizeof(int);
        memset_params.width = 16;
        memset_params.height = 1;
        memset_params.value = 0;
        hipGraphNode_t drv_memset_node = nullptr;
        const bool have_memset =
            hipDrvGraphAddMemsetNode(&drv_memset_node, built, nullptr, 0,
                                     &memset_params, mutation_ctx)
            == hipSuccess;

        HIP_MEMCPY3D drv_copy{};
        drv_copy.srcMemoryType = hipMemoryTypeDevice;
        drv_copy.srcDevice = reinterpret_cast<hipDeviceptr_t>(d);
        drv_copy.srcPitch = kSZ;
        drv_copy.srcHeight = 1;
        drv_copy.dstMemoryType = hipMemoryTypeDevice;
        drv_copy.dstDevice = reinterpret_cast<hipDeviceptr_t>(d_copy);
        drv_copy.dstPitch = kSZ;
        drv_copy.dstHeight = 1;
        drv_copy.WidthInBytes = kSZ;
        drv_copy.Height = 1;
        drv_copy.Depth = 1;
        hipGraphNode_t drv_memcpy_node = nullptr;
        const bool have_drv_memcpy =
            hipDrvGraphAddMemcpyNode(&drv_memcpy_node, built, nullptr, 0,
                                     &drv_copy, mutation_ctx) == hipSuccess;
        if (have_drv_memcpy)
          (void)hipDrvGraphMemcpyNodeSetParams(drv_memcpy_node, &drv_copy);

        hipGraphExec_t built_exec = nullptr;
        if (hipGraphInstantiate(&built_exec, built, nullptr, nullptr, 0)
                == hipSuccess) {
          if (have_batch)
            (void)hipGraphExecBatchMemOpNodeSetParams(built_exec, batch_node,
                                                      &batch_params);
          if (have_memset)
            (void)hipDrvGraphExecMemsetNodeSetParams(
                built_exec, drv_memset_node, &memset_params, mutation_ctx);
          if (have_drv_memcpy)
            (void)hipDrvGraphExecMemcpyNodeSetParams(
                built_exec, drv_memcpy_node, &drv_copy, mutation_ctx);
          (void)hipGraphExecDestroy(built_exec);
        }
        (void)hipGraphDestroy(built);
      }
      if (mutation_ctx) (void)hipCtxDestroy(mutation_ctx);
    }

    HRR_HIP_CHECK(hipGraphLaunch(exec, s));
    HRR_HIP_CHECK(hipStreamSynchronize(s));
    HRR_HIP_CHECK(hipGraphExecDestroy(exec));
    HRR_HIP_CHECK(hipGraphDestroy(graph));
    HRR_HIP_CHECK(hipFree(d_copy));
  }

  HRR_HIP_CHECK(hipDeviceSynchronize());
  std::vector<int> h(kN, 0);
  HRR_HIP_CHECK(hipMemcpy(h.data(), d, kSZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < kN; ++i) REQUIRE(h[i] == 5);

  HRR_HIP_CHECK(hipStreamDestroy(s));
  HRR_HIP_CHECK(hipFree(d));
}

TEST_CASE("Unit_HRR_ApiMatrix_MultiGpu_Direct", "[.][hrr-direct]") {
  int ndev = 0;
  HRR_HIP_CHECK(hipGetDeviceCount(&ndev));
  if (ndev < 2) HRR_SKIP("workload requires at least two visible GPUs");

  constexpr int kVal = 0x5A5A5A5A;
  const int src_dev = 0;
  const int dst_dev = 1;

  HRR_HIP_CHECK(hipSetDevice(src_dev));

  int can_access = 0;
  HRR_HIP_CHECK(hipDeviceCanAccessPeer(&can_access, src_dev, dst_dev));
  {
    int p2p_value = 0;
    (void)hipDeviceGetP2PAttribute(&p2p_value, hipDevP2PAttrAccessSupported,
                                   src_dev, dst_dev);
    uint32_t link_type = 0, hop_count = 0;
    (void)hipExtGetLinkTypeAndHopCount(src_dev, dst_dev, &link_type, &hop_count);
  }

  int* d0 = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d0, kSZ));
  HRR_HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d0), kVal, kN));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  HRR_HIP_CHECK(hipSetDevice(dst_dev));
  int* d1 = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d1, kSZ));
  HRR_HIP_CHECK(hipMemset(d1, 0, kSZ));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  HRR_HIP_CHECK(hipSetDevice(src_dev));
  const bool peer_enabled =
      can_access && hipDeviceEnablePeerAccess(dst_dev, 0) == hipSuccess;

  HRR_HIP_CHECK(hipMemcpyPeer(d1, dst_dev, d0, src_dev, kSZ));
  HRR_HIP_CHECK(hipDeviceSynchronize());
  {
    hipStream_t peer_stream = nullptr;
    HRR_HIP_CHECK(hipStreamCreate(&peer_stream));
    (void)hipMemcpyPeerAsync(d1, dst_dev, d0, src_dev, kSZ, peer_stream);
    (void)hipStreamSynchronize(peer_stream);
    HRR_HIP_CHECK(hipStreamDestroy(peer_stream));
  }

  // hipMemCreate's properties are carried inline, so replay uses the recorded device, not device 0.
  if (device_attr(hipDeviceAttributeVirtualMemoryManagementSupported, src_dev)) {
    hipMemAllocationProp prop{};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = src_dev;

    size_t granularity = 0;
    if (hipMemGetAllocationGranularity(&granularity, &prop,
                                       hipMemAllocationGranularityRecommended)
            == hipSuccess && granularity > 0) {
      const size_t vmm_size = granularity;
      hipDeviceptr_t va = 0;
      hipMemGenericAllocationHandle_t handle{};
      if (hipMemAddressReserve(&va, vmm_size, 0, 0, 0) == hipSuccess) {
        if (hipMemCreate(&handle, vmm_size, &prop, 0) == hipSuccess) {
          if (hipMemMap(va, vmm_size, 0, handle, 0) == hipSuccess) {
            hipMemAccessDesc access{};
            access.location.type = hipMemLocationTypeDevice;
            access.location.id = src_dev;
            access.flags = hipMemAccessFlagsProtReadWrite;
            (void)hipMemSetAccess(va, vmm_size, &access, 1);

            unsigned long long access_flags = 0;
            (void)hipMemGetAccess(&access_flags, &access.location, va);
            (void)hipMemsetD32(va, kVal, static_cast<int>(vmm_size /
                                                          sizeof(int)));
            (void)hipDeviceSynchronize();
            (void)hipMemUnmap(va, vmm_size);
          }
          (void)hipMemRelease(handle);
        }
        (void)hipMemAddressFree(va, vmm_size);
      }
    }
  }


  HRR_HIP_CHECK(hipSetDevice(dst_dev));
  std::vector<int> h(kN, 0);
  HRR_HIP_CHECK(hipMemcpy(h.data(), d1, kSZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < kN; ++i) REQUIRE(h[i] == kVal);

  HRR_HIP_CHECK(hipFree(d1));
  HRR_HIP_CHECK(hipSetDevice(src_dev));
  if (peer_enabled) (void)hipDeviceDisablePeerAccess(dst_dev);
  HRR_HIP_CHECK(hipFree(d0));
}

TEST_CASE("Unit_HRR_ApiMatrix_ShareableHandle_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  if (!device_attr(hipDeviceAttributeVirtualMemoryManagementSupported, 0))
    HRR_SKIP("virtual memory management is unsupported on this device");

  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, kSZ));
  HRR_HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d), 0x11, kN));

  // dmabuf export of an ordinary allocation, as in MoRI's RDMA path.
  // hipMemGetHandleForAddressRange is absent from amdhip.def.in, so HrrTest.exe cannot link it on Windows.
#ifndef _WIN32
  {
    int range_handle = -1;
    (void)hipMemGetHandleForAddressRange(
        &range_handle, reinterpret_cast<hipDeviceptr_t>(d), kSZ,
        hipMemRangeHandleTypeDmaBufFd, 0);
  }
#endif

  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = 0;
  // Without an exportable handle type the export fails, and failed calls are not recorded.
  // The whole family would then drop out of the matrix.
  prop.requestedHandleType = hipMemHandleTypePosixFileDescriptor;

  size_t granularity = 0;
  if (hipMemGetAllocationGranularity(&granularity, &prop,
                                     hipMemAllocationGranularityRecommended)
          == hipSuccess && granularity > 0) {
    hipDeviceptr_t va = 0;
    hipMemGenericAllocationHandle_t handle{};
    if (hipMemAddressReserve(&va, granularity, 0, 0, 0) == hipSuccess) {
      if (hipMemCreate(&handle, granularity, &prop, 0) == hipSuccess) {
        if (hipMemMap(va, granularity, 0, handle, 0) == hipSuccess) {
          hipMemAccessDesc access{};
          access.location.type = hipMemLocationTypeDevice;
          access.location.id = 0;
          access.flags = hipMemAccessFlagsProtReadWrite;
          (void)hipMemSetAccess(va, granularity, &access, 1);

          hipMemGenericAllocationHandle_t retained{};
          (void)hipMemRetainAllocationHandle(&retained,
                                             reinterpret_cast<void*>(va));

          hipMemAllocationProp read_back{};
          (void)hipMemGetAllocationPropertiesFromHandle(&read_back, handle);

          (void)hipMemUnmap(va, granularity);
        }

        int fd = -1;
        if (hipMemExportToShareableHandle(
                &fd, handle, hipMemHandleTypePosixFileDescriptor, 0)
                == hipSuccess) {
          hipMemGenericAllocationHandle_t imported{};
          (void)hipMemImportFromShareableHandle(
              &imported, reinterpret_cast<void*>(static_cast<intptr_t>(fd)),
              hipMemHandleTypePosixFileDescriptor);
        }
        (void)hipMemRelease(handle);
      }
      (void)hipMemAddressFree(va, granularity);
    }
  }

  HRR_HIP_CHECK(hipDeviceSynchronize());
  HRR_HIP_CHECK(hipFree(d));
}

TEST_CASE("Unit_HRR_ApiMatrix_Breadth_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, kSZ));
  hipStream_t s = nullptr;
  HRR_HIP_CHECK(hipStreamCreate(&s));

  {
    (void)hipGetLastError();
    (void)hipPeekAtLastError();
    (void)hipExtGetLastError();
    (void)hipGetErrorName(hipSuccess);
    (void)hipGetErrorString(hipSuccess);
    const char* drv_name = nullptr;
    const char* drv_str = nullptr;
    (void)hipDrvGetErrorName(hipSuccess, &drv_name);
    (void)hipDrvGetErrorString(hipSuccess, &drv_str);
    (void)hipApiName(0);
  }

  {
    int stream_dev = 0;
    unsigned long long stream_id = 0;
    hipDevice_t dev_of_stream = 0;
    (void)hipGetStreamDeviceId(s);
    (void)hipStreamGetId(s, &stream_id);
    (void)hipStreamGetDevice(s, &dev_of_stream);
    (void)stream_dev;

    hipStream_t copy_to = nullptr;
    if (hipStreamCreate(&copy_to) == hipSuccess) {
      (void)hipStreamCopyAttributes(copy_to, s);
      (void)hipStreamDestroy(copy_to);
    }
  }

  // hipExtStreamGetCUMask rejects a buffer smaller than one bit per CU, so size the mask from the device.
  // A fixed 8 words misses gfx94X's 304 CUs, and a rejected call is not recorded.
  {
    const uint32_t cu_count =
        static_cast<uint32_t>(device_attr(hipDeviceAttributeMultiprocessorCount));
    const uint32_t mask_words = (cu_count + 31) / 32;
    hipStream_t cu_stream = nullptr;
    std::vector<uint32_t> cu_mask(mask_words ? mask_words : 1, 0xFFFFFFFFu);
    if (hipExtStreamCreateWithCUMask(&cu_stream,
                                     static_cast<uint32_t>(cu_mask.size()),
                                     cu_mask.data()) == hipSuccess) {
      std::vector<uint32_t> read_back(cu_mask.size(), 0);
      (void)hipExtStreamGetCUMask(cu_stream,
                                  static_cast<uint32_t>(read_back.size()),
                                  read_back.data());
      (void)hipStreamDestroy(cu_stream);
    }
  }

  // Both record a host function pointer that means nothing in the replaying process.
  {
    int callback_hits = 0;
    (void)hipStreamAddCallback(s, mtx_stream_callback, &callback_hits, 0);
    (void)hipLaunchHostFunc(s, mtx_host_fn, &callback_hits);
    (void)hipStreamSynchronize(s);
  }

  {
    hipUserObject_t obj = nullptr;
    int payload = 0;
    if (hipUserObjectCreate(&obj, &payload, mtx_host_fn, 1,
                            hipUserObjectNoDestructorSync) == hipSuccess) {
      (void)hipUserObjectRetain(obj, 1);
      (void)hipUserObjectRelease(obj, 1);
      (void)hipUserObjectRelease(obj, 1);
    }
  }

  {
    std::vector<char> co = compile_rtc();
    hipLibrary_t lib = nullptr;
    if (hipLibraryLoadData(&lib, co.data(), nullptr, nullptr, 0, nullptr,
                           nullptr, 0) == hipSuccess) {
      unsigned int kernel_count = 0;
      (void)hipLibraryGetKernelCount(&kernel_count, lib);
      hipKernel_t kernel = nullptr;
      if (hipLibraryGetKernel(&kernel, lib, "mtx_rtc_fill") == hipSuccess) {
        int attr_value = 0;
        (void)hipKernelGetAttribute(&attr_value,
                                    HIP_FUNC_ATTRIBUTE_NUM_REGS, kernel, 0);
        (void)hipKernelGetName(nullptr, kernel);
        hipFunction_t as_function = nullptr;
        (void)hipKernelGetFunction(&as_function, kernel);
        hipLibrary_t owning = nullptr;
        (void)hipKernelGetLibrary(&owning, kernel);
      }
      // hipLibraryGetGlobal and hipLibraryGetManaged are in the dispatch table but not in the SDK headers, so they have no call site.
      (void)hipLibraryUnload(lib);
    }
  }

  {
    std::vector<char> co = compile_rtc();
    hipModule_t mod = nullptr;
    if (hipModuleLoadData(&mod, co.data()) == hipSuccess) {
      unsigned int fn_count = 0;
      (void)hipModuleGetFunctionCount(&fn_count, mod);
      (void)hipModuleUnload(mod);
    }
  }

  {
    hipLinkState_t link_state = nullptr;
    if (hipLinkCreate(0, nullptr, nullptr, &link_state) == hipSuccess) {
      std::vector<char> co = compile_rtc();
      (void)hipLinkAddData(link_state, hipJitInputLLVMBundledBitcode,
                           co.data(), co.size(), "mtx", 0, nullptr, nullptr);
      void* linked = nullptr;
      size_t linked_size = 0;
      (void)hipLinkComplete(link_state, &linked, &linked_size);
      (void)hipLinkDestroy(link_state);
    }
  }

  // The green-context family (hipDeviceGetDevResource, hipGreenCtxCreate, hipExecutionCtx*) is in the dispatch table
  // but not in the SDK headers, so it has no call site.

  (void)hipProfilerStart();
  (void)hipProfilerStop();

  {
    HRR_HIP_CHECK(hipMemsetAsync(d, 0, kSZ, hipStreamPerThread));
    hipLaunchKernelGGL(hrr_mtx_fill, dim3((kN + 255) / 256), dim3(256), 0,
                       hipStreamPerThread, d, 11, kN);
    HRR_HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
  }

  HRR_HIP_CHECK(hipDeviceSynchronize());
  std::vector<int> h(kN, 0);
  HRR_HIP_CHECK(hipMemcpy(h.data(), d, kSZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < kN; ++i) REQUIRE(h[i] == 11);

  HRR_HIP_CHECK(hipStreamDestroy(s));
  HRR_HIP_CHECK(hipFree(d));
}

TEST_CASE("Unit_HRR_ApiMatrix_GraphNodes_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  int* d = nullptr;
  int* d_copy = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, kSZ));
  HRR_HIP_CHECK(hipMalloc(&d_copy, kSZ));
  hipStream_t s = nullptr;
  HRR_HIP_CHECK(hipStreamCreate(&s));
  hipEvent_t ev_a = nullptr, ev_b = nullptr;
  HRR_HIP_CHECK(hipEventCreateWithFlags(&ev_a, hipEventDisableTiming));
  HRR_HIP_CHECK(hipEventCreateWithFlags(&ev_b, hipEventDisableTiming));

  hipGraph_t graph = nullptr;
  HRR_HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t empty_node = nullptr;
  (void)hipGraphAddEmptyNode(&empty_node, graph, nullptr, 0);

  int fill_val = 17;
  int n_arg = kN;
  void* kernel_args[] = {&d, &fill_val, &n_arg};
  hipKernelNodeParams knp{};
  knp.func = reinterpret_cast<void*>(hrr_mtx_fill);
  knp.gridDim = dim3((kN + 255) / 256);
  knp.blockDim = dim3(256);
  knp.sharedMemBytes = 0;
  knp.kernelParams = kernel_args;
  knp.extra = nullptr;
  hipGraphNode_t kernel_node = nullptr;
  (void)hipGraphAddKernelNode(&kernel_node, graph, &empty_node, 1, &knp);

  hipMemsetParams msp{};
  msp.dst = d_copy;
  msp.elementSize = sizeof(int);
  msp.value = 0;
  msp.width = kN;
  msp.height = 1;
  hipGraphNode_t memset_node = nullptr;
  (void)hipGraphAddMemsetNode(&memset_node, graph, nullptr, 0, &msp);

  hipGraphNode_t memcpy1d_node = nullptr;
  (void)hipGraphAddMemcpyNode1D(&memcpy1d_node, graph, &kernel_node, 1,
                                d_copy, d, kSZ, hipMemcpyDeviceToDevice);

  hipMemcpy3DParms cp3{};
  cp3.srcPtr = make_hipPitchedPtr(d, kSZ, kN, 1);
  cp3.dstPtr = make_hipPitchedPtr(d_copy, kSZ, kN, 1);
  cp3.extent = make_hipExtent(kSZ, 1, 1);
  cp3.kind = hipMemcpyDeviceToDevice;
  hipGraphNode_t memcpy_node = nullptr;
  (void)hipGraphAddMemcpyNode(&memcpy_node, graph, nullptr, 0, &cp3);

  int host_hits = 0;
  hipHostNodeParams hnp{};
  hnp.fn = mtx_host_fn;
  hnp.userData = &host_hits;
  hipGraphNode_t host_node = nullptr;
  (void)hipGraphAddHostNode(&host_node, graph, nullptr, 0, &hnp);

  hipGraphNode_t record_node = nullptr, wait_node = nullptr;
  (void)hipGraphAddEventRecordNode(&record_node, graph, nullptr, 0, ev_a);
  (void)hipGraphAddEventWaitNode(&wait_node, graph, &record_node, 1, ev_a);
  {
    hipEvent_t read_back = nullptr;
    (void)hipGraphEventRecordNodeGetEvent(record_node, &read_back);
    (void)hipGraphEventRecordNodeSetEvent(record_node, ev_b);
    (void)hipGraphEventWaitNodeGetEvent(wait_node, &read_back);
    (void)hipGraphEventWaitNodeSetEvent(wait_node, ev_b);
  }

  const int symbol_seed[4] = {5, 6, 7, 8};
  int symbol_read_back[4] = {0, 0, 0, 0};
  hipGraphNode_t to_symbol_node = nullptr, from_symbol_node = nullptr;
  (void)hipGraphAddMemcpyNodeToSymbol(&to_symbol_node, graph, nullptr, 0,
                                      HIP_SYMBOL(hrr_mtx_symbol), symbol_seed,
                                      sizeof(symbol_seed), 0,
                                      hipMemcpyHostToDevice);
  (void)hipGraphAddMemcpyNodeFromSymbol(&from_symbol_node, graph, nullptr, 0,
                                        symbol_read_back,
                                        HIP_SYMBOL(hrr_mtx_symbol),
                                        sizeof(symbol_read_back), 0,
                                        hipMemcpyDeviceToHost);
  (void)hipGraphMemcpyNodeSetParamsToSymbol(to_symbol_node,
                                            HIP_SYMBOL(hrr_mtx_symbol),
                                            symbol_seed, sizeof(symbol_seed),
                                            0, hipMemcpyHostToDevice);
  (void)hipGraphMemcpyNodeSetParamsFromSymbol(from_symbol_node,
                                              symbol_read_back,
                                              HIP_SYMBOL(hrr_mtx_symbol),
                                              sizeof(symbol_read_back), 0,
                                              hipMemcpyDeviceToHost);

  // Nodes that are only constructed and queried go in a second graph: instantiating a graph with them crashes HIP on this stack.
  // This keeps the executable-graph half of the workload running.
  hipGraph_t build_only = nullptr;
  (void)hipGraphCreate(&build_only, 0);

  // The pool a graph allocates from is rebuilt at replay, so these nodes' addresses differ from the recorded ones.
  hipGraphNode_t alloc_node = nullptr, free_node = nullptr;
  {
    hipMemAllocNodeParams anp{};
    anp.poolProps.allocType = hipMemAllocationTypePinned;
    anp.poolProps.location.type = hipMemLocationTypeDevice;
    anp.poolProps.location.id = 0;
    anp.bytesize = kSZ;
    if (hipGraphAddMemAllocNode(&alloc_node, build_only, nullptr, 0, &anp)
            == hipSuccess) {
      hipMemAllocNodeParams read_back{};
      (void)hipGraphMemAllocNodeGetParams(alloc_node, &read_back);
      if (anp.dptr != nullptr &&
          hipGraphAddMemFreeNode(&free_node, build_only, &alloc_node, 1,
                                 anp.dptr) == hipSuccess) {
        void* freed = nullptr;
        (void)hipGraphMemFreeNodeGetParams(free_node, &freed);
      }
    }
  }

  hipGraph_t child = nullptr;
  hipGraphNode_t child_node = nullptr;
  if (hipGraphCreate(&child, 0) == hipSuccess) {
    hipGraphNode_t child_kernel = nullptr;
    (void)hipGraphAddKernelNode(&child_kernel, child, nullptr, 0, &knp);
    if (hipGraphAddChildGraphNode(&child_node, graph, nullptr, 0, child)
            == hipSuccess) {
      hipGraph_t got = nullptr;
      (void)hipGraphChildGraphNodeGetGraph(child_node, &got);
    }
  }
  {
    hipGraphNodeParams np{};
    np.type = hipGraphNodeTypeEmpty;
    hipGraphNode_t generic_node = nullptr;
    (void)hipGraphAddNode(&generic_node, build_only, nullptr, 0, &np);
  }

  {
    hipStreamBatchMemOpParams op{};
    op.operation = hipStreamMemOpWriteValue32;
    op.writeValue.address = reinterpret_cast<hipDeviceptr_t>(d);
    op.writeValue.value = 1;
    hipBatchMemOpNodeParams bnp{};
    bnp.count = 1;
    bnp.paramArray = &op;
    hipGraphNode_t batch_node = nullptr;
    if (hipGraphAddBatchMemOpNode(&batch_node, build_only, nullptr, 0, &bnp)
            == hipSuccess) {
      hipBatchMemOpNodeParams read_back{};
      (void)hipGraphBatchMemOpNodeGetParams(batch_node, &read_back);
      (void)hipGraphBatchMemOpNodeSetParams(batch_node, &bnp);
    }
  }

  {
    HIP_MEMCPY3D drv_copy{};
    drv_copy.srcMemoryType = hipMemoryTypeDevice;
    drv_copy.srcDevice = reinterpret_cast<hipDeviceptr_t>(d);
    drv_copy.srcPitch = kSZ;
    drv_copy.srcHeight = 1;
    drv_copy.dstMemoryType = hipMemoryTypeDevice;
    drv_copy.dstDevice = reinterpret_cast<hipDeviceptr_t>(d_copy);
    drv_copy.dstPitch = kSZ;
    drv_copy.dstHeight = 1;
    drv_copy.WidthInBytes = kSZ;
    drv_copy.Height = 1;
    drv_copy.Depth = 1;
    hipGraphNode_t drv_memcpy_node = nullptr;
    if (hipDrvGraphAddMemcpyNode(&drv_memcpy_node, build_only, nullptr, 0,
                                 &drv_copy, nullptr) == hipSuccess) {
      HIP_MEMCPY3D read_back{};
      (void)hipDrvGraphMemcpyNodeGetParams(drv_memcpy_node, &read_back);
    }
    hipGraphNode_t drv_memset_node = nullptr;
    (void)hipDrvGraphAddMemsetNode(&drv_memset_node, build_only, nullptr, 0,
                                   &msp, nullptr);
    hipGraphNode_t drv_free_node = nullptr;
    (void)hipDrvGraphAddMemFreeNode(&drv_free_node, build_only, nullptr, 0,
                                    reinterpret_cast<hipDeviceptr_t>(d_copy));
  }

  (void)hipGraphAddDependencies(graph, &memset_node, &host_node, 1);
  (void)hipGraphRemoveDependencies(graph, &memset_node, &host_node, 1);
  {
    size_t num_edges = 0;
    (void)hipGraphGetEdges(graph, nullptr, nullptr, &num_edges);
    std::vector<hipGraphNode_t> from(num_edges), to(num_edges);
    if (num_edges)
      (void)hipGraphGetEdges(graph, from.data(), to.data(), &num_edges);

    size_t num_roots = 0;
    (void)hipGraphGetRootNodes(graph, nullptr, &num_roots);
    std::vector<hipGraphNode_t> roots(num_roots);
    if (num_roots)
      (void)hipGraphGetRootNodes(graph, roots.data(), &num_roots);

    size_t num_deps = 0;
    (void)hipGraphNodeGetDependencies(kernel_node, nullptr, &num_deps);
    std::vector<hipGraphNode_t> deps(num_deps);
    if (num_deps)
      (void)hipGraphNodeGetDependencies(kernel_node, deps.data(), &num_deps);

    size_t num_dependents = 0;
    (void)hipGraphNodeGetDependentNodes(kernel_node, nullptr, &num_dependents);
    std::vector<hipGraphNode_t> dependents(num_dependents);
    if (num_dependents)
      (void)hipGraphNodeGetDependentNodes(kernel_node, dependents.data(),
                                          &num_dependents);
  }

  {
    hipKernelNodeAttrValue attr{};
    (void)hipGraphKernelNodeGetAttribute(kernel_node,
                                         hipKernelNodeAttributeCooperative,
                                         &attr);
    (void)hipGraphKernelNodeSetAttribute(kernel_node,
                                         hipKernelNodeAttributeCooperative,
                                         &attr);
    hipGraphNode_t attr_target = nullptr;
    if (hipGraphAddKernelNode(&attr_target, graph, nullptr, 0, &knp)
            == hipSuccess) {
      (void)hipGraphKernelNodeCopyAttributes(kernel_node, attr_target);
      (void)hipGraphDestroyNode(attr_target);
    }
  }

  {
    hipGraph_t clone = nullptr;
    if (hipGraphClone(&clone, graph) == hipSuccess) {
      hipGraphNode_t in_clone = nullptr;
      (void)hipGraphNodeFindInClone(&in_clone, kernel_node, clone);
      (void)hipGraphDestroy(clone);
    }
  }

  {
    hipUserObject_t obj = nullptr;
    int payload = 0;
    if (hipUserObjectCreate(&obj, &payload, mtx_host_fn, 1,
                            hipUserObjectNoDestructorSync) == hipSuccess) {
      (void)hipGraphRetainUserObject(graph, obj, 1, 0);
      (void)hipGraphReleaseUserObject(graph, obj, 1);
      (void)hipUserObjectRelease(obj, 1);
    }
  }

  hipGraphExec_t exec = nullptr;
  if (hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == hipSuccess) {
    (void)hipGraphExecEventRecordNodeSetEvent(exec, record_node, ev_a);
    (void)hipGraphExecEventWaitNodeSetEvent(exec, wait_node, ev_a);
    (void)hipGraphExecMemcpyNodeSetParamsToSymbol(
        exec, to_symbol_node, HIP_SYMBOL(hrr_mtx_symbol), symbol_seed,
        sizeof(symbol_seed), 0, hipMemcpyHostToDevice);
    (void)hipGraphExecMemcpyNodeSetParamsFromSymbol(
        exec, from_symbol_node, symbol_read_back, HIP_SYMBOL(hrr_mtx_symbol),
        sizeof(symbol_read_back), 0, hipMemcpyDeviceToHost);
    if (child != nullptr && child_node != nullptr)
      (void)hipGraphExecChildGraphNodeSetParams(exec, child_node, child);
    (void)hipGraphLaunch(exec, s);
    (void)hipStreamSynchronize(s);
    (void)hipGraphExecDestroy(exec);
  }

  {
    hipGraphInstantiateParams params{};
    hipGraphExec_t exec_with_params = nullptr;
    if (hipGraphInstantiateWithParams(&exec_with_params, graph, &params)
            == hipSuccess) {
      (void)hipGraphLaunch(exec_with_params, s);
      (void)hipStreamSynchronize(s);
      (void)hipGraphExecDestroy(exec_with_params);
    }
  }

  {
    hipGraph_t target = nullptr;
    if (hipGraphCreate(&target, 0) == hipSuccess) {
      if (hipStreamBeginCaptureToGraph(s, target, nullptr, nullptr, 0,
                                       hipStreamCaptureModeThreadLocal)
              == hipSuccess) {
        hipLaunchKernelGGL(hrr_mtx_fill, dim3((kN + 255) / 256), dim3(256), 0,
                           s, d, 17, kN);
        hipGraph_t captured = nullptr;
        (void)hipStreamEndCapture(s, &captured);
      }
      (void)hipGraphDestroy(target);
    }
  }

  {
    size_t used = 0;
    (void)hipDeviceGetGraphMemAttribute(0, hipGraphMemAttrUsedMemHigh, &used);
    size_t reset = 0;
    (void)hipDeviceSetGraphMemAttribute(0, hipGraphMemAttrUsedMemHigh, &reset);
  }

  if (child != nullptr) (void)hipGraphDestroy(child);
  if (build_only != nullptr) (void)hipGraphDestroy(build_only);
  (void)hipGraphDestroy(graph);

  hipLaunchKernelGGL(hrr_mtx_fill, dim3((kN + 255) / 256), dim3(256), 0, s, d,
                     17, kN);
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  HRR_HIP_CHECK(hipDeviceSynchronize());
  std::vector<int> h(kN, 0);
  HRR_HIP_CHECK(hipMemcpy(h.data(), d, kSZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < kN; ++i) REQUIRE(h[i] == 17);

  (void)hipEventDestroy(ev_a);
  (void)hipEventDestroy(ev_b);
  HRR_HIP_CHECK(hipStreamDestroy(s));
  HRR_HIP_CHECK(hipFree(d_copy));
  HRR_HIP_CHECK(hipFree(d));
}

TEST_CASE("Unit_HRR_ApiMatrix_LegacySurface_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, kSZ));
  hipStream_t s = nullptr;
  HRR_HIP_CHECK(hipStreamCreate(&s));
  hipEvent_t ev = nullptr;
  HRR_HIP_CHECK(hipEventCreateWithFlags(&ev, hipEventDisableTiming));

  (void)hipSetDeviceFlags(hipDeviceScheduleAuto);

  {
    (void)hipConfigureCall(dim3(1), dim3(64), 0, s);
  }

  {
    (void)hipFuncSetAttribute(reinterpret_cast<const void*>(hrr_mtx_fill),
                              hipFuncAttributeMaxDynamicSharedMemorySize, 0);
    (void)hipFuncSetCacheConfig(reinterpret_cast<const void*>(hrr_mtx_fill),
                                hipFuncCachePreferNone);
    (void)hipFuncSetSharedMemConfig(reinterpret_cast<const void*>(hrr_mtx_fill),
                                    hipSharedMemBankSizeDefault);
    size_t dynamic_smem = 0;
    (void)hipOccupancyAvailableDynamicSMemPerBlock(
        &dynamic_smem, reinterpret_cast<const void*>(hrr_mtx_fill), 1, 256);
    // Absent from amdhip.def.in, so HrrTest.exe cannot link it on Windows.
#ifndef _WIN32
    (void)hipKernelNameRefByPtr(reinterpret_cast<const void*>(hrr_mtx_fill), s);
#endif
  }

  {
    std::vector<char> co = compile_rtc();
    hipModule_t fat_module = nullptr;
    // A code object is not a fat binary, so this call fails and is not recorded.
    // The matrix then reports the API as unreachable.
    (void)hipModuleLoadFatBinary(&fat_module, co.data());

    hipModule_t mod = nullptr;
    if (hipModuleLoadData(&mod, co.data()) == hipSuccess) {
      textureReference* tex_ref = nullptr;
      (void)hipModuleGetTexRef(&tex_ref, mod, "mtx_rtc_tex");
      hipFunction_t fn = nullptr;
      if (hipModuleGetFunction(&fn, mod, "mtx_rtc_fill") == hipSuccess)
        (void)hipKernelNameRef(fn);
      (void)hipModuleUnload(mod);
    }

    hipLibrary_t lib = nullptr;
    if (hipLibraryLoadData(&lib, co.data(), nullptr, nullptr, 0, nullptr,
                           nullptr, 0) == hipSuccess) {
      unsigned int count = 0;
      (void)hipLibraryGetKernelCount(&count, lib);
      std::vector<hipKernel_t> kernels(count ? count : 1, nullptr);
      (void)hipLibraryEnumerateKernels(kernels.data(), count, lib);
      if (count && kernels[0] != nullptr) {
        size_t param_offset = 0, param_size = 0;
        (void)hipKernelGetParamInfo(kernels[0], 0, &param_offset, &param_size);
        (void)hipKernelSetAttribute(HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                    0, kernels[0], 0);
      }
      (void)hipLibraryUnload(lib);
    }
  }

  {
    hipDeviceptr_t base = 0;
    size_t range_size = 0;
    (void)hipMemGetAddressRange(&base, &range_size,
                                reinterpret_cast<hipDeviceptr_t>(d));

    hipPointer_attribute attrs[] = {HIP_POINTER_ATTRIBUTE_DEVICE_POINTER,
                                    HIP_POINTER_ATTRIBUTE_MEMORY_TYPE};
    void* device_ptr = nullptr;
    unsigned int mem_type = 0;
    void* attr_data[] = {&device_ptr, &mem_type};
    (void)hipDrvPointerGetAttributes(2, attrs, attr_data,
                                     reinterpret_cast<hipDeviceptr_t>(d));
    // Absent from amdhip.def.in, so HrrTest.exe cannot link it on Windows.
#ifndef _WIN32
    unsigned int sync_memops = 1;
    (void)hipPointerSetAttribute(&sync_memops,
                                 HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                 reinterpret_cast<hipDeviceptr_t>(d));
#endif
  }

  {
    hipDeviceProp_tR0000 old_props{};
    if (hipGetDevicePropertiesR0000(&old_props, 0) == hipSuccess) {
      int chosen = 0;
      (void)hipChooseDeviceR0000(&chosen, &old_props);
    }
  }

  {
    int* batch_dst = nullptr;
    if (hipMalloc(&batch_dst, kSZ) == hipSuccess) {
      void* dsts[] = {batch_dst};
      void* srcs[] = {d};
      size_t sizes[] = {kSZ};
      hipMemcpyAttributes attrs{};
      size_t attr_idx[] = {0};
      size_t fail_idx = 0;
      (void)hipMemcpyBatchAsync(dsts, srcs, sizes, 1, &attrs, attr_idx, 1,
                                &fail_idx, s);
      (void)hipStreamSynchronize(s);
      (void)hipFree(batch_dst);
    }
  }

  {
    int* unaligned_dst = nullptr;
    if (hipMalloc(&unaligned_dst, kSZ) == hipSuccess) {
      hip_Memcpy2D copy{};
      copy.srcMemoryType = hipMemoryTypeDevice;
      copy.srcDevice = reinterpret_cast<hipDeviceptr_t>(d);
      copy.srcPitch = kSZ;
      copy.dstMemoryType = hipMemoryTypeDevice;
      copy.dstDevice = reinterpret_cast<hipDeviceptr_t>(unaligned_dst);
      copy.dstPitch = kSZ;
      copy.WidthInBytes = kSZ;
      copy.Height = 1;
      (void)hipDrvMemcpy2DUnaligned(&copy);
      (void)hipFree(unaligned_dst);
    }
  }

  // Both are NOOP at replay.
  {
    void* entry = nullptr;
    hipDriverEntryPointQueryResult status = hipDriverEntryPointSuccess;
    (void)hipGetDriverEntryPoint("hipMalloc", &entry, 0, &status);
    void* spt_entry = nullptr;
    hipDriverEntryPointQueryResult spt_status = hipDriverEntryPointSuccess;
    (void)hipGetDriverEntryPoint_spt("hipMalloc", &spt_entry, 0, &spt_status);
  }

  // The _spt kernel launches live in the SptKernelLaunch and SptCoopLaunch workloads.
  // The callback pair below is UNREPLAYABLE, like its plain spellings.
  {
    int callback_hits = 0;
    (void)hipStreamAddCallback_spt(hipStreamPerThread, mtx_stream_callback,
                                   &callback_hits, 0);
    (void)hipLaunchHostFunc_spt(hipStreamPerThread, mtx_host_fn,
                                &callback_hits);
    (void)hipEventRecord(ev, hipStreamPerThread);
    (void)hipStreamWaitEvent_spt(hipStreamPerThread, ev, 0);
    (void)hipStreamSynchronize(hipStreamPerThread);

    hipMemcpy3DParms cp3{};
    int* copy_dst = nullptr;
    if (hipMalloc(&copy_dst, kSZ) == hipSuccess) {
      cp3.srcPtr = make_hipPitchedPtr(d, kSZ, kN, 1);
      cp3.dstPtr = make_hipPitchedPtr(copy_dst, kSZ, kN, 1);
      cp3.extent = make_hipExtent(kSZ, 1, 1);
      cp3.kind = hipMemcpyDeviceToDevice;
      (void)hipMemcpy3D_spt(&cp3);
      (void)hipMemcpy3DAsync_spt(&cp3, hipStreamPerThread);
      (void)hipStreamSynchronize(hipStreamPerThread);
      (void)hipFree(copy_dst);
    }

    int symbol_read_back[4] = {0, 0, 0, 0};
    (void)hipMemcpyFromSymbol_spt(symbol_read_back,
                                  HIP_SYMBOL(hrr_mtx_symbol),
                                  sizeof(symbol_read_back), 0,
                                  hipMemcpyDeviceToHost);
    (void)hipMemcpyFromSymbolAsync_spt(symbol_read_back,
                                       HIP_SYMBOL(hrr_mtx_symbol),
                                       sizeof(symbol_read_back), 0,
                                       hipMemcpyDeviceToHost,
                                       hipStreamPerThread);
    (void)hipStreamSynchronize(hipStreamPerThread);
  }

  {
    if (hipStreamBeginCapture_spt(hipStreamPerThread,
                                  hipStreamCaptureModeThreadLocal)
            == hipSuccess) {
      hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
      unsigned long long capture_id = 0;
      hipGraph_t capturing_graph = nullptr;
      const hipGraphNode_t* deps = nullptr;
      size_t num_deps = 0;
      (void)hipStreamGetCaptureInfo_spt(hipStreamPerThread, &status,
                                        &capture_id);
      (void)hipStreamGetCaptureInfo_v2(hipStreamPerThread, &status,
                                       &capture_id, &capturing_graph, &deps,
                                       &num_deps);
      (void)hipStreamGetCaptureInfo_v2_spt(hipStreamPerThread, &status,
                                           &capture_id, &capturing_graph,
                                           &deps, &num_deps);
      hipLaunchKernelGGL(hrr_mtx_fill, dim3((kN + 255) / 256), dim3(256), 0,
                         hipStreamPerThread, d, 19, kN);
      if (num_deps)
        (void)hipStreamUpdateCaptureDependencies(
            hipStreamPerThread, const_cast<hipGraphNode_t*>(deps), num_deps,
            0);

      hipGraph_t captured = nullptr;
      if (hipStreamEndCapture_spt(hipStreamPerThread, &captured)
              == hipSuccess && captured != nullptr) {
        hipGraphExec_t exec = nullptr;
        if (hipGraphInstantiate(&exec, captured, nullptr, nullptr, 0)
                == hipSuccess) {
          (void)hipGraphLaunch_spt(exec, hipStreamPerThread);
          (void)hipStreamSynchronize(hipStreamPerThread);
          (void)hipGraphExecDestroy(exec);
        }
        (void)hipGraphDestroy(captured);
      }
    }
  }

  hipLaunchKernelGGL(hrr_mtx_fill, dim3((kN + 255) / 256), dim3(256), 0, s, d,
                     19, kN);
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  HRR_HIP_CHECK(hipDeviceSynchronize());
  std::vector<int> h(kN, 0);
  HRR_HIP_CHECK(hipMemcpy(h.data(), d, kSZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < kN; ++i) REQUIRE(h[i] == 19);

  (void)hipEventDestroy(ev);
  HRR_HIP_CHECK(hipStreamDestroy(s));
  HRR_HIP_CHECK(hipFree(d));
}

TEST_CASE("Unit_HRR_ApiMatrix_SptKernelLaunch_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, kSZ));
  int fill_val = 19;
  int n_arg = kN;
  void* args[] = {&d, &fill_val, &n_arg};

  (void)hipLaunchKernel_spt(reinterpret_cast<const void*>(hrr_mtx_fill),
                            dim3((kN + 255) / 256), dim3(256), args, 0,
                            hipStreamPerThread);
  (void)hipStreamSynchronize(hipStreamPerThread);
  HRR_HIP_CHECK(hipFree(d));
}

TEST_CASE("Unit_HRR_ApiMatrix_SptCoopLaunch_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  if (!device_attr(hipDeviceAttributeCooperativeLaunch, 0))
    HRR_SKIP("cooperative launch is unsupported on this device");

  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, kSZ));
  int n_arg = kN;
  void* args[] = {&d, &n_arg};

  (void)hipLaunchCooperativeKernel_spt(
      reinterpret_cast<const void*>(hrr_mtx_coop), dim3(1), dim3(64), args, 0,
      hipStreamPerThread);
  (void)hipStreamSynchronize(hipStreamPerThread);
  HRR_HIP_CHECK(hipFree(d));
}

TEST_CASE("Unit_HRR_ApiMatrix_ProcAddress_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  void* entry = nullptr;
  hipDriverEntryPointQueryResult status = hipDriverEntryPointSuccess;
  (void)hipGetDriverEntryPoint("hipMalloc", &entry, 0, &status);

  void* proc = nullptr;
  hipDriverProcAddressQueryResult proc_status = HIP_GET_PROC_ADDRESS_SUCCESS;
  (void)hipGetProcAddress("hipMalloc", &proc, HIP_VERSION, 0, &proc_status);

  void* spt_proc = nullptr;
  hipDriverProcAddressQueryResult spt_status = HIP_GET_PROC_ADDRESS_SUCCESS;
  (void)hipGetProcAddress_spt("hipMalloc", &spt_proc, HIP_VERSION, 0,
                              &spt_status);
}

TEST_CASE("Unit_HRR_ApiMatrix_LegacyCtx_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  hipDevice_t dev = 0;
  (void)hipDeviceGet(&dev, 0);
  (void)hipSetDeviceFlags(hipDeviceScheduleAuto);

  hipCtx_t primary = nullptr;
  if (hipDevicePrimaryCtxRetain(&primary, dev) == hipSuccess) {
    (void)hipDevicePrimaryCtxSetFlags(dev, hipDeviceScheduleAuto);
    (void)hipDevicePrimaryCtxRelease(dev);
  }

  hipCtx_t ctx = nullptr;
  if (hipCtxCreate(&ctx, 0, dev) == hipSuccess) {
    unsigned int api_version = 0;
    hipDevice_t ctx_dev = 0;
    unsigned int ctx_flags = 0;
    hipFuncCache_t cache_config = hipFuncCachePreferNone;
    (void)hipCtxGetApiVersion(ctx, &api_version);
    (void)hipCtxGetDevice(&ctx_dev);
    (void)hipCtxGetFlags(&ctx_flags);
    (void)hipCtxGetCacheConfig(&cache_config);
    (void)hipCtxSetCacheConfig(cache_config);
    (void)hipCtxSetSharedMemConfig(hipSharedMemBankSizeDefault);
    (void)hipCtxSynchronize();
    (void)hipCtxDestroy(ctx);
  }

  int probe_device = -1;
  (void)hipGetDevice(&probe_device);

  hipCtx_t ctx2 = nullptr;
  if (hipCtxCreate(&ctx2, 0, dev) == hipSuccess) {
    hipCtx_t popped = nullptr;
    if (hipCtxPopCurrent(&popped) == hipSuccess)
      (void)hipCtxPushCurrent(popped);
  }
}

TEST_CASE("Unit_HRR_ApiMatrix_LegacyLaunch_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, kSZ));
  hipStream_t s = nullptr;
  HRR_HIP_CHECK(hipStreamCreate(&s));

  hipLaunchKernelGGL(hrr_mtx_noargs, dim3(1), dim3(64), 0, s);
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  int fill_val = 19;
  int n_arg = kN;
  if (hipConfigureCall(dim3((kN + 255) / 256), dim3(256), 0, s)
          == hipSuccess) {
    (void)hipSetupArgument(&d, sizeof(d), 0);
    (void)hipSetupArgument(&fill_val, sizeof(fill_val), sizeof(d));
    (void)hipSetupArgument(&n_arg, sizeof(n_arg), sizeof(d) + sizeof(fill_val));
  }

  HRR_HIP_CHECK(hipStreamDestroy(s));
  HRR_HIP_CHECK(hipFree(d));
}

TEST_CASE("Unit_HRR_ApiMatrix_MemPoolShare_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, kSZ));
  hipStream_t s = nullptr;
  HRR_HIP_CHECK(hipStreamCreate(&s));

  hipMemLocation location{};
  location.type = hipMemLocationTypeDevice;
  location.id = 0;

  {
    hipMemPool_t current = nullptr;
    (void)hipMemGetMemPool(&current, &location, hipMemAllocationTypePinned);
    if (current != nullptr)
      (void)hipMemSetMemPool(&location, hipMemAllocationTypePinned, current);
  }

  hipMemPoolProps props{};
  props.allocType = hipMemAllocationTypePinned;
  props.handleTypes = hipMemHandleTypePosixFileDescriptor;
  props.location = location;

  hipMemPool_t pool = nullptr;
  if (hipMemPoolCreate(&pool, &props) == hipSuccess) {
    void* pool_ptr = nullptr;
    if (hipMallocFromPoolAsync(&pool_ptr, kSZ, pool, s) == hipSuccess) {
      HRR_HIP_CHECK(hipStreamSynchronize(s));

      hipMemPoolPtrExportData export_data{};
      if (hipMemPoolExportPointer(&export_data, pool_ptr) == hipSuccess) {
        void* imported = nullptr;
        (void)hipMemPoolImportPointer(&imported, pool, &export_data);
      }
      (void)hipFreeAsync(pool_ptr, s);
      HRR_HIP_CHECK(hipStreamSynchronize(s));
    }

    int pool_fd = -1;
    if (hipMemPoolExportToShareableHandle(
            &pool_fd, pool, hipMemHandleTypePosixFileDescriptor, 0)
            == hipSuccess) {
      hipMemPool_t imported_pool = nullptr;
      (void)hipMemPoolImportFromShareableHandle(
          &imported_pool,
          reinterpret_cast<void*>(static_cast<intptr_t>(pool_fd)),
          hipMemHandleTypePosixFileDescriptor, 0);
    }
    (void)hipMemPoolDestroy(pool);
  }

  hipLaunchKernelGGL(hrr_mtx_fill, dim3((kN + 255) / 256), dim3(256), 0, s, d,
                     23, kN);
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  HRR_HIP_CHECK(hipDeviceSynchronize());
  std::vector<int> h(kN, 0);
  HRR_HIP_CHECK(hipMemcpy(h.data(), d, kSZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < kN; ++i) REQUIRE(h[i] == 23);

  HRR_HIP_CHECK(hipStreamDestroy(s));
  HRR_HIP_CHECK(hipFree(d));
}


TEST_CASE("Unit_HRR_ApiMatrix_Breadth2_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  int* d = nullptr;
  int* d2 = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, kSZ));
  HRR_HIP_CHECK(hipMalloc(&d2, kSZ));
  hipStream_t s = nullptr;
  HRR_HIP_CHECK(hipStreamCreate(&s));
  hipEvent_t ev = nullptr;
  HRR_HIP_CHECK(hipEventCreateWithFlags(&ev, hipEventDisableTiming));
  std::vector<int> host(kN, 7);

  {
    (void)hipProfilerStart();
    (void)hipProfilerStop();
    (void)hipGetStreamDeviceId(s);
    // hipExtHostAlloc has an HRR event type and a dispatch-table slot but no exported symbol, so it cannot be called.
  }

  // The green-context family is absent: its declarations are in the rocm-systems headers but not in the ROCm SDK these tests compile against.
  // api_matrix.yaml records the reason for each API.

  {
    void* dsts[1] = {d2};
    void* srcs[1] = {d};
    size_t sizes[1] = {kSZ};
    size_t fail_index = 0;

    hipMemcpyAttributes copy_attrs{};
    copy_attrs.srcAccessOrder = hipMemcpySrcAccessOrderStream;
    size_t attr_indices[1] = {0};
    (void)hipMemcpyBatchAsync(dsts, srcs, sizes, 1, &copy_attrs, attr_indices,
                              1, &fail_index, s);

    // Both operands need a location hint, or the runtime rejects the batch before the capture shim records it.
    hipMemLocation op_loc{};
    op_loc.type = hipMemLocationTypeDevice;
    op_loc.id = 0;
    hipMemcpy3DBatchOp batch_op{};
    batch_op.src.type = hipMemcpyOperandTypePointer;
    batch_op.src.op.ptr.ptr = d;
    batch_op.src.op.ptr.rowLength = kSZ;
    batch_op.src.op.ptr.layerHeight = 1;
    batch_op.src.op.ptr.locHint = op_loc;
    batch_op.dst = batch_op.src;
    batch_op.dst.op.ptr.ptr = d2;
    batch_op.extent = make_hipExtent(kSZ, 1, 1);
    batch_op.srcAccessOrder = hipMemcpySrcAccessOrderStream;
    (void)hipMemcpy3DBatchAsync(1, &batch_op, &fail_index, 0, s);

    hipMemLocation prefetch_loc{};
    prefetch_loc.type = hipMemLocationTypeDevice;
    prefetch_loc.id = 0;
    size_t prefetch_indices[1] = {0};
    (void)hipMemPrefetchBatchAsync(dsts, sizes, 1, &prefetch_loc,
                                   prefetch_indices, 1, 0, s);
    (void)hipStreamSynchronize(s);
  }

  // The driver-flavoured node constructors reject a null hipCtx_t, so this block needs a real context.
  {
    hipDevice_t node_device = 0;
    (void)hipDeviceGet(&node_device, 0);
    hipCtx_t node_ctx = nullptr;
    (void)hipCtxCreate(&node_ctx, 0, node_device);

    hipGraph_t graph = nullptr;
    if (hipGraphCreate(&graph, 0) == hipSuccess) {
      hipGraphNode_t generic_node = nullptr;
      hipGraphNodeParams node_params{};
      node_params.type = hipGraphNodeTypeMemset;
      node_params.memset.dst = d;
      node_params.memset.elementSize = sizeof(int);
      node_params.memset.width = 16;
      node_params.memset.height = 1;
      node_params.memset.value = 0;
      (void)hipGraphAddNode(&generic_node, graph, nullptr, 0, &node_params);

      hipStreamBatchMemOpParams memop{};
      memop.waitValue.operation = hipStreamMemOpWaitValue32;
      memop.waitValue.address = reinterpret_cast<hipDeviceptr_t>(d);
      memop.waitValue.value = 0;
      memop.waitValue.flags = hipStreamWaitValueGte;
      hipBatchMemOpNodeParams batch_params{};
      batch_params.ctx = node_ctx;
      batch_params.count = 1;
      batch_params.paramArray = &memop;
      batch_params.flags = 0;
      hipGraphNode_t batch_node = nullptr;
      if (hipGraphAddBatchMemOpNode(&batch_node, graph, nullptr, 0,
                                    &batch_params) == hipSuccess) {
        hipBatchMemOpNodeParams read_back{};
        (void)hipGraphBatchMemOpNodeGetParams(batch_node, &read_back);
        (void)hipGraphBatchMemOpNodeSetParams(batch_node, &batch_params);
      }

      // The eight external-semaphore graph-node APIs are not called: their dispatch-table slots are null in this build.
      // The call jumps to address zero and crashes the recording program (hip_table_interface.cpp:3083).

      HIP_MEMCPY3D drv_copy{};
      drv_copy.srcMemoryType = hipMemoryTypeDevice;
      drv_copy.srcDevice = reinterpret_cast<hipDeviceptr_t>(d);
      drv_copy.srcPitch = kSZ;
      drv_copy.srcHeight = 1;
      drv_copy.dstMemoryType = hipMemoryTypeDevice;
      drv_copy.dstDevice = reinterpret_cast<hipDeviceptr_t>(d2);
      drv_copy.dstPitch = kSZ;
      drv_copy.dstHeight = 1;
      drv_copy.WidthInBytes = kSZ;
      drv_copy.Height = 1;
      drv_copy.Depth = 1;
      hipGraphNode_t drv_node = nullptr;
      if (hipDrvGraphAddMemcpyNode(&drv_node, graph, nullptr, 0, &drv_copy,
                                   node_ctx) == hipSuccess) {
        HIP_MEMCPY3D drv_read_back{};
        (void)hipDrvGraphMemcpyNodeGetParams(drv_node, &drv_read_back);
      }

      (void)hipGraphDestroy(graph);
    }
    if (node_ctx) (void)hipCtxDestroy(node_ctx);
  }

  {
    std::vector<char> code_object = compile_rtc();

    // hipLinkAddFile and hipLibraryLoadFromFile both read a file, so write the code object once up front.
    // The file must exist before hipLinkAddFile runs, because capture records only successful calls.
    const fs::path co_path = fs::temp_directory_path() / "hrr_matrix_rtc.co";
    {
      std::ofstream out(co_path, std::ios::binary);
      out.write(code_object.data(),
                static_cast<std::streamsize>(code_object.size()));
    }

    hipLinkState_t link_state = nullptr;
    if (hipLinkCreate(0, nullptr, nullptr, &link_state) == hipSuccess) {
      (void)hipLinkAddData(link_state, hipJitInputLLVMBundledBitcode,
                           code_object.data(), code_object.size(), "mtx_rtc",
                           0, nullptr, nullptr);
      (void)hipLinkAddFile(link_state, hipJitInputLLVMBundledBitcode,
                           co_path.string().c_str(), 0, nullptr, nullptr);
      (void)hipLinkDestroy(link_state);
    }

    hipLibrary_t library = nullptr;
    if (hipLibraryLoadData(&library, code_object.data(), nullptr, nullptr, 0,
                           nullptr, nullptr, 0) == hipSuccess) {
      // hipLibraryGetGlobal and hipLibraryGetManaged are omitted for the same header gap.
      hipKernel_t kernel = nullptr;
      if (hipLibraryGetKernel(&kernel, library, "mtx_rtc_fill")
              == hipSuccess) {
        const char* kernel_name = nullptr;
        (void)hipKernelGetName(&kernel_name, kernel);
      }
      (void)hipLibraryUnload(library);
    }

    {
      hipLibrary_t file_library = nullptr;
      if (hipLibraryLoadFromFile(&file_library, co_path.string().c_str(),
                                 nullptr, nullptr, 0, nullptr, nullptr, 0)
              == hipSuccess)
        (void)hipLibraryUnload(file_library);
      std::error_code ec;
      fs::remove(co_path, ec);
    }

    hipModule_t module = nullptr;
    if (hipModuleLoadData(&module, code_object.data()) == hipSuccess) {
      hipFunction_t func = nullptr;
      if (hipModuleGetFunction(&func, module, "mtx_rtc_fill") == hipSuccess) {
        int fill_value = 29;
        int count = kN;
        void* args[] = {&d, &fill_value, &count};
        (void)hipHccModuleLaunchKernel(func, kN, 1, 1, 256, 1, 1, 0, s, args,
                                       nullptr, nullptr, nullptr);
        (void)hipStreamSynchronize(s);
        // hipModuleLaunchCooperativeKernelMultiDevice is omitted for the same null-dispatch-slot reason as the external-semaphore nodes.
      }
      textureReference* tex_ref = nullptr;
      (void)hipModuleGetTexRef(&tex_ref, module, "mtx_rtc_global");
      (void)hipModuleUnload(module);
    }

    int fill_value = 29;
    int count = kN;
    void* args[] = {&d, &fill_value, &count};
    hipLaunchParams launch_params{};
    launch_params.func = reinterpret_cast<void*>(hrr_mtx_fill);
    launch_params.gridDim = dim3(1);
    launch_params.blockDim = dim3(64);
    launch_params.args = args;
    launch_params.sharedMem = 0;
    launch_params.stream = s;
    (void)hipExtLaunchMultiKernelMultiDevice(&launch_params, 1, 0);
  }


  {
    hipChannelFormatDesc channel = hipCreateChannelDesc<int>();
    hipArray_t array = nullptr;
    hipArray_t array2 = nullptr;
    if (hipMallocArray(&array, &channel, kN, 0, hipArrayDefault)
            == hipSuccess) {
      (void)hipMemcpyHtoA(array, 0, host.data(), kSZ);
      std::vector<int> read_back(kN, 0);
      (void)hipMemcpyAtoH(read_back.data(), array, 0, kSZ);
      (void)hipMemcpyAtoD(reinterpret_cast<hipDeviceptr_t>(d), array, 0, kSZ);
      (void)hipMemcpyDtoA(array, 0, reinterpret_cast<hipDeviceptr_t>(d), kSZ);
      (void)hipMemcpyAtoHAsync(read_back.data(), array, 0, kSZ, s);
      (void)hipMemcpyHtoAAsync(array, 0, host.data(), kSZ, s);
      (void)hipStreamSynchronize(s);

      if (hipMallocArray(&array2, &channel, kN, 0, hipArrayDefault)
              == hipSuccess) {
        (void)hipMemcpyAtoA(array2, 0, array, 0, kSZ);
        (void)hipFreeArray(array2);
      }

      hipResourceDesc resource_desc{};
      resource_desc.resType = hipResourceTypeArray;
      resource_desc.res.array.array = array;
      hipTextureDesc texture_desc{};
      texture_desc.readMode = hipReadModeElementType;
      hipTextureObject_t texture = 0;
      if (hipCreateTextureObject(&texture, &resource_desc, &texture_desc,
                                 nullptr) == hipSuccess) {
        hipResourceDesc resource_read_back{};
        hipTextureDesc texture_read_back{};
        hipResourceViewDesc view_read_back{};
        (void)hipGetTextureObjectResourceDesc(&resource_read_back, texture);
        (void)hipGetTextureObjectTextureDesc(&texture_read_back, texture);
        (void)hipGetTextureObjectResourceViewDesc(&view_read_back, texture);
        (void)hipDestroyTextureObject(texture);
      }
      (void)hipFreeArray(array);
    }
  }

  // The legacy-surface workload reaches hipStreamUpdateCaptureDependencies only when capture reports a dependency, so exercise it here.
  {
    if (hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal)
            == hipSuccess) {
      hipLaunchKernelGGL(hrr_mtx_fill, dim3((kN + 255) / 256), dim3(256), 0, s,
                         d, 29, kN);
      hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
      unsigned long long capture_id = 0;
      hipGraph_t capturing_graph = nullptr;
      const hipGraphNode_t* deps = nullptr;
      size_t num_deps = 0;
      (void)hipStreamGetCaptureInfo_v2(s, &status, &capture_id,
                                       &capturing_graph, &deps, &num_deps);
      if (num_deps) {
        std::vector<hipGraphNode_t> dep_copy(deps, deps + num_deps);
        (void)hipStreamUpdateCaptureDependencies(s, dep_copy.data(),
                                                 dep_copy.size(), 0);
      }
      hipGraph_t captured = nullptr;
      if (hipStreamEndCapture(s, &captured) == hipSuccess && captured) {
        hipGraphExec_t exec = nullptr;
        if (hipGraphInstantiate(&exec, captured, nullptr, nullptr, 0)
                == hipSuccess) {
          (void)hipGraphLaunch(exec, s);
          (void)hipStreamSynchronize(s);
          (void)hipGraphExecDestroy(exec);
        }
        (void)hipGraphDestroy(captured);
      }
    }
  }

  hipLaunchKernelGGL(hrr_mtx_fill, dim3((kN + 255) / 256), dim3(256), 0, s, d,
                     29, kN);
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  HRR_HIP_CHECK(hipDeviceSynchronize());
  std::vector<int> final_host(kN, 0);
  HRR_HIP_CHECK(hipMemcpy(final_host.data(), d, kSZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < kN; ++i) REQUIRE(final_host[i] == 29);

  (void)hipEventDestroy(ev);
  HRR_HIP_CHECK(hipStreamDestroy(s));
  HRR_HIP_CHECK(hipFree(d2));
  HRR_HIP_CHECK(hipFree(d));
}

TEST_CASE("Unit_HRR_ApiMatrix_Reset_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, kSZ));
  hipLaunchKernelGGL(hrr_mtx_fill, dim3((kN + 255) / 256), dim3(256), 0, 0, d,
                     31, kN);
  HRR_HIP_CHECK(hipDeviceSynchronize());

  hipDevice_t dev = 0;
  (void)hipDeviceGet(&dev, 0);
  (void)hipDevicePrimaryCtxReset(dev);

  (void)hipDeviceReset();
}

TEST_CASE("Unit_HRR_ApiMatrix_Deprioritised_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, kSZ));
  hipStream_t s = nullptr;
  HRR_HIP_CHECK(hipStreamCreate(&s));

  {
    int* managed = nullptr;
    if (hipMallocManaged(reinterpret_cast<void**>(&managed), kSZ,
                         hipMemAttachGlobal) == hipSuccess && managed) {
      for (int i = 0; i < kN; ++i) managed[i] = 1;
      (void)hipMemAdvise(managed, kSZ, hipMemAdviseSetReadMostly, 0);
      (void)hipMemPrefetchAsync(managed, kSZ, 0, s);
      (void)hipStreamAttachMemAsync(s, managed, 0, hipMemAttachGlobal);
      (void)hipStreamSynchronize(s);
      unsigned int attr_value = 0;
      (void)hipMemRangeGetAttribute(&attr_value, sizeof(attr_value),
                                    hipMemRangeAttributeReadMostly, managed,
                                    kSZ);

      // The _v2 spellings take a hipMemLocation instead of a device ordinal.
      // Managed memory works on this part, so these are ordinary covered APIs.
      hipMemLocation loc{};
      loc.type = hipMemLocationTypeDevice;
      loc.id = 0;
      (void)hipMemAdvise_v2(managed, kSZ, hipMemAdviseSetReadMostly, loc);
      (void)hipMemPrefetchAsync_v2(managed, kSZ, loc, 0, s);
      (void)hipStreamSynchronize(s);

      unsigned int read_mostly = 0;
      unsigned int last_loc = 0;
      void* attr_data[2] = {&read_mostly, &last_loc};
      size_t attr_sizes[2] = {sizeof(read_mostly), sizeof(last_loc)};
      hipMemRangeAttribute attrs[2] = {
          hipMemRangeAttributeReadMostly,
          hipMemRangeAttributeLastPrefetchLocation};
      (void)hipMemRangeGetAttributes(attr_data, attr_sizes, attrs, 2, managed,
                                     kSZ);
      (void)hipFree(managed);
    }
  }

  // Outside the image-support guard so they run on every device. gfx950 has no image support and returns hipErrorNotSupported,
  // so there they reach no archive and the T5 floor does not count them.
  {
    HIP_ARRAY_DESCRIPTOR desc{};
    desc.Width = 64;
    desc.Height = 1;
    desc.Format = HIP_AD_FORMAT_FLOAT;
    desc.NumChannels = 1;
    hipArray_t drv_array = nullptr;
    if (hipArrayCreate(&drv_array, &desc) == hipSuccess)
      (void)hipArrayDestroy(drv_array);

    HIP_ARRAY3D_DESCRIPTOR desc3{};
    desc3.Width = 8;
    desc3.Height = 8;
    desc3.Depth = 1;
    desc3.Format = HIP_AD_FORMAT_FLOAT;
    desc3.NumChannels = 1;
    hipArray_t drv_array3 = nullptr;
    if (hipArray3DCreate(&drv_array3, &desc3) == hipSuccess)
      (void)hipArrayDestroy(drv_array3);
  }

  if (device_attr(hipDeviceAttributeImageSupport)) {
    hipChannelFormatDesc channel =
        hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);

    hipArray_t array = nullptr;
    if (hipMallocArray(&array, &channel, 64, 1, hipArrayDefault) == hipSuccess) {
      hipChannelFormatDesc read_back{};
      (void)hipGetChannelDesc(&read_back, array);

      hipResourceDesc res_desc{};
      res_desc.resType = hipResourceTypeArray;
      res_desc.res.array.array = array;
      hipTextureDesc tex_desc{};
      tex_desc.addressMode[0] = hipAddressModeClamp;
      tex_desc.filterMode = hipFilterModePoint;
      tex_desc.readMode = hipReadModeElementType;

      hipTextureObject_t tex_obj = 0;
      if (hipCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr)
              == hipSuccess) {
        hipResourceDesc out_res{};
        hipTextureDesc out_tex{};
        (void)hipGetTextureObjectResourceDesc(&out_res, tex_obj);
        (void)hipGetTextureObjectTextureDesc(&out_tex, tex_obj);
        (void)hipDestroyTextureObject(tex_obj);
      }

      hipSurfaceObject_t surf_obj = 0;
      if (hipCreateSurfaceObject(&surf_obj, &res_desc) == hipSuccess)
        (void)hipDestroySurfaceObject(surf_obj);
      (void)hipFreeArray(array);
    }

    hipMipmappedArray_t mipmap = nullptr;
    hipExtent mip_extent = make_hipExtent(8, 8, 0);
    if (hipMallocMipmappedArray(&mipmap, &channel, mip_extent, 2, 0)
            == hipSuccess) {
      hipArray_t level = nullptr;
      (void)hipGetMipmappedArrayLevel(&level, mipmap, 0);
      (void)hipFreeMipmappedArray(mipmap);
    }
  }

  // Creating either object needs image support, so the handle here is null.
  // The destroy entry points still return success and are recorded, which covers their capture and replay paths.
  {
    (void)hipDestroyTextureObject(0);
    (void)hipDestroySurfaceObject(0);
    (void)hipTexObjectDestroy(0);
  }

  {
    int n_arg = kN;
    int fill_val = 13;
    void* args[] = {&d, &fill_val, &n_arg};
    if (device_attr(hipDeviceAttributeCooperativeLaunch)) {
      (void)hipLaunchCooperativeKernel(
          reinterpret_cast<const void*>(hrr_mtx_fill),
          dim3((kN + 255) / 256), dim3(256), args, 0, s);
      (void)hipStreamSynchronize(s);
    }
    (void)hipExtLaunchKernel(reinterpret_cast<const void*>(hrr_mtx_fill),
                             dim3((kN + 255) / 256), dim3(256), args, 0, s,
                             nullptr, nullptr, 0);
    (void)hipStreamSynchronize(s);
  }

  hipLaunchKernelGGL(hrr_mtx_fill, dim3((kN + 255) / 256), dim3(256), 0, s,
                     d, 13, kN);
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  HRR_HIP_CHECK(hipDeviceSynchronize());
  std::vector<int> h(kN, 0);
  HRR_HIP_CHECK(hipMemcpy(h.data(), d, kSZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < kN; ++i) REQUIRE(h[i] == 13);

  HRR_HIP_CHECK(hipStreamDestroy(s));
  HRR_HIP_CHECK(hipFree(d));
}

/**
 * @}
 */
