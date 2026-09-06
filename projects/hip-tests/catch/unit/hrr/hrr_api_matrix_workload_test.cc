/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR API-matrix workloads (direct GPU implementations)
 * @{
 * @ingroup HRRTest
 * Hidden ([.][hrr-direct]) GPU workloads for the per-API replay matrix. Each
 * one is a call site for a family of HIP APIs; hrr_api_matrix_test.cc captures
 * them, replays the archive, and asserts that every API behaved as
 * api_matrix.yaml says it should.
 *
 * These differ from the workloads in hrr_workload_test.cc in intent rather
 * than in kind. Those exist to produce a D2H buffer that must survive replay
 * byte-for-byte. These exist to make a *named* API appear in the archive so
 * its replay class can be observed, and they deliberately reach APIs whose
 * replay is known to be wrong — a captured no-op is exactly what the matrix
 * needs to see.
 *
 * Every workload therefore:
 *   - opens with hipSetDevice(0), so the first hipMalloc is not the in-flight
 *     first HIP call that capture installs its shims from and misses;
 *   - calls incidental APIs through (void) rather than HIP_CHECK, because an
 *     API failing on this device is information for the matrix, not a reason
 *     to abandon the capture;
 *   - closes with a D2H memcpy, so the roundtrip has a blob to validate and a
 *     silently empty archive cannot pass.
 *
 * Run one directly for manual inspection:
 *   HrrTest "Unit_HRR_ApiMatrix_UC1Dense_Direct"
 */

#include <hip_test_common.hh>
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

// Distinct names: hrr_workload_test.cc has its own file-scope kernels and both
// translation units link into HrrTest.
__global__ void hrr_mtx_fill(int* out, int val, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = val;
}

__global__ void hrr_mtx_add(const int* a, const int* b, int* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

// Statically registered __device__ symbol, reached through hipMemcpyToSymbol
// and friends. It has to be a real compile-time symbol rather than an RTC one:
// the hipGetSymbol* / hipMemcpy*Symbol family takes HIP_SYMBOL(...) of a host
// shadow, which only exists for statically registered variables. This is the
// shape MoRI's globalGpuStates arrives in (section 8.6).
__device__ int hrr_mtx_symbol[4] = {0, 0, 0, 0};

// No-parameter kernel. hipLaunchByPtr launches whatever the exec stack holds,
// so a kernel that takes nothing is the only way to exercise it without first
// calling hipSetupArgument — which crashes replay, and would take the rest of
// the archive with it (see Unit_HRR_ApiMatrix_LegacyLaunch_Direct).
__global__ void hrr_mtx_noargs() {}

// Grid-wide cooperative barrier: every workgroup must be co-resident, so an
// unfaithful replay of the launch deadlocks here rather than returning a wrong
// answer. That is the property the T2 test is guarding.
__global__ void hrr_mtx_coop(int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = i;
}

// Runtime-compiled source. Module-path APIs (hipModuleLoadData,
// hipModuleGetFunction, hipModuleLaunchKernel and friends) need a code object
// that arrived through the module path rather than through static
// registration, which is what HIPRTC gives us.
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

// Compile kRtcSource and return the code object bytes.
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

// hipLaunchHostFunc / hipStreamAddCallback payload. Nothing observable happens
// here on purpose: the point is whether the *call* is recorded and what replay
// does with a recorded function pointer, not what the callback computes.
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

// ===========================================================================
// T0 — UC1 dense LLM inference / serving substrate.
//
// The rank-1 use-case surface: PyTorch eager + hipBLASLt + a stream-captured
// graph. Most of it is already covered elsewhere; what this workload adds is
// the launch and module holes UC1 actually depends on and no existing workload
// reaches — hipDrvLaunchKernelEx (Triton's default modern launch path),
// hipLaunchKernelExC (CK cluster launches), hipLaunchHostFunc recorded into a
// captured graph (CK's FMHA-backward launcher via aiter's mha_bwd) and
// hipModuleUnload.
//
// Final blob: d_out[i] == 7 + 7.
// ===========================================================================
TEST_CASE("Unit_HRR_ApiMatrix_UC1Dense_Direct", "[.][hrr-direct]") {
  HIP_CHECK(hipSetDevice(0));

  // ---- Device and runtime identity ---------------------------------------
  {
    int count = 0, dev = 0, rt = 0, drv = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    HIP_CHECK(hipGetDevice(&dev));
    (void)hipRuntimeGetVersion(&rt);
    (void)hipDriverGetVersion(&drv);
    size_t free_bytes = 0, total_bytes = 0;
    HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
    REQUIRE(total_bytes > 0);
  }

  // ---- Allocation --------------------------------------------------------
  int *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_a, kSZ));
  HIP_CHECK(hipMalloc(&d_b, kSZ));
  HIP_CHECK(hipMalloc(&d_out, kSZ));

  int* h_pinned = nullptr;
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h_pinned), kSZ,
                          hipHostMallocDefault));
  for (int i = 0; i < kN; ++i) h_pinned[i] = 7;

  // hipHostRegister / hipHostGetDevicePointer / hipHostUnregister — the
  // zero-copy path torch uses for pinned staging buffers.
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

  // ---- Streams and events ------------------------------------------------
  hipStream_t s_default = nullptr, s_prio = nullptr;
  HIP_CHECK(hipStreamCreate(&s_default));
  hipStream_t s_nb = nullptr;
  HIP_CHECK(hipStreamCreateWithFlags(&s_nb, hipStreamNonBlocking));
  {
    int lo = 0, hi = 0;
    (void)hipDeviceGetStreamPriorityRange(&lo, &hi);
    HIP_CHECK(hipStreamCreateWithPriority(&s_prio, hipStreamDefault, hi));
    int prio = 0;
    unsigned int flags = 0;
    (void)hipStreamGetPriority(s_prio, &prio);
    (void)hipStreamGetFlags(s_prio, &flags);
  }

  hipEvent_t ev_start = nullptr, ev_stop = nullptr, ev_flag = nullptr;
  HIP_CHECK(hipEventCreate(&ev_start));
  HIP_CHECK(hipEventCreate(&ev_stop));
  HIP_CHECK(hipEventCreateWithFlags(&ev_flag, hipEventDisableTiming));

  // ---- Data movement -----------------------------------------------------
  HIP_CHECK(hipMemcpy(d_a, h_pinned, kSZ, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpyAsync(d_b, h_pinned, kSZ, hipMemcpyHostToDevice, s_default));
  HIP_CHECK(hipMemcpyWithStream(d_out, h_pinned, kSZ, hipMemcpyHostToDevice, s_nb));
  HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(d_a), h_pinned, kSZ));
  HIP_CHECK(hipMemcpyHtoDAsync(reinterpret_cast<hipDeviceptr_t>(d_b), h_pinned,
                               kSZ, s_default));
  HIP_CHECK(hipMemcpyDtoD(reinterpret_cast<hipDeviceptr_t>(d_out),
                          reinterpret_cast<hipDeviceptr_t>(d_a), kSZ));
  HIP_CHECK(hipMemset(d_out, 0, kSZ));
  HIP_CHECK(hipMemsetAsync(d_out, 0, kSZ, s_default));
  HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d_out), 0, kN));
  HIP_CHECK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(d_out), 0, kN,
                              s_default));
  HIP_CHECK(hipStreamSynchronize(s_default));

  // ---- Chevron launch (the __hipPushCallConfiguration path) --------------
  HIP_CHECK(hipEventRecord(ev_start, s_default));
  hipLaunchKernelGGL(hrr_mtx_add, dim3((kN + 255) / 256), dim3(256), 0, s_default,
                     d_a, d_b, d_out, kN);
  HIP_CHECK(hipEventRecord(ev_stop, s_default));
  HIP_CHECK(hipEventSynchronize(ev_stop));
  {
    float ms = 0.f;
    (void)hipEventElapsedTime(&ms, ev_start, ev_stop);
    (void)hipEventQuery(ev_stop);
    (void)hipStreamQuery(s_default);
  }
  HIP_CHECK(hipStreamWaitEvent(s_nb, ev_stop, 0));

  // hipFuncGetAttribute against the chevron kernel's device function.
  {
    hipFuncAttributes fattr{};
    (void)hipFuncGetAttributes(&fattr, reinterpret_cast<const void*>(hrr_mtx_add));
  }

  // ---- Module path: load, launch four ways, unload -----------------------
  std::vector<char> co = compile_rtc();
  hipModule_t mod = nullptr;
  HIP_CHECK(hipModuleLoadData(&mod, co.data()));
  hipFunction_t fn = nullptr;
  HIP_CHECK(hipModuleGetFunction(&fn, mod, "mtx_rtc_fill"));
  {
    int attr_value = 0;
    (void)hipFuncGetAttribute(&attr_value, HIP_FUNC_ATTRIBUTE_NUM_REGS, fn);
  }

  const int blocks = (kN + 255) / 256;
  int  fill_val = 7;
  int  n_arg    = kN;
  void* kargs[] = {&d_out, &fill_val, &n_arg};

  // 1. hipModuleLaunchKernel — the baseline module launch.
  HIP_CHECK(hipModuleLaunchKernel(fn, blocks, 1, 1, 256, 1, 1, 0, s_default,
                                  kargs, nullptr));

  // 2. hipExtModuleLaunchKernel — global-work-size form, with event timing.
  HIP_CHECK(hipExtModuleLaunchKernel(fn, kN, 1, 1, 256, 1, 1, 0, s_default,
                                     kargs, nullptr, nullptr, nullptr, 0));

  // 3. hipDrvLaunchKernelEx — Triton's default launch path since the
  //    extensible-launch migration. The config is a const-struct pointer, so
  //    it does not reach the archive (section 8.3, P2c) and the handler fails
  //    at replay; the matrix declares that and asserts it.
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

  // 4. hipLaunchKernelExC — the runtime-API spelling, used by CK for cluster
  //    launches. Same const-struct-pointer loss.
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
  HIP_CHECK(hipStreamSynchronize(s_default));

  // hipModuleUnload — UC1's one consequential gap. Replay does not reproduce
  // module lifetime, which matters at UC3's load/unload rates.
  HIP_CHECK(hipModuleUnload(mod));

  // ---- Dispatch-table lookups torch and RCCL use -------------------------
  {
    void* sym = nullptr;
    (void)hipGetProcAddress("hipMalloc", &sym, 0, 0, nullptr);
    hipFunction_t by_symbol = nullptr;
    (void)hipGetFuncBySymbol(&by_symbol,
                             reinterpret_cast<const void*>(hrr_mtx_add));
  }

  // ---- Stream-captured graph, with a host function recorded into it -------
  {
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    hipGraph_t graph = nullptr;
    hipGraphExec_t exec = nullptr;
    int host_fn_hits = 0;

    hipStreamCaptureMode mode = hipStreamCaptureModeThreadLocal;
    (void)hipThreadExchangeStreamCaptureMode(&mode);

    HIP_CHECK(hipStreamBeginCapture(s_nb, hipStreamCaptureModeThreadLocal));
    hipLaunchKernelGGL(hrr_mtx_add, dim3(blocks), dim3(256), 0, s_nb,
                       d_a, d_b, d_out, kN);
    // UC1 fidelity risk 2: CK's FMHA-backward launcher records a host callback
    // into the captured graph. A recorded function pointer cannot be
    // re-executed at replay, so this is the call site that proves it.
    (void)hipLaunchHostFunc(s_nb, mtx_host_fn, &host_fn_hits);
    (void)hipStreamIsCapturing(s_nb, &status);
    {
      unsigned long long capture_id = 0;
      (void)hipStreamGetCaptureInfo(s_nb, &status, &capture_id);
    }
    HIP_CHECK(hipStreamEndCapture(s_nb, &graph));
    (void)hipThreadExchangeStreamCaptureMode(&mode);

    REQUIRE(graph != nullptr);
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(exec, s_nb));
    HIP_CHECK(hipStreamSynchronize(s_nb));
    HIP_CHECK(hipGraphExecDestroy(exec));
    HIP_CHECK(hipGraphDestroy(graph));
  }

  // ---- Final D2H: the blob replay must reproduce -------------------------
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<int> h_out(kN, 0);
  HIP_CHECK(hipMemcpy(h_out.data(), d_out, kSZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < kN; ++i) REQUIRE(h_out[i] == 14);
  HIP_CHECK(hipMemcpyDtoH(h_out.data(), reinterpret_cast<hipDeviceptr_t>(d_out),
                          kSZ));
  HIP_CHECK(hipMemcpyDtoHAsync(h_out.data(),
                               reinterpret_cast<hipDeviceptr_t>(d_out), kSZ,
                               s_default));
  HIP_CHECK(hipStreamSynchronize(s_default));

  HIP_CHECK(hipEventDestroy(ev_flag));
  HIP_CHECK(hipEventDestroy(ev_stop));
  HIP_CHECK(hipEventDestroy(ev_start));
  HIP_CHECK(hipStreamDestroy(s_prio));
  HIP_CHECK(hipStreamDestroy(s_nb));
  HIP_CHECK(hipStreamDestroy(s_default));
  HIP_CHECK(hipHostFree(h_pinned));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_a));
}

// ===========================================================================
// T1 — the payload-loss class (section 8.3).
//
// Every API here has a real playback handler and is counted in the "273
// faithfully replayed" figure, yet cannot be replayed: the generator lowers a
// const-struct pointer to a bare capture-time address, or a >8-byte by-value
// struct to a single 8-byte field. The workload's job is to call them with
// arguments whose loss is *observable*, so the T1 test can assert on the
// specific truncation rather than on a general "it did not work".
//
// hipIpcGetMemHandle is the sharpest case: the handle is 64 bytes and only 8
// survive, so the test fills it with a recognisable pattern.
//
// Final blob: d[i] == 3.
// ===========================================================================
TEST_CASE("Unit_HRR_ApiMatrix_PayloadLoss_Direct", "[.][hrr-direct]") {
  HIP_CHECK(hipSetDevice(0));

  int* d = nullptr;
  HIP_CHECK(hipMalloc(&d, kSZ));
  hipStream_t s = nullptr;
  HIP_CHECK(hipStreamCreate(&s));
  hipLaunchKernelGGL(hrr_mtx_fill, dim3((kN + 255) / 256), dim3(256), 0, s,
                     d, 3, kN);
  HIP_CHECK(hipStreamSynchronize(s));

  // ---- P2(a): the 64-byte IPC handle -------------------------------------
  // hipIpcMemHandle_t is HIP_IPC_HANDLE_SIZE (64) bytes passed by value, and
  // the generated struct field is one uint64_t, so 56 bytes are discarded at
  // capture. Open it in-process: HIP rejects a same-process open, and that is
  // fine — the event is recorded either way and the recorded *argument* is
  // what the matrix is about.
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
        (void)hipIpcOpenEventHandle(&reopened, event_handle);
      }
      (void)hipEventDestroy(ipc_event);
    }
  }

  // ---- P2(b): hipMemCreate's allocation properties ------------------------
  // hipMemAllocationProp never reaches the archive, and playback hardcodes
  // location.id = 0 and type = Pinned. Setting location.id to a non-zero
  // device here would make the loss visible on a multi-GPU host; on one GPU
  // the recorded-vs-replayed type is still the observable difference.
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

  // ---- P2(d): hipStreamBatchMemOp's op list -------------------------------
  // The op array is a plain (non-const) pointer, so the mechanical detector in
  // derive_manifest.py does not flag it — api_matrix.yaml marks it by hand.
  // The ops themselves never reach the archive.
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

  // ---- External memory / semaphore import ---------------------------------
  // Both take a const descriptor pointer. There is no portable handle to
  // import here, so these fail at capture time — which is the point: the
  // recorded event still carries a dangling host address for the descriptor.
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

  // ---- Multi-device launch descriptors ------------------------------------
  // hipLaunchParams carries the kernel as a host function address, which is
  // the payload-loss shape this tier is about, in the launch family that has
  // the least chance of surviving it. One device is enough to record it.
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

  // ---- Final D2H ----------------------------------------------------------
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<int> h(kN, 0);
  HIP_CHECK(hipMemcpy(h.data(), d, kSZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < kN; ++i) REQUIRE(h[i] == 3);

  HIP_CHECK(hipStreamDestroy(s));
  HIP_CHECK(hipFree(d));
}

// ===========================================================================
// T2 — silent failures: handlers that return hipSuccess and do nothing.
//
// Section 9 ranks these above breadth, because a replay that reports success
// while producing wrong numbers is worse than one that refuses to run. Four
// families:
//
//   graph mutation      ggml/llama.cpp mutates its instantiated graph per
//                       token; the graph was legitimately stream-captured, so
//                       H1's fail-loud gate never fires and every token
//                       replays the first token's parameters (section 8.4a).
//   stream value ops    XLA's VMM allocator waits on a value the no-op never
//                       writes; replay hangs (section 7, H2).
//   hipHostAlloc        allocates nothing. cudaHostAlloc hipifies here while
//                       cudaMallocHost hipifies to the working spelling
//                       (section 8.6).
//   __device__ symbols  MoRI's globalGpuStates arrives through
//                       hipModuleGetGlobal / hipMemcpyToSymbol, both no-ops,
//                       so replayed kernels dereference zeros.
//
// Note the D2H at the end is written by a *chevron* launch, not by any of the
// above: the archive must still have a valid blob for the roundtrip to mean
// anything.
//
// Final blob: d[i] == 5.
// ===========================================================================
TEST_CASE("Unit_HRR_ApiMatrix_SilentFailure_Direct", "[.][hrr-direct]") {
  HIP_CHECK(hipSetDevice(0));

  int* d = nullptr;
  HIP_CHECK(hipMalloc(&d, kSZ));
  hipStream_t s = nullptr;
  HIP_CHECK(hipStreamCreate(&s));

  // ---- Legacy pinned-alloc spellings --------------------------------------
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

  // ---- __device__ symbol access -------------------------------------------
  {
    std::vector<char> co = compile_rtc();
    hipModule_t mod = nullptr;
    if (hipModuleLoadData(&mod, co.data()) == hipSuccess) {
      hipDeviceptr_t sym = 0;
      size_t sym_bytes = 0;
      // No-op at replay: the replayed module never yields the symbol address,
      // so anything the program writes through it lands nowhere.
      (void)hipModuleGetGlobal(&sym, &sym_bytes, mod, "mtx_rtc_global");
      (void)hipModuleUnload(mod);
    }
  }

  // The statically registered spelling of the same hazard. hipMemcpyToSymbol
  // is how a program initialises a __device__ global before the kernels that
  // read it run; if it is a no-op at replay, every replayed kernel sees zeros
  // and produces plausible wrong numbers rather than failing.
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

  // ---- Stream value operations (H2) ---------------------------------------
  // Write then wait on the same location. At capture both succeed; at replay
  // the write is a no-op, so a real waiter would never be satisfied. Use a
  // pinned host flag so this cannot wedge the *capture* run.
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

  // ---- Cooperative module launch ------------------------------------------
  // Live on Instinct through MIOpen's Winograd Fury RxS solver. The document
  // predicted a deadlock at the grid-wide barrier; the handler in fact errors,
  // which is the better failure and is what the matrix asserts.
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

  // ---- Graph mutation family ----------------------------------------------
  // Capture a graph the supported way, instantiate it, then mutate it the way
  // llama.cpp does per token. Every mutation below is a no-op at replay.
  {
    hipGraph_t graph = nullptr;
    hipGraphExec_t exec = nullptr;
    int* d_copy = nullptr;
    HIP_CHECK(hipMalloc(&d_copy, kSZ));
    int host_hits = 0;

    HIP_CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));
    HIP_CHECK(hipMemsetAsync(d, 0, kSZ, s));
    hipLaunchKernelGGL(hrr_mtx_fill, dim3((kN + 255) / 256), dim3(256), 0, s,
                       d, 5, kN);
    // A memcpy node and a host node so the mutation loop below has one of each
    // kind to set params on. Without them the captured graph is kernel and
    // memset only, and the memcpy/host halves of the NOOP family — which is
    // most of section 8.4a — have no call site at all.
    HIP_CHECK(hipMemcpyAsync(d_copy, d, kSZ, hipMemcpyDeviceToDevice, s));
    (void)hipLaunchHostFunc(s, mtx_host_fn, &host_hits);
    HIP_CHECK(hipStreamEndCapture(s, &graph));
    REQUIRE(graph != nullptr);
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

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
          // The per-token mutation: same node, new arguments.
          (void)hipGraphKernelNodeSetParams(nodes[i], &kp);
          (void)hipGraphExecKernelNodeSetParams(exec, nodes[i], &kp);

          // The generic hipGraphNodeParams spelling of the same mutation. It
          // is a separate entry point with its own NOOP handler, so it needs
          // its own call site. There is no hipGraphNodeGetParams to read the
          // union back, hence building it from the kernel params above — and
          // it has to be a valid setting, because the capture shims only
          // record calls that return hipSuccess.
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
        // The 1D convenience spelling, and the driver-API spelling that takes
        // HIP_MEMCPY3D instead of hipMemcpy3DParms. Same node, three different
        // ways in, all no-ops at replay.
        (void)hipGraphMemcpyNodeSetParams1D(nodes[i], d_copy, d, kSZ,
                                            hipMemcpyDeviceToDevice);
        (void)hipGraphExecMemcpyNodeSetParams1D(exec, nodes[i], d_copy, d, kSZ,
                                                hipMemcpyDeviceToDevice);
        HIP_MEMCPY3D drv{};
        drv.srcMemoryType = hipMemoryTypeDevice;
        drv.srcDevice = reinterpret_cast<hipDeviceptr_t>(d);
        drv.dstMemoryType = hipMemoryTypeDevice;
        drv.dstDevice = reinterpret_cast<hipDeviceptr_t>(d_copy);
        drv.WidthInBytes = kSZ;
        drv.Height = 1;
        drv.Depth = 1;
        (void)hipDrvGraphMemcpyNodeSetParams(nodes[i], &drv);
        (void)hipDrvGraphExecMemcpyNodeSetParams(exec, nodes[i], &drv,
                                                 nullptr);
      } else if (type == hipGraphNodeTypeHost) {
        hipHostNodeParams hp{};
        if (hipGraphHostNodeGetParams(nodes[i], &hp) == hipSuccess) {
          (void)hipGraphHostNodeSetParams(nodes[i], &hp);
          (void)hipGraphExecHostNodeSetParams(exec, nodes[i], &hp);
        }
      }
    }

    // hipGraphExecUpdate — the whole-graph form of the same mutation.
    {
      hipGraphExecUpdateResult update_result{};
      hipGraphNode_t error_node = nullptr;
      (void)hipGraphExecUpdate(exec, graph, &error_node, &update_result);
    }

    // The batch-memop and driver-memset members of the same NOOP family. A
    // stream capture never produces either node kind, so this is the one place
    // they can be built: explicitly, into a second graph, which is then
    // instantiated so the exec-level spellings have an exec to mutate. Both
    // node constructors reject a null hipCtx_t, hence the context.
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

        hipGraphExec_t built_exec = nullptr;
        if (hipGraphInstantiate(&built_exec, built, nullptr, nullptr, 0)
                == hipSuccess) {
          if (have_batch)
            (void)hipGraphExecBatchMemOpNodeSetParams(built_exec, batch_node,
                                                      &batch_params);
          if (have_memset)
            (void)hipDrvGraphExecMemsetNodeSetParams(
                built_exec, drv_memset_node, &memset_params, mutation_ctx);
          (void)hipGraphExecDestroy(built_exec);
        }
        (void)hipGraphDestroy(built);
      }
      if (mutation_ctx) (void)hipCtxDestroy(mutation_ctx);
    }

    HIP_CHECK(hipGraphLaunch(exec, s));
    HIP_CHECK(hipStreamSynchronize(s));
    HIP_CHECK(hipGraphExecDestroy(exec));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipFree(d_copy));
  }

  // ---- Final D2H ----------------------------------------------------------
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<int> h(kN, 0);
  HIP_CHECK(hipMemcpy(h.data(), d, kSZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < kN; ++i) REQUIRE(h[i] == 5);

  HIP_CHECK(hipStreamDestroy(s));
  HIP_CHECK(hipFree(d));
}

// ===========================================================================
// T3 — multi-GPU: device identity, peer access, IPC and VMM.
//
// Section 5: the multi-GPU delta is 42 APIs at roughly 31% faithful, and the
// problem is structural rather than per-API — events carry no device ID and
// alloc_map has no device field, so a replay cannot know which GPU an
// allocation belonged to. Needs two visible devices; the runner supplies them
// with HIP_VISIBLE_DEVICES=6,7.
//
// Final blob: d1[i] == 0x5A5A5A5A, copied device-to-device across the pair.
// ===========================================================================
TEST_CASE("Unit_HRR_ApiMatrix_MultiGpu_Direct", "[.][hrr-direct]") {
  int ndev = 0;
  HIP_CHECK(hipGetDeviceCount(&ndev));
  if (ndev < 2) HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);

  constexpr int kVal = 0x5A5A5A5A;
  const int src_dev = 0;
  const int dst_dev = 1;

  HIP_CHECK(hipSetDevice(src_dev));

  // ---- Topology queries ----------------------------------------------------
  int can_access = 0;
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access, src_dev, dst_dev));
  {
    int p2p_value = 0;
    (void)hipDeviceGetP2PAttribute(&p2p_value, hipDevP2PAttrAccessSupported,
                                   src_dev, dst_dev);
    uint32_t link_type = 0, hop_count = 0;
    (void)hipExtGetLinkTypeAndHopCount(src_dev, dst_dev, &link_type, &hop_count);
  }

  int* d0 = nullptr;
  HIP_CHECK(hipMalloc(&d0, kSZ));
  HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d0), kVal, kN));
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipSetDevice(dst_dev));
  int* d1 = nullptr;
  HIP_CHECK(hipMalloc(&d1, kSZ));
  HIP_CHECK(hipMemset(d1, 0, kSZ));
  HIP_CHECK(hipDeviceSynchronize());

  // ---- Peer access and peer copies ----------------------------------------
  HIP_CHECK(hipSetDevice(src_dev));
  const bool peer_enabled =
      can_access && hipDeviceEnablePeerAccess(dst_dev, 0) == hipSuccess;

  HIP_CHECK(hipMemcpyPeer(d1, dst_dev, d0, src_dev, kSZ));
  HIP_CHECK(hipDeviceSynchronize());
  {
    hipStream_t peer_stream = nullptr;
    HIP_CHECK(hipStreamCreate(&peer_stream));
    (void)hipMemcpyPeerAsync(d1, dst_dev, d0, src_dev, kSZ, peer_stream);
    (void)hipStreamSynchronize(peer_stream);
    HIP_CHECK(hipStreamDestroy(peer_stream));
  }

  // ---- VMM: reserve, create, map, set access, unmap -----------------------
  // hipMemCreate's properties never reach the archive and playback hardcodes
  // location.id = 0, so on a two-GPU capture the replayed heap is on the wrong
  // device. That is the P2(b) failure, observed here rather than argued.
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

  // The shareable-handle family is deliberately absent here: replaying
  // hipMemExportToShareableHandle kills the replay process, and one such event
  // in this archive would cost every observation after it. It has its own
  // workload, Unit_HRR_ApiMatrix_ShareableHandle_Direct, for that reason.

  // ---- Verify the peer copy landed, and leave the blob for replay ---------
  HIP_CHECK(hipSetDevice(dst_dev));
  std::vector<int> h(kN, 0);
  HIP_CHECK(hipMemcpy(h.data(), d1, kSZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < kN; ++i) REQUIRE(h[i] == kVal);

  HIP_CHECK(hipFree(d1));
  HIP_CHECK(hipSetDevice(src_dev));
  if (peer_enabled) (void)hipDeviceDisablePeerAccess(dst_dev);
  HIP_CHECK(hipFree(d0));
}

// ===========================================================================
// T3 — the fabric / shareable-handle family, on its own.
//
// MoRI's VMHeap and the PyTorch symmetric-memory path both hand memory between
// ranks with these four (UC2a, "Fabric / shareable handles"). Three of them are
// NOOP at replay. The fourth, hipMemExportToShareableHandle, has a real
// generated handler that writes the exported fd through the *capture-time*
// address of the caller's `int fd` — an address that means nothing in the
// replay process — so replaying it does not fail, it segfaults.
//
// That is why this is a separate workload rather than part of
// Unit_HRR_ApiMatrix_MultiGpu_Direct: a crash ends the replay, and every event
// recorded after it goes unobserved. Isolating it costs one small archive and
// keeps the crash from erasing the peer and VMM coverage next door. The three
// NOOP calls come first for the same reason, so only the import — which needs
// an fd the export has to produce — is left behind the crash.
//
// No final blob: nothing here writes device memory.
// ===========================================================================
TEST_CASE("Unit_HRR_ApiMatrix_ShareableHandle_Direct", "[.][hrr-direct]") {
  HIP_CHECK(hipSetDevice(0));
  if (!device_attr(hipDeviceAttributeVirtualMemoryManagementSupported, 0))
    HIP_SKIP_TEST(HipTest::SkipReason::kVmmUnsupported);

  int* d = nullptr;
  HIP_CHECK(hipMalloc(&d, kSZ));
  HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d), 0x11, kN));

  // dmabuf export of an ordinary allocation: MoRI's RDMA path
  // (src/application/transport/rdma/rdma.cpp:463).
  // Absent from amdhip.def.in, so HrrTest.exe cannot link it on Windows.
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
  // Without an exportable handle type the export below fails, and a failed
  // call is never recorded — the whole family would drop out of the matrix
  // rather than being observed as broken.
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

        // Everything below this point is recorded but unobservable: the export
        // is the event that ends the replay.
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

  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipFree(d));
}

// ===========================================================================
// T4 — breadth: the APIs no existing workload reaches.
//
// The suite's 42 workloads already cover the device, stream, event, memcpy,
// memset, mempool, module, occupancy and graph families, so this workload is
// deliberately the remainder rather than another pass over the same ground:
// error-string and last-error handling, the library and link loaders, batch
// memory operations, user objects, device resource partitioning, and the
// stream-per-thread (_spt) spellings.
//
// Final blob: d[i] == 11.
// ===========================================================================
TEST_CASE("Unit_HRR_ApiMatrix_Breadth_Direct", "[.][hrr-direct]") {
  HIP_CHECK(hipSetDevice(0));

  int* d = nullptr;
  HIP_CHECK(hipMalloc(&d, kSZ));
  hipStream_t s = nullptr;
  HIP_CHECK(hipStreamCreate(&s));

  // ---- Error handling ------------------------------------------------------
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

  // ---- Stream identity and attributes --------------------------------------
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

  // ---- CU-masked streams (RCCL's channel isolation path) -------------------
  // hipExtStreamGetCUMask rejects a buffer smaller than one bit per CU, so the
  // word count has to come from the device rather than a constant: a fixed 8
  // words covers parts up to 256 CUs but not gfx94X's 304, and a rejected call
  // is never recorded, which drops the API out of the tier's coverage.
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

  // ---- Host callbacks ------------------------------------------------------
  // Both record a host function pointer belonging to this process, which means
  // nothing in the replaying one.
  {
    int callback_hits = 0;
    (void)hipStreamAddCallback(s, mtx_stream_callback, &callback_hits, 0);
    (void)hipLaunchHostFunc(s, mtx_host_fn, &callback_hits);
    (void)hipStreamSynchronize(s);
  }

  // ---- User objects --------------------------------------------------------
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

  // ---- Library loader (the hipModule successor) ----------------------------
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
      // hipLibraryGetGlobal / hipLibraryGetManaged exist in the in-tree
      // dispatch table but not in the SDK headers this suite compiles
      // against, so they have no call site here. The matrix reports them as
      // not-exercised, which is the accurate answer.
      (void)hipLibraryUnload(lib);
    }
  }

  // ---- Module-level introspection -----------------------------------------
  {
    std::vector<char> co = compile_rtc();
    hipModule_t mod = nullptr;
    if (hipModuleLoadData(&mod, co.data()) == hipSuccess) {
      unsigned int fn_count = 0;
      (void)hipModuleGetFunctionCount(&fn_count, mod);
      (void)hipModuleUnload(mod);
    }
  }

  // ---- Link-time code assembly ---------------------------------------------
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

  // The device-resource / green-context family (hipDeviceGetDevResource,
  // hipDevSmResourceSplitByCount, hipGreenCtxCreate, hipExecutionCtx*) is in
  // the in-tree dispatch table but not in the SDK headers this suite compiles
  // against, so it has no call site. Those APIs come back as not-exercised.

  // ---- Profiler control ----------------------------------------------------
  (void)hipProfilerStart();
  (void)hipProfilerStop();

  // ---- Stream-per-thread spellings -----------------------------------------
  {
    HIP_CHECK(hipMemsetAsync(d, 0, kSZ, hipStreamPerThread));
    hipLaunchKernelGGL(hrr_mtx_fill, dim3((kN + 255) / 256), dim3(256), 0,
                       hipStreamPerThread, d, 11, kN);
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
  }

  // ---- Final D2H -----------------------------------------------------------
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<int> h(kN, 0);
  HIP_CHECK(hipMemcpy(h.data(), d, kSZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < kN; ++i) REQUIRE(h[i] == 11);

  HIP_CHECK(hipStreamDestroy(s));
  HIP_CHECK(hipFree(d));
}

// ===========================================================================
// T4 — the explicit graph node-construction API.
//
// The other graph coverage in this suite builds graphs by stream capture,
// which is how PyTorch, vLLM and llama.cpp build theirs. This workload builds
// one the other way, node by node, because that is the half of the graph
// surface HRR answers with ERROR_STUB: "explicit (node-API) graph construction
// is NOT supported". Around 55 APIs sit behind that sentence, and a matrix
// that never calls them cannot notice if one quietly starts doing something.
//
// The external-semaphore node family is absent on purpose: those nodes need an
// imported semaphore, which needs a real fd from another process, and a call
// that fails is never recorded. They stay not-exercised, which is accurate.
//
// Final blob: d[i] == 17, written by a graph launched from an explicit node.
// ===========================================================================
TEST_CASE("Unit_HRR_ApiMatrix_GraphNodes_Direct", "[.][hrr-direct]") {
  HIP_CHECK(hipSetDevice(0));

  int* d = nullptr;
  int* d_copy = nullptr;
  HIP_CHECK(hipMalloc(&d, kSZ));
  HIP_CHECK(hipMalloc(&d_copy, kSZ));
  hipStream_t s = nullptr;
  HIP_CHECK(hipStreamCreate(&s));
  hipEvent_t ev_a = nullptr, ev_b = nullptr;
  HIP_CHECK(hipEventCreateWithFlags(&ev_a, hipEventDisableTiming));
  HIP_CHECK(hipEventCreateWithFlags(&ev_b, hipEventDisableTiming));

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // ---- One node of every kind ---------------------------------------------
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

  // Symbol nodes: the graph spelling of the UC2 risk-5 hazard.
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

  // A second graph for the node kinds that are only ever constructed and
  // queried here. Instantiating a graph that contains them crashes inside HIP
  // on this stack, and the matrix wants the *construction* APIs recorded, not
  // a demonstration of that. Keeping them out of `graph` is what lets the
  // executable-graph half of this workload run at all.
  hipGraph_t build_only = nullptr;
  (void)hipGraphCreate(&build_only, 0);

  // Graph-owned allocations: the pool a graph allocates from is rebuilt at
  // replay, so the addresses these nodes hand out are not the recorded ones.
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

  // Child graph, and the generic hipGraphAddNode spelling.
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

  // Batch memory operations as a graph node: XLA's VMM allocator shape.
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

  // Driver-API node spellings.
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

  // ---- Topology queries and edits -----------------------------------------
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

  // Kernel-node attributes: the cooperative flag and the priority CK sets.
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

  // Clone, and the node-identity mapping between original and clone.
  {
    hipGraph_t clone = nullptr;
    if (hipGraphClone(&clone, graph) == hipSuccess) {
      hipGraphNode_t in_clone = nullptr;
      (void)hipGraphNodeFindInClone(&in_clone, kernel_node, clone);
      (void)hipGraphDestroy(clone);
    }
  }

  // User objects tied to a graph's lifetime.
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

  // ---- Instantiate, mutate the executable, launch -------------------------
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

  // The params-struct spelling of instantiation, which reports which node
  // failed rather than only that something did.
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

  // ---- Capturing into a graph that already has nodes ----------------------
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

  // Graph memory pool accounting.
  {
    size_t used = 0;
    (void)hipDeviceGetGraphMemAttribute(0, hipGraphMemAttrUsedMemHigh, &used);
    size_t reset = 0;
    (void)hipDeviceSetGraphMemAttribute(0, hipGraphMemAttrUsedMemHigh, &reset);
  }

  if (child != nullptr) (void)hipGraphDestroy(child);
  if (build_only != nullptr) (void)hipGraphDestroy(build_only);
  (void)hipGraphDestroy(graph);

  // ---- Final D2H -----------------------------------------------------------
  hipLaunchKernelGGL(hrr_mtx_fill, dim3((kN + 255) / 256), dim3(256), 0, s, d,
                     17, kN);
  HIP_CHECK(hipStreamSynchronize(s));
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<int> h(kN, 0);
  HIP_CHECK(hipMemcpy(h.data(), d, kSZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < kN; ++i) REQUIRE(h[i] == 17);

  (void)hipEventDestroy(ev_a);
  (void)hipEventDestroy(ev_b);
  HIP_CHECK(hipStreamDestroy(s));
  HIP_CHECK(hipFree(d_copy));
  HIP_CHECK(hipFree(d));
}

// ===========================================================================
// T4 — the legacy driver-context ABI and the stream-per-thread spellings.
//
// Two families no modern workload reaches, for opposite reasons. The hipCtx*
// and hipConfigureCall/hipSetupArgument/hipLaunchByPtr surfaces are the
// pre-runtime-API spellings, still exported and still reachable from old
// code. The _spt spellings are what any translation unit compiled with
// -fgpu-default-stream=per-thread calls instead of the plain names — the same
// operations under different API IDs, and a recording made by such a program
// contains those IDs rather than the ones every other workload here produces.
//
// Final blob: d[i] == 19.
// ===========================================================================
TEST_CASE("Unit_HRR_ApiMatrix_LegacySurface_Direct", "[.][hrr-direct]") {
  HIP_CHECK(hipSetDevice(0));

  int* d = nullptr;
  HIP_CHECK(hipMalloc(&d, kSZ));
  hipStream_t s = nullptr;
  HIP_CHECK(hipStreamCreate(&s));
  hipEvent_t ev = nullptr;
  HIP_CHECK(hipEventCreateWithFlags(&ev, hipEventDisableTiming));

  // The driver-context family is NOT here: replaying it leaves the process
  // without a usable current device and the next API to ask for one segfaults.
  // It has its own workload, Unit_HRR_ApiMatrix_LegacyCtx_Direct.
  (void)hipSetDeviceFlags(hipDeviceScheduleAuto);

  // ---- The pre-chevron launch ABI, minus the argument pushes --------------
  // hipConfigureCall with a kernel that takes no arguments, so no
  // hipSetupArgument is needed. That half of the ABI lives in
  // Unit_HRR_ApiMatrix_LegacyLaunch_Direct because it segfaults replay, and
  // keeping it out of here is what lets the ~80 APIs below be observed at all.
  // hipLaunchByPtr itself is lowered away before capture; it is here to keep
  // the sequence a real one rather than for its own event.
  //
  // The launch it issues is NOT here: hipLaunchByPtr is recorded as an
  // ordinary hipModuleLaunchKernel, and that event segfaults replay, which
  // would cost the ~80 APIs below. It lives with the crash in
  // Unit_HRR_ApiMatrix_LegacyLaunch_Direct instead.
  {
    (void)hipConfigureCall(dim3(1), dim3(64), 0, s);
  }

  // ---- Function and kernel attributes -------------------------------------
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

  // ---- Module and library loaders in their remaining spellings ------------
  {
    std::vector<char> co = compile_rtc();
    hipModule_t fat_module = nullptr;
    // A code object is not a fat binary, so this is expected to fail. It is
    // called anyway: a failing call is not recorded, and the matrix reporting
    // the API as unreachable is the accurate outcome rather than a silent gap.
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

  // ---- Pointer and allocation introspection -------------------------------
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

  // ---- Deprecated device-property spellings -------------------------------
  {
    hipDeviceProp_tR0000 old_props{};
    if (hipGetDevicePropertiesR0000(&old_props, 0) == hipSuccess) {
      int chosen = 0;
      (void)hipChooseDeviceR0000(&chosen, &old_props);
    }
  }

  // ---- Batched memory operations ------------------------------------------
  // The 1D batch form is what a collective's gather/scatter step turns into.
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

  // ---- Driver 2D copy with no alignment requirement ------------------------
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

  // ---- Driver entry-point lookup ------------------------------------------
  // Both of these are NOOP at replay. Their hipGetProcAddress_spt sibling is
  // not, and it segfaults, so it lives in
  // Unit_HRR_ApiMatrix_ProcAddress_Direct.
  {
    void* entry = nullptr;
    hipDriverEntryPointQueryResult status = hipDriverEntryPointSuccess;
    (void)hipGetDriverEntryPoint("hipMalloc", &entry, 0, &status);
    void* spt_entry = nullptr;
    hipDriverEntryPointQueryResult spt_status = hipDriverEntryPointSuccess;
    (void)hipGetDriverEntryPoint_spt("hipMalloc", &spt_entry, 0, &spt_status);
  }

  // ---- Stream-per-thread spellings ----------------------------------------
  // The two _spt kernel-launch entry points are not here: both replay a
  // capture-time host function address and segfault, so they have a workload
  // each (Unit_HRR_ApiMatrix_SptKernelLaunch_Direct,
  // Unit_HRR_ApiMatrix_SptCoopLaunch_Direct). The callback pair below takes a
  // recorded host pointer too and does not crash — their plain spellings are
  // already measured as a real handler and a clean error return.
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

  // ---- Capture, in the _spt spellings and with the v2 query ---------------
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

  // ---- Final D2H -----------------------------------------------------------
  hipLaunchKernelGGL(hrr_mtx_fill, dim3((kN + 255) / 256), dim3(256), 0, s, d,
                     19, kN);
  HIP_CHECK(hipStreamSynchronize(s));
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<int> h(kN, 0);
  HIP_CHECK(hipMemcpy(h.data(), d, kSZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < kN; ++i) REQUIRE(h[i] == 19);

  (void)hipEventDestroy(ev);
  HIP_CHECK(hipStreamDestroy(s));
  HIP_CHECK(hipFree(d));
}

// ===========================================================================
// T4 — the stream-per-thread kernel launches, one archive each.
//
// A translation unit compiled -fgpu-default-stream=per-thread calls these
// instead of the plain names, so this is not an exotic surface: it is what an
// ordinary chevron launch becomes under a common compiler flag. The plain
// hipLaunchKernel is lowered to hipModuleLaunchKernel before capture sees it
// and replays faithfully; the _spt spelling is recorded under its own name
// with the host function address as the identity of the kernel, and replay
// hands that address to the runtime in a process where it means nothing.
//
// Two workloads rather than one because a crash ends the archive it is in, so
// two crashing APIs cannot be measured in the same recording. Only the plain
// launch turned out to crash — the cooperative one fails cleanly with 13
// (invalid device symbol), the same missing information caught a layer
// earlier. It keeps its own archive so that regressing to a crash would cost
// nothing but its own coverage.
// ===========================================================================
TEST_CASE("Unit_HRR_ApiMatrix_SptKernelLaunch_Direct", "[.][hrr-direct]") {
  HIP_CHECK(hipSetDevice(0));

  int* d = nullptr;
  HIP_CHECK(hipMalloc(&d, kSZ));
  int fill_val = 19;
  int n_arg = kN;
  void* args[] = {&d, &fill_val, &n_arg};

  // Last recorded event of this archive, on purpose.
  (void)hipLaunchKernel_spt(reinterpret_cast<const void*>(hrr_mtx_fill),
                            dim3((kN + 255) / 256), dim3(256), args, 0,
                            hipStreamPerThread);
  (void)hipStreamSynchronize(hipStreamPerThread);
  HIP_CHECK(hipFree(d));
}

TEST_CASE("Unit_HRR_ApiMatrix_SptCoopLaunch_Direct", "[.][hrr-direct]") {
  HIP_CHECK(hipSetDevice(0));
  if (!device_attr(hipDeviceAttributeCooperativeLaunch, 0))
    HIP_SKIP_TEST(HipTest::SkipReason::kCooperativeLaunchUnsupported);

  int* d = nullptr;
  HIP_CHECK(hipMalloc(&d, kSZ));
  int n_arg = kN;
  void* args[] = {&d, &n_arg};

  // Last recorded event of this archive, on purpose.
  (void)hipLaunchCooperativeKernel_spt(
      reinterpret_cast<const void*>(hrr_mtx_coop), dim3(1), dim3(64), args, 0,
      hipStreamPerThread);
  (void)hipStreamSynchronize(hipStreamPerThread);
  HIP_CHECK(hipFree(d));
}

// ===========================================================================
// T4 — symbol lookup by name, which is a string the archive never stored.
//
// hipGetProcAddress_spt(const char* symbol, ...) is recorded as a bare address
// with none of the characters behind it, and the generated handler replays the
// address. Measured with a backtrace: SIGSEGV inside
// hip::hipGetProcAddress_common at hip_device.cpp:880, with gdb reporting
// "Cannot access memory" for the symbol argument. Same defect as
// hipSetupArgument, on an input string rather than an argument blob.
//
// Its plain and _spt driver-entry-point siblings take the same const char* and
// survive only because they are NOOP at replay. They are called here too, to
// keep the comparison in one archive.
//
// No final blob: nothing here touches device memory.
// ===========================================================================
TEST_CASE("Unit_HRR_ApiMatrix_ProcAddress_Direct", "[.][hrr-direct]") {
  HIP_CHECK(hipSetDevice(0));

  void* entry = nullptr;
  hipDriverEntryPointQueryResult status = hipDriverEntryPointSuccess;
  (void)hipGetDriverEntryPoint("hipMalloc", &entry, 0, &status);

  void* proc = nullptr;
  hipDriverProcAddressQueryResult proc_status = HIP_GET_PROC_ADDRESS_SUCCESS;
  (void)hipGetProcAddress("hipMalloc", &proc, HIP_VERSION, 0, &proc_status);

  // Last recorded event of this archive, on purpose.
  void* spt_proc = nullptr;
  hipDriverProcAddressQueryResult spt_status = HIP_GET_PROC_ADDRESS_SUCCESS;
  (void)hipGetProcAddress_spt("hipMalloc", &spt_proc, HIP_VERSION, 0,
                              &spt_status);
}

// ===========================================================================
// T4 — the driver-context family, which poisons the replay it runs in.
//
// Measured, with a backtrace: replaying this sequence ends with the next
// hipGetDevice taking SIGSEGV inside hip::hipGetDevice at
// hip_device_runtime.cpp:705 (`*deviceId = device->deviceId()`) — there is no
// current device left to ask. The family replays asymmetrically:
// hipCtxCreate and hipCtxPopCurrent are NOOP, hipCtxPushCurrent and
// hipCtxDestroy are real. So replay pushes a capture-time hipCtx_t that means
// nothing here, then destroys a context it never created, and the thread is
// left pointing at nothing. Nothing reports an error; the process just dies
// several events later, at an innocent API.
//
// A probe run narrowed it down: the replay dies at the hipGetDevice that
// follows hipCtxPushCurrent, before ever reaching hipCtxDestroy. So it is the
// NOOP-pop / real-push pair that does the damage — replay pushes a hipCtx_t
// value that only meant something in the recording process — and hipCtxDestroy
// is survivable on its own. The two are separated here accordingly: the
// destroy runs against its own context and is observed, and the push is the
// last event in the archive so nothing sits behind the crash.
//
// The hipGetDevice calls are probes rather than coverage. The first one
// standing between destroy and push is what makes the attribution above
// re-checkable on every run instead of a claim in a comment.
//
// No final blob: this workload deliberately ends by breaking its own context.
// ===========================================================================
TEST_CASE("Unit_HRR_ApiMatrix_LegacyCtx_Direct", "[.][hrr-direct]") {
  HIP_CHECK(hipSetDevice(0));

  hipDevice_t dev = 0;
  (void)hipDeviceGet(&dev, 0);
  (void)hipSetDeviceFlags(hipDeviceScheduleAuto);

  hipCtx_t primary = nullptr;
  if (hipDevicePrimaryCtxRetain(&primary, dev) == hipSuccess) {
    (void)hipDevicePrimaryCtxSetFlags(dev, hipDeviceScheduleAuto);
    (void)hipDevicePrimaryCtxRelease(dev);
  }

  // First context: everything except the push, then torn down.
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

  // Still alive after a replayed hipCtxDestroy? Anything after this line in
  // the verbose replay log is the evidence that the destroy is not the
  // problem.
  int probe_device = -1;
  (void)hipGetDevice(&probe_device);

  // Second context, for the pair that is. hipCtxPushCurrent is the last
  // recorded event of this archive, on purpose.
  hipCtx_t ctx2 = nullptr;
  if (hipCtxCreate(&ctx2, 0, dev) == hipSuccess) {
    hipCtx_t popped = nullptr;
    if (hipCtxPopCurrent(&popped) == hipSuccess)
      (void)hipCtxPushCurrent(popped);
  }
}

// ===========================================================================
// T4 — the half of the pre-chevron launch ABI that segfaults replay.
//
// hipSetupArgument(const void* arg, size_t size, size_t offset) pushes `size`
// bytes from `arg` into the pending kernarg buffer. HRR records `arg` as a
// bare uint64_t and the generated handler replays it verbatim
// (hip_playback_generated.cpp: hipSetupArgument((const void*)a->arg, ...)),
// so the real handler reads `size` bytes from a capture-time host address in a
// process that never allocated it. That is a segfault, not an error return.
//
// Isolated for the same reason the shareable-handle export is: a crash ends
// the replay, and this one used to sit at event 31 of the legacy surface
// archive, hiding the ~80 APIs recorded after it. hipLaunchByPtr cannot be
// exercised here — it has to come after the pushes — so it is covered in
// Unit_HRR_ApiMatrix_LegacySurface_Direct with a kernel that takes no
// arguments instead.
//
// No final blob: the launch this configures is never issued.
// ===========================================================================
TEST_CASE("Unit_HRR_ApiMatrix_LegacyLaunch_Direct", "[.][hrr-direct]") {
  HIP_CHECK(hipSetDevice(0));

  int* d = nullptr;
  HIP_CHECK(hipMalloc(&d, kSZ));
  hipStream_t s = nullptr;
  HIP_CHECK(hipStreamCreate(&s));

  // A chevron launch of the same no-argument kernel, first, to separate two
  // explanations for the crash the legacy path produces: a kernel with an
  // empty kernarg buffer, or the pre-chevron ABI specifically. This one is an
  // ordinary hipModuleLaunchKernel event, so if the empty buffer were the
  // problem the replay would die here instead.
  hipLaunchKernelGGL(hrr_mtx_noargs, dim3(1), dim3(64), 0, s);
  HIP_CHECK(hipStreamSynchronize(s));

  int fill_val = 19;
  int n_arg = kN;
  if (hipConfigureCall(dim3((kN + 255) / 256), dim3(256), 0, s)
          == hipSuccess) {
    // Three pushes so the observation is unambiguous: the crash is in the
    // handler, not in some interaction with an empty argument buffer.
    (void)hipSetupArgument(&d, sizeof(d), 0);
    (void)hipSetupArgument(&fill_val, sizeof(fill_val), sizeof(d));
    (void)hipSetupArgument(&n_arg, sizeof(n_arg), sizeof(d) + sizeof(fill_val));
  }

  // Drop the configured launch rather than issuing it: hipLaunchByPtr would be
  // recorded after the crashing event and could never be observed here.
  HIP_CHECK(hipStreamDestroy(s));
  HIP_CHECK(hipFree(d));
}

// ===========================================================================
// T4 — memory-pool sharing, on its own for the same reason the VMM export is.
//
// hipMemPoolExportToShareableHandle has the same generated shape as
// hipMemExportToShareableHandle: a `void*` out-parameter replayed as the
// capture-time address. Whether it takes the process down the same way is
// exactly what this workload is here to measure, and keeping it out of the
// main breadth archive means the answer costs one small capture instead of
// everything recorded after it.
//
// The NOOP members of the family are called first so they are observed even
// if the export does crash.
//
// Final blob: d[i] == 23.
// ===========================================================================
TEST_CASE("Unit_HRR_ApiMatrix_MemPoolShare_Direct", "[.][hrr-direct]") {
  HIP_CHECK(hipSetDevice(0));

  int* d = nullptr;
  HIP_CHECK(hipMalloc(&d, kSZ));
  hipStream_t s = nullptr;
  HIP_CHECK(hipStreamCreate(&s));

  hipMemLocation location{};
  location.type = hipMemLocationTypeDevice;
  location.id = 0;

  // The default pool for the device, and the setter that redirects it.
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
      HIP_CHECK(hipStreamSynchronize(s));

      // Export data for a pointer inside the pool, and the import that
      // consumes it. Both are NOOP at replay.
      hipMemPoolPtrExportData export_data{};
      if (hipMemPoolExportPointer(&export_data, pool_ptr) == hipSuccess) {
        void* imported = nullptr;
        (void)hipMemPoolImportPointer(&imported, pool, &export_data);
      }
      (void)hipFreeAsync(pool_ptr, s);
      HIP_CHECK(hipStreamSynchronize(s));
    }

    // Everything after this line may be unobservable: if the export handler
    // dereferences the recorded address the way the VMM one does, the replay
    // ends here.
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

  // ---- Final D2H -----------------------------------------------------------
  hipLaunchKernelGGL(hrr_mtx_fill, dim3((kN + 255) / 256), dim3(256), 0, s, d,
                     23, kN);
  HIP_CHECK(hipStreamSynchronize(s));
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<int> h(kN, 0);
  HIP_CHECK(hipMemcpy(h.data(), d, kSZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < kN; ++i) REQUIRE(h[i] == 23);

  HIP_CHECK(hipStreamDestroy(s));
  HIP_CHECK(hipFree(d));
}

// ===========================================================================
// T4 — the second breadth pass: the families the first one left with no call
// site at all.
//
// Everything here is called for one reason — to find out what replay does with
// it — so almost every call is unchecked. What matters is that the call
// happens and, if it succeeds, that the archive gets an event for it.
//
// Worth knowing while reading this: HRR's capture shims record only on
// hipSuccess. An API that cannot succeed on this machine therefore cannot be
// captured, and cannot be given a replay class by any amount of test code.
// Those come back as not-exercised and are answered in api_matrix.yaml with a
// measured reason rather than another call site.
//
// Final blob: d[i] == 29.
// ===========================================================================

TEST_CASE("Unit_HRR_ApiMatrix_Breadth2_Direct", "[.][hrr-direct]") {
  HIP_CHECK(hipSetDevice(0));

  int* d = nullptr;
  int* d2 = nullptr;
  HIP_CHECK(hipMalloc(&d, kSZ));
  HIP_CHECK(hipMalloc(&d2, kSZ));
  hipStream_t s = nullptr;
  HIP_CHECK(hipStreamCreate(&s));
  hipEvent_t ev = nullptr;
  HIP_CHECK(hipEventCreateWithFlags(&ev, hipEventDisableTiming));
  std::vector<int> host(kN, 7);

  // ---- Profiler control and odds and ends ---------------------------------
  {
    (void)hipProfilerStart();
    (void)hipProfilerStop();
    (void)hipGetStreamDeviceId(s);
    // hipExtHostAlloc has an HRR event type and a dispatch-table slot but no
    // exported symbol to link against, so no caller can reach it directly.
  }

  // The green-context and device-resource-partitioning family is absent, and
  // not by choice: hipDeviceGetDevResource, hipGreenCtxCreate, the
  // hipExecutionCtx* set and their hipDevResource types are declared in the
  // rocm-systems headers but not in the ROCm SDK these tests compile against,
  // so there is no way to write the call. api_matrix.yaml carries the same
  // reason for each of them.

  // ---- Batch memory operations --------------------------------------------
  // The batched spellings of copy, prefetch and discard. A recording made by a
  // program that uses them contains these IDs and nothing else, so a gap here
  // is a gap for that whole program.
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

    // rowLength and layerHeight are in elements and the extent is in bytes,
    // and both operands need a location hint: without one the runtime rejects
    // the batch before the capture shim records anything, which is how this
    // call spent a while looking like a missing call site rather than a wrong
    // one.
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
    // The four discard spellings are missing for the same header reason as
    // the green-context family above.
    (void)hipStreamSynchronize(s);
  }

  // ---- Graph node kinds the first graph workload does not build -----------
  // The driver-flavoured node constructors take a hipCtx_t and reject a null
  // one with hipErrorInvalidValue, so this block needs a real context to build
  // anything at all.
  {
    hipDevice_t node_device = 0;
    (void)hipDeviceGet(&node_device, 0);
    hipCtx_t node_ctx = nullptr;
    (void)hipCtxCreate(&node_ctx, 0, node_device);

    hipGraph_t graph = nullptr;
    if (hipGraphCreate(&graph, 0) == hipSuccess) {
      // The generic node constructor, which takes the node kind as data.
      hipGraphNode_t generic_node = nullptr;
      hipGraphNodeParams node_params{};
      node_params.type = hipGraphNodeTypeMemset;
      node_params.memset.dst = d;
      node_params.memset.elementSize = sizeof(int);
      node_params.memset.width = 16;
      node_params.memset.height = 1;
      node_params.memset.value = 0;
      (void)hipGraphAddNode(&generic_node, graph, nullptr, 0, &node_params);

      // Batch memory operation node.
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

      // The eight external-semaphore graph-node entry points are not called.
      // They are declared and they have HRR event types, but their slots in
      // the HIP dispatch table are null in this build, so the call jumps to
      // address zero and takes the *recording program* down — measured, with
      // a backtrace through hip_table_interface.cpp:3083. There is nothing to
      // record and no way to record it.

      // Driver-flavoured memcpy node.
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

  // ---- JIT link and the rest of the library loader ------------------------
  {
    std::vector<char> code_object = compile_rtc();

    hipLinkState_t link_state = nullptr;
    if (hipLinkCreate(0, nullptr, nullptr, &link_state) == hipSuccess) {
      (void)hipLinkAddData(link_state, hipJitInputLLVMBundledBitcode,
                           code_object.data(), code_object.size(), "mtx_rtc",
                           0, nullptr, nullptr);
      (void)hipLinkAddFile(link_state, hipJitInputLLVMBundledBitcode,
                           "/tmp/hrr_matrix_rtc.co", 0, nullptr, nullptr);
      (void)hipLinkDestroy(link_state);
    }

    hipLibrary_t library = nullptr;
    if (hipLibraryLoadData(&library, code_object.data(), nullptr, nullptr, 0,
                           nullptr, nullptr, 0) == hipSuccess) {
      // hipLibraryGetGlobal / hipLibraryGetManaged would go here; same header
      // gap as the green-context family.
      hipKernel_t kernel = nullptr;
      if (hipLibraryGetKernel(&kernel, library, "mtx_rtc_fill")
              == hipSuccess) {
        const char* kernel_name = nullptr;
        (void)hipKernelGetName(&kernel_name, kernel);
      }
      (void)hipLibraryUnload(library);
    }

    // hipLibraryLoadFromFile needs a path, so give it one rather than leaving
    // the only file-based loader in the API untested.
    {
      const fs::path co_path =
          fs::temp_directory_path() / "hrr_matrix_rtc.co";
      std::ofstream out(co_path, std::ios::binary);
      out.write(code_object.data(),
                static_cast<std::streamsize>(code_object.size()));
      out.close();
      hipLibrary_t file_library = nullptr;
      if (hipLibraryLoadFromFile(&file_library, co_path.string().c_str(),
                                 nullptr, nullptr, 0, nullptr, nullptr, 0)
              == hipSuccess)
        (void)hipLibraryUnload(file_library);
      std::error_code ec;
      fs::remove(co_path, ec);
    }

    // The legacy hcc launch spelling, which needs a module-loaded function.
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
        // hipModuleLaunchCooperativeKernelMultiDevice belongs here and is
        // omitted for the same null-dispatch-slot reason as the
        // external-semaphore nodes above.
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

  // The two cluster-occupancy queries are absent for the same header reason.

  // ---- The hipArray copy spellings and the texture-object queries ---------
  // Tagged T4 rather than T5 because they are ordinary memcpy entry points
  // that happen to take an array; the texture *sampling* path is what section
  // 10 rules out, not these.
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

  // ---- Capture-dependency mutation, unconditionally -----------------------
  // The legacy-surface workload only reaches this when the capture reports a
  // dependency to replace, which it does not always do.
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

  // ---- Final D2H -----------------------------------------------------------
  hipLaunchKernelGGL(hrr_mtx_fill, dim3((kN + 255) / 256), dim3(256), 0, s, d,
                     29, kN);
  HIP_CHECK(hipStreamSynchronize(s));
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<int> final_host(kN, 0);
  HIP_CHECK(hipMemcpy(final_host.data(), d, kSZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < kN; ++i) REQUIRE(final_host[i] == 29);

  (void)hipEventDestroy(ev);
  HIP_CHECK(hipStreamDestroy(s));
  HIP_CHECK(hipFree(d2));
  HIP_CHECK(hipFree(d));
}

// ===========================================================================
// T4 — the two resets, which end the archive they are in.
//
// hipDeviceReset and hipDevicePrimaryCtxReset tear down everything the process
// has on the device. Recording them anywhere else would mean every event after
// them replays against a device that was just reset, so they get a workload
// whose whole content is the teardown.
// ===========================================================================
TEST_CASE("Unit_HRR_ApiMatrix_Reset_Direct", "[.][hrr-direct]") {
  HIP_CHECK(hipSetDevice(0));

  int* d = nullptr;
  HIP_CHECK(hipMalloc(&d, kSZ));
  hipLaunchKernelGGL(hrr_mtx_fill, dim3((kN + 255) / 256), dim3(256), 0, 0, d,
                     31, kN);
  HIP_CHECK(hipDeviceSynchronize());

  hipDevice_t dev = 0;
  (void)hipDeviceGet(&dev, 0);
  (void)hipDevicePrimaryCtxReset(dev);

  // Last recorded event of this archive, on purpose.
  (void)hipDeviceReset();
}

// ===========================================================================
// T5 — deprioritised families, skipped unless --include-deprioritised.
//
// Section 10 records verified-negative evidence for everything here: MIOpen
// has zero texture occurrences across its whole repository, shipped
// rocFFT/rocSPARSE/rocRAND import zero texture symbols, and the managed-memory
// caller that does exist (FBGEMM_GPU) belongs to a recommender workload that
// does not appear in Instinct MLPerf submissions. The workload exists so the
// matrix can still make a statement about these APIs when asked, not because
// an AI/ML path reaches them.
//
// Final blob: d[i] == 13.
// ===========================================================================
TEST_CASE("Unit_HRR_ApiMatrix_Deprioritised_Direct", "[.][hrr-direct]") {
  HIP_CHECK(hipSetDevice(0));

  int* d = nullptr;
  HIP_CHECK(hipMalloc(&d, kSZ));
  hipStream_t s = nullptr;
  HIP_CHECK(hipStreamCreate(&s));

  // ---- Managed memory ------------------------------------------------------
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

      // The _v2 spellings take a hipMemLocation where the originals take a
      // plain device ordinal. Managed memory works on this part, so unlike the
      // rest of T5 these three are ordinary covered APIs.
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

  // ---- Driver-flavoured array creation ------------------------------------
  // Deliberately outside the image-support guard below. gfx950 reports no
  // image support and both of these return hipErrorNotSupported, but unlike
  // their runtime counterparts the capture shim records the call before the
  // runtime rejects it, so they are the only two members of the array family
  // that reach an archive at all.
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

  // ---- Textures, arrays and surfaces --------------------------------------
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

  // ---- Texture and surface teardown ---------------------------------------
  // Creating either object needs image support, which this part does not have,
  // so the handle is necessarily null. The three destroy entry points accept
  // that, return success and are recorded, which is enough to hold their
  // capture and replay paths under assertion while the create side cannot be
  // reached. Both halves are declared: these here, the create side in the
  // no-image-support unreachable group.
  {
    (void)hipDestroyTextureObject(0);
    (void)hipDestroySurfaceObject(0);
    (void)hipTexObjectDestroy(0);
  }

  // ---- Cooperative and extended launch spellings with no library caller ----
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

  // Guarantee the final value regardless of which optional paths above ran.
  hipLaunchKernelGGL(hrr_mtx_fill, dim3((kN + 255) / 256), dim3(256), 0, s,
                     d, 13, kN);
  HIP_CHECK(hipStreamSynchronize(s));

  // ---- Final D2H -----------------------------------------------------------
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<int> h(kN, 0);
  HIP_CHECK(hipMemcpy(h.data(), d, kSZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < kN; ++i) REQUIRE(h[i] == 13);

  HIP_CHECK(hipStreamDestroy(s));
  HIP_CHECK(hipFree(d));
}

/**
 * @}
 */
