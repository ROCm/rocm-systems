/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_process.hh>
#include <resource_guards.hh>

#include "hipIpcHandleHex.hh"

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <thread>

namespace {
/* Owns a device pointer obtained from hipIpcOpenMemHandle, closing it on destruction so a failed
   assertion cannot leave the mapping behind. */
class IpcMappingGuard {
 public:
  IpcMappingGuard(const hipIpcMemHandle_t& handle, unsigned int flags) {
    HIP_CHECK(hipIpcOpenMemHandle(&ptr_, handle, flags));
  }

  IpcMappingGuard(const IpcMappingGuard&) = delete;
  IpcMappingGuard(IpcMappingGuard&&) = delete;
  IpcMappingGuard& operator=(const IpcMappingGuard&) = delete;
  IpcMappingGuard& operator=(IpcMappingGuard&&) = delete;

  ~IpcMappingGuard() {
    // No Catch macros, don't want to possibly throw in the destructor
    if (ptr_ != nullptr) {
      static_cast<void>(hipIpcCloseMemHandle(ptr_));
    }
  }

  /* Closes the mapping early, so that the caller can assert on the result. The destructor then has
     nothing left to do. */
  hipError_t close() {
    void* const ptr = ptr_;
    ptr_ = nullptr;
    return hipIpcCloseMemHandle(ptr);
  }

  void* ptr() const { return ptr_; }

 private:
  void* ptr_ = nullptr;
};
}  // namespace

/**
 * @addtogroup hipIpcOpenMemHandle hipIpcOpenMemHandle
 * @{
 * @ingroup DeviceTest
 * `hipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t handle, unsigned int flags)` -
 * Opens an interprocess memory handle exported from another process
 * and returns a device pointer usable in the local process.
 */

/**
 * Test Description
 * ------------------------
 *  - Handle the attempt to open memory handle in the same process
 *    that has created it.
 *      -# When the process is the same
 *        - Expected output: return `hipErrorInvalidContext`
 * Test source
 * ------------------------
 *  - unit/device/hipIpcOpenMemHandle.cc
 * Test requirements
 * ------------------------
 *  - Host specific (LINUX)
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipIpcOpenMemHandle_Negative_Open_In_Creating_Process) {
  hipDeviceptr_t ptr2;
  hipIpcMemHandle_t handle;
  LinearAllocGuard<char> alloc(LinearAllocs::hipMalloc, 1024);
  HIP_CHECK(hipIpcGetMemHandle(&handle, alloc.ptr()));
  HIP_CHECK_ERROR(
      hipIpcOpenMemHandle(reinterpret_cast<void**>(&ptr2), handle, hipIpcMemLazyEnablePeerAccess),
      hipErrorInvalidContext);
}


#if HT_LINUX
/**
 * Test Description
 * ------------------------
 *  - Checks that opening the same memory handle from a different context
 *    returns error
 *    -# When different context
 *      - Expected output: return `hipErrorInvalidResourceHandle`
 * Test source
 * ------------------------
 *  - unit/device/hipIpcOpenMemHandle.cc
 * Test requirements
 * ------------------------
 *  - Host specific (LINUX)
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipIpcOpenMemHandle_Negative_Open_In_Two_Contexts_Same_Device) {
  int fd[2];
  REQUIRE(pipe(fd) == 0);

  // The fork must be performed before the runtime is initialized(so before any API that implicitly
  // initializes it). The pipe in conjunction with wait is then used to impose total ordering
  // between parent and child process. Because total ordering is imposed regular CATCH assertions
  // should be safe to use
  auto pid = fork();
  REQUIRE(pid >= 0);
  if (pid == 0) {  // child
    REQUIRE(close(fd[1]) == 0);

    hipIpcMemHandle_t handle;
    REQUIRE(read(fd[0], &handle, sizeof(handle)) >= 0);
    REQUIRE(close(fd[0]) == 0);

    hipDeviceptr_t ptr_child;
    HIP_CHECK(hipIpcOpenMemHandle(reinterpret_cast<void**>(&ptr_child), handle,
                                  hipIpcMemLazyEnablePeerAccess));

    HIP_CHECK(hipInit(0));
    hipCtx_t ctx;
    HIP_CHECK(hipCtxCreate(&ctx, 0, 0));

    hipDeviceptr_t ptr_child_ctx;
    HIP_CHECK_ERROR(hipIpcOpenMemHandle(reinterpret_cast<void**>(&ptr_child_ctx), handle,
                                        hipIpcMemLazyEnablePeerAccess),
                    hipErrorInvalidResourceHandle);

    exit(0);
  } else {  // parent
    REQUIRE(close(fd[0]) == 0);

    hipIpcMemHandle_t handle;
    LinearAllocGuard<char> alloc(LinearAllocs::hipMalloc, 1024);
    HIP_CHECK(hipIpcGetMemHandle(&handle, alloc.ptr()));

    REQUIRE(write(fd[1], &handle, sizeof(handle)) >= 0);
    REQUIRE(close(fd[1]) == 0);

    REQUIRE(wait(NULL) >= 0);
  }
}
#endif

constexpr int DATA_SIZE = 1024 * 1024;
constexpr size_t BYTE_SIZE = DATA_SIZE * sizeof(int);
constexpr int THREADS_PER_BLOCK = 512;

static __global__ void square_kernel(int* buf) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  buf[i] = buf[i] * buf[i];
}

/**
 * Test Description
 * ------------------------
 *  - Parent process: allocates device memory, fills with i%1024,
 *    exports the IPC handle via environment variable, spawns a child
 *    process that squares the data through the shared handle, then
 *    validates the result.
 * Test source
 * ------------------------
 *  - unit/device/hipIpcOpenMemHandle.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipIpcOpenMemHandle_Multiproc) {
  LinearAllocGuard<int> d_buf(LinearAllocs::hipMalloc, BYTE_SIZE);

  std::vector<int> h_buf(DATA_SIZE);
  for (int i = 0; i < DATA_SIZE; i++) {
    h_buf[i] = i % 1024;
  }
  HIP_CHECK(hipMemcpy(d_buf.ptr(), h_buf.data(), BYTE_SIZE, hipMemcpyHostToDevice));

  hipIpcMemHandle_t handle;
  HIP_CHECK(hipIpcGetMemHandle(&handle, d_buf.ptr()));

  std::string hex = ipcHandleToHex(handle);

  hip::SpawnProc child(getSelfExePath());
  child.setEnv("HIP_IPC_HANDLE", hex);
  REQUIRE(child.spawn("Unit_hipIpcOpenMemHandle_Multiproc_Child") == 0);
  REQUIRE(child.wait() == 0);

  HIP_CHECK(hipMemcpy(h_buf.data(), d_buf.ptr(), BYTE_SIZE, hipMemcpyDeviceToHost));
  for (int i = 0; i < DATA_SIZE; i++) {
    int expected = (i % 1024) * (i % 1024);
    REQUIRE(h_buf[i] == expected);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Child process: reads the IPC handle from the HIP_IPC_HANDLE
 *    environment variable, opens the shared device memory, squares
 *    each element via a kernel, validates the result, then closes
 *    the handle. Skipped when not launched by the parent test.
 * Test source
 * ------------------------
 *  - unit/device/hipIpcOpenMemHandle.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipIpcOpenMemHandle_Multiproc_Child) {
  const char* hex = std::getenv("HIP_IPC_HANDLE");
  if (hex == nullptr || hex[0] == '\0') {
    HIP_SKIP_TEST("This test must be launched by parent multiprocess test.");
  }

  hipIpcMemHandle_t handle = hexToIpcHandle<hipIpcMemHandle_t>(hex);

  IpcMappingGuard mapping(handle, hipIpcMemLazyEnablePeerAccess);

  square_kernel<<<dim3(DATA_SIZE / THREADS_PER_BLOCK), dim3(THREADS_PER_BLOCK)>>>(
      reinterpret_cast<int*>(mapping.ptr()));
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<int> result(DATA_SIZE);
  HIP_CHECK(hipMemcpy(result.data(), mapping.ptr(), BYTE_SIZE, hipMemcpyDeviceToHost));
  for (int i = 0; i < DATA_SIZE; i++) {
    int expected = (i % 1024) * (i % 1024);
    REQUIRE(result[i] == expected);
  }

  HIP_CHECK(mapping.close());
}

#if HT_AMD
/**
 * Cross-GPU coherency over a legacy IPC mapping.
 *
 * Communication libraries do not export plain `hipMalloc` buffers over IPC: they export
 * fine-grained (or uncached) allocations and then poll flags in them from a *different* GPU in a
 * *different* process, without any process-level barrier in between. The tests below pin that
 * property — a write made by the importing GPU must become visible to the exporting GPU, and vice
 * versa, with ordering established only by `__threadfence_system()`.
 *
 * Every wait is bounded by a wall-clock deadline so a coherency failure reports as a test failure
 * rather than a hang.
 */
namespace {
constexpr uint32_t kChildToken = 0x00C0FFEEu;
constexpr uint32_t kParentToken = 0x00BEEF01u;
constexpr uint32_t kChildDataValue = 0x12345678u;

constexpr int kSlotChildFlag = 0;
constexpr int kSlotChildData = 1;
constexpr int kSlotParentAck = 2;
constexpr size_t kSlotCount = 4;

constexpr char kHandleEnv[] = "HIP_IPC_HANDLE";
constexpr char kPeerDeviceEnv[] = "HIP_IPC_PEER_DEVICE";

/* Polls one slot of the shared buffer from the host until it reads `expected`. Returns false if
   the deadline expires first. */
bool PollSlot(const uint32_t* buf, int slot, uint32_t expected, std::chrono::seconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  do {
    uint32_t value = 0;
    const hipError_t error =
        hipMemcpy(&value, buf + slot, sizeof(value), hipMemcpyDeviceToHost);
    if (error != hipSuccess) {
      (void)hipGetLastError();
      return false;
    }
    if (value == expected) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  } while (std::chrono::steady_clock::now() < deadline);
  return false;
}
}  // namespace

/* Run by the importing process on the peer GPU: publish a data word, then the flag. */
static __global__ void ipc_publish_kernel(uint32_t* buf) {
  buf[kSlotChildData] = kChildDataValue;
  __threadfence_system();
  buf[kSlotChildFlag] = kChildToken;
  __threadfence_system();
}

/* Run by the exporting process on its own GPU: read back what the peer GPU wrote and acknowledge
   only if it is the expected value. */
static __global__ void ipc_ack_kernel(uint32_t* buf) {
  const uint32_t observed = buf[kSlotChildData];
  __threadfence_system();
  buf[kSlotParentAck] = (observed == kChildDataValue) ? kParentToken : 0u;
  __threadfence_system();
}

/**
 * Test Description
 * ------------------------
 *  - Exports a fine-grained (and, in a second section, an uncached) allocation from device 0 over
 *    a legacy IPC handle, and has a child process open it on device 1 and run a kernel against it.
 *  - Verifies a full two-way handshake with no process-level barrier:
 *    -# The importing GPU writes a data word and a flag; the exporting process must observe the
 *       flag.
 *    -# The exporting GPU reads the data word and writes an acknowledgement; the importing
 *       process must observe the acknowledgement before it exits.
 *  - Each wait is bounded, so a coherency failure fails the test instead of hanging.
 * Test source
 * ------------------------
 *  - unit/device/hipIpcOpenMemHandle.cc
 * Test requirements
 * ------------------------
 *  - Host specific (LINUX)
 *  - AMD specific (hipExtMallocWithFlags allocation flags)
 *  - Multi-device, with peer access between device 0 and device 1
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipIpcOpenMemHandle_Multiproc_CoherentAlloc_Handshake) {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  if (deviceCount < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }

  constexpr int kExporterDevice = 0;
  constexpr int kImporterDevice = 1;
  int canAccessPeer = 0;
  HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, kImporterDevice, kExporterDevice));
  if (canAccessPeer == 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kPeerAccessUnavailable);
  }

  unsigned int allocFlags = hipDeviceMallocFinegrained;
  SECTION("Fine-grained allocation") { allocFlags = hipDeviceMallocFinegrained; }
  SECTION("Uncached allocation") { allocFlags = hipDeviceMallocUncached; }

  HIP_CHECK(hipSetDevice(kExporterDevice));

  ExtMallocGuard<uint32_t> alloc(kSlotCount * sizeof(uint32_t), allocFlags);
  if (alloc.ptr() == nullptr) {
    WARN("Skipping section: this allocation flavor is not supported on this device.");
    return;
  }
  uint32_t* const buf = alloc.ptr();
  HIP_CHECK(hipMemset(buf, 0, kSlotCount * sizeof(uint32_t)));
  HIP_CHECK(hipDeviceSynchronize());

  hipIpcMemHandle_t handle;
  HIP_CHECK(hipIpcGetMemHandle(&handle, buf));

  hip::SpawnProc child(getSelfExePath());
  child.setEnv(kHandleEnv, ipcHandleToHex(handle));
  child.setEnv(kPeerDeviceEnv, std::to_string(kImporterDevice));
  REQUIRE(child.spawn("Unit_hipIpcOpenMemHandle_Multiproc_CoherentAlloc_Child") == 0);

  // The peer GPU's write must reach the exporting allocation without a process-level barrier.
  const bool sawChildFlag =
      PollSlot(buf, kSlotChildFlag, kChildToken, std::chrono::seconds(60));

  if (sawChildFlag) {
    ipc_ack_kernel<<<1, 1>>>(buf);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
  }

  const int childExit = child.wait();

  // Assert after reaping the child so a failure cannot leave it polling until its own deadline.
  REQUIRE(sawChildFlag);
  REQUIRE(childExit == 0);

  uint32_t ack = 0;
  HIP_CHECK(hipMemcpy(&ack, buf + kSlotParentAck, sizeof(ack), hipMemcpyDeviceToHost));
  REQUIRE(ack == kParentToken);
}

/**
 * Test Description
 * ------------------------
 *  - Child half of @ref Unit_hipIpcOpenMemHandle_Multiproc_CoherentAlloc_Handshake. Opens the
 *    handle published by the parent on the peer device named by `HIP_IPC_PEER_DEVICE`, publishes
 *    a data word and a flag from a kernel, then waits for the parent's acknowledgement.
 *  - Skipped when not launched by the parent test.
 * Test source
 * ------------------------
 *  - unit/device/hipIpcOpenMemHandle.cc
 * Test requirements
 * ------------------------
 *  - Host specific (LINUX)
 *  - AMD specific
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipIpcOpenMemHandle_Multiproc_CoherentAlloc_Child) {
  const char* hex = std::getenv(kHandleEnv);
  const char* peerDevice = std::getenv(kPeerDeviceEnv);
  if (hex == nullptr || hex[0] == '\0' || peerDevice == nullptr) {
    HIP_SKIP_TEST("This test must be launched by parent multiprocess test.");
  }

  // Parsed strictly: a malformed ordinal must fail the test, not silently select device 0.
  char* parse_end = nullptr;
  errno = 0;
  const long peer_ordinal = std::strtol(peerDevice, &parse_end, 10);
  INFO(kPeerDeviceEnv << "=" << peerDevice);
  REQUIRE(errno == 0);
  REQUIRE(parse_end != peerDevice);
  REQUIRE(*parse_end == '\0');
  REQUIRE(peer_ordinal >= 0);
  REQUIRE(peer_ordinal <= std::numeric_limits<int>::max());

  HIP_CHECK(hipSetDevice(static_cast<int>(peer_ordinal)));

  hipIpcMemHandle_t handle = hexToIpcHandle<hipIpcMemHandle_t>(hex);
  IpcMappingGuard mapping(handle, hipIpcMemLazyEnablePeerAccess);
  uint32_t* buf = reinterpret_cast<uint32_t*>(mapping.ptr());

  ipc_publish_kernel<<<1, 1>>>(buf);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  // The exporting GPU's acknowledgement must become visible here, again with no barrier.
  const bool sawAck = PollSlot(buf, kSlotParentAck, kParentToken, std::chrono::seconds(60));

  HIP_CHECK(mapping.close());
  REQUIRE(sawAck);
}
#endif  // HT_AMD

/**
 * End doxygen group DeviceTest.
 * @}
 */
