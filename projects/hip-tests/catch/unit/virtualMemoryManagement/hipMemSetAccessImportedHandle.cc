/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipMemSetAccess hipMemSetAccess
 * @{
 * @ingroup VirtualMemoryManagementTest
 * `hipError_t hipMemSetAccess(void* ptr, size_t size, const hipMemAccessDesc* desc,
 *                             size_t count)` -
 * 	Sets the access flags for each location specified in desc for the given virtual
 * address range.
 *
 * These tests cover granting *CPU* access (hipMemLocationTypeHost) to a mapping backed by
 * an allocation that was created in another process and brought in with
 * hipMemImportFromShareableHandle. An imported allocation carries no CPU mapping of its
 * own, so this exercises a different path than the same call on a locally created handle
 * (covered by Unit_hipMemSetAccessHost_devicealloc).
 */

#include <hip_test_common.hh>
#include "hip_vmm_common.hh"

#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>  // strsignal
#include <unistd.h>

#include <string>

#define DATA_SIZE (1 << 13)

// Device access on the imported mapping is exercised with hipMemsetD32 rather than a
// kernel launch: these tests run their HIP work in a fork()ed child, and loading a code
// object there is unreliable.
#define MEMSET_PATTERN 0xdeadbeef

namespace {

/* Short transfers are legal, and a >= 0 check also accepts read() returning 0 (peer gone), which
 * leaves the child running on an uninitialised size. */
bool WriteExact(int fd, const void* buf, size_t len) {
  const char* p = static_cast<const char*>(buf);
  while (len) {
    ssize_t n = write(fd, p, len);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    p += n;
    len -= static_cast<size_t>(n);
  }
  return true;
}

bool ReadExact(int fd, void* buf, size_t len) {
  char* p = static_cast<char*>(buf);
  while (len) {
    ssize_t n = read(fd, p, len);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    p += n;
    len -= static_cast<size_t>(n);
  }
  return true;
}

constexpr int kChildAssertFailed = 42;

/* Catch2 assertions and HIP_CHECK throw. In a forked child that unwinds past exit() and re-runs
 * the runner's teardown, turning a plain failure into a stray signal. Confine it to an exit code. */
template <typename F> void RunChildBody(F&& body) {
  int rc = 0;
  try {
    body();
  } catch (...) {
    rc = kChildAssertFailed;
  }
  fflush(nullptr);
  _exit(rc);
}

/* A raw "status == 0" check reports a child killed by SIGBUS as "135 == 0", indistinguishable
 * from a failed assertion. Say which one happened. */
void RequireChildExitedCleanly(int status) {
  if (WIFSIGNALED(status)) {
    INFO("Child process terminated by signal " << WTERMSIG(status) << " ("
                                               << strsignal(WTERMSIG(status)) << ")");
    REQUIRE(!WIFSIGNALED(status));
  }
  REQUIRE(WIFEXITED(status));
  INFO("Child process exited with code " << WEXITSTATUS(status));
  REQUIRE(WEXITSTATUS(status) == 0);
}

/* Parent -> child handshake payload. size_mem == 0 means "stand down": the parent skipped or
 * failed before exporting, so the child exits without touching HIP or the socket. */
struct ChildParams {
  size_t size_mem;
  int device;
};

/* Parent-side handle on the forked child. Nothing can wake a child parked in recvmsg() on a
 * SOCK_DGRAM socket, so the destructor unwinds whatever handoff the parent did not finish. */
class ChildProcess {
 public:
  ChildProcess(pid_t pid, int notify_fd) : pid_(pid), notify_fd_(notify_fd) {}

  ChildProcess(const ChildProcess&) = delete;
  ChildProcess& operator=(const ChildProcess&) = delete;

  ~ChildProcess() {
    if (joined_) return;
    // The parent threw before the handoff, so the child is stuck in recvmsg().
    if (committed_) {
      kill(pid_, SIGKILL);
    } else {
      StandDown();
    }
    Join(nullptr);
  }

  pid_t pid() const { return pid_; }

  // Hand the child its parameters. Only the first call has any effect.
  bool Notify(const ChildParams& params) {
    if (notified_) return false;
    notified_ = true;
    committed_ = params.size_mem != 0;
    return WriteExact(notify_fd_, &params, sizeof(params));
  }

  /* Release a child the parent never committed to. It exits 0 without making a HIP call. */
  bool StandDown() {
    ChildParams params = {};
    return Notify(params);
  }

  /* Reap the child. @p status may be null. Returns false only if waitpid() itself failed. */
  bool Join(int* status) {
    if (!joined_) {
      int raw = 0;
      pid_t reaped;
      do {
        reaped = waitpid(pid_, &raw, 0);
      } while (reaped < 0 && errno == EINTR);
      joined_ = true;
      join_ok_ = reaped == pid_;
      status_ = join_ok_ ? raw : 0;
    }
    if (status != nullptr) *status = status_;
    return join_ok_;
  }

 private:
  pid_t pid_;
  int notify_fd_;  // Only used by Notify(), so the caller may close it once the child is reaped.
  bool notified_ = false;
  bool committed_ = false;
  bool joined_ = false;
  bool join_ok_ = false;
  int status_ = 0;
};

/* checkVMMSupported() skips immediately, which between fork() and the handshake would strand the
 * child on the pipe. Report instead, so the caller can stand the child down first. */
bool VmmSupported(int device) {
  int value = 0;
  HIP_CHECK(
      hipDeviceGetAttribute(&value, hipDeviceAttributeVirtualMemoryManagementSupported, device));
  return value != 0;
}

enum class Exporter { kDeviceZero, kLastDevice };

/* Parent allocates and seeds device memory on the exporting device and exports it; the child
 * imports it, grants CPU access, reads it back and writes through it for the parent to verify. */
void ImportedDeviceMemHostAccessTest(Exporter exporter) {
  constexpr int N = DATA_SIZE;
  const size_t buffer_size = N * sizeof(int);

  int fd[2], fdSig[2];
  REQUIRE(pipe(fd) == 0);
  REQUIRE(pipe(fdSig) == 0);

  /* Drain stdio first: under ctest it is fully buffered, so a child that flushes would re-emit
   * whatever the parent had queued. */
  fflush(nullptr);

  /* fork() before the first HIP call: libhsakmt latches the fork in a pthread_atfork handler and
   * then fails every thunk call until KFD is reopened, which neither ROCr nor HIP ever does. */
  auto pid = fork();
  REQUIRE(pid >= 0);

  if (pid == 0) {  // child
    RunChildBody([&]() {
      REQUIRE(close(fd[1]) == 0);
      REQUIRE(close(fdSig[0]) == 0);

      // Wait for the parent to pick a device, size the allocation and bring up the socket.
      ChildParams params = {};
      REQUIRE(ReadExact(fd[0], &params, sizeof(params)));
      if (params.size_mem == 0) {
        // Parent skipped or failed before exporting: there is nothing to import or receive.
        REQUIRE(close(fd[0]) == 0);
        REQUIRE(close(fdSig[1]) == 0);
        return;
      }
      REQUIRE(params.size_mem >= buffer_size);

      CTX_CREATE();
      HIP_CHECK(hipSetDevice(params.device));

      // Open Socket as client
      ipcSocketCom sockObj(false);
      hipShareableHdl shHandle;
      // Signal Parent process that Child is ready to receive msg
      int sig = 0;
      REQUIRE(WriteExact(fdSig[1], &sig, sizeof(sig)));
      // receive message from parent process
      checkSysCallErrors(sockObj.recvShareableHdl(&shHandle));
      hipMemGenericAllocationHandle_t imported_handle;
      // import the shareable handle
      HIP_CHECK(hipMemImportFromShareableHandle(
          &imported_handle, reinterpret_cast<void*>(static_cast<uintptr_t>(shHandle)),
          hipMemHandleTypePosixFileDescriptor));
      // Allocate virtual address range and map the imported allocation into it
      void* ptrA;
      HIP_CHECK(hipMemAddressReserve(&ptrA, params.size_mem, 0, 0, 0));
      HIP_CHECK(hipMemMap(ptrA, params.size_mem, 0, imported_handle, 0));

      // Grant CPU access to the imported mapping. This is the case under test: the imported
      // allocation has no CPU mapping of its own, so one has to be established here.
      hipMemAccessDesc accHost = {};
      accHost.location.type = hipMemLocationTypeHost;
      accHost.location.id = 0;
      accHost.flags = hipMemAccessFlagsProtReadWrite;
#if HT_AMD
      HIP_CHECK(hipMemSetAccess(ptrA, params.size_mem, &accHost, 1));

      // Read what the parent seeded and write back. A mapping made through the wrong GPU's DRM
      // context faults here rather than failing above.
      int* hostPtr = reinterpret_cast<int*>(ptrA);
      std::vector<int> expected(N);
      for (size_t idx = 0; idx < N; idx++) expected[idx] = idx;
      REQUIRE(true == std::equal(expected.begin(), expected.end(), hostPtr));

      for (size_t idx = 0; idx < N; idx++) hostPtr[idx] = static_cast<int>(idx) * 2;
#else
      // CUDA does not allow host access to a device located allocation.
      HIP_CHECK_ERROR(hipMemSetAccess(ptrA, params.size_mem, &accHost, 1), hipErrorNotSupported);
#endif

      // free resources
      HIP_CHECK(hipMemUnmap(ptrA, params.size_mem));
      HIP_CHECK(hipMemAddressFree(ptrA, params.size_mem));
      HIP_CHECK(hipMemRelease(imported_handle));
      CTX_DESTROY();
      checkSysCallErrors(sockObj.closeThisSock());
      REQUIRE(close(fd[0]) == 0);
      REQUIRE(close(fdSig[1]) == 0);
    });
  } else {  // parent
    REQUIRE(close(fd[0]) == 0);
    REQUIRE(close(fdSig[1]) == 0);
    ChildProcess child(pid, fd[1]);

    // Decide before touching the current device, so a skip leaves the process as it found it.
    int device = 0;
    std::string skip_reason;
    if (exporter == Exporter::kLastDevice) {
      int device_count = 0;
      HIP_CHECK(hipGetDeviceCount(&device_count));
      if (device_count < 2) {
        skip_reason = "Test needs at least 2 devices. Skipping Test..";
      } else {
        device = device_count - 1;
      }
    }
    if (skip_reason.empty() && !VmmSupported(device)) {
      skip_reason = HipTest::SkipReason::kVmmUnsupported;
    }

    if (!skip_reason.empty()) {
      REQUIRE(child.StandDown());
      int status = 0;
      REQUIRE(child.Join(&status));
      RequireChildExitedCleanly(status);
      REQUIRE(close(fd[1]) == 0);
      REQUIRE(close(fdSig[0]) == 0);
      HIP_SKIP_TEST(skip_reason);
      return;
    }

    CTX_CREATE();
    HIP_CHECK(hipSetDevice(device));

    // Set property
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = device;
    // Set Granularity of the VMM memory
    size_t granularity;
    HIP_CHECK(
        hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
    REQUIRE(granularity > 0);
    size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;
    hipMemGenericAllocationHandle_t handle;
    HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));

    // Map it locally and seed it so the child has something to read back.
    void* ptrA;
    HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));
    HIP_CHECK(hipMemMap(ptrA, size_mem, 0, handle, 0));
    hipMemAccessDesc accessDesc = {};
    accessDesc.location.type = hipMemLocationTypeDevice;
    accessDesc.location.id = device;
    accessDesc.flags = hipMemAccessFlagsProtReadWrite;
    HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));

    std::vector<int> A_h(N), B_h(N);
    for (size_t idx = 0; idx < N; idx++) A_h[idx] = idx;
    HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptrA), A_h.data(), buffer_size));

    hipShareableHdl shareable_handle;
    HIP_CHECK(hipMemExportToShareableHandle(&shareable_handle, handle,
                                            hipMemHandleTypePosixFileDescriptor, 0));
    // Create the socket for communication as Server
    ipcSocketCom sockObj(true);
    // Release the child: everything that can fail before sending the handle is done.
    ChildParams params = {};
    params.size_mem = size_mem;
    params.device = device;
    REQUIRE(child.Notify(params));
    // Wait for the child process to be ready to receive msg
    int sig = 0;
    REQUIRE(ReadExact(fdSig[0], &sig, sizeof(sig)));
    checkSysCallErrors(sockObj.sendShareableHdl(shareable_handle, child.pid()));
    // Wait for child process to exit.
    int status = 0;
    REQUIRE(child.Join(&status));
    RequireChildExitedCleanly(status);

#if HT_AMD
    // Check what the child wrote through its CPU mapping of the imported allocation.
    HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptrA), buffer_size));
    std::vector<int> expected(N);
    for (size_t idx = 0; idx < N; idx++) expected[idx] = static_cast<int>(idx) * 2;
    REQUIRE(true == std::equal(B_h.begin(), B_h.end(), expected.data()));
#endif

    // Free all resources
    HIP_CHECK(hipMemUnmap(ptrA, size_mem));
    HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
    HIP_CHECK(hipMemRelease(handle));
    checkSysCallErrors(sockObj.closeThisSock());
    CTX_DESTROY();
    REQUIRE(close(fd[1]) == 0);
    REQUIRE(close(fdSig[0]) == 0);

    // Do not leak a non-zero device into the next test.
    HIP_CHECK(hipSetDevice(0));
  }
}

}  // namespace

/**
 * Test Description
 * ------------------------
 *    - Multiprocess functionality test. The Parent Process creates a device backed Vmm
 * allocation, seeds it and exports it to the Child Process over a socket. The Child
 * Process imports the handle, maps it into its own reserved address range and grants
 * *CPU* access to the imported mapping. The Child then reads the Parent's data and writes
 * back through the CPU mapping, and the Parent verifies the result.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemSetAccessImportedHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMemSetAccess_MulProc_ImportedDeviceMem_HostAccess) {
  ImportedDeviceMemHostAccessTest(Exporter::kDeviceZero);
}

/**
 * Test Description
 * ------------------------
 *    - Same as Unit_hipMemSetAccess_MulProc_ImportedDeviceMem_HostAccess, but exporting from the
 * last device, so a wrong pick of the exporting GPU is not masked by everything agreeing. Skips
 * on a single device system.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemSetAccessImportedHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - Multiple devices
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMemSetAccess_MulProc_ImportedDeviceMem_HostAccess_ExporterNotDeviceZero) {
  ImportedDeviceMemHostAccessTest(Exporter::kLastDevice);
}

/**
 * Test Description
 * ------------------------
 *    - Multiprocess functionality test. Same import as above, but the Child Process grants
 * CPU access to the imported mapping *before* granting GPU access, seeds it through the CPU
 * mapping and then overwrites it from the device with hipMemsetD32. This checks that the host
 * and device mappings of an imported allocation coexist and that granting host access first
 * does not disturb the subsequent device mapping.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemSetAccessImportedHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMemSetAccess_MulProc_ImportedDeviceMem_HostThenDeviceAccess) {
  constexpr int N = DATA_SIZE;
  const size_t buffer_size = N * sizeof(int);
  constexpr int device = 0;

  int fd[2], fdSig[2];
  REQUIRE(pipe(fd) == 0);
  REQUIRE(pipe(fdSig) == 0);

  /* Drain stdio first: under ctest it is fully buffered, so a child that flushes would re-emit
   * whatever the parent had queued. */
  fflush(nullptr);

  /* fork() before the first HIP call; see ImportedDeviceMemHostAccessTest. */
  auto pid = fork();
  REQUIRE(pid >= 0);

  if (pid == 0) {  // child
    RunChildBody([&]() {
      REQUIRE(close(fd[1]) == 0);
      REQUIRE(close(fdSig[0]) == 0);

      // Wait for the parent to size the allocation and bring up the socket.
      ChildParams params = {};
      REQUIRE(ReadExact(fd[0], &params, sizeof(params)));
      if (params.size_mem == 0) {
        // Parent skipped or failed before exporting: there is nothing to import or receive.
        REQUIRE(close(fd[0]) == 0);
        REQUIRE(close(fdSig[1]) == 0);
        return;
      }
      REQUIRE(params.size_mem >= buffer_size);

      CTX_CREATE();
      HIP_CHECK(hipSetDevice(params.device));

      // Open Socket as client
      ipcSocketCom sockObj(false);
      hipShareableHdl shHandle;
      // Signal Parent process that Child is ready to receive msg
      int sig = 0;
      REQUIRE(WriteExact(fdSig[1], &sig, sizeof(sig)));
      // receive message from parent process
      checkSysCallErrors(sockObj.recvShareableHdl(&shHandle));
      hipMemGenericAllocationHandle_t imported_handle;
      // import the shareable handle
      HIP_CHECK(hipMemImportFromShareableHandle(
          &imported_handle, reinterpret_cast<void*>(static_cast<uintptr_t>(shHandle)),
          hipMemHandleTypePosixFileDescriptor));
      // Allocate virtual address range and map the imported allocation into it
      void* ptrA;
      HIP_CHECK(hipMemAddressReserve(&ptrA, params.size_mem, 0, 0, 0));
      HIP_CHECK(hipMemMap(ptrA, params.size_mem, 0, imported_handle, 0));

      // Grant CPU access first, then GPU access, on the same imported mapping.
      hipMemAccessDesc accHost = {};
      accHost.location.type = hipMemLocationTypeHost;
      accHost.location.id = 0;
      accHost.flags = hipMemAccessFlagsProtReadWrite;
#if HT_AMD
      HIP_CHECK(hipMemSetAccess(ptrA, params.size_mem, &accHost, 1));

      hipMemAccessDesc accDev = {};
      accDev.location.type = hipMemLocationTypeDevice;
      accDev.location.id = params.device;
      accDev.flags = hipMemAccessFlagsProtReadWrite;
      HIP_CHECK(hipMemSetAccess(ptrA, params.size_mem, &accDev, 1));

      // Seed through the CPU mapping, overwrite from the GPU, then read back through the CPU
      // mapping. Both mappings of the imported allocation have to be live for this to hold.
      int* hostPtr = reinterpret_cast<int*>(ptrA);
      for (size_t idx = 0; idx < N; idx++) hostPtr[idx] = static_cast<int>(idx);

      HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(ptrA), MEMSET_PATTERN, N));
      HIP_CHECK(hipDeviceSynchronize());

      std::vector<int> expected(N, static_cast<int>(MEMSET_PATTERN));
      REQUIRE(true == std::equal(expected.begin(), expected.end(), hostPtr));

      // And the device side observes the same memory through hipMemcpyDtoH.
      std::vector<int> readback(N, 0);
      HIP_CHECK(
          hipMemcpyDtoH(readback.data(), reinterpret_cast<hipDeviceptr_t>(ptrA), buffer_size));
      REQUIRE(true == std::equal(readback.begin(), readback.end(), expected.data()));
#else
      // CUDA does not allow host access to a device located allocation.
      HIP_CHECK_ERROR(hipMemSetAccess(ptrA, params.size_mem, &accHost, 1), hipErrorNotSupported);
#endif

      // free resources
      HIP_CHECK(hipMemUnmap(ptrA, params.size_mem));
      HIP_CHECK(hipMemAddressFree(ptrA, params.size_mem));
      HIP_CHECK(hipMemRelease(imported_handle));
      CTX_DESTROY();
      checkSysCallErrors(sockObj.closeThisSock());
      REQUIRE(close(fd[0]) == 0);
      REQUIRE(close(fdSig[1]) == 0);
    });
  } else {  // parent
    REQUIRE(close(fd[0]) == 0);
    REQUIRE(close(fdSig[1]) == 0);
    ChildProcess child(pid, fd[1]);

    if (!VmmSupported(device)) {
      REQUIRE(child.StandDown());
      int status = 0;
      REQUIRE(child.Join(&status));
      RequireChildExitedCleanly(status);
      REQUIRE(close(fd[1]) == 0);
      REQUIRE(close(fdSig[0]) == 0);
      HIP_SKIP_TEST(HipTest::SkipReason::kVmmUnsupported);
      return;
    }

    CTX_CREATE();
    HIP_CHECK(hipSetDevice(device));

    // Set property
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = device;
    // Set Granularity of the VMM memory
    size_t granularity;
    HIP_CHECK(
        hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
    REQUIRE(granularity > 0);
    size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;
    hipMemGenericAllocationHandle_t handle;
    HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));

    hipShareableHdl shareable_handle;
    HIP_CHECK(hipMemExportToShareableHandle(&shareable_handle, handle,
                                            hipMemHandleTypePosixFileDescriptor, 0));
    // Create the socket for communication as Server
    ipcSocketCom sockObj(true);
    // Release the child: everything that can fail before sending the handle is done.
    ChildParams params = {};
    params.size_mem = size_mem;
    params.device = device;
    REQUIRE(child.Notify(params));
    // Wait for the child process to be ready to receive msg
    int sig = 0;
    REQUIRE(ReadExact(fdSig[0], &sig, sizeof(sig)));
    checkSysCallErrors(sockObj.sendShareableHdl(shareable_handle, child.pid()));
    // Wait for child process to exit.
    int status = 0;
    REQUIRE(child.Join(&status));
    RequireChildExitedCleanly(status);

    // Free all resources
    HIP_CHECK(hipMemRelease(handle));
    checkSysCallErrors(sockObj.closeThisSock());
    CTX_DESTROY();
    REQUIRE(close(fd[1]) == 0);
    REQUIRE(close(fdSig[0]) == 0);
  }
}

/**
 * End doxygen group VirtualMemoryManagementTest.
 * @}
 */
