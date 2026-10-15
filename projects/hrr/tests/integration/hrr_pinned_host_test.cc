/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR pinned host snapshots
 * @{
 * @ingroup HRRTest
 * A kernel reads pinned host memory that the host refills with plain stores
 * between launches, which no HIP call records. Capture snapshots it before
 * each launch and replay writes it back, so the replayed D2H checks match.
 */

#include "hrr_test_common.hh"

#include <string>
#include <vector>
#ifndef _WIN32
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <thread>
#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif
#endif

namespace {
constexpr int kPinnedN     = 256;
constexpr int kPinnedIters = 4;
// The input changes on every second launch: two snapshot records, four checks.
int pinned_value(int it, int i) { return (it / 2) * 1000 + i; }
}  // namespace

__global__ void hrr_pinned_triple(int* out, const int* in, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * 3 + 1;
}

// The same kernel with the pinned pointer inside a by-value struct.
struct PinnedIn { const int* in; int n; };
__global__ void hrr_pinned_triple_struct(int* out, PinnedIn s) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < s.n) out[i] = s.in[i] * 3 + 1;
}

__global__ void hrr_pinned_ignore(uint64_t) {}

static void hrr_pinned_workload(int* in, bool by_struct = false) {
  int* dev_in = nullptr;
  HRR_HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&dev_in), in, 0));
  int* dout = nullptr;
  HRR_HIP_CHECK(hipMalloc(&dout, kPinnedN * sizeof(int)));
  std::vector<int> hout(kPinnedN);
  auto launch = [&](hipStream_t s) {
    if (by_struct)
      hipLaunchKernelGGL(hrr_pinned_triple_struct, dim3(1), dim3(kPinnedN), 0, s,
                         dout, PinnedIn{dev_in, kPinnedN});
    else
      hipLaunchKernelGGL(hrr_pinned_triple, dim3(1), dim3(kPinnedN), 0, s, dout, dev_in, kPinnedN);
    HRR_HIP_CHECK(hipGetLastError());
  };
  for (int it = 0; it < kPinnedIters; ++it) {
    for (int i = 0; i < kPinnedN; ++i) in[i] = pinned_value(it, i);  // uncaptured stores
    // Replay restores nothing in a graph capture. A launch on another stream
    // meanwhile is restored, and leaves the next launch nothing to record.
    if (it == 2) {
      hipStream_t s = nullptr, s2 = nullptr;
      hipGraph_t g  = nullptr;
      HRR_HIP_CHECK(hipStreamCreate(&s));
      HRR_HIP_CHECK(hipStreamCreate(&s2));
      HRR_HIP_CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
      launch(s);
      launch(s2);
      HRR_HIP_CHECK(hipStreamEndCapture(s, &g));
      HRR_HIP_CHECK(hipGraphDestroy(g));
      HRR_HIP_CHECK(hipStreamDestroy(s));
      HRR_HIP_CHECK(hipStreamDestroy(s2));
    }
    launch(nullptr);
    HRR_HIP_CHECK(hipMemcpy(hout.data(), dout, kPinnedN * sizeof(int), hipMemcpyDeviceToHost));
    for (int i = 0; i < kPinnedN; ++i) REQUIRE(hout[i] == pinned_value(it, i) * 3 + 1);
  }
  HRR_HIP_CHECK(hipFree(dout));
}

TEST_CASE("Unit_HRR_PinnedHostMalloc_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  int* in = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&in), kPinnedN * sizeof(int), 0));
  hrr_pinned_workload(in);
  HRR_HIP_CHECK(hipHostFree(in));
}

TEST_CASE("Unit_HRR_PinnedHostStruct_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  int* in = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&in), kPinnedN * sizeof(int), 0));
  hrr_pinned_workload(in, true);
  HRR_HIP_CHECK(hipHostFree(in));
}

// A freed allocation's address passed on as a plain number: capture must not
// snapshot memory that is gone.
TEST_CASE("Unit_HRR_PinnedHostFreed_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  void* p = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(&p, kPinnedN * sizeof(int), 0));
  HRR_HIP_CHECK(hipHostFree(p));
  hipLaunchKernelGGL(hrr_pinned_ignore, dim3(1), dim3(1), 0, nullptr,
                     reinterpret_cast<uint64_t>(p));
  HRR_HIP_CHECK(hipGetLastError());
  HRR_HIP_CHECK(hipDeviceSynchronize());
}

TEST_CASE("Unit_HRR_PinnedHostRegister_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  std::vector<int> in(kPinnedN);
  HRR_HIP_CHECK(hipHostRegister(in.data(), kPinnedN * sizeof(int), hipHostRegisterDefault));
  hrr_pinned_workload(in.data());
  HRR_HIP_CHECK(hipHostUnregister(in.data()));
}

#ifndef _WIN32
// The struct case from pinned memory whose low 32 bits are zero, an address
// replay's device-pointer shape check rejects.
TEST_CASE("Unit_HRR_PinnedHostStructLowBits_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  void* at = reinterpret_cast<void*>(uintptr_t{0x7e0000000000});
  void* p  = mmap(at, 1 << 16, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
  if (p != at) {
    const int err = errno;
    // A kernel older than 4.17 takes the address as a hint only.
    if (p != MAP_FAILED) munmap(p, 1 << 16);
    if (p != MAP_FAILED || err == EEXIST) HRR_SKIP("0x7e0000000000 is already mapped");
    FAIL("mmap: " << std::strerror(err));
  }
  HRR_HIP_CHECK(hipHostRegister(p, kPinnedN * sizeof(int), hipHostRegisterDefault));
  hrr_pinned_workload(static_cast<int*>(p), true);
  HRR_HIP_CHECK(hipHostUnregister(p));
  munmap(p, 1 << 16);
}

// Forks while another thread snapshots a 32 MiB pinned allocation, so most forks
// land while it holds the pinned tracking lock. Each child frees an address
// capture does not track, which takes that lock and fails without device work.
// A child that inherited the lock held never exits.
TEST_CASE("Unit_HRR_PinnedHostFork_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr size_t kBig = 32u << 20;
  void* p = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(&p, kBig, 0));
  std::memset(p, 1, kBig);
  // Nothing may REQUIRE before the join.
  std::atomic<bool> stop{false};
  std::thread launcher([&] {
    while (!stop.load()) {
      hipLaunchKernelGGL(hrr_pinned_ignore, dim3(1), dim3(1), 0, nullptr,
                         reinterpret_cast<uint64_t>(p));
      (void)hipStreamSynchronize(nullptr);
    }
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  int hung = 0, failed = 0;
  for (int i = 0; i < 20 && !hung; ++i) {
    const pid_t pid = fork();
    if (pid == 0) {
      int x = 0;
      _exit(hipFree(&x) == hipSuccess);
    }
    int status = 0;
    pid_t got = 0;
    for (int ms = 0; pid > 0 && ms < 10000 && (got = waitpid(pid, &status, WNOHANG)) == 0; ++ms)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (pid > 0 && got == 0) {
      kill(pid, SIGKILL);
      waitpid(pid, &status, 0);
      ++hung;
    } else if (got != pid || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      ++failed;
    }
  }
  stop = true;
  launcher.join();
  HRR_HIP_CHECK(hipHostFree(p));
  INFO("hung " << hung << ", failed " << failed);
  REQUIRE(hung == 0);
  REQUIRE(failed == 0);
}
#endif

#if defined(HRR_PLAYBACK_EXE) && defined(HRR_TEST_EXE)
// Replay matches the recording only if it wrote each launch's input back; its
// own buffer never sees the host's stores. With HIP_HRR_HOST_SNAPSHOTS=0 there
// is nothing to write back and the checks fail.
HRR_TEST_CASE(Unit_HRR_PinnedHostRoundtrip) {
  for (const char* direct : {"Unit_HRR_PinnedHostMalloc_Direct",
                             "Unit_HRR_PinnedHostRegister_Direct",
                             "Unit_HRR_PinnedHostStruct_Direct",
#ifndef _WIN32
                             // Last, since a skip ends the case.
                             "Unit_HRR_PinnedHostStructLowBits_Direct",
#endif
                            }) {
    for (const char* snapshots : {"1", "0"}) {
      // Without snapshots the struct would carry a host address replay never mapped.
      if (std::string(direct).find("Struct") != std::string::npos && *snapshots == '0')
        continue;
      INFO(direct << " HIP_HRR_HOST_SNAPSHOTS=" << snapshots);
      ScopedDir cap{fs::temp_directory_path() / "hrr_pinned_host"};
      hrr_capture_direct(direct, cap.path, 5, {{"HIP_HRR_HOST_SNAPSHOTS", snapshots}});
      auto [ret, out] = hrr_playback_env(cap.path, {{"HIP_HRR_D2H_EXACT", "1"}});
      INFO("Playback exit " << ret << ", stdout:\n" << out);
      int pass = 0, fail = 0;
      REQUIRE(hrr_parse_d2h_summary(out, pass, fail));
      if (std::string(snapshots) == "1") {
        CHECK(ret == 0);
        CHECK(pass == kPinnedIters);
        CHECK(fail == 0);
        CHECK(out.find("Pinned host snapshots: 2 restored, 0 refused, 1 skipped") !=
              std::string::npos);
      } else {
        CHECK(ret != 0);
        CHECK(fail > 0);
      }
    }
  }
}

// Capture stops tracking an allocation when it is freed: had it snapshotted the
// stale address, replay would refuse the record for an allocation it freed.
HRR_TEST_CASE(Unit_HRR_PinnedHostFreedRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_pinned_host_freed"};
  hrr_capture_direct("Unit_HRR_PinnedHostFreed_Direct", cap.path, 3);
  auto [ret, out] = hrr_playback_env(cap.path, {});
  INFO("Playback exit " << ret << ", stdout:\n" << out);
  CHECK(ret == 0);
  CHECK(out.find("Pinned host snapshots") == std::string::npos);
}

#ifndef _WIN32
// The pinned tracking lock is held across fork(), so no child inherits it locked.
HRR_TEST_CASE(Unit_HRR_PinnedHostFork) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_pinned_host_fork"};
  hrr::test::SpawnProc proc(hrr_self_exe());
  proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
  set_proc_search_path(proc);
  const int ret = proc.runWithTimeout("\"Unit_HRR_PinnedHostFork_Direct\"", 300);
  INFO("Capture exit code: " << ret);
  REQUIRE(ret == 0);
}
#endif
#endif  // HRR_PLAYBACK_EXE && HRR_TEST_EXE

/**
 * @}
 */
