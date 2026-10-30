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

// by_module launches through hipModuleLaunchKernel with an extra[] buffer, the
// path where capture finds the arguments by the kernel's offsets.
static void hrr_pinned_workload(int* in, bool by_struct = false, bool by_module = false) {
  int* dev_in = nullptr;
  HRR_HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&dev_in), in, 0));
  int* dout = nullptr;
  HRR_HIP_CHECK(hipMalloc(&dout, kPinnedN * sizeof(int)));
  std::vector<int> hout(kPinnedN);
  hipFunction_t f = nullptr;
  if (by_module)
    HRR_HIP_CHECK(hipGetFuncBySymbol(&f, reinterpret_cast<const void*>(hrr_pinned_triple)));
  auto launch = [&](hipStream_t s) {
    if (by_module) {
      struct { int* out; const int* in; int n; } a{dout, dev_in, kPinnedN};
      size_t sz = sizeof(a);
      void* extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &a, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz,
                       HIP_LAUNCH_PARAM_END};
      HRR_HIP_CHECK(hipModuleLaunchKernel(f, 1, 1, 1, kPinnedN, 1, 1, 0, s, nullptr, extra));
    } else if (by_struct)
      hipLaunchKernelGGL(hrr_pinned_triple_struct, dim3(1), dim3(kPinnedN), 0, s,
                         dout, PinnedIn{dev_in, kPinnedN});
    else
      hipLaunchKernelGGL(hrr_pinned_triple, dim3(1), dim3(kPinnedN), 0, s, dout, dev_in, kPinnedN);
    HRR_HIP_CHECK(hipGetLastError());
  };
  for (int it = 0; it < kPinnedIters; ++it) {
    for (int i = 0; i < kPinnedN; ++i) in[i] = pinned_value(it, i);  // uncaptured stores
    // Replay restores nothing in a graph capture. A launch on another stream
    // meanwhile is restored on that stream, so the next launch records again.
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

__global__ void hrr_pinned_wait(int* flag) {
  while (atomicAdd(flag, 0) == 0) __builtin_amdgcn_s_sleep(8);
}
__global__ void hrr_pinned_signal(int* flag) { atomicExch(flag, 1); }

// The pinned-input kernel on stream a queues behind a kernel waiting for a flag
// that b sets later, after a launch on b that reads the same unchanged input.
TEST_CASE("Unit_HRR_PinnedHostStreams_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  int *in = nullptr, *dev_in = nullptr, *flag = nullptr, *dout = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&in), kPinnedN * sizeof(int), 0));
  HRR_HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&dev_in), in, 0));
  HRR_HIP_CHECK(hipMalloc(&flag, sizeof(int)));
  HRR_HIP_CHECK(hipMemset(flag, 0, sizeof(int)));
  HRR_HIP_CHECK(hipMalloc(&dout, 2 * kPinnedN * sizeof(int)));
  hipStream_t a = nullptr, b = nullptr;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&a, hipStreamNonBlocking));
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&b, hipStreamNonBlocking));
  std::vector<int> hout(2 * kPinnedN);
  for (int it = 0; it < 2; ++it) {
    for (int i = 0; i < kPinnedN; ++i) in[i] = pinned_value(2 * it, i);
    if (it == 1) {
      hipLaunchKernelGGL(hrr_pinned_wait, dim3(1), dim3(1), 0, a, flag);
      hipLaunchKernelGGL(hrr_pinned_triple, dim3(1), dim3(kPinnedN), 0, a,
                         dout + kPinnedN, dev_in, kPinnedN);
    }
    hipLaunchKernelGGL(hrr_pinned_triple, dim3(1), dim3(kPinnedN), 0, b, dout, dev_in, kPinnedN);
    if (it == 1) hipLaunchKernelGGL(hrr_pinned_signal, dim3(1), dim3(1), 0, b, flag);
    HRR_HIP_CHECK(hipGetLastError());
    HRR_HIP_CHECK(hipStreamSynchronize(a));
    HRR_HIP_CHECK(hipStreamSynchronize(b));
    HRR_HIP_CHECK(hipMemcpy(hout.data(), dout, (it + 1) * kPinnedN * sizeof(int),
                            hipMemcpyDeviceToHost));
    for (int i = 0; i < (it + 1) * kPinnedN; ++i)
      REQUIRE(hout[i] == pinned_value(2 * it, i % kPinnedN) * 3 + 1);
  }
  HRR_HIP_CHECK(hipStreamDestroy(a));
  HRR_HIP_CHECK(hipStreamDestroy(b));
  HRR_HIP_CHECK(hipFree(dout));
  HRR_HIP_CHECK(hipFree(flag));
  HRR_HIP_CHECK(hipHostFree(in));
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

TEST_CASE("Unit_HRR_PinnedHostModule_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  int* in = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&in), kPinnedN * sizeof(int), 0));
  hrr_pinned_workload(in, false, true);
  HRR_HIP_CHECK(hipHostFree(in));
}

// The deprecated hipMallocHost and hipFreeHost, which replay still replays.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
TEST_CASE("Unit_HRR_PinnedMallocHost_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  int* in = nullptr;
  HRR_HIP_CHECK(hipMallocHost(reinterpret_cast<void**>(&in), kPinnedN * sizeof(int)));
  hrr_pinned_workload(in);
  HRR_HIP_CHECK(hipFreeHost(in));
}
#pragma clang diagnostic pop

// A 2 MiB allocation: over HIP_HRR_HOST_SNAPSHOT_MAX_MB=1, within 2.
TEST_CASE("Unit_HRR_PinnedHostLarge_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  int* in = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&in), size_t{2} << 20, 0));
  hrr_pinned_workload(in);
  HRR_HIP_CHECK(hipHostFree(in));
}

// A released allocation's address passed on as a plain number, once for each
// way to release it: capture must not snapshot memory that is no longer pinned.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
TEST_CASE("Unit_HRR_PinnedHostFreed_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  std::vector<int> reg(kPinnedN);
  for (int how = 0; how < 4; ++how) {
    void* p = reg.data();
    if (how == 3)
      HRR_HIP_CHECK(hipHostRegister(p, kPinnedN * sizeof(int), hipHostRegisterDefault));
    else
      HRR_HIP_CHECK(hipHostMalloc(&p, kPinnedN * sizeof(int), 0));
    if (how == 0) HRR_HIP_CHECK(hipHostFree(p));
    if (how == 1) HRR_HIP_CHECK(hipFree(p));
    if (how == 2) HRR_HIP_CHECK(hipFreeHost(p));
    if (how == 3) HRR_HIP_CHECK(hipHostUnregister(p));
    hipLaunchKernelGGL(hrr_pinned_ignore, dim3(1), dim3(1), 0, nullptr,
                       reinterpret_cast<uint64_t>(p));
    HRR_HIP_CHECK(hipGetLastError());
  }
  HRR_HIP_CHECK(hipDeviceSynchronize());
}
#pragma clang diagnostic pop

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
                             "Unit_HRR_PinnedMallocHost_Direct",
                             "Unit_HRR_PinnedHostModule_Direct",
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
      auto [ret, out] = hrr_playback_env(cap.path, {});
      INFO("Playback exit " << ret << ", stdout:\n" << out);
      int pass = 0, fail = 0;
      REQUIRE(hrr_parse_d2h_summary(out, pass, fail));
      if (std::string(snapshots) == "1") {
        CHECK(ret == 0);
        CHECK(pass == kPinnedIters);
        CHECK(fail == 0);
        CHECK(out.find("Pinned host snapshots: 3 restored, 0 refused, 1 skipped") !=
              std::string::npos);
      } else {
        CHECK(ret != 0);
        CHECK(fail > 0);
      }
    }
  }
}

// Replay queues each restore on its launch's stream rather than waiting for
// the stream, here held by a kernel that waits for one replay submits later.
// The launch on b records the input too, since replay restores it on a.
HRR_TEST_CASE(Unit_HRR_PinnedHostStreamsRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_pinned_host_streams"};
  hrr_capture_direct("Unit_HRR_PinnedHostStreams_Direct", cap.path, 5);
  auto [ret, out] = hrr_playback_watchdog(cap.path, 120);
  INFO("Playback exit " << ret << ", output:\n" << out);
  REQUIRE(ret != kHrrWatchdogKilled);
  CHECK(ret == 0);
  // Exact, since a HIP_HRR_D2H_TOL float tolerance would take these ints for
  // tiny floats.
  CHECK(out.find("2 pass (2 exact, 0 within tol), 0 fail") != std::string::npos);
  CHECK(out.find("Pinned host snapshots: 3 restored, 0 refused, 0 skipped") !=
        std::string::npos);
}

// An allocation larger than HIP_HRR_HOST_SNAPSHOT_MAX_MB is not snapshotted and
// is listed as unreplayable, so replay cannot match. Within the cap it is.
HRR_TEST_CASE(Unit_HRR_PinnedHostSnapshotCap) {
  for (const char* max_mb : {"2", "1"}) {
    const bool over = *max_mb == '1';
    INFO("HIP_HRR_HOST_SNAPSHOT_MAX_MB=" << max_mb);
    ScopedDir cap{fs::temp_directory_path() / "hrr_pinned_host_cap"};
    hrr_capture_direct("Unit_HRR_PinnedHostLarge_Direct", cap.path, 5,
                       {{"HIP_HRR_HOST_SNAPSHOT_MAX_MB", max_mb}});
    const std::string manifest =
        read_text_file(hrr_single_process_archive(cap.path) / "manifest.json");
    INFO("manifest.json:\n" << manifest);
    CHECK((manifest.find("larger than HIP_HRR_HOST_SNAPSHOT_MAX_MB") != std::string::npos) ==
          over);
    auto [ret, out] = hrr_playback_env(cap.path, {});
    INFO("Playback exit " << ret << ", stdout:\n" << out);
    int pass = 0, fail = 0;
    REQUIRE(hrr_parse_d2h_summary(out, pass, fail));
    if (over) {
      CHECK(ret != 0);
      CHECK(fail > 0);
      CHECK(out.find("Pinned host snapshots") == std::string::npos);
    } else {
      CHECK(ret == 0);
      CHECK(pass == kPinnedIters);
      CHECK(fail == 0);
      CHECK(out.find("Pinned host snapshots: 3 restored, 0 refused, 1 skipped") !=
            std::string::npos);
    }
  }
}

// Metadata-only capture takes no pinned snapshot: over the cap no allocation is
// noted as unreplayable, and within it no launch carries a snapshot record.
HRR_TEST_CASE(Unit_HRR_PinnedHostMetadataOnlyNoSnapshot) {
  for (const char* max_mb : {"1", "64"}) {
    INFO("HIP_HRR_HOST_SNAPSHOT_MAX_MB=" << max_mb);
    ScopedDir cap{fs::temp_directory_path() / "hrr_pinned_host_metadata_only"};
    hrr_capture_direct("Unit_HRR_PinnedHostLarge_Direct", cap.path, 5,
                       {{"HIP_HRR_HOST_SNAPSHOT_MAX_MB", max_mb},
                        {"HIP_HRR_CAPTURE_CONTENT", "0"}});
    const std::string manifest =
        read_text_file(hrr_single_process_archive(cap.path) / "manifest.json");
    INFO("manifest.json:\n" << manifest);
    CHECK(manifest.find("larger than HIP_HRR_HOST_SNAPSHOT_MAX_MB") == std::string::npos);
    hrr::Archive arc;
    REQUIRE(hrr::load_archive(cap.path.string(), arc));
    REQUIRE((arc.flags & HRR_FILE_FLAG_METADATA_ONLY) != 0);
    size_t launches = 0, snapshots = 0;
    for (const auto& ev : arc.events) {
      if (!ev.kernel_launch) continue;
      ++launches;
      snapshots += ev.kernel_launch->snapshots.size();
    }
    CHECK(launches > 0);
    CHECK(snapshots == 0);
  }
}

// A snapshot blob over --max-file-bytes ends the replay before the kernel that
// reads it is launched on input replay never restored.
HRR_TEST_CASE(Unit_HRR_PinnedHostBlobRefused) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_pinned_host_refused"};
  hrr_capture_direct("Unit_HRR_PinnedHostLarge_Direct", cap.path, 5);
  auto [ret, out] = hrr_playback_merged(cap.path, "--trace-kernels --max-file-bytes 2097151");
  INFO("Playback exit " << ret << ", output:\n" << out);
  CHECK(ret != 0);
  CHECK(out.find("refusing to read") != std::string::npos);
  CHECK(out.find("[HRR launch]") == std::string::npos);
}

// Capture stops tracking an allocation on hipHostFree, hipFree, hipFreeHost and
// hipHostUnregister: had it snapshotted the stale address, replay would refuse
// the record for an allocation it released.
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
  proc.setEnv("HIP_HRR_CAPTURE_CONTENT", "1");
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
