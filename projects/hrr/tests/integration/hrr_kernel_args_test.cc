/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR kernel-argument translation
 * @{
 * @ingroup HRRTest
 * Replay keeps a small literal in a pointer slot (a sentinel; aiter's MLA decode passes 1) and nulls an address-shaped pointer no allocation explains.
 * Passing a lost address through could scribble over an unrelated live buffer, since the driver tends to reproduce VA layout across runs.
 */

#include "hrr_test_common.hh"
#include "hrr_test_process.hh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

#if defined(HRR_PLAYBACK_EXE) && defined(HRR_TEST_EXE)

namespace {
constexpr uint64_t kSentinel = 1;
constexpr size_t   kProbeBytes = 64 * 1024;
// Past the probe allocation so no replay allocation covers it, yet inside device VA space
// so the value stays address-shaped.
constexpr size_t   kFarOff = 1ull << 32;

#define HRR_KARGS_MARKER "HRR_KARGS_LOST"
}  // namespace

__global__ void hrr_kargs_probe(int* out, int* sentinel, int* lost) {
// Neither pointer is dereferenced; they only need to sit in the kernarg segment for HRR to record and translate them.
  (void)sentinel;
  (void)lost;
  if (threadIdx.x == 0) out[0] = 1;
}

TEST_CASE("Unit_HRR_KernelArgs_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  int* out = nullptr;
  HRR_HIP_CHECK(hipMalloc(&out, sizeof(int)));
  HRR_HIP_CHECK(hipMemset(out, 0, sizeof(int)));

  char* probe = nullptr;
  HRR_HIP_CHECK(hipMalloc(&probe, kProbeBytes));

  int* lost     = reinterpret_cast<int*>(probe + kFarOff);
  int* sentinel = reinterpret_cast<int*>(kSentinel);

// The parent needs this address; the archive holds no allocation for it.
  printf(HRR_KARGS_MARKER " 0x%llx\n",
         static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(lost)));
  fflush(stdout);

  hipLaunchKernelGGL(hrr_kargs_probe, dim3(1), dim3(64), 0, nullptr, out,
                     sentinel, lost);
  HRR_HIP_CHECK(hipGetLastError());
  HRR_HIP_CHECK(hipDeviceSynchronize());

  int host = 0;
  HRR_HIP_CHECK(hipMemcpy(&host, out, sizeof(int), hipMemcpyDeviceToHost));
  REQUIRE(host == 1);

  HRR_HIP_CHECK(hipFree(probe));
  HRR_HIP_CHECK(hipFree(out));
}

namespace {
inline uint64_t hrr_capture_direct_kargs(const fs::path& cap_path) {
  std::string out;
  { hrr::test::SpawnProc proc(hrr_self_exe(), /*capture_stdout=*/true);
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap_path.string());
    set_proc_search_path(proc);
    int ret = proc.run("\"Unit_HRR_KernelArgs_Direct\"");
    out = proc.getOutput();
    INFO("Capture exit: " << ret << "\n" << out);
    REQUIRE(ret == 0); }

  const size_t at = out.find(HRR_KARGS_MARKER);
  if (at == std::string::npos) return 0;
  unsigned long long lost = 0;
  if (sscanf(out.c_str() + at, HRR_KARGS_MARKER " 0x%llx", &lost) != 1)
    return 0;
  return lost;
}

// The dump's live side uses %p, whose rendering differs by platform (glibc "0x1" and "(nil)", MSVC bare padded hex).
// Match on the HRR-formatted recorded side and parse the live side instead.
inline bool hrr_live_arg(const std::string& out, uint64_t recorded,
                         uint64_t* live) {
  char needle[64];
  snprintf(needle, sizeof(needle), "recorded=0x%llx -> live=",
           static_cast<unsigned long long>(recorded));
  const size_t at = out.find(needle);
  if (at == std::string::npos) return false;

  const char* p = out.c_str() + at + strlen(needle);
  if (strncmp(p, "(nil)", 5) == 0) { *live = 0; return true; }
  // Base 16 accepts both the bare digits MSVC prints and glibc's 0x prefix.
  *live = strtoull(p, nullptr, 16);
  return true;
}
}  // namespace

TEST_CASE("Unit_HRR_KernelArgs_SentinelKeptLostPointerNulled", "[.][hrr]") {
  ScopedDir cap(fs::temp_directory_path() / "hrr_kernel_args.hrr");
  const uint64_t lost = hrr_capture_direct_kargs(cap.path);
  REQUIRE(lost != 0);

  const fs::path archive = hrr_single_process_archive(cap.path);

  // Ordinal 1 is the workload's only launch.
  auto [rc, out] = hrr_playback_merged(
      archive, "--warn-untranslated-args",
      {{"HIP_HRR_REPLAY_DUMP_PTRS_ORDINAL", "1"}});
  INFO("Replay:\n" << out);
  CHECK(rc == 0);

  uint64_t sentinel_live = 0;
  REQUIRE(hrr_live_arg(out, kSentinel, &sentinel_live));
  CHECK(sentinel_live == kSentinel);
  CHECK(out.find("not an address, passing it through unchanged") !=
        std::string::npos);

  uint64_t lost_live = 1;
  REQUIRE(hrr_live_arg(out, lost, &lost_live));
  CHECK(lost_live == 0);
  CHECK(out.find("passing null so it faults at first use") !=
        std::string::npos);

  CHECK(out.find("Untranslated   : 0 ") == std::string::npos);
}

// HIP_HRR_DEBUG_ARGS dumps kernel argument bytes to the log, a second copy of the archive.
// Only a Debug runtime honours it and says so on stderr; a release runtime says it ignored it.
// When a Debug build reads it as on, the log level is 0 and the runtime's own raise to LOG_INFO shows the dumps.
// Otherwise AMD_LOG_LEVEL=3 shows them if anything else enabled them. AMD_LOG_MASK=0 keeps per-API traces out.
// Env and log file are always set or unset, so the parent's environment cannot decide the outcome.
HRR_TEST_CASE(Unit_HRR_KernelArgs_DebugArgsOnlyInDebugRuntime) {
  struct Run { bool dumped, noticed, ignored; };
  // A null debug_args leaves the variable unset; on is whether a Debug build
  // reads the value as on.
  auto capture = [](const char* debug_args, bool on) {
    ScopedDir cap(fs::temp_directory_path() / "hrr_kernel_args_debug.hrr");
    hrr::test::SpawnProc proc(HRR_TEST_EXE, /*capture_stdout=*/true,
                              /*capture_stderr=*/true);
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    if (debug_args)
      proc.setEnv("HIP_HRR_DEBUG_ARGS", debug_args);
    else
      proc.unsetEnv("HIP_HRR_DEBUG_ARGS");
    proc.setEnv("AMD_LOG_LEVEL", on ? "0" : "3");
    proc.setEnv("AMD_LOG_MASK", "0");
    proc.unsetEnv("AMD_LOG_LEVEL_FILE");
    proc.unsetEnv("AMD_LOG_ASYNC");
    set_proc_search_path(proc);
    const int ret = proc.run("\"Unit_HRR_KernelArgs_Direct\"");
    const std::string out = proc.getOutput();
    INFO("Capture output:\n" << out);
    REQUIRE(ret == 0);
    // An absent dump means something only if capture ran.
    hrr_single_process_archive(cap.path);
    return Run{out.find("[HRR args] ") != std::string::npos,
               out.find("[HRR capture] HIP_HRR_DEBUG_ARGS is set: kernel "
                        "argument bytes") != std::string::npos,
               out.find("[HRR capture] HIP_HRR_DEBUG_ARGS ignored") !=
                   std::string::npos};
  };

  const Run unset = capture(nullptr, false);
  CHECK_FALSE(unset.dumped);
  CHECK_FALSE(unset.noticed);
  CHECK_FALSE(unset.ignored);

  // A Debug build reads "1" as on and "0" as off; a release build reports both.
  for (const bool on : {true, false}) {
    const char* value = on ? "1" : "0";
    INFO("HIP_HRR_DEBUG_ARGS=" << value);
    const Run set = capture(value, on);
#if defined(HRR_TEST_CLR_DEBUG_FLAGS)
    CHECK(set.dumped == on);
    CHECK(set.noticed == on);
    CHECK_FALSE(set.ignored);
#else
    CHECK_FALSE(set.dumped);
    CHECK_FALSE(set.noticed);
    CHECK(set.ignored);
#endif
  }
}

#endif  // HRR_PLAYBACK_EXE && HRR_TEST_EXE

/**
 * @}
 */
