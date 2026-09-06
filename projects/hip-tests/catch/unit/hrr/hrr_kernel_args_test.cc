/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR kernel-argument translation
 * @{
 * @ingroup HRRTest
 * Tests for what replay does with a pointer-typed kernel argument the
 * allocation map cannot explain.
 *
 * Two things reach a kernel through the same kind of slot and mean opposite
 * things. A sentinel is a small literal a kernel branches on — aiter's MLA
 * decode kernel takes a 1 in one of its pointer slots — and substituting null
 * for it changes what the kernel does, silently and only on replay. A pointer
 * HRR genuinely lost is address-shaped, and handing it to the GPU unchanged is
 * worse than null: the driver tends to reproduce VA layout across runs, so the
 * recorded address is likely to land in an unrelated live buffer and be
 * scribbled over instead of faulting.
 *
 * Replay tells them apart by shape, and this asserts both halves of that.
 */

#include <hip_test_common.hh>
#include <hip_test_process.hh>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

namespace fs = std::filesystem;

#include "hrr_test_common.h"

#if defined(HRR_PLAYBACK_EXE) && defined(HRR_TEST_EXE)

namespace {
// Small enough to be certainly not an address, and the value aiter actually
// passes.
constexpr uint64_t kSentinel = 1;
constexpr size_t   kProbeBytes = 64 * 1024;
// Far enough past the probe allocation that no allocation of the replay's own
// can cover it, while staying inside device VA space so the value is
// address-shaped — which is the whole distinction under test.
constexpr size_t   kFarOff = 1ull << 32;

#define HRR_KARGS_MARKER "HRR_KARGS_LOST"
}  // namespace

__global__ void hrr_kargs_probe(int* out, int* sentinel, int* lost) {
  // Neither `sentinel` nor `lost` is dereferenced. At capture one is a literal
  // and the other points at nothing mapped; at replay the second arrives as
  // null. The kernel only has to carry them in its kernarg segment for HRR to
  // record and translate them.
  (void)sentinel;
  (void)lost;
  if (threadIdx.x == 0) out[0] = 1;
}

// ===========================================================================
// The captured workload: one launch carrying both cases in pointer slots.
// ===========================================================================
TEST_CASE("Unit_HRR_KernelArgs_Direct", "[.][hrr-direct]") {
  HIP_CHECK(hipSetDevice(0));

  int* out = nullptr;
  HIP_CHECK(hipMalloc(&out, sizeof(int)));
  HIP_CHECK(hipMemset(out, 0, sizeof(int)));

  char* probe = nullptr;
  HIP_CHECK(hipMalloc(&probe, kProbeBytes));

  int* lost     = reinterpret_cast<int*>(probe + kFarOff);
  int* sentinel = reinterpret_cast<int*>(kSentinel);

  // The address has to reach the parent: nothing in the archive records an
  // allocation for it, which is exactly what makes it the case under test.
  printf(HRR_KARGS_MARKER " 0x%llx\n",
         static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(lost)));
  fflush(stdout);

  hipLaunchKernelGGL(hrr_kargs_probe, dim3(1), dim3(64), 0, nullptr, out,
                     sentinel, lost);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  int host = 0;
  HIP_CHECK(hipMemcpy(&host, out, sizeof(int), hipMemcpyDeviceToHost));
  REQUIRE(host == 1);

  HIP_CHECK(hipFree(probe));
  HIP_CHECK(hipFree(out));
}

namespace {
// Capture the workload, keeping the child's stdout so the parent learns the
// address it passed.
inline uint64_t hrr_capture_direct_kargs(const fs::path& cap_path) {
  std::string out;
  { hip::SpawnProc proc(HRR_TEST_EXE, /*capture_stdout=*/true);
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

// Recover the live value the ptr-dump paired with a recorded one.
//
// The dump prints the live side with %p, whose rendering is not portable:
// glibc emits "0x1" and "(nil)", the MSVC CRT emits zero-padded hex with no
// prefix and no null spelling ("0000000000000001", "0000000000000000"). Only
// the recorded side is formatted by HRR itself, so match on that and parse
// whatever the platform produced instead of comparing formatted text.
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

// ---------------------------------------------------------------------------
// Both halves, from one archive and one replay.
//
// Before replay distinguished them, every unresolvable pointer took the same
// branch, so one of these two assertions failed whichever branch that was: the
// sentinel arrived as null, or the lost pointer arrived as the recorded
// address.
// ---------------------------------------------------------------------------
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

  // A value no device VA could be keeps its meaning.
  uint64_t sentinel_live = 0;
  REQUIRE(hrr_live_arg(out, kSentinel, &sentinel_live));
  CHECK(sentinel_live == kSentinel);
  CHECK(out.find("not an address, passing it through unchanged") !=
        std::string::npos);

  // An address-shaped value that resolves nowhere does not.
  uint64_t lost_live = 1;
  REQUIRE(hrr_live_arg(out, lost, &lost_live));
  CHECK(lost_live == 0);
  CHECK(out.find("passing null so it faults at first use") !=
        std::string::npos);

  // Both were counted, and neither was quietly treated as translated.
  CHECK(out.find("Untranslated   : 0 ") == std::string::npos);
}

#endif  // HRR_PLAYBACK_EXE && HRR_TEST_EXE

/**
 * @}
 */
