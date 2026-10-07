/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR capture-address placement
 * @{
 * @ingroup HRRTest
 * Tests for replaying device allocations at their capture-time address.
 *
 * Replay translates every pointer it sees in a HIP call. It cannot see one the
 * program stored in device memory: that pointer reaches the GPU inside an H2D
 * payload, byte for byte, and names memory in the capturing process. vLLM's
 * block table is the case that made this matter (ROCM-31827). Placement makes
 * the stored value true again by putting each allocation where it was.
 *
 * The workload is the minimal form of that case: a buffer, a pointer cell
 * that holds the buffer's address and is filled with hipMemcpy, and a kernel
 * that reads the buffer through the cell. The kernel is also handed the
 * buffer's address as an argument, which replay does translate, and reads
 * through the cell only when the two agree. A stale cell then gives a wrong
 * answer in a D2H check instead of a GPU page fault, which a test can assert
 * on without taking the runner down.
 */

#include "hrr_test_common.hh"
#include "hrr_test_process.hh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#if defined(HRR_PLAYBACK_EXE) && defined(HRR_TEST_EXE)

namespace {
constexpr int    kElems = 1024;
constexpr size_t kBytes = kElems * sizeof(int);
constexpr int    kStale = -1;

#define HRR_PLACE_MARKER "HRR_PLACE_ADDRS"
}  // namespace

// Reads src through the cell, and only when the cell holds what the argument
// says the buffer is.
__global__ void hrr_place_deref(int* out, int* const* cell, const int* expect,
                                int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int* p = *cell;
  out[i] = (p == expect) ? p[i] * 2 : kStale;
}

namespace {
// One buffer, one cell holding its address, and the launch that reads one
// through the other. Returns the buffer.
int* hrr_place_round(int* out, int seed, int** cell_out) {
  int* buf = nullptr;
  HRR_HIP_CHECK(hipMalloc(&buf, kBytes));
  std::vector<int> host(kElems);
  for (int i = 0; i < kElems; ++i) host[i] = seed + i;
  HRR_HIP_CHECK(hipMemcpy(buf, host.data(), kBytes, hipMemcpyHostToDevice));

  // The cell is filled the way vLLM fills block_table_ptrs: an H2D copy whose
  // payload is a device address.
  int** cell = nullptr;
  HRR_HIP_CHECK(hipMalloc(&cell, sizeof(int*)));
  HRR_HIP_CHECK(hipMemcpy(cell, &buf, sizeof(int*), hipMemcpyHostToDevice));

  hipLaunchKernelGGL(hrr_place_deref, dim3(kElems / 256), dim3(256), 0, nullptr,
                     out, cell, buf, kElems);
  HRR_HIP_CHECK(hipGetLastError());
  HRR_HIP_CHECK(hipDeviceSynchronize());

  std::vector<int> got(kElems);
  HRR_HIP_CHECK(hipMemcpy(got.data(), out, kBytes, hipMemcpyDeviceToHost));
  for (int i = 0; i < kElems; ++i) REQUIRE(got[i] == (seed + i) * 2);
  *cell_out = reinterpret_cast<int*>(cell);
  return buf;
}
}  // namespace

// ===========================================================================
// The captured workload.
// ===========================================================================
TEST_CASE("Unit_HRR_VaPlacement_Direct", "[.][hrr-direct]") {
  // ROCM-30200: the first HIP call in a captured process must not be the
  // allocation under test.
  HRR_HIP_CHECK(hipFree(nullptr));
  HRR_HIP_CHECK(hipSetDevice(0));

  int* out = nullptr;
  HRR_HIP_CHECK(hipMalloc(&out, kBytes));

  int* cell1 = nullptr;
  int* buf = hrr_place_round(out, 1, &cell1);

  // Replay has to answer this on memory it mapped itself.
  hipPointerAttribute_t attr{};
  HRR_HIP_CHECK(hipPointerGetAttributes(&attr, buf));
  REQUIRE(attr.type == hipMemoryTypeDevice);

  // Free and allocate the same size again. The allocator normally hands back
  // the address it just took, so replay has to unmap and map again at the
  // same place, inside a reservation it kept.
  void* freed = nullptr;
  HRR_HIP_CHECK(hipMalloc(&freed, kBytes));
  HRR_HIP_CHECK(hipFree(freed));
  int* cell2 = nullptr;
  int* again = hrr_place_round(out, 1000, &cell2);
  HRR_HIP_CHECK(hipPointerGetAttributes(&attr, again));
  REQUIRE(attr.type == hipMemoryTypeDevice);

  printf(HRR_PLACE_MARKER " buf=0x%llx freed=0x%llx again=0x%llx\n",
         static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(buf)),
         static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(freed)),
         static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(again)));
  fflush(stdout);

  HRR_HIP_CHECK(hipFree(cell2));
  HRR_HIP_CHECK(hipFree(again));
  HRR_HIP_CHECK(hipFree(cell1));
  HRR_HIP_CHECK(hipFree(buf));
  HRR_HIP_CHECK(hipFree(out));
}

namespace {
struct PlaceCapture {
  uint64_t buf = 0, freed = 0, again = 0;
  fs::path archive;
};

// Capture once per process. Catch2 runs the test case body again for every
// section, and each section only needs a fresh replay.
const PlaceCapture& hrr_place_capture() {
  static ScopedDir cap(fs::temp_directory_path() / "hrr_va_placement.hrr");
  static PlaceCapture pc;
  if (!pc.archive.empty()) return pc;

  std::string out;
  { hrr::test::SpawnProc proc(HRR_TEST_EXE, /*capture_stdout=*/true);
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    set_proc_search_path(proc);
    int ret = proc.run("\"Unit_HRR_VaPlacement_Direct\"");
    out = proc.getOutput();
    INFO("Capture exit: " << ret << "\n" << out);
    REQUIRE(ret == 0); }

  const size_t at = out.find(HRR_PLACE_MARKER);
  REQUIRE(at != std::string::npos);
  unsigned long long b = 0, f = 0, a = 0;
  REQUIRE(sscanf(out.c_str() + at, HRR_PLACE_MARKER " buf=0x%llx freed=0x%llx again=0x%llx",
                 &b, &f, &a) == 3);
  pc.buf = b; pc.freed = f; pc.again = a;
  pc.archive = hrr_single_process_archive(cap.path);
  return pc;
}

// The live value the pointer dump paired with a recorded one. See
// hrr_kernel_args_test.cc for why the live side is parsed rather than matched.
bool hrr_place_live_arg(const std::string& out, uint64_t recorded, uint64_t* live) {
  char needle[64];
  snprintf(needle, sizeof(needle), "recorded=0x%llx -> live=",
           static_cast<unsigned long long>(recorded));
  const size_t at = out.find(needle);
  if (at == std::string::npos) return false;
  const char* p = out.c_str() + at + strlen(needle);
  if (strncmp(p, "(nil)", 5) == 0) { *live = 0; return true; }
  *live = strtoull(p, nullptr, 16);
  return true;
}

// The counts from "[HRR]   Placement      : N placed at capture address, M fell back".
bool hrr_place_counts(const std::string& out, int* placed, int* fell) {
  const size_t at = out.find("Placement      :");
  if (at == std::string::npos) return false;
  return sscanf(out.c_str() + at, "Placement      : %d placed at capture address, %d fell back",
                placed, fell) == 2;
}

std::string hex(uint64_t v) {
  char b[32];
  snprintf(b, sizeof(b), "0x%llx", static_cast<unsigned long long>(v));
  return b;
}

// A replay without placement gets the stored pointer wrong, unless the runtime
// happened to return the recorded address anyway. The pointer dump of the
// first launch says which, so the negative assertions are skipped only when
// there was nothing to catch.
void hrr_require_stale(int rc, const std::string& out, uint64_t buf) {
  uint64_t live = 0;
  REQUIRE(hrr_place_live_arg(out, buf, &live));
  if (live == buf) {
    WARN("the runtime returned the recorded address by itself; nothing to catch");
    return;
  }
  int pass = 0, fail = 0;
  REQUIRE(hrr_parse_d2h_summary(out, pass, fail));
  CHECK(fail >= 1);
  CHECK(rc != 0);
}
}  // namespace

HRR_TEST_CASE(Unit_HRR_VaPlacement_StoredPointer) {
#ifdef _WIN32
  HRR_SKIP("placement needs mmap(MAP_FIXED_NOREPLACE) and is off on Windows");
#endif
  // Placement maps every allocation through the VMM API. Without it every
  // allocation falls back, and there is nothing here to assert.
  int vmm = 0;
  HRR_HIP_CHECK(hipDeviceGetAttribute(
      &vmm, hipDeviceAttributeVirtualMemoryManagementSupported, 0));
  if (!vmm) {
    HRR_SKIP("the device does not support virtual memory management");
  }
  const PlaceCapture& pc = hrr_place_capture();
  INFO("capture: buf=" << hex(pc.buf) << " freed=" << hex(pc.freed)
                       << " again=" << hex(pc.again));
  const std::vector<std::pair<std::string, std::string>> dump = {
      {"HIP_HRR_REPLAY_DUMP_PTRS_ORDINAL", "1"}};

  SECTION("placement on: the stored pointer is right and nothing falls back") {
    auto [rc, out] = hrr_playback_merged(pc.archive, "", dump);
    INFO("Replay:\n" << out);
    CHECK(rc == 0);
    int pass = 0, fail = 0;
    REQUIRE(hrr_parse_d2h_summary(out, pass, fail));
    CHECK(pass >= 2);
    CHECK(fail == 0);
    // out, buf, cell1, freed, again, cell2.
    int placed = 0, fell = -1;
    REQUIRE(hrr_place_counts(out, &placed, &fell));
    CHECK(placed >= 6);
    CHECK(fell == 0);
    uint64_t live = 0;
    REQUIRE(hrr_place_live_arg(out, pc.buf, &live));
    CHECK(live == pc.buf);
  }

  SECTION("--no-placement: the stored pointer is stale") {
    auto [rc, out] = hrr_playback_merged(pc.archive, "--no-placement", dump);
    INFO("Replay:\n" << out);
    CHECK(out.find("Placement : off (--no-placement)") != std::string::npos);
    CHECK(out.find("placed at capture address") == std::string::npos);
    hrr_require_stale(rc, out, pc.buf);
  }

  SECTION("free and allocate again lands at the recorded address again") {
    if (pc.freed != pc.again)
      WARN("the capture's allocator did not reuse the freed address ("
           << hex(pc.freed) << " then " << hex(pc.again)
           << "); only the free path is exercised");
    // The second round's launch is ordinal 2. Its buffer argument is `again`.
    auto [rc, out] = hrr_playback_merged(pc.archive, "",
                                         {{"HIP_HRR_REPLAY_DUMP_PTRS_ORDINAL", "2"}});
    INFO("Replay:\n" << out);
    CHECK(rc == 0);
    uint64_t live = 0;
    REQUIRE(hrr_place_live_arg(out, pc.again, &live));
    CHECK(live == pc.again);
    int placed = 0, fell = -1;
    REQUIRE(hrr_place_counts(out, &placed, &fell));
    CHECK(fell == 0);
  }

  SECTION("a range that is taken falls back and is named") {
    auto [rc, out] = hrr_playback_merged(
        pc.archive, "",
        {{"HIP_HRR_REPLAY_DUMP_PTRS_ORDINAL", "1"},
         {"HIP_HRR_REPLAY_PLACE_DENY", hex(pc.buf)},
         {"HIP_HRR_REPLAY_SCAN_H2D", "1"}});
    INFO("Replay:\n" << out);
    CHECK(out.find("treated as taken") != std::string::npos);
    // The allocation is named when it falls back...
    CHECK(out.find("hipMalloc " + hex(pc.buf) + " (" + std::to_string(kBytes) +
                   " bytes) not placed") != std::string::npos);
    // ...and the payload that stored its address is named by the scan.
    CHECK(out.find("a recorded address in allocation " + hex(pc.buf)) !=
          std::string::npos);
    // Only the denied one moved, so the scan names nothing else.
    CHECK(out.find("a recorded address in allocation " + hex(pc.again)) ==
          std::string::npos);
    int placed = 0, fell = 0;
    REQUIRE(hrr_place_counts(out, &placed, &fell));
    CHECK(fell == 1);
    hrr_require_stale(rc, out, pc.buf);
  }

  SECTION("hipPointerGetAttributes answers on placed memory") {
    // --continue-on-error turns a failing replayed call into a counted one,
    // so the summary says which call it was instead of the replay stopping.
    auto [rc, out] = hrr_playback_merged(pc.archive, "--continue-on-error");
    INFO("Replay:\n" << out);
    CHECK(rc == 0);
    CHECK(out.find("Events failed") == std::string::npos);
    CHECK(out.find("hipPointerGetAttributes") == std::string::npos);
    int placed = 0, fell = -1;
    REQUIRE(hrr_place_counts(out, &placed, &fell));
    CHECK(fell == 0);
  }

  SECTION("--guard-segments turns placement off") {
    auto [rc, out] = hrr_playback_merged(pc.archive, "--guard-segments", dump);
    INFO("Replay:\n" << out);
    CHECK(out.find("Placement : off (--guard-segments") != std::string::npos);
    CHECK(out.find("placed at capture address") == std::string::npos);
    hrr_require_stale(rc, out, pc.buf);
  }
}

#endif  // HRR_PLAYBACK_EXE && HRR_TEST_EXE

/**
 * @}
 */
