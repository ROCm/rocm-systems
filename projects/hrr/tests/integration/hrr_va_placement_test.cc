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

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
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

  // What the capture's hipPointerGetAttributes said about buf, for replay
  // under --verbose to be compared with.
  HRR_HIP_CHECK(hipPointerGetAttributes(&attr, buf));
  printf(HRR_PLACE_MARKER " buf=0x%llx freed=0x%llx again=0x%llx type=%d device=%d ptr=0x%llx\n",
         static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(buf)),
         static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(freed)),
         static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(again)),
         static_cast<int>(attr.type), attr.device,
         static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(attr.devicePointer)));
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
  int attr_type = -1, attr_device = -1;
  uint64_t attr_ptr = 0;
  fs::path archive;
};

// Capture once per process. Catch2 runs the test case body again for every
// section, and each section only needs a fresh replay.
const PlaceCapture& hrr_place_capture() {
  static ScopedDir cap(fs::temp_directory_path() / "hrr_va_placement.hrr");
  static PlaceCapture pc;
  if (!pc.archive.empty()) return pc;

  std::string out;
  { hrr::test::SpawnProc proc(hrr_test_exe(), /*capture_stdout=*/true);
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    set_proc_search_path(proc);
    int ret = proc.run("\"Unit_HRR_VaPlacement_Direct\"");
    out = proc.getOutput();
    INFO("Capture exit: " << ret << "\n" << out);
    REQUIRE(ret == 0); }

  const size_t at = out.find(HRR_PLACE_MARKER);
  REQUIRE(at != std::string::npos);
  unsigned long long b = 0, f = 0, a = 0, ap = 0;
  REQUIRE(sscanf(out.c_str() + at,
                 HRR_PLACE_MARKER " buf=0x%llx freed=0x%llx again=0x%llx type=%d device=%d "
                 "ptr=0x%llx",
                 &b, &f, &a, &pc.attr_type, &pc.attr_device, &ap) == 6);
  pc.buf = b; pc.freed = f; pc.again = a; pc.attr_ptr = ap;
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

size_t count_of(const std::string& text, const std::string& what) {
  size_t n = 0;
  for (size_t at = text.find(what); at != std::string::npos; at = text.find(what, at + 1)) ++n;
  return n;
}

std::string hex(uint64_t v) {
  char b[32];
  snprintf(b, sizeof(b), "0x%llx", static_cast<unsigned long long>(v));
  return b;
}

// A replay that does not place `buf` gets the stored pointer wrong. The
// caller denies `buf` with HIP_HRR_REPLAY_PLACE_DENY, which holds its range
// even with placement off, so the runtime cannot return the recorded address
// by chance. The pointer dump of the first launch says where it went.
void hrr_require_stale(int rc, const std::string& out, uint64_t buf) {
  uint64_t live = 0;
  REQUIRE(hrr_place_live_arg(out, buf, &live));
  REQUIRE(live != buf);
  int pass = 0, fail = 0;
  REQUIRE(hrr_parse_d2h_summary(out, pass, fail));
  CHECK(fail >= 1);
  CHECK(rc != 0);
}

// Placement maps every allocation through the VMM API. MI300 and MI350 have
// it, so there a runtime that says otherwise is a failure. Elsewhere the test
// skips, and says so.
void hrr_place_require_vmm() {
#ifdef _WIN32
  SKIP("placement needs mmap(MAP_FIXED_NOREPLACE) and is off on Windows");
#endif
  int vmm = 0;
  HRR_HIP_CHECK(hipDeviceGetAttribute(
      &vmm, hipDeviceAttributeVirtualMemoryManagementSupported, 0));
  if (vmm) return;
  hipDeviceProp_t props{};
  HRR_HIP_CHECK(hipGetDeviceProperties(&props, 0));
  const std::string arch = props.gcnArchName;
  if (arch.rfind("gfx942", 0) == 0 || arch.rfind("gfx950", 0) == 0)
    FAIL(arch << " supports virtual memory management, but the runtime says it does not");
  SKIP(arch << " does not support virtual memory management");
}

// How many frees the summary says were deferred to a later device sync: 0 when the
// clause is absent, -1 when the summary line is.
int hrr_place_deferred(const std::string& out) {
  const size_t at = out.find("Placement      :");
  if (at == std::string::npos) return -1;
  int placed = 0, fell = 0, deferred = 0;
  if (sscanf(out.c_str() + at,
             "Placement      : %d placed at capture address, %d fell back, %d frees deferred",
             &placed, &fell, &deferred) == 3)
    return deferred;
  return 0;
}

// The start of the line naming an allocation that fell back.
std::string hrr_place_named(const char* api, uint64_t rec, size_t size) {
  return std::string("[HRR] Placement: ") + api + " " + hex(rec) + " (" +
         std::to_string(size) + " bytes) not placed at its recorded address: ";
}

std::string hrr_place_attr_line(uint64_t rec, int type, int device, uint64_t ptr) {
  return "[HRR] hipPointerGetAttributes " + hex(rec) + " -> type=" + std::to_string(type) +
         " device=" + std::to_string(device) + " devicePointer=" + hex(ptr);
}
}  // namespace

HRR_TEST_CASE(Unit_HRR_VaPlacement_StoredPointer) {
  hrr_place_require_vmm();
  const PlaceCapture& pc = hrr_place_capture();
  INFO("capture: buf=" << hex(pc.buf) << " freed=" << hex(pc.freed)
                       << " again=" << hex(pc.again));
  const std::vector<std::pair<std::string, std::string>> dump = {
      {"HIP_HRR_REPLAY_DUMP_PTRS_ORDINAL", "1"}};
  // The sections that turn placement off deny `buf`, so it cannot come back
  // at its recorded address by chance.
  const std::vector<std::pair<std::string, std::string>> deny = {
      {"HIP_HRR_REPLAY_DUMP_PTRS_ORDINAL", "1"},
      {"HIP_HRR_REPLAY_PLACE_DENY", hex(pc.buf)}};

  SECTION("placement on: the stored pointer is right and nothing falls back") {
    auto [rc, out] = hrr_playback_merged(pc.archive, "", dump);
    INFO("Replay:\n" << out);
    CHECK(rc == 0);
    int pass = 0, fail = 0;
    REQUIRE(hrr_parse_d2h_summary(out, pass, fail));
    CHECK(pass >= 2);
    CHECK(fail == 0);
    CHECK(out.find(", 0 could not be") != std::string::npos);
    // out, buf, cell1, freed, again, cell2.
    int placed = 0, fell = -1;
    REQUIRE(hrr_place_counts(out, &placed, &fell));
    CHECK(placed >= 6);
    CHECK(fell == 0);
    uint64_t live = 0;
    REQUIRE(hrr_place_live_arg(out, pc.buf, &live));
    CHECK(live == pc.buf);
    // Nothing moved, so the scan for stored addresses never ran.
    CHECK(out.find("a recorded address in allocation") == std::string::npos);
  }

  SECTION("--no-placement: the stored pointer is stale") {
    auto [rc, out] = hrr_playback_merged(pc.archive, "--no-placement", deny);
    INFO("Replay:\n" << out);
    CHECK(out.find("Placement : off (--no-placement)") != std::string::npos);
    CHECK(out.find("1 range(s) treated as taken") != std::string::npos);
    CHECK(out.find("placed at capture address") == std::string::npos);
    hrr_require_stale(rc, out, pc.buf);
  }

  SECTION("free and allocate again lands at the recorded address again") {
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
    if (pc.freed != pc.again)
      SKIP("the capture's allocator did not reuse the freed address ("
           << hex(pc.freed) << " then " << hex(pc.again)
           << "); only the free path was exercised");
  }

  SECTION("a range that is taken falls back and is named") {
    // No HIP_HRR_REPLAY_SCAN_H2D: the scan starts by itself at the first
    // fallback.
    auto [rc, out] = hrr_playback_merged(
        pc.archive, "",
        {{"HIP_HRR_REPLAY_DUMP_PTRS_ORDINAL", "1"},
         {"HIP_HRR_REPLAY_PLACE_DENY", hex(pc.buf)}});
    INFO("Replay:\n" << out);
    CHECK(out.find("treated as taken") != std::string::npos);
    // The allocation is named when it falls back...
    CHECK(out.find(hrr_place_named("hipMalloc", pc.buf, kBytes)) != std::string::npos);
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

  SECTION("hipPointerGetAttributes on placed memory answers what the capture got") {
    // --continue-on-error turns a failing replayed call into a counted one,
    // so the summary says which call it was instead of the replay stopping.
    // --verbose prints what the replayed call answered.
    auto [rc, out] = hrr_playback_merged(pc.archive, "--verbose --continue-on-error");
    INFO("Replay:\n" << out);
    CHECK(rc == 0);
    CHECK(out.find("Events failed") == std::string::npos);
    CHECK(out.find(hrr_place_attr_line(pc.buf, pc.attr_type, pc.attr_device, pc.attr_ptr)) !=
          std::string::npos);
    int placed = 0, fell = -1;
    REQUIRE(hrr_place_counts(out, &placed, &fell));
    CHECK(fell == 0);
  }

  SECTION("--guard-segments turns placement off") {
    auto [rc, out] = hrr_playback_merged(pc.archive, "--guard-segments", deny);
    INFO("Replay:\n" << out);
    CHECK(out.find("Placement : off (--guard-segments") != std::string::npos);
    CHECK(out.find("1 range(s) treated as taken") != std::string::npos);
    CHECK(out.find("placed at capture address") == std::string::npos);
    hrr_require_stale(rc, out, pc.buf);
  }

  SECTION("HIP_HRR_REPLAY_ALLOC_PAD_FACTOR above 1 turns placement off") {
    auto [rc, out] = hrr_playback_merged(pc.archive, "",
                                         {{"HIP_HRR_REPLAY_DUMP_PTRS_ORDINAL", "1"},
                                          {"HIP_HRR_REPLAY_PLACE_DENY", hex(pc.buf)},
                                          {"HIP_HRR_REPLAY_ALLOC_PAD_FACTOR", "2"}});
    INFO("Replay:\n" << out);
    CHECK(out.find("Placement : off (HIP_HRR_REPLAY_ALLOC_PAD_FACTOR=2") != std::string::npos);
    CHECK(out.find("1 range(s) treated as taken") != std::string::npos);
    CHECK(out.find("placed at capture address") == std::string::npos);
    hrr_require_stale(rc, out, pc.buf);
  }

  SECTION("--kernel-filter: the timed pass places every allocation again") {
    // The silent warm-up pass replays everything first. What it left mapped
    // is released and the counts start again, so the timed pass reports its
    // own allocations and none of them collides with the warm-up's.
    auto [rc, out] = hrr_playback_merged(pc.archive, "--kernel-filter hrr_place_deref");
    INFO("Replay:\n" << out);
    CHECK(out.find(", 0 could not be") != std::string::npos);
    int placed = 0, fell = -1;
    REQUIRE(hrr_place_counts(out, &placed, &fell));
    CHECK(placed == 6);
    CHECK(fell == 0);
  }
}

// ===========================================================================
// Every placed API, and the ones that are not placed.
// ===========================================================================
namespace {
#define HRR_APIS_MARKER "HRR_PLACE_APIS"

// Fill `buf`, store its address in a fresh cell with an H2D copy, and read it
// back through the cell on `stream`. Returns the cell.
int** hrr_place_check(int* out, int* buf, int seed, hipStream_t stream) {
  std::vector<int> host(kElems);
  for (int i = 0; i < kElems; ++i) host[i] = seed + i;
  HRR_HIP_CHECK(hipMemcpy(buf, host.data(), kBytes, hipMemcpyHostToDevice));
  int** cell = nullptr;
  HRR_HIP_CHECK(hipMalloc(&cell, sizeof(int*)));
  HRR_HIP_CHECK(hipMemcpy(cell, &buf, sizeof(int*), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(hrr_place_deref, dim3(kElems / 256), dim3(256), 0, stream,
                     out, cell, buf, kElems);
  HRR_HIP_CHECK(hipGetLastError());
  HRR_HIP_CHECK(stream ? hipStreamSynchronize(stream) : hipDeviceSynchronize());
  std::vector<int> got(kElems);
  HRR_HIP_CHECK(hipMemcpy(got.data(), out, kBytes, hipMemcpyDeviceToHost));
  for (int i = 0; i < kElems; ++i) REQUIRE(got[i] == (seed + i) * 2);
  return cell;
}

unsigned long long u64(const void* p) {
  return static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(p));
}
}  // namespace

TEST_CASE("Unit_HRR_VaPlacement_Apis_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipFree(nullptr));
  HRR_HIP_CHECK(hipSetDevice(0));
  hipStream_t s = nullptr;
  HRR_HIP_CHECK(hipStreamCreate(&s));
  int* out = nullptr;
  HRR_HIP_CHECK(hipMalloc(&out, kBytes));
  std::vector<int**> cells;

  // hipExtMallocWithFlags without a flag is an ordinary allocation.
  int* ext = nullptr;
  HRR_HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&ext), kBytes,
                                      hipDeviceMallocDefault));
  cells.push_back(hrr_place_check(out, ext, 10, nullptr));

  // Stream-ordered, from the device's default pool.
  int* async = nullptr;
  HRR_HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&async), kBytes, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  cells.push_back(hrr_place_check(out, async, 20, s));
  HRR_HIP_CHECK(hipFreeAsync(async, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  // From a pool of its own. The two small allocations after it are likely
  // to share a page, which replay cannot map twice.
  hipMemPoolProps props{};
  props.allocType     = hipMemAllocationTypePinned;
  props.location.type = hipMemLocationTypeDevice;
  props.location.id   = 0;
  hipMemPool_t pool = nullptr;
  HRR_HIP_CHECK(hipMemPoolCreate(&pool, &props));
  int* frompool = nullptr;
  HRR_HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&frompool), kBytes, pool, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  cells.push_back(hrr_place_check(out, frompool, 30, s));
  void* small1 = nullptr;
  void* small2 = nullptr;
  HRR_HIP_CHECK(hipMallocFromPoolAsync(&small1, 256, pool, s));
  HRR_HIP_CHECK(hipMallocFromPoolAsync(&small2, 256, pool, s));
  HRR_HIP_CHECK(hipFreeAsync(small2, s));
  HRR_HIP_CHECK(hipFreeAsync(small1, s));
  HRR_HIP_CHECK(hipFreeAsync(frompool, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  HRR_HIP_CHECK(hipMemPoolDestroy(pool));

  // A VMM reservation with memory mapped in it. Replay reserves at the
  // recorded address as a hint.
  hipMemAllocationProp prop{};
  prop.type          = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id   = 0;
  size_t gran = 0;
  HRR_HIP_CHECK(hipMemGetAllocationGranularity(&gran, &prop,
                                               hipMemAllocationGranularityMinimum));
  const size_t vsz = (kBytes + gran - 1) / gran * gran;
  void* va = nullptr;
  HRR_HIP_CHECK(hipMemAddressReserve(&va, vsz, 0, nullptr, 0));
  hipMemGenericAllocationHandle_t handle{};
  HRR_HIP_CHECK(hipMemCreate(&handle, vsz, &prop, 0));
  HRR_HIP_CHECK(hipMemMap(va, vsz, 0, handle, 0));
  hipMemAccessDesc desc{};
  desc.location = prop.location;
  desc.flags    = hipMemAccessFlagsProtReadWrite;
  HRR_HIP_CHECK(hipMemSetAccess(va, vsz, &desc, 1));
  cells.push_back(hrr_place_check(out, static_cast<int*>(va), 40, nullptr));

  // Never placed: managed, fine-grained, and memory exported over IPC.
  void* managed = nullptr;
  HRR_HIP_CHECK(hipMallocManaged(&managed, kBytes));
  HRR_HIP_CHECK(hipMemset(managed, 0, kBytes));
  void* fine = nullptr;
  HRR_HIP_CHECK(hipExtMallocWithFlags(&fine, kBytes, hipDeviceMallocFinegrained));
  void* ipc = nullptr;
  HRR_HIP_CHECK(hipMalloc(&ipc, kBytes));
  hipIpcMemHandle_t ih{};
  // Some containers refuse IPC. The test then skips that check.
  const bool exported = hipIpcGetMemHandle(&ih, ipc) == hipSuccess;
  if (!exported) (void)hipGetLastError();

  // A free inside a relaxed-mode graph capture. Replay cannot unmap there:
  // hipMemUnmap waits for every stream, the capturing one included.
  void* doomed = nullptr;
  HRR_HIP_CHECK(hipMalloc(&doomed, kBytes));
  hipStream_t cs = nullptr;
  HRR_HIP_CHECK(hipStreamCreate(&cs));
  HRR_HIP_CHECK(hipStreamBeginCapture(cs, hipStreamCaptureModeRelaxed));
  HRR_HIP_CHECK(hipMemsetAsync(out, 0, kBytes, cs));
  HRR_HIP_CHECK(hipFree(doomed));
  hipGraph_t graph = nullptr;
  HRR_HIP_CHECK(hipStreamEndCapture(cs, &graph));
  hipGraphExec_t exec = nullptr;
  HRR_HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  HRR_HIP_CHECK(hipGraphLaunch(exec, cs));
  HRR_HIP_CHECK(hipStreamSynchronize(cs));
  HRR_HIP_CHECK(hipGraphExecDestroy(exec));
  HRR_HIP_CHECK(hipGraphDestroy(graph));
  HRR_HIP_CHECK(hipStreamDestroy(cs));
  // After the capture, the same size again, with its address stored.
  int* cell_after = nullptr;
  int* after = hrr_place_round(out, 50, &cell_after);

  printf(HRR_APIS_MARKER " ext=0x%llx async=0x%llx pool=0x%llx small1=0x%llx small2=0x%llx "
         "va=0x%llx vsz=%zu managed=0x%llx fine=0x%llx ipc=0x%llx doomed=0x%llx\n",
         u64(ext), u64(async), u64(frompool), u64(small1), u64(small2), u64(va), vsz,
         u64(managed), u64(fine), exported ? u64(ipc) : 0ull, u64(doomed));
  fflush(stdout);

  HRR_HIP_CHECK(hipFree(cell_after));
  HRR_HIP_CHECK(hipFree(after));
  HRR_HIP_CHECK(hipFree(ipc));
  HRR_HIP_CHECK(hipFree(fine));
  HRR_HIP_CHECK(hipFree(managed));
  HRR_HIP_CHECK(hipMemUnmap(va, vsz));
  HRR_HIP_CHECK(hipMemRelease(handle));
  HRR_HIP_CHECK(hipMemAddressFree(va, vsz));
  for (int** c : cells) HRR_HIP_CHECK(hipFree(c));
  HRR_HIP_CHECK(hipFree(ext));
  HRR_HIP_CHECK(hipFree(out));
  HRR_HIP_CHECK(hipStreamDestroy(s));
}

namespace {
struct ApisCapture {
  uint64_t ext = 0, async = 0, pool = 0, small1 = 0, small2 = 0, va = 0;
  uint64_t managed = 0, fine = 0, ipc = 0, doomed = 0;
  size_t vsz = 0;
  fs::path archive;
};

const ApisCapture& hrr_apis_capture() {
  static ScopedDir cap(fs::temp_directory_path() / "hrr_va_placement_apis.hrr");
  static ApisCapture c;
  if (!c.archive.empty()) return c;

  std::string out;
  { hrr::test::SpawnProc proc(hrr_test_exe(), /*capture_stdout=*/true);
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    set_proc_search_path(proc);
    int ret = proc.run("\"Unit_HRR_VaPlacement_Apis_Direct\"");
    out = proc.getOutput();
    INFO("Capture exit: " << ret << "\n" << out);
    REQUIRE(ret == 0); }

  const size_t at = out.find(HRR_APIS_MARKER);
  REQUIRE(at != std::string::npos);
  unsigned long long v[10] = {};
  size_t vsz = 0;
  REQUIRE(sscanf(out.c_str() + at,
                 HRR_APIS_MARKER " ext=0x%llx async=0x%llx pool=0x%llx small1=0x%llx "
                 "small2=0x%llx va=0x%llx vsz=%zu managed=0x%llx fine=0x%llx ipc=0x%llx "
                 "doomed=0x%llx",
                 &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &vsz, &v[6], &v[7], &v[8],
                 &v[9]) == 11);
  c.ext = v[0]; c.async = v[1]; c.pool = v[2]; c.small1 = v[3]; c.small2 = v[4];
  c.va = v[5]; c.vsz = vsz; c.managed = v[6]; c.fine = v[7]; c.ipc = v[8]; c.doomed = v[9];
  c.archive = hrr_single_process_archive(cap.path);
  return c;
}

// Whether two allocations touch a common unit of the mapping granularity,
// which replay cannot map twice.
bool hrr_share_page(uint64_t a, size_t asz, uint64_t b, size_t bsz) {
  hipMemAllocationProp prop{};
  prop.type          = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id   = 0;
  size_t gran = 0;
  HRR_HIP_CHECK(hipMemGetAllocationGranularity(&gran, &prop,
                                               hipMemAllocationGranularityMinimum));
  if (gran < 4096) gran = 4096;
  return a / gran <= (b + bsz - 1) / gran && b / gran <= (a + asz - 1) / gran;
}

// Which of the two small pool allocations replay leaves to the runtime. Each
// falls back when it shares a page with an allocation placed before it and
// still live: the pool allocation, or small1 if small1 was placed.
struct SmallFate {
  bool small1_fell = false, small2_fell = false;
  int fell() const { return (small1_fell ? 1 : 0) + (small2_fell ? 1 : 0); }
};
SmallFate hrr_small_fate(const ApisCapture& c) {
  SmallFate f;
  f.small1_fell = hrr_share_page(c.small1, 256, c.pool, kBytes);
  f.small2_fell = hrr_share_page(c.small2, 256, c.pool, kBytes) ||
                  (!f.small1_fell && hrr_share_page(c.small2, 256, c.small1, 256));
  return f;
}
}  // namespace

// Whether the loaded libamdhip64 logs a hinted hipMemAddressReserve only when
// the hint was missed. Before that fix in projects/clr/hipamd/src/hip_vm.cpp,
// every hinted reserve logged "Requested address was not allocated", so a
// section that checks the log is quiet for a placed reservation has nothing
// to check. The probe reserves anywhere, frees, and reserves again at that
// address, under AMD_LOG_LEVEL=1.
#define HRR_LOG_PROBE_MARKER "HRR_PLACE_LOGPROBE"

TEST_CASE("Unit_HRR_VaPlacement_LogProbe_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipFree(nullptr));
  hipMemAllocationProp prop{};
  prop.type          = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id   = 0;
  size_t gran = 0;
  HRR_HIP_CHECK(hipMemGetAllocationGranularity(&gran, &prop,
                                               hipMemAllocationGranularityMinimum));
  void* first = nullptr;
  HRR_HIP_CHECK(hipMemAddressReserve(&first, gran, 0, nullptr, 0));
  HRR_HIP_CHECK(hipMemAddressFree(first, gran));
  fflush(stderr);
  printf(HRR_LOG_PROBE_MARKER " hinted\n");
  fflush(stdout);
  void* again = nullptr;
  HRR_HIP_CHECK(hipMemAddressReserve(&again, gran, 0, first, 0));
  fflush(stderr);
  printf(HRR_LOG_PROBE_MARKER " honored=%d\n", again == first ? 1 : 0);
  fflush(stdout);
  HRR_HIP_CHECK(hipMemAddressFree(again, gran));
}

namespace {
enum class HintLog { Fixed, Unfixed, Unknown };

HintLog hrr_hint_log() {
  static const HintLog verdict = [] {
    hrr::test::SpawnProc proc(hrr_test_exe(), /*capture_stdout=*/true, /*capture_stderr=*/true);
    proc.setEnv("AMD_LOG_LEVEL", "1");
    set_proc_search_path(proc);
    if (proc.run("\"Unit_HRR_VaPlacement_LogProbe_Direct\"") != 0) return HintLog::Unknown;
    const std::string out = proc.getOutput();
    const size_t from = out.find(HRR_LOG_PROBE_MARKER " hinted");
    const size_t to   = out.find(HRR_LOG_PROBE_MARKER " honored=1");
    // A missed hint logs in every version, so it tells nothing.
    if (from == std::string::npos || to == std::string::npos) return HintLog::Unknown;
    return out.substr(from, to - from).find("Requested address was not allocated") ==
                   std::string::npos
               ? HintLog::Fixed
               : HintLog::Unfixed;
  }();
  return verdict;
}
}  // namespace

HRR_TEST_CASE(Unit_HRR_VaPlacement_Apis) {
  hrr_place_require_vmm();
  const ApisCapture& c = hrr_apis_capture();
  INFO("capture: ext=" << hex(c.ext) << " async=" << hex(c.async) << " pool=" << hex(c.pool)
       << " small1=" << hex(c.small1) << " small2=" << hex(c.small2) << " va=" << hex(c.va)
       << " managed=" << hex(c.managed) << " fine=" << hex(c.fine) << " ipc=" << hex(c.ipc)
       << " doomed=" << hex(c.doomed));

  SECTION("every placed API lands where it was recorded, and the rest are named") {
    auto [rc, out] = hrr_playback_merged(c.archive);
    INFO("Replay:\n" << out);
    CHECK(rc == 0);
    // One stored-pointer check each for hipExtMallocWithFlags, hipMallocAsync,
    // hipMallocFromPoolAsync, the VMM mapping, and the allocation after the
    // capture. A cell read stale fails its D2H.
    int pass = 0, fail = 0;
    REQUIRE(hrr_parse_d2h_summary(out, pass, fail));
    CHECK(pass >= 5);
    CHECK(fail == 0);
    CHECK(out.find(hrr_place_named("hipMallocManaged", c.managed, kBytes) +
                   "managed memory has no VMM equivalent") != std::string::npos);
    CHECK(out.find(hrr_place_named("hipExtMallocWithFlags", c.fine, kBytes) + "flags 0x1") !=
          std::string::npos);
    // `doomed` was freed inside the capture and unmapped after it.
    CHECK(hrr_place_deferred(out) >= 1);
    // out, ext and its cell, async and its cell, the pool allocation and its
    // cell, the reservation and its cell, doomed, after and its cell: 12, and
    // whichever small pool allocations do not share a page. Left to the
    // runtime: managed, fine, the IPC export, and the small ones that do.
    const SmallFate f = hrr_small_fate(c);
    const int expect_fell = 2 + (c.ipc != 0 ? 1 : 0) + f.fell();
    int placed = 0, fell = -1;
    REQUIRE(hrr_place_counts(out, &placed, &fell));
    CHECK(placed == 12 + (2 - f.fell()));
    CHECK(fell == expect_fell);
  }

  SECTION("memory exported over IPC is left to the runtime") {
    if (!c.ipc) SKIP("the capture could not export memory over IPC here");
    auto [rc, out] = hrr_playback_merged(c.archive);
    INFO("Replay:\n" << out);
    CHECK(out.find(hrr_place_named("hipMalloc", c.ipc, kBytes) +
                   "the recording exports it to another process") != std::string::npos);
  }

  SECTION("an allocation that shares a page with a live one falls back") {
    const SmallFate f = hrr_small_fate(c);
    if (f.fell() == 0)
      SKIP("the pool put " << hex(c.small1) << " and " << hex(c.small2)
                           << " in pages of their own");
    auto [rc, out] = hrr_playback_merged(c.archive);
    INFO("Replay:\n" << out);
    const uint64_t small = f.small1_fell ? c.small1 : c.small2;
    const std::string line =
        hrr_place_named("hipMallocFromPoolAsync", small, 256) + "it shares a page with live allocation ";
    const bool by_pool  = out.find(line + hex(c.pool)) != std::string::npos;
    const bool by_small = !f.small1_fell && out.find(line + hex(c.small1)) != std::string::npos;
    CHECK((by_pool || by_small));
  }

  SECTION("AMD_LOG_LEVEL=1: a reservation that gets its hint logs nothing") {
    const HintLog log = hrr_hint_log();
    if (log == HintLog::Unfixed)
      SKIP("the loaded libamdhip64 logs every hinted hipMemAddressReserve, so a placed "
           "reservation is not quiet; it lacks the hip_vm.cpp log fix");
    if (log == HintLog::Unknown)
      SKIP("the probe could not tell whether the loaded libamdhip64 logs a hint it met");
    auto [rc, out] = hrr_playback_merged(c.archive, "", {{"AMD_LOG_LEVEL", "1"}});
    INFO("Replay:\n" << out);
    CHECK(out.find("placed at capture address") != std::string::npos);
    CHECK(out.find("Requested address was not allocated") == std::string::npos);
  }

  SECTION("a reservation denied its address falls back, and the runtime log says so") {
    auto [rc, out] = hrr_playback_merged(
        c.archive, "", {{"AMD_LOG_LEVEL", "1"}, {"HIP_HRR_REPLAY_PLACE_DENY", hex(c.va)}});
    INFO("Replay:\n" << out);
    CHECK(out.find("Requested address was not allocated") != std::string::npos);
    CHECK(out.find(hrr_place_named("hipMemAddressReserve", c.va, c.vsz) +
                   "the runtime reserved ") != std::string::npos);
  }
}

// ===========================================================================
// When placed memory is released: frees deferred to a device sync, a capture that
// does not end cleanly, and a reservation still live after the warm-up pass.
// ===========================================================================
namespace {
#define HRR_LIFE_MARKER "HRR_PLACE_LIFE"
}  // namespace

TEST_CASE("Unit_HRR_VaPlacement_Lifetimes_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipFree(nullptr));
  HRR_HIP_CHECK(hipSetDevice(0));
  int* out = nullptr;
  HRR_HIP_CHECK(hipMalloc(&out, kBytes));

  // (a) A free inside a capture that a synchronous hipMemset then
  // invalidates, so hipStreamEndCapture fails. Failed calls are not recorded,
  // so the archive has neither; the capture ends at hipStreamDestroy, and the
  // free is unmapped at the hipDeviceSynchronize after it. The next
  // allocation at the freed address must be placed.
  void* doomed = nullptr;
  HRR_HIP_CHECK(hipMalloc(&doomed, kBytes));
  hipStream_t cs = nullptr;
  HRR_HIP_CHECK(hipStreamCreate(&cs));
  HRR_HIP_CHECK(hipStreamBeginCapture(cs, hipStreamCaptureModeRelaxed));
  HRR_HIP_CHECK(hipMemsetAsync(out, 0, kBytes, cs));
  HRR_HIP_CHECK(hipFree(doomed));
  REQUIRE(hipMemset(out, 0, kBytes) == hipErrorStreamCaptureImplicit);
  hipGraph_t graph = nullptr;
  REQUIRE(hipStreamEndCapture(cs, &graph) == hipErrorStreamCaptureInvalidated);
  (void)hipGetLastError();
  HRR_HIP_CHECK(hipStreamDestroy(cs));
  HRR_HIP_CHECK(hipDeviceSynchronize());
  int* cell_again = nullptr;
  int* again = hrr_place_round(out, 90, &cell_again);

  // (b) Stream-ordered frees. A stream sync does not unmap a1; the pool
  // usually hands its address straight back to a2, which then takes the
  // mapping back. The device sync unmaps what is left.
  hipStream_t s = nullptr;
  HRR_HIP_CHECK(hipStreamCreate(&s));
  void* a1 = nullptr;
  HRR_HIP_CHECK(hipMallocAsync(&a1, kBytes, s));
  HRR_HIP_CHECK(hipMemsetAsync(a1, 1, kBytes, s));
  HRR_HIP_CHECK(hipFreeAsync(a1, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  void* a2 = nullptr;
  HRR_HIP_CHECK(hipMallocAsync(&a2, kBytes, s));
  HRR_HIP_CHECK(hipMemsetAsync(a2, 2, kBytes, s));
  HRR_HIP_CHECK(hipFreeAsync(a2, s));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  // (f) A free inside a capture opened by hipStreamBeginCaptureToGraph. It
  // waits for the device sync after the capture ends.
  void* x = nullptr;
  HRR_HIP_CHECK(hipMalloc(&x, kBytes));
  hipStream_t s2 = nullptr;
  HRR_HIP_CHECK(hipStreamCreate(&s2));
  hipGraph_t g = nullptr;
  HRR_HIP_CHECK(hipGraphCreate(&g, 0));
  HRR_HIP_CHECK(hipStreamBeginCaptureToGraph(s2, g, nullptr, nullptr, 0,
                                             hipStreamCaptureModeRelaxed));
  HRR_HIP_CHECK(hipMemsetAsync(out, 0, kBytes, s2));
  HRR_HIP_CHECK(hipFree(x));
  hipGraph_t g2 = nullptr;
  HRR_HIP_CHECK(hipStreamEndCapture(s2, &g2));
  HRR_HIP_CHECK(hipGraphDestroy(g));
  HRR_HIP_CHECK(hipStreamDestroy(s2));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  // (g) A free inside a capture of this thread's default stream, opened and
  // closed with the _spt calls code built for per-thread streams makes. It
  // waits for the device sync after the capture ends.
  void* z = nullptr;
  HRR_HIP_CHECK(hipMalloc(&z, kBytes));
  HRR_HIP_CHECK(hipStreamBeginCapture_spt(nullptr, hipStreamCaptureModeRelaxed));
  HRR_HIP_CHECK(hipFree(z));
  hipGraph_t g3 = nullptr;
  HRR_HIP_CHECK(hipStreamEndCapture_spt(nullptr, &g3));
  if (g3) HRR_HIP_CHECK(hipGraphDestroy(g3));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  // (d) The same device address copied into 40 cells, one H2D copy each: 40
  // payloads for the scan to find it in once that allocation has moved.
  void* stored = nullptr;
  HRR_HIP_CHECK(hipMalloc(&stored, kBytes));
  void** cells = nullptr;
  HRR_HIP_CHECK(hipMalloc(&cells, 40 * sizeof(void*)));
  for (int i = 0; i < 40; ++i)
    HRR_HIP_CHECK(hipMemcpy(cells + i, &stored, sizeof(void*), hipMemcpyHostToDevice));
  HRR_HIP_CHECK(hipFree(cells));
  HRR_HIP_CHECK(hipFree(stored));

  // (e) A VMM reservation whose address is stored in device memory, still
  // reserved and mapped when the program exits.
  hipMemAllocationProp prop{};
  prop.type          = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id   = 0;
  size_t gran = 0;
  HRR_HIP_CHECK(hipMemGetAllocationGranularity(&gran, &prop,
                                               hipMemAllocationGranularityMinimum));
  const size_t vsz = (kBytes + gran - 1) / gran * gran;
  void* va = nullptr;
  HRR_HIP_CHECK(hipMemAddressReserve(&va, vsz, 0, nullptr, 0));
  hipMemGenericAllocationHandle_t handle{};
  HRR_HIP_CHECK(hipMemCreate(&handle, vsz, &prop, 0));
  HRR_HIP_CHECK(hipMemMap(va, vsz, 0, handle, 0));
  hipMemAccessDesc desc{};
  desc.location = prop.location;
  desc.flags    = hipMemAccessFlagsProtReadWrite;
  HRR_HIP_CHECK(hipMemSetAccess(va, vsz, &desc, 1));
  int** cell_va = hrr_place_check(out, static_cast<int*>(va), 100, nullptr);

  printf(HRR_LIFE_MARKER " doomed=0x%llx again=0x%llx va=0x%llx vsz=%zu stored=0x%llx"
         " a1=0x%llx a2=0x%llx\n",
         u64(doomed), u64(again), u64(va), vsz, u64(stored), u64(a1), u64(a2));
  fflush(stdout);

  HRR_HIP_CHECK(hipFree(cell_va));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  HRR_HIP_CHECK(hipFree(cell_again));
  HRR_HIP_CHECK(hipFree(again));
  HRR_HIP_CHECK(hipFree(out));
}

namespace {
struct LifeCapture {
  uint64_t doomed = 0, again = 0, va = 0, stored = 0, a1 = 0, a2 = 0;
  size_t vsz = 0;
  fs::path archive;
};

const LifeCapture& hrr_life_capture() {
  static ScopedDir cap(fs::temp_directory_path() / "hrr_va_placement_life.hrr");
  static LifeCapture c;
  if (!c.archive.empty()) return c;

  std::string out;
  { hrr::test::SpawnProc proc(hrr_test_exe(), /*capture_stdout=*/true);
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    set_proc_search_path(proc);
    int ret = proc.run("\"Unit_HRR_VaPlacement_Lifetimes_Direct\"");
    out = proc.getOutput();
    INFO("Capture exit: " << ret << "\n" << out);
    REQUIRE(ret == 0); }

  const size_t at = out.find(HRR_LIFE_MARKER);
  REQUIRE(at != std::string::npos);
  unsigned long long d = 0, a = 0, v = 0, st = 0, x1 = 0, x2 = 0;
  size_t vsz = 0;
  REQUIRE(sscanf(out.c_str() + at,
                 HRR_LIFE_MARKER " doomed=0x%llx again=0x%llx va=0x%llx vsz=%zu stored=0x%llx"
                 " a1=0x%llx a2=0x%llx",
                 &d, &a, &v, &vsz, &st, &x1, &x2) == 7);
  c.doomed = d; c.again = a; c.va = v; c.vsz = vsz; c.stored = st; c.a1 = x1; c.a2 = x2;
  c.archive = hrr_single_process_archive(cap.path);
  return c;
}
}  // namespace

HRR_TEST_CASE(Unit_HRR_VaPlacement_Lifetimes) {
  hrr_place_require_vmm();
  const LifeCapture& c = hrr_life_capture();
  INFO("capture: doomed=" << hex(c.doomed) << " again=" << hex(c.again) << " va=" << hex(c.va));

  SECTION("deferred frees are unmapped at the next device sync, and nowhere else") {
    // One --verbose replay serves sections (a), (b), (f) and (g) of the
    // workload.
    auto [rc, out] = hrr_playback_merged(c.archive, "--verbose");
    INFO("Replay:\n" << out);
    CHECK(rc == 0);
    int pass = 0, fail = 0;
    REQUIRE(hrr_parse_d2h_summary(out, pass, fail));
    CHECK(fail == 0);
    int placed = 0, fell = -1;
    REQUIRE(hrr_place_counts(out, &placed, &fell));
    CHECK(fell == 0);
    // Nothing is still mapped where a later allocation goes, so each lands.
    CHECK(out.find("is still mapped") == std::string::npos);

    // (a) The capture ends where its stream is destroyed, so the free made
    // inside it is unmapped at the next device sync, the first in the replay.
    const std::string want = "[HRR] Placement: unmapped 1 deferred free(s) at hipDeviceSynchronize";
    const size_t first = out.find("[HRR] Placement: unmapped ");
    REQUIRE(first != std::string::npos);
    CHECK(out.compare(first, want.size(), want) == 0);
    CHECK(out.find("deferred free(s) at hipStreamDestroy") == std::string::npos);
    if (c.doomed != c.again)
      WARN("the capture's allocator did not reuse the freed address ("
           << hex(c.doomed) << " then " << hex(c.again) << "), so nothing was placed over it");

    // (b) hipMemUnmap waits for every stream on the device: a stream sync
    // waited for one, so it unmaps nothing.
    CHECK(out.find("deferred free(s) at hipStreamSynchronize") == std::string::npos);
    // a2 took a1's mapping back when the pool reused its address, so the
    // device sync after them unmaps one; otherwise both. This assumes a2
    // either lands on a1's pages or misses them entirely: one that shared
    // only some of them would unmap a1 first, and the count would be 1 with
    // a1 != a2. A 4 KB allocation from a pool block makes that unlikely.
    const std::string b_line = std::string("[HRR] Placement: unmapped ") +
                               (c.a1 == c.a2 ? "1" : "2") +
                               " deferred free(s) at hipDeviceSynchronize";
    CHECK(out.find(b_line) != std::string::npos);
    if (c.a1 != c.a2)
      WARN("the pool did not hand a1's address to a2 (" << hex(c.a1) << " then " << hex(c.a2)
           << "), so no mapping was taken back");

    // doomed, x, z, and a2 when it took a1's mapping back.
    CHECK(count_of(out, "[HRR] Placement: unmapped 1 deferred free(s) at hipDeviceSynchronize") ==
          (c.a1 == c.a2 ? 4 : 3));
    // doomed, a1, a2, x, freed inside the capture hipStreamBeginCaptureToGraph
    // opened, and z, freed inside the one hipStreamBeginCapture_spt opened.
    CHECK(hrr_place_deferred(out) == 5);
  }

  SECTION("the H2D scan names the first 16 payloads, says why it runs, and counts the rest") {
    auto [rc, out] = hrr_playback_merged(c.archive, "",
                                         {{"HIP_HRR_REPLAY_PLACE_DENY", hex(c.stored)}});
    INFO("Replay:\n" << out);
    // The scan says once that it is on, and which allocation turned it on.
    CHECK(count_of(out, "replay scans the payload of each host-to-device hipMemcpy,") == 1);
    // The workload may reuse the address of an earlier allocation; the line
    // then names that one, at the same address.
    CHECK(out.find(" " + hex(c.stored) + " did not land at its recorded address") !=
          std::string::npos);
    // The fallback line itself no longer claims the scan.
    CHECK(out.find("now scans") == std::string::npos);
    CHECK(count_of(out, "recorded addresses in 8 bytes") == 16);
    CHECK(count_of(out, "further payloads holding recorded addresses are only counted") == 1);
    CHECK(out.find("H2D scan       : 40 payload(s) held a recorded address") != std::string::npos);
  }

  SECTION("--kernel-filter: a reservation the warm-up left live is reserved there again") {
    // The warm-up pass replays the hipMemAddressReserve the program never
    // freed. The timed pass replays it again and must get the same address.
    auto [rc, out] = hrr_playback_merged(c.archive, "--kernel-filter hrr_place_deref");
    INFO("Replay:\n" << out);
    CHECK(rc == 0);
    CHECK(out.find(hrr_place_named("hipMemAddressReserve", c.va, c.vsz)) == std::string::npos);
    int placed = 0, fell = -1;
    REQUIRE(hrr_place_counts(out, &placed, &fell));
    CHECK(fell == 0);
  }
}

// ===========================================================================
// A freed mapping taken back on another stream waits for the free.
// ===========================================================================
// The pool hands a block freed on one stream to another only once the free is
// done, or ordered after it. Replay takes the placed mapping back at once, so
// the new stream has to wait for the free there, or its writes race the old
// stream's last kernel. The recording sleeps, which is not a HIP call and is
// not replayed, so in replay that kernel is still running when the second
// stream takes the memory.
namespace {
#define HRR_ORDER_MARKER "HRR_PLACE_ORDER"
// The bit patterns of 1.0f and 3.0f: the D2H check compares 4-byte words as
// floats within a tolerance, so the two values must differ as floats too.
constexpr int kOldFill = 0x3f800000;
constexpr int kNewFill = 0x40400000;
}  // namespace

// Spins for `ticks` of the constant-rate wall clock, then fills `p` with `v`.
__global__ void hrr_place_late_fill(int* p, int v, unsigned long long ticks, int n) {
  const unsigned long long t0 = wall_clock64();
  while (wall_clock64() - t0 < ticks) {
  }
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = v;
}

__global__ void hrr_place_fill(int* p, int v, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = v;
}

TEST_CASE("Unit_HRR_VaPlacement_StreamOrder_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipFree(nullptr));
  HRR_HIP_CHECK(hipSetDevice(0));
  int khz = 0;
  HRR_HIP_CHECK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, 0));
  REQUIRE(khz > 0);
  const unsigned long long one_second = static_cast<unsigned long long>(khz) * 1000;

  hipStream_t s1 = nullptr, s2 = nullptr;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking));
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s2, hipStreamNonBlocking));

  // The old stream's last kernel writes x a second after it starts, and x is
  // freed behind it.
  int* x = nullptr;
  HRR_HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&x), kBytes, s1));
  hipLaunchKernelGGL(hrr_place_late_fill, dim3(kElems / 256), dim3(256), 0, s1, x,
                     kOldFill, one_second, kElems);
  HRR_HIP_CHECK(hipGetLastError());
  HRR_HIP_CHECK(hipFreeAsync(x, s1));
  std::this_thread::sleep_for(std::chrono::seconds(2));

  // By now that kernel is done, so the pool hands x's block to s2.
  int* y = nullptr;
  HRR_HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&y), kBytes, s2));
  hipLaunchKernelGGL(hrr_place_fill, dim3(kElems / 256), dim3(256), 0, s2, y, kNewFill,
                     kElems);
  HRR_HIP_CHECK(hipGetLastError());
  HRR_HIP_CHECK(hipDeviceSynchronize());
  std::vector<int> got(kElems);
  HRR_HIP_CHECK(hipMemcpy(got.data(), y, kBytes, hipMemcpyDeviceToHost));
  for (int i = 0; i < kElems; ++i) REQUIRE(got[i] == kNewFill);

  printf(HRR_ORDER_MARKER " x=0x%llx y=0x%llx\n", u64(x), u64(y));
  fflush(stdout);

  HRR_HIP_CHECK(hipFreeAsync(y, s2));
  HRR_HIP_CHECK(hipStreamSynchronize(s2));
  HRR_HIP_CHECK(hipStreamDestroy(s2));
  HRR_HIP_CHECK(hipStreamDestroy(s1));
}

HRR_TEST_CASE(Unit_HRR_VaPlacement_StreamOrder) {
  hrr_place_require_vmm();
  ScopedDir cap(fs::temp_directory_path() / "hrr_va_placement_order.hrr");
  std::string cout_;
  { hrr::test::SpawnProc proc(hrr_test_exe(), /*capture_stdout=*/true);
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    set_proc_search_path(proc);
    int ret = proc.run("\"Unit_HRR_VaPlacement_StreamOrder_Direct\"");
    cout_ = proc.getOutput();
    INFO("Capture exit: " << ret << "\n" << cout_);
    REQUIRE(ret == 0); }
  const size_t at = cout_.find(HRR_ORDER_MARKER);
  REQUIRE(at != std::string::npos);
  unsigned long long x = 0, y = 0;
  REQUIRE(sscanf(cout_.c_str() + at, HRR_ORDER_MARKER " x=0x%llx y=0x%llx", &x, &y) == 2);
  if (x != y)
    SKIP("the pool did not hand x's block to the second stream (" << hex(x) << " then "
         << hex(y) << "), so no mapping is taken back across streams");

  // Unordered, the second stream fills y at once, and the old kernel then
  // overwrites it a second later: the D2H check of y fails.
  auto [rc, out] = hrr_playback_merged(hrr_single_process_archive(cap.path), "");
  INFO("Replay:\n" << out);
  CHECK(rc == 0);
  int pass = 0, fail = 0;
  REQUIRE(hrr_parse_d2h_summary(out, pass, fail));
  CHECK(pass >= 1);
  CHECK(fail == 0);
  int placed = 0, fell = -1;
  REQUIRE(hrr_place_counts(out, &placed, &fell));
  CHECK(placed >= 2);
  CHECK(fell == 0);
}

// ===========================================================================
// Two GPUs: allocations on each, and device 0 reading device 1's memory.
// ===========================================================================
// Kernels run on device 0 only. Replay loads each code object for the device
// current at load time, so a launch on device 1 is a separate limitation.
namespace {
constexpr size_t kMultiGpuBig = 256ull << 20;
}  // namespace

TEST_CASE("Unit_HRR_VaPlacement_MultiGpu_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipFree(nullptr));
  int n = 0;
  HRR_HIP_CHECK(hipGetDeviceCount(&n));
  if (n < 2) SKIP("needs at least 2 GPUs; " << n << " visible");

  HRR_HIP_CHECK(hipSetDevice(1));
  hipStream_t s1 = nullptr;
  HRR_HIP_CHECK(hipStreamCreate(&s1));
  HRR_HIP_CHECK(hipDeviceEnablePeerAccess(0, 0));
  int* buf1 = nullptr;
  HRR_HIP_CHECK(hipMalloc(&buf1, kBytes));

  HRR_HIP_CHECK(hipSetDevice(0));
  HRR_HIP_CHECK(hipDeviceEnablePeerAccess(1, 0));
  int* out0 = nullptr;
  HRR_HIP_CHECK(hipMalloc(&out0, kBytes));
  int* buf0 = nullptr;
  HRR_HIP_CHECK(hipMalloc(&buf0, kBytes));
  // Pool memory is not covered by hipDeviceEnablePeerAccess: device 0 gets
  // into device 1's default pool by its own grant.
  hipMemPool_t pool1 = nullptr;
  HRR_HIP_CHECK(hipDeviceGetDefaultMemPool(&pool1, 1));
  hipMemAccessDesc desc{};
  desc.location.type = hipMemLocationTypeDevice;
  desc.location.id   = 0;
  desc.flags         = hipMemAccessFlagsProtReadWrite;
  HRR_HIP_CHECK(hipMemPoolSetAccess(pool1, &desc, 1));
  // A large allocation on device 1's stream, made with device 0 current,
  // between hipMemGetInfo calls on both devices. Replay under --verbose
  // prints what each call says there, which shows whose memory it took. It
  // comes first and is freed at once, because the pool packs its blocks
  // together, and one sharing a granule with a live block falls back.
  size_t free_b = 0, total_b = 0;
  HRR_HIP_CHECK(hipSetDevice(1));
  HRR_HIP_CHECK(hipMemGetInfo(&free_b, &total_b));
  HRR_HIP_CHECK(hipSetDevice(0));
  HRR_HIP_CHECK(hipMemGetInfo(&free_b, &total_b));
  void* big1 = nullptr;
  HRR_HIP_CHECK(hipMallocAsync(&big1, kMultiGpuBig, s1));
  HRR_HIP_CHECK(hipStreamSynchronize(s1));
  HRR_HIP_CHECK(hipSetDevice(1));
  HRR_HIP_CHECK(hipMemGetInfo(&free_b, &total_b));
  HRR_HIP_CHECK(hipSetDevice(0));
  HRR_HIP_CHECK(hipMemGetInfo(&free_b, &total_b));
  HRR_HIP_CHECK(hipFreeAsync(big1, s1));
  HRR_HIP_CHECK(hipStreamSynchronize(s1));
  // Once that free is done, the pool hands big1's block to another stream of
  // device 1. Replay recorded an event at the free on s1, with device 0
  // current, and the new stream waits for it before taking the mapping back.
  HRR_HIP_CHECK(hipSetDevice(1));
  hipStream_t s1b = nullptr;
  HRR_HIP_CHECK(hipStreamCreate(&s1b));
  HRR_HIP_CHECK(hipSetDevice(0));
  void* again = nullptr;
  HRR_HIP_CHECK(hipMallocAsync(&again, kMultiGpuBig, s1b));
  HRR_HIP_CHECK(hipFreeAsync(again, s1b));
  HRR_HIP_CHECK(hipStreamSynchronize(s1b));

  // Allocated while device 0 is current, on device 1's stream: it lives on
  // device 1, and replay has to map it there.
  int* async1 = nullptr;
  HRR_HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&async1), kBytes, s1));
  HRR_HIP_CHECK(hipStreamSynchronize(s1));
  hipPointerAttribute_t attr{};
  HRR_HIP_CHECK(hipPointerGetAttributes(&attr, async1));
  REQUIRE(attr.device == 1);

  // Device 0 reads its own buffer and both of device 1's through stored
  // pointers. Without peer access in replay, the last two fault.
  int** cell0 = hrr_place_check(out0, buf0, 60, nullptr);
  int** cell1 = hrr_place_check(out0, buf1, 70, nullptr);
  int** cella = hrr_place_check(out0, async1, 80, nullptr);
  // And a peer copy from device 1 back to device 0.
  HRR_HIP_CHECK(hipMemcpyPeer(buf0, 0, async1, 1, kBytes));
  std::vector<int> got(kElems);
  HRR_HIP_CHECK(hipMemcpy(got.data(), buf0, kBytes, hipMemcpyDeviceToHost));
  for (int i = 0; i < kElems; ++i) REQUIRE(got[i] == 80 + i);

  printf(HRR_PLACE_MARKER " buf0=0x%llx async1=0x%llx type=%d big1=0x%llx again=0x%llx\n",
         u64(buf0), u64(async1), static_cast<int>(attr.type), u64(big1), u64(again));
  fflush(stdout);

  HRR_HIP_CHECK(hipFree(cella));
  HRR_HIP_CHECK(hipFree(cell1));
  HRR_HIP_CHECK(hipFree(cell0));
  HRR_HIP_CHECK(hipFreeAsync(async1, s1));
  HRR_HIP_CHECK(hipStreamSynchronize(s1));
  HRR_HIP_CHECK(hipFree(buf0));
  HRR_HIP_CHECK(hipFree(out0));
  HRR_HIP_CHECK(hipFree(buf1));
  HRR_HIP_CHECK(hipStreamDestroy(s1b));
  HRR_HIP_CHECK(hipStreamDestroy(s1));
}

HRR_TEST_CASE(Unit_HRR_VaPlacement_MultiGpu) {
  int n = 0;
  HRR_HIP_CHECK(hipGetDeviceCount(&n));
  if (n < 2) SKIP("needs at least 2 GPUs; " << n << " visible");
  int can01 = 0, can10 = 0;
  HRR_HIP_CHECK(hipDeviceCanAccessPeer(&can01, 0, 1));
  HRR_HIP_CHECK(hipDeviceCanAccessPeer(&can10, 1, 0));
  if (!can01 || !can10) SKIP("devices 0 and 1 cannot reach each other's memory");
  hrr_place_require_vmm();

  ScopedDir cap(fs::temp_directory_path() / "hrr_va_placement_multigpu.hrr");
  std::string cout_;
  { hrr::test::SpawnProc proc(hrr_test_exe(), /*capture_stdout=*/true);
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    set_proc_search_path(proc);
    int ret = proc.run("\"Unit_HRR_VaPlacement_MultiGpu_Direct\"");
    cout_ = proc.getOutput();
    INFO("Capture exit: " << ret << "\n" << cout_);
    REQUIRE(ret == 0); }
  const size_t at = cout_.find(HRR_PLACE_MARKER);
  REQUIRE(at != std::string::npos);
  unsigned long long buf0 = 0, async1 = 0, big1 = 0, again = 0;
  int type = -1;
  REQUIRE(sscanf(cout_.c_str() + at,
                 HRR_PLACE_MARKER " buf0=0x%llx async1=0x%llx type=%d big1=0x%llx again=0x%llx",
                 &buf0, &async1, &type, &big1, &again) == 5);

  auto [rc, out] = hrr_playback_merged(hrr_single_process_archive(cap.path), "--verbose");
  INFO("Replay:\n" << out);
  CHECK(rc == 0);
  int pass = 0, fail = 0;
  REQUIRE(hrr_parse_d2h_summary(out, pass, fail));
  CHECK(pass >= 4);
  CHECK(fail == 0);
  // buf1, out0, buf0, async1, big1 and three cells.
  int placed = 0, fell = -1;
  REQUIRE(hrr_place_counts(out, &placed, &fell));
  CHECK(placed >= 8);
  CHECK(fell == 0);
  // The async allocation was mapped on its stream's device, not the current one.
  CHECK(out.find(hrr_place_attr_line(async1, type, 1, async1)) != std::string::npos);
  // And its memory came from that device: hipMemCreate takes it from the
  // current one, whatever the allocation's location says. The four
  // hipMemGetInfo lines are device 1, device 0, before big1, then after it.
  std::vector<std::pair<int, unsigned long long>> info;
  const std::string tag = "[HRR] hipMemGetInfo device=";
  for (size_t p = out.find(tag); p != std::string::npos; p = out.find(tag, p + 1)) {
    int d = -1;
    unsigned long long f = 0;
    if (sscanf(out.c_str() + p + tag.size(), "%d free=%llu", &d, &f) == 2) info.push_back({d, f});
  }
  REQUIRE(info.size() == 4);
  REQUIRE(info[0].first == 1);
  REQUIRE(info[1].first == 0);
  REQUIRE(info[2].first == 1);
  REQUIRE(info[3].first == 0);
  const long long took1 = static_cast<long long>(info[0].second - info[2].second);
  const long long took0 = static_cast<long long>(info[1].second - info[3].second);
  INFO("device 1 gave " << took1 << " bytes, device 0 " << took0);
  // With a margin: another process may allocate or free on either device
  // meanwhile.
  CHECK(took1 >= static_cast<long long>(kMultiGpuBig / 4 * 3));
  CHECK(took0 < static_cast<long long>(kMultiGpuBig / 4));
  // big1 was freed on device 1's stream with device 0 current. The event
  // recorded at that free is on device 1 too, so the other stream that got
  // the block back waits for it rather than unmapping the mapping first.
  if (again == big1) {
    CHECK(out.find("no event could be recorded after the hipFreeAsync of " + hex(big1)) ==
          std::string::npos);
    CHECK(out.find("[HRR] Placement: hipMallocAsync " + hex(big1) +
                   " takes back the mapping freed there (its stream waits for the free)") !=
          std::string::npos);
  } else {
    WARN("the pool did not hand big1's block to the other stream; the take-back is not "
         "checked");
  }
}

#endif  // HRR_PLAYBACK_EXE && HRR_TEST_EXE

/**
 * @}
 */
