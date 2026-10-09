/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR capabilities and addresses kept out of the archive
 * @{
 * @ingroup HRRTest
 * Tests that capture records that a call happened without recording what the
 * call handed back when that is a capability or an address in the capturing
 * process.
 *
 * An IPC memory handle lets any process on the machine open the allocation
 * while its exporter lives, so the archive holds a digest of it: two calls
 * with the same handle record the same digest and two with different handles
 * do not, which is all replay needs to pair an import with its export. A
 * resolved entry point from hipGetProcAddress or hipGetDriverEntryPoint is an
 * address in the runtime as the capturing process loaded it, and replay looks
 * the name up again, so it is not recorded at all.
 */

#include "hrr_test_common.hh"
#include "hrr_test_process.hh"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#if defined(HRR_PLAYBACK_EXE) && defined(HRR_TEST_EXE)

namespace {
#define HRR_CAPS_IPC "HRR_CAPS_IPC"
#define HRR_CAPS_PROC "HRR_CAPS_PROC"
#define HRR_CAPS_ENTRY "HRR_CAPS_ENTRY"

constexpr size_t kHandleBytes = sizeof(hipIpcMemHandle_t);

void print_handle(int idx, const hipIpcMemHandle_t& h) {
  const auto* p = reinterpret_cast<const uint8_t*>(&h);
  printf(HRR_CAPS_IPC " %d ", idx);
  for (size_t i = 0; i < kHandleBytes; ++i) printf("%02x", p[i]);
  printf("\n");
}
}  // namespace

// ===========================================================================
// The captured workload: three handle exports, two of them of one allocation,
// and both entry-point lookups. What each call returned goes to stdout, so
// the parent can look for it in the archive.
// ===========================================================================
TEST_CASE("Unit_HRR_CaptureCapabilities_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  void* a = nullptr;
  void* b = nullptr;
  HRR_HIP_CHECK(hipMalloc(&a, 1 << 20));
  HRR_HIP_CHECK(hipMalloc(&b, 1 << 20));

  void* const exported[] = {a, a, b};
  for (int i = 0; i < 3; ++i) {
    hipIpcMemHandle_t h{};
    if (hipIpcGetMemHandle(&h, exported[i]) == hipSuccess) print_handle(i, h);
  }

  void* proc = nullptr;
  hipDriverProcAddressQueryResult proc_status{};
  HRR_HIP_CHECK(hipGetProcAddress("hipMalloc", &proc,
                                HIP_VERSION_MAJOR * 100 + HIP_VERSION_MINOR,
                                hipEnableDefault, &proc_status));
  printf(HRR_CAPS_PROC " 0x%llx\n",
         static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(proc)));

  void* entry = nullptr;
  hipDriverEntryPointQueryResult entry_status = hipDriverEntryPointSuccess;
  if (hipGetDriverEntryPoint("hipMalloc", &entry, 0, &entry_status) == hipSuccess)
    printf(HRR_CAPS_ENTRY " 0x%llx\n",
           static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(entry)));
  fflush(stdout);

  HRR_HIP_CHECK(hipFree(b));
  HRR_HIP_CHECK(hipFree(a));
}

namespace {
std::vector<uint8_t> parse_hex(const std::string& hex) {
  std::vector<uint8_t> out;
  for (size_t i = 0; i + 1 < hex.size(); i += 2)
    out.push_back(static_cast<uint8_t>(std::stoul(hex.substr(i, 2), nullptr, 16)));
  return out;
}

bool all_zero(const uint8_t* p, size_t n) {
  for (size_t i = 0; i < n; ++i)
    if (p[i]) return false;
  return true;
}
}  // namespace

// ---------------------------------------------------------------------------
// One capture, then the archive record by record, then a replay of it.
// ---------------------------------------------------------------------------
HRR_TEST_CASE(Unit_HRR_CaptureCapabilities) {
  ScopedDir cap(fs::temp_directory_path() / "hrr_capture_capabilities.hrr");
  std::string out;
  { hrr::test::SpawnProc proc(HRR_TEST_EXE, /*capture_stdout=*/true);
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    set_proc_search_path(proc);
    int ret = proc.run("\"Unit_HRR_CaptureCapabilities_Direct\"");
    out = proc.getOutput();
    INFO("Capture exit: " << ret << "\n" << out);
    REQUIRE(ret == 0); }
  INFO("Capture output:\n" << out);

  // What the workload saw.
  std::map<int, std::vector<uint8_t>> handles;
  unsigned long long proc_addr = 0, entry_addr = 0;
  bool have_entry = false;
  { std::istringstream lines(out);
    std::string line;
    while (std::getline(lines, line)) {
      std::istringstream ls(line);
      std::string tag;
      ls >> tag;
      if (tag == HRR_CAPS_IPC) {
        int idx = -1;
        std::string hex;
        ls >> idx >> hex;
        handles[idx] = parse_hex(hex);
      } else if (tag == HRR_CAPS_PROC) {
        ls >> std::hex >> proc_addr;
      } else if (tag == HRR_CAPS_ENTRY) {
        ls >> std::hex >> entry_addr;
        have_entry = true;
      }
    } }
  // Something to leak, or the checks below prove nothing.
  REQUIRE(proc_addr != 0);
#ifndef _WIN32
  REQUIRE(handles.size() == 3);
#endif

  hrr::Archive arc;
  REQUIRE(hrr::load_archive(hrr_single_process_archive(cap.path).string(), arc));

  std::vector<const hrr_args_hipIpcGetMemHandle*> ipc;
  const hrr_args_hipGetProcAddress* lookup = nullptr;
  const hrr_args_hipGetDriverEntryPoint* entry_ev = nullptr;
  for (const auto& ev : arc.events) {
    const uint16_t t = ev.header().event_type;
    if (t == HRR_API_HIPIPCGETMEMHANDLE) {
      REQUIRE(ev.raw_payload.size() >= sizeof(hrr_args_hipIpcGetMemHandle));
      ipc.push_back(reinterpret_cast<const hrr_args_hipIpcGetMemHandle*>(ev.raw_payload.data()));
    } else if (t == HRR_API_HIPGETPROCADDRESS) {
      REQUIRE(ev.raw_payload.size() >= sizeof(hrr_args_hipGetProcAddress));
      lookup = reinterpret_cast<const hrr_args_hipGetProcAddress*>(ev.raw_payload.data());
    } else if (t == HRR_API_HIPGETDRIVERENTRYPOINT) {
      REQUIRE(ev.raw_payload.size() >= sizeof(hrr_args_hipGetDriverEntryPoint));
      entry_ev = reinterpret_cast<const hrr_args_hipGetDriverEntryPoint*>(ev.raw_payload.data());
    }
  }

  // Each export is recorded, as a digest and zeros rather than the handle,
  // and the digests compare the way the handles did.
  REQUIRE(ipc.size() == handles.size());
  for (size_t i = 0; i < ipc.size(); ++i) {
    INFO("export " << i);
    CHECK(ipc[i]->handle_present == 1);
    CHECK(std::memcmp(ipc[i]->handle_bytes, handles[int(i)].data(), kHandleBytes) != 0);
    CHECK(all_zero(ipc[i]->handle_bytes + 16, kHandleBytes - 16));
    CHECK_FALSE(all_zero(ipc[i]->handle_bytes, 16));
  }
  for (size_t i = 0; i < ipc.size(); ++i)
    for (size_t j = i + 1; j < ipc.size(); ++j) {
      INFO("exports " << i << " and " << j);
      const bool same_handle = handles[int(i)] == handles[int(j)];
      const bool same_digest =
          std::memcmp(ipc[i]->handle_bytes, ipc[j]->handle_bytes, 16) == 0;
      CHECK(same_handle == same_digest);
    }
#ifndef _WIN32
  // The same allocation exported twice is the case replay pairs on.
  CHECK(handles[0] == handles[1]);
  CHECK(handles[0] != handles[2]);
#endif

  // The lookup keeps its name and flags and loses its answer.
  REQUIRE(lookup != nullptr);
  CHECK(lookup->pfn == 0);
  CHECK(lookup->symbol_present == 1);
  CHECK(std::string(reinterpret_cast<const char*>(lookup->symbol_bytes)) == "hipMalloc");
  if (have_entry) {
    REQUIRE(entry_ev != nullptr);
    CHECK(entry_addr != 0);
    CHECK(entry_ev->funcPtr == 0);
  }

  // Replay re-exports and looks the name up again from what is left.
  hrr::test::SpawnProc replay(HRR_PLAYBACK_EXE, /*capture_stdout=*/true);
  set_proc_search_path(replay);
  const int ret = replay.run(hrr_quote_path(cap.path));
  INFO("Playback stdout:\n" << replay.getOutput());
  CHECK(ret == 0);
}

#endif  // HRR_PLAYBACK_EXE && HRR_TEST_EXE

/**
 * @}
 */
