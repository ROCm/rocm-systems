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
 * An IPC handle lets any process on the machine open the allocation while its
 * exporter lives, so the archive leaves its bytes zero. A resolved entry point
 * from hipGetProcAddress or hipGetDriverEntryPoint is an address in the runtime
 * as the capturing process loaded it, and replay looks the name up again, so
 * it is not recorded either.
 */

#include "hrr_test_common.hh"
#include "hrr_test_process.hh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#if defined(HRR_PLAYBACK_EXE) && defined(HRR_TEST_EXE)

namespace {
#define HRR_CAPS_IPC "HRR_CAPS_IPC"
#define HRR_CAPS_PROC "HRR_CAPS_PROC"
#define HRR_CAPS_PROC_SPT "HRR_CAPS_PROC_SPT"
#define HRR_CAPS_ENTRY "HRR_CAPS_ENTRY"
#define HRR_CAPS_ENTRY_SPT "HRR_CAPS_ENTRY_SPT"
#define HRR_CAPS_EXPORT_MEM "HRR_CAPS_EXPORT_MEM"
#define HRR_CAPS_EXPORT_EVENT "HRR_CAPS_EXPORT_EVENT"
#define HRR_CAPS_OPEN_MEM "HRR_CAPS_OPEN_MEM"
#define HRR_CAPS_OPEN_EVENT "HRR_CAPS_OPEN_EVENT"
#define HRR_CAPS_IMPORTER "HRR_CAPS_IMPORTER"

constexpr size_t kHandleBytes = sizeof(hipIpcMemHandle_t);
constexpr size_t kEventHandleBytes = sizeof(hipIpcEventHandle_t);

std::string to_hex(const void* data, size_t n) {
  const auto* p = static_cast<const uint8_t*>(data);
  std::string out;
  char b[3];
  for (size_t i = 0; i < n; ++i) {
    snprintf(b, sizeof(b), "%02x", p[i]);
    out += b;
  }
  return out;
}

std::vector<uint8_t> parse_hex(const std::string& hex) {
  std::vector<uint8_t> out;
  for (size_t i = 0; i + 1 < hex.size(); i += 2)
    out.push_back(static_cast<uint8_t>(std::stoul(hex.substr(i, 2), nullptr, 16)));
  return out;
}

void print_addr(const char* tag, void* p) {
  printf("%s 0x%llx\n", tag, static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(p)));
}

bool all_zero(const uint8_t* p, size_t n) {
  for (size_t i = 0; i < n; ++i)
    if (p[i]) return false;
  return true;
}
}  // namespace

// ===========================================================================
// The captured workload: two handle exports, an event export, the entry-point
// lookups plain and _spt, and a second process that exports one allocation of
// its own and opens one handle of each kind. Capture records only a call that
// succeeded, and a process cannot open its own handle, so only another process
// reaches the by-value open shims. What each call returned goes to stdout.
// ===========================================================================
TEST_CASE("Unit_HRR_CaptureCapabilities_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  void* mem[2] = {nullptr, nullptr};
  for (auto& m : mem) HRR_HIP_CHECK(hipMalloc(&m, 1 << 20));
  std::string first;
  for (void* m : mem) {
    hipIpcMemHandle_t h{};
    if (hipIpcGetMemHandle(&h, m) != hipSuccess) continue;
    printf(HRR_CAPS_IPC " %s\n", to_hex(&h, kHandleBytes).c_str());
    if (first.empty()) first = to_hex(&h, kHandleBytes);
  }
  hipEvent_t ev = nullptr;
  HRR_HIP_CHECK(hipEventCreateWithFlags(&ev, hipEventDisableTiming | hipEventInterprocess));
  hipIpcEventHandle_t eh{};
  HRR_HIP_CHECK(hipIpcGetEventHandle(&eh, ev));

  const int version = HIP_VERSION_MAJOR * 100 + HIP_VERSION_MINOR;
  hipDriverProcAddressQueryResult proc_status{};
  void* proc = nullptr;
  HRR_HIP_CHECK(hipGetProcAddress("hipMalloc", &proc, version, hipEnableDefault, &proc_status));
  print_addr(HRR_CAPS_PROC, proc);
  // The _spt forms are what a per-thread-default-stream build reaches; they
  // are declared whatever the build, so they are called by name here.
  void* proc_spt = nullptr;
  HRR_HIP_CHECK(hipGetProcAddress_spt("hipMalloc", &proc_spt, version, hipEnableDefault,
                                      &proc_status));
  print_addr(HRR_CAPS_PROC_SPT, proc_spt);
  hipDriverEntryPointQueryResult entry_status = hipDriverEntryPointSuccess;
  void* entry = nullptr;
  if (hipGetDriverEntryPoint("hipMalloc", &entry, 0, &entry_status) == hipSuccess)
    print_addr(HRR_CAPS_ENTRY, entry);
  void* entry_spt = nullptr;
  if (hipGetDriverEntryPoint_spt("hipMalloc", &entry_spt, 0, &entry_status) == hipSuccess)
    print_addr(HRR_CAPS_ENTRY_SPT, entry_spt);
  fflush(stdout);

  // The importer runs while this process waits, which keeps the handles alive.
  if (!first.empty()) {
    hrr::test::SpawnProc proc_imp(HRR_TEST_EXE, /*capture_stdout=*/true);
    proc_imp.setEnv(HRR_CAPS_EXPORT_MEM, first);
    proc_imp.setEnv(HRR_CAPS_EXPORT_EVENT, to_hex(&eh, kEventHandleBytes));
    const int ret = proc_imp.run("\"Unit_HRR_CaptureCapabilitiesImport_Direct\"");
    printf("%s" HRR_CAPS_IMPORTER " %d\n", proc_imp.getOutput().c_str(), ret);
    fflush(stdout);
    // An open the machine refuses still exits 0; anything else is a failure.
    REQUIRE(ret == 0);
  }

  HRR_HIP_CHECK(hipEventDestroy(ev));
  for (void* m : mem) HRR_HIP_CHECK(hipFree(m));
}

TEST_CASE("Unit_HRR_CaptureCapabilitiesImport_Direct", "[.][hrr-direct]") {
  const char* mem_hex = getenv(HRR_CAPS_EXPORT_MEM);
  const char* event_hex = getenv(HRR_CAPS_EXPORT_EVENT);
  if (!mem_hex || !event_hex) HRR_SKIP("started by Unit_HRR_CaptureCapabilities_Direct");
  const std::vector<uint8_t> mem_bytes = parse_hex(mem_hex);
  const std::vector<uint8_t> event_bytes = parse_hex(event_hex);
  REQUIRE(mem_bytes.size() == kHandleBytes);
  REQUIRE(event_bytes.size() == kEventHandleBytes);
  hipIpcMemHandle_t mh{};
  hipIpcEventHandle_t eh{};
  std::memcpy(&mh, mem_bytes.data(), kHandleBytes);
  std::memcpy(&eh, event_bytes.data(), kEventHandleBytes);

  HRR_HIP_CHECK(hipSetDevice(0));
  // An export of its own, so its archive holds an export and an import.
  void* own = nullptr;
  HRR_HIP_CHECK(hipMalloc(&own, 1 << 20));
  hipIpcMemHandle_t own_h{};
  HRR_HIP_CHECK(hipIpcGetMemHandle(&own_h, own));
  void* mem = nullptr;
  const hipError_t rm = hipIpcOpenMemHandle(&mem, mh, hipIpcMemLazyEnablePeerAccess);
  printf(HRR_CAPS_OPEN_MEM " %d\n", static_cast<int>(rm));
  hipEvent_t ev = nullptr;
  const hipError_t re = hipIpcOpenEventHandle(&ev, eh);
  printf(HRR_CAPS_OPEN_EVENT " %d\n", static_cast<int>(re));
  fflush(stdout);
  if (re == hipSuccess) HRR_HIP_CHECK(hipEventDestroy(ev));
  if (rm == hipSuccess) HRR_HIP_CHECK(hipIpcCloseMemHandle(mem));
  HRR_HIP_CHECK(hipFree(own));
}

// ---------------------------------------------------------------------------
// One capture of both processes, then each archive record by record, then a
// replay of the importer's.
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

  // What the workload saw. The exports are in call order, and only those that
  // succeeded, as in the archive.
  std::vector<std::vector<uint8_t>> handles;
  std::map<std::string, std::string> tags;
  { std::istringstream lines(out);
    std::string line;
    while (std::getline(lines, line)) {
      std::istringstream ls(line);
      std::string tag, value;
      ls >> tag >> value;
      if (tag == HRR_CAPS_IPC) {
        handles.push_back(parse_hex(value));
        REQUIRE(handles.back().size() == kHandleBytes);
        // Something to leak, or the checks below prove nothing.
        REQUIRE_FALSE(all_zero(handles.back().data(), kHandleBytes));
      } else if (tag.rfind("HRR_CAPS_", 0) == 0) {
        tags[tag] = value;
      }
    } }
  REQUIRE(tags.count(HRR_CAPS_PROC));
  REQUIRE(tags.count(HRR_CAPS_PROC_SPT));
#ifndef _WIN32
  REQUIRE(handles.size() == 2);
#endif

  std::vector<hrr_args_hipIpcGetMemHandle> exports;
  std::vector<hrr_args_hipIpcOpenMemHandle> mem_opens;
  std::vector<hrr_args_hipIpcOpenEventHandle> event_opens;
  std::map<uint16_t, uint64_t> addr_fields;  // event type -> recorded pfn/funcPtr
  fs::path importer;
  for (const auto& dir : hrr_process_archives(cap.path)) {
    hrr::Archive arc;
    REQUIRE(hrr::load_archive(dir.string(), arc));
    for (const auto& ev : arc.events) {
      const uint16_t t = ev.header().event_type;
      const uint8_t* pl = ev.raw_payload.data();
      const size_t n = ev.raw_payload.size();
      if (t == HRR_API_HIPIPCGETMEMHANDLE) {
        REQUIRE(n >= sizeof(hrr_args_hipIpcGetMemHandle));
        exports.push_back(*reinterpret_cast<const hrr_args_hipIpcGetMemHandle*>(pl));
      } else if (t == HRR_API_HIPIPCOPENMEMHANDLE) {
        REQUIRE(n >= sizeof(hrr_args_hipIpcOpenMemHandle));
        mem_opens.push_back(*reinterpret_cast<const hrr_args_hipIpcOpenMemHandle*>(pl));
        importer = dir;
      } else if (t == HRR_API_HIPIPCOPENEVENTHANDLE) {
        REQUIRE(n >= sizeof(hrr_args_hipIpcOpenEventHandle));
        event_opens.push_back(*reinterpret_cast<const hrr_args_hipIpcOpenEventHandle*>(pl));
      } else if (t == HRR_API_HIPGETPROCADDRESS) {
        REQUIRE(n >= sizeof(hrr_args_hipGetProcAddress));
        const auto* a = reinterpret_cast<const hrr_args_hipGetProcAddress*>(pl);
        CHECK(std::string(reinterpret_cast<const char*>(a->symbol_bytes)) == "hipMalloc");
        addr_fields[t] = a->pfn;
      } else if (t == HRR_API_HIPGETPROCADDRESS_SPT) {
        REQUIRE(n >= sizeof(hrr_args_hipGetProcAddress_spt));
        const auto* a = reinterpret_cast<const hrr_args_hipGetProcAddress_spt*>(pl);
        CHECK(std::string(reinterpret_cast<const char*>(a->symbol_bytes)) == "hipMalloc");
        addr_fields[t] = a->pfn;
      } else if (t == HRR_API_HIPGETDRIVERENTRYPOINT) {
        REQUIRE(n >= sizeof(hrr_args_hipGetDriverEntryPoint));
        addr_fields[t] = reinterpret_cast<const hrr_args_hipGetDriverEntryPoint*>(pl)->funcPtr;
      } else if (t == HRR_API_HIPGETDRIVERENTRYPOINT_SPT) {
        REQUIRE(n >= sizeof(hrr_args_hipGetDriverEntryPoint_spt));
        addr_fields[t] =
            reinterpret_cast<const hrr_args_hipGetDriverEntryPoint_spt*>(pl)->funcPtr;
      }
    }
  }

  // Each export is recorded, and its handle is not; the importer adds one.
  const bool imported = tags[HRR_CAPS_OPEN_MEM] == "0" && tags[HRR_CAPS_OPEN_EVENT] == "0";
  REQUIRE(exports.size() == handles.size() + (tags.count(HRR_CAPS_IMPORTER) ? 1 : 0));
  for (const auto& a : exports) {
    CHECK(a.handle_present == 1);
    CHECK(all_zero(a.handle_bytes, sizeof(a.handle_bytes)));
  }

  // The lookups keep their name and lose their answer.
  REQUIRE(addr_fields.count(HRR_API_HIPGETPROCADDRESS));
  REQUIRE(addr_fields.count(HRR_API_HIPGETPROCADDRESS_SPT));
  CHECK(addr_fields.count(HRR_API_HIPGETDRIVERENTRYPOINT) == tags.count(HRR_CAPS_ENTRY));
  CHECK(addr_fields.count(HRR_API_HIPGETDRIVERENTRYPOINT_SPT) == tags.count(HRR_CAPS_ENTRY_SPT));
  for (const auto& [type, value] : addr_fields) {
    INFO("event type " << type);
    CHECK(value == 0);
  }

  if (tags.count(HRR_CAPS_IMPORTER)) REQUIRE(tags[HRR_CAPS_IMPORTER] == "0");
  if (!imported) {
    WARN("this machine cannot open an IPC handle from another process; the open "
         "records and their replay are not checked");
    return;
  }

  // The importer's opens, by value, are recorded without the handle.
  REQUIRE(mem_opens.size() == 1);
  REQUIRE(event_opens.size() == 1);
  CHECK(all_zero(mem_opens[0].handle_bytes, sizeof(mem_opens[0].handle_bytes)));
  CHECK(all_zero(event_opens[0].handle_bytes, sizeof(event_opens[0].handle_bytes)));

  // Its export and its import both hold zeros, and replay must not pair them:
  // the import is reported as unmatched instead of opening the local export.
  // The event open cannot be replayed from zeros either; keep going past it.
  hrr::test::SpawnProc replay(HRR_PLAYBACK_EXE, /*capture_stdout=*/true,
                              /*capture_stderr=*/true);
  set_proc_search_path(replay);
  (void)replay.run("--continue-on-error " + hrr_quote_path(importer));
  const std::string log = replay.getOutput();
  INFO("Playback output:\n" << log);
  CHECK(log.find("hipIpcOpenMemHandle: no replayed export matches") != std::string::npos);
  CHECK(log.find("(hipIpcOpenMemHandle) returned") == std::string::npos);
}

#endif  // HRR_PLAYBACK_EXE && HRR_TEST_EXE

/**
 * @}
 */
