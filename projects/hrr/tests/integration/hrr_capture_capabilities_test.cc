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
 * do not, which is all replay needs to pair an import with its export. A handle
 * has little entropy, so the digest is keyed with a secret each capturing
 * process draws and never writes down: nobody can hash candidate handles and
 * compare. A resolved entry point from hipGetProcAddress or
 * hipGetDriverEntryPoint is an address in the runtime as the capturing process
 * loaded it, and replay looks the name up again, so it is not recorded at all.
 */

#include "hrr_test_common.hh"
#include "hrr_test_process.hh"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#if defined(HRR_PLAYBACK_EXE) && defined(HRR_TEST_EXE)

namespace {
#define HRR_CAPS_IPC "HRR_CAPS_IPC"
#define HRR_CAPS_PROC "HRR_CAPS_PROC"
#define HRR_CAPS_ENTRY "HRR_CAPS_ENTRY"
#define HRR_CAPS_PROC_SPT "HRR_CAPS_PROC_SPT"
#define HRR_CAPS_ENTRY_SPT "HRR_CAPS_ENTRY_SPT"
#define HRR_CAPS_EXPORT_MEM "HRR_CAPS_EXPORT_MEM"
#define HRR_CAPS_EXPORT_EVENT "HRR_CAPS_EXPORT_EVENT"
#define HRR_CAPS_OPEN_MEM "HRR_CAPS_OPEN_MEM"
#define HRR_CAPS_OPEN_EVENT "HRR_CAPS_OPEN_EVENT"
#define HRR_CAPS_IMPORTER "HRR_CAPS_IMPORTER"

constexpr size_t kHandleBytes = sizeof(hipIpcMemHandle_t);
constexpr size_t kEventHandleBytes = sizeof(hipIpcEventHandle_t);
constexpr size_t kDigestBytes = 16;

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

void print_handle(int idx, const hipIpcMemHandle_t& h) {
  printf(HRR_CAPS_IPC " %d %s\n", idx, to_hex(&h, kHandleBytes).c_str());
}

void print_addr(const char* tag, void* p) {
  printf("%s 0x%llx\n", tag,
         static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(p)));
}
}  // namespace

// ===========================================================================
// The captured workload: three handle exports, two of them of one allocation,
// and the entry-point lookups, plain and _spt. What each call returned goes to
// stdout, so the parent can look for it in the archive.
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

  const int version = HIP_VERSION_MAJOR * 100 + HIP_VERSION_MINOR;
  void* proc = nullptr;
  hipDriverProcAddressQueryResult proc_status{};
  HRR_HIP_CHECK(hipGetProcAddress("hipMalloc", &proc, version, hipEnableDefault,
                                  &proc_status));
  print_addr(HRR_CAPS_PROC, proc);
  // The _spt forms are what a per-thread-default-stream build reaches; they
  // are declared whatever the build, so they are called by name here.
  void* proc_spt = nullptr;
  HRR_HIP_CHECK(hipGetProcAddress_spt("hipMalloc", &proc_spt, version,
                                      hipEnableDefault, &proc_status));
  print_addr(HRR_CAPS_PROC_SPT, proc_spt);

  void* entry = nullptr;
  hipDriverEntryPointQueryResult entry_status = hipDriverEntryPointSuccess;
  if (hipGetDriverEntryPoint("hipMalloc", &entry, 0, &entry_status) == hipSuccess)
    print_addr(HRR_CAPS_ENTRY, entry);
  void* entry_spt = nullptr;
  if (hipGetDriverEntryPoint_spt("hipMalloc", &entry_spt, 0, &entry_status) == hipSuccess)
    print_addr(HRR_CAPS_ENTRY_SPT, entry_spt);
  fflush(stdout);

  HRR_HIP_CHECK(hipFree(b));
  HRR_HIP_CHECK(hipFree(a));
}

// ===========================================================================
// The two halves of an import. Capture records only a call that succeeded, and
// a process cannot open its own handle, so the importer is another process:
// the exporter starts it and waits for it, which keeps the handles alive while
// it opens them. Both are captured into the same root, one archive each.
// ===========================================================================
TEST_CASE("Unit_HRR_CaptureCapabilitiesExport_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  void* mem = nullptr;
  HRR_HIP_CHECK(hipMalloc(&mem, 1 << 20));
  hipIpcMemHandle_t mh{};
  HRR_HIP_CHECK(hipIpcGetMemHandle(&mh, mem));
  hipEvent_t ev = nullptr;
  HRR_HIP_CHECK(hipEventCreateWithFlags(&ev, hipEventDisableTiming | hipEventInterprocess));
  hipIpcEventHandle_t eh{};
  HRR_HIP_CHECK(hipIpcGetEventHandle(&eh, ev));
  printf(HRR_CAPS_EXPORT_MEM " %s\n", to_hex(&mh, kHandleBytes).c_str());
  printf(HRR_CAPS_EXPORT_EVENT " %s\n", to_hex(&eh, kEventHandleBytes).c_str());
  fflush(stdout);

  int ret = -1;
  std::string out;
  { hrr::test::SpawnProc proc(HRR_TEST_EXE, /*capture_stdout=*/true);
    proc.setEnv(HRR_CAPS_EXPORT_MEM, to_hex(&mh, kHandleBytes));
    proc.setEnv(HRR_CAPS_EXPORT_EVENT, to_hex(&eh, kEventHandleBytes));
    ret = proc.run("\"Unit_HRR_CaptureCapabilitiesImport_Direct\"");
    out = proc.getOutput(); }
  printf("%s" HRR_CAPS_IMPORTER " %d\n", out.c_str(), ret);
  fflush(stdout);

  HRR_HIP_CHECK(hipEventDestroy(ev));
  HRR_HIP_CHECK(hipFree(mem));
}

TEST_CASE("Unit_HRR_CaptureCapabilitiesImport_Direct", "[.][hrr-direct]") {
  const char* mem_hex = getenv(HRR_CAPS_EXPORT_MEM);
  const char* event_hex = getenv(HRR_CAPS_EXPORT_EVENT);
  if (!mem_hex || !event_hex) HRR_SKIP("started by Unit_HRR_CaptureCapabilitiesExport_Direct");
  const std::vector<uint8_t> mem_bytes = parse_hex(mem_hex);
  const std::vector<uint8_t> event_bytes = parse_hex(event_hex);
  REQUIRE(mem_bytes.size() == kHandleBytes);
  REQUIRE(event_bytes.size() == kEventHandleBytes);
  hipIpcMemHandle_t mh{};
  hipIpcEventHandle_t eh{};
  std::memcpy(&mh, mem_bytes.data(), kHandleBytes);
  std::memcpy(&eh, event_bytes.data(), kEventHandleBytes);

  HRR_HIP_CHECK(hipSetDevice(0));
  void* mem = nullptr;
  const hipError_t rm = hipIpcOpenMemHandle(&mem, mh, hipIpcMemLazyEnablePeerAccess);
  printf(HRR_CAPS_OPEN_MEM " %d\n", static_cast<int>(rm));
  hipEvent_t ev = nullptr;
  const hipError_t re = hipIpcOpenEventHandle(&ev, eh);
  printf(HRR_CAPS_OPEN_EVENT " %d\n", static_cast<int>(re));
  fflush(stdout);
  if (re == hipSuccess) HRR_HIP_CHECK(hipEventDestroy(ev));
  if (rm == hipSuccess) HRR_HIP_CHECK(hipIpcCloseMemHandle(mem));
}

namespace {
bool all_zero(const uint8_t* p, size_t n) {
  for (size_t i = 0; i < n; ++i)
    if (p[i]) return false;
  return true;
}

using Digest = std::array<uint8_t, kDigestBytes>;

Digest digest_of(const uint8_t* bytes) {
  Digest d;
  std::memcpy(d.data(), bytes, kDigestBytes);
  return d;
}

Digest from_words(uint64_t lo, uint64_t hi) {
  Digest d;
  std::memcpy(d.data(), &lo, sizeof(lo));
  std::memcpy(d.data() + sizeof(lo), &hi, sizeof(hi));
  return d;
}

// The unkeyed FNV-1a pair the archive's blob names use: anyone holding the
// archive can compute it, so a recorded digest must not be it.
Digest public_digest(const std::vector<uint8_t>& v) {
  uint64_t h1 = 0xcbf29ce484222325ULL, h2 = 0x100000001b3ULL;
  for (uint8_t c : v) {
    h1 ^= c; h1 *= 0x100000001b3ULL;
    h2 ^= c; h2 *= 0xcbf29ce484222325ULL;
  }
  return from_words(h1, h2);
}

// SipHash-2-4 with 128-bit output, the keyed digest capture uses.
uint64_t rotl(uint64_t x, int b) { return (x << b) | (x >> (64 - b)); }

Digest siphash128(const uint8_t key[16], const std::vector<uint8_t>& msg) {
  uint64_t k0 = 0, k1 = 0;
  for (int i = 0; i < 8; ++i) {
    k0 |= uint64_t{key[i]} << (8 * i);
    k1 |= uint64_t{key[8 + i]} << (8 * i);
  }
  uint64_t v0 = k0 ^ 0x736f6d6570736575ULL, v1 = k1 ^ 0x646f72616e646f6dULL ^ 0xee;
  uint64_t v2 = k0 ^ 0x6c7967656e657261ULL, v3 = k1 ^ 0x7465646279746573ULL;
  auto round = [&] {
    v0 += v1; v1 = rotl(v1, 13); v1 ^= v0; v0 = rotl(v0, 32);
    v2 += v3; v3 = rotl(v3, 16); v3 ^= v2;
    v0 += v3; v3 = rotl(v3, 21); v3 ^= v0;
    v2 += v1; v1 = rotl(v1, 17); v1 ^= v2; v2 = rotl(v2, 32);
  };
  auto absorb = [&](uint64_t m) { v3 ^= m; round(); round(); v0 ^= m; };
  const size_t whole = msg.size() & ~size_t{7};
  for (size_t i = 0; i < whole; i += 8) {
    uint64_t m = 0;
    for (int j = 0; j < 8; ++j) m |= uint64_t{msg[i + j]} << (8 * j);
    absorb(m);
  }
  uint64_t last = uint64_t{msg.size()} << 56;
  for (size_t j = 0; whole + j < msg.size(); ++j) last |= uint64_t{msg[whole + j]} << (8 * j);
  absorb(last);
  v2 ^= 0xee;
  for (int r = 0; r < 4; ++r) round();
  const uint64_t lo = v0 ^ v1 ^ v2 ^ v3;
  v1 ^= 0xdd;
  for (int r = 0; r < 4; ++r) round();
  return from_words(lo, v0 ^ v1 ^ v2 ^ v3);
}

struct Recorded {
  std::vector<uint8_t> handle;
  Digest digest;
};

int hex_value(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

// The file under `root` that holds the key of one of the recorded digests,
// as 16 raw bytes or as 32 hex digits, or an empty string. Every 16 bytes of
// every file are tried as a key.
std::string file_holding_key(const fs::path& root, const std::vector<Recorded>& recs,
                             size_t* scanned) {
  auto keys = [&](const uint8_t key[16]) {
    for (const auto& r : recs)
      if (siphash128(key, r.handle) == r.digest) return true;
    return false;
  };
  for (const auto& ent : fs::recursive_directory_iterator(root)) {
    if (!ent.is_regular_file()) continue;
    std::ifstream in(ent.path(), std::ios::binary);
    const std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const auto* p = reinterpret_cast<const uint8_t*>(s.data());
    *scanned += s.size();
    size_t hex_run = 0;
    for (size_t i = 0; i < s.size(); ++i) {
      if (i + 16 <= s.size() && keys(p + i)) return ent.path().string();
      hex_run = hex_value(s[i]) >= 0 ? hex_run + 1 : 0;
      if (hex_run >= 32) {
        uint8_t k[16];
        const char* h = s.data() + i + 1 - 32;
        for (int j = 0; j < 16; ++j) k[j] = uint8_t(hex_value(h[2 * j]) << 4 | hex_value(h[2 * j + 1]));
        if (keys(k)) return ent.path().string();
      }
    }
  }
  return {};
}

std::map<std::string, std::string> tagged_lines(const std::string& out,
                                                std::vector<std::string>* ipc = nullptr) {
  std::map<std::string, std::string> tags;
  std::istringstream lines(out);
  std::string line;
  while (std::getline(lines, line)) {
    std::istringstream ls(line);
    std::string tag, value;
    ls >> tag;
    if (tag == HRR_CAPS_IPC && ipc) {
      int idx = -1;
      ls >> idx >> value;
      ipc->push_back(value);
    } else if (tag.rfind("HRR_CAPS_", 0) == 0) {
      ls >> value;
      tags[tag] = value;
    }
  }
  return tags;
}

unsigned long long addr(const std::map<std::string, std::string>& tags, const char* tag) {
  auto it = tags.find(tag);
  return it == tags.end() ? 0 : std::stoull(it->second, nullptr, 16);
}

// What every recorded handle field must look like: a digest that is neither
// the handle nor the public hash of it, then zeros.
void check_digest_layout(const uint8_t* field, const std::vector<uint8_t>& handle) {
  REQUIRE(handle.size() <= 64);
  CHECK(std::memcmp(field, handle.data(), handle.size()) != 0);
  CHECK(all_zero(field + kDigestBytes, 64 - kDigestBytes));
  CHECK_FALSE(all_zero(field, kDigestBytes));
  CHECK(digest_of(field) != public_digest(handle));
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

  // What the workload saw. The exports are in call order, and only those that
  // succeeded, as in the archive.
  std::vector<std::string> ipc_hex;
  const auto tags = tagged_lines(out, &ipc_hex);
  std::vector<std::vector<uint8_t>> handles;
  for (const auto& hex : ipc_hex) {
    handles.push_back(parse_hex(hex));
    REQUIRE(handles.back().size() == kHandleBytes);
  }
  // Something to leak, or the checks below prove nothing.
  REQUIRE(addr(tags, HRR_CAPS_PROC) != 0);
  REQUIRE(addr(tags, HRR_CAPS_PROC_SPT) != 0);
#ifndef _WIN32
  REQUIRE(handles.size() == 3);
#endif

  hrr::Archive arc;
  REQUIRE(hrr::load_archive(hrr_single_process_archive(cap.path).string(), arc));

  std::vector<const hrr_args_hipIpcGetMemHandle*> ipc;
  const hrr_args_hipGetProcAddress* lookup = nullptr;
  const hrr_args_hipGetProcAddress_spt* lookup_spt = nullptr;
  const hrr_args_hipGetDriverEntryPoint* entry_ev = nullptr;
  const hrr_args_hipGetDriverEntryPoint_spt* entry_spt_ev = nullptr;
  for (const auto& ev : arc.events) {
    const uint16_t t = ev.header().event_type;
    if (t == HRR_API_HIPIPCGETMEMHANDLE) {
      REQUIRE(ev.raw_payload.size() >= sizeof(hrr_args_hipIpcGetMemHandle));
      ipc.push_back(reinterpret_cast<const hrr_args_hipIpcGetMemHandle*>(ev.raw_payload.data()));
    } else if (t == HRR_API_HIPGETPROCADDRESS) {
      REQUIRE(ev.raw_payload.size() >= sizeof(hrr_args_hipGetProcAddress));
      lookup = reinterpret_cast<const hrr_args_hipGetProcAddress*>(ev.raw_payload.data());
    } else if (t == HRR_API_HIPGETPROCADDRESS_SPT) {
      REQUIRE(ev.raw_payload.size() >= sizeof(hrr_args_hipGetProcAddress_spt));
      lookup_spt = reinterpret_cast<const hrr_args_hipGetProcAddress_spt*>(ev.raw_payload.data());
    } else if (t == HRR_API_HIPGETDRIVERENTRYPOINT) {
      REQUIRE(ev.raw_payload.size() >= sizeof(hrr_args_hipGetDriverEntryPoint));
      entry_ev = reinterpret_cast<const hrr_args_hipGetDriverEntryPoint*>(ev.raw_payload.data());
    } else if (t == HRR_API_HIPGETDRIVERENTRYPOINT_SPT) {
      REQUIRE(ev.raw_payload.size() >= sizeof(hrr_args_hipGetDriverEntryPoint_spt));
      entry_spt_ev =
          reinterpret_cast<const hrr_args_hipGetDriverEntryPoint_spt*>(ev.raw_payload.data());
    }
  }

  // Each export is recorded, as a keyed digest and zeros rather than the
  // handle, and the digests compare the way the handles did.
  REQUIRE(ipc.size() == handles.size());
  std::vector<Recorded> recorded;
  for (size_t i = 0; i < ipc.size(); ++i) {
    INFO("export " << i);
    CHECK(ipc[i]->handle_present == 1);
    check_digest_layout(ipc[i]->handle_bytes, handles[i]);
    recorded.push_back({handles[i], digest_of(ipc[i]->handle_bytes)});
  }
  for (size_t i = 0; i < ipc.size(); ++i)
    for (size_t j = i + 1; j < ipc.size(); ++j) {
      INFO("exports " << i << " and " << j);
      const bool same_handle = handles[i] == handles[j];
      const bool same_digest =
          std::memcmp(ipc[i]->handle_bytes, ipc[j]->handle_bytes, kDigestBytes) == 0;
      CHECK(same_handle == same_digest);
    }
#ifndef _WIN32
  // The same allocation exported twice is the case replay pairs on.
  CHECK(handles[0] == handles[1]);
  CHECK(handles[0] != handles[2]);
#endif

  // The key is nowhere in the archive.
  size_t scanned = 0;
  CHECK(file_holding_key(cap.path, recorded, &scanned) == "");
  CHECK(scanned > 0);

  // The lookups keep their name and flags and lose their answer.
  REQUIRE(lookup != nullptr);
  CHECK(lookup->pfn == 0);
  CHECK(lookup->symbol_present == 1);
  CHECK(std::string(reinterpret_cast<const char*>(lookup->symbol_bytes)) == "hipMalloc");
  REQUIRE(lookup_spt != nullptr);
  CHECK(lookup_spt->pfn == 0);
  CHECK(lookup_spt->symbol_present == 1);
  CHECK(std::string(reinterpret_cast<const char*>(lookup_spt->symbol_bytes)) == "hipMalloc");
  if (tags.count(HRR_CAPS_ENTRY)) {
    REQUIRE(entry_ev != nullptr);
    CHECK(addr(tags, HRR_CAPS_ENTRY) != 0);
    CHECK(entry_ev->funcPtr == 0);
  }
  if (tags.count(HRR_CAPS_ENTRY_SPT)) {
    REQUIRE(entry_spt_ev != nullptr);
    CHECK(addr(tags, HRR_CAPS_ENTRY_SPT) != 0);
    CHECK(entry_spt_ev->funcPtr == 0);
  }

  // Replay re-exports and looks the name up again from what is left.
  hrr::test::SpawnProc replay(HRR_PLAYBACK_EXE, /*capture_stdout=*/true);
  set_proc_search_path(replay);
  const int ret = replay.run(hrr_quote_path(cap.path));
  INFO("Playback stdout:\n" << replay.getOutput());
  CHECK(ret == 0);
}

// ---------------------------------------------------------------------------
// An export in one process and its import in another, both captured. The
// importer's two by-value opens record the digest layout, and its digest of
// the memory handle differs from the exporter's: each process keys its own.
// ---------------------------------------------------------------------------
HRR_TEST_CASE(Unit_HRR_CaptureCapabilitiesAcrossProcesses) {
  ScopedDir cap(fs::temp_directory_path() / "hrr_capture_capabilities_ipc.hrr");
  std::string out;
  { hrr::test::SpawnProc proc(HRR_TEST_EXE, /*capture_stdout=*/true);
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    set_proc_search_path(proc);
    int ret = proc.run("\"Unit_HRR_CaptureCapabilitiesExport_Direct\"");
    out = proc.getOutput();
    INFO("Capture exit: " << ret << "\n" << out);
    REQUIRE(ret == 0); }
  INFO("Capture output:\n" << out);

  const auto tags = tagged_lines(out);
  REQUIRE(tags.count(HRR_CAPS_IMPORTER));
  REQUIRE(tags.at(HRR_CAPS_IMPORTER) == "0");
  const std::vector<uint8_t> mem = parse_hex(tags.at(HRR_CAPS_EXPORT_MEM));
  const std::vector<uint8_t> event = parse_hex(tags.at(HRR_CAPS_EXPORT_EVENT));
  REQUIRE(mem.size() == kHandleBytes);
  REQUIRE(event.size() == kEventHandleBytes);
  REQUIRE(tags.count(HRR_CAPS_OPEN_MEM));
  REQUIRE(tags.count(HRR_CAPS_OPEN_EVENT));
  // Capture records only an open that succeeded.
  if (tags.at(HRR_CAPS_OPEN_MEM) != "0" || tags.at(HRR_CAPS_OPEN_EVENT) != "0")
    HRR_SKIP("this machine cannot open an IPC handle from another process");

  const hrr_args_hipIpcGetMemHandle* exported = nullptr;
  const hrr_args_hipIpcOpenMemHandle* opened_mem = nullptr;
  const hrr_args_hipIpcOpenEventHandle* opened_event = nullptr;
  std::vector<hrr::Archive> arcs(2);
  const auto dirs = hrr_process_archives(cap.path);
  REQUIRE(dirs.size() == 2);
  for (size_t i = 0; i < dirs.size(); ++i) {
    REQUIRE(hrr::load_archive(dirs[i].string(), arcs[i]));
    for (const auto& ev : arcs[i].events) {
      const uint16_t t = ev.header().event_type;
      const uint8_t* pl = ev.raw_payload.data();
      if (t == HRR_API_HIPIPCGETMEMHANDLE) {
        REQUIRE(ev.raw_payload.size() >= sizeof(hrr_args_hipIpcGetMemHandle));
        exported = reinterpret_cast<const hrr_args_hipIpcGetMemHandle*>(pl);
      } else if (t == HRR_API_HIPIPCOPENMEMHANDLE) {
        REQUIRE(ev.raw_payload.size() >= sizeof(hrr_args_hipIpcOpenMemHandle));
        opened_mem = reinterpret_cast<const hrr_args_hipIpcOpenMemHandle*>(pl);
      } else if (t == HRR_API_HIPIPCOPENEVENTHANDLE) {
        REQUIRE(ev.raw_payload.size() >= sizeof(hrr_args_hipIpcOpenEventHandle));
        opened_event = reinterpret_cast<const hrr_args_hipIpcOpenEventHandle*>(pl);
      }
    }
  }
  REQUIRE(exported != nullptr);
  REQUIRE(opened_mem != nullptr);
  REQUIRE(opened_event != nullptr);

  { INFO("hipIpcOpenMemHandle");
    check_digest_layout(opened_mem->handle_bytes, mem); }
  { INFO("hipIpcOpenEventHandle");
    check_digest_layout(opened_event->handle_bytes, event); }
  { INFO("hipIpcGetMemHandle");
    check_digest_layout(exported->handle_bytes, mem); }
  // One handle, two capturing processes, two keys.
  CHECK(digest_of(exported->handle_bytes) != digest_of(opened_mem->handle_bytes));

  size_t scanned = 0;
  CHECK(file_holding_key(cap.path,
                         {{mem, digest_of(exported->handle_bytes)},
                          {mem, digest_of(opened_mem->handle_bytes)},
                          {event, digest_of(opened_event->handle_bytes)}},
                         &scanned) == "");
  CHECK(scanned > 0);
}

#endif  // HRR_PLAYBACK_EXE && HRR_TEST_EXE

/**
 * @}
 */
