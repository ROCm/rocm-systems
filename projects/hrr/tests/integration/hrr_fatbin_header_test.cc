/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR fat binary header checks
 * @{
 * @ingroup HRRTest
 * Tests for what capture stores of a registered fat binary.
 *
 * Capture stores none of the bundle: HIP is handed a bare pointer, so the
 * bundle's extent is known only from its header, and HIP itself reads just the
 * header and the code object it picks for this GPU. Capture records that code
 * object, with the size HIP knows, and the register event without a blob.
 * A bundle whose header declares an entry far past the real image is captured
 * without reading there.
 *
 * Two paths record fat binaries and both are exercised:
 *   - registered before HIP initialises: swept from HIP's list in
 *     hip_capture_init, after HIP has loaded what it needed for globals;
 *   - registered after: seen by the __hipRegisterFatBinary shim, once HIP has
 *     digested the bundle.
 * The workload calls __hipRegisterFatBinary itself with bundles laid out in
 * memory the way clang-offload-bundler lays them out, valid and altered.
 */

#include "hrr_test_common.hh"
#include "hrr_test_process.hh"

#include <hip/hiprtc.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <sys/mman.h>
#endif

// Exported by libamdhip64 and called by the module constructor clang emits for
// every HIP translation unit; no header declares it.
extern "C" void** __hipRegisterFatBinary(const void* data);

#define HRR_HIPRTC_CHECK(expr)                                                 \
  do {                                                                         \
    hiprtcResult _hrr_rtc = (expr);                                            \
    INFO("hipRTC call failed: " #expr);                                        \
    INFO("hiprtcResult: " << static_cast<int>(_hrr_rtc));                      \
    INFO("hiprtcGetErrorString: " << hiprtcGetErrorString(_hrr_rtc));          \
    REQUIRE(_hrr_rtc == HIPRTC_SUCCESS);                                       \
  } while (0)

#if defined(HRR_TEST_EXE)

namespace {

constexpr char     kBundleMagic[]   = "__CLANG_OFFLOAD_BUNDLE__";
constexpr size_t   kBundleMagicLen  = sizeof(kBundleMagic) - 1;
constexpr size_t   kTableStart      = kBundleMagicLen + sizeof(uint64_t);
constexpr size_t   kEntryHeaderSize = 3 * sizeof(uint64_t);
// HIP hands COMGR only this much of the bundle to find the entry table in.
constexpr size_t   kTableLimit      = 4096;
// Where clang-offload-bundler puts the first code object.
constexpr size_t   kPayloadOffset   = 4096;
constexpr size_t   kMarkerPayload   = 64;
constexpr char     kHostId[]        = "host-x86_64-unknown-linux-gnu-";
constexpr char     kTargetPrefix[]  = "hipv4-amdgcn-amd-amdhsa--";
// A real target no runner has, so HIP finds no code object for its GPUs.
constexpr char     kForeignId[]     = "hipv4-amdgcn-amd-amdhsa--gfx600";
// The compressed header as HIP reads it (version 2): 24 bytes up to the data.
constexpr size_t   kCompressedHeaderSize = 24;
constexpr uint32_t kHipfMagic       = 0x48495046u;  // "HIPF"

struct BundleEntry {
  uint64_t    offset;
  uint64_t    size;
  std::string id;
};

void put_u64(std::vector<uint8_t>& b, size_t at, uint64_t v) {
  std::memcpy(b.data() + at, &v, sizeof(v));
}

// An uncompressed bundle of total_size bytes: magic, entry count and entry
// table at the front, zero elsewhere. Callers place the payloads.
std::vector<uint8_t> uncompressed_bundle(const std::vector<BundleEntry>& entries,
                                         size_t total_size) {
  std::vector<uint8_t> b(total_size, 0);
  std::memcpy(b.data(), kBundleMagic, kBundleMagicLen);
  put_u64(b, kBundleMagicLen, entries.size());
  size_t pos = kTableStart;
  for (const auto& e : entries) {
    put_u64(b, pos, e.offset);
    put_u64(b, pos + 8, e.size);
    put_u64(b, pos + 16, e.id.size());
    std::memcpy(b.data() + pos + kEntryHeaderSize, e.id.data(), e.id.size());
    pos += kEntryHeaderSize + e.id.size();
  }
  return b;
}

void put_marker(std::vector<uint8_t>& b, size_t at, const char* marker) {
  std::memcpy(b.data() + at, marker, std::strlen(marker));
}

// Well-formed, with a code object only for a GPU that is not here. HIP
// registers it lazily before it initialises and refuses it after.
std::vector<uint8_t> foreign_bundle(const char* marker) {
  auto b = uncompressed_bundle({{kPayloadOffset, 0, kHostId},
                                {kPayloadOffset, kMarkerPayload, kForeignId}},
                               kPayloadOffset + kMarkerPayload);
  put_marker(b, kPayloadOffset, marker);
  return b;
}

// A compressed header whose totalSize does not even cover the header. The
// hash field carries the tag, so the bytes a capture stored can be told apart.
std::vector<uint8_t> short_compressed_bundle(uint64_t tag) {
  std::vector<uint8_t> b(kCompressedHeaderSize, 0);
  std::memcpy(b.data(), "CCOB", 4);
  const uint16_t version = 2, method = 1;
  const uint32_t total_size = kCompressedHeaderSize - 1, uncompressed_size = 0;
  std::memcpy(b.data() + 4, &version, 2);
  std::memcpy(b.data() + 6, &method, 2);
  std::memcpy(b.data() + 8, &total_size, 4);
  std::memcpy(b.data() + 12, &uncompressed_size, 4);
  put_u64(b, 16, tag);
  return b;
}

// One entry whose identifier runs past the first 4096 bytes. The entry it
// describes is in range, so only the table bound refuses it.
std::vector<uint8_t> long_id_bundle(const char* marker) {
  std::string id(kTableLimit, '\0');
  std::memcpy(&id[0], kForeignId, sizeof(kForeignId) - 1);
  const size_t payload = kTableStart + kEntryHeaderSize + id.size();
  auto b = uncompressed_bundle({{payload, kMarkerPayload, id}}, payload + kMarkerPayload);
  put_marker(b, payload, marker);
  return b;
}

// A host entry whose offset plus size wraps around, beside a code object for
// target_id. HIP never loads the host entry and does not check it.
std::vector<uint8_t> host_overflow_bundle(const std::string& target_id,
                                          const std::vector<uint8_t>& payload) {
  auto b = uncompressed_bundle({{1, UINT64_MAX, kHostId},
                                {kPayloadOffset, payload.size(), target_id}},
                               kPayloadOffset + payload.size());
  std::memcpy(b.data() + kPayloadOffset, payload.data(), payload.size());
  return b;
}

// A host entry declaring a terabyte from the start of the bundle, beside a code
// object for target_id. Nothing past the code object is mapped, and HIP never
// loads the host entry.
std::vector<uint8_t> far_end_bundle(const std::string& target_id,
                                    const std::vector<uint8_t>& payload) {
  auto b = uncompressed_bundle({{0, uint64_t(1) << 40, kHostId},
                                {kPayloadOffset, payload.size(), target_id}},
                               kPayloadOffset + payload.size());
  std::memcpy(b.data() + kPayloadOffset, payload.data(), payload.size());
  return b;
}

// The code object for target_id, whose entry declares a terabyte. HIP picks
// that entry for this GPU and keeps the declared size: nothing bounds it.
std::vector<uint8_t> oversized_bundle(const std::string& target_id,
                                      const std::vector<uint8_t>& payload) {
  auto b = uncompressed_bundle({{kPayloadOffset, 0, kHostId},
                                {kPayloadOffset, uint64_t(1) << 40, target_id}},
                               kPayloadOffset + payload.size());
  std::memcpy(b.data() + kPayloadOffset, payload.data(), payload.size());
  return b;
}

// What clang emits for one target: an empty host entry, then the code object.
std::vector<uint8_t> valid_bundle(const std::string& target_id,
                                  const std::vector<uint8_t>& payload) {
  auto b = uncompressed_bundle({{kPayloadOffset, 0, kHostId},
                                {kPayloadOffset, payload.size(), target_id}},
                               kPayloadOffset + payload.size());
  std::memcpy(b.data() + kPayloadOffset, payload.data(), payload.size());
  return b;
}

std::vector<uint8_t> marker_payload(const char* marker) {
  std::vector<uint8_t> p(kMarkerPayload, 0);
  std::memcpy(p.data(), marker, std::strlen(marker));
  return p;
}

// The bundles registered before HIP initialises; the parent rebuilds the same
// bytes to recognise what the capture stored.
constexpr char     kPreInitValidMarker[]    = "HRR fatbin pre-init valid";
constexpr char     kPreInitLongIdMarker[]   = "HRR fatbin pre-init long id";
constexpr char     kPreInitOverflowMarker[] = "HRR fatbin pre-init overflow";
constexpr char     kPreInitFarEndMarker[]   = "HRR fatbin pre-init far end";
constexpr uint64_t kPreInitCompressedTag    = 0x4852524650524531ull;
// And the ones after: three HIP loads, told apart by their kernel, two it refuses.
constexpr char     kLiveKernel[]            = "hrr_fatbin_live_probe";
constexpr char     kFarEndKernel[]          = "hrr_fatbin_far_end_probe";
constexpr char     kOversizedKernel[]       = "hrr_fatbin_oversized_probe";
// Registered with device 1 current, to replay on that device.
constexpr char     kDevice1Kernel[]         = "hrr_fatbin_device1_probe";
constexpr char     kLiveForeignMarker[]     = "HRR fatbin live foreign";
constexpr uint64_t kLiveCompressedTag       = 0x485252464C495631ull;

std::vector<uint8_t> preinit_valid()      { return foreign_bundle(kPreInitValidMarker); }
std::vector<uint8_t> preinit_compressed() { return short_compressed_bundle(kPreInitCompressedTag); }
std::vector<uint8_t> preinit_long_id()    { return long_id_bundle(kPreInitLongIdMarker); }
std::vector<uint8_t> preinit_overflow() {
  return host_overflow_bundle(kForeignId, marker_payload(kPreInitOverflowMarker));
}
std::vector<uint8_t> preinit_far_end() {
  return far_end_bundle(kForeignId, marker_payload(kPreInitFarEndMarker));
}
std::vector<uint8_t> live_foreign()       { return foreign_bundle(kLiveForeignMarker); }
std::vector<uint8_t> live_compressed()    { return short_compressed_bundle(kLiveCompressedTag); }

// Copy a bundle to page-aligned memory that stays mapped for the life of the
// process, as a fat binary in .hip_fatbin does: HIP keeps the pointer, and
// capture reads it again when HIP initialises. HIP also hands COMGR the first
// 4096 bytes whatever the bundle's size, so at least that much is readable.
const void* pin(const std::vector<uint8_t>& bytes) {
  const size_t span = bytes.size() > 2 * kTableLimit ? bytes.size() : 2 * kTableLimit;
  auto* raw = new uint8_t[span + kTableLimit]();
  auto* p = reinterpret_cast<uint8_t*>(
      (reinterpret_cast<uintptr_t>(raw) + kTableLimit - 1) & ~uintptr_t(kTableLimit - 1));
  std::memcpy(p, bytes.data(), bytes.size());
  return p;
}

// Like pin(), but the bundle's last page is followed by one that cannot be
// read, so a read that runs past the bundle faults. Windows keeps pin().
const void* pin_before_guard(const std::vector<uint8_t>& bytes) {
#if defined(_WIN32)
  return pin(bytes);
#else
  constexpr size_t kGuard = 65536;  // a multiple of any page size in use
  const size_t span = (bytes.size() + kGuard - 1) / kGuard * kGuard;
  void* m = mmap(nullptr, span + kGuard, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  REQUIRE(m != MAP_FAILED);
  auto* p = static_cast<uint8_t*>(m);
  REQUIRE(mprotect(p + span, kGuard, PROT_NONE) == 0);
  std::memcpy(p, bytes.data(), bytes.size());
  return p;
#endif
}

void** register_fat_binary(const void* bundle) {
  // __CudaFatBinaryWrapper; HIP keeps this pointer too.
  struct Wrapper { uint32_t magic; uint32_t version; const void* binary; const void* dummy; };
  return __hipRegisterFatBinary(new Wrapper{kHipfMagic, 1, bundle, nullptr});
}

std::vector<uint8_t> compile_code_object(const std::string& kernel) {
  const std::string src =
      "extern \"C\" __global__ void " + kernel + "(int* out) { out[0] = 1; }\n";
  hiprtcProgram prog = nullptr;
  HRR_HIPRTC_CHECK(hiprtcCreateProgram(&prog, src.c_str(), "hrr_fatbin_probe.hip", 0,
                                       nullptr, nullptr));
  if (hiprtcCompileProgram(prog, 0, nullptr) != HIPRTC_SUCCESS) {
    size_t log_sz = 0;
    (void)hiprtcGetProgramLogSize(prog, &log_sz);
    std::string log(log_sz, '\0');
    (void)hiprtcGetProgramLog(prog, &log[0]);
    (void)hiprtcDestroyProgram(&prog);
    FAIL("hiprtcCompileProgram failed: " + log);
  }
  size_t co_size = 0;
  HRR_HIPRTC_CHECK(hiprtcGetCodeSize(prog, &co_size));
  std::vector<uint8_t> co(co_size);
  HRR_HIPRTC_CHECK(hiprtcGetCode(prog, reinterpret_cast<char*>(co.data())));
  HRR_HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
  return co;
}

}  // namespace

// ===========================================================================
// The captured workload.
// ===========================================================================
TEST_CASE("Unit_HRR_FatBinHeader_Direct", "[.][hrr-direct]") {
  // HIP is not initialised yet: it only notes these and returns a handle, and
  // capture sees them when it starts, by sweeping HIP's list.
  REQUIRE(register_fat_binary(pin(preinit_valid())) != nullptr);
  REQUIRE(register_fat_binary(pin(preinit_compressed())) != nullptr);
  REQUIRE(register_fat_binary(pin(preinit_long_id())) != nullptr);
  REQUIRE(register_fat_binary(pin(preinit_overflow())) != nullptr);
  REQUIRE(register_fat_binary(pin(preinit_far_end())) != nullptr);

  // Initialises HIP, and with it capture.
  HRR_HIP_CHECK(hipSetDevice(0));
  hipDeviceProp_t prop{};
  HRR_HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  const std::string target = std::string(kTargetPrefix) + prop.gcnArchName;
  const std::vector<uint8_t> live_co = compile_code_object(kLiveKernel);
  const std::vector<uint8_t> far_end_co = compile_code_object(kFarEndKernel);
  const std::vector<uint8_t> oversized_co = compile_code_object(kOversizedKernel);
  for (const auto* co : {&live_co, &far_end_co, &oversized_co}) {
    REQUIRE(co->size() > 4);
    REQUIRE(std::memcmp(co->data(), "\x7f" "ELF", 4) == 0);
  }

  // From here HIP digests each bundle inside the call. These three it loads.
  REQUIRE(register_fat_binary(pin(valid_bundle(target, live_co))) != nullptr);
  REQUIRE(register_fat_binary(pin(far_end_bundle(target, far_end_co))) != nullptr);
  REQUIRE(register_fat_binary(pin_before_guard(oversized_bundle(target, oversized_co))) !=
          nullptr);
  // And these it refuses: no code object for this GPU, a bad header.
  REQUIRE(register_fat_binary(pin(live_foreign())) == nullptr);
  REQUIRE(register_fat_binary(pin(live_compressed())) == nullptr);
}

namespace {
bool contains(const std::vector<uint8_t>& hay, const void* needle, size_t n) {
  const auto* p = static_cast<const uint8_t*>(needle);
  return std::search(hay.begin(), hay.end(), p, p + n) != hay.end();
}

bool contains(const std::vector<uint8_t>& hay, const char* needle) {
  return contains(hay, needle, std::strlen(needle));
}
}  // namespace

// ---------------------------------------------------------------------------
// One capture covers both paths. Before capture stopped storing bundles it
// stored each one whole, as far as its header said, and the terabyte entry
// took the capture down.
// ---------------------------------------------------------------------------
HRR_TEST_CASE(Unit_HRR_FatBinHeaderChecks) {
  ScopedDir cap(fs::temp_directory_path() / "hrr_fatbin_header.hrr");
  std::string out;
  { hrr::test::SpawnProc proc(hrr_self_exe(), /*capture_stdout=*/true,
                              /*capture_stderr=*/true);
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    set_proc_search_path(proc);
    int ret = proc.run("\"Unit_HRR_FatBinHeader_Direct\"");
    out = proc.getOutput();
    INFO("Capture exit: " << ret << "\n" << out);
    REQUIRE(ret == 0); }
  INFO("Capture output:\n" << out);

  hrr::Archive arc;
  REQUIRE(hrr::load_archive(hrr_single_process_archive(cap.path).string(), arc));

  // Every registration is recorded, none with a blob: the seven the workload
  // made with a null handle (five before HIP initialised, two HIP refused),
  // the three HIP loaded with a real one, each naming its code object.
  size_t null_handle = 0, loaded = 0;
  for (const auto& ev : arc.events) {
    if (ev.header().event_type != HRR_API_HIPREGISTERFATBINARY) continue;
    REQUIRE(ev.raw_payload.size() >= sizeof(hrr_args___hipRegisterFatBinary));
    const auto* a =
        reinterpret_cast<const hrr_args___hipRegisterFatBinary*>(ev.raw_payload.data());
    CHECK(a->blob_size == 0);
    // A handle means HIP loaded a code object; a swept one may name one too.
    if (a->ret || a->blob_hash_lo || a->blob_hash_hi)
      CHECK(arc.code_objects.count(hrr::hash_hex(a->blob_hash_lo, a->blob_hash_hi)) == 1);
    ++(a->ret ? loaded : null_handle);
  }
  CHECK(loaded == 3);
  CHECK(null_handle >= 7);

  std::vector<std::vector<uint8_t>> stored;
  for (const auto* files : {&arc.blobs, &arc.code_objects}) {
    for (const auto& [hex, path] : *files) {
      std::vector<uint8_t> bytes;
      REQUIRE(hrr::read_file_capped(path, bytes));
      stored.push_back(std::move(bytes));
    }
  }

  // The code object of each bundle HIP loaded is there, an ELF on its own
  // rather than inside a bundle, and only as long as the ELF: it ends with its
  // section header table, as the linker lays it out, even where the bundle
  // entry declared a terabyte. The page after that bundle cannot be read, so
  // a capture that hashed the declared size did not get this far.
  for (const char* kernel : {kLiveKernel, kFarEndKernel, kOversizedKernel}) {
    INFO("Kernel: " << kernel);
    size_t hits = 0;
    for (const auto& bytes : stored) {
      if (!contains(bytes, kernel)) continue;
      ++hits;
      REQUIRE(bytes.size() >= 64);
      CHECK(std::memcmp(bytes.data(), "\x7f" "ELF", 4) == 0);
      uint64_t shoff = 0;
      uint16_t shentsize = 0, shnum = 0;
      std::memcpy(&shoff, bytes.data() + 0x28, sizeof(shoff));
      std::memcpy(&shentsize, bytes.data() + 0x3a, sizeof(shentsize));
      std::memcpy(&shnum, bytes.data() + 0x3c, sizeof(shnum));
      CHECK(shoff + uint64_t(shentsize) * shnum == bytes.size());
    }
    CHECK(hits == 1);
  }

  // No file in the archive holds a bundle, or bytes of one HIP never loaded.
  for (const auto& bytes : stored) {
    CHECK_FALSE(contains(bytes, kBundleMagic, kBundleMagicLen));
    CHECK_FALSE(contains(bytes, "CCOB", 4));
    for (const char* marker : {kPreInitValidMarker, kPreInitLongIdMarker,
                               kPreInitOverflowMarker, kPreInitFarEndMarker,
                               kLiveForeignMarker}) {
      INFO("Marker: " << marker);
      CHECK_FALSE(contains(bytes, marker));
    }
  }
}

// ===========================================================================
// Which device replay loads a registration's code object on. HIP loaded it on
// the device current when the fat binary was registered, and on a host with
// GPUs of more than one kind it may load on no other. The workload registers
// with device 1 current, then makes device 0 current again.
// ===========================================================================
TEST_CASE("Unit_HRR_FatBinLoadDevice_Direct", "[.][hrr-direct]") {
  int n = 0;
  HRR_HIP_CHECK(hipGetDeviceCount(&n));
  if (n < 2) HRR_SKIP("needs two GPUs");
  HRR_HIP_CHECK(hipSetDevice(1));
  hipDeviceProp_t prop{};
  HRR_HIP_CHECK(hipGetDeviceProperties(&prop, 1));
  const std::vector<uint8_t> co = compile_code_object(kDevice1Kernel);
  const std::string target = std::string(kTargetPrefix) + prop.gcnArchName;
  REQUIRE(register_fat_binary(pin(valid_bundle(target, co))) != nullptr);
  HRR_HIP_CHECK(hipSetDevice(0));
}

// Replay pre-loads every code object before the first event, so it does not
// replay the switch to device 1 first: it has to load this one there itself.
// Run on GPUs of one kind, this shows where the load goes, not that a load
// elsewhere fails.
HRR_TEST_CASE(Unit_HRR_FatBinLoadDevice) {
  int n = 0;
  if (hipGetDeviceCount(&n) != hipSuccess || n < 2) HRR_SKIP("needs two GPUs");
  ScopedDir cap(fs::temp_directory_path() / "hrr_fatbin_load_device.hrr");
  { hrr::test::SpawnProc proc(hrr_self_exe(), /*capture_stdout=*/true,
                              /*capture_stderr=*/true);
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    set_proc_search_path(proc);
    int ret = proc.run("\"Unit_HRR_FatBinLoadDevice_Direct\"");
    INFO("Capture exit: " << ret << "\n" << proc.getOutput());
    REQUIRE(ret == 0); }

  const fs::path archive = hrr_single_process_archive(cap.path);
  hrr::Archive arc;
  REQUIRE(hrr::load_archive(archive.string(), arc));
  std::string hex;
  for (const auto& ev : arc.events) {
    if (ev.header().event_type != HRR_API_HIPREGISTERFATBINARY) continue;
    REQUIRE(ev.raw_payload.size() >= sizeof(hrr_args___hipRegisterFatBinary));
    const auto* a =
        reinterpret_cast<const hrr_args___hipRegisterFatBinary*>(ev.raw_payload.data());
    if (a->ret) hex = hrr::hash_hex(a->blob_hash_lo, a->blob_hash_hi);
  }
  INFO("Code object: " << hex);
  REQUIRE(arc.code_objects.count(hex) == 1);

  auto [rc, out] = hrr_playback_merged(archive);
  INFO("Replay:\n" << out);
  CHECK(rc == 0);
  CHECK(out.find("Pre-loading code object " + hex + " on device 1,") != std::string::npos);
}

#endif  // HRR_TEST_EXE

/**
 * @}
 */
