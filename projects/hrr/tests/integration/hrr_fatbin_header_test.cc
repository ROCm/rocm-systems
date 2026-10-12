/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR fat binary header checks
 * @{
 * @ingroup HRRTest
 * Tests for which registered fat binaries capture stores as blobs.
 *
 * Capture stores each registered fat binary so replay can load it with
 * hipModuleLoadData, and the byte count it stores comes from the bundle
 * header. HIP checks that header before it loads the bundle, and capture
 * applies the same checks: a bundle whose header fails them is not stored,
 * and neither is a bundle HIP refused to register after it initialised.
 *
 * Two paths record fat binaries and both are exercised:
 *   - registered before HIP initialises: swept from HIP's list in
 *     hip_capture_init, where only the header checks apply;
 *   - registered after: seen by the __hipRegisterFatBinary shim, which also
 *     has HIP's verdict.
 * The workload calls __hipRegisterFatBinary itself with bundles laid out in
 * memory the way clang-offload-bundler lays them out, valid and altered.
 */

#include "hrr_test_common.hh"
#include "hrr_test_process.hh"

#include <hip/hiprtc.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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
constexpr uint64_t kPreInitCompressedTag    = 0x4852524650524531ull;
// And the ones after, that HIP refuses.
constexpr char     kLiveForeignMarker[]     = "HRR fatbin live foreign";
constexpr uint64_t kLiveCompressedTag       = 0x485252464C495631ull;

std::vector<uint8_t> preinit_valid()      { return foreign_bundle(kPreInitValidMarker); }
std::vector<uint8_t> preinit_compressed() { return short_compressed_bundle(kPreInitCompressedTag); }
std::vector<uint8_t> preinit_long_id()    { return long_id_bundle(kPreInitLongIdMarker); }
std::vector<uint8_t> preinit_overflow() {
  return host_overflow_bundle(kForeignId, marker_payload(kPreInitOverflowMarker));
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

void** register_fat_binary(const void* bundle) {
  // __CudaFatBinaryWrapper; HIP keeps this pointer too.
  struct Wrapper { uint32_t magic; uint32_t version; const void* binary; const void* dummy; };
  return __hipRegisterFatBinary(new Wrapper{kHipfMagic, 1, bundle, nullptr});
}

std::vector<uint8_t> compile_code_object() {
  static const char* src = R"(
extern "C" __global__ void hrr_fatbin_probe(int* out) { out[0] = 1; }
)";
  hiprtcProgram prog = nullptr;
  HRR_HIPRTC_CHECK(hiprtcCreateProgram(&prog, src, "hrr_fatbin_probe.hip", 0, nullptr,
                                       nullptr));
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

uint64_t read_u64(const std::vector<uint8_t>& b, size_t at) {
  uint64_t v = 0;
  std::memcpy(&v, b.data() + at, sizeof(v));
  return v;
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

  // Initialises HIP, and with it capture.
  HRR_HIP_CHECK(hipSetDevice(0));
  hipDeviceProp_t prop{};
  HRR_HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  const std::string target = std::string(kTargetPrefix) + prop.gcnArchName;
  const std::vector<uint8_t> co = compile_code_object();
  REQUIRE(co.size() > 4);
  REQUIRE(std::memcmp(co.data(), "\x7f" "ELF", 4) == 0);

  // From here HIP digests each bundle inside the call. These two it loads.
  REQUIRE(register_fat_binary(pin(valid_bundle(target, co))) != nullptr);
  REQUIRE(register_fat_binary(pin(host_overflow_bundle(target, co))) != nullptr);
  // And these it refuses: no code object for this GPU, a bad header.
  REQUIRE(register_fat_binary(pin(live_foreign())) == nullptr);
  REQUIRE(register_fat_binary(pin(live_compressed())) == nullptr);
}

namespace {
struct RecordedFatBinary {
  uint64_t             ret = 0;
  uint64_t             blob_size = 0;
  bool                 has_blob = false;
  std::vector<uint8_t> blob;
};

// True when the capture stored bytes from the start of `bundle`: all of it, or
// the leading part a header overstating or understating its size would give.
bool stored_from(const RecordedFatBinary& r, const std::vector<uint8_t>& bundle) {
  return r.has_blob && !r.blob.empty() && r.blob.size() <= bundle.size() &&
         std::memcmp(r.blob.data(), bundle.data(), r.blob.size()) == 0;
}
}  // namespace

// ---------------------------------------------------------------------------
// One capture covers both paths. Before the header checks, every altered
// bundle here was stored: the short compressed headers as their first 23
// bytes, the others whole, and the two HIP refused after it initialised as
// well, although no launch from them can succeed.
// ---------------------------------------------------------------------------
HRR_TEST_CASE(Unit_HRR_FatBinHeaderChecks) {
  ScopedDir cap(fs::temp_directory_path() / "hrr_fatbin_header.hrr");
  std::string out;
  { hrr::test::SpawnProc proc(hrr_self_exe(), /*capture_stdout=*/true,
                              /*capture_stderr=*/true);
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    // The refusals are reported as warnings.
    proc.setEnv("AMD_LOG_LEVEL", "2");
    set_proc_search_path(proc);
    int ret = proc.run("\"Unit_HRR_FatBinHeader_Direct\"");
    out = proc.getOutput();
    INFO("Capture exit: " << ret << "\n" << out);
    REQUIRE(ret == 0); }
  INFO("Capture output:\n" << out);

  hrr::Archive arc;
  REQUIRE(hrr::load_archive(hrr_single_process_archive(cap.path).string(), arc));

  std::vector<RecordedFatBinary> recorded;
  for (const auto& ev : arc.events) {
    if (ev.header().event_type != HRR_API_HIPREGISTERFATBINARY) continue;
    REQUIRE(ev.raw_payload.size() >= sizeof(hrr_args___hipRegisterFatBinary));
    const auto* a =
        reinterpret_cast<const hrr_args___hipRegisterFatBinary*>(ev.raw_payload.data());
    RecordedFatBinary r;
    r.ret = a->ret;
    r.blob_size = a->blob_size;
    r.has_blob = a->blob_hash_lo != 0 || a->blob_hash_hi != 0;
    if (r.has_blob) REQUIRE(hrr::read_blob(arc, a->blob_hash_lo, a->blob_hash_hi, r.blob));
    recorded.push_back(std::move(r));
  }

  // Registered before HIP initialised (handle recorded as 0): a bundle with a
  // sound header is stored whole, even with no code object for this GPU.
  const auto pre_valid = preinit_valid();
  size_t pre_valid_hits = 0;
  for (const auto& r : recorded) {
    if (r.ret == 0 && r.has_blob && r.blob == pre_valid) {
      CHECK(r.blob_size == pre_valid.size());
      ++pre_valid_hits;
    }
  }
  CHECK(pre_valid_hits == 1);

  // Registered after HIP initialised and loaded by it (a real handle): the
  // valid bundle is stored whole, the one with the wrapping host entry not at
  // all, though HIP loaded it, because no end can be computed for it.
  std::vector<const RecordedFatBinary*> loaded_live;
  for (const auto& r : recorded)
    if (r.ret != 0) loaded_live.push_back(&r);
  REQUIRE(loaded_live.size() == 2);
  const RecordedFatBinary* valid = nullptr;
  const RecordedFatBinary* overflow = nullptr;
  for (const auto* r : loaded_live) (r->has_blob ? valid : overflow) = r;
  REQUIRE(valid != nullptr);
  REQUIRE(overflow != nullptr);
  CHECK(overflow->blob_size == 0);

  // The stored valid bundle is the one the workload built: its code object
  // entry ends exactly where the blob ends.
  const auto& vb = valid->blob;
  REQUIRE(vb.size() > kPayloadOffset + 4);
  CHECK(std::memcmp(vb.data(), kBundleMagic, kBundleMagicLen) == 0);
  CHECK(read_u64(vb, kBundleMagicLen) == 2);
  const size_t second = kTableStart + kEntryHeaderSize + (sizeof(kHostId) - 1);
  CHECK(read_u64(vb, second) == kPayloadOffset);
  CHECK(read_u64(vb, second + 8) == vb.size() - kPayloadOffset);
  CHECK(valid->blob_size == vb.size());
  CHECK(std::memcmp(vb.data() + kPayloadOffset, "\x7f" "ELF", 4) == 0);

  // Refused by HIP after it initialised: the register event is still written,
  // without a blob, so replay treats it as a no-op.
  size_t refused_live = 0;
  for (const auto& r : recorded)
    if (r.ret == 0 && !r.has_blob && r.blob_size == 0) ++refused_live;
  CHECK(refused_live == 2);

  // No bytes from any altered or refused bundle reached the archive.
  const struct {
    const char*          name;
    std::vector<uint8_t> bytes;
  } rejected[] = {
      {"pre-init compressed totalSize below its header", preinit_compressed()},
      {"pre-init entry table past 4096 bytes", preinit_long_id()},
      {"pre-init host entry whose offset plus size wraps", preinit_overflow()},
      {"live bundle with no code object for this GPU", live_foreign()},
      {"live compressed totalSize below its header", live_compressed()},
  };
  for (const auto& bad : rejected) {
    INFO("Bundle: " << bad.name);
    size_t hits = 0;
    for (const auto& r : recorded)
      if (stored_from(r, bad.bytes)) ++hits;
    CHECK(hits == 0);
  }

  // Each refusal says why.
  CHECK(out.find("is smaller than its 24-byte header") != std::string::npos);
  CHECK(out.find("past the first 4096 bytes") != std::string::npos);
  CHECK(out.find("an offset and size that overflow") != std::string::npos);
  CHECK(out.find("HIP did not register it") != std::string::npos);
}

#endif  // HRR_TEST_EXE

/**
 * @}
 */
