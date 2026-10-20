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
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <sys/mman.h>
#endif

// Exported by libamdhip64 and called by the module constructor clang emits for
// every HIP translation unit; no header declares it.
extern "C" void** __hipRegisterFatBinary(const void* data);
extern "C" void __hipRegisterFunction(void** modules, const void* hostFunction,
                                      char* deviceFunction, const char* deviceName,
                                      unsigned int threadLimit, uint3* tid, uint3* bid,
                                      dim3* blockDim, dim3* gridDim, int* wSize);
extern "C" void __hipRegisterVar(void** modules, void* var, char* hostVar, char* deviceVar,
                                 int ext, size_t size, int constant, int global);

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

// The code object with its first section the loader never reads (PROGBITS,
// not allocated: .comment as lld emits it) declaring a terabyte from the start
// of the file. HIP looks only at the ELF header of a shared object it is given.
std::vector<uint8_t> overstated_section(std::vector<uint8_t> co) {
  uint64_t shoff = 0, flags = 0;
  uint16_t shnum = 0;
  uint32_t type = 0;
  std::memcpy(&shoff, co.data() + 0x28, sizeof(shoff));
  std::memcpy(&shnum, co.data() + 0x3c, sizeof(shnum));
  for (size_t sh = shoff; sh < shoff + size_t(shnum) * 64; sh += 64) {
    std::memcpy(&type, co.data() + sh + 4, sizeof(type));
    std::memcpy(&flags, co.data() + sh + 8, sizeof(flags));
    if (type != 1 /* SHT_PROGBITS */ || (flags & 2 /* SHF_ALLOC */)) continue;
    put_u64(co, sh + 24, 0);
    put_u64(co, sh + 32, uint64_t(1) << 40);
    return co;
  }
  FAIL("no unallocated PROGBITS section to overstate");
  return co;
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
// And the ones after: four HIP loads, told apart by their kernel, two it refuses.
constexpr char     kLiveKernel[]            = "hrr_fatbin_live_probe";
constexpr char     kFarEndKernel[]          = "hrr_fatbin_far_end_probe";
constexpr char     kOversizedKernel[]       = "hrr_fatbin_oversized_probe";
constexpr char     kOverstatedKernel[]      = "hrr_fatbin_overstated_probe";
// Registered with device 1 current, launched on device 0.
constexpr char     kDevice1Kernel[]         = "hrr_fatbin_device1_probe";
// Registered before HIP initialises, from a bundle the parent builds, and read
// on both devices: a __device__ global and the kernel that returns it. Replay
// compares a copy back with a float tolerance, so the values written differ
// as floats too: 2.0f and 1.0f.
constexpr char     kGlobalKernel[]          = "hrr_fatbin_global_probe";
constexpr char     kDeviceGlobal[]          = "hrr_fatbin_device_global";
constexpr int      kGlobalInit              = 0x40000000;
constexpr int      kGlobalWritten           = 0x3f800000;
constexpr char     kGlobalBundleEnv[]       = "HRR_FATBIN_GLOBAL_BUNDLE";
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
// read, so a read that runs past the bundle faults.
const void* pin_before_guard(const std::vector<uint8_t>& bytes) {
  constexpr size_t kGuard = 65536;  // a multiple of any page size in use
  const size_t span = (bytes.size() + kGuard - 1) / kGuard * kGuard;
#if defined(_WIN32)
  // Reserved but not committed: no access.
  auto* p = static_cast<uint8_t*>(VirtualAlloc(nullptr, span + kGuard, MEM_RESERVE,
                                               PAGE_NOACCESS));
  REQUIRE(p != nullptr);
  REQUIRE(VirtualAlloc(p, span, MEM_COMMIT, PAGE_READWRITE) == p);
#else
  void* m = mmap(nullptr, span + kGuard, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  REQUIRE(m != MAP_FAILED);
  auto* p = static_cast<uint8_t*>(m);
  REQUIRE(mprotect(p + span, kGuard, PROT_NONE) == 0);
#endif
  std::memcpy(p, bytes.data(), bytes.size());
  return p;
}

void** register_fat_binary(const void* bundle) {
  // __CudaFatBinaryWrapper; HIP keeps this pointer too.
  struct Wrapper { uint32_t magic; uint32_t version; const void* binary; const void* dummy; };
  return __hipRegisterFatBinary(new Wrapper{kHipfMagic, 1, bundle, nullptr});
}

// A kernel writing 1, or with `global`, the value of that __device__ global.
std::vector<uint8_t> compile_code_object(const std::string& kernel,
                                         const std::string& global = "") {
  const std::string src =
      (global.empty() ? std::string() : "extern \"C\" __device__ int " + global + " = " +
                                            std::to_string(kGlobalInit) + ";\n") +
      "extern \"C\" __global__ void " + kernel + "(int* out) { out[0] = " +
      (global.empty() ? std::string("1") : global) + "; }\n";
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

// What clang's module constructor does for a kernel: only the stub's address matters.
void register_function(void** handle, const void* stub, const char* kernel) {
  __hipRegisterFunction(handle, stub, const_cast<char*>(kernel), kernel,
                        static_cast<unsigned>(-1), nullptr, nullptr, nullptr, nullptr, nullptr);
}

// Run the kernel registered for `stub` on `device` and return what it wrote.
int launch_probe(int device, const void* stub) {
  HRR_HIP_CHECK(hipSetDevice(device));
  int* out = nullptr;
  HRR_HIP_CHECK(hipMalloc(&out, sizeof(int)));
  HRR_HIP_CHECK(hipMemset(out, 0, sizeof(int)));
  void* args[] = {&out};
  HRR_HIP_CHECK(hipLaunchKernel(stub, dim3(1), dim3(1), args, 0, nullptr));
  int host = 0;
  HRR_HIP_CHECK(hipMemcpy(&host, out, sizeof(int), hipMemcpyDeviceToHost));
  HRR_HIP_CHECK(hipFree(out));
  return host;
}

// Whether GPUs 0 and 1 are of one kind, so a code object built for one runs on both.
bool devices_0_1_alike() {
  hipDeviceProp_t p0{}, p1{};
  HRR_HIP_CHECK(hipGetDeviceProperties(&p0, 0));
  HRR_HIP_CHECK(hipGetDeviceProperties(&p1, 1));
  return std::strcmp(p0.gcnArchName, p1.gcnArchName) == 0;
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

  // From here HIP digests each bundle inside the call. These four it loads.
  void** live = register_fat_binary(pin(valid_bundle(target, live_co)));
  REQUIRE(live != nullptr);
  REQUIRE(register_fat_binary(pin(far_end_bundle(target, far_end_co))) != nullptr);
  REQUIRE(register_fat_binary(pin_before_guard(oversized_bundle(target, oversized_co))) !=
          nullptr);
  REQUIRE(register_fat_binary(pin_before_guard(oversized_bundle(
              target, overstated_section(compile_code_object(kOverstatedKernel))))) != nullptr);
  // And these it refuses: no code object for this GPU, a bad header.
  REQUIRE(register_fat_binary(pin(live_foreign())) == nullptr);
  REQUIRE(register_fat_binary(pin(live_compressed())) == nullptr);

  // A launch for the replay to resolve.
  static const char live_stub = 0;
  register_function(live, &live_stub, kLiveKernel);
  CHECK(launch_probe(0, &live_stub) == 1);
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
  // the four HIP loaded with a real one, each naming its code object but the
  // one whose ELF overstates a section.
  size_t null_handle = 0, loaded = 0;
  for (const auto& ev : arc.events) {
    if (ev.header().event_type != HRR_API_HIPREGISTERFATBINARY) continue;
    REQUIRE(ev.raw_payload.size() >= sizeof(hrr_args___hipRegisterFatBinary));
    const auto* a =
        reinterpret_cast<const hrr_args___hipRegisterFatBinary*>(ev.raw_payload.data());
    CHECK(a->blob_size == 0);
    if (a->blob_hash_lo || a->blob_hash_hi)
      CHECK(arc.code_objects.count(hrr::hash_hex(a->blob_hash_lo, a->blob_hash_hi)) == 1);
    ++(a->ret ? loaded : null_handle);
  }
  CHECK(loaded == 4);
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
  // a capture that hashed the declared size did not get this far, nor one that
  // trusted the ELF's overstated section, whose code object is not stored.
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
    CHECK_FALSE(contains(bytes, kOverstatedKernel));
  }

  // An archive from before held the bundle as a blob, which replay loads as a
  // module, and a launch recorded without a code object finds its kernel by
  // name. Rewrite the live bundle and its launch that way: its code object
  // becomes the blob, and the launch names none.
  const fs::path archive = hrr_single_process_archive(cap.path);
  std::string events = read_text_file(archive / "events.bin");
  auto at = [&](const hrr::Event& ev) {
    const size_t pos = events.find(std::string(ev.raw_payload.begin(), ev.raw_payload.end()));
    REQUIRE(pos != std::string::npos);
    return pos;
  };
  std::string hex;
  for (const auto& ev : arc.events) {
    if (!ev.kernel_launch || ev.kernel_launch->kernel_name != kLiveKernel) continue;
    hex = hrr::hash_hex(ev.kernel_launch->co_hash_lo, ev.kernel_launch->co_hash_hi);
    // After the stream, the name's length and the name.
    const uint64_t none[2] = {0, 0};
    std::memcpy(&events[at(ev) + sizeof(hrr_event_header) + 10 + std::strlen(kLiveKernel)],
                none, sizeof(none));
  }
  INFO("Code object: " << hex);
  REQUIRE(arc.code_objects.count(hex) == 1);
  const fs::path blob = archive / "blobs" / hex.substr(0, 2) / (hex + ".blob");
  fs::create_directories(blob.parent_path());
  fs::rename(archive / "code_objects" / (hex + ".hsaco"), blob);
  const uint64_t blob_size = fs::file_size(blob);
  size_t registrations = 0;
  for (const auto& ev : arc.events) {
    if (ev.header().event_type != HRR_API_HIPREGISTERFATBINARY) continue;
    const auto* a =
        reinterpret_cast<const hrr_args___hipRegisterFatBinary*>(ev.raw_payload.data());
    if (hrr::hash_hex(a->blob_hash_lo, a->blob_hash_hi) != hex) continue;
    std::memcpy(&events[at(ev) + offsetof(hrr_args___hipRegisterFatBinary, blob_size)],
                &blob_size, sizeof(blob_size));
    ++registrations;
  }
  REQUIRE(registrations == 1);
  std::ofstream(archive / "events.bin", std::ios::binary | std::ios::trunc)
      .write(events.data(), static_cast<std::streamsize>(events.size()));
  auto [rc, replay] = hrr_playback_merged(archive);
  INFO("Replay:\n" << replay);
  CHECK(rc == 0);
}

// ===========================================================================
// Which devices replay loads a registration's code object on. HIP loaded it on
// the device current when the fat binary was registered; on a host with GPUs
// of more than one kind it loads only on GPUs of its own kind. The workload registers
// with device 1 current, then launches the kernel on device 0, which on GPUs of
// one kind uses the same code object; on GPUs of two kinds it cannot, and both
// halves skip.
// ===========================================================================
namespace {
// Register a __device__ global and the kernel that returns it, from the bundle
// at kGlobalBundleEnv's path, before HIP initialises so that capture records
// them; the parent compiled it, since compiling initialises HIP.
void register_global_bundle(const void* stub, int* shadow) {
  const char* path = std::getenv(kGlobalBundleEnv);
  if (!path) HRR_SKIP(kGlobalBundleEnv << " is not set");
  std::vector<uint8_t> bytes;
  REQUIRE(hrr::read_file_capped(path, bytes));
  void** handle = register_fat_binary(pin(bytes));
  REQUIRE(handle != nullptr);
  register_function(handle, stub, kGlobalKernel);
  __hipRegisterVar(handle, shadow, const_cast<char*>(kDeviceGlobal),
                   const_cast<char*>(kDeviceGlobal), 0, sizeof(int), 0, 0);
}

// The bundle register_global_bundle() reads, for device 0's GPU.
void write_global_bundle(const fs::path& path) {
  hipDeviceProp_t prop{};
  HRR_HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  const auto b = valid_bundle(std::string(kTargetPrefix) + prop.gcnArchName,
                              compile_code_object(kGlobalKernel, kDeviceGlobal));
  std::ofstream(path, std::ios::binary)
      .write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
}
}  // namespace

TEST_CASE("Unit_HRR_FatBinLoadDevice_Direct", "[.][hrr-direct]") {
  static int global_shadow = 0;
  static const char global_stub = 0;
  register_global_bundle(&global_stub, &global_shadow);

  int n = 0;
  HRR_HIP_CHECK(hipGetDeviceCount(&n));
  if (n < 2) HRR_SKIP("needs two GPUs");
  if (!devices_0_1_alike()) HRR_SKIP("GPUs 0 and 1 are of different kinds");
  HRR_HIP_CHECK(hipSetDevice(1));
  hipDeviceProp_t prop{};
  HRR_HIP_CHECK(hipGetDeviceProperties(&prop, 1));
  const std::vector<uint8_t> co = compile_code_object(kDevice1Kernel);
  const std::string target = std::string(kTargetPrefix) + prop.gcnArchName;
  void** handle = register_fat_binary(pin(valid_bundle(target, co)));
  REQUIRE(handle != nullptr);
  static const char stub = 0;
  register_function(handle, &stub, kDevice1Kernel);
  CHECK(launch_probe(0, &stub) == 1);

  // Each device has its own copy of the global. Write device 0's, through the
  // address HIP gives for it there, and read both.
  HRR_HIP_CHECK(hipSetDevice(0));
  void* global_addr = nullptr;
  HRR_HIP_CHECK(hipGetSymbolAddress(&global_addr, global_shadow));
  HRR_HIP_CHECK(
      hipMemcpy(global_addr, &kGlobalWritten, sizeof(int), hipMemcpyHostToDevice));
  CHECK(launch_probe(0, &global_stub) == kGlobalWritten);
  CHECK(launch_probe(1, &global_stub) == kGlobalInit);
}

// Replay pre-loads every code object before the first event, so it does not
// replay the switch to device 1 first: it has to load this one there itself,
// and on device 0 too, where the launch runs and a module on device 1 cannot.
// The global's code object is loaded on both devices too, and its registration
// has to bind device 0's copy, the one capture wrote: either read differs
// from capture's otherwise.
HRR_TEST_CASE(Unit_HRR_FatBinLoadDevice) {
  int n = 0;
  if (hipGetDeviceCount(&n) != hipSuccess || n < 2) HRR_SKIP("needs two GPUs");
  if (!devices_0_1_alike()) HRR_SKIP("GPUs 0 and 1 are of different kinds");
  ScopedDir cap(fs::temp_directory_path() / "hrr_fatbin_load_device.hrr");
  ScopedDir work(fs::temp_directory_path() / "hrr_fatbin_load_device.work");
  fs::create_directories(work.path);
  const fs::path global_bundle = work.path / "global.bundle";
  write_global_bundle(global_bundle);
  { hrr::test::SpawnProc proc(hrr_self_exe(), /*capture_stdout=*/true,
                              /*capture_stderr=*/true);
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    proc.setEnv(kGlobalBundleEnv, global_bundle.string());
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
  // The launch on device 0 names the code object the registration named.
  size_t launches = 0;
  for (const auto& ev : arc.events) {
    if (!ev.kernel_launch || ev.kernel_launch->kernel_name != kDevice1Kernel) continue;
    ++launches;
    CHECK(hrr::hash_hex(ev.kernel_launch->co_hash_lo, ev.kernel_launch->co_hash_hi) == hex);
  }
  REQUIRE(launches == 1);

  auto [rc, out] = hrr_playback_merged(archive);
  INFO("Replay:\n" << out);
  CHECK(rc == 0);
  CHECK(out.find("Pre-loading code object " + hex + " on device 1,") != std::string::npos);

  // The warm-up pass a kernel filter adds ends on device 1; the timed pass
  // still starts on device 0, where capture registered the global.
  auto [rc_filtered, out_filtered] = hrr_playback_merged(archive, "--kernel-filter hrr_fatbin");
  INFO("Filtered replay:\n" << out_filtered);
  CHECK(rc_filtered == 0);
}

// ===========================================================================
// A __device__ global registered before HIP initialises, copied to and back
// with nothing launched. Capture records its code object only if it resolves
// the global before it sweeps the fat binaries, since resolving it is what
// makes HIP load that code object; replay binds the global there. Replay does
// not replay symbol copies, so the binding is what is checked.
// ===========================================================================
TEST_CASE("Unit_HRR_FatBinGlobalCopy_Direct", "[.][hrr-direct]") {
  static int shadow = 0;
  static const char stub = 0;
  register_global_bundle(&stub, &shadow);
  HRR_HIP_CHECK(hipSetDevice(0));
  HRR_HIP_CHECK(hipMemcpyToSymbol(shadow, &kGlobalWritten, sizeof(int)));
  int back = 0;
  HRR_HIP_CHECK(hipMemcpyFromSymbol(&back, shadow, sizeof(int)));
  CHECK(back == kGlobalWritten);
}

HRR_TEST_CASE(Unit_HRR_FatBinGlobalCopy) {
  ScopedDir cap(fs::temp_directory_path() / "hrr_fatbin_global_copy.hrr");
  ScopedDir work(fs::temp_directory_path() / "hrr_fatbin_global_copy.work");
  fs::create_directories(work.path);
  const fs::path global_bundle = work.path / "global.bundle";
  write_global_bundle(global_bundle);
  hrr_capture_direct("Unit_HRR_FatBinGlobalCopy_Direct", cap.path, 1,
                     {{kGlobalBundleEnv, global_bundle.string()}});
  const fs::path archive = hrr_single_process_archive(cap.path);
  hrr::Archive arc;
  REQUIRE(hrr::load_archive(archive.string(), arc));
  // The global's code object is in the archive.
  size_t holding = 0;
  for (const auto& [hex, path] : arc.code_objects) {
    std::vector<uint8_t> bytes;
    REQUIRE(hrr::read_file_capped(path, bytes));
    holding += contains(bytes, kDeviceGlobal);
  }
  CHECK(holding == 1);
  auto [rc, out] = hrr_playback_merged(archive);
  INFO("Replay:\n" << out);
  CHECK(rc == 0);
  CHECK(out.find("symbol '" + std::string(kDeviceGlobal) + "' is not in any") ==
        std::string::npos);
}

#endif  // HRR_TEST_EXE

/**
 * @}
 */
