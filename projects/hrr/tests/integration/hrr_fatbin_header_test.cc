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
 * object, cut to the extent its ELF headers give, and the register event
 * without a blob. A bundle whose header declares an entry far past the real
 * image is captured without reading there.
 *
 * Two paths record fat binaries and both are exercised:
 *   - registered before HIP initialises: swept from HIP's list in
 *     hip_capture_init, after HIP has loaded what it needed for globals;
 *   - registered after: seen by the __hipRegisterFatBinary shim, once HIP has
 *     digested the bundle.
 * The workload calls __hipRegisterFatBinary itself with bundles laid out in
 * memory the way clang-offload-bundler lays them out, valid and altered.
 *
 * Every code object capture stores under code_objects/ has its .comment and
 * .debug* sections zeroed: they hold the compiler build and the build paths,
 * and the GPU never loads them.
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
// These two take names as char* but only read them, so the calls below
// const_cast string literals to match the C signature.
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
constexpr uint32_t kHipfMagic      = 0x48495046u;  // "HIPF"
constexpr uint64_t kTerabyte        = uint64_t(1) << 40;

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
    put_u64(co, sh + 32, kTerabyte);
    return co;
  }
  FAIL("no unallocated PROGBITS section to overstate");
  return co;
}

// What clang emits for one target: an empty host entry, then the code object.
// A non-zero host_size or co_size is declared in that entry instead. HIP never
// loads the host entry, and keeps the size the code object's entry declares.
std::vector<uint8_t> target_bundle(const std::string& target_id,
                                   const std::vector<uint8_t>& payload,
                                   uint64_t host_size = 0, uint64_t co_size = 0) {
  auto b = uncompressed_bundle({{kPayloadOffset, host_size, kHostId},
                                {kPayloadOffset, co_size ? co_size : payload.size(), target_id}},
                               kPayloadOffset + payload.size());
  std::memcpy(b.data() + kPayloadOffset, payload.data(), payload.size());
  return b;
}

std::vector<uint8_t> marker_payload(const char* marker) {
  std::vector<uint8_t> p(kMarkerPayload, 0);
  std::memcpy(p.data(), marker, std::strlen(marker));
  return p;
}

// The payloads of the bundles HIP loads no code object from carry a marker,
// which the parent looks for in every file of the archive. Two are registered
// before HIP initialises, one after.
constexpr char     kPreInitValidMarker[]    = "HRR fatbin pre-init valid";
constexpr char     kPreInitFarEndMarker[]   = "HRR fatbin pre-init far end";
constexpr char     kLiveForeignMarker[]     = "HRR fatbin live foreign";
// The four bundles HIP loads, told apart by their kernel.
constexpr char     kLiveKernel[]            = "hrr_fatbin_live_probe";
constexpr char     kFarEndKernel[]          = "hrr_fatbin_far_end_probe";
constexpr char     kOversizedKernel[]       = "hrr_fatbin_oversized_probe";
constexpr char     kOverstatedKernel[]      = "hrr_fatbin_overstated_probe";
// Registered before HIP initialises, from a bundle the parent builds: a
// __device__ global, beside a kernel that returns it.
constexpr char     kGlobalKernel[]          = "hrr_fatbin_global_probe";
constexpr char     kDeviceGlobal[]          = "hrr_fatbin_device_global";
constexpr int      kGlobalInit              = 0x40000000;
constexpr int      kGlobalWritten           = 0x3f800000;
constexpr char     kGlobalBundleEnv[]       = "HRR_FATBIN_GLOBAL_BUNDLE";

std::vector<uint8_t> preinit_valid()      { return foreign_bundle(kPreInitValidMarker); }
std::vector<uint8_t> preinit_far_end() {
  return target_bundle(kForeignId, marker_payload(kPreInitFarEndMarker), kTerabyte);
}
std::vector<uint8_t> live_foreign()       { return foreign_bundle(kLiveForeignMarker); }

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

// Like pin(), but followed by 64 KiB that cannot be read, so reading the
// terabyte a bundle entry declares faults.
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

// A kernel writing 1, or with `global`, the value of that __device__ global,
// compiled with `options`.
std::vector<uint8_t> compile_code_object(const std::string& kernel,
                                         const std::string& global = "",
                                         std::vector<const char*> options = {}) {
  const std::string src =
      (global.empty() ? std::string() : "extern \"C\" __device__ int " + global + " = " +
                                            std::to_string(kGlobalInit) + ";\n") +
      "extern \"C\" __global__ void " + kernel + "(int* out) { out[0] = " +
      (global.empty() ? std::string("1") : global) + "; }\n";
  hiprtcProgram prog = nullptr;
  HRR_HIPRTC_CHECK(hiprtcCreateProgram(&prog, src.c_str(), "hrr_fatbin_probe.hip", 0,
                                       nullptr, nullptr));
  if (hiprtcCompileProgram(prog, static_cast<int>(options.size()), options.data()) !=
      HIPRTC_SUCCESS) {
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

// Run the kernel registered for `stub` and return what it wrote.
int launch_probe(const void* stub) {
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

}  // namespace

// ===========================================================================
// The captured workload.
// ===========================================================================
TEST_CASE("Unit_HRR_FatBinHeader_Direct", "[.][hrr-direct]") {
  // HIP is not initialised yet: it only notes these and returns a handle, and
  // capture sees them when it starts, by sweeping HIP's list.
  REQUIRE(register_fat_binary(pin(preinit_valid())) != nullptr);
  REQUIRE(register_fat_binary(pin(preinit_far_end())) != nullptr);

  // Initialises HIP, and with it capture.
  HRR_HIP_CHECK(hipSetDevice(0));
  hipDeviceProp_t prop{};
  HRR_HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  const std::string target = std::string(kTargetPrefix) + prop.gcnArchName;
  const std::vector<uint8_t> live_co = compile_code_object(kLiveKernel);
  const std::vector<uint8_t> far_end_co = compile_code_object(kFarEndKernel);
  const std::vector<uint8_t> oversized_co = compile_code_object(kOversizedKernel);

  // From here HIP digests each bundle inside the call. These four it loads.
  void** live = register_fat_binary(pin(target_bundle(target, live_co)));
  REQUIRE(live != nullptr);
  REQUIRE(register_fat_binary(pin(target_bundle(target, far_end_co, kTerabyte))) != nullptr);
  REQUIRE(register_fat_binary(pin_before_guard(
              target_bundle(target, oversized_co, 0, kTerabyte))) != nullptr);
  REQUIRE(register_fat_binary(pin_before_guard(target_bundle(
              target, overstated_section(compile_code_object(kOverstatedKernel)), 0,
              kTerabyte))) != nullptr);
  // And this one it refuses: no code object for this GPU.
  REQUIRE(register_fat_binary(pin(live_foreign())) == nullptr);

  // A launch for the replay to resolve.
  static const char live_stub = 0;
  register_function(live, &live_stub, kLiveKernel);
  CHECK(launch_probe(&live_stub) == 1);
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
// One capture covers both paths, and its archive replays, also with the live
// bundle stored as a blob.
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

  const fs::path archive = hrr_single_process_archive(cap.path);
  hrr::Archive arc;
  REQUIRE(hrr::load_archive(archive.string(), arc));

  // Every registration is recorded, none with a blob: the three the workload
  // made with a null handle (two before HIP initialised, one HIP refused),
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
  CHECK(null_handle >= 3);

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
  // entry declared a terabyte. A guard that cannot be read follows the bundle, so
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
    for (const char* marker : {kPreInitValidMarker, kPreInitFarEndMarker, kLiveForeignMarker}) {
      INFO("Marker: " << marker);
      CHECK_FALSE(contains(bytes, marker));
    }
    CHECK_FALSE(contains(bytes, kOverstatedKernel));
  }

  auto [rc, replay] = hrr_playback_merged(archive);
  INFO("Replay:\n" << replay);
  CHECK(rc == 0);

  // A registration whose bundle is stored as a blob replays through the
  // register handler, which loads the blob as a module. Rewrite the live
  // kernel's registration that way, and its launch to name no code object, so
  // only that module holds the kernel.
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
  std::vector<uint8_t> co;
  REQUIRE(hrr::read_file_capped(arc.code_objects.at(hex), co));
  hipDeviceProp_t prop{};
  HRR_HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  const auto bundle = target_bundle(std::string(kTargetPrefix) + prop.gcnArchName, co);
  const fs::path blob = archive / "blobs" / hex.substr(0, 2) / (hex + ".blob");
  fs::create_directories(blob.parent_path());
  std::ofstream(blob, std::ios::binary)
      .write(reinterpret_cast<const char*>(bundle.data()),
             static_cast<std::streamsize>(bundle.size()));
  fs::remove(arc.code_objects.at(hex));
  const uint64_t blob_size = bundle.size();
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
  auto [rc_old, replay_old] = hrr_playback_merged(archive);
  INFO("Replay of the rewritten archive:\n" << replay_old);
  CHECK(rc_old == 0);
}

// ===========================================================================
// A __device__ global registered before HIP initialises, copied to with
// nothing launched. Capture records its code object only if it resolves
// the global before it sweeps the fat binaries, since resolving it is what
// makes HIP load that code object.
// ===========================================================================
namespace {
// Register a __device__ global from the bundle at kGlobalBundleEnv's path,
// before HIP initialises so that capture records it; the parent compiled it,
// since compiling initialises HIP.
void register_global_bundle(int* shadow) {
  const char* path = std::getenv(kGlobalBundleEnv);
  if (!path) HRR_SKIP(kGlobalBundleEnv << " is not set");
  std::vector<uint8_t> bytes;
  REQUIRE(hrr::read_file_capped(path, bytes));
  void** handle = register_fat_binary(pin(bytes));
  REQUIRE(handle != nullptr);
  __hipRegisterVar(handle, shadow, const_cast<char*>(kDeviceGlobal),
                   const_cast<char*>(kDeviceGlobal), 0, sizeof(int), 0, 0);
}

// The bundle register_global_bundle() reads, for device 0's GPU.
void write_global_bundle(const fs::path& path) {
  hipDeviceProp_t prop{};
  HRR_HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  const auto b = target_bundle(std::string(kTargetPrefix) + prop.gcnArchName,
                               compile_code_object(kGlobalKernel, kDeviceGlobal));
  std::ofstream(path, std::ios::binary)
      .write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
}
}  // namespace

TEST_CASE("Unit_HRR_FatBinGlobalCopy_Direct", "[.][hrr-direct]") {
  static int shadow = 0;
  register_global_bundle(&shadow);
  HRR_HIP_CHECK(hipSetDevice(0));
  HRR_HIP_CHECK(hipMemcpyToSymbol(shadow, &kGlobalWritten, sizeof(int)));
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
}

// ===========================================================================
// Code objects are stored without their build paths and compiler build.
// ===========================================================================
namespace {
// One probe is registered in a fat binary, the other loaded as a module, so
// each leaves its own code object.
constexpr char kScrubFatBinKernel[] = "hrr_scrub_fatbin_probe";
constexpr char kScrubModuleKernel[] = "hrr_scrub_module_probe";
}  // namespace

TEST_CASE("Unit_HRR_CodeObjectScrub_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  hipDeviceProp_t prop{};
  HRR_HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  const std::string target = std::string(kTargetPrefix) + prop.gcnArchName;
  // Both are built with debug information, which is what puts the source file
  // name and build directory into a code object. This one is registered in a
  // fat binary, of which capture stores the code object HIP loads.
  REQUIRE(register_fat_binary(pin(target_bundle(
              target, compile_code_object(kScrubFatBinKernel, "", {"-g"})))) != nullptr);

  // Loaded as a module, and launched so replay runs the stored copy.
  const std::vector<uint8_t> co = compile_code_object(kScrubModuleKernel, "", {"-g"});
  hipModule_t mod = nullptr;
  HRR_HIP_CHECK(hipModuleLoadData(&mod, co.data()));
  hipFunction_t fn = nullptr;
  HRR_HIP_CHECK(hipModuleGetFunction(&fn, mod, kScrubModuleKernel));
  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, sizeof(int)));
  HRR_HIP_CHECK(hipMemset(d, 0, sizeof(int)));
  void* args[] = {&d};
  HRR_HIP_CHECK(hipModuleLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
  int h = 0;
  HRR_HIP_CHECK(hipMemcpy(&h, d, sizeof(int), hipMemcpyDeviceToHost));
  REQUIRE(h == 1);
  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipModuleUnload(mod));
}

namespace {
struct ScrubCount {
  size_t elves = 0;
  size_t sections = 0;  // non-empty .comment and .debug* sections
  size_t left = 0;      // of those, still holding a non-zero byte
  size_t typed = 0;     // of those, not SHT_NOBITS, so DWARF readers parse them
};

// Walks the section headers of an ELF64 image independently of capture's own
// parser and counts the sections capture has to scrub. False if the image does
// not parse, including a section name that does not end inside the string table.
bool count_scrub_sections(const uint8_t* p, size_t size, ScrubCount& c) {
  auto u16 = [&](size_t at) { uint16_t v; std::memcpy(&v, p + at, 2); return v; };
  auto u32 = [&](size_t at) { uint32_t v; std::memcpy(&v, p + at, 4); return v; };
  auto u64 = [&](size_t at) { uint64_t v; std::memcpy(&v, p + at, 8); return v; };
  if (size < 64 || std::memcmp(p, "\x7f" "ELF", 4) != 0 || u16(0x3a) != 64) return false;
  const uint64_t shoff = u64(0x28);
  const uint16_t shnum = u16(0x3c);
  const uint16_t shstrndx = u16(0x3e);
  if (shoff > size || uint64_t{shnum} * 64 > size - shoff || shstrndx >= shnum) return false;
  const uint64_t names = u64(shoff + shstrndx * 64 + 24);
  const uint64_t names_size = u64(shoff + shstrndx * 64 + 32);
  if (names > size || names_size > size - names) return false;
  ++c.elves;
  for (uint16_t i = 0; i < shnum; ++i) {
    const size_t sh = shoff + size_t{i} * 64;
    const uint32_t at = u32(sh);
    if (at >= names_size) return false;
    const char* s = reinterpret_cast<const char*>(p + names + at);
    const size_t name_len = strnlen(s, names_size - at);
    if (name_len == names_size - at) return false;
    const std::string name(s, name_len);
    if (name != ".comment" && name.rfind(".debug", 0) != 0) continue;
    const uint64_t off = u64(sh + 24), len = u64(sh + 32);
    if (len == 0) continue;
    if (off > size || len > size - off) return false;
    ++c.sections;
    if (u32(sh + 4) != 8 /*SHT_NOBITS*/) ++c.typed;
    if (std::any_of(p + off, p + off + len, [](uint8_t b) { return b != 0; })) ++c.left;
  }
  return true;
}
}  // namespace

HRR_TEST_CASE(Unit_HRR_CodeObjectScrub) {
  ScopedDir cap(fs::temp_directory_path() / "hrr_code_object_scrub.hrr");
  // Replay checks the kernel's result, so the scrubbed copy loads and runs.
  hrr_run_roundtrip("Unit_HRR_CodeObjectScrub_Direct", cap.path);

  hrr::Archive arc;
  REQUIRE(hrr::load_archive(hrr_single_process_archive(cap.path).string(), arc));

  // The code objects the loaded fat binaries' registrations name.
  std::vector<std::string> registered;
  for (const auto& ev : arc.events) {
    if (ev.header().event_type != HRR_API_HIPREGISTERFATBINARY) continue;
    const auto* a =
        reinterpret_cast<const hrr_args___hipRegisterFatBinary*>(ev.raw_payload.data());
    if (a->ret && (a->blob_hash_lo || a->blob_hash_hi))
      registered.push_back(hrr::hash_hex(a->blob_hash_lo, a->blob_hash_hi));
  }

  // Every stored code object is scrubbed, the module's and the one HIP took
  // from the fat binary alike; each probe is counted where it is found.
  ScrubCount module_count, bundle_count;
  size_t module_files = 0, bundle_files = 0;
  for (const auto& [hex, path] : arc.code_objects) {
    INFO("Code object " << path);
    std::vector<uint8_t> image;
    REQUIRE(hrr::read_file_capped(path, image));
    CHECK_FALSE(contains(image, "clang version"));
    CHECK_FALSE(contains(image, "hrr_fatbin_probe.hip"));
    if (contains(image, kScrubFatBinKernel)) {
      CHECK(std::find(registered.begin(), registered.end(), hex) != registered.end());
      ++bundle_files;
      REQUIRE(count_scrub_sections(image.data(), image.size(), bundle_count));
    } else {
      if (contains(image, kScrubModuleKernel)) ++module_files;
      REQUIRE(count_scrub_sections(image.data(), image.size(), module_count));
    }
  }
  CHECK(module_files == 1);
  CHECK(bundle_files == 1);

  INFO("Module code objects: " << module_count.elves << " ELF, " << module_count.sections
                               << " sections, " << module_count.left << " not zeroed, "
                               << module_count.typed << " not SHT_NOBITS");
  INFO("Fat binary code objects: " << bundle_count.elves << " ELF, " << bundle_count.sections
                                   << " sections, " << bundle_count.left << " not zeroed, "
                                   << bundle_count.typed << " not SHT_NOBITS");
  // The probe was built with -g, so both copies had something to zero.
  CHECK(module_count.sections >= 2);
  CHECK(bundle_count.sections >= 2);
  CHECK(module_count.left == 0);
  CHECK(bundle_count.left == 0);
  // Retyped, so a debugger on the replay sees no debug information rather than zeros.
  CHECK(module_count.typed == 0);
  CHECK(bundle_count.typed == 0);
}

// The walker reads a section name only inside its string table, even when the
// bytes after the table would complete a name it looks for.
HRR_TEST_CASE(Unit_HRR_CodeObjectScrubWalker) {
  std::vector<uint8_t> elf(80 + 2 * 64, 0);
  auto put = [&](size_t at, uint64_t v, size_t n) { std::memcpy(elf.data() + at, &v, n); };
  std::memcpy(elf.data(), "\x7f" "ELF", 4);
  elf[4] = 2;  // ELFCLASS64
  elf[5] = 1;  // ELFDATA2LSB
  put(0x28, 80, 8);  // e_shoff
  put(0x3a, 64, 2);  // e_shentsize
  put(0x3c, 2, 2);   // e_shnum
  put(0x3e, 1, 2);   // e_shstrndx
  std::memcpy(elf.data() + 64, "\0.comment", 10);
  // Section 1 is the string table, named at offset 1, and declares only "\0.comm".
  const size_t sh = 80 + 64;
  put(sh, 1, 4);
  put(sh + 4, 3 /*SHT_STRTAB*/, 4);
  put(sh + 24, 64, 8);
  put(sh + 32, 6, 8);
  ScrubCount c;
  CHECK_FALSE(count_scrub_sections(elf.data(), elf.size(), c));
  // Declared up to its terminator, the same table parses.
  put(sh + 32, 10, 8);
  c = ScrubCount{};
  CHECK(count_scrub_sections(elf.data(), elf.size(), c));
  CHECK(c.sections == 1);
}

#endif  // HRR_TEST_EXE

/**
 * @}
 */
