////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in the
//    documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#include "core/inc/hotswap.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <dlfcn.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include "core/inc/hotswap_gfx_query.hpp"
#include "core/util/os.h"

namespace rocr {
namespace hotswap {
namespace {

#ifdef ROCR_HOTSWAP_TESTING
std::atomic<bool> g_force_retarget_code_object_failure_for_testing{false};
std::mutex g_comgr_cache_fingerprint_override_mutex;
bool g_has_comgr_cache_fingerprint_override = false;
DiskCacheDigest g_comgr_cache_fingerprint_override{};
#endif

std::string g_comgr_lib_path;

constexpr char kGfx1250[] = "gfx1250";
constexpr char kGfx1250B0Feature[] = ":gfx1250-b0-specific+";
constexpr char kGfx1250A0Feature[] = ":gfx1250-b0-specific-";

enum class Gfx1250Stepping {
  kB0,
  kA0,
};

bool IsEnvFlagEnabled(const char* name) {
  if (!os::IsEnvVarSet(name)) {
    return false;
  }

  std::string value = os::GetEnvVar(name);
  if (value.empty()) {
    return false;
  }

  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value != "0" && value != "off" && value != "false" && value != "no" && value != "n" &&
      value != "f";
}

bool IsHotswapDisabledByEnv() { return IsEnvFlagEnabled("HSA_HOTSWAP_DISABLE"); }

bool AreEntryTrampolinesRequested() {
  constexpr char kEnvName[] = "AMD_COMGR_HOTSWAP_ENTRY_TRAMPOLINES";
  if (!os::IsEnvVarSet(kEnvName)) {
    return kDefaultEntryTrampolinesEnabled;
  }
  return IsEnvFlagEnabled(kEnvName);
}

bool IsStrictModeRequested() {
  return IsEnvFlagEnabled("HSA_HOTSWAP_STRICT_MODE");
}

bool IsVerboseLoggingEnabled() {
  static const bool verbose = IsEnvFlagEnabled("HSA_HOTSWAP_VERBOSE");
  return verbose;
}

#define HOTSWAP_LOG(...)                                                                           \
  do {                                                                                             \
    if (IsVerboseLoggingEnabled()) {                                                               \
      fprintf(stderr, __VA_ARGS__);                                                                \
    }                                                                                              \
  } while (false)

constexpr size_t kDigestSize = 32;
constexpr size_t kBlake2bBlockSize = 128;

uint64_t LoadLittleEndian64(const uint8_t* bytes) {
  uint64_t value = 0;
  for (size_t i = 0; i < sizeof(value); ++i) {
    value |= static_cast<uint64_t>(bytes[i]) << (i * 8);
  }
  return value;
}

void StoreLittleEndian64(uint64_t value, uint8_t* bytes) {
  for (size_t i = 0; i < sizeof(value); ++i) {
    bytes[i] = static_cast<uint8_t>(value >> (i * 8));
  }
}

uint64_t RotateRight(uint64_t value, unsigned int bits) {
  return (value >> bits) | (value << (64 - bits));
}

// Small dependency-free BLAKE2b-256 implementation for persistent cache
// identity and integrity. BLAKE2b is optimized for 64-bit CPUs and avoids
// adding a crypto-library dependency to ROCr.
class Blake2b256 {
 public:
  Blake2b256() {
    static constexpr uint64_t kInitializationVector[8] = {
        0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
        0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL};
    std::copy(std::begin(kInitializationVector), std::end(kInitializationVector), state_.begin());
    state_[0] ^= 0x01010000U | kDigestSize;
  }

  void Update(const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    while (size != 0) {
      if (buffer_size_ == buffer_.size()) {
        AddCount(buffer_.size());
        Compress(false);
        buffer_size_ = 0;
      }
      const size_t count = std::min(size, buffer_.size() - buffer_size_);
      std::memcpy(buffer_.data() + buffer_size_, bytes, count);
      buffer_size_ += count;
      bytes += count;
      size -= count;
    }
  }

  void UpdateUint64(uint64_t value) {
    uint8_t bytes[sizeof(value)];
    StoreLittleEndian64(value, bytes);
    Update(bytes, sizeof(bytes));
  }

  DiskCacheDigest Final() {
    AddCount(buffer_size_);
    std::fill(buffer_.begin() + buffer_size_, buffer_.end(), 0);
    Compress(true);

    DiskCacheDigest digest{};
    for (size_t i = 0; i < digest.size() / sizeof(uint64_t); ++i) {
      StoreLittleEndian64(state_[i], digest.data() + i * sizeof(uint64_t));
    }
    return digest;
  }

 private:
  static void Mix(uint64_t* a, uint64_t* b, uint64_t* c, uint64_t* d, uint64_t x, uint64_t y) {
    *a = *a + *b + x;
    *d = RotateRight(*d ^ *a, 32);
    *c += *d;
    *b = RotateRight(*b ^ *c, 24);
    *a = *a + *b + y;
    *d = RotateRight(*d ^ *a, 16);
    *c += *d;
    *b = RotateRight(*b ^ *c, 63);
  }

  void AddCount(size_t bytes) {
    const uint64_t previous = byte_count_low_;
    byte_count_low_ += bytes;
    if (byte_count_low_ < previous) {
      ++byte_count_high_;
    }
  }

  void Compress(bool last) {
    static constexpr uint64_t kInitializationVector[8] = {
        0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
        0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL};
    static constexpr uint8_t kPermutation[12][16] = {
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
        {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
        {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
        {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
        {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
        {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
        {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
        {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
        {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
        {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
        {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}};

    uint64_t message[16];
    uint64_t work[16];
    for (size_t i = 0; i < 16; ++i) {
      message[i] = LoadLittleEndian64(buffer_.data() + i * sizeof(uint64_t));
    }
    for (size_t i = 0; i < 8; ++i) {
      work[i] = state_[i];
      work[i + 8] = kInitializationVector[i];
    }
    work[12] ^= byte_count_low_;
    work[13] ^= byte_count_high_;
    if (last) {
      work[14] = ~work[14];
    }

    for (size_t round = 0; round < 12; ++round) {
      const uint8_t* p = kPermutation[round];
      Mix(&work[0], &work[4], &work[8], &work[12], message[p[0]], message[p[1]]);
      Mix(&work[1], &work[5], &work[9], &work[13], message[p[2]], message[p[3]]);
      Mix(&work[2], &work[6], &work[10], &work[14], message[p[4]], message[p[5]]);
      Mix(&work[3], &work[7], &work[11], &work[15], message[p[6]], message[p[7]]);
      Mix(&work[0], &work[5], &work[10], &work[15], message[p[8]], message[p[9]]);
      Mix(&work[1], &work[6], &work[11], &work[12], message[p[10]], message[p[11]]);
      Mix(&work[2], &work[7], &work[8], &work[13], message[p[12]], message[p[13]]);
      Mix(&work[3], &work[4], &work[9], &work[14], message[p[14]], message[p[15]]);
    }
    for (size_t i = 0; i < 8; ++i) {
      state_[i] ^= work[i] ^ work[i + 8];
    }
  }

  std::array<uint64_t, 8> state_{};
  std::array<uint8_t, kBlake2bBlockSize> buffer_{};
  size_t buffer_size_ = 0;
  uint64_t byte_count_low_ = 0;
  uint64_t byte_count_high_ = 0;
};

DiskCacheDigest DigestBytes(const void* data, size_t size) {
  Blake2b256 hasher;
  if (size != 0) {
    hasher.Update(data, size);
  }
  return hasher.Final();
}

void HashString(Blake2b256* hasher, const std::string& value) {
  hasher->UpdateUint64(value.size());
  hasher->Update(value.data(), value.size());
}

std::string DigestToHex(const DiskCacheDigest& digest) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string text(digest.size() * 2, '0');
  for (size_t i = 0; i < digest.size(); ++i) {
    text[i * 2] = kHex[digest[i] >> 4];
    text[i * 2 + 1] = kHex[digest[i] & 0xf];
  }
  return text;
}

struct ComgrData {
  uint64_t handle;
};

struct ComgrHotswapRewriteOptions {
  size_t size;
  uint64_t flags;
};

constexpr int kComgrStatusSuccess = 0;
constexpr int kComgrDataKindExecutable = 0x8;
constexpr uint64_t kComgrHotswapRewriteFlagEntryTrampolines = 0x1;
constexpr uint64_t kComgrHotswapRewriteFlagStrictMode = 0x2;

struct ComgrApi {
  os::LibHandle lib = nullptr;
  DiskCacheDigest disk_cache_fingerprint{};
  bool has_disk_cache_fingerprint = false;
  int (*create_data)(int kind, ComgrData* data) = nullptr;
  int (*release_data)(ComgrData data) = nullptr;
  int (*set_data)(ComgrData data, size_t size, const char* bytes) = nullptr;
  int (*get_data)(ComgrData data, size_t* size, char* bytes) = nullptr;
  int (*get_data_isa_name)(ComgrData data, size_t* size, char* isa_name) = nullptr;
  int (*get_cache_identifier)(const char** identifier) = nullptr;
  int (*hotswap_rewrite)(ComgrData input, const char* source_isa_name, const char* target_isa_name,
                         ComgrData* output) = nullptr;
  int (*hotswap_rewrite_with_options)(ComgrData input, const char* source_isa_name,
                                      const char* target_isa_name,
                                      const ComgrHotswapRewriteOptions* rewrite_options,
                                      ComgrData* output) = nullptr;
};

template <typename T> bool ResolveComgrSymbol(os::LibHandle lib, const char* name, T* symbol) {
  *symbol = reinterpret_cast<T>(os::GetExportAddress(lib, name));
  return *symbol != nullptr;
}

bool ResolveComgrApi(os::LibHandle lib, ComgrApi* api) {
  api->lib = lib;
  const bool base_api_ready =
      ResolveComgrSymbol(lib, "amd_comgr_create_data", &api->create_data) &&
      ResolveComgrSymbol(lib, "amd_comgr_release_data", &api->release_data) &&
      ResolveComgrSymbol(lib, "amd_comgr_set_data", &api->set_data) &&
      ResolveComgrSymbol(lib, "amd_comgr_get_data", &api->get_data) &&
      ResolveComgrSymbol(lib, "amd_comgr_get_data_isa_name", &api->get_data_isa_name) &&
      ResolveComgrSymbol(lib, "amd_comgr_hotswap_rewrite", &api->hotswap_rewrite);
  if (!base_api_ready) {
    return false;
  }

  ResolveComgrSymbol(lib, "amd_comgr_hotswap_rewrite_with_options",
                     &api->hotswap_rewrite_with_options);
  ResolveComgrSymbol(lib, "amd_comgr_get_cache_identifier", &api->get_cache_identifier);
  return true;
}

std::string GetRuntimeLibraryDirectory() {
#if defined(_WIN32) || defined(_WIN64)
  HMODULE module = nullptr;
  if (!GetModuleHandleExA(
          GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
          reinterpret_cast<LPCSTR>(&GetRuntimeLibraryDirectory), &module)) {
    return {};
  }

  char path[MAX_PATH] = {};
  const DWORD len = GetModuleFileNameA(module, path, sizeof(path));
  if (len == 0 || len >= sizeof(path)) {
    return {};
  }

  std::string path_str(path);
  const std::string::size_type slash = path_str.find_last_of("\\/");
  return slash == std::string::npos ? std::string{} : path_str.substr(0, slash);
#else
  Dl_info info = {};
  if (dladdr(reinterpret_cast<void*>(&GetRuntimeLibraryDirectory), &info) == 0 || !info.dli_fname ||
      info.dli_fname[0] == '\0') {
    return {};
  }

  std::string path(info.dli_fname);
  const std::string::size_type slash = path.find_last_of('/');
  return slash == std::string::npos ? std::string{} : path.substr(0, slash);
#endif
}

std::vector<std::string> GetComgrLibraryCandidates() {
  std::vector<std::string> names;
  const std::string runtime_dir = GetRuntimeLibraryDirectory();
  if (!runtime_dir.empty()) {
#if defined(_WIN32) || defined(_WIN64)
    names.push_back(runtime_dir + "\\amd_comgr.dll");
    names.push_back(runtime_dir + "\\..\\bin\\amd_comgr.dll");
#else
    names.push_back(runtime_dir + "/libamd_comgr.so.3");
    names.push_back(runtime_dir + "/libamd_comgr.so");
    names.push_back(runtime_dir + "/../lib/libamd_comgr.so.3");
    names.push_back(runtime_dir + "/../lib/libamd_comgr.so");
    names.push_back(runtime_dir + "/../lib64/libamd_comgr.so.3");
    names.push_back(runtime_dir + "/../lib64/libamd_comgr.so");
#endif
  }

#if defined(_WIN32) || defined(_WIN64)
  names.push_back("amd_comgr.dll");
#else
  names.push_back("libamd_comgr.so.3");
  names.push_back("libamd_comgr.so");
#endif
  return names;
}

ComgrApi* GetComgrApi() {
  static std::once_flag once;
  static ComgrApi api;
  static bool ready = false;

  std::call_once(once, [] {
    auto try_load_comgr_api = [](const char* name) {
      if (!name || name[0] == '\0') {
        return false;
      }

      os::LibHandle lib = os::LoadLib(name);
      if (!lib) {
        return false;
      }

      if (ResolveComgrApi(lib, &api)) {
        ready = true;
#if !defined(_WIN32) && !defined(_WIN64)
        Dl_info info = {};
        g_comgr_lib_path =
            (dladdr(reinterpret_cast<void*>(api.create_data), &info) != 0 &&
             info.dli_fname && info.dli_fname[0] != '\0')
                ? info.dli_fname
                : name;
#else
        g_comgr_lib_path = name;
#endif
        const char* cache_identifier = nullptr;
        if (api.get_cache_identifier &&
            api.get_cache_identifier(&cache_identifier) == kComgrStatusSuccess &&
            cache_identifier && cache_identifier[0] != '\0') {
          api.disk_cache_fingerprint = DigestBytes(cache_identifier, std::strlen(cache_identifier));
          api.has_disk_cache_fingerprint = true;
        }
        HOTSWAP_LOG("hotswap: loaded COMGR from %s (cache identifier %s)\n",
                    g_comgr_lib_path.c_str(),
                    api.has_disk_cache_fingerprint ? "available" : "unavailable");
        return true;
      }

      os::CloseLib(lib);
      api = ComgrApi{};
      return false;
    };

    for (const std::string& name : GetComgrLibraryCandidates()) {
      if (try_load_comgr_api(name.c_str())) {
        return;
      }
    }
    HOTSWAP_LOG("hotswap: COMGR hotswap entry points unavailable\n");
  });

  return ready ? &api : nullptr;
}

const char* Gfx1250SteppingFeature(Gfx1250Stepping stepping) {
  return stepping == Gfx1250Stepping::kB0 ? kGfx1250B0Feature
                                          : kGfx1250A0Feature;
}

std::string WithGfx1250SteppingFeature(const std::string& isa_name,
                                       Gfx1250Stepping stepping) {
  if (ExtractGfxTarget(isa_name) != kGfx1250 ||
      isa_name.find(kGfx1250B0Feature) != std::string::npos ||
      isa_name.find(kGfx1250A0Feature) != std::string::npos) {
    return isa_name;
  }
  return isa_name + Gfx1250SteppingFeature(stepping);
}

bool HasCandidateHotswapRewrite(const AgentGfxRevision& gfx,
                                const RewriteOptions& options) {
  return IsHotswapSupportedGfxRevision(gfx) ||
      (options.strict_mode_enabled && gfx.gfx_target == kGfx1250) ||
      (options.entry_trampolines_enabled && IsGfx12_5Target(gfx.gfx_target));
}

std::optional<RewriteDecision> DecideHotswapRewrite(
    const AgentGfxRevision& gfx, const std::string& source_isa,
    const std::string& target_isa, const RewriteOptions& options) {
  if (source_isa.empty() || target_isa.empty()) {
    return std::nullopt;
  }

  const std::string source_gfx = ExtractGfxTarget(source_isa);
  const std::string target_gfx = ExtractGfxTarget(target_isa);
  if (IsHotswapSupportedGfxRevision(gfx) && source_gfx == kGfx1250 &&
      target_gfx == kGfx1250) {
    // Keep A0 retargeting on COMGR's legacy rewrite path. The B0 source and A0
    // target ISA features select the required instruction patches without
    // strict mode; B0 strict rewrites use hotswap_rewrite_with_options().
    RewriteDecision decision;
    decision.source_isa =
        WithGfx1250SteppingFeature(source_isa, Gfx1250Stepping::kB0);
    decision.target_isa =
        WithGfx1250SteppingFeature(target_isa, Gfx1250Stepping::kA0);
    // Preserve the legacy A0 fallback when COMGR cannot rewrite a code object.
    // Required behavior remains opt-in through strict mode for non-A0 targets.
    decision.rewrite_required = false;
    return decision;
  }

  const bool request_entry_trampolines = options.entry_trampolines_enabled &&
      IsGfx12_5Target(gfx.gfx_target) && IsGfx12_5Target(source_gfx);
  const bool request_strict_mode =
      options.strict_mode_enabled && gfx.gfx_target == kGfx1250 &&
      source_gfx == kGfx1250;
  if (!request_entry_trampolines && !request_strict_mode) {
    return std::nullopt;
  }

  RewriteDecision decision;
  decision.source_isa = source_isa;
  decision.target_isa = source_isa;
  decision.request_entry_trampolines = request_entry_trampolines;
  decision.request_strict_mode = request_strict_mode;
  decision.rewrite_required = request_strict_mode;
  if (source_gfx == kGfx1250) {
    decision.source_isa =
        WithGfx1250SteppingFeature(source_isa, Gfx1250Stepping::kB0);
    decision.target_isa =
        WithGfx1250SteppingFeature(source_isa, Gfx1250Stepping::kB0);
  }
  return decision;
}


DiskCacheDigest ComputeRetargetCacheDigest(const void* elf_data, size_t elf_size,
                                           const std::string& source_isa,
                                           const std::string& target_isa, bool entry_trampolines,
                                           bool strict_mode,
                                           const DiskCacheDigest& comgr_fingerprint) {
  static constexpr char kDomain[] = "ROCR-HOTSWAP-RETARGET-CACHE-V2";
  Blake2b256 hasher;
  hasher.Update(kDomain, sizeof(kDomain) - 1);
  hasher.Update(comgr_fingerprint.data(), comgr_fingerprint.size());
  hasher.UpdateUint64(elf_size);
  hasher.Update(elf_data, elf_size);
  HashString(&hasher, source_isa);
  HashString(&hasher, target_isa);
  hasher.UpdateUint64(entry_trampolines ? 1 : 0);
  hasher.UpdateUint64(strict_mode ? 1 : 0);
  return hasher.Final();
}


#if !defined(_WIN32) && !defined(_WIN64)
#define HOTSWAP_DISK_CACHE_SUPPORTED 1
#else
#define HOTSWAP_DISK_CACHE_SUPPORTED 0
#endif

#if HOTSWAP_DISK_CACHE_SUPPORTED

constexpr char kDiskCacheMagic[8] = {'H', 'S', 'H', 'O', 'T', 'S', 'W', '4'};
constexpr uint32_t kDiskCacheFormatVersion = 2;

struct DiskCacheHeader {
  char magic[8];
  uint32_t format_version;
  uint32_t header_size;
  uint64_t payload_size;
  uint8_t input_digest[kDigestSize];
  uint8_t output_digest[kDigestSize];
  uint8_t comgr_fingerprint[kDigestSize];
};
static_assert(sizeof(DiskCacheHeader) == 120, "DiskCacheHeader layout must be exactly 120 bytes");

bool IsDiskCacheEnabledByEnv() {
  if (IsHotswapDisabledByEnv()) {
    return false;
  }
  return !os::IsEnvVarSet("HSA_HOTSWAP_DISK_CACHE") || IsEnvFlagEnabled("HSA_HOTSWAP_DISK_CACHE");
}

// Resolves the cache root directory, or "" if none is usable.
std::string GetDiskCacheDir() {
  if (os::IsEnvVarSet("XDG_CACHE_HOME")) {
    const std::string base = os::GetEnvVar("XDG_CACHE_HOME");
    if (!base.empty()) {
      return base + "/rocm/hotswap";
    }
  }
  if (os::IsEnvVarSet("HOME")) {
    const std::string home = os::GetEnvVar("HOME");
    if (!home.empty()) {
      return home + "/.cache/rocm/hotswap";
    }
  }
  return {};
}

bool GetComgrDiskCacheFingerprint(DiskCacheDigest* fingerprint) {
#ifdef ROCR_HOTSWAP_TESTING
  {
    std::scoped_lock lock(g_comgr_cache_fingerprint_override_mutex);
    if (g_has_comgr_cache_fingerprint_override) {
      *fingerprint = g_comgr_cache_fingerprint_override;
      return true;
    }
  }
#endif
  ComgrApi* api = GetComgrApi();
  if (!api || !api->has_disk_cache_fingerprint) {
    HOTSWAP_LOG("hotswap: disk cache disabled (COMGR cache identifier unavailable)\n");
    return false;
  }
  *fingerprint = api->disk_cache_fingerprint;
  return true;
}

// Recursive mkdir (like `mkdir -p`); tolerates existing dirs. Best-effort.
bool MakeDirs(const std::string& path) {
  if (path.empty()) {
    return false;
  }
  std::string partial;
  partial.reserve(path.size());
  for (size_t i = 0; i < path.size(); ++i) {
    partial.push_back(path[i]);
    const bool at_end = (i + 1 == path.size());
    if (path[i] == '/' || at_end) {
      if (partial == "/" || partial.empty()) {
        continue;
      }
      if (mkdir(partial.c_str(), 0755) != 0 && errno != EEXIST) {
        return false;
      }
    }
  }
  return true;
}

std::string DiskCacheSubdir(const std::string& dir, const DiskCacheDigest& comgr_fingerprint) {
  return dir + "/" + DigestToHex(comgr_fingerprint);
}

std::string DiskCachePath(const std::string& dir, const DiskCacheDigest& key,
                          const DiskCacheDigest& comgr_fingerprint) {
  return DiskCacheSubdir(dir, comgr_fingerprint) + "/" + DigestToHex(key) + ".co";
}

// Reads and validates a disk cache entry. On success returns the payload as a
// shared buffer; on any mismatch or I/O failure returns nullptr (cold miss).
std::shared_ptr<std::vector<uint8_t>> ReadDiskCache(const std::string& path,
                                                    const DiskCacheDigest& key,
                                                    const DiskCacheDigest& comgr_fingerprint) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) {
    return nullptr;
  }
  DiskCacheHeader header;
  if (std::fread(&header, sizeof(header), 1, f) != 1) {
    std::fclose(f);
    return nullptr;
  }
  if (std::memcmp(header.magic, kDiskCacheMagic, sizeof(kDiskCacheMagic)) != 0 ||
      header.format_version != kDiskCacheFormatVersion ||
      header.header_size != sizeof(DiskCacheHeader) || header.payload_size == 0 ||
      header.payload_size > static_cast<uint64_t>(SIZE_MAX) ||
      std::memcmp(header.input_digest, key.data(), key.size()) != 0 ||
      std::memcmp(header.comgr_fingerprint, comgr_fingerprint.data(), comgr_fingerprint.size()) !=
          0) {
    std::fclose(f);
    return nullptr;
  }
  // Bound the declared payload_size against the actual file size before
  // allocating, so a corrupt/garbage header cannot drive a multi-TB malloc.
  // The file must contain exactly header + payload_size bytes.
  struct stat st;
  if (fstat(fileno(f), &st) != 0 || st.st_size < 0 ||
      static_cast<uint64_t>(st.st_size) < sizeof(DiskCacheHeader) ||
      header.payload_size != static_cast<uint64_t>(st.st_size) - sizeof(DiskCacheHeader)) {
    std::fclose(f);
    return nullptr;
  }
  std::shared_ptr<std::vector<uint8_t>> bytes;
  try {
    bytes = std::make_shared<std::vector<uint8_t>>(header.payload_size);
  } catch (const std::bad_alloc&) {
    std::fclose(f);
    return nullptr;
  }
  const size_t read =
      std::fread(bytes->data(), 1, header.payload_size, f);
  std::fclose(f);
  if (read != header.payload_size) {
    return nullptr;
  }
  const DiskCacheDigest output_digest = DigestBytes(bytes->data(), bytes->size());
  if (std::memcmp(header.output_digest, output_digest.data(), output_digest.size()) != 0) {
    HOTSWAP_LOG("hotswap: disk cache payload digest mismatch for %s\n", path.c_str());
    return nullptr;
  }
  return bytes;
}

// Writes an entry atomically: header + payload to a unique temp file, then
// rename() into place. Best-effort; removes the temp on any failure.
bool WriteDiskCache(const std::string& dir, const DiskCacheDigest& key,
                    const DiskCacheDigest& comgr_fingerprint, const void* payload,
                    size_t payload_size) {
  if (!payload || payload_size == 0) {
    return false;
  }
  const std::string subdir = DiskCacheSubdir(dir, comgr_fingerprint);
  if (!MakeDirs(subdir)) {
    HOTSWAP_LOG("hotswap: disk cache mkdir failed for %s\n", subdir.c_str());
    return false;
  }
  const std::string final_path = DiskCachePath(dir, key, comgr_fingerprint);
  static std::atomic<uint64_t> tmp_counter{0};
  const uint64_t uniq = tmp_counter.fetch_add(1, std::memory_order_relaxed);
  char tmp[48];
  std::snprintf(tmp, sizeof(tmp), ".%d.%016llx.tmp", static_cast<int>(getpid()),
                static_cast<unsigned long long>(uniq));
  const std::string tmp_path = final_path + tmp;

  FILE* f = std::fopen(tmp_path.c_str(), "wb");
  if (f == nullptr) {
    HOTSWAP_LOG("hotswap: disk cache tmp open failed for %s\n", tmp_path.c_str());
    return false;
  }
  DiskCacheHeader header{};
  std::memcpy(header.magic, kDiskCacheMagic, sizeof(kDiskCacheMagic));
  header.format_version = kDiskCacheFormatVersion;
  header.header_size = sizeof(DiskCacheHeader);
  header.payload_size = payload_size;
  std::memcpy(header.input_digest, key.data(), key.size());
  const DiskCacheDigest output_digest = DigestBytes(payload, payload_size);
  std::memcpy(header.output_digest, output_digest.data(), output_digest.size());
  std::memcpy(header.comgr_fingerprint, comgr_fingerprint.data(), comgr_fingerprint.size());

  bool ok = std::fwrite(&header, sizeof(header), 1, f) == 1;
  if (ok) {
    ok = std::fwrite(payload, 1, payload_size, f) == payload_size;
  }
  // Flush libc buffers, then fsync the file's data to stable storage before the
  // rename publishes it, so a crash can't expose a correctly-named but
  // partially-written entry. (The directory entry from rename is not itself
  // fsync'd, so the entry's existence isn't crash-durable; that's acceptable
  // for a best-effort read-many cache — a lost entry is just a future miss.)
  if (ok) {
    ok = (std::fflush(f) == 0);
  }
  if (ok) {
    ok = (fsync(fileno(f)) == 0);
  }
  std::fclose(f);
  if (!ok) {
    std::remove(tmp_path.c_str());
    HOTSWAP_LOG("hotswap: disk cache write failed for %s\n", tmp_path.c_str());
    return false;
  }
  if (std::rename(tmp_path.c_str(), final_path.c_str()) != 0) {
    std::remove(tmp_path.c_str());
    HOTSWAP_LOG("hotswap: disk cache rename failed for %s\n", final_path.c_str());
    return false;
  }
  HOTSWAP_LOG("hotswap: disk cache stored key=%s bytes=%zu\n", DigestToHex(key).c_str(),
              payload_size);
  return true;
}

// -- Background disk writer (owned by ROCr Runtime lifecycle) -----------------
//
// A single lazily-created writer keeps persistence off the load critical path.
// Tasks alias the RetargetedElf allocation, avoiding a second output copy.
// Outstanding task and byte limits include the write currently in progress.
struct DiskWriteTask {
  std::string dir;
  DiskCacheDigest key{};
  DiskCacheDigest comgr_fingerprint{};
  std::shared_ptr<const void> payload_owner;
  size_t payload_size = 0;
};

constexpr size_t kDefaultMaxDiskWriteTasks = 16;
constexpr size_t kDefaultMaxDiskWriteBytes = 1ULL << 30;

class DiskWriter {
 public:
  ~DiskWriter() {
    RequestStop();
    WaitForStop();
  }

  bool Enqueue(DiskWriteTask task) {
    std::scoped_lock lock(mutex_);
    const uint64_t bytes = task.payload_size;
    if (!task.payload_owner || task.payload_size == 0 || stopping_ ||
        metrics_.queued_tasks >= max_tasks_ || bytes > max_bytes_ ||
        metrics_.queued_bytes > max_bytes_ - bytes) {
      RecordDropLocked(bytes);
      return false;
    }

    try {
      queue_.push_back(std::move(task));
    } catch (const std::bad_alloc&) {
      RecordDropLocked(bytes);
      HOTSWAP_LOG("hotswap: disk write enqueue OOM; dropping task\n");
      return false;
    }
    ++metrics_.queued_tasks;
    metrics_.queued_bytes += bytes;
    metrics_.peak_queued_tasks = std::max(metrics_.peak_queued_tasks, metrics_.queued_tasks);
    metrics_.peak_queued_bytes = std::max(metrics_.peak_queued_bytes, metrics_.queued_bytes);

    if (!running_) {
      try {
        thread_ = std::thread([this] { DrainLoop(); });
        running_ = true;
      } catch (const std::system_error&) {
        DropQueuedLocked();
        HOTSWAP_LOG("hotswap: disk writer thread creation failed; dropping writes\n");
        return false;
      }
    }
    cv_.notify_one();
    return true;
  }

  void RequestStop() {
    std::scoped_lock lock(mutex_);
    if (!running_) {
      return;
    }
    stopping_ = true;
    DropQueuedLocked();
    cv_.notify_all();
  }

  void WaitForStop() {
    std::scoped_lock join_lock(join_mutex_);
    RequestStop();
    if (thread_.joinable()) {
      thread_.join();
    }
    std::scoped_lock lock(mutex_);
    running_ = false;
    stopping_ = false;
    idle_cv_.notify_all();
  }

  DiskCacheMetrics Snapshot() const {
    std::scoped_lock lock(mutex_);
    return metrics_;
  }

#ifdef ROCR_HOTSWAP_TESTING
  void Configure(size_t max_tasks, size_t max_bytes) {
    std::scoped_lock lock(mutex_);
    if (!running_ && queue_.empty()) {
      max_tasks_ = max_tasks;
      max_bytes_ = max_bytes;
      metrics_ = {};
    }
  }

  void BlockWrites(bool block) {
    std::scoped_lock lock(mutex_);
    block_writes_ = block;
    cv_.notify_all();
  }

  bool WaitForIdle(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return idle_cv_.wait_for(lock, timeout, [this] { return metrics_.queued_tasks == 0; });
  }
#endif

 private:
  void RecordDropLocked(uint64_t bytes) {
    ++metrics_.dropped_tasks;
    metrics_.dropped_bytes += bytes;
  }

  void DropQueuedLocked() {
    for (const DiskWriteTask& task : queue_) {
      RecordDropLocked(task.payload_size);
      --metrics_.queued_tasks;
      metrics_.queued_bytes -= task.payload_size;
    }
    queue_.clear();
  }

  void DrainLoop() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
      cv_.wait(lock, [this] { return !queue_.empty() || stopping_; });
      if (queue_.empty()) {
        return;
      }

      DiskWriteTask task = std::move(queue_.front());
      queue_.pop_front();
#ifdef ROCR_HOTSWAP_TESTING
      cv_.wait(lock, [this] { return !block_writes_ || stopping_; });
      const bool should_write = !stopping_;
#else
      const bool should_write = true;
#endif
      lock.unlock();
      bool write_succeeded = false;
      if (should_write) {
        try {
          write_succeeded = WriteDiskCache(task.dir, task.key, task.comgr_fingerprint,
                                           task.payload_owner.get(), task.payload_size);
        } catch (const std::bad_alloc&) {
          HOTSWAP_LOG("hotswap: disk write OOM; dropping persistence result\n");
        }
      }
      lock.lock();
      --metrics_.queued_tasks;
      metrics_.queued_bytes -= task.payload_size;
      if (write_succeeded) {
        ++metrics_.completed_tasks;
        metrics_.completed_bytes += task.payload_size;
      } else {
        RecordDropLocked(task.payload_size);
      }
      idle_cv_.notify_all();
      if (stopping_) {
        return;
      }
    }
  }

  // Runtime::Release() waits outside ROCr's bootstrap lock. Serialize joins
  // in case another runtime epoch completes while an earlier write is ending.
  std::mutex join_mutex_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable idle_cv_;
  std::list<DiskWriteTask> queue_;
  std::thread thread_;
  size_t max_tasks_ = kDefaultMaxDiskWriteTasks;
  uint64_t max_bytes_ = kDefaultMaxDiskWriteBytes;
  DiskCacheMetrics metrics_{};
  bool running_ = false;
  bool stopping_ = false;
#ifdef ROCR_HOTSWAP_TESTING
  bool block_writes_ = false;
#endif
};

// ROCR is built with -fno-threadsafe-statics (see hsa-runtime/CMakeLists.txt),
// so explicit global synchronization protects lazy construction.
std::mutex g_disk_writer_guard;
std::unique_ptr<DiskWriter> g_disk_writer;

DiskWriter* GetOrCreateDiskWriter() {
  std::scoped_lock lock(g_disk_writer_guard);
  if (!g_disk_writer) {
    g_disk_writer.reset(new (std::nothrow) DiskWriter());
  }
  return g_disk_writer.get();
}

DiskWriter* GetDiskWriterIfConstructed() {
  std::scoped_lock lock(g_disk_writer_guard);
  return g_disk_writer.get();
}

#endif  // HOTSWAP_DISK_CACHE_SUPPORTED


}  // namespace

void HotswapCacheShutdown() {
#if HOTSWAP_DISK_CACHE_SUPPORTED
  if (DiskWriter* writer = GetDiskWriterIfConstructed()) {
    writer->RequestStop();
  }
#endif
}

void HotswapCacheWaitForShutdown() {
#if HOTSWAP_DISK_CACHE_SUPPORTED
  if (DiskWriter* writer = GetDiskWriterIfConstructed()) {
    writer->WaitForStop();
  }
#endif
}

DiskCacheMetrics GetDiskCacheMetrics() {
#if HOTSWAP_DISK_CACHE_SUPPORTED
  if (DiskWriter* writer = GetDiskWriterIfConstructed()) {
    return writer->Snapshot();
  }
#endif
  return {};
}

std::string GetCodeObjectIsaName(const void* elf_data, size_t elf_size) {
  ComgrApi* api = GetComgrApi();
  if (!api || !elf_data || elf_size == 0) {
    return {};
  }

  ComgrData data = {};
  if (api->create_data(kComgrDataKindExecutable, &data) != kComgrStatusSuccess) {
    return {};
  }

  std::string isa;
  if (api->set_data(data, elf_size, static_cast<const char*>(elf_data)) == kComgrStatusSuccess) {
    size_t isa_len = 0;
    if (api->get_data_isa_name(data, &isa_len, nullptr) == kComgrStatusSuccess && isa_len > 0) {
      isa.resize(isa_len);
      if (api->get_data_isa_name(data, &isa_len, isa.data()) == kComgrStatusSuccess) {
        if (!isa.empty() && isa.back() == '\0') {
          isa.pop_back();
        }
      } else {
        isa.clear();
      }
    }
  }

  api->release_data(data);
  return isa;
}

namespace {

bool IsAgentEligibleForHotswap(const AgentGfxRevision& gfx,
                               const RewriteOptions& options) {
  HOTSWAP_LOG("hotswap: agent gfx=%s asic_revision=%u (valid=%s)\n",
              gfx.gfx_target.empty() ? "?" : gfx.gfx_target.c_str(), gfx.asic_revision,
              gfx.has_asic_revision ? "yes" : "no");
  return HasCandidateHotswapRewrite(gfx, options);
}

void LogRewrittenCodeObjectLoadFailure(hsa_status_t status) {
  HOTSWAP_LOG(
      "hotswap: rewritten load failed (status=%d), falling back to "
      "original code object\n",
      static_cast<int>(status));
}

void LogRequiredRewriteFailure() {
  HOTSWAP_LOG("hotswap: required rewrite failed, not falling back to original "
              "code object\n");
}

void LogRequiredRewrittenLoadFailure(hsa_status_t status) {
  HOTSWAP_LOG("hotswap: required rewritten load failed (status=%d), not falling "
              "back to original code object\n",
              static_cast<int>(status));
}

}  // namespace

RetargetOperationResult RetargetCodeObject(const void* elf_data, size_t elf_size,
                                           const char* source_isa, const char* target_isa,
                                           bool request_entry_trampolines,
                                           bool request_strict_mode,
                                           SourceSnapshotRef source_snapshot) {
  if (!elf_data || elf_size == 0 || !source_isa || !target_isa) {
    return {{}, RetargetError::kInvalidArgument};
  }

  ComgrApi* api = GetComgrApi();
  if (!api) {
    return {{}, RetargetError::kComgrUnavailable};
  }

  ComgrData input = {};
  if (api->create_data(kComgrDataKindExecutable, &input) != kComgrStatusSuccess) {
    return {{}, RetargetError::kComgrFailure};
  }

  if (api->set_data(input, elf_size, static_cast<const char*>(elf_data)) != kComgrStatusSuccess) {
    api->release_data(input);
    return {{}, RetargetError::kComgrFailure};
  }

  ComgrData output = {};
  int status = kComgrStatusSuccess;
  const uint64_t rewrite_flags =
      (request_entry_trampolines ? kComgrHotswapRewriteFlagEntryTrampolines : 0) |
      (request_strict_mode ? kComgrHotswapRewriteFlagStrictMode : 0);
  if (rewrite_flags != 0) {
    if (!api->hotswap_rewrite_with_options) {
      api->release_data(input);
      HOTSWAP_LOG("hotswap: COMGR rewrite-with-options entry point unavailable\n");
      return {{}, RetargetError::kComgrUnavailable};
    }
    const ComgrHotswapRewriteOptions options{
        sizeof(ComgrHotswapRewriteOptions),
        rewrite_flags};
    status = api->hotswap_rewrite_with_options(input, source_isa, target_isa,
                                               &options, &output);
  } else {
    status = api->hotswap_rewrite(input, source_isa, target_isa, &output);
  }
  api->release_data(input);
  if (status != kComgrStatusSuccess) {
    HOTSWAP_LOG("hotswap: COMGR rewrite failed for %s -> %s (rc=%d)\n", source_isa, target_isa,
                status);
    return {{}, RetargetError::kComgrFailure};
  }

  size_t output_size = 0;
  if (api->get_data(output, &output_size, nullptr) != kComgrStatusSuccess || output_size == 0) {
    api->release_data(output);
    return {{}, RetargetError::kComgrFailure};
  }

  OwnedElfBuffer output_buffer(std::malloc(output_size), &std::free);
  if (!output_buffer) {
    api->release_data(output);
    return {{}, RetargetError::kOutOfResources};
  }

  size_t copy_size = output_size;
  if (api->get_data(output, &copy_size, static_cast<char*>(output_buffer.get())) !=
          kComgrStatusSuccess ||
      copy_size != output_size) {
    api->release_data(output);
    return {{}, RetargetError::kComgrFailure};
  }

  api->release_data(output);
  try {
    return {std::make_shared<const RetargetedElf>(std::move(output_buffer), output_size,
                                                  std::move(source_snapshot)),
            RetargetError::kNone};
  } catch (const std::bad_alloc&) {
    return {{}, RetargetError::kOutOfResources};
  }
}

RetargetCodeObjectResult TryRetargetCodeObject(const CodeObjectView& code_object,
                                               hsa_agent_t agent) {
  if (IsHotswapDisabledByEnv() || !code_object.data || code_object.size == 0) {
    return {};
  }

  const AgentGfxRevision gfx = GetAgentGfxRevision(agent);
  RewriteOptions options;
  options.entry_trampolines_enabled = AreEntryTrampolinesRequested();
  options.strict_mode_enabled = IsStrictModeRequested();
  if (!IsAgentEligibleForHotswap(gfx, options)) {
    return {};
  }

  const std::string source_isa = GetCodeObjectIsaName(code_object.data, code_object.size);
  const std::string target_isa = GetAgentIsaName(agent);
  const std::optional<RewriteDecision> decision =
      DecideHotswapRewrite(gfx, source_isa, target_isa, options);
  if (!decision) {
    HOTSWAP_LOG("hotswap: rewrite skipped, no decision (src='%s' tgt='%s')\n",
                source_isa.c_str(), target_isa.c_str());
    return {};
  }

  const RetargetCacheKey cache_key{decision->source_isa, decision->target_isa,
                                   decision->request_entry_trampolines,
                                   decision->request_strict_mode};

  auto producer = [&](const SourceSnapshotRef& source_snapshot) -> RetargetOperationResult {
    const void* source_data = source_snapshot ? source_snapshot->data() : code_object.data;
    const size_t source_size = source_snapshot ? source_snapshot->size() : code_object.size;

    // This producer runs only on the single-flight leader, outside all cache
    // mutexes, so the disk read/write below is coalesced across waiters and
    // never serializes the in-memory cache.
#if HOTSWAP_DISK_CACHE_SUPPORTED
    std::string disk_dir;
    DiskCacheDigest comgr_fingerprint{};
    DiskCacheDigest disk_key{};
    bool disk_ok = false;
    if (IsDiskCacheEnabledByEnv() && GetComgrDiskCacheFingerprint(&comgr_fingerprint)) {
      disk_dir = GetDiskCacheDir();
      disk_ok = !disk_dir.empty();
      if (disk_ok) {
        disk_key = ComputeRetargetCacheDigest(
            source_data, source_size, decision->source_isa, decision->target_isa,
            decision->request_entry_trampolines, decision->request_strict_mode, comgr_fingerprint);
      }
    }

    // Disk hit: build a RetargetedElf from the persisted bytes, WITHOUT COMGR.
    // It must carry `source_snapshot` so GetOrCompute's snapshot-identity check
    // accepts it.
    if (disk_ok) {
      std::shared_ptr<std::vector<uint8_t>> disk_bytes = ReadDiskCache(
          DiskCachePath(disk_dir, disk_key, comgr_fingerprint), disk_key, comgr_fingerprint);
      if (disk_bytes) {
        const size_t n = disk_bytes->size();
        OwnedElfBuffer buf(std::malloc(n), &std::free);
        if (buf) {
          std::memcpy(buf.get(), disk_bytes->data(), n);
          try {
            RetargetedElfRef elf = std::make_shared<const RetargetedElf>(
                std::move(buf), n, source_snapshot);
            HOTSWAP_LOG("hotswap: disk cache hit key=%s out=%zu\n", DigestToHex(disk_key).c_str(),
                        n);
            return {std::move(elf), RetargetError::kNone};
          } catch (const std::bad_alloc&) {
            // Fall through to a fresh COMGR retarget.
          }
        }
      }
    }
#endif  // HOTSWAP_DISK_CACHE_SUPPORTED

#ifdef ROCR_HOTSWAP_TESTING
    if (g_force_retarget_code_object_failure_for_testing.load(std::memory_order_relaxed)) {
      HOTSWAP_LOG("hotswap: forcing retarget failure for test\n");
      return {{}, RetargetError::kComgrFailure};
    }
#endif
    RetargetOperationResult result =
        RetargetCodeObject(source_data, source_size, decision->source_isa.c_str(),
                           decision->target_isa.c_str(), decision->request_entry_trampolines,
                           decision->request_strict_mode, source_snapshot);

#if HOTSWAP_DISK_CACHE_SUPPORTED
    // Cold COMGR success: retain the existing immutable output allocation until
    // the bounded background write completes.
    if (result.succeeded() && disk_ok) {
      try {
        DiskWriteTask task;
        task.dir = disk_dir;
        task.key = disk_key;
        task.comgr_fingerprint = comgr_fingerprint;
        task.payload_owner = std::shared_ptr<const void>(result.elf, result.elf->data());
        task.payload_size = result.elf->size();
        if (DiskWriter* writer = GetOrCreateDiskWriter()) {
          writer->Enqueue(std::move(task));
        }
      } catch (const std::bad_alloc&) {
        // Persistence is best-effort; skip on OOM.
      }
    }
#endif  // HOTSWAP_DISK_CACHE_SUPPORTED
    return result;
  };

  ContentRetargetCache* cache = code_object.retarget_cache;
  if (!cache) {
    try {
      cache = &GetProcessRetargetCache();
    } catch (const std::bad_alloc&) {
      HOTSWAP_LOG("hotswap: process cache unavailable (OOM), rewriting without cache\n");
    }
  }

  const RetargetOperationResult operation = cache
      ? cache->GetOrCompute(code_object.data, code_object.size, code_object.reader_id, cache_key,
                            producer)
      : producer({});

  HOTSWAP_LOG(
      "hotswap: rewrite src=%s tgt=%s entry_trampolines=%d strict=%d required=%d "
      "in=%zu out=%zu changed=%d source=%d error=%d\n",
      decision->source_isa.c_str(), decision->target_isa.c_str(),
      decision->request_entry_trampolines, decision->request_strict_mode,
      decision->rewrite_required, code_object.size,
      operation.succeeded() ? operation.elf->size() : 0, operation.succeeded() ? 1 : 0,
      static_cast<int>(operation.source), static_cast<int>(operation.error));
  if (cache && IsVerboseLoggingEnabled()) {
    const RetargetCacheMetrics metrics = cache->SnapshotMetrics();
    fprintf(stderr,
            "hotswap cache: producer_calls=%llu producer_failures=%llu ready_hits=%llu "
            "cross_reader_results=%llu coalesced_results=%llu reentrant_rejections=%llu "
            "hash_bytes=%llu hash_nanoseconds=%llu exact_compare_bytes=%llu "
            "exact_compare_nanoseconds=%llu wait_nanoseconds=%llu lock_hold_nanoseconds=%llu "
            "source_snapshot_allocations=%llu source_snapshot_bytes=%llu "
            "live_source_snapshot_bytes=%llu peak_live_source_snapshot_bytes=%llu "
            "produced_output_bytes=%llu live_output_bytes=%llu peak_live_output_bytes=%llu "
            "content_bucket_entries=%zu transform_bucket_entries=%zu ready_entries=%zu "
            "in_flight_entries=%zu\n",
            static_cast<unsigned long long>(metrics.producer_calls),
            static_cast<unsigned long long>(metrics.producer_failures),
            static_cast<unsigned long long>(metrics.ready_hits),
            static_cast<unsigned long long>(metrics.cross_reader_results),
            static_cast<unsigned long long>(metrics.coalesced_results),
            static_cast<unsigned long long>(metrics.reentrant_rejections),
            static_cast<unsigned long long>(metrics.hash_bytes),
            static_cast<unsigned long long>(metrics.hash_nanoseconds),
            static_cast<unsigned long long>(metrics.exact_compare_bytes),
            static_cast<unsigned long long>(metrics.exact_compare_nanoseconds),
            static_cast<unsigned long long>(metrics.wait_nanoseconds),
            static_cast<unsigned long long>(metrics.lock_hold_nanoseconds),
            static_cast<unsigned long long>(metrics.source_snapshot_allocations),
            static_cast<unsigned long long>(metrics.source_snapshot_bytes),
            static_cast<unsigned long long>(metrics.live_source_snapshot_bytes),
            static_cast<unsigned long long>(metrics.peak_live_source_snapshot_bytes),
            static_cast<unsigned long long>(metrics.produced_output_bytes),
            static_cast<unsigned long long>(metrics.live_output_bytes),
            static_cast<unsigned long long>(metrics.peak_live_output_bytes),
            metrics.content_bucket_entries, metrics.transform_bucket_entries,
            metrics.ready_entries, metrics.in_flight_entries);
  }
  if (IsVerboseLoggingEnabled()) {
    const DiskCacheMetrics disk_metrics = GetDiskCacheMetrics();
    fprintf(stderr,
            "hotswap disk cache: queued_tasks=%llu queued_bytes=%llu "
            "peak_queued_tasks=%llu peak_queued_bytes=%llu "
            "dropped_tasks=%llu dropped_bytes=%llu completed_tasks=%llu "
            "completed_bytes=%llu\n",
            static_cast<unsigned long long>(disk_metrics.queued_tasks),
            static_cast<unsigned long long>(disk_metrics.queued_bytes),
            static_cast<unsigned long long>(disk_metrics.peak_queued_tasks),
            static_cast<unsigned long long>(disk_metrics.peak_queued_bytes),
            static_cast<unsigned long long>(disk_metrics.dropped_tasks),
            static_cast<unsigned long long>(disk_metrics.dropped_bytes),
            static_cast<unsigned long long>(disk_metrics.completed_tasks),
            static_cast<unsigned long long>(disk_metrics.completed_bytes));
  }
  if (operation.succeeded()) {
    return {RetargetCodeObjectStatus::kRewritten, decision->rewrite_required, operation.elf,
            RetargetError::kNone};
  }
  if (decision->rewrite_required) {
    return {RetargetCodeObjectStatus::kRequiredRewriteFailed, true, {}, operation.error};
  }
  return {RetargetCodeObjectStatus::kSkipped, false, {}, operation.error};
}

RetargetCodeObjectResult TryRetargetCodeObject(amd::hsa::loader::CodeObjectReaderImpl* reader,
                                               hsa_agent_t agent) {
  if (!reader) {
    return {};
  }

  CodeObjectView code_object;
  code_object.data = reader->GetCodeObjectMemory();
  code_object.size = reader->GetCodeObjectSize();
  code_object.uri = reader->GetUri();
  code_object.reader_id = reader->GetRetargetReaderId();
  return TryRetargetCodeObject(code_object, agent);
}

hsa_status_t LoadAgentCodeObjectWithHotswap(hsa_executable_t /*executable*/, hsa_agent_t agent,
                                            const CodeObjectView& code_object, const char* options,
                                            hsa_loaded_code_object_t* loaded_code_object,
                                            const LoadAgentCodeObjectCallbacks& callbacks) {
  if (!callbacks.load_original_code_object || !callbacks.load_rewritten_code_object) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  hsa_code_object_t original_code_object = {reinterpret_cast<uint64_t>(code_object.data)};

  const RetargetCodeObjectResult retarget_result = TryRetargetCodeObject(code_object, agent);
  if (retarget_result.status == RetargetCodeObjectStatus::kRewritten) {
    amd::hsa::loader::CodeObjectMemoryOwner code_object_owner(retarget_result.elf,
                                                              retarget_result.elf->data());
    hsa_code_object_t rewritten_code_object = {
        reinterpret_cast<uint64_t>(retarget_result.elf->data())};
    hsa_status_t status = callbacks.load_rewritten_code_object(
        callbacks.context, agent, rewritten_code_object, retarget_result.elf->size(),
        std::move(code_object_owner), options, code_object.uri, loaded_code_object);
    if (status == HSA_STATUS_SUCCESS) {
      return status;
    }
    if (retarget_result.rewrite_required) {
      LogRequiredRewrittenLoadFailure(status);
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
    }
    LogRewrittenCodeObjectLoadFailure(status);
  } else if (retarget_result.status ==
             RetargetCodeObjectStatus::kRequiredRewriteFailed) {
    LogRequiredRewriteFailure();
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }

  return callbacks.load_original_code_object(callbacks.context, agent, original_code_object,
                                             options, code_object.uri, loaded_code_object);
}

#ifdef ROCR_HOTSWAP_TESTING
std::optional<RewriteDecision> DecideHotswapRewriteForTesting(
    const AgentGfxRevision& gfx, const std::string& source_isa,
    const std::string& target_isa, const RewriteOptions& options) {
  return DecideHotswapRewrite(gfx, source_isa, target_isa, options);
}

bool HotswapRewriteWithOptionsAvailableForTesting() {
  ComgrApi* api = GetComgrApi();
  return api && api->hotswap_rewrite_with_options;
}

bool ComgrCacheIdentifierAvailableForTesting() {
  ComgrApi* api = GetComgrApi();
  return api && api->has_disk_cache_fingerprint;
}

void SetComgrCacheFingerprintForTesting(const DiskCacheDigest* fingerprint) {
  std::scoped_lock lock(g_comgr_cache_fingerprint_override_mutex);
  g_has_comgr_cache_fingerprint_override = fingerprint != nullptr;
  if (fingerprint) {
    g_comgr_cache_fingerprint_override = *fingerprint;
  }
}

void ForceRetargetCodeObjectFailureForTesting(bool force) {
  g_force_retarget_code_object_failure_for_testing.store(force, std::memory_order_relaxed);
}

DiskCacheDigest DigestBytesForTesting(const void* data, size_t size) {
  return DigestBytes(data, size);
}

DiskCacheDigest ComputeRetargetCacheDigestForTesting(const void* elf_data, size_t elf_size,
                                                     const std::string& source_isa,
                                                     const std::string& target_isa,
                                                     bool entry_trampolines, bool strict_mode,
                                                     const DiskCacheDigest& comgr_fingerprint) {
  return ComputeRetargetCacheDigest(elf_data, elf_size, source_isa, target_isa, entry_trampolines,
                                    strict_mode, comgr_fingerprint);
}

bool DiskCacheWriteForTesting(const std::string& dir, const DiskCacheDigest& key,
                              const DiskCacheDigest& comgr_fingerprint,
                              const std::vector<uint8_t>& payload) {
#if HOTSWAP_DISK_CACHE_SUPPORTED
  return WriteDiskCache(dir, key, comgr_fingerprint, payload.data(), payload.size());
#else
  (void)dir;
  (void)key;
  (void)comgr_fingerprint;
  (void)payload;
  return false;
#endif
}

bool DiskCacheReadForTesting(const std::string& dir, const DiskCacheDigest& key,
                             const DiskCacheDigest& comgr_fingerprint,
                             std::vector<uint8_t>* out_payload) {
#if HOTSWAP_DISK_CACHE_SUPPORTED
  const std::string path = DiskCachePath(dir, key, comgr_fingerprint);
  std::shared_ptr<std::vector<uint8_t>> bytes = ReadDiskCache(path, key, comgr_fingerprint);
  if (!bytes) {
    return false;
  }
  if (out_payload != nullptr) {
    *out_payload = *bytes;
  }
  return true;
#else
  (void)dir;
  (void)key;
  (void)comgr_fingerprint;
  (void)out_payload;
  return false;
#endif
}

std::string DiskCachePathForTesting(const std::string& dir, const DiskCacheDigest& key,
                                    const DiskCacheDigest& comgr_fingerprint) {
#if HOTSWAP_DISK_CACHE_SUPPORTED
  return DiskCachePath(dir, key, comgr_fingerprint);
#else
  (void)dir;
  (void)key;
  (void)comgr_fingerprint;
  return {};
#endif
}

bool DiskWriterConstructedForTesting() {
#if HOTSWAP_DISK_CACHE_SUPPORTED
  return GetDiskWriterIfConstructed() != nullptr;
#else
  return false;
#endif
}

void ConfigureDiskWriterForTesting(size_t max_tasks, size_t max_bytes) {
#if HOTSWAP_DISK_CACHE_SUPPORTED
  if (DiskWriter* writer = GetOrCreateDiskWriter()) {
    writer->Configure(max_tasks, max_bytes);
  }
#else
  (void)max_tasks;
  (void)max_bytes;
#endif
}

bool EnqueueDiskWriteForTesting(const std::string& dir, const DiskCacheDigest& key,
                                const DiskCacheDigest& comgr_fingerprint,
                                const std::vector<uint8_t>& payload) {
#if HOTSWAP_DISK_CACHE_SUPPORTED
  try {
    auto owner = std::make_shared<std::vector<uint8_t>>(payload);
    DiskWriteTask task;
    task.dir = dir;
    task.key = key;
    task.comgr_fingerprint = comgr_fingerprint;
    task.payload_owner = std::shared_ptr<const void>(owner, owner->data());
    task.payload_size = owner->size();
    DiskWriter* writer = GetOrCreateDiskWriter();
    return writer && writer->Enqueue(std::move(task));
  } catch (const std::bad_alloc&) {
    return false;
  }
#else
  (void)dir;
  (void)key;
  (void)comgr_fingerprint;
  (void)payload;
  return false;
#endif
}

void BlockDiskWritesForTesting(bool block) {
#if HOTSWAP_DISK_CACHE_SUPPORTED
  if (DiskWriter* writer = GetDiskWriterIfConstructed()) {
    writer->BlockWrites(block);
  }
#else
  (void)block;
#endif
}

bool WaitForDiskWriterIdleForTesting(uint64_t timeout_milliseconds) {
#if HOTSWAP_DISK_CACHE_SUPPORTED
  DiskWriter* writer = GetDiskWriterIfConstructed();
  return !writer || writer->WaitForIdle(std::chrono::milliseconds(timeout_milliseconds));
#else
  (void)timeout_milliseconds;
  return true;
#endif
}

bool DiskCacheEnabledForTesting() {
#if HOTSWAP_DISK_CACHE_SUPPORTED
  return IsDiskCacheEnabledByEnv();
#else
  return false;
#endif
}

void ResetDiskWriterForTesting() {
#if HOTSWAP_DISK_CACHE_SUPPORTED
  std::unique_ptr<DiskWriter> writer;
  {
    std::scoped_lock lock(g_disk_writer_guard);
    writer = std::move(g_disk_writer);
  }
  if (writer) {
    writer->RequestStop();
    writer->WaitForStop();
  }
#endif
}

#endif

}  // namespace hotswap
}  // namespace rocr
