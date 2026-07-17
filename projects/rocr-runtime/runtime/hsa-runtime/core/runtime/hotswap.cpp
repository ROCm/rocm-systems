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
#include <atomic>
#include <cctype>
#include <cerrno>
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
#endif

// Absolute path of the COMGR library actually loaded by GetComgrApi(), captured
// once at load time and read by the disk cache to derive its toolchain salt.
// Empty until COMGR is loaded (see EnsureComgrLoadedForSalt).
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
  int (*create_data)(int kind, ComgrData* data) = nullptr;
  int (*release_data)(ComgrData data) = nullptr;
  int (*set_data)(ComgrData data, size_t size, const char* bytes) = nullptr;
  int (*get_data)(ComgrData data, size_t* size, char* bytes) = nullptr;
  int (*get_data_isa_name)(ComgrData data, size_t* size, char* isa_name) = nullptr;
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
        // Resolve the absolute path the loader actually mapped so the disk
        // cache salt's stat() can find it (a bare soname would not resolve).
        // dladdr on a resolved symbol yields the real file; fall back to the
        // candidate name only if that fails.
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
        HOTSWAP_LOG("hotswap: loaded COMGR from %s\n", g_comgr_lib_path.c_str());
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


// -- Content hash for the disk-cache key (independent of the in-memory
//    single-flight cache's internal content hashing). --------------------
uint64_t FnvHash(const void* data, size_t size) {
  constexpr uint64_t kFnvOffset = 14695981039346656037ULL;
  constexpr uint64_t kFnvPrime = 1099511628211ULL;
  uint64_t hash = kFnvOffset;
  const auto* bytes = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= kFnvPrime;
  }
  return hash;
}

uint64_t ComputeRetargetCacheKey(const void* elf_data, size_t elf_size,
                                 const std::string& source_isa,
                                 const std::string& target_isa,
                                 bool entry_trampolines, bool strict_mode) {
  uint64_t hash = FnvHash(elf_data, elf_size);
  hash ^= FnvHash(source_isa.data(), source_isa.size()) * 31;
  hash ^= FnvHash(target_isa.data(), target_isa.size()) * 37;
  hash ^= entry_trampolines ? 0xDEADBEEF12345678ULL : 0x0ULL;
  hash ^= strict_mode ? 0x9E3779B97F4A7C15ULL : 0x0ULL;
  return hash;
}



#if !defined(_WIN32) && !defined(_WIN64)
#define HOTSWAP_DISK_CACHE_SUPPORTED 1
#else
#define HOTSWAP_DISK_CACHE_SUPPORTED 0
#endif

#if HOTSWAP_DISK_CACHE_SUPPORTED

constexpr char kDiskCacheMagic[8] = {'H', 'S', 'H', 'O', 'T', 'S', 'W', '3'};
constexpr uint32_t kDiskCacheFormatVersion = 1;

struct DiskCacheHeader {
  char magic[8];
  uint32_t format_version;
  uint32_t reserved;
  uint64_t comgr_salt;
  uint64_t payload_size;
};
// The header is written and read as a raw byte image; its layout must be
// stable. 8 + 4 + 4 + 8 + 8 = 32 with no padding on any supported ABI.
static_assert(sizeof(DiskCacheHeader) == 32,
              "DiskCacheHeader layout must be exactly 32 bytes");

bool IsDiskCacheDisabledByEnv() {
  if (!os::IsEnvVarSet("HSA_HOTSWAP_DISK_CACHE")) {
    return false;
  }
  std::string value = os::GetEnvVar("HSA_HOTSWAP_DISK_CACHE");
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value == "0" || value == "off" || value == "false" || value == "no" ||
         value == "n" || value == "f";
}

// Resolves the cache root directory, or "" if none is usable.
std::string GetDiskCacheDir() {
  if (os::IsEnvVarSet("HSA_HOTSWAP_CACHE_DIR")) {
    const std::string dir = os::GetEnvVar("HSA_HOTSWAP_CACHE_DIR");
    if (!dir.empty()) {
      return dir;
    }
  }
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

// Ensures COMGR has been loaded so g_comgr_lib_path is populated before the
// salt is derived. Without this, the first (cold) load computes the salt before
// GetComgrApi() runs, persisting under a wrong/empty-path salt that later reads
// (post-COMGR-load) never match. Returns true if a usable lib path is known.
bool EnsureComgrLoadedForSalt() {
  GetComgrApi();  // idempotent; populates g_comgr_lib_path on success
  return !g_comgr_lib_path.empty();
}

// Derives a stable per-toolchain salt from the loaded COMGR library identity.
// Returns 0 if the library path is unknown or cannot be stat'd, which callers
// treat as "disk cache unavailable this run" rather than persisting unsalted.
uint64_t ComgrIdentitySalt() {
  if (g_comgr_lib_path.empty()) {
    return 0;
  }
  struct stat st;
  if (stat(g_comgr_lib_path.c_str(), &st) != 0) {
    HOTSWAP_LOG("hotswap: disk cache disabled (cannot stat COMGR lib %s)\n",
                g_comgr_lib_path.c_str());
    return 0;
  }
  uint64_t salt = static_cast<uint64_t>(kDiskCacheFormatVersion) * 1000003ULL;
  salt ^= FnvHash(g_comgr_lib_path.data(), g_comgr_lib_path.size());
  salt ^= static_cast<uint64_t>(st.st_size) * 2654435761ULL;
  salt ^= static_cast<uint64_t>(st.st_mtime) * 40503ULL;
  // 0 is the "salt unavailable" sentinel used by callers; force any real salt
  // to be nonzero so a valid toolchain can never masquerade as unavailable.
  return salt == 0 ? 1 : salt;
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

std::string ToHex(uint64_t v) {
  char buf[17];
  std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(v));
  return std::string(buf);
}

std::string DiskCacheSubdir(const std::string& dir, uint64_t salt) {
  return dir + "/" + ToHex(salt);
}

std::string DiskCachePath(const std::string& dir, uint64_t key, uint64_t salt) {
  return DiskCacheSubdir(dir, salt) + "/" + ToHex(key) + ".co";
}

// Reads and validates a disk cache entry. On success returns the payload as a
// shared buffer; on any mismatch or I/O failure returns nullptr (cold miss).
std::shared_ptr<std::vector<uint8_t>> ReadDiskCache(const std::string& path,
                                                    uint64_t salt) {
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
      header.comgr_salt != salt || header.payload_size == 0) {
    std::fclose(f);
    return nullptr;
  }
  // Bound the declared payload_size against the actual file size before
  // allocating, so a corrupt/garbage header cannot drive a multi-TB malloc.
  // The file must contain exactly header + payload_size bytes.
  struct stat st;
  if (fstat(fileno(f), &st) != 0 ||
      static_cast<uint64_t>(st.st_size) !=
          sizeof(DiskCacheHeader) + header.payload_size) {
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
  return bytes;
}

// Writes an entry atomically: header + payload to a unique temp file, then
// rename() into place. Best-effort; removes the temp on any failure.
void WriteDiskCache(const std::string& dir, uint64_t key, uint64_t salt,
                    const std::vector<uint8_t>& payload) {
  const std::string subdir = DiskCacheSubdir(dir, salt);
  if (!MakeDirs(subdir)) {
    HOTSWAP_LOG("hotswap: disk cache mkdir failed for %s\n", subdir.c_str());
    return;
  }
  const std::string final_path = DiskCachePath(dir, key, salt);
  // Temp name is unique per (pid, key, monotonic counter) so two writers in the
  // same process (e.g. the async writer and a synchronous test hook) targeting
  // the same key never share an in-progress temp file.
  static std::atomic<uint64_t> tmp_counter{0};
  const uint64_t uniq = tmp_counter.fetch_add(1, std::memory_order_relaxed);
  // ".<pid>.<key:16>.<uniq:16>.tmp": up to 1+10+1+16+1+16+1+4 = 50 chars + NUL.
  char tmp[64];
  std::snprintf(tmp, sizeof(tmp), ".%d.%016llx.%016llx.tmp",
                static_cast<int>(getpid()),
                static_cast<unsigned long long>(key),
                static_cast<unsigned long long>(uniq));
  const std::string tmp_path = final_path + tmp;

  FILE* f = std::fopen(tmp_path.c_str(), "wb");
  if (f == nullptr) {
    HOTSWAP_LOG("hotswap: disk cache tmp open failed for %s\n", tmp_path.c_str());
    return;
  }
  DiskCacheHeader header;
  std::memcpy(header.magic, kDiskCacheMagic, sizeof(kDiskCacheMagic));
  header.format_version = kDiskCacheFormatVersion;
  header.reserved = 0;
  header.comgr_salt = salt;
  header.payload_size = payload.size();

  bool ok = std::fwrite(&header, sizeof(header), 1, f) == 1;
  if (ok && !payload.empty()) {
    ok = std::fwrite(payload.data(), 1, payload.size(), f) == payload.size();
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
    return;
  }
  if (std::rename(tmp_path.c_str(), final_path.c_str()) != 0) {
    std::remove(tmp_path.c_str());
    HOTSWAP_LOG("hotswap: disk cache rename failed for %s\n", final_path.c_str());
    return;
  }
  HOTSWAP_LOG("hotswap: disk cache stored key=0x%llx bytes=%zu\n",
              static_cast<unsigned long long>(key), payload.size());
}

// -- Background disk writer (owned by ROCr Runtime lifecycle) -----------------
//
// A single writer thread drains a queue of pending writes so the ~6s disk
// persist stays off the load critical path. Each task holds a shared_ptr to the
// payload, so an in-memory eviction before the write completes cannot lose the
// bytes. Started by HotswapCacheStartup() (Runtime::Load) and stopped by
// HotswapCacheShutdown() (Runtime::Unload) using ROCr's standard
// stop-flag + condition-variable pattern.
//
// The writer is compiled only into the real runtime, not the unit-test binary:
// it depends on ROCr's os:: thread layer (core/util/lnx/os_linux.cpp), which
// pulls in the full Runtime graph and is not linked by the standalone
// hotswap_rewrite test. Unit tests exercise the disk format synchronously via
// DiskCacheWriteForTesting/DiskCacheReadForTesting; the async writer path is
// covered by the real hsa-runtime64 build and on-device runtime testing.
struct DiskWriteTask {
  std::string dir;
  uint64_t key;
  uint64_t salt;
  std::shared_ptr<std::vector<uint8_t>> payload;
};

// Background writer that persists retarget results off the load critical path.
//
// Uses std::thread (not ROCr's os:: layer) so the identical code compiles and
// runs in both the real runtime and the standalone hotswap_rewrite unit test —
// the concurrency logic (drain-at-shutdown, enqueue/stop ordering) is therefore
// directly unit-tested rather than only exercised on device.
//
// Synchronization uses a std::condition_variable so the wait predicate
// (queue non-empty OR stopping) and every mutation of queue_/stopping_ share
// mutex_ in a single critical section. Shutdown is race-free: no wakeup can be
// lost, the cv/thread lifetimes are tied to the object, and any task enqueued
// before the thread exits is guaranteed to be drained.
class DiskWriter {
 public:
  ~DiskWriter() { Stop(); }

  void Start() {
    std::scoped_lock lock(mutex_);
    if (running_) {
      return;  // already started
    }
    stopping_ = false;
    try {
      thread_ = std::thread([this] { DrainLoop(); });
      running_ = true;
    } catch (const std::system_error&) {
      HOTSWAP_LOG("hotswap: disk writer thread creation failed; writes disabled\n");
    }
  }

  void Stop() {
    {
      std::scoped_lock lock(mutex_);
      if (!running_) {
        return;
      }
      // Signal shutdown under the lock, then notify: the drain loop's predicate
      // is evaluated while holding mutex_, so this notify cannot be lost.
      stopping_ = true;
      cv_.notify_all();
    }
    if (thread_.joinable()) {
      thread_.join();
    }
    std::scoped_lock lock(mutex_);
    running_ = false;
    stopping_ = false;  // ready for a clean restart on a later Start()
  }

  // Enqueues a write. If the writer isn't running (creation failed, not
  // started, or already stopping), silently drops the task — persistence is
  // best-effort.
  void Enqueue(DiskWriteTask task) {
    std::scoped_lock lock(mutex_);
    if (!running_ || stopping_) {
      return;
    }
    try {
      queue_.push_back(std::move(task));
    } catch (const std::bad_alloc&) {
      HOTSWAP_LOG("hotswap: disk write enqueue OOM; dropping task\n");
      return;
    }
    cv_.notify_one();
  }

 private:
  void DrainLoop() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
      // Wait until there is work or we are stopping. Predicate + wait share the
      // lock, so no enqueue/stop between the check and the wait can be missed.
      cv_.wait(lock, [this] { return !queue_.empty() || stopping_; });

      // Drain all queued tasks — even while stopping — so pending writes land.
      while (!queue_.empty()) {
        DiskWriteTask task = std::move(queue_.front());
        queue_.pop_front();
        lock.unlock();  // perform the large write without holding the lock
        WriteDiskCache(task.dir, task.key, task.salt, *task.payload);
        lock.lock();
      }

      // Exit only once the queue is fully drained AND shutdown was requested.
      // A task enqueued during the unlocked write above is caught by the
      // while-loop re-check before this test.
      if (stopping_) {
        return;
      }
    }
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  std::list<DiskWriteTask> queue_;
  std::thread thread_;
  bool running_ = false;
  bool stopping_ = false;
};

DiskWriter& GetDiskWriter() {
  static DiskWriter writer;
  return writer;
}

#endif  // HOTSWAP_DISK_CACHE_SUPPORTED


}  // namespace

void HotswapCacheStartup() {
#if HOTSWAP_DISK_CACHE_SUPPORTED
  if (IsDiskCacheDisabledByEnv()) {
    return;
  }
  GetDiskWriter().Start();
#endif
}

void HotswapCacheShutdown() {
#if HOTSWAP_DISK_CACHE_SUPPORTED
  // Always safe to call: Stop() is a no-op if the writer was never started.
  GetDiskWriter().Stop();
#endif
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
#ifdef ROCR_HOTSWAP_TESTING
    if (g_force_retarget_code_object_failure_for_testing.load(std::memory_order_relaxed)) {
      HOTSWAP_LOG("hotswap: forcing retarget failure for test\n");
      return {{}, RetargetError::kComgrFailure};
    }
#endif
    const void* source_data = source_snapshot ? source_snapshot->data() : code_object.data;
    const size_t source_size = source_snapshot ? source_snapshot->size() : code_object.size;

    // This producer runs only on the single-flight leader, outside all cache
    // mutexes, so the disk read/write below is coalesced across waiters and
    // never serializes the in-memory cache.
#if HOTSWAP_DISK_CACHE_SUPPORTED
    const uint64_t disk_key = ComputeRetargetCacheKey(
        source_data, source_size, decision->source_isa, decision->target_isa,
        decision->request_entry_trampolines, decision->request_strict_mode);
    std::string disk_dir;
    uint64_t salt = 0;
    bool disk_ok = false;
    if (!IsDiskCacheDisabledByEnv() && EnsureComgrLoadedForSalt()) {
      disk_dir = GetDiskCacheDir();
      salt = ComgrIdentitySalt();
      disk_ok = (!disk_dir.empty() && salt != 0);
    }

    // Disk hit: build a RetargetedElf from the persisted bytes, WITHOUT COMGR.
    // It must carry `source_snapshot` so GetOrCompute's snapshot-identity check
    // accepts it.
    if (disk_ok) {
      std::shared_ptr<std::vector<uint8_t>> disk_bytes =
          ReadDiskCache(DiskCachePath(disk_dir, disk_key, salt), salt);
      if (disk_bytes) {
        const size_t n = disk_bytes->size();
        OwnedElfBuffer buf(std::malloc(n), &std::free);
        if (buf) {
          std::memcpy(buf.get(), disk_bytes->data(), n);
          try {
            RetargetedElfRef elf = std::make_shared<const RetargetedElf>(
                std::move(buf), n, source_snapshot);
            HOTSWAP_LOG("hotswap: disk cache hit key=0x%llx out=%zu\n",
                        (unsigned long long)disk_key, n);
            return {std::move(elf), RetargetError::kNone};
          } catch (const std::bad_alloc&) {
            // Fall through to a fresh COMGR retarget.
          }
        }
      }
    }
#endif  // HOTSWAP_DISK_CACHE_SUPPORTED

    RetargetOperationResult result =
        RetargetCodeObject(source_data, source_size, decision->source_isa.c_str(),
                           decision->target_isa.c_str(), decision->request_entry_trampolines,
                           decision->request_strict_mode, source_snapshot);

#if HOTSWAP_DISK_CACHE_SUPPORTED
    // Cold COMGR success: persist to disk on the background writer. The task
    // owns its own copy of the bytes so its lifetime is independent of the
    // RetargetedElf and the cache.
    if (result.succeeded() && disk_ok) {
      const auto* ebytes = static_cast<const uint8_t*>(result.elf->data());
      const size_t esize = result.elf->size();
      try {
        DiskWriteTask task;
        task.dir = disk_dir;
        task.key = disk_key;
        task.salt = salt;
        task.payload =
            std::make_shared<std::vector<uint8_t>>(ebytes, ebytes + esize);
        GetDiskWriter().Enqueue(std::move(task));
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

void ForceRetargetCodeObjectFailureForTesting(bool force) {
  g_force_retarget_code_object_failure_for_testing.store(force, std::memory_order_relaxed);
}

// Synchronously writes a disk cache entry under `dir` (bypasses the async
// writer) so the disk format/salt/atomic-publish path can be tested directly.
// Returns false if disk cache support is compiled out.
bool DiskCacheWriteForTesting(const std::string& dir, uint64_t key,
                              uint64_t salt,
                              const std::vector<uint8_t>& payload) {
#if HOTSWAP_DISK_CACHE_SUPPORTED
  WriteDiskCache(dir, key, salt, payload);
  return true;
#else
  (void)dir; (void)key; (void)salt; (void)payload;
  return false;
#endif
}

// Reads a disk cache entry written under `dir`. Returns true and fills
// `out_payload` on a validated hit; false on miss/mismatch/unsupported.
bool DiskCacheReadForTesting(const std::string& dir, uint64_t key, uint64_t salt,
                            std::vector<uint8_t>* out_payload) {
#if HOTSWAP_DISK_CACHE_SUPPORTED
  const std::string path = DiskCachePath(dir, key, salt);
  std::shared_ptr<std::vector<uint8_t>> bytes = ReadDiskCache(path, salt);
  if (!bytes) {
    return false;
  }
  if (out_payload != nullptr) {
    *out_payload = *bytes;
  }
  return true;
#else
  (void)dir; (void)key; (void)salt; (void)out_payload;
  return false;
#endif
}

// Drives the background DiskWriter directly: starts it, enqueues `n` writes of
// `payload` under `dir` (keys 0..n-1, salt fixed), then Stop() which must drain
// all pending writes before joining. Returns the number of entries that are
// readable back afterward (should equal `n` if the drain-at-shutdown path is
// correct). Exercises the real async writer concurrency logic.
int DiskWriterDrainRoundTripForTesting(const std::string& dir, int n,
                                       const std::vector<uint8_t>& payload) {
#if HOTSWAP_DISK_CACHE_SUPPORTED
  constexpr uint64_t kSalt = 0xA5A5A5A5A5A5A5A5ULL;
  DiskWriter& writer = GetDiskWriter();
  writer.Start();
  for (int i = 0; i < n; ++i) {
    DiskWriteTask task;
    task.dir = dir;
    task.key = static_cast<uint64_t>(i);
    task.salt = kSalt;
    task.payload = std::make_shared<std::vector<uint8_t>>(payload);
    writer.Enqueue(std::move(task));
  }
  writer.Stop();  // must drain every enqueued task before returning
  int found = 0;
  for (int i = 0; i < n; ++i) {
    if (ReadDiskCache(DiskCachePath(dir, static_cast<uint64_t>(i), kSalt),
                      kSalt) != nullptr) {
      ++found;
    }
  }
  return found;
#else
  (void)dir; (void)n; (void)payload;
  return -1;
#endif
}

#endif

}  // namespace hotswap
}  // namespace rocr
