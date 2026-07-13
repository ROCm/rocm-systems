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
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
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

std::mutex g_retained_rewritten_elf_buffers_mutex;
std::unordered_map<uint64_t, std::vector<OwnedElfBuffer>> g_retained_rewritten_elf_buffers;

// -- Per-code-object retarget cache -------------------------------------------
//
// Caches the output of RetargetCodeObject keyed by (code object content,
// source ISA, target ISA, entry trampoline flag). When the same code object
// is loaded for multiple GPUs of the same stepping, the COMGR retarget runs
// once; subsequent loads return a shared reference to the cached result.
// Only deterministic failures (COMGR returned an error) are cached; transient
// allocation failures are not, so a later attempt can succeed.

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
                                 bool entry_trampolines) {
  uint64_t hash = FnvHash(elf_data, elf_size);
  hash ^= FnvHash(source_isa.data(), source_isa.size()) * 31;
  hash ^= FnvHash(target_isa.data(), target_isa.size()) * 37;
  hash ^= entry_trampolines ? 0xDEADBEEF12345678ULL : 0x0ULL;
  return hash;
}

struct CachedRetargetResult {
  bool succeeded = false;
  // Ref-counted so callers can grab a cheap handle under the mutex
  // and copy into the output buffer after releasing it.
  std::shared_ptr<std::vector<uint8_t>> elf_bytes;
};

constexpr size_t kMaxRetargetCacheEntries = 256;

std::mutex g_retarget_cache_mutex;
std::unordered_map<uint64_t, CachedRetargetResult> g_retarget_cache;

// Path of the COMGR library that actually loaded; set by GetComgrApi. Folded
// into the disk-cache salt so a toolchain swap invalidates persisted entries.
std::string g_comgr_lib_path;

// -- Disk-persistent tier for the retarget cache ------------------------------
//
// The in-memory cache above only lives for the current process, so every fresh
// launch re-pays the full COMGR retarget (dominated by disassembling a large
// .text). This tier persists the retargeted ELF to disk keyed by the same
// content hash, so repeat runs -- and unrelated processes -- skip COMGR
// entirely. Entries are namespaced by a salt derived from the loaded COMGR
// library (path + size + mtime) plus a format version, so a toolchain change
// naturally invalidates stale results. POSIX-only; a no-op on Windows.

constexpr char kDiskCacheMagic[8] = {'H', 'S', 'H', 'O', 'T', 'S', 'W', '2'};
constexpr uint32_t kDiskCacheFormatVersion = 1;

struct DiskCacheHeader {
  char magic[8];
  uint32_t format_version;
  uint32_t reserved;
  uint64_t comgr_salt;
  uint64_t payload_size;
};

bool IsDiskCacheDisabledByEnv() {
  // Default-on; only an explicit false-like value opts out.
  if (!os::IsEnvVarSet("HSA_HOTSWAP_DISK_CACHE")) {
    return false;
  }
  std::string v = os::GetEnvVar("HSA_HOTSWAP_DISK_CACHE");
  std::transform(v.begin(), v.end(), v.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return v == "0" || v == "off" || v == "false" || v == "no" || v == "n" || v == "f";
}

std::string GetDiskCacheDir() {
#if defined(_WIN32) || defined(_WIN64)
  return {};
#else
  if (IsDiskCacheDisabledByEnv()) {
    return {};
  }
  if (os::IsEnvVarSet("HSA_HOTSWAP_CACHE_DIR")) {
    const std::string d = os::GetEnvVar("HSA_HOTSWAP_CACHE_DIR");
    if (!d.empty()) {
      return d;
    }
  }
  if (os::IsEnvVarSet("XDG_CACHE_HOME")) {
    const std::string d = os::GetEnvVar("XDG_CACHE_HOME");
    if (!d.empty()) {
      return d + "/rocm/hotswap";
    }
  }
  if (os::IsEnvVarSet("HOME")) {
    const std::string h = os::GetEnvVar("HOME");
    if (!h.empty()) {
      return h + "/.cache/rocm/hotswap";
    }
  }
  return {};
#endif
}

#if !defined(_WIN32) && !defined(_WIN64)
uint64_t ComgrIdentitySalt() {
  uint64_t salt = static_cast<uint64_t>(kDiskCacheFormatVersion) * 1000003ULL;
  if (!g_comgr_lib_path.empty()) {
    salt ^= FnvHash(g_comgr_lib_path.data(), g_comgr_lib_path.size());
    struct stat st = {};
    if (::stat(g_comgr_lib_path.c_str(), &st) == 0) {
      salt ^= static_cast<uint64_t>(st.st_size) * 2654435761ULL;
      salt ^= static_cast<uint64_t>(st.st_mtime) * 40503ULL;
    }
  }
  return salt;
}

bool MakeDirs(const std::string& path) {
  if (path.empty()) {
    return false;
  }
  for (size_t i = 1; i < path.size(); ++i) {
    if (path[i] == '/') {
      const std::string sub = path.substr(0, i);
      if (::mkdir(sub.c_str(), 0755) != 0 && errno != EEXIST) {
        return false;
      }
    }
  }
  return ::mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

std::string DiskCacheSubdir(const std::string& dir, uint64_t salt) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "/%016llx", static_cast<unsigned long long>(salt));
  return dir + buf;
}

std::string DiskCachePath(const std::string& dir, uint64_t key, uint64_t salt) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "/%016llx.co", static_cast<unsigned long long>(key));
  return DiskCacheSubdir(dir, salt) + buf;
}

std::shared_ptr<std::vector<uint8_t>> ReadDiskCache(const std::string& path, uint64_t salt) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    return nullptr;
  }
  std::shared_ptr<std::vector<uint8_t>> result;
  DiskCacheHeader hdr = {};
  if (std::fread(&hdr, sizeof(hdr), 1, f) == 1 &&
      std::memcmp(hdr.magic, kDiskCacheMagic, sizeof(hdr.magic)) == 0 &&
      hdr.format_version == kDiskCacheFormatVersion && hdr.comgr_salt == salt &&
      hdr.payload_size > 0) {
    try {
      auto bytes = std::make_shared<std::vector<uint8_t>>(hdr.payload_size);
      if (std::fread(bytes->data(), 1, hdr.payload_size, f) == hdr.payload_size) {
        result = std::move(bytes);
      }
    } catch (const std::bad_alloc&) {
      result = nullptr;
    }
  }
  std::fclose(f);
  return result;
}

void WriteDiskCache(const std::string& dir, uint64_t key, uint64_t salt, const void* data,
                    size_t size) {
  if (dir.empty() || !data || size == 0) {
    return;
  }
  if (!MakeDirs(DiskCacheSubdir(dir, salt))) {
    return;
  }
  const std::string final_path = DiskCachePath(dir, key, salt);
  char suffix[64];
  std::snprintf(suffix, sizeof(suffix), ".%d.%llu.tmp", static_cast<int>(::getpid()),
                static_cast<unsigned long long>(key));
  const std::string tmp_path = final_path + suffix;

  std::FILE* f = std::fopen(tmp_path.c_str(), "wb");
  if (!f) {
    return;
  }
  DiskCacheHeader hdr = {};
  std::memcpy(hdr.magic, kDiskCacheMagic, sizeof(hdr.magic));
  hdr.format_version = kDiskCacheFormatVersion;
  hdr.reserved = 0;
  hdr.comgr_salt = salt;
  hdr.payload_size = size;
  const bool ok = std::fwrite(&hdr, sizeof(hdr), 1, f) == 1 &&
                  std::fwrite(data, 1, size, f) == size;
  std::fclose(f);

  // Atomic publish: a partial writer never exposes a truncated entry.
  if (!ok || ::rename(tmp_path.c_str(), final_path.c_str()) != 0) {
    ::remove(tmp_path.c_str());
  }
}
#endif  // !_WIN32 && !_WIN64

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

bool IsGfx12_5RewriteRequested() {
  constexpr char kEnvName[] = "AMD_COMGR_HOTSWAP_ENTRY_TRAMPOLINES";
  // Default-on policy owned by ROCR: only literal "0" opts out.
  return !os::IsEnvVarSet(kEnvName) || os::GetEnvVar(kEnvName) != "0";
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
        g_comgr_lib_path = name;
        HOTSWAP_LOG("hotswap: loaded COMGR from %s\n", name);
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
         (options.gfx12_5_rewrite_enabled &&
          IsGfx12_5Target(gfx.gfx_target));
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
    return RewriteDecision{
        WithGfx1250SteppingFeature(source_isa, Gfx1250Stepping::kB0),
        WithGfx1250SteppingFeature(target_isa, Gfx1250Stepping::kA0),
        false};
  }

  if (!options.gfx12_5_rewrite_enabled ||
      !IsGfx12_5Target(gfx.gfx_target) || !IsGfx12_5Target(source_gfx)) {
    return std::nullopt;
  }

  RewriteDecision decision{source_isa, source_isa, true};
  if (source_gfx == kGfx1250) {
    decision.source_isa =
        WithGfx1250SteppingFeature(source_isa, Gfx1250Stepping::kB0);
    decision.target_isa =
        WithGfx1250SteppingFeature(source_isa, Gfx1250Stepping::kB0);
  }
  return decision;
}

}  // namespace

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

}  // namespace

bool RetargetCodeObject(const void* elf_data, size_t elf_size, const char* source_isa,
                        const char* target_isa, OwnedElfBuffer* out_elf_buffer,
                        size_t* out_elf_size, bool request_entry_trampolines) {
  ComgrApi* api = GetComgrApi();
  if (!api || !elf_data || elf_size == 0 || !source_isa || !target_isa || !out_elf_buffer ||
      !out_elf_size) {
    return false;
  }

  ComgrData input = {};
  if (api->create_data(kComgrDataKindExecutable, &input) != kComgrStatusSuccess) {
    return false;
  }

  if (api->set_data(input, elf_size, static_cast<const char*>(elf_data)) != kComgrStatusSuccess) {
    api->release_data(input);
    return false;
  }

  ComgrData output = {};
  int status = kComgrStatusSuccess;
  if (request_entry_trampolines) {
    if (!api->hotswap_rewrite_with_options) {
      api->release_data(input);
      HOTSWAP_LOG("hotswap: COMGR entry-trampoline rewrite entry point unavailable\n");
      return false;
    }
    const ComgrHotswapRewriteOptions options{
        sizeof(ComgrHotswapRewriteOptions),
        kComgrHotswapRewriteFlagEntryTrampolines};
    status = api->hotswap_rewrite_with_options(input, source_isa, target_isa,
                                               &options, &output);
  } else {
    status = api->hotswap_rewrite(input, source_isa, target_isa, &output);
  }
  api->release_data(input);
  if (status != kComgrStatusSuccess) {
    HOTSWAP_LOG("hotswap: COMGR rewrite failed for %s -> %s (rc=%d)\n", source_isa, target_isa,
                status);
    return false;
  }

  size_t output_size = 0;
  if (api->get_data(output, &output_size, nullptr) != kComgrStatusSuccess || output_size == 0) {
    api->release_data(output);
    return false;
  }

  OwnedElfBuffer output_buffer(std::malloc(output_size), &std::free);
  if (!output_buffer) {
    api->release_data(output);
    return false;
  }

  size_t copy_size = output_size;
  if (api->get_data(output, &copy_size, static_cast<char*>(output_buffer.get())) !=
      kComgrStatusSuccess) {
    api->release_data(output);
    return false;
  }

  api->release_data(output);
  *out_elf_buffer = std::move(output_buffer);
  *out_elf_size = output_size;
  return true;
}

bool TryRetargetCodeObject(const CodeObjectView& code_object, hsa_agent_t agent,
                           OwnedElfBuffer* out_elf_buffer, size_t* out_elf_size) {
  if (IsHotswapDisabledByEnv() || !code_object.data || code_object.size == 0) {
    return false;
  }

  const AgentGfxRevision gfx = GetAgentGfxRevision(agent);
  const RewriteOptions options{IsGfx12_5RewriteRequested()};
  if (!IsAgentEligibleForHotswap(gfx, options)) {
    return false;
  }

  const std::string source_isa = GetCodeObjectIsaName(code_object.data, code_object.size);
  const std::string target_isa = GetAgentIsaName(agent);
  const std::optional<RewriteDecision> decision =
      DecideHotswapRewrite(gfx, source_isa, target_isa, options);
  if (!decision) {
    HOTSWAP_LOG("hotswap: rewrite skipped, no decision (src='%s' tgt='%s')\n",
                source_isa.c_str(), target_isa.c_str());
    return false;
  }

  const uint64_t cache_key = ComputeRetargetCacheKey(
      code_object.data, code_object.size, decision->source_isa,
      decision->target_isa, decision->request_entry_trampolines);

#if !defined(_WIN32) && !defined(_WIN64)
  const std::string disk_cache_dir = GetDiskCacheDir();
  const uint64_t disk_cache_salt = disk_cache_dir.empty() ? 0ULL : ComgrIdentitySalt();
#endif

  // Cache lookup: grab a shared_ptr under the lock, then copy outside it.
  {
    std::shared_ptr<std::vector<uint8_t>> cached_bytes;
    bool cached_failed = false;

    {
      std::scoped_lock lock(g_retarget_cache_mutex);
      auto it = g_retarget_cache.find(cache_key);
      if (it != g_retarget_cache.end()) {
        if (it->second.succeeded) {
          cached_bytes = it->second.elf_bytes;
        } else {
          cached_failed = true;
        }
      }
    }

    if (cached_failed) {
      HOTSWAP_LOG("hotswap: cache hit (failed) src=%s tgt=%s entry_trampolines=%d in=%zu\n",
                  decision->source_isa.c_str(), decision->target_isa.c_str(),
                  decision->request_entry_trampolines, code_object.size);
      return false;
    }

    if (cached_bytes) {
      OwnedElfBuffer buf(std::malloc(cached_bytes->size()), &std::free);
      if (buf) {
        std::memcpy(buf.get(), cached_bytes->data(), cached_bytes->size());
        *out_elf_buffer = std::move(buf);
        *out_elf_size = cached_bytes->size();
        HOTSWAP_LOG("hotswap: cache hit (success) src=%s tgt=%s entry_trampolines=%d "
                    "in=%zu out=%zu\n",
                    decision->source_isa.c_str(), decision->target_isa.c_str(),
                    decision->request_entry_trampolines, code_object.size,
                    cached_bytes->size());
        return true;
      }
    }
  }

#if !defined(_WIN32) && !defined(_WIN64)
  // Disk tier: another process (or a previous run) may have already persisted
  // the retargeted ELF. On a hit, copy it out and promote it into the in-memory
  // cache so the rest of this process avoids even the disk read.
  if (!disk_cache_dir.empty()) {
    const std::string disk_path = DiskCachePath(disk_cache_dir, cache_key, disk_cache_salt);
    std::shared_ptr<std::vector<uint8_t>> disk_bytes = ReadDiskCache(disk_path, disk_cache_salt);
    if (disk_bytes) {
      OwnedElfBuffer buf(std::malloc(disk_bytes->size()), &std::free);
      if (buf) {
        std::memcpy(buf.get(), disk_bytes->data(), disk_bytes->size());
        *out_elf_buffer = std::move(buf);
        *out_elf_size = disk_bytes->size();
        try {
          std::scoped_lock lock(g_retarget_cache_mutex);
          if (g_retarget_cache.find(cache_key) == g_retarget_cache.end() &&
              g_retarget_cache.size() < kMaxRetargetCacheEntries) {
            CachedRetargetResult entry;
            entry.succeeded = true;
            entry.elf_bytes = disk_bytes;
            g_retarget_cache.emplace(cache_key, std::move(entry));
          }
        } catch (const std::bad_alloc&) {
          // Promotion is best-effort; the disk result is already returned.
        }
        HOTSWAP_LOG("hotswap: disk cache hit src=%s tgt=%s entry_trampolines=%d in=%zu out=%zu\n",
                    decision->source_isa.c_str(), decision->target_isa.c_str(),
                    decision->request_entry_trampolines, code_object.size, disk_bytes->size());
        return true;
      }
    }
  }
#endif

  const bool rewritten =
      RetargetCodeObject(code_object.data, code_object.size,
                         decision->source_isa.c_str(), decision->target_isa.c_str(),
                         out_elf_buffer, out_elf_size,
                         decision->request_entry_trampolines);

  // Cache the result. Only deterministic COMGR failures are cached;
  // transient allocation failures in this function are not, so a
  // later attempt with the same code object can still succeed.
  try {
    std::scoped_lock lock(g_retarget_cache_mutex);
    if (g_retarget_cache.size() < kMaxRetargetCacheEntries) {
      CachedRetargetResult entry;
      entry.succeeded = rewritten;
      if (rewritten) {
        const auto* data = static_cast<const uint8_t*>((*out_elf_buffer).get());
        entry.elf_bytes = std::make_shared<std::vector<uint8_t>>(data, data + *out_elf_size);
      }
      g_retarget_cache.emplace(cache_key, std::move(entry));
    }
  } catch (const std::bad_alloc&) {
    // OOM during caching — the retarget result in out_elf_buffer is
    // still valid, we just can't cache it for future loads.
    HOTSWAP_LOG("hotswap: retarget cache store skipped (out of memory); "
                "returning uncached result for src=%s tgt=%s\n",
                decision->source_isa.c_str(), decision->target_isa.c_str());
  }

#if !defined(_WIN32) && !defined(_WIN64)
  // Persist a successful retarget so future launches (this or any process) can
  // skip COMGR entirely. Best-effort: failures never affect the returned result.
  if (rewritten && !disk_cache_dir.empty()) {
    WriteDiskCache(disk_cache_dir, cache_key, disk_cache_salt, (*out_elf_buffer).get(),
                   *out_elf_size);
    HOTSWAP_LOG("hotswap: disk cache store src=%s tgt=%s in=%zu out=%zu\n",
                decision->source_isa.c_str(), decision->target_isa.c_str(), code_object.size,
                *out_elf_size);
  }
#endif

  HOTSWAP_LOG("hotswap: rewrite src=%s tgt=%s entry_trampolines=%d in=%zu out=%zu changed=%d\n",
              decision->source_isa.c_str(), decision->target_isa.c_str(),
              decision->request_entry_trampolines, code_object.size,
              rewritten ? *out_elf_size : 0, rewritten ? 1 : 0);
  return rewritten;
}

bool TryRetargetCodeObject(amd::hsa::loader::CodeObjectReaderImpl* reader, hsa_agent_t agent,
                           OwnedElfBuffer* out_elf_buffer, size_t* out_elf_size) {
  if (!reader) {
    return false;
  }

  CodeObjectView code_object;
  code_object.data = reader->GetCodeObjectMemory();
  code_object.size = reader->GetCodeObjectSize();
  code_object.uri = reader->GetUri();
  return TryRetargetCodeObject(code_object, agent, out_elf_buffer, out_elf_size);
}

hsa_status_t LoadAgentCodeObjectWithHotswap(hsa_executable_t executable, hsa_agent_t agent,
                                            const CodeObjectView& code_object, const char* options,
                                            hsa_loaded_code_object_t* loaded_code_object,
                                            const LoadAgentCodeObjectCallbacks& callbacks) {
  if (!callbacks.load_original_code_object || !callbacks.load_rewritten_code_object) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  hsa_code_object_t original_code_object = {reinterpret_cast<uint64_t>(code_object.data)};

  OwnedElfBuffer rewritten_elf_buffer(nullptr, &std::free);
  size_t rewritten_elf_size = 0;
  if (TryRetargetCodeObject(code_object, agent, &rewritten_elf_buffer, &rewritten_elf_size)) {
    hsa_code_object_t rewritten_code_object = {
        reinterpret_cast<uint64_t>(rewritten_elf_buffer.get())};
    hsa_status_t status = callbacks.load_rewritten_code_object(
        callbacks.context, agent, rewritten_code_object, rewritten_elf_size, options,
        code_object.uri, loaded_code_object);
    if (status == HSA_STATUS_SUCCESS) {
      RetainRewrittenElfBuffer(executable, std::move(rewritten_elf_buffer));
      return status;
    }
    LogRewrittenCodeObjectLoadFailure(status);
  }

  return callbacks.load_original_code_object(callbacks.context, agent, original_code_object,
                                             options, code_object.uri, loaded_code_object);
}

void RetainRewrittenElfBuffer(hsa_executable_t executable, OwnedElfBuffer elf_buffer) {
  try {
    std::scoped_lock lock(g_retained_rewritten_elf_buffers_mutex);
    g_retained_rewritten_elf_buffers[executable.handle].push_back(std::move(elf_buffer));
  } catch (const std::bad_alloc&) {
    // If the keepalive container cannot grow, preserve the loaded code object's
    // raw ELF pointer by intentionally leaking this allocation.
    (void)elf_buffer.release();
  }
}

void ReleaseRetainedRewrittenElfBuffers(hsa_executable_t executable) {
  std::scoped_lock lock(g_retained_rewritten_elf_buffers_mutex);
  g_retained_rewritten_elf_buffers.erase(executable.handle);
}

#ifdef ROCR_HOTSWAP_TESTING
std::optional<RewriteDecision> DecideHotswapRewriteForTesting(
    const AgentGfxRevision& gfx, const std::string& source_isa,
    const std::string& target_isa, const RewriteOptions& options) {
  return DecideHotswapRewrite(gfx, source_isa, target_isa, options);
}

size_t RetainedRewrittenElfBufferCountForTesting(hsa_executable_t executable) {
  std::scoped_lock lock(g_retained_rewritten_elf_buffers_mutex);
  const auto it = g_retained_rewritten_elf_buffers.find(executable.handle);
  return it == g_retained_rewritten_elf_buffers.end() ? 0 : it->second.size();
}

bool EntryTrampolineRewriteAvailableForTesting() {
  ComgrApi* api = GetComgrApi();
  return api && api->hotswap_rewrite_with_options;
}

size_t RetargetCacheSizeForTesting() {
  std::scoped_lock lock(g_retarget_cache_mutex);
  return g_retarget_cache.size();
}

void ClearRetargetCacheForTesting() {
  std::scoped_lock lock(g_retarget_cache_mutex);
  g_retarget_cache.clear();
}
#endif

}  // namespace hotswap
}  // namespace rocr
