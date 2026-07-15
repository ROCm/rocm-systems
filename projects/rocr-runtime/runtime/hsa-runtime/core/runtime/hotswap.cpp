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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
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
#endif

#include "core/inc/hotswap_gfx_query.hpp"
#include "core/util/os.h"

namespace rocr {
namespace hotswap {
namespace {

std::mutex g_retained_rewritten_elf_buffers_mutex;
std::unordered_map<uint64_t, std::vector<OwnedElfBuffer>> g_retained_rewritten_elf_buffers;
#ifdef ROCR_HOTSWAP_TESTING
std::atomic<bool> g_force_retarget_code_object_failure_for_testing{false};
#endif

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
                                 bool entry_trampolines, bool strict_mode) {
  uint64_t hash = FnvHash(elf_data, elf_size);
  hash ^= FnvHash(source_isa.data(), source_isa.size()) * 31;
  hash ^= FnvHash(target_isa.data(), target_isa.size()) * 37;
  hash ^= entry_trampolines ? 0xDEADBEEF12345678ULL : 0x0ULL;
  hash ^= strict_mode ? 0x9E3779B97F4A7C15ULL : 0x0ULL;
  return hash;
}

struct CachedRetargetResult {
  bool succeeded = false;
  // Ref-counted so callers can grab a cheap handle under the mutex
  // and copy into the output buffer after releasing it.
  std::shared_ptr<std::vector<uint8_t>> elf_bytes;
};

// Default in-memory cache byte budget (4 GiB). Bounds resident RAM held by
// cached retarget results. Overridable at runtime via HSA_HOTSWAP_MEM_BUDGET
// (parsed as a byte count; see ParseMemBudgetEnv). A value of 0 disables the
// budget (unbounded), matching prior behavior for callers that opt out.
constexpr size_t kDefaultRetargetCacheMemBudget = 4ULL * 1024 * 1024 * 1024;

// LRU-by-bytes cache of retarget results, keyed by ComputeRetargetCacheKey.
//
// - Successful entries hold a ref-counted ELF buffer and consume `bytes()`
//   worth of budget; failure sentinels hold no buffer and consume nothing.
// - On insert, least-recently-used entries are evicted until the total held
//   bytes fit within the budget. A single entry larger than the whole budget
//   is still stored (evicting everything else); refusing it would force an
//   unbounded COMGR re-run on every load of that object.
// - Access (hit) and insert both move the entry to the most-recently-used
//   position. Failure sentinels participate in LRU ordering too, so a stale
//   failure can be evicted like any other entry.
//
// All public methods are internally synchronized; the returned shared_ptr
// lets callers perform the large copy after releasing the lock.
class RetargetCache {
 public:
  explicit RetargetCache(size_t byte_budget) : byte_budget_(byte_budget) {}

  enum class Lookup { kMiss, kHitSuccess, kHitFailure };

  // Looks up `key`. On a success hit, `out_bytes` receives a shared handle to
  // the cached buffer (copy it outside the lock). Touches LRU order on hit.
  Lookup Get(uint64_t key, std::shared_ptr<std::vector<uint8_t>>* out_bytes) {
    if (out_bytes != nullptr) {
      out_bytes->reset();  // clear on all paths so callers never see stale data
    }
    std::scoped_lock lock(mutex_);
    auto it = map_.find(key);
    if (it == map_.end()) {
      return Lookup::kMiss;
    }
    TouchLocked(it);
    if (it->second.result.succeeded) {
      if (out_bytes != nullptr) {
        *out_bytes = it->second.result.elf_bytes;
      }
      return Lookup::kHitSuccess;
    }
    return Lookup::kHitFailure;
  }

  // Inserts (or replaces) `key` with a success entry owning `bytes`, evicting
  // LRU entries as needed to respect the budget. `bytes` must be non-null.
  void PutSuccess(uint64_t key, std::shared_ptr<std::vector<uint8_t>> bytes) {
    const size_t entry_bytes = bytes ? bytes->size() : 0;
    CachedRetargetResult result;
    result.succeeded = true;
    result.elf_bytes = std::move(bytes);
    std::scoped_lock lock(mutex_);
    InsertLocked(key, std::move(result), entry_bytes);
  }

  // Inserts (or replaces) `key` with a failure sentinel (consumes no budget).
  void PutFailure(uint64_t key) {
    CachedRetargetResult result;
    result.succeeded = false;
    std::scoped_lock lock(mutex_);
    InsertLocked(key, std::move(result), 0);
  }

  size_t Size() {
    std::scoped_lock lock(mutex_);
    return map_.size();
  }

  size_t Bytes() {
    std::scoped_lock lock(mutex_);
    return held_bytes_;
  }

  void Clear() {
    std::scoped_lock lock(mutex_);
    map_.clear();
    lru_.clear();
    held_bytes_ = 0;
  }

#ifdef ROCR_HOTSWAP_TESTING
  // Testing hook: reports residency without perturbing LRU order.
  bool ContainsForTesting(uint64_t key) {
    std::scoped_lock lock(mutex_);
    return map_.find(key) != map_.end();
  }

  // Testing hook: change the byte budget and immediately re-evict so the new
  // limit takes effect. Used to exercise eviction without allocating gigabytes.
  void SetByteBudgetForTesting(size_t budget) {
    std::scoped_lock lock(mutex_);
    byte_budget_ = budget;
    // Evict from the tail until within the new budget. Pass a sentinel that
    // matches no key so nothing is protected.
    EvictToBudgetLocked(/*protect=*/0, /*protect_valid=*/false);
  }

  // Testing hook: insert a synthetic success entry of `size` zero-filled bytes
  // under `key`, so eviction ordering/accounting can be tested with controlled
  // sizes and without real code objects.
  void PutSyntheticForTesting(uint64_t key, size_t size) {
    auto bytes = std::make_shared<std::vector<uint8_t>>(size, 0);
    PutSuccess(key, std::move(bytes));
  }
#endif

 private:
  struct Entry {
    CachedRetargetResult result;
    size_t bytes = 0;                     // budget contribution of this entry
    std::list<uint64_t>::iterator lru_it; // position in lru_ (front = MRU)
  };

  // Moves the entry to the front (most-recently-used). Caller holds mutex_.
  void TouchLocked(std::unordered_map<uint64_t, Entry>::iterator it) {
    lru_.splice(lru_.begin(), lru_, it->second.lru_it);
  }

  // Inserts/replaces key, then evicts from the LRU tail until within budget.
  // Caller holds mutex_.
  void InsertLocked(uint64_t key, CachedRetargetResult result,
                    size_t entry_bytes) {
    auto existing = map_.find(key);
    if (existing != map_.end()) {
      // Replace in place, adjusting the byte total and refreshing LRU order.
      held_bytes_ -= existing->second.bytes;
      existing->second.result = std::move(result);
      existing->second.bytes = entry_bytes;
      held_bytes_ += entry_bytes;
      TouchLocked(existing);
    } else {
      lru_.push_front(key);
      Entry entry;
      entry.result = std::move(result);
      entry.bytes = entry_bytes;
      entry.lru_it = lru_.begin();
      map_.emplace(key, std::move(entry));
      held_bytes_ += entry_bytes;
    }
    EvictToBudgetLocked(key, /*protect_valid=*/true);
  }

  // Evicts least-recently-used entries until held_bytes_ <= byte_budget_.
  // When `protect_valid` is true, never evicts `protect` (the just-inserted
  // key), so a lone oversized entry is retained rather than dropped and forced
  // to re-run COMGR on every load. A budget of 0 means unbounded (no
  // eviction). Caller holds mutex_.
  void EvictToBudgetLocked(uint64_t protect, bool protect_valid) {
    if (byte_budget_ == 0) {
      return;
    }
    while (held_bytes_ > byte_budget_ && !lru_.empty()) {
      uint64_t victim = lru_.back();
      if (protect_valid && victim == protect) {
        // The protected (just-inserted, MRU) entry sits at the tail only when
        // it is the sole remaining entry. Stop rather than evict it: keeping a
        // lone oversized entry is preferable to discarding what we just added.
        break;
      }
      auto it = map_.find(victim);
      if (it != map_.end()) {
        held_bytes_ -= it->second.bytes;
        map_.erase(it);
      }
      lru_.pop_back();
    }
  }

  std::mutex mutex_;
  std::unordered_map<uint64_t, Entry> map_;
  std::list<uint64_t> lru_;  // front = most-recently-used, back = least
  size_t held_bytes_ = 0;
  size_t byte_budget_;  // mutable only under mutex_ (see SetByteBudgetForTesting)
};

size_t ParseMemBudgetEnv() {
  if (!os::IsEnvVarSet("HSA_HOTSWAP_MEM_BUDGET")) {
    return kDefaultRetargetCacheMemBudget;
  }
  std::string value = os::GetEnvVar("HSA_HOTSWAP_MEM_BUDGET");
  // Trim leading whitespace so a leading '-' is detectable below.
  size_t first = value.find_first_not_of(" \t");
  if (first == std::string::npos) {
    return kDefaultRetargetCacheMemBudget;  // empty / all-whitespace
  }
  value = value.substr(first);
  // Reject negatives explicitly: strtoull would wrap "-1" to ULLONG_MAX, which
  // would read as an effectively-unbounded budget rather than a parse error.
  if (value[0] == '-') {
    return kDefaultRetargetCacheMemBudget;
  }
  // Accept a plain byte count. errno/strtoull guard against garbage; on any
  // parse failure fall back to the default rather than an accidental 0
  // (unbounded) or overflow. An explicit "0" is honored as unbounded.
  errno = 0;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
  if (end == value.c_str() || errno != 0) {
    return kDefaultRetargetCacheMemBudget;
  }
  return static_cast<size_t>(parsed);
}

RetargetCache& GetRetargetCache() {
  static RetargetCache cache(ParseMemBudgetEnv());
  return cache;
}

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
    decision.rewrite_required = true;
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

bool RetargetCodeObject(const void* elf_data, size_t elf_size, const char* source_isa,
                        const char* target_isa, OwnedElfBuffer* out_elf_buffer,
                        size_t* out_elf_size, bool request_entry_trampolines,
                        bool request_strict_mode) {
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
  const uint64_t rewrite_flags =
      (request_entry_trampolines ? kComgrHotswapRewriteFlagEntryTrampolines : 0) |
      (request_strict_mode ? kComgrHotswapRewriteFlagStrictMode : 0);
  if (rewrite_flags != 0) {
    if (!api->hotswap_rewrite_with_options) {
      api->release_data(input);
      HOTSWAP_LOG("hotswap: COMGR rewrite-with-options entry point unavailable\n");
      return false;
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

RetargetCodeObjectResult TryRetargetCodeObject(const CodeObjectView& code_object,
                                               hsa_agent_t agent,
                                               OwnedElfBuffer* out_elf_buffer,
                                               size_t* out_elf_size) {
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

  const uint64_t cache_key = ComputeRetargetCacheKey(
      code_object.data, code_object.size, decision->source_isa,
      decision->target_isa, decision->request_entry_trampolines,
      decision->request_strict_mode);

  // Cache lookup: grab a shared_ptr under the lock, then copy outside it.
  {
    std::shared_ptr<std::vector<uint8_t>> cached_bytes;
    const RetargetCache::Lookup lookup =
        GetRetargetCache().Get(cache_key, &cached_bytes);

    if (lookup == RetargetCache::Lookup::kHitFailure) {
      HOTSWAP_LOG("hotswap: cache hit (failed) src=%s tgt=%s entry_trampolines=%d strict=%d "
                  "required=%d in=%zu\n",
                  decision->source_isa.c_str(), decision->target_isa.c_str(),
                  decision->request_entry_trampolines, decision->request_strict_mode,
                  decision->rewrite_required, code_object.size);
      // A deterministic COMGR failure is cached. If the rewrite was required,
      // fail loudly rather than falling back to the unpatched original.
      if (decision->rewrite_required) {
        return {RetargetCodeObjectStatus::kRequiredRewriteFailed, true};
      }
      return {};
    }

    if (lookup == RetargetCache::Lookup::kHitSuccess && cached_bytes) {
      OwnedElfBuffer buf(std::malloc(cached_bytes->size()), &std::free);
      if (buf) {
        std::memcpy(buf.get(), cached_bytes->data(), cached_bytes->size());
        *out_elf_buffer = std::move(buf);
        *out_elf_size = cached_bytes->size();
        HOTSWAP_LOG("hotswap: cache hit (success) src=%s tgt=%s entry_trampolines=%d strict=%d "
                    "in=%zu out=%zu\n",
                    decision->source_isa.c_str(), decision->target_isa.c_str(),
                    decision->request_entry_trampolines, decision->request_strict_mode,
                    code_object.size, cached_bytes->size());
        return {RetargetCodeObjectStatus::kRewritten, decision->rewrite_required};
      }
      // malloc failed: fall through to a fresh retarget rather than caching a
      // spurious failure. The cached entry stays intact for a later attempt.
    }
  }

  bool rewritten = false;
  bool retarget_attempted = true;
#ifdef ROCR_HOTSWAP_TESTING
  if (g_force_retarget_code_object_failure_for_testing.load(std::memory_order_relaxed)) {
    HOTSWAP_LOG("hotswap: forcing retarget failure for test\n");
    retarget_attempted = false;
  } else
#endif
  {
    rewritten = RetargetCodeObject(code_object.data, code_object.size,
                                   decision->source_isa.c_str(), decision->target_isa.c_str(),
                                   out_elf_buffer, out_elf_size,
                                   decision->request_entry_trampolines,
                                   decision->request_strict_mode);
  }

  // Cache the result. Only deterministic COMGR outcomes are cached; transient
  // allocation failures and test-forced failures are not, so a later attempt
  // with the same code object can still succeed.
  if (retarget_attempted) {
    try {
      if (rewritten) {
        const auto* data = static_cast<const uint8_t*>((*out_elf_buffer).get());
        auto bytes =
            std::make_shared<std::vector<uint8_t>>(data, data + *out_elf_size);
        GetRetargetCache().PutSuccess(cache_key, std::move(bytes));
      } else {
        GetRetargetCache().PutFailure(cache_key);
      }
      HOTSWAP_LOG("hotswap: cached key=0x%llx src=%s tgt=%s entry_trampolines=%d strict=%d "
                  "in=%zu succeeded=%d\n",
                  (unsigned long long)cache_key, decision->source_isa.c_str(),
                  decision->target_isa.c_str(), decision->request_entry_trampolines,
                  decision->request_strict_mode, code_object.size, rewritten ? 1 : 0);
    } catch (const std::bad_alloc&) {
      HOTSWAP_LOG("hotswap: cache store skipped (OOM) key=0x%llx src=%s tgt=%s in=%zu\n",
                  (unsigned long long)cache_key, decision->source_isa.c_str(),
                  decision->target_isa.c_str(), code_object.size);
    }
  }

  HOTSWAP_LOG("hotswap: rewrite src=%s tgt=%s entry_trampolines=%d strict=%d required=%d "
              "in=%zu out=%zu changed=%d\n",
              decision->source_isa.c_str(), decision->target_isa.c_str(),
              decision->request_entry_trampolines, decision->request_strict_mode,
              decision->rewrite_required, code_object.size, rewritten ? *out_elf_size : 0,
              rewritten ? 1 : 0);
  if (rewritten) {
    return {RetargetCodeObjectStatus::kRewritten,
            decision->rewrite_required};
  }
  if (decision->rewrite_required) {
    return {RetargetCodeObjectStatus::kRequiredRewriteFailed, true};
  }
  return {};
}

RetargetCodeObjectResult TryRetargetCodeObject(
    amd::hsa::loader::CodeObjectReaderImpl* reader, hsa_agent_t agent,
    OwnedElfBuffer* out_elf_buffer, size_t* out_elf_size) {
  if (!reader) {
    return {};
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
  const RetargetCodeObjectResult retarget_result =
      TryRetargetCodeObject(code_object, agent, &rewritten_elf_buffer, &rewritten_elf_size);
  if (retarget_result.status == RetargetCodeObjectStatus::kRewritten) {
    hsa_code_object_t rewritten_code_object = {
        reinterpret_cast<uint64_t>(rewritten_elf_buffer.get())};
    hsa_status_t status = callbacks.load_rewritten_code_object(
        callbacks.context, agent, rewritten_code_object, rewritten_elf_size, options,
        code_object.uri, loaded_code_object);
    if (status == HSA_STATUS_SUCCESS) {
      RetainRewrittenElfBuffer(executable, std::move(rewritten_elf_buffer));
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

bool HotswapRewriteWithOptionsAvailableForTesting() {
  ComgrApi* api = GetComgrApi();
  return api && api->hotswap_rewrite_with_options;
}

void ForceRetargetCodeObjectFailureForTesting(bool force) {
  g_force_retarget_code_object_failure_for_testing.store(force, std::memory_order_relaxed);
}

size_t RetargetCacheSizeForTesting() { return GetRetargetCache().Size(); }

void ClearRetargetCacheForTesting() { GetRetargetCache().Clear(); }

size_t RetargetCacheBytesForTesting() { return GetRetargetCache().Bytes(); }

void SetRetargetCacheByteBudgetForTesting(size_t budget) {
  GetRetargetCache().SetByteBudgetForTesting(budget);
}

void PutSyntheticRetargetCacheEntryForTesting(uint64_t key, size_t size) {
  GetRetargetCache().PutSyntheticForTesting(key, size);
}

void PutFailureRetargetCacheEntryForTesting(uint64_t key) {
  GetRetargetCache().PutFailure(key);
}

bool RetargetCacheContainsForTesting(uint64_t key) {
  return GetRetargetCache().ContainsForTesting(key);
}

bool RetargetCacheGetForTesting(
    uint64_t key, std::shared_ptr<std::vector<uint8_t>>* out_bytes) {
  return GetRetargetCache().Get(key, out_bytes) ==
         RetargetCache::Lookup::kHitSuccess;
}
#endif

}  // namespace hotswap
}  // namespace rocr
