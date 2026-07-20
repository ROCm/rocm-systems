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
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <new>
#include <string>
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

#ifdef ROCR_HOTSWAP_TESTING
std::atomic<bool> g_force_retarget_code_object_failure_for_testing{false};
#endif

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
    return RetargetCodeObject(source_data, source_size, decision->source_isa.c_str(),
                              decision->target_isa.c_str(), decision->request_entry_trampolines,
                              decision->request_strict_mode, source_snapshot);
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

#endif

}  // namespace hotswap
}  // namespace rocr
