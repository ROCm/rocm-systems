//===- hotswap_rewrite_test.cc - HotSwap rewrite tests -------------------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include "core/inc/hotswap.hpp"
#include "core/inc/hotswap_gfx_query.hpp"
#include "core/inc/hsa_internal.h"
#include "core/util/os.h"
#include "gfx1250_min_hsaco.h"
#include "gtest/gtest.h"

#if defined(_WIN32) || defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

#include <filesystem>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

namespace {

constexpr const char* kGfx1250Isa = "amdgcn-amd-amdhsa--gfx1250";
constexpr const char* kGfx1251Isa = "amdgcn-amd-amdhsa--gfx1251";
constexpr const char* kGfx12_5GenericIsa =
    "amdgcn-amd-amdhsa--gfx12-5-generic";
constexpr const char* kGfx942Isa = "amdgcn-amd-amdhsa--gfx942";
constexpr const char* kGfx1250B0Isa =
    "amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+";
constexpr const char* kGfx1250A0Isa =
    "amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific-";

struct FakeHsaEnv {
  std::string isa_name = kGfx1250Isa;
  bool asic_rev_ok = true;
  uint32_t asic_revision = 0;
};

FakeHsaEnv g_fake_hsa_env;
std::unordered_map<std::string, std::string> g_fake_env_vars;

}  // namespace

namespace rocr {
namespace os {

LibHandle LoadLib(std::string filename) {
#if defined(_WIN32) || defined(_WIN64)
  return LoadLibraryA(filename.c_str());
#else
  int flags = RTLD_LAZY;
#ifdef RTLD_NODELETE
  flags |= RTLD_NODELETE;
#endif
  return dlopen(filename.c_str(), flags);
#endif
}

void* GetExportAddress(LibHandle lib, std::string export_name) {
#if defined(_WIN32) || defined(_WIN64)
  return reinterpret_cast<void*>(
      GetProcAddress(static_cast<HMODULE>(lib), export_name.c_str()));
#else
  return dlsym(lib, export_name.c_str());
#endif
}

bool CloseLib(LibHandle lib) {
#if defined(_WIN32) || defined(_WIN64)
  return FreeLibrary(static_cast<HMODULE>(lib)) != 0;
#else
  return dlclose(lib) == 0;
#endif
}

bool IsEnvVarSet(std::string env_var_name) {
  return g_fake_env_vars.find(env_var_name) != g_fake_env_vars.end();
}

std::string GetEnvVar(std::string env_var_name) {
  const auto it = g_fake_env_vars.find(env_var_name);
  return it == g_fake_env_vars.end() ? "" : it->second;
}

}  // namespace os

namespace HSA {

hsa_status_t hsa_agent_iterate_isas(hsa_agent_t /*agent*/,
                                    hsa_status_t (*callback)(hsa_isa_t isa,
                                                             void* data),
                                    void* data) {
  hsa_isa_t isa{};
  isa.handle = 1;
  return callback(isa, data);
}

hsa_status_t hsa_isa_get_info_alt(hsa_isa_t /*isa*/, hsa_isa_info_t attribute,
                                  void* value) {
  if (attribute == HSA_ISA_INFO_NAME_LENGTH) {
    *static_cast<uint32_t*>(value) =
        static_cast<uint32_t>(g_fake_hsa_env.isa_name.size() + 1);
    return HSA_STATUS_SUCCESS;
  }
  if (attribute == HSA_ISA_INFO_NAME) {
    std::memcpy(value, g_fake_hsa_env.isa_name.c_str(),
                g_fake_hsa_env.isa_name.size() + 1);
    return HSA_STATUS_SUCCESS;
  }
  return HSA_STATUS_ERROR;
}

hsa_status_t hsa_agent_get_info(hsa_agent_t /*agent*/,
                                hsa_agent_info_t attribute, void* value) {
  if (attribute ==
      static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_ASIC_REVISION)) {
    if (!g_fake_hsa_env.asic_rev_ok) {
      return HSA_STATUS_ERROR;
    }
    *static_cast<uint32_t*>(value) = g_fake_hsa_env.asic_revision;
    return HSA_STATUS_SUCCESS;
  }
  return HSA_STATUS_ERROR;
}

}  // namespace HSA
}  // namespace rocr

namespace {

enum class LoadPath {
  kOriginal,
  kRewritten,
};

struct LoadCall {
  LoadPath path;
  const void* code_object;
  size_t code_object_size;
  std::string uri;
};

struct LoadRecorder {
  std::vector<LoadCall> calls;
  std::vector<rocr::amd::hsa::loader::CodeObjectMemoryOwner> retained_owners;
  hsa_status_t original_status = HSA_STATUS_SUCCESS;
  hsa_status_t rewritten_status = HSA_STATUS_SUCCESS;
};

hsa_status_t RecordOriginalLoad(void* context, hsa_agent_t /*agent*/,
                                hsa_code_object_t code_object,
                                const char* /*options*/, const std::string& uri,
                                hsa_loaded_code_object_t* loaded_code_object) {
  auto* recorder = static_cast<LoadRecorder*>(context);
  recorder->calls.push_back({LoadPath::kOriginal,
                             reinterpret_cast<const void*>(code_object.handle),
                             0, uri});
  if (loaded_code_object) {
    loaded_code_object->handle = 0x1000 + recorder->calls.size();
  }
  return recorder->original_status;
}

hsa_status_t RecordRewrittenLoad(void* context, hsa_agent_t /*agent*/,
                                 hsa_code_object_t code_object, size_t code_object_size,
                                 rocr::amd::hsa::loader::CodeObjectMemoryOwner code_object_owner,
                                 const char* /*options*/, const std::string& uri,
                                 hsa_loaded_code_object_t* loaded_code_object) {
  auto* recorder = static_cast<LoadRecorder*>(context);
  recorder->calls.push_back({LoadPath::kRewritten,
                             reinterpret_cast<const void*>(code_object.handle),
                             code_object_size, uri});
  if (loaded_code_object) {
    loaded_code_object->handle = 0x2000 + recorder->calls.size();
  }
  if (recorder->rewritten_status == HSA_STATUS_SUCCESS) {
    recorder->retained_owners.push_back(std::move(code_object_owner));
  }
  return recorder->rewritten_status;
}

rocr::hotswap::LoadAgentCodeObjectCallbacks MakeLoadCallbacks(
    LoadRecorder* recorder) {
  rocr::hotswap::LoadAgentCodeObjectCallbacks callbacks;
  callbacks.context = recorder;
  callbacks.load_original_code_object = RecordOriginalLoad;
  callbacks.load_rewritten_code_object = RecordRewrittenLoad;
  return callbacks;
}

void ResetRuntimeTestEnv() {
  g_fake_hsa_env = FakeHsaEnv{};
  g_fake_env_vars.clear();
  rocr::hotswap::ResetAgentGfxRevisionCache();
  rocr::hotswap::SetComgrCacheFingerprintForTesting(nullptr);
  rocr::hotswap::ForceRetargetCodeObjectFailureForTesting(false);
}

bool ComgrHotswapOptionsApiAvailable() {
  if (rocr::hotswap::HotswapRewriteWithOptionsAvailableForTesting()) {
    return true;
  }
  SUCCEED() << "requires COMGR with amd_comgr_hotswap_rewrite_with_options";
  return false;
}

bool ComgrStrictModeApiAvailable() {
  if (!ComgrHotswapOptionsApiAvailable()) return false;

  if (rocr::hotswap::RetargetCodeObject(kGfx1250MinCo, sizeof(kGfx1250MinCo), kGfx1250B0Isa,
                                        kGfx1250B0Isa, false, true)
          .succeeded()) {
    return true;
  }

  SUCCEED() << "requires COMGR accepting AMD_COMGR_HOTSWAP_REWRITE_FLAG_STRICT_MODE";
  return false;
}

hsa_agent_t MakeTestAgent() {
  hsa_agent_t agent{};
  agent.handle = 1;
  return agent;
}

hsa_executable_t MakeTestExecutable(uint64_t handle) {
  hsa_executable_t executable{};
  executable.handle = handle;
  return executable;
}

rocr::hotswap::CodeObjectView MakeRealCodeObjectView() {
  rocr::hotswap::CodeObjectView code_object;
  code_object.data = kGfx1250MinCo;
  code_object.size = sizeof(kGfx1250MinCo);
  code_object.uri = "memory://gfx1250_min.hsaco";
  return code_object;
}

rocr::hotswap::RetargetOperationResult MakeTestRetargetedElf(
    const rocr::hotswap::SourceSnapshotRef& source = {}) {
  constexpr size_t kSize = 16;
  rocr::hotswap::OwnedElfBuffer bytes(std::malloc(kSize), &std::free);
  if (!bytes) {
    return {{}, rocr::hotswap::RetargetError::kOutOfResources};
  }
  std::memcpy(bytes.get(), kGfx1250MinCo, kSize);
  return {std::make_shared<const rocr::hotswap::RetargetedElf>(std::move(bytes), kSize, source),
          rocr::hotswap::RetargetError::kNone};
}

rocr::hotswap::AgentGfxRevision MakeRevision(const std::string& gfx_target,
                                             uint32_t asic_revision,
                                             bool has_asic_revision = true) {
  rocr::hotswap::AgentGfxRevision revision;
  revision.gfx_target = gfx_target;
  revision.asic_revision = asic_revision;
  revision.has_asic_revision = has_asic_revision;
  return revision;
}

TEST(HotswapRewriteDecision, A0RetargetsWithoutStrictModeRegardlessOfOptions) {
  rocr::hotswap::RewriteOptions entry_trampolines_enabled;
  entry_trampolines_enabled.entry_trampolines_enabled = true;
  rocr::hotswap::RewriteOptions strict_mode_enabled;
  strict_mode_enabled.strict_mode_enabled = true;
  const rocr::hotswap::RewriteOptions options[] = {
      {}, entry_trampolines_enabled, strict_mode_enabled};
  for (const auto& option : options) {
    SCOPED_TRACE(option.entry_trampolines_enabled
                     ? "entry trampolines enabled"
                     : (option.strict_mode_enabled ? "strict mode enabled" : "default options"));
    const auto decision = rocr::hotswap::DecideHotswapRewriteForTesting(
        MakeRevision("gfx1250", 0), kGfx1250Isa, kGfx1250Isa, option);

    ASSERT_TRUE(decision.has_value());
    EXPECT_EQ(decision->source_isa, kGfx1250B0Isa);
    EXPECT_EQ(decision->target_isa, kGfx1250A0Isa);
    EXPECT_FALSE(decision->request_entry_trampolines);
    EXPECT_FALSE(decision->request_strict_mode);
    EXPECT_FALSE(decision->rewrite_required);
  }
}

TEST(HotswapRewriteDecision, StrictModeDisabledDoesNotBlockA0Retarget) {
  rocr::hotswap::RewriteOptions options;
  options.strict_mode_enabled = false;

  const auto decision = rocr::hotswap::DecideHotswapRewriteForTesting(
      MakeRevision("gfx1250", 0), kGfx1250Isa, kGfx1250Isa, options);

  ASSERT_TRUE(decision.has_value());
  EXPECT_EQ(decision->source_isa, kGfx1250B0Isa);
  EXPECT_EQ(decision->target_isa, kGfx1250A0Isa);
  EXPECT_FALSE(decision->request_strict_mode);
  EXPECT_FALSE(decision->rewrite_required);
}

TEST(HotswapRewriteDecision, EntryTrampolinesDefaultOffBlocksNonA0Gfx1250) {
  const auto decision = rocr::hotswap::DecideHotswapRewriteForTesting(
      MakeRevision("gfx1250", 1), kGfx1250Isa, kGfx1250Isa, {});

  EXPECT_FALSE(decision.has_value());
}

TEST(HotswapRewriteDecision, EntryTrampolinesEnabledRoutesNonA0Gfx1250) {
  rocr::hotswap::RewriteOptions options;
  options.entry_trampolines_enabled = true;

  const auto decision = rocr::hotswap::DecideHotswapRewriteForTesting(
      MakeRevision("gfx1250", 1), kGfx1250Isa, kGfx1250Isa, options);

  ASSERT_TRUE(decision.has_value());
  EXPECT_EQ(decision->source_isa, kGfx1250B0Isa);
  EXPECT_EQ(decision->target_isa, kGfx1250B0Isa);
  EXPECT_TRUE(decision->request_entry_trampolines);
  EXPECT_FALSE(decision->request_strict_mode);
  EXPECT_FALSE(decision->rewrite_required);
}

TEST(HotswapRewriteDecision, StrictModeEnabledRoutesNonA0Gfx1250Strict) {
  rocr::hotswap::RewriteOptions options;
  options.strict_mode_enabled = true;

  const auto decision = rocr::hotswap::DecideHotswapRewriteForTesting(
      MakeRevision("gfx1250", 1), kGfx1250Isa, kGfx1250Isa, options);

  ASSERT_TRUE(decision.has_value());
  EXPECT_EQ(decision->source_isa, kGfx1250B0Isa);
  EXPECT_EQ(decision->target_isa, kGfx1250B0Isa);
  EXPECT_FALSE(decision->request_entry_trampolines);
  EXPECT_TRUE(decision->request_strict_mode);
  EXPECT_TRUE(decision->rewrite_required);
}

TEST(HotswapRewriteDecision, EntryTrampolinesEnabledKeepsNonA0Gfx1250StrictWhenEnabled) {
  rocr::hotswap::RewriteOptions options;
  options.entry_trampolines_enabled = true;
  options.strict_mode_enabled = true;

  const auto decision = rocr::hotswap::DecideHotswapRewriteForTesting(
      MakeRevision("gfx1250", 1), kGfx1250Isa, kGfx1250Isa, options);

  ASSERT_TRUE(decision.has_value());
  EXPECT_EQ(decision->source_isa, kGfx1250B0Isa);
  EXPECT_EQ(decision->target_isa, kGfx1250B0Isa);
  EXPECT_TRUE(decision->request_entry_trampolines);
  EXPECT_TRUE(decision->request_strict_mode);
  EXPECT_TRUE(decision->rewrite_required);
}

TEST(HotswapRewriteDecision, NonA0Gfx1250NoDecisionWhenEntryAndStrictDisabled) {
  rocr::hotswap::RewriteOptions options;
  options.entry_trampolines_enabled = false;
  options.strict_mode_enabled = false;

  const auto decision = rocr::hotswap::DecideHotswapRewriteForTesting(
      MakeRevision("gfx1250", 1), kGfx1250Isa, kGfx1250Isa, options);

  EXPECT_FALSE(decision.has_value());
}

TEST(HotswapRewriteDecision, EntryTrampolinesRouteGfx12_5Family) {
  rocr::hotswap::RewriteOptions options;
  options.entry_trampolines_enabled = true;
  const auto concrete = rocr::hotswap::DecideHotswapRewriteForTesting(
      MakeRevision("gfx1251", 1), kGfx1251Isa, kGfx1251Isa, options);
  const auto generic = rocr::hotswap::DecideHotswapRewriteForTesting(
      MakeRevision("gfx12-5-generic", 1), kGfx12_5GenericIsa, kGfx12_5GenericIsa, options);

  ASSERT_TRUE(concrete.has_value());
  EXPECT_EQ(concrete->source_isa, kGfx1251Isa);
  EXPECT_EQ(concrete->target_isa, kGfx1251Isa);
  EXPECT_TRUE(concrete->request_entry_trampolines);
  EXPECT_FALSE(concrete->request_strict_mode);
  EXPECT_FALSE(concrete->rewrite_required);
  ASSERT_TRUE(generic.has_value());
  EXPECT_EQ(generic->source_isa, kGfx12_5GenericIsa);
  EXPECT_EQ(generic->target_isa, kGfx12_5GenericIsa);
  EXPECT_TRUE(generic->request_entry_trampolines);
  EXPECT_FALSE(generic->request_strict_mode);
  EXPECT_FALSE(generic->rewrite_required);
}

TEST(HotswapRewriteDecision, EntryTrampolinesUseGenericSourceAsTarget) {
  rocr::hotswap::RewriteOptions options;
  options.entry_trampolines_enabled = true;
  const auto decision = rocr::hotswap::DecideHotswapRewriteForTesting(
      MakeRevision("gfx1251", 1), kGfx12_5GenericIsa, kGfx1251Isa, options);

  ASSERT_TRUE(decision.has_value());
  EXPECT_EQ(decision->source_isa, kGfx12_5GenericIsa);
  EXPECT_EQ(decision->target_isa, kGfx12_5GenericIsa);
  EXPECT_TRUE(decision->request_entry_trampolines);
  EXPECT_FALSE(decision->request_strict_mode);
  EXPECT_FALSE(decision->rewrite_required);
}

TEST(HotswapRewriteDecision, EntryTrampolinesBlockNonGfx12_5) {
  rocr::hotswap::RewriteOptions options;
  options.entry_trampolines_enabled = true;
  const auto decision = rocr::hotswap::DecideHotswapRewriteForTesting(
      MakeRevision("gfx942", 0), kGfx942Isa, kGfx942Isa, options);

  EXPECT_FALSE(decision.has_value());
}

TEST(HotswapRewrite, GetIsaNameRealCodeObject) {
  const std::string isa =
      rocr::hotswap::GetCodeObjectIsaName(kGfx1250MinCo, sizeof(kGfx1250MinCo));
  EXPECT_EQ(isa, kGfx1250Isa);
}

TEST(HotswapRewrite, GetIsaNameInvalidCodeObject) {
  const unsigned char fake_elf[] = {0x7f, 'E',  'L',  'F',  0x02, 0x01,
                                    0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
                                    0x00, 0x00, 0x00, 0x00};

  const std::string isa =
      rocr::hotswap::GetCodeObjectIsaName(fake_elf, sizeof(fake_elf));

  EXPECT_TRUE(isa.empty());
}

TEST(HotswapRewrite, RetargetRealCodeObject) {
  const auto rewritten = rocr::hotswap::RetargetCodeObject(kGfx1250MinCo, sizeof(kGfx1250MinCo),
                                                           kGfx1250Isa, kGfx1250Isa);

  ASSERT_TRUE(rewritten.succeeded());
  ASSERT_NE(rewritten.elf->data(), nullptr);
  EXPECT_NE(rewritten.elf->data(), static_cast<const void*>(kGfx1250MinCo));
  EXPECT_GT(rewritten.elf->size(), 0u);
  EXPECT_EQ(rocr::hotswap::GetCodeObjectIsaName(rewritten.elf->data(), rewritten.elf->size()),
            kGfx1250Isa);
}

TEST(HotswapRewrite, RetargetInvalidCodeObjectFails) {
  const unsigned char fake_elf[] = {0x7f, 'E',  'L',  'F',  0x02, 0x01,
                                    0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
                                    0x00, 0x00, 0x00, 0x00};
  const auto rewritten =
      rocr::hotswap::RetargetCodeObject(fake_elf, sizeof(fake_elf), kGfx1250Isa, kGfx1250Isa);

  EXPECT_FALSE(rewritten.succeeded());
  EXPECT_EQ(rewritten.error, rocr::hotswap::RetargetError::kComgrFailure);
}

TEST(HotswapRewrite, RetargetNullInputs) {
  const auto rewritten = rocr::hotswap::RetargetCodeObject(nullptr, 0, kGfx1250Isa, kGfx1250Isa);

  EXPECT_FALSE(rewritten.succeeded());
  EXPECT_EQ(rewritten.error, rocr::hotswap::RetargetError::kInvalidArgument);
}

TEST(HotswapRewrite, RetargetNullSourceOrTarget) {
  const unsigned char fake_elf[] = {0x7f, 'E', 'L', 'F'};

  const auto source_missing =
      rocr::hotswap::RetargetCodeObject(fake_elf, sizeof(fake_elf), nullptr, kGfx1250Isa);
  const auto target_missing =
      rocr::hotswap::RetargetCodeObject(fake_elf, sizeof(fake_elf), kGfx1250Isa, nullptr);

  EXPECT_EQ(source_missing.error, rocr::hotswap::RetargetError::kInvalidArgument);
  EXPECT_EQ(target_missing.error, rocr::hotswap::RetargetError::kInvalidArgument);
}

TEST(HotswapRewrite, RuntimeLoadUsesRewrittenCodeObject) {
  ResetRuntimeTestEnv();
  if (!ComgrHotswapOptionsApiAvailable()) return;
  LoadRecorder load;
  hsa_loaded_code_object_t loaded{};
  const hsa_executable_t executable = MakeTestExecutable(0x501);

  const hsa_status_t status = rocr::hotswap::LoadAgentCodeObjectWithHotswap(
      executable, MakeTestAgent(), MakeRealCodeObjectView(), nullptr, &loaded,
      MakeLoadCallbacks(&load));

  EXPECT_EQ(status, HSA_STATUS_SUCCESS);
  ASSERT_EQ(load.calls.size(), 1u);
  EXPECT_EQ(load.calls[0].path, LoadPath::kRewritten);
  EXPECT_NE(load.calls[0].code_object, static_cast<const void*>(kGfx1250MinCo));
  EXPECT_GT(load.calls[0].code_object_size, 0u);
  EXPECT_EQ(load.calls[0].uri, "memory://gfx1250_min.hsaco");
  ASSERT_EQ(load.retained_owners.size(), 1u);
  EXPECT_EQ(load.retained_owners[0].get(), load.calls[0].code_object);
}

TEST(HotswapRewrite, RuntimeLoadNonA0DefaultsToOriginalWhenEntryTrampolinesUnset) {
  ResetRuntimeTestEnv();
  g_fake_hsa_env.asic_revision = 1;
  LoadRecorder load;
  const hsa_executable_t executable = MakeTestExecutable(0x502);

  const hsa_status_t status = rocr::hotswap::LoadAgentCodeObjectWithHotswap(
      executable, MakeTestAgent(), MakeRealCodeObjectView(), nullptr, nullptr,
      MakeLoadCallbacks(&load));

  EXPECT_EQ(status, HSA_STATUS_SUCCESS);
  ASSERT_EQ(load.calls.size(), 1u);
  EXPECT_EQ(load.calls[0].path, LoadPath::kOriginal);
  EXPECT_EQ(load.calls[0].code_object, static_cast<const void*>(kGfx1250MinCo));
  EXPECT_TRUE(load.retained_owners.empty());
}

TEST(HotswapRewrite, RuntimeLoadNonA0FallsBackWhenEntryTrampolinesDisabledAndStrictUnset) {
  const char* const env_values[] = {"0", "false", "off", ""};
  uint64_t executable_handle = 0x503;
  for (const char* env_value : env_values) {
    SCOPED_TRACE(env_value);
    ResetRuntimeTestEnv();
    g_fake_hsa_env.asic_revision = 1;
    g_fake_env_vars["AMD_COMGR_HOTSWAP_ENTRY_TRAMPOLINES"] = env_value;
    LoadRecorder load;
    const hsa_executable_t executable = MakeTestExecutable(executable_handle++);

    const hsa_status_t status = rocr::hotswap::LoadAgentCodeObjectWithHotswap(
        executable, MakeTestAgent(), MakeRealCodeObjectView(), nullptr, nullptr,
        MakeLoadCallbacks(&load));

    EXPECT_EQ(status, HSA_STATUS_SUCCESS);
    ASSERT_EQ(load.calls.size(), 1u);
    EXPECT_EQ(load.calls[0].path, LoadPath::kOriginal);
    EXPECT_EQ(load.calls[0].code_object, static_cast<const void*>(kGfx1250MinCo));
    EXPECT_TRUE(load.retained_owners.empty());
  }
}

TEST(HotswapRewrite, RuntimeLoadNonA0UsesStrictModeEnvWhenEntryTrampolinesAreZero) {
  ResetRuntimeTestEnv();
  if (!ComgrStrictModeApiAvailable()) return;
  g_fake_hsa_env.asic_revision = 1;
  g_fake_env_vars["AMD_COMGR_HOTSWAP_ENTRY_TRAMPOLINES"] = "0";
  g_fake_env_vars["HSA_HOTSWAP_STRICT_MODE"] = "1";
  LoadRecorder load;
  const hsa_executable_t executable = MakeTestExecutable(0x504);

  const hsa_status_t status = rocr::hotswap::LoadAgentCodeObjectWithHotswap(
      executable, MakeTestAgent(), MakeRealCodeObjectView(), nullptr, nullptr,
      MakeLoadCallbacks(&load));

  EXPECT_EQ(status, HSA_STATUS_SUCCESS);
  ASSERT_EQ(load.calls.size(), 1u);
  EXPECT_EQ(load.calls[0].path, LoadPath::kRewritten);
  EXPECT_NE(load.calls[0].code_object, static_cast<const void*>(kGfx1250MinCo));
  EXPECT_GT(load.calls[0].code_object_size, 0u);
  ASSERT_EQ(load.retained_owners.size(), 1u);
  EXPECT_EQ(load.retained_owners[0].get(), load.calls[0].code_object);
}

TEST(HotswapRewrite, RuntimeLoadNonA0UsesEntryTrampolinesWhenEnabled) {
  if (!ComgrHotswapOptionsApiAvailable()) return;
  const char* const env_values[] = {"1", "true", "on"};
  uint64_t executable_handle = 0x507;
  for (const char* env_value : env_values) {
    SCOPED_TRACE(env_value);
    ResetRuntimeTestEnv();
    g_fake_hsa_env.asic_revision = 1;
    g_fake_env_vars["AMD_COMGR_HOTSWAP_ENTRY_TRAMPOLINES"] = env_value;
    LoadRecorder load;
    const hsa_executable_t executable =
        MakeTestExecutable(executable_handle++);

    const hsa_status_t status = rocr::hotswap::LoadAgentCodeObjectWithHotswap(
        executable, MakeTestAgent(), MakeRealCodeObjectView(), nullptr, nullptr,
        MakeLoadCallbacks(&load));

    EXPECT_EQ(status, HSA_STATUS_SUCCESS);
    ASSERT_EQ(load.calls.size(), 1u);
    EXPECT_EQ(load.calls[0].path, LoadPath::kRewritten);
    ASSERT_EQ(load.retained_owners.size(), 1u);
    EXPECT_EQ(load.retained_owners[0].get(), load.calls[0].code_object);
  }
}

TEST(HotswapRewrite, RuntimeLoadDisableEnvFallsBackToOriginal) {
  ResetRuntimeTestEnv();
  g_fake_env_vars["HSA_HOTSWAP_DISABLE"] = "1";
  LoadRecorder load;
  const hsa_executable_t executable = MakeTestExecutable(0x506);

  const hsa_status_t status = rocr::hotswap::LoadAgentCodeObjectWithHotswap(
      executable, MakeTestAgent(), MakeRealCodeObjectView(), nullptr, nullptr,
      MakeLoadCallbacks(&load));

  EXPECT_EQ(status, HSA_STATUS_SUCCESS);
  ASSERT_EQ(load.calls.size(), 1u);
  EXPECT_EQ(load.calls[0].path, LoadPath::kOriginal);
  EXPECT_TRUE(load.retained_owners.empty());
}

TEST(HotswapRewrite, RuntimeLoadRewriteFailureFallsBackToOriginal) {
  ResetRuntimeTestEnv();
  const unsigned char fake_elf[] = {0x7f, 'E',  'L',  'F',  0x02, 0x01,
                                    0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
                                    0x00, 0x00, 0x00, 0x00};
  rocr::hotswap::CodeObjectView code_object;
  code_object.data = fake_elf;
  code_object.size = sizeof(fake_elf);
  code_object.uri = "memory://invalid.hsaco";
  LoadRecorder load;
  const hsa_executable_t executable = MakeTestExecutable(0x507);

  const hsa_status_t status = rocr::hotswap::LoadAgentCodeObjectWithHotswap(
      executable, MakeTestAgent(), code_object, nullptr, nullptr,
      MakeLoadCallbacks(&load));

  EXPECT_EQ(status, HSA_STATUS_SUCCESS);
  ASSERT_EQ(load.calls.size(), 1u);
  EXPECT_EQ(load.calls[0].path, LoadPath::kOriginal);
  EXPECT_EQ(load.calls[0].code_object, fake_elf);
  EXPECT_TRUE(load.retained_owners.empty());
}

TEST(HotswapRewrite, RuntimeLoadOptionalRewriteFailureFallsBackToOriginal) {
  ResetRuntimeTestEnv();
  if (!ComgrHotswapOptionsApiAvailable()) return;
  g_fake_hsa_env.isa_name = kGfx1251Isa;
  g_fake_env_vars["AMD_COMGR_HOTSWAP_ENTRY_TRAMPOLINES"] = "1";
  rocr::hotswap::ForceRetargetCodeObjectFailureForTesting(true);
  LoadRecorder load;
  const hsa_executable_t executable = MakeTestExecutable(0x508);

  const hsa_status_t status = rocr::hotswap::LoadAgentCodeObjectWithHotswap(
      executable, MakeTestAgent(), MakeRealCodeObjectView(), nullptr, nullptr,
      MakeLoadCallbacks(&load));

  EXPECT_EQ(status, HSA_STATUS_SUCCESS);
  ASSERT_EQ(load.calls.size(), 1u);
  EXPECT_EQ(load.calls[0].path, LoadPath::kOriginal);
  EXPECT_EQ(load.calls[0].code_object, static_cast<const void*>(kGfx1250MinCo));
  EXPECT_TRUE(load.retained_owners.empty());

  rocr::hotswap::ForceRetargetCodeObjectFailureForTesting(false);
}

TEST(HotswapRewrite, RuntimeLoadRequiredStrictRewriteFailureReturnsError) {
  ResetRuntimeTestEnv();
  if (!ComgrHotswapOptionsApiAvailable()) return;
  g_fake_hsa_env.asic_revision = 1;
  g_fake_env_vars["HSA_HOTSWAP_STRICT_MODE"] = "1";
  rocr::hotswap::ForceRetargetCodeObjectFailureForTesting(true);
  LoadRecorder load;
  const hsa_executable_t executable = MakeTestExecutable(0x509);

  const hsa_status_t status = rocr::hotswap::LoadAgentCodeObjectWithHotswap(
      executable, MakeTestAgent(), MakeRealCodeObjectView(), nullptr, nullptr,
      MakeLoadCallbacks(&load));

  EXPECT_EQ(status, HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  EXPECT_TRUE(load.calls.empty());
  EXPECT_TRUE(load.retained_owners.empty());

  rocr::hotswap::ForceRetargetCodeObjectFailureForTesting(false);
}

TEST(HotswapRewrite, RuntimeLoadOptionalA0RewriteFailureFallsBackToOriginal) {
  ResetRuntimeTestEnv();
  if (!ComgrHotswapOptionsApiAvailable()) return;
  rocr::hotswap::ForceRetargetCodeObjectFailureForTesting(true);
  LoadRecorder load;
  const hsa_executable_t executable = MakeTestExecutable(0x512);

  const hsa_status_t status = rocr::hotswap::LoadAgentCodeObjectWithHotswap(
      executable, MakeTestAgent(), MakeRealCodeObjectView(), nullptr, nullptr,
      MakeLoadCallbacks(&load));

  EXPECT_EQ(status, HSA_STATUS_SUCCESS);
  ASSERT_EQ(load.calls.size(), 1u);
  EXPECT_EQ(load.calls[0].path, LoadPath::kOriginal);
  EXPECT_EQ(load.calls[0].code_object, static_cast<const void*>(kGfx1250MinCo));
  EXPECT_TRUE(load.retained_owners.empty());

  rocr::hotswap::ForceRetargetCodeObjectFailureForTesting(false);
}

TEST(HotswapRewrite, RuntimeLoadOptionalRewrittenLoadFailureFallsBackToOriginal) {
  ResetRuntimeTestEnv();
  if (!ComgrHotswapOptionsApiAvailable()) return;
  g_fake_hsa_env.isa_name = kGfx1251Isa;
  g_fake_env_vars["AMD_COMGR_HOTSWAP_ENTRY_TRAMPOLINES"] = "1";
  LoadRecorder load;
  load.rewritten_status = HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  const hsa_executable_t executable = MakeTestExecutable(0x510);

  const hsa_status_t status = rocr::hotswap::LoadAgentCodeObjectWithHotswap(
      executable, MakeTestAgent(), MakeRealCodeObjectView(), nullptr, nullptr,
      MakeLoadCallbacks(&load));

  EXPECT_EQ(status, HSA_STATUS_SUCCESS);
  ASSERT_EQ(load.calls.size(), 2u);
  EXPECT_EQ(load.calls[0].path, LoadPath::kRewritten);
  EXPECT_EQ(load.calls[1].path, LoadPath::kOriginal);
  EXPECT_EQ(load.calls[1].code_object, static_cast<const void*>(kGfx1250MinCo));
  EXPECT_TRUE(load.retained_owners.empty());
}

TEST(HotswapRewrite, RetargetCacheServesSecondLoadFromCache) {
  ResetRuntimeTestEnv();
  if (!ComgrHotswapOptionsApiAvailable()) return;
  rocr::hotswap::ContentRetargetCache cache;
  auto first_code_object = MakeRealCodeObjectView();
  first_code_object.reader_id = 1;
  first_code_object.retarget_cache = &cache;
  auto second_code_object = MakeRealCodeObjectView();
  second_code_object.reader_id = 2;
  second_code_object.retarget_cache = &cache;
  LoadRecorder first_load;
  LoadRecorder second_load;

  EXPECT_EQ(rocr::hotswap::LoadAgentCodeObjectWithHotswap(MakeTestExecutable(0x601),
                                                          MakeTestAgent(), first_code_object, nullptr,
                                                          nullptr, MakeLoadCallbacks(&first_load)),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(rocr::hotswap::LoadAgentCodeObjectWithHotswap(MakeTestExecutable(0x602),
                                                          MakeTestAgent(), second_code_object, nullptr,
                                                          nullptr, MakeLoadCallbacks(&second_load)),
            HSA_STATUS_SUCCESS);

  ASSERT_EQ(first_load.retained_owners.size(), 1u);
  ASSERT_EQ(second_load.retained_owners.size(), 1u);
  EXPECT_EQ(first_load.retained_owners[0].get(), second_load.retained_owners[0].get());
  EXPECT_EQ(cache.ReadyEntryCountForTesting(), 1u);
  EXPECT_EQ(cache.SnapshotMetrics().cross_reader_results, 1u);

  first_load.retained_owners.clear();
  EXPECT_EQ(cache.ReadyEntryCountForTesting(), 1u);
  second_load.retained_owners.clear();
  EXPECT_EQ(cache.ReadyEntryCountForTesting(), 0u);
}

TEST(HotswapRewrite, RetargetCacheCoalescesConcurrentMisses) {
  constexpr size_t kThreadCount = 8;
  rocr::hotswap::ContentRetargetCache cache;
  const rocr::hotswap::RetargetCacheKey key{kGfx1250B0Isa, kGfx1250A0Isa, false, false};
  std::atomic<size_t> producer_calls{0};
  std::atomic<size_t> started{0};
  std::atomic<bool> release_producer{false};
  std::vector<rocr::hotswap::RetargetOperationResult> results(kThreadCount);
  std::vector<std::thread> threads;

  auto producer = [&](const rocr::hotswap::SourceSnapshotRef& source) {
    producer_calls.fetch_add(1, std::memory_order_relaxed);
    while (!release_producer.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    return MakeTestRetargetedElf(source);
  };

  for (size_t i = 0; i < kThreadCount; ++i) {
    threads.emplace_back([&, i] {
      started.fetch_add(1, std::memory_order_release);
      results[i] = cache.GetOrCompute(kGfx1250MinCo, sizeof(kGfx1250MinCo), 0,
                                      key, producer);
    });
  }

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  bool all_waiters_observed = false;
  while (std::chrono::steady_clock::now() < deadline) {
    if (started.load(std::memory_order_acquire) == kThreadCount &&
        cache.WaiterCountForTesting(kGfx1250MinCo, sizeof(kGfx1250MinCo), key) ==
            kThreadCount - 1) {
      all_waiters_observed = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  release_producer.store(true, std::memory_order_release);
  for (auto& thread : threads) thread.join();

  EXPECT_TRUE(all_waiters_observed);
  EXPECT_EQ(producer_calls.load(std::memory_order_relaxed), 1u);
  size_t computed_results = 0;
  size_t coalesced_results = 0;
  for (const auto& result : results) {
    ASSERT_TRUE(result.succeeded());
    EXPECT_EQ(result.elf.get(), results[0].elf.get());
    computed_results += result.source == rocr::hotswap::RetargetResultSource::kComputed;
    coalesced_results += result.source == rocr::hotswap::RetargetResultSource::kCoalesced;
  }
  EXPECT_EQ(computed_results, 1u);
  EXPECT_EQ(coalesced_results, kThreadCount - 1);
}

TEST(HotswapRewrite, RetargetCacheRetriesFailures) {
  rocr::hotswap::ContentRetargetCache cache;
  const rocr::hotswap::RetargetCacheKey key{kGfx1250B0Isa, kGfx1250A0Isa, false, false};
  size_t producer_calls = 0;

  const auto first = cache.GetOrCompute(kGfx1250MinCo, sizeof(kGfx1250MinCo), 0,
                                       key, [&](const rocr::hotswap::SourceSnapshotRef&) {
    ++producer_calls;
    return rocr::hotswap::RetargetOperationResult{{},
                                                  rocr::hotswap::RetargetError::kOutOfResources};
  });
  const auto second = cache.GetOrCompute(kGfx1250MinCo, sizeof(kGfx1250MinCo), 0,
                                        key, [&](const rocr::hotswap::SourceSnapshotRef& source) {
    ++producer_calls;
    return MakeTestRetargetedElf(source);
  });

  EXPECT_FALSE(first.succeeded());
  EXPECT_EQ(first.error, rocr::hotswap::RetargetError::kOutOfResources);
  EXPECT_TRUE(second.succeeded());
  EXPECT_EQ(producer_calls, 2u);
}

TEST(HotswapRewrite, RuntimeLoadOptionalA0RewrittenLoadFailureFallsBackToOriginal) {
  ResetRuntimeTestEnv();
  if (!ComgrHotswapOptionsApiAvailable()) return;
  LoadRecorder load;
  load.rewritten_status = HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  const hsa_executable_t executable = MakeTestExecutable(0x513);

  const hsa_status_t status = rocr::hotswap::LoadAgentCodeObjectWithHotswap(
      executable, MakeTestAgent(), MakeRealCodeObjectView(), nullptr, nullptr,
      MakeLoadCallbacks(&load));

  EXPECT_EQ(status, HSA_STATUS_SUCCESS);
  ASSERT_EQ(load.calls.size(), 2u);
  EXPECT_EQ(load.calls[0].path, LoadPath::kRewritten);
  EXPECT_EQ(load.calls[1].path, LoadPath::kOriginal);
  EXPECT_EQ(load.calls[1].code_object, static_cast<const void*>(kGfx1250MinCo));
  EXPECT_TRUE(load.retained_owners.empty());
}

TEST(HotswapRewrite, RuntimeLoadRequiredStrictRewrittenLoadFailureReturnsError) {
  ResetRuntimeTestEnv();
  if (!ComgrStrictModeApiAvailable()) return;
  g_fake_hsa_env.asic_revision = 1;
  g_fake_env_vars["HSA_HOTSWAP_STRICT_MODE"] = "1";
  LoadRecorder load;
  load.rewritten_status = HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  const hsa_executable_t executable = MakeTestExecutable(0x511);

  const hsa_status_t status = rocr::hotswap::LoadAgentCodeObjectWithHotswap(
      executable, MakeTestAgent(), MakeRealCodeObjectView(), nullptr, nullptr,
      MakeLoadCallbacks(&load));

  EXPECT_EQ(status, HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  ASSERT_EQ(load.calls.size(), 1u);
  EXPECT_EQ(load.calls[0].path, LoadPath::kRewritten);
  EXPECT_TRUE(load.retained_owners.empty());
}


#if !defined(_WIN32) && !defined(_WIN64)
namespace {
int g_next_test_dir_counter = 0;

// A temp dir that removes itself on scope exit, so the disk-cache tests never
// leak /tmp/hotswap-disk-test-* directories. Naming follows the rocrtst
// convention (/tmp/<name>_<pid>, cf. gpu_coredump.cc) plus a counter so tests
// in the same process don't collide.
class ScopedTempDir {
 public:
  ScopedTempDir()
      : path_("/tmp/hotswap-disk-test-" +
              std::to_string(static_cast<long>(::getpid())) + "-" +
              std::to_string(g_next_test_dir_counter++)) {}
  ~ScopedTempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);  // best-effort; ignore errors
  }
  const std::string& str() const { return path_; }

 private:
  std::string path_;
};

rocr::hotswap::DiskCacheDigest TestDigest(uint8_t seed) {
  rocr::hotswap::DiskCacheDigest digest{};
  for (size_t i = 0; i < digest.size(); ++i) {
    digest[i] = static_cast<uint8_t>(seed + i);
  }
  return digest;
}

std::string DigestToHexForTesting(const rocr::hotswap::DiskCacheDigest& digest) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string result;
  result.reserve(digest.size() * 2);
  for (uint8_t byte : digest) {
    result.push_back(kHex[byte >> 4]);
    result.push_back(kHex[byte & 0xf]);
  }
  return result;
}

class ScopedDiskWriterReset {
 public:
  ScopedDiskWriterReset() { rocr::hotswap::ResetDiskWriterForTesting(); }
  ~ScopedDiskWriterReset() {
    rocr::hotswap::SetComgrCacheFingerprintForTesting(nullptr);
    rocr::hotswap::ForceRetargetCodeObjectFailureForTesting(false);
    rocr::hotswap::ResetDiskWriterForTesting();
  }
};
}  // namespace

TEST(RetargetDiskCache, Blake2b256MatchesKnownVector) {
  EXPECT_EQ(DigestToHexForTesting(rocr::hotswap::DigestBytesForTesting(nullptr, 0)),
            "0e5751c026e543b2e8ab2eb06099daa1d1e5df47778f7787faab45cdf12fe3a8");
  constexpr char kAbc[] = "abc";
  EXPECT_EQ(DigestToHexForTesting(rocr::hotswap::DigestBytesForTesting(kAbc, sizeof(kAbc) - 1)),
            "bddd813c634239723171ef3fee98579b94964e3bb1cb3e427262c8c068d52319");
  const std::vector<uint8_t> multi_block(129, 'a');
  EXPECT_EQ(DigestToHexForTesting(
                rocr::hotswap::DigestBytesForTesting(multi_block.data(), multi_block.size())),
            "2f64744a6de0d2c0b56e64cf6e29a5aaa255010d415d51c75ccc82f73dccd865");
}

TEST(RetargetDiskCache, CacheDigestCoversCanonicalInputs) {
  const std::vector<uint8_t> source = {1, 2, 3, 4};
  const auto fingerprint = TestDigest(0x20);
  const auto base = rocr::hotswap::ComputeRetargetCacheDigestForTesting(
      source.data(), source.size(), "source", "target", false, false, fingerprint);
  auto changed_source = source;
  changed_source.back() ^= 1;

  EXPECT_NE(base,
            rocr::hotswap::ComputeRetargetCacheDigestForTesting(
                changed_source.data(), changed_source.size(), "source", "target", false, false,
                fingerprint));
  EXPECT_NE(base,
            rocr::hotswap::ComputeRetargetCacheDigestForTesting(
                source.data(), source.size(), "source-2", "target", false, false, fingerprint));
  EXPECT_NE(base,
            rocr::hotswap::ComputeRetargetCacheDigestForTesting(
                source.data(), source.size(), "source", "target-2", false, false, fingerprint));
  EXPECT_NE(base,
            rocr::hotswap::ComputeRetargetCacheDigestForTesting(
                source.data(), source.size(), "source", "target", true, false, fingerprint));
  EXPECT_NE(base,
            rocr::hotswap::ComputeRetargetCacheDigestForTesting(
                source.data(), source.size(), "source", "target", false, true, fingerprint));
  EXPECT_NE(base,
            rocr::hotswap::ComputeRetargetCacheDigestForTesting(
                source.data(), source.size(), "source", "target", false, false, TestDigest(0x21)));
}

TEST(RetargetDiskCache, WriteThenReadRoundTrips) {
  ScopedTempDir tmp;
  const std::string& dir = tmp.str();
  const auto key = TestDigest(0x10);
  const auto fingerprint = TestDigest(0x80);
  std::vector<uint8_t> payload = {0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x11, 0x22};
  ASSERT_TRUE(rocr::hotswap::DiskCacheWriteForTesting(dir, key, fingerprint, payload));
  std::vector<uint8_t> out;
  ASSERT_TRUE(rocr::hotswap::DiskCacheReadForTesting(dir, key, fingerprint, &out));
  EXPECT_EQ(out, payload);
}

TEST(RetargetDiskCache, ComgrFingerprintMismatchIsMiss) {
  ScopedTempDir tmp;
  const std::string& dir = tmp.str();
  const auto key = TestDigest(1);
  const auto first_fingerprint = TestDigest(2);
  const auto second_fingerprint = TestDigest(3);
  std::vector<uint8_t> payload = {1, 2, 3, 4};
  ASSERT_TRUE(rocr::hotswap::DiskCacheWriteForTesting(dir, key, first_fingerprint, payload));
  std::vector<uint8_t> out;
  EXPECT_FALSE(rocr::hotswap::DiskCacheReadForTesting(dir, key, second_fingerprint, &out));
  EXPECT_TRUE(rocr::hotswap::DiskCacheReadForTesting(dir, key, first_fingerprint, &out));
}

TEST(RetargetDiskCache, MissingEntryIsMiss) {
  ScopedTempDir tmp;
  std::vector<uint8_t> out;
  EXPECT_FALSE(
      rocr::hotswap::DiskCacheReadForTesting(tmp.str(), TestDigest(4), TestDigest(5), &out));
}

TEST(RetargetDiskCache, DistinctKeysDoNotCollide) {
  ScopedTempDir tmp;
  const std::string& dir = tmp.str();
  const auto first_key = TestDigest(6);
  const auto second_key = TestDigest(7);
  const auto fingerprint = TestDigest(8);
  std::vector<uint8_t> a = {0xAA, 0xAA};
  std::vector<uint8_t> b = {0xBB, 0xBB, 0xBB};
  ASSERT_TRUE(rocr::hotswap::DiskCacheWriteForTesting(dir, first_key, fingerprint, a));
  ASSERT_TRUE(rocr::hotswap::DiskCacheWriteForTesting(dir, second_key, fingerprint, b));
  std::vector<uint8_t> out;
  ASSERT_TRUE(rocr::hotswap::DiskCacheReadForTesting(dir, first_key, fingerprint, &out));
  EXPECT_EQ(out, a);
  ASSERT_TRUE(rocr::hotswap::DiskCacheReadForTesting(dir, second_key, fingerprint, &out));
  EXPECT_EQ(out, b);
}

TEST(RetargetDiskCache, OverwriteSameKeyIsIdempotent) {
  ScopedTempDir tmp;
  const std::string& dir = tmp.str();
  const auto key = TestDigest(9);
  const auto fingerprint = TestDigest(10);
  std::vector<uint8_t> v1 = {1, 1, 1};
  std::vector<uint8_t> v2 = {2, 2, 2, 2, 2};
  ASSERT_TRUE(rocr::hotswap::DiskCacheWriteForTesting(dir, key, fingerprint, v1));
  ASSERT_TRUE(rocr::hotswap::DiskCacheWriteForTesting(dir, key, fingerprint, v2));
  std::vector<uint8_t> out;
  ASSERT_TRUE(rocr::hotswap::DiskCacheReadForTesting(dir, key, fingerprint, &out));
  EXPECT_EQ(out, v2);
}

TEST(RetargetDiskCache, CorruptPayloadIsRejected) {
  ScopedTempDir tmp;
  const std::string& dir = tmp.str();
  const auto key = TestDigest(11);
  const auto fingerprint = TestDigest(12);
  const std::vector<uint8_t> payload(4096, 0x5a);
  ASSERT_TRUE(rocr::hotswap::DiskCacheWriteForTesting(dir, key, fingerprint, payload));

  const std::string path = rocr::hotswap::DiskCachePathForTesting(dir, key, fingerprint);
  std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
  ASSERT_TRUE(file.is_open());
  file.seekg(-1, std::ios::end);
  char byte = 0;
  file.read(&byte, 1);
  ASSERT_TRUE(file.good());
  byte ^= 1;
  file.seekp(-1, std::ios::end);
  file.write(&byte, 1);
  file.close();

  std::vector<uint8_t> out;
  EXPECT_FALSE(rocr::hotswap::DiskCacheReadForTesting(dir, key, fingerprint, &out));
}

TEST(RetargetDiskCache, ShutdownWithoutUseDoesNotConstructWriter) {
  ScopedDiskWriterReset writer_reset;
  EXPECT_FALSE(rocr::hotswap::DiskWriterConstructedForTesting());
  rocr::hotswap::HotswapCacheShutdown();
  rocr::hotswap::HotswapCacheWaitForShutdown();
  EXPECT_FALSE(rocr::hotswap::DiskWriterConstructedForTesting());
}

TEST(RetargetDiskCache, DefaultOnWithOptionalKillSwitch) {
  ResetRuntimeTestEnv();
  EXPECT_TRUE(rocr::hotswap::DiskCacheEnabledForTesting());
  g_fake_env_vars["HSA_HOTSWAP_DISK_CACHE"] = "0";
  EXPECT_FALSE(rocr::hotswap::DiskCacheEnabledForTesting());
  ResetRuntimeTestEnv();
}

TEST(RetargetDiskCache, QueueTaskLimitDropsBeforeUnboundedRetention) {
  ScopedDiskWriterReset writer_reset;
  ScopedTempDir tmp;
  constexpr size_t kPayloadSize = 64 * 1024;
  const std::vector<uint8_t> payload(kPayloadSize, 0x7e);
  const auto fingerprint = TestDigest(13);
  rocr::hotswap::ConfigureDiskWriterForTesting(2, 3 * kPayloadSize);
  rocr::hotswap::BlockDiskWritesForTesting(true);

  EXPECT_TRUE(
      rocr::hotswap::EnqueueDiskWriteForTesting(tmp.str(), TestDigest(14), fingerprint, payload));
  EXPECT_TRUE(
      rocr::hotswap::EnqueueDiskWriteForTesting(tmp.str(), TestDigest(15), fingerprint, payload));
  EXPECT_FALSE(
      rocr::hotswap::EnqueueDiskWriteForTesting(tmp.str(), TestDigest(16), fingerprint, payload));

  const auto before_shutdown = rocr::hotswap::GetDiskCacheMetrics();
  EXPECT_EQ(before_shutdown.queued_tasks, 2u);
  EXPECT_EQ(before_shutdown.queued_bytes, 2 * kPayloadSize);
  EXPECT_EQ(before_shutdown.peak_queued_tasks, 2u);
  EXPECT_EQ(before_shutdown.peak_queued_bytes, 2 * kPayloadSize);
  EXPECT_EQ(before_shutdown.dropped_tasks, 1u);

  const auto start = std::chrono::steady_clock::now();
  rocr::hotswap::HotswapCacheShutdown();
  const auto request_duration = std::chrono::steady_clock::now() - start;
  rocr::hotswap::HotswapCacheWaitForShutdown();
  EXPECT_LT(request_duration, std::chrono::seconds(1));
  const auto after_shutdown = rocr::hotswap::GetDiskCacheMetrics();
  EXPECT_EQ(after_shutdown.queued_tasks, 0u);
  EXPECT_EQ(after_shutdown.queued_bytes, 0u);
  EXPECT_EQ(after_shutdown.dropped_tasks, 3u);
}

TEST(RetargetDiskCache, QueueByteLimitDropsBeforeUnboundedRetention) {
  ScopedDiskWriterReset writer_reset;
  ScopedTempDir tmp;
  constexpr size_t kPayloadSize = 64 * 1024;
  const std::vector<uint8_t> payload(kPayloadSize, 0x7e);
  const auto fingerprint = TestDigest(17);
  rocr::hotswap::ConfigureDiskWriterForTesting(3, 2 * kPayloadSize);
  rocr::hotswap::BlockDiskWritesForTesting(true);

  EXPECT_TRUE(
      rocr::hotswap::EnqueueDiskWriteForTesting(tmp.str(), TestDigest(18), fingerprint, payload));
  EXPECT_TRUE(
      rocr::hotswap::EnqueueDiskWriteForTesting(tmp.str(), TestDigest(19), fingerprint, payload));
  EXPECT_FALSE(
      rocr::hotswap::EnqueueDiskWriteForTesting(tmp.str(), TestDigest(20), fingerprint, payload));

  const auto before_shutdown = rocr::hotswap::GetDiskCacheMetrics();
  EXPECT_EQ(before_shutdown.queued_tasks, 2u);
  EXPECT_EQ(before_shutdown.queued_bytes, 2 * kPayloadSize);
  EXPECT_EQ(before_shutdown.peak_queued_tasks, 2u);
  EXPECT_EQ(before_shutdown.peak_queued_bytes, 2 * kPayloadSize);
  EXPECT_EQ(before_shutdown.dropped_tasks, 1u);

  rocr::hotswap::HotswapCacheShutdown();
  rocr::hotswap::HotswapCacheWaitForShutdown();
  const auto after_shutdown = rocr::hotswap::GetDiskCacheMetrics();
  EXPECT_EQ(after_shutdown.queued_tasks, 0u);
  EXPECT_EQ(after_shutdown.queued_bytes, 0u);
  EXPECT_EQ(after_shutdown.dropped_tasks, 3u);
}

TEST(RetargetDiskCache, ConcurrentShutdownWaitersAreSafe) {
  ScopedDiskWriterReset writer_reset;
  ScopedTempDir tmp;
  const std::vector<uint8_t> payload(64 * 1024, 0x7e);
  rocr::hotswap::ConfigureDiskWriterForTesting(1, payload.size());
  rocr::hotswap::BlockDiskWritesForTesting(true);
  ASSERT_TRUE(rocr::hotswap::EnqueueDiskWriteForTesting(tmp.str(), TestDigest(21), TestDigest(22),
                                                        payload));

  rocr::hotswap::HotswapCacheShutdown();
  std::thread first_waiter([] { rocr::hotswap::HotswapCacheWaitForShutdown(); });
  std::thread second_waiter([] { rocr::hotswap::HotswapCacheWaitForShutdown(); });
  first_waiter.join();
  second_waiter.join();

  const auto metrics = rocr::hotswap::GetDiskCacheMetrics();
  EXPECT_EQ(metrics.queued_tasks, 0u);
  EXPECT_EQ(metrics.queued_bytes, 0u);
}

TEST(RetargetDiskCache, ProductionColdWriteThenHitWithoutComgrRewrite) {
  ResetRuntimeTestEnv();
  ScopedDiskWriterReset writer_reset;
  ScopedTempDir tmp;
  if (!rocr::hotswap::ComgrCacheIdentifierAvailableForTesting()) {
    const auto fingerprint = TestDigest(0x60);
    rocr::hotswap::SetComgrCacheFingerprintForTesting(&fingerprint);
  }
  g_fake_env_vars["XDG_CACHE_HOME"] = tmp.str();
  g_fake_hsa_env.asic_revision = 0;

  rocr::hotswap::ContentRetargetCache first_cache;
  auto first_view = MakeRealCodeObjectView();
  first_view.retarget_cache = &first_cache;
  LoadRecorder first_load;
  ASSERT_EQ(rocr::hotswap::LoadAgentCodeObjectWithHotswap(MakeTestExecutable(0x601),
                                                          MakeTestAgent(), first_view, nullptr,
                                                          nullptr, MakeLoadCallbacks(&first_load)),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(first_load.calls.size(), 1u);
  ASSERT_EQ(first_load.calls[0].path, LoadPath::kRewritten);
  ASSERT_TRUE(rocr::hotswap::WaitForDiskWriterIdleForTesting(/*timeout_ms=*/10000));
  EXPECT_EQ(rocr::hotswap::GetDiskCacheMetrics().completed_tasks, 1u);
  first_load.retained_owners.clear();

  rocr::hotswap::ContentRetargetCache second_cache;
  auto second_view = MakeRealCodeObjectView();
  second_view.retarget_cache = &second_cache;
  LoadRecorder second_load;
  rocr::hotswap::ForceRetargetCodeObjectFailureForTesting(true);
  ASSERT_EQ(rocr::hotswap::LoadAgentCodeObjectWithHotswap(MakeTestExecutable(0x602),
                                                          MakeTestAgent(), second_view, nullptr,
                                                          nullptr, MakeLoadCallbacks(&second_load)),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(second_load.calls.size(), 1u);
  EXPECT_EQ(second_load.calls[0].path, LoadPath::kRewritten);
}
#endif  // POSIX


}  // namespace
