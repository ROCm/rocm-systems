//===- hotswap_rewrite_test.cc - HotSwap rewrite tests -------------------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdlib>
#include <cstring>

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
#endif

#include <string>
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
                                 hsa_code_object_t code_object,
                                 size_t code_object_size,
                                 const char* /*options*/,
                                 const std::string& uri,
                                 hsa_loaded_code_object_t* loaded_code_object) {
  auto* recorder = static_cast<LoadRecorder*>(context);
  recorder->calls.push_back({LoadPath::kRewritten,
                             reinterpret_cast<const void*>(code_object.handle),
                             code_object_size, uri});
  if (loaded_code_object) {
    loaded_code_object->handle = 0x2000 + recorder->calls.size();
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
}

bool NewComgrHotswapApiAvailable() {
  if (rocr::hotswap::EntryTrampolineRewriteAvailableForTesting()) {
    return true;
  }
  SUCCEED() << "requires COMGR with amd_comgr_hotswap_rewrite_with_options";
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
  rocr::hotswap::ReleaseRetainedRewrittenElfBuffers(executable);
  return executable;
}

rocr::hotswap::CodeObjectView MakeRealCodeObjectView() {
  rocr::hotswap::CodeObjectView code_object;
  code_object.data = kGfx1250MinCo;
  code_object.size = sizeof(kGfx1250MinCo);
  code_object.uri = "memory://gfx1250_min.hsaco";
  return code_object;
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

TEST(HotswapRewriteDecision, A0RetargetsUsesLegacyPathUnlessOptedIn) {
  struct TestCase {
    rocr::hotswap::RewriteOptions options;
    bool request_entry_trampolines;
  };
  // {entry_trampolines_enabled}.
  const TestCase cases[] = {
      {{}, false},       // default: B0->A0 legacy path
      {{false}, false},  // explicitly disabled: legacy path
      {{true}, true},    // opted in: entry trampolines
  };
  for (const TestCase& test_case : cases) {
    SCOPED_TRACE(test_case.request_entry_trampolines
                     ? "entry trampolines enabled"
                     : "entry trampolines disabled");
    const auto decision = rocr::hotswap::DecideHotswapRewriteForTesting(
        MakeRevision("gfx1250", 0), kGfx1250Isa, kGfx1250Isa,
        test_case.options);

    ASSERT_TRUE(decision.has_value());
    EXPECT_EQ(decision->source_isa, kGfx1250B0Isa);
    EXPECT_EQ(decision->target_isa, kGfx1250A0Isa);
    EXPECT_EQ(decision->request_entry_trampolines,
              test_case.request_entry_trampolines);
  }
}

TEST(HotswapRewriteDecision, EntryTrampolinesDefaultOffBlocksNonA0Gfx1250) {
  const auto decision = rocr::hotswap::DecideHotswapRewriteForTesting(
      MakeRevision("gfx1250", 1), kGfx1250Isa, kGfx1250Isa, {});

  EXPECT_FALSE(decision.has_value());
}

TEST(HotswapRewriteDecision, EntryTrampolinesEnabledRoutesNonA0Gfx1250) {
  const auto decision = rocr::hotswap::DecideHotswapRewriteForTesting(
      MakeRevision("gfx1250", 1), kGfx1250Isa, kGfx1250Isa, {true});

  ASSERT_TRUE(decision.has_value());
  EXPECT_EQ(decision->source_isa, kGfx1250B0Isa);
  EXPECT_EQ(decision->target_isa, kGfx1250B0Isa);
  EXPECT_TRUE(decision->request_entry_trampolines);
}

TEST(HotswapRewriteDecision, EntryTrampolinesDisabledBlocksNonA0Gfx1250) {
  const auto decision = rocr::hotswap::DecideHotswapRewriteForTesting(
      MakeRevision("gfx1250", 1), kGfx1250Isa, kGfx1250Isa, {false});

  EXPECT_FALSE(decision.has_value());
}

TEST(HotswapRewriteDecision, EntryTrampolinesRouteGfx12_5Family) {
  const auto concrete = rocr::hotswap::DecideHotswapRewriteForTesting(
      MakeRevision("gfx1251", 1), kGfx1251Isa, kGfx1251Isa, {true});
  const auto generic = rocr::hotswap::DecideHotswapRewriteForTesting(
      MakeRevision("gfx12-5-generic", 1), kGfx12_5GenericIsa,
      kGfx12_5GenericIsa, {true});

  ASSERT_TRUE(concrete.has_value());
  EXPECT_EQ(concrete->source_isa, kGfx1251Isa);
  EXPECT_EQ(concrete->target_isa, kGfx1251Isa);
  EXPECT_TRUE(concrete->request_entry_trampolines);
  ASSERT_TRUE(generic.has_value());
  EXPECT_EQ(generic->source_isa, kGfx12_5GenericIsa);
  EXPECT_EQ(generic->target_isa, kGfx12_5GenericIsa);
  EXPECT_TRUE(generic->request_entry_trampolines);
}

TEST(HotswapRewriteDecision, EntryTrampolinesUseGenericSourceAsTarget) {
  const auto decision = rocr::hotswap::DecideHotswapRewriteForTesting(
      MakeRevision("gfx1251", 1), kGfx12_5GenericIsa, kGfx1251Isa, {true});

  ASSERT_TRUE(decision.has_value());
  EXPECT_EQ(decision->source_isa, kGfx12_5GenericIsa);
  EXPECT_EQ(decision->target_isa, kGfx12_5GenericIsa);
  EXPECT_TRUE(decision->request_entry_trampolines);
}

TEST(HotswapRewriteDecision, EntryTrampolinesBlockNonGfx12_5) {
  const auto decision = rocr::hotswap::DecideHotswapRewriteForTesting(
      MakeRevision("gfx942", 0), kGfx942Isa, kGfx942Isa, {true});

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
  rocr::hotswap::OwnedElfBuffer rewritten_elf_buffer(nullptr, &std::free);
  size_t rewritten_elf_size = 0;

  const bool rewritten = rocr::hotswap::RetargetCodeObject(
      kGfx1250MinCo, sizeof(kGfx1250MinCo), kGfx1250Isa, kGfx1250Isa,
      &rewritten_elf_buffer, &rewritten_elf_size);

  ASSERT_TRUE(rewritten);
  ASSERT_NE(rewritten_elf_buffer.get(), nullptr);
  EXPECT_NE(rewritten_elf_buffer.get(),
            static_cast<const void*>(kGfx1250MinCo));
  EXPECT_GT(rewritten_elf_size, 0u);
  EXPECT_EQ(rocr::hotswap::GetCodeObjectIsaName(rewritten_elf_buffer.get(),
                                                rewritten_elf_size),
            kGfx1250Isa);
}

TEST(HotswapRewrite, RetargetInvalidCodeObjectFails) {
  const unsigned char fake_elf[] = {0x7f, 'E',  'L',  'F',  0x02, 0x01,
                                    0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
                                    0x00, 0x00, 0x00, 0x00};
  rocr::hotswap::OwnedElfBuffer rewritten_elf_buffer(nullptr, &std::free);
  size_t rewritten_elf_size = 0;

  const bool rewritten = rocr::hotswap::RetargetCodeObject(
      fake_elf, sizeof(fake_elf), kGfx1250Isa, kGfx1250Isa,
      &rewritten_elf_buffer, &rewritten_elf_size);

  EXPECT_FALSE(rewritten);
  EXPECT_EQ(rewritten_elf_buffer.get(), nullptr);
  EXPECT_EQ(rewritten_elf_size, 0u);
}

TEST(HotswapRewrite, RetargetNullOutputPointers) {
  const unsigned char fake_elf[] = {0x7f, 'E', 'L', 'F'};

  const bool rewritten = rocr::hotswap::RetargetCodeObject(
      fake_elf, sizeof(fake_elf), kGfx1250Isa, kGfx1250Isa, nullptr, nullptr);

  EXPECT_FALSE(rewritten);
}

TEST(HotswapRewrite, RetargetNullInputs) {
  rocr::hotswap::OwnedElfBuffer rewritten_elf_buffer(nullptr, &std::free);
  size_t rewritten_elf_size = 0;

  const bool rewritten = rocr::hotswap::RetargetCodeObject(
      nullptr, 0, kGfx1250Isa, kGfx1250Isa, &rewritten_elf_buffer,
      &rewritten_elf_size);

  EXPECT_FALSE(rewritten);
}

TEST(HotswapRewrite, RetargetNullSourceOrTarget) {
  const unsigned char fake_elf[] = {0x7f, 'E', 'L', 'F'};
  rocr::hotswap::OwnedElfBuffer rewritten_elf_buffer(nullptr, &std::free);
  size_t rewritten_elf_size = 0;

  const bool source_missing_rewritten = rocr::hotswap::RetargetCodeObject(
      fake_elf, sizeof(fake_elf), nullptr, kGfx1250Isa, &rewritten_elf_buffer,
      &rewritten_elf_size);
  const bool target_missing_rewritten = rocr::hotswap::RetargetCodeObject(
      fake_elf, sizeof(fake_elf), kGfx1250Isa, nullptr, &rewritten_elf_buffer,
      &rewritten_elf_size);

  EXPECT_FALSE(source_missing_rewritten);
  EXPECT_FALSE(target_missing_rewritten);
}

TEST(HotswapRewrite, RuntimeLoadUsesRewrittenCodeObject) {
  ResetRuntimeTestEnv();
  if (!NewComgrHotswapApiAvailable()) return;
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
  EXPECT_EQ(
      rocr::hotswap::RetainedRewrittenElfBufferCountForTesting(executable), 1u);

  rocr::hotswap::ReleaseRetainedRewrittenElfBuffers(executable);
  EXPECT_EQ(
      rocr::hotswap::RetainedRewrittenElfBufferCountForTesting(executable), 0u);
}

TEST(HotswapRewrite, RuntimeLoadNonA0DefaultsToOriginalWhenEntryTrampolinesUnset) {
  ResetRuntimeTestEnv();
  g_fake_hsa_env.asic_revision = 1;
  LoadRecorder load;
  const hsa_executable_t executable = MakeTestExecutable(0x504);

  const hsa_status_t status = rocr::hotswap::LoadAgentCodeObjectWithHotswap(
      executable, MakeTestAgent(), MakeRealCodeObjectView(), nullptr, nullptr,
      MakeLoadCallbacks(&load));

  EXPECT_EQ(status, HSA_STATUS_SUCCESS);
  ASSERT_EQ(load.calls.size(), 1u);
  EXPECT_EQ(load.calls[0].path, LoadPath::kOriginal);
  EXPECT_EQ(load.calls[0].code_object, static_cast<const void*>(kGfx1250MinCo));
  EXPECT_EQ(
      rocr::hotswap::RetainedRewrittenElfBufferCountForTesting(executable), 0u);
}

TEST(HotswapRewrite, RuntimeLoadNonA0FallsBackWhenEntryTrampolinesDisabled) {
  const char* const env_values[] = {"0", "false", "off", ""};
  uint64_t executable_handle = 0x510;
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
    EXPECT_EQ(load.calls[0].path, LoadPath::kOriginal);
    EXPECT_EQ(load.calls[0].code_object, static_cast<const void*>(kGfx1250MinCo));
    EXPECT_EQ(
        rocr::hotswap::RetainedRewrittenElfBufferCountForTesting(executable), 0u);
  }
}

TEST(HotswapRewrite, RuntimeLoadNonA0UsesEntryTrampolinesWhenEnabled) {
  if (!NewComgrHotswapApiAvailable()) return;
  const char* const env_values[] = {"1", "true", "on"};
  uint64_t executable_handle = 0x509;
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
    EXPECT_EQ(
        rocr::hotswap::RetainedRewrittenElfBufferCountForTesting(executable),
        1u);

    rocr::hotswap::ReleaseRetainedRewrittenElfBuffers(executable);
    EXPECT_EQ(
        rocr::hotswap::RetainedRewrittenElfBufferCountForTesting(executable),
        0u);
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
  EXPECT_EQ(
      rocr::hotswap::RetainedRewrittenElfBufferCountForTesting(executable), 0u);
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
  EXPECT_EQ(
      rocr::hotswap::RetainedRewrittenElfBufferCountForTesting(executable), 0u);
}

TEST(HotswapRewrite, RuntimeLoadRewrittenLoadFailureFallsBackToOriginal) {
  ResetRuntimeTestEnv();
  if (!NewComgrHotswapApiAvailable()) return;
  LoadRecorder load;
  load.rewritten_status = HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  const hsa_executable_t executable = MakeTestExecutable(0x508);

  const hsa_status_t status = rocr::hotswap::LoadAgentCodeObjectWithHotswap(
      executable, MakeTestAgent(), MakeRealCodeObjectView(), nullptr, nullptr,
      MakeLoadCallbacks(&load));

  EXPECT_EQ(status, HSA_STATUS_SUCCESS);
  ASSERT_EQ(load.calls.size(), 2u);
  EXPECT_EQ(load.calls[0].path, LoadPath::kRewritten);
  EXPECT_EQ(load.calls[1].path, LoadPath::kOriginal);
  EXPECT_EQ(load.calls[1].code_object, static_cast<const void*>(kGfx1250MinCo));
  EXPECT_EQ(
      rocr::hotswap::RetainedRewrittenElfBufferCountForTesting(executable), 0u);
}

}  // namespace
