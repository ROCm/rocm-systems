//===- hotswap_rewrite_test.cpp - HotSwap rewrite tests -------------------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "core/inc/hsa_internal.h"
#include "core/runtime/hotswap.hpp"
#include "core/runtime/hotswap_gfx_query.hpp"
#include "core/util/os.h"

#include "gfx1250_min_hsaco.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

static constexpr const char* kGfx1250Isa = "amdgcn-amd-amdhsa--gfx1250";

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
  return dlopen(filename.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

void* GetExportAddress(LibHandle lib, std::string export_name) {
#if defined(_WIN32) || defined(_WIN64)
  return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(lib), export_name.c_str()));
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
                                    hsa_status_t (*callback)(hsa_isa_t isa, void* data),
                                    void* data) {
  hsa_isa_t isa{};
  isa.handle = 1;
  return callback(isa, data);
}

hsa_status_t hsa_isa_get_info_alt(hsa_isa_t /*isa*/, hsa_isa_info_t attribute, void* value) {
  if (attribute == HSA_ISA_INFO_NAME_LENGTH) {
    *static_cast<uint32_t*>(value) = static_cast<uint32_t>(g_fake_hsa_env.isa_name.size() + 1);
    return HSA_STATUS_SUCCESS;
  }
  if (attribute == HSA_ISA_INFO_NAME) {
    std::memcpy(value, g_fake_hsa_env.isa_name.c_str(), g_fake_hsa_env.isa_name.size() + 1);
    return HSA_STATUS_SUCCESS;
  }
  return HSA_STATUS_ERROR;
}

hsa_status_t hsa_agent_get_info(hsa_agent_t /*agent*/, hsa_agent_info_t attribute, void* value) {
  if (attribute == static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_ASIC_REVISION)) {
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

int tests_passed = 0;
int tests_failed = 0;

void check(bool cond, const char* name) {
  if (cond) {
    ++tests_passed;
    printf("  PASS: %s\n", name);
  } else {
    ++tests_failed;
    fprintf(stderr, "  FAIL: %s\n", name);
  }
}

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

hsa_status_t RecordOriginalLoad(void* context, hsa_agent_t /*agent*/, hsa_code_object_t code_object,
                                const char* /*options*/, const std::string& uri,
                                hsa_loaded_code_object_t* loaded_code_object) {
  auto* recorder = static_cast<LoadRecorder*>(context);
  recorder->calls.push_back(
      {LoadPath::kOriginal, reinterpret_cast<const void*>(code_object.handle), 0, uri});
  if (loaded_code_object) {
    loaded_code_object->handle = 0x1000 + recorder->calls.size();
  }
  return recorder->original_status;
}

hsa_status_t RecordRewrittenLoad(void* context, hsa_agent_t /*agent*/,
                                 hsa_code_object_t code_object, size_t code_object_size,
                                 const char* /*options*/, const std::string& uri,
                                 hsa_loaded_code_object_t* loaded_code_object) {
  auto* recorder = static_cast<LoadRecorder*>(context);
  recorder->calls.push_back({LoadPath::kRewritten,
                             reinterpret_cast<const void*>(code_object.handle), code_object_size,
                             uri});
  if (loaded_code_object) {
    loaded_code_object->handle = 0x2000 + recorder->calls.size();
  }
  return recorder->rewritten_status;
}

rocr::hotswap::LoadAgentCodeObjectCallbacks MakeLoadCallbacks(LoadRecorder* recorder) {
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

void test_GetIsaNameRealCodeObject() {
  printf("TEST GetIsaNameRealCodeObject...\n");
  const std::string isa = rocr::hotswap::GetCodeObjectIsaName(kGfx1250MinCo, sizeof(kGfx1250MinCo));
  check(isa == kGfx1250Isa, "GetCodeObjectIsaName reads gfx1250 from a real CO");
}

void test_GetIsaNameInvalidCodeObject() {
  printf("TEST GetIsaNameInvalidCodeObject...\n");
  const unsigned char fake_elf[] = {0x7f, 'E',  'L',  'F',  0x02, 0x01, 0x01, 0x00,
                                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  const std::string isa = rocr::hotswap::GetCodeObjectIsaName(fake_elf, sizeof(fake_elf));
  check(isa.empty(), "GetCodeObjectIsaName returns empty for an invalid CO");
}

void test_RetargetRealCodeObject() {
  printf("TEST RetargetRealCodeObject...\n");
  rocr::hotswap::OwnedElfBuffer rewritten_elf_buffer(nullptr, &std::free);
  size_t rewritten_elf_size = 0;
  const bool rewritten =
      rocr::hotswap::RetargetCodeObject(kGfx1250MinCo, sizeof(kGfx1250MinCo), kGfx1250Isa,
                                        kGfx1250Isa, &rewritten_elf_buffer, &rewritten_elf_size);

  check(rewritten, "RetargetCodeObject rewrites a real CO");
  check(rewritten_elf_buffer != nullptr &&
            rewritten_elf_buffer.get() != static_cast<const void*>(kGfx1250MinCo),
        "rewritten_elf_buffer is a fresh allocation");
  check(rewritten_elf_size > 0, "rewritten_elf_size is non-zero");
  check(rocr::hotswap::GetCodeObjectIsaName(rewritten_elf_buffer.get(), rewritten_elf_size) ==
            kGfx1250Isa,
        "rewritten CO keeps the expected ISA name");
}

void test_RetargetInvalidCodeObjectFails() {
  printf("TEST RetargetInvalidCodeObjectFails...\n");
  const unsigned char fake_elf[] = {0x7f, 'E',  'L',  'F',  0x02, 0x01, 0x01, 0x00,
                                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  rocr::hotswap::OwnedElfBuffer rewritten_elf_buffer(nullptr, &std::free);
  size_t rewritten_elf_size = 0;
  const bool rewritten =
      rocr::hotswap::RetargetCodeObject(fake_elf, sizeof(fake_elf), kGfx1250Isa, kGfx1250Isa,
                                        &rewritten_elf_buffer, &rewritten_elf_size);

  check(!rewritten, "RetargetCodeObject fails for an un-rewritable CO");
  check(rewritten_elf_buffer == nullptr, "rewritten_elf_buffer remains empty after failure");
  check(rewritten_elf_size == 0, "rewritten_elf_size remains zero after failure");
}

void test_RetargetNullOutputPointers() {
  printf("TEST RetargetNullOutputPointers...\n");
  const unsigned char fake_elf[] = {0x7f, 'E', 'L', 'F'};
  const bool rewritten = rocr::hotswap::RetargetCodeObject(fake_elf, sizeof(fake_elf), kGfx1250Isa,
                                                           kGfx1250Isa, nullptr, nullptr);
  check(!rewritten, "RetargetCodeObject rejects null output pointers");
}

void test_RetargetNullInputs() {
  printf("TEST RetargetNullInputs...\n");
  rocr::hotswap::OwnedElfBuffer rewritten_elf_buffer(nullptr, &std::free);
  size_t rewritten_elf_size = 0;
  const bool rewritten = rocr::hotswap::RetargetCodeObject(
      nullptr, 0, kGfx1250Isa, kGfx1250Isa, &rewritten_elf_buffer, &rewritten_elf_size);
  check(!rewritten, "RetargetCodeObject rejects null elf_data");
}

void test_RetargetNullSourceOrTarget() {
  printf("TEST RetargetNullSourceOrTarget...\n");
  const unsigned char fake_elf[] = {0x7f, 'E', 'L', 'F'};
  rocr::hotswap::OwnedElfBuffer rewritten_elf_buffer(nullptr, &std::free);
  size_t rewritten_elf_size = 0;
  const bool source_missing_rewritten = rocr::hotswap::RetargetCodeObject(
      fake_elf, sizeof(fake_elf), nullptr, kGfx1250Isa, &rewritten_elf_buffer, &rewritten_elf_size);
  check(!source_missing_rewritten, "RetargetCodeObject rejects null source_isa");
  const bool target_missing_rewritten = rocr::hotswap::RetargetCodeObject(
      fake_elf, sizeof(fake_elf), kGfx1250Isa, nullptr, &rewritten_elf_buffer, &rewritten_elf_size);
  check(!target_missing_rewritten, "RetargetCodeObject rejects null target_isa");
}

void test_RuntimeLoadUsesRewrittenCodeObject() {
  printf("TEST RuntimeLoadUsesRewrittenCodeObject...\n");
  ResetRuntimeTestEnv();
  LoadRecorder load;
  hsa_loaded_code_object_t loaded{};
  const hsa_executable_t executable = MakeTestExecutable(0x501);
  const hsa_status_t status = rocr::hotswap::LoadAgentCodeObjectWithHotswap(
      executable, MakeTestAgent(), MakeRealCodeObjectView(), nullptr, &loaded,
      MakeLoadCallbacks(&load));

  check(status == HSA_STATUS_SUCCESS, "runtime helper reports rewritten load success");
  check(load.calls.size() == 1, "runtime helper made one load call");
  check(load.calls.size() == 1 && load.calls[0].path == LoadPath::kRewritten,
        "runtime helper chose rewritten code object");
  check(load.calls.size() == 1 &&
            load.calls[0].code_object != static_cast<const void*>(kGfx1250MinCo),
        "rewritten load receives a fresh code object pointer");
  check(load.calls.size() == 1 && load.calls[0].code_object_size > 0,
        "rewritten load receives a non-zero code object size");
  check(load.calls.size() == 1 && load.calls[0].uri == "memory://gfx1250_min.hsaco",
        "runtime helper preserves the reader URI");
  check(rocr::hotswap::RetainedRewrittenElfBufferCountForTesting(executable) == 1,
        "successful rewritten load retains the rewritten ELF");
  rocr::hotswap::ReleaseRetainedRewrittenElfBuffers(executable);
  check(rocr::hotswap::RetainedRewrittenElfBufferCountForTesting(executable) == 0,
        "executable destroy releases retained rewritten ELFs");
}

void test_RuntimeLoadNonA0FallsBackToOriginal() {
  printf("TEST RuntimeLoadNonA0FallsBackToOriginal...\n");
  ResetRuntimeTestEnv();
  g_fake_hsa_env.asic_revision = 1;
  LoadRecorder load;
  const hsa_executable_t executable = MakeTestExecutable(0x502);
  const hsa_status_t status = rocr::hotswap::LoadAgentCodeObjectWithHotswap(
      executable, MakeTestAgent(), MakeRealCodeObjectView(), nullptr, nullptr,
      MakeLoadCallbacks(&load));

  check(status == HSA_STATUS_SUCCESS, "non-A0 runtime load succeeds");
  check(load.calls.size() == 1 && load.calls[0].path == LoadPath::kOriginal,
        "non-A0 runtime load uses the original code object");
  check(load.calls.size() == 1 &&
            load.calls[0].code_object == static_cast<const void*>(kGfx1250MinCo),
        "non-A0 original load receives the input code object");
  check(rocr::hotswap::RetainedRewrittenElfBufferCountForTesting(executable) == 0,
        "non-A0 fallback does not retain a rewritten ELF");
}

void test_RuntimeLoadDisableEnvFallsBackToOriginal() {
  printf("TEST RuntimeLoadDisableEnvFallsBackToOriginal...\n");
  ResetRuntimeTestEnv();
  g_fake_env_vars["HSA_HOTSWAP_DISABLE"] = "1";
  LoadRecorder load;
  const hsa_executable_t executable = MakeTestExecutable(0x503);
  const hsa_status_t status = rocr::hotswap::LoadAgentCodeObjectWithHotswap(
      executable, MakeTestAgent(), MakeRealCodeObjectView(), nullptr, nullptr,
      MakeLoadCallbacks(&load));

  check(status == HSA_STATUS_SUCCESS, "disabled runtime load succeeds");
  check(load.calls.size() == 1 && load.calls[0].path == LoadPath::kOriginal,
        "disable env var uses the original code object");
  check(rocr::hotswap::RetainedRewrittenElfBufferCountForTesting(executable) == 0,
        "disable env var does not retain a rewritten ELF");
}

void test_RuntimeLoadRewriteFailureFallsBackToOriginal() {
  printf("TEST RuntimeLoadRewriteFailureFallsBackToOriginal...\n");
  ResetRuntimeTestEnv();
  const unsigned char fake_elf[] = {0x7f, 'E',  'L',  'F',  0x02, 0x01, 0x01, 0x00,
                                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  rocr::hotswap::CodeObjectView code_object;
  code_object.data = fake_elf;
  code_object.size = sizeof(fake_elf);
  code_object.uri = "memory://invalid.hsaco";

  LoadRecorder load;
  const hsa_executable_t executable = MakeTestExecutable(0x504);
  const hsa_status_t status = rocr::hotswap::LoadAgentCodeObjectWithHotswap(
      executable, MakeTestAgent(), code_object, nullptr, nullptr, MakeLoadCallbacks(&load));

  check(status == HSA_STATUS_SUCCESS, "rewrite failure runtime fallback succeeds");
  check(load.calls.size() == 1 && load.calls[0].path == LoadPath::kOriginal,
        "rewrite failure uses the original code object");
  check(load.calls.size() == 1 && load.calls[0].code_object == fake_elf,
        "rewrite failure fallback receives the input code object");
  check(rocr::hotswap::RetainedRewrittenElfBufferCountForTesting(executable) == 0,
        "rewrite failure does not retain a rewritten ELF");
}

void test_RuntimeLoadRewrittenLoadFailureFallsBackToOriginal() {
  printf("TEST RuntimeLoadRewrittenLoadFailureFallsBackToOriginal...\n");
  ResetRuntimeTestEnv();
  LoadRecorder load;
  load.rewritten_status = HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  const hsa_executable_t executable = MakeTestExecutable(0x505);
  const hsa_status_t status = rocr::hotswap::LoadAgentCodeObjectWithHotswap(
      executable, MakeTestAgent(), MakeRealCodeObjectView(), nullptr, nullptr,
      MakeLoadCallbacks(&load));

  check(status == HSA_STATUS_SUCCESS, "rewritten-load failure returns original-load status");
  check(load.calls.size() == 2, "rewritten-load failure makes two load calls");
  check(load.calls.size() == 2 && load.calls[0].path == LoadPath::kRewritten &&
            load.calls[1].path == LoadPath::kOriginal,
        "rewritten-load failure falls back to original after rewritten attempt");
  check(load.calls.size() == 2 &&
            load.calls[1].code_object == static_cast<const void*>(kGfx1250MinCo),
        "rewritten-load failure fallback receives the input code object");
  check(rocr::hotswap::RetainedRewrittenElfBufferCountForTesting(executable) == 0,
        "failed rewritten load does not retain the rewritten ELF");
}

}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IOLBF, 0);

  test_GetIsaNameRealCodeObject();
  test_GetIsaNameInvalidCodeObject();
  test_RetargetRealCodeObject();
  test_RetargetInvalidCodeObjectFails();
  test_RetargetNullOutputPointers();
  test_RetargetNullInputs();
  test_RetargetNullSourceOrTarget();
  test_RuntimeLoadUsesRewrittenCodeObject();
  test_RuntimeLoadNonA0FallsBackToOriginal();
  test_RuntimeLoadDisableEnvFallsBackToOriginal();
  test_RuntimeLoadRewriteFailureFallsBackToOriginal();
  test_RuntimeLoadRewrittenLoadFailureFallsBackToOriginal();

  printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
  return tests_failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
