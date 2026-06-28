//===- hotswap_rewrite_test.cpp - HotSwap rewrite tests -------------------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "core/inc/hsa_internal.h"
#include "core/runtime/hotswap.hpp"
#include "core/util/os.h"

#include "gfx1250_min_hsaco.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>

namespace rocr {
namespace os {

LibHandle LoadLib(std::string filename) {
  return dlopen(filename.c_str(), RTLD_NOW | RTLD_LOCAL);
}

void* GetExportAddress(LibHandle lib, std::string export_name) {
  return dlsym(lib, export_name.c_str());
}

bool CloseLib(LibHandle lib) { return dlclose(lib) == 0; }

bool IsEnvVarSet(std::string env_var_name) {
  return std::getenv(env_var_name.c_str()) != nullptr;
}

std::string GetEnvVar(std::string env_var_name) {
  const char* value = std::getenv(env_var_name.c_str());
  return value ? value : "";
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
  static constexpr const char* kGfx1250Isa = "amdgcn-amd-amdhsa--gfx1250";
  if (attribute == HSA_ISA_INFO_NAME_LENGTH) {
    *static_cast<uint32_t*>(value) = std::strlen(kGfx1250Isa) + 1;
    return HSA_STATUS_SUCCESS;
  }
  if (attribute == HSA_ISA_INFO_NAME) {
    std::memcpy(value, kGfx1250Isa, std::strlen(kGfx1250Isa) + 1);
    return HSA_STATUS_SUCCESS;
  }
  return HSA_STATUS_ERROR;
}

hsa_status_t hsa_agent_get_info(hsa_agent_t /*agent*/,
                                hsa_agent_info_t attribute, void* value) {
  if (attribute ==
      static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_ASIC_REVISION)) {
    *static_cast<uint32_t*>(value) = 0;
    return HSA_STATUS_SUCCESS;
  }
  return HSA_STATUS_ERROR;
}

}  // namespace HSA
}  // namespace rocr

namespace {

int tests_passed = 0;
int tests_failed = 0;

static constexpr const char* kGfx1250Isa = "amdgcn-amd-amdhsa--gfx1250";

void check(bool cond, const char* name) {
  if (cond) {
    ++tests_passed;
    printf("  PASS: %s\n", name);
  } else {
    ++tests_failed;
    fprintf(stderr, "  FAIL: %s\n", name);
  }
}

void test_GetIsaNameRealCodeObject() {
  printf("TEST GetIsaNameRealCodeObject...\n");
  const std::string isa = rocr::hotswap::GetCodeObjectIsaName(
      kGfx1250MinCo, sizeof(kGfx1250MinCo));
  check(isa == kGfx1250Isa, "GetCodeObjectIsaName reads gfx1250 from a real CO");
}

void test_GetIsaNameInvalidCodeObject() {
  printf("TEST GetIsaNameInvalidCodeObject...\n");
  const unsigned char fake_elf[] = {0x7f, 'E', 'L', 'F', 0x02, 0x01, 0x01,
                                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                    0x00, 0x00};
  const std::string isa =
      rocr::hotswap::GetCodeObjectIsaName(fake_elf, sizeof(fake_elf));
  check(isa.empty(), "GetCodeObjectIsaName returns empty for an invalid CO");
}

void test_RetargetRealCodeObject() {
  printf("TEST RetargetRealCodeObject...\n");
  rocr::hotswap::OwnedElf out_data(nullptr, &std::free);
  size_t out_size = 0;
  const bool rewritten = rocr::hotswap::RetargetCodeObject(
      kGfx1250MinCo, sizeof(kGfx1250MinCo), kGfx1250Isa, kGfx1250Isa,
      &out_data, &out_size);

  check(rewritten, "RetargetCodeObject rewrites a real CO");
  check(out_data != nullptr &&
            out_data.get() != static_cast<const void*>(kGfx1250MinCo),
        "out_data is a fresh allocation");
  check(out_size > 0, "out_size is non-zero");
  check(rocr::hotswap::GetCodeObjectIsaName(out_data.get(), out_size) ==
            kGfx1250Isa,
        "rewritten CO keeps the expected ISA name");
}

void test_RetargetInvalidCodeObjectFails() {
  printf("TEST RetargetInvalidCodeObjectFails...\n");
  const unsigned char fake_elf[] = {0x7f, 'E', 'L', 'F', 0x02, 0x01, 0x01,
                                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                    0x00, 0x00};
  rocr::hotswap::OwnedElf out_data(nullptr, &std::free);
  size_t out_size = 0;
  const bool rewritten = rocr::hotswap::RetargetCodeObject(
      fake_elf, sizeof(fake_elf), kGfx1250Isa, kGfx1250Isa, &out_data,
      &out_size);

  check(!rewritten, "RetargetCodeObject fails for an un-rewritable CO");
  check(out_data == nullptr, "out_data remains empty after failure");
  check(out_size == 0, "out_size remains zero after failure");
}

void test_RetargetNullOutputPointers() {
  printf("TEST RetargetNullOutputPointers...\n");
  const unsigned char fake_elf[] = {0x7f, 'E', 'L', 'F'};
  const bool rewritten = rocr::hotswap::RetargetCodeObject(
      fake_elf, sizeof(fake_elf), kGfx1250Isa, kGfx1250Isa, nullptr, nullptr);
  check(!rewritten, "RetargetCodeObject rejects null output pointers");
}

void test_RetargetNullInputs() {
  printf("TEST RetargetNullInputs...\n");
  rocr::hotswap::OwnedElf out_data(nullptr, &std::free);
  size_t out_size = 0;
  const bool rewritten = rocr::hotswap::RetargetCodeObject(
      nullptr, 0, kGfx1250Isa, kGfx1250Isa, &out_data, &out_size);
  check(!rewritten, "RetargetCodeObject rejects null elf_data");
}

void test_RetargetNullSourceOrTarget() {
  printf("TEST RetargetNullSourceOrTarget...\n");
  const unsigned char fake_elf[] = {0x7f, 'E', 'L', 'F'};
  rocr::hotswap::OwnedElf out_data(nullptr, &std::free);
  size_t out_size = 0;
  const bool src_rewritten = rocr::hotswap::RetargetCodeObject(
      fake_elf, sizeof(fake_elf), nullptr, kGfx1250Isa, &out_data, &out_size);
  check(!src_rewritten, "RetargetCodeObject rejects null source_isa");
  const bool tgt_rewritten = rocr::hotswap::RetargetCodeObject(
      fake_elf, sizeof(fake_elf), kGfx1250Isa, nullptr, &out_data, &out_size);
  check(!tgt_rewritten, "RetargetCodeObject rejects null target_isa");
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

  printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
  return tests_failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
