//===- hotswap_gfx_query_test.cpp - HotSwap gfx / ASIC gate tests ---------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

struct FakeHsaEnv {
  std::string isa_name;
  bool asic_rev_ok = true;
  uint32_t asic_revision = 0;
  int isa_query_calls = 0;
  int asic_rev_calls = 0;
};

FakeHsaEnv g_fake_hsa_env;

}  // namespace

#include "core/runtime/hotswap_gfx_query.hpp"
#include "core/inc/hsa_internal.h"

namespace rocr {
namespace HSA {

hsa_status_t hsa_agent_iterate_isas(hsa_agent_t /*agent*/,
                                    hsa_status_t (*callback)(hsa_isa_t isa, void* data),
                                    void* data) {
  ++g_fake_hsa_env.isa_query_calls;
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
    ++g_fake_hsa_env.asic_rev_calls;
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

using rocr::hotswap::AgentGfxRevision;
using rocr::hotswap::GetAgentGfxRevision;
using rocr::hotswap::IsHotswapSupportedGfxRevision;
using rocr::hotswap::ResetAgentGfxRevisionCache;

namespace {

int tests_total = 0;
int tests_failed = 0;

void check(const char* name, bool cond) {
  printf("  %s: %s\n", cond ? "PASS" : "FAIL", name);
  if (!cond) {
    ++tests_failed;
  }
  ++tests_total;
}

void ResetTestEnv() {
  g_fake_hsa_env = FakeHsaEnv{};
  ResetAgentGfxRevisionCache();
}

constexpr uint64_t kFirstTestAgentHandle = 1;
uint64_t g_next_handle = kFirstTestAgentHandle;

hsa_agent_t MakeFreshAgent() {
  hsa_agent_t a{};
  a.handle = g_next_handle++;
  return a;
}

const char* kGfx1250Isa = "amdgcn-amd-amdhsa--gfx1250";
const char* kGfx1250IsaWithFeatures = "amdgcn-amd-amdhsa--gfx1250:sramecc+:xnack-";
const char* kGfx942Isa = "amdgcn-amd-amdhsa--gfx942";
const char* kGfx1251Isa = "amdgcn-amd-amdhsa--gfx1251";

void test_Gfx1250A0Passes() {
  printf("TEST Gfx1250A0Passes...\n");
  ResetTestEnv();
  g_fake_hsa_env.isa_name = kGfx1250Isa;
  g_fake_hsa_env.asic_revision = 0;
  const AgentGfxRevision revision = GetAgentGfxRevision(MakeFreshAgent());
  check("gfx target parsed as gfx1250", revision.gfx_target == "gfx1250");
  check("ASIC revision is A0 (0)", revision.has_asic_revision && revision.asic_revision == 0);
  check("gate allows gfx1250 A0", IsHotswapSupportedGfxRevision(revision));
}

void test_Gfx1250FeatureSuffixParsed() {
  printf("TEST Gfx1250FeatureSuffixParsed...\n");
  ResetTestEnv();
  g_fake_hsa_env.isa_name = kGfx1250IsaWithFeatures;
  g_fake_hsa_env.asic_revision = 0;
  const AgentGfxRevision revision = GetAgentGfxRevision(MakeFreshAgent());
  check("feature suffix stripped to gfx1250", revision.gfx_target == "gfx1250");
  check("gate allows suffixed gfx1250 A0", IsHotswapSupportedGfxRevision(revision));
}

void test_NonGfx1250Blocks() {
  printf("TEST NonGfx1250Blocks...\n");
  ResetTestEnv();
  g_fake_hsa_env.isa_name = kGfx942Isa;
  g_fake_hsa_env.asic_revision = 0;
  const AgentGfxRevision revision = GetAgentGfxRevision(MakeFreshAgent());
  check("gfx target parsed as gfx942", revision.gfx_target == "gfx942");
  check("gate blocks gfx942", !IsHotswapSupportedGfxRevision(revision));
}

void test_NearMissTargetBlocks() {
  printf("TEST NearMissTargetBlocks...\n");
  ResetTestEnv();
  g_fake_hsa_env.isa_name = kGfx1251Isa;
  g_fake_hsa_env.asic_revision = 0;
  const AgentGfxRevision revision = GetAgentGfxRevision(MakeFreshAgent());
  check("gfx target parsed as gfx1251", revision.gfx_target == "gfx1251");
  check("gate blocks gfx1251 exact-match miss", !IsHotswapSupportedGfxRevision(revision));
}

void test_Gfx1250NonA0Blocks() {
  printf("TEST Gfx1250NonA0Blocks...\n");
  ResetTestEnv();
  g_fake_hsa_env.isa_name = kGfx1250Isa;
  g_fake_hsa_env.asic_revision = 1;
  const AgentGfxRevision revision = GetAgentGfxRevision(MakeFreshAgent());
  check("ASIC revision is non-A0", revision.has_asic_revision && revision.asic_revision == 1);
  check("gate blocks gfx1250 non-A0", !IsHotswapSupportedGfxRevision(revision));
}

void test_AsicRevisionQueryFailure() {
  printf("TEST AsicRevisionQueryFailure...\n");
  ResetTestEnv();
  g_fake_hsa_env.isa_name = kGfx1250Isa;
  g_fake_hsa_env.asic_rev_ok = false;
  const AgentGfxRevision revision = GetAgentGfxRevision(MakeFreshAgent());
  check("gfx target still parsed", revision.gfx_target == "gfx1250");
  check("revision marked invalid", !revision.has_asic_revision);
  check("gate blocks when ASIC revision is unavailable", !IsHotswapSupportedGfxRevision(revision));
  check("ASIC revision query was attempted once",
        revision.asic_revision == 0 && g_fake_hsa_env.asic_rev_calls == 1);
}

void test_ResultIsCachedPerHandle() {
  printf("TEST ResultIsCachedPerHandle...\n");
  ResetTestEnv();
  g_fake_hsa_env.isa_name = kGfx1250Isa;
  g_fake_hsa_env.asic_revision = 0;
  const hsa_agent_t agent = MakeFreshAgent();
  const AgentGfxRevision first = GetAgentGfxRevision(agent);
  const int isa_after_first = g_fake_hsa_env.isa_query_calls;
  const int rev_after_first = g_fake_hsa_env.asic_rev_calls;
  const AgentGfxRevision second = GetAgentGfxRevision(agent);
  check("cached result stays gfx1250",
        first.gfx_target == "gfx1250" && second.gfx_target == "gfx1250");
  check("second call served from cache",
        g_fake_hsa_env.isa_query_calls == isa_after_first &&
            g_fake_hsa_env.asic_rev_calls == rev_after_first);
}

void test_DistinctHandlesIndependent() {
  printf("TEST DistinctHandlesIndependent...\n");
  ResetTestEnv();
  g_fake_hsa_env.isa_name = kGfx1250Isa;
  g_fake_hsa_env.asic_revision = 0;
  const AgentGfxRevision first_revision = GetAgentGfxRevision(MakeFreshAgent());

  g_fake_hsa_env.isa_name = kGfx942Isa;
  const AgentGfxRevision second_revision = GetAgentGfxRevision(MakeFreshAgent());
  check("first handle passes", IsHotswapSupportedGfxRevision(first_revision));
  check("distinct handle evaluated independently",
        second_revision.gfx_target == "gfx942" && !IsHotswapSupportedGfxRevision(second_revision));
}

}  // namespace

int main() {
  test_Gfx1250A0Passes();
  test_Gfx1250FeatureSuffixParsed();
  test_NonGfx1250Blocks();
  test_NearMissTargetBlocks();
  test_Gfx1250NonA0Blocks();
  test_AsicRevisionQueryFailure();
  test_ResultIsCachedPerHandle();
  test_DistinctHandlesIndependent();

  printf("\n%d passed, %d failed\n", tests_total - tests_failed, tests_failed);
  return tests_failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
