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

struct FakeEnv {
  std::string isa_name;
  bool asic_rev_ok = true;
  uint32_t asic_revision = 0;
  int isa_query_calls = 0;
  int asic_rev_calls = 0;
};

FakeEnv g_env;

}  // namespace

#include "core/runtime/hotswap_gfx_query.hpp"
#include "core/inc/hsa_internal.h"

namespace rocr {
namespace HSA {

hsa_status_t hsa_agent_iterate_isas(hsa_agent_t /*agent*/,
                                    hsa_status_t (*callback)(hsa_isa_t isa,
                                                             void* data),
                                    void* data) {
  ++g_env.isa_query_calls;
  hsa_isa_t isa{};
  isa.handle = 1;
  return callback(isa, data);
}

hsa_status_t hsa_isa_get_info_alt(hsa_isa_t /*isa*/, hsa_isa_info_t attribute,
                                  void* value) {
  if (attribute == HSA_ISA_INFO_NAME_LENGTH) {
    *static_cast<uint32_t*>(value) =
        static_cast<uint32_t>(g_env.isa_name.size() + 1);
    return HSA_STATUS_SUCCESS;
  }
  if (attribute == HSA_ISA_INFO_NAME) {
    std::memcpy(value, g_env.isa_name.c_str(), g_env.isa_name.size() + 1);
    return HSA_STATUS_SUCCESS;
  }
  return HSA_STATUS_ERROR;
}

hsa_status_t hsa_agent_get_info(hsa_agent_t /*agent*/,
                                hsa_agent_info_t attribute, void* value) {
  if (attribute ==
      static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_ASIC_REVISION)) {
    ++g_env.asic_rev_calls;
    if (!g_env.asic_rev_ok) {
      return HSA_STATUS_ERROR;
    }
    *static_cast<uint32_t*>(value) = g_env.asic_revision;
    return HSA_STATUS_SUCCESS;
  }
  return HSA_STATUS_ERROR;
}

}  // namespace HSA
}  // namespace rocr

using rocr::hotswap::AgentGfxRevision;
using rocr::hotswap::GateAllowsHotswap;
using rocr::hotswap::QueryAgentGfxRevision;
using rocr::hotswap::ResetGfxRevisionCache;

namespace {

int tests_total = 0;
int tests_failed = 0;

void run(const char* name, bool cond) {
  printf("  %s: %s\n", cond ? "PASS" : "FAIL", name);
  if (!cond) {
    ++tests_failed;
  }
  ++tests_total;
}

void reset_env() {
  g_env = FakeEnv{};
  ResetGfxRevisionCache();
}

constexpr uint64_t kFirstTestAgentHandle = 1;
uint64_t g_next_handle = kFirstTestAgentHandle;

hsa_agent_t fresh_agent() {
  hsa_agent_t a{};
  a.handle = g_next_handle++;
  return a;
}

const char* kGfx1250Isa = "amdgcn-amd-amdhsa--gfx1250";
const char* kGfx1250IsaWithFeatures =
    "amdgcn-amd-amdhsa--gfx1250:sramecc+:xnack-";
const char* kGfx942Isa = "amdgcn-amd-amdhsa--gfx942";
const char* kGfx1251Isa = "amdgcn-amd-amdhsa--gfx1251";

void test_Gfx1250A0Passes() {
  printf("TEST Gfx1250A0Passes...\n");
  reset_env();
  g_env.isa_name = kGfx1250Isa;
  g_env.asic_revision = 0;
  const AgentGfxRevision g = QueryAgentGfxRevision(fresh_agent());
  run("gfx target parsed as gfx1250", g.gfx_target == "gfx1250");
  run("ASIC revision is A0 (0)", g.revision_valid && g.asic_revision == 0);
  run("gate allows gfx1250 A0", GateAllowsHotswap(g));
}

void test_Gfx1250FeatureSuffixParsed() {
  printf("TEST Gfx1250FeatureSuffixParsed...\n");
  reset_env();
  g_env.isa_name = kGfx1250IsaWithFeatures;
  g_env.asic_revision = 0;
  const AgentGfxRevision g = QueryAgentGfxRevision(fresh_agent());
  run("feature suffix stripped to gfx1250", g.gfx_target == "gfx1250");
  run("gate allows suffixed gfx1250 A0", GateAllowsHotswap(g));
}

void test_NonGfx1250Blocks() {
  printf("TEST NonGfx1250Blocks...\n");
  reset_env();
  g_env.isa_name = kGfx942Isa;
  g_env.asic_revision = 0;
  const AgentGfxRevision g = QueryAgentGfxRevision(fresh_agent());
  run("gfx target parsed as gfx942", g.gfx_target == "gfx942");
  run("gate blocks gfx942", !GateAllowsHotswap(g));
}

void test_NearMissTargetBlocks() {
  printf("TEST NearMissTargetBlocks...\n");
  reset_env();
  g_env.isa_name = kGfx1251Isa;
  g_env.asic_revision = 0;
  const AgentGfxRevision g = QueryAgentGfxRevision(fresh_agent());
  run("gfx target parsed as gfx1251", g.gfx_target == "gfx1251");
  run("gate blocks gfx1251 exact-match miss", !GateAllowsHotswap(g));
}

void test_Gfx1250NonA0Blocks() {
  printf("TEST Gfx1250NonA0Blocks...\n");
  reset_env();
  g_env.isa_name = kGfx1250Isa;
  g_env.asic_revision = 1;
  const AgentGfxRevision g = QueryAgentGfxRevision(fresh_agent());
  run("ASIC revision is non-A0", g.revision_valid && g.asic_revision == 1);
  run("gate blocks gfx1250 non-A0", !GateAllowsHotswap(g));
}

void test_AsicRevisionQueryFailure() {
  printf("TEST AsicRevisionQueryFailure...\n");
  reset_env();
  g_env.isa_name = kGfx1250Isa;
  g_env.asic_rev_ok = false;
  const AgentGfxRevision g = QueryAgentGfxRevision(fresh_agent());
  run("gfx target still parsed", g.gfx_target == "gfx1250");
  run("revision marked invalid", !g.revision_valid);
  run("gate blocks when ASIC revision is unavailable", !GateAllowsHotswap(g));
  run("ASIC revision query was attempted once",
      g.asic_revision == 0 && g_env.asic_rev_calls == 1);
}

void test_ResultIsCachedPerHandle() {
  printf("TEST ResultIsCachedPerHandle...\n");
  reset_env();
  g_env.isa_name = kGfx1250Isa;
  g_env.asic_revision = 0;
  const hsa_agent_t agent = fresh_agent();
  const AgentGfxRevision first = QueryAgentGfxRevision(agent);
  const int isa_after_first = g_env.isa_query_calls;
  const int rev_after_first = g_env.asic_rev_calls;
  const AgentGfxRevision second = QueryAgentGfxRevision(agent);
  run("cached result stays gfx1250",
      first.gfx_target == "gfx1250" && second.gfx_target == "gfx1250");
  run("second call served from cache",
      g_env.isa_query_calls == isa_after_first &&
          g_env.asic_rev_calls == rev_after_first);
}

void test_DistinctHandlesIndependent() {
  printf("TEST DistinctHandlesIndependent...\n");
  reset_env();
  g_env.isa_name = kGfx1250Isa;
  g_env.asic_revision = 0;
  const AgentGfxRevision a1 = QueryAgentGfxRevision(fresh_agent());

  g_env.isa_name = kGfx942Isa;
  const AgentGfxRevision a2 = QueryAgentGfxRevision(fresh_agent());
  run("first handle passes", GateAllowsHotswap(a1));
  run("distinct handle evaluated independently",
      a2.gfx_target == "gfx942" && !GateAllowsHotswap(a2));
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
