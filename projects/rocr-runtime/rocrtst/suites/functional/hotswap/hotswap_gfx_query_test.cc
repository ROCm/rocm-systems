//===- hotswap_gfx_query_test.cc - HotSwap gfx / ASIC gate tests ---------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "core/inc/hotswap_gfx_query.hpp"

#include <cstdint>
#include <cstring>
#include <string>

#include "core/inc/hsa_internal.h"
#include "gtest/gtest.h"

namespace {

struct FakeHsaEnv {
  std::string isa_name;
  bool asic_rev_ok = true;
  uint32_t asic_revision = 0;
  int isa_query_calls = 0;
  int asic_rev_calls = 0;
};

FakeHsaEnv g_fake_hsa_env;

constexpr uint64_t kFirstTestAgentHandle = 1;
uint64_t g_next_handle = kFirstTestAgentHandle;

const char* kGfx1250Isa = "amdgcn-amd-amdhsa--gfx1250";
const char* kGfx1250IsaWithFeatures =
    "amdgcn-amd-amdhsa--gfx1250:sramecc+:xnack-";
const char* kGfx942Isa = "amdgcn-amd-amdhsa--gfx942";
const char* kGfx1251Isa = "amdgcn-amd-amdhsa--gfx1251";
const char* kGfx12_5GenericIsaWithFeatures =
    "amdgcn-amd-amdhsa--gfx12-5-generic:sramecc+";

void ResetTestEnv() {
  g_fake_hsa_env = FakeHsaEnv{};
  g_next_handle = kFirstTestAgentHandle;
  rocr::hotswap::ResetAgentGfxRevisionCache();
}

hsa_agent_t MakeFreshAgent() {
  hsa_agent_t agent{};
  agent.handle = g_next_handle++;
  return agent;
}

}  // namespace

namespace rocr {
namespace HSA {

hsa_status_t hsa_agent_iterate_isas(hsa_agent_t /*agent*/,
                                    hsa_status_t (*callback)(hsa_isa_t isa,
                                                             void* data),
                                    void* data) {
  ++g_fake_hsa_env.isa_query_calls;
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

namespace {

using rocr::hotswap::AgentGfxRevision;
using rocr::hotswap::GetAgentGfxRevision;
using rocr::hotswap::IsGfx12_5Target;
using rocr::hotswap::IsHotswapSupportedGfxRevision;

TEST(HotswapGfxQuery, Gfx1250A0Passes) {
  ResetTestEnv();
  g_fake_hsa_env.isa_name = kGfx1250Isa;
  g_fake_hsa_env.asic_revision = 0;

  const AgentGfxRevision revision = GetAgentGfxRevision(MakeFreshAgent());

  EXPECT_EQ(revision.gfx_target, "gfx1250");
  EXPECT_TRUE(revision.has_asic_revision);
  EXPECT_EQ(revision.asic_revision, 0u);
  EXPECT_TRUE(IsHotswapSupportedGfxRevision(revision));
}

TEST(HotswapGfxQuery, Gfx1250FeatureSuffixParsed) {
  ResetTestEnv();
  g_fake_hsa_env.isa_name = kGfx1250IsaWithFeatures;
  g_fake_hsa_env.asic_revision = 0;

  const AgentGfxRevision revision = GetAgentGfxRevision(MakeFreshAgent());

  EXPECT_EQ(revision.gfx_target, "gfx1250");
  EXPECT_TRUE(IsHotswapSupportedGfxRevision(revision));
}

TEST(HotswapGfxQuery, NonGfx1250Blocks) {
  ResetTestEnv();
  g_fake_hsa_env.isa_name = kGfx942Isa;
  g_fake_hsa_env.asic_revision = 0;

  const AgentGfxRevision revision = GetAgentGfxRevision(MakeFreshAgent());

  EXPECT_EQ(revision.gfx_target, "gfx942");
  EXPECT_FALSE(IsHotswapSupportedGfxRevision(revision));
}

TEST(HotswapGfxQuery, NearMissTargetBlocks) {
  ResetTestEnv();
  g_fake_hsa_env.isa_name = kGfx1251Isa;
  g_fake_hsa_env.asic_revision = 0;

  const AgentGfxRevision revision = GetAgentGfxRevision(MakeFreshAgent());

  EXPECT_EQ(revision.gfx_target, "gfx1251");
  EXPECT_FALSE(IsHotswapSupportedGfxRevision(revision));
}

TEST(HotswapGfxQuery, Gfx12_5GenericFeatureSuffixParsed) {
  ResetTestEnv();
  g_fake_hsa_env.isa_name = kGfx12_5GenericIsaWithFeatures;
  g_fake_hsa_env.asic_revision = 0;

  const AgentGfxRevision revision = GetAgentGfxRevision(MakeFreshAgent());

  EXPECT_EQ(revision.gfx_target, "gfx12-5-generic");
  EXPECT_FALSE(IsHotswapSupportedGfxRevision(revision));
}

TEST(HotswapGfxQuery, Gfx12_5TargetPredicateIsStrict) {
  EXPECT_TRUE(IsGfx12_5Target("gfx1250"));
  EXPECT_TRUE(IsGfx12_5Target("gfx1251"));
  EXPECT_TRUE(IsGfx12_5Target("gfx12-5-generic"));
  EXPECT_FALSE(IsGfx12_5Target("gfx125"));
  EXPECT_FALSE(IsGfx12_5Target("gfx125foo"));
  EXPECT_FALSE(IsGfx12_5Target("gfx942"));
}

TEST(HotswapGfxQuery, Gfx1250NonA0Blocks) {
  ResetTestEnv();
  g_fake_hsa_env.isa_name = kGfx1250Isa;
  g_fake_hsa_env.asic_revision = 1;

  const AgentGfxRevision revision = GetAgentGfxRevision(MakeFreshAgent());

  EXPECT_TRUE(revision.has_asic_revision);
  EXPECT_EQ(revision.asic_revision, 1u);
  EXPECT_FALSE(IsHotswapSupportedGfxRevision(revision));
}

TEST(HotswapGfxQuery, AsicRevisionQueryFailure) {
  ResetTestEnv();
  g_fake_hsa_env.isa_name = kGfx1250Isa;
  g_fake_hsa_env.asic_rev_ok = false;

  const AgentGfxRevision revision = GetAgentGfxRevision(MakeFreshAgent());

  EXPECT_EQ(revision.gfx_target, "gfx1250");
  EXPECT_FALSE(revision.has_asic_revision);
  EXPECT_FALSE(IsHotswapSupportedGfxRevision(revision));
  EXPECT_EQ(revision.asic_revision, 0u);
  EXPECT_EQ(g_fake_hsa_env.asic_rev_calls, 1);
}

TEST(HotswapGfxQuery, ResultIsCachedPerHandle) {
  ResetTestEnv();
  g_fake_hsa_env.isa_name = kGfx1250Isa;
  g_fake_hsa_env.asic_revision = 0;
  const hsa_agent_t agent = MakeFreshAgent();

  const AgentGfxRevision first = GetAgentGfxRevision(agent);
  const int isa_after_first = g_fake_hsa_env.isa_query_calls;
  const int rev_after_first = g_fake_hsa_env.asic_rev_calls;
  const AgentGfxRevision second = GetAgentGfxRevision(agent);

  EXPECT_EQ(first.gfx_target, "gfx1250");
  EXPECT_EQ(second.gfx_target, "gfx1250");
  EXPECT_EQ(g_fake_hsa_env.isa_query_calls, isa_after_first);
  EXPECT_EQ(g_fake_hsa_env.asic_rev_calls, rev_after_first);
}

TEST(HotswapGfxQuery, DistinctHandlesIndependent) {
  ResetTestEnv();
  g_fake_hsa_env.isa_name = kGfx1250Isa;
  g_fake_hsa_env.asic_revision = 0;
  const AgentGfxRevision first_revision = GetAgentGfxRevision(MakeFreshAgent());

  g_fake_hsa_env.isa_name = kGfx942Isa;
  const AgentGfxRevision second_revision =
      GetAgentGfxRevision(MakeFreshAgent());

  EXPECT_TRUE(IsHotswapSupportedGfxRevision(first_revision));
  EXPECT_EQ(second_revision.gfx_target, "gfx942");
  EXPECT_FALSE(IsHotswapSupportedGfxRevision(second_revision));
}

}  // namespace
