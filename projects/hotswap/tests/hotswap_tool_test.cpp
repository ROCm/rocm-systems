//===- hotswap_tool_test.cpp - Tests for gfx-target / ASIC-rev query ------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for query_agent_gfx_revision() in hotswap_gfx_query.cpp and the
// COMGR requests selected by hotswap_loader_policy.cpp.
//
// The HSA entry points used by the query helper are replaced with in-file stubs
// (linked in place of the real libraries) so query and policy behavior can be
// driven entirely from the test without GPU hardware:
//
//   * ISA name        <- hsa_agent_iterate_isas / hsa_isa_get_info_alt
//   * ASIC revision   <- hsa_agent_get_info(HSA_AMD_AGENT_INFO_ASIC_REVISION)
//
// This path is portable: it depends only on HSA so the same query/gate logic is 
// exercised for both Linux and Windows builds.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>

// ---------------------------------------------------------------------------
// Test-controlled fake environment. Set before each call into the tool.
// ---------------------------------------------------------------------------
namespace {
struct FakeEnv {
  std::string isa_name;          // ISA reported for the agent
  bool asic_rev_ok = true;       // hsa_agent_get_info(ASIC_REVISION) success
  uint32_t asic_revision = 0;    // reported ASIC revision (0 == A0)

  // Counters used to assert short-circuiting and caching behavior.
  int isa_query_calls = 0;       // get_agent_isa_name -> iterate_isas
  int asic_rev_calls = 0;        // ASIC revision queries
};
FakeEnv g_env;
}  // namespace

// The units under test (bring in HSA query helpers and rewrite policy).
#include "hotswap_gfx_query.hpp"
#include "hotswap_loader_policy.hpp"

using rocr::hotswap::AgentGfxRevision;
using rocr::hotswap::decide_hotswap_rewrite;
using rocr::hotswap::gate_allows_hotswap;
using rocr::hotswap::has_candidate_hotswap_rewrite;
using rocr::hotswap::query_agent_gfx_revision;
using rocr::hotswap::reset_gfx_revision_cache;
using rocr::hotswap::RewriteDecision;
using rocr::hotswap::RewriteOptions;

// ---------------------------------------------------------------------------
// Stubs replacing the real HSA symbols referenced by the tool.
// ---------------------------------------------------------------------------
extern "C" {

hsa_status_t hsa_agent_iterate_isas(hsa_agent_t /*agent*/,
                                    hsa_status_t (*callback)(hsa_isa_t isa,
                                                             void *data),
                                    void *data) {
  ++g_env.isa_query_calls;
  hsa_isa_t isa{};
  isa.handle = 1;
  return callback(isa, data);
}

hsa_status_t hsa_isa_get_info_alt(hsa_isa_t /*isa*/, hsa_isa_info_t attribute,
                                  void *value) {
  if (attribute == HSA_ISA_INFO_NAME_LENGTH) {
    *static_cast<uint32_t *>(value) =
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
                                hsa_agent_info_t attribute, void *value) {
  // The tool only queries the ASIC revision through this entry point.
  if (attribute ==
      static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_ASIC_REVISION)) {
    ++g_env.asic_rev_calls;
    if (!g_env.asic_rev_ok) {
      return HSA_STATUS_ERROR;
    }
    *static_cast<uint32_t *>(value) = g_env.asic_revision;
    return HSA_STATUS_SUCCESS;
  }
  return HSA_STATUS_ERROR;
}

}  // extern "C"

// ---------------------------------------------------------------------------
// Minimal test harness.
// ---------------------------------------------------------------------------
namespace {

int tests_total = 0;
int tests_failed = 0;

void run(const char *name, bool cond) {
  printf("  %s: %s\n", cond ? "PASS" : "FAIL", name);
  if (!cond)
    ++tests_failed;
  ++tests_total;
}

// Reset both the fake HSA environment and the query module's per-handle cache
// before each test, so results never leak across tests via a reused handle.
void reset_env() {
  g_env = FakeEnv{};
  reset_gfx_revision_cache();
}

// Arbitrary non-zero seed for synthesized test agent handles. The value is not
// significant; it only needs to be non-zero (so a synthesized handle is never
// confused with a default-constructed hsa_agent_t whose handle is 0) and unique
// per fresh_agent() call.
constexpr uint64_t kFirstTestAgentHandle = 1;
uint64_t g_next_handle = kFirstTestAgentHandle;
hsa_agent_t fresh_agent() {
  hsa_agent_t a{};
  a.handle = g_next_handle++;
  return a;
}

const char *kGfx1250Isa = "amdgcn-amd-amdhsa--gfx1250";
const char *kGfx1250IsaWithFeatures =
    "amdgcn-amd-amdhsa--gfx1250:sramecc+:xnack-";
const char *kGfx942Isa = "amdgcn-amd-amdhsa--gfx942";
const char *kGfx1251Isa = "amdgcn-amd-amdhsa--gfx1251";
const char *kGfx125MalformedIsa = "amdgcn-amd-amdhsa--gfx125foo";
const char *kGfx12_5GenericIsa = "amdgcn-amd-amdhsa--gfx12-5-generic";
const char *kGfx12_5GenericIsaWithFeatures =
    "amdgcn-amd-amdhsa--gfx12-5-generic:sramecc+";

AgentGfxRevision make_gfx_revision(const char *gfx_target,
                                   uint32_t asic_revision,
                                   bool revision_valid = true) {
  AgentGfxRevision gfx;
  gfx.gfx_target = gfx_target;
  gfx.revision_valid = revision_valid;
  gfx.asic_revision = asic_revision;
  return gfx;
}

void run_decision_pair(const char *name,
                       const std::optional<RewriteDecision> &decision,
                       const std::string &source_isa,
                       const std::string &target_isa) {
  run(name, decision && decision->source_isa == source_isa &&
                decision->target_isa == target_isa);
}

// The rewrite-policy helpers are exercised directly below so the tests and the
// loader can never drift apart.

// gfx1250 silicon at ASIC revision A0 -> parsed target + revision, gate passes.
void test_Gfx1250A0Passes() {
  printf("TEST Gfx1250A0Passes...\n");
  reset_env();
  g_env.isa_name = kGfx1250Isa;
  g_env.asic_revision = 0;
  const AgentGfxRevision g = query_agent_gfx_revision(fresh_agent());
  run("gfx target parsed as gfx1250", g.gfx_target == "gfx1250");
  run("ASIC revision is A0 (0)", g.revision_valid && g.asic_revision == 0);
  run("gate allows gfx1250 A0", gate_allows_hotswap(g) == true);
}

// Feature-suffixed ISA names still resolve to the bare gfx target.
void test_Gfx1250FeatureSuffixParsed() {
  printf("TEST Gfx1250FeatureSuffixParsed...\n");
  reset_env();
  g_env.isa_name = kGfx1250IsaWithFeatures;
  g_env.asic_revision = 0;
  const AgentGfxRevision g = query_agent_gfx_revision(fresh_agent());
  run("feature suffix stripped -> gfx1250", g.gfx_target == "gfx1250");
  run("gate allows suffixed gfx1250 A0", gate_allows_hotswap(g) == true);
}

// A different gfx target -> gate blocks (and the ASIC revision is irrelevant).
void test_NonGfx1250Blocks() {
  printf("TEST NonGfx1250Blocks...\n");
  reset_env();
  g_env.isa_name = kGfx942Isa;
  g_env.asic_revision = 0;
  const AgentGfxRevision g = query_agent_gfx_revision(fresh_agent());
  run("gfx target parsed as gfx942", g.gfx_target == "gfx942");
  run("gate blocks gfx942", gate_allows_hotswap(g) == false);
}

// Exact-match gating: a near-miss target (gfx1251) must not be treated as
// gfx1250 even though it shares a prefix.
void test_NearMissTargetBlocks() {
  printf("TEST NearMissTargetBlocks...\n");
  reset_env();
  g_env.isa_name = kGfx1251Isa;
  g_env.asic_revision = 0;
  const AgentGfxRevision g = query_agent_gfx_revision(fresh_agent());
  run("gfx target parsed as gfx1251", g.gfx_target == "gfx1251");
  run("gate blocks gfx1251 (exact match)", gate_allows_hotswap(g) == false);
}

// Hyphenated generic processor names must be preserved while stripping feature
// suffixes; otherwise gfx12.5 opt-in routing would see only "gfx12".
void test_Gfx12_5GenericFeatureSuffixParsed() {
  printf("TEST Gfx12_5GenericFeatureSuffixParsed...\n");
  reset_env();
  g_env.isa_name = kGfx12_5GenericIsaWithFeatures;
  g_env.asic_revision = 0;
  const AgentGfxRevision g = query_agent_gfx_revision(fresh_agent());
  run("feature suffix stripped -> gfx12-5-generic",
      g.gfx_target == "gfx12-5-generic");
  run("baseline route blocks gfx12-5-generic",
      gate_allows_hotswap(g) == false);
}

// gfx1250 but a non-A0 stepping -> gate blocks.
void test_Gfx1250NonA0Blocks() {
  printf("TEST Gfx1250NonA0Blocks...\n");
  reset_env();
  g_env.isa_name = kGfx1250Isa;
  g_env.asic_revision = 1;  // A1
  const AgentGfxRevision g = query_agent_gfx_revision(fresh_agent());
  run("ASIC revision is A1 (1)", g.revision_valid && g.asic_revision == 1);
  run("gate blocks gfx1250 A1", gate_allows_hotswap(g) == false);
}

// The explicit opt-in opens a gfx1250 route independent of the baseline A0
// agent route.
void test_OptInAllowsGfx1250NonA0() {
  printf("TEST OptInAllowsGfx1250NonA0...\n");
  reset_env();
  g_env.isa_name = kGfx1250Isa;
  g_env.asic_revision = 1;  // A1/B0-side path, not A0.
  const AgentGfxRevision g = query_agent_gfx_revision(fresh_agent());
  run("baseline route blocks non-A0",
      has_candidate_hotswap_rewrite(g, RewriteOptions{false}) == false);
  run("opt-in route allows gfx1250 non-A0",
      has_candidate_hotswap_rewrite(g, RewriteOptions{true}) == true);
}

// The explicit opt-in covers the broader gfx125* family, while the baseline
// route remains exact gfx1250 A0 only.
void test_OptInAllowsGfx125Family() {
  printf("TEST OptInAllowsGfx125Family...\n");
  reset_env();
  g_env.isa_name = kGfx1251Isa;
  g_env.asic_revision = 1;
  const AgentGfxRevision g = query_agent_gfx_revision(fresh_agent());
  run("baseline route blocks gfx1251",
      has_candidate_hotswap_rewrite(g, RewriteOptions{false}) == false);
  run("opt-in route allows gfx1251",
      has_candidate_hotswap_rewrite(g, RewriteOptions{true}) == true);
}

void test_OptInRejectsMalformedGfx125Prefix() {
  printf("TEST OptInRejectsMalformedGfx125Prefix...\n");
  reset_env();
  g_env.isa_name = kGfx125MalformedIsa;
  g_env.asic_revision = 1;
  const AgentGfxRevision g = query_agent_gfx_revision(fresh_agent());
  run("malformed gfx125 prefix is parsed",
      g.gfx_target == "gfx125foo");
  run("opt-in route rejects malformed gfx125 prefix",
      has_candidate_hotswap_rewrite(g, RewriteOptions{true}) == false);
}

// The HSA tool routes gfx12-5-generic when explicitly requested.
void test_OptInAllowsGfx12_5Generic() {
  printf("TEST OptInAllowsGfx12_5Generic...\n");
  reset_env();
  g_env.isa_name = kGfx12_5GenericIsa;
  g_env.asic_revision = 1;
  const AgentGfxRevision g = query_agent_gfx_revision(fresh_agent());
  run("baseline route blocks gfx12-5-generic",
      has_candidate_hotswap_rewrite(g, RewriteOptions{false}) == false);
  run("opt-in route allows gfx12-5-generic",
      has_candidate_hotswap_rewrite(g, RewriteOptions{true}) == true);
}

// If ASIC revision cannot be queried, the explicit opt-in can still route
// gfx12.5 targets through COMGR. The baseline A0 route remains disabled.
void test_OptInAllowsGfx1250UnknownRevision() {
  printf("TEST OptInAllowsGfx1250UnknownRevision...\n");
  reset_env();
  g_env.isa_name = kGfx1250Isa;
  g_env.asic_rev_ok = false;
  const AgentGfxRevision g = query_agent_gfx_revision(fresh_agent());
  run("baseline route blocks unknown revision",
      has_candidate_hotswap_rewrite(g, RewriteOptions{false}) == false);
  run("opt-in route allows gfx1250 unknown revision",
      has_candidate_hotswap_rewrite(g, RewriteOptions{true}) == true);
}

// The opt-in is not a global rewrite enable; non-gfx12.5 targets still
// load unchanged.
void test_OptInBlocksOtherTargets() {
  printf("TEST OptInBlocksOtherTargets...\n");
  reset_env();
  g_env.isa_name = kGfx942Isa;
  g_env.asic_revision = 0;
  const AgentGfxRevision g = query_agent_gfx_revision(fresh_agent());
  run("opt-in route blocks gfx942",
      has_candidate_hotswap_rewrite(g, RewriteOptions{true}) == false);
}

// A gfx1250 A0 agent uses the baseline COMGR request without the opt-in.
void test_RewriteDecisionSelectsBaselineA0Request() {
  printf("TEST RewriteDecisionSelectsBaselineA0Request...\n");
  const AgentGfxRevision gfx1250_a0 = make_gfx_revision("gfx1250", 0);
  const auto d =
      decide_hotswap_rewrite(gfx1250_a0, kGfx1250Isa, kGfx1250Isa,
                             RewriteOptions{false});
  run_decision_pair("baseline request uses expected source and target ISA", d,
                    std::string(kGfx1250Isa) + ":gfx1250-b0-specific+",
                    std::string(kGfx1250Isa) + ":gfx1250-b0-specific-");
}

// The explicit opt-in does not change the baseline gfx1250 A0 request.
void test_RewriteDecisionA0WithOptInKeepsBaselineRequest() {
  printf("TEST RewriteDecisionA0WithOptInKeepsBaselineRequest...\n");
  const AgentGfxRevision gfx1250_a0 = make_gfx_revision("gfx1250", 0);
  const auto d =
      decide_hotswap_rewrite(gfx1250_a0, kGfx1250Isa, kGfx1250Isa,
                             RewriteOptions{true});
  run_decision_pair("opt-in keeps baseline source and target ISA", d,
                    std::string(kGfx1250Isa) + ":gfx1250-b0-specific+",
                    std::string(kGfx1250Isa) + ":gfx1250-b0-specific-");
}

// A non-A0 gfx1250 opt-in uses a same-processor COMGR request.
void test_RewriteDecisionOptInSelectsGfx1250SameProcessorRequest() {
  printf("TEST RewriteDecisionOptInSelectsGfx1250SameProcessorRequest...\n");
  const AgentGfxRevision gfx1250_b0 = make_gfx_revision("gfx1250", 1);
  const auto d =
      decide_hotswap_rewrite(gfx1250_b0, kGfx1250Isa, kGfx1250Isa,
                             RewriteOptions{true});
  const std::string b0_isa =
      std::string(kGfx1250Isa) + ":gfx1250-b0-specific+";
  run_decision_pair("gfx1250 request keeps source and target on source ISA", d,
                    b0_isa, b0_isa);
}

// Concrete gfx125* targets other than gfx1250 use the original ISA pair.
void test_RewriteDecisionOptInSelectsGfx125FamilyRequest() {
  printf("TEST RewriteDecisionOptInSelectsGfx125FamilyRequest...\n");
  const AgentGfxRevision gfx1251 = make_gfx_revision("gfx1251", 1);
  const auto d =
      decide_hotswap_rewrite(gfx1251, kGfx1251Isa, kGfx1251Isa,
                             RewriteOptions{true});
  run_decision_pair("gfx1251 ISA pair is preserved", d, kGfx1251Isa,
                    kGfx1251Isa);
}

// The opt-in path keeps COMGR on the source processor when the agent reports a
// different concrete or generic gfx12.5 ISA.
void test_RewriteDecisionOptInKeepsSourceProcessor() {
  printf("TEST RewriteDecisionOptInKeepsSourceProcessor...\n");
  const AgentGfxRevision gfx1251 = make_gfx_revision("gfx1251", 1);
  auto d =
      decide_hotswap_rewrite(gfx1251, kGfx12_5GenericIsa, kGfx1251Isa,
                             RewriteOptions{true});
  run_decision_pair("generic source keeps generic target", d,
                    kGfx12_5GenericIsa, kGfx12_5GenericIsa);

  d = decide_hotswap_rewrite(gfx1251, kGfx1250Isa, kGfx1251Isa,
                             RewriteOptions{true});
  const std::string b0_isa =
      std::string(kGfx1250Isa) + ":gfx1250-b0-specific+";
  run_decision_pair("concrete mismatch keeps source processor", d, b0_isa,
                    b0_isa);
}

// The opt-in never routes non-gfx12.5 source code objects through COMGR.
void test_RewriteDecisionRejectsNonGfx12_5Source() {
  printf("TEST RewriteDecisionRejectsNonGfx12_5Source...\n");
  const AgentGfxRevision gfx1251 = make_gfx_revision("gfx1251", 1);
  const auto d =
      decide_hotswap_rewrite(gfx1251, kGfx942Isa, kGfx1251Isa,
                             RewriteOptions{true});
  run("non-gfx12.5 source on gfx125 agent is not rewritten",
      !d.has_value());
}

// ASIC revision query failure -> revision_valid false and gate blocks, even for
// gfx1250. The query is still attempted exactly once.
void test_AsicRevisionQueryFailure() {
  printf("TEST AsicRevisionQueryFailure...\n");
  reset_env();
  g_env.isa_name = kGfx1250Isa;
  g_env.asic_rev_ok = false;
  const AgentGfxRevision g = query_agent_gfx_revision(fresh_agent());
  run("gfx target still parsed", g.gfx_target == "gfx1250");
  run("revision marked invalid on query failure", g.revision_valid == false);
  run("gate blocks when ASIC revision is unavailable",
      gate_allows_hotswap(g) == false);
  run("ASIC revision query was attempted once",
      g.asic_revision == 0 && g_env.asic_rev_calls == 1);
}

// Result is cached per agent handle: a repeat call does not re-query HSA.
void test_ResultIsCachedPerHandle() {
  printf("TEST ResultIsCachedPerHandle...\n");
  reset_env();
  g_env.isa_name = kGfx1250Isa;
  g_env.asic_revision = 0;
  const hsa_agent_t agent = fresh_agent();
  const AgentGfxRevision first = query_agent_gfx_revision(agent);
  const int isa_after_first = g_env.isa_query_calls;
  const int rev_after_first = g_env.asic_rev_calls;
  const AgentGfxRevision second = query_agent_gfx_revision(agent);
  run("cached result stays gfx1250",
      first.gfx_target == "gfx1250" && second.gfx_target == "gfx1250");
  run("second call served from cache (no re-query)",
      g_env.isa_query_calls == isa_after_first &&
          g_env.asic_rev_calls == rev_after_first);
}

// Distinct handles are evaluated independently (not conflated by the cache).
void test_DistinctHandlesIndependent() {
  printf("TEST DistinctHandlesIndependent...\n");
  reset_env();
  g_env.isa_name = kGfx1250Isa;
  g_env.asic_revision = 0;
  const AgentGfxRevision a1 = query_agent_gfx_revision(fresh_agent());

  // Different agent, different ISA: must be re-evaluated, not cached as a1.
  g_env.isa_name = kGfx942Isa;
  const AgentGfxRevision a2 = query_agent_gfx_revision(fresh_agent());
  run("first handle (gfx1250 A0) passes", gate_allows_hotswap(a1) == true);
  run("distinct handle (gfx942) evaluated independently",
      a2.gfx_target == "gfx942" && gate_allows_hotswap(a2) == false);
}

}  // namespace

int main() {
  test_Gfx1250A0Passes();
  test_Gfx1250FeatureSuffixParsed();
  test_NonGfx1250Blocks();
  test_NearMissTargetBlocks();
  test_Gfx12_5GenericFeatureSuffixParsed();
  test_Gfx1250NonA0Blocks();
  test_OptInAllowsGfx1250NonA0();
  test_OptInAllowsGfx125Family();
  test_OptInRejectsMalformedGfx125Prefix();
  test_OptInAllowsGfx12_5Generic();
  test_OptInAllowsGfx1250UnknownRevision();
  test_OptInBlocksOtherTargets();
  test_RewriteDecisionSelectsBaselineA0Request();
  test_RewriteDecisionA0WithOptInKeepsBaselineRequest();
  test_RewriteDecisionOptInSelectsGfx1250SameProcessorRequest();
  test_RewriteDecisionOptInSelectsGfx125FamilyRequest();
  test_RewriteDecisionOptInKeepsSourceProcessor();
  test_RewriteDecisionRejectsNonGfx12_5Source();
  test_AsicRevisionQueryFailure();
  test_ResultIsCachedPerHandle();
  test_DistinctHandlesIndependent();

  printf("\n%d passed, %d failed\n", tests_total - tests_failed, tests_failed);
  return tests_failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
