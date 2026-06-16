//===- hotswap_tool_test.cpp - Tests for gfx-target / ASIC-rev query ------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for query_agent_gfx_revision() in hotswap_gfx_query.cpp, plus the
// gfx1250-A0 gate that hotswap_load_agent_code_object() applies on top of it.
//
// The query helpers — query_agent_gfx_revision(), extract_gfx_target() and the
// AgentGfxRevision type — live in their own translation unit (compiled in
// alongside this test), so the test includes only the small
// hotswap_gfx_query.hpp header. The HSA entry
// points the query calls are replaced with in-file stubs (linked in place of
// the real libraries) so the query can be driven entirely from the test without
// GPU hardware:
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

// The unit under test (brings in hsa.h and the query helper declarations).
#include "hotswap_gfx_query.hpp"

using rocr::hotswap::AgentGfxRevision;
using rocr::hotswap::gate_allows_hotswap;
using rocr::hotswap::query_agent_gfx_revision;
using rocr::hotswap::reset_gfx_revision_cache;

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

// The gate applied in hotswap_load_agent_code_object() is the shared
// rocr::hotswap::gate_allows_hotswap(), exercised directly below so the tests
// and the tool can never drift apart.

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
  test_Gfx1250NonA0Blocks();
  test_AsicRevisionQueryFailure();
  test_ResultIsCachedPerHandle();
  test_DistinctHandlesIndependent();

  printf("\n%d passed, %d failed\n", tests_total - tests_failed, tests_failed);
  return tests_failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
