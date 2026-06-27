//===- hotswap_loader_test.cpp - Tests for HSA tools loader path ----------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The test support header includes hotswap_tool.cpp directly so these tests can
// drive the wrapped HSA API-table entry points without a GPU, real HSA runtime,
// or real COMGR.
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <cstdlib>

#include "hotswap_loader_test_utils.hpp"

namespace {

void test_OptInDisabledLoadsOriginal() {
  begin_test("OptInDisabledLoadsOriginal",
             "Unset, empty, and 0 opt-in values must leave non-A0 gfx1250 "
             "on the original reader.");
  struct FlagCase {
    const char *flag_value;
    const char *expectation;
  };
  const FlagCase cases[] = {
      {nullptr, "unset opt-in skips non-A0 gfx1250 rewrite"},
      {"", "empty opt-in skips non-A0 gfx1250 rewrite"},
      {"0", "opt-in value 0 skips non-A0 gfx1250 rewrite"},
  };
  for (const FlagCase &c : cases) {
    const LoadResult result =
        load_once(kGfx1250Isa, kGfx1250Isa, c.flag_value);
    check_original_load(result, c.expectation);
  }
}

void test_OptInRoutesGfx1250SameProcessor() {
  begin_test("OptInRoutesGfx1250SameProcessor",
             "The opt-in must route non-A0 gfx1250 through COMGR with a "
             "same-processor request.");
  const LoadResult result = load_once(kGfx1250Isa, kGfx1250Isa, "1");
  check(result.status == HSA_STATUS_SUCCESS, "load succeeds");
  check(result.retarget_calls == 1,
        "opt-in routes non-A0 gfx1250 through COMGR");
  check(result.source_isa == kGfx1250B0Isa,
        "source ISA is tagged as B0");
  check(result.target_isa == kGfx1250B0Isa,
        "non-A0 target ISA is tagged as B0");
  check(result.loaded_reader != result.original_reader,
        "rewritten reader is loaded instead of original reader");
  check(result.retained_elfs == 1,
        "rewritten ELF is retained after successful load");
}

void test_OptInRoutesGfx12_5Family() {
  begin_test("OptInRoutesGfx12_5Family",
             "The opt-in must route gfx125* and gfx12-5-generic "
             "without adding gfx1250 stepping features.");
  const char *cases[] = {kGfx1251Isa, kGfx12_5GenericIsa};
  for (const char *isa : cases) {
    const LoadResult result = load_once(isa, isa, "1");
    check(result.status == HSA_STATUS_SUCCESS, "load succeeds");
    check(result.retarget_calls == 1,
          "opt-in routes gfx12.5 target through COMGR");
    check(result.source_isa == isa, "source ISA is preserved");
    check(result.target_isa == isa, "target ISA is preserved");
  }
}

void test_GenericSourceUsesGenericTarget() {
  begin_test("GenericSourceUsesGenericTarget",
             "A gfx12-5-generic source loaded on a concrete gfx125* agent "
             "must stay generic to avoid processor retargeting.");
  const LoadResult result = load_once(kGfx12_5GenericIsa, kGfx1251Isa, "1");
  check(result.status == HSA_STATUS_SUCCESS, "load succeeds");
  check(result.retarget_calls == 1,
        "generic source on concrete gfx125 agent routes through COMGR");
  check(result.source_isa == kGfx12_5GenericIsa,
        "generic source ISA is preserved");
  check(result.target_isa == kGfx12_5GenericIsa,
        "generic target ISA is preserved to avoid processor retargeting");
}

void test_ConcreteSourceUsesSourceTarget() {
  begin_test("ConcreteSourceUsesSourceTarget",
             "A concrete gfx125* source loaded on a different gfx125* agent "
             "must stay on the source processor to avoid retargeting.");
  const LoadResult result = load_once(kGfx1250Isa, kGfx1251Isa, "1");
  check(result.status == HSA_STATUS_SUCCESS, "load succeeds");
  check(result.retarget_calls == 1,
        "concrete source on different gfx125 agent routes through COMGR");
  check(result.source_isa == kGfx1250B0Isa,
        "source ISA is tagged as B0");
  check(result.target_isa == kGfx1250B0Isa,
        "target ISA stays on the source processor");
}

void test_A0UsesBaselineRouteWithoutOptIn() {
  begin_test("A0UsesBaselineRouteWithoutOptIn",
             "The existing gfx1250 A0 route must still call COMGR without "
             "the opt-in.");
  const LoadResult result = load_once(kGfx1250Isa, kGfx1250Isa, nullptr, 0);
  check(result.status == HSA_STATUS_SUCCESS, "load succeeds");
  check(result.retarget_calls == 1, "A0 gfx1250 keeps baseline route");
  check(result.source_isa == kGfx1250B0Isa,
        "source code object ISA is tagged as B0");
  check(result.target_isa == kGfx1250A0Isa,
        "A0 agent ISA is tagged as A0");
}

void test_A0WithOptInKeepsBaselinePair() {
  begin_test("A0WithOptInKeepsBaselinePair",
             "The opt-in on gfx1250 A0 must preserve the baseline ISA pair "
             "while routing through COMGR.");
  const LoadResult result = load_once(kGfx1250Isa, kGfx1250Isa, "1", 0);
  check(result.status == HSA_STATUS_SUCCESS, "load succeeds");
  check(result.retarget_calls == 1, "A0 gfx1250 routes through COMGR");
  check(result.source_isa == kGfx1250B0Isa,
        "source code object ISA is tagged as B0");
  check(result.target_isa == kGfx1250A0Isa,
        "A0 agent ISA remains tagged as A0");
}

void test_OptInBlocksNonGfx12_5() {
  begin_test("OptInBlocksNonGfx12_5",
             "The opt-in must not become a global rewrite enable for "
             "unsupported agents or source code objects.");
  LoadResult result = load_once(kGfx942Isa, kGfx942Isa, "1", 0);
  check_original_load(result, "non-gfx12.5 agent does not route");

  result = load_once(kGfx942Isa, kGfx1251Isa, "1");
  check_original_load(result, "non-gfx12.5 source does not route");
}

void test_RetargetFailureFallsBackToOriginalReader() {
  begin_test("RetargetFailureFallsBackToOriginalReader",
             "If COMGR rejects a gated rewrite, the loader must still load "
             "the original reader.");
  const LoadResult result = load_once(kGfx1250Isa, kGfx1250Isa, "1", 1, -1);
  check(result.status == HSA_STATUS_SUCCESS, "fallback load succeeds");
  check(result.retarget_calls == 1, "COMGR retarget was attempted");
  check(result.loaded_reader == result.original_reader,
        "retarget failure falls back to original reader");
}

} // namespace

int main() {
  test_OptInDisabledLoadsOriginal();
  test_OptInRoutesGfx1250SameProcessor();
  test_OptInRoutesGfx12_5Family();
  test_GenericSourceUsesGenericTarget();
  test_ConcreteSourceUsesSourceTarget();
  test_A0UsesBaselineRouteWithoutOptIn();
  test_A0WithOptInKeepsBaselinePair();
  test_OptInBlocksNonGfx12_5();
  test_RetargetFailureFallsBackToOriginalReader();
  reset_state();

  std::printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
  return tests_failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
