// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file isa_diff_test.cpp
/// @brief Validator tests for rj_isa_diff (positive, negative and rejection cases).
///
/// Fixtures are LLVM-assembled gfx950 words (fixtures/fixtures.asm -> fixtures.inc);
/// the tests themselves never run LLVM.

#include "case_io.h"
#include "effects.h"
#include "input_policy.h"
#include "kernel_model.h"
#include "mutation.h"
#include "region_runner.h"

#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <vector>

// clang-format off
#include "fixtures/fixtures.inc"
// clang-format on

namespace {

using namespace rocjitsu;
using namespace rocjitsu::isa_diff;

LocSet locs(std::string_view text) {
  auto parsed = parse_locs(text);
  EXPECT_TRUE(parsed.has_value()) << text;
  return parsed.value_or(LocSet{});
}

CaseData analyze(const std::vector<uint32_t> &old_words, const std::vector<uint32_t> &new_words,
                 const PlanOptions &options = {}) {
  std::string error;
  auto data = analyze_artifacts(artifact_from_words(old_words, "old"),
                                artifact_from_words(new_words, "new"), options, error);
  EXPECT_TRUE(data.has_value()) << error;
  return data.value_or(CaseData{});
}

struct TestRun {
  Json report;
  std::vector<RegionResult> results;
  [[nodiscard]] std::string overall() const { return report["overall"]["test_status"].as_string(); }
  [[nodiscard]] std::string coverage() const {
    return report["overall"]["coverage_scope"].as_string();
  }
};

TestRun run(const CaseData &data, uint64_t cases = 400, uint64_t seed = 1,
            const std::string &cex_dir = "") {
  TestOptions opt;
  opt.cases = cases;
  opt.seed = seed;
  opt.counterexample_dir = cex_dir;
  TestRun r;
  r.report = test_case(data, opt, "", &r.results);
  return r;
}

std::string temp_dir(const std::string &name) {
  auto dir = std::filesystem::temp_directory_path() /
             ("rj_isa_diff_test_" + name + "_" +
              std::to_string(::testing::UnitTest::GetInstance()->random_seed()));
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  return dir.string();
}

std::vector<std::unique_ptr<Instruction>> decode_all(const std::vector<uint32_t> &words) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  std::vector<std::unique_ptr<Instruction>> out;
  std::vector<uint32_t> padded = words;
  padded.resize(padded.size() + 4, 0);
  size_t off = 0;
  while (off < words.size()) {
    auto d = decoder->decode_window(std::span<const uint32_t>(padded).subspan(off), off * 4);
    if (d.failed()) {
      ADD_FAILURE() << "decode failed at word " << off;
      break;
    }
    off += static_cast<size_t>(d.value()->size()) / 4;
    out.push_back(std::move(d).value());
  }
  return out;
}

// ---------------------------------------------------------------------------
// Effect model

TEST(IsaDiffEffects, ClassifiesSaluOperandsPrecisely) {
  auto insts = decode_all(k_effects);
  ASSERT_EQ(insts.size(), 21u);
  auto fx = [&](size_t i) { return classify(*insts[i]); };

  Effects add = fx(0); // s_add_u32 s4, s2, s3
  EXPECT_TRUE(add.p0);
  EXPECT_EQ(add.reads, locs("s2,s3"));
  EXPECT_EQ(add.writes, locs("s4,scc"));

  Effects addc = fx(1); // s_addc_u32 s5, s0, s1
  EXPECT_EQ(addc.reads, locs("s0,s1,scc"));
  EXPECT_EQ(addc.writes, locs("s5,scc"));

  Effects csel = fx(2); // s_cselect_b64 s[14:15], s[2:3], vcc
  EXPECT_EQ(csel.reads, locs("s2,s3,vcc_lo,vcc_hi,scc"));
  EXPECT_EQ(csel.writes, locs("s14,s15"));
  EXPECT_EQ(csel.wide_reads, locs("s2,vcc_lo"));

  Effects vcchi = fx(3); // s_mov_b32 vcc_hi, s1: only the high half
  EXPECT_TRUE(vcchi.p0);
  EXPECT_EQ(vcchi.writes, locs("vcc_hi"));

  Effects exec_and = fx(4); // s_and_b64 exec, exec, s[2:3]
  EXPECT_EQ(exec_and.reads, locs("s2,s3,exec_lo,exec_hi"));
  EXPECT_EQ(exec_and.writes, locs("exec_lo,exec_hi,scc"));

  Effects saveexec = fx(5); // s_and_saveexec_b64 s[2:3], s[4:5]
  EXPECT_TRUE(saveexec.p0);
  EXPECT_EQ(saveexec.reads, locs("s4,s5,exec_lo,exec_hi"));
  EXPECT_EQ(saveexec.writes, locs("s2,s3,exec_lo,exec_hi,scc"));

  Effects cmov = fx(6); // s_cmov_b32 s6, s7: conditional def also reads s6
  EXPECT_EQ(cmov.reads, locs("s6,s7,scc"));
  EXPECT_EQ(cmov.writes, locs("s6"));

  EXPECT_EQ(fx(7).writes, locs("m0"));        // s_mov_b32 m0, s2
  EXPECT_EQ(fx(8).reads, locs("scc"));        // s_mov_b32 s9, src_scc
  EXPECT_EQ(fx(9).writes, locs("s6,s7,scc")); // s_lshl_b64
  EXPECT_EQ(fx(9).wide_reads, locs("s4"));

  for (size_t i : {10u, 11u, 12u}) { // s_cbranch_scc0 / execz / vccnz
    EXPECT_FALSE(fx(i).p0);
    EXPECT_TRUE(fx(i).control_flow);
    EXPECT_FALSE(fx(i).implicit_special_reads) << i;
  }
  EXPECT_EQ(fx(10).reads, locs("scc"));
  EXPECT_EQ(fx(11).reads, locs("exec_lo,exec_hi"));
  EXPECT_EQ(fx(12).reads, locs("vcc_lo,vcc_hi"));

  Effects getpc = fx(13);
  EXPECT_FALSE(getpc.p0);
  EXPECT_TRUE(getpc.reads_pc);
  EXPECT_FALSE(fx(14).p0); // s_waitcnt
  EXPECT_FALSE(fx(14).implicit_special_reads);
  Effects load = fx(15);
  EXPECT_FALSE(load.p0);
  EXPECT_TRUE(load.implicit_special_reads);
  EXPECT_EQ(load.reads, locs("s0,s1"));
  EXPECT_FALSE(fx(16).p0); // v_mov_b32
  EXPECT_TRUE(fx(16).implicit_special_reads);
  EXPECT_NE(fx(17).reject_reason.find("trap temporary"), std::string::npos) << fx(17).reject_reason;
  EXPECT_NE(fx(18).reject_reason.find("allowlist"), std::string::npos) << fx(18).reject_reason;
  EXPECT_TRUE(fx(19).unknown_effect); // s_movrels_b32: indirect SGPR
  EXPECT_TRUE(fx(20).control_flow);   // s_endpgm
  EXPECT_FALSE(fx(20).implicit_special_reads);
}

// ---------------------------------------------------------------------------
// Liveness, taint, CFG

TEST(IsaDiffKernelModel, SccLivenessDependsOnTheConsumer) {
  std::string error;
  auto live = KernelModel::build(artifact_from_words(k_scclive_old, "a"), error);
  auto dead = KernelModel::build(artifact_from_words(k_sccdead_old, "b"), error);
  ASSERT_TRUE(live && dead) << error;
  EXPECT_TRUE(live->live_after(0).test(kLocScc));
  EXPECT_FALSE(dead->live_after(0).test(kLocScc));
  // Nothing is live after s_endpgm.
  EXPECT_TRUE(live->live_after(live->insts().size() - 1).none());
}

TEST(IsaDiffKernelModel, UnknownSuccessorMakesEverythingLive) {
  std::string error;
  auto model = KernelModel::build(artifact_from_words(k_setpc_old, "a"), error);
  ASSERT_TRUE(model) << error;
  EXPECT_FALSE(model->cfg_complete());
  EXPECT_EQ(model->live_after(1), all_locs());
}

TEST(IsaDiffKernelModel, PcTaintFollowsTheAddressChain) {
  std::string error;
  auto model = KernelModel::build(artifact_from_words(k_getpc_old, "a"), error);
  ASSERT_TRUE(model) << error;
  EXPECT_EQ(model->pc_tainted_before(1) & locs("s0,s1"), locs("s0,s1"));
  EXPECT_TRUE(model->pc_tainted_before(2).test(1)); // s1 still tainted after s_add_u32 s0
}

// ---------------------------------------------------------------------------
// Alignment and classification

TEST(IsaDiffAlign, IdenticalKernelsHaveNoSemanticDiff) {
  CaseData d = analyze(k_shift_old, k_shift_old);
  EXPECT_EQ(d.changes.size(), 0u);
  EXPECT_TRUE(d.regions.empty());
  TestRun r = run(d);
  EXPECT_EQ(r.overall(), "NO_SEMANTIC_DIFF");
}

TEST(IsaDiffAlign, ChangedBranchTargetIsControlFlow) {
  std::vector<uint32_t> retarget = k_rename_old;
  bool found = false;
  for (auto &w : retarget)
    if ((w & 0xffff0000u) == 0xbf850000u) { // s_cbranch_scc1
      w = (w & 0xffff0000u) | ((w + 1) & 0xffffu);
      found = true;
    }
  ASSERT_TRUE(found);
  CaseData d = analyze(k_rename_old, retarget);
  ASSERT_FALSE(d.unresolved.empty());
  bool control = false;
  for (const auto &u : d.unresolved)
    control |= u.kind == "control_flow";
  EXPECT_TRUE(control);
  TestRun r = run(d);
  EXPECT_NE(r.overall(), "NO_MISMATCH_FOUND");
}

// ---------------------------------------------------------------------------
// Handoff section 13 table

TEST(IsaDiffValidator, EquivalentRewriteHasNoMismatch) {
  CaseData d = analyze(k_shift_old, k_shift_new);
  ASSERT_EQ(d.regions.size(), 1u);
  EXPECT_EQ(d.regions[0].status, RegionStatus::Ready);
  EXPECT_EQ(d.regions[0].contract.status, ContractStatus::Established);
  TestRun r = run(d, 2000);
  ASSERT_EQ(r.results.size(), 1u);
  EXPECT_EQ(r.results[0].test_status, TestStatus::NoMismatchFound);
  EXPECT_EQ(r.results[0].valid_pairs, 2000u);
  // Only the region executed: 1 old and 2 new instructions per case.
  EXPECT_EQ(r.results[0].old_insts_executed, 2000u);
  EXPECT_EQ(r.results[0].new_insts_executed, 4000u);
  EXPECT_EQ(r.overall(), "NO_MISMATCH_FOUND");
  EXPECT_EQ(r.coverage(), "COMPLETE_FOR_DECLARED_SCOPE");
}

TEST(IsaDiffValidator, IdenticalRegionIsStableAndLeakFree) {
  CaseData d = analyze(k_shift_old, k_shift_new);
  ASSERT_EQ(d.regions.size(), 1u);
  RegionPair same = d.regions[0];
  same.new_side = same.old_side;
  same.contract.declared_writes_new = same.contract.declared_writes_old;
  same.prepared_key = compute_prepared_key(same);
  BaselinePolicy p1(5), p2(5);
  TestOptions opt;
  opt.cases = 1500;
  opt.seed = 5;
  RegionResult a = test_region(same, opt, p1);
  RegionResult b = test_region(same, opt, p2);
  EXPECT_EQ(a.test_status, TestStatus::NoMismatchFound);
  EXPECT_EQ(a.valid_pairs, 1500u);
  EXPECT_EQ(a.valid_pairs, b.valid_pairs);
  EXPECT_EQ(a.mismatches, 0u);
}

TEST(IsaDiffValidator, DeadTemporaryDifferenceIsNotReported) {
  CaseData d = analyze(k_deadtemp_old, k_deadtemp_new);
  ASSERT_EQ(d.regions.size(), 1u);
  EXPECT_FALSE(d.regions[0].contract.observable.test(10));
  EXPECT_TRUE(d.regions[0].contract.canary.test(10));
  TestRun r = run(d);
  EXPECT_EQ(r.results[0].test_status, TestStatus::NoMismatchFound);
}

TEST(IsaDiffValidator, LiveSccDifferenceIsDetectedAndDeadSccIsNot) {
  TestRun live = run(analyze(k_scclive_old, k_scclive_new));
  ASSERT_EQ(live.results.size(), 1u);
  EXPECT_EQ(live.results[0].test_status, TestStatus::LocalMismatch);
  ASSERT_FALSE(live.results[0].examples.empty());
  EXPECT_TRUE(live.results[0].examples[0].locs.test(kLocScc));
  EXPECT_EQ(live.results[0].witness_scope, WitnessScope::LocalOnly);
  EXPECT_EQ(live.overall(), "LOCAL_MISMATCH");

  TestRun dead = run(analyze(k_sccdead_old, k_sccdead_new), 2000);
  EXPECT_EQ(dead.results[0].test_status, TestStatus::NoMismatchFound);
}

TEST(IsaDiffValidator, ClobberedLiveThroughRegisterIsDetected) {
  CaseData d = analyze(k_clobber_old, k_clobber_new);
  ASSERT_EQ(d.regions.size(), 1u);
  EXPECT_TRUE(d.regions[0].old_side.insts.empty()); // pure insertion
  EXPECT_TRUE(d.regions[0].contract.observable.test(20));
  TestRun r = run(d);
  EXPECT_EQ(r.results[0].test_status, TestStatus::LocalMismatch);
  EXPECT_TRUE(r.results[0].examples[0].locs.test(20));
}

TEST(IsaDiffValidator, HalfRegistersPairsAndOverlapCompareCorrectly) {
  TestRun live = run(analyze(k_vcclive_old, k_vcclive_new));
  EXPECT_EQ(live.results[0].test_status, TestStatus::LocalMismatch);
  EXPECT_EQ(live.results[0].examples[0].locs, locs("vcc_hi"));

  CaseData dead_case = analyze(k_vccdead_old, k_vccdead_new);
  EXPECT_FALSE(dead_case.regions[0].contract.observable.test(kLocVccHi));
  EXPECT_EQ(run(dead_case).results[0].test_status, TestStatus::NoMismatchFound);

  CaseData overlap = analyze(k_overlap_old, k_overlap_new);
  ASSERT_EQ(overlap.regions.size(), 1u); // the two hunks merge within the block
  TestRun o = run(overlap, 2000);
  EXPECT_EQ(o.results[0].test_status, TestStatus::NoMismatchFound);
}

TEST(IsaDiffValidator, SameSeedIsDeterministicAcrossRunsAndRunners) {
  CaseData d = analyze(k_scclive_old, k_scclive_new);
  TestRun a = run(d, 300, 11);
  TestRun b = run(d, 300, 11);
  ASSERT_EQ(a.results.size(), b.results.size());
  EXPECT_EQ(a.results[0].mismatches, b.results[0].mismatches);
  EXPECT_EQ(a.results[0].examples[0].case_index, b.results[0].examples[0].case_index);
  EXPECT_EQ(a.results[0].examples[0].final_new, b.results[0].examples[0].final_new);

  // Interleaved reuse of two runners matches fresh runners exactly.
  std::string error;
  auto r1 = RegionRunner::create(d.regions[0].old_side.insts, "il_a", error);
  auto r2 = RegionRunner::create(d.regions[0].new_side.insts, "il_b", error);
  ASSERT_TRUE(r1 && r2) << error;
  std::mt19937_64 rng(3);
  for (int i = 0; i < 200; ++i) {
    MachineState s;
    s.frame = r1->baseline_frame();
    for (unsigned l = 0; l < kNumLocs; ++l)
      s.set(l, static_cast<uint32_t>(rng()));
    r1->load(s);
    ASSERT_EQ(r1->run(8).status, RunStatus::Ok);
    const MachineState first = r1->capture();
    r2->load(s);
    ASSERT_EQ(r2->run(8).status, RunStatus::Ok);
    if (i % 10 == 0) {
      auto fresh = RegionRunner::create(d.regions[0].old_side.insts, "il_fresh", error);
      ASSERT_TRUE(fresh) << error;
      fresh->load(s);
      ASSERT_EQ(fresh->run(8).status, RunStatus::Ok);
      EXPECT_EQ(first, fresh->capture());
    }
    r1->load(s);
    ASSERT_EQ(r1->run(8).status, RunStatus::Ok);
    EXPECT_EQ(first, r1->capture());
  }
}

TEST(IsaDiffValidator, ConstraintDependentRewriteSeparatesUnrestrictedFromConstrained) {
  // Unrestricted entry: s_lshr vs s_bfe differ once s4 >= 256.
  CaseData bare = analyze(k_constraint_old, k_constraint_new);
  ASSERT_EQ(bare.regions.size(), 1u);
  EXPECT_EQ(run(bare).results[0].test_status, TestStatus::LocalMismatch);

  // Executing the common prefix that establishes the range: ESTABLISHED, no mismatch.
  PlanOptions ctx;
  ctx.context_insts = 1;
  CaseData with_ctx = analyze(k_constraint_old, k_constraint_new, ctx);
  ASSERT_EQ(with_ctx.regions.size(), 1u);
  EXPECT_EQ(with_ctx.regions[0].old_side.insts.size(), 2u);
  EXPECT_EQ(with_ctx.regions[0].contract.status, ContractStatus::Established);
  EXPECT_EQ(run(with_ctx, 2000).results[0].test_status, TestStatus::NoMismatchFound);

  // A hand-written constraint is an assumption: ASSUMED after re-reading the case.
  Json j = bare.to_json();
  std::string error;
  auto edited_region = RegionPair::from_json(j["regions"].items()[0], error);
  ASSERT_TRUE(edited_region) << error;
  Constraint c;
  c.loc = 4;
  c.umax = 0xffu;
  edited_region->contract.constraints.push_back(c);
  Json regions = Json::array();
  regions.push(edited_region->to_json());
  j.set("regions", std::move(regions));
  auto reread = CaseData::from_json(j, error);
  ASSERT_TRUE(reread) << error;
  EXPECT_EQ(reread->regions[0].contract.status, ContractStatus::Assumed);
  TestRun constrained = run(*reread, 1000);
  EXPECT_EQ(constrained.results[0].test_status, TestStatus::NoMismatchFound);
  EXPECT_EQ(constrained.results[0].contract_status, ContractStatus::Assumed);
  EXPECT_GT(constrained.results[0].valid_pairs, 0u);
}

TEST(IsaDiffValidator, ContradictoryConstraintsNeverPass) {
  CaseData d = analyze(k_constraint_old, k_constraint_new);
  Constraint c;
  c.loc = 4;
  c.known_zero = 1;
  c.known_one = 1;
  d.regions[0].contract.constraints.push_back(c);
  TestRun r = run(d, 200);
  EXPECT_EQ(r.results[0].test_status, TestStatus::Inconclusive);
  EXPECT_EQ(r.results[0].valid_pairs, 0u);
  EXPECT_EQ(r.results[0].accepted, 0u);
  EXPECT_NE(r.overall(), "NO_MISMATCH_FOUND");
}

TEST(IsaDiffValidator, DependentHunksMergeInsteadOfPassingIndependently) {
  CaseData merged = analyze(k_twohunk_old, k_twohunk_new);
  ASSERT_EQ(merged.regions.size(), 1u);
  EXPECT_EQ(merged.regions[0].hunks.size(), 2u);
  EXPECT_EQ(merged.regions[0].old_side.insts.size(), 3u); // includes the common s_mov
  EXPECT_EQ(run(merged, 2000).results[0].test_status, TestStatus::NoMismatchFound);

  PlanOptions no_merge;
  no_merge.merge_same_block = false;
  CaseData split = analyze(k_twohunk_old, k_twohunk_new, no_merge);
  ASSERT_EQ(split.regions.size(), 2u);
  EXPECT_EQ(split.regions[1].contract.status, ContractStatus::Assumed);
  ASSERT_FALSE(split.regions[1].contract.depends_on.empty());
  EXPECT_EQ(split.regions[1].contract.depends_on[0], "r0");
  TestRun r = run(split);
  // The intermediate s4 differs: a local mismatch, never an independent false pass.
  EXPECT_EQ(r.results[0].test_status, TestStatus::LocalMismatch);
}

TEST(IsaDiffValidator, RenamingWithoutValueMappingIsInconclusive) {
  CaseData d = analyze(k_rename_old, k_rename_new);
  ASSERT_GE(d.regions.size(), 2u);
  EXPECT_EQ(d.regions[0].status, RegionStatus::Inconclusive);
  EXPECT_NE(d.regions[0].reason.find("INCONCLUSIVE_MAPPING"), std::string::npos);
  TestRun r = run(d);
  EXPECT_EQ(r.overall(), "INCONCLUSIVE");
}

TEST(IsaDiffValidator, UnknownSuccessorDescriptorAndUncoveredChangesKeepScopePartial) {
  CaseData setpc = analyze(k_setpc_old, k_setpc_new);
  ASSERT_EQ(setpc.regions.size(), 1u);
  EXPECT_EQ(setpc.regions[0].contract.status, ContractStatus::Assumed);
  TestRun r = run(setpc);
  EXPECT_EQ(r.results[0].test_status, TestStatus::NoMismatchFound);
  EXPECT_EQ(r.overall(), "INCONCLUSIVE");
  EXPECT_EQ(r.coverage(), "PARTIAL");

  // Descriptor resource change: contract UNRESOLVED, scope partial.
  Artifact a = artifact_from_words(k_shift_old, "old");
  Artifact b = artifact_from_words(k_shift_new, "new");
  a.descriptor.assign(64, 0);
  b.descriptor.assign(64, 0);
  b.descriptor[48] = 0x41; // compute_pgm_rsrc1
  std::string error;
  auto desc = analyze_artifacts(a, b, {}, error);
  ASSERT_TRUE(desc) << error;
  EXPECT_EQ(desc->regions[0].contract.status, ContractStatus::Unresolved);
  TestRun rd = run(*desc);
  EXPECT_EQ(rd.overall(), "INCONCLUSIVE");
  EXPECT_EQ(rd.coverage(), "PARTIAL");

  // Entry-offset-only descriptor change is layout, not semantics.
  b.descriptor.assign(64, 0);
  b.descriptor[16] = 0x40;
  auto layout = analyze_artifacts(a, b, {}, error);
  ASSERT_TRUE(layout) << error;
  EXPECT_TRUE(layout->unresolved.empty());
  EXPECT_EQ(run(*layout).overall(), "NO_MISMATCH_FOUND");

  // Unreachable bytes changed: nothing to test, but never a pass.
  TestRun un = run(analyze(k_unreachable_old, k_unreachable_new));
  EXPECT_EQ(un.overall(), "INCONCLUSIVE");
  EXPECT_EQ(un.coverage(), "NOT_ESTABLISHED");
}

TEST(IsaDiffValidator, ExecZeroIsExecutedNotFiltered) {
  CaseData d = analyze(k_exec_old, k_exec_new);
  ASSERT_EQ(d.regions.size(), 1u);
  BaselinePolicy policy(9);
  bool saw_zero_exec = false;
  for (uint64_t k = 0; k < 4000 && !saw_zero_exec; ++k) {
    Sample s = policy.propose({}, d.regions[0].contract, k);
    saw_zero_exec = s.value[kLocExecLo] == 0 && s.value[kLocExecHi] == 0;
  }
  EXPECT_TRUE(saw_zero_exec);
  TestRun r = run(d, 3000, 9);
  EXPECT_EQ(r.results[0].test_status, TestStatus::NoMismatchFound);
  EXPECT_EQ(r.results[0].valid_pairs, r.results[0].accepted);
}

TEST(IsaDiffValidator, WaitcntMemoryPcAndHwregRegionsAreUnsupported) {
  struct Case {
    const std::vector<uint32_t> &o, &n;
    const char *needle;
  };
  const Case cases[] = {
      {k_waitcnt_old, k_waitcnt_new, "s_waitcnt"},
      {k_smem_old, k_smem_new, "s_load_dword"},
      {k_getpc_old, k_getpc_new, "PC-derived"},
      {k_getreg_old, k_getreg_new, "allowlist"},
  };
  for (const auto &c : cases) {
    SCOPED_TRACE(c.needle);
    CaseData d = analyze(c.o, c.n);
    ASSERT_EQ(d.regions.size(), 1u);
    EXPECT_EQ(d.regions[0].status, RegionStatus::Unsupported);
    EXPECT_NE(d.regions[0].reason.find(c.needle), std::string::npos) << d.regions[0].reason;
    TestRun r = run(d);
    EXPECT_EQ(r.results[0].test_status, TestStatus::Unsupported);
    EXPECT_EQ(r.overall(), "UNSUPPORTED");
  }
}

TEST(IsaDiffValidator, DecodeFailureAndModelOnlyAreClassifiedNotNops) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  std::optional<uint32_t> invalid;
  for (uint32_t op = 0x7f; op >= 0x20 && !invalid; --op) {
    const std::vector<uint32_t> w = {0xbf800000u | (op << 16), 0, 0, 0};
    if (decoder->decode_window(w).failed())
      invalid = w[0];
  }
  ASSERT_TRUE(invalid.has_value()) << "no undecodable SOPP opcode found";
  std::string error;
  RegionInst bad_inst;
  bad_inst.pc = 0x100;
  bad_inst.words.push_back(*invalid);
  bad_inst.text = "<bad>";
  auto bad = RegionRunner::create(std::vector<RegionInst>(1, bad_inst), "bad", error);
  ASSERT_TRUE(bad) << error;
  bad->load(MachineState{});
  RunOutcome out = bad->run(4);
  EXPECT_EQ(out.status, RunStatus::DecodeFailure);
  EXPECT_EQ(out.executed, 0u);

  CaseData d = analyze(k_shift_old, k_shift_new);
  auto model_only = RegionRunner::create(d.regions[0].new_side.insts, "mo", error);
  ASSERT_TRUE(model_only) << error;
  model_only->simulate_model_only_for_test(1);
  MachineState s;
  s.frame = model_only->baseline_frame();
  model_only->load(s);
  out = model_only->run(4);
  EXPECT_EQ(out.status, RunStatus::MissingExecutionBackend);
  EXPECT_EQ(out.failed_index, 1u);
  EXPECT_EQ(out.executed, 1u);

  // A region whose words do not decode is UNSUPPORTED, never a pass.
  RegionPair broken = d.regions[0];
  // (vector(1, x) rather than {x}: gcc-13 -Warray-bounds misfires on the one-element list.)
  broken.new_side.insts[0].words = std::vector<uint32_t>(1, *invalid);
  broken.prepared_key = compute_prepared_key(broken);
  BaselinePolicy policy(1);
  TestOptions opt;
  opt.cases = 20;
  RegionResult r = test_region(broken, opt, policy);
  EXPECT_EQ(r.test_status, TestStatus::Unsupported);
  EXPECT_EQ(r.valid_pairs, 0u);
}

TEST(IsaDiffValidator, LivenessChangeInvalidatesThePreparedKey) {
  CaseData dead = analyze(k_cachedead_old, k_cachedead_new);
  CaseData live = analyze(k_cachelive_old, k_cachelive_new);
  ASSERT_EQ(dead.regions.size(), 1u);
  ASSERT_EQ(live.regions.size(), 1u);
  // Same instruction words at the same PCs ...
  EXPECT_EQ(dead.regions[0].old_side.insts[0].words, live.regions[0].old_side.insts[0].words);
  EXPECT_EQ(dead.regions[0].new_side.insts[0].words, live.regions[0].new_side.insts[0].words);
  EXPECT_EQ(dead.regions[0].old_side.insts[0].pc, live.regions[0].old_side.insts[0].pc);
  // ... but a different consumer: different contract, different key, different verdict.
  EXPECT_NE(dead.regions[0].prepared_key, live.regions[0].prepared_key);
  EXPECT_EQ(run(dead).results[0].test_status, TestStatus::NoMismatchFound);
  EXPECT_EQ(run(live).results[0].test_status, TestStatus::LocalMismatch);
}

TEST(IsaDiffValidator, InjectedErrorsAreDetectedReplayableAndSynthetic) {
  CaseData d = analyze(k_shift_old, k_shift_new);
  ASSERT_EQ(d.regions.size(), 1u);
  // new inst 1 is s_lshl_b64 s[8:9], s[8:9], 1 (SOP2: op[29:23], ssrc1[15:8], ssrc0[7:0]).
  const struct {
    const char *label;
    uint32_t mask;
  } mutations[] = {
      {"immediate", 0x00000300u}, // ssrc1 inline 1 -> 2
      {"opcode", 0x01000000u},    // s_lshl_b64 -> s_lshr_b64
      {"operand", 0x0000000eu},   // ssrc0 s[8:9] -> s[6:7]
  };
  const std::string dir = temp_dir("inject");
  for (const auto &m : mutations) {
    SCOPED_TRACE(m.label);
    MutationSpec spec;
    spec.new_side = true;
    spec.inst = 1;
    spec.xor_mask = m.mask;
    spec.label = m.label;
    std::string error;
    auto syn = make_synthetic_mutation(d.regions[0], spec, error);
    ASSERT_TRUE(syn) << error;
    EXPECT_TRUE(syn->synthetic);
    BaselinePolicy policy(2);
    TestOptions opt;
    opt.cases = 300;
    opt.seed = 2;
    opt.counterexample_dir = dir;
    RegionResult r = test_region(*syn, opt, policy);
    EXPECT_EQ(r.test_status, TestStatus::LocalMismatch);
    EXPECT_TRUE(r.synthetic);
    ASSERT_FALSE(r.counterexample_files.empty());
    auto text = read_text_file(r.counterexample_files[0]);
    ASSERT_TRUE(text);
    auto bundle = Json::parse(*text, &error);
    ASSERT_TRUE(bundle) << error;
    EXPECT_TRUE((*bundle)["synthetic"].as_bool());
    ReplayOutcome replay = replay_counterexample(*bundle, error);
    EXPECT_TRUE(error.empty()) << error;
    EXPECT_TRUE(replay.reproduced) << replay.summary;
  }
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffValidator, ReplayRejectsAnEditedBundle) {
  CaseData d = analyze(k_scclive_old, k_scclive_new);
  const std::string dir = temp_dir("tamper");
  TestRun r = run(d, 100, 1, dir);
  ASSERT_FALSE(r.results[0].counterexample_files.empty());
  auto text = read_text_file(r.results[0].counterexample_files[0]);
  std::string error;
  auto bundle = Json::parse(*text, &error);
  ASSERT_TRUE(bundle) << error;
  ReplayOutcome ok = replay_counterexample(*bundle, error);
  EXPECT_TRUE(ok.reproduced) << ok.summary << error;
  // Change one instruction word inside the recorded region.
  std::string edited = *text;
  const std::string word =
      (*bundle)["region"]["new"]["insts"].items()[0]["words"].items()[0].as_string();
  const std::string anchor = "\"words\": [\"" + word;
  const size_t pos = edited.find(anchor);
  ASSERT_NE(pos, std::string::npos);
  const size_t last = pos + anchor.size() - 1;
  edited[last] = edited[last] == '0' ? '1' : '0';
  auto tampered = Json::parse(edited, &error);
  ASSERT_TRUE(tampered) << error;
  error.clear();
  ReplayOutcome bad = replay_counterexample(*tampered, error);
  EXPECT_FALSE(bad.reproduced);
  EXPECT_NE(error.find("prepared key"), std::string::npos) << error;
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffFormat, CaseJsonRoundTripsAndKeepsKeys) {
  CaseData d = analyze(k_twohunk_old, k_twohunk_new);
  std::string error;
  auto text = d.to_json().dump();
  auto parsed = Json::parse(text, &error);
  ASSERT_TRUE(parsed) << error;
  auto back = CaseData::from_json(*parsed, error);
  ASSERT_TRUE(back) << error;
  ASSERT_EQ(back->regions.size(), d.regions.size());
  for (size_t i = 0; i < d.regions.size(); ++i) {
    EXPECT_EQ(back->regions[i].prepared_key, d.regions[i].prepared_key);
    EXPECT_EQ(back->regions[i].contract.status, d.regions[i].contract.status);
    EXPECT_EQ(back->regions[i].contract.observable, d.regions[i].contract.observable);
  }
  EXPECT_FALSE(Json::parse("{\"a\": [1, 2,]}", &error).has_value());
  auto j = Json::parse("{\"s\":\"x\\u0041\\n\",\"n\":-3,\"u\":18446744073709551615}", &error);
  ASSERT_TRUE(j) << error;
  EXPECT_EQ((*j)["s"].as_string(), "xA\n");
  EXPECT_EQ((*j)["n"].as_int(), -3);
  EXPECT_EQ((*j)["u"].as_uint(), UINT64_MAX);
}

} // namespace
