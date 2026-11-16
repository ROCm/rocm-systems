// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file isa_diff_test.cpp
/// @brief Validator tests for rj_isa_diff (positive, negative and rejection cases).
///
/// Fixtures are LLVM-assembled gfx950 words (fixtures/fixtures.asm -> fixtures.inc);
/// the tests themselves never run LLVM.

#include "boundary.h"
#include "case_io.h"
#include "effects.h"
#include "input_policy.h"
#include "kernel_model.h"
#include "mutation.h"
#include "region_runner.h"
#include "result_cache.h"

#include "rocjitsu/code/kernel_descriptor_abi.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <random>
#include <sstream>
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

TEST(IsaDiffTargets, TableNamesEveryRocjitsuTargetWithItsWaveSizes) {
  const auto gfx1100 = target_by_name("gfx1100");
  ASSERT_TRUE(gfx1100);
  EXPECT_EQ(gfx1100->arch, ROCJITSU_CODE_ARCH_RDNA3);
  EXPECT_EQ(gfx1100->wave_size, 32u);
  EXPECT_EQ(gfx1100->major, 11u);
  EXPECT_EQ(gfx1100->sgprs(), 106u);
  EXPECT_EQ(gfx1100->m0_selector(), 125);
  EXPECT_EQ(gfx1100->null_selector(), 124);
  const auto gfx942 = target_by_elf_mach(0x4c);
  ASSERT_TRUE(gfx942);
  EXPECT_EQ(gfx942->name, "gfx942");
  EXPECT_EQ(gfx942->arch, ROCJITSU_CODE_ARCH_CDNA3);
  EXPECT_EQ(gfx942->wave_size, 64u);
  EXPECT_EQ(gfx942->sgprs(), 102u);
  EXPECT_EQ(gfx942->m0_selector(), 124);
  EXPECT_EQ(target_by_name("gfx1030")->null_selector(), 125);
  EXPECT_EQ(target_by_name("gfx908")->arch, ROCJITSU_CODE_ARCH_CDNA1);
  EXPECT_FALSE(target_by_name("gfx908")->unified_vgprs);
  EXPECT_TRUE(target_by_name("gfx90a")->unified_vgprs);
  EXPECT_EQ(target_by_name("gfx1151")->arch, ROCJITSU_CODE_ARCH_RDNA3_5);
  EXPECT_EQ(target_by_name("gfx1201")->arch, ROCJITSU_CODE_ARCH_RDNA4);
  EXPECT_EQ(target_by_name("gfx1250")->arch, ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_FALSE(target_by_name("gfx1250")->has_ieee_dx10());
  EXPECT_TRUE(target_by_name("gfx1151")->has_ieee_dx10());
  // No RocJITsu ISA: Vega, SI/CI/VI, GFX13.
  for (const char *t : {"gfx900", "gfx906", "gfx803", "gfx600", "gfx1310"})
    EXPECT_FALSE(target_by_name(t)) << t;
  EXPECT_FALSE(target_by_elf_mach(0x2c)); // gfx900
  // Wave sizes: GFX9 is wave64 only, gfx1250 wave32 only, RDNA both.
  EXPECT_FALSE(with_wave_size(*gfx942, 32));
  EXPECT_FALSE(with_wave_size(*target_by_name("gfx1250"), 64));
  EXPECT_EQ(with_wave_size(*gfx1100, 64)->wave_size, 64u);
  EXPECT_EQ(parse_target("gfx1030:w64")->wave_size, 64u);
  EXPECT_EQ(parse_target("gfx1030:wave32")->wave_size, 32u);
  EXPECT_FALSE(parse_target("gfx950:w32"));
  EXPECT_FALSE(parse_target("gfx1030:w16"));
  EXPECT_EQ(parse_target("gfx1100:w64")->label(), "gfx1100 wave64");
  // A contract names its target; unsupported pairs are rejected.
  RegionContract k;
  k.set_gpu(*with_wave_size(*gfx1100, 64));
  EXPECT_EQ(k.target, "gfx1100");
  EXPECT_EQ(k.wave, "wave64");
  EXPECT_EQ(k.gpu()->wave_size, 64u);
  k.wave = "wave32";
  k.target = "gfx950";
  EXPECT_FALSE(k.gpu());
}

TEST(IsaDiffEffects, ClassifiesSaluOperandsPrecisely) {
  auto insts = decode_all(k_effects);
  ASSERT_EQ(insts.size(), 21u);
  auto fx = [&](size_t i) { return classify(*insts[i], Target{}); };

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

  for (size_t i : {10u, 11u, 12u}) { // s_cbranch_scc0 / execz / vccnz: regions follow them
    EXPECT_TRUE(fx(i).p0);
    EXPECT_TRUE(fx(i).branch);
    EXPECT_TRUE(fx(i).control_flow);
    EXPECT_FALSE(fx(i).implicit_special_reads) << i;
  }
  EXPECT_EQ(fx(10).reads, locs("scc"));
  EXPECT_EQ(fx(11).reads, locs("exec_lo,exec_hi"));
  EXPECT_EQ(fx(12).reads, locs("vcc_lo,vcc_hi"));

  Effects getpc = fx(13);
  EXPECT_FALSE(getpc.p0);
  EXPECT_TRUE(getpc.reads_pc);
  EXPECT_TRUE(fx(14).p0); // s_waitcnt: executed in regions (memory model)
  EXPECT_FALSE(fx(14).implicit_special_reads);
  Effects load = fx(15);
  EXPECT_TRUE(load.p0); // s_load_dword: a region-executable scalar load
  EXPECT_TRUE(load.mem_read);
  EXPECT_FALSE(load.implicit_special_reads); // SMEM: its operands name everything it reads
  EXPECT_TRUE(load.implicit_reads.none());
  EXPECT_EQ(load.reads, locs("s0,s1"));
  // v_mov_b32 v0, s1: a plain VALU instruction is region-executable. It reads s1 and
  // EXEC (the lanes it writes) and writes v0.
  Effects vmov = fx(16);
  EXPECT_TRUE(vmov.p0) << vmov.reject_reason;
  EXPECT_TRUE(vmov.valu);
  EXPECT_EQ(vmov.reads, locs("s1,exec_lo,exec_hi"));
  EXPECT_TRUE(vmov.writes.none());
  EXPECT_EQ(format_vgprs(vmov.vwrites), "v0");
  EXPECT_TRUE(vmov.vreads.none());
  EXPECT_FALSE(vmov.implicit_special_reads);
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

TEST(IsaDiffKernelModel, ImplicitStoreBaseReadsKeepLivenessPrecise) {
  // global_store_* reports its SGPR base pair as implicit uses. Those are now translated
  // into tracked reads instead of an unknown effect, so a store after the region no
  // longer makes every location live, and a renamed dead SALU temporary (s8 vs s10) is
  // not reported as a mismatch.
  auto insts = decode_all(k_valutemp_old);
  ASSERT_EQ(insts.size(), 7u);
  RegisterSet implicit;
  insts[5]->implicit_uses(implicit);
  EXPECT_FALSE(implicit.none()); // the decoder does report hidden uses here
  const Effects fx = classify(*insts[5], Target{});
  EXPECT_FALSE(fx.unknown_effect);
  EXPECT_TRUE(fx.reads.test(6));
  EXPECT_TRUE(fx.reads.test(7));
  // Region-executable now (memory model): the EXEC read is an ordinary read, and a store
  // never reads VCC. (The decoder lists M0 among a global access's implicit uses; keeping
  // that read is conservative.)
  EXPECT_TRUE(fx.p0);
  EXPECT_TRUE(fx.mem_write);
  EXPECT_TRUE(fx.reject_reason.empty()) << fx.reject_reason;
  EXPECT_TRUE(fx.implicit_reads.none());
  EXPECT_EQ(fx.reads & locs("exec_lo,exec_hi,vcc_lo,vcc_hi"), locs("exec_lo,exec_hi"));
  CaseData d = analyze(k_valutemp_old, k_valutemp_new);
  ASSERT_EQ(d.regions.size(), 1u);
  const RegionContract &k = d.regions[0].contract;
  EXPECT_EQ(k.status, ContractStatus::Established);
  EXPECT_TRUE(k.observable.test(5));
  EXPECT_FALSE(k.observable.test(8));
  EXPECT_FALSE(k.observable.test(10));
  TestRun t = run(d, 500);
  ASSERT_EQ(t.results.size(), 1u);
  EXPECT_EQ(t.results[0].test_status, TestStatus::NoMismatchFound);
  EXPECT_EQ(t.overall(), "NO_MISMATCH_FOUND");
}

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
  // The branch is a region of its own; its taken exit now goes elsewhere than old's, which
  // has no matched partner: a mapping inconclusive, never a pass.
  ASSERT_FALSE(d.regions.empty());
  EXPECT_EQ(d.regions[0].status, RegionStatus::Inconclusive);
  EXPECT_NE(d.regions[0].reason.find("exit"), std::string::npos) << d.regions[0].reason;
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
  // The mandatory boundary sweep runs first, then exactly 2000 random cases.
  EXPECT_GT(r.results[0].boundary.valid, 0u);
  EXPECT_EQ(r.results[0].random.valid, 2000u);
  EXPECT_EQ(r.results[0].valid_pairs, r.results[0].boundary.valid + 2000u);
  // Only the region executed: 1 old and 2 new instructions per case.
  EXPECT_EQ(r.results[0].old_insts_executed, r.results[0].executed_pairs);
  EXPECT_EQ(r.results[0].new_insts_executed, 2 * r.results[0].executed_pairs);
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
  EXPECT_EQ(a.random.valid, 1500u);
  EXPECT_EQ(a.valid_pairs, a.boundary.valid + 1500u);
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
  PlanOptions no_derive;
  no_derive.derive_constraints = false;
  CaseData bare = analyze(k_constraint_old, k_constraint_new, no_derive);
  ASSERT_EQ(bare.regions.size(), 1u);
  EXPECT_EQ(run(bare).results[0].test_status, TestStatus::LocalMismatch);

  // The common prefix (s_and_b32 s4, s4, 0xff) proves s4 < 256: a derived, established
  // constraint. The verdict is the constrained one; the unrestricted run is reported
  // separately and flags the rewrite as constraint-dependent.
  CaseData derived = analyze(k_constraint_old, k_constraint_new);
  ASSERT_EQ(derived.regions.size(), 1u);
  const RegionContract &k = derived.regions[0].contract;
  ASSERT_EQ(k.constraints.size(), 1u);
  EXPECT_EQ(k.constraints[0].loc, 4u);
  EXPECT_EQ(k.constraints[0].known_zero, 0xffffff00u);
  EXPECT_EQ(k.constraints[0].source, "common_prefix_analysis");
  EXPECT_EQ(k.status, ContractStatus::Established);
  TestRun d = run(derived, 1000);
  EXPECT_EQ(d.results[0].test_status, TestStatus::NoMismatchFound);
  ASSERT_FALSE(d.results[0].unrestricted.is_null());
  EXPECT_EQ(d.results[0].unrestricted["test_status"].as_string(), "LOCAL_MISMATCH");
  EXPECT_TRUE(d.results[0].unrestricted["constraint_dependent"].as_bool());
  EXPECT_EQ(d.overall(), "NO_MISMATCH_FOUND");

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

TEST(IsaDiffValidator, DerivedConstraintsComeOnlyFromCommonCode) {
  PlanOptions no_merge;
  no_merge.merge_same_block = false;
  CaseData split = analyze(k_twohunk_old, k_twohunk_new, no_merge);
  ASSERT_EQ(split.regions.size(), 2u);
  const RegionContract &k = split.regions[1].contract;
  bool s9_exact = false, s4_constrained = false;
  for (const auto &c : k.constraints) {
    EXPECT_EQ(c.source, "common_prefix_analysis");
    s9_exact |= c.loc == 9 && c.known_one == 7u && c.known_zero == ~7u; // common s_mov_b32 s9, 7
    s4_constrained |= c.loc == 4; // written by the changed region r0: nothing may be derived
  }
  EXPECT_TRUE(s9_exact);
  EXPECT_FALSE(s4_constrained);
  // Kernel inputs are never constrained: the first region's inputs are unknown.
  for (const auto &c : split.regions[0].contract.constraints)
    EXPECT_NE(c.loc, 2u);
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
  // r1 is changed code downstream that reads s4, so it may compensate the difference: the
  // local mismatch is a witness, not a verdict, and the overall result says so.
  EXPECT_EQ(split.regions[0].contract.downstream_changed, std::vector<std::string>{"r1"});
  EXPECT_TRUE(split.regions[0].contract.downstream_reads.test(4));
  EXPECT_TRUE(r.results[0].compensable);
  EXPECT_EQ(r.overall(), "INCONCLUSIVE");
  EXPECT_EQ(r.report["overall"]["witness_scope"].as_string(), "LOCAL_ONLY");
  bool listed = false;
  for (const Json &u : r.report["unresolved"].items())
    listed |= u["what"].as_string().find("may be compensated by changed code downstream (r1)") !=
              std::string::npos;
  EXPECT_TRUE(listed);
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

  // Register-allocation-only change (granulated VGPR and SGPR counts in rsrc1) while both
  // kernels' registers fit their allocation: informational.
  Artifact a = artifact_from_words(k_shift_old, "old");
  Artifact b = artifact_from_words(k_shift_new, "new");
  a.descriptor.assign(64, 0);
  b.descriptor.assign(64, 0);
  b.descriptor[48] = 0x41; // compute_pgm_rsrc1
  std::string error;
  auto alloc = analyze_artifacts(a, b, {}, error);
  ASSERT_TRUE(alloc) << error;
  EXPECT_TRUE(alloc->unresolved.empty());
  EXPECT_EQ(alloc->regions[0].contract.status, ContractStatus::Established);
  EXPECT_EQ(run(*alloc).overall(), "NO_MISMATCH_FOUND");

  // Any other rsrc1 bit (IEEE_MODE, bit 23): contract UNRESOLVED, scope partial.
  b.descriptor.assign(64, 0);
  b.descriptor[50] = 0x80;
  auto desc = analyze_artifacts(a, b, {}, error);
  ASSERT_TRUE(desc) << error;
  EXPECT_EQ(desc->regions[0].contract.status, ContractStatus::Unresolved);
  TestRun rd = run(*desc);
  EXPECT_EQ(rd.overall(), "INCONCLUSIVE");
  EXPECT_EQ(rd.coverage(), "PARTIAL");

  // The same allocation change is semantic when a kernel names more registers than its
  // descriptor allocates (s21 plus the six extra SGPRs exceed granule 0's 16).
  Artifact c = artifact_from_words(k_clobber_old, "old");
  Artifact e = artifact_from_words(k_clobber_old, "new");
  c.descriptor.assign(64, 0);
  e.descriptor.assign(64, 0);
  e.descriptor[48] = 0x41;
  auto over = analyze_artifacts(c, e, {}, error);
  ASSERT_TRUE(over) << error;
  EXPECT_FALSE(over->unresolved.empty());
  EXPECT_EQ(run(*over).overall(), "INCONCLUSIVE");

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
    Sample s = policy.propose({}, d.regions[0].contract, History{}, k);
    saw_zero_exec = s.value[kLocExecLo] == 0 && s.value[kLocExecHi] == 0;
  }
  EXPECT_TRUE(saw_zero_exec);
  TestRun r = run(d, 3000, 9);
  EXPECT_EQ(r.results[0].test_status, TestStatus::NoMismatchFound);
  EXPECT_EQ(r.results[0].valid_pairs, r.results[0].accepted);
}

TEST(IsaDiffValidator, PcAndHwregRegionsAreUnsupportedWhileMemoryRuns) {
  // Waits and scalar loads are executed (memory model): an extra vmcnt field changes
  // nothing, and a load from offset 0x4 instead of 0x0 reads a different input word.
  CaseData wait = analyze(k_waitcnt_old, k_waitcnt_new);
  ASSERT_EQ(wait.regions.size(), 1u);
  ASSERT_EQ(wait.regions[0].status, RegionStatus::Ready) << wait.regions[0].reason;
  EXPECT_EQ(run(wait).results[0].test_status, TestStatus::NoMismatchFound);
  CaseData smem = analyze(k_smem_old, k_smem_new);
  ASSERT_EQ(smem.regions.size(), 1u);
  ASSERT_EQ(smem.regions[0].status, RegionStatus::Ready) << smem.regions[0].reason;
  EXPECT_TRUE(smem.regions[0].contract.memory_reads);
  bool pointer = false;
  for (const auto &c : smem.regions[0].contract.constraints)
    pointer |= c.source == "memory_model" && c.loc == 1 && c.known_one == 0x10u;
  EXPECT_TRUE(pointer); // s[0:1] is a memory base: a canonical pointer input
  TestRun sr = run(smem);
  ASSERT_EQ(sr.results[0].test_status, TestStatus::LocalMismatch);
  EXPECT_TRUE(sr.results[0].examples.at(0).locs.test(4));

  struct Case {
    const std::vector<uint32_t> &o, &n;
    const char *needle;
  };
  const Case cases[] = {
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

// ---------------------------------------------------------------------------
// Mandatory boundary / NaN / corner-case sweep

std::vector<RegionInst> region_from_words(const std::vector<uint32_t> &words) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  std::vector<uint32_t> padded = words;
  padded.resize(words.size() + 4, 0);
  std::vector<RegionInst> out;
  size_t off = 0;
  while (off < words.size()) {
    auto d = decoder->decode_window(std::span<const uint32_t>(padded).subspan(off), off * 4);
    if (d.failed()) {
      ADD_FAILURE() << "decode failed at word " << off;
      break;
    }
    const size_t n = static_cast<size_t>(d.value()->size()) / 4;
    RegionInst ri;
    ri.pc = off * 4;
    ri.words.assign(words.begin() + static_cast<std::ptrdiff_t>(off),
                    words.begin() + static_cast<std::ptrdiff_t>(off + n));
    ri.text = d.value()->disassemble();
    if (std::string(d.value()->mnemonic()) != "s_endpgm")
      out.push_back(std::move(ri));
    off += n;
  }
  return out;
}

bool sweep_has(const BoundarySweep &sweep,
               std::initializer_list<std::pair<unsigned, uint32_t>> want) {
  for (const auto &c : sweep.cases) {
    const Sample s = sweep.materialize(c);
    bool all = true;
    for (const auto &[l, v] : want)
      all = all && s.value[l] == v;
    if (all)
      return true;
  }
  return false;
}

TEST(IsaDiffBoundary, EveryEntryRegisterSeesEveryCatalogValue) {
  CaseData d = analyze(k_exec_old, k_exec_new);
  ASSERT_EQ(d.regions.size(), 1u);
  const RegionPair &r = d.regions[0];
  const RegionFeatures f = make_region_features(r.old_side.insts, r.new_side.insts);
  const BoundarySweep sweep = build_boundary_sweep(f.roles, r.contract, {});
  EXPECT_FALSE(sweep.truncated);
  for (unsigned l = 0; l < kNumLocs; ++l) {
    if (!r.contract.entry_equal.test(l) || l == kLocScc)
      continue;
    for (const NamedValue &v : catalog32())
      EXPECT_TRUE(sweep_has(sweep, {{l, v.value}})) << loc_name(l) << "=" << v.name;
  }
  // 64-bit operands (EXEC and s[2:3]) see f64 specials and EXEC lane patterns.
  for (uint64_t v : {0x7ff8000000000000ull, 0xfff0000000000000ull, 0x8000000000000000ull,
                     0x00000000ffffffffull, 0xffffffff00000000ull, 0x0000000000000001ull}) {
    const auto lo = static_cast<uint32_t>(v), hi = static_cast<uint32_t>(v >> 32);
    EXPECT_TRUE(sweep_has(sweep, {{kLocExecLo, lo}, {kLocExecHi, hi}})) << hex64(v);
    EXPECT_TRUE(sweep_has(sweep, {{2, lo}, {3, hi}})) << hex64(v);
  }
  // When SCC is an input, corner values run with both SCC values.
  RegionContract with_scc = r.contract;
  with_scc.entry_equal.set(kLocScc);
  with_scc.canary.reset(kLocScc);
  const BoundarySweep s2 = build_boundary_sweep(f.roles, with_scc, {});
  EXPECT_TRUE(sweep_has(s2, {{2, 0x7fc00000u}, {kLocScc, 0}}));
  EXPECT_TRUE(sweep_has(s2, {{2, 0x7fc00000u}, {kLocScc, 1}}));
  // Deterministic: the same inputs build the same sweep.
  const BoundarySweep again = build_boundary_sweep(f.roles, r.contract, {});
  ASSERT_EQ(again.cases.size(), sweep.cases.size());
  for (size_t i = 0; i < sweep.cases.size(); ++i)
    EXPECT_EQ(again.cases[i].label, sweep.cases[i].label);
}

TEST(IsaDiffBoundary, RoleValuesAreAimedAtTheSensitiveOperands) {
  const auto insts = region_from_words(k_roles);
  ASSERT_EQ(insts.size(), 7u);
  const RegionRoles roles = infer_roles(insts, insts);
  EXPECT_TRUE(roles.decoded);
  EXPECT_TRUE(roles.roles[3] & kRoleShiftAmount);  // s_lshl_b32 s4, s2, s3
  EXPECT_TRUE(roles.roles[7] & kRoleBfeControl);   // s_bfe_u32 s5, s6, s7
  EXPECT_TRUE(roles.roles[9] & kRoleSextInput);    // s_sext_i32_i8 s8, s9
  EXPECT_TRUE(roles.roles[18] & kRoleShiftAmount); // s_lshl_b64 ..., s18
  ASSERT_EQ(roles.const_relations.size(), 1u);
  EXPECT_EQ(roles.const_relations[0].loc, 10u);
  EXPECT_EQ(roles.const_relations[0].constant, 0x12345u);
  EXPECT_NE(std::ranges::find(roles.pairs, OperandPair{12, 13, false, ""}), roles.pairs.end());
  EXPECT_NE(std::ranges::find(roles.pairs, OperandPair{20, 22, true, ""}), roles.pairs.end());

  RegionContract k;
  k.entry_equal = locs("s2,s3,s6,s7,s9,s10,s12,s13,s16,s17,s18,s20,s21,s22,s23");
  k.canary = ~k.entry_equal;
  k.wide_reads = locs("s16,s20,s22");
  const BoundarySweep sweep = build_boundary_sweep(roles, k, {});
  EXPECT_FALSE(sweep.truncated);
  // Shift amounts: at and past the width, and high garbage over a zero low field.
  for (uint32_t v : {32u, 33u, 63u, 64u, 65u, 0xffffffe0u, 0xffffffc0u})
    EXPECT_TRUE(sweep_has(sweep, {{3, v}})) << hex32(v);
  EXPECT_TRUE(sweep_has(sweep, {{18, 64u}}));
  // S_BFE control: offset 31 width 32, width 0, offset beyond the field.
  for (uint32_t v : {(32u << 16) | 31u, (0u << 16) | 16u, (1u << 16) | 64u})
    EXPECT_TRUE(sweep_has(sweep, {{7, v}})) << hex32(v);
  // Sign-extension edges.
  for (uint32_t v : {0x7fu, 0x80u, 0x17fu, 0x12345680u})
    EXPECT_TRUE(sweep_has(sweep, {{9, v}})) << hex32(v);
  // Values around the literal it is compared with.
  for (uint32_t v : {0x12345u, 0x12344u, 0x12346u, 0u - 0x12345u, ~0x12345u})
    EXPECT_TRUE(sweep_has(sweep, {{10, v}})) << hex32(v);
  // Carry / overflow / borrow pairs on the two add sources.
  EXPECT_TRUE(sweep_has(sweep, {{12, 0xffffffffu}, {13, 1u}}));
  EXPECT_TRUE(sweep_has(sweep, {{12, 0x7fffffffu}, {13, 1u}}));
  EXPECT_TRUE(sweep_has(sweep, {{12, 0u}, {13, 1u}}));
  EXPECT_TRUE(sweep_has(sweep, {{12, 0x7fc00000u}, {13, 0x7fc00000u}}));
  // 64-bit compare pair: equal high halves, low halves differing by one.
  EXPECT_TRUE(sweep_has(sweep, {{20, 0u}, {21, 1u}, {22, 1u}, {23, 1u}}));
}

TEST(IsaDiffBoundary, OffByOneAgainstALiteralIsAlwaysCaught) {
  // x > 0x12345 rewritten as x >= 0x12345: wrong only at x == 0x12345, which random
  // 32-bit sampling essentially never hits. The sweep hits it on every run.
  CaseData bug = analyze(k_offbyone_old, k_offbyone_bug);
  ASSERT_EQ(bug.regions.size(), 1u);
  for (uint64_t seed : {1u, 2u, 3u, 99u}) {
    SCOPED_TRACE(seed);
    TestRun r = run(bug, /*cases=*/0, seed);
    ASSERT_EQ(r.results.size(), 1u);
    EXPECT_EQ(r.results[0].random.cases, 0u);
    EXPECT_EQ(r.results[0].test_status, TestStatus::LocalMismatch);
    EXPECT_GT(r.results[0].boundary.mismatches, 0u);
    ASSERT_FALSE(r.results[0].examples.empty());
    EXPECT_EQ(r.results[0].examples[0].phase, "boundary");
    EXPECT_EQ(r.results[0].examples[0].init_old.get(2), 0x12345u);
    EXPECT_TRUE(r.results[0].examples[0].locs.test(kLocScc));
  }
  // The correct rewrite (x >= 0x12346) passes the same sweep.
  TestRun fixed = run(analyze(k_offbyone_old, k_offbyone_fixed), /*cases=*/0);
  EXPECT_EQ(fixed.results[0].test_status, TestStatus::NoMismatchFound);
  EXPECT_GT(fixed.results[0].boundary.valid, 0u);
}

TEST(IsaDiffBoundary, ConstraintEdgesRunAndOutOfContractValuesAreRejectedNotChanged) {
  PlanOptions manual_only; // this test is about a hand-written range constraint
  manual_only.derive_constraints = false;
  CaseData d = analyze(k_constraint_old, k_constraint_new, manual_only);
  ASSERT_EQ(d.regions.size(), 1u);
  Constraint c;
  c.loc = 4;
  c.umax = 0xffu;
  d.regions[0].contract.constraints.push_back(c);
  TestRun r = run(d, /*cases=*/0);
  const RegionResult &res = r.results[0];
  EXPECT_EQ(res.test_status, TestStatus::NoMismatchFound);
  EXPECT_GT(res.boundary.valid, 0u);
  EXPECT_GT(res.boundary.cases, res.boundary.accepted);
  bool rejected_above_umax = false;
  for (const auto &[reason, count] : res.rejections)
    rejected_above_umax |= reason.rfind("boundary constraint: above umax", 0) == 0 && count > 0;
  EXPECT_TRUE(rejected_above_umax);
  EXPECT_GT(res.boundary_families.count("constraint_edge"), 0u);
  const RegionFeatures f =
      make_region_features(d.regions[0].old_side.insts, d.regions[0].new_side.insts);
  const BoundarySweep sweep = build_boundary_sweep(f.roles, d.regions[0].contract, {});
  EXPECT_TRUE(sweep_has(sweep, {{4, 0xffu}}));
  EXPECT_TRUE(sweep_has(sweep, {{4, 0xfeu}}));
  EXPECT_TRUE(sweep_has(sweep, {{4, 0x100u}})); // generated, then rejected by the contract
}

TEST(IsaDiffBoundary, TruncatedSweepIsReportedAsPartialCoverage) {
  CaseData d = analyze(k_shift_old, k_shift_new);
  TestOptions opt;
  opt.cases = 100;
  opt.max_boundary_cases = 10;
  std::vector<RegionResult> results;
  Json report = test_case(d, opt, "", &results);
  ASSERT_EQ(results.size(), 1u);
  EXPECT_TRUE(results[0].boundary_truncated);
  EXPECT_EQ(results[0].boundary.cases, 10u);
  EXPECT_GT(results[0].boundary_generated, 10u);
  EXPECT_EQ(results[0].coverage_scope, CoverageScope::Partial);
  EXPECT_NE(results[0].reason.find("truncated"), std::string::npos);
  EXPECT_EQ(report["overall"]["coverage_scope"].as_string(), "PARTIAL");
  EXPECT_TRUE(report["regions"].items()[0]["boundary_sweep"]["truncated"].as_bool());
}

// ---------------------------------------------------------------------------
// Memory operations outstanding at the cutpoint (handoff 6.1)

TEST(IsaDiffPending, OutstandingLoadIntoARegionInputLeavesTheContextUnresolved) {
  CaseData d = analyze(k_pending_conflict_old, k_pending_conflict_new);
  ASSERT_EQ(d.regions.size(), 1u);
  const RegionContract &k = d.regions[0].contract;
  EXPECT_EQ(k.pending_effects, "CONFLICT");
  EXPECT_EQ(k.status, ContractStatus::Unresolved);
  TestRun r = run(d, 200);
  // The commuted add shows no local mismatch, but the isolated run assumed s4 was ready:
  // the result is never a pass.
  EXPECT_EQ(r.results[0].test_status, TestStatus::NoMismatchFound);
  EXPECT_EQ(r.results[0].pending_effects, "CONFLICT");
  EXPECT_EQ(r.overall(), "INCONCLUSIVE");
  EXPECT_EQ(r.coverage(), "PARTIAL");
}

TEST(IsaDiffPending, DisjointWaitedAndPartiallyWaitedLoads) {
  CaseData disjoint = analyze(k_pending_disjoint_old, k_pending_disjoint_new);
  ASSERT_EQ(disjoint.regions.size(), 1u);
  EXPECT_EQ(disjoint.regions[0].contract.pending_effects, "DISJOINT");
  EXPECT_EQ(disjoint.regions[0].contract.status, ContractStatus::Established);
  EXPECT_EQ(run(disjoint, 200).overall(), "NO_MISMATCH_FOUND");

  CaseData waited = analyze(k_pending_waited_old, k_pending_waited_new);
  ASSERT_EQ(waited.regions.size(), 1u);
  EXPECT_EQ(waited.regions[0].contract.pending_effects, "NONE");
  EXPECT_EQ(waited.regions[0].contract.status, ContractStatus::Established);

  // lgkmcnt(1) after two SMEM loads guarantees neither: SMEM returns out of order.
  CaseData partial = analyze(k_pending_partial_old, k_pending_partial_new);
  ASSERT_EQ(partial.regions.size(), 1u);
  EXPECT_EQ(partial.regions[0].contract.pending_effects, "CONFLICT");
  std::string error;
  auto model = KernelModel::build(artifact_from_words(k_pending_partial_old, "p"), error);
  ASSERT_TRUE(model) << error;
  const PendingState &p = model->pending_before(3); // s_add_u32 s5, s4, 1
  EXPECT_EQ(p.count[PendingState::kDs], 1u);
  EXPECT_EQ(p.all_dests() & locs("s4,s10"), locs("s4,s10"));
  EXPECT_FALSE(model->pending_before(5).any()); // after s_waitcnt lgkmcnt(0)
}

// ---------------------------------------------------------------------------
// Composition (handoff 5.2)

TEST(IsaDiffComposition, UntestedUpstreamChangeMattersOnlyIfItReachesTheEntryRelation) {
  CaseData ok = analyze(k_comp_disjoint_old, k_comp_disjoint_new);
  ASSERT_EQ(ok.regions.size(), 2u);
  EXPECT_EQ(ok.regions[0].status, RegionStatus::Unsupported); // the s_getreg change
  EXPECT_EQ(ok.regions[1].contract.composition, "HOLDS");
  EXPECT_EQ(ok.regions[1].contract.status, ContractStatus::Assumed);
  ASSERT_FALSE(ok.regions[1].contract.depends_on.empty());
  EXPECT_EQ(ok.regions[1].contract.depends_on[0], "r0");

  CaseData bad = analyze(k_comp_conflict_old, k_comp_conflict_new);
  ASSERT_EQ(bad.regions.size(), 2u);
  EXPECT_EQ(bad.regions[1].contract.composition, "FAILS");
  EXPECT_EQ(bad.regions[1].contract.status, ContractStatus::Unresolved);
  bool names_s4 = false;
  for (const auto &s : bad.regions[1].contract.sources)
    names_s4 |= s.source == "composition_check" && s.what.find("{s4}") != std::string::npos;
  EXPECT_TRUE(names_s4);

  // Renamed registers: the downstream region's inputs come from inconclusive regions.
  CaseData rename = analyze(k_rename_old, k_rename_new);
  ASSERT_EQ(rename.regions.size(), 3u);
  EXPECT_EQ(rename.regions[2].contract.composition, "FAILS");
  EXPECT_EQ(rename.regions[2].contract.status, ContractStatus::Unresolved);

  // Two dependent hunks verified separately: r1's input s4 is part of r0's R_out.
  PlanOptions no_merge;
  no_merge.merge_same_block = false;
  CaseData split = analyze(k_twohunk_old, k_twohunk_new, no_merge);
  ASSERT_EQ(split.regions.size(), 2u);
  EXPECT_EQ(split.regions[1].contract.composition, "HOLDS");
  EXPECT_EQ(split.regions[0].contract.composition, "HOLDS");
}

// ---------------------------------------------------------------------------
// Automatic context expansion (handoff 5.2)

TEST(IsaDiffExpansion, CompensatedLocalMismatchIsResolvedByContext) {
  // s4 = s2 << 1 vs s2 << 2 differs locally, but the common s_and_b32 s4, s4, 1 masks it.
  CaseData d = analyze(k_expand_masked_old, k_expand_masked_new);
  ASSERT_EQ(d.regions.size(), 1u);
  ASSERT_FALSE(d.variants.empty());
  TestRun r = run(d, 300);
  const RegionResult &res = r.results[0];
  EXPECT_EQ(res.test_status, TestStatus::LocalMismatch); // the local fact is kept
  EXPECT_EQ(res.resolution, "RESOLVED_BY_CONTEXT");
  EXPECT_EQ(res.resolved_by, "r0.ctx-b0f1");
  EXPECT_EQ(res.resolved_contract, ContractStatus::Established);
  EXPECT_EQ(r.overall(), "NO_MISMATCH_FOUND");
  EXPECT_EQ(r.report["overall"]["witness_scope"].as_string(), "NONE");
  EXPECT_EQ(r.report["regions"].items()[0]["expansion"]["resolution"].as_string(),
            "RESOLVED_BY_CONTEXT");
}

TEST(IsaDiffExpansion, RealDifferencePersistsAndPrerequisiteContextResolves) {
  TestRun live = run(analyze(k_scclive_old, k_scclive_new));
  EXPECT_EQ(live.results[0].resolution, "PERSISTS_WITH_CONTEXT");
  EXPECT_GE(live.results[0].expansion_tried.size(), 2u);
  EXPECT_EQ(live.overall(), "LOCAL_MISMATCH");

  // s_lshr vs s_bfe is only equal after the common s_and_b32 s4, s4, 0xff.
  PlanOptions no_derive; // exercise expansion, not the derived prefix constraint
  no_derive.derive_constraints = false;
  TestRun constraint = run(analyze(k_constraint_old, k_constraint_new, no_derive));
  EXPECT_EQ(constraint.results[0].test_status, TestStatus::LocalMismatch);
  EXPECT_EQ(constraint.results[0].resolution, "RESOLVED_BY_CONTEXT");
  EXPECT_NE(constraint.results[0].resolved_by.find("b1"), std::string::npos);
  EXPECT_EQ(constraint.overall(), "NO_MISMATCH_FOUND");
}

// ---------------------------------------------------------------------------
// Counterexample reduction (handoff 10)

uint64_t entry_popcount(const MachineState &s, const LocSet &entry) {
  uint64_t n = 0;
  for (unsigned l = 0; l < kNumLocs; ++l)
    if (entry.test(l))
      n += static_cast<uint64_t>(std::popcount(s.get(l)));
  return n;
}

TEST(IsaDiffReduction, CounterexamplesAreShrunkSeedIndependentAndStillReplay) {
  CaseData d = analyze(k_scclive_old, k_scclive_new);
  ASSERT_EQ(d.regions.size(), 1u);
  const LocSet entry = d.regions[0].contract.entry_equal;
  const std::string dir = temp_dir("reduce");
  TestRun r = run(d, 300, 1, dir);
  ASSERT_FALSE(r.results[0].examples.empty());
  const Mismatch &m = r.results[0].examples[0];
  ASSERT_FALSE(m.reduction.is_null());
  EXPECT_GT(m.reduction["runs"].as_uint(), 0u);
  EXPECT_TRUE(m.reduction["seed_independent"].as_bool());
  EXPECT_TRUE(m.locs.test(kLocScc));
  // Only the two add sources stay non-zero, and the inputs got no larger.
  for (unsigned l = 0; l < kNumLocs; ++l) {
    if (entry.test(l) && l != 2 && l != 3) {
      EXPECT_EQ(m.init_old.get(l), 0u) << loc_name(l);
    }
  }
  std::string error;
  auto original = MachineState::from_json(m.reduction["original_initial_state"]["old"], error);
  ASSERT_TRUE(original) << error;
  EXPECT_LE(entry_popcount(m.init_old, entry), entry_popcount(*original, entry));
  // The bundle holds the reduced state and replays it exactly.
  ASSERT_FALSE(r.results[0].counterexample_files.empty());
  auto text = read_text_file(r.results[0].counterexample_files[0]);
  auto bundle = Json::parse(*text, &error);
  ASSERT_TRUE(bundle) << error;
  EXPECT_TRUE((*bundle)["reduction"]["seed_independent"].as_bool());
  ReplayOutcome replay = replay_counterexample(*bundle, error);
  EXPECT_TRUE(replay.reproduced) << replay.summary << error;

  // The literal off-by-one cannot shrink its trigger value, only everything else.
  TestRun obo = run(analyze(k_offbyone_old, k_offbyone_bug), 0);
  const Mismatch &o = obo.results[0].examples[0];
  EXPECT_EQ(o.init_old.get(2), 0x12345u);
  EXPECT_EQ(o.reduction["reduced_entry_values"].size(), 1u);

  // Reduction can be switched off.
  TestOptions off;
  off.cases = 100;
  off.reduce = false;
  BaselinePolicy policy(1);
  RegionResult raw = test_region(d.regions[0], off, policy);
  ASSERT_FALSE(raw.examples.empty());
  EXPECT_TRUE(raw.examples[0].reduction.is_null());
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffReduction, ReductionNeverLeavesTheContract) {
  CaseData d = analyze(k_scclive_old, k_scclive_new);
  Constraint c;
  c.loc = 3;
  c.known_one = 0x10u; // s3 must keep bit 4
  d.regions[0].contract.constraints.push_back(c);
  TestRun r = run(d, 300);
  ASSERT_EQ(r.results[0].test_status, TestStatus::LocalMismatch);
  for (const Mismatch &m : r.results[0].examples) {
    EXPECT_EQ(m.init_old.get(3) & 0x10u, 0x10u);
    Sample s;
    for (unsigned l = 0; l < kNumLocs; ++l)
      s.value[l] = m.init_old.get(l);
    EXPECT_FALSE(ConstraintChecker::validate(s, d.regions[0].contract).has_value());
  }
}

// ---------------------------------------------------------------------------
// Learning hooks and per-case log (handoff 8)

class RecordingPolicy final : public InputPolicy {
public:
  [[nodiscard]] std::string name() const override { return "recording"; }
  Sample propose(const RegionFeatures &f, const RegionContract &k, const History &h,
                 uint64_t case_index) override {
    history_sizes.push_back(h.cases);
    window_sizes.push_back(h.recent.size());
    return inner.propose(f, k, h, case_index);
  }
  void observe(const Feedback &fb) override { seen.push_back(fb); }

  BaselinePolicy inner{7};
  std::vector<Feedback> seen;
  std::vector<uint64_t> history_sizes;
  std::vector<size_t> window_sizes;
};

TEST(IsaDiffLearning, PolicyGetsHistoryRichFeedbackAndEveryCaseIsLogged) {
  CaseData d = analyze(k_scclive_old, k_scclive_new);
  RecordingPolicy policy;
  TestOptions opt;
  opt.cases = 50;
  opt.case_log_dir = temp_dir("caselog");
  RegionResult r = test_region(d.regions[0], opt, policy);

  // observe() saw every case of both phases; propose() saw the growing, bounded history.
  EXPECT_EQ(policy.seen.size(), r.boundary.cases + r.random.cases);
  ASSERT_EQ(policy.history_sizes.size(), 50u);
  EXPECT_EQ(policy.history_sizes.front(), r.boundary.cases);
  EXPECT_EQ(policy.history_sizes.back(), r.boundary.cases + 49u);
  for (size_t w : policy.window_sizes)
    EXPECT_LE(w, History::kWindow);

  bool saw_mismatch = false, saw_ok = false;
  for (const Feedback &fb : policy.seen) {
    EXPECT_TRUE(fb.phase == "boundary" || fb.phase == "random");
    if (!fb.accepted)
      continue;
    EXPECT_EQ(fb.exit_old, "OK");
    EXPECT_EQ(fb.insts_old, 1u);
    EXPECT_EQ(fb.insts_new, 1u);
    EXPECT_NE(fb.bucket, 0u);
    saw_mismatch = saw_mismatch || (fb.mismatch && fb.locs.test(kLocScc));
    saw_ok = saw_ok || fb.category == "ok";
  }
  EXPECT_TRUE(saw_mismatch);
  EXPECT_TRUE(saw_ok);
  // Novelty is a bounded classification, not one class per random value.
  EXPECT_GT(r.novelty_buckets, 1u);
  EXPECT_LT(r.novelty_buckets, 64u);
  EXPECT_EQ(r.novel_boundary + r.novel_random, r.novelty_buckets);

  // The log: a header with the opcode/effect summary, one line per case, a summary.
  ASSERT_FALSE(r.case_log.empty());
  auto text = read_text_file(r.case_log);
  ASSERT_TRUE(text);
  std::vector<Json> lines;
  std::istringstream in(*text);
  std::string line;
  std::string error;
  while (std::getline(in, line)) {
    auto j = Json::parse(line, &error);
    ASSERT_TRUE(j) << error << ": " << line;
    lines.push_back(std::move(*j));
  }
  ASSERT_EQ(lines.size(), policy.seen.size() + 2);
  const Json &header = lines.front();
  EXPECT_EQ(header["type"].as_string(), "header");
  EXPECT_EQ(header["seed"].as_uint(), opt.seed);
  EXPECT_EQ(header["old"].items()[0]["mnemonic"].as_string(), "s_add_u32");
  EXPECT_EQ(header["old"].items()[0]["writes"].as_string(), "s4,scc");
  EXPECT_EQ(header["new"].items()[0]["mnemonic"].as_string(), "s_add_i32");
  EXPECT_FALSE(header["pairs"].items().empty());
  const Json &first_case = lines[1];
  EXPECT_EQ(first_case["type"].as_string(), "case");
  EXPECT_EQ(first_case["phase"].as_string(), "boundary");
  EXPECT_FALSE(first_case["events"].items().empty());
  EXPECT_EQ(first_case["cost_ns"].size(), 5u);
  EXPECT_EQ(lines.back()["type"].as_string(), "summary");
  EXPECT_EQ(lines.back()["novelty_buckets"].as_uint(), r.novelty_buckets);
  std::filesystem::remove_all(opt.case_log_dir);
}

// ---------------------------------------------------------------------------
// Manual region and register map (handoff 3.1, 12)

TEST(IsaDiffManual, RegisterMapParsingAndPermutation) {
  std::string error;
  auto m = parse_register_map("s4=s8,s8=s9", error);
  ASSERT_TRUE(m) << error;
  RegionContract k;
  k.entry_map = *m;
  const auto f = k.entry_forward();
  const auto inv = k.entry_inverse();
  EXPECT_EQ(k.exit_forward()[4], 4u); // the exit map is independent
  EXPECT_EQ(f[4], 8u);
  EXPECT_EQ(f[8], 9u);
  EXPECT_EQ(f[9], 4u); // chain completed back to its start: still a permutation
  EXPECT_EQ(f[5], 5u);
  for (unsigned l = 0; l < kNumLocs; ++l)
    EXPECT_EQ(inv[f[l]], l);
  EXPECT_EQ(format_register_map(*m), "s4=s8,s8=s9");
  EXPECT_FALSE(parse_register_map("s4=s8,s5=s8", error)); // two sources, one target
  EXPECT_FALSE(parse_register_map("s4=s8,s4=s9", error)); // one source twice
  EXPECT_FALSE(parse_register_map("exec_lo=s8", error));  // specials keep identity
  EXPECT_FALSE(parse_register_map("s4", error));
  // Round trip through the contract JSON.
  k.entry_equal = all_locs();
  k.canary = LocSet{};
  k.status = ContractStatus::Assumed;
  auto back = RegionContract::from_json(k.to_json(), error);
  ASSERT_TRUE(back) << error;
  EXPECT_EQ(back->entry_map, k.entry_map);
  EXPECT_TRUE(back->exit_map.empty());
}

TEST(IsaDiffManual, ManualRegionStaysAssumedAndLeavesOtherHunksUnresolved) {
  PlanOptions opt;
  opt.manual_region = "old:5-6,new:0x14-0x18"; // s_add_u32 s6, s4/s8, s5
  CaseData d = analyze(k_rename_old, k_rename_new, opt);
  ASSERT_EQ(d.regions.size(), 1u);
  const RegionPair &r = d.regions[0];
  EXPECT_EQ(r.id, "m0");
  EXPECT_EQ(r.status, RegionStatus::Ready) << r.reason;
  EXPECT_EQ(r.contract.status, ContractStatus::Assumed);
  EXPECT_EQ(r.contract.composition, "NOT_CHECKED");
  // Without a map the planner uses identity: s4 and s8 are separate entry values.
  EXPECT_TRUE(r.contract.entry_equal.test(4));
  EXPECT_TRUE(r.contract.entry_equal.test(8));
  EXPECT_TRUE(d.variants.empty());
  // The two upstream hunks are not covered: they stay unresolved.
  size_t uncovered = 0;
  for (const auto &u : d.unresolved)
    uncovered += u.what.find("not covered by the manual region") != std::string::npos;
  EXPECT_EQ(uncovered, 2u);
  // Identity is the wrong correspondence here: old s4+s5 vs new s8+s5.
  TestRun t = run(d, 200);
  ASSERT_EQ(t.results.size(), 1u);
  EXPECT_EQ(t.results[0].test_status, TestStatus::LocalMismatch);
  EXPECT_NE(t.coverage(), "COMPLETE_FOR_DECLARED_SCOPE");
}

TEST(IsaDiffManual, RegisterMapReconcilesRenameButStaysAssumed) {
  PlanOptions opt;
  opt.manual_region = "old:5-6,new:5-6";
  opt.register_map = "s4=s8";
  CaseData d = analyze(k_rename_old, k_rename_new, opt);
  ASSERT_EQ(d.regions.size(), 1u);
  const RegionPair &r = d.regions[0];
  ASSERT_EQ(r.status, RegionStatus::Ready) << r.reason;
  EXPECT_EQ(r.contract.mapping, "user_register_map");
  EXPECT_EQ(r.contract.status, ContractStatus::Assumed);
  EXPECT_TRUE(r.contract.entry_equal.test(4));
  EXPECT_FALSE(r.contract.entry_equal.test(8)); // new s8 is old s4 under the map
  bool user_source = false;
  for (const auto &s : r.contract.sources)
    user_source |= s.source == "user_assumption" && s.what.find("s4=s8") != std::string::npos;
  EXPECT_TRUE(user_source);
  // The initial new state receives old s4's value at s8.
  MachineState so, sn;
  Sample sample;
  sample.value[4] = 0x1234;
  build_initial_states(r.contract, sample, Frame{}, 1, 0, so, sn);
  EXPECT_EQ(so.get(4), 0x1234u);
  EXPECT_EQ(sn.get(8), 0x1234u);
  TestRun t = run(d, 400);
  ASSERT_EQ(t.results.size(), 1u);
  EXPECT_EQ(t.results[0].test_status, TestStatus::NoMismatchFound) << t.results[0].reason;
  EXPECT_EQ(t.results[0].contract_status, ContractStatus::Assumed);
  EXPECT_GT(t.results[0].valid_pairs, 0u);

  // A wrong map is caught, not trusted.
  opt.register_map = "s4=s9";
  CaseData wrong = analyze(k_rename_old, k_rename_new, opt);
  ASSERT_EQ(wrong.regions.size(), 1u);
  TestRun tw = run(wrong, 200);
  EXPECT_EQ(tw.results[0].test_status, TestStatus::LocalMismatch);
}

TEST(IsaDiffManual, ManualRegionRejectsBadSpecsAndBlockCrossing) {
  std::string error;
  PlanOptions opt;
  opt.register_map = "s4=s8";
  EXPECT_FALSE(analyze_artifacts(artifact_from_words(k_rename_old, "old"),
                                 artifact_from_words(k_rename_new, "new"), opt, error));
  EXPECT_NE(error.find("--manual-region"), std::string::npos);
  for (const char *bad : {"old:5-6", "old:6-5,new:5-6", "old:5-6,new:0x15-0x18", "mid:5-6,new:5-6",
                          "old:5-6,new:5-99", "old:8-8,new:8-8"}) {
    PlanOptions o;
    o.manual_region = bad;
    EXPECT_FALSE(analyze_artifacts(artifact_from_words(k_rename_old, "old"),
                                   artifact_from_words(k_rename_new, "new"), o, error))
        << bad;
  }
  // A hand-picked range across a block boundary is not mapped by guesswork.
  PlanOptions cross;
  cross.manual_region = "old:3-6,new:3-6";
  CaseData d = analyze(k_rename_old, k_rename_new, cross);
  ASSERT_EQ(d.regions.size(), 1u);
  EXPECT_EQ(d.regions[0].status, RegionStatus::Inconclusive);
  EXPECT_NE(d.regions[0].reason.find("INCONCLUSIVE_MAPPING"), std::string::npos);
}

TEST(IsaDiffArtifact, MetadataFieldsAreComparedPerKernel) {
  Artifact a = artifact_from_words(k_rename_old, "old");
  Artifact b = artifact_from_words(k_rename_old, "new");
  a.metadata_parsed = b.metadata_parsed = true;
  a.metadata_sha256 = "1";
  b.metadata_sha256 = "2";
  a.kernel_metadata = {{".name", "\"k\""}, {".sgpr_count", "18"}, {".args[0].size", "8"}};
  b.kernel_metadata = {{".name", "\"k\""}, {".sgpr_count", "20"}, {".args[0].size", "8"}};
  std::string error;
  auto d = analyze_artifacts(a, b, {}, error);
  ASSERT_TRUE(d) << error;
  ASSERT_EQ(d->unresolved.size(), 1u);
  EXPECT_EQ(d->unresolved[0].kind, "metadata");
  EXPECT_EQ(d->unresolved[0].what, "kernel metadata field differs: .sgpr_count (old 18, new 20)");
  // The note differs only in another kernel's entry: informational, not a semantic change.
  b.kernel_metadata = a.kernel_metadata;
  auto e = analyze_artifacts(a, b, {}, error);
  ASSERT_TRUE(e) << error;
  EXPECT_TRUE(e->unresolved.empty());
  ASSERT_EQ(e->notes.size(), 1u);
  EXPECT_FALSE(e->notes[0].semantic);
  // Unparsed metadata falls back to the whole-note hash, conservatively semantic.
  b.metadata_parsed = false;
  auto f = analyze_artifacts(a, b, {}, error);
  ASSERT_TRUE(f) << error;
  ASSERT_EQ(f->unresolved.size(), 1u);
  EXPECT_NE(f->unresolved[0].what.find("not parsed field by field"), std::string::npos);
}

TEST(IsaDiffArtifact, AnalysisTimeLimitLeavesContractsUnresolved) {
  PlanOptions opt;
  opt.max_analysis_us = 1; // reached before the optional steps
  CaseData d = analyze(k_rename_old, k_rename_new, opt);
  ASSERT_FALSE(d.regions.empty());
  EXPECT_TRUE(d.variants.empty());
  for (const auto &r : d.regions) {
    EXPECT_EQ(r.contract.status, ContractStatus::Unresolved);
    EXPECT_EQ(r.contract.composition, "NOT_CHECKED");
  }
  ASSERT_TRUE(d.analysis.has("skipped_for_time"));
  EXPECT_EQ(d.analysis["skipped_for_time"].items().back().as_string(), "composition check");
  const Json &timing = d.analysis["timing"];
  for (const char *k : {"parse_ns", "decode_cfg_ns", "dataflow_ns", "align_classify_ns", "plan_ns"})
    EXPECT_TRUE(timing.has(k)) << k;
}

TEST(IsaDiffCache, ResultCacheHitsOnlyForTheSamePreparedRegionAndBudget) {
  const std::string cache = temp_dir("result_cache");
  const std::string cex = temp_dir("result_cache_cex");
  PlanOptions manual;
  manual.manual_region = "old:5-6,new:5-6"; // identity mapping is wrong here: mismatch
  CaseData d = analyze(k_rename_old, k_rename_new, manual);
  TestOptions opt;
  opt.cases = 100;
  opt.cache_dir = cache;
  opt.counterexample_dir = cex;
  std::vector<RegionResult> first, second;
  Json r1 = test_case(d, opt, "", &first);
  ASSERT_EQ(first.size(), 1u);
  EXPECT_TRUE(first[0].cached.is_null());
  EXPECT_EQ(r1["cache"]["misses"].as_uint(), 1u);
  ASSERT_EQ(first[0].test_status, TestStatus::LocalMismatch);
  Json r2 = test_case(d, opt, "", &second);
  ASSERT_EQ(second.size(), 1u);
  ASSERT_FALSE(second[0].cached.is_null());
  EXPECT_EQ(r2["cache"]["hits"].as_uint(), 1u);
  EXPECT_TRUE(second[0].cached["cache"]["hit"].as_bool());
  EXPECT_EQ(second[0].test_status, first[0].test_status);
  EXPECT_EQ(second[0].valid_pairs, first[0].valid_pairs);
  EXPECT_EQ(r2["overall"]["test_status"].as_string(), r1["overall"]["test_status"].as_string());
  // The cached counterexample is a self-contained copy that still replays.
  ASSERT_FALSE(second[0].counterexample_files.empty());
  EXPECT_EQ(second[0].counterexample_files[0].rfind(cache, 0), 0u);
  auto text = read_text_file(second[0].counterexample_files[0]);
  ASSERT_TRUE(text);
  std::string error;
  auto bundle = Json::parse(*text, &error);
  ASSERT_TRUE(bundle) << error;
  EXPECT_TRUE(replay_counterexample(*bundle, error).reproduced) << error;
  // Another seed, or the same bytes under a different contract, is a miss.
  TestOptions other_seed = opt;
  other_seed.seed = 2;
  std::vector<RegionResult> third;
  (void)test_case(d, other_seed, "", &third);
  EXPECT_TRUE(third[0].cached.is_null());
  CaseData changed = d;
  changed.regions[0].contract.observable.set(9);
  changed.regions[0].prepared_key = compute_prepared_key(changed.regions[0]);
  std::vector<RegionResult> fourth;
  (void)test_case(changed, opt, "", &fourth);
  EXPECT_TRUE(fourth[0].cached.is_null());
  EXPECT_FALSE(build_fingerprint().empty());
  EXPECT_NE(build_fingerprint(), "unavailable");
  std::filesystem::remove_all(cache);
  std::filesystem::remove_all(cex);
}

TEST(IsaDiffCache, AnalysisCacheIsKeyedByArtifactsAndPlanOptions) {
  const std::string cache = temp_dir("analysis_cache");
  const Artifact a = artifact_from_words(k_shift_old, "old");
  const Artifact b = artifact_from_words(k_shift_new, "new");
  std::string error;
  bool hit = true;
  auto d1 = analyze_cached(a, b, {}, cache, hit, error);
  ASSERT_TRUE(d1) << error;
  EXPECT_FALSE(hit);
  auto d2 = analyze_cached(a, b, {}, cache, hit, error);
  ASSERT_TRUE(d2) << error;
  EXPECT_TRUE(hit);
  EXPECT_EQ(d2->to_json()["regions"].dump(-1), d1->to_json()["regions"].dump(-1));
  PlanOptions other;
  other.context_insts = 1;
  auto d3 = analyze_cached(a, b, other, cache, hit, error);
  ASSERT_TRUE(d3) << error;
  EXPECT_FALSE(hit);
  const Artifact c = artifact_from_words(k_shift_old, "new"); // identical sides
  auto d4 = analyze_cached(a, c, {}, cache, hit, error);
  ASSERT_TRUE(d4) << error;
  EXPECT_FALSE(hit);
  std::filesystem::remove_all(cache);
}

TEST(IsaDiffDiagnostics, EntryPrefixRegionsRunOnlyWhenRequestedAndStayOutOfTheVerdict) {
  CaseData plain = analyze(k_shift_old, k_shift_new);
  EXPECT_TRUE(plain.diagnostics.empty());
  PlanOptions opt;
  opt.diagnostic_entry_prefix = true;
  CaseData d = analyze(k_shift_old, k_shift_new, opt);
  ASSERT_EQ(d.diagnostics.size(), 1u);
  const RegionPair &diag = d.diagnostics[0];
  EXPECT_EQ(diag.id, "r0.diag-entry");
  EXPECT_EQ(diag.parent, "r0");
  ASSERT_EQ(diag.status, RegionStatus::Ready) << diag.reason;
  EXPECT_EQ(diag.old_side.begin, 0u);
  EXPECT_EQ(diag.contract.status, ContractStatus::Assumed);
  TestRun t = run(d, 200);
  ASSERT_EQ(t.results.size(), 1u); // the verdict covers r0 only
  ASSERT_EQ(t.report["diagnostics"].items().size(), 1u);
  const Json &dj = t.report["diagnostics"].items()[0];
  EXPECT_EQ(dj["test_status"].as_string(), "NO_MISMATCH_FOUND");
  EXPECT_GT(dj["counts"]["valid_pairs"].as_uint(), 0u);
  // Round trip through case.json keeps the diagnostic regions.
  std::string error;
  auto back = CaseData::from_json(d.to_json(), error);
  ASSERT_TRUE(back) << error;
  EXPECT_EQ(back->diagnostics.size(), 1u);
  // A region behind control flow, or with SMEM in the prefix, is reported, not run.
  CaseData rename = analyze(k_rename_old, k_rename_new, opt);
  bool not_entry = false;
  for (const auto &r : rename.diagnostics)
    not_entry |= r.status == RegionStatus::Unsupported &&
                 r.reason.find("not in the kernel's entry block") != std::string::npos;
  EXPECT_TRUE(not_entry);
}

// ---------------------------------------------------------------------------
// VALU support: VGPRs are tracked per lane, inactive lanes are compared, MODE per side.

TEST(IsaDiffValu, EquivalentVectorRewriteIsExecutedOnAllLanes) {
  CaseData d = analyze(k_valu_old, k_valu_new);
  ASSERT_EQ(d.regions.size(), 1u);
  const RegionPair &r = d.regions[0];
  ASSERT_EQ(r.status, RegionStatus::Ready) << r.reason;
  const RegionContract &k = r.contract;
  EXPECT_EQ(k.status, ContractStatus::Established); // VGPR composition from the kernel entry
  EXPECT_EQ(k.vgpr_count, 8u);                      // v0..v4 named, allocation granule 8
  EXPECT_TRUE(k.ventry_equal.test(0) && k.ventry_equal.test(1));
  EXPECT_TRUE(k.vobservable.test(3));          // stored after the region
  EXPECT_TRUE(k.vobservable.test(4));          // live through: must be preserved
  EXPECT_FALSE(k.vobservable.test(2));         // dead temporary
  EXPECT_TRUE(k.entry_equal.test(kLocExecLo)); // VALU reads EXEC
  TestRun t = run(d, 500);
  ASSERT_EQ(t.results.size(), 1u);
  const RegionResult &res = t.results[0];
  EXPECT_EQ(res.test_status, TestStatus::NoMismatchFound) << res.reason;
  EXPECT_EQ(t.overall(), "NO_MISMATCH_FOUND");
  EXPECT_EQ(res.harness_errors, 0u);
  for (const char *family : {"vector.lanes", "vector.cross", "vector.exec"})
    EXPECT_GT(res.boundary_families.count(family), 0u) << family;
  EXPECT_EQ(res.old_insts_executed, 2 * res.executed_pairs); // only the region ran
  EXPECT_EQ(res.new_insts_executed, 3 * res.executed_pairs);
}

TEST(IsaDiffValu, WrongShiftIsCaughtOnLanesAndReplays) {
  const std::string cex = temp_dir("valu_cex");
  CaseData d = analyze(k_valu_old, k_valuwrong_new);
  TestRun t = run(d, 200, 1, cex);
  ASSERT_EQ(t.results.size(), 1u);
  const RegionResult &res = t.results[0];
  ASSERT_EQ(res.test_status, TestStatus::LocalMismatch);
  ASSERT_FALSE(res.examples.empty());
  EXPECT_TRUE(res.examples[0].vlocs.test(3));
  EXPECT_NE(res.examples[0].detail.find("v3 lane"), std::string::npos) << res.examples[0].detail;
  ASSERT_FALSE(res.counterexample_files.empty());
  auto text = read_text_file(res.counterexample_files[0]);
  ASSERT_TRUE(text);
  std::string error;
  auto bundle = Json::parse(*text, &error);
  ASSERT_TRUE(bundle) << error;
  EXPECT_EQ((*bundle)["vgpr_locations"].as_string(), "v3");
  bool lane_entry = false;
  for (const Json &dj : (*bundle)["differences"].items())
    lane_entry |= dj["loc"].as_string() == "v3" && dj.has("lane");
  EXPECT_TRUE(lane_entry);
  EXPECT_TRUE((*bundle)["initial_state"]["old"]["vgprs"].has("v0"));
  const ReplayOutcome rep = replay_counterexample(*bundle, error);
  EXPECT_TRUE(rep.reproduced) << rep.summary << error;
  std::filesystem::remove_all(cex);
}

TEST(IsaDiffValu, InactiveLaneClobberIsDetected) {
  // The new side writes v3 with EXEC forced to all lanes: identical results on active
  // lanes, but lanes outside the original EXEC lose their value. At a kernel end nothing
  // reads them again (the store writes active lanes only): not observable.
  {
    CaseData end = analyze(k_valuclobber_old, k_valuclobber_new);
    ASSERT_EQ(end.regions.size(), 1u);
    EXPECT_TRUE(end.regions[0].contract.vactive_only.test(3));
    EXPECT_EQ(run(end, 300).results[0].test_status, TestStatus::NoMismatchFound);
  }
  // Here EXEC widens after the store and v3 is stored again on every lane: the inactive
  // lanes are observed, so all lanes are compared.
  CaseData d = analyze(k_valuclobberlater_old, k_valuclobberlater_new);
  ASSERT_EQ(d.regions.size(), 1u);
  ASSERT_EQ(d.regions[0].status, RegionStatus::Ready) << d.regions[0].reason;
  EXPECT_FALSE(d.regions[0].contract.vactive_only.test(3));
  TestRun t = run(d, 300);
  ASSERT_EQ(t.results[0].test_status, TestStatus::LocalMismatch);
  const Mismatch &m = t.results[0].examples.at(0);
  EXPECT_TRUE(m.vlocs.test(3));
  const uint64_t exec =
      uint64_t{m.init_old.get(kLocExecLo)} | (uint64_t{m.init_old.get(kLocExecHi)} << 32);
  EXPECT_NE(exec, ~uint64_t{0});
  for (unsigned lane = 0; lane < kWaveLanes; ++lane) {
    if (m.final_old.vget(3, lane) != m.final_new.vget(3, lane)) {
      EXPECT_FALSE(exec >> lane & 1u) << "lane " << lane << " was active";
    }
  }
  // With EXEC constrained to all lanes the rewrite is equivalent.
  CaseData full = d;
  for (unsigned l : {unsigned{kLocExecLo}, unsigned{kLocExecHi}}) {
    Constraint c;
    c.loc = l;
    c.known_one = 0xffffffffu;
    full.regions[0].contract.constraints.push_back(c);
  }
  TestRun tf = run(full, 300);
  EXPECT_EQ(tf.results[0].test_status, TestStatus::NoMismatchFound);
}

TEST(IsaDiffValu, DeadVectorTemporaryIsNotObservable) {
  CaseData d = analyze(k_valudead_old, k_valudead_new);
  ASSERT_EQ(d.regions.size(), 1u);
  const RegionContract &k = d.regions[0].contract;
  EXPECT_FALSE(k.vobservable.test(5));
  EXPECT_FALSE(k.vobservable.test(6));
  EXPECT_TRUE(k.vobservable.test(3));
  TestRun t = run(d, 300);
  EXPECT_EQ(t.results[0].test_status, TestStatus::NoMismatchFound) << t.results[0].reason;
}

TEST(IsaDiffValu, OutstandingLoadIntoAVectorInputLeavesTheContextUnresolved) {
  CaseData d = analyze(k_valupending_old, k_valupending_new);
  ASSERT_EQ(d.regions.size(), 1u);
  EXPECT_EQ(d.regions[0].contract.pending_effects, "CONFLICT");
  EXPECT_EQ(d.regions[0].contract.status, ContractStatus::Unresolved);
  TestRun t = run(d, 100);
  EXPECT_NE(t.overall(), "NO_MISMATCH_FOUND");
}

TEST(IsaDiffValu, ModeIsPerSideAndMattersForFloatConversion) {
  PlanOptions opt;
  opt.manual_region = "old:0-1,new:0-1"; // v_cvt_f32_f16 v1, v0 on both sides
  CaseData d = analyze(k_valucvt, k_valucvt, opt);
  ASSERT_EQ(d.regions.size(), 1u);
  ASSERT_EQ(d.regions[0].status, RegionStatus::Ready) << d.regions[0].reason;
  EXPECT_EQ(d.regions[0].contract.mode_old, kDefaultInitialMode);
  TestRun same = run(d, 200);
  EXPECT_EQ(same.results[0].test_status, TestStatus::NoMismatchFound);
  // The new side flushes f16 denormals: an f16 denormal input converts to 0 there.
  CaseData flush = d;
  flush.regions[0].contract.mode_new = kDefaultInitialMode & ~(3u << 6);
  TestRun t = run(flush, 200);
  ASSERT_EQ(t.results[0].test_status, TestStatus::LocalMismatch);
  const Mismatch &m = t.results[0].examples.at(0);
  EXPECT_TRUE(m.vlocs.test(1));
  EXPECT_EQ(m.init_new.frame.mode, kDefaultInitialMode & ~(3u << 6));
}

TEST(IsaDiffValu, RegisterCountMetadataIsInformationalOnlyWithinTheAllocation) {
  // k_valu_old/new name v0..v4. A descriptor with ACCUM_OFFSET 1 allocates 8 VGPRs.
  auto with_descriptor = [](std::span<const uint32_t> words, const char *label, uint32_t accum,
                            const char *vgpr_count) {
    Artifact a = artifact_from_words(words, label);
    a.descriptor.assign(sizeof(amdhsa::kernel_descriptor_t), 0);
    amdhsa::kernel_descriptor_t kd{};
    kd.compute_pgm_rsrc3 = accum; // ACCUM_OFFSET field [5:0]
    std::memcpy(a.descriptor.data(), &kd, sizeof(kd));
    a.metadata_parsed = true;
    a.metadata_sha256 = label;
    a.kernel_metadata = {{".name", "\"k\""}, {".vgpr_count", vgpr_count}};
    return a;
  };
  std::string error;
  auto fits = analyze_artifacts(with_descriptor(k_valu_old, "old", 1, "5"),
                                with_descriptor(k_valu_new, "new", 1, "4"), {}, error);
  ASSERT_TRUE(fits) << error;
  for (const auto &u : fits->unresolved)
    EXPECT_EQ(u.what.find(".vgpr_count"), std::string::npos) << u.what;
  bool informational = false;
  for (const auto &n : fits->notes)
    informational |= n.what.find(".vgpr_count") != std::string::npos && !n.semantic;
  EXPECT_TRUE(informational);
  EXPECT_EQ(fits->regions.at(0).contract.status, ContractStatus::Established);
  // ACCUM_OFFSET 0 allocates 4 VGPRs, but the kernels name v4: semantic, never hidden.
  auto beyond = analyze_artifacts(with_descriptor(k_valu_old, "old", 0, "5"),
                                  with_descriptor(k_valu_new, "new", 0, "4"), {}, error);
  ASSERT_TRUE(beyond) << error;
  bool reported = false;
  for (const auto &u : beyond->unresolved)
    reported |= u.what.find("beyond its descriptor allocation of 4 VGPRs") != std::string::npos;
  EXPECT_TRUE(reported);
}

TEST(IsaDiffMemory, StoredBytesAreTheObservableOutput) {
  // A renamed temporary stored through the same address: equal stored bytes.
  CaseData d = analyze(k_memst_old, k_memst_new);
  ASSERT_EQ(d.regions.size(), 1u);
  ASSERT_EQ(d.regions[0].status, RegionStatus::Ready) << d.regions[0].reason;
  EXPECT_TRUE(d.regions[0].contract.memory_writes);
  TestRun t = run(d, 300);
  EXPECT_EQ(t.results[0].test_status, TestStatus::NoMismatchFound) << t.results[0].reason;
  EXPECT_GT(t.results[0].valid_pairs, 300u);
  // The same value stored at offset 4: a memory mismatch, replayable.
  const std::string cex = temp_dir("mem_cex");
  CaseData w = analyze(k_memst_old, k_memstoff_new);
  TestRun tw = run(w, 100, 1, cex);
  ASSERT_EQ(tw.results[0].test_status, TestStatus::LocalMismatch);
  const Mismatch &m = tw.results[0].examples.at(0);
  EXPECT_TRUE(m.memory);
  EXPECT_NE(m.detail.find("memory difference"), std::string::npos) << m.detail;
  ASSERT_FALSE(tw.results[0].counterexample_files.empty());
  auto text = read_text_file(tw.results[0].counterexample_files[0]);
  ASSERT_TRUE(text);
  std::string error;
  auto bundle = Json::parse(*text, &error);
  ASSERT_TRUE(bundle) << error;
  EXPECT_TRUE((*bundle)["memory_differs"].as_bool());
  EXPECT_FALSE((*bundle)["memory_differences"].items().empty());
  const ReplayOutcome rep = replay_counterexample(*bundle, error);
  EXPECT_TRUE(rep.reproduced) << rep.summary << error;
  std::filesystem::remove_all(cex);
}

TEST(IsaDiffMemory, LoadsReadTheSameInputOnBothSides) {
  CaseData d = analyze(k_memld_old, k_memld_new);
  ASSERT_EQ(d.regions.size(), 1u);
  ASSERT_EQ(d.regions[0].status, RegionStatus::Ready) << d.regions[0].reason;
  EXPECT_TRUE(d.regions[0].contract.memory_reads);
  EXPECT_TRUE(d.regions[0].contract.memory_writes);
  TestRun t = run(d, 300);
  ASSERT_EQ(t.results[0].test_status, TestStatus::NoMismatchFound) << t.results[0].reason;
  EXPECT_EQ(t.results[0].harness_errors, 0u);
  // The initial image is recorded in states: 64 lanes read different words.
  MachineState probe;
  probe.mem_seed = 1;
  EXPECT_NE(initial_memory_word(1, 100), initial_memory_word(1, 101));
}

TEST(IsaDiffMemory, VectorMemoryCompletesInOrder) {
  // vmcnt(1) after two loads retires the first one: reading its destination is fine.
  CaseData ok = analyze(k_vmorder_old, k_vmorder_new);
  ASSERT_EQ(ok.regions.size(), 1u);
  EXPECT_NE(ok.regions[0].contract.pending_effects, "CONFLICT");
  EXPECT_EQ(run(ok, 200).results[0].test_status, TestStatus::NoMismatchFound);
  // Reading the second load's destination after vmcnt(1) may see a stale value.
  CaseData early = analyze(k_vmorder_old, k_vmearly_new);
  ASSERT_EQ(early.regions.size(), 1u);
  EXPECT_EQ(early.regions[0].contract.pending_effects, "CONFLICT");
  EXPECT_EQ(early.regions[0].contract.status, ContractStatus::Unresolved);
}

// ---------------------------------------------------------------------------
// Callable functions and comment-only inline asm.

TEST(IsaDiffInlineAsm, DefAndUseCommentsAreParsedIntoSlots) {
  const InlineAsmEffect d = parse_inline_asm("; def v[0:1]");
  ASSERT_EQ(d.defs.size(), 2u);
  EXPECT_TRUE(d.defs[0].vgpr);
  EXPECT_EQ(d.defs[1].index, 1u);
  EXPECT_TRUE(d.uses.empty());
  EXPECT_FALSE(d.opaque);

  const InlineAsmEffect u = parse_inline_asm("; use s[4:5], vcc, scc");
  ASSERT_EQ(u.uses.size(), 5u);
  EXPECT_EQ(u.uses[0], (AsmSlot{false, 4}));
  EXPECT_EQ(u.uses[2], (AsmSlot{false, kLocVccLo}));
  EXPECT_EQ(u.uses[4], (AsmSlot{false, kLocScc}));

  // Several lines: uses and defs accumulate.
  const InlineAsmEffect both = parse_inline_asm("; use v3\n; def s7");
  EXPECT_EQ(both.uses.size(), 1u);
  EXPECT_EQ(both.defs.size(), 1u);

  EXPECT_TRUE(parse_inline_asm("; def a0").untracked);
  EXPECT_TRUE(parse_inline_asm("; use alloca0 v1").opaque);
  EXPECT_TRUE(parse_inline_asm("; frobnicate v1").opaque); // names a register: not ignorable
  const InlineAsmEffect none = parse_inline_asm("; just a comment");
  EXPECT_FALSE(none.opaque);
  EXPECT_TRUE(none.uses.empty() && none.defs.empty());
  EXPECT_TRUE(parse_inline_asm("v_mov_b32 v0, 0").opaque); // not comment-only
}

TEST(IsaDiffInlineAsm, EffectsReadExecForVectorOperandsAndShapesIgnoreRenaming) {
  const Effects use = classify_inline_asm("; use v2");
  EXPECT_TRUE(use.p0);
  EXPECT_TRUE(use.vreads.test(2));
  EXPECT_EQ(use.reads, locs("exec_lo,exec_hi"));
  EXPECT_TRUE(use.implicit_reads.none());

  const Effects def = classify_inline_asm("; def v[4:5], s9");
  EXPECT_TRUE(def.vwrites.test(4) && def.vwrites.test(5));
  EXPECT_EQ(def.writes, locs("s9"));
  EXPECT_EQ(def.reads, locs("exec_lo,exec_hi")); // lanes outside EXEC keep their value

  const Effects agpr = classify_inline_asm("; def a[0:1]");
  EXPECT_FALSE(agpr.p0);
  EXPECT_FALSE(agpr.unknown_effect);
  const Effects odd = classify_inline_asm("; use v1-2");
  EXPECT_TRUE(odd.unknown_effect);

  EXPECT_EQ(inline_asm_shape("; def v[0:1]"), inline_asm_shape("; def v[2:3]"));
  EXPECT_NE(inline_asm_shape("; def v[0:1]"), inline_asm_shape("; def v[0:2]"));
  EXPECT_NE(inline_asm_shape("; def v0"), inline_asm_shape("; def s0"));
}

TEST(IsaDiffInlineAsm, ReturnObservablesFollowTheCallingConvention) {
  const LocSet l = return_observable_locs();
  EXPECT_TRUE(l.test(30) && l.test(31) && l.test(39) && l.test(48) && l.test(101));
  EXPECT_FALSE(l.test(0) || l.test(29) || l.test(40) || l.test(47));
  EXPECT_TRUE(l.test(kLocExecLo) && l.test(kLocExecHi));
  EXPECT_FALSE(l.test(kLocVccLo) || l.test(kLocM0) || l.test(kLocScc));
  const VSet v = return_observable_vgprs(2);
  EXPECT_TRUE(v.test(0) && v.test(1));
  EXPECT_FALSE(v.test(2) || v.test(39));
  EXPECT_TRUE(v.test(40) && v.test(47) && v.test(56) && v.test(255));
  EXPECT_FALSE(v.test(48) || v.test(55));
  EXPECT_FALSE(return_observable_vgprs(0).test(0));
}

TEST(IsaDiffInlineAsm, RunnerDefinesActiveLanesAndRecordsUses) {
  // def s[4:5]; def v1 (lanes in EXEC); use v1, s5.
  std::vector<RegionInst> insts = {
      {0x1000, {}, "inline asm: ; def s[4:5]", 0, "; def s[4:5]"},
      {0x1000, {}, "inline asm: ; def v1", 1, "; def v1"},
      {0x1000, {}, "inline asm: ; use v1, s5", 2, "; use v1, s5"},
  };
  std::string error;
  auto runner = RegionRunner::create(insts, "asm_runner", error, 8);
  ASSERT_TRUE(runner) << error;
  MachineState s;
  s.frame = runner->baseline_frame();
  s.resize_vgprs(8);
  for (unsigned lane = 0; lane < kWaveLanes; ++lane)
    s.vset(1, lane, 0xabc00000u + lane);
  s.set(kLocExecLo, 0x0000ffffu); // lanes 0..15
  s.set(kLocExecHi, 0);
  s.asm_seed = 0x1234567u;
  runner->load(s);
  const RunOutcome out = runner->run(16);
  ASSERT_EQ(out.status, RunStatus::Ok) << out.detail;
  EXPECT_EQ(out.executed, 3u);
  const MachineState f = runner->capture();
  EXPECT_EQ(f.get(4), inline_asm_word(s.asm_seed, 0, 0, 0));
  EXPECT_EQ(f.get(5), inline_asm_word(s.asm_seed, 0, 1, 0));
  EXPECT_EQ(f.vget(1, 3), inline_asm_word(s.asm_seed, 1, 0, 3));
  EXPECT_EQ(f.vget(1, 40), 0xabc00000u + 40); // outside EXEC: kept
  ASSERT_EQ(f.asm_uses.size(), 1u);
  const AsmUse &u = f.asm_uses[0];
  EXPECT_EQ(u.block, 2u);
  ASSERT_EQ(u.values.size(), 64u + 1u + 2u); // v1 lanes, s5, EXEC lo/hi
  EXPECT_EQ(u.values[3], inline_asm_word(s.asm_seed, 1, 0, 3));
  EXPECT_EQ(u.values[40], 0u); // lanes outside EXEC are not observed
  EXPECT_EQ(u.values[64], f.get(5));
  EXPECT_EQ(u.values[65], 0x0000ffffu);

  // The record round-trips through JSON (bundles replay it).
  auto back = MachineState::from_json(f.to_json(), error);
  ASSERT_TRUE(back) << error;
  EXPECT_EQ(back->asm_seed, f.asm_seed);
  EXPECT_EQ(back->asm_uses, f.asm_uses);
}

TEST(IsaDiffInlineAsm, RegionInstructionsKeepTheirBlockThroughJson) {
  RegionPair r;
  r.id = "r0";
  r.status = RegionStatus::Ready;
  r.old_side.insts = {{0x10, {}, "inline asm: ; def v0", 3, "; def v0"},
                      {0x10, {0xbf800000u}, "s_nop 0", -1, ""}};
  r.new_side.insts = r.old_side.insts;
  r.contract.canary = all_locs(); // a valid contract initialises every location
  std::string error;
  auto back = RegionPair::from_json(r.to_json(), error);
  ASSERT_TRUE(back) << error;
  ASSERT_EQ(back->old_side.insts.size(), 2u);
  EXPECT_EQ(back->old_side.insts[0].inline_asm, 3);
  EXPECT_EQ(back->old_side.insts[0].asm_text, "; def v0");
  EXPECT_TRUE(back->old_side.insts[0].words.empty());
  EXPECT_EQ(back->old_side.insts[1].inline_asm, -1);
  // Different asm text, same words: a different prepared region.
  RegionPair other = r;
  other.new_side.insts[0].asm_text = "; def v1";
  EXPECT_NE(compute_prepared_key(r), compute_prepared_key(other));
}

// ---------------------------------------------------------------------------
// NaN equivalence, poison analogues.

TEST(IsaDiffNan, ResultKindsFollowTheDestinationFormat) {
  EXPECT_EQ(fp_result_kind("v_add_f32_e32", "v_add_f32_e32 v0, v1, v2"), FpResult::F32);
  EXPECT_EQ(fp_result_kind("v_fma_f64", "v_fma_f64 v[0:1], ..."), FpResult::F64);
  EXPECT_EQ(fp_result_kind("v_add_f16_e32", "v_add_f16_e32 v0, v1, v2"), FpResult::F16Lo);
  EXPECT_EQ(fp_result_kind("v_add_f16_e64", "v_add_f16_e64 v0, v1, v2 op_sel:[0,0,1]"),
            FpResult::F16Hi);
  EXPECT_EQ(fp_result_kind("v_pk_add_f16", "v_pk_add_f16 v0, v1, v2"), FpResult::F16x2);
  EXPECT_EQ(fp_result_kind("v_pk_fma_f32", "v_pk_fma_f32 v[0:1], ..."), FpResult::F32);
  EXPECT_EQ(fp_result_kind("v_cvt_f16_f32_e32", "v_cvt_f16_f32_e32 v0, v1"), FpResult::F16Lo);
  EXPECT_EQ(fp_result_kind("v_cvt_i32_f32_e32", "v_cvt_i32_f32_e32 v0, v1"), FpResult::None);
  EXPECT_EQ(fp_result_kind("v_cmp_lt_f32_e32", "v_cmp_lt_f32_e32 vcc, v0, v1"), FpResult::None);
  EXPECT_EQ(fp_result_kind("v_xor_b32_e32", "v_xor_b32_e32 v0, v1, v2"), FpResult::None);
  EXPECT_EQ(fp_result_kind("v_mov_b32_e32", "v_mov_b32_e32 v0, v1"), FpResult::Move);
  EXPECT_EQ(fp_result_kind("v_mov_b32_dpp", "v_mov_b32_dpp v0, v1 row_shr:1"), FpResult::None);
  EXPECT_EQ(fp_result_kind("v_cndmask_b32_e32", "v_cndmask_b32_e32 v0, v1, v2, vcc"),
            FpResult::CndMask);
  EXPECT_EQ(fp_source_kind("v_cvt_i32_f32_e32"), FpResult::F32);
  EXPECT_EQ(fp_source_kind("v_and_b32_e32"), FpResult::None);
}

TEST(IsaDiffNan, ResidualKeepsEverythingButSameFormatNans) {
  // f32: two different NaNs from floating-point results are equal; a NaN and a number not.
  EXPECT_EQ(nan_residual(0x7fc00000u, 0xffc00001u, kFpF32, kFpF32), 0u);
  EXPECT_NE(nan_residual(0x7fc00000u, 0x3f800000u, kFpF32, kFpF32), 0u);
  // Only one side is a floating-point result: bitwise.
  EXPECT_NE(nan_residual(0x7fc00000u, 0xffc00001u, kFpF32, 0), 0u);
  // f16 halves: the low half differs only in NaN payload, the high half in value.
  EXPECT_EQ(nan_residual(0x3c007e00u, 0x3c007d01u, kFpF16Lo | kFpF16Hi, kFpF16Lo | kFpF16Hi), 0u);
  EXPECT_EQ(nan_residual(0x3c007e00u, 0x40007d01u, kFpF16Lo | kFpF16Hi, kFpF16Lo | kFpF16Hi),
            0x7c000000u);
  // bf16.
  EXPECT_EQ(nan_residual(0x00007fc0u, 0x0000ffc1u, kFpBF16Lo, kFpBF16Lo), 0u);
}

TEST(IsaDiffPoison, KindsParseAndMapToInstructions) {
  EXPECT_EQ(parse_poison_kinds("shift,nsw"), kPoisonShift | kPoisonNsw);
  EXPECT_FALSE(parse_poison_kinds("shift,bogus"));
  EXPECT_EQ(format_poison_kinds(*parse_poison_kinds("all")),
            "shift,fptoint,ctz,nsw,nuw,disjoint,exact,nnan,ninf");
  EXPECT_EQ(poison_check_kind("v_lshlrev_b32_e32"), kPoisonShift);
  EXPECT_EQ(poison_check_kind("s_lshr_b32"), kPoisonShift | kPoisonExact);
  EXPECT_TRUE(poison_check_kind("v_cvt_i32_f32_e32") & kPoisonFpToInt);
  EXPECT_EQ(poison_check_kind("v_ffbh_u32_e32"), kPoisonCtz);
  EXPECT_EQ(poison_check_kind("v_add_u32_e32"), kPoisonNsw | kPoisonNuw);
  EXPECT_EQ(poison_check_kind("v_add_co_u32_e32"), 0u); // a carry chain: not an i32 add
  EXPECT_EQ(poison_check_kind("v_or_b32_e32"), kPoisonDisjoint);
  EXPECT_EQ(poison_check_kind("v_mul_f32_e32"), kPoisonNnan | kPoisonNinf);
  EXPECT_EQ(poison_check_kind("v_mov_b32_e32"), 0u);
}

} // namespace
