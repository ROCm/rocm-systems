#include "rocjitsu_fuzzer/afl_dbi_plan.h"

#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <string_view>

namespace {

void check(bool condition, const char *message) {
  if (!condition) {
    fprintf(stderr, "edge_plan_unit: %s\n", message);
    std::exit(1);
  }
}

} // namespace

int main() {
  using namespace rocjitsu::fuzzer::afl_dbi;

  check(build_rdna4_s_load_b64_s0_word0(/*sdst_lo=*/100) == 0xF4003900u,
        "state pointer s_load word0 encoding changed");
  check(build_rdna4_s_load_b64_s0_word0(/*sdst_lo=*/4) == 0xF4002100u,
        "entry counter s_load word0 encoding changed");
  check(build_rdna4_sop1(/*op=*/0, /*sdst=*/102, kScalarExecLo) == 0xBEE6007Eu,
        "s_mov_b32 builder encoding changed");
  check(build_rdna4_sop1(/*op=*/0x19, /*sdst=*/103, /*ssrc0=*/102) == 0xBEE71966u,
        "s_bcnt1_i32_b64 builder encoding changed");
  check(build_rdna4_sop2(/*op=*/22, /*sdst=*/104, kScalarExecLo, kScalarVccLo) ==
            0x8B686A7Eu,
        "s_and_b32 builder encoding changed");
  check(build_rdna4_sop2(/*op=*/0x30, /*sdst=*/8,
                         amdgpu_positive_inline_const(1),
                         amdgpu_positive_inline_const(0)) == 0x98088081u,
        "s_cselect_b32 builder encoding changed");
  check(build_rdna4_sopc(/*op=*/7, /*ssrc0=*/8,
                         amdgpu_positive_inline_const(0)) == 0xBF078008u,
        "s_cmp_lg_u32 builder encoding changed");
  check(build_rdna4_sopp(/*op=*/0x25, /*imm=*/17) == 0xBFA50011u,
        "s_cbranch_execz builder encoding changed");
  check(build_rdna4_v_lshlrev_b32(/*vdst=*/121, amdgpu_positive_inline_const(2),
                                  /*vsrc1=*/120) == 0x30F2F082u,
        "v_lshlrev builder encoding changed");
  check(build_rdna4_v_add_nc_u32(/*vdst=*/121, amdgpu_vgpr_src(122),
                                 /*vsrc1=*/121) == 0x4AF2F37Au,
        "v_add_nc builder encoding changed");
  check(build_rdna4_v_xor_b32(/*vdst=*/122, amdgpu_vgpr_src(122),
                              /*vsrc1=*/123) == 0x3AF4F77Au,
        "v_xor builder encoding changed");
  check(build_rdna4_v_and_b32(/*vdst=*/122, kScalarLiteral, /*vsrc1=*/122) ==
            0x36F4F4FFu,
        "v_and builder encoding changed");
  check(build_rdna4_v_mbcnt_lo_u32_b32_word0(/*vdst=*/121) == 0xD71F0079u,
        "v_mbcnt word0 builder encoding changed");
  check(build_rdna4_v_mbcnt_lo_u32_b32_word1(/*ssrc0=*/103) == 0x00010067u,
        "v_mbcnt word1 builder encoding changed");
  check(build_rdna4_v_mbcnt_hi_u32_b32_word0(/*vdst=*/121) == 0xD7200079u,
        "v_mbcnt_hi word0 builder encoding changed");
  check(build_rdna4_v_mbcnt_hi_u32_b32_word1(/*src0=*/103, amdgpu_vgpr_src(121)) ==
            0x0002F267u,
        "v_mbcnt_hi word1 builder encoding changed");
  check(build_rdna4_v_cmp_eq_u32_vcc(amdgpu_positive_inline_const(0),
                                     /*vsrc1=*/121) == 0x7C94F280u,
        "v_cmp builder encoding changed");
  check(build_rdna4_v_cmp_eq_u32_sdst_word0(/*sdst=*/104) == 0xD44A0068u,
        "v_cmp e64 sdst word0 builder encoding changed");
  check(build_rdna4_v_cmp_eq_u32_sdst_word1(amdgpu_positive_inline_const(0),
                                            amdgpu_vgpr_src(121)) == 0x0002F280u,
        "v_cmp e64 sdst word1 builder encoding changed");
  check(build_rdna4_v_cmpx_eq_u32_exec(amdgpu_positive_inline_const(0),
                                       /*vsrc1=*/121) == 0x7D94F280u,
        "v_cmpx builder encoding changed");

  auto entry_probe = rdna4_counter_probe(kEntryCounterSlot);
  check(entry_probe.has_value(), "entry probe should be encodable");
  check(entry_probe->size() == 30, "entry probe word count changed");
  check((*entry_probe)[9] == build_rdna4_v_mbcnt_hi_u32_b32_word0(/*vdst=*/2),
        "entry probe should compute wave64 high-lane ranks");
  check((*entry_probe)[18] == 0xBFA50008u, "entry probe first-active skip changed");

  auto edge_probe = rdna4_counter_probe(3);
  check(edge_probe.has_value(), "edge probe should be encodable");
  check(edge_probe->size() == 30, "edge probe word count changed");
  check((*edge_probe)[7] == build_rdna4_v_mbcnt_hi_u32_b32_word0(/*vdst=*/121),
        "edge probe should compute wave64 high-lane ranks");
  check((*edge_probe)[16] == 0xBFA5000Au, "edge probe first-active skip changed");
  check((*edge_probe)[21] ==
            build_rdna4_v_mov_b32(/*vdst=*/120, amdgpu_positive_inline_const(12)),
        "edge probe slot byte offset move changed");
  auto literal_slot_edge_probe = rdna4_counter_probe(kMaxInlineCounterSlot + 1);
  check(literal_slot_edge_probe.has_value(),
        "edge probe should materialize literal slot offsets");
  check(literal_slot_edge_probe->size() == edge_probe->size() + 1,
        "literal slot edge probe should add one literal word");
  check((*literal_slot_edge_probe)[21] == build_rdna4_v_mov_b32(/*vdst=*/120, kScalarLiteral),
        "literal slot edge probe should use literal address offset");
  check((*literal_slot_edge_probe)[22] == (kMaxInlineCounterSlot + 1) * sizeof(uint32_t),
        "literal slot edge probe offset changed");

  auto edge_entry_probe = rdna4_edge_entry_probe();
  check(edge_entry_probe.size() == 30, "edge entry probe word count changed");
  check(edge_entry_probe[0] == 0xF4003900u, "edge entry probe should initialize s[100:101]");

  Rdna4ProbeRegisters custom_edge_entry_regs;
  custom_edge_entry_regs.state_sgpr = 20;
  custom_edge_entry_regs.saved_exec_sgpr = 22;
  custom_edge_entry_regs.tmp0_sgpr = 24;
  custom_edge_entry_regs.tmp1_sgpr = 24;
  custom_edge_entry_regs.scc_sgpr = 26;
  custom_edge_entry_regs.workitem_vgpr = 30;
  custom_edge_entry_regs.tmp0_vgpr = 31;
  custom_edge_entry_regs.tmp1_vgpr = 32;
  custom_edge_entry_regs.tmp2_vgpr = 33;
  auto custom_edge_entry =
      rdna4_edge_entry_probe(ROCJITSU_CODE_ARCH_RDNA4, custom_edge_entry_regs);
  check(custom_edge_entry[0] == build_rdna4_s_load_b64_s0_word0(/*sdst_lo=*/20),
        "custom edge entry state load SGPR changed");
  check(custom_edge_entry[5] ==
            build_rdna4_v_mbcnt_lo_u32_b32_word0(/*vdst=*/31),
        "custom edge entry low-lane mbcnt destination changed");
  check(custom_edge_entry[9] ==
            build_rdna4_v_mbcnt_hi_u32_b32_word0(/*vdst=*/31),
        "custom edge entry high-lane mbcnt destination changed");
  check(custom_edge_entry[10] ==
            build_rdna4_v_mbcnt_hi_u32_b32_word1(/*src0=*/24, amdgpu_vgpr_src(31)),
        "custom edge entry high-lane mbcnt source changed");
  check(custom_edge_entry[11] == build_rdna4_v_cmp_eq_u32_sdst_word0(/*sdst=*/24),
        "custom edge entry e64 compare destination changed");
  check(custom_edge_entry[12] ==
            build_rdna4_v_cmp_eq_u32_sdst_word1(amdgpu_positive_inline_const(0),
                                                amdgpu_vgpr_src(31)),
        "custom edge entry e64 compare source changed");
  check(custom_edge_entry[16] == build_s_mov_b32_word(/*sdst=*/kScalarExecLo, /*src=*/24,
                                                      ROCJITSU_CODE_ARCH_RDNA4),
        "custom edge entry EXEC_LO mask changed");
  check(custom_edge_entry[17] == build_s_mov_b32_word(/*sdst=*/kScalarExecHi, /*src=*/25,
                                                      ROCJITSU_CODE_ARCH_RDNA4),
        "custom edge entry EXEC_HI mask changed");
  check(custom_edge_entry[19] == build_rdna4_sop1(/*op=*/0x19, /*sdst=*/24, /*ssrc0=*/22),
        "custom edge entry active-lane count changed");
  check(custom_edge_entry[27] == build_rdna4_sop2(/*op=*/24, /*sdst=*/kScalarExecLo,
                                                  /*ssrc0=*/kScalarExecLo, /*ssrc1=*/22),
        "custom edge entry EXEC_LO restore changed");
  check(custom_edge_entry[28] == build_rdna4_sop2(/*op=*/24, /*sdst=*/kScalarExecHi,
                                                  /*ssrc0=*/kScalarExecHi, /*ssrc1=*/23),
        "custom edge entry EXEC_HI restore changed");
  check(custom_edge_entry[6] == build_rdna4_v_mbcnt_lo_u32_b32_word1(/*ssrc0=*/24),
        "custom edge entry mbcnt SGPR source changed");
  const ProbeRegisterRequirements custom_edge_requirements =
      probe_register_requirements(custom_edge_entry_regs);
  check(custom_edge_requirements.sgprs == 26,
        "custom probe SGPR high-water mark changed");
  check(custom_edge_requirements.vgprs == 34,
        "custom probe VGPR high-water mark changed");
  check(kAdaptiveFixedCounterBranchEdgeFallbackLimit ==
            kFixedCounterBranchEdgeFallbackMapBudget,
        "adaptive fixed branch fallback should be capped by the fixed counter map");
  check(adaptive_fixed_counter_branch_edge_fallback_budget(/*candidate_edges=*/309) == 309,
        "adaptive fixed branch fallback should derive from candidate edges");
  check(adaptive_fixed_counter_branch_edge_fallback_budget(
            kFixedCounterBranchEdgeFallbackMapBudget + 1) ==
            kFixedCounterBranchEdgeFallbackMapBudget,
        "adaptive fixed branch fallback should not exceed the fixed counter map");
  constexpr uint32_t site_derived_edge_budget =
      previous_bb_branch_site_derived_edge_budget(kAdaptivePreviousBbBranchSiteLimit);
  check(site_derived_edge_budget == std::numeric_limits<uint32_t>::max(),
        "previous-BB candidate-derived logical edge budget changed");
  constexpr PreviousBbBranchAggregateBudget candidate_budget =
      make_previous_bb_branch_aggregate_budget(
          /*candidate_edges=*/101, /*candidate_sites=*/54,
          /*edge_budget=*/101,
          /*site_budget=*/54,
          /*edge_budget_auto=*/true, /*site_budget_auto=*/true,
          /*previous_bb_policy=*/true,
          /*fixed_counter_fallback_enabled=*/true,
          /*fixed_counter_fallback_budget=*/
          adaptive_fixed_counter_branch_edge_fallback_budget(101));
  check(candidate_budget.edge_over_budget == 0,
        "candidate-derived edge budget should cover the split previous-BB case");
  check(candidate_budget.site_over_budget == 0,
        "split previous-BB case should fit the candidate-derived writer budget");
  check(candidate_budget.limit_kind() ==
            PreviousBbBranchAggregateLimitKind::None,
        "split previous-BB case should fit the auto candidate budget");
  constexpr PreviousBbBranchAggregateBudget mt128_like_candidate_budget =
      make_previous_bb_branch_aggregate_budget(
          /*candidate_edges=*/309, /*candidate_sites=*/173,
          /*edge_budget=*/309,
          /*site_budget=*/173,
          /*edge_budget_auto=*/true, /*site_budget_auto=*/true,
          /*previous_bb_policy=*/true,
          /*fixed_counter_fallback_enabled=*/true,
          /*fixed_counter_fallback_budget=*/
          adaptive_fixed_counter_branch_edge_fallback_budget(309));
  check(mt128_like_candidate_budget.edge_over_budget == 0 &&
            mt128_like_candidate_budget.site_over_budget == 0,
        "MT128-like previous-BB candidate budget should be uncapped by default");
  check(mt128_like_candidate_budget.limit_kind() ==
            PreviousBbBranchAggregateLimitKind::None,
        "MT128-like previous-BB candidate budget should report no aggregate cap");
  constexpr PreviousBbBranchAggregateBudget configured_over_cap_budget =
      make_previous_bb_branch_aggregate_budget(
          /*candidate_edges=*/309, /*candidate_sites=*/173,
          /*edge_budget=*/256,
          /*site_budget=*/128,
          /*edge_budget_auto=*/true, /*site_budget_auto=*/true,
          /*previous_bb_policy=*/true,
          /*fixed_counter_fallback_enabled=*/true,
          /*fixed_counter_fallback_budget=*/
          adaptive_fixed_counter_branch_edge_fallback_budget(309));
  check(configured_over_cap_budget.edge_over_budget == 53,
        "previous-BB aggregate edge overage changed");
  check(configured_over_cap_budget.site_over_budget == 45,
        "previous-BB aggregate site overage changed for configured cap");
  check(configured_over_cap_budget.limit_kind() ==
            PreviousBbBranchAggregateLimitKind::EdgeAndSiteCap,
        "previous-BB aggregate limit kind changed");
  check(std::string_view(previous_bb_branch_aggregate_limit_kind_name(
            configured_over_cap_budget.limit_kind())) == "edge-and-site-cap",
        "previous-BB aggregate limit kind name changed");
  check(!configured_over_cap_budget.edge_budget_exhausted(/*selected_edges=*/254,
                                                          /*site_edge_count=*/2),
        "previous-BB edge budget should allow the final in-budget conditional site");
  check(configured_over_cap_budget.edge_budget_exhausted(/*selected_edges=*/255,
                                                         /*site_edge_count=*/2),
        "previous-BB edge budget should reject over-budget conditional sites");
  check(configured_over_cap_budget.previous_bb_site_budget_exhausted(
            /*selected_sites=*/128),
        "previous-BB site budget should reject the first over-budget writer");
  check(configured_over_cap_budget.fixed_counter_fallback_available(
            /*selected_fallback_edges=*/307, /*site_edge_count=*/2),
        "fixed fallback should allow the final in-budget conditional site");
  check(!configured_over_cap_budget.fixed_counter_fallback_available(
            /*selected_fallback_edges=*/308, /*site_edge_count=*/2),
        "fixed fallback should reject over-budget conditional sites");
  constexpr PreviousBbBranchAggregateBudget fixed_policy_budget =
      make_previous_bb_branch_aggregate_budget(
          /*candidate_edges=*/101, /*candidate_sites=*/54,
          /*edge_budget=*/kAdaptiveBranchEdgeSiteLimit,
          /*site_budget=*/kAdaptivePreviousBbBranchSiteLimit,
          /*edge_budget_auto=*/true, /*site_budget_auto=*/true,
          /*previous_bb_policy=*/false,
          /*fixed_counter_fallback_enabled=*/true,
          /*fixed_counter_fallback_budget=*/
          adaptive_fixed_counter_branch_edge_fallback_budget(101));
  check(fixed_policy_budget.limit_kind() ==
            PreviousBbBranchAggregateLimitKind::NotPreviousBbPolicy,
        "fixed branch policy should not report a previous-BB aggregate cap");
  check(fixed_policy_budget.edge_over_budget == 0 &&
            fixed_policy_budget.site_over_budget == 0,
        "fixed branch policy should not report previous-BB overage");

  auto previous_bb_entry_probe =
      rdna4_previous_bb_edge_probe(/*bb_id=*/0x101u, /*load_state_base=*/true);
  check(previous_bb_entry_probe.size() == 55,
        "previous-BB entry probe word count changed");
  check(previous_bb_entry_probe[0] == 0xF4003900u,
        "previous-BB entry probe should initialize s[100:101]");
  check(previous_bb_entry_probe[1] ==
            build_rdna4_s_load_b64_s100_s0_word1(/*byte_offset=*/0),
        "previous-BB entry probe state load offset changed");
  check(previous_bb_entry_probe[8] ==
            build_rdna4_sop2(/*op=*/0x30, /*sdst=*/108,
                             amdgpu_positive_inline_const(1),
                             amdgpu_positive_inline_const(0)),
        "previous-BB probe SCC save changed");
  check(previous_bb_entry_probe[13] == kPreviousBbByteOffset,
        "previous-BB probe state offset literal changed");
  check(previous_bb_entry_probe[21] == 0x101u,
        "previous-BB probe BB id literal changed");
  check(previous_bb_entry_probe[24] == kHashEdgeSlotMask,
        "previous-BB probe edge mask literal changed");
  check(previous_bb_entry_probe[27] == previous_bb_after(0x101u),
        "previous-BB probe next previous-BB literal changed");
  check(previous_bb_entry_probe[36] == build_rdna4_v_mbcnt_hi_u32_b32_word0(/*vdst=*/121),
        "previous-BB probe should compute wave64 high-lane ranks");
  check(previous_bb_entry_probe[37] ==
            build_rdna4_v_mbcnt_hi_u32_b32_word1(/*src0=*/104, amdgpu_vgpr_src(121)),
        "previous-BB probe high-lane mbcnt source changed");
  check(previous_bb_entry_probe[38] == build_rdna4_v_cmp_eq_u32_sdst_word0(/*sdst=*/104),
        "previous-BB probe EXEC compare destination changed");
  check(previous_bb_entry_probe[45] == 0xBFA50005u,
        "previous-BB probe first-active-lane skip changed");
  check(previous_bb_entry_probe[54] ==
            build_rdna4_sopc(/*op=*/7, /*ssrc0=*/108,
                             amdgpu_positive_inline_const(0)),
        "previous-BB probe SCC restore changed");

  auto previous_bb_block_probe =
      rdna4_previous_bb_edge_probe(/*bb_id=*/0x101u, /*load_state_base=*/false);
  check(previous_bb_block_probe.size() == 53,
        "previous-BB block probe word count changed");
  check(previous_bb_block_probe[0] != 0xF4003900u,
        "block probe should reuse the entry-loaded state pointer");

  auto hidden_state_entry_probe =
      rdna4_previous_bb_edge_probe(/*bb_id=*/0x101u, /*load_state_base=*/true,
                                   /*state_pointer_kernarg_offset=*/288);
  check(hidden_state_entry_probe[1] ==
            build_rdna4_s_load_b64_s100_s0_word1(/*byte_offset=*/288),
        "hidden state pointer load offset should be encodable");

  auto literal_state_entry_probe =
      rdna4_previous_bb_edge_probe_with_state_pointer(/*bb_id=*/0x101u,
                                                      /*state_pointer=*/0x1234567887654321ull);
  check(literal_state_entry_probe.size() == 57,
        "literal state pointer entry probe word count changed");
  check(literal_state_entry_probe[0] == build_rdna4_s_mov_b32_literal_word(100),
        "literal state pointer low SGPR move changed");
  check(literal_state_entry_probe[1] == 0x87654321u,
        "literal state pointer low literal changed");
  check(literal_state_entry_probe[2] == build_rdna4_s_mov_b32_literal_word(101),
        "literal state pointer high SGPR move changed");
  check(literal_state_entry_probe[3] == 0x12345678u,
        "literal state pointer high literal changed");

  Rdna4ProbeRegisters previous_bb_regs;
  previous_bb_regs.state_sgpr = 20;
  previous_bb_regs.saved_exec_sgpr = 22;
  previous_bb_regs.tmp0_sgpr = 24;
  previous_bb_regs.tmp1_sgpr = 24;
  previous_bb_regs.scc_sgpr = 26;
  previous_bb_regs.workitem_vgpr = 30;
  previous_bb_regs.tmp0_vgpr = 31;
  previous_bb_regs.tmp1_vgpr = 32;
  previous_bb_regs.tmp2_vgpr = 33;
  auto custom_previous_bb_probe =
      rdna4_previous_bb_edge_probe(/*bb_id=*/0x101u, /*load_state_base=*/true,
                                   /*state_pointer_kernarg_offset=*/288,
                                   ROCJITSU_CODE_ARCH_RDNA4, previous_bb_regs);
  check(custom_previous_bb_probe.size() == 55,
        "custom previous-BB probe word count changed");
  check(custom_previous_bb_probe[0] == build_rdna4_s_load_b64_s0_word0(/*sdst_lo=*/20),
        "custom previous-BB state load SGPR changed");
  check(custom_previous_bb_probe[11] ==
            build_rdna4_v_lshlrev_b32(/*vdst=*/31, amdgpu_positive_inline_const(2),
                                      /*vsrc1=*/30),
        "custom previous-BB workitem scale changed");
  check(custom_previous_bb_probe[14] ==
            build_rdna4_v_add_nc_u32(/*vdst=*/31, amdgpu_vgpr_src(32), /*vsrc1=*/31),
        "custom previous-BB previous-state address add changed");
  check(custom_previous_bb_probe[32] == build_rdna4_v_mbcnt_lo_u32_b32_word0(/*vdst=*/31),
        "custom previous-BB mbcnt destination changed");
  check(custom_previous_bb_probe[33] == build_rdna4_v_mbcnt_lo_u32_b32_word1(/*ssrc0=*/24),
        "custom previous-BB mbcnt SGPR source changed");
  check(custom_previous_bb_probe[36] == build_rdna4_v_mbcnt_hi_u32_b32_word0(/*vdst=*/31),
        "custom previous-BB high-lane mbcnt destination changed");
  check(custom_previous_bb_probe[37] ==
            build_rdna4_v_mbcnt_hi_u32_b32_word1(/*src0=*/24, amdgpu_vgpr_src(31)),
        "custom previous-BB high-lane mbcnt source changed");
  check(custom_previous_bb_probe[38] == build_rdna4_v_cmp_eq_u32_sdst_word0(/*sdst=*/24),
        "custom previous-BB first-active-lane compare changed");

  auto flagless_counter_probe = rdna4_flagless_counter_probe(/*slot=*/0x123u);
  check(flagless_counter_probe.has_value(), "flagless counter probe should be encodable");
  check(flagless_counter_probe->size() == 8,
        "flagless counter probe word count changed");
  check((*flagless_counter_probe)[0] == build_rdna4_v_mov_b32(/*vdst=*/120, kScalarLiteral),
        "flagless counter probe should use a literal slot offset");
  check((*flagless_counter_probe)[1] == 0x123u * sizeof(uint32_t),
        "flagless counter probe slot offset literal changed");
  check((*flagless_counter_probe)[2] ==
            build_rdna4_v_mov_b32(/*vdst=*/121, amdgpu_positive_inline_const(1)),
        "flagless counter probe increment move changed");
  check(!rdna4_flagless_counter_probe(kCoverageSlots).has_value(),
        "flagless counter probe should reject out-of-map slots");

  Rdna4ProbeRegisters custom_regs;
  custom_regs.state_sgpr = 20;
  custom_regs.tmp0_vgpr = 30;
  custom_regs.workitem_vgpr = 31;
  auto custom_flagless_counter =
      rdna4_flagless_counter_probe(/*slot=*/0x123u, ROCJITSU_CODE_ARCH_RDNA4, custom_regs);
  check(custom_flagless_counter.has_value(),
        "custom-register flagless counter probe should be encodable");
  check((*custom_flagless_counter)[0] ==
            build_rdna4_v_mov_b32(/*vdst=*/31, kScalarLiteral),
        "custom flagless counter should use custom address VGPR");
  check((*custom_flagless_counter)[2] ==
            build_rdna4_v_mov_b32(/*vdst=*/30, amdgpu_positive_inline_const(1)),
        "custom flagless counter should use custom data VGPR");
  auto custom_flagless_state =
      rdna4_flagless_counter_probe_with_state_pointer(/*slot=*/0x123u,
                                                      /*state_pointer=*/0x1234567887654321ull,
                                                      ROCJITSU_CODE_ARCH_RDNA4, custom_regs);
  check(custom_flagless_state.has_value(),
        "custom-register state-pointer flagless counter should be encodable");
  check((*custom_flagless_state)[0] == build_rdna4_s_mov_b32_literal_word(20),
        "custom flagless state pointer low SGPR move changed");
  check((*custom_flagless_state)[2] == build_rdna4_s_mov_b32_literal_word(21),
        "custom flagless state pointer high SGPR move changed");
  check((*custom_flagless_state)[4] ==
            build_rdna4_v_mov_b32(/*vdst=*/31, kScalarLiteral),
        "custom flagless state probe should preserve custom address VGPR");

  InstrumentationPlanOptions default_options;
  check(default_options.block_entry_site_limit == 12,
        "default block-entry site limit changed");
  check(default_options.branch_edge_site_limit == 8,
        "default branch-edge site limit changed");
  check(kAdaptiveBranchEdgeSiteLimit == 100,
        "adaptive previous-BB baseline branch budget changed");
  check(kAdaptivePreviousBbBranchSiteLimit == std::numeric_limits<uint32_t>::max(),
        "adaptive previous-BB branch-site writer budget should be candidate-derived");
  check(kAdaptiveBranchEdgeSiteLimit <=
            previous_bb_branch_site_derived_edge_budget(
                kAdaptivePreviousBbBranchSiteLimit),
        "site-derived previous-BB branch edge budget should cover the baseline");
  check(default_options.block_entry_slot_policy == EdgeSlotPolicyKind::PreviousBbHash,
        "block entries should use hashed slots by default");
  check(default_options.branch_terminator_slot_policy == EdgeSlotPolicyKind::PreviousBbHash,
        "branch terminators should prefer hashed slots by default");
  check(!default_options.fixed_edge_slots,
        "fixed edge slots should stay opt-in by default");
  check(!default_options.branch_edge_slots,
        "branch edge slots should stay opt-in by default");
  check(default_options.liveness_registers,
        "liveness register planning should stay enabled by default");
  check(!default_options.require_liveness_registers,
        "strict liveness register requirements should stay opt-in by default");
  check(default_options.fixed_counter_fallback_for_branch_liveness,
        "adaptive branch fallback should stay enabled by default");
  check(!default_options.self_contained_edge_probes,
        "edge probes should not assume self-contained state loads by default");

  check(register_count_to_granulated(/*registers=*/1, /*granularity=*/8) == 0,
        "one register should round up to the first granulated bucket");
  const ProbeRegisterRequirements default_requirements = probe_register_requirements({});
  check(default_requirements.sgprs == 106,
        "default probe SGPR high-water mark changed");
  check(default_requirements.vgprs == 124,
        "default probe VGPR high-water mark changed");
  const ProbeRegisterRequirements default_flagless_requirements =
      flagless_counter_probe_register_requirements({});
  check(default_flagless_requirements.sgprs == 102,
        "flagless counter SGPR high-water mark changed");
  check(default_flagless_requirements.vgprs == 122,
        "flagless counter VGPR high-water mark changed");
  const ProbeRegisterRequirements default_first_active_requirements =
      first_active_counter_probe_register_requirements({});
  check(default_first_active_requirements.sgprs == 106,
        "first-active counter SGPR high-water mark changed");
  check(default_first_active_requirements.vgprs == 122,
        "first-active counter VGPR high-water mark changed");
  const ProbeRegisterRequirements default_previous_bb_requirements =
      previous_bb_probe_register_requirements({});
  check(default_previous_bb_requirements.sgprs == 109,
        "previous-BB SGPR high-water mark changed");
  check(default_previous_bb_requirements.vgprs == 124,
        "previous-BB VGPR high-water mark changed");
  check(register_count_to_granulated(default_requirements.vgprs, /*granularity=*/8) == 15,
        "wave32 VGPR probe requirement granulation changed");
  check(register_count_to_granulated(default_requirements.vgprs, /*granularity=*/4) == 30,
        "wave64 VGPR probe requirement granulation changed");
  check(register_count_to_granulated(default_requirements.sgprs, /*granularity=*/8) == 13,
        "SGPR probe requirement granulation changed");
  check(granulated_to_register_count(/*granulated=*/0, /*granularity=*/8) == 8,
        "first granulated register bucket should report one granule");
  check(granulated_to_register_count(/*granulated=*/15, /*granularity=*/8) == 128,
        "wave32 VGPR granulated count conversion changed");
  check(granulated_to_register_count(/*granulated=*/31, /*granularity=*/4) == 128,
        "wave64 VGPR granulated count conversion changed");

  FixedEdgeSlotAllocator slots;
  check(slots.reserve(1).value_or(0) == kFirstEdgeCounterSlot,
        "fixed slot allocator should start at first edge counter slot");
  check(slots.reserve(2).value_or(0) == kFirstEdgeCounterSlot + 1,
        "fixed slot allocator should reserve adjacent slot ranges");
  check(slots.used() == 3, "fixed slot allocator used count changed");
  check(slots.reserve(kMaxInlineCounterSlot).has_value(),
        "fixed slot allocator should allow literal-offset slots beyond inline immediates");
  check(slots.reserve(kMaxFixedCounterSlot - slots.used()).has_value(),
        "fixed slot allocator should fill the map up to the last device counter");
  check(!slots.reserve(1).has_value(),
        "fixed slot allocator should reject ranges beyond the device counter map");
  FixedEdgeSlotAllocator stable_slots;
  const uint32_t fixed_slot_budget = FixedEdgeSlotAllocator::fixed_slot_budget();
  auto stable_slot = stable_slots.reserve_stable(0x1234u);
  check(stable_slot.has_value(), "stable fixed slot assignment should succeed");
  check(stable_slot->primary_slot ==
            FixedEdgeSlotAllocator::fixed_counter_slot_for_id(0x1234u),
        "stable fixed slot should derive from the edge id");
  auto colliding_stable_slot = stable_slots.reserve_stable(0x1234u + fixed_slot_budget);
  check(colliding_stable_slot.has_value(),
        "stable fixed slot collision should still assign a slot");
  check(colliding_stable_slot->primary_slot == stable_slot->primary_slot,
        "fixed ids in the same bucket should map to the same stable slot");
  check(colliding_stable_slot->collisions == 1,
        "stable fixed slot collision should be reported");
  auto stable_pair = stable_slots.reserve_stable(0x99u, 0x99u + fixed_slot_budget,
                                                 /*count=*/2);
  check(stable_pair.has_value(), "stable two-edge assignment should succeed");
  check(stable_pair->primary_slot != stable_pair->secondary_slot,
        "taken and fallthrough fixed counters should not share one slot within a site");
  check(std::string_view(edge_patch_kind_name(EdgePatchKind::BlockEntry)) == "block",
        "block edge kind name changed");
  check(std::string_view(edge_patch_kind_name(EdgePatchKind::ConditionalBlockEntry)) ==
            "cond-block",
        "conditional block edge kind name changed");
  check(std::string_view(edge_patch_kind_name(EdgePatchKind::ConditionalBranchTerminator)) ==
            "cond-branch",
        "conditional branch edge kind name changed");
  check(std::string_view(edge_slot_policy_name(EdgeSlotPolicyKind::PreviousBbHash)) ==
            "previous-bb-hash",
        "hashed edge slot policy name changed");

  EdgeSite fixed_branch_site;
  fixed_branch_site.kind = EdgePatchKind::BranchTerminator;
  fixed_branch_site.slot_policy = EdgeSlotPolicyKind::FixedCounter;
  fixed_branch_site.self_contained_probe = true;
  ProbeRegisterRequirements fixed_branch_requirements =
      edge_site_probe_register_requirements(fixed_branch_site);
  check(fixed_branch_requirements.sgprs == default_flagless_requirements.sgprs,
        "fixed branch edge should use flagless SGPR requirements");
  check(fixed_branch_requirements.vgprs == default_flagless_requirements.vgprs,
        "fixed branch edge should use flagless VGPR requirements");

  EdgeSite hashed_branch_site;
  hashed_branch_site.kind = EdgePatchKind::BranchTerminator;
  hashed_branch_site.slot_policy = EdgeSlotPolicyKind::PreviousBbHash;
  hashed_branch_site.self_contained_probe = true;
  ProbeRegisterRequirements hashed_branch_requirements =
      edge_site_probe_register_requirements(hashed_branch_site);
  check(hashed_branch_requirements.sgprs == default_previous_bb_requirements.sgprs,
        "hashed branch edge should use previous-BB SGPR requirements");
  check(hashed_branch_requirements.vgprs == default_previous_bb_requirements.vgprs,
        "hashed branch edge should use previous-BB VGPR requirements");

  EdgeSlotPolicy hashed_policy(EdgeSlotPolicyKind::PreviousBbHash,
                               EdgeSlotPolicyKind::PreviousBbHash);
  auto hashed_block = hashed_policy.assign_block_entry();
  check(hashed_block.has_value(), "hashed block-entry assignment should succeed");
  check(hashed_block->policy == EdgeSlotPolicyKind::PreviousBbHash,
        "hashed block-entry assignment policy changed");
  check(hashed_block->primary_slot == 0,
        "hashed block-entry assignment should not consume fixed slots");
  auto hashed_summary = hashed_policy.summary();
  check(hashed_summary.hashed_edge_sites == 1,
        "hashed block-entry summary changed");
  check(hashed_summary.inline_slots_reserved == 0,
        "hashed assignment should not reserve inline slots");
  auto fallback_branch = hashed_policy.assign_branch_terminator_fallback(
      EdgeSlotPolicyKind::FixedCounter, /*edge_count=*/1);
  check(fallback_branch.has_value(),
        "adaptive fixed branch fallback should reserve a fixed slot");
  check(fallback_branch->policy == EdgeSlotPolicyKind::FixedCounter,
        "adaptive fallback policy changed");
  hashed_summary = hashed_policy.summary();
  check(hashed_summary.hashed_edge_sites == 1,
        "adaptive fallback should not erase prior hashed accounting");
  check(hashed_summary.fixed_edge_sites == 1,
        "adaptive fallback fixed-site accounting changed");

  EdgeSlotPolicy fixed_policy(EdgeSlotPolicyKind::FixedCounter,
                              EdgeSlotPolicyKind::FixedCounter);
  auto fixed_branch = fixed_policy.assign_branch_terminator(/*edge_count=*/2);
  check(fixed_branch.has_value(), "fixed branch assignment should succeed");
  check(fixed_branch->primary_slot == kFirstEdgeCounterSlot,
        "fixed branch primary slot changed");
  check(fixed_branch->secondary_slot == kFirstEdgeCounterSlot + 1,
        "fixed branch fallthrough slot changed");
  auto fixed_summary = fixed_policy.summary();
  check(fixed_summary.fixed_edge_sites == 1, "fixed branch site summary changed");
  check(fixed_summary.fixed_slot_requests == 2,
        "fixed branch request count changed");
  check(fixed_summary.fixed_slots_reserved == 2,
        "fixed branch reserved count changed");
  check(fixed_summary.inline_slot_requests == 2,
        "legacy inline branch request alias changed");
  check(fixed_summary.inline_slots_reserved == 2,
        "legacy inline branch reserved alias changed");

  EdgeSlotPolicy stable_fixed_policy(EdgeSlotPolicyKind::FixedCounter,
                                     EdgeSlotPolicyKind::FixedCounter);
  const uint32_t stable_edge_id = 0x4000u;
  auto stable_fixed_branch = stable_fixed_policy.assign_branch_terminator(
      /*edge_count=*/2, stable_edge_id, stable_edge_id + fixed_slot_budget);
  check(stable_fixed_branch.has_value(),
        "stable fixed branch assignment should succeed");
  check(stable_fixed_branch->primary_slot ==
            FixedEdgeSlotAllocator::fixed_counter_slot_for_id(stable_edge_id),
        "stable fixed branch primary slot changed");
  check(stable_fixed_branch->secondary_slot != stable_fixed_branch->primary_slot,
        "stable fixed branch fallthrough slot should be disambiguated");
  auto stable_fixed_collision = stable_fixed_policy.assign_branch_terminator(
      /*edge_count=*/1, stable_edge_id);
  check(stable_fixed_collision.has_value(),
        "colliding stable fixed assignment should succeed");
  check(stable_fixed_collision->primary_slot == stable_fixed_branch->primary_slot,
        "colliding stable fixed assignment should keep deterministic slot");
  check(stable_fixed_collision->fixed_slot_collisions == 1,
        "colliding stable fixed assignment should report the collision");
  check(stable_fixed_policy.summary().fixed_slot_collisions == 1,
        "stable fixed slot collision summary changed");

  FixedEdgeSlotTracker shared_fixed_slots;
  EdgeSlotPolicy first_kernel_fixed_policy(EdgeSlotPolicyKind::FixedCounter,
                                           EdgeSlotPolicyKind::FixedCounter,
                                           &shared_fixed_slots);
  EdgeSlotPolicy second_kernel_fixed_policy(EdgeSlotPolicyKind::FixedCounter,
                                            EdgeSlotPolicyKind::FixedCounter,
                                            &shared_fixed_slots);
  check(first_kernel_fixed_policy.assign_branch_terminator(/*edge_count=*/1,
                                                           stable_edge_id)
            .has_value(),
        "first shared stable fixed assignment should succeed");
  auto cross_kernel_collision = second_kernel_fixed_policy.assign_branch_terminator(
      /*edge_count=*/1, stable_edge_id + fixed_slot_budget);
  check(cross_kernel_collision.has_value(),
        "cross-kernel stable fixed collision should still assign a slot");
  check(cross_kernel_collision->fixed_slot_collisions == 1,
        "shared fixed slot tracker should report cross-kernel collisions");
  check(second_kernel_fixed_policy.summary().fixed_slot_collisions == 1,
        "cross-kernel fixed slot collision summary changed");

  EdgeSlotPolicy exhausting_policy(EdgeSlotPolicyKind::FixedCounter,
                                   EdgeSlotPolicyKind::FixedCounter);
  check(exhausting_policy.assign_branch_terminator(kMaxFixedCounterSlot).has_value(),
        "full fixed slot reservation should fit");
  check(!exhausting_policy.assign_branch_terminator(/*edge_count=*/1).has_value(),
        "fixed slot reservation should report exhaustion");
  check(exhausting_policy.summary().fixed_slot_exhaustions == 1,
        "fixed slot exhaustion summary changed");
  check(exhausting_policy.summary().inline_slot_exhaustions == 1,
        "legacy inline slot exhaustion alias changed");

  std::vector<uint8_t> text_with_caves(128, 0xff);
  std::fill(text_with_caves.begin() + 16, text_with_caves.begin() + 80, 0);
  std::fill(text_with_caves.begin() + 96, text_with_caves.end(), 0);
  LocalTextCaveAllocator local_caves(text_with_caves);
  LocalTextCaveSummary cave_summary = local_caves.summary();
  check(cave_summary.range_count == 2, "local cave range count changed");
  check(cave_summary.total_bytes == 96, "local cave byte budget changed");
  check(cave_summary.largest_range_bytes == 64, "largest local cave changed");
  auto branch_reachable_cave =
      local_caves.allocate(/*ideal_offset=*/20, /*size_bytes=*/32,
                           [](uint64_t offset) { return offset >= 32; });
  check(branch_reachable_cave.has_value(), "branch-reachable local cave should be found");
  check(*branch_reachable_cave == 32, "unexpected branch-reachable local cave");
  cave_summary = local_caves.summary();
  check(cave_summary.range_count == 1, "local cave range count after allocation changed");
  check(cave_summary.total_bytes == 32, "local cave byte budget after allocation changed");
  check(!local_caves.allocate(/*ideal_offset=*/0, /*size_bytes=*/128,
                              [](uint64_t) { return true; })
             .has_value(),
        "oversized local cave allocation should fail");

  LocalTextCaveAllocator interior_only_caves(text_with_caves);
  auto interior_only_cave =
      interior_only_caves.allocate(/*ideal_offset=*/20, /*size_bytes=*/32,
                                   [](uint64_t offset) { return offset == 32; });
  check(interior_only_cave.has_value(), "interior-only local cave should be found");
  check(*interior_only_cave == 32, "unexpected interior-only local cave");

  constexpr uint32_t kEntryBb = 0x101u;
  constexpr uint32_t kEvenPathBb = 0x201u;
  constexpr uint32_t kOddPathBb = 0x401u;
  constexpr uint32_t kJoinBb = 0x801u;
  constexpr uint64_t kAllLanes = 0b1111u;
  constexpr uint64_t kEvenLanes = 0b0101u;
  constexpr uint64_t kOddLanes = 0b1010u;

  std::vector<uint32_t> previous_bb(4, 0);
  std::vector<uint32_t> counters(kCoverageSlots, 0);
  model_record_edge(kEntryBb, kAllLanes, previous_bb, counters);
  for (uint32_t previous : previous_bb) {
    check(previous == previous_bb_after(kEntryBb),
          "entry should update every active lane's previous BB");
  }

  model_record_edge(kEvenPathBb, kEvenLanes, previous_bb, counters);
  check(previous_bb[0] == previous_bb_after(kEvenPathBb),
        "even lane should take even-path previous BB");
  check(previous_bb[2] == previous_bb_after(kEvenPathBb),
        "second even lane should take even-path previous BB");
  check(previous_bb[1] == previous_bb_after(kEntryBb),
        "masked-off odd lane should keep entry previous BB");
  check(previous_bb[3] == previous_bb_after(kEntryBb),
        "second masked-off odd lane should keep entry previous BB");

  model_record_edge(kOddPathBb, kOddLanes, previous_bb, counters);
  check(previous_bb[1] == previous_bb_after(kOddPathBb),
        "odd lane should take odd-path previous BB");
  check(previous_bb[3] == previous_bb_after(kOddPathBb),
        "second odd lane should take odd-path previous BB");

  const std::vector<uint32_t> previous_at_join = previous_bb;
  const uint32_t even_join_slot =
      hashed_edge_counter_slot(previous_bb_after(kEvenPathBb), kJoinBb);
  const uint32_t odd_join_slot =
      hashed_edge_counter_slot(previous_bb_after(kOddPathBb), kJoinBb);
  check(even_join_slot != odd_join_slot, "divergent join edges should hash apart");

  model_record_edge(kJoinBb, kAllLanes, previous_bb, counters,
                    EdgeCounterPolicy::FirstActiveLane);
  check(counters[even_join_slot] == 1,
        "first-active-lane policy should count the first join edge");
  check(counters[odd_join_slot] == 0,
        "first-active-lane policy intentionally collapses other join edges");

  std::vector<uint32_t> extension_previous = previous_at_join;
  std::vector<uint32_t> extension_counters(kCoverageSlots, 0);
  model_record_edge(kJoinBb, kAllLanes, extension_previous, extension_counters,
                    EdgeCounterPolicy::DistinctPreviousBb);
  check(extension_counters[even_join_slot] == 1,
        "distinct-previous-BB policy should count the even join edge");
  check(extension_counters[odd_join_slot] == 1,
        "distinct-previous-BB policy should count the odd join edge");

  constexpr uint64_t kHighLaneOnly = 1ull << 40;
  std::vector<uint32_t> wave64_previous(64, 0);
  std::vector<uint32_t> wave64_counters(kCoverageSlots, 0);
  const uint32_t high_lane_slot = hashed_edge_counter_slot(/*previous_bb=*/0, kOddPathBb);
  model_record_edge(kOddPathBb, kHighLaneOnly, wave64_previous, wave64_counters,
                    EdgeCounterPolicy::FirstActiveLane);
  check(wave64_previous[40] == previous_bb_after(kOddPathBb),
        "high-lane-only wave64 path should update that lane's previous BB");
  check(wave64_previous[39] == 0 && wave64_previous[41] == 0,
        "inactive wave64 neighbor lanes should keep their previous BB");
  check(wave64_counters[high_lane_slot] == 1,
        "first-active-lane policy should count high-lane-only wave64 masks");

  check(counter_slot_byte_offset(kMaxInlineCounterSlot).has_value(),
        "last inline slot should be encodable");
  check(counter_slot_uses_inline_offset(kMaxInlineCounterSlot),
        "last inline slot should use inline offset encoding");
  check(counter_slot_byte_offset(kMaxInlineCounterSlot + 1).has_value(),
        "slot beyond inline limit should still be encodable with a literal");
  check(!counter_slot_uses_inline_offset(kMaxInlineCounterSlot + 1),
        "slot beyond inline limit should not claim inline offset encoding");
  check(!counter_slot_byte_offset(kCoverageSlots).has_value(),
        "slot beyond the device counter map should be rejected");

  check(s_branch_offset_dwords(100, 104).value_or(1) == 0,
        "branch to next instruction should have zero offset");
  check(s_branch_offset_dwords(100, 100).value_or(0) == -1,
        "branch to self should have -1 offset");
  check(!s_branch_offset_dwords(100, 102).has_value(),
        "unaligned branch target should be rejected");
  check(!s_branch_offset_dwords(0, static_cast<uint64_t>(1) << 32).has_value(),
        "out-of-range branch target should be rejected");

  return 0;
}
