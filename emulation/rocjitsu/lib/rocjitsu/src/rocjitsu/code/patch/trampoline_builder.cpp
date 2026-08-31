// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/trampoline_builder.h"

#include "rocjitsu/code/analysis/free_registers.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/builders/spill_builders.h"
#include "rocjitsu/code/patch/error_report.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <string>
#include <vector>

namespace rocjitsu {

namespace {

[[nodiscard]] bool check_size_and_words(const TrampolinePlan &plan, std::string *err) {
  if (plan.arch == ROCJITSU_CODE_ARCH_INVALID) {
    report(err, "trampoline plan: arch was not set");
    return false;
  }
  if (plan.original_size != 4 && plan.original_size != 8) {
    report(err, "trampoline plan: original_size must be 4 or 8");
    return false;
  }
  const size_t expected_words = plan.original_size / sizeof(uint32_t);
  if (plan.original_words.size() != expected_words) {
    report(err, "trampoline plan: original_words count does not match original_size");
    return false;
  }
  return true;
}

// Appends @p w to @p dst in host byte order. AMDGPU code objects are little-
// endian and rocjitsu only supports little-endian hosts (matches DBT's
// memcpy convention in binary_translator.cpp); if either invariant ever
// changes, this helper needs an explicit byte-swap.
void append_word(std::vector<uint8_t> &dst, uint32_t w) {
  uint8_t buf[sizeof(w)];
  std::memcpy(buf, &w, sizeof(w));
  dst.insert(dst.end(), buf, buf + sizeof(w));
}

// Spill/fill bracket. prologue saves each register before the call; epilogue
// restores each after the call and waits for the loads before the original runs.
// SGPRs bridge through one VGPR (writelane/readlane); since the single bridge is
// reused, each SGPR restore is a load/wait/readlane of its own.
struct SpillBracket {
  std::vector<uint32_t> prologue;
  std::vector<uint32_t> epilogue;
};

[[nodiscard]] SpillBracket build_spill_bracket(const std::vector<SpillSlot> &vgpr_spills,
                                               const std::vector<SpillSlot> &sgpr_spills,
                                               const std::vector<SpillSlot> &acc_spills,
                                               uint16_t bridge_vgpr, rj_code_arch_t arch) {
  constexpr uint16_t kUniformLane = 0; // An SGPR is uniform; one lane suffices.
  SpillBracket bracket;

  const bool has_vgpr = !vgpr_spills.empty();
  const bool has_sgpr = !sgpr_spills.empty();
  const bool has_acc = !acc_spills.empty();

  // The in-flight-load drain that protects to-be-spilled registers from a pending
  // pre-anchor load is emitted by the caller (emit_probe_call) at the very top of
  // the probe-call envelope, ahead of the special-state/temp writes, so it also
  // covers no-spill sites. The bracket therefore opens straight into the stores.

  // VGPRs: batch stores in the prologue.
  for (const SpillSlot &slot : vgpr_spills) {
    const std::vector<uint32_t> store = build_scratch_store_dword(slot.reg, slot.byte_offset, arch);
    bracket.prologue.insert(bracket.prologue.end(), store.begin(), store.end());
  }

  // AccVGPRs: like VGPRs but store/load directly out of the accumulator file via
  // the CDNA scratch `acc` bit -- no writelane/readlane bridge needed. Stores batch
  // with the VGPR stores; loads batch with the VGPR loads under the shared wait below.
  for (const SpillSlot &slot : acc_spills) {
    const auto store = build_scratch_store_dword(slot.reg, slot.byte_offset, arch, /*acc=*/true);
    bracket.prologue.insert(bracket.prologue.end(), store.begin(), store.end());
  }

  // Drain the VGPR stores before the writelane phase whenever both classes spill --
  // the only time the bridge could be a just-stored VGPR (WAR, see above).
  if (has_vgpr && has_sgpr)
    bracket.prologue.push_back(build_wait_stores_complete(arch));

  // SGPRs: writelane into the bridge then store.
  for (const SpillSlot &slot : sgpr_spills) {
    const std::array<uint32_t, 2> wl =
        build_v_writelane_b32(bridge_vgpr, slot.reg, kUniformLane, arch);
    bracket.prologue.insert(bracket.prologue.end(), wl.begin(), wl.end());
    const std::vector<uint32_t> store =
        build_scratch_store_dword(bridge_vgpr, slot.byte_offset, arch);
    bracket.prologue.insert(bracket.prologue.end(), store.begin(), store.end());
  }

  // The matching drain of the probe's in-flight loads is emitted by the caller
  // (emit_probe_call) immediately after the call returns, ahead of any restoration,
  // so it also covers no-spill sites. The bracket epilogue therefore opens straight
  // into the fills.

  // Epilogue: restore SGPRs first (load/wait/readlane each, since the single bridge
  // is reused), then the VGPRs and AccVGPRs, so a reused bridge's reload lands last.
  for (const SpillSlot &slot : sgpr_spills) {
    const std::vector<uint32_t> load =
        build_scratch_load_dword(bridge_vgpr, slot.byte_offset, arch);
    bracket.epilogue.insert(bracket.epilogue.end(), load.begin(), load.end());
    bracket.epilogue.push_back(build_wait_loads_complete(arch));
    const std::array<uint32_t, 2> rl =
        build_v_readlane_b32(slot.reg, bridge_vgpr, kUniformLane, arch);
    bracket.epilogue.insert(bracket.epilogue.end(), rl.begin(), rl.end());
  }
  for (const SpillSlot &slot : vgpr_spills) {
    const std::vector<uint32_t> load = build_scratch_load_dword(slot.reg, slot.byte_offset, arch);
    bracket.epilogue.insert(bracket.epilogue.end(), load.begin(), load.end());
  }
  for (const SpillSlot &slot : acc_spills) {
    const auto load = build_scratch_load_dword(slot.reg, slot.byte_offset, arch, /*acc=*/true);
    bracket.epilogue.insert(bracket.epilogue.end(), load.begin(), load.end());
  }
  if (has_vgpr || has_acc)
    bracket.epilogue.push_back(build_wait_loads_complete(arch));

  // Drain all scratch stores before the call: the store's async read of the source
  // register must finish before the probe clobbers it, which also orders each store
  // ahead of its reload. RDNA4 stores live on STORECNT, which s_wait_loadcnt misses.
  if (has_vgpr || has_sgpr || has_acc)
    bracket.prologue.push_back(build_wait_stores_complete(arch));

  return bracket;
}

} // namespace

std::optional<TrampolineBytes> TrampolineBuilder::build(const TrampolinePlan &plan,
                                                        std::string *error_out) {
  if (!check_size_and_words(plan, error_out))
    return std::nullopt;

  // Forward branch: from the anchor to the trampoline.
  const auto fwd = compute_sopp_branch_simm16(plan.anchor_offset, plan.trampoline_offset);
  if (!fwd) {
    report(error_out, "relocation trampoline forward branch exceeds s_branch simm16");
    return std::nullopt;
  }

  // Lay out trampoline body so we can compute the return branch offset. The
  // generic loops below handle any multi-item inline-asm shape; no reserve
  // hint because the per-item word counts aren't known up front and
  // vector::insert handles growth.
  std::vector<uint32_t> body;
  for (const InlineAsmItem &item : plan.before_items)
    body.insert(body.end(), item.words.begin(), item.words.end());
  if (plan.emit_original)
    body.insert(body.end(), plan.original_words.begin(), plan.original_words.end());
  for (const InlineAsmItem &item : plan.after_items)
    body.insert(body.end(), item.words.begin(), item.words.end());

  const uint64_t return_branch_pc = plan.trampoline_offset + body.size() * sizeof(uint32_t);
  const auto ret = compute_sopp_branch_simm16(return_branch_pc, plan.return_target);
  if (!ret) {
    report(error_out, "relocation trampoline return branch exceeds s_branch simm16");
    return std::nullopt;
  }

  TrampolineBytes out;
  out.patched_anchor_bytes.reserve(plan.original_size);
  append_word(out.patched_anchor_bytes, build_s_branch(*fwd, plan.arch));
  if (plan.original_size == 8)
    append_word(out.patched_anchor_bytes, build_s_nop(0, plan.arch));

  out.trampoline_words = std::move(body);
  out.trampoline_words.push_back(build_s_branch(*ret, plan.arch));
  return out;
}

bool TrampolineBuilder::plan_probe_call(TrampolinePlan &plan, ProbeCallingConvention cc,
                                        const RegisterSet &live_at_anchor,
                                        const RegisterSet &probe_body_clobbers,
                                        std::string *error_out) {
  // The link pair is whatever the probe's calling convention returns through,
  // so the call site and the probe body agree on one pair. An unknown
  // convention cannot be called.
  const std::optional<uint16_t> link_base = link_pair_for(cc);
  if (!link_base) {
    report(error_out, "probe-call resource planning: unknown probe calling convention; cannot "
                      "derive the return-link pair");
    return false;
  }
  const uint16_t kLinkPairBase = *link_base;

  // Reject if either lane of the link pair is live at the anchor; saving a live
  // link pair is deferred.
  if (any_in_range(live_at_anchor, RegClass::SGPR, kLinkPairBase, 2)) {
    report(error_out, ("probe-call resource planning: return-link pair s[" +
                       std::to_string(kLinkPairBase) + ":" + std::to_string(kLinkPairBase + 1) +
                       "] is live at the anchor; cannot yet save a live link pair")
                          .c_str());
    return false;
  }

  RegisterSet link_pair;
  link_pair.expand(RegisterRef{RegClass::SGPR, kLinkPairBase, 2});

  // Target/scc/special-state selection is capped at plan.kernel_sgpr_count so a
  // temp never lands past the patched kernel's actual .sgpr_count (a wider kernel
  // is not synthesized). The orchestrator sets the bound; it defaults to the
  // conservative cross-ISA allocatable limit.
  const uint32_t sgpr_bound =
      std::min<uint32_t>(plan.kernel_sgpr_count, REGISTER_SET_ALLOCATABLE_SGPRS);

  // Target-address pair: dead, even-aligned, and not the link pair. It is
  // read by s_swappc before the probe body runs, so it may overlap
  // probe_body_clobbers.
  const RegisterSet target_unavail = live_at_anchor | link_pair;
  const std::optional<uint16_t> target_pair = find_free_sgpr_pair(target_unavail, sgpr_bound);
  if (!target_pair) {
    report(error_out, "probe-call resource planning: no dead SGPR pair available for the probe "
                      "target address");
    return false;
  }

  RegisterSet target_pair_set;
  target_pair_set.expand(RegisterRef{RegClass::SGPR, *target_pair, 2});

  // Registers unavailable for the whole call: the live set, link + target pairs,
  // and the probe body clobbers (a temp lives across the call, so a probe clobber
  // would corrupt its saved value). The SCC and special-state temps are drawn
  // from this pool, each added back as it is picked so they never overlap.
  // TODO: allow for reuse of target_pair if unavailable
  RegisterSet reserved = target_unavail | target_pair_set | probe_body_clobbers;

  // SCC temp: one dead SGPR, only when preserving SCC.
  std::optional<uint16_t> scc_temp;
  if (plan.preserve_scc) {
    scc_temp = find_free_sgpr(reserved, sgpr_bound);
    if (!scc_temp) {
      report(error_out, "probe-call resource planning: no dead SGPR available for the SCC "
                        "preservation temp");
      return false;
    }
    reserved.expand(RegisterRef{RegClass::SGPR, *scc_temp, 1});
  }

  // Special-state temps: one dead SGPR (pair for the wave64 EXEC/VCC masks,
  // single for M0) per register the probe clobbers. Shared reserve path so EXEC,
  // VCC, and M0 differ only by their row below, not by a branch each.
  std::vector<SpecialStateSlot> special_saves;
  auto reserve_special = [&](bool requested, uint16_t operand, uint8_t width,
                             const char *name) -> bool {
    if (!requested)
      return true;
    const std::optional<uint16_t> temp = width == 2 ? find_free_sgpr_pair(reserved, sgpr_bound)
                                                    : find_free_sgpr(reserved, sgpr_bound);
    if (!temp) {
      report(error_out,
             (std::string("probe-call resource planning: no dead SGPR available for the ") + name +
              " preservation temp")
                 .c_str());
      return false;
    }
    reserved.expand(RegisterRef{RegClass::SGPR, *temp, width});
    special_saves.push_back(SpecialStateSlot{operand, *temp, width});
    return true;
  };
  // A spilled register is live at the anchor and clobbered by the probe (builder
  // clobbers are dead, so never in the spill set). Spilling forces EXEC=-1 around
  // the store/load, so EXEC must be saved even if the probe never touches it.
  const bool will_spill = live_at_anchor.intersects(probe_body_clobbers);

  // EXEC/VCC/M0 operand codes are resolved per-arch, but only when actually
  // reserving that register -- so a plan with no special-state saves (and no
  // spill) stays arch-agnostic, as the resource-planning tests rely on. EXEC also
  // rides this path when the site spills.
  const bool save_exec = plan.preserve_exec || will_spill;
  if (!reserve_special(save_exec, save_exec ? scalar_operand_exec_lo(plan.arch) : 0, 2, "EXEC") ||
      !reserve_special(plan.preserve_vcc, plan.preserve_vcc ? scalar_operand_vcc_lo(plan.arch) : 0,
                       2, "VCC") ||
      !reserve_special(plan.preserve_m0, plan.preserve_m0 ? scalar_operand_m0(plan.arch) : 0, 1,
                       "M0"))
    return false;

  // Word count is derived from the resource decisions, not a fixed envelope size.
  // Each add/addc uses the 32-bit literal form (instruction + literal word) so the
  // count is independent of the (layout-dependent) addend values.
  uint32_t before_words = 0;
  before_words += 1;     // s_getpc_b64
  before_words += 2 + 2; // s_add_u32 + literal, s_addc_u32 + literal
  before_words += 1;     // s_swappc_b64
  if (plan.preserve_scc)
    before_words += 2; // s_cselect_b32 (save) + s_cmp_lg_u32 (restore)
  // Each special-state register adds one s_mov save + one s_mov restore.
  before_words += static_cast<uint32_t>(special_saves.size()) * 2;
  // Spilling adds three EXEC toggles: widen before the stores, restore the anchor
  // mask before the call (probe runs under the anchor mask), re-widen before the loads.
  if (will_spill)
    before_words += 3;

  plan.is_probe_call = true;
  plan.link_pair_base = kLinkPairBase;
  plan.target_pair_base = *target_pair;
  if (scc_temp)
    plan.scc_temp = *scc_temp;
  plan.special_state_saves = std::move(special_saves);
  plan.before_word_count = before_words;

  plan.builder_clobbers = link_pair | target_pair_set;
  if (scc_temp)
    plan.builder_clobbers.expand(RegisterRef{RegClass::SGPR, *scc_temp, 1});
  for (const SpecialStateSlot &s : plan.special_state_saves)
    plan.builder_clobbers.expand(RegisterRef{RegClass::SGPR, s.temp_base, s.width});
  return true;
}

std::optional<TrampolineBytes> TrampolineBuilder::emit_probe_call(const TrampolinePlan &plan,
                                                                  std::string *error_out) {
  if (!plan.is_probe_call) {
    report(error_out, "emit_probe_call: plan is not a probe call (run plan_probe_call first)");
    return std::nullopt;
  }

  const uint16_t link = plan.link_pair_base;
  const uint16_t target_lo = plan.target_pair_base;
  const uint16_t target_hi = static_cast<uint16_t>(plan.target_pair_base + 1);
  // Literal-constant scalar source code; the 32-bit literal follows the word.
  constexpr uint16_t kLiteralConstant = 0xFF;

  const SpillBracket spill = build_spill_bracket(
      plan.vgpr_spills, plan.sgpr_spills, plan.acc_spills, plan.spill_bridge_vgpr, plan.arch);

  // Architecture-complete load drain, emitted once at each probe-call boundary
  // regardless of whether the site spills. Liveness does not model a pre-anchor
  // asynchronous load still targeting an otherwise-dead register, so such a load
  // could retire over a special-state/temp value we save below; a probe can
  // likewise return with an outstanding load that would escape the call boundary.
  // Emitted at the very top of the envelope (before any special-state/target/link
  // write) and again immediately after the call returns (before any restoration).
  const std::vector<uint32_t> boundary_drain = build_wait_all_loads_complete(plan.arch);

  std::vector<uint32_t> env;
  env.insert(env.end(), boundary_drain.begin(), boundary_drain.end());

  // The site spills iff there is anything to spill; this drives the EXEC full-mask
  // toggles only. EXEC save/restore is decided by special_state_saves membership.
  const bool full_mask_exec =
      !plan.vgpr_spills.empty() || !plan.sgpr_spills.empty() || !plan.acc_spills.empty();

  // SGPR pair holding the saved anchor EXEC (populated by the save loop below).
  // Reused to restore the anchor mask before the call; always present when spilling.
  uint16_t exec_temp = 0;
  bool exec_temp_found = false;
  for (const SpecialStateSlot &s : plan.special_state_saves)
    if (s.operand == scalar_operand_exec_lo(plan.arch)) {
      exec_temp = s.temp_base;
      exec_temp_found = true;
    }
  // full_mask_exec implies will_spill implies save_exec, so plan_probe_call must
  // have reserved an EXEC temp for any spilling site. Verify rather than trust the
  // default. Fail closed instead.
  if (full_mask_exec && !exec_temp_found) {
    report(error_out, "emit_probe_call: spilling site has no saved EXEC temp to restore the anchor "
                      "mask; plan_probe_call must reserve one when spilling");
    return std::nullopt;
  }

  // Special-state saves: copy each preserved EXEC/VCC/M0 into its dead temp. Before
  // the stores so EXEC is captured before we force it to -1. Plain s_mov, SCC-safe.
  for (const SpecialStateSlot &s : plan.special_state_saves)
    env.push_back(s.width == 2 ? build_s_mov_b64(s.temp_base, s.operand, plan.arch)
                               : build_s_mov_b32(s.temp_base, s.operand, plan.arch));

  // Full-mask the spill store so a probe that widens EXEC cannot leave inactive-
  // lane copies unsaved. EXEC was just saved and is restored after the loads.
  if (full_mask_exec)
    env.push_back(build_s_mov_b64(scalar_operand_exec_lo(plan.arch),
                                  scalar_inline_neg_one(plan.arch), plan.arch));

  // Spill saves: store each live+clobbered register before the call.
  env.insert(env.end(), spill.prologue.begin(), spill.prologue.end());

  // Restore the anchor EXEC before the call so the probe runs under the anchor mask,
  // not the full mask used to bracket the stores. The loads are re-widened after.
  if (full_mask_exec)
    env.push_back(build_s_mov_b64(scalar_operand_exec_lo(plan.arch), exec_temp, plan.arch));

  // SCC save (prologue): capture SCC into the temp without disturbing it. The
  // matching restore is emitted after the call but still before the relocated
  // original.
  if (plan.preserve_scc)
    env.push_back(build_s_cselect_b32(plan.scc_temp, scalar_positive_inline_u32(1),
                                      scalar_positive_inline_u32(0), plan.arch));

  // Target-address materialization. s_getpc_b64 writes the runtime VA of the
  // *next* instruction (the s_add_u32 below) into the target pair; the
  // build-time delta to the probe body is then folded in via the 64-bit add
  // chain (s_add_u32 sets carry -> SCC, s_addc_u32 consumes it). Both sides are
  // .text-relative and share the load base, so the delta is a pure layout
  // distance. The adds always use the literal form so the word count is
  // independent of the (layout-dependent) delta value (see before_word_count).
  const size_t getpc_index = env.size();
  env.push_back(build_s_getpc_b64(target_lo, plan.arch));
  const uint64_t va_after_getpc =
      plan.trampoline_offset + static_cast<uint64_t>(getpc_index + 1) * sizeof(uint32_t);
  const uint64_t delta = static_cast<uint64_t>(static_cast<int64_t>(plan.probe_target_offset) -
                                               static_cast<int64_t>(va_after_getpc));
  env.push_back(build_s_add_u32(target_lo, target_lo, kLiteralConstant, plan.arch));
  env.push_back(static_cast<uint32_t>(delta & 0xFFFFFFFFu));
  env.push_back(build_s_addc_u32(target_hi, target_hi, kLiteralConstant, plan.arch));
  env.push_back(static_cast<uint32_t>(delta >> 32));

  // The call: writes the return PC into the cc-derived link pair, jumps to the
  // materialized target. The probe returns here via s_setpc_b64 of the same pair.
  env.push_back(build_s_swappc_b64(link, target_lo, plan.arch));

  // Drain the probe's in-flight loads immediately on return, before any restoration
  // (SCC/EXEC, spill fills, special-state) or the relocated host code -- a probe may
  // return with an unwaited load still targeting a register we are about to restore
  // or that the host reads. Mirrors the boundary drain at the envelope top.
  env.insert(env.end(), boundary_drain.begin(), boundary_drain.end());

  // SCC restore (epilogue): set SCC from the saved temp before the relocated
  // original runs.
  if (plan.preserve_scc)
    env.push_back(build_s_cmp_lg_u32(plan.scc_temp, scalar_positive_inline_u32(0), plan.arch));

  // Full-mask the spill load to match the store (the probe may have changed EXEC).
  if (full_mask_exec)
    env.push_back(build_s_mov_b64(scalar_operand_exec_lo(plan.arch),
                                  scalar_inline_neg_one(plan.arch), plan.arch));

  // Spill fills: reload each saved register after the call and wait for the loads.
  env.insert(env.end(), spill.epilogue.begin(), spill.epilogue.end());

  // Special-state restores: copy each temp back into its register. After the spill
  // loads so EXEC is restored to its anchor mask only once the full-mask loads are
  // done; all run before the relocated original.
  for (const SpecialStateSlot &s : plan.special_state_saves)
    env.push_back(s.width == 2 ? build_s_mov_b64(s.operand, s.temp_base, plan.arch)
                               : build_s_mov_b32(s.operand, s.temp_base, plan.arch));

  // Plan/emit drift guard: the planner committed to this many envelope words and
  // the orchestrator sized the layout around it. A mismatch means the two
  // disagree about the envelope shape. before_word_count is the arch-agnostic
  // envelope; the spill bracket and the two boundary drains are accounted
  // separately since their sizes are arch- and site-specific.
  if (env.size() != plan.before_word_count + spill.prologue.size() + spill.epilogue.size() +
                        2 * boundary_drain.size()) {
    report(error_out, "emit_probe_call: synthesized envelope word count does not match the planned "
                      "before_word_count");
    return std::nullopt;
  }

  // Hand the synthesized envelope to build() for layout and branch math so the
  // SOPP range checks are shared with the inline path.
  TrampolinePlan emit_plan = plan;
  emit_plan.before_items.assign(1, InlineAsmItem{std::move(env)});
  emit_plan.after_items.clear();
  emit_plan.emit_original = true;
  return build(emit_plan, error_out);
}

} // namespace rocjitsu
