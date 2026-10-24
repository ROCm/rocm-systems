// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/trampoline_builder.h"

#include "rocjitsu/code/analysis/free_registers.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/builders/spill_builders.h"
#include "rocjitsu/code/builders/vector_builders.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/code/patch/error_report.h"
#include "rocjitsu/isa/isa_traits.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <string>
#include <vector>

namespace rocjitsu {

namespace {

bool validate_register_saves(const TrampolinePlan &plan, const RegisterSet &live,
                             const RegisterSet &probe_clobbers, RegisterSet &destinations,
                             std::string *error) {
  RegisterSet sources;
  for (const RegisterSave &save : plan.register_saves) {
    if (save.cls != RegClass::SGPR && save.cls != RegClass::VGPR) {
      report(error, "register preservation requires an ordinary SGPR or VGPR");
      return false;
    }
    const uint32_t bound = save.cls == RegClass::SGPR
                               ? ordinary_sgpr_bound(plan.arch, plan.kernel_sgpr_count)
                               : std::min(plan.kernel_vgpr_count, 256u);
    const RegisterRef destination{save.cls, save.destination, 1};
    if (save.source >= bound || save.destination >= bound || save.source == save.destination ||
        destinations.intersects(destination) || live.intersects(destination) ||
        probe_clobbers.intersects(destination)) {
      report(error, "register preservation destination is unavailable or outside the allocation");
      return false;
    }
    destinations.expand(destination);
    sources.expand(RegisterRef{save.cls, save.source, 1});
  }
  if (sources.intersects(destinations)) {
    report(error, "register preservation destinations overlap source registers");
    return false;
  }
  return true;
}

bool plan_argument_moves(std::span<const ProbeArgValue> args, const ProbeAbi &abi,
                         uint32_t vgpr_bound, const RegisterSet &live,
                         std::vector<ProbeArgumentMove> &moves, std::optional<uint16_t> &temp,
                         std::string *error) {
  vgpr_bound = std::min(vgpr_bound, 256u);
  std::vector<ProbeArgumentMove> pending;
  RegisterSet unavailable = live | arg_registers(abi) | probe_argument_registers(args);
  for (size_t index = 0; index < args.size(); ++index) {
    const ProbeArgValue &arg = args[index];
    if (arg.source != ProbeArgSource::AnchorVgpr)
      continue;
    const uint16_t destination = abi.arg_vgpr_base + index;
    if (arg.register_index >= vgpr_bound || destination >= vgpr_bound) {
      report(error, "guest VGPR argument exceeds the kernel's ordinary VGPR allocation");
      return false;
    }
    if (destination != arg.register_index)
      pending.push_back({destination, arg.register_index});
  }
  while (!pending.empty()) {
    const std::vector<ProbeArgumentMove>::iterator ready =
        std::find_if(pending.begin(), pending.end(), [&](const auto &move) {
          return std::none_of(pending.begin(), pending.end(),
                              [&](const auto &other) { return other.source == move.destination; });
        });
    if (ready != pending.end()) {
      moves.push_back(*ready);
      pending.erase(ready);
      continue;
    }
    if (!temp)
      temp = find_free_run(unavailable, RegClass::VGPR, 1, 0, 1, vgpr_bound);
    if (!temp || *temp >= vgpr_bound ||
        unavailable.intersects(RegisterRef{RegClass::VGPR, *temp, 1})) {
      report(error, "guest VGPR argument copy cycle has no dead temporary VGPR");
      return false;
    }
    const uint16_t source = pending.front().source;
    moves.push_back({*temp, source});
    for (auto &move : pending)
      if (move.source == source)
        move.source = *temp;
  }
  return true;
}

[[nodiscard]] bool check_size_and_words(const TrampolinePlan &plan, std::string *err) {
  if (plan.arch == ROCJITSU_CODE_ARCH_INVALID) {
    report(err, "trampoline plan: arch was not set");
    return false;
  }
  if (plan.original_size == 0 || plan.original_size % 4 != 0 ||
      ((plan.emit_original || plan.patch_anchor) && plan.original_size != 4 &&
       plan.original_size != 8)) {
    report(err, "trampoline plan: original_size must be 4 or 8 when patching or relocating");
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

// GFX9 (CDNA1-4) does not interlock a VALU write of an SGPR against a later
// vector-memory read or lane select of it, nor a SALU write of M0 against a later
// M0 reader; software spaces them with wait states (s_nop N supplies N+1, any other
// instruction 1).
[[nodiscard]] constexpr bool needs_restore_wait_states(rj_code_arch_t arch) {
  return arch_is_cdna_4_or_lower(arch);
}

// After the v_readlane_b32 that restores an SGPR: a vector-memory read of it
// (resource, soffset, saddr) needs 5, which also covers a v_readlane/v_writelane
// lane select (4) and a gfx94x VALU read (2). LLVM GCNHazardRecognizer's
// checkVMEMHazards, checkRWLaneHazards and checkVALUHazards.
constexpr uint32_t kSgprRestoreWaitStates = 5;

// After the s_mov_b32 that restores M0: s_movrel*, s_sendmsg, GDS, LDS DMA and
// LDS-direct reads of M0 need 1 (LLVM GCNHazardRecognizer::checkReadM0Hazards).
constexpr uint32_t kM0RestoreWaitStates = 1;

// Spill/fill bracket. prologue saves each register before the call; epilogue
// restores each after the call and waits for the loads before the original runs.
// SGPRs bridge through one VGPR (writelane/readlane); since the single bridge is
// reused, each SGPR restore is a load/wait/readlane of its own.
struct SpillBracket {
  std::vector<uint32_t> prologue;
  std::vector<uint32_t> epilogue;
  // Instructions, not words, the epilogue emits after its last v_readlane_b32.
  // emit_probe_call counts them toward the SGPR restore's wait states.
  uint32_t instructions_after_sgpr_restore = 0;
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
    ++bracket.instructions_after_sgpr_restore;
  }
  for (const SpillSlot &slot : acc_spills) {
    const auto load = build_scratch_load_dword(slot.reg, slot.byte_offset, arch, /*acc=*/true);
    bracket.epilogue.insert(bracket.epilogue.end(), load.begin(), load.end());
    ++bracket.instructions_after_sgpr_restore;
  }
  if (has_vgpr || has_acc) {
    bracket.epilogue.push_back(build_wait_loads_complete(arch));
    ++bracket.instructions_after_sgpr_restore;
  }

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
  const auto fwd = plan.patch_anchor
                       ? compute_sopp_branch_simm16(plan.anchor_offset, plan.trampoline_offset)
                       : std::optional<int16_t>{};
  if (plan.patch_anchor && !fwd) {
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
  const uint32_t original_offset = body.size() * sizeof(uint32_t);
  if (plan.emit_original)
    body.insert(body.end(), plan.original_words.begin(), plan.original_words.end());
  for (const InlineAsmItem &item : plan.after_items)
    body.insert(body.end(), item.words.begin(), item.words.end());

  const uint64_t return_branch_pc = plan.trampoline_offset + body.size() * sizeof(uint32_t);
  const auto ret = compute_sopp_branch_simm16(return_branch_pc, plan.return_target);
  if (plan.returns_to_kernel && !ret) {
    report(error_out, "relocation trampoline return branch exceeds s_branch simm16");
    return std::nullopt;
  }

  TrampolineBytes out;
  if (plan.patch_anchor) {
    out.patched_anchor_bytes.reserve(plan.original_size);
    append_word(out.patched_anchor_bytes, build_s_branch(*fwd, plan.arch));
    if (plan.original_size == 8)
      append_word(out.patched_anchor_bytes, build_s_nop(0, plan.arch));
  }

  out.trampoline_words = std::move(body);
  out.original_byte_offset = original_offset;
  if (plan.returns_to_kernel)
    out.trampoline_words.push_back(build_s_branch(*ret, plan.arch));
  return out;
}

bool TrampolineBuilder::plan_probe_call(TrampolinePlan &plan, const ProbeAbi &abi,
                                        const RegisterSet &live_at_anchor,
                                        const RegisterSet &probe_body_clobbers,
                                        std::string *error_out) {
  // The link pair is whatever the probe's calling convention returns through,
  // so the call site and the probe body agree on one pair. An ABI that could
  // not have come from derive_probe_abi() names no pair worth trusting.
  if (!is_valid_probe_abi(abi)) {
    report(error_out, "probe-call resource planning: unusable probe ABI; cannot derive the "
                      "return-link pair");
    return false;
  }
  const uint16_t kLinkPairBase = abi.link_pair_base;

  // The orchestrator derives both sides from one request, so a mismatch means a
  // direct caller built the two independently and they disagree about how many
  // VGPRs the call writes.
  if (plan.probe_args.size() != abi.num_arg_vgprs) {
    report(error_out, ("probe-call resource planning: " + std::to_string(plan.probe_args.size()) +
                       " argument values but the ABI declares " + std::to_string(abi.num_arg_vgprs))
                          .c_str());
    return false;
  }
  for (size_t i = 0; i < plan.probe_args.size(); ++i) {
    if (!is_declared_probe_arg_source(plan.probe_args[i].source)) {
      report(error_out,
             ("probe-call resource planning: argument " + std::to_string(i) + " names source " +
              std::to_string(static_cast<int>(plan.probe_args[i].source)) +
              ", which is not a declared ProbeArgSource")
                 .c_str());
      return false;
    }
  }
  const RegisterSet arg_regs = arg_registers(abi);
  const RegisterSet observed = probe_argument_registers(plan.probe_args);
  RegisterSet call_clobbers = probe_body_clobbers | arg_regs;
  call_clobbers.expand(probe_link_pair(abi));
  RegisterSet save_destinations;
  if (!validate_register_saves(plan, live_at_anchor | observed, call_clobbers, save_destinations,
                               error_out))
    return false;
  std::vector<ProbeArgumentMove> argument_moves;
  std::optional<uint16_t> argument_copy_temp;
  if (!plan_argument_moves(plan.probe_args, abi, plan.kernel_vgpr_count,
                           live_at_anchor | save_destinations, argument_moves, argument_copy_temp,
                           error_out))
    return false;

  // Target/scc/special-state selection stays within the kernel's ordinary SGPRs,
  // so a temp never lands past the planned .sgpr_count or in a CDNA kernel's
  // special-register tail. The
  // orchestrator sets plan.kernel_sgpr_count; it defaults to the conservative
  // cross-ISA allocatable limit.
  const uint32_t sgpr_bound = ordinary_sgpr_bound(plan.arch, plan.kernel_sgpr_count);

  // The entry-storage pair holds a value produced once at kernel entry and read
  // at arbitrary later sites, so both lanes must be inside the range the temps
  // are drawn from.
  RegisterSet entry_storage;
  if (plan.entry_storage_base) {
    const uint32_t last = static_cast<uint32_t>(*plan.entry_storage_base) + 1u;
    if (last >= sgpr_bound) {
      report(error_out, ("probe-call resource planning: entry-storage pair s[" +
                         std::to_string(*plan.entry_storage_base) + ":" + std::to_string(last) +
                         "] is not below s" + std::to_string(sgpr_bound) +
                         ", where the kernel's ordinary SGPRs end")
                            .c_str());
      return false;
    }
    entry_storage.expand(RegisterRef{RegClass::SGPR, *plan.entry_storage_base, 2});
    RegisterSet captured_vgprs;
    for (const EntryVgprCapture &capture : plan.entry_vgpr_captures) {
      if (capture.sgpr < *plan.entry_storage_base + 2 ||
          capture.sgpr >= *plan.entry_storage_base + 4 || capture.sgpr >= sgpr_bound ||
          capture.vgpr >= std::min(plan.kernel_vgpr_count, 256u) ||
          entry_storage.intersects({RegClass::SGPR, capture.sgpr, 1}) ||
          captured_vgprs.intersects({RegClass::VGPR, capture.vgpr, 1})) {
        report(error_out, "entry VGPR capture is outside the framework-owned register run");
        return false;
      }
      entry_storage.expand(RegisterRef{RegClass::SGPR, capture.sgpr, 1});
      captured_vgprs.expand(RegisterRef{RegClass::VGPR, capture.vgpr, 1});
    }
    for (const ProbeArgValue &arg : plan.probe_args)
      if (arg.source == ProbeArgSource::EntryVgprLane0 &&
          std::ranges::none_of(plan.entry_vgpr_captures, [&](const auto &capture) {
            return capture.vgpr == arg.register_index;
          })) {
        report(error_out, "entry VGPR argument has no framework-owned entry capture");
        return false;
      }
    // Excluding the pair from the envelope's temps is not enough on its own: the
    // probe body runs with the pair still holding the pointer, and a body that
    // writes it destroys the value for every site after this one. Nothing saves
    // and restores it, so such a probe is rejected rather than bracketed.
    if (probe_body_clobbers.intersects(entry_storage)) {
      report(error_out,
             ("probe-call resource planning: the probe body writes the framework's "
              "entry-storage pair s[" +
              std::to_string(*plan.entry_storage_base) + ":" + std::to_string(last) + "]")
                 .c_str());
      return false;
    }
  } else if (std::any_of(plan.probe_args.begin(), plan.probe_args.end(),
                         [](const ProbeArgValue &a) { return reads_entry_storage(a.source); })) {
    report(error_out, "probe-call resource planning: an argument reads the framework's entry "
                      "storage, but the plan names no entry-storage pair");
    return false;
  }

  // A live fixed link pair needs an explicit register backup before the call.
  RegisterSet saved_sources;
  for (const RegisterSave &save : plan.register_saves)
    saved_sources.expand(RegisterRef{save.cls, save.source, 1});
  if ((live_at_anchor - saved_sources).intersects(probe_link_pair(abi))) {
    report(error_out, ("probe-call resource planning: return-link pair s[" +
                       std::to_string(kLinkPairBase) + ":" + std::to_string(kLinkPairBase + 1) +
                       "] is live at the anchor; cannot yet save a live link pair")
                          .c_str());
    return false;
  }

  RegisterSet link_pair;
  link_pair.expand(probe_link_pair(abi));

  // Target-address pair: dead, even-aligned, and not the link pair. It is
  // read by s_swappc before the probe body runs, so it may overlap
  // probe_body_clobbers.
  if (entry_storage.intersects(save_destinations)) {
    report(error_out, "register preservation overlaps the framework's entry storage");
    return false;
  }
  const RegisterSet target_unavail = live_at_anchor | link_pair | entry_storage | save_destinations;
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
  // A spilled register is live at the anchor and clobbered by the probe or by
  // the envelope's argument writes. The argument VGPRs are the one builder
  // clobber that is not chosen dead -- the ABI fixes them at arg_vgpr_base --
  // so unlike the SGPR temps they can be live and need spilling. Spilling forces
  // EXEC=-1 around the store/load, so EXEC must be saved even if the probe never
  // touches it.
  const bool will_spill = (live_at_anchor | observed).intersects(probe_body_clobbers | arg_regs);

  // The full-mask window the spill stores/loads run under. Argument writes join
  // it: a v_mov_b32 under the anchor mask leaves the argument undefined in the
  // inactive lanes, which an EXEC-independent read in the probe (v_readlane of a
  // fixed lane, or anything the probe runs after widening EXEC itself) would then
  // see. Writing under EXEC=-1 defines every lane. It costs nothing in guest
  // state: a live argument VGPR is stored before the write and reloaded after the
  // call, both under the same full mask.
  //
  // A full-exec site opens the window for its own sake: the probe runs inside it
  // rather than under the anchor mask.
  const bool needs_full_mask = will_spill || !plan.register_saves.empty() ||
                               !plan.probe_args.empty() || plan.force_full_exec;

  // EXEC/VCC/M0 operand codes are resolved per-arch, but only when actually
  // reserving that register -- so a plan with no special-state saves (and no
  // full-mask window) stays arch-agnostic, as the resource-planning tests rely
  // on. EXEC also rides this path whenever that window is needed, since the
  // anchor mask has to be restored from somewhere.
  const bool save_exec = plan.preserve_exec || needs_full_mask;
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
  // Each argument is one v_mov_b32; an immediate adds the literal word its src0
  // field names, a register source does not.
  for (const ProbeArgValue &arg : plan.probe_args) {
    switch (arg.source) {
    case ProbeArgSource::AnchorVgpr:
      break; // Parallel copies are counted together, including cycle breaks.
    case ProbeArgSource::Immediate:
      before_words += 2;
      break;
    case ProbeArgSource::AnchorExecLo:
    case ProbeArgSource::AnchorExecHi:
    case ProbeArgSource::LogBufferPtrLo:
    case ProbeArgSource::LogBufferPtrHi:
    case ProbeArgSource::EntryVgprLane0:
      before_words += 1;
      break;
    }
  }
  before_words += argument_moves.size();
  before_words += 2 * plan.register_saves.size();
  if (plan.preserve_scc)
    before_words += 2; // s_cselect_b32 (save) + s_cmp_lg_u32 (restore)
  // Each special-state register adds one s_mov save + one s_mov restore.
  before_words += static_cast<uint32_t>(special_saves.size()) * 2;
  // The full-mask window's EXEC toggles, counted one per s_mov so this mirrors
  // emit_probe_call's three conditionals in the same order. A count that does not
  // match what the emitter produces trips the drift guard there.
  if (needs_full_mask) {
    before_words += 1; // Widen ahead of the spill stores and the argument writes.
    // Restore the anchor mask so the probe runs under it. A force_full_exec site
    // skips this; leaving the window open across the call is the whole policy.
    if (!plan.force_full_exec)
      before_words += 1;
    // Re-widen for the spill loads. Emitted even when the epilogue is empty (an
    // argument site that spills nothing), and on a force_full_exec site too,
    // since the probe may have narrowed EXEC while it ran.
    before_words += 1;
  }

  plan.is_probe_call = true;
  plan.link_pair_base = kLinkPairBase;
  plan.arg_vgpr_base = abi.arg_vgpr_base;
  plan.target_pair_base = *target_pair;
  if (scc_temp)
    plan.scc_temp = *scc_temp;
  plan.special_state_saves = std::move(special_saves);
  plan.before_word_count = before_words;
  plan.argument_moves = std::move(argument_moves);
  plan.argument_copy_temp = argument_copy_temp;

  plan.builder_clobbers = link_pair | target_pair_set | arg_regs | save_destinations;
  if (argument_copy_temp)
    plan.builder_clobbers.expand(RegisterRef{RegClass::VGPR, *argument_copy_temp, 1});
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
  std::vector<uint32_t> boundary_drain = build_wait_all_loads_complete(plan.arch);
  if (plan.arch == ROCJITSU_CODE_ARCH_RDNA4)
    boundary_drain.push_back(build_sopp_encoding(plan.arch, rdna4::kSWaitAluSopp, 0xf1fc));

  std::vector<uint32_t> env;
  env.insert(env.end(), boundary_drain.begin(), boundary_drain.end());

  // The full-mask window: needed to spill, and to define the argument VGPRs in
  // every lane rather than only the ones active at the anchor. Mirrors
  // plan_probe_call's needs_full_mask, which is what sized the envelope; this
  // drives the EXEC toggles only, since EXEC save/restore is decided by
  // special_state_saves membership.
  const bool full_mask_exec = !plan.vgpr_spills.empty() || !plan.sgpr_spills.empty() ||
                              !plan.acc_spills.empty() || !plan.probe_args.empty() ||
                              !plan.register_saves.empty() || plan.force_full_exec;

  // SGPR pair holding the saved anchor EXEC (populated by the save loop below).
  // Reused to restore the anchor mask before the call; always present when the
  // full-mask window is open.
  uint16_t exec_temp = 0;
  bool exec_temp_found = false;
  for (const SpecialStateSlot &s : plan.special_state_saves)
    if (s.operand == scalar_operand_exec_lo(plan.arch)) {
      exec_temp = s.temp_base;
      exec_temp_found = true;
    }
  // full_mask_exec implies needs_full_mask implies save_exec, so plan_probe_call
  // must have reserved an EXEC temp for any such site. Verify rather than trust
  // the default. Fail closed instead.
  if (full_mask_exec && !exec_temp_found) {
    report(error_out, "emit_probe_call: site needs the full-mask window but has no saved EXEC temp "
                      "to restore the anchor mask; plan_probe_call must reserve one");
    return std::nullopt;
  }

  // Same shape as the EXEC temp above: planning rejects this combination, so
  // reaching it means the plan came from somewhere else. Checked once here rather
  // than dereferenced per argument in the loop below.
  if (!plan.entry_storage_base &&
      std::any_of(plan.probe_args.begin(), plan.probe_args.end(),
                  [](const ProbeArgValue &a) { return reads_entry_storage(a.source); })) {
    report(error_out, "probe-call emission: an argument reads the framework's entry storage, but "
                      "the plan names no entry-storage pair");
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
  if (plan.probe_args.size() > kMaxProbeArgVgprs) {
    report(error_out, "probe-call emission: argument block does not fit the call ABI");
    return std::nullopt;
  }
  const std::optional<ProbeAbi> argument_abi = derive_probe_abi(
      ProbeCallingConvention::AmdGpuFuncReturnS30S31, static_cast<uint8_t>(plan.probe_args.size()));
  if (!argument_abi) {
    report(error_out, "probe-call emission: argument block does not fit the call ABI");
    return std::nullopt;
  }
  RegisterSet unavailable_backups = arg_registers(*argument_abi);
  unavailable_backups.expand(probe_link_pair(*argument_abi));
  unavailable_backups.expand({RegClass::SGPR, target_lo, 2});
  if (plan.preserve_scc)
    unavailable_backups.expand({RegClass::SGPR, plan.scc_temp, 1});
  for (const SpecialStateSlot &save : plan.special_state_saves)
    unavailable_backups.expand({RegClass::SGPR, save.temp_base, save.width});
  if (plan.entry_storage_base)
    unavailable_backups.expand({RegClass::SGPR, *plan.entry_storage_base, 2});
  for (const EntryVgprCapture &capture : plan.entry_vgpr_captures)
    unavailable_backups.expand({RegClass::SGPR, capture.sgpr, 1});
  RegisterSet register_save_destinations;
  if (!validate_register_saves(plan, {}, unavailable_backups, register_save_destinations,
                               error_out))
    return std::nullopt;
  for (const RegisterSave &save : plan.register_saves)
    env.push_back(save.cls == RegClass::SGPR
                      ? build_s_mov_b32(save.destination, save.source, plan.arch)
                      : build_v_mov_b32_src(save.destination, 256 + save.source, plan.arch));

  // Capture guest operands before constants or scalar-sourced arguments can
  // overwrite them. The order implements a parallel assignment, including cycles.
  std::vector<ProbeArgumentMove> expected_moves;
  std::optional<uint16_t> copy_temp = plan.argument_copy_temp;
  if (!plan_argument_moves(plan.probe_args, *argument_abi, plan.kernel_vgpr_count,
                           register_save_destinations, expected_moves, copy_temp, error_out) ||
      expected_moves != plan.argument_moves) {
    report(error_out, "probe-call emission: guest VGPR argument copy plan is inconsistent");
    return std::nullopt;
  }
  for (const ProbeArgumentMove &move : plan.argument_moves)
    env.push_back(build_v_mov_b32_src(move.destination, 256 + move.source, plan.arch));

  // Argument materialization: consecutive VGPRs from the ABI's base, in order.
  //
  // Three things fix this position. It follows the spill prologue, so an argument
  // VGPR that was live at the anchor is already stored -- and the prologue ends in
  // a store-completion wait, so the store has finished reading the register before
  // the v_mov overwrites it. It runs inside the full-mask window, so every lane's
  // copy is defined rather than only the lanes active at the anchor. And
  // v_mov_b32 writes no SCC, so it cannot disturb the save/restore pair below.
  for (size_t i = 0; i < plan.probe_args.size(); ++i) {
    const ProbeArgValue &arg = plan.probe_args[i];
    const uint16_t vdst = static_cast<uint16_t>(plan.arg_vgpr_base + i);
    // Screened again here rather than inherited from planning: emit_probe_call is
    // callable on a plan this builder did not produce.
    if (!is_declared_probe_arg_source(arg.source)) {
      report(error_out, ("probe-call emission: argument " + std::to_string(i) + " names source " +
                         std::to_string(static_cast<int>(arg.source)) +
                         ", which is not a declared ProbeArgSource")
                            .c_str());
      return std::nullopt;
    }
    // The anchor mask comes from the saved pair, not from `exec`: the widen above
    // already overwrote the register. exec_temp is the pair base, so the high
    // dword is the next SGPR.
    //
    // So an EXEC-sourced argument needs a site that saved EXEC. Nothing asks for
    // that on its own: any argument at all opens the full-mask window, which
    // reserves the temp, and the guard above fails closed if it is somehow absent.
    switch (arg.source) {
    case ProbeArgSource::EntryVgprLane0: {
      const std::vector<EntryVgprCapture>::const_iterator captured =
          std::ranges::find_if(plan.entry_vgpr_captures,
                               [&](const auto &value) { return value.vgpr == arg.register_index; });
      if (captured == plan.entry_vgpr_captures.end()) {
        report(error_out, "probe-call emission: missing entry VGPR capture");
        return std::nullopt;
      }
      env.push_back(build_v_mov_b32_src(vdst, captured->sgpr, plan.arch));
      break;
    }
    case ProbeArgSource::AnchorVgpr:
      break; // Already captured by the parallel assignment above.
    case ProbeArgSource::Immediate: {
      const auto words = build_v_mov_b32_imm(vdst, arg.immediate, plan.arch);
      env.insert(env.end(), words.begin(), words.end());
      break;
    }
    case ProbeArgSource::AnchorExecLo:
      env.push_back(build_v_mov_b32_src(vdst, exec_temp, plan.arch));
      break;
    case ProbeArgSource::AnchorExecHi:
      env.push_back(build_v_mov_b32_src(vdst, static_cast<uint16_t>(exec_temp + 1), plan.arch));
      break;
    // The pair is read, never written, so it needs no save/restore here: the
    // prologue defined it at kernel entry and planning kept the envelope and the
    // probe body off it.
    case ProbeArgSource::LogBufferPtrLo:
      env.push_back(build_v_mov_b32_src(vdst, *plan.entry_storage_base, plan.arch));
      break;
    case ProbeArgSource::LogBufferPtrHi:
      env.push_back(build_v_mov_b32_src(vdst, static_cast<uint16_t>(*plan.entry_storage_base + 1),
                                        plan.arch));
      break;
    }
  }

  // Restore the anchor EXEC before the call so the probe runs under the anchor mask,
  // not the full mask used to bracket the stores and argument writes. The loads are
  // re-widened after. Skipped under force_full_exec, which is what leaves the probe
  // running with every lane enabled.
  if (full_mask_exec && !plan.force_full_exec)
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
  // Emitted on a force_full_exec site too: it entered the call at -1, but the probe
  // may have narrowed EXEC while running, which is the case this guards against.
  if (full_mask_exec)
    env.push_back(build_s_mov_b64(scalar_operand_exec_lo(plan.arch),
                                  scalar_inline_neg_one(plan.arch), plan.arch));

  // Spill fills: reload each saved register after the call and wait for the loads.
  env.insert(env.end(), spill.epilogue.begin(), spill.epilogue.end());
  for (const RegisterSave &save : plan.register_saves)
    env.push_back(save.cls == RegClass::SGPR
                      ? build_s_mov_b32(save.source, save.destination, plan.arch)
                      : build_v_mov_b32_src(save.source, 256 + save.destination, plan.arch));

  // Special-state restores: copy each temp back into its register. After the spill
  // loads so EXEC is restored to its anchor mask only once the full-mask loads are
  // done; all run before the relocated original.
  for (const SpecialStateSlot &s : plan.special_state_saves)
    env.push_back(s.width == 2 ? build_s_mov_b64(s.operand, s.temp_base, plan.arch)
                               : build_s_mov_b32(s.operand, s.temp_base, plan.arch));

  // GFX9 wait states the restores still owe the relocated original, topped up with
  // one s_nop covering the larger shortfall. Counted rather than fixed, so a site
  // whose reloads and restores already fill the gap pays nothing. The guest code
  // the return branch reaches is further away still, so it is covered too.
  uint32_t owed_wait_states = 0;
  if (needs_restore_wait_states(plan.arch)) {
    if (!plan.sgpr_spills.empty()) {
      const uint32_t provided = spill.instructions_after_sgpr_restore +
                                static_cast<uint32_t>(plan.special_state_saves.size());
      if (provided < kSgprRestoreWaitStates)
        owed_wait_states = kSgprRestoreWaitStates - provided;
    }
    // Nothing is counted after the M0 restore: plan_probe_call reserves M0 last,
    // so its restore is the final instruction above. A plan ordering it earlier
    // is only over-padded.
    const bool restores_m0 =
        std::ranges::any_of(plan.special_state_saves, [&](const SpecialStateSlot &s) {
          return s.operand == scalar_operand_m0(plan.arch);
        });
    if (restores_m0)
      owed_wait_states = std::max(owed_wait_states, kM0RestoreWaitStates);
  }
  const size_t wait_state_pad_words = owed_wait_states > 0 ? 1 : 0;
  if (owed_wait_states > 0)
    env.push_back(build_s_nop(static_cast<uint16_t>(owed_wait_states - 1), plan.arch));
  const size_t restore_drain_words = plan.arch == ROCJITSU_CODE_ARCH_RDNA4 ? 1 : 0;
  if (restore_drain_words)
    env.push_back(build_sopp_encoding(plan.arch, rdna4::kSWaitAluSopp, 0xf1fc));

  // Plan/emit drift guard: the planner committed to this many envelope words and
  // the orchestrator sized the layout around it. A mismatch means the two
  // disagree about the envelope shape. before_word_count is the arch-agnostic
  // envelope; the spill bracket, the two boundary drains and the wait-state pad
  // are accounted separately since their sizes are arch- and site-specific.
  if (env.size() != plan.before_word_count + spill.prologue.size() + spill.epilogue.size() +
                        2 * boundary_drain.size() + wait_state_pad_words + restore_drain_words) {
    report(error_out, "emit_probe_call: synthesized envelope word count does not match the planned "
                      "before_word_count");
    return std::nullopt;
  }

  // Hand the synthesized envelope to build() for layout and branch math so the
  // SOPP range checks are shared with the inline path.
  TrampolinePlan emit_plan = plan;
  emit_plan.before_items.assign(1, InlineAsmItem{std::move(env)});
  emit_plan.after_items.clear();
  return build(emit_plan, error_out);
}

} // namespace rocjitsu
