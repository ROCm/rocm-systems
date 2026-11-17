// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "differential.h"

#include "boundary.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <unordered_set>

#ifndef RJ_ISA_DIFF_RJ_REVISION
#define RJ_ISA_DIFF_RJ_REVISION "unknown"
#endif
#ifndef RJ_ISA_DIFF_RJ_VERSION
#define RJ_ISA_DIFF_RJ_VERSION "unknown"
#endif

namespace rocjitsu::isa_diff {
namespace {

using Clock = std::chrono::steady_clock;

uint64_t ns_since(Clock::time_point start) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

bool is_unsupported_status(RunStatus s) {
  return s == RunStatus::DecodeFailure || s == RunStatus::MissingExecutionBackend ||
         s == RunStatus::Unimplemented || s == RunStatus::UnsupportedOperand ||
         s == RunStatus::IllegalInstruction;
}

bool is_excluded_status(RunStatus s) { return s == RunStatus::StepLimit; }

/// Loop step limit multiplier of a re-run past the step limit, and the number of
/// divergences after which further one-sided step limits are counted without a re-run.
constexpr size_t kStepLimitRerunScale = 16;
constexpr uint64_t kStepLimitReruns = 4;

/// One side exited and the other ran past the loop step limit: re-run the looping side
/// with kStepLimitRerunScale times the limit, so a slower loop that terminates is compared
/// like any case. The runner keeps the larger limit until the next case resets it.
/// Returns whether a side was re-run.
bool rerun_past_step_limit(RegionRunner &old_runner, RegionRunner &new_runner,
                           const MachineState &init_old, const MachineState &init_new,
                           size_t old_budget, size_t new_budget, RunOutcome &oo, RunOutcome &on,
                           std::vector<TraceStep> *trace_old = nullptr,
                           std::vector<TraceStep> *trace_new = nullptr) {
  const bool lim_old = oo.status == RunStatus::StepLimit;
  const bool lim_new = on.status == RunStatus::StepLimit;
  if (lim_old == lim_new || (lim_old ? on : oo).status != RunStatus::Ok)
    return false;
  RegionRunner &runner = lim_old ? old_runner : new_runner;
  if (runner.loop_scale() >= kStepLimitRerunScale)
    return false;
  runner.set_loop_scale(kStepLimitRerunScale);
  runner.load(lim_old ? init_old : init_new);
  std::vector<TraceStep> *trace = lim_old ? trace_old : trace_new;
  if (trace)
    trace->clear();
  (lim_old ? oo : on) = runner.run(lim_old ? old_budget : new_budget, trace);
  return true;
}

/// One side exited normally, the other ran past the loop step limit.
bool one_sided_step_limit(const RunOutcome &oo, const RunOutcome &on) {
  const bool lim_old = oo.status == RunStatus::StepLimit;
  const bool lim_new = on.status == RunStatus::StepLimit;
  return lim_old != lim_new && (lim_old ? on : oo).status == RunStatus::Ok;
}

bool is_harness_status(RunStatus s) {
  return s == RunStatus::HostException || s == RunStatus::ExecutionError ||
         s == RunStatus::InvalidRegisterAccess || s == RunStatus::BudgetExceeded ||
         s == RunStatus::PcWritten || s == RunStatus::WaveStateChanged;
}

Json trace_json(const std::vector<TraceStep> &trace) {
  Json arr = Json::array();
  for (const auto &step : trace) {
    Json s = Json::object();
    s.set("index", static_cast<uint64_t>(step.index));
    s.set("pc", hex64(step.pc));
    s.set("asm", step.text);
    Json changed = Json::object();
    for (unsigned l = 0; l < kNumLocs; ++l)
      if (step.changed.test(l))
        changed.set(loc_name(l), hex32(step.after.get(l)));
    s.set("changed", std::move(changed));
    if (step.vchanged.any()) {
      s.set("vgprs_changed", format_vgprs(step.vchanged));
      // Lane 0 of each changed vector register (the bundle's states hold every lane).
      Json lane0 = Json::object();
      for (unsigned r = 0; r < kNumVgprs; ++r)
        if (step.vchanged.test(r) && r < step.after.vgpr_count)
          lane0.set(vgpr_name(r), hex32(step.after.vget(r, 0)));
      s.set("vgprs_lane0", std::move(lane0));
    }
    if (!step.frame_changed.empty()) {
      Json f = Json::array();
      for (const auto &name : step.frame_changed)
        f.push(name);
      s.set("frame_changed", std::move(f));
    }
    arr.push(std::move(s));
  }
  return arr;
}

Json outcome_json(const RunOutcome &o) {
  Json j = Json::object();
  j.set("status", run_status_name(o.status));
  j.set("executed", static_cast<uint64_t>(o.executed));
  if (o.status != RunStatus::Ok) {
    j.set("failed_index", static_cast<uint64_t>(o.failed_index));
    j.set("detail", o.detail);
  }
  return j;
}

uint8_t initial_byte(const MachineState &s, uint64_t x) {
  auto it = std::lower_bound(s.mem_init.begin(), s.mem_init.end(),
                             std::pair<uint64_t, uint32_t>{x >> 2, 0});
  if (it != s.mem_init.end() && it->first == (x >> 2))
    return static_cast<uint8_t>(it->second >> ((x & 3) * 8));
  return initial_memory_byte(s.mem_seed, x);
}

/// Bytes either side stored whose final values differ. A byte one side did not store
/// keeps its initial content on that side.
/// Calls @p f(address, byte on a, byte on b) for every byte either side stored, in address
/// order: a linear merge of the two sorted mem_written lists.
template <typename F> void for_each_stored(const MachineState &a, const MachineState &b, F f) {
  const auto &wa = a.mem_written, &wb = b.mem_written;
  size_t i = 0, j = 0;
  while (i < wa.size() || j < wb.size()) {
    if (j == wb.size() || (i < wa.size() && wa[i].first < wb[j].first)) {
      f(wa[i].first, wa[i].second, initial_byte(b, wa[i].first));
      ++i;
    } else if (i == wa.size() || wb[j].first < wa[i].first) {
      f(wb[j].first, initial_byte(a, wb[j].first), wb[j].second);
      ++j;
    } else {
      f(wa[i].first, wa[i].second, wb[j].second);
      ++i;
      ++j;
    }
  }
}

std::vector<uint64_t> differing_memory(const MachineState &a, const MachineState &b) {
  std::vector<uint64_t> out;
  // mem_written and mem_dead are sorted (capture, from_json).
  auto dead = [&](uint64_t x) { // dead where the run left (memory_dead_at_exit)
    return std::binary_search(a.mem_dead.begin(), a.mem_dead.end(), x) ||
           std::binary_search(b.mem_dead.begin(), b.mem_dead.end(), x);
  };
  for_each_stored(a, b, [&](uint64_t x, uint8_t va, uint8_t vb) {
    if (va != vb && !dead(x))
      out.push_back(x);
  });
  return out;
}

uint8_t memory_byte(const MachineState &s, uint64_t x) {
  // mem_written is sorted by address (capture, from_json).
  const auto it = std::lower_bound(s.mem_written.begin(), s.mem_written.end(),
                                   std::pair<uint64_t, uint8_t>{x, 0});
  return it != s.mem_written.end() && it->first == x ? it->second : initial_byte(s, x);
}

/// Don't-care bits of old's outputs (LLVM undef): those that changed when old re-ran with
/// its undefined registers (never written since the function entry) refilled. Any value
/// there is a valid refinement.
struct UndefMask {
  bool any = false;
  /// Old leaves the region through another exit under some refill or NaN choice: its
  /// control flow depends on an undefined value (branching on undef is undefined in LLVM).
  bool exit_varies = false;
  bool from_undef = false; ///< Built from refills of undefined registers.
  bool from_nan = false;   ///< Built from alternative NaN choices.
  std::array<uint32_t, kNumLocs> loc{};
  std::vector<uint32_t> vgpr; ///< Register-major lanes, as MachineState::vgpr.
  std::map<uint64_t, uint8_t> mem;
  std::vector<std::vector<uint32_t>> asm_uses;
  /// Undefined inputs old passes through unchanged (its output is the input, in the base
  /// run and every refill): any value there is exact, not a per-bit approximation.
  LocSet passthrough;
  VSet vpassthrough;
};

struct CompareConfig {
  bool strict_nan = false;
  const UndefMask *undef = nullptr;
};

struct CompareStats {
  bool nan = false;    ///< A difference vanished through NaN equivalence.
  bool undef = false;  ///< A difference vanished through undefined-bit masking.
  bool masked = false; ///< A difference vanished under the per-bit don't-care mask.
};

/// A difference vanished under the mask: credit the mask's sources. @p exact: the location
/// is an undefined input old passes through (any value is right), not a per-bit
/// approximation.
void note_mask(const UndefMask &m, CompareStats &st, bool exact = false) {
  st.undef = st.undef || m.from_undef;
  st.nan = st.nan || m.from_nan;
  st.masked = st.masked || !exact;
}

/// The stored floating-point unit containing byte @p x, if any.
const FpUnit *fp_unit_at(const MachineState &s, uint64_t x) {
  for (const FpUnit &u : s.mem_fp)
    if (x >= u.addr && x < u.addr + u.size)
      return &u;
  return nullptr;
}

/// Both sides stored the same floating-point unit over byte @p x and both values are NaNs.
bool nan_unit_equal(const MachineState &a, const MachineState &b, uint64_t x) {
  const FpUnit *ua = fp_unit_at(a, x), *ub = fp_unit_at(b, x);
  if (!ua || !ub || !(*ua == *ub))
    return false;
  uint64_t va = 0, vb = 0;
  for (unsigned i = 0; i < ua->size; ++i) {
    va |= uint64_t{memory_byte(a, ua->addr + i)} << (8 * i);
    vb |= uint64_t{memory_byte(b, ua->addr + i)} << (8 * i);
  }
  switch (ua->format) {
  case kFpF32:
    return is_nan_f32(static_cast<uint32_t>(va)) && is_nan_f32(static_cast<uint32_t>(vb));
  case kFpF16Lo:
    return is_nan_f16(static_cast<uint32_t>(va)) && is_nan_f16(static_cast<uint32_t>(vb));
  case kFpBF16Lo:
    return is_nan_bf16(static_cast<uint32_t>(va)) && is_nan_bf16(static_cast<uint32_t>(vb));
  case kFpF64Lo:
    return is_nan_f64(va) && is_nan_f64(vb);
  default:
    return false;
  }
}

/// First difference between what the two sides' inline-asm `use` blocks observed, or empty.
/// With barriers: the first difference in what other waves of the workgroup could observe
/// ("" for none). One wave runs, so the order of its shared-memory accesses relative to the
/// barriers stands in for that: the sides execute as many barriers, store to the same
/// shared dwords in each epoch, and load an LDS dword both load in the same epochs (a load only
/// one side makes, e.g. through a wider access, is not compared).
std::string barrier_difference(const MachineState &a, const MachineState &b) {
  if (a.barriers != b.barriers)
    return "the sides execute different numbers of barriers: old " + std::to_string(a.barriers) +
           ", new " + std::to_string(b.barriers);
  if (a.mem_sync == b.mem_sync)
    return {};
  auto where = [](uint64_t addr) {
    return addr & kLdsTag ? "LDS " + hex64(addr & ~kLdsTag) : hex64(addr);
  };
  std::vector<SyncAccess> sa, sb;
  std::map<uint64_t, std::vector<uint32_t>> la, lb;
  auto split = [](const MachineState &s, std::vector<SyncAccess> &stores,
                  std::map<uint64_t, std::vector<uint32_t>> &loads) {
    for (const SyncAccess &x : s.mem_sync) {
      if (x.store)
        stores.push_back(x);
      else
        loads[x.addr].push_back(x.epoch);
    }
  };
  split(a, sa, la);
  split(b, sb, lb);
  if (sa != sb) {
    std::vector<SyncAccess> only;
    std::ranges::set_symmetric_difference(sa, sb, std::back_inserter(only));
    const SyncAccess &x = only.front();
    const bool old_only = std::ranges::binary_search(sa, x);
    return std::string(old_only ? "old" : "new") + " stores " + where(x.addr) +
           " in barrier epoch " + std::to_string(x.epoch) + " and " + (old_only ? "new" : "old") +
           " does not: other waves of the workgroup would see different memory after that "
           "barrier";
  }
  auto epochs = [](const std::vector<uint32_t> &e) {
    std::string s;
    for (uint32_t x : e)
      s += (s.empty() ? "" : ",") + std::to_string(x);
    return s;
  };
  for (const auto &[addr, ea] : la) {
    const auto it = lb.find(addr);
    if (it != lb.end() && it->second != ea)
      return "old loads " + where(addr) + " in barrier epoch(s) " + epochs(ea) + ", new in " +
             epochs(it->second) +
             ": a load moved across a barrier reads what other waves "
             "wrote at another time";
  }
  return {};
}

std::string asm_use_difference(const MachineState &a, const MachineState &b,
                               const CompareConfig &cfg, CompareStats &st) {
  const size_t n = std::max(a.asm_uses.size(), b.asm_uses.size());
  for (size_t i = 0; i < n; ++i) {
    if (i >= a.asm_uses.size() || i >= b.asm_uses.size() ||
        a.asm_uses[i].block != b.asm_uses[i].block ||
        a.asm_uses[i].values.size() != b.asm_uses[i].values.size())
      return "inline asm use blocks executed differently (old " +
             std::to_string(a.asm_uses.size()) + ", new " + std::to_string(b.asm_uses.size()) +
             " observations)";
    const AsmUse &ua = a.asm_uses[i];
    const AsmUse &ub = b.asm_uses[i];
    for (size_t k = 0; k < ua.values.size(); ++k) {
      uint32_t d = ua.values[k] ^ ub.values[k];
      if (!d)
        continue;
      if (!cfg.strict_nan) {
        const uint8_t ta = k < ua.fp.size() ? ua.fp[k] : 0, tb = k < ub.fp.size() ? ub.fp[k] : 0;
        const uint32_t nd = nan_residual(ua.values[k], ub.values[k], ta, tb) & d;
        if (!nd) {
          st.nan = true;
          continue;
        }
        d = nd;
      }
      if (cfg.undef && i < cfg.undef->asm_uses.size() && k < cfg.undef->asm_uses[i].size() &&
          (d & cfg.undef->asm_uses[i][k])) {
        d &= ~cfg.undef->asm_uses[i][k];
        if (!d) {
          note_mask(*cfg.undef, st);
          continue;
        }
      }
      return "inline asm block " + std::to_string(ua.block) + " use differs at value " +
             std::to_string(k) + ": old " + hex32(ua.values[k]) + " new " + hex32(ub.values[k]);
    }
  }
  return {};
}

/// Compare one executed pair. Returns the mismatch kind, or empty when they agree. Bits in
/// the undef mask and the contract's exit masks are don't care; NaN results of the same
/// floating-point format compare equal unless cfg.strict_nan.
std::string compare_pair(const RegionPair &region, const MachineState &init_old,
                         const MachineState &init_new, const MachineState &final_old,
                         const MachineState &final_new, Mismatch &m, const CompareConfig &cfg = {},
                         CompareStats *stats = nullptr) {
  CompareStats local;
  CompareStats &st = stats ? *stats : local;
  const UndefMask *undef = cfg.undef && cfg.undef->any ? cfg.undef : nullptr;
  const RegionContract &k = region.contract;
  const auto fwd = k.exit_forward();
  // Control flow: both sides must leave through corresponding exits; that exit's R_out
  // is compared.
  const RegionExit *exit = nullptr;
  if (!k.exits.empty()) {
    for (const RegionExit &e : k.exits)
      if (e.old_pc == final_old.exit_pc)
        exit = &e;
    if (!exit || exit->new_pc != final_new.exit_pc) {
      m.detail = std::string("the sides leave through different exits: old ") +
                 (final_old.exit_pc ? hex64(final_old.exit_pc) : "end") + ", new " +
                 (final_new.exit_pc ? hex64(final_new.exit_pc) : "end") +
                 (exit ? "" : " (old's exit is not declared)");
      return "exit_divergence";
    }
  }
  const LocSet &observable = exit ? exit->observable : k.observable;
  const VSet &vobservable = exit ? exit->vobservable : k.vobservable;
  const VSet &vactive_only = exit ? exit->vactive_only : k.vactive_only;
  // Effect model: execution must not write anything the model did not declare.
  const LocSet extra_old = differing_locs(init_old, final_old) & ~k.declared_writes_old;
  const LocSet extra_new = differing_locs(init_new, final_new) & ~k.declared_writes_new;
  const VSet vextra_old = differing_vgprs(init_old, final_old) & ~k.vwrites_old;
  const VSet vextra_new = differing_vgprs(init_new, final_new) & ~k.vwrites_new;
  if (extra_old.any() || extra_new.any() || vextra_old.any() || vextra_new.any()) {
    m.locs = extra_old | extra_new;
    m.vlocs = vextra_old | vextra_new;
    auto both = [](const LocSet &l, const VSet &v) {
      std::string s = format_locs(l);
      if (v.any())
        s += (s.empty() ? "" : ",") + format_vgprs(v);
      return s;
    };
    m.detail = "execution wrote undeclared locations (old: " + both(extra_old, vextra_old) +
               "; new: " + both(extra_new, vextra_new) + ")";
    return "effect_model";
  }
  // Frame preservation on each side.
  auto fo = differing_frame_fields(init_old.frame, final_old.frame);
  auto fn = differing_frame_fields(init_new.frame, final_new.frame);
  if (!fo.empty() || !fn.empty()) {
    m.frame_fields = fo;
    m.frame_fields.insert(m.frame_fields.end(), fn.begin(), fn.end());
    m.detail = "frame state changed by the region";
    return "frame_preservation";
  }
  // R_out: equality on the observable set (old naming; the new side is read at map(l)).
  LocSet diff;
  std::array<uint32_t, kNumLocs> smask;
  smask.fill(0xffffffffu);
  if (!exit) // narrow return values (the fallthrough exit of a straight region)
    for (const auto &[l, mask] : k.exit_masks)
      smask[l] = mask;
  for (unsigned l = 0; l < kNumLocs; ++l) {
    if (!observable.test(l))
      continue;
    uint32_t d = (final_old.get(l) ^ final_new.get(fwd[l])) & smask[l];
    if (!d)
      continue;
    // Exact rule first: a NaN that a floating-point operation produced on both sides
    // (SALU float, or one moved into an SGPR) has unspecified bits.
    if (!cfg.strict_nan) {
      d &= nan_residual(final_old.get(l), final_new.get(fwd[l]), final_old.sfp_tag(l),
                        final_new.sfp_tag(fwd[l]));
      if (!d) {
        st.nan = true;
        continue;
      }
    }
    // Then the per-bit don't-care mask (heuristic, unless old passes the undefined input
    // through; marks the result).
    if (undef && !(d & ~undef->loc[l])) {
      note_mask(*undef, st, undef->passthrough.test(l));
      continue;
    }
    diff.set(l);
  }
  // VGPRs: every lane, active or not (an inactive lane must keep its value), except bits
  // the exit masks (partial return values), the undef mask, or NaN equivalence release.
  std::map<unsigned, uint32_t> exit_mask(k.vexit_masks.begin(), k.vexit_masks.end());
  VSet vdiff;
  unsigned first_r = 0, first_lane = 0, first_count = 0;
  const VSet vcand = differing_vgprs(final_old, final_new) & vobservable;
  const unsigned nv = std::min(final_old.vgpr_count, final_new.vgpr_count);
  for (unsigned r = 0; r < kNumVgprs; ++r) {
    if (!vcand.test(r))
      continue;
    if (r >= nv) {
      vdiff.set(r);
      continue;
    }
    const auto em = exit_mask.find(r);
    const uint32_t emask = em == exit_mask.end() ? 0xffffffffu : em->second;
    // Lanes inactive at an exit after which EXEC never changes are not observable.
    const uint64_t active = vactive_only.test(r) ? (uint64_t{final_old.get(kLocExecLo)} |
                                                    uint64_t{final_old.get(kLocExecHi)} << 32 |
                                                    uint64_t{final_new.get(fwd[kLocExecLo])} |
                                                    uint64_t{final_new.get(fwd[kLocExecHi])} << 32)
                                                 : ~uint64_t{0};
    unsigned lanes = 0, lane0 = 0;
    for (unsigned lane = 0; lane < kWaveLanes; ++lane) {
      if (!((active >> lane) & 1u))
        continue;
      const uint32_t a = final_old.vget(r, lane), b = final_new.vget(r, lane);
      uint32_t d = (a ^ b) & emask;
      if (!d)
        continue;
      if (!cfg.strict_nan) {
        const uint8_t ta = final_old.fp_tag(r, lane), tb = final_new.fp_tag(r, lane);
        uint32_t nd = d;
        auto pair_nan = [&](unsigned lo) {
          const uint64_t va =
              uint64_t{final_old.vget(lo, lane)} | (uint64_t{final_old.vget(lo + 1, lane)} << 32);
          const uint64_t vb =
              uint64_t{final_new.vget(lo, lane)} | (uint64_t{final_new.vget(lo + 1, lane)} << 32);
          return is_nan_f64(va) && is_nan_f64(vb);
        };
        if ((ta & tb & kFpF64Lo) && r + 1 < nv &&
            (final_old.fp_tag(r + 1, lane) & final_new.fp_tag(r + 1, lane) & kFpF64Hi))
          nd = pair_nan(r) ? 0 : d;
        else if ((ta & tb & kFpF64Hi) && r >= 1 &&
                 (final_old.fp_tag(r - 1, lane) & final_new.fp_tag(r - 1, lane) & kFpF64Lo))
          nd = pair_nan(r - 1) ? 0 : d;
        else
          nd = nan_residual(a, b, ta, tb) & d;
        if (!nd) {
          st.nan = true;
          continue;
        }
        d = nd;
      }
      // Then the per-bit don't-care mask (heuristic; marks the result).
      if (undef && r * kWaveLanes + lane < undef->vgpr.size() &&
          !(d & ~undef->vgpr[size_t{r} * kWaveLanes + lane])) {
        note_mask(*undef, st, undef->vpassthrough.test(r));
        continue;
      }
      if (!lanes)
        lane0 = lane;
      ++lanes;
    }
    if (!lanes)
      continue;
    if (vdiff.none()) {
      first_r = r;
      first_lane = lane0;
      first_count = lanes;
    }
    vdiff.set(r);
  }
  // Memory: every byte either side stored.
  std::vector<uint64_t> mdiff;
  for (uint64_t x : differing_memory(final_old, final_new)) {
    uint8_t d = memory_byte(final_old, x) ^ memory_byte(final_new, x);
    if (!cfg.strict_nan && nan_unit_equal(final_old, final_new, x)) {
      st.nan = true;
      continue;
    }
    if (undef) {
      const auto it = undef->mem.find(x);
      if (it != undef->mem.end() && !(d & ~it->second)) {
        note_mask(*undef, st);
        continue;
      }
    }
    mdiff.push_back(x);
  }
  if (!mdiff.empty()) {
    m.memory = true;
    m.mem_addrs.assign(mdiff.begin(), mdiff.begin() + std::min<size_t>(mdiff.size(), 16));
  }
  // Inline asm: what matched `use` blocks observed.
  const std::string adiff = asm_use_difference(final_old, final_new, cfg, st);
  if (diff.none() && vdiff.none() && mdiff.empty() && !adiff.empty()) {
    m.detail = adiff;
    return "observable";
  }
  // Barriers: what other waves of the workgroup could observe, epoch by epoch.
  const std::string bdiff = barrier_difference(final_old, final_new);
  if (diff.none() && vdiff.none() && mdiff.empty() && !bdiff.empty()) {
    m.detail = bdiff;
    return "observable";
  }
  if (diff.none() && vdiff.none() && !mdiff.empty()) {
    const uint64_t x = mdiff.front();
    m.detail = "first memory difference at " + hex64(x) + ": old " +
               hex32(memory_byte(final_old, x)) + " new " + hex32(memory_byte(final_new, x)) +
               " (" + std::to_string(mdiff.size()) + " stored bytes differ)";
    return "observable";
  }
  if (diff.none() && vdiff.any()) {
    m.vlocs = vdiff;
    if (first_r < nv)
      m.detail = "first observable difference " + vgpr_name(first_r) + " lane " +
                 std::to_string(first_lane) + ": old " +
                 hex32(final_old.vget(first_r, first_lane)) + " new " +
                 hex32(final_new.vget(first_r, first_lane)) + " (" + std::to_string(first_count) +
                 " of 64 lanes differ)";
    else
      m.detail = "VGPR counts differ";
    return "observable";
  }
  if (diff.any()) {
    m.locs = diff;
    m.vlocs = vdiff;
    for (unsigned l = 0; l < kNumLocs; ++l)
      if (diff.test(l)) {
        m.detail = "first observable difference " + loc_name(l) + ": old " +
                   hex32(final_old.get(l)) + " new " +
                   (fwd[l] == l ? "" : "(" + loc_name(fwd[l]) + ") ") +
                   hex32(final_new.get(fwd[l]));
        break;
      }
    return "observable";
  }
  return {};
}

/// Alternative fillings of undefined registers: all zeros, all ones, then random. Every
/// bit of a value copied from an undefined register differs between the first two, and a
/// random one also exposes bits of derived values (x + 1, x & m, ...).
constexpr unsigned kUndefAlternatives = 3;

/// Refill @p regs (the wave's lanes) and @p locs with values of alternative @p alt
/// (1-based). Lanes and SGPRs the target does not have stay as they are (zero).
void refill_undef(MachineState &s, const VSet &regs, const LocSet &locs, uint64_t seed,
                  uint64_t case_index, unsigned alt, bool new_side, const Target &target) {
  for (unsigned l = 0; l < kNumLocs; ++l)
    if (locs.test(l) && (l < target.sgprs() || l >= kNumSgprs))
      s.set(l, alt == 1   ? 0u
               : alt == 2 ? 0xffffffffu
                          : canary_value(seed ^ (0x756e646566ull * alt), case_index, l, new_side));
  for (unsigned r = 0; r < s.vgpr_count; ++r)
    if (regs.test(r))
      for (unsigned lane = 0; lane < target.wave_size; ++lane)
        s.vset(r, lane,
               alt == 1   ? 0u
               : alt == 2 ? 0xffffffffu
                          : canary_value(seed ^ (0x756e646566ull * alt), case_index,
                                         kNumLocs + r * kWaveLanes + lane, new_side));
}

void merge_undef(UndefMask &m, const MachineState &a, const MachineState &b) {
  for (unsigned l = 0; l < kNumLocs; ++l)
    m.loc[l] |= a.loc[l] ^ b.loc[l];
  if (m.vgpr.size() != a.vgpr.size())
    m.vgpr.assign(a.vgpr.size(), 0);
  for (size_t i = 0; i < a.vgpr.size(); ++i)
    m.vgpr[i] |= a.vgpr[i] ^ (i < b.vgpr.size() ? b.vgpr[i] : 0u);
  for_each_stored(a, b, [&](uint64_t x, uint8_t va, uint8_t vb) {
    if (const uint8_t d = va ^ vb)
      m.mem[x] |= d;
  });
  if (m.asm_uses.size() < a.asm_uses.size())
    m.asm_uses.resize(a.asm_uses.size());
  for (size_t i = 0; i < a.asm_uses.size(); ++i) {
    auto &mu = m.asm_uses[i];
    mu.resize(a.asm_uses[i].values.size(), 0);
    const bool same = i < b.asm_uses.size() && b.asm_uses[i].block == a.asm_uses[i].block &&
                      b.asm_uses[i].values.size() == a.asm_uses[i].values.size();
    for (size_t k = 0; k < mu.size(); ++k)
      mu[k] |= same ? a.asm_uses[i].values[k] ^ b.asm_uses[i].values[k] : 0xffffffffu;
  }
  m.any = m.any || std::ranges::any_of(m.loc, [](uint32_t v) { return v != 0; }) ||
          std::ranges::any_of(m.vgpr, [](uint32_t v) { return v != 0; }) || !m.mem.empty() ||
          std::ranges::any_of(m.asm_uses, [](const auto &u) {
            return std::ranges::any_of(u, [](uint32_t v) { return v != 0; });
          });
}

/// The seed of NaN choice @p alt in case @p case_index. Old and new use the same one: the
/// runner keys each random NaN by the floating-point operation's position in execution
/// order, so the same operation draws the same NaN on both sides.
uint64_t nan_seed_for(uint64_t seed, uint64_t case_index, unsigned alt) {
  return splitmix64(seed ^ (case_index << 2) ^ alt);
}

/// Old's don't-care bits for one case: old re-runs kUndefAlternatives times, with its
/// undefined registers refilled and (when it produced a NaN, @p nan) every NaN a
/// floating-point instruction produces replaced by an alternative one. Output bits that
/// change depend on an undefined value or on a NaN's unspecified bits.
UndefMask compute_undef_mask(const RegionPair &region, RegionRunner &old_runner,
                             const MachineState &init_old, const MachineState &final_old,
                             uint64_t seed, uint64_t case_index, bool nan) {
  UndefMask m;
  const RegionContract &k = region.contract;
  const bool undef = k.vundef_old.any() || k.undef_old.any();
  if (!undef && !nan)
    return m;
  // Pass-through candidates: undefined inputs whose output equals the input.
  auto unchanged = [](const MachineState &in, const MachineState &out, LocSet &locs, VSet &regs) {
    for (unsigned l = 0; l < kNumLocs; ++l)
      if (locs.test(l) && in.get(l) != out.get(l))
        locs.reset(l);
    for (unsigned r = 0; r < kNumVgprs; ++r)
      if (regs.test(r)) {
        bool same = r < in.vgpr_count && r < out.vgpr_count;
        for (unsigned lane = 0; same && lane < kWaveLanes; ++lane)
          same = in.vget(r, lane) == out.vget(r, lane);
        if (!same)
          regs.reset(r);
      }
  };
  m.passthrough = k.undef_old;
  m.vpassthrough = k.vundef_old;
  unchanged(init_old, final_old, m.passthrough, m.vpassthrough);
  for (unsigned alt = 1; alt <= kUndefAlternatives; ++alt) {
    MachineState s = init_old;
    refill_undef(s, k.vundef_old, k.undef_old, seed, case_index, alt, false,
                 k.gpu().value_or(Target{}));
    old_runner.set_nan_choice(nan ? alt : 0, nan_seed_for(seed, case_index, alt));
    old_runner.load(s);
    const bool ok = old_runner.run(region.old_side.insts.size()).status == RunStatus::Ok;
    if (ok) {
      const MachineState alt = old_runner.capture();
      m.exit_varies = m.exit_varies || alt.exit_pc != final_old.exit_pc;
      merge_undef(m, final_old, alt);
      unchanged(s, alt, m.passthrough, m.vpassthrough);
    } else {
      m.passthrough.reset();
      m.vpassthrough.reset();
    }
  }
  old_runner.set_nan_choice(0, 0);
  m.from_undef = undef;
  m.from_nan = nan;
  return m;
}

/// New must not depend on an undefined value or a NaN's bits where old does not: new
/// re-runs with its own undefined registers refilled and alternative NaNs (@p nan), and
/// each run must still match old under the mask. Both sides draw alternative NaNs from the
/// same seed, keyed by the operation's position in execution order: when new runs the same
/// floating-point operations as old, it makes the same choices old's mask was built from.
/// On a mismatch, @p alt_init / @p alt_final hold the new side's run.
std::string check_new_undef(const RegionPair &region, RegionRunner &new_runner,
                            const MachineState &init_old, const MachineState &init_new,
                            const MachineState &final_old, const CompareConfig &cfg, uint64_t seed,
                            uint64_t case_index, bool nan, Mismatch &m, MachineState &alt_init,
                            MachineState &alt_final, CompareStats *stats = nullptr) {
  const RegionContract &k = region.contract;
  if (k.vundef_new.none() && k.undef_new.none() && !nan)
    return {};
  std::string kind;
  for (unsigned alt = 1; alt <= kUndefAlternatives && kind.empty(); ++alt) {
    MachineState s = init_new;
    refill_undef(s, k.vundef_new, k.undef_new, seed, case_index, alt, true,
                 k.gpu().value_or(Target{}));
    const uint64_t nan_seed = nan ? nan_seed_for(seed, case_index, alt) : 0;
    new_runner.set_nan_choice(nan ? alt : 0, nan_seed);
    new_runner.load(s);
    if (new_runner.run(region.new_side.insts.size()).status != RunStatus::Ok)
      continue;
    MachineState f = new_runner.capture();
    kind = compare_pair(region, init_old, s, final_old, f, m, cfg, stats);
    if (!kind.empty()) {
      m.new_nan_choice = nan ? alt : 0;
      m.new_nan_seed = nan_seed;
      m.detail = std::string("new depends on ") +
                 (nan ? "a NaN's unspecified bits or an undefined value"
                      : "an undefined value (" + format_vgprs(k.vundef_new) +
                            (k.undef_new.any() ? " " + format_locs(k.undef_new) : "") + ")") +
                 ": " + m.detail;
      alt_init = std::move(s);
      alt_final = std::move(f);
    }
  }
  new_runner.set_nan_choice(0, 0);
  return kind;
}

// ---------------------------------------------------------------------------
// Novelty (handoff 8): a bounded classification of an outcome. Raw register values never
// form a class; only the outcome kind and, per written location, a coarse value class
// and whether old and new agree.

unsigned value_class(uint32_t v) {
  if (v == 0)
    return 0;
  if (v == 1)
    return 1;
  if (v == 0xffffffffu)
    return 2;
  if (v & 0x80000000u)
    return 3;
  return 4;
}

uint64_t novelty_bucket(const RegionContract &k, const std::string &category,
                        const MachineState *final_old, const MachineState *final_new) {
  uint64_t h = splitmix64(0x6e6f76656c747900ull);
  for (char c : category)
    h = splitmix64(h ^ static_cast<uint8_t>(c));
  if (!final_old || !final_new)
    return h;
  const auto fwd = k.exit_forward();
  const auto inv = k.exit_inverse();
  LocSet watched = k.declared_writes_old;
  for (unsigned l = 0; l < kNumLocs; ++l)
    if (k.declared_writes_new.test(l))
      watched.set(inv[l]);
  watched.set(kLocScc);
  for (unsigned l = 0; l < kNumLocs; ++l) {
    if (!watched.test(l))
      continue;
    const unsigned eq = final_old->get(l) == final_new->get(fwd[l]) ? 1u : 0u;
    h = splitmix64(h ^ (uint64_t{l} << 8) ^ (value_class(final_old->get(l)) << 1) ^ eq);
  }
  const VSet vwatched = (k.vwrites_old | k.vwrites_new) & k.vobservable;
  for (unsigned r = 0; r < kNumVgprs; ++r)
    if (vwatched.test(r)) {
      const uint64_t lanes = differing_lanes(*final_old, *final_new, r);
      h = splitmix64(h ^ (uint64_t{r + kNumLocs} << 8) ^ (lanes == 0 ? 1u : 0u) ^
                     (lanes == ~uint64_t{0} ? 2u : 0u));
    }
  return h;
}

// ---------------------------------------------------------------------------
// Counterexample reduction (handoff 10). Shrink the inputs while the same kind of
// mismatch persists under the contract. The program is never reduced here: changing it
// would require re-planning the region and contract.

bool simpler(uint32_t candidate, uint32_t current) {
  const int pc = std::popcount(candidate), pv = std::popcount(current);
  return pc < pv || (pc == pv && candidate < current);
}

Json reduce_mismatch(const RegionPair &region, RegionRunner &old_runner, RegionRunner &new_runner,
                     Mismatch &m, size_t budget, uint64_t seed) {
  // The new side runs as it did when it showed the difference.
  new_runner.set_nan_choice(m.new_nan_choice, m.new_nan_seed);
  struct ResetNan {
    RegionRunner &r;
    ~ResetNan() { r.set_nan_choice(0, 0); }
  } reset_nan{new_runner};
  const RegionContract &k = region.contract;
  const auto fwd = k.entry_forward();
  const MachineState original_old = m.init_old, original_new = m.init_new;
  // A difference found under an alternative ("new depends on ...: ") keeps that cause.
  const size_t cause_end = m.detail.rfind("new depends on ", 0) == 0 ? m.detail.find(": ") : 0;
  const std::string cause =
      cause_end && cause_end != std::string::npos ? m.detail.substr(0, cause_end + 2) : "";
  const LocSet original_locs = m.locs;
  const VSet original_vlocs = m.vlocs;
  MachineState cur_o = m.init_old, cur_n = m.init_new;
  size_t runs = 0, steps = 0;
  bool seed_independent = false;
  auto attempt = [&](const MachineState &a, const MachineState &b) {
    if (runs >= budget)
      return false;
    ++runs;
    Sample s;
    for (unsigned l = 0; l < kNumLocs; ++l)
      s.value[l] = a.get(l);
    if (ConstraintChecker::validate(s, k))
      return false; // never leave the contract
    old_runner.load(a);
    const RunOutcome oo = old_runner.run(region.old_side.insts.size());
    new_runner.load(b);
    const RunOutcome on = new_runner.run(region.new_side.insts.size());
    if (oo.status != RunStatus::Ok || on.status != RunStatus::Ok)
      return false;
    if (!old_runner.poison_hit().empty() || !new_runner.poison_hit().empty())
      return false; // a poison input: excluded, never a reduction step
    const MachineState fa = old_runner.capture(), fb = new_runner.capture();
    const UndefMask umask = compute_undef_mask(region, old_runner, a, fa, seed, m.case_index,
                                               !m.strict_nan && old_runner.nan_produced());
    if (umask.exit_varies)
      return false; // old's control flow depends on an undefined value
    Mismatch t;
    const std::string kind =
        compare_pair(region, a, b, fa, fb, t, CompareConfig{m.strict_nan, &umask});
    if (kind != m.kind)
      return false;
    m.locs = t.locs;
    m.vlocs = t.vlocs;
    m.memory = t.memory;
    m.mem_addrs = t.mem_addrs;
    m.detail = cause + t.detail;
    m.frame_fields = t.frame_fields;
    m.init_old = a;
    m.init_new = b;
    m.final_old = fa;
    m.final_new = fb;
    m.out_old = oo;
    m.out_new = on;
    return true;
  };
  // 1. Seed independence: replace the seed-derived canaries with fixed distinct values.
  {
    MachineState a = cur_o, b = cur_n;
    const unsigned sgprs = k.gpu().value_or(Target{}).sgprs();
    for (unsigned l = 0; l < kNumLocs; ++l)
      if (k.canary.test(l) && (l < sgprs || l >= kNumSgprs)) { // not s102-s105 on GFX9
        a.set(l, 0);
        b.set(fwd[l], l == kLocScc ? 1u : 0xffffffffu);
      }
    const unsigned lanes = k.gpu().value_or(Target{}).wave_size;
    for (unsigned r = 0; r < k.vgpr_count; ++r)
      if (!k.ventry_equal.test(r))
        for (unsigned lane = 0; lane < lanes; ++lane) {
          a.vset(r, lane, 0);
          b.vset(r, lane, 0xffffffffu);
        }
    if (!(a == cur_o && b == cur_n) && attempt(a, b)) {
      cur_o = a;
      cur_n = b;
      seed_independent = true;
      ++steps;
    } else if (a == cur_o && b == cur_n) {
      seed_independent = true;
    }
  }
  // 1b. VALU: keep only the mismatching lanes. Every other lane of every shared VGPR
  // becomes 0, so the bundle shows the inputs of the lanes that differ.
  uint64_t focus = 0;
  if (k.vgpr_count && m.kind == "observable") {
    for (unsigned r = 0; r < kNumVgprs; ++r)
      if (m.vlocs.test(r))
        focus |= differing_lanes(m.final_old, m.final_new, r);
    if (focus && focus != ~uint64_t{0}) {
      MachineState a = cur_o, b = cur_n;
      for (unsigned r = 0; r < k.vgpr_count; ++r)
        if (k.ventry_equal.test(r))
          for (unsigned lane = 0; lane < kWaveLanes; ++lane)
            if (!(focus >> lane & 1u)) {
              a.vset(r, lane, 0);
              b.vset(r, lane, 0);
            }
      if (!(a == cur_o && b == cur_n) && attempt(a, b)) {
        cur_o = a;
        cur_n = b;
        ++steps;
      }
    }
  }
  // 2. Shrink each shared entry value: 0, 1, halvings, then clearing single bits.
  bool progress = true;
  for (int pass = 0; progress && pass < 8 && runs < budget; ++pass) {
    progress = false;
    for (unsigned l = 0; l < kNumLocs; ++l) {
      if (!k.entry_equal.test(l))
        continue;
      const uint32_t v = cur_o.get(l);
      if (v == 0)
        continue;
      std::vector<uint32_t> cands = {0u, 1u};
      for (uint32_t h = v >> 1; h; h >>= 1)
        cands.push_back(h);
      for (int bit = 31; bit >= 0; --bit)
        if (v & (1u << bit))
          cands.push_back(v & ~(1u << bit));
      for (uint32_t c : cands) {
        if (c == v || !simpler(c, v))
          continue;
        MachineState a = cur_o, b = cur_n;
        a.set(l, c);
        b.set(fwd[l], c);
        if (attempt(a, b)) {
          cur_o = a;
          cur_n = b;
          ++steps;
          progress = true;
          break;
        }
      }
    }
  }
  // 2b. VALU: shrink each shared VGPR at the first mismatching lane the same way.
  if (focus) {
    const unsigned lane = static_cast<unsigned>(std::countr_zero(focus));
    for (int pass = 0; pass < 4 && runs < budget; ++pass) {
      bool vprogress = false;
      for (unsigned r = 0; r < k.vgpr_count && runs < budget; ++r) {
        if (!k.ventry_equal.test(r))
          continue;
        const uint32_t v = cur_o.vget(r, lane);
        if (v == 0)
          continue;
        std::vector<uint32_t> cands = {0u, 1u};
        for (uint32_t h = v >> 1; h; h >>= 1)
          cands.push_back(h);
        for (int bit = 31; bit >= 0; --bit)
          if (v & (1u << bit))
            cands.push_back(v & ~(1u << bit));
        for (uint32_t c : cands) {
          if (c == v || !simpler(c, v))
            continue;
          MachineState a = cur_o, b = cur_n;
          a.vset(r, lane, c);
          b.vset(r, lane, c);
          if (attempt(a, b)) {
            cur_o = a;
            cur_n = b;
            ++steps;
            vprogress = true;
            break;
          }
        }
      }
      if (!vprogress)
        break;
    }
  }
  // attempt() leaves m at the last accepted state; make sure that is the current one.
  if (steps == 0) {
    m.init_old = original_old;
    m.init_new = original_new;
  }
  Json red = Json::object();
  red.set("runs", static_cast<uint64_t>(runs));
  red.set("accepted_steps", static_cast<uint64_t>(steps));
  red.set("budget_exhausted", runs >= budget);
  red.set("seed_independent", seed_independent);
  red.set("program_reduced", false);
  red.set("original_locations", format_locs(original_locs));
  if (original_vlocs.any())
    red.set("original_vgpr_locations", format_vgprs(original_vlocs));
  Json orig = Json::object();
  orig.set("old", original_old.to_json());
  orig.set("new", original_new.to_json());
  red.set("original_initial_state", std::move(orig));
  Json nonzero = Json::object();
  for (unsigned l = 0; l < kNumLocs; ++l)
    if (k.entry_equal.test(l) && cur_o.get(l) != 0)
      nonzero.set(loc_name(l), hex32(cur_o.get(l)));
  size_t listed = 0;
  for (unsigned r = 0; r < k.vgpr_count && listed < 64; ++r)
    if (k.ventry_equal.test(r))
      for (unsigned lane = 0; lane < kWaveLanes && listed < 64; ++lane)
        if (cur_o.vget(r, lane) != 0) {
          nonzero.set(vgpr_name(r) + "[" + std::to_string(lane) + "]", hex32(cur_o.vget(r, lane)));
          ++listed;
        }
  red.set("reduced_entry_values", std::move(nonzero));
  return red;
}

// ---------------------------------------------------------------------------
// Per-case learning log (handoff 8).

Json summary_json(const std::vector<InstSummary> &insts) {
  Json arr = Json::array();
  for (const auto &i : insts) {
    Json o = Json::object();
    o.set("pc", hex64(i.pc));
    o.set("mnemonic", i.mnemonic);
    o.set("asm", i.text);
    o.set("reads", format_locs(i.reads));
    o.set("writes", format_locs(i.writes));
    if (i.vreads.any())
      o.set("vreads", format_vgprs(i.vreads));
    if (i.vwrites.any())
      o.set("vwrites", format_vgprs(i.vwrites));
    arr.push(std::move(o));
  }
  return arr;
}

Json log_header(const RegionPair &region, const RegionFeatures &f, const TestOptions &options,
                const std::string &policy, size_t sweep_cases) {
  Json h = Json::object();
  h.set("type", "header");
  h.set("schema", "rj_isa_diff.case_log.v1");
  h.set("region", region.id);
  if (!region.parent.empty())
    h.set("parent", region.parent);
  h.set("synthetic", region.synthetic);
  h.set("prepared_key", region.prepared_key);
  h.set("seed", options.seed);
  h.set("policy", policy);
  Json k = Json::object();
  k.set("status", contract_status_name(region.contract.status));
  k.set("entry_equal", format_locs(region.contract.entry_equal));
  k.set("observable", format_locs(region.contract.observable));
  k.set("pending_effects", region.contract.pending_effects);
  k.set("composition", region.contract.composition);
  h.set("contract", std::move(k));
  h.set("old", summary_json(f.roles.old_summary));
  h.set("new", summary_json(f.roles.new_summary));
  Json roles = Json::object();
  for (unsigned l = 0; l < kNumLocs; ++l)
    if (f.roles.roles[l])
      roles.set(loc_name(l), role_names(f.roles.roles[l]));
  h.set("roles", std::move(roles));
  Json pairs = Json::array();
  for (const auto &p : f.roles.pairs)
    pairs.push(p.mnemonic + " " + loc_name(p.a) + "," + loc_name(p.b) +
               (p.wide ? " (64-bit)" : ""));
  h.set("pairs", std::move(pairs));
  Json rel = Json::array();
  for (const auto &c : f.roles.const_relations)
    rel.push(c.mnemonic + " " + loc_name(c.loc) + " vs " + hex32(c.constant));
  h.set("const_relations", std::move(rel));
  h.set("boundary_cases", static_cast<uint64_t>(sweep_cases));
  return h;
}

Json feedback_json(const Feedback &fb) {
  Json o = Json::object();
  o.set("type", "case");
  o.set("case", fb.case_index);
  o.set("phase", fb.phase);
  o.set("accepted", fb.accepted);
  if (!fb.rejection.empty())
    o.set("rejection", fb.rejection);
  Json ev = Json::array();
  for (const auto &e : fb.events)
    ev.push(e);
  o.set("events", std::move(ev));
  if (fb.accepted) {
    Json ex = Json::array();
    ex.push(fb.exit_old);
    ex.push(fb.exit_new);
    o.set("exit", std::move(ex));
    o.set("category", fb.category);
    if (fb.locs.any())
      o.set("locs", format_locs(fb.locs));
    if (fb.vlocs.any())
      o.set("vlocs", format_vgprs(fb.vlocs));
    o.set("bucket", hex64(fb.bucket));
    o.set("novel", fb.novel);
    Json cost = Json::array();
    for (uint64_t c : {fb.propose_ns, fb.load_ns, fb.old_ns, fb.new_ns, fb.compare_ns})
      cost.push(c);
    o.set("cost_ns", std::move(cost)); // propose, load, old, new, compare
    Json insts = Json::array();
    insts.push(fb.insts_old);
    insts.push(fb.insts_new);
    o.set("insts", std::move(insts));
  }
  return o;
}

} // namespace

const char *test_status_name(TestStatus s) {
  switch (s) {
  case TestStatus::NoMismatchFound:
    return "NO_MISMATCH_FOUND";
  case TestStatus::LocalMismatch:
    return "LOCAL_MISMATCH";
  case TestStatus::Inconclusive:
    return "INCONCLUSIVE";
  case TestStatus::Unsupported:
    return "UNSUPPORTED";
  case TestStatus::HarnessError:
    return "HARNESS_ERROR";
  case TestStatus::NoSemanticDiff:
    return "NO_SEMANTIC_DIFF";
  }
  return "?";
}

const char *witness_scope_name(WitnessScope s) {
  switch (s) {
  case WitnessScope::None:
    return "NONE";
  case WitnessScope::LocalOnly:
    return "LOCAL_ONLY";
  case WitnessScope::ReachableAtCut:
    return "REACHABLE_AT_CUT";
  case WitnessScope::ObservableKernelDivergence:
    return "OBSERVABLE_KERNEL_DIVERGENCE";
  }
  return "?";
}

const char *coverage_scope_name(CoverageScope s) {
  switch (s) {
  case CoverageScope::CompleteForDeclaredScope:
    return "COMPLETE_FOR_DECLARED_SCOPE";
  case CoverageScope::Partial:
    return "PARTIAL";
  case CoverageScope::NotEstablished:
    return "NOT_ESTABLISHED";
  }
  return "?";
}

Json build_identity() {
  Json j = Json::object();
  j.set("tool", kToolName);
  j.set("tool_version", kToolVersion);
  j.set("schema_version", kSchemaVersion);
  j.set("rocjitsu_revision", RJ_ISA_DIFF_RJ_REVISION);
  j.set("rocjitsu_version", RJ_ISA_DIFF_RJ_VERSION);
  j.set("semantics", "RocJITsu ComputeUnitCore::execute_instruction on the contract's target "
                     "(ISA family, wave size), default arithmetic semantics");
#if defined(__VERSION__)
  j.set("compiler", __VERSION__);
#endif
  return j;
}

void build_initial_states(const RegionContract &contract, const Sample &sample, const Frame &frame,
                          uint64_t seed, uint64_t case_index, MachineState &old_state,
                          MachineState &new_state) {
  old_state = {};
  new_state = {};
  old_state.frame = frame;
  new_state.frame = frame;
  // Sample and contract sets use old naming; the new side receives each value at map(l)
  // (identity without a register map). entry_forward() is a permutation, so every new
  // location is initialised exactly once.
  const auto fwd = contract.entry_forward();
  for (unsigned l = 0; l < kNumLocs; ++l) {
    if (contract.entry_equal.test(l)) {
      old_state.set(l, sample.value[l]);
      new_state.set(fwd[l], sample.value[l]);
    } else {
      old_state.set(l, canary_value(seed, case_index, l, /*new_side=*/false));
      new_state.set(fwd[l], canary_value(seed, case_index, l, /*new_side=*/true));
    }
  }
  // VGPRs (VALU support): shared lanes for vgpr entry_equal, per-side canaries otherwise.
  old_state.resize_vgprs(contract.vgpr_count);
  new_state.resize_vgprs(contract.vgpr_count);
  const bool have_lanes = sample.vvalue.size() == size_t{contract.vgpr_count} * kWaveLanes;
  for (unsigned r = 0; r < contract.vgpr_count; ++r)
    for (unsigned lane = 0; lane < kWaveLanes; ++lane) {
      if (contract.ventry_equal.test(r)) {
        uint32_t v = have_lanes ? sample.vvalue[size_t{r} * kWaveLanes + lane] : 0u;
        for (const auto &[mr, mask] : contract.vinput_masks)
          if (mr == r)
            v &= mask;
        old_state.vset(r, lane, v);
        new_state.vset(r, lane, v);
      } else {
        const unsigned id = kNumLocs + r * kWaveLanes + lane;
        old_state.vset(r, lane, canary_value(seed, case_index, id, false));
        new_state.vset(r, lane, canary_value(seed, case_index, id, true));
      }
    }
  // Memory model: both sides start from the same initial memory content.
  if (contract.memory_reads || contract.memory_writes) {
    const uint64_t mseed = splitmix64(seed ^ splitmix64(case_index ^ 0x6d656d6f7279ull)) | 1u;
    old_state.mem_seed = mseed;
    new_state.mem_seed = mseed;
  }
  // Inline asm: matched def blocks write the same per-case values on both sides.
  if (contract.inline_asm) {
    const uint64_t aseed = splitmix64(seed ^ splitmix64(case_index ^ 0x696e6c696e65ull)) | 1u;
    old_state.asm_seed = aseed;
    new_state.asm_seed = aseed;
  }
  // Each side starts in its own kernel's MODE (FP rounding/denormals, DX10_CLAMP, IEEE).
  old_state.frame.mode = contract.mode_old;
  new_state.frame.mode = contract.mode_new;
  // What the target does not have stays zero: s102-s105 on GFX9, lanes 32-63 on wave32.
  const Target t = contract.gpu().value_or(Target{});
  for (unsigned l = t.sgprs(); l < kNumSgprs; ++l) {
    old_state.set(l, 0);
    new_state.set(l, 0);
  }
  for (unsigned r = 0; r < contract.vgpr_count; ++r)
    for (unsigned lane = t.wave_size; lane < kWaveLanes; ++lane) {
      old_state.vset(r, lane, 0);
      new_state.vset(r, lane, 0);
    }
}

Json RegionResult::to_json() const {
  if (!cached.is_null())
    return cached;
  Json j = Json::object();
  j.set("region", region_id);
  j.set("synthetic", synthetic);
  j.set("test_status", test_status_name(test_status));
  j.set("contract_status", contract_status_name(contract_status));
  j.set("witness_scope", witness_scope_name(witness_scope));
  j.set("coverage_scope", coverage_scope_name(coverage_scope));
  j.set("proof_status", "NOT_RUN");
  if (!reason.empty())
    j.set("reason", reason);
  j.set("prepared_key", prepared_key);
  if (!pending_effects.empty())
    j.set("pending_effects", pending_effects);
  if (!composition.empty())
    j.set("composition", composition);
  if (!resolution.empty()) {
    Json e = Json::object();
    e.set("resolution", resolution);
    if (!resolved_by.empty()) {
      e.set("resolved_by", resolved_by);
      e.set("resolved_contract_status", contract_status_name(resolved_contract));
      e.set("resolved_coverage_scope", coverage_scope_name(resolved_coverage));
    }
    e.set("tried", expansion_tried);
    j.set("expansion", std::move(e));
  }
  j.set("policy", policy);
  j.set("seed", seed);
  Json cmp = Json::object();
  cmp.set("nan", strict_nan ? "strict" : "llvm");
  cmp.set("poison_filter", format_poison_kinds(poison_filter));
  j.set("comparison", std::move(cmp));
  Json counts = Json::object();
  counts.set("attempted", attempted);
  counts.set("accepted", accepted);
  counts.set("executed_pairs", executed_pairs);
  counts.set("valid_pairs", valid_pairs);
  counts.set("mismatches", mismatches);
  counts.set("harness_errors", harness_errors);
  counts.set("step_limit_divergences", step_limit_divergences);
  counts.set("step_limit_reruns", step_limit_reruns);
  counts.set("nan_tolerated", nan_tolerated);
  counts.set("undef_masked", undef_masked);
  counts.set("dont_care_masked", dont_care_masked);
  Json rej = Json::object();
  for (const auto &[k, v] : rejections)
    rej.set(k, v);
  counts.set("rejections", std::move(rej));
  Json mk = Json::object();
  for (const auto &[k, v] : mismatch_kinds)
    mk.set(k, v);
  counts.set("mismatch_kinds", std::move(mk));
  j.set("counts", std::move(counts));
  auto phase_json = [](const PhaseCounts &p) {
    Json o = Json::object();
    o.set("cases", p.cases);
    o.set("accepted", p.accepted);
    o.set("valid", p.valid);
    o.set("mismatches", p.mismatches);
    return o;
  };
  Json bj = phase_json(boundary);
  bj.set("mandatory", true);
  bj.set("generated", boundary_generated);
  bj.set("duplicates_dropped", boundary_duplicates);
  bj.set("truncated", boundary_truncated);
  Json fam = Json::object();
  for (const auto &[k, v] : boundary_families)
    fam.set(k, v);
  bj.set("valid_by_family", std::move(fam));
  j.set("boundary_sweep", std::move(bj));
  j.set("random_phase", phase_json(random));
  Json nov = Json::object();
  nov.set("distinct_buckets", novelty_buckets);
  nov.set("first_seen_in_boundary", novel_boundary);
  nov.set("first_seen_in_random", novel_random);
  j.set("novelty", std::move(nov));
  if (!case_log.empty())
    j.set("case_log", case_log);
  if (!unrestricted.is_null())
    j.set("unrestricted", unrestricted);
  Json exec = Json::object();
  exec.set("old_region_insts", static_cast<uint64_t>(old_len));
  exec.set("new_region_insts", static_cast<uint64_t>(new_len));
  exec.set("old_insts_executed", old_insts_executed);
  exec.set("new_insts_executed", new_insts_executed);
  j.set("execution", std::move(exec));
  Json tj = Json::object();
  tj.set("setup_ns", t.setup_ns);
  tj.set("propose_ns", t.propose_ns);
  tj.set("state_load_ns", t.load_ns);
  tj.set("old_execute_ns", t.old_exec_ns);
  tj.set("new_execute_ns", t.new_exec_ns);
  tj.set("compare_ns", t.compare_ns);
  tj.set("triage_ns", t.triage_ns);
  j.set("timing", std::move(tj));
  if (mismatch_locs.any() || mismatch_vlocs.any() || mismatch_memory) {
    j.set("mismatch_locations", format_locs(mismatch_locs));
    j.set("vgpr_mismatch_locations", format_vgprs(mismatch_vlocs));
    if (mismatch_memory)
      j.set("memory_mismatch", true);
  }
  if (compensable)
    j.set("compensable_downstream", true);
  if (!examples.empty()) {
    const Mismatch &m = examples.front();
    Json first = Json::object();
    first.set("case_index", m.case_index);
    first.set("phase", m.phase);
    first.set("kind", m.kind);
    first.set("locations", format_locs(m.locs));
    if (m.vlocs.any())
      first.set("vgpr_locations", format_vgprs(m.vlocs));
    first.set("detail", m.detail);
    j.set("first_mismatch", std::move(first));
  }
  Json files = Json::array();
  for (const auto &f : counterexample_files)
    files.push(f);
  j.set("counterexamples", std::move(files));
  return j;
}

Json counterexample_json(const RegionPair &region, const Mismatch &m, uint64_t seed,
                         const std::string &policy, const std::vector<TraceStep> &trace_old,
                         const std::vector<TraceStep> &trace_new) {
  Json j = Json::object();
  j.set("schema_version", "rj_isa_diff.counterexample.v1");
  j.set("synthetic", region.synthetic);
  j.set("evidence", region.synthetic
                        ? "synthetic mismatch from injected mutation; not an LLVM finding"
                        : "region-level reproducer under the recorded contract; kernel/context "
                          "reproducer PENDING (reachability not established in P0)");
  j.set("witness_scope", "LOCAL_ONLY");
  j.set("region_id", region.id);
  j.set("prepared_key", region.prepared_key);
  j.set("seed", seed);
  j.set("policy", policy);
  j.set("case_index", m.case_index);
  j.set("phase", m.phase);
  j.set("kind", m.kind);
  j.set("nan_policy", m.strict_nan ? "strict" : "llvm");
  if (m.new_nan_choice) {
    Json nc = Json::object();
    nc.set("side", "new");
    nc.set("alternative", static_cast<uint64_t>(m.new_nan_choice));
    nc.set("seed", hex64(m.new_nan_seed));
    j.set("nan_choice", std::move(nc));
  }
  if (m.poison_filter)
    j.set("poison_filter", format_poison_kinds(m.poison_filter));
  j.set("locations", format_locs(m.locs));
  if (m.vlocs.any())
    j.set("vgpr_locations", format_vgprs(m.vlocs));
  if (m.memory) {
    j.set("memory_differs", true);
    Json md = Json::array();
    for (uint64_t x : m.mem_addrs) {
      Json e = Json::object();
      e.set("address", hex64(x));
      e.set("old", hex32(memory_byte(m.final_old, x)));
      e.set("new", hex32(memory_byte(m.final_new, x)));
      md.push(std::move(e));
    }
    j.set("memory_differences", std::move(md));
  }
  j.set("detail", m.detail);
  // Observable locations use old naming (the new side is read at map(l)); effect-model
  // locations are physical on their own side.
  const auto fwd = region.contract.exit_forward();
  const bool observable = m.kind == "observable";
  Json firsts = Json::array();
  for (unsigned l = 0; l < kNumLocs; ++l) {
    if (!m.locs.test(l))
      continue;
    const unsigned nl = observable ? fwd[l] : l;
    Json d = Json::object();
    d.set("loc", loc_name(l));
    if (nl != l)
      d.set("new_loc", loc_name(nl));
    d.set("old", hex32(m.final_old.get(l)));
    d.set("new", hex32(m.final_new.get(nl)));
    firsts.push(std::move(d));
  }
  for (unsigned r = 0; r < kNumVgprs; ++r) {
    if (!m.vlocs.test(r))
      continue;
    uint64_t lanes = differing_lanes(m.final_old, m.final_new, r);
    for (unsigned listed = 0; lanes && listed < 8; ++listed) {
      const unsigned lane = static_cast<unsigned>(std::countr_zero(lanes));
      lanes &= lanes - 1;
      Json d = Json::object();
      d.set("loc", vgpr_name(r));
      d.set("lane", lane);
      d.set("old", hex32(m.final_old.vget(r, lane)));
      d.set("new", hex32(m.final_new.vget(r, lane)));
      firsts.push(std::move(d));
    }
  }
  j.set("differences", std::move(firsts));
  Json events = Json::array();
  for (const auto &e : m.events)
    events.push(e);
  j.set("input_events", std::move(events));
  j.set("region", region.to_json());
  Json init = Json::object();
  init.set("old", m.init_old.to_json());
  init.set("new", m.init_new.to_json());
  j.set("initial_state", std::move(init));
  Json fin = Json::object();
  fin.set("old", m.final_old.to_json());
  fin.set("new", m.final_new.to_json());
  j.set("final_state", std::move(fin));
  Json exit = Json::object();
  exit.set("old", outcome_json(m.out_old));
  exit.set("new", outcome_json(m.out_new));
  j.set("exit", std::move(exit));
  Json tr = Json::object();
  tr.set("old", trace_json(trace_old));
  tr.set("new", trace_json(trace_new));
  j.set("trace", std::move(tr));
  if (!m.reduction.is_null())
    j.set("reduction", m.reduction);
  j.set("identity", build_identity());
  return j;
}

RegionResult test_region(const RegionPair &region, const TestOptions &options,
                         InputPolicy &policy) {
  RegionResult r;
  r.region_id = region.id;
  r.synthetic = region.synthetic;
  r.contract_status = region.contract.status;
  r.prepared_key = region.prepared_key;
  r.pending_effects = region.contract.pending_effects;
  r.composition = region.contract.composition;
  r.policy = policy.name();
  r.seed = options.seed;
  r.old_len = region.old_side.insts.size();
  r.new_len = region.new_side.insts.size();
  if (region.status != RegionStatus::Ready) {
    r.test_status = region.status == RegionStatus::Unsupported ? TestStatus::Unsupported
                                                               : TestStatus::Inconclusive;
    r.reason = region.reason;
    return r;
  }

  auto setup_start = Clock::now();
  std::string error;
  const unsigned vgprs = region.contract.vgpr_count;
  const auto target = region.contract.gpu();
  if (!target) {
    r.test_status = TestStatus::HarnessError;
    r.reason = "unsupported contract target " + region.contract.target + "/" + region.contract.wave;
    return r;
  }
  auto old_runner =
      RegionRunner::create(region.old_side.insts, region.id + "_old", error, vgprs, *target);
  auto new_runner = old_runner ? RegionRunner::create(region.new_side.insts, region.id + "_new",
                                                      error, vgprs, *target)
                               : nullptr;
  if (old_runner && new_runner) {
    auto image = std::make_shared<MemoryImage>();
    image->set_small_scalars(old_runner->loops() || new_runner->loops());
    share_code_guard(*old_runner, *new_runner);
    old_runner->set_memory_image(image);
    new_runner->set_memory_image(image);
    old_runner->set_poison_filter(options.poison_filter);
    new_runner->set_poison_filter(options.poison_filter);
    old_runner->set_memory_dead_at_exit(region.contract.memory_dead_at_exit);
    new_runner->set_memory_dead_at_exit(region.contract.memory_dead_at_exit);
  }
  r.strict_nan = options.strict_nan;
  r.poison_filter = options.poison_filter;
  r.t.setup_ns = ns_since(setup_start);
  if (!old_runner || !new_runner) {
    r.test_status = TestStatus::HarnessError;
    r.reason = error;
    return r;
  }
  if (!(old_runner->baseline_frame() == new_runner->baseline_frame())) {
    r.test_status = TestStatus::HarnessError;
    r.reason = "old and new runners start from different frames";
    return r;
  }
  const Frame frame = old_runner->baseline_frame();

  const auto new_to_old = region.contract.entry_inverse();
  const RegionFeatures features =
      make_region_features(region.old_side.insts, region.new_side.insts,
                           region.contract.entry_map.empty() ? nullptr : &new_to_old, *target);
  auto sweep_start = Clock::now();
  const BoundarySweep sweep = build_boundary_sweep(features.roles, region.contract,
                                                   {options.seed, options.max_boundary_cases});
  r.t.propose_ns += ns_since(sweep_start);
  r.boundary_generated = sweep.generated;
  r.boundary_duplicates = sweep.duplicates;
  r.boundary_truncated = sweep.truncated;

  std::map<std::string, uint64_t> unsupported_exclusions;
  uint64_t undefined = 0;
  MachineState init_old, init_new;
  History history;
  std::unordered_set<uint64_t> buckets;

  // Optional per-case learning log (handoff 8).
  std::ofstream log;
  if (!options.case_log_dir.empty()) {
    std::filesystem::create_directories(options.case_log_dir);
    r.case_log = options.case_log_dir + "/" + region.id + ".jsonl";
    log.open(r.case_log, std::ios::trunc);
    if (log)
      log << log_header(region, features, options, policy.name(), sweep.cases.size()).dump(-1)
          << '\n';
    else
      r.case_log.clear();
  }
  auto finish = [&](Feedback &fb) {
    policy.observe(fb);
    history.record(fb);
    if (log)
      log << feedback_json(fb).dump(-1) << '\n';
  };

  // Execute one validated sample on both sides and classify the outcome.
  auto run_case = [&](const Sample &sample, uint64_t case_index, PhaseCounts &phase,
                      const char *phase_name, const std::string &family, Feedback &fb) {
    ++r.accepted;
    ++phase.accepted;
    fb.accepted = true;
    auto t1 = Clock::now();
    build_initial_states(region.contract, sample, frame, options.seed, case_index, init_old,
                         init_new);
    old_runner->set_loop_scale(1);
    new_runner->set_loop_scale(1);
    old_runner->load(init_old);
    new_runner->load(init_new);
    fb.load_ns = ns_since(t1);
    r.t.load_ns += fb.load_ns;

    auto t2 = Clock::now();
    RunOutcome out_old = old_runner->run(region.old_side.insts.size());
    fb.old_ns = ns_since(t2);
    r.t.old_exec_ns += fb.old_ns;
    auto t3 = Clock::now();
    RunOutcome out_new = new_runner->run(region.new_side.insts.size());
    fb.new_ns = ns_since(t3);
    r.t.new_exec_ns += fb.new_ns;
    if (r.step_limit_divergences < kStepLimitReruns) {
      // A slower loop rather than a hang: the looping side gets a larger limit.
      auto tr = Clock::now();
      if (rerun_past_step_limit(*old_runner, *new_runner, init_old, init_new,
                                region.old_side.insts.size(), region.new_side.insts.size(), out_old,
                                out_new)) {
        ++r.step_limit_reruns;
        const uint64_t ns = ns_since(tr);
        (old_runner->loop_scale() > 1 ? fb.old_ns : fb.new_ns) += ns;
        (old_runner->loop_scale() > 1 ? r.t.old_exec_ns : r.t.new_exec_ns) += ns;
      }
    }
    ++r.executed_pairs;
    r.old_insts_executed += out_old.executed;
    r.new_insts_executed += out_new.executed;
    fb.insts_old = out_old.executed;
    fb.insts_new = out_new.executed;
    fb.exit_old = run_status_name(out_old.status);
    fb.exit_new = run_status_name(out_new.status);

    auto t4 = Clock::now();
    Mismatch m;
    m.case_index = case_index;
    m.phase = phase_name;
    m.out_old = out_old;
    m.out_new = out_new;
    std::string kind;
    std::string excluded_as = "excluded";
    auto note_novelty = [&](const MachineState *fo, const MachineState *fn) {
      fb.bucket = novelty_bucket(region.contract, fb.category, fo, fn);
      fb.novel = buckets.insert(fb.bucket).second;
      if (fb.novel)
        ++(std::string_view(phase_name) == "boundary" ? r.novel_boundary : r.novel_random);
    };
    const bool fault_old = out_old.status == RunStatus::MemoryFault;
    const bool fault_new = out_new.status == RunStatus::MemoryFault;
    if (fault_new && !fault_old && out_old.status == RunStatus::Ok) {
      // New faults where old completes: the change introduced a fault. (Old faulting where
      // new completes is an input outside old's defined behaviour, excluded below.)
      ++r.valid_pairs;
      ++phase.valid;
      kind = "exit_status_divergence";
      m.kind = kind;
      m.detail = "new side: " + out_new.detail + "; the old side completed";
      m.final_old = old_runner->capture();
      m.final_new = new_runner->capture();
      ++r.mismatches;
      ++phase.mismatches;
      fb.compare_ns = ns_since(t4);
      r.t.compare_ns += fb.compare_ns;
      fb.mismatch = true;
      fb.category = kind;
      note_novelty(nullptr, nullptr);
    } else if (out_old.status != RunStatus::Ok || out_new.status != RunStatus::Ok) {
      const std::string key = std::string("exit: old ") + run_status_name(out_old.status) +
                              " / new " + run_status_name(out_new.status);
      ++r.rejections[key];
      if (is_harness_status(out_old.status) || is_harness_status(out_new.status)) {
        ++r.harness_errors;
        kind = "harness_exit";
      } else if (is_unsupported_status(out_old.status) || is_unsupported_status(out_new.status)) {
        ++unsupported_exclusions[key];
      } else if (out_old.status == RunStatus::UndefinedBehavior ||
                 out_new.status == RunStatus::UndefinedBehavior) {
        ++undefined;
      } else if (fault_old && fault_new) {
        // An input outside the memory model's address domain: excluded, counted.
        ++r.rejections["memory fault on both sides (excluded input): " + out_old.detail];
      } else if (fault_old && out_new.status == RunStatus::Ok) {
        // Old faults where new completes: a fault is an access LLVM leaves undefined (outside
        // the allocation, wrapping, misaligned), so the input is outside old's defined
        // behaviour and new may do anything there (as with reported undefined behaviour).
        ++r.rejections["memory fault on the old side only (excluded input): " + out_old.detail];
      } else if (one_sided_step_limit(out_old, out_new)) {
        // One side exits, the other still loops past kStepLimitRerunScale times the limit:
        // possibly a non-termination the change introduced or removed. Not a witness (the
        // looping side might finish later), and never offset by other cases: the region
        // is inconclusive.
        ++r.step_limit_divergences;
        if (r.step_limit_first.empty())
          r.step_limit_first = "case " + std::to_string(case_index) + " (" + phase_name +
                               "): old " + run_status_name(out_old.status) + " after " +
                               std::to_string(out_old.executed) + " insts, new " +
                               run_status_name(out_new.status) + " after " +
                               std::to_string(out_new.executed) + " insts";
        excluded_as = "step_limit_divergence";
      } else if (is_excluded_status(out_old.status) || is_excluded_status(out_new.status)) {
        // Both sides past the step limit: excluded (outside the declared scope, like
        // inputs that fault on both sides), never read as non-termination.
        ++r.rejections["loop step limit (excluded input)"];
      }
      fb.compare_ns = ns_since(t4);
      r.t.compare_ns += fb.compare_ns;
      fb.category = kind.empty() ? excluded_as : kind;
      note_novelty(nullptr, nullptr);
      if (kind.empty())
        return;
      m.kind = kind;
      m.detail = key + ": " + out_old.detail + " " + out_new.detail;
    } else if (!old_runner->poison_hit().empty() || !new_runner->poison_hit().empty()) {
      // An input that is poison in the IR (--poison-filter): any result is allowed.
      const std::string &hit =
          old_runner->poison_hit().empty() ? new_runner->poison_hit() : old_runner->poison_hit();
      ++r.rejections["poison input (excluded): " + hit.substr(0, hit.find(':'))];
      fb.compare_ns = ns_since(t4);
      r.t.compare_ns += fb.compare_ns;
      fb.category = "excluded";
      note_novelty(nullptr, nullptr);
      return;
    } else {
      ++r.valid_pairs;
      ++phase.valid;
      if (!family.empty())
        ++r.boundary_families[family];
      const MachineState final_old = old_runner->capture();
      MachineState final_new = new_runner->capture();
      const bool nan_old = !options.strict_nan && old_runner->nan_produced();
      const bool nan_new = !options.strict_nan && new_runner->nan_produced();
      const UndefMask umask = compute_undef_mask(region, *old_runner, init_old, final_old,
                                                 options.seed, case_index, nan_old);
      if (umask.exit_varies) {
        // Old's branch depends on an undefined value (or a NaN's bits): undefined in LLVM,
        // so any behaviour of new is allowed. Excluded, counted.
        --r.valid_pairs;
        --phase.valid;
        ++r.rejections["branch on an undefined value (excluded input)"];
        fb.compare_ns = ns_since(t4);
        r.t.compare_ns += fb.compare_ns;
        fb.category = "excluded";
        note_novelty(nullptr, nullptr);
        return;
      }
      const CompareConfig cfg{options.strict_nan, &umask};
      CompareStats cst;
      kind = compare_pair(region, init_old, init_new, final_old, final_new, m, cfg, &cst);
      if (kind.empty()) {
        MachineState alt_init, alt_final;
        kind = check_new_undef(region, *new_runner, init_old, init_new, final_old, cfg,
                               options.seed, case_index, nan_new, m, alt_init, alt_final, &cst);
        if (!kind.empty()) {
          init_new = std::move(alt_init);
          final_new = std::move(alt_final);
        }
      }
      if (kind.empty()) {
        r.nan_tolerated += cst.nan;
        r.undef_masked += cst.undef;
        r.dont_care_masked += cst.masked;
      }
      fb.compare_ns = ns_since(t4);
      r.t.compare_ns += fb.compare_ns;
      fb.mismatch = !kind.empty();
      fb.category = kind.empty() ? "ok" : kind;
      fb.locs = m.locs;
      fb.vlocs = m.vlocs;
      note_novelty(&final_old, &final_new);
      if (kind.empty())
        return;
      m.kind = kind;
      m.final_old = final_old;
      m.final_new = final_new;
      if (kind == "effect_model" || kind == "frame_preservation") {
        ++r.harness_errors;
      } else {
        ++r.mismatches;
        ++phase.mismatches;
        r.mismatch_locs |= m.locs;
        r.mismatch_vlocs |= m.vlocs;
        r.mismatch_memory = r.mismatch_memory || m.memory;
      }
    }
    ++r.mismatch_kinds[kind];
    m.init_old = init_old;
    m.init_new = init_new;
    m.strict_nan = options.strict_nan;
    m.poison_filter = options.poison_filter;
    if (!m.final_old.mem_init.empty()) {
      m.init_old.mem_init = m.final_old.mem_init;
      m.init_new.mem_init = m.final_old.mem_init;
    }
    m.events = sample.events;
    if (r.examples.size() < options.max_counterexamples) {
      auto t5 = Clock::now();
      // Shrink the inputs first (handoff 10), so bundles and traces show the reduced case.
      if (options.reduce && kind == "observable")
        m.reduction = reduce_mismatch(region, *old_runner, *new_runner, m, options.reduction_budget,
                                      options.seed);
      if (!options.counterexample_dir.empty()) {
        // Re-execute with tracing on the same runners; determinism makes this the same run.
        std::vector<TraceStep> trace_old, trace_new;
        old_runner->load(m.init_old);
        old_runner->run(region.old_side.insts.size(), &trace_old);
        new_runner->set_nan_choice(m.new_nan_choice, m.new_nan_seed);
        new_runner->load(m.init_new);
        new_runner->run(region.new_side.insts.size(), &trace_new);
        new_runner->set_nan_choice(0, 0);
        Json bundle =
            counterexample_json(region, m, options.seed, policy.name(), trace_old, trace_new);
        if (options.artifact_info.has("artifacts"))
          bundle.set("artifacts", options.artifact_info["artifacts"]);
        if (options.artifact_info.has("provenance"))
          bundle.set("provenance", options.artifact_info["provenance"]);
        std::filesystem::create_directories(options.counterexample_dir);
        const std::string path = options.counterexample_dir + "/" + region.id + "-case" +
                                 std::to_string(case_index) + ".json";
        if (write_text_file(path, bundle.dump()))
          r.counterexample_files.push_back(path);
      }
      r.examples.push_back(std::move(m));
      r.t.triage_ns += ns_since(t5);
    }
  };

  // Phase 1: the boundary sweep always runs in full (exact values; never projected).
  uint64_t case_index = 0;
  auto stopped = [&] {
    return options.stop_at_first_mismatch && (r.mismatches || r.harness_errors);
  };
  for (const SweepCase &c : sweep.cases) {
    if (stopped())
      break;
    auto t0 = Clock::now();
    const Sample sample = sweep.materialize(c);
    ++r.attempted;
    ++r.boundary.cases;
    Feedback fb;
    fb.case_index = case_index;
    fb.phase = "boundary";
    fb.events = sample.events;
    auto rejection = ConstraintChecker::validate(sample, region.contract);
    fb.propose_ns = ns_since(t0);
    r.t.propose_ns += fb.propose_ns;
    if (rejection) {
      ++r.rejections["boundary constraint: " + *rejection];
      fb.category = "rejected";
      fb.rejection = *rejection;
    } else {
      run_case(sample, case_index, r.boundary, "boundary", c.family, fb);
    }
    finish(fb);
    ++case_index;
  }

  // Phase 2: random cases from the policy.
  const uint64_t max_attempts = options.max_attempts ? options.max_attempts : 4 * options.cases;
  while (r.random.accepted < options.cases && r.random.cases < max_attempts && !stopped()) {
    auto t0 = Clock::now();
    Sample sample = policy.propose(features, region.contract, history, case_index);
    project_onto_constraints(sample, region.contract, options.seed ^ (case_index << 1));
    ++r.attempted;
    ++r.random.cases;
    Feedback fb;
    fb.case_index = case_index;
    fb.phase = "random";
    fb.events = sample.events;
    auto rejection = ConstraintChecker::validate(sample, region.contract);
    fb.propose_ns = ns_since(t0);
    r.t.propose_ns += fb.propose_ns;
    if (rejection) {
      ++r.rejections["constraint: " + *rejection];
      fb.category = "rejected";
      fb.rejection = *rejection;
    } else {
      run_case(sample, case_index, r.random, "random", "", fb);
    }
    finish(fb);
    ++case_index;
  }
  r.novelty_buckets = buckets.size();
  if (log) {
    Json s = Json::object();
    s.set("type", "summary");
    s.set("cases", history.cases);
    s.set("accepted", history.accepted);
    s.set("valid", history.valid);
    s.set("mismatches", history.mismatches);
    s.set("novelty_buckets", static_cast<uint64_t>(buckets.size()));
    Json cats = Json::object();
    for (const auto &[k, v] : history.categories)
      cats.set(k, v);
    s.set("categories", std::move(cats));
    log << s.dump(-1) << '\n';
  }

  // Status.
  if (r.harness_errors) {
    r.test_status = TestStatus::HarnessError;
    r.reason = "harness or model error during execution (see mismatch_kinds)";
  } else if (r.mismatches) {
    r.test_status = TestStatus::LocalMismatch;
    r.witness_scope = WitnessScope::LocalOnly;
  } else if (r.step_limit_divergences) {
    r.test_status = TestStatus::Inconclusive;
    r.reason = "possible non-termination: in " + std::to_string(r.step_limit_divergences) +
               " case(s) one side exited and the other ran past the loop step limit, also at " +
               std::to_string(kStepLimitRerunScale) + "x the limit (first: " + r.step_limit_first +
               ")";
  } else if (r.valid_pairs == 0) {
    if (!unsupported_exclusions.empty()) {
      r.test_status = TestStatus::Unsupported;
      r.reason = "no case could execute: " + unsupported_exclusions.begin()->first;
    } else {
      r.test_status = TestStatus::Inconclusive;
      r.reason = r.accepted == 0 ? "no sample satisfied the constraints (zero valid cases)"
                                 : "no case exited normally on both sides";
    }
  } else if (undefined || !unsupported_exclusions.empty()) {
    r.test_status = TestStatus::Inconclusive;
    r.reason = "some cases ended in undefined behaviour or unsupported execution";
  } else {
    r.test_status = TestStatus::NoMismatchFound;
  }
  if (r.valid_pairs > 0 && r.test_status != TestStatus::HarnessError)
    r.coverage_scope = r.boundary_truncated || r.step_limit_divergences
                           ? CoverageScope::Partial
                           : CoverageScope::CompleteForDeclaredScope;
  if (r.boundary_truncated) {
    if (!r.reason.empty())
      r.reason += "; ";
    r.reason += "boundary sweep truncated at " + std::to_string(sweep.cases.size()) + " of " +
                std::to_string(sweep.generated) + " generated cases (--max-boundary-cases)";
  }
  return r;
}

ReplayOutcome replay_counterexample(const Json &bundle, std::string &error) {
  ReplayOutcome out;
  auto region = RegionPair::from_json(bundle["region"], error);
  if (!region)
    return out;
  if (compute_prepared_key(*region) != bundle["prepared_key"].as_string()) {
    error = "prepared key mismatch: the bundle's region or contract was edited";
    return out;
  }
  auto init_old = MachineState::from_json(bundle["initial_state"]["old"], error);
  auto init_new = MachineState::from_json(bundle["initial_state"]["new"], error);
  if (!init_old || !init_new)
    return out;
  const Target target = region->contract.gpu().value_or(Target{});
  auto old_runner = RegionRunner::create(region->old_side.insts, region->id + "_replay_old", error,
                                         region->contract.vgpr_count, target);
  auto new_runner = old_runner
                        ? RegionRunner::create(region->new_side.insts, region->id + "_replay_new",
                                               error, region->contract.vgpr_count, target)
                        : nullptr;
  if (!old_runner || !new_runner)
    return out;
  auto image = std::make_shared<MemoryImage>();
  image->set_small_scalars(old_runner->loops() || new_runner->loops());
  share_code_guard(*old_runner, *new_runner);
  old_runner->set_memory_image(image);
  new_runner->set_memory_image(image);
  old_runner->set_memory_dead_at_exit(region->contract.memory_dead_at_exit);
  new_runner->set_memory_dead_at_exit(region->contract.memory_dead_at_exit);
  const auto poison = parse_poison_kinds(bundle["poison_filter"].as_string()).value_or(0);
  old_runner->set_poison_filter(poison);
  new_runner->set_poison_filter(poison);
  const bool strict_nan = bundle["nan_policy"].as_string() == "strict";
  std::vector<TraceStep> trace_old, trace_new;
  old_runner->load(*init_old);
  RunOutcome oo = old_runner->run(region->old_side.insts.size(), &trace_old);
  // The recorded NaN choice of the new side (a difference shown only under it).
  if (bundle["nan_choice"].is_object())
    new_runner->set_nan_choice(static_cast<unsigned>(bundle["nan_choice"]["alternative"].as_uint()),
                               json_u64(bundle["nan_choice"]["seed"]).value_or(0));
  new_runner->load(*init_new);
  RunOutcome on = new_runner->run(region->new_side.insts.size(), &trace_new);
  // The test's re-run of a side past the loop step limit (a slower loop).
  rerun_past_step_limit(*old_runner, *new_runner, *init_old, *init_new,
                        region->old_side.insts.size(), region->new_side.insts.size(), oo, on,
                        &trace_old, &trace_new);
  Mismatch m;
  m.out_old = oo;
  m.out_new = on;
  std::string kind;
  const bool replay_fo = oo.status == RunStatus::MemoryFault;
  const bool replay_fn = on.status == RunStatus::MemoryFault;
  if (replay_fo != replay_fn && (replay_fo ? on.status : oo.status) == RunStatus::Ok) {
    kind = "exit_status_divergence";
  } else if (oo.status != RunStatus::Ok || on.status != RunStatus::Ok) {
    kind = "harness_exit";
  }
  const MachineState final_old = old_runner->capture(), final_new = new_runner->capture();
  const bool nan_old = !strict_nan && old_runner->nan_produced();
  if (kind.empty() && oo.status == RunStatus::Ok && on.status == RunStatus::Ok) {
    // The same comparison as the test: NaN policy and the case's undef mask (old re-run
    // with its undefined registers refilled, derived from the recorded seed and case).
    const UndefMask umask =
        compute_undef_mask(*region, *old_runner, *init_old, final_old, bundle["seed"].as_uint(),
                           bundle["case_index"].as_uint(), nan_old);
    kind = compare_pair(*region, *init_old, *init_new, final_old, final_new, m,
                        CompareConfig{strict_nan, &umask});
  }
  const std::string recorded_kind = bundle["kind"].as_string();
  const std::string recorded_locs = bundle["locations"].as_string();
  out.reproduced = kind == recorded_kind && format_locs(m.locs) == recorded_locs &&
                   format_vgprs(m.vlocs) == bundle["vgpr_locations"].as_string() &&
                   m.memory == bundle["memory_differs"].as_bool();
  if (kind != "harness_exit" && oo.status == RunStatus::Ok && on.status == RunStatus::Ok) {
    // Final values must match the bundle exactly, not just the kind. The initial words a run
    // read (mem_init) are an input record, not an output: the bundle's may also hold words
    // that reduction's re-runs of the same case read.
    auto fo = MachineState::from_json(bundle["final_state"]["old"], error);
    auto fn = MachineState::from_json(bundle["final_state"]["new"], error);
    auto same = [](MachineState a, MachineState b) {
      a.mem_init.clear();
      b.mem_init.clear();
      return a == b;
    };
    if (!fo || !fn || !same(final_old, *fo) || !same(final_new, *fn))
      out.reproduced = false;
  }
  out.summary = std::string(out.reproduced ? "REPRODUCED" : "NOT_REPRODUCED") +
                ": kind=" + (kind.empty() ? "none" : kind) + " locations=" + format_locs(m.locs) +
                " (recorded kind=" + recorded_kind + " locations=" + recorded_locs + ")";
  out.detail = Json::object();
  out.detail.set("reproduced", out.reproduced);
  out.detail.set("kind", kind);
  out.detail.set("locations", format_locs(m.locs));
  out.detail.set("detail", m.detail);
  out.detail.set("synthetic", region->synthetic);
  Json tr = Json::object();
  tr.set("old", trace_json(trace_old));
  tr.set("new", trace_json(trace_new));
  out.detail.set("trace", std::move(tr));
  return out;
}

} // namespace rocjitsu::isa_diff
