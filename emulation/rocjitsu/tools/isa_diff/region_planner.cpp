// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "region_planner.h"

#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <optional>
#include <set>

namespace rocjitsu::isa_diff {
namespace {

struct Span {
  size_t begin = 0, end = 0, cut = 0;
};

struct Candidate {
  Span o, n;
  std::vector<size_t> hunks;
  size_t expansions = 0;
  uint32_t before = 0, after = 0; ///< Common context added (variants).
};

/// The block a (possibly empty) side span executes in.
uint32_t span_block(const KernelModel &m, const Span &s) {
  const auto &insts = m.insts();
  if (s.begin < s.end)
    return insts[s.begin].block;
  return insts[std::min(s.cut, insts.size() - 1)].block;
}

size_t anchor_index(const Span &s) { return s.begin < s.end ? s.begin : s.cut; }

/// All instructions of [begin, end) lie in one block and only begin may be a leader.
bool single_block(const KernelModel &m, size_t begin, size_t end) {
  const auto &insts = m.insts();
  for (size_t i = begin; i < end; ++i) {
    if (insts[i].block != insts[begin].block)
      return false;
    if (i != begin && (insts[i].leader || insts[i].branch_target))
      return false;
  }
  return true;
}

std::optional<std::string> first_non_p0(const KernelModel &m, size_t begin, size_t end,
                                        const char *side) {
  for (size_t i = begin; i < end; ++i) {
    const InstRef &ref = m.insts()[i];
    if (!ref.fx.p0)
      return std::string(side) + " " + hex64(ref.pc) + " '" + ref.text +
             "': " + ref.fx.reject_reason;
  }
  return std::nullopt;
}

std::optional<std::string> pc_tainted_read(const KernelModel &m, size_t begin, size_t end,
                                           const char *side) {
  for (size_t i = begin; i < end; ++i) {
    const InstRef &ref = m.insts()[i];
    const LocSet hit = m.pc_tainted_before(i) & ref.fx.reads;
    const VSet vhit = m.pc_vtainted_before(i) & ref.fx.vreads;
    if (ref.fx.reads_pc || hit.any() || vhit.any())
      return std::string(side) + " " + hex64(ref.pc) + " '" + ref.text +
             "' reads a PC-derived value (" + format_locs(hit) +
             (vhit.any() ? (hit.any() ? "," : "") + format_vgprs(vhit) : "") + ")";
  }
  return std::nullopt;
}

/// Handoff 4: relocated bytes depend on the linker/loader; P0 does not model them.
std::optional<std::string> first_relocated(const KernelModel &m, size_t begin, size_t end,
                                           const char *side) {
  for (size_t i = begin; i < end; ++i)
    if (!m.insts()[i].relocations.empty())
      return std::string("relocated instruction (") + side + " " + hex64(m.insts()[i].pc) + " " +
             m.insts()[i].text + ": " + m.insts()[i].relocations +
             "): its executed bytes depend on relocation processing, not modelled in P0";
  return std::nullopt;
}

std::vector<RegionInst> region_insts(const KernelModel &m, size_t begin, size_t end) {
  std::vector<RegionInst> out;
  for (size_t i = begin; i < end; ++i) {
    const InstRef &ref = m.insts()[i];
    out.push_back({ref.pc, ref.words, ref.text, ref.inline_asm, ref.asm_text});
  }
  return out;
}

LocSet entry_live(const KernelModel &m, const Span &s) { return m.live_before(anchor_index(s)); }

VSet ventry_live(const KernelModel &m, const Span &s) { return m.vlive_before(anchor_index(s)); }

VSet vexit_live(const KernelModel &m, const Span &s) {
  return s.begin < s.end ? m.vlive_after(s.end - 1) : m.vlive_before(s.cut);
}

/// VGPR exit liveness without partial return slots (Artifact::return_masks).
VSet vexit_live_strict(const KernelModel &m, const Span &s) {
  return s.begin < s.end ? m.vlive_after_strict(s.end - 1) : m.vlive_before_strict(s.cut);
}

VSet union_vwrites(const KernelModel &m, const Span &s) {
  VSet w;
  for (size_t i = s.begin; i < s.end; ++i)
    w |= m.insts()[i].fx.vwrites;
  return w;
}

VSet union_vtouch(const KernelModel &m, const Span &s) {
  VSet t;
  for (size_t i = s.begin; i < s.end; ++i)
    t |= m.insts()[i].fx.vreads | m.insts()[i].fx.vwrites;
  return t;
}

VSet union_vwide_reads(const KernelModel &m, const Span &s) {
  VSet w;
  for (size_t i = s.begin; i < s.end; ++i)
    w |= m.insts()[i].fx.vwide_reads;
  return w;
}

LocSet exit_live(const KernelModel &m, const Span &s) {
  return s.begin < s.end ? m.live_after(s.end - 1) : m.live_before(s.cut);
}

LocSet union_writes(const KernelModel &m, const Span &s) {
  LocSet w;
  for (size_t i = s.begin; i < s.end; ++i)
    w |= m.insts()[i].fx.writes;
  return w;
}

LocSet union_reads(const KernelModel &m, const Span &s) {
  LocSet r;
  for (size_t i = s.begin; i < s.end; ++i)
    r |= m.insts()[i].fx.reads;
  return r;
}

LocSet union_wide_reads(const KernelModel &m, const Span &s) {
  LocSet w;
  for (size_t i = s.begin; i < s.end; ++i)
    w |= m.insts()[i].fx.wide_reads;
  return w;
}

bool in_cycle(const KernelModel &m, uint32_t block) {
  for (uint32_t s : m.blocks()[block].successors)
    if (m.reaches(s, block))
      return true;
  return !m.cfg_complete();
}

// ---------------------------------------------------------------------------
// Pending memory effects at the cutpoint (handoff 6.1).

std::string describe_pending(const char *side, const PendingState &p, const Target &target) {
  std::string s = std::string(side) + ":";
  for (unsigned c = 0; c < PendingState::kNumCounters; ++c)
    if (p.count[c])
      s += " " + pending_counter_name(c, target) + "<=" + std::to_string(p.count[c]);
  s += ", destinations {" + format_locs(p.all_dests()) + "}";
  if (p.example_pc)
    s += ", e.g. issued at " + hex64(p.example_pc);
  return s;
}

void check_pending(const KernelModel &om, const KernelModel &nm, const Candidate &c,
                   RegionContract &k) {
  const PendingState &po = om.pending_before(anchor_index(c.o));
  const PendingState &pn = nm.pending_before(anchor_index(c.n));
  const LocSet touch_old = union_reads(om, c.o) | union_writes(om, c.o);
  const LocSet touch_new = union_reads(nm, c.n) | union_writes(nm, c.n);
  (void)touch_old;
  (void)touch_new;
  // A destination still pending when a region instruction reads or writes it conflicts:
  // the isolated run sees it ready. Waits inside the region retire earlier operations,
  // and loads inside the region are pending for later region instructions until a wait.
  LocSet conflict;
  VSet vconflict;
  for (const auto *side : {&c.o, &c.n}) {
    const KernelModel &m = side == &c.o ? om : nm;
    for (size_t i = side->begin; i < side->end; ++i) {
      const PendingState &p = m.pending_before(i);
      const InstRef &ref = m.insts()[i];
      const Effects &fx = ref.fx;
      // A vector load writes its destinations after every earlier vector load (vmcnt
      // returns in issue order): overwriting a destination pending only on that in-order
      // counter is no conflict (write after write, as LLVM's waitcnt insertion assumes).
      LocSet wpend = p.all_dests();
      VSet vwpend = p.all_vdests();
      const std::string_view mn = ref.mnemonic;
      const bool vector_load = fx.mem_read && !fx.mem_write &&
                               (mn.starts_with("global_load") || mn.starts_with("buffer_load") ||
                                mn.starts_with("scratch_load"));
      if (vector_load && !p.unordered[0]) {
        wpend.reset();
        vwpend.reset();
        for (unsigned c = 0; c < PendingState::kNumCounters; ++c)
          if (c != PendingState::kLoad) {
            wpend |= p.dests[c];
            vwpend |= p.vdests[c];
          }
      }
      // A predicated destination also counts as read (inactive lanes keep it): for such a
      // load it lands in order like the write. Its source operands (the address, read at
      // issue) still conflict, also when they are among its destinations.
      LocSet own;
      VSet vown;
      if (vector_load && ref.inst) {
        own = fx.writes;
        vown = fx.vwrites;
        for (const SourceOperand &so : source_operands(*ref.inst, m.artifact().target)) {
          own &= ~so.locs;
          vown &= ~so.vlocs;
        }
      }
      conflict |= (p.all_dests() & fx.reads & ~own) | (wpend & (fx.writes | (fx.reads & own)));
      vconflict |=
          (p.all_vdests() & fx.vreads & ~vown) | (vwpend & (fx.vwrites | (fx.vreads & vown)));
    }
  }
  const std::string state = describe_pending("old", po, om.artifact().target) + "; " +
                            describe_pending("new", pn, nm.artifact().target);
  if (po.any_unknown() || pn.any_unknown()) {
    k.pending_effects = "UNKNOWN";
    k.sources.push_back({"context unresolved: a memory operation with unknown effects may be "
                         "outstanding at the region entry (" +
                             state + ")",
                         "pending_effects_analysis"});
  } else if (conflict.any() || vconflict.any()) {
    k.pending_effects = "CONFLICT";
    k.sources.push_back(
        {"context unresolved: outstanding memory writes to {" + format_locs(conflict) +
             (vconflict.any() ? (conflict.any() ? "," : "") + format_vgprs(vconflict) : "") +
             "} may land during the region, which reads or writes them; the "
             "isolated run starts with every input ready (" +
             state + ")",
         "pending_effects_analysis"});
  } else if (po.any() || pn.any()) {
    k.pending_effects = "DISJOINT";
    k.sources.push_back({"memory operations are outstanding at the region entry, but none "
                         "writes a location the region reads or writes, so starting the "
                         "isolated run with nothing pending matches the region's view (" +
                             state + ")",
                         "pending_effects_analysis"});
  } else {
    k.pending_effects = "NONE";
    k.sources.push_back({"no memory operation outstanding at the region entry (s_waitcnt "
                         "dataflow over the CFG)",
                         "pending_effects_analysis"});
  }
}

// ---------------------------------------------------------------------------
// Region construction (shared by base regions and expansion variants).

/// A kernel: an artifact with a descriptor, or a "callable" function that ends with
/// s_endpgm on both sides. A function of the C or gfx convention never does; one that does
/// is a kernel listed without its descriptor (as rj_isa_diff_pr writes them). A shader
/// part's s_endpgm returns to its epilog.
bool is_kernel(const KernelModel &om, const KernelModel &nm) {
  const Artifact &a = om.artifact();
  if (!a.callable)
    return true;
  return a.calling_convention != "shader" && !om.insts().empty() && !nm.insts().empty() &&
         om.insts().back().mnemonic == "s_endpgm" && nm.insts().back().mnemonic == "s_endpgm";
}

/// The control flow of instructions [b, e) on one side: one entry (b), direct branches
/// only (and an early s_endpgm when the region reaches the final one). Sets @p exits to the
/// instruction indices control can leave to (e: falling off the end) and @p branches when
/// there is a branch. Returns why it is not such a region.
std::optional<std::string> region_flow(const KernelModel &m, size_t b, size_t e, const char *side,
                                       std::vector<size_t> &exits, bool &branches) {
  const auto &in = m.insts();
  const auto &blocks = m.blocks();
  exits.clear();
  branches = false;
  auto add = [&](size_t t) {
    if (std::ranges::find(exits, t) == exits.end())
      exits.push_back(t);
  };
  if (b >= e) {
    add(e);
    return std::nullopt;
  }
  bool falls = true;
  for (size_t i = b; i < e; ++i) {
    const InstRef &r = in[i];
    if (!r.fx.control_flow)
      continue;
    // An s_endpgm before the last instruction ends the wave as the final s_endpgm does (a
    // kernel's early exit): when the region reaches the final one, it is an exit there.
    if (r.mnemonic == "s_endpgm" && e + 1 == in.size() && in.back().mnemonic == "s_endpgm") {
      branches = true;
      add(e);
      continue;
    }
    if (!r.fx.branch || r.branch_target_index < 0)
      return std::string(side) + " " + hex64(r.pc) + " '" + r.text +
             "': only direct branches can be inside a region";
    branches = true;
    const auto t = static_cast<size_t>(r.branch_target_index);
    if (t < b || t >= e)
      add(t);
    if (i + 1 == e)
      falls = r.mnemonic != "s_branch";
  }
  if (falls) {
    if (e >= in.size())
      return std::string(side) + " region falls off the end of the kernel";
    add(e);
  }
  // Single entry: a block starting inside the region is reached from inside only.
  for (size_t i = b + 1; i < e; ++i) {
    const BlockInfo &blk = blocks[in[i].block];
    if (blk.first != i)
      continue;
    for (uint32_t pred : blk.predecessors) {
      const size_t last = blocks[pred].end - 1;
      if (last < b || last >= e)
        return std::string(side) + " control enters the region at " + hex64(in[i].pc) +
               " from outside it";
    }
  }
  return std::nullopt;
}

RegionPair build_region(const KernelModel &om, const KernelModel &nm, const Candidate &c,
                        const PlanOptions &options, const std::string &id,
                        const std::vector<std::pair<unsigned, unsigned>> &entry_map = {},
                        const std::vector<std::pair<unsigned, unsigned>> &exit_map = {},
                        const DiffResult *diff = nullptr) {
  RegionPair region;
  region.id = id;
  region.hunks = c.hunks;
  region.old_side = {c.o.begin, c.o.end, c.o.cut, region_insts(om, c.o.begin, c.o.end)};
  region.new_side = {c.n.begin, c.n.end, c.n.cut, region_insts(nm, c.n.begin, c.n.end)};
  region.status = RegionStatus::Ready;
  region.context_before = c.before;
  region.context_after = c.after;

  std::optional<std::string> reject;
  // Control flow: each side's exits, paired through the instruction matching (falling off
  // the end of both regions is the cut point, which is matched by construction).
  std::vector<size_t> oexits, nexits;
  bool obranch = false, nbranch = false;
  std::vector<std::pair<size_t, size_t>> exit_pairs;
  auto flow_reject = [&]() -> std::optional<std::string> {
    if (auto bad = region_flow(om, c.o.begin, c.o.end, "old", oexits, obranch))
      return bad;
    if (auto bad = region_flow(nm, c.n.begin, c.n.end, "new", nexits, nbranch))
      return bad;
    for (size_t to : oexits) {
      int64_t tn = -1;
      if (to == c.o.end)
        tn = static_cast<int64_t>(c.n.end);
      else if (diff && to < diff->old_to_new.size())
        tn = diff->old_to_new[to];
      if (tn < 0 || std::ranges::find(nexits, static_cast<size_t>(tn)) == nexits.end())
        return "INCONCLUSIVE_MAPPING: the old exit to " +
               hex64(to < om.insts().size() ? om.insts()[to].pc : 0) +
               " has no corresponding new exit";
      exit_pairs.emplace_back(to, static_cast<size_t>(tn));
    }
    if (exit_pairs.size() != nexits.size())
      return std::string("INCONCLUSIVE_MAPPING: a new exit has no corresponding old exit");
    return std::nullopt;
  };
  const bool multi_block =
      !single_block(om, c.o.begin, c.o.end) || !single_block(nm, c.n.begin, c.n.end);
  if (auto bad = flow_reject()) {
    region.status = RegionStatus::Inconclusive;
    reject = bad->rfind("INCONCLUSIVE_MAPPING", 0) == 0 ? *bad : "INCONCLUSIVE_MAPPING: " + *bad;
  } else if (multi_block && !obranch && !nbranch && exit_pairs.size() != 1) {
    region.status = RegionStatus::Inconclusive;
    reject = "INCONCLUSIVE_MAPPING: region spans basic blocks without a common exit";
  } else if (c.o.end - c.o.begin > options.max_region_insts ||
             c.n.end - c.n.begin > options.max_region_insts) {
    reject = "region exceeds max_region_insts";
  } else if (auto bad = first_non_p0(om, c.o.begin, c.o.end, "old")) {
    reject = *bad;
  } else if (auto bad2 = first_non_p0(nm, c.n.begin, c.n.end, "new")) {
    reject = *bad2;
  } else if (auto rel = first_relocated(om, c.o.begin, c.o.end, "old")) {
    reject = *rel;
  } else if (auto rel2 = first_relocated(nm, c.n.begin, c.n.end, "new")) {
    reject = *rel2;
  } else if (auto pc = pc_tainted_read(om, c.o.begin, c.o.end, "old")) {
    reject = *pc;
  } else if (auto pc2 = pc_tainted_read(nm, c.n.begin, c.n.end, "new")) {
    reject = *pc2;
  }
  if (reject) {
    if (region.status == RegionStatus::Ready)
      region.status = RegionStatus::Unsupported;
    region.reason = *reject;
  }

  RegionContract &k = region.contract;
  k.set_gpu(om.artifact().target);
  // Contract sets use old naming; with register maps the new side's sets are translated,
  // live-ins through the entry map and live-outs through the exit map.
  k.entry_map = entry_map;
  k.exit_map = exit_map;
  const bool mapped = k.mapped();
  const auto inv_in = k.entry_inverse();
  const auto inv_out = k.exit_inverse();
  auto to_old = [](const LocSet &s, const std::array<unsigned, kNumLocs> &inv) {
    LocSet o;
    for (unsigned l = 0; l < kNumLocs; ++l)
      if (s.test(l))
        o.set(inv[l]);
    return o;
  };
  k.entry_equal = entry_live(om, c.o) | to_old(entry_live(nm, c.n), inv_in);
  // Undefined scalar inputs (LLVM undef): live-ins written on no path from the entry and
  // not defined there. Not with a register map (the sides name different locations).
  if (!mapped) {
    k.undef_old = entry_live(om, c.o) & om.undef_before(anchor_index(c.o));
    k.undef_new = entry_live(nm, c.n) & nm.undef_before(anchor_index(c.n));
  }
  k.canary = ~k.entry_equal;
  // EXEC is still the dispatch (or call) EXEC on every path here, on both sides: a wave
  // runs with at least one lane, so EXEC 0 is not an input.
  if (k.entry_equal.test(kLocExecLo) && om.exec_is_entry_before(anchor_index(c.o)) &&
      nm.exec_is_entry_before(anchor_index(c.n))) {
    k.exec_nonzero = true;
    // A matrix instruction reads every lane's operands: run with inactive lanes, its result
    // depends on whatever those lanes hold. LLVM issues MFMA with the whole wave active, so
    // an entry EXEC that reaches one is all ones.
    bool mfma = false;
    for (const auto *side : {&c.o, &c.n}) {
      const KernelModel &m = side == &c.o ? om : nm;
      for (size_t i = side->begin; i < side->end; ++i)
        mfma = mfma || m.insts()[i].mnemonic.rfind("v_mfma", 0) == 0 ||
               m.insts()[i].mnemonic.rfind("v_smfmac", 0) == 0;
    }
    if (mfma) {
      const unsigned lanes = om.artifact().target.wave_size;
      for (unsigned l : {kLocExecLo, kLocExecHi}) {
        const uint32_t ones = l == kLocExecLo || lanes == 64 ? 0xffffffffu : 0u;
        Constraint full;
        full.loc = l;
        full.known_one = ones;
        full.known_zero = ~ones;
        full.source = "mfma_full_exec";
        k.constraints.push_back(full);
      }
      k.sources.push_back({"mfma_full_exec: the region issues a matrix instruction under the "
                           "entry EXEC, which LLVM keeps whole-wave: EXEC is all ones (inactive "
                           "lanes would feed the matrix product)",
                           "mfma_full_exec"});
    }
    k.sources.push_back({"exec_nonzero: EXEC is the entry EXEC on every path to the region "
                         "(a wave runs with at least one active lane)",
                         "KernelModel forward may-written LocSet analysis"});
  }
  const LocSet out_old = exit_live(om, c.o);
  const LocSet out_new = to_old(exit_live(nm, c.n), inv_out);
  k.observable = out_old | out_new;
  // SGPRs observable only through a return value narrower than the register.
  {
    const LocSet full =
        (c.o.begin < c.o.end ? om.live_after_strict(c.o.end - 1) : om.live_before_strict(c.o.cut)) |
        (c.n.begin < c.n.end ? nm.live_after_strict(c.n.end - 1) : nm.live_before_strict(c.n.cut));
    const auto &masks = om.artifact().return_sgpr_masks;
    for (unsigned s = 0; s < masks.size() && s < om.artifact().return_sgprs; ++s)
      if (masks[s] != 0xffffffffu && k.observable.test(s) && !full.test(s))
        k.exit_masks.emplace_back(s, masks[s]);
  }
  // The two continuations read different locations: they differ downstream (typically
  // a register renaming that crosses blocks), so equality on the same physical
  // locations at this exit is not a justified relation. Do not guess a mapping.
  const bool flow = obranch || nbranch || multi_block;
  if (region.status == RegionStatus::Ready && out_old != out_new && !flow) {
    region.status = RegionStatus::Inconclusive;
    region.reason = "INCONCLUSIVE_MAPPING: continuation liveness differs at the region exit "
                    "(old-only: " +
                    format_locs(out_old & ~out_new) +
                    "; new-only: " + format_locs(out_new & ~out_old) + "); " +
                    (mapped ? std::string("the user register map does not reconcile them")
                            : std::string("identity R_out is not justified and no checked value "
                                          "mapping exists"));
  }
  // VGPRs (VALU support): the tracked range covers every VGPR either side names, rounded
  // up to the allocation granule; live-ins and live-outs are restricted to it (a VGPR
  // beyond it is not allocated, so the region cannot touch it).
  const unsigned vextent = vgpr_extent(union_vtouch(om, c.o) | union_vtouch(nm, c.n));
  k.vgpr_count = vextent == 0 ? 0u : std::min(kNumVgprs, (vextent + 7u) & ~7u);
  if (k.vgpr_count) {
    VSet range;
    for (unsigned r = 0; r < k.vgpr_count; ++r)
      range.set(r);
    k.ventry_equal = (ventry_live(om, c.o) | ventry_live(nm, c.n)) & range;
    const VSet vout_old = vexit_live(om, c.o) & range;
    const VSet vout_new = vexit_live(nm, c.n) & range;
    k.vobservable = vout_old | vout_new;
    // Undefined inputs (LLVM undef): live-in VGPRs written on no path from the entry and
    // not defined at it.
    k.vundef_old = ventry_live(om, c.o) & om.vundef_before(anchor_index(c.o)) & range;
    k.vundef_new = ventry_live(nm, c.n) & nm.vundef_before(anchor_index(c.n)) & range;
    // VGPRs observable only through a return value narrower than the register.
    const VSet full = (vexit_live_strict(om, c.o) | vexit_live_strict(nm, c.n)) & range;
    const auto &masks = om.artifact().return_masks;
    for (unsigned r = 0; r < masks.size() && r < om.artifact().return_vgprs; ++r)
      if (masks[r] != 0xffffffffu && k.vobservable.test(r) && !full.test(r))
        k.vexit_masks.emplace_back(r, masks[r]);
    // Inactive lanes at an exit after which EXEC never changes are never read again; a
    // callable function's callee-saved VGPRs must still keep every lane.
    auto stable = [](const KernelModel &m, const Span &s) {
      return s.begin < s.end ? m.exec_stable_after(s.end - 1) : m.exec_stable_before(s.cut);
    };
    if (stable(om, c.o) && stable(nm, c.n)) {
      const VSet csr = om.callee_saved_vgprs() | nm.callee_saved_vgprs();
      k.vactive_only = k.vobservable & ~csr;
    }
    k.vwrites_old = union_vwrites(om, c.o);
    k.vwrites_new = union_vwrites(nm, c.n);
    k.vrange = range;
    k.vwide_reads = (union_vwide_reads(om, c.o) | union_vwide_reads(nm, c.n)) & range;
    if (region.status == RegionStatus::Ready && vout_old != vout_new && !flow) {
      region.status = RegionStatus::Inconclusive;
      region.reason = "INCONCLUSIVE_MAPPING: VGPR continuation liveness differs at the region "
                      "exit (old-only: " +
                      format_vgprs(vout_old & ~vout_new) +
                      "; new-only: " + format_vgprs(vout_new & ~vout_old) +
                      "); identity R_out is not justified";
    }
    k.sources.push_back({"vector registers v0.." + vgpr_name(k.vgpr_count - 1) +
                             " are tracked on all 64 lanes; vgpr_entry_equal/vgpr_observable "
                             "come from VGPR liveness, in which a write never kills (lanes "
                             "outside EXEC keep their value)",
                         "KernelModel backward VSet liveness"});
    if (k.vundef_old.any() || k.vundef_new.any())
      k.sources.push_back({"undefined inputs (old " + format_vgprs(k.vundef_old) + "; new " +
                               format_vgprs(k.vundef_new) +
                               "): never written since the entry and not defined there; old's "
                               "outputs that depend on them are don't care",
                           "KernelModel forward may-written VSet analysis"});
  }
  std::vector<int> asm_blocks[2];
  for (const auto *side : {&region.old_side, &region.new_side}) {
    const KernelModel &m = side == &region.old_side ? om : nm;
    for (size_t i = side->begin; i < side->end; ++i) {
      k.memory_reads = k.memory_reads || m.insts()[i].fx.mem_read;
      k.memory_writes = k.memory_writes || m.insts()[i].fx.mem_write;
      if (m.insts()[i].inline_asm >= 0)
        asm_blocks[side == &region.old_side ? 0 : 1].push_back(m.insts()[i].inline_asm);
    }
  }
  if (!asm_blocks[0].empty() || !asm_blocks[1].empty()) {
    k.inline_asm = true;
    k.sources.push_back({"inline asm: blocks are paired by their index in each function's "
                         "sequence; a def block writes per-case values, equal for both "
                         "blocks of a pair; what use blocks observe (lanes in EXEC) is compared",
                         "inline-asm markers from the .s source"});
    if (asm_blocks[0] != asm_blocks[1] && region.status == RegionStatus::Ready) {
      region.status = RegionStatus::Inconclusive;
      region.reason = "INCONCLUSIVE_MAPPING: the region's sides contain different inline asm "
                      "blocks, so their values cannot be paired";
    }
  }
  bool barrier = false;
  for (const auto *side : {&region.old_side, &region.new_side}) {
    const KernelModel &m = side == &region.old_side ? om : nm;
    for (size_t i = side->begin; i < side->end; ++i)
      barrier = barrier || m.insts()[i].fx.barrier;
  }
  if (barrier)
    k.sources.push_back({"s_barrier: one wave of the workgroup runs, so a barrier waits for no "
                         "other wave; it divides the run into epochs, and the sides must "
                         "execute as many barriers, store to the same shared dwords (LDS, "
                         "global memory outside the private segment) in each epoch, and load an "
                         "LDS dword both load in the same epochs (global loads are not "
                         "tracked). What other waves store is not modelled: their part of "
                         "memory keeps its initial content",
                         "barrier epochs"});
  // Stack-frame bytes the region stores are dead at its exit when nothing outside it can
  // read private memory (spill slots reloaded inside the region, or not at all).
  if (k.memory_writes) {
    bool private_reads = false;
    for (const auto *side : {&region.old_side, &region.new_side}) {
      const KernelModel &m = side == &region.old_side ? om : nm;
      for (size_t i = 0; i < m.insts().size() && !private_reads; ++i)
        if (i < side->begin || i >= side->end)
          private_reads = may_read_private(m.insts()[i].mnemonic) || m.insts()[i].fx.unknown_effect;
    }
    if (!private_reads) {
      // A callable function's VGPR-addressed private stores may reach its caller's frame;
      // a kernel has no caller, so all its private memory is its own.
      const bool kernel = is_kernel(om, nm);
      k.memory_dead_at_exit = kernel ? "private_memory" : "frame";
      k.sources.push_back({kernel ? "memory: no instruction outside the region reads private "
                                    "memory, so the private bytes this kernel stores are dead "
                                    "at its exit and not compared"
                                  : "memory: no instruction outside the region reads private "
                                    "memory, so the stack-frame bytes it stores are dead at its "
                                    "exit and not compared",
                           "memory_model"});
    }
  }
  if (k.memory_reads || k.memory_writes) {
    k.sources.push_back({"memory: loads read one initial memory image shared by both sides "
                         "(an input); the bytes either side stores are compared at the exit",
                         "memory model"});
    // An SGPR pair the region uses as a memory base before writing it is a pointer input:
    // canonical, 256-byte aligned, far above the code (high half in [0x10, 0x7f]).
    LocSet bases;
    for (const auto *side : {&region.old_side, &region.new_side}) {
      const KernelModel &m = side == &region.old_side ? om : nm;
      LocSet written;
      for (size_t i = side->begin; i < side->end; ++i) {
        const InstRef &ref = m.insts()[i];
        if (ref.fx.mem_read || ref.fx.mem_write)
          for (const SourceOperand &so : source_operands(*ref.inst, m.artifact().target))
            if (so.locs.count() == 2) {
              if ((so.locs & written).none())
                bases |= so.locs;
              break;
            }
        written |= ref.fx.writes;
      }
    }
    // Buffer resources (V#) and scratch offsets that are region inputs: the calling
    // convention's private segment buffer (s[0:3] in a function), a base far above the
    // code with unlimited records and ADD_TID with a dword swizzle, so lanes interleave by
    // dword as the hardware's scratch does; stack-pointer offsets are 16-byte aligned.
    {
      const Target &t = om.artifact().target;
      const uint32_t swizzle = t.major >= 11 ? 0x40000000u : 0x80000000u;
      const uint32_t word3 = t.major <= 9 ? 0x00e00000u : t.wave32() ? 0x31c16000u : 0x31e16000u;
      auto pin = [&](unsigned l, uint32_t one, uint32_t zero) {
        if (!k.entry_equal.test(l))
          return;
        for (const Constraint &c : k.constraints)
          if (c.loc == l && c.source == "memory_model")
            return;
        Constraint c;
        c.loc = l;
        c.known_one = one;
        c.known_zero = zero;
        c.source = "memory_model";
        k.constraints.push_back(c);
      };
      bool resource = false;
      for (const auto *side : {&region.old_side, &region.new_side}) {
        const KernelModel &m = side == &region.old_side ? om : nm;
        LocSet written;
        for (size_t i = side->begin; i < side->end; ++i) {
          const InstRef &ref = m.insts()[i];
          const bool buffer = ref.mnemonic.rfind("buffer_", 0) == 0;
          const bool scratch = ref.mnemonic.rfind("scratch_", 0) == 0;
          if ((buffer || scratch) && (ref.fx.mem_read || ref.fx.mem_write))
            for (const SourceOperand &so : source_operands(*ref.inst, m.artifact().target)) {
              const auto n = so.locs.count();
              unsigned first = 0;
              while (first < kNumLocs && !so.locs.test(first))
                ++first;
              if (n == 4 && buffer && first + 3 < kNumSgprs) {
                const unsigned words[4] = {first, first + 1, first + 2, first + 3};
                if (!written.test(words[0]))
                  pin(words[0], 0, 0xffu);
                if (!written.test(words[1]))
                  pin(words[1], swizzle | 0x10u, ~(swizzle | 0x7fu));
                if (!written.test(words[2]))
                  pin(words[2], 0xffffffffu, 0);
                if (!written.test(words[3]))
                  pin(words[3], word3, ~word3);
                resource = true;
              } else if (n == 1 && first < kNumSgprs && !written.test(first)) {
                pin(first, 0, 0xfu); // soffset / saddr: an aligned stack offset
              }
            }
          written |= ref.fx.writes;
        }
      }
      // LDS addresses that are region inputs: inside the window, dword aligned, with room
      // for the instruction offsets.
      VSet lds_addr;
      for (const auto *side : {&region.old_side, &region.new_side}) {
        const KernelModel &m = side == &region.old_side ? om : nm;
        VSet vwritten;
        for (size_t i = side->begin; i < side->end; ++i) {
          const InstRef &ref = m.insts()[i];
          if (ref.mnemonic.rfind("ds_", 0) == 0 && (ref.fx.mem_read || ref.fx.mem_write))
            for (const SourceOperand &so : source_operands(*ref.inst, m.artifact().target))
              if (so.vlocs.any()) {
                lds_addr |= so.vlocs & ~vwritten & k.ventry_equal;
                break; // the address is the first VGPR source
              }
          vwritten |= ref.fx.vwrites;
        }
      }
      for (unsigned r = 0; r < kNumVgprs; ++r)
        if (lds_addr.test(r))
          k.vinput_masks.emplace_back(r, 0x7ffcu);
      if (lds_addr.any())
        k.sources.push_back({"LDS addresses that are region inputs (" + format_vgprs(lds_addr) +
                                 ") lie inside the LDS window: every lane is masked with "
                                 "0x7ffc (outside the allocation LLVM is undefined)",
                             "memory_model"});
      if (resource)
        k.sources.push_back({"buffer resources that are region inputs are the private segment "
                             "buffer of the calling convention: a 256-byte aligned base far "
                             "above the code, unlimited records, ADD_TID with a dword swizzle "
                             "(word 3 " +
                                 hex32(word3) + ")",
                             "memory_model"});
    }
    for (unsigned l = 0; l + 1 < kNumSgprs; ++l) {
      if (!bases.test(l) || !bases.test(l + 1) || (l > 0 && bases.test(l - 1) && l % 2 == 1))
        continue;
      if (!k.entry_equal.test(l) || !k.entry_equal.test(l + 1))
        continue;
      Constraint lo, hi;
      lo.loc = l;
      lo.known_zero = 0xffu;
      lo.source = "memory_model";
      hi.loc = l + 1;
      hi.known_one = 0x10u;
      hi.known_zero = 0xffffff80u;
      hi.source = "memory_model";
      k.constraints.push_back(lo);
      k.constraints.push_back(hi);
      ++l;
    }
  }
  // Regions with control flow: R_out per exit pair, from the liveness at each exit's
  // target; observable/vobservable become their union.
  if (!reject && (obranch || nbranch || multi_block)) {
    const VSet csr = om.callee_saved_vgprs() | nm.callee_saved_vgprs();
    const VSet vrange = k.vrange;
    LocSet obs_union;
    VSet vobs_union;
    for (const auto &[to, tn] : exit_pairs) {
      RegionExit ex;
      ex.old_pc = to == c.o.end ? 0 : om.insts()[to].pc;
      ex.new_pc = tn == c.n.end ? 0 : nm.insts()[tn].pc;
      const LocSet lo = om.live_before(to);
      const LocSet ln = to_old(nm.live_before(tn), inv_out);
      ex.observable = lo | ln;
      const VSet vo = om.vlive_before(to) & vrange, vn = nm.vlive_before(tn) & vrange;
      ex.vobservable = vo | vn;
      if (om.exec_stable_before(to) && nm.exec_stable_before(tn))
        ex.vactive_only = ex.vobservable & ~csr;
      if (region.status == RegionStatus::Ready && (lo != ln || vo != vn)) {
        region.status = RegionStatus::Inconclusive;
        region.reason = "INCONCLUSIVE_MAPPING: continuation liveness differs at the exit to " +
                        hex64(om.insts()[to].pc) + " (old-only: " + format_locs(lo & ~ln) +
                        format_vgprs(vo & ~vn) + "; new-only: " + format_locs(ln & ~lo) +
                        format_vgprs(vn & ~vo) + ")";
      }
      obs_union |= ex.observable;
      vobs_union |= ex.vobservable;
      k.exits.push_back(ex);
    }
    k.observable = obs_union;
    k.vobservable = vobs_union;
    k.vactive_only = VSet{};
    k.sources.push_back({std::to_string(k.exits.size()) +
                             " exit pair(s): the runner follows the region's branches; each exit "
                             "is compared on the liveness at its target, and the two sides "
                             "must leave through corresponding exits",
                         "region control flow"});
  }
  k.mode_old = initial_mode(om.artifact());
  k.mode_new = initial_mode(nm.artifact());
  k.sources.push_back({"mode_old/mode_new = MODE at dispatch from each kernel descriptor's "
                       "compute_pgm_rsrc1 (assembler defaults without a descriptor)",
                       "kernel descriptor"});
  k.declared_writes_old = union_writes(om, c.o);
  k.declared_writes_new = union_writes(nm, c.n); // physical new naming (effect model)
  k.wide_reads = union_wide_reads(om, c.o) | to_old(union_wide_reads(nm, c.n), inv_in);
  k.sources.push_back({"entry_equal = live-in(old) | live-in(new) at the region entry",
                       "KernelModel backward LocSet liveness over BasicBlock::build_reachable "
                       "CFG"});
  k.sources.push_back({"observable = live-out(old) | live-out(new) at the region exit",
                       "KernelModel backward LocSet liveness"});
  if (mapped) {
    k.mapping = "user_register_map";
    auto show = [](const std::vector<std::pair<unsigned, unsigned>> &m) {
      return m.empty() ? std::string("identity") : format_register_map(m);
    };
    k.sources.push_back({"mapping = entry map " + show(entry_map) + ", exit map " + show(exit_map) +
                             " (old=new; other locations identity), asserted by hand and not "
                             "checked",
                         "user_assumption"});
  } else {
    k.sources.push_back({"mapping = same physical location on both sides",
                         "identity at a common cutpoint; matched anchors from LCS alignment"});
  }
  k.sources.push_back({"canary = locations dead at entry on both sides",
                       "complement of entry_equal; independent per-side values"});
  if (c.before)
    k.sources.push_back({"region includes " + std::to_string(c.before) +
                             " common instruction(s) before the change as context",
                         "common_prefix_analysis"});
  if (c.after)
    k.sources.push_back({"region includes " + std::to_string(c.after) +
                             " common instruction(s) after the change as context",
                         "common_suffix_context"});
  if (!om.cfg_complete() || !nm.cfg_complete())
    k.sources.push_back(
        {"some CFG successors are unknown; liveness made them all-live", "conservative widening"});
  check_pending(om, nm, c, k);
  return region;
}

// ---------------------------------------------------------------------------
// Context expansion steps over matched common P0 instructions in the same block.

bool step_back(const KernelModel &om, const KernelModel &nm, const DiffResult &diff, Candidate &c) {
  const auto &oi = om.insts();
  const auto &ni = nm.insts();
  const size_t ob = anchor_index(c.o), nb = anchor_index(c.n);
  if (ob == 0 || nb == 0 || ob >= oi.size() || nb >= ni.size())
    return false;
  const size_t po = ob - 1, pn = nb - 1;
  if (diff.old_to_new[po] != static_cast<int64_t>(pn))
    return false;
  if (oi[po].block != oi[ob].block || ni[pn].block != ni[nb].block)
    return false;
  if (oi[ob].leader || ni[nb].leader || oi[ob].branch_target || ni[nb].branch_target)
    return false;
  if (!oi[po].fx.p0 || !ni[pn].fx.p0)
    return false;
  if (c.o.begin == c.o.end)
    c.o.end = ob;
  if (c.n.begin == c.n.end)
    c.n.end = nb;
  c.o.begin = po;
  c.n.begin = pn;
  ++c.before;
  return true;
}

bool step_forward(const KernelModel &om, const KernelModel &nm, const DiffResult &diff,
                  Candidate &c) {
  const auto &oi = om.insts();
  const auto &ni = nm.insts();
  const size_t oe = c.o.end, ne = c.n.end;
  if (oe >= oi.size() || ne >= ni.size())
    return false;
  if (diff.old_to_new[oe] != static_cast<int64_t>(ne))
    return false;
  if (oi[oe].block != span_block(om, c.o) || ni[ne].block != span_block(nm, c.n))
    return false;
  if (oi[oe].leader || ni[ne].leader || oi[oe].branch_target || ni[ne].branch_target)
    return false;
  if (!oi[oe].fx.p0 || !ni[ne].fx.p0)
    return false;
  if (c.o.begin == c.o.end)
    c.o.begin = oe;
  if (c.n.begin == c.n.end)
    c.n.begin = ne;
  c.o.end = c.o.cut = oe + 1;
  c.n.end = c.n.cut = ne + 1;
  ++c.after;
  return true;
}

// ---------------------------------------------------------------------------
// Composition check (handoff 5.2): forward must-equality over the old CFG.
//
// State: locations known equal on both sides, plus whether memory and VGPR state are
// known equal. Kernel entry: everything equal, unless kernel metadata differs.
// Common instructions are identical on both sides: their writes are equal when all
// their inputs are. A READY region contributes its R_out (equality on `observable`)
// and preserves untouched locations. An unsupported region removes what it writes.
// A control-flow change or divergent branch makes nothing known.

struct EqState {
  LocSet eq;
  bool mem = true;
  VSet veq;        ///< VGPRs equal on every lane.
  bool top = true; ///< Not reached yet (identity of the meet).
  bool operator==(const EqState &) const = default;
};

EqState eq_bottom() { return {LocSet{}, false, VSet{}, false}; }

EqState eq_all() {
  EqState s{all_locs(), true, VSet{}, false};
  s.veq.set();
  return s;
}

EqState eq_meet(const EqState &a, const EqState &b) {
  if (a.top)
    return b;
  if (b.top)
    return a;
  return {a.eq & b.eq, a.mem && b.mem, a.veq & b.veq, false};
}

struct Atom {
  enum Kind { ReadyRegion, OpaqueRegion, Diverge } kind = Diverge;
  size_t region = SIZE_MAX; ///< Index into regions for region atoms.
  size_t anchor = 0;
  size_t begin = 0, end = 0; ///< Consumed old instructions.
  bool boundary = false;     ///< Empty old side at a block boundary: no R_out claim.
  LocSet writes;
  LocSet observable;
  VSet vwrites;     ///< VGPRs the changed code may write.
  VSet vobservable; ///< VGPRs a READY region establishes equal (its R_out).
  bool unknown = false, mem = false;
};

struct AtomMap {
  std::vector<Atom> atoms;
  std::vector<std::vector<size_t>> atoms_at; ///< Atoms applied before each old instruction.
  std::vector<int64_t> owner;                ///< Atom consuming each old instruction, or -1.
};

/// Changed code on the old instruction stream: READY regions, untested regions, and
/// changes outside any region, each applied atomically at its anchor.
AtomMap build_atoms(const KernelModel &om, const KernelModel &nm, const DiffResult &diff,
                    const PlanResult &plan) {
  const auto &oi = om.insts();
  AtomMap map;
  auto &atoms = map.atoms;
  auto &atoms_at = map.atoms_at;
  auto &owner = map.owner;
  atoms_at.assign(oi.size() + 1, {});
  owner.assign(oi.size(), -1);
  auto add_atom = [&](Atom a) {
    const size_t id = atoms.size();
    for (size_t i = a.begin; i < a.end && i < oi.size(); ++i)
      owner[i] = static_cast<int64_t>(id);
    atoms_at[std::min(a.anchor, oi.size())].push_back(id);
    atoms.push_back(std::move(a));
  };
  std::vector<bool> hunk_in_region(diff.hunks.size(), false);
  for (size_t r = 0; r < plan.regions.size(); ++r) {
    const RegionPair &reg = plan.regions[r];
    for (size_t h : reg.hunks)
      hunk_in_region[h] = true;
    Atom a;
    a.region = r;
    a.begin = reg.old_side.begin;
    a.end = reg.old_side.end;
    a.anchor = reg.old_side.begin < reg.old_side.end ? reg.old_side.begin : reg.old_side.cut;
    a.boundary = reg.old_side.begin == reg.old_side.end && a.anchor > 0 && a.anchor < oi.size() &&
                 oi[a.anchor].leader && oi[a.anchor - 1].block != oi[a.anchor].block;
    const bool one_block = single_block(om, reg.old_side.begin, reg.old_side.end);
    if (!one_block || !reg.contract.exits.empty()) {
      a.kind = Atom::Diverge;
    } else if (reg.status == RegionStatus::Ready) {
      a.kind = Atom::ReadyRegion;
      a.writes = reg.contract.declared_writes_old | reg.contract.declared_writes_new;
      a.observable = reg.contract.observable;
      a.vwrites = reg.contract.vwrites_old | reg.contract.vwrites_new;
      a.vobservable = reg.contract.vobservable;
    } else {
      a.kind = Atom::OpaqueRegion;
      auto absorb = [&](const KernelModel &m, size_t b, size_t e) {
        for (size_t i = b; i < e; ++i) {
          const InstRef &ref = m.insts()[i];
          a.writes |= ref.fx.writes;
          a.unknown = a.unknown || ref.fx.unknown_effect;
          a.mem = a.mem || (ref.inst && (ref.inst->flags() & MEMORY_OP));
          a.vwrites |= ref.fx.vwrites;
        }
      };
      absorb(om, reg.old_side.begin, reg.old_side.end);
      absorb(nm, reg.new_side.begin, reg.new_side.end);
    }
    add_atom(std::move(a));
  }
  for (size_t h = 0; h < diff.hunks.size(); ++h) {
    if (hunk_in_region[h] || diff.hunks[h].kind == ChangeKind::Hint)
      continue; // a hint hunk has no effect: its old instructions are common code
    const Hunk &hunk = diff.hunks[h];
    Atom a;
    a.kind = Atom::Diverge;
    a.begin = hunk.old_begin;
    a.end = hunk.old_end;
    a.anchor = hunk.old_begin < hunk.old_end ? hunk.old_begin : hunk.old_end;
    add_atom(std::move(a));
  }
  return map;
}

/// The edge from old instruction @p from to @p to lies inside one READY region with control
/// flow: its runner executes it (a loop back to the region's entry).
bool region_runs_edge(const PlanResult &plan, const AtomMap &map, size_t from, size_t to) {
  const int64_t a = map.owner[from];
  if (a < 0 || map.owner[to] != a)
    return false;
  const size_t r = map.atoms[static_cast<size_t>(a)].region;
  return r < plan.regions.size() && plan.regions[r].status == RegionStatus::Ready &&
         !plan.regions[r].contract.exits.empty();
}

struct CompositionResult {
  std::vector<EqState> anchor_state; ///< Per region index: state at its entry.
  std::vector<EqState> before;       ///< Per old instruction: state before it.
};

CompositionResult analyse_composition(const KernelModel &om, const KernelModel &nm,
                                      const DiffResult &diff, const PlanResult &plan,
                                      bool metadata_differs) {
  const auto &oi = om.insts();
  const auto &ni = nm.insts();
  const auto &blocks = om.blocks();
  const AtomMap map = build_atoms(om, nm, diff, plan);
  const auto &atoms = map.atoms;
  const auto &atoms_at = map.atoms_at;
  const auto &owner = map.owner;
  (void)ni;

  auto common = [&](size_t i, EqState &s) {
    const InstRef &ref = oi[i];
    if (ref.fx.unknown_effect) {
      s = eq_bottom();
      return;
    }
    const uint64_t flags = ref.inst ? ref.inst->flags() : 0;
    const bool is_mem = (flags & MEMORY_OP) != 0;
    LocSet reads = ref.fx.reads;
    reads |= ref.fx.implicit_reads;
    // A VGPR write keeps lanes outside EXEC: the destination's old value is an input too.
    const VSet vin = ref.fx.vreads | ref.fx.vwrites;
    const bool inputs_equal = (reads & ~s.eq).none() && (!is_mem || s.mem) &&
                              (vin & ~s.veq).none() && !(ref.fx.reads_pc && diff.layout_shift);
    if (inputs_equal) {
      s.eq |= ref.fx.writes;
      s.veq |= ref.fx.vwrites;
    } else {
      s.eq &= ~ref.fx.writes;
      s.veq &= ~ref.fx.vwrites;
    }
    if (is_mem && !inputs_equal)
      s.mem = false;
  };
  std::vector<EqState> anchor_state(plan.regions.size());
  auto apply_atom = [&](const Atom &a, EqState &s) {
    if (a.region != SIZE_MAX)
      anchor_state[a.region] = s;
    if (s.top)
      return;
    switch (a.kind) {
    case Atom::Diverge:
      s = eq_bottom();
      break;
    case Atom::OpaqueRegion:
      if (a.unknown) {
        s = eq_bottom();
        break;
      }
      s.eq &= ~a.writes;
      s.mem = s.mem && !a.mem;
      s.veq &= ~a.vwrites;
      break;
    case Atom::ReadyRegion:
      s.eq &= ~a.writes;
      s.veq &= ~a.vwrites;
      if (!a.boundary) {
        s.eq |= a.observable;
        s.veq |= a.vobservable;
      }
      break;
    }
  };
  std::vector<EqState> before(oi.size());
  auto run_block = [&](uint32_t b, EqState s, bool record) {
    for (size_t i = blocks[b].first; i < blocks[b].end; ++i) {
      for (size_t id : atoms_at[i])
        apply_atom(atoms[id], s);
      if (owner[i] >= 0) {
        const Atom &a = atoms[static_cast<size_t>(owner[i])];
        if (a.anchor < oi.size() && oi[a.anchor].block != b && !s.top)
          s = eq_bottom(); // multi-block change: nothing known
        continue;
      }
      if (record)
        before[i] = s;
      if (!s.top)
        common(i, s);
    }
    return s;
  };
  auto diverges = [&](uint32_t b, const EqState &s) {
    const InstRef &last = oi[blocks[b].end - 1];
    if (!last.inst || !(last.inst->flags() & COND_BRANCH))
      return false;
    if (owner[blocks[b].end - 1] >= 0)
      return true;
    LocSet reads = last.fx.reads;
    reads |= last.fx.implicit_reads;
    return (reads & ~s.eq).any();
  };

  const EqState initial = metadata_differs ? eq_bottom() : eq_all();
  const size_t nb = blocks.size();
  std::vector<EqState> out(nb);
  auto runs_edge = [&](uint32_t p, uint32_t b) {
    return region_runs_edge(plan, map, blocks[p].end - 1, blocks[b].first);
  };
  bool changed = true;
  for (unsigned round = 0; changed && round < 4096; ++round) {
    changed = false;
    for (uint32_t b = 0; b < nb; ++b) {
      EqState in = b == 0 ? initial : EqState{};
      for (uint32_t p : blocks[b].predecessors) {
        // An edge inside a READY region with control flow (a loop back to its entry) is run
        // by the region itself.
        if (runs_edge(p, b))
          continue;
        EqState po = out[p];
        if (!po.top && diverges(p, po))
          po = eq_bottom();
        in = eq_meet(in, po);
      }
      if (!om.cfg_complete() && b != 0)
        in = eq_meet(in, eq_bottom()); // unknown edges may enter anywhere
      const EqState o = run_block(b, in, false);
      if (!(o == out[b])) {
        out[b] = o;
        changed = true;
      }
    }
  }
  for (uint32_t b = 0; b < nb; ++b) {
    EqState in = b == 0 ? initial : EqState{};
    for (uint32_t p : blocks[b].predecessors) {
      if (runs_edge(p, b))
        continue; // run inside one READY region
      EqState po = out[p];
      if (!po.top && diverges(p, po))
        po = eq_bottom();
      in = eq_meet(in, po);
    }
    if (!om.cfg_complete() && b != 0)
      in = eq_meet(in, eq_bottom());
    run_block(b, in, true);
  }
  return {std::move(anchor_state), std::move(before)};
}

// ---------------------------------------------------------------------------
// Known bits from the common prefix (handoff 3.3): a forward must-analysis over the old
// CFG. Common code is identical on both sides, so a fact proven here holds in both
// kernels at the cutpoint. Changed code (regions, untested changes) makes its writes
// unknown, and a control-flow change makes everything unknown. Path conditions are not
// used. Only a few exact transfer functions are modelled; every other write is unknown.

struct Bits {
  uint32_t kz = 0, ko = 0; ///< Known-zero and known-one masks.
  bool operator==(const Bits &) const = default;
};

struct BitsState {
  std::array<Bits, kNumLocs> v{};
  bool top = true;
  bool operator==(const BitsState &) const = default;
};

BitsState bits_unknown() {
  BitsState s;
  s.top = false;
  return s;
}

BitsState bits_meet(const BitsState &a, const BitsState &b) {
  if (a.top)
    return b;
  if (b.top)
    return a;
  BitsState r;
  r.top = false;
  for (unsigned l = 0; l < kNumLocs; ++l)
    r.v[l] = {a.v[l].kz & b.v[l].kz, a.v[l].ko & b.v[l].ko};
  return r;
}

Bits exact(uint32_t c) { return {~c, c}; }

uint32_t leading_known_zero(const Bits &b) { return static_cast<uint32_t>(std::countl_one(b.kz)); }

uint32_t top_mask(uint32_t n) { return n == 0 ? 0u : n >= 32 ? ~0u : ~0u << (32 - n); }

std::optional<Bits> transfer_bits(const InstRef &ref, const BitsState &s, const Target &target) {
  const Effects &fx = ref.fx;
  if (!fx.p0 || !ref.inst || (ref.inst->flags() & PREDICATED_DEF))
    return std::nullopt;
  LocSet dst = fx.writes;
  dst.reset(kLocScc);
  if (dst.count() != 1)
    return std::nullopt;
  std::vector<SourceOperand> src = source_operands(*ref.inst, target);
  std::erase_if(src, [](const SourceOperand &o) { return o.fieldless; });
  auto bits_of = [&](size_t i) -> std::optional<Bits> {
    if (i >= src.size())
      return std::nullopt;
    if (src[i].constant)
      return exact(*src[i].constant);
    if (src[i].locs.count() == 1)
      for (unsigned l = 0; l < kNumLocs; ++l)
        if (src[i].locs.test(l))
          return s.v[l];
    return std::nullopt;
  };
  const std::string &m = ref.mnemonic;
  auto a = bits_of(0), b = bits_of(1);
  if ((m == "s_mov_b32" || m == "s_movk_i32") && a)
    return a;
  if (m == "s_not_b32" && a)
    return Bits{a->ko, a->kz};
  if (!a || !b)
    return std::nullopt;
  if (m == "s_and_b32")
    return Bits{a->kz | b->kz, a->ko & b->ko};
  if (m == "s_andn2_b32")
    return Bits{a->kz | b->ko, a->ko & b->kz};
  if (m == "s_or_b32")
    return Bits{a->kz & b->kz, a->ko | b->ko};
  if (m == "s_xor_b32")
    return Bits{(a->kz & b->kz) | (a->ko & b->ko), (a->kz & b->ko) | (a->ko & b->kz)};
  if (m == "s_cselect_b32")
    return Bits{a->kz & b->kz, a->ko & b->ko};
  const bool b_exact = (b->kz | b->ko) == ~0u;
  if (m == "s_lshl_b32" && b_exact) {
    const uint32_t c = b->ko & 31u;
    return Bits{(a->kz << c) | ((c ? (1u << c) : 1u) - 1u), a->ko << c};
  }
  if (m == "s_lshr_b32" && b_exact) {
    const uint32_t c = b->ko & 31u;
    return Bits{(a->kz >> c) | top_mask(c), a->ko >> c};
  }
  if (m == "s_bfe_u32" && b_exact) {
    const uint32_t off = b->ko & 31u, width = (b->ko >> 16) & 0x7fu;
    if (width == 0)
      return exact(0);
    if (width >= 32)
      return Bits{(a->kz >> off) | top_mask(off), a->ko >> off};
    const uint32_t mask = (1u << width) - 1u;
    return Bits{((a->kz >> off) & mask) | ~mask, (a->ko >> off) & mask};
  }
  if (m == "s_min_u32")
    return Bits{top_mask(std::max(leading_known_zero(*a), leading_known_zero(*b))), 0};
  if (m == "s_max_u32")
    return Bits{top_mask(std::min(leading_known_zero(*a), leading_known_zero(*b))), 0};
  return std::nullopt;
}

struct KnownBitsResult {
  std::vector<BitsState> anchor_state; ///< Per region index.
  std::vector<BitsState> before;       ///< Per old instruction.
};

KnownBitsResult analyse_known_bits(const KernelModel &om, const KernelModel &nm,
                                   const DiffResult &diff, const PlanResult &plan) {
  const auto &oi = om.insts();
  const auto &blocks = om.blocks();
  const AtomMap map = build_atoms(om, nm, diff, plan);
  std::vector<BitsState> anchor_state(plan.regions.size());
  auto forget = [](BitsState &s, const LocSet &w) {
    for (unsigned l = 0; l < kNumLocs; ++l)
      if (w.test(l))
        s.v[l] = {};
  };
  auto apply_atom = [&](const Atom &a, BitsState &s) {
    if (a.region != SIZE_MAX)
      anchor_state[a.region] = s;
    if (s.top)
      return;
    if (a.kind == Atom::Diverge || a.unknown)
      s = bits_unknown();
    else
      forget(s, a.writes);
  };
  std::vector<BitsState> before(oi.size());
  auto run_block = [&](uint32_t b, BitsState s, bool record) {
    for (size_t i = blocks[b].first; i < blocks[b].end; ++i) {
      for (size_t id : map.atoms_at[i])
        apply_atom(map.atoms[id], s);
      if (map.owner[i] >= 0) {
        const Atom &a = map.atoms[static_cast<size_t>(map.owner[i])];
        if (a.anchor < oi.size() && oi[a.anchor].block != b && !s.top)
          s = bits_unknown();
        continue;
      }
      if (record)
        before[i] = s;
      if (s.top)
        continue;
      const InstRef &ref = oi[i];
      if (ref.fx.unknown_effect) {
        s = bits_unknown();
        continue;
      }
      const auto computed = transfer_bits(ref, s, om.artifact().target);
      forget(s, ref.fx.writes);
      if (computed) {
        LocSet dst = ref.fx.writes;
        dst.reset(kLocScc);
        for (unsigned l = 0; l < kNumLocs; ++l)
          if (dst.test(l))
            s.v[l] = *computed;
      }
    }
    return s;
  };
  const BitsState initial = bits_unknown();
  const size_t nb = blocks.size();
  std::vector<BitsState> out(nb);
  // An edge inside one region (a loop back to its entry) is run by the region itself.
  auto inside_edge = [&](uint32_t p, uint32_t b) {
    return region_runs_edge(plan, map, blocks[p].end - 1, blocks[b].first);
  };
  bool changed = true;
  for (unsigned round = 0; changed && round < 4096; ++round) {
    changed = false;
    for (uint32_t b = 0; b < nb; ++b) {
      BitsState in = b == 0 ? initial : BitsState{};
      for (uint32_t p : blocks[b].predecessors)
        if (!inside_edge(p, b))
          in = bits_meet(in, out[p]);
      if (!om.cfg_complete() && b != 0)
        in = bits_meet(in, bits_unknown());
      const BitsState o = run_block(b, in, false);
      if (!(o == out[b])) {
        out[b] = o;
        changed = true;
      }
    }
  }
  for (uint32_t b = 0; b < nb; ++b) {
    BitsState in = b == 0 ? initial : BitsState{};
    for (uint32_t p : blocks[b].predecessors)
      if (!inside_edge(p, b))
        in = bits_meet(in, out[p]);
    if (!om.cfg_complete() && b != 0)
      in = bits_meet(in, bits_unknown());
    run_block(b, in, true);
  }
  return {std::move(anchor_state), std::move(before)};
}

/// Turn known bits at a region entry into constraints on its entry_equal locations.
void derive_constraints(RegionPair &region, const BitsState &s) {
  if (s.top)
    return;
  RegionContract &k = region.contract;
  size_t added = 0;
  for (unsigned l = 0; l < kNumLocs; ++l) {
    if (l == kLocScc || !k.entry_equal.test(l))
      continue;
    const Bits &b = s.v[l];
    if ((b.kz | b.ko) == 0)
      continue;
    Constraint c;
    c.loc = l;
    c.known_zero = b.kz;
    c.known_one = b.ko;
    c.source = "common_prefix_analysis";
    k.constraints.push_back(c);
    ++added;
  }
  if (added)
    k.sources.push_back({std::to_string(added) +
                             " known-bits constraint(s) on entry locations proven by the common "
                             "prefix (forward must-analysis; changed code makes its writes "
                             "unknown)",
                         "common_prefix_analysis"});
}

void record_composition(RegionPair &region, const EqState &s, const std::string &how) {
  RegionContract &k = region.contract;
  const LocSet missing = s.top ? LocSet{} : (k.entry_equal & ~s.eq);
  const VSet vmissing = s.top ? VSet{} : (k.ventry_equal & ~s.veq);
  const bool mmissing = !s.top && k.memory_reads && !s.mem;
  if (missing.none() && vmissing.none() && !mmissing) {
    k.composition = "HOLDS";
    k.sources.push_back({"composition: every entry_equal location is established equal at the "
                         "entry by must-equality from the kernel entry through common code " +
                             how,
                         "composition_check"});
  } else {
    k.composition = "FAILS";
    std::string what = format_locs(missing);
    if (vmissing.any())
      what += (what.empty() ? "" : ",") + format_vgprs(vmissing);
    if (mmissing)
      what += (what.empty() ? "" : ",") + std::string("memory");
    k.sources.push_back({"composition: {" + what +
                             "} at the entry are not established equal by the kernel entry, "
                             "common code and upstream regions' R_out " +
                             how,
                         "composition_check"});
  }
}

} // namespace

PlanResult plan_regions(const KernelModel &old_model, const KernelModel &new_model,
                        const DiffResult &diff, const PlanOptions &options) {
  PlanResult result;
  const auto &oi = old_model.insts();
  const auto &ni = new_model.insts();
  const auto plan_start = std::chrono::steady_clock::now();
  auto out_of_time = [&] {
    return options.max_analysis_us != 0 && std::chrono::steady_clock::now() - plan_start >=
                                               std::chrono::microseconds(options.max_analysis_us);
  };

  bool metadata_differs = false;
  for (const auto &note : diff.notes) {
    if (note.semantic)
      result.unresolved.push_back({change_kind_name(note.kind), note.what, ""});
    else
      result.notes.push_back(note);
    if (note.semantic && note.kind == ChangeKind::Metadata)
      metadata_differs = true;
  }

  // 1. Seed candidates from instruction and control-flow hunks (a region with branches must
  // pass region_flow: one entry, direct branches, corresponding exits).
  std::vector<Candidate> candidates;
  std::vector<uint32_t> unresolved_blocks; // old blocks of control-flow changes
  size_t hint_hunks = 0;
  for (size_t h = 0; h < diff.hunks.size(); ++h) {
    const Hunk &hunk = diff.hunks[h];
    if (hunk.kind == ChangeKind::Hint) {
      ++hint_hunks; // no functional effect: nothing to test, common code to composition
      continue;
    }
    if (hunk.old_end >= oi.size() || hunk.new_end >= ni.size()) {
      result.unresolved.push_back(
          {change_kind_name(ChangeKind::ControlFlow),
           "hunk " + std::to_string(h) + " reaches the end of the kernel (no common exit anchor)",
           ""});
      unresolved_blocks.push_back(oi.empty() ? 0 : oi.back().block);
      continue;
    }
    if (hunk.kind != ChangeKind::Instruction && hunk.kind != ChangeKind::ControlFlow) {
      result.unresolved.push_back(
          {change_kind_name(hunk.kind),
           "hunk " + std::to_string(h) + " at old " +
               hex64(oi[hunk.old_begin < hunk.old_end ? hunk.old_begin : hunk.old_end].pc) + ": " +
               hunk.note,
           ""});
      unresolved_blocks.push_back(oi[std::min(hunk.old_begin, oi.size() - 1)].block);
      continue;
    }
    Candidate c;
    c.o = {hunk.old_begin, hunk.old_end, hunk.old_end};
    c.n = {hunk.new_begin, hunk.new_end, hunk.new_end};
    c.hunks.push_back(h);
    candidates.push_back(std::move(c));
  }
  if (hint_hunks)
    result.notes.push_back(
        {ChangeKind::Hint,
         std::to_string(hint_hunks) +
             " hunk(s) change only scheduling hints or dependency waits (s_delay_alu, s_nop, "
             "s_wait_alu, ...): no functional effect on an in-order wave, not tested; hardware "
             "hazards are not checked",
         false});

  // 2. Merge hunks that share a block pair, with the common code between them.
  std::vector<Candidate> merged;
  for (auto &c : candidates) {
    if (options.merge_same_block && !merged.empty()) {
      Candidate &prev = merged.back();
      const size_t ob = prev.o.begin, oe = c.o.end, nb = prev.n.begin, ne = c.n.end;
      const bool same_old = span_block(old_model, prev.o) == span_block(old_model, c.o) &&
                            single_block(old_model, ob, oe);
      const bool same_new = span_block(new_model, prev.n) == span_block(new_model, c.n) &&
                            single_block(new_model, nb, ne);
      const bool gap_p0 = !first_non_p0(old_model, prev.o.end, c.o.begin, "old") &&
                          !first_non_p0(new_model, prev.n.end, c.n.begin, "new");
      const bool fits = oe - ob <= options.max_region_insts && ne - nb <= options.max_region_insts;
      if (same_old && same_new && gap_p0 && fits) {
        prev.o.end = oe;
        prev.o.cut = c.o.cut;
        prev.n.end = ne;
        prev.n.cut = c.n.cut;
        prev.hunks.insert(prev.hunks.end(), c.hunks.begin(), c.hunks.end());
        continue;
      }
    }
    merged.push_back(std::move(c));
  }

  // 3. Manual backward context.
  for (auto &c : merged)
    for (size_t k = 0; k < options.context_insts && c.before < options.max_expansions; ++k)
      if (!step_back(old_model, new_model, diff, c))
        break;

  // 4. Build base regions.
  for (size_t r = 0; r < merged.size(); ++r) {
    RegionPair region = build_region(old_model, new_model, merged[r], options,
                                     "r" + std::to_string(r), {}, {}, &diff);
    if (region.status != RegionStatus::Ready)
      result.unresolved.push_back({"region", region.reason, region.id});
    result.regions.push_back(std::move(region));
  }

  // 5. Expansion variants (smallest first): backward, forward and both-sided common context.
  std::vector<size_t> variant_base;
  const bool skip_variants = options.auto_expand && out_of_time();
  if (skip_variants)
    result.skipped_for_time.push_back("expansion variants");
  if (options.auto_expand && !skip_variants) {
    for (size_t r = 0; r < merged.size(); ++r) {
      if (result.regions[r].status != RegionStatus::Ready)
        continue;
      Candidate probe = merged[r];
      size_t max_b = 0, max_f = 0;
      while (max_b < options.max_expansions && step_back(old_model, new_model, diff, probe))
        ++max_b;
      probe = merged[r];
      while (max_f < options.max_expansions && step_forward(old_model, new_model, diff, probe))
        ++max_f;
      std::set<std::pair<size_t, size_t>> shapes;
      for (size_t k : {1u, 2u, 4u, 8u, 16u, 32u, 64u}) {
        shapes.insert({std::min(k, max_b), 0});
        shapes.insert({0, std::min(k, max_f)});
        shapes.insert({std::min(k, max_b), std::min(k, max_f)});
      }
      shapes.insert({max_b, max_f});
      shapes.erase({0, 0});
      std::vector<std::pair<size_t, size_t>> ordered(shapes.begin(), shapes.end());
      std::ranges::stable_sort(ordered, {}, [](const auto &s) { return s.first + s.second; });
      if (ordered.size() > options.max_variants) {
        // Keep the largest contexts as well: when the small ones fail, the compensating
        // code is often further away (a reused VGPR overwritten a dozen instructions on).
        std::vector<std::pair<size_t, size_t>> largest;
        for (auto s : {std::pair{size_t{0}, max_f}, std::pair{max_b, max_f}})
          if (s != std::pair<size_t, size_t>{0, 0} &&
              std::ranges::find(largest, s) == largest.end())
            largest.push_back(s);
        ordered.resize(options.max_variants > largest.size() ? options.max_variants - largest.size()
                                                             : 0);
        for (const auto &s : largest)
          if (std::ranges::find(ordered, s) == ordered.end())
            ordered.push_back(s);
      }
      for (const auto &[b, f] : ordered) {
        Candidate v = merged[r];
        v.before = 0;
        for (size_t i = 0; i < b; ++i)
          step_back(old_model, new_model, diff, v);
        for (size_t i = 0; i < f; ++i)
          step_forward(old_model, new_model, diff, v);
        RegionPair variant =
            build_region(old_model, new_model, v, options,
                         result.regions[r].id + ".ctx-b" + std::to_string(v.before) + "f" +
                             std::to_string(v.after),
                         {}, {}, &diff);
        variant.parent = result.regions[r].id;
        result.variants.push_back(std::move(variant));
        variant_base.push_back(r);
      }
    }
  }

  // 6. Known bits from the common prefix become derived entry constraints.
  if (options.derive_constraints && out_of_time())
    result.skipped_for_time.push_back("derived constraints");
  else if (options.derive_constraints) {
    const KnownBitsResult kb = analyse_known_bits(old_model, new_model, diff, result);
    for (size_t r = 0; r < result.regions.size(); ++r)
      if (result.regions[r].status == RegionStatus::Ready)
        derive_constraints(result.regions[r], kb.anchor_state[r]);
    for (size_t v = 0; v < result.variants.size(); ++v) {
      RegionPair &variant = result.variants[v];
      const size_t r = variant_base[v];
      const RegionPair &base = result.regions[r];
      const size_t va = variant.old_side.begin < variant.old_side.end ? variant.old_side.begin
                                                                      : variant.old_side.cut;
      const size_t ba =
          base.old_side.begin < base.old_side.end ? base.old_side.begin : base.old_side.cut;
      if (variant.status == RegionStatus::Ready)
        derive_constraints(variant,
                           va == ba || va >= kb.before.size() ? kb.anchor_state[r] : kb.before[va]);
    }
  }

  // 7. Composition check and contract status. Without it no entry relation is established.
  if (out_of_time()) {
    result.skipped_for_time.push_back("composition check");
    auto unresolve = [](RegionPair &region) {
      region.contract.status = ContractStatus::Unresolved;
      region.contract.composition = "NOT_CHECKED";
      region.contract.sources.push_back({"entry relation not established: the analysis time limit "
                                         "was reached before the composition "
                                         "check",
                                         "planner"});
      region.prepared_key = compute_prepared_key(region);
    };
    for (auto &region : result.regions) {
      unresolve(region);
      if (region.status == RegionStatus::Ready)
        result.unresolved.push_back(
            {"context", "analysis time limit: composition not checked for " + region.id,
             region.id});
    }
    for (auto &variant : result.variants)
      unresolve(variant);
    if (options.diagnostic_entry_prefix)
      result.skipped_for_time.push_back("diagnostic entry-prefix regions");
    return result;
  }
  const CompositionResult comp =
      analyse_composition(old_model, new_model, diff, result, metadata_differs);
  // Changed code downstream of each base region's exit, and what it reads (either side).
  struct Downstream {
    std::vector<std::string> ids;
    LocSet reads;
    VSet vreads;
    bool mem = false;
  };
  std::vector<Downstream> downstream(result.regions.size());
  for (size_t r = 0; r < result.regions.size(); ++r) {
    const RegionPair &a = result.regions[r];
    const Span as{a.old_side.begin, a.old_side.end, a.old_side.cut};
    const uint32_t ab = span_block(old_model, as);
    const size_t aend = a.old_side.end;
    Downstream &d = downstream[r];
    auto absorb = [&](const KernelModel &m, size_t b, size_t e) {
      for (size_t i = b; i < e; ++i) {
        const Effects &fx = m.insts()[i].fx;
        if (fx.unknown_effect) {
          d.reads = all_locs();
          d.vreads.set();
        }
        d.reads |= fx.reads | fx.implicit_reads;
        d.vreads |= fx.vreads;
        d.mem = d.mem || fx.mem_read ||
                (m.insts()[i].inst && (m.insts()[i].inst->flags() & MEMORY_OP) != 0);
      }
    };
    for (size_t q = 0; q < result.regions.size(); ++q) {
      if (q == r)
        continue;
      const RegionPair &other = result.regions[q];
      const Span qs{other.old_side.begin, other.old_side.end, other.old_side.cut};
      const uint32_t qb = span_block(old_model, qs);
      const bool after = qb == ab ? anchor_index(qs) >= aend || in_cycle(old_model, ab)
                                  : old_model.reaches(ab, qb);
      if (!after)
        continue;
      d.ids.push_back(other.id);
      absorb(old_model, other.old_side.begin, other.old_side.end);
      absorb(new_model, other.new_side.begin, other.new_side.end);
    }
    for (uint32_t ub : unresolved_blocks)
      if (ub == ab || old_model.reaches(ab, ub)) {
        d.ids.push_back("control-flow change");
        d.reads = all_locs();
        d.vreads.set();
        d.mem = true;
      }
  }

  auto decide = [&](RegionPair &region, size_t base_index, const EqState &entry_state,
                    bool is_base) {
    RegionContract &k = region.contract;
    const RegionPair &base = result.regions[base_index];
    k.downstream_changed = downstream[base_index].ids;
    k.downstream_reads = downstream[base_index].reads;
    k.vdownstream_reads = downstream[base_index].vreads;
    k.downstream_memory_reads = downstream[base_index].mem;
    const uint32_t ob =
        span_block(old_model, {base.old_side.begin, base.old_side.end, base.old_side.cut});
    const size_t opos =
        base.old_side.begin < base.old_side.end ? base.old_side.begin : base.old_side.cut;
    bool cf_upstream = false;
    for (uint32_t ub : unresolved_blocks)
      if (ub == ob || old_model.reaches(ub, ob))
        cf_upstream = true;
    for (size_t q = 0; q < result.regions.size(); ++q) {
      if (q == base_index)
        continue;
      const RegionPair &other = result.regions[q];
      const Span qs{other.old_side.begin, other.old_side.end, other.old_side.cut};
      const uint32_t qb = span_block(old_model, qs);
      const size_t qpos = anchor_index(qs);
      const bool upstream =
          qb == ob ? qpos < opos || in_cycle(old_model, ob) : old_model.reaches(qb, ob);
      if (upstream)
        k.depends_on.push_back(other.id);
    }
    // A cycle through the entry block is a dependency on this region's own R_out, unless
    // every edge back to it comes from inside the region (the runner executes that loop).
    bool loop_inside = !base.contract.exits.empty();
    for (uint32_t pred : old_model.blocks()[ob].predecessors) {
      const size_t last = old_model.blocks()[pred].end - 1;
      if (old_model.reaches(ob, pred) && (last < base.old_side.begin || last >= base.old_side.end))
        loop_inside = false;
    }
    if (in_cycle(old_model, ob) && !loop_inside)
      k.depends_on.push_back(base.id + " (loop back-edge)");
    record_composition(
        region, entry_state,
        k.depends_on.empty() ? "(no upstream change)" : "(given the R_out of: " + [&] {
          std::string s;
          for (const auto &d : k.depends_on)
            s += (s.empty() ? "" : ", ") + d;
          return s;
        }() + ")");
    if (metadata_differs) {
      k.status = ContractStatus::Unresolved;
      k.sources.push_back({"entry relation not established: kernel metadata differs", "planner"});
    } else if (cf_upstream) {
      k.status = ContractStatus::Unresolved;
      k.sources.push_back({"entry relation not established: a control-flow change can reach "
                           "the region entry",
                           "planner"});
    } else if (k.composition == "FAILS") {
      k.status = ContractStatus::Unresolved;
    } else if (k.pending_effects == "CONFLICT" || k.pending_effects == "UNKNOWN") {
      k.status = ContractStatus::Unresolved;
    } else if (!k.depends_on.empty()) {
      k.status = ContractStatus::Assumed;
      bool opaque_upstream = false;
      for (const auto &d : k.depends_on)
        for (const auto &other : result.regions)
          opaque_upstream |= other.id == d && other.status != RegionStatus::Ready;
      k.sources.push_back(
          {"R_in relies on upstream regions' R_out" +
               std::string(opaque_upstream ? " and on untested upstream changes writing "
                                             "only their decoded destinations"
                                           : ""),
           "relation_summary (composition)"});
    } else if (!old_model.cfg_complete() || !new_model.cfg_complete()) {
      k.status = ContractStatus::Assumed;
      k.sources.push_back(
          {"single entry assumed: unknown successors could enter the region", "user_assumption"});
    } else {
      k.status = ContractStatus::Established;
      k.sources.push_back({"every path to the region entry runs byte-identical code from "
                           "identical kernel metadata, so all state is equal at the cutpoint",
                           "common_prefix_analysis"});
    }
    // A tested region whose entry context is not established keeps the scope partial.
    if (is_base && region.status == RegionStatus::Ready) {
      if (k.composition == "FAILS")
        result.unresolved.push_back(
            {"context", "composition fails at the entry of " + region.id, region.id});
      if (k.pending_effects == "CONFLICT" || k.pending_effects == "UNKNOWN")
        result.unresolved.push_back({"context",
                                     "memory operations outstanding at the entry of " + region.id +
                                         " (" + k.pending_effects + ")",
                                     region.id});
    }
    region.prepared_key = compute_prepared_key(region);
  };
  for (size_t r = 0; r < result.regions.size(); ++r)
    decide(result.regions[r], r, comp.anchor_state[r], true);
  for (size_t v = 0; v < result.variants.size(); ++v) {
    RegionPair &variant = result.variants[v];
    const size_t r = variant_base[v];
    const size_t va = variant.old_side.begin < variant.old_side.end ? variant.old_side.begin
                                                                    : variant.old_side.cut;
    const RegionPair &base = result.regions[r];
    const size_t ba =
        base.old_side.begin < base.old_side.end ? base.old_side.begin : base.old_side.cut;
    const EqState &s =
        va == ba || va >= comp.before.size() ? comp.anchor_state[r] : comp.before[va];
    decide(variant, r, s, false);
  }

  // 8. Explicit diagnostic path: the region preceded by everything from the kernel entry,
  // so the region's inputs are produced by executing the common prefix instead of being
  // sampled at the cut. Arbitrary entry values stand in for the dispatch ABI.
  if (options.diagnostic_entry_prefix) {
    for (const RegionPair &base : result.regions) {
      if (base.status != RegionStatus::Ready)
        continue;
      Candidate c;
      c.o = {0, base.old_side.end, base.old_side.cut};
      c.n = {0, base.new_side.end, base.new_side.cut};
      c.hunks = base.hunks;
      c.before = static_cast<uint32_t>(base.old_side.begin);
      RegionPair d =
          build_region(old_model, new_model, c, options, base.id + ".diag-entry", {}, {}, &diff);
      d.parent = base.id;
      const auto entry_block = [](const KernelModel &m, const RegionSide &s) {
        const size_t a = s.begin < s.end ? s.begin : s.cut;
        return a < m.insts().size() && m.insts()[a].block == m.insts()[0].block;
      };
      if (!entry_block(old_model, base.old_side) || !entry_block(new_model, base.new_side)) {
        d.status = RegionStatus::Unsupported;
        d.reason = "entry-prefix diagnostic: the region is not in the kernel's entry block, so "
                   "the prefix has control flow P0 does not execute";
      }
      RegionContract &k = d.contract;
      k.composition = "NOT_CHECKED";
      k.status = metadata_differs ? ContractStatus::Unresolved : ContractStatus::Assumed;
      k.sources.push_back({"diagnostic: execution starts at the kernel entry with arbitrary values "
                           "in every entry-live location; the dispatch ABI (kernarg pointer, "
                           "workgroup ids, ...) is not modelled",
                           "user_assumption"});
      d.prepared_key = compute_prepared_key(d);
      result.diagnostics.push_back(std::move(d));
    }
  }

  // 9. The whole function, from its entry to its final return. There is no cut: both
  // sides start from the same call or dispatch state, and the exit is the return, where
  // liveness is the calling convention's on both sides.
  const bool changed =
      std::ranges::any_of(diff.hunks, [](const Hunk &h) { return h.kind != ChangeKind::Hint; });
  if (options.whole_function && changed && !oi.empty() && !ni.empty()) {
    Candidate c;
    c.o = {0, oi.size() - 1, oi.size() - 1};
    c.n = {0, ni.size() - 1, ni.size() - 1};
    bool in_return = false;
    for (size_t h = 0; h < diff.hunks.size(); ++h) {
      const Hunk &hunk = diff.hunks[h];
      if (hunk.kind == ChangeKind::Hint)
        continue;
      c.hunks.push_back(h);
      in_return = in_return || hunk.old_end >= oi.size() || hunk.new_end >= ni.size();
    }
    PlanOptions fo = options;
    fo.max_region_insts = options.max_function_insts;
    RegionPair f = build_region(old_model, new_model, c, fo, "f0", {}, {}, &diff);
    auto returns = [](const KernelModel &m) {
      const InstRef &last = m.insts().back();
      return last.fx.control_flow && !last.fx.branch;
    };
    if (!returns(old_model) || !returns(new_model)) {
      f.status = RegionStatus::Unsupported;
      f.reason = "whole function: the last instruction is not a return or s_endpgm";
    } else if (in_return) {
      f.status = RegionStatus::Unsupported;
      f.reason = "whole function: the change reaches the final return itself";
    }
    RegionContract &k = f.contract;
    k.composition = "NOT_CHECKED";
    const bool kernel = is_kernel(old_model, new_model);
    if (kernel && k.vgpr_count) {
      // Its entry VGPRs are the workitem IDs (v0-v2, or x | y << 10 | z << 20 in v0): below
      // the workgroup size. Arbitrary values put LDS addresses outside the allocation; IDs
      // below 256 (four waves of 64, a one-dimensional workgroup) stay in its range.
      for (unsigned r = 0; r < 3 && r < k.vgpr_count; ++r)
        if (k.ventry_equal.test(r))
          k.vinput_masks.emplace_back(r, 0xffu);
      k.sources.push_back({"workitem IDs: a kernel's entry VGPRs v0-v2 are ANDed with 0xff (IDs "
                           "below 256 in one dimension; the dispatch's real IDs are below the "
                           "workgroup size)",
                           "whole_function"});
    }
    if (k.memory_reads || k.memory_writes) {
      // At the return a function's own stack frame is gone; at s_endpgm so are a kernel's
      // private memory and LDS.
      k.memory_dead_at_exit = kernel ? "private" : "frame";
      k.sources.push_back({k.memory_dead_at_exit == "frame"
                               ? "memory: bytes the function stores into its own stack frame "
                                 "(through the stack or frame pointer, no VGPR address) are dead "
                                 "at its return and not compared"
                               : "memory: the kernel's private memory and LDS are dead at "
                                 "s_endpgm and not compared",
                           "whole_function"});
    }
    if (metadata_differs) {
      k.status = ContractStatus::Unresolved;
      k.sources.push_back({"entry relation not established: kernel metadata differs", "planner"});
    } else if (k.pending_effects == "CONFLICT" || k.pending_effects == "UNKNOWN") {
      k.status = ContractStatus::Unresolved;
    } else if (!old_model.cfg_complete() || !new_model.cfg_complete()) {
      k.status = ContractStatus::Assumed;
      k.sources.push_back({"single entry assumed: unknown successors could enter the function "
                           "anywhere",
                           "user_assumption"});
    } else {
      k.status = ContractStatus::Established;
      k.sources.push_back({"whole function: execution starts at the entry, where both sides "
                           "receive the same state from the call or dispatch; entry values are "
                           "arbitrary (the ABI's specific values are not modelled), which only "
                           "widens the inputs",
                           "whole_function"});
    }
    f.prepared_key = compute_prepared_key(f);
    result.whole_function = std::move(f);
  }
  return result;
}

namespace {

/// Instruction index for a manual-region bound: a decimal index or a 0x-prefixed PC.
/// The kernel end (one past the last instruction) is accepted as an exclusive bound.
std::optional<size_t> manual_bound(const KernelModel &m, std::string_view tok) {
  const auto &insts = m.insts();
  if (tok.empty())
    return std::nullopt;
  const bool hex = tok.size() > 2 && tok[0] == '0' && (tok[1] == 'x' || tok[1] == 'X');
  uint64_t v = 0;
  const char *first = tok.data() + (hex ? 2 : 0);
  const char *last = tok.data() + tok.size();
  auto [ptr, ec] = std::from_chars(first, last, v, hex ? 16 : 10);
  if (ec != std::errc{} || ptr != last)
    return std::nullopt;
  if (!hex)
    return v <= insts.size() ? std::optional<size_t>(v) : std::nullopt;
  for (size_t i = 0; i < insts.size(); ++i)
    if (insts[i].pc == v)
      return i;
  if (!insts.empty() && v == insts.back().pc + 4 * insts.back().words.size())
    return insts.size();
  return std::nullopt;
}

} // namespace

std::optional<ManualRegion> parse_manual_region(std::string_view spec, const KernelModel &om,
                                                const KernelModel &nm, std::string &error) {
  ManualRegion mr;
  bool have_old = false, have_new = false;
  while (!spec.empty()) {
    const size_t comma = spec.find(',');
    const std::string_view item = spec.substr(0, comma);
    const size_t colon = item.find(':');
    const size_t dash = colon == std::string_view::npos ? colon : item.find('-', colon);
    if (colon == std::string_view::npos || dash == std::string_view::npos) {
      error = "manual region item '" + std::string(item) + "' is not side:B-E";
      return std::nullopt;
    }
    const std::string_view side = item.substr(0, colon);
    const bool is_old = side == "old";
    if (!is_old && side != "new") {
      error = "manual region side must be 'old' or 'new', got '" + std::string(side) + "'";
      return std::nullopt;
    }
    const KernelModel &m = is_old ? om : nm;
    auto b = manual_bound(m, item.substr(colon + 1, dash - colon - 1));
    auto e = manual_bound(m, item.substr(dash + 1));
    if (!b || !e || *b > *e) {
      error = "manual region '" + std::string(item) +
              "': bounds are not instruction indices/PCs of the kernel, or B > E";
      return std::nullopt;
    }
    if (*b == *e && *e >= m.insts().size()) {
      error = "manual region '" + std::string(item) +
              "': an empty side needs a following instruction as its cut point";
      return std::nullopt;
    }
    (is_old ? mr.old_begin : mr.new_begin) = *b;
    (is_old ? mr.old_end : mr.new_end) = *e;
    (is_old ? have_old : have_new) = true;
    if (comma == std::string_view::npos)
      break;
    spec.remove_prefix(comma + 1);
  }
  if (!have_old || !have_new) {
    error = "manual region needs both old:B-E and new:B-E";
    return std::nullopt;
  }
  if (mr.old_begin == mr.old_end && mr.new_begin == mr.new_end) {
    error = "manual region is empty on both sides";
    return std::nullopt;
  }
  return mr;
}

PlanResult plan_manual_region(const KernelModel &old_model, const KernelModel &new_model,
                              const DiffResult &diff, const ManualRegion &manual,
                              const PlanOptions &options) {
  PlanResult result;
  bool metadata_differs = false;
  for (const auto &note : diff.notes) {
    if (note.semantic)
      result.unresolved.push_back({change_kind_name(note.kind), note.what, ""});
    else
      result.notes.push_back(note);
    if (note.semantic && note.kind == ChangeKind::Metadata)
      metadata_differs = true;
  }
  Candidate c;
  c.o = {manual.old_begin, manual.old_end, manual.old_end};
  c.n = {manual.new_begin, manual.new_end, manual.new_end};
  for (size_t h = 0; h < diff.hunks.size(); ++h) {
    const Hunk &hunk = diff.hunks[h];
    const bool covered = manual.old_begin <= hunk.old_begin && hunk.old_end <= manual.old_end &&
                         manual.new_begin <= hunk.new_begin && hunk.new_end <= manual.new_end;
    if (covered) {
      c.hunks.push_back(h);
      continue;
    }
    if (hunk.kind == ChangeKind::Hint)
      continue;
    result.unresolved.push_back(
        {change_kind_name(hunk.kind),
         "hunk " + std::to_string(h) + " is not covered by the manual region (old [" +
             std::to_string(hunk.old_begin) + "," + std::to_string(hunk.old_end) + "), new [" +
             std::to_string(hunk.new_begin) + "," + std::to_string(hunk.new_end) + "))",
         ""});
  }
  RegionPair region = build_region(old_model, new_model, c, options, "m0", manual.entry_map,
                                   manual.exit_map, &diff);
  RegionContract &k = region.contract;
  k.composition = "NOT_CHECKED";
  k.sources.push_back(
      {"region boundaries chosen by hand: old [" + std::to_string(manual.old_begin) + "," +
           std::to_string(manual.old_end) + "), new [" + std::to_string(manual.new_begin) + "," +
           std::to_string(manual.new_end) + "); entry relation not checked by composition",
       "user_assumption"});
  if (metadata_differs) {
    k.status = ContractStatus::Unresolved;
    k.sources.push_back({"entry relation not established: kernel metadata differs", "planner"});
  } else if (k.pending_effects == "CONFLICT" || k.pending_effects == "UNKNOWN") {
    k.status = ContractStatus::Unresolved;
  } else {
    k.status = ContractStatus::Assumed;
  }
  if (region.status != RegionStatus::Ready)
    result.unresolved.push_back({"region", region.reason, region.id});
  region.prepared_key = compute_prepared_key(region);
  result.regions.push_back(std::move(region));
  return result;
}

} // namespace rocjitsu::isa_diff
