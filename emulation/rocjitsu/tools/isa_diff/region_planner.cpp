// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "region_planner.h"

#include <algorithm>
#include <optional>

namespace rocjitsu::isa_diff {
namespace {

struct Span {
  size_t begin = 0, end = 0, cut = 0;
};

struct Candidate {
  Span o, n;
  std::vector<size_t> hunks;
  size_t expansions = 0;
};

/// The block a (possibly empty) side span executes in.
uint32_t span_block(const KernelModel &m, const Span &s) {
  const auto &insts = m.insts();
  if (s.begin < s.end)
    return insts[s.begin].block;
  return insts[std::min(s.cut, insts.size() - 1)].block;
}

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
    if (ref.fx.reads_pc || hit.any())
      return std::string(side) + " " + hex64(ref.pc) + " '" + ref.text +
             "' reads a PC-derived value (" + format_locs(hit) + ")";
  }
  return std::nullopt;
}

std::vector<RegionInst> region_insts(const KernelModel &m, size_t begin, size_t end) {
  std::vector<RegionInst> out;
  for (size_t i = begin; i < end; ++i) {
    const InstRef &ref = m.insts()[i];
    out.push_back({ref.pc, ref.words, ref.text});
  }
  return out;
}

LocSet entry_live(const KernelModel &m, const Span &s) {
  return m.live_before(s.begin < s.end ? s.begin : s.cut);
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

} // namespace

PlanResult plan_regions(const KernelModel &old_model, const KernelModel &new_model,
                        const DiffResult &diff, const PlanOptions &options) {
  PlanResult result;
  const auto &oi = old_model.insts();
  const auto &ni = new_model.insts();

  for (const auto &note : diff.notes) {
    if (note.semantic)
      result.unresolved.push_back({change_kind_name(note.kind), note.what, ""});
    else
      result.notes.push_back(note);
  }

  // 1. Seed candidates from instruction hunks; control-flow hunks are unresolved in P0.
  std::vector<Candidate> candidates;
  std::vector<std::pair<uint32_t, uint32_t>> unresolved_blocks; // (old block, new block)
  for (size_t h = 0; h < diff.hunks.size(); ++h) {
    const Hunk &hunk = diff.hunks[h];
    if (hunk.old_end >= oi.size() || hunk.new_end >= ni.size()) {
      result.unresolved.push_back(
          {change_kind_name(ChangeKind::ControlFlow),
           "hunk " + std::to_string(h) + " reaches the end of the kernel (no common exit anchor)",
           ""});
      unresolved_blocks.emplace_back(oi.empty() ? 0 : oi.back().block,
                                     ni.empty() ? 0 : ni.back().block);
      continue;
    }
    if (hunk.kind != ChangeKind::Instruction) {
      result.unresolved.push_back(
          {change_kind_name(hunk.kind),
           "hunk " + std::to_string(h) + " at old " +
               hex64(oi[hunk.old_begin < hunk.old_end ? hunk.old_begin : hunk.old_end].pc) + ": " +
               hunk.note,
           ""});
      unresolved_blocks.emplace_back(oi[std::min(hunk.old_begin, oi.size() - 1)].block,
                                     ni[std::min(hunk.new_begin, ni.size() - 1)].block);
      continue;
    }
    Candidate c;
    c.o = {hunk.old_begin, hunk.old_end, hunk.old_end};
    c.n = {hunk.new_begin, hunk.new_end, hunk.new_end};
    c.hunks.push_back(h);
    candidates.push_back(std::move(c));
  }

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

  // 3. Backward context over matched common instructions in the same block.
  for (auto &c : merged) {
    for (size_t k = 0; k < options.context_insts && c.expansions < options.max_expansions; ++k) {
      const size_t ob = c.o.begin < c.o.end ? c.o.begin : c.o.cut;
      const size_t nb = c.n.begin < c.n.end ? c.n.begin : c.n.cut;
      if (ob == 0 || nb == 0)
        break;
      const size_t po = ob - 1, pn = nb - 1;
      if (diff.old_to_new[po] != static_cast<int64_t>(pn))
        break;
      if (oi[po].block != oi[ob].block || ni[pn].block != ni[nb].block)
        break;
      if (oi[ob].leader || ni[nb].leader || oi[ob].branch_target || ni[nb].branch_target)
        break;
      if (!oi[po].fx.p0 || !ni[pn].fx.p0)
        break;
      if (c.o.begin == c.o.end)
        c.o.end = ob;
      if (c.n.begin == c.n.end)
        c.n.end = nb;
      c.o.begin = po;
      c.n.begin = pn;
      ++c.expansions;
    }
  }

  // 4. Validate and build contracts.
  for (size_t r = 0; r < merged.size(); ++r) {
    const Candidate &c = merged[r];
    RegionPair region;
    region.id = "r" + std::to_string(r);
    region.hunks = c.hunks;
    region.old_side = {c.o.begin, c.o.end, c.o.cut, region_insts(old_model, c.o.begin, c.o.end)};
    region.new_side = {c.n.begin, c.n.end, c.n.cut, region_insts(new_model, c.n.begin, c.n.end)};
    region.status = RegionStatus::Ready;

    std::optional<std::string> reject;
    if (!single_block(old_model, c.o.begin, c.o.end) ||
        !single_block(new_model, c.n.begin, c.n.end)) {
      // Renamed registers across blocks typically land here: no checked value mapping
      // exists, so the result is a mapping inconclusive rather than a guess.
      region.status = RegionStatus::Inconclusive;
      reject = "INCONCLUSIVE_MAPPING: region spans basic blocks; P0 only maps state at a "
               "common single-block cutpoint";
    } else if (c.o.end - c.o.begin > options.max_region_insts ||
               c.n.end - c.n.begin > options.max_region_insts) {
      reject = "region exceeds max_region_insts";
    } else if (auto bad = first_non_p0(old_model, c.o.begin, c.o.end, "old")) {
      reject = *bad;
    } else if (auto bad2 = first_non_p0(new_model, c.n.begin, c.n.end, "new")) {
      reject = *bad2;
    } else if (auto pc = pc_tainted_read(old_model, c.o.begin, c.o.end, "old")) {
      reject = *pc;
    } else if (auto pc2 = pc_tainted_read(new_model, c.n.begin, c.n.end, "new")) {
      reject = *pc2;
    }
    if (reject) {
      if (region.status == RegionStatus::Ready)
        region.status = RegionStatus::Unsupported;
      region.reason = *reject;
      result.unresolved.push_back({"region", *reject, region.id});
    }

    RegionContract &k = region.contract;
    k.entry_equal = entry_live(old_model, c.o) | entry_live(new_model, c.n);
    k.canary = ~k.entry_equal;
    const LocSet out_old = exit_live(old_model, c.o);
    const LocSet out_new = exit_live(new_model, c.n);
    k.observable = out_old | out_new;
    // The two continuations read different locations: they differ downstream (typically
    // a register renaming that crosses blocks), so equality on the same physical
    // locations at this exit is not a justified relation. Do not guess a mapping.
    if (region.status == RegionStatus::Ready && out_old != out_new) {
      region.status = RegionStatus::Inconclusive;
      region.reason = "INCONCLUSIVE_MAPPING: continuation liveness differs at the region exit "
                      "(old-only: " +
                      format_locs(out_old & ~out_new) +
                      "; new-only: " + format_locs(out_new & ~out_old) +
                      "); identity R_out is not justified and no checked value mapping exists";
      result.unresolved.push_back({"region", region.reason, region.id});
    }
    k.declared_writes_old = union_writes(old_model, c.o);
    k.declared_writes_new = union_writes(new_model, c.n);
    k.wide_reads = union_wide_reads(old_model, c.o) | union_wide_reads(new_model, c.n);
    k.sources.push_back({"entry_equal = live-in(old) | live-in(new) at the region entry",
                         "KernelModel backward LocSet liveness over BasicBlock::build_reachable "
                         "CFG"});
    k.sources.push_back({"observable = live-out(old) | live-out(new) at the region exit",
                         "KernelModel backward LocSet liveness"});
    k.sources.push_back({"mapping = same physical location on both sides",
                         "identity at a common cutpoint; matched anchors from LCS alignment"});
    k.sources.push_back({"canary = locations dead at entry on both sides",
                         "complement of entry_equal; independent per-side values"});
    if (c.expansions)
      k.sources.push_back({"region includes " + std::to_string(c.expansions) +
                               " common-prefix instruction(s) as context",
                           "common_prefix_analysis"});
    if (!old_model.cfg_complete() || !new_model.cfg_complete())
      k.sources.push_back({"some CFG successors are unknown; liveness made them all-live",
                           "conservative widening"});

    // Contract status from upstream changes.
    const uint32_t ob = span_block(old_model, c.o);
    const size_t opos = c.o.begin < c.o.end ? c.o.begin : c.o.cut;
    bool upstream_unresolved = false;
    for (const auto &note : diff.notes)
      if (note.semantic && note.kind == ChangeKind::Metadata)
        upstream_unresolved = true;
    for (const auto &[ub, nb2] : unresolved_blocks) {
      (void)nb2;
      if (ub != ob ? old_model.reaches(ub, ob) : true)
        upstream_unresolved = true;
    }
    for (size_t q = 0; q < merged.size(); ++q) {
      if (q == r)
        continue;
      const uint32_t qb = span_block(old_model, merged[q].o);
      const size_t qpos = merged[q].o.begin < merged[q].o.end ? merged[q].o.begin : merged[q].o.cut;
      const bool upstream =
          qb == ob ? qpos < opos || in_cycle(old_model, ob) : old_model.reaches(qb, ob);
      if (upstream)
        k.depends_on.push_back("r" + std::to_string(q));
    }
    if (in_cycle(old_model, ob))
      k.depends_on.push_back(region.id + " (loop back-edge)");
    if (upstream_unresolved) {
      k.status = ContractStatus::Unresolved;
      k.sources.push_back({"entry relation not established: an unresolved change (metadata, "
                           "control flow or unsupported region) can reach the region entry",
                           "planner"});
    } else if (!k.depends_on.empty()) {
      k.status = ContractStatus::Assumed;
      k.sources.push_back({"R_in relies on the R_out of upstream regions: " +
                               std::to_string(k.depends_on.size()) + " dependency(ies)",
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
    result.regions.push_back(std::move(region));
  }

  // An upstream region that is not READY has no tested R_out to compose with.
  for (auto &region : result.regions) {
    for (const auto &dep : region.contract.depends_on)
      for (const auto &other : result.regions)
        if (other.id == dep && other.status != RegionStatus::Ready &&
            region.contract.status != ContractStatus::Unresolved) {
          region.contract.status = ContractStatus::Unresolved;
          region.contract.sources.push_back(
              {"upstream region " + dep + " is " + region_status_name(other.status) +
                   ", so its R_out cannot establish this entry relation",
               "planner"});
        }
    region.prepared_key = compute_prepared_key(region);
  }
  return result;
}

} // namespace rocjitsu::isa_diff
