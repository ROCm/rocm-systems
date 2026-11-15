// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "kernel_model.h"

#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/code_object.h"
#include "rocjitsu/isa/arch/amdgpu/shared/wait_counter_policy.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "util/diagnostic.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <deque>
#include <map>
#include <unordered_map>

namespace rocjitsu::isa_diff {
namespace {

class TextSection : public Section {
public:
  TextSection(std::unique_ptr<char[]> data, size_t size, uint64_t vaddr)
      : Section(".text", std::move(data)), size_(size), vaddr_(vaddr) {}
  size_t size() const override { return size_; }
  uint64_t vaddr() const override { return vaddr_; }
  uint32_t sectionHeaderNameIdx() const override { return 0; }
  uint64_t sectionOffset() const override { return 0; }

private:
  size_t size_;
  uint64_t vaddr_;
};

/// A code object holding exactly the artifact's .text, so code objects and raw
/// word streams take the same CFG path.
class TextCodeObject : public CodeObject {
public:
  TextCodeObject(const std::vector<uint8_t> &text, uint64_t vaddr) {
    image_.resize(text.size());
    std::memcpy(image_.data(), text.data(), text.size());
    auto data = std::make_unique<char[]>(text.size());
    std::memcpy(data.get(), text.data(), text.size());
    sections_.push_back(std::make_unique<TextSection>(std::move(data), text.size(), vaddr));
    text_sections_.push_back(sections_.back().get());
  }
};

int64_t direct_branch_offset(const Instruction &inst, uint64_t offset) {
  // SOPP direct branches: target = pc + 4 + simm16 * 4.
  const uint64_t flags = inst.flags();
  if (!(flags & (BRANCH | COND_BRANCH)) || (flags & (INDIRECT_BRANCH | INDIRECT_CALL)))
    return -1;
  const uint32_t *raw = inst.raw_encoding();
  if (!raw || inst.size() != 4)
    return -1;
  const auto simm = static_cast<int16_t>(raw[0] & 0xffffu);
  return static_cast<int64_t>(offset) + 4 + int64_t{simm} * 4;
}

} // namespace

KernelModel::~KernelModel() = default;

LocSet KernelModel::live_transfer(const Effects &fx, const LocSet &live_after) {
  if (fx.unknown_effect)
    return all_locs();
  if (fx.p0)
    return (live_after & ~fx.writes) | fx.reads;
  LocSet live = live_after | fx.reads;
  // Outside the executable subset: never a kill, plus the implicit special reads of the
  // instruction class (see Effects::implicit_reads).
  return live | fx.implicit_reads;
}

std::unique_ptr<KernelModel> KernelModel::build(const Artifact &artifact, std::string &error) {
  const auto t_decode = std::chrono::steady_clock::now();
  std::unique_ptr<KernelModel> model(new KernelModel());
  model->artifact_ = artifact;
  model->code_object_ = std::make_unique<TextCodeObject>(artifact.text, artifact.text_vaddr);
  model->decoder_ = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  if (!model->decoder_) {
    error = "no gfx950 decoder linked";
    return nullptr;
  }
  if (artifact.kernel_end <= artifact.kernel_begin || artifact.kernel_end > artifact.text.size()) {
    error = "empty or out-of-range kernel";
    return nullptr;
  }
  const uint64_t entries[] = {artifact.kernel_begin};
  const BasicBlock::CodeRange ranges[] = {
      {artifact.kernel_begin, artifact.kernel_end - artifact.kernel_begin}};
  util::StringDiagnostic diag;
  auto built =
      BasicBlock::build_reachable(*model->code_object_, *model->decoder_, ROCJITSU_CODE_ARCH_CDNA4,
                                  entries, diag.emitter(), ranges);
  if (built.failed()) {
    error = "CFG construction failed: " + diag.message();
    return nullptr;
  }
  model->owned_blocks_ = std::move(built).value();

  // Linear order by offset.
  std::vector<std::pair<uint64_t, std::pair<const Instruction *, BasicBlock *>>> order;
  for (auto &block : model->owned_blocks_)
    for (const Instruction &inst : block->instructions())
      order.push_back({inst.src_loc(), {&inst, block.get()}});
  std::ranges::sort(order, {}, [](const auto &e) { return e.first; });

  std::unordered_map<const BasicBlock *, uint32_t> block_index;
  std::unordered_map<uint64_t, size_t> index_of_offset;
  auto &insts = model->insts_;
  for (const auto &[offset, pair] : order) {
    const auto [inst, block] = pair;
    if (!insts.empty() && insts.back().offset + insts.back().words.size() * 4 > offset) {
      error = "overlapping instruction boundaries at offset " + std::to_string(offset);
      return nullptr;
    }
    InstRef ref;
    ref.inst = inst;
    ref.offset = offset;
    ref.pc = artifact.text_vaddr + offset;
    const size_t nwords = static_cast<size_t>(inst->size()) / 4;
    ref.words.resize(nwords);
    std::memcpy(ref.words.data(), artifact.text.data() + offset, nwords * 4);
    ref.mnemonic = std::string(inst->mnemonic());
    ref.text = inst->disassemble();
    ref.fx = classify(*inst);
    for (const TextRelocation &rel : artifact.text_relocations)
      if (rel.offset >= offset && rel.offset < offset + nwords * 4)
        ref.relocations += (ref.relocations.empty() ? "" : "; ") + describe_relocation(rel);
    auto [it, inserted] = block_index.try_emplace(block, static_cast<uint32_t>(block_index.size()));
    ref.block = it->second;
    index_of_offset[offset] = insts.size();
    insts.push_back(std::move(ref));
  }
  if (insts.empty()) {
    error = "kernel decoded to no instructions";
    return nullptr;
  }

  // Blocks in first-instruction order (block_index follows linear order of first insts).
  auto &blocks = model->blocks_;
  blocks.resize(block_index.size());
  for (size_t i = 0; i < insts.size(); ++i) {
    BlockInfo &b = blocks[insts[i].block];
    if (i == 0 || insts[i - 1].block != insts[i].block) {
      if (b.end != 0) {
        error = "non-contiguous basic block";
        return nullptr;
      }
      b.first = i;
      insts[i].leader = true;
    }
    b.end = i + 1;
  }
  for (const auto &[block, index] : block_index) {
    BlockInfo &b = blocks[index];
    for (const BasicBlock *s : block->successors()) {
      auto it = block_index.find(s);
      if (it == block_index.end())
        b.unknown_successor = true;
      else
        b.successors.push_back(it->second);
    }
    if (block->successor_issue() != BasicBlock::SuccessorIssue::None ||
        !block->call_edges().empty())
      b.unknown_successor = true;
    const InstRef &last = insts[b.end - 1];
    if (last.fx.unknown_effect)
      b.unknown_successor = true;
    if (b.unknown_successor)
      model->cfg_complete_ = false;
  }
  for (uint32_t i = 0; i < blocks.size(); ++i)
    for (uint32_t s : blocks[i].successors)
      blocks[s].predecessors.push_back(i);

  // Direct branch targets.
  for (size_t i = 0; i < insts.size(); ++i) {
    const int64_t target = direct_branch_offset(*insts[i].inst, insts[i].offset);
    if (target < 0)
      continue;
    auto it = index_of_offset.find(static_cast<uint64_t>(target));
    if (it == index_of_offset.end())
      continue;
    insts[i].branch_target_index = static_cast<int64_t>(it->second);
    insts[it->second].branch_target = true;
  }
  for (const auto &ref : insts)
    if (ref.fx.reads_pc)
      model->reads_pc_ = true;

  // Bytes of the kernel range not covered by reachable instructions.
  uint64_t cursor = artifact.kernel_begin;
  for (const auto &ref : insts) {
    if (ref.offset > cursor)
      model->uncovered_.insert(model->uncovered_.end(),
                               artifact.text.begin() + static_cast<std::ptrdiff_t>(cursor),
                               artifact.text.begin() + static_cast<std::ptrdiff_t>(ref.offset));
    cursor = std::max(cursor, ref.offset + ref.words.size() * 4);
  }
  if (artifact.kernel_end > cursor)
    model->uncovered_.insert(
        model->uncovered_.end(), artifact.text.begin() + static_cast<std::ptrdiff_t>(cursor),
        artifact.text.begin() + static_cast<std::ptrdiff_t>(artifact.kernel_end));

  const auto t_dataflow = std::chrono::steady_clock::now();
  model->decode_cfg_ns_ = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(t_dataflow - t_decode).count());
  model->compute_liveness();
  model->compute_taint();
  model->compute_pending();
  model->compute_reachability();
  model->dataflow_ns_ = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                  std::chrono::steady_clock::now() - t_dataflow)
                                                  .count());
  return model;
}

LocSet KernelModel::live_before(size_t i) const {
  return live_transfer(insts_[i].fx, live_after_[i]);
}

VSet KernelModel::vlive_transfer(const Effects &fx, const VSet &live_after) {
  if (fx.unknown_effect) {
    VSet all;
    all.set();
    return all;
  }
  return live_after | fx.vreads;
}

VSet KernelModel::vlive_before(size_t i) const {
  return vlive_transfer(insts_[i].fx, vlive_after_[i]);
}

void KernelModel::compute_liveness() {
  const size_t nb = blocks_.size();
  std::vector<LocSet> live_in(nb), live_out(nb);
  auto block_out = [&](uint32_t b) {
    if (blocks_[b].unknown_successor)
      return all_locs();
    LocSet out;
    for (uint32_t s : blocks_[b].successors)
      out |= live_in[s];
    return out;
  };
  auto block_in = [&](uint32_t b, LocSet live) {
    for (size_t i = blocks_[b].end; i-- > blocks_[b].first;)
      live = live_transfer(insts_[i].fx, live);
    return live;
  };
  std::deque<uint32_t> work;
  std::vector<bool> queued(nb, true);
  for (uint32_t b = static_cast<uint32_t>(nb); b-- > 0;)
    work.push_back(b);
  while (!work.empty()) {
    const uint32_t b = work.front();
    work.pop_front();
    queued[b] = false;
    live_out[b] = block_out(b);
    const LocSet in = block_in(b, live_out[b]);
    if (in == live_in[b])
      continue;
    live_in[b] = in;
    for (uint32_t p : blocks_[b].predecessors)
      if (!queued[p]) {
        queued[p] = true;
        work.push_back(p);
      }
  }
  live_after_.assign(insts_.size(), LocSet{});
  for (uint32_t b = 0; b < nb; ++b) {
    LocSet live = block_out(b);
    for (size_t i = blocks_[b].end; i-- > blocks_[b].first;) {
      live_after_[i] = live;
      live = live_transfer(insts_[i].fx, live);
    }
  }

  // VGPRs: the same backward dataflow; writes never kill (partial under EXEC).
  std::vector<VSet> vin(nb), vout(nb);
  VSet vall;
  vall.set();
  auto vblock_out = [&](uint32_t b) {
    if (blocks_[b].unknown_successor)
      return vall;
    VSet out;
    for (uint32_t s : blocks_[b].successors)
      out |= vin[s];
    return out;
  };
  for (uint32_t b = static_cast<uint32_t>(nb); b-- > 0;)
    work.push_back(b);
  std::fill(queued.begin(), queued.end(), true);
  while (!work.empty()) {
    const uint32_t b = work.front();
    work.pop_front();
    queued[b] = false;
    vout[b] = vblock_out(b);
    VSet in = vout[b];
    for (size_t i = blocks_[b].end; i-- > blocks_[b].first;)
      in = vlive_transfer(insts_[i].fx, in);
    if (in == vin[b])
      continue;
    vin[b] = in;
    for (uint32_t p : blocks_[b].predecessors)
      if (!queued[p]) {
        queued[p] = true;
        work.push_back(p);
      }
  }
  vlive_after_.assign(insts_.size(), VSet{});
  for (uint32_t b = 0; b < nb; ++b) {
    VSet live = vblock_out(b);
    for (size_t i = blocks_[b].end; i-- > blocks_[b].first;) {
      vlive_after_[i] = live;
      live = vlive_transfer(insts_[i].fx, live);
    }
  }
}

void KernelModel::compute_taint() {
  // Forward may-analysis: which locations can hold a PC-derived value.
  const size_t nb = blocks_.size();
  std::vector<LocSet> taint_out(nb);
  std::vector<LocSet> taint_in(nb);
  // Scalar and vector taint travel together: a VGPR holding a PC-derived value can flow
  // back into SGPRs (v_readfirstlane) and the reverse. VGPR writes never clear taint
  // (inactive lanes keep their value).
  std::vector<VSet> vtaint_out(nb);
  VSet vall;
  vall.set();
  VSet vt_cur;
  auto transfer = [&](const Effects &fx, LocSet taint) {
    if (fx.unknown_effect) {
      vt_cur = vall;
      return all_locs();
    }
    const bool derived = fx.reads_pc || (taint & fx.reads).any() || (vt_cur & fx.vreads).any();
    if (derived)
      vt_cur |= fx.vwrites;
    if (fx.p0)
      return derived ? (taint | fx.writes) : (taint & ~fx.writes);
    return derived ? (taint | fx.writes) : taint;
  };
  bool changed = true;
  while (changed) {
    changed = false;
    for (uint32_t b = 0; b < nb; ++b) {
      LocSet in;
      for (uint32_t p : blocks_[b].predecessors)
        in |= taint_out[p];
      VSet vin;
      for (uint32_t p : blocks_[b].predecessors)
        vin |= vtaint_out[p];
      vt_cur = vin;
      LocSet t = in;
      for (size_t i = blocks_[b].first; i < blocks_[b].end; ++i)
        t = transfer(insts_[i].fx, t);
      if (in != taint_in[b] || t != taint_out[b] || vt_cur != vtaint_out[b]) {
        taint_in[b] = in;
        taint_out[b] = t;
        vtaint_out[b] = vt_cur;
        changed = true;
      }
    }
  }
  taint_before_.assign(insts_.size(), LocSet{});
  vtaint_before_.assign(insts_.size(), VSet{});
  for (uint32_t b = 0; b < nb; ++b) {
    LocSet t = taint_in[b];
    vt_cur = VSet{};
    for (uint32_t p : blocks_[b].predecessors)
      vt_cur |= vtaint_out[p];
    for (size_t i = blocks_[b].first; i < blocks_[b].end; ++i) {
      taint_before_[i] = t;
      vtaint_before_[i] = vt_cur;
      t = transfer(insts_[i].fx, t);
    }
  }
}

namespace {

/// Counters an issued memory operation increments on gfx9 (CDNA): SMEM and LDS/GDS use
/// lgkmcnt, VMEM uses vmcnt, FLAT uses both; an unclassified producer is assumed to use all.
std::array<bool, PendingState::kNumCounters> producer_counters(const Instruction &inst) {
  const uint64_t f = inst.flags();
  std::array<bool, PendingState::kNumCounters> c{};
  if (!(f & (MEMORY_OP | MEMORY_WAIT_PRODUCER | SMEM | VMEM | LDS)))
    return c;
  if (f & (SMEM | LDS))
    c[PendingState::kLgkm] = true;
  if (f & VMEM)
    c[PendingState::kVm] = true;
  if (!c[0] && !c[1] && !c[2])
    c = {true, true, true};
  return c;
}

PendingState pending_transfer(const InstRef &ref, PendingState s) {
  const Instruction &inst = *ref.inst;
  if (ref.mnemonic == "s_waitcnt" && inst.num_src_operands() >= 1) {
    if (auto v = inst.src_operand(0)->const_value()) {
      const auto f = amdgpu::decode_gfx9_waitcnt(static_cast<uint32_t>(*v));
      const uint32_t limit[PendingState::kNumCounters] = {f.vmcnt, f.lgkmcnt, f.expcnt};
      for (unsigned c = 0; c < PendingState::kNumCounters; ++c) {
        s.count[c] = static_cast<uint8_t>(std::min<uint32_t>(s.count[c], limit[c]));
        if (s.count[c] == 0) {
          s.dests[c].reset();
          s.vdests[c].reset();
          s.unknown[c] = false;
        }
      }
    }
    return s;
  }
  const auto counters = producer_counters(inst);
  for (unsigned c = 0; c < PendingState::kNumCounters; ++c) {
    if (!counters[c])
      continue;
    s.count[c] = static_cast<uint8_t>(std::min<unsigned>(s.count[c] + 1u, 63u));
    s.dests[c] |= ref.fx.writes;
    s.vdests[c] |= ref.fx.vwrites;
    s.unknown[c] = s.unknown[c] || ref.fx.unknown_effect;
    s.example_pc = ref.pc;
  }
  return s;
}

PendingState pending_join(const PendingState &a, const PendingState &b) {
  PendingState r;
  for (unsigned c = 0; c < PendingState::kNumCounters; ++c) {
    r.count[c] = std::max(a.count[c], b.count[c]);
    r.dests[c] = a.dests[c] | b.dests[c];
    r.vdests[c] = a.vdests[c] | b.vdests[c];
    r.unknown[c] = a.unknown[c] || b.unknown[c];
  }
  r.example_pc = a.example_pc ? a.example_pc : b.example_pc;
  return r;
}

} // namespace

void KernelModel::compute_pending() {
  const size_t nb = blocks_.size();
  std::vector<PendingState> in(nb), out(nb);
  std::vector<bool> visited(nb, false);
  // Forward may-analysis to a fixpoint; counts saturate, so the lattice is finite.
  bool changed = true;
  for (unsigned round = 0; changed && round < 4096; ++round) {
    changed = false;
    for (uint32_t b = 0; b < nb; ++b) {
      PendingState s;
      bool have = b == 0; // the kernel entry starts with nothing outstanding
      for (uint32_t p : blocks_[b].predecessors) {
        if (!visited[p])
          continue;
        s = have ? pending_join(s, out[p]) : out[p];
        have = true;
      }
      if (!have)
        continue;
      PendingState t = s;
      for (size_t i = blocks_[b].first; i < blocks_[b].end; ++i)
        t = pending_transfer(insts_[i], t);
      if (!visited[b] || !(s == in[b]) || !(t == out[b])) {
        in[b] = s;
        out[b] = t;
        visited[b] = true;
        changed = true;
      }
    }
  }
  pending_before_.assign(insts_.size(), PendingState{});
  for (uint32_t b = 0; b < nb; ++b) {
    PendingState t = in[b];
    if (!visited[b]) {
      // Reached only through edges the CFG could not resolve: assume the worst.
      for (unsigned c = 0; c < PendingState::kNumCounters; ++c) {
        t.count[c] = 63;
        t.dests[c] = all_locs();
        t.vdests[c].set();
        t.unknown[c] = true;
      }
    }
    for (size_t i = blocks_[b].first; i < blocks_[b].end; ++i) {
      pending_before_[i] = t;
      t = pending_transfer(insts_[i], t);
    }
  }
}

void KernelModel::compute_reachability() {
  const size_t nb = blocks_.size();
  reach_.assign(nb, std::vector<bool>(nb, false));
  for (uint32_t s = 0; s < nb; ++s) {
    std::deque<uint32_t> work{s};
    reach_[s][s] = true;
    while (!work.empty()) {
      const uint32_t b = work.front();
      work.pop_front();
      for (uint32_t n : blocks_[b].successors)
        if (!reach_[s][n]) {
          reach_[s][n] = true;
          work.push_back(n);
        }
    }
  }
}

bool KernelModel::reaches(uint32_t from, uint32_t to) const {
  if (from >= reach_.size() || to >= reach_.size())
    return true;
  // An unknown successor anywhere may reach anything.
  return reach_[from][to] || !cfg_complete_;
}

} // namespace rocjitsu::isa_diff
