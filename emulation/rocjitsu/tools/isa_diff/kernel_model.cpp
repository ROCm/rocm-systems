// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "kernel_model.h"

#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/code_object.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "util/diagnostic.h"

#include <algorithm>
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
  // Outside P0: never a kill, and implicit EXEC/VCC/M0 consumers are assumed unless the
  // instruction is on the short list whose operands name everything it reads.
  if (fx.implicit_special_reads) {
    live.set(kLocExecLo);
    live.set(kLocExecHi);
    live.set(kLocVccLo);
    live.set(kLocVccHi);
    live.set(kLocM0);
  }
  return live;
}

std::unique_ptr<KernelModel> KernelModel::build(const Artifact &artifact, std::string &error) {
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

  model->compute_liveness();
  model->compute_taint();
  model->compute_reachability();
  return model;
}

LocSet KernelModel::live_before(size_t i) const {
  return live_transfer(insts_[i].fx, live_after_[i]);
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
}

void KernelModel::compute_taint() {
  // Forward may-analysis: which locations can hold a PC-derived value.
  const size_t nb = blocks_.size();
  std::vector<LocSet> taint_out(nb);
  std::vector<LocSet> taint_in(nb);
  auto transfer = [&](const Effects &fx, LocSet taint) {
    if (fx.unknown_effect)
      return all_locs();
    const bool derived = fx.reads_pc || (taint & fx.reads).any();
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
      LocSet t = in;
      for (size_t i = blocks_[b].first; i < blocks_[b].end; ++i)
        t = transfer(insts_[i].fx, t);
      if (in != taint_in[b] || t != taint_out[b]) {
        taint_in[b] = in;
        taint_out[b] = t;
        changed = true;
      }
    }
  }
  taint_before_.assign(insts_.size(), LocSet{});
  for (uint32_t b = 0; b < nb; ++b) {
    LocSet t = taint_in[b];
    for (size_t i = blocks_[b].first; i < blocks_[b].end; ++i) {
      taint_before_[i] = t;
      t = transfer(insts_[i].fx, t);
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
