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

std::string pending_counter_name(unsigned c, const Target &target) {
  if (target.major < 12) {
    if (c == PendingState::kLoad)
      return "vmcnt";
    if (c == PendingState::kDs)
      return "lgkmcnt";
    if (c == static_cast<unsigned>(amdgpu::WaitCounterKind::Store))
      return "vscnt";
  }
  return std::string(amdgpu::wait_counter_name(static_cast<amdgpu::WaitCounterKind>(c)));
}

LocSet return_observable_locs(const Target &target) {
  // CSR_AMDGPU_SGPRs (AMDGPUCallingConv.td): s30-s39, s48-s55, s64-s71, s80-s87, s96-s105;
  // GFX9 addresses s0-s101. The caller also expects its EXEC back.
  LocSet l;
  for (auto [lo, hi] : {std::pair{30u, 39u}, {48u, 55u}, {64u, 71u}, {80u, 87u}, {96u, 105u}})
    for (unsigned s = lo; s <= hi && s < target.sgprs(); ++s)
      l.set(s);
  l.set(kLocExecLo);
  l.set(kLocExecHi);
  return l;
}

VSet return_observable_vgprs(unsigned return_vgprs) {
  // RetCC_AMDGPU_Func returns in v0..v31; CSR_AMDGPU_VGPRs are v40-v47, v56-v63, ...
  VSet v;
  for (unsigned r = 0; r < std::min(return_vgprs, 32u); ++r)
    v.set(r);
  for (unsigned base = 40; base + 8 <= kNumArchVgprs; base += 16)
    for (unsigned r = base; r < base + 8; ++r)
      v.set(r);
  return v;
}

LocSet KernelModel::return_live_locs() const {
  const Target &t = artifact_.target;
  LocSet l;
  if (artifact_.calling_convention == "shader") {
    for (unsigned s = 0; s < artifact_.return_sgprs && s < t.sgprs(); ++s)
      l.set(s);
  } else if (artifact_.calling_convention == "gfx") {
    // CSR_AMDGPU_SI_Gfx_SGPRs: s4-s31, s64-s105.
    for (auto [lo, hi] : {std::pair{4u, 31u}, {64u, 105u}})
      for (unsigned s = lo; s <= hi && s < t.sgprs(); ++s)
        l.set(s);
  } else {
    return return_observable_locs(t);
  }
  l.set(kLocExecLo);
  l.set(kLocExecHi);
  return l;
}

namespace {
/// CSR_AMDGPU_AGPRs: a32..a255 are callee-saved on targets with gfx90a instructions (CDNA2
/// and later); gfx908 saves none.
VSet callee_saved_agprs(const Target &t) {
  VSet v;
  if (!t.unified_vgprs)
    return v;
  for (unsigned a = 32; a < kNumArchVgprs; ++a)
    v.set(kAgprBase + a);
  return v;
}
} // namespace

VSet KernelModel::callee_saved_vgprs() const {
  return artifact_.callable && artifact_.calling_convention != "shader"
             ? return_observable_vgprs(0) | callee_saved_agprs(artifact_.target)
             : VSet{};
}

VSet KernelModel::return_live_vgprs() const {
  if (artifact_.calling_convention != "shader")
    return return_observable_vgprs(artifact_.return_vgprs) | callee_saved_agprs(artifact_.target);
  VSet v;
  for (unsigned r = 0; r < artifact_.return_vgprs && r < kNumVgprs; ++r)
    v.set(r);
  return v;
}

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
  const Target &target = artifact.target;
  model->decoder_ = Decoder::create(target.arch);
  if (!model->decoder_) {
    error = "no " + target.name + " decoder linked";
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
  auto built = BasicBlock::build_reachable(*model->code_object_, *model->decoder_, target.arch,
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
    ref.fx = classify(*inst, target);
    if (ref.mnemonic == "s_set_vgpr_msb")
      model->vgpr_msb_ = true;
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

  // Comment-only inline-asm blocks become zero-size pseudo-instructions. A block runs
  // before the instruction at its offset, in that instruction's basic block, unless a label
  // separates them: then it ends the preceding block (fall-through into the label). After
  // a branch it would sit on the fall-through edge alone, which no block represents; its
  // effects are then unknown. Blocks not directly before a reachable instruction are dead.
  if (!artifact.inline_asm.empty()) {
    std::vector<InstRef> merged;
    merged.reserve(insts.size() + artifact.inline_asm.size());
    std::vector<bool> ends_previous;
    size_t k = 0;
    const auto &blocks_asm = artifact.inline_asm;
    for (size_t i = 0; i < insts.size(); ++i) {
      while (k < blocks_asm.size() && blocks_asm[k].offset <= insts[i].offset) {
        const InlineAsmBlock &b = blocks_asm[k];
        const int index = static_cast<int>(k++);
        if (b.offset != insts[i].offset)
          continue; // not directly before reachable code
        InstRef ref;
        ref.offset = b.offset;
        ref.pc = insts[i].pc;
        ref.mnemonic = "inline_asm";
        ref.inline_asm = index;
        ref.asm_text = b.text;
        std::string flat = b.text;
        std::ranges::replace(flat, '\n', ' ');
        ref.text = "inline asm: " + flat;
        ref.fx = classify_inline_asm(b.text);
        ref.block = insts[i].block;
        bool previous = false;
        if (b.before_label) {
          const InstRef *prev = merged.empty() ? nullptr : &merged.back();
          if (prev && !prev->fx.control_flow) {
            ref.block = prev->block;
            previous = true;
          } else {
            ref.fx = Effects{};
            ref.fx.unknown_effect = true;
            ref.fx.reject_reason = "inline asm on a branch's fall-through edge";
          }
        }
        merged.push_back(std::move(ref));
        ends_previous.push_back(previous);
      }
      merged.push_back(std::move(insts[i]));
      ends_previous.push_back(false);
    }
    insts = std::move(merged);
    index_of_offset.clear();
    for (size_t i = 0; i < insts.size(); ++i)
      if (!ends_previous[i])
        index_of_offset.try_emplace(insts[i].offset, i);
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
    // A callable function returns through the address it was called with (s_set_pc_i64 is
    // the GFX12.5 name).
    // A shader part returns to its epilog at s_endpgm (the listing ends there).
    if (artifact.callable && artifact.calling_convention == "shader" && last.inst &&
        (last.inst->flags() & PROGRAM_TERMINATOR)) {
      b.returns = true;
      b.unknown_successor = false;
      b.successors.clear();
    }
    if (artifact.callable && last.inst &&
        (last.mnemonic == "s_setpc_b64" || last.mnemonic == "s_set_pc_i64") &&
        !last.fx.unknown_effect) {
      LocSet ra;
      ra.set(30);
      ra.set(31);
      if (last.fx.reads == ra) {
        b.returns = true;
        b.unknown_successor = false;
        b.successors.clear();
      }
    }
    if (b.unknown_successor)
      model->cfg_complete_ = false;
  }
  for (uint32_t i = 0; i < blocks.size(); ++i)
    for (uint32_t s : blocks[i].successors)
      blocks[s].predecessors.push_back(i);

  // Direct branch targets.
  for (size_t i = 0; i < insts.size(); ++i) {
    if (!insts[i].inst)
      continue;
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

  // GFX12.5 s_set_vgpr_msb selects which 256-VGPR bank later operands name: decoded VGPR
  // numbers are then not the registers accessed, so nothing that names a VGPR is modelled.
  if (model->vgpr_msb_)
    for (InstRef &ref : insts)
      if (ref.inst &&
          (ref.fx.vreads.any() || ref.fx.vwrites.any() || ref.mnemonic == "s_set_vgpr_msb")) {
        ref.fx.unknown_effect = true;
        ref.fx.p0 = false;
        ref.fx.reject_reason = "VGPR bank selected by s_set_vgpr_msb (GFX12.5 VGPRs above "
                               "v255): register numbers are dynamic";
        ref.fx.implicit_reads = all_locs();
        ref.fx.implicit_special_reads = true;
      }

  const auto t_dataflow = std::chrono::steady_clock::now();
  model->decode_cfg_ns_ = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(t_dataflow - t_decode).count());
  model->compute_exec_stability();
  model->compute_liveness();
  model->compute_undefined();
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

VSet KernelModel::vlive_step(size_t i, const VSet &live_after) const {
  const Effects &fx = insts_[i].fx;
  if (fx.unknown_effect) {
    VSet all;
    all.set();
    return all;
  }
  return (i < vkill_.size() ? live_after & ~vkill_[i] : live_after) | fx.vreads;
}

VSet KernelModel::vlive_before(size_t i) const { return vlive_step(i, vlive_after_[i]); }

LocSet KernelModel::live_before_strict(size_t i) const {
  return live_transfer(insts_[i].fx, live_after_strict_[i]);
}

void KernelModel::compute_liveness() {
  // Scalars: a backward dataflow; a second pass leaves out the return SGPRs a partial
  // return mask covers.
  const LocSet sret = return_live_locs();
  LocSet spartial;
  for (size_t s = 0; s < artifact_.return_sgpr_masks.size() && s < artifact_.return_sgprs; ++s)
    if (artifact_.return_sgpr_masks[s] != 0xffffffffu)
      spartial.set(s);
  live_after_ = scalar_liveness(sret);
  live_after_strict_ = spartial.any() ? scalar_liveness(sret & ~spartial) : live_after_;

  // VGPRs: the same backward dataflow; writes never kill (partial under EXEC). A second
  // pass leaves out the return VGPRs a partial return mask covers, so the planner can tell
  // which VGPRs are observable only through such a narrow return value.
  const VSet ret = return_live_vgprs();
  VSet partial;
  for (size_t r = 0; r < artifact_.return_masks.size() && r < artifact_.return_vgprs; ++r)
    if (artifact_.return_masks[r] != 0xffffffffu)
      partial.set(r);
  vlive_after_ = vgpr_liveness(ret);
  vlive_after_strict_ = partial.any() ? vgpr_liveness(ret & ~partial) : vlive_after_;
}

std::vector<LocSet> KernelModel::scalar_liveness(const LocSet &return_live) const {
  const size_t nb = blocks_.size();
  std::vector<LocSet> live_in(nb), live_out(nb);
  auto block_out = [&](uint32_t b) {
    if (blocks_[b].unknown_successor)
      return all_locs();
    if (blocks_[b].returns)
      return return_live;
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
  std::vector<LocSet> after(insts_.size());
  for (uint32_t b = 0; b < nb; ++b) {
    LocSet live = block_out(b);
    for (size_t i = blocks_[b].end; i-- > blocks_[b].first;) {
      after[i] = live;
      live = live_transfer(insts_[i].fx, live);
    }
  }
  return after;
}

std::vector<VSet> KernelModel::vgpr_liveness(const VSet &return_live) const {
  const size_t nb = blocks_.size();
  std::vector<VSet> vin(nb), vout(nb);
  VSet vall;
  vall.set();
  auto vblock_out = [&](uint32_t b) {
    if (blocks_[b].unknown_successor)
      return vall;
    if (blocks_[b].returns)
      return return_live;
    VSet out;
    for (uint32_t s : blocks_[b].successors)
      out |= vin[s];
    return out;
  };
  std::deque<uint32_t> work;
  std::vector<bool> queued(nb, true);
  for (uint32_t b = static_cast<uint32_t>(nb); b-- > 0;)
    work.push_back(b);
  while (!work.empty()) {
    const uint32_t b = work.front();
    work.pop_front();
    queued[b] = false;
    vout[b] = vblock_out(b);
    VSet in = vout[b];
    for (size_t i = blocks_[b].end; i-- > blocks_[b].first;)
      in = vlive_step(i, in);
    if (in == vin[b])
      continue;
    vin[b] = in;
    for (uint32_t p : blocks_[b].predecessors)
      if (!queued[p]) {
        queued[p] = true;
        work.push_back(p);
      }
  }
  std::vector<VSet> after(insts_.size());
  for (uint32_t b = 0; b < nb; ++b) {
    VSet live = vblock_out(b);
    for (size_t i = blocks_[b].end; i-- > blocks_[b].first;) {
      after[i] = live;
      live = vlive_step(i, live);
    }
  }
  return after;
}

VSet KernelModel::vlive_before_strict(size_t i) const {
  return vlive_step(i, vlive_after_strict_[i]);
}

VSet KernelModel::vundef_before(size_t i) const {
  return i < vwritten_before_.size() ? entry_vundef_ & ~vwritten_before_[i] : VSet{};
}

LocSet KernelModel::undef_before(size_t i) const {
  return i < written_before_.size() ? entry_undef_ & ~written_before_[i] : LocSet{};
}

void KernelModel::compute_undefined() {
  // VGPRs defined at the entry: the argument VGPRs, and for a callable function the
  // workitem IDs (v31) and the callee-saved VGPRs (the caller's values). The rest hold
  // whatever the caller or the previous wave left: LLVM undef until written.
  if (artifact_.entry_vgprs < 0 || !cfg_complete_)
    return; // all defined (raw words), or edges the analysis cannot see
  // Scalars: s{entry_sgprs}.. (but a callable function's callee-saved SGPRs), VCC, M0, SCC.
  if (artifact_.entry_sgprs >= 0) {
    LocSet sdef;
    for (int l = 0; l < artifact_.entry_sgprs && l < static_cast<int>(artifact_.target.sgprs());
         ++l)
      sdef.set(static_cast<size_t>(l));
    if (artifact_.callable && artifact_.calling_convention != "shader")
      sdef |= return_live_locs(); // callee-saved SGPRs and EXEC
    sdef.set(kLocExecLo);
    sdef.set(kLocExecHi);
    entry_undef_ = ~sdef;
  }
  VSet defined;
  for (int r = 0; r < artifact_.entry_vgprs && r < static_cast<int>(kNumVgprs); ++r)
    defined.set(static_cast<size_t>(r));
  if (artifact_.callable && artifact_.calling_convention != "shader") {
    defined.set(31);
    defined |= callee_saved_vgprs(); // callee-saved stripes
  }
  entry_vundef_ = ~defined;
  // Forward may-analysis: VGPRs written on some path from the entry.
  const size_t nb = blocks_.size();
  std::vector<VSet> in(nb), out(nb);
  auto block_writes = [&](uint32_t b, VSet w) {
    for (size_t i = blocks_[b].first; i < blocks_[b].end; ++i) {
      if (insts_[i].fx.unknown_effect)
        w.set();
      w |= insts_[i].fx.vwrites;
    }
    return w;
  };
  for (bool changed = true; changed;) { // union of writes: monotone, converges
    changed = false;
    for (uint32_t b = 0; b < nb; ++b) {
      VSet i;
      for (uint32_t p : blocks_[b].predecessors)
        i |= out[p];
      const VSet o = block_writes(b, i);
      if (i != in[b] || o != out[b]) {
        in[b] = i;
        out[b] = o;
        changed = true;
      }
    }
  }
  vwritten_before_.assign(insts_.size(), VSet{});
  for (uint32_t b = 0; b < nb; ++b) {
    VSet w = in[b];
    for (size_t i = blocks_[b].first; i < blocks_[b].end; ++i) {
      vwritten_before_[i] = w;
      if (insts_[i].fx.unknown_effect)
        w.set();
      w |= insts_[i].fx.vwrites;
    }
  }
  // The same forward may-written analysis over scalar locations.
  std::vector<LocSet> sin(nb), sout(nb);
  auto sblock_writes = [&](uint32_t b, LocSet w) {
    for (size_t i = blocks_[b].first; i < blocks_[b].end; ++i)
      w |= insts_[i].fx.unknown_effect ? all_locs() : insts_[i].fx.writes;
    return w;
  };
  for (bool changed = true; changed;) {
    changed = false;
    for (uint32_t b = 0; b < nb; ++b) {
      LocSet i;
      for (uint32_t p : blocks_[b].predecessors)
        i |= sout[p];
      const LocSet o = sblock_writes(b, i);
      if (i != sin[b] || o != sout[b]) {
        sin[b] = i;
        sout[b] = o;
        changed = true;
      }
    }
  }
  written_before_.assign(insts_.size(), LocSet{});
  for (uint32_t b = 0; b < nb; ++b) {
    LocSet w = sin[b];
    for (size_t i = blocks_[b].first; i < blocks_[b].end; ++i) {
      written_before_[i] = w;
      w |= insts_[i].fx.unknown_effect ? all_locs() : insts_[i].fx.writes;
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

static_assert(PendingState::kNumCounters == amdgpu::kWaitCounterCount);
static_assert(PendingState::kDs == static_cast<unsigned>(amdgpu::WaitCounterKind::Ds));
static_assert(PendingState::kLoad == static_cast<unsigned>(amdgpu::WaitCounterKind::Load));

/// Wait-counter domains an issued memory operation increments, from RocJITsu's memory-issue
/// metadata (SMEM, LDS, VMEM, FLAT on every generation). @p in_order[q] is cleared when its
/// obligation on ordered counter PendingState::kOrdered[q] is outside that counter's
/// in-order class (VMEM for Load, LDS for Ds). A memory-flagged instruction without
/// metadata is assumed to use every counter.
std::array<bool, PendingState::kNumCounters>
producer_counters(const Instruction &inst, rj_code_arch_t arch, std::array<bool, 2> &in_order) {
  std::array<bool, PendingState::kNumCounters> c{};
  in_order.fill(true);
  const auto model = amdgpu::waitcnt_model(arch);
  if (const auto *issue = inst.amdgpu_memory_issue_info(); issue && !model.failed()) {
    for (const auto &ob : issue->counter_obligations()) {
      const auto kind = amdgpu::canonical_wait_counter(ob.wait_counter_type(), model.value());
      const size_t k = amdgpu::WaitCounterPolicy::counter_index(kind);
      if (k >= PendingState::kNumCounters)
        continue;
      c[k] = true;
      const int q = PendingState::ordered_index(static_cast<unsigned>(k));
      const auto ordered_class = k == PendingState::kLoad ? amdgpu::MemoryCompletionClass::VMEM
                                                          : amdgpu::MemoryCompletionClass::LDS;
      if (q >= 0 && ob.completion_class() != ordered_class)
        in_order[static_cast<size_t>(q)] = false;
    }
    return c;
  }
  const uint64_t f = inst.flags();
  // The LDS-crossbar permutes (ds_bpermute, ds_permute, ds_swizzle) carry no metadata: they
  // count on the LDS counter only (lgkmcnt/dscnt), in order with the LDS operations.
  if (std::string_view(inst.mnemonic()).starts_with("ds_") && (f & (VMEM | SMEM)) == 0) {
    c[PendingState::kDs] = true;
    return c;
  }
  if (f & (MEMORY_OP | MEMORY_WAIT_PRODUCER | SMEM | VMEM | LDS))
    c.fill(true);
  return c;
}

} // namespace

PendingState pending_transfer(const InstRef &ref, PendingState s, rj_code_arch_t arch) {
  if (!ref.inst)
    return s; // inline asm: no memory operation
  const Instruction &inst = *ref.inst;
  // A counter wait of any generation (s_waitcnt, s_waitcnt_vscnt, s_wait_loadcnt, ...): each
  // field it names caps that counter. An undecodable wait changes nothing (conservative).
  const auto waits = amdgpu::WaitCounterPolicy::explicit_wait_fields(inst, arch);
  if (!waits.failed() && waits.value()) {
    const amdgpu::WaitFields &fields = *waits.value();
    for (unsigned c = 0; c < PendingState::kNumCounters; ++c) {
      if (!fields[c])
        continue;
      const uint32_t limit = *fields[c];
      s.count[c] = static_cast<uint8_t>(std::min<uint32_t>(s.count[c], limit));
      const int q = PendingState::ordered_index(c);
      if (s.count[c] == 0) {
        s.dests[c].reset();
        s.vdests[c].reset();
        s.unknown[c] = false;
        if (q >= 0) {
          s.ops[static_cast<size_t>(q)].clear();
          s.unordered[static_cast<size_t>(q)] = false;
        }
      } else if (q >= 0 && !s.unordered[static_cast<size_t>(q)] &&
                 s.ops[static_cast<size_t>(q)].size() > limit) {
        // In order: only the newest `limit` operations can still be outstanding.
        auto &ops = s.ops[static_cast<size_t>(q)];
        ops.erase(ops.begin(), ops.end() - static_cast<std::ptrdiff_t>(limit));
        if (!s.unknown[c]) {
          s.dests[c].reset();
          s.vdests[c].reset();
          for (const auto &[l, v] : ops) {
            s.dests[c] |= l;
            s.vdests[c] |= v;
          }
        }
      }
    }
    return s;
  }
  std::array<bool, 2> in_order{};
  const auto counters = producer_counters(inst, arch, in_order);
  for (unsigned c = 0; c < PendingState::kNumCounters; ++c) {
    if (!counters[c])
      continue;
    s.count[c] = static_cast<uint8_t>(std::min<unsigned>(s.count[c] + 1u, 63u));
    s.dests[c] |= ref.fx.writes;
    s.vdests[c] |= ref.fx.vwrites;
    if (const int q = PendingState::ordered_index(c); q >= 0) {
      const auto qi = static_cast<size_t>(q);
      s.unordered[qi] = s.unordered[qi] || !in_order[qi];
      auto &ops = s.ops[qi];
      ops.emplace_back(ref.fx.writes, ref.fx.vwrites);
      if (ops.size() > 63) { // saturate: fold the two oldest
        ops[1].first |= ops[0].first;
        ops[1].second |= ops[0].second;
        ops.erase(ops.begin());
      }
    }
    s.unknown[c] = s.unknown[c] || ref.fx.unknown_effect;
    s.example_pc = ref.pc;
  }
  return s;
}

namespace {

PendingState pending_join(const PendingState &a, const PendingState &b) {
  PendingState r;
  for (unsigned c = 0; c < PendingState::kNumCounters; ++c) {
    r.count[c] = std::max(a.count[c], b.count[c]);
    r.dests[c] = a.dests[c] | b.dests[c];
    r.vdests[c] = a.vdests[c] | b.vdests[c];
    r.unknown[c] = a.unknown[c] || b.unknown[c];
  }
  // Align each in-order queue at its newest end and join element-wise.
  for (size_t q = 0; q < r.ops.size(); ++q) {
    r.unordered[q] = a.unordered[q] || b.unordered[q];
    const auto &qa = a.ops[q], &qb = b.ops[q];
    const size_t n = std::max(qa.size(), qb.size());
    r.ops[q].assign(n, {});
    for (size_t i = 0; i < n; ++i) {
      auto &dst = r.ops[q][n - 1 - i];
      if (i < qa.size()) {
        dst.first |= qa[qa.size() - 1 - i].first;
        dst.second |= qa[qa.size() - 1 - i].second;
      }
      if (i < qb.size()) {
        dst.first |= qb[qb.size() - 1 - i].first;
        dst.second |= qb[qb.size() - 1 - i].second;
      }
    }
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
        t = pending_transfer(insts_[i], t, artifact_.target.arch);
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
      t = pending_transfer(insts_[i], t, artifact_.target.arch);
    }
  }
}

namespace {

/// Reads or writes lanes regardless of EXEC: lane reads/writes by index, DPP, permutes,
/// LDS swizzles. Unknown and opaque instructions count as such.
bool crosses_lanes(const InstRef &r) {
  if (r.inline_asm >= 0)
    return r.fx.unknown_effect;
  if (!r.inst || r.fx.unknown_effect)
    return true;
  const std::string_view m = r.mnemonic;
  auto starts = [&](std::string_view p) { return m.substr(0, p.size()) == p; };
  // MFMA/SMFMAC multiply matrices spread over the wave: every lane's A, B and C feed the
  // result, whatever EXEC holds.
  if (starts("v_readlane") || starts("v_writelane") || starts("v_permlane") ||
      starts("ds_swizzle") || starts("ds_bpermute") || starts("ds_permute") || starts("v_swap") ||
      starts("v_movrel") || starts("v_mfma") || starts("v_smfmac") ||
      m.find("_dpp") != std::string_view::npos || is_lds_transpose(m))
    return true;
  const std::string_view t = r.text;
  return t.find("row_") != std::string_view::npos ||
         t.find("quad_perm") != std::string_view::npos ||
         t.find("dpp8") != std::string_view::npos || t.find("fi:") != std::string_view::npos;
}

/// Writes every bit of each destination VGPR on the active lanes: not a 16-bit result
/// (which may keep the other half), SDWA/DPP/op_sel/true16 forms, d16 loads, mix
/// instructions or lane writes.
bool full_vgpr_write(const InstRef &r) {
  if (!r.inst || r.fx.unknown_effect || r.fx.vwrites.none())
    return false;
  const std::string_view m = r.mnemonic;
  for (std::string_view p : {"_sdwa", "_dpp", "d16", "v_writelane", "_mix", "_b16", "_i16", "_u16"})
    if (m.find(p) != std::string_view::npos)
      return false;
  const std::string_view t = r.text;
  if (t.find("op_sel") != std::string_view::npos || t.find(".h") != std::string_view::npos ||
      t.find(".l") != std::string_view::npos)
    return false;
  switch (fp_result_kind(m, t)) {
  case FpResult::F16Lo:
  case FpResult::F16Hi:
  case FpResult::BF16Lo:
  case FpResult::BF16Hi:
    return false;
  default:
    return true;
  }
}

} // namespace

void KernelModel::compute_exec_stability() {
  const size_t n = insts_.size();
  exec_stable_after_.assign(n, false);
  exec_stable_before_.assign(n, false);
  if (n == 0)
    return;
  LocSet exec;
  exec.set(kLocExecLo);
  exec.set(kLocExecHi);
  // Greatest fixpoint: start from "stable" everywhere and remove what reaches a change.
  std::vector<bool> block_in(blocks_.size(), true);
  bool changed = true;
  for (unsigned round = 0; changed && round < 4096; ++round) {
    changed = false;
    for (size_t bi = blocks_.size(); bi-- > 0;) {
      const BlockInfo &b = blocks_[bi];
      bool stable = !b.unknown_successor;
      for (uint32_t s : b.successors)
        stable = stable && block_in[s];
      for (size_t i = b.end; i-- > b.first;) {
        exec_stable_after_[i] = stable;
        const InstRef &r = insts_[i];
        stable = stable && !(r.fx.writes & exec).any() && !crosses_lanes(r);
        exec_stable_before_[i] = stable;
      }
      if (block_in[bi] != stable) {
        block_in[bi] = stable;
        changed = true;
      }
    }
  }
  // Kills: full-width writes under an EXEC that stays (callable functions keep every lane
  // of their callee-saved VGPRs for the caller).
  const VSet keep = callee_saved_vgprs();
  vkill_.assign(n, VSet{});
  for (size_t i = 0; i < n; ++i)
    if (exec_stable_before_[i] && full_vgpr_write(insts_[i]))
      vkill_[i] = insts_[i].fx.vwrites & ~keep;
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
