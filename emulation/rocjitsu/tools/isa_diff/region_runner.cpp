// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "region_runner.h"

#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"
#include "util/diagnostic.h"

#include "effects.h"
#include "input_policy.h"

#include <algorithm>
#include <exception>
#include <set>
#include <unordered_set>

namespace rocjitsu::isa_diff {
namespace {

constexpr uint32_t kSgprsPerWave = 104;
constexpr uint32_t kVgprFile = 256;   ///< Per-wave VGPR file the CU is configured with.
constexpr uint32_t kMinVgprAlloc = 8; ///< Allocation for a region with no VGPR operand.
constexpr uint32_t kWaveSize = 64;

std::array<uint8_t, 9> pack_counters(const amdgpu::WaitCounters &c) {
  return {c.vmcnt, c.lgkmcnt,   c.expcnt,   c.vscnt,    c.dscnt,
          c.kmcnt, c.tensorcnt, c.asynccnt, c.samplecnt};
}

void unpack_counters(const std::array<uint8_t, 9> &p, amdgpu::WaitCounters &c) {
  c.vmcnt = p[0];
  c.lgkmcnt = p[1];
  c.expcnt = p[2];
  c.vscnt = p[3];
  c.dscnt = p[4];
  c.kmcnt = p[5];
  c.tensorcnt = p[6];
  c.asynccnt = p[7];
  c.samplecnt = p[8];
}

RunStatus map_error(amdgpu::InstructionExecutionError e) {
  using E = amdgpu::InstructionExecutionError;
  switch (e) {
  case E::None:
    return RunStatus::ExecutionError;
  case E::InvalidInstructionEncoding:
  case E::DecodeFailure:
    return RunStatus::DecodeFailure;
  case E::UnsupportedOperandValue:
    return RunStatus::UnsupportedOperand;
  case E::MissingExecutionBackend:
    return RunStatus::MissingExecutionBackend;
  case E::InvalidRegisterAccess:
    return RunStatus::InvalidRegisterAccess;
  case E::UnimplementedInstruction:
    return RunStatus::Unimplemented;
  case E::IllegalInstruction:
    return RunStatus::IllegalInstruction;
  case E::PluginFailure:
    return RunStatus::ExecutionError;
  }
  return RunStatus::ExecutionError;
}

} // namespace

void MemoryImage::begin(uint64_t seed, const std::vector<std::pair<uint64_t, uint32_t>> &preset) {
  if (!started_ || seed != seed_) {
    words_.clear();
    seed_ = seed;
    started_ = true;
  }
  for (const auto &[w, v] : preset)
    words_.try_emplace(w, v);
}

uint32_t MemoryImage::word(uint64_t w, int pointer_half) {
  auto it = words_.find(w);
  if (it != words_.end())
    return it->second;
  uint32_t v = initial_memory_word(seed_, w);
  if (pointer_half >= 0) {
    // Canonical and far above the code: high half in [0x10, 0x7f], low half 64 KiB aligned.
    const uint64_t h = splitmix64(seed_ ^ (w * 0x9e3779b97f4a7c15ull) ^ 0x7074725f);
    v = pointer_half == 0 ? static_cast<uint32_t>(h) & 0xffff0000u
                          : 0x10u + static_cast<uint32_t>((h >> 32) % 0x70u);
  }
  words_.emplace(w, v);
  return v;
}

std::vector<std::pair<uint64_t, uint32_t>> MemoryImage::snapshot() const {
  std::vector<std::pair<uint64_t, uint32_t>> out(words_.begin(), words_.end());
  std::sort(out.begin(), out.end());
  return out;
}

/// Seeds load bytes from the initial memory content before the pipeline reads them, and
/// records the bytes stores write. Observations fire after routing, before the access.
class RegionRunner::MemoryModel final : public ExecutionPlugin {
public:
  MemoryModel(amdgpu::GpuMemory *mem, uint64_t code_lo, uint64_t code_hi)
      : ExecutionPlugin("rj_isa_diff_memory"), mem_(mem), code_lo_(code_lo), code_hi_(code_hi) {}
  bool observes_before_execute_instruction() const override { return false; }
  bool observes_after_execute_instruction() const override { return false; }
  bool observes_async_instruction_issued() const override { return false; }
  bool observes_memory_instruction_routing() const override { return false; }
  bool observes_vgpr_reads() const override { return false; }
  bool observes_vgpr_writes() const override { return false; }
  bool observes_scalar_register_writes() const override { return false; }
  bool observes_sgpr_reads() const override { return false; }
  bool observes_memory_routing() const override { return true; }

  void set_access(MemoryImage *image, uint32_t pointer_lo, uint32_t pointer_hi) {
    image_ = image;
    pointer_lo_ = pointer_lo;
    pointer_hi_ = pointer_hi;
  }

  void onAmdgpuMemoryAccessRouted(const amdgpu::MemoryAccessObservation &a) override {
    if (a.atomic_op != amdgpu::AtomicOp::NONE || !a.secondary_addresses.empty()) {
      unsupported_ = true;
      return;
    }
    const uint64_t bytes = uint64_t{a.element_size_bytes} * a.elements_per_lane;
    for (size_t lane = 0; lane < a.addresses.size() && lane < 64; ++lane) {
      if (!(a.request_lane_mask >> lane & 1u))
        continue;
      const uint64_t base = a.addresses[lane];
      if (bytes && base + (bytes - 1) < base) {
        // The access wraps around the 64-bit address space: hardware faults (the address
        // is far outside any GPUVM range); there are no bytes to seed or record.
        wraps_ = true;
        continue;
      }
      if (a.is_load) {
        for (uint64_t w = base >> 2; w <= (base + bytes - 1) >> 2; ++w) {
          const uint64_t k = w - (base >> 2); // dword of this lane's access
          const int half = k < 32 ? (pointer_lo_ >> k & 1u   ? 0
                                     : pointer_hi_ >> k & 1u ? 1
                                                             : -1)
                                  : -1;
          seed_word(w, half);
        }
      } else {
        for (uint64_t b = base; b < base + bytes; ++b)
          stored_.insert(b);
      }
    }
  }

  /// Undo the previous case (touched bytes back to 0, the unwritten state) and set the
  /// seed of the next one.
  void reset(uint64_t seed) {
    static constexpr uint8_t kZero[4] = {};
    for (uint64_t w : seeded_)
      mem_->write_block(w << 2, std::span<const uint8_t>(kZero, 4));
    for (uint64_t b : stored_)
      mem_->write_block(b, std::span<const uint8_t>(kZero, 1));
    seeded_.clear();
    stored_.clear();
    seed_ = seed;
    unsupported_ = false;
    wraps_ = false;
  }
  [[nodiscard]] uint64_t seed() const { return seed_; }
  [[nodiscard]] bool unsupported() const { return unsupported_; }
  [[nodiscard]] bool wraps() const { return wraps_; }
  [[nodiscard]] std::vector<std::pair<uint64_t, uint8_t>> written() const {
    std::vector<std::pair<uint64_t, uint8_t>> out;
    for (uint64_t b : std::set<uint64_t>(stored_.begin(), stored_.end())) {
      uint8_t v = 0;
      mem_->read_block(b, std::span<uint8_t>(&v, 1));
      out.emplace_back(b, v);
    }
    return out;
  }
  [[nodiscard]] bool touches_code() const {
    for (uint64_t b : stored_)
      if (b >= code_lo_ && b < code_hi_)
        return true;
    return false;
  }

private:
  void seed_word(uint64_t w, int pointer_half) {
    if (!seeded_.insert(w).second)
      return;
    const uint32_t value = image_ ? image_->word(w, pointer_half) : initial_memory_word(seed_, w);
    for (unsigned k = 0; k < 4; ++k) {
      const uint64_t addr = (w << 2) + k;
      if (stored_.count(addr) || (addr >= code_lo_ && addr < code_hi_))
        continue; // a byte this run stored, or the region's own code
      const uint8_t byte = static_cast<uint8_t>(value >> (8 * k));
      mem_->write_block(addr, std::span<const uint8_t>(&byte, 1));
    }
  }

  amdgpu::GpuMemory *mem_;
  MemoryImage *image_ = nullptr;
  uint32_t pointer_lo_ = 0, pointer_hi_ = 0;
  uint64_t code_lo_, code_hi_;
  uint64_t seed_ = 0;
  bool unsupported_ = false;
  bool wraps_ = false;
  std::unordered_set<uint64_t> seeded_; ///< 4-byte words seeded this run.
  std::unordered_set<uint64_t> stored_; ///< Bytes stored this run.
};

const char *run_status_name(RunStatus s) {
  switch (s) {
  case RunStatus::Ok:
    return "OK";
  case RunStatus::DecodeFailure:
    return "DECODE_FAILURE";
  case RunStatus::MissingExecutionBackend:
    return "MISSING_EXECUTION_BACKEND";
  case RunStatus::Unimplemented:
    return "UNIMPLEMENTED";
  case RunStatus::UnsupportedOperand:
    return "UNSUPPORTED_OPERAND";
  case RunStatus::InvalidRegisterAccess:
    return "INVALID_REGISTER_ACCESS";
  case RunStatus::MemoryFault:
    return "MEMORY_FAULT";
  case RunStatus::IllegalInstruction:
    return "ILLEGAL_INSTRUCTION";
  case RunStatus::ExecutionError:
    return "EXECUTION_ERROR";
  case RunStatus::HostException:
    return "HOST_EXCEPTION";
  case RunStatus::UndefinedBehavior:
    return "UNDEFINED_BEHAVIOR";
  case RunStatus::PcWritten:
    return "PC_WRITTEN";
  case RunStatus::WaveStateChanged:
    return "WAVE_STATE_CHANGED";
  case RunStatus::BudgetExceeded:
    return "BUDGET_EXCEEDED";
  }
  return "?";
}

RegionRunner::RegionRunner() = default;

RegionRunner::~RegionRunner() {
  // Release the wave's register blocks before the CU goes away.
  if (wave_ && !wave_->is_halted())
    wave_->halt(amdgpu::Wavefront::CpCompletionNotice::Suppress);
  insts_.clear();
}

std::unique_ptr<RegionRunner> RegionRunner::create(const std::vector<RegionInst> &insts,
                                                   const std::string &label, std::string &error,
                                                   unsigned vgpr_count) {
  std::unique_ptr<RegionRunner> r(new RegionRunner());
  if (vgpr_count > kVgprFile) {
    error = "region needs more than 256 VGPRs";
    return nullptr;
  }
  r->vgpr_count_ = vgpr_count;
  for (const RegionInst &ri : insts) {
    // Classification needs a decode; the memory decision only needs the mnemonic.
    const size_t sp = ri.text.find(' ');
    const std::string mnemonic = sp == std::string::npos ? ri.text : ri.text.substr(0, sp);
    r->is_mem_.push_back(is_region_memory(mnemonic));
    r->has_memory_ = r->has_memory_ || r->is_mem_.back();
  }
  r->memory_ = std::make_unique<amdgpu::GpuMemory>(label + "_mem");
  r->l2_ = std::make_unique<amdgpu::L2Cache>(label + "_l2");
  r->l2_->set_backing_memory(r->memory_.get()); // global accesses reach this memory
  amdgpu::ComputeUnitCore::Config config{};
  config.arch = ROCJITSU_CODE_ARCH_CDNA4;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = kSgprsPerWave;
  config.vgprs_per_wf = kVgprFile;
  config.lds_size_kb = 64;
  // Waits are modelled by the region planner (pending effects); a region replays the same
  // PCs for every case, which the CU's own wait scoreboard would report as hazards.
  config.memory_wait_diagnostics = amdgpu::MemoryWaitDiagnostics::Off;
  util::StringDiagnostic diag;
  r->cu_ = amdgpu::ComputeUnitCore::create(label, config, r->memory_.get(), r->l2_.get(),
                                           simdojo::ExecMode::FUNCTIONAL, diag.emitter());
  if (!r->cu_) {
    error = "cannot create a gfx950 compute unit: " + diag.message();
    return nullptr;
  }
  const uint64_t entry = insts.empty() ? 0 : insts.front().pc;
  if (r->has_memory_) {
    // step() fetches from memory: place the region's words at their PCs, and observe
    // memory accesses to seed loads and record stores.
    uint64_t lo = ~uint64_t{0}, hi = 0;
    for (const RegionInst &ri : insts) {
      r->memory_->write_block(
          ri.pc, std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(ri.words.data()),
                                          ri.words.size() * 4));
      lo = std::min(lo, ri.pc);
      hi = std::max<uint64_t>(hi, ri.pc + ri.words.size() * 4);
    }
    auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
    auto *model = new MemoryModel(r->memory_.get(), lo, hi);
    if (!group->add(OwnedPlugin(model, PluginDeleter{&delete_execution_plugin}))) {
      error = "cannot attach the memory-model plugin";
      return nullptr;
    }
    r->mem_ = model;
    r->cu_->set_plugin_group(group);
    r->plugin_group_ = group;
  }
  r->entry_pc_ = entry;
  r->vgpr_alloc_ = vgpr_count ? vgpr_count : kMinVgprAlloc;
  r->wave_ = r->cu_->dispatch_wf(0, entry, kSgprsPerWave, r->vgpr_alloc_, kWaveSize);
  if (!r->wave_) {
    error = "cannot dispatch a wave64";
    return nullptr;
  }
  r->decoder_ = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  if (!r->decoder_) {
    error = "no gfx950 decoder";
    return nullptr;
  }
  r->asm_.assign(insts.size(), std::nullopt);
  r->asm_block_.assign(insts.size(), 0);
  for (size_t i = 0; i < insts.size(); ++i) {
    const RegionInst &ri = insts[i];
    r->pcs_.push_back(ri.pc);
    r->texts_.push_back(ri.text);
    if (ri.inline_asm >= 0) {
      // Comment-only inline asm: no machine code; run_inline_asm() gives it its meaning.
      r->asm_[i] = parse_inline_asm(ri.asm_text);
      r->asm_block_[i] = static_cast<uint32_t>(ri.inline_asm);
      r->insts_.push_back(nullptr);
      r->model_only_.push_back(false);
      continue;
    }
    util::StringDiagnostic ddiag;
    std::vector<uint32_t> window = ri.words;
    window.resize(window.size() + 3, 0); // decoder may look ahead for literals
    auto decoded = r->decoder_->decode_window(window, ri.pc, ddiag.emitter());
    if (decoded.failed() || static_cast<size_t>(decoded.value()->size()) != ri.words.size() * 4) {
      if (!r->decode_failed_) {
        r->decode_failed_ = true;
        r->decode_failed_index_ = i;
        r->decode_error_ = decoded.failed()
                               ? "decode failed: " + ddiag.message()
                               : "decoded size does not match the recorded instruction words";
      }
      r->insts_.push_back(nullptr);
      r->model_only_.push_back(false);
      continue;
    }
    r->insts_.push_back(std::move(decoded).value());
    r->model_only_.push_back(false);
  }
  // Pointer dataflow (memory model): a scalar load whose destination later serves as a
  // memory base (directly or through SALU moves/adds) gets pointer-shaped content there.
  r->pointer_lo_.assign(insts.size(), 0);
  r->pointer_hi_.assign(insts.size(), 0);
  if (r->has_memory_) {
    std::vector<std::pair<int, int>> origin(kNumSgprs, {-1, -1}); // (load index, dword)
    for (size_t i = 0; i < r->insts_.size(); ++i) {
      const Instruction *inst = r->insts_[i].get();
      if (!inst && r->asm_[i]) {
        // An inline-asm def is an opaque producer, like a scalar load: if its SGPRs later
        // serve as a memory base, it produces pointer-shaped values there.
        const auto &defs = r->asm_[i]->defs;
        for (size_t j = 0; j < defs.size(); ++j)
          if (!defs[j].vgpr && defs[j].index < kNumSgprs)
            origin[defs[j].index] =
                j < 32 ? std::pair<int, int>{static_cast<int>(i), static_cast<int>(j)}
                       : std::pair<int, int>{-1, -1};
        continue;
      }
      if (!inst)
        break;
      const Effects fx = classify(*inst);
      const std::vector<SourceOperand> src = source_operands(*inst);
      std::vector<unsigned> writes;
      for (unsigned l = 0; l < kNumSgprs; ++l)
        if (fx.writes.test(l))
          writes.push_back(l);
      if (r->is_mem_[i]) {
        for (const SourceOperand &so : src) {
          if (so.locs.count() != 2)
            continue;
          unsigned lo = kNumSgprs;
          for (unsigned l = 0; l < kNumSgprs; ++l)
            if (so.locs.test(l)) {
              lo = l;
              break;
            }
          for (unsigned p : {lo, lo + 1})
            if (p < kNumSgprs && origin[p].first >= 0) {
              const auto [li, k] = origin[p];
              (p == lo ? r->pointer_lo_ : r->pointer_hi_)[static_cast<size_t>(li)] |= 1u << k;
            }
          break; // the first SGPR pair is the base (sbase / saddr)
        }
        const bool scalar_load =
            fx.mem_read && std::string_view(inst->mnemonic()).substr(0, 6) == "s_load";
        for (size_t idx = 0; idx < writes.size(); ++idx)
          origin[writes[idx]] = scalar_load && idx < 32 ? std::pair<int, int>{static_cast<int>(i),
                                                                              static_cast<int>(idx)}
                                                        : std::pair<int, int>{-1, -1};
        continue;
      }
      const SourceOperand *first = nullptr;
      for (const SourceOperand &so : src)
        if (so.locs.any()) {
          first = &so;
          break;
        }
      unsigned first_lo = kNumSgprs;
      if (first)
        for (unsigned l = 0; l < kNumSgprs; ++l)
          if (first->locs.test(l)) {
            first_lo = l;
            break;
          }
      for (size_t idx = 0; idx < writes.size(); ++idx) {
        const unsigned from = first_lo + static_cast<unsigned>(idx);
        origin[writes[idx]] = first && from < kNumSgprs && first->locs.test(from)
                                  ? origin[from]
                                  : std::pair<int, int>{-1, -1};
      }
    }
  }
  // Fix the frame to the post-dispatch values.
  r->baseline_ = r->capture().frame;
  return r;
}

void RegionRunner::simulate_model_only_for_test(size_t index) {
  if (index < model_only_.size())
    model_only_[index] = true;
}

void RegionRunner::load(const MachineState &s) {
  if (wave_->state() != amdgpu::WfState::RUNNING) // a memory fault halted it: re-dispatch
    wave_ = cu_->dispatch_wf(0, entry_pc_, kSgprsPerWave, vgpr_alloc_, kWaveSize);
  if (has_memory_)
    image_->begin(s.mem_seed, s.mem_init);
  amdgpu::Wavefront &wf = *wave_;
  const uint32_t base = wf.sgpr_alloc().base;
  for (unsigned i = 0; i < kNumSgprs; ++i)
    cu_->write_sgpr(base + i, s.loc[i]);
  cu_->write_sgpr(base + 102, s.frame.s102);
  cu_->write_sgpr(base + 103, s.frame.s103);
  wf.set_vcc_raw(uint64_t{s.loc[kLocVccLo]} | (uint64_t{s.loc[kLocVccHi]} << 32));
  wf.set_exec_raw(uint64_t{s.loc[kLocExecLo]} | (uint64_t{s.loc[kLocExecHi]} << 32));
  wf.set_m0(s.loc[kLocM0]);
  wf.set_status_raw((s.frame.status_except_scc & ~1u) | (s.loc[kLocScc] & 1u));
  wf.set_mode_raw(s.frame.mode);
  wf.set_trapsts(s.frame.trapsts);
  for (uint32_t i = 0; i < 16; ++i)
    wf.set_ttmp(i, s.frame.ttmp[i]);
  wf.set_scratch_base(s.frame.scratch_base);
  unpack_counters(s.frame.wait_counters, wf.wait_counters());
  wf.set_in_trap_handler(s.frame.in_trap_handler);
  if (mem_) {
    // Write back the previous case's stores, undo its memory, and drop every cached line
    // (L1 scalar/vector caches and the write-back L2), so this case reads its own input.
    cu_->flush_all();
    mem_->reset(s.mem_seed);
    l2_->invalidate_all();
    cu_->flush_l1();
  }
  const uint32_t vbase = wf.vgpr_alloc().base;
  for (unsigned r = 0; r < vgpr_count_ && r < s.vgpr_count; ++r)
    for (unsigned lane = 0; lane < kWaveLanes; ++lane)
      cu_->write_vgpr(vbase + r, lane, s.vget(r, lane));
  wf.begin_instruction_execution();
  wf.pc = pcs_.empty() ? 0 : pcs_.front();
  asm_seed_ = s.asm_seed;
  asm_uses_.clear();
}

MachineState RegionRunner::capture() const {
  const amdgpu::Wavefront &wf = *wave_;
  MachineState s;
  const uint32_t base = wf.sgpr_alloc().base;
  for (unsigned i = 0; i < kNumSgprs; ++i)
    s.loc[i] = cu_->read_sgpr_storage(base + i);
  s.frame.s102 = cu_->read_sgpr_storage(base + 102);
  s.frame.s103 = cu_->read_sgpr_storage(base + 103);
  const uint64_t vcc = wf.vcc();
  const uint64_t exec = wf.exec_raw();
  s.loc[kLocVccLo] = static_cast<uint32_t>(vcc);
  s.loc[kLocVccHi] = static_cast<uint32_t>(vcc >> 32);
  s.loc[kLocExecLo] = static_cast<uint32_t>(exec);
  s.loc[kLocExecHi] = static_cast<uint32_t>(exec >> 32);
  s.loc[kLocM0] = wf.m0();
  s.loc[kLocScc] = wf.read_scc() ? 1u : 0u;
  s.frame.status_except_scc = wf.status_raw() & ~1u;
  s.frame.mode = wf.mode_raw();
  s.frame.trapsts = wf.trapsts();
  for (uint32_t i = 0; i < 16; ++i)
    s.frame.ttmp[i] = wf.ttmp(i);
  s.frame.scratch_base = wf.scratch_base();
  s.frame.wait_counters = pack_counters(wf.wait_counters());
  s.frame.in_trap_handler = wf.in_trap_handler();
  s.frame.wave_state = static_cast<uint8_t>(wf.state());
  s.frame.sleep_cycles = wf.sleep_cycles();
  s.resize_vgprs(vgpr_count_);
  const uint32_t vbase = wf.vgpr_alloc().base;
  for (unsigned r = 0; r < vgpr_count_; ++r)
    for (unsigned lane = 0; lane < kWaveLanes; ++lane)
      s.vset(r, lane, cu_->read_vgpr_storage(vbase + r, lane));
  if (mem_) {
    cu_->flush_all(); // stores may still sit in the write-back L2
    s.mem_seed = mem_->seed();
    s.mem_written = mem_->written();
    s.mem_init = image_->snapshot();
  }
  s.asm_seed = asm_seed_;
  s.asm_uses = asm_uses_;
  return s;
}

uint32_t inline_asm_word(uint64_t seed, uint32_t block, uint32_t slot, uint32_t lane) {
  return initial_memory_word(seed ^ 0x61736d0000000000ull,
                             (uint64_t{block} << 40) | (uint64_t{slot} << 8) | lane);
}

RunOutcome RegionRunner::run_inline_asm(size_t i) {
  RunOutcome out;
  const InlineAsmEffect &e = *asm_[i];
  const uint32_t block = asm_block_[i];
  amdgpu::Wavefront &wf = *wave_;
  const uint32_t sbase = wf.sgpr_alloc().base;
  const uint32_t vbase = wf.vgpr_alloc().base;
  if (e.opaque || e.untracked) {
    out.status = RunStatus::UnsupportedOperand;
    out.failed_index = i;
    out.detail = e.reason;
    return out;
  }
  for (const auto *slots : {&e.uses, &e.defs})
    for (const AsmSlot &s : *slots)
      if (s.vgpr && s.index >= vgpr_count_) {
        out.status = RunStatus::InvalidRegisterAccess;
        out.failed_index = i;
        out.detail = "inline asm names v" + std::to_string(s.index) + " beyond the tracked VGPRs";
        return out;
      }
  auto read_loc = [&](unsigned l) -> uint32_t {
    if (l < kNumSgprs)
      return cu_->read_sgpr_storage(sbase + l);
    switch (l) {
    case kLocVccLo:
      return static_cast<uint32_t>(wf.vcc());
    case kLocVccHi:
      return static_cast<uint32_t>(wf.vcc() >> 32);
    case kLocExecLo:
      return static_cast<uint32_t>(wf.exec_raw());
    case kLocExecHi:
      return static_cast<uint32_t>(wf.exec_raw() >> 32);
    case kLocM0:
      return wf.m0();
    default:
      return wf.read_scc() ? 1u : 0u;
    }
  };
  auto write_loc = [&](unsigned l, uint32_t v) {
    if (l < kNumSgprs) {
      cu_->write_sgpr(sbase + l, v);
      return;
    }
    switch (l) {
    case kLocVccLo:
      wf.set_vcc_raw((wf.vcc() & ~uint64_t{0xffffffffu}) | v);
      break;
    case kLocVccHi:
      wf.set_vcc_raw((wf.vcc() & uint64_t{0xffffffffu}) | (uint64_t{v} << 32));
      break;
    case kLocExecLo:
      wf.set_exec_raw((wf.exec_raw() & ~uint64_t{0xffffffffu}) | v);
      break;
    case kLocExecHi:
      wf.set_exec_raw((wf.exec_raw() & uint64_t{0xffffffffu}) | (uint64_t{v} << 32));
      break;
    case kLocM0:
      wf.set_m0(v);
      break;
    default:
      wf.set_status_raw((wf.status_raw() & ~1u) | (v & 1u));
      break;
    }
  };
  if (!e.uses.empty()) {
    AsmUse u;
    u.block = block;
    const uint64_t exec = wf.exec_raw();
    bool vector = false;
    for (const AsmSlot &s : e.uses) {
      if (!s.vgpr) {
        u.values.push_back(read_loc(s.index));
        continue;
      }
      vector = true;
      for (unsigned lane = 0; lane < kWaveLanes; ++lane)
        u.values.push_back((exec >> lane) & 1u ? cu_->read_vgpr_storage(vbase + s.index, lane)
                                               : 0u);
    }
    if (vector) {
      u.values.push_back(static_cast<uint32_t>(exec));
      u.values.push_back(static_cast<uint32_t>(exec >> 32));
    }
    asm_uses_.push_back(std::move(u));
  }
  for (size_t j = 0; j < e.defs.size(); ++j) {
    const AsmSlot &s = e.defs[j];
    const auto slot = static_cast<uint32_t>(j);
    if (s.vgpr) {
      // Per-thread semantics: the lanes in EXEC get the value, the others keep theirs (as
      // for a VALU def; LLVM may keep another thread's value in an inactive lane).
      const uint64_t exec = wf.exec_raw();
      for (unsigned lane = 0; lane < kWaveLanes; ++lane)
        if ((exec >> lane) & 1u)
          cu_->write_vgpr(vbase + s.index, lane, inline_asm_word(asm_seed_, block, slot, lane));
    } else {
      uint32_t v = inline_asm_word(asm_seed_, block, slot, 0);
      // Pointer-shaped where the region uses it as a memory base (see MemoryImage::word).
      const uint64_t h = splitmix64(asm_seed_ ^ (uint64_t{block} << 32) ^ slot ^ 0x7074725f);
      if (j < 32 && ((pointer_lo_[i] >> j) & 1u))
        v = static_cast<uint32_t>(h) & 0xffff0000u;
      else if (j < 32 && ((pointer_hi_[i] >> j) & 1u))
        v = 0x10u + static_cast<uint32_t>((h >> 32) % 0x70u);
      write_loc(s.index, v & loc_value_mask(s.index));
    }
  }
  return out;
}

RunOutcome RegionRunner::run(size_t budget, std::vector<TraceStep> *trace) {
  RunOutcome out;
  amdgpu::Wavefront &wf = *wave_;
  MachineState prev;
  if (trace)
    prev = capture();
  for (size_t i = 0; i < insts_.size(); ++i) {
    if (out.executed >= budget) {
      out.status = RunStatus::BudgetExceeded;
      out.failed_index = i;
      return out;
    }
    if (decode_failed_ && i >= decode_failed_index_) {
      out.status = RunStatus::DecodeFailure;
      out.failed_index = decode_failed_index_;
      out.detail = decode_error_;
      return out;
    }
    if (asm_[i]) {
      RunOutcome a = run_inline_asm(i);
      if (a.status != RunStatus::Ok) {
        a.executed = out.executed;
        return a;
      }
      ++out.executed;
      ++total_executed_;
      if (trace) {
        MachineState now = capture();
        TraceStep step;
        step.index = i;
        step.pc = pcs_[i];
        step.text = texts_[i];
        step.changed = differing_locs(prev, now);
        step.vchanged = differing_vgprs(prev, now);
        step.frame_changed = differing_frame_fields(prev.frame, now.frame);
        step.after = now;
        trace->push_back(std::move(step));
        prev = std::move(now);
      }
      continue;
    }
    Instruction *inst = insts_[i].get();
    if (is_mem_[i] && inst && inst->execute && !model_only_[i]) {
      // The full issue path: fetch at the PC, execute, route, complete (functional).
      wf.pc = pcs_[i];
      if (mem_) {
        mem_->set_access(image_.get(), pointer_lo_[i], pointer_hi_[i]);
        // Seeding writes backing memory when the access is observed (before the pipeline
        // reads it). Write back and drop every cached line first, so the access fills
        // its lines from the seeded memory instead of an earlier, partially seeded fill.
        cu_->flush_all();
        l2_->invalidate_all();
        cu_->flush_l1();
      }
      const uint64_t ub_before = cu_->isa_diagnostic_count();
      try {
        (void)cu_->step();
      } catch (const std::exception &e) {
        out.status = RunStatus::HostException;
        out.failed_index = i;
        out.detail = e.what();
        return out;
      }
      ++out.executed;
      ++total_executed_;
      if (wf.instruction_execution_failed()) {
        out.status = map_error(wf.instruction_execution_error());
        out.failed_index = i;
        out.detail = "memory instruction failed";
        return out;
      }
      if (mem_ && (mem_->unsupported() || mem_->touches_code())) {
        out.status = RunStatus::UnsupportedOperand;
        out.failed_index = i;
        out.detail = mem_->unsupported() ? "atomic or dual-address memory access"
                                         : "store into the region's own code";
        return out;
      }
      if (cu_->isa_diagnostic_count() != ub_before) {
        out.status = RunStatus::UndefinedBehavior;
        out.failed_index = i;
        out.detail = "RocJITsu reported ISA undefined behaviour (see isa-undefined log)";
        return out;
      }
      if (mem_ && mem_->wraps()) {
        out.status = RunStatus::MemoryFault;
        out.failed_index = i;
        out.detail = "memory access wraps around the 64-bit address space at '" + texts_[i] + "'";
        return out;
      }
      if (wf.state() == amdgpu::WfState::HALTED || wf.state() == amdgpu::WfState::MEMORY_TRAP) {
        out.status = RunStatus::MemoryFault;
        out.failed_index = i;
        out.detail = "memory access fault at '" + texts_[i] + "'";
        return out;
      }
      if (wf.state() != amdgpu::WfState::RUNNING) {
        out.status = RunStatus::WaveStateChanged;
        out.failed_index = i;
        out.detail = "the wave stopped running at a memory instruction (state " +
                     std::to_string(static_cast<unsigned>(wf.state())) + ")";
        return out;
      }
      if (wf.pc != pcs_[i] + inst->size()) {
        out.status = RunStatus::PcWritten;
        out.failed_index = i;
        return out;
      }
      if (trace) {
        MachineState now = capture();
        TraceStep step;
        step.index = i;
        step.pc = pcs_[i];
        step.text = texts_[i];
        step.changed = differing_locs(prev, now);
        step.vchanged = differing_vgprs(prev, now);
        step.frame_changed = differing_frame_fields(prev.frame, now.frame);
        step.after = now;
        trace->push_back(std::move(step));
        prev = std::move(now);
      }
      continue;
    }
    if (!inst->execute || model_only_[i]) {
      out.status = RunStatus::MissingExecutionBackend;
      out.failed_index = i;
      out.detail = "instruction has no execution callback";
      return out;
    }
    wf.pc = pcs_[i];
    const uint64_t ub_before = cu_->isa_diagnostic_count();
    std::string detail;
    util::Result result = util::Result::failure();
    try {
      result = cu_->execute_instruction(inst, wf, &detail);
    } catch (const std::exception &e) {
      out.status = RunStatus::HostException;
      out.failed_index = i;
      out.detail = e.what();
      return out;
    } catch (...) {
      out.status = RunStatus::HostException;
      out.failed_index = i;
      out.detail = "unknown exception";
      return out;
    }
    ++out.executed;
    ++total_executed_;
    if (result.failed()) {
      out.status = map_error(wf.instruction_execution_error());
      out.failed_index = i;
      out.detail = detail;
      return out;
    }
    if (cu_->isa_diagnostic_count() != ub_before) {
      out.status = RunStatus::UndefinedBehavior;
      out.failed_index = i;
      out.detail = "RocJITsu reported ISA undefined behaviour (see isa-undefined log)";
      return out;
    }
    if (wf.pc != pcs_[i]) {
      out.status = RunStatus::PcWritten;
      out.failed_index = i;
      return out;
    }
    if (wf.state() != amdgpu::WfState::RUNNING) {
      out.status = RunStatus::WaveStateChanged;
      out.failed_index = i;
      return out;
    }
    if (trace) {
      MachineState now = capture();
      TraceStep step;
      step.index = i;
      step.pc = pcs_[i];
      step.text = texts_[i];
      step.changed = differing_locs(prev, now);
      step.vchanged = differing_vgprs(prev, now);
      step.frame_changed = differing_frame_fields(prev.frame, now.frame);
      step.after = now;
      trace->push_back(std::move(step));
      prev = std::move(now);
    }
  }
  return out;
}

} // namespace rocjitsu::isa_diff
