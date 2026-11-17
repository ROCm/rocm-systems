// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file region_runner.h
/// @brief Execute only a region's instructions through RocJITsu's instruction execution.
///
/// @details Each runner owns a private GpuMemory, L2Cache, ComputeUnitCore and one
/// dispatched wave64, so an old and a new runner share no mutable state. The region is
/// decoded once, from its recorded words, when the runner is created. Each case then:
///   1. load(): writes every tracked location and every frame field, and clears the
///      per-instruction error state;
///   2. run(): sets the PC to the recorded address of each instruction and calls
///      ComputeUnitCore::execute_instruction, the same entry point the CU's own issue
///      loop uses, so no semantics are re-implemented here;
///   3. capture(): reads the state back.
/// A failure is classified and never treated as a no-op or a success.
#pragma once

#include "effects.h"
#include "machine_state.h"
#include "region.h"

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocjitsu {
class Decoder;
class Instruction;
namespace amdgpu {
class ComputeUnitCore;
class GpuMemory;
class L2Cache;
class Wavefront;
} // namespace amdgpu
} // namespace rocjitsu

namespace rocjitsu::isa_diff {

enum class RunStatus {
  Ok,
  DecodeFailure,
  MissingExecutionBackend,
  Unimplemented,
  UnsupportedOperand,
  InvalidRegisterAccess,
  MemoryFault, ///< A memory access faulted (the wave halted or trapped).
  IllegalInstruction,
  ExecutionError,
  HostException,
  UndefinedBehavior,
  PcWritten,
  WaveStateChanged,
  BudgetExceeded,
  /// A loop inside the region ran past the step limit: the case is excluded (never read as
  /// non-termination, never a pass).
  StepLimit,
};
[[nodiscard]] const char *run_status_name(RunStatus s);

struct RunOutcome {
  RunStatus status = RunStatus::Ok;
  size_t executed = 0;     ///< Instructions executed this run.
  uint64_t exit_pc = 0;    ///< 0: fell off the end; else the PC a branch left the region to.
  size_t failed_index = 0; ///< Region index of the failing instruction.
  std::string detail;
};

struct TraceStep {
  size_t index = 0;
  uint64_t pc = 0;
  std::string text;
  LocSet changed;
  VSet vchanged;
  std::vector<std::string> frame_changed;
  MachineState after;
};

/// Initial memory content of one case, shared by the old and new runners so both read
/// the same input. Words are generated on first read (data or pointer-shaped) and kept.
class MemoryImage {
public:
  /// Start a case. The content is kept while the seed is unchanged (re-executions of the
  /// same case: reduction, traces), otherwise cleared; preset words are added if absent.
  void begin(uint64_t seed, const std::vector<std::pair<uint64_t, uint32_t>> &preset);
  /// Initial value of word @p w; @p pointer_half is -1 for data, 0/1 for the low/high half
  /// of a pointer (canonical, away from the region's code). @p scalar: a scalar load reads
  /// it (kernel arguments, constants).
  uint32_t word(uint64_t w, int pointer_half, bool scalar = false);
  /// Every word read so far (word index, value), sorted. Words are only added, so the
  /// sorted list is kept and the words added since are merged in: a state capture costs a
  /// copy, not a sort of the case's whole input (a GEMM case reads tens of thousands).
  [[nodiscard]] std::vector<std::pair<uint64_t, uint32_t>> snapshot() const;
  /// Regions with a loop: in three cases of four, data words that scalar loads read are
  /// small (below 2^k, k up to 12), so loop bounds and sizes read from kernel arguments
  /// let loops end; the other cases keep the full catalog and random values.
  void set_small_scalars(bool on) { small_scalars_ = on; }
  /// The current case draws small scalar data words.
  [[nodiscard]] bool small_case() const;

private:
  uint64_t seed_ = 0;
  bool started_ = false;
  bool small_scalars_ = false;
  std::unordered_map<uint64_t, uint32_t> words_;
  void added(uint64_t w, uint32_t v) { added_.emplace_back(w, v); }
  mutable std::vector<std::pair<uint64_t, uint32_t>> sorted_; ///< words_ as of the last snapshot
  mutable std::vector<std::pair<uint64_t, uint32_t>> added_;  ///< words added since
};

class RegionRunner {
public:
  /// @brief Build a runner. Decode failures do not fail creation; run() reports them.
  /// @param vgpr_count VGPRs v0..v{n-1} the wave allocates, loads and captures; any
  /// access beyond them is an invalid register access. 0: no VGPR state (SALU region).
  /// @param target GPU and wave size: the compute unit's ISA, the decoder, the SGPRs the
  /// wave allocates (s0-s103 on GFX9, s0-s105 from GFX10) and the lanes it runs.
  static std::unique_ptr<RegionRunner> create(const std::vector<RegionInst> &insts,
                                              const std::string &label, std::string &error,
                                              unsigned vgpr_count = 0, const Target &target = {});
  ~RegionRunner();
  RegionRunner(const RegionRunner &) = delete;
  RegionRunner &operator=(const RegionRunner &) = delete;

  /// Frame values right after dispatch; contracts fix the frame to these.
  [[nodiscard]] const Frame &baseline_frame() const { return baseline_; }

  void load(const MachineState &state);
  /// Share the initial memory image with the other side's runner (one per case).
  void set_memory_image(std::shared_ptr<MemoryImage> image) { image_ = std::move(image); }
  /// Stored bytes captured as dead at the exit (RegionContract::memory_dead_at_exit).
  void set_memory_dead_at_exit(std::string kind) { dead_kind_ = std::move(kind); }
  /// LDS window of the wave (one workgroup's allocation at offset 0).
  static constexpr uint64_t kLdsWindow = 64 * 1024;
  /// Scratch base (FLAT_SCRATCH): canonical, far above the code and pointer inputs.
  static constexpr uint64_t kScratchBase = 0x0000'0080'0000'0000ull;
  RunOutcome run(size_t budget, std::vector<TraceStep> *trace = nullptr);
  /// Multiplier of the loop step limit (1 by default). A case in which one side exits and
  /// the other runs past the limit re-runs the looping side with a larger limit, kept for
  /// the rest of the case (undefined-value refills, reduction, trace).
  void set_loop_scale(size_t scale) { loop_scale_ = std::max<size_t>(scale, 1); }
  [[nodiscard]] size_t loop_scale() const { return loop_scale_; }
  /// Base loop step limit for @p budget instructions: 64 steps per instruction, at most
  /// max(budget, 65536); times loop_scale().
  [[nodiscard]] size_t step_limit(size_t budget) const;
  /// The region has a backward branch (a loop).
  [[nodiscard]] bool loops() const { return back_edge_; }
  /// [lo, hi) of the region's code in the runner's memory (empty without memory accesses).
  [[nodiscard]] std::pair<uint64_t, uint64_t> code_range() const { return {code_lo_, code_hi_}; }
  /// A load from [lo, hi) reads code and makes the case UNSUPPORTED_OPERAND: give both
  /// runners of a pair the union of their code ranges (share_code_guard), so both exclude
  /// the same inputs (each side's code bytes differ).
  void guard_code(uint64_t lo, uint64_t hi);
  [[nodiscard]] MachineState capture() const;

  [[nodiscard]] size_t size() const { return pcs_.size(); }
  [[nodiscard]] uint64_t total_executed() const { return total_executed_; }
  /// Test seam: drop an instruction's execution callback (simulates a model-only provider).
  void simulate_model_only_for_test(size_t index);

  /// Poison filter (PoisonKind mask): instructions are checked for the machine analogues of
  /// IR poison conditions before (inputs) and after (floating-point results) they execute.
  void set_poison_filter(uint32_t kinds) { poison_filter_ = kinds; }
  /// NaN choice: LLVM leaves the bits of a NaN that a floating-point operation produces
  /// unspecified. Alternative @p alt (1 canonical quiet NaN, 2 all ones, 3 random quiet
  /// NaN from @p seed) replaces every NaN a floating-point instruction produces; 0 keeps
  /// the hardware's. Outputs that change with the choice depend on it (don't care).
  /// A random NaN is drawn from (seed, the NaN the hardware produced, destination
  /// component, lane): two programs that run the same floating-point operation on the same
  /// inputs draw the same NaN, whatever order they run the operations in and whatever
  /// hints, moves or register names lie between them.
  void set_nan_choice(unsigned alt, uint64_t seed) {
    nan_choice_ = alt;
    nan_seed_ = seed;
  }
  /// The last run produced a NaN from a floating-point instruction (in some active lane).
  [[nodiscard]] bool nan_produced() const { return nan_produced_; }
  /// The first poison condition the last run met, or empty.
  [[nodiscard]] const std::string &poison_hit() const { return poison_hit_; }

private:
  RegionRunner();

  std::unique_ptr<amdgpu::GpuMemory> memory_;
  std::unique_ptr<amdgpu::L2Cache> l2_;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu_;
  amdgpu::Wavefront *wave_ = nullptr;
  std::unique_ptr<Decoder> decoder_;
  std::vector<std::unique_ptr<Instruction>> insts_;
  std::vector<bool> model_only_;
  std::vector<uint64_t> pcs_;
  std::vector<std::string> texts_;
  bool decode_failed_ = false;
  size_t decode_failed_index_ = 0;
  std::string decode_error_;
  Frame baseline_;
  unsigned vgpr_count_ = 0;
  Target target_;
  std::vector<bool> branch_; ///< Direct branch (followed by the runner).
  /// s_barrier: one wave runs, so nothing to wait for; it starts the next barrier epoch.
  std::vector<bool> barrier_;
  /// s_endpgm inside the region (an early exit; the planner admits it only when the region
  /// reaches the final s_endpgm): the wave ends, which is the region's end exit.
  std::vector<bool> endpgm_;
  uint32_t barriers_ = 0;                         ///< Barriers the last run executed.
  uint64_t code_lo_ = 0, code_hi_ = 0;            ///< The region's code in memory_.
  std::unordered_map<uint64_t, size_t> pc_index_; ///< Region instruction at each PC.
  size_t loop_scale_ = 1;
  bool back_edge_ = false; ///< Some branch goes backwards (a loop).
  uint64_t exit_pc_ = 0;   ///< The last run's exit (see RunOutcome).
  /// Lanes of the wave: all 64 on wave64, the low 32 on wave32.
  uint64_t lane_mask_ = ~uint64_t{0};
  /// EXEC restricted to the wave's lanes.
  [[nodiscard]] uint64_t exec_lanes() const;
  /// Memory model: memory instructions are issued through ComputeUnitCore::step() (the
  /// full fetch/execute/route path, synchronous in functional mode); a plugin seeds the
  /// bytes a load reads first from the initial content and records stored bytes.
  class MemoryModel;
  bool has_memory_ = false;
  std::vector<bool> is_mem_;
  MemoryModel *mem_ = nullptr;
  std::string dead_kind_;
  std::vector<bool> frame_access_; ///< Per instruction: a stack-frame private access.
  void configure_wave();
  std::shared_ptr<void> plugin_group_;
  std::shared_ptr<MemoryImage> image_ = std::make_shared<MemoryImage>();
  /// Per instruction: dwords of a scalar load that later serve as a memory base
  /// (bit k: dword k is a pointer's low half / high half).
  std::vector<uint32_t> pointer_lo_, pointer_hi_;
  uint64_t entry_pc_ = 0;
  uint32_t vgpr_alloc_ = 0;
  uint64_t total_executed_ = 0;
  /// Inline-asm blocks by region index (empty for machine instructions), and per case the
  /// seed of their def values and what their uses observed.
  std::vector<std::optional<InlineAsmEffect>> asm_;
  std::vector<uint32_t> asm_block_;
  uint64_t asm_seed_ = 0;
  std::vector<AsmUse> asm_uses_;
  /// A source operand: VGPR/SGPR registers (first register and count) or a constant.
  struct OperandRef {
    enum Kind : uint8_t { Constant, Vgpr, Scalar, Other } kind = Other;
    unsigned reg = 0;   ///< First VGPR, or lowest Loc for scalars.
    unsigned count = 0; ///< Registers (2 for a 64-bit operand).
    uint32_t value = 0; ///< Constants.
  };
  /// Floating-point provenance plan of one instruction (see FpResult).
  struct FpPlan {
    FpResult kind = FpResult::None;
    std::vector<unsigned> dsts;            ///< Destination VGPRs, ascending.
    int mask_loc = -1;                     ///< v_cndmask: lowest Loc of the lane-select mask.
    unsigned op_sel_lo = 0, op_sel_hi = 0; ///< v_pk_mov_b32 source dword selection.
    bool store = false;                    ///< A vector store: data tags go to the memory model.
    FpResult skind = FpResult::None;       ///< Scalar results (fp_scalar_result_kind).
    std::vector<unsigned> sdsts;           ///< Scalar destinations (not SCC), ascending.
    bool store_hi16 = false;               ///< `_d16_hi` store: the high half is stored.
  };
  std::vector<std::vector<OperandRef>> srcs_; ///< Non-fieldless sources per instruction.
  std::vector<FpPlan> fp_plan_;
  std::vector<uint32_t> poison_kind_;
  std::vector<std::string> mnemonics_;
  std::vector<uint8_t> vfp_;            ///< FpTag per VGPR lane during a run.
  std::array<uint8_t, kNumLocs> sfp_{}; ///< FpTag per scalar location during a run.
  std::vector<uint8_t> scratch_stags_;
  /// Tags of instruction @p i's scalar results (computed before it runs).
  [[nodiscard]] std::vector<uint8_t> plan_stags(size_t i) const;
  void apply_stags(size_t i, const std::vector<uint8_t> &tags);
  /// Write a scalar location (SGPR, VCC/EXEC half, M0).
  void write_loc(unsigned l, uint32_t v);
  uint32_t poison_filter_ = 0;
  unsigned nan_choice_ = 0;
  uint64_t nan_seed_ = 0;
  bool nan_produced_ = false;
  /// Note NaN results of instruction @p i (lanes in @p exec) and apply the NaN choice.
  void choose_nans(size_t i, uint64_t exec);
  std::string poison_hit_;
  [[nodiscard]] uint32_t read_loc(unsigned l) const;
  [[nodiscard]] uint32_t read_vgpr(unsigned r, unsigned lane) const;
  /// Value of a source operand in @p lane (64-bit when @p wide and the operand has two
  /// registers); nullopt when it cannot be read (64-bit constants, other state).
  [[nodiscard]] std::optional<uint64_t> operand_value(const OperandRef &op, unsigned lane,
                                                      bool wide) const;
  uint8_t &tag(unsigned r, unsigned lane) { return vfp_[size_t{r} * kWaveLanes + lane]; }
  /// The tags instruction @p i will give its destinations (moves read their sources, so
  /// this runs before execution), applied to the lanes in @p exec after it.
  void plan_tags(size_t i, std::vector<std::array<uint8_t, kWaveLanes>> &out) const;
  void apply_tags(size_t i, uint64_t exec,
                  const std::vector<std::array<uint8_t, kWaveLanes>> &tags);
  std::vector<std::array<uint8_t, kWaveLanes>> scratch_tags_;
  /// First poison condition instruction @p i meets on its inputs (before) or its
  /// floating-point results (after), or empty.
  [[nodiscard]] std::string poison_before(size_t i) const;
  [[nodiscard]] std::string poison_after(size_t i, uint64_t exec) const;
  /// Run inline-asm block @p i: record its uses, then write its defs.
  RunOutcome run_inline_asm(size_t i);
};

/// Value a matched inline-asm `def` writes: piece @p slot of block @p block, lane @p lane
/// (0 for scalars), for the case's inline-asm seed. Half catalog values, half random.
[[nodiscard]] uint32_t inline_asm_word(uint64_t seed, uint32_t block, uint32_t slot, uint32_t lane);

/// Both runners of a pair treat loads from either side's code as reading code.
void share_code_guard(RegionRunner &a, RegionRunner &b);

} // namespace rocjitsu::isa_diff
