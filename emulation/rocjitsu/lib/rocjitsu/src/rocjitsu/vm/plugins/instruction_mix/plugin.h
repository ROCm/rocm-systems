// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "rocjitsu/vm/plugins/instruction_family.h"
#include "rocjitsu/vm/plugins/kernel_dispatch_info.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace rocjitsu::plugins::instruction_mix {

/// The instruction-mix report shares its instruction families with the
/// throughput report so the two are joinable; see
/// plugins/instruction_family.h.
using InstructionFamily = plugins::InstructionFamily;

inline constexpr size_t kInstructionFamilyCount = plugins::kInstructionFamilyCount;

/// One executed mnemonic. A mnemonic reachable through more than one encoding
/// is still one entry, because the coverage denominators (the ISA XML and the
/// generated sources) enumerate mnemonics rather than encodings.
///
/// The encoding fields then have to pick one of those encodings, and the pick
/// must not depend on which wavefront happened to finish first: the report is
/// meant to be diffable between runs. `merge()` keeps the entry that is
/// smallest under `sighting_precedes()`, which orders on every field the record
/// emits and is therefore a total order over distinguishable sightings.
struct MnemonicStats {
  uint64_t executions = 0;
  uint16_t encoding_id = 0;
  uint16_t opcode = 0;
  uint8_t encoding_bytes = 0;
  InstructionFamily family = InstructionFamily::Other;
  uint32_t first_dispatch_id = 0;
};

/// Transparent hash so the hot path can look a mnemonic up by
/// `std::string_view` without materialising a `std::string`.
struct MnemonicHash {
  using is_transparent = void;
  size_t operator()(std::string_view mnemonic) const {
    return std::hash<std::string_view>{}(mnemonic);
  }
};

/// Keyed by an *owned* mnemonic string.
///
/// `Instruction::mnemonic()` documents its result as pointing to static storage
/// (isa/instruction.h:95-99), but that does not hold for every instruction: the
/// generated FLAT encoding on every architecture, and VOPD on gfx11/gfx12 and
/// CDNA5, synthesise the mnemonic into a per-instruction `std::string` member
/// and point `mnemonic_` at it (e.g. generated/cdna4/encodings.cpp:1149-1153,
/// generated/cdna5/vopd.cpp:261). A view stored past the instruction's lifetime
/// therefore dangles. Copying on first sight costs one allocation per distinct
/// mnemonic; heterogeneous lookup keeps the repeat path allocation-free.
using MnemonicMap = std::unordered_map<std::string, MnemonicStats, MnemonicHash, std::equal_to<>>;

/// Per-wavefront accumulation. Kept wavefront-local so the hot before-execute
/// hook touches no memory shared with another simulation partition and the
/// plugin never needs the group's callback lock on the instruction path.
struct InstructionMixWavefrontState final : WavefrontState {
  MnemonicMap counts;
};

/// Records which ISA mnemonics a run actually executed, and how often.
///
/// One execution is counted each time a wavefront reaches the before-execute
/// hook, matching the throughput plugin: counts are executed *wave*
/// instructions, not active-lane operations. A mnemonic with a non-zero count
/// was reached by at least one wavefront; a mnemonic absent from a *complete*
/// summary was not executed by this run at all (see `complete` below).
///
/// The plugin deliberately reports no architecture name. It observes decoded
/// instructions, not the target that produced them; the harness that chose the
/// config knows the architecture and attributes the report.
///
/// @note Single-XCD scope. Dispatch identity in the plugin callback API is the
///       command processor's `next_dispatch_id_`, which every XCD's CP counts
///       independently from 1, while one plugin group is shared by the whole
///       SoC (vm/soc.cpp SoC::set_plugin_group). On a multi-XCD config two
///       concurrent dispatches on different XCDs can therefore share an id and
///       have their records combined. Mnemonic *coverage* -- the union of what
///       executed, which is what this plugin exists to answer -- survives that,
///       because merging is associative and the summary is the union either
///       way; the per-dispatch attribution and `dispatches` count do not.
///       Fixing it means making dispatch identity unique SoC-wide in the
///       callback API, which the throughput and race plugins key on too. See
///       https://github.com/ROCm/rocm-systems/issues/11774 for the follow-up.
class InstructionMixPlugin final : public ExecutionPlugin {
public:
  /// @param config_json Plugin configuration object as a JSON string. May be
  ///        null, in which case the defaults apply.
  explicit InstructionMixPlugin(const char *config_json = nullptr);
  ~InstructionMixPlugin() override;

  void onShutdown() override;
  void onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) override;
  void onAmdgpuDispatchExecutionEnd(uint32_t dispatch_id) override;
  void onAmdgpuWavefrontDispatched(amdgpu::Wavefront &wf) override;
  void onAmdgpuWavefrontHalted(amdgpu::Wavefront &wf) override;
  void onAmdgpuBeforeExecuteInstruction(uint64_t pc, const Instruction &inst,
                                        amdgpu::Wavefront &wf) override;

  static InstructionFamily classify(const Instruction &inst);
  static std::string_view family_name(InstructionFamily family);

private:
  struct DispatchState {
    KernelDispatchInfo info;
    MnemonicMap counts;
  };

  static void merge(MnemonicMap &destination, const MnemonicMap &source);
  void emit_record(std::string_view record, const KernelDispatchInfo *info,
                   const MnemonicMap &counts);

  /// In-flight dispatches, erased as each one ends. A dispatch that never
  /// reaches onAmdgpuDispatchExecutionEnd stays here until shutdown, which
  /// folds whatever it collected into the summary and counts it in
  /// `incomplete_dispatches_` -- the report's contract is that absence means
  /// non-execution, so an abandoned dispatch's mnemonics must not vanish. It
  /// still gets no `"record":"dispatch"` line, since its per-dispatch totals
  /// are not final. The retained state is a whole mnemonic map rather than a
  /// fixed array, so a run that abandons many dispatches holds more memory
  /// than the throughput plugin would.
  std::unordered_map<uint32_t, DispatchState> dispatches_;
  MnemonicMap aggregate_;
  uint64_t completed_dispatches_ = 0;
  uint64_t incomplete_dispatches_ = 0;
  bool summary_emitted_ = false;
};

} // namespace rocjitsu::plugins::instruction_mix
