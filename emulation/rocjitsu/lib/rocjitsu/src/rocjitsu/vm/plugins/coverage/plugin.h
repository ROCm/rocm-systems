// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "rocjitsu/vm/plugins/kernel_dispatch_info.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace rocjitsu::plugins::coverage {

/// Exclusive instruction families used by the coverage report. The order is
/// part of the JSONL schema and should remain stable. It mirrors the
/// throughput plugin's families so the two reports can be joined.
enum class InstructionFamily : size_t {
  Scalar,
  Vector,
  Matrix,
  Lds,
  Global,
  Control,
  Other,
  Count,
};

inline constexpr size_t kInstructionFamilyCount = static_cast<size_t>(InstructionFamily::Count);

/// One covered mnemonic. A mnemonic reachable through more than one encoding is
/// still one entry, because the coverage denominators (the ISA XML and the
/// generated sources) enumerate mnemonics rather than encodings.
///
/// The encoding fields then have to pick one of those encodings, and the pick
/// must not depend on which wavefront happened to finish first: the report is
/// meant to be diffable between runs. `merge()` keeps the entry that is
/// smallest by (first_dispatch_id, encoding_id, opcode), which is a total
/// order over sightings and therefore stable.
struct MnemonicCoverage {
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
using CoverageMap =
    std::unordered_map<std::string, MnemonicCoverage, MnemonicHash, std::equal_to<>>;

/// Per-wavefront accumulation. Kept wavefront-local so the hot before-execute
/// hook touches no memory shared with another simulation partition and the
/// plugin never needs the group's callback lock on the instruction path.
struct CoverageWavefrontState final : WavefrontState {
  CoverageMap counts;
};

/// Records which ISA mnemonics a run actually executed.
///
/// One execution is counted each time a wavefront reaches the before-execute
/// hook, matching the throughput plugin: counts are executed *wave*
/// instructions, not active-lane operations. A mnemonic with a non-zero count
/// was reached by at least one wavefront; a mnemonic absent from the report was
/// not executed by this run at all.
///
/// The plugin deliberately reports no architecture name. It observes decoded
/// instructions, not the target that produced them; the harness that chose the
/// config knows the architecture and attributes the report.
class CoveragePlugin final : public ExecutionPlugin {
public:
  /// @param config_json Plugin configuration object as a JSON string. May be
  ///        null, in which case the defaults apply.
  explicit CoveragePlugin(const char *config_json = nullptr);
  ~CoveragePlugin() override;

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
    CoverageMap counts;
  };

  static void merge(CoverageMap &destination, const CoverageMap &source);
  void emit_record(std::string_view record, const KernelDispatchInfo *info,
                   const CoverageMap &counts);

  /// In-flight dispatches, erased as each one ends. A dispatch that never
  /// reaches onAmdgpuDispatchExecutionEnd stays here until shutdown and is
  /// absent from the per-dispatch records, matching the throughput plugin --
  /// but here the retained state is a whole mnemonic map rather than a fixed
  /// array, so a run that abandons many dispatches holds more memory.
  std::unordered_map<uint32_t, DispatchState> dispatches_;
  CoverageMap aggregate_;
  uint64_t completed_dispatches_ = 0;
  bool summary_emitted_ = false;
};

} // namespace rocjitsu::plugins::coverage
