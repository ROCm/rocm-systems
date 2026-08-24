// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file waitcheck.h
/// @brief Static waitcnt dependency checker for decoded AMDGPU code.

#pragma once

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/register_set.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {

class CodeObject;
struct WaitcheckKernelInfo;

/// @brief Internal wait counters tracked by waitcheck.
///
/// @details GFX12 targets use the split counter names directly. Legacy gfx9
/// style targets reuse Load for vmcnt, Ds for lgkmcnt, and Exp for expcnt.
enum class WaitCounterKind : uint8_t {
  Load = 0,
  Store,
  Ds,
  Km,
  Sample,
  Bvh,
  Exp,
  X,
  Async,
  Tensor,
  VmVsrc,
  VaVdst,
  Depctr,
  Count,
};

/// @brief How a later instruction conflicts with an outstanding event.
enum class WaitcheckAccessKind : uint8_t {
  Use,
  Def,
  MemoryOrder,
  ProgramEnd,
  ControlTransfer,
};

/// @brief Stable semantic family for one waitcheck diagnostic.
enum class WaitcheckDiagnosticKind : uint8_t {
  WaitCounter = 1,
  SgprDepctr = 2,
  AsyncBarrierPreWait = 3,
  AsyncBarrierPostWait = 4,
  VaVdst = 5,
};

/// @brief One static waitcnt hazard diagnostic.
struct WaitcheckDiagnostic {
  bool has_kernel = false;
  std::string kernel_name;
  uint64_t kernel_entry_offset = 0;
  WaitcheckDiagnosticKind kind = WaitcheckDiagnosticKind::WaitCounter;
  WaitCounterKind counter = WaitCounterKind::Load;
  WaitcheckAccessKind access = WaitcheckAccessKind::Use;
  RegisterRef reg{RegClass::VGPR, 0, 1};
  std::string section_name;
  uint64_t section_offset = 0;
  uint64_t file_offset = 0;
  std::string instruction;
  uint64_t producer_section_offset = 0;
  uint64_t producer_file_offset = 0;
  std::string producer_instruction;
  uint32_t required_count = 0;
  std::string message;
};

/// @brief One emitted wait that is stronger than waitcheck's modeled requirement.
///
/// @details Lower counter values represent stronger waits. For LLVM-produced
/// kernels, required_count > emitted_count is evidence that waitcheck may be
/// under-accounting the dependency protected by the emitted wait.
struct WaitcheckCounterUnderaccountingDiagnostic {
  bool has_kernel = false;
  std::string kernel_name;
  uint64_t kernel_entry_offset = 0;
  WaitCounterKind counter = WaitCounterKind::Load;
  uint32_t emitted_count = 0;
  uint32_t required_count = 0;
  /// @brief Whether waitcheck found a modeled dependency requiring this counter.
  ///
  /// @details When false, required_count is the architecture's no-wait
  /// sentinel for the counter.
  bool has_required_dependency = false;
  WaitcheckAccessKind access = WaitcheckAccessKind::Use;
  RegisterRef reg{RegClass::VGPR, 0, 1};
  std::string section_name;
  uint64_t wait_section_offset = 0;
  uint64_t wait_file_offset = 0;
  std::string wait_instruction;
  uint64_t consumer_section_offset = 0;
  uint64_t consumer_file_offset = 0;
  std::string consumer_instruction;
  uint64_t producer_section_offset = 0;
  uint64_t producer_file_offset = 0;
  std::string producer_instruction;
  std::string message;
};

/// @brief Controls for waitcheck analysis.
struct WaitcheckOptions {
  /// @brief Maximum diagnostics to retain. Analysis still reports failure after
  /// the first hazard when this is zero.
  size_t max_diagnostics = std::numeric_limits<size_t>::max();
  /// @brief Stop checking the current input after the first observed diagnostic.
  ///
  /// @details This is intended for large corpus sweeps where only hazard
  /// presence is needed. Report counts become lower bounds when enabled.
  bool stop_after_first_diagnostic = false;
  /// @brief Compare intact emitted waits with waitcheck's pre-wait requirements.
  ///
  /// @details This opt-in analysis does not alter ordinary hazard diagnostics
  /// or WaitcheckReport::passed(). It normalizes every supported target's
  /// legacy, split, combined, embedded, and implied counter fields.
  bool check_counter_parity = false;
  /// @brief Maximum counter-underaccounting diagnostics to retain.
  size_t max_counter_parity_diagnostics = std::numeric_limits<size_t>::max();
  /// @brief Maximum combined bytes retained for forward and reverse CFG
  /// reachability memoization by one analyzer.
  ///
  /// @details Lossless diagnostic runs can query many distinct producer and
  /// consumer blocks in a large call-context graph. The cache is only a speed
  /// optimization; entries beyond this budget are computed transiently, so
  /// reducing the budget does not change diagnostic results.
  size_t max_reachability_cache_bytes = 16 * 1024 * 1024;
  /// @brief Optional callback invoked after each kernel descriptor is fully
  /// analyzed. The callback is not invoked for symbol-less whole-section
  /// fallback analysis.
  std::function<void()> kernel_analyzed_callback;
  /// @brief Optional callback with the identity and wall time of each analyzed kernel.
  std::function<void(const WaitcheckKernelInfo &, std::chrono::nanoseconds)> kernel_timing_callback;
};

/// @brief One AMDHSA kernel discovered in a final code object.
struct WaitcheckKernelInfo {
  std::string name;
  /// @brief ELF virtual address of the kernel descriptor (`.kd`) symbol.
  uint64_t descriptor_vaddr = 0;
  /// @brief Byte offset of the kernel entry point within the `.text` section.
  uint64_t entry_offset = 0;
  /// @brief Size of the kernel's ELF function symbol, or zero when unavailable.
  uint64_t code_size = 0;
  /// @brief Wavefront size selected by the AMDHSA kernel descriptor.
  uint32_t wavefront_size = 64;
};

/// @brief Result of one waitcheck analysis run.
struct WaitcheckReport {
  bool supported = true;
  /// @brief Whether every wait dependency encountered was covered by the active model.
  ///
  /// @details A supported decoder can still encounter a dependency class whose
  /// inputs are not precise enough to prove a pass or emit a definite hazard.
  /// Such a report retains any definite diagnostics but must not be treated as
  /// clean.
  bool analysis_complete = true;
  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_INVALID;
  size_t instructions_analyzed = 0;
  size_t memory_events_tracked = 0;
  size_t kernels_discovered = 0;
  size_t kernels_analyzed = 0;
  size_t diagnostics_observed = 0;
  bool diagnostics_truncated = false;
  bool stopped_early = false;
  size_t counter_parity_wait_groups = 0;
  size_t counter_parity_fields_checked = 0;
  size_t counter_parity_exact = 0;
  size_t counter_underaccounting_observed = 0;
  size_t counter_unmodeled_wait_observed = 0;
  size_t counter_parity_indeterminate_groups = 0;
  bool counter_parity_diagnostics_truncated = false;
  std::string analysis_error;
  size_t incomplete_observations = 0;
  std::string incomplete_reason;
  std::vector<WaitcheckDiagnostic> diagnostics;
  std::vector<WaitcheckCounterUnderaccountingDiagnostic> counter_underaccounting_diagnostics;

  [[nodiscard]] bool passed() const {
    return supported && analysis_complete && diagnostics_observed == 0;
  }
};

/// @brief Human-readable split counter name, e.g. "loadcnt".
[[nodiscard]] std::string_view wait_counter_name(WaitCounterKind counter);

/// @brief Return the RocJITsu ISA arch supported by waitcheck for @p target.
///
/// @details Unsupported targets return ROCJITSU_CODE_ARCH_INVALID.
[[nodiscard]] rj_code_arch_t waitcheck_arch_for_target(rj_code_target_id_t target);

/// @brief Discover independently analyzable AMDHSA kernels in a code object.
///
/// @details The descriptor virtual address can be combined with the loader's
/// load delta to identify the `kernel_object` field in an AQL dispatch packet.
[[nodiscard]] std::vector<WaitcheckKernelInfo> waitcheck_kernels(const CodeObject &code_object);

/// @brief Analyze one stream of 32-bit instruction words.
///
/// @details Unsupported architectures return a report with supported=false.
[[nodiscard]] WaitcheckReport analyze_waitcnts(std::span<const uint32_t> words, rj_code_arch_t arch,
                                               WaitcheckOptions options = {});

/// @brief Analyze all executable sections in a code object.
///
/// @details Offsets in diagnostics are relative to each section and to the ELF
/// file offset when the section exposes one.
[[nodiscard]] WaitcheckReport analyze_waitcnts(const CodeObject &code_object, rj_code_arch_t arch,
                                               WaitcheckOptions options = {});

/// @brief Analyze only the kernel whose entry point is at @p kernel_entry_offset.
///
/// @details Reachable functions are included, while sibling kernels and
/// unrelated executable sections are skipped. This is the runtime-oriented
/// counterpart to the exhaustive code-object overload above.
[[nodiscard]] WaitcheckReport analyze_waitcnts_for_kernel(const CodeObject &code_object,
                                                          rj_code_arch_t arch,
                                                          uint64_t kernel_entry_offset,
                                                          WaitcheckOptions options = {});

/// @brief Analyze an already-discovered kernel without repeating descriptor discovery.
///
/// @details This overload is intended for schedulers and runtime hooks that
/// obtained @p kernel from waitcheck_kernels().
[[nodiscard]] WaitcheckReport analyze_waitcnts_for_kernel(const CodeObject &code_object,
                                                          rj_code_arch_t arch,
                                                          const WaitcheckKernelInfo &kernel,
                                                          WaitcheckOptions options = {});

/// @brief Analyze an already-discovered batch of kernels with shared setup.
///
/// @details Each kernel is still analyzed independently, but the decoder,
/// diagnostic state, and report are reused across the batch. This is intended
/// for bounded-granularity offline schedulers.
[[nodiscard]] WaitcheckReport
analyze_waitcnts_for_kernels(const CodeObject &code_object, rj_code_arch_t arch,
                             std::span<const WaitcheckKernelInfo> kernels,
                             WaitcheckOptions options = {});

} // namespace rocjitsu
