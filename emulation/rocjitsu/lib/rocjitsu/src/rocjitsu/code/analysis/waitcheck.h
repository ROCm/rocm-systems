// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file waitcheck.h
/// @brief Static waitcnt dependency checker for decoded AMDGPU code.

#pragma once

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/shared/wait_counter_policy.h"
#include "rocjitsu/isa/register_set.h"
#include "rocjitsu/isa/target_registry.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {

using amdgpu::WaitCounterKind;

class CodeObject;
class AmdGpuCodeObject;
struct WaitcheckKernelInfo;
namespace amdhsa {
struct kernel_descriptor_t;
}

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
  /// @brief Wavefront size when no AMDHSA descriptor supplies one.
  ///
  /// @details Raw streams default to wave64, matching WaitcheckStreamOptions.
  /// Descriptor-less ELF code objects, including helper-only modules, retain
  /// the target's conventional default (wave32 on RDNA3/3.5/4 and CDNA5;
  /// wave64 on CDNA3/4). Supply 32 or 64
  /// when the actual ELF wave mode is known. For executable descriptorless
  /// code, a complete result also requires an explicit entry offset;
  /// otherwise the fallback is diagnostic-only. Described kernels always use
  /// their descriptor, even when this override is present.
  std::optional<uint32_t> fallback_wavefront_size = std::nullopt;
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
  /// @brief Caller-selected ISA providers for decoding and concrete GPU capabilities.
  std::reference_wrapper<const IsaTargetRegistry> registry = default_isa_target_registry();
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
  /// Other descriptors selecting the same executable entry and wave size in
  /// an offline group. The group's allocation bounds are not a dispatch ABI.
  std::vector<uint64_t> alias_descriptor_vaddrs;
  /// Bounds across descriptors sharing this entry and wave size, in dwords.
  uint32_t min_vgprs = 0, max_vgprs = 0;
  uint32_t min_sgprs = 0, max_sgprs = 0;
  /// Kernel argument SGPR pair and byte extent of the first descriptor.
  /// An offline group with aliases cannot use these for a shared argument proof.
  std::optional<uint16_t> kernarg_sgpr = std::nullopt;
  uint32_t kernarg_size = 0;
  /// Largest ordinary and accumulator banks across descriptors. A zero
  /// ordinary limit denotes missing descriptor evidence.
  uint16_t max_memory_vgprs = 0, max_memory_acc_vgprs = 0;
};

/// @brief Result of one waitcheck analysis run.
struct WaitcheckReport {
  bool supported = true;
  /// False when reachable control flow has unknown successors.
  bool complete = true;
  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_INVALID;
  size_t instructions_analyzed = 0;
  size_t memory_events_tracked = 0;
  size_t kernels_discovered = 0;
  size_t kernels_analyzed = 0;
  /// @brief Exact observed count, even when storage is capped. A lower bound
  /// when analysis stops early.
  size_t diagnostics_observed = 0;
  /// @brief Details were omitted, or analysis stopped early.
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
  std::vector<WaitcheckDiagnostic> diagnostics;
  std::vector<WaitcheckCounterUnderaccountingDiagnostic> counter_underaccounting_diagnostics;

  [[nodiscard]] bool passed() const { return supported && complete && diagnostics_observed == 0; }
};

/// @brief Return the RocJITsu ISA arch supported by waitcheck for @p target.
///
/// @details Unsupported targets return ROCJITSU_CODE_ARCH_INVALID.
[[nodiscard]] rj_code_arch_t waitcheck_arch_for_target(rj_code_target_id_t target);

/// @brief Conventional wavefront size for a descriptor-less code object.
///
/// @details A descriptor or explicit fallback override takes precedence. Raw
/// word streams independently default to wave64.
[[nodiscard]] uint32_t waitcheck_default_wavefront_size(rj_code_arch_t arch);

/// @brief Build exact register-allocation and wavefront bounds for one descriptor.
///
/// @param entry_text_offset Byte offset of the entry within `.text`, not the
/// signed descriptor-relative kernel_code_entry_byte_offset. Use
/// waitcheck_kernels_exact() when a whole AMDHSA code object is available.
[[nodiscard]] WaitcheckKernelInfo
waitcheck_kernel_from_descriptor(std::string name, uint64_t descriptor_vaddr, uint64_t entry_text_offset,
                                 uint64_t code_size, rj_code_arch_t arch,
                                 const amdhsa::kernel_descriptor_t &descriptor);

/// @brief Discover independently analyzable AMDHSA entry and wave groups.
///
/// @details Descriptors sharing an entry and wave are merged conservatively for
/// offline analysis. The allocation and argument fields of a merged group do
/// not describe any particular dispatch. Runtime hooks must build an exact
/// WaitcheckKernelInfo with waitcheck_kernel_from_descriptor() from the launched
/// descriptor, identified by AQL kernel_object and the loader's load delta.
[[nodiscard]] std::vector<WaitcheckKernelInfo>
waitcheck_kernels(const AmdGpuCodeObject &code_object);

/// @brief Discover one kernel record for each AMDHSA dispatch descriptor.
///
/// @details Each result keeps its descriptor's exact allocation and wavefront
/// bounds. Runtime dispatch hooks should match descriptor_vaddr to the packet's
/// kernel_object after applying the loader's address delta.
[[nodiscard]] std::vector<WaitcheckKernelInfo>
waitcheck_kernels_exact(const AmdGpuCodeObject &code_object);

/// @brief Analyze one stream of 32-bit instruction words.
///
/// @details The stream has no descriptor; options.fallback_wavefront_size selects
/// wave32 or wave64 and defaults to wave64. Unsupported architectures or wave
/// sizes return a report with supported=false.
[[nodiscard]] WaitcheckReport analyze_waitcnts(std::span<const uint32_t> words, rj_code_arch_t arch,
                                               WaitcheckOptions options = {});

/// @brief Analyze all executable sections in a code object.
///
/// @details Offsets in diagnostics are relative to each section and to the ELF
/// file offset when the section exposes one. Executable descriptorless code
/// without both an explicit entry offset and wavefront size remains incomplete,
/// although reachable diagnostics are retained.
[[nodiscard]] WaitcheckReport analyze_waitcnts(const CodeObject &code_object, rj_code_arch_t arch,
                                               WaitcheckOptions options = {});

/// @brief Analyze all merged entry/wave groups at @p kernel_entry_offset.
///
/// @details waitcheck_kernels() merges same-wave descriptors at one entry and
/// conservatively combines their allocation bounds; distinct wavefront modes
/// are analyzed separately. Reachable functions are included, while kernels
/// at other entries and unrelated executable sections are skipped. Descriptorless
/// callers must also supply fallback_wavefront_size for a complete result.
/// Dispatch callers with a descriptor must use the WaitcheckKernelInfo overload
/// below with exact descriptor bounds and wavefront mode.
[[nodiscard]] WaitcheckReport analyze_waitcnts_for_kernel(const CodeObject &code_object,
                                                          rj_code_arch_t arch,
                                                          uint64_t kernel_entry_offset,
                                                          WaitcheckOptions options = {});

/// @brief Analyze an already-discovered kernel without repeating descriptor discovery.
///
/// @details Offline schedulers may pass a merged entry and wave group from
/// waitcheck_kernels(). Runtime hooks must pass exact descriptor information
/// built with waitcheck_kernel_from_descriptor().
[[nodiscard]] WaitcheckReport analyze_waitcnts_for_kernel(const CodeObject &code_object,
                                                          rj_code_arch_t arch,
                                                          const WaitcheckKernelInfo &kernel,
                                                          WaitcheckOptions options = {});

/// @brief Analyze an already-discovered batch of kernels with shared setup.
///
/// @details Each kernel is still analyzed independently, but the decoder,
/// diagnostic state, and report are reused across the batch. This is intended
/// for bounded-granularity offline schedulers. It retains whole-object
/// executable-section coverage; use the single-kernel overload to scope a
/// dispatch when unrelated executable sections are present. An empty span
/// with executable text is unsupported; callers that receive no entries from
/// waitcheck_kernels() can use analyze_waitcnts() for diagnostic-only function
/// or section coverage.
[[nodiscard]] WaitcheckReport
analyze_waitcnts_for_kernels(const CodeObject &code_object, rj_code_arch_t arch,
                             std::span<const WaitcheckKernelInfo> kernels,
                             WaitcheckOptions options = {});

} // namespace rocjitsu
