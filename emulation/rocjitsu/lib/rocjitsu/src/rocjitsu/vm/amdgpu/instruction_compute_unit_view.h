// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file instruction_compute_unit_view.h
/// @brief Instruction-facing compute-unit service view.

#ifndef ROCJITSU_VM_AMDGPU_INSTRUCTION_COMPUTE_UNIT_VIEW_H_
#define ROCJITSU_VM_AMDGPU_INSTRUCTION_COMPUTE_UNIT_VIEW_H_

#include "rocjitsu/code/rj_code.h"
#include "simdojo/sim/sim_types.h"
#include "util/result.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace simdojo {
class SimulationEngine;
}

namespace rocjitsu {
template <typename Isa> class AmdgpuIsaOperand;

namespace amdgpu {
class ComputeUnitCore;
struct ImageAddressConfig;
class InstructionCache;
class L1ScalarCache;
class L1VectorCache;
class L2Cache;
class Lds;
class OperandExecutionAccess;
class RegisterAccess;
struct TensorDmaIssueObservation;
struct TensorDmaMemoryAccessObservation;
class Wavefront;

/// @brief Completion contract for a deferred runtime queue exception.
enum class QueueExceptionDeliveryPolicy {
  /// Preserve the trap-handler acknowledgment and debugger-ownership protocol.
  WaitForAcknowledgment,
  /// Publish without waiting for ROCr; fail the engine if publication fails.
  PublishOrFail,
  /// Publish without waiting; the caller cancels all queue replicas and fails
  /// the engine after their acknowledgments, even if publication fails.
  PublishAndCancel,
};

/// One shader export presented to the graphics front end. Only active lanes
/// and enabled components are defined. Compressed components contain 16 bits.
struct ShaderExport {
  uint8_t target;
  uint8_t enabled;
  bool compressed;
  bool valid_mask;
  bool done;
  uint64_t lanes;
  std::array<std::array<uint32_t, 4>, 64> values{};
};

/// @brief Narrow CU API exposed to AMDGPU instruction emulation code.
///
/// @details This view deliberately omits public physical register storage APIs.
/// VM storage code can still use ComputeUnitCore directly; instruction bodies
/// should reach registers through Operand or RegisterAccess.
class InstructionComputeUnitView {
public:
  InstructionComputeUnitView(ComputeUnitCore &cu, Wavefront &wf) : cu_(&cu), wf_(&wf) {}

  InstructionCache &instruction_cache();
  L1ScalarCache &l1_scalar();
  L1VectorCache &l1_vector();
  L2Cache *l2() const;
  Lds &lds();
  bool sram_ecc() const;
  bool setreg_vgpr_msb_fixup() const;
  /// Whether register accesses have a diagnostic or plugin consumer.
  bool observes_register_access() const;
  bool debug_active() const;
  rj_code_arch_t arch() const;
  uint64_t arithmetic_flags() const;
  uint64_t shader_cycles() const;
  ImageAddressConfig image_address_config() const;
  uint32_t cus_per_shader_array() const;
  uint32_t shader_array_cu_id() const;
  uint32_t shader_array_id() const;
  uint32_t shader_engine_id() const;
  uint32_t shader_engine_cu_count() const;
  std::optional<uint32_t> xcc_id() const;
  uint32_t wf_size() const;
  uint32_t sgprs_per_wf() const;
  uint32_t vgpr_allocation_block_size() const;
  std::string full_path() const;
  simdojo::ComponentID id() const;
  simdojo::SimulationEngine *engine() const;
  uint32_t fetch_instruction_word(uint64_t address, uint32_t process_id) const;
  void request_functional_yield();
  void wake_sleeping_workgroup(const Wavefront &issuer,
                               std::optional<int32_t> named_barrier = std::nullopt);
  void wake_sleeping_workgroup() { wake_sleeping_workgroup(raw_wavefront()); }
  bool in_multiwave_workgroup() const;
  void update_shader_perf_level(uint32_t id, int change);
  void emit_shader_trace_data(uint32_t data);
  util::FailureOr<uint64_t> return_message(uint32_t message);
  uint64_t trap_message_address(uint32_t message) const;
  bool update_trap_address(bool memory, uint64_t mask, uint64_t value);
  uint64_t cycle_count() const;
  /// Configured TBA and TMA byte addresses for the issuing process.
  std::pair<uint64_t, uint64_t> trap_addresses() const;
  bool deliver_sendmsg(Wavefront &wf, uint32_t message);
  bool shader_export_ready() const;
  void export_shader_outputs(Wavefront &wf, uint8_t target, uint8_t enabled, bool compressed,
                             bool valid_mask, bool done, const std::array<uint8_t, 4> &sources);
  void notify_trap_complete(Wavefront &wf);
  /// Queue a runtime exception for delivery after instruction execution drops
  /// the compute unit's wave-state lock. A caller without a debugger fallback
  /// can require engine failure if the runtime cannot take ownership. Set
  /// retain_wave=false when the dispatch will abort before delivery completes.
  bool signal_queue_exception(uint32_t queue_id, uint32_t process_id, uint64_t status,
                              bool clear_debug_stop_on_success = false,
                              bool retain_failure_for_debugger = true,
                              QueueExceptionDeliveryPolicy delivery_policy =
                                  QueueExceptionDeliveryPolicy::WaitForAcknowledgment,
                              bool retain_wave = true);
  bool observes_tensor_dma_memory_access() const;
  void report_tensor_dma_memory_access(const TensorDmaMemoryAccessObservation &access);
  bool observes_tensor_dma_issue() const;
  void report_tensor_dma_issue(const TensorDmaIssueObservation &issue);

private:
  ComputeUnitCore &raw_cu() { return *cu_; }
  const ComputeUnitCore &raw_cu() const { return *cu_; }
  Wavefront &raw_wavefront() { return *wf_; }
  const Wavefront &raw_wavefront() const { return *wf_; }

  ComputeUnitCore *cu_ = nullptr;
  Wavefront *wf_ = nullptr;

  // RegisterAccess unwraps both the CU and the owning wave. This keeps legacy
  // instruction helpers that accept a CU-shaped service view wave-bound:
  // physical register indices are validated against the executing wave rather
  // than re-attributed through the CU's reverse ownership maps.
  friend class RegisterAccess;

  // The non-split ISA operand fallback and the execution-only access key are
  // the only instruction-side code allowed to reach private CU register hooks.
  template <typename Isa> friend class ::rocjitsu::AmdgpuIsaOperand;
  friend class OperandExecutionAccess;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_INSTRUCTION_COMPUTE_UNIT_VIEW_H_
