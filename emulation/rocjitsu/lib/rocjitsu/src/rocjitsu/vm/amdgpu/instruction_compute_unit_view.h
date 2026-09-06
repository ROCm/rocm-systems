// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file instruction_compute_unit_view.h
/// @brief Instruction-facing compute-unit service view.

#ifndef ROCJITSU_VM_AMDGPU_INSTRUCTION_COMPUTE_UNIT_VIEW_H_
#define ROCJITSU_VM_AMDGPU_INSTRUCTION_COMPUTE_UNIT_VIEW_H_

#include "rocjitsu/code/rj_code.h"
#include "simdojo/sim/sim_types.h"

#include <cstdint>
#include <string>

namespace simdojo {
class SimulationEngine;
}

namespace rocjitsu {
template <typename Isa> class AmdgpuIsaOperand;

namespace amdgpu {
class ComputeUnitCore;
class InstructionCache;
class L1ScalarCache;
class L1VectorCache;
class L2Cache;
class Lds;
class OperandExecutionAccess;
class RegisterAccess;
class Wavefront;

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
  rj_code_arch_t arch() const;
  uint32_t wf_size() const;
  uint32_t sgprs_per_wf() const;
  uint32_t vgpr_allocation_block_size() const;
  std::string full_path() const;
  simdojo::ComponentID id() const;
  simdojo::SimulationEngine *engine() const;
  uint32_t fetch_instruction_word(uint64_t address, uint32_t process_id) const;
  void request_functional_yield();
  bool handle_sendmsg(Wavefront &wf, uint32_t message);
  void notify_trap_complete(Wavefront &wf);
  /// Queue a runtime exception for delivery after instruction execution drops
  /// the compute unit's wave-state lock.
  bool signal_queue_exception(uint32_t queue_id, uint32_t process_id, uint64_t status);

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
