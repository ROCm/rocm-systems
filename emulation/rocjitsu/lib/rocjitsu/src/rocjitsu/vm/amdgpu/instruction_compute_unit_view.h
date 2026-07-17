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
class GpuMemory;
class L1ScalarCache;
class L1VectorCache;
class L2Cache;
class Lds;
class RegisterAccess;

/// @brief Narrow CU API exposed to AMDGPU instruction emulation code.
///
/// @details This view deliberately omits public physical register storage APIs.
/// VM storage code can still use ComputeUnitCore directly; instruction bodies
/// should reach registers through Operand or RegisterAccess.
class InstructionComputeUnitView {
public:
  explicit InstructionComputeUnitView(ComputeUnitCore &cu) : cu_(&cu) {}

  GpuMemory *memory() const;
  L1ScalarCache &l1_scalar();
  L1VectorCache &l1_vector();
  L2Cache *l2() const;
  Lds &lds();
  bool sram_ecc() const;
  rj_code_arch_t arch() const;
  uint32_t wf_size() const;
  uint32_t sgprs_per_wf() const;
  std::string full_path() const;
  simdojo::ComponentID id() const;
  simdojo::SimulationEngine *engine() const;
  void request_functional_yield();

private:
  uint32_t read_sgpr(uint32_t reg_idx) const;
  void write_sgpr(uint32_t reg_idx, uint32_t value);

  ComputeUnitCore &raw_cu() { return *cu_; }
  const ComputeUnitCore &raw_cu() const { return *cu_; }

  ComputeUnitCore *cu_ = nullptr;

  // RegisterAccess unwraps the view so instruction helpers can use the same
  // observed physical-register facade whether they were passed a full CU or an
  // instruction-facing CU view.
  friend class RegisterAccess;

  // The ISA operand backend is the only instruction-side code allowed to reach
  // private CU register hooks. It remains private behind Operand/RegisterAccess.
  template <typename Isa> friend class ::rocjitsu::AmdgpuIsaOperand;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_INSTRUCTION_COMPUTE_UNIT_VIEW_H_
