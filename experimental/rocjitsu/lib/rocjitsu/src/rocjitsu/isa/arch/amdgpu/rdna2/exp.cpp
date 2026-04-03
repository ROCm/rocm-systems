// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/rdna2/exp.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/except.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace rocjitsu {
namespace rdna2 {

ExpExp::ExpExp(const MachineInst *inst)
    : Exp("exp", reinterpret_cast<const OpEncoding *>(inst)),
      tgt(128, OperandType::OPR_TGT, reinterpret_cast<const OpEncoding *>(inst)->tgt),
      vsrc0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1),
      vsrc2(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc2),
      vsrc3(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc3) {
  dst_operands_.emplace_back(&tgt);
  src_operands_.emplace_back(&vsrc0);
  src_operands_.emplace_back(&vsrc1);
  src_operands_.emplace_back(&vsrc2);
  src_operands_.emplace_back(&vsrc3);
}

void ExpExp::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Export: no-op in compute simulation.
}

} // namespace rdna2
} // namespace rocjitsu
