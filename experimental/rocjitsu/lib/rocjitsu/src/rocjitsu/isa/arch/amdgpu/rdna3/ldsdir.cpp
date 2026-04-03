// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/rdna3/ldsdir.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/except.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace rocjitsu {
namespace rdna3 {

LdsParamLoadLdsdir::LdsParamLoadLdsdir(const MachineInst *inst)
    : Ldsdir("lds_param_load", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      attr(32, OperandType::OPR_ATTR, reinterpret_cast<const OpEncoding *>(inst)->attr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&attr);
}

void LdsParamLoadLdsdir::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Interpolation/LDS-direct: no-op in compute simulation.
}

LdsDirectLoadLdsdir::LdsDirectLoadLdsdir(const MachineInst *inst)
    : Ldsdir("lds_direct_load", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst) {
  dst_operands_.emplace_back(&vdst);
}

void LdsDirectLoadLdsdir::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Interpolation/LDS-direct: no-op in compute simulation.
}

} // namespace rdna3
} // namespace rocjitsu
