// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/rdna4/vdsdir.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/except.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace rocjitsu {
namespace rdna4 {

DsParamLoadVdsdir::DsParamLoadVdsdir(const MachineInst *inst)
    : Vdsdir("ds_param_load", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      attr(32, OperandType::OPR_ATTR, reinterpret_cast<const OpEncoding *>(inst)->attr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&attr);
}

void DsParamLoadVdsdir::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Interpolation/LDS-direct: no-op in compute simulation.
}

DsDirectLoadVdsdir::DsDirectLoadVdsdir(const MachineInst *inst)
    : Vdsdir("ds_direct_load", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst) {
  dst_operands_.emplace_back(&vdst);
}

void DsDirectLoadVdsdir::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Interpolation/LDS-direct: no-op in compute simulation.
}

} // namespace rdna4
} // namespace rocjitsu
