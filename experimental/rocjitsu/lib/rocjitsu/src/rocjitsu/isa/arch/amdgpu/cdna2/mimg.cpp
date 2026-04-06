// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/cdna2/mimg.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/except.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace rocjitsu {
namespace cdna2 {

ImageLoadMimg::ImageLoadMimg(const MachineInst *inst)
    : Mimg("image_load", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageLoadMimg::execute(amdgpu::Wavefront &wf) {
  // Minimal image load stub — not yet implemented.
  (void)wf;
}

ImageLoadMipMimg::ImageLoadMipMimg(const MachineInst *inst)
    : Mimg("image_load_mip", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageLoadMipMimg::execute(amdgpu::Wavefront &wf) {
  // Minimal image load stub — not yet implemented.
  (void)wf;
}

ImageLoadPckMimg::ImageLoadPckMimg(const MachineInst *inst)
    : Mimg("image_load_pck", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageLoadPckMimg::execute(amdgpu::Wavefront &wf) {
  // Minimal image load stub — not yet implemented.
  (void)wf;
}

ImageLoadPckSgnMimg::ImageLoadPckSgnMimg(const MachineInst *inst)
    : Mimg("image_load_pck_sgn", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageLoadPckSgnMimg::execute(amdgpu::Wavefront &wf) {
  // Minimal image load stub — not yet implemented.
  (void)wf;
}

ImageLoadMipPckMimg::ImageLoadMipPckMimg(const MachineInst *inst)
    : Mimg("image_load_mip_pck", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageLoadMipPckMimg::execute(amdgpu::Wavefront &wf) {
  // Minimal image load stub — not yet implemented.
  (void)wf;
}

ImageLoadMipPckSgnMimg::ImageLoadMipPckSgnMimg(const MachineInst *inst)
    : Mimg("image_load_mip_pck_sgn", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageLoadMipPckSgnMimg::execute(amdgpu::Wavefront &wf) {
  // Minimal image load stub — not yet implemented.
  (void)wf;
}

ImageStoreMimg::ImageStoreMimg(const MachineInst *inst)
    : Mimg("image_store", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageStoreMimg::execute(amdgpu::Wavefront &wf) {
  // Minimal image store stub — not yet implemented.
  (void)wf;
}

ImageStoreMipMimg::ImageStoreMipMimg(const MachineInst *inst)
    : Mimg("image_store_mip", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageStoreMipMimg::execute(amdgpu::Wavefront &wf) {
  // Minimal image store stub — not yet implemented.
  (void)wf;
}

ImageStorePckMimg::ImageStorePckMimg(const MachineInst *inst)
    : Mimg("image_store_pck", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageStorePckMimg::execute(amdgpu::Wavefront &wf) {
  // Minimal image store stub — not yet implemented.
  (void)wf;
}

ImageStoreMipPckMimg::ImageStoreMipPckMimg(const MachineInst *inst)
    : Mimg("image_store_mip_pck", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageStoreMipPckMimg::execute(amdgpu::Wavefront &wf) {
  // Minimal image store stub — not yet implemented.
  (void)wf;
}

ImageGetResinfoMimg::ImageGetResinfoMimg(const MachineInst *inst)
    : Mimg("image_get_resinfo", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageGetResinfoMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Image pipeline not yet implemented.
}

ImageAtomicSwapMimg::ImageAtomicSwapMimg(const MachineInst *inst)
    : Mimg("image_atomic_swap", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageAtomicSwapMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Image pipeline not yet implemented.
}

ImageAtomicCmpswapMimg::ImageAtomicCmpswapMimg(const MachineInst *inst)
    : Mimg("image_atomic_cmpswap", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageAtomicCmpswapMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Image pipeline not yet implemented.
}

ImageAtomicAddMimg::ImageAtomicAddMimg(const MachineInst *inst)
    : Mimg("image_atomic_add", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageAtomicAddMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Image pipeline not yet implemented.
}

ImageAtomicSubMimg::ImageAtomicSubMimg(const MachineInst *inst)
    : Mimg("image_atomic_sub", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageAtomicSubMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Image pipeline not yet implemented.
}

ImageAtomicSminMimg::ImageAtomicSminMimg(const MachineInst *inst)
    : Mimg("image_atomic_smin", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageAtomicSminMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Image pipeline not yet implemented.
}

ImageAtomicUminMimg::ImageAtomicUminMimg(const MachineInst *inst)
    : Mimg("image_atomic_umin", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageAtomicUminMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Image pipeline not yet implemented.
}

ImageAtomicSmaxMimg::ImageAtomicSmaxMimg(const MachineInst *inst)
    : Mimg("image_atomic_smax", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageAtomicSmaxMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Image pipeline not yet implemented.
}

ImageAtomicUmaxMimg::ImageAtomicUmaxMimg(const MachineInst *inst)
    : Mimg("image_atomic_umax", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageAtomicUmaxMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Image pipeline not yet implemented.
}

ImageAtomicAndMimg::ImageAtomicAndMimg(const MachineInst *inst)
    : Mimg("image_atomic_and", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageAtomicAndMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Image pipeline not yet implemented.
}

ImageAtomicOrMimg::ImageAtomicOrMimg(const MachineInst *inst)
    : Mimg("image_atomic_or", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageAtomicOrMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Image pipeline not yet implemented.
}

ImageAtomicXorMimg::ImageAtomicXorMimg(const MachineInst *inst)
    : Mimg("image_atomic_xor", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageAtomicXorMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Image pipeline not yet implemented.
}

ImageAtomicIncMimg::ImageAtomicIncMimg(const MachineInst *inst)
    : Mimg("image_atomic_inc", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageAtomicIncMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Image pipeline not yet implemented.
}

ImageAtomicDecMimg::ImageAtomicDecMimg(const MachineInst *inst)
    : Mimg("image_atomic_dec", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageAtomicDecMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Image pipeline not yet implemented.
}

ImageSampleMimg::ImageSampleMimg(const MachineInst *inst)
    : Mimg("image_sample", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(96, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Image pipeline not yet implemented.
}

} // namespace cdna2
} // namespace rocjitsu
