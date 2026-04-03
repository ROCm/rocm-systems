// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/rdna1/mimg.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/except.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace rocjitsu {
namespace rdna1 {

ImageLoadMimg::ImageLoadMimg(const MachineInst *inst)
    : Mimg("image_load", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageLoadMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_load
}

ImageLoadMipMimg::ImageLoadMipMimg(const MachineInst *inst)
    : Mimg("image_load_mip", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageLoadMipMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_load
}

ImageLoadPckMimg::ImageLoadPckMimg(const MachineInst *inst)
    : Mimg("image_load_pck", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageLoadPckMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_load
}

ImageLoadPckSgnMimg::ImageLoadPckSgnMimg(const MachineInst *inst)
    : Mimg("image_load_pck_sgn", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageLoadPckSgnMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_load
}

ImageLoadMipPckMimg::ImageLoadMipPckMimg(const MachineInst *inst)
    : Mimg("image_load_mip_pck", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageLoadMipPckMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_load
}

ImageLoadMipPckSgnMimg::ImageLoadMipPckSgnMimg(const MachineInst *inst)
    : Mimg("image_load_mip_pck_sgn", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageLoadMipPckSgnMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_load
}

ImageStoreMimg::ImageStoreMimg(const MachineInst *inst)
    : Mimg("image_store", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageStoreMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_store
}

ImageStoreMipMimg::ImageStoreMipMimg(const MachineInst *inst)
    : Mimg("image_store_mip", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageStoreMipMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_store
}

ImageStorePckMimg::ImageStorePckMimg(const MachineInst *inst)
    : Mimg("image_store_pck", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageStorePckMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_store
}

ImageStoreMipPckMimg::ImageStoreMipPckMimg(const MachineInst *inst)
    : Mimg("image_store_mip_pck", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageStoreMipPckMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_store
}

ImageGetResinfoMimg::ImageGetResinfoMimg(const MachineInst *inst)
    : Mimg("image_get_resinfo", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageGetResinfoMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_query
}

ImageAtomicSwapMimg::ImageAtomicSwapMimg(const MachineInst *inst)
    : Mimg("image_atomic_swap", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageAtomicSwapMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_atomic
}

ImageAtomicCmpswapMimg::ImageAtomicCmpswapMimg(const MachineInst *inst)
    : Mimg("image_atomic_cmpswap", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageAtomicCmpswapMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_atomic
}

ImageAtomicAddMimg::ImageAtomicAddMimg(const MachineInst *inst)
    : Mimg("image_atomic_add", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageAtomicAddMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_atomic
}

ImageAtomicSubMimg::ImageAtomicSubMimg(const MachineInst *inst)
    : Mimg("image_atomic_sub", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageAtomicSubMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_atomic
}

ImageAtomicSminMimg::ImageAtomicSminMimg(const MachineInst *inst)
    : Mimg("image_atomic_smin", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageAtomicSminMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_atomic
}

ImageAtomicUminMimg::ImageAtomicUminMimg(const MachineInst *inst)
    : Mimg("image_atomic_umin", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageAtomicUminMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_atomic
}

ImageAtomicSmaxMimg::ImageAtomicSmaxMimg(const MachineInst *inst)
    : Mimg("image_atomic_smax", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageAtomicSmaxMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_atomic
}

ImageAtomicUmaxMimg::ImageAtomicUmaxMimg(const MachineInst *inst)
    : Mimg("image_atomic_umax", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageAtomicUmaxMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_atomic
}

ImageAtomicAndMimg::ImageAtomicAndMimg(const MachineInst *inst)
    : Mimg("image_atomic_and", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageAtomicAndMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_atomic
}

ImageAtomicOrMimg::ImageAtomicOrMimg(const MachineInst *inst)
    : Mimg("image_atomic_or", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageAtomicOrMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_atomic
}

ImageAtomicXorMimg::ImageAtomicXorMimg(const MachineInst *inst)
    : Mimg("image_atomic_xor", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageAtomicXorMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_atomic
}

ImageAtomicIncMimg::ImageAtomicIncMimg(const MachineInst *inst)
    : Mimg("image_atomic_inc", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageAtomicIncMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_atomic
}

ImageAtomicDecMimg::ImageAtomicDecMimg(const MachineInst *inst)
    : Mimg("image_atomic_dec", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageAtomicDecMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_atomic
}

ImageAtomicFcmpswapMimg::ImageAtomicFcmpswapMimg(const MachineInst *inst)
    : Mimg("image_atomic_fcmpswap", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageAtomicFcmpswapMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_atomic
}

ImageAtomicFminMimg::ImageAtomicFminMimg(const MachineInst *inst)
    : Mimg("image_atomic_fmin", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageAtomicFminMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_atomic
}

ImageAtomicFmaxMimg::ImageAtomicFmaxMimg(const MachineInst *inst)
    : Mimg("image_atomic_fmax", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageAtomicFmaxMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_atomic
}

ImageSampleMimg::ImageSampleMimg(const MachineInst *inst)
    : Mimg("image_sample", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(96, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleClMimg::ImageSampleClMimg(const MachineInst *inst)
    : Mimg("image_sample_cl", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleClMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleDMimg::ImageSampleDMimg(const MachineInst *inst)
    : Mimg("image_sample_d", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(288, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleDMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleDClMimg::ImageSampleDClMimg(const MachineInst *inst)
    : Mimg("image_sample_d_cl", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(320, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleDClMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleLMimg::ImageSampleLMimg(const MachineInst *inst)
    : Mimg("image_sample_l", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleLMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleBMimg::ImageSampleBMimg(const MachineInst *inst)
    : Mimg("image_sample_b", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleBMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleBClMimg::ImageSampleBClMimg(const MachineInst *inst)
    : Mimg("image_sample_b_cl", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(160, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleBClMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleLzMimg::ImageSampleLzMimg(const MachineInst *inst)
    : Mimg("image_sample_lz", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(96, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleLzMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCMimg::ImageSampleCMimg(const MachineInst *inst)
    : Mimg("image_sample_c", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCClMimg::ImageSampleCClMimg(const MachineInst *inst)
    : Mimg("image_sample_c_cl", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(160, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCClMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCDMimg::ImageSampleCDMimg(const MachineInst *inst)
    : Mimg("image_sample_c_d", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(320, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCDMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCDClMimg::ImageSampleCDClMimg(const MachineInst *inst)
    : Mimg("image_sample_c_d_cl", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(352, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCDClMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCLMimg::ImageSampleCLMimg(const MachineInst *inst)
    : Mimg("image_sample_c_l", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(160, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCLMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCBMimg::ImageSampleCBMimg(const MachineInst *inst)
    : Mimg("image_sample_c_b", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(160, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCBMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCBClMimg::ImageSampleCBClMimg(const MachineInst *inst)
    : Mimg("image_sample_c_b_cl", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(192, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCBClMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCLzMimg::ImageSampleCLzMimg(const MachineInst *inst)
    : Mimg("image_sample_c_lz", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCLzMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleOMimg::ImageSampleOMimg(const MachineInst *inst)
    : Mimg("image_sample_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleOMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleClOMimg::ImageSampleClOMimg(const MachineInst *inst)
    : Mimg("image_sample_cl_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(160, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleClOMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleDOMimg::ImageSampleDOMimg(const MachineInst *inst)
    : Mimg("image_sample_d_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(320, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleDOMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleDClOMimg::ImageSampleDClOMimg(const MachineInst *inst)
    : Mimg("image_sample_d_cl_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(352, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleDClOMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleLOMimg::ImageSampleLOMimg(const MachineInst *inst)
    : Mimg("image_sample_l_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(160, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleLOMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleBOMimg::ImageSampleBOMimg(const MachineInst *inst)
    : Mimg("image_sample_b_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(160, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleBOMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleBClOMimg::ImageSampleBClOMimg(const MachineInst *inst)
    : Mimg("image_sample_b_cl_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(192, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleBClOMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleLzOMimg::ImageSampleLzOMimg(const MachineInst *inst)
    : Mimg("image_sample_lz_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleLzOMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCOMimg::ImageSampleCOMimg(const MachineInst *inst)
    : Mimg("image_sample_c_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(160, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCOMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCClOMimg::ImageSampleCClOMimg(const MachineInst *inst)
    : Mimg("image_sample_c_cl_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(192, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCClOMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCDOMimg::ImageSampleCDOMimg(const MachineInst *inst)
    : Mimg("image_sample_c_d_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(352, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCDOMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCDClOMimg::ImageSampleCDClOMimg(const MachineInst *inst)
    : Mimg("image_sample_c_d_cl_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(384, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCDClOMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCLOMimg::ImageSampleCLOMimg(const MachineInst *inst)
    : Mimg("image_sample_c_l_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(192, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCLOMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCBOMimg::ImageSampleCBOMimg(const MachineInst *inst)
    : Mimg("image_sample_c_b_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(192, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCBOMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCBClOMimg::ImageSampleCBClOMimg(const MachineInst *inst)
    : Mimg("image_sample_c_b_cl_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(224, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCBClOMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCLzOMimg::ImageSampleCLzOMimg(const MachineInst *inst)
    : Mimg("image_sample_c_lz_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(160, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCLzOMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageGather4Mimg::ImageGather4Mimg(const MachineInst *inst)
    : Mimg("image_gather4", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(96, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageGather4Mimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageGather4ClMimg::ImageGather4ClMimg(const MachineInst *inst)
    : Mimg("image_gather4_cl", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageGather4ClMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageLoadBy2Mimg::ImageLoadBy2Mimg(const MachineInst *inst)
    : Mimg("image_load_by2", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageLoadBy2Mimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_load
}

ImageLoadBy4Mimg::ImageLoadBy4Mimg(const MachineInst *inst)
    : Mimg("image_load_by4", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageLoadBy4Mimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_load
}

ImageGather4LMimg::ImageGather4LMimg(const MachineInst *inst)
    : Mimg("image_gather4_l", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageGather4LMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageGather4BMimg::ImageGather4BMimg(const MachineInst *inst)
    : Mimg("image_gather4_b", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageGather4BMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageGather4BClMimg::ImageGather4BClMimg(const MachineInst *inst)
    : Mimg("image_gather4_b_cl", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(160, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageGather4BClMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageGather4LzMimg::ImageGather4LzMimg(const MachineInst *inst)
    : Mimg("image_gather4_lz", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(96, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageGather4LzMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageGather4CMimg::ImageGather4CMimg(const MachineInst *inst)
    : Mimg("image_gather4_c", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageGather4CMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageGather4CClMimg::ImageGather4CClMimg(const MachineInst *inst)
    : Mimg("image_gather4_c_cl", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(160, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageGather4CClMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageLoadMipBy2Mimg::ImageLoadMipBy2Mimg(const MachineInst *inst)
    : Mimg("image_load_mip_by2", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageLoadMipBy2Mimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_load
}

ImageLoadMipBy4Mimg::ImageLoadMipBy4Mimg(const MachineInst *inst)
    : Mimg("image_load_mip_by4", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageLoadMipBy4Mimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_load
}

ImageGather4CLMimg::ImageGather4CLMimg(const MachineInst *inst)
    : Mimg("image_gather4_c_l", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(160, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageGather4CLMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageGather4CBMimg::ImageGather4CBMimg(const MachineInst *inst)
    : Mimg("image_gather4_c_b", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(160, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageGather4CBMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageGather4CBClMimg::ImageGather4CBClMimg(const MachineInst *inst)
    : Mimg("image_gather4_c_b_cl", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(192, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageGather4CBClMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageGather4CLzMimg::ImageGather4CLzMimg(const MachineInst *inst)
    : Mimg("image_gather4_c_lz", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageGather4CLzMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageGather4OMimg::ImageGather4OMimg(const MachineInst *inst)
    : Mimg("image_gather4_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageGather4OMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageGather4ClOMimg::ImageGather4ClOMimg(const MachineInst *inst)
    : Mimg("image_gather4_cl_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(160, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageGather4ClOMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageStoreBy2Mimg::ImageStoreBy2Mimg(const MachineInst *inst)
    : Mimg("image_store_by2", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageStoreBy2Mimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_store
}

ImageStoreBy4Mimg::ImageStoreBy4Mimg(const MachineInst *inst)
    : Mimg("image_store_by4", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageStoreBy4Mimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_store
}

ImageGather4LOMimg::ImageGather4LOMimg(const MachineInst *inst)
    : Mimg("image_gather4_l_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(160, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageGather4LOMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageGather4BOMimg::ImageGather4BOMimg(const MachineInst *inst)
    : Mimg("image_gather4_b_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(160, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageGather4BOMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageGather4BClOMimg::ImageGather4BClOMimg(const MachineInst *inst)
    : Mimg("image_gather4_b_cl_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(192, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageGather4BClOMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageGather4LzOMimg::ImageGather4LzOMimg(const MachineInst *inst)
    : Mimg("image_gather4_lz_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageGather4LzOMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageGather4COMimg::ImageGather4COMimg(const MachineInst *inst)
    : Mimg("image_gather4_c_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(160, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageGather4COMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageGather4CClOMimg::ImageGather4CClOMimg(const MachineInst *inst)
    : Mimg("image_gather4_c_cl_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(192, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageGather4CClOMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageStoreMipBy2Mimg::ImageStoreMipBy2Mimg(const MachineInst *inst)
    : Mimg("image_store_mip_by2", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageStoreMipBy2Mimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_store
}

ImageStoreMipBy4Mimg::ImageStoreMipBy4Mimg(const MachineInst *inst)
    : Mimg("image_store_mip_by4", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageStoreMipBy4Mimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_store
}

ImageGather4CLOMimg::ImageGather4CLOMimg(const MachineInst *inst)
    : Mimg("image_gather4_c_l_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(192, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageGather4CLOMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageGather4CBOMimg::ImageGather4CBOMimg(const MachineInst *inst)
    : Mimg("image_gather4_c_b_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(192, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageGather4CBOMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageGather4CBClOMimg::ImageGather4CBClOMimg(const MachineInst *inst)
    : Mimg("image_gather4_c_b_cl_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(224, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageGather4CBClOMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageGather4CLzOMimg::ImageGather4CLzOMimg(const MachineInst *inst)
    : Mimg("image_gather4_c_lz_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(160, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageGather4CLzOMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageGetLodMimg::ImageGetLodMimg(const MachineInst *inst)
    : Mimg("image_get_lod", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(96, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageGetLodMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_query
}

ImageGather4hMimg::ImageGather4hMimg(const MachineInst *inst)
    : Mimg("image_gather4h", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(96, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageGather4hMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageGather4hPckMimg::ImageGather4hPckMimg(const MachineInst *inst)
    : Mimg("image_gather4h_pck", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(96, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageGather4hPckMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageGather8hPckMimg::ImageGather8hPckMimg(const MachineInst *inst)
    : Mimg("image_gather8h_pck", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(96, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageGather8hPckMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCdMimg::ImageSampleCdMimg(const MachineInst *inst)
    : Mimg("image_sample_cd", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(288, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCdMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCdClMimg::ImageSampleCdClMimg(const MachineInst *inst)
    : Mimg("image_sample_cd_cl", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(320, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCdClMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCCdMimg::ImageSampleCCdMimg(const MachineInst *inst)
    : Mimg("image_sample_c_cd", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(320, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCCdMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCCdClMimg::ImageSampleCCdClMimg(const MachineInst *inst)
    : Mimg("image_sample_c_cd_cl", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(352, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCCdClMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCdOMimg::ImageSampleCdOMimg(const MachineInst *inst)
    : Mimg("image_sample_cd_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(320, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCdOMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCdClOMimg::ImageSampleCdClOMimg(const MachineInst *inst)
    : Mimg("image_sample_cd_cl_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(352, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCdClOMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCCdOMimg::ImageSampleCCdOMimg(const MachineInst *inst)
    : Mimg("image_sample_c_cd_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(352, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCCdOMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCCdClOMimg::ImageSampleCCdClOMimg(const MachineInst *inst)
    : Mimg("image_sample_c_cd_cl_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(384, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCCdClOMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageLoadPck2Mimg::ImageLoadPck2Mimg(const MachineInst *inst)
    : Mimg("image_load_pck2", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageLoadPck2Mimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_load
}

ImageLoadPck4Mimg::ImageLoadPck4Mimg(const MachineInst *inst)
    : Mimg("image_load_pck4", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageLoadPck4Mimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_load
}

ImageLoadMipPck2Mimg::ImageLoadMipPck2Mimg(const MachineInst *inst)
    : Mimg("image_load_mip_pck2", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageLoadMipPck2Mimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_load
}

ImageLoadMipPck4Mimg::ImageLoadMipPck4Mimg(const MachineInst *inst)
    : Mimg("image_load_mip_pck4", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageLoadMipPck4Mimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_load
}

ImageStorePck2Mimg::ImageStorePck2Mimg(const MachineInst *inst)
    : Mimg("image_store_pck2", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageStorePck2Mimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_store
}

ImageStorePck4Mimg::ImageStorePck4Mimg(const MachineInst *inst)
    : Mimg("image_store_pck4", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageStorePck4Mimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_store
}

ImageStoreMipPck2Mimg::ImageStoreMipPck2Mimg(const MachineInst *inst)
    : Mimg("image_store_mip_pck2", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageStoreMipPck2Mimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_store
}

ImageStoreMipPck4Mimg::ImageStoreMipPck4Mimg(const MachineInst *inst)
    : Mimg("image_store_mip_pck4", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageStoreMipPck4Mimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_store
}

ImageMsaaLoadMimg::ImageMsaaLoadMimg(const MachineInst *inst)
    : Mimg("image_msaa_load", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
}

void ImageMsaaLoadMimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_load
}

ImageSampleDG16Mimg::ImageSampleDG16Mimg(const MachineInst *inst)
    : Mimg("image_sample_d_g16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(224, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleDG16Mimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleDClG16Mimg::ImageSampleDClG16Mimg(const MachineInst *inst)
    : Mimg("image_sample_d_cl_g16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(256, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleDClG16Mimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCDG16Mimg::ImageSampleCDG16Mimg(const MachineInst *inst)
    : Mimg("image_sample_c_d_g16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(256, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCDG16Mimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCDClG16Mimg::ImageSampleCDClG16Mimg(const MachineInst *inst)
    : Mimg("image_sample_c_d_cl_g16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(288, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCDClG16Mimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleDOG16Mimg::ImageSampleDOG16Mimg(const MachineInst *inst)
    : Mimg("image_sample_d_o_g16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(256, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleDOG16Mimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleDClOG16Mimg::ImageSampleDClOG16Mimg(const MachineInst *inst)
    : Mimg("image_sample_d_cl_o_g16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(288, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleDClOG16Mimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCDOG16Mimg::ImageSampleCDOG16Mimg(const MachineInst *inst)
    : Mimg("image_sample_c_d_o_g16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(288, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCDOG16Mimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCDClOG16Mimg::ImageSampleCDClOG16Mimg(const MachineInst *inst)
    : Mimg("image_sample_c_d_cl_o_g16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(320, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCDClOG16Mimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCdG16Mimg::ImageSampleCdG16Mimg(const MachineInst *inst)
    : Mimg("image_sample_cd_g16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(224, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCdG16Mimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCdClG16Mimg::ImageSampleCdClG16Mimg(const MachineInst *inst)
    : Mimg("image_sample_cd_cl_g16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(256, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCdClG16Mimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCCdG16Mimg::ImageSampleCCdG16Mimg(const MachineInst *inst)
    : Mimg("image_sample_c_cd_g16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(256, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCCdG16Mimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCCdClG16Mimg::ImageSampleCCdClG16Mimg(const MachineInst *inst)
    : Mimg("image_sample_c_cd_cl_g16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(288, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCCdClG16Mimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCdOG16Mimg::ImageSampleCdOG16Mimg(const MachineInst *inst)
    : Mimg("image_sample_cd_o_g16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(256, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCdOG16Mimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCdClOG16Mimg::ImageSampleCdClOG16Mimg(const MachineInst *inst)
    : Mimg("image_sample_cd_cl_o_g16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(288, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCdClOG16Mimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCCdOG16Mimg::ImageSampleCCdOG16Mimg(const MachineInst *inst)
    : Mimg("image_sample_c_cd_o_g16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(288, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCCdOG16Mimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

ImageSampleCCdClOG16Mimg::ImageSampleCCdClOG16Mimg(const MachineInst *inst)
    : Mimg("image_sample_c_cd_cl_o_g16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(320, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(256, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      ssamp(128, OperandType::OPR_SREG_NONULL, reinterpret_cast<const OpEncoding *>(inst)->ssamp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&ssamp);
}

void ImageSampleCCdClOG16Mimg::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_sample
}

} // namespace rdna1
} // namespace rocjitsu
