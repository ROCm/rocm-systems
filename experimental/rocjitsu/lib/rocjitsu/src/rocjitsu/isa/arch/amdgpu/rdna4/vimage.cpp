// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/rdna4/vimage.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/except.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace rocjitsu {
namespace rdna4 {

ImageLoadVimage::ImageLoadVimage(const MachineInst *inst)
    : Vimage("image_load", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageLoadVimage::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_load
}

ImageLoadMipVimage::ImageLoadMipVimage(const MachineInst *inst)
    : Vimage("image_load_mip", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageLoadMipVimage::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_load
}

ImageLoadPckVimage::ImageLoadPckVimage(const MachineInst *inst)
    : Vimage("image_load_pck", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageLoadPckVimage::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_load
}

ImageLoadPckSgnVimage::ImageLoadPckSgnVimage(const MachineInst *inst)
    : Vimage("image_load_pck_sgn", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageLoadPckSgnVimage::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_load
}

ImageLoadMipPckVimage::ImageLoadMipPckVimage(const MachineInst *inst)
    : Vimage("image_load_mip_pck", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageLoadMipPckVimage::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_load
}

ImageLoadMipPckSgnVimage::ImageLoadMipPckSgnVimage(const MachineInst *inst)
    : Vimage("image_load_mip_pck_sgn", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageLoadMipPckSgnVimage::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_load
}

ImageStoreVimage::ImageStoreVimage(const MachineInst *inst)
    : Vimage("image_store", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageStoreVimage::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_store
}

ImageStoreMipVimage::ImageStoreMipVimage(const MachineInst *inst)
    : Vimage("image_store_mip", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageStoreMipVimage::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_store
}

ImageStorePckVimage::ImageStorePckVimage(const MachineInst *inst)
    : Vimage("image_store_pck", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageStorePckVimage::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_store
}

ImageStoreMipPckVimage::ImageStoreMipPckVimage(const MachineInst *inst)
    : Vimage("image_store_mip_pck", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageStoreMipPckVimage::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_store
}

ImageAtomicSwapVimage::ImageAtomicSwapVimage(const MachineInst *inst)
    : Vimage("image_atomic_swap", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageAtomicSwapVimage::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_atomic
}

ImageAtomicCmpswapVimage::ImageAtomicCmpswapVimage(const MachineInst *inst)
    : Vimage("image_atomic_cmpswap", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageAtomicCmpswapVimage::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_atomic
}

ImageAtomicAddUintVimage::ImageAtomicAddUintVimage(const MachineInst *inst)
    : Vimage("image_atomic_add_uint", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageAtomicAddUintVimage::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_atomic
}

ImageAtomicSubUintVimage::ImageAtomicSubUintVimage(const MachineInst *inst)
    : Vimage("image_atomic_sub_uint", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageAtomicSubUintVimage::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_atomic
}

ImageAtomicMinIntVimage::ImageAtomicMinIntVimage(const MachineInst *inst)
    : Vimage("image_atomic_min_int", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageAtomicMinIntVimage::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_atomic
}

ImageAtomicMinUintVimage::ImageAtomicMinUintVimage(const MachineInst *inst)
    : Vimage("image_atomic_min_uint", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageAtomicMinUintVimage::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_atomic
}

ImageAtomicMaxIntVimage::ImageAtomicMaxIntVimage(const MachineInst *inst)
    : Vimage("image_atomic_max_int", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageAtomicMaxIntVimage::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_atomic
}

ImageAtomicMaxUintVimage::ImageAtomicMaxUintVimage(const MachineInst *inst)
    : Vimage("image_atomic_max_uint", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageAtomicMaxUintVimage::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_atomic
}

ImageAtomicAndVimage::ImageAtomicAndVimage(const MachineInst *inst)
    : Vimage("image_atomic_and", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageAtomicAndVimage::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_atomic
}

ImageAtomicOrVimage::ImageAtomicOrVimage(const MachineInst *inst)
    : Vimage("image_atomic_or", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageAtomicOrVimage::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_atomic
}

ImageAtomicXorVimage::ImageAtomicXorVimage(const MachineInst *inst)
    : Vimage("image_atomic_xor", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageAtomicXorVimage::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_atomic
}

ImageAtomicIncUintVimage::ImageAtomicIncUintVimage(const MachineInst *inst)
    : Vimage("image_atomic_inc_uint", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageAtomicIncUintVimage::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_atomic
}

ImageAtomicDecUintVimage::ImageAtomicDecUintVimage(const MachineInst *inst)
    : Vimage("image_atomic_dec_uint", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageAtomicDecUintVimage::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_atomic
}

ImageGetResinfoVimage::ImageGetResinfoVimage(const MachineInst *inst)
    : Vimage("image_get_resinfo", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageGetResinfoVimage::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_query
}

ImageBvhIntersectRayVimage::ImageBvhIntersectRayVimage(const MachineInst *inst)
    : Vimage("image_bvh_intersect_ray", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageBvhIntersectRayVimage::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_bvh
}

ImageBvh64IntersectRayVimage::ImageBvh64IntersectRayVimage(const MachineInst *inst)
    : Vimage("image_bvh64_intersect_ray", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageBvh64IntersectRayVimage::execute(amdgpu::Wavefront &wf) { (void)wf; }

ImageBvhDualIntersectRayVimage::ImageBvhDualIntersectRayVimage(const MachineInst *inst)
    : Vimage("image_bvh_dual_intersect_ray", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(320, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageBvhDualIntersectRayVimage::execute(amdgpu::Wavefront &wf) { (void)wf; }

ImageBvh8IntersectRayVimage::ImageBvh8IntersectRayVimage(const MachineInst *inst)
    : Vimage("image_bvh8_intersect_ray", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(320, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageBvh8IntersectRayVimage::execute(amdgpu::Wavefront &wf) { (void)wf; }

ImageAtomicAddFltVimage::ImageAtomicAddFltVimage(const MachineInst *inst)
    : Vimage("image_atomic_add_flt", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageAtomicAddFltVimage::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_atomic
}

ImageAtomicMinFltVimage::ImageAtomicMinFltVimage(const MachineInst *inst)
    : Vimage("image_atomic_min_flt", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageAtomicMinFltVimage::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_atomic
}

ImageAtomicMaxFltVimage::ImageAtomicMaxFltVimage(const MachineInst *inst)
    : Vimage("image_atomic_max_flt", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageAtomicMaxFltVimage::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_atomic
}

ImageAtomicPkAddF16Vimage::ImageAtomicPkAddF16Vimage(const MachineInst *inst)
    : Vimage("image_atomic_pk_add_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageAtomicPkAddF16Vimage::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_atomic
}

ImageAtomicPkAddBf16Vimage::ImageAtomicPkAddBf16Vimage(const MachineInst *inst)
    : Vimage("image_atomic_pk_add_bf16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageAtomicPkAddBf16Vimage::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: image_atomic
}

} // namespace rdna4
} // namespace rocjitsu
