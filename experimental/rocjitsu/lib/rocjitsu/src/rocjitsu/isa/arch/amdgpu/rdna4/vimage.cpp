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
  // Minimal image load stub — Phase C placeholder.
  (void)wf;
}

ImageLoadMipVimage::ImageLoadMipVimage(const MachineInst *inst)
    : Vimage("image_load_mip", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageLoadMipVimage::execute(amdgpu::Wavefront &wf) {
  // Minimal image load stub — Phase C placeholder.
  (void)wf;
}

ImageLoadPckVimage::ImageLoadPckVimage(const MachineInst *inst)
    : Vimage("image_load_pck", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageLoadPckVimage::execute(amdgpu::Wavefront &wf) {
  // Minimal image load stub — Phase C placeholder.
  (void)wf;
}

ImageLoadPckSgnVimage::ImageLoadPckSgnVimage(const MachineInst *inst)
    : Vimage("image_load_pck_sgn", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageLoadPckSgnVimage::execute(amdgpu::Wavefront &wf) {
  // Minimal image load stub — Phase C placeholder.
  (void)wf;
}

ImageLoadMipPckVimage::ImageLoadMipPckVimage(const MachineInst *inst)
    : Vimage("image_load_mip_pck", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageLoadMipPckVimage::execute(amdgpu::Wavefront &wf) {
  // Minimal image load stub — Phase C placeholder.
  (void)wf;
}

ImageLoadMipPckSgnVimage::ImageLoadMipPckSgnVimage(const MachineInst *inst)
    : Vimage("image_load_mip_pck_sgn", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageLoadMipPckSgnVimage::execute(amdgpu::Wavefront &wf) {
  // Minimal image load stub — Phase C placeholder.
  (void)wf;
}

ImageStoreVimage::ImageStoreVimage(const MachineInst *inst)
    : Vimage("image_store", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageStoreVimage::execute(amdgpu::Wavefront &wf) {
  // Minimal image store stub — Phase C placeholder.
  (void)wf;
}

ImageStoreMipVimage::ImageStoreMipVimage(const MachineInst *inst)
    : Vimage("image_store_mip", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageStoreMipVimage::execute(amdgpu::Wavefront &wf) {
  // Minimal image store stub — Phase C placeholder.
  (void)wf;
}

ImageStorePckVimage::ImageStorePckVimage(const MachineInst *inst)
    : Vimage("image_store_pck", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageStorePckVimage::execute(amdgpu::Wavefront &wf) {
  // Minimal image store stub — Phase C placeholder.
  (void)wf;
}

ImageStoreMipPckVimage::ImageStoreMipPckVimage(const MachineInst *inst)
    : Vimage("image_store_mip_pck", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageStoreMipPckVimage::execute(amdgpu::Wavefront &wf) {
  // Minimal image store stub — Phase C placeholder.
  (void)wf;
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
  (void)wf; // Deferred to Phase E (image pipeline).
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
  (void)wf; // Deferred to Phase E (image pipeline).
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
  (void)wf; // Deferred to Phase E (image pipeline).
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
  (void)wf; // Deferred to Phase E (image pipeline).
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
  (void)wf; // Deferred to Phase E (image pipeline).
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
  (void)wf; // Deferred to Phase E (image pipeline).
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
  (void)wf; // Deferred to Phase E (image pipeline).
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
  (void)wf; // Deferred to Phase E (image pipeline).
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
  (void)wf; // Deferred to Phase E (image pipeline).
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
  (void)wf; // Deferred to Phase E (image pipeline).
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
  (void)wf; // Deferred to Phase E (image pipeline).
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
  (void)wf; // Deferred to Phase E (image pipeline).
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
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageGetResinfoVimage::ImageGetResinfoVimage(const MachineInst *inst)
    : Vimage("image_get_resinfo", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageGetResinfoVimage::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageBvhIntersectRayVimage::ImageBvhIntersectRayVimage(const MachineInst *inst)
    : Vimage("image_bvh_intersect_ray", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageBvhIntersectRayVimage::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
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
  (void)wf; // Deferred to Phase E (image pipeline).
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
  (void)wf; // Deferred to Phase E (image pipeline).
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
  (void)wf; // Deferred to Phase E (image pipeline).
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
  (void)wf; // Deferred to Phase E (image pipeline).
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
  (void)wf; // Deferred to Phase E (image pipeline).
}

} // namespace rdna4
} // namespace rocjitsu
