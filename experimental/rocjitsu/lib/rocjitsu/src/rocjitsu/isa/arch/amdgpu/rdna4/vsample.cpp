// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/rdna4/vsample.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/except.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace rocjitsu {
namespace rdna4 {

ImageMsaaLoadVsample::ImageMsaaLoadVsample(const MachineInst *inst)
    : Vsample("image_msaa_load", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
}

void ImageMsaaLoadVsample::execute(amdgpu::Wavefront &wf) {
  // Minimal image load stub — Phase C placeholder.
  (void)wf;
}

ImageSampleVsample::ImageSampleVsample(const MachineInst *inst)
    : Vsample("image_sample", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleDVsample::ImageSampleDVsample(const MachineInst *inst)
    : Vsample("image_sample_d", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleDVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleLVsample::ImageSampleLVsample(const MachineInst *inst)
    : Vsample("image_sample_l", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleLVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleBVsample::ImageSampleBVsample(const MachineInst *inst)
    : Vsample("image_sample_b", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleBVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleLzVsample::ImageSampleLzVsample(const MachineInst *inst)
    : Vsample("image_sample_lz", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleLzVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleCVsample::ImageSampleCVsample(const MachineInst *inst)
    : Vsample("image_sample_c", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleCVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleCDVsample::ImageSampleCDVsample(const MachineInst *inst)
    : Vsample("image_sample_c_d", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleCDVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleCLVsample::ImageSampleCLVsample(const MachineInst *inst)
    : Vsample("image_sample_c_l", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleCLVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleCBVsample::ImageSampleCBVsample(const MachineInst *inst)
    : Vsample("image_sample_c_b", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleCBVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleCLzVsample::ImageSampleCLzVsample(const MachineInst *inst)
    : Vsample("image_sample_c_lz", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleCLzVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleOVsample::ImageSampleOVsample(const MachineInst *inst)
    : Vsample("image_sample_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleOVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleDOVsample::ImageSampleDOVsample(const MachineInst *inst)
    : Vsample("image_sample_d_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleDOVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleLOVsample::ImageSampleLOVsample(const MachineInst *inst)
    : Vsample("image_sample_l_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleLOVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleBOVsample::ImageSampleBOVsample(const MachineInst *inst)
    : Vsample("image_sample_b_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleBOVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleLzOVsample::ImageSampleLzOVsample(const MachineInst *inst)
    : Vsample("image_sample_lz_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleLzOVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleCOVsample::ImageSampleCOVsample(const MachineInst *inst)
    : Vsample("image_sample_c_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleCOVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleCDOVsample::ImageSampleCDOVsample(const MachineInst *inst)
    : Vsample("image_sample_c_d_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleCDOVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleCLOVsample::ImageSampleCLOVsample(const MachineInst *inst)
    : Vsample("image_sample_c_l_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleCLOVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleCBOVsample::ImageSampleCBOVsample(const MachineInst *inst)
    : Vsample("image_sample_c_b_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleCBOVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleCLzOVsample::ImageSampleCLzOVsample(const MachineInst *inst)
    : Vsample("image_sample_c_lz_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleCLzOVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageGather4Vsample::ImageGather4Vsample(const MachineInst *inst)
    : Vsample("image_gather4", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageGather4Vsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageGather4LVsample::ImageGather4LVsample(const MachineInst *inst)
    : Vsample("image_gather4_l", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageGather4LVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageGather4BVsample::ImageGather4BVsample(const MachineInst *inst)
    : Vsample("image_gather4_b", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageGather4BVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageGather4LzVsample::ImageGather4LzVsample(const MachineInst *inst)
    : Vsample("image_gather4_lz", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageGather4LzVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageGather4CVsample::ImageGather4CVsample(const MachineInst *inst)
    : Vsample("image_gather4_c", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageGather4CVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageGather4CLzVsample::ImageGather4CLzVsample(const MachineInst *inst)
    : Vsample("image_gather4_c_lz", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageGather4CLzVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageGather4OVsample::ImageGather4OVsample(const MachineInst *inst)
    : Vsample("image_gather4_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageGather4OVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageGather4LzOVsample::ImageGather4LzOVsample(const MachineInst *inst)
    : Vsample("image_gather4_lz_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageGather4LzOVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageGather4CLzOVsample::ImageGather4CLzOVsample(const MachineInst *inst)
    : Vsample("image_gather4_c_lz_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageGather4CLzOVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageGetLodVsample::ImageGetLodVsample(const MachineInst *inst)
    : Vsample("image_get_lod", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageGetLodVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleDG16Vsample::ImageSampleDG16Vsample(const MachineInst *inst)
    : Vsample("image_sample_d_g16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleDG16Vsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleCDG16Vsample::ImageSampleCDG16Vsample(const MachineInst *inst)
    : Vsample("image_sample_c_d_g16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleCDG16Vsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleDOG16Vsample::ImageSampleDOG16Vsample(const MachineInst *inst)
    : Vsample("image_sample_d_o_g16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleDOG16Vsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleCDOG16Vsample::ImageSampleCDOG16Vsample(const MachineInst *inst)
    : Vsample("image_sample_c_d_o_g16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleCDOG16Vsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleClVsample::ImageSampleClVsample(const MachineInst *inst)
    : Vsample("image_sample_cl", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleClVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleDClVsample::ImageSampleDClVsample(const MachineInst *inst)
    : Vsample("image_sample_d_cl", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleDClVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleBClVsample::ImageSampleBClVsample(const MachineInst *inst)
    : Vsample("image_sample_b_cl", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleBClVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleCClVsample::ImageSampleCClVsample(const MachineInst *inst)
    : Vsample("image_sample_c_cl", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleCClVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleCDClVsample::ImageSampleCDClVsample(const MachineInst *inst)
    : Vsample("image_sample_c_d_cl", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleCDClVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleCBClVsample::ImageSampleCBClVsample(const MachineInst *inst)
    : Vsample("image_sample_c_b_cl", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleCBClVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleClOVsample::ImageSampleClOVsample(const MachineInst *inst)
    : Vsample("image_sample_cl_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleClOVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleDClOVsample::ImageSampleDClOVsample(const MachineInst *inst)
    : Vsample("image_sample_d_cl_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleDClOVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleBClOVsample::ImageSampleBClOVsample(const MachineInst *inst)
    : Vsample("image_sample_b_cl_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleBClOVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleCClOVsample::ImageSampleCClOVsample(const MachineInst *inst)
    : Vsample("image_sample_c_cl_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleCClOVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleCDClOVsample::ImageSampleCDClOVsample(const MachineInst *inst)
    : Vsample("image_sample_c_d_cl_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleCDClOVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleCBClOVsample::ImageSampleCBClOVsample(const MachineInst *inst)
    : Vsample("image_sample_c_b_cl_o", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleCBClOVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleCDClG16Vsample::ImageSampleCDClG16Vsample(const MachineInst *inst)
    : Vsample("image_sample_c_d_cl_g16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleCDClG16Vsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleDClOG16Vsample::ImageSampleDClOG16Vsample(const MachineInst *inst)
    : Vsample("image_sample_d_cl_o_g16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleDClOG16Vsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleCDClOG16Vsample::ImageSampleCDClOG16Vsample(const MachineInst *inst)
    : Vsample("image_sample_c_d_cl_o_g16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleCDClOG16Vsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageSampleDClG16Vsample::ImageSampleDClG16Vsample(const MachineInst *inst)
    : Vsample("image_sample_d_cl_g16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageSampleDClG16Vsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageGather4ClVsample::ImageGather4ClVsample(const MachineInst *inst)
    : Vsample("image_gather4_cl", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageGather4ClVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageGather4BClVsample::ImageGather4BClVsample(const MachineInst *inst)
    : Vsample("image_gather4_b_cl", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageGather4BClVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageGather4CClVsample::ImageGather4CClVsample(const MachineInst *inst)
    : Vsample("image_gather4_c_cl", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageGather4CClVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageGather4CLVsample::ImageGather4CLVsample(const MachineInst *inst)
    : Vsample("image_gather4_c_l", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageGather4CLVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageGather4CBVsample::ImageGather4CBVsample(const MachineInst *inst)
    : Vsample("image_gather4_c_b", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageGather4CBVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageGather4CBClVsample::ImageGather4CBClVsample(const MachineInst *inst)
    : Vsample("image_gather4_c_b_cl", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageGather4CBClVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

ImageGather4hVsample::ImageGather4hVsample(const MachineInst *inst)
    : Vsample("image_gather4h", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      rsrc(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      samp(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->samp) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&samp);
}

void ImageGather4hVsample::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Deferred to Phase E (image pipeline).
}

} // namespace rdna4
} // namespace rocjitsu
