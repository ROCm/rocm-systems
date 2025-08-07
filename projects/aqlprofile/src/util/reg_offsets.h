// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef SRC_UTIL_AQLPROFILE_REG_OFFSET_H_
#define SRC_UTIL_AQLPROFILE_REG_OFFSET_H_

#include <iostream>
#include <array>
#include <vector>

#include "util/hsa_rsrc_factory.h"

/* Define the HW IP blocks will be used in driver , add more if necessary */
enum amd_hw_ip_block_type {
  GC_HWIP = 1,
  HDP_HWIP,
  SDMA0_HWIP,
  SDMA1_HWIP,
  SDMA2_HWIP,
  SDMA3_HWIP,
  SDMA4_HWIP,
  SDMA5_HWIP,
  SDMA6_HWIP,
  SDMA7_HWIP,
  LSDMA_HWIP,
  MMHUB_HWIP,
  ATHUB_HWIP,
  NBIO_HWIP,
  MP0_HWIP,
  MP1_HWIP,
  UVD_HWIP,
  VCN_HWIP = UVD_HWIP,
  JPEG_HWIP = VCN_HWIP,
  VCN1_HWIP,
  VCE_HWIP,
  VPE_HWIP,
  DF_HWIP,
  DCE_HWIP,
  OSSSYS_HWIP,
  SMUIO_HWIP,
  PWR_HWIP,
  NBIF_HWIP,
  THM_HWIP,
  CLK_HWIP,
  UMC_HWIP,
  RSMU_HWIP,
  XGMI_HWIP,
  DCI_HWIP,
  PCIE_HWIP,
  ISP_HWIP,
  MAX_HWIP
};

#define HWIP_MAX_INSTANCE 44
#define HWIP_MAX_SEGMENT 32

struct Register {
  uint32_t hwip;
  uint32_t ip_inst;
  uint32_t offset;
  uint32_t base_idx;

  explicit constexpr Register()
      : hwip(MAX_HWIP), ip_inst(HWIP_MAX_INSTANCE), offset(0), base_idx(HWIP_MAX_SEGMENT) {}

  explicit constexpr Register(uint32_t value)
      : hwip(MAX_HWIP), ip_inst(HWIP_MAX_INSTANCE), offset(value), base_idx(HWIP_MAX_SEGMENT) {}

  explicit constexpr Register(uint32_t hwip_val, uint32_t ip_inst_val, uint32_t offset_val,
                              uint32_t base_idx_val)
      : hwip(hwip_val), ip_inst(ip_inst_val), offset(offset_val), base_idx(base_idx_val) {}
};

struct reg_base_offset_table {
  using segment_array_t = std::array<uint32_t, HWIP_MAX_SEGMENT>;
  using instance_array_t = std::array<segment_array_t, HWIP_MAX_INSTANCE>;
  using hwip_array_t = std::array<instance_array_t, MAX_HWIP>;

  hwip_array_t reg_offset{};

  reg_base_offset_table() {}

  uint32_t operator[](const Register& reg) const {
    try {
      return reg_offset.at(reg.hwip).at(reg.ip_inst).at(reg.base_idx) + reg.offset;
    } catch (const std::out_of_range& e) {
      std::cerr << "Invalid Register used: hwip=" + std::to_string(reg.hwip) +
                       ", ip_inst=" + std::to_string(reg.ip_inst) +
                       ", base_idx=" + std::to_string(reg.base_idx) + ". Exception: " + e.what()
                << std::endl;
      std::abort();
    }
  }
};

const reg_base_offset_table* acquire_ip_offset_table(const AgentInfo* agent_info);

#endif