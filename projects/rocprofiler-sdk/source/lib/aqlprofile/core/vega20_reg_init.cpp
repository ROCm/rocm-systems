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

#include <cstdint>
#include <cstring>
#include <mutex>
#include "linux/registers/vega20_ip_offset.h"
#include "util/reg_offsets.h"
#include "util/soc15_common.h"

const reg_base_offset_table* vega20_reg_base_init() {
  static_assert(HWIP_MAX_INSTANCE >= MAX_INSTANCE,
                "HWIP_MAX_INSTANCE must be greater than MAX_INSTANCE");
  static_assert(HWIP_MAX_SEGMENT >= MAX_SEGMENT,
                "HWIP_MAX_SEGMENT must be greater than MAX_SEGMENT");

  static const auto* vega20_reg_table = []() {
    auto* reg_table = new reg_base_offset_table();

    // helper lambda to initialize blocks
    auto init_hwip = [&](amd_hw_ip_block_type hwip, const auto& base) {
      for (uint32_t i = 0; i < MAX_INSTANCE; i++) {
        std::copy(std::begin(base.instance[i].segment), std::end(base.instance[i].segment),
                  std::begin(reg_table->reg_offset[hwip][i]));
      }
    };

    // Initialize all HWIP blocks
    init_hwip(GC_HWIP, GC_BASE);
    init_hwip(HDP_HWIP, HDP_BASE);
    init_hwip(MMHUB_HWIP, MMHUB_BASE);
    init_hwip(ATHUB_HWIP, ATHUB_BASE);
    init_hwip(NBIO_HWIP, NBIO_BASE);
    init_hwip(MP0_HWIP, MP0_BASE);
    init_hwip(MP1_HWIP, MP1_BASE);
    init_hwip(UVD_HWIP, UVD_BASE);
    init_hwip(VCE_HWIP, VCE_BASE);
    init_hwip(DF_HWIP, DF_BASE);
    init_hwip(DCE_HWIP, DCE_BASE);
    init_hwip(OSSSYS_HWIP, OSSSYS_BASE);
    init_hwip(SDMA0_HWIP, SDMA0_BASE);
    init_hwip(SDMA1_HWIP, SDMA1_BASE);
    init_hwip(SMUIO_HWIP, SMUIO_BASE);
    init_hwip(NBIF_HWIP, NBIO_BASE);
    init_hwip(THM_HWIP, THM_BASE);
    init_hwip(CLK_HWIP, CLK_BASE);
    init_hwip(UMC_HWIP, UMC_BASE);
    init_hwip(RSMU_HWIP, RSMU_BASE);

    return reg_table;
  }();

  return vega20_reg_table;
}
