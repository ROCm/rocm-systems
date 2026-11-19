// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/kernel_descriptor_rsrc3.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>

namespace rocjitsu {
namespace {

// The field masks are written out from the compute_pgm_rsrc3 tables in
// AMDGPUUsage rather than taken from the ABI header, so a wrong width or shift
// fails here: GFX11 holds INST_PREF_SIZE in 6 bits at 9:4, GFX12 and later in 8
// bits at 11:4.
TEST(KernelDescriptorRsrc3, InstPrefSizeUsesEachLayoutsField) {
  struct Case {
    rj_code_arch_t arch;
    uint32_t max_lines;
    uint32_t field_mask;
  };
  const Case cases[] = {
      {ROCJITSU_CODE_ARCH_RDNA3, 63, 0x3f0u},
      {ROCJITSU_CODE_ARCH_RDNA3_5, 63, 0x3f0u},
      {ROCJITSU_CODE_ARCH_RDNA4, 255, 0xff0u},
      {ROCJITSU_CODE_ARCH_CDNA5, 255, 0xff0u},
  };
  for (const Case &c : cases) {
    SCOPED_TRACE(c.arch);
    uint32_t rsrc3 = 0xffffffffu;
    set_inst_pref_size(rsrc3, c.arch, 0);
    EXPECT_EQ(rsrc3, ~c.field_mask) << "only the field's bits change";
    set_inst_pref_size(rsrc3, c.arch, c.max_lines);
    EXPECT_EQ(rsrc3, 0xffffffffu);
    EXPECT_EQ(inst_pref_size(rsrc3, c.arch), c.max_lines);
    EXPECT_EQ(inst_pref_size(5u << 4, c.arch), 5u);
  }

  // CDNA encodes ACCUM_OFFSET in these bits, and GFX10 reserves them.
  for (const rj_code_arch_t arch :
       {ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3,
        ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2}) {
    SCOPED_TRACE(arch);
    EXPECT_EQ(inst_pref_size(0xffffffffu, arch), std::nullopt);
  }
}

} // namespace
} // namespace rocjitsu
