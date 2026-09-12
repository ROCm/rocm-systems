// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "lib/aqlprofile/aqlprofile.hpp"
#include "lib/aqlprofile/def/gpu_block_info.h"
#include "lib/aqlprofile/def/gfx12_def.h"

namespace
{
using gfxip::gfx12::gfx1200::gfx12_cntx_prim;

GpuBlockInfo
make_block_info(uint32_t attr)
{
    GpuBlockInfo info{nullptr,
                      0,
                      0,
                      0,
                      0,
                      nullptr,
                      nullptr,
                      attr,
                      BLOCK_DELAY_NONE,
                      0,
                      0};
    info.attr = attr;
    return info;
}
}  // namespace

TEST(SpmGfx12HelpersTest, EncodeDecodeWgpInstance)
{
    const auto block_info = make_block_info(CounterBlockSaAttr | CounterBlockWgpAttr);

    const uint32_t encoded0 = gfx12_cntx_prim::encode_spm_block_index(/*inst_index=*/0,
                                                                       /*sa_index=*/3,
                                                                       /*wgp_index=*/1);
    const uint32_t encoded1 = gfx12_cntx_prim::encode_spm_block_index(/*inst_index=*/1,
                                                                       /*sa_index=*/3,
                                                                       /*wgp_index=*/1);

    EXPECT_EQ(gfx12_cntx_prim::decode_spm_instance_index(&block_info, encoded0), 0u);
    EXPECT_EQ(gfx12_cntx_prim::decode_spm_instance_index(&block_info, encoded1), 1u);
}

TEST(SpmGfx12HelpersTest, EncodeDecodeSaOnlyInstance)
{
    const auto block_info = make_block_info(CounterBlockSaAttr);

    const uint32_t encoded = gfx12_cntx_prim::encode_spm_block_index(/*inst_index=*/17,
                                                                      /*sa_index=*/1,
                                                                      /*wgp_index=*/0);

    EXPECT_EQ(gfx12_cntx_prim::decode_spm_instance_index(&block_info, encoded), 17u);
}

TEST(SpmGfx12HelpersTest, WgpDecodeTakesPriorityOverSaAttr)
{
    const auto block_info = make_block_info(CounterBlockSaAttr | CounterBlockWgpAttr);

    const uint32_t encoded = gfx12_cntx_prim::encode_spm_block_index(/*inst_index=*/1,
                                                                      /*sa_index=*/1,
                                                                      /*wgp_index=*/0);

    EXPECT_EQ(gfx12_cntx_prim::decode_spm_instance_index(&block_info, encoded), 1u);
}
