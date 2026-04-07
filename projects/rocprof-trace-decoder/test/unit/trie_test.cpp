// MIT License
//
// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "trie.h"
#include <gtest/gtest.h>

class TrieTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Each test gets a fresh trie instance
    }
};

TEST_F(TrieTest, ValuInstructionsReturnVALU)
{
    EXPECT_EQ(Trie::inst_type("v_add_f32", 10), InstCategory::VALU);
    EXPECT_EQ(Trie::inst_type("v_mul_f16", 11), InstCategory::VALU);
    EXPECT_EQ(Trie::inst_type("v_cmp_eq_u32", 12), InstCategory::VALU);
    EXPECT_EQ(Trie::inst_type("v_mfma_f32_32x32x8f16", 9), InstCategory::VALU);
}

TEST_F(TrieTest, ScaleMfmaDetectedOnGfx9)
{
    EXPECT_EQ(Trie::inst_type("v_mfma_scale_f32_32x32x64_f8f6f4", 9), InstCategory::MFMA_SCALE);
    // On other GFXIPs, should still be VALU
    EXPECT_EQ(Trie::inst_type("v_mfma_scale_f32_32x32x64_f8f6f4", 10), InstCategory::VALU);
}

TEST_F(TrieTest, WaitInstructionsReturnIMMED)
{
    EXPECT_EQ(Trie::inst_type("s_waitcnt vmcnt(0)", 10), InstCategory::IMMED);
    EXPECT_EQ(Trie::inst_type("s_wait_idle", 11), InstCategory::IMMED);
    EXPECT_EQ(Trie::inst_type("s_wait_kmcnt 0", 12), InstCategory::IMMED);
    EXPECT_EQ(Trie::inst_type("s_wait_loadcnt 0", 12), InstCategory::IMMED);
    EXPECT_EQ(Trie::inst_type("s_wait_storecnt 0", 12), InstCategory::IMMED);
}

TEST_F(TrieTest, BranchInstructionsReturnBRANCH)
{
    EXPECT_EQ(Trie::inst_type("s_cbranch_scc0 -5", 10), InstCategory::BRANCH);
    EXPECT_EQ(Trie::inst_type("s_cbranch_execz 10", 11), InstCategory::BRANCH);
    EXPECT_EQ(Trie::inst_type("s_branch 100", 9), InstCategory::BRANCH);
}

TEST_F(TrieTest, SmemInstructionsReturnSMEM)
{
    EXPECT_EQ(Trie::inst_type("s_load_dwordx4 s[0:3], s[4:5], 0x0", 10), InstCategory::SMEM);
    EXPECT_EQ(Trie::inst_type("s_buffer_load_dword", 11), InstCategory::SMEM);
    EXPECT_EQ(Trie::inst_type("s_atomic_add", 12), InstCategory::SMEM);
}

TEST_F(TrieTest, VmemInstructionsReturnVMEM)
{
    EXPECT_EQ(Trie::inst_type("buffer_load_dword v0, v1, s[0:3], 0", 10), InstCategory::VMEM);
    EXPECT_EQ(Trie::inst_type("global_load_dword v0, v[1:2], off", 11), InstCategory::VMEM);
    EXPECT_EQ(Trie::inst_type("scratch_load_dword v0, v1, off", 12), InstCategory::VMEM);
}

TEST_F(TrieTest, LdsInstructionsReturnLDS)
{
    EXPECT_EQ(Trie::inst_type("ds_read_b32 v0, v1", 10), InstCategory::LDS);
    EXPECT_EQ(Trie::inst_type("ds_write_b64 v0, v[1:2]", 11), InstCategory::LDS);
}

TEST_F(TrieTest, FlatInstructionsReturnFLAT)
{
    EXPECT_EQ(Trie::inst_type("flat_load_dword v0, v[1:2]", 10), InstCategory::FLAT);
    EXPECT_EQ(Trie::inst_type("flat_store_dword v[0:1], v2", 11), InstCategory::FLAT);
}

TEST_F(TrieTest, SaluInstructionsReturnSALU)
{
    EXPECT_EQ(Trie::inst_type("s_add_u32 s0, s1, s2", 10), InstCategory::SALU);
    EXPECT_EQ(Trie::inst_type("s_mov_b32 s0, 0", 11), InstCategory::SALU);
    EXPECT_EQ(Trie::inst_type("s_and_b64 s[0:1], s[2:3], s[4:5]", 12), InstCategory::SALU);
    EXPECT_EQ(Trie::inst_type("s_cmp_eq_u32 s0, s1", 9), InstCategory::SALU);
}

TEST_F(TrieTest, SkipInstructionsReturnSKIP)
{
    EXPECT_EQ(Trie::inst_type("s_wait_dep 0", 12), InstCategory::SKIP);
    EXPECT_EQ(Trie::inst_type("s_wait_alu 0x0", 12), InstCategory::SKIP);
    EXPECT_EQ(Trie::inst_type("s_delay_alu", 11), InstCategory::SKIP);
}

TEST_F(TrieTest, BvhInstructionsReturnBVH)
{
    EXPECT_EQ(Trie::inst_type("image_bvh_intersect_ray", 10), InstCategory::BVH);
    EXPECT_EQ(Trie::inst_type("image_bvh64_intersect_ray", 11), InstCategory::BVH);
}

TEST_F(TrieTest, UnknownSaluFallsbackToSALU)
{
    // Unknown s_ prefix instructions should fall back to SALU
    EXPECT_EQ(Trie::inst_type("s_unknown_instruction", 10), InstCategory::SALU);
}

TEST_F(TrieTest, ControlFlowInstructionsReturnIMMED)
{
    EXPECT_EQ(Trie::inst_type("s_endpgm", 10), InstCategory::IMMED);
    EXPECT_EQ(Trie::inst_type("s_trap 0x1", 11), InstCategory::IMMED);
    EXPECT_EQ(Trie::inst_type("s_nop 0", 12), InstCategory::IMMED);
    EXPECT_EQ(Trie::inst_type("s_sleep 10", 9), InstCategory::IMMED);
}

// Edge case: empty string
TEST_F(TrieTest, EmptyStringReturnsDontKnow) { EXPECT_EQ(Trie::inst_type("", 10), InstCategory::DONT_KNOW); }

// Edge case: instruction with only whitespace
TEST_F(TrieTest, WhitespaceOnlyReturnsDontKnow) { EXPECT_EQ(Trie::inst_type("   ", 10), InstCategory::DONT_KNOW); }

// Edge case: completely unrecognized instruction
TEST_F(TrieTest, UnrecognizedInstructionReturnsDontKnow)
{
    EXPECT_EQ(Trie::inst_type("xyz_unknown_instruction", 10), InstCategory::DONT_KNOW);
}

// Edge case: image instructions
TEST_F(TrieTest, ImageInstructionsReturnCorrectCategory)
{
    EXPECT_EQ(Trie::inst_type("image_load v0, v[1:2], s[0:7]", 10), InstCategory::VMEM);
    EXPECT_EQ(Trie::inst_type("image_store v0, v[1:2], s[0:7]", 11), InstCategory::VMEM);
    EXPECT_EQ(Trie::inst_type("image_atomic_add v0, v[1:2], s[0:7]", 12), InstCategory::VMEM);
    EXPECT_EQ(Trie::inst_type("image_gather4 v[0:3], v[1:2], s[0:7]", 10), InstCategory::VMEM);
    EXPECT_EQ(Trie::inst_type("image_sample v0, v[1:2], s[0:7], s[8:11]", 11), InstCategory::VMEM);
}

// MFMA scale on non-gfx9 returns VALU
TEST_F(TrieTest, MfmaScaleOnNonGfx9ReturnsValu)
{
    EXPECT_EQ(Trie::inst_type("v_mfma_scale_f32_32x32x64_f8f6f4", 11), InstCategory::VALU);
    EXPECT_EQ(Trie::inst_type("v_mfma_scale_f32_32x32x64_f8f6f4", 12), InstCategory::VALU);
}

// Branch detection is case sensitive and works with operands
TEST_F(TrieTest, BranchDetectionInVariousContexts)
{
    EXPECT_EQ(Trie::inst_type("s_cbranch_vccz 0x10", 10), InstCategory::BRANCH);
    EXPECT_EQ(Trie::inst_type("s_cbranch_vccnz label", 11), InstCategory::BRANCH);
    EXPECT_EQ(Trie::inst_type("s_cbranch_cdbgsys", 12), InstCategory::BRANCH);
}
