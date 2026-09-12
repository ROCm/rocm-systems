// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "GoogleTestExtension.hpp"

#include "../../../src/libhsakmt.h"

class QueueContextSizingTest : public testing::Test {
 protected:
    QueueContextSizingTest() {
        node.NumShaderBanks = 4;
        node.NumArrays = 2;
        node.NumSIMDPerCU = 4;
        node.MaxWavesPerSIMD = 10;
    }

    HsaNodeProperties node = {};
};

TEST_F(QueueContextSizingTest, PreNaviUsesFortyWavesPerCu) {
    TEST_START(TESTPROFILE_RUNALL)

    EXPECT_EQ(800u, hsakmt_get_num_waves(&node, GFX_VERSION_NAVI10 - 1, 20));

    TEST_END
}

TEST_F(QueueContextSizingTest, PreNaviCapsWavesPerShaderEngine) {
    TEST_START(TESTPROFILE_RUNALL)

    EXPECT_EQ(1024u, hsakmt_get_num_waves(&node, GFX_VERSION_NAVI10 - 1, 64));

    TEST_END
}

TEST_F(QueueContextSizingTest, Navi14MatchesKfdWaveCount) {
    TEST_START(TESTPROFILE_RUNALL)

    EXPECT_EQ(640u, hsakmt_get_num_waves(&node, GFX_VERSION_NAVI14, 20));

    TEST_END
}

TEST_F(QueueContextSizingTest, FixedWaveCountIncludesNavi10AndExcludesGfx1250) {
    TEST_START(TESTPROFILE_RUNALL)

    node.MaxWavesPerSIMD = 16;
    EXPECT_EQ(640u, hsakmt_get_num_waves(&node, GFX_VERSION_NAVI10, 20));
    EXPECT_EQ(640u, hsakmt_get_num_waves(&node, GFX_VERSION_GFX1250 - 1, 20));
    EXPECT_EQ(1280u, hsakmt_get_num_waves(&node, GFX_VERSION_GFX1250, 20));

    TEST_END
}
