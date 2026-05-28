// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "leaf_context.h"

#include <gtest/gtest.h>

TEST(RoctxRecordFnLeafContext, ForwardTopLevelUsesAtenSentinel)
{
    EXPECT_STREQ(
        roctx_recordfn::default_leaf_context(false, /*seq_nr=*/42, /*stack_was_empty=*/true),
        roctx_recordfn::kAtenTopLevelLeaf
    );
}

TEST(RoctxRecordFnLeafContext, ForwardNestedUsesAtenNestedSentinel)
{
    EXPECT_STREQ(
        roctx_recordfn::default_leaf_context(false, /*seq_nr=*/42, /*stack_was_empty=*/false),
        roctx_recordfn::kAtenNestedLeaf
    );
}

TEST(RoctxRecordFnLeafContext, BackwardWithSeqUsesAutogradBackwardSentinel)
{
    EXPECT_STREQ(
        roctx_recordfn::default_leaf_context(true, /*seq_nr=*/7, /*stack_was_empty=*/true),
        roctx_recordfn::kAutogradBackwardLeaf
    );
}

TEST(RoctxRecordFnLeafContext, BackwardWithoutSeqUsesAutogradEngineSentinel)
{
    EXPECT_STREQ(
        roctx_recordfn::default_leaf_context(true, /*seq_nr=*/-1, /*stack_was_empty=*/true),
        roctx_recordfn::kAutogradEngineLeaf
    );
}
