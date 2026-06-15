// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_pc_string_table.h"

TEST_F(test_pc_string_table_t, ProvidedFreshTable_InsertReturnsZeroAndStoresPair)
{
    EXPECT_EQ(m_string_table.insert("a", "x"), 0u);
    ASSERT_EQ(m_string_table.instructions().size(), 1u);
    ASSERT_EQ(m_string_table.comments().size(), 1u);
    EXPECT_EQ(m_string_table.instructions()[0], "a");
    EXPECT_EQ(m_string_table.comments()[0], "x");
}

TEST_F(test_pc_string_table_t, ProvidedSamePairTwice_DedupsToSameIndexWithoutGrowing)
{
    EXPECT_EQ(m_string_table.insert("a", "x"), 0u);
    EXPECT_EQ(m_string_table.insert("a", "x"), 0u);
    EXPECT_EQ(m_string_table.instructions().size(), 1u);
    EXPECT_EQ(m_string_table.comments().size(), 1u);
}

TEST_F(test_pc_string_table_t, ProvidedNewPair_ReturnsNextIndexAndGrowsBothArrays)
{
    EXPECT_EQ(m_string_table.insert("a", "x"), 0u);
    EXPECT_EQ(m_string_table.insert("b", "y"), 1u);
    ASSERT_EQ(m_string_table.instructions().size(), 2u);
    ASSERT_EQ(m_string_table.comments().size(), 2u);
    EXPECT_EQ(m_string_table.instructions()[1], "b");
    EXPECT_EQ(m_string_table.comments()[1], "y");
}

TEST_F(test_pc_string_table_t, ProvidedPartialMatches_KeyIsTheFullPairNotEitherFieldAlone)
{
    EXPECT_EQ(m_string_table.insert("a", "x"), 0u);

    // Same text, different comment -> new entry.
    const size_t same_text_idx = m_string_table.insert("a", "z");
    EXPECT_NE(same_text_idx, 0u);

    // Different text, same comment -> also new entry.
    const size_t same_comment_idx = m_string_table.insert("c", "x");
    EXPECT_NE(same_comment_idx, 0u);
    EXPECT_NE(same_comment_idx, same_text_idx);

    EXPECT_EQ(m_string_table.instructions().size(), 3u);
    EXPECT_EQ(m_string_table.comments().size(), 3u);
}

TEST_F(test_pc_string_table_t, ProvidedSeveralPairs_ReturnedIndexEqualsPositionInBothParallelArrays)
{
    const size_t i0 = m_string_table.insert("mov", "// a");
    const size_t i1 = m_string_table.insert("add", "// b");
    const size_t i2 = m_string_table.insert("mul", "// c");

    EXPECT_EQ(m_string_table.instructions()[i0], "mov");
    EXPECT_EQ(m_string_table.comments()[i0], "// a");
    EXPECT_EQ(m_string_table.instructions()[i1], "add");
    EXPECT_EQ(m_string_table.comments()[i1], "// b");
    EXPECT_EQ(m_string_table.instructions()[i2], "mul");
    EXPECT_EQ(m_string_table.comments()[i2], "// c");

    ASSERT_EQ(m_string_table.instructions().size(), m_string_table.comments().size());
    EXPECT_EQ(m_string_table.insert("add", "// b"), i1);
}
