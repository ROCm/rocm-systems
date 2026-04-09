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

#include "segment.hpp"
#include <gtest/gtest.h>

// Tests for address_range_t
class AddressRangeTest : public ::testing::Test
{
protected:
    address_range_t range1{100, 50, 1}; // [100, 150)
    address_range_t range2{200, 50, 2}; // [200, 250)
};

TEST_F(AddressRangeTest, InrangeReturnsCorrectly)
{
    EXPECT_TRUE(range1.inrange(100));
    EXPECT_TRUE(range1.inrange(125));
    EXPECT_TRUE(range1.inrange(149));
    EXPECT_FALSE(range1.inrange(150));
    EXPECT_FALSE(range1.inrange(99));
}

TEST_F(AddressRangeTest, EqualityForOverlappingRanges)
{
    // Ranges are "equal" if they overlap
    address_range_t overlap{125, 50, 3}; // [125, 175) overlaps with range1 [100, 150)
    EXPECT_TRUE(range1 == overlap);
    EXPECT_FALSE(range1 == range2); // No overlap
}

TEST_F(AddressRangeTest, LessThanForNonOverlapping)
{
    EXPECT_TRUE(range1 < range2);
    EXPECT_FALSE(range2 < range1);

    // Overlapping ranges are neither less than
    address_range_t overlap{125, 50, 3};
    EXPECT_FALSE(range1 < overlap);
    EXPECT_FALSE(overlap < range1);
}

// Tests for CodeobjTableTranslator
class CodeobjTableTranslatorTest : public ::testing::Test
{
protected:
    CodeobjTableTranslator translator;

    void SetUp() override
    {
        translator.insert({1000, 100, 1}); // [1000, 1100), id=1
        translator.insert({2000, 200, 2}); // [2000, 2200), id=2
        translator.insert({3000, 50, 3});  // [3000, 3050), id=3
    }
};

TEST_F(CodeobjTableTranslatorTest, FindCodeobjInRangeSuccess)
{
    address_range_t out;

    EXPECT_TRUE(translator.find_codeobj_in_range(1050, out));
    EXPECT_EQ(out.id, 1);
    EXPECT_EQ(out.vbegin, 1000);

    EXPECT_TRUE(translator.find_codeobj_in_range(2100, out));
    EXPECT_EQ(out.id, 2);

    EXPECT_TRUE(translator.find_codeobj_in_range(3000, out));
    EXPECT_EQ(out.id, 3);
}

TEST_F(CodeobjTableTranslatorTest, FindCodeobjInRangeFailure)
{
    address_range_t out;

    EXPECT_FALSE(translator.find_codeobj_in_range(500, out));
    EXPECT_FALSE(translator.find_codeobj_in_range(1500, out));
    EXPECT_FALSE(translator.find_codeobj_in_range(4000, out));
}

TEST_F(CodeobjTableTranslatorTest, CacheWorksCorrectly)
{
    address_range_t out;

    // First lookup
    EXPECT_TRUE(translator.find_codeobj_in_range(1050, out));
    EXPECT_EQ(out.id, 1);

    // Second lookup in same range should use cache
    EXPECT_TRUE(translator.find_codeobj_in_range(1075, out));
    EXPECT_EQ(out.id, 1);

    // Lookup in different range
    EXPECT_TRUE(translator.find_codeobj_in_range(2050, out));
    EXPECT_EQ(out.id, 2);
}

TEST_F(CodeobjTableTranslatorTest, RemoveByAddress)
{
    address_range_t out;

    EXPECT_TRUE(translator.find_codeobj_in_range(1050, out));
    EXPECT_TRUE(translator.remove(1000));
    EXPECT_FALSE(translator.find_codeobj_in_range(1050, out));
}

TEST_F(CodeobjTableTranslatorTest, RemoveByRange)
{
    address_range_t out;

    EXPECT_TRUE(translator.find_codeobj_in_range(2100, out));
    EXPECT_TRUE(translator.remove({2000, 200, 2}));
    EXPECT_FALSE(translator.find_codeobj_in_range(2100, out));
}

TEST_F(CodeobjTableTranslatorTest, ClearCacheAfterModification)
{
    address_range_t out;

    // Populate cache
    EXPECT_TRUE(translator.find_codeobj_in_range(1050, out));

    // Remove should clear cache
    translator.remove(1000);

    // Now the cached segment should not be used
    EXPECT_FALSE(translator.find_codeobj_in_range(1075, out));
}

TEST_F(CodeobjTableTranslatorTest, ToPcV2ReturnsValidPc)
{
    pcinfo_t pc = translator.ToPcV2(1050);
    EXPECT_EQ(pc.code_object_id, 1);
    EXPECT_EQ(pc.address, 50); // Offset from base
}

TEST_F(CodeobjTableTranslatorTest, ToPcV2ReturnsInvalidPcForOutOfRange)
{
    // When PC is not in any code object, returns raw address with code_object_id=0
    pcinfo_t pc = translator.ToPcV2(500);
    EXPECT_EQ(pc.code_object_id, 0);
    EXPECT_EQ(pc.address, 500); // Raw address preserved
}

TEST_F(CodeobjTableTranslatorTest, ToPcV2AtBoundary)
{
    // Test at exact start of range
    pcinfo_t pc_start = translator.ToPcV2(1000);
    EXPECT_EQ(pc_start.code_object_id, 1);
    EXPECT_EQ(pc_start.address, 0);

    // Test at end boundary (should be out of range - returns raw address)
    pcinfo_t pc_end = translator.ToPcV2(1100);
    EXPECT_EQ(pc_end.code_object_id, 0);
    EXPECT_EQ(pc_end.address, 1100); // Raw address preserved
}

// Test for pcinfo_t hash and equality
TEST(PcinfoTest, EqualityOperators)
{
    pcinfo_t a{100, 1};
    pcinfo_t b{100, 1};
    pcinfo_t c{100, 2};
    pcinfo_t d{200, 1};

    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
    EXPECT_FALSE(a == d);
    EXPECT_TRUE(a != c);
    EXPECT_TRUE(a != d);
}

TEST(PcinfoTest, HashFunction)
{
    pcinfo_t a{100, 1};
    pcinfo_t b{100, 1};
    pcinfo_t c{100, 2};

    std::hash<pcinfo_t> hasher;
    EXPECT_EQ(hasher(a), hasher(b));
    EXPECT_NE(hasher(a), hasher(c));
}

// Edge case tests for address_range_t
TEST(AddressRangeEdgeCaseTest, ZeroSizeRange)
{
    address_range_t zero_size{100, 0, 1}; // [100, 100) - empty range

    EXPECT_FALSE(zero_size.inrange(100));
    EXPECT_FALSE(zero_size.inrange(99));
}

TEST(AddressRangeEdgeCaseTest, MaxUint64Values)
{
    uint64_t max_val = UINT64_MAX;
    uint64_t near_max = max_val - 100;

    address_range_t range{near_max, 50, 1}; // Near max value

    EXPECT_TRUE(range.inrange(near_max));
    EXPECT_TRUE(range.inrange(near_max + 25));
    EXPECT_FALSE(range.inrange(near_max + 50));
}

TEST(AddressRangeEdgeCaseTest, AdjacentRangesNotEqual)
{
    address_range_t range1{100, 50, 1}; // [100, 150)
    address_range_t range2{150, 50, 2}; // [150, 200) - adjacent but not overlapping

    EXPECT_FALSE(range1 == range2);
    EXPECT_TRUE(range1 < range2);
}

// Edge case tests for CodeobjTableTranslator
TEST(CodeobjTableTranslatorEdgeCaseTest, LookupAtExactBoundaries)
{
    CodeobjTableTranslator translator;
    translator.insert({1000, 100, 1}); // [1000, 1100)

    address_range_t out;

    // Exact start
    EXPECT_TRUE(translator.find_codeobj_in_range(1000, out));
    EXPECT_EQ(out.id, 1);

    // One before end
    EXPECT_TRUE(translator.find_codeobj_in_range(1099, out));
    EXPECT_EQ(out.id, 1);

    // Exact end (out of range)
    EXPECT_FALSE(translator.find_codeobj_in_range(1100, out));

    // One before start (out of range)
    EXPECT_FALSE(translator.find_codeobj_in_range(999, out));
}

TEST(CodeobjTableTranslatorEdgeCaseTest, EmptyTranslator)
{
    CodeobjTableTranslator translator;
    address_range_t out;

    EXPECT_FALSE(translator.find_codeobj_in_range(100, out));
    EXPECT_FALSE(translator.remove(100));
}

TEST(CodeobjTableTranslatorEdgeCaseTest, MultipleRemoves)
{
    CodeobjTableTranslator translator;
    translator.insert({1000, 100, 1});
    translator.insert({2000, 100, 2});
    translator.insert({3000, 100, 3});

    // Remove middle
    EXPECT_TRUE(translator.remove(2000));

    address_range_t out;
    EXPECT_TRUE(translator.find_codeobj_in_range(1050, out));
    EXPECT_FALSE(translator.find_codeobj_in_range(2050, out));
    EXPECT_TRUE(translator.find_codeobj_in_range(3050, out));

    // Remove first
    EXPECT_TRUE(translator.remove(1000));
    EXPECT_FALSE(translator.find_codeobj_in_range(1050, out));
}

//=============================================================================
// Overlap insertion tests
// Check for violation of std::set ordering invariants
//=============================================================================

TEST(CodeobjOverlapTest, RightOverlapDeletesExisting)
{
    // Insert [100, 200), then [150, 300)
    // First range overlaps and is deleted entirely
    CodeobjTableTranslator translator;
    translator.insert({100, 100, 1}); // [100, 200)
    translator.insert({150, 150, 2}); // [150, 300) -> deletes range 1

    address_range_t out;
    EXPECT_FALSE(translator.find_codeobj_in_range(125, out)); // range 1 gone

    EXPECT_TRUE(translator.find_codeobj_in_range(150, out));
    EXPECT_EQ(out.id, 2);

    EXPECT_TRUE(translator.find_codeobj_in_range(299, out));
    EXPECT_EQ(out.id, 2);
}

TEST(CodeobjOverlapTest, LeftOverlapDeletesExisting)
{
    // Insert [200, 300), then [100, 250)
    // First range overlaps and is deleted entirely
    CodeobjTableTranslator translator;
    translator.insert({200, 100, 1}); // [200, 300)
    translator.insert({100, 150, 2}); // [100, 250) -> deletes range 1

    address_range_t out;
    EXPECT_TRUE(translator.find_codeobj_in_range(150, out));
    EXPECT_EQ(out.id, 2);

    EXPECT_TRUE(translator.find_codeobj_in_range(249, out));
    EXPECT_EQ(out.id, 2);

    EXPECT_FALSE(translator.find_codeobj_in_range(250, out)); // range 1 gone
    EXPECT_FALSE(translator.find_codeobj_in_range(299, out));
}

TEST(CodeobjOverlapTest, FullyContainedDeletesExisting)
{
    // Insert [100, 300), then [150, 250) inside it
    // First range is deleted entirely, second inserted
    CodeobjTableTranslator translator;
    translator.insert({100, 200, 1}); // [100, 300)
    translator.insert({150, 100, 2}); // [150, 250) -> deletes range 1

    address_range_t out;
    EXPECT_FALSE(translator.find_codeobj_in_range(125, out)); // range 1 gone

    EXPECT_TRUE(translator.find_codeobj_in_range(200, out));
    EXPECT_EQ(out.id, 2);

    EXPECT_FALSE(translator.find_codeobj_in_range(275, out)); // range 1 gone
}

TEST(CodeobjOverlapTest, NewRangeDeletesMultipleExisting)
{
    // Insert [100, 200) and [300, 400), then [150, 350) overlaps both
    // Both existing ranges are deleted
    CodeobjTableTranslator translator;
    translator.insert({100, 100, 1}); // [100, 200)
    translator.insert({300, 100, 2}); // [300, 400)
    translator.insert({150, 200, 3}); // [150, 350) -> deletes both

    address_range_t out;
    EXPECT_FALSE(translator.find_codeobj_in_range(125, out)); // range 1 gone

    EXPECT_TRUE(translator.find_codeobj_in_range(200, out));
    EXPECT_EQ(out.id, 3);

    EXPECT_TRUE(translator.find_codeobj_in_range(349, out));
    EXPECT_EQ(out.id, 3);

    EXPECT_FALSE(translator.find_codeobj_in_range(350, out)); // range 2 gone
}

TEST(CodeobjOverlapTest, ExactDuplicateReplacesExisting)
{
    CodeobjTableTranslator translator;
    translator.insert({100, 100, 1});
    translator.insert({100, 100, 2}); // Same range, new id wins

    address_range_t out;
    EXPECT_TRUE(translator.find_codeobj_in_range(150, out));
    EXPECT_EQ(out.id, 2);
}

TEST(CodeobjOverlapTest, NonOverlappingPreserved)
{
    // Ranges that don't overlap should not be affected
    CodeobjTableTranslator translator;
    translator.insert({100, 50, 1}); // [100, 150)
    translator.insert({200, 50, 2}); // [200, 250)
    translator.insert({300, 50, 3}); // [300, 350)
    translator.insert({160, 30, 4}); // [160, 190) - no overlap with any

    address_range_t out;
    EXPECT_TRUE(translator.find_codeobj_in_range(125, out));
    EXPECT_EQ(out.id, 1);
    EXPECT_TRUE(translator.find_codeobj_in_range(175, out));
    EXPECT_EQ(out.id, 4);
    EXPECT_TRUE(translator.find_codeobj_in_range(225, out));
    EXPECT_EQ(out.id, 2);
    EXPECT_TRUE(translator.find_codeobj_in_range(325, out));
    EXPECT_EQ(out.id, 3);
}

TEST(CodeobjOverlapTest, TransitivityPreserved)
{
    CodeobjTableTranslator translator;
    translator.insert({100, 100, 1}); // [100, 200)
    translator.insert({150, 100, 2}); // [150, 250) -> deletes 1
    translator.insert({225, 75, 3});  // [225, 300) -> deletes 2

    address_range_t out;
    EXPECT_FALSE(translator.find_codeobj_in_range(125, out)); // range 1 gone
    EXPECT_FALSE(translator.find_codeobj_in_range(200, out)); // range 2 gone

    EXPECT_TRUE(translator.find_codeobj_in_range(275, out));
    EXPECT_EQ(out.id, 3);

    EXPECT_FALSE(translator.find_codeobj_in_range(300, out));
}
