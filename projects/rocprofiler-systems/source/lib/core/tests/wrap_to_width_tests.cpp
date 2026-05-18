// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "core/output/summary_renderer.hpp"

#include <cstddef>
#include <string>

TEST(wrap_to_width, breaks_on_word_boundary_when_available)
{
    const auto chunks = rocprofsys::output::wrap_to_width(
        "the quick brown fox jumps over the lazy dog", 16);
    ASSERT_FALSE(chunks.empty());
    for(const auto& c : chunks)
    {
        EXPECT_LE(c.size(), 16u);
        EXPECT_FALSE(c.empty());
    }
    // Joining with " " round-trips the original string (since the
    // wrapper consumes the breaking space).
    std::string joined;
    for(std::size_t i = 0; i < chunks.size(); ++i)
    {
        if(i > 0) joined += ' ';
        joined += chunks[i];
    }
    EXPECT_EQ(joined, "the quick brown fox jumps over the lazy dog");
}

TEST(wrap_to_width, falls_back_to_byte_chunking_for_path_without_spaces)
{
    const std::string path   = "/very/long/path/with/many/components/and/no/spaces.txt";
    const auto        chunks = rocprofsys::output::wrap_to_width(path, 20);
    ASSERT_FALSE(chunks.empty());
    for(const auto& c : chunks)
        EXPECT_LE(c.size(), 20u);
    std::string joined;
    for(const auto& c : chunks)
        joined += c;
    EXPECT_EQ(joined, path);
}

TEST(wrap_to_width, never_splits_a_utf8_codepoint)
{
    // 6 box-drawing chars (3 bytes each) = 18 bytes. Width 10 forces
    // a split mid-string; the split must land between code points,
    // i.e. no chunk after the first may START with a UTF-8
    // continuation byte (10xxxxxx).
    const std::string content = "──────";  // 6 × U+2500
    const auto        chunks  = rocprofsys::output::wrap_to_width(content, 10);
    ASSERT_FALSE(chunks.empty());
    for(std::size_t i = 1; i < chunks.size(); ++i)
    {
        ASSERT_FALSE(chunks[i].empty());
        const auto first_byte = static_cast<unsigned char>(chunks[i].front());
        EXPECT_NE(first_byte & 0xC0, 0x80);
    }
    std::string joined;
    for(const auto& c : chunks)
        joined += c;
    EXPECT_EQ(joined, content);
}

TEST(wrap_to_width, never_splits_a_utf8_codepoint_in_mixed_ascii_input)
{
    // Mixed ASCII + 3-byte UTF-8 (a─b─c─d─...): exercises the
    // backoff over an ASCII byte adjacent to a multi-byte code point.
    const std::string content = "a─b─c─d─e─f─g─h─i─j";
    const auto        chunks  = rocprofsys::output::wrap_to_width(content, 7);
    ASSERT_FALSE(chunks.empty());
    for(std::size_t i = 1; i < chunks.size(); ++i)
    {
        ASSERT_FALSE(chunks[i].empty());
        const auto first_byte = static_cast<unsigned char>(chunks[i].front());
        EXPECT_NE(first_byte & 0xC0, 0x80);
    }
    std::string joined;
    for(const auto& c : chunks)
        joined += c;
    EXPECT_EQ(joined, content);
}
