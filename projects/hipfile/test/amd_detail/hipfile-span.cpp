/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hipfile-span.h"
#include "hipfile-warnings.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <span>
#include <string>
#include <string_view>

using namespace hipFile;

HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

namespace {

TEST(HipFileSpan, SpanFromPointerViewsTheObjects)
{
    std::array<int, 4>   values{{1, 2, 3, 4}};
    const std::span<int> view = spanFromPointer(values.data(), 3);
    EXPECT_EQ(view.data(), values.data());
    ASSERT_EQ(view.size(), 3U);
    view[2] = 30;
    EXPECT_EQ(values[2], 30);
}

TEST(HipFileSpan, CStringViewStopsAtTheNul)
{
    const std::string text = "token";
    EXPECT_EQ(cStringView(text.c_str(), 100), "token");
    EXPECT_EQ(cStringView("", 100), "");
    /* The NUL itself is within the limit */
    EXPECT_EQ(cStringView(text.c_str(), text.size() + 1), "token");
}

TEST(HipFileSpan, CStringViewCutsALongString)
{
    /* No NUL in the first maxLen bytes, so it scans no further */
    const std::array<char, 4> unterminated{{'a', 'b', 'c', 'd'}};
    EXPECT_EQ(cStringView(unterminated.data(), unterminated.size()), "abcd");
    EXPECT_EQ(cStringView("token", 3), "tok");
    EXPECT_EQ(cStringView("token", 0), "");
}

TEST(HipFileSpan, CStringViewOfAnArrayStopsAtTheNulOrTheEnd)
{
    const std::array<char, 8> terminated{{'e', 'x', 't', '4', '\0', 'x', 'x', 'x'}};
    EXPECT_EQ(cStringView(terminated), "ext4");

    const std::array<char, 4> unterminated{{'a', 'b', 'c', 'd'}};
    EXPECT_EQ(cStringView(unterminated), "abcd");
}

TEST(HipFileSpan, CStringViewOfAnEmptySpanIsEmpty)
{
    /* A default-constructed span's data() is null */
    EXPECT_EQ(cStringView(std::span<const char>{}), "");
    EXPECT_EQ(cStringView(nullptr, 0), "");
}

TEST(HipFileSpan, AsStringViewViewsTheSameCharacters)
{
    const std::array<char, 5> chars{{'h', 'e', 'l', 'l', 'o'}};
    const std::string_view    text = asStringView(std::span(chars).first(4));
    EXPECT_EQ(text, "hell");
    EXPECT_EQ(text.data(), chars.data());
}

TEST(HipFileSpan, ArgSpanViewsTheArguments)
{
    std::string           prog = "prog";
    std::string           arg  = "--flag";
    std::array<char *, 3> argv{{prog.data(), arg.data(), nullptr}};

    const std::span<char *> args = argSpan(2, argv.data());
    ASSERT_EQ(args.size(), 2U);
    EXPECT_EQ(std::string_view(args[1]), "--flag");

    EXPECT_TRUE(argSpan(0, argv.data()).empty());
    EXPECT_TRUE(argSpan(-1, argv.data()).empty());
}

/* Padding after c and after s, on every ABI hipFile builds for */
struct Padded {
    char     c;
    uint64_t u;
    uint16_t s;
};

/* Padding after c. Small enough to be passed and returned in registers,
 * where a copy is the most likely to drop the padding. */
struct SmallPadded {
    char     c;
    uint16_t s;
};

/* Fill obj with a nonzero pattern, zero it, and check every byte */
template <typename T>
bool
zeroObjectZeroesEveryByte(T &obj)
{
    const std::span<std::byte, sizeof(T)> bytes = std::as_writable_bytes(std::span<T, 1>{&obj, 1});
    std::ranges::fill(bytes, std::byte{0xa5});
    zeroObject(obj);
    return std::ranges::all_of(bytes, [](std::byte b) noexcept { return b == std::byte{0}; });
}

TEST(HipFileSpan, ZeroObjectZeroesThePadding)
{
    Padded padded{};
    EXPECT_TRUE(zeroObjectZeroesEveryByte(padded));
}

TEST(HipFileSpan, ZeroObjectZeroesThePaddingOfASmallStruct)
{
    SmallPadded padded{};
    EXPECT_TRUE(zeroObjectZeroesEveryByte(padded));
}

}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
