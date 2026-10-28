// Unit tests for cpu_snap::core::format_hex.

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>

#include "cpu_snap/core/format_hex.hpp"

using cpu_snap::core::format_hex;

TEST(FormatHexTest, ZeroIsSingleDigit)
{
    EXPECT_EQ(format_hex(0), "0x0");
}

TEST(FormatHexTest, IsLowercaseAndUnpadded)
{
    EXPECT_EQ(format_hex(0xabc), "0xabc");
    EXPECT_EQ(format_hex(0x1), "0x1");
}

TEST(FormatHexTest, MaxUint64UsesAllSixteenDigits)
{
    EXPECT_EQ(format_hex(std::numeric_limits<std::uint64_t>::max()), "0xffffffffffffffff");
}

TEST(FormatHexTest, MatchesPrintfForRepresentativeValues)
{
    for(const std::uint64_t value : std::array<std::uint64_t, 4>{
            0x7f1234567890, 0x400000, 0x8000000000000000, 0xdeadbeefcafef00d, })
    {
        std::array<char, 32> buf{};
        std::snprintf(buf.data(), buf.size(), "0x%lx", static_cast<unsigned long>(value));
        EXPECT_EQ(format_hex(value), buf.data());
    }
}
