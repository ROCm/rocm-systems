// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <chrono>
#include <type_traits>

#include "common/units/data_size.hpp"
#include "common/units/format.hpp"
#include "common/units/frequency.hpp"
#include "common/units/power.hpp"

using namespace rocprofsys::common::units;
using namespace rocprofsys::common::units::literals;

// ---------------------------------------------------------------------------
// frequency: construction and count
// ---------------------------------------------------------------------------

TEST(UnitsFrequency, ConstructionAndCount)
{
    EXPECT_DOUBLE_EQ(hertz{ 50.0 }.count(), 50.0);
    EXPECT_DOUBLE_EQ(kilohertz{ 1.5 }.count(), 1.5);
    EXPECT_DOUBLE_EQ(megahertz{ 2.0 }.count(), 2.0);
}

// Construction from long/long long (signed or unsigned) must be explicit at
// the call site, since those can't always be represented exactly as a double.
// Other implicit conversions (e.g. int) are still allowed.
static_assert(!std::is_constructible_v<hertz, long>);
static_assert(!std::is_constructible_v<hertz, unsigned long>);
static_assert(!std::is_constructible_v<hertz, long long>);
static_assert(!std::is_constructible_v<hertz, unsigned long long>);
static_assert(std::is_constructible_v<hertz, int>);
static_assert(std::is_constructible_v<hertz, double>);

TEST(UnitsFrequency, Literals)
{
    EXPECT_DOUBLE_EQ((50_hz).count(), 50.0);
    EXPECT_DOUBLE_EQ((1500_khz).count(), 1500.0);
    EXPECT_DOUBLE_EQ((2_mhz).count(), 2.0);
}

TEST(UnitsFrequency, GhzConstructionAndLiterals)
{
    EXPECT_DOUBLE_EQ(gigahertz{ 2.4 }.count(), 2.4);
    EXPECT_DOUBLE_EQ((2_ghz).count(), 2.0);
    const auto result = frequency_cast<megahertz>(1_ghz);
    EXPECT_DOUBLE_EQ(result.count(), 1000.0);
}

TEST(UnitsFrequency, CastHzToKhz)
{
    const auto result = frequency_cast<kilohertz>(2000_hz);
    EXPECT_DOUBLE_EQ(result.count(), 2.0);
}

TEST(UnitsFrequency, CastMhzToHz)
{
    const auto result = frequency_cast<hertz>(2_mhz);
    EXPECT_DOUBLE_EQ(result.count(), 2'000'000.0);
}

TEST(UnitsFrequency, CastKhzToMhz)
{
    const auto result = frequency_cast<megahertz>(1500_khz);
    EXPECT_DOUBLE_EQ(result.count(), 1.5);
}

TEST(UnitsFrequency, EqualityAcrossUnits)
{
    EXPECT_EQ(1000_hz, 1_khz);
    EXPECT_EQ(1000_khz, 1_mhz);
}

TEST(UnitsFrequency, OrderingAcrossUnits)
{
    EXPECT_LT(999_hz, 1_khz);
    EXPECT_GT(2_mhz, 1999_khz);
}

// ---------------------------------------------------------------------------
// data_size: construction, count, to_bytes
// ---------------------------------------------------------------------------

TEST(UnitsDataSize, ConstructionAndCount)
{
    EXPECT_DOUBLE_EQ(bytes{ 512.0 }.count(), 512.0);
    EXPECT_DOUBLE_EQ(kilobytes{ 4.0 }.count(), 4.0);
    EXPECT_DOUBLE_EQ(megabytes{ 2.0 }.count(), 2.0);
}

// Construction from long/long long (signed or unsigned) must be explicit at
// the call site, since those can't always be represented exactly as a double.
// Other implicit conversions (e.g. int) are still allowed.
static_assert(!std::is_constructible_v<bytes, long>);
static_assert(!std::is_constructible_v<bytes, unsigned long>);
static_assert(!std::is_constructible_v<bytes, long long>);
static_assert(!std::is_constructible_v<bytes, unsigned long long>);
static_assert(std::is_constructible_v<bytes, int>);
static_assert(std::is_constructible_v<bytes, double>);

TEST(UnitsDataSize, Literals)
{
    EXPECT_DOUBLE_EQ((512_b).count(), 512.0);
    EXPECT_DOUBLE_EQ((4_kb).count(), 4.0);
    EXPECT_DOUBLE_EQ((2_mb).count(), 2.0);
    EXPECT_DOUBLE_EQ((1_gb).count(), 1.0);
    EXPECT_DOUBLE_EQ((1_tb).count(), 1.0);
}

TEST(UnitsDataSize, ToBytesDecimal)
{
    EXPECT_DOUBLE_EQ((1_kb).to_bytes(), 1000.0);
    EXPECT_DOUBLE_EQ((1_mb).to_bytes(), 1000.0 * 1000.0);
    EXPECT_DOUBLE_EQ((1_gb).to_bytes(), 1000.0 * 1000.0 * 1000.0);
}

TEST(UnitsDataSize, ToBytesBinary)
{
    EXPECT_DOUBLE_EQ((1_kib).to_bytes(), 1024.0);
    EXPECT_DOUBLE_EQ((1_mib).to_bytes(), 1024.0 * 1024.0);
    EXPECT_DOUBLE_EQ((1_gib).to_bytes(), 1024.0 * 1024.0 * 1024.0);
}

TEST(UnitsDataSize, CastBToKb)
{
    const auto result = data_size_cast<kilobytes>(2000_b);
    EXPECT_DOUBLE_EQ(result.count(), 2.0);
}

TEST(UnitsDataSize, CastBToKib)
{
    const auto result = data_size_cast<kibibytes>(2048_b);
    EXPECT_DOUBLE_EQ(result.count(), 2.0);
}

TEST(UnitsDataSize, CastMbToKb)
{
    const auto result = data_size_cast<kilobytes>(2_mb);
    EXPECT_DOUBLE_EQ(result.count(), 2000.0);
}

TEST(UnitsDataSize, CastMibToKib)
{
    const auto result = data_size_cast<kibibytes>(2_mib);
    EXPECT_DOUBLE_EQ(result.count(), 2048.0);
}

// The two families are distinct types that never alias: 1 KiB is 24 B larger.
TEST(UnitsDataSize, CastAcrossFamilies)
{
    EXPECT_DOUBLE_EQ(data_size_cast<bytes>(1_kib).count(), 1024.0);
    EXPECT_DOUBLE_EQ(data_size_cast<bytes>(1_kb).count(), 1000.0);
    EXPECT_GT(1_kib, 1_kb);
}

TEST(UnitsDataSize, EqualityAcrossUnits)
{
    EXPECT_EQ(1000_b, 1_kb);
    EXPECT_EQ(1000_kb, 1_mb);
    EXPECT_EQ(1024_b, 1_kib);
    EXPECT_EQ(1024_kib, 1_mib);
}

TEST(UnitsDataSize, OrderingAcrossUnits)
{
    EXPECT_LT(999_b, 1_kb);
    EXPECT_GT(2_mb, 1999_kb);
    EXPECT_LT(1023_b, 1_kib);
    EXPECT_GT(2_mib, 2047_kib);
}

// ---------------------------------------------------------------------------
// fmt formatting: frequency
// ---------------------------------------------------------------------------

TEST(UnitsFmtFrequency, PlainHz) { EXPECT_EQ(fmt::format("{}", 50_hz), "50 Hz"); }

TEST(UnitsFmtFrequency, PlainMhz) { EXPECT_EQ(fmt::format("{}", 2_mhz), "2 MHz"); }

TEST(UnitsFmtFrequency, AutoscaleHzToMhz)
{
    EXPECT_EQ(fmt::format("{:~}", hertz{ 50'000'000.0 }), "50 MHz");
}

TEST(UnitsFmtFrequency, AutoscaleHzToKhz)
{
    EXPECT_EQ(fmt::format("{:~}", 1500_hz), "1.5 kHz");
}

TEST(UnitsFmtFrequency, AutoscaleWithPrecision)
{
    EXPECT_EQ(fmt::format("{:~.2f}", 1500_hz), "1.50 kHz");
}

TEST(UnitsFmtFrequency, AutoscaleHzToGhz)
{
    EXPECT_EQ(fmt::format("{:~}", hertz{ 3'000'000'000.0 }), "3 GHz");
}

TEST(UnitsFmtFrequency, AutoscaleGhzLiteral)
{
    EXPECT_EQ(fmt::format("{:~}", 2.4_ghz), "2.4 GHz");
}

// ---------------------------------------------------------------------------
// fmt formatting: data_size
// ---------------------------------------------------------------------------

TEST(UnitsFmtDataSize, PlainBytes) { EXPECT_EQ(fmt::format("{}", 512_b), "512 B"); }

TEST(UnitsFmtDataSize, PlainMb) { EXPECT_EQ(fmt::format("{}", 5_mb), "5 MB"); }

TEST(UnitsFmtDataSize, PlainMib) { EXPECT_EQ(fmt::format("{}", 5_mib), "5 MiB"); }

TEST(UnitsFmtDataSize, AutoscaleBToMb)
{
    EXPECT_EQ(fmt::format("{:~}", bytes{ 5'000'000.0 }), "5 MB");
}

TEST(UnitsFmtDataSize, AutoscaleBToKb) { EXPECT_EQ(fmt::format("{:~}", 2000_b), "2 KB"); }

TEST(UnitsFmtDataSize, AutoscaleWithPrecision)
{
    EXPECT_EQ(fmt::format("{:~.1f}", 1500_b), "1.5 KB");
}

// Autoscaling always reports decimal units, whatever family the input used:
// the value is converted to bytes first, so number and suffix stay consistent.
TEST(UnitsFmtDataSize, AutoscaleBinaryInputReportsDecimal)
{
    EXPECT_EQ(fmt::format("{:~.3f}", 1_mib), "1.049 MB");
}

// ---------------------------------------------------------------------------
// fmt formatting: std::chrono::duration (via chrono.hpp)
// ---------------------------------------------------------------------------

TEST(UnitsFmtChrono, PlainMs)
{
    using std::chrono::milliseconds;
    EXPECT_EQ(fmt::format("{}", milliseconds{ 1500 }), "1500 ms");
}

TEST(UnitsFmtChrono, AutoscaleMsToS)
{
    using std::chrono::milliseconds;
    EXPECT_EQ(fmt::format("{:~}", milliseconds{ 1500 }), "1.5 s");
}

TEST(UnitsFmtChrono, AutoscaleNsToUs)
{
    using std::chrono::nanoseconds;
    EXPECT_EQ(fmt::format("{:~}", nanoseconds{ 250'000 }), "250 us");
}

TEST(UnitsFmtChrono, AutoscaleWithPrecision)
{
    using std::chrono::microseconds;
    EXPECT_EQ(fmt::format("{:~.3f}", microseconds{ 1500 }), "1.500 ms");
}

// ---------------------------------------------------------------------------
// power: construction, cast, literals, comparison
// ---------------------------------------------------------------------------

TEST(UnitsPower, ConstructionAndCount)
{
    EXPECT_DOUBLE_EQ(watt{ 1.5 }.count(), 1.5);
    EXPECT_DOUBLE_EQ(milliwatt{ 250.0 }.count(), 250.0);
    EXPECT_DOUBLE_EQ(kilowatt{ 2.0 }.count(), 2.0);
}

// Construction from long/long long (signed or unsigned) must be explicit at
// the call site, since those can't always be represented exactly as a double.
// Other implicit conversions (e.g. int) are still allowed.
static_assert(!std::is_constructible_v<watt, long>);
static_assert(!std::is_constructible_v<watt, unsigned long>);
static_assert(!std::is_constructible_v<watt, long long>);
static_assert(!std::is_constructible_v<watt, unsigned long long>);
static_assert(std::is_constructible_v<watt, int>);
static_assert(std::is_constructible_v<watt, double>);

TEST(UnitsPower, Literals)
{
    EXPECT_DOUBLE_EQ((500_mw).count(), 500.0);
    EXPECT_DOUBLE_EQ((1_w).count(), 1.0);
    EXPECT_DOUBLE_EQ((2_kw).count(), 2.0);
    EXPECT_DOUBLE_EQ((100_nw).count(), 100.0);
}

TEST(UnitsPower, CastWToMw)
{
    const auto result = power_cast<milliwatt>(1_w);
    EXPECT_DOUBLE_EQ(result.count(), 1000.0);
}

TEST(UnitsPower, CastKwToW)
{
    const auto result = power_cast<watt>(2_kw);
    EXPECT_DOUBLE_EQ(result.count(), 2000.0);
}

TEST(UnitsPower, EqualityAcrossUnits)
{
    EXPECT_EQ(1000_mw, 1_w);
    EXPECT_EQ(1000_w, 1_kw);
}

TEST(UnitsPower, OrderingAcrossUnits)
{
    EXPECT_LT(999_mw, 1_w);
    EXPECT_GT(2_kw, 1999_w);
}

// ---------------------------------------------------------------------------
// fmt formatting: power
// ---------------------------------------------------------------------------

TEST(UnitsFmtPower, PlainW) { EXPECT_EQ(fmt::format("{}", 1_w), "1 W"); }

TEST(UnitsFmtPower, PlainMw) { EXPECT_EQ(fmt::format("{}", 250_mw), "250 mW"); }

TEST(UnitsFmtPower, AutoscaleMwToW)
{
    EXPECT_EQ(fmt::format("{:~}", watt{ 1500.0 }), "1.5 kW");
}

TEST(UnitsFmtPower, AutoscaleWToMw)
{
    EXPECT_EQ(fmt::format("{:~}", watt{ 0.5 }), "500 mW");
}

TEST(UnitsFmtPower, AutoscaleWithPrecision)
{
    EXPECT_EQ(fmt::format("{:~.2f}", milliwatt{ 1500.0 }), "1.50 W");
}

TEST(UnitsFmtPower, AutoscaleZeroIsWattsNotNanowatts)
{
    EXPECT_EQ(fmt::format("{:~}", watt{ 0.0 }), "0 W");
}
