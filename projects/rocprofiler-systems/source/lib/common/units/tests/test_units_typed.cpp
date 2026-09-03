// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <fmt/format.h>

// provides fmt::formatter for std::chrono::duration, exercised by the
// UnitsFmtChrono coexistence test
#include <spdlog/fmt/chrono.h>  // NOLINT(misc-include-cleaner)

#include <chrono>
#include <type_traits>

#include "common/units/data_size.hpp"
#include "common/units/frequency.hpp"
#include "common/units/power.hpp"

using namespace rocprofsys::common::units;
using namespace rocprofsys::common::units::literals;

// ---------------------------------------------------------------------------
// cross-family type firewall
//
// data_size_like / frequency_like / power_like are the only barrier between the
// three families, so pin that mixing them is ill-formed. The probes go through
// named concepts rather than a bare `requires` in the static_assert because gcc
// 13 turns an unsatisfied constrained call in a non-template context into a hard
// error instead of an unsatisfied requirement.
// ---------------------------------------------------------------------------

template <typename To, typename From>
concept data_size_castable = requires(From from) { data_size_cast<To>(from); };

template <typename LHS, typename RHS>
concept units_equality_comparable = requires(LHS lhs, RHS rhs) { lhs == rhs; };

static_assert(!data_size_castable<megabytes, hertz>);
static_assert(!data_size_castable<megabytes, watt>);
static_assert(data_size_castable<megabytes, bytes>);

static_assert(!units_equality_comparable<bytes, hertz>);
static_assert(!units_equality_comparable<watt, bytes>);
static_assert(!units_equality_comparable<hertz, watt>);
static_assert(units_equality_comparable<bytes, kilobytes>);

// ---------------------------------------------------------------------------
// frequency: construction and count
// ---------------------------------------------------------------------------

TEST(UnitsFrequency, ConstructionAndCount)
{
    EXPECT_DOUBLE_EQ(hertz{ 50.0 }.count(), 50.0);
    EXPECT_DOUBLE_EQ(kilohertz{ 1.5 }.count(), 1.5);
    EXPECT_DOUBLE_EQ(megahertz{ 2.0 }.count(), 2.0);
}

// The constructor is explicit and admits a source type only when every value it
// can hold is exactly representable in the rep, so `int` and `float` pass for a
// `double` rep while `long`, `long long` (signed or unsigned) and `long double`
// do not and the caller has to write the narrowing cast where it is visible.
static_assert(!std::is_constructible_v<hertz, long>);
static_assert(!std::is_constructible_v<hertz, unsigned long>);
static_assert(!std::is_constructible_v<hertz, long long>);
static_assert(!std::is_constructible_v<hertz, unsigned long long>);
static_assert(!std::is_constructible_v<hertz, long double>);
static_assert(!std::is_constructible_v<gigahertz, long>);
static_assert(std::is_constructible_v<hertz, int>);
static_assert(std::is_constructible_v<hertz, float>);
static_assert(std::is_constructible_v<hertz, double>);
static_assert(!std::is_convertible_v<double, hertz>);

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

TEST(UnitsFrequency, DefaultConstruction) { EXPECT_DOUBLE_EQ(hertz{}.count(), 0.0); }

TEST(UnitsFrequency, ZeroAndNegative)
{
    EXPECT_DOUBLE_EQ(frequency_cast<kilohertz>(hertz{ 0.0 }).count(), 0.0);
    EXPECT_DOUBLE_EQ(frequency_cast<hertz>(kilohertz{ 0.0 }).count(), 0.0);
    EXPECT_DOUBLE_EQ(frequency_cast<hertz>(kilohertz{ -2.5 }).count(), -2500.0);
    EXPECT_DOUBLE_EQ(frequency_cast<megahertz>(hertz{ -1'500'000.0 }).count(), -1.5);
}

TEST(UnitsFrequency, RoundTripHzToGhz)
{
    const auto original = hertz{ 2'400'000'000.0 };
    const auto restored = frequency_cast<hertz>(frequency_cast<gigahertz>(original));
    EXPECT_DOUBLE_EQ(restored.count(), original.count());
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

static_assert(!std::is_constructible_v<bytes, long>);
static_assert(!std::is_constructible_v<bytes, unsigned long>);
static_assert(!std::is_constructible_v<bytes, long long>);
static_assert(!std::is_constructible_v<bytes, unsigned long long>);
static_assert(!std::is_constructible_v<bytes, long double>);
static_assert(!std::is_constructible_v<megabytes, long>);
static_assert(!std::is_constructible_v<kibibytes, long>);
static_assert(std::is_constructible_v<bytes, int>);
static_assert(std::is_constructible_v<bytes, float>);
static_assert(std::is_constructible_v<bytes, double>);
static_assert(!std::is_convertible_v<double, bytes>);

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

TEST(UnitsDataSize, DefaultConstruction) { EXPECT_DOUBLE_EQ(bytes{}.count(), 0.0); }

TEST(UnitsDataSize, ZeroAndNegative)
{
    EXPECT_DOUBLE_EQ(data_size_cast<megabytes>(bytes{ 0.0 }).count(), 0.0);
    EXPECT_DOUBLE_EQ(data_size_cast<bytes>(kibibytes{ 0.0 }).count(), 0.0);
    EXPECT_DOUBLE_EQ(data_size_cast<bytes>(kilobytes{ -1.5 }).count(), -1500.0);
    EXPECT_DOUBLE_EQ(data_size_cast<bytes>(kibibytes{ -2.0 }).count(), -2048.0);
    EXPECT_DOUBLE_EQ(kilobytes{ -1.5 }.to_bytes(), -1500.0);
}

TEST(UnitsDataSize, RoundTripBytesToMegabytes)
{
    const auto original = bytes{ 2'500'000.0 };
    const auto restored = data_size_cast<bytes>(data_size_cast<megabytes>(original));
    EXPECT_DOUBLE_EQ(restored.count(), original.count());
}

TEST(UnitsDataSize, RoundTripBytesToGibibytes)
{
    const auto original = bytes{ 1024.0 * 1024.0 * 1024.0 };
    const auto restored = data_size_cast<bytes>(data_size_cast<gibibytes>(original));
    EXPECT_DOUBLE_EQ(restored.count(), original.count());
}

// Decimal and binary units are separate types, so a decimal value can never be
// silently consumed where a binary one is expected (and vice versa).
static_assert(!std::is_same_v<kilobytes, kibibytes>);
static_assert(!std::is_same_v<megabytes, mebibytes>);
static_assert(!std::is_same_v<gigabytes, gibibytes>);
static_assert(!std::is_same_v<terabytes, tebibytes>);

TEST(UnitsDataSize, DecimalAndBinaryAreDistinctValues)
{
    EXPECT_NE(1_tb, 1_tib);
    EXPECT_GT(1_mib, 1_mb);
    EXPECT_GT(1_gib, 1_gb);
}

// ---------------------------------------------------------------------------
// fmt formatting: frequency
// ---------------------------------------------------------------------------

TEST(UnitsFmtFrequency, PlainHz) { EXPECT_EQ(fmt::format("{}", 50_hz), "50 Hz"); }

TEST(UnitsFmtFrequency, PlainMhz) { EXPECT_EQ(fmt::format("{}", 2_mhz), "2 MHz"); }

TEST(UnitsFmtFrequency, PlainKhz) { EXPECT_EQ(fmt::format("{}", 1_khz), "1 kHz"); }

TEST(UnitsFmtFrequency, PlainGhz) { EXPECT_EQ(fmt::format("{}", 3_ghz), "3 GHz"); }

TEST(UnitsFmtFrequency, Precision)
{
    EXPECT_EQ(fmt::format("{:.2f}", megahertz{ 1.5 }), "1.50 MHz");
}

// ---------------------------------------------------------------------------
// fmt formatting: data_size
// ---------------------------------------------------------------------------

TEST(UnitsFmtDataSize, PlainBytes) { EXPECT_EQ(fmt::format("{}", 512_b), "512 B"); }

TEST(UnitsFmtDataSize, PlainMb) { EXPECT_EQ(fmt::format("{}", 5_mb), "5 MB"); }

TEST(UnitsFmtDataSize, PlainMib) { EXPECT_EQ(fmt::format("{}", 5_mib), "5 MiB"); }

TEST(UnitsFmtDataSize, DecimalSuffixes)
{
    EXPECT_EQ(fmt::format("{}", 1_kb), "1 KB");
    EXPECT_EQ(fmt::format("{}", 1_gb), "1 GB");
    EXPECT_EQ(fmt::format("{}", 1_tb), "1 TB");
}

TEST(UnitsFmtDataSize, BinarySuffixes)
{
    EXPECT_EQ(fmt::format("{}", 1_kib), "1 KiB");
    EXPECT_EQ(fmt::format("{}", 1_gib), "1 GiB");
    EXPECT_EQ(fmt::format("{}", 1_tib), "1 TiB");
}

TEST(UnitsFmtDataSize, Precision)
{
    EXPECT_EQ(fmt::format("{:.2f}", megabytes{ 1.5 }), "1.50 MB");
}

// The inherited fmt::formatter<Rep> pads before the suffix is appended, so a
// width spec sizes the numeric part only and the result is longer than the
// requested width. Pinned here so a change in that layering is caught.
TEST(UnitsFmtDataSize, WidthCoversNumericPartOnly)
{
    EXPECT_EQ(fmt::format("{:>10}", megabytes{ 5.0 }), "         5 MB");
}

// ---------------------------------------------------------------------------
// fmt formatting: std::chrono::duration (owned by fmt, via <fmt/chrono.h>)
// ---------------------------------------------------------------------------

// Regression guard: <spdlog/fmt/chrono.h> must keep coexisting with the unit
// formatters in one TU - an earlier revision made that a hard #error.
TEST(UnitsFmtChrono, PlainMs)
{
    using std::chrono::milliseconds;
    EXPECT_EQ(fmt::format("{}", milliseconds{ 1500 }), "1500ms");
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

static_assert(!std::is_constructible_v<watt, long>);
static_assert(!std::is_constructible_v<watt, unsigned long>);
static_assert(!std::is_constructible_v<watt, long long>);
static_assert(!std::is_constructible_v<watt, unsigned long long>);
static_assert(!std::is_constructible_v<watt, long double>);
static_assert(!std::is_constructible_v<kilowatt, long>);
static_assert(std::is_constructible_v<watt, int>);
static_assert(std::is_constructible_v<watt, float>);
static_assert(std::is_constructible_v<watt, double>);
static_assert(!std::is_convertible_v<double, watt>);

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

TEST(UnitsPower, DefaultConstruction) { EXPECT_DOUBLE_EQ(watt{}.count(), 0.0); }

TEST(UnitsPower, ZeroAndNegative)
{
    EXPECT_DOUBLE_EQ(power_cast<milliwatt>(watt{ 0.0 }).count(), 0.0);
    EXPECT_DOUBLE_EQ(power_cast<watt>(kilowatt{ 0.0 }).count(), 0.0);
    EXPECT_DOUBLE_EQ(power_cast<milliwatt>(watt{ -0.25 }).count(), -250.0);
    EXPECT_DOUBLE_EQ(power_cast<watt>(kilowatt{ -1.5 }).count(), -1500.0);
}

// Exactly the expression sampling.cpp relies on to store watts as nanowatts.
TEST(UnitsPower, RoundTripWattToNanowatt)
{
    const auto original = watt{ 0.25 };
    const auto as_nano  = power_cast<nanowatt>(original);
    EXPECT_DOUBLE_EQ(as_nano.count(), 250'000'000.0);
    EXPECT_DOUBLE_EQ(power_cast<watt>(as_nano).count(), original.count());
}

// ---------------------------------------------------------------------------
// fmt formatting: power
// ---------------------------------------------------------------------------

TEST(UnitsFmtPower, PlainW) { EXPECT_EQ(fmt::format("{}", 1_w), "1 W"); }

TEST(UnitsFmtPower, PlainMw) { EXPECT_EQ(fmt::format("{}", 250_mw), "250 mW"); }

TEST(UnitsFmtPower, SubWattSuffixes)
{
    EXPECT_EQ(fmt::format("{}", 1_nw), "1 nW");
    EXPECT_EQ(fmt::format("{}", 1_uw), "1 uW");
}

TEST(UnitsFmtPower, PlainKw) { EXPECT_EQ(fmt::format("{}", 2_kw), "2 kW"); }

TEST(UnitsFmtPower, Precision)
{
    EXPECT_EQ(fmt::format("{:.2f}", milliwatt{ 1.5 }), "1.50 mW");
}
