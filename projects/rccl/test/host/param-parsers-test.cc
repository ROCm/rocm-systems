/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/include/param/parser_default.h,
// src/include/param/parser_enum.h, src/include/param/parser_bitset.h, and
// src/include/param/parser_list.h: the four parser factories that
// src/include/param/param.h wires every DEFINE_NCCL_PARAM up to. Second PR
// in the AICOMRCCL-2820 stacked chain; see param-utils-test.cc for the
// shared ncclParamParser<T> wrapper and option-set builders these factories
// return/consume.

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

#include "param/parser_bitset.h"
#include "param/parser_common.h"
#include "param/parser_default.h"
#include "param/parser_enum.h"
#include "param/parser_list.h"

namespace {

// ---------------------------------------------------------------------------
// ncclParamParserDefault<T>: bool
// ---------------------------------------------------------------------------

TEST(ParamParsersMicrotest, Bool_ResolvesEveryAcceptedSpelling) {
  bool out = false;
  for (const char* truthy : {"1", "T", "t", "TRUE", "true", "TrUe"}) {
    out = false;
    EXPECT_EQ(ncclSuccess, ncclParamParserDefault<bool>::resolve(truthy, out)) << truthy;
    EXPECT_TRUE(out) << truthy;
  }
  for (const char* falsy : {"0", "F", "f", "FALSE", "false", "FaLsE"}) {
    out = true;
    EXPECT_EQ(ncclSuccess, ncclParamParserDefault<bool>::resolve(falsy, out)) << falsy;
    EXPECT_FALSE(out) << falsy;
  }
}

TEST(ParamParsersMicrotest, Bool_RejectsNullAndUnrecognizedSpellings) {
  bool out = false;
  EXPECT_EQ(ncclInvalidArgument, ncclParamParserDefault<bool>::resolve(nullptr, out));
  EXPECT_EQ(ncclInvalidArgument, ncclParamParserDefault<bool>::resolve("yes", out));
  EXPECT_EQ(ncclInvalidArgument, ncclParamParserDefault<bool>::resolve("", out));
}

TEST(ParamParsersMicrotest, Bool_ValidateAcceptsBothValues) {
  EXPECT_TRUE(ncclParamParserDefault<bool>::validate(true));
  EXPECT_TRUE(ncclParamParserDefault<bool>::validate(false));
}

TEST(ParamParsersMicrotest, Bool_ToStringIsUppercase) {
  EXPECT_EQ("TRUE", ncclParamParserDefault<bool>::toString(true));
  EXPECT_EQ("FALSE", ncclParamParserDefault<bool>::toString(false));
}

TEST(ParamParsersMicrotest, Bool_DescribesAcceptedSpellings) {
  EXPECT_STREQ("Boolean: 1/T/TRUE or 0/F/FALSE", ncclParamParserDefault<bool>::desc);
}

// ---------------------------------------------------------------------------
// ncclParamParserDefault<T>: const char*
// ---------------------------------------------------------------------------

TEST(ParamParsersMicrotest, Cstr_ResolvePassesThePointerThrough) {
  const char* out = nullptr;
  EXPECT_EQ(ncclSuccess, ncclParamParserDefault<const char*>::resolve("hello", out));
  EXPECT_STREQ("hello", out);
}

TEST(ParamParsersMicrotest, Cstr_ValidateAlwaysTrue) {
  EXPECT_TRUE(ncclParamParserDefault<const char*>::validate("anything"));
  EXPECT_TRUE(ncclParamParserDefault<const char*>::validate(nullptr));
}

TEST(ParamParsersMicrotest, Cstr_ToStringEmptyForNull) {
  EXPECT_EQ("", ncclParamParserDefault<const char*>::toString(nullptr));
  EXPECT_EQ("hello", ncclParamParserDefault<const char*>::toString("hello"));
}

// ---------------------------------------------------------------------------
// ncclIntegerParser / ncclParamParserDefault<integer types>
// ---------------------------------------------------------------------------

TEST(ParamParsersMicrotest, Integer_ResolvesSignedAndUnsigned) {
  int32_t s = 0;
  EXPECT_EQ(ncclSuccess, ncclParamParserDefault<int32_t>::resolve("-42", s));
  EXPECT_EQ(-42, s);

  uint32_t u = 0;
  EXPECT_EQ(ncclSuccess, ncclParamParserDefault<uint32_t>::resolve("42", u));
  EXPECT_EQ(42u, u);
}

TEST(ParamParsersMicrotest, Integer_RejectsNullEmptyAndTrailingGarbage) {
  int32_t out = 0;
  EXPECT_EQ(ncclInvalidArgument, ncclParamParserDefault<int32_t>::resolve(nullptr, out));
  EXPECT_EQ(ncclInvalidArgument, ncclParamParserDefault<int32_t>::resolve("", out));
  EXPECT_EQ(ncclInvalidArgument, ncclParamParserDefault<int32_t>::resolve("12abc", out));
  EXPECT_EQ(ncclInvalidArgument, ncclParamParserDefault<int32_t>::resolve("abc", out));
}

TEST(ParamParsersMicrotest, Integer_RejectsOverflowOfTheIntermediateLongLong) {
  // Exercises strtoll/strtoull's own ERANGE, as distinct from the T-narrowing
  // truncation in Integer_NarrowerTypeTruncatesRatherThanErrors below: this
  // magnitude overflows (unsigned) long long itself, not just int32_t/uint32_t.
  int32_t signedOut = 0;
  EXPECT_EQ(ncclInvalidArgument,
            ncclParamParserDefault<int32_t>::resolve("999999999999999999999999999", signedOut));
  uint32_t unsignedOut = 0;
  EXPECT_EQ(ncclInvalidArgument,
            ncclParamParserDefault<uint32_t>::resolve("999999999999999999999999999", unsignedOut));
}

TEST(ParamParsersMicrotest, Integer_NegativeUnsignedWrapsRatherThanErrors) {
  // strtoull() accepts a leading '-' and negates the parsed magnitude rather
  // than rejecting it; pinning that passthrough behavior here since
  // ncclIntegerParser<uint32_t>::resolve() does not special-case the sign.
  uint32_t out = 0;
  EXPECT_EQ(ncclSuccess, ncclParamParserDefault<uint32_t>::resolve("-1", out));
  EXPECT_EQ(0xFFFFFFFFu, out);
}

TEST(ParamParsersMicrotest, Integer_NarrowerTypeTruncatesRatherThanErrors) {
  // resolve() only checks errno/ERANGE on the intermediate long long, not
  // against T's own range; a value that overflows int8_t but not long long
  // silently truncates via static_cast, matching current production behavior.
  int8_t out = 0;
  EXPECT_EQ(ncclSuccess, ncclParamParserDefault<int8_t>::resolve("200", out));
  EXPECT_EQ(static_cast<int8_t>(200), out);
}

TEST(ParamParsersMicrotest, Integer_ValidateAcceptsAnInRangeValue) {
  EXPECT_TRUE(ncclParamParserDefault<int32_t>::validate(0));
  EXPECT_TRUE(ncclParamParserDefault<uint32_t>::validate(0));
}

TEST(ParamParsersMicrotest, Integer_ToStringMatchesStdToString) {
  EXPECT_EQ(std::to_string(-7), ncclParamParserDefault<int32_t>::toString(-7));
  EXPECT_EQ(std::to_string(7u), ncclParamParserDefault<uint32_t>::toString(7u));
}

// ---------------------------------------------------------------------------
// ncclParamParserDefault<T>: primary template (unsupported types)
// ---------------------------------------------------------------------------

TEST(ParamParsersMicrotest, UnsupportedType_AlwaysFailsAndDescribesItself) {
  double out = 0.0;
  EXPECT_EQ(ncclInvalidArgument, ncclParamParserDefault<double>::resolve("1.5", out));
  EXPECT_FALSE(ncclParamParserDefault<double>::validate(1.5));
  EXPECT_EQ("<unsupported>", ncclParamParserDefault<double>::toString(1.5));
  EXPECT_STREQ("Unsupported parser", ncclParamParserDefault<double>::desc);
}

// ---------------------------------------------------------------------------
// ncclParamDefault<T> factory
// ---------------------------------------------------------------------------

TEST(ParamParsersMicrotest, DefaultFactory_WiresTheMatchingSpecialization) {
  const ncclParamParser<bool>& parser = ncclParamDefault<bool>();
  bool out = false;
  EXPECT_EQ(ncclSuccess, parser.resolve("TRUE", out));
  EXPECT_TRUE(out);
  EXPECT_TRUE(parser.validate(out));
  EXPECT_EQ("TRUE", parser.toString(true));
  EXPECT_TRUE(static_cast<bool>(parser));
}

TEST(ParamParsersMicrotest, DefaultFactory_ReturnsTheSameInstanceEveryCall) {
  EXPECT_EQ(&ncclParamDefault<int32_t>(), &ncclParamDefault<int32_t>());
}

// ---------------------------------------------------------------------------
// ncclParamBounded<T>
// ---------------------------------------------------------------------------

TEST(ParamParsersMicrotest, Bounded_ValidatesWithinClosedRangeOnly) {
  ncclParamParser<int32_t> parser = ncclParamBounded<int32_t>(1, 10);
  EXPECT_TRUE(parser.validate(1));
  EXPECT_TRUE(parser.validate(10));
  EXPECT_TRUE(parser.validate(5));
  EXPECT_FALSE(parser.validate(0));
  EXPECT_FALSE(parser.validate(11));
}

TEST(ParamParsersMicrotest, Bounded_ResolveAndToStringDelegateToTheDefaultParser) {
  ncclParamParser<int32_t> parser = ncclParamBounded<int32_t>(1, 10);
  int32_t out = 0;
  EXPECT_EQ(ncclSuccess, parser.resolve("7", out));
  EXPECT_EQ(7, out);
  EXPECT_EQ("7", parser.toString(7));
}

TEST(ParamParsersMicrotest, Bounded_DescribesTheClosedRange) {
  ncclParamParser<int32_t> parser = ncclParamBounded<int32_t>(1, 10);
  EXPECT_EQ("Integer in range [1, 10]", parser.desc);
}

TEST(ParamParsersMicrotest, Bounded_SingleArgOverloadDefaultsUpperToTypeMax) {
  ncclParamParser<uint8_t> parser = ncclParamBounded<uint8_t>(5);
  EXPECT_FALSE(parser.validate(4));
  EXPECT_TRUE(parser.validate(5));
  EXPECT_TRUE(parser.validate(std::numeric_limits<uint8_t>::max()));
}

// ---------------------------------------------------------------------------
// ncclParamOneOf (parser_enum.h)
// ---------------------------------------------------------------------------

enum class Mode : int32_t { Off = 0, On = 1, Auto = 2 };

ncclParamParser<Mode> MakeModeParser() {
  auto opts = makeOptions(makeOption<Mode>("OFF", Mode::Off, "Disable the feature"),
                          makeOption<Mode>("ON", Mode::On, "Enable the feature"),
                          makeOption<Mode>("AUTO", Mode::Auto));
  return ncclParamOneOf(opts);
}

TEST(ParamParsersMicrotest, OneOf_ResolveMatchesCaseInsensitivelyAndTrims) {
  auto parser = MakeModeParser();
  Mode out = Mode::Off;
  EXPECT_EQ(ncclSuccess, parser.resolve("on", out));
  EXPECT_EQ(Mode::On, out);
  EXPECT_EQ(ncclSuccess, parser.resolve("  AUTO  ", out));
  EXPECT_EQ(Mode::Auto, out);
}

TEST(ParamParsersMicrotest, OneOf_RejectsNullEmptyAndUnknownTokens) {
  auto parser = MakeModeParser();
  Mode out = Mode::Off;
  EXPECT_EQ(ncclInvalidArgument, parser.resolve(nullptr, out));
  EXPECT_EQ(ncclInvalidArgument, parser.resolve("   ", out));
  EXPECT_EQ(ncclInvalidArgument, parser.resolve("MAYBE", out));
}

TEST(ParamParsersMicrotest, OneOf_ValidateAlwaysTrue) {
  auto parser = MakeModeParser();
  EXPECT_TRUE(parser.validate(Mode::On));
}

TEST(ParamParsersMicrotest, OneOf_ToStringFindsTheNameByValue) {
  auto parser = MakeModeParser();
  EXPECT_EQ("ON", parser.toString(Mode::On));
}

TEST(ParamParsersMicrotest, OneOf_ToStringOfAnUnlistedValueIsUnknown) {
  auto parser = MakeModeParser();
  EXPECT_EQ("<unknown>", parser.toString(static_cast<Mode>(99)));
}

TEST(ParamParsersMicrotest, OneOf_DescribesEachOptionWithOrWithoutADescription) {
  auto parser = MakeModeParser();
  EXPECT_EQ(
      "One of:\n        OFF - Disable the feature\n        ON - Enable the feature\n        AUTO",
      parser.desc);
}

// ---------------------------------------------------------------------------
// ncclParamBitsetOf (parser_bitset.h)
// ---------------------------------------------------------------------------

enum class Flag : uint32_t { None = 0, A = 1, B = 2, C = 4, AB = 3 };

ncclParamParser<uint32_t> MakeFlagParser(bool ignoreUnknown = false) {
  // NONE is a real, explicitly listed zero-valued option (mirroring how
  // production bitset params commonly spell their all-clear alias), so the
  // decomposition loop's optVal != 0 / isSingleBit(0) short-circuits run for
  // every toString() call, not just the exact-match "NONE" case below.
  auto opts = makeOptions(makeOption<Flag>("NONE", Flag::None), makeOption<Flag>("A", Flag::A),
                          makeOption<Flag>("B", Flag::B), makeOption<Flag>("C", Flag::C),
                          makeOption<Flag>("AB", Flag::AB));
  return ncclParamBitsetOf<Flag, uint32_t>(opts, ',', ignoreUnknown);
}

TEST(ParamParsersMicrotest, Bitset_ResolveOrsEachCommaSeparatedBit) {
  auto parser = MakeFlagParser();
  uint32_t out = 0;
  EXPECT_EQ(ncclSuccess, parser.resolve("A,C", out));
  EXPECT_EQ(5u, out);
}

TEST(ParamParsersMicrotest, Bitset_ResolveTrimsAndSkipsEmptyTokens) {
  auto parser = MakeFlagParser();
  uint32_t out = 0;
  EXPECT_EQ(ncclSuccess, parser.resolve(" A , ,B ", out));
  EXPECT_EQ(3u, out);
}

TEST(ParamParsersMicrotest, Bitset_CaretPrefixInvertsTheMask) {
  auto parser = MakeFlagParser();
  uint32_t out = 0;
  // ^A starts from all-bits-set and clears only A's single bit, not any
  // composite alias that happens to include it.
  EXPECT_EQ(ncclSuccess, parser.resolve("^A", out));
  EXPECT_EQ(~uint32_t(0) & ~uint32_t(1), out);
}

TEST(ParamParsersMicrotest, Bitset_UnknownTokenIsRejectedByDefault) {
  auto parser = MakeFlagParser(/*ignoreUnknown=*/false);
  uint32_t out = 0;
  EXPECT_EQ(ncclInvalidArgument, parser.resolve("A,ZZZ", out));
}

TEST(ParamParsersMicrotest, Bitset_UnknownTokenIsSkippedWhenIgnored) {
  auto parser = MakeFlagParser(/*ignoreUnknown=*/true);
  uint32_t out = 0;
  EXPECT_EQ(ncclSuccess, parser.resolve("A,ZZZ,C", out));
  EXPECT_EQ(5u, out);
}

TEST(ParamParsersMicrotest, Bitset_RejectsNullInput) {
  auto parser = MakeFlagParser();
  uint32_t out = 0;
  EXPECT_EQ(ncclInvalidArgument, parser.resolve(nullptr, out));
}

TEST(ParamParsersMicrotest, Bitset_EmptyStringResolvesToZeroRatherThanErroring) {
  // Unlike nullptr, "" passes the null check and the token loop simply never
  // finds a non-empty token to OR in, so this succeeds with no bits set --
  // distinct from a rejected unknown token.
  auto parser = MakeFlagParser();
  uint32_t out = 0xFFu;
  EXPECT_EQ(ncclSuccess, parser.resolve("", out));
  EXPECT_EQ(0u, out);
}

TEST(ParamParsersMicrotest, Bitset_ValidateAlwaysTrue) {
  auto parser = MakeFlagParser();
  EXPECT_TRUE(parser.validate(0xFFu));
}

TEST(ParamParsersMicrotest, Bitset_ToStringPrefersAnExactCompositeMatch) {
  auto parser = MakeFlagParser();
  // 3 (A|B) is itself a listed option (AB); the exact-match pass finds it
  // before decomposition ever runs.
  EXPECT_EQ("AB", parser.toString(3u));
}

TEST(ParamParsersMicrotest, Bitset_ToStringDecomposesAnUnlistedCombination) {
  auto parser = MakeFlagParser();
  // 5 (A|C) has no listed alias, so it decomposes into single-bit options,
  // skipping AB (not a single bit) even though AB's own bit (A) is set.
  EXPECT_EQ("A,C", parser.toString(5u));
}

TEST(ParamParsersMicrotest, Bitset_ToStringOfZeroIsNone) {
  auto parser = MakeFlagParser();
  EXPECT_EQ("NONE", parser.toString(0u));
}

// ---------------------------------------------------------------------------
// ncclParamListOf (parser_list.h)
// ---------------------------------------------------------------------------

TEST(ParamParsersMicrotest, ListOf_ResolveIntoAVectorPreservesOrder) {
  auto parser = ncclParamListOf<std::vector<std::string>>(',');
  std::vector<std::string> out;
  EXPECT_EQ(ncclSuccess, parser.resolve("one,two,three", out));
  EXPECT_EQ((std::vector<std::string>{"one", "two", "three"}), out);
}

TEST(ParamParsersMicrotest, ListOf_ResolveTrimsAndSkipsEmptyTokens) {
  auto parser = ncclParamListOf<std::vector<std::string>>(',');
  std::vector<std::string> out;
  EXPECT_EQ(ncclSuccess, parser.resolve(" one , , two ", out));
  EXPECT_EQ((std::vector<std::string>{"one", "two"}), out);
}

TEST(ParamParsersMicrotest, ListOf_ResolveIntoASetDeduplicates) {
  auto parser = ncclParamListOf<std::unordered_set<std::string>>(',');
  std::unordered_set<std::string> out;
  EXPECT_EQ(ncclSuccess, parser.resolve("ALL,all,ALL", out));
  std::unordered_set<std::string> expected{"ALL", "all"};
  EXPECT_EQ(expected, out);
}

TEST(ParamParsersMicrotest, ListOf_RejectsNullAndEmptyInput) {
  auto parser = ncclParamListOf<std::vector<std::string>>(',');
  std::vector<std::string> out;
  EXPECT_EQ(ncclInvalidArgument, parser.resolve(nullptr, out));
  EXPECT_EQ(ncclInvalidArgument, parser.resolve("", out));
}

TEST(ParamParsersMicrotest, ListOf_ValidateAlwaysTrue) {
  auto parser = ncclParamListOf<std::vector<std::string>>(',');
  std::vector<std::string> any{"x"};
  EXPECT_TRUE(parser.validate(any));
}

TEST(ParamParsersMicrotest, ListOf_ToStringJoinsWithTheDelimiter) {
  auto parser = ncclParamListOf<std::vector<std::string>>(',');
  EXPECT_EQ("one,two,three", parser.toString({"one", "two", "three"}));
}

TEST(ParamParsersMicrotest, ListOf_DescribesItsDelimiter) {
  auto parser = ncclParamListOf<std::vector<std::string>>(';');
  EXPECT_EQ("Delimiter-separated list (delimiter=';')", parser.desc);
}

}  // namespace
