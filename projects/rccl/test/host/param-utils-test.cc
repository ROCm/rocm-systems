/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/include/param/utils.h and
// src/include/param/parser_common.h: the type-id mapping, flag-string
// formatting, string helpers (iequals/trim/split/stringFormat), the
// ncclParamParser<T> wrapper, and the option-set builders (makeOption(s),
// ncclOptionSetAssertUnique) every typed/enum/bitset/list parser factory
// builds on. The rest of the param subsystem (param.h's ncclParam<T>, the
// registry, the C API, env loading) is covered by follow-up microtests in
// this stacked chain.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "../common/LogCapture.hpp"
#include "param/parser_common.h"
#include "param/utils.h"

namespace {

using RcclUnitTesting::CaptureLog;
using RcclUnitTesting::LogHas;
using nccl::param::utils::flagsStr;
using nccl::param::utils::iequals;
using nccl::param::utils::split;
using nccl::param::utils::srcDefault;
using nccl::param::utils::srcEnvPlugin;
using nccl::param::utils::stringFormat;
using nccl::param::utils::trim;

ncclResult_t ResolveDoubled(const void*, const char* input, int32_t& out) {
  out = std::atoi(input) * 2;
  return ncclSuccess;
}
ncclResult_t ResolveAlwaysFails(const void*, const char*, int32_t&) {
  return ncclInvalidArgument;
}
bool ValidateNonNegative(const void*, const int32_t& val) {
  return val >= 0;
}
std::string ToStringWithSuffix(const void*, const int32_t& val) {
  return std::to_string(val) + "x";
}

// ---------------------------------------------------------------------------
// ncclParamTypeIdOf<T>
// ---------------------------------------------------------------------------

TEST(ParamUtilsMicrotest, TypeIdOf_MapsEveryFixedWidthIntegerAndBoolAndCstr) {
  EXPECT_EQ(NCCL_PARAM_TYPE_I8, ncclParamTypeIdOf<int8_t>());
  EXPECT_EQ(NCCL_PARAM_TYPE_I16, ncclParamTypeIdOf<int16_t>());
  EXPECT_EQ(NCCL_PARAM_TYPE_I32, ncclParamTypeIdOf<int32_t>());
  EXPECT_EQ(NCCL_PARAM_TYPE_I64, ncclParamTypeIdOf<int64_t>());
  EXPECT_EQ(NCCL_PARAM_TYPE_U8, ncclParamTypeIdOf<uint8_t>());
  EXPECT_EQ(NCCL_PARAM_TYPE_U16, ncclParamTypeIdOf<uint16_t>());
  EXPECT_EQ(NCCL_PARAM_TYPE_U32, ncclParamTypeIdOf<uint32_t>());
  EXPECT_EQ(NCCL_PARAM_TYPE_U64, ncclParamTypeIdOf<uint64_t>());
  EXPECT_EQ(NCCL_PARAM_TYPE_BOOL, ncclParamTypeIdOf<bool>());
  EXPECT_EQ(NCCL_PARAM_TYPE_CSTR, ncclParamTypeIdOf<const char*>());
}

TEST(ParamUtilsMicrotest, TypeIdOf_FallsBackToRawForUnlistedTypes) {
  EXPECT_EQ(NCCL_PARAM_TYPE_RAW, ncclParamTypeIdOf<double>());
  EXPECT_EQ(NCCL_PARAM_TYPE_RAW, ncclParamTypeIdOf<std::string>());
  // Plain char is a distinct type from both signed char (int8_t) and unsigned
  // char (uint8_t), so it isn't matched by either is_same check and falls
  // through the same as any other unlisted type.
  EXPECT_EQ(NCCL_PARAM_TYPE_RAW, ncclParamTypeIdOf<char>());
}

// ---------------------------------------------------------------------------
// flagsStr
// ---------------------------------------------------------------------------

TEST(ParamUtilsMicrotest, FlagsStr_PublishedWithNoOtherFlagsIsEmpty) {
  EXPECT_EQ("", flagsStr(NCCL_PARAM_FLAG_PUBLISHED));
}

TEST(ParamUtilsMicrotest, FlagsStr_UnpublishedWithNoOtherFlagsIsPrivate) {
  EXPECT_EQ("Private", flagsStr(NCCL_PARAM_FLAG_NONE));
}

TEST(ParamUtilsMicrotest, FlagsStr_PublishedCachedDropsThePrivatePrefix) {
  EXPECT_EQ("Cached", flagsStr(NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_CACHED));
}

TEST(ParamUtilsMicrotest, FlagsStr_UnpublishedCachedJoinsPrivateWithAComma) {
  EXPECT_EQ("Private, Cached", flagsStr(NCCL_PARAM_FLAG_CACHED));
}

TEST(ParamUtilsMicrotest, FlagsStr_OrdersMultipleFlagsDeprecatedBeforeSensitive) {
  EXPECT_EQ("Deprecated, Sensitive",
            flagsStr(NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_DEPRECATED | NCCL_PARAM_FLAG_SENSITIVE));
}

TEST(ParamUtilsMicrotest, FlagsStr_AllFlagsJoinInDeclarationOrder) {
  uint64_t all = NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_DEPRECATED | NCCL_PARAM_FLAG_CACHED |
                 NCCL_PARAM_FLAG_UNUSED | NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT | NCCL_PARAM_FLAG_SENSITIVE;
  EXPECT_EQ("Deprecated, Cached, Unused, NoEnvPluginInit, Sensitive", flagsStr(all));
}

// ---------------------------------------------------------------------------
// srcDefault / srcEnvPlugin
// ---------------------------------------------------------------------------

TEST(ParamUtilsMicrotest, SrcLabels_AreTheirFixedStrings) {
  EXPECT_STREQ("Default", srcDefault());
  EXPECT_STREQ("EnvPlugin", srcEnvPlugin());
}

// ---------------------------------------------------------------------------
// stringFormat
// ---------------------------------------------------------------------------

TEST(ParamUtilsMicrotest, StringFormat_SubstitutesArguments) {
  EXPECT_EQ("key=NCCL_DEBUG value=3", stringFormat("key=%s value=%d", "NCCL_DEBUG", 3));
}

TEST(ParamUtilsMicrotest, StringFormat_NoSubstitutionsReturnsTheLiteral) {
  EXPECT_EQ("literal text", stringFormat("literal text"));
}

TEST(ParamUtilsMicrotest, StringFormat_HandlesOutputLongerThanATypicalStackBuffer) {
  std::string longArg(600, 'x');
  EXPECT_EQ(longArg, stringFormat("%s", longArg.c_str()));
}

// ---------------------------------------------------------------------------
// iequals
// ---------------------------------------------------------------------------

TEST(ParamUtilsMicrotest, Iequals_MatchesIgnoringAsciiCase) {
  EXPECT_TRUE(iequals("TRUE", "true"));
  EXPECT_TRUE(iequals("MiXeD", "mixed"));
}

TEST(ParamUtilsMicrotest, Iequals_RejectsDifferentLength) {
  EXPECT_FALSE(iequals("abc", "ab"));
}

TEST(ParamUtilsMicrotest, Iequals_RejectsSameLengthDifferentContent) {
  EXPECT_FALSE(iequals("abc", "abd"));
}

TEST(ParamUtilsMicrotest, Iequals_EmptyStringsMatch) {
  EXPECT_TRUE(iequals("", ""));
}

// ---------------------------------------------------------------------------
// trim
// ---------------------------------------------------------------------------

TEST(ParamUtilsMicrotest, Trim_RemovesLeadingAndTrailingWhitespace) {
  EXPECT_EQ("value", trim("  \tvalue \t "));
}

TEST(ParamUtilsMicrotest, Trim_AllWhitespaceCollapsesToEmpty) {
  EXPECT_EQ("", trim("   \t  "));
}

TEST(ParamUtilsMicrotest, Trim_NoWhitespaceIsUnchanged) {
  EXPECT_EQ("value", trim("value"));
}

TEST(ParamUtilsMicrotest, Trim_EmptyInputStaysEmpty) {
  EXPECT_EQ("", trim(""));
}

// ---------------------------------------------------------------------------
// split
// ---------------------------------------------------------------------------

TEST(ParamUtilsMicrotest, Split_SeparatesOnDelimiter) {
  std::vector<std::string> expected{"a", "b", "c"};
  EXPECT_EQ(expected, split("a,b,c", ','));
}

TEST(ParamUtilsMicrotest, Split_PreservesEmptyTokensFromAdjacentDelimiters) {
  std::vector<std::string> expected{"a", "", "b"};
  EXPECT_EQ(expected, split("a,,b", ','));
}

TEST(ParamUtilsMicrotest, Split_LeadingAndTrailingDelimiterYieldEmptyEnds) {
  std::vector<std::string> expected{"", "a", ""};
  EXPECT_EQ(expected, split(",a,", ','));
}

TEST(ParamUtilsMicrotest, Split_NoDelimiterYieldsTheWholeString) {
  std::vector<std::string> expected{"solo"};
  EXPECT_EQ(expected, split("solo", ','));
}

TEST(ParamUtilsMicrotest, Split_EmptyInputYieldsOneEmptyToken) {
  std::vector<std::string> expected{""};
  EXPECT_EQ(expected, split("", ','));
}

// ---------------------------------------------------------------------------
// ncclParamParser<T>: resolve/validate/toString delegation + explicit bool
// ---------------------------------------------------------------------------

TEST(ParamUtilsMicrotest, Parser_DelegatesThroughTheFunctionPointerTrio) {
  ncclParamParser<int32_t> parser{ResolveDoubled, ValidateNonNegative, ToStringWithSuffix, nullptr, "doubled"};

  int32_t value = 0;
  EXPECT_EQ(ncclSuccess, parser.resolve("21", value));
  EXPECT_EQ(42, value);
  EXPECT_TRUE(parser.validate(42));
  EXPECT_FALSE(parser.validate(-1));
  EXPECT_EQ("42x", parser.toString(42));
}

TEST(ParamUtilsMicrotest, Parser_ResolvePropagatesTheCalleesFailure) {
  ncclParamParser<int32_t> parser{ResolveAlwaysFails, ValidateNonNegative, ToStringWithSuffix, nullptr, "fails"};
  int32_t value = 0;
  EXPECT_EQ(ncclInvalidArgument, parser.resolve("21", value));
}

TEST(ParamUtilsMicrotest, Parser_DefaultConstructedIsFalsy) {
  ncclParamParser<int32_t> parser{};
  EXPECT_FALSE(static_cast<bool>(parser));
}

TEST(ParamUtilsMicrotest, Parser_ConstructedWithAllFieldsIsTruthy) {
  ncclParamParser<int32_t> parser{ResolveDoubled, ValidateNonNegative, ToStringWithSuffix, nullptr, "doubled"};
  EXPECT_TRUE(static_cast<bool>(parser));
}

// ---------------------------------------------------------------------------
// makeOption / makeOptions / ncclOptionSetAssertUnique
// ---------------------------------------------------------------------------

TEST(ParamUtilsMicrotest, MakeOption_TwoArgLeavesDescriptionNull) {
  auto opt = makeOption<int32_t>("OFF", 0);
  EXPECT_STREQ("OFF", opt.name);
  EXPECT_EQ(0, opt.value);
  EXPECT_EQ(nullptr, opt.desc);
}

TEST(ParamUtilsMicrotest, MakeOption_ThreeArgKeepsTheDescription) {
  auto opt = makeOption<int32_t>("ON", 1, "Enable feature");
  EXPECT_STREQ("Enable feature", opt.desc);
}

TEST(ParamUtilsMicrotest, MakeOptions_PreservesOrderAndSize) {
  auto opts =
      makeOptions(makeOption<int32_t>("OFF", 0), makeOption<int32_t>("ON", 1), makeOption<int32_t>("AUTO", 2));
  ASSERT_EQ(3u, opts.size());
  EXPECT_STREQ("OFF", opts.options[0].name);
  EXPECT_EQ(0, opts.options[0].value);
  EXPECT_STREQ("ON", opts.options[1].name);
  EXPECT_EQ(1, opts.options[1].value);
  EXPECT_STREQ("AUTO", opts.options[2].name);
  EXPECT_EQ(2, opts.options[2].value);

  // begin()/end() walk the same backing array in the same order as indexing.
  const std::vector<int32_t> expectedValues{0, 1, 2};
  int seen = 0;
  for (const auto& opt : opts) {
    ASSERT_LT(seen, static_cast<int>(expectedValues.size()));
    EXPECT_EQ(expectedValues[seen], opt.value);
    ++seen;
  }
  EXPECT_EQ(3, seen);
}

TEST(ParamUtilsMicrotest, MakeOptions_UniqueNamesLogNoWarning) {
  std::string log = CaptureLog([&]() {
    auto opts = makeOptions(makeOption<int32_t>("OFF", 0), makeOption<int32_t>("ON", 1));
    (void)opts;
  });
  EXPECT_FALSE(LogHas(log, "Duplicate option name"));
}

TEST(ParamUtilsMicrotest, MakeOptions_DuplicateNameWarns) {
  std::string log = CaptureLog([&]() {
    auto opts = makeOptions(makeOption<int32_t>("ON", 1), makeOption<int32_t>("ON", 2));
    (void)opts;
  });
  EXPECT_TRUE(LogHas(log, "Duplicate option name \"ON\""));
}

}  // namespace
