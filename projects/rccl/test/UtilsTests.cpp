/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "utils.h"
#include <gtest/gtest.h>

namespace RcclUnitTesting {

// ---------------------------------------------------------------------------
// busIdToInt64 / int64ToBusId
//
// Expected values below follow the encoding implemented by busIdToInt64 in
// src/misc/utils.cc (the source of truth): the ':' and '.' separators are
// stripped from the PCI address DDDD:BB:DD.F, and the remaining hex digits are
// concatenated and parsed as a single integer via strtol(..., 16).
//
//   "0000:03:00.0" -> "000003000" -> 0x3000
//   "0001:0a:1f.3" -> "00010a1f3" -> 0x10a1f3
//
// int64ToBusId is the inverse, laying the integer back out as
// domain[35:20] bus[19:12] device[11:4] function[3:0].
// ---------------------------------------------------------------------------

TEST(BusIdConversion, BusIdToInt64ZeroAddress) {
    int64_t id = -1;
    EXPECT_EQ(busIdToInt64("0000:00:00.0", &id), ncclSuccess);
    EXPECT_EQ(id, 0);
}

TEST(BusIdConversion, BusIdToInt64NonZeroBus) {
    int64_t id = -1;
    EXPECT_EQ(busIdToInt64("0000:03:00.0", &id), ncclSuccess);
    EXPECT_EQ(id, 0x3000);
}

TEST(BusIdConversion, BusIdToInt64NonZeroDomain) {
    int64_t id = -1;
    EXPECT_EQ(busIdToInt64("0001:0a:1f.3", &id), ncclSuccess);
    EXPECT_EQ(id, 0x10a1f3);
}

TEST(BusIdConversion, Int64ToBusIdZero) {
    char busId[32] = {};
    EXPECT_EQ(int64ToBusId(0, busId), ncclSuccess);
    EXPECT_STREQ(busId, "0000:00:00.0");
}

TEST(BusIdConversion, Int64ToBusIdNonZeroBus) {
    char busId[32] = {};
    EXPECT_EQ(int64ToBusId(0x3000, busId), ncclSuccess);
    EXPECT_STREQ(busId, "0000:03:00.0");
}

TEST(BusIdConversion, RoundTripBusId) {
    const char* inputs[] = {
        "0000:00:00.0",
        "0000:03:00.0",
        "0001:0a:1f.3",
        "ffff:ff:1f.7",
    };
    for (const char* input : inputs) {
        int64_t id = -1;
        char output[32] = {};
        EXPECT_EQ(busIdToInt64(input, &id), ncclSuccess) << "input=" << input;
        EXPECT_EQ(int64ToBusId(id, output), ncclSuccess) << "input=" << input;
        EXPECT_STREQ(output, input) << "round-trip failed for " << input;
    }
}

// busIdToInt64 does not validate its input: it performs strtol-style parsing
// that stops at the first character which is neither a hex digit nor a ':'/'.'
// separator, then parses whatever digits it collected. It therefore always
// returns ncclSuccess, even for input that is not a PCI bus ID at all. The
// tests below pin that *observed* behavior (traced through src/misc/utils.cc)
// rather than asserting a rejection that the implementation never performs.

TEST(BusIdConversion, BusIdToInt64NonBusIdStringYieldsZero) {
    // 'n' is not a hex digit, so parsing stops immediately and the collected
    // (empty) digit string parses as 0.
    int64_t id = -1;
    EXPECT_EQ(busIdToInt64("not-a-bus-id", &id), ncclSuccess);
    EXPECT_EQ(id, 0);
}

TEST(BusIdConversion, BusIdToInt64EmptyStringYieldsZero) {
    int64_t id = -1;
    EXPECT_EQ(busIdToInt64("", &id), ncclSuccess);
    EXPECT_EQ(id, 0);
}

TEST(BusIdConversion, BusIdToInt64MissingSeparatorsParsesSameAsSeparated) {
    // Separators are only skipped, never required, so the unseparated form of
    // "0000:03:00.0" collects the identical digit string and yields 0x3000.
    int64_t id = -1;
    EXPECT_EQ(busIdToInt64("000003000", &id), ncclSuccess);
    EXPECT_EQ(id, 0x3000);
}

TEST(BusIdConversion, BusIdToInt64StopsAtGarbageTail) {
    // Parsing halts at 'z'; the valid prefix still yields 0x3000 and the
    // trailing garbage is silently ignored.
    int64_t id = -1;
    EXPECT_EQ(busIdToInt64("0000:03:00.0zzz", &id), ncclSuccess);
    EXPECT_EQ(id, 0x3000);
}

// ---------------------------------------------------------------------------
// parseStringList
// ---------------------------------------------------------------------------

TEST(ParseStringList, NullInputReturnsZero) {
    struct netIf ifList[8];
    EXPECT_EQ(parseStringList(nullptr, ifList, 8), 0);
}

TEST(ParseStringList, EmptyStringReturnsZero) {
    struct netIf ifList[8];
    EXPECT_EQ(parseStringList("", ifList, 8), 0);
}

TEST(ParseStringList, SingleInterface) {
    struct netIf ifList[8];
    int n = parseStringList("eth0", ifList, 8);
    EXPECT_EQ(n, 1);
    EXPECT_STREQ(ifList[0].prefix, "eth0");
    EXPECT_EQ(ifList[0].port, -1);
}

TEST(ParseStringList, InterfaceWithPort) {
    struct netIf ifList[8];
    int n = parseStringList("eth0:1", ifList, 8);
    EXPECT_EQ(n, 1);
    EXPECT_STREQ(ifList[0].prefix, "eth0");
    EXPECT_EQ(ifList[0].port, 1);
}

TEST(ParseStringList, MultipleInterfaces) {
    struct netIf ifList[8];
    int n = parseStringList("eth0,eth1,ib0", ifList, 8);
    EXPECT_EQ(n, 3);
    EXPECT_STREQ(ifList[0].prefix, "eth0");
    EXPECT_EQ(ifList[0].port, -1);
    EXPECT_STREQ(ifList[1].prefix, "eth1");
    EXPECT_EQ(ifList[1].port, -1);
    EXPECT_STREQ(ifList[2].prefix, "ib0");
    EXPECT_EQ(ifList[2].port, -1);
}

TEST(ParseStringList, MixedPortAndNoPort) {
    struct netIf ifList[8];
    int n = parseStringList("eth0:1,eth1", ifList, 8);
    EXPECT_EQ(n, 2);
    EXPECT_STREQ(ifList[0].prefix, "eth0");
    EXPECT_EQ(ifList[0].port, 1);
    EXPECT_STREQ(ifList[1].prefix, "eth1");
    EXPECT_EQ(ifList[1].port, -1);
}

// parseStringList only commits an entry when it has accumulated at least one
// prefix character, so runs of commas simply produce no entry rather than an
// empty one. It does no whitespace trimming: a space after a comma becomes part
// of the next prefix. Both behaviors are confirmed from src/misc/utils.cc.

TEST(ParseStringList, LeadingAndTrailingCommasProduceNoEmptyEntries) {
    struct netIf ifList[8];
    int n = parseStringList(",eth0,", ifList, 8);
    EXPECT_EQ(n, 1);
    EXPECT_STREQ(ifList[0].prefix, "eth0");
    EXPECT_EQ(ifList[0].port, -1);
}

TEST(ParseStringList, DoubleCommasProduceNoEmptyEntries) {
    struct netIf ifList[8];
    int n = parseStringList("eth0,,eth1", ifList, 8);
    EXPECT_EQ(n, 2);
    EXPECT_STREQ(ifList[0].prefix, "eth0");
    EXPECT_STREQ(ifList[1].prefix, "eth1");
}

TEST(ParseStringList, WhitespaceAfterCommaIsNotTrimmed) {
    struct netIf ifList[8];
    int n = parseStringList("eth0, eth1", ifList, 8);
    EXPECT_EQ(n, 2);
    EXPECT_STREQ(ifList[0].prefix, "eth0");
    // The space is treated as an ordinary prefix character, so it is retained.
    EXPECT_STREQ(ifList[1].prefix, " eth1");
}

TEST(ParseStringList, MaxListClamps) {
    struct netIf ifList[2];
    // Give 3 interfaces but max is 2
    int n = parseStringList("eth0,eth1,eth2", ifList, 2);
    EXPECT_EQ(n, 2);
    EXPECT_STREQ(ifList[0].prefix, "eth0");
    EXPECT_STREQ(ifList[1].prefix, "eth1");
}

} // namespace RcclUnitTesting
