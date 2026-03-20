/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "utils.h"
#include <gtest/gtest.h>
#include <cstring>
#include <cstdlib>

namespace RcclUnitTesting {

// ---------------------------------------------------------------------------
// busIdToInt64 / int64ToBusId
// ---------------------------------------------------------------------------

TEST(BusIdConversion, BusIdToInt64ZeroAddress) {
    int64_t id = -1;
    EXPECT_EQ(busIdToInt64("0000:00:00.0", &id), ncclSuccess);
    EXPECT_EQ(id, 0);
}

TEST(BusIdConversion, BusIdToInt64NonZeroBus) {
    // "0000:03:00.0" => strip separators => "000003000" hex = 0x3000
    int64_t id = -1;
    EXPECT_EQ(busIdToInt64("0000:03:00.0", &id), ncclSuccess);
    EXPECT_EQ(id, 0x3000);
}

TEST(BusIdConversion, BusIdToInt64NonZeroDomain) {
    // "0001:0a:1f.3" => "00010a1f3" hex = 0x10a1f3
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

TEST(ParseStringList, MaxListClamps) {
    struct netIf ifList[2];
    // Give 3 interfaces but max is 2
    int n = parseStringList("eth0,eth1,eth2", ifList, 2);
    EXPECT_EQ(n, 2);
    EXPECT_STREQ(ifList[0].prefix, "eth0");
    EXPECT_STREQ(ifList[1].prefix, "eth1");
}

} // namespace RcclUnitTesting
