/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Regression tests for NCCL_PROTO / NCCL_ALGO prefix parsing in graph/tuning.cc.
//
// ncclFuncStr[] must stay aligned with ncclFunc_t enum order: parseList() resolves
// a prefix like "broadcast" to an index p and writes protoEnable[p * nelems + e].
// When ncclFuncStr[ncclFuncBroadcast] was not "Broadcast" (pre-fix ordering placed
// Broadcast at index 5 while NCCL_NUM_FUNCTIONS is 5), NCCL_PROTO=broadcast:simple
// could not match any prefix and init failed or misconfigured protocol enablement.

#include "device.h"
#include "gtest/gtest.h"
#include "nccl_common.h"
#include "plugin/nccl_tuner.h"

#include <cstring>

// parseList() lives in graph/tuning.cc and is exported from librccl in Debug builds.
ncclResult_t parseList(const char* str, const char* prefixElems[], int nprefixes, const char* elems[],
                       int nelems, int* list);

namespace RcclUnitTesting
{
namespace
{

constexpr const char* kCoreFuncNames[NCCL_NUM_FUNCTIONS] = {
    "Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce",
};

TEST(TuningParseList, NcclFuncStrMatchesEnumOrder)
{
    for(int f = 0; f < NCCL_NUM_FUNCTIONS; ++f)
    {
        ASSERT_NE(ncclFuncStr[f], nullptr);
        EXPECT_STRCASEEQ(ncclFuncStr[f], kCoreFuncNames[f])
            << "ncclFuncStr[" << f << "] must match ncclFunc_t value " << f;
    }
}

TEST(TuningParseList, BroadcastSimpleSelectsBroadcastRow)
{
    int protoEnable[NCCL_NUM_FUNCTIONS * NCCL_NUM_PROTOCOLS];
    for(int i = 0; i < NCCL_NUM_FUNCTIONS * NCCL_NUM_PROTOCOLS; ++i)
        protoEnable[i] = 1;

    ASSERT_EQ(parseList("broadcast:simple", ncclFuncStr, NCCL_NUM_FUNCTIONS, ncclProtoStr, NCCL_NUM_PROTOCOLS,
                        protoEnable),
              ncclSuccess);

    for(int p = 0; p < NCCL_NUM_PROTOCOLS; ++p)
    {
        const int broadcastSlot = ncclFuncBroadcast * NCCL_NUM_PROTOCOLS + p;
        if(p == NCCL_PROTO_SIMPLE)
            EXPECT_EQ(protoEnable[broadcastSlot], 1) << "Broadcast should enable Simple only";
        else
            EXPECT_EQ(protoEnable[broadcastSlot], 0) << "Broadcast should disable non-Simple protocols";
    }

    // Other core collectives are untouched by a broadcast-prefixed entry.
    for(int f = 1; f < NCCL_NUM_FUNCTIONS; ++f)
        for(int p = 0; p < NCCL_NUM_PROTOCOLS; ++p)
            EXPECT_EQ(protoEnable[f * NCCL_NUM_PROTOCOLS + p], 1)
                << "Unprefixed rows should retain their prior enablement";
}

TEST(TuningParseList, BroadcastSimpleCaseInsensitive)
{
    int protoEnable[NCCL_NUM_FUNCTIONS * NCCL_NUM_PROTOCOLS] = {};
    for(int i = 0; i < NCCL_NUM_FUNCTIONS * NCCL_NUM_PROTOCOLS; ++i)
        protoEnable[i] = 1;

    ASSERT_EQ(parseList("Broadcast:Simple", ncclFuncStr, NCCL_NUM_FUNCTIONS, ncclProtoStr, NCCL_NUM_PROTOCOLS,
                        protoEnable),
              ncclSuccess);
    EXPECT_EQ(protoEnable[ncclFuncBroadcast * NCCL_NUM_PROTOCOLS + NCCL_PROTO_SIMPLE], 1);
    EXPECT_EQ(protoEnable[ncclFuncBroadcast * NCCL_NUM_PROTOCOLS + NCCL_PROTO_LL], 0);
    EXPECT_EQ(protoEnable[ncclFuncBroadcast * NCCL_NUM_PROTOCOLS + NCCL_PROTO_LL128], 0);
}

TEST(TuningParseList, GlobalSimpleThenBroadcastOverride)
{
    // Mirrors NCCL_PROTO="Simple;broadcast:LL" from tuning.cc comments: enable Simple
    // globally, then restrict Broadcast to LL.
    int protoEnable[NCCL_NUM_FUNCTIONS * NCCL_NUM_PROTOCOLS] = {};
    for(int i = 0; i < NCCL_NUM_FUNCTIONS * NCCL_NUM_PROTOCOLS; ++i)
        protoEnable[i] = 2; // non-zero default, distinct from parseList set/unset values

    ASSERT_EQ(parseList("Simple;broadcast:LL", ncclFuncStr, NCCL_NUM_FUNCTIONS, ncclProtoStr, NCCL_NUM_PROTOCOLS,
                        protoEnable),
              ncclSuccess);

    for(int f = 0; f < NCCL_NUM_FUNCTIONS; ++f)
    {
        for(int p = 0; p < NCCL_NUM_PROTOCOLS; ++p)
        {
            const int slot = f * NCCL_NUM_PROTOCOLS + p;
            if(f == ncclFuncBroadcast)
            {
                if(p == NCCL_PROTO_LL)
                    EXPECT_EQ(protoEnable[slot], 1);
                else
                    EXPECT_EQ(protoEnable[slot], 0);
            }
            else if(p == NCCL_PROTO_SIMPLE)
            {
                EXPECT_EQ(protoEnable[slot], 1);
            }
            else
            {
                EXPECT_EQ(protoEnable[slot], 0);
            }
        }
    }
}

} // namespace
} // namespace RcclUnitTesting
