/*
 * Copyright (C) 2014-2018 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#include "KFDPerfCounters.hpp"

void KFDPerfCountersTest::SetUp() {
    ROUTINE_START

    KFDBaseComponentTest::SetUp();

    ROUTINE_END
}

void KFDPerfCountersTest::TearDown() {
    ROUTINE_START

    KFDBaseComponentTest::TearDown();

    ROUTINE_END
}

static struct block_name_table {
    char name[32];
    HSA_UUID uuid;
} block_lookup_table[] = {
    {"CB     ", {0x9ba429c6, 0xaf2d, 0x4b38, 0xb3, 0x49, 0x15, 0x72, 0x71, 0xbe, 0xac, 0x6a}},
    {"CPF    ", {0x2b0ad2b5, 0x1c43, 0x4f46, 0xa7, 0xbc, 0xe1, 0x19, 0x41, 0x1e, 0xa6, 0xc9}},
    {"CPG    ", {0x590ec94d, 0x20f0, 0x448f, 0x8d, 0xff, 0x31, 0x6c, 0x67, 0x9d, 0xe7, 0xff}},
    {"DB     ", {0x3d1a47fc, 0x0013, 0x4ed4, 0x83, 0x06, 0x82, 0x2c, 0xa0, 0xb7, 0xa6, 0xc2}},
    {"GDS    ", {0xf59276ec, 0x2526, 0x4bf8, 0x8e, 0xc0, 0x11, 0x8f, 0x77, 0x70, 0x0d, 0xc9}},
    {"GRBM   ", {0x8f00933c, 0xc33d, 0x4801, 0x97, 0xb7, 0x70, 0x07, 0xf7, 0x85, 0x73, 0xad}},
    {"GRBMSE ", {0x34ebd8d7, 0x7c8b, 0x4d15, 0x88, 0xfa, 0x0e, 0x4e, 0x4a, 0xf5, 0x9a, 0xc1}},
    {"IA     ", {0x34276944, 0x4264, 0x4fcd, 0x9d, 0x6e, 0xae, 0x26, 0x45, 0x82, 0xec, 0x51}},
    {"MC     ", {0x13900b57, 0x4956, 0x4d98, 0x81, 0xd0, 0x68, 0x52, 0x19, 0x37, 0xf5, 0x9c}},
    {"PASC   ", {0xb0e7fb5d, 0x0efc, 0x4744, 0xb5, 0x16, 0x5d, 0x23, 0xdc, 0x1f, 0xd5, 0x6c}},
    {"PASU   ", {0x9a152b6a, 0x1fad, 0x45f2, 0xa5, 0xbf, 0xf1, 0x63, 0x82, 0x6b, 0xd0, 0xcd}},
    {"SPI    ", {0xeda81044, 0xd62c, 0x47eb, 0xaf, 0x89, 0x4f, 0x6f, 0xbf, 0x3b, 0x38, 0xe0}},
    {"SRBM   ", {0x9f8040e0, 0x6830, 0x4019, 0xac, 0xc8, 0x46, 0x3c, 0x9e, 0x44, 0x5b, 0x89}},
    {"SQ     ", {0xb5c396b6, 0xd310, 0x47e4, 0x86, 0xfc, 0x5c, 0xc3, 0x4, 0x3a, 0xf5, 0x8}},
    {"SX     ", {0xbdb8d737, 0x43cc, 0x4162, 0xbe, 0x52, 0x51, 0xcf, 0xb8, 0x47, 0xbe, 0xaf}},
    {"TA     ", {0xc01ee43d, 0xad92, 0x44b1, 0x8a, 0xb9, 0xbe, 0x5e, 0x69, 0x6c, 0xee, 0xa7}},
    {"TCA    ", {0x333e393f, 0xe147, 0x4f49, 0xa6, 0xd1, 0x60, 0x91, 0x4c, 0x70, 0x86, 0xb0}},
    {"TCC    ", {0x848ce855, 0xd805, 0x4566, 0xa8, 0xab, 0x73, 0xe8, 0x84, 0xcc, 0x6b, 0xff}},
    {"TCP    ", {0xe10a013b, 0x17d4, 0x4bf5, 0xb0, 0x89, 0x42, 0x95, 0x91, 0x05, 0x9b, 0x60}},
    {"TCS    ", {0x4126245c, 0x4d96, 0x4d1a, 0x8a, 0xed, 0xa9, 0x39, 0xd4, 0xcc, 0x8e, 0xc9}},
    {"TD     ", {0x7d7c0fe4, 0xfe41, 0x4fea, 0x92, 0xc9, 0x45, 0x44, 0xd7, 0x70, 0x6d, 0xc6}},
    {"VGT    ", {0x0b6a8cb7, 0x7a01, 0x409f, 0xa2, 0x2c, 0x30, 0x14, 0x85, 0x4f, 0x13, 0x59}},
    {"WD     ", {0x0e176789, 0x46ed, 0x4b02, 0x97, 0x2a, 0x91, 0x6d, 0x2f, 0xac, 0x24, 0x4a}},
    {"DRIVER ", {0xea9b5ae1, 0x6c3f, 0x44b3, 0x89, 0x54, 0xda, 0xf0, 0x75, 0x65, 0xa9, 0xa}}
};

static void GetBlockName(HSA_UUID uuid, char *name, uint32_t name_len,
                                       char *uuid_str, uint32_t uuid_str_len) {
    uint32_t i, table_size;

    table_size = sizeof(block_lookup_table) / sizeof(struct block_name_table);

    snprintf(name, name_len, "unknown");
    for (i = 0; i < table_size; i++) {
        if (!memcmp(&block_lookup_table[i].uuid, &uuid, sizeof(HSA_UUID))) {
            if (name)
                snprintf(name, name_len, "%s", block_lookup_table[i].name);
            break;
        }
    }

    if (uuid_str)
        snprintf(uuid_str, uuid_str_len,
                 "%08x-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                 uuid.Data1, uuid.Data2, uuid.Data3,
                 uuid.Data4[0], uuid.Data4[1], uuid.Data4[2],
                 uuid.Data4[3], uuid.Data4[4], uuid.Data4[5],
                 uuid.Data4[6], uuid.Data4[7]);
}

static void GetCounterProperties(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDPerfCountersTest* pKFDPerfCountersTest =
                         (KFDPerfCountersTest*)pTestParamters->pTestObject;

    HsaCounterProperties* pProps = NULL;
    ASSERT_SUCCESS(hsaKmtPmcGetCounterProperties(gpuNode, &pProps));
    /* Verifying that there is at least one block */
    ASSERT_NE(0, pProps->NumBlocks) << "No performance counters blocks";

    LOG() << std::dec << pProps->NumBlocks << " blocks found." << std::endl;

    HsaCounterBlockProperties *block;
    block = &pProps->Blocks[0];
    for (HSAuint32 i = 0; i < pProps->NumBlocks; i++) {
        char uuid_string[37] = "";
        char name[32] = "";
        GetBlockName(block->BlockId, name, 32, uuid_string, 37);

        char type[32];
        switch (block->Counters[0].Type) {
        case HSA_PROFILE_TYPE_PRIVILEGED_IMMEDIATE:
            snprintf(type, sizeof(type), "Priv Immediate");
            break;
        case HSA_PROFILE_TYPE_PRIVILEGED_STREAMING:
            snprintf(type, sizeof(type), "Priv Streaming");
            break;
        case HSA_PROFILE_TYPE_NONPRIV_IMMEDIATE:
            snprintf(type, sizeof(type), "Non-priv Immediate");
            break;
        case HSA_PROFILE_TYPE_NONPRIV_STREAMING:
            snprintf(type, sizeof(type), "Non-priv Immediate");
            break;
        default:
            snprintf(type, sizeof(type), "Unknown");
            break;
        }

        LOG() << name << " (" << uuid_string << "): " << type << ", " <<
            block->NumCounters << " counter IDs" << std::endl;
        block = reinterpret_cast<HsaCounterBlockProperties *>(&block->Counters[block->NumCounters]);
    }
}

TEST_F(KFDPerfCountersTest, GetCounterProperties) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(GetCounterProperties));

    TEST_END
}

static void RegisterTrace(KFDTEST_PARAMETERS* pTestParamters) {

    HsaCounterProperties* pProps;

    int gpuNode = pTestParamters->gpuNode;
    KFDPerfCountersTest* pKFDPerfCountersTest =
                         (KFDPerfCountersTest*)pTestParamters->pTestObject;

    HsaPmcTraceRoot root;

    pProps = NULL;
    ASSERT_SUCCESS(hsaKmtPmcGetCounterProperties(gpuNode, &pProps));

    /* Verifying that there is at least one block */
    ASSERT_NE(0, pProps->NumBlocks) << "No performance counters blocks";

    HsaCounterBlockProperties *block = &pProps->Blocks[0];
    bool priv_block_found = false;
    for (HSAuint32 i = 0; i < pProps->NumBlocks; i++) {
        if (block->Counters[0].Type <= HSA_PROFILE_TYPE_PRIVILEGED_STREAMING) {
            priv_block_found = true;
            break;
        }
        block = reinterpret_cast<HsaCounterBlockProperties *>(&block->Counters[block->NumCounters]);
    }

    if (!priv_block_found) {
        LOG() << "Skipping test: No privileged block is found."
            << std::endl;
        return;
    }

    /* Registering trace */
    ASSERT_SUCCESS(hsaKmtPmcRegisterTrace(gpuNode,
                                          block->NumConcurrent,
                                          block->Counters,
                                          &root));
    EXPECT_SUCCESS(hsaKmtPmcUnregisterTrace(gpuNode, root.TraceId));
}

TEST_F(KFDPerfCountersTest, RegisterTrace) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(RegisterTrace));

    TEST_END
}

static const unsigned int START_STOP_DELAY = 10000;     // 10 sec tracing

static void StartStopQueryTrace(KFDTEST_PARAMETERS* pTestParamters){

    HsaPmcTraceRoot root;
    HsaCounterProperties* pProps;

    int gpuNode = pTestParamters->gpuNode;
    KFDPerfCountersTest* pKFDPerfCountersTest =
                         (KFDPerfCountersTest*)pTestParamters->pTestObject;

    pProps = NULL;
    ASSERT_SUCCESS(hsaKmtPmcGetCounterProperties(gpuNode, &pProps));

    /* Verifying that there is at least one block */
    ASSERT_NE(0, pProps->NumBlocks) << "No performance counters blocks";

    HsaCounterBlockProperties *block = &pProps->Blocks[0];
    bool priv_block_found = false;
    for (HSAuint32 i = 0; i < pProps->NumBlocks; i++) {
        if (block->Counters[0].Type <= HSA_PROFILE_TYPE_PRIVILEGED_STREAMING) {
            priv_block_found = true;
            break;
        }
        block = reinterpret_cast<HsaCounterBlockProperties *>(&block->Counters[block->NumCounters]);
    }

    if (!priv_block_found) {
        LOG() << "Skipping test: No privileged block is found."
             << std::endl;
        return;
    }

    if (getuid()) { /* Non-root */
        LOG() << "Skipping test: Privileged counters requires the user as root." << std::endl;
        return;
    }

    /* Registering trace */
    ASSERT_SUCCESS(hsaKmtPmcRegisterTrace(gpuNode,
                                          block->NumConcurrent,
                                          block->Counters,
                                          &root));

    /* Acquiring access for the trace */
    ASSERT_SUCCESS(hsaKmtPmcAcquireTraceAccess(gpuNode, root.TraceId));

    /* Allocating memory buffer for the trace */
    HsaMemoryBuffer membuf(PAGE_SIZE, gpuNode);

    /* Starting the trace */
    ASSERT_SUCCESS(hsaKmtPmcStartTrace(root.TraceId,
                                       membuf.As<void*>(),
                                       membuf.Size()));

    /* Delay between START and STOP tracing */
    Delay(START_STOP_DELAY);

    /* Stopping the trace */
    ASSERT_SUCCESS(hsaKmtPmcStopTrace(root.TraceId));

    /* Querying the trace */
    ASSERT_SUCCESS(hsaKmtPmcQueryTrace(root.TraceId));
    uint64_t *buf = membuf.As<uint64_t*>();
    for (uint32_t i = 0; i < block->NumConcurrent; i++, buf++)
        LOG() << "Counter " << std::dec << i << ": " << *buf << std::endl;

    /* Releasing the trace */
    EXPECT_SUCCESS(hsaKmtPmcReleaseTraceAccess(0, root.TraceId));

    EXPECT_SUCCESS(hsaKmtPmcUnregisterTrace(gpuNode, root.TraceId));
}

TEST_F(KFDPerfCountersTest, StartStopQueryTrace) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(RegisterTrace));

    TEST_END
}

static void ClockCountersBasicTest(KFDTEST_PARAMETERS* pTestParamters){

    int gpuNode = pTestParamters->gpuNode;
    KFDPerfCountersTest* pKFDPerfCountersTest =
					  (KFDPerfCountersTest*)pTestParamters->pTestObject;

    HsaClockCounters counters1;
    HsaClockCounters counters2;

    EXPECT_SUCCESS(hsaKmtGetClockCounters(gpuNode, &counters1));

    Delay(100);

    EXPECT_SUCCESS(hsaKmtGetClockCounters(gpuNode, &counters2));

    EXPECT_NE(0, counters1.GPUClockCounter);
    EXPECT_NE(0, counters2.GPUClockCounter);
    EXPECT_NE(0, counters1.SystemClockCounter);
    EXPECT_NE(0, counters2.SystemClockCounter);

    EXPECT_GT(counters2.GPUClockCounter, counters1.GPUClockCounter);
    EXPECT_GT(counters2.SystemClockCounter, counters1.SystemClockCounter);

}

TEST_F(KFDPerfCountersTest, ClockCountersBasicTest) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(ClockCountersBasicTest));

    TEST_END
}

