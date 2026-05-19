/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */
#include "hipfile-test.h"
#include "include_internal/hipfile-stats.h"
#include "msys.h"
#include "stats.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace hipFile;
using ::testing::StrictMock;

// Put tests inside the macros to suppress the global constructor
// warnings
HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

class MStatsClient : public IStatsClient {
public:
    MOCK_METHOD(bool, pollProcess, (int timeout), (const, override));
    MOCK_METHOD(bool, connectServer, (), (override));
    MOCK_METHOD(bool, generateReport, (std::ostream & stream), (const, override));
};

struct HipFileStatsApi : public HipFileUnopened {};

TEST_F(HipFileStatsApi, CreateContextInvalidArgument)
{
    hipFileStatsContext_t *context;
    EXPECT_EQ(hipFileStatsCreateContext(nullptr, 1234), hipFileStatsInvalidArgument);
    EXPECT_EQ(hipFileStatsCreateContext(&context, -1), hipFileStatsInvalidArgument);
    EXPECT_EQ(context, nullptr);
    EXPECT_EQ(hipFileStatsCreateContext(&context, 0), hipFileStatsInvalidArgument);
    EXPECT_EQ(context, nullptr);
}

TEST_F(HipFileStatsApi, CreateContextInvalidTarget)
{
    StrictMock<MSys> msys{};
    EXPECT_CALL(msys, pidfd_open)
        .WillOnce(testing::Throw(std::system_error(EINVAL, std::generic_category())));
    hipFileStatsContext_t *context;
    EXPECT_EQ(hipFileStatsCreateContext(&context, 1234), hipFileStatsTargetProcessNotFound);
    EXPECT_EQ(context, nullptr);
}

TEST_F(HipFileStatsApi, ConnectToTargetProcessInvalidArgument)
{
    EXPECT_EQ(hipFileStatsConnectToTargetProcess(nullptr), hipFileStatsInvalidArgument);
}

TEST_F(HipFileStatsApi, ConnectToTargetProcessFailure)
{
    StrictMock<MStatsClient> client{};
    EXPECT_CALL(client, connectServer).WillOnce(testing::Return(false));
    EXPECT_EQ(hipFileStatsConnectToTargetProcess(reinterpret_cast<hipFileStatsContext_t *>(&client)),
              hipFileStatsTargetProcessNotAccessible);
}

TEST_F(HipFileStatsApi, ConnectToTargetProcessSuccess)
{
    StrictMock<MStatsClient> client{};
    EXPECT_CALL(client, connectServer).WillOnce(testing::Return(true));
    EXPECT_EQ(hipFileStatsConnectToTargetProcess(reinterpret_cast<hipFileStatsContext_t *>(&client)),
              hipFileStatsSuccess);
}

TEST_F(HipFileStatsApi, PollTargetProcessInvalidArgument)
{
    EXPECT_EQ(hipFileStatsPollTargetProcess(nullptr, false), hipFileStatsInvalidArgument);
}

TEST_F(HipFileStatsApi, PollTargetProcessFailure)
{
    StrictMock<MStatsClient> client{};
    EXPECT_CALL(client, pollProcess).WillOnce(testing::Return(false));
    EXPECT_EQ(hipFileStatsPollTargetProcess(reinterpret_cast<const hipFileStatsContext_t *>(&client), false),
              hipFileStatsTargetProcessNotAccessible);
}

TEST_F(HipFileStatsApi, PollTargetProcessSuccess)
{
    StrictMock<MStatsClient> client{};
    EXPECT_CALL(client, pollProcess).WillOnce(testing::Return(true));
    EXPECT_EQ(hipFileStatsPollTargetProcess(reinterpret_cast<const hipFileStatsContext_t *>(&client), false),
              hipFileStatsSuccess);
}

TEST_F(HipFileStatsApi, GenerateReportInvalidArgument)
{
    EXPECT_EQ(hipFileStatsGenerateReport(nullptr, 1), hipFileStatsInvalidArgument);
    EXPECT_EQ(hipFileStatsGenerateReport(reinterpret_cast<const hipFileStatsContext_t *>(0x1234), -1),
              hipFileStatsInvalidArgument);
}

TEST_F(HipFileStatsApi, GenerateReportSuccess)
{
    StrictMock<MStatsClient> client{};
    EXPECT_CALL(client, generateReport).WillOnce(testing::Return(true));
    EXPECT_EQ(hipFileStatsGenerateReport(reinterpret_cast<const hipFileStatsContext_t *>(&client), 1),
              hipFileStatsSuccess);
}

TEST_F(HipFileStatsApi, GenerateReportFailure)
{
    StrictMock<MStatsClient> client{};
    EXPECT_CALL(client, generateReport).WillOnce(testing::Return(false));
    EXPECT_EQ(hipFileStatsGenerateReport(reinterpret_cast<const hipFileStatsContext_t *>(&client), 1),
              hipFileStatsReportGenerationFailed);
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
