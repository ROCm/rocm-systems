/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hipfile-test.h"
#include "mconfiguration.h"
#include "mhip.h"
#include "mstats.h"
#include "stats.h"
#include "msys.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sstream>

using namespace hipFile;

using ::testing::StrictMock;

// Put tests inside the macros to suppress the global constructor
// warnings
HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

struct HipFileStats : public HipFileUnopened {};

TEST_F(HipFileStats, StatsCollectionAddIo)
{
    Stats                    stats{};
    StrictMock<MStatsServer> mstats{};
    StrictMock<MHip>         mhip{};
    EXPECT_CALL(mstats, getStats).WillRepeatedly(testing::Return(&stats));
    EXPECT_CALL(mhip, hipGetDevice).WillRepeatedly(testing::Return(0));
    stats.setLevel(StatsLevel::Basic);
    Context<StatsCollection>::get()->addIo(IoType::Read, StatsBackend::Fastpath, 10, 0);
    ASSERT_EQ(10, stats.getPerGpuStats(0, StatsBackend::Fastpath)
                      ->ioSizeBytes[static_cast<size_t>(IoType::Read)]
                      .buckets[0]
                      .load());
    Context<StatsCollection>::get()->addIo(IoType::Write, StatsBackend::Fallback, 10, 0);
    ASSERT_EQ(10, stats.getPerGpuStats(0, StatsBackend::Fallback)
                      ->ioSizeBytes[static_cast<size_t>(IoType::Write)]
                      .buckets[0]
                      .load());
    Context<StatsCollection>::get()->addIo(IoType::Read, StatsBackend::Fastpath, 10, 0);
    ASSERT_EQ(20, stats.getPerGpuStats(0, StatsBackend::Fastpath)
                      ->ioSizeBytes[static_cast<size_t>(IoType::Read)]
                      .buckets[0]
                      .load());
    stats.setLevel(StatsLevel::Disabled);
    Context<StatsCollection>::get()->addIo(IoType::Write, StatsBackend::Fallback, 10, 0);
    ASSERT_EQ(10, stats.getPerGpuStats(0, StatsBackend::Fallback)
                      ->ioSizeBytes[static_cast<size_t>(IoType::Write)]
                      .buckets[0]
                      .load());
}

TEST_F(HipFileStats, StatsCollectionError)
{
    Stats                    stats{};
    StrictMock<MStatsServer> mstats{};
    StrictMock<MHip>         mhip{};
    EXPECT_CALL(mstats, getStats).WillRepeatedly(testing::Return(&stats));
    EXPECT_CALL(mhip, hipGetDevice).WillRepeatedly(testing::Return(0));
    stats.setLevel(StatsLevel::Basic);
    Context<StatsCollection>::get()->error(IoType::Read, StatsBackend::Fastpath, 10);
    ASSERT_EQ(1, stats.getPerGpuStats(0, StatsBackend::Fastpath)
                     ->errorCount[static_cast<size_t>(IoType::Read)]
                     .buckets[0]
                     .load());
    Context<StatsCollection>::get()->error(IoType::Write, StatsBackend::Fallback, 10);
    ASSERT_EQ(1, stats.getPerGpuStats(0, StatsBackend::Fallback)
                     ->errorCount[static_cast<size_t>(IoType::Write)]
                     .buckets[0]
                     .load());
    Context<StatsCollection>::get()->error(IoType::Read, StatsBackend::Fastpath, 10);
    ASSERT_EQ(2, stats.getPerGpuStats(0, StatsBackend::Fastpath)
                     ->errorCount[static_cast<size_t>(IoType::Read)]
                     .buckets[0]
                     .load());
    stats.setLevel(StatsLevel::Disabled);
    Context<StatsCollection>::get()->error(IoType::Write, StatsBackend::Fallback, 10);
    ASSERT_EQ(1, stats.getPerGpuStats(0, StatsBackend::Fallback)
                     ->errorCount[static_cast<size_t>(IoType::Write)]
                     .buckets[0]
                     .load());
}

TEST_F(HipFileStats, StatsCollectionFileRegistration)
{
    Stats                    stats{};
    StrictMock<MStatsServer> mstats{};
    EXPECT_CALL(mstats, getStats).WillRepeatedly(testing::Return(&stats));
    stats.setLevel(StatsLevel::Basic);
    Context<StatsCollection>::get()->fileRegistration();
    ASSERT_EQ(1, stats.getFileRegistrations().load());
    Context<StatsCollection>::get()->fileRegistration();
    ASSERT_EQ(2, stats.getFileRegistrations().load());
    stats.setLevel(StatsLevel::Disabled);
    Context<StatsCollection>::get()->fileRegistration();
    ASSERT_EQ(2, stats.getFileRegistrations().load());
}

TEST_F(HipFileStats, StatsCollectionBufferRegistration)
{
    Stats                    stats{};
    StrictMock<MStatsServer> mstats{};
    EXPECT_CALL(mstats, getStats).WillRepeatedly(testing::Return(&stats));
    stats.setLevel(StatsLevel::Basic);
    Context<StatsCollection>::get()->bufferRegistration();
    ASSERT_EQ(1, stats.getBufferRegistrations().load());
    Context<StatsCollection>::get()->bufferRegistration();
    ASSERT_EQ(2, stats.getBufferRegistrations().load());
    stats.setLevel(StatsLevel::Disabled);
    Context<StatsCollection>::get()->bufferRegistration();
    ASSERT_EQ(2, stats.getBufferRegistrations().load());
}

TEST_F(HipFileStats, StatsCollectionFastpathRejection)
{
    Stats                    stats{};
    StrictMock<MStatsServer> mstats{};
    EXPECT_CALL(mstats, getStats).WillRepeatedly(testing::Return(&stats));
    stats.setLevel(StatsLevel::Basic);
    Context<StatsCollection>::get()->fastpathRejection();
    ASSERT_EQ(1, stats.getFastpathRejections().load());
    Context<StatsCollection>::get()->fastpathRejection();
    ASSERT_EQ(2, stats.getFastpathRejections().load());
    stats.setLevel(StatsLevel::Disabled);
    Context<StatsCollection>::get()->fastpathRejection();
    ASSERT_EQ(2, stats.getFastpathRejections().load());
}

TEST_F(HipFileStats, StatsContainer)
{
    StrictMock<MSys>           msys{};
    StrictMock<MConfiguration> mcfg{};
    alignas(Stats) char        buff[sizeof(Stats)];
    EXPECT_CALL(msys, memfd_create).WillOnce(testing::Return(10));
    EXPECT_CALL(msys, fcntl).WillOnce(testing::Return(0));
    EXPECT_CALL(msys, ftruncate);
    EXPECT_CALL(msys, mmap).WillOnce(testing::Return(&buff));
    EXPECT_CALL(msys, munmap).Times(1);
    EXPECT_CALL(msys, close).Times(1);
    EXPECT_CALL(mcfg, statsLevel()).WillOnce(testing::Return(1));
    StatsContainer sc{};
    ASSERT_EQ(reinterpret_cast<Stats *>(&buff), sc.getStats());
    ASSERT_EQ(10, sc.getFd());
}

TEST_F(HipFileStats, StatsContainerNoF_SEAL_FUTURE_WRITE)
{
    StrictMock<MSys>           msys{};
    StrictMock<MConfiguration> mcfg{};
    EXPECT_CALL(msys, memfd_create).WillOnce(testing::Return(10));
    EXPECT_CALL(mcfg, statsLevel()).WillOnce(testing::Return(1));
    EXPECT_CALL(msys, ftruncate);
    EXPECT_CALL(msys, mmap).WillOnce(testing::Return(nullptr));
    EXPECT_CALL(msys, fcntl).WillOnce(testing::Throw(std::system_error(EINVAL, std::generic_category())));
    EXPECT_CALL(msys, munmap).Times(1);
    EXPECT_CALL(msys, close).Times(1);
    StatsContainer sc{};
    ASSERT_EQ(nullptr, sc.getStats());
}

TEST_F(HipFileStats, GenerateReportV1)
{
    Stats              stats{};
    std::ostringstream os{};
    stats.getPerGpuStats(0, StatsBackend::Fastpath)
        ->ioSizeBytes[static_cast<size_t>(IoType::Read)]
        .buckets[0] = 2;
    stats.getPerGpuStats(0, StatsBackend::Fastpath)
        ->ioSizeBytes[static_cast<size_t>(IoType::Write)]
        .buckets[0] = 4;
    stats.getPerGpuStats(0, StatsBackend::Fallback)
        ->ioSizeBytes[static_cast<size_t>(IoType::Read)]
        .buckets[0] = 6;
    stats.getPerGpuStats(0, StatsBackend::Fallback)
        ->ioSizeBytes[static_cast<size_t>(IoType::Write)]
        .buckets[0] = 8;
    stats.getPerGpuStats(0, StatsBackend::Fastpath)
        ->errorCount[static_cast<size_t>(IoType::Read)]
        .buckets[0] = 1;
    stats.getPerGpuStats(0, StatsBackend::Fastpath)
        ->errorCount[static_cast<size_t>(IoType::Write)]
        .buckets[0] = 2;
    stats.getPerGpuStats(0, StatsBackend::Fallback)
        ->errorCount[static_cast<size_t>(IoType::Read)]
        .buckets[0] = 3;
    stats.getPerGpuStats(0, StatsBackend::Fallback)
        ->errorCount[static_cast<size_t>(IoType::Write)]
        .buckets[0]                = 4;
    stats.getBufferRegistrations() = 10;
    stats.getFileRegistrations()   = 20;
    stats.getFastpathRejections()  = 30;
    StatsClient::generateReportV1(os, &stats);
    std::string str{os.str()};
    ASSERT_GT(std::string::npos, str.find("Total Fastpath Read Size (B): 2"));
    ASSERT_GT(std::string::npos, str.find("Total Fastpath Write Size (B): 4"));
    ASSERT_GT(std::string::npos, str.find("Total Fallback Read Size (B): 6"));
    ASSERT_GT(std::string::npos, str.find("Total Fallback Write Size (B): 8"));
    ASSERT_GT(std::string::npos, str.find("Total Fastpath Read Errors: 1"));
    ASSERT_GT(std::string::npos, str.find("Total Fastpath Write Errors: 2"));
    ASSERT_GT(std::string::npos, str.find("Total Fallback Read Errors: 3"));
    ASSERT_GT(std::string::npos, str.find("Total Fallback Write Errors: 4"));
    ASSERT_GT(std::string::npos, str.find("Buffer Registrations: 10"));
    ASSERT_GT(std::string::npos, str.find("File Handle Registrations: 20"));
    ASSERT_GT(std::string::npos, str.find("Fastpath Rejections: 30"));
}

TEST_F(HipFileStats, GenerateReportV1TwoGpus)
{
    Stats              stats{};
    std::ostringstream os{};
    stats.getPerGpuStats(0, StatsBackend::Fastpath)
        ->ioSizeBytes[static_cast<size_t>(IoType::Read)]
        .buckets[0] = 2;
    stats.getPerGpuStats(1, StatsBackend::Fastpath)
        ->ioSizeBytes[static_cast<size_t>(IoType::Read)]
        .buckets[0] = 4;
    stats.getPerGpuStats(0, StatsBackend::Fastpath)
        ->errorCount[static_cast<size_t>(IoType::Read)]
        .buckets[0] = 1;
    stats.getPerGpuStats(1, StatsBackend::Fastpath)
        ->errorCount[static_cast<size_t>(IoType::Read)]
        .buckets[0] = 2;
    StatsClient::generateReportV1(os, &stats);
    std::string str{os.str()};
    ASSERT_GT(std::string::npos, str.find("Total Fastpath Read Size (B): 6"));
    ASSERT_GT(std::string::npos, str.find("Total Fastpath Read Errors: 3"));
}

struct HipFileStatsHistogram : public HipFileUnopened {};

TEST_F(HipFileStatsHistogram, ToHistogramBucket)
{
    ASSERT_EQ(0, StatsHistogram::toHistogramBucket(0));
    ASSERT_EQ(0, StatsHistogram::toHistogramBucket(1));
    ASSERT_EQ(0, StatsHistogram::toHistogramBucket(4095));
    ASSERT_EQ(1, StatsHistogram::toHistogramBucket(4096));
    ASSERT_EQ(1, StatsHistogram::toHistogramBucket(8191));
    ASSERT_EQ(2, StatsHistogram::toHistogramBucket(8192));
    ASSERT_EQ(2, StatsHistogram::toHistogramBucket(16383));
    ASSERT_EQ(3, StatsHistogram::toHistogramBucket(16384));
    ASSERT_EQ(14, StatsHistogram::toHistogramBucket(uint64_t{1} << 25));
    ASSERT_EQ(15, StatsHistogram::toHistogramBucket(uint64_t{1} << 30));
    ASSERT_EQ(15, StatsHistogram::toHistogramBucket(uint64_t{1} << 40));
}

TEST_F(HipFileStatsHistogram, BucketRange)
{
    auto [lower0, upper0] = StatsHistogram::bucketRange(0);
    ASSERT_EQ(0, lower0);
    ASSERT_EQ(4096, upper0);
    auto [lower1, upper1] = StatsHistogram::bucketRange(1);
    ASSERT_EQ(4096, lower1);
    ASSERT_EQ(8192, upper1);
    auto [lower2, upper2] = StatsHistogram::bucketRange(2);
    ASSERT_EQ(8192, lower2);
    ASSERT_EQ(16384, upper2);
    auto [lower3, upper3] = StatsHistogram::bucketRange(3);
    ASSERT_EQ(16384, lower3);
    ASSERT_EQ(32768, upper3);
    auto [lower14, upper14] = StatsHistogram::bucketRange(14);
    ASSERT_EQ(uint64_t{1} << 25, lower14);
    ASSERT_EQ(uint64_t{1} << 26, upper14);
    auto [lower15, upper15] = StatsHistogram::bucketRange(15);
    ASSERT_EQ(uint64_t{1} << 26, lower15);
    ASSERT_EQ(UINT64_MAX, upper15);
}

struct HipFileStatsPerGpuStatsV1 : public HipFileUnopened {};

TEST_F(HipFileStatsPerGpuStatsV1, GetHistograms)
{
    PerGpuStatsV1 stats{};
    auto [readSize, readCount, readTime, readError]     = stats.getHistograms(IoType::Read);
    auto [writeSize, writeCount, writeTime, writeError] = stats.getHistograms(IoType::Write);
    auto [invalidSize, invalidCount, invalidTime, invalidError] =
        stats.getHistograms(static_cast<IoType>(-1));
    ASSERT_EQ(&stats.ioSizeBytes[0], readSize);
    ASSERT_EQ(&stats.ioCount[0], readCount);
    ASSERT_EQ(&stats.ioTimeUs[0], readTime);
    ASSERT_EQ(&stats.errorCount[0], readError);
    ASSERT_EQ(&stats.ioSizeBytes[1], writeSize);
    ASSERT_EQ(&stats.ioCount[1], writeCount);
    ASSERT_EQ(&stats.ioTimeUs[1], writeTime);
    ASSERT_EQ(&stats.errorCount[1], writeError);
    ASSERT_EQ(nullptr, invalidSize);
    ASSERT_EQ(nullptr, invalidCount);
    ASSERT_EQ(nullptr, invalidTime);
    ASSERT_EQ(nullptr, invalidError);
}

struct HipFileStatsV1 : public HipFileUnopened {};

TEST_F(HipFileStatsV1, getPerGpuStats)
{
    StatsV1 stats{};
    ASSERT_EQ(nullptr, stats.getPerGpuStats(StatsV1::MaxGpus, StatsBackend::Fastpath));
    ASSERT_EQ(nullptr, stats.getPerGpuStats(0, static_cast<StatsBackend>(-1)));
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
