/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hipfile-test.h"
#include "mconfiguration.h"
#include "mstats.h"
#include "stats.h"
#include "msys.h"

#include <atomic>
#include <chrono>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sstream>
#include <thread>
#include <vector>

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
    EXPECT_CALL(mstats, getStats).WillRepeatedly(testing::Return(&stats));
    stats.setLevel(StatsLevel::Basic);
    Context<StatsCollection>::get()->addIo(IoType::Read, StatsBackend::Fastpath, 10, 0, true, 0);
    ASSERT_EQ(10, stats.getPerGpuStats(0, StatsBackend::Fastpath)
                      ->ioSizeBytes[static_cast<size_t>(IoType::Read)]
                      .buckets[0]
                      .load());
    Context<StatsCollection>::get()->addIo(IoType::Write, StatsBackend::Fallback, 10, 0, true, 0);
    ASSERT_EQ(10, stats.getPerGpuStats(0, StatsBackend::Fallback)
                      ->ioSizeBytes[static_cast<size_t>(IoType::Write)]
                      .buckets[0]
                      .load());
    Context<StatsCollection>::get()->addIo(IoType::Read, StatsBackend::Fastpath, 10, 0, true, 0);
    ASSERT_EQ(20, stats.getPerGpuStats(0, StatsBackend::Fastpath)
                      ->ioSizeBytes[static_cast<size_t>(IoType::Read)]
                      .buckets[0]
                      .load());
    stats.setLevel(StatsLevel::Disabled);
    Context<StatsCollection>::get()->addIo(IoType::Write, StatsBackend::Fallback, 10, 0, true, 0);
    ASSERT_EQ(10, stats.getPerGpuStats(0, StatsBackend::Fallback)
                      ->ioSizeBytes[static_cast<size_t>(IoType::Write)]
                      .buckets[0]
                      .load());
}

TEST_F(HipFileStats, StatsCollectionAddIoNonZeroDevice)
{
    Stats                    stats{};
    StrictMock<MStatsServer> mstats{};
    EXPECT_CALL(mstats, getStats).WillRepeatedly(testing::Return(&stats));
    stats.setLevel(StatsLevel::Basic);
    Context<StatsCollection>::get()->addIo(IoType::Read, StatsBackend::Fastpath, 10, 0, true, 3);
    ASSERT_EQ(10, stats.getPerGpuStats(3, StatsBackend::Fastpath)
                      ->ioSizeBytes[static_cast<size_t>(IoType::Read)]
                      .buckets[0]
                      .load());
    ASSERT_EQ(0, stats.getPerGpuStats(0, StatsBackend::Fastpath)
                     ->ioSizeBytes[static_cast<size_t>(IoType::Read)]
                     .buckets[0]
                     .load());
}

TEST_F(HipFileStats, StatsCollectionAddIoUnaligned)
{
    Stats                    stats{};
    StrictMock<MStatsServer> mstats{};
    EXPECT_CALL(mstats, getStats).WillRepeatedly(testing::Return(&stats));
    stats.setLevel(StatsLevel::Basic);
    Context<StatsCollection>::get()->addIo(IoType::Read, StatsBackend::Fastpath, 10, 0, false, 0);
    ASSERT_EQ(1, stats.getPerGpuStats(0, StatsBackend::Fastpath)
                     ->unalignedCount[static_cast<size_t>(IoType::Read)]
                     .load());
    Context<StatsCollection>::get()->addIo(IoType::Read, StatsBackend::Fastpath, 10, 0, true, 0);
    ASSERT_EQ(1, stats.getPerGpuStats(0, StatsBackend::Fastpath)
                     ->unalignedCount[static_cast<size_t>(IoType::Read)]
                     .load());
}

TEST_F(HipFileStats, StatsCollectionError)
{
    Stats                    stats{};
    StrictMock<MStatsServer> mstats{};
    EXPECT_CALL(mstats, getStats).WillRepeatedly(testing::Return(&stats));
    stats.setLevel(StatsLevel::Basic);
    Context<StatsCollection>::get()->error(IoType::Read, StatsBackend::Fastpath, 10, 0);
    ASSERT_EQ(1, stats.getPerGpuStats(0, StatsBackend::Fastpath)
                     ->errorCount[static_cast<size_t>(IoType::Read)]
                     .buckets[0]
                     .load());
    Context<StatsCollection>::get()->error(IoType::Write, StatsBackend::Fallback, 10, 0);
    ASSERT_EQ(1, stats.getPerGpuStats(0, StatsBackend::Fallback)
                     ->errorCount[static_cast<size_t>(IoType::Write)]
                     .buckets[0]
                     .load());
    Context<StatsCollection>::get()->error(IoType::Read, StatsBackend::Fastpath, 10, 0);
    ASSERT_EQ(2, stats.getPerGpuStats(0, StatsBackend::Fastpath)
                     ->errorCount[static_cast<size_t>(IoType::Read)]
                     .buckets[0]
                     .load());
    stats.setLevel(StatsLevel::Disabled);
    Context<StatsCollection>::get()->error(IoType::Write, StatsBackend::Fallback, 10, 0);
    ASSERT_EQ(1, stats.getPerGpuStats(0, StatsBackend::Fallback)
                     ->errorCount[static_cast<size_t>(IoType::Write)]
                     .buckets[0]
                     .load());
}

TEST_F(HipFileStats, StatsCollectionErrorNonZeroDevice)
{
    Stats                    stats{};
    StrictMock<MStatsServer> mstats{};
    EXPECT_CALL(mstats, getStats).WillRepeatedly(testing::Return(&stats));
    stats.setLevel(StatsLevel::Basic);
    Context<StatsCollection>::get()->error(IoType::Read, StatsBackend::Fastpath, 10, 3);
    ASSERT_EQ(1, stats.getPerGpuStats(3, StatsBackend::Fastpath)
                     ->errorCount[static_cast<size_t>(IoType::Read)]
                     .buckets[0]
                     .load());
    ASSERT_EQ(0, stats.getPerGpuStats(0, StatsBackend::Fastpath)
                     ->errorCount[static_cast<size_t>(IoType::Read)]
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
        .buckets[0] = 4;

    stats.getBufferRegistrations() = 10;
    stats.getFileRegistrations()   = 20;
    stats.getFastpathRejections()  = 30;

    stats.getPerGpuStats(0, StatsBackend::Fallback)->unalignedCount[static_cast<size_t>(IoType::Read)]  = 40;
    stats.getPerGpuStats(0, StatsBackend::Fallback)->unalignedCount[static_cast<size_t>(IoType::Write)] = 50;

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
    ASSERT_GT(std::string::npos, str.find("Total Fallback Read Unaligned: 40"));
    ASSERT_GT(std::string::npos, str.find("Total Fallback Write Unaligned: 50"));
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

namespace {
uint64_t
activeCount(const BusyWindow *window)
{
    return window->state.load() >> BusyWindow::TimestampBits;
}
}

// Two IOs that overlap in time must count the union of their intervals, not the
// sum of their durations. [0,100] and [50,150] overlap, so busy = 150, not 200.
TEST_F(HipFileStats, BusyTimeOverlappingIos)
{
    Stats                    stats{};
    StrictMock<MStatsServer> mstats{};
    EXPECT_CALL(mstats, getStats).WillRepeatedly(testing::Return(&stats));
    stats.setLevel(StatsLevel::Basic);

    const auto *col{Context<StatsCollection>::get()};
    ASSERT_TRUE(col->beginIo(IoType::Read, StatsBackend::Fastpath, 0, 0));
    ASSERT_TRUE(col->beginIo(IoType::Read, StatsBackend::Fastpath, 0, 50));
    col->endIo(IoType::Read, StatsBackend::Fastpath, 0, 100);
    col->endIo(IoType::Read, StatsBackend::Fastpath, 0, 150);

    // Every grouping that this IO belongs to sees the same union.
    EXPECT_EQ(150u, stats.busyByType(IoType::Read)->busyUs.load());
    EXPECT_EQ(150u, stats.busyByGpuType(0, IoType::Read)->busyUs.load());
    EXPECT_EQ(150u, stats.busyByBackendType(StatsBackend::Fastpath, IoType::Read)->busyUs.load());
    // The window is closed again.
    EXPECT_EQ(0u, activeCount(stats.busyByType(IoType::Read)));
}

// A fully-nested interval adds nothing beyond the outer one: [0,100] over [25,75]
// stays 100.
TEST_F(HipFileStats, BusyTimeNestedIos)
{
    Stats                    stats{};
    StrictMock<MStatsServer> mstats{};
    EXPECT_CALL(mstats, getStats).WillRepeatedly(testing::Return(&stats));
    stats.setLevel(StatsLevel::Basic);

    const auto *col{Context<StatsCollection>::get()};
    ASSERT_TRUE(col->beginIo(IoType::Write, StatsBackend::Fallback, 0, 0));
    ASSERT_TRUE(col->beginIo(IoType::Write, StatsBackend::Fallback, 0, 25));
    col->endIo(IoType::Write, StatsBackend::Fallback, 0, 75);
    col->endIo(IoType::Write, StatsBackend::Fallback, 0, 100);

    EXPECT_EQ(100u, stats.busyByType(IoType::Write)->busyUs.load());
}

// Disjoint intervals add their individual lengths and the idle gap between them is
// excluded: [0,100] + [200,300] = 200.
TEST_F(HipFileStats, BusyTimeDisjointIos)
{
    Stats                    stats{};
    StrictMock<MStatsServer> mstats{};
    EXPECT_CALL(mstats, getStats).WillRepeatedly(testing::Return(&stats));
    stats.setLevel(StatsLevel::Basic);

    const auto *col{Context<StatsCollection>::get()};
    ASSERT_TRUE(col->beginIo(IoType::Read, StatsBackend::Fastpath, 0, 0));
    col->endIo(IoType::Read, StatsBackend::Fastpath, 0, 100);
    ASSERT_TRUE(col->beginIo(IoType::Read, StatsBackend::Fastpath, 0, 200));
    col->endIo(IoType::Read, StatsBackend::Fastpath, 0, 300);

    EXPECT_EQ(200u, stats.busyByType(IoType::Read)->busyUs.load());
}

// Busy time is attributed to the IO's own grouping and nowhere else.
TEST_F(HipFileStats, BusyTimeGroupingIsolation)
{
    Stats                    stats{};
    StrictMock<MStatsServer> mstats{};
    EXPECT_CALL(mstats, getStats).WillRepeatedly(testing::Return(&stats));
    stats.setLevel(StatsLevel::Basic);

    const auto *col{Context<StatsCollection>::get()};
    ASSERT_TRUE(col->beginIo(IoType::Read, StatsBackend::Fastpath, 3, 0));
    col->endIo(IoType::Read, StatsBackend::Fastpath, 3, 100);

    EXPECT_EQ(100u, stats.busyByGpuType(3, IoType::Read)->busyUs.load());
    EXPECT_EQ(100u, stats.busyByBackendType(StatsBackend::Fastpath, IoType::Read)->busyUs.load());
    // Not GPU 0, not Write, not Fallback.
    EXPECT_EQ(0u, stats.busyByGpuType(0, IoType::Read)->busyUs.load());
    EXPECT_EQ(0u, stats.busyByType(IoType::Write)->busyUs.load());
    EXPECT_EQ(0u, stats.busyByBackendType(StatsBackend::Fallback, IoType::Read)->busyUs.load());
}

// With stats disabled, beginIo opens nothing and reports that it did not, so the
// tracker knows to skip the matching endIo.
TEST_F(HipFileStats, BusyTimeDisabledLevelDoesNothing)
{
    Stats                    stats{};
    StrictMock<MStatsServer> mstats{};
    EXPECT_CALL(mstats, getStats).WillRepeatedly(testing::Return(&stats));
    stats.setLevel(StatsLevel::Disabled);

    const auto *col{Context<StatsCollection>::get()};
    EXPECT_FALSE(col->beginIo(IoType::Read, StatsBackend::Fastpath, 0, 0));
    EXPECT_EQ(0u, stats.busyByType(IoType::Read)->busyUs.load());
    EXPECT_EQ(0u, activeCount(stats.busyByType(IoType::Read)));
}

// sample() includes the active tail so a live query during an ongoing IO is not
// understated by the duration of the in-flight IO.
TEST_F(HipFileStats, BusyTimeSampleIncludesActiveTail)
{
    StatsV1     stats{};
    BusyWindow *window{stats.busyByType(IoType::Read)};

    window->transition(1, 0); // IO starts at t=0, no prior busy time
    EXPECT_EQ(0u, window->busyUs.load());
    // sample() at t=1000 must include the 1000µs tail even though no transition
    // has committed it yet.
    EXPECT_GE(window->sample(1000), 1000u);

    window->transition(-1, 1000); // IO ends
    EXPECT_GE(window->busyUs.load(), 1000u);
    EXPECT_EQ(0u, activeCount(window));
    // With no active IO, sample() equals committed busyUs.
    EXPECT_EQ(window->busyUs.load(), window->sample(2000));
}

// beginIo() must reject device ordinals that addIo() would discard, so the
// busy-time denominator cannot grow without a matching byte count in the numerator.
TEST_F(HipFileStats, BeginIoReturnsFalseForInvalidDevice)
{
    Stats                    stats{};
    StrictMock<MStatsServer> mstats{};
    EXPECT_CALL(mstats, getStats).WillRepeatedly(testing::Return(&stats));
    stats.setLevel(StatsLevel::Basic);

    const auto *col{Context<StatsCollection>::get()};
    // Device 100 is beyond MaxGpus (16): addIo() discards its bytes.
    EXPECT_FALSE(col->beginIo(IoType::Read, StatsBackend::Fastpath, 100, 0));
    // No window should have been opened.
    EXPECT_EQ(0u, activeCount(stats.busyByType(IoType::Read)));
    EXPECT_EQ(0u, stats.busyByType(IoType::Read)->busyUs.load());
    EXPECT_EQ(0u, stats.busyByBackendType(StatsBackend::Fastpath, IoType::Read)->busyUs.load());
}

// complete() must close the busy-time window at IO completion, before the dtor
// runs, so a scheduler delay between IO return and stack unwind does not inflate
// the windows of concurrent IOs that begin after this IO ends.
TEST_F(HipFileStats, CompleteClosesWindowBeforeDestructor)
{
    Stats                    stats{};
    StrictMock<MStatsServer> mstats{};
    EXPECT_CALL(mstats, getStats).WillRepeatedly(testing::Return(&stats));
    stats.setLevel(StatsLevel::Basic);

    {
        StatsIoTracker tracker{IoType::Read, StatsBackend::Fastpath, 0};
        EXPECT_EQ(1u, activeCount(stats.busyByType(IoType::Read)));

        tracker.complete(1024);
        // Window must be closed by complete(), not deferred to the dtor.
        EXPECT_EQ(0u, activeCount(stats.busyByType(IoType::Read)));
    }
    // Dtor must not reopen or double-close the window.
    EXPECT_EQ(0u, activeCount(stats.busyByType(IoType::Read)));
}

// The lock-free transition path must be race-free and leave the window balanced
// (active count back to zero) under concurrency. Runs under the sanitizers.
TEST_F(HipFileStats, BusyTimeConcurrentBalanced)
{
    Stats                    stats{};
    StrictMock<MStatsServer> mstats{};
    EXPECT_CALL(mstats, getStats).WillRepeatedly(testing::Return(&stats));
    stats.setLevel(StatsLevel::Basic);

    const auto   *col{Context<StatsCollection>::get()};
    constexpr int kThreads{8};
    constexpr int kIters{2000};
    uint64_t      wallStart{StatsIoTracker::nowUs()};
    // EXPECT_* is not thread-safe in GoogleTest; collect failures atomically
    // and assert on the main thread after joining.
    std::atomic<bool>        allOpened{true};
    std::vector<std::thread> threads{};
    for (int t{}; t < kThreads; ++t) {
        threads.emplace_back([col, &allOpened] {
            for (int i{}; i < kIters; ++i) {
                uint64_t start{StatsIoTracker::nowUs()};
                bool     opened{col->beginIo(IoType::Read, StatsBackend::Fastpath, 0, start)};
                if (opened) {
                    col->endIo(IoType::Read, StatsBackend::Fastpath, 0, StatsIoTracker::nowUs());
                }
                else {
                    allOpened.store(false, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto &thread : threads) {
        thread.join();
    }
    EXPECT_TRUE(allOpened.load());
    uint64_t wallElapsed{StatsIoTracker::nowUs() - wallStart};

    EXPECT_EQ(0u, activeCount(stats.busyByType(IoType::Read)));
    EXPECT_EQ(0u, activeCount(stats.busyByGpuType(0, IoType::Read)));
    EXPECT_EQ(0u, activeCount(stats.busyByBackendType(StatsBackend::Fastpath, IoType::Read)));

    // busyUs must not exceed wall-clock elapsed time: a wrap-around from a stale
    // timestamp on CAS retry would inflate it to ~2^48 µs (nearly 9 years).
    uint64_t busyUs{stats.busyByType(IoType::Read)->busyUs.load()};
    EXPECT_GT(busyUs, 0u);
    EXPECT_LE(busyUs, wallElapsed);
}

// The aggregate bandwidth line divides total bytes by the backend+ioType busy
// time, not the summed per-IO latency.
TEST_F(HipFileStats, GenerateReportBandwidthUsesBusyTime)
{
    Stats              stats{};
    std::ostringstream os{};
    // 1 GiB over 1 s of busy time -> 1 GiB/s. A large summed latency would give a
    // very different number if it were (wrongly) used as the denominator.
    stats.getPerGpuStats(0, StatsBackend::Fastpath)
        ->ioSizeBytes[static_cast<size_t>(IoType::Read)]
        .buckets[0] = 1024ull * 1024 * 1024;
    stats.getPerGpuStats(0, StatsBackend::Fastpath)->ioTimeUs[static_cast<size_t>(IoType::Read)].buckets[0] =
        500000;
    stats.busyByBackendType(StatsBackend::Fastpath, IoType::Read)->busyUs = 1000000;

    StatsClient::generateReportV1(os, &stats);
    EXPECT_NE(std::string::npos, os.str().find("Average Fastpath Read Bandwidth (GiB/s): 1\n"));
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
        // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange): invalid value tests rejection
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
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange): invalid value tests rejection
    ASSERT_EQ(nullptr, stats.getPerGpuStats(0, static_cast<StatsBackend>(-1)));
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
