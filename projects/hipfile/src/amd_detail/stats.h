/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "file-descriptor.h"

#include <array>
#include <atomic>
#include <memory>
#include <ostream>
#include <thread>

namespace hipFile {

enum class StatsLevel : uint64_t {
    Disabled,
    Basic,
    Detailed,
    Max,
};

/// When adding new fields, remember to increment Stats::version
enum class StatsCounters {
    TotalFastPathReadBytes,
    TotalFastPathWriteBytes,
    TotalFallbackPathReadBytes,
    TotalFallbackPathWriteBytes,

    Max,
    Capacity = 64,
};

static_assert(StatsCounters::Max <= StatsCounters::Capacity, "Increase StatsCounters::Capacity");

struct Stats {
    using Array = std::array<std::atomic_uint64_t, static_cast<std::size_t>(StatsCounters::Capacity)>;
    StatsLevel     level{};
    const uint64_t version{1};
    Array          counters;

    std::atomic_uint64_t &getCounter(StatsCounters index)
    {
        return counters[static_cast<std::size_t>(index)];
    }
    const std::atomic_uint64_t &getCounter(StatsCounters index) const
    {
        return counters[static_cast<std::size_t>(index)];
    }
};

void statsAddFastPathRead(uint64_t bytes);
void statsAddFastPathWrite(uint64_t bytes);
void statsAddFallbackPathRead(uint64_t bytes);
void statsAddFallbackPathWrite(uint64_t bytes);

class StatsServer {
public:
    StatsServer();
    virtual ~StatsServer();
    virtual Stats *getStats()
    {
        return m_stats.get();
    }

    static void statsDeleter(Stats *s);
    using UniqueStats = std::unique_ptr<Stats, decltype(&StatsServer::statsDeleter)>;

private:
    void           threadFn();
    FileDescriptor m_fd;
    FileDescriptor m_efd;
    UniqueStats    m_stats;
    std::thread    m_thread;
};

class IStatsClient {
public:
    virtual ~IStatsClient()                                 = default;
    virtual bool pollProcess(int timeout) const             = 0;
    virtual bool connectServer()                            = 0;
    virtual bool generateReport(std::ostream &stream) const = 0;
};

class StatsClient : public IStatsClient {
public:
    explicit StatsClient(pid_t p);
    bool pollProcess(int timeout) const override;
    bool connectServer() override;
    bool generateReport(std::ostream &stream) const override;

    static void generateReportV1(std::ostream &stream, const Stats *stats);

private:
    FileDescriptor m_pfd, m_sfd;
    pid_t          m_pid{0};
};
}
