/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "buffer.h"
#include "file.h"
#include "file-descriptor.h"
#include "io.h"

#include <array>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <numeric>
#include <ostream>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>

namespace hipFile {

enum class StatsLevel : uint64_t {
    Disabled,
    Basic,
    Detailed,
    Max,
};

enum class StatsBackend {
    Fastpath,
    Fallback,
    Count,
};

struct StatsHistogram {
    static constexpr size_t   MaxBuckets{16};
    static constexpr uint64_t FirstBucketBits{12};
    using Buckets = std::array<std::atomic_uint64_t, MaxBuckets>;
    Buckets buckets{};

    static size_t toHistogramBucket(uint64_t bytes) noexcept
    {
        bytes >>= FirstBucketBits;
        int idx{bytes == 0 ? 0 : 64 - __builtin_clzll(bytes)}; // log2(); clz(0) is UB
        return std::min(static_cast<size_t>(idx), MaxBuckets - 1);
    }

    static constexpr std::pair<uint64_t, uint64_t> bucketRange(size_t bucket) noexcept
    {
        if (bucket == 0) {
            return std::make_pair(uint64_t{0}, uint64_t{1} << FirstBucketBits);
        }
        else if (bucket < MaxBuckets - 1) {
            uint64_t lower{uint64_t{1} << (bucket + FirstBucketBits - 1)};
            return std::make_pair(lower, lower << 1);
        }
        else {
            return std::make_pair(uint64_t{1} << (MaxBuckets + FirstBucketBits - 2), UINT64_MAX);
        }
    }

    uint64_t accumulate() const noexcept
    {
        return std::accumulate(buckets.begin(), buckets.end(), uint64_t{0});
    }
};

struct PerGpuStatsV1 {
    static constexpr uint64_t           version{1};
    std::atomic_uint64_t                inUse{};
    std::array<StatsHistogram, 2>       ioSizeBytes{};
    std::array<StatsHistogram, 2>       ioCount{};
    std::array<StatsHistogram, 2>       ioTimeUs{};
    std::array<StatsHistogram, 2>       errorCount{};
    std::array<std::atomic_uint64_t, 2> unalignedCount{};

    using Histograms = std::tuple<StatsHistogram *, StatsHistogram *, StatsHistogram *, StatsHistogram *>;
    using ConstHistograms = std::tuple<const StatsHistogram *, const StatsHistogram *, const StatsHistogram *,
                                       const StatsHistogram *>;

    Histograms getHistograms(IoType ioType) noexcept
    {
        if (ioType != IoType::Read && ioType != IoType::Write) {
            return Histograms{nullptr, nullptr, nullptr, nullptr};
        }
        return Histograms{&ioSizeBytes[static_cast<size_t>(ioType)], &ioCount[static_cast<size_t>(ioType)],
                          &ioTimeUs[static_cast<size_t>(ioType)], &errorCount[static_cast<size_t>(ioType)]};
    }

    ConstHistograms getHistograms(IoType ioType) const noexcept
    {
        if (ioType != IoType::Read && ioType != IoType::Write) {
            return ConstHistograms{nullptr, nullptr, nullptr, nullptr};
        }
        return ConstHistograms{&ioSizeBytes[static_cast<size_t>(ioType)],
                               &ioCount[static_cast<size_t>(ioType)], &ioTimeUs[static_cast<size_t>(ioType)],
                               &errorCount[static_cast<size_t>(ioType)]};
    }
};

// A lock-free accumulator of wall-clock "busy time": the length of the union of
// the [start, end] intervals of overlapping IOs. Bandwidth is bytes / busyUs, so
// concurrent IOs that overlap in time are not double counted (unlike summing each
// IO's duration, which inflates the denominator by the concurrency factor).
//
// m_state packs the number of currently-in-flight IOs in the top 16 bits and the
// timestamp of the last transition (microseconds, masked to 48 bits) in the low
// bits. Each successful compare-exchange "owns" the interval [oldTs, nowUs] and,
// if an IO was in flight over it, adds its length to m_busyUs. The CAS serializes
// transitions, so the owned intervals neither overlap nor leave gaps.
struct BusyWindow {
    static constexpr uint64_t TimestampBits{48};
    static constexpr uint64_t TimestampMask{(uint64_t{1} << TimestampBits) - 1};
    static constexpr uint64_t ActiveMask{0xFFFF};

    std::atomic_uint64_t state{};
    std::atomic_uint64_t busyUs{};

    // Approximate snapshot: committed busyUs plus the active tail up to nowUs.
    // busyUs is read before state so that if a transition fires between the two
    // loads it commits the interval to busyUs and clears active, giving a
    // correct result with no double-counting. The residual race is bounded by
    // one IO's duration and is acceptable for a live snapshot.
    uint64_t sample(uint64_t nowUs) const noexcept
    {
        nowUs &= TimestampMask;
        // seq_cst prevents the compiler from moving this load after the acquire
        // load below, which would allow a concurrent transition to commit to
        // busyUs between the two loads and be counted by both committed and tail.
        uint64_t committed{busyUs.load(std::memory_order_seq_cst)};
        uint64_t s{state.load(std::memory_order_acquire)};
        uint64_t active{s >> TimestampBits};
        uint64_t ts{s & TimestampMask};
        uint64_t tail{active > 0 && nowUs > ts ? nowUs - ts : 0};
        return committed + tail;
    }

    // delta is +1 when an IO starts and -1 when it completes.
    void transition(int delta, uint64_t nowUs) noexcept
    {
        nowUs &= TimestampMask;
        uint64_t old{state.load(std::memory_order_acquire)};
        for (;;) {
            uint64_t active{old >> TimestampBits};
            // Saturate rather than wrap: an extra endIo (active==0, delta==-1)
            // would wrap to 65535 and permanently corrupt sample(); an extra
            // beginIo beyond 65535 concurrent IOs would wrap to 0, losing all
            // in-flight accounting. Neither should happen in practice, but
            // mismatched calls must not silently corrupt the window.
            if ((delta < 0 && active == 0) || (delta > 0 && active == ActiveMask)) {
                return;
            }
            uint64_t ts{old & TimestampMask};
            // Clamp to prevent wrap-around if our nowUs is stale relative to ts.
            // This can happen on CAS retry when another thread's successful
            // transition has already advanced the stored timestamp past our nowUs.
            uint64_t safeNow{nowUs > ts ? nowUs : ts};
            uint64_t pending{active > 0 ? (safeNow - ts) : 0};
            uint64_t newActive{(static_cast<uint64_t>(static_cast<int64_t>(active) + delta)) & ActiveMask};
            uint64_t next{(newActive << TimestampBits) | safeNow};
            if (state.compare_exchange_weak(old, next, std::memory_order_acq_rel,
                                            std::memory_order_acquire)) {
                if (pending != 0) {
                    busyUs.fetch_add(pending, std::memory_order_relaxed);
                }
                return;
            }
        }
    }
};

static_assert(std::is_standard_layout_v<BusyWindow>, "BusyWindow must be standard layout");

template <typename PerGpuStatsT, size_t gpus> class StatsTemplate {
public:
    static constexpr size_t MaxGpus{gpus};
    using PerGpuStats      = PerGpuStatsT;
    using PerBackendStats  = std::array<PerGpuStats, static_cast<size_t>(StatsBackend::Count)>;
    using PerGpuStatsArray = std::array<PerBackendStats, MaxGpus>;

    static_assert(std::is_standard_layout_v<PerGpuStats>, "PerGpuStats must be standard layout");

    uint64_t getVersion() const noexcept
    {
        return m_version;
    }

    StatsLevel getLevel() const noexcept
    {
        return m_level;
    }

    void setLevel(StatsLevel level) noexcept
    {
        m_level = level;
    }

    PerGpuStats *getPerGpuStats(size_t gpuId, StatsBackend backend) noexcept
    {
        if (gpuId >= MaxGpus || static_cast<size_t>(backend) >= static_cast<size_t>(StatsBackend::Count)) {
            return nullptr;
        }
        return &m_perGpuStats[gpuId][static_cast<size_t>(backend)];
    }

    const PerGpuStats *getPerGpuStats(size_t gpuId, StatsBackend backend) const noexcept
    {
        if (gpuId >= MaxGpus || static_cast<size_t>(backend) >= static_cast<size_t>(StatsBackend::Count)) {
            return nullptr;
        }
        return &m_perGpuStats[gpuId][static_cast<size_t>(backend)];
    }

    bool gpuInUse(size_t gpuId) const noexcept
    {
        if (gpuId >= MaxGpus) {
            return false;
        }
        return std::any_of(m_perGpuStats[gpuId].begin(), m_perGpuStats[gpuId].end(),
                           [](const PerGpuStats &stats) { return stats.inUse.load() != 0; });
    }

    std::atomic_uint64_t &getFileRegistrations() noexcept
    {
        return m_fileRegistrations;
    }

    const std::atomic_uint64_t &getFileRegistrations() const noexcept
    {
        return m_fileRegistrations;
    }

    std::atomic_uint64_t &getBufferRegistrations() noexcept
    {
        return m_bufferRegistrations;
    }

    const std::atomic_uint64_t &getBufferRegistrations() const noexcept
    {
        return m_bufferRegistrations;
    }

    std::atomic_uint64_t &getFastpathRejections() noexcept
    {
        return m_fastpathRejections;
    }

    const std::atomic_uint64_t &getFastpathRejections() const noexcept
    {
        return m_fastpathRejections;
    }

    // Busy-time windows measured at each grouping the reports expose (per ioType,
    // per GPU+ioType, per backend+ioType). The union of intervals is not additive,
    // so it is measured separately at each grouping rather than summed from a finer
    // one. See BusyWindow and the three bandwidth report sites.
    BusyWindow *busyByType(IoType ioType) noexcept
    {
        if (ioType != IoType::Read && ioType != IoType::Write) {
            return nullptr;
        }
        return &m_busyByType[static_cast<size_t>(ioType)];
    }

    const BusyWindow *busyByType(IoType ioType) const noexcept
    {
        if (ioType != IoType::Read && ioType != IoType::Write) {
            return nullptr;
        }
        return &m_busyByType[static_cast<size_t>(ioType)];
    }

    BusyWindow *busyByGpuType(size_t gpuId, IoType ioType) noexcept
    {
        if (gpuId >= MaxGpus || (ioType != IoType::Read && ioType != IoType::Write)) {
            return nullptr;
        }
        return &m_busyByGpuType[gpuId][static_cast<size_t>(ioType)];
    }

    const BusyWindow *busyByGpuType(size_t gpuId, IoType ioType) const noexcept
    {
        if (gpuId >= MaxGpus || (ioType != IoType::Read && ioType != IoType::Write)) {
            return nullptr;
        }
        return &m_busyByGpuType[gpuId][static_cast<size_t>(ioType)];
    }

    BusyWindow *busyByBackendType(StatsBackend backend, IoType ioType) noexcept
    {
        if (static_cast<size_t>(backend) >= static_cast<size_t>(StatsBackend::Count) ||
            (ioType != IoType::Read && ioType != IoType::Write)) {
            return nullptr;
        }
        return &m_busyByBackendType[static_cast<size_t>(backend)][static_cast<size_t>(ioType)];
    }

    const BusyWindow *busyByBackendType(StatsBackend backend, IoType ioType) const noexcept
    {
        if (static_cast<size_t>(backend) >= static_cast<size_t>(StatsBackend::Count) ||
            (ioType != IoType::Read && ioType != IoType::Write)) {
            return nullptr;
        }
        return &m_busyByBackendType[static_cast<size_t>(backend)][static_cast<size_t>(ioType)];
    }

private:
    const uint64_t       m_version{PerGpuStatsT::version};
    StatsLevel           m_level{};
    std::atomic_uint64_t m_fileRegistrations{};
    std::atomic_uint64_t m_bufferRegistrations{};
    std::atomic_uint64_t m_fastpathRejections{};
    PerGpuStatsArray     m_perGpuStats{};

    std::array<BusyWindow, 2>                                                       m_busyByType{};
    std::array<std::array<BusyWindow, 2>, MaxGpus>                                  m_busyByGpuType{};
    std::array<std::array<BusyWindow, 2>, static_cast<size_t>(StatsBackend::Count)> m_busyByBackendType{};
};

using StatsV1 = StatsTemplate<PerGpuStatsV1, 16>;

using Stats = StatsV1;

static_assert(std::is_standard_layout_v<Stats>, "Stats must be standard layout");

class StatsContainer {
public:
    StatsContainer();
    ~StatsContainer();

    StatsContainer(const StatsContainer &)            = delete;
    StatsContainer &operator=(const StatsContainer &) = delete;

    StatsContainer(StatsContainer &&other) noexcept : m_fd(std::move(other.m_fd)), m_stats(other.m_stats)
    {
        other.m_stats = nullptr;
    }

    StatsContainer &operator=(StatsContainer &&other) noexcept
    {
        if (this != &other) {
            std::swap(m_fd, other.m_fd);
            std::swap(m_stats, other.m_stats);
        }
        return *this;
    }

    Stats *getStats() noexcept
    {
        return m_stats;
    }

    const Stats *getStats() const noexcept
    {
        return m_stats;
    }

    int getFd() const noexcept
    {
        return m_fd.get();
    }

private:
    FileDescriptor m_fd;
    Stats         *m_stats;
};

class IStatsServer {
public:
    virtual Stats *getStats() noexcept = 0;
    virtual ~IStatsServer()            = default;
};

class StatsServer : public IStatsServer {
public:
    StatsServer();
    virtual ~StatsServer() override;
    virtual Stats *getStats() noexcept override
    {
        return m_stats.getStats();
    }

private:
    void           threadFn();
    FileDescriptor m_efd;
    StatsContainer m_stats;
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

    static void generateReportV1(std::ostream &stream, const StatsV1 *stats);

private:
    FileDescriptor m_pfd, m_sfd;
    pid_t          m_pid{0};
};

// Tracks a single IO for stats. Construction opens the IO's busy-time window;
// complete() closes it at IO completion time and records the size, count, latency,
// and alignment histograms. Closing at completion rather than at destruction ensures
// a scheduler delay between IO return and stack unwind does not inflate concurrent
// IOs' windows. The destructor closes the window on the error path (when complete()
// was never called).
class StatsIoTracker {
public:
    StatsIoTracker(IoType ioType, StatsBackend backend, int deviceId) noexcept;

    StatsIoTracker(IoType ioType, StatsBackend backend, const std::shared_ptr<IFile> &file,
                   const std::shared_ptr<IBuffer> &buffer, size_t size, hoff_t file_offset,
                   hoff_t buffer_offset) noexcept
        : StatsIoTracker(ioType, backend, buffer->getGpuId())
    {
        const uint32_t dio_offset_align{file->dioOffsetAlign()};
        m_aligned &= dio_offset_align && !(file_offset & (dio_offset_align - 1));
        m_aligned &= dio_offset_align && !(size & (dio_offset_align - 1));

        const uint32_t dio_mem_align{file->dioMemAlign()};
        const auto     mem_addr{reinterpret_cast<intptr_t>(buffer->getBuffer()) + buffer_offset};
        m_aligned &= dio_mem_align && !(mem_addr & (dio_mem_align - 1));
    }
    ~StatsIoTracker();

    StatsIoTracker(const StatsIoTracker &)            = delete;
    StatsIoTracker &operator=(const StatsIoTracker &) = delete;
    StatsIoTracker(StatsIoTracker &&)                 = delete;
    StatsIoTracker &operator=(StatsIoTracker &&)      = delete;

    void complete(uint64_t bytes) noexcept;

    // Microseconds on the steady clock's own epoch (boot time on Linux), masked to
    // fit a BusyWindow timestamp. The epoch is shared across processes so ais-stats
    // can compare its own nowUs() against timestamps the library wrote.
    static uint64_t nowUs() noexcept;

private:
    IoType                                m_ioType;
    StatsBackend                          m_backend;
    std::chrono::steady_clock::time_point m_startTime;
    int                                   m_deviceId;
    bool                                  m_aligned;
    bool                                  m_windowOpen;
};

class StatsCollection {
public:
    virtual ~StatsCollection() = default;
    virtual void addIo(IoType ioType, StatsBackend backend, uint64_t bytes, uint64_t timeUs, bool aligned,
                       int deviceId) const noexcept;
    virtual void error(IoType ioType, StatsBackend backend, uint64_t bytes, int deviceId) const noexcept;
    virtual void fileRegistration() const noexcept;
    virtual void bufferRegistration() const noexcept;
    virtual void fastpathRejection() const noexcept;

    // Open an IO's busy-time windows at the three reported groupings. Returns true
    // if the windows were opened (stats enabled), in which case endIo() must be
    // called exactly once to close them. endIo() does not re-check the stats level,
    // so a mid-flight level change cannot leave a window stuck open.
    virtual bool beginIo(IoType ioType, StatsBackend backend, int deviceId, uint64_t nowUs) const noexcept;
    virtual void endIo(IoType ioType, StatsBackend backend, int deviceId, uint64_t nowUs) const noexcept;
};
}
