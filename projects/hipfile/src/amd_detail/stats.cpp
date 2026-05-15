/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "configuration.h"
#include "context.h"
#include "hip.h"
#include "stats.h"
#include "sys.h"

#include <cerrno>
#include <fcntl.h>
#include <iomanip>
#include <poll.h>
#include <sstream>
#include <unistd.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>

static int
sendFd(int sock, int fd) noexcept
{
    int      data{1};
    iovec    iov{&data, sizeof(data)};
    msghdr   msgh{};
    cmsghdr *cmsgp{};

    union {
        char    buff[CMSG_SPACE(sizeof(int))];
        cmsghdr align;
    } controlMsg{};

    msgh.msg_name       = nullptr;
    msgh.msg_namelen    = 0;
    msgh.msg_iov        = &iov;
    msgh.msg_iovlen     = 1;
    msgh.msg_control    = controlMsg.buff;
    msgh.msg_controllen = sizeof(controlMsg.buff);
    msgh.msg_flags      = 0;

    cmsgp             = CMSG_FIRSTHDR(&msgh);
    cmsgp->cmsg_level = SOL_SOCKET;
    cmsgp->cmsg_type  = SCM_RIGHTS;
    cmsgp->cmsg_len   = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cmsgp), &fd, sizeof(int));

    if (sendmsg(sock, &msgh, 0) == -1)
        return -1;

    return 0;
}

static int
recvFd(int sockfd) noexcept
{
    int      data, fd;
    iovec    iov{&data, sizeof(data)};
    msghdr   msgh{};
    cmsghdr *cmsgp{};

    union {
        char    buff[CMSG_SPACE(sizeof(int))];
        cmsghdr align;
    } controlMsg{};

    msgh.msg_name       = nullptr;
    msgh.msg_namelen    = 0;
    msgh.msg_iov        = &iov;
    msgh.msg_iovlen     = 1;
    msgh.msg_control    = controlMsg.buff;
    msgh.msg_controllen = sizeof(controlMsg.buff);
    msgh.msg_flags      = 0;

    if (recvmsg(sockfd, &msgh, 0) == -1)
        return -1;

    cmsgp = CMSG_FIRSTHDR(&msgh);
    if (cmsgp == NULL || cmsgp->cmsg_len != CMSG_LEN(sizeof(int)) || cmsgp->cmsg_level != SOL_SOCKET ||
        cmsgp->cmsg_type != SCM_RIGHTS) {
        errno = EINVAL;
        return -1;
    }
    memcpy(&fd, CMSG_DATA(cmsgp), sizeof(int));
    return fd;
}

static void
populateSocketAddr(sockaddr_un &addr, pid_t pid) noexcept
{
    memset(&addr, 0, sizeof(addr));
    addr.sun_family  = AF_UNIX;
    addr.sun_path[0] = '\0'; // abstract namespace
    snprintf(&(addr.sun_path[1]), sizeof(addr.sun_path) - 1, "AISSTATS%08x", static_cast<uint32_t>(pid));
}

namespace hipFile {

// This flag is not defined on kernels before 5.1
#ifndef F_SEAL_FUTURE_WRITE
#define F_SEAL_FUTURE_WRITE 0x0010
#endif

StatsContainer::StatsContainer()
    : m_fd{FileDescriptor::make_managed(Context<Sys>::get()->memfd_create("AISSTATS", MFD_ALLOW_SEALING))},
      m_stats{nullptr}
{
    StatsLevel level{static_cast<StatsLevel>(Context<Configuration>::get()->statsLevel())};
    Context<Sys>::get()->ftruncate(m_fd.get(), sizeof(Stats));
    void *shm =
        Context<Sys>::get()->mmap(nullptr, sizeof(Stats), PROT_READ | PROT_WRITE, MAP_SHARED, m_fd.get(), 0);
    try {
        Context<Sys>::get()->fcntl(m_fd.get(), F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_FUTURE_WRITE);
    }
    catch (const std::system_error &e) {
        Context<Sys>::get()->munmap(shm, sizeof(Stats));
        if (e.code().value() == EINVAL)
            return;
        throw;
    }
    catch (...) {
        Context<Sys>::get()->munmap(shm, sizeof(Stats));
        throw;
    }
    m_stats = new (shm) Stats;
    m_stats->setLevel(std::min(level, StatsLevel::Max));
}

StatsContainer::~StatsContainer()
{
    if (m_stats == nullptr)
        return;
    m_stats->~Stats();
    Context<Sys>::get()->munmap(m_stats, sizeof(Stats));
    m_stats = nullptr;
}

StatsServer::StatsServer()
    : m_efd{FileDescriptor::make_managed(Context<Sys>::get()->eventfd(0, 0))}, m_stats{}
{
    if (m_stats.getStats())
        m_thread = std::thread(&StatsServer::threadFn, this);
}

StatsServer::~StatsServer()
{
    if (m_thread.joinable()) {
        uint64_t i{1};
        while (write(m_efd.get(), &i, sizeof(i)) == -1 && errno == EINTR) {
        }
        m_thread.join();
    }
}

void
StatsServer::threadFn()
{
    FileDescriptor sock{FileDescriptor::make_managed(socket(AF_UNIX, SOCK_STREAM, 0))};
    int            fd{m_stats.getFd()};
    pid_t          pid{getpid()};
    if (sock.get() == -1) {
        return;
    }
    sockaddr_un addr;
    populateSocketAddr(addr, pid);
    if (bind(sock.get(), reinterpret_cast<const sockaddr *>(&addr), sizeof(addr)) == -1) {
        return;
    }
    if (listen(sock.get(), 64) == -1) {
        return;
    }
    while (true) {
        pollfd pfd[2]{{sock.get(), POLLIN, 0}, {m_efd.get(), POLLIN, 0}};

        int ret = poll(&pfd[0], 2, -1);

        if (ret == 0) {
            continue;
        }
        else if (ret < 0) {
            if (errno == EINTR) {
                // Signal interrupt
                continue;
            }
            else {
                // Badness
                break;
            }
        }

        if (pfd[0].revents & POLLIN) {
            socklen_t addrlen{sizeof(addr)};
            int       conn{accept(sock.get(), reinterpret_cast<sockaddr *>(&addr), &addrlen)};
            if (conn == -1) {
                continue;
            }
            sendFd(conn, fd);
            close(conn);
        }
        if (pfd[1].revents & (POLLIN | POLLERR | POLLNVAL)) {
            break;
        }
    }
}

StatsClient::StatsClient(pid_t p)
    : m_pfd{FileDescriptor::make_managed(Context<Sys>::get()->pidfd_open(p, 0))}, m_pid{p}
{
}

bool
StatsClient::pollProcess(int timeout) const
{
    if (m_pfd.get() == -1) {
        return true;
    }
    pollfd pfd{m_pfd.get(), POLLIN, 0};
    return poll(&pfd, 1, timeout) > 0;
}

bool
StatsClient::connectServer()
{
    FileDescriptor sock{FileDescriptor::make_managed(socket(AF_UNIX, SOCK_STREAM, 0))};
    if (sock.get() == -1) {
        return false;
    }
    int success{-1};

    sockaddr_un addr;
    populateSocketAddr(addr, m_pid);
    for (int timeout{0}; !pollProcess(timeout); ++timeout) { // backoff on connect attempts
        success = connect(sock.get(), reinterpret_cast<sockaddr *>(&addr), sizeof(struct sockaddr_un));
        if (success == 0) {
            break;
        }
    }
    if (success == -1) {
        return false;
    }
    m_sfd = FileDescriptor::make_managed(recvFd(sock.get()));
    return true;
}

bool
StatsClient::generateReport(std::ostream &stream) const
{
    if (m_sfd.get() == -1) {
        return false;
    }
    void *shm = mmap(nullptr, sizeof(Stats), PROT_READ, MAP_SHARED, m_sfd.get(), 0);
    if (shm == reinterpret_cast<void *>(-1)) {
        return false;
    }
    Stats *stats = reinterpret_cast<Stats *>(shm);
    if (stats == nullptr) {
        return false;
    }
    stream << "AIS-STATS Version: " << stats->getVersion()
           << "\nHipFile Stats Level: " << static_cast<uint64_t>(stats->getLevel()) << '\n';
    switch (stats->getVersion()) {
        case 1:
            generateReportV1(stream, stats);
            break;
        default:
            break;
    }
    munmap(stats, sizeof(Stats));
    return true;
}

namespace reportUtil {
    const char *toString(IoType) noexcept;
    const char *toString(StatsBackend) noexcept;
    double      bandwidthGiBs(uint64_t bytes, uint64_t timeUs) noexcept;
    double      latencyUs(uint64_t timeUs, uint64_t count) noexcept;

    const char *toString(IoType ioType) noexcept
    {
        switch (ioType) {
            case IoType::Read:
                return "Read";
            case IoType::Write:
                return "Write";
            default:
                return "Unknown";
        }
    }

    const char *toString(StatsBackend backend) noexcept
    {
        switch (backend) {
            case StatsBackend::Fastpath:
                return "Fastpath";
            case StatsBackend::Fallback:
                return "Fallback";
            case StatsBackend::Count:
                [[fallthrough]];
            default:
                return "Unknown";
        }
    }

    double bandwidthGiBs(uint64_t bytes, uint64_t timeUs) noexcept
    {
        double time{static_cast<double>(timeUs) / 1000000.0};
        double gigaBytes{static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0)};
        return time > 0 ? (gigaBytes / time) : 0.0;
    }

    double latencyUs(uint64_t timeUs, uint64_t count) noexcept
    {
        return count > 0 ? (static_cast<double>(timeUs) / static_cast<double>(count)) : 0.0;
    }
}

static void
generateReportHistogramV1(
    std::ostream &stream, const StatsV1 *stats, size_t gpuId, const char *name, const char *op,
    std::function<void(std::ostream &, PerGpuStatsV1::ConstHistograms, size_t bucket)> printFn)
{
    if (stats == nullptr) {
        return;
    }
    static constexpr IoType       ioTypes[]{IoType::Read, IoType::Write};
    static constexpr StatsBackend backends[]{StatsBackend::Fastpath, StatsBackend::Fallback};
    int                           fw{16}, w{34};
    stream << "IO " << name << " Histogram\n";
    stream << std::left << std::setw(fw) << "IO Size (KiB)";
    for (const auto &backend : backends) {
        for (const auto &ioType : ioTypes) {
            std::stringstream header{};
            header << reportUtil::toString(backend) << ' ' << reportUtil::toString(ioType) << ' ' << op;
            stream << std::right << std::setw(w) << header.str();
        }
    }
    stream << '\n';
    for (size_t i{}; i < StatsHistogram::MaxBuckets; ++i) {
        auto [lower, upper] = StatsHistogram::bucketRange(i);
        std::stringstream rowName{};
        rowName << (lower >> 10) << '-'
                << (i == StatsHistogram::MaxBuckets - 1 ? "..." : std::to_string(upper >> 10));
        stream << std::left << std::setw(fw) << rowName.str();
        for (const auto &backend : backends) {
            for (const auto &ioType : ioTypes) {
                if (const auto *perGpuStats{stats->getPerGpuStats(gpuId, backend)}) {
                    stream << std::right << std::setw(w);
                    printFn(stream, perGpuStats->getHistograms(ioType), i);
                }
            }
        }
        stream << '\n';
    }
}

void
StatsClient::generateReportV1(std::ostream &stream, const StatsV1 *stats)
{
    if (stats == nullptr) {
        return;
    }
    static constexpr IoType       ioTypes[]{IoType::Read, IoType::Write};
    static constexpr StatsBackend backends[]{StatsBackend::Fastpath, StatsBackend::Fallback};
    for (const auto &backend : backends) {
        for (const auto &ioType : ioTypes) {
            uint64_t totalBytes{}, totalCount{}, totalTimeUs{}, totalErrors{};
            for (size_t i{}; i < StatsV1::MaxGpus; ++i) {
                if (const auto *perGpuStats{stats->getPerGpuStats(i, backend)}) {
                    if (const auto [sizeHist, countHist, timeHist, errorCountHist] =
                            perGpuStats->getHistograms(ioType);
                        sizeHist != nullptr && countHist != nullptr && timeHist != nullptr &&
                        errorCountHist != nullptr) {
                        totalBytes += sizeHist->accumulate();
                        totalCount += countHist->accumulate();
                        totalTimeUs += timeHist->accumulate();
                        totalErrors += errorCountHist->accumulate();
                    }
                }
            }
            stream << "Total " << reportUtil::toString(backend) << ' ' << reportUtil::toString(ioType)
                   << " Size (B): " << totalBytes << '\n';
            stream << "Average " << reportUtil::toString(backend) << ' ' << reportUtil::toString(ioType)
                   << " Bandwidth (GiB/s): " << reportUtil::bandwidthGiBs(totalBytes, totalTimeUs) << '\n';
            stream << "Average " << reportUtil::toString(backend) << ' ' << reportUtil::toString(ioType)
                   << " Latency (us): " << reportUtil::latencyUs(totalTimeUs, totalCount) << '\n';
            stream << "Total " << reportUtil::toString(backend) << ' ' << reportUtil::toString(ioType)
                   << " Errors: " << totalErrors << "\n\n";
        }
    }
    for (size_t gpuId{}; gpuId < StatsV1::MaxGpus; ++gpuId) {
        if (!stats->gpuInUse(gpuId)) {
            continue;
        }
        stream << "GPU " << gpuId << ":\n";
        auto ioSize = [](std::ostream &str, PerGpuStatsV1::ConstHistograms histograms, size_t bucket) {
            const auto [sizeHist, countHist, timeHist, errorCountHist] = histograms;
            (void)countHist;
            (void)timeHist;
            (void)errorCountHist;
            uint64_t size{};
            if (sizeHist != nullptr) {
                size = sizeHist->buckets[bucket].load();
            }
            str << size;
        };
        auto bandwidth = [](std::ostream &str, PerGpuStatsV1::ConstHistograms histograms, size_t bucket) {
            const auto [sizeHist, countHist, timeHist, errorCountHist] = histograms;
            (void)countHist;
            (void)errorCountHist;
            uint64_t size{}, timeUs{};
            if (sizeHist != nullptr) {
                size = sizeHist->buckets[bucket].load();
            }
            if (timeHist != nullptr) {
                timeUs = timeHist->buckets[bucket].load();
            }
            str << reportUtil::bandwidthGiBs(size, timeUs);
        };
        auto latency = [](std::ostream &str, PerGpuStatsV1::ConstHistograms histograms, size_t bucket) {
            const auto [sizeHist, countHist, timeHist, errorCountHist] = histograms;
            (void)sizeHist;
            (void)errorCountHist;
            uint64_t timeUs{}, count{};
            if (timeHist != nullptr) {
                timeUs = timeHist->buckets[bucket].load();
            }
            if (countHist != nullptr) {
                count = countHist->buckets[bucket].load();
            }
            str << reportUtil::latencyUs(timeUs, count);
        };
        auto errorCount = [](std::ostream &str, PerGpuStatsV1::ConstHistograms histograms, size_t bucket) {
            const auto [sizeHist, countHist, timeHist, errorCountHist] = histograms;
            (void)sizeHist;
            (void)countHist;
            (void)timeHist;
            uint64_t errors{};
            if (errorCountHist != nullptr) {
                errors = errorCountHist->buckets[bucket].load();
            }
            str << errors;
        };
        generateReportHistogramV1(stream, stats, gpuId, "Size", "Size (B)", ioSize);
        generateReportHistogramV1(stream, stats, gpuId, "Bandwidth", "Bandwidth (GiB/s)", bandwidth);
        generateReportHistogramV1(stream, stats, gpuId, "Latency", "Latency (us)", latency);
        generateReportHistogramV1(stream, stats, gpuId, "Errors", "Error Count", errorCount);
    }
}

void
StatsIoTracker::complete(uint64_t bytes) const noexcept
{
    auto endTime{std::chrono::steady_clock::now()};
    auto duration{std::chrono::duration_cast<std::chrono::microseconds>(endTime - m_startTime)};
    Context<StatsCollection>::get()->addIo(m_ioType, m_backend, bytes,
                                           static_cast<uint64_t>(duration.count()));
}

void
StatsCollection::addIo(IoType ioType, StatsBackend backend, uint64_t bytes, uint64_t timeUs) const noexcept
{
    Stats *stats{Context<IStatsServer>::get()->getStats()};
    if (stats == nullptr || stats->getLevel() < StatsLevel::Basic) {
        return;
    }
    size_t device{};
    try {
        device = static_cast<size_t>(Context<Hip>::get()->hipGetDevice());
    }
    catch (...) {
        return;
    }
    auto *perGpuStats{stats->getPerGpuStats(device, backend)};
    if (perGpuStats == nullptr) {
        return;
    }
    auto [sizeHist, countHist, timeHist, errorCountHist] = perGpuStats->getHistograms(ioType);
    (void)errorCountHist;
    if (sizeHist == nullptr || countHist == nullptr || timeHist == nullptr) {
        return;
    }
    size_t bucket = StatsHistogram::toHistogramBucket(bytes);
    sizeHist->buckets[bucket].fetch_add(bytes, std::memory_order_relaxed);
    countHist->buckets[bucket].fetch_add(1, std::memory_order_relaxed);
    timeHist->buckets[bucket].fetch_add(timeUs, std::memory_order_relaxed);
    perGpuStats->inUse.store(1, std::memory_order_relaxed);
}

void
StatsCollection::error(IoType ioType, StatsBackend backend, uint64_t bytes) const noexcept
{
    Stats *stats{Context<IStatsServer>::get()->getStats()};
    if (stats == nullptr || stats->getLevel() < StatsLevel::Basic) {
        return;
    }
    size_t device{};
    try {
        device = static_cast<size_t>(Context<Hip>::get()->hipGetDevice());
    }
    catch (...) {
        return;
    }
    auto *perGpuStats{stats->getPerGpuStats(device, backend)};
    if (perGpuStats == nullptr) {
        return;
    }
    auto [sizeHist, countHist, timeHist, errorCountHist] = perGpuStats->getHistograms(ioType);
    (void)sizeHist;
    (void)countHist;
    (void)timeHist;
    if (errorCountHist == nullptr) {
        return;
    }
    size_t bucket = StatsHistogram::toHistogramBucket(bytes);
    errorCountHist->buckets[bucket].fetch_add(1, std::memory_order_relaxed);
    perGpuStats->inUse.store(1, std::memory_order_relaxed);
}
}
