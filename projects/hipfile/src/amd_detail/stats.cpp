/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "configuration.h"
#include "context.h"
#include "stats.h"
#include "sys.h"

#include <cerrno>
#include <fcntl.h>
#include <poll.h>
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
StatsServer::StatsServer()
    : m_fd{FileDescriptor::make_managed(Context<Sys>::get()->memfd_create("AISSTATS", MFD_ALLOW_SEALING))},
      m_efd{FileDescriptor::make_managed(Context<Sys>::get()->eventfd(0, 0))}, m_stats{nullptr, &statsDeleter}
{
    int fd{m_fd.get()};
    Context<Sys>::get()->ftruncate(fd, sizeof(Stats));
    void *shm = Context<Sys>::get()->mmap(nullptr, sizeof(Stats), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    Context<Sys>::get()->fcntl(fd, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_FUTURE_WRITE);
    m_stats = UniqueStats{new (shm) Stats{}, &statsDeleter};
    m_stats->level =
        std::min(static_cast<StatsLevel>(Context<Configuration>::get()->statsLevel()), StatsLevel::Max);
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
StatsServer::statsDeleter(Stats *s)
{
    if (s == nullptr) {
        return;
    }
    s->~Stats();
    Context<Sys>::get()->munmap(s, sizeof(Stats));
}

void
StatsServer::threadFn()
{
    FileDescriptor sock{FileDescriptor::make_managed(socket(AF_UNIX, SOCK_STREAM, 0))};
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
            sendFd(conn, m_fd.get());
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
    stream << "AIS-STATS Version: " << stats->version
           << "\nHipFile Stats Level: " << static_cast<uint64_t>(stats->level) << '\n';
    switch (stats->version) {
        case 1:
            generateReportV1(stream, stats);
            break;
        default:
            break;
    }
    munmap(stats, sizeof(Stats));
    return true;
}

void
StatsClient::generateReportV1(std::ostream &stream, const Stats *stats)
{
    if (stats == nullptr) {
        return;
    }
    stream << "Total fast path reads (B): " << stats->getCounter(StatsCounters::TotalFastPathReadBytes).load()
           << "\nTotal fast path writes (B): "
           << stats->getCounter(StatsCounters::TotalFastPathWriteBytes).load()
           << "\nTotal fallback path reads (B): "
           << stats->getCounter(StatsCounters::TotalFallbackPathReadBytes).load()
           << "\nTotal fallback path writes (B): "
           << stats->getCounter(StatsCounters::TotalFallbackPathWriteBytes).load() << '\n';
}

void
statsAddFastPathRead(uint64_t bytes)
{
    Stats *stats{Context<StatsServer>::get()->getStats()};
    if (stats && stats->level >= StatsLevel::Basic) {
        stats->getCounter(StatsCounters::TotalFastPathReadBytes) += bytes;
    }
}

void
statsAddFastPathWrite(uint64_t bytes)
{
    Stats *stats{Context<StatsServer>::get()->getStats()};
    if (stats && stats->level >= StatsLevel::Basic) {
        stats->getCounter(StatsCounters::TotalFastPathWriteBytes) += bytes;
    }
}

void
statsAddFallbackPathRead(uint64_t bytes)
{
    Stats *stats{Context<StatsServer>::get()->getStats()};
    if (stats && stats->level >= StatsLevel::Basic) {
        stats->getCounter(StatsCounters::TotalFallbackPathReadBytes) += bytes;
    }
}

void
statsAddFallbackPathWrite(uint64_t bytes)
{
    Stats *stats{Context<StatsServer>::get()->getStats()};
    if (stats && stats->level >= StatsLevel::Basic) {
        stats->getCounter(StatsCounters::TotalFallbackPathWriteBytes) += bytes;
    }
}
}
