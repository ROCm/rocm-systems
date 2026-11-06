// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#include "debugger_control_transport.hpp"

#include <poll.h>
#include <pthread.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/timerfd.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <future>
#include <stdexcept>

namespace rocprofiler::tool::debugger_control
{
namespace
{
pthread_mutex_t  fork_mutex  = PTHREAD_MUTEX_INITIALIZER;
worker*          fork_worker = nullptr;
constexpr size_t max_frame   = 65536;

void
check(bool success, const char* operation)
{
    if(!success) throw std::runtime_error(std::string{operation} + ": " + strerror(errno));
}
uint64_t
number(const json& message, const char* key)
{
    const auto& value = message.at(key);
    if(!value.is_number_unsigned()) throw std::runtime_error("invalid unsigned field");
    return value.get<uint64_t>();
}
}  // namespace

uint64_t
monotonic_ns()
{
    timespec value{};
    clock_gettime(CLOCK_MONOTONIC, &value);
    return uint64_t(value.tv_sec) * 1000000000 + value.tv_nsec;
}

worker::worker(std::string endpoint, std::string session, json identity, control_fn control)
: endpoint_{std::move(endpoint)}
, session_{std::move(session)}
, identity_(std::move(identity))
, control_{std::move(control)}
, owner_{getpid()}
{}

worker::~worker() { finish(); }

void
worker::close_after_fork()
{
    // close, never shutdown: a child's inherited descriptor shares the parent's socket.
    if(socket_ >= 0) close(socket_);
    if(timer_ >= 0) close(timer_);
    if(wake_ >= 0) close(wake_);
    socket_ = timer_ = wake_ = -1;
}

void
worker::start()
{
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    if(endpoint_.size() >= sizeof(address.sun_path))
        throw std::runtime_error("ATT session socket path too long");
    std::strcpy(address.sun_path, endpoint_.c_str());
    socket_ = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    check(socket_ >= 0, "socket");
    check(connect(socket_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0,
          "connect to ATT session");
    ucred     peer{};
    socklen_t length = sizeof(peer);
    check(
        getsockopt(socket_, SOL_SOCKET, SO_PEERCRED, &peer, &length) == 0 && peer.uid == geteuid(),
        "ATT session peer credentials");
    // A failed controller must not indefinitely block a participant writing a reply.
    timeval send_timeout{10, 0};
    check(setsockopt(socket_, SOL_SOCKET, SO_SNDTIMEO, &send_timeout, sizeof(send_timeout)) == 0,
          "socket send timeout");
    timer_ = timerfd_create(CLOCK_MONOTONIC, TFD_CLOEXEC);
    wake_  = eventfd(0, EFD_CLOEXEC);
    check(timer_ >= 0 && wake_ >= 0, "ATT control descriptors");
    static const int atfork_status = pthread_atfork([] { pthread_mutex_lock(&fork_mutex); },
                                                    [] { pthread_mutex_unlock(&fork_mutex); },
                                                    [] {
                                                        if(fork_worker)
                                                            fork_worker->close_after_fork();
                                                        pthread_mutex_unlock(&fork_mutex);
                                                    });
    check(atfork_status == 0, "pthread_atfork");
    pthread_mutex_lock(&fork_mutex);
    fork_worker = this;
    pthread_mutex_unlock(&fork_mutex);
    // Publish registration before initialization returns, without waiting for a debugger
    // reply (which could deadlock profiler attach/bootstrap).
    std::promise<void> registered;
    auto               ready = registered.get_future();
    thread_                  = std::thread{[this, &registered] {
        pthread_setname_np(pthread_self(), "rocp-gdb-ctrl");
        auto hello       = identity_;
        hello["op"]      = "REGISTER";
        hello["version"] = 3;
        hello["pid"]     = owner_;
        hello["tid"]     = syscall(SYS_gettid);
        if(send(std::move(hello)))
            registered.set_value();
        else
            registered.set_exception(
                std::make_exception_ptr(std::runtime_error("ATT registration failed")));
        serve();
    }};
    ready.get();
}

bool
worker::send(json message)
{
    message["session"]  = session_;
    message["instance"] = identity_.at("instance");
    auto data           = message.dump() + '\n';
    if(data.size() > max_frame) return false;
    size_t offset = 0;
    while(offset < data.size())
    {
        auto count = ::send(socket_, data.data() + offset, data.size() - offset, MSG_NOSIGNAL);
        if(count < 0 && errno == EINTR) continue;
        if(count <= 0) return false;
        offset += count;
    }
    return true;
}

void
worker::stop(const char* reason)
{
    itimerspec off{};
    timerfd_settime(timer_, 0, &off, nullptr);
    if(active_)
    {
        control_(false, capture_);
        active_    = false;
        last_stop_ = {{"op", "STOPPED"},
                      {"capture", capture_},
                      {"reason", reason},
                      {"timestamp", monotonic_ns()}};
    }
    if(!last_stop_.is_null()) send(last_stop_);
}

void
worker::serve()
{
    std::string input;
    const char* ending = "disconnect";
    try
    {
        bool connected = true;
        while(connected)
        {
            pollfd fds[]  = {{socket_, POLLIN, 0}, {timer_, POLLIN, 0}, {wake_, POLLIN, 0}};
            auto   result = poll(fds, 3, -1);
            if(result < 0 && errno == EINTR) continue;
            if(result < 0) break;
            if(fds[2].revents)
            {
                ending = "exit";
                break;
            }
            if(fds[1].revents & POLLIN)
            {
                uint64_t count = 0;
                if(read(timer_, &count, sizeof(count)) != sizeof(count)) break;
                stop("timeout");
            }
            if(!(fds[0].revents & (POLLIN | POLLHUP | POLLERR))) continue;
            char buffer[4096];
            auto count = read(socket_, buffer, sizeof(buffer));
            if(count < 0 && errno == EINTR) continue;
            if(count <= 0) break;
            input.append(buffer, count);
            size_t end = 0;
            while((end = input.find('\n')) != std::string::npos)
            {
                if(end >= max_frame) throw std::runtime_error("oversized ATT control frame");
                auto message = json::parse(input.substr(0, end));
                input.erase(0, end + 1);
                if(message.at("session") != session_ ||
                   message.at("instance") != identity_.at("instance"))
                    throw std::runtime_error("wrong ATT session or worker generation");
                auto op = message.at("op").get<std::string>();
                if(op == "REGISTERED") continue;
                auto id = number(message, "capture");
                try
                {
                    if(op == "START" && id > capture_ && !active_)
                    {
                        capture_   = id;
                        deadline_  = 0;
                        last_stop_ = nullptr;
                        // Treat an exception as a possibly partial start; always attempt stop.
                        active_ = true;
                        control_(true, id);
                        connected = send(
                            {{"op", "STARTED"}, {"capture", id}, {"timestamp", monotonic_ns()}});
                    }
                    else if(op == "DEADLINE" && id == capture_ && active_)
                    {
                        auto deadline = number(message, "deadline");
                        if(!deadline || (deadline_ && deadline_ != deadline))
                            throw std::runtime_error("invalid or changed deadline");
                        if(!deadline_)
                        {
                            deadline_ = deadline;
                            itimerspec spec{};
                            spec.it_value.tv_sec  = deadline / 1000000000;
                            spec.it_value.tv_nsec = deadline % 1000000000;
                            check(timerfd_settime(timer_, TFD_TIMER_ABSTIME, &spec, nullptr) == 0,
                                  "timer setup");
                        }
                        connected =
                            send({{"op", "TIMED"}, {"capture", id}, {"deadline", deadline_}});
                    }
                    else if(op == "STOP" && id == capture_ && id)
                        stop("request");
                    else
                        throw std::runtime_error("invalid or stale command");
                } catch(const std::exception& error)
                {
                    connected = send({{"op", "ERROR"}, {"capture", id}, {"message", error.what()}});
                }
            }
            if(input.size() >= max_frame) throw std::runtime_error("unterminated ATT frame");
        }
    } catch(const std::exception& error)
    {
        send({{"op", "ERROR"}, {"capture", capture_}, {"message", error.what()}});
    }
    try
    {
        if(active_) stop(ending);
    } catch(const std::exception& error)
    {
        std::fprintf(stderr, "[att] participant cleanup failed: %s\n", error.what());
    }
    send({{"op", "BYE"}, {"capture", capture_}, {"reason", ending}});
    // Keep descriptor ownership in finish(); no close/FD-reuse race with its wakeup.
}

void
worker::finish()
{
    if(getpid() != owner_) return;  // An inherited runtime is not a new participant.
    if(thread_.joinable())
    {
        uint64_t one = 1;
        if(wake_ >= 0) (void) write(wake_, &one, sizeof(one));
        thread_.join();
    }
    // A concurrent fork must not follow a worker pointer while it is being destroyed.
    pthread_mutex_lock(&fork_mutex);
    if(fork_worker == this) fork_worker = nullptr;
    close_after_fork();
    pthread_mutex_unlock(&fork_mutex);
}
}  // namespace rocprofiler::tool::debugger_control
