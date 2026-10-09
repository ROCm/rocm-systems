// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
// Prototype-only ATT ROCTx bridge. No SDK client or tool control API is added here.

#include <dlfcn.h>
#include <poll.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/timerfd.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace
{
int listener = -1;
int client   = -1;
int timer    = -1;

uint64_t
now()
{
    timespec value{};
    clock_gettime(CLOCK_MONOTONIC, &value);
    return uint64_t(value.tv_sec) * 1000000000 + value.tv_nsec;
}

bool
send_message(const std::string& message)
{
    size_t offset = 0;
    while(offset < message.size())
    {
        auto count = send(client, message.data() + offset, message.size() - offset, MSG_NOSIGNAL);
        if(count < 0 && errno == EINTR) continue;
        if(count <= 0) return false;
        offset += count;
    }
    return true;
}

void
close_in_child()
{
    if(listener >= 0) close(listener);
    if(client >= 0) close(client);
    if(timer >= 0) close(timer);
    listener = client = timer = -1;
}

void*
serve(void*)
{
    pthread_setname_np(pthread_self(), "rocp-gdb-ctrl");
    auto*       path = std::getenv("ROCPROFV3_GDB_SOCKET");
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    if(!path || strlen(path) >= sizeof(address.sun_path)) return nullptr;
    std::strcpy(address.sun_path, path);
    listener = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if(listener < 0 || bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) ||
       listen(listener, 1))
    {
        std::perror("[att] helper socket");
        if(listener >= 0) close(listener);
        listener = -1;
        return nullptr;
    }
    do
    {
        client = accept4(listener, nullptr, nullptr, SOCK_CLOEXEC);
    } while(client < 0 && errno == EINTR);
    close(listener);
    listener = -1;
    if(client < 0) return nullptr;
    ucred     peer{};
    socklen_t length = sizeof(peer);
    if(getsockopt(client, SOL_SOCKET, SO_PEERCRED, &peer, &length) || peer.uid != geteuid())
    {
        close(client);
        client = -1;
        return nullptr;
    }
    timer                = timerfd_create(CLOCK_MONOTONIC, TFD_CLOEXEC);
    using control_fn     = int (*)(uint64_t);
    auto resume          = reinterpret_cast<control_fn>(dlsym(RTLD_DEFAULT, "roctxProfilerResume"));
    auto pause           = reinterpret_cast<control_fn>(dlsym(RTLD_DEFAULT, "roctxProfilerPause"));
    using initialized_fn = int (*)(int*);
    auto initialized =
        reinterpret_cast<initialized_fn>(dlsym(RTLD_DEFAULT, "rocprofiler_is_initialized"));
    if(timer < 0 || !resume || !pause || !initialized)
    {
        send_message("ERROR 0 missing ROCTx, SDK, or timerfd\n");
        close_in_child();
        return nullptr;
    }
    send_message("HELLO 2 " + std::to_string(getpid()) + " " + std::to_string(syscall(SYS_gettid)) +
                 "\n");
    bool        active   = false;
    uint64_t    capture  = 0;
    uint64_t    timeout  = 0;
    uint64_t    deadline = 0;
    std::string last_stop;
    auto        stop = [&](const char* reason) {
        itimerspec off{};
        timerfd_settime(timer, 0, &off, nullptr);
        if(active)
        {
            if(pause(0) != 0)
            {
                send_message("ERROR " + std::to_string(capture) + " ROCTx Pause failed\n");
                return false;
            }
            active    = false;
            last_stop = "STOPPED " + std::to_string(capture) + " " + reason + " " +
                        std::to_string(now()) + "\n";
        }
        return send_message(last_stop);
    };
    std::string input;
    bool        connected = true;
    while(connected)
    {
        pollfd descriptors[] = {{client, POLLIN, 0}, {timer, POLLIN, 0}};
        auto   result        = poll(descriptors, 2, -1);
        if(result < 0 && errno == EINTR) continue;
        if(result < 0) break;
        if(descriptors[1].revents & POLLIN)
        {
            uint64_t expirations = 0;
            if(read(timer, &expirations, sizeof(expirations)) != sizeof(expirations)) break;
            if(active && !stop("timeout")) break;
        }
        if(descriptors[0].revents & (POLLIN | POLLHUP | POLLERR))
        {
            char buffer[256];
            auto count = read(client, buffer, sizeof(buffer));
            if(count <= 0) break;
            input.append(buffer, count);
            if(input.size() > 1024) break;
            size_t end = 0;
            while((end = input.find('\n')) != std::string::npos)
            {
                auto line = input.substr(0, end);
                input.erase(0, end + 1);
                uint64_t id = 0, duration = 0;
                char     extra = 0;
                if(sscanf(
                       line.c_str(), "START %" SCNu64 " %" SCNu64 " %c", &id, &duration, &extra) ==
                   2)
                {
                    int ready = 0;
                    if(active || id <= capture || duration > 3600ULL * 1000000000 ||
                       initialized(&ready) != 0 || !ready)
                    {
                        send_message("ERROR " + std::to_string(id) +
                                     " invalid START or SDK not initialized\n");
                        continue;
                    }
                    capture  = id;
                    timeout  = duration;
                    deadline = 0;
                    last_stop.clear();
                    if(resume(0) != 0)
                    {
                        send_message("ERROR " + std::to_string(id) + " ROCTx Resume failed\n");
                        continue;
                    }
                    active       = true;
                    auto started = now();
                    connected    = send_message("STARTED " + std::to_string(id) + " " +
                                             std::to_string(started) + "\n");
                }
                else if(sscanf(line.c_str(), "CONTINUED %" SCNu64 " %c", &id, &extra) == 1 &&
                        id == capture && active && timeout)
                {
                    // Exclude both profiler setup and debugger continuation from the interval.
                    // A repeated acknowledgement must not extend an already armed deadline.
                    if(!deadline)
                    {
                        deadline = now() + timeout;
                        itimerspec spec{};
                        spec.it_value.tv_sec  = deadline / 1000000000;
                        spec.it_value.tv_nsec = deadline % 1000000000;
                        if(timerfd_settime(timer, TFD_TIMER_ABSTIME, &spec, nullptr))
                        {
                            stop("timer-error");
                            send_message("ERROR " + std::to_string(id) + " timer setup failed\n");
                            continue;
                        }
                    }
                    connected = send_message("TIMED " + std::to_string(id) + " " +
                                             std::to_string(deadline) + "\n");
                }
                else if(sscanf(line.c_str(), "STOP %" SCNu64 " %c", &id, &extra) == 1 &&
                        id == capture && id)
                    connected = stop("request");
                else
                    connected =
                        send_message("ERROR " + std::to_string(id) + " invalid or stale command\n");
                if(!connected) break;
            }
        }
    }
    if(active) stop("disconnect");
    close_in_child();
    unlink(path);
    return nullptr;
}

__attribute__((constructor)) void
initialize()
{
    if(!std::getenv("ROCPROFV3_GDB_SOCKET")) return;
    auto* owner = std::getenv("ROCPROFV3_GDB_OWNER");
    if(owner && std::strtol(owner, nullptr, 10) != getpid()) return;
    char pid[32];
    std::snprintf(pid, sizeof(pid), "%d", getpid());
    setenv("ROCPROFV3_GDB_OWNER", pid, 1);
    pthread_atfork(nullptr, nullptr, close_in_child);
    pthread_t thread;
    auto      error = pthread_create(&thread, nullptr, serve, nullptr);
    if(!error)
        pthread_detach(thread);
    else
        std::fprintf(stderr, "[att] unable to create helper thread: %s\n", strerror(error));
}
}  // namespace
