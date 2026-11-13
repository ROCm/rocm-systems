// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/rocprofiler-sdk-collector/launch.hpp"

#include "lib/common/logging.hpp"

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <string>
#include <vector>

namespace rocprofiler
{
namespace collector
{
namespace
{
// This process's environment with the extra variables; built before fork, since the child
// may only make async-signal-safe calls before exec.
std::vector<std::string>
make_environment(const std::vector<std::string>& extra_env)
{
    auto result = std::vector<std::string>{};
    for(char** itr = environ; itr != nullptr && *itr != nullptr; ++itr)
    {
        auto entry    = std::string{*itr};
        auto name     = entry.substr(0, entry.find('='));
        auto replaced = std::any_of(extra_env.begin(), extra_env.end(), [&name](const auto& extra) {
            return extra.compare(0, name.size() + 1, name + "=") == 0;
        });
        if(!replaced) result.push_back(std::move(entry));
    }
    result.insert(result.end(), extra_env.begin(), extra_env.end());
    return result;
}
}  // namespace

rocprofiler_collector_status_t
launch_process(const char* const* argv, const std::vector<std::string>& extra_env, int* pid)
{
    if(!argv || !argv[0] || !pid) return ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_ARGUMENT;

    auto environment = make_environment(extra_env);
    auto envp        = std::vector<char*>{};
    envp.reserve(environment.size() + 1);
    for(auto& itr : environment)
        envp.push_back(itr.data());
    envp.push_back(nullptr);

    int fds[2] = {-1, -1};
    if(::pipe2(fds, O_CLOEXEC) != 0)
    {
        ROCP_ERROR << "[rocprofiler-sdk-collector] pipe2 failed: " << std::strerror(errno);
        return ROCPROFILER_COLLECTOR_STATUS_ERROR_LAUNCH;
    }

    auto child = ::fork();
    if(child < 0)
    {
        ROCP_ERROR << "[rocprofiler-sdk-collector] fork failed: " << std::strerror(errno);
        ::close(fds[0]);
        ::close(fds[1]);
        return ROCPROFILER_COLLECTOR_STATUS_ERROR_LAUNCH;
    }
    if(child == 0)
    {
        ::close(fds[0]);
        // execvp's argv is char* const[] for historical reasons; it does not modify it.
        ::execvpe(argv[0],
                  const_cast<char* const*>(argv),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                  envp.data());
        auto err = errno;
        auto ret = ::write(fds[1], &err, sizeof(err));
        (void) ret;
        ::_exit(127);
    }

    ::close(fds[1]);
    auto err   = 0;
    auto nread = ssize_t{0};
    do
    {
        nread = ::read(fds[0], &err, sizeof(err));
    } while(nread < 0 && errno == EINTR);
    ::close(fds[0]);

    if(nread == static_cast<ssize_t>(sizeof(err)))
    {
        // exec failed: reap the child, which exits immediately.
        auto status = 0;
        while(::waitpid(child, &status, 0) < 0 && errno == EINTR)
        {}
        ROCP_ERROR << "[rocprofiler-sdk-collector] cannot execute " << argv[0] << ": "
                   << std::strerror(err);
        return ROCPROFILER_COLLECTOR_STATUS_ERROR_LAUNCH;
    }

    *pid = child;
    return ROCPROFILER_COLLECTOR_STATUS_SUCCESS;
}

rocprofiler_collector_status_t
wait_process(int pid, int* exit_status)
{
    auto status = 0;
    auto ret    = pid_t{0};
    do
    {
        ret = ::waitpid(pid, &status, 0);
    } while(ret < 0 && errno == EINTR);

    if(ret != pid)
    {
        ROCP_ERROR << "[rocprofiler-sdk-collector] waitpid(" << pid
                   << ") failed: " << std::strerror(errno);
        return ROCPROFILER_COLLECTOR_STATUS_ERROR;
    }

    if(WIFEXITED(status))
        *exit_status = WEXITSTATUS(status);
    else if(WIFSIGNALED(status))
        *exit_status = 128 + WTERMSIG(status);
    else
        *exit_status = -1;
    return ROCPROFILER_COLLECTOR_STATUS_SUCCESS;
}
}  // namespace collector
}  // namespace rocprofiler
