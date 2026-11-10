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

#include "lib/rocprofiler-sdk-collector/daemon.hpp"

#include "lib/common/logging.hpp"

#include <lttng/lttng.h>

#include <fcntl.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <thread>
#include <vector>

extern char** environ;  // NOLINT(readability-redundant-declaration)

namespace rocprofiler
{
namespace collector
{
namespace
{
namespace fs = std::filesystem;

constexpr auto daemon_ready_timeout = std::chrono::seconds{10};
constexpr auto daemon_stop_timeout  = std::chrono::seconds{10};
constexpr auto poll_interval        = std::chrono::milliseconds{20};

bool
is_executable(const fs::path& path)
{
    return !path.empty() && ::access(path.c_str(), X_OK) == 0 && fs::is_regular_file(path);
}

fs::path
find_in_path(const char* program)
{
    const auto* path_env = std::getenv("PATH");
    if(!path_env) return {};

    auto stream = std::istringstream{path_env};
    auto dir    = std::string{};
    while(std::getline(stream, dir, ':'))
    {
        if(dir.empty()) continue;
        auto candidate = fs::path{dir} / program;
        if(is_executable(candidate)) return candidate;
    }
    return {};
}

// Wait for pid to exit, up to timeout; true if it did (and was reaped).
bool
wait_for_exit(pid_t pid, std::chrono::steady_clock::duration timeout)
{
    auto deadline = std::chrono::steady_clock::now() + timeout;
    while(std::chrono::steady_clock::now() < deadline)
    {
        auto status = 0;
        auto ret    = ::waitpid(pid, &status, WNOHANG);
        if(ret == pid || (ret < 0 && errno == ECHILD)) return true;
        std::this_thread::sleep_for(poll_interval);
    }
    return false;
}
}  // namespace

rocprofiler_collector_status_t
find_lttng_tools(const std::string& prefix, lttng_tools* tools)
{
    auto sessiond = fs::path{};
    auto base     = fs::path{};
    if(!prefix.empty())
    {
        base     = fs::path{prefix};
        sessiond = base / "bin" / "lttng-sessiond";
        if(!is_executable(sessiond)) sessiond.clear();
    }
    else
    {
        sessiond = find_in_path("lttng-sessiond");
        if(!sessiond.empty()) base = sessiond.parent_path().parent_path();
    }

    if(sessiond.empty())
    {
        ROCP_ERROR << "[rocprofiler-sdk-collector] lttng-sessiond not found "
                   << (prefix.empty() ? std::string{"in PATH"} : "under " + prefix);
        return ROCPROFILER_COLLECTOR_STATUS_ERROR_LTTNG_NOT_FOUND;
    }

    tools->sessiond    = sessiond.string();
    auto consumerd     = base / "lib" / "lttng" / "libexec" / "lttng-consumerd";
    tools->consumerd64 = is_executable(consumerd) ? consumerd.string() : std::string{};
    auto xsd_dir       = base / "share" / "xml" / "lttng";
    tools->xsd_dir     = fs::exists(xsd_dir / "session.xsd") ? xsd_dir.string() : std::string{};
    return ROCPROFILER_COLLECTOR_STATUS_SUCCESS;
}

rocprofiler_collector_status_t
make_lttng_home(std::string* home)
{
    // The daemon's sockets are under LTTNG_HOME/.lttng; a Unix socket path is limited to
    // 108 bytes, so prefer a short temporary directory.
    auto tmpdir = fs::path{"/tmp"};
    if(const auto* env = std::getenv("TMPDIR");
       env != nullptr && std::strlen(env) > 0 && std::strlen(env) < 48 && fs::is_directory(env))
        tmpdir = env;

    auto templ = (tmpdir / "rocprofiler-collector-XXXXXX").string();
    auto buf   = std::vector<char>{templ.begin(), templ.end()};
    buf.emplace_back('\0');
    if(!::mkdtemp(buf.data()))
    {
        ROCP_ERROR << "[rocprofiler-sdk-collector] cannot create " << templ << ": "
                   << std::strerror(errno);
        return ROCPROFILER_COLLECTOR_STATUS_ERROR_SESSION_DAEMON;
    }
    *home = std::string{buf.data()};
    return ROCPROFILER_COLLECTOR_STATUS_SUCCESS;
}

rocprofiler_collector_status_t
start_session_daemon(const lttng_tools& tools,
                     const std::string& home,
                     const std::string& log_path,
                     session_daemon*    daemon)
{
    daemon->home     = home;
    daemon->log_path = log_path;

    auto args = std::vector<std::string>{tools.sessiond, "--no-kernel"};
    // The daemon's compiled-in consumer path is where lttng-tools was built; the bundled
    // tools are installed elsewhere.
    if(!tools.consumerd64.empty()) args.emplace_back("--consumerd64-path=" + tools.consumerd64);
    auto argv = std::vector<char*>{};
    for(auto& itr : args)
        argv.emplace_back(itr.data());
    argv.emplace_back(nullptr);

    // The daemon's environment: this process's, with the schema directory when known (the
    // daemon validates session configurations at start and exits if the schema is missing).
    // Built before fork: the child of a multi-threaded process must not allocate.
    auto env_strings = std::vector<std::string>{};
    for(char** itr = environ; itr && *itr; ++itr)
    {
        if(std::strncmp(*itr, "LTTNG_SESSION_CONFIG_XSD_PATH=", 30) != 0)
            env_strings.emplace_back(*itr);
    }
    if(!tools.xsd_dir.empty())
        env_strings.emplace_back("LTTNG_SESSION_CONFIG_XSD_PATH=" + tools.xsd_dir);
    auto envp = std::vector<char*>{};
    for(auto& itr : env_strings)
        envp.emplace_back(itr.data());
    envp.emplace_back(nullptr);

    auto parent = ::getpid();
    auto pid    = ::fork();
    if(pid < 0)
    {
        ROCP_ERROR << "[rocprofiler-sdk-collector] fork failed: " << std::strerror(errno);
        return ROCPROFILER_COLLECTOR_STATUS_ERROR_SESSION_DAEMON;
    }
    if(pid == 0)
    {
        // The daemon must not outlive the collector's process.
        ::prctl(PR_SET_PDEATHSIG, SIGTERM);
        if(::getppid() != parent) ::_exit(1);
        auto fd = ::open(log_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        if(fd >= 0)
        {
            ::dup2(fd, STDOUT_FILENO);
            ::dup2(fd, STDERR_FILENO);
        }
        ::execve(argv.front(), argv.data(), envp.data());
        ::_exit(127);
    }
    daemon->pid = pid;

    auto deadline = std::chrono::steady_clock::now() + daemon_ready_timeout;
    while(std::chrono::steady_clock::now() < deadline)
    {
        if(::lttng_session_daemon_alive() == 1) return ROCPROFILER_COLLECTOR_STATUS_SUCCESS;

        auto status = 0;
        if(::waitpid(pid, &status, WNOHANG) == pid)
        {
            daemon->pid = -1;
            ROCP_ERROR << "[rocprofiler-sdk-collector] lttng-sessiond exited during start "
                       << "(status " << status << "); see " << log_path;
            return ROCPROFILER_COLLECTOR_STATUS_ERROR_SESSION_DAEMON;
        }
        std::this_thread::sleep_for(poll_interval);
    }

    ROCP_ERROR << "[rocprofiler-sdk-collector] lttng-sessiond did not accept commands within "
               << daemon_ready_timeout.count() << " s; see " << log_path;
    stop_session_daemon(daemon);
    return ROCPROFILER_COLLECTOR_STATUS_ERROR_SESSION_DAEMON;
}

void
stop_session_daemon(session_daemon* daemon)
{
    if(daemon->pid > 0)
    {
        ::kill(daemon->pid, SIGTERM);
        if(!wait_for_exit(daemon->pid, daemon_stop_timeout))
        {
            ROCP_WARNING << "[rocprofiler-sdk-collector] lttng-sessiond did not exit; killing it";
            ::kill(daemon->pid, SIGKILL);
            wait_for_exit(daemon->pid, daemon_stop_timeout);
        }
        daemon->pid = -1;
    }

    if(!daemon->home.empty())
    {
        auto ec = std::error_code{};
        fs::remove_all(daemon->home, ec);
        if(ec)
            ROCP_WARNING << "[rocprofiler-sdk-collector] cannot remove " << daemon->home << ": "
                         << ec.message();
        daemon->home.clear();
    }
}
}  // namespace collector
}  // namespace rocprofiler
