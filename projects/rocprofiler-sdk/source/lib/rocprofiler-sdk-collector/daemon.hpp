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

#pragma once

#include <rocprofiler-sdk-collector/types.h>

#include <sys/types.h>

#include <string>

namespace rocprofiler
{
namespace collector
{
// The lttng-tools programs a collector runs.
struct lttng_tools
{
    std::string sessiond    = {};  // lttng-sessiond
    std::string consumerd64 = {};  // lttng-consumerd; empty: the daemon's compiled-in path
    std::string xsd_dir     = {};  // session.xsd's directory; empty: the compiled-in path
};

// Find lttng-sessiond: under prefix (bin/lttng-sessiond) when prefix is set, otherwise in
// PATH. lttng-consumerd (lib/lttng/libexec/lttng-consumerd) and the session schema
// (share/xml/lttng/session.xsd) are under the same prefix (for a PATH match, the directory
// above bin/). The daemon has their build-time paths compiled in, which do not exist when
// the tools are installed somewhere else, as ROCm's bundled lttng-tools are.
rocprofiler_collector_status_t
find_lttng_tools(const std::string& prefix, lttng_tools* tools);

// A private session daemon: its own LTTNG_HOME, so its sessions and the applications that
// register with it are separate from any other LTTng user on the node.
struct session_daemon
{
    pid_t       pid      = -1;
    std::string home     = {};  // LTTNG_HOME of the daemon and of the traced processes
    std::string log_path = {};
};

// Create a private LTTNG_HOME under the temporary directory (short, since the daemon's
// Unix socket paths derive from it) and return its path.
rocprofiler_collector_status_t
make_lttng_home(std::string* home);

// Start lttng-sessiond with LTTNG_HOME=daemon->home (the caller sets it in this process's
// environment first, so liblttng-ctl reaches the same daemon) and wait until it accepts
// commands. Its output goes to log_path. The daemon is stopped if this process dies.
rocprofiler_collector_status_t
start_session_daemon(const lttng_tools& tools,
                     const std::string& home,
                     const std::string& log_path,
                     session_daemon*    daemon);

// Stop the daemon (SIGTERM, then SIGKILL after a timeout) and remove its LTTNG_HOME.
void
stop_session_daemon(session_daemon* daemon);
}  // namespace collector
}  // namespace rocprofiler
