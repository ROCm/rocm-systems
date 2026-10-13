// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <dlfcn.h>
#include <pthread.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "common/fwd.hpp"

void
run(const std::string& name, pthread_barrier_t* _barrier)
{
    if(_barrier) pthread_barrier_wait(_barrier);

    if(hsa_init_fn)
    {
        if(_barrier) pthread_barrier_wait(_barrier);
        hsa_init_fn();
    }

    if(hip_init_fn)
    {
        if(_barrier) pthread_barrier_wait(_barrier);
        hip_init_fn();
    }

    if(roctxRangePush_fn)
    {
        if(_barrier) pthread_barrier_wait(_barrier);
        roctxRangePush_fn(name.c_str());
    }

    if(roctxRangePop_fn)
    {
        if(_barrier) pthread_barrier_wait(_barrier);
        roctxRangePop_fn(name.c_str());
    }
}

auto
run_threads(unsigned long n)
{
    resolve_symbols<ROCP_REG_TEST_HIP>();

    auto threads = std::vector<std::thread>{};
    auto names   = std::vector<std::string>{};

    for(unsigned long i = 0; i < n; ++i)
        names.emplace_back(std::string{ "thread-" } + std::to_string(i));

    auto _barrier = pthread_barrier_t{};
    pthread_barrier_init(&_barrier, nullptr, n);

    for(unsigned long i = 0; i < n; ++i)
        threads.emplace_back(run, names.at(i), &_barrier);

    for(auto& itr : threads)
        itr.join();

    pthread_barrier_destroy(&_barrier);

    return n;
}

namespace
{
constexpr auto logging_environment_names = std::array{
    "GLOG_minloglevel",
    "GLOG_logtostderr",
    "GLOG_alsologtostderr",
    "GLOG_stderrthreshold",
    "GOOGLE_LOG_DIR",
    "GLOG_log_dir",
    "GLOG_vmodule",
};

using environment_value_t = std::optional<std::string>;
using environment_snapshot_t =
    std::array<environment_value_t, logging_environment_names.size()>;

environment_snapshot_t
snapshot_logging_environment()
{
    auto snapshot = environment_snapshot_t{};
    for(size_t i = 0; i < logging_environment_names.size(); ++i)
    {
        if(const auto* value = std::getenv(logging_environment_names.at(i)))
            snapshot.at(i) = value;
    }
    return snapshot;
}

bool
logging_environment_matches(const environment_snapshot_t& expected)
{
    const auto actual = snapshot_logging_environment();
    for(size_t i = 0; i < logging_environment_names.size(); ++i)
    {
        if(actual.at(i) != expected.at(i))
        {
            fprintf(stderr,
                    "rocprofiler-register changed environment variable %s during "
                    "initialization\n",
                    logging_environment_names.at(i));
            return false;
        }
    }
    return true;
}
}  // namespace

auto logging_environment = snapshot_logging_environment();
auto run_n = run_threads(4);

int
main()
{
    return (run_n == 4 && logging_environment_matches(logging_environment))
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}
