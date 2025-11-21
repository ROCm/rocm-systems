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
#include <string>
#include <thread>
#include <vector>

#include "common/fwd.hpp"

void
run(const std::string& name, pthread_barrier_t* _barrier)
{
    if(_barrier) pthread_barrier_wait(_barrier);

    if(hip_init_fn)
    {
        if(_barrier) pthread_barrier_wait(_barrier);
        hip_init_fn();
    }

    if(hsa_init_fn)
    {
        if(_barrier) pthread_barrier_wait(_barrier);
        hsa_init_fn();
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

void
run_threads(unsigned long n)
{
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
}

int
main(int argc, char** argv)
{
    unsigned long n = 4;
    if(argc > 1) n = std::stoul(argv[1]);

    resolve_symbols<ROCP_REG_TEST_HIP | ROCP_REG_TEST_ROCTX>();

    run_threads(n);

    return 0;
}
