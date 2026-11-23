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

#include "lib/common/container/pool.hpp"

#include <gtest/gtest.h>

#include <sched.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <limits>
#include <thread>
#include <vector>

// The HSA signal pool is released on the async signal handler thread while queue write
// interceptors acquire from it. An exhausted pool grows inside acquire(), which reallocates the
// stable_vector chunk array that release() reads to bounds-check the index. Each round grows a
// fresh pool through its first, smallest reallocations while other threads keep calling
// release(), so an unlocked bounds check reads a freed chunk array and crashes. Those threads
// release an index the pool does not own, which release() ignores, so they never wait for an
// object and keep hitting the bounds check while the pool grows.
TEST(pool, release_while_growing)
{
    using pool_t = rocprofiler::common::container::pool<size_t>;

    constexpr size_t batch       = 32;
    constexpr size_t num_rounds  = 8000;
    constexpr size_t grow_to     = 3 * batch;
    constexpr size_t unowned_idx = std::numeric_limits<size_t>::max();

    auto _cpus = cpu_set_t{};
    CPU_ZERO(&_cpus);
    ASSERT_EQ(sched_getaffinity(0, sizeof(_cpus), &_cpus), 0);
    const size_t num_releasers = std::clamp<size_t>(CPU_COUNT(&_cpus), 2, 5) - 1;

    auto _pool    = std::atomic<pool_t*>{nullptr};
    auto _round   = std::atomic<size_t>{0};
    auto _running = std::atomic<bool>{false};
    auto _busy    = std::atomic<size_t>{0};
    auto _threads = std::vector<std::thread>{};

    for(size_t i = 0; i < num_releasers; ++i)
        _threads.emplace_back([&]() {
            for(size_t round = 1; round <= num_rounds; ++round)
            {
                while(_round.load() < round)
                    std::this_thread::yield();
                auto* _p = _pool.load();
                while(_running.load())
                    _p->release(unowned_idx);
                _busy.fetch_sub(1);
            }
        });

    for(size_t round = 1; round <= num_rounds; ++round)
    {
        auto _p = pool_t{std::piecewise_construct, batch, [](size_t& v) { v = 0; }};
        _pool.store(&_p);
        _busy.store(num_releasers);
        _running.store(true);
        _round.store(round);
        for(size_t i = 0; i < grow_to; ++i)
            _p.acquire();
        _running.store(false);
        while(_busy.load() > 0)
            std::this_thread::yield();
    }

    for(auto& itr : _threads)
        itr.join();
}
