// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "library/cpu_load.hpp"
#include "library/components/cpu_load.hpp"

#include <chrono>
#include <cstdint>
#include <deque>
#include <map>
#include <utility>

namespace rocprofsys
{
namespace cpu_load
{
namespace
{
struct cpu_load_sample
{
    uint64_t                   timestamp_ns;  // Nanoseconds since epoch
    std::map<uint64_t, double> loads;         // CPU ID -> load percentage
};

std::deque<cpu_load_sample> g_samples;

// Helper: Get current timestamp in nanoseconds
uint64_t
get_timestamp_ns()
{
    using namespace std::chrono;
    auto now = steady_clock::now();
    auto ns  = duration_cast<nanoseconds>(now.time_since_epoch());
    return static_cast<uint64_t>(ns.count());
}

}  // anonymous namespace

void
setup()
{
    // Initialize any global state if needed
    // For now, no Perfetto integration
}

void
config()
{
    // Configure the component
    component::cpu_load::configure();
}

void
sample()
{
    // Get current timestamp
    uint64_t timestamp = get_timestamp_ns();

    // Collect CPU load data
    component::cpu_load loader;
    loader.sample();

    // Store sample
    cpu_load_sample sample_data;
    sample_data.timestamp_ns = timestamp;
    sample_data.loads        = loader.get_loads();

    g_samples.push_back(std::move(sample_data));
}

void
shutdown()
{
    // Clean up
    g_samples.clear();
}

void
post_process()
{
    // Process collected data
    // TODO: Write to Perfetto/ROCPD traces when integrated

    // For now, just clear the data
    g_samples.clear();
}

}  // namespace cpu_load
}  // namespace rocprofsys
