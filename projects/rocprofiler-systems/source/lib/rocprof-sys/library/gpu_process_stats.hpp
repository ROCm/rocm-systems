// MIT License
//
// Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include <cstdint>
#include <string>
#include <sys/types.h>

namespace rocprofsys
{
namespace gpu_process_stats
{
/// Process-specific GPU metrics data structure
struct process_info_t
{
    uint64_t timestamp;
    uint32_t process_id;
    uint64_t vram_usage;
    uint64_t sdma_usage;
    uint32_t cu_occupancy;
};

/// Configuration settings for process-specific GPU metrics
struct process_settings
{
    bool vram_usage   = false;
    bool sdma_usage   = false;
    bool cu_occupancy = false;
};

/// Get the process settings (singleton accessor)
process_settings&
get_process_settings();

/// Sample GPU process metrics for the given PID
void
sample(pid_t target_pid);

/// Post-process collected process samples (write to Perfetto)
void
post_process();

/// Initialize Perfetto tracks for process metrics
void
initialize_perfetto_tracks(pid_t target_pid);

/// Initialize PMC (Performance Monitoring Counters) info for process metrics
void
initialize_pmc();

/// Configure process settings from environment variables
void
configure(const std::string& env_value);

/// Initialize the module with the target process ID
void
initialize(pid_t target_pid);

}  // namespace gpu_process_stats
}  // namespace rocprofsys
