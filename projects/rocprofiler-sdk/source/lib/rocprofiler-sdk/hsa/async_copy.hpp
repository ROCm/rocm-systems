// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include "lib/rocprofiler-sdk/hsa/hsa.hpp"

#include <optional>
#include <utility>

namespace rocprofiler
{
namespace hsa
{
namespace async_copy
{
using copy_time_t = std::pair<hsa_status_t, hsa_amd_profiling_async_copy_time_t>;

void
create_copy_signal(hsa_signal_t signal);

void
retain_copy_signal(hsa_signal_t signal);

void
release_copy_signal(hsa_signal_t signal);

void
record_copy_time(hsa_signal_t                        signal,
                 hsa_status_t                        status,
                 hsa_amd_profiling_async_copy_time_t time);

std::optional<copy_time_t>
find_copy_time(hsa_signal_t signal);

void
clear_copy_time_cache();

const char*
name_by_id(uint32_t id);

uint32_t
id_by_name(const char* name);

std::vector<const char*>
get_names();

std::vector<uint32_t>
get_ids();
}  // namespace async_copy

void
async_copy_init(hsa_api_table_t* _orig, uint64_t _tbl_instance);

void
async_copy_sync();

void
async_copy_fini();

void
async_copy_reset();
}  // namespace hsa
}  // namespace rocprofiler
