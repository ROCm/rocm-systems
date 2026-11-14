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

namespace rocprofiler
{
namespace hsa
{
namespace async_copy
{
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

// Drain outstanding memory copy completion callbacks. Returns the number still undelivered
// when the wait expires, so ZERO means drained.
uint64_t
async_copy_sync(const uint64_t timeout_ns = 0);

void
async_copy_fini();

// Copy completion callbacks still outstanding, without waiting.
uint64_t
async_copy_peek_count();
}  // namespace hsa
}  // namespace rocprofiler
