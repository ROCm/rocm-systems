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

#include <rocprofiler-sdk/fwd.h>

ROCPROFILER_EXTERN_C_INIT

size_t
get_number_of_agents() ROCPROFILER_EXPORT;
void
agent_handles(uint64_t* agent_handles, size_t num_agents) ROCPROFILER_EXPORT;
uint64_t
get_agent_id(uint64_t agent_handle, int id_type) ROCPROFILER_EXPORT;
size_t
get_number_of_agent_counters(uint64_t agent_handle) ROCPROFILER_EXPORT;
void
agent_counter_handles(uint64_t* counter_handles,
                      uint64_t  agent_handle,
                      size_t    number_of_agent_counters) ROCPROFILER_EXPORT;
void
counter_info(uint64_t     counter_handle,
             const char** counter_name,
             const char** counter_description,
             uint8_t*     is_derived,
             uint8_t*     is_hw_constant) ROCPROFILER_EXPORT;
void
counter_block(uint64_t counter_handle, const char** counter_block) ROCPROFILER_EXPORT;

void
counter_expression(uint64_t counter_handle, const char** counter_expr) ROCPROFILER_EXPORT;

size_t
get_number_of_dimensions(uint64_t counter_handle) ROCPROFILER_EXPORT;

void
counter_dimension_ids(uint64_t  counter_handle,
                      uint64_t* dimension_ids,
                      size_t    num_dimensions) ROCPROFILER_EXPORT;

void
counter_dimension(uint64_t     counter_handle,
                  uint64_t     dimension_handle,
                  const char** dimension_name,
                  uint64_t*    dimension_instance) ROCPROFILER_EXPORT;

size_t
get_number_of_pc_sample_configs(uint64_t agent_handle) ROCPROFILER_EXPORT;

void
pc_sample_config(uint64_t  agent_handle,
                 uint64_t  config_idx,
                 uint64_t* method,
                 uint64_t* unit,
                 uint64_t* min_interval,
                 uint64_t* max_interval,
                 uint64_t* flags) ROCPROFILER_EXPORT;
bool
is_counter_set(uint64_t* counter_handles,
               uint64_t  agent_handle,
               size_t    num_counters) ROCPROFILER_EXPORT;

void
agent_info(uint64_t agent_handle, const char** agent_info_str) ROCPROFILER_EXPORT;

ROCPROFILER_EXTERN_C_FINI
