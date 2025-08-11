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

#include <rocprofiler-sdk/fwd.h>

#include <stdint.h>

/**
 * ######## ROCR Definitions ########
 * Some data types have been modified for better type safety.
 */

enum packet_header_t
{
    AMD_GENERIC_SAMPLE        = 0,
    AMD_DOORBELL_TO_QUEUE_MAP = 3,
    AMD_DISPATCH_PKT_WRAP,
    AMD_UPCOMING_SAMPLES,
    AMD_DISPATCH_PKT_ID,
};

enum upcoming_sample_t
{
    AMD_HOST_TRAP_V1 = 1,
    AMD_SNAPSHOT_V1  = 2
};

typedef uint32_t sample_enum;
typedef struct
{
    uint32_t handle;
} device_handle;
typedef uint32_t upcoming_sample_enum;
typedef struct
{
    uint32_t _;
} reserved_type;

typedef struct
{
    sample_enum   type;
    reserved_type _[15];
} generic_sample_t;

typedef struct
{
    sample_enum   type;
    device_handle device;
    uint32_t      doorbell_id;
    uint64_t      queue_size;
    uint64_t      write_index;
    uint64_t      read_index;
    /// both internal and external correlation ID.
    rocprofiler_async_correlation_id_t correlation_id;
    rocprofiler_dispatch_id_t          dispatch_id;
} dispatch_pkt_id_t;

typedef struct
{
    sample_enum          type;
    device_handle        device;
    upcoming_sample_enum which_sample_type;
    reserved_type        reserved0;
    uint64_t             num_samples;
    reserved_type        _[10];
} upcoming_samples_t;

typedef struct
{
    uint64_t      pc;
    uint64_t      exec_mask;
    uint32_t      workgroup_id_x;
    uint32_t      workgroup_id_y;
    uint32_t      workgroup_id_z;
    uint32_t      chiplet_and_wave_id;
    uint32_t      hw_id;
    reserved_type reserved[3];
    uint64_t      timestamp;
    uint64_t      correlation_id;
} perf_sample_host_trap_v1;

typedef struct
{
    uint64_t pc;
    uint64_t exec_mask;
    uint32_t workgroup_id_x;
    uint32_t workgroup_id_y;
    uint32_t workgroup_id_z;
    uint32_t chiplet_and_wave_id;
    uint32_t hw_id;
    uint32_t perf_snapshot_data;
    uint32_t perf_snapshot_data1;
    uint32_t perf_snapshot_data2;
    uint64_t timestamp;
    uint64_t correlation_id;
} perf_sample_snapshot_v1;

typedef union
{
    generic_sample_t         generic;
    perf_sample_snapshot_v1  snap;
    perf_sample_host_trap_v1 host;
    upcoming_samples_t       upcoming;
    dispatch_pkt_id_t        dispatch_id;
} packet_union_t;
