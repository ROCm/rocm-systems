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

#include <stddef.h>
#include <stdint.h>

typedef enum rocprofiler_thread_trace_decoder_info_t
{
    ROCPROFILER_THREAD_TRACE_DECODER_INFO_NONE = 0,
    ROCPROFILER_THREAD_TRACE_DECODER_INFO_DATA_LOST,
    ROCPROFILER_THREAD_TRACE_DECODER_INFO_STITCH_INCOMPLETE,
    ROCPROFILER_THREAD_TRACE_DECODER_INFO_LAST
} rocprofiler_thread_trace_decoder_info_t;

typedef struct rocprofiler_thread_trace_decoder_pc_t
{
    size_t addr;
    size_t marker_id;
} rocprofiler_thread_trace_decoder_pc_t;

typedef struct rocprofiler_thread_trace_decoder_perfevent_t
{
    int64_t  time;
    uint16_t events0;
    uint16_t events1;
    uint16_t events2;
    uint16_t events3;
    uint8_t  CU;
    uint8_t  bank;
} rocprofiler_thread_trace_decoder_perfevent_t;

typedef struct rocprofiler_thread_trace_decoder_occupancy_t
{
    rocprofiler_thread_trace_decoder_pc_t pc;
    uint64_t                              time;
    uint8_t                               se;
    uint8_t                               cu;
    uint8_t                               simd;
    uint8_t                               slot;
    uint32_t                              start : 1;
    uint32_t                              _rsvd : 31;
} rocprofiler_thread_trace_decoder_occupancy_t;

typedef enum rocprofiler_thread_trace_decoder_wstate_type_t
{
    ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_EMPTY = 0,
    ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_IDLE,
    ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_EXEC,
    ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_WAIT,
    ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_STALL,
    ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_LAST,
} rocprofiler_thread_trace_decoder_wstate_type_t;

typedef struct rocprofiler_thread_trace_decoder_wave_state_t
{
    int32_t type;  // One of rocprofiler_thread_trace_decoder_waveslot_state_type_t
    int32_t duration;
} rocprofiler_thread_trace_decoder_wave_state_t;

typedef enum rocprofiler_thread_trace_decoder_inst_category_t
{
    ROCPROFILER_THREAD_TRACE_DECODER_INST_NONE = 0,
    ROCPROFILER_THREAD_TRACE_DECODER_INST_SMEM,
    ROCPROFILER_THREAD_TRACE_DECODER_INST_SALU,
    ROCPROFILER_THREAD_TRACE_DECODER_INST_VMEM,
    ROCPROFILER_THREAD_TRACE_DECODER_INST_FLAT,
    ROCPROFILER_THREAD_TRACE_DECODER_INST_LDS,
    ROCPROFILER_THREAD_TRACE_DECODER_INST_VALU,
    ROCPROFILER_THREAD_TRACE_DECODER_INST_JUMP,
    ROCPROFILER_THREAD_TRACE_DECODER_INST_NEXT,
    ROCPROFILER_THREAD_TRACE_DECODER_INST_IMMED,
    ROCPROFILER_THREAD_TRACE_DECODER_INST_CONTEXT,
    ROCPROFILER_THREAD_TRACE_DECODER_INST_MESSAGE,
    ROCPROFILER_THREAD_TRACE_DECODER_INST_BVH,
    ROCPROFILER_THREAD_TRACE_DECODER_INST_LAST
} rocprofiler_thread_trace_decoder_inst_category_t;

typedef struct rocprofiler_thread_trace_decoder_inst_t
{
    uint32_t category : 8;  // One of rocprofiler_thread_trace_decoder_inst_category_t
    uint32_t stall    : 24;
    int32_t  duration;
    int64_t  time;
    rocprofiler_thread_trace_decoder_pc_t pc;
} rocprofiler_thread_trace_decoder_inst_t;

typedef struct rocprofiler_thread_trace_decoder_wave_t
{
    uint8_t cu;
    uint8_t simd;
    uint8_t wave_id;
    uint8_t contexts;

    uint32_t _rsvd1;
    uint32_t _rsvd2;
    uint32_t _rsvd3;

    int64_t begin_time;
    int64_t end_time;

    size_t                                         timeline_size;
    size_t                                         instructions_size;
    rocprofiler_thread_trace_decoder_wave_state_t* timeline_array;
    rocprofiler_thread_trace_decoder_inst_t*       instructions_array;
} rocprofiler_thread_trace_decoder_wave_t;

typedef enum rocprofiler_thread_trace_decoder_record_type_t
{
    ROCPROFILER_THREAD_TRACE_DECODER_RECORD_GFXIP =
        0,  // Record is size_t representing the gfxip_major
    ROCPROFILER_THREAD_TRACE_DECODER_RECORD_OCCUPANCY,  // Record is pointer to
                                                        // rocprofiler_thread_trace_decoder_occupancy_t
    ROCPROFILER_THREAD_TRACE_DECODER_RECORD_PERFEVENT,  // Record is pointer to
                                                        // rocprofiler_thread_trace_decoder_perfevent_t
    ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE,   // Record is pointer to
                                                    // rocprofiler_thread_trace_decoder_wave_t
    ROCPROFILER_THREAD_TRACE_DECODER_RECORD_INFO,   // Record is pointer to
                                                    // rocprofiler_thread_trace_decoder_info_t
    ROCPROFILER_THREAD_TRACE_DECODER_RECORD_DEBUG,  // Debug
    ROCPROFILER_THREAD_TRACE_DECODER_RECORD_LAST
} rocprofiler_thread_trace_decoder_record_type_t;
