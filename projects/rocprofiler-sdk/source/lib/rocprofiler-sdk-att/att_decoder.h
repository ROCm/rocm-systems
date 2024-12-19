// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifdef __cplusplus
#    include <cstddef>
#    include <cstdint>
extern "C" {
#else
#    include <stddef.h>
#    include <stdint.h>
#endif

typedef enum
{
    ROCPROFILER_ATT_DECODER_STATUS_SUCCESS = 0,
    ROCPROFILER_ATT_DECODER_STATUS_ERROR,
    ROCPROFILER_ATT_DECODER_STATUS_ERROR_OUT_OF_RESOURCES,
    ROCPROFILER_ATT_DECODER_STATUS_ERROR_INVALID_ARGUMENT,
    ROCPROFILER_ATT_DECODER_STATUS_ERROR_INVALID_SHADER_DATA,
    ROCPROFILER_ATT_DECODER_STATUS_LAST
} rocprofiler_att_decoder_status_t;

typedef enum
{
    ROCPROFILER_ATT_DECODER_INFO_NONE = 0,
    ROCPROFILER_ATT_DECODER_INFO_DATA_LOST,
    ROCPROFILER_ATT_DECODER_INFO_STITCH_INCOMPLETE,
    ROCPROFILER_ATT_DECODER_INFO_LAST
} rocprofiler_att_decoder_info_t;

typedef enum
{
    ROCPROFILER_ATT_DECODER_TYPE_GFXIP = 0,
    ROCPROFILER_ATT_DECODER_TYPE_OCCUPANCY,
    ROCPROFILER_ATT_DECODER_TYPE_PERFEVENT,
    ROCPROFILER_ATT_DECODER_TYPE_WAVE,
    ROCPROFILER_ATT_DECODER_TYPE_INFO,
    ROCPROFILER_ATT_DECODER_TYPE_DEBUG,
    ROCPROFILER_ATT_DECODER_TYPE_LAST
} rocprofiler_att_decoder_record_type_t;

typedef struct
{
    size_t addr;
    size_t marker_id;
} pcinfo_t;

typedef struct
{
    pcinfo_t pc;
    uint64_t time;
    uint8_t  se;
    uint8_t  cu;
    uint8_t  simd;
    uint8_t  slot;
    uint32_t start : 1;
    uint32_t _rsvd : 31;
} att_occupancy_info_v2_t;

typedef struct
{
    int32_t type;
    int32_t duration;
} att_wave_state_t;

typedef struct
{
    uint32_t category : 8;
    uint32_t stall    : 24;
    int32_t  duration;
    int64_t  time;
    pcinfo_t pc;
} att_wave_instruction_t;

typedef enum
{
    ATT_WAVE_STATE_EMPTY = 0,
    ATT_WAVE_STATE_LAST  = 5
} att_waveslot_state_t;

typedef enum
{
    ATT_INST_NONE = 0,
    ATT_INST_LAST = 11,
} att_wave_inst_category_t;

typedef struct
{
    uint8_t cu;
    uint8_t simd;
    uint8_t wave_id;
    uint8_t contexts;

    uint32_t _rsvd;
    size_t   traceID;

    int64_t begin_time;
    int64_t end_time;

    size_t                  timeline_size;
    size_t                  instructions_size;
    att_wave_state_t*       timeline_array;
    att_wave_instruction_t* instructions_array;
} att_wave_data_t;

typedef rocprofiler_att_decoder_status_t (*rocprofiler_att_decoder_isa_callback_t)(
    char*     instruction,
    uint64_t* memory_size,
    uint64_t* size,
    pcinfo_t  address,
    void*     userdata);

typedef rocprofiler_att_decoder_status_t (*rocprofiler_att_decoder_trace_callback_t)(
    rocprofiler_att_decoder_record_type_t record_type_id,
    int                                   shader_engine_id,
    void*                                 trace_events,
    uint64_t                              trace_size,
    void*                                 userdata);

typedef uint64_t (*rocprofiler_att_decoder_se_data_callback_t)(int*      shader_engine_id,
                                                               uint8_t** buffer,
                                                               uint64_t* buffer_size,
                                                               void*     userdata);

rocprofiler_att_decoder_status_t
rocprofiler_att_decoder_parse_data(rocprofiler_att_decoder_se_data_callback_t se_data_callback,
                                   rocprofiler_att_decoder_trace_callback_t   trace_callback,
                                   rocprofiler_att_decoder_isa_callback_t     isa_callback,
                                   void*                                      userdata);

const char*
rocprofiler_att_decoder_get_info_string(rocprofiler_att_decoder_info_t info);

const char*
rocprofiler_att_decoder_get_status_string(rocprofiler_att_decoder_status_t status);

#ifdef __cplusplus
}
#endif
