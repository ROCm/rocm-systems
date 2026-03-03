// MIT License
//
// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "trace_decoder_types.h"

enum WaveslotState
{
    WS_EMPTY = ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_EMPTY,
    WS_IDLE = ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_IDLE,
    WS_EXEC = ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_EXEC,
    WS_WAIT = ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_WAIT,
    WS_STALL = ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_STALL,
    WS_UNKNOWN = ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_LAST
};

enum WaveInstCategory
{
    NONE = ROCPROFILER_THREAD_TRACE_DECODER_INST_NONE,
    SMEM = ROCPROFILER_THREAD_TRACE_DECODER_INST_SMEM,
    SALU = ROCPROFILER_THREAD_TRACE_DECODER_INST_SALU,
    VMEM = ROCPROFILER_THREAD_TRACE_DECODER_INST_VMEM,
    FLAT = ROCPROFILER_THREAD_TRACE_DECODER_INST_FLAT,
    LDS = ROCPROFILER_THREAD_TRACE_DECODER_INST_LDS,
    VALU = ROCPROFILER_THREAD_TRACE_DECODER_INST_VALU,
    JUMP = ROCPROFILER_THREAD_TRACE_DECODER_INST_JUMP,
    NEXT = ROCPROFILER_THREAD_TRACE_DECODER_INST_NEXT,
    IMMED = ROCPROFILER_THREAD_TRACE_DECODER_INST_IMMED,
    TRAP = ROCPROFILER_THREAD_TRACE_DECODER_INST_CONTEXT,
    MSG = ROCPROFILER_THREAD_TRACE_DECODER_INST_MESSAGE,
    BVH = ROCPROFILER_THREAD_TRACE_DECODER_INST_BVH,
    DPMACC1,
    LD_SCALE,
    WAVE_NOT_FINISHED,
};

static_assert(int(ROCPROFILER_THREAD_TRACE_DECODER_INST_LAST) < int(WAVE_NOT_FINISHED));

enum WaveTrapStatus
{
    TRAP_RESTORED = 0,
    TRAP_REQUEST,
    TRAP_STANDBY,
    TRAP_WAIT_FOR_NEW_PC
};

typedef rocprofiler_thread_trace_decoder_pc_t pcinfo_t;
typedef rocprofiler_thread_trace_decoder_wave_state_t att_wave_state_t;
typedef rocprofiler_thread_trace_decoder_perfevent_t att_perfevent_t;
typedef rocprofiler_thread_trace_decoder_realtime_t att_decoder_realtime_t;
typedef rocprofiler_thread_trace_decoder_shaderdata_t att_shader_data_t;
typedef rocprofiler_thread_trace_decoder_inst_other_simd_t att_other_simd_t;
