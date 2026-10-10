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

// The C half of the range-replay ABI test, mirroring
// kernel_replay/tests/replay_abi_c.c. See that file for why a C translation unit exists at
// all: the public headers are consumed by C tools, but nothing else in the tree compiles
// them as C, so a C++-only construct reaching one would go unnoticed until a downstream
// tool failed to build. Compiling this file is that check, and the offsets it reports let
// the C++ side confirm both languages derive the same layout from the same header.

#include <rocprofiler-sdk/experimental/range_replay.h>

#include <stddef.h>
#include <stdint.h>

typedef struct rocprofiler_callback_tracing_range_replay_data_t range_data_c_t;

size_t
rocprofiler_test_c_range_record_size(void)
{
    return sizeof(range_data_c_t);
}

size_t
rocprofiler_test_c_range_record_align(void)
{
    return _Alignof(range_data_c_t);
}

size_t
rocprofiler_test_c_range_offset_size(void)
{
    return offsetof(range_data_c_t, size);
}

size_t
rocprofiler_test_c_range_offset_range_id(void)
{
    return offsetof(range_data_c_t, range_id);
}

size_t
rocprofiler_test_c_range_offset_pass_count_cb(void)
{
    return offsetof(range_data_c_t, pass_count_cb);
}

size_t
rocprofiler_test_c_range_offset_replay_continue_cb(void)
{
    return offsetof(range_data_c_t, replay_continue_cb);
}

size_t
rocprofiler_test_c_range_offset_current_pass(void)
{
    return offsetof(range_data_c_t, current_pass);
}

size_t
rocprofiler_test_c_range_offset_total_passes(void)
{
    return offsetof(range_data_c_t, total_passes);
}

size_t
rocprofiler_test_c_range_offset_local_start_context(void)
{
    return offsetof(range_data_c_t, replay_local_start_context_cb);
}

size_t
rocprofiler_test_c_range_offset_local_stop_context(void)
{
    return offsetof(range_data_c_t, replay_local_stop_context_cb);
}

size_t
rocprofiler_test_c_range_offset_agent_id(void)
{
    return offsetof(range_data_c_t, agent_id);
}

size_t
rocprofiler_test_c_range_offset_dispatch_count(void)
{
    return offsetof(range_data_c_t, dispatch_count);
}

size_t
rocprofiler_test_c_range_offset_status(void)
{
    return offsetof(range_data_c_t, status);
}

size_t
rocprofiler_test_c_range_offset_divergence_count(void)
{
    return offsetof(range_data_c_t, divergence_count);
}

// Enum values as C sees them. An enum whose underlying type differs between the languages would
// show up as a mismatch here rather than as a wrong branch taken at runtime in a tool.
int
rocprofiler_test_c_range_operation_last(void)
{
    return (int) ROCPROFILER_RANGE_REPLAY_LAST;
}

int
rocprofiler_test_c_range_status_last(void)
{
    return (int) ROCPROFILER_RANGE_REPLAY_STATUS_LAST;
}

int
rocprofiler_test_c_range_tracing_kind(void)
{
    return (int) ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY;
}

// The C compiler's view of the status enum's size. The SDK writes `status` as a member of the
// record, so a language disagreement about the enum's underlying type shifts every field after it.
size_t
rocprofiler_test_c_range_status_enum_size(void)
{
    return sizeof(rocprofiler_range_replay_status_t);
}

// Fill a CLOSE record the way the SDK does, from C, so the C++ side can verify it reads back the
// same values through its own view of the struct.
void
rocprofiler_test_c_range_fill_close(void*    record,
                                    uint64_t range_id,
                                    uint64_t dispatch_count,
                                    int      status,
                                    uint64_t divergence_count)
{
    range_data_c_t* rec = (range_data_c_t*) record;

    rec->size             = sizeof(range_data_c_t);
    rec->range_id         = range_id;
    rec->dispatch_count   = dispatch_count;
    rec->status           = (rocprofiler_range_replay_status_t) status;
    rec->divergence_count = divergence_count;
}
