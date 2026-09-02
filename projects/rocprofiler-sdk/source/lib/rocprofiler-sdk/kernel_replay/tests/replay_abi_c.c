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

// The C half of the kernel-replay ABI test.
//
// This file exists to be *compiled*, in C, against only the public header. Most rocprofiler-sdk
// tools are C or are built by a C toolchain, but the SDK's own tests are C++ throughout (the
// directory named c-tool is a CXX project), so nothing else in the tree would notice if a C++-only
// construct -- a default argument, a `bool` without <stdbool.h>, a missing extern "C" guard, an
// enum used as a bitfield type -- reached a public header. Building this translation unit is that
// check: if the header stops being valid C99, the test target fails to build.
//
// It also reports what C thinks the record's layout is. The C++ side compares those numbers against
// its own, which catches a header change that both languages accept but interpret differently --
// the failure mode where a C tool silently reads the wrong bytes rather than failing to compile.

#include <rocprofiler-sdk/experimental/kernel_replay.h>

#include <stddef.h>
#include <stdint.h>

typedef struct rocprofiler_callback_tracing_kernel_replay_data_t replay_data_c_t;

size_t
rocprofiler_test_c_replay_record_size(void)
{
    return sizeof(replay_data_c_t);
}

size_t
rocprofiler_test_c_replay_record_align(void)
{
    return _Alignof(replay_data_c_t);
}

size_t
rocprofiler_test_c_replay_offset_size(void)
{
    return offsetof(replay_data_c_t, size);
}

size_t
rocprofiler_test_c_replay_offset_dispatch_info(void)
{
    return offsetof(replay_data_c_t, dispatch_info);
}

size_t
rocprofiler_test_c_replay_offset_pass_count_cb(void)
{
    return offsetof(replay_data_c_t, pass_count_cb);
}

size_t
rocprofiler_test_c_replay_offset_replay_continue_cb(void)
{
    return offsetof(replay_data_c_t, replay_continue_cb);
}

size_t
rocprofiler_test_c_replay_offset_current_pass(void)
{
    return offsetof(replay_data_c_t, current_pass);
}

size_t
rocprofiler_test_c_replay_offset_total_passes(void)
{
    return offsetof(replay_data_c_t, total_passes);
}

size_t
rocprofiler_test_c_replay_offset_local_enable_cb(void)
{
    return offsetof(replay_data_c_t, replay_local_enable_context_cb);
}

size_t
rocprofiler_test_c_replay_offset_local_disable_cb(void)
{
    return offsetof(replay_data_c_t, replay_local_disable_context_cb);
}

size_t
rocprofiler_test_c_replay_offset_reserved_padding(void)
{
    return offsetof(replay_data_c_t, reserved_padding);
}

// Enum values as C sees them. An enum whose underlying type differs between the languages would
// show up as a mismatch here rather than as a wrong branch taken at runtime in a tool.
int
rocprofiler_test_c_replay_operation_last(void)
{
    return (int) ROCPROFILER_KERNEL_REPLAY_LAST;
}

int
rocprofiler_test_c_replay_tracing_kind(void)
{
    return (int) ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY;
}

// Fill a record the way the SDK does, from C, so the C++ side can verify it reads back the same
// values through its own view of the struct.
void
rocprofiler_test_c_replay_fill(void* record, uint64_t current_pass, uint64_t total_passes)
{
    replay_data_c_t* rec = (replay_data_c_t*) record;

    rec->size         = offsetof(replay_data_c_t, reserved_padding);
    rec->current_pass = current_pass;
    rec->total_passes = total_passes;
}
