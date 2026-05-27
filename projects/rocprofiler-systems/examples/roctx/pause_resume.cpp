// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Demonstrates the roctx Pause/Resume feature for selective profiling.
//
// Code flow:
//   CodeBlock_Z          (profiled)
//   CodeBlock_A          (profiled)
//   roctxProfilerPause
//   CodeBlock_B          (NOT profiled — paused)
//   roctxProfilerResume
//   CodeBlock_C          (profiled)
//   CodeBlock_D          (profiled)
//
// Run with profiling:
//   rocprof-sys -- ./pause_resume
//
// Expected: profiling data recorded for {CodeBlock_Z, CodeBlock_A, CodeBlock_C,
// CodeBlock_D}

#include "roctx_example_kernels.hpp"

DEFINE_KERNEL(CodeBlock_Z, 10)
DEFINE_KERNEL(CodeBlock_A, 20)
DEFINE_KERNEL(CodeBlock_B, 30)
DEFINE_KERNEL(CodeBlock_C, 40)
DEFINE_KERNEL(CodeBlock_D, 50)

int
main()
{
    gpu_buffer buf;
    float*     d = buf.get();

    roctx_thread_id_t roctx_tid{};
    roctxGetThreadId(&roctx_tid);

    LAUNCH_KERNEL(CodeBlock_Z, d);

    LAUNCH_KERNEL(CodeBlock_A, d);

    roctxProfilerPause(roctx_tid);

    LAUNCH_KERNEL(CodeBlock_B, d);

    roctxProfilerResume(roctx_tid);

    LAUNCH_KERNEL(CodeBlock_C, d);

    LAUNCH_KERNEL(CodeBlock_D, d);

    return 0;
}
