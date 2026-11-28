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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

// The boundary between the rocprofiler_synchronize_context tools and the applications they
// profile. The application publishes the GPU operations it can perform; the tool drives them and
// judges the records they produce. A tool holds no HIP, an application holds no rocprofiler.
//
// The application resolves SYNC_TEST_RUN_SYMBOL out of the preloaded tool and hands it the
// workload. The return value is the number of failed expectations.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// A stream of GPU work the tool can enqueue onto and wait on. One lane is one ordered
/// sequence; separate lanes make progress independently.
typedef struct sync_test_workload
{
    /// Stamped by the application with its own sizeof, so a tool can tell which members it
    /// may read.
    uint64_t size;

    /// The most lanes a tool may hold open at once. An application backed by the default
    /// stream advertises 1.
    uint64_t lanes;

    /// Open a lane and return an opaque handle. `index` is the tool's own lane number and
    /// must be less than `lanes`; the application may use it to select a lane or ignore it,
    /// so the same index does not necessarily name the same lane twice. Call it on the
    /// thread that will use the lane: a lane may carry thread-affine state.
    void* (*open_lane)(uint64_t index);
    void (*close_lane)(void* lane);

    /// One kernel dispatch on `lane`, enqueued and not waited on.
    void (*launch_kernel)(void* lane);

    /// One host-to-device copy on `lane`, enqueued and not waited on, large enough and from
    /// pinned host memory to route to the copy engine rather than a blit kernel. NULL when
    /// the application has no copy buffer.
    void (*launch_copy)(void* lane);

    /// Block until everything previously enqueued on `lane` has completed.
    void (*wait)(void* lane);
} sync_test_workload_t;

/// Implemented by every tool here, resolved by the application through dlsym. Returns the
/// number of failed expectations.
typedef uint64_t (*sync_test_run_fn)(const sync_test_workload_t*);

#define SYNC_TEST_RUN_SYMBOL "sync_test_run"

#ifdef __cplusplus
}
#endif
