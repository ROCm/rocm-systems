// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// DEC-13: thread_context — opaque incomplete type in the public header.
// NFR-PORT-3: public headers MUST NOT include <ucontext.h>, <signal.h>,
//             <pthread.h>, or <linux/perf_event.h>.
//
// The complete definition lives in src/linux/platform_thread_context.hpp.
// Public callers traffic in pointers/references to thread_context only.
// Only trigger and unwinder implementations dereference the context.

namespace rocprofsys::sampling
{
struct thread_context;
}  // namespace rocprofsys::sampling
