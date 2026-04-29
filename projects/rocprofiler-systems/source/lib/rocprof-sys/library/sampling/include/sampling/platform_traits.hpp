// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// NFR-PORT-3: public headers MUST NOT include <ucontext.h>, <signal.h>,
//             <pthread.h>, or <linux/perf_event.h>.
//
// The unwinder concept exposes its `unwind(void const* ctx)` parameter as
// `void const*` to keep this header platform-agnostic; production wires a
// `ucontext_t const*` through the same pointer.

namespace rocprofsys::sampling
{

// Opaque handle for sigset_t passed across the signal_dispatcher policy seam.
// Public concept headers must not include <signal.h> (NFR-PORT-1); production
// implementations cast to sigset_t*/sigset_t const*.
using signal_set_handle         = void const*;
using signal_set_mutable_handle = void*;
}  // namespace rocprofsys::sampling
