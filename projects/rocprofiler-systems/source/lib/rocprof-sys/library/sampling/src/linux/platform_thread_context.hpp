// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Complete definition of thread_context for Linux.
// Only included by src/linux/ files (trigger, unwinder implementations).
// NFR-PORT-3: this header MUST NOT appear under include/sampling/.

#include <ucontext.h>

namespace rocprofsys::sampling
{

struct thread_context
{
    ucontext_t value;
};

}  // namespace rocprofsys::sampling
