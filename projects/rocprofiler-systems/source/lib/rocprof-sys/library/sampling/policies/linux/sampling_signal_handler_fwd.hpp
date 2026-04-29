// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Single forward declaration of the production sampling signal handler.
// Defined exactly once in services_accessor.cpp; declared here so policy headers
// (real_overflow_trigger, real_timer_trigger) and the default-policies aggregator
// can reference it via sigaction()/F_SETSIG without each carrying its own
// independent extern "C" declaration. Linux-only by construction (siginfo_t).

#include <csignal>

extern "C" void
rocprofsys_sampling_signal_handler(int, siginfo_t*, void*);
