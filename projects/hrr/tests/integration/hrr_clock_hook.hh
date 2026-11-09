/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

// The test binary defines clock_gettime() itself, in hrr_clock_hook.cc. The
// dynamic linker binds the HIP runtime's calls to it before libc's. The capture
// writer calls it to timestamp each record, before it takes the buffer lock.
//
// A thread that sets t_hrr_clock_hook gets it called on each clock_gettime()
// that thread makes, before the clock is read. Other threads are not affected.
// POSIX only.
extern thread_local void (*t_hrr_clock_hook)();
