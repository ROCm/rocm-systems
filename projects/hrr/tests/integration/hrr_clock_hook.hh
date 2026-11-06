/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

// The test binary defines clock_gettime() in hrr_clock_hook.cc; the dynamic linker binds the HIP
// runtime's calls to it before libc's. The capture writer calls it to timestamp each record, before it takes the buffer lock.
// A thread that sets t_hrr_clock_hook has it called before each clock_gettime() that thread makes.
// Other threads are not affected. POSIX only.
extern thread_local void (*t_hrr_clock_hook)();
