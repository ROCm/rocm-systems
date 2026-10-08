/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Data symbols alloc.h declares extern. Production defines them in src/init.cc;
// a binary that inlines the alloc.h cuMem helpers without compiling init.cc
// links them from here.

#include "alloc.h"

// Per-device counters the cuMem alloc/free helpers bump. Zero-initialised.
struct allocationTracker allocTracker[MAX_ALLOC_TRACK_NGPU] = {};
