/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// alloc.h data symbols that src/init.cc defines in production.

#include "alloc.h"

struct allocationTracker allocTracker[MAX_ALLOC_TRACK_NGPU] = {};
