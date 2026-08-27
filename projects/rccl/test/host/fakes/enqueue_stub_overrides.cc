/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Replacements for the shared nccl_stubs.cc entries that the enqueue microtest
// omits. Colocating them with the omission keeps the mapping reviewable:
//
//   omitted from nccl_stubs.cc            | supplied instead by
//   --------------------------------------+---------------------------------
//   ncclInitKernelsForDevice              | src/enqueue.cc itself (the UUT)
//   ncclParamGraphStreamOrdering          | src/enqueue.cc itself (NCCL_PARAM)
//   rcclUseAinic                          | THIS FILE (needs a real value)
//
// The first two need no definition here -- the unit under test provides them,
// and the omit macros exist only to avoid a duplicate symbol. Only the third
// requires a replacement: rcclEffectiveP2pBatchEnable's gfx950 arm reads it, and
// the shared stub is fail-loud.

#include "enqueue_fakes.h"

bool g_rcclUseAinic = false;
bool rcclUseAinic() { return g_rcclUseAinic; }
