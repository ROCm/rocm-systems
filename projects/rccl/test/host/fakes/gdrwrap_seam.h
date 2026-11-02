/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Installs controllable trampolines over gdrwrap.h's file-static helpers.
// Include immediately before a unit under test and pair with
// gdrwrap_seam_undef.h immediately afterwards. There is intentionally no
// include guard: this is a positional rename pair.

#include "fakes/gdrwrap_fakes.h"

#define ncclGdrCudaMapPointer micro_ncclGdrCudaMapPointer
#define ncclGdrCudaUnmapPointer micro_ncclGdrCudaUnmapPointer
#define wrap_gdr_copy_to_mapping micro_wrapGdrCopyToMapping
#define wrap_gdr_copy_from_mapping micro_wrapGdrCopyFromMapping
