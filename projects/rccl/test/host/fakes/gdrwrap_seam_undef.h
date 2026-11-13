/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Removes the positional rename seam installed by gdrwrap_seam.h.

#undef wrap_gdr_copy_from_mapping
#undef wrap_gdr_copy_to_mapping
#undef ncclGdrCudaUnmapPointer
#undef ncclGdrCudaMapPointer
