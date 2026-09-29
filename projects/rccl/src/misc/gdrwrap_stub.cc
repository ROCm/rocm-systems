/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include "gdrwrap.h"

// AMD stub implementations for GDR copy functions.
// RCCL on ROCm does not build misc/gdrwrap.cc (see gdrwrap.h); Host RMA
// transport (rma_socket*.cc, new in NCCL 2.32) references these symbols.
// Return ncclInternalError so any unexpected call path fails gracefully.

ncclResult_t wrap_gdr_copy_from_mapping(gdr_mh_t handle, void* h_ptr, const void* map_d_ptr, size_t size) {
  return ncclInternalError;
}

ncclResult_t wrap_gdr_copy_to_mapping(gdr_mh_t handle, void* map_d_ptr, const void* h_ptr, size_t size) {
  return ncclInternalError;
}
