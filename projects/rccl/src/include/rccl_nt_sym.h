/*************************************************************************
 * Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 * Modifications Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_NT_SYM_H_
#define RCCL_NT_SYM_H_

// [RCCL] Device-symbol suffixing for the alternate gfx950 512-thread kernel set.
// When the device sources are compiled with -DRCCL_NTHREADS_512, every device
// symbol that must be unique across the two coexisting kernel sets is renamed
// with a "_512" suffix via RCCL_NT_SYM(). This lets both the default 256-thread
// set and the 512-thread set live in the same device ELF without collisions.
// When RCCL_NTHREADS_512 is not defined, RCCL_NT_SYM(x) == x (no change).
//
// Kept in its own header (included from device.h) to minimize merge conflicts.
#define RCCL_CAT_(a, b) a##b
#define RCCL_CAT(a, b) RCCL_CAT_(a, b)
#if defined(RCCL_NTHREADS_512)
#define RCCL_NT_SUFFIX _512
// Redirect the shared-memory objects too: their layout depends on NCCL_MAX_GROUPS,
// so the 512 set needs its own LDS allocation. Renaming the identifier here means
// every reference in the (unmodified) device sources resolves to the _512 object.
#define ncclShmem ncclShmem_512
#define ncclShmemPerWarp ncclShmemPerWarp_512
#else
#define RCCL_NT_SUFFIX
#endif
#define RCCL_NT_SYM(name) RCCL_CAT(name, RCCL_NT_SUFFIX)

#endif // RCCL_NT_SYM_H_
