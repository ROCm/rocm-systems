/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Controllable seams for sym_kernels.cc's kernel-index/table symbols, for binaries that fake rather than
// compile the real sym_kernels_host.cc (which defines these for real and would collide at link time).

#ifndef RCCL_TEST_HOST_FAKES_SYM_KERNELS_INDEX_FAKES_H_
#define RCCL_TEST_HOST_FAKES_SYM_KERNELS_INDEX_FAKES_H_

#include <cstdint>
#include <functional>

#include "nccl.h"
#include "sym_kernels.h"  // ncclSymkKernelId

struct ncclComm;
struct ncclTaskColl;
struct ncclSymkDevWork;

// sym_kernels.cc's ncclSymkLLKernelMask: default has no bits set (the LL-kernel-init-once check never fires).
extern std::function<uint64_t()> g_symkLLKernelMask;  // ncclSymkKernelMask (NCCL 2.32.6: uint64_t)

// sym_kernels.cc's ncclSymkDynamicSmemKernelMask: default has no bits set (kernelDynSmem stays 0).
extern std::function<uint64_t()> g_symkDynamicSmemKernelMask;  // ncclSymkKernelMask (NCCL 2.32.6: uint64_t)

// sym_kernels.cc's ncclSymkTmaKernelMask: default has no bits set, so the scheduler's tile-staging
// LDS check never fires.
extern std::function<uint64_t()> g_symkTmaKernelMask;  // ncclSymkKernelMask (NCCL 2.32.6: uint64_t)

// sym_kernels.cc's ncclSymkGetKernelIndex: default always selects index 0.
extern std::function<int(ncclSymkKernelId, int, ncclDataType_t)> g_symkGetKernelIndex;

// sym_kernels.cc's ncclSymkKernelIdToString: default returns a fixed, recognizable name.
extern std::function<const char*(int)> g_symkKernelIdToString;

// sym_kernels.cc's ncclSymkMakeDevWork: trivial success; does not populate *outDevWork.
extern std::function<ncclResult_t(struct ncclComm*, struct ncclTaskColl*, struct ncclSymkDevWork*)>
    g_symkMakeDevWork;

// NCCL 2.32.6 Genk seams: mask 0 and index -1 (AMD builds no Genk kernels); init start/end succeed.
extern std::function<uint64_t()> g_genkKernelMask;
extern std::function<int(ncclSymkKernelId, int, ncclDataType_t)> g_genkGetKernelIndex;
extern std::function<ncclResult_t(struct ncclComm*, bool*)> g_genkInitStart;
extern std::function<ncclResult_t(struct ncclComm*)> g_genkInitEnd;

void ResetSymKernelsIndexFakes();

#endif  // RCCL_TEST_HOST_FAKES_SYM_KERNELS_INDEX_FAKES_H_
