/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Reusable stubs for the RCCL device-runtime layer (devr / gin / rma / team /
// shadow-pool / space-allocator / symmetric-kernel symbols). See devr_fakes.h
// for the philosophy. Real headers are included so every signature is checked
// against the production declaration rather than hand-transcribed.

#include <cstdlib>

#include "comm.h"
#include "sym_kernels.h"
#include "allocator.h"
#include "utils.h"
#include "dev_runtime_internal.h"
#include "gin/gin_host.h"
#include "rma/rma_proxy.h"
#include "nccl_device/core_tmp.h"
#include "nccl_device/lsa_barrier.h"
#include "nccl_device/gin_barrier.h"

#include "devr_fakes.h"

// ---------------------------------------------------------------------------
// devcomm compat tables (defined in devcomm/devcomm_v*.cc in the real build).
// ---------------------------------------------------------------------------
struct ncclDevCommCompat ncclDevCommCompat_v22902 = {};
struct ncclDevCommCompat ncclDevCommCompat_v22907 = {};
struct ncclDevCommCompat ncclDevCommCompat_v23000 = {};

// ---------------------------------------------------------------------------
// Symmetric kernels.
// ---------------------------------------------------------------------------
ncclResult_t ncclSymkInitOnce(struct ncclComm*) { return ncclSuccess; }

// ---------------------------------------------------------------------------
// Space allocator.
// ---------------------------------------------------------------------------
void         ncclSpaceConstruct(struct ncclSpace*) {}
void         ncclSpaceDestruct(struct ncclSpace*) {}
ncclResult_t ncclSpaceAlloc(struct ncclSpace*, int64_t, int64_t, int, int64_t* outOffset) {
  if (outOffset) *outOffset = 0;
  return ncclSuccess;
}
ncclResult_t ncclSpaceFree(struct ncclSpace*, int64_t, int64_t) { return ncclSuccess; }

// ---------------------------------------------------------------------------
// Shadow pool.
// ---------------------------------------------------------------------------
void         ncclShadowPoolConstruct(struct ncclShadowPool*) {}
ncclResult_t ncclShadowPoolDestruct(struct ncclShadowPool*, hipStream_t) { return ncclSuccess; }
ncclResult_t ncclShadowPoolAlloc(struct ncclShadowPool*, size_t, void** outDevObj, void** outHostObj, hipStream_t) {
  if (outDevObj) *outDevObj = nullptr;
  if (outHostObj) *outHostObj = nullptr;
  return ncclSuccess;
}
ncclResult_t ncclShadowPoolFree(struct ncclShadowPool*, void*, hipStream_t) { return ncclSuccess; }
ncclResult_t ncclShadowPoolToHost(struct ncclShadowPool*, void*, void** outHostObj) {
  if (outHostObj) *outHostObj = nullptr;
  return ncclSuccess;
}

// ---------------------------------------------------------------------------
// Intrusive address map.
// ---------------------------------------------------------------------------
ncclResult_t ncclIntruAddressMapInsert_untyped(struct ncclIntruAddressMap_untyped*, int, int, int, uintptr_t, void*) {
  return ncclSuccess;
}
ncclResult_t ncclIntruAddressMapFind_untyped(struct ncclIntruAddressMap_untyped*, int, int, int, uintptr_t, void** out) {
  if (out) *out = nullptr;
  return ncclSuccess;
}
ncclResult_t ncclIntruAddressMapRemove_untyped(struct ncclIntruAddressMap_untyped*, int, int, int, uintptr_t) {
  return ncclSuccess;
}

// ---------------------------------------------------------------------------
// Memory stack spill (must return real memory to avoid a crash if ever hit).
// ---------------------------------------------------------------------------
void* ncclMemoryStack::allocateSpilled(struct ncclMemoryStack*, size_t size, size_t align) {
  void* p = nullptr;
  if (align < sizeof(void*)) align = sizeof(void*);
  if (posix_memalign(&p, align, size) != 0) return nullptr;
  return p;  // intentionally leaked; process is short-lived
}

// ---------------------------------------------------------------------------
// GIN host.
// ---------------------------------------------------------------------------
ncclResult_t ncclGetGinType(struct ncclComm*, ncclGinType_t* ginType) {
  if (ginType) *ginType = NCCL_GIN_TYPE_NONE;
  return ncclSuccess;
}
ncclResult_t ncclGetRailedGinType(struct ncclComm*, ncclGinType_t* ginType) {
  if (ginType) *ginType = NCCL_GIN_TYPE_NONE;
  return ncclSuccess;
}
ncclResult_t ncclGinConnectOnce(struct ncclComm*) { return ncclSuccess; }
ncclResult_t ncclGinDevCommSetup(struct ncclComm*, struct ncclDevCommRequirements const*, struct ncclDevComm*) {
  return ncclSuccess;
}
ncclResult_t ncclGinDevCommFree(struct ncclComm*, struct ncclDevComm const*) { return ncclSuccess; }
ncclResult_t ncclGinRegister(struct ncclComm*, void*, size_t, void*[NCCL_GIN_MAX_CONNECTIONS],
                             ncclGinWindow_t[NCCL_GIN_MAX_CONNECTIONS], int, bool, int) {
  return ncclSuccess;
}
ncclResult_t ncclGinDeregister(struct ncclComm*, void*[NCCL_GIN_MAX_CONNECTIONS]) { return ncclSuccess; }

// ---------------------------------------------------------------------------
// RMA proxy.
// ---------------------------------------------------------------------------
ncclResult_t ncclRmaProxyConnectOnce(struct ncclComm*) { return ncclSuccess; }
ncclResult_t ncclRmaProxyRegister(struct ncclComm*, void*, size_t, void*[NCCL_GIN_MAX_CONNECTIONS]) {
  return ncclSuccess;
}
ncclResult_t ncclRmaProxyDeregister(struct ncclComm*, void*[NCCL_GIN_MAX_CONNECTIONS]) { return ncclSuccess; }

// ---------------------------------------------------------------------------
// devr internal helpers (defined elsewhere in the real build).
// ---------------------------------------------------------------------------
ncclResult_t ncclDevrPopulateSegmentSizes(struct ncclDevrMemory*, int) { return ncclSuccess; }
ncclResult_t ncclDevrAllocAndPopulateSegmentWindows(struct ncclDevrState*, struct ncclDevrMemory*, hipStream_t,
                                                    struct ncclSegmentWindow** out) {
  if (out) *out = nullptr;
  return ncclSuccess;
}

// ---------------------------------------------------------------------------
// Team accessors (host variants).
// ---------------------------------------------------------------------------
extern "C" ncclTeam_t ncclTeamWorld(ncclComm_t) { return ncclTeam_t{}; }
extern "C" ncclTeam_t ncclTeamLsa(ncclComm_t) { return ncclTeam_t{}; }
extern "C" ncclTeam_t ncclTeamRail(ncclComm_t) { return ncclTeam_t{}; }

// ---------------------------------------------------------------------------
// Barrier requirement builders (host variants).
// ---------------------------------------------------------------------------
extern "C" ncclResult_t ncclLsaBarrierCreateRequirement(ncclTeam_t, int, ncclLsaBarrierHandle_t*,
                                                        ncclDevResourceRequirements_t*) {
  return ncclSuccess;
}
extern "C" ncclResult_t ncclGinBarrierCreateRequirement(ncclComm_t, ncclTeam_t, int, ncclGinBarrierHandle_t*,
                                                        ncclDevResourceRequirements_t*) {
  return ncclSuccess;
}
