/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// src/diagnostics/device/p2p.cu launchers; each defaults to its Emulate* twin.

#ifndef RCCL_TEST_HOST_FAKES_DIAGNOSTICS_P2P_DEVICE_FAKES_H_
#define RCCL_TEST_HOST_FAKES_DIAGNOSTICS_P2P_DEVICE_FAKES_H_

#include <cstdint>
#include <functional>

#include "diagnostics/p2p.h"

ncclResult_t EmulateDiagP2pInitSlots(struct ncclDiagP2pSlot* slots, const int* slotRanks, int slotCount, int dstRank,
                                     hipStream_t stream);
ncclResult_t EmulateDiagP2pRemoteWrite(const struct ncclDiagP2pRemoteOp* ops, int opCount, hipStream_t stream);
ncclResult_t EmulateDiagP2pVerifyWrites(struct ncclDiagP2pSlot* slots, int slotCount, hipStream_t stream);
ncclResult_t EmulateDiagP2pRemoteRead(const struct ncclDiagP2pRemoteOp* ops, int opCount, uint64_t* readback,
                                      hipStream_t stream);

extern std::function<ncclResult_t(struct ncclDiagP2pSlot*, const int*, int, int, hipStream_t)> g_ncclDiagP2pInitSlots;
extern std::function<ncclResult_t(const struct ncclDiagP2pRemoteOp*, int, hipStream_t)> g_ncclDiagP2pRemoteWrite;
extern std::function<ncclResult_t(struct ncclDiagP2pSlot*, int, hipStream_t)> g_ncclDiagP2pVerifyWrites;
extern std::function<ncclResult_t(const struct ncclDiagP2pRemoteOp*, int, uint64_t*, hipStream_t)>
    g_ncclDiagP2pRemoteRead;

void ResetDiagnosticsP2pDeviceFakes();

#endif  // RCCL_TEST_HOST_FAKES_DIAGNOSTICS_P2P_DEVICE_FAKES_H_
