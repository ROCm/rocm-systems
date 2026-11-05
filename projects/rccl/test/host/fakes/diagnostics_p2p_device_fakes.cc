/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "diagnostics_p2p_device_fakes.h"

#include "signature-drift.h"

ASSERT_HOOK_MATCHES_PROD(g_ncclDiagP2pInitSlots, ncclDiagP2pInitSlots);
ASSERT_HOOK_MATCHES_PROD(g_ncclDiagP2pRemoteWrite, ncclDiagP2pRemoteWrite);
ASSERT_HOOK_MATCHES_PROD(g_ncclDiagP2pVerifyWrites, ncclDiagP2pVerifyWrites);
ASSERT_HOOK_MATCHES_PROD(g_ncclDiagP2pRemoteRead, ncclDiagP2pRemoteRead);

ncclResult_t EmulateDiagP2pInitSlots(struct ncclDiagP2pSlot* slots, const int* slotRanks, int slotCount, int dstRank,
                                     hipStream_t) {
  for (int i = 0; i < slotCount; i++) {
    slots[i].writeValue = 0;
    slots[i].verifyValue = 0;
    slots[i].readPattern = ncclDiagP2pReadPattern(dstRank, slotRanks[i]);
  }
  return ncclSuccess;
}

ncclResult_t EmulateDiagP2pRemoteWrite(const struct ncclDiagP2pRemoteOp* ops, int opCount, hipStream_t) {
  for (int i = 0; i < opCount; i++) {
    ops[i].remoteSlots[ops[i].srcSlot].writeValue = ncclDiagP2pWritePattern(ops[i].srcRank, ops[i].dstRank);
  }
  return ncclSuccess;
}

ncclResult_t EmulateDiagP2pVerifyWrites(struct ncclDiagP2pSlot* slots, int slotCount, hipStream_t) {
  for (int i = 0; i < slotCount; i++) {
    slots[i].verifyValue = slots[i].writeValue;
  }
  return ncclSuccess;
}

ncclResult_t EmulateDiagP2pRemoteRead(const struct ncclDiagP2pRemoteOp* ops, int opCount, uint64_t* readback,
                                      hipStream_t) {
  for (int i = 0; i < opCount; i++) {
    readback[i] = ops[i].remoteSlots[ops[i].srcSlot].readPattern;
  }
  return ncclSuccess;
}

std::function<ncclResult_t(struct ncclDiagP2pSlot*, const int*, int, int, hipStream_t)> g_ncclDiagP2pInitSlots =
    EmulateDiagP2pInitSlots;
std::function<ncclResult_t(const struct ncclDiagP2pRemoteOp*, int, hipStream_t)> g_ncclDiagP2pRemoteWrite =
    EmulateDiagP2pRemoteWrite;
std::function<ncclResult_t(struct ncclDiagP2pSlot*, int, hipStream_t)> g_ncclDiagP2pVerifyWrites =
    EmulateDiagP2pVerifyWrites;
std::function<ncclResult_t(const struct ncclDiagP2pRemoteOp*, int, uint64_t*, hipStream_t)>
    g_ncclDiagP2pRemoteRead = EmulateDiagP2pRemoteRead;

ncclResult_t ncclDiagP2pInitSlots(struct ncclDiagP2pSlot* slots, const int* slotRanks, int slotCount, int dstRank,
                                  hipStream_t stream) {
  return g_ncclDiagP2pInitSlots(slots, slotRanks, slotCount, dstRank, stream);
}

ncclResult_t ncclDiagP2pRemoteWrite(const struct ncclDiagP2pRemoteOp* ops, int opCount, hipStream_t stream) {
  return g_ncclDiagP2pRemoteWrite(ops, opCount, stream);
}

ncclResult_t ncclDiagP2pVerifyWrites(struct ncclDiagP2pSlot* slots, int slotCount, hipStream_t stream) {
  return g_ncclDiagP2pVerifyWrites(slots, slotCount, stream);
}

ncclResult_t ncclDiagP2pRemoteRead(const struct ncclDiagP2pRemoteOp* ops, int opCount, uint64_t* readback,
                                   hipStream_t stream) {
  return g_ncclDiagP2pRemoteRead(ops, opCount, readback, stream);
}

void ResetDiagnosticsP2pDeviceFakes() {
  g_ncclDiagP2pInitSlots = EmulateDiagP2pInitSlots;
  g_ncclDiagP2pRemoteWrite = EmulateDiagP2pRemoteWrite;
  g_ncclDiagP2pVerifyWrites = EmulateDiagP2pVerifyWrites;
  g_ncclDiagP2pRemoteRead = EmulateDiagP2pRemoteRead;
}
