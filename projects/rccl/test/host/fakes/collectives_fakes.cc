/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Data and string helpers defined by src/collectives.cc. Pure data with no
// behaviour to drive, so no seam and no reset entry point.
//
// Distinct from collective_stubs.cc, which is a fail-loud floor for the
// collective LAUNCH pipeline (ncclLaunchKernel and friends) and therefore
// cannot link into a target whose unit under test defines those.
//
// Real names rather than placeholders: production logs through these helpers,
// and tests asserting log content must see the same text production emits.
// Keep these switches faithful to src/collectives.cc.

#include "device.h"
#include "info.h"
#include "nccl.h"

// Algorithm/protocol name tables. Passed to the RCCL tuning override hooks and
// logged through.
const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS] = {
    "Tree", "Ring", "CollNetDirect", "CollNetChain", "NVLS", "NVLSTree", "PAT"};
const char* ncclProtoStr[NCCL_NUM_PROTOCOLS] = {"LL", "LL128", "Simple"};

const char* ncclFuncToString(ncclFunc_t op) {
  switch (op) {
    case ncclFuncAllGather: return "AllGather";
    case ncclFuncAllReduce: return "AllReduce";
    case ncclFuncAlltoAll: return "AlltoAll";
    case ncclFuncAlltoAllv: return "AlltoAllv";
    case ncclFuncBroadcast: return "Broadcast";
    case ncclFuncGather: return "Gather";
    case ncclFuncRecv: return "Recv";
    case ncclFuncReduce: return "Reduce";
    case ncclFuncReduceScatter: return "ReduceScatter";
    case ncclFuncScatter: return "Scatter";
    case ncclFuncSendRecv: return "SendRecv";
    case ncclFuncSend: return "Send";
    case ncclFuncPutSignal: return "PutSignal";
    case ncclFuncSignal: return "Signal";
    case ncclFuncWaitSignal: return "WaitSignal";
    default: return "Invalid";
  }
}

const char* ncclAlgoToString(int algo) {
  switch (algo) {
    case NCCL_ALGO_TREE: return "TREE";
    case NCCL_ALGO_RING: return "RING";
    case NCCL_ALGO_COLLNET_DIRECT: return "COLLNET_DIRECT";
    case NCCL_ALGO_COLLNET_CHAIN: return "COLLNET_CHAIN";
    case NCCL_ALGO_NVLS: return "NVLS";
    case NCCL_ALGO_NVLS_TREE: return "NVLS_TREE";
    case NCCL_ALGO_PAT: return "PAT";
    default: return "Unknown";
  }
}
// If NCCL_NUM_ALGORITHMS grows, this switch needs the new name -- otherwise a
// real algorithm silently logs as "Unknown", which is what happened with PAT.
static_assert(NCCL_NUM_ALGORITHMS == 7,
              "ncclAlgoToString above must name every algorithm; add the new case");

const char* ncclProtoToString(int proto) {
  switch (proto) {
    case NCCL_PROTO_LL: return "LL";
    case NCCL_PROTO_LL128: return "LL128";
    case NCCL_PROTO_SIMPLE: return "SIMPLE";
    default: return "Unknown";
  }
}

// Same failure mode as ncclAlgoToString: a new enumerator silently logs as the
// default arm. Protocols get the same guard.
static_assert(NCCL_NUM_PROTOCOLS == 3,
              "ncclProtoToString above must name every protocol; add the new case");
const char* ncclDatatypeToString(ncclDataType_t type) {
  switch (type) {
    case ncclInt8: return "ncclInt8";
    case ncclInt32: return "ncclInt32";
    case ncclUint32: return "ncclUint32";
    case ncclInt64: return "ncclInt64";
    case ncclUint64: return "ncclUint64";
    case ncclFloat16: return "ncclFloat16";
    case ncclFloat32: return "ncclFloat32";
    case ncclFloat64: return "ncclFloat64";
    case ncclBfloat16: return "ncclBfloat16";
    case ncclFloat8e4m3: return "ncclFloat8e4m3";
    case ncclFloat8e5m2: return "ncclFloat8e5m2";
    default: return "Unknown";
  }
}

// Red-op text is not consumed by this target; retain the existing placeholder.
const char* ncclDevRedOpToString(ncclDevRedOp_t) { return "redop"; }
