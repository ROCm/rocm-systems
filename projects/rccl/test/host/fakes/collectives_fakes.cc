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
// Real names rather than placeholders for the two tables and the three switches
// below: production logs through those, and a test asserting on a log line should
// see what production would print. ncclDatatypeToString and ncclDevRedOpToString
// are the deliberate exceptions -- no test asserts on a datatype or redop in a
// log line, so they return a fixed token rather than a transcribed table.

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
    case ncclFuncBroadcast: return "Broadcast";
    case ncclFuncReduce: return "Reduce";
    case ncclFuncAllGather: return "AllGather";
    case ncclFuncReduceScatter: return "ReduceScatter";
    case ncclFuncAllReduce: return "AllReduce";
    case ncclFuncSendRecv: return "SendRecv";
    case ncclFuncSend: return "Send";
    case ncclFuncRecv: return "Recv";
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
// ncclFuncToString gets NO count guard on purpose. ncclNumFuncs is 19
// (nccl_common.h:95) and includes the alltoallv/scatter/gather family; the switch
// names the 8 that enqueue.cc logs and lets the rest fall to "Invalid", which
// mirrors what production's own table does. A guard here would fire on an
// enumerator this fake never needed to name.

// Deliberate placeholders; see the exception noted in the file header.
const char* ncclDatatypeToString(ncclDataType_t) { return "dtype"; }
const char* ncclDevRedOpToString(ncclDevRedOp_t) { return "redop"; }
