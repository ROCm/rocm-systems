/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 ************************************************************************/

#include "cpu_func_decode.h"

#include "device.h"

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

static const char* kCollNames[] = {
    "Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce", "SendRecv",
    "", "", "", "", "", "AlltoAllPivot", "AlltoAllGda", "AlltoAllvGda"};
static const char* kAlgoNames[] = {"TREE", "RING", "", "", "", "", "PAT"};
static const char* kProtoNames[] = {"LL", "LL128", "SIMPLE"};
static const char* kRedOpNames[] = {"Sum", "Prod", "MinMax", "PreMulSum", "SumPostDiv"};
static const char* kTypeNames[] = {
    "i8", "u8", "i32", "u32", "i64", "u64", "f16", "f32", "f64", "bf16", "f8e4m3", "f8e5m2"};

static const ncclDataType_t kTypes[] = {
    ncclInt8, ncclUint8, ncclInt32, ncclUint32, ncclInt64, ncclUint64,
    ncclFloat16, ncclFloat32, ncclFloat64, ncclBfloat16, ncclFloat8e4m3, ncclFloat8e5m2};

static const ncclDevRedOp_t kRedOps[] = {
    ncclDevSum, ncclDevProd, ncclDevMinMax, ncclDevPreMulSum, ncclDevSumPostDiv};

static std::mutex g_decodeMutex;
static std::vector<rcclCpuFuncDesc> g_idToDesc;
static bool g_initialized = false;

static int nameIndex(const char* name, const char* const* table, int n) {
  for (int i = 0; i < n; i++) {
    if (table[i][0] != '\0' && name == std::string(table[i])) return i;
  }
  return -1;
}

static void decodeKey(uint64_t key, rcclCpuFuncDesc* d) {
  d->valid = true;
  int collIdx = (key >> RCCL_COLL_SHIFT) & RCCL_FUNC_ID_MASK;
  int algoIdx = (key >> RCCL_ALGO_SHIFT) & RCCL_FUNC_ID_MASK;
  int protoIdx = (key >> RCCL_PROTO_SHIFT) & RCCL_FUNC_ID_MASK;
  int redIdx = (key >> RCCL_REDOP_SHIFT) & RCCL_FUNC_ID_MASK;
  int tyIdx = (key >> RCCL_DTYPE_SHIFT) & RCCL_FUNC_ID_MASK;
  int accIdx = (key >> RCCL_ACC_SHIFT) & RCCL_FUNC_ID_MASK;
  int pipeIdx = (key >> RCCL_PIPELINE_SHIFT) & RCCL_FUNC_ID_MASK;

  if (collIdx >= 0 && collIdx < static_cast<int>(sizeof(kCollNames) / sizeof(kCollNames[0]))) {
    const char* cn = kCollNames[collIdx];
    if (cn[0] == '\0') { d->valid = false; return; }
    if (std::string(cn) == "Broadcast") d->coll = ncclFuncBroadcast;
    else if (cn == std::string("Reduce")) d->coll = ncclFuncReduce;
    else if (cn == std::string("AllGather")) d->coll = ncclFuncAllGather;
    else if (cn == std::string("ReduceScatter")) d->coll = ncclFuncReduceScatter;
    else if (cn == std::string("AllReduce")) d->coll = ncclFuncAllReduce;
    else if (cn == std::string("SendRecv")) d->coll = ncclFuncSendRecv;
    else if (cn == std::string("AlltoAllPivot")) d->coll = ncclFuncAlltoAllPivot;
    else if (cn == std::string("AlltoAllGda")) d->coll = ncclFuncAlltoAllGda;
    else if (cn == std::string("AlltoAllvGda")) d->coll = ncclFuncAlltoAllvGda;
    else d->valid = false;
  }

  d->algo = (algoIdx == 0) ? NCCL_ALGO_TREE :
            (algoIdx == 1) ? NCCL_ALGO_RING :
            (algoIdx == 6) ? NCCL_ALGO_PAT : NCCL_ALGO_UNDEF;
  d->proto = (protoIdx == 0) ? NCCL_PROTO_LL :
             (protoIdx == 1) ? NCCL_PROTO_LL128 :
             (protoIdx == 2) ? NCCL_PROTO_SIMPLE : NCCL_PROTO_UNDEF;
  d->devRedOp = (redIdx < 5) ? kRedOps[redIdx] : ncclDevSum;
  d->datatype = (tyIdx < 12) ? kTypes[tyIdx] : ncclFloat32;
  d->acc = accIdx;
  d->pipeline = pipeIdx;

  if (d->coll == ncclFuncBroadcast) {
    d->algo = NCCL_ALGO_RING;
    d->devRedOp = ncclDevSum;
  } else if (d->coll == ncclFuncSendRecv || d->coll == ncclFuncAlltoAllPivot ||
             d->coll == ncclFuncAlltoAllGda || d->coll == ncclFuncAlltoAllvGda) {
    d->algo = NCCL_ALGO_RING;
  }
}

}  // namespace

void rcclCpuFuncDecodeInit() {
  std::lock_guard<std::mutex> lock(g_decodeMutex);
  if (g_initialized) return;

  int maxId = -1;
  for (const auto& kv : ncclDevFuncNameToId) {
    maxId = std::max(maxId, kv.second);
  }
  g_idToDesc.assign(static_cast<size_t>(maxId + 1), rcclCpuFuncDesc{});
  for (const auto& kv : ncclDevFuncNameToId) {
    if (kv.second < 0) continue;
    rcclCpuFuncDesc d{};
    d.valid = false;
    decodeKey(kv.first, &d);
    g_idToDesc[static_cast<size_t>(kv.second)] = d;
  }
  g_initialized = true;
}

bool rcclCpuDecodeFuncId(unsigned funcId, struct rcclCpuFuncDesc* out) {
  rcclCpuFuncDecodeInit();
  if (out == nullptr) return false;
  if (funcId >= g_idToDesc.size()) {
    out->valid = false;
    return false;
  }
  *out = g_idToDesc[funcId];
  return out->valid;
}
