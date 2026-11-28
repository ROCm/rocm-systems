/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Scenes and helpers shared by more than one enqueue-test.d/ file. Included into
// enqueue-test.cc's translation unit before every enqueue-test.d/*.inc; see README.md.

namespace {
constexpr ssize_t kBatch = sizeof(struct ncclDevWorkBatch);
ncclKernelPlanBudget MakeBudget(ssize_t in, ssize_t out) {
  ncclKernelPlanBudget b{};
  b.inArgsBytes = in;
  b.outArgsBytes = out;
  return b;
}
}  // namespace

namespace {
// LOAD-BEARING, twice over:
//  1. The protocol loop at :2534 reads `comm->topo->type` for the LL128 XGMI
//     gate, so comm->topo may not be null -- a zero-initialised ncclComm
//     segfaults there rather than failing an assertion.
//  2. ncclTopoSystem is far too large to hold by value in a test fixture: a
//     stack instance overflows and crashes inside the CONSTRUCTOR, before any
//     production code runs. Both it and ncclComm are heap-allocated here.
struct CostComm {
  std::unique_ptr<ncclTopoSystem> topoStorage{new ncclTopoSystem{}};
  std::unique_ptr<ncclComm> commStorage{new ncclComm{}};
  CostComm(int nRanks = 8, int nNodes = 1) {
    ncclComm& comm = *commStorage;
    comm.nRanks = nRanks;
    comm.nNodes = nNodes;
    comm.maxLocalRanks = 8;
    comm.localRanks = 8;
    // XGMI_ALL set: the LL128 gate at :2535 passes, so LL128 cells are filled
    // like every other protocol. Tests that care about the gate clear it.
    topoStorage->type = RCCL_TOPO_XGMI_ALL;
    comm.topo = topoStorage.get();
  }
  ncclComm* get() { return commStorage.get(); }
  ncclTopoSystem& topo() { return *topoStorage; }
};

// A table wrapper that starts fully IGNOREd, like production's initCollCostTable.
struct CostTable {
  float t[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  CostTable() { initCollCostTable((float**)t); }
  float** ptr() { return (float**)t; }
  bool written(int a, int p) const { return t[a][p] != NCCL_ALGO_PROTO_IGNORE; }
  int countWritten() const {
    int n = 0;
    for (int a = 0; a < NCCL_NUM_ALGORITHMS; ++a) {
      for (int p = 0; p < NCCL_NUM_PROTOCOLS; ++p) {
        if (t[a][p] != NCCL_ALGO_PROTO_IGNORE) {
          ++n;
        }
      }
    }
    return n;
  }
};

// A general row's mask bit, for the per-call algSelection cases.
constexpr rcclTuningMask GeneralBit(int a, int p) {
  return rcclTuningMask::bit(a * NCCL_NUM_PROTOCOLS + p);
}

ncclTaskColl CostTask(ncclFunc_t f, ncclDataType_t dt = ncclFloat32,
                      ncclDevRedOp_t devOp = ncclDevSum) {
  ncclTaskColl t{};
  t.func = f;
  t.datatype = dt;
  t.opDev.op = devOp;
  // Match the nccl.h.in default (1), so a case relying on 0 has to say so.
  t.forceAlgSelection = 1;
  return t;
}

// Scripts every (algo, proto) pair to a fixed cost, so "was this cell reached?"
// is answerable.
void ScriptAllTimes(float cost) {
  g_topoGetAlgoTime = [cost](struct ncclComm*, int, int, int, size_t, int, float* time) {
    if (time) *time = cost;
    return ncclSuccess;
  };
}
}  // namespace

namespace {
// One place that stamps a topology's single GPU node with an arch string. Every
// fixture that needs one sets the same pair of fields, and a copy that forgets the
// count leaves the gcn string unreachable rather than failing.
void SetSingleGpuArch(ncclTopoSystem* topo, const char* gcn) {
  topo->nodes[GPU].count = 1;
  auto& gpu = topo->nodes[GPU].nodes[0].gpu;
  static_assert(sizeof(gpu.gcn) == GCN_ARCH_NAME_LEN);
  std::memset(gpu.gcn, 0, sizeof(gpu.gcn));
  size_t const length = std::strlen(gcn);
  EXPECT_LT(length, sizeof(gpu.gcn)) << "topology gcn must be normalized before storage";
  if (length < sizeof(gpu.gcn)) std::memcpy(gpu.gcn, gcn, length + 1);
}
}  // namespace

namespace {
// topoGetAlgoInfo needs slightly more comm state than updateCollCostTable: it
// reads archName for the arch-specific overrides and the maxThreads table when
// it converts an algorithm choice into a warp count.
struct AlgoInfoComm {
  // Holds a CostComm for the comm/topo pair, the rank counts and the XGMI_ALL
  // stamp; adds only what topoGetAlgoInfo needs beyond updateCollCostTable.
  //
  // NOT parameterised, deliberately, even though topoGetAlgoInfo reads nRanks and
  // nNodes. CostComm hardcodes maxLocalRanks and localRanks to 8 regardless of
  // nRanks (:1387-1392), so CostComm(64, 8) would describe 64 ranks over 8 nodes
  // with 8 local ranks and one GPU node -- a comm that cannot exist. Exposing a
  // pair of parameters that produce an incoherent fixture is worse than the fixed
  // 8/1 here. Varying either one needs CostComm to derive the local counts first,
  // and that changes what the existing CostComm tests cover (e.g. :1582).
  CostComm base{8, 1};
  AlgoInfoComm() {
    ncclComm* comm = base.get();
    comm->WarpSize = 32;
    SetSingleGpuArch(&base.topo(), "gfx942");
    for (int a = 0; a < NCCL_NUM_ALGORITHMS; ++a) {
      for (int p = 0; p < NCCL_NUM_PROTOCOLS; ++p) {
        comm->maxThreads[a][p] = 256;
      }
    }
  }
  ncclComm* get() { return base.get(); }
};
}  // namespace

namespace {
// ncclGroupSimulateEnd estimatedTime: scripts one (algo, proto) cell of the cost model.
constexpr float kChosenModelUs = 25.0f;

void ScriptCellTime(int algo, int proto, float chosen, float other) {
  g_topoGetAlgoTime = [=](struct ncclComm*, int, int a, int p, size_t, int, float* time) {
    if (time) *time = (a == algo && p == proto) ? chosen : other;
    return ncclSuccess;
  };
}
}  // namespace

namespace {
// One param setter for every enqueue test: three GetImplicitOrder tests used to
// re-roll this same lambda with a different key, which risks the copies drifting
// on the `deft` fallback.
void SetParam(const char* key, int64_t v) {
  std::string k(key);
  g_loadParam = [k, v](const char* env, int64_t deft) {
    return std::strcmp(env, k.c_str()) == 0 ? v : deft;
  };
}
}  // namespace

namespace {
// ncclComm is far too large for the stack (see CostComm), and memScoped needs
// real backing memory for ncclMemoryStackAlloc.
struct BatchPlanComm {
  std::unique_ptr<ncclComm> comm{new ncclComm{}};
  std::unique_ptr<ncclKernelPlan> plan{new ncclKernelPlan{}};
  BatchPlanComm(int nNodes = 1, int cudaArch = 942) {
    comm->nNodes = nNodes;
    comm->cudaArch = cudaArch;
    ncclMemoryStackConstruct(&comm->memScoped);
    // memPermanent is load-bearing, not decorative: addProxyOpIfNeeded allocates
    // its queued copy from memPool_ncclProxyOp against &comm->memPermanent. A
    // value-initialised stack has bumper == end == 0, so allocate() falls through
    // to allocateSpilled and mallocs a 64 KiB hunk -- and because topFrame.hunk is
    // null it is never linked into the frame chain, so nothing can reclaim it.
    ncclMemoryStackConstruct(&comm->memPermanent);
  }
  ~BatchPlanComm() {
    ncclMemoryStackDestruct(&comm->memScoped);
    ncclMemoryStackDestruct(&comm->memPermanent);
  }
  ncclComm* c() { return comm.get(); }
  ncclKernelPlan* p() { return plan.get(); }
  ncclKernelPlanner::WipPlan::Channel* chan(int id = 0) {
    return &comm->planner.wipPlan.channels[id];
  }
  // Number of batch nodes currently queued on a channel.
  int queueLength(int id = 0) {
    int n = 0;
    for (auto* node = chan(id)->workBatchQueue.head; node != nullptr; node = node->next) ++n;
    return n;
  }
  ncclDevWorkBatch* tailBatch(int id = 0) {
    auto* t = chan(id)->workBatchQueue.tail;
    return t ? &t->batch : nullptr;
  }
};
}  // namespace

namespace {
void InstallSendRecvDevFuncIds() {
  auto key = [](int reg) -> uint64_t {
    return (uint64_t(ncclFuncSendRecv & RCCL_FUNC_ID_MASK) << RCCL_COLL_SHIFT) |
           (uint64_t(reg & RCCL_FUNC_ID_MASK) << RCCL_REG_SHIFT);
  };
  ncclDevFuncNameToId[key(0)] = 100;  // legacy LL kernel
  ncclDevFuncNameToId[key(1)] = 101;  // LL128 kernel
}
}  // namespace

namespace {
// addProxyOpIfNeeded enqueues to the op's CHANNEL queue
// (comm->planner.wipPlan.channels[op->channelId].proxyOpQueue) -- NOT to
// plan->proxyOpQueue. finishPlan is what later merges the channel queues into
// the plan. Looking in the wrong queue makes every one of these read "empty".
ncclProxyOp* ChannelProxyHead(BatchPlanComm& bp, int channelId) {
  return ncclIntruQueueHead(&bp.chan(channelId)->proxyOpQueue);
}
bool ChannelProxyEmpty(BatchPlanComm& bp, int channelId) {
  return ncclIntruQueueEmpty(&bp.chan(channelId)->proxyOpQueue);
}
}  // namespace

namespace {
// Minimal comm that satisfies collTaskAppend without entering the Broadcast /
// AllGatherV branch and without GPU allocation.
struct CollTaskComm {
  std::unique_ptr<ncclComm> storage{new ncclComm{}};
  CollTaskComm() {
    storage->nRanks = 4;
    ncclMemoryStackConstruct(&storage->memScoped);
    // memPermanent is load-bearing, not decorative; see BatchPlanComm for why.
    ncclMemoryStackConstruct(&storage->memPermanent);
  }
  ~CollTaskComm() {
    ncclMemoryStackDestruct(&storage->memScoped);
    ncclMemoryStackDestruct(&storage->memPermanent);
  }
  ncclComm* get() { return storage.get(); }
};
}  // namespace

namespace {
// A comm that passes CommCheck + ncclCommEnsureReady. The real argcheck.cc is
// compiled in as an oracle, so these fields must genuinely satisfy it.
struct RedOpComm {
  std::unique_ptr<ncclComm> storage{new ncclComm{}};
  RedOpComm() {
    ncclComm& c = *storage;
    // CommCheck validates startMagic AND endMagic -- the sentinels bracketing
    // the struct, not a single `magic` field.
    c.startMagic = NCCL_MAGIC;
    c.endMagic = NCCL_MAGIC;
    c.rank = 0;
    c.nRanks = 8;
    c.userRedOps = nullptr;
    c.userRedOpCapacity = 0;
    c.userRedOpFreeHead = 0;
    c.initState = ncclSuccess;
    c.finalizeCalled = 0;
  }
  ~RedOpComm() { delete[] storage->userRedOps; }
  ncclComm* get() { return storage.get(); }
};
}  // namespace

namespace {
std::unique_ptr<ncclComm> P2pPartsComm(int nNodes, int nRanks, int p2pnChannels, int perPeer) {
  std::unique_ptr<ncclComm> comm{new ncclComm{}};
  comm->nNodes = nNodes;
  comm->nRanks = nRanks;
  comm->p2pnChannels = p2pnChannels;
  comm->p2pnChannelsPerPeer = perPeer;
  comm->config.nChannelsPerNetPeer = NCCL_CONFIG_UNDEF_INT;
  return comm;
}
}  // namespace

namespace {
// Every param at its default.
void ClearParams() {
  g_loadParam = [](const char*, int64_t deft) { return deft; };
}

// Per-peer payload for the AllToAll peer-layer cases: large enough to be layered.
constexpr ssize_t kLayerPeerBytes = ssize_t(16) << 20;
// Test-only marker: leave RCCL_A2A_P2P_PEERS_IN_FLIGHT unset.
constexpr int64_t kPeersInFlightUnset = INT64_MIN;

void SetPeersInFlight(int64_t k) {
  if (k == kPeersInFlightUnset) {
    ClearParams();
  } else {
    SetParam("RCCL_A2A_P2P_PEERS_IN_FLIGHT", k);
  }
}
}  // namespace
