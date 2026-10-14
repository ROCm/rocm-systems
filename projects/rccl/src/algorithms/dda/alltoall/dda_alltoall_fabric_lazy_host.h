/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Lazy creation of the DDA AllToAll fabric buffers: the push staging buffer
 * (dda_alltoall_fabric_push_host.h) and the Direct flag buffer
 * (dda_alltoall_fabric_direct_host.h). Included at the end of dda_alltoall_fabric.cu,
 * after both.
 *
 * Neither buffer is created at comm init: comms that never run an AllToAll in the push / Direct
 * windows allocate nothing (the push buffer is RCCL_DDA_A2A_PUSH_BUFFER_SIZE, 1 GiB by default;
 * the Direct flag buffer 2 MiB). The first ncclAlltoAll that push or Direct would take once the
 * buffers exist (a collective input: count, datatype, nRanks, in-place-ness, alignment and, for
 * Direct, a symmetric-window recvbuff are the same on every rank) creates both on every rank, with
 * one agreement (dda_alltoall_lazy.h): each rank allocates and exports what it can, one allgather
 * of {parts, size, export descriptors}, peer imports, one allgather of the import results. A part
 * that failed anywhere is disabled everywhere; nothing is retried.
 * If any rank's comm is nonblocking they are created at init instead (an API call of a nonblocking
 * comm must not wait for its peers; ncclDdaFabricCommInit gathers the flag so all ranks agree).
 *
 * Host rendezvous contract: the triggering call blocks in bootstrap until every rank makes it, so
 * every input of the trigger must be the same on all ranks (rcclSelectAlltoAllFabric). Per-rank
 * state that skips the trigger on one rank only -- a call inside ncclGroupStart/End on one rank
 * only, per-rank RCCL_DDA_* env, or a rank whose communicator already reported an AllToAll device
 * error (ncclAlltoAll refuses before selecting) -- leaves the other ranks waiting in bootstrap
 * until ncclCommAbort. Those cases already selected different kernels on different ranks before
 * (a device-side hang); they are contract violations or follow an error, not new failure modes.
 *
 * Safe inside a stream capture of another stream: the thread switches to relaxed capture mode, and
 * all device work (zeroing, peer table upload) runs on a private non-blocking stream that is
 * synchronized before returning -- no null-stream or device-wide synchronization (the pattern of
 * ncclDevrFinalize / ncclCuMemAlloc).
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/alltoall/dda_alltoall_lazy.h"

namespace {

struct DdaA2aLazyBlob {
  ncclCuDesc push;
  ncclCuDesc direct;
  uint64_t pushAlloc;    // exporter's mapped size (the importer maps the same size)
  uint64_t directAlloc;
};

class DdaA2aLazyOps {
 public:
  using Blob = DdaA2aLazyBlob;
  using Rec = dda_a2a_lazy::Rec<Blob>;

  DdaA2aLazyOps(ncclComm* comm, cudaStream_t stream) : comm_(comm), stream_(stream) {}
  int rank() const { return comm_->rank; }
  int nRanks() const { return comm_->nRanks; }
  ncclResult_t allGather(void* buf, size_t bytes) { return bootstrapAllGather(comm_->bootstrap, buf, (int)bytes); }

  void setup(uint32_t want, int64_t pushBytes, Rec* mine) {
    using namespace dda_a2a_lazy;
    // Export descriptors travel through bootstrap: only FABRIC handles are meaningful in a peer
    // (the fabric DDA path runs on MNNVL comms, which use them).
    if (stream_ == nullptr || ncclCuMemHandleType != CU_MEM_HANDLE_TYPE_FABRIC) return;
    // Created at the first large AllToAll, i.e. possibly with most of the device memory already in
    // use: if the configured size does not fit with kPushMinFreeBytes left for p2p, offer the
    // largest half-size that does (down to kPushMinBytes); the agreement takes the minimum over
    // ranks. A rank with no size that fits offers none, and push is off on every rank.
    size_t freeB = 0, totalB = 0;
    if ((want & kPush) && cudaMemGetInfo(&freeB, &totalB) != cudaSuccess) freeB = 0;
    (void)cudaGetLastError();
    const std::vector<int64_t> sizes = (want & kPush) ? pushSizes(pushBytes, comm_->nRanks,
                                                                   (int64_t)dda::common::kDdaA2aPushFlagBytes,
                                                                   (int64_t)freeB)
                                                      : std::vector<int64_t>{};
    for (int64_t b : sizes) {
      push_.alloc = dda::common::kDdaA2aPushFlagBytes + (size_t)b;
      if (createExport(&push_, &mine->blob.push)) {
        mine->ok |= kPush;
        mine->pushBytes = b;
        mine->blob.pushAlloc = push_.alloc;
        break;
      }
      releasePart(&push_);
    }
    if ((want & kPush) && sizes.empty()) {
      INFO(NCCL_INIT, "DDA A2A: rank %d offers no push buffer: %zu bytes free, needs %lld more than the buffer",
           comm_->rank, freeB, (long long)kPushMinFreeBytes);
    }
    if (want & kDirect) {
      direct_.alloc = dda::common::kDdaA2aDirectFlagBytes;
      if (createExport(&direct_, &mine->blob.direct)) {
        mine->ok |= kDirect;
        mine->blob.directAlloc = direct_.alloc;
      } else {
        releasePart(&direct_);
      }
    }
    // Zeroing done before any peer learns that this rank is ready.
    if (mine->ok && cudaStreamSynchronize(stream_) != cudaSuccess) {
      releasePart(&push_);
      releasePart(&direct_);
      mine->ok = 0;
    }
    (void)cudaGetLastError();
  }

  uint32_t import(const Rec* all, uint32_t parts, int64_t /*pushBytes*/) {
    using namespace dda_a2a_lazy;
    uint32_t ok = 0;
    if ((parts & kPush) && importPart(&push_, all, true)) ok |= kPush;
    if ((parts & kDirect) && importPart(&direct_, all, false)) ok |= kDirect;
    if (ok && cudaStreamSynchronize(stream_) != cudaSuccess) ok = 0;
    (void)cudaGetLastError();
    return ok;
  }

  void release(uint32_t parts) {
    if (parts & dda_a2a_lazy::kPush) releasePart(&push_);
    if (parts & dda_a2a_lazy::kDirect) releasePart(&direct_);
  }

  // Successful parts change owner (the comm state); release() no longer touches them.
  struct Part {
    void* base = nullptr;
    size_t alloc = 0;
    CUmemGenericAllocationHandle handle{};
    std::vector<void*> peerMaps;  // imported peer buffers (self slot = base)
    char** peerDev = nullptr;     // device table of peerMaps
  };
  Part push_, direct_;

 private:
  bool createExport(Part* p, ncclCuDesc* desc) {
    // Zeroed by ncclCuMemAlloc (the flags must start at 0; the payload needs no initial value).
    const ncclResult_t r = ncclCuMemAlloc(&p->base, &p->handle, ncclCuMemHandleType, p->alloc, comm_->memManager);
    (void)cudaGetLastError();
    if (r != ncclSuccess || p->base == nullptr) {
      p->base = nullptr;
      INFO(NCCL_INIT, "DDA A2A: rank %d could not allocate %zu bytes for the AllToAll fabric buffers", comm_->rank,
           p->alloc);
      return false;
    }
    return cuMemExportToShareableHandle(desc, p->handle, ncclCuMemHandleType, 0) == CUDA_SUCCESS;
  }

  bool importPart(Part* p, const Rec* all, bool push) {
    const int n = comm_->nRanks;
    p->peerMaps.assign(n, nullptr);
    p->peerMaps[comm_->rank] = p->base;
    for (int i = 0; i < n; ++i) {
      if (i == comm_->rank) continue;
      ncclCuDesc desc = push ? all[i].blob.push : all[i].blob.direct;
      const size_t size = push ? all[i].blob.pushAlloc : all[i].blob.directAlloc;
      CUmemGenericAllocationHandle h{};
      if (cuMemImportFromShareableHandle(&h, (void*)&desc, ncclCuMemHandleType) != CUDA_SUCCESS) return false;
      void* ptr = nullptr;
      const ncclResult_t r = ncclCuMemAllocAddr(&ptr, &h, size);
      (void)cuMemRelease(h);
      if (r != ncclSuccess || ptr == nullptr) return false;
      p->peerMaps[i] = ptr;
    }
    if (cudaMalloc((void**)&p->peerDev, n * sizeof(char*)) != cudaSuccess) {
      p->peerDev = nullptr;
      return false;
    }
    return cudaMemcpyAsync(p->peerDev, p->peerMaps.data(), n * sizeof(char*), cudaMemcpyHostToDevice, stream_) ==
           cudaSuccess;
  }

  void releasePart(Part* p) {
    for (size_t i = 0; i < p->peerMaps.size(); ++i) {
      if ((int)i != comm_->rank && p->peerMaps[i] != nullptr) (void)ncclCuMemFreeAddr(p->peerMaps[i], comm_->memManager);
    }
    p->peerMaps.clear();
    if (p->peerDev != nullptr) CUDACHECKIGNORE(cudaFree(p->peerDev));
    p->peerDev = nullptr;
    if (p->base != nullptr) (void)ncclCuMemFree(p->base, comm_->memManager);
    p->base = nullptr;
    (void)cudaGetLastError();
  }

  ncclComm* comm_;
  cudaStream_t stream_;
};

}  // namespace

ncclResult_t ncclDdaA2aLazyInit(ncclComm* comm) {
  using namespace dda_a2a_lazy;
  if (comm == nullptr || comm->ddaA2aLazy != kUntried) return ncclSuccess;
  comm->ddaA2aLazy = kOff;  // also on every early return below: never retried
  // Only on a comm whose fabric DDA init succeeded (ncclDdaFabricCommInit sets kUntried last).
  int64_t pushBytes = rcclParamDdaA2aPushBufferSize();
  pushBytes = pushBytes > 0 ? pushBytes / 4096 * 4096 : 0;
  uint32_t want = 0;
  if (rcclParamDdaA2aPush() != 0 && comm->nRanks <= dda::common::kDdaA2aPushMaxRanks && pushBytes > 0) want |= kPush;
  if (rcclParamDdaA2aDirect() != 0 && comm->nRanks <= dda::common::kDdaA2aDirectMaxRanks &&
      comm->localRanks == comm->nRanks)
    want |= kDirect;
  // Parts no rank wants still go through the agreement (a rank with a part disabled by env joins as
  // a failure): the bootstrap call sequence must not depend on per-rank env.
  // Host state objects first: nothing may fail after the agreement.
  auto* pst = new (std::nothrow) DdaA2aPushState();
  auto* dst = new (std::nothrow) DdaA2aDirectState();
  if (pst == nullptr) want &= ~kPush;
  if (dst == nullptr) want &= ~kDirect;

  const uint64_t t0 = clockNano();
  // From here on no early return: a local failure joins the agreement as "nothing set up".
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  const bool modeSet = cudaThreadExchangeStreamCaptureMode(&mode) == cudaSuccess;
  cudaStream_t stream = nullptr;
  if (!modeSet || cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) stream = nullptr;
  (void)cudaGetLastError();
  DdaA2aLazyOps ops(comm, stream);
  Result res;
  const ncclResult_t r = agree(ops, want, pushBytes, &res);
  if (stream != nullptr) CUDACHECKIGNORE(cudaStreamDestroy(stream));
  (void)cudaGetLastError();
  if (modeSet) (void)cudaThreadExchangeStreamCaptureMode(&mode);
  if (r != ncclSuccess || !(res.parts & kPush)) delete pst;
  if (r != ncclSuccess || !(res.parts & kDirect)) delete dst;
  if (r != ncclSuccess) {
    WARN("DDA AllToAll: rank %d: agreement on the fabric buffers failed (bootstrap error %d)", comm->rank, (int)r);
    return r;
  }

  if (res.parts & kPush) {
    auto* st = pst;
    st->peerMaps = std::move(ops.push_.peerMaps);
    st->base = ops.push_.base;
    st->dataBytes = (size_t)res.pushBytes;  // <= every rank's allocation
    st->peerBaseDev = ops.push_.peerDev;
    // Same-package sibling from topology (CU-split kernel), from the gathered peerInfo: the same
    // value on every rank for every rank, so allPaired is comm-uniform.
    auto hostOf = [comm](int r) { return comm->peerInfo[r].hostHash; };
    auto busOf = [comm](int r) { return comm->peerInfo[r].busId; };
    st->sibRank = rcclDdaA2aTopoSibling(comm->nRanks, comm->rank, hostOf, busOf);
    st->allPaired = comm->peerInfo != nullptr;
    for (int r = 0; st->allPaired && r < comm->nRanks; ++r) {
      st->allPaired = rcclDdaA2aTopoSibling(comm->nRanks, r, hostOf, busOf) >= 0;
    }
    comm->ddaA2aPushState = st;
    ops.push_ = DdaA2aLazyOps::Part{};
  }
  if (res.parts & kDirect) {
    auto* st = dst;
    st->peerMaps = std::move(ops.direct_.peerMaps);
    st->base = ops.direct_.base;
    st->peerFlagsDev = ops.direct_.peerDev;
    comm->ddaA2aDirectState = st;
    ops.direct_ = DdaA2aLazyOps::Part{};
  }
  comm->ddaA2aLazy = kDone;
  {
    const auto* st = static_cast<const DdaA2aPushState*>(comm->ddaA2aPushState);
    INFO(NCCL_INIT,
         "DDA AllToAll: rank %d created the fabric buffers in %.3f ms: push %s (data %lld bytes), direct %s, "
         "same-package sibling rank %d (topology), every rank paired %d (CU-split push %s)",
         comm->rank, (clockNano() - t0) / 1e6, (res.parts & kPush) ? "on" : "off", (long long)res.pushBytes,
         (res.parts & kDirect) ? "on" : "off",
         st != nullptr ? st->sibRank : -1, st != nullptr && st->allPaired ? 1 : 0,
         st != nullptr && st->allPaired ? "allowed" : "off: no verified sibling for every rank");
  }
  if ((want & kPush) && !(res.parts & kPush)) {
    WARN("DDA AllToAll: rank %d: the push staging buffer could not be created on every rank (device memory: "
         "each rank needs the buffer plus 2 GiB free); "
         "out-of-place AllToAll above the LL128 window takes p2p on this communicator "
         "(RCCL_DDA_A2A_PUSH_BUFFER_SIZE)", comm->rank);
  } else if ((want & kPush) && res.pushBytes < pushBytes) {
    INFO(NCCL_INIT, "DDA AllToAll: rank %d: push buffer %lld bytes instead of %lld (largest size every rank could allocate)",
         comm->rank, (long long)res.pushBytes, (long long)pushBytes);
  }
  if ((want & kDirect) && !(res.parts & kDirect))
    WARN("DDA AllToAll: rank %d: the zero-copy flag buffer could not be created on every rank; Direct is off on "
         "this communicator", comm->rank);
  return ncclSuccess;
}
