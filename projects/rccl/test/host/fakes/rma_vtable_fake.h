/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// One scripted ncclRma_t vtable for the host microtests that drive an RMA plugin. Every entry point records
// its arguments and returns its *Result knob; gin-host-proxy-test.cc is the first user.

#ifndef RCCL_TEST_HOST_RMA_VTABLE_FAKE_H_
#define RCCL_TEST_HOST_RMA_VTABLE_FAKE_H_

#include <cstdint>
#include <cstring>
#include <deque>
#include <set>
#include <utility>
#include <vector>

#include "nccl.h"
#include "plugin/nccl_rma.h"

struct FakeRmaVtable {
  // The handles the plugin hands out.
  static inline void* const kListenComm = reinterpret_cast<void*>(0x20);
  static inline void* const kCollComm = reinterpret_cast<void*>(0x30);
  static inline void* const kRmaCtx = reinterpret_cast<void*>(0x40);

  struct MrReg {
    void* collComm;
    void* addr;
    size_t size;
    int type;
    uint64_t mrFlags;
    int fd;  // -1 for a plain registration
    void* mhandle;
  };

  struct Op {
    char kind;  // 'p' iput, 's' iputSignal, 'g' iget, 'f' iflush
    void* rmaCtx;
    int context;
    uint64_t srcOff;
    void* srcHandle;
    size_t size;
    uint64_t dstOff;
    void* dstHandle;
    uint32_t rank;
    uint64_t signalOff;
    void* signalHandle;
    uint64_t signalValue;
    uint32_t signalOp;
    bool isStrong;
    uint32_t optFlags;
    void* request;
  };

  int maxRecvs = 1;
  int ptrSupport = NCCL_PTR_HOST | NCCL_PTR_CUDA;
  bool flushesAll = false;
  bool pluginHasError = false;
  bool flushReturnsNullRequest = false;
  ncclResult_t listenResult = ncclSuccess, connectResult = ncclSuccess, getPropertiesResult = ncclSuccess,
               closeListenResult = ncclSuccess, closeCollResult = ncclSuccess, createContextResult = ncclSuccess,
               destroyContextResult = ncclSuccess, getRmaPropertiesResult = ncclSuccess, regMrSymResult = ncclSuccess,
               deregMrSymResult = ncclSuccess, opResult = ncclSuccess, testResult = ncclSuccess,
               queryLastErrorResult = ncclSuccess;
  std::deque<ncclResult_t> regMrSymDmaBufResults;  // empty means success

  int listenDev = -1;
  void* listenHandle = nullptr;
  void* connectListenComm = nullptr;
  int connectNranks = -1, connectRank = -1;
  int getPropertiesDev = -1;
  std::vector<void*> closedListens, closedColls, destroyedContexts, progressed, queried, tested;
  ncclRmaConfig_t createConfig{};
  std::vector<MrReg> regs;
  std::vector<std::pair<void*, void*>> deregs;
  std::vector<Op> ops;
  std::set<void*> done;

  ncclRma_t vtable{};

  static FakeRmaVtable*& currentPtr() {
    static FakeRmaVtable* p = nullptr;
    return p;
  }
  static FakeRmaVtable& current() { return *currentPtr(); }
  static void setCurrent(FakeRmaVtable* p) { currentPtr() = p; }

  static ncclResult_t Init(void**, uint64_t, ncclDebugLogger_t) { return ncclSuccess; }
  static ncclResult_t Devices(int* ndev) {
    *ndev = 1;
    return ncclSuccess;
  }
  static ncclResult_t Finalize(void*) { return ncclSuccess; }
  static ncclResult_t GetProperties(int dev, ncclNetProperties_t* props) {
    FakeRmaVtable& f = current();
    f.getPropertiesDev = dev;
    std::memset(props, 0, sizeof(*props));
    props->maxRecvs = f.maxRecvs;
    props->ptrSupport = f.ptrSupport;
    return f.getPropertiesResult;
  }
  // Entry points taking a collComm reject the proxy's wrapper: the plugin must get its own back.
  static ncclResult_t GetRmaProperties(void* collComm, ncclRmaProperties_t* rmaProps) {
    FakeRmaVtable& f = current();
    if (collComm != kCollComm) return ncclInternalError;
    rmaProps->flushesAllPutsOnAnySignal = f.flushesAll;
    return f.getRmaPropertiesResult;
  }
  static ncclResult_t Listen(void*, int dev, void* handle, void** listenComm) {
    FakeRmaVtable& f = current();
    f.listenDev = dev;
    f.listenHandle = handle;
    if (f.listenResult != ncclSuccess) return f.listenResult;
    *listenComm = kListenComm;
    return ncclSuccess;
  }
  static ncclResult_t Connect(void*, void**, int nranks, int rank, void* listenComm, void** collComm) {
    FakeRmaVtable& f = current();
    f.connectNranks = nranks;
    f.connectRank = rank;
    f.connectListenComm = listenComm;
    if (f.connectResult != ncclSuccess) return f.connectResult;
    *collComm = kCollComm;
    return ncclSuccess;
  }
  static ncclResult_t CreateContext(void* collComm, ncclRmaConfig_t* config, void** rmaCtx) {
    FakeRmaVtable& f = current();
    if (collComm != kCollComm) return ncclInternalError;
    f.createConfig = *config;
    if (f.createContextResult != ncclSuccess) return f.createContextResult;
    *rmaCtx = kRmaCtx;
    return ncclSuccess;
  }
  static ncclResult_t RegMrSym(void* collComm, void* data, size_t size, int type, uint64_t mrFlags, void** mhandle) {
    FakeRmaVtable& f = current();
    if (f.regMrSymResult != ncclSuccess) return f.regMrSymResult;
    *mhandle = reinterpret_cast<void*>(0xA000 + f.regs.size());
    f.regs.push_back({collComm, data, size, type, mrFlags, -1, *mhandle});
    return ncclSuccess;
  }
  static ncclResult_t RegMrSymDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd,
                                     uint64_t mrFlags, void** mhandle) {
    FakeRmaVtable& f = current();
    ncclResult_t res = ncclSuccess;
    if (!f.regMrSymDmaBufResults.empty()) {
      res = f.regMrSymDmaBufResults.front();
      f.regMrSymDmaBufResults.pop_front();
    }
    if (offset != 0 || fd < 0) return ncclInternalError;
    if (res != ncclSuccess) return res;
    *mhandle = reinterpret_cast<void*>(0xA000 + f.regs.size());
    f.regs.push_back({collComm, data, size, type, mrFlags, fd, *mhandle});
    return ncclSuccess;
  }
  static ncclResult_t DeregMrSym(void* collComm, void* mhandle) {
    FakeRmaVtable& f = current();
    f.deregs.push_back({collComm, mhandle});
    return f.deregMrSymResult;
  }
  static ncclResult_t DestroyContext(void* rmaCtx) {
    FakeRmaVtable& f = current();
    f.destroyedContexts.push_back(rmaCtx);
    return f.destroyContextResult;
  }
  static ncclResult_t CloseColl(void* collComm) {
    FakeRmaVtable& f = current();
    f.closedColls.push_back(collComm);
    return f.closeCollResult;
  }
  static ncclResult_t CloseListen(void* listenComm) {
    FakeRmaVtable& f = current();
    f.closedListens.push_back(listenComm);
    return f.closeListenResult;
  }
  static ncclResult_t Submit(Op op, void** request) {
    FakeRmaVtable& f = current();
    op.request = reinterpret_cast<void*>(0x5000 + f.ops.size());
    if (op.kind == 'f' && f.flushReturnsNullRequest) op.request = nullptr;
    f.ops.push_back(op);
    if (f.opResult != ncclSuccess) return f.opResult;
    *request = op.request;
    return ncclSuccess;
  }
  static ncclResult_t Iput(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size, uint64_t dstOff,
                           void* dstMhandle, uint32_t rank, uint32_t optFlags, void** request) {
    return Submit({'p', rmaCtx, context, srcOff, srcMhandle, size, dstOff, dstMhandle, rank, 0, nullptr, 0, 0, false,
                   optFlags, nullptr},
                  request);
  }
  static ncclResult_t IputSignal(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size,
                                 uint64_t dstOff, void* dstMhandle, uint32_t rank, uint64_t signalOff,
                                 void* signalMhandle, uint64_t signalValue, uint32_t signalOp, bool isStrongSignal,
                                 uint32_t optFlags, void** request) {
    return Submit({'s', rmaCtx, context, srcOff, srcMhandle, size, dstOff, dstMhandle, rank, signalOff, signalMhandle,
                   signalValue, signalOp, isStrongSignal, optFlags, nullptr},
                  request);
  }
  static ncclResult_t Iget(void* rmaCtx, int context, uint64_t remoteOff, void* remoteMhandle, size_t size,
                           uint64_t localOff, void* localMhandle, uint32_t rank, uint32_t optFlags, void** request) {
    return Submit({'g', rmaCtx, context, remoteOff, remoteMhandle, size, localOff, localMhandle, rank, 0, nullptr, 0,
                   0, false, optFlags, nullptr},
                  request);
  }
  static ncclResult_t Iflush(void* rmaCtx, int context, void* mhandle, uint32_t rank, void** request) {
    return Submit({'f', rmaCtx, context, 0, nullptr, 0, 0, mhandle, rank, 0, nullptr, 0, 0, false, 0, nullptr},
                  request);
  }
  static ncclResult_t Test(void* collComm, void* request, int* done) {
    FakeRmaVtable& f = current();
    f.tested.push_back(request);
    if (collComm != kCollComm) return ncclInternalError;
    if (f.testResult != ncclSuccess) return f.testResult;
    *done = f.done.count(request) ? 1 : 0;
    return ncclSuccess;
  }
  static ncclResult_t RmaProgress(void* rmaCtx) {
    current().progressed.push_back(rmaCtx);
    return ncclSuccess;
  }
  static ncclResult_t QueryLastError(void* rmaCtx, bool* hasError) {
    FakeRmaVtable& f = current();
    f.queried.push_back(rmaCtx);
    *hasError = f.pluginHasError;
    return f.queryLastErrorResult;
  }

  FakeRmaVtable() {
    vtable.name = "FakeRmaVtable";
    vtable.init = &Init;
    vtable.devices = &Devices;
    vtable.getRmaProperties = &GetRmaProperties;
    vtable.getProperties = &GetProperties;
    vtable.listen = &Listen;
    vtable.connect = &Connect;
    vtable.createContext = &CreateContext;
    vtable.regMrSym = &RegMrSym;
    vtable.regMrSymDmaBuf = &RegMrSymDmaBuf;
    vtable.deregMrSym = &DeregMrSym;
    vtable.destroyContext = &DestroyContext;
    vtable.closeColl = &CloseColl;
    vtable.closeListen = &CloseListen;
    vtable.iput = &Iput;
    vtable.iputSignal = &IputSignal;
    vtable.iget = &Iget;
    vtable.iflush = &Iflush;
    vtable.test = &Test;
    vtable.rmaProgress = &RmaProgress;
    vtable.queryLastError = &QueryLastError;
    vtable.finalize = &Finalize;
  }
};

#endif  // RCCL_TEST_HOST_RMA_VTABLE_FAKE_H_
