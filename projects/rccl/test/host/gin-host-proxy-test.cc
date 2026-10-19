/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/gin/gin_host_proxy.cc, the GIN proxy backend over an RMA plugin, and the
// versioned GPU-context writers in src/gin/proxy_gpucontext/. The RMA plugin, device allocations and the
// DMA-BUF export are seamed in this TU; GFDs are posted straight into the host-side queues.

#include <gtest/gtest.h>

#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "nccl.h"
#include "alloc.h"
#include "gdrwrap.h"
#include "comm.h"
#include "plugin/nccl_rma.h"
#include "gin/gin_host_proxy.h"
#include "gin/proxy_gpucontext/gpucontext.h"

// Logging and params: this binary links no shared fakes.
uint32_t ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED;  // log everything
uint64_t ncclDebugMask = ~uint64_t(0);
thread_local int ncclDebugNoWarn = 0;
std::function<int64_t(const char*, int64_t)> g_loadParam = [](const char*, int64_t deft) { return deft; };

namespace {
std::vector<std::string> g_logs;

bool Logged(const char* needle) {
  for (const auto& m : g_logs) {
    if (m.find(needle) != std::string::npos) return true;
  }
  return false;
}

int64_t g_dmaBufEnable = 0;
}  // namespace

void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int, const char* fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  g_logs.push_back(buf);
}

int64_t ncclParamDmaBufEnable() { return g_dmaBufEnable; }
int64_t ncclParamIbDataDirect() { return 0; }
gdr_t ncclGdrCopy = nullptr;

// HIP: no runtime is linked, so these are the only definitions.
namespace {
hipError_t g_memsetResult = hipSuccess;
std::deque<hipError_t> g_exportResults;  // one per DMA-BUF export; empty means success
std::vector<size_t> g_exportSizes;
std::vector<int> g_exportedFds;
}  // namespace

const char* hipGetErrorString(hipError_t) { return "fake hip error"; }

hipError_t hipMemset(void* dst, int value, size_t sizeBytes) {
  if (g_memsetResult != hipSuccess) return g_memsetResult;
  std::memset(dst, value, sizeBytes);
  return hipSuccess;
}

// hipMemGetHandleForAddressRange needs HIP >= 7.0; rma_proxy.cc guards it the same way.
#if HIP_VERSION >= 70000000
hipError_t hipMemGetHandleForAddressRange(void* handle, hipDeviceptr_t, size_t size, hipMemRangeHandleType,
                                          unsigned long long) {
  g_exportSizes.push_back(size);
  hipError_t res = hipSuccess;
  if (!g_exportResults.empty()) {
    res = g_exportResults.front();
    g_exportResults.pop_front();
  }
  if (res != hipSuccess) return res;
  int fd = open("/dev/null", O_RDONLY);
  g_exportedFds.push_back(fd);
  *static_cast<int*>(handle) = fd;
  return hipSuccess;
}
#endif

// Device allocations: host memory standing in for GPU/GDR buffers.
namespace {
struct AllocCall {
  std::string kind;
  size_t bytes;
  bool forceHost;
};

struct FakeMem {
  std::map<void*, std::string> live;
  std::vector<AllocCall> calls;
  int failAt = -1;  // index into calls that returns an error
  int memcpyCalls = 0;
  ncclResult_t memcpyResult = ncclSuccess;

  bool Record(const std::string& kind, size_t bytes, bool forceHost = false) {
    calls.push_back({kind, bytes, forceHost});
    return static_cast<int>(calls.size()) - 1 != failAt;
  }
  void* Alloc(const std::string& kind, size_t bytes) {
    void* p = std::calloc(1, bytes ? bytes : 1);
    live[p] = kind;
    return p;
  }
  ncclResult_t Free(void* p) {
    if (p == nullptr) return ncclSuccess;
    if (live.erase(p) == 0) return ncclInternalError;  // double free or foreign pointer
    std::free(p);
    return ncclSuccess;
  }
  void Reset() {
    for (auto& [p, kind] : live) std::free(p);
    *this = FakeMem{};
  }
};
FakeMem g_mem;
}  // namespace

template <typename T>
ncclResult_t FakeAllocMemCPUAccessible(T** ptr, T** devPtr, size_t nelem, int, void** gdrHandle,
                                       struct ncclMemManager*, bool forceHost = false) {
  if (!g_mem.Record("cpuAccessible", nelem * sizeof(T), forceHost)) return ncclUnhandledCudaError;
  *ptr = static_cast<T*>(g_mem.Alloc("cpuAccessible", nelem * sizeof(T)));
  if (ncclGdrCopy && !forceHost) {
    // GDR: the GPU sees a separate mapping, tracked through the handle.
    *devPtr = static_cast<T*>(g_mem.Alloc("gdrDev", nelem * sizeof(T)));
    *gdrHandle = *devPtr;
  } else {
    *devPtr = *ptr;
    if (gdrHandle) *gdrHandle = nullptr;
  }
  return ncclSuccess;
}

template <typename T>
ncclResult_t FakeFreeMemCPUAccessible(T* ptr, void* gdrHandle, struct ncclMemManager*) {
  if (gdrHandle) NCCLCHECK(g_mem.Free(gdrHandle));
  return g_mem.Free(ptr);
}

ncclResult_t FakeCuMemAlloc(void** ptr, hipMemGenericAllocationHandle_t*, hipMemAllocationHandleType, size_t size,
                            struct ncclMemManager* = nullptr) {
  if (!g_mem.Record("cuMem", size)) return ncclUnhandledCudaError;
  *ptr = g_mem.Alloc("cuMem", size);
  std::memset(*ptr, 0xff, size);  // like fresh device memory, not pre-zeroed
  return ncclSuccess;
}

template <typename T>
ncclResult_t FakeCudaCalloc(T** ptr, size_t nelem) {
  if (!g_mem.Record("cudaCalloc", nelem * ncclSizeOfT<T>())) return ncclUnhandledCudaError;
  *ptr = static_cast<T*>(g_mem.Alloc("cudaCalloc", nelem * ncclSizeOfT<T>()));
  return ncclSuccess;
}

template <typename T>
ncclResult_t FakeCudaFree(T* ptr, struct ncclMemManager*) {
  return g_mem.Free((void*)ptr);
}

template <typename T>
ncclResult_t FakeCudaMemcpy(T* dst, T* src, size_t nelem) {
  g_mem.memcpyCalls++;
  if (g_mem.memcpyResult != ncclSuccess) return g_mem.memcpyResult;
  std::memcpy((void*)dst, (void*)src, nelem * ncclSizeOfT<T>());
  return ncclSuccess;
}

#define allocMemCPUAccessible FakeAllocMemCPUAccessible
#define freeMemCPUAccessible FakeFreeMemCPUAccessible
#define ncclCuMemAlloc FakeCuMemAlloc
#undef ncclCudaCalloc
#define ncclCudaCalloc(ptr, nelem, ...) FakeCudaCalloc(ptr, nelem)
#define ncclCudaFree FakeCudaFree
#define ncclCudaMemcpy FakeCudaMemcpy

// Builds against HIP >= 7.12 compile the DMA-BUF export arm; test it on any HIP that has the export API.
#if HIP_VERSION >= 70000000
#undef NCCL_CUMEM_DMABUF_EXPORT_GATE
#define NCCL_CUMEM_DMABUF_EXPORT_GATE 1
#endif

#include "fakes/param_redirect.h"

#include GIN_HOST_PROXY_CC_PATH
#include GPUCONTEXT_CC_PATH
#include GPUCONTEXT_V1_CC_PATH
#include GPUCONTEXT_V2_CC_PATH

namespace {

void* const kPluginCtx = reinterpret_cast<void*>(0x10);
void* const kPluginListenComm = reinterpret_cast<void*>(0x20);
void* const kPluginCollComm = reinterpret_cast<void*>(0x30);
void* const kPluginRmaCtx = reinterpret_cast<void*>(0x40);

struct MrReg {
  void* collComm;
  void* addr;
  size_t size;
  int type;
  uint64_t mrFlags;
  int fd;  // -1 for a plain registration
  void* mhandle;
};

struct RmaOp {
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

// Scripted ncclRma_t: every entry point records its arguments and returns its *Result knob.
struct FakeRma {
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
  std::vector<RmaOp> ops;
  std::set<void*> done;

  ncclRma_t vtable{};

  static FakeRma*& Current() {
    static FakeRma* p = nullptr;
    return p;
  }

  static ncclResult_t Init(void**, uint64_t, ncclDebugLogger_t) { return ncclSuccess; }
  static ncclResult_t Devices(int* ndev) {
    *ndev = 1;
    return ncclSuccess;
  }
  static ncclResult_t Finalize(void*) { return ncclSuccess; }
  static ncclResult_t GetProperties(int dev, ncclNetProperties_t* props) {
    FakeRma& f = *Current();
    f.getPropertiesDev = dev;
    std::memset(props, 0, sizeof(*props));
    props->maxRecvs = f.maxRecvs;
    props->ptrSupport = f.ptrSupport;
    return f.getPropertiesResult;
  }
  // Entry points taking a collComm reject the proxy's wrapper: the plugin must get its own back.
  static ncclResult_t GetRmaProperties(void* collComm, ncclRmaProperties_t* rmaProps) {
    FakeRma& f = *Current();
    if (collComm != kPluginCollComm) return ncclInternalError;
    rmaProps->flushesAllPutsOnAnySignal = f.flushesAll;
    return f.getRmaPropertiesResult;
  }
  static ncclResult_t Listen(void*, int dev, void* handle, void** listenComm) {
    FakeRma& f = *Current();
    f.listenDev = dev;
    f.listenHandle = handle;
    if (f.listenResult != ncclSuccess) return f.listenResult;
    *listenComm = kPluginListenComm;
    return ncclSuccess;
  }
  static ncclResult_t Connect(void*, void**, int nranks, int rank, void* listenComm, void** collComm) {
    FakeRma& f = *Current();
    f.connectNranks = nranks;
    f.connectRank = rank;
    f.connectListenComm = listenComm;
    if (f.connectResult != ncclSuccess) return f.connectResult;
    *collComm = kPluginCollComm;
    return ncclSuccess;
  }
  static ncclResult_t CreateContext(void* collComm, ncclRmaConfig_t* config, void** rmaCtx) {
    FakeRma& f = *Current();
    if (collComm != kPluginCollComm) return ncclInternalError;
    f.createConfig = *config;
    if (f.createContextResult != ncclSuccess) return f.createContextResult;
    *rmaCtx = kPluginRmaCtx;
    return ncclSuccess;
  }
  static ncclResult_t RegMrSym(void* collComm, void* data, size_t size, int type, uint64_t mrFlags, void** mhandle) {
    FakeRma& f = *Current();
    if (f.regMrSymResult != ncclSuccess) return f.regMrSymResult;
    *mhandle = reinterpret_cast<void*>(0xA000 + f.regs.size());
    f.regs.push_back({collComm, data, size, type, mrFlags, -1, *mhandle});
    return ncclSuccess;
  }
  static ncclResult_t RegMrSymDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd,
                                     uint64_t mrFlags, void** mhandle) {
    FakeRma& f = *Current();
    ncclResult_t res = ncclSuccess;
    if (!f.regMrSymDmaBufResults.empty()) {
      res = f.regMrSymDmaBufResults.front();
      f.regMrSymDmaBufResults.pop_front();
    }
    if (offset != 0) return ncclInternalError;
    if (res != ncclSuccess) return res;
    *mhandle = reinterpret_cast<void*>(0xA000 + f.regs.size());
    f.regs.push_back({collComm, data, size, type, mrFlags, fd, *mhandle});
    return ncclSuccess;
  }
  static ncclResult_t DeregMrSym(void* collComm, void* mhandle) {
    FakeRma& f = *Current();
    f.deregs.push_back({collComm, mhandle});
    return f.deregMrSymResult;
  }
  static ncclResult_t DestroyContext(void* rmaCtx) {
    FakeRma& f = *Current();
    f.destroyedContexts.push_back(rmaCtx);
    return f.destroyContextResult;
  }
  static ncclResult_t CloseColl(void* collComm) {
    FakeRma& f = *Current();
    f.closedColls.push_back(collComm);
    return f.closeCollResult;
  }
  static ncclResult_t CloseListen(void* listenComm) {
    FakeRma& f = *Current();
    f.closedListens.push_back(listenComm);
    return f.closeListenResult;
  }
  static ncclResult_t Submit(RmaOp op, void** request) {
    FakeRma& f = *Current();
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
    FakeRma& f = *Current();
    f.tested.push_back(request);
    if (collComm != kPluginCollComm) return ncclInternalError;
    if (f.testResult != ncclSuccess) return f.testResult;
    *done = f.done.count(request) ? 1 : 0;
    return ncclSuccess;
  }
  static ncclResult_t RmaProgress(void* rmaCtx) {
    Current()->progressed.push_back(rmaCtx);
    return ncclSuccess;
  }
  static ncclResult_t QueryLastError(void* rmaCtx, bool* hasError) {
    FakeRma& f = *Current();
    f.queried.push_back(rmaCtx);
    *hasError = f.pluginHasError;
    return f.queryLastErrorResult;
  }

  FakeRma() {
    vtable.name = "FakeRma";
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

bool FdIsOpen(int fd) { return fcntl(fd, F_GETFD) != -1; }

class GinHostProxyMicrotest : public ::testing::Test {
 protected:
  FakeRma rma_;
  std::unique_ptr<ncclComm> comm_ = std::make_unique<ncclComm>();
  int64_t queueSizeParam_ = -1;
  int64_t pollBatchParam_ = 32;
  char handle_[NCCL_NET_HANDLE_MAXSIZE] = {};
  void* handles_[2] = {handle_, handle_};
  void* listenComm_ = nullptr;
  void* collComm_ = nullptr;

  void SetUp() override {
    FakeRma::Current() = &rma_;
    g_logs.clear();
    g_mem.Reset();
    g_dmaBufEnable = 0;
    g_memsetResult = hipSuccess;
    g_exportResults.clear();
    g_exportSizes.clear();
    g_exportedFds.clear();
    ncclGdrCopy = nullptr;
    g_loadParam = [this](const char* env, int64_t deft) -> int64_t {
      if (std::strcmp(env, "GIN_PROXY_QUEUE_SIZE") == 0) return queueSizeParam_;
      if (std::strcmp(env, "GIN_PROXY_POLL_BATCH") == 0) return pollBatchParam_;
      return deft;
    };
    comm_->rmaState.rmaProxyState.ncclRma = &rma_.vtable;
    comm_->rmaState.rmaProxyState.rmaVersion = 16;
    ASSERT_EQ(ncclSuccess, ncclGinProxyInit(comm_.get()));
  }

  void TearDown() override {
    g_mem.Reset();
    ncclGdrCopy = nullptr;
    g_loadParam = [](const char*, int64_t deft) { return deft; };
    FakeRma::Current() = nullptr;
  }

  ncclResult_t ListenAndConnect(int nRanks) {
    NCCLCHECK(ncclGinProxy.listen(kPluginCtx, /*dev=*/3, handle_, &listenComm_));
    return ncclGinProxy.connect(kPluginCtx, handles_, nRanks, /*rank=*/1, listenComm_, &collComm_);
  }

  void CloseAll() {
    if (collComm_) EXPECT_EQ(ncclSuccess, ncclGinProxy.closeColl(collComm_));
    if (listenComm_) EXPECT_EQ(ncclSuccess, ncclGinProxy.closeListen(listenComm_));
    collComm_ = listenComm_ = nullptr;
  }

  ncclGinProxyCollComm* CollComm() { return static_cast<ncclGinProxyCollComm*>(collComm_); }
};

TEST_F(GinHostProxyMicrotest, Init_WithoutAnRmaBackendFails) {
  comm_->rmaState.rmaProxyState.ncclRma = nullptr;
  EXPECT_EQ(ncclInvalidArgument, ncclGinProxyInit(comm_.get()));
  EXPECT_TRUE(Logged("RMA backend not found"));
}

TEST_F(GinHostProxyMicrotest, Init_ForwardsThePluginEntryPointsAndVersion) {
  comm_->rmaState.rmaProxyState.rmaVersion = 15;
  ASSERT_EQ(ncclSuccess, ncclGinProxyInit(comm_.get()));
  EXPECT_STREQ("FakeRma", ncclGinProxy.name);
  EXPECT_EQ(rma_.vtable.init, ncclGinProxy.init);
  EXPECT_EQ(rma_.vtable.devices, ncclGinProxy.devices);
  EXPECT_EQ(rma_.vtable.getProperties, ncclGinProxy.getProperties);
  EXPECT_EQ(rma_.vtable.finalize, ncclGinProxy.finalize);
  EXPECT_EQ(15, ncclGinProxyVersion);
}

TEST_F(GinHostProxyMicrotest, GetGinProperties_SupportsStrongAndVaSignals) {
  ncclGinProperties_t props{};
  ASSERT_EQ(ncclSuccess, ncclGinProxy.getGinProperties(&props));
  EXPECT_TRUE(props.supportsStrongSignals);
  EXPECT_TRUE(props.supportsVASignals);
}

TEST_F(GinHostProxyMicrotest, ListenConnectClose_WrapThePluginComms) {
  rma_.maxRecvs = 4;
  ASSERT_EQ(ncclSuccess, ListenAndConnect(/*nRanks=*/2));
  EXPECT_EQ(3, rma_.listenDev);
  EXPECT_EQ(handle_, rma_.listenHandle);
  EXPECT_EQ(3, rma_.getPropertiesDev);
  EXPECT_EQ(kPluginListenComm, rma_.connectListenComm);
  EXPECT_EQ(2, rma_.connectNranks);
  EXPECT_EQ(1, rma_.connectRank);
  ASSERT_NE(nullptr, collComm_);
  EXPECT_EQ(kPluginCollComm, CollComm()->collComm);
  EXPECT_EQ(2, CollComm()->nRanks);
  EXPECT_EQ(4, CollComm()->props.maxRecvs);

  CloseAll();
  EXPECT_EQ(std::vector<void*>{kPluginCollComm}, rma_.closedColls);
  EXPECT_EQ(std::vector<void*>{kPluginListenComm}, rma_.closedListens);
}

TEST_F(GinHostProxyMicrotest, Listen_PluginFailureLeavesListenCommUnset) {
  rma_.listenResult = ncclSystemError;
  void* listenComm = kPluginCtx;
  EXPECT_EQ(ncclSystemError, ncclGinProxy.listen(kPluginCtx, 0, handle_, &listenComm));
  EXPECT_EQ(kPluginCtx, listenComm);
}

TEST_F(GinHostProxyMicrotest, Connect_GetPropertiesFailureSkipsConnect) {
  ASSERT_EQ(ncclSuccess, ncclGinProxy.listen(kPluginCtx, 0, handle_, &listenComm_));
  rma_.getPropertiesResult = ncclInternalError;
  void* collComm = kPluginCtx;
  EXPECT_EQ(ncclInternalError, ncclGinProxy.connect(kPluginCtx, handles_, 2, 0, listenComm_, &collComm));
  EXPECT_EQ(kPluginCtx, collComm);
  EXPECT_EQ(nullptr, rma_.connectListenComm);
  CloseAll();
}

TEST_F(GinHostProxyMicrotest, Connect_PluginFailureLeavesCollCommUnset) {
  ASSERT_EQ(ncclSuccess, ncclGinProxy.listen(kPluginCtx, 0, handle_, &listenComm_));
  rma_.connectResult = ncclRemoteError;
  void* collComm = kPluginCtx;
  EXPECT_EQ(ncclRemoteError, ncclGinProxy.connect(kPluginCtx, handles_, 2, 0, listenComm_, &collComm));
  EXPECT_EQ(kPluginCtx, collComm);
  CloseAll();
}

TEST_F(GinHostProxyMicrotest, Close_PropagatesPluginErrors) {
  ASSERT_EQ(ncclSuccess, ListenAndConnect(2));
  rma_.closeCollResult = ncclSystemError;
  rma_.closeListenResult = ncclSystemError;
  EXPECT_EQ(ncclSystemError, ncclGinProxy.closeColl(collComm_));
  EXPECT_EQ(ncclSystemError, ncclGinProxy.closeListen(listenComm_));
  rma_.closeCollResult = rma_.closeListenResult = ncclSuccess;
  CloseAll();
}

TEST_F(GinHostProxyMicrotest, FlushesAllPutsOnAnySignal_ReportsThePluginProperty) {
  ASSERT_EQ(ncclSuccess, ListenAndConnect(2));
  bool flushesAll = false;
  rma_.flushesAll = true;
  ASSERT_EQ(ncclSuccess, ncclGinProxyFlushesAllPutsOnAnySignal(collComm_, &flushesAll));
  EXPECT_TRUE(flushesAll);
  rma_.flushesAll = false;
  ASSERT_EQ(ncclSuccess, ncclGinProxyFlushesAllPutsOnAnySignal(collComm_, &flushesAll));
  EXPECT_FALSE(flushesAll);
  rma_.getRmaPropertiesResult = ncclInternalError;
  EXPECT_EQ(ncclInternalError, ncclGinProxyFlushesAllPutsOnAnySignal(collComm_, &flushesAll));
  CloseAll();
}

class GinHostProxyRegMrMicrotest : public GinHostProxyMicrotest {
 protected:
  char buf_[100] = {};
  void* mhandle_ = nullptr;
  void* ginHandle_ = nullptr;

  void SetUp() override {
    GinHostProxyMicrotest::SetUp();
    ASSERT_EQ(ncclSuccess, ListenAndConnect(2));
    g_dmaBufEnable = 1;
    CollComm()->props.ptrSupport |= NCCL_PTR_DMABUF;
  }
  void TearDown() override {
    CloseAll();
    GinHostProxyMicrotest::TearDown();
  }
  ncclResult_t Reg(int type) {
    return ncclGinProxy.regMrSym(collComm_, buf_, sizeof(buf_), type, /*mrFlags=*/7, &mhandle_, &ginHandle_);
  }
};

TEST_F(GinHostProxyRegMrMicrotest, Host_RegistersPlainWithoutDmaBuf) {
  ASSERT_EQ(ncclSuccess, Reg(NCCL_PTR_HOST));
  ASSERT_EQ(1u, rma_.regs.size());
  EXPECT_EQ(kPluginCollComm, rma_.regs[0].collComm);
  EXPECT_EQ(buf_, rma_.regs[0].addr);
  EXPECT_EQ(sizeof(buf_), rma_.regs[0].size);
  EXPECT_EQ(NCCL_PTR_HOST, rma_.regs[0].type);
  EXPECT_EQ(7u, rma_.regs[0].mrFlags);
  EXPECT_EQ(-1, rma_.regs[0].fd);
  EXPECT_EQ(rma_.regs[0].mhandle, mhandle_);
  EXPECT_EQ(mhandle_, ginHandle_);
  EXPECT_TRUE(g_exportSizes.empty());
}

#if HIP_VERSION >= 70000000
TEST_F(GinHostProxyRegMrMicrotest, Cuda_RegistersThroughAnExportedDmaBufAndClosesIt) {
  ASSERT_EQ(ncclSuccess, Reg(NCCL_PTR_CUDA));
  ASSERT_EQ(1u, g_exportedFds.size());
  EXPECT_EQ(std::vector<size_t>{(size_t)sysconf(_SC_PAGESIZE)}, g_exportSizes);  // page-aligned length
  ASSERT_EQ(1u, rma_.regs.size());
  EXPECT_EQ(g_exportedFds[0], rma_.regs[0].fd);
  EXPECT_EQ(NCCL_PTR_CUDA, rma_.regs[0].type);
  EXPECT_EQ(7u, rma_.regs[0].mrFlags);
  EXPECT_EQ(mhandle_, ginHandle_);
  EXPECT_FALSE(FdIsOpen(g_exportedFds[0]));
}
#endif

TEST_F(GinHostProxyRegMrMicrotest, Cuda_DmaBufDisabledRegistersPlain) {
  g_dmaBufEnable = 0;
  ASSERT_EQ(ncclSuccess, Reg(NCCL_PTR_CUDA));
  EXPECT_TRUE(g_exportSizes.empty());
  ASSERT_EQ(1u, rma_.regs.size());
  EXPECT_EQ(-1, rma_.regs[0].fd);
}

TEST_F(GinHostProxyRegMrMicrotest, Cuda_PluginWithoutDmaBufSupportRegistersPlain) {
  CollComm()->props.ptrSupport = NCCL_PTR_HOST | NCCL_PTR_CUDA;
  ASSERT_EQ(ncclSuccess, Reg(NCCL_PTR_CUDA));
  EXPECT_TRUE(g_exportSizes.empty());
  ASSERT_EQ(1u, rma_.regs.size());
  EXPECT_EQ(-1, rma_.regs[0].fd);
}

#if HIP_VERSION >= 70000000
TEST_F(GinHostProxyRegMrMicrotest, Cuda_ExportFailureFallsBackToPlain) {
  g_exportResults = {hipErrorNotSupported};
  ASSERT_EQ(ncclSuccess, Reg(NCCL_PTR_CUDA));
  EXPECT_EQ(1u, g_exportSizes.size());
  ASSERT_EQ(1u, rma_.regs.size());
  EXPECT_EQ(-1, rma_.regs[0].fd);
  EXPECT_EQ(mhandle_, ginHandle_);
}

// A rejected DMA-BUF registration retries once with a fresh, non-DataDirect export.
TEST_F(GinHostProxyRegMrMicrotest, Cuda_RejectedDmaBufRetriesWithASecondExport) {
  rma_.regMrSymDmaBufResults = {ncclSystemError, ncclSuccess};
  ASSERT_EQ(ncclSuccess, Reg(NCCL_PTR_CUDA));
  ASSERT_EQ(2u, g_exportedFds.size());
  ASSERT_EQ(1u, rma_.regs.size());
  EXPECT_EQ(g_exportedFds[1], rma_.regs[0].fd);
  EXPECT_FALSE(FdIsOpen(g_exportedFds[0]));
  EXPECT_FALSE(FdIsOpen(g_exportedFds[1]));
}

TEST_F(GinHostProxyRegMrMicrotest, Cuda_RejectedDmaBufAndFailedReexportFallsBackToPlain) {
  rma_.regMrSymDmaBufResults = {ncclSystemError};
  g_exportResults = {hipSuccess, hipErrorNotSupported};
  ASSERT_EQ(ncclSuccess, Reg(NCCL_PTR_CUDA));
  EXPECT_EQ(2u, g_exportSizes.size());
  ASSERT_EQ(1u, rma_.regs.size());
  EXPECT_EQ(-1, rma_.regs[0].fd);
}

// LATENT BUG (gin_host_proxy.cc:470): when the retry is rejected too, NCCLCHECK returns before the second
// export's fd is closed.
TEST_F(GinHostProxyRegMrMicrotest, Cuda_BothDmaBufRegistrationsRejected_LeaksTheSecondFd) {
  rma_.regMrSymDmaBufResults = {ncclSystemError, ncclRemoteError};
  EXPECT_EQ(ncclRemoteError, Reg(NCCL_PTR_CUDA));
  ASSERT_EQ(2u, g_exportedFds.size());
  EXPECT_TRUE(FdIsOpen(g_exportedFds[1]));
  EXPECT_TRUE(rma_.regs.empty());
  close(g_exportedFds[1]);
}
#endif

TEST_F(GinHostProxyRegMrMicrotest, UnknownPointerTypeIsRejected) {
  ginHandle_ = kPluginCtx;
  EXPECT_EQ(ncclInvalidUsage, Reg(NCCL_PTR_DMABUF));
  EXPECT_TRUE(rma_.regs.empty());
  EXPECT_EQ(kPluginCtx, ginHandle_);
}

TEST_F(GinHostProxyRegMrMicrotest, PlainRegistrationFailurePropagates) {
  rma_.regMrSymResult = ncclSystemError;
  ginHandle_ = kPluginCtx;
  EXPECT_EQ(ncclSystemError, Reg(NCCL_PTR_HOST));
  g_dmaBufEnable = 0;
  EXPECT_EQ(ncclSystemError, Reg(NCCL_PTR_CUDA));
  EXPECT_EQ(kPluginCtx, ginHandle_);
}

TEST_F(GinHostProxyRegMrMicrotest, Dereg_ForwardsThePluginCollCommAndHandle) {
  void* mh = reinterpret_cast<void*>(0x77);
  ASSERT_EQ(ncclSuccess, ncclGinProxy.deregMrSym(collComm_, mh));
  rma_.deregMrSymResult = ncclSystemError;
  EXPECT_EQ(ncclSystemError, ncclGinProxy.deregMrSym(collComm_, mh));
  ASSERT_EQ(2u, rma_.deregs.size());
  EXPECT_EQ(kPluginCollComm, rma_.deregs[0].first);
  EXPECT_EQ(mh, rma_.deregs[0].second);
}

class GinHostProxyContextMicrotest : public GinHostProxyMicrotest {
 protected:
  void* ginCtx_ = nullptr;
  ncclNetDeviceHandle_t* devHandle_ = nullptr;

  void SetUp() override {
    GinHostProxyMicrotest::SetUp();
    ASSERT_EQ(ncclSuccess, ListenAndConnect(/*nRanks=*/2));
  }
  void TearDown() override {
    if (ginCtx_) EXPECT_EQ(ncclSuccess, ncclGinProxy.destroyContext(ginCtx_));
    CloseAll();
    GinHostProxyMicrotest::TearDown();
  }

  static ncclGinConfig_t Config(int version, int nContexts = 2, int nSignals = 3, int nCounters = 4) {
    ncclGinConfig_t c{};
    c.nSignals = nSignals;
    c.nCounters = nCounters;
    c.nContexts = nContexts;
    c.trafficClass = 5;
    c.backendVersion = version;
    c.rankStride = 2;
    return c;
  }
  ncclResult_t Create(ncclGinConfig_t config) {
    return ncclGinProxy.createContext(collComm_, &config, &ginCtx_, &devHandle_);
  }
  void Destroy() {
    ASSERT_EQ(ncclSuccess, ncclGinProxy.destroyContext(ginCtx_));
    ginCtx_ = nullptr;
  }
  ginProxyCtx* Ctx() { return static_cast<ginProxyCtx*>(ginCtx_); }
  ncclGinProxyGpuCtx_t* GpuCtx() { return static_cast<ncclGinProxyGpuCtx_t*>(devHandle_->handle); }
  size_t QueueSizeFor(int64_t param) {
    queueSizeParam_ = param;
    void* ctx = nullptr;
    ncclNetDeviceHandle_t* dh = nullptr;
    ncclGinConfig_t c = Config(3, 1, 0, 0);
    if (ncclGinProxy.createContext(collComm_, &c, &ctx, &dh) != ncclSuccess) return 0;
    size_t q = static_cast<ginProxyCtx*>(ctx)->hostGpuCtx[0].queueSize;
    EXPECT_EQ(ncclSuccess, ncclGinProxy.destroyContext(ctx));
    return q;
  }
};

TEST_F(GinHostProxyContextMicrotest, Create_RejectsOutOfRangeBackendVersions) {
  EXPECT_EQ(ncclInternalError, Create(Config(-1)));
  EXPECT_EQ(ncclInternalError, Create(Config(NCCL_GIN_PROXY_GPU_CONTEXT_VERSION + 1)));
  EXPECT_EQ(nullptr, ginCtx_);
  EXPECT_TRUE(g_mem.calls.empty());
  EXPECT_EQ(0, rma_.createConfig.nContexts);
}

TEST_F(GinHostProxyContextMicrotest, Create_PublishesOneGpuContextPerContext) {
  rma_.flushesAll = true;
  ASSERT_EQ(ncclSuccess, Create(Config(3)));
  EXPECT_EQ(2, rma_.createConfig.nContexts);
  EXPECT_EQ(5, rma_.createConfig.trafficClass);
  EXPECT_EQ(2, rma_.createConfig.rankStride);

  ASSERT_NE(nullptr, devHandle_);
  EXPECT_EQ(NCCL_NET_DEVICE_GIN_PROXY, devHandle_->netDeviceType);
  EXPECT_EQ(NCCL_GIN_PROXY_VERSION, devHandle_->netDeviceVersion);
  EXPECT_EQ(0u, devHandle_->size);
  EXPECT_EQ(1, devHandle_->needsProxyProgress);
  EXPECT_EQ(1, g_mem.memcpyCalls);

  for (int c = 0; c < 2; c++) {
    SCOPED_TRACE(c);
    ncclGinProxyGpuCtx_t& g = GpuCtx()[c];
    ginProxyHostGpuCtx& h = Ctx()->hostGpuCtx[c];
    EXPECT_EQ(c, h.contextId);
    EXPECT_EQ(2, g.nranks);
    EXPECT_EQ(32u, g.queueSize);  // NCCL_NET_MAX_REQUESTS * maxRecvs
    EXPECT_EQ(h.queues, g.queues);
    EXPECT_EQ(h.cis, g.cis);
    EXPECT_EQ(h.pis, g.pis);
    EXPECT_EQ(h.lastIssuedGet, g.lastIssuedGet);
    EXPECT_EQ(h.lastVisibleGet, g.lastVisibleGet);
    EXPECT_EQ(Ctx()->countersDev + c * 4, g.counters);
    EXPECT_EQ(Ctx()->signalsDev + c * 3, g.signals);
    EXPECT_EQ(Ctx()->signalOffsetsDev + c * 3, g.signalOffsets);
    EXPECT_TRUE(g.flushesAllPutsOnAnySignal);
  }
  EXPECT_NE(GpuCtx()[0].queues, GpuCtx()[1].queues);
  EXPECT_EQ(32, Ctx()->pollBatch);
}

TEST_F(GinHostProxyContextMicrotest, Create_ForcesOnlyTheQueuesIntoHostMemory) {
  ASSERT_EQ(ncclSuccess, Create(Config(3, /*nContexts=*/1)));
  ASSERT_EQ(9u, g_mem.calls.size());
  EXPECT_EQ("cpuAccessible", g_mem.calls[0].kind);  // counters
  EXPECT_EQ(4 * sizeof(uint64_t), g_mem.calls[0].bytes);
  EXPECT_FALSE(g_mem.calls[0].forceHost);
  EXPECT_EQ("cuMem", g_mem.calls[1].kind);  // signals
  EXPECT_EQ(3 * sizeof(uint64_t), g_mem.calls[1].bytes);
  EXPECT_EQ("cudaCalloc", g_mem.calls[2].kind);  // signal offsets
  EXPECT_EQ(3 * sizeof(uint64_t), g_mem.calls[2].bytes);
  // pis, lastIssuedGet, lastVisibleGet: one uint32 per rank each.
  for (int i = 3; i < 6; i++) EXPECT_EQ(2 * sizeof(uint32_t), g_mem.calls[i].bytes) << i;
  EXPECT_EQ(32 * 2 * sizeof(ncclGinProxyGfd_t), g_mem.calls[6].bytes);  // queues
  EXPECT_TRUE(g_mem.calls[6].forceHost);
  EXPECT_EQ(2 * sizeof(uint32_t), g_mem.calls[7].bytes);  // cis
  EXPECT_FALSE(g_mem.calls[7].forceHost);
  EXPECT_EQ("cudaCalloc", g_mem.calls[8].kind);  // the GPU-context array
  EXPECT_EQ(NCCL_GIN_PROXY_GPU_CONTEXT_MAX_SIZE, g_mem.calls[8].bytes);
}

TEST_F(GinHostProxyContextMicrotest, Create_ZeroesAndRegistersSignalsStronglyOrderedAndNeverReset) {
  ASSERT_EQ(ncclSuccess, Create(Config(3)));
  ASSERT_EQ(3u, rma_.regs.size());  // signals, then inlines per context
  EXPECT_EQ(Ctx()->signalsDev, rma_.regs[0].addr);
  EXPECT_EQ(2 * 3 * sizeof(uint64_t), rma_.regs[0].size);
  EXPECT_EQ(NCCL_PTR_CUDA, rma_.regs[0].type);
  EXPECT_EQ((uint64_t)(NCCL_NET_MR_FLAG_FORCE_SO | NCCL_NET_MR_FLAG_SIGNAL_NEVER_RESET), rma_.regs[0].mrFlags);
  EXPECT_EQ(rma_.regs[0].mhandle, Ctx()->signalsGinHandle);
  for (int c = 0; c < 2; c++) {
    const MrReg& r = rma_.regs[1 + c];
    EXPECT_EQ(Ctx()->hostGpuCtx[c].inlines, r.addr);
    EXPECT_EQ(32 * 2 * sizeof(uint64_t), r.size);
    EXPECT_EQ(NCCL_PTR_HOST, r.type);
    EXPECT_EQ(0u, r.mrFlags);
    EXPECT_EQ(r.mhandle, Ctx()->hostGpuCtx[c].inlinesMhandle);
  }
}

// The fake cuMem allocation is 0xff-filled, so only the explicit memset can zero it.
TEST_F(GinHostProxyContextMicrotest, Create_ZeroesTheSignalBuffer) {
  ASSERT_EQ(ncclSuccess, Create(Config(3)));
  for (int i = 0; i < 2 * 3; i++) EXPECT_EQ(0u, Ctx()->signalsDev[i]) << i;
  Destroy();

  g_memsetResult = hipErrorInvalidValue;
  EXPECT_EQ(ncclUnhandledCudaError, Create(Config(3)));
}

TEST_F(GinHostProxyContextMicrotest, Create_SignalMrFlagsFollowTheBackendVersion) {
  ASSERT_EQ(ncclSuccess, Create(Config(2)));
  EXPECT_EQ((uint64_t)(NCCL_NET_MR_FLAG_FORCE_SO | NCCL_NET_MR_FLAG_SIGNAL_NEVER_RESET), rma_.regs[0].mrFlags);
  Destroy();
  rma_.regs.clear();
  ASSERT_EQ(ncclSuccess, Create(Config(1)));
  EXPECT_EQ((uint64_t)NCCL_NET_MR_FLAG_FORCE_SO, rma_.regs[0].mrFlags);  // v1 always reset signals
}

TEST_F(GinHostProxyContextMicrotest, Create_OlderBackendVersionsUseTheirOwnGpuContextLayout) {
  ASSERT_EQ(ncclSuccess, Create(Config(1)));
  auto* v1 = static_cast<ncclGinProxyGpuCtx_v1_t*>(devHandle_->handle);
  EXPECT_EQ(2, v1[1].nranks);
  EXPECT_EQ(Ctx()->hostGpuCtx[1].queues, v1[1].queues);
  EXPECT_EQ(Ctx()->signalsDev + 3, v1[1].signals);
  Destroy();

  ASSERT_EQ(ncclSuccess, Create(Config(2)));
  auto* v2 = static_cast<ncclGinProxyGpuCtx_v2_t*>(devHandle_->handle);
  EXPECT_EQ(Ctx()->hostGpuCtx[1].queues, v2[1].queues);
  EXPECT_EQ(Ctx()->signalOffsetsDev + 3, v2[1].signalOffsets);
  EXPECT_EQ(Ctx()->hostGpuCtx[1].lastVisibleGet, v2[1].lastVisibleGet);
}

// LATENT BUG (gin_host_proxy.cc:499, :604): 0 passes the version check up front, so the failure comes from
// the GPU-context writer after the RMA context and every buffer exist, and none of them is released.
TEST_F(GinHostProxyContextMicrotest, Create_BackendVersionZero_FailsLateAndLeaksEverything) {
  EXPECT_EQ(ncclInternalError, Create(Config(0)));
  EXPECT_EQ(nullptr, ginCtx_);
  EXPECT_TRUE(Logged("Invalid GIN proxy backend version 0"));
  EXPECT_EQ(2, rma_.createConfig.nContexts);  // the RMA context was created ...
  EXPECT_TRUE(rma_.destroyedContexts.empty());  // ... and never destroyed
  EXPECT_FALSE(g_mem.live.empty());
  EXPECT_FALSE(rma_.regs.empty());
  EXPECT_TRUE(rma_.deregs.empty());
}

TEST_F(GinHostProxyContextMicrotest, Create_WithoutSignalsOrCountersAllocatesNeither) {
  ASSERT_EQ(ncclSuccess, Create(Config(3, 1, /*nSignals=*/0, /*nCounters=*/0)));
  EXPECT_EQ(nullptr, Ctx()->counters);
  EXPECT_EQ(nullptr, Ctx()->signalsDev);
  EXPECT_EQ(nullptr, Ctx()->signalOffsetsDev);
  EXPECT_EQ(nullptr, GpuCtx()[0].counters);
  EXPECT_EQ(nullptr, GpuCtx()[0].signals);
  for (const auto& c : g_mem.calls) EXPECT_NE("cuMem", c.kind);
  ASSERT_EQ(1u, rma_.regs.size());
  EXPECT_EQ(NCCL_PTR_HOST, rma_.regs[0].type);
}

// Where counters live is allocMemCPUAccessible's call; the proxy logs the choice and must hand the GPU
// the device-side pointer either way.
TEST_F(GinHostProxyContextMicrotest, Create_GdrCopyChoosesTheCountersMappingTheGpuSees) {
  ASSERT_EQ(ncclSuccess, Create(Config(3)));
  EXPECT_TRUE(Logged("will not be using GDRCopy"));
  EXPECT_EQ(nullptr, Ctx()->countersGdrHandle);
  EXPECT_EQ(Ctx()->counters, GpuCtx()[0].counters);
  Destroy();
  g_logs.clear();
  ncclGdrCopy = reinterpret_cast<gdr_t>(0x1);
  ASSERT_EQ(ncclSuccess, Create(Config(3)));
  EXPECT_FALSE(Logged("GDRCopy"));
  EXPECT_NE(nullptr, Ctx()->countersGdrHandle);
  EXPECT_NE(Ctx()->counters, GpuCtx()[0].counters);
  EXPECT_EQ(Ctx()->countersDev, GpuCtx()[0].counters);
}

TEST_F(GinHostProxyContextMicrotest, Create_QueueSizeMustBeAPowerOfTwoWithinThePluginLimit) {
  EXPECT_EQ(32u, QueueSizeFor(-1));  // default
  EXPECT_EQ(16u, QueueSizeFor(16));
  EXPECT_EQ(1u, QueueSizeFor(1));
  EXPECT_EQ(32u, QueueSizeFor(32));
  EXPECT_EQ(32u, QueueSizeFor(64));
  EXPECT_TRUE(Logged("greater than the maximum outstanding requests"));
  EXPECT_EQ(32u, QueueSizeFor(-5));  // wraps to a huge unsigned value
  EXPECT_EQ(32u, QueueSizeFor(0));
  EXPECT_TRUE(Logged("less than 1"));
  EXPECT_EQ(32u, QueueSizeFor(12));
  EXPECT_TRUE(Logged("not a power of two"));
}

TEST_F(GinHostProxyContextMicrotest, Create_QueueLimitScalesWithMaxRecvs) {
  CollComm()->props.maxRecvs = 4;
  EXPECT_EQ(128u, QueueSizeFor(-1));
  EXPECT_EQ(64u, QueueSizeFor(64));
}

// LATENT BUG (gin_host_proxy.cc:528-546): the fallback size is never checked itself, so a plugin with a
// non-power-of-two maxRecvs gets a queue the index mask (queueSize - 1) cannot wrap.
TEST_F(GinHostProxyContextMicrotest, Create_NonPowerOfTwoMaxRecvs_KeepsANonPowerOfTwoQueue) {
  CollComm()->props.maxRecvs = 3;
  EXPECT_EQ(96u, QueueSizeFor(-1));
  EXPECT_EQ(96u, QueueSizeFor(12));
}

TEST_F(GinHostProxyContextMicrotest, Create_PollBatchIsAtLeastOne) {
  for (int64_t param : {0, -3}) {
    pollBatchParam_ = param;
    ASSERT_EQ(ncclSuccess, Create(Config(3)));
    EXPECT_EQ(1, Ctx()->pollBatch) << param;
    Destroy();
  }
  pollBatchParam_ = 5;
  ASSERT_EQ(ncclSuccess, Create(Config(3)));
  EXPECT_EQ(5, Ctx()->pollBatch);
}

TEST_F(GinHostProxyContextMicrotest, Create_PropagatesPluginAndAllocationFailures) {
  rma_.createContextResult = ncclSystemError;
  EXPECT_EQ(ncclSystemError, Create(Config(3)));
  rma_.createContextResult = ncclSuccess;
  rma_.getRmaPropertiesResult = ncclRemoteError;
  EXPECT_EQ(ncclRemoteError, Create(Config(3)));
  rma_.getRmaPropertiesResult = ncclSuccess;
  rma_.regMrSymResult = ncclInvalidUsage;
  EXPECT_EQ(ncclInvalidUsage, Create(Config(3)));
  rma_.regMrSymResult = ncclSuccess;
  g_mem.memcpyResult = ncclUnhandledCudaError;
  EXPECT_EQ(ncclUnhandledCudaError, Create(Config(3)));
  g_mem.memcpyResult = ncclSuccess;

  g_mem.Reset();
  ASSERT_EQ(ncclSuccess, Create(Config(3)));
  const int nAllocs = static_cast<int>(g_mem.calls.size());
  Destroy();
  // CreateContext has no cleanup on its error paths, so each attempt starts from a clean fake heap.
  for (int failAt = 0; failAt < nAllocs; failAt++) {
    g_mem.Reset();
    g_mem.failAt = failAt;
    EXPECT_EQ(ncclUnhandledCudaError, Create(Config(3))) << failAt;
    EXPECT_EQ((size_t)failAt + 1, g_mem.calls.size()) << "stopped at the failing allocation";
  }
}

TEST_F(GinHostProxyContextMicrotest, Destroy_NullIsANoOp) {
  EXPECT_EQ(ncclSuccess, ncclGinProxy.destroyContext(nullptr));
  EXPECT_TRUE(rma_.destroyedContexts.empty());
}

// LATENT BUG (gin_host_proxy.cc:575, :634-682): destroy releases everything CreateContext allocated except
// the signal-offsets buffer.
TEST_F(GinHostProxyContextMicrotest, Destroy_ReleasesAllButTheSignalOffsets) {
  ASSERT_EQ(ncclSuccess, Create(Config(3)));
  void* signalOffsets = Ctx()->signalOffsetsDev;
  void* signalsMh = Ctx()->signalsMhandle;
  void* inlinesMh0 = Ctx()->hostGpuCtx[0].inlinesMhandle;
  void* inlinesMh1 = Ctx()->hostGpuCtx[1].inlinesMhandle;
  Destroy();

  EXPECT_EQ(std::vector<void*>{kPluginRmaCtx}, rma_.destroyedContexts);
  std::vector<std::pair<void*, void*>> deregs = {
    {kPluginCollComm, signalsMh}, {kPluginCollComm, inlinesMh0}, {kPluginCollComm, inlinesMh1}};
  EXPECT_EQ(deregs, rma_.deregs);
  ASSERT_EQ(1u, g_mem.live.size());
  EXPECT_EQ(signalOffsets, g_mem.live.begin()->first);
}

TEST_F(GinHostProxyContextMicrotest, Destroy_FreesGdrCountersThroughTheirHandle) {
  ncclGdrCopy = reinterpret_cast<gdr_t>(0x1);
  ASSERT_EQ(ncclSuccess, Create(Config(3, 1, 0, /*nCounters=*/2)));
  void* gdrDev = Ctx()->countersGdrHandle;
  ASSERT_EQ(1u, g_mem.live.count(gdrDev));
  Destroy();
  EXPECT_TRUE(g_mem.live.empty());
}

TEST_F(GinHostProxyContextMicrotest, Destroy_PluginFailureReturnsBeforeFreeing) {
  ASSERT_EQ(ncclSuccess, Create(Config(3)));
  rma_.destroyContextResult = ncclSystemError;
  EXPECT_EQ(ncclSystemError, ncclGinProxy.destroyContext(ginCtx_));
  EXPECT_TRUE(rma_.deregs.empty());
  rma_.destroyContextResult = ncclSuccess;
}

TEST_F(GinHostProxyContextMicrotest, Destroy_IgnoresDeregistrationFailures) {
  ASSERT_EQ(ncclSuccess, Create(Config(3)));
  rma_.deregMrSymResult = ncclSystemError;
  EXPECT_EQ(ncclSuccess, ncclGinProxy.destroyContext(ginCtx_));
  ginCtx_ = nullptr;  // deregistration failures are ignored, so the context is gone
  EXPECT_EQ(3u, rma_.deregs.size());
}

class GinHostProxyProgressMicrotest : public GinHostProxyContextMicrotest {
 protected:
  static constexpr int kSignals = 4;
  static constexpr int kCounters = 5;
  std::map<std::pair<int, int>, uint32_t> pi_;

  void SetUp() override {
    GinHostProxyContextMicrotest::SetUp();
    queueSizeParam_ = 4;
    ASSERT_EQ(ncclSuccess, Create(Config(3, 2, kSignals, kCounters)));
  }

  static ncclGinProxyGfd_t Gfd(uint16_t op, uint64_t size = 0, int version = NCCL_GIN_PROXY_GFD_VERSION) {
    ncclGinProxyGfd_t g{};
    g.qword[ncclGinProxyGfdHeader].header.version = version;
    g.qword[ncclGinProxyGfdHeader].header.size = size;
    g.qword[ncclGinProxyGfdHeaderExt].headerExt.op = op;
    return g;
  }
  static ncclGinProxyGfd_t PutGfd(uint16_t op, uint64_t size) {
    ncclGinProxyGfd_t g = Gfd(ncclGinProxyOpPut | op, size);
    g.qword[ncclGinProxyGfdSrcOff].srcOff.srcOff = 0x10;
    g.qword[ncclGinProxyGfdSrcHandle].srcHandle.srcHandle = 0x2000;
    g.qword[ncclGinProxyGfdDstOff].dstOff.dstOff = 0x20;
    g.qword[ncclGinProxyGfdDstHandle].dstHandle.dstHandle = 0x3000;
    return g;
  }
  static void SetSignal(ncclGinProxyGfd_t& g, uint32_t signalId, uint64_t value, bool strong) {
    g.qword[ncclGinProxyGfdCompletion].completion.signalId = signalId;
    g.qword[ncclGinProxyGfdCompletion].completion.signalValLow = value & 0xffff;
    g.qword[ncclGinProxyGfdSignalVal].signalVal.signalValLow2 = (value >> 16) & 0xffff;
    g.qword[ncclGinProxyGfdSignalVal].signalVal.signalValHigh = value >> 32;
    g.qword[ncclGinProxyGfdSignalVal].signalVal.isStrongSignal = strong;
  }

  ncclGinProxyGfd_t* Slot(int context, int rank, uint32_t index) {
    ginProxyHostGpuCtx& h = Ctx()->hostGpuCtx[context];
    return h.queues + rank * h.queueSize + (index & (h.queueSize - 1));
  }
  // Producer side: every qword carries the flag bit, as the GPU writes it.
  void Post(int context, int rank, ncclGinProxyGfd_t g) {
    for (auto& q : g.qword) q.raw |= 1;
    *Slot(context, rank, pi_[{context, rank}]++) = g;
  }
  ncclResult_t Progress() { return ncclGinProxy.ginProgress(ginCtx_); }
  bool HasError() {
    bool hasError = false;
    EXPECT_EQ(ncclSuccess, ncclGinProxy.queryLastError(ginCtx_, &hasError));
    return hasError;
  }
  uint32_t Ci(int context, int rank) { return Ctx()->hostGpuCtx[context].cis[rank]; }
};

TEST_F(GinHostProxyProgressMicrotest, Put_SubmitsAnIputAndReleasesTheSlot) {
  Post(1, 1, PutGfd(0, 64));
  ASSERT_EQ(ncclSuccess, Progress());
  ASSERT_EQ(1u, rma_.ops.size());
  const RmaOp& op = rma_.ops[0];
  EXPECT_EQ('p', op.kind);
  EXPECT_EQ(kPluginRmaCtx, op.rmaCtx);
  EXPECT_EQ(1, op.context);
  EXPECT_EQ(0x10u, op.srcOff);
  EXPECT_EQ(reinterpret_cast<void*>(0x2000), op.srcHandle);
  EXPECT_EQ(64u, op.size);
  EXPECT_EQ(0x20u, op.dstOff);
  EXPECT_EQ(reinterpret_cast<void*>(0x3000), op.dstHandle);
  EXPECT_EQ(1u, op.rank);
  EXPECT_EQ((uint32_t)ncclRmaOptFlagsDefault, op.optFlags);
  for (const auto& q : Slot(1, 1, 0)->qword) EXPECT_EQ(0u, q.raw);
  EXPECT_EQ(1u, Ctx()->hostGpuCtx[1].sis[1]);
  EXPECT_EQ(std::vector<void*>{kPluginRmaCtx}, rma_.progressed);
}

// The header flag makes a GFD visible; the proxy then waits for each later qword's flag before reading it.
TEST_F(GinHostProxyProgressMicrotest, Poll_WaitsForTheRestOfAPartiallyWrittenGfd) {
  ncclGinProxyGfd_t g = PutGfd(0, 64);
  for (auto& q : g.qword) q.raw |= 1;
  ncclGinProxyGfd_t* slot = Slot(0, 0, 0);
  __atomic_store_n(&slot->qword[0].raw, g.qword[0].raw, __ATOMIC_RELEASE);
  std::thread producer([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    for (int k = ncclGinProxyGfdQwords - 1; k > 0; k--) {
      __atomic_store_n(&slot->qword[k].raw, g.qword[k].raw, __ATOMIC_RELEASE);
    }
  });
  EXPECT_EQ(ncclSuccess, Progress());
  producer.join();
  ASSERT_EQ(1u, rma_.ops.size());
  EXPECT_EQ(0x20u, rma_.ops[0].dstOff);
  EXPECT_EQ(reinterpret_cast<void*>(0x3000), rma_.ops[0].dstHandle);
}

TEST_F(GinHostProxyProgressMicrotest, EmptyQueuesSubmitNothing) {
  ASSERT_EQ(ncclSuccess, Progress());
  EXPECT_TRUE(rma_.ops.empty());
  EXPECT_TRUE(rma_.tested.empty());
  EXPECT_EQ(1u, rma_.progressed.size());
}

TEST_F(GinHostProxyProgressMicrotest, PutWithSignal_TargetsThisContextsSignalSlot) {
  const uint64_t value = 0x123456789abcdef0ull;
  ncclGinProxyGfd_t inc = PutGfd(ncclGinProxyOpWithSignalInc, 8);
  SetSignal(inc, /*signalId=*/2, value, /*strong=*/true);
  ncclGinProxyGfd_t add = PutGfd(ncclGinProxyOpWithSignalAdd, 8);
  SetSignal(add, /*signalId=*/3, 7, /*strong=*/false);
  Post(1, 0, inc);
  Post(1, 0, add);
  ASSERT_EQ(ncclSuccess, Progress());
  ASSERT_EQ(2u, rma_.ops.size());
  EXPECT_EQ('s', rma_.ops[0].kind);
  EXPECT_EQ((2 + 1 * kSignals) * sizeof(uint64_t), rma_.ops[0].signalOff);
  EXPECT_EQ(Ctx()->signalsGinHandle, rma_.ops[0].signalHandle);
  EXPECT_EQ(value, rma_.ops[0].signalValue);
  EXPECT_EQ((uint32_t)NCCL_NET_SIGNAL_OP_INC, rma_.ops[0].signalOp);
  EXPECT_TRUE(rma_.ops[0].isStrong);
  EXPECT_EQ(0x10u, rma_.ops[0].srcOff);
  EXPECT_EQ(0x20u, rma_.ops[0].dstOff);
  EXPECT_EQ((3 + 1 * kSignals) * sizeof(uint64_t), rma_.ops[1].signalOff);
  EXPECT_EQ(7u, rma_.ops[1].signalValue);
  EXPECT_EQ((uint32_t)NCCL_NET_SIGNAL_OP_ADD, rma_.ops[1].signalOp);
  EXPECT_FALSE(rma_.ops[1].isStrong);
}

TEST_F(GinHostProxyProgressMicrotest, PutWithSignal_VersionOneGfdsAreAlwaysStrong) {
  ncclGinProxyGfd_t g = PutGfd(ncclGinProxyOpWithSignalInc, 8);
  SetSignal(g, 0, 1, /*strong=*/false);
  g.qword[ncclGinProxyGfdHeader].header.version = 1;
  Post(0, 0, g);
  ASSERT_EQ(ncclSuccess, Progress());
  ASSERT_EQ(1u, rma_.ops.size());
  EXPECT_TRUE(rma_.ops[0].isStrong);
}

TEST_F(GinHostProxyProgressMicrotest, InlinePut_SendsTheReassembledValueFromTheInlinesMr) {
  const struct {
    uint64_t size, expected;
  } cases[] = {{4, 0x89abcdefull}, {6, 0x456789abcdefull}, {8, 0x0123456789abcdefull}};
  for (const auto& c : cases) {
    ncclGinProxyGfd_t g = PutGfd(ncclGinProxyOpWithInline, c.size);
    g.qword[ncclGinProxyGfdInlineLow].inlineLow.inlineValLow = 0x89abcdef;
    g.qword[ncclGinProxyGfdInlineLow].inlineLow.inlineValLow2 = 0x4567;
    g.qword[ncclGinProxyGfdInlineHigh].inlineHigh.inlineValHigh = 0x0123;
    Post(0, 1, g);
  }
  ASSERT_EQ(ncclSuccess, Progress());
  ASSERT_EQ(3u, rma_.ops.size());
  ginProxyHostGpuCtx& h = Ctx()->hostGpuCtx[0];
  for (int i = 0; i < 3; i++) {
    SCOPED_TRACE(i);
    const size_t stateIdx = 1 * h.queueSize + i;  // rank 1, slot i
    EXPECT_EQ('p', rma_.ops[i].kind);
    EXPECT_EQ(stateIdx * sizeof(uint64_t), rma_.ops[i].srcOff);
    EXPECT_EQ(h.inlinesMhandle, rma_.ops[i].srcHandle);
    EXPECT_EQ(cases[i].size, rma_.ops[i].size);
    EXPECT_EQ(cases[i].expected, h.inlines[stateIdx]);
  }
}

TEST_F(GinHostProxyProgressMicrotest, VaSignal_SignalsTheGfdsAddressWithoutAPut) {
  ncclGinProxyGfd_t g = Gfd(ncclGinProxyOpVASignal | ncclGinProxyOpWithSignalAdd);
  g.qword[ncclGinProxyGfdVASignalOff].vaSignalOff.vaSignalOff = 0x40;
  g.qword[ncclGinProxyGfdVASignalHandle].vaSignalHandle.vaSignalHandle = 0x7000;
  SetSignal(g, 0, 0xabcdef012345ull, /*strong=*/true);
  Post(1, 1, g);
  ASSERT_EQ(ncclSuccess, Progress());
  ASSERT_EQ(1u, rma_.ops.size());
  const RmaOp& op = rma_.ops[0];
  EXPECT_EQ('s', op.kind);
  EXPECT_EQ(1, op.context);
  EXPECT_EQ(0u, op.srcOff);
  EXPECT_EQ(nullptr, op.srcHandle);
  EXPECT_EQ(0u, op.size);
  EXPECT_EQ(nullptr, op.dstHandle);
  EXPECT_EQ(1u, op.rank);
  EXPECT_EQ(0x40u, op.signalOff);
  EXPECT_EQ(reinterpret_cast<void*>(0x7000), op.signalHandle);
  EXPECT_EQ(0xabcdef012345ull, op.signalValue);
  EXPECT_EQ((uint32_t)NCCL_NET_SIGNAL_OP_ADD, op.signalOp);
  EXPECT_TRUE(op.isStrong);
}

TEST_F(GinHostProxyProgressMicrotest, Get_SubmitsAnIget) {
  ncclGinProxyGfd_t g = PutGfd(0, 128);
  g.qword[ncclGinProxyGfdHeaderExt].headerExt.op = ncclGinProxyOpGet;
  Post(0, 1, g);
  ASSERT_EQ(ncclSuccess, Progress());
  ASSERT_EQ(1u, rma_.ops.size());
  EXPECT_EQ('g', rma_.ops[0].kind);
  EXPECT_EQ(0x10u, rma_.ops[0].srcOff);
  EXPECT_EQ(reinterpret_cast<void*>(0x2000), rma_.ops[0].srcHandle);
  EXPECT_EQ(128u, rma_.ops[0].size);
  EXPECT_EQ(0x20u, rma_.ops[0].dstOff);
  EXPECT_EQ(reinterpret_cast<void*>(0x3000), rma_.ops[0].dstHandle);
  EXPECT_EQ(1u, rma_.ops[0].rank);
}

TEST_F(GinHostProxyProgressMicrotest, Get_PluginWithoutIgetIsAnError) {
  rma_.vtable.iget = nullptr;
  Post(0, 0, Gfd(ncclGinProxyOpGet, 8));
  EXPECT_EQ(ncclInvalidUsage, Progress());
  EXPECT_TRUE(Logged("does not support GET"));
  EXPECT_TRUE(HasError());
}

TEST_F(GinHostProxyProgressMicrotest, Flush_SubmitsAnIflushOnTheSignalsMr) {
  Post(1, 0, Gfd(ncclGinProxyOpFlush));
  ASSERT_EQ(ncclSuccess, Progress());
  ASSERT_EQ(1u, rma_.ops.size());
  EXPECT_EQ('f', rma_.ops[0].kind);
  EXPECT_EQ(1, rma_.ops[0].context);
  EXPECT_EQ(Ctx()->signalsGinHandle, rma_.ops[0].dstHandle);
  EXPECT_EQ(0u, rma_.ops[0].rank);
  EXPECT_EQ(0u, Ci(1, 0));
}

TEST_F(GinHostProxyProgressMicrotest, Flush_WithoutARequestCompletesWithoutATest) {
  rma_.flushReturnsNullRequest = true;
  Post(1, 0, Gfd(ncclGinProxyOpFlush));
  ASSERT_EQ(ncclSuccess, Progress());
  ASSERT_EQ(ncclSuccess, Progress());
  EXPECT_TRUE(rma_.tested.empty());
  EXPECT_EQ(1u, Ci(1, 0));
}

TEST_F(GinHostProxyProgressMicrotest, Flush_PluginWithoutIflushIsAnError) {
  rma_.vtable.iflush = nullptr;
  Post(0, 0, Gfd(ncclGinProxyOpFlush));
  EXPECT_EQ(ncclInvalidUsage, Progress());
  EXPECT_TRUE(Logged("does not support FLUSH"));
}

TEST_F(GinHostProxyProgressMicrotest, UnknownOpIsAnInternalError) {
  Post(0, 0, Gfd(ncclGinProxyOpWithCounter, 8));
  EXPECT_EQ(ncclInternalError, Progress());
  EXPECT_TRUE(Logged("Unexpected GIN proxy operation"));
  EXPECT_TRUE(HasError());
}

TEST_F(GinHostProxyProgressMicrotest, SubmitFailureSetsTheErrorFlag) {
  rma_.opResult = ncclRemoteError;
  Post(0, 0, PutGfd(0, 8));
  EXPECT_EQ(ncclRemoteError, Progress());
  EXPECT_TRUE(HasError());
  EXPECT_TRUE(rma_.queried.empty()) << "a local error answers without asking the plugin";
}

TEST_F(GinHostProxyProgressMicrotest, Completion_AdvancesTheCiAndBumpsTheGfdsCounter) {
  ncclGinProxyGfd_t g = PutGfd(ncclGinProxyOpWithCounter, 8);
  g.qword[ncclGinProxyGfdCompletion].completion.counterId = 3;
  Post(1, 1, g);
  ASSERT_EQ(ncclSuccess, Progress());
  ASSERT_EQ(1u, rma_.ops.size());
  void* request = rma_.ops[0].request;

  ASSERT_EQ(ncclSuccess, Progress());  // still in flight
  EXPECT_EQ(std::vector<void*>{request}, rma_.tested);
  EXPECT_EQ(0u, Ci(1, 1));

  rma_.done.insert(request);
  ASSERT_EQ(ncclSuccess, Progress());
  EXPECT_EQ(1u, Ci(1, 1));
  EXPECT_EQ(1u, Ctx()->hostGpuCtx[1].cisShadow[1]);
  EXPECT_EQ(1u, Ctx()->counters[1 * kCounters + 3]);
  for (int i = 0; i < 2 * kCounters; i++) {
    if (i != 1 * kCounters + 3) EXPECT_EQ(0u, Ctx()->counters[i]) << i;
  }

  ASSERT_EQ(ncclSuccess, Progress());
  EXPECT_EQ(2u, rma_.tested.size()) << "a consumed GFD is not tested again";
}

TEST_F(GinHostProxyProgressMicrotest, Completion_OutOfOrderHoldsTheCiUntilTheHoleCloses) {
  Post(0, 1, PutGfd(0, 8));
  Post(0, 1, PutGfd(ncclGinProxyOpWithCounter, 8));
  ASSERT_EQ(ncclSuccess, Progress());
  ASSERT_EQ(2u, rma_.ops.size());

  rma_.done.insert(rma_.ops[1].request);
  ASSERT_EQ(ncclSuccess, Progress());
  EXPECT_EQ(0u, Ci(0, 1));
  EXPECT_EQ(1u, Ctx()->counters[0]) << "the counter moves as soon as its GFD completes";

  rma_.tested.clear();
  rma_.done.insert(rma_.ops[0].request);
  ASSERT_EQ(ncclSuccess, Progress());
  EXPECT_EQ(std::vector<void*>{rma_.ops[0].request}, rma_.tested);
  EXPECT_EQ(2u, Ci(0, 1));
  EXPECT_EQ(1u, Ctx()->counters[0]) << "the counterless GFD leaves it alone";
}

TEST_F(GinHostProxyProgressMicrotest, Completion_TestFailureSetsTheErrorFlag) {
  Post(0, 0, PutGfd(0, 8));
  ASSERT_EQ(ncclSuccess, Progress());
  rma_.testResult = ncclSystemError;
  EXPECT_EQ(ncclSystemError, Progress());
  EXPECT_TRUE(HasError());
  EXPECT_EQ(1u, rma_.progressed.size()) << "plugin progress is skipped after a failed test";
}

TEST_F(GinHostProxyProgressMicrotest, QueueIndicesWrapAroundTheQueue) {
  for (int i = 0; i < 6; i++) {
    Post(0, 0, PutGfd(0, 100 + i));
    ASSERT_EQ(ncclSuccess, Progress());
    rma_.done.insert(rma_.ops.back().request);
    ASSERT_EQ(ncclSuccess, Progress());
  }
  ASSERT_EQ(6u, rma_.ops.size());
  for (int i = 0; i < 6; i++) EXPECT_EQ(100u + i, rma_.ops[i].size);
  EXPECT_EQ(6u, Ci(0, 0));
  EXPECT_EQ(6u, Ctx()->hostGpuCtx[0].sis[0]);
}

// Requests are aggregated only while the peer's next GFD is already queued and the batch isn't over.
TEST_F(GinHostProxyProgressMicrotest, PollBatch_AggregatesAllButTheLastOfABatch) {
  for (int i = 0; i < 3; i++) Post(0, 0, PutGfd(0, 8));
  ASSERT_EQ(ncclSuccess, Progress());
  ASSERT_EQ(3u, rma_.ops.size());
  EXPECT_EQ((uint32_t)ncclRmaOptFlagsAggregateRequests, rma_.ops[0].optFlags);
  EXPECT_EQ((uint32_t)ncclRmaOptFlagsAggregateRequests, rma_.ops[1].optFlags);
  EXPECT_EQ((uint32_t)ncclRmaOptFlagsDefault, rma_.ops[2].optFlags);
}

TEST_F(GinHostProxyProgressMicrotest, PollBatch_BoundsGfdsPerPeerPerCall) {
  Ctx()->pollBatch = 2;  // the fixture's context is already created; Create_PollBatchIsAtLeastOne covers the param
  for (int i = 0; i < 3; i++) Post(0, 0, PutGfd(0, 8));
  Post(0, 1, PutGfd(0, 8));
  ASSERT_EQ(ncclSuccess, Progress());
  ASSERT_EQ(3u, rma_.ops.size());
  EXPECT_EQ(0u, rma_.ops[0].rank);
  EXPECT_EQ((uint32_t)ncclRmaOptFlagsAggregateRequests, rma_.ops[0].optFlags);
  EXPECT_EQ(0u, rma_.ops[1].rank);
  EXPECT_EQ((uint32_t)ncclRmaOptFlagsDefault, rma_.ops[1].optFlags) << "last of the batch";
  EXPECT_EQ(1u, rma_.ops[2].rank);
  ASSERT_EQ(ncclSuccess, Progress());
  ASSERT_EQ(4u, rma_.ops.size());
  EXPECT_EQ(0u, rma_.ops[3].rank);
}

TEST_F(GinHostProxyProgressMicrotest, Progress_PluginWithoutRmaProgressIsSkipped) {
  rma_.vtable.rmaProgress = nullptr;
  EXPECT_EQ(ncclSuccess, Progress());
  EXPECT_TRUE(rma_.progressed.empty());
}

TEST_F(GinHostProxyProgressMicrotest, QueryLastError_AsksThePluginWhenNoLocalError) {
  EXPECT_FALSE(HasError());
  EXPECT_EQ(std::vector<void*>{kPluginRmaCtx}, rma_.queried);
  rma_.pluginHasError = true;
  EXPECT_TRUE(HasError());
  rma_.queryLastErrorResult = ncclSystemError;
  bool hasError = false;
  EXPECT_EQ(ncclSystemError, ncclGinProxy.queryLastError(ginCtx_, &hasError));
  rma_.vtable.queryLastError = nullptr;
  EXPECT_FALSE(HasError());
}

TEST(GinHostProxyGpuCtxMicrotest, Init_WritesEachVersionsLayoutAtItsIndex) {
  alignas(16) char array[3 * NCCL_GIN_PROXY_GPU_CONTEXT_MAX_SIZE];
  auto* queues = reinterpret_cast<ncclGinProxyGfd_t*>(0x100);
  auto* pis = reinterpret_cast<uint32_t*>(0x200);
  auto* cis = reinterpret_cast<uint32_t*>(0x300);
  auto* counters = reinterpret_cast<uint64_t*>(0x400);
  auto* signals = reinterpret_cast<uint64_t*>(0x500);
  auto* offsets = reinterpret_cast<uint64_t*>(0x600);
  auto* issued = reinterpret_cast<uint32_t*>(0x700);
  auto* visible = reinterpret_cast<uint32_t*>(0x800);
  auto init = [&](int version) {
    std::memset(array, 0xee, sizeof(array));
    return ncclGinProxyGpuCtx_init(version, array, 1, 8, 64, queues, pis, cis, counters, signals, offsets, issued,
                                   visible, true);
  };

  ASSERT_EQ(ncclSuccess, init(1));
  auto* v1 = reinterpret_cast<ncclGinProxyGpuCtx_v1_t*>(array);
  EXPECT_EQ(8, v1[1].nranks);
  EXPECT_EQ(64u, v1[1].queueSize);
  EXPECT_EQ(queues, v1[1].queues);
  EXPECT_EQ(pis, v1[1].pis);
  EXPECT_EQ(cis, v1[1].cis);
  EXPECT_EQ(counters, v1[1].counters);
  EXPECT_EQ(signals, v1[1].signals);
  EXPECT_EQ(0xeeeeeeee, (uint32_t)v1[0].nranks);
  EXPECT_EQ(0xeeeeeeee, (uint32_t)v1[2].nranks);

  ASSERT_EQ(ncclSuccess, init(2));
  auto* v2 = reinterpret_cast<ncclGinProxyGpuCtx_v2_t*>(array);
  EXPECT_EQ(8, v2[1].nranks);
  EXPECT_EQ(64u, v2[1].queueSize);
  EXPECT_EQ(queues, v2[1].queues);
  EXPECT_EQ(pis, v2[1].pis);
  EXPECT_EQ(cis, v2[1].cis);
  EXPECT_EQ(counters, v2[1].counters);
  EXPECT_EQ(signals, v2[1].signals);
  EXPECT_EQ(offsets, v2[1].signalOffsets);
  EXPECT_EQ(issued, v2[1].lastIssuedGet);
  EXPECT_EQ(visible, v2[1].lastVisibleGet);
  EXPECT_EQ(0xeeeeeeee, (uint32_t)v2[0].nranks);

  ASSERT_EQ(ncclSuccess, init(NCCL_GIN_PROXY_GPU_CONTEXT_VERSION));
  auto* v3 = reinterpret_cast<ncclGinProxyGpuCtx_t*>(array);
  EXPECT_EQ(8, v3[1].nranks);
  EXPECT_EQ(64u, v3[1].queueSize);
  EXPECT_EQ(queues, v3[1].queues);
  EXPECT_EQ(pis, v3[1].pis);
  EXPECT_EQ(cis, v3[1].cis);
  EXPECT_EQ(counters, v3[1].counters);
  EXPECT_EQ(signals, v3[1].signals);
  EXPECT_EQ(offsets, v3[1].signalOffsets);
  EXPECT_EQ(issued, v3[1].lastIssuedGet);
  EXPECT_EQ(visible, v3[1].lastVisibleGet);
  EXPECT_TRUE(v3[1].flushesAllPutsOnAnySignal);
  EXPECT_EQ(0xeeeeeeee, (uint32_t)v3[0].nranks);
}

TEST(GinHostProxyGpuCtxMicrotest, Init_RejectsUnknownVersionsWithoutWriting) {
  char array[NCCL_GIN_PROXY_GPU_CONTEXT_MAX_SIZE];
  for (int version : {0, NCCL_GIN_PROXY_GPU_CONTEXT_VERSION + 1}) {
    std::memset(array, 0xee, sizeof(array));
    EXPECT_EQ(ncclInternalError, ncclGinProxyGpuCtx_init(version, array, 0, 8, 64, nullptr, nullptr, nullptr, nullptr,
                                                         nullptr, nullptr, nullptr, nullptr, false));
    for (char c : array) ASSERT_EQ((char)0xee, c);
  }
}

}  // namespace
