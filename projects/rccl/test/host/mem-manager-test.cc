/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/mem_manager.cc: entry teardown, Suspend/Resume and the peer handle
// exchange, and the public Suspend/Resume/MemStats entry points. Bootstrap, proxy and group calls
// are seamed in this TU; the cuMem calls go through hip_fakes.

#include <gtest/gtest.h>

#include <fcntl.h>
#include <unistd.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "comm.h"
#include "debug.h"
#include "group.h"
#include "fakes/fail_loud.h"
#include "fakes/hip_fakes.h"

uint32_t ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED;
uint64_t ncclDebugMask = ~uint64_t(0);
thread_local int ncclDebugNoWarn = 0;
std::function<int64_t(const char*, int64_t)> g_loadParam;

namespace {
std::vector<std::string> g_warns;
}  // namespace

void ncclDebugLog(ncclDebugLogLevel level, unsigned long, const char*, int, const char* fmt, ...) {
  if (level != NCCL_LOG_WARN) return;
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  g_warns.push_back(buf);
}

thread_local int ncclGroupDepth = 0;
thread_local ncclResult_t ncclGroupError = ncclSuccess;
thread_local struct ncclComm* ncclGroupCommHead[ncclGroupTaskTypeNum] = {};
thread_local int ncclGroupBlocking = -1;

std::function<ncclResult_t(int, int)> g_barrier;
std::function<ncclResult_t(void*, int)> g_allGather;
std::function<ncclResult_t(int, int, void*, int)> g_send;
std::function<ncclResult_t(int, int, void*, int)> g_recv;
std::function<ncclResult_t(int, void*, ncclIpcFd*)> g_getFd;
std::function<ncclResult_t(ncclComm*)> g_commCheck;
std::function<ncclResult_t()> g_groupEnd;
std::function<ncclResult_t(ncclAsyncJob*, ncclResult_t (*)(ncclAsyncJob*), void (*)(void*))> g_mgmtEnqueue;
std::vector<int> g_closedFds;
int64_t g_rearch = 0;
ncclResult_t g_asyncError = ncclSuccess;
ncclResult_t g_asyncErrorResult = ncclSuccess;

// Counts deletes of one watched pointer and keeps its memory, so a double delete is observable.
// new is replaced too so the pair stays malloc/free, which ASan expects.
void* g_watchedDelete = nullptr;
int g_watchedDeletes = 0;
void* operator new(size_t n) {
  void* p = std::malloc(n ? n : 1);
  if (p == nullptr) throw std::bad_alloc();
  return p;
}
void operator delete(void* p) noexcept {
  if (p != nullptr && p == g_watchedDelete) {
    g_watchedDeletes++;
    return;
  }
  std::free(p);
}
void operator delete(void* p, size_t) noexcept {
  operator delete(p);
}
// mem_manager.cc allocates its jobs with new (std::nothrow), so the nothrow and array forms are replaced too.
void* operator new[](size_t n) { return ::operator new(n); }
void* operator new(size_t n, const std::nothrow_t&) noexcept { return std::malloc(n ? n : 1); }
void* operator new[](size_t n, const std::nothrow_t&) noexcept { return std::malloc(n ? n : 1); }
void operator delete[](void* p) noexcept { operator delete(p); }
void operator delete[](void* p, size_t) noexcept { operator delete(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { operator delete(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { operator delete(p); }

ncclResult_t bootstrapBarrier(void*, int, int, int tag) { return g_barrier(tag, 0); }
ncclResult_t bootstrapAllGather(void*, void* data, int size) { return g_allGather(data, size); }
ncclResult_t bootstrapSend(void*, int peer, int tag, void* data, int size) { return g_send(peer, tag, data, size); }
ncclResult_t bootstrapRecv(void*, int peer, int tag, void* data, int size) { return g_recv(peer, tag, data, size); }
ncclResult_t ncclProxyClientGetFdBlocking(struct ncclComm*, int rank, void* handle, ncclIpcFd* fd) {
  return g_getFd(rank, handle, fd);
}
int ncclIpcFdClose(ncclIpcFd fd) {
  g_closedFds.push_back(fd);
  return 0;
}
ncclResult_t CommCheck(struct ncclComm* comm, const char*, const char*) { return g_commCheck(comm); }
ncclResult_t ncclCommEnsureReady(ncclComm_t) { return ncclSuccess; }
ncclResult_t ncclGroupStartInternal() {
  ncclGroupDepth++;
  return ncclSuccess;
}
ncclResult_t ncclGroupEndInternal(ncclSimInfo_t*) {
  ncclGroupDepth--;
  ncclResult_t ret = g_groupEnd();
  ncclGroupError = ncclSuccess;
  return ret;
}
ncclResult_t ncclCommGetAsyncError(ncclComm_t, ncclResult_t* asyncError) {
  *asyncError = g_asyncError;
  return g_asyncErrorResult;
}
int64_t ncclParamEnqueueRearchEnable() { return g_rearch; }
// Only ncclGroupCommJoin's collective arm reaches this; Suspend/Resume join as SymRegister.
void* ncclMemoryStack::allocateSpilled(ncclMemoryStack*, size_t, size_t) {
  FailLoudUnfaked("mem-manager-test", "ncclMemoryStack::allocateSpilled");
}
ncclResult_t ncclMgmtTaskEnqueue(struct ncclAsyncJob* job, ncclResult_t (*func)(struct ncclAsyncJob*),
                                 void (*destructor)(void*), ncclComm_t) {
  return g_mgmtEnqueue(job, func, destructor);
}

int g_mallocFailAt = 0;
int g_mallocCalls = 0;
constexpr unsigned char kMallocPoison = 0xee;
// Fills fresh blocks with kMallocPoison so fields the UUT leaves unset read the same every run.
void* MemManagerMalloc(size_t n) {
  if (++g_mallocCalls == g_mallocFailAt) return nullptr;
  void* p = std::malloc(n);
  if (p != nullptr) std::memset(p, kMallocPoison, n);
  return p;
}

#include "fakes/param_redirect.h"
#include "fakes/nvtx_redirect.h"
#define malloc(n) MemManagerMalloc(n)
#include MEM_MANAGER_CC_PATH
#undef malloc

// hipify renames the CUDA FABRIC names wherever HIP has them; p2p.h only defines them otherwise.
#ifdef CU_MEM_HANDLE_TYPE_FABRIC
const hipMemAllocationHandleType kFabricHandle = CU_MEM_HANDLE_TYPE_FABRIC;
#else
const hipMemAllocationHandleType kFabricHandle = hipMemHandleTypeFabric;
#endif

namespace {

constexpr int kDev = 2;
constexpr size_t kSize = 1 << 16;
#if defined(HIP_VMM_UNCACHED_MEMORY)
constexpr hipMemAllocationType kVmmType = hipMemAllocationTypeUncached;
#else
constexpr hipMemAllocationType kVmmType = hipMemAllocationTypePinned;
#endif

hipMemGenericAllocationHandle_t Handle(uintptr_t v) {
  return reinterpret_cast<hipMemGenericAllocationHandle_t>(v);
}

void* Addr(uintptr_t v) {
  return reinterpret_cast<void*>(v);
}

bool FdOpen(int fd) {
  return fcntl(fd, F_GETFD) != -1;
}

struct MapCall {
  void* ptr;
  size_t size;
  hipMemGenericAllocationHandle_t handle;
};

class MemManagerMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetHipFakes();
    g_warns.clear();
    g_closedFds.clear();
    g_rearch = 0;
    g_asyncError = ncclSuccess;
    g_asyncErrorResult = ncclSuccess;
    g_mallocFailAt = 0;
    g_mallocCalls = 0;
    params_.clear();
    g_loadParam = [this](const char* env, int64_t deftVal) {
      auto it = params_.find(env);
      return it == params_.end() ? deftVal : it->second;
    };
    g_barrier = [this](int tag, int) {
      barriers_.push_back(tag);
      return barrierResult_.count(tag) ? barrierResult_[tag] : ncclSuccess;
    };
    g_allGather = [this](void* data, int size) {
      EXPECT_EQ(size, (int)sizeof(int));
      int* counts = static_cast<int*>(data);
      for (int r = 0; r < comm_->nRanks; r++)
        if (r != comm_->rank) counts[r] = (int)inbox_[r].size();
      return allGatherResult_;
    };
    g_send = [this](int peer, int tag, void* data, int size) {
      EXPECT_EQ(tag, 0xFEED);
      auto* infos = static_cast<ncclDynMemP2pHandleInfo*>(data);
      sent_[peer].assign(infos, infos + size / sizeof(ncclDynMemP2pHandleInfo));
      return sendResult_;
    };
    g_recv = [this](int peer, int tag, void* data, int size) {
      EXPECT_EQ(tag, 0xFEED);
      EXPECT_EQ((size_t)size, inbox_[peer].size() * sizeof(ncclDynMemP2pHandleInfo));
      std::memcpy(data, inbox_[peer].data(), size);
      recvPeers_.push_back(peer);
      return recvResult_;
    };
    g_getFd = [this](int rank, void* handle, ncclIpcFd* fd) {
      fdRequests_.push_back({rank, *static_cast<uint64_t*>(handle)});
      *fd = fd_;
      return getFdResult_;
    };
    g_commCheck = [](ncclComm*) { return ncclSuccess; };
    g_groupEnd = [this] {
      groupErrors_.push_back(ncclGroupError);
      return ncclSuccess;
    };
    g_mgmtEnqueue = [this](ncclAsyncJob* job, ncclResult_t (*func)(ncclAsyncJob*), void (*destructor)(void*)) {
      jobs_.push_back({job, func, destructor});
      return ncclSuccess;
    };

    g_hipDeviceSynchronize = [] { return hipSuccess; };
    g_hipThreadExchangeStreamCaptureMode = [](hipStreamCaptureMode*) { return hipSuccess; };
    g_hipMemUnmap = [this](void* ptr, size_t size) {
      unmaps_.push_back({ptr, size, nullptr});
      return unmapFail_.count(ptr) ? hipErrorInvalidValue : hipSuccess;
    };
    g_hipMemRelease = [this](hipMemGenericAllocationHandle_t h) {
      releases_.push_back(h);
      return releaseFail_.count(h) ? hipErrorInvalidValue : hipSuccess;
    };
    g_hipMemCreate = [this](hipMemGenericAllocationHandle_t* h, size_t size, const hipMemAllocationProp* prop,
                            unsigned long long) {
      creates_.push_back({*prop, size});
      *h = Handle(nextHandle_++);
      return createResult_;
    };
    g_hipMemMap = [this](void* ptr, size_t size, size_t, hipMemGenericAllocationHandle_t h, unsigned long long) {
      maps_.push_back({ptr, size, h});
      return mapResult_;
    };
    g_hipMemSetAccess = [this](void*, size_t, const hipMemAccessDesc* desc, size_t count) {
      EXPECT_EQ(count, 1u);
      accessDevs_.push_back(desc->location.id);
      return hipSuccess;
    };
    g_hipMemcpy = [this](void* dst, const void* src, size_t n, hipMemcpyKind kind) {
      copies_.push_back(kind);
      if (copyResult_ != hipSuccess) return copyResult_;
      std::memcpy(dst, src, n);
      return hipSuccess;
    };
    g_hipMemAddressFree = [this](void* ptr, size_t size) {
      addressFrees_.push_back({ptr, size, nullptr});
      return hipSuccess;
    };
    g_hipHostFree = [this](void* ptr) {
      hostFrees_.push_back(ptr);
      std::free(ptr);
      return hipSuccess;
    };
    g_hipMemImportFromShareableHandle = [this](hipMemGenericAllocationHandle_t* h, void* shareable,
                                               hipMemAllocationHandleType type) {
      imports_.push_back({shareable, type});
      *h = Handle(nextHandle_++);
      return importResult_;
    };
    g_hipMemExportToShareableHandle = [this](void* shareable, hipMemGenericAllocationHandle_t,
                                             hipMemAllocationHandleType type, unsigned long long) {
      EXPECT_EQ(type, kFabricHandle);
      std::memset(shareable, 0xab, sizeof(hipMemFabricHandle_compat_t));
      return exportResult_;
    };

    BuildComm(0, 1);
    ASSERT_EQ(ncclMemManagerInit(comm_.get()), ncclSuccess);
    mgr_ = comm_->memManager;
  }

  void TearDown() override {
    params_.clear();
    if (comm_->memManager != nullptr) {
      comm_->memManager->refCount = 1;
      ncclMemManagerDestroy(comm_.get());
    }
    ncclIntruQueueConstruct(&comm_->suspendTaskQueue);
    ncclIntruQueueConstruct(&comm_->resumeTaskQueue);
    for (auto* t : tasks_) free(t);
    ncclGroupCommHead[ncclGroupTaskTypeSymRegister] = nullptr;
    g_watchedDelete = nullptr;
    ResetHipFakes();
  }

  void BuildComm(int rank, int nRanks) {
    comm_ = std::make_unique<ncclComm>();
    peers_.assign(nRanks, ncclPeerInfo{});
    for (int r = 0; r < nRanks; r++) {
      peers_[r].cudaDev = 10 + r;
      peers_[r].hostHash = 0x40;
      peers_[r].pidHash = 0x50;
    }
    comm_->rank = rank;
    comm_->nRanks = nRanks;
    comm_->cudaDev = kDev;
    comm_->peerInfo = peers_.data();
    comm_->bootstrap = Addr(0xb007);
    comm_->config.blocking = 1;
    for (auto& next : comm_->groupNext) next = reinterpret_cast<ncclComm*>(NCCL_COMM_GROUP_INVALID);
  }

  // Rebuilds the comm as `rank` of `nRanks`, keeping the manager.
  void Regroup(int rank, int nRanks) {
    BuildComm(rank, nRanks);
    comm_->memManager = mgr_;
  }

  ncclDynMemEntry* Track(void* ptr, ncclMemType_t type, uintptr_t handle,
                         hipMemAllocationHandleType htype = hipMemHandleTypePosixFileDescriptor) {
    EXPECT_EQ(ncclMemTrack(mgr_, ptr, kSize, Handle(handle), htype, type), ncclSuccess);
    return mgr_->entries;
  }

  ncclDynMemEntry* Import(void* ptr, ncclMemType_t type, uintptr_t handle, int owner, void* ownerPtr) {
    EXPECT_EQ(ncclMemTrackImportFromPeer(mgr_, ptr, kSize, Handle(handle), hipMemHandleTypePosixFileDescriptor,
                                         type, owner, 10 + owner, ownerPtr),
              ncclSuccess);
    return mgr_->entries;
  }

  ncclDynMemP2pHandleInfo Info(int owner, void* ptr, int handleType, uint64_t handleData) {
    ncclDynMemP2pHandleInfo info{};
    info.ptr = ptr;
    info.ownerRank = owner;
    info.ownerDev = 10 + owner;
    info.size = kSize;
    info.handleType = handleType;
    info.handleData = handleData;
    return info;
  }

  // Moves the manager to the suspended state without driving Suspend.
  void MarkSuspended(std::initializer_list<ncclDynMemEntry*> entries) {
    for (auto* e : entries) {
      e->state = ncclDynMemStateReleased;
      e->handle = nullptr;
    }
    mgr_->released = 1;
  }

  void ClearCalls() {
    unmaps_.clear();
    releases_.clear();
    maps_.clear();
    creates_.clear();
    copies_.clear();
    barriers_.clear();
    accessDevs_.clear();
  }

  struct Job {
    ncclAsyncJob* job;
    ncclResult_t (*func)(ncclAsyncJob*);
    void (*destructor)(void*);
  };
  struct Create {
    hipMemAllocationProp prop;
    size_t size;
  };
  struct ImportCall {
    void* shareable;
    hipMemAllocationHandleType type;
  };

  std::unique_ptr<ncclComm> comm_;
  std::vector<ncclPeerInfo> peers_;
  ncclMemManager* mgr_ = nullptr;
  std::map<std::string, int64_t> params_;
  std::map<int, ncclResult_t> barrierResult_;
  std::vector<int> barriers_;
  std::map<int, std::vector<ncclDynMemP2pHandleInfo>> inbox_;
  std::map<int, std::vector<ncclDynMemP2pHandleInfo>> sent_;
  std::vector<int> recvPeers_;
  std::vector<std::pair<int, uint64_t>> fdRequests_;
  std::vector<Job> jobs_;
  std::vector<ncclMemManagerTask*> tasks_;
  std::vector<ncclResult_t> groupErrors_;
  ncclResult_t allGatherResult_ = ncclSuccess;
  ncclResult_t sendResult_ = ncclSuccess;
  ncclResult_t recvResult_ = ncclSuccess;
  ncclResult_t getFdResult_ = ncclSuccess;
  ncclIpcFd fd_ = 77;
  std::vector<MapCall> unmaps_, maps_, addressFrees_;
  std::vector<hipMemGenericAllocationHandle_t> releases_;
  std::vector<Create> creates_;
  std::vector<hipMemcpyKind> copies_;
  std::vector<int> accessDevs_;
  std::vector<void*> hostFrees_;
  std::vector<ImportCall> imports_;
  std::map<void*, bool> unmapFail_;
  std::map<hipMemGenericAllocationHandle_t, bool> releaseFail_;
  hipError_t createResult_ = hipSuccess;
  hipError_t mapResult_ = hipSuccess;
  hipError_t copyResult_ = hipSuccess;
  hipError_t importResult_ = hipSuccess;
  hipError_t exportResult_ = hipSuccess;
  uintptr_t nextHandle_ = 0x100;
};

// Disabled manager

TEST_F(MemManagerMicrotest, Disabled_EveryEntryPointIsANoOpOrRejected) {
  params_["DISABLE_MEM_MANAGER"] = 1;
  auto other = std::make_unique<ncclComm>();
  EXPECT_EQ(ncclMemManagerInit(other.get()), ncclSuccess);
  EXPECT_EQ(other->memManager, nullptr);
  EXPECT_EQ(ncclMemManagerDestroy(comm_.get()), ncclSuccess);
  EXPECT_EQ(comm_->memManager, mgr_);

  ncclMemUntrackInfo info{ncclMemOffload, ncclDynMemStateReleased, 5};
  EXPECT_EQ(ncclMemTrack(nullptr, nullptr, 1, nullptr, hipMemHandleTypePosixFileDescriptor, ncclMemScratch),
            ncclSuccess);
  EXPECT_EQ(ncclMemUntrackDynamic(nullptr, nullptr, &info), ncclSuccess);
  EXPECT_EQ(info.memType, ncclMemPersist);
  EXPECT_EQ(info.dynMemSize, 0u);
  EXPECT_EQ(ncclMemUntrackPersist(nullptr, nullptr, 1), ncclSuccess);
  EXPECT_EQ(ncclDynMemMarkExportToPeer(nullptr, nullptr, 1), ncclSuccess);
  EXPECT_EQ(ncclCommMemSuspend(comm_.get()), ncclInvalidUsage);
  EXPECT_EQ(ncclCommMemResume(comm_.get()), ncclInvalidUsage);
  EXPECT_EQ(mgr_->entries, nullptr);

  uint64_t value = 7;
  EXPECT_EQ(ncclCommMemStats_impl(comm_.get(), ncclStatGpuMemTotal, &value), ncclInvalidUsage);
  EXPECT_EQ(value, 7u);
  EXPECT_EQ(ncclCommSuspend_impl(comm_.get(), NCCL_SUSPEND_MEM), ncclInvalidUsage);
  EXPECT_EQ(ncclCommResume_impl(comm_.get()), ncclInvalidUsage);
  EXPECT_TRUE(jobs_.empty());
  EXPECT_EQ(ncclGroupDepth, 0);
}

// Destroy and Untrack teardown

TEST_F(MemManagerMicrotest, Destroy_TearsDownEveryEntryKind) {
  int fds[2];
  ASSERT_EQ(pipe(fds), 0);
  ncclDynMemEntry* local = Track(Addr(0x1000), ncclMemOffload, 0x11);
  local->cpuBackup = std::malloc(8);
  local->desc.local.shareableHandleValid = true;
  local->desc.local.shareableHandle.fd = fds[0];
  ASSERT_EQ(ncclDynMemMarkExportToPeer(mgr_, Addr(0x1000), 3), ncclSuccess);
  ncclDynMemEntry* suspended = Track(Addr(0x2000), ncclMemScratch, 0x12);
  suspended->state = ncclDynMemStateReleased;
  Import(Addr(0x3000), ncclMemScratch, 0x13, 1, Addr(0x9000));
  void* backup = local->cpuBackup;

  ASSERT_EQ(ncclMemManagerDestroy(comm_.get()), ncclSuccess);
  EXPECT_EQ(comm_->memManager, nullptr);
  EXPECT_EQ(hostFrees_, std::vector<void*>{backup});
  EXPECT_FALSE(FdOpen(fds[0]));
  ASSERT_EQ(addressFrees_.size(), 1u);
  EXPECT_EQ(addressFrees_[0].ptr, Addr(0x2000));
  EXPECT_EQ(addressFrees_[0].size, kSize);
  close(fds[1]);
}

// Only a valid POSIX fd on a local entry is closed; an fd field that is not a live handle is left alone.
TEST_F(MemManagerMicrotest, Destroy_LeavesFdsThatAreNotLiveLocalPosixHandles) {
  int fds[2];
  ASSERT_EQ(pipe(fds), 0);
  ncclDynMemEntry* notValid = Track(Addr(0x1000), ncclMemScratch, 0x11);
  notValid->desc.local.shareableHandle.fd = fds[0];
  ncclDynMemEntry* fabric = Track(Addr(0x2000), ncclMemScratch, 0x12, kFabricHandle);
  fabric->desc.local.shareableHandleValid = true;
  fabric->desc.local.shareableHandle.fd = fds[1];
  ncclDynMemEntry* negative = Track(Addr(0x3000), ncclMemScratch, 0x13);
  negative->desc.local.shareableHandleValid = true;

  ASSERT_EQ(ncclMemManagerDestroy(comm_.get()), ncclSuccess);
  EXPECT_TRUE(FdOpen(fds[0]));
  EXPECT_TRUE(FdOpen(fds[1]));
  EXPECT_TRUE(addressFrees_.empty());
  close(fds[0]);
  close(fds[1]);
}

TEST_F(MemManagerMicrotest, UntrackDynamic_FreesBackupAndClosesFd) {
  int fds[2];
  ASSERT_EQ(pipe(fds), 0);
  ncclDynMemEntry* e = Track(Addr(0x1000), ncclMemOffload, 0x11);
  e->cpuBackup = std::malloc(8);
  e->state = ncclDynMemStateReleased;
  e->desc.local.shareableHandleValid = true;
  e->desc.local.shareableHandle.fd = fds[0];
  void* backup = e->cpuBackup;
  mgr_->cpuBackupUsage = kSize + 3;

  ncclMemUntrackInfo info{};
  ASSERT_EQ(ncclMemUntrackDynamic(mgr_, Addr(0x1000), &info), ncclSuccess);
  EXPECT_EQ(info.memType, ncclMemOffload);
  EXPECT_EQ(info.dynMemState, ncclDynMemStateReleased);
  EXPECT_EQ(info.dynMemSize, kSize);
  EXPECT_EQ(mgr_->cpuBackupUsage, 3u);
  EXPECT_EQ(hostFrees_, std::vector<void*>{backup});
  EXPECT_FALSE(FdOpen(fds[0]));
  EXPECT_EQ(mgr_->totalOffload, 0u);
  close(fds[1]);
}

TEST_F(MemManagerMicrotest, UntrackDynamic_ImportedEntryDecrementsImportedCounters) {
  Import(Addr(0x1000), ncclMemScratch, 0x11, 1, Addr(0x9000));
  Import(Addr(0x2000), ncclMemOffload, 0x12, 1, Addr(0x9100));
  Import(Addr(0x3000), ncclMemPersist, 0x13, 1, Addr(0x9200));
  ASSERT_EQ(mgr_->totalScratchImported, kSize);
  ASSERT_EQ(mgr_->totalOffloadImported, kSize);

  ncclMemUntrackInfo info{};
  ASSERT_EQ(ncclMemUntrackDynamic(mgr_, Addr(0x1000), &info), ncclSuccess);
  EXPECT_EQ(info.memType, ncclMemScratch);
  ASSERT_EQ(ncclMemUntrackDynamic(mgr_, Addr(0x2000), nullptr), ncclSuccess);
  EXPECT_EQ(mgr_->totalScratchImported, 0u);
  EXPECT_EQ(mgr_->totalOffloadImported, 0u);
  EXPECT_EQ(mgr_->totalPersistImported, kSize);
  EXPECT_EQ(mgr_->numEntries, 0);
}

TEST_F(MemManagerMicrotest, Track_EntryAllocationFailure) {
  g_mallocFailAt = 1;
  EXPECT_EQ(ncclMemTrack(mgr_, Addr(0x1000), kSize, nullptr, hipMemHandleTypePosixFileDescriptor, ncclMemScratch),
            ncclSystemError);
  EXPECT_EQ(mgr_->numEntries, 0);
  EXPECT_EQ(mgr_->totalScratch, 0u);
}

TEST_F(MemManagerMicrotest, UntrackPersist_ZeroSizeOrUninitializedLeavesTheCounter) {
  ASSERT_EQ(ncclMemTrack(mgr_, Addr(0x1000), kSize, nullptr, hipMemHandleTypePosixFileDescriptor, ncclMemPersist),
            ncclSuccess);
  EXPECT_EQ(ncclMemUntrackPersist(mgr_, Addr(0x1000), 0), ncclSuccess);
  mgr_->initialized = 0;
  EXPECT_EQ(ncclMemUntrackPersist(mgr_, Addr(0x1000), kSize), ncclInternalError);
  mgr_->initialized = 1;
  EXPECT_EQ(mgr_->totalPersist, kSize);
}

// Suspend

TEST_F(MemManagerMicrotest, Suspend_RejectsMissingOrSuspendedManager) {
  EXPECT_EQ(ncclCommMemSuspend(nullptr), ncclInvalidArgument);
  comm_->memManager = nullptr;
  EXPECT_EQ(ncclCommMemSuspend(comm_.get()), ncclInvalidUsage);
  comm_->memManager = mgr_;
  mgr_->released = 1;
  EXPECT_EQ(ncclCommMemSuspend(comm_.get()), ncclInvalidUsage);
  EXPECT_TRUE(barriers_.empty());
}

TEST_F(MemManagerMicrotest, Suspend_FailsBeforeTouchingEntriesWhenSyncOrBarrierFails) {
  Track(Addr(0x1000), ncclMemScratch, 0x11);
  g_hipDeviceSynchronize = [] { return hipErrorInvalidValue; };
  EXPECT_EQ(ncclCommMemSuspend(comm_.get()), ncclUnhandledCudaError);
  EXPECT_TRUE(barriers_.empty());

  g_hipDeviceSynchronize = [] { return hipSuccess; };
  barrierResult_[0xBEEF] = ncclRemoteError;
  EXPECT_EQ(ncclCommMemSuspend(comm_.get()), ncclRemoteError);
  EXPECT_EQ(barriers_, std::vector<int>{0xBEEF});
  EXPECT_TRUE(unmaps_.empty());
  EXPECT_EQ(mgr_->released, 0);
}

TEST_F(MemManagerMicrotest, Suspend_ReleasesImportsThenLocalsAndBacksUpOffload) {
  std::vector<char> device(kSize, 'x');
  int fds[2];
  ASSERT_EQ(pipe(fds), 0);
  ncclDynMemEntry* scratch = Track(Addr(0x1000), ncclMemScratch, 0x11);
  scratch->desc.local.shareableHandleValid = true;
  scratch->desc.local.shareableHandle.fd = fds[0];
  ncclDynMemEntry* offload = Track(device.data(), ncclMemOffload, 0x12);
  ncclDynMemEntry* imported = Import(Addr(0x3000), ncclMemScratch, 0x13, 1, Addr(0x9000));
  ncclDynMemEntry* importedNoRef = Import(Addr(0x4000), ncclMemScratch, 0, 1, Addr(0x9100));

  ASSERT_EQ(ncclCommMemSuspend(comm_.get()), ncclSuccess);
  EXPECT_EQ(mgr_->released, 1);
  ASSERT_EQ(unmaps_.size(), 4u);
  EXPECT_EQ(unmaps_[0].ptr, Addr(0x4000));
  EXPECT_EQ(unmaps_[1].ptr, Addr(0x3000));
  EXPECT_EQ(unmaps_[2].ptr, (void*)device.data());
  EXPECT_EQ(unmaps_[3].ptr, Addr(0x1000));
  EXPECT_EQ(releases_, (std::vector<hipMemGenericAllocationHandle_t>{Handle(0x13), Handle(0x12), Handle(0x11)}));
  for (auto* e : {scratch, offload, imported, importedNoRef}) {
    EXPECT_EQ(e->state, ncclDynMemStateReleased);
    EXPECT_EQ(e->handle, nullptr);
  }
  ASSERT_NE(offload->cpuBackup, nullptr);
  EXPECT_EQ(static_cast<char*>(offload->cpuBackup)[kSize - 1], 'x');
  EXPECT_EQ(copies_, std::vector<hipMemcpyKind>{hipMemcpyDeviceToHost});
  EXPECT_EQ(mgr_->cpuBackupUsage, kSize);
  EXPECT_EQ(scratch->cpuBackup, nullptr);
  EXPECT_FALSE(FdOpen(fds[0]));
  EXPECT_FALSE(scratch->desc.local.shareableHandleValid);
  EXPECT_EQ(scratch->desc.local.shareableHandle.fd, -1);
  close(fds[1]);
}

TEST_F(MemManagerMicrotest, Suspend_UnmapFailureKeepsTheEntryActiveAndTheManagerUnsuspended) {
  ncclDynMemEntry* local = Track(Addr(0x1000), ncclMemScratch, 0x11);
  ncclDynMemEntry* imported = Import(Addr(0x3000), ncclMemScratch, 0x13, 1, Addr(0x9000));
  ncclDynMemEntry* other = Track(Addr(0x5000), ncclMemScratch, 0x15);
  unmapFail_[Addr(0x1000)] = true;
  unmapFail_[Addr(0x3000)] = true;

  EXPECT_EQ(ncclCommMemSuspend(comm_.get()), ncclUnhandledCudaError);
  EXPECT_EQ(mgr_->released, 0);
  EXPECT_EQ(local->state, ncclDynMemStateActive);
  EXPECT_EQ(local->handle, Handle(0x11));
  EXPECT_EQ(imported->state, ncclDynMemStateActive);
  EXPECT_EQ(imported->handle, Handle(0x13));
  EXPECT_EQ(other->state, ncclDynMemStateReleased);
  EXPECT_EQ(releases_, std::vector<hipMemGenericAllocationHandle_t>{Handle(0x15)});

  // A retry only revisits what is still Active.
  unmapFail_.clear();
  ClearCalls();
  ASSERT_EQ(ncclCommMemSuspend(comm_.get()), ncclSuccess);
  EXPECT_EQ(releases_, (std::vector<hipMemGenericAllocationHandle_t>{Handle(0x13), Handle(0x11)}));
  EXPECT_EQ(mgr_->released, 1);
}

TEST_F(MemManagerMicrotest, Suspend_ReleaseFailureStillMarksTheEntryReleased) {
  for (uintptr_t failing : {0x11, 0x13}) {
    SCOPED_TRACE(failing);
    ncclDynMemEntry* local = Track(Addr(0x1000), ncclMemScratch, 0x11);
    ncclDynMemEntry* imported = Import(Addr(0x3000), ncclMemScratch, 0x13, 1, Addr(0x9000));
    releaseFail_ = {{Handle(failing), true}};
    mgr_->released = 0;

    EXPECT_EQ(ncclCommMemSuspend(comm_.get()), ncclUnhandledCudaError);
    EXPECT_EQ(local->state, ncclDynMemStateReleased);
    EXPECT_EQ(local->handle, nullptr);
    EXPECT_EQ(imported->state, ncclDynMemStateReleased);
    EXPECT_EQ(imported->handle, nullptr);
    EXPECT_EQ(mgr_->released, 0);
    ASSERT_EQ(ncclMemUntrackDynamic(mgr_, Addr(0x1000), nullptr), ncclSuccess);
    ASSERT_EQ(ncclMemUntrackDynamic(mgr_, Addr(0x3000), nullptr), ncclSuccess);
  }
}

TEST_F(MemManagerMicrotest, Suspend_OffloadBackupFailureStopsAtThatEntry) {
  std::vector<char> device(kSize);
  ncclDynMemEntry* offload = Track(device.data(), ncclMemOffload, 0x12);
  ncclDynMemEntry* later = Track(Addr(0x5000), ncclMemOffload, 0x15);
  g_hipHostMalloc = [](void** ptr, size_t, unsigned) {
    *ptr = nullptr;
    return hipErrorOutOfMemory;
  };
  EXPECT_EQ(ncclCommMemSuspend(comm_.get()), ncclUnhandledCudaError);
  EXPECT_EQ(later->state, ncclDynMemStateActive);
  EXPECT_EQ(offload->state, ncclDynMemStateActive);
  EXPECT_TRUE(unmaps_.empty());
  EXPECT_EQ(mgr_->released, 0);
}

// A zero-byte offload entry gets no backup buffer, which Suspend treats as an allocation failure.
TEST_F(MemManagerMicrotest, Suspend_ZeroByteOffloadEntryFailsForWantOfABackup) {
  ASSERT_EQ(ncclMemTrack(mgr_, Addr(0x1000), 0, Handle(0x11), hipMemHandleTypePosixFileDescriptor,
                         ncclMemOffload),
            ncclSuccess);
  EXPECT_EQ(ncclCommMemSuspend(comm_.get()), ncclSystemError);
  EXPECT_EQ(mgr_->entries->state, ncclDynMemStateActive);
  EXPECT_TRUE(copies_.empty());
}

TEST_F(MemManagerMicrotest, Suspend_OffloadCopyFailureFreesTheBackup) {
  std::vector<char> device(kSize);
  ncclDynMemEntry* offload = Track(device.data(), ncclMemOffload, 0x12);
  copyResult_ = hipErrorInvalidValue;
  EXPECT_EQ(ncclCommMemSuspend(comm_.get()), ncclUnhandledCudaError);
  EXPECT_EQ(offload->cpuBackup, nullptr);
  EXPECT_EQ(hostFrees_.size(), 1u);
  EXPECT_EQ(mgr_->cpuBackupUsage, 0u);
  EXPECT_TRUE(unmaps_.empty());
}

// The retry backs the entry up again, leaking the first backup and counting it twice; a fix flips this pin.
TEST_F(MemManagerMicrotest, Suspend_KnownBug_RetryAfterOffloadUnmapFailureLeaksTheFirstBackup) {
  std::vector<char> device(kSize);
  ncclDynMemEntry* offload = Track(device.data(), ncclMemOffload, 0x12);
  unmapFail_[device.data()] = true;
  ASSERT_EQ(ncclCommMemSuspend(comm_.get()), ncclUnhandledCudaError);
  void* first = offload->cpuBackup;
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(mgr_->cpuBackupUsage, kSize);

  unmapFail_.clear();
  ASSERT_EQ(ncclCommMemSuspend(comm_.get()), ncclSuccess);
  EXPECT_NE(offload->cpuBackup, first);
  EXPECT_TRUE(hostFrees_.empty());
  EXPECT_EQ(mgr_->cpuBackupUsage, 2 * kSize);
  std::free(first);
}

// Resume

TEST_F(MemManagerMicrotest, Resume_RejectsMissingOrActiveManager) {
  EXPECT_EQ(ncclCommMemResume(nullptr), ncclInvalidArgument);
  comm_->memManager = nullptr;
  EXPECT_EQ(ncclCommMemResume(comm_.get()), ncclInvalidUsage);
  comm_->memManager = mgr_;
  EXPECT_EQ(ncclCommMemResume(comm_.get()), ncclInvalidUsage);
  EXPECT_TRUE(barriers_.empty());
}

TEST_F(MemManagerMicrotest, Resume_RecreatesLocalMemoryAndRestoresOffload) {
  std::vector<char> device(kSize, 0);
  ncclDynMemEntry* scratch = Track(Addr(0x1000), ncclMemScratch, 0x11);
  ncclDynMemEntry* offload = Track(device.data(), ncclMemOffload, 0x12);
  ncclDynMemEntry* active = Track(Addr(0x5000), ncclMemScratch, 0x15);
  MarkSuspended({scratch, offload});
  offload->cpuBackup = std::malloc(kSize);
  std::memset(offload->cpuBackup, 'y', kSize);
  void* backup = offload->cpuBackup;
  mgr_->cpuBackupUsage = kSize;
  scratch->cudaDev = 5;

  ASSERT_EQ(ncclCommMemResume(comm_.get()), ncclSuccess);
  EXPECT_EQ(mgr_->released, 0);
  ASSERT_EQ(creates_.size(), 2u);
  for (auto& c : creates_) {
    EXPECT_EQ(c.size, kSize);
    EXPECT_EQ(c.prop.type, kVmmType);
    EXPECT_EQ(c.prop.location.type, hipMemLocationTypeDevice);
    EXPECT_EQ(c.prop.requestedHandleType, hipMemHandleTypePosixFileDescriptor);
    EXPECT_EQ(c.prop.allocFlags.gpuDirectRDMACapable, 1);
  }
  EXPECT_EQ(creates_[0].prop.location.id, kDev);
  EXPECT_EQ(creates_[1].prop.location.id, 5);
  ASSERT_EQ(maps_.size(), 2u);
  EXPECT_EQ(maps_[0].ptr, (void*)device.data());
  EXPECT_EQ(maps_[0].handle, offload->handle);
  EXPECT_EQ(maps_[1].ptr, Addr(0x1000));
  EXPECT_EQ(maps_[1].handle, scratch->handle);
  EXPECT_EQ(accessDevs_, (std::vector<int>{kDev, 5}));
  EXPECT_EQ(scratch->state, ncclDynMemStateActive);
  EXPECT_EQ(offload->state, ncclDynMemStateActive);
  EXPECT_EQ(active->handle, Handle(0x15));
  EXPECT_EQ(device[kSize - 1], 'y');
  EXPECT_EQ(copies_, std::vector<hipMemcpyKind>{hipMemcpyHostToDevice});
  EXPECT_EQ(offload->cpuBackup, nullptr);
  EXPECT_EQ(hostFrees_, std::vector<void*>{backup});
  EXPECT_EQ(mgr_->cpuBackupUsage, 0u);
  EXPECT_EQ(barriers_, (std::vector<int>{0xBEEF, 0xCAFE}));
}

TEST_F(MemManagerMicrotest, Resume_SuspendThenResumeRoundTrip) {
  std::vector<char> device(kSize, 'z');
  ncclDynMemEntry* offload = Track(device.data(), ncclMemOffload, 0x12);
  ASSERT_EQ(ncclCommMemSuspend(comm_.get()), ncclSuccess);
  std::fill(device.begin(), device.end(), 0);
  ASSERT_EQ(ncclCommMemResume(comm_.get()), ncclSuccess);
  EXPECT_EQ(device[0], 'z');
  EXPECT_EQ(offload->state, ncclDynMemStateActive);
  EXPECT_EQ(mgr_->cpuBackupUsage, 0u);
}

TEST_F(MemManagerMicrotest, Resume_CreateFailureLeavesTheManagerSuspended) {
  ncclDynMemEntry* e = Track(Addr(0x1000), ncclMemScratch, 0x11);
  MarkSuspended({e});
  createResult_ = hipErrorOutOfMemory;
  EXPECT_EQ(ncclCommMemResume(comm_.get()), ncclUnhandledCudaError);
  EXPECT_EQ(mgr_->released, 1);
  EXPECT_EQ(e->state, ncclDynMemStateReleased);
  EXPECT_TRUE(maps_.empty());
  EXPECT_TRUE(barriers_.empty());
}

TEST_F(MemManagerMicrotest, Resume_MapFailureReleasesTheNewHandle) {
  ncclDynMemEntry* e = Track(Addr(0x1000), ncclMemScratch, 0x11);
  MarkSuspended({e});
  mapResult_ = hipErrorInvalidValue;
  EXPECT_EQ(ncclCommMemResume(comm_.get()), ncclUnhandledCudaError);
  EXPECT_EQ(releases_, std::vector<hipMemGenericAllocationHandle_t>{Handle(0x100)});
  EXPECT_EQ(e->handle, nullptr);
  EXPECT_EQ(e->state, ncclDynMemStateReleased);
}

TEST_F(MemManagerMicrotest, Resume_RestoresAccessOnlyForSameProcessExportedPeers) {
  Regroup(0, 4);
  peers_[2].pidHash = 0x51;
  peers_[3].hostHash = 0x41;
  peers_.push_back(peers_[1]);  // a same-process slot just past nRanks, which must not be read
  comm_->peerInfo = peers_.data();
  ncclDynMemEntry* e = Track(Addr(0x1000), ncclMemScratch, 0x11);
  for (int peer : {1, 2, 3, 4, -1}) ASSERT_EQ(ncclDynMemMarkExportToPeer(mgr_, Addr(0x1000), peer), ncclSuccess);
  MarkSuspended({e});
  for (int r = 1; r < 4; r++) inbox_[r] = {};

  ASSERT_EQ(ncclCommMemResume(comm_.get()), ncclSuccess);
  EXPECT_EQ(accessDevs_, (std::vector<int>{kDev, 11}));
}

// A failed restore leaves the new handle mapped on a Released entry, so it is never released; a fix flips this pin.
TEST_F(MemManagerMicrotest, Resume_KnownBug_BackupRestoreFailureKeepsTheMappingButStaysReleased) {
  std::vector<char> device(kSize);
  ncclDynMemEntry* offload = Track(device.data(), ncclMemOffload, 0x12);
  MarkSuspended({offload});
  offload->cpuBackup = std::malloc(kSize);
  copyResult_ = hipErrorInvalidValue;

  EXPECT_EQ(ncclCommMemResume(comm_.get()), ncclUnhandledCudaError);
  EXPECT_NE(offload->cpuBackup, nullptr);
  EXPECT_EQ(offload->handle, Handle(0x100));
  EXPECT_EQ(offload->state, ncclDynMemStateReleased);
  EXPECT_TRUE(releases_.empty());
  EXPECT_TRUE(unmaps_.empty());

  copyResult_ = hipSuccess;
  ASSERT_EQ(ncclCommMemResume(comm_.get()), ncclSuccess);
  ASSERT_EQ(maps_.size(), 2u);
  EXPECT_EQ(maps_[1].ptr, maps_[0].ptr);
  EXPECT_TRUE(releases_.empty());  // handle 0x100 is never released
}

TEST_F(MemManagerMicrotest, Resume_ReexportsFabricHandles) {
  ncclDynMemEntry* e = Track(Addr(0x1000), ncclMemScratch, 0x11, kFabricHandle);
  MarkSuspended({e});
  ASSERT_EQ(ncclCommMemResume(comm_.get()), ncclSuccess);
  EXPECT_TRUE(e->desc.local.shareableHandleValid);
  EXPECT_EQ(e->desc.local.shareableHandle.fabricHandle.data[0], 0xab);
}

TEST_F(MemManagerMicrotest, Resume_FabricExportFailureUndoesTheMapping) {
  ncclDynMemEntry* e = Track(Addr(0x1000), ncclMemScratch, 0x11, kFabricHandle);
  MarkSuspended({e});
  exportResult_ = hipErrorInvalidValue;
  EXPECT_EQ(ncclCommMemResume(comm_.get()), ncclUnhandledCudaError);
  ASSERT_EQ(unmaps_.size(), 1u);
  EXPECT_EQ(unmaps_[0].ptr, Addr(0x1000));
  EXPECT_EQ(releases_, std::vector<hipMemGenericAllocationHandle_t>{Handle(0x100)});
  EXPECT_EQ(e->handle, nullptr);
  EXPECT_FALSE(e->desc.local.shareableHandleValid);
  EXPECT_EQ(e->state, ncclDynMemStateReleased);
}

TEST_F(MemManagerMicrotest, Resume_BarrierFailuresAreReported) {
  ncclDynMemEntry* e = Track(Addr(0x1000), ncclMemScratch, 0x11);
  MarkSuspended({e});
  barrierResult_[0xBEEF] = ncclRemoteError;
  EXPECT_EQ(ncclCommMemResume(comm_.get()), ncclRemoteError);
  EXPECT_EQ(mgr_->released, 1);

  MarkSuspended({e});
  barrierResult_.clear();
  barrierResult_[0xCAFE] = ncclSystemError;
  EXPECT_EQ(ncclCommMemResume(comm_.get()), ncclSystemError);
  EXPECT_EQ(mgr_->released, 0);  // flipped before the final barrier
}

TEST_F(MemManagerMicrotest, Resume_WithoutBootstrapSkipsBarriersAndExchange) {
  Regroup(0, 2);
  comm_->bootstrap = nullptr;
  ncclDynMemEntry* e = Track(Addr(0x1000), ncclMemScratch, 0x11);
  ASSERT_EQ(ncclDynMemMarkExportToPeer(mgr_, Addr(0x1000), 1), ncclSuccess);
  MarkSuspended({e});
  ASSERT_EQ(ncclCommMemResume(comm_.get()), ncclSuccess);
  EXPECT_TRUE(barriers_.empty());
  EXPECT_TRUE(sent_.empty());
}

// Handle exchange

TEST_F(MemManagerMicrotest, Exchange_SendsExportedLocalHandlesToEveryPeer) {
  Regroup(1, 3);
  Track(Addr(0x1000), ncclMemScratch, 0x11);
  ncclDynMemEntry* fabric = Track(Addr(0x2000), ncclMemOffload, 0x12, kFabricHandle);
  Track(Addr(0x3000), ncclMemScratch, 0x13);  // never exported: not broadcast
  ASSERT_EQ(ncclDynMemMarkExportToPeer(mgr_, Addr(0x1000), 0), ncclSuccess);
  ASSERT_EQ(ncclDynMemMarkExportToPeer(mgr_, Addr(0x2000), 2), ncclSuccess);
  fabric->desc.local.shareableHandleValid = true;
  std::memset(&fabric->desc.local.shareableHandle.fabricHandle, 0x5c, sizeof(hipMemFabricHandle_compat_t));
  fabric->cudaDev = 6;
  mgr_->released = 1;
  inbox_[0] = {};
  inbox_[2] = {Info(2, Addr(0x7000), hipMemHandleTypePosixFileDescriptor, 0x99)};

  ASSERT_EQ(ncclCommMemResume(comm_.get()), ncclSuccess);
  ASSERT_EQ(sent_.size(), 2u);
  EXPECT_EQ(sent_.count(1), 0u);
  ASSERT_EQ(sent_[0].size(), 2u);
  const ncclDynMemP2pHandleInfo& f = sent_[0][0];
  EXPECT_EQ(f.ptr, Addr(0x2000));
  EXPECT_EQ(f.ownerRank, 1);
  EXPECT_EQ(f.ownerDev, 6);
  EXPECT_EQ(f.size, kSize);
  EXPECT_EQ(f.handleType, kFabricHandle);
  EXPECT_EQ(f.fabricHandle.data[63], 0x5c);
  const ncclDynMemP2pHandleInfo& p = sent_[0][1];
  EXPECT_EQ(p.ptr, Addr(0x1000));
  EXPECT_EQ(p.ownerDev, kDev);
  EXPECT_EQ(p.handleType, hipMemHandleTypePosixFileDescriptor);
  EXPECT_EQ(p.handleData, (uint64_t)0x11);
  EXPECT_EQ(sent_[2].size(), 2u);
  EXPECT_EQ(recvPeers_, std::vector<int>{2});
}

// A FABRIC entry without a valid exported handle is still broadcast, with its handle bytes unset.
TEST_F(MemManagerMicrotest, Exchange_FabricEntryWithoutAnExportedHandleIsSentAnyway) {
  Regroup(0, 2);
  ncclDynMemEntry* fabric = Track(Addr(0x2000), ncclMemScratch, 0x12, kFabricHandle);
  ASSERT_EQ(ncclDynMemMarkExportToPeer(mgr_, Addr(0x2000), 1), ncclSuccess);
  std::memset(&fabric->desc.local.shareableHandle.fabricHandle, 0x5c, sizeof(hipMemFabricHandle_compat_t));
  mgr_->released = 1;
  inbox_[1] = {};
  ASSERT_EQ(ncclCommMemResume(comm_.get()), ncclSuccess);
  ASSERT_EQ(sent_[1].size(), 1u);
  EXPECT_EQ(sent_[1][0].handleType, kFabricHandle);
  EXPECT_EQ(sent_[1][0].fabricHandle.data[63], kMallocPoison);
}

TEST_F(MemManagerMicrotest, Exchange_ReceivesOnlyFromPeersWithHandles) {
  Regroup(0, 3);
  ncclDynMemEntry* imported = Import(Addr(0x3000), ncclMemScratch, 0x13, 2, Addr(0x9000));
  MarkSuspended({imported});
  inbox_[1] = {};
  inbox_[2] = {Info(2, Addr(0x9000), hipMemHandleTypePosixFileDescriptor, 0x99)};
  ASSERT_EQ(ncclCommMemResume(comm_.get()), ncclSuccess);
  EXPECT_TRUE(sent_.empty());
  EXPECT_EQ(recvPeers_, std::vector<int>{2});
}

TEST_F(MemManagerMicrotest, Exchange_NothingToExchangeSkipsSendAndRecv) {
  Regroup(0, 2);
  inbox_[1] = {};
  mgr_->released = 1;
  ASSERT_EQ(ncclCommMemResume(comm_.get()), ncclSuccess);
  EXPECT_TRUE(sent_.empty());
  EXPECT_TRUE(recvPeers_.empty());
}

TEST_F(MemManagerMicrotest, Exchange_BootstrapFailuresAbortTheResume) {
  Regroup(0, 2);
  Track(Addr(0x1000), ncclMemScratch, 0x11);
  ASSERT_EQ(ncclDynMemMarkExportToPeer(mgr_, Addr(0x1000), 1), ncclSuccess);
  inbox_[1] = {Info(1, Addr(0x7000), hipMemHandleTypePosixFileDescriptor, 0x99)};

  mgr_->released = 1;
  allGatherResult_ = ncclRemoteError;
  EXPECT_EQ(ncclCommMemResume(comm_.get()), ncclRemoteError);
  allGatherResult_ = ncclSuccess;
  sendResult_ = ncclSystemError;
  EXPECT_EQ(ncclCommMemResume(comm_.get()), ncclSystemError);
  sendResult_ = ncclSuccess;
  recvResult_ = ncclInternalError;
  EXPECT_EQ(ncclCommMemResume(comm_.get()), ncclInternalError);
  EXPECT_EQ(mgr_->released, 1);
  EXPECT_EQ(barriers_, (std::vector<int>{0xBEEF, 0xBEEF, 0xBEEF}));
}

// The four exchange buffers, in allocation order: counts, offsets, local infos, all infos.
TEST_F(MemManagerMicrotest, Exchange_BufferAllocationFailures) {
  Regroup(0, 2);
  Track(Addr(0x1000), ncclMemScratch, 0x11);
  ASSERT_EQ(ncclDynMemMarkExportToPeer(mgr_, Addr(0x1000), 1), ncclSuccess);
  inbox_[1] = {};
  for (int failAt = 1; failAt <= 4; failAt++) {
    SCOPED_TRACE(failAt);
    mgr_->released = 1;
    g_mallocCalls = 0;
    g_mallocFailAt = failAt;
    EXPECT_EQ(ncclCommMemResume(comm_.get()), ncclSystemError);
    EXPECT_TRUE(sent_.empty());
  }
  g_mallocCalls = 0;
  g_mallocFailAt = 0;
  ASSERT_EQ(ncclCommMemResume(comm_.get()), ncclSuccess);
  EXPECT_EQ(g_mallocCalls, 4);
}

// Peer re-import

TEST_F(MemManagerMicrotest, Reimport_PosixPeerBufferThroughTheOwnersProxy) {
  Regroup(0, 2);
  ncclDynMemEntry* imported = Import(Addr(0x3000), ncclMemScratch, 0x13, 1, Addr(0x9000));
  MarkSuspended({imported});
  inbox_[1] = {Info(1, Addr(0x8000), hipMemHandleTypePosixFileDescriptor, 0x98),
               Info(1, Addr(0x9000), hipMemHandleTypePosixFileDescriptor, 0x99)};

  ASSERT_EQ(ncclCommMemResume(comm_.get()), ncclSuccess);
  EXPECT_EQ(fdRequests_, (std::vector<std::pair<int, uint64_t>>{{1, 0x99}}));
  ASSERT_EQ(imports_.size(), 1u);
  EXPECT_EQ(imports_[0].shareable, Addr(77));
  EXPECT_EQ(imports_[0].type, hipMemHandleTypePosixFileDescriptor);
  EXPECT_EQ(g_closedFds, std::vector<int>{77});
  ASSERT_EQ(maps_.size(), 1u);
  EXPECT_EQ(maps_[0].ptr, Addr(0x3000));
  EXPECT_EQ(accessDevs_, std::vector<int>{kDev});
  EXPECT_EQ(imported->handle, Handle(0x100));
  EXPECT_EQ(imported->state, ncclDynMemStateActive);
  EXPECT_EQ(mgr_->released, 0);
}

TEST_F(MemManagerMicrotest, Reimport_FabricPeerBufferFromTheExchangedHandle) {
  Regroup(0, 2);
  ncclDynMemEntry* imported = Import(Addr(0x3000), ncclMemScratch, 0x13, 1, Addr(0x9000));
  MarkSuspended({imported});
  inbox_[1] = {Info(1, Addr(0x9000), kFabricHandle, 0)};
  inbox_[1][0].fabricHandle.data[0] = 0x42;
  g_hipMemImportFromShareableHandle = [this](hipMemGenericAllocationHandle_t* h, void* shareable,
                                             hipMemAllocationHandleType type) {
    EXPECT_EQ(type, kFabricHandle);
    EXPECT_EQ(static_cast<unsigned char*>(shareable)[0], 0x42);
    *h = Handle(0x200);
    return hipSuccess;
  };
  ASSERT_EQ(ncclCommMemResume(comm_.get()), ncclSuccess);
  EXPECT_TRUE(fdRequests_.empty());
  EXPECT_EQ(imported->handle, Handle(0x200));
  EXPECT_EQ(imported->state, ncclDynMemStateActive);
}

// Each failing peer leaves its entry Released and the manager suspended; the first error is kept.
TEST_F(MemManagerMicrotest, Reimport_FailuresLeaveTheEntryReleased) {
  struct Case {
    const char* name;
    std::function<void()> arm;
    ncclResult_t expected;
  };
  const std::vector<Case> cases = {
      {"no matching info", [this] { inbox_[1][0].size = kSize + 1; }, ncclSystemError},
      {"owner on another node",
       [this] {
         comm_->peerInfoValid = true;
         peers_[1].hostHash = 0x41;
       },
       ncclInvalidUsage},
      {"fd request fails", [this] { getFdResult_ = ncclRemoteError; }, ncclRemoteError},
      {"fd request returns no fd", [this] { fd_ = -1; }, ncclSystemError},
      {"unknown handle type", [this] { inbox_[1][0].handleType = 3; }, ncclInvalidUsage},
      {"import fails", [this] { importResult_ = hipErrorInvalidValue; }, ncclUnhandledCudaError},
      {"map fails", [this] { mapResult_ = hipErrorInvalidValue; }, ncclUnhandledCudaError},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(c.name);
    Regroup(0, 2);
    ncclDynMemEntry* imported = Import(Addr(0x3000), ncclMemScratch, 0x13, 1, Addr(0x9000));
    MarkSuspended({imported});
    inbox_[1] = {Info(1, Addr(0x9000), hipMemHandleTypePosixFileDescriptor, 0x99)};
    getFdResult_ = ncclSuccess;
    fd_ = 77;
    importResult_ = hipSuccess;
    mapResult_ = hipSuccess;
    releases_.clear();
    c.arm();
    EXPECT_EQ(ncclCommMemResume(comm_.get()), c.expected);
    EXPECT_EQ(imported->state, ncclDynMemStateReleased);
    EXPECT_EQ(imported->handle, nullptr);
    EXPECT_EQ(mgr_->released, 1);
    EXPECT_EQ(barriers_.back(), 0xCAFE);
    if (c.expected == ncclUnhandledCudaError && mapResult_ != hipSuccess)
      EXPECT_EQ(releases_, std::vector<hipMemGenericAllocationHandle_t>{Handle(nextHandle_ - 1)});
    ASSERT_EQ(ncclMemUntrackDynamic(mgr_, Addr(0x3000), nullptr), ncclSuccess);
  }
}

// Entries are walked newest first: the unknown handle type fails before the missing info and the import.
TEST_F(MemManagerMicrotest, Reimport_KeepsTheFirstErrorAcrossEntries) {
  Regroup(0, 2);
  ncclDynMemEntry* importFails = Import(Addr(0x2000), ncclMemScratch, 0x12, 1, Addr(0x8000));
  ncclDynMemEntry* noInfo = Import(Addr(0x3000), ncclMemScratch, 0x13, 1, Addr(0x9000));
  ncclDynMemEntry* unknownType = Import(Addr(0x4000), ncclMemScratch, 0x14, 1, Addr(0x9100));
  MarkSuspended({importFails, noInfo, unknownType});
  inbox_[1] = {Info(1, Addr(0x9100), 3, 0), Info(1, Addr(0x8000), kFabricHandle, 0)};
  importResult_ = hipErrorInvalidValue;
  EXPECT_EQ(ncclCommMemResume(comm_.get()), ncclInvalidUsage);
  EXPECT_EQ(imports_.size(), 1u);
}

// Infos from each peer land at that peer's offset, and owners are told apart even at the same address.
TEST_F(MemManagerMicrotest, Reimport_MatchesInfosByOwnerRank) {
  Regroup(2, 3);
  ncclDynMemEntry* from0 = Import(Addr(0x3000), ncclMemScratch, 0x13, 0, Addr(0x9000));
  ncclDynMemEntry* from1 = Import(Addr(0x4000), ncclMemScratch, 0x14, 1, Addr(0x9000));
  MarkSuspended({from0, from1});
  inbox_[0] = {Info(0, Addr(0x9000), hipMemHandleTypePosixFileDescriptor, 0x90)};
  inbox_[1] = {Info(1, Addr(0x9000), hipMemHandleTypePosixFileDescriptor, 0x91)};
  ASSERT_EQ(ncclCommMemResume(comm_.get()), ncclSuccess);
  EXPECT_EQ(fdRequests_, (std::vector<std::pair<int, uint64_t>>{{1, 0x91}, {0, 0x90}}));
  EXPECT_EQ(recvPeers_, (std::vector<int>{0, 1}));
}

// The POSIX same-node check only applies once peer info has been exchanged.
TEST_F(MemManagerMicrotest, Reimport_SkipsTheNodeCheckWithoutPeerInfo) {
  Regroup(0, 2);
  peers_[1].hostHash = 0x41;
  ncclDynMemEntry* imported = Import(Addr(0x3000), ncclMemScratch, 0x13, 1, Addr(0x9000));
  MarkSuspended({imported});
  inbox_[1] = {Info(1, Addr(0x9000), hipMemHandleTypePosixFileDescriptor, 0x99)};
  ASSERT_EQ(ncclCommMemResume(comm_.get()), ncclSuccess);
  EXPECT_EQ(imported->state, ncclDynMemStateActive);
}

// Public Suspend/Resume entry points

TEST_F(MemManagerMicrotest, SuspendImpl_RejectsBadCommAndFlagsBeforeTheGroup) {
  g_commCheck = [](ncclComm*) { return ncclInvalidArgument; };
  EXPECT_EQ(ncclCommSuspend_impl(comm_.get(), NCCL_SUSPEND_MEM), ncclInvalidArgument);
  EXPECT_EQ(ncclCommResume_impl(comm_.get()), ncclInvalidArgument);
  uint64_t value = 0;
  EXPECT_EQ(ncclCommMemStats_impl(comm_.get(), ncclStatGpuMemTotal, &value), ncclInvalidArgument);
  g_commCheck = [](ncclComm*) { return ncclSuccess; };
  EXPECT_EQ(ncclCommSuspend_impl(comm_.get(), 0), ncclInvalidArgument);
  EXPECT_EQ(ncclCommSuspend_impl(comm_.get(), NCCL_SUSPEND_MEM | 0x10), ncclInvalidArgument);
  EXPECT_EQ(ncclGroupDepth, 0);
  EXPECT_TRUE(jobs_.empty());
}

TEST_F(MemManagerMicrotest, SuspendImpl_QueuesATaskAndJoinsTheGroup) {
  g_currentDevice = 6;
  ASSERT_EQ(ncclCommSuspend_impl(comm_.get(), NCCL_SUSPEND_MEM), ncclSuccess);
  ncclMemManagerTask* task = ncclIntruQueueHead(&comm_->suspendTaskQueue);
  ASSERT_NE(task, nullptr);
  tasks_.push_back(task);
  EXPECT_EQ(task->comm, comm_.get());
  EXPECT_EQ(ncclGroupCommHead[ncclGroupTaskTypeSymRegister], comm_.get());
  EXPECT_EQ(ncclGroupDepth, 0);
  EXPECT_EQ(g_currentDevice, 6);
}

TEST_F(MemManagerMicrotest, ResumeImpl_QueuesATaskAndJoinsTheGroup) {
  mgr_->released = 1;
  ASSERT_EQ(ncclCommResume_impl(comm_.get()), ncclSuccess);
  ncclMemManagerTask* task = ncclIntruQueueHead(&comm_->resumeTaskQueue);
  ASSERT_NE(task, nullptr);
  tasks_.push_back(task);
  EXPECT_EQ(task->comm, comm_.get());
  EXPECT_EQ(ncclGroupCommHead[ncclGroupTaskTypeSymRegister], comm_.get());
}

TEST_F(MemManagerMicrotest, SuspendImpl_RearchEnqueuesAJobThatSuspends) {
  g_rearch = 1;
  Track(Addr(0x1000), ncclMemScratch, 0x11);
  g_currentDevice = 6;
  ASSERT_EQ(ncclCommSuspend_impl(comm_.get(), NCCL_SUSPEND_MEM), ncclSuccess);
  ASSERT_EQ(jobs_.size(), 1u);
  EXPECT_TRUE(ncclIntruQueueEmpty(&comm_->suspendTaskQueue));
  EXPECT_EQ(g_currentDevice, 6);

  EXPECT_EQ(jobs_[0].func(jobs_[0].job), ncclSuccess);
  EXPECT_EQ(g_currentDevice, kDev);
  EXPECT_EQ(mgr_->released, 1);
  EXPECT_EQ(jobs_[0].func(jobs_[0].job), ncclInvalidUsage);  // already suspended
  jobs_[0].destructor(jobs_[0].job);
}

TEST_F(MemManagerMicrotest, ResumeImpl_RearchEnqueuesAJobThatResumes) {
  g_rearch = 1;
  mgr_->released = 1;
  ASSERT_EQ(ncclCommResume_impl(comm_.get()), ncclSuccess);
  ASSERT_EQ(jobs_.size(), 1u);
  EXPECT_EQ(jobs_[0].func(jobs_[0].job), ncclSuccess);
  EXPECT_EQ(mgr_->released, 0);
  g_hipSetDevice = [](int) { return hipErrorInvalidValue; };
  EXPECT_EQ(jobs_[0].func(jobs_[0].job), ncclUnhandledCudaError);
  g_hipSetDevice = [](int dev) {
    g_currentDevice = dev;
    return hipSuccess;
  };
  jobs_[0].destructor(jobs_[0].job);
}

TEST_F(MemManagerMicrotest, SuspendImpl_JobReportsSetDeviceFailure) {
  g_rearch = 1;
  ASSERT_EQ(ncclCommSuspend_impl(comm_.get(), NCCL_SUSPEND_MEM), ncclSuccess);
  ASSERT_EQ(jobs_.size(), 1u);
  g_hipSetDevice = [](int) { return hipErrorInvalidValue; };
  EXPECT_EQ(jobs_[0].func(jobs_[0].job), ncclUnhandledCudaError);
  EXPECT_EQ(mgr_->released, 0);
  jobs_[0].destructor(jobs_[0].job);
}

// Suspend deletes a job ncclMgmtTaskEnqueue already destroyed on rejecting it (Resume doesn't); a fix flips this pin.
TEST_F(MemManagerMicrotest, Impl_KnownBug_RejectedJobIsDeletedTwiceBySuspend) {
  g_rearch = 1;
  g_mgmtEnqueue = [](ncclAsyncJob* job, ncclResult_t (*)(ncclAsyncJob*), void (*destructor)(void*)) {
    g_watchedDelete = job;
    destructor(job);
    return ncclInvalidArgument;
  };
  g_watchedDeletes = 0;
  EXPECT_EQ(ncclCommSuspend_impl(comm_.get(), NCCL_SUSPEND_MEM), ncclInvalidArgument);
  EXPECT_EQ(g_watchedDeletes, 2);
  std::free(g_watchedDelete);

  mgr_->released = 1;
  g_watchedDeletes = 0;
  EXPECT_EQ(ncclCommResume_impl(comm_.get()), ncclInvalidArgument);
  EXPECT_EQ(g_watchedDeletes, 1);
  std::free(g_watchedDelete);
  g_watchedDelete = nullptr;
  EXPECT_EQ(groupErrors_, (std::vector<ncclResult_t>{ncclInvalidArgument, ncclInvalidArgument}));
}

TEST_F(MemManagerMicrotest, Impl_RejectsSharedManagersAndWrongStates) {
  mgr_->refCount = 2;
  EXPECT_EQ(ncclCommSuspend_impl(comm_.get(), NCCL_SUSPEND_MEM), ncclInvalidUsage);
  EXPECT_EQ(ncclCommResume_impl(comm_.get()), ncclInvalidUsage);
  mgr_->refCount = 1;
  EXPECT_EQ(ncclCommResume_impl(comm_.get()), ncclInvalidUsage);
  mgr_->released = 1;
  EXPECT_EQ(ncclCommSuspend_impl(comm_.get(), NCCL_SUSPEND_MEM), ncclInvalidUsage);
  EXPECT_TRUE(ncclIntruQueueEmpty(&comm_->suspendTaskQueue));
  EXPECT_TRUE(ncclIntruQueueEmpty(&comm_->resumeTaskQueue));
  EXPECT_EQ(groupErrors_, std::vector<ncclResult_t>(4, ncclInvalidUsage));
}

// A queued opposite operation makes the state check pass: it flips `released` when the group drains.
TEST_F(MemManagerMicrotest, Impl_AcceptsAPendingOppositeTask) {
  mgr_->released = 1;
  ncclMemManagerTask pendingResume{};
  ncclIntruQueueEnqueue(&comm_->resumeTaskQueue, &pendingResume);
  ASSERT_EQ(ncclCommSuspend_impl(comm_.get(), NCCL_SUSPEND_MEM), ncclSuccess);
  tasks_.push_back(ncclIntruQueueHead(&comm_->suspendTaskQueue));

  mgr_->released = 0;
  comm_->groupNext[ncclGroupTaskTypeSymRegister] = reinterpret_cast<ncclComm*>(NCCL_COMM_GROUP_INVALID);
  ncclGroupCommHead[ncclGroupTaskTypeSymRegister] = nullptr;
  ASSERT_EQ(ncclCommResume_impl(comm_.get()), ncclSuccess);
  ncclMemManagerTask* resumeTask = ncclIntruQueueHead(&comm_->resumeTaskQueue)->next;
  ASSERT_NE(resumeTask, nullptr);
  tasks_.push_back(resumeTask);
}

TEST_F(MemManagerMicrotest, Impl_DeviceSwitchFailures) {
  g_hipGetDevice = [](int*) { return hipErrorInvalidValue; };
  EXPECT_EQ(ncclCommSuspend_impl(comm_.get(), NCCL_SUSPEND_MEM), ncclUnhandledCudaError);
  EXPECT_EQ(ncclCommResume_impl(comm_.get()), ncclUnhandledCudaError);
  EXPECT_EQ(ncclGroupDepth, 0);

  g_hipGetDevice = [](int* dev) {
    *dev = 6;
    return hipSuccess;
  };
  std::vector<int> sets;
  g_hipSetDevice = [&](int dev) {
    sets.push_back(dev);
    return dev == kDev ? hipErrorInvalidValue : hipSuccess;
  };
  EXPECT_EQ(ncclCommSuspend_impl(comm_.get(), NCCL_SUSPEND_MEM), ncclUnhandledCudaError);
  mgr_->released = 1;
  EXPECT_EQ(ncclCommResume_impl(comm_.get()), ncclUnhandledCudaError);
  EXPECT_EQ(sets, (std::vector<int>{kDev, 6, kDev, 6}));
  EXPECT_TRUE(ncclIntruQueueEmpty(&comm_->suspendTaskQueue));
  EXPECT_TRUE(ncclIntruQueueEmpty(&comm_->resumeTaskQueue));
}

TEST_F(MemManagerMicrotest, Impl_RestoreFailureIsReportedAfterSuccess) {
  g_rearch = 1;
  g_hipSetDevice = [](int dev) { return dev == kDev ? hipSuccess : hipErrorInvalidValue; };
  EXPECT_EQ(ncclCommSuspend_impl(comm_.get(), NCCL_SUSPEND_MEM), ncclUnhandledCudaError);
  mgr_->released = 1;
  EXPECT_EQ(ncclCommResume_impl(comm_.get()), ncclUnhandledCudaError);
  g_hipSetDevice = [](int) { return hipSuccess; };
  for (auto& j : jobs_) j.destructor(j.job);
}

TEST_F(MemManagerMicrotest, Impl_GroupEndAndAsyncErrorsPropagate) {
  g_rearch = 1;
  g_groupEnd = [] { return ncclRemoteError; };
  EXPECT_EQ(ncclCommSuspend_impl(comm_.get(), NCCL_SUSPEND_MEM), ncclRemoteError);
  mgr_->released = 1;
  EXPECT_EQ(ncclCommResume_impl(comm_.get()), ncclRemoteError);

  g_groupEnd = [] { return ncclSuccess; };
  comm_->config.blocking = 0;
  g_asyncError = ncclInProgress;
  EXPECT_EQ(ncclCommResume_impl(comm_.get()), ncclInProgress);
  mgr_->released = 0;
  EXPECT_EQ(ncclCommSuspend_impl(comm_.get(), NCCL_SUSPEND_MEM), ncclInProgress);
  g_asyncErrorResult = ncclInvalidArgument;
  EXPECT_EQ(ncclCommSuspend_impl(comm_.get(), NCCL_SUSPEND_MEM), ncclInvalidArgument);
  comm_->config.blocking = 1;
  EXPECT_EQ(ncclCommSuspend_impl(comm_.get(), NCCL_SUSPEND_MEM), ncclSuccess);
  for (auto& j : jobs_) j.destructor(j.job);
}

// MemStats

TEST_F(MemManagerMicrotest, MemStats_SumsLocalCountersOnly) {
  ASSERT_EQ(ncclMemTrack(mgr_, Addr(0x1000), 100, nullptr, hipMemHandleTypePosixFileDescriptor, ncclMemPersist),
            ncclSuccess);
  Track(Addr(0x2000), ncclMemScratch, 0x12);
  Track(Addr(0x3000), ncclMemOffload, 0x13);
  Import(Addr(0x4000), ncclMemScratch, 0x14, 1, Addr(0x9000));
  uint64_t total = 0, persist = 0, suspendable = 0, suspended = 9;
  ASSERT_EQ(ncclCommMemStats_impl(comm_.get(), ncclStatGpuMemTotal, &total), ncclSuccess);
  ASSERT_EQ(ncclCommMemStats_impl(comm_.get(), ncclStatGpuMemPersist, &persist), ncclSuccess);
  ASSERT_EQ(ncclCommMemStats_impl(comm_.get(), ncclStatGpuMemSuspend, &suspendable), ncclSuccess);
  ASSERT_EQ(ncclCommMemStats_impl(comm_.get(), ncclStatGpuMemSuspended, &suspended), ncclSuccess);
  EXPECT_EQ(total, 100 + 2 * kSize);
  EXPECT_EQ(persist, 100u);
  EXPECT_EQ(suspendable, 2 * kSize);
  EXPECT_EQ(suspended, 0u);
  EXPECT_EQ(ncclCommMemStats_impl(comm_.get(), ncclStatGpuMemTotal, nullptr), ncclInvalidArgument);
  EXPECT_EQ(ncclCommMemStats_impl(comm_.get(), (ncclCommMemStat_t)99, &total), ncclInvalidArgument);
}

}  // namespace
