/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for the helpers in src/include/alloc.h: shutdown and skip-free gates, side
// streams, host/device/VMM allocation and free, and the allocation tracker. HIP goes through
// hip_fakes; the memory-manager calls and getBusId are seamed in this TU.

#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "alloc.h"
#include "debug.h"
#include "fakes/hip_fakes.h"

// hipify renames the CUDA FABRIC names wherever HIP has them; p2p.h only defines them otherwise.
#ifdef CU_MEM_HANDLE_TYPE_FABRIC
const hipMemAllocationHandleType kFabricHandle = CU_MEM_HANDLE_TYPE_FABRIC;
#else
const hipMemAllocationHandleType kFabricHandle = hipMemHandleTypeFabric;
#endif

uint32_t ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED;
uint64_t ncclDebugMask = ~uint64_t(0);
thread_local int ncclDebugNoWarn = 0;
struct allocationTracker allocTracker[MAX_ALLOC_TRACK_NGPU] = {};
hipMemAllocationHandleType ncclCuMemHandleType = hipMemHandleTypePosixFileDescriptor;

namespace {

std::vector<std::string> g_warns;
int g_cuMemEnable = 0;
ncclResult_t g_busIdResult = ncclSuccess;

struct TrackCall {
  void* ptr;
  size_t size;
  hipMemAllocationHandleType type;
  ncclMemType_t memType;
};
std::vector<TrackCall> g_tracked;
ncclResult_t g_trackResult = ncclSuccess;
ncclMemUntrackInfo g_untrackInfo{};
ncclResult_t g_untrackResult = ncclSuccess;
std::vector<std::pair<void*, size_t>> g_untrackedPersist;

bool Warned(const char* needle) {
  for (const std::string& w : g_warns)
    if (w.find(needle) != std::string::npos) return true;
  return false;
}

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

int ncclCuMemEnable() { return g_cuMemEnable; }
size_t ncclOsGetPageSize() { return 4096; }
ncclResult_t getBusId(int cudaDev, int64_t* busId) {
  *busId = 0x100 * cudaDev;
  return g_busIdResult;
}
ncclResult_t ncclMemTrack(struct ncclMemManager*, void* ptr, size_t size, hipMemGenericAllocationHandle_t,
                          hipMemAllocationHandleType type, ncclMemType_t memType) {
  g_tracked.push_back({ptr, size, type, memType});
  return g_trackResult;
}
ncclResult_t ncclMemUntrackDynamic(struct ncclMemManager*, void*, struct ncclMemUntrackInfo* info) {
  *info = g_untrackInfo;
  return g_untrackResult;
}
ncclResult_t ncclMemUntrackPersist(struct ncclMemManager*, void* ptr, size_t size) {
  g_untrackedPersist.push_back({ptr, size});
  return ncclSuccess;
}

namespace {

constexpr int kDev = 3;
constexpr hipDevice_t kHipDev = 13;
constexpr size_t kGran = 4096;
#if defined(HIP_VMM_UNCACHED_MEMORY)
constexpr hipMemAllocationType kVmmType = hipMemAllocationTypeUncached;
#else
constexpr hipMemAllocationType kVmmType = hipMemAllocationTypePinned;
#endif
constexpr uintptr_t kVa = 0x100000;
const hipStream_t kPrivateStream = reinterpret_cast<hipStream_t>(0x5);
const hipStream_t kSideStream = reinterpret_cast<hipStream_t>(0x6);
const hipMemGenericAllocationHandle_t kHandle = reinterpret_cast<hipMemGenericAllocationHandle_t>(0x77);

void* Va(uintptr_t offset = 0) {
  return reinterpret_cast<void*>(kVa + offset);
}

class AllocMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetHipFakes();
    // Memoized per process (gfx942 from the default fakes: no skip); settle it before calls are logged.
    ASSERT_FALSE(rcclSkipCuMemFree());
    ASSERT_FALSE(rcclSkipLsaFlatAddressFree());
    rcclShutdownFlag().store(false);
    sideStream.clear();
    std::memset(allocTracker, 0, sizeof(allocTracker));
    g_warns.clear();
    g_cuMemEnable = 0;
    g_busIdResult = ncclSuccess;
    g_tracked.clear();
    g_trackResult = ncclSuccess;
    g_untrackInfo = {};
    g_untrackResult = ncclSuccess;
    g_untrackedPersist.clear();
    ncclCuMemHandleType = hipMemHandleTypePosixFileDescriptor;

    g_hipGetDevice = [this](int* d) {
      *d = kDev;
      return Call("getDevice");
    };
    g_hipDeviceGet = [this](hipDevice_t* d, int ordinal) {
      EXPECT_EQ(ordinal, kDev);
      *d = kHipDev;
      return Call("deviceGet");
    };
    g_hipDeviceGetAttribute = [this](int* v, hipDeviceAttribute_t, int) {
      *v = -1;  // host NUMA id: none
      return Call("numa");
    };
    g_hipMemGetAllocationGranularity = [this](size_t* g, const hipMemAllocationProp* prop,
                                              hipMemAllocationGranularity_flags) {
      prop_ = *prop;
      *g = kGran;
      return Call("granularity");
    };
    g_hipMemGetAllocationPropertiesFromHandle = [this](hipMemAllocationProp* prop, hipMemGenericAllocationHandle_t h) {
      EXPECT_EQ(h, kHandle);
      *prop = hipMemAllocationProp{};
      prop->location.type = handleLocation_;
      return Call("props");
    };
    g_hipMemCreate = [this](hipMemGenericAllocationHandle_t* h, size_t size, const hipMemAllocationProp* prop,
                            unsigned long long) {
      createProp_ = *prop;
      sizes_["create"] = size;
      *h = kHandle;
      return Call("create");
    };
    g_hipMemAddressReserve = [this](void** p, size_t size, size_t align, void*, unsigned long long) {
      EXPECT_EQ(align, kGran);
      sizes_["reserve"] = size;
      *p = Va();
      return Call("reserve");
    };
    g_hipMemMap = [this](void* p, size_t size, size_t, hipMemGenericAllocationHandle_t h, unsigned long long) {
      EXPECT_EQ(h, kHandle);
      sizes_["map"] = size;
      return Call("map");
    };
    g_hipMemSetAccess = [this](void*, size_t, const hipMemAccessDesc* desc, size_t count) {
      EXPECT_EQ(count, 1u);
      EXPECT_EQ(desc->flags, hipMemAccessFlagsProtReadWrite);
      access_.push_back({desc->location.type, desc->location.id});
      return Call("access");
    };
    g_hipMemUnmap = [this](void* p, size_t size) {
      unmaps_.push_back({p, size});
      return Call("unmap");
    };
    g_hipMemRelease = [this](hipMemGenericAllocationHandle_t h) {
      EXPECT_EQ(h, kHandle);
      return Call("release");
    };
    g_hipMemAddressFree = [this](void* p, size_t size) {
      addressFrees_.push_back({p, size});
      return Call("addrfree");
    };
    g_hipMemRetainAllocationHandle = [this](hipMemGenericAllocationHandle_t* h, void*) {
      *h = kHandle;
      return Call("retain");
    };
    g_hipMemGetAddressRange = [this](hipDeviceptr_t* base, size_t* size, hipDeviceptr_t ptr) {
      auto it = ranges_.find(ptr);
      *base = it == ranges_.end() ? ptr : it->second.first;
      *size = it == ranges_.end() ? segment_ : it->second.second;
      return Call("range");
    };
    g_hipThreadExchangeStreamCaptureMode = [this](hipStreamCaptureMode* mode) {
      EXPECT_EQ(*mode, hipStreamCaptureModeRelaxed);
      return Call("capture");
    };
    g_hipDeviceGetStreamPriorityRange = [this](int* least, int* greatest) {
      *least = 0;
      *greatest = -3;
      return Call("priorityRange");
    };
    g_hipStreamCreateWithPriority = [this](hipStream_t* s, unsigned flags, int priority) {
      EXPECT_EQ(flags, hipStreamNonBlocking);
      priorities_.push_back(priority);
      *s = kSideStream;
      return Call("sideCreate");
    };
    g_hipStreamCreateWithFlags = [this](hipStream_t* s, unsigned flags) {
      EXPECT_EQ(flags, hipStreamNonBlocking);
      *s = kPrivateStream;
      return Call("streamCreate");
    };
    g_hipStreamDestroy = [this](hipStream_t s) {
      destroyed_.push_back(s);
      return Call("streamDestroy");
    };
    g_hipStreamSynchronize = [this](hipStream_t s) {
      synced_.push_back(s);
      return Call("sync");
    };
    g_hipMemsetAsync = [this](void* p, int value, size_t n, hipStream_t s) {
      EXPECT_EQ(value, 0);
      memsets_.push_back({p, n});
      memsetStreams_.push_back(s);
      if (!failing_.count("memset") && heap_.count(p)) std::memset(p, 0, n);
      return Call("memset");
    };
    g_hipMemcpyAsync = [this](void*, const void*, size_t n, hipMemcpyKind kind, hipStream_t s) {
      EXPECT_EQ(kind, hipMemcpyDefault);
      sizes_["memcpy"] = n;
      memsetStreams_.push_back(s);
      return Call("memcpy");
    };
    g_hipExtMallocWithFlags = [this](void** p, size_t n, unsigned flags) {
      sizes_["extMalloc"] = n;
      flags_ = flags;
      Status s = Call("extMalloc");
      *p = s == hipSuccess ? Heap(n) : nullptr;
      return s;
    };
    g_hipHostMalloc = [this](void** p, size_t n, unsigned flags) {
      sizes_["hostMalloc"] = n;
      flags_ = flags;
      Status s = Call("hostMalloc");
      *p = s == hipSuccess ? Heap(n) : nullptr;
      return s;
    };
    g_hipHostFree = [this](void* p) {
      Unheap(p);
      return Call("hostFree");
    };
    g_hipFree = [this](void* p) {
      Unheap(p);
      return Call("free");
    };
  }

  void TearDown() override {
    EXPECT_TRUE(heap_.empty()) << heap_.size() << " fake allocations leaked";
    for (void* p : heap_) std::free(p);
    rcclShutdownFlag().store(false);
    sideStream.clear();
    ResetHipFakes();
  }

  using Status = hipError_t;

  // Logs one fake HIP call by name; names in failing_ fail with hipErrorInvalidValue.
  Status Call(const char* name) {
    calls_.push_back(name);
    return failing_.count(name) ? hipErrorInvalidValue : hipSuccess;
  }
  void* Heap(size_t n) {
    void* p = std::malloc(n);
    std::memset(p, 0xab, n);
    heap_.insert(p);
    return p;
  }
  void Unheap(void* p) {
    if (heap_.erase(p)) std::free(p);
  }
  int Count(const std::string& name) const {
    int n = 0;
    for (const std::string& c : calls_) n += c == name;
    return n;
  }
  // Every capture-mode exchange is undone by a second one.
  bool CaptureModeRestored() const { return Count("capture") % 2 == 0; }

  std::vector<std::string> calls_;
  std::set<std::string> failing_;
  std::map<std::string, size_t> sizes_;
  std::set<void*> heap_;
  std::map<hipDeviceptr_t, std::pair<hipDeviceptr_t, size_t>> ranges_;
  size_t segment_ = 2 * kGran;
  hipMemLocationType handleLocation_ = hipMemLocationTypeDevice;
  hipMemAllocationProp prop_{}, createProp_{};
  std::vector<std::pair<hipMemLocationType, int>> access_;
  std::vector<std::pair<void*, size_t>> unmaps_, addressFrees_, memsets_;
  std::vector<hipStream_t> destroyed_, synced_, memsetStreams_;
  std::vector<int> priorities_;
  unsigned flags_ = 0;
};

// Shutdown flag and skip-free gates

TEST_F(AllocMicrotest, Shutdown_FreesBecomeNoOps) {
  rcclShutdownHandler();
  EXPECT_TRUE(rcclShutdownFlag().load());
  int x = 0;
  EXPECT_EQ(ncclCudaHostFree(&x), ncclSuccess);
  EXPECT_EQ(ncclCudaFree(&x, nullptr), ncclSuccess);
#if ROCM_VERSION >= 70000
  EXPECT_EQ(ncclCuMemFree(&x, nullptr), ncclSuccess);
  EXPECT_EQ(ncclCuMemFreeAddr(&x, nullptr), ncclSuccess);
#endif
  for (const char* call : {"hostFree", "free", "range", "unmap", "release", "addrfree"}) EXPECT_EQ(Count(call), 0);
}

// atexit runs handlers in reverse order, so a checker registered first sees the flag.
TEST_F(AllocMicrotest, Shutdown_RegisteredHandlerRunsAtExit) {
  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
  EXPECT_EXIT(
      {
        std::atexit([] { _exit(rcclShutdownFlag().load() ? 42 : 1); });
        rcclRegisterShutdownHandler();
        rcclRegisterShutdownHandler();
        std::exit(0);
      },
      ::testing::ExitedWithCode(42), "");
  rcclRegisterShutdownHandler();  // here too: it only arms the flag for this process's own exit
  EXPECT_FALSE(rcclShutdownFlag().load());
}

TEST_F(AllocMicrotest, SkipFree_ArchPredicates) {
  hipDeviceProp_t prop{};
  const std::vector<std::tuple<const char*, bool, bool>> cases = {
      {"gfx950:sramecc+:xnack-", true, false}, {"gfx1250", true, true}, {"gfx942", false, false}};
  for (const auto& c : cases) {
    std::snprintf(prop.gcnArchName, sizeof(prop.gcnArchName), "%s", std::get<0>(c));
    EXPECT_EQ(rcclSkipCuMemFreeFromArch(prop), std::get<1>(c)) << std::get<0>(c);
    EXPECT_EQ(rcclSkipLsaFlatAddressFreeFromArch(prop), std::get<2>(c)) << std::get<0>(c);
  }
}

TEST_F(AllocMicrotest, QueryDeviceProp_ReportsHipFailures) {
  hipDeviceProp_t prop;
  std::vector<int> devs;
  g_hipGetDeviceProperties = [&](hipDeviceProp_t* p, int dev) {
    devs.push_back(dev);
    std::snprintf(p->gcnArchName, sizeof(p->gcnArchName), "gfx950");
    return Call("properties");
  };
  ASSERT_TRUE(rcclQueryDeviceProp(&prop));
  EXPECT_STREQ(prop.gcnArchName, "gfx950");
  EXPECT_EQ(devs, std::vector<int>{kDev});
  failing_ = {"properties"};
  EXPECT_FALSE(rcclQueryDeviceProp(&prop));
  failing_ = {"getDevice"};
  EXPECT_FALSE(rcclQueryDeviceProp(&prop));
  EXPECT_STREQ(prop.gcnArchName, "");
}

// The decision is memoized per process, so each case runs in a fresh child (no fixture, which
// settles it): the exit code is rcclSkipCuMemFree() * 2 + rcclSkipLsaFlatAddressFree().
TEST(AllocSkipFreeMicrotest, EnvOverridesArchAndIsComputedOnce) {
  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
  ResetHipFakes();
  auto arch = [](const char* name) {
    g_hipGetDeviceProperties = [name](hipDeviceProp_t* p, int) {
      std::snprintf(p->gcnArchName, sizeof(p->gcnArchName), "%s", name);
      return hipSuccess;
    };
  };
  auto run = [] { _exit(rcclSkipCuMemFree() * 2 + rcclSkipLsaFlatAddressFree()); };
  EXPECT_EXIT((unsetenv("NCCL_CUMEM_SKIP_FREE"), arch("gfx950"), run()), ::testing::ExitedWithCode(2), "");
  EXPECT_EXIT((unsetenv("NCCL_CUMEM_SKIP_FREE"), arch("gfx1250"), run()), ::testing::ExitedWithCode(3), "");
  EXPECT_EXIT((setenv("NCCL_CUMEM_SKIP_FREE", "1", 1), arch("gfx942"), run()), ::testing::ExitedWithCode(3), "");
  EXPECT_EXIT((setenv("NCCL_CUMEM_SKIP_FREE", "0", 1), arch("gfx1250"), run()), ::testing::ExitedWithCode(0), "");
  EXPECT_EXIT((unsetenv("NCCL_CUMEM_SKIP_FREE"), g_hipGetDevice = [](int*) { return hipErrorInvalidValue; }, run()),
              ::testing::ExitedWithCode(0), "");
  EXPECT_EXIT(
      {
        unsetenv("NCCL_CUMEM_SKIP_FREE");
        arch("gfx942");
        bool first = rcclSkipCuMemFree();
        arch("gfx950");
        setenv("NCCL_CUMEM_SKIP_FREE", "1", 1);
        _exit(first == rcclSkipCuMemFree() ? 7 : 8);
      },
      ::testing::ExitedWithCode(7), "");
}

// Host-side calloc/realloc and owning pointers

TEST_F(AllocMicrotest, SizeOfT_VoidCountsBytes) {
  EXPECT_EQ(ncclSizeOfT<void>(), 1u);
  EXPECT_EQ(ncclSizeOfT<uint64_t>(), 8u);
}

TEST_F(AllocMicrotest, Calloc_ZeroesAndOwnsThroughUniquePtrs) {
  uint32_t* raw = reinterpret_cast<uint32_t*>(this);
  ASSERT_EQ(ncclCalloc(&raw, 0), ncclSuccess);
  EXPECT_EQ(raw, nullptr);
  ASSERT_EQ(ncclCallocQuiet(&raw, 3), ncclSuccess);
  EXPECT_EQ(raw[0] | raw[1] | raw[2], 0u);
  free(raw);

  ncclUniquePtr<uint64_t> one;
  ASSERT_EQ(ncclCalloc(one, 1), ncclSuccess);
  ASSERT_NE(one, nullptr);
  EXPECT_EQ(*one, 0u);
  ncclUniqueArrayPtr<char> many;
  ASSERT_EQ(ncclCalloc(many, 16), ncclSuccess);
  EXPECT_EQ(many[15], 0);
  ASSERT_EQ(ncclCalloc(many, 0), ncclSuccess);
  EXPECT_EQ(many, nullptr);
}

// Frees a block of `bytes` full of 0xff so the next malloc of that size likely reuses dirty memory.
void LeaveDirtyBlock(size_t bytes) {
  void* dirty = std::malloc(bytes);
  std::memset(dirty, 0xff, bytes);
  std::free(dirty);
}

TEST_F(AllocMicrotest, Realloc_GrowsKeepingContentsAndZeroingTheTail) {
  int* p = nullptr;
  ASSERT_EQ(ncclRealloc(&p, 0, 2), ncclSuccess);
  p[0] = 5;
  p[1] = 6;
  LeaveDirtyBlock(64 * sizeof(int));
  ASSERT_EQ(ncclReallocQuiet(&p, 2, 64), ncclSuccess);
  EXPECT_EQ(p[0], 5);
  EXPECT_EQ(p[1], 6);
  for (int i = 2; i < 64; i++) ASSERT_EQ(p[i], 0) << i;
  int* same = p;
  ASSERT_EQ(ncclRealloc(&p, 64, 64), ncclSuccess);
  EXPECT_EQ(p, same);
  EXPECT_EQ(ncclRealloc(&p, 64, 3), ncclInternalError);
  free(p);
  int* none = nullptr;
  EXPECT_EQ(ncclRealloc(&none, 2, 4), ncclInternalError);
}

TEST_F(AllocMicrotest, CallocAndRealloc_ReportMallocFailure) {
  const size_t huge = SIZE_MAX / 16;  // past PTRDIFF_MAX bytes: malloc refuses it
  uint64_t* p = reinterpret_cast<uint64_t*>(this);
  EXPECT_EQ(ncclCalloc(&p, huge), ncclSystemError);
  EXPECT_EQ(p, reinterpret_cast<uint64_t*>(this));
  p = nullptr;
  ASSERT_EQ(ncclRealloc(&p, 0, 1), ncclSuccess);
  uint64_t* kept = p;
  EXPECT_EQ(ncclRealloc(&p, 1, huge), ncclSystemError);
  EXPECT_EQ(p, kept);
  EXPECT_TRUE(Warned("Failed to malloc"));
  free(p);
}

TEST_F(AllocMicrotest, IbMalloc_PageAlignedAndZeroed) {
  void* p = this;
  ASSERT_EQ(ncclIbMalloc(&p, 0), ncclSuccess);
  EXPECT_EQ(p, nullptr);
  void* dirty = nullptr;
  ASSERT_EQ(posix_memalign(&dirty, 4096, 4096), 0);
  std::memset(dirty, 0xff, 4096);
  free(dirty);
  ASSERT_EQ(ncclIbMalloc(&p, 100), ncclSuccess);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % 4096, 0u);
  for (int i = 0; i < 100; i++) ASSERT_EQ(static_cast<char*>(p)[i], 0) << i;
  free(p);
}

// Side streams

TEST_F(AllocMicrotest, ClampStreamPriority_IntoTheDeviceRange) {
  EXPECT_EQ(ncclClampStreamPriority(-5), -3);
  EXPECT_EQ(ncclClampStreamPriority(2), 0);
  EXPECT_EQ(ncclClampStreamPriority(-1), -1);
  failing_ = {"priorityRange"};
  EXPECT_EQ(ncclClampStreamPriority(-1), 0);
  EXPECT_TRUE(Warned("defaulting side-stream priority clamp to 0"));
}

TEST_F(AllocMicrotest, SideStream_CreatedOncePerKeyAndDestroyedWithTheLastScope) {
  ASSERT_EQ(ncclSideStreamAcquire(kDev, -9), ncclSuccess);
  ASSERT_EQ(ncclSideStreamAcquire(kDev, -3), ncclSuccess);  // same key once clamped
  ASSERT_EQ(ncclSideStreamAcquire(kDev), ncclSuccess);
  EXPECT_EQ(priorities_, (std::vector<int>{-3, 0}));
  hipStream_t s = nullptr;
  ASSERT_EQ(getSideStream(&s, -3), ncclSuccess);
  EXPECT_EQ(s, kSideStream);

  ASSERT_EQ(ncclSideStreamRelease(kDev, -3), ncclSuccess);
  EXPECT_TRUE(destroyed_.empty());
  ASSERT_EQ(ncclSideStreamRelease(kDev, -3), ncclSuccess);
  EXPECT_EQ(destroyed_, std::vector<hipStream_t>{kSideStream});
  ASSERT_EQ(getSideStream(&s, -3), ncclSuccess);
  EXPECT_EQ(s, nullptr);
  ASSERT_EQ(getSideStream(&s), ncclSuccess);
  EXPECT_EQ(s, kSideStream);
  ASSERT_EQ(ncclSideStreamRelease(kDev), ncclSuccess);
  EXPECT_TRUE(sideStream.empty());
}

TEST_F(AllocMicrotest, SideStream_KeyedByBusId) {
  ASSERT_EQ(ncclSideStreamAcquire(1), ncclSuccess);
  ASSERT_EQ(ncclSideStreamAcquire(2), ncclSuccess);
  EXPECT_EQ(Count("sideCreate"), 2);
  hipStream_t s = nullptr;
  ASSERT_EQ(getSideStream(&s), ncclSuccess);  // current device kDev has none
  EXPECT_EQ(s, nullptr);
  ASSERT_EQ(ncclSideStreamRelease(1), ncclSuccess);
  ASSERT_EQ(ncclSideStreamRelease(2), ncclSuccess);
}

TEST_F(AllocMicrotest, SideStream_Failures) {
  failing_ = {"sideCreate"};
  EXPECT_EQ(ncclSideStreamAcquire(kDev), ncclUnhandledCudaError);
  EXPECT_TRUE(Warned("StreamCreateWithPriority failed"));
  EXPECT_TRUE(sideStream.empty());

  failing_ = {"streamDestroy"};
  ASSERT_EQ(ncclSideStreamAcquire(kDev), ncclSuccess);
  EXPECT_EQ(ncclSideStreamRelease(kDev), ncclUnhandledCudaError);
  EXPECT_TRUE(Warned("StreamDestroy failed"));
  EXPECT_TRUE(sideStream.empty());  // dropped even though destroy failed

  g_warns.clear();
  EXPECT_EQ(ncclSideStreamRelease(kDev), ncclSuccess);
  EXPECT_TRUE(Warned("was not found for release"));

  g_busIdResult = ncclSystemError;
  hipStream_t s = nullptr;
  EXPECT_EQ(ncclSideStreamAcquire(kDev), ncclSystemError);
  EXPECT_EQ(ncclSideStreamRelease(kDev), ncclSystemError);
  EXPECT_EQ(getSideStream(&s), ncclSystemError);
  g_busIdResult = ncclSuccess;
  failing_ = {"getDevice"};
  EXPECT_EQ(getSideStream(&s), ncclUnhandledCudaError);
}

TEST_F(AllocMicrotest, SideStreamScope_HoldsTheStreamWhileAlive) {
  hipStream_t s = nullptr;
  {
    ncclSideStreamScope scope(kDev);
    EXPECT_TRUE(scope.active);
    ASSERT_EQ(getSideStream(&s), ncclSuccess);
    EXPECT_EQ(s, kSideStream);
  }
  EXPECT_EQ(destroyed_, std::vector<hipStream_t>{kSideStream});
  failing_ = {"sideCreate"};
  {
    ncclSideStreamScope scope(kDev);
    EXPECT_FALSE(scope.active);
  }
  EXPECT_EQ(destroyed_.size(), 1u);
}

// Pinned host memory

TEST_F(AllocMicrotest, HostCalloc_MapsZeroesAndRestoresCaptureMode) {
  uint32_t* p = nullptr;
  ASSERT_EQ(ncclCudaHostCalloc(&p, 4), ncclSuccess);
  EXPECT_EQ(sizes_["hostMalloc"], 16u);
#if defined(HIP_HOST_UNCACHED_MEMORY)
  EXPECT_EQ(flags_, hipHostMallocMapped | hipHostMallocUncached);
#else
  EXPECT_EQ(flags_, hipHostMallocMapped);
#endif
  EXPECT_EQ(p[0] | p[3], 0u);
  EXPECT_TRUE(CaptureModeRestored());
  ASSERT_EQ(ncclCudaHostFree(p), ncclSuccess);
  EXPECT_EQ(Count("hostFree"), 1);
  EXPECT_EQ(ncclCudaHostFree(nullptr), ncclSuccess);
  EXPECT_EQ(Count("hostFree"), 1);

  ASSERT_EQ(ncclCudaHostCalloc(&p, 0), ncclSuccess);
  EXPECT_EQ(p, nullptr);
  EXPECT_EQ(Count("hostMalloc"), 1);
}

TEST_F(AllocMicrotest, HostCalloc_Failures) {
  char* p = reinterpret_cast<char*>(this);
  failing_ = {"hostMalloc"};
  EXPECT_EQ(ncclCudaHostCalloc(&p, 8), ncclUnhandledCudaError);
  EXPECT_EQ(p, nullptr);
  EXPECT_TRUE(Warned("Failed to CUDA host alloc 8 bytes"));
  EXPECT_TRUE(CaptureModeRestored());
  failing_ = {"capture"};
  EXPECT_EQ(ncclCudaHostCalloc(&p, 8), ncclUnhandledCudaError);
  EXPECT_EQ(Count("hostMalloc"), 1);

  failing_ = {"hostFree"};
  ASSERT_EQ((failing_.clear(), ncclCudaHostCalloc(&p, 8)), ncclSuccess);
  failing_ = {"hostFree"};
  EXPECT_EQ(ncclCudaHostFree(p), ncclUnhandledCudaError);
}

#if ROCM_VERSION >= 71200

// In this arm the ncclCuMemHostAlloc macro expands to an ncclCuMemHostAllocDebug nothing defines, so
// these tests call the function by its parenthesized name.
TEST_F(AllocMicrotest, CuMemHostAlloc_MapsHostMemoryForTheDeviceAndTheCpu) {
  void* p = nullptr;
  hipMemGenericAllocationHandle_t h = nullptr;
  ASSERT_EQ((ncclCuMemHostAlloc)(&p, &h, 100), ncclSuccess);
  EXPECT_EQ(p, Va());
  EXPECT_EQ(h, kHandle);
  EXPECT_EQ(prop_.location.type, hipMemLocationTypeHost);
  EXPECT_EQ(prop_.location.id, 0);
  EXPECT_EQ(sizes_["create"], kGran);
  EXPECT_EQ(access_, (std::vector<std::pair<hipMemLocationType, int>>{{hipMemLocationTypeDevice, kDev},
                                                                       {hipMemLocationTypeHost, 0}}));
  ASSERT_EQ(ncclCuMemHostFree(p), ncclSuccess);
  EXPECT_EQ(Count("release"), 2);
  EXPECT_EQ(addressFrees_.back(), std::make_pair(p, segment_));
  EXPECT_EQ(ncclCuMemHostFree(nullptr), ncclSuccess);
}

TEST_F(AllocMicrotest, CuMemHostAlloc_FailuresUnwind) {
  void* untouched = Va(1);
  for (const char* step : {"getDevice", "deviceGet", "numa", "granularity"}) {
    failing_ = {step};
    EXPECT_EQ((ncclCuMemHostAlloc)(&untouched, nullptr, 1), ncclUnhandledCudaError) << step;
    EXPECT_EQ(untouched, Va(1)) << step;
  }
  const std::vector<std::pair<std::string, std::vector<std::string>>> cases = {
      {"create", {}},
      {"reserve", {"release"}},
      {"map", {"addrfree", "release"}},
      {"access", {"unmap", "addrfree", "release"}},
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(c.first);
    failing_ = {c.first};
    calls_.clear();
    void* p = Va();
    EXPECT_EQ((ncclCuMemHostAlloc)(&p, nullptr, 1), ncclUnhandledCudaError);
    EXPECT_EQ(p, nullptr);
    std::vector<std::string> undo(std::find(calls_.begin(), calls_.end(), c.first) + 1, calls_.end());
    undo.erase(std::remove(undo.begin(), undo.end(), "access"), undo.end());
    EXPECT_EQ(undo, c.second);
  }
}

#else

TEST_F(AllocMicrotest, CuMemHost_UnsupportedBeforeRocm712) {
  void* p = nullptr;
  EXPECT_EQ(ncclCuMemHostAlloc(&p, nullptr, 100), ncclInternalError);
  EXPECT_EQ(ncclCuMemHostFree(p), ncclInternalError);
  EXPECT_TRUE(Warned("CUMEM Host is not supported"));
  EXPECT_TRUE(calls_.empty());
}

#endif  // ROCM_VERSION >= 71200

#if ROCM_VERSION >= 70000

// VMM mapping helpers

TEST_F(AllocMicrotest, MapAndSetAccess_GrantsTheGivenDevice) {
  ASSERT_EQ(ncclCuMemMapAndSetAccess(Va(), kGran, kHandle, 6), ncclSuccess);
  EXPECT_EQ(access_, (std::vector<std::pair<hipMemLocationType, int>>{{hipMemLocationTypeDevice, 6}}));
  failing_ = {"access"};
  EXPECT_EQ(ncclCuMemMapAndSetAccess(Va(), kGran, kHandle, 6), ncclUnhandledCudaError);
  failing_ = {"map"};
  EXPECT_EQ(ncclCuMemMapAndSetAccess(Va(), kGran, kHandle, 6), ncclUnhandledCudaError);
  EXPECT_EQ(Count("access"), 2);
}

TEST_F(AllocMicrotest, AllocAddr_MapsAnImportedHandleAndTracksIt) {
  void* p = nullptr;
  hipMemGenericAllocationHandle_t h = kHandle;
  ASSERT_EQ(ncclCuMemAllocAddr(&p, &h, 100), ncclSuccess);
  EXPECT_EQ(p, Va());
  EXPECT_EQ(sizes_["reserve"], kGran);
  EXPECT_EQ(sizes_["map"], kGran);
  EXPECT_EQ(access_, (std::vector<std::pair<hipMemLocationType, int>>{{hipMemLocationTypeDevice, kDev}}));
  EXPECT_EQ(allocTracker[kDev].totalAlloc, 1u);
  EXPECT_EQ(allocTracker[kDev].totalAllocSize, kGran);
}

TEST_F(AllocMicrotest, AllocAddr_FailuresUnwind) {
  const std::vector<std::pair<std::string, std::vector<std::string>>> cases = {
      {"getDevice", {}}, {"props", {}}, {"granularity", {}}, {"reserve", {}},
      {"map", {"addrfree"}}, {"access", {"unmap", "addrfree"}},
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(c.first);
    failing_ = {c.first};
    calls_.clear();
    void* p = Va();
    hipMemGenericAllocationHandle_t h = kHandle;
    EXPECT_EQ(ncclCuMemAllocAddr(&p, &h, 1), ncclUnhandledCudaError);
    std::vector<std::string> undo(std::find(calls_.begin(), calls_.end(), c.first) + 1, calls_.end());
    EXPECT_EQ(undo, c.second);
    if (c.second.size() > 0 || c.first == "reserve") EXPECT_EQ(p, nullptr);
  }
  EXPECT_EQ(allocTracker[kDev].totalAlloc, 0u);
}

TEST_F(AllocMicrotest, FreeAddr_UnmapsEverySegmentThenFreesTheRange) {
  ranges_[Va()] = {Va(), kGran};
  ranges_[Va(kGran)] = {Va(kGran), 3 * kGran};
  allocTracker[kDev].totalAlloc = 1;
  allocTracker[kDev].totalAllocSize = 4 * kGran;
  ASSERT_EQ(ncclCuMemFreeAddr(Va(), nullptr, 2), ncclSuccess);
  EXPECT_EQ(unmaps_, (std::vector<std::pair<void*, size_t>>{{Va(), kGran}, {Va(kGran), 3 * kGran}}));
  EXPECT_EQ(addressFrees_, (std::vector<std::pair<void*, size_t>>{{Va(), 4 * kGran}}));
  EXPECT_EQ(Count("release"), 0);  // an imported mapping: the owner releases the handle
  EXPECT_EQ(allocTracker[kDev].totalAlloc, 0u);
  EXPECT_EQ(allocTracker[kDev].totalAllocSize, 0u);
  EXPECT_EQ(ncclCuMemFreeAddr(nullptr, nullptr), ncclSuccess);
}

TEST_F(AllocMicrotest, FreeAddr_UntracksThroughTheManager) {
  ncclMemManager mgr{};
  g_untrackInfo = {ncclMemPersist, ncclDynMemStateActive, 0};
  ASSERT_EQ(ncclCuMemFreeAddr(Va(), &mgr), ncclSuccess);
  EXPECT_EQ(g_untrackedPersist, (std::vector<std::pair<void*, size_t>>{{Va(), segment_}}));

  // A dynamic entry Suspend already released: only the reservation is left to free.
  calls_.clear();
  g_untrackInfo = {ncclMemScratch, ncclDynMemStateReleased, 5 * kGran};
  ASSERT_EQ(ncclCuMemFreeAddr(Va(), &mgr), ncclSuccess);
  EXPECT_EQ(calls_, (std::vector<std::string>{"addrfree", "getDevice"}));
  EXPECT_EQ(addressFrees_.back(), std::make_pair(Va(), 5 * kGran));

  // An active dynamic entry is unmapped like persistent memory, but not untracked as persistent.
  unmaps_.clear();
  g_untrackInfo = {ncclMemOffload, ncclDynMemStateActive, 5 * kGran};
  ASSERT_EQ(ncclCuMemFreeAddr(Va(), &mgr), ncclSuccess);
  EXPECT_EQ(unmaps_.size(), 1u);
  EXPECT_EQ(g_untrackedPersist.size(), 1u);
}

TEST_F(AllocMicrotest, FreeAddr_SkipsWhatSuspendReleased) {
  ncclMemManager mgr{};
  ncclDynMemEntry entry{};
  entry.ptr = Va();
  entry.state = ncclDynMemStateReleased;
  mgr.entries = &entry;
  mgr.released = 1;
  EXPECT_EQ(ncclCuMemFreeAddr(Va(), &mgr), ncclSuccess);
  EXPECT_EQ(ncclCuMemFree(Va(), &mgr), ncclSuccess);
  EXPECT_EQ(ncclCudaFree(static_cast<char*>(Va()), &mgr), ncclSuccess);
  for (const char* call : {"free", "retain", "range", "unmap", "release", "addrfree"}) EXPECT_EQ(Count(call), 0);
}

// BUG: when a step fails, the shared exit still calls cuMemAddressFree with only the size of the
// segments already unmapped -- zero if the first one failed -- on a range that may still be mapped.
TEST_F(AllocMicrotest, FreeAddr_FailureFreesTheRangeWithThePartialSize) {
  ranges_[Va()] = {Va(), kGran};
  failing_ = {"range"};
  EXPECT_EQ(ncclCuMemFreeAddr(Va(), nullptr), ncclUnhandledCudaError);
  EXPECT_EQ(addressFrees_.back(), std::make_pair(Va(), (size_t)0));

  failing_ = {"unmap"};
  EXPECT_EQ(ncclCuMemFreeAddr(Va(), nullptr), ncclUnhandledCudaError);
  EXPECT_EQ(addressFrees_.back(), std::make_pair(Va(), (size_t)0));

  ncclMemManager mgr{};
  failing_.clear();
  g_untrackResult = ncclInternalError;
  EXPECT_EQ(ncclCuMemFreeAddr(Va(), &mgr), ncclInternalError);
  EXPECT_EQ(addressFrees_.back(), std::make_pair(Va(), (size_t)0));
  failing_ = {"addrfree"};
  g_untrackResult = ncclSuccess;
  EXPECT_EQ(ncclCuMemFreeAddr(Va(), nullptr), ncclUnhandledCudaError);
}

// cuMem allocation

TEST_F(AllocMicrotest, CuMemAlloc_ZeroesTheMappingOnAPrivateStream) {
  void* p = nullptr;
  hipMemGenericAllocationHandle_t h = nullptr;
  ASSERT_EQ(ncclCuMemAlloc(&p, &h, kFabricHandle, 100), ncclSuccess);
  EXPECT_EQ(p, Va());
  EXPECT_EQ(h, kHandle);
  EXPECT_EQ(createProp_.type, kVmmType);
  EXPECT_EQ(createProp_.location.type, hipMemLocationTypeDevice);
  EXPECT_EQ(createProp_.location.id, kHipDev);
  EXPECT_EQ(createProp_.requestedHandleType, kFabricHandle);
  EXPECT_EQ(createProp_.allocFlags.gpuDirectRDMACapable, 1);
  EXPECT_EQ(sizes_["create"], kGran);
  EXPECT_EQ(access_, (std::vector<std::pair<hipMemLocationType, int>>{{hipMemLocationTypeDevice, kHipDev}}));
  EXPECT_EQ(memsets_, (std::vector<std::pair<void*, size_t>>{{Va(), kGran}}));
  EXPECT_EQ(memsetStreams_, std::vector<hipStream_t>{kPrivateStream});
  EXPECT_EQ(synced_, std::vector<hipStream_t>{kPrivateStream});
  EXPECT_EQ(destroyed_, std::vector<hipStream_t>{kPrivateStream});
  EXPECT_TRUE(CaptureModeRestored());
  EXPECT_TRUE(g_tracked.empty());
  EXPECT_EQ(allocTracker[kDev].totalAlloc, 1u);
  EXPECT_EQ(allocTracker[kDev].totalAllocSize, kGran);
}

TEST_F(AllocMicrotest, CuMemAlloc_UsesTheSideStreamAndTracksWithTheManager) {
  ncclMemManager mgr{};
  ncclSideStreamScope scope(kDev);
  void* p = nullptr;
  ASSERT_EQ(ncclCuMemAlloc(&p, nullptr, hipMemHandleTypePosixFileDescriptor, kGran, &mgr, ncclMemScratch),
            ncclSuccess);
  EXPECT_EQ(memsetStreams_, std::vector<hipStream_t>{kSideStream});
  EXPECT_EQ(Count("streamCreate"), 0);
  EXPECT_TRUE(destroyed_.empty());
  ASSERT_EQ(g_tracked.size(), 1u);
  EXPECT_EQ(g_tracked[0].ptr, Va());
  EXPECT_EQ(g_tracked[0].size, kGran);
  EXPECT_EQ(g_tracked[0].type, hipMemHandleTypePosixFileDescriptor);
  EXPECT_EQ(g_tracked[0].memType, ncclMemScratch);
}

TEST_F(AllocMicrotest, CuMemAlloc_FailuresUnwindTheAllocation) {
  ncclMemManager mgr{};
  const std::vector<std::pair<std::string, std::vector<std::string>>> cases = {
      {"getDevice", {}},
      {"deviceGet", {}},
      {"granularity", {}},
      {"create", {}},
      {"reserve", {"release"}},
      {"map", {"addrfree", "release"}},
      {"access", {"unmap", "addrfree", "release"}},
      {"capture", {"unmap", "addrfree", "release"}},
      {"streamCreate", {"capture", "unmap", "addrfree", "release"}},
      {"memset", {"streamDestroy", "capture", "unmap", "addrfree", "release"}},
      {"sync", {"streamDestroy", "capture", "unmap", "addrfree", "release"}},
      {"track", {"unmap", "addrfree", "release"}},
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(c.first);
    failing_ = {c.first};
    g_trackResult = c.first == "track" ? ncclInternalError : ncclSuccess;
    calls_.clear();
    void* p = Va(1);
    ncclResult_t ret = ncclCuMemAlloc(&p, nullptr, hipMemHandleTypePosixFileDescriptor, 1, &mgr);
    EXPECT_EQ(ret, c.first == "track" ? ncclInternalError : ncclUnhandledCudaError);
    auto from = c.first == "track" ? std::find(calls_.begin(), calls_.end(), "streamDestroy") + 2
                                   : std::find(calls_.begin(), calls_.end(), c.first) + 1;
    std::vector<std::string> undo(from, calls_.end());
    EXPECT_EQ(undo, c.second);
    if (!c.second.empty() || c.first == "create") EXPECT_EQ(p, nullptr);
  }
  EXPECT_EQ(allocTracker[kDev].totalAlloc, 0u);
}

TEST_F(AllocMicrotest, CuMemAlloc_SideStreamLookupFailureUnwinds) {
  g_busIdResult = ncclSystemError;
  void* p = nullptr;
  EXPECT_EQ(ncclCuMemAlloc(&p, nullptr, hipMemHandleTypePosixFileDescriptor, 1), ncclSystemError);
  EXPECT_EQ(p, nullptr);
  EXPECT_TRUE(CaptureModeRestored());
  EXPECT_EQ(Count("release"), 1);
}

// cuMem free

TEST_F(AllocMicrotest, CuMemFree_ReleasesEachSegmentAndUntracksPersist) {
  ncclMemManager mgr{};
  ranges_[Va()] = {Va(), kGran};
  ranges_[Va(kGran)] = {Va(kGran), kGran};
  allocTracker[kDev].totalAlloc = 1;
  allocTracker[kDev].totalAllocSize = 2 * kGran;
  ASSERT_EQ(ncclCuMemFree(Va(), &mgr, 2), ncclSuccess);
  EXPECT_EQ(calls_, (std::vector<std::string>{"retain", "release", "range", "unmap", "release", "retain", "release",
                                              "range", "unmap", "release", "addrfree", "getDevice"}));
  EXPECT_EQ(addressFrees_, (std::vector<std::pair<void*, size_t>>{{Va(), 2 * kGran}}));
  EXPECT_EQ(g_untrackedPersist, (std::vector<std::pair<void*, size_t>>{{Va(), 2 * kGran}}));
  EXPECT_EQ(allocTracker[kDev].totalAlloc, 0u);
  EXPECT_EQ(allocTracker[kDev].totalAllocSize, 0u);
  EXPECT_EQ(ncclCuMemFree(nullptr, &mgr), ncclSuccess);
}

TEST_F(AllocMicrotest, CuMemFree_SuspendedDynamicEntryOnlyFreesTheRange) {
  ncclMemManager mgr{};
  g_untrackInfo = {ncclMemOffload, ncclDynMemStateReleased, 3 * kGran};
  ASSERT_EQ(ncclCuMemFree(Va(), &mgr), ncclSuccess);
  EXPECT_EQ(calls_, (std::vector<std::string>{"addrfree", "getDevice"}));
  EXPECT_EQ(addressFrees_.back(), std::make_pair(Va(), 3 * kGran));
  EXPECT_TRUE(g_untrackedPersist.empty());
}

TEST_F(AllocMicrotest, CuMemFree_FailuresStillFreeTheRange) {
  for (const char* step : {"retain", "release", "range", "unmap"}) {
    SCOPED_TRACE(step);
    failing_ = {step};
    calls_.clear();
    EXPECT_EQ(ncclCuMemFree(Va(), nullptr), ncclUnhandledCudaError);
    EXPECT_EQ(addressFrees_.back(), std::make_pair(Va(), (size_t)0));
  }
  ncclMemManager mgr{};
  failing_.clear();
  g_untrackResult = ncclInternalError;
  EXPECT_EQ(ncclCuMemFree(Va(), &mgr), ncclInternalError);
  failing_ = {"addrfree"};
  g_untrackResult = ncclSuccess;
  EXPECT_EQ(ncclCuMemFree(Va(), nullptr), ncclUnhandledCudaError);
}

// cuMemGetAddressRange over several segments

TEST_F(AllocMicrotest, GetAddressRange_SpansEverySegmentOfTheBuffer) {
  ranges_[Va(kGran)] = {Va(), 2 * kGran};
  ranges_[Va(2 * kGran)] = {Va(2 * kGran), kGran};
  hipDeviceptr_t base = nullptr;
  size_t total = 0;
  int segments = 0;
  ASSERT_EQ(ncclCuMemGetAddressRange(Va(kGran), 2 * kGran, &base, &total, &segments), ncclSuccess);
  EXPECT_EQ(base, Va());
  EXPECT_EQ(total, 3 * kGran);
  EXPECT_EQ(segments, 2);
  ASSERT_EQ(ncclCuMemGetAddressRange(Va(kGran), 1, &base, &total, nullptr), ncclSuccess);
  EXPECT_EQ(total, 2 * kGran);
  failing_ = {"range"};
  EXPECT_EQ(ncclCuMemGetAddressRange(Va(kGran), 1, &base, &total, &segments), ncclUnhandledCudaError);
}

// Some drivers report a zero base for CPU-backed ranges: the walk then advances from the buffer.
TEST_F(AllocMicrotest, GetAddressRange_ZeroBaseFallsBackToTheBuffer) {
  ranges_[Va()] = {nullptr, kGran};
  ranges_[Va(kGran)] = {nullptr, kGran};
  hipDeviceptr_t base = nullptr;
  size_t total = 0;
  int segments = 0;
  ASSERT_EQ(ncclCuMemGetAddressRange(Va(), 2 * kGran, &base, &total, &segments), ncclSuccess);
  EXPECT_EQ(base, Va());
  EXPECT_EQ(total, 2 * kGran);
  EXPECT_EQ(segments, 2);
}

TEST_F(AllocMicrotest, GetAddressRange_ProbesForSysmemUntilFound) {
  ranges_[Va()] = {Va(), kGran};
  ranges_[Va(kGran)] = {Va(kGran), kGran};
  hipDeviceptr_t base = nullptr;
  size_t total = 0;
  bool sysmem = true;
  ASSERT_EQ(ncclCuMemGetAddressRange(Va(), 2 * kGran, &base, &total, nullptr, &sysmem), ncclSuccess);
  EXPECT_FALSE(sysmem);
  EXPECT_EQ(Count("retain"), 2);
  EXPECT_EQ(Count("release"), 2);

  calls_.clear();
  handleLocation_ = hipMemLocationTypeHost;
  ASSERT_EQ(ncclCuMemGetAddressRange(Va(), 2 * kGran, &base, &total, nullptr, &sysmem), ncclSuccess);
#if ROCM_VERSION >= 71200
  EXPECT_TRUE(sysmem);
  EXPECT_EQ(Count("retain"), 1);  // no more probing once a host segment is seen
#else
  EXPECT_FALSE(sysmem);  // host VMM segments only exist from ROCm 7.12
#endif
  for (const char* step : {"retain", "props", "release"}) {
    failing_ = {step};
    EXPECT_EQ(ncclCuMemGetAddressRange(Va(), kGran, &base, &total, nullptr, &sysmem), ncclUnhandledCudaError) << step;
  }
}

// BUG: if reading the retained handle's properties fails, the handle reference taken to probe for
// sysmem is never released.
TEST_F(AllocMicrotest, GetAddressRange_PropertiesFailureLeaksTheRetainedHandle) {
  hipDeviceptr_t base = nullptr;
  size_t total = 0;
  bool sysmem = false;
  failing_ = {"props"};
  EXPECT_EQ(ncclCuMemGetAddressRange(Va(), kGran, &base, &total, nullptr, &sysmem), ncclUnhandledCudaError);
  EXPECT_EQ(Count("retain"), 1);
  EXPECT_EQ(Count("release"), 0);
}

// ncclCudaMalloc / ncclCudaCalloc / ncclCudaCallocAsync

TEST_F(AllocMicrotest, CudaMalloc_WithoutCuMemUsesExtMalloc) {
  uint32_t* p = nullptr;
  ASSERT_EQ(ncclCudaMalloc(&p, 4, nullptr, ncclMemPersist, hipDeviceMallocUncached), ncclSuccess);
  EXPECT_EQ(sizes_["extMalloc"], 16u);
  EXPECT_EQ(flags_, (unsigned)hipDeviceMallocUncached);
  EXPECT_TRUE(CaptureModeRestored());
  EXPECT_EQ(allocTracker[kDev].totalAlloc, 1u);
  EXPECT_EQ(allocTracker[kDev].totalAllocSize, 16u);
  ranges_[p] = {p, 16};
  ASSERT_EQ(ncclCudaFree(p, nullptr), ncclSuccess);
  EXPECT_EQ(Count("free"), 1);
  EXPECT_EQ(allocTracker[kDev].totalAlloc, 0u);
  EXPECT_EQ(allocTracker[kDev].totalAllocSize, 0u);
}

TEST_F(AllocMicrotest, CudaCalloc_WithoutCuMemZeroesOnAStream) {
  uint32_t* p = nullptr;
  ASSERT_EQ(ncclCudaCalloc(&p, 4, nullptr), ncclSuccess);
  EXPECT_EQ(p[3], 0u);
  EXPECT_EQ(memsetStreams_, std::vector<hipStream_t>{kPrivateStream});
  EXPECT_EQ(destroyed_, std::vector<hipStream_t>{kPrivateStream});
  {
    ncclSideStreamScope scope(kDev);
    uint32_t* q = nullptr;
    ASSERT_EQ(ncclCudaCalloc(&q, 4, nullptr), ncclSuccess);
    EXPECT_EQ(memsetStreams_.back(), kSideStream);
    EXPECT_EQ(Count("streamCreate"), 1);
    ranges_[q] = {q, 16};
    ASSERT_EQ(ncclCudaFree(q, nullptr), ncclSuccess);
  }
  EXPECT_TRUE(CaptureModeRestored());
  ranges_[p] = {p, 16};
  ASSERT_EQ(ncclCudaFree(p, nullptr), ncclSuccess);
}

TEST_F(AllocMicrotest, CudaCallocAsync_WithoutCuMemZeroesOnTheCallersStream) {
  uint32_t* p = nullptr;
  ASSERT_EQ(ncclCudaCallocAsync(&p, 4, kSideStream, nullptr), ncclSuccess);
  EXPECT_EQ(memsetStreams_, std::vector<hipStream_t>{kSideStream});
  EXPECT_EQ(Count("sync"), 0);
  EXPECT_TRUE(CaptureModeRestored());
  EXPECT_EQ(allocTracker[kDev].totalAlloc, 1u);
  ranges_[p] = {p, 16};
  ASSERT_EQ(ncclCudaFree(p, nullptr), ncclSuccess);
}

// With cuMem every helper goes through ncclCuMemAlloc, passing the manager and memory type on.
TEST_F(AllocMicrotest, CudaAllocs_WithCuMemDispatchToCuMemAlloc) {
  g_cuMemEnable = 1;
  ncclCuMemHandleType = kFabricHandle;
  ncclMemManager mgr{};
  char* a = nullptr;
  char* b = nullptr;
  char* c = nullptr;
  ASSERT_EQ(ncclCudaMalloc(&a, 10, &mgr, ncclMemScratch), ncclSuccess);
  ASSERT_EQ(ncclCudaCalloc(&b, 10, &mgr, ncclMemOffload), ncclSuccess);
  ASSERT_EQ(ncclCudaCallocAsync(&c, 10, kSideStream, &mgr), ncclSuccess);
  ASSERT_EQ(g_tracked.size(), 3u);
  EXPECT_EQ(g_tracked[0].memType, ncclMemScratch);
  EXPECT_EQ(g_tracked[1].memType, ncclMemOffload);
  EXPECT_EQ(g_tracked[2].memType, ncclMemPersist);
  for (const TrackCall& t : g_tracked) EXPECT_EQ(t.type, kFabricHandle);
  EXPECT_EQ(Count("extMalloc"), 0);
  EXPECT_EQ(Count("memset"), 3);  // ncclCuMemAlloc's own zeroing only
}

TEST_F(AllocMicrotest, CudaAllocs_FailuresWarnAndReturnNull) {
  char* p = reinterpret_cast<char*>(this);
  failing_ = {"extMalloc"};
  EXPECT_EQ(ncclCudaMalloc(&p, 8, nullptr), ncclUnhandledCudaError);
  EXPECT_EQ(p, nullptr);
  EXPECT_TRUE(Warned("Failed to CUDA malloc 8 bytes"));
  EXPECT_EQ(ncclCudaCallocAsync(&p, 8, kSideStream, nullptr), ncclUnhandledCudaError);
  EXPECT_TRUE(Warned("Failed to CUDA calloc async 8 bytes"));
  EXPECT_TRUE(CaptureModeRestored());
  g_cuMemEnable = 1;
  failing_ = {"create"};
  EXPECT_EQ(ncclCudaMalloc(&p, 8, nullptr), ncclUnhandledCudaError);
  EXPECT_EQ(ncclCudaCalloc(&p, 8, nullptr), ncclUnhandledCudaError);
  EXPECT_TRUE(Warned("Failed to CUDA calloc 8 bytes"));
  EXPECT_EQ(ncclCudaCallocAsync(&p, 8, kSideStream, nullptr), ncclUnhandledCudaError);
  failing_ = {"capture"};
  EXPECT_EQ(ncclCudaMalloc(&p, 8, nullptr), ncclUnhandledCudaError);
  EXPECT_EQ(ncclCudaCalloc(&p, 8, nullptr), ncclUnhandledCudaError);
  EXPECT_EQ(ncclCudaCallocAsync(&p, 8, kSideStream, nullptr), ncclUnhandledCudaError);
  EXPECT_EQ(allocTracker[kDev].totalAlloc, 0u);
}

TEST_F(AllocMicrotest, CudaCalloc_ZeroingFailuresWithoutCuMem) {
  for (const char* step : {"extMalloc", "memset", "sync", "streamDestroy"}) {
    SCOPED_TRACE(step);
    failing_ = {step};
    char* p = nullptr;
    EXPECT_EQ(ncclCudaCalloc(&p, 8, nullptr), ncclUnhandledCudaError);
    EXPECT_TRUE(CaptureModeRestored());
    Unheap(p);
  }
  char* p = nullptr;
  failing_ = {"memset"};
  EXPECT_EQ(ncclCudaCallocAsync(&p, 8, kSideStream, nullptr), ncclUnhandledCudaError);
  Unheap(p);
  failing_ = {"streamCreate"};
  EXPECT_EQ(ncclCudaCalloc(&p, 8, nullptr), ncclUnhandledCudaError);
  Unheap(p);
}

// BUG: a zero-element allocation allocates nothing but still counts one allocation, and freeing the
// null pointer it returns takes nothing back off.
TEST_F(AllocMicrotest, CudaAllocs_ZeroElementsStillCountAnAllocation) {
  char* p = reinterpret_cast<char*>(this);
  ASSERT_EQ(ncclCudaMalloc(&p, 0, nullptr), ncclSuccess);
  EXPECT_EQ(p, nullptr);
  ASSERT_EQ(ncclCudaCalloc(&p, 0, nullptr), ncclSuccess);
  ASSERT_EQ(ncclCudaCallocAsync(&p, 0, kSideStream, nullptr), ncclSuccess);
  EXPECT_EQ(allocTracker[kDev].totalAlloc, 3u);
  EXPECT_EQ(Count("extMalloc"), 0);
  ASSERT_EQ(ncclCudaFree(p, nullptr), ncclSuccess);
  EXPECT_EQ(allocTracker[kDev].totalAlloc, 3u);
}

// BUG: with cuMem both ncclCudaMalloc and ncclCuMemAlloc count the allocation, by requested and by
// granularity-aligned size, and ncclCudaFree and ncclCuMemFree both take the mapped size back off:
// the count balances but the byte total drifts by the alignment padding on every allocation.
TEST_F(AllocMicrotest, CudaAllocs_CuMemCountsEveryAllocationTwice) {
  g_cuMemEnable = 1;
  char* p = nullptr;
  ASSERT_EQ(ncclCudaMalloc(&p, 100, nullptr), ncclSuccess);
  EXPECT_EQ(allocTracker[kDev].totalAlloc, 2u);
  EXPECT_EQ(allocTracker[kDev].totalAllocSize, 100 + kGran);
  ranges_[p] = {p, kGran};
  ASSERT_EQ(ncclCudaFree(p, nullptr), ncclSuccess);
  EXPECT_EQ(allocTracker[kDev].totalAlloc, 0u);
  EXPECT_EQ((int64_t)allocTracker[kDev].totalAllocSize, (int64_t)(100 - kGran));
}

// BUG: on a zeroing failure the buffer is neither freed nor returned null, so the caller cannot
// free it, and the allocation is counted as if it had succeeded; a private zeroing stream leaks too.
TEST_F(AllocMicrotest, CudaCalloc_ZeroingFailureLeaksTheBuffer) {
  failing_ = {"memset"};
  char* p = nullptr;
  EXPECT_EQ(ncclCudaCalloc(&p, 8, nullptr), ncclUnhandledCudaError);
  EXPECT_NE(p, nullptr);
  EXPECT_EQ(Count("free"), 0);
  EXPECT_TRUE(destroyed_.empty());
  EXPECT_EQ(allocTracker[kDev].totalAlloc, 1u);
  Unheap(p);
}

// BUG: a failed side-stream lookup returns with the thread still switched to relaxed capture mode.
TEST_F(AllocMicrotest, CudaCalloc_SideStreamFailureLeavesTheCaptureModeSwitched) {
  g_busIdResult = ncclSystemError;
  char* p = nullptr;
  EXPECT_EQ(ncclCudaCalloc(&p, 8, nullptr), ncclSystemError);
  EXPECT_EQ(Count("capture"), 1);
  calls_.clear();
  EXPECT_EQ(ncclCudaMemcpy(p, p, 8), ncclSystemError);
  EXPECT_EQ(Count("capture"), 1);
}

// Copies

TEST_F(AllocMicrotest, CudaMemcpy_CopiesOnASideOrPrivateStreamAndWaits) {
  uint32_t a[4], b[4];
  ASSERT_EQ(ncclCudaMemcpy(a, b, 4), ncclSuccess);
  EXPECT_EQ(sizes_["memcpy"], 16u);
  EXPECT_EQ(synced_, std::vector<hipStream_t>{kPrivateStream});
  EXPECT_EQ(destroyed_, std::vector<hipStream_t>{kPrivateStream});
  {
    ncclSideStreamScope scope(kDev);
    ASSERT_EQ(ncclCudaMemcpy(a, b, 4), ncclSuccess);
    EXPECT_EQ(synced_, (std::vector<hipStream_t>{kPrivateStream, kSideStream}));
    EXPECT_EQ(destroyed_.size(), 1u);
  }
  EXPECT_TRUE(CaptureModeRestored());
  ASSERT_EQ(ncclCudaMemcpyAsync(a, b, 2, kSideStream), ncclSuccess);
  EXPECT_EQ(sizes_["memcpy"], 8u);
  EXPECT_EQ(memsetStreams_.back(), kSideStream);
}

TEST_F(AllocMicrotest, CudaMemcpy_Failures) {
  uint32_t a[4], b[4];
  for (const char* step : {"streamCreate", "sync", "streamDestroy"}) {
    failing_ = {step};
    EXPECT_EQ(ncclCudaMemcpy(a, b, 4), ncclUnhandledCudaError) << step;
    EXPECT_TRUE(CaptureModeRestored()) << step;
  }
  failing_ = {"capture"};
  EXPECT_EQ(ncclCudaMemcpy(a, b, 4), ncclUnhandledCudaError);
  EXPECT_EQ(ncclCudaMemcpyAsync(a, b, 4, kSideStream), ncclUnhandledCudaError);
  failing_ = {"memcpy"};
  EXPECT_EQ(ncclCudaMemcpyAsync(a, b, 4, kSideStream), ncclUnhandledCudaError);
  EXPECT_TRUE(CaptureModeRestored());
}

// BUG: when the copy itself fails, the private stream ncclCudaMemcpy created is never destroyed.
TEST_F(AllocMicrotest, CudaMemcpy_CopyFailureLeaksThePrivateStream) {
  uint32_t a[4], b[4];
  failing_ = {"memcpy"};
  EXPECT_EQ(ncclCudaMemcpy(a, b, 4), ncclUnhandledCudaError);
  EXPECT_EQ(Count("streamCreate"), 1);
  EXPECT_TRUE(destroyed_.empty());
}

// ncclCudaFree

TEST_F(AllocMicrotest, CudaFree_TracksOnlyBaseAddresses) {
  char* p = static_cast<char*>(Va(16));
  ranges_[p] = {Va(), kGran};
  allocTracker[kDev].totalAlloc = 1;
  ASSERT_EQ(ncclCudaFree(p, nullptr), ncclSuccess);
  EXPECT_EQ(allocTracker[kDev].totalAlloc, 1u);  // an interior pointer is not the allocation
  EXPECT_EQ(Count("free"), 1);
  EXPECT_EQ(ncclCudaFree(static_cast<char*>(nullptr), nullptr), ncclSuccess);
}

TEST_F(AllocMicrotest, CudaFree_Failures) {
  char* p = static_cast<char*>(Va());
  failing_ = {"range"};
  EXPECT_EQ(ncclCudaFree(p, nullptr), ncclUnhandledCudaError);
  EXPECT_EQ(Count("capture"), 0);
  for (const char* step : {"capture", "free"}) {
    failing_ = {step};
    EXPECT_EQ(ncclCudaFree(p, nullptr), ncclUnhandledCudaError) << step;
  }
  failing_.clear();
  calls_.clear();
  EXPECT_EQ(ncclCudaFree(p, nullptr, 2), ncclUnhandledCudaError);  // segments need cuMem
  EXPECT_EQ(Count("free"), 0);
  EXPECT_TRUE(CaptureModeRestored());
  g_cuMemEnable = 1;
  failing_ = {"retain"};
  EXPECT_EQ(ncclCudaFree(p, nullptr), ncclUnhandledCudaError);
  EXPECT_TRUE(CaptureModeRestored());
}

#else  // ROCM_VERSION < 70000

TEST_F(AllocMicrotest, CuMem_UnsupportedBeforeRocm7) {
  void* p = nullptr;
  hipMemGenericAllocationHandle_t h = nullptr;
  hipDeviceptr_t base = nullptr;
  size_t total = 0;
  EXPECT_EQ(ncclCuMemAlloc(&p, nullptr, 0, 1, nullptr), ncclInternalError);
  EXPECT_EQ(ncclCuMemFree(p, nullptr), ncclInternalError);
  EXPECT_EQ(ncclCuMemAllocAddr(&p, &h, 1), ncclInternalError);
  EXPECT_EQ(ncclCuMemFreeAddr(p, nullptr), ncclInternalError);
  EXPECT_EQ(ncclCuMemGetAddressRange(base, 1, &base, &total, nullptr), ncclInternalError);
  EXPECT_EQ(ncclCuMemMapAndSetAccess(p, 1, h, 0), ncclInternalError);
  EXPECT_TRUE(Warned("CUMEM requires ROCM_VERSION >= 7.0.0"));
}

#endif  // ROCM_VERSION >= 70000

}  // namespace
