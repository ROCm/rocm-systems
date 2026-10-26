/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/allocator.cc: the ncclMemAlloc/ncclMemFree VMM paths, the ncclSpace
// offset allocator and the ncclShadowPool device/host object pool. HIP goes through hip_fakes.

#include <gtest/gtest.h>

#include <algorithm>
#include <csignal>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "comm.h"
#include "allocator.h"
#include "debug.h"
#include "ExpectedCrash.h"
#include "ScopedHook.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/rocmwrap_fakes.h"

namespace {
bool g_hashToZero = false;
}  // namespace

// utils.cc's hash; with hbits == 0 it returns bucket 0 instead of shifting by 64.
uint64_t ncclHashPointer(int hbits, void* key) {
  if (g_hashToZero || hbits == 0) return 0;
  uintptr_t h = reinterpret_cast<uintptr_t>(key);
  h ^= h >> 32;
  h *= 0x9e3779b97f4a7c13;
  return (uint64_t)h >> (64 - hbits);
}

#include "fakes/param_redirect.h"
#include "fakes/nvtx_redirect.h"
#include ALLOCATOR_CC_PATH

struct allocationTracker allocTracker[MAX_ALLOC_TRACK_NGPU] = {};

// hipify renames the CUDA FABRIC names wherever HIP has them; p2p.h only defines them otherwise.
#ifdef CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED
const hipDeviceAttribute_t kFabricSupportedAttr = (hipDeviceAttribute_t)CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED;
#else
const hipDeviceAttribute_t kFabricSupportedAttr = hipDeviceAttributeHandleTypeFabricSupported;
#endif

namespace {

constexpr int kDev = 3;
constexpr hipDevice_t kHipDev = 13;
constexpr size_t kGran = 1 << 16;
const hipStream_t kStream = reinterpret_cast<hipStream_t>(0x5);
const hipMemPool_t kPool = reinterpret_cast<hipMemPool_t>(0x7);

hipMemGenericAllocationHandle_t Handle(uintptr_t v) {
  return reinterpret_cast<hipMemGenericAllocationHandle_t>(v);
}

class AllocatorMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetHipFakes();
    ResetNcclFakes();
    ResetRocmWrapFakes();
    g_hashToZero = false;
    params_.clear();
    g_loadParam = [this](const char* env, int64_t deftVal) {
      auto it = params_.find(env);
      return it == params_.end() ? deftVal : it->second;
    };
    g_currentDevice = kDev;
    g_deviceCount = 4;
    g_hipDeviceGet = [](hipDevice_t* d, int ordinal) {
      EXPECT_EQ(ordinal, kDev);
      *d = kHipDev;
      return hipSuccess;
    };
    g_hipMalloc = [this](void** p, size_t n) {
      mallocs_.push_back(n);
      if (mallocResult_ != hipSuccess) return mallocResult_;
      *p = std::malloc(n);
      allocated_.push_back(*p);
      return hipSuccess;
    };
    g_hipFree = [this](void* p) {
      frees_.push_back(p);
      Release(p);
      return hipSuccess;
    };
    g_hipDeviceGetAttribute = [this](int* v, hipDeviceAttribute_t attr, int dev) {
      attrs_.push_back(attr);
      *v = attr == hipDeviceAttributeMemoryPoolsSupported ? poolsSupported_ : fabricSupported_;
      return attrResult_;
    };
    g_hipMemPoolCreate = [this](hipMemPool_t* pool, const hipMemPoolProps* props) {
      poolProps_.push_back(*props);
      *pool = kPool;
      return poolCreateResult_;
    };
    g_hipMemPoolDestroy = [this](hipMemPool_t pool) {
      EXPECT_EQ(pool, kPool);
      poolDestroys_++;
      return hipSuccess;
    };
    g_hipMallocFromPoolAsync = [this](void** p, size_t n, hipMemPool_t pool, hipStream_t stream) {
      EXPECT_EQ(pool, kPool);
      EXPECT_EQ(stream, kStream);
      mallocs_.push_back(n);
      if (mallocResult_ != hipSuccess) {
        *p = nullptr;
        return mallocResult_;
      }
      *p = std::malloc(n);
      allocated_.push_back(*p);
      return hipSuccess;
    };
    g_hipFreeAsync = [this](void* p, hipStream_t stream) {
      EXPECT_EQ(stream, kStream);
      asyncFrees_.push_back(p);
      Release(p);
      return hipSuccess;
    };
    g_hipMemsetAsync = [this](void* p, int v, size_t n, hipStream_t) {
      memsets_.push_back(n);
      if (memsetResult_ == hipSuccess) std::memset(p, v, n);
      return memsetResult_;
    };
    g_hipStreamSynchronize = [this](hipStream_t stream) {
      EXPECT_EQ(stream, kStream);
      syncs_++;
      return hipSuccess;
    };
  }

  void TearDown() override {
    EXPECT_TRUE(allocated_.empty()) << allocated_.size() << " fake device allocations leaked";
    for (void* p : allocated_) std::free(p);
    ResetHipFakes();
    ResetNcclFakes();  // g_loadParam captures this fixture
  }

  // Frees only what the fakes handed out, so a stray free of a made-up address fails the test.
  void Release(void* p) {
    if (p == nullptr) return;
    auto it = std::find(allocated_.begin(), allocated_.end(), p);
    if (it == allocated_.end()) {
      ADD_FAILURE() << "free of " << p << ", which no fake allocated";
      return;
    }
    allocated_.erase(it);
    std::free(p);
  }

  // VMM fakes that record each call; `failAt` names the one call to fail.
  void InstallVmm(const std::string& failAt = "") {
    auto result = [failAt](const char* call) { return failAt == call ? hipErrorInvalidValue : hipSuccess; };
    g_hipMemGetAllocationGranularity = [this, result](size_t* g, const hipMemAllocationProp* prop,
                                                      hipMemAllocationGranularity_flags flags) {
      prop_ = *prop;
      granularityFlags_ = flags;
      *g = kGran;
      return result("granularity");
    };
    g_hipGetDeviceCount = [result](int* n) {
      *n = 4;
      return result("count");
    };
    g_hipMemCreate = [this, result](hipMemGenericAllocationHandle_t* h, size_t size, const hipMemAllocationProp*,
                                    unsigned long long) {
      calls_.push_back("create");
      createSize_ = size;
      *h = Handle(0x77);
      return result("create");
    };
    g_hipMemAddressReserve = [this, result](void** p, size_t size, size_t align, void*, unsigned long long) {
      calls_.push_back("reserve");
      EXPECT_EQ(size, createSize_);
      EXPECT_EQ(align, kGran);
      *p = reinterpret_cast<void*>(0x40000);
      return result("reserve");
    };
    g_hipMemMap = [this, result](void* p, size_t size, size_t, hipMemGenericAllocationHandle_t h,
                                 unsigned long long) {
      calls_.push_back("map");
      EXPECT_EQ(h, Handle(0x77));
      return result("map");
    };
    g_hipMemSetAccess = [this, result](void*, size_t, const hipMemAccessDesc* desc, size_t count) {
      calls_.push_back("access" + std::to_string(desc->location.id));
      EXPECT_EQ(count, 1u);
      EXPECT_EQ(desc->flags, hipMemAccessFlagsProtReadWrite);
      return result("access");
    };
    g_hipMemUnmap = [this](void*, size_t) {
      calls_.push_back("unmap");
      return hipSuccess;
    };
    g_hipMemAddressFree = [this](void*, size_t) {
      calls_.push_back("addrfree");
      return hipSuccess;
    };
    g_hipMemRelease = [this](hipMemGenericAllocationHandle_t h) {
      calls_.push_back("release");
      EXPECT_EQ(h, Handle(0x77));
      return hipSuccess;
    };
    g_hipDeviceCanAccessPeer = [this](int* can, int dev, int peer) {
      EXPECT_EQ(peer, kDev);
      *can = dev != 1;
      return dev == 2 ? hipErrorInvalidValue : hipSuccess;
    };
  }

  std::map<std::string, int64_t> params_;
  std::vector<size_t> mallocs_;
  std::vector<void*> allocated_, frees_, asyncFrees_;
  std::vector<size_t> memsets_;
  std::vector<hipDeviceAttribute_t> attrs_;
  std::vector<hipMemPoolProps> poolProps_;
  std::vector<std::string> calls_;
  hipMemAllocationProp prop_{};
  hipMemAllocationGranularity_flags granularityFlags_{};
  size_t createSize_ = 0;
  int poolDestroys_ = 0;
  int syncs_ = 0;
  int poolsSupported_ = 1;
  int fabricSupported_ = 0;
  hipError_t attrResult_ = hipSuccess;
  hipError_t poolCreateResult_ = hipSuccess;
  hipError_t mallocResult_ = hipSuccess;
  hipError_t memsetResult_ = hipSuccess;
};

// ncclMemAlloc / ncclMemFree

#if ROCM_VERSION >= 70000
TEST_F(AllocatorMicrotest, MemAlloc_FallsBackToHipMallocWithoutCuMem) {
  void* p = nullptr;
  ASSERT_EQ(ncclMemAlloc_impl(&p, 100), ncclSuccess);
  EXPECT_NE(p, nullptr);
  EXPECT_EQ(mallocs_, std::vector<size_t>{100});
  EXPECT_EQ(g_rocmLibraryInitCalls, 1);

  // A null out-pointer or a zero size goes straight to hipMalloc, whose error is the caller's.
  mallocResult_ = hipErrorInvalidValue;
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  EXPECT_EQ(ncclMemAlloc_impl(nullptr, 100), ncclUnhandledCudaError);
  EXPECT_EQ(ncclMemAlloc_impl(&p, 0), ncclUnhandledCudaError);
  EXPECT_EQ(g_rocmLibraryInitCalls, 1);
  Release(p);
}

TEST_F(AllocatorMicrotest, MemAlloc_DeviceLookupFailures) {
  void* p = nullptr;
  ScopedHook hipGetDevice(g_hipGetDevice, [](int*) { return hipErrorInvalidValue; });
  EXPECT_EQ(ncclMemAlloc_impl(&p, 100), ncclUnhandledCudaError);
  ScopedHook hipGetDevice2(g_hipGetDevice, [](int* d) {
    *d = kDev;
    return hipSuccess;
  });
  ScopedHook hipDeviceGet(g_hipDeviceGet, [](hipDevice_t*, int) { return hipErrorInvalidValue; });
  EXPECT_EQ(ncclMemAlloc_impl(&p, 100), ncclUnhandledCudaError);
  EXPECT_TRUE(mallocs_.empty());
}

TEST_F(AllocatorMicrotest, MemAlloc_MapsAlignedVmmMemoryForEveryPeerThatCanReachIt) {
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  InstallVmm();
  void* p = nullptr;
  ASSERT_EQ(ncclMemAlloc_impl(&p, kGran + 1), ncclSuccess);
  EXPECT_EQ(p, reinterpret_cast<void*>(0x40000));
  EXPECT_EQ(createSize_, 2 * kGran);
  EXPECT_EQ(prop_.type, kVmmType);
  EXPECT_EQ(prop_.location.type, hipMemLocationTypeDevice);
  EXPECT_EQ(prop_.location.id, kHipDev);
  EXPECT_EQ(prop_.requestedHandleType, hipMemHandleTypePosixFileDescriptor);
#if HIP_VERSION > 70000000
  EXPECT_EQ(prop_.allocFlags.gpuDirectRDMACapable, 1);
#endif
  EXPECT_EQ(granularityFlags_, hipMemAllocationGranularityRecommended);
  // Device 1 has no P2P to kDev and device 2's query fails: only 0 and kDev itself get access.
  EXPECT_EQ(calls_, (std::vector<std::string>{"create", "reserve", "map", "access0", "access3"}));
  EXPECT_TRUE(attrs_.empty());
  EXPECT_TRUE(mallocs_.empty());
}

TEST_F(AllocatorMicrotest, MemAlloc_FabricHandlesOnlyWhenTheDeviceSupportsThem) {
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  ncclCuMemHandleType = kFabricHandle;
  InstallVmm();
  void* p = nullptr;
  fabricSupported_ = 1;
  ASSERT_EQ(ncclMemAlloc_impl(&p, 1), ncclSuccess);
  EXPECT_EQ(prop_.requestedHandleType, kFabricHandle);
  ASSERT_EQ(attrs_.size(), 1u);
  EXPECT_EQ(attrs_[0], kFabricSupportedAttr);

  fabricSupported_ = 0;
  ASSERT_EQ(ncclMemAlloc_impl(&p, 1), ncclSuccess);
  EXPECT_EQ(prop_.requestedHandleType, hipMemHandleTypePosixFileDescriptor);

  fabricSupported_ = 1;
  attrResult_ = hipErrorInvalidValue;
  ASSERT_EQ(ncclMemAlloc_impl(&p, 1), ncclSuccess);
  EXPECT_EQ(prop_.requestedHandleType, hipMemHandleTypePosixFileDescriptor);
}

// Each failure undoes exactly the steps that succeeded before it and nulls the pointer.
TEST_F(AllocatorMicrotest, MemAlloc_VmmFailuresUnwindWhatWasDone) {
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  const std::vector<std::pair<std::string, std::vector<std::string>>> cases = {
      {"granularity", {}},
      {"count", {}},
      {"create", {"create"}},
      {"reserve", {"create", "reserve", "release"}},
      {"map", {"create", "reserve", "map", "addrfree", "release"}},
      {"access", {"create", "reserve", "map", "access0", "unmap", "addrfree", "release"}},
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(c.first);
    InstallVmm(c.first);
    calls_.clear();
    void* p = reinterpret_cast<void*>(0x1);
    EXPECT_EQ(ncclMemAlloc_impl(&p, 1), ncclUnhandledCudaError);
    EXPECT_EQ(calls_, c.second);
    if (c.first != "granularity" && c.first != "count") EXPECT_EQ(p, nullptr);
  }
  EXPECT_TRUE(mallocs_.empty());
}

TEST_F(AllocatorMicrotest, MemFree_FreesThroughHipWithoutCuMemAndRestoresTheDevice) {
  void* p = std::malloc(8);
  allocated_.push_back(p);
  std::vector<int> sets;
  ScopedHook ordinal(g_hipPointerGetAttribute, [](void* data, hipPointer_attribute attr, hipDeviceptr_t) {
    EXPECT_EQ(attr, HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL);
    *static_cast<int*>(data) = 1;
    return hipSuccess;
  });
  ScopedHook hipSetDevice(g_hipSetDevice, [&](int dev) {
    sets.push_back(dev);
    return hipSuccess;
  });
  ASSERT_EQ(ncclMemFree_impl(p), ncclSuccess);
  EXPECT_EQ(frees_, std::vector<void*>{p});
  EXPECT_EQ(g_rocmLibraryInitCalls, 1);
  EXPECT_EQ(sets, (std::vector<int>{1, kDev}));

  // A null pointer skips the ordinal lookup and goes to hipFree.
  sets.clear();
  ASSERT_EQ(ncclMemFree_impl(nullptr), ncclSuccess);
  EXPECT_EQ(frees_.back(), nullptr);
  EXPECT_EQ(sets, std::vector<int>{kDev});
}

TEST_F(AllocatorMicrotest, MemFree_ReleasesVmmMemoryWithCuMem) {
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  InstallVmm();
  void* p = reinterpret_cast<void*>(0x40000);
  ScopedHook hipPointerGetAttribute(g_hipPointerGetAttribute, [](void* data, hipPointer_attribute, hipDeviceptr_t) {
    *static_cast<int*>(data) = kDev;
    return hipSuccess;
  });
  ScopedHook retain(g_hipMemRetainAllocationHandle, [](hipMemGenericAllocationHandle_t* h, void* addr) {
    EXPECT_EQ(addr, reinterpret_cast<void*>(0x40000));
    *h = Handle(0x77);
    return hipSuccess;
  });
  ScopedHook hipMemGetAddressRange(g_hipMemGetAddressRange, [](hipDeviceptr_t* base, size_t* size, hipDeviceptr_t ptr) {
    *base = ptr;
    *size = kGran;
    return hipSuccess;
  });
  ASSERT_EQ(ncclMemFree_impl(p), ncclSuccess);
  EXPECT_EQ(calls_, (std::vector<std::string>{"release", "unmap", "release", "addrfree"}));
  EXPECT_TRUE(frees_.empty());

  calls_.clear();
  ScopedHook retainFails(g_hipMemRetainAllocationHandle,
                         [](hipMemGenericAllocationHandle_t*, void*) { return hipErrorInvalidValue; });
  EXPECT_EQ(ncclMemFree_impl(p), ncclUnhandledCudaError);
}

TEST_F(AllocatorMicrotest, MemFree_LookupFailuresStillRestoreTheDevice) {
  void* p = reinterpret_cast<void*>(0x40000);
  std::vector<int> sets;
  ScopedHook hipSetDevice(g_hipSetDevice, [&](int dev) {
    sets.push_back(dev);
    return dev == kDev ? hipSuccess : hipErrorInvalidValue;
  });
  ScopedHook ordinal(g_hipPointerGetAttribute,
                     [](void*, hipPointer_attribute, hipDeviceptr_t) { return hipErrorInvalidValue; });
  EXPECT_EQ(ncclMemFree_impl(p), ncclUnhandledCudaError);
  EXPECT_EQ(sets, std::vector<int>{kDev});

  sets.clear();
  ScopedHook hipPointerGetAttribute2(g_hipPointerGetAttribute, [](void* data, hipPointer_attribute, hipDeviceptr_t) {
    *static_cast<int*>(data) = 1;
    return hipSuccess;
  });
  EXPECT_EQ(ncclMemFree_impl(p), ncclUnhandledCudaError);
  EXPECT_EQ(sets, (std::vector<int>{1, kDev}));
  EXPECT_TRUE(frees_.empty());

  ScopedHook hipGetDevice(g_hipGetDevice, [](int*) { return hipErrorInvalidValue; });
  EXPECT_EQ(ncclMemFree_impl(p), ncclUnhandledCudaError);
}
#else
// Below ROCm 7.0 both are plain hipMalloc/hipFree, even with cuMem enabled.
TEST_F(AllocatorMicrotest, MemAllocAndFree_PlainHipBelowRocm7) {
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  void* p = nullptr;
  ASSERT_EQ(ncclMemAlloc_impl(&p, 100), ncclSuccess);
  EXPECT_EQ(mallocs_, std::vector<size_t>{100});
  ASSERT_EQ(ncclMemFree_impl(p), ncclSuccess);
  EXPECT_EQ(frees_, std::vector<void*>{p});
  EXPECT_EQ(g_rocmLibraryInitCalls, 0);
}
#endif  // ROCM_VERSION >= 70000

// ncclSpace

class AllocatorSpaceMicrotest : public ::testing::Test {
 protected:
  void SetUp() override { ncclSpaceConstruct(&space_); }
  void TearDown() override { ncclSpaceDestruct(&space_); }

  int64_t Alloc(int64_t size, int align = 1, int64_t limit = 1 << 20) {
    int64_t off = -1;
    EXPECT_EQ(ncclSpaceAlloc(&space_, limit, size, align, &off), ncclSuccess);
    return off;
  }
  std::vector<int64_t> Cuts() const { return std::vector<int64_t>(space_.cuts, space_.cuts + space_.count); }

  ncclSpace space_;
};

TEST_F(AllocatorSpaceMicrotest, Alloc_PacksFromZeroAndHonoursAlignment) {
  EXPECT_EQ(Alloc(10), 0);
  EXPECT_EQ(Alloc(5), 10);
  EXPECT_EQ(Alloc(4, 8), 16);
  EXPECT_EQ(Cuts(), std::vector<int64_t>{20});
}

TEST_F(AllocatorSpaceMicrotest, Alloc_FailsWhenNothingFitsUnderTheLimit) {
  EXPECT_EQ(Alloc(60, 1, 64), 0);
  int64_t off = -1;
  EXPECT_EQ(ncclSpaceTryAlloc(&space_, 64, 8, 1, &off), ncclInternalError);
  EXPECT_EQ(ncclSpaceAlloc(&space_, 64, 8, 1, &off), ncclInternalError);
  EXPECT_EQ(off, -1);
  EXPECT_EQ(Alloc(4, 1, 64), 60);  // exactly fills the frontier
}

TEST_F(AllocatorSpaceMicrotest, Alloc_ReusesTheFirstHoleThatFits) {
  for (int i = 0; i < 5; i++) Alloc(10);
  ASSERT_EQ(ncclSpaceFree(&space_, 10, 10), ncclSuccess);
  ASSERT_EQ(ncclSpaceFree(&space_, 30, 10), ncclSuccess);
  EXPECT_EQ(Cuts(), (std::vector<int64_t>{10, 20, 30, 40, 50}));
  EXPECT_EQ(Alloc(15), 50);  // neither 10-byte hole fits
  EXPECT_EQ(Alloc(4), 10);
  EXPECT_EQ(Alloc(6), 14);  // closes the hole: its two cuts fuse away
  EXPECT_EQ(Cuts(), (std::vector<int64_t>{30, 40, 65}));
  EXPECT_EQ(Alloc(10), 30);
  EXPECT_EQ(Cuts(), std::vector<int64_t>{65});
}

// Freeing the top of the last segment hands the space back to the frontier.
TEST_F(AllocatorSpaceMicrotest, Free_TopOfTheLastSegmentRejoinsTheFrontier) {
  for (int i = 0; i < 4; i++) Alloc(10);
  ASSERT_EQ(ncclSpaceFree(&space_, 30, 10), ncclSuccess);
  EXPECT_EQ(Cuts(), std::vector<int64_t>{30});
  EXPECT_EQ(Alloc(15), 30);
}

TEST_F(AllocatorSpaceMicrotest, Alloc_SkipsAllocatedSegments) {
  Alloc(10);
  Alloc(10);
  Alloc(40);
  ASSERT_EQ(ncclSpaceFree(&space_, 10, 10), ncclSuccess);
  EXPECT_EQ(Alloc(15), 60);
}

// The two cuts a zero-byte object adds on an existing cut make three equal ones, of which one stays.
TEST_F(AllocatorSpaceMicrotest, Alloc_ZeroBytesOnASegmentStartLeavesTheCuts) {
  for (int i = 0; i < 3; i++) Alloc(16);
  ASSERT_EQ(ncclSpaceFree(&space_, 16, 16), ncclSuccess);
  ASSERT_EQ(Cuts(), (std::vector<int64_t>{16, 32, 48}));
  EXPECT_EQ(Alloc(0, 32), 32);
  EXPECT_EQ(Cuts(), (std::vector<int64_t>{16, 32, 48}));
}

TEST_F(AllocatorSpaceMicrotest, Free_ShrinksFromEitherEndOrSplits) {
  Alloc(100);
  ASSERT_EQ(ncclSpaceFree(&space_, 0, 10), ncclSuccess);  // bottom of the first segment
  EXPECT_EQ(Cuts(), (std::vector<int64_t>{10, 100}));
  ASSERT_EQ(ncclSpaceFree(&space_, 10, 10), ncclSuccess);  // bottom again, now via the fast path
  EXPECT_EQ(Cuts(), (std::vector<int64_t>{20, 100}));
  ASSERT_EQ(ncclSpaceFree(&space_, 90, 10), ncclSuccess);  // top: the frontier moves down
  EXPECT_EQ(Cuts(), (std::vector<int64_t>{20, 90}));
  ASSERT_EQ(ncclSpaceFree(&space_, 40, 10), ncclSuccess);  // middle: split in two
  EXPECT_EQ(Cuts(), (std::vector<int64_t>{20, 40, 50, 90}));
  ASSERT_EQ(ncclSpaceFree(&space_, 20, 20), ncclSuccess);  // a whole segment
  EXPECT_EQ(Cuts(), (std::vector<int64_t>{50, 90}));
  ASSERT_EQ(ncclSpaceFree(&space_, 50, 40), ncclSuccess);
  EXPECT_TRUE(Cuts().empty());
}

TEST_F(AllocatorSpaceMicrotest, Free_RejectsUnallocatedRanges) {
  EXPECT_EQ(ncclSpaceFree(&space_, 0, 1), ncclInternalError);
  Alloc(10);
  Alloc(10);
  ASSERT_EQ(ncclSpaceFree(&space_, 0, 10), ncclSuccess);
  EXPECT_EQ(ncclSpaceFree(&space_, 20, 1), ncclInternalError);
  EXPECT_EQ(ncclSpaceFree(&space_, 5, 1), ncclInternalError);  // in the hole below
  EXPECT_EQ(ncclSpaceFree(&space_, 15, 6), ncclInternalError);
  EXPECT_EQ(Cuts(), (std::vector<int64_t>{10, 20}));
}

TEST_F(AllocatorSpaceMicrotest, ManySegmentsGrowTheCutArray) {
  for (int i = 0; i < 40; i++) Alloc(1);
  for (int i = 0; i < 40; i += 2) ASSERT_EQ(ncclSpaceFree(&space_, i, 1), ncclSuccess);
  EXPECT_EQ(space_.count, 40);
  EXPECT_EQ(space_.capacity, 64);
  for (int i = 0; i < 40; i += 2) EXPECT_EQ(Alloc(1), i);
  EXPECT_EQ(Cuts(), std::vector<int64_t>{40});
}

// Padding appended to a full segment is never freed, so the space never empties.
// Pins AICOMRCCL-3039; a fix flips this pin.
TEST_F(AllocatorSpaceMicrotest, Alloc_KnownBug_AlignmentPaddingIsNeverFreed) {
  EXPECT_EQ(Alloc(1), 0);
  EXPECT_EQ(Alloc(1, 16), 16);
  ASSERT_EQ(ncclSpaceFree(&space_, 0, 1), ncclSuccess);
  ASSERT_EQ(ncclSpaceFree(&space_, 16, 1), ncclSuccess);
  EXPECT_EQ(Cuts(), (std::vector<int64_t>{1, 16}));
  int64_t off = -1;
  EXPECT_EQ(ncclSpaceTryAlloc(&space_, 16, 16, 1, &off), ncclInternalError);
}

// ncclShadowPool

class AllocatorShadowPoolMicrotest : public AllocatorMicrotest {
 protected:
  void SetUp() override {
    AllocatorMicrotest::SetUp();
    ncclShadowPoolConstruct(&pool_);
  }
  void TearDown() override {
    EXPECT_EQ(ncclShadowPoolDestruct(&pool_, kStream), ncclSuccess);
    AllocatorMicrotest::TearDown();
  }

  void* Alloc(size_t size, void** host = nullptr) {
    void* dev = nullptr;
    void* h = nullptr;
    EXPECT_EQ(ncclShadowPoolAlloc(&pool_, size, &dev, &h, kStream), ncclSuccess);
    if (host) *host = h;
    return dev;
  }

  ncclShadowPool pool_;
};

TEST_F(AllocatorShadowPoolMicrotest, Construct_IsEmptyAndDestructIsANoOp) {
  EXPECT_EQ(pool_.hbits, 0);
  EXPECT_EQ(pool_.count, 0);
  EXPECT_EQ(pool_.table, nullptr);
  EXPECT_EQ(pool_.memPool, nullptr);
  EXPECT_EQ(pool_.pages, nullptr);
  EXPECT_EQ(ncclShadowPoolDestruct(&pool_, kStream), ncclSuccess);
  EXPECT_EQ(syncs_, 0);
}

TEST_F(AllocatorShadowPoolMicrotest, Alloc_ZeroSizeReturnsNulls) {
  void* dev = this;
  void* host = this;
  EXPECT_EQ(ncclShadowPoolAlloc(&pool_, 0, &dev, &host, kStream), ncclSuccess);
  EXPECT_EQ(dev, nullptr);
  EXPECT_EQ(host, nullptr);
  EXPECT_EQ(ncclShadowPoolAlloc(&pool_, 0, nullptr, nullptr, kStream), ncclSuccess);
  EXPECT_EQ(pool_.hbits, 0);
}

TEST_F(AllocatorShadowPoolMicrotest, Alloc_FirstUseCreatesTheMemPool) {
  params_["SHADOW_MEMPOOL_MAX_SIZE"] = 12345;
  void* host = nullptr;
  void* dev = Alloc(100, &host);
  ASSERT_EQ(poolProps_.size(), 1u);
  EXPECT_EQ(poolProps_[0].allocType, hipMemAllocationTypePinned);
  EXPECT_EQ(poolProps_[0].handleTypes, hipMemHandleTypeNone);
  EXPECT_EQ(poolProps_[0].location.type, hipMemLocationTypeDevice);
  EXPECT_EQ(poolProps_[0].location.id, kDev);
  EXPECT_EQ(poolProps_[0].maxSize, 12345u);
  EXPECT_EQ(pool_.memPool, kPool);
  EXPECT_EQ(pool_.hbits, 4);
  EXPECT_EQ(pool_.count, 1);
  // 100 bytes round up to 104 (1/16 of the next power of two): 64 of them fill a 6656-byte page.
  EXPECT_EQ(mallocs_, std::vector<size_t>{64 * 104});
  EXPECT_EQ(memsets_, std::vector<size_t>{64 * 104});
  EXPECT_EQ(reinterpret_cast<uintptr_t>(host) % alignof(max_align_t), 0u);
  void* h2 = nullptr;
  ASSERT_EQ(ncclShadowPoolToHost(&pool_, dev, &h2), ncclSuccess);
  EXPECT_EQ(h2, host);
  EXPECT_EQ(Alloc(100), static_cast<char*>(dev) + 104);
  EXPECT_EQ(mallocs_.size(), 1u);
  EXPECT_EQ(poolProps_.size(), 1u);
}

TEST_F(AllocatorShadowPoolMicrotest, Alloc_WithoutPoolSupportUsesHipMalloc) {
  poolsSupported_ = 0;
  void* small = Alloc(8);
  void* big = Alloc(32 << 10);
  EXPECT_TRUE(poolProps_.empty());
  EXPECT_EQ(pool_.memPool, nullptr);
  EXPECT_EQ(mallocs_, (std::vector<size_t>{64 * 8, 32 << 10}));
  ASSERT_EQ(ncclShadowPoolFree(&pool_, big, kStream), ncclSuccess);
  EXPECT_EQ(frees_, std::vector<void*>{big});
  ASSERT_EQ(ncclShadowPoolFree(&pool_, small, kStream), ncclSuccess);
  EXPECT_EQ(frees_.size(), 1u);  // page memory stays until Destruct
}

TEST_F(AllocatorShadowPoolMicrotest, Alloc_FirstUseFailures) {
  void* dev = nullptr;
  ScopedHook hipGetDevice(g_hipGetDevice, [](int*) { return hipErrorInvalidValue; });
  EXPECT_EQ(ncclShadowPoolAlloc(&pool_, 8, &dev, nullptr, kStream), ncclUnhandledCudaError);
  ScopedHook hipGetDevice2(g_hipGetDevice, [](int* d) {
    *d = kDev;
    return hipSuccess;
  });
  attrResult_ = hipErrorInvalidValue;
  EXPECT_EQ(ncclShadowPoolAlloc(&pool_, 8, &dev, nullptr, kStream), ncclUnhandledCudaError);
  attrResult_ = hipSuccess;
  poolCreateResult_ = hipErrorOutOfMemory;
  EXPECT_EQ(ncclShadowPoolAlloc(&pool_, 8, &dev, nullptr, kStream), ncclUnhandledCudaError);
  EXPECT_EQ(pool_.hbits, 0);
  EXPECT_TRUE(mallocs_.empty());
  pool_.memPool = nullptr;
}

TEST_F(AllocatorShadowPoolMicrotest, Alloc_LargeObjectsGetTheirOwnAllocation) {
  // Pages are used only while at least three objects fit in 64 KiB.
  Alloc(21845);
  void* direct = Alloc(21846);
  EXPECT_EQ(mallocs_, (std::vector<size_t>{64 << 10, 21846}));
  ASSERT_EQ(ncclShadowPoolFree(&pool_, direct, kStream), ncclSuccess);
  EXPECT_EQ(asyncFrees_, std::vector<void*>{direct});
}

TEST_F(AllocatorShadowPoolMicrotest, Alloc_DeviceAllocationFailuresAreReported) {
  mallocResult_ = hipErrorOutOfMemory;
  void* dev = nullptr;
  EXPECT_EQ(ncclShadowPoolAlloc(&pool_, 1 << 20, &dev, nullptr, kStream), ncclUnhandledCudaError);
  EXPECT_EQ(pool_.count, 0);
  EXPECT_TRUE(allocated_.empty());
}

// A large object whose zeroing fails is neither returned nor tracked, so it leaks.
// Pins AICOMRCCL-3040; a fix flips this pin.
TEST_F(AllocatorShadowPoolMicrotest, Alloc_KnownBug_LargeObjectLeaksWhenZeroingFails) {
  memsetResult_ = hipErrorInvalidValue;
  void* dev = nullptr;
  EXPECT_EQ(ncclShadowPoolAlloc(&pool_, 1 << 20, &dev, nullptr, kStream), ncclUnhandledCudaError);
  ASSERT_EQ(allocated_.size(), 1u);
  EXPECT_EQ(pool_.count, 0);
  ASSERT_EQ(ncclShadowPoolDestruct(&pool_, kStream), ncclSuccess);
  EXPECT_TRUE(asyncFrees_.empty());
  Release(allocated_[0]);
  ncclShadowPoolConstruct(&pool_);
}

// A page whose device allocation failed stays listed and backs the next allocation.
// Pins AICOMRCCL-3040; a fix flips this pin.
TEST_F(AllocatorShadowPoolMicrotest, Alloc_KnownBug_FailedPageIsReusedWithoutDeviceMemory) {
  void* dev = this;
  mallocResult_ = hipErrorOutOfMemory;
  EXPECT_EQ(ncclShadowPoolAlloc(&pool_, 64, &dev, nullptr, kStream), ncclUnhandledCudaError);
  ASSERT_NE(pool_.pages, nullptr);
  mallocResult_ = hipSuccess;
  ASSERT_EQ(ncclShadowPoolAlloc(&pool_, 64, &dev, nullptr, kStream), ncclSuccess);
  EXPECT_EQ(dev, nullptr);
  EXPECT_EQ(mallocs_.size(), 1u);
  ASSERT_EQ(ncclShadowPoolFree(&pool_, dev, kStream), ncclSuccess);  // a null devObj is a no-op
  EXPECT_EQ(pool_.count, 1);
}

TEST_F(AllocatorShadowPoolMicrotest, Page_FillsUpLeavesTheListAndReturnsWhenASlotFrees) {
  std::vector<void*> objs;
  for (int i = 0; i < 64; i++) objs.push_back(Alloc(16));
  for (int i = 1; i < 64; i++) EXPECT_EQ(objs[i], static_cast<char*>(objs[0]) + 16 * i);
  EXPECT_EQ(pool_.pages, nullptr);  // the full page left the list
  Alloc(16);
  EXPECT_EQ(mallocs_.size(), 2u);
  ASSERT_EQ(ncclShadowPoolFree(&pool_, objs[5], kStream), ncclSuccess);
  ASSERT_EQ(ncclShadowPoolFree(&pool_, objs[9], kStream), ncclSuccess);
  EXPECT_EQ(Alloc(16), objs[5]);
  EXPECT_EQ(Alloc(16), objs[9]);
  EXPECT_EQ(mallocs_.size(), 2u);
}

TEST_F(AllocatorShadowPoolMicrotest, Page_SizeClassesDoNotShare) {
  void* a = Alloc(16);
  void* b = Alloc(24);
  void* c = Alloc(17);  // rounds up to 18: a third class
  EXPECT_EQ(mallocs_, (std::vector<size_t>{64 * 16, 64 * 24, 64 * 18}));
  EXPECT_EQ(Alloc(16), static_cast<char*>(a) + 16);
  EXPECT_EQ(Alloc(24), static_cast<char*>(b) + 24);
  EXPECT_EQ(Alloc(18), static_cast<char*>(c) + 18);
}

TEST_F(AllocatorShadowPoolMicrotest, HostShadowsAreZeroedAndDistinct) {
  void* h1 = nullptr;
  void* h2 = nullptr;
  Alloc(48, &h1);
  std::memset(h1, 0xff, 48);
  void* d2 = Alloc(48, &h2);
  EXPECT_NE(h1, h2);
  for (int i = 0; i < 48; i++) ASSERT_EQ(static_cast<char*>(h2)[i], 0);
  void* got = nullptr;
  ASSERT_EQ(ncclShadowPoolToHost(&pool_, d2, &got), ncclSuccess);
  EXPECT_EQ(got, h2);
}

TEST_F(AllocatorShadowPoolMicrotest, HashTableDoublesAtTwoObjectsPerBucket) {
  std::vector<void*> devs, hosts;
  for (int i = 0; i < 33; i++) {
    void* h = nullptr;
    devs.push_back(Alloc(8, &h));
    hosts.push_back(h);
    EXPECT_EQ(pool_.hbits, i < 32 ? 4 : 5);
  }
  for (int i = 0; i < 33; i++) {
    void* h = nullptr;
    ASSERT_EQ(ncclShadowPoolToHost(&pool_, devs[i], &h), ncclSuccess);
    EXPECT_EQ(h, hosts[i]);
  }
  for (int i = 0; i < 33; i++) ASSERT_EQ(ncclShadowPoolFree(&pool_, devs[i], kStream), ncclSuccess);
  EXPECT_EQ(pool_.count, 0);
}

// Every object in one bucket: lookups walk the chain past the other entries.
TEST_F(AllocatorShadowPoolMicrotest, CollidingObjectsAreFoundAlongTheChain) {
  g_hashToZero = true;
  void* h0 = nullptr;
  void* d0 = Alloc(8, &h0);
  void* d1 = Alloc(8);
  void* d2 = Alloc(8);
  void* h = nullptr;
  ASSERT_EQ(ncclShadowPoolToHost(&pool_, d0, &h), ncclSuccess);
  EXPECT_EQ(h, h0);
  ASSERT_EQ(ncclShadowPoolFree(&pool_, d1, kStream), ncclSuccess);
  ASSERT_EQ(ncclShadowPoolFree(&pool_, d0, kStream), ncclSuccess);
  EXPECT_EQ(ncclShadowPoolToHost(&pool_, d0, &h), ncclInternalError);
  ASSERT_EQ(ncclShadowPoolToHost(&pool_, d2, &h), ncclSuccess);
  EXPECT_EQ(pool_.count, 1);
}

TEST_F(AllocatorShadowPoolMicrotest, UnknownObjectsAreRejected) {
  Alloc(8);
  int other = 0;
  void* h = this;
  EXPECT_EQ(ncclShadowPoolToHost(&pool_, &other, &h), ncclInternalError);
  EXPECT_EQ(ncclShadowPoolFree(&pool_, &other, kStream), ncclInternalError);
  ASSERT_EQ(ncclShadowPoolToHost(&pool_, nullptr, &h), ncclSuccess);
  EXPECT_EQ(h, nullptr);
  EXPECT_EQ(ncclShadowPoolFree(&pool_, nullptr, kStream), ncclSuccess);
  EXPECT_EQ(pool_.count, 1);
}

// With no table yet, an unknown object crashes the lookup instead of being rejected. The pool is
// hidden from the optimizer so it can't see the null table and drop the load.
// Pins AICOMRCCL-3041; a fix flips this pin.
TEST_F(AllocatorShadowPoolMicrotest, FreeAndToHost_KnownBug_UnknownObjectOnAnEmptyPoolCrashes) {
  int other = 0;
  void* h = nullptr;
  ncclShadowPool* pool = &pool_;
  asm volatile("" : "+r"(pool) : : "memory");
  ScopedDeathTestStyle style("threadsafe");
  EXPECT_EXIT(ncclShadowPoolToHost(pool, &other, &h), DEATH_BY_SEGV, "");
  EXPECT_EXIT(ncclShadowPoolFree(pool, &other, kStream), DEATH_BY_SEGV, "");
}

TEST_F(AllocatorShadowPoolMicrotest, Destruct_FreesObjectsPagesAndPool) {
  std::vector<void*> page;
  for (int i = 0; i < 64; i++) page.push_back(Alloc(32));  // a full page, off the list
  void* partial = Alloc(64);
  void* big = Alloc(1 << 20);
  ASSERT_EQ(ncclShadowPoolDestruct(&pool_, kStream), ncclSuccess);
  std::set<void*> freed(asyncFrees_.begin(), asyncFrees_.end());
  EXPECT_EQ(freed, (std::set<void*>{page[0], partial, big}));
  EXPECT_EQ(asyncFrees_.size(), 3u);
  EXPECT_EQ(syncs_, 1);
  EXPECT_EQ(poolDestroys_, 1);
  ncclShadowPoolConstruct(&pool_);
}

TEST_F(AllocatorShadowPoolMicrotest, Destruct_WithoutAPoolUsesHipFree) {
  poolsSupported_ = 0;
  void* small = Alloc(8);
  void* big = Alloc(1 << 20);
  ASSERT_EQ(ncclShadowPoolDestruct(&pool_, kStream), ncclSuccess);
  std::set<void*> freed(frees_.begin(), frees_.end());
  EXPECT_EQ(freed, (std::set<void*>{small, big}));
  EXPECT_EQ(poolDestroys_, 0);
  EXPECT_EQ(syncs_, 1);
  ncclShadowPoolConstruct(&pool_);
}

// Destruct only frees; an emptied pool (count 0, table still allocated) still drops its pages.
TEST_F(AllocatorShadowPoolMicrotest, Destruct_EmptiedPoolStillFreesPages) {
  void* dev = Alloc(8);
  ASSERT_EQ(ncclShadowPoolFree(&pool_, dev, kStream), ncclSuccess);
  ASSERT_EQ(ncclShadowPoolDestruct(&pool_, kStream), ncclSuccess);
  EXPECT_EQ(asyncFrees_, std::vector<void*>{dev});
  ncclShadowPoolConstruct(&pool_);
}

}  // namespace
