/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/register/register.cc, the per-comm user-buffer
// registration cache. This TU #includes the hipified register.cc
// (REGISTER_CC_PATH) so its file-static helpers are reachable. No GPU, no
// librccl.so.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include "ScopedHook.h"
#include "fakes/argcheck_fakes.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/recorder_fakes.h"
#include "fakes/transport_p2p_reg_fakes.h"
#include "fakes/transport_stubs.h"

// Routes every NCCL_PARAM in register.cc through g_loadParam on each call.
#include "fakes/param_redirect.h"

#include "fakes/register_seam.h"
#include "fakes/transport_p2p_reg_seam.h"
#include REGISTER_CC_PATH
#include "fakes/transport_p2p_reg_seam_undef.h"
#include "fakes/register_seam_undef.h"

namespace {

void* Ptr(uintptr_t addr) { return reinterpret_cast<void*>(addr); }

constexpr uintptr_t kPage = 0x1000;
constexpr int kCommDev = 3;

// hipMemGetAddressRange reporting one allocation that starts at the queried address.
hipError_t WholeAllocation(hipDeviceptr_t* base, size_t* size, hipDeviceptr_t ptr) {
  *base = ptr;
  *size = 1 << 20;
  return hipSuccess;
}

// A comm-queue callback that records what the cache looked like when it ran.
struct ProbeCallback {
  ncclCommCallback base{};
  ncclResult_t result = ncclSuccess;
  int calls = 0;
  int populationSeen = -1;

  static ncclResult_t Fn(ncclComm* comm, ncclCommCallback* cb) {
    auto* self = reinterpret_cast<ProbeCallback*>(cb);
    ++self->calls;
    self->populationSeen = comm->regCache.population;
    return self->result;
  }
  ProbeCallback() { base.fn = Fn; }
};

class RegisterMicrotestBase : public ::testing::Test {
 protected:
  std::unique_ptr<ncclComm> comm_ = std::make_unique<ncclComm>();

  void SetUp() override {
    comm_->regCache.pageSize = kPage;
    comm_->cudaDev = kCommDev;
    ncclIntruQueueMpscConstruct(&comm_->callbackQueue);
  }

  void TearDown() override {
    ncclRegCache* cache = &comm_->regCache;
    for (int i = 0; i < cache->population; ++i) free(cache->slots[i]);
    free(cache->slots);
    ResetArgcheckFakes();
    ResetHipFakes();
    ResetNcclFakes();
    ResetRecorderFakes();
    ResetTransportP2pRegFakes();
    ResetTransportStubs();
  }

  // Appends a record straight into the cache, as production allocates it (calloc, freed with free()).
  ncclReg* Seed(uintptr_t beg, uintptr_t end, int localRefs = 1, int graphRefs = 0) {
    ncclRegCache* cache = &comm_->regCache;
    if (cache->population == cache->capacity) {
      cache->capacity = cache->capacity ? 2 * cache->capacity : 8;
      cache->slots = static_cast<ncclReg**>(realloc(cache->slots, cache->capacity * sizeof(ncclReg*)));
    }
    auto* reg = static_cast<ncclReg*>(calloc(1, sizeof(ncclReg)));
    reg->begAddr = beg;
    reg->endAddr = end;
    reg->localRefs = localRefs;
    reg->graphRefs = graphRefs;
    cache->slots[cache->population++] = reg;
    return reg;
  }

  // Runs ncclRegCleanup and, on success, forgets the cache it freed.
  ncclResult_t CleanupAll() {
    ncclResult_t res = ncclRegCleanup(comm_.get());
    if (res == ncclSuccess) comm_->regCache = {nullptr, 0, 0, kPage};
    return res;
  }

  int Population() const { return comm_->regCache.population; }
  ncclReg* Slot(int i) const { return comm_->regCache.slots[i]; }
};

// ---------------------------------------------------------------------------
// ncclRegLocalIsValid
// ---------------------------------------------------------------------------

using RegisterLocalIsValidMicrotest = RegisterMicrotestBase;

TEST_F(RegisterLocalIsValidMicrotest, LocalRefHeld_ReportsValid) {
  ncclReg reg{};
  reg.localRefs = 1;
  bool valid = false;
  EXPECT_EQ(micro_ncclRegLocalIsValid(&reg, &valid), ncclSuccess);
  EXPECT_TRUE(valid);
}

TEST_F(RegisterLocalIsValidMicrotest, OnlyGraphRefHeld_ReportsInvalid) {
  ncclReg reg{};
  reg.graphRefs = 1;
  bool valid = true;
  EXPECT_EQ(micro_ncclRegLocalIsValid(&reg, &valid), ncclSuccess);
  EXPECT_FALSE(valid);
}

TEST_F(RegisterLocalIsValidMicrotest, NullRecord_LeavesFlagUntouched) {
  bool valid = true;
  EXPECT_EQ(micro_ncclRegLocalIsValid(nullptr, &valid), ncclSuccess);
  EXPECT_TRUE(valid);
}

TEST_F(RegisterLocalIsValidMicrotest, NullFlag_Succeeds) {
  ncclReg reg{};
  reg.localRefs = 1;
  EXPECT_EQ(micro_ncclRegLocalIsValid(&reg, nullptr), ncclSuccess);
}

// ---------------------------------------------------------------------------
// ncclRegister: cache insert / lookup
// ---------------------------------------------------------------------------

using RegisterCacheMicrotest = RegisterMicrotestBase;

TEST_F(RegisterCacheMicrotest, EmptyCache_AddsPageAlignedRecordWithOneLocalRef) {
  void* handle = nullptr;
  ASSERT_EQ(ncclRegister(comm_.get(), Ptr(0x10010), 0x2000, /*isGraph=*/false, &handle), ncclSuccess);

  ASSERT_EQ(Population(), 1);
  ncclReg* reg = Slot(0);
  EXPECT_EQ(handle, reg);
  EXPECT_EQ(reg->begAddr, 0x10000u);
  EXPECT_EQ(reg->endAddr, 0x13000u);
  EXPECT_EQ(reg->localRefs, 1);
  EXPECT_EQ(reg->graphRefs, 0);
}

TEST_F(RegisterCacheMicrotest, GraphRegistration_CountsGraphRefOnly) {
  void* handle = nullptr;
  ASSERT_EQ(ncclRegister(comm_.get(), Ptr(0x10000), kPage, /*isGraph=*/true, &handle), ncclSuccess);

  ASSERT_EQ(Population(), 1);
  EXPECT_EQ(Slot(0)->graphRefs, 1);
  EXPECT_EQ(Slot(0)->localRefs, 0);
}

TEST_F(RegisterCacheMicrotest, PageAlignedBuffer_KeepsEndUnrounded) {
  void* handle = nullptr;
  ASSERT_EQ(ncclRegister(comm_.get(), Ptr(0x10000), 2 * kPage, false, &handle), ncclSuccess);
  EXPECT_EQ(Slot(0)->begAddr, 0x10000u);
  EXPECT_EQ(Slot(0)->endAddr, 0x12000u);
}

TEST_F(RegisterCacheMicrotest, RangeInsideExistingRecord_ReusesRecordAndCountsRef) {
  ncclReg* existing = Seed(0x10000, 0x20000, /*localRefs=*/1, /*graphRefs=*/0);

  void* local = nullptr;
  ASSERT_EQ(ncclRegister(comm_.get(), Ptr(0x11000), kPage, false, &local), ncclSuccess);
  EXPECT_EQ(local, existing);
  EXPECT_EQ(existing->localRefs, 2);
  EXPECT_EQ(existing->graphRefs, 0);

  void* graph = nullptr;
  ASSERT_EQ(ncclRegister(comm_.get(), Ptr(0x1f000), kPage, true, &graph), ncclSuccess);
  EXPECT_EQ(graph, existing);
  EXPECT_EQ(existing->localRefs, 2);
  EXPECT_EQ(existing->graphRefs, 1);
  EXPECT_EQ(Population(), 1);
}

TEST_F(RegisterCacheMicrotest, RangeInsideLaterRecord_ReusesThatRecord) {
  Seed(0x10000, 0x11000);
  ncclReg* later = Seed(0x20000, 0x30000);
  void* handle = nullptr;
  ASSERT_EQ(ncclRegister(comm_.get(), Ptr(0x21000), kPage, false, &handle), ncclSuccess);
  EXPECT_EQ(Population(), 2);
  EXPECT_EQ(handle, later);
  EXPECT_EQ(later->localRefs, 2);
}

TEST_F(RegisterCacheMicrotest, RangeMatchingRecordExactly_ReusesRecord) {
  ncclReg* existing = Seed(0x10000, 0x12000);
  void* handle = nullptr;
  ASSERT_EQ(ncclRegister(comm_.get(), Ptr(0x10000), 2 * kPage, false, &handle), ncclSuccess);
  EXPECT_EQ(Population(), 1);
  EXPECT_EQ(handle, existing);
}

TEST_F(RegisterCacheMicrotest, LowerAddress_InsertsBeforeExistingRecord) {
  ncclReg* high = Seed(0x20000, 0x21000);
  void* handle = nullptr;
  ASSERT_EQ(ncclRegister(comm_.get(), Ptr(0x10000), kPage, false, &handle), ncclSuccess);

  ASSERT_EQ(Population(), 2);
  EXPECT_EQ(Slot(0), handle);
  EXPECT_EQ(Slot(0)->begAddr, 0x10000u);
  EXPECT_EQ(Slot(1), high);
}

TEST_F(RegisterCacheMicrotest, AddressBetweenRecords_InsertsInOrder) {
  ncclReg* low = Seed(0x10000, 0x11000);
  ncclReg* high = Seed(0x30000, 0x31000);
  void* handle = nullptr;
  ASSERT_EQ(ncclRegister(comm_.get(), Ptr(0x20000), kPage, false, &handle), ncclSuccess);

  ASSERT_EQ(Population(), 3);
  EXPECT_EQ(Slot(0), low);
  EXPECT_EQ(Slot(1), handle);
  EXPECT_EQ(Slot(2), high);
}

TEST_F(RegisterCacheMicrotest, RangeOverrunningRecordEnd_AddsSeparateRecord) {
  ncclReg* existing = Seed(0x10000, 0x12000);
  void* handle = nullptr;
  ASSERT_EQ(ncclRegister(comm_.get(), Ptr(0x11000), 2 * kPage, false, &handle), ncclSuccess);

  ASSERT_EQ(Population(), 2);
  EXPECT_EQ(Slot(0), existing);
  EXPECT_EQ(Slot(1), handle);
  EXPECT_EQ(existing->localRefs, 1);
  EXPECT_EQ(Slot(1)->begAddr, 0x11000u);
  EXPECT_EQ(Slot(1)->endAddr, 0x13000u);
}

TEST_F(RegisterCacheMicrotest, CacheFull_GrowsAndKeepsEveryRecord) {
  constexpr int kRecords = 70;
  std::vector<void*> handles(kRecords);
  for (int i = 0; i < kRecords; ++i) {
    ASSERT_EQ(ncclRegister(comm_.get(), Ptr(0x100000 + i * 0x10000), kPage, false, &handles[i]), ncclSuccess);
  }

  ASSERT_EQ(Population(), kRecords);
  EXPECT_GE(comm_->regCache.capacity, kRecords);
  for (int i = 0; i < kRecords; ++i) {
    EXPECT_EQ(Slot(i), handles[i]) << i;
    EXPECT_EQ(Slot(i)->begAddr, 0x100000u + i * 0x10000) << i;
  }
}

TEST_F(RegisterCacheMicrotest, InvalidComm_ReturnsErrorWithoutCaching) {
  ScopedHook check(g_commCheck, [](ncclComm*, const char*, const char*) { return ncclInvalidArgument; });
  void* handle = nullptr;
  EXPECT_EQ(ncclRegister(comm_.get(), Ptr(0x10000), kPage, false, &handle), ncclInvalidArgument);
  EXPECT_EQ(Population(), 0);
}

TEST_F(RegisterCacheMicrotest, DefaultCheckMode_SkipsPointerCheck) {
  ScopedHook ptrCheck(g_cudaPtrCheck,
                      [](const void*, ncclComm*, const char*, const char*) { return ncclInvalidArgument; });
  void* handle = nullptr;
  EXPECT_EQ(ncclRegister(comm_.get(), Ptr(0x10000), kPage, false, &handle), ncclSuccess);
  EXPECT_EQ(ptrCheck.calls, 0);
}

TEST_F(RegisterCacheMicrotest, DebugCheckMode_RejectsBadPointerWithoutCaching) {
  comm_->checkMode = ncclCheckModeDebugLocal;
  const void* checked = nullptr;
  ScopedHook ptrCheck(g_cudaPtrCheck, [&](const void* p, ncclComm*, const char*, const char*) {
    checked = p;
    return ncclInvalidArgument;
  });
  void* handle = nullptr;
  EXPECT_EQ(ncclRegister(comm_.get(), Ptr(0x10040), kPage, false, &handle), ncclInvalidArgument);
  EXPECT_EQ(checked, Ptr(0x10040));
  EXPECT_EQ(Population(), 0);
}

// ---------------------------------------------------------------------------
// ncclRegister: CPU-backed buffers are not registered (cuMem enabled)
// ---------------------------------------------------------------------------

class RegisterSysmemMicrotest : public RegisterMicrotestBase {
 protected:
  void SetUp() override {
    RegisterMicrotestBase::SetUp();
    g_cuMemEnable = [] { return 1; };
    g_hipMemGetAddressRange = WholeAllocation;
  }
};

TEST_F(RegisterSysmemMicrotest, HostMemory_SkipsRegistration) {
  g_hipPointerGetAttribute = ForceMemoryType(hipMemoryTypeHost);
  void* handle = nullptr;
  EXPECT_EQ(ncclRegister(comm_.get(), Ptr(0x10000), kPage, false, &handle), ncclSuccess);
  EXPECT_EQ(Population(), 0);
}

TEST_F(RegisterSysmemMicrotest, AddressRangeQueryFails_ReturnsCudaErrorWithoutCaching) {
  // Legacy-IPC device memory, so no later query can fail in its place.
  g_hipPointerGetAttribute = ForceLegacyIpcCapable();
  g_hipMemGetAddressRange = [](hipDeviceptr_t*, size_t*, hipDeviceptr_t) { return hipErrorInvalidValue; };
  void* handle = nullptr;
  EXPECT_EQ(ncclRegister(comm_.get(), Ptr(0x10000), kPage, false, &handle), ncclUnhandledCudaError);
  EXPECT_EQ(Population(), 0);
}

TEST_F(RegisterSysmemMicrotest, MemoryTypeQueryFails_ReturnsCudaErrorWithoutCaching) {
  // Legacy-IPC device memory, so no later query can fail in its place.
  auto deviceMemory = ForceLegacyIpcCapable();
  g_hipPointerGetAttribute = [deviceMemory](void* data, hipPointer_attribute attr, hipDeviceptr_t ptr) {
    return attr == HIP_POINTER_ATTRIBUTE_MEMORY_TYPE ? hipErrorInvalidValue : deviceMemory(data, attr, ptr);
  };
  void* handle = nullptr;
  EXPECT_EQ(ncclRegister(comm_.get(), Ptr(0x10000), kPage, false, &handle), ncclUnhandledCudaError);
  EXPECT_EQ(Population(), 0);
}

TEST_F(RegisterSysmemMicrotest, CuMemDisabled_RegistersWithoutQueryingMemoryType) {
  g_cuMemEnable = [] { return 0; };
  ScopedHook attrs(g_hipPointerGetAttribute, ForceMemoryType(hipMemoryTypeHost));
  void* handle = nullptr;
  EXPECT_EQ(ncclRegister(comm_.get(), Ptr(0x10000), kPage, false, &handle), ncclSuccess);
  EXPECT_EQ(Population(), 1);
  EXPECT_EQ(attrs.calls, 0);
}

#if HIP_VERSION >= 71260540
// Newer HIP: a non-legacy-IPC device buffer is walked segment by segment.

TEST_F(RegisterSysmemMicrotest, LegacyIpcDeviceMemory_RegistersWithoutWalkingSegments) {
  g_hipPointerGetAttribute = ForceLegacyIpcCapable();
  ScopedHook retain(g_hipMemRetainAllocationHandle,
                    [](hipMemGenericAllocationHandle_t*, void*) { return hipErrorInvalidValue; });
  void* handle = nullptr;
  EXPECT_EQ(ncclRegister(comm_.get(), Ptr(0x10000), kPage, false, &handle), ncclSuccess);
  EXPECT_EQ(Population(), 1);
  EXPECT_EQ(retain.calls, 0);
}

TEST_F(RegisterSysmemMicrotest, LegacyIpcQueryFails_ReturnsCudaErrorWithoutCaching) {
  auto deviceMemory = ForceLegacyIpcCapable();
  g_hipPointerGetAttribute = [deviceMemory](void* data, hipPointer_attribute attr, hipDeviceptr_t ptr) {
    return attr == HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE ? hipErrorInvalidValue
                                                                    : deviceMemory(data, attr, ptr);
  };
  void* handle = nullptr;
  EXPECT_EQ(ncclRegister(comm_.get(), Ptr(0x10000), kPage, false, &handle), ncclUnhandledCudaError);
  EXPECT_EQ(Population(), 0);
}

class RegisterSegmentWalkMicrotest : public RegisterSysmemMicrotest {
 protected:
  hipMemLocationType segmentLocation_ = hipMemLocationTypeDevice;

  void SetUp() override {
    RegisterSysmemMicrotest::SetUp();
    g_hipPointerGetAttribute = ForceMemoryType(hipMemoryTypeDevice);
    g_hipMemRetainAllocationHandle = [](hipMemGenericAllocationHandle_t* h, void*) {
      *h = reinterpret_cast<hipMemGenericAllocationHandle_t>(0x1);
      return hipSuccess;
    };
    g_hipMemGetAllocationPropertiesFromHandle = [this](hipMemAllocationProp* prop, hipMemGenericAllocationHandle_t) {
      *prop = {};
      prop->location.type = segmentLocation_;
      return hipSuccess;
    };
    g_hipMemRelease = [](hipMemGenericAllocationHandle_t) { return hipSuccess; };
  }
};

TEST_F(RegisterSegmentWalkMicrotest, DeviceSegments_Registers) {
  void* handle = nullptr;
  EXPECT_EQ(ncclRegister(comm_.get(), Ptr(0x10000), kPage, false, &handle), ncclSuccess);
  EXPECT_EQ(Population(), 1);
}

TEST_F(RegisterSegmentWalkMicrotest, SegmentQueryFails_ReturnsCudaErrorWithoutCaching) {
  g_hipMemRetainAllocationHandle = [](hipMemGenericAllocationHandle_t*, void*) { return hipErrorInvalidValue; };
  void* handle = nullptr;
  EXPECT_EQ(ncclRegister(comm_.get(), Ptr(0x10000), kPage, false, &handle), ncclUnhandledCudaError);
  EXPECT_EQ(Population(), 0);
}

#if ROCM_VERSION >= 71200
TEST_F(RegisterSegmentWalkMicrotest, HostSegment_SkipsRegistration) {
  segmentLocation_ = hipMemLocationTypeHost;
  void* handle = nullptr;
  EXPECT_EQ(ncclRegister(comm_.get(), Ptr(0x10000), kPage, false, &handle), ncclSuccess);
  EXPECT_EQ(Population(), 0);
}
#endif

#else
// Older HIP: only plain device memory is registered.

TEST_F(RegisterSysmemMicrotest, DeviceMemory_Registers) {
  g_hipPointerGetAttribute = ForceMemoryType(hipMemoryTypeDevice);
  void* handle = nullptr;
  EXPECT_EQ(ncclRegister(comm_.get(), Ptr(0x10000), kPage, false, &handle), ncclSuccess);
  EXPECT_EQ(Population(), 1);
}

TEST_F(RegisterSysmemMicrotest, ManagedMemory_SkipsRegistration) {
  g_hipPointerGetAttribute = ForceMemoryType(hipMemoryTypeManaged);
  void* handle = nullptr;
  EXPECT_EQ(ncclRegister(comm_.get(), Ptr(0x10000), kPage, false, &handle), ncclSuccess);
  EXPECT_EQ(Population(), 0);
}
#endif

// ---------------------------------------------------------------------------
// ncclCommRegister / ncclCommGraphRegister
// ---------------------------------------------------------------------------

using RegisterCommRegisterMicrotest = RegisterMicrotestBase;

TEST_F(RegisterCommRegisterMicrotest, LocalRegisterDisabled_ReturnsNullHandleWithoutCaching) {
  g_loadParam = ParamOverride({{"LOCAL_REGISTER", 0}});
  void* handle = Ptr(0xdead);
  EXPECT_EQ(ncclCommRegister_impl(comm_.get(), Ptr(0x10000), kPage, &handle), ncclSuccess);
  EXPECT_EQ(handle, nullptr);
  EXPECT_EQ(Population(), 0);
}

TEST_F(RegisterCommRegisterMicrotest, LocalRegisterEnabled_ReturnsCachedLocalRecord) {
  g_loadParam = ParamOverride({{"LOCAL_REGISTER", 1}});
  void* handle = nullptr;
  EXPECT_EQ(ncclCommRegister_impl(comm_.get(), Ptr(0x10000), kPage, &handle), ncclSuccess);
  ASSERT_EQ(Population(), 1);
  EXPECT_EQ(handle, Slot(0));
  EXPECT_EQ(Slot(0)->localRefs, 1);
}

TEST_F(RegisterCommRegisterMicrotest, PendingCallbacks_RunBeforeLookup) {
  g_loadParam = ParamOverride({{"LOCAL_REGISTER", 1}});
  ProbeCallback probe;
  ncclIntruQueueMpscEnqueue(&comm_->callbackQueue, &probe.base);
  void* handle = nullptr;
  EXPECT_EQ(ncclCommRegister_impl(comm_.get(), Ptr(0x10000), kPage, &handle), ncclSuccess);
  EXPECT_EQ(probe.calls, 1);
  EXPECT_EQ(probe.populationSeen, 0);
  EXPECT_EQ(Population(), 1);
}

TEST_F(RegisterCommRegisterMicrotest, CallbackFails_ReturnsErrorWithoutCaching) {
  g_loadParam = ParamOverride({{"LOCAL_REGISTER", 1}});
  ProbeCallback probe;
  probe.result = ncclInternalError;
  ncclIntruQueueMpscEnqueue(&comm_->callbackQueue, &probe.base);
  void* handle = nullptr;
  EXPECT_EQ(ncclCommRegister_impl(comm_.get(), Ptr(0x10000), kPage, &handle), ncclInternalError);
  EXPECT_EQ(Population(), 0);
}

TEST_F(RegisterCommRegisterMicrotest, InvalidComm_ReturnsErrorWithoutCaching) {
  g_loadParam = ParamOverride({{"LOCAL_REGISTER", 1}});
  g_commCheck = [](ncclComm*, const char*, const char*) { return ncclInvalidArgument; };
  void* handle = nullptr;
  EXPECT_EQ(ncclCommRegister_impl(comm_.get(), Ptr(0x10000), kPage, &handle), ncclInvalidArgument);
  EXPECT_EQ(Population(), 0);
}

TEST_F(RegisterCommRegisterMicrotest, BadPointer_ReturnsErrorWithoutCaching) {
  g_loadParam = ParamOverride({{"LOCAL_REGISTER", 1}});
  comm_->checkMode = ncclCheckModeDebugLocal;
  g_cudaPtrCheck = [](const void*, ncclComm*, const char*, const char*) { return ncclInvalidArgument; };
  void* handle = nullptr;
  EXPECT_EQ(ncclCommRegister_impl(comm_.get(), Ptr(0x10000), kPage, &handle), ncclInvalidArgument);
  EXPECT_EQ(Population(), 0);
}

TEST_F(RegisterCommRegisterMicrotest, InvalidComm_LeavesCallbacksPending) {
  g_loadParam = ParamOverride({{"LOCAL_REGISTER", 1}});
  g_commCheck = [](ncclComm*, const char*, const char*) { return ncclInvalidArgument; };
  ProbeCallback probe;
  ncclIntruQueueMpscEnqueue(&comm_->callbackQueue, &probe.base);
  void* handle = nullptr;
  EXPECT_EQ(ncclCommRegister_impl(comm_.get(), Ptr(0x10000), kPage, &handle), ncclInvalidArgument);
  EXPECT_EQ(probe.calls, 0);
}

TEST_F(RegisterCommRegisterMicrotest, RecorderFails_ReturnsRecorderError) {
  g_loadParam = ParamOverride({{"LOCAL_REGISTER", 1}});
  g_recorderResult = ncclSystemError;
  void* handle = nullptr;
  EXPECT_EQ(ncclCommRegister_impl(comm_.get(), Ptr(0x10000), kPage, &handle), ncclSystemError);
}

TEST_F(RegisterCommRegisterMicrotest, GraphRegister_ReturnsCachedGraphRecord) {
  void* handle = nullptr;
  EXPECT_EQ(micro_ncclCommGraphRegister(comm_.get(), Ptr(0x10000), kPage, &handle), ncclSuccess);
  ASSERT_EQ(Population(), 1);
  EXPECT_EQ(handle, Slot(0));
  EXPECT_EQ(Slot(0)->graphRefs, 1);
  EXPECT_EQ(Slot(0)->localRefs, 0);
}

TEST_F(RegisterCommRegisterMicrotest, GraphRegisterWithMemcpyP2p_ReturnsNullHandleWithoutCaching) {
  g_ncclP2pUsesMemcpy = [] { return true; };
  void* handle = Ptr(0xdead);
  EXPECT_EQ(micro_ncclCommGraphRegister(comm_.get(), Ptr(0x10000), kPage, &handle), ncclSuccess);
  EXPECT_EQ(handle, nullptr);
  EXPECT_EQ(Population(), 0);
}

TEST_F(RegisterCommRegisterMicrotest, GraphRegisterInvalidComm_ReturnsError) {
  g_commCheck = [](ncclComm*, const char*, const char*) { return ncclInvalidArgument; };
  void* handle = nullptr;
  EXPECT_EQ(micro_ncclCommGraphRegister(comm_.get(), Ptr(0x10000), kPage, &handle), ncclInvalidArgument);
}

// ---------------------------------------------------------------------------
// ncclCommDeregister / ncclCommGraphDeregister
// ---------------------------------------------------------------------------

using RegisterDeregisterMicrotest = RegisterMicrotestBase;

TEST_F(RegisterDeregisterMicrotest, NullHandle_SucceedsWithoutTouchingDevice) {
  ScopedHook setDevice(g_hipSetDevice, [](int) { return hipSuccess; });
  EXPECT_EQ(ncclCommDeregister_impl(comm_.get(), nullptr), ncclSuccess);
  EXPECT_EQ(setDevice.calls, 0);
}

TEST_F(RegisterDeregisterMicrotest, UnknownHandle_ReturnsInvalidUsage) {
  Seed(0x10000, 0x11000);
  ncclReg stranger{};
  stranger.localRefs = 1;
  EXPECT_EQ(ncclCommDeregister_impl(comm_.get(), &stranger), ncclInvalidUsage);
  EXPECT_EQ(Population(), 1);
}

TEST_F(RegisterDeregisterMicrotest, SharedRecord_DropsOneRefAndKeepsRecord) {
  ncclReg* reg = Seed(0x10000, 0x11000, /*localRefs=*/2);
  EXPECT_EQ(ncclCommDeregister_impl(comm_.get(), reg), ncclSuccess);
  ASSERT_EQ(Population(), 1);
  EXPECT_EQ(Slot(0), reg);
  EXPECT_EQ(reg->localRefs, 1);
}

TEST_F(RegisterDeregisterMicrotest, GraphRefStillHeld_KeepsRecordAfterLocalRelease) {
  ncclReg* reg = Seed(0x10000, 0x11000, /*localRefs=*/1, /*graphRefs=*/1);
  EXPECT_EQ(ncclCommDeregister_impl(comm_.get(), reg), ncclSuccess);
  ASSERT_EQ(Population(), 1);
  EXPECT_EQ(reg->localRefs, 0);
  EXPECT_EQ(reg->graphRefs, 1);
}

TEST_F(RegisterDeregisterMicrotest, LocalRefStillHeld_KeepsRecordAfterGraphRelease) {
  ncclReg* reg = Seed(0x10000, 0x11000, /*localRefs=*/1, /*graphRefs=*/1);
  EXPECT_EQ(micro_ncclCommGraphDeregister(comm_.get(), reg), ncclSuccess);
  ASSERT_EQ(Population(), 1);
  EXPECT_EQ(reg->localRefs, 1);
  EXPECT_EQ(reg->graphRefs, 0);
}

TEST_F(RegisterDeregisterMicrotest, LastRef_RemovesRecordAndKeepsOthersInOrder) {
  ncclReg* first = Seed(0x10000, 0x11000);
  ncclReg* middle = Seed(0x20000, 0x21000);
  ncclReg* last = Seed(0x30000, 0x31000);
  EXPECT_EQ(ncclCommDeregister_impl(comm_.get(), middle), ncclSuccess);
  ASSERT_EQ(Population(), 2);
  EXPECT_EQ(Slot(0), first);
  EXPECT_EQ(Slot(1), last);
}

TEST_F(RegisterDeregisterMicrotest, LastGraphRef_RemovesRecord) {
  ncclReg* reg = Seed(0x10000, 0x11000, /*localRefs=*/0, /*graphRefs=*/1);
  EXPECT_EQ(micro_ncclCommGraphDeregister(comm_.get(), reg), ncclSuccess);
  EXPECT_EQ(Population(), 0);
}

TEST_F(RegisterDeregisterMicrotest, LastRef_ReleasesOnCommDeviceAndRestoresCaller) {
  g_currentDevice = 5;
  ncclReg* reg = Seed(0x10000, 0x11000);
  reg->state = COLLNET_REG_COMPLETE;
  int deviceDuringRelease = -1;
  g_ncclCollnetDeregBuffer = [&](ncclComm*, ncclProxyConnector*, void*) {
    deviceDuringRelease = g_currentDevice;
    return ncclSuccess;
  };
  EXPECT_EQ(ncclCommDeregister_impl(comm_.get(), reg), ncclSuccess);
  EXPECT_EQ(deviceDuringRelease, kCommDev);
  EXPECT_EQ(g_currentDevice, 5);
}

TEST_F(RegisterDeregisterMicrotest, SetDeviceFails_ReturnsCudaErrorAndKeepsRecord) {
  ncclReg* reg = Seed(0x10000, 0x11000);
  g_hipSetDevice = [](int) { return hipErrorInvalidDevice; };
  EXPECT_EQ(ncclCommDeregister_impl(comm_.get(), reg), ncclUnhandledCudaError);
  EXPECT_EQ(Population(), 1);
  EXPECT_EQ(reg->localRefs, 1);
}

TEST_F(RegisterDeregisterMicrotest, GetDeviceFails_ReturnsCudaErrorAndKeepsRecord) {
  ncclReg* reg = Seed(0x10000, 0x11000);
  g_hipGetDevice = [](int*) { return hipErrorInvalidDevice; };
  EXPECT_EQ(ncclCommDeregister_impl(comm_.get(), reg), ncclUnhandledCudaError);
  EXPECT_EQ(Population(), 1);
}

TEST_F(RegisterDeregisterMicrotest, CleanupFails_ReturnsErrorAndKeepsRecord) {
  ncclReg* reg = Seed(0x10000, 0x11000);
  reg->state = NVLS_REG_COMPLETE;
  g_ncclNvlsUbDeregister = [](ncclComm*, ncclReg*) { return ncclInternalError; };
  EXPECT_EQ(ncclCommDeregister_impl(comm_.get(), reg), ncclInternalError);
  EXPECT_EQ(Population(), 1);
}

TEST_F(RegisterDeregisterMicrotest, PendingCallbacks_RunBeforeLookup) {
  ncclReg* reg = Seed(0x10000, 0x11000);
  ProbeCallback probe;
  ncclIntruQueueMpscEnqueue(&comm_->callbackQueue, &probe.base);
  EXPECT_EQ(ncclCommDeregister_impl(comm_.get(), reg), ncclSuccess);
  EXPECT_EQ(probe.calls, 1);
  EXPECT_EQ(probe.populationSeen, 1);
  EXPECT_EQ(Population(), 0);
}

TEST_F(RegisterDeregisterMicrotest, CallbackFails_ReturnsErrorAndKeepsRecord) {
  ncclReg* reg = Seed(0x10000, 0x11000);
  ProbeCallback probe;
  probe.result = ncclInternalError;
  ncclIntruQueueMpscEnqueue(&comm_->callbackQueue, &probe.base);
  EXPECT_EQ(ncclCommDeregister_impl(comm_.get(), reg), ncclInternalError);
  EXPECT_EQ(Population(), 1);
}

TEST_F(RegisterDeregisterMicrotest, InvalidComm_ReturnsErrorAndKeepsRecord) {
  ncclReg* reg = Seed(0x10000, 0x11000);
  g_commCheck = [](ncclComm*, const char*, const char*) { return ncclInvalidArgument; };
  EXPECT_EQ(ncclCommDeregister_impl(comm_.get(), reg), ncclInvalidArgument);
  EXPECT_EQ(Population(), 1);
}

TEST_F(RegisterDeregisterMicrotest, GraphDeregisterInvalidComm_ReturnsErrorAndKeepsRecord) {
  ncclReg* reg = Seed(0x10000, 0x11000, /*localRefs=*/0, /*graphRefs=*/1);
  g_commCheck = [](ncclComm*, const char*, const char*) { return ncclInvalidArgument; };
  EXPECT_EQ(micro_ncclCommGraphDeregister(comm_.get(), reg), ncclInvalidArgument);
  EXPECT_EQ(Population(), 1);
}

TEST_F(RegisterDeregisterMicrotest, InvalidComm_LeavesCallbacksPending) {
  ncclReg* reg = Seed(0x10000, 0x11000);
  g_commCheck = [](ncclComm*, const char*, const char*) { return ncclInvalidArgument; };
  ProbeCallback probe;
  ncclIntruQueueMpscEnqueue(&comm_->callbackQueue, &probe.base);
  EXPECT_EQ(ncclCommDeregister_impl(comm_.get(), reg), ncclInvalidArgument);
  EXPECT_EQ(probe.calls, 0);
}

TEST_F(RegisterDeregisterMicrotest, RecorderFails_ReturnsErrorAndKeepsRecord) {
  ncclReg* reg = Seed(0x10000, 0x11000);
  g_recorderResult = ncclSystemError;
  EXPECT_EQ(ncclCommDeregister_impl(comm_.get(), reg), ncclSystemError);
  EXPECT_EQ(Population(), 1);
}

// ---------------------------------------------------------------------------
// Record teardown (regCleanup via ncclRegCleanup)
// ---------------------------------------------------------------------------

class RegisterCleanupMicrotest : public RegisterMicrotestBase {
 protected:
  ncclProxyConnector proxyA_{};
  ncclProxyConnector proxyB_{};

  // Prepends a NET handle node; production frees the nodes.
  void AddNetHandle(ncclReg* reg, void* handle, ncclProxyConnector* proxy) {
    auto* node = static_cast<ncclRegNetHandles*>(calloc(1, sizeof(ncclRegNetHandles)));
    node->handle = handle;
    node->proxyConn = proxy;
    node->next = reg->netHandleHead;
    reg->netHandleHead = node;
  }

  // An ipcInfos array of `size` slots; production frees the array and every entry.
  std::vector<ncclIpcRegInfo*> AddIpcInfos(ncclReg* reg, std::initializer_list<bool> present) {
    reg->ipcInfosSize = static_cast<int>(present.size());
    reg->ipcInfos = static_cast<ncclIpcRegInfo**>(calloc(present.size(), sizeof(ncclIpcRegInfo*)));
    std::vector<ncclIpcRegInfo*> infos;
    int i = 0;
    for (bool p : present) {
      if (p) {
        reg->ipcInfos[i] = static_cast<ncclIpcRegInfo*>(calloc(1, sizeof(ncclIpcRegInfo)));
        infos.push_back(reg->ipcInfos[i]);
      }
      ++i;
    }
    return infos;
  }
};

TEST_F(RegisterCleanupMicrotest, EmptyCache_Succeeds) { EXPECT_EQ(CleanupAll(), ncclSuccess); }

TEST_F(RegisterCleanupMicrotest, NetRegistered_DeregistersEveryNetHandle) {
  ncclReg* reg = Seed(0x10000, 0x11000);
  reg->state = NET_REG_COMPLETE;
  AddNetHandle(reg, Ptr(0xa1), &proxyA_);
  AddNetHandle(reg, Ptr(0xb2), &proxyB_);
  std::vector<std::pair<void*, ncclProxyConnector*>> seen;
  g_ncclNetDeregBuffer = [&](ncclComm* c, ncclProxyConnector* p, void* h) {
    EXPECT_EQ(c, comm_.get());
    seen.emplace_back(h, p);
    return ncclSuccess;
  };
  EXPECT_EQ(CleanupAll(), ncclSuccess);
  std::vector<std::pair<void*, ncclProxyConnector*>> expected{{Ptr(0xb2), &proxyB_}, {Ptr(0xa1), &proxyA_}};
  EXPECT_EQ(seen, expected);
}

TEST_F(RegisterCleanupMicrotest, NetDeregisterFails_StillDeregistersRemainingHandles) {
  ncclReg* reg = Seed(0x10000, 0x11000);
  reg->state = NET_REG_COMPLETE;
  AddNetHandle(reg, Ptr(0xa1), &proxyA_);
  AddNetHandle(reg, Ptr(0xb2), &proxyB_);
  ScopedHook dereg(g_ncclNetDeregBuffer, [](ncclComm*, ncclProxyConnector*, void*) { return ncclSystemError; });
  EXPECT_EQ(CleanupAll(), ncclSuccess);
  EXPECT_EQ(dereg.calls, 2);
}

TEST_F(RegisterCleanupMicrotest, NvlsRegistered_DeregistersRecord) {
  ncclReg* reg = Seed(0x10000, 0x11000);
  reg->state = NVLS_REG_COMPLETE;
  ncclReg* seen = nullptr;
  g_ncclNvlsUbDeregister = [&](ncclComm*, ncclReg* r) {
    seen = r;
    return ncclSuccess;
  };
  EXPECT_EQ(CleanupAll(), ncclSuccess);
  EXPECT_EQ(seen, reg);
}

TEST_F(RegisterCleanupMicrotest, NvlsDeregisterFails_ReturnsError) {
  ncclReg* reg = Seed(0x10000, 0x11000);
  reg->state = NVLS_REG_COMPLETE;
  g_ncclNvlsUbDeregister = [](ncclComm*, ncclReg*) { return ncclInternalError; };
  EXPECT_EQ(CleanupAll(), ncclInternalError);
}

TEST_F(RegisterCleanupMicrotest, CollnetRegistered_DeregistersHandle) {
  ncclReg* reg = Seed(0x10000, 0x11000);
  reg->state = COLLNET_REG_COMPLETE;
  reg->collnetHandle = Ptr(0xc3);
  reg->collnetProxyconn = &proxyA_;
  ScopedHook dereg(g_ncclCollnetDeregBuffer, [&](ncclComm*, ncclProxyConnector* p, void* h) {
    EXPECT_EQ(p, &proxyA_);
    EXPECT_EQ(h, Ptr(0xc3));
    return ncclSuccess;
  });
  EXPECT_EQ(CleanupAll(), ncclSuccess);
  EXPECT_EQ(dereg.calls, 1);
}

TEST_F(RegisterCleanupMicrotest, CollnetDeregisterFails_StillSucceeds) {
  ncclReg* reg = Seed(0x10000, 0x11000);
  reg->state = COLLNET_REG_COMPLETE;
  reg->collnetProxyconn = &proxyA_;
  g_ncclCollnetDeregBuffer = [](ncclComm*, ncclProxyConnector*, void*) { return ncclSystemError; };
  EXPECT_EQ(CleanupAll(), ncclSuccess);
}

TEST_F(RegisterCleanupMicrotest, NoTransportRegistered_DeregistersNothing) {
  ncclReg* reg = Seed(0x10000, 0x11000);
  reg->state = NVLS_REG_POSSIBLE | IPC_REG_COMPLETE;
  AddNetHandle(reg, Ptr(0xa1), &proxyA_);
  ncclRegNetHandles* netNode = reg->netHandleHead;  // only freed once NET registration completed
  reg->collnetHandle = Ptr(0xc3);
  reg->collnetProxyconn = &proxyA_;
  ScopedHook net(g_ncclNetDeregBuffer, [](ncclComm*, ncclProxyConnector*, void*) { return ncclSuccess; });
  ScopedHook nvls(g_ncclNvlsUbDeregister, [](ncclComm*, ncclReg*) { return ncclSuccess; });
  ScopedHook collnet(g_ncclCollnetDeregBuffer, [](ncclComm*, ncclProxyConnector*, void*) { return ncclSuccess; });
  EXPECT_EQ(CleanupAll(), ncclSuccess);
  EXPECT_EQ(net.calls, 0);
  EXPECT_EQ(nvls.calls, 0);
  EXPECT_EQ(collnet.calls, 0);
  free(netNode);
}

TEST_F(RegisterCleanupMicrotest, IpcInfosWithoutCompleteFlag_DeregistersEveryPresentPeer) {
  ncclReg* reg = Seed(0x10000, 0x11000);
  std::vector<ncclIpcRegInfo*> infos = AddIpcInfos(reg, {true, false, true});
  std::vector<ncclIpcRegInfo*> seen;
  g_ncclIpcDeregBuffer = [&](ncclComm*, ncclIpcRegInfo* info) {
    seen.push_back(info);
    return ncclSuccess;
  };
  EXPECT_EQ(CleanupAll(), ncclSuccess);
  EXPECT_EQ(seen, infos);
}

TEST_F(RegisterCleanupMicrotest, IpcDeregisterFails_StillDeregistersRemainingPeers) {
  ncclReg* reg = Seed(0x10000, 0x11000);
  AddIpcInfos(reg, {true, true});
  ScopedHook dereg(g_ncclIpcDeregBuffer, [](ncclComm*, ncclIpcRegInfo*) { return ncclSystemError; });
  EXPECT_EQ(CleanupAll(), ncclSuccess);
  EXPECT_EQ(dereg.calls, 2);
}

TEST_F(RegisterCleanupMicrotest, PeerAddressTables_FreesDeviceCopy) {
  ncclReg* reg = Seed(0x10000, 0x11000);
  reg->regIpcAddrs.hostPeerRmtAddrs = static_cast<uintptr_t*>(calloc(4, sizeof(uintptr_t)));
  reg->regIpcAddrs.devPeerRmtAddrs = reinterpret_cast<uintptr_t*>(0xd0000);
  g_hipMemGetAddressRange = WholeAllocation;
  g_hipAsyncOpsResult = hipSuccess;
  void* freed = nullptr;
  g_hipFree = [&](void* p) {
    freed = p;
    return hipSuccess;
  };
  EXPECT_EQ(CleanupAll(), ncclSuccess);
  EXPECT_EQ(freed, Ptr(0xd0000));
}

TEST_F(RegisterCleanupMicrotest, DeviceFreeFails_ReturnsError) {
  ncclReg* reg = Seed(0x10000, 0x11000);
  reg->regIpcAddrs.devPeerRmtAddrs = reinterpret_cast<uintptr_t*>(0xd0000);
  g_hipMemGetAddressRange = WholeAllocation;
  g_hipAsyncOpsResult = hipSuccess;
  g_hipFree = [](void*) { return hipErrorInvalidValue; };
  EXPECT_EQ(CleanupAll(), ncclUnhandledCudaError);
}

TEST_F(RegisterCleanupMicrotest, EveryRecord_IsTornDown) {
  for (uintptr_t base : {0x10000, 0x20000, 0x30000}) {
    ncclReg* reg = Seed(base, base + kPage);
    reg->state = COLLNET_REG_COMPLETE;
    reg->collnetHandle = Ptr(base);
    reg->collnetProxyconn = &proxyA_;
  }
  std::vector<void*> seen;
  g_ncclCollnetDeregBuffer = [&](ncclComm*, ncclProxyConnector*, void* h) {
    seen.push_back(h);
    return ncclSuccess;
  };
  EXPECT_EQ(CleanupAll(), ncclSuccess);
  EXPECT_EQ(seen, (std::vector<void*>{Ptr(0x10000), Ptr(0x20000), Ptr(0x30000)}));
}

}  // namespace
