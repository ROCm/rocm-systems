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

#include "../common/LogCapture.hpp"
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

using RcclUnitTesting::CaptureLog;
using RcclUnitTesting::LogHas;

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
  AddIpcInfos(reg, {true});
  ScopedHook collnet(g_ncclCollnetDeregBuffer,
                     [](ncclComm*, ncclProxyConnector*, void*) { return ncclSystemError; });
  ScopedHook ipc(g_ncclIpcDeregBuffer, [](ncclComm*, ncclIpcRegInfo*) { return ncclSuccess; });

  EXPECT_EQ(CleanupAll(), ncclSuccess);
  EXPECT_EQ(collnet.calls, 1);
  EXPECT_EQ(ipc.calls, 1);
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

// ---------------------------------------------------------------------------
// rcclRegAutoLocal / rcclRegAutoRelease: records RCCL makes for unregistered ring buffers
// ---------------------------------------------------------------------------

class RegisterAutoLocalMicrotest : public RegisterMicrotestBase {
 protected:
  // What hipPointerGetAttribute reports for every address.
  hipMemoryType memType_ = hipMemoryTypeDevice;
  unsigned long long bufferId_ = 7;
  hipError_t bufferIdResult_ = hipSuccess;
  // hipMemGetAddressRange: every address sits in an allocSize_-aligned allocation of allocSize_ bytes.
  uintptr_t allocSize_ = 0x10000;
  hipError_t addressRangeResult_ = hipSuccess;
  // 1 for a hipMalloc allocation, 0 for a cuMem one.
  int legacyIpcCap_ = 0;
  std::vector<ncclIpcRegInfo*> deregistered_;

  void SetUp() override {
    RegisterMicrotestBase::SetUp();
    g_hipPointerGetAttribute = [this](void* data, hipPointer_attribute attr, hipDeviceptr_t) {
      if (attr == HIP_POINTER_ATTRIBUTE_MEMORY_TYPE) *static_cast<hipMemoryType*>(data) = memType_;
      if (attr == HIP_POINTER_ATTRIBUTE_BUFFER_ID) {
        if (bufferIdResult_ != hipSuccess) return bufferIdResult_;
        *static_cast<unsigned long long*>(data) = bufferId_;
      }
      if (attr == HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE) *static_cast<int*>(data) = legacyIpcCap_;
      return hipSuccess;
    };
    g_hipMemGetAddressRange = [this](hipDeviceptr_t* base, size_t* size, hipDeviceptr_t ptr) {
      if (addressRangeResult_ != hipSuccess) return addressRangeResult_;
      *base = reinterpret_cast<hipDeviceptr_t>(reinterpret_cast<uintptr_t>(ptr) & ~(allocSize_ - 1));
      *size = allocSize_;
      return hipSuccess;
    };
    g_ncclIpcDeregBuffer = [this](ncclComm*, ncclIpcRegInfo* info) {
      deregistered_.push_back(info);
      return ncclSuccess;
    };
  }

  void TearDown() override {
    ncclRegCache* cache = &comm_->regCache;
    while (cache->rcclRetired) {
      ncclReg* reg = cache->rcclRetired;
      cache->rcclRetired = reg->rcclAuto.retiredNext;
      free(reg);
    }
    RegisterMicrotestBase::TearDown();
  }

  ncclReg* Auto(uintptr_t addr, size_t size) {
    ncclReg* reg = reinterpret_cast<ncclReg*>(0x1);
    EXPECT_EQ(rcclRegAutoLocal(comm_.get(), Ptr(addr), size, &reg), ncclSuccess);
    return reg;
  }

  std::vector<ncclReg*> Retired() const {
    std::vector<ncclReg*> out;
    for (ncclReg* r = comm_->regCache.rcclRetired; r; r = r->rcclAuto.retiredNext) out.push_back(r);
    return out;
  }

  // Gives reg one peer import, as a completed IPC registration leaves it.
  ncclIpcRegInfo* AddPeerImport(ncclReg* reg) {
    reg->ipcInfosSize = 1;
    reg->ipcInfos = static_cast<ncclIpcRegInfo**>(calloc(1, sizeof(ncclIpcRegInfo*)));
    reg->ipcInfos[0] = static_cast<ncclIpcRegInfo*>(calloc(1, sizeof(ncclIpcRegInfo)));
    return reg->ipcInfos[0];
  }
};

TEST_F(RegisterAutoLocalMicrotest, UnregisteredDeviceBuffer_CreatesOwnedRecord) {
  ncclReg* reg = Auto(0x10010, 0x2000);
  ASSERT_NE(reg, nullptr);
  ASSERT_EQ(Population(), 1);
  EXPECT_EQ(Slot(0), reg);
  // The record spans the whole allocation, not just the call's range.
  EXPECT_EQ(reg->begAddr, 0x10000u);
  EXPECT_EQ(reg->endAddr, 0x20000u);
  EXPECT_EQ(reg->localRefs, 1);
  EXPECT_TRUE(reg->rcclAuto.owned);
  EXPECT_EQ(reg->rcclAuto.bufferId, 7u);
}

TEST_F(RegisterAutoLocalMicrotest, SameAllocation_ReusesRecordWithoutNewRef) {
  ncclReg* first = Auto(0x10000, 0x2000);
  EXPECT_EQ(Auto(0x11000, kPage), first);
  EXPECT_EQ(Auto(0x18000, 0x8000), first);
  EXPECT_EQ(Population(), 1);
  EXPECT_EQ(first->localRefs, 1);
}

TEST_F(RegisterAutoLocalMicrotest, BufferSpansTwoAllocations_LeavesUnregistered) {
  EXPECT_EQ(Auto(0x1F000, 0x2000), nullptr);
  EXPECT_EQ(Population(), 0);
}

TEST_F(RegisterAutoLocalMicrotest, AddressRangeQueryFails_LeavesUnregistered) {
  addressRangeResult_ = hipErrorInvalidValue;
  EXPECT_EQ(Auto(0x10000, kPage), nullptr);
  EXPECT_EQ(Population(), 0);
}

#if HIP_VERSION >= 71260540
TEST_F(RegisterAutoLocalMicrotest, LegacyAllocation_LeavesUnregistered) {
  legacyIpcCap_ = 1;
  EXPECT_EQ(Auto(0x10000, kPage), nullptr);
  EXPECT_EQ(Population(), 0);
}
#endif

TEST_F(RegisterAutoLocalMicrotest, CuMemAllocation_CreatesOwnedRecord) {
  ncclReg* reg = Auto(0x10000, kPage);
  ASSERT_NE(reg, nullptr);
  EXPECT_TRUE(reg->rcclAuto.owned);
}

TEST_F(RegisterAutoLocalMicrotest, UserRegisteredBuffer_ReturnsUserRecordUntouched) {
  ncclReg* user = Seed(0x10000, 0x20000);
  EXPECT_EQ(Auto(0x11000, kPage), user);
  EXPECT_EQ(user->localRefs, 1);
  EXPECT_FALSE(user->rcclAuto.owned);
}

TEST_F(RegisterAutoLocalMicrotest, ReusedAddress_RetiresStaleRecordWithoutTouchingPeers) {
  ncclReg* stale = Auto(0x10000, 0x2000);
  AddPeerImport(stale);
  bufferId_ = 8;  // freed and reallocated at the same address

  ncclReg* fresh = Auto(0x10000, 0x2000);
  ASSERT_NE(fresh, nullptr);
  EXPECT_NE(fresh, stale);
  ASSERT_EQ(Population(), 1);
  EXPECT_EQ(Slot(0), fresh);
  EXPECT_EQ(fresh->rcclAuto.bufferId, 8u);
  EXPECT_EQ(Retired(), (std::vector<ncclReg*>{stale}));
  // Unmapping a peer import here would wait on the peer's kernel; it waits for ncclRegCleanup.
  EXPECT_TRUE(deregistered_.empty());
  // The retired record's pages stay mapped on the peers until the communicator is destroyed.
  EXPECT_EQ(comm_->regCache.rcclRetiredBytes, 0x10000u);
  free(stale->ipcInfos[0]);
  free(stale->ipcInfos);
}

TEST_F(RegisterAutoLocalMicrotest, ReusedAddressAlsoHeldLocally_DropsOnlyRcclRefAndLeavesUnregistered) {
  ncclReg* reg = Auto(0x10000, 0x2000);
  ++reg->localRefs;  // someone else also holds the record
  bufferId_ = 8;

  // The record still maps the freed allocation on the peers, so the ring must not use it.
  EXPECT_EQ(Auto(0x10000, 0x2000), nullptr);
  ASSERT_EQ(Population(), 1);
  EXPECT_EQ(Slot(0), reg);
  EXPECT_EQ(reg->localRefs, 1);
  EXPECT_FALSE(reg->rcclAuto.owned);
  EXPECT_TRUE(Retired().empty());
}

TEST_F(RegisterAutoLocalMicrotest, ReusedAddressAlsoHeldByGraph_DropsOnlyRcclRefAndLeavesUnregistered) {
  ncclReg* reg = Auto(0x10000, 0x2000);
  ++reg->graphRefs;
  bufferId_ = 8;

  EXPECT_EQ(Auto(0x10000, 0x2000), nullptr);
  ASSERT_EQ(Population(), 1);
  EXPECT_EQ(Slot(0), reg);
  EXPECT_EQ(reg->localRefs, 0);
  EXPECT_EQ(reg->graphRefs, 1);
  EXPECT_FALSE(reg->rcclAuto.owned);
  EXPECT_TRUE(Retired().empty());
}

TEST_F(RegisterAutoLocalMicrotest, LookupOutsideRingAutoPath_SkipsRecordOnlyRcclHolds) {
  ncclReg* reg = Auto(0x10000, 0x2000);
  ASSERT_NE(reg, nullptr);

  ncclReg* found = reinterpret_cast<ncclReg*>(0x1);
  EXPECT_EQ(ncclRegFind(comm_.get(), Ptr(0x14000), kPage, &found), ncclSuccess);
  EXPECT_EQ(found, nullptr);
  EXPECT_EQ(Population(), 1);
  EXPECT_TRUE(reg->rcclAuto.owned);
  EXPECT_FALSE(comm_->regCache.rcclAutoVisible);
}

TEST_F(RegisterAutoLocalMicrotest, LookupInsideScope_ReturnsOwnedRecord) {
  ncclReg* reg = Auto(0x10000, 0x2000);
  ncclReg* found = nullptr;
  {
    rcclRegAutoScope visible(&comm_->regCache);
    EXPECT_EQ(ncclRegFind(comm_.get(), Ptr(0x14000), kPage, &found), ncclSuccess);
  }
  EXPECT_EQ(found, reg);
  EXPECT_TRUE(Retired().empty());
  EXPECT_FALSE(comm_->regCache.rcclAutoVisible);
}

TEST_F(RegisterAutoLocalMicrotest, DisabledScope_KeepsRecordHidden) {
  Auto(0x10000, 0x2000);
  ncclReg* found = reinterpret_cast<ncclReg*>(0x1);
  {
    rcclRegAutoScope visible(&comm_->regCache, /*enable=*/false);
    EXPECT_EQ(ncclRegFind(comm_.get(), Ptr(0x10000), kPage, &found), ncclSuccess);
  }
  EXPECT_EQ(found, nullptr);
}

TEST_F(RegisterAutoLocalMicrotest, NestedScope_RestoresOuterVisibility) {
  {
    rcclRegAutoScope outer(&comm_->regCache);
    Auto(0x10000, 0x2000);
    EXPECT_TRUE(comm_->regCache.rcclAutoVisible);
  }
  EXPECT_FALSE(comm_->regCache.rcclAutoVisible);
}

TEST_F(RegisterAutoLocalMicrotest, UserRegistersInsideAutoRegisteredAllocation_GetsOwnRecord) {
  ncclReg* reg = Auto(0x10000, 0x2000);
  void* handle = nullptr;
  ASSERT_EQ(ncclRegister(comm_.get(), Ptr(0x14000), kPage, false, &handle), ncclSuccess);
  ASSERT_NE(handle, nullptr);
  EXPECT_NE(handle, reg);
  EXPECT_FALSE(static_cast<ncclReg*>(handle)->rcclAuto.owned);
  EXPECT_EQ(reg->localRefs, 1);
  ASSERT_EQ(Population(), 2);

  // Send/recv outside the user's range must stay unregistered on this rank.
  ncclReg* found = reinterpret_cast<ncclReg*>(0x1);
  EXPECT_EQ(ncclRegFind(comm_.get(), Ptr(0x18000), kPage, &found), ncclSuccess);
  EXPECT_EQ(found, nullptr);
  EXPECT_EQ(ncclRegFind(comm_.get(), Ptr(0x14000), kPage, &found), ncclSuccess);
  EXPECT_EQ(found, handle);
}

TEST_F(RegisterAutoLocalMicrotest, GraphRegistersInsideAutoRegisteredAllocation_GetsOwnRecord) {
  ncclReg* reg = Auto(0x10000, 0x2000);
  void* handle = nullptr;
  ASSERT_EQ(ncclRegister(comm_.get(), Ptr(0x14000), kPage, /*isGraph=*/true, &handle), ncclSuccess);
  ASSERT_NE(handle, nullptr);
  EXPECT_NE(handle, reg);
  EXPECT_EQ(static_cast<ncclReg*>(handle)->graphRefs, 1);
  EXPECT_EQ(reg->graphRefs, 0);
  EXPECT_EQ(Population(), 2);
}

TEST_F(RegisterAutoLocalMicrotest, LookupPastRetiredRecord_FindsUserRecordOfNewAllocation) {
  allocSize_ = 0x40000;
  ncclReg* stale = Auto(0x40000, kPage);  // [0x40000, 0x80000)
  ASSERT_NE(stale, nullptr);
  bufferId_ = 8;
  void* user = nullptr;
  ASSERT_EQ(ncclRegister(comm_.get(), Ptr(0x60000), 0x40000, false, &user), ncclSuccess);
  ASSERT_EQ(Population(), 2);
  ASSERT_EQ(Slot(1), user);

  // The stale record comes first and covers the buffer; the user's record after it covers it too.
  EXPECT_EQ(Auto(0x70000, kPage), user);
  EXPECT_EQ(static_cast<ncclReg*>(user)->localRefs, 1);
  ASSERT_EQ(Population(), 1);
  EXPECT_EQ(Retired(), (std::vector<ncclReg*>{stale}));
}

TEST_F(RegisterAutoLocalMicrotest, ReusedAddress_DeregistersNetHandlesButKeepsPeerImports) {
  ncclReg* stale = Auto(0x10000, 0x2000);
  ncclIpcRegInfo* import = AddPeerImport(stale);
  ncclProxyConnector proxyConn = {};
  auto* netHandle = static_cast<ncclRegNetHandles*>(calloc(1, sizeof(ncclRegNetHandles)));
  netHandle->handle = reinterpret_cast<void*>(0x1234);
  netHandle->proxyConn = &proxyConn;
  stale->netHandleHead = netHandle;
  stale->state |= NET_REG_COMPLETE;
  std::vector<void*> netDeregistered;
  ScopedHook net(g_ncclNetDeregBuffer, [&](ncclComm*, ncclProxyConnector* p, void* h) {
    EXPECT_EQ(p, &proxyConn);
    netDeregistered.push_back(h);
    return ncclSuccess;
  });
  bufferId_ = 8;

  ASSERT_NE(Auto(0x10000, 0x2000), nullptr);
  // A later NET registration at this address must not get the freed allocation's MR from the IB cache.
  EXPECT_EQ(netDeregistered, (std::vector<void*>{reinterpret_cast<void*>(0x1234)}));
  EXPECT_EQ(stale->netHandleHead, nullptr);
  EXPECT_FALSE(stale->state & NET_REG_COMPLETE);
  EXPECT_EQ(Retired(), (std::vector<ncclReg*>{stale}));
  EXPECT_TRUE(deregistered_.empty());

  EXPECT_EQ(rcclRegAutoRelease(comm_.get()), ncclSuccess);
  EXPECT_EQ(netDeregistered.size(), 1u);
  EXPECT_EQ(deregistered_, (std::vector<ncclIpcRegInfo*>{import}));
}

TEST_F(RegisterAutoLocalMicrotest, LookupInsideScopeOfReusedAddress_RetiresStaleRecordAndReturnsNothing) {
  ncclReg* stale = Auto(0x10000, 0x2000);
  bufferId_ = 8;

  ncclReg* found = reinterpret_cast<ncclReg*>(0x1);
  {
    rcclRegAutoScope visible(&comm_->regCache);
    EXPECT_EQ(ncclRegFind(comm_.get(), Ptr(0x10000), kPage, &found), ncclSuccess);
  }
  EXPECT_EQ(found, nullptr);
  EXPECT_EQ(Population(), 0);
  EXPECT_EQ(Retired(), (std::vector<ncclReg*>{stale}));
}

TEST_F(RegisterAutoLocalMicrotest, UserRegistersReusedAddress_GetsFreshRecordThatTheRingThenUses) {
  ncclReg* stale = Auto(0x10000, 0x2000);
  bufferId_ = 8;

  void* handle = nullptr;
  ASSERT_EQ(ncclRegister(comm_.get(), Ptr(0x10000), 0x2000, false, &handle), ncclSuccess);
  ASSERT_NE(handle, nullptr);
  EXPECT_NE(handle, stale);
  EXPECT_FALSE(static_cast<ncclReg*>(handle)->rcclAuto.owned);
  // ncclRegister passes over the hidden record; the next Ring lookup retires it.
  ASSERT_EQ(Population(), 2);
  EXPECT_TRUE(Retired().empty());

  EXPECT_EQ(Auto(0x10000, 0x2000), handle);
  ASSERT_EQ(Population(), 1);
  EXPECT_EQ(Slot(0), handle);
  EXPECT_EQ(Retired(), (std::vector<ncclReg*>{stale}));
}

TEST_F(RegisterAutoLocalMicrotest, HostMemory_LeavesUnregistered) {
  memType_ = hipMemoryTypeHost;
  EXPECT_EQ(Auto(0x10000, kPage), nullptr);
  EXPECT_EQ(Population(), 0);
}

TEST_F(RegisterAutoLocalMicrotest, BufferIdQueryFails_LeavesUnregistered) {
  bufferIdResult_ = hipErrorInvalidValue;
  EXPECT_EQ(Auto(0x10000, kPage), nullptr);
  EXPECT_EQ(Population(), 0);
}

TEST_F(RegisterAutoLocalMicrotest, NoBufferId_LeavesUnregistered) {
  bufferId_ = 0;
  EXPECT_EQ(Auto(0x10000, kPage), nullptr);
  EXPECT_EQ(Population(), 0);
}

TEST_F(RegisterAutoLocalMicrotest, NullOrEmptyBuffer_LeavesUnregistered) {
  EXPECT_EQ(Auto(0, kPage), nullptr);
  EXPECT_EQ(Auto(0x10000, 0), nullptr);
  EXPECT_EQ(Population(), 0);
}

TEST_F(RegisterAutoLocalMicrotest, RetiredBytesPassWarnThreshold_WarnsOnceAndKeepsRegistering) {
  g_loadParam = ParamOverride({{"RCCL_RING_AUTO_REGISTER_WARN_BYTES", 0x18000}});
  std::string log = CaptureLog([&]() {
    // Each round frees the allocation and gets a new one at the same address, retiring a whole 0x10000 record.
    for (unsigned long long id = 7; id < 11; ++id) {
      bufferId_ = id;
      ncclReg* reg = Auto(0x10000, kPage);
      ASSERT_NE(reg, nullptr);
      EXPECT_TRUE(reg->rcclAuto.owned);
    }
  });
  EXPECT_EQ(comm_->regCache.rcclRetiredBytes, 0x30000u);
  EXPECT_EQ(Retired().size(), 3u);
  EXPECT_TRUE(LogHas(log, "keeps freeing and reallocating them"));
  EXPECT_TRUE(LogHas(log, "Set RCCL_RING_AUTO_REGISTER=0"));
  EXPECT_EQ(log.find("RCCL_RING_AUTO_REGISTER is holding"), log.rfind("RCCL_RING_AUTO_REGISTER is holding"));
}

TEST_F(RegisterAutoLocalMicrotest, RegisterRetiresStaleRecordAndInserts_ReturnsNewOwnedRecord) {
  ASSERT_NE(Auto(0x50000, kPage), nullptr);  // [0x50000, 0x60000)
  bufferId_ = 8;
  allocSize_ = 0x40000;
  ASSERT_NE(Auto(0x40000, kPage), nullptr);  // [0x40000, 0x80000); lookup stops before the stale record above
  ASSERT_EQ(Population(), 2);
  bufferId_ = 9;
  allocSize_ = 0x10000;
  // Lookup retires the wide record; ncclRegister then retires the first one and inserts a new record in its place,
  // so the cache population is unchanged.
  ncclReg* reg = Auto(0x50000, kPage);
  ASSERT_NE(reg, nullptr);
  EXPECT_TRUE(reg->rcclAuto.owned);
  EXPECT_EQ(reg->rcclAuto.bufferId, 9u);
  EXPECT_EQ(reg->localRefs, 1);
  ASSERT_EQ(Population(), 1);
  EXPECT_EQ(Slot(0), reg);
  EXPECT_EQ(Retired().size(), 2u);
}

TEST_F(RegisterAutoLocalMicrotest, RetiredBytesBelowWarnThreshold_DoesNotWarn) {
  std::string log = CaptureLog([&]() {
    Auto(0x10000, kPage);
    bufferId_ = 8;
    Auto(0x10000, kPage);
  });
  EXPECT_EQ(comm_->regCache.rcclRetiredBytes, 0x10000u);
  EXPECT_FALSE(LogHas(log, "RCCL_RING_AUTO_REGISTER is holding"));
}

TEST_F(RegisterAutoLocalMicrotest, RegisterFails_LeavesUnregistered) {
  ScopedHook check(g_commCheck, [](ncclComm*, const char*, const char*) { return ncclInvalidArgument; });
  EXPECT_EQ(Auto(0x10000, kPage), nullptr);
  EXPECT_EQ(Population(), 0);
}

TEST_F(RegisterAutoLocalMicrotest, Release_TearsDownOwnedAndRetiredRecordsOnly) {
  ncclReg* user = Seed(0x40000, 0x41000);
  ncclReg* stale = Auto(0x10000, 0x2000);
  ncclIpcRegInfo* staleImport = AddPeerImport(stale);
  bufferId_ = 8;
  ncclReg* fresh = Auto(0x10000, 0x2000);
  ncclIpcRegInfo* freshImport = AddPeerImport(fresh);

  EXPECT_EQ(rcclRegAutoRelease(comm_.get()), ncclSuccess);
  EXPECT_EQ(deregistered_, (std::vector<ncclIpcRegInfo*>{freshImport, staleImport}));
  ASSERT_EQ(Population(), 1);
  EXPECT_EQ(Slot(0), user);
  EXPECT_TRUE(Retired().empty());
  EXPECT_EQ(comm_->regCache.rcclRetiredBytes, 0u);
}

TEST_F(RegisterAutoLocalMicrotest, Release_LeavesOwnedRecordSomeoneElseHoldsToThem) {
  ncclReg* reg = Auto(0x10000, kPage);
  ++reg->localRefs;
  EXPECT_EQ(rcclRegAutoRelease(comm_.get()), ncclSuccess);
  ASSERT_EQ(Population(), 1);
  EXPECT_EQ(Slot(0), reg);
  EXPECT_EQ(reg->localRefs, 1);
  EXPECT_FALSE(reg->rcclAuto.owned);
  EXPECT_TRUE(deregistered_.empty());
}

TEST_F(RegisterAutoLocalMicrotest, Release_LeavesOwnedRecordAGraphHoldsToTheGraph) {
  ncclReg* reg = Auto(0x10000, kPage);
  AddPeerImport(reg);
  ++reg->graphRefs;

  EXPECT_EQ(rcclRegAutoRelease(comm_.get()), ncclSuccess);
  ASSERT_EQ(Population(), 1);
  EXPECT_EQ(Slot(0), reg);
  EXPECT_EQ(reg->localRefs, 0);
  EXPECT_EQ(reg->graphRefs, 1);
  EXPECT_FALSE(reg->rcclAuto.owned);
  EXPECT_TRUE(deregistered_.empty());
  free(reg->ipcInfos[0]);
  free(reg->ipcInfos);
}

TEST_F(RegisterAutoLocalMicrotest, Cleanup_DropsOwnedAndRetiredRecordsWithoutProxyCalls) {
  ncclReg* user = Seed(0x40000, 0x41000);
  ncclIpcRegInfo* userImport = AddPeerImport(user);
  ncclReg* stale = Auto(0x10000, kPage);
  AddPeerImport(stale);
  bufferId_ = 8;
  ncclReg* fresh = Auto(0x10000, kPage);
  ASSERT_NE(fresh, nullptr);
  AddPeerImport(fresh);
  auto* netHandle = static_cast<ncclRegNetHandles*>(calloc(1, sizeof(ncclRegNetHandles)));
  fresh->netHandleHead = netHandle;
  fresh->state |= NET_REG_COMPLETE;
  int netDeregs = 0;
  ScopedHook net(g_ncclNetDeregBuffer, [&](ncclComm*, ncclProxyConnector*, void*) {
    ++netDeregs;
    return ncclSuccess;
  });

  // After an abort the proxies are gone, so only the user's record may still make proxy calls.
  EXPECT_EQ(CleanupAll(), ncclSuccess);
  EXPECT_EQ(deregistered_, (std::vector<ncclIpcRegInfo*>{userImport}));
  EXPECT_EQ(netDeregs, 0);
  EXPECT_TRUE(Retired().empty());
  EXPECT_EQ(comm_->regCache.rcclRetiredBytes, 0u);
}

}  // namespace
