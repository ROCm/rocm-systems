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
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ScopedHook.h"
#include "fakes/argcheck_fakes.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/recorder_fakes.h"
#include "fakes/transport_p2p_reg_fakes.h"

// Routes every NCCL_PARAM in register.cc through g_loadParam on each call.
#include "fakes/param_redirect.h"

#include "register.h"

// nccl_fakes.cc keeps these names as seams for p2p-test.cc, so the unit's own definitions are renamed.
#define ncclRegLocalIsValid micro_ncclRegLocalIsValid
#define ncclCommGraphRegister micro_ncclCommGraphRegister
#define ncclCommGraphDeregister micro_ncclCommGraphDeregister
#include "fakes/transport_p2p_reg_seam.h"
#include REGISTER_CC_PATH
#include "fakes/transport_p2p_reg_seam_undef.h"
#undef ncclCommGraphDeregister
#undef ncclCommGraphRegister
#undef ncclRegLocalIsValid

namespace {

constexpr uintptr_t kPage = 0x1000;

void* Ptr(uintptr_t addr) { return reinterpret_cast<void*>(addr); }

// Param values for the redirected NCCL_PARAM bodies; unset names keep their defaults.
std::function<int64_t(const char*, int64_t)> ParamOverride(std::initializer_list<std::pair<const char*, int64_t>> kv) {
  std::vector<std::pair<std::string, int64_t>> table(kv.begin(), kv.end());
  return [table](const char* env, int64_t deft) {
    for (const auto& [name, value] : table) {
      if (name == env) return value;
    }
    return deft;
  };
}

// hipPointerGetAttribute answering the two attributes register.cc queries.
std::function<hipError_t(void*, hipPointer_attribute, hipDeviceptr_t)> PointerAttrs(hipMemoryType memType,
                                                                                     int legacyIpcCapable) {
  return [=](void* data, hipPointer_attribute attr, hipDeviceptr_t) {
    if (attr == HIP_POINTER_ATTRIBUTE_MEMORY_TYPE) *static_cast<hipMemoryType*>(data) = memType;
    if (attr == HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE) *static_cast<int*>(data) = legacyIpcCapable;
    return hipSuccess;
  };
}

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
  g_hipPointerGetAttribute = PointerAttrs(hipMemoryTypeHost, /*legacyIpcCapable=*/0);
  void* handle = nullptr;
  EXPECT_EQ(ncclRegister(comm_.get(), Ptr(0x10000), kPage, false, &handle), ncclSuccess);
  EXPECT_EQ(Population(), 0);
}

TEST_F(RegisterSysmemMicrotest, AddressRangeQueryFails_ReturnsCudaErrorWithoutCaching) {
  // Legacy-IPC device memory, so no later query can fail in its place.
  g_hipPointerGetAttribute = PointerAttrs(hipMemoryTypeDevice, /*legacyIpcCapable=*/1);
  g_hipMemGetAddressRange = [](hipDeviceptr_t*, size_t*, hipDeviceptr_t) { return hipErrorInvalidValue; };
  void* handle = nullptr;
  EXPECT_EQ(ncclRegister(comm_.get(), Ptr(0x10000), kPage, false, &handle), ncclUnhandledCudaError);
  EXPECT_EQ(Population(), 0);
}

TEST_F(RegisterSysmemMicrotest, MemoryTypeQueryFails_ReturnsCudaErrorWithoutCaching) {
  // Legacy-IPC device memory, so no later query can fail in its place.
  auto deviceMemory = PointerAttrs(hipMemoryTypeDevice, /*legacyIpcCapable=*/1);
  g_hipPointerGetAttribute = [deviceMemory](void* data, hipPointer_attribute attr, hipDeviceptr_t ptr) {
    return attr == HIP_POINTER_ATTRIBUTE_MEMORY_TYPE ? hipErrorInvalidValue : deviceMemory(data, attr, ptr);
  };
  void* handle = nullptr;
  EXPECT_EQ(ncclRegister(comm_.get(), Ptr(0x10000), kPage, false, &handle), ncclUnhandledCudaError);
  EXPECT_EQ(Population(), 0);
}

TEST_F(RegisterSysmemMicrotest, CuMemDisabled_RegistersWithoutQueryingMemoryType) {
  g_cuMemEnable = [] { return 0; };
  ScopedHook attrs(g_hipPointerGetAttribute, PointerAttrs(hipMemoryTypeHost, 0));
  void* handle = nullptr;
  EXPECT_EQ(ncclRegister(comm_.get(), Ptr(0x10000), kPage, false, &handle), ncclSuccess);
  EXPECT_EQ(Population(), 1);
  EXPECT_EQ(attrs.calls, 0);
}

#if HIP_VERSION >= 71260540
// Newer HIP: a non-legacy-IPC device buffer is walked segment by segment.

TEST_F(RegisterSysmemMicrotest, LegacyIpcDeviceMemory_RegistersWithoutWalkingSegments) {
  g_hipPointerGetAttribute = PointerAttrs(hipMemoryTypeDevice, /*legacyIpcCapable=*/1);
  ScopedHook retain(g_hipMemRetainAllocationHandle,
                    [](hipMemGenericAllocationHandle_t*, void*) { return hipErrorInvalidValue; });
  void* handle = nullptr;
  EXPECT_EQ(ncclRegister(comm_.get(), Ptr(0x10000), kPage, false, &handle), ncclSuccess);
  EXPECT_EQ(Population(), 1);
  EXPECT_EQ(retain.calls, 0);
}

TEST_F(RegisterSysmemMicrotest, LegacyIpcQueryFails_ReturnsCudaErrorWithoutCaching) {
  auto deviceMemory = PointerAttrs(hipMemoryTypeDevice, /*legacyIpcCapable=*/1);
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
    g_hipPointerGetAttribute = PointerAttrs(hipMemoryTypeDevice, /*legacyIpcCapable=*/0);
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
  g_hipPointerGetAttribute = PointerAttrs(hipMemoryTypeDevice, 0);
  void* handle = nullptr;
  EXPECT_EQ(ncclRegister(comm_.get(), Ptr(0x10000), kPage, false, &handle), ncclSuccess);
  EXPECT_EQ(Population(), 1);
}

TEST_F(RegisterSysmemMicrotest, ManagedMemory_SkipsRegistration) {
  g_hipPointerGetAttribute = PointerAttrs(hipMemoryTypeManaged, 0);
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

}  // namespace
