/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Single-GPU ncclGin::put backend-mask dispatch tests with recording ncclGinApi_Put stubs; connection 0 only.

#ifdef __CLANG_RDC__
#error "GinBackendMaskDispatch_test.cpp must stay non-RDC: its macro overrides would ODR-merge with other TUs"
#endif

// Compile only GPI and EFA_GDA so the mask has two bits; skip gin_device_api.h since their real headers are CUDA-only.
#undef NCCL_GIN_PROXY_ENABLE
#define NCCL_GIN_PROXY_ENABLE 0
#undef NCCL_GIN_GDAKI_ENABLE
#define NCCL_GIN_GDAKI_ENABLE 0
#undef NCCL_GIN_ROCSHMEM_GDA_ENABLE
#define NCCL_GIN_ROCSHMEM_GDA_ENABLE 0
#undef NCCL_GIN_ANVIL_SDMA_ENABLE
#define NCCL_GIN_ANVIL_SDMA_ENABLE 0
#undef NCCL_GIN_GPI_ENABLE
#define NCCL_GIN_GPI_ENABLE 1
#undef NCCL_GIN_EFA_GDA_ENABLE
#define NCCL_GIN_EFA_GDA_ENABLE 1
#define _NCCL_GIN_DEVICE_API_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "DeviceTestBase.hpp"

#include "nccl_device/coop.h"
#include "nccl_device/gin/gin_device_common.h"
#include "nccl_device/gin/gin_device_host_common.h"

namespace RcclUnitTesting {
namespace {

// Arguments one recording backend stub saw on its last ncclGinApi_Put::call.
struct PutRecord {
  int calls;
  int peer;
  ncclGinWindow_t dstWin;
  size_t dstOff;
  ncclGinWindow_t srcWin;
  size_t srcOff;
  size_t bytes;
  ncclGinSignalType signalType;
  ncclGinSignal_t signalId;
  bool isStrong;
  ncclGinSignalOp_t signalOp;
  uint64_t signalOpArg;
  bool hasCounter;
  ncclGinCounter_t counterId;
  cuda::thread_scope required;
  cuda::thread_scope given;
  uint32_t optFlags;
};

// ctx.handle points at PutRecord[kNumSlots]; each stub backend writes its own slot.
constexpr int kGpiSlot = 0;
constexpr int kEfaSlot = 1;
constexpr int kNumSlots = 2;

template <int kSlot>
struct RecordingPutStub {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop, int peer, bool, ncclGinWindow_t dstWin, size_t dstOff,
                                      ncclGinWindow_t srcWin, size_t srcOff, size_t bytes,
                                      ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasCounter, ncclGinCounter_t counterId, bool,
                                      ncclGinDescriptorSmem*, cuda::thread_scope required, cuda::thread_scope given,
                                      uint32_t optFlags = ncclGinOptFlagsDefault) {
    PutRecord* rec = static_cast<PutRecord*>(ctx.handle) + kSlot;
    rec->calls += 1;
    rec->peer = peer;
    rec->dstWin = dstWin;
    rec->dstOff = dstOff;
    rec->srcWin = srcWin;
    rec->srcOff = srcOff;
    rec->bytes = bytes;
    rec->signalType = signal.type;
    rec->signalId = signal.indexedSignal.signalId;
    rec->isStrong = signal.isStrong;
    rec->signalOp = signalOp;
    rec->signalOpArg = signalOpArg;
    rec->hasCounter = hasCounter;
    rec->counterId = counterId;
    rec->required = required;
    rec->given = given;
    rec->optFlags = optFlags;
  }
};

// The primary GetSignalPtr/GetCounterPtr declare 3 params but gin__funcs.h's ncclGin_C helpers pass 2, so stub both.
struct UnusedOffsetPtrStub {
  NCCL_DEVICE_INLINE static ncclGinOffsetPtr call(ncclGinCtx, uint32_t) {
    return {nullptr, 0};
  }
};

}  // namespace
}  // namespace RcclUnitTesting

// Stand-ins for the GPI and EFA_GDA leaves; must precede gin__funcs.h so put() instantiates them.
template <>
struct ncclGinApi_Put<NCCL_NET_DEVICE_GIN_GPI> : RcclUnitTesting::RecordingPutStub<RcclUnitTesting::kGpiSlot> {};
template <>
struct ncclGinApi_Put<NCCL_NET_DEVICE_GIN_EFA_GDA> : RcclUnitTesting::RecordingPutStub<RcclUnitTesting::kEfaSlot> {};
template <>
struct ncclGinApi_GetSignalPtr<NCCL_NET_DEVICE_GIN_GPI> : RcclUnitTesting::UnusedOffsetPtrStub {};
template <>
struct ncclGinApi_GetSignalPtr<NCCL_NET_DEVICE_GIN_EFA_GDA> : RcclUnitTesting::UnusedOffsetPtrStub {};
template <>
struct ncclGinApi_GetCounterPtr<NCCL_NET_DEVICE_GIN_GPI> : RcclUnitTesting::UnusedOffsetPtrStub {};
template <>
struct ncclGinApi_GetCounterPtr<NCCL_NET_DEVICE_GIN_EFA_GDA> : RcclUnitTesting::UnusedOffsetPtrStub {};

#include "nccl_device/impl/core__funcs.h"
#include "nccl_device/impl/gin__funcs.h"

namespace RcclUnitTesting {
namespace {

static_assert(NCCL_GIN_BACKEND_MASK_ALL ==
                  ((1u << NCCL_NET_DEVICE_GIN_GPI) | (1u << NCCL_NET_DEVICE_GIN_EFA_GDA)),
              "this TU must compile exactly the GPI and EFA_GDA backends");

struct PutArgs {
  int peer;
  size_t dstOff;
  size_t srcOff;
  size_t bytes;
  ncclGinSignal_t signalId;
  ncclGinCounter_t counterId;
  cuda::thread_scope given;
  cuda::thread_scope required;
  uint32_t optFlags;
};

template <typename Segment, typename RemoteAction>
__global__ void kernelBackendMaskPut(ncclDevComm comm, ncclWindow_t dstWin, ncclWindow_t srcWin, PutArgs args,
                                     RemoteAction remoteAction) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  ncclGin net(comm, 0);
  net.put(ncclTeamWorld(comm), args.peer, dstWin, args.dstOff, srcWin, args.srcOff, args.bytes,
          remoteAction, ncclGin_CounterInc{args.counterId}, ncclCoopThread{}, ncclGin_None{},
          args.given, args.required, args.optFlags, Segment{});
}

// put()'s action helpers take ncclGin (NCCL_GIN_BACKEND_MASK_ALL), so call ncclGinCall directly with a one-bit mask.
__global__ void kernelOneBitMaskCall(PutRecord* records) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  ncclGinCtx_M<(1u << NCCL_NET_DEVICE_GIN_EFA_GDA)> ctx{};
  ctx.backend = NCCL_NET_DEVICE_GIN_GPI;  // Only the singleton path reaches EFA; dispatching on ctx.backend would not.
  ctx.handle = records;
  ncclGinSignalDescriptor signal{};
  signal.type = NCCL_GIN_SIGNAL_TYPE_NONE;
  ncclGinCall<ncclGinApi_Put>(ctx, ncclCoopThread{}, /*peer=*/1, /*hasWins=*/true, /*dstWin=*/ncclGinWindow_t{},
                              /*dstOff=*/size_t{0}, /*srcWin=*/ncclGinWindow_t{}, /*srcOff=*/size_t{0},
                              /*bytes=*/size_t{8}, signal, /*signalOp=*/ncclGinSignalInc, /*signalOpArg=*/uint64_t{0},
                              /*hasCounter=*/false, /*counterId=*/ncclGinCounter_t{0}, /*hasDescriptor=*/false,
                              /*descriptor=*/static_cast<ncclGinDescriptorSmem*>(nullptr),
                              /*required=*/cuda::thread_scope_device, /*given=*/cuda::thread_scope_device);
}

class GinBackendMaskDispatchTest : public DeviceTestBase {
protected:
  static constexpr uintptr_t kDstToken = 0xD57000;
  static constexpr uintptr_t kSrcToken = 0x5C2000;
  static constexpr uint32_t kDstOffset4K = 3;
  static constexpr uint32_t kSrcOffset4K = 7;
  static constexpr int kSignalCount = 8;
  static constexpr uint64_t kSignalAddValue = 7;

  void SetUp() override {
    ASSERT_EQ(setDeviceErr_, hipSuccess);
    ASSERT_NO_FATAL_FAILURE(DeviceTestBase::SetUp());
    ASSERT_NO_FATAL_FAILURE(d_records_.zero());
    ASSERT_NO_FATAL_FAILURE(d_signalShadows_.zero());
    ASSERT_NO_FATAL_FAILURE(d_dstWin_.upload(makeWindow(kDstToken, kDstOffset4K)));
    ASSERT_NO_FATAL_FAILURE(d_srcWin_.upload(makeWindow(kSrcToken, kSrcOffset4K)));
  }

  static size_t ginOffset(uint32_t offset4K, size_t offset) {
    return 4096 * size_t{offset4K} + offset;
  }

  static ncclWindow_vidmem makeWindow(uintptr_t token, uint32_t offset4K) {
    ncclWindow_vidmem win{};
    win.ginOffset4K = offset4K;
    win.ginWinsDefaultBackend[0] = reinterpret_cast<ncclGinWindow_t>(token);
    win.numSegments = 1;
    return win;
  }

  ncclDevComm makeComm(ncclNetDeviceType backend, bool strongSignals = true) {
    ncclDevComm comm{};
    comm.nRanks = 2;
    comm.ginConnectionCount = 1;
    comm.ginConnectionStride = 1;
    comm.backendIndex = 0;
    comm.ginNetDeviceTypes[0] = backend;
    comm.ginHandles[0] = d_records_.ptr;
    comm.ginSignalCount = kSignalCount;
    comm.ginSignalShadows = d_signalShadows_.ptr;
    comm.ginStrongLegacySignals = strongSignals;
    return comm;
  }

  static PutArgs defaultArgs() {
    PutArgs args{};
    args.peer = 1;
    args.dstOff = 40;
    args.srcOff = 24;
    args.bytes = 256;
    args.signalId = 5;
    args.counterId = 3;
    args.given = cuda::thread_scope_device;
    args.required = cuda::thread_scope_block;
    args.optFlags = ncclGinOptFlagsAggregateRequests;
    return args;
  }

  template <typename Segment, typename RemoteAction>
  std::vector<PutRecord> runPut(const ncclDevComm& comm, const PutArgs& args, RemoteAction remoteAction) {
    d_records_.zero();
    kernelBackendMaskPut<Segment><<<1, 1>>>(comm, d_dstWin_.ptr, d_srcWin_.ptr, args, remoteAction);
    syncAndCheck();
    if (HasFatalFailure()) {
      return std::vector<PutRecord>(kNumSlots);
    }
    return d_records_.copyTo();
  }

  // Exactly one backend stub ran, once.
  static void expectOnlyCalled(const std::vector<PutRecord>& records, int slot) {
    for (int i = 0; i < kNumSlots; ++i) {
      EXPECT_EQ(records[i].calls, i == slot ? 1 : 0) << "slot " << i;
    }
  }

  // The backend saw the user's put, with only the required release scope possibly escalated.
  static void expectForwarded(const PutRecord& rec, const PutArgs& args, ncclGinSignalOp_t signalOp,
                              uint64_t signalOpArg, cuda::thread_scope required, bool strongSignals) {
    EXPECT_EQ(rec.peer, args.peer);
    EXPECT_EQ(rec.dstWin, reinterpret_cast<ncclGinWindow_t>(kDstToken));
    EXPECT_EQ(rec.srcWin, reinterpret_cast<ncclGinWindow_t>(kSrcToken));
    EXPECT_EQ(rec.dstOff, ginOffset(kDstOffset4K, args.dstOff));
    EXPECT_EQ(rec.srcOff, ginOffset(kSrcOffset4K, args.srcOff));
    EXPECT_EQ(rec.bytes, args.bytes);
    EXPECT_EQ(rec.signalType, NCCL_GIN_SIGNAL_TYPE_INDEXED);
    EXPECT_EQ(rec.signalId, args.signalId);
    EXPECT_EQ(rec.isStrong, strongSignals);
    EXPECT_EQ(rec.signalOp, signalOp);
    EXPECT_EQ(rec.signalOpArg, signalOpArg);
    EXPECT_TRUE(rec.hasCounter);
    EXPECT_EQ(rec.counterId, args.counterId);
    EXPECT_EQ(rec.required, required);
    EXPECT_EQ(rec.given, args.given);
    EXPECT_EQ(rec.optFlags, args.optFlags);
  }

  hipError_t setDeviceErr_ = hipSetDevice(0);  // Must precede the DeviceBuffers; they allocate on the current device.
  DeviceBuffer<PutRecord> d_records_{kNumSlots};
  DeviceBuffer<ncclWindow_vidmem> d_dstWin_{1};
  DeviceBuffer<ncclWindow_vidmem> d_srcWin_{1};
  DeviceBuffer<uint64_t> d_signalShadows_{kSignalCount};
};

// With two backends compiled, ctx.backend (from comm.ginNetDeviceTypes) picks the ncclGinApi_Put specialization.
TEST_F(GinBackendMaskDispatchTest, MultiBitMaskDispatchesOnCtxBackend) {
  struct Case {
    ncclNetDeviceType backend;
    int slot;
  };
  const Case cases[] = {
      {NCCL_NET_DEVICE_GIN_GPI, kGpiSlot},
      {NCCL_NET_DEVICE_GIN_EFA_GDA, kEfaSlot},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(::testing::Message() << "backend=" << static_cast<int>(c.backend));
    const PutArgs args = defaultArgs();
    const std::vector<PutRecord> records =
        runPut<ncclGin_SegmentDevice>(makeComm(c.backend), args, ncclGin_SignalInc{args.signalId});
    expectOnlyCalled(records, c.slot);
  }
}

// A one-bit mask dispatches to its only backend even when ctx.backend names another one.
TEST_F(GinBackendMaskDispatchTest, OneBitMaskDispatchesToItsBackend) {
  kernelOneBitMaskCall<<<1, 1>>>(d_records_.ptr);
  syncAndCheck();
  ASSERT_FALSE(HasFatalFailure());
  expectOnlyCalled(d_records_.copyTo(), kEfaSlot);
}

// Device-only put forwards peer, 4K-scaled offsets, windows, SignalAdd, counter, flags and release scopes unchanged.
TEST_F(GinBackendMaskDispatchTest, DevicePutForwardsArguments) {
  const PutArgs args = defaultArgs();
  for (bool strongSignals : {true, false}) {
    SCOPED_TRACE(::testing::Message() << "strongSignals=" << strongSignals);
    const std::vector<PutRecord> records =
        runPut<ncclGin_SegmentDevice>(makeComm(NCCL_NET_DEVICE_GIN_GPI, strongSignals), args,
                                      ncclGin_SignalAdd{args.signalId, kSignalAddValue});
    expectOnlyCalled(records, kGpiSlot);
    expectForwarded(records[kGpiSlot], args, ncclGinSignalAdd, kSignalAddValue, args.required, strongSignals);
  }
}

// Non-device buffers with single-segment windows escalate the required release scope to system and keep the rest.
TEST_F(GinBackendMaskDispatchTest, MixedSingleSegmentPutEscalatesRequiredToSystem) {
  const PutArgs args = defaultArgs();
  auto check = [&](auto remoteAction, ncclGinSignalOp_t signalOp, uint64_t signalOpArg) {
    SCOPED_TRACE(::testing::Message() << "signalOp=" << static_cast<int>(signalOp));
    const std::vector<PutRecord> records =
        runPut<ncclGin_SegmentMixed>(makeComm(NCCL_NET_DEVICE_GIN_EFA_GDA), args, remoteAction);
    expectOnlyCalled(records, kEfaSlot);
    expectForwarded(records[kEfaSlot], args, signalOp, signalOpArg, cuda::thread_scope_system, /*strongSignals=*/true);
  };
  check(ncclGin_SignalInc{args.signalId}, ncclGinSignalInc, /*signalOpArg=*/1);
  check(ncclGin_SignalAdd{args.signalId, kSignalAddValue}, ncclGinSignalAdd, kSignalAddValue);
}

}  // namespace
}  // namespace RcclUnitTesting
