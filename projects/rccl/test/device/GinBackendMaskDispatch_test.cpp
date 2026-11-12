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
#include <cstring>
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

// ctx.handle points at PutRecord[kNumSlots] (each backend's last call), then a kMaxLoggedCalls log per slot.
constexpr int kGpiSlot = 0;
constexpr int kEfaSlot = 1;
constexpr int kNumSlots = 2;
constexpr int kMaxLoggedCalls = 8;
constexpr int kNumRecords = kNumSlots * (1 + kMaxLoggedCalls);

constexpr int logIndex(int slot, int call) {
  return kNumSlots + slot * kMaxLoggedCalls + call;
}

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
    if (rec->calls <= kMaxLoggedCalls) {
      static_cast<PutRecord*>(ctx.handle)[logIndex(kSlot, rec->calls - 1)] = *rec;
    }
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

template <typename Segment, typename RemoteAction, typename LocalAction>
__global__ void kernelBackendMaskPut(ncclDevComm comm, ncclWindow_t dstWin, ncclWindow_t srcWin, PutArgs args,
                                     RemoteAction remoteAction, LocalAction localAction) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  ncclGin net(comm, 0);
  net.put(ncclTeamWorld(comm), args.peer, dstWin, args.dstOff, srcWin, args.srcOff, args.bytes,
          remoteAction, localAction, ncclCoopThread{}, ncclGin_None{},
          args.given, args.required, args.optFlags, Segment{});
}

// ncclGinPut_v3 arguments that the template put derives from its RemoteAction, LocalAction and Segment types.
struct CPutAction {
  ncclGinSignalType signalType;
  bool isStrong;
  ncclGinSignalOp_t signalOp;
  uint64_t signalOpArg;
  bool isCounter;
  ncclGinSegmentType_t segmentType;
};

// The C API reaches the same backends through ncclGinPut_C_impl, which has its own copy of the multi-segment loop.
__global__ void kernelCPut(ncclDevComm comm, ncclWindow_t dstWin, ncclWindow_t srcWin, PutArgs args, CPutAction a) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  ncclGin_C net(comm, NCCL_GIN_BACKEND_MASK_ALL, 0);
  ncclGinPut_v3(&net, ncclTeamWorld(comm), args.peer, dstWin, args.dstOff, srcWin, args.srcOff, args.bytes,
                a.signalType, /*signalWin=*/nullptr, /*signalOffset=*/0, args.signalId, a.isStrong, a.signalOp,
                a.signalOpArg, a.isCounter, args.counterId, ncclCoopThread{}, /*isDescriptor=*/false,
                /*descriptor=*/nullptr, args.given, args.required, args.optFlags, a.segmentType);
}

constexpr ncclGinSegmentType_t segmentTypeOf(ncclGin_SegmentDevice) {
  return ncclGinSegmentTypeDevice;
}
constexpr ncclGinSegmentType_t segmentTypeOf(ncclGin_SegmentMixed) {
  return ncclGinSegmentTypeMixed;
}
constexpr ncclGinSegmentType_t segmentTypeOf(ncclGin_SegmentHostNuma) {
  return ncclGinSegmentTypeHostNuma;
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

  template <typename Segment, typename RemoteAction, typename LocalAction>
  std::vector<PutRecord> runPut(const ncclDevComm& comm, const PutArgs& args, RemoteAction remote, LocalAction local) {
    d_records_.zero();
    kernelBackendMaskPut<Segment><<<1, 1>>>(comm, d_dstWin_.ptr, d_srcWin_.ptr, args, remote, local);
    syncAndCheck();
    if (HasFatalFailure()) {
      return std::vector<PutRecord>(kNumRecords);
    }
    return d_records_.copyTo();
  }

  std::vector<PutRecord> runCPut(const ncclDevComm& comm, const PutArgs& args, const CPutAction& action) {
    d_records_.zero();
    kernelCPut<<<1, 1>>>(comm, d_dstWin_.ptr, d_srcWin_.ptr, args, action);
    syncAndCheck();
    if (HasFatalFailure()) {
      return std::vector<PutRecord>(kNumRecords);
    }
    return d_records_.copyTo();
  }

  // Exactly one backend stub ran, calls times.
  static void expectOnlyCalled(const std::vector<PutRecord>& records, int slot, int calls = 1) {
    for (int i = 0; i < kNumSlots; ++i) {
      EXPECT_EQ(records[i].calls, i == slot ? calls : 0) << "slot " << i;
    }
  }

  using MemType = decltype(ncclSegmentWindow::memType);
  static_assert(sizeof(MemType) == sizeof(int), "memType is stored through its int representation");
  static constexpr int kDev = hipMemLocationTypeDevice;
  // Before ROCm 7.12 hip_compat.h defines the CUDA name as 3, outside the HIP enum, so it is stored as a raw int.
#ifdef CU_MEM_LOCATION_TYPE_HOST_NUMA
  static constexpr int kNuma = CU_MEM_LOCATION_TYPE_HOST_NUMA;
#else
  static constexpr int kNuma = hipMemLocationTypeHostNuma;
#endif
  static constexpr int kHost = 2; // CU_MEM_LOCATION_TYPE_HOST; ROCm < 7.12 headers have no HIP enumerator for it.

  struct SegmentSpec {
    uintptr_t token;
    size_t size;
    int memType;
  };

  // One chunk the multi-segment put must hand to the backend, in segment-local coordinates.
  struct ExpectedChunk {
    uintptr_t dstToken;
    size_t dstOff;
    uintptr_t srcToken;
    size_t srcOff;
    size_t bytes;
    cuda::thread_scope required;
  };

  // Connections 1..3 get a poisoned token, so reading the wrong connection or the default-backend window shows up.
  // specs is numBackends blocks of numSegments, backend-major like ncclDevrAllocAndPopulateSegmentWindows.
  static void uploadMultiSegmentWindow(DeviceBuffer<ncclWindow_vidmem>& d_win, DeviceBuffer<ncclSegmentWindow>& d_segs,
                                       uintptr_t defaultToken, uint32_t offset4K, const std::vector<SegmentSpec>& specs,
                                       int numBackends = 1) {
    ASSERT_LE(specs.size(), d_segs.count);
    ASSERT_EQ(specs.size() % numBackends, 0u);
    std::vector<ncclSegmentWindow> segs(d_segs.count);
    for (size_t i = 0; i < specs.size(); ++i) {
      for (int c = 0; c < NCCL_GIN_MAX_CONNECTIONS; ++c) {
        segs[i].ginWins[c] = reinterpret_cast<ncclGinWindow_t>(c == 0 ? specs[i].token : specs[i].token ^ 0xC0);
      }
      segs[i].segmentSize = specs[i].size;
      std::memcpy(&segs[i].memType, &specs[i].memType, sizeof(int));
    }
    d_segs.copyFrom(segs);
    ncclWindow_vidmem win = makeWindow(defaultToken, offset4K);
    win.ginMultiSegmentWins = d_segs.ptr;
    win.numSegments = static_cast<int>(specs.size()) / numBackends;
    d_win.upload(win);
  }

  // The slot's per-call log holds exactly the expected chunks; only the last one carries the signal and counter.
  static void expectChunks(const std::vector<PutRecord>& records, int slot, const PutArgs& args,
                           const std::vector<ExpectedChunk>& chunks, ncclGinSignalType lastSignalType,
                           ncclGinSignalOp_t signalOp, uint64_t signalOpArg, bool strongSignals, bool lastHasCounter) {
    ASSERT_LE(chunks.size(), static_cast<size_t>(kMaxLoggedCalls));
    expectOnlyCalled(records, slot, static_cast<int>(chunks.size()));
    for (size_t i = 0; i < chunks.size(); ++i) {
      SCOPED_TRACE(::testing::Message() << "chunk " << i);
      const PutRecord& rec = records[logIndex(slot, static_cast<int>(i))];
      const ExpectedChunk& want = chunks[i];
      const bool isLast = i + 1 == chunks.size();
      EXPECT_EQ(rec.peer, args.peer);
      EXPECT_EQ(rec.dstWin, reinterpret_cast<ncclGinWindow_t>(want.dstToken));
      EXPECT_EQ(rec.dstOff, want.dstOff);
      EXPECT_EQ(rec.srcWin, reinterpret_cast<ncclGinWindow_t>(want.srcToken));
      EXPECT_EQ(rec.srcOff, want.srcOff);
      EXPECT_EQ(rec.bytes, want.bytes);
      EXPECT_EQ(rec.signalType, isLast ? lastSignalType : NCCL_GIN_SIGNAL_TYPE_NONE);
      if (isLast && lastSignalType != NCCL_GIN_SIGNAL_TYPE_NONE) {
        EXPECT_EQ(rec.signalId, args.signalId);
        EXPECT_EQ(rec.isStrong, strongSignals);
      }
      EXPECT_EQ(rec.signalOp, signalOp);
      EXPECT_EQ(rec.signalOpArg, signalOpArg);
      EXPECT_EQ(rec.hasCounter, isLast && lastHasCounter);
      if (lastHasCounter) {
        EXPECT_EQ(rec.counterId, args.counterId);
      }
      EXPECT_EQ(rec.required, want.required);
      EXPECT_EQ(rec.given, args.given);
      EXPECT_EQ(rec.optFlags, args.optFlags);
    }
  }

  // One backend call carried the user's whole put, with only the required release scope possibly escalated.
  static void expectForwarded(const std::vector<PutRecord>& records, int slot, const PutArgs& args,
                              ncclGinSignalOp_t signalOp, uint64_t signalOpArg, cuda::thread_scope required,
                              bool strongSignals) {
    expectChunks(records, slot, args,
                 {{kDstToken, ginOffset(kDstOffset4K, args.dstOff), kSrcToken, ginOffset(kSrcOffset4K, args.srcOff),
                   args.bytes, required}},
                 NCCL_GIN_SIGNAL_TYPE_INDEXED, signalOp, signalOpArg, strongSignals, /*lastHasCounter=*/true);
  }

  // Runs the put through ncclGin::put and ncclGinPut_v3 and expects the same chunks from both.
  template <typename Segment, typename RemoteAction, typename LocalAction>
  void expectPutChunks(const ncclDevComm& comm, int slot, const PutArgs& args, RemoteAction remote, LocalAction local,
                       const std::vector<ExpectedChunk>& chunks, ncclGinSignalType lastSignalType,
                       ncclGinSignalOp_t signalOp, uint64_t signalOpArg, bool strongSignals, bool lastHasCounter) {
    {
      SCOPED_TRACE("ncclGin::put");
      expectChunks(runPut<Segment>(comm, args, remote, local), slot, args, chunks, lastSignalType, signalOp,
                   signalOpArg, strongSignals, lastHasCounter);
    }
    {
      SCOPED_TRACE("ncclGinPut_v3");
      const CPutAction action{lastSignalType, strongSignals, signalOp,
                              signalOpArg,    lastHasCounter, segmentTypeOf(Segment{})};
      expectChunks(runCPut(comm, args, action), slot, args, chunks, lastSignalType, signalOp, signalOpArg,
                   strongSignals, lastHasCounter);
    }
  }

  hipError_t setDeviceErr_ = hipSetDevice(0);  // Must precede the DeviceBuffers; they allocate on the current device.
  DeviceBuffer<PutRecord> d_records_{kNumRecords};
  DeviceBuffer<ncclWindow_vidmem> d_dstWin_{1};
  DeviceBuffer<ncclWindow_vidmem> d_srcWin_{1};
  DeviceBuffer<uint64_t> d_signalShadows_{kSignalCount};
  DeviceBuffer<ncclSegmentWindow> d_dstSegs_{4};
  DeviceBuffer<ncclSegmentWindow> d_srcSegs_{4};
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
        runPut<ncclGin_SegmentDevice>(makeComm(c.backend), args, ncclGin_None{}, ncclGin_None{});
    expectOnlyCalled(records, c.slot);
    EXPECT_EQ(records[c.slot].signalType, NCCL_GIN_SIGNAL_TYPE_NONE) << "ncclGin_None must send no signal";
    EXPECT_FALSE(records[c.slot].hasCounter) << "ncclGin_None must not request a counter";
  }
}

// A one-bit mask dispatches to its only backend even when ctx.backend names another one.
TEST_F(GinBackendMaskDispatchTest, OneBitMaskDispatchesToItsBackend) {
  kernelOneBitMaskCall<<<1, 1>>>(d_records_.ptr);
  syncAndCheck();
  ASSERT_FALSE(HasFatalFailure());
  expectOnlyCalled(d_records_.copyTo(), kEfaSlot);
}

// ginConnectionStride 2 maps world rank 3 to gin rank 3 / 2 = 1 through idivFast32.
TEST_F(GinBackendMaskDispatchTest, ConnectionStrideDividesPeerRank) {
  ncclDevComm comm = makeComm(NCCL_NET_DEVICE_GIN_GPI);
  comm.nRanks = 4;
  comm.ginConnectionStride = 2;
  comm.ginConnectionStride_rcp32 = nccl::utility::idivRcp32(2);
  PutArgs args = defaultArgs();
  args.peer = 3;
  const std::vector<PutRecord> records = runPut<ncclGin_SegmentDevice>(comm, args, ncclGin_None{}, ncclGin_None{});
  expectOnlyCalled(records, kGpiSlot);
  EXPECT_EQ(records[kGpiSlot].peer, 1);
}

// Device-only put forwards peer, 4K-scaled offsets, windows, SignalAdd, counter, flags and release scopes unchanged.
TEST_F(GinBackendMaskDispatchTest, DevicePutForwardsArguments) {
  const PutArgs args = defaultArgs();
  for (bool strongSignals : {true, false}) {
    SCOPED_TRACE(::testing::Message() << "strongSignals=" << strongSignals);
    const std::vector<PutRecord> records =
        runPut<ncclGin_SegmentDevice>(makeComm(NCCL_NET_DEVICE_GIN_GPI, strongSignals), args,
                                      ncclGin_SignalAdd{args.signalId, kSignalAddValue},
                                      ncclGin_CounterInc{args.counterId});
    expectForwarded(records, kGpiSlot, args, ncclGinSignalAdd, kSignalAddValue, args.required, strongSignals);
  }
}

// Non-device buffers with single-segment windows escalate the required release scope to system and keep the rest.
TEST_F(GinBackendMaskDispatchTest, MixedSingleSegmentPutEscalatesRequiredToSystem) {
  const PutArgs args = defaultArgs();
  auto check = [&](auto remoteAction, ncclGinSignalOp_t signalOp, uint64_t signalOpArg) {
    SCOPED_TRACE(::testing::Message() << "signalOp=" << static_cast<int>(signalOp));
    const std::vector<PutRecord> records =
        runPut<ncclGin_SegmentMixed>(makeComm(NCCL_NET_DEVICE_GIN_EFA_GDA), args, remoteAction,
                                     ncclGin_CounterInc{args.counterId});
    expectForwarded(records, kEfaSlot, args, signalOp, signalOpArg, cuda::thread_scope_system, /*strongSignals=*/true);
  };
  check(ncclGin_SignalInc{args.signalId}, ncclGinSignalInc, /*signalOpArg=*/1);
  check(ncclGin_SignalAdd{args.signalId, kSignalAddValue}, ncclGinSignalAdd, kSignalAddValue);
}

// Chunks end where either segment ends; the first HOST_NUMA src chunk escalates to system, later ones drop to thread.
TEST_F(GinBackendMaskDispatchTest, MultiSegmentPutChunksAcrossSegmentsAndEscalatesOnFirstHostNuma) {
  constexpr uintptr_t kS0 = 0x5C0100, kS1 = 0x5C0200, kS2 = 0x5C0300, kS3 = 0x5C0400;
  constexpr uintptr_t kD0 = 0xD50100, kD1 = 0xD50200, kD2 = 0xD50300, kD3 = 0xD50400;
  // Segment 0 of each window is skipped, so the start offsets resolve to (1, 60) and (1, 150).
  ASSERT_NO_FATAL_FAILURE(
      uploadMultiSegmentWindow(d_srcWin_, d_srcSegs_, kSrcToken, /*offset4K=*/0,
                               {{kS0, 40, kNuma}, {kS1, 100, kDev}, {kS2, 300, kNuma}, {kS3, 50, kNuma}}));
  ASSERT_NO_FATAL_FAILURE(
      uploadMultiSegmentWindow(d_dstWin_, d_dstSegs_, kDstToken, /*offset4K=*/0,
                               {{kD0, 80, kDev}, {kD1, 200, kDev}, {kD2, 290, kDev}, {kD3, 100, kDev}}));
  PutArgs args = defaultArgs();
  args.srcOff = 40 + 60;
  args.dstOff = 80 + 150;
  args.bytes = 360;
  // 40 ends src seg 1, 10 ends dst seg 1, 290 ends both seg 2s together, 20 is the remainder in seg 3.
  const std::vector<ExpectedChunk> chunks = {
      {kD1, 150, kS1, 60, 40, args.required},
      {kD1, 190, kS2, 0, 10, cuda::thread_scope_system},
      {kD2, 0, kS2, 10, 290, cuda::thread_scope_thread},
      {kD3, 0, kS3, 0, 20, cuda::thread_scope_thread},
  };
  expectPutChunks<ncclGin_SegmentMixed>(makeComm(NCCL_NET_DEVICE_GIN_GPI, /*strongSignals=*/false), kGpiSlot, args,
      ncclGin_SignalAdd{args.signalId, kSignalAddValue}, ncclGin_CounterInc{args.counterId}, chunks,
      NCCL_GIN_SIGNAL_TYPE_INDEXED, ncclGinSignalAdd, kSignalAddValue, /*strongSignals=*/false,
      /*lastHasCounter=*/true);
}

// Multi-segment src with a one-segment dst still chunks; with no HOST_NUMA src segment required is never escalated.
TEST_F(GinBackendMaskDispatchTest, MultiSegmentPutWithoutHostNumaKeepsRequiredOnFirstChunkOnly) {
  constexpr uintptr_t kS0 = 0x5C1100, kS1 = 0x5C1200, kS2 = 0x5C1300;
  constexpr uintptr_t kD0 = 0xD51100;
  ASSERT_NO_FATAL_FAILURE(uploadMultiSegmentWindow(d_srcWin_, d_srcSegs_, kSrcToken, /*offset4K=*/0,
                                                   {{kS0, 64, kDev}, {kS1, 64, kDev}, {kS2, 64, kDev}}));
  ASSERT_NO_FATAL_FAILURE(
      uploadMultiSegmentWindow(d_dstWin_, d_dstSegs_, kDstToken, /*offset4K=*/0, {{kD0, 1000, kDev}}));
  PutArgs args = defaultArgs();
  args.srcOff = 32;
  args.dstOff = 500;
  args.bytes = 150;
  const std::vector<ExpectedChunk> chunks = {
      {kD0, 500, kS0, 32, 32, args.required},
      {kD0, 532, kS1, 0, 64, cuda::thread_scope_thread},
      {kD0, 596, kS2, 0, 54, cuda::thread_scope_thread},
  };
  {
    SCOPED_TRACE("SignalInc, no counter");
    expectPutChunks<ncclGin_SegmentHostNuma>(makeComm(NCCL_NET_DEVICE_GIN_EFA_GDA), kEfaSlot, args,
        ncclGin_SignalInc{args.signalId}, ncclGin_None{}, chunks, NCCL_GIN_SIGNAL_TYPE_INDEXED, ncclGinSignalInc,
        /*signalOpArg=*/1, /*strongSignals=*/true, /*lastHasCounter=*/false);
  }
  {
    SCOPED_TRACE("no signal, counter");
    expectPutChunks<ncclGin_SegmentHostNuma>(makeComm(NCCL_NET_DEVICE_GIN_EFA_GDA), kEfaSlot, args, ncclGin_None{},
        ncclGin_CounterInc{args.counterId}, chunks, NCCL_GIN_SIGNAL_TYPE_NONE, static_cast<ncclGinSignalOp_t>(0),
        /*signalOpArg=*/0, /*strongSignals=*/true, /*lastHasCounter=*/true);
  }
}

// Offsets exactly on segment boundaries start in the next segment; a >= in findSegmentFromWindow adds a 0-byte chunk.
TEST_F(GinBackendMaskDispatchTest, MultiSegmentPutStartingOnSegmentBoundaries) {
  constexpr uintptr_t kS0 = 0x5C2100, kS1 = 0x5C2200, kS2 = 0x5C2300;
  constexpr uintptr_t kD0 = 0xD52100, kD1 = 0xD52200;
  ASSERT_NO_FATAL_FAILURE(uploadMultiSegmentWindow(d_srcWin_, d_srcSegs_, kSrcToken, /*offset4K=*/0,
                                                   {{kS0, 64, kDev}, {kS1, 64, kDev}, {kS2, 64, kDev}}));
  ASSERT_NO_FATAL_FAILURE(
      uploadMultiSegmentWindow(d_dstWin_, d_dstSegs_, kDstToken, /*offset4K=*/0, {{kD0, 100, kDev}, {kD1, 100, kDev}}));
  PutArgs args = defaultArgs();
  args.srcOff = 64;
  args.dstOff = 100;
  args.bytes = 100;
  const std::vector<ExpectedChunk> chunks = {
      {kD1, 0, kS1, 0, 64, args.required},
      {kD1, 64, kS2, 0, 36, cuda::thread_scope_thread},
  };
  expectPutChunks<ncclGin_SegmentMixed>(makeComm(NCCL_NET_DEVICE_GIN_GPI), kGpiSlot, args,
      ncclGin_SignalInc{args.signalId}, ncclGin_None{}, chunks, NCCL_GIN_SIGNAL_TYPE_INDEXED, ncclGinSignalInc,
      /*signalOpArg=*/1, /*strongSignals=*/true, /*lastHasCounter=*/false);
}

// Segments are registered from the allocation base, but the multi-segment branch looks up the window offset alone.
TEST_F(GinBackendMaskDispatchTest, MultiSegmentPutCurrentlyIgnoresWindowOffset4K) {
  constexpr uintptr_t kS0 = 0x5C3100, kS1 = 0x5C3200;
  constexpr uintptr_t kD0 = 0xD53100;
  ASSERT_NO_FATAL_FAILURE(uploadMultiSegmentWindow(d_srcWin_, d_srcSegs_, kSrcToken, /*offset4K=*/1,
                                                   {{kS0, 4096, kDev}, {kS1, 4096, kNuma}}));
  ASSERT_NO_FATAL_FAILURE(
      uploadMultiSegmentWindow(d_dstWin_, d_dstSegs_, kDstToken, /*offset4K=*/0, {{kD0, 1000, kDev}}));
  PutArgs args = defaultArgs();
  args.srcOff = 100;
  args.dstOff = 0;
  args.bytes = 200;
  // BUG: findSegmentFromWindow gets srcOffset, not 4096 * ginOffset4K + srcOffset, so it reads seg 0; correct is below.
  const std::vector<ExpectedChunk> chunks = {{kD0, 0, kS0, 100, 200, args.required}};
  // const std::vector<ExpectedChunk> chunks = {{kD0, 0, kS1, 100, 200, cuda::thread_scope_system}};
  expectPutChunks<ncclGin_SegmentMixed>(makeComm(NCCL_NET_DEVICE_GIN_GPI), kGpiSlot, args,
      ncclGin_SignalInc{args.signalId}, ncclGin_None{}, chunks, NCCL_GIN_SIGNAL_TYPE_INDEXED, ncclGinSignalInc,
      /*signalOpArg=*/1, /*strongSignals=*/true, /*lastHasCounter=*/false);
}

// AMD ROCm >= 7.12 records host segments as CU_MEM_LOCATION_TYPE_HOST, but the escalation only checks HOST_NUMA.
TEST_F(GinBackendMaskDispatchTest, MultiSegmentPutCurrentlyDoesNotEscalateOnHostSegment) {
  constexpr uintptr_t kS0 = 0x5C4100, kS1 = 0x5C4200;
  constexpr uintptr_t kD0 = 0xD54100;
  ASSERT_NO_FATAL_FAILURE(uploadMultiSegmentWindow(d_srcWin_, d_srcSegs_, kSrcToken, /*offset4K=*/0,
                                                   {{kS0, 64, kHost}, {kS1, 64, kHost}}));
  ASSERT_NO_FATAL_FAILURE(
      uploadMultiSegmentWindow(d_dstWin_, d_dstSegs_, kDstToken, /*offset4K=*/0, {{kD0, 1000, kDev}}));
  PutArgs args = defaultArgs();
  args.srcOff = 32;
  args.dstOff = 0;
  args.bytes = 64;
  // BUG: gin__funcs.h escalates only on HOST_NUMA, so an AMD host src segment keeps the user's scope; correct is below.
  const std::vector<ExpectedChunk> chunks = {
      {kD0, 0, kS0, 32, 32, args.required},
      // {kD0, 0, kS0, 32, 32, cuda::thread_scope_system},
      {kD0, 32, kS1, 0, 32, cuda::thread_scope_thread},
  };
  expectPutChunks<ncclGin_SegmentMixed>(makeComm(NCCL_NET_DEVICE_GIN_GPI), kGpiSlot, args,
      ncclGin_SignalInc{args.signalId}, ncclGin_None{}, chunks, NCCL_GIN_SIGNAL_TYPE_INDEXED, ncclGinSignalInc,
      /*signalOpArg=*/1, /*strongSignals=*/true, /*lastHasCounter=*/false);
}

// backendIndex 1 reads windows from ginMultiSegmentWins[numSegments + seg]; sizes still come from backend 0.
TEST_F(GinBackendMaskDispatchTest, MultiSegmentPutUsesWindowsOfItsBackendIndex) {
  constexpr uintptr_t kS0 = 0x5C5100, kS1 = 0x5C5200, kB1S0 = 0x5C5300, kB1S1 = 0x5C5400;
  constexpr uintptr_t kD0 = 0xD55100, kD1 = 0xD55200, kB1D0 = 0xD55300, kB1D1 = 0xD55400;
  ASSERT_NO_FATAL_FAILURE(uploadMultiSegmentWindow(
      d_srcWin_, d_srcSegs_, kSrcToken, /*offset4K=*/0,
      {{kS0, 64, kDev}, {kS1, 64, kNuma}, {kB1S0, 64, kDev}, {kB1S1, 64, kNuma}}, /*numBackends=*/2));
  ASSERT_NO_FATAL_FAILURE(uploadMultiSegmentWindow(
      d_dstWin_, d_dstSegs_, kDstToken, /*offset4K=*/0,
      {{kD0, 100, kDev}, {kD1, 100, kDev}, {kB1D0, 100, kDev}, {kB1D1, 100, kDev}}, /*numBackends=*/2));
  ncclDevComm comm = makeComm(NCCL_NET_DEVICE_GIN_EFA_GDA);
  comm.backendIndex = 1;
  PutArgs args = defaultArgs();
  args.srcOff = 32;
  args.dstOff = 50;
  args.bytes = 80;
  const std::vector<ExpectedChunk> chunks = {
      {kB1D0, 50, kB1S0, 32, 32, args.required},
      {kB1D0, 82, kB1S1, 0, 18, cuda::thread_scope_system},
      {kB1D1, 0, kB1S1, 18, 30, cuda::thread_scope_thread},
  };
  expectPutChunks<ncclGin_SegmentMixed>(comm, kEfaSlot, args, ncclGin_SignalInc{args.signalId},
      ncclGin_CounterInc{args.counterId}, chunks, NCCL_GIN_SIGNAL_TYPE_INDEXED, ncclGinSignalInc, /*signalOpArg=*/1,
      /*strongSignals=*/true, /*lastHasCounter=*/true);
}

}  // namespace
}  // namespace RcclUnitTesting
