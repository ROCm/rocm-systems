/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Suite F: SDMA factory unit tests (real HSA/KFD queue setup, HW-dependent).
// Exercises gin_anvil_sdma_probe/create/destroy against real SDMA hardware.
// Calls the factory C API directly — does not go through the RCCL GIN plugin
// layer (see Suite G for plugin vtable tests).
// Validation tests (null args, bad ranks) run on any machine; queue creation
// tests skip via GTEST_SKIP when probe() fails (no GPU or SDMA unavailable).

#include "gin/gin_anvil_sdma_factory.h"
#include "sdma/anvil.hpp"

#include "../common/ScopedEnv.hpp"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Stubs: gin_anvil_sdma_factory.cc uses WARN() which references these.
// Self-contained test — no librccl.so link — so provide no-op fallbacks.
#include "debug.h"
uint32_t ncclDebugLevelMask = 0;
uint64_t ncclDebugMask = 0;
thread_local int ncclDebugNoWarn = 0;
void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int, const char*, ...) {}

// Defined in rocmwrap.cc in the real build; referenced by common/EnvVars.cpp's
// GPU probe. This binary never needs cuMem, so report it unsupported.
int ncclCuMemRuntimeSupported() { return 0; }

using RcclUnitTesting::ScopedEnv;

namespace {

int mockAllgatherFail(void*, void*, size_t) { return -1; }

// Mock allgather callbacks receive nRanks via the context pointer (cast to
// intptr_t). bytes_per_rank is the per-rank element size
// (sizeof(gin_anvil_sdma_peer_t)), not the total buffer size — nRanks cannot be
// derived from it.

constexpr uint64_t kHostA = 0xA0A0;

// The record the factory filled in before calling allgather.
const gin_anvil_sdma_peer_t* ownRecord(const gin_anvil_sdma_peer_t* peers, int nRanks) {
  for (int i = 0; i < nRanks; ++i) {
    if (peers[i].device_id >= 0) return &peers[i];
  }
  return nullptr;
}

int mockAllgatherLeaveInvalidPeer(void* ctx, void* buf, size_t) {
  const int nRanks = static_cast<int>(reinterpret_cast<intptr_t>(ctx));
  auto* peers = static_cast<gin_anvil_sdma_peer_t*>(buf);
  for (int i = 0; i < nRanks; ++i) {
    if (i == 1) peers[i].device_id = -1;
  }
  return 0;
}

int mockAllgatherFillDevices(void* ctx, void* buf, size_t) {
  const int nRanks = static_cast<int>(reinterpret_cast<intptr_t>(ctx));
  auto* peers = static_cast<gin_anvil_sdma_peer_t*>(buf);
  const gin_anvil_sdma_peer_t* self = ownRecord(peers, nRanks);
  const uint64_t host = self ? self->host_id : 0;
  for (int i = 0; i < nRanks; ++i) {
    if (peers[i].device_id < 0) peers[i] = gin_anvil_sdma_peer_t{i, 0, host};
  }
  return 0;
}

// Every other rank sits on another host and reports g_remoteDevice, an ordinal on
// its own host that need not exist here.
int g_remoteDevice = 0;
int mockAllgatherRemoteHost(void* ctx, void* buf, size_t) {
  const int nRanks = static_cast<int>(reinterpret_cast<intptr_t>(ctx));
  auto* peers = static_cast<gin_anvil_sdma_peer_t*>(buf);
  const gin_anvil_sdma_peer_t* self = ownRecord(peers, nRanks);
  const uint64_t host = self ? self->host_id : 0;
  for (int i = 0; i < nRanks; ++i) {
    if (peers[i].device_id < 0) peers[i] = gin_anvil_sdma_peer_t{g_remoteDevice, 0, host + 1};
  }
  return 0;
}

// Rank i (other than the caller) reports g_peerTable[i]: an ordinal and an offset added to the
// caller's host id, so an offset of 0 is a peer on this host.
struct PeerSpec {
  int device;
  uint64_t hostOffset;
};
std::vector<PeerSpec> g_peerTable;
int mockAllgatherPeerTable(void* ctx, void* buf, size_t) {
  const int nRanks = static_cast<int>(reinterpret_cast<intptr_t>(ctx));
  auto* peers = static_cast<gin_anvil_sdma_peer_t*>(buf);
  const gin_anvil_sdma_peer_t* self = ownRecord(peers, nRanks);
  const uint64_t host = self ? self->host_id : 0;
  for (int i = 0; i < nRanks; ++i) {
    if (peers[i].device_id < 0) {
      const PeerSpec& s = g_peerTable[static_cast<size_t>(i)];
      peers[i] = gin_anvil_sdma_peer_t{s.device, 0, host + s.hostOffset};
    }
  }
  return 0;
}

struct CreateOut {
  gin_anvil_sdma_handle_t handle{nullptr};
  void* gpu_handles{nullptr};
  uint64_t* sdma_dirty{nullptr};
};

void* mockCtx(int nRanks) { return reinterpret_cast<void*>(static_cast<intptr_t>(nRanks)); }

int tryCreate(int nRanks, int myRank, int deviceId, int (*allgather)(void*, void*, size_t),
              int num_channels, CreateOut* out, int remoteHostsOk = 0) {
  return gin_anvil_sdma_create(nRanks, myRank, deviceId, kHostA, remoteHostsOk, allgather, mockCtx(nRanks),
                               num_channels, &out->handle, &out->gpu_handles, &out->sdma_dirty);
}

void destroyOut(CreateOut* out) {
  if (out->handle) gin_anvil_sdma_destroy(out->handle);
  out->handle = nullptr;
  out->gpu_handles = nullptr;
  out->sdma_dirty = nullptr;
}

class GinAnvilSdmaFactoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int ndev = 0;
    hipError_t err = hipGetDeviceCount(&ndev);
    if (err == hipErrorNoDevice || ndev < 1) return;
    ASSERT_EQ(err, hipSuccess);
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
  }
};

}  // namespace

// F1: probe returns 0 or 1 depending on HIP + Anvil init.
TEST_F(GinAnvilSdmaFactoryTest, Probe_ReturnsZeroOrOne) {
  const int rc = gin_anvil_sdma_probe();
  EXPECT_TRUE(rc == 0 || rc == 1);
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev < 1) {
    EXPECT_EQ(rc, 0);
  }
}

// F2: null / invalid output and rank arguments.
TEST_F(GinAnvilSdmaFactoryTest, Create_NullOutParams) {
  CreateOut out{};
  EXPECT_EQ(gin_anvil_sdma_create(1, 0, 0, kHostA, 0, mockAllgatherFillDevices, mockCtx(1), 1,
                                  nullptr, &out.gpu_handles, &out.sdma_dirty),
            -1);
  EXPECT_EQ(gin_anvil_sdma_create(1, 0, 0, kHostA, 0, mockAllgatherFillDevices, mockCtx(1), 1, &out.handle,
                                  nullptr, &out.sdma_dirty),
            -1);
  EXPECT_EQ(gin_anvil_sdma_create(1, 0, 0, kHostA, 0, mockAllgatherFillDevices, mockCtx(1), 1, &out.handle,
                                  &out.gpu_handles, nullptr),
            -1);
  EXPECT_EQ(gin_anvil_sdma_create(1, 0, 0, kHostA, 0, nullptr, mockCtx(1), 1, &out.handle, &out.gpu_handles,
                                  &out.sdma_dirty),
            -1);
}

TEST_F(GinAnvilSdmaFactoryTest, Create_InvalidRankArgs) {
  CreateOut out{};
  EXPECT_EQ(tryCreate(0, 0, 0, mockAllgatherFillDevices, 1, &out), -1);
  EXPECT_EQ(tryCreate(2, -1, 0, mockAllgatherFillDevices, 1, &out), -1);
  EXPECT_EQ(tryCreate(2, 2, 0, mockAllgatherFillDevices, 1, &out), -1);
}

// F3: allgather failure.
TEST_F(GinAnvilSdmaFactoryTest, Create_AllgatherFail) {
  CreateOut out{};
  EXPECT_EQ(tryCreate(2, 0, 0, mockAllgatherFail, 1, &out), -1);
  EXPECT_EQ(out.handle, nullptr);
}

// F4: post-allgather invalid device id.
TEST_F(GinAnvilSdmaFactoryTest, Create_InvalidPeerDev) {
  CreateOut out{};
  EXPECT_EQ(tryCreate(2, 0, 0, mockAllgatherLeaveInvalidPeer, 1, &out), -1);
  EXPECT_EQ(out.handle, nullptr);
}

// F5–F7: success path (skipped when probe fails).
TEST_F(GinAnvilSdmaFactoryTest, Create_Success_1Rank) {
  if (gin_anvil_sdma_probe() <= 0) {
    GTEST_SKIP() << "Anvil SDMA probe failed (no GPU or USE_SDMA off)";
  }

  CreateOut out{};
  ASSERT_EQ(tryCreate(1, 0, 0, mockAllgatherFillDevices, 1, &out), 0);
  EXPECT_NE(out.handle, nullptr);
  EXPECT_NE(out.gpu_handles, nullptr);
  EXPECT_NE(out.sdma_dirty, nullptr);

  EXPECT_EQ(gin_anvil_sdma_get_n_ranks(out.handle), 1);
  EXPECT_EQ(gin_anvil_sdma_get_num_channels(out.handle), 1);
  EXPECT_GE(gin_anvil_sdma_get_channel_stride(out.handle), 0);
  EXPECT_LE(gin_anvil_sdma_get_channel_stride(out.handle), 1);

  uint64_t dirty = 0xDEADBEEFULL;
  ASSERT_EQ(hipMemcpy(&dirty, out.sdma_dirty, sizeof(dirty), hipMemcpyDeviceToHost), hipSuccess);
  EXPECT_EQ(dirty, 0ULL);

  destroyOut(&out);
}

TEST_F(GinAnvilSdmaFactoryTest, Create_NumChannelsClamp) {
  if (gin_anvil_sdma_probe() <= 0) {
    GTEST_SKIP() << "Anvil SDMA probe failed";
  }

  CreateOut out{};
  ASSERT_EQ(tryCreate(1, 0, 0, mockAllgatherFillDevices, 0, &out), 0);
  EXPECT_EQ(gin_anvil_sdma_get_num_channels(out.handle), 1);
  destroyOut(&out);

  ASSERT_EQ(tryCreate(1, 0, 0, mockAllgatherFillDevices, 99, &out), 0);
  EXPECT_EQ(gin_anvil_sdma_get_num_channels(out.handle), 8);
  destroyOut(&out);
}

TEST_F(GinAnvilSdmaFactoryTest, SpreadChannels_Env) {
  if (gin_anvil_sdma_probe() <= 0) {
    GTEST_SKIP() << "Anvil SDMA probe failed";
  }

  {
    ScopedEnv spread("NCCL_GIN_ANVIL_SDMA_SPREAD_CHANNELS", "0");
    CreateOut out{};
    ASSERT_EQ(tryCreate(1, 0, 0, mockAllgatherFillDevices, 1, &out), 0);
    EXPECT_EQ(gin_anvil_sdma_get_channel_stride(out.handle), 0);
    destroyOut(&out);
  }

  {
    ScopedEnv spread("NCCL_GIN_ANVIL_SDMA_SPREAD_CHANNELS", "1");
    CreateOut out{};
    ASSERT_EQ(tryCreate(1, 0, 0, mockAllgatherFillDevices, 1, &out), 0);
    EXPECT_EQ(gin_anvil_sdma_get_channel_stride(out.handle), 1);
    destroyOut(&out);
  }

  unsetenv("NCCL_GIN_ANVIL_SDMA_SPREAD_CHANNELS");
  CreateOut outDefault{};
  ASSERT_EQ(tryCreate(1, 0, 0, mockAllgatherFillDevices, 1, &outDefault), 0);
  EXPECT_EQ(gin_anvil_sdma_get_channel_stride(outDefault.handle), 1);
  destroyOut(&outDefault);
}

// F8: destroy null and valid.
TEST_F(GinAnvilSdmaFactoryTest, Destroy_NullAndValid) {
  gin_anvil_sdma_destroy(nullptr);
  if (gin_anvil_sdma_probe() <= 0) {
    GTEST_SKIP() << "Anvil SDMA probe failed";
  }
  CreateOut out{};
  ASSERT_EQ(tryCreate(1, 0, 0, mockAllgatherFillDevices, 1, &out), 0);
  destroyOut(&out);
}

// F9: getters on null handle.
TEST_F(GinAnvilSdmaFactoryTest, Getters_NullHandle) {
  EXPECT_EQ(gin_anvil_sdma_get_n_ranks(nullptr), 0);
  EXPECT_EQ(gin_anvil_sdma_get_num_channels(nullptr), 0);
  EXPECT_EQ(gin_anvil_sdma_get_channel_stride(nullptr), 0);
}

// F10: multi-channel loopback (exercises channel stride + multi-queue setup).
// Multi-rank (cross-device) SDMA queue creation requires peer access setup
// that only works within RCCL's full init path; tested via alltoall_perf -D3.
TEST_F(GinAnvilSdmaFactoryTest, Create_MultiChannel) {
  if (gin_anvil_sdma_probe() <= 0) {
    GTEST_SKIP() << "Anvil SDMA probe failed";
  }

  CreateOut out{};
  ASSERT_EQ(tryCreate(1, 0, 0, mockAllgatherFillDevices, 4, &out), 0);
  EXPECT_EQ(gin_anvil_sdma_get_n_ranks(out.handle), 1);
  EXPECT_EQ(gin_anvil_sdma_get_num_channels(out.handle), 4);
  EXPECT_GE(gin_anvil_sdma_get_channel_stride(out.handle), 0);
  EXPECT_LE(gin_anvil_sdma_get_channel_stride(out.handle), 1);
  destroyOut(&out);
}

namespace {
std::vector<void*> queueHandles(const CreateOut& out, int nRanks) {
  std::vector<void*> handles(static_cast<size_t>(nRanks), nullptr);
  EXPECT_EQ(hipMemcpy(handles.data(), out.gpu_handles, handles.size() * sizeof(void*), hipMemcpyDeviceToHost),
            hipSuccess);
  return handles;
}
}  // namespace

// In an MI450 fabric pod, a peer on another host reports an ordinal from its own
// host, which may not exist here or may equal this rank's own. Either way it gets
// a queue of its own: a queue is an in-order ring, and sharing this rank's would
// serialize its copies behind the self copies.
TEST_F(GinAnvilSdmaFactoryTest, Create_RemoteHostPeerGetsItsOwnQueue) {
  if (gin_anvil_sdma_probe() <= 0) {
    GTEST_SKIP() << "Anvil SDMA probe failed";
  }
  int ndev = 0;
  ASSERT_EQ(hipGetDeviceCount(&ndev), hipSuccess);
  for (int ordinal : {0, ndev + 1}) {
    SCOPED_TRACE("remote ordinal " + std::to_string(ordinal));
    g_remoteDevice = ordinal;
    CreateOut out{};
    ASSERT_EQ(tryCreate(2, 0, 0, mockAllgatherRemoteHost, 1, &out, /*remoteHostsOk=*/1), 0);
    EXPECT_EQ(gin_anvil_sdma_get_n_ranks(out.handle), 2);
    const std::vector<void*> h = queueHandles(out, 2);
    EXPECT_NE(h[0], nullptr);
    EXPECT_NE(h[1], nullptr);
    EXPECT_NE(h[1], h[0]);
    destroyOut(&out);
  }
}

// The 2P8G case: a peer on this host and a peer on another host with the same
// ordinal must not share a queue.
TEST_F(GinAnvilSdmaFactoryTest, Create_LocalAndRemotePeerWithTheSameOrdinalGetSeparateQueues) {
  if (gin_anvil_sdma_probe() <= 0) {
    GTEST_SKIP() << "Anvil SDMA probe failed";
  }
  int ndev = 0;
  ASSERT_EQ(hipGetDeviceCount(&ndev), hipSuccess);
  if (ndev < 2) GTEST_SKIP() << "needs a second local GPU for the same-host peer";
  g_peerTable = {{0, 0}, {1, 0}, {1, 0x10}};

  CreateOut out{};
  ASSERT_EQ(tryCreate(3, 0, 0, mockAllgatherPeerTable, 1, &out, /*remoteHostsOk=*/1), 0);
  const std::vector<void*> h = queueHandles(out, 3);
  EXPECT_NE(h[1], nullptr);
  EXPECT_NE(h[2], nullptr);
  EXPECT_NE(h[2], h[1]);
  EXPECT_NE(h[2], h[0]);
  destroyOut(&out);
}

TEST_F(GinAnvilSdmaFactoryTest, Create_RemotePeersOnDifferentHostsWithTheSameOrdinalGetSeparateQueues) {
  if (gin_anvil_sdma_probe() <= 0) {
    GTEST_SKIP() << "Anvil SDMA probe failed";
  }
  g_peerTable = {{0, 0}, {1, 0x20}, {1, 0x21}};

  CreateOut out{};
  ASSERT_EQ(tryCreate(3, 0, 0, mockAllgatherPeerTable, 1, &out, /*remoteHostsOk=*/1), 0);
  const std::vector<void*> h = queueHandles(out, 3);
  EXPECT_NE(h[1], nullptr);
  EXPECT_NE(h[2], nullptr);
  EXPECT_NE(h[2], h[1]);
  destroyOut(&out);
}

// Two GPUs on the same remote host: the key must name the GPU as well as the host.
TEST_F(GinAnvilSdmaFactoryTest, Create_RemotePeersOnTheSameHostWithDifferentOrdinalsGetSeparateQueues) {
  if (gin_anvil_sdma_probe() <= 0) {
    GTEST_SKIP() << "Anvil SDMA probe failed";
  }
  g_peerTable = {{0, 0}, {0, 0x40}, {1, 0x40}};

  CreateOut out{};
  ASSERT_EQ(tryCreate(3, 0, 0, mockAllgatherPeerTable, 1, &out, /*remoteHostsOk=*/1), 0);
  const std::vector<void*> h = queueHandles(out, 3);
  EXPECT_NE(h[1], nullptr);
  EXPECT_NE(h[2], nullptr);
  EXPECT_NE(h[2], h[1]);
  destroyOut(&out);
}

// Queues live in the process-global Anvil singleton and a later communicator
// reuses them, so a second create must not take more queues for the same peer.
TEST_F(GinAnvilSdmaFactoryTest, Create_RemotePeerQueueReusedByLaterCreate) {
  if (gin_anvil_sdma_probe() <= 0) {
    GTEST_SKIP() << "Anvil SDMA probe failed";
  }
  g_peerTable = {{0, 0}, {3, 0x30}};

  const uint32_t before = sdma_anvil::anvil.queuesUsed();
  CreateOut first{};
  ASSERT_EQ(tryCreate(2, 0, 0, mockAllgatherPeerTable, 1, &first, /*remoteHostsOk=*/1), 0);
  const uint32_t afterFirst = sdma_anvil::anvil.queuesUsed();
  EXPECT_GT(afterFirst, before);
  CreateOut second{};
  ASSERT_EQ(tryCreate(2, 0, 0, mockAllgatherPeerTable, 1, &second, /*remoteHostsOk=*/1), 0);
  EXPECT_EQ(sdma_anvil::anvil.queuesUsed(), afterFirst);
  EXPECT_EQ(queueHandles(second, 2)[1], queueHandles(first, 2)[1]);
  destroyOut(&first);
  destroyOut(&second);
}

// rocSHMEM's anvil_engine_map_gtest pins the engine-choice helpers; these pin
// that connect and connectRemotePeer feed them this process's live queue load. Earlier cases leave
// queues on the singleton, so the expected engine is computed from the load seen
// just before each connect rather than assumed.
TEST(AnvilEngineChoice, RemotePeerQueueGoesOnTheLeastLoadedUsableEngine) {
  if (gin_anvil_sdma_probe() <= 0) {
    GTEST_SKIP() << "Anvil SDMA probe failed";
  }
  auto& anvil = sdma_anvil::anvil;
  const uint64_t keys[] = {0xE5C0000000000001ull, 0xE5C0000000000002ull};
  // The second connect's expected engine includes the first's queue in its load.
  for (uint64_t key : keys) {
    ASSERT_EQ(anvil.getRemotePeerQueue(key, 0), nullptr);
    const int expected = sdma_anvil::leastLoadedUsableEngine(anvil.queuesPerEngine());
    ASSERT_TRUE(anvil.connectRemotePeer(/*srcDeviceId=*/0, key, /*numChannels=*/1));
    const sdma_anvil::SdmaQueue* q = anvil.getRemotePeerQueue(key, 0);
    ASSERT_NE(q, nullptr);
    EXPECT_EQ(q->engineId() % 2, 0u);
    EXPECT_EQ(static_cast<int>(q->engineId()), expected);
  }
  for (uint64_t key : keys) anvil.disconnectRemotePeer(key);
}

TEST(AnvilEngineChoice, LocalPeerQueuesGoOnTheLeastLoadedPreferredEngine) {
  if (gin_anvil_sdma_probe() <= 0) {
    GTEST_SKIP() << "Anvil SDMA probe failed";
  }
  int ndev = 0;
  ASSERT_EQ(hipGetDeviceCount(&ndev), hipSuccess);
  auto& anvil = sdma_anvil::anvil;
  std::vector<int> dsts;
  for (int d = 1; d < ndev && dsts.size() < 2; ++d) {
    if (anvil.getSdmaQueue(0, d, 0) == nullptr) dsts.push_back(d);
  }
  if (dsts.size() < 2) GTEST_SKIP() << "needs two local GPUs that device 0 has no queue to yet";

  int checked = 0;
  for (int d : dsts) {
    SCOPED_TRACE("0 -> " + std::to_string(d));
    const std::vector<uint32_t> load = anvil.queuesPerEngine();
    sdma_anvil::EnablePeerAccess(0, d);
    ASSERT_TRUE(anvil.connect(0, d, /*numChannels=*/1));
    const sdma_anvil::SdmaQueue* q = anvil.getSdmaQueue(0, d, 0);
    ASSERT_NE(q, nullptr);
    const sdma_anvil::EngineSelection& sel = q->selection();
    if (!sel.usedPreferred) continue;  // HSA gave no mask; the OAM map chose
    EXPECT_EQ(static_cast<int>(q->engineId()), sdma_anvil::leastLoadedPreferredEngine(sel.preferredMask, load))
        << "mask=0x" << std::hex << sel.preferredMask;
    ++checked;
  }
  for (int d : dsts) anvil.disconnectDevice(d);
  if (checked == 0) GTEST_SKIP() << "HSA reported no preferred engine for these pairs";
}

// Outside a fabric pod (MI300, MI350, or MI450 without one) SDMA cannot reach
// another host, so a remote peer fails creation rather than aliasing a local GPU.
TEST_F(GinAnvilSdmaFactoryTest, Create_RemoteHostPeerRejectedOutsidePod) {
  g_remoteDevice = 0;
  CreateOut out{};
  EXPECT_EQ(tryCreate(2, 0, 0, mockAllgatherRemoteHost, 1, &out, /*remoteHostsOk=*/0), -1);
  EXPECT_EQ(out.handle, nullptr);
}

// The CPX fold is not visible on a queue handle, and getSdmaEngineIdFromOamMap is
// private. These assert when the fold runs, including the in-range diagonal a
// partition must still fold. The engine the fold picks is rocSHMEM's contract and
// is pinned by its anvil_engine_map_gtest.
TEST(AnvilOamMapEngine, CpxPartitionFoldsTheInRangeDiagonal) {
  EXPECT_TRUE(sdma_anvil::oamMapEngineNeedsFold(/*numSdmaXgmiEngines=*/0, /*numSdmaEnginesTotal=*/2,
                                               /*doubledEngineId=*/0));
}

TEST(AnvilOamMapEngine, SpxInRangeDoesNotFold) {
  EXPECT_FALSE(sdma_anvil::oamMapEngineNeedsFold(/*numSdmaXgmiEngines=*/8, /*numSdmaEnginesTotal=*/16,
                                                /*doubledEngineId=*/0));
  EXPECT_FALSE(sdma_anvil::oamMapEngineNeedsFold(8, 16, 14));
}

TEST(AnvilOamMapEngine, OutOfRangeFoldsEvenWithXgmiEngines) {
  EXPECT_TRUE(sdma_anvil::oamMapEngineNeedsFold(/*numSdmaXgmiEngines=*/8, /*numSdmaEnginesTotal=*/2,
                                               /*doubledEngineId=*/14));
}

TEST(AnvilOamMapEngine, ZeroEnginesDoesNotFold) {
  EXPECT_FALSE(sdma_anvil::oamMapEngineNeedsFold(0, 0, 0));
}

// pciFunctionBus decides getOamId's fallback: physBusId differing from busId is what gives it a
// second candidate to read, so a tail it refuses to parse means the physical-BDF read never runs.
// getBusId lowercases an "00000000:00:00.0" buffer, so the function digit is the last character.
TEST(AnvilPciFunctionBus, PartitionTailYieldsFunctionAndPhysicalBdf) {
  const auto loc = sdma_anvil::pciFunctionBus("00000000:1b:00.1");
  EXPECT_EQ(loc.function, 1);
  EXPECT_EQ(loc.busId, "00000000:1b:00.1");
  EXPECT_EQ(loc.physBusId, "00000000:1b:00.0");  // the second candidate getOamId reads
}

// PCI function numbers are three bits, so '0'-'7' is the whole range and both ends must parse.
TEST(AnvilPciFunctionBus, WholeDigitRangeParses) {
  for (int fn = 0; fn <= 7; ++fn) {
    const std::string bdf = "00000000:1b:00." + std::to_string(fn);
    const auto loc = sdma_anvil::pciFunctionBus(bdf);
    EXPECT_EQ(loc.function, fn) << bdf;
    EXPECT_EQ(loc.physBusId, "00000000:1b:00.0") << bdf;
  }
}

// Function 0 parses but rewrites to itself, so physBusId == busId collapses getOamId's candidate
// list to one entry -- the same shape as an unparsable tail, reached for a different reason.
TEST(AnvilPciFunctionBus, FunctionZeroLeavesOneCandidate) {
  const auto loc = sdma_anvil::pciFunctionBus("00000000:1b:00.0");
  EXPECT_EQ(loc.function, 0);
  EXPECT_EQ(loc.physBusId, loc.busId);
}

// A tail outside the range is not function 0: it stays -1 so foldOamMapEngine's unreadable-function
// path is what runs, and busId is left untouched rather than rewritten to something that does not
// exist. '8' and '9' are the ones that matter -- they are digits, so a std::isdigit test here would
// accept them and invent a function PCI cannot express.
TEST(AnvilPciFunctionBus, TailOutsideRangeIsNotFunctionZero) {
  for (const char* bdf : {"00000000:1b:00.8", "00000000:1b:00.9", "00000000:1b:00.f",
                          "00000000:1b:00.", "00000000:1b:00.a"}) {
    const auto loc = sdma_anvil::pciFunctionBus(bdf);
    EXPECT_EQ(loc.function, -1) << bdf;
    EXPECT_EQ(loc.physBusId, loc.busId) << bdf;
    EXPECT_EQ(loc.busId, bdf) << bdf;
  }
}

// The parse is the last character and nothing else, so a BDF with no function field at all is read
// as function 0 whenever it happens to end in a function digit. getBusId always writes a
// "00000000:00:00.0" buffer so this does not arise today, and pinning it says which invariant
// getOamId leans on: if that format ever loses its ".N" suffix, this returns a function and a
// physBusId equal to busId rather than refusing, and the fallback read silently stops happening.
TEST(AnvilPciFunctionBus, ParsesTheLastCharacterOnlyNotTheBdfShape) {
  const auto noFn = sdma_anvil::pciFunctionBus("0000:1b:00");
  EXPECT_EQ(noFn.function, 0);
  EXPECT_EQ(noFn.physBusId, noFn.busId);
  // A bare digit is accepted for the same reason: nothing checks the shape.
  EXPECT_EQ(sdma_anvil::pciFunctionBus("3").function, 3);
}

// back() on an empty string is undefined, so the empty guard is load-bearing rather than defensive.
TEST(AnvilPciFunctionBus, EmptyInputIsRefusedWithoutReadingTheTail) {
  const auto loc = sdma_anvil::pciFunctionBus("");
  EXPECT_EQ(loc.function, -1);
  EXPECT_TRUE(loc.busId.empty());
  EXPECT_TRUE(loc.physBusId.empty());
}

// The budget observable. Without it the refusal in connect() and the per-destination teardown in
// disconnectDevice() are only visible in a log line, so neither the up-front refusal nor the
// rollback that follows a part-way failure has anything a case can assert.
//
// Both of connect()'s failure exits have to land on the same count: the budget compare returns
// before touching sdma_channels_, and a create that fails after some channels succeeded runs
// rollback(), which gives back only the valid queues it charged. The request below is far past any
// partition's budget, so the first exit is the expected one and the second is the fallback when a
// node does not report a budget at all.
//
// disconnectDevice() is deliberately not called on the success branch: the channel map is
// process-global, so dropping device 0 would destroy queues the fixture cases still hold.
TEST(AnvilQueueBudget, FailedConnectDoesNotChargeTheBudget) {
  if (gin_anvil_sdma_probe() <= 0) {
    GTEST_SKIP() << "Anvil SDMA probe failed";
  }
  auto& anvil = sdma_anvil::anvil;
  const uint32_t before = anvil.queuesUsed();
  if (anvil.connect(/*srcDeviceId=*/0, /*dstDeviceId=*/0, /*numChannels=*/1 << 20)) {
    GTEST_SKIP() << "node satisfied an over-subscribed connect; neither failure exit was reached";
  }
  EXPECT_EQ(anvil.queuesUsed(), before);
}

// F11: spread env atoi paths (non-zero/non-one numeric and invalid string).
TEST_F(GinAnvilSdmaFactoryTest, SpreadChannels_EnvAtoi) {
  if (gin_anvil_sdma_probe() <= 0) {
    GTEST_SKIP() << "Anvil SDMA probe failed";
  }

  {
    ScopedEnv spread("NCCL_GIN_ANVIL_SDMA_SPREAD_CHANNELS", "2");
    CreateOut out{};
    ASSERT_EQ(tryCreate(1, 0, 0, mockAllgatherFillDevices, 1, &out), 0);
    EXPECT_EQ(gin_anvil_sdma_get_channel_stride(out.handle), 1);
    destroyOut(&out);
  }

  {
    ScopedEnv spread("NCCL_GIN_ANVIL_SDMA_SPREAD_CHANNELS", "x");
    CreateOut out{};
    ASSERT_EQ(tryCreate(1, 0, 0, mockAllgatherFillDevices, 1, &out), 0);
    EXPECT_EQ(gin_anvil_sdma_get_channel_stride(out.handle), 0);
    destroyOut(&out);
  }
}
