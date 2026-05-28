/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include "TestBed.hpp"
#include "TransportUtils.hpp"

namespace RcclUnitTesting
{

TEST(TransportTest, CollNetRecvSetup) {
  constexpr int nranks = 2;
  constexpr int nNodes = 2;

  // --- Setup comm ---
  ncclComm comm = {};
  comm.rank = 0;
  comm.nRanks = nranks;
  comm.nNodes = nNodes;
  comm.node = 0;
  ncclPeerInfo peerInfo[3] = {};
  comm.peerInfo = peerInfo;

  // --- Setup channel ---
  ncclChannel channel = {};
  static ncclChannelPeer peerArray[3];
  static ncclChannelPeer* peerPtrs[3] = {
    &peerArray[0], &peerArray[1], &peerArray[2]
  };
  channel.peers = peerPtrs;

  // Step 1: Allocate device-side array of ncclDevChannelPeer
  ncclDevChannelPeer* devPeerArrayDevice;
  hipMalloc(&devPeerArrayDevice, sizeof(ncclDevChannelPeer) * 3);

  // Step 2: Create host-side array of device pointers
  ncclDevChannelPeer* devPeerPtrsHost[3] = {
    devPeerArrayDevice + 0,
    devPeerArrayDevice + 1,
    devPeerArrayDevice + 2
  };

  // Step 3: Allocate device-side array of device pointers
  ncclDevChannelPeer** devPeerPtrsDevice;
  hipMalloc(&devPeerPtrsDevice, sizeof(ncclDevChannelPeer*) * 3);

  // Step 4: Copy host-side array of device pointers to device
  hipMemcpy(devPeerPtrsDevice, devPeerPtrsHost, sizeof(ncclDevChannelPeer*) * 3, hipMemcpyHostToDevice);

  // Step 5: Set in channel
  channel.devPeers = devPeerPtrsDevice;
  // --- Setup transportComm ---
  static struct ncclTransportComm dummyTransport = {
    .setup = mockSetup,
    .connect = mockConnect,
  };
  collNetTransport.recv = dummyTransport;

  // --- Dummy inputs ---
  ncclTopoGraph topoGraph = {};
  ncclConnect connect = {};
  int masterRank = 0;
  int masterPeer = 1;
  int channelId = 0;
  int type = collNetRecv;

  // --- Run the function ---
  bool failed = ncclTransportCollNetSetup(&comm, &topoGraph, &channel, masterRank, masterPeer, channelId, type, &connect);

  // --- Assert: function should succeed (return false) ---
  ASSERT_FALSE(failed);

  // --- Cleanup ---
  hipFree(devPeerArrayDevice);
  hipFree(devPeerPtrsDevice);
}

TEST(TransportTest, CollNetSendSetup) {
  constexpr int nranks = 2;
  constexpr int nNodes = 2;

  // --- Setup comm ---
  ncclComm comm = {};
  comm.rank = 0;
  comm.nRanks = nranks;
  comm.nNodes = nNodes;
  comm.node = 0;

  ncclPeerInfo peerInfo[3] = {};
  comm.peerInfo = peerInfo;

  // --- Setup channel ---
  ncclChannel channel = {};
  static ncclChannelPeer peerArray[3];
  static ncclChannelPeer* peerPtrs[3] = {
    &peerArray[0], &peerArray[1], &peerArray[2]
  };
  channel.peers = peerPtrs;

  // Step 1: Allocate device-side array of ncclDevChannelPeer
  ncclDevChannelPeer* devPeerArrayDevice;
  hipMalloc(&devPeerArrayDevice, sizeof(ncclDevChannelPeer) * 3);

  // Step 2: Create host-side array of device pointers
  ncclDevChannelPeer* devPeerPtrsHost[3] = {
    devPeerArrayDevice + 0,
    devPeerArrayDevice + 1,
    devPeerArrayDevice + 2
  };

  // Step 3: Allocate device-side array of device pointers
  ncclDevChannelPeer** devPeerPtrsDevice;
  hipMalloc(&devPeerPtrsDevice, sizeof(ncclDevChannelPeer*) * 3);

  // Step 4: Copy host-side array of device pointers to device
  hipMemcpy(devPeerPtrsDevice, devPeerPtrsHost, sizeof(ncclDevChannelPeer*) * 3, hipMemcpyHostToDevice);

  // Step 5: Set in channel
  channel.devPeers = devPeerPtrsDevice;

  // --- Setup transportComm ---
  static struct ncclTransportComm dummyTransport = {
    .setup = mockSetup,
    .connect = mockConnect,
  };
  collNetTransport.send = dummyTransport;

  // --- Dummy inputs ---
  ncclTopoGraph topoGraph = {};
  ncclConnect connect = {};  // IMPORTANT: non-null since this is memcpy’d into masterConnects
  int masterRank = 0;
  int masterPeer = 1;
  int channelId = 0;
  int type = collNetSend;

  // --- Run the function ---
  bool failed = ncclTransportCollNetSetup(&comm, &topoGraph, &channel, masterRank, masterPeer, channelId, type, &connect);

  // --- Assert: function should succeed (return false) ---
  ASSERT_FALSE(failed);

  // --- Cleanup ---
  hipFree(devPeerArrayDevice);
  hipFree(devPeerPtrsDevice);
}


  TEST(TransportTest, NcclTransportCollNetCheckTestSuccess) {
    struct ncclComm comm = {};
    struct ncclBootstrap dummyBootstrap;

    int rankMap[1] = {0};
    comm.localRank = 0;
    comm.localRanks = 1;
    comm.localRankToRank = rankMap;
    comm.bootstrap = &dummyBootstrap;
    int collNetSetupFail = 0;
    ncclResult_t result = ncclTransportCollNetCheck(&comm, collNetSetupFail);
    EXPECT_EQ(result, ncclSuccess);
  }

  TEST(TransportTest, NcclTransportCollNetCheckTestFails) {
    ncclComm comm;
    int rankMap[1] = {0};
    ncclBootstrap bootstrap;

    comm.localRank = 0;
    comm.localRanks = 1;
    comm.localRankToRank = rankMap;
    comm.bootstrap = &bootstrap;

    int collNetSetupFail = 1; // simulate failure on this rank
    ncclResult_t result = ncclTransportCollNetCheck(&comm, collNetSetupFail);
    EXPECT_EQ(result, ncclSystemError);
}

// Test for ncclTransportCollNetFree
TEST(TransportTest, CollNetFreeTest) {
  ncclComm comm = {};
  comm.nChannels = 1;
  comm.nRanks = 0; // So comm.channels[0].peers[0] is accessed

  // Access embedded array directly (don't assign to comm.channels)
  ncclChannel* channel = &comm.channels[0];

  // Allocate peer array for the channel (comm.nRanks + 1 for dummy peer)
  channel->peers = new ncclChannelPeer*[comm.nRanks + 1];
  for (int r = 0; r <= comm.nRanks; ++r) {
    channel->peers[r] = nullptr;
  }

  // Create dummy peer at index nRanks
  ncclChannelPeer* peer = new ncclChannelPeer();
  channel->peers[comm.nRanks] = peer;
  peer->refCount = 1;

  // Setup dummy ncclTransportComm with only `free` implemented
  static ncclTransportComm dummyTransportComm = {};
  dummyTransportComm.free = [](ncclConnector* conn) -> ncclResult_t {
    if (conn && conn->transportResources) {
      free(conn->transportResources);
      conn->transportResources = nullptr;
    }
    return ncclSuccess;
  };

  // Fill in dummy send and recv connectors
  for (int b = 0; b < NCCL_MAX_CONNS; ++b) {
    peer->send[b].transportResources = malloc(1);
    peer->send[b].transportComm = &dummyTransportComm;

    peer->recv[b].transportResources = malloc(1);
    peer->recv[b].transportComm = &dummyTransportComm;
  }

  // Call the function under test
  ncclResult_t result = ncclTransportCollNetFree(&comm);
  ASSERT_EQ(result, ncclSuccess);

  // Clean up
  delete peer;
  delete[] channel->peers;
  channel->peers = nullptr;
}

TEST(TransportTest, DumpDataTest) {
  ncclConnect conn;
  memset(&conn, 0xAB, sizeof(conn));  // Fill with a known byte pattern

  auto output = captureStdout([&]() {
    dumpData(&conn, 1);
  });

  // Basic checks on output
  EXPECT_TRUE(output.find("[0]") != std::string::npos);         // Label exists
  EXPECT_GT(output.length(), 10);                               // Should print something
  EXPECT_NE(output.find("ab"), std::string::npos);              // Should include 0xAB hex
}
}
