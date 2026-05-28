/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <rccl/rccl.h>

#include "TestBed.hpp"

#include "CollRegUtils.hpp"

namespace RcclUnitTesting
{
  
TEST(CollReg, RegisterCheckP2P_ConnectedWithP2PFlag) {
  ncclComm comm = {};
  ncclConnector conn = {};
  conn.connected = 1;
  conn.conn.flags = NCCL_P2P_WRITE;

  bool needReg = false;
  EXPECT_EQ(registerCheckP2PConnection(&comm, &conn, nullptr, 1, &needReg), ncclSuccess);
  EXPECT_TRUE(needReg);
}

TEST(CollReg, RegisterCheckP2P_ConnectedWithNoP2PFlag) {
  ncclComm comm = {};
  ncclConnector conn = {};
  conn.connected = 1;
  conn.conn.flags = 0;

  bool needReg = true;
  EXPECT_EQ(registerCheckP2PConnection(&comm, &conn, nullptr, 1, &needReg), ncclSuccess);
  EXPECT_FALSE(needReg);
}

TEST(CollReg, RegisterCheckP2P_NotConnected_CanConnectTrue) {
  // Save original canConnect
  ScopedCanConnectOverride _(&(*ncclTransports[0]), mockCanConnect);
  auto* originalCanConnect = ncclTransports[0]->canConnect;
  ncclTransports[0]->canConnect = mockCanConnect;

  ncclComm comm = {};
  comm.rank = 0;

  ncclPeerInfo peers[2] = {};
  comm.peerInfo = peers;

  // Set myInfo and peerInfo
  peers[0].hostHash = 123;
  peers[0].pidHash = 999;
  peers[0].cudaDev = 0;
  peers[0].busId = 0x1;
  peers[0].comm = &comm;

  peers[1].hostHash = 123;
  peers[1].pidHash = 888;
  peers[1].cudaDev = 1;
  peers[1].busId = 0x2;
  peers[1].comm = &comm;

  ncclConnector conn = {};
  conn.connected = 0;

  ncclTopoGraph graph = {};

  bool needReg = false;

  EXPECT_EQ(registerCheckP2PConnection(&comm, &conn, &graph, 1, &needReg), ncclSuccess);
  EXPECT_TRUE(needReg);  // Because mockCanConnect returns true

  // Restore original canConnect
  ncclTransports[0]->canConnect = originalCanConnect;
}

TEST(CollReg, RegisterCheckP2P_NotConnected_CanConnectFalse) {
  ScopedCanConnectOverride _(&(*ncclTransports[0]), mockCanConnectFalse); 

  ncclComm comm = {};
  comm.rank = 0;
  ncclPeerInfo peers[2] = {};
  comm.peerInfo = peers;

  // Set up myInfo and peerInfo to block P2P
  peers[0].hostHash = 123; peers[0].pidHash = 999;
  //peers[0].dev = 0;
  peers[1].hostHash = 456; peers[1].pidHash = 888;  // different host
  //peers[1].dev = 1;

  ncclConnector conn = {};
  conn.connected = 0;

  ncclTopoGraph graph = {};

  bool needReg = true;
  EXPECT_EQ(registerCheckP2PConnection(&comm, &conn, &graph, 1, &needReg), ncclSuccess);
  EXPECT_FALSE(needReg);  // should be false for different host
}

}
