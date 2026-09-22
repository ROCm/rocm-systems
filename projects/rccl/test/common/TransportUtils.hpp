/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef TRANSPORT_UTILS_H
#define TRANSPORT_UTILS_H

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <transport.h>  
#include "TestBed.hpp"

void dumpData(struct ncclConnect* data, int ndata);
ncclResult_t bootstrapAllGather(void* bootstrap, void* data, int size) {
  memcpy((char*)data + size, data, size); // Simulate copying rank 0 connect to rank 1
  return ncclSuccess;
}

namespace RcclUnitTesting
{
//Mock functions for CollNetRecvSetup and CollNetSendSetup
ncclResult_t mockSetup(struct ncclComm* comm, struct ncclTopoGraph* graph,
                       struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo,
                       struct ncclConnect* connect, struct ncclConnector* connector,
                       int channelId, int type) {
  memset(connect, 42, sizeof(struct ncclConnect)); // dummy data
  return ncclSuccess;
}

ncclResult_t mockConnect(struct ncclComm* comm, struct ncclConnect* connect,
                         int nranks, int rank, struct ncclConnector* connector) {
  memset(&connector->conn, 99, sizeof(connector->conn)); // dummy
  return ncclSuccess;
}

// Dummy bootstrap implementation for testing NcclTransportCollNetCheckTestSuccess and NcclTransportCollNetCheckTestFails
struct ncclBootstrap {};

ncclResult_t bootstrapIntraNodeAllGather(
struct ncclBootstrap* bootstrap,
int* localRankToRank,
int localRank,
int localRanks,
int* data,
size_t size
) {
  data[0] = 0; // Rank 0 is fine
  data[1] = 1; // Rank 1 reports failure
  return ncclSuccess;
}

//Helper function for capturing the output for DumpDataTest
std::string captureStdout(std::function<void()> func) {
  int pipefd[2];
  pipe(pipefd);

  int saved_stdout = dup(STDOUT_FILENO);
  dup2(pipefd[1], STDOUT_FILENO);
  close(pipefd[1]);

  func();

  fflush(stdout);
  dup2(saved_stdout, STDOUT_FILENO);
  close(saved_stdout);

  char buffer[4096];
  ssize_t count = read(pipefd[0], buffer, sizeof(buffer) - 1);
  close(pipefd[0]);
  buffer[count] = '\0';

  return std::string(buffer);
}

} //namespace RcclUnitTesting
#endif
