/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/* Planner-level PIPE graph coverage. These tests use an MPI/NCCL communicator
 * only to obtain real world ranks; they do not allocate transfer buffers or
 * launch a reshard kernel. */

#include <gtest/gtest.h>

#include <cstdlib>

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include "reshard_internal.h"
#include "staging_buffer.h"
#include "staging_types.h"
#include "test_helpers.h"

namespace {

int countRdmaSources(const StagingTransferDescriptor& desc) {
  int count = 0;
  for (int i = 0; i < desc.numSources; ++i) count += desc.sources[i].isRdma;
  return count;
}

int countRdmaTargets(const StagingTransferDescriptor& desc) {
  int count = 0;
  for (int i = 0; i < desc.numTargets; ++i) count += desc.targets[i].isRdma;
  return count;
}

int countRdmaSourcesFromPeerRange(const StagingTransferDescriptor& desc, int firstRank, int lastRank) {
  int count = 0;
  for (int i = 0; i < desc.numSources; ++i) {
    const StagingPeerDescriptor& source = desc.sources[i];
    count += source.isRdma && source.peerWorldRank >= firstRank && source.peerWorldRank < lastRank;
  }
  return count;
}

int countActiveEdges(const StagingPeerInfo (&edges)[STAGING_MAX_CHANNELS][MAX_SOURCES], int edgeCount,
                     int channels) {
  int count = 0;
  for (int ch = 0; ch < channels; ++ch) {
    for (int edge = 0; edge < edgeCount; ++edge) count += edges[ch][edge].active;
  }
  return count;
}

int countActiveEdges(const StagingPeerInfo (&edges)[STAGING_MAX_CHANNELS][MAX_TARGETS], int edgeCount,
                     int channels) {
  int count = 0;
  for (int ch = 0; ch < channels; ++ch) {
    for (int edge = 0; edge < edgeCount; ++edge) count += edges[ch][edge].active;
  }
  return count;
}

StagingBufferState makePlannerOnlyStagingState() {
  StagingBufferState state = {};
  state.buffer = reinterpret_cast<void*>(0x1000);
  state.numChannels = 4;
  state.capacityChannels = state.numChannels;
  state.controlSlotCount = 1;
  state.controlRegionSize = STAGING_CTRL_REGION_SIZE;
  state.chunkSize = 4096;
  state.channelDataSize = 4 * state.chunkSize;
  state.channelSize = state.controlRegionSize + state.channelDataSize;
  state.totalSize = (size_t)state.numChannels * state.channelSize;
  state.dataCapacity = (size_t)state.numChannels * state.channelDataSize;
  state.peersPerChannel = 1;
  state.initialized = true;
  return state;
}

TEST(PipeGraphPathPlannerTest, UsesActualRdmaTargetsForSourceLocalFifo) {
  const ReshardCopyAlgorithm savedCopyAlgorithm = gReshardCopyAlgorithm;
  gReshardCopyAlgorithm = RESHARD_COPY_ALGO_PIPE;

  StagingBufferState state = makePlannerOnlyStagingState();
  state.numChannels = 8;
  state.capacityChannels = state.numChannels;
  state.chunkSize = 32ULL * 1024ULL * 1024ULL;
  state.channelDataSize = 128ULL * 1024ULL * 1024ULL;
  state.channelSize = state.controlRegionSize + state.channelDataSize;
  state.totalSize = (size_t)state.numChannels * state.channelSize;
  state.dataCapacity = (size_t)state.numChannels * state.channelDataSize;

  StagingTransferDescriptor desc = {};
  desc.isSource = true;
  desc.myWorldRank = 0;
  desc.peerGroupSizeBound = 64;
  desc.numTargets = 1;
  desc.targets[0].isRdma = true;
  desc.targets[0].peerWorldRank = 1;
  desc.sourceIndexOnDest[0] = 0;
  desc.destNumSources[0] = 1;
  desc.destNumRdmaSources[0] = 1;
  desc.rdmaSourceIndexOnDest[0] = 0;
  desc.pipeGinPeerCapacity = 16;
  desc.pipeGinChannelsPerPeer = 8;

  StagingKernelParams params = {};
  EXPECT_EQ(ncclSuccess, stagingPrepareTransfer(&state, &desc, nullptr, nullptr, &params));
  EXPECT_TRUE(params.rdmaTargets[0][0].active);
  EXPECT_EQ(4, params.localRdmaFc[0][0].peerNumSlots);
  EXPECT_EQ(128ULL * 1024ULL * 1024ULL, params.localRdmaFc[0][0].peerChunkSize *
                                               (size_t)params.localRdmaFc[0][0].peerNumSlots);

  gReshardCopyAlgorithm = savedCopyAlgorithm;
}

TEST(PipeGraphPathPlannerTest, KeepsPackSourceFifoLayout) {
  const ReshardCopyAlgorithm savedCopyAlgorithm = gReshardCopyAlgorithm;
  gReshardCopyAlgorithm = RESHARD_COPY_ALGO_PACK;

  StagingBufferState state = makePlannerOnlyStagingState();
  state.numChannels = 1;
  state.capacityChannels = state.numChannels;
  state.chunkSize = 32ULL * 1024ULL * 1024ULL;
  state.channelDataSize = 128ULL * 1024ULL * 1024ULL;
  state.channelSize = state.controlRegionSize + state.channelDataSize;
  state.totalSize = state.channelSize;
  state.dataCapacity = state.channelDataSize;

  StagingTransferDescriptor desc = {};
  desc.isSource = true;
  desc.myWorldRank = 0;
  desc.numTargets = 2;
  desc.targets[0].isRdma = true;
  desc.targets[0].peerWorldRank = 1;
  desc.targets[1].isRdma = false;
  desc.targets[1].peerWorldRank = 2;
  desc.targets[1].peerLocalRank = 1;
  desc.sourceIndexOnDest[0] = 0;
  desc.sourceIndexOnDest[1] = 0;
  desc.destNumSources[0] = 1;
  desc.destNumSources[1] = 1;
  desc.destNumRdmaSources[0] = 1;
  desc.rdmaSourceIndexOnDest[0] = 0;

  StagingKernelParams params = {};
  EXPECT_EQ(ncclSuccess, stagingPrepareTransfer(&state, &desc, nullptr, nullptr, &params));
  EXPECT_EQ(1, params.localRdmaFc[0][0].peerNumSlots);
  EXPECT_EQ(params.localRdmaFc[0][0].peerNumSlots, params.localLsaFc[0][0].peerNumSlots);

  gReshardCopyAlgorithm = savedCopyAlgorithm;
}

class PipeGraphPathTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_EQ(MPI_SUCCESS, MPI_Comm_rank(MPI_COMM_WORLD, &rank_));
    ASSERT_EQ(MPI_SUCCESS, MPI_Comm_size(MPI_COMM_WORLD, &world_));
  }

  void initComm(ncclComm_t* comm) {
    ASSERT_NE(nullptr, comm);
    int deviceCount = 0;
    ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&deviceCount));
    ASSERT_GT(deviceCount, 0);
    ASSERT_EQ(cudaSuccess, cudaSetDevice(rank_ % deviceCount));

    ncclUniqueId id = {};
    if (rank_ == 0) ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(MPI_SUCCESS, MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    ASSERT_NCCL(ncclCommInitRankConfig(comm, world_, id, rank_, nullptr), *comm);
  }

  void preparePipeParams(ncclComm_t comm, ncclDistTensor_t* src, ncclDistTensor_t* dst,
                         int srcGpusPerDomain, int dstGpusPerDomain, StagingTransferDescriptor* desc,
                         StagingKernelParams* params) {
    ASSERT_EQ(ncclSuccess,
              buildStagingTransferDescriptor(comm, src->dataPtr, src->localShape, 2, src, dst->dataPtr,
                                             dst->localShape, dst, srcGpusPerDomain, dstGpusPerDomain, 0, desc));
    desc->controlSlot = 0;
    desc->pipeGinPeerCapacity = 4;
    desc->pipeGinChannelsPerPeer = 4;

    StagingBufferState state = makePlannerOnlyStagingState();
    ASSERT_EQ(ncclSuccess, stagingPrepareTransfer(&state, desc, nullptr, nullptr, params));
  }

  int rank_ = -1;
  int world_ = 0;
};

TEST_F(PipeGraphPathTest, Dsv3LikeFourSourceShardsTwoLocalDestinationReplicas) {
  if (world_ != 8) {
    GTEST_SKIP() << "requires 8 MPI ranks (got " << world_ << ")";
  }

  ncclComm_t comm = nullptr;
  initComm(&comm);
  ASSERT_NE(nullptr, comm);

  const ReshardCopyAlgorithm savedCopyAlgorithm = gReshardCopyAlgorithm;
  gReshardCopyAlgorithm = RESHARD_COPY_ALGO_PIPE;

  int srcMeshDims[2] = {4, 1};
  int dstMeshDims[2] = {2, 2};
  int srcPlacements[2] = {NCCL_RESHARD_SHARD(0), NCCL_RESHARD_REPLICATE};
  int dstPlacements[2] = {NCCL_RESHARD_SHARD(1), NCCL_RESHARD_REPLICATE};
  size_t srcLocalShape[2] = {1, 4};
  size_t dstLocalShape[2] = {4, 2};

  ncclMesh_t srcMesh = NCCL_M2N_MESH_INITIALIZER;
  srcMesh.ndims = 2;
  srcMesh.dims = srcMeshDims;
  srcMesh.startRank = 0;
  ncclMesh_t dstMesh = NCCL_M2N_MESH_INITIALIZER;
  dstMesh.ndims = 2;
  dstMesh.dims = dstMeshDims;
  dstMesh.startRank = 4;

  ncclDistTensor_t src = NCCL_M2N_DIST_TENSOR_INITIALIZER;
  src.dataPtr = rank_ < 4 ? reinterpret_cast<void*>(0x2000) : nullptr;
  src.localShape = srcLocalShape;
  src.ndims = 2;
  src.dtype = ncclUint8;
  src.mesh = &srcMesh;
  src.placements = srcPlacements;
  ncclDistTensor_t dst = NCCL_M2N_DIST_TENSOR_INITIALIZER;
  dst.dataPtr = rank_ >= 4 ? reinterpret_cast<void*>(0x3000) : nullptr;
  dst.localShape = dstLocalShape;
  dst.ndims = 2;
  dst.dtype = ncclUint8;
  dst.mesh = &dstMesh;
  dst.placements = dstPlacements;

  StagingTransferDescriptor desc = {};
  StagingKernelParams params = {};
  preparePipeParams(comm, &src, &dst, 1, 2, &desc, &params);

  if (rank_ < 4) {
    EXPECT_TRUE(desc.isSource);
    EXPECT_FALSE(desc.isDest);
    EXPECT_EQ(2, desc.numTargets);
    EXPECT_EQ(2, countRdmaTargets(desc));
    EXPECT_EQ(0, desc.numRingTargets);
    EXPECT_EQ(2, params.numRdmaTargets);
    EXPECT_EQ(0, params.numLsaTargets);
    EXPECT_EQ(2, countActiveEdges(params.rdmaTargets, params.numRdmaTargets, params.numChannels));
  } else {
    EXPECT_FALSE(desc.isSource);
    EXPECT_TRUE(desc.isDest);
    EXPECT_EQ(4, desc.numSources);
    EXPECT_EQ(2, countRdmaSources(desc));
    EXPECT_EQ(2, desc.numSources - countRdmaSources(desc));
    EXPECT_EQ(1, desc.numLsaFollowers);
    EXPECT_EQ(2, desc.numTargets);
    EXPECT_EQ(0, countRdmaTargets(desc));

    EXPECT_EQ(2, params.numRdmaSources);
    EXPECT_EQ(2, params.numLsaSources);
    EXPECT_EQ(2, params.numLsaTargets);
    EXPECT_EQ(1, params.numLsaFollowers);
    EXPECT_EQ(params.rdmaRegions[0].dataOffset, params.lsaRegions[0].dataOffset);
    EXPECT_EQ(params.rdmaRegions[0].regionSize, params.lsaRegions[0].regionSize);
    for (int ch = 0; ch < params.numChannels; ++ch) {
      for (int source = 0; source < params.numLsaSources; ++source) {
        const StagingPeerInfo& peer = params.lsaSources[ch][source];
        if (peer.active) {
          EXPECT_GE(peer.fc.peerDataOffset, params.rdmaRegions[ch].dataOffset);
          EXPECT_LT(peer.fc.peerDataOffset, params.rdmaRegions[ch].dataOffset + params.rdmaRegions[ch].regionSize);
        }
      }
      for (int target = 0; target < params.numLsaTargets; ++target) {
        const StagingPeerInfo& peer = params.lsaTargets[ch][target];
        if (peer.active) {
          EXPECT_GE(peer.fc.peerDataOffset, params.rdmaRegions[ch].dataOffset);
          EXPECT_LT(peer.fc.peerDataOffset, params.rdmaRegions[ch].dataOffset + params.rdmaRegions[ch].regionSize);
        }
      }
    }
    EXPECT_EQ(0, params.numRingTargets);
    EXPECT_GT(params.ginSignalCount, 0);
    EXPECT_EQ(2, countActiveEdges(params.rdmaSources, params.numRdmaSources, params.numChannels));
    EXPECT_EQ(2, countActiveEdges(params.lsaSources, params.numLsaSources, params.numChannels));
    EXPECT_EQ(2, countActiveEdges(params.lsaTargets, params.numLsaTargets, params.numChannels));
  }

  gReshardCopyAlgorithm = savedCopyAlgorithm;
  ASSERT_EQ(MPI_SUCCESS, MPI_Barrier(MPI_COMM_WORLD));
  ASSERT_NCCL(ncclCommDestroy(comm));
}

TEST_F(PipeGraphPathTest, FourRanksCoverPointToPointTrainerFaninAndTrainerFanout) {
  if (world_ != 4) {
    GTEST_SKIP() << "requires 4 MPI ranks (got " << world_ << ")";
  }

  ncclComm_t comm = nullptr;
  initComm(&comm);
  ASSERT_NE(nullptr, comm);

  const ReshardCopyAlgorithm savedCopyAlgorithm = gReshardCopyAlgorithm;
  gReshardCopyAlgorithm = RESHARD_COPY_ALGO_PIPE;

  int srcMeshDims[2] = {2, 1};
  int srcPlacements[2] = {NCCL_RESHARD_SHARD(0), NCCL_RESHARD_REPLICATE};
  size_t srcLocalShape[2] = {1, 2};
  ncclMesh_t srcMesh = NCCL_M2N_MESH_INITIALIZER;
  srcMesh.ndims = 2;
  srcMesh.dims = srcMeshDims;
  srcMesh.startRank = 0;
  ncclDistTensor_t src = NCCL_M2N_DIST_TENSOR_INITIALIZER;
  src.dataPtr = rank_ < 2 ? reinterpret_cast<void*>(0x6000) : nullptr;
  src.localShape = srcLocalShape;
  src.ndims = 2;
  src.dtype = ncclUint8;
  src.mesh = &srcMesh;
  src.placements = srcPlacements;

  {
    int dstMeshDims[2] = {2, 1};
    int dstPlacements[2] = {NCCL_RESHARD_SHARD(0), NCCL_RESHARD_REPLICATE};
    size_t dstLocalShape[2] = {1, 2};
    ncclMesh_t dstMesh = NCCL_M2N_MESH_INITIALIZER;
    dstMesh.ndims = 2;
    dstMesh.dims = dstMeshDims;
    dstMesh.startRank = 2;
    ncclDistTensor_t dst = NCCL_M2N_DIST_TENSOR_INITIALIZER;
    dst.dataPtr = rank_ >= 2 ? reinterpret_cast<void*>(0x7000) : nullptr;
    dst.localShape = dstLocalShape;
    dst.ndims = 2;
    dst.dtype = ncclUint8;
    dst.mesh = &dstMesh;
    dst.placements = dstPlacements;

    StagingTransferDescriptor desc = {};
    StagingKernelParams params = {};
    preparePipeParams(comm, &src, &dst, 1, 1, &desc, &params);
    EXPECT_EQ(1, desc.isSource ? countRdmaTargets(desc) : countRdmaSources(desc));
    EXPECT_EQ(0, desc.numLsaFollowers);
    EXPECT_EQ(0, params.numLsaTargets + params.numLsaSources);
    EXPECT_EQ(0, params.numRingTargets);
  }

  {
    int dstMeshDims[2] = {1, 1};
    int dstPlacements[2] = {NCCL_RESHARD_REPLICATE, NCCL_RESHARD_REPLICATE};
    size_t dstLocalShape[2] = {2, 2};
    ncclMesh_t dstMesh = NCCL_M2N_MESH_INITIALIZER;
    dstMesh.ndims = 2;
    dstMesh.dims = dstMeshDims;
    dstMesh.startRank = 2;
    ncclDistTensor_t dst = NCCL_M2N_DIST_TENSOR_INITIALIZER;
    dst.dataPtr = rank_ == 2 ? reinterpret_cast<void*>(0x8000) : nullptr;
    dst.localShape = dstLocalShape;
    dst.ndims = 2;
    dst.dtype = ncclUint8;
    dst.mesh = &dstMesh;
    dst.placements = dstPlacements;

    StagingTransferDescriptor desc = {};
    StagingKernelParams params = {};
    preparePipeParams(comm, &src, &dst, 1, 1, &desc, &params);
    if (rank_ == 2) {
      EXPECT_EQ(2, countRdmaSources(desc));
      EXPECT_EQ(2, params.numRdmaSources);
    }
    EXPECT_EQ(0, desc.numLsaFollowers);
    EXPECT_EQ(0, params.numLsaTargets + params.numLsaSources);
  }

  {
    int dstMeshDims[2] = {1, 2};
    int dstPlacements[2] = {NCCL_RESHARD_SHARD(1), NCCL_RESHARD_REPLICATE};
    size_t dstLocalShape[2] = {2, 2};
    ncclMesh_t dstMesh = NCCL_M2N_MESH_INITIALIZER;
    dstMesh.ndims = 2;
    dstMesh.dims = dstMeshDims;
    dstMesh.startRank = 2;
    ncclDistTensor_t dst = NCCL_M2N_DIST_TENSOR_INITIALIZER;
    dst.dataPtr = rank_ >= 2 ? reinterpret_cast<void*>(0x9000) : nullptr;
    dst.localShape = dstLocalShape;
    dst.ndims = 2;
    dst.dtype = ncclUint8;
    dst.mesh = &dstMesh;
    dst.placements = dstPlacements;

    StagingTransferDescriptor desc = {};
    StagingKernelParams params = {};
    preparePipeParams(comm, &src, &dst, 1, 2, &desc, &params);
    if (rank_ >= 2) {
      EXPECT_EQ(1, countRdmaSources(desc));
      EXPECT_EQ(1, desc.numSources - countRdmaSources(desc));
      EXPECT_EQ(1, desc.numLsaFollowers);
      EXPECT_EQ(1, params.numRdmaSources);
      EXPECT_EQ(1, params.numLsaSources);
      EXPECT_EQ(1, params.numLsaTargets);
    }
  }

  gReshardCopyAlgorithm = savedCopyAlgorithm;
  ASSERT_EQ(MPI_SUCCESS, MPI_Barrier(MPI_COMM_WORLD));
  ASSERT_NCCL(ncclCommDestroy(comm));
}

TEST_F(PipeGraphPathTest, ThreeDestinationDomainsCoverGeneratorRingIngressForwardingAndFanout) {
  if (world_ != 8) {
    GTEST_SKIP() << "requires 8 MPI ranks (got " << world_ << ")";
  }

  ncclComm_t comm = nullptr;
  initComm(&comm);
  ASSERT_NE(nullptr, comm);

  const ReshardCopyAlgorithm savedCopyAlgorithm = gReshardCopyAlgorithm;
  gReshardCopyAlgorithm = RESHARD_COPY_ALGO_PIPE;

  int srcMeshDims[2] = {2, 1};
  int dstMeshDims[2] = {1, 6};
  int srcPlacements[2] = {NCCL_RESHARD_SHARD(0), NCCL_RESHARD_REPLICATE};
  int dstPlacements[2] = {NCCL_RESHARD_SHARD(1), NCCL_RESHARD_REPLICATE};
  size_t srcLocalShape[2] = {1, 3};
  size_t dstLocalShape[2] = {2, 3};

  ncclMesh_t srcMesh = NCCL_M2N_MESH_INITIALIZER;
  srcMesh.ndims = 2;
  srcMesh.dims = srcMeshDims;
  srcMesh.startRank = 0;
  ncclMesh_t dstMesh = NCCL_M2N_MESH_INITIALIZER;
  dstMesh.ndims = 2;
  dstMesh.dims = dstMeshDims;
  dstMesh.startRank = 2;

  ncclDistTensor_t src = NCCL_M2N_DIST_TENSOR_INITIALIZER;
  src.dataPtr = rank_ < 2 ? reinterpret_cast<void*>(0x4000) : nullptr;
  src.localShape = srcLocalShape;
  src.ndims = 2;
  src.dtype = ncclUint8;
  src.mesh = &srcMesh;
  src.placements = srcPlacements;
  ncclDistTensor_t dst = NCCL_M2N_DIST_TENSOR_INITIALIZER;
  dst.dataPtr = rank_ >= 2 ? reinterpret_cast<void*>(0x5000) : nullptr;
  dst.localShape = dstLocalShape;
  dst.ndims = 2;
  dst.dtype = ncclUint8;
  dst.mesh = &dstMesh;
  dst.placements = dstPlacements;

  StagingTransferDescriptor desc = {};
  StagingKernelParams params = {};
  preparePipeParams(comm, &src, &dst, 1, 2, &desc, &params);

  if (rank_ < 2) {
    EXPECT_TRUE(desc.isSource);
    EXPECT_EQ(1, desc.numTargets);
    EXPECT_EQ(1, countRdmaTargets(desc));
    EXPECT_EQ(0, desc.numRingTargets);
  } else {
    EXPECT_TRUE(desc.isDest);
    EXPECT_EQ(2, desc.numSources);
    EXPECT_EQ(1, countRdmaSources(desc));
    EXPECT_EQ(1, desc.numSources - countRdmaSources(desc));
    EXPECT_EQ(1, desc.numLsaFollowers);
    EXPECT_EQ(1, params.numRdmaSources);
    EXPECT_EQ(1, params.numLsaSources);
    EXPECT_EQ(1, params.numLsaTargets);

    const bool hasRingSuccessor = rank_ < 6;
    EXPECT_EQ(hasRingSuccessor ? 1 : 0, desc.numRingTargets);
    EXPECT_EQ(hasRingSuccessor ? 1 : 0, params.numRingTargets);

    const bool receivesFromGenerator = rank_ >= 4;
    EXPECT_EQ(receivesFromGenerator ? 1 : 0, countRdmaSourcesFromPeerRange(desc, 2, 8));
    EXPECT_EQ(receivesFromGenerator ? 0 : 1, countRdmaSourcesFromPeerRange(desc, 0, 2));
  }

  gReshardCopyAlgorithm = savedCopyAlgorithm;
  ASSERT_EQ(MPI_SUCCESS, MPI_Barrier(MPI_COMM_WORLD));
  ASSERT_NCCL(ncclCommDestroy(comm));
}

} // namespace
