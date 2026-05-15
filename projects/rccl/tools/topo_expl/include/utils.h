/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef UTILS_H_
#define UTILS_H_

struct graphInfo {
  int pattern;
  int nChannels;
  int sameChannels;
  float bwIntra;
  float bwInter;
  int typeIntra;
  int typeInter;
};

struct allGatherInfo {
  struct graphInfo graphInfo[NCCL_NUM_ALGORITHMS];
  struct ncclTopoRanks topoRanks;
  int nc;
  bool pivotA2AEnabled;
  bool ll128Enabled;
  bool mscclEnabled;
};

void initCollNet();

ncclResult_t ncclTopoGetSystem(const char* xmlTopoFile, struct ncclTopoSystem** system);

ncclResult_t ncclTopoGetSystemFromXml(struct ncclXml* xml, struct ncclTopoSystem** topoSystem);

ncclResult_t fillInfo(struct ncclComm* comm, struct ncclPeerInfo* info, uint64_t commHash);

ncclResult_t initTransportsRank_1(struct ncclComm* comm, struct allGatherInfo *allGather3Data,
  struct ncclTopoGraph& treeGraph, struct ncclTopoGraph& ringGraph, struct ncclTopoGraph& collNetGraph, struct ncclTopoGraph& nvlsGraph, struct ncclComm* parent = NULL);

ncclResult_t initTransportsRank_3(struct ncclComm* comm, struct allGatherInfo *allGather3Data,
  struct ncclTopoGraph& treeGraph, struct ncclTopoGraph& ringGraph, struct ncclTopoGraph& collNetGraph, struct ncclTopoGraph& nvlsGraph);

#define TIME_START(index)

#define TIME_STOP(index)

#define TIME_CANCEL(index)

#define TIME_PRINT(name)

#endif