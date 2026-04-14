/*
Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "nccl.h"
#include "channel.h"
#include "nvmlwrap.h"
#include "bootstrap.h"
#include "transport.h"
#include "group.h"
#include "net.h"
#include "graph.h"
#include "argcheck.h"
#include <sched.h>
#include <fcntl.h>
#include <unistd.h>
#include <hip/hip_runtime.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <dlfcn.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <iostream>
#include <cstring>
#include "model.h"
#include "utils.h"
#include "topo.h"
#include "graph.h"
#include "rccl_common.h"

NodeModel *node_model;
extern ncclNet_t* ncclNet;

int64_t ncclParamWorkArgsBytes() { return INT64_MAX; }

char* getCmdOption(char ** begin, char ** end, const std::string & option) {
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option) {
    return std::find(begin, end, option) != end;
}

typedef struct NodeModelDesc {
    const char *filename;
    const char *description;
} NodeModelDesc;

NodeModelDesc model_descs[] = {
  // GFX 906
  {"topo_4p1h.xml",                      " 4gfx906 1H2XGMI  1NIC 1Intel A"},
  {"topo_4p1h_1.xml",                    " 4gfx906 1H2XGMI  2NIC 2Intel A"},
  {"topo_8p_rome.xml",                   " 8gfx906 2H2XGMI  1NIC 2AMD   A"},
  {"topo_8p_rome_n2.xml",                " 8gfx906 2H2XGMI  1NIC 4AMD   A"},
  {"topo_8p_rome_n4.xml",                " 8gfx906 2H2XGMI  1NIC 7AMD   A"},
  {"topo_4p2h.xml",                      " 8gfx906 2H2XGMI  1NIC 1Intel A"},
  {"topo_4p2h_1.xml",                    " 8gfx906 2H2XGMI  1NIC 1Intel B"},
  {"topo_4p2h_2nic.xml",                 " 8gfx906 2H2XGMI  2NIC 1Intel A"},
  {"topo_8p_rome_n2_1.xml",              " 8gfx906 2H2XGMI  2NIC 4AMD   A"},
  {"topo_8p_rome_n2_2.xml",              " 8gfx906 2H2XGMI  2NIC 4AMD   B"},
  {"topo_8p_ts1.xml",                    " 8gfx906 2H2XGMI  2NIC 4AMD   C"},
  {"topo_8p_ts1_1.xml",                  " 8gfx906 2H2XGMI  2NIC 4AMD   D"},
  {"topo_8p_ts1_n4.xml",                 " 8gfx906 2H2XGMI  2NIC 8AMD   A"},
  {"topo_8p_ts1_n4_1.xml",               " 8gfx906 2H2XGMI  2NIC 8AMD   B"},
  {"topo_8p_ts1_n4_2.xml",               " 8gfx906 2H2XGMI  3NIC 8AMD   C"},
  {"topo_8p_pcie.xml",                   " 8gfx906 PCIe     1NIC 1Intel A"},
  {"topo_8p_pcie_1.xml",                 " 8gfx906 PCIe     1NIC 1Intel B"},
  {"topo_8p_pcie_2nic.xml",              " 8gfx906 PCIe     2NIC 1Intel A"},
  {"topo_8p_rome_pcie.xml",              " 8gfx906 PCIe     2NIC 2AMD2  A"},
  // GFX 908
  {"topo_4p3l.xml",                      " 4gfx908 1H3XGMI  2NIC 1Intel A"},
  {"topo_8p6l.xml",                      " 8gfx908 1H6XGMI  1NIC 2AMD   A"},
  {"topo_8p6l_1nic.xml",                 " 8gfx908 1H6XGMI  1NIC 2AMD   B"},
  {"topo_8p6l_2nic.xml",                 " 8gfx908 1H6XGMI  2NIC 2AMD   A"},
  {"topo_8p6l_3nic.xml",                 " 8gfx908 1H6XGMI  3NIC 2AMD   A"},
  {"topo_8p6l_4nic.xml",                 " 8gfx908 1H6XGMI  4NIC 2AMD   A"},
  {"topo_8p6l_5nic.xml",                 " 8gfx908 1H6XGMI  5NIC 2AMD   A"},
  {"topo_8p6l_6nic.xml",                 " 8gfx908 1H6XGMI  6NIC 2AMD   A"},
  {"topo_4p3l_ia.xml",                   " 8gfx908 2H3XGMI  1NIC 1Intel A"},
  {"topo_4p3l_2h.xml",                   " 8gfx908 2H3XGMI  1NIC 4AMD   A"},
  {"topo_4p3l_n2.xml",                   " 8gfx908 2H3XGMI  1NIC 4AMD   B"},
  {"topo_4p3l_n2_1.xml",                 " 8gfx908 2H3XGMI  1NIC 4AMD   C"},
  {"topo_collnet_n1.xml",                " 8gfx908 2H3XGMI  1NIC 4AMD   D"},
  {"topo_8p_rome_vm1.xml",               " 8gfx908 2H3XGMI  1NIC 4AMD   E"},
  {"topo_4p3l_n4.xml",                   " 8gfx908 2H3XGMI  1NIC 7AMD   A"},
  {"topo_8p_rome_n4_1.xml",              " 8gfx908 2H3XGMI  1NIC 7AMD   B"},
  {"topo_8p_rome_4nics.xml",             " 8gfx908 2H3XGMI  4NIC 4AMD   A"},
  {"topo_collnet_n4.xml",                " 8gfx908 2H3XGMI  4NIC 4AMD   B"},
  {"topo_8p_rome_4n_1.xml",              " 8gfx908 2H3XGMI  4NIC 4AMD   C"},
  {"topo_8p_rome_4n_2.xml",              " 8gfx908 2H3XGMI  4NIC 4AMD   D"},
  {"topo_8p_4nics.xml",                  " 8gfx908 2H3XGMI  4NIC 4AMD   E"},
  {"topo_4p4h.xml",                      "16gfx908 2H3XGMI 16NIC 1AMD   A"},
  // GFX 910
  {"topo_3p_pcie.xml",                   " 3gfx910 PCIe     1NIC 2AMD   A"},
  {"topo_3p_pcie_1.xml",                 " 3gfx910 PCIe     1NIC 2AMD   B"},
  {"topo_8p_90a.xml",                    " 8gfx910 2H3XGMI  1NIC 1AMD   A"},
  {"topo_8p_90a_1.xml",                  " 8gfx910 2H3XGMI  1NIC 3AMD   A"},
  {"topo_8p1h_2.xml",                    " 8gfx910 2H3XGMI  2NIC 4AMD   A"},
  {"topo_8p1h.xml",                      " 8gfx910 2H3XGMI  4NIC 2AMD   A"},
  {"topo_8p1h_n1.xml",                   " 8gfx910 2H3XGMI  4NIC 2AMD   B"},
  {"topo_8p1h_1.xml",                    " 8gfx910 2H3XGMI  4NIC 2AMD   C"},
  {"topo_8p1h_3.xml",                    " 8gfx910 2H3XGMI  4NIC 4AMD   A"},
  {"topo_8p1h_4.xml",                    " 8gfx910 2H3XGMI  8NIC 2AMD   A"},
  {"topo_8p1h_5.xml",                    " 8gfx910 2H3XGMI  8NIC 2AMD   B"},
  {"topo_16p1h.xml",                     "16gfx910 2H3XGMI  8NIC 4AMD   A"},
  {"topo_16p1h_vm.xml",                  "16gfx910 2H3XGMI  8NIC 4AMD   B"},
  // GFX 942
  {"topo_4p_942.xml",                    " 4gfx942 1H3XGMI  4NIC 4AMD2  A"},
  {"topo_8p_942.xml",                    " 8gfx942 1H7XGMI  8NIC 2Intel A"},
  {"topo_8p_942vm.xml",                  " 8gfx942 1H7XGMI  8NIC 2Intel B"},
  {"topo_16p_gio-1s-1rp-cascade.xml",    "16gfx942 2H7XGMI  1NIC 2AMD   A"},
  {"topo_16p_gio-3s-1rp-split-flat.xml", "16gfx942 2H7XGMI  1NIC 2AMD   B"},
  // GFX 950
  {"topo_8p_950.xml",                    " 8gfx950 1H7XGMI  8NIC 2AMD   A"},
};

NCCL_PARAM(MaxCTAs, "MAX_CTAS", MAXCHANNELS);
NCCL_PARAM(MinCTAs, "MIN_CTAS", 1);

int main(int argc,char* argv[])
{
  struct ncclComm *comm;
  const int num_models = sizeof(model_descs) / sizeof(*model_descs);
  int minCTAsEnv;
  int maxCTAsEnv;

  if (!cmdOptionExists(argv, argv + argc, "-m")) {
    printf("Usage: ./topo_expl -m model_id [-n numNodes=1]\n");
    printf("List of model_id:\n");
    for (int i = 0; i < num_models; i++)
      printf("  %2d: %24s [%s]\n", i, model_descs[i].description, model_descs[i].filename);
    exit(0);
  }

  int model_id = 0;
  char *mi = getCmdOption(argv, argv + argc, "-m");
  if (mi)
    model_id = atol(mi);

  if (model_id >= num_models) {
      printf("Invalid model_id %d\n", model_id);
      exit(0);
  }

  NetworkModel network;
  NodeModel* node;

  initCollNet();

  NodeModelDesc *desc = &model_descs[model_id];
  int numNodes = 1;
  if (cmdOptionExists(argv, argv + argc, "-n")) {
    char *numNodesStr = getCmdOption(argv, argv + argc, "-n");
    if (numNodesStr)
      numNodes = atol(numNodesStr);
  }
  for (int i=0; i < numNodes; i++) {
      node = new NodeModel(desc->filename);
      network.AddNode(node);
  }

  printf("Generating topology using %d: %s\n", model_id, desc->description);

  int nranks = network.GetNRanks();
  int nnodes = network.GetNNodes();

  printf("nnodes = %d, nranks = %d\n", nnodes, nranks);
  for (int i = 0; i < nranks; i++) {
    node_model = network.GetNode(i);
    assert(node_model!=0);
    printf("Rank %d: node %d cudaDev %d GPU busId %lx\n", i, node_model->nodeId,
      node_model->rankToCudaDev(i), node_model->getGpuBusId(i));
  }

  minCTAsEnv = ncclParamMinCTAs();
  maxCTAsEnv = ncclParamMaxCTAs();

  NCCLCHECK(ncclCalloc(&comm, nranks));

  struct ncclPeerInfo *peerInfo;
  NCCLCHECK(ncclCalloc(&peerInfo, nranks+1)); // Extra rank to represent CollNet root

  struct allGatherInfo* allGather3Data;
  NCCLCHECK(ncclCalloc(&allGather3Data, nranks));

  struct ncclTopoGraph *treeGraph, *ringGraph, *collNetGraph, *nvlsGraph;
  NCCLCHECK(ncclCalloc(&treeGraph, nranks));
  NCCLCHECK(ncclCalloc(&ringGraph, nranks));
  NCCLCHECK(ncclCalloc(&collNetGraph, nranks));
  NCCLCHECK(ncclCalloc(&nvlsGraph, nranks));

  for (int i = 0; i < nranks; i++) {
    comm[i].rank = i;
    comm[i].nRanks = nranks;
    NCCLCHECK(ncclCalloc(&comm[i].connectSend, NCCL_MAX_CONNS*comm->nRanks));
    NCCLCHECK(ncclCalloc(&comm[i].connectRecv, NCCL_MAX_CONNS*comm->nRanks));
    node_model = network.GetNode(i);
    assert(node_model!=0);
    comm[i].busId = node_model->getGpuBusId(i);
    comm[i].topo = node_model->getSystem(i);
    comm[i].peerInfo = peerInfo;
    comm[i].ncclNet = ncclNet;
    comm[i].config.maxCTAs = maxCTAsEnv;
    comm[i].config.minCTAs = minCTAsEnv;
    if (comm[i].topParentRanks == NULL) {
      NCCLCHECK(ncclCalloc(&comm[i].topParentRanks, comm->nRanks));
      for (int j = 0; j < comm->nRanks; ++j)
        comm[i].topParentRanks[j] = j;
    }
    struct ncclSharedResources* sharedRes = NULL;
    NCCLCHECK(ncclCalloc(&sharedRes, 1));
    /* most of attributes are assigned later in initTransportsRank(). */
    sharedRes->owner = &comm[i];
    sharedRes->tpNRanks = comm[i].nRanks;
    NCCLCHECK(ncclCalloc(&sharedRes->tpRankToLocalRank, comm[i].nRanks));
    comm[i].sharedRes = sharedRes;
    sharedRes->refCount = 1;
    ncclMemoryStackConstruct(&comm[i].memPermanent);
   // Mark channels as non initialized.
    for (int c=0; c<MAXCHANNELS; c++) comm[i].channels[c].id = -1;
    NCCLCHECK(fillInfo(&comm[i], comm[i].peerInfo+comm[i].rank, 0));
  }

  for (int i = 0; i < nranks; i++) {
    node_model = network.GetNode(i);
    assert(node_model!=0);
    initTransportsRank_1(&comm[i], allGather3Data, treeGraph[i], ringGraph[i], collNetGraph[i], nvlsGraph[i]);
  }

  for (int i = 0; i < nranks; i++) {
    node_model = network.GetNode(i);
    assert(node_model!=0);
    initTransportsRank_3(&comm[i], allGather3Data, treeGraph[i], ringGraph[i], collNetGraph[i], nvlsGraph[i]);
    CUDACHECK(hipDeviceGetAttribute(&comm[i].WarpSize, hipDeviceAttributeWarpSize, comm[i].cudaDev));
  }
  for (uint64_t len = 8; len <= 4294967296L; len *= 2) {
    struct ncclInfo info;
    float minTime = 3600000000.0;
    info.comm = &comm[0];

    info.coll = ncclFuncAllReduce;
    // Find algorithm / protocol.
    int algorithm = -1;
    int protocol = -1;
    int nAlgos = NCCL_NUM_ALGORITHMS;
    for (int a=0; a<nAlgos; a++) {
      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        float time;
        NCCLCHECK(ncclTopoGetAlgoTime(info.comm, info.coll, a, p, len, 1, &time));
        if (time >= 0 && time < minTime) {
          algorithm = a;
          protocol = p;
          minTime = time;
        }
      }
    }
    if (algorithm == -1 || protocol == -1) {
      WARN("Error : no algorithm/protocol available");
      return ncclInternalError;
    }
    INFO(NCCL_TUNING, "%10ld %s %s time %f", len, ncclAlgoStr[algorithm], ncclProtoStr[protocol], minTime);
  }

  // Arrays to store function types for ncclFuncAllReduce, ReduceScatter, and AllGather
  std::vector<ncclFunc_t> ncclFuncTypes = {
    ncclFuncAllReduce,
    ncclFuncReduceScatter,
    ncclFuncAllGather,
    ncclFuncReduce,
    ncclFuncBroadcast
  };

  std::cout << "Running fp32 production choices for algorithm/protocol/maxChannels" << std::endl;
  // RCCL tuning results
  printf("| %-15s | %-15s | %-15s | %-10s | %-10s | %-12s |\n", "Max Size(B)", "Count", "Collective", "Algorithm", "Protocol", "Max Channels");
  printf("|-----------------|-----------------|-----------------|------------|------------|--------------|\n");
  for(int i = 0; i < ncclFuncTypes.size(); ++i) {
    for (uint64_t count = 8; count <= 1073741824L; count *= 2) { // Up to 1 gigabyte
      int algo, proto, nChannels;
      NCCLCHECK(rcclGetAlgoInfo(&comm[0], ncclFuncTypes[i], count, ncclFloat32 , 0, 0, 1, &algo, &proto, &nChannels));
      uint64_t maxCount;
      NCCLCHECK(rcclFuncMaxSendRecvCount(ncclFuncTypes[i], comm[0].nRanks, count, maxCount));
      printf("| %-15ld | %-15ld | %-15s | %-10s | %-10s | %-12d |\n",
        maxCount * sizeof(float),
        count,
        ncclFuncStr[ncclFuncTypes[i]],
        ncclAlgoStr[algo],
        ncclProtoStr[proto],
        nChannels);
    }
  }



  for (int i = 0; i < nranks; i++) {
    free(comm[i].connectSend);
    free(comm[i].connectRecv);
  }

  free(treeGraph);
  free(ringGraph);
  free(collNetGraph);
  free(allGather3Data);
  free(peerInfo);

  free(comm);
  printf("Done generating topology using %d: %s\n", model_id, desc->description);

  return 0;
}
