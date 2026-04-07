/*
Copyright (c) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.

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

#include "topo_expl_api.h"
#include "rccl_common.h"
#include "nccl.h"
#include <memory>
#include "model.h"
#include "topo_expl_impl.h"
#include "topo.h"
#include "graph.h"
#include "channel.h"
#include "nvmlwrap.h"
#include "bootstrap.h"
#include "transport.h"
#include "group.h"
#include "net.h"
#include "argcheck.h"

// Error checking macros that match NCCLCHECK and CUDACHECK
#define TOPO_NCCLCHECK(call) do { \
  ncclResult_t RES = call; \
  if (RES != ncclSuccess && RES != ncclInProgress) { \
    if (ncclDebugNoWarn == 0) INFO(NCCL_ALL, "%s:%d -> %d", __FILE__, __LINE__, RES); \
    return TOPO_EXPL_ERROR; \
  } \
} while (0)

#define TOPO_CUDACHECK(call) do { \
  hipError_t err = call; \
  if (err != hipSuccess) { \
    ncclResult_t RES = rcclCudaErrorHandler(err); \
    if (ncclDebugNoWarn == 0) INFO(NCCL_ALL, "%s:%d -> %d", __FILE__, __LINE__, RES); \
    return TOPO_EXPL_ERROR; \
  } \
} while (0)

NodeModel *node_model = nullptr;
extern ncclNet_t* ncclNet;

NCCL_PARAM(MinCTAs, "MIN_CTAS", 1);
NCCL_PARAM(MaxCTAs, "MAX_CTAS", 64);

struct TopoExplContext {
  NetworkModel* network;
  struct ncclComm* comms;
  int nRanks;
  int nNodes;
  struct ncclPeerInfo* peerInfo;
  struct allGatherInfo* allGather3Data;
  struct ncclTopoGraph* treeGraph;
  struct ncclTopoGraph* ringGraph;
  struct ncclTopoGraph* collNetGraph;
  struct ncclTopoGraph* nvlsGraph;
  
  TopoExplContext() 
    : network(nullptr), comms(nullptr), nRanks(0), nNodes(0),
      peerInfo(nullptr), allGather3Data(nullptr),
      treeGraph(nullptr), ringGraph(nullptr), 
      collNetGraph(nullptr), nvlsGraph(nullptr) {}
};

static ncclFunc_t toNcclFunc(TopoExplFunc func) {
  switch (func) {
    case TOPO_FUNC_ALLGATHER: return ncclFuncAllGather;
    case TOPO_FUNC_ALLREDUCE: return ncclFuncAllReduce;
    case TOPO_FUNC_ALLTOALL: return ncclFuncAlltoAllPivot;
    case TOPO_FUNC_ALLTOALL_GDA: return ncclFuncAlltoAllGda;
    case TOPO_FUNC_ALLTOALLV_GDA: return ncclFuncAlltoAllvGda;
    case TOPO_FUNC_BROADCAST: return ncclFuncBroadcast;
    case TOPO_FUNC_REDUCE: return ncclFuncReduce;
    case TOPO_FUNC_REDUCESCATTER: return ncclFuncReduceScatter;
    case TOPO_FUNC_SENDRECV: return ncclFuncSendRecv;
    default: return ncclFuncAllReduce;
  }
}

static ncclDataType_t toNcclDataType(TopoExplDataType dtype) {
  switch (dtype) {
    case TOPO_DTYPE_FLOAT32: return ncclFloat32;
    case TOPO_DTYPE_FLOAT64: return ncclFloat64;
    case TOPO_DTYPE_FLOAT16: return ncclFloat16;
    case TOPO_DTYPE_INT32: return ncclInt32;
    case TOPO_DTYPE_INT64: return ncclInt64;
    case TOPO_DTYPE_INT8: return ncclInt8;
    case TOPO_DTYPE_UINT8: return ncclUint8;
    case TOPO_DTYPE_BFLOAT16: return ncclBfloat16;
    default: return ncclFloat32;
  }
}

static TopoExplAlgo fromNcclAlgo(int algo) {
  if (algo >= 0 && algo <= 6) {
    return static_cast<TopoExplAlgo>(algo);
  }
  return TOPO_ALGO_RING;
}

static TopoExplProto fromNcclProto(int proto) {
  if (proto >= 0 && proto <= 2) {
    return static_cast<TopoExplProto>(proto);
  }
  return TOPO_PROTO_SIMPLE;
}

struct TopoExplContextDestroy {
  void operator()(TopoExplContext* ctx) const {
    topoExplDestroy(ctx);
  }
};

TopoExplResult topoExplCreate(
    const TopoExplConfig* config,
    TopoExplContext** context) {
  
  if (!config || !context || !config->xmlTopoFile) {
    return TOPO_EXPL_INVALID_ARG;
  }

  try {
    // Use unique_ptr with custom deleter to ensure proper cleanup on all error paths
    std::unique_ptr<TopoExplContext, TopoExplContextDestroy> ctx(new TopoExplContext());
    
    // Initialize CollNet
    initCollNet();
    
    // Create network model from XML file
    ctx->network = new NetworkModel();
    int numNodes = config->numNodes > 0 ? config->numNodes : 1;
    
    for (int i = 0; i < numNodes; i++) {
      NodeModel* node = new NodeModel(config->xmlTopoFile);
      ctx->network->AddNode(node);
    }
    
    ctx->nRanks = ctx->network->GetNRanks();
    ctx->nNodes = ctx->network->GetNNodes();
    
    // Allocate comm structures
    TOPO_NCCLCHECK(ncclCalloc(&ctx->comms, ctx->nRanks));
    
    // Allocate peer info
    TOPO_NCCLCHECK(ncclCalloc(&ctx->peerInfo, ctx->nRanks + 1)); // Extra for CollNet root
    
    // Allocate allGather3Data
    TOPO_NCCLCHECK(ncclCalloc(&ctx->allGather3Data, ctx->nRanks));
    
    // Allocate graphs
    TOPO_NCCLCHECK(ncclCalloc(&ctx->treeGraph, ctx->nRanks));
    TOPO_NCCLCHECK(ncclCalloc(&ctx->ringGraph, ctx->nRanks));
    TOPO_NCCLCHECK(ncclCalloc(&ctx->collNetGraph, ctx->nRanks));
    TOPO_NCCLCHECK(ncclCalloc(&ctx->nvlsGraph, ctx->nRanks));
    
    // Initialize comm structures
    int minCTAs = static_cast<int>(ncclParamMinCTAs());
    int maxCTAs = static_cast<int>(ncclParamMaxCTAs());
    
    for (int i = 0; i < ctx->nRanks; i++) {
      ctx->comms[i].rank = i;
      ctx->comms[i].nRanks = ctx->nRanks;
      TOPO_NCCLCHECK(ncclCalloc(&ctx->comms[i].connectSend, NCCL_MAX_CONNS * ctx->nRanks));
      TOPO_NCCLCHECK(ncclCalloc(&ctx->comms[i].connectRecv, NCCL_MAX_CONNS * ctx->nRanks));
      
      node_model = ctx->network->GetNode(i);
      if (!node_model) {
        return TOPO_EXPL_INTERNAL_ERROR;
      }
      
      ctx->comms[i].busId = node_model->getGpuBusId(i);
      ctx->comms[i].topo = node_model->getSystem(i);
      ctx->comms[i].peerInfo = ctx->peerInfo;
      ctx->comms[i].ncclNet = ncclNet;
      ctx->comms[i].config.maxCTAs = maxCTAs;
      ctx->comms[i].config.minCTAs = minCTAs;
      
      if (ctx->comms[i].topParentRanks == NULL) {
        TOPO_NCCLCHECK(ncclCalloc(&ctx->comms[i].topParentRanks, ctx->nRanks));
        for (int j = 0; j < ctx->nRanks; ++j) {
          ctx->comms[i].topParentRanks[j] = j;
        }
      }
      
      struct ncclSharedResources* sharedRes = NULL;
      TOPO_NCCLCHECK(ncclCalloc(&sharedRes, 1));
      sharedRes->owner = &ctx->comms[i];
      sharedRes->tpNRanks = ctx->comms[i].nRanks;
      TOPO_NCCLCHECK(ncclCalloc(&sharedRes->tpRankToLocalRank, ctx->comms[i].nRanks));
      ctx->comms[i].sharedRes = sharedRes;
      sharedRes->refCount = 1;
      ncclMemoryStackConstruct(&ctx->comms[i].memPermanent);
      
      // Mark channels as non-initialized
      for (int c = 0; c < MAXCHANNELS; c++) {
        ctx->comms[i].channels[c].id = -1;
      }
      
      TOPO_NCCLCHECK(fillInfo(&ctx->comms[i], ctx->comms[i].peerInfo + ctx->comms[i].rank, 0));
    }
    
    // Initialize transports
    for (int i = 0; i < ctx->nRanks; i++) {
      node_model = ctx->network->GetNode(i);
      if (!node_model) {
        return TOPO_EXPL_INTERNAL_ERROR;
      }
      TOPO_NCCLCHECK(initTransportsRank_1(&ctx->comms[i], ctx->allGather3Data, 
                          ctx->treeGraph[i], ctx->ringGraph[i], 
                          ctx->collNetGraph[i], ctx->nvlsGraph[i]));
    }
    
    for (int i = 0; i < ctx->nRanks; i++) {
      node_model = ctx->network->GetNode(i);
      if (!node_model) {
        return TOPO_EXPL_INTERNAL_ERROR;
      }
      TOPO_NCCLCHECK(initTransportsRank_3(&ctx->comms[i], ctx->allGather3Data,
                          ctx->treeGraph[i], ctx->ringGraph[i],
                          ctx->collNetGraph[i], ctx->nvlsGraph[i]));
      TOPO_CUDACHECK(hipDeviceGetAttribute(&ctx->comms[i].WarpSize, 
                                       hipDeviceAttributeWarpSize, 
                                       ctx->comms[i].cudaDev));
    }
    
    *context = ctx.release();
    return TOPO_EXPL_SUCCESS;
    
  } catch (...) {
    // unique_ptr will automatically call topoExplDestroy() via custom deleter
    return TOPO_EXPL_INTERNAL_ERROR;
  }
}

void topoExplDestroy(TopoExplContext* context) {
  if (!context) return;
  
  // Free comm structures
  if (context->comms) {
    for (int i = 0; i < context->nRanks; i++) {
      ncclMemoryStackDestruct(&context->comms[i].memPermanent);
      if (context->comms[i].sharedRes) {
        if (context->comms[i].sharedRes->tpRankToLocalRank) {
          free(context->comms[i].sharedRes->tpRankToLocalRank);
        }
        free(context->comms[i].sharedRes);
        context->comms[i].sharedRes = nullptr;
      }
      if (context->comms[i].topParentRanks) {
        free(context->comms[i].topParentRanks);
        context->comms[i].topParentRanks = nullptr;
      }
      if (context->comms[i].connectSend) {
        free(context->comms[i].connectSend);
      }
      if (context->comms[i].connectRecv) {
        free(context->comms[i].connectRecv);
      }
    }
    free(context->comms);
  }
  
  // Free graphs
  if (context->treeGraph) free(context->treeGraph);
  if (context->ringGraph) free(context->ringGraph);
  if (context->collNetGraph) free(context->collNetGraph);
  if (context->nvlsGraph) free(context->nvlsGraph);
  
  // Free other allocations
  if (context->allGather3Data) free(context->allGather3Data);
  if (context->peerInfo) free(context->peerInfo);
  
  // Free network model
  if (context->network) {
    delete context->network;
  }
  
  delete context;
}

TopoExplResult topoExplGetAlgoInfo(
    TopoExplContext* context,
    TopoExplFunc func,
    uint64_t count,
    TopoExplDataType dataType,
    TopoExplAlgoInfo* info) {
  
  if (!context || !info) {
    return TOPO_EXPL_INVALID_ARG;
  }
  
  // Use rank 0's comm for algorithm selection
  struct ncclComm* comm = &context->comms[0];
  
  ncclFunc_t ncclFunc = toNcclFunc(func);
  ncclDataType_t ncclDataType = toNcclDataType(dataType);
  
  int algo, proto, nChannels;
  TOPO_NCCLCHECK(rcclGetAlgoInfo(comm, ncclFunc, count, ncclDataType, 
                                      0, 0, 1, &algo, &proto, &nChannels));
  
  info->algo = fromNcclAlgo(algo);
  info->proto = fromNcclProto(proto);
  info->maxChannels = nChannels;
  
  // Get max send/recv count
  uint64_t maxCount;
  TOPO_NCCLCHECK(rcclFuncMaxSendRecvCount(ncclFunc, comm->nRanks, count, maxCount));

  // Support only fp32 production table
  info->maxSizeBytes = maxCount * sizeof(float);
  
  return TOPO_EXPL_SUCCESS;
}

int topoExplGetNumAlgos(void) {
  return NCCL_NUM_ALGORITHMS;
}

int topoExplGetNumProtos(void) {
  return NCCL_NUM_PROTOCOLS;
}

TopoExplResult topoExplGetRankInfo(
    TopoExplContext* context,
    int rank,
    int* nodeId,
    int* cudaDev,
    uint64_t* busId) {
  
  if (!context || !nodeId || !cudaDev || !busId) {
    return TOPO_EXPL_INVALID_ARG;
  }
  
  if (rank < 0 || rank >= context->nRanks) {
    return TOPO_EXPL_INVALID_ARG;
  }
  
  node_model = context->network->GetNode(rank);
  if (!node_model) {
    return TOPO_EXPL_INTERNAL_ERROR;
  }
  
  *nodeId = node_model->nodeId;
  *cudaDev = node_model->rankToCudaDev(rank);
  *busId = node_model->getGpuBusId(rank);
  
  return TOPO_EXPL_SUCCESS;
}

TopoExplResult topoExplGetAlgoTime(
    TopoExplContext* context,
    TopoExplFunc func,
    TopoExplAlgo algo,
    TopoExplProto proto,
    uint64_t count,
    float* time) {
  
  if (!context || !time) {
    return TOPO_EXPL_INVALID_ARG;
  }
  
  struct ncclComm* comm = &context->comms[0];
  
  ncclFunc_t ncclFunc = toNcclFunc(func);
  int ncclAlgo = static_cast<int>(algo);
  int ncclProto = static_cast<int>(proto);
  
  TOPO_NCCLCHECK(ncclTopoGetAlgoTime(comm, ncclFunc, ncclAlgo, ncclProto, count, 1, time));
  
  return TOPO_EXPL_SUCCESS;
}
