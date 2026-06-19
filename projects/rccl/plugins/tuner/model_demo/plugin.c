/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "tuner.h"
#define __hidden __attribute__ ((visibility("hidden")))
#define HOPPER_COMPCAP_IDX 2
// NVLink, PCI, Network
#define NCCL_HW_NVLINK 0
#define NCCL_HW_PCI 1
#define NCCL_HW_NET 2

static long log2i(long n) {
 long l = 0;
 while (n>>=1) l++;
 return l;
}
// Latencies in us, Bandwidths in GB/s
// Tree { LL, LL128, Simple } , Ring { LL, LL128, Simple }
static const float baseLat  [NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = { 
       { 12.0, 12.0, 17.0 }, { 12.0, 12.0, 17.0 },   // Tree, Ring
       { 12.0, 12.0, 17.0 }, { 12.0, 12.0, 17.0 },   // Collnet Direct, Chain
       {    0,    0,    0 }, {    0,    0,    0 }};  // NVLS, NVLS Tree

struct tuningModel {
  float hwLat[3][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float bwRatio[2][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float treeCorrectionFactor[NCCL_NUM_PROTOCOLS][27];
  float ringCorrectionFactor[NCCL_NUM_PROTOCOLS][27];
};

static struct tuningModel tuning_model = {
  {
    /* NVLINK */
    { /* Tree (LL/LL128/Simple)*/ { 0.8, 0.0, 2.5 }, /* Ring (LL/LL128/Simple)*/ { 0.8, 0.0, 3.6 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 0.8 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 0.0 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
    /* PCI */
    { /* Tree (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* Ring (LL/LL128/Simple)*/ { 2.2, 2.2, 5.7 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 5.7 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 5.7 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
    /* NET */
    { /* Tree (LL/LL128/Simple)*/ { 12.5, 0.0, 22.4 }, /* Ring (LL/LL128/Simple)*/ { 9.5, 0.0, 19.8 }, /* CollNetDirect (Simple)*/ { 0.0, 0.0, 12.5 }, /* CollNetChain (Simple)*/ { 0.0, 0.0, 0.0 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
  },

  {
    /* 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.41, 0.00, 1.00 }, /* Ring (LL/LL128/Simple)*/ { 0.41, 0.00, 1.00 }, /* CollNetDirect (Simple)*/ { 0.00, 0.00, 1.00 }, /* CollNetChain (Simple)*/ { 0.00, 0.00, 1.00 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
    /* more than 2 nodes */
    { /* Tree (LL/LL128/Simple)*/ { 0.41, 0.00, 0.86 }, /* Ring (LL/LL128/Simple)*/ { 0.41, 0.00, 1.00 }, /* CollNetDirect (Simple)*/ { 0.00, 0.00, 1.00 }, /* CollNetChain (Simple)*/ { 0.00, 0.00, 1.00 }, /* NVLS */ { 0, 0, 0 }, /* NVLS Tree */ { 0, 0, 0 } },
  },

  {
    { 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 0.8, 0.1, 0.4, 0.5, 1.0, 0.6, 0.4, 0.6, 0.1, 0.3, 0.4, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, },
    { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, },
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 0.4, 1.0, 1.0, 1.0, 0.2, 0.7, 1.0, 1.0, 1.0, 0.8, 0.7, 0.7, 0.8, 0.8, 0.8, 0.9, },
  },

  {
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.2, 0.2, 0.1, 0.5, 0.8, 1.0, 0.2, 0.4, 0.5, 0.4, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, },
    { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, },
    { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, },
  },
};

float latencies[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
float bandwidths[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];

ncclResult_t ncclTopoGetAlgoTime_Tuner(ncclFunc_t collType, int algorithm, int protocol, int numPipeOps, float* time, size_t nBytes) {
  float bw = bandwidths[collType][algorithm][protocol];
  float lat = latencies[collType][algorithm][protocol];

  if (bw == 0) {
    *time = -1.0; return ncclSuccess;
  }
  int logSize = log2i(nBytes>>6);
  if (algorithm == NCCL_ALGO_TREE) {
    if (logSize < 27) bw *= tuning_model.treeCorrectionFactor[protocol][logSize];
    else bw *= tuning_model.treeCorrectionFactor[protocol][26];
  }
  else if (algorithm == NCCL_ALGO_RING) {
    if(logSize < 27) bw *= tuning_model.ringCorrectionFactor[protocol][logSize];
    else bw *= tuning_model.ringCorrectionFactor[protocol][26];
  }

  int latCount = 1;
  *time = lat * latCount + (nBytes) / (1000 * bw);
  return ncclSuccess;
}

__hidden ncclResult_t pluginInit(size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction) { 
  if (nRanks <= 1) return ncclSuccess;
  int compCapIndex = HOPPER_COMPCAP_IDX;
  int index2 = nNodes <= 2 ? nNodes-1 : 2;
  int index1 = nNodes == 1 ? compCapIndex : 1;
  float ppn = (float)nRanks / nNodes; // if ppn < 2, then we are sending/receiving at the same GPU through the NIC, apply some bw discount

  int intraHw[NCCL_NUM_ALGORITHMS], hw[NCCL_NUM_ALGORITHMS];
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) intraHw[a] = NCCL_HW_NVLINK;
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) hw[a] = nNodes == 1 ? intraHw[a] : NCCL_HW_NET;
  for (int coll=0; coll<NCCL_NUM_FUNCTIONS; coll++) {
    int nsteps = coll == ncclFuncAllReduce ? 2*(nRanks-1) :
      coll == ncclFuncReduceScatter || coll == ncclFuncAllGather ? nRanks-1 :
      nRanks;
    int nInterSteps = coll == ncclFuncAllReduce ? (nNodes > 1 ? 2*nNodes :0) :
      coll == ncclFuncReduceScatter || coll == ncclFuncAllGather ? nNodes-1 :
      nNodes;

    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      if (coll == ncclFuncBroadcast && a != NCCL_ALGO_RING) continue;
      if (coll == ncclFuncReduce && a != NCCL_ALGO_RING) continue;
      if (coll == ncclFuncReduceScatter && a != NCCL_ALGO_RING && a != NCCL_ALGO_NVLS && a != NCCL_ALGO_COLLNET_DIRECT) continue;
      if (coll == ncclFuncAllGather && a != NCCL_ALGO_RING && a != NCCL_ALGO_NVLS && a != NCCL_ALGO_COLLNET_DIRECT) continue;

      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        if (a == NCCL_ALGO_TREE && p == NCCL_PROTO_SIMPLE && nNodes == 1) continue;
        if ((a == NCCL_ALGO_NVLS || a == NCCL_ALGO_NVLS_TREE) && p != NCCL_PROTO_SIMPLE) continue;
        int collnet = (a == NCCL_ALGO_COLLNET_DIRECT || a == NCCL_ALGO_COLLNET_CHAIN) ? 1 : 0;
        float bw = nNodes <= 2 || collnet ? 12.0 : 12.0; //graphs[a]->bwIntra : graphs[a]->bwInter
        if (a == NCCL_ALGO_NVLS) bw = 0.0;
        if (a == NCCL_ALGO_NVLS_TREE) bw = 0.0;
        if (collnet == 1) bw = 0.0;
        int nChannels = 28; //nNodes==1 && MI300
        float busBw = nChannels * bw; //comm->topo->baseBw != 0.0 ? comm->topo->baseBw : graphs[a]->nChannels * bw
        
        // Various model refinements
        if (nNodes <= 2)
          busBw *= tuning_model.bwRatio[0][a][p];
        else
          busBw *= tuning_model.bwRatio[1][a][p];
        if (a == NCCL_ALGO_RING && p == NCCL_PROTO_LL && (coll == ncclFuncBroadcast || coll == ncclFuncReduce) && nNodes == 1) { busBw = busBw * 1.65; }

        // Convert bus BW to algorithm BW
        if (!(a == NCCL_ALGO_COLLNET_DIRECT && (coll == ncclFuncAllGather || coll == ncclFuncReduceScatter))) {
          float ratio = 1.0f;
          if (a == NCCL_ALGO_RING) ratio *= (1.0 * nRanks) / nsteps;
          else if (a == NCCL_ALGO_NVLS || a == NCCL_ALGO_NVLS_TREE) ratio *= 5.0/6.0;
          else ratio *= .5;
          busBw *= ratio;
        }
        bandwidths[coll][a][p] = busBw;
        latencies[coll][a][p] = baseLat[a][p];
        float intraLat = tuning_model.hwLat[intraHw[a]][a][p];
        float interLat = tuning_model.hwLat[NCCL_HW_NET][a][p];

        if (a == NCCL_ALGO_RING) {
          float lat = tuning_model.hwLat[hw[a]][a][p];
          if ((coll == ncclFuncReduce || coll == ncclFuncBroadcast)) {
            latencies[coll][a][p] += lat;
          } else {
            // Inter-node rings still have to launch nsteps * net overhead.
            float netOverhead = 0.0;
            if (nNodes > 1) {
              netOverhead = 1;
              if (p == NCCL_PROTO_SIMPLE) netOverhead *= 3;
            }
            if (intraLat < netOverhead) intraLat = netOverhead;
            latencies[coll][a][p] += (nsteps-nInterSteps)*intraLat + nInterSteps*interLat;
          }
        } else if (a == NCCL_ALGO_TREE) {
          latencies[coll][a][p] +=
            2 * ((nRanks/nNodes-1) * intraLat + log2i(nNodes) * interLat);
        } else if (a == NCCL_ALGO_COLLNET_DIRECT) {
          int minimum = 1;
          if ((nRanks/nNodes-1) < 1) minimum = (nRanks/nNodes-1);
          latencies[coll][a][p] +=
            2 * (minimum * intraLat + (nRanks/nNodes-1) * 0.4) + interLat;  // Add 0.4 us arity serialization latency
        } else if (a == NCCL_ALGO_COLLNET_CHAIN) {
          latencies[coll][a][p] += 2 * (nRanks/nNodes-1) * intraLat + interLat;
        } else if (a == NCCL_ALGO_NVLS) {
          if (nNodes > 1) latencies[coll][a][p] += tuning_model.hwLat[NCCL_HW_NET][a][p];
        } else if (a == NCCL_ALGO_NVLS_TREE) {
          latencies[coll][a][p] += 2*(nNodes-1)*tuning_model.hwLat[NCCL_HW_NET][a][p];
        }
      }
    }
  }
  // Protocols/Algorithms enable/disable, and user overrides.
  // All are enabled except ll128 which is enabled by default only in certain cases.
  int protoEnable[NCCL_NUM_PROTOCOLS] = { 1, 2, 1 };
  int algoEnable[NCCL_NUM_ALGORITHMS] = { 1, 1, 1, 1, 1, 1 };

  // MNNVL: NVLS not yet supported
  algoEnable[NCCL_ALGO_NVLS_TREE] = 0;
  algoEnable[NCCL_ALGO_COLLNET_DIRECT] = 0;
  algoEnable[NCCL_ALGO_COLLNET_CHAIN] = 0;
  algoEnable[NCCL_ALGO_NVLS] = 0;

  for (int c=0; c<NCCL_NUM_FUNCTIONS; c++) for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    int pEnable = protoEnable[p];
    if (p == NCCL_PROTO_LL128) {
      pEnable = 0;
    }
    if (pEnable == 0) bandwidths[c][a][p] = 0;
    if (algoEnable[a] == 0) bandwidths[c][a][p] = 0;
  }
  return ncclSuccess;
}

__hidden ncclResult_t pluginGetCollInfo(void* context, ncclFunc_t collType, size_t nBytes,
                              int collNetSupport, int nvlsSupport, int numPipeOps,
                              int *algorithm, int *protocol, int* nChannels) {
                                
  float minTime = 3600000000.0; // Hopefully no operation will take an hour to complete.
  // Find algorithm / protocol.
  *algorithm = -1;
  *protocol = -1;
  int nAlgos = NCCL_NUM_ALGORITHMS;
  for (int a=0; a<nAlgos; a++) {
    if ((a == NCCL_ALGO_COLLNET_DIRECT || a == NCCL_ALGO_COLLNET_CHAIN) && collNetSupport != 1) continue;
    if ((a == NCCL_ALGO_NVLS || a == NCCL_ALGO_NVLS_TREE) && nvlsSupport != 1) continue;
    if (a == NCCL_ALGO_NVLS && collNetSupport != 1) continue;
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      if (p == NCCL_PROTO_LL128) continue;
      float time;
      ncclTopoGetAlgoTime_Tuner(collType, a, p, numPipeOps, &time, nBytes);
        if (time >= 0 && time < minTime) {
          *algorithm = a;
          *protocol = p;
          minTime = time;
        }
    }
  }
  return ncclSuccess;
}

__hidden ncclResult_t pluginDestroy(void* context) { return ncclSuccess; }

#define PLUGIN_NAME "Example"

const ncclTuner_v4_t ncclTunerPlugin_v4 = {
  .name = PLUGIN_NAME,
  .init = pluginInit,
  .getCollInfo = pluginGetCollInfo,
  .destroy = pluginDestroy
};
