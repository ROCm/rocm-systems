/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "device.h"
#include "graph.h"
#include "transport.h"
#include "trees.h"
#include "rings.h"
#include "topo.h"

#include "msccl/msccl_lifecycle.h"

/******************************************************************/
/********************* Internode connection ***********************/
/******************************************************************/

ncclResult_t ncclTopoPreset(struct ncclComm* comm, struct ncclTopoGraph** graphs, struct ncclTopoRanks* topoRanks) {
  int rank = comm->rank;
  int localRanks = comm->topo->nodes[GPU].count;
  int nChannels = comm->nChannels;

  topoRanks->nvlsHeadNum = 0;
  for (int c=0; c<nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    channel->ring.prev = channel->ring.next = -1;
    channel->tree.up = -1;
    channel->collnetChain.up = -1;
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) channel->tree.down[i] = -1;
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) channel->collnetChain.down[i] = -1;
    channel->collnetDirect.out = -1;
    channel->collnetDirect.headRank = -1;
    channel->collnetDirect.nHeads = 0;
    channel->collnetDirect.shift = 0;
    for (int i=0; i<NCCL_MAX_DIRECT_ARITY+1; i++) channel->collnetDirect.heads[i] = -1;
    for (int i=0; i<NCCL_MAX_DIRECT_ARITY; i++) channel->collnetDirect.up[i] = -1;
    for (int i=0; i<NCCL_MAX_DIRECT_ARITY; i++) channel->collnetDirect.down[i] = -1;

    int* ringIntra = graphs[NCCL_ALGO_RING]->intra+c*localRanks;
    int* treeIntra = graphs[NCCL_ALGO_TREE]->intra+c*localRanks;
    int* collNetIntra = graphs[NCCL_ALGO_COLLNET_CHAIN]->intra+c*localRanks;

    for (int i=0; i<localRanks; i++) {
      if (ringIntra[i] == rank) {
        topoRanks->ringRecv[c] = ringIntra[0];
        topoRanks->ringSend[c] = ringIntra[localRanks-1];
        topoRanks->ringPrev[c] = (i == 0) ? -1 : ringIntra[i-1];
        topoRanks->ringNext[c] = (i == localRanks-1) ? -1 : ringIntra[i+1];
      }
      if (treeIntra[i] == rank) {
        int parentIndex = 0;
        int child0Index = graphs[NCCL_ALGO_TREE]->pattern == NCCL_TOPO_PATTERN_TREE ? 0 : 1;
        int child1Index = graphs[NCCL_ALGO_TREE]->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE ? 1 : 0;

        topoRanks->treeToParent[c] = treeIntra[parentIndex];
        topoRanks->treeToChild0[c] = treeIntra[child0Index];
        topoRanks->treeToChild1[c] = treeIntra[child1Index];
        channel->tree.up         = i == 0 ? -1 : treeIntra[i-1];
        channel->tree.down[0]    = i == localRanks-1 ? -1 : treeIntra[i+1];
      }
      if (collNetIntra[i] == rank) {
        channel->collnetChain.up      = i == 0 ? comm->nRanks : collNetIntra[i-1];
        channel->collnetChain.down[0] = i == localRanks-1 ? -1 : collNetIntra[i+1];
      }
    }
  }
  // Duplicate channels trees
  struct ncclChannel* channel0 = comm->channels;
  struct ncclChannel* channel1 = (nChannels > MAXCHANNELS/2) ? 0 : channel0+nChannels;
  if (channel1) memcpy(channel1, channel0, nChannels*sizeof(struct ncclChannel));

  // Get nvls heads and the number of heads. Duplicate head is not allowed.
  for (int c = 0; c < graphs[NCCL_ALGO_NVLS]->nChannels; ++c) {
    bool addHead = true;
    int* nvlsIntra = graphs[NCCL_ALGO_NVLS]->intra + c * localRanks;

    for (int dup = 0; dup < topoRanks->nvlsHeadNum; dup++) {
      if (topoRanks->nvlsHeads[dup] == nvlsIntra[0]) {
        addHead = false;
        break;
      }
    }
    if (addHead) {
      topoRanks->nvlsHeads[topoRanks->nvlsHeadNum++] = nvlsIntra[0];
    }
  }
  memcpy(comm->nvlsHeads, topoRanks->nvlsHeads, sizeof(int) * topoRanks->nvlsHeadNum);

  return ncclSuccess;
}

bool isRankHere(const char* s, int start, int end, int rank) {
  if (end <= start || start < 0 || end < 0)
    return false;
  int num = 0;
  while (start < end) {
    char currChar = s[start];
    if (isdigit(currChar)) {
      num = num * 10 + (currChar - '0');
      if (isdigit(s[start+1])) {
        start++;
        continue;
      }
    }
    else if (currChar == '(' || currChar == ')') {
      start++;
      num = 0;
      continue;
    }
    if (num == rank) return true;
    start++;
  }
  return false;
}

ncclResult_t ncclTreeBasePostset(struct ncclComm* comm,
    struct ncclTopoGraph* treeGraph) {
  int x=0;
  for (int i=0;  treeGraph->treeBase[i][0]!=0; i++)
  {
    x=i+1;
  }
  if( treeGraph->treeBase[0][0] == 0) return ncclSuccess;
  int nChannels = comm->nChannels;
  //int localRanks = comm->topo->nodes[GPU].count; // unused variable - compiler warning
  //new tree
  for (int c=0; c<nChannels; c++) { // in here
    int buff = c%x;
    char tempString[NCCL_TOPO_MAX_NODES*4];
    int ko=0;
    while (treeGraph->treeBase[buff][ko] != 0) {
      tempString[ko] = treeGraph->treeBase[buff][ko];
      ko++;
    }
    tempString[ko]=0;
    int start = 0;
    int curRank = comm->rank;
    struct ncclChannel* channel = comm->channels+c;
    int end = 0;
    while (tempString[end] != 0) end++;
    int parent = -1;
    // constructing a number from the continuous digits
    while (start < end) {
      int num = 0, num_found = 0;
      start++;
      while (start < end && tempString[start] != '('
         && tempString[start] != ')') {
        int num_here = (int)(tempString[start] - '0');
        num = num * 10 + num_here;
        start = start + 1;
        if (tempString[start] == '(' || tempString[start] == ')' || start == end) num_found = 1;
      }
      if (num_found != 0 && num == curRank) {
        channel->tree.up = parent;
        int depth = 0;
        for (int childId = 0; childId < NCCL_MAX_TREE_ARITY; childId++) {
          int or_start = start;
          int child = -1;
          channel->tree.down[childId] = -1;
          if (or_start >= end -1) continue;
          num=0;
          or_start++;
          while (tempString[or_start] != 0 && tempString[or_start] != '('
             && tempString[or_start] != ')') {
            int num_here = (int)(tempString[or_start] - '0');
            num = num * 10 + num_here;
            or_start++;
          }
          child = num;
          // find next child start
          while (start < end) {
            if (tempString[start] == '(' ) depth++;
            else if(tempString[start] == ')') depth--;
            if (depth == 0) break; // next child
            start++;
          }
          start++;
          channel->tree.down[childId] = child;
          // get kids, update numbers, get out of this string
        }
        break;
      }
      else { //go to the next one
        parent = num;
        int start_c = start;
        int end_c = start_c;
        while (end_c < end) {
          int depth = 0;
          while (end_c < end) {
            if (tempString[end_c] == '(' ) depth++;
            else if(tempString[end_c] == ')') depth--;
            if (depth == 0) break; // next child
            end_c++;
          }
          if (isRankHere(tempString, start_c, end_c, curRank)) {
            start = start_c;
            end = end_c;
            break;
          }
          else {
            end_c++;
            start_c = end_c;
          }
        }
      }
    }

  }
  return ncclSuccess;
}

static ncclResult_t connectRings(struct ncclComm* comm, int* ringRecv, int* ringSend, int* ringPrev, int* ringNext) {
  int nChannels = comm->nChannels;
  int nNodes = comm->nNodes;
  for (int c=0; c<nChannels; c++) {
    int* recv = ringRecv+c*comm->nNodes;
    int* send = ringSend+c*comm->nNodes;
    int* prev = ringPrev+c*comm->nRanks;
    int* next = ringNext+c*comm->nRanks;
    for (int n=0; n<nNodes; n++) {
      int recvRank = recv[n];
      int prevSendRank = send[(n-1+nNodes)%nNodes];
      prev[recvRank] = prevSendRank;
      int sendRank = send[n];
      int nextRecvRank = recv[(n+1)%nNodes];
      next[sendRank] = nextRecvRank;
    }
  }

  // [RCCL] Print off the recv/send local ranks per node, per channel
  if (comm->rank == 0)
  {
    char buff[2048] = "";
    int offset = 0;
    int inc;
    int numChannels = (nChannels > MAXCHANNELS/2) ? 2 * nChannels : nChannels;

    for (int c = 0; c < numChannels; c++) {
      sprintf(buff + offset, "     %02d%n", c, &inc);
      offset += inc;
    }
    INFO(NCCL_GRAPH, "[RINGS] %s", buff);

    for (int n = 0; n < nNodes; n++) {
      offset = 0;
      for (int c = 0; c < nChannels; c++) {
        int recvRank = comm->rankToLocalRank[ringRecv[c*comm->nNodes+n]];
        int sendRank = comm->rankToLocalRank[ringSend[c*comm->nNodes+n]];
        sprintf(buff + offset, " %02d->%02d%n",  recvRank, sendRank, &inc);
        offset += inc;
      }
      INFO(NCCL_GRAPH, "[RINGS] %s", buff);
    }
  }

  return ncclSuccess;
}

static ncclResult_t getIndexes(int* ranks, int* indexes, int nNodes) {
 for (int n=0; n<nNodes; n++) indexes[n] = ranks[n];
 return ncclSuccess;
}

static ncclResult_t setTreeUp(struct ncclTree* tree, int* indexes, int u) {
  if (u == -1) return ncclSuccess;
  tree->up = indexes[u];
  return ncclSuccess;
}

static ncclResult_t setTreeDown(struct ncclTree* tree, int* indexes, int d) {
  if (d == -1) return ncclSuccess;
  int x = 0;
  while (x < NCCL_MAX_TREE_ARITY && tree->down[x] >= 0) x++;
  if (x == NCCL_MAX_TREE_ARITY) {
    WARN("Internal error : tree already has %d children (%d %d %d)", x, tree->down[0], tree->down[1], tree->down[2]);
    return ncclInternalError;
  }
  tree->down[x] = indexes[d];
  return ncclSuccess;
}

static ncclResult_t connectTrees(struct ncclComm* comm, int* treeToParent, int* treeToChild0, int* treeToChild1, int* treePatterns) {

  const int channelLimit = (IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942") || IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950")) ? 2*CHANNEL_LIMIT : CHANNEL_LIMIT;
  const int nChannels = (comm->nChannels > channelLimit) ? comm->nChannels / 2 : comm->nChannels;
  const int nNodes = comm->nNodes, node = comm->node;

  // Compute tree depth. Not an exact value but a good approximation in most
  // cases
  int depth = comm->nRanks/nNodes - 1 + log2i(nNodes);

  int t0u, t0d0, t0d1, t0ChildType, t1u, t1d0, t1d1, t1ChildType;
  int* ttp, *ttc0, *ttc1;
  NCCLCHECK(ncclGetDtree(nNodes, node, &t0u, &t0d0, &t0d1, &t0ChildType, &t1u, &t1d0, &t1d1, &t1ChildType));
  if (nChannels == comm->nChannels) {
    for (int c=0; c<nChannels; c++) {
       struct ncclChannel* channel0 = comm->channels+c;
       struct ncclChannel* channel1 = channel0+nChannels;
       ttp = treeToParent+c*comm->nNodes;
       ttc0 = treeToChild0+c*comm->nNodes;
       ttc1 = treeToChild1+c*comm->nNodes;
       if (comm->rank == ttp[node]) {
         NCCLCHECK(setTreeUp(&channel0->tree, t0ChildType == 0 ? ttc0 : ttc1, t0u));
         NCCLCHECK(setTreeUp(&channel1->tree, t1ChildType == 0 ? ttc0 : ttc1, t1u));
       }
       if (comm->rank == ttc0[node]) {
         NCCLCHECK(setTreeDown(&channel0->tree, ttp, t0d0));
         NCCLCHECK(setTreeDown(&channel1->tree, ttp, t1d0));
       }
       if (comm->rank == ttc1[node]) {
         NCCLCHECK(setTreeDown(&channel0->tree, ttp, t0d1));
         NCCLCHECK(setTreeDown(&channel1->tree, ttp, t1d1));
       }
       if (comm->rank == ttp[node] ||
           comm->rank == ttc0[node] ||
           comm->rank == ttc1[node]) {
         INFO(NCCL_GRAPH, "Tree %d : %d -> %d -> %d/%d/%d", c,           channel0->tree.up, comm->rank, channel0->tree.down[0], channel0->tree.down[1], channel0->tree.down[2]);
         INFO(NCCL_GRAPH, "Tree %d : %d -> %d -> %d/%d/%d", c+nChannels, channel1->tree.up, comm->rank, channel1->tree.down[0], channel1->tree.down[1], channel1->tree.down[2]);
       }
       channel0->tree.depth = channel1->tree.depth = depth;
    }
  } else {
    for (int c=0; c<nChannels; c++) {
       struct ncclChannel* channel0 = comm->channels+c;
       ttp = treeToParent+c*comm->nNodes;
       ttc0 = treeToChild0+c*comm->nNodes;
       ttc1 = treeToChild1+c*comm->nNodes;
       if (comm->rank == ttp[node]) {
         NCCLCHECK(setTreeUp(&channel0->tree, t0ChildType == 0 ? ttc0 : ttc1, t0u));
       }
       if (comm->rank == ttc0[node]) {
         NCCLCHECK(setTreeDown(&channel0->tree, ttp, t0d0));
       }
       if (comm->rank == ttc1[node]) {
         NCCLCHECK(setTreeDown(&channel0->tree, ttp, t0d1));
       }
       if (comm->rank == ttp[node] ||
           comm->rank == ttc0[node] ||
           comm->rank == ttc1[node]) {
         INFO(NCCL_GRAPH, "Tree %d : %d -> %d -> %d/%d/%d", c,           channel0->tree.up, comm->rank, channel0->tree.down[0], channel0->tree.down[1], channel0->tree.down[2]);
       }
       channel0->tree.depth = depth;
    }
    for (int c=nChannels; c<nChannels*2; c++) {
       struct ncclChannel* channel1 = comm->channels+c;
       ttp = treeToParent+c*comm->nNodes;
       ttc0 = treeToChild0+c*comm->nNodes;
       ttc1 = treeToChild1+c*comm->nNodes;
       if (comm->rank == ttp[node]) {
         NCCLCHECK(setTreeUp(&channel1->tree, t1ChildType == 0 ? ttc0 : ttc1, t1u));
       }
       if (comm->rank == ttc0[node]) {
         NCCLCHECK(setTreeDown(&channel1->tree, ttp, t1d0));
       }
       if (comm->rank == ttc1[node]) {
         NCCLCHECK(setTreeDown(&channel1->tree, ttp, t1d1));
       }
       if (comm->rank == ttp[node] ||
           comm->rank == ttc0[node] ||
           comm->rank == ttc1[node]) {
         INFO(NCCL_GRAPH, "Tree %d : %d -> %d -> %d/%d/%d", c+nChannels, channel1->tree.up, comm->rank, channel1->tree.down[0], channel1->tree.down[1], channel1->tree.down[2]);
       }
       channel1->tree.depth = depth;
    }
  }
  return ncclSuccess;
}

static ncclResult_t connectCollNet(struct ncclComm* comm, struct ncclTopoGraph* collNetGraph) {
  int rank = comm->rank;
  int localRanks = comm->localRanks;
  int nHeads = 0;
  int *heads;
  NCCLCHECK(ncclCalloc(&heads, localRanks));
  // Find all head ranks
  // Head index is always 0
  for (int c=0; c<collNetGraph->nChannels; c++) {
    int* collNetIntra = collNetGraph->intra+c*localRanks;
    int head = collNetIntra[0];
    for (int h=0; h<nHeads; h++) if (heads[h] == head) head = -1;
    if (head != -1) heads[nHeads++] = collNetIntra[0];
  }
  // For all channels
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    char line[1024];
    sprintf(line, "CollNetDirect channel %d rank %d ", c, rank);
    int nDown = 0;
    for (int i=0; i<nHeads; i++) {
      if (rank == heads[i]) { // is head
        channel->collnetDirect.headRank = i; // Mark the index for deciding offset in the CUDA kernel
        channel->collnetDirect.out = comm->nRanks; // Set root of collnetDirect to id nranks
        int* collNetIntra = collNetGraph->intra+i*localRanks;
        sprintf(line+strlen(line), "down ");
        for (int r=0; r<localRanks; r++) {
          if (collNetIntra[r] == rank) continue;
          channel->collnetDirect.down[nDown++] = collNetIntra[r];  // connect to all peers
          sprintf(line+strlen(line), " %d ", collNetIntra[r]);
        }
        sprintf(line+strlen(line), "nDown %d ", nDown);
        break;
      }
    }
    // Connect to all heads
    int nUp = 0;
    sprintf(line+strlen(line), "up ");
    for (int h=0; h<nHeads; h++) {
      if (rank == heads[h]) continue;
      channel->collnetDirect.up[nUp++] = heads[h];
      sprintf(line+strlen(line), " %d ", heads[h]);
    }
    sprintf(line+strlen(line), "heads ");
    { // heads[] is the list of heads ordered in head order startubg with self
      int h0 = (channel->collnetDirect.headRank == -1) ? 0 : channel->collnetDirect.headRank;
      for (int h1=0; h1 < nHeads; h1++) {
        int h = (h0+h1)%nHeads;
        channel->collnetDirect.heads[h1] = heads[h];
        sprintf(line+strlen(line), " %d ", heads[h]);
      }
    }
    channel->collnetDirect.nHeads = nHeads;
    // nHeads should always be greater than 0.
    // coverity[divide_by_zero]
    channel->collnetDirect.shift = (rank%localRanks)%nHeads; // Shift by intraRank so that leaves don't send to same head simultaneously
    channel->collnetDirect.depth = (nUp == 0 && nDown == 0) ? 1 : 2;
    sprintf(line+strlen(line), "nUp %d nHeads %d ", nUp, nHeads);
    sprintf(line+strlen(line), "headRank %d out %d shift %d", channel->collnetDirect.headRank, channel->collnetDirect.out, channel->collnetDirect.shift);
    INFO(NCCL_GRAPH, "%s", line);
    channel->collnetChain.depth = comm->nRanks/comm->nNodes;
  }
  free(heads);
  return ncclSuccess;
}

static ncclResult_t connectNvls(struct ncclComm* comm, int* nvlsHeads, int nHeads) {
  int headRank = -1;
  if (nHeads == 0) {
    comm->nvlsChannels = 0;
    return ncclSuccess;
  }

  for (int h = 0; h < nHeads; h++) {
    if (nvlsHeads[h * comm->nNodes + comm->node] == comm->rank) headRank = h;
  }

  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    channel->nvls.nHeads = nHeads;
    for (int h=0; h<nHeads; h++) channel->nvls.up[h] = comm->nRanks+1+h;
    for (int h=nHeads; h<NCCL_MAX_NVLS_ARITY; h++) channel->nvls.up[h] = -1;
    channel->nvls.down = comm->nRanks+1+headRank;
    channel->nvls.out = -1;       // NVLS+SHARP not yet implemented.
    channel->nvls.headRank = headRank;
    channel->nvls.treeUp = channel->nvls.treeDown[0] = channel->nvls.treeDown[1] = channel->nvls.treeDown[2] = -1;
    if (comm->config.collnetEnable && channel->nvls.headRank != -1) channel->nvls.out = comm->nRanks;
  }
  if (comm->nNodes == 1) return ncclSuccess;

  // Connect Trees
  int tree0Parent, tree0Child0, tree0Child1, tree1Parent, tree1Child0, tree1Child1;
  int pc0, pc1; // ignored
  NCCLCHECK(ncclGetDtree(comm->nNodes, comm->node,
        &tree0Parent, &tree0Child0, &tree0Child1, &pc0,
        &tree1Parent, &tree1Child0, &tree1Child1, &pc1));

  int* heads = NULL;
  int treeUp[2] = { -1, -1 };
  int treeDown0[2] = { -1, -1 };
  int treeDown1[2] = { -1, -1 };

  if (comm->node == 0) {
    for (int h=0; h<nHeads; h++) {
      char line[1024];
      sprintf(line, "NVLS Head %2d:", h);
      heads = nvlsHeads+h*comm->nNodes;
      for (int n=0; n<comm->nNodes && n<20; n++) {
        sprintf(line+strlen(line), " %2d", heads[n]);
      }
      INFO(NCCL_INIT, "%s", line);
    }
  }

  // Find the heads where I'm the head rank and retain tree up/down
  for (int h=0; h<nHeads; h++) {
    heads = nvlsHeads+h*comm->nNodes;
    if (heads[comm->node] == comm->rank) {
      treeUp[0] = tree0Parent == -1 ? -1: heads[tree0Parent];
      treeDown0[0] = tree0Child0 == -1 ? -1 : heads[tree0Child0];
      treeDown1[0] = tree0Child1 == -1 ? -1 : heads[tree0Child1];
      treeUp[1] = tree1Parent == -1 ? -1 : heads[tree1Parent];
      treeDown0[1] = tree1Child0 == -1 ? -1 : heads[tree1Child0];
      treeDown1[1] = tree1Child1 == -1 ? -1 : heads[tree1Child1];
      break;
    }
  }
  // Set prev/next in all channels (NVLS compute channels work
  // orthogonally to NVLS search channels).
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    channel->nvls.treeUp = treeUp[c%2];
    channel->nvls.treeDown[0] = channel->nvls.down;
    int ix = 1;
    if (treeDown0[c%2] != -1) channel->nvls.treeDown[ix++] = treeDown0[c%2];
    if (treeDown1[c%2] != -1) channel->nvls.treeDown[ix] = treeDown1[c%2];
  }

  struct ncclNvls* nvls0 = &comm->channels[0].nvls;
  struct ncclNvls* nvls1 = &comm->channels[1].nvls;
  INFO(NCCL_GRAPH, "NVLS Trees : %d/%d/%d->%d->%d %d/%d/%d->%d->%d",
      nvls0->treeDown[0], nvls0->treeDown[1], nvls0->treeDown[2], comm->rank, nvls0->treeUp,
      nvls1->treeDown[0], nvls1->treeDown[1], nvls1->treeDown[2], comm->rank, nvls1->treeUp);
  return ncclSuccess;
}

// Legacy naming
NCCL_PARAM(MinNrings, "MIN_NRINGS", -2);
NCCL_PARAM(MaxNrings, "MAX_NRINGS", -2);
// New naming
NCCL_PARAM(MinNchannels, "MIN_NCHANNELS", -2);
NCCL_PARAM(MaxNchannels, "MAX_NCHANNELS", -2);

int ncclMinNchannels() {
  int minNchannels = 2;
  if (ncclParamMinNrings() != -2) minNchannels = ncclParamMinNrings();
  if (ncclParamMinNchannels() != -2) minNchannels = ncclParamMinNchannels();
  if (minNchannels > MAXCHANNELS) {
    INFO(NCCL_GRAPH|NCCL_ENV, "User asked for a minimum of %d channels, limiting to %d", minNchannels, MAXCHANNELS);
    minNchannels = MAXCHANNELS;
  }
  if (minNchannels < 0) minNchannels = 0;
  return minNchannels;
}

extern int64_t ncclParamWorkArgsBytes();

int ncclMaxNchannels() {
  int maxNchannels = MAXCHANNELS;
  if (ncclParamMaxNrings() != -2) maxNchannels = ncclParamMaxNrings();
  if (ncclParamMaxNchannels() != -2) maxNchannels = ncclParamMaxNchannels();
  maxNchannels = std::min(maxNchannels, ncclDevMaxChannelsForArgsBytes(ncclParamWorkArgsBytes()));
  if (maxNchannels > MAXCHANNELS) maxNchannels = MAXCHANNELS;
  if (maxNchannels < 1) {
    INFO(NCCL_GRAPH|NCCL_ENV, "User asked for a maximum of %d channels, setting it to 1", maxNchannels);
    maxNchannels = 1;
  }
  return maxNchannels;
}

static int copyChannels(struct ncclComm* comm, int start, int end, int* ringPrev, int* ringNext) {
  int nranks = comm->nRanks;
  int c;
  for (c=start; c<end; c++) {
    memcpy(ringPrev+c*nranks, ringPrev+(c-start)*nranks, nranks*sizeof(int));
    memcpy(ringNext+c*nranks, ringNext+(c-start)*nranks, nranks*sizeof(int));
    memcpy(comm->channels+c, comm->channels+c-start, sizeof(struct ncclChannel));
  }
  return c;
}

void exchangeValues(int* v0, int* v1) {
  int tmp = *v1;
  *v1 = *v0;
  *v0 = tmp;
}

int getTreeNodeParity(int treeDir, int nNodes, int node)
{
  if (node == -1) return -1;

  int parentNodes[2], child0Nodes[2], child1Nodes[2], childTypes[2];
  ncclGetDtree(nNodes, node,
               &parentNodes[0], &child0Nodes[0], &child1Nodes[0], &childTypes[0],
               &parentNodes[1], &child0Nodes[1], &child1Nodes[1], &childTypes[1]);

  // Uptree and downtree have different parity
  if (parentNodes[treeDir] == -1) return treeDir;

  // Recurse and swap parity if this is child that exits from 2nd intranode rank (childType == 0)
  return ((childTypes[treeDir] + 1) + getTreeNodeParity(treeDir, nNodes, parentNodes[treeDir])) % 2;
}

// [RCCL] Build rail-optimized trees
ncclResult_t connectRailOptimizedTrees(struct ncclComm* comm, int* treeToParent, int* treeToChild0, int* treeToChild1)
{
  INFO(NCCL_GRAPH, "Building rail-optimized trees for %d nodes", comm->nNodes);

  /* Rail-optimized trees are implemented in RCCL via a set of specially crafted complimentary
     pairs of intra-node XGMI paths such that:
        A) Complimentary pairs alternate their first two elements
        B) Cover all the XGMI links exactly once

     E.g: For MI300X
       Path 1A 0 1 2 4 3 6 5 7       Path 2A 2 3 0 5 6 1 7 4
       Path 1B 1 0 4 7 3 5 2 6       Path 2B 3 2 7 0 6 4 1 5
               ^ ^                           ^ ^

       Path 3A 4 5 1 6 0 3 7 2       Path 4A 6 7 5 3 4 0 2 1
       Path 3B 5 4 6 2 0 7 1 3       Path 4B 7 6 3 1 4 2 5 0
               ^ ^                           ^ ^

     Due to the balanced tree pattern, the 1st rank in the Path gets connected to the first
     rank of the path of the left child node, while the 2nd rank in the path connects to the first
     rank of the right child node.

     In order to avoid crossing rails, any time a right child is visited, the channel should
     be swapped:

                             AB              BA
                            /  \            /  \
                           /    \          /    \
                          AB     BA       BA     AB
                         /  \   /  \     / \    /  \
                        AB  BA BA   AB  BA  AB AB   BA
  */

  const int nChannels = (comm->nChannels / 2);
  const int nNodes = comm->nNodes, node = comm->node, rank = comm->rank;
  const int depth = comm->nRanks/nNodes - 1 + log2i(nNodes);
  const int nGpus = comm->topo->nodes[GPU].count;
  struct ncclTopoGraph* treeGraph = &comm->graphs[NCCL_ALGO_TREE];

  // Compute parent/child nodes for this current node, for uptree and downtree
  int parentNodes[2], child0Nodes[2], child1Nodes[2], childTypes[2];
  NCCLCHECK(ncclGetDtree(nNodes, node,
                         &parentNodes[0], &child0Nodes[0], &child1Nodes[0], &childTypes[0],
                         &parentNodes[1], &child0Nodes[1], &child1Nodes[1], &childTypes[1]));

  // Loop over up-tree / down-tree
  for (int treeDir = 0; treeDir < 2; treeDir++) {
    // Collect the parent / child nodes for this tree direction
    int parentNode = parentNodes[treeDir];
    int child0Node = child0Nodes[treeDir];
    int child1Node = child1Nodes[treeDir];

    int* treeToChild = (childTypes[treeDir] == 0) ? treeToChild0 : treeToChild1;

    // Compute the parity for nodes for this tree direction
    int nodeParity   = getTreeNodeParity(treeDir, nNodes, node);
    int parentParity = getTreeNodeParity(treeDir, nNodes, parentNode);
    int child0Parity = getTreeNodeParity(treeDir, nNodes, child0Node);
    int child1Parity = getTreeNodeParity(treeDir, nNodes, child1Node);

    // Loop over pairs of complimentary channels
    for (int ch = 0; ch < nChannels; ch += 2) {
      int ch0 = treeDir * nChannels + ch;
      int ch1 = ch0 + 1;

      ncclChannel* channel[2] = {&comm->channels[ch0], &comm->channels[ch1]};
      channel[0]->tree.depth = channel[1]->tree.depth = depth;

      // Determine ranks that connect to other nodes for each of the two channels
      int rankToParent[2] = {treeToParent[ch0 * nNodes + node], treeToParent[ch1 * nNodes + node]};
      int rankToChild0[2] = {treeToChild0[ch0 * nNodes + node], treeToChild0[ch1 * nNodes + node]};
      int rankToChild1[2] = {treeToChild1[ch0 * nNodes + node], treeToChild1[ch1 * nNodes + node]};

      // All ranks swizzle channels.  This maintains internal rank structures setup during TopoPreset
      if (nodeParity) {
        std::swap(channel[0]->tree, channel[1]->tree);
        std::swap(rankToParent[0], rankToParent[1]);
        std::swap(rankToChild0[0], rankToChild0[1]);
        std::swap(rankToChild1[0], rankToChild1[1]);

        // Swap NICs
        std::swap(treeGraph->inter[ch0 * 2    ], treeGraph->inter[ch1 * 2    ]);
        std::swap(treeGraph->inter[ch0 * 2 + 1], treeGraph->inter[ch1 * 2 + 1]);

        // Swap lines
        for (int j = 0; j < nGpus; j++)
          std::swap(treeGraph->intra[ch0 * nGpus + j], treeGraph->intra[ch1 * nGpus + j]);
      }

      // Connect ranks that connect to other nodes for each of the two channels
      for (int i = 0; i < 2; i++) {
        if (rank == rankToParent[i] && parentNode != -1) {
          // Connect this rank to correct child rank on parent node
          int parentChannel = (parentParity + i) % 2 == 0 ? ch0 : ch1;
          channel[i]->tree.up = treeToChild[parentChannel * nNodes + parentNode];
        }

        if (rank == rankToChild0[i] && child0Node != -1) {
          // Connect this rank to the parent rank on child0 node
          int child0Channel = (child0Parity + i) % 2 == 0 ? ch0 : ch1;
          setTreeDown(&channel[i]->tree, treeToParent + child0Channel * nNodes, child0Node);
        }

        if (rank == rankToChild1[i] && child1Node != -1) {
          // Connect this rank to the parent rank on child1 node
          int child1Channel = (child1Parity + i) % 2 == 0 ? ch0 : ch1;
          setTreeDown(&channel[i]->tree, treeToParent + child1Channel * nNodes, child1Node);
        }
        if (rank == rankToParent[i] ||
            rank == rankToChild0[i] ||
            rank == rankToChild1[i]) {
          INFO(NCCL_GRAPH, "Tree %d : %d -> %d -> %d/%d/%d", (i == 0 ? ch0 : ch1),
               channel[i]->tree.up, rank,
               channel[i]->tree.down[0],
               channel[i]->tree.down[1],
               channel[i]->tree.down[2]);
        }
      }
    }
  }
  return ncclSuccess;
}

NCCL_PARAM(UnpackDoubleNChannels, "UNPACK_DOUBLE_NCHANNELS", 1);
RCCL_PARAM(OutputTrees, "OUTPUT_TREES", 0);

ncclResult_t ncclTopoPostset(struct ncclComm* comm, int* firstRanks, int* treePatterns, struct ncclTopoRanks** allTopoRanks, int* rings, struct ncclTopoGraph** graphs, struct ncclComm* parent, int nc) {
  // Gather data from all ranks
  ncclResult_t ret = ncclSuccess;
  int *ringRecv = NULL, *ringSend = NULL, *ringPrev = NULL, *ringNext = NULL, *treeToParent = NULL, *treeToChild0 = NULL, *treeToChild1 = NULL, *nvlsHeads = NULL;
  int nranks = comm->nRanks;
  int nNodes = comm->nNodes;
  int nChannels = comm->nChannels;
  int minHeadNum = INT_MAX;
  int shared = parent && parent->nvlsSupport  && parent->shareResources;
  int maxChannels;
  int minNchannels, maxNchannels;
  NCCLCHECK(ncclCalloc(&ringRecv, nNodes*MAXCHANNELS));
  NCCLCHECKGOTO(ncclCalloc(&ringSend, nNodes*MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&ringPrev, nranks*MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&ringNext, nranks*MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&treeToParent, nNodes*MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&treeToChild0, nNodes*MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&treeToChild1, nNodes*MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&nvlsHeads, nNodes*MAXCHANNELS), ret, fail);

  // Alternate rings to avoid crossing rails
  if (graphs[NCCL_ALGO_RING]->crossNic == 2 && (nChannels % 2) == 0) {
    for (int r=0; r<comm->nRanks; r++) {
      if (comm->rankToNode[r] % 2 == 1) {
        // Exchange rings
        for (int c=0; c<nChannels; c+=2) {
          exchangeValues(allTopoRanks[r]->ringRecv+c, allTopoRanks[r]->ringRecv+(c^1));
          exchangeValues(allTopoRanks[r]->ringSend+c, allTopoRanks[r]->ringSend+(c^1));
          exchangeValues(allTopoRanks[r]->ringPrev+c, allTopoRanks[r]->ringPrev+(c^1));
          exchangeValues(allTopoRanks[r]->ringNext+c, allTopoRanks[r]->ringNext+(c^1));
        }
      }
    }
  }

  for (int c=0; c<nChannels;c++) {
    for (int n=0; n<nNodes; n++) {
      int r = firstRanks[n];
      ringRecv[c*nNodes+n] = allTopoRanks[r]->ringRecv[c];
      ringSend[c*nNodes+n] = allTopoRanks[r]->ringSend[c];
      treeToParent[c*nNodes+n] = allTopoRanks[r]->treeToParent[c];
      treeToChild0[c*nNodes+n] = allTopoRanks[r]->treeToChild0[c];
      treeToChild1[c*nNodes+n] = allTopoRanks[r]->treeToChild1[c];
    }
    for (int r=0; r<nranks; r++) {
      ringPrev[c*nranks+r] = allTopoRanks[r]->ringPrev[c];
      ringNext[c*nranks+r] = allTopoRanks[r]->ringNext[c];
    }
  }

  for (int n = 0; n < nNodes; n++) {
    int r = firstRanks[n];
    if (minHeadNum > allTopoRanks[r]->nvlsHeadNum)
      minHeadNum = allTopoRanks[r]->nvlsHeadNum;
  }

  for (int c = 0; c < minHeadNum; c++) {
    for (int n = 0; n < nNodes; n++) {
      int r = firstRanks[n];
      nvlsHeads[c * nNodes + n] = allTopoRanks[r]->nvlsHeads[c];
    }
  }

  // Connect rings and trees. This should also duplicate the channels.
  NCCLCHECK(connectRings(comm, ringRecv, ringSend, ringPrev, ringNext));

  // [RCCL] Connect rail-optimized trees
  if (comm->topo->useRailOptimizedTrees) {
    NCCLCHECK(connectRailOptimizedTrees(comm, treeToParent, treeToChild0, treeToChild1));
  } else {
    NCCLCHECK(connectTrees(comm, treeToParent, treeToChild0, treeToChild1, treePatterns));
  }

  // Dump graphviz-friendly trees
  if (rcclParamOutputTrees()) {
    int rank = comm->rank;
    char color[8][16] =
      {"red", "orange", "yellow", "yellowgreen", "green", "cyan", "deepskyblue", "violet"};

    for (int i = 0; i < comm->nChannels; i++) {
      INFO(NCCL_GRAPH, "[TREE] %d.%d [style=filled, fillcolor=%s]", i, rank, color[rank % comm->localRanks]);
      for (int j = 0; j < 3; j++) {
        if (comm->channels[i].tree.down[j] != -1) {
	  bool sameNode = (comm->rankToNode[rank] == comm->rankToNode[comm->channels[i].tree.down[j]]);
          INFO(NCCL_GRAPH, "[TREE] %d.%d->%d.%d [style=%s,width=10,color=%s,label=\"%s\"]",
	       i, rank, i, comm->channels[i].tree.down[j],
	       sameNode ? "solid" : "dashed",
	       sameNode ? "black" : color[rank % comm->localRanks],
	       sameNode ? ""  : (std::string("N") + std::to_string(graphs[NCCL_ALGO_TREE]->inter[i*2+1])).c_str());
        }
      }
    }
  }

  // Only use full MAXCHANNELS for gfx942 (MI300X) and gfx950
  maxChannels = (IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942") ||
                 IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950"))
                 ? std::min(comm->topo->nodes[GPU].nodes[0].gpu.cu, MAXCHANNELS) : 2*CHANNEL_LIMIT;

  if (graphs[NCCL_ALGO_RING]->nIntraChannels > 0 || comm->nNodes > 1) {
    maxChannels = std::min(64, maxChannels);
  }

  // Duplicate ringPrev/ringNext for ncclBuildRing
  if (nChannels <= maxChannels/2) memcpy(ringPrev+nChannels*nranks, ringPrev, nChannels*nranks*sizeof(int));
  if (nChannels <= maxChannels/2) memcpy(ringNext+nChannels*nranks, ringNext, nChannels*nranks*sizeof(int));

  // Get number of channels after duplication
  maxNchannels = std::min((int)ncclMaxNchannels(), maxChannels);
  nc = std::min(maxNchannels/comm->nChannels, nc);
  nc *= comm->nChannels;

  // Set ring prev/next for my rank
  for (int c=0; c<nChannels; c++) {
    struct ncclChannel* channel0 = comm->channels+c;
    struct ncclChannel* channel1 = channel0+nChannels;
    channel0->ring.prev = ringPrev[c*nranks+comm->rank];
    channel0->ring.next = ringNext[c*nranks+comm->rank];

    if (c + nChannels < MAXCHANNELS) {
      channel1->ring.prev = channel0->ring.prev;
      channel1->ring.next = channel0->ring.next;
    }
  }

  // Duplication should be complete now
  nChannels = comm->nChannels = std::min(maxChannels, (nChannels <= maxChannels/2) ? nChannels*2 : nChannels);

  // Setup CollNet
  if (comm->config.collnetEnable) {
    struct ncclTopoGraph* collNetChainGraph = graphs[NCCL_ALGO_COLLNET_CHAIN];
    // Add more channels to saturate intra-node bandwidth, except the 1 PPN case
    if (collNetChainGraph->bwIntra > collNetChainGraph->bwInter && comm->nRanks > comm->nNodes) {
      int collNetNchannels = std::min(maxChannels, nChannels+nChannels/2);
      nChannels = comm->nChannels = copyChannels(comm, nChannels, collNetNchannels, ringPrev, ringNext);
    }
    NCCLCHECKGOTO(connectCollNet(comm, graphs[NCCL_ALGO_COLLNET_DIRECT]), ret, fail);
  }

  // Use 4 compute channels per search channel to reach peak BW on <8 PPN
  if (comm->minCompCap >= 90 && comm->nNodes > 1 && graphs[NCCL_ALGO_RING]->bwIntra > 45.0 && 2*nChannels <= maxChannels) {
     nChannels = comm->nChannels = copyChannels(comm, nChannels, 2*nChannels, ringPrev, ringNext);
  }

  // Double the number of channels when using unpack networking (greater than 1 node)
  // We won't automatically double past 16 channels, users can specify 32 if they want
  if (comm->netDeviceType == NCCL_NET_DEVICE_UNPACK && comm->nNodes > 1 && nChannels < 16 && ncclParamUnpackDoubleNChannels()) {
     nChannels = comm->nChannels = copyChannels(comm, nChannels, 2*nChannels, ringPrev, ringNext);
  }

  minNchannels = ncclMinNchannels();
  if (comm->nNodes > 1) {
    minNchannels = std::min(64, minNchannels);
  }
  if (comm->nRanks < 8 && 64 < minNchannels) {
    minNchannels = 2;
    WARN("NCCL_MIN_NCHANNELS set by environment is ignored due to less than 8 GPUs.");
  }
  if (minNchannels > maxChannels) {
    minNchannels = 2;
    WARN("NCCL_MIN_NCHANNELS set by environment is ignored due to greater than max allowed %d channels.", maxChannels);
  }

  if (mscclEnabled() && (comm->topo->mscclEnabled || mscclForceEnabled())) {
    int mscclNumChannelsRequired = maxNchannels;
    mscclSchedulerInit(comm, &mscclNumChannelsRequired);
    if (comm->mscclCompatible) {
      minNchannels = std::max(minNchannels, mscclNumChannelsRequired);
    }
  }

  // Honor NCCL_MIN_NRINGS/NCCL_MAX_NRINGS.
  // We permit combining max, then min, to only use the first channels, then duplicate them.
  if (comm->sharedRes->owner != comm) {
    /* child comm #channels cannot exceed top parent #channels. */
    nChannels = comm->nChannels = std::min(std::min(std::min(ncclMaxNchannels(), nChannels), comm->config.maxCTAs), comm->sharedRes->tpNChannels);
    nChannels = comm->nChannels = copyChannels(comm, nChannels, std::min(std::max(minNchannels, std::max(nc, comm->config.minCTAs)), comm->sharedRes->tpNChannels), ringPrev, ringNext);
  } else {
    nChannels = comm->nChannels = std::min(std::min(ncclMaxNchannels(), nChannels), comm->config.maxCTAs);
    nChannels = comm->nChannels = copyChannels(comm, nChannels, std::max(minNchannels, std::max(nc, comm->config.minCTAs)), ringPrev, ringNext);
  }

  comm->collChannels = comm->nChannels;
#if CUDART_VERSION >= 12010
  // Support maximal channel usage for aggregation
  if (shared && comm->nvlsChannels > parent->nvlsResources->nChannels) {
    comm->nvlsChannels = parent->nvlsResources->nChannels;
  }
  if (comm->nChannels < comm->nvlsChannels) {
    nChannels = comm->nChannels = copyChannels(comm, comm->nChannels, comm->nvlsChannels, ringPrev, ringNext);
  }
  NCCLCHECKGOTO(connectNvls(comm, nvlsHeads, minHeadNum), ret, fail);
#endif
  if (shared && comm->nChannels > parent->sharedRes->tpNChannels) {
    nChannels = comm->nChannels = parent->sharedRes->tpNChannels;
    comm->collChannels = std::min(comm->collChannels, comm->nChannels);
  }

  // Create rings array and check all is fine
  NCCLCHECKGOTO(ncclBuildRings(nChannels, rings, comm->rank, comm->nRanks, ringPrev, ringNext), ret, fail);

exit:
  if (ringRecv) free(ringRecv);
  if (ringSend) free(ringSend);
  if (ringPrev) free(ringPrev);
  if (ringNext) free(ringNext);
  if (treeToParent) free(treeToParent);
  if (treeToChild0) free(treeToChild0);
  if (treeToChild1) free(treeToChild1);
  if (nvlsHeads) free(nvlsHeads);
  return ret;
fail:
  goto exit;
}
