/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * [RCCL] DIRECT_A2A mesh graph.
 *
 * Topology graph for the NCCL_ALGO_DIRECT_A2A allreduce: a full mesh where
 * every rank reaches every other rank over the net in a single hop (e.g.
 * 4 nodes fully interconnected over USB4/Thunderbolt with the odl_tb5 net
 * plugin). Unlike ring/tree graphs there is no path search: the mesh is
 * implicit, every rank's peer list is "all other ranks" (filled per channel
 * in setupChannel(), see src/init.cc).
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "graph.h"
#include "topo.h"
#include "debug.h"

ncclResult_t ncclTopoComputeMesh(struct ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  // NOTE: system->nodes[GPU].count is the number of *local* GPUs (1 on STX
  // Halo), not the communicator size. system->nRanks is the global rank count.
  int nranks = system->nRanks;
  int nNets = system->nodes[NET].count;

  // DIRECT_A2A requires:
  //  - a multi-node run (single node is served by IPC/DDA paths),
  //  - net connectivity (nNets > 0),
  //  - nranks <= NCCL_MAX_MESH_PEERS+1 (device kernel fan bound).
  //
  // TODO(cluster): verify *full-mesh* reachability instead of just "a NET
  // device exists" — e.g. from the path tables or from the odl device file —
  // so that partially-wired clusters fall back to RING instead of failing
  // at connect time.
  if (system->nHosts <= 1 || nNets == 0 || nranks > NCCL_MAX_MESH_PEERS + 1) {
    graph->nChannels = 0;
    INFO(NCCL_GRAPH, "MESH : not applicable (nHosts=%d nNets=%d nranks=%d), DIRECT_A2A disabled",
         system->nHosts, nNets, nranks);
    return ncclSuccess;
  }

  // Two channels for large-message bandwidth: the fan-out writes and proxy
  // progress of the 3 peers are split across two channel thread groups.
  graph->nChannels = 2;
  graph->sameChannels = 1;
  graph->nHops = 1;
  graph->typeIntra = PATH_NET; // 1 GPU per node: there is no intra-node path
  graph->typeInter = PATH_NET;

  // Take bandwidth/latency from the first NET node (odl_tb5 reports
  // speed=20000 Mbps ~= 2.5 GB/s and latency=20 us).
  struct ncclTopoNode* net = system->nodes[NET].nodes;
  graph->bwInter = net->net.bw;
  graph->bwIntra = net->net.bw;
  graph->latencyInter = net->net.latency;

  // Net device used for every mesh connector (ncclTopoGetNetDev reads
  // inter[channel*2+{0,1}] when a graph is attached to the transport setup).
  // With the odl_tb5 plugin's virtual single device this is always dev 0.
  for (int c = 0; c < graph->nChannels; c++) {
    graph->inter[c*2] = net->id;
    graph->inter[c*2+1] = net->id;
  }

  // Mesh membership for channel 0: all ranks in ascending order. The MESH
  // preset does not consume intra[] (peer lists are built in setupChannel);
  // this is only so ncclTopoPrintGraph / graph dumps show something sane.
  for (int c = 0; c < graph->nChannels; c++) {
    for (int r = 0; r < nranks && r < NCCL_TOPO_MAX_NODES; r++) {
      graph->intra[c*nranks + r] = r;
    }
  }

  INFO(NCCL_GRAPH, "MESH : full mesh over net, %d ranks, bw=%.2f GB/s, lat=%.1f us",
       nranks, graph->bwInter, graph->latencyInter);
  return ncclSuccess;
}
