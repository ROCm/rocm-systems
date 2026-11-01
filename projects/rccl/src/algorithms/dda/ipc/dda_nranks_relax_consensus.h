/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_ALGORITHMS_DDA_IPC_NRANKS_RELAX_CONSENSUS_H_
#define RCCL_ALGORITHMS_DDA_IPC_NRANKS_RELAX_CONSENSUS_H_

#include "nccl.h"
#include <functional>

// RCCL_DDA_NRANKS_RELAX is read per process (RCCL_PARAM), so nothing stops it
// from being set on only some ranks of a communicator. Before this knob
// existed, every rank derived the DDA IPC comm-init decision from comm->nRanks
// alone, which is identical on every rank by construction; ncclDdaIpcCommInit()
// now also depends on this per-process value, so a mismatched setting makes
// some ranks enter its bootstrap allgather (ipc_mem_handler.cc exchangeMemPtrs)
// while the rest return early -- the entering ranks then block forever waiting
// on peers that never arrive.
//
// Mirrors rcclCheckRomeTopoModelIdxConsensus (rome_topo_consensus.h): callers
// gather every rank's value into an array via the bootstrap allgather that
// already runs unconditionally for every communicator (allGather3Data in
// init.cc), then call this to verify agreement from that already-gathered
// data -- no new communication, so the check itself cannot introduce a hang.
// Callers supply a getter so this stays independent of ncclComm / allGatherInfo.
ncclResult_t ncclCheckDdaNranksRelaxConsensus(int nranks, std::function<bool(int /*rank*/)> getDdaNranksRelax,
                                              std::function<const char*(int /*rank*/)> getHostname);

#endif
