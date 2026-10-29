/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * Tensor Reshard — Split-Comm RING Param Builder
 *
 * Translates a parent-comm-relative ncclReshardParams (as built by
 * prepareReshardParams, including any PACK contiguous-plan rewrite)
 * into a ncclReshardParamsSplit whose peer ranks are expressed in the
 * sub-comm each operation runs on:
 *
 *   - Source -> first-domain-leader puts ride commA, so the target
 *     leader rank and the wait signal of first-domain dests are
 *     commA-relative.
 *   - The cross-NVL ring forward + LSA fan-out ride commB, so ringNext
 *     and followers are commB-relative and use commB-relative signal
 *     bases keyed by srcShardIdx (consistent across every ring hop
 *     because the same rep-slot handles the same source subset on every
 *     generator NVL domain).
 *
 * This is a pure post-process so the heavy mesh/load-balance logic in
 * prepareReshardParams is reused rather than duplicated.
 ************************************************************************/

#include <cstring>

#include "reshard_types.h"
#include "reshard_internal.h"
#include "reshard_split.h"

void buildSplitReshardParams(const ncclReshardParams* base, const ReshardSplitComms* sc, int numCtas,
                             ncclWindow_t windowA, ncclWindow_t windowB, ncclReshardParamsSplit* out) {
  memset(out, 0, sizeof(*out));

  const int srcSize = sc->srcMeshSize;
  const int dstStart = sc->dstStartRank;
  const int lsaSize = sc->lsaSize;
  const int ndims = base->ndims;

  out->windowA = windowA;
  out->windowB = windowB;

  for (int d = 0; d < ndims; d++) {
    out->srcDims[d] = base->srcDims[d];
    out->dstDims[d] = base->dstDims[d];
    out->srcStrides[d] = base->srcStrides[d];
    out->dstStrides[d] = base->dstStrides[d];
  }
  out->ndims = ndims;
  out->srcShardLayout = base->srcShardLayout;
  out->dstShardLayout = base->dstShardLayout;
  out->srcShardCount = base->srcShardCount;
  out->dstShardCount = base->dstShardCount;

  out->isSource = base->isSource;
  out->isDest = base->isDest;
  out->mySrcShardIdx = base->mySrcShardIdx;
  out->myDstShardIdx = base->myDstShardIdx;
  out->mySrcRepIdx = base->mySrcRepIdx;
  out->myDstRepIdx = base->myDstRepIdx;

  out->myRankInA = sc->rankInA;
  out->myRankInB = sc->rankInB;

  out->elementsPerChunk = base->elementsPerChunk;
  out->chunkSizeBytes = base->chunkSizeBytes;
  out->totalCtas = base->totalCtas;

  /* A dest receives directly from the trainer over commA only when it is the
   * head of its ring chain (a source injects it directly).  Being positionally
   * inside the first K injection NVL domains is necessary but NOT sufficient:
   * with strided/mixed-domain layouts a dest can sit in the commA gen block yet
   * still be reached by the commB ring from an upstream leader.  Flagging such a
   * rank commA-direct makes it wait on a commA signal no source produces ->
   * deadlock.  base->destInjectedDirectly is the source-consistent condition
   * (leader && myNode == firstRepNode), so gate on it as well. */
  const int commAGenRanks = sc->numInjectionDomains * lsaSize;
  out->destRecvViaCommA = base->isDest && (sc->parentRank < dstStart + commAGenRanks) && base->destInjectedDirectly;

  /* ---- Source side: targets (leader ranks -> commA-relative). ------ */
  out->numTargets = base->numTargets;
  for (int t = 0; t < base->numTargets; t++) {
    out->targets[t] = base->targets[t];
    /* The leader is a first-gen-domain dst rank; its commA rank is the
     * src block (size srcSize) followed by the first-domain gen block
     * starting at dstStart. */
    const int leaderParent = base->targets[t].dstWorldRank;
    out->targets[t].dstWorldRank = srcSize + (leaderParent - dstStart);
  }

  /* ---- Dest side: sources (wait + ring-forward signal bases). ------ */
  out->numSources = base->numSources;
  for (int s = 0; s < base->numSources; s++) {
    out->sources[s] = base->sources[s];

    const int srcShardIdx = base->sources[s].srcShardIdx;
    const unsigned int forwardBaseB = (unsigned int)(srcShardIdx * numCtas);
    out->sourceSignalBaseB[s] = forwardBaseB;

    if (out->destRecvViaCommA) {
      /* prepareReshardParams already expresses signalBase relative to the
       * source mesh. commA orders source ranks first, so the value is also
       * the commA-relative signal base even when the source mesh starts at a
       * nonzero parent rank. */
      out->sources[s].signalBase = base->sources[s].signalBase;
    } else {
      /* Wait on commB: same index the upstream leader forwards to. */
      out->sources[s].signalBase = forwardBaseB;
    }
  }

  /* ---- Replication peers -> commB-relative. ----------------------- */
  out->numLocalFollowers = base->numLocalFollowers;
  for (int f = 0; f < base->numLocalFollowers; f++) {
    out->localFollowerRanksB[f] = base->localFollowerWorldRanks[f] - dstStart;
    out->localFollowerWindowOffsets[f] = base->localFollowerWindowOffsets[f];
  }
  out->ringNextRankB = (base->ringNextWorldRank >= 0) ? (base->ringNextWorldRank - dstStart) : -1;
  out->isRingLast = base->isRingLast;

  out->localRepIdx = base->localRepIdx;
  out->numLocalReps = base->numLocalReps;
  out->isLeaderForSources = base->isLeaderForSources;

  out->myWindowOffset = base->myWindowOffset;
  out->ringNextWindowOffset = base->ringNextWindowOffset;

  /* Per-rank split-plan summary (TRACE; one line per rank per param, so
   * kept off the default INFO path) so the strided ring/injection neighbors
   * can be validated against the probed topology:
   *   - source ranks: the commA-relative leader(s) they inject into;
   *   - dest leaders: whether they receive over commA (injection domain)
   *     or the commB ring, plus their commB ring-next rank and LSA
   *     follower count.  K=numInjectionDomains, lsaSize from commB. */
  RESHARD_TRACE(sc->parentRank,
                "split-plan: K=%d lsa=%d strided=%d | src=%d dst=%d srcRep=%d dstRep=%d | rankInA=%d rankInB=%d "
                "recvViaA=%d numTargets=%d leaderA0=%d numSrc=%d ringNextB=%d isRingLast=%d numFollowers=%d",
                sc->numInjectionDomains, sc->lsaSize, (int)sc->strided, (int)out->isSource, (int)out->isDest,
                out->mySrcRepIdx, out->myDstRepIdx, out->myRankInA, out->myRankInB, (int)out->destRecvViaCommA,
                out->numTargets, (out->numTargets > 0) ? out->targets[0].dstWorldRank : -1, out->numSources,
                out->ringNextRankB, (int)out->isRingLast, out->numLocalFollowers);
}
