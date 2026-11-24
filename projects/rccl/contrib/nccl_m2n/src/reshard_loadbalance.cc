/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#include "reshard_types.h"
#include "reshard_internal.h"

int getNodeOfDestRep(const ncclReshardRepLoadBalancer* lb, int dstRepIdx) {
  int dstRank = lb->dstRepStartRank + dstRepIdx * lb->dstRepStride;
  /* dstNodeAnchor==0 reproduces the absolute mapping used by the non-split
   * path.  The split path anchors at the dst mesh origin so a gen domain whose
   * origin is not a multiple of dstGpusPerDomain (asymmetric-PP) is not split
   * into phantom nodes. */
  return (dstRank - lb->dstNodeAnchor) / lb->dstGpusPerDomain;
}

/* Split-comm strided injection helper: the NVL-domain offset (0-based,
 * relative to the first destination domain) that a given dst rep sits in. */
static int getDomainOffsetOfDestRep(const ncclReshardRepLoadBalancer* lb, int dstRepIdx) {
  return getNodeOfDestRep(lb, dstRepIdx) - getNodeOfDestRep(lb, 0);
}

/* Split-comm strided injection helper: the index of the FIRST dst rep that
 * lands in NVL-domain offset `offset`.  Returns -1 if none. */
static int getFirstDestRepAtDomainOffset(const ncclReshardRepLoadBalancer* lb, int offset) {
  for (int rep = 0; rep < lb->dstRepCount; rep++) {
    if (getDomainOffsetOfDestRep(lb, rep) == offset) {
      return rep;
    }
  }
  return -1;
}

int getNumDestNodes(const ncclReshardRepLoadBalancer* lb) {
  if (lb->dstRepCount <= 0 || lb->dstGpusPerDomain <= 0) return 1;
  int firstNode = getNodeOfDestRep(lb, 0);
  int lastNode = getNodeOfDestRep(lb, lb->dstRepCount - 1);
  return lastNode - firstNode + 1;
}

void getDestRepsOnNode(const ncclReshardRepLoadBalancer* lb, int targetNode, int* repStart, int* repEnd) {
  *repStart = -1;
  *repEnd = -1;
  for (int r = 0; r < lb->dstRepCount; r++) {
    if (getNodeOfDestRep(lb, r) == targetNode) {
      if (*repStart == -1) *repStart = r;
      *repEnd = r + 1;
    }
  }
  if (*repStart == -1) {
    *repStart = 0;
    *repEnd = 0;
  }
}

void getDestRepsOnNodeRange(const ncclReshardRepLoadBalancer* lb, int firstNode, int lastNode, int* repStart, int* repEnd) {
  *repStart = -1;
  *repEnd = -1;
  for (int r = 0; r < lb->dstRepCount; r++) {
    int node = getNodeOfDestRep(lb, r);
    if (node >= firstNode && node <= lastNode) {
      if (*repStart == -1) *repStart = r;
      *repEnd = r + 1;
    }
  }
  if (*repStart == -1) {
    *repStart = 0;
    *repEnd = 0;
  }
}

void getTargetRepRange(const ncclReshardRepLoadBalancer* lb, int srcRepIdx, int* repStart, int* repEnd) {
  if (lb->mode == RESHARD_LB_NODE_AWARE) {
    /* Split-comm strided injection: source rep i injects the replica that
     * heads at NVL-domain offset i*domainsPerRep (one of the first
     * numInjectionReps = K/domainsPerRep injection replicas) and rings to
     * replicas i+numInjectionReps, i+2*numInjectionReps, ...  The returned
     * range starts at the first rep of the injection replica's lead domain
     * and spans to the end of the rep list; downstream consumers in
     * prepareReshardParams filter by exact node (and step by K for the
     * ring), so this superset is correct for both source and dest.
     *
     * domainsPerRep == 1 (reps/domain >= 1) reduces to the original
     * behavior: numInjectionReps == K and the injection offset is just
     * srcRepIdx. */
    if (lb->strided) {
      int dpr = (lb->domainsPerRep > 0) ? lb->domainsPerRep : 1;
      int numInjectionReps = (dpr > 0) ? lb->numInjectionDomains / dpr : lb->numInjectionDomains;
      if (numInjectionReps <= 0 || srcRepIdx >= numInjectionReps) {
        *repStart = 0;
        *repEnd = 0;
        return;
      }
      int firstRep = getFirstDestRepAtDomainOffset(lb, srcRepIdx * dpr);
      if (firstRep < 0) {
        *repStart = 0;
        *repEnd = 0;
        return;
      }
      *repStart = firstRep;
      *repEnd = lb->dstRepCount;
      return;
    }
    int numDstNodes = getNumDestNodes(lb);
    int firstDstNode = getNodeOfDestRep(lb, 0);
    if (lb->srcRepCount >= numDstNodes) {
      if (srcRepIdx >= numDstNodes) {
        *repStart = 0;
        *repEnd = 0;
        return;
      }
      int targetNode = firstDstNode + srcRepIdx;
      getDestRepsOnNode(lb, targetNode, repStart, repEnd);
    } else {
      int nodesPerSrc = numDstNodes / lb->srcRepCount;
      int extraNodes = numDstNodes % lb->srcRepCount;
      int firstNodeOffset, lastNodeOffset;
      if (srcRepIdx < extraNodes) {
        firstNodeOffset = srcRepIdx * (nodesPerSrc + 1);
        lastNodeOffset = firstNodeOffset + nodesPerSrc;
      } else {
        firstNodeOffset = extraNodes * (nodesPerSrc + 1) + (srcRepIdx - extraNodes) * nodesPerSrc;
        lastNodeOffset = firstNodeOffset + nodesPerSrc - 1;
      }
      getDestRepsOnNodeRange(lb, firstDstNode + firstNodeOffset, firstDstNode + lastNodeOffset, repStart, repEnd);
    }
    return;
  }
  // UNIFORM mode
  if (lb->srcRepCount > lb->dstRepCount) {
    int perDst = lb->srcRepCount / lb->dstRepCount;
    int extra = lb->srcRepCount % lb->dstRepCount;
    int threshold = extra * (perDst + 1);
    int dstRep;
    bool isSender;
    if (srcRepIdx < threshold) {
      int groupSize = perDst + 1;
      dstRep = srcRepIdx / groupSize;
      isSender = (srcRepIdx % groupSize == 0);
    } else {
      int adjustedIdx = srcRepIdx - threshold;
      dstRep = extra + adjustedIdx / perDst;
      isSender = (adjustedIdx % perDst == 0);
    }
    if (isSender && dstRep < lb->dstRepCount) {
      *repStart = dstRep;
      *repEnd = dstRep + 1;
    } else {
      *repStart = 0;
      *repEnd = 0;
    }
  } else if (lb->srcRepCount == lb->dstRepCount) {
    *repStart = srcRepIdx;
    *repEnd = srcRepIdx + 1;
  } else {
    int perSrc = lb->dstRepCount / lb->srcRepCount;
    int extra = lb->dstRepCount % lb->srcRepCount;
    if (srcRepIdx < extra) {
      *repStart = srcRepIdx * (perSrc + 1);
      *repEnd = *repStart + perSrc + 1;
    } else {
      *repStart = extra * (perSrc + 1) + (srcRepIdx - extra) * perSrc;
      *repEnd = *repStart + perSrc;
    }
  }
}

int getSourceRepForDest(const ncclReshardRepLoadBalancer* lb, int dstRepIdx) {
  if (lb->mode == RESHARD_LB_NODE_AWARE) {
    /* Split-comm strided injection: a dst rep whose replica heads at
     * NVL-domain offset o is served by source rep (o/domainsPerRep) %
     * numInjectionReps -- the injection replica that heads the ring chain
     * {r, r+numInjectionReps, ...} this replica sits on.  domainsPerRep == 1
     * reduces to o % numInjectionDomains (the original expression). */
    if (lb->strided && lb->numInjectionDomains > 0) {
      int dpr = (lb->domainsPerRep > 0) ? lb->domainsPerRep : 1;
      int numInjectionReps = lb->numInjectionDomains / dpr;
      if (numInjectionReps <= 0) numInjectionReps = 1;
      return (getDomainOffsetOfDestRep(lb, dstRepIdx) / dpr) % numInjectionReps;
    }
    int numDstNodes = getNumDestNodes(lb);
    int firstDstNode = getNodeOfDestRep(lb, 0);
    int myNode = getNodeOfDestRep(lb, dstRepIdx);
    int nodeOffset = myNode - firstDstNode;
    if (lb->srcRepCount >= numDstNodes) return nodeOffset;
    int nodesPerSrc = numDstNodes / lb->srcRepCount;
    int extraNodes = numDstNodes % lb->srcRepCount;
    int threshold = extraNodes * (nodesPerSrc + 1);
    if (nodeOffset < threshold) return nodeOffset / (nodesPerSrc + 1);
    return extraNodes + (nodeOffset - threshold) / nodesPerSrc;
  }
  // UNIFORM mode
  if (lb->srcRepCount > lb->dstRepCount) {
    int perDst = lb->srcRepCount / lb->dstRepCount;
    int extra = lb->srcRepCount % lb->dstRepCount;
    if (dstRepIdx < extra) return dstRepIdx * (perDst + 1);
    return extra * (perDst + 1) + (dstRepIdx - extra) * perDst;
  }
  if (lb->srcRepCount == lb->dstRepCount) return dstRepIdx;
  int perSrc = lb->dstRepCount / lb->srcRepCount;
  int extra = lb->dstRepCount % lb->srcRepCount;
  int threshold = extra * (perSrc + 1);
  if (dstRepIdx < threshold) return dstRepIdx / (perSrc + 1);
  return extra + (dstRepIdx - threshold) / perSrc;
}
