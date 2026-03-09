/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"

#define RANK_TO_INDEX(r) (rank > root ? rank-1 : rank)

/* Btree which alternates leaves and nodes.
 * Assumes root is 0, which conveniently builds a tree on powers of two,
 * (because we have pow2-1 ranks) which lets us manipulate bits.
 * Find first non-zero bit, then :
 * Find the parent :
 *   xx01[0] -> xx10[0] (1,5,9 below) or xx00[0] if xx10[0] is out of bounds (13 below)
 *   xx11[0] -> xx10[0] (3,7,11 below)
 * Find the children :
 *   xx10[0] -> xx01[0] (2,4,6,8,10,12) or -1 (1,3,5,7,9,11,13)
 *   xx10[0] -> xx11[0] (2,4,6,8,10) or xx101[0] (12) or xx1001[0] ... or -1 (1,3,5,7,9,11,13)
 *
 * Illustration :
 * 0---------------8
 *          ______/ \______
 *         4               12
 *       /   \            /  \
 *     2       6       10     \
 *    / \     / \     /  \     \
 *   1   3   5   7   9   11    13
 */
ncclResult_t ncclGetBtree(int nranks, int rank, int* u, int* d0, int* d1, int* parentChildType) {
  int up, down0, down1;
  int bit;
  for (bit=1; bit<nranks; bit<<=1) {
    if (bit & rank) break;
  }

  if (rank == 0) {
    *u = -1;
    *d0 = -1;
    // Child rank is > 0 so it has to be our child 1, not 0.
    *d1 = nranks > 1 ? bit >> 1 : -1;
    return ncclSuccess;
  }

  up = (rank ^ bit) | (bit << 1);
  // if smaller than the parent, we are his first child, otherwise we're his second
  if (up >= nranks) up = (rank ^ bit);
  *parentChildType = (rank < up) ? 0 : 1;
  *u = up;

  int lowbit = bit >> 1;
  // down0 is always within bounds
  down0 = lowbit == 0 ? -1 : rank-lowbit;

  down1 = lowbit == 0 ? -1 : rank+lowbit;
  // Make sure down1 is within bounds
  while (down1 >= nranks) {
    down1 = lowbit == 0 ? -1 : rank+lowbit;
    lowbit >>= 1;
  }
  *d0 = down0; *d1 = down1;

  return ncclSuccess;
}

/* Build a double binary tree. Take the previous tree for the first tree.
 * For the second tree, we use a mirror tree (if nranks is even)
 *
 * 0---------------8                   3----------------11
 *          ______/ \                 / \______
 *         4         \               /         7
 *       /   \        \             /        /   \
 *     2       6       10         1        5      9
 *    / \     / \     /  \       / \      / \    / \
 *   1   3   5   7   9   11     0   2    4   6  8   10
 *
 * or shift it by one rank (if nranks is odd).
 *
 * 0---------------8            1---------------9
 *          ______/ \______              ______/ \______
 *         4               12           5                0
 *       /   \            /           /   \            /
 *     2       6       10           3       7       11
 *    / \     / \     /  \         / \     / \     /  \
 *   1   3   5   7   9   11       2   4   6   8  10   12
 */
ncclResult_t ncclGetDtree(int nranks, int rank, int* s0, int* d0_0, int* d0_1, int* parentChildType0, int* s1, int* d1_0, int* d1_1, int* parentChildType1) {
  // First tree ... use a btree
  ncclGetBtree(nranks, rank, s0, d0_0, d0_1, parentChildType0);
  // Second tree ... mirror or shift
  if (nranks % 2 == 1) {
    // shift
    int shiftrank = (rank-1+nranks) % nranks;
    int u, d0, d1;
    ncclGetBtree(nranks, shiftrank, &u, &d0, &d1, parentChildType1);
    *s1 = u == -1 ? -1 : (u+1) % nranks;
    *d1_0 = d0 == -1 ? -1 : (d0+1) % nranks;
    *d1_1 = d1 == -1 ? -1 : (d1+1) % nranks;
  } else {
    // mirror
    int u, d0, d1;
    ncclGetBtree(nranks, nranks-1-rank, &u, &d0, &d1, parentChildType1);
    *s1 = u == -1 ? -1 : nranks-1-u;
    *d1_0 = d0 == -1 ? -1 : nranks-1-d0;
    *d1_1 = d1 == -1 ? -1 : nranks-1-d1;
  }
  return ncclSuccess;
}

/* Compact tree algorithms optimized for sorted domain ordering.
 *
 * When nodes are sorted by domain (e.g., ranks 0-4 in domain A, 5-7 in domain B),
 * the standard binary tree creates cross-domain hops along the spine, adding
 * sequential latency. This optimization uses "half-interleave" remapping to
 * build a tree where:
 *   - The tree spine stays in the first half of ranks (typically same domain)
 *   - Cross-domain hops are pushed to leaf level where they execute in parallel
 *
 * This works for ANY domain distribution (5+3, 7+1, 6+1+1, etc.) without needing
 * to know domain boundaries.
 *
 * Example with 8 nodes:
 *   Standard btree:        Compact tree (sorted-optimized):
 *        0                        0
 *        |                        |
 *        4                        2
 *       / \                      / \
 *      2   6                    1   3
 *     / \ / \                  / \ / \
 *    1  3 5  7                4  5 6  7
 *
 * For domain split [0-3]=A, [4-7]=B:
 *   - Standard: spine 0→4 crosses domains immediately
 *   - Compact: spine 0→2→1 stays in domain A, cross-domain only at leaves
 */

// Remap rank using half-interleave: first half to even positions, second half to odd
static int remapRankForCompactTree(int rank, int nranks) {
  if (nranks <= 1) return rank;

  int half = nranks / 2;
  if (rank < half) {
    // First half of sorted ranks -> even positions in remapped space
    return rank * 2;
  } else {
    // Second half of sorted ranks -> odd positions in remapped space
    return (rank - half) * 2 + 1;
  }
}

// Inverse remap: even positions to first half, odd positions to second half
static int inverseRemapRankForCompactTree(int remappedRank, int nranks) {
  if (nranks <= 1) return remappedRank;

  int half = nranks / 2;
  if (remappedRank % 2 == 0) {
    // Even positions -> first half of sorted ranks
    return remappedRank / 2;
  } else {
    // Odd positions -> second half of sorted ranks
    return half + remappedRank / 2;
  }
}

// Compact Btree optimized for sorted domain ordering
ncclResult_t ncclGetBtreeCompact(int nranks, int rank, int* u, int* d0, int* d1,
                                  int* parentChildType) {
  // Remap to half-interleaved space
  int remappedRank = remapRankForCompactTree(rank, nranks);

  // Get tree structure in remapped space
  int remappedU, remappedD0, remappedD1;
  ncclGetBtree(nranks, remappedRank, &remappedU, &remappedD0, &remappedD1, parentChildType);

  // Map connections back to sorted ordering
  *u = remappedU == -1 ? -1 : inverseRemapRankForCompactTree(remappedU, nranks);
  *d0 = remappedD0 == -1 ? -1 : inverseRemapRankForCompactTree(remappedD0, nranks);
  *d1 = remappedD1 == -1 ? -1 : inverseRemapRankForCompactTree(remappedD1, nranks);

  return ncclSuccess;
}

// Compact Dtree optimized for sorted domain ordering
ncclResult_t ncclGetDtreeCompact(int nranks, int rank, int* s0, int* d0_0, int* d0_1,
                                  int* parentChildType0, int* s1, int* d1_0, int* d1_1,
                                  int* parentChildType1) {
  // Remap to half-interleaved space
  int remappedRank = remapRankForCompactTree(rank, nranks);

  // Get double tree structure in remapped space
  int remappedS0, remappedD0_0, remappedD0_1;
  int remappedS1, remappedD1_0, remappedD1_1;
  ncclGetDtree(nranks, remappedRank,
               &remappedS0, &remappedD0_0, &remappedD0_1, parentChildType0,
               &remappedS1, &remappedD1_0, &remappedD1_1, parentChildType1);

  // Map connections back to sorted ordering
  *s0 = remappedS0 == -1 ? -1 : inverseRemapRankForCompactTree(remappedS0, nranks);
  *d0_0 = remappedD0_0 == -1 ? -1 : inverseRemapRankForCompactTree(remappedD0_0, nranks);
  *d0_1 = remappedD0_1 == -1 ? -1 : inverseRemapRankForCompactTree(remappedD0_1, nranks);
  *s1 = remappedS1 == -1 ? -1 : inverseRemapRankForCompactTree(remappedS1, nranks);
  *d1_0 = remappedD1_0 == -1 ? -1 : inverseRemapRankForCompactTree(remappedD1_0, nranks);
  *d1_1 = remappedD1_1 == -1 ? -1 : inverseRemapRankForCompactTree(remappedD1_1, nranks);

  return ncclSuccess;
}
