/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"

#include <stdio.h>      
#include <stdlib.h>     
#include <stdint.h>    
#include <string.h> 

void dumpLine(int* values, int nranks, const char* prefix) {
  constexpr int line_length = 128;
  char line[line_length];
  int num_width = snprintf(nullptr, 0, "%d", nranks-1);  // safe as per "man snprintf"
  int n = snprintf(line, line_length, "%s", prefix);
  for (int i = 0; i < nranks && n < line_length-1; i++) {
    n += snprintf(line + n, line_length - n, " %*d", num_width, values[i]);
    // At this point n may be more than line_length-1, so don't use it
    // for indexing into "line".
  }
  if (n >= line_length) {
    // Sprintf wanted to write more than would fit in the buffer. Assume
    // line_length is at least 4 and replace the end with "..." to
    // indicate that it was truncated.
    snprintf(line+line_length-4, 4, "...");
  }
  INFO(NCCL_INIT, "%s", line);
}

ncclResult_t ncclBuildRings(int nrings, int* rings, int rank, int nranks, int* prev, int* next) {
  ncclResult_t ret = ncclSuccess;
  uint64_t* rankFound;
  int rankFoundSize = DIVUP(nranks, 64);
  NCCLCHECK(ncclCalloc(&rankFound, rankFoundSize));

  for (int r=0; r<nrings; r++) {
    char prefix[40];
    /*sprintf(prefix, "[%d] Channel %d Prev : ", rank, r);
    dumpLine(prev+r*nranks, nranks, prefix);
    sprintf(prefix, "[%d] Channel %d Next : ", rank, r);
    dumpLine(next+r*nranks, nranks, prefix);*/

    int current = rank;
    for (int i=0; i<nranks; i++) {
      rankFound[current/64] |= (1<<(current%64));
      rings[r*nranks+i] = current;
      current = next[r*nranks+current];
    }
    snprintf(prefix, sizeof(prefix), "Channel %02d/%02d :", r, nrings);
    if (rank == 0) dumpLine(rings+r*nranks, nranks, prefix);
    if (current != rank) {
      WARN("Error : ring %d does not loop back to start (%d != %d)", r, current, rank);
      ret = ncclInternalError;
      goto end;
    }
    // Check that all ranks are there
    for (int i=0; i<nranks; i++) {
      uint64_t bits = rankFound[i/64], mask = 1<<(i%64);
      // Fast check 64 ranks at a time
      if (mask == 1 && bits == 0xffffffffffffffff) { i += 63; continue; }
      if ((bits & mask) == 0) {
        WARN("Error : ring %d does not contain rank %d", r, i);
        ret = ncclInternalError;
        goto end;
      }
    }
    memset(rankFound, 0, rankFoundSize*sizeof(uint64_t));
  }
end:
  free(rankFound);
  return ret;
}

/**
 * rcclBuildRings: Functionally same as ncclBuildRings, Linearizes linked-list neighbor pointers into a rank array.
 * This function converts 'next' and 'prev' adjacency arrays into a flat list 
 * of ranks (a ring) for each communication channel.
 *
 * PRE-CONDITIONS & ASSUMPTIONS:
 * 1. Global Connectivity: It assumes that 'next' and 'prev' arrays represent 
 * a complete graph of all 'nranks' global participants.
 * 2. Array Sizing: 
 * - 'rings' must be allocated to at least (nrings * nranks * sizeof(int)).
 * - 'prev' and 'next' must be (nrings * nranks) in size.
 * 3. Identity: 'rank' must be the global rank of the local process, and 
 * 0 <= rank < nranks.
 * 4. Path Discovery: It assumes the caller has already performed topology search 
 * to populate 'next' and 'prev' such that they form a Hamiltonian cycle.
 *
 * DESCRIPTION:
 * - Loop-back: Ensures the ring returns to the starting rank after exactly 'nranks' steps.
 * - Full Coverage: Ensures no rank is skipped or duplicated (O(N) check).
 * - Bi-directional: Verifies that 'prev' and 'next' are perfect mirrors.
 * - O( nRings * nRanks ) Algorithmic complexity 
 */
ncclResult_t rcclBuildRings(int nrings, int* rings, int rank, int nranks, int* prev, int* next) {
  // Use a stack-allocated buffer for O(N) validation.
  // If nranks > 1024, consider using malloc/free
  if( (nrings < 0) || (rank < 0) ||  (nranks < 0) || ( rings == NULL) || (prev == NULL) || (next == NULL)) return ncclInvalidArgument;
  ncclResult_t res = ncclSuccess;
  uint8_t* found = (uint8_t*)malloc(nranks * sizeof(uint8_t));
  if (found == NULL) return ncclInternalError;

  for (int r = 0; r < nrings; r++) {
    int* current_ring = rings + (r * nranks);
    int* current_next = next + (r * nranks);
    int* current_prev = prev + (r * nranks);

    int current = rank;
    // Initilize to not found
    memset(found, 0, nranks * sizeof(uint8_t));

    for (int i = 0; i < nranks; i++) {
      // Safety: Check for out-of-bounds rank pointers
      if (current < 0 || current >= nranks) {
        WARN("Ring %d: Found invalid rank index %d", r, current);
        res = ncclInternalError;
        goto exit;
      }

      // Check for sub-loops/cycles before we finish (Safety check)
      if (found[current] == 1) {
        WARN("Ring %d: Unexpected sub-loop detected at rank %d", r, current);
        res = ncclInternalError;
        goto exit;
      }

      current_ring[i] = current;
      found[current] = 1;

      // --- The Consistency Check ---
      int next_rank = current_next[current];
      if (next_rank >= 0 && next_rank < nranks) {
        // Verify that if I think 'next_rank' is my next, 
        // 'next_rank' must think I am its previous.
        if (current_prev[next_rank] != current) {
          WARN("Ring %d: Asymmetric link! Rank %d -> %d, but %d -> %d", 
               r, current, next_rank, next_rank, current_prev[next_rank]);
          res = ncclInternalError;
          goto exit;
        }
      } else {
        WARN("Ring %d: Rank %d pointed to invalid next_rank %d", r, current, next_rank);
        res = ncclInternalError;
        goto exit;
      }

      current = next_rank;
    }

    // Assumptions check
    if (rank == 0) {
      char prefix[40];
      snprintf(prefix, sizeof(prefix), "Channel %02d/%02d :", r, nrings);
      dumpLine(rings + r * nranks, nranks, prefix);
    }

    // 1. Must close the loop
    if (current != rank) {
      WARN("Ring %d: Failed to loop back. Ended at %d instead of %d", r, current, rank);
      res = ncclInternalError;
      goto exit;
    }

    // 2. Must visit every rank (already partially checked by sub-loop check)
    for (int i = 0; i < nranks; i++) {
      if (found[i] == 0) {
        WARN("Ring %d: Incomplete. Rank %d is missing.", r, i);
        res = ncclInternalError;
        goto exit;
      }
    }
  }
exit:
  if(found) free(found);
  return res;
}
