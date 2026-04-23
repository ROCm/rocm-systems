/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"

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
  for (int r=0; r<nrings; r++) {
    char prefix[40];
    /*sprintf(prefix, "[%d] Channel %d Prev : ", rank, r);
    dumpLine(prev+r*nranks, nranks, prefix);
    sprintf(prefix, "[%d] Channel %d Next : ", rank, r);
    dumpLine(next+r*nranks, nranks, prefix);*/

    int current = rank;
    for (int i=0; i<nranks; i++) {
      rings[r*nranks+i] = current;
      current = next[r*nranks+current];
    }
    snprintf(prefix, sizeof(prefix), "Channel %02d/%02d :", r, nrings);
    if (rank == 0) dumpLine(rings+r*nranks, nranks, prefix);
    if (current != rank) {
      WARN("Error : ring %d does not loop back to start (%d != %d)", r, current, rank);
      return ncclInternalError;
    }
    // Check that all ranks are there
    for (int i=0; i<nranks; i++) {
      int found = 0;
      for (int j=0; j<nranks; j++) {
        if (rings[r*nranks+j] == i) {
          found = 1;
          break;
        }
      }
      if (found == 0) {
        WARN("Error : ring %d does not contain rank %d", r, i);
        return ncclInternalError;
      }
    }
  }
  return ncclSuccess;
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
  // Use a bitmask/flag array for O(N) validation instead of O(N^2)
  std::vector<char> found(nranks, 0);

  for (int r = 0; r < nrings; r++) {
    int* current_ring = rings + (r * nranks);
    int* current_next = next + (r * nranks);
    int* current_prev = prev + (r * nranks);

    int current = rank;
    std::fill(found.begin(), found.end(), 0);

    for (int i = 0; i < nranks; i++) {
      // Safety: Check for out-of-bounds rank pointers
      if (current < 0 || current >= nranks) {
        WARN("Ring %d: Found invalid rank index %d", r, current);
        return ncclInternalError;
      }

      current_ring[i] = current;
      found[current] = 1;

      // --- The Consistency Check (Where current_prev is used) ---
      int next_rank = current_next[current];
      if (next_rank >= 0 && next_rank < nranks) {
        if (current_prev[next_rank] != current) {
          WARN("Ring %d: Asymmetric link detected! Rank %d -> %d, but %d -> %d", 
               r, current, next_rank, next_rank, current_prev[next_rank]);
          return ncclInternalError;
        }
      }

      current = next_rank;
    }

    if (rank == 0) {
      char prefix[40];
      snprintf(prefix, sizeof(prefix), "Channel %02d/%02d :", r, nrings);
      dumpLine(rings+r*nranks, nranks, prefix);
    }

    // Assumption: The path must close a perfect circle
    if (current != rank) {
      WARN("Ring %d: Failed to loop back. Ended at %d instead of %d", r, current, rank);
      return ncclInternalError;
    }

    // Assumption: Every single rank in the communicator must be present in the ring
    for (int i = 0; i < nranks; i++) {
      if (found[i] == 0) {
        WARN("Ring %d: Incomplete. Rank %d is missing from the ring.", r, i);
        return ncclInternalError;
      }
    }
  }
  return ncclSuccess;
}
