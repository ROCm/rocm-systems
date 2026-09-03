
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
#include "nccl.h"
#include "debug.h"
#include "rccl_graph_gen.h"

#include <stdio.h>      // For NULL and
#include <stdlib.h>     // For malloc(), calloc(), and free()
#include <stdint.h>     // For uint8_t and other fixed-width types
#include <string.h>     // For memset()
#include <limits.h>     // For INT_MAX
#include <vector>

RCCL_PARAM(IntraGraphGen, "INTRA_GRAPH_GEN", 0);
RCCL_PARAM(InterGraphGen, "INTER_GRAPH_GEN", 0);

/**
 * Permutes `input` in-place according to `permutation` using gather semantics.
 *
 * After this function returns:
 *
 *     input[i] == old_input[permutation[i]]
 *
 * Preconditions:
 *   - `length >= 0`.
 *   - If `length > 0`, `input` and `permutation` must point to arrays
 *     containing at least `length` elements.
 *   - `permutation` must initially contain a valid permutation of
 *     the integers [0, length). In other words:
 *       * 0 <= permutation[i] < length for every i.
 *       * Every value in [0, length) occurs exactly once.
 *   - `input` must be writable.
 *   - `input` and `permutation` must not overlap.
 *
 * Notes:
 *   - The permutation array is temporarily modified internally to mark
 *     processed indices, but is restored to its original contents before
 *     returning.
 *   - The operation is performed in-place and requires O(1) auxiliary
 *     storage apart from the input arrays.
 */
void permute_array_inplace(int* input, int length, int* permutation) {
  for (int i = 0; i < length; i++) {
    // Skip if this index is already processed (marked negative)
    if (permutation[i] < 0) continue;

    // If the element is already in the right place, just mark and move on
    if (permutation[i] == i) {
      permutation[i] = -permutation[i] - 1;
      continue;
    }
  
    // Standard cycle-following for a "Gather" operation
    int current_hole = i;
    int save_val = input[i]; // Take the value out to create a 'hole'

    while (true) {
      int source_idx = permutation[current_hole];
      // Mark the current index as done
      permutation[current_hole] = -permutation[current_hole] - 1;
      if (source_idx == i) {
        // The cycle is complete; plug the hole with the saved value
        input[current_hole] = save_val;
        break;
      }

      // Pull the value from the source into the current hole
      input[current_hole] = input[source_idx];

      // The source_idx is now the new hole we need to fill
      current_hole = source_idx;
    }
  }

  // Restore the permutation array
  for (int i = 0; i < length; i++) {
    permutation[i] = -permutation[i] - 1;
  }
}

static void generateWalecki(int nNodes, int channel, int* order) {
  if (nNodes <= 0 || !order) return;

  // For Walecki, if N is even, we treat it as (N-1) nodes + 1 fixed pivot
  int m = (nNodes % 2) ? nNodes : (nNodes - 1);
  int left = 0;
  int right = m - 1;

  for (int i = 0; i < m; i++) {
    int val;
    if (i % 2 == 0) {
      val = (left + channel) % m;
      left++;
    } else {
      val = (right + channel) % m;
      right--;
    }
    order[i] = val;
  }
  if (nNodes % 2 == 0) {
    order[nNodes - 1] = nNodes - 1;
  }
}

/**
 * isPrime
 * Uses the 6k+/-1 optimization.
 * Time Complexity: O(sqrt(n))
 */
static bool isPrime(int n) {
    // 1. Handle the simplest cases
  if (n <= 1) return false;
  if (n <= 3) return true;

    // 2. Eliminate multiples of 2 and 3 immediately
  if (n % 2 == 0 || n % 3 == 0) return false;

    // 3. Check for divisors up to sqrt(n)
    // We skip even numbers and multiples of 3 by stepping by 6
  for (int i = 5; i * i <= n; i += 6) {
    if (n % i == 0 || n % (i + 2) == 0) {
      return false;
    }
  }

  return true;
}

/**
 * genRingsN_prime
 * For prime p, generates (p-1) perfectly balanced rings.
 *
 * @param p: The number of nodes (MUST be prime)
 * @param nChannels: Number of rings requested
 * @param nodeOrder: Buffer of size nChannels * p
 */
static int genRingsN_prime(int p, int nChannels, int* nodeOrder) {
  if (p <= 0 || nChannels <= 0 || nodeOrder == NULL) return 0;

    // There are (p-1) unique strides that produce Hamiltonian cycles
  int totalAvailable = p - 1;
  int numRingsCopied = 0;
  for (int c = 0; c < nChannels; c++) {
        // stride 's' must be in [1, p-1]
        // We use (c % totalAvailable) to loop through strides
    int s = (c % totalAvailable) + 1;
    int* currentRing = &nodeOrder[c * p];
    for (int i = 0; i < p; i++) {
            // The i-th rank in the ring is (i * s) mod p
      currentRing[i] = (i * s) % p;
    }
    numRingsCopied++;
  }
  return numRingsCopied;
}

/**
 * genRingsN_4
 * @param nodeOrder: Pre-allocated buffer of size nChannelsRequested * 4
 * @param nChannelsRequested: Number of rings the user wants
 * @return The number of rings successfully written
 */
static int genRingsN_4(int* nodeOrder, int nChannelsRequested) {
  const int totalAvailable = 6;
  const int nNodes = 4;
  const int optimizedRings[totalAvailable][nNodes] = {{0, 1, 2, 3}, {0, 2, 1, 3}, {0, 1, 3, 2},
                                                      {0, 3, 2, 1}, {0, 3, 1, 2}, {0, 2, 3, 1}};
  if (nodeOrder == NULL || nChannelsRequested <= 0) return 0;
  int numRingsCopied = 0;
  for (int c = 0; c < nChannelsRequested; c++) {
    int sourceIdx = c % totalAvailable;
    memcpy(&nodeOrder[c * nNodes], optimizedRings[sourceIdx], nNodes * sizeof(int));
    numRingsCopied++;
  }
  return numRingsCopied;
}

/**
 * genRingsN_6
 * @param nodeOrder: Pre-allocated buffer of size nChannelsRequested * 6
 * @param nChannelsRequested: Number of rings the user wants
 * @return The number of rings successfully written
 */
static int genRingsN_6(int* nodeOrder, int nChannelsRequested) {
  const int totalAvailable = 15;
  const int nNodes = 6;
  const int optimizedRings[totalAvailable][nNodes] = {{5, 3, 4, 1, 2, 0}, {4, 0, 3, 1, 5, 2}, {5, 1, 0, 4, 3, 2},
                                                      {2, 0, 1, 3, 5, 4}, {3, 0, 2, 1, 4, 5}, {1, 5, 0, 2, 3, 4},
                                                      {0, 4, 1, 3, 5, 2}, {1, 0, 2, 4, 5, 3}, {2, 5, 0, 1, 4, 3},
                                                      {1, 2, 3, 4, 0, 5}, {3, 2, 1, 4, 5, 0}, {2, 1, 3, 0, 5, 4},
                                                      {3, 0, 1, 5, 2, 4}, {4, 0, 3, 1, 2, 5}, {0, 4, 2, 3, 5, 1}};

  if (nodeOrder == NULL || nChannelsRequested <= 0) return 0;
  int numRingsCopied = 0;
  for (int c = 0; c < nChannelsRequested; c++) {
    int sourceIdx = c % totalAvailable;
    memcpy(&nodeOrder[c * nNodes], optimizedRings[sourceIdx], nNodes * sizeof(int));
    numRingsCopied++;
  }
  return numRingsCopied;
}

/**
 * genRingsN_8
 * @param nodeOrder: Pre-allocated buffer of size nChannelsRequested * 8
 * @param nChannelsRequested: Number of rings the user wants
 * @return The number of rings successfully written
 */
static int genRingsN_8(int* nodeOrder, int nChannelsRequested) {
  const int totalAvailable = 14;
  const int nNodes = 8;
  const int optimizedRings[totalAvailable][nNodes] = {
    {0, 1, 2, 3, 4, 5, 6, 7}, {0, 2, 4, 6, 1, 7, 3, 5}, {0, 5, 3, 7, 1, 6, 4, 2}, {0, 7, 6, 5, 4, 3, 2, 1},
    {0, 6, 4, 7, 2, 5, 1, 3}, {0, 2, 4, 1, 3, 6, 5, 7}, {0, 3, 5, 1, 6, 2, 7, 4}, {0, 7, 5, 6, 3, 1, 4, 2},
    {0, 3, 1, 5, 2, 7, 4, 6}, {0, 4, 7, 2, 6, 1, 5, 3}, {0, 5, 2, 6, 3, 7, 1, 4}, {0, 1, 2, 3, 4, 5, 7, 6},
    {0, 6, 7, 5, 4, 3, 2, 1}, {0, 4, 1, 7, 3, 6, 2, 5}
  };

  if (nodeOrder == NULL || nChannelsRequested <= 0) return 0;
  int numRingsCopied = 0;
  for (int c = 0; c < nChannelsRequested; c++) {
    // Use modulo to cycle through the optimized patterns if nChannels > 13
    int sourceIdx = c % totalAvailable;
    // Copy 8 integers (32 bytes) into the correct offset
    memcpy(&nodeOrder[c * nNodes], optimizedRings[sourceIdx], nNodes * sizeof(int));
    numRingsCopied++;
  }
  return numRingsCopied;
}

static ncclResult_t greedyRingGen(int nNodes, uint8_t nChannels, int* nodeOrder) {
    // Choose to augment with Greedy approach only if nNodes/2 channels are not sufficient. Most systems
    // do not execute below code as we have MAXCHANNELS = 128.
    // Optimization: uint8_t is sufficient for nChannels <= 255, In RCCL we limit it to 128 channels now.
  uint8_t* edgeUsage = (uint8_t*)calloc(nNodes * nNodes, sizeof(uint8_t));
  uint8_t* visited = (uint8_t*)malloc(nNodes * sizeof(uint8_t));

  if (!edgeUsage || !visited) {
    if (edgeUsage) free(edgeUsage);
    if (visited) free(visited);
    WARN("Unable to allocate memory with malloc/calloc");
    return ncclInternalError;
  }

  int startNode = 0;
  for (int c = 0; c < nChannels; c++) {
    if (c < nNodes / 2) {
      generateWalecki(nNodes, c, &nodeOrder[c * nNodes]);
      for (int i = 0; i < nNodes; i++) {
        int u = nodeOrder[c * nNodes + i];
        int v = nodeOrder[c * nNodes + ((i + 1) % nNodes)];
        edgeUsage[u * nNodes + v]++;
      }
      continue;
    }
        // Set all non-visited
    memset(visited, 0, nNodes * sizeof(uint8_t));
        // Only to reduce the pressure on starting node
    startNode = (startNode + 1) % nNodes;
    int curr = startNode;
    nodeOrder[c * nNodes + 0] = curr;
    visited[curr] = 1;

    for (int step = 1; step < nNodes; step++) {
      int bestNext = -1;
      int minCost = INT_MAX;

      for (int next = 0; next < nNodes; next++) {
        if (!visited[next]) {
          int cost = (int)edgeUsage[curr * nNodes + next];

          // Without this check, a greedy algorithm might pick a "cheap" bestNext for the second-to-last
          // node, but that node might have a very "expensive" (heavily used) link back to the start.
          if (step == nNodes - 1) {
            cost += (int)edgeUsage[next * nNodes + startNode];
          }

          if (cost < minCost) {
            minCost = cost;
            bestNext = next;
          }
        }
      }
      nodeOrder[c * nNodes + step] = bestNext;
      visited[bestNext] = 1;
      edgeUsage[curr * nNodes + bestNext]++;
      curr = bestNext;
    }
    edgeUsage[curr * nNodes + startNode]++;
  }
  free(visited);
  free(edgeUsage);
  return ncclSuccess;
}

/**
 * This function takes number of nodes in a fully connected graph and number target channels, and generates upto nChannel Hamiltonian cycles
 * In this function we use known constructions for 2,4,6,8 nodes in graph. For prime number of nodes, simple modulo construction creates
 * perfect set of Hamiltonian rings load balancing accross all channels, We use Walecki + greedy in rest of the cases.
 *
 * Assumptions : nodeOrder is pointer to flattened 2D array of size nNodes*nChannels*sizeof(int), and is pre-allocated before invoking this function.
 *
 */
ncclResult_t generateRings(int nNodes, uint8_t nChannels, int* nodeOrder) {
    // --- SAFETY CHECK: Guard against invalid cluster sizes ---
  if (nNodes <= 0 || nChannels <= 0 || nodeOrder == NULL) return ncclInvalidArgument;
  if (nChannels >= 255) {
    WARN(" generateRings is implemented with an assumption nChannels [=%d] < 255 as an optimization. Update the "
         "implementaion to accept uint16/32 for nChannels ",
         nChannels);
    return ncclInvalidArgument;
  }
    // Handle degenerate cases (N=1, N=2) where Hamiltonian diversity is impossible
  if (nNodes < 3) {
    for (int c = 0; c < nChannels; c++) {
      for (int n = 0; n < nNodes; n++) {
        nodeOrder[c * nNodes + n] = n;
      }
    }
    return ncclSuccess;
  }
  if (nNodes == 4) {
    genRingsN_4(nodeOrder, nChannels);
    return ncclSuccess;
  }
  if (nNodes == 6) {
    genRingsN_6(nodeOrder, nChannels);
    return ncclSuccess;
  }
  if (nNodes == 8) {
    genRingsN_8(nodeOrder, nChannels);
    return ncclSuccess;
  }
  if (isPrime(nNodes)) {
    genRingsN_prime(nNodes, nChannels, nodeOrder);
    return ncclSuccess;
  }

  if (nChannels <= (nNodes / 2)) {
    for (int c = 0; c < nChannels; c++) {
      generateWalecki(nNodes, c, &nodeOrder[c * nNodes]);
    }
    return ncclSuccess;
  }
  ncclResult_t res = greedyRingGen(nNodes, nChannels, nodeOrder);
  return res;
}

/**
 * findRingCutIndices
 * Calculates the optimal index to cut within each ring to maintain load balance.
 *
 * @param nChannels          Number of channels (rings)
 * @param nNodes             Number of nodes per ring
 * @param flattenedRings     Input array of size (k * n) containing the rings
 * @param cutIndices         Output array of size (k) allocated by caller.
 *                           The cut happens AFTER cutIndices[ch], making:
 *                           Exit  = currentRing[cutIndices[ch]]
 *                           Entry = currentRing[(cutIndices[ch] + 1) % n]
 */
void findRingCutIndices(int nChannels, int nNodes /*nodes in the ring graph*/,
                        const int* flattenedRings /* Hamiltonian rings*/, int* cutIndices) {
  if (nChannels <= 0 || nNodes <= 0 || !flattenedRings || !cutIndices) return;

    // Track historical balancing state across all allocations
  std::vector<int> exitCounts(nNodes, 0);
  std::vector<int> entryCounts(nNodes, 0);

  for (int channel = 0; channel < nChannels; channel++) {
    const int* currentRing = &flattenedRings[channel * nNodes];

    int bestCutIdx = 0;
    int minScore = 2e9; // Sentinel high value

        // Evaluate each possible edge cut inside the current ring
    for (int i = 0; i < nNodes; i++) {
      int u = currentRing[i];            // Potential Exit Node
      int v = currentRing[(i + 1) % nNodes];  // Potential Entry Node

            // Quadratic penalty function to enforce strict balance
      int potentialExitScore = (exitCounts[u] + 1) * (exitCounts[u] + 1);
      int potentialEntryScore = (entryCounts[v] + 1) * (entryCounts[v] + 1);
      int totalCost = potentialExitScore + potentialEntryScore;

      if (totalCost < minScore) {
        minScore = totalCost;
        bestCutIdx = i;
      }
    }

        // Commit the index for this channel
    cutIndices[channel] = bestCutIdx;

        // Keep our internal balance metrics updated
    int finalExit = currentRing[bestCutIdx];
    int finalEntry = currentRing[(bestCutIdx + 1) % nNodes];
    exitCounts[finalExit]++;
    entryCounts[finalEntry]++;
  }
}