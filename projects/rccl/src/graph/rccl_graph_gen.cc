
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

RCCL_PARAM(IntraGraphGen, "INTRA_GRAPH_GEN", 0);

static void generateWalecki(int nNodes, int channel, int* order) {
  if (nNodes <= 0 || !order) return;

  // For Walecki, if N is even, we treat it as (N-1) nodes + 1 fixed pivot
  int m = (nNodes % 2) ? nNodes : (nNodes - 1) ;
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
  if( nNodes % 2 == 0) {
    order[nNodes - 1] = nNodes - 1;
  }
}

/**
 * This function takes number of nodes in a fully connected graph and number target channels, and generates upto nChannel Hamiltonian cycles 
 * In this function we initially generate Walecki Construction depending on (nNodes mod 2), upto nNodes / 2 channels. Then, based on edge usage
 * heuristic, we construct rest of the cycles.
 * 
 * Assumptions : nodeOrder is pointer to flattened 2D array of size nNodes*nChannels*sizeof(int), and is pre-allocated before invoking this function.
 * 
 */
ncclResult_t generateRings(int nNodes, uint8_t nChannels, int* nodeOrder) {
    // --- SAFETY CHECK: Guard against invalid cluster sizes ---
    if (nNodes <= 0 || nChannels <= 0 || nodeOrder == NULL) return ncclInvalidArgument;
    if ( nChannels >= 255 ) {
        WARN(" generateRings is implemented with an assumption nChannels [=%d] < 255 as an optimization. Update the implementaion to accept uint16/32 for nChannels ",nChannels );
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

    if (nChannels <= (nNodes / 2)) {
        for (int c = 0; c < nChannels; c++) {
            generateWalecki(nNodes, c, &nodeOrder[c * nNodes]);
        }
        return ncclSuccess;
    }

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
        //Set all non-visited
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
