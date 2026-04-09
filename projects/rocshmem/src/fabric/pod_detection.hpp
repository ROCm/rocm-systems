/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef ROCSHMEM_POD_DETECTION_HPP_
#define ROCSHMEM_POD_DETECTION_HPP_

#include <cstdint>
#include <vector>

namespace rocshmem {

/**
 * Structure containing physical and virtual pod IDs for a GPU.
 * Used to determine fabric topology and IPC capability.
 */
struct PodIds {
  uint32_t physicalPodId;  ///< Physical pod ID (ppod_id from AMD SMI)
  uint32_t virtualPodId;   ///< Virtual pod ID (vpod_id from AMD SMI)
};

/**
 * Detect the pod IDs for the current rank's GPU.
 *
 * This function:
 * - Gets the current HIP device
 * - Retrieves the BDF ID (PCI Bus ID)
 * - Loads AMD SMI library dynamically
 * - Queries fabric information to get ppod_id and vpod_id
 *
 * @return PodIds structure with physical and virtual pod IDs.
 *         Returns {0, 0} if detection fails at any step.
 */
PodIds detectLocalPodIds();

/**
 * Match IPC-capable ranks based on pod IDs.
 *
 * Ranks are considered IPC-capable if they share the same physical
 * and virtual pod IDs, indicating they are in the same fabric domain.
 *
 * @param rank Current rank
 * @param allPodIds Vector of PodIds from all ranks (already gathered)
 * @return Vector of rank indices that are IPC-capable with the current rank
 */
std::vector<int> matchIpcCapableRanks(int rank, const std::vector<PodIds>& allPodIds);

}  // namespace rocshmem

#endif  // ROCSHMEM_POD_DETECTION_HPP_
