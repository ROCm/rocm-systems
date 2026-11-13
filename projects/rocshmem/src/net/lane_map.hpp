/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *****************************************************************************/

#ifndef ROCSHMEM_LIBRARY_SRC_NET_LANE_MAP_HPP_
#define ROCSHMEM_LIBRARY_SRC_NET_LANE_MAP_HPP_

#include <cassert>

namespace rocshmem {
namespace net {

/**
 * @brief Maps a (context, peer-PE) pair to a flat "lane" index.
 *
 * A lane is the unit of independent network resource a conduit owns: one QP +
 * CQ (verbs), one tx-context + counter (OFI), or one device (LCI). Using
 * lane = ctx * num_pes + pe gives each (context, destination) its own ordered
 * stream and its own completion counter, so quiet(ctx) drains only the lanes
 * that context posted to rather than flushing every peer (the O(PEs)-per-put
 * cost of the MPI window path).
 *
 * Transport-neutral: pure integer arithmetic, no NIC headers.
 */
struct LaneMap {
  int num_ctx{0};  //!< number of device/host contexts
  int num_pes{0};  //!< number of PEs

  LaneMap() = default;
  LaneMap(int contexts, int pes) : num_ctx{contexts}, num_pes{pes} {}

  //! Flat lane index for a (context, peer) pair.
  int lane(int ctx, int pe) const {
    assert(ctx >= 0 && ctx < num_ctx);
    assert(pe >= 0 && pe < num_pes);
    return ctx * num_pes + pe;
  }

  //! Total number of lanes the conduit must allocate resources for.
  int num_lanes() const { return num_ctx * num_pes; }

  //! First and one-past-last lane owned by a context (for quiet(ctx)).
  int lane_begin(int ctx) const { return ctx * num_pes; }
  int lane_end(int ctx) const { return (ctx + 1) * num_pes; }
};

}  // namespace net
}  // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_NET_LANE_MAP_HPP_
