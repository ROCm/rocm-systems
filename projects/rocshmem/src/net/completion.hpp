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

#ifndef ROCSHMEM_LIBRARY_SRC_NET_COMPLETION_HPP_
#define ROCSHMEM_LIBRARY_SRC_NET_COMPLETION_HPP_

#include <cassert>
#include <cstdint>
#include <vector>

namespace rocshmem {
namespace net {

/**
 * @brief Per-lane posted/completed bookkeeping for the standalone conduits.
 *
 * The raw verbs conduit (and the OFI conduit when using a plain CQ) gets no
 * completion aggregation from its library: ibv_post_send / ibv_poll_cq only
 * hand back individual work completions. This tracks, per lane, how many
 * operations were posted vs. reaped, which is exactly what quiet(lane) needs:
 * spin the poller until outstanding(lane) == 0.
 *
 * This is intentionally NOT the MPI backend's B2/B3/B4 progress machinery --
 * it is plain counters. Threading is the conduit's concern; this type does no
 * locking (a conduit that posts and polls on the same progress thread needs
 * none).
 *
 * Transport-neutral: no NIC headers.
 */
class CompletionCounters {
 public:
  CompletionCounters() = default;
  explicit CompletionCounters(int num_lanes)
      : posted_(static_cast<size_t>(num_lanes), 0),
        completed_(static_cast<size_t>(num_lanes), 0) {}

  //! Record @p n newly posted operations on @p lane.
  void on_post(int lane, uint64_t n = 1) {
    assert(lane >= 0 && static_cast<size_t>(lane) < posted_.size());
    posted_[lane] += n;
  }

  //! Record @p n newly completed operations on @p lane.
  void on_complete(int lane, uint64_t n = 1) {
    assert(lane >= 0 && static_cast<size_t>(lane) < completed_.size());
    completed_[lane] += n;
  }

  //! Operations posted to @p lane but not yet completed.
  uint64_t outstanding(int lane) const {
    return posted_[lane] - completed_[lane];
  }

  //! True once every op posted to @p lane has completed.
  bool drained(int lane) const { return posted_[lane] == completed_[lane]; }

  uint64_t posted(int lane) const { return posted_[lane]; }
  uint64_t completed(int lane) const { return completed_[lane]; }

  int num_lanes() const { return static_cast<int>(posted_.size()); }

 private:
  std::vector<uint64_t> posted_;
  std::vector<uint64_t> completed_;
};

}  // namespace net
}  // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_NET_COMPLETION_HPP_
