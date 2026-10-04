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

#ifndef ROCSHMEM_LIBRARY_SRC_NET_REMOTE_REGION_HPP_
#define ROCSHMEM_LIBRARY_SRC_NET_REMOTE_REGION_HPP_

#include <cstdint>

namespace rocshmem {
namespace net {

/**
 * @brief A peer PE's registered symmetric-heap region.
 *
 * Transport-neutral by design: it carries only the peer's registered base VA
 * and a network key, in plain integer types, so it pulls in no libibverbs /
 * libfabric / LCI headers. Concrete conduits fill it from their own registration
 * exchange:
 *   - verbs : remote_base = peer heap VA,  key = ibv_mr::rkey (32-bit, widened)
 *   - ofi   : remote_base = peer heap VA,  key = fi_mr key   (may be 64-bit)
 * The LCI conduit uses its own opaque remote-MR handle (rmr_t) and does not use
 * this struct.
 *
 * Because rocSHMEM's heap is symmetric, a local symmetric VA maps to a peer VA
 * by preserving the offset from the local heap base; see remote_addr().
 */
struct RemoteRegion {
  uintptr_t remote_base{0};  //!< peer's registered symmetric-heap base VA
  uint64_t key{0};           //!< verbs rkey (widened) or OFI mr key
};

/**
 * @brief Translate a local symmetric-heap VA to the peer's remote VA.
 *
 * @param region          The peer region (its remote_base).
 * @param local_va        A VA within this PE's symmetric heap.
 * @param local_heap_base This PE's symmetric-heap base VA.
 * @return The corresponding VA in the peer's symmetric heap.
 */
inline uintptr_t remote_addr(const RemoteRegion &region, uintptr_t local_va,
                             uintptr_t local_heap_base) {
  return region.remote_base + (local_va - local_heap_base);
}

}  // namespace net
}  // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_NET_REMOTE_REGION_HPP_
