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

#include "wf_coal_policy.hpp"

#include "util.hpp"

namespace rocshmem {

/**
 * TODO: Determine tradeoffs between number of instructions, LDS utilization,
 * and quality of coalescing.
 */

/**
 * About the current algorithm:
 *   1) The algorithm is low overhead. It uses no LDS (shared) space or
 *      rounds of a tree reduction (which may cause it to miss some
 *      scenarios).
 *   2) The algorithm can identify multiple groups of coalescable packets
 *      within a wavefront.
 *      As an example:
 *          Threads 0-8 are coalescable.
 *          Threads 10-63 are coalescable.
 *          The logic will generate 3 messages.
 *      +-----------------------+----------+-------------------------+
 *      | Thread_0 ... Thread_8 | Thread_9 | Thread_10 ... Thread_63 |
 *      +-----------------------+----------+-------------------------+
 *      |          msg_1        |  msg 2   |          msg 3          |
 *      +-----------------------+----------+-------------------------+
 *      However, threads must be contiguous by thread id to be coalesced.
 *      For example, Thread_0 and Thread_2 are coalescable, but Thread_1 is
 *      not coalescable (see coalescability conditions) then none of the
 *      three threads can be coalesced.
 *   3) The algorithm misses opportunities when threads have coalescable
 *      messages but the message sizes are different.
 */
__device__ bool WfCoalOn::coalesce(int pe, const void *source, const void *dest,
                                   size_t *size) {
  const uint64_t src = (const uint64_t)source;
  const uint64_t dst = (const uint64_t)dest;

  /**
   * Split 64-bit values into high and low for 32-bit shuffles.
   * Unfortunately, the shuffle operations only support 32-bit widths.
   */
  uint32_t src_low = uint32_t(src & 0xFFFFFFFF);
  uint32_t src_high = uint32_t((src >> 32) & 0xFFFFFFFF);
  uint32_t dst_low = uint32_t(dst & 0xFFFFFFFF);
  uint32_t dst_high = uint32_t((dst >> 32) & 0xFFFFFFFF);

  /**
   * Shuffle message info to upwards neighboring threads.
   * +----------------------------------------------------------------+
   * | Thread_0 Thread_1  ...   ...   ...   ...   Thread_62 Thread_63 |
   * +----------------------------------------------------------------+
   * | Upwards                                              Downwards |
   * +----------------------------------------------------------------+
   *
   * The implementation of __shfl_up comes from the hip header files.
   * In rocm 2.10, the filename is device_functions.h.
   */
  uint64_t lower_src_low = __shfl_up(src_low, 1);
  uint64_t lower_src_high = __shfl_up(src_high, 1);
  uint64_t lower_dst_low = __shfl_up(dst_low, 1);
  uint64_t lower_dst_high = __shfl_up(dst_high, 1);
  int lower_pe = __shfl_up(pe, 1);
  size_t lower_size = __shfl_up((unsigned int)*size, 1);

  /**
   * Recombine the incoming 64-bit values from neighbor.
   */
  uint64_t lower_src = (lower_src_high << 32) | lower_src_low;
  uint64_t lower_dst = (lower_dst_high << 32) | lower_dst_low;

  /**
   * The mask variable tells us which threads are active in the wavefront.
   * An active thread will call into this function with a value '1' which
   * notifies the other threads that the lane is active.
   */
  uint64_t mask = __ballot(1);

  /**
   * The wv_id variable holds the wavefront id number. To set it, we
   * flatten the thread block out (to make it one-dimensional) and then
   * modulo based of the wavefront size (which is a characteristic of
   * the hardware).
   */
  int wv_id = get_flat_block_id() % WF_SIZE;

  /**
   * If coalescable evaluates to true, this thread is __NOT__ responsible
   * for sending a message (another thread will send the message on its
   * behalf). In other words, the thread's message is coalesced, yay.
   *
   * If coalescable evaluates to false, this thread is responsible for
   * sending a message (which means that it is not coalescable with its
   * lower neighbor).
   */
  bool coalescable =
      (mask & (1LL << (wv_id - 1))) &&  // Ensure lower lane is active
      (lower_size == *size) &&          // Ensure lower lane size is equal
      ((lower_src + *size) == src) &&   // Ensure I cover lower src
      ((lower_dst + *size) == dst) &&   // Ensure I cover lower dst
      (pe == lower_pe) &&               // Must be sending to the same pe
      (wv_id != 0);                     // Thread_0 is never coalescable

  /**
   * Share the lower neighbor coalescability status with all the active
   * threads in the wavefront.
   *
   * Inactive threads will not participate in the ballot which returns '0'
   * in their position.
   */
  uint64_t lowerNeighborCoal = __ballot(coalescable);

  /**
   * If the thread is not coalescable, it must send a message.
   * It needs to check how many threads are considered coalesced above it
   * to adjust its message size.
   *
   * Do this by counting the number of contiguous '1's greater than its
   * thread ID from the ballot function. It will coalesce the messages for
   * all contiguous higher threads which report that they are coalescable
   * with their immediate lower neighbor.
   */
  if (!coalescable) {
    int coal_size = *size;

    /**
     * Ignore the lower threads and the thread's own position.
     */
    lowerNeighborCoal >>= (wv_id + 1);

    /**
     * Invert and find the first bit index set to zero. This bit
     * indicates the first higher thread which is not coalescable with
     * its lower neighbor.
     *
     * This thread is now responsible for coalescing everything
     * between its own index and that one.
     */
    uint32_t coalMsgs = __ffsll((unsigned long long)~lowerNeighborCoal);

    if (coalMsgs) {
      coal_size += *size * (coalMsgs - 1);
    }

    *size = coal_size;
  }

  return !coalescable;
}

}  // namespace rocshmem
