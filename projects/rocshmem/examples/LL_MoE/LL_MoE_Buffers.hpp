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

#include "../util.h"

/******************************************************************************
 * Buffer sizing and allocation (DeepEP LL mode style)
 *
 * DeepEP low-latency mode pre-allocates all communication buffers up
 * front, sized for the WORST-CASE decode micro-batch.
 *
 * This avoids:
 *   - dynamic allocation on the critical path
 *   - buffer resizing between iterations
 *
 * In DeepEP, this idea is exposed via a size-hint helper for LL RDMA
 * buffers, and the documentation notes that LL mode consumes more
 * memory and typically limits max dispatch tokens per rank.
 *
 * This example mirrors that strategy by computing a maximum required
 * size once and reusing the same buffers across all iterations.
 *****************************************************************************/

/**
 * Low-latency buffer structure
 * - Separate dispatch and combine buffers
 * - Separate send and receive buffers
 * - Separate signaling buffers for dispatch and combine
 */
struct LLMoEBuffer {
  // Number of signaling elements = number of experts
  int num_sig_elems {0};

  /**
   * Dispatch buffers
   * Dimensions:
   * - Send buffer: [num_tokens, hidden + 1]
   * - Recv buffer: [num_experts * num_tokens, hidden + 1
   * - Recv count buffer: [num_experts]
   */
  void*    dispatch_send_buffer {nullptr};
  void*    dispatch_recv_buffer {nullptr};
  int64_t* dispatch_recv_count_buffer {nullptr};

  /**
   * Combine buffers
   * Dimensions:
   * - Send buffer: [num_experts * num_tokens, hidden + 1]
   * - Recv buffer: [num_experts * num_tokens, hidden + 1]
   * - Recv flag buffer: [num_experts]
   */
  void*    combine_send_buffer {nullptr};
  void*    combine_recv_buffer {nullptr};
  int64_t* combine_recv_flag_buffer {nullptr};

  std::pair<int64_t*, int> clean_meta() {
    ASSERT(dispatch_recv_count_buffer == combine_recv_flag_buffer);
    return {dispatch_recv_count_buffer, num_sig_elems};
  }
};

/**
 * Low-latency buffer layout
 * - Two sets of LLMoEBuffer for double buffering
 * - Calculates total bytes required for allocation
 */
template <typename T>
struct LLMoEBufferLayout {
  size_t total_bytes {0};
  LLMoEBuffer buffers[2];

  template <typename out_ptr_t = void*,
            typename count_ptr_t = uint8_t*,
            typename in_ptr_t = void*>
  out_ptr_t advance(const in_ptr_t& ptr, size_t count) {
      return reinterpret_cast<out_ptr_t>(
          reinterpret_cast<count_ptr_t>(ptr) + count);
  }

  LLMoEBufferLayout(void* rdma_buffer, const int num_tokens, const int hidden,
      const int num_ranks, const int num_experts) {
    
    const int num_local_experts = num_experts / num_ranks;

    // Message sizes
    size_t num_bytes_per_dispatch_msg = sizeof(int) + hidden * sizeof(T);
    size_t num_bytes_per_combine_msg  = sizeof(int) + hidden * sizeof(T);

    // Send buffers sizes
    size_t dispatch_send_buffer_bytes = num_tokens *
                                        num_bytes_per_dispatch_msg;
    size_t combine_send_buffer_bytes  = num_experts * num_tokens *
                                        num_bytes_per_combine_msg;
    size_t send_buffer_bytes = std::max(dispatch_send_buffer_bytes,
                                        combine_send_buffer_bytes);

    ASSERT(send_buffer_bytes % sizeof(int) == 0);
    total_bytes += send_buffer_bytes * 2;

    // receive buffers sizes
    size_t dispatch_recv_buffer_bytes = num_experts * num_tokens *
                                        num_bytes_per_dispatch_msg;
    size_t combine_recv_buffer_bytes  = num_experts * num_tokens *
                                        num_bytes_per_combine_msg;
    size_t recv_buffer_bytes = std::max(dispatch_recv_buffer_bytes,
                                        combine_recv_buffer_bytes);
    ASSERT(recv_buffer_bytes % sizeof(int) == 0);
    total_bytes += recv_buffer_bytes * 2;

    // Symmetric signaling buffers
    size_t signaling_buffer_bytes = num_experts * sizeof(int64_t);
    total_bytes += signaling_buffer_bytes * 2;

    // Assign pointers
    for (int i = 0; i < 2; ++ i) {
        buffers[i] = {
            num_experts,
            advance(rdma_buffer, send_buffer_bytes * i),
            advance(rdma_buffer,
                    send_buffer_bytes * 2 + recv_buffer_bytes * i),
            advance<int64_t*>(rdma_buffer,
                    send_buffer_bytes * 2 + recv_buffer_bytes * 2 +
                    signaling_buffer_bytes * i),
            advance(rdma_buffer, send_buffer_bytes * i),
            advance(rdma_buffer,
                    send_buffer_bytes * 2 + recv_buffer_bytes * i),
            advance<int64_t*>(rdma_buffer,
                    send_buffer_bytes * 2 + recv_buffer_bytes * 2 +
                    signaling_buffer_bytes * i)
        };
    }
  }
};

/**
 * Get RDMA size hint for low-latency buffers
 * - Used for allocating rocSHMEM symmetric memory buffer
 */
template <typename T>
size_t get_rdma_size_hint(int num_max_dispatch_tokens_per_rank, int hidden,
    int num_ranks, int num_experts) {
  LLMoEBufferLayout<T> ll_buffer_layout(nullptr, num_max_dispatch_tokens_per_rank,
                        hidden, num_ranks, num_experts);
  return ll_buffer_layout.total_bytes;
}