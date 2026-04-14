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

#include <mpi.h>

#include "LL_MoE_Data.hpp"
#include "LL_MoE_Buffers.hpp"
#include "LL_MoE_Kernels.hpp"

using namespace rocshmem;

template<typename T>
class LLMoE {
 private:
  int rank {0}, num_ranks {0};
  int device_id {0};

  // rocSHMEM buffer
  int64_t num_rdma_bytes {0};
  void*   rdma_buffer_ptr {nullptr};

  // Workspace (32 MiB)
  void* workspace {nullptr};

  // LL_DeepEP parameters
  const int num_tokens {0};
  const int hidden {0};
  const int num_topk {0};
  const int num_experts {0};

  // LL_DeepEP data
  LLMoEData<T> ll_moe_data;

  // LL Buffer index
  int ll_buffer_idx {0};

  // Dispatch buffers
  T*        packed_recv_x {nullptr};
  int*      packed_recv_src_info {nullptr};
  int64_t*  packed_recv_layout_range {nullptr};
  int*      packed_recv_count {nullptr};
  int*      global_atomic_counter {nullptr};

  // Combine buffers
  T*        combined_x {nullptr};

  // HIP stream to launch kernels
  hipStream_t stream;

  // init mode
  InitMode init_mode {InitMode::Deterministic};


 public:
  LLMoE(int num_tokens_, int hidden_, int num_topk_, int num_experts_,
    InitMode init_mode_ = InitMode::Deterministic)
      : num_tokens(num_tokens_), hidden(hidden_), num_topk(num_topk_),
        num_experts(num_experts_), init_mode(init_mode_),
        ll_moe_data(num_tokens_, hidden_, num_topk_, num_experts_,init_mode_) {

    // Initialize rocSHMEM
    comm_init();

    // print rank and gpu id
    std::cout << "Rank " << rank << " using GPU " << device_id << std::endl;

    // Create HIP stream
    CHECK_HIP(hipStreamCreate(&stream));

    // num_experts must be divisible by num_ranks
    ASSERT(num_experts % num_ranks == 0);

    /**
     * Generate input data after COMM init, because it sets different device
     * IDs for different ranks.
     */
    ll_moe_data.generate_data();

    CHECK_HIP(hipExtMallocWithFlags(&workspace, NUM_WORKSPACE_BYTES,
              hipDeviceMallocUncached));
    // Mmemset workspace to zero
    CHECK_HIP(hipMemsetAsync(workspace, 0, NUM_WORKSPACE_BYTES, stream));

    // Allocate rocSHMEM buffer
    num_rdma_bytes = get_rdma_size_hint<T>(
        num_tokens, hidden, num_ranks, num_experts);
    rdma_buffer_ptr = rocshmem_malloc(num_rdma_bytes);
    if (rdma_buffer_ptr == nullptr) {
      std::cerr << "Rank " << rank
                << ": Error in rocshmem_malloc. Aborting." << std::endl;
      comm_finalize();
      // Clean up other resources and exit
      exit(EXIT_FAILURE);
    }

    // Mmemset rdma_buffer_ptr to zero
    CHECK_HIP(hipMemsetAsync(rdma_buffer_ptr, 0, num_rdma_bytes, stream));

    // Allocate dispatch buffers
    allocate_dispatch_buffers();

    // Synchronize to ensure all allocations are done
    CHECK_HIP(hipStreamSynchronize(stream));
  }

  ~LLMoE() {
    CHECK_HIP(hipFree(workspace));

    // Free dispatch buffers
    CHECK_HIP(hipFree(packed_recv_x));
    CHECK_HIP(hipFree(packed_recv_src_info));
    CHECK_HIP(hipFree(packed_recv_layout_range));
    CHECK_HIP(hipFree(packed_recv_count));

    // Free combine buffers
    CHECK_HIP(hipFree(combined_x));

    // Destroy HIP stream
    CHECK_HIP(hipStreamDestroy(stream));

    // Free rocSHMEM buffer
    rocshmem_free(rdma_buffer_ptr);
    comm_finalize();
  }

  int get_rank() const {
    return rank;
  }

  int get_num_ranks() const {
    return num_ranks;
  }

  void ll_dispatch() {
    int num_local_experts {num_experts / num_ranks};

    // Buffer control
    LLMoEBufferLayout<T> ll_layout(rdma_buffer_ptr, num_tokens,
                       hidden, num_ranks, num_experts);
    LLMoEBuffer& buffer = ll_layout.buffers[ll_buffer_idx];
    LLMoEBuffer& next_buffer = ll_layout.buffers[ll_buffer_idx ^= 1];

    // Hip memset global_atomic_counter to zero
    CHECK_HIP(hipMemsetAsync(global_atomic_counter, 0, sizeof(int), stream));

    // Launch dispatch kernel
    ll_kernels::dispatch<T>(packed_recv_x, packed_recv_src_info,
      packed_recv_layout_range, packed_recv_count, global_atomic_counter,
      buffer.dispatch_recv_buffer, buffer.dispatch_recv_count_buffer,
      buffer.dispatch_send_buffer, ll_moe_data.X, ll_moe_data.topk_idx,
      next_buffer.clean_meta().first, next_buffer.clean_meta().second,
      num_tokens, hidden, num_topk, num_experts, rank, num_ranks, workspace,
      stream);

    // hipStreamSynchronize
    CHECK_HIP(hipStreamSynchronize(stream));

    //--- DEBUG FUNCTIONS ---
    // Print rdma_x (buffer.dispatch_send_buffer) for debugging
    // print_rdma_x(buffer.dispatch_send_buffer);

    // Print rdma_recv_x (buffer.dispatch_recv_buffer) for debugging
    // print_rdma_recv_x(buffer.dispatch_recv_buffer, num_experts / num_ranks);

    // print atomic counters for debugging
    // print_atomic_counters();

    // Print dispatch recv counts
    // print_dispatch_recv_counts(num_local_experts,
    //     buffer.dispatch_recv_count_buffer);

    // Print packed recv buffers
    // print_packed_recv_buffers(num_local_experts);
  }

  void ll_combine() { // hipStream_t stream : TODO: pass stream
    int num_local_experts {num_experts / num_ranks};

    // Buffer control
    LLMoEBufferLayout<T> ll_layout(rdma_buffer_ptr, num_tokens,
                       hidden, num_ranks, num_experts);
    LLMoEBuffer& buffer = ll_layout.buffers[ll_buffer_idx];
    LLMoEBuffer& next_buffer = ll_layout.buffers[ll_buffer_idx ^= 1];

    // Hip memset global_atomic_counter to zero
    CHECK_HIP(hipMemsetAsync(global_atomic_counter, 0,
      sizeof(int), stream));

    // Hip memset combined_x to zero
    CHECK_HIP(hipMemsetAsync(combined_x, 0,
      num_tokens * hidden * sizeof(T), stream));

    // Launch combine kernel
    ll_kernels::combine<T>(combined_x, buffer.combine_recv_buffer,
      buffer.combine_recv_flag_buffer, buffer.combine_send_buffer,
      packed_recv_x, ll_moe_data.topk_idx, packed_recv_src_info,
      packed_recv_layout_range, global_atomic_counter,
      next_buffer.clean_meta().first, next_buffer.clean_meta().second,
      num_tokens, num_topk, hidden, num_experts, rank, num_ranks,
      workspace, stream);

    // hipStreamSynchronize
    CHECK_HIP(hipStreamSynchronize(stream));

    //--- DEBUG FUNCTIONS ---
    // Print combine send buffer for debugging
    // print_combine_send_buffer(buffer.combine_send_buffer);

    // Print combined_x for debugging
    // print_combined_x(combined_x);

    // Verify combined_x data based on the experts it is routed to
    // verify_combined_x(combined_x);

    // Verify if init mode is Deterministic
    if (init_mode == InitMode::Deterministic) {
      verify_combined_x(combined_x);
    }
  }

 private:
  // Initialize rocSHMEM
  void comm_init() {

    // Initialize MPI
    int mpi_rank {0}, mpi_size {0};
    int ret {0};
    int provided {0};
    MPI_Init_thread (nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE) {
      std::cerr << "MPI_THREAD_MULTIPLE support disabled.\n";
    }
    MPI_Comm_rank (MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size (MPI_COMM_WORLD, &mpi_size);

    // Set GPU id
    int device_count {0};
    CHECK_HIP(hipGetDeviceCount(&device_count));
    CHECK_HIP(hipSetDevice(mpi_rank % device_count));

    // Get GPU id
    CHECK_HIP(hipGetDevice(&device_id));

    // Initialize rocSHMEM with unique ID
    rocshmem_uniqueid_t uid;
    rocshmem_init_attr_t attr;
    if (mpi_rank == 0) {
      ret = rocshmem_get_uniqueid (&uid);
      if (ret != ROCSHMEM_SUCCESS) {
        std::cout << mpi_rank
        << ": Error in rocshmem_get_uniqueid. Aborting." << std::endl;
        MPI_Abort (MPI_COMM_WORLD, ret);
      }
    }

    // Broadcast the unique ID to all ranks
    MPI_Bcast (&uid, sizeof(rocshmem_uniqueid_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    ret = rocshmem_set_attr_uniqueid_args(mpi_rank, mpi_size, &uid, &attr);
    if (ret != ROCSHMEM_SUCCESS) {
      std::cout << mpi_rank
                << ": Error in rocshmem_set_attr_uniqueid_args. Aborting"
                << std::endl;
      MPI_Abort (MPI_COMM_WORLD, ret);
    }

    ret = rocshmem_init_attr(ROCSHMEM_INIT_WITH_UNIQUEID, &attr);
    if (ret != ROCSHMEM_SUCCESS) {
      std::cout << mpi_rank << ": Error in rocshmem_init_attr. Aborting."
                << std::endl;
      MPI_Abort (MPI_COMM_WORLD, ret);
    }

    rank      = rocshmem_my_pe();
    num_ranks = rocshmem_n_pes();
  }

  void comm_finalize() {
    rocshmem_finalize();
    MPI_Finalize();
  }

  // Allocate GPU buffers/tensors for dispatch
  void allocate_dispatch_buffers() {
    int num_local_experts = num_experts / num_ranks;
    size_t packed_recv_x_bytes =
        num_local_experts * num_ranks * num_tokens * hidden * sizeof(T);
    size_t packed_recv_src_info_bytes =
        num_local_experts * num_ranks * num_tokens * sizeof(int);
    size_t packed_recv_layout_range_bytes =
        num_local_experts * num_ranks * sizeof(int64_t);
    size_t packed_recv_count_bytes =
        num_local_experts * sizeof(int);

    size_t combined_x_bytes = num_tokens * hidden * sizeof(T);

    // Dimemsions: [num_local_experts][num_ranks][num_tokens][hidden]
    CHECK_HIP(hipMalloc(&packed_recv_x, packed_recv_x_bytes));
    // Dimensions: [num_local_experts][num_ranks][num_tokens]
    CHECK_HIP(hipMalloc(&packed_recv_src_info, packed_recv_src_info_bytes));
    // Dimensions: [num_local_experts][num_ranks]
    CHECK_HIP(hipMalloc(&packed_recv_layout_range,
                        packed_recv_layout_range_bytes));
    // Dimensions: [num_local_experts]
    CHECK_HIP(hipMalloc(&packed_recv_count, packed_recv_count_bytes));
    // Dimensions: [num_tokens][hidden]
    CHECK_HIP(hipMalloc(&combined_x, combined_x_bytes));
    CHECK_HIP(hipMalloc(&global_atomic_counter, sizeof(int)));
  }

  /**
   * ------------ DEBUG FUNCTIONS ------------
   */
  // Verify combined_x data based on the experts it is routed to
  void verify_combined_x(void* combined_x) {
    T* combined_x_t = reinterpret_cast<T*>(combined_x);
    bool all_correct = true;
    for (int i = 0; i < num_tokens; i++) {
      for (int h = 0; h < hidden; h++) {
        T expected_value = ll_moe_data.X[i * hidden + h] * num_topk;
        T actual_value = combined_x_t[i * hidden + h];
        if (actual_value != expected_value) {
          std::cout << "Mismatch at Token " << i << ", Hidden " << h
                    << ": Expected " << expected_value
                    << ", Actual " << actual_value << std::endl;
          all_correct = false;
        }
      }
    }
    if (! all_correct) {
      std::cout << "Verification failed: combined_x has mismatches." << std::endl;
    }
  }

  // Print combined_x of dimensions [num_tokens][hidden]
  void print_combined_x(void* combined_x) {
    T* combined_x_t = reinterpret_cast<T*>(combined_x);
    for (int i = 0; i < num_tokens; i++) {
      std::cout << "Token " << i << ": ";
      for (int j = 0; j < hidden; j++) {
        std::cout << combined_x_t[i * hidden + j] << " ";
      }
      std::cout << std::endl;
    }
  }

  // Print combine send buffer for debugging
  void print_combine_send_buffer(void* combine_send_buffer) {
    T* combine_send_buffer_t =
        reinterpret_cast<T*>(combine_send_buffer);
    int* packed_send_src_info_t =
        reinterpret_cast<int*>(packed_recv_src_info);
    int num_local_experts = num_experts / num_ranks;
    const size_t num_T_per_slot = hidden + (sizeof(int) / sizeof(T));
    for (int e = 0; e < num_local_experts; e++) {
      std::cout << "Expert " << (rank * num_local_experts + e)
                << "[local expert " << e << "]:\n";
      for (int r = 0; r < num_ranks; r++) {
        std::cout << "  To Rank " << r << ":\n";
        // Starting offset and number of tokens sent to rank r
        int64_t *layout_range = reinterpret_cast<int64_t*>(
            packed_recv_layout_range + (e * num_ranks + r));
        int num_tokens_from_r = reinterpret_cast<int*>(
            layout_range)[0];
        int offset_for_r = reinterpret_cast<int*>(
            layout_range)[1];
        std::cout << "    Num tokens: " << num_tokens_from_r
                  << ", Offset: " << offset_for_r
                  << ", packed: " << *layout_range
                  << ", addr: " << layout_range <<"\n";
        for (int t = 0; t < num_tokens_from_r; t++) {
          // int offset = e * num_ranks * num_tokens * hidden +
          //              (offset_for_r + t) * hidden + sizeof(int)/sizeof(T);
          int offset = e * num_ranks * num_tokens * num_T_per_slot +
                       (offset_for_r + t) * num_T_per_slot +
                       sizeof(int)/sizeof(T);
          std::cout << "      Token " << t << ": (Index: "
                    << packed_send_src_info_t[e * num_ranks * num_tokens +
                                              (offset_for_r + t)]
                    << ") " << " Addr: " << &combine_send_buffer_t[offset] << " ";
          for (int h = 0; h < hidden; h++) {
            std::cout << combine_send_buffer_t[offset + h] << " ";
          }
          std::cout << std::endl;
        }
      }
    }
  }

  // Print Ranks and the experts assigned to them
  void print_expert_assignment() {
    int num_local_experts = num_experts / num_ranks;
    for (int r = 0; r < num_ranks; r++) {
      // if (r == rank) {
        std::cout << "Rank [" << r << "]: Experts: ";
        for (int e = 0; e < num_local_experts; e++) {
          std::cout << (r * num_local_experts + e) << " ";
        }
        std::cout << std::endl;
      // }
    }
  }

  /**
   * Function to print Atomic counters per expert from workspace
   */
  void print_atomic_counters() {
    int* atomic_counter_per_expert = reinterpret_cast<int*>(workspace);
    int* atomic_finish_counter_per_expert = atomic_counter_per_expert +
                                            num_experts;
    std::cout << "Atomic counters per expert:\n";
    for (int i = 0; i < num_experts; i++) {
      std::cout << "Expert " << i << ": "
                << atomic_counter_per_expert[i]
                << ", Finish: "
                << atomic_finish_counter_per_expert[i]
                << std::endl;
    }
  }

  /**
   * Function to print packed_recv_x and packed_recv_src_info
   * Dimensions:
   *    packed_recv_x: [num_local_experts][num_ranks][num_tokens][hidden]
   *    packed_recv_src_info: [num_local_experts][num_ranks][num_tokens]
   * packed_recv_layout_range: [num_local_experts][num_ranks]
   */
  void print_packed_recv_buffers(int num_local_experts) {
    T* packed_recv_x_t = reinterpret_cast<T*>(packed_recv_x);
    int* packed_recv_src_info_t =
        reinterpret_cast<int*>(packed_recv_src_info);
    for (int e = 0; e < num_local_experts; e++) {
      std::cout << "Expert " << (rank * num_local_experts + e)
                << "[local expert " << e << "]:\n";
      for (int r = 0; r < num_ranks; r++) {
        // Starting offset and number of tokens received from rank r
        std::cout << "  From Rank " << r << ":\n";
        int64_t *layout_range = reinterpret_cast<int64_t*>(
            packed_recv_layout_range + (e * num_ranks + r));
        int num_tokens_from_r = reinterpret_cast<int*>(
            layout_range)[0];
        int offset_for_r = reinterpret_cast<int*>(
            layout_range)[1];
        std::cout << "    Num tokens: " << num_tokens_from_r
                  << ", Offset: " << offset_for_r
                  << ", packed: " << *layout_range
                  << ", addr: " << layout_range <<"\n";
        for (int t = 0; t < num_tokens_from_r; t++) {
          int offset = e * num_ranks * num_tokens * hidden +
                       (offset_for_r + t) * hidden;
          std::cout << "      Token " << t << ": (Index: "
                    << packed_recv_src_info_t[e * num_ranks * num_tokens +
                                              (offset_for_r + t)]
                    << ") " << " Addr: " << &packed_recv_x_t[offset] << " ";
          for (int h = 0; h < hidden; h++) {
            std::cout << packed_recv_x_t[offset + h] << " ";
          }
          std::cout << std::endl;
        }
      }
    }
  }

  /**
   * Function to print number of tokens received by each expert
   * Dimensions: [num_local_experts][num_ranks]
   */
  void print_dispatch_recv_counts(int num_local_experts,
      void* recv_count_buffer) {
    int64_t* recv_count = reinterpret_cast<int64_t*>(recv_count_buffer);
    for (int e = 0; e < num_local_experts; e++) {
      std::cout << "Expert " << (rank * num_local_experts + e) << ":\n";
      for (int r = 0; r < num_ranks; r++) {
        std::cout << "  From Rank " << r << ": "
                  << -recv_count[e * num_ranks + r] - 1
                  << std::endl;
      }
    }
  }

  /**
   * Function to print rdma_x of dimensions [num_tokens][int + hidden]
   * for debugging purposes.
   */
  void print_rdma_x(void* rdma_x) {
    T* rdma_x_t = reinterpret_cast<T*>(rdma_x);
    for (int i = 0; i < num_tokens; i++) {
      std::cout << "Token " << i << ": ";
      int token_idx = *(reinterpret_cast<int*>(
          reinterpret_cast<uint8_t*>(rdma_x) + i * (sizeof(int) +
          hidden * sizeof(T))));
      int *token_addr = reinterpret_cast<int*>(
          reinterpret_cast<uint8_t*>(rdma_x) + i * (sizeof(int) +
          hidden * sizeof(T)));
      std::cout << "[Addr: " << token_addr << "] ";
      std::cout << "(Index: " << token_idx << ") ";
      for (int j = 0; j < hidden; j++) {
        std::cout << rdma_x_t[i * (hidden + sizeof(int)/sizeof(T)) +
                              sizeof(int)/sizeof(T) + j] << " ";
      }
      std::cout << std::endl;
    }
  }

  /**
 * Function to print rdma_recv_x of dimensions
 * [num_local_experts][num_ranks][num_tokens][int + hidden]
 */
  void print_rdma_recv_x(void* rdma_recv_x, int num_local_experts) {

    // Dimensions:
    std::cout << "rdma_recv_x dimensions: ["
              << num_local_experts << "]["
              << num_ranks << "]["
              << num_tokens << "]["
              << sizeof(int) + hidden * sizeof(T) << "]\n";

    T* rdma_recv_x_t = reinterpret_cast<T*>(rdma_recv_x);
    const size_t msg_size = sizeof(int) + hidden * sizeof(T);
    for (int expert = 0; expert < num_local_experts; expert++) {
      for (int rank = 0; rank < num_ranks; rank++) {
        std::cout << "Expert " << expert << ", Rank " << rank << ":\n";
        for (int token = 0; token < num_tokens; token++) {
          int offset = expert * num_ranks * num_tokens * msg_size +
                      rank * num_tokens * msg_size +
                      token * msg_size;
          int token_idx = *(reinterpret_cast<int*>(
              reinterpret_cast<uint8_t*>(rdma_recv_x) + offset));
          // Print the address of the token index for debugging
          int *token_addr = reinterpret_cast<int*>(
              reinterpret_cast<uint8_t*>(rdma_recv_x) + offset);
          std::cout << "[Addr: " << token_addr << "] ";
          std::cout << "  Token " << token << ": (Index: " << token_idx << ") ";
          for (int j = 0; j < hidden; j++) {
            std::cout << rdma_recv_x_t[offset / sizeof(T) +
                         sizeof(int)/sizeof(T) + j] << " ";
          }
          std::cout << std::endl;
        }
      }
    }
  }
};