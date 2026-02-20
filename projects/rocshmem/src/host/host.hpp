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

#ifndef LIBRARY_SRC_HOST_HOST_HPP_
#define LIBRARY_SRC_HOST_HOST_HPP_

/**
 * @file host.hpp
 * Defines the HostInterface class.
 *
 * The file contains the HostInterface class that defines all the
 * host-facing functions that will be used by all host contexts of
 * any backend type.
 */

#include <map>

#include "rocshmem/rocshmem.hpp"
#include "hdp_policy.hpp"
#include "memory/symmetric_heap.hpp"
#include "memory/window_info.hpp"
#include "bootstrap/bootstrap.hpp"
#include "mpi_instance.hpp"

namespace rocshmem {

class HostContextWindowInfo {
 public:
  /**
   * @brief Constructor with default members
   */
  HostContextWindowInfo() = default;

  /**
   * @brief Constructor with initialized members
   *
   * @param[in] team pointer used to track team info
   * @param[in] team_info information about participating PEs
   */
  HostContextWindowInfo(MPI_Comm comm_world, SymmetricHeap* heap);
  HostContextWindowInfo(SymmetricHeap* heap);

  /**
   * @brief Destructor
   */
  __host__ ~HostContextWindowInfo();

  /**
   * @brief Retrieve a pointer to the internal WindowInfo
   *
   * @return WindowInfo pointer
   */
  WindowInfo* get() { return window_info_; }

  /**
   * @brief Mark the window info as avaialable (not allocated)
   */
  void mark_avail() { avail_ = true; }

  /**
   * @brief Mark the window info as unavaialble (allocated)
   */
  void mark_unavail() { avail_ = false; }

  /**
   * @brief Check if the window info has been allocated
   */
  bool is_avail() { return avail_; }

 private:
  /**
   * @brief Flag to state whether or not this window is available to be assigned
   * to a Host Context
   */
  bool avail_{true};

  /**
   * @brief Pointer to the WindowInfo object that manages the MPI Window for
   * this context
   */
  WindowInfo* window_info_{nullptr};
};

class HostInterface {
 public:
  /**
   * @brief Primary constructors
   */
  __host__ HostInterface(HdpPolicy* hdp_policy, MPI_Comm rocshmem_comm,
                         SymmetricHeap* heap);

  __host__ HostInterface(HdpPolicy* hdp_policy, TcpBootstrap *bootstrap,
                         SymmetricHeap* heap);

  /**
   * @brief Destructor
   */
  __host__ ~HostInterface();

  /**
   * @brief Accessor for copy of comm world
   *
   * @return MPI_Comm containing host comm world
   */
  MPI_Comm get_comm_world() { return host_comm_world_; }

  /**
   * @brief Get a window context from the pool
   *
   * @return Pointer to the WindowInfo in the allocated one from the pool
   */
  WindowInfo* acquire_window_context();

  /**
   * @brief Return a window context back to the pool
   */
  void release_window_context(WindowInfo* window_info);

  /**************************************************************************
   ***************************** HOST FUNCTIONS *****************************
   *************************************************************************/
  template <typename T>
  __host__ void p(T* dest, T value, int pe, WindowInfo* window_info);

  template <typename T>
  __host__ T g(const T* source, int pe, WindowInfo* window_info);

  template <typename T>
  __host__ void put(T* dest, const T* source, size_t nelems, int pe,
                    WindowInfo* window_info);

  template <typename T>
  __host__ void get(T* dest, const T* source, size_t nelems, int pe,
                    WindowInfo* window_info);

  template <typename T>
  __host__ void put_nbi(T* dest, const T* source, size_t nelems, int pe,
                        WindowInfo* window_info);

  template <typename T>
  __host__ void get_nbi(T* dest, const T* source, size_t nelems, int pe,
                        WindowInfo* window_info);

  __host__ void putmem(void* dest, const void* source, size_t nelems, int pe,
                       WindowInfo* window_info);

  __host__ void getmem(void* dest, const void* source, size_t nelems, int pe,
                       WindowInfo* window_info);

  __host__ void putmem_nbi(void* dest, const void* source, size_t nelems,
                           int pe, WindowInfo* window_info);

  __host__ void getmem_nbi(void* dest, const void* source, size_t size, int pe,
                           WindowInfo* window_info);

  template <typename T>
  __host__ void amo_add(void* dst, T value, int pe, WindowInfo* window_info);

  template <typename T>
  __host__ void amo_cas(void* dst, T value, T cond, int pe,
                        WindowInfo* window_info);

  template <typename T>
  __host__ T amo_fetch_add(void* dst, T value, int pe, WindowInfo* window_info);

  template <typename T>
  __host__ T amo_fetch_cas(void* dst, T value, T cond, int pe,
                           WindowInfo* window_info);

  __host__ void fence(WindowInfo* window_info);

  __host__ void quiet(WindowInfo* window_info);

  __host__ void barrier_all(WindowInfo* window_info);
  
  __host__ void barrier_all_on_stream(hipStream_t stream);

  __host__ void alltoallmem_on_stream(rocshmem_team_t team, void *dest,
                                      const void *source, size_t size,
                                      hipStream_t stream);

  __host__ void broadcastmem_on_stream(rocshmem_team_t team, void *dest,
                                       const void *source, size_t nelems,
                                       int pe_root, hipStream_t stream);

  __host__ void getmem_on_stream(void *dest, const void *source, size_t nelems,
                                 int pe, hipStream_t stream);

  __host__ void putmem_on_stream(void *dest, const void *source, size_t nelems,
                                 int pe, hipStream_t stream);

  __host__ void putmem_signal_on_stream(void *dest, const void *source,
                                        size_t nelems, uint64_t *sig_addr,
                                        uint64_t signal, int sig_op, int pe,
                                        hipStream_t stream);

  __host__ void signal_wait_until_on_stream(uint64_t *sig_addr, int cmp,
                                            uint64_t cmp_value,
                                            hipStream_t stream);

  __host__ void barrier_for_sync();

  __host__ void sync_all(WindowInfo* window_info);

  template <typename T>
  __host__ void broadcast(T* dest, const T* source, int nelems, int pe_root,
                          int pe_start, int log_pe_stride, int pe_size,
                          long* p_sync);  // NOLINT(runtime/int)

  template <typename T>
  __host__ void broadcast(rocshmem_team_t team, T* dest, const T* source,
                          int nelems, int pe_root);

  template <typename T, ROCSHMEM_OP Op>
  __host__ void to_all(T* dest, const T* source, int nreduce, int pe_start,
                       int log_pe_stride, int pe_size, T* p_wrk,
                       long* p_sync);  // NOLINT(runtime/int)

  template <typename T, ROCSHMEM_OP Op>
  __host__ int reduce(rocshmem_team_t team, T* dest, const T* source, int nreduce);

  template <typename T>
  __host__ void wait_until(T *ivars, int cmp, T val,
                           WindowInfo* window_info);

  template <typename T>
  __host__ void wait_until_all(T *ivars, size_t nelems, const int* status,
                               int cmp, T val,
                               WindowInfo* window_info);

  template <typename T>
  __host__ size_t wait_until_any(T *ivars, size_t nelems, const int* status,
                                 int cmp, T val,
                                 WindowInfo* window_info);

  template <typename T>
  __host__ size_t wait_until_some(T *ivars, size_t nelems, size_t* indices,
                                  const int* status, int cmp, T val,
                                  WindowInfo* window_info);

  template <typename T>
  __host__ void wait_until_all_vector(T *ivars, size_t nelems, const int* status,
                                      int cmp, T* vals,
                                      WindowInfo* window_info);

  template <typename T>
  __host__ size_t wait_until_any_vector(T *ivars, size_t nelems,
                                        const int* status, int cmp,
                                        T* vals, WindowInfo* window_info);

  template <typename T>
  __host__ size_t wait_until_some_vector(T *ivars, size_t nelems, size_t* indices,
                                         const int* status, int cmp,
                                         T* vals, WindowInfo* window_info);

  template <typename T>
  __host__ int test(T *ivars, int cmp, T val, WindowInfo* window_info);

#if defined USE_HDP_FLUSH
  __host__ void create_hdp_window();
#endif // USE_HDP_FLUSH

 private:
  /**************************************************************************
   **************************** INTERNAL METHODS ****************************
   *************************************************************************/
  __host__ void flush_remote_hdps() {
#if defined USE_HDP_FLUSH
    unsigned flush_val{HdpPolicy::HDP_FLUSH_VAL};
    for (size_t i{0}; i < num_pes_; i++) {
      if (i == my_pe_) {
        continue;
      }
      mpilib_ftable_.Put(&flush_val, 1, MPI_UNSIGNED, i, 0, 1, MPI_UNSIGNED, hdp_win);
    }
    mpilib_ftable_.Win_flush_all(hdp_win);
#endif // USE_HDP_FLUSH
  }

  __host__ void flush_remote_hdp(int pe) {
#if defined USE_HDP_FLUSH
    unsigned flush_val{HdpPolicy::HDP_FLUSH_VAL};
    mpilib_ftable_.Put(&flush_val, 1, MPI_UNSIGNED, pe, 0, 1, MPI_UNSIGNED, hdp_win);
    mpilib_ftable_.Win_flush(pe, hdp_win);
#endif // USE_HDP_FLUSH
  }

  __host__ void initiate_put(void* dest, const void* source, size_t nelems,
                             int pe, WindowInfoMPI* window_info);

  __host__ void initiate_get(void* dest, const void* source, size_t nelems,
                             int pe, WindowInfoMPI* window_info);

  __host__ void complete_all(MPI_Win win);

  __host__ MPI_Aint compute_offset(const void* dest, void* win_start,
                                   void* win_end);

  __host__ MPI_Comm get_mpi_comm(int pe_start, int log_pe_stride, int pe_size);

  __host__ MPI_Op get_mpi_op(ROCSHMEM_OP Op);

  template <typename T>
  __host__ MPI_Datatype get_mpi_type();

  template <typename T>
  __host__ int compare(int cmp, T input_val, T target_val);

  template <typename T>
  __host__ int test_and_compare(MPI_Aint offset, MPI_Datatype mpi_type,
                                int cmp, T val, MPI_Win win);

  template <typename T, ROCSHMEM_OP Op>
  __host__ void to_all_internal(MPI_Comm mpi_comm, T* dest, const T* source,
                                int nreduce);

  template <typename T>
  __host__ void broadcast_internal(MPI_Comm mpi_comm, T* dest, const T* source,
                                   int nelems, int pe_root);

  /**************************************************************************
   **************************** INTERNAL MEMBERS ****************************
   *************************************************************************/
  /**
   * @brief Duplicate to the Backend's hdp policy pointer
   */
  HdpPolicy* hdp_policy_{nullptr};

  /**
   * @brief Global MPI communicator for those host API
   */
  MPI_Comm host_comm_world_{MPI_COMM_NULL};

  /**
   * @brief Bootstrap object used in the non-mpi workloads
   */
  TcpBootstrap *host_bootstrap_{nullptr};

  /**
   * @brief Duplicate of this processing element's id within global rank
   */
  int my_pe_{-1};

  /**
   * @brief Duplicate of global number of processing elements
   */
  int num_pes_{0};

#if defined USE_HDP_FLUSH
  /**
   * @brief MPI window for hdp flushing
   */
  MPI_Win hdp_win;
#endif  // USE_HDP_FLUSH

  /**
   * @brief Pool of HostContexWindowInfos
   */
  HostContextWindowInfo** host_window_context_pool_{nullptr};

  int find_win_info_in_pool(WindowInfo* window_info);

  int find_avail_pool_entry();

  /*
   * @brief Used by comm_map map for active sets.
   *
   * This data structure that stores the parameters defining the active
   * set of PEs in a collective. This struct also serves as a key
   * into the comm_map map.
   */
  class ActiveSetKey {
   public:
    /**
     * @brief Primary constructor
     */
    ActiveSetKey(int pe_start, int log_pe_stride, int pe_size)
        : pe_start_(pe_start),
          log_pe_stride_(log_pe_stride),
          pe_size_(pe_size) {}

    bool operator<(const ActiveSetKey& key) const {
      return pe_start_ < key.pe_start_ ||
             (pe_start_ == key.pe_start_ &&
              log_pe_stride_ < key.log_pe_stride_) ||
             (pe_start_ == key.pe_start_ &&
              log_pe_stride_ == key.log_pe_stride_ && pe_size_ < key.pe_size_);
    }

   private:
    /**
     * @brief Records start location in (logical) active set bitmask
     */
    int pe_start_{-1};

    /**
     * @brief Records stride in (logical) active set bitmask
     */
    int log_pe_stride_{-1};

    /**
     * @brief Records (logical) active set bitmask size
     */
    int pe_size_{-1};
  };

  /*
   * @brief Map of active set descriptors to MPI communicators
   */
  std::map<ActiveSetKey, MPI_Comm> comm_map{};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_HOST_HOST_HPP_
