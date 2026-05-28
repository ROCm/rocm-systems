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

#ifndef LIBRARY_SRC_HOST_HOST_TEMPLATES_HPP_
#define LIBRARY_SRC_HOST_HOST_TEMPLATES_HPP_

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "host_helpers.hpp"
#include "memory/window_info.hpp"
#include "team.hpp"

#include <utility>
#include <cassert>

namespace rocshmem {

template <typename T>
__host__ void HostInterface::p(T* dest, T value, int pe,
                               WindowInfo* window_info) {
  DPRINTF("Function: host_p\n");
  putmem(dest, &value, sizeof(T), pe, window_info);
}

template <typename T>
__host__ void HostInterface::put(T* dest, const T* source, size_t nelems,
                                 int pe, WindowInfo* window_info) {
  DPRINTF("Function: host_put\n");
  putmem(dest, source, sizeof(T) * nelems, pe, window_info);
}

template <typename T>
__host__ void HostInterface::put_nbi(T* dest, const T* source, size_t nelems,
                                     int pe, WindowInfo* window_info) {
  DPRINTF("Function: host_put_nbi\n");
  putmem_nbi(dest, source, sizeof(T) * nelems, pe, window_info);
}

template <typename T>
__host__ T HostInterface::g(const T* source, int pe, WindowInfo* window_info) {
  WindowInfoMPI* window_info_mpi = dynamic_cast<WindowInfoMPI*>(window_info);
  if (!window_info_mpi) {
    abort();
  }
  DPRINTF("Function: host_g\n");

  T ret{};

  /*
   * We don't call getmem directly here
   * since it flushes the local HDP. We
   * don't need the flush because the
   * destination buffer is on the CPU.
   */
  getmem_nbi(&ret, source, sizeof(T), pe, window_info);

  mpilib_ftable_.Win_flush_local(pe, window_info_mpi->get_win());

  return ret;
}

template <typename T>
__host__ void HostInterface::get(T* dest, const T* source, size_t nelems,
                                 int pe, WindowInfo* window_info) {
  DPRINTF("Function: host_get\n");
  getmem(dest, source, sizeof(T) * nelems, pe, window_info);
}

template <typename T>
__host__ void HostInterface::get_nbi(T* dest, const T* source, size_t nelems,
                                     int pe, WindowInfo* window_info) {
  DPRINTF("Function: host_get_nbi\n");
  getmem_nbi(dest, source, sizeof(T) * nelems, pe, window_info);
}

__host__ MPI_Comm HostInterface::get_mpi_comm(int pe_start, int log_pe_stride,
                                              int pe_size) {
  MPI_Comm active_set_comm{};

  /*
   * First, check to see if the active set is the same as COMM_WORLD
   */
  int comm_world_size{-1};
  mpilib_ftable_.Comm_size(host_comm_world_, &comm_world_size);

  if (pe_start == 0 && log_pe_stride == 0 && pe_size == comm_world_size) {
    /*
     * Use the host interface's copy of MPI_COMM_WORLD
     * TODO: replace with a per-context copy of MPI_COMM_WORLD when we
     * have multiple contexts
     */
    active_set_comm = host_comm_world_;
    return active_set_comm;
  }

  /*
   * Then, check to see if we had already created a communicator for
   * this active set
   */
  ActiveSetKey key(pe_start, log_pe_stride, pe_size);

  auto it{comm_map.find(key)};
  if (it != comm_map.end()) {
    DPRINTF("Using cached communicator\n");
    return it->second;
  }

  /*
   * If there is not one cached, create a new one (expensive)
   */
  std::vector<int> active_set_ranks(pe_size);
  int stride{1 << log_pe_stride};
  active_set_ranks[0] = pe_start;

  for (int i{1}; i < pe_size; i++) {
    active_set_ranks[i] = active_set_ranks[i - 1] + stride;
  }

  MPI_Group comm_world_group{};
  MPI_Group active_set_group{};

  mpilib_ftable_.Comm_group(host_comm_world_, &comm_world_group);

  mpilib_ftable_.Group_incl(comm_world_group, pe_size, active_set_ranks.data(),
                            &active_set_group);

  mpilib_ftable_.Comm_create_group(host_comm_world_, active_set_group, 0,
                        &active_set_comm);

  /*
   * Cache the new communicator
   */
  DPRINTF("Created a new communicator. Now caching it\n");
  comm_map.insert(std::pair<ActiveSetKey, MPI_Comm>(key, active_set_comm));

  return active_set_comm;
}

template <typename T>
__host__ void HostInterface::broadcast_internal(MPI_Comm mpi_comm, T* dest,
                                                const T* source, int nelems,
                                                int pe_root) {
  DPRINTF("Function: host_broadcast_internal\n");

  /*
   * Choose the right pointer for my buffer depending
   * on whether or not I am the root.
   */
  int active_set_rank{-1};
  void* buffer{nullptr};
  mpilib_ftable_.Comm_rank(mpi_comm, &active_set_rank);
  if (pe_root == active_set_rank) {
    buffer = const_cast<T*>(source);
  } else {
    buffer = const_cast<T*>(dest);
  }

  /*
   * Flush my HDP so that the NIC does not read stale values
   */
  hdp_policy_->hdp_flush();

  /*
   * Offload the broadcast to MPI
   */
  mpilib_ftable_.Bcast(buffer, nelems * sizeof(T), MPI_CHAR, pe_root, mpi_comm);

  return;
}

template <typename T>
__host__ void HostInterface::broadcast(T* dest, const T* source, int nelems,
                                       int pe_root, int pe_start,
                                       int log_pe_stride, int pe_size,
                                       [[maybe_unused]] long* p_sync) {
  DPRINTF("Function: host_broadcast\n");

  /*
   * Get an MPI communicator for active set of PEs
   * Note: pe_root is w.r.t the active set, hence
   * the MPI communicator contains the root as well.
   */
  MPI_Comm mpi_comm{get_mpi_comm(pe_start, log_pe_stride, pe_size)};

  broadcast_internal<T>(mpi_comm, dest, source, nelems, pe_root);

  return;
}

template <typename T>
__host__ void HostInterface::broadcast(rocshmem_team_t team, T* dest,
                                       const T* source, int nelems,
                                       int pe_root) {
  DPRINTF("Function: Team-based host_broadcast\n");

  /*
   * Get the MPI communicator of this team
   */
  Team* team_obj{get_internal_team(team)};
  MPI_Comm mpi_comm{team_obj->mpi_comm};

  broadcast_internal<T>(mpi_comm, dest, source, nelems, pe_root);

  return;
}

__host__ inline MPI_Op HostInterface::get_mpi_op(ROCSHMEM_OP Op) {
  switch (Op) {
    case ROCSHMEM_SUM:
      return MPI_SUM;
    case ROCSHMEM_MAX:
      return MPI_MAX;
    case ROCSHMEM_MIN:
      return MPI_MIN;
    case ROCSHMEM_PROD:
      return MPI_PROD;
    case ROCSHMEM_AND:
      return MPI_BAND;
    case ROCSHMEM_OR:
      return MPI_BOR;
    case ROCSHMEM_XOR:
      return MPI_BXOR;
    default:
      fprintf(stderr, "Unknown rocSHMEM op MPI conversion %d\n", Op);
      abort();
      return 0;
  }
}

template <typename T>
__host__ inline MPI_Datatype HostInterface::get_mpi_type() {
  fprintf(stderr, "Unknown or unimplemented datatype \n");
}

#define GET_MPI_TYPE(T, MPI_T)                                    \
  template <>                                                     \
  __host__ inline MPI_Datatype HostInterface::get_mpi_type<T>() { \
    return MPI_T;                                                 \
  }

GET_MPI_TYPE(int, MPI_INT)
GET_MPI_TYPE(unsigned int, MPI_UNSIGNED)
GET_MPI_TYPE(short, MPI_SHORT)
GET_MPI_TYPE(unsigned short, MPI_UNSIGNED_SHORT)
GET_MPI_TYPE(long, MPI_LONG)
GET_MPI_TYPE(unsigned long, MPI_UNSIGNED_LONG)
GET_MPI_TYPE(long long, MPI_LONG_LONG)
GET_MPI_TYPE(unsigned long long, MPI_UNSIGNED_LONG_LONG)
GET_MPI_TYPE(float, MPI_FLOAT)
GET_MPI_TYPE(double, MPI_DOUBLE)
GET_MPI_TYPE(char, MPI_CHAR)
GET_MPI_TYPE(signed char, MPI_SIGNED_CHAR)
GET_MPI_TYPE(unsigned char, MPI_UNSIGNED_CHAR)

template <typename T>
__host__ void HostInterface::amo_add(void* dst, T value, int pe,
                                     WindowInfo* window_info) {
  /*
   * Most MPI implementations tend to use active messages to implement
   * MPI_Accumulate. So, to eliminate the potential involvement of the
   * target PE, we instead use fetch_add and disregard the return value.
   */
  [[maybe_unused]] T ret{amo_fetch_add(dst, value, pe, window_info)};
}

template <typename T>
__host__ void HostInterface::amo_cas(void* dst, T value, T cond, int pe,
                                     WindowInfo* window_info) {
  /* Perform the compare and swap and disregard the return value */
  [[maybe_unused]] T ret{amo_fetch_cas(dst, value, cond, pe, window_info)};
}

template <typename T>
__host__ T HostInterface::amo_fetch_add(void* dst, T value, int pe,
                                        WindowInfo* window_info) {
  WindowInfoMPI* window_info_mpi = dynamic_cast<WindowInfoMPI*>(window_info);
  if (!window_info_mpi) {
    abort();
  }

  /* Calculate offset of remote dest from base address of window */
  MPI_Aint offset{
      compute_offset(dst, window_info->get_start(), window_info->get_end())};

  /*
   * Flush the HDP of the remote PE so that the NIC does not
   * read stale values
   */
  flush_remote_hdp(pe);

  /* Offload remote fetch and op operation to MPI */
  T ret{};
  MPI_Win win{window_info_mpi->get_win()};
  MPI_Datatype mpi_type{get_mpi_type<T>()};
  mpilib_ftable_.Fetch_and_op(&value, &ret, mpi_type, pe, offset, MPI_SUM, win);

  mpilib_ftable_.Win_flush_local(pe, win);

  return ret;
}

template <typename T>
__host__ T HostInterface::amo_fetch_cas(void* dst, T value, T cond, int pe,
                                        WindowInfo* window_info) {
  WindowInfoMPI* window_info_mpi = dynamic_cast<WindowInfoMPI*>(window_info);
  if (!window_info_mpi) {
    abort();
  }

  /* Calculate offset of remote dest from base address of window */
  MPI_Aint offset{
      compute_offset(dst, window_info->get_start(), window_info->get_end())};

  /*
   * Flush the HDP of the remote PE so that the NIC does not
   * read stale values
   */
  flush_remote_hdp(pe);

  /* Offload remote compare and swap operation to MPI */
  T ret{};
  MPI_Win win{window_info_mpi->get_win()};
  MPI_Datatype mpi_type{get_mpi_type<T>()};
  mpilib_ftable_.Compare_and_swap(&value, &cond, &ret, mpi_type, pe, offset, win);

  mpilib_ftable_.Win_flush_local(pe, win);

  return ret;
}

template <typename T, ROCSHMEM_OP Op>
__host__ void HostInterface::to_all_internal(MPI_Comm mpi_comm, T* dest,
                                             const T* source, int nreduce) {
  DPRINTF("Function: host_to_all_internal\n");

  MPI_Op mpi_op{get_mpi_op(Op)};

  MPI_Datatype mpi_type{get_mpi_type<T>()};

  void* send_buf{const_cast<T*>(source)};
  void* recv_buf{const_cast<T*>(dest)};

  /*
   * Flush my HDP so that the NIC does not read stale values
   */
  hdp_policy_->hdp_flush();

  /*
   * Offload the allreduce to MPI
   */
  mpilib_ftable_.Allreduce((dest == source) ? MPI_IN_PLACE : send_buf, recv_buf, nreduce,
                           mpi_type, mpi_op, mpi_comm);

  return;
}

template <typename T, ROCSHMEM_OP Op>
__host__ void HostInterface::to_all(T* dest, const T* source, int nreduce,
                                    int pe_start, int log_pe_stride,
                                    int pe_size, [[maybe_unused]] T* p_wrk,
                                    [[maybe_unused]] long* p_sync) {
  DPRINTF("Function: host_to_all\n");

  /*
   * Get an MPI communicator for active set of PEs
   * Note: pe_root is w.r.t. the active set, hence
   * the MPI communicator contains the root as well.
   */
  MPI_Comm mpi_comm{get_mpi_comm(pe_start, log_pe_stride, pe_size)};

  to_all_internal<T, Op>(mpi_comm, dest, source, nreduce);

  return;
}

template <typename T, ROCSHMEM_OP Op>
__host__ int HostInterface::reduce(rocshmem_team_t team, T* dest,
                                    const T* source, int nreduce) {
  DPRINTF("Function: Team-based host_reduce\n");

  /*
   * Get the MPI communicator of this team
   */
  Team* team_obj{get_internal_team(team)};
  MPI_Comm mpi_comm{team_obj->mpi_comm};

  to_all_internal<T, Op>(mpi_comm, dest, source, nreduce);

  return ROCSHMEM_SUCCESS;
}

template <typename T>
__host__ inline int HostInterface::compare(int cmp, T input_val,
                                           T target_val) {
  int cond_satisfied{0};

  switch (cmp) {
    case ROCSHMEM_CMP_EQ:
      cond_satisfied = (input_val == target_val) ? 1 : 0;
      break;
    case ROCSHMEM_CMP_NE:
      cond_satisfied = (input_val != target_val) ? 1 : 0;
      break;
    case ROCSHMEM_CMP_GT:
      cond_satisfied = (input_val > target_val) ? 1 : 0;
      break;
    case ROCSHMEM_CMP_GE:
      cond_satisfied = (input_val >= target_val) ? 1 : 0;
      break;
    case ROCSHMEM_CMP_LT:
      cond_satisfied = (input_val < target_val) ? 1 : 0;
      break;
    case ROCSHMEM_CMP_LE:
      cond_satisfied = (input_val <= target_val) ? 1 : 0;
      break;
    default:
      assert(cmp >= ROCSHMEM_CMP_EQ && cmp <= ROCSHMEM_CMP_LE);
      break;
  }

  return cond_satisfied;
}

template <typename T>
__host__ inline int HostInterface::test_and_compare(MPI_Aint offset,
                                                    MPI_Datatype mpi_type,
                                                    int cmp, T val,
                                                    MPI_Win win) {
  T fetched_val{};

  /*
   * Flush the HDP so that the CPU doesn't read stale values
   */
  hdp_policy_->hdp_flush();

  mpilib_ftable_.Fetch_and_op(nullptr,  // because no operation happening here
                              &fetched_val, mpi_type, my_pe_, offset, MPI_NO_OP, win);
  mpilib_ftable_.Win_flush_local(my_pe_, win);

  /*
   * Compare based on the operation
   */
  return compare(cmp, fetched_val, val);
}

template <typename T>
__host__ void HostInterface::wait_until(T *ivars, int cmp, T val,
                                        WindowInfo* window_info) {
  WindowInfoMPI* window_info_mpi = dynamic_cast<WindowInfoMPI*>(window_info);
  if (!window_info_mpi) {
    abort();
  }
  DPRINTF("Function: host_wait_until\n");

  /*
   * Find the offset of this memory in the window
   */
  MPI_Aint offset{
      compute_offset(ivars, window_info->get_start(), window_info->get_end())};

  MPI_Datatype mpi_type{get_mpi_type<T>()};
  MPI_Win win{window_info_mpi->get_win()};

  /*
   * Continuously read the ivars atomically until it satisfies the condition
   */
  while (1) {
    int cond_satisfied{test_and_compare(offset, mpi_type, cmp, val, win)};

    if (cond_satisfied) {
      break;
    }
  }
}

__host__ size_t status_entry(size_t nelems,
                             const int *status,
                             bool* done_flags) {
  size_t i{0};
  size_t pos{SIZE_MAX};
  while (i < nelems) {
    if (status[i]) {
      done_flags[i] = 1;
    } else {
      pos = min(i, pos);
    }
    i++;
  }
  return pos;
}

__host__ size_t status_entry(size_t nelems,
                             const int *status) {
  size_t i{0};
  while (i < nelems) {
    if (status[i] == 0) {
      return i;
    }
    i++;
  }
  return i;
}

template <typename T>
__host__ size_t HostInterface::wait_until_any(T* ivars, size_t nelems,
                                              const int *status,
                                              int cmp, T val,
                                              WindowInfo* window_info) {
  DPRINTF("Function: host_wait_until_any\n");

  // zero nelems error condition
  if (!nelems) {
    return SIZE_MAX;
  }

  size_t pos{status_entry(nelems, status)};

  // invalid (empty) status array error condition
  if (pos == nelems) {
    return SIZE_MAX;
  }

  while (true) {
    for (size_t i{pos}; i < nelems; i++) {
      // skip entries marked with non-zero status
      if (status[i]) {
        continue;
      }
      if (test(ivars + i, cmp, val, window_info)) {
        return i;
      }
    }
  }
}

template <typename T>
__host__ void HostInterface::wait_until_all(T* ivars, size_t nelems,
                                            const int *status,
                                            int cmp, T val,
                                            WindowInfo* window_info) {
  DPRINTF("Function: host_wait_until_all\n");

  // zero nelems error condition
  if (!nelems) {
    return;
  }

  size_t pos{status_entry(nelems, status)};

  // invalid (empty) status array error condition
  if (pos == nelems) {
    return;
  }

  for (size_t i{pos}; i < nelems; i++) {
    if (status[i]) {
      continue;
    }
    while (!test(ivars + i, cmp, val, window_info)) {
    }
  }
}

template <typename T>
__host__ size_t HostInterface::wait_until_some(T* ivars, size_t nelems,
                                             size_t* indices,
                                             const int *status,
                                             int cmp, T val,
                                             WindowInfo* window_info) {
  DPRINTF("Function: host_wait_until_some\n");

  // zero nelems error condition
  if (!nelems) {
    return 0;
  }

  size_t pos{status_entry(nelems, status)};

  // invalid (empty) status array error condition
  if (pos == nelems) {
    return 0;
  }

  bool done {false};
  size_t ncompleted {0};
  while (!done) {
    for (size_t i{pos}; i < nelems; i++) {
      // skip entries marked with non-zero status
      if (status[i]) {
        continue;
      }
      if (test(ivars + i, cmp, val, window_info)) {
        done = true;
        indices[ncompleted] = i;
        ncompleted++;
      }
    }
  }
  return ncompleted;
}

template <typename T>
__host__ void HostInterface::wait_until_all_vector(T* ivars, size_t nelems,
                                                   const int *status,
                                                   int cmp, T* vals,
                                                   WindowInfo* window_info) {
  DPRINTF("Function: host_wait_until_all_vector\n");
}

template <typename T>
__host__ size_t HostInterface::wait_until_any_vector(T* ivars, size_t nelems,
                                                     const int *status,
                                                     int cmp, T* vals,
                                                     WindowInfo* window_info) {
  DPRINTF("Function: host_wait_until_any_vector\n");
  return 0;
}

template <typename T>
__host__ size_t HostInterface::wait_until_some_vector(T* ivars, size_t nelems,
                                                    size_t* indices,
                                                    const int *status,
                                                    int cmp, T* vals,
                                                    WindowInfo* window_info) {
  DPRINTF("Function: host_wait_until_some_vector\n");
  return 0;
}

template <typename T>
__host__ int HostInterface::test(T* ivars, int cmp, T val,
                                 WindowInfo* window_info) {
  WindowInfoMPI* window_info_mpi = dynamic_cast<WindowInfoMPI*>(window_info);
  if (!window_info_mpi) {
    abort();
  }
  DPRINTF("Function: host_test\n");

  /*
   * Find the offset of this memory in the window
   */
  MPI_Aint offset{
      compute_offset(ivars, window_info->get_start(), window_info->get_end())};

  MPI_Datatype mpi_type{get_mpi_type<T>()};

  return test_and_compare(offset, mpi_type, cmp, val, window_info_mpi->get_win());
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_HOST_HOST_TEMPLATES_HPP_
