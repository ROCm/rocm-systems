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

#ifndef LIBRARY_SRC_MPI_INSTANCE_HPP_
#define LIBRARY_SRC_MPI_INSTANCE_HPP_

#include "rocshmem/rocshmem_config.h"
#include "rocshmem/rocshmem_mpi.hpp"
#include <memory>

/**
 * @file mpi_instance.hpp
 *
 * @brief Contains MPI library initialization code
 */

namespace rocshmem {

struct mpilib_funcs_t {
  int (*Init_thread)(int *argc, char ***argv, int required, int *provided);
  int (*Initialized)(int *flag);
  int (*Finalize)(void);
  int (*Finalized)(int *flag);
  int (*Comm_rank)(MPI_Comm comm, int *rank);
  int (*Comm_size)(MPI_Comm comm, int *size);
  int (*Abort)(MPI_Comm comm, int errorcode);
  int (*Get_address)(const void *location, MPI_Aint *address);
  int (*Type_size)(MPI_Datatype type, int *size);
  int (*Iprobe)(int source, int tag, MPI_Comm comm, int *flag, MPI_Status *status);
  int (*Testsome)(int incount, MPI_Request array_of_requests[], int *outcount, int array_of_indices[],
                  MPI_Status array_of_statuses[]);
  int (*Comm_split)(MPI_Comm comm, int color, int key, MPI_Comm *newcomm);
  int (*Comm_split_type)(MPI_Comm comm, int split_type, int key, MPI_Info info, MPI_Comm *newcomm);
  int (*Comm_group)(MPI_Comm comm, MPI_Group *group);
  int (*Comm_create_group)(MPI_Comm comm, MPI_Group group, int tag, MPI_Comm *newcomm);
  int (*Comm_dup)(MPI_Comm comm, MPI_Comm *newcomm);
  int (*Comm_free)(MPI_Comm *comm);
  int (*Group_free)(MPI_Group *group);
  int (*Group_translate_ranks)(MPI_Group group1, int n, const int ranks1[], MPI_Group group2, int ranks2[]);
  int (*Group_incl)(MPI_Group group, int n, const int ranks[], MPI_Group *newgroup);
  int (*Allgather)(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
  int (*Allreduce)(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                   MPI_Op op, MPI_Comm comm);
  int (*Alltoall)(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount,
                  MPI_Datatype recvtype, MPI_Comm comm);
  int (*Bcast)(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
  int (*Barrier)(MPI_Comm comm);
  int (*Iallreduce)(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                    MPI_Op op, MPI_Comm comm, MPI_Request *request);
  int (*Ibarrier)(MPI_Comm comm, MPI_Request *request);
  int (*Win_create)(void *base, MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, MPI_Win *win);
  int (*Win_free)(MPI_Win *win);
  int (*Win_flush)(int rank, MPI_Win win);
  int (*Win_flush_all)(MPI_Win win);
  int (*Win_flush_local)(int rank, MPI_Win win);
  int (*Win_lock)(int lock_type, int rank, int mpi_assert, MPI_Win win);
  int (*Win_lock_all)(int mpi_assert, MPI_Win win);
  int (*Win_sync)(MPI_Win win);
  int (*Win_unlock)(int rank, MPI_Win win);
  int (*Win_unlock_all)(MPI_Win win);
  int (*Get)(void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank,
             MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win);
  int (*Rget)(void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp,
              int target_count, MPI_Datatype target_datatype,  MPI_Win win, MPI_Request *request);
  int (*Put)(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp,
             int target_count, MPI_Datatype target_datatype, MPI_Win win);
  int (*Rput)(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp,
              int target_cout, MPI_Datatype target_datatype, MPI_Win win, MPI_Request *request);
  int (*Compare_and_swap)(const void *origin_addr, const void *compare_addr, void *result_addr, MPI_Datatype datatype, int target_rank,
                          MPI_Aint target_disp, MPI_Win win);
  int (*Fetch_and_op)(const void *origin_addr, void *result_addr, MPI_Datatype datatype,
                      int target_rank, MPI_Aint target_disp, MPI_Op op, MPI_Win win);
};
extern struct mpilib_funcs_t mpilib_ftable_;

class MPIInstance {
  public:
    /**
     * @brief Primary constructor
     */
    MPIInstance(MPI_Comm comm);

    /**
     * @brief Destructor
     */
    ~MPIInstance();

    /**
     * @brief Accessor for my COMM_WORLD rank identifier
     *
     * @return My COMM_WORLD rank identifier
     */
    int get_rank();

    /**
     * @brief Accessor for number or processes in COMM_WORLD
     *
     * @return Number of processes in COMM_WORLD
     */
    int get_nprocs();

    /**
     * @brief dlopen the MPI library and set
     *        function pointers.
     * @return ROCSHMEM_SUCCESS on success,
     *         ROCSHMEM_ERROR otherwise.
     */
    static int mpilib_dl_init(void);

    /**
     * @brief dlclose the MPI library
     */
    static void mpilib_dl_close(void);

  private:
    /**
     * @brief My MPI rank identifier
     */
    int my_rank_{-1};

    /**
     * @brief Number of MPI processes
     */
    int nprocs_{-1};

    /**
     * @brief Was MPI initialized in this class
     */
    int init_in_this_class{0};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_MPI_INSTANCE_HPP_
