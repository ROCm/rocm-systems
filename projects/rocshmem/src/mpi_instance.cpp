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

#include <dlfcn.h>

#include "rocshmem/rocshmem.hpp"
#include "mpi_instance.hpp"
#include "util.hpp"

#if !defined(HAVE_EXTERNAL_MPI)
// Open MPI specific symbols
struct ompi_internal_symbols_t ompi_symbols_;
#endif //!defined(HAVE_EXTERNAL_MPI)

namespace rocshmem {

static void* mpilib_handle_{nullptr};
struct mpilib_funcs_t mpilib_ftable_;

int MPIInstance::mpilib_dl_init() {
  if (mpilib_handle_ != nullptr)
      return ROCSHMEM_SUCCESS;

  mpilib_handle_ = dlopen("libmpi.so", RTLD_LAZY);
  if (!mpilib_handle_) {
    printf("Could not open libmpi.so. Returning\n");
    return ROCSHMEM_ERROR;
  }

  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Init_thread);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Initialized);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Finalize);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Finalized);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Comm_rank);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Comm_size);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Abort);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Get_address);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Type_size);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Iprobe);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Testsome);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Comm_split);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Comm_split_type);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Comm_group);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Comm_create_group);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Comm_dup);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Comm_free);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Group_free);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Group_translate_ranks);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Group_incl);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Allgather);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Alltoall);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Allreduce);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Bcast);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Barrier);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Iallreduce);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Ibarrier);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Win_create);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Win_free);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Win_flush);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Win_flush_all);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Win_flush_local);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Win_lock);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Win_lock_all);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Win_unlock);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Win_unlock_all);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Win_lock);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Win_sync);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Get);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Rget);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Put);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Rput);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Compare_and_swap);
  DLSYM_HELPER(mpilib_ftable_, MPI_, mpilib_handle_, Fetch_and_op);

#if !defined(HAVE_EXTERNAL_MPI)
  DLSYM_VAR_HELPER(ompi_symbols_, mpilib_handle_, ompi_mpi_comm_world);
  DLSYM_VAR_HELPER(ompi_symbols_, mpilib_handle_, ompi_mpi_comm_null);
  DLSYM_VAR_HELPER(ompi_symbols_, mpilib_handle_, ompi_mpi_datatype_null);
  DLSYM_VAR_HELPER(ompi_symbols_, mpilib_handle_, ompi_request_null);
  DLSYM_VAR_HELPER(ompi_symbols_, mpilib_handle_, ompi_mpi_info_null);

  DLSYM_VAR_HELPER(ompi_symbols_, mpilib_handle_, ompi_mpi_op_max);
  DLSYM_VAR_HELPER(ompi_symbols_, mpilib_handle_, ompi_mpi_op_min);
  DLSYM_VAR_HELPER(ompi_symbols_, mpilib_handle_, ompi_mpi_op_sum);
  DLSYM_VAR_HELPER(ompi_symbols_, mpilib_handle_, ompi_mpi_op_prod);
  DLSYM_VAR_HELPER(ompi_symbols_, mpilib_handle_, ompi_mpi_op_band);
  DLSYM_VAR_HELPER(ompi_symbols_, mpilib_handle_, ompi_mpi_op_bor);
  DLSYM_VAR_HELPER(ompi_symbols_, mpilib_handle_, ompi_mpi_op_bxor);
  DLSYM_VAR_HELPER(ompi_symbols_, mpilib_handle_, ompi_mpi_op_replace);
  DLSYM_VAR_HELPER(ompi_symbols_, mpilib_handle_, ompi_mpi_op_no_op);

  DLSYM_VAR_HELPER(ompi_symbols_, mpilib_handle_, ompi_mpi_char);
  DLSYM_VAR_HELPER(ompi_symbols_, mpilib_handle_, ompi_mpi_unsigned_char);
  DLSYM_VAR_HELPER(ompi_symbols_, mpilib_handle_, ompi_mpi_signed_char);
  DLSYM_VAR_HELPER(ompi_symbols_, mpilib_handle_, ompi_mpi_short);
  DLSYM_VAR_HELPER(ompi_symbols_, mpilib_handle_, ompi_mpi_unsigned_short);
  DLSYM_VAR_HELPER(ompi_symbols_, mpilib_handle_, ompi_mpi_int);
  DLSYM_VAR_HELPER(ompi_symbols_, mpilib_handle_, ompi_mpi_unsigned);
  DLSYM_VAR_HELPER(ompi_symbols_, mpilib_handle_, ompi_mpi_long);
  DLSYM_VAR_HELPER(ompi_symbols_, mpilib_handle_, ompi_mpi_unsigned_long);
  DLSYM_VAR_HELPER(ompi_symbols_, mpilib_handle_, ompi_mpi_long_long_int);
  DLSYM_VAR_HELPER(ompi_symbols_, mpilib_handle_, ompi_mpi_unsigned_long_long);
  DLSYM_VAR_HELPER(ompi_symbols_, mpilib_handle_, ompi_mpi_float);
  DLSYM_VAR_HELPER(ompi_symbols_, mpilib_handle_, ompi_mpi_double);
  DLSYM_VAR_HELPER(ompi_symbols_, mpilib_handle_, ompi_mpi_long_double);
#endif //!defined(HAVE_EXTERNAL_MPI)

  return ROCSHMEM_SUCCESS;
}

void MPIInstance::mpilib_dl_close() {
  if (mpilib_handle_ != nullptr) {
    dlclose(mpilib_handle_);
    mpilib_handle_ = nullptr;
  }
}

MPIInstance::MPIInstance(MPI_Comm comm) {
  int is_init{0};

  assert (nullptr != mpilib_handle_);

  mpilib_ftable_.Initialized(&is_init);

  if (!is_init) {
    int provided;
    mpilib_ftable_.Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    init_in_this_class = 1;
  }

  mpilib_ftable_.Comm_size(comm, &nprocs_);
  mpilib_ftable_.Comm_rank(comm, &my_rank_);
}

MPIInstance::~MPIInstance() {
  int finalized{0};
  mpilib_ftable_.Finalized(&finalized);
  if (!finalized && init_in_this_class) {
    mpilib_ftable_.Finalize();
  }
}

int MPIInstance::get_rank() { return my_rank_; }

int MPIInstance::get_nprocs() { return nprocs_; }

}  // namespace rocshmem
