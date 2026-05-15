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

#ifndef LIBRARY_INCLUDE_ROCSHMEM_MPI_HPP
#define LIBRARY_INCLUDE_ROCSHMEM_MPI_HPP

#if defined(HAVE_EXTERNAL_MPI)
#include <mpi.h>
#endif

#if defined(c_plusplus) || defined(__cplusplus)
extern "C" {
#endif

#if !defined(MPI_VERSION)
// Open MPI based values for the constants/handles etc.
// Even though we did not include an external MPI header file
// The includer may have (e.g., a unit test).

typedef void* MPI_Comm;
typedef void* MPI_Win;
typedef void* MPI_Group;
typedef void* MPI_Op;
typedef void* MPI_Datatype;
typedef void* MPI_Request;
typedef void* MPI_Info;

struct ompi_status_public_t {
    int MPI_SOURCE;
    int MPI_TAG;
    int MPI_ERROR;
    int _cancelled;
    size_t _ucount;
};
typedef struct ompi_status_public_t MPI_Status;

#define MPI_Aint     uint64_t

#define MPI_UNDEFINED -32766
#define MPI_THREAD_MULTIPLE 3
#define MPI_SUCCESS 0
#define MPI_IN_PLACE (void*)1
#define MPI_MODE_NOCHECK 1
#define MPI_COMM_TYPE_SHARED 0
#define MPI_ANY_SOURCE -1
#define MPI_STATUSES_IGNORE (static_cast<MPI_Status*>(0))

#define MPI_Aint_diff(addr1, addr2) ((MPI_Aint) ((char *) (addr1) - (char *) (addr2)))

struct ompi_internal_symbols_t  {
  void *ompi_mpi_comm_world;
  void *ompi_mpi_comm_null;
  void *ompi_request_null;
  void *ompi_mpi_info_null;
  void *ompi_mpi_datatype_null;

  void *ompi_mpi_op_max;
  void *ompi_mpi_op_min;
  void *ompi_mpi_op_sum;
  void *ompi_mpi_op_prod;
  void *ompi_mpi_op_band;
  void *ompi_mpi_op_bor;
  void *ompi_mpi_op_bxor;
  void *ompi_mpi_op_replace;
  void *ompi_mpi_op_no_op;

  void *ompi_mpi_char;
  void *ompi_mpi_unsigned_char;
  void *ompi_mpi_signed_char;
  void *ompi_mpi_short;
  void *ompi_mpi_unsigned_short;
  void *ompi_mpi_int;
  void *ompi_mpi_unsigned;
  void *ompi_mpi_long;
  void *ompi_mpi_unsigned_long;
  void *ompi_mpi_long_long_int;
  void *ompi_mpi_unsigned_long_long;
  void *ompi_mpi_float;
  void *ompi_mpi_double;
  void *ompi_mpi_long_double;
};

extern struct ompi_internal_symbols_t ompi_symbols_;

#define OMPI_PREDEFINED_GLOBAL(type, global) (static_cast<type> (global))
#define MPI_COMM_WORLD OMPI_PREDEFINED_GLOBAL(MPI_Comm, ompi_symbols_.ompi_mpi_comm_world)
#define MPI_COMM_NULL OMPI_PREDEFINED_GLOBAL(MPI_Comm, ompi_symbols_.ompi_mpi_comm_null)
#define MPI_REQUEST_NULL OMPI_PREDEFINED_GLOBAL(MPI_Request, ompi_symbols_.ompi_request_null)
#define MPI_WIN_NULL OMPI_PREDEFINED_GLOBAL(MPI_Win, ompi_symbols_.ompi_mpi_win_null)
#define MPI_INFO_NULL OMPI_PREDEFINED_GLOBAL(MPI_Info, ompi_symbols_.ompi_mpi_info_null)

#define MPI_MAX OMPI_PREDEFINED_GLOBAL(MPI_Op, ompi_symbols_.ompi_mpi_op_max)
#define MPI_MIN OMPI_PREDEFINED_GLOBAL(MPI_Op, ompi_symbols_.ompi_mpi_op_min)
#define MPI_SUM OMPI_PREDEFINED_GLOBAL(MPI_Op, ompi_symbols_.ompi_mpi_op_sum)
#define MPI_PROD OMPI_PREDEFINED_GLOBAL(MPI_Op, ompi_symbols_.ompi_mpi_op_prod)
#define MPI_BAND OMPI_PREDEFINED_GLOBAL(MPI_Op, ompi_symbols_.ompi_mpi_op_band)
#define MPI_BOR OMPI_PREDEFINED_GLOBAL(MPI_Op, ompi_symbols_.ompi_mpi_op_bor)
#define MPI_BXOR OMPI_PREDEFINED_GLOBAL(MPI_Op, ompi_symbols_.ompi_mpi_op_bxor)
#define MPI_REPLACE OMPI_PREDEFINED_GLOBAL(MPI_Op, ompi_symbols_.ompi_mpi_op_replace)
#define MPI_NO_OP OMPI_PREDEFINED_GLOBAL(MPI_Op, ompi_symbols_.ompi_mpi_op_no_op)

#define MPI_DATATYPE_NULL OMPI_PREDEFINED_GLOBAL(MPI_Datatype, ompi_symbols_.ompi_mpi_datatype_null)
#define MPI_CHAR OMPI_PREDEFINED_GLOBAL(MPI_Datatype, ompi_symbols_.ompi_mpi_char)
#define MPI_UNSIGNED_CHAR OMPI_PREDEFINED_GLOBAL(MPI_Datatype, ompi_symbols_.ompi_mpi_unsigned_char)
#define MPI_SIGNED_CHAR OMPI_PREDEFINED_GLOBAL(MPI_Datatype, ompi_symbols_.ompi_mpi_signed_char)
#define MPI_SHORT OMPI_PREDEFINED_GLOBAL(MPI_Datatype, ompi_symbols_.ompi_mpi_short)
#define MPI_UNSIGNED_SHORT OMPI_PREDEFINED_GLOBAL(MPI_Datatype, ompi_symbols_.ompi_mpi_unsigned_short)
#define MPI_INT OMPI_PREDEFINED_GLOBAL(MPI_Datatype, ompi_symbols_.ompi_mpi_int)
#define MPI_UNSIGNED OMPI_PREDEFINED_GLOBAL(MPI_Datatype, ompi_symbols_.ompi_mpi_unsigned)
#define MPI_LONG OMPI_PREDEFINED_GLOBAL(MPI_Datatype, ompi_symbols_.ompi_mpi_long)
#define MPI_UNSIGNED_LONG OMPI_PREDEFINED_GLOBAL(MPI_Datatype, ompi_symbols_.ompi_mpi_unsigned_long)
#define MPI_LONG_LONG OMPI_PREDEFINED_GLOBAL(MPI_Datatype, ompi_symbols_.ompi_mpi_long_long_int)
#define MPI_UNSIGNED_LONG_LONG OMPI_PREDEFINED_GLOBAL(MPI_Datatype, ompi_symbols_.ompi_mpi_unsigned_long_long)
#define MPI_FLOAT OMPI_PREDEFINED_GLOBAL(MPI_Datatype, ompi_symbols_.ompi_mpi_float)
#define MPI_DOUBLE OMPI_PREDEFINED_GLOBAL(MPI_Datatype, ompi_symbols_.ompi_mpi_double)
#define MPI_LONG_DOUBLE OMPI_PREDEFINED_GLOBAL(MPI_Datatype, ompi_symbols_.ompi_mpi_long_double)

#endif //!defined(MPI_VERSION)

#if defined(c_plusplus) || defined(__cplusplus)
}
#endif

#endif //LIBRARY_INCLUDE_ROCSHMEM_MPI_HPP
