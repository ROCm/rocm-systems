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

#ifndef LIBRARY_INCLUDE_ROCSHMEM_COMMON_HPP
#define LIBRARY_INCLUDE_ROCSHMEM_COMMON_HPP

namespace rocshmem {

#ifdef USE_FUNC_CALL
#define ATTR_NO_INLINE __attribute__((noinline))
#else
#define ATTR_NO_INLINE
#endif


enum ROCSHMEM_STATUS {
  ROCSHMEM_SUCCESS = 0,
  ROCSHMEM_ERROR = 1,
};

enum ROCSHMEM_OP {
  ROCSHMEM_SUM,
  ROCSHMEM_MAX,
  ROCSHMEM_MIN,
  ROCSHMEM_PROD,
  ROCSHMEM_AND,
  ROCSHMEM_OR,
  ROCSHMEM_XOR,
  ROCSHMEM_REPLACE
};

enum ROCSHMEM_SIGNAL_OPS {
  ROCSHMEM_SIGNAL_SET,
  ROCSHMEM_SIGNAL_ADD,
};

/**
 * @brief Types defined for rocshmem_wait() operations.
 */
enum rocshmem_cmps {
  ROCSHMEM_CMP_EQ,
  ROCSHMEM_CMP_NE,
  ROCSHMEM_CMP_GT,
  ROCSHMEM_CMP_GE,
  ROCSHMEM_CMP_LT,
  ROCSHMEM_CMP_LE,
};

enum rocshmem_thread_ops {
  ROCSHMEM_THREAD_SINGLE,
  ROCSHMEM_THREAD_FUNNELED,
  ROCSHMEM_THREAD_WG_FUNNELED,
  ROCSHMEM_THREAD_SERIALIZED,
  ROCSHMEM_THREAD_MULTIPLE
};

/**
 * @brief Bitwise flags to mask configuration parameters.
 */
enum rocshmem_team_configs {
  ROCSHMEM_TEAM_DEFAULT_CONFIGS,
  ROCSHMEM_TEAM_NUM_CONTEXTS
};

typedef struct {
  int num_contexts;
} rocshmem_team_config_t;

constexpr size_t ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE = 1024;
constexpr size_t ROCSHMEM_BARRIER_SYNC_SIZE = 256;
constexpr size_t ROCSHMEM_REDUCE_SYNC_SIZE = 256;
// Internally calls sync function, which matches barrier implementation
constexpr size_t ROCSHMEM_BCAST_SYNC_SIZE = ROCSHMEM_BARRIER_SYNC_SIZE;
constexpr size_t ROCSHMEM_ALLTOALL_SYNC_SIZE = ROCSHMEM_BARRIER_SYNC_SIZE + 1;
constexpr size_t ROCSHMEM_FCOLLECT_SYNC_SIZE = ROCSHMEM_ALLTOALL_SYNC_SIZE;
constexpr size_t ROCSHMEM_SYNC_VALUE = 0;

const int ROCSHMEM_CTX_ZERO = 0;
const int ROCSHMEM_CTX_NOSTORE = 1;
const int ROCSHMEM_CTX_SERIALIZED = 2;
const int ROCSHMEM_CTX_WG_PRIVATE = 4;
const int ROCSHMEM_CTX_SHARED = 8;

/**
 * @brief GPU side OpenSHMEM context created from each work-groups'
 * rocshmem_wg_handle_t
 */
typedef struct rocshmem_ctx{
  void *ctx_opaque;
  void *team_opaque;

  __host__ __device__ bool operator==(const struct rocshmem_ctx& other) const {
    return (ctx_opaque == other.ctx_opaque &&
            team_opaque == other.team_opaque);
  }

  __host__ __device__ bool operator!=(const struct rocshmem_ctx& other) const {
    return !(*this == other);
  }
} rocshmem_ctx_t;

/**
 * Shmem default context.
 */
extern "C" __device__  rocshmem_ctx_t __attribute__((visibility("default"))) ROCSHMEM_CTX_DEFAULT;

/**
 * A value corresponding to an invalid communication context. This value can be
 * used to initialize or update context handles to indicate that they do not
 * reference a valid context. When managed in this way, applications can use an
 * equality comparison to test whether a given context handle references a
 * valid context.
 */
extern __constant__ rocshmem_ctx_t ROCSHMEM_CTX_INVALID;
/**
 * Used internally to set default context.
 */
void set_internal_ctx(rocshmem_ctx_t *ctx);

/**
 * Used internally to query the loaded backend
 */
//TODO: this should remain internal?
enum class BackendType { GDA_BACKEND, RO_BACKEND, IPC_BACKEND };
BackendType get_backend_type();

typedef uint64_t *rocshmem_team_t;
extern rocshmem_team_t ROCSHMEM_TEAM_WORLD;

const rocshmem_team_t ROCSHMEM_TEAM_INVALID = nullptr;

/**
 * @brief  Data structure defining the unqiueId
 */


/// Unique ID for a process. This is a ROCSHMEM_UNIQUE_ID_BYTES byte array that uniquely identifies a process.
#define ROCSHMEM_UNIQUE_ID_BYTES 128
using rocshmem_uniqueid_t = std::array<uint8_t, ROCSHMEM_UNIQUE_ID_BYTES>;

/**
 * @brief Data structure used for attribute based 
 *        initialization
 */
struct rocshmem_init_attr_t  {
  int32_t rank;
  int32_t nranks;
  rocshmem_uniqueid_t uid;
  void* mpi_comm;
};
typedef struct rocshmem_init_attr_t rocshmem_init_attr_t;

constexpr unsigned int ROCSHMEM_INIT_WITH_MPI_COMM = 0;
constexpr unsigned int ROCSHMEM_INIT_WITH_UNIQUEID = 1;

}  // namespace rocshmem

#endif  // LIBRARY_INCLUDE_ROCSHMEM_COMMON_HPP
