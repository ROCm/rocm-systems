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

#ifndef LIBRARY_INCLUDE_ROCSHMEM_HPP
#define LIBRARY_INCLUDE_ROCSHMEM_HPP

#include <hip/hip_runtime.h>

#include "rocshmem_config.h"
#include "rocshmem_common.hpp"
#include "rocshmem_RMA.hpp"
#include "rocshmem_AMO.hpp"
#include "rocshmem_SIG_OP.hpp"
#include "rocshmem_COLL.hpp"
#include "rocshmem_P2P_SYNC.hpp"
#include "rocshmem_RMA_X.hpp"
#include "rocshmem_TILE.hpp"
#if defined(HAVE_EXTERNAL_MPI)
#include <mpi.h>
#endif

/**
 * @file rocshmem.hpp
 * @brief Public header for rocSHMEM device and host libraries.
 *
 * This file contains all the callable functions and data structures for both
 * the device-side runtime and host-side runtime.
 *
 * The comments on these functions are sparse, but the semantics are the same
 * as those implemented in OpenSHMEM unless otherwise documented. Please see
 * the OpenSHMEM 1.4 standards documentation for more details:
 *
 * http://openshmem.org/site/sites/default/site_files/OpenSHMEM-1.4.pdf
 */

namespace rocshmem {

#define ROCSHMEM_MAJOR_VERSION 1
#define ROCSHMEM_MINOR_VERSION 4
#define ROCSHMEM_MAX_NAME_LEN  64

constexpr char VERSION[] = ROCSHMEM_VERSION;

/******************************************************************************
 **************************** HOST INTERFACE **********************************
 *****************************************************************************/
#if defined(HAVE_EXTERNAL_MPI)
/**
 * @brief Initialize the rocSHMEM runtime and underlying transport layer.
 *
 * @param[in] comm      MPI Communicator that rocSHMEM will be using
 *                      If MPI_COMM_NULL, rocSHMEM will be using MPI_COMM_WORLD
 */
[[deprecated]] __host__ void rocshmem_init(MPI_Comm comm);
#endif

/**
 * @brief Initialize the rocSHMEM runtime and underlying transport layer.
 *        This is equivalent to the previous function, using implicitly
 *        MPI_COMM_WORLD for initialization
 */
__host__ void rocshmem_init(void);


/**
 * @brief Query rocSHMEM context from host API (DEPRECATED)
 *
 * @deprecated This API is deprecated. Use rocshmem_hipmodule_init() instead,
 *             which provides better CUDA graph compatibility and automatic
 *             module initialization.
 *
 * @param[out] ctx      Returns ROCSHMEM_CTX_DEFAULT device pointer that users
 *                      can query from one instance of rocshmem host library and
 *                      use later for dynamic module initialization in
 *                      kernel bitcode device library in the same application
 */
[[deprecated("Use rocshmem_hipmodule_init() instead")]]
__host__ void * rocshmem_get_device_ctx();

/**
 * @brief Initialize rocSHMEM device context for a specific HIP module
 *
 * This function queries the ROCSHMEM_CTX_DEFAULT symbol from the provided
 * HIP module and initializes it with the host-side context using stream-ordered
 * memory operations. This is required for CUDA graph compatibility.
 *
 * @param[in] module    HIP module containing rocSHMEM device code
 * @param[in] stream    HIP stream to use for context initialization (optional,
 *                      uses hipStreamPerThread if nullptr)
 *
 * @return int          0 on success, non-zero on failure
 *
 * @note This function does not synchronize the stream, allowing it to be
 *       used in CUDA graph capture contexts. The caller is responsible for
 *       synchronization if needed.
 */
__host__ int rocshmem_hipmodule_init(hipModule_t module, hipStream_t stream = nullptr);

/**
 * @brief Query rocSHMEM remote symmetric heap pointer
 *
 * @param[in]  dest     local symmetric heap allocation pointer for current pe/device
 *
 * @param[in]  pe       remote PE
 *
 * @param[out] ptr      Returns remote symmetric heap device pointer from host-side API.
 *                      This can be used to issue load/store from custom kernels
 *                      instead of using rocshmem device side get/put APIs for RMA operations.
 */
__host__ void* rocshmem_ptr(const void *dest, int pe);
__device__ ATTR_NO_INLINE void* rocshmem_ptr(const void *dest, int pe);

#if defined(HAVE_EXTERNAL_MPI)
/**
 * @brief Initialize the rocSHMEM runtime and underlying transport layer
 *        with an attempt to enable the requested thread support.
 *
 * @param[in] requested Requested thread mode (from rocshmem_thread_ops)
 *                      for host-facing functions.
 * @param[out] provided Thread mode selected by the runtime. May not be equal
 *                      to requested thread mode.
 * @param[in] comm      (Optional) MPI Communicator that rocSHMEM will be using
 *                      If MPI_COMM_NULL, rocSHMEM will be using MPI_COMM_WORLD
 *
 * @return int          returns 0 upon success; otherwise, it returns a nonzero
 *                      value
 */
[[deprecated]] __host__ int rocshmem_init_thread(int requested, int *provided,
                                                 MPI_Comm comm);
#endif

/**
 * @brief Initialize the rocSHMEM runtime and underlying transport layer
 *        using the provided mode and attributes
 *
 * @param[in] flags   initialization method to be used.
 *                    Valid values are ROCSHMEM_INIT_WITH_UNIQUEID and
 *                    ROCSHMEM_INIT_WITH_MPI_COMM
 * @param[in] attr    attribute structure specifying input characteristics
 *
 * @return int        returns 0 upon success; otherwise, it returns a nonzero
 *                    value
 */
__host__ int rocshmem_init_attr(unsigned int flags, rocshmem_init_attr_t *attr);

/**
 * @brief Return a uniqueID
 *
 * @return int        returns 0 upon success; otherwise, it returns a nonzero
 *                    value
 */
__host__ int rocshmem_get_uniqueid(rocshmem_uniqueid_t *uid);

/**
 * @brief Initializes the rocshmem_init_attr_t struct
 *
 * @param[in] rank     rank of the calling process
 * @param[in] nranks   number of pes
 * @param[in] uid      unique ID used to identify the group processes.
 *                     All processes that
 * @param[out] attr    attribute structure to be passed to rocshmem_init_attr
 *
 * @return int         returns 0 upon success; otherwise, it returns a nonzero
 *                     value
 */
__host__ int rocshmem_set_attr_uniqueid_args(int rank, int nranks,
                                             rocshmem_uniqueid_t *uid,
                                             rocshmem_init_attr_t *attr);
/**
 * @brief Query the thread mode used by the runtime.
 *
 * @param[out] provided Thread mode the runtime is operating in.
 *
 * @return void.
 */
__host__ void rocshmem_query_thread(int *provided);

/**
 * @brief Function that dumps internal stats to stdout.
 */
__host__ void rocshmem_dump_stats();

/**
 * @brief Reset all internal stats.
 */
__host__ void rocshmem_reset_stats();

/**
 * @brief Finalize the rocSHMEM runtime.
 */
__host__ void rocshmem_finalize();

/**
 * @brief Allocate memory of \p size bytes from the symmetric heap.
 * This is a collective operation and must be called by all PEs.
 *
 * @param[in] size Memory allocation size in bytes.
 *
 * @return A pointer to the allocated memory on the symmetric heap.
 *
 * @todo Return error code instead of ptr.
 */
__host__ void *rocshmem_malloc(size_t size);

/**
 * @brief Allocate memory of \p size bytes from the symmetric heap with the
 * requested \p alignment. This is a collective operation and must be called by
 * all PEs with identical \p alignment and \p size arguments.
 *
 * The returned pointer (and, by symmetry, every peer's pointer for the same
 * collective call) is aligned to \p alignment bytes.
 *
 * Constraints on @p alignment (matching OpenSHMEM ``shmem_align`` and POSIX
 * ``posix_memalign`` semantics):
 *   - \p alignment must be a power of two.
 *   - \p alignment must be a multiple of ``sizeof(void *)``.
 *
 *   In practice, any power of two that is at least ``sizeof(void *)`` (i.e.
 *   \c 8 on a 64-bit build) satisfies both rules. Therefore the smallest
 *   accepted value is ``sizeof(void *)`` and accepted values are
 *   ``sizeof(void *), 2*sizeof(void *), 4*sizeof(void *), ...``.
 *
 *   If \p alignment violates either rule (including \c 0, \c 1, \c 2, \c 4 on
 *   64-bit, or any non power-of-two), this routine returns \c NULL and emits
 *   a warning. The collective barrier is still entered so other PEs do not
 *   deadlock.
 *
 *   \p alignment values less than or equal to the default symmetric-heap
 *   alignment (128 bytes) yield a 128-byte aligned pointer (identical to
 *   ::rocshmem_malloc).
 *
 * @param[in] alignment Required pointer alignment in bytes. Must be a power
 *                      of two and a multiple of ``sizeof(void *)``.
 * @param[in] size      Memory allocation size in bytes.
 *
 * @return A pointer to the allocated memory on the symmetric heap, or
 *         \c NULL if \p alignment is invalid or the allocation cannot be
 *         satisfied.
 */
__host__ void *rocshmem_align(size_t alignment, size_t size);

/**
 * @brief Allocate memory for an array of \p count elements of \p size bytes
 * each from the symmetric heap and zero-initialize the allocation. This is a
 * collective operation and must be called by all PEs with identical \p count
 * and \p size arguments.
 *
 * Mirrors OpenSHMEM ``shmem_calloc`` semantics: the returned buffer holds
 * ``count * size`` bytes, every byte set to ``0``. The pointer satisfies the
 * same default symmetric-heap alignment as ::rocshmem_malloc.
 *
 * Returns ``NULL`` when:
 *   - \p count is \c 0,
 *   - \p size is \c 0,
 *   - ``count * size`` would overflow ``size_t`` (a warning is emitted in
 *     this case), or
 *   - the underlying allocation cannot be satisfied.
 *
 * The collective barrier is still entered on the NULL-return paths so that
 * other PEs do not deadlock.
 *
 * @param[in] count Number of elements to allocate.
 * @param[in] size  Size of each element in bytes.
 *
 * @return A pointer to the zero-initialized allocation on the symmetric
 *         heap, or \c NULL on the failure conditions described above.
 */
__host__ void *rocshmem_calloc(size_t count, size_t size);

/**
 * @brief Free a memory allocation from the symmetric heap.
 * This is a collective operation and must be called by all PEs.
 *
 * @param[in] ptr Pointer to previously allocated memory on the symmetric heap.
 */
__host__ void rocshmem_free(void *ptr);

/**
 * @brief Registers a non-symmetric buffer
 *
 * @param[in] addr Pointer to previously allocated user memory
 * @param[in] length Length of addr
 */
__host__ int rocshmem_buffer_register(void *addr, size_t length);

/**
 * @brief Deregisters previously registered user memory
 *
 * @param[in] addr Pointer to previously registered memory
 */
__host__ int rocshmem_buffer_unregister(void *addr);

/**
 * @brief Deregisters all previously registered user memory
 */
__host__ void rocshmem_buffer_unregister_all();

/**
 * @brief Query for the number of PEs.
 *
 * @return Number of PEs.
 */
__host__ int rocshmem_n_pes();

/**
 * @brief Query the PE ID of the caller.
 *
 * @return PE ID of the caller.
 */
__host__ int rocshmem_my_pe();

/**
 * @brief Query the OpenSHMEM specification version this library implements.
 *
 * @param[out] major  ROCSHMEM_MAJOR_VERSION.
 * @param[out] minor  ROCSHMEM_MINOR_VERSION.
 */
__host__ void rocshmem_info_get_version(int *major, int *minor);

/**
 * @brief Returns the vendor-defined string identifying this library.
 *
 * Copies ROCSHMEM_VENDOR_STRING into @p name. The caller must provide
 * a buffer of at least ROCSHMEM_MAX_NAME_LEN bytes.
 *
 * @param[out] name  Buffer to receive the vendor string.
 */
__host__ void rocshmem_info_get_name(char *name);

/**
 * @brief Returns the vendor-specific library version of rocSHMEM.
 *
 * Writes ROCSHMEM_VENDOR_MAJOR_VERSION into @p major,
 * ROCSHMEM_VENDOR_MINOR_VERSION into @p minor, and
 * ROCSHMEM_VENDOR_PATCH_VERSION into @p patch.
 *
 * @param[out] major  Vendor major version.
 * @param[out] minor  Vendor minor version.
 * @param[out] patch  Vendor patch version.
 */
__host__ void rocshmem_vendor_get_version_info(int *major, int *minor,
                                               int *patch);

/**
 * @brief Creates an OpenSHMEM context.
 *
 * @param[in] options Options for context creation. Ignored in current design.
 * @param[out] ctx    Context handle.
 *
 * @return Zero on success and nonzero otherwise.
 */
__host__ int rocshmem_ctx_create(int64_t options, rocshmem_ctx_t *ctx);

/**
 * @brief Destroys an OpenSHMEM context.
 *
 * @param[out] ctx    Context handle.
 *
 * @return void.
 */
__host__ void rocshmem_ctx_destroy(rocshmem_ctx_t ctx);

/**
 * @brief Translate the PE in src_team to that in dest_team.
 *
 * @param[in] src_team  Handle of the team from which to translate
 * @param[in] src_pe    PE-of-interest's index in src_team
 * @param[in] dest_team Handle of the team to which to translate
 *
 * @return PE of src_pe in dest_team. If any input is invalid
 *         or if src_pe is not in both source and destination
 *         teams, a value of -1 is returned.
 */
__host__ int rocshmem_team_translate_pe(rocshmem_team_t src_team, int src_pe,
                                         rocshmem_team_t dest_team);

/**
 * @brief Query the number of PEs in a team.
 *
 * @param[in] team The team to query the number of PEs from.
 *
 * @return Number of PEs in the provided team.
 */
__host__ int rocshmem_team_n_pes(rocshmem_team_t team);

/**
 * @brief Query the PE ID of the caller in a team.
 *
 * @param[in] team The team to query PE ID in.
 *
 * @return PE ID of the caller in the provided team.
 */
__host__ int rocshmem_team_my_pe(rocshmem_team_t team);

/**
 * @brief Create a new a team of PEs. Must be called by all PEs
 * in the parent team.
 *
 * @param[in] parent_team The team to split from.
 * @param[in] start       The lowest PE number of the subset of the PEs
 *                        from the parent team that will form the new
 *                        team.
 * @param[in] stide       The stride between team PE members in the
 *                        parent team that comprise the subset of PEs
 *                        that will form the new team.
 * @param[in] size        The number of PEs in the new team.
 * @param[in] config      Pointer to the config parameters for the new
 *                        team.
 * @param[in] config_mask Bitwise mask representing parameters to use
 *                        from config
 * @param[out] new_team   Pointer to the newly created team. If an error
 *                        occurs during team creation, or if the PE in
 *                        the parent team is not in the new team, the
 *                        value will be ROCSHMEM_TEAM_INVALID.
 *
 * @return Zero upon successful team creation; non-zero if erroneous.
 */
__host__ int rocshmem_team_split_strided(rocshmem_team_t parent_team,
                                          int start, int stride, int size,
                                          const rocshmem_team_config_t *config,
                                          long config_mask,
                                          rocshmem_team_t *new_team);

/**
 * @brief Destroy a team. Must be called by all PEs in the team.
 * The user must destroy all private contexts created in the
 * team before destroying this team. Otherwise, the behavior
 * is undefined. This call will destroy only the shareable contexts
 * created from the referenced team.
 *
 * @param[in] team The team to destroy.
 *                 ROCSHMEM_TEAM_INVALID, ROCSHMEM_TEAM_WORLD, and
 *                 ROCSHMEM_TEAM_SHARED are silently ignored (no-op).
 *                 Passing a handle that was already destroyed or
 *                 never created results in undefined behavior.
 *
 * @return None.
 */
__host__ void rocshmem_team_destroy(rocshmem_team_t team);

/**
 * @brief Guarantees order between messages in this context in accordance with
 * OpenSHMEM semantics.
 *
 * @param[in] ctx     Context with which to perform this operation.
 *
 * @return void.
 */
__host__ void rocshmem_ctx_fence(rocshmem_ctx_t ctx);

__host__ void rocshmem_fence();

/**
 * @brief Completes all previous operations posted on the host.
 *
 * @param[in] ctx     Context with which to perform this operation.
 *
 * @return void.
 */
__host__ void rocshmem_ctx_quiet(rocshmem_ctx_t ctx);

__host__ void rocshmem_quiet();

/**
 * @brief enqueues a quiet operation on given stream.
 *
 * @param[in] stream  HIP stream on which to enqueue the operation.
 *
 * @return void
 */
__host__ void rocshmem_quiet_on_stream(hipStream_t stream);

/**
 * @brief perform a collective barrier between all PEs in the system.
 * The caller is blocked until the barrier is resolved.
 *
 * @return void
 */
__host__ void rocshmem_barrier_all();

/**
 * @brief enqueues a collective barrier on given stream.
 *
 * @return void
 */
__host__ void rocshmem_barrier_all_on_stream(hipStream_t stream);

/**
 * @brief enqueues a sync_all operation on given stream.
 *
 * @param[in] stream  HIP stream on which to enqueue the operation.
 *
 * @return void
 */
__host__ void rocshmem_sync_all_on_stream(hipStream_t stream);

/**
 * @brief enqueues an alltoall collective operation on given stream.
 *
 * @param[in] team    The team participating in the collective.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] size    Number of bytes to transfer per pair of PEs.
 * @param[in] stream  HIP stream on which to enqueue the operation.
 *
 * @return void
 */
__host__ void rocshmem_alltoallmem_on_stream(rocshmem_team_t team, void *dest,
                                             const void *source, size_t size,
                                             hipStream_t stream);

/**
 * @brief enqueues a broadcast collective operation on given stream.
 *
 * @param[in] team    The team participating in the collective.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Number of bytes to broadcast.
 * @param[in] pe_root Root PE (relative to team) from which to broadcast.
 * @param[in] stream  HIP stream on which to enqueue the operation.
 *
 * @return void
 */
__host__ void rocshmem_broadcastmem_on_stream(rocshmem_team_t team, void *dest,
                                              const void *source, size_t nelems,
                                              int pe_root, hipStream_t stream);

/**
 * @brief enqueues a getmem RMA operation on given stream.
 *
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 * @param[in] stream  HIP stream on which to enqueue the operation.
 *
 * @return void
 */
__host__ void rocshmem_getmem_on_stream(void *dest, const void *source,
                                        size_t nelems, int pe,
                                        hipStream_t stream);

/**
 * @brief enqueues a putmem RMA operation on given stream.
 *
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 * @param[in] stream  HIP stream on which to enqueue the operation.
 *
 * @return void
 */
__host__ void rocshmem_putmem_on_stream(void *dest, const void *source,
                                        size_t nelems, int pe,
                                        hipStream_t stream);

/**
 * @brief Perform a put operation with signal on a HIP stream.
 *
 * This routine initiates a remote memory transfer on a specified HIP stream.
 * The source data is copied from the local PE to the remote PE's destination
 * address. After the put operation completes, a signal operation is performed
 * on a remote symmetric signal variable.
 *
 * @param[in] dest      Destination address on the remote PE
 * @param[in] source    Source address on the local PE
 * @param[in] nelems    Size of the transfer in bytes
 * @param[in] sig_addr  Address of signal variable on the remote PE
 * @param[in] signal    Signal value to be written
 * @param[in] sig_op    Signal operation (ROCSHMEM_SIGNAL_SET or
 * ROCSHMEM_SIGNAL_ADD)
 * @param[in] pe        PE number of the remote PE
 * @param[in] stream    HIP stream on which to enqueue the operation
 *
 * @return void
 */
__host__ void rocshmem_putmem_signal_on_stream(void *dest, const void *source,
                                               size_t nelems,
                                               uint64_t *sig_addr,
                                               uint64_t signal, int sig_op,
                                               int pe, hipStream_t stream);

/**
 * @brief Wait on a signal variable until it satisfies the specified condition,
 * with the operation enqueued on a HIP stream.
 *
 * This function blocks the calling thread until the signal variable at
 * \p sig_addr satisfies the comparison condition (* \p sig_addr \p cmp
 * \p cmp_value). The wait operation is executed asynchronously on the
 * specified HIP stream.
 *
 * @param[in] sig_addr  Address of the signal variable on the symmetric heap
 * @param[in] cmp       Comparison operator (e.g., ROCSHMEM_CMP_EQ,
 * ROCSHMEM_CMP_GE, ROCSHMEM_CMP_NE, etc.)
 * @param[in] cmp_value Value to compare against
 * @param[in] stream    HIP stream on which to enqueue the operation
 *
 * @return void
 */
__host__ void rocshmem_signal_wait_until_on_stream(uint64_t *sig_addr, int cmp,
                                                   uint64_t cmp_value,
                                                   hipStream_t stream);

/**
 * @brief registers the arrival of a PE at a barrier.
 * The caller is blocked until the synchronization is resolved.
 *
 * In contrast with the shmem_barrier_all routine, shmem_sync_all only ensures
 * completion and visibility of previously issued memory stores and does not
 * ensure completion of remote memory updates issued via OpenSHMEM routines.
 *
 * @return void
 */
__host__ void rocshmem_sync_all();

/**
 * @brief allows any PE to force the termination of an entire program.
 *
 * @param[in] status    The exit status from the main program.
 *
 * @return void
 */
__host__ void rocshmem_global_exit(int status);

/******************************************************************************
 **************************** DEVICE INTERFACE ********************************
 *****************************************************************************/

/**
 * @brief Initializes device-side rocSHMEM resources. Must be called before
 * any threads in this work-group invoke other rocSHMEM functions.
 *
 * Must be called collectively by all threads in the work-group.
 *
 * @return void.
 */
[[deprecated]] __device__ void rocshmem_wg_init();

/**
 * @brief Finalizes device-side rocSHMEM resources. Must be called before
 * work-group completion if the work-group also called rocshmem_wg_init().
 *
 * Must be called collectively by all threads in the work-group.
 *
 * @return void.
 */
[[deprecated]] __device__ void rocshmem_wg_finalize();

/**
 * @brief Initializes device-side rocSHMEM resources. Must be called before
 * any threads in this work-group invoke other rocSHMEM functions. This is
 * a variant of rocshmem_wg_init that allows the caller to request a
 * threading mode.
 *
 * @param[in] requested Requested thread mode from rocshmem_thread_ops.
 * @param[out] provided Thread mode selected by the runtime. May not be equal
 *                      to requested thread mode.
 *
 * Must be called collectively by all threads in the work-group.
 *
 * @return void.
 */
[[deprecated]] __device__ void rocshmem_wg_init_thread(int requested, int *provided);

/**
 * @brief Query the thread mode used by the runtime.
 *
 * @param[out] provided Thread mode the runtime is operating in.
 *
 * @return void.
 */
__device__ void rocshmem_query_thread(int *provided);

/**
 * @brief Creates an OpenSHMEM context. By design, the context is private
 * to the calling work-group.
 *
 * Must be called collectively by all threads in the work-group.
 *
 * @param[in] options Options for context creation. Ignored in current design.
 * @param[out] ctx    Context handle.
 *
 * @return All threads returns 0 if the context was created successfully. If any
 * thread returns non-zero value, the operation failed and a higher number of
 * `ROCSHMEM_MAX_NUM_CONTEXTS` is required.
 */
__device__ ATTR_NO_INLINE int rocshmem_wg_ctx_create(int64_t options,
                                                      rocshmem_ctx_t *ctx);

__device__ ATTR_NO_INLINE int rocshmem_wg_team_create_ctx(
    rocshmem_team_t team, long options, rocshmem_ctx_t *ctx);

/**
 * @brief Destroys an OpenSHMEM context.
 *
 * Must be called collectively by all threads in the work-group.
 *
 * @param[in] The context to destroy.
 *
 * @return void.
 */
__device__ ATTR_NO_INLINE void rocshmem_wg_ctx_destroy(rocshmem_ctx_t *ctx);

/**
 * @brief Guarantees order between messages in this context in accordance with
 * OpenSHMEM semantics.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * rocSHMEM function.
 *
 * @param[in] ctx Context with which to perform this operation.
 *
 * @return void.
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_fence(rocshmem_ctx_t ctx);

__device__ ATTR_NO_INLINE void rocshmem_fence();

/**
 * @brief Guarantees order between messages in this context in accordance with
 * OpenSHMEM semantics.
 *
 * This function  is an extension as it is per PE. has same semantics as default
 * API but it is per PE
 *
 * @param[in] ctx Context with which to perform this operation.
 * @param[in] pe destination pe.
 *
 * @return void.
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_fence(rocshmem_ctx_t ctx, int pe);

__device__ ATTR_NO_INLINE void rocshmem_fence(int pe);

/**
 * @brief Completes all previous operations posted to this context.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * rocSHMEM function.
 *
 * @param[in] ctx Context with which to perform this operation.
 *
 * @return void.
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_quiet(rocshmem_ctx_t ctx);

__device__ ATTR_NO_INLINE void rocshmem_quiet();

/**
 * @brief Completes all previous operations posted to this context for PEs in the
 *        `target_pes` array.
 *
 * @param[in] ctx Context with which to perform this operation.
 *
 * @param[in] target_pes Address of target PE array where the operations need to be completed.
 *
 * @param[in] npes The number of PEs in the target PE array.
 *
 * @return void.
 */

__device__ ATTR_NO_INLINE void rocshmem_ctx_pe_quiet(rocshmem_ctx_t ctx, const int *target_pes, size_t npes);

__device__ ATTR_NO_INLINE void rocshmem_pe_quiet(const int *target_pes, size_t npes);

/**
 * @brief Query the total number of PEs.
 *
 * Can be called per thread with no performance penalty.
 *
 * @param[in] ctx GPU side handle.
 *
 * @return Total number of PEs.
 */
__device__ int rocshmem_ctx_n_pes(rocshmem_ctx_t ctx);

/**
 * @brief Query for the number of PEs.
 *
 * @return Number of PEs.
 */
__device__ int rocshmem_n_pes();

/**
 * @brief Query the PE ID of the caller.
 *
 * Can be called per thread with no performance penalty.
 *
 * @param[in] ctx GPU side handle
 *
 * @return PE ID of the caller.
 */
__device__ int rocshmem_ctx_my_pe(rocshmem_ctx_t ctx);

/**
 * @brief Query the PE ID of the caller.
 *
 * @return PE ID of the caller.
 */
__device__ int rocshmem_my_pe();

/**
 * @brief Query the number of PEs in a team.
 *
 * @param[in] team The team to query the number of PEs from.
 *
 * @return Number of PEs in the provided team.
 */
__device__ int rocshmem_team_n_pes(rocshmem_team_t team);

/**
 * @brief Query the PE ID of the caller in a team.
 *
 * @param[in] team The team to query PE ID in.
 *
 * @return PE ID of the caller in the provided team.
 */
__device__ int rocshmem_team_my_pe(rocshmem_team_t team);

/**
 * @brief Translate the PE in src_team to that in dest_team.
 *
 * @param[in] src_team  Handle of the team from which to translate
 * @param[in] src_pe    PE-of-interest's index in src_team
 * @param[in] dest_team Handle of the team to which to translate
 *
 * @return PE of src_pe in dest_team. If any input is invalid
 *         or if src_pe is not in both source and destination
 *         teams, a value of -1 is returned.
 */
__device__ int rocshmem_team_translate_pe(rocshmem_team_t src_team,
                                           int src_pe,
                                           rocshmem_team_t dest_team);

__device__ ATTR_NO_INLINE void rocshmem_ctx_threadfence_system(
    rocshmem_ctx_t ctx);

__device__ ATTR_NO_INLINE void rocshmem_threadfence_system();

}  // namespace rocshmem

#endif  // LIBRARY_INCLUDE_ROCSHMEM_HPP
