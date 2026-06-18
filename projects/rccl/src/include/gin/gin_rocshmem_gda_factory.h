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

#ifndef GIN_QP_FACTORY_H_
#define GIN_QP_FACTORY_H_

#include <stdint.h>
#include <stddef.h>

/**
 * @file gin_rocshmem_gda_factory.h
 *
 * @section DESCRIPTION
 * Standalone C-linkage API for creating rocshmem QueuePairs for use by
 * RCCL's GIN backend. Does NOT require rocshmem_init() — the factory
 * opens IB devices, loads DV libraries, and creates QPs independently.
 *
 * Lifecycle:
 *   1. rocshmem_gin_create_qps()  — discover NIC, create + connect QPs
 *   2. rocshmem_gin_reg_mr()      — register buffers for RDMA
 *   3. (use QPs from device code)
 *   4. rocshmem_gin_dereg_mr()    — deregister buffers
 *   5. rocshmem_gin_destroy_qps() — tear down everything
 */

#define GIN_QP_API __attribute__((visibility("default")))

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque handle representing a set of GIN QPs and their IB resources.
 */
typedef struct rocshmem_gin_qp_set* rocshmem_gin_qp_set_t;

/**
 * @brief Create a set of QPs for GIN use.
 *
 * Creates nRanks QueuePair objects (one per peer), connected via the provided
 * bootstrap allgather function. Each QP is GPU-accessible and ready for
 * RDMA write and atomic operations.
 *
 * Does not require rocshmem_init(). Opens IB devices and loads DV libraries
 * independently.
 *
 * @param[in]  nRanks       Number of peers (one QP per peer).
 * @param[in]  myRank       This rank's index.
 * @param[in]  allgather    Bootstrap allgather function for exchanging QP info.
 *                          Signature: int (*)(void* ctx, void* buf, size_t size)
 *                          Performs in-place allgather of size bytes per rank.
 * @param[in]  allgather_ctx Opaque context passed to allgather.
 * @param[out] out_qp_set   Opaque handle to the created QP set.
 * @param[out] out_gpu_qps  GPU-accessible array of nRanks QueuePair pointers.
 *
 * @return 0 on success, non-zero on failure.
 */
GIN_QP_API int rocshmem_gin_create_qps(int nRanks, int myRank,
                             int (*allgather)(void* ctx, void* buf, size_t size),
                             void* allgather_ctx,
                             rocshmem_gin_qp_set_t* out_qp_set,
                             void*** out_gpu_qps);

/**
 * @brief Destroy a set of GIN QPs and release all IB resources.
 *
 * @param[in] qp_set Handle returned by rocshmem_gin_create_qps.
 */
GIN_QP_API void rocshmem_gin_destroy_qps(rocshmem_gin_qp_set_t qp_set);

/**
 * @brief Register a memory buffer for RDMA using a QP set's PD.
 *
 * @param[in]  qp_set  QP set whose PD to use for registration.
 * @param[in]  addr    Buffer address (GPU memory).
 * @param[in]  size    Buffer size in bytes.
 * @param[in]  atomic  If true, register with IBV_ACCESS_REMOTE_ATOMIC.
 * @param[out] out_mr  Opaque MR handle (for deregistration).
 * @param[out] out_lkey Local key for this registration.
 * @param[out] out_rkey Remote key for this registration.
 *
 * @return 0 on success, non-zero on failure.
 */
GIN_QP_API int rocshmem_gin_reg_mr(rocshmem_gin_qp_set_t qp_set,
                         void* addr, size_t size, int atomic,
                         void** out_mr, uint32_t* out_lkey, uint32_t* out_rkey);

/**
 * @brief Register a VMM-allocated memory buffer for RDMA.
 *
 * Uses hipMemGetHandleForAddressRange for dmabuf FD extraction (works
 * for both hipMalloc and VMM allocations), then ibv_reg_dmabuf_mr.
 * Falls back to ibv_reg_mr_iova2. Both use iova=VA.
 */
GIN_QP_API int rocshmem_gin_reg_mr_vmm(rocshmem_gin_qp_set_t qp_set,
                              void* addr, size_t size, int atomic,
                              void** out_mr, uint32_t* out_lkey, uint32_t* out_rkey);

/**
 * @brief Deregister a memory buffer.
 *
 * @param[in] mr Handle returned by rocshmem_gin_reg_mr.
 */
GIN_QP_API void rocshmem_gin_dereg_mr(void* mr);

/**
 * @brief Query the GDA provider type from a QP set.
 *
 * @param[in] qp_set QP set to query.
 * @return Provider enum value (GDAProvider::IONIC/BNXT/MLX5), or -1 if invalid.
 */
GIN_QP_API int rocshmem_gin_get_provider(rocshmem_gin_qp_set_t qp_set);

/**
 * @brief Probe for usable IB devices with active ports and recognized vendor IDs.
 *
 * Lightweight check — opens devices, checks vendor ID and port state, closes immediately.
 * Reuses gin_rocshmem_gda_factory's existing vendor detection (BNXT/IONIC/MLX5).
 *
 * @return Number of usable devices (0 if none found).
 */
GIN_QP_API int rocshmem_gin_probe_devices(void);

/**
 * @brief Initialize rocshmem __constant__ device memory for the GIN QP path.
 *
 * Sets constmem.gda_provider and logd_constants so that QueuePair device
 * dispatchers use the fast scalar constant cache and error reporting works.
 * Called automatically after QP creation.
 *
 * @param[in] provider GDAProvider enum value from rocshmem_gin_get_provider().
 * @param[in] rank     This rank's index (for device-side log messages).
 */
GIN_QP_API void rocshmem_gin_init_constmem(int provider, int rank);


#ifdef __cplusplus
}
#endif

#endif  // GIN_QP_FACTORY_H_
