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

#include "segment_builder.hpp"

#include "util.hpp"
#include "gda/endian.hpp"

namespace rocshmem {

__device__ SegmentBuilder::SegmentBuilder(uint64_t wqe_idx, void *base) {
  mlx5_segment *base_ptr = static_cast<mlx5_segment*>(base);
  size_t segment_offset = wqe_idx * SEGMENTS_PER_WQE;
  segp = &base_ptr[segment_offset];
}

/*
 * Control segment - contains some control information for the current WQE.
 *
 * Output:
 *      seg       - control segment to be filled
 * Input:
 *      pi        - WQEBB number of the first block of this WQE.
 *                  This number should wrap at 0xffff, regardless of
 *                  size of the WQ.
 *      opcode    - Opcode of this WQE. Encodes the type of operation
 *                  to be executed on the QP.
 *      opmod     - Opcode modifier.
 *      qp_num    - QP/SQ number this WQE is posted to.
 *      fm_ce_se  - FM (fence mode), CE (completion and event mode)
 *                  and SE (solicited event).
 *      ds        - WQE size in octowords (16-byte units). DS accounts for all
 *                  the segments in the WQE as summarized in WQE construction.
 *      signature - WQE signature.
 *      imm       - Immediate data/Invalidation key/UMR mkey.
 */
/*
 * static MLX5DV_ALWAYS_INLINE
 * void mlx5dv_set_ctrl_seg(struct mlx5_wqe_ctrl_seg *seg, uint16_t pi, uint8_t opcode, uint8_t opmod, uint32_t qp_num, uint8_t fm_ce_se, uint8_t ds, uint8_t signature, uint32_t imm)
 * {
 *   seg->opmod_idx_opcode   = htobe32(((uint32_t)opmod << 24) | ((uint32_t)pi << 8) | opcode);
 *   seg->qpn_ds             = htobe32((qp_num << 8) | ds);
 *   seg->fm_ce_se           = fm_ce_se;
 *   seg->signature          = signature;
 *   // The caller should prepare "imm" in advance based on WR opcode.
 *   // For IBV_WR_SEND_WITH_IMM and IBV_WR_RDMA_WRITE_WITH_IMM,
 *   // the "imm" should be assigned as is.
 *   // For the IBV_WR_SEND_WITH_INV, it should be htobe32(imm).
 *   seg->imm                = imm;
 * }
 */
__device__ void SegmentBuilder::update_ctrl_seg(uint16_t pi, uint8_t opcode, uint8_t opmod, uint32_t qp_num, uint8_t fm_ce_se, uint8_t ds, uint8_t signature, uint32_t imm) {
  segp->ctrl_seg = {0};
  segp->ctrl_seg.opmod_idx_opcode = byteswap<uint32_t>(((uint32_t)opmod << 24) | ((uint32_t)pi << 8) | opcode);
  segp->ctrl_seg.qpn_ds = byteswap<uint32_t>(qp_num << 8 | ds);
  segp->ctrl_seg.fm_ce_se = fm_ce_se;
  segp->ctrl_seg.signature = signature;
  segp->ctrl_seg.imm = imm;
  segp++;
}

__device__ void SegmentBuilder::update_raddr_seg(uint64_t raddr, uint32_t rkey) {
  segp->raddr_seg = {0};
  segp->raddr_seg.raddr = byteswap<uint64_t>(raddr);
  segp->raddr_seg.rkey = rkey;
  segp++;
}

/*
 * Data Segments - contain pointers and a byte count for the scatter/gather list.
 * They can optionally contain data, which will save a memory read access for
 * gather Work Requests.
 */
/*
 * static MLX5DV_ALWAYS_INLINE
 * void mlx5dv_set_data_seg(struct mlx5_wqe_data_seg *seg, uint32_t length, uint32_t lkey, uintptr_t address) {
 *   seg->byte_count = htobe32(length);
 *   seg->lkey       = htobe32(lkey);
 *   seg->addr       = htobe64(address);
 * }
 */
__device__ void SegmentBuilder::update_data_seg(uint64_t laddr, uint32_t size, uint32_t lkey) {
  segp->data_seg = {0};
  segp->data_seg.byte_count = byteswap<uint32_t>(size);
  segp->data_seg.lkey = lkey;
  segp->data_seg.addr = byteswap<uint64_t>(laddr);
  segp++;
}

__device__ void SegmentBuilder::update_inl_data_seg(const void* laddr, int32_t size) {
  // size is masked with 0x3FF because only the first 10 bits of byte_count are valid
  segp->inl_data_seg.byte_count = byteswap<uint32_t>((size & 0x3FF) | MLX5_INLINE_SEG);
  // + 1 because we start packing the segment with data after the byte_count parameter
  memcpy(&segp->inl_data_seg + 1, laddr, size);
  segp++;
}

__device__ void SegmentBuilder::update_atomic_seg(uint64_t atomic_data, uint64_t atomic_cmp) {
  segp->atomic_seg = {0};
  segp->atomic_seg.swap_add = byteswap<uint64_t>(atomic_data);
  segp->atomic_seg.compare = byteswap<uint64_t>(atomic_cmp);
  segp++;
}

}  // namespace rocshmem
