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

#ifndef LIBRARY_SRC_GDA_GDA_SYMM_TABLE_HPP_
#define LIBRARY_SRC_GDA_GDA_SYMM_TABLE_HPP_

#include <cstddef>
#include <cstdint>

namespace rocshmem {

/**
 * @brief One symmetric user-buffer registration, pre-specialized to a single
 * QueuePair's fixed (dest_pe, nic_idx).
 *
 * Because every QP talks to exactly one peer over exactly one NIC, all the
 * per-PE / per-NIC indexing can be resolved on the host at registration time,
 * leaving the device with a self-contained 32-byte record: matching an address
 * and producing the remote address + keys needs no pointer chasing. The host
 * fills one entry per (pe, nic) per registration (see GDABackend's flat entry
 * table), and each QP is handed the contiguous slice for its own (dest_pe,
 * nic_idx), scanned by registration slot.
 */
struct QpSymmEntry {
  /**
   * @brief This PE's registered alias base (identical across all slices).
   *
   * Used to recognize whether a symmetric address falls in this registration
   * and to compute the intra-registration offset.
   */
  uintptr_t local_base;

  /**
   * @brief The peer's alias base for this slice's dest_pe.
   *
   * Remote address for a transfer is remote_base + (sym_addr - local_base).
   */
  uintptr_t remote_base;

  /**
   * @brief Registered length in bytes.
   */
  uint64_t length;

  /**
   * @brief Remote key for this slice's (dest_pe, nic_idx).
   */
  uint32_t rkey;

  /**
   * @brief Local key for this slice's nic_idx (locally-sourced buffers).
   */
  uint32_t lkey;
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_GDA_SYMM_TABLE_HPP_
