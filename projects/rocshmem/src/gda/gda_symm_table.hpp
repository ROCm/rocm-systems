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
 * @brief A single symmetrically-registered user buffer for the GDA backend.
 *
 * Unlike the symmetric heap (whose per-PE bases and NIC keys are exchanged
 * once at init), each registered alias is a freshly reserved virtual address
 * that differs per PE, so the remote bases and remote keys must be exchanged
 * per registration. All pointer members reference device-resident arrays.
 */
struct GDASymmRegion {
  /**
   * @brief This PE's registered alias base address.
   *
   * Used to recognize whether a symmetric address falls in this region and to
   * compute the intra-region offset.
   */
  uintptr_t local_base;

  /**
   * @brief Registered length in bytes.
   */
  size_t length;

  /**
   * @brief Device array[num_pes] of each peer's alias base address.
   *
   * The remote address for a transfer to PE p is
   * remote_bases[p] + (sym_addr - local_base).
   */
  uintptr_t *remote_bases;

  /**
   * @brief Device array[num_pes * num_nics] of peer remote keys.
   *
   * Indexed as rkeys[pe * num_nics + nic_idx], mirroring the symmetric heap's
   * heap_rkey layout, so a QP selects the remote key for its own NIC.
   */
  uint32_t *rkeys;

  /**
   * @brief Device array[num_nics] of this PE's local keys.
   *
   * Indexed by the issuing QP's NIC so a locally-sourced buffer supplies the
   * matching lkey.
   */
  uint32_t *lkeys;
};

/**
 * @brief Device-visible table of symmetric user-buffer registrations.
 *
 * Allocated once in device memory; its contents are mutated by the
 * (collective) register/unregister calls. The pointer is shared by all
 * contexts so updates are observed without re-propagation. @c regions points
 * to a device-resident array of @c capacity entries (configured via
 * ROCSHMEM_MAX_SYMM_REGIONS, see envvar::max_symm_regions).
 */
struct GDASymmTable {
  int count;
  int capacity;
  int num_nics;
  GDASymmRegion *regions;
};

/**
 * @brief Shared, device-visible view of the symmetric address space used to
 * resolve remote addresses and NIC keys.
 *
 * Bundles the process-global inputs a QueuePair needs to translate a symmetric
 * address for a peer: the per-PE symmetric heap bases and the registered-buffer
 * table. A single instance is owned by the backend and pointed to by every
 * QueuePair (see QueuePair::addr_space_). Because the members are pointers, the
 * QPs observe register/unregister updates without re-propagation. The per-QP
 * NIC selection (rkey/lkey/nic_idx) stays on the QueuePair itself.
 */
struct SymmAddrSpace {
  char *const *heap_bases{nullptr};   // device array[num_pes] of per-PE heap bases
  GDASymmTable *symm_table{nullptr};  // registration table (null when unavailable)
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_GDA_SYMM_TABLE_HPP_
