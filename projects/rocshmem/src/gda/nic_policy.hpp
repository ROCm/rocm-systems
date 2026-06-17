/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

#ifndef ROCSHMEM_NIC_POLICY_HPP
#define ROCSHMEM_NIC_POLICY_HPP

#include <cassert>

namespace rocshmem {

enum class NicPolicy { ROUND_ROBIN, PER_CONTEXT };

/**
 * Map a global QP index to a NIC index.
 *
 * ROUND_ROBIN: (qp_row % num_nics) -- successive rows alternate NICs.
 * PER_CONTEXT: (ctx_id % num_nics) -- all QPs in a context share one NIC.
 */
inline int ComputeNicIdxForQp(
    int qp_idx, int num_pes, int num_nics,
    int qps_per_pe_default_ctx, int qps_per_pe_usr_ctx,
    NicPolicy policy)
{
  assert(num_pes > 0 && "num_pes must be positive");
  assert(num_nics > 0 && "num_nics must be positive");

  if (policy == NicPolicy::PER_CONTEXT) {
    int default_block = qps_per_pe_default_ctx * num_pes;
    int ctx_id = 0;
    if (qp_idx >= default_block && qps_per_pe_usr_ctx > 0) {
      int usr_block = qps_per_pe_usr_ctx * num_pes;
      ctx_id = 1 + (qp_idx - default_block) / usr_block;
    }
    return ctx_id % num_nics;
  } else {
    // ROUND_ROBIN: interleave QP rows across NICs
    return (qp_idx / num_pes) % num_nics;
  }
}

}  // namespace rocshmem

#endif  // ROCSHMEM_NIC_POLICY_HPP
