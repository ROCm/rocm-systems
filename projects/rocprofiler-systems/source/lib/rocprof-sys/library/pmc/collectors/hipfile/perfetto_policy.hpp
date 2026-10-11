// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/pmc/collectors/hipfile/types.hpp"

#include <cstddef>
#include <cstdint>

namespace rocprofsys::pmc::collectors::hipfile
{

/**
 * @brief No-op Perfetto policy for the hipFile collector.
 *
 * The AMD SMI GPU, NIC, and CPU collectors predate the PMC path and still write their
 * Perfetto counter tracks directly through a per-collector Perfetto policy, gated on
 * ROCPROFSYS_USE_PERFETTO. hipFile has no such legacy path: its tracks reach Perfetto
 * from the same hipfile_pmc_sample records that populate RocPD, so there is exactly one
 * producer per track.
 *
 * base::collector calls PerfettoApi::store_sample directly rather than through the
 * traits, so this type exists to satisfy that call and do nothing. Without it, enabling
 * Perfetto would emit each hipFile counter twice.
 */
struct perfetto_policy
{
    static void store_sample(std::size_t /*device_id*/, const metrics& /*values*/,
                             std::uint64_t /*timestamp*/)
    {}
};

}  // namespace rocprofsys::pmc::collectors::hipfile
