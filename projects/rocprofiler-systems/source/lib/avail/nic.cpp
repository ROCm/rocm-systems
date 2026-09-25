// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "avail/nic.hpp"
#include "avail/records.hpp"
#include "avail/smi_session.hpp"

#include <exception>
#include <string>

namespace rocprofsys::avail
{

nic_listing_result
query_nic_inventory(bool include_devices, bool include_metrics)
{
#if defined(ROCPROFSYS_BUILD_AINIC) && ROCPROFSYS_BUILD_AINIC == 1
    try
    {
        const smi_session session;
        return inventory::nic_inventory(session.backend(), include_devices,
                                        include_metrics);
    } catch(const std::exception& err)
    {
        return nic_listing_result{
            .issue =
                diagnostic{
                    .capability = include_devices ? capability_kind::nic_devices
                                                  : capability_kind::nic_metrics,
                    .source     = source_id::amd_smi,
                    .message    = std::string{ err.what() },
                },
        };
    }
#else
    static_cast<void>(include_devices);
    static_cast<void>(include_metrics);
    return nic_listing_result{
        .issue =
            diagnostic{
                .capability = include_devices ? capability_kind::nic_devices
                                              : capability_kind::nic_metrics,
                .source     = source_id::amd_smi,
                .message    = std::string{ k_ainic_unavailable_message },
            },
    };
#endif
}

}  // namespace rocprofsys::avail
