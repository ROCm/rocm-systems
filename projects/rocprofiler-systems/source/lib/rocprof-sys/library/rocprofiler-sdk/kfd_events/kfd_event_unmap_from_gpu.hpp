// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "domains_poc/domains/types.hpp"
#include "domains_poc/utils/record_printer.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <cstddef>
#include <cstdint>

namespace domains_poc::domains::buffered
{

// NOLINTBEGIN
inline void
on_kfd_event_unmap_from_gpu(rocprofiler_context_id_t      context,
                            rocprofiler_buffer_id_t       buffer_id,
                            rocprofiler_record_header_t** headers,
                            std::size_t num_headers, void* data, std::uint64_t drop_count)
{
    utils::print_records(context, buffer_id, headers, num_headers, data, drop_count);
}
// NOLINTEND

inline constexpr auto kfd_event_unmap_from_gpu = buffered_domain_definition{
    .meta =
        domain_descriptor{
            .name  = "kfd_event_unmap_from_gpu",
            .id    = ROCPROFILER_BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU,
            .mode  = collection_mode::buffered,
            .group = domain_group{ .name = "kfd_events" },
        },
    .on_records = on_kfd_event_unmap_from_gpu
};

}  // namespace domains_poc::domains::buffered
