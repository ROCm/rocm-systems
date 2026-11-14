// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/trace_cache/collection_sink.hpp"
#include "core/collection/schema.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/counter_samples.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "policies/trace_cache/buffer_storage.hpp"
#include "policies/trace_cache/metadata_registry.hpp"

namespace rocprofsys::trace_cache
{
static_assert(policies::trace_cache::record_storage<buffer_storage_t, counter_batch_view,
                                                    gap_marker>);
static_assert(policies::trace_cache::counter_schema_registry<metadata_registry,
                                                             collection::schema>);

template class collection_sink<buffer_storage_t, metadata_registry>;
}  // namespace rocprofsys::trace_cache
