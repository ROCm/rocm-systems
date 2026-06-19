// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/tracing/annotation.hpp"

namespace rocprofsys
{
namespace tracing
{
void
add_perfetto_annotation(perfetto_event_context_t&      ctx,
                        const rocprofsys_annotation_t& _annotation)
{
    add_perfetto_annotation(
        ctx, _annotation, utility::make_index_sequence_range<1, ROCPROFSYS_VALUE_LAST>{});
}
}  // namespace tracing
}  // namespace rocprofsys
