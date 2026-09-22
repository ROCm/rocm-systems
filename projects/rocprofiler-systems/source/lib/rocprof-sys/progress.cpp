// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "api.hpp"
#include "core/categories.hpp"
#include "core/config.hpp"
#include "library/components/category_region.hpp"
#include "library/tracing.hpp"

extern "C" void
rocprofsys_progress_hidden(const char* _name)
{
    // mark the progress point
    rocprofsys::component::category_region<rocprofsys::category::causal>::mark<
        rocprofsys::quirk::causal>(_name);
}

extern "C" void
rocprofsys_annotated_progress_hidden(const char*              _name,
                                     rocprofsys_annotation_t* _annotations,
                                     size_t                   _annotation_count)
{
    // mark the progress point
    rocprofsys::component::category_region<rocprofsys::category::causal>::mark<
        rocprofsys::quirk::causal>(_name, _annotations, _annotation_count);
}
