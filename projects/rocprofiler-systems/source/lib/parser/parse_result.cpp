// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/parse_result.hpp"
#include "parser/config_store.hpp"

namespace rocprofsys::parser
{
void
apply_parsed(config_store& store, const parse_result& parsed, config_source source)
{
    for(const auto& [param_id, value] : parsed.values)
    {
        store.set(param_id, value, source);
    }
}

}  // namespace rocprofsys::parser
