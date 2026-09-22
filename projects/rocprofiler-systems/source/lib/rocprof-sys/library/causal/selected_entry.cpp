// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/causal/selected_entry.hpp"
#include "core/common.hpp"
#include "core/timemory.hpp"

namespace rocprofsys
{
namespace causal
{
template <typename ArchiveT>
void
selected_entry::serialize(ArchiveT& ar, const unsigned int)
{
    using ::tim::cereal::make_nvp;
    ar(make_nvp("address", address), make_nvp("symbol_address", symbol_address),
       make_nvp("info", symbol));
}

template void
selected_entry::serialize<cereal::JSONInputArchive>(cereal::JSONInputArchive&,
                                                    const unsigned int);

template void
selected_entry::serialize<cereal::MinimalJSONOutputArchive>(
    cereal::MinimalJSONOutputArchive&, const unsigned int);

template void
selected_entry::serialize<cereal::PrettyJSONOutputArchive>(
    cereal::PrettyJSONOutputArchive&, const unsigned int);
}  // namespace causal
}  // namespace rocprofsys
