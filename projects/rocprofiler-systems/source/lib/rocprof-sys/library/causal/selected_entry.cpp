// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
