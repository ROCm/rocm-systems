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

#pragma once

#include "binary/dwarf_entry.hpp"
#include "binary/symbol.hpp"
#include "core/binary/fwd.hpp"
#include "core/debug.hpp"
#include "core/defines.hpp"
#include "library/causal/fwd.hpp"

#include <timemory/hash/types.hpp>
#include <timemory/unwind/dlinfo.hpp>
#include <timemory/unwind/stack.hpp>
#include <timemory/utility/macros.hpp>
#include <timemory/utility/procfs/maps.hpp>

#include <cstddef>
#include <cstdint>
#include <dlfcn.h>
#include <map>
#include <utility>

namespace rocprofsys
{
namespace causal
{
struct selected_entry
{
    ROCPROFSYS_DEFAULT_OBJECT(selected_entry)

    uintptr_t      address        = 0x0;
    uintptr_t      symbol_address = 0x0;
    binary::symbol symbol         = {};

    template <typename ArchiveT>
    void serialize(ArchiveT&, const unsigned int);

    bool     contains(uintptr_t) const;
    explicit operator bool() const { return (address > 0 && symbol.address); }
};

inline bool
selected_entry::contains(uintptr_t _v) const
{
    return (_v == address || (symbol_address > 0 && _v == symbol_address) ||
            symbol.ipaddr().contains(_v));
}
}  // namespace causal
}  // namespace rocprofsys
