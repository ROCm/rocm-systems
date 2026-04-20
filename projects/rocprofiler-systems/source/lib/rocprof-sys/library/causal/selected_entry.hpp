// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "binary/dwarf_entry.hpp"
#include "binary/symbol.hpp"
#include "core/binary/fwd.hpp"
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
