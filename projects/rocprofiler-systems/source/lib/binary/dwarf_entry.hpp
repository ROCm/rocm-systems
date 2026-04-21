// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/binary/address_range.hpp"
#include "core/binary/fwd.hpp"

namespace rocprofsys
{
namespace binary
{
struct dwarf_entry
{
    // tuple of dwarf line info, address ranges, and breakpoints
    using dwarf_tuple_t = std::tuple<std::deque<dwarf_entry>, std::vector<address_range>,
                                     std::vector<uintptr_t>>;

    ROCPROFSYS_DEFAULT_OBJECT(dwarf_entry)

    bool          begin_statement = false;
    bool          end_sequence    = false;
    bool          line_block      = false;
    bool          prologue_end    = false;
    bool          epilogue_begin  = false;
    unsigned int  line            = 0;
    int           col             = 0;
    unsigned int  vliw_op_index   = 0;
    unsigned int  isa             = 0;
    unsigned int  discriminator   = 0;
    address_range address         = { 0, 0 };
    std::string   file            = {};

    bool is_valid() const;

    bool     operator<(const dwarf_entry&) const;
    bool     operator==(const dwarf_entry&) const;
    bool     operator!=(const dwarf_entry&) const;
    explicit operator bool() const { return is_valid(); }

    static dwarf_tuple_t process_dwarf(int _fd);

    template <typename ArchiveT>
    void serialize(ArchiveT&, const unsigned int);
};
}  // namespace binary
}  // namespace rocprofsys
