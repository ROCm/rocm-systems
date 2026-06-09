// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <deque>
#include <optional>
#include <string_view>
#include <vector>

namespace profiler_hub::shared_types
{

using timestamp_ns_t = size_t;

// --------------------- Call Stack & Line Info Abstract Data Types ------------------

/**
 * @brief Memory address range representing a loaded code object region.
 */
struct address_range_info_t
{
    size_t           address_base;  ///< Base load address of the code object
    size_t           address_low;  ///< Lower bound of the address range (>= address_base)
    size_t           address_high;  ///< Upper bound of the address range (>= address_low)
    std::string_view extdata = "{}";
};

/**
 * @brief Program counter information representing a location in executable code.
 */
struct program_counter_info_t
{
    std::optional<std::string_view>
        function;  ///< Function or symbol name at this program counter
    std::optional<std::string_view> filename;  ///< Source file path (if available)
    std::optional<size_t> line_number;  ///< Line number in source file (if available)
    std::string_view      extdata = "{}";
};

/**
 * @brief A single frame in a call stack.
 */
struct stack_frame_t
{
    std::optional<program_counter_info_t>
        program_counter;                                ///< Location info for this frame
    std::optional<address_range_info_t> address_range;  ///< Code object memory range
    std::string_view                    extdata = "{}";
};

/**
 * @brief Complete call stack as an ordered collection of stack frames.
 * @note Front element (index 0) is the top of the stack (most recent call).
 * Back element is the bottom of the stack (e.g., main). Depth is implicit
 * from position in the deque. Maps to multiple rocpd_call_stack rows in schema v4.
 */
using call_stack_t = std::deque<stack_frame_t>;

/**
 * @brief Source code context containing actual source lines and disassembly.
 */
struct source_code_info_t
{
    std::optional<std::string_view> filename;  ///< Source file path
    std::optional<size_t>
        starting_line_number;  ///< First line number in source_code_lines
    std::vector<std::string_view> source_code_lines;  ///< Actual source code lines
    std::vector<std::string_view>
                     assembly_instruction_lines;  ///< Disassembled instructions
    std::string_view extdata = "{}";
};

/**
 * @brief Line info entry linking source code context with program counter location.
 */
struct line_info_entry_t
{
    std::optional<source_code_info_t>     source_code;      ///< Source code context
    std::optional<program_counter_info_t> program_counter;  ///< Code location
    std::optional<address_range_info_t>
        address_range;  ///< Code object for program_counter
};

/**
 * @brief Collection of line info entries for an event.
 * @note An event may have multiple source contexts (e.g., inlined functions,
 * multiple relevant source locations). Each entry provides source code and
 * location information. Maps to multiple rocpd_line_info rows in schema v4.
 */
using source_context_list_t = std::vector<line_info_entry_t>;

}  // namespace profiler_hub::shared_types
