// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "function_signature.hpp"
#include "fwd.hpp"

#include <timemory/mpl/concepts.hpp>
#include <timemory/tpls/cereal/cereal/cereal.hpp>

#include <sstream>
#include <string>

struct module_function
{
    using width_t   = std::array<size_t, 4>;
    using address_t = Dyninst::Address;

    static constexpr size_t absolute_max_width = 80;
    static width_t&         get_width();
    static void             reset_width();
    static void             update_width(const module_function& rhs);
    static void             write_header(std::ostream& os);

    TIMEMORY_DEFAULT_OBJECT(module_function)

    module_function(module_t* mod, procedure_t* proc);

    std::pair<size_t, size_t> operator()(address_space_t* _addr_space,
                                         procedure_t*     _entr_trace,
                                         procedure_t*     _exit_trace) const;

    // applies logic for all "is_*" and "can_*" checks below
    bool should_instrument() const;

    // hard constraints
    bool is_instrumentable() const;       // checks whether can instrument
    bool can_instrument_entry() const;    // checks for entry points
    bool can_instrument_exit() const;     // checks for exit points
    bool is_module_constrained() const;   // checks module constraints
    bool is_routine_constrained() const;  // checks function constraints

    // user bypass of heuristics
    bool is_user_restricted() const;  // checks user restrict regexes
    bool is_user_included() const;    // checks user include regexes
    bool is_user_excluded() const;    // checks user exclude regexes

    // applied before dynamic-callsite constraint
    bool is_overlapping_constrained() const;  // checks overlapping constrains
    bool is_entry_trap_constrained() const;   // checks entry trap constraint
    bool is_exit_trap_constrained() const;    // checks exit trap constraint

    // applied before address range and # instruction constraints
    bool is_dynamic_callsite_forced() const;  // checks dynamic callsites

    // estimate the size/work of the function
    bool is_address_range_constrained() const;     // checks address range constraint
    bool is_num_instructions_constrained() const;  // check # instructions constraint

    size_t                                  start_address    = 0;
    uint64_t                                address_range    = 0;
    uint64_t                                num_instructions = 0;
    module_t*                               module           = nullptr;
    procedure_t*                            function         = nullptr;
    flow_graph_t*                           flow_graph       = nullptr;
    string_t                                module_name      = {};
    string_t                                function_name    = {};
    function_signature                      signature        = {};
    basic_block_set_t                       basic_blocks     = {};
    basic_loop_vec_t                        loop_blocks      = {};
    std::vector<std::vector<instruction_t>> instructions     = {};

    using str_msg_t     = std::tuple<int, string_t, string_t, string_t>;
    using str_msg_vec_t = std::vector<str_msg_t>;

    mutable str_msg_vec_t messages = {};

    bool is_overlapping() const;  // checks if func overlaps

private:
    bool is_loop_address_range_constrained() const;  // checks loop addr range constraint
    bool contains_dynamic_callsites() const;

public:
    template <typename ArchiveT>
    void serialize(ArchiveT& ar, const unsigned);

    friend bool operator<(const module_function& lhs, const module_function& rhs)
    {
        return (lhs.module_name == rhs.module_name)
                   ? ((lhs.function_name == rhs.function_name)
                          ? (lhs.signature.get() < rhs.signature.get())
                          : (lhs.function_name < rhs.function_name))
                   : (lhs.module_name < rhs.module_name);
    }

    friend bool operator==(const module_function& lhs, const module_function& rhs)
    {
        return std::tie(lhs.module_name, lhs.function_name, lhs.signature,
                        lhs.address_range, lhs.num_instructions) ==
               std::tie(rhs.module_name, rhs.function_name, rhs.signature,
                        rhs.address_range, rhs.num_instructions);
    }

    friend std::ostream& operator<<(std::ostream& os, const module_function& rhs)
    {
        std::stringstream ss;

        auto w0 = std::min<size_t>(get_width()[0], absolute_max_width);
        auto w1 = std::min<size_t>(get_width()[1], absolute_max_width);
        auto w2 = std::min<size_t>(get_width()[2], absolute_max_width);

        auto _get_str = [](const std::string& _inc) {
            if(_inc.length() > absolute_max_width)
                return _inc.substr(0, absolute_max_width - 3) + "...";
            return _inc;
        };

        std::stringstream _addr{};
        _addr << "0x" << std::hex << rhs.start_address;
        // clang-format off
        ss << std::setw(14) << _addr.str() << " "
           << std::setw(14) << rhs.address_range << " "
           << std::setw(14) << rhs.num_instructions << " "
           << std::setw(6) << std::setprecision(2) << std::fixed << (rhs.address_range / static_cast<double>(rhs.num_instructions)) << "  "
           << std::setw(w0 + 8) << std::left << _get_str(rhs.module_name) << " "
           << std::setw(w1 + 8) << std::left << _get_str(rhs.function_name) << " "
           << std::setw(w2 + 8) << std::left << _get_str(rhs.signature.get());
        // clang-format on

        os << ss.str();
        return os;
    }
};

template <typename ArchiveT>
void
module_function::serialize(ArchiveT& ar, const unsigned)
{
    namespace cereal = tim::cereal;
    if constexpr(tim::concepts::is_output_archive<ArchiveT>::value)
    {
        std::stringstream _addr{};
        _addr << "0x" << std::hex << start_address;
        ar(cereal::make_nvp("start_address", _addr.str()));
    }

    ar(cereal::make_nvp("address_range", address_range),
       cereal::make_nvp("instructions", num_instructions),
       cereal::make_nvp("module", module_name),
       cereal::make_nvp("function", function_name),
       cereal::make_nvp("signature", signature));

    if constexpr(tim::concepts::is_output_archive<ArchiveT>::value)
    {
        ar.setNextName("heuristics");
        ar.startNode();
        ar(cereal::make_nvp("should_instrument", should_instrument()),
           cereal::make_nvp("is_instrumentable", is_instrumentable()),
           cereal::make_nvp("can_instrument_entry", can_instrument_entry()),
           cereal::make_nvp("can_instrument_exit", can_instrument_exit()),
           cereal::make_nvp("contains_dynamic_callsites", contains_dynamic_callsites()),
           cereal::make_nvp("is_module_constrained", is_module_constrained()),
           cereal::make_nvp("is_routine_constrained", is_routine_constrained()),
           cereal::make_nvp("is_user_restricted", is_user_restricted()),
           cereal::make_nvp("is_user_included", is_user_included()),
           cereal::make_nvp("is_user_excluded", is_user_excluded()),
           cereal::make_nvp("is_overlapping_constrained", is_overlapping_constrained()),
           cereal::make_nvp("is_entry_trap_constrained", is_entry_trap_constrained()),
           cereal::make_nvp("is_exit_trap_constrained", is_exit_trap_constrained()),
           cereal::make_nvp("is_dynamic_callsite_forced", is_dynamic_callsite_forced()),
           cereal::make_nvp("is_address_range_constrained",
                            is_address_range_constrained()),
           cereal::make_nvp("is_loop_address_range_constrained",
                            is_loop_address_range_constrained()),
           cereal::make_nvp("is_num_instructions_constrained",
                            is_num_instructions_constrained()));
        ar.finishNode();
        // instructions can inflate JSON size so only output when verbosity is increased
        // above default
        if(debug_print || verbose_level > 3 || instr_print)
        {
            std::vector<std::vector<std::string>> _instructions{};
            _instructions.reserve(instructions.size());
            for(auto&& itr : instructions)
            {
                std::vector<std::string> _subinstr{};
                _subinstr.reserve(itr.size());
                for(auto&& iitr : itr)
                    _subinstr.emplace_back(iitr.format());
                _instructions.emplace_back(std::move(_subinstr));
            }
            ar(cereal::make_nvp("instructions", _instructions));
        }
    }
}
