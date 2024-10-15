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
#include <tuple>

struct module_function
{
    using width_t           = std::array<size_t, 4>;
    using address_t         = Dyninst::Address;
    using instr_addr_pair_t = std::pair<instruction_t, address_t>;
    using str_msg_t         = std::tuple<int, string_t, string_t, string_t, string_t>;
    using str_msg_vec_t     = std::vector<str_msg_t>;

    static constexpr size_t absolute_max_width = 80;
    static width_t&         get_width();
    static void             reset_width();
    static void             update_width(const module_function& rhs);
    static void             write_header(std::ostream& os);

    TIMEMORY_DEFAULT_OBJECT(module_function)

    module_function(module_t* mod, procedure_t* proc);

    // code coverage
    void register_source(address_space_t* _addr_space, procedure_t* _entr_trace,
                         const std::vector<point_t*>&) const;
    std::pair<size_t, size_t> register_coverage(address_space_t* _addr_space,
                                                procedure_t*     _entr_trace) const;

    // instrumentation
    std::pair<size_t, size_t> operator()(address_space_t* _addr_space,
                                         procedure_t*     _entr_trace,
                                         procedure_t*     _exit_trace) const;

    // applies logic for all "is_*" and "can_*" checks below
    bool should_instrument() const;
    bool should_coverage_instrument() const;

    // hard constraints
    bool is_instrumentable() const;        // checks whether can instrument
    bool can_instrument_entry() const;     // checks for entry points
    bool can_instrument_exit() const;      // checks for exit points
    bool is_internal_constrained() const;  // checks internal usage constraint
    bool is_module_constrained() const;    // checks module constraints
    bool is_routine_constrained() const;   // checks function constraints

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

    // user exclusion based on instructions
    bool is_instruction_constrained() const;

    // estimate the size/work of the function
    bool is_address_range_constrained() const;     // checks address range constraint
    bool is_num_instructions_constrained() const;  // check # instructions constraint

    bool is_visibility_constrained() const;
    bool is_linkage_constrained() const;

    size_t                                      start_address     = 0;
    uint64_t                                    address_range     = 0;
    uint64_t                                    num_instructions  = 0;
    module_t*                                   module            = nullptr;
    procedure_t*                                function          = nullptr;
    symtab_func_t*                              symtab_function   = nullptr;
    flow_graph_t*                               flow_graph        = nullptr;
    string_t                                    module_name       = {};
    string_t                                    function_name     = {};
    function_signature                          signature         = {};
    basic_block_set_t                           basic_blocks      = {};
    basic_loop_vec_t                            loop_blocks       = {};
    std::map<instruction_category_t, int64_t>   instruction_types = {};
    std::vector<std::vector<instr_addr_pair_t>> instructions      = {};

    mutable str_msg_vec_t messages = {};

    bool is_overlapping() const;  // checks if func overlaps

private:
    symbol_linkage_t    get_linkage() const;
    symbol_visibility_t get_visibility() const;
    bool is_loop_num_instructions_constrained() const;  // checks loop instr constraint
    bool is_loop_address_range_constrained() const;  // checks loop addr range constraint
    bool contains_dynamic_callsites() const;
    bool should_instrument(bool _coverage) const;
    bool contains_user_callsite() const;  // checks user caller regexes

public:
    template <typename ArchiveT>
    void serialize(ArchiveT& ar, const unsigned);

    friend bool operator<(const module_function& lhs, const module_function& rhs)
    {
        return std::tie(lhs.module_name, lhs.function_name, lhs.start_address,
                        lhs.address_range, lhs.num_instructions) <
               std::tie(rhs.module_name, rhs.function_name, rhs.start_address,
                        rhs.address_range, rhs.num_instructions);
    }

    friend bool operator==(const module_function& lhs, const module_function& rhs)
    {
        return std::tie(lhs.start_address, lhs.address_range, lhs.num_instructions,
                        lhs.module_name, lhs.function_name) ==
               std::tie(rhs.start_address, rhs.address_range, rhs.num_instructions,
                        rhs.module_name, rhs.function_name);
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
           << std::setw(7) << std::to_string(rhs.get_linkage()) << " "
           << std::setw(10) << std::to_string(rhs.get_visibility()) << " "
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
       cereal::make_nvp("num_instructions", num_instructions),
       cereal::make_nvp("module", module_name),
       cereal::make_nvp("function", function_name),
       cereal::make_nvp("signature", signature));

    if constexpr(tim::concepts::is_output_archive<ArchiveT>::value)
    {
        ar(cereal::make_nvp("linkage", std::to_string(get_linkage())),
           cereal::make_nvp("visibility", std::to_string(get_visibility())),
           cereal::make_nvp("num_basic_blocks", basic_blocks.size()),
           cereal::make_nvp("num_outer_loops", loop_blocks.size()));
        ar.setNextName("heuristics");
        ar.startNode();
        ar(cereal::make_nvp("should_instrument", should_instrument()),
           cereal::make_nvp("should_coverage_instrument", should_coverage_instrument()),
           cereal::make_nvp("is_instrumentable", is_instrumentable()),
           cereal::make_nvp("can_instrument_entry", can_instrument_entry()),
           cereal::make_nvp("can_instrument_exit", can_instrument_exit()),
           cereal::make_nvp("contains_dynamic_callsites", contains_dynamic_callsites()),
           cereal::make_nvp("is_internal_constrained", is_internal_constrained()),
           cereal::make_nvp("is_module_constrained", is_module_constrained()),
           cereal::make_nvp("is_routine_constrained", is_routine_constrained()),
           cereal::make_nvp("is_user_restricted", is_user_restricted()),
           cereal::make_nvp("is_user_included", is_user_included()),
           cereal::make_nvp("contains_user_callsite", contains_user_callsite()),
           cereal::make_nvp("is_user_excluded", is_user_excluded()),
           cereal::make_nvp("is_overlapping_constrained", is_overlapping_constrained()),
           cereal::make_nvp("is_entry_trap_constrained", is_entry_trap_constrained()),
           cereal::make_nvp("is_exit_trap_constrained", is_exit_trap_constrained()),
           cereal::make_nvp("is_dynamic_callsite_forced", is_dynamic_callsite_forced()),
           cereal::make_nvp("is_linkage_constrained", is_linkage_constrained()),
           cereal::make_nvp("is_visibility_constrained", is_visibility_constrained()),
           cereal::make_nvp("is_address_range_constrained",
                            is_address_range_constrained()),
           cereal::make_nvp("is_num_instructions_constrained",
                            is_num_instructions_constrained()),
           cereal::make_nvp("is_instruction_constrained", is_instruction_constrained()),
           cereal::make_nvp("is_loop_address_range_constrained",
                            is_loop_address_range_constrained()),
           cereal::make_nvp("is_loop_num_instructions_constrained",
                            is_loop_num_instructions_constrained()));
        ar.finishNode();
        ar.setNextName("instruction_breakdown");
        ar.startNode();
        for(auto itr : instruction_types)
            ar(cereal::make_nvp(std::to_string(itr.first).c_str(), itr.second));
        ar.finishNode();
        // instructions can inflate JSON size so only output when verbosity is increased
        // above default
        if(debug_print || verbose_level > 3 || instr_print)
        {
            ar.setNextName("instructions");
            ar.startNode();
            ar.makeArray();
            for(auto&& itr : instructions)
            {
                ar.startNode();
                for(auto&& iitr : itr)
                {
                    std::stringstream _addr{};
                    _addr << "0x" << std::hex << iitr.second;
                    ar(cereal::make_nvp(_addr.str().c_str(), iitr.first.format()));
                }
                ar.finishNode();
            }
            ar.finishNode();
        }
    }
}
