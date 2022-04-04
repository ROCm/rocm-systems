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

#include "module_function.hpp"
#include "fwd.hpp"
#include "omnitrace.hpp"

module_function::width_t&
module_function::get_width()
{
    static width_t _instance = []() {
        width_t _tmp;
        _tmp.fill(0);
        return _tmp;
    }();
    return _instance;
}

void
module_function::reset_width()
{
    get_width().fill(0);
}

void
module_function::update_width(const module_function& rhs)
{
    get_width()[0] = std::max<size_t>(get_width()[0], rhs.module_name.length());
    get_width()[1] = std::max<size_t>(get_width()[1], rhs.function_name.length());
    get_width()[2] = std::max<size_t>(get_width()[2], rhs.signature.get().length());
}

module_function::module_function(module_t* mod, procedure_t* proc)
: module{ mod }
, function{ proc }
, flow_graph{ proc->getCFG() }
{
    if(flow_graph)
    {
        flow_graph->getAllBasicBlocks(basic_blocks);
        flow_graph->getOuterLoops(loop_blocks);
    }

    for(const auto& itr : basic_blocks)
    {
        std::vector<instruction_t> _instructions{};
        itr->getInstructions(_instructions);
        num_instructions += _instructions.size();
        instructions.reserve(instructions.size() + _instructions.size());
        for(auto&& iitr : _instructions)
            instructions.emplace_back(iitr);
    }

    char modname[FUNCNAMELEN];
    char fname[FUNCNAMELEN];
    module->getFullName(modname, FUNCNAMELEN);
    function->getName(fname, FUNCNAMELEN);
    module_name   = modname;
    function_name = fname;
    signature     = get_func_file_line_info(module, function);

    if(!function->isInstrumentable())
    {
        verbprintf(0,
                   "Warning! module function generated for un-instrumentable "
                   "function: %s [%s]\n",
                   function_name.c_str(), module_name.c_str());
    }
    std::pair<address_t, address_t> _range{};
    if(function->getAddressRange(_range.first, _range.second))
    {
        start_address = _range.first;
        address_range = _range.second - _range.first;
    }
}

void
module_function::write_header(std::ostream& os)
{
    auto w0 = std::min<size_t>(get_width()[0], absolute_max_width);
    auto w1 = std::min<size_t>(get_width()[1], absolute_max_width);
    auto w2 = std::min<size_t>(get_width()[2], absolute_max_width);

    std::stringstream ss;
    ss << std::setw(14) << "StartAddress"
       << " " << std::setw(14) << "AddressRange"
       << " " << std::setw(14) << "#Instructions"
       << " " << std::setw(6) << "Ratio"
       << "  " << std::setw(w0 + 8) << std::left << "Module"
       << " " << std::setw(w1 + 8) << std::left << "Function"
       << " " << std::setw(w2 + 8) << std::left << "FunctionSignature"
       << "\n";
    os << ss.str();
}

bool
module_function::should_instrument() const
{
    // hard constraints
    if(!is_instrumentable()) return false;
    if(!can_instrument_entry()) return false;
    if(!can_instrument_exit()) return false;
    if(is_module_constrained()) return false;
    if(is_routine_constrained()) return false;

    // should be before user selection
    constexpr int absolute_min_instructions = 4;
    if(num_instructions < absolute_min_instructions)
    {
        messages.emplace_back(
            2, "Skipping", "function",
            TIMEMORY_JOIN("-", "less-than", absolute_min_instructions, "instructions"));
        return false;
    }

    // user selection
    if(is_user_excluded()) return false;
    if(is_user_restricted()) return true;
    if(is_user_included()) return true;

    // should be applied before dynamic-callsite check
    if(is_overlapping_constrained()) return false;
    if(is_entry_trap_constrained()) return false;
    if(is_exit_trap_constrained()) return false;

    // needs to be applied before address range and number of instruction constraints
    if(is_dynamic_callsite_forced()) return true;

    if(is_address_range_constrained()) return false;
    if(is_num_instructions_constrained()) return false;

    return true;
}

bool
module_function::is_instrumentable() const
{
    if(!function->isInstrumentable())
    {
        messages.emplace_back(2, "Skipping", "module", "not-instrumentable");
        return false;
    }

    return true;
}

namespace
{
bool
check_regex_restrictions(const std::string& _name, const regexvec_t& _regexes)
{
    // NOLINTNEXTLINE
    for(auto& itr : _regexes)
        if(std::regex_search(_name, itr)) return true;
    return false;
}
}  // namespace

bool
module_function::is_user_restricted() const
{
    if(!file_restrict.empty())
    {
        if(check_regex_restrictions(module_name, file_restrict))
        {
            messages.emplace_back(2, "Forcing", "module", "module-restrict-regex");
            return false;
        }
        else
        {
            messages.emplace_back(3, "Skipping", "module", "module-restrict-regex");
            return true;
        }
    }

    if(!func_restrict.empty())
    {
        if(check_regex_restrictions(module_name, func_restrict))
        {
            messages.emplace_back(2, "Forcing", "function", "function-restrict-regex");
            return false;
        }
        else if(check_regex_restrictions(signature.get(), func_restrict))
        {
            messages.emplace_back(2, "Forcing", "function", "function-restrict-regex");
            return false;
        }
        else
        {
            messages.emplace_back(3, "Skipping", "function", "function-restrict-regex");
            return true;
        }
    }

    return false;
}

bool
module_function::is_user_included() const
{
    if(!file_include.empty())
    {
        if(check_regex_restrictions(module_name, file_include))
        {
            messages.emplace_back(2, "Forcing", "module", "module-include-regex");
            return true;
        }
    }

    if(!func_include.empty())
    {
        if(check_regex_restrictions(function_name, func_include))
        {
            messages.emplace_back(2, "Forcing", "function", "function-include-regex");
            return true;
        }
        else if(check_regex_restrictions(signature.get(), func_include))
        {
            messages.emplace_back(2, "Forcing", "function", "function-include-regex");
            return true;
        }
    }

    return false;
}

bool
module_function::is_user_excluded() const
{
    if(!file_exclude.empty())
    {
        if(check_regex_restrictions(module_name, file_exclude))
        {
            messages.emplace_back(2, "Skipping", "module", "module-exclude-regex");
            return true;
        }
    }

    if(!func_exclude.empty())
    {
        if(check_regex_restrictions(function_name, func_exclude))
        {
            messages.emplace_back(2, "Skipping", "function", "function-exclude-regex");
            return true;
        }
        else if(check_regex_restrictions(signature.get(), func_exclude))
        {
            messages.emplace_back(2, "Skipping", "function", "function-exclude-regex");
            return true;
        }
    }

    return false;
}

bool
module_function::is_overlapping() const
{
    procedure_vec_t _overlapping{};
    return function->findOverlapping(_overlapping);
}

bool
module_function::is_module_constrained() const
{
    if(!instrument_module(module_name) || module_constraint(module_name.c_str()))
    {
        messages.emplace_back(2, "Skipping", "module", "module-constraint");
        return true;
    }
    return false;
}

bool
module_function::is_routine_constrained() const
{
    if(!instrument_entity(function_name) || !instrument_entity(signature.get()) ||
       routine_constraint(function_name) || routine_constraint(signature.get()))
    {
        messages.emplace_back(2, "Skipping", "function", "function-constraint");
        return true;
    }
    return false;
}

bool
module_function::is_overlapping_constrained() const
{
    if(!allow_overlapping && is_overlapping())
    {
        messages.emplace_back(2, "Skipping", "function", "overlapping");
        return true;
    }

    return false;
}

bool
module_function::contains_dynamic_callsites() const
{
    if(flow_graph) return flow_graph->containsDynamicCallsites();

    return false;
}

bool
module_function::is_dynamic_callsite_forced() const
{
    if(instr_dynamic_callsites && contains_dynamic_callsites())
    {
        messages.emplace_back(2, "Forcing", "function", "dynamic-callsites");
        return true;
    }

    return false;
}

bool
module_function::is_address_range_constrained() const
{
    if(!loop_blocks.empty()) return is_loop_address_range_constrained();

    if(address_range < min_address_range)
    {
        messages.emplace_back(2, "Skipping", "function", "min-address-range");
        return true;
    }
    return false;
}

bool
module_function::is_loop_address_range_constrained() const
{
    if(loop_blocks.empty()) return false;

    if(address_range < min_loop_address_range)
    {
        messages.emplace_back(2, "Skipping", "function", "min-address-range-loop");
        return true;
    }

    return false;
}

bool
module_function::is_num_instructions_constrained() const
{
    if(num_instructions < min_instructions)
    {
        messages.emplace_back(2, "Skipping", "function", "min-instructions");
        return true;
    }

    return false;
}

bool
module_function::can_instrument_entry() const
{
    size_t _num_points = 0;
    size_t _num_traps  = 0;

    std::tie(_num_points, _num_traps) = query_instr(function, BPatch_entry);

    if(_num_points == 0)
    {
        messages.emplace_back(3, "Skipping", "function", "no-instrumentable-entry-point");
        return false;
    }

    return true;
}

bool
module_function::can_instrument_exit() const
{
    size_t _num_points = 0;
    size_t _num_traps  = 0;

    std::tie(_num_points, _num_traps) = query_instr(function, BPatch_exit);

    if(_num_points == 0)
    {
        messages.emplace_back(3, "Skipping", "function", "no-instrumentable-exit-point");
        return false;
    }

    return true;
}

bool
module_function::is_entry_trap_constrained() const
{
    if(instr_traps) return false;

    size_t _num_points = 0;
    size_t _num_traps  = 0;

    std::tie(_num_points, _num_traps) = query_instr(function, BPatch_entry);

    if(!instr_traps && (_num_points - _num_traps) == 0)
    {
        messages.emplace_back(3, "Skipping", "function",
                              "entry-point-trap-instrumentation");
        return true;
    }

    return false;
}

bool
module_function::is_exit_trap_constrained() const
{
    if(instr_traps) return false;

    size_t _num_points = 0;
    size_t _num_traps  = 0;

    std::tie(_num_points, _num_traps) = query_instr(function, BPatch_exit);

    if((_num_points - _num_traps) == 0)
    {
        messages.emplace_back(3, "Skipping", "function",
                              "exit-point-trap-instrumentation");
        return true;
    }

    return false;
}

std::pair<size_t, size_t>
module_function::operator()(address_space_t* _addr_space, procedure_t* _entr_trace,
                            procedure_t* _exit_trace) const
{
    std::pair<size_t, size_t> _count = { 0, 0 };

    auto _name       = signature.get();
    auto _trace_entr = omnitrace_call_expr(_name.c_str());
    auto _trace_exit = omnitrace_call_expr(_name.c_str());
    auto _entr       = _trace_entr.get(_entr_trace);
    auto _exit       = _trace_exit.get(_exit_trace);

    if(insert_instr(_addr_space, function, _entr, BPatch_entry, nullptr, nullptr,
                    instr_traps) &&
       insert_instr(_addr_space, function, _exit, BPatch_exit, nullptr, nullptr,
                    instr_traps))
    {
        messages.emplace_back(1, "Instrumenting", "function", "no-constraint");
        ++_count.first;
    }

    for(auto* itr : loop_blocks)
    {
        if(!loop_level_instr) continue;

        auto _is_constrained = [this](bool _v, const std::string& _label) {
            if(!_v)
            {
                messages.emplace_back(3, "Skipping", "function", _label);
                return true;
            }
            return false;
        };

        size_t _points             = 0;
        size_t _ntraps             = 0;
        std::tie(_points, _ntraps) = query_instr(function, BPatch_entry, flow_graph, itr);

        if(_is_constrained(_points == 0, "no-instrumentable-loop-entry-point")) continue;
        if(_is_constrained(!instr_traps && (_points - _ntraps) == 0,
                           "loop-entry-point-trap-instrumentation"))
            continue;

        std::tie(_points, _ntraps) = query_instr(function, BPatch_exit, flow_graph, itr);

        if(_is_constrained(_points == 0, "no-instrumentable-loop-exit-point")) continue;
        if(_is_constrained(!instr_traps && (_points - _ntraps) == 0,
                           "loop-exit-point-trap-instrumentation"))
            continue;

        auto lname  = get_loop_file_line_info(module, function, flow_graph, itr);
        auto _lname = lname.get();

        messages.emplace_back(1, "Loop Instrumenting", "function", "no-constraint");
        ++_count.second;

        auto _ltrace_entr = omnitrace_call_expr(_lname.c_str());
        auto _ltrace_exit = omnitrace_call_expr(_lname.c_str());
        auto _lentr       = _ltrace_entr.get(_entr_trace);
        auto _lexit       = _ltrace_exit.get(_exit_trace);

        insert_instr(_addr_space, function, _lentr, BPatch_entry, flow_graph, itr,
                     instr_loop_traps);
        insert_instr(_addr_space, function, _lexit, BPatch_exit, flow_graph, itr,
                     instr_loop_traps);
    }

    return _count;
}
