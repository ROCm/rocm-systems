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

    instructions.reserve(basic_blocks.size());
    for(const auto& itr : basic_blocks)
    {
        std::vector<instruction_t> _instructions{};
        itr->getInstructions(_instructions);
        num_instructions += _instructions.size();
        if(debug_print || verbose_level > 3 || instr_print)
            instructions.emplace_back(std::move(_instructions));
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
    return should_instrument(false);
}

bool
module_function::should_coverage_instrument() const
{
    return should_instrument(true);
}

bool
module_function::should_instrument(bool coverage) const
{
    // hard constraints
    if(!is_instrumentable()) return false;
    if(!can_instrument_entry()) return false;
    if(!coverage && !can_instrument_exit()) return false;
    if(is_module_constrained()) return false;
    if(is_routine_constrained()) return false;

    // should be before user selection
    constexpr int absolute_min_instructions = 2;
    if(num_instructions < absolute_min_instructions)
    {
        messages.emplace_back(
            2, "Skipping", "function",
            TIMEMORY_JOIN("-", "less-than", absolute_min_instructions, "instructions"));
        return false;
    }

    // user selection
    if(is_user_excluded()) return false;

    // should be applied before dynamic-callsite check
    if(is_overlapping_constrained()) return false;
    if(is_entry_trap_constrained()) return false;
    if(is_exit_trap_constrained()) return false;

    // needs to be applied before address range and number of instruction constraints
    if(is_dynamic_callsite_forced()) return true;

    // user selection
    if(!file_restrict.empty() || !func_restrict.empty()) return !is_user_restricted();
    if(is_user_included()) return true;

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
        if(check_regex_restrictions(function_name, func_restrict))
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
    auto regex_opts = std::regex_constants::egrep | std::regex_constants::optimize;
    auto _report    = [&](const string_t& _action, const string_t& _reason, int _lvl) {
        messages.emplace_back(_lvl, _action, "module", _reason);
        return true;
    };

    if(module_constraint(function_name.c_str())) return true;

    // always instrument these modules
    if(module_name == "DEFAULT_MODULE" || module_name == "LIBRARY_MODULE")
        return _report("Skipping", "default module", 2);

    static std::regex ext_regex{ "\\.(s|S)$", regex_opts };
    static std::regex sys_regex{ "^(s|k|e|w)_[A-Za-z_0-9\\-]+\\.(c|C)$", regex_opts };
    static std::regex sys_build_regex{ "^(\\.\\./sysdeps/|/build/)", regex_opts };
    static std::regex dyninst_regex{ "(dyninst|DYNINST|(^|/)RT[[:graph:]]+\\.c$)",
                                     regex_opts };
    static std::regex dependlib_regex{ "^(lib|)(omnitrace|pthread|caliper|gotcha|papi|"
                                       "cupti|TAU|likwid|pfm|nvperf|unwind)",
                                       regex_opts };
    static std::regex core_cmod_regex{
        "^(malloc|(f|)lock|sig|sem)[a-z_]+(|64|_r|_l)\\.c$"
    };
    static std::regex core_lib_regex{
        "^(lib|)(c|dl|dw|pthread|tcmalloc|profiler|"
        "tbbmalloc|tbbmalloc_proxy|malloc|stdc\\+\\+)(-|\\.)",
        regex_opts
    };
    static std::regex prefix_regex{ "^(_|\\.[a-zA-Z0-9])", regex_opts };

    // file extensions that should not be instrumented
    if(std::regex_search(module_name, ext_regex))
        return _report("Excluding", "file extension", 3);

    // system modules that should not be instrumented (wastes time)
    if(std::regex_search(module_name, sys_regex) ||
       std::regex_search(module_name, sys_build_regex))
        return _report("Excluding", "system module", 3);

    // dyninst modules that must not be instrumented
    if(std::regex_search(module_name, dyninst_regex))
        return _report("Excluding", "dyninst module", 3);

    // modules used by omnitrace and dependent libraries
    if(std::regex_search(module_name, core_lib_regex) ||
       std::regex_search(module_name, core_cmod_regex))
        return _report("Excluding", "core module", 3);

    // modules used by omnitrace and dependent libraries
    if(std::regex_search(module_name, dependlib_regex))
        return _report("Excluding", "dependency module", 3);

    // known set of modules whose starting sequence of characters suggest it should not be
    // instrumented (wastes time)
    if(std::regex_search(module_name, prefix_regex))
        return _report("Excluding", "prefix match", 3);

    return false;
}

bool
module_function::is_routine_constrained() const
{
    auto regex_opts = std::regex_constants::egrep | std::regex_constants::optimize;
    auto _report    = [&](const string_t& _action, const string_t& _reason, int _lvl) {
        messages.emplace_back(_lvl, _action, "function", _reason);
        return true;
    };

    if(routine_constraint(function_name.c_str())) return true;

    auto npos = std::string::npos;
    if(function_name.find("omnitrace") != npos)
    {
        return _report("Skipping", "omnitrace-function", 1);
    }

    if(function_name.find("FunctionInfo") != npos ||
       function_name.find("_L_lock") != npos || function_name.find("_L_unlock") != npos)
    {
        return _report("Skipping", "function-constraint", 2);
    }

    static std::regex exclude(
        "(omnitrace|tim::|N3tim|MPI_Init|MPI_Finalize|dyninst|tm_clones)", regex_opts);
    static std::regex exclude_cxx(
        "(std::_Sp_counted_base|std::(use|has)_facet|std::locale|::sentry|^std::_|::_(M|"
        "S)_|::basic_string[a-zA-Z,<>: ]+::_M_create|::__|::_(Alloc|State)|"
        "std::(basic_|)(ifstream|ios|istream|ostream|stream))",
        regex_opts);
    static std::regex leading("^(_|\\.|frame_dummy|transaction clone|virtual "
                              "thunk|non-virtual thunk|\\(|targ|kmp_threadprivate_)",
                              regex_opts);
    static std::regex trailing(
        "(_|\\.part\\.[0-9]+|\\.constprop\\.[0-9]+|\\.|\\.[0-9]+)$", regex_opts);
    static strset_t whole = []() {
        auto _v   = get_whole_function_names();
        auto _ret = _v;
        for(std::string _ext : { "64", "_l", "_r" })
            for(const auto& itr : _v)
                _ret.emplace(itr + _ext);
        return _ret;
    }();

    // don't instrument the functions when key is found anywhere in function name
    if(std::regex_search(function_name, exclude) ||
       std::regex_search(function_name, exclude_cxx))
    {
        return _report("Excluding", "critical", 3);
    }

    if(whole.count(function_name) > 0)
    {
        return _report("Excluding", "critical-whole-match", 3);
    }

    // don't instrument the functions when key is found at the start of the function name
    if(std::regex_search(function_name, leading))
    {
        return _report("Excluding", "recommended-leading-match", 3);
    }

    // don't instrument the functions when key is found at the end of the function name
    if(std::regex_search(function_name, trailing))
    {
        return _report("Excluding", "recommended-trailing-match", 3);
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
    if(!loop_blocks.empty()) return is_loop_num_instructions_constrained();

    if(num_instructions < min_instructions)
    {
        messages.emplace_back(2, "Skipping", "function", "min-instructions");
        return true;
    }

    return false;
}

bool
module_function::is_loop_num_instructions_constrained() const
{
    if(loop_blocks.empty()) return false;

    if(num_instructions < min_loop_instructions)
    {
        messages.emplace_back(2, "Skipping", "function", "min-instructions-loop");
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

    if(insert_instr(_addr_space, function, _entr, BPatch_entry) &&
       insert_instr(_addr_space, function, _exit, BPatch_exit))
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

void
module_function::register_source(address_space_t* _addr_space, procedure_t* _entr_trace,
                                 const std::vector<point_t*>& _entr_points) const
{
    switch(coverage_mode)
    {
        case CODECOV_FUNCTION:
        {
            auto _name = signature.get_coverage(false);
            auto _trace_entr =
                omnitrace_call_expr(signature.m_file, signature.m_name,
                                    signature.m_row.first, start_address, _name);
            auto _entr = _trace_entr.get(_entr_trace);

            if(insert_instr(_addr_space, _entr_points, _entr, BPatch_entry))
            {
                messages.emplace_back(1, "Code Coverage", "function", "no-constraint");
            }
            break;
        }
        case CODECOV_BASIC_BLOCK:
        {
            for(auto&& itr : get_basic_block_file_line_info(module, function))
            {
                auto  _start_addr = itr.second.start_address;
                auto& _signature  = itr.second.signature;
                auto  _name       = _signature.get_coverage(true);
                auto  _trace_entr =
                    omnitrace_call_expr(_signature.m_file, _signature.m_name,
                                        _signature.m_row.first, _start_addr, _name);
                auto _entr = _trace_entr.get(_entr_trace);

                if(insert_instr(_addr_space, _entr_points, _entr, BPatch_entry))
                {
                    messages.emplace_back(1, "Code Coverage", "basic_block",
                                          "no-constraint");
                }
            }
            break;
        }
        case CODECOV_NONE: break;
    }
}

std::pair<size_t, size_t>
module_function::register_coverage(address_space_t* _addr_space,
                                   procedure_t*     _entr_trace) const
{
    std::pair<size_t, size_t> _count = { 0, 0 };
    switch(coverage_mode)
    {
        case CODECOV_FUNCTION:
        {
            auto _trace_entr =
                omnitrace_call_expr(signature.m_file, signature.m_name, start_address);
            auto _entr = _trace_entr.get(_entr_trace);

            if(insert_instr(_addr_space, function, _entr, BPatch_entry))
            {
                messages.emplace_back(1, "Code Coverage", "function", "no-constraint");
                ++_count.first;
            }
            break;
        }
        case CODECOV_BASIC_BLOCK:
        {
            for(auto&& itr : get_basic_block_file_line_info(module, function))
            {
                auto  _start_addr = itr.second.start_address;
                auto& _signature  = itr.second.signature;
                auto  _trace_entr = omnitrace_call_expr(_signature.m_file,
                                                       _signature.m_name, _start_addr);
                auto  _entr       = _trace_entr.get(_entr_trace);

                if(insert_instr(_addr_space, _entr, BPatch_entry, itr.first))
                {
                    ++_count.second;
                    messages.emplace_back(1, "Code Coverage", "basic_block",
                                          "no-constraint");
                }
            }
            verbprintf(0, "Basic-block code coverage is not available yet\n");
            break;
        }
        case CODECOV_NONE: break;
    }
    return _count;
}
