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

#include "library/causal/data.hpp"
#include "binary/address_multirange.hpp"
#include "binary/analysis.hpp"
#include "binary/binary_info.hpp"
#include "binary/link_map.hpp"
#include "binary/scope_filter.hpp"
#include "core/binary/fwd.hpp"
#include "core/config.hpp"
#include "core/containers/c_array.hpp"
#include "core/debug.hpp"
#include "core/state.hpp"
#include "core/utility.hpp"
#include "library/causal/delay.hpp"
#include "library/causal/experiment.hpp"
#include "library/causal/fwd.hpp"
#include "library/causal/sample_data.hpp"
#include "library/causal/sampling.hpp"
#include "library/causal/selected_entry.hpp"
#include "library/ptl.hpp"
#include "library/runtime.hpp"
#include "library/thread_data.hpp"
#include "library/thread_info.hpp"

#include <timemory/data/atomic_ring_buffer.hpp>
#include <timemory/hash/types.hpp>
#include <timemory/log/logger.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/units.hpp>
#include <timemory/unwind/dlinfo.hpp>
#include <timemory/unwind/processed_entry.hpp>
#include <timemory/utility/procfs/maps.hpp>
#include <timemory/utility/types.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <random>
#include <regex>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

namespace rocprofsys
{
namespace causal
{
namespace
{
using random_engine_t    = std::mt19937_64;
using progress_bundles_t = component_bundle_cache<component::progress_point>;

auto speedup_seeds     = std::vector<size_t>{};
auto speedup_divisions = get_env<uint16_t>("ROCPROFSYS_CAUSAL_SPEEDUP_DIVISIONS", 5);
auto speedup_dist      = []() {
    size_t                _n = std::max<size_t>(1, 100 / speedup_divisions);
    std::vector<uint16_t> _v(_n, uint16_t{ 0 });
    std::generate(_v.begin(), _v.end(),
                       [_value = 0]() mutable { return (_value += speedup_divisions); });
    // approximately 25% of bins should be zero speedup
    size_t _nzero = std::ceil(_v.size() / 4.0);
    _v.resize(_v.size() + _nzero, 0);
    std::sort(_v.begin(), _v.end());
    ROCPROFSYS_CI_THROW(_v.back() > 100, "Error! last value is too large: %i\n",
                             (int) _v.back());
    return _v;
}();

auto perform_experiment_impl_completed = std::unique_ptr<std::promise<void>>{};
auto num_progress_points               = std::atomic<size_t>{ 0 };

auto&
get_progress_bundles(int64_t _tid = utility::get_thread_index())
{
    return progress_bundles_t::instance(construct_on_thread{ _tid });
}

template <typename ContextT>
auto&
get_engine()
{
    static auto _seed = []() -> hash_value_t {
        auto _seed_v =
            config::get_setting_value<uint64_t>("ROCPROFSYS_CAUSAL_RANDOM_SEED")
                .value_or(0);
        if(_seed_v == 0) _seed_v = std::random_device{}();
        return _seed_v;
    }();

    static thread_local auto _v =
        random_engine_t{ tim::get_hash_id(_seed, utility::get_thread_index()) };
    return _v;
}

binary::address_multirange&
get_eligible_address_ranges()
{
    static auto _v = binary::address_multirange{};
    return _v;
}

using sf = binary::scope_filter;

auto
get_filters(const std::set<binary::scope_filter::filter_scope>& _scopes = {
                sf::BINARY_FILTER, sf::SOURCE_FILTER, sf::FUNCTION_FILTER })
{
    auto _filters = std::vector<binary::scope_filter>{};

    // exclude internal libraries used by rocprof-sys
    if(_scopes.count(sf::BINARY_FILTER) > 0)
        _filters.emplace_back(sf{ sf::FILTER_EXCLUDE, sf::BINARY_FILTER,
                                  "lib(rocprof-sys[-\\.]|dyninst|"
                                  "tbbmalloc|gotcha\\.|unwind\\.so\\.99)" });

    // in function mode, it generally doesn't help to experiment on main function since
    // telling the user to "make the main function" faster is literally useless since it
    // contains everything that could be made faster
    if(config::get_causal_mode() == CausalMode::Function &&
       _scopes.count(sf::FUNCTION_FILTER) > 0)
        _filters.emplace_back(sf{ sf::FILTER_EXCLUDE, sf::FUNCTION_FILTER,
                                  "( main\\(|^main$|^main\\.cold$)" });

    bool _use_default_excludes =
        config::get_setting_value<bool>("ROCPROFSYS_CAUSAL_FUNCTION_EXCLUDE_DEFAULTS")
            .value_or(true);

    if(_use_default_excludes && _scopes.count(sf::FUNCTION_FILTER) > 0)
    {
        // symbols starting with leading underscore are generally system functions
        _filters.emplace_back(sf{ sf::FILTER_EXCLUDE, sf::FUNCTION_FILTER, "^_" });

        if(config::get_causal_mode() == CausalMode::Function)
        {
            // exclude STL implementation functions
            _filters.emplace_back(sf{ sf::FILTER_EXCLUDE, sf::FUNCTION_FILTER, "::_M" });
        }
    }

    // in function mode, it generally doesn't help to claim
    // "make main function" faster since it contains everything
    // that could be made faster
    if(config::get_causal_mode() == CausalMode::Function &&
       _scopes.count(sf::FUNCTION_FILTER) > 0)
    {
        _filters.emplace_back(sf{ sf::FILTER_EXCLUDE, sf::FUNCTION_FILTER,
                                  "(^main$|^main.cold$|int main\\()" });
    }

    using utility::get_regex_or;

    auto _source_end_converter = [](const std::string& _v) { return _v + "$"; };

    // include handling
    {
        auto _binary_include = get_regex_or(config::get_causal_binary_scope(), "");
        auto _source_include =
            get_regex_or(config::get_causal_source_scope(), _source_end_converter, "");
        auto _function_include = get_regex_or(config::get_causal_function_scope(), "");

        auto _current_include =
            std::make_tuple(_binary_include, _source_include, _function_include);
        static auto _former_include = decltype(_current_include){};

        if(_former_include != _current_include)
        {
            if(!_binary_include.empty())
                ROCPROFSYS_VERBOSE(0, "[causal] binary scope     : %s\n",
                                   _binary_include.c_str());
            if(!_source_include.empty())
                ROCPROFSYS_VERBOSE(0, "[causal] source scope     : %s\n",
                                   _source_include.c_str());
            if(!_function_include.empty())
                ROCPROFSYS_VERBOSE(0, "[causal] function scope   : %s\n",
                                   _function_include.c_str());
            _former_include = _current_include;
        }

        if(!_binary_include.empty() && _scopes.count(sf::BINARY_FILTER) > 0)
            _filters.emplace_back(
                sf{ sf::FILTER_INCLUDE, sf::BINARY_FILTER, _binary_include });

        if(!_source_include.empty() && _scopes.count(sf::SOURCE_FILTER) > 0)
            _filters.emplace_back(
                sf{ sf::FILTER_INCLUDE, sf::SOURCE_FILTER, _source_include });

        if(!_function_include.empty() && _scopes.count(sf::FUNCTION_FILTER) > 0)
            _filters.emplace_back(
                sf{ sf::FILTER_INCLUDE, sf::FUNCTION_FILTER, _function_include });
    }

    // exclude handling
    {
        auto _binary_exclude = get_regex_or(config::get_causal_binary_exclude(), "");
        auto _source_exclude =
            get_regex_or(config::get_causal_source_exclude(), _source_end_converter, "");
        auto _function_exclude = get_regex_or(config::get_causal_function_exclude(), "");

        auto _current_exclude =
            std::make_tuple(_binary_exclude, _source_exclude, _function_exclude);
        static auto _former_exclude = decltype(_current_exclude){};

        if(_former_exclude != _current_exclude)
        {
            if(!_binary_exclude.empty())
                ROCPROFSYS_VERBOSE(0, "[causal] binary exclude   : %s\n",
                                   _binary_exclude.c_str());
            if(!_source_exclude.empty())
                ROCPROFSYS_VERBOSE(0, "[causal] source exclude   : %s\n",
                                   _source_exclude.c_str());
            if(!_function_exclude.empty())
                ROCPROFSYS_VERBOSE(0, "[causal] function exclude : %s\n",
                                   _function_exclude.c_str());
            _former_exclude = _current_exclude;
        }

        if(!_binary_exclude.empty() && _scopes.count(sf::BINARY_FILTER) > 0)
            _filters.emplace_back(
                sf{ sf::FILTER_EXCLUDE, sf::BINARY_FILTER, _binary_exclude });

        if(!_source_exclude.empty() && _scopes.count(sf::SOURCE_FILTER) > 0)
            _filters.emplace_back(
                sf{ sf::FILTER_EXCLUDE, sf::SOURCE_FILTER, _source_exclude });

        if(!_function_exclude.empty() && _scopes.count(sf::FUNCTION_FILTER) > 0)
            _filters.emplace_back(
                sf{ sf::FILTER_EXCLUDE, sf::FUNCTION_FILTER, _function_exclude });
    }

    return _filters;
}

using binary_info_t = std::vector<binary::binary_info>;
std::pair<binary_info_t, binary_info_t>&
get_cached_binary_info()
{
    static auto _v = []() {
        // get the linked binaries for the exe (excluding ones from librocprof-sys)
        auto _link_map = binary::get_link_map();
        auto _files    = std::vector<std::string>{};
        _files.reserve(_link_map.size());
        for(const auto& itr : _link_map)
            _files.emplace_back(itr.real());

        auto _discarded = std::vector<binary::binary_info>{};
        auto _requested = binary::get_binary_info(_files, get_filters());
        return std::make_pair(_requested, _discarded);
    }();
    return _v;
}

bool
satisfies_filter(const binary::scope_filter::filter_scope& _scope,
                 const std::string&                        _value)
{
    static auto _filters = get_filters();
    return binary::scope_filter::satisfies_filter(_filters, _scope, _value);
}

auto
compute_eligible_lines_impl()
{
    const auto& _binary_info = get_cached_binary_info().first;
    auto&       _scoped_info = get_cached_binary_info().second;
    auto        _filters     = get_filters();

    for(const auto& litr : _binary_info)
    {
        auto& _scoped    = _scoped_info.emplace_back();
        _scoped.bfd      = litr.bfd;
        _scoped.mappings = litr.mappings;
        _scoped.sections = litr.sections;

        for(const auto& ditr : litr.symbols)
        {
            auto _sym = ditr.clone();

            _sym.inlines =
                ditr.get_inline_symbols<std::vector<binary::inlined_symbol>>(_filters);

            _sym.dwarf_info =
                ditr.get_debug_line_info<std::vector<binary::dwarf_entry>>(_filters);

            if(ditr(_filters) || (_sym.inlines.size() + _sym.dwarf_info.size()) > 0)
            {
                _scoped.ranges.emplace_back(_sym.ipaddr());
                _scoped.symbols.emplace_back(_sym);
            }
        }

        for(const auto& ditr : litr.debug_info)
        {
            if(sf::satisfies_filter(_filters, sf::SOURCE_FILTER, ditr.file) ||
               sf::satisfies_filter(_filters, sf::SOURCE_FILTER,
                                    join(':', ditr.file, ditr.line)))
            {
                _scoped.debug_info.emplace_back(ditr);
            }
        }

        _scoped.sort();
    }

    auto& _eligible_ar = get_eligible_address_ranges();
    for(const auto& litr : _scoped_info)
    {
        for(const auto& ditr : litr.symbols)
        {
            _eligible_ar += ditr.ipaddr();
        }

        for(auto ditr : litr.ranges)
        {
            _eligible_ar += ditr;
        }
    }

    ROCPROFSYS_VERBOSE(
        0, "[causal] eligible address ranges: %zu, coarse address range: %zu [%s]\n",
        _eligible_ar.size(), _eligible_ar.range_size(),
        _eligible_ar.get_coarse_range().as_string().c_str());

    if(_eligible_ar.empty())
    {
        auto _cfg         = settings::compose_filename_config{};
        _cfg.subdirectory = "causal/binary-info";
        _cfg.use_suffix   = config::get_use_pid();
        save_line_info(_cfg, config::get_verbose());
    }

    ROCPROFSYS_CONDITIONAL_THROW(
        _eligible_ar.empty(),
        "Error! binary analysis (after filters) resulted in zero eligible instruction "
        "pointer addresses for causal experimentation");
}

void
save_maps_info_impl(std::ostream& _ofs)
{
    auto _maps_file = join("/", "/proc", process::get_id(), "maps");
    auto _ifs       = std::ifstream{ _maps_file };
    auto _maps      = std::stringstream{};
    if(_ifs)
    {
        _maps << _maps_file << "\n";
        while(_ifs)
        {
            std::string _line{};
            getline(_ifs, _line);
            if(!_line.empty()) _maps << "    " << _line << "\n";
        }
    }
    _ofs << _maps.str();
}

void
save_line_info_impl(std::ostream&                           _ofs,
                    const std::vector<binary::binary_info>& _binary_data,
                    const std::array<bool, 3>&              _info = { true, true, true })
{
    auto _write_impl = [&_ofs, &_info](const binary::binary_info& _data) {
        for(const auto& itr : _data.mappings)
        {
            _ofs << itr.pathname << " [" << as_hex(itr.load_address) << " - "
                 << as_hex(itr.last_address) << "]\n";
        }

        auto _emitted_dwarf_addresses = std::set<uintptr_t>{};
        for(const auto& itr : _data.symbols)
        {
            auto _addr     = itr.address;
            auto _addr_off = itr.address + itr.load_address;
            _ofs << "    " << as_hex(_addr_off) << " [" << as_hex(_addr)
                 << "] :: " << itr.file;
            if(itr.line > 0) _ofs << ":" << itr.line;
            if(!itr.func.empty()) _ofs << " [" << tim::demangle(itr.func) << "]";
            _ofs << "\n";

            if(std::get<0>(_info))
            {
                for(const auto& ditr : itr.inlines)
                {
                    _ofs << "        " << ditr.file << ":" << ditr.line;
                    if(!ditr.func.empty())
                        _ofs << " [" << tim::demangle(ditr.func) << "]";
                    _ofs << "\n";
                }
            }

            if(std::get<1>(_info))
            {
                for(const auto& ditr : itr.dwarf_info)
                {
                    _ofs << "        " << as_hex(ditr.address) << " :: " << ditr.file
                         << ":" << ditr.line;
                    _ofs << "\n";
                    _emitted_dwarf_addresses.emplace(ditr.address.low);
                }
            }
        }

        if(std::get<2>(_info))
        {
            for(const auto& itr : _data.debug_info)
            {
                if(_emitted_dwarf_addresses.count(itr.address.low) > 0) continue;
                _ofs << "    " << as_hex(itr.address) << " :: " << itr.file << ":"
                     << itr.line;
                _ofs << "\n";
            }
        }

        _ofs << "\n" << std::flush;
    };

    for(const auto& itr : _binary_data)
        _write_impl(itr);
}

void
compute_eligible_lines()
{
    static auto _once = std::once_flag{};
    std::call_once(_once, []() {
        compute_eligible_lines_impl();
        auto _cfg         = settings::compose_filename_config{};
        _cfg.subdirectory = "causal/binary-info";
        _cfg.use_suffix   = config::get_use_pid();
        save_line_info(_cfg, config::get_verbose());
    });
}

auto eligible_pc_history    = std::map<uintptr_t, size_t>{};
auto eligible_pc_idx        = std::atomic<size_t>{ 0 };
auto eligible_pc_candidates = std::atomic<size_t>{ 0 };

void
perform_experiment_impl(std::shared_ptr<std::promise<void>> _started)  // NOLINT
{
    using clock_type      = std::chrono::high_resolution_clock;
    using duration_nsec_t = std::chrono::duration<double, std::nano>;
    using duration_sec_t  = std::chrono::duration<double, std::ratio<1>>;

    const auto& _thr_info = thread_info::init(true);
    set_thread_state(ThreadState::Disabled);
    ROCPROFSYS_CONDITIONAL_THROW(!_thr_info->is_offset,
                                 "Error! causal profiling thread should be offset");

    if(!perform_experiment_impl_completed)
        perform_experiment_impl_completed = std::make_unique<std::promise<void>>();

    perform_experiment_impl_completed->set_value_at_thread_exit();

    compute_eligible_lines();

    // notify that thread has started
    if(_started) _started->set_value();

    if(!config::get_causal_end_to_end())
    {
        // wait for at least one progress point to start
        while(num_progress_points.load(std::memory_order_relaxed) == 0)
        {
            std::this_thread::yield();
            std::this_thread::sleep_for(std::chrono::milliseconds{ 1 });
        }
    }

    // allow ~10 samples to be collected
    std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds{ 10 });

    double _delay_sec =
        config::get_setting_value<double>("ROCPROFSYS_CAUSAL_DELAY").value_or(0.0);
    double _duration_sec =
        config::get_setting_value<double>("ROCPROFSYS_CAUSAL_DURATION").value_or(0.0);
    auto _duration_nsec = duration_nsec_t{ _duration_sec * units::sec };

    if(_delay_sec > 0.0)
    {
        ROCPROFSYS_VERBOSE(1, "[causal] delaying experimentation for %.2f seconds...\n",
                           _delay_sec);
        uint64_t _delay_nsec = _delay_sec * units::sec;
        std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::nanoseconds{ _delay_nsec });
    }

    auto _impl_count        = 0;
    auto _start_time        = clock_type::now();
    auto _exceeded_duration = [&]() {
        if(_duration_sec > 1.0e-3)
        {
            auto _elapsed = clock_type::now() - _start_time;
            if(_elapsed >= _duration_nsec)
            {
                ROCPROFSYS_VERBOSE(
                    1,
                    "[causal] stopping experimentation after %.2f seconds "
                    "(elapsed: %.2f seconds)...\n",
                    _duration_sec,
                    std::chrono::duration_cast<duration_sec_t>(_elapsed).count());
                causal::sampling::post_process();
                return true;
            }
        }
        return false;
    };

    while(get_state() < State::Finalized)
    {
        auto _impl_no = _impl_count++;
        auto _experim = experiment{};

        // loop until started or finalized
        while(!_experim.start())
        {
            if(get_state() == State::Finalized)
            {
                if(_impl_no > 0) return;

                ROCPROFSYS_VERBOSE(
                    0,
                    "[causal] experiment failed to start. Number of PC candidates: %zu\n",
                    eligible_pc_candidates.load());

                auto _memory   = std::stringstream{};
                auto _binary   = std::stringstream{};
                auto _scoped   = std::stringstream{};
                auto _sample   = std::stringstream{};
                auto _eligible = std::stringstream{};
                save_maps_info_impl(_memory);
                save_line_info_impl(_binary, get_cached_binary_info().first,
                                    { true, true, false });
                save_line_info_impl(_scoped, get_cached_binary_info().second,
                                    { true, true, false });

                auto _samples_map = std::map<uintptr_t, size_t>{};
                for(const auto& itr : get_samples())
                {
                    for(const auto& iitr : itr.second)
                    {
                        _samples_map[iitr.address] += iitr.count;
                    }
                }

                auto _eligible_pc_hist = std::vector<std::pair<uintptr_t, size_t>>{};
                for(const auto& itr : eligible_pc_history)
                {
                    _eligible_pc_hist.emplace_back(std::make_pair(itr.first, itr.second));
                }

                std::sort(
                    _eligible_pc_hist.begin(), _eligible_pc_hist.end(),
                    [](auto&& _lhs, auto&& _rhs) { return _lhs.second > _rhs.second; });

                for(const auto& itr : _eligible_pc_hist)
                {
                    _eligible << "    " << std::setw(8) << itr.second
                              << " :: " << as_hex(itr.first) << "\n";
                }

                auto _samples = std::vector<std::pair<uintptr_t, size_t>>{};
                for(const auto& itr : _samples_map)
                    _samples.emplace_back(std::make_pair(itr.first, itr.second));

                // sort by most samples
                std::sort(_samples.begin(), _samples.end(),
                          [](const auto& _lhs, const auto& _rhs) {
                              return _lhs.second > _rhs.second;
                          });

                for(const auto& itr : _samples)
                {
                    if(itr.second > 0)
                    {
                        auto _is_eligible = is_eligible_address(itr.first) &&
                                            !get_line_info(itr.first, false).empty();
                        auto _linfo = binary::lookup_ipaddr_entry<true>(itr.first);
                        if(_linfo)
                        {
                            _sample << "    " << std::setw(8) << itr.second
                                    << " :: " << std::setw(5) << std::boolalpha
                                    << _is_eligible << " :: " << as_hex(itr.first) << " "
                                    << _linfo->location << ":" << _linfo->lineno << " ["
                                    << demangle(_linfo->name) << "]\n";
                            for(const auto& iitr : _linfo->lineinfo.lines)
                            {
                                _sample << "    " << std::setw(8) << itr.second
                                        << " :: " << std::setw(5) << std::boolalpha
                                        << _is_eligible << " :: " << as_hex(itr.first)
                                        << " " << iitr.location << ":" << iitr.line
                                        << " [" << demangle(iitr.name) << "]\n";
                            }
                        }
                    }
                }

                ROCPROFSYS_PRINT_COLOR(fatal, "causal experiment never started\n");

                std::cerr << std::flush;
                auto _cerr = tim::log::warning_stream(std::cerr);
                _cerr << "\npc samples:\n\n" << _sample.str() << "\n";
                _cerr << "\neligible pcs:\n\n" << _eligible.str() << "\n";
                _cerr << "\nscoped pcs:\n\n" << _scoped.str() << "\n";
                if(get_verbose() >= 1)
                {
                    _cerr << "\nbinary pcs:\n\n" << _binary.str() << "\n";
                    _cerr << "\nmaps:\n\n" << _memory.str() << "\n";
                }
                std::cerr << std::flush;

                // if launched via rocprof-sys-causal, allow end-to-end runs that do not
                // start experiments
                auto _omni_causal_launcher =
                    get_env<std::string>("ROCPROFSYS_LAUNCHER", "", false) ==
                    "rocprof-sys-causal";

                if(!(get_causal_end_to_end() && _omni_causal_launcher))
                {
                    ROCPROFSYS_CONDITIONAL_THROW(_impl_no == 0,
                                                 "causal experiment never started");
                }

                return;
            }
            else
            {
                ROCPROFSYS_VERBOSE(
                    1,
                    "[causal] experiment failed to start. Number of PC candidates: %zu\n",
                    eligible_pc_candidates.load());
            }
        }

        ROCPROFSYS_VERBOSE(3,
                           "[causal] experiment started. Number of PC candidates: %zu\n",
                           eligible_pc_candidates.load());

        reset_sample_selection();

        // wait for the experiment to complete
        if(config::get_causal_end_to_end())
        {
            mark_progress_point(config::get_exe_name(), true);
            while(get_state() < State::Finalized)
            {
                std::this_thread::yield();
                std::this_thread::sleep_for(std::chrono::milliseconds{ 100 });
                if(_exceeded_duration()) return;
            }
        }
        else
        {
            _experim.wait();
        }

        while(!_experim.stop())
        {
            if(get_state() == State::Finalized) return;
        }

        if(_exceeded_duration()) return;
    }
}

// latest_eligible_pcs is an array of unwind_depth size -> samples will
// use lowest indexes for most recent functions address in the call-stack
auto latest_eligible_pc = []() {
    using atomic_uintptr_t      = std::atomic<uintptr_t>;
    constexpr size_t array_size = unwind_depth;
    auto             _arr = std::array<std::unique_ptr<atomic_uintptr_t>, array_size>{};
    for(auto& itr : _arr)
        itr = std::make_unique<std::atomic<uintptr_t>>(0);
    return _arr;
}();
}  // namespace

//--------------------------------------------------------------------------------------//

bool
is_eligible_address(uintptr_t _v)
{
    return get_eligible_address_ranges().contains(_v);
}

void
save_line_info(const settings::compose_filename_config& _cfg, int _verbose)
{
    auto _write = [_verbose](const std::string& ofname, const auto& _data,
                             const std::array<bool, 3>& _info) {
        auto _ofs = std::ofstream{};
        if(tim::filepath::open(_ofs, ofname))
        {
            if(_verbose >= 0)
                operation::file_output_message<binary::symbol>{}(
                    ofname, std::string{ "causal_symbol_info" });
            save_line_info_impl(_ofs, _data, _info);
            save_maps_info_impl(_ofs);
        }
        else
        {
            throw ::rocprofsys::exception<std::runtime_error>("Error opening " + ofname);
        }
    };

    _write(tim::settings::compose_output_filename(
               join('-', config::get_causal_output_filename(), "binary"), "txt", _cfg),
           get_cached_binary_info().first, { true, true, true });
    _write(tim::settings::compose_output_filename(
               join('-', config::get_causal_output_filename(), "scoped"), "txt", _cfg),
           get_cached_binary_info().second, { true, true, false });
}

size_t
set_current_selection(unwind_addr_t _stack)
{
    for(auto itr : _stack)
    {
        if(itr == 0) continue;
        ++eligible_pc_candidates;
        if(is_eligible_address(itr))
        {
            auto _idx = eligible_pc_idx++ % latest_eligible_pc.size();
            latest_eligible_pc.at(_idx)->store(itr);
        }
    }

    return eligible_pc_idx.load(std::memory_order_relaxed);
}

size_t
set_current_selection(container::c_array<uint64_t> _stack)
{
    for(auto itr : _stack)
    {
        if(itr == 0) continue;
        ++eligible_pc_candidates;
        if(is_eligible_address(itr))
        {
            auto _idx = eligible_pc_idx++ % latest_eligible_pc.size();
            latest_eligible_pc.at(_idx)->store(itr);
        }
    }

    return eligible_pc_idx.load(std::memory_order_relaxed);
}

void
reset_sample_selection()
{
    eligible_pc_idx.store(0);
    eligible_pc_candidates.store(0);
    for(auto& itr : latest_eligible_pc)
    {
        if(itr) itr->store(0);
    }
}

selected_entry
sample_selection(size_t _nitr, size_t _wait_ns)
{
    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);

    auto _select_address = [&](auto& _address_vec) {
        // this isn't necessary bc of check before calling this lambda but
        // kept because of size() - 1 in distribution range
        if(ROCPROFSYS_UNLIKELY(_address_vec.empty()))
        {
            ROCPROFSYS_WARNING(0, "no addresses for sample selection...\n");
            return selected_entry{};
        }

        while(!_address_vec.empty())
        {
            // randomly select an address
            auto _dist =
                std::uniform_int_distribution<size_t>{ 0, _address_vec.size() - 1 };
            auto _idx = _dist(get_engine<selected_entry>());

            uintptr_t _addr        = _address_vec.at(_idx);
            uintptr_t _sym_addr    = 0;
            uintptr_t _lookup_addr = _addr;
            auto      _dl_info     = unwind::dlinfo::construct(_addr);

            _address_vec.erase(_address_vec.begin() + _idx);

            eligible_pc_history[_addr] += 1;

            if(get_causal_mode() == CausalMode::Function)
                _sym_addr = (_dl_info.symbol) ? _dl_info.symbol.address() : _addr;

            // lookup the PC line info at either the address or the symbol address
            auto linfo = get_line_info(_lookup_addr, false);

            // unlikely this will be empty but just in case
            if(linfo.empty()) continue;

            // debugging for continuous integration
            if(ROCPROFSYS_UNLIKELY(config::get_is_continuous_integration() ||
                                   config::get_debug()))
            {
                auto _location =
                    (_dl_info.location)
                        ? filepath::realpath(std::string{ _dl_info.location.name },
                                             nullptr, false)
                        : std::string{};
                for(const auto& itr : linfo)
                {
                    if(ROCPROFSYS_UNLIKELY(config::get_debug()))
                    {
                        ROCPROFSYS_WARNING(
                            0, "[%s][%s][%s][%s] %s [%s:%i][%s][%zu]\n",
                            as_hex(_lookup_addr).c_str(), as_hex(_addr).c_str(),
                            as_hex(_sym_addr).c_str(),
                            (_location.empty()) ? "" : _location.data(),
                            demangle(itr.func).c_str(), itr.file.c_str(), itr.line,
                            itr.address.as_string().c_str(), itr.address.size());
                    }
                }
            }

            auto& _linfo_v = (config::get_causal_mode() == CausalMode::Function)
                                 ? linfo.front()
                                 : linfo.back();
            return selected_entry{ _addr, _sym_addr, _linfo_v };
        }
        return selected_entry{};
    };

    while(eligible_pc_idx.load(std::memory_order_relaxed) == 0)
    {
        std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::nanoseconds{ _wait_ns });
    }

    for(size_t _n = 0; _n < _nitr; ++_n)
    {
        auto _addresses = std::deque<uintptr_t>{};
        for(auto& aitr : latest_eligible_pc)
        {
            if(ROCPROFSYS_UNLIKELY(!aitr))
            {
                ROCPROFSYS_WARNING(0, "invalid atomic pc...\n");
                continue;
            }

            uintptr_t _addr = aitr->load();
            if(_addr > 0) _addresses.emplace_back(_addr);
        }

        if(!_addresses.empty())
        {
            auto _selection = _select_address(_addresses);
            if(_selection) return _selection;
        }
    }

    return selected_entry{};
}

std::deque<binary::symbol>
get_line_info(uintptr_t _addr, bool _include_discarded)
{
    static auto _glob_filters  = get_filters({ sf::BINARY_FILTER });
    static auto _scope_filters = get_filters();
    auto        _data          = std::deque<binary::symbol>{};
    auto        _get_line_info = [&](const auto& _info, const auto& _filters) {
        // search for exact matches first
        for(const binary::binary_info& litr : _info)
        {
            auto _local_data = std::deque<binary::symbol>{};

            // make sure the address is in the coarse grained mapped regions
            // before performing an exhaustive search
            bool _is_mapped = std::find_if(litr.mappings.begin(), litr.mappings.end(),
                                                  [_addr](const auto& mitr) {
                                               return address_range_t{ mitr.load_address,
                                                                       mitr.last_address }
                                                   .contains(_addr);
                                           }) != litr.mappings.end();

            if(!_is_mapped) return;

            for(const auto& ditr : litr.symbols)
            {
                // skip if load address is greater than address
                if(_addr < ditr.load_address) continue;
                // compute the symbols ip address range
                auto _ipaddr = ditr.ipaddr();
                // if the lower bound of the ip address range is greater than the address,
                // all following symbols are not worth searching since they are at higher
                // addresses than this symbol (sorted by address)
                // if(_ipaddr.low > _addr) break;

                if(!_ipaddr.contains(_addr)) continue;

                if(_include_discarded ||
                   config::get_causal_mode() == CausalMode::Function)
                {
                    // check if the primary symbol satisfy the constraints
                    if(ditr(_filters)) _local_data.emplace_back(ditr);

                    // the primary symbol may not satisfy the constraints but the inlined
                    // functions may
                    utility::combine(_local_data, ditr.get_inline_symbols(_filters));
                }

                if(_include_discarded || config::get_causal_mode() == CausalMode::Line)
                {
                    auto _debug_data = std::deque<binary::symbol>{};
                    for(const auto& itr : ditr.get_debug_line_info(_filters))
                    {
                        if(!_ipaddr.contains(itr.ipaddr()))
                            ROCPROFSYS_THROW(
                                "Error! debug line info ipaddr (%s) is not contained in "
                                       "symbol ipaddr (%s)",
                                as_hex(itr.ipaddr()).c_str(), as_hex(_ipaddr).c_str());
                        if(itr.ipaddr().contains(_addr)) _debug_data.emplace_back(itr);
                    }
                    utility::combine(_local_data, _debug_data);
                }
            }

            if(!_local_data.empty())
            {
                // combine and only allow first match
                utility::combine(_data, _local_data);
                if(!_include_discarded) break;
            }
        }
    };

    if(_include_discarded)
        _get_line_info(get_cached_binary_info().first, _glob_filters);
    else
        _get_line_info(get_cached_binary_info().second, _scope_filters);

    return _data;
}

void
push_progress_point(std::string_view _name)
{
    if(config::get_causal_end_to_end()) return;

    ++num_progress_points;

    auto  _hash = tim::add_hash_id(_name);
    auto& _data = get_progress_bundles();
    if(ROCPROFSYS_LIKELY(_data != nullptr))
    {
        auto* _bundle = _data->construct(_hash);
        _bundle->push();
        _bundle->start();
    }
}

void
pop_progress_point(std::string_view _name)
{
    if(config::get_causal_end_to_end()) return;

    auto& _data = get_progress_bundles();
    if(ROCPROFSYS_UNLIKELY(!_data || _data->empty())) return;
    if(_name.empty())
    {
        auto* itr = _data->back();
        itr->stop();
        itr->pop();
        _data->pop_back();
        return;
    }
    else
    {
        auto _hash = tim::add_hash_id(_name);
        for(auto itr = _data->rbegin(); itr != _data->rend(); ++itr)
        {
            if((*itr)->get_hash() == _hash)
            {
                (*itr)->stop();
                (*itr)->pop();
                _data->destroy(itr);
                return;
            }
        }
    }
}

void
mark_progress_point(std::string_view _name, bool _force)
{
    if(config::get_causal_end_to_end() && !_force) return;

    ++num_progress_points;

    auto  _hash = tim::add_hash_id(_name);
    auto& _data = get_progress_bundles();
    if(ROCPROFSYS_LIKELY(_data != nullptr))
    {
        auto* _bundle = _data->construct(_hash);
        _bundle->push();
        _bundle->mark();
        _bundle->pop();
        _data->pop_back();
    }
}

uint16_t
sample_virtual_speedup()
{
    if(speedup_dist.empty())
        return 0;
    else if(speedup_dist.size() == 1)
        return speedup_dist.front();
    else
    {
        struct virtual_speedup
        {};
        auto _dist =
            std::uniform_int_distribution<size_t>{ size_t{ 0 }, speedup_dist.size() - 1 };
        return speedup_dist.at(_dist(get_engine<virtual_speedup>()));
    }
}

void
start_experimenting()
{
    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);

    auto _user_speedup_dist = config::get_causal_fixed_speedup();
    if(!_user_speedup_dist.empty())
    {
        speedup_dist.clear();
        for(auto itr : _user_speedup_dist)
        {
            ROCPROFSYS_CONDITIONAL_ABORT_F(itr > 100,
                                           "Virtual speedups must be in range [0, 100]. "
                                           "Invalid virtual speedup: %lu\n",
                                           itr);
            speedup_dist.emplace_back(static_cast<uint16_t>(itr));
        }
    }

    delay::setup();
    compute_eligible_lines();

    if(get_state() < State::Finalized)
    {
        ROCPROFSYS_SCOPED_SAMPLING_ON_CHILD_THREADS(false);
        auto _promise = std::make_shared<std::promise<void>>();
        std::thread{ perform_experiment_impl, _promise }.detach();
        _promise->get_future().wait_for(std::chrono::seconds{ 2 });
    }
}

void
finish_experimenting()
{
    if(perform_experiment_impl_completed)
    {
        perform_experiment_impl_completed->get_future().wait_for(
            std::chrono::seconds{ 5 });
        perform_experiment_impl_completed.reset();
    }
    sampling::post_process();
    experiment::save_experiments();
}
}  // namespace causal
}  // namespace rocprofsys
