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

#include "library/coverage.hpp"
#include "library/api.hpp"
#include "library/config.hpp"
#include "library/debug.hpp"
#include "library/thread_data.hpp"

#include <timemory/backends/threading.hpp>
#include <timemory/tpls/cereal/cereal.hpp>
#include <timemory/utility/popen.hpp>

#include <algorithm>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>

#define OMNITRACE_SERIALIZE(MEMBER_VARIABLE)                                             \
    ar(::tim::cereal::make_nvp(#MEMBER_VARIABLE, MEMBER_VARIABLE))

namespace omnitrace
{
namespace coverage
{
namespace
{
template <typename... Tp>
using uomap_t = std::unordered_map<Tp...>;
//
using coverage_thread_data_type =
    uomap_t<std::string_view, uomap_t<std::string_view, std::map<size_t, size_t>>>;
//
template <typename Tp, typename... Args>
inline std::set<Tp, Args...>
get_uncovered(const std::set<Tp, Args...>& _covered,
              const std::set<Tp, Args...>& _possible)
{
    std::set<Tp, Args...> _v{};
    for(auto&& itr : _possible)
    {
        if(_covered.count(itr) == 0) _v.emplace(itr);
    }
    return _v;
}
//
template <typename Tp, typename... Args>
inline std::vector<Tp, Args...>
get_uncovered(const std::vector<Tp, Args...>& _covered,
              const std::vector<Tp, Args...>& _possible)
{
    std::vector<Tp, Args...> _v{};
    for(auto&& itr : _possible)
    {
        if(!std::any_of(_covered.begin(), _covered.end(),
                        [itr](auto&& _entry) { return _entry == itr; }))
            _v.emplace_back(itr);
    }
    return _v;
}
//
using coverage_thread_data =
    omnitrace::thread_data<coverage_thread_data_type, code_coverage>;
//
auto&
get_code_coverage()
{
    static auto _v = code_coverage{};
    return _v;
}
//
auto&
get_coverage_data()
{
    static auto _v = std::vector<coverage_data>{};
    return _v;
}
//
auto&
get_coverage_count(int64_t _tid = tim::threading::get_id())
{
    static auto& _v =
        coverage_thread_data::instances(coverage_thread_data::construct_on_init{});
    return _v.at(_tid);
}
}  // namespace

//--------------------------------------------------------------------------------------//

double
code_coverage::operator()(Category _c) const
{
    switch(_c)
    {
        case STANDARD: return static_cast<double>(count) / static_cast<double>(size);
        case ADDRESS:
            return static_cast<double>(covered.addresses.size()) /
                   static_cast<double>(possible.addresses.size());
        case MODULE:
            return static_cast<double>(covered.modules.size()) /
                   static_cast<double>(possible.modules.size());
        case FUNCTION:
            return static_cast<double>(covered.functions.size()) /
                   static_cast<double>(possible.functions.size());
    }
    return 0.0;
}

code_coverage::int_set_t
code_coverage::get_uncovered_addresses() const
{
    return get_uncovered(covered.addresses, possible.addresses);
}

code_coverage::str_set_t
code_coverage::get_uncovered_modules() const
{
    return get_uncovered(covered.modules, possible.modules);
}

code_coverage::str_set_t
code_coverage::get_uncovered_functions() const
{
    return get_uncovered(covered.functions, possible.functions);
}

//--------------------------------------------------------------------------------------//

coverage_data&
coverage_data::operator+=(const coverage_data& rhs)
{
    count += rhs.count;
    return *this;
}

bool
coverage_data::operator==(const coverage_data& rhs) const
{
    return std::tie(module, function, address) ==
           std::tie(rhs.module, rhs.function, rhs.address);
}

bool
coverage_data::operator==(const data_tuple_t& rhs) const
{
    return std::tie(module, function, address) ==
           std::tie(std::get<0>(rhs), std::get<1>(rhs), std::get<2>(rhs));
}

bool
coverage_data::operator!=(const coverage_data& rhs) const
{
    return !(*this == rhs);
}

bool
coverage_data::operator<(const coverage_data& rhs) const
{
    if(count != rhs.count) return count < rhs.count;
    if(module != rhs.module) return module < rhs.module;
    if(function != rhs.function) return function < rhs.function;
    if(address != rhs.address) return address < rhs.address;
    if(line != rhs.line) return line < rhs.line;
    return source < rhs.source;
}

bool
coverage_data::operator<=(const coverage_data& rhs) const
{
    return (*this == rhs || *this < rhs);
}

bool
coverage_data::operator>(const coverage_data& rhs) const
{
    return (*this != rhs && !(*this < rhs));
}

bool
coverage_data::operator>=(const coverage_data& rhs) const
{
    return !(*this < rhs);
}

//--------------------------------------------------------------------------------------//

void
post_process()
{
    if(!config::get_use_code_coverage()) return;

    code_coverage& _coverage = get_code_coverage();
    if(_coverage.size == 0)
    {
        OMNITRACE_VERBOSE_F(
            0,
            "Warning! Code coverage enabled but no code coverage data is available!\n");
        return;
    }

    using data_tuple_t   = coverage_data::data_tuple_t;
    auto& _coverage_data = get_coverage_data();

    auto _find = [&_coverage_data](data_tuple_t&& _v) {
        for(auto itr = _coverage_data.begin(); itr != _coverage_data.end(); ++itr)
        {
            if(*itr == _v) return std::make_pair(itr, true);
        }
        return std::make_pair(_coverage_data.end(), false);
    };

    auto _data = coverage_thread_data_type{};
    for(size_t i = 0; i < coverage_thread_data::size(); ++i)
    {
        const auto& _thr_data = *get_coverage_count(i);
        for(const auto& file : _thr_data)
        {
            for(const auto& func : file.second)
            {
                for(const auto& addr : func.second)
                {
                    _data[file.first][func.first][addr.first] += addr.second;
                    auto&& _v = _find({ file.first, func.first, addr.first });
                    if(_v.second)
                    {
                        _v.first->count += addr.second;
                    }
                    else
                    {
                        OMNITRACE_VERBOSE_F(
                            0, "Warning! No matching coverage data for %s :: %s (0x%x)\n",
                            func.first.data(), file.first.data(),
                            (unsigned int) addr.first);
                    }
                }
            }
        }
    }

    for(const auto& file : _data)
    {
        for(const auto& func : file.second)
        {
            for(const auto& addr : func.second)
            {
                if(addr.second > 0)
                {
                    _coverage.count += 1;
                    _coverage.covered.modules.emplace(file.first);
                    _coverage.covered.functions.emplace(func.first);
                    _coverage.covered.addresses.emplace(addr.first);
                }
            }
        }
    }

    std::sort(_coverage_data.begin(), _coverage_data.end(),
              std::greater<coverage_data>{});

    {
        auto _tmp         = std::decay_t<decltype(_coverage_data)>{};
        auto _find_in_tmp = [&_tmp](const auto& _v) {
            for(auto itr = _tmp.begin(); itr != _tmp.end(); ++itr)
            {
                if(itr->source == _v.source && itr->address != _v.address &&
                   itr->count == _v.count)
                    return std::make_pair(itr, true);
            }
            return std::make_pair(_tmp.end(), false);
        };
        for(auto&& itr : _coverage_data)
        {
            if(!_find_in_tmp(itr).second) _tmp.emplace_back(itr);
        }
        std::swap(_coverage_data, _tmp);
    }

    OMNITRACE_VERBOSE(0, "code coverage     :: %6.2f%s\n", _coverage() * 100.0, "%");
    OMNITRACE_VERBOSE(0, "module coverage   :: %6.2f%s\n",
                      _coverage(code_coverage::MODULE) * 100.0, "%");
    OMNITRACE_VERBOSE(0, "function coverage :: %6.2f%s\n",
                      _coverage(code_coverage::FUNCTION) * 100.0, "%");

    if(get_verbose() >= 0) fprintf(stderr, "\n");

    std::sort(_coverage_data.begin(), _coverage_data.end(),
              std::greater<coverage_data>{});

    auto _get_setting = [](const std::string& _v) {
        auto&& _b = config::get_setting_value<bool>(_v);
        OMNITRACE_CI_THROW(!_b.first, "Error! No configuration setting named '%s'",
                           _v.c_str());
        return (_b.first) ? _b.second : true;
    };

    auto _text_output = _get_setting("OMNITRACE_TEXT_OUTPUT");
    auto _json_output = _get_setting("OMNITRACE_JSON_OUTPUT");

    if(_text_output)
    {
        auto          _fname = tim::settings::compose_output_filename("coverage", ".txt");
        std::ofstream ofs{};
        if(tim::filepath::open(ofs, _fname))
        {
            if(get_verbose() >= 0)
                fprintf(stderr, "[%s][coverage]|%i> Outputting '%s'...\n",
                        TIMEMORY_PROJECT_NAME, dmp::rank(), _fname.c_str());
            for(auto& itr : _coverage_data)
            {
                // if(get_debug() && get_verbose() >= 2)
                if(true)
                {
                    auto _addr = TIMEMORY_JOIN("", "0x", std::hex, itr.address);
                    ofs << std::setw(8) << itr.count << "  " << std::setw(8) << _addr
                        << "  " << itr.source << "\n";
                }
                else
                {
                    ofs << std::setw(8) << itr.count << "  " << itr.source << "\n";
                }
            }
        }
        else
        {
            OMNITRACE_THROW("Error opening coverage output file: %s", _fname.c_str());
        }
    }

    if(_json_output)
    {
        auto _fname = tim::settings::compose_output_filename("coverage", ".json");
        std::ofstream ofs{};
        if(tim::filepath::open(ofs, _fname))
        {
            if(get_verbose() >= 0)
                fprintf(stderr, "[%s][coverage]|%i> Outputting '%s'...\n",
                        TIMEMORY_PROJECT_NAME, dmp::rank(), _fname.c_str());
            {
                namespace cereal = tim::cereal;
                auto ar =
                    tim::policy::output_archive<cereal::PrettyJSONOutputArchive>::get(
                        ofs);

                ar->setNextName("omnitrace");
                ar->startNode();
                ar->setNextName("coverage");
                ar->startNode();
                (*ar)(cereal::make_nvp("summary", _coverage));
                (*ar)(cereal::make_nvp("details", _coverage_data));
                ar->finishNode();
                ar->finishNode();
            }
        }
        else
        {
            OMNITRACE_THROW("Error opening coverage output file: %s", _fname.c_str());
        }
    }

    if(get_verbose() >= 0) fprintf(stderr, "\n");
}
}  // namespace coverage
}  // namespace omnitrace

//--------------------------------------------------------------------------------------//

namespace coverage = omnitrace::coverage;

extern "C" void
omnitrace_register_source_hidden(const char* file, const char* func, size_t line,
                                 size_t address, const char* source)
{
    using coverage_data = coverage::coverage_data;

    OMNITRACE_BASIC_VERBOSE_F(2, "[0x%x] :: %-20s :: %20s:%zu :: %s\n",
                              (unsigned int) address, func, file, line, source);

    coverage::get_coverage_data().emplace_back(
        coverage_data{ size_t{ 0 }, address, line, file, func,
                       (source && strlen(source) > 0) ? source : func });

    coverage::get_code_coverage().size += 1;
    coverage::get_code_coverage().possible.modules.emplace(file);
    coverage::get_code_coverage().possible.functions.emplace(func);
    coverage::get_code_coverage().possible.addresses.emplace(address);

    // initialize
    for(size_t i = 0; i < coverage::coverage_thread_data::size(); ++i)
        (*coverage::get_coverage_count(i))[file][func][address] = 0;
}

//--------------------------------------------------------------------------------------//

extern "C" void
omnitrace_register_coverage_hidden(const char* file, const char* func, size_t address)
{
    OMNITRACE_BASIC_VERBOSE_F(3, "[0x%x] %-20s :: %20s\n", (unsigned int) address, func,
                              file);
    coverage::get_coverage_count()->at(file).at(func).at(address) += 1;
}

//--------------------------------------------------------------------------------------//
