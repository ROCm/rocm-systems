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

#include "library/coverage.hpp"
#include "api.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "library/coverage/impl.hpp"
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

#define ROCPROFSYS_SERIALIZE(MEMBER_VARIABLE)                                            \
    ar(::tim::cereal::make_nvp(#MEMBER_VARIABLE, MEMBER_VARIABLE))

namespace rocprofsys
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
using coverage_data_vector = std::vector<coverage_data>;
//
using coverage_data_map =
    uomap_t<std::string_view,
            uomap_t<std::string_view, std::map<size_t, coverage_data_vector::iterator>>>;
//
using coverage_thread_data =
    rocprofsys::thread_data<coverage_thread_data_type, code_coverage>;
//
auto&
get_code_coverage()
{
    static auto _v = code_coverage{};
    return _v;
}
//
auto&
get_post_processed()
{
    static auto* _v = new bool{ false };
    return *_v;
}
//
auto&
get_coverage_data()
{
    static auto _v = coverage_data_vector{};
    return _v;
}
//
auto&
get_coverage_count(int64_t _tid = tim::threading::get_id())
{
    return coverage_thread_data::instance(construct_on_thread{ _tid });
}
}  // namespace

//--------------------------------------------------------------------------------------//

void
post_process()
{
    using data_tuple_t = coverage_data::data_tuple_t;

    if(get_post_processed()) return;
    get_post_processed() = true;

    if(!config::get_use_code_coverage()) return;

    auto& _coverage      = get_code_coverage();
    auto& _coverage_data = get_coverage_data();

    if(_coverage.size == 0)
    {
        ROCPROFSYS_VERBOSE_F(
            0,
            "Warning! Code coverage enabled but no code coverage data is available!\n");
        return;
    }

    auto _data = coverage_thread_data_type{};
    {
        auto _coverage_map = coverage_data_map{};
        auto _find         = [&_coverage_data, &_coverage_map](data_tuple_t&& _v) {
            auto& _cache = _coverage_map[std::get<0>(_v)][std::get<1>(_v)];
            auto  mitr   = _cache.find(std::get<2>(_v));
            if(mitr != _cache.end()) return std::make_pair(mitr->second, true);

            for(auto itr = _coverage_data.begin(); itr != _coverage_data.end(); ++itr)
            {
                if(*itr == _v)
                {
                    _cache[std::get<2>(_v)] = itr;
                    return std::make_pair(itr, true);
                }
            }
            return std::make_pair(_coverage_data.end(), false);
        };

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
                            ROCPROFSYS_VERBOSE_F(0,
                                                 "Warning! No matching coverage data for "
                                                 "%s :: %s (0x%x)\n",
                                                 func.first.data(), file.first.data(),
                                                 (unsigned int) addr.first);
                        }
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
        auto _tmp_map     = coverage_data_map{};
        auto _tmp         = std::decay_t<decltype(_coverage_data)>{};
        auto _find_in_tmp = [&_tmp, &_tmp_map](const auto& _v) {
            auto& _cache = _tmp_map[_v.module][_v.function];
            auto  mitr   = _cache.find(_v.address);
            if(mitr != _cache.end()) return std::make_pair(mitr->second, true);

            for(auto titr = _tmp.begin(); titr != _tmp.end(); ++titr)
            {
                if(titr->source == _v.source && titr->address != _v.address &&
                   titr->count == _v.count)
                {
                    _cache[_v.address] = titr;
                    return std::make_pair(titr, true);
                }
            }
            return std::make_pair(_tmp.end(), false);
        };
        for(auto&& itr : _coverage_data)
        {
            if(!_find_in_tmp(itr).second) _tmp.emplace_back(itr);
        }
        std::swap(_coverage_data, _tmp);
    }

    ROCPROFSYS_VERBOSE(0, "code coverage     :: %6.2f%s\n", _coverage() * 100.0, "%");
    ROCPROFSYS_VERBOSE(0, "module coverage   :: %6.2f%s\n",
                       _coverage(code_coverage::MODULE) * 100.0, "%");
    ROCPROFSYS_VERBOSE(0, "function coverage :: %6.2f%s\n",
                       _coverage(code_coverage::FUNCTION) * 100.0, "%");

    if(get_verbose() >= 0) fprintf(stderr, "\n");

    std::sort(_coverage_data.begin(), _coverage_data.end(),
              std::greater<coverage_data>{});

    auto _get_setting = [](const std::string& _v) {
        auto&& _b = config::get_setting_value<bool>(_v);
        ROCPROFSYS_CI_THROW(!_b, "Error! No configuration setting named '%s'",
                            _v.c_str());
        return _b.value_or(true);
    };

    auto _text_output = _get_setting("ROCPROFSYS_TEXT_OUTPUT");
    auto _json_output = _get_setting("ROCPROFSYS_JSON_OUTPUT");

    if(_text_output)
    {
        auto          _fname = tim::settings::compose_output_filename("coverage", ".txt");
        std::ofstream ofs{};
        if(tim::filepath::open(ofs, _fname))
        {
            if(get_verbose() >= 0)
                operation::file_output_message<code_coverage>{}(
                    _fname, std::string{ "coverage" });
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
            ROCPROFSYS_THROW("Error opening coverage output file: %s", _fname.c_str());
        }
    }

    if(_json_output)
    {
        std::stringstream oss{};
        {
            namespace cereal = tim::cereal;
            auto ar =
                tim::policy::output_archive<cereal::PrettyJSONOutputArchive>::get(oss);

            ar->setNextName("rocprofsys");
            ar->startNode();
            ar->setNextName("coverage");
            ar->startNode();
            (*ar)(cereal::make_nvp("summary", _coverage));
            (*ar)(cereal::make_nvp("details", _coverage_data));
            ar->finishNode();
            ar->finishNode();
        }
        auto _fname = tim::settings::compose_output_filename("coverage", ".json");
        std::ofstream ofs{};
        if(tim::filepath::open(ofs, _fname))
        {
            if(get_verbose() >= 0)
                operation::file_output_message<code_coverage>{}(
                    _fname, std::string{ "coverage" });
            ofs << oss.str() << "\n";
        }
        else
        {
            ROCPROFSYS_THROW("Error opening coverage output file: %s", _fname.c_str());
        }
    }

    if(get_verbose() >= 0) fprintf(stderr, "\n");
}
}  // namespace coverage
}  // namespace rocprofsys

//--------------------------------------------------------------------------------------//

namespace coverage = rocprofsys::coverage;

extern "C" void
rocprofsys_register_source_hidden(const char* file, const char* func, size_t line,
                                  size_t address, const char* source)
{
    if(coverage::get_post_processed()) return;

    using coverage_data = coverage::coverage_data;

    ROCPROFSYS_BASIC_VERBOSE_F(4, "[0x%x] :: %-20s :: %20s:%zu :: %s\n",
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
    {
        (*coverage::get_coverage_count(i))[file][func].emplace(address, 0);
    }
}

//--------------------------------------------------------------------------------------//

extern "C" void
rocprofsys_register_coverage_hidden(const char* file, const char* func, size_t address)
{
    if(coverage::get_post_processed()) return;
    if(rocprofsys::get_state() < rocprofsys::State::Active &&
       !rocprofsys_init_tooling_hidden())
        return;
    else if(rocprofsys::get_state() >= rocprofsys::State::Finalized)
        return;

    ROCPROFSYS_BASIC_VERBOSE_F(3, "[0x%x] %-20s :: %20s\n", (unsigned int) address, func,
                               file);
    (*coverage::get_coverage_count())[file][func][address] += 1;
}

//--------------------------------------------------------------------------------------//
