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

#pragma once

#include "categories.hpp"
#include "common.hpp"

#if defined(TIMEMORY_USE_PERFETTO)
#    include <timemory/components/perfetto/backends.hpp>
#else
#    include <perfetto.h>
PERFETTO_DEFINE_CATEGORIES(ROCPROFSYS_PERFETTO_CATEGORIES);
#endif

#include "debug.hpp"

#include <timemory/process/process.hpp>

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace rocprofsys
{
std::unique_ptr<::perfetto::TracingSession>& get_perfetto_session(
    pid_t = process::get_id());

template <typename Tp>
struct perfetto_counter_track
{
    using track_map_t = std::map<uint32_t, std::vector<::perfetto::CounterTrack>>;
    using name_map_t  = std::map<uint32_t, std::vector<std::unique_ptr<std::string>>>;
    using data_t      = std::pair<name_map_t, track_map_t>;

    static auto   init() { (void) get_data(); }
    static auto   exists(size_t _idx, int64_t _n = -1);
    static size_t size(size_t _idx);
    static auto emplace(size_t _idx, const std::string& _v, const char* _units = nullptr,
                        const char* _category = nullptr, int64_t _mult = 1,
                        bool _incr = false);

    static auto& at(size_t _idx, size_t _n) { return get_data().second.at(_idx).at(_n); }

private:
    static data_t& get_data()
    {
        static auto _v = data_t{};
        return _v;
    }
};

template <typename Tp>
auto
perfetto_counter_track<Tp>::exists(size_t _idx, int64_t _n)
{
    bool _v = get_data().second.count(_idx) != 0;
    if(_n < 0 || !_v) return _v;
    return static_cast<size_t>(_n) < get_data().second.at(_idx).size();
}

template <typename Tp>
size_t
perfetto_counter_track<Tp>::size(size_t _idx)
{
    bool _v = get_data().second.count(_idx) != 0;
    if(!_v) return 0;
    return get_data().second.at(_idx).size();
}

template <typename Tp>
auto
perfetto_counter_track<Tp>::emplace(size_t _idx, const std::string& _v,
                                    const char* _units, const char* _category,
                                    int64_t _mult, bool _incr)
{
    auto& _name_data  = get_data().first[_idx];
    auto& _track_data = get_data().second[_idx];
    std::vector<std::tuple<std::string, const char*, bool>> _missing = {};
    if(config::get_is_continuous_integration())
    {
        for(const auto& itr : _name_data)
        {
            _missing.emplace_back(std::make_tuple(*itr, itr->c_str(), false));
            // TODO: _missing.emplace_back(*itr, itr->c_str(), false);
        }
    }
    auto        _index     = _track_data.size();
    auto&       _name      = _name_data.emplace_back(std::make_unique<std::string>(_v));
    const char* _unit_name = (_units && strlen(_units) > 0) ? _units : nullptr;
    _track_data.emplace_back(
        ::perfetto::CounterTrack{ ::perfetto::DynamicString{ _name->c_str() } }
            .set_unit_name(_unit_name)
            .set_category(_category)
            .set_unit_multiplier(_mult)
            .set_is_incremental(_incr));
    if(config::get_is_continuous_integration())
    {
        for(auto& itr : _missing)
        {
            const char* citr = std::get<1>(itr);
            for(const auto& ditr : _name_data)
            {
                if(citr == ditr->c_str() && strcmp(citr, ditr->c_str()) == 0)
                {
                    std::get<2>(itr) = true;
                    break;
                }
            }
            if(!std::get<2>(itr))
            {
                std::set<void*> _prev = {};
                std::set<void*> _curr = {};
                for(const auto& eitr : _missing)
                    _prev.emplace(
                        static_cast<void*>(const_cast<char*>(std::get<1>(eitr))));
                for(const auto& eitr : _name_data)
                    _curr.emplace(static_cast<void*>(const_cast<char*>(eitr->c_str())));
                std::stringstream _pss{};
                for(auto&& eitr : _prev)
                    _pss << " " << std::hex << std::setw(12) << std::left << eitr;
                std::stringstream _css{};
                for(auto&& eitr : _curr)
                    _css << " " << std::hex << std::setw(12) << std::left << eitr;
                ROCPROFSYS_THROW("perfetto_counter_track emplace method for '%s' (%p) "
                                 "invalidated C-string '%s' (%p).\n%8s: %s\n%8s: %s\n",
                                 _v.c_str(), (void*) _name->c_str(),
                                 std::get<0>(itr).c_str(),
                                 (void*) std::get<0>(itr).c_str(), "previous",
                                 _pss.str().c_str(), "current", _css.str().c_str());
            }
        }
    }
    return _index;
}
}  // namespace rocprofsys
