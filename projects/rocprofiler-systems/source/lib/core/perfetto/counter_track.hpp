// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <perfetto.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace rocprofsys::core::perfetto
{
// Per-counter-category track table. Each <Tp> instantiation owns its own
// map of CounterTrack handles keyed by (numeric index, sub-index) so a
// caller writing TRACE_COUNTER for tag <Tp> can look up or lazily create
// the track without colliding with other tag types. Mutex-protected
// because emplace() mutates the underlying maps; at() is only safe while
// no concurrent emplace fires on the same _idx, and production call sites
// guard emplace under std::call_once before the first at().
template <typename Tp>
struct counter_track
{
    using category_type = Tp;
    using track_map_t   = std::map<std::uint32_t, std::vector<::perfetto::CounterTrack>>;
    using name_map_t = std::map<std::uint32_t, std::vector<std::unique_ptr<std::string>>>;
    using data_t     = std::pair<name_map_t, track_map_t>;

    static auto   init() { (void) get_data(); }
    static auto   exists(size_t _idx, std::int64_t _n = -1);
    static size_t size(size_t _idx);
    static auto emplace(size_t _idx, const std::string& _v, const char* _units = nullptr,
                        const char* _category = nullptr, std::int64_t _mult = 1,
                        bool _incr = false);

    static ::perfetto::CounterTrack at(size_t _idx, size_t _n)
    {
        std::lock_guard<std::mutex> _lk{ get_mutex() };
        return get_data().second.at(_idx).at(_n);
    }

private:
    static data_t& get_data()
    {
        static auto _v = data_t{};
        return _v;
    }
    static std::mutex& get_mutex()
    {
        static std::mutex _m{};
        return _m;
    }
};

template <typename Tp>
auto
counter_track<Tp>::exists(size_t _idx, std::int64_t _n)
{
    std::lock_guard<std::mutex> _lk{ get_mutex() };
    bool                        _v = get_data().second.count(_idx) != 0;
    if(_n < 0 || !_v) return _v;
    return static_cast<size_t>(_n) < get_data().second.at(_idx).size();
}

template <typename Tp>
size_t
counter_track<Tp>::size(size_t _idx)
{
    std::lock_guard<std::mutex> _lk{ get_mutex() };
    bool                        _v = get_data().second.count(_idx) != 0;
    if(!_v) return 0;
    return get_data().second.at(_idx).size();
}

template <typename Tp>
auto
counter_track<Tp>::emplace(size_t _idx, const std::string& _v, const char* _units,
                           const char* _category, std::int64_t _mult, bool _incr)
{
    std::lock_guard<std::mutex> _lk{ get_mutex() };
    auto&                       _name_data  = get_data().first[_idx];
    auto&                       _track_data = get_data().second[_idx];

    auto        _index     = _track_data.size();
    auto&       _name      = _name_data.emplace_back(std::make_unique<std::string>(_v));
    const char* _name_cstr = _name->c_str();
    const char* _unit_name = nullptr;
    if(_units && strlen(_units) > 0)
    {
        auto& _unit_str = _name_data.emplace_back(std::make_unique<std::string>(_units));
        _unit_name      = _unit_str->c_str();
    }
    _track_data.emplace_back(
        ::perfetto::CounterTrack{ ::perfetto::DynamicString{ _name_cstr } }
            .set_unit_name(_unit_name)
            .set_category(_category)
            .set_unit_multiplier(_mult)
            .set_is_incremental(_incr));
    return _index;
}
}  // namespace rocprofsys::core::perfetto
