#include "cpu_snap/core/module_map.hpp"

#include <algorithm>
#include <iterator>

namespace cpu_snap::core
{

void
module_map::apply(const maps_init_record& rec)
{
    for(const auto& mod : rec.modules)
    {
        add_module(mod, rec.wall_ns);
    }
}

void
module_map::apply(const maps_delta_record& rec)
{
    if(rec.operation == maps_delta_record::op::add)
    {
        add_module(rec.module, rec.wall_ns);
        return;
    }
    remove_module(rec.module, rec.wall_ns);
}

void
module_map::add_module(const module_info& mod, std::uint64_t load_ts)
{
    interval iv;
    iv.mod.base_va     = mod.base_va;
    iv.mod.size        = mod.size;
    iv.mod.file_offset = mod.file_offset;
    iv.mod.build_id    = mod.build_id;
    iv.mod.path        = mod.path;
    iv.load_ts         = load_ts;
    m_max_size         = std::max(m_max_size, mod.size);

    const auto pos = std::upper_bound(m_intervals.begin(), m_intervals.end(), mod.base_va,
                                      [](std::uint64_t va, const interval& other) {
                                          return va < other.mod.base_va;
                                      });
    m_intervals.insert(pos, std::move(iv));
}

void
module_map::remove_module(const module_info& mod, std::uint64_t unload_ts)
{
    // Intervals sharing a base_va stay in insertion order, so the last
    // open match is the most recent load at that address.
    const auto last = std::upper_bound(m_intervals.begin(), m_intervals.end(), mod.base_va,
                                       [](std::uint64_t va, const interval& other) {
                                           return va < other.mod.base_va;
                                       });
    const std::reverse_iterator rbegin{ last };
    const auto it = std::find_if(rbegin, m_intervals.rend(), [&](const interval& iv) {
        return iv.mod.base_va == mod.base_va && !iv.unload_ts.has_value()
            && iv.mod.path == mod.path;
    });
    if(it == m_intervals.rend())
    {
        return;
    }
    it->unload_ts = unload_ts;
}

std::optional<module_map::module_ref>
module_map::resolve(std::uint64_t ip, std::uint64_t timestamp_ns) const
{
    const auto last = std::upper_bound(m_intervals.begin(), m_intervals.end(), ip,
                                       [](std::uint64_t va, const interval& iv) {
                                           return va < iv.mod.base_va;
                                       });
    for(auto it = last; it != m_intervals.begin();)
    {
        --it;
        const std::uint64_t offset = ip - it->mod.base_va;
        if(offset >= m_max_size)
        {
            break;
        }
        if(offset >= it->mod.size || timestamp_ns < it->load_ts)
        {
            continue;
        }
        if(it->unload_ts.has_value() && timestamp_ns >= *it->unload_ts)
        {
            continue;
        }
        return it->mod;
    }
    return std::nullopt;
}

}  // namespace cpu_snap::core
