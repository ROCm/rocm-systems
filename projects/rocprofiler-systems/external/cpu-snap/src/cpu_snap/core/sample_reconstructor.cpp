#include "cpu_snap/core/sample_reconstructor.hpp"

#include <algorithm>
#include <cstdint>
#include <variant>

namespace cpu_snap::core
{

sample_reconstructor::sample_reconstructor()
{
    m_reader.on_event([this](const record_variant& rec) { handle_event(rec); });
}

sample_reconstructor::~sample_reconstructor() = default;

void
sample_reconstructor::feed(std::span<const std::byte> data) noexcept
{
    m_reader.feed(data);
}

const std::unordered_map<pid_t, std::vector<raw_sample>>&
sample_reconstructor::samples() const noexcept
{
    return m_samples;
}

const module_map&
sample_reconstructor::modules() const noexcept
{
    return m_modules;
}

const reader_counters&
sample_reconstructor::counters() const noexcept
{
    return m_reader.counters();
}

std::uint64_t
sample_reconstructor::unresolved_samples() const noexcept
{
    return m_unresolved_samples;
}

void
sample_reconstructor::handle_event(const record_variant& rec)
{
    switch(type_of(rec))
    {
    case record_type::sample:
    {
        const auto& sample = std::get<sample_record>(rec);
        if(sample.leaf_node_id == 0)
        {
            return;
        }
        auto ips = m_reader.resolve_stack(sample.leaf_node_id);
        if(ips.empty())
        {
            ++m_unresolved_samples;
            return;
        }

        raw_sample rs{};
        rs.tid          = static_cast<pid_t>(sample.tid);
        rs.timestamp_ns = sample.timestamp_ns;
        rs.ip           = ips.front();
        rs.frames       = std::move(ips);
        m_samples[rs.tid].push_back(std::move(rs));
        return;
    }
    case record_type::maps_init:
        m_modules.apply(std::get<maps_init_record>(rec));
        return;
    case record_type::maps_delta:
        m_modules.apply(std::get<maps_delta_record>(rec));
        return;
    default:
        return;
    }
}

}  // namespace cpu_snap::core
