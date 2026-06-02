// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ranges.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace rocprofsys::pmc::collectors::gpu_perf_counter
{

using counter_id_t = std::uint64_t;

struct dimension_position
{
    std::string name;
    size_t      position{ 0 };
};

struct counter_metadata
{
    counter_id_t                    counter_id{ 0 };
    std::string                     name;
    std::string                     description;
    std::string                     block;
    std::string                     expression;
    bool                            is_constant = false;
    bool                            is_derived  = false;
    std::vector<dimension_position> dimensions;
};

struct counter_value
{
    counter_id_t counter_id{ 0 };
    double       value{ 0.0 };
};

using metrics = std::vector<counter_value>;

struct counter_definition
{
    std::string name{};
    size_t      device_index{ 0 };

    [[nodiscard]] std::string to_string() const
    {
        return fmt::format("{}:device={}", name, device_index);
    }

    bool operator==(const counter_definition& other) const
    {
        return name == other.name && device_index == other.device_index;
    }
};

struct counter_definition_hash
{
    size_t operator()(const counter_definition& def) const noexcept
    {
        static constexpr size_t hash_mix_constant = 0x9e3779b97f4a7c15ULL;
        static constexpr size_t name_hash_offset  = 6;
        static constexpr size_t name_hash_shift   = 2;

        const size_t name_hash   = std::hash<std::string>{}(def.name);
        const size_t device_hash = std::hash<size_t>{}(def.device_index);

        return name_hash ^
               (device_hash + hash_mix_constant + (name_hash << name_hash_offset) +
                (name_hash >> name_hash_shift));
    }
};

struct gpu_perf_counter_settings
{
    std::vector<counter_definition> explicit_counters;
    std::vector<std::string>        broadcast_names;
};

struct enabled_metrics
{
    enabled_metrics() = default;

    explicit enabled_metrics(std::vector<counter_definition> counters)
    : m_counters{ counters.begin(), counters.end() }
    {}

    [[nodiscard]] bool is_counter_enabled(const counter_definition& def) const
    {
        return m_counters.count(def) > 0;
    }

private:
    std::unordered_set<counter_definition, counter_definition_hash> m_counters;
};

inline enabled_metrics
to_enabled_metrics(enabled_metrics enabled)
{
    return enabled;
}

inline enabled_metrics
to_enabled_metrics(gpu_perf_counter_settings settings)
{
    return enabled_metrics{ std::move(settings.explicit_counters) };
}

inline std::string
make_qualified_name(const counter_metadata& meta)
{
    if(meta.dimensions.empty()) return meta.name;
    return fmt::format("{}[{}]", meta.name, fmt::join(meta.dimensions, ","));
}

inline std::string
format_track_name(size_t gpu_id, const std::string& qualified_name)
{
    return fmt::format("GPU [{}] {} (S)", gpu_id, qualified_name);
}

}  // namespace rocprofsys::pmc::collectors::gpu_perf_counter

template <>
struct fmt::formatter<rocprofsys::pmc::collectors::gpu_perf_counter::dimension_position>
: fmt::formatter<std::string_view>
{
    template <typename FormatContext>
    auto format(
        const rocprofsys::pmc::collectors::gpu_perf_counter::dimension_position& dim,
        FormatContext& ctx) const
    {
        return fmt::format_to(ctx.out(), "{}={}", dim.name, dim.position);
    }
};
