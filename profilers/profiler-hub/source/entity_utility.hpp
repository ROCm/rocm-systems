// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "traits.hpp"

#include "logger.hpp"

#include <cstddef>
#include <functional>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace profiler_hub
{

template <typename EntityContainerType, typename PrimaryKey = size_t>
class entity_utility
{
public:
    template <typename Entity>
    [[nodiscard]] bool is_entry_registered(const Entity& entity) const noexcept
    {
        const typename EntityContainerType::key_type key{ entity };
        const std::lock_guard<std::mutex>            lock(m_mutex);
        return m_entity_container.count(key) > 0;
    }

    template <typename... Args>
    void emplace_entity(Args&&... args)
    {
        std::lock_guard<std::mutex> const lock(m_mutex);
        m_entity_container.emplace(std::forward<Args>(args)...);
        m_last_key.reset();
    }

    template <typename Entity>
    [[nodiscard]] PrimaryKey get_primary_key_value_for_entity(const Entity& entity) const
    {
        const std::lock_guard<std::mutex> lock(m_mutex);
        if constexpr(common::traits::is_unordered_map_v<EntityContainerType>)
        {
            const typename EntityContainerType::key_type key{ entity };
            if(m_last_key && *m_last_key == key)
            {
                return m_last_value;
            }
            auto it = m_entity_container.find(key);
            if(it == m_entity_container.end())
            {
                throw std::runtime_error(fmt::format("Primary key not found for entity"));
            }
            m_last_key   = key;
            m_last_value = it->second;
            return it->second;
        }
        else
        {
            static_assert(common::traits::is_unordered_map_v<EntityContainerType>,
                          "EntityContainerType is not an unordered map");
        }
    }

private:
    EntityContainerType m_entity_container;
    mutable std::mutex  m_mutex;

    /// Single-slot last-resolved cache. Bulk inserts with an unchanging
    /// trace_environment_t (the dominant pattern) hit this 100% on the hot
    /// path and skip the unordered_map find/at.
    mutable std::optional<typename EntityContainerType::key_type> m_last_key;
    mutable PrimaryKey                                            m_last_value{};
};

/// Specialization for string-keyed maps enabling lookup from
/// std::string_view without per-lookup heap allocation.
///
/// Uses a pre-allocated std::string buffer (protected by the existing mutex)
/// to convert string_view to string for find(). After the buffer reaches
/// steady-state capacity, assign() is a memcpy with zero allocation.
template <typename ValueType, typename PrimaryKey>
class entity_utility<std::unordered_map<std::string, ValueType>, PrimaryKey>
{
    static constexpr size_t initial_lookup_buffer_capacity = 64;

    using container_type_t = std::unordered_map<std::string, ValueType>;

public:
    entity_utility() { m_lookup_buffer.reserve(initial_lookup_buffer_capacity); }

    [[nodiscard]] bool is_entry_registered(const std::string& key) const noexcept
    {
        const std::lock_guard<std::mutex> lock(m_mutex);
        return m_entity_container.count(key) > 0;
    }

    [[nodiscard]] bool is_entry_registered(std::string_view key) const noexcept
    {
        const std::lock_guard<std::mutex> lock(m_mutex);
        m_lookup_buffer.assign(key.data(), key.size());
        return m_entity_container.count(m_lookup_buffer) > 0;
    }

    template <typename... Args>
    void emplace_entity(Args&&... args)
    {
        std::lock_guard<std::mutex> const lock(m_mutex);
        m_entity_container.emplace(std::forward<Args>(args)...);
        m_last_key.reset();
    }

    [[nodiscard]] PrimaryKey get_primary_key_value_for_entity(
        const std::string& key) const
    {
        const std::lock_guard<std::mutex> lock(m_mutex);
        if(m_last_key && *m_last_key == key)
        {
            return m_last_value;
        }
        auto it = m_entity_container.find(key);
        if(it == m_entity_container.end())
        {
            throw std::runtime_error(
                fmt::format("Primary key not found for key: {}", key));
        }
        m_last_key   = key;
        m_last_value = it->second;
        return it->second;
    }

    [[nodiscard]] PrimaryKey get_primary_key_value_for_entity(std::string_view key) const
    {
        const std::lock_guard<std::mutex> lock(m_mutex);
        if(m_last_key && *m_last_key == key)
        {
            return m_last_value;
        }
        m_lookup_buffer.assign(key.data(), key.size());
        auto it = m_entity_container.find(m_lookup_buffer);

        if(it == m_entity_container.end())
        {
            throw std::runtime_error(
                fmt::format("Primary key not found for key: {}", key));
        }
        m_last_key   = m_lookup_buffer;
        m_last_value = it->second;
        return it->second;
    }

private:
    container_type_t                   m_entity_container;
    mutable std::string                m_lookup_buffer;
    mutable std::mutex                 m_mutex;
    mutable std::optional<std::string> m_last_key;
    mutable PrimaryKey                 m_last_value{};
};

}  // namespace profiler_hub
