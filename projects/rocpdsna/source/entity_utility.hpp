// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "traits.hpp"

#include <cstddef>
#include <mutex>

namespace rocpdsna
{

template <typename EntityContainerType, typename PrimaryKey = size_t>
class entity_utility
{
public:
    template <typename Entity>
    [[nodiscard]] bool is_entry_registered(const Entity& entity) const noexcept
    {
        const std::lock_guard<std::mutex> lock(m_mutex);
        return m_entity_container.count(entity) > 0;
    }

    template <typename... Args>
    void emplace_entity(Args&&... args)
    {
        std::lock_guard<std::mutex> const lock(m_mutex);
        m_entity_container.emplace(std::forward<Args>(args)...);
    }

    template <typename Entity>
    [[nodiscard]] PrimaryKey get_primary_key_value_for_entity(
        const Entity& entity) const noexcept
    {
        const std::lock_guard<std::mutex> lock(m_mutex);
        if constexpr(common::traits::is_unordered_map_v<EntityContainerType>)
        {
            return m_entity_container.at(entity);
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
};

}  // namespace rocpdsna
