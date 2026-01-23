// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "traits.hpp"

namespace rocstorage
{

template <typename EntityContainerType, typename PrimaryKey = size_t>
class entity_utility
{
public:
    template <typename Entity>
    bool is_entry_registered(const Entity& entity) const noexcept
    {
        return m_entity_container.count(entity) > 0;
    }

    template <typename... Entity>
    void emplace_entity(Entity&&... entity)
    {
        m_entity_container.emplace(std::forward<Entity>(entity)...);
    }

    template <typename Entity>
    PrimaryKey get_primary_key_value_for_entity(const Entity& entity) const noexcept
    {
        if constexpr(common::traits::is_unordered_map_v<EntityContainerType>)
        {
            return m_entity_container.at(entity);
        }
        else
        {
            static_assert(false, "EntityContainerType is not an unordered map");
        }
    }

private:
    EntityContainerType m_entity_container;
};

}  // namespace rocstorage
