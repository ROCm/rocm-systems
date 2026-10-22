// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include "core/trace_cache/archive.hpp"
#include "core/trace_cache/cache_type_traits.hpp"
#include <cstdint>

#include <functional>
#include <map>
#include <optional>
#include <utility>

namespace rocprofsys::trace_cache
{

template <typename TypeIdentifierEnum, typename... SupportedTypes>
class type_registry
{
    static_assert(type_traits::is_enum_class_v<TypeIdentifierEnum>,
                  "TypeIdentifierEnum must be an enum class");

public:
    using variant_t = std::variant<SupportedTypes...>;

    type_registry() { (register_type<SupportedTypes>(), ...); }

    std::optional<variant_t> get_type(TypeIdentifierEnum id, std::uint8_t*& data)
    {
        auto const found = m_deserializers.find(id);
        if(found != m_deserializers.end())
        {
            return found->second(data);
        }
        return std::nullopt;
    }

private:
    // Converts to T by reading it, so the record is built inside the returned optional.
    template <typename T>
    struct reader
    {
        std::uint8_t*& data;

        explicit operator T() const { return deserialize<T>(data); }
    };

    std::map<TypeIdentifierEnum, std::function<std::optional<variant_t>(std::uint8_t*&)>>
        m_deserializers;

    template <typename T>
        requires type_traits::cacheable<T, TypeIdentifierEnum>
    void register_type()
    {
        m_deserializers[T::type_identifier] = [](std::uint8_t*& data) {
            return std::optional<variant_t>{ std::in_place, std::in_place_type<T>,
                                             reader<T>{ data } };
        };
    }
};

}  // namespace rocprofsys::trace_cache
