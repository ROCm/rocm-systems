// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include "core/trace_cache/archive.hpp"
#include "core/trace_cache/cache_type_traits.hpp"
#include <cstdint>

#include <algorithm>
#include <array>
#include <functional>
#include <map>
#include <optional>
#include <utility>
#include <variant>

namespace rocprofsys::trace_cache
{

namespace detail
{
template <auto... Identifiers>
[[nodiscard]] constexpr bool
are_unique()
{
    std::array identifiers{ Identifiers... };
    std::ranges::sort(identifiers);
    return std::ranges::adjacent_find(identifiers) == identifiers.end();
}
}  // namespace detail

template <typename TypeIdentifierEnum, typename... SupportedTypes>
class type_registry
{
    static_assert(type_traits::is_enum_class_v<TypeIdentifierEnum>,
                  "TypeIdentifierEnum must be an enum class");
    static_assert(detail::are_unique<SupportedTypes::type_identifier...>(),
                  "two record types share a type_identifier");

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
