// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include "core/trace_cache/archive.hpp"
#include "core/trace_cache/cache_type_traits.hpp"
#include <cstdint>

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

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
    /**
     * Deserializes the record whose type_identifier matches `type_id` and passes it to
     * `visitor`.
     * @throws std::runtime_error when no supported type has that id.
     */
    template <typename Visitor>
        requires type_traits::sample_visitor<Visitor, SupportedTypes...>
    void dispatch(TypeIdentifierEnum type_id, std::uint8_t*& data,
                  Visitor&& visitor) const
    {
        const auto try_type = [&]<typename T>(std::type_identity<T>) {
            if(type_id != T::type_identifier)
            {
                return false;
            }
            visitor(deserialize<T>(data));
            return true;
        };
        if(!(try_type(std::type_identity<SupportedTypes>{}) || ...))
        {
            throw std::runtime_error(
                "No record type registered for type id " +
                std::to_string(
                    static_cast<std::underlying_type_t<TypeIdentifierEnum>>(type_id)));
        }
    }
};

}  // namespace rocprofsys::trace_cache
