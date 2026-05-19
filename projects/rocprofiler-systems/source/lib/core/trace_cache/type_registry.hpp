// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include "core/trace_cache/cache_type_traits.hpp"
#include <cstdint>

#include <functional>
#include <map>
#include <optional>

namespace rocprofsys
{
namespace trace_cache
{

template <typename TypeIdentifierEnum, typename... SupportedTypes>
class type_registry
{
    static_assert(type_traits::is_enum_class_v<TypeIdentifierEnum>,
                  "TypeIdentifierEnum must be an enum class");

public:
    using variant_t = typename std::variant<SupportedTypes...>;

    type_registry() { (register_type<SupportedTypes>(), ...); }

    std::optional<variant_t> get_type(TypeIdentifierEnum id, std::uint8_t*& data)
    {
        auto it = deserializers.find(id);
        if(it != deserializers.end())
        {
            return it->second(data);
        }
        return std::nullopt;
    }

private:
    std::map<TypeIdentifierEnum, std::function<variant_t(std::uint8_t*&)>> deserializers;

    template <typename T>
    inline void register_type()
    {
        static_assert(type_traits::has_type_identifier<T, TypeIdentifierEnum>::value,
                      "Type must have type_identifier");
        static_assert(type_traits::has_deserialize<T>::value,
                      "Type must have deserialize function");
        deserializers[T::type_identifier] = [](std::uint8_t*& data) -> variant_t {
            return deserialize<T>(data);
        };
    }
};

}  // namespace trace_cache
}  // namespace rocprofsys
