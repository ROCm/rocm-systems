// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once
#include "core/trace_cache/cache_type_traits.hpp"
#include <functional>
#include <map>
#include <optional>

namespace rocprofsys
{
namespace trace_cache
{

/**
 * @brief Registry for managing type deserialization based on type identifiers
 *
 * This template class provides a type-safe registry that maps type identifier enums
 * to their corresponding deserialization functions. It automatically registers all
 * supported types at construction time and provides runtime type lookup and
 * deserialization capabilities.
 *
 * The registry uses a variant to hold any of the supported types and maintains
 * a mapping from type identifiers to deserialization functions for efficient
 * runtime type resolution.
 *
 * @tparam TypeIdentifierEnum Enum class that identifies different types
 * @tparam SupportedTypes Variadic template pack of types that can be registered
 *
 * @note All SupportedTypes must:
 *       - Have a static member `type_identifier` of type TypeIdentifierEnum
 *       - Have a deserialize function available via ADL or in the global scope
 *
 * Example usage:
 * @code
 * enum class MyTypeId { TYPE_A, TYPE_B };
 * type_registry<MyTypeId, TypeA, TypeB> registry;
 *
 * uint8_t* data = get_serialized_data();
 * auto result = registry.get_type(MyTypeId::TYPE_A, data);
 * if (result) {
 *     // Handle deserialized type
 * }
 * @endcode
 */
template <typename TypeIdentifierEnum, typename... SupportedTypes>
class type_registry
{
    static_assert(type_traits::is_enum_class_v<TypeIdentifierEnum>,
                  "TypeIdentifierEnum must be an enum class");

public:
    using variant_t = typename std::variant<SupportedTypes...>;

    type_registry() { (register_type<SupportedTypes>(), ...); }

    /**
     * @brief Deserialize data based on type identifier
     *
     * Looks up the appropriate deserializer function based on the provided
     * type identifier and attempts to deserialize the data. The data pointer
     * is passed by reference and may be modified by the deserialization process
     * to point to the next data element.
     *
     * @param id Type identifier enum value that specifies which type to deserialize
     * @param data Reference to pointer to serialized data. May be advanced during
     * deserialization
     * @return std::optional<variant_t> Variant containing the deserialized object if
     * successful, or std::nullopt if the type identifier is not registered
     *
     * @note The data pointer may be advanced during deserialization to facilitate
     *       sequential reading of multiple serialized objects
     */
    std::optional<variant_t> get_type(TypeIdentifierEnum id, uint8_t*& data)
    {
        auto it = deserializers.find(id);
        if(it != deserializers.end())
        {
            return it->second(data);
        }
        return std::nullopt;
    }

private:
    std::map<TypeIdentifierEnum, std::function<variant_t(uint8_t*&)>> deserializers;

    /**
     * @brief Register a single type with its deserialization function
     *
     * Creates and stores a lambda function that can deserialize the specified type T.
     * The lambda captures the deserialize function for type T and returns the
     * deserialized object wrapped in the registry's variant type.
     *
     * @tparam T Type to register (must satisfy the requirements below)
     *
     * @note Type T must satisfy the following requirements:
     *       - Must have a static member `type_identifier` of type TypeIdentifierEnum
     *       - Must have a deserialize function available (checked via type traits)
     *
     * @see type_traits::has_type_identifier
     * @see type_traits::has_deserialize
     */
    template <typename T>
    inline void register_type()
    {
        static_assert(type_traits::has_type_identifier<T, TypeIdentifierEnum>::value,
                      "Type must have type_identifier");
        static_assert(type_traits::has_deserialize<T>::value,
                      "Type must have deserialize function");
        deserializers[T::type_identifier] = [](uint8_t*& data) -> variant_t {
            return deserialize<T>(data);
        };
    }
};

}  // namespace trace_cache
}  // namespace rocprofsys
