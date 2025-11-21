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

template <typename TypeIdentifierEnum, typename... SupportedTypes>
class type_registry
{
    static_assert(type_traits::is_enum_class_v<TypeIdentifierEnum>,
                  "TypeIdentifierEnum must be an enum class");

public:
    using variant_t = typename std::variant<SupportedTypes...>;

    type_registry() { (register_type<SupportedTypes>(), ...); }

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
