/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <type_traits>
#include <variant>

namespace hipFile {

template <typename> inline constexpr bool is_variant_of_T_and_T_ptr = false;

template <typename T1, typename T2>
inline constexpr bool is_variant_of_T_and_T_ptr<std::variant<T1, T2>> =
    (!std::is_pointer_v<T1> && std::is_pointer_v<T2> &&
     std::is_same_v<std::remove_const_t<T1>, std::remove_pointer_t<T2>>) ||
    (std::is_pointer_v<T1> && !std::is_pointer_v<T2> &&
     std::is_same_v<std::remove_pointer_t<T1>, std::remove_const_t<T2>>);

/// @brief Retrieves a pointer to the current set value
/// @param v A variant
/// @details Takes a variant of the form <T, T*> or the reverse. T may be const. Will return a pointer to the
/// non-pointer entry if it is currently held, otherwise it will return the pointer entry.
template <typename T1, typename T2,
          typename = std::enable_if_t<is_variant_of_T_and_T_ptr<std::variant<T1, T2>>>>
auto
get_variant_ptr(std::variant<T1, T2> &v)
{
    if constexpr (!std::is_pointer_v<T1>) {
        T1 *ptr = std::get_if<T1>(&v);
        if (!ptr) {
            ptr = std::get<T2>(v);
        }
        return ptr;
    }
    else {
        T2 *ptr = std::get_if<T2>(&v);
        if (!ptr) {
            ptr = std::get<T1>(v);
        }
        return ptr;
    }
}

}
