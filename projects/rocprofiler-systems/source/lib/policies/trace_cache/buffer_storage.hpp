// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <cstddef>
#include <string>
#include <sys/types.h>
#include <type_traits>

namespace rocprofsys::policies::trace_cache
{

template <typename TypeIdentifierEnum>
concept buffered_type_identifier_enum =
    std::is_enum_v<TypeIdentifierEnum> &&
    !std::is_convertible_v<TypeIdentifierEnum,
                           std::underlying_type_t<TypeIdentifierEnum>> &&
    requires { TypeIdentifierEnum::fragmented_space; };

template <typename Value, typename TypeIdentifierEnum>
concept buffered_value = buffered_type_identifier_enum<TypeIdentifierEnum> && requires {
    { Value::type_identifier } -> std::convertible_to<TypeIdentifierEnum>;
};

template <typename Storage, typename TypeIdentifierEnum, typename Value>
concept buffer_storage_policy = buffered_value<Value, TypeIdentifierEnum> &&
                                requires(Storage& storage, const Storage& const_storage,
                                         const pid_t& current_pid, const Value& value) {
                                    { Storage(std::string{}) };
                                    { storage.start(current_pid) };
                                    { storage.shutdown(current_pid) };
                                    { storage.store(value) };
                                    {
                                        const_storage.is_running()
                                    } -> std::convertible_to<bool>;
                                };

}  // namespace rocprofsys::policies::trace_cache
