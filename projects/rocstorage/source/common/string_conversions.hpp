// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include <rocstorage/writer_types.hpp>

#include "debug.hpp"

#include <optional>
#include <string>

namespace std
{
std::string
to_string(const rocstorage::writer_api::agent_unique_id_t& agent_unique_id)
{
    return fmt::format("[agent_type={}, type_index={}]",
                       agent_unique_id.agent_type,
                       agent_unique_id.type_index);
}
}  // namespace std

namespace rocstorage
{

template <typename T>
[[nodiscard]] std::string
to_error_string(const std::optional<T>& opt)
{
    return opt.has_value() ? std::to_string(opt.value()) : "[NULL]";
}
}  // namespace rocstorage
