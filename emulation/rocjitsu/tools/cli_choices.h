// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>
#include <span>
#include <string_view>
#include <utility>

namespace rocjitsu::tools {

template <typename T> using NamedChoice = std::pair<std::string_view, T>;

template <typename T>
std::optional<T> find_choice(std::span<const NamedChoice<T>> choices, std::string_view name) {
  for (const auto &[candidate_name, value] : choices)
    if (candidate_name == name)
      return value;
  return std::nullopt;
}

template <typename T>
std::optional<std::string_view> choice_name(std::span<const NamedChoice<T>> choices,
                                            const T &value) {
  for (const auto &[name, candidate] : choices)
    if (candidate == value)
      return name;
  return std::nullopt;
}

} // namespace rocjitsu::tools
