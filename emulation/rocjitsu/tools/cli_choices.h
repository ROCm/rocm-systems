// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>
#include <span>
#include <string_view>

namespace rocjitsu::tools {

template <typename T> struct NamedChoice {
  std::string_view name;
  T value;
};

template <typename T>
std::optional<T> find_choice(std::span<const NamedChoice<T>> choices, std::string_view name) {
  for (const NamedChoice<T> &choice : choices)
    if (choice.name == name)
      return choice.value;
  return std::nullopt;
}

template <typename T>
std::optional<std::string_view> choice_name(std::span<const NamedChoice<T>> choices,
                                            const T &value) {
  for (const NamedChoice<T> &choice : choices)
    if (choice.value == value)
      return choice.name;
  return std::nullopt;
}

} // namespace rocjitsu::tools
