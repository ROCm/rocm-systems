// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string_view>
#include <type_traits>

namespace rocjitsu {

/// One entry in a closed enum's authoritative iterable and diagnostic vocabulary.
template <typename Enum> struct ConSanEnumVocabularyEntry {
  Enum value;
  std::string_view name;
};

template <typename Enum>
[[nodiscard]] constexpr ConSanEnumVocabularyEntry<Enum> consan_enum(Enum value,
                                                                    std::string_view name) {
  return {value, name};
}

/// A closed enum's values and stable spellings, kept in one declaration.
template <typename Enum, size_t Size> class ConSanEnumVocabulary {
public:
  constexpr ConSanEnumVocabulary(std::array<ConSanEnumVocabularyEntry<Enum>, Size> entries,
                                 std::string_view invalid_name)
      : invalid_name_(invalid_name) {
    for (size_t i = 0; i < Size; ++i) {
      values_[i] = entries[i].value;
      names_[i] = entries[i].name;
    }
  }

  [[nodiscard]] constexpr auto begin() const { return values_.begin(); }
  [[nodiscard]] constexpr auto end() const { return values_.end(); }
  [[nodiscard]] constexpr size_t size() const { return Size; }
  [[nodiscard]] constexpr Enum operator[](size_t index) const { return values_[index]; }

  [[nodiscard]] constexpr std::string_view name(Enum value) const {
    for (size_t i = 0; i < Size; ++i)
      if (values_[i] == value)
        return names_[i];
    return invalid_name_;
  }

  [[nodiscard]] constexpr std::optional<Enum> parse(std::string_view name) const {
    for (size_t i = 0; i < Size; ++i)
      if (names_[i] == name)
        return values_[i];
    return std::nullopt;
  }

private:
  std::array<Enum, Size> values_{};
  std::array<std::string_view, Size> names_{};
  std::string_view invalid_name_;
};

template <typename Enum, typename... Entries>
[[nodiscard]] constexpr auto
make_consan_enum_vocabulary(std::string_view invalid_name,
                            ConSanEnumVocabularyEntry<Enum> first, Entries... rest) {
  static_assert((std::is_same_v<ConSanEnumVocabularyEntry<Enum>, Entries> && ...));
  return ConSanEnumVocabulary(std::array{first, rest...}, invalid_name);
}

} // namespace rocjitsu
