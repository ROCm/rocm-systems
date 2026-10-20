// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file race_checking.h
/// @brief Independently selectable core address-race check families.

#ifndef ROCJITSU_RACE_CHECKING_H_
#define ROCJITSU_RACE_CHECKING_H_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace rocjitsu {

using RaceCheckMask = uint32_t;

/// @brief Address domains and optional memory-operation sources tracked by the checker.
enum class RaceCheckFamily : RaceCheckMask {
  Global = 1u << 0,
  LdsSameWave = 1u << 1,
  LdsCrossWave = 1u << 2,
  DirectToLds = 1u << 3,
  TensorDma = 1u << 4,
};

constexpr RaceCheckMask race_check_bit(RaceCheckFamily family) {
  return static_cast<RaceCheckMask>(family);
}

inline constexpr RaceCheckMask kRaceCheckAddressDomains =
    race_check_bit(RaceCheckFamily::Global) | race_check_bit(RaceCheckFamily::LdsSameWave) |
    race_check_bit(RaceCheckFamily::LdsCrossWave);
inline constexpr RaceCheckMask kRaceCheckLdsDomains =
    race_check_bit(RaceCheckFamily::LdsSameWave) |
    race_check_bit(RaceCheckFamily::LdsCrossWave);
inline constexpr RaceCheckMask kAllRaceChecks =
    kRaceCheckAddressDomains | race_check_bit(RaceCheckFamily::DirectToLds) |
    race_check_bit(RaceCheckFamily::TensorDma);

constexpr bool race_check_enabled(RaceCheckMask mask, RaceCheckFamily family) {
  return (mask & race_check_bit(family)) != 0;
}

/// @brief Parse `all` or a comma-separated list of race-check family names.
///
/// Source switches (`direct-to-lds` and `tensor-dma`) require at least one
/// address domain. Empty lists, duplicate names, unknown names, and source-only
/// selections are rejected.
inline std::optional<RaceCheckMask> parse_race_checking_checks(std::string_view value) {
  if (value == "all")
    return kAllRaceChecks;

  RaceCheckMask mask = 0;
  while (!value.empty()) {
    const size_t comma = value.find(',');
    std::string_view token = value.substr(0, comma);
    const size_t first = token.find_first_not_of(" \t");
    const size_t last = token.find_last_not_of(" \t");
    if (first == std::string_view::npos)
      return std::nullopt;
    token = token.substr(first, last - first + 1);

    RaceCheckFamily family;
    if (token == "global")
      family = RaceCheckFamily::Global;
    else if (token == "lds-same-wave")
      family = RaceCheckFamily::LdsSameWave;
    else if (token == "lds-cross-wave")
      family = RaceCheckFamily::LdsCrossWave;
    else if (token == "direct-to-lds")
      family = RaceCheckFamily::DirectToLds;
    else if (token == "tensor-dma")
      family = RaceCheckFamily::TensorDma;
    else
      return std::nullopt;

    const RaceCheckMask bit = race_check_bit(family);
    if (mask & bit)
      return std::nullopt;
    mask |= bit;
    if (comma == std::string_view::npos)
      break;
    if (comma + 1 == value.size())
      return std::nullopt;
    value.remove_prefix(comma + 1);
  }

  if ((mask & kRaceCheckAddressDomains) == 0)
    return std::nullopt;
  return mask;
}

/// @brief Return the canonical comma-separated spelling for a valid mask.
inline std::string format_race_checking_checks(RaceCheckMask mask) {
  if (mask == kAllRaceChecks)
    return "all";
  std::string result;
  const auto append = [&](std::string_view name) {
    if (!result.empty())
      result.push_back(',');
    result.append(name);
  };
  if (race_check_enabled(mask, RaceCheckFamily::Global))
    append("global");
  if (race_check_enabled(mask, RaceCheckFamily::LdsSameWave))
    append("lds-same-wave");
  if (race_check_enabled(mask, RaceCheckFamily::LdsCrossWave))
    append("lds-cross-wave");
  if (race_check_enabled(mask, RaceCheckFamily::DirectToLds))
    append("direct-to-lds");
  if (race_check_enabled(mask, RaceCheckFamily::TensorDma))
    append("tensor-dma");
  return result;
}

} // namespace rocjitsu

#endif // ROCJITSU_RACE_CHECKING_H_
