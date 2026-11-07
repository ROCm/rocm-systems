// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "smi_nic_hwmon.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <map>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>

namespace amd::smi::nic::hwmon {

namespace {

namespace fs = std::filesystem;

constexpr auto kHwmonDirPrefix = std::string_view{"hwmon"};
constexpr auto kLabelSuffix = std::string_view{"label"};
constexpr auto kDigits = std::string_view{"0123456789"};

struct KindSpec_t {
  std::string_view prefix;
  Category_t category;
  Unit_t unit;
};

constexpr auto kKinds = std::array<KindSpec_t, 3>{{
    {"temp", Category_t::Temperature, Unit_t::MilliDegreeCelsius},
    {"in", Category_t::Voltage, Unit_t::MilliVolt},
    {"power", Category_t::Power, Unit_t::MicroWatt},
}};

struct AttributeSpec_t {
  std::string_view suffix;
  Attribute_t attribute;
};

constexpr auto kAttributes = std::array<AttributeSpec_t, 15>{{
    {"input", Attribute_t::Input},
    {"min", Attribute_t::Min},
    {"max", Attribute_t::Max},
    {"lcrit", Attribute_t::LCrit},
    {"crit", Attribute_t::Crit},
    {"emergency", Attribute_t::Emergency},
    {"shutdown", Attribute_t::Shutdown},
    {"cap", Attribute_t::Cap},
    {"min_alarm", Attribute_t::AlarmMin},
    {"max_alarm", Attribute_t::AlarmMax},
    {"lcrit_alarm", Attribute_t::AlarmLCrit},
    {"crit_alarm", Attribute_t::AlarmCrit},
    {"emergency_alarm", Attribute_t::AlarmEmergency},
    {"shutdown_alarm", Attribute_t::AlarmShutdown},
    {"cap_alarm", Attribute_t::AlarmCap},
}};

// An alarm attribute is a 0 or 1 flag whatever the sensor kind, so it does not take the kind's
// unit.
auto is_alarm(Attribute_t attribute) -> bool {
  return (static_cast<uint32_t>(attribute) >= static_cast<uint32_t>(Attribute_t::AlarmMin));
}

struct ParsedName_t {
  const KindSpec_t* kind;
  uint32_t channel;
  std::string_view suffix;
};

// "temp3_crit" -> {temp, 3, "crit"}; nullopt for anything that is not <kind><N>_<suffix>.
auto parse_name(std::string_view name) -> std::optional<ParsedName_t> {
  for (const auto& kind : kKinds) {
    if (name.substr(0, kind.prefix.size()) != kind.prefix) {
      continue;
    }
    const auto rest = name.substr(kind.prefix.size());
    const auto digits_end = rest.find_first_not_of(kDigits);
    if ((digits_end == 0) || (digits_end == std::string_view::npos) || (rest[digits_end] != '_')) {
      continue;
    }
    auto channel = uint32_t{0};
    const auto result = std::from_chars(rest.data(), (rest.data() + digits_end), channel);
    if (result.ec != std::errc{}) {
      continue;
    }
    return ParsedName_t{&kind, channel, rest.substr(digits_end + 1)};
  }
  return std::nullopt;
}

auto lookup_attribute(std::string_view suffix) -> std::optional<Attribute_t> {
  for (const auto& entry : kAttributes) {
    if (entry.suffix == suffix) {
      return entry.attribute;
    }
  }
  return std::nullopt;
}

auto read_first_line(const fs::path& path) -> std::optional<std::string> {
  auto file = std::ifstream{path};
  auto line = std::string{};
  if (!std::getline(file, line)) {
    return std::nullopt;
  }
  return line;
}

// SmiSysfsReader holds an int, which cannot carry a power reading above 2147 W in microwatts.
auto read_int64(const fs::path& path) -> std::optional<int64_t> {
  const auto line = read_first_line(path);
  if (!line.has_value()) {
    return std::nullopt;
  }
  const auto* const begin = line->data();
  const auto* const end = (begin + line->size());
  auto value = int64_t{0};
  const auto result = std::from_chars(begin, end, value);
  if ((result.ec != std::errc{}) || (result.ptr != end)) {
    return std::nullopt;
  }
  return value;
}

struct Instance_t {
  uint32_t index;
  fs::path path;
};

// hwmon<N> directories under `hwmon_root`, ascending by N.
auto list_instances(const std::string& hwmon_root) -> std::vector<Instance_t> {
  auto instances = std::vector<Instance_t>{};
  auto ec = std::error_code{};
  if (!fs::is_directory(hwmon_root, ec)) {
    return instances;
  }

  // Manual increment(ec): operator++ throws on a mid-scan read error.
  for (auto it = fs::directory_iterator(hwmon_root, ec); (!ec && (it != fs::directory_iterator{}));
       it.increment(ec)) {
    const auto name = it->path().filename().string();
    const auto view = std::string_view{name};
    if (view.substr(0, kHwmonDirPrefix.size()) != kHwmonDirPrefix) {
      continue;
    }
    const auto digits = view.substr(kHwmonDirPrefix.size());
    const auto* const digits_end = (digits.data() + digits.size());
    auto index = uint32_t{0};
    const auto result = std::from_chars(digits.data(), digits_end, index);
    if (digits.empty() || (result.ec != std::errc{}) || (result.ptr != digits_end)) {
      continue;
    }
    instances.push_back({index, it->path()});
  }

  std::sort(instances.begin(), instances.end(),
            [](const Instance_t& lhs, const Instance_t& rhs) { return (lhs.index < rhs.index); });
  return instances;
}

}  // namespace

auto scan_sensors(const std::string& hwmon_root) -> std::vector<SensorRecord_t> {
  auto records = std::vector<SensorRecord_t>{};

  for (const auto& instance : list_instances(hwmon_root)) {
    auto labels = std::map<std::pair<Category_t, uint32_t>, std::string>{};
    const auto first_of_instance = records.size();

    auto ec = std::error_code{};
    for (auto it = fs::directory_iterator(instance.path, ec);
         (!ec && (it != fs::directory_iterator{})); it.increment(ec)) {
      const auto name = it->path().filename().string();
      const auto parsed = parse_name(name);
      if (!parsed.has_value()) {
        continue;
      }

      if (parsed->suffix == kLabelSuffix) {
        auto label = read_first_line(it->path());
        if (label.has_value()) {
          labels[std::make_pair(parsed->kind->category, parsed->channel)] =
              std::move(label.value());
        }
        continue;
      }

      const auto attribute = lookup_attribute(parsed->suffix);
      if (!attribute.has_value()) {
        continue;
      }
      const auto value = read_int64(it->path());
      if (!value.has_value()) {
        continue;
      }
      const auto unit = is_alarm(attribute.value()) ? Unit_t::Bool : parsed->kind->unit;
      records.push_back({parsed->kind->category, attribute.value(), unit, parsed->channel,
                         instance.index, std::string{}, value.value()});
    }

    for (auto i = first_of_instance; i < records.size(); ++i) {
      const auto found = labels.find(std::make_pair(records[i].category, records[i].channel));
      if (found != labels.end()) {
        records[i].label = found->second;
      }
    }
  }

  std::sort(records.begin(), records.end(),
            [](const SensorRecord_t& lhs, const SensorRecord_t& rhs) {
              return (std::tie(lhs.instance, lhs.category, lhs.channel, lhs.attribute) <
                      std::tie(rhs.instance, rhs.category, rhs.channel, rhs.attribute));
            });
  return records;
}

}  // namespace amd::smi::nic::hwmon
