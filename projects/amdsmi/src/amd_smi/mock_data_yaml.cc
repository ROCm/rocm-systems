/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

// Mock-data YAML loader.
//
// Schema (intentionally narrow; this is NOT a general YAML parser):
//
//   gpu_metrics:
//     temperature_edge: 45
//     temperature_hotspot: 52
//     temperature_mem: 48
//     average_socket_power: 120
//     average_gfx_activity: 75
//     average_umc_activity: 30
//     current_socket_power: 125
//     current_gfxclk: 1800
//     energy_accumulator: 1234567
//     system_clock_counter: 1000000000
//     firmware_timestamp: 9876543210
//     temperature_hbm: [50, 51, 52, 53]
//     vcn_activity:    [10, 20, 30, 40]
//
// Supported syntax:
//   * One top-level section header `gpu_metrics:` (required for any gpu_metrics
//     fields to be loaded).
//   * Indented `key: value` pairs (scalars).
//   * Flow-style integer arrays in square brackets.
//   * Lines starting with `#` and blank lines are ignored. Trailing `# ...`
//     comments after a value are also stripped.
//
// Mapped fields cover ints (uint16/uint32-equiv), uint64s, and arrays - the
// representative subset listed in slice-2 brief. Add more keys here as the
// backend grows; do not silently widen this without updating the schema doc.

#include "amd_smi/impl/mock_data_yaml.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace amd::smi {

namespace {

std::string strip_comment(const std::string& line) {
  auto hash = line.find('#');
  if (hash == std::string::npos) return line;
  return line.substr(0, hash);
}

std::string trim(const std::string& s) {
  auto begin = s.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) return {};
  auto end = s.find_last_not_of(" \t\r\n");
  return s.substr(begin, end - begin + 1);
}

size_t leading_spaces(const std::string& line) {
  size_t i = 0;
  while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
  return i;
}

// Try parse a uint64 in decimal. Returns nullopt on any error or trailing junk.
std::optional<uint64_t> parse_u64(const std::string& s) {
  std::string t = trim(s);
  if (t.empty()) return std::nullopt;
  for (char c : t) {
    if (!std::isdigit(static_cast<unsigned char>(c))) return std::nullopt;
  }
  try {
    return std::stoull(t);
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<std::vector<uint16_t>> parse_u16_array(const std::string& s) {
  std::string t = trim(s);
  if (t.size() < 2 || t.front() != '[' || t.back() != ']') return std::nullopt;
  std::string inner = t.substr(1, t.size() - 2);
  std::vector<uint16_t> out;
  std::stringstream ss(inner);
  std::string item;
  while (std::getline(ss, item, ',')) {
    auto v = parse_u64(item);
    if (!v || *v > 0xFFFFu) return std::nullopt;
    out.push_back(static_cast<uint16_t>(*v));
  }
  return out;
}

bool assign_u16(std::optional<uint16_t>& field, const std::string& value, std::string& err,
                const std::string& key) {
  auto v = parse_u64(value);
  if (!v || *v > 0xFFFFu) {
    err = "invalid uint16 value for '" + key + "': '" + value + "'";
    return false;
  }
  field = static_cast<uint16_t>(*v);
  return true;
}

bool assign_u64(std::optional<uint64_t>& field, const std::string& value, std::string& err,
                const std::string& key) {
  auto v = parse_u64(value);
  if (!v) {
    err = "invalid uint64 value for '" + key + "': '" + value + "'";
    return false;
  }
  field = *v;
  return true;
}

bool assign_u16_array(std::optional<std::vector<uint16_t>>& field, const std::string& value,
                      std::string& err, const std::string& key) {
  auto arr = parse_u16_array(value);
  if (!arr) {
    err = "invalid uint16 array for '" + key + "': '" + value + "'";
    return false;
  }
  field = std::move(*arr);
  return true;
}

bool apply_gpu_metrics_kv(MockGPUMetricsData& m, const std::string& key, const std::string& value,
                          std::string& err) {
  if (key == "temperature_edge") return assign_u16(m.temperature_edge, value, err, key);
  if (key == "temperature_hotspot") return assign_u16(m.temperature_hotspot, value, err, key);
  if (key == "temperature_mem") return assign_u16(m.temperature_mem, value, err, key);
  if (key == "average_socket_power") return assign_u16(m.average_socket_power, value, err, key);
  if (key == "average_gfx_activity") return assign_u16(m.average_gfx_activity, value, err, key);
  if (key == "average_umc_activity") return assign_u16(m.average_umc_activity, value, err, key);
  if (key == "current_socket_power") return assign_u16(m.current_socket_power, value, err, key);
  if (key == "current_gfxclk") return assign_u16(m.current_gfxclk, value, err, key);
  if (key == "energy_accumulator") return assign_u64(m.energy_accumulator, value, err, key);
  if (key == "system_clock_counter") return assign_u64(m.system_clock_counter, value, err, key);
  if (key == "firmware_timestamp") return assign_u64(m.firmware_timestamp, value, err, key);
  if (key == "temperature_hbm") return assign_u16_array(m.temperature_hbm, value, err, key);
  if (key == "vcn_activity") return assign_u16_array(m.vcn_activity, value, err, key);
  // Unknown keys are tolerated (forward compatibility): silently skip.
  return true;
}

}  // namespace

MockDataLoadResult load_mock_data_yaml(const std::string& path) {
  MockDataLoadResult result;

  std::ifstream file(path);
  if (!file.is_open()) {
    result.error = "cannot open mock data file: " + path;
    return result;
  }

  std::string line;
  std::string current_section;
  size_t section_indent = 0;
  size_t line_no = 0;

  while (std::getline(file, line)) {
    ++line_no;
    std::string stripped = strip_comment(line);
    std::string trimmed = trim(stripped);
    if (trimmed.empty()) continue;

    size_t indent = leading_spaces(stripped);
    auto colon = trimmed.find(':');
    if (colon == std::string::npos) {
      result.error = "line " + std::to_string(line_no) + ": missing ':'";
      return result;
    }

    std::string key = trim(trimmed.substr(0, colon));
    std::string value = trim(trimmed.substr(colon + 1));

    // Top-level section header (no indentation, empty value).
    if (indent == 0 && value.empty()) {
      current_section = key;
      section_indent = 0;
      continue;
    }

    // Indented key: must belong to a known section.
    if (indent > section_indent && !current_section.empty() && !value.empty()) {
      if (current_section == "gpu_metrics") {
        std::string err;
        if (!apply_gpu_metrics_kv(result.data.gpu_metrics, key, value, err)) {
          result.error = "line " + std::to_string(line_no) + ": " + err;
          return result;
        }
      }
      continue;
    }

    // Anything else is a parse error in this minimal grammar.
    result.error = "line " + std::to_string(line_no) + ": unexpected syntax";
    return result;
  }

  result.ok = true;
  return result;
}

}  // namespace amd::smi
