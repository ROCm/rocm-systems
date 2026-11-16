// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "location.h"

#include <charconv>

namespace rocjitsu::isa_diff {

LocSet all_locs() {
  LocSet locs;
  locs.set();
  return locs;
}

std::string loc_name(unsigned loc) {
  if (loc < kNumSgprs)
    return "s" + std::to_string(loc);
  switch (loc) {
  case kLocVccLo:
    return "vcc_lo";
  case kLocVccHi:
    return "vcc_hi";
  case kLocExecLo:
    return "exec_lo";
  case kLocExecHi:
    return "exec_hi";
  case kLocM0:
    return "m0";
  case kLocScc:
    return "scc";
  default:
    return "<invalid>";
  }
}

std::optional<unsigned> parse_loc(std::string_view name) {
  if (name.size() > 1 && name.front() == 's' && name != "scc") {
    unsigned index = 0;
    const auto *end = name.data() + name.size();
    auto [ptr, ec] = std::from_chars(name.data() + 1, end, index);
    if (ec != std::errc{} || ptr != end || index >= kNumSgprs)
      return std::nullopt;
    return index;
  }
  for (unsigned loc = kNumSgprs; loc < kNumLocs; ++loc)
    if (loc_name(loc) == name)
      return loc;
  return std::nullopt;
}

std::string format_locs(const LocSet &locs) {
  std::string out;
  for (unsigned loc = 0; loc < kNumLocs; ++loc) {
    if (!locs.test(loc))
      continue;
    if (!out.empty())
      out += ',';
    out += loc_name(loc);
  }
  return out;
}

std::string vgpr_name(unsigned r) {
  return r < kAgprBase ? "v" + std::to_string(r) : "a" + std::to_string(r - kAgprBase);
}

std::optional<unsigned> parse_vgpr_name(std::string_view name) {
  if (name.size() < 2 || (name.front() != 'v' && name.front() != 'a'))
    return std::nullopt;
  unsigned index = 0;
  const auto *end = name.data() + name.size();
  auto [ptr, ec] = std::from_chars(name.data() + 1, end, index);
  if (ec != std::errc{} || ptr != end || index >= kNumArchVgprs)
    return std::nullopt;
  return name.front() == 'a' ? kAgprBase + index : index;
}

std::string format_vgprs(const VSet &v) {
  std::string out;
  for (unsigned r = 0; r < kNumVgprs; ++r) {
    if (!v.test(r))
      continue;
    if (!out.empty())
      out += ',';
    out += vgpr_name(r);
  }
  return out;
}

std::optional<VSet> parse_vgprs(std::string_view text) {
  VSet v;
  while (!text.empty()) {
    const size_t comma = text.find(',');
    const std::string_view item = text.substr(0, comma);
    if (!item.empty()) {
      const auto index = parse_vgpr_name(item);
      if (!index)
        return std::nullopt;
      v.set(*index);
    }
    if (comma == std::string_view::npos)
      break;
    text.remove_prefix(comma + 1);
  }
  return v;
}

unsigned vgpr_extent(const VSet &v) {
  for (unsigned r = kNumVgprs; r-- > 0;)
    if (v.test(r))
      return r + 1;
  return 0;
}

std::optional<LocSet> parse_locs(std::string_view text) {
  LocSet locs;
  while (!text.empty()) {
    const size_t comma = text.find(',');
    const std::string_view item = text.substr(0, comma);
    if (!item.empty()) {
      auto loc = parse_loc(item);
      if (!loc)
        return std::nullopt;
      locs.set(*loc);
    }
    if (comma == std::string_view::npos)
      break;
    text.remove_prefix(comma + 1);
  }
  return locs;
}

} // namespace rocjitsu::isa_diff
