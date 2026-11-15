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
