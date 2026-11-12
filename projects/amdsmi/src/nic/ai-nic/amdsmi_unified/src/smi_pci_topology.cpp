// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "smi_pci_topology.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <system_error>
#include <variant>

#include "smi_sysfs.h"

namespace amd::smi::nic::pci {

namespace {

namespace fs = std::filesystem;

constexpr auto kBdfLength = size_t{12};
constexpr auto kBdfBusSeparatorPos = size_t{4};
constexpr auto kBdfDeviceSeparatorPos = size_t{7};
constexpr auto kBdfFunctionSeparatorPos = size_t{10};
constexpr auto kDomainShift = unsigned{16};
constexpr auto kBusShift = unsigned{8};
constexpr auto kDeviceShift = unsigned{3};
constexpr auto kDomainMask = uint64_t{0xFFFFFFFF};
constexpr auto kBusMask = uint64_t{0xFF};
constexpr auto kDeviceMask = uint64_t{0x1F};
constexpr auto kFunctionMask = uint64_t{0x7};
constexpr auto kDomainWidth = 4;
constexpr auto kBusWidth = 2;
constexpr auto kDeviceWidth = 2;
constexpr auto kFunctionWidth = 1;
constexpr auto kMaxPcieAncestors = 64;
constexpr auto kNumaNodeFile = "numa_node";

auto is_hex(char digit) -> bool { return (std::isxdigit(static_cast<unsigned char>(digit)) != 0); }

}  // namespace

auto is_valid_bdf(const std::string& bdf) -> bool {
  if (bdf.size() != kBdfLength) {
    return false;
  }
  for (auto i = size_t{0}; i < kBdfLength; ++i) {
    const auto is_separator = ((i == kBdfBusSeparatorPos) || (i == kBdfDeviceSeparatorPos) ||
                               (i == kBdfFunctionSeparatorPos));
    if (!is_separator) {
      if (!is_hex(bdf[i])) {
        return false;
      }
      continue;
    }
    const auto expected = (i == kBdfFunctionSeparatorPos) ? '.' : ':';
    if (bdf[i] != expected) {
      return false;
    }
  }
  return true;
}

auto format_bdf(uint64_t packed_bdf) -> std::string {
  auto text = std::ostringstream{};
  text << std::hex << std::setfill('0') << std::setw(kDomainWidth)
       << ((packed_bdf >> kDomainShift) & kDomainMask) << ':' << std::setw(kBusWidth)
       << ((packed_bdf >> kBusShift) & kBusMask) << ':' << std::setw(kDeviceWidth)
       << ((packed_bdf >> kDeviceShift) & kDeviceMask) << '.' << std::setw(kFunctionWidth)
       << (packed_bdf & kFunctionMask);
  return text.str();
}

auto get_pcie_parent(const std::string& bdf, const std::string& pci_root)
    -> std::optional<std::string> {
  if (!is_valid_bdf(bdf)) {
    return std::nullopt;
  }
  // The link names the device's real path, e.g. .../0000:00:01.0/0000:01:00.0; the directory above
  // it is the parent bridge, or the domain root (not a BDF) for a device under the root complex.
  auto ec = std::error_code{};
  const auto target = fs::read_symlink(fs::path(pci_root) / bdf, ec);
  if (ec) {
    return std::nullopt;
  }
  auto parent = target.parent_path().filename().string();
  if (!is_valid_bdf(parent)) {
    return std::nullopt;
  }
  return parent;
}

auto get_pcie_ancestors(const std::string& bdf, const std::string& pci_root)
    -> std::vector<std::string> {
  auto ancestors = std::vector<std::string>{};
  auto current = bdf;
  for (auto hop = 0; hop < kMaxPcieAncestors; ++hop) {
    auto parent = get_pcie_parent(current, pci_root);
    if (!parent.has_value()) {
      break;
    }
    current = parent.value();
    ancestors.push_back(current);
  }
  return ancestors;
}

auto is_same_pcie_switch(const std::string& first_bdf, const std::string& second_bdf,
                         const std::string& pci_root) -> bool {
  if (!is_valid_bdf(first_bdf) || !is_valid_bdf(second_bdf)) {
    return false;
  }
  const auto first_ancestors = get_pcie_ancestors(first_bdf, pci_root);
  for (const auto& ancestor : get_pcie_ancestors(second_bdf, pci_root)) {
    if (std::find(first_ancestors.begin(), first_ancestors.end(), ancestor) ==
        first_ancestors.end()) {
      continue;
    }
    // The nearest shared bridge decides: a switch has a bridge above it, the root port does not.
    return get_pcie_parent(ancestor, pci_root).has_value();
  }
  return false;
}

auto get_numa_node(const std::string& bdf, const std::string& pci_root) -> std::optional<int> {
  if (!is_valid_bdf(bdf)) {
    return std::nullopt;
  }
  auto value = SmiSysfsReader::SysfsValue{};
  const auto path = (fs::path(pci_root) / bdf / kNumaNodeFile).string();
  if (SmiSysfsReader::readLine(path, value, /*is_success_logged=*/false) !=
      SmiSysfsReader::SysfsStatus::Success) {
    return std::nullopt;
  }
  const auto* node = std::get_if<int>(&value);
  if ((node == nullptr) || (*node < 0)) {
    return std::nullopt;
  }
  return *node;
}

auto get_link_type(const std::string& first_bdf, const std::string& second_bdf,
                   const std::string& pci_root) -> LinkType_t {
  if (!is_valid_bdf(first_bdf) || !is_valid_bdf(second_bdf)) {
    return LinkType_t::Unknown;
  }
  if (is_same_pcie_switch(first_bdf, second_bdf, pci_root)) {
    return LinkType_t::Pcie;
  }
  const auto first_node = get_numa_node(first_bdf, pci_root);
  const auto second_node = get_numa_node(second_bdf, pci_root);
  if (!first_node.has_value() || !second_node.has_value()) {
    return LinkType_t::Unknown;
  }
  return (first_node.value() == second_node.value()) ? LinkType_t::Numa : LinkType_t::Xnuma;
}

}  // namespace amd::smi::nic::pci
