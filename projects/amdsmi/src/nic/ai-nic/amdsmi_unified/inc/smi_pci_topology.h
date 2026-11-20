// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMDSMI_UNIFIED_PCI_TOPOLOGY_H_
#define AMDSMI_UNIFIED_PCI_TOPOLOGY_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace amd::smi::nic::pci {

/** Where sysfs lists PCI devices. Every lookup takes the root, so a test can use a fake tree. */
inline const auto kDefaultPciRoot = std::string{"/sys/bus/pci/devices"};

/** How two PCI devices are connected. The values equal smi_nic_link_type_t. */
enum class LinkType_t : uint8_t { Unknown = 0, Pcie = 1, Numa = 2, Xnuma = 3 };

/** True for the sysfs form dddd:bb:dd.f with hexadecimal digits. */
auto is_valid_bdf(const std::string& bdf) -> bool;

/** A packed BDF (domain << 16 | bus << 8 | device << 3 | function) as dddd:bb:dd.f. */
auto format_bdf(uint64_t packed_bdf) -> std::string;

/** The bridge above a device, or nullopt for a device under the root complex or not in sysfs. */
auto get_pcie_parent(const std::string& bdf, const std::string& pci_root = kDefaultPciRoot)
    -> std::optional<std::string>;

/** Bridges above a device, nearest first, ending at the root port. */
auto get_pcie_ancestors(const std::string& bdf, const std::string& pci_root = kDefaultPciRoot)
    -> std::vector<std::string>;

/**
 * True when the nearest bridge the two devices share is a switch. A shared root port is the root
 * complex, which is not a switch.
 */
auto is_same_pcie_switch(const std::string& first_bdf, const std::string& second_bdf,
                         const std::string& pci_root = kDefaultPciRoot) -> bool;

/** NUMA node of a device; nullopt when sysfs has none or reports -1 (unknown). */
auto get_numa_node(const std::string& bdf, const std::string& pci_root = kDefaultPciRoot)
    -> std::optional<int>;

/**
 * Pcie when the devices share a switch, otherwise Numa on one NUMA node and Xnuma across nodes.
 * Unknown when a BDF is invalid or a NUMA node cannot be read.
 */
auto get_link_type(const std::string& first_bdf, const std::string& second_bdf,
                   const std::string& pci_root = kDefaultPciRoot) -> LinkType_t;

}  // namespace amd::smi::nic::pci

#endif  // AMDSMI_UNIFIED_PCI_TOPOLOGY_H_
