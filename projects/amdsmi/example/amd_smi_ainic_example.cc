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

/**
 * amd_smi_ainic_example.cc
 *
 * Developer tool: discover all AI NIC (Pensando Pollara) devices visible to
 * amd-smi and print every piece of information the public API exposes.
 *
 * Uses ONLY the public header <amd_smi/amdsmi.h> — no internal headers —
 * so it works both on systems with real hardware and against a simulated
 * sysfs tree (set SMI_NIC_SYSFS_ROOT before running; see run_simulated_ainic.sh).
 *
 * Build: compiled as part of the amdsmi example CMake target amd_smi_ainic_info.
 */

#include <amd_smi/amdsmi.h>

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static const char* SEP = "  ";

static void print_section(const char* title) {
  std::cout << "\n" << SEP << "--- " << title << " ---\n";
}

static void print_field(const char* name, const char* value) {
  std::cout << SEP << SEP << std::left << std::setw(28) << name << ": " << value << "\n";
}

static void print_field(const char* name, uint64_t value, bool hex = false) {
  std::cout << SEP << SEP << std::left << std::setw(28) << name << ": ";
  if (hex) {
    std::cout << "0x" << std::hex << value << std::dec;
  } else {
    std::cout << value;
  }
  std::cout << "\n";
}

static void print_field(const char* name, uint32_t value, bool hex = false) {
  print_field(name, static_cast<uint64_t>(value), hex);
}

static void print_field(const char* name, uint16_t value, bool hex = false) {
  print_field(name, static_cast<uint64_t>(value), hex);
}

static void print_field(const char* name, uint8_t value) {
  print_field(name, static_cast<uint64_t>(value));
}

// ---------------------------------------------------------------------------
// Per-device display functions
// ---------------------------------------------------------------------------

static void show_asic_info(amdsmi_processor_handle handle) {
  amdsmi_nic_asic_info_t info{};
  if (amdsmi_get_nic_asic_info(handle, &info) != AMDSMI_STATUS_SUCCESS) {
    std::cout << SEP << SEP << "(ASIC info unavailable)\n";
    return;
  }
  print_section("ASIC");
  print_field("Vendor name", info.vendor_name);
  print_field("Product name", info.product_name);
  print_field("Part number", info.part_number);
  print_field("Serial number", info.serial_number);
  print_field("Vendor ID", info.vendor_id, /*hex=*/true);
  print_field("Subvendor ID", info.subvendor_id, /*hex=*/true);
  print_field("Device ID", info.device_id, /*hex=*/true);
  print_field("Subsystem ID", info.subsystem_id, /*hex=*/true);
  print_field("Revision", info.revision);
  print_field("Permanent addr", info.permanent_address);
}

static void show_bus_info(amdsmi_processor_handle handle) {
  amdsmi_nic_bus_info_t info{};
  if (amdsmi_get_nic_bus_info(handle, &info) != AMDSMI_STATUS_SUCCESS) {
    std::cout << SEP << SEP << "(bus info unavailable)\n";
    return;
  }
  print_section("PCIe Bus");
  std::cout << SEP << SEP << std::left << std::setw(28) << "BDF"
            << ": " << std::setfill('0') << std::hex << std::setw(4) << info.bdf.domain_number
            << ":" << std::setw(2) << (unsigned)info.bdf.bus_number << ":" << std::setw(2)
            << (unsigned)info.bdf.device_number << "." << std::setw(1)
            << (unsigned)info.bdf.function_number << std::dec << std::setfill(' ') << "\n";
  print_field("Max PCIe width (lanes)", info.max_pcie_width);
  print_field("Max PCIe speed (GT/s)", info.max_pcie_speed);
  print_field("PCIe interface version", info.pcie_interface_version);
  print_field("Slot type", info.slot_type);
}

static void show_driver_info(amdsmi_processor_handle handle) {
  amdsmi_nic_driver_info_t info{};
  if (amdsmi_get_nic_driver_info(handle, &info) != AMDSMI_STATUS_SUCCESS) {
    std::cout << SEP << SEP << "(driver info unavailable)\n";
    return;
  }
  print_section("Driver");
  print_field("Driver name", info.name);
  print_field("Driver version", info.version);
}

static void show_numa_info(amdsmi_processor_handle handle) {
  amdsmi_nic_numa_info_t info{};
  if (amdsmi_get_nic_numa_info(handle, &info) != AMDSMI_STATUS_SUCCESS) {
    std::cout << SEP << SEP << "(NUMA info unavailable)\n";
    return;
  }
  print_section("NUMA");
  print_field("NUMA node", info.node);
  print_field("CPU affinity", info.affinity);
}

static void show_port_info(amdsmi_processor_handle handle) {
  amdsmi_nic_port_info_t info{};
  if (amdsmi_get_nic_port_info(handle, &info) != AMDSMI_STATUS_SUCCESS) {
    std::cout << SEP << SEP << "(port info unavailable)\n";
    return;
  }
  print_section("Ports");
  std::cout << SEP << SEP << info.num_ports << " port(s)\n";
  for (uint32_t i = 0; i < info.num_ports; ++i) {
    const amdsmi_nic_port_t& p = info.ports[i];
    std::cout << SEP << SEP << "  [port " << i << "]\n";
    auto pf = [&](const char* name, const char* val) {
      std::cout << SEP << SEP << "    " << std::left << std::setw(26) << name << ": " << val
                << "\n";
    };
    auto pfn = [&](const char* name, uint64_t val) {
      std::cout << SEP << SEP << "    " << std::left << std::setw(26) << name << ": " << val
                << "\n";
    };
    pf("netdev", p.netdev);
    pf("type", p.type);
    pf("MAC address", p.mac_address);
    pf("link state", p.link_state);
    pf("autoneg", p.autoneg);
    pf("pause autoneg", p.pause_autoneg);
    pf("pause rx", p.pause_rx);
    pf("pause tx", p.pause_tx);
    pfn("port number", p.port_num);
    pfn("ifindex", p.ifindex);
    pfn("carrier", p.carrier);
    pfn("MTU", p.mtu);
    pfn("link speed", p.link_speed);
    pfn("active FEC", p.active_fec);
  }
}

static void show_rdma_info(amdsmi_processor_handle handle) {
  amdsmi_nic_rdma_devices_info_t info{};
  if (amdsmi_get_nic_rdma_dev_info(handle, &info) != AMDSMI_STATUS_SUCCESS) {
    std::cout << SEP << SEP << "(RDMA info unavailable)\n";
    return;
  }
  print_section("RDMA Devices");
  std::cout << SEP << SEP << info.num_rdma_dev << " RDMA device(s)\n";

  for (uint8_t d = 0; d < info.num_rdma_dev; ++d) {
    const amdsmi_nic_rdma_dev_info_t& dev = info.rdma_dev_info[d];
    std::cout << SEP << SEP << "  [rdma_dev " << (int)d << ": " << dev.rdma_dev << "]\n";
    auto df = [&](const char* name, const char* val) {
      std::cout << SEP << SEP << "    " << std::left << std::setw(26) << name << ": " << val
                << "\n";
    };
    df("node GUID", dev.node_guid);
    df("node type", dev.node_type);
    df("sys image GUID", dev.sys_image_guid);
    df("firmware version", dev.fw_ver);
    std::cout << SEP << SEP << "    " << dev.num_rdma_ports << " RDMA port(s)\n";

    for (uint8_t p = 0; p < dev.num_rdma_ports; ++p) {
      const amdsmi_nic_rdma_port_info_t& port = dev.rdma_port_info[p];
      std::cout << SEP << SEP << "      [rdma port " << (int)p << "]\n";
      auto pf = [&](const char* name, const char* val) {
        std::cout << SEP << SEP << "        " << std::left << std::setw(22) << name << ": " << val
                  << "\n";
      };
      auto pfn = [&](const char* name, uint64_t val) {
        std::cout << SEP << SEP << "        " << std::left << std::setw(22) << name << ": " << val
                  << "\n";
      };
      pf("netdev", port.netdev);
      pf("state", port.state);
      pfn("rdma port", port.rdma_port);
      pfn("max MTU", port.max_mtu);
      pfn("active MTU", port.active_mtu);

      // Fetch live RDMA statistics
      uint32_t num_stats = 0;
      amdsmi_status_t s = amdsmi_get_nic_rdma_port_statistics(handle, p, &num_stats, nullptr);
      if (s == AMDSMI_STATUS_SUCCESS && num_stats > 0) {
        std::vector<amdsmi_nic_stat_t> stats(num_stats);
        s = amdsmi_get_nic_rdma_port_statistics(handle, p, &num_stats, stats.data());
        if (s == AMDSMI_STATUS_SUCCESS) {
          std::cout << SEP << SEP << "        --- hw_counters (" << num_stats << ") ---\n";
          for (uint32_t k = 0; k < num_stats; ++k) {
            std::cout << SEP << SEP << "        " << std::left << std::setw(36) << stats[k].name
                      << ": " << stats[k].value << "\n";
          }
        }
      } else {
        std::cout << SEP << SEP << "        (RDMA statistics unavailable)\n";
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Main discovery loop
// ---------------------------------------------------------------------------

static void show_all_nic_devices() {
  uint32_t soc_count = 0;
  amdsmi_status_t status = amdsmi_get_socket_handles(&soc_count, nullptr);
  if (status != AMDSMI_STATUS_SUCCESS) {
    std::cerr << "amdsmi_get_socket_handles failed: " << (int)status << "\n";
    return;
  }

  if (soc_count == 0) {
    std::cout << "No sockets found.\n";
    return;
  }

  std::vector<amdsmi_socket_handle> sockets(soc_count);
  status = amdsmi_get_socket_handles(&soc_count, sockets.data());
  if (status != AMDSMI_STATUS_SUCCESS) {
    std::cerr << "amdsmi_get_socket_handles (data) failed: " << (int)status << "\n";
    return;
  }

  uint32_t total_nics = 0;
  for (uint32_t si = 0; si < soc_count; ++si) {
    uint32_t nic_count = 0;
    status = amdsmi_get_processor_handles_by_type(sockets[si], AMDSMI_PROCESSOR_TYPE_AMD_NIC,
                                                  nullptr, &nic_count);
    if (status != AMDSMI_STATUS_SUCCESS || nic_count == 0) continue;

    std::vector<amdsmi_processor_handle> handles(nic_count);
    status = amdsmi_get_processor_handles_by_type(sockets[si], AMDSMI_PROCESSOR_TYPE_AMD_NIC,
                                                  handles.data(), &nic_count);
    if (status != AMDSMI_STATUS_SUCCESS) continue;

    for (uint32_t ni = 0; ni < nic_count; ++ni) {
      std::cout << "\n=== AI NIC device " << total_nics << " (socket " << si << ", device " << ni
                << ") ===\n";
      show_asic_info(handles[ni]);
      show_bus_info(handles[ni]);
      show_driver_info(handles[ni]);
      show_numa_info(handles[ni]);
      show_port_info(handles[ni]);
      show_rdma_info(handles[ni]);
      ++total_nics;
    }
  }

  std::cout << "\nTotal AI NIC devices found: " << total_nics << "\n";
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

int main() {
  const char* sysfs_root = std::getenv("SMI_NIC_SYSFS_ROOT");
  if (sysfs_root && sysfs_root[0] != '\0') {
    std::cout << "SMI_NIC_SYSFS_ROOT = " << sysfs_root << "  (simulation mode)\n";
  } else {
    std::cout << "SMI_NIC_SYSFS_ROOT not set  (real hardware mode)\n";
  }

  amdsmi_status_t status = amdsmi_init(AMDSMI_INIT_AMD_NICS);
  if (status != AMDSMI_STATUS_SUCCESS) {
    std::cerr << "amdsmi_init(AMDSMI_INIT_AMD_NICS) failed: " << (int)status << "\n";
    return 1;
  }
  std::cout << "amd-smi initialized\n";

  show_all_nic_devices();

  amdsmi_shut_down();
  std::cout << "\namd-smi shut down\n";
  return 0;
}
