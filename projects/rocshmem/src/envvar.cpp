/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "envvar.hpp"

#include <algorithm>
#include <istream>
#include <list>
#include <mutex>
#include <ostream>
#include <string>
#include <tuple>
#include <unordered_map>

#include <unistd.h>

namespace rocshmem {
namespace envvar {
  inline namespace _base {
    const var<bool> uniqueid_with_mpi("UNIQUEID_WITH_MPI",
      "Defines whether rocSHMEM is expected to use MPI internally when using the uniqueId based initialization. 0: Do not use MPI; 1: Use MPI",
      false);
    const var<types::debug_level> debug_level("DEBUG_LEVEL",
      "Debug output level (NONE, VERSION, WARN, ENV:MODIFIED, ENV:ALL, ENV:FULL, INFO, TRACE)",
      types::debug_level::NONE);
    const var<size_t> heap_size("HEAP_SIZE",
      "Defines the size of the rocSHMEM symmetric heap in bytes (per PE). Size in bytes (per PE); Note: the heap is on GPU memory",
      1L << 30);
    const var<std::string> backend("BACKEND",
      "When rocSHMEM is compiled for all backends, this environment variable selects which backend to execute. The default value is an empty string and rocSHMEM auto-selects the most appropriate backend. ipc: IPC Backend; ro: Reverse Offload Backend; gda: GPU Direct Async Backend");
    const var<size_t> max_num_contexts("MAX_NUM_CONTEXTS",
      "Defines the number of contexts an application can use.",
      32);
    const var<size_t> max_num_host_contexts("MAX_NUM_HOST_CONTEXTS",
      "Maximum number of host-side communication contexts",
      1);
    const var<size_t> max_num_teams("MAX_NUM_TEAMS",
      "Defines the number of teams an application can use.",
      40);
    const var<std::string> hca_list("HCA_LIST",
      "Comma separated list of NIC names that can be used by rocSHMEM. Unlike ROCSHMEM_USE_IB_HCA, when this variable is set, NIC auto-detection and mapping still executes, but NICs that are not in the list are discarded before auto-detection runs. Prefixing the list with ^ turns the list in an exclude list, NICs that are in the list are discarded before auto-detection runs. The default value is an empty string and rocSHMEM auto-detects the most appropriate NIC. Example value: bnxt_re1,bnxt_re11, ^mlx5_0,mlx5_3");
    const var<std::string> requested_nic("USE_IB_HCA",
      "Forces the NIC that this PE uses. When this value is set NIC auto-detection and mapping is disabled, the NIC specified in the variable will be selected. The default value is an empty string and rocSHMEM auto-detects the most appropriate NIC. Example value: bnxt_re0");
    const var<bool> disable_mixed_ipc("DISABLE_MIXED_IPC",
      "Defines whether to force using the network conduit even when IPC is available. 0: Use IPC when available; 1: Force network conduit",
      false);
    const var<bool> disable_ipc("DISABLE_IPC",
      "DEPRECATED: Synonym for ROCSHMEM_DISABLE_MIXED_IPC. Force using network conduit even when IPC is available",
      false);
    const var<size_t> max_wavefront_buffers("MAX_WF_BUFFERS",
      "Maximum number of wavefront buffer arrays in default context (determines size of status, return, and atomic return buffers)",
      1024);
  }  // inline namespace _base

  namespace bootstrap {
    const var<int64_t> timeout("TIMEOUT",
      "Bootstrap initialization timeout in seconds",
      5);
    const var<std::string> hostid("HOSTID",
      "Override host identifier for bootstrap. Empty string uses hostname.");
    const var<types::socket_family> socket_family("SOCKET_FAMILY",
      "Socket family for bootstrap (AF_UNSPEC, AF_INET, AF_INET6)",
      types::socket_family::UNSPEC);
    const var<std::string> socket_ifname("SOCKET_IFNAME",
      "Chooses the interface to bootstrap rocSHMEM with. Only valid when not using MPI. The default value is an empty string and rocSHMEM auto-detects the most appropriate interface. Example value: eno8303");
  }  // namespace bootstrap

  namespace ro {
    const var<bool> disable_ipc("DISABLE_IPC",
      "DEPRECATED: Synonym for ROCSHMEM_DISABLE_MIXED_IPC. Force using network conduit even when IPC is available",
      false);
    const var<useconds_t> progress_delay("PROGRESS_DELAY",
      "Progress engine delay in microseconds (reduces memory subsystem load for performance)",
      3);
    const var<bool> net_cpu_queue("NET_CPU_QUEUE",
      "Use CPU queue for network operations in Reverse Offload backend",
      false);
  }  // namespace ro

  namespace gda {
    const var<std::string> provider("PROVIDER",
      "When rocSHMEM is compiled with support for multiple NIC vendors, the environment variable selects the desired provider. The default value is an empty string and rocSHMEM auto-detects the most appropriate NIC. bnxt: Broadcom Thor 2; pensando: AMD Pensando Pollara; ionic: AMD Pensando Pollara (alias); mlx5: Mellanox ConnectX-7");
    const var<bool> alternate_qp_ports("ALTERNATE_QP_PORTS",
      "Enables or disables alternating QP mappings across rocSHMEM contexts. 0: Disabled; 1: Enabled. This helps saturate bandwidth on multiport bonded interfaces",
      true);
    const var<uint8_t> traffic_class("TRAFFIC_CLASS",
      "When using an NIC with an Ethernet link layer, this sets the traffic class for the QPs.",
      0);
    const var<bool> pcie_relaxed_ordering("PCIE_RELAXED_ORDERING",
      "Enables PCIe Relaxed Ordering when registering the symmetric heap with the RDMA NICs. 0: Disabled; 1: Enabled",
      false);
    const var<bool> enable_dmabuf("ENABLE_DMABUF",
      "Enable dmabuf support for memory registration. 0: Disabled; 1: Enabled",
      false);
    const var<bool> override_nic_firmware_check("OVERRIDE_NIC_FIRMWARE_CHECK",
      "This environment variable should be used with caution. It overrides the NIC firmware check if a user wants to use an unsupported NIC firmware. If the firmware check is disabled rocSHMEM is not guaranteed to work. 0: Disabled; 1: Enabled",
      false);
    const var<std::string> alltoallv_wg_algo("ALLTOALLV_WG_ALGO",
      "Selects between two algorithms to use for GDA based alltoallv. The GET algorithm uses an initial round of alltoallv communication to distribute displacements then a second round to get transfer data. This algorithm has a higher latency but has better performance for large messages. The COPY algorithm does an alltoallv communication pattern into a staging buffer then does a copy into the destination buffers. This reduces latency but requires more memory, this algorithm only works for small messages.");
    const var<uint32_t> sq_size("SQ_SIZE",
      "This environment variable sets the length of the SQ for GDA. Maximum number of Work Queue Entries (WQEs) posted on the Send Queue (SQ)",
      1024);
    const var<size_t> num_qps_per_pe_default_ctx("NUM_QPS_PER_PE_DEFAULT_CTX",
      "Sets the number of Queue Pairs (QPs) to create per PE for the default context.",
      1);
    const var<size_t> num_qps_per_pe_usr_ctx("NUM_QPS_PER_PE_USR_CTX",
      "Sets the number of Queue Pairs (QPs) to create per PE for each user context.",
      1);
  }  // namespace gda

  namespace _detail {
    std::tuple<var_map_t&, std::mutex&> get_var_map() {
      // construct on first use idiom
      // allocate variable_map on heap to prevent static initialization order fiasco
      static auto variable_map = new var_map_t();
      static std::mutex map_mutex;
      // use std::tie to return a tuple of references
      return std::tie(*variable_map, map_mutex);
    }
  }  // namespace _detail

  namespace types {
    inline namespace _sf {
      std::istream& operator>>(std::istream& is, socket_family& family) {
        std::string family_str;
        is >> family_str;
        if (family_str == "AF_INET" ||
            family_str == "INET") {
          family = socket_family::INET;
        } else if (family_str == "AF_INET6" ||
                   family_str == "INET6") {
          family = socket_family::INET6;
        } else if (family_str == "AF_UNSPEC" ||
                   family_str == "UNSPEC") {
          family = socket_family::UNSPEC;
        } else {
          // all other inputs are invalid
          is.setstate(std::ios_base::failbit);
          family = socket_family::UNSPEC;
        }
        return is;
      }

      std::ostream& operator<<(std::ostream& os, const socket_family& family) {
        switch (family) {
        case socket_family::UNSPEC:
          return os << "AF_UNSPEC";
        case socket_family::INET:
          return os << "AF_INET";
        case socket_family::INET6:
          return os << "AF_INET6";
        }
      }
    }  // inline namespace _sf

    inline namespace _debug {
      std::istream& operator>>(std::istream& is, debug_level& level) {
        std::string level_str;
        is >> level_str;
        std::transform(level_str.begin(), level_str.end(), level_str.begin(), ::toupper);
        if (level_str == "NONE") {
          level = debug_level::NONE;
        } else if (level_str == "VERSION") {
          level = debug_level::VERSION;
        } else if (level_str == "WARN") {
          level = debug_level::WARN;
        } else if (level_str == "ENV") {
          level = debug_level::ENV;
        } else if (level_str == "ENV:MODIFIED") {
          level = debug_level::ENV;
        } else if (level_str == "ENV:ALL") {
          level = debug_level::ENV_ALL;
        } else if (level_str == "ENV:FULL") {
          level = debug_level::ENV_FULL;
        } else if (level_str == "INFO") {
          level = debug_level::INFO;
        } else if (level_str == "TRACE") {
          level = debug_level::TRACE;
        } else {
          // all other inputs are invalid
          is.setstate(std::ios_base::failbit);
          level = debug_level::NONE;
        }
        return is;
      }

      std::ostream& operator<<(std::ostream& os, const debug_level& level) {
        switch (level) {
        case debug_level::NONE:
          return os << "NONE";
        case debug_level::VERSION:
          return os << "VERSION";
        case debug_level::WARN:
          return os << "WARN";
        case debug_level::ENV:
          return os << "ENV:MODIFIED";
        case debug_level::ENV_ALL:
          return os << "ENV:ALL";
        case debug_level::ENV_FULL:
          return os << "ENV:FULL";
        case debug_level::INFO:
          return os << "INFO";
        case debug_level::TRACE:
          return os << "TRACE";
        }
      }
    }  // inline namespace _debug
  }  // namespace types

  void print_envvars(print_mode mode, std::ostream& os) {
    auto [var_map, map_mutex] = _detail::get_var_map();
    std::lock_guard map_lock(map_mutex);

    // Category names for display
    const std::unordered_map<category::tag, std::string> category_names = {
      {category::tag::ROCSHMEM, "ROCSHMEM"},
      {category::tag::BOOTSTRAP, "BOOTSTRAP"},
      {category::tag::REVERSE_OFFLOAD, "REVERSE_OFFLOAD"},
      {category::tag::GDA, "GDA (GPU Direct Async)"}
    };

    const std::unordered_map<category::tag, std::string> category_descriptions = {
      {category::tag::ROCSHMEM, "Core rocSHMEM configuration variables"},
      {category::tag::BOOTSTRAP, "Bootstrap initialization variables"},
      {category::tag::REVERSE_OFFLOAD, "Reverse Offload backend variables"},
      {category::tag::GDA, "GPU Direct Async backend variables"}
    };

    os << "################################################################################\n";
    os << "#                   rocSHMEM Environment Variables                             #\n";
    os << "################################################################################\n";

    // Iterate through each category
    for (const auto& cat : {category::tag::ROCSHMEM,
                            category::tag::BOOTSTRAP,
                            category::tag::REVERSE_OFFLOAD,
                            category::tag::GDA}) {

      // Skip if category not in map
      if (var_map.find(cat) == var_map.end()) {
        continue;
      }

      const auto& var_list = var_map.at(cat);

      // Print category header only for FULL_DOCUMENTATION mode
      if (mode == print_mode::FULL_DOCUMENTATION) {
        os << "#" << std::string(78, '-') << "#\n";
        os << "# " << category_names.at(cat) << ": "
           << category_descriptions.at(cat) << "\n";
        os << "#" << std::string(78, '-') << "#\n";
      }

      // Iterate through each variable in the category
      for (const auto& var_ref : var_list) {
        // Visit the variant to extract information from the var
        std::visit([&os, mode](const auto& v) {
          // For MODIFIED mode, skip variables using default values
          if (mode == print_mode::MODIFIED && v.get().is_default()) {
            return;
          }

          // Helper lambda to format value
          auto format_value = [](const auto& value) -> std::string {
            using ValueType = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<ValueType, bool>) {
              return value ? "true" : "false";
            } else {
              std::ostringstream oss;
              oss << value;
              return oss.str();
            }
          };

          if (mode == print_mode::MODIFIED || mode == print_mode::ALL_VALUES) {
            // Simple format: NAME=value
            os << "# " << v.get().get_name() << "=" << format_value(v.get().get_value()) << "\n";
          } else {
            // FULL_DOCUMENTATION mode
            os << "# " << v.get().get_name() << "\n";

            // Print documentation if available
            if (!v.get().get_doc().empty()) {
              os << "#   Description:   " << v.get().get_doc() << "\n";
            }

            // Print default value
            os << "#   Default:       " << format_value(v.get().get_default()) << "\n";

            // Print current value and whether it was set
            os << "#   Current:       " << format_value(v.get().get_value());

            if (v.get().is_default()) {
              os << " (using default)";
            } else {
              os << " (set from environment)";
            }
            os << "\n";
          }
        }, var_ref);
      }
    }

    if (mode != print_mode::MODIFIED) {
      os << "#\n";
      os << "#------------------------------------------------------------------------------#\n";
      os << "# For more information, see:\n";
      os << "# https://rocm.docs.amd.com/projects/rocSHMEM/en/latest/api/env_variables.html\n";
      os << "#------------------------------------------------------------------------------#\n";
    }
    os << "################################################################################\n";
  }

}  // namespace envvar
}  // namespace rocshmem
