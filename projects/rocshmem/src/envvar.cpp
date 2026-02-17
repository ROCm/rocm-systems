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
    const var<bool> uniqueid_with_mpi("UNIQUEID_WITH_MPI", "", false);
    const var<types::debug_level> debug_level("DEBUG_LEVEL", "", types::debug_level::NONE);
    const var<size_t> heap_size("HEAP_SIZE", "", 1L << 30);
    const var<size_t> max_num_teams("MAX_NUM_TEAMS", "", 40);
    const var<size_t> max_num_host_contexts("MAX_NUM_HOST_CONTEXTS", "", 1);
    const var<size_t> max_num_contexts("MAX_NUM_CONTEXTS", "", 32);
    const var<size_t> max_wavefront_buffers("MAX_WF_BUFFERS", "", 1024);
    const var<std::string> requested_nic("USE_IB_HCA", "");
    const var<std::string> hca_list("HCA_LIST", "");
    const var<uint32_t> sq_size("SQ_SIZE", "", 1024);
    const var<std::string> backend("BACKEND", "");
    const var<bool> disable_mixed_ipc("DISABLE_MIXED_IPC", "", false);
    const var<bool> disable_ipc("DISABLE_IPC", "", false);
  }  // inline namespace _base

  namespace bootstrap {
    const var<int64_t> timeout("TIMEOUT", "", 5);
    const var<std::string> hostid("HOSTID", "");
    const var<types::socket_family> socket_family("SOCKET_FAMILY", "", types::socket_family::UNSPEC);
    const var<std::string> socket_ifname("SOCKET_IFNAME", "");
  }  // namespace bootstrap

  namespace ro {
    const var<bool> disable_ipc("DISABLE_IPC", "", false);
    const var<useconds_t> progress_delay("PROGRESS_DELAY", "", 3);
    const var<bool> net_cpu_queue("NET_CPU_QUEUE", "", false);
  }  // namespace ro

  namespace gda {
    const var<std::string> provider("PROVIDER", "");
    const var<bool> alternate_qp_ports("ALTERNATE_QP_PORTS", "", true);
    const var<uint8_t> traffic_class("TRAFFIC_CLASS", "", 0);
    const var<bool> pcie_relaxed_ordering("PCIE_RELAXED_ORDERING", "", false);
    const var<bool> enable_dmabuf("ENABLE_DMABUF", "", false);
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
        if (level_str == "NONE") {
          level = debug_level::NONE;
        } else if (level_str == "VERSION") {
          level = debug_level::VERSION;
        } else if (level_str == "WARN") {
          level = debug_level::WARN;
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
        case debug_level::INFO:
          return os << "INFO";
        case debug_level::TRACE:
          return os << "TRACE";
        }
      }
    }  // inline namespace _debug
  }  // namespace types
}  // namespace envvar
}  // namespace rocshmem
