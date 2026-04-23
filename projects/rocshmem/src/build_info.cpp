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

#include "build_info.hpp"
#include "rocshmem/rocshmem.hpp"
#include "rocshmem_config_embedded.hpp"
#include "util.hpp"

#include <rocm-core/rocm_version.h>

#include <cstring>
#include <iomanip>
#include <set>
#include <sstream>
#include <string>

namespace rocshmem {

static constexpr int NAME_COL = 28;
static constexpr int INFO_COL = 47;

template <typename T>
static void print_entry(std::ostream& os, const char* name, const T& info) {
  os << "# " << std::left << std::setw(NAME_COL) << name
     << ": " << std::setw(INFO_COL) << info << "#\n";
}

static void print_arch_info(std::ostream& os) {
  hipDeviceProp_t prop;
  std::string compiled_arch;
  std::string system_arch;

  int n_compiled_arch = 1;
  bool supported_arch = false;
  std::istringstream compiled_arch_list(ROCSHMEM_OFFLOAD_TARGETS);

  CHECK_HIP(hipGetDeviceProperties(&prop, 0));

  system_arch = std::string(prop.gcnArchName, strcspn(prop.gcnArchName, ":"));

  while (compiled_arch_list >> compiled_arch) {
    if (1 == n_compiled_arch) {
      print_entry(os, "Compiled Arch(s)", compiled_arch.c_str());
    } else {
      print_entry(os, " ", compiled_arch.c_str());
    }
    if (compiled_arch.find(system_arch) != std::string::npos)
      supported_arch = true;
    n_compiled_arch++;
  }

  print_entry(os, "System Arch", prop.gcnArchName);
  print_entry(os, "System Arch is supported", supported_arch ? "Yes" : "No");
}

static void print_rocm_info(std::ostream& os) {
  std::string rocm_version = std::to_string(ROCM_VERSION_MAJOR) + "."
                           + std::to_string(ROCM_VERSION_MINOR) + "."
                           + std::to_string(ROCM_VERSION_PATCH);
  print_entry(os, "ROCm", rocm_version.c_str());
}

static void print_mpi_info(std::ostream& os) {
#if defined(OMPI_MAJOR_VERSION)
  std::string mpi_version = "Open MPI "
                          + std::to_string(OMPI_MAJOR_VERSION) + "."
                          + std::to_string(OMPI_MINOR_VERSION) + "."
                          + std::to_string(OMPI_RELEASE_VERSION);
  print_entry(os, "External MPI", mpi_version.c_str());
#elif defined(MPICH_VERSION)
  print_entry(os, "External MPI", "MPICH " MPICH_VERSION);
#elif defined(MPI_VERSION)
  std::string mpi_version = "MPI (unknown vendor) "
                          + std::to_string(MPI_VERSION) + "."
                          + std::to_string(MPI_SUBVERSION);
  print_entry(os, "External MPI", mpi_version.c_str());
#else
  print_entry(os, "External MPI", "No");
#endif
}

static void parse_config(std::ostream& os) {
  const std::string define     = "#define ";
  const std::string undef_pre  = "/* #undef ";
  const std::string undef_post = " */";

  // Already printed at the top of the output; suppress duplicates here.
  const std::set<std::string> skip = {
    "ROCSHMEM_VENDOR_STRING",
    "ROCSHMEM_VENDOR_MAJOR_VERSION",
    "ROCSHMEM_VENDOR_MINOR_VERSION",
    "ROCSHMEM_VENDOR_PATCH_VERSION",
    "ROCSHMEM_VERSION",
    "ROCSHMEM_GIT_HASH",
    "ROCSHMEM_INSTALL_PREFIX",
    "ROCSHMEM_OFFLOAD_TARGETS",
    "ROCSHMEM_BUILD_TYPE",
  };

  os << "#------------------------------------------------------------------------------#\n";
  os << "#                              Build Configuration                             #\n";
  os << "#------------------------------------------------------------------------------#\n";

  std::istringstream stream(rocshmem_config_h_content);
  std::string line;
  while (std::getline(stream, line)) {
    if (line.find(undef_pre) != std::string::npos) {
      line.replace(line.find(undef_pre), undef_pre.length(), "");
      line.replace(line.find(undef_post), undef_post.length(), "");
      print_entry(os, line.c_str(), "OFF");
      continue;
    }

    size_t def_pos = line.find(define);
    if (def_pos != std::string::npos) {
      std::string rest = line.substr(def_pos + define.length());
      size_t space = rest.find(' ');
      if (space != std::string::npos) {
        std::string name  = rest.substr(0, space);
        if (skip.count(name)) continue;
        std::string value = rest.substr(space + 1);
        if (value.size() >= 2 && value.front() == '"' && value.back() == '"')
          value = value.substr(1, value.size() - 2);
        print_entry(os, name.c_str(), value.c_str());
      } else {
        if (skip.count(rest)) continue;
        print_entry(os, rest.c_str(), "ON");
      }
    }
  }
}

void print_build_info(std::ostream& os) {
  os << "################################################################################\n";
  os << "#                                rocSHMEM Info                                 #\n";
  os << "################################################################################\n";

  print_entry(os, "Version", rocshmem::VERSION);
  print_entry(os, "Vendor String", ROCSHMEM_VENDOR_STRING);
  print_entry(os, "Vendor Major Version", ROCSHMEM_VENDOR_MAJOR_VERSION);
  print_entry(os, "Vendor Minor Version", ROCSHMEM_VENDOR_MINOR_VERSION);
  print_entry(os, "Vendor Patch Version", ROCSHMEM_VENDOR_PATCH_VERSION);
  print_entry(os, "Git Hash", ROCSHMEM_GIT_HASH);
  print_entry(os, "Build Type", ROCSHMEM_BUILD_TYPE);
  print_entry(os, "Install Prefix", ROCSHMEM_INSTALL_PREFIX);

  print_arch_info(os);
  print_rocm_info(os);
  print_mpi_info(os);
  parse_config(os);

  os << "################################################################################\n";
}

}  // namespace rocshmem
