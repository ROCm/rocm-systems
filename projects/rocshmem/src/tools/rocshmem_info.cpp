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

#include "util.hpp"
#include "envvar.hpp"

#include "rocshmem_config_embedded.hpp"

#include <stdio.h>
#include <cstring>
#include <iostream>
#include <string>
#include <sstream>

#include <rocm-core/rocm_version.h>
#include <rocshmem/rocshmem.hpp>
#if defined(USE_GDA)
#include "gda/topology.hpp"
#endif // defined(USE_GDA)

#define NAME_COLUMN_WIDTH (28)
#define INFO_COLUMN_WIDTH (47)

#define PRINT_ENTRY(NAME, INFO)   \
  printf("# %-*s: %-*s#\n", NAME_COLUMN_WIDTH, NAME, INFO_COLUMN_WIDTH, INFO)

void parse_config_file() {
  std::string line;
  std::istringstream stream(rocshmem_config_h_content);

  const std::string define     = "#define ";
  const std::string undef_pre  = "/* #undef ";
  const std::string undef_post = " */";

  printf("#------------------------------------------------------------------------------#\n");
  printf("#                              Build Configuration                             #\n");
  printf("#------------------------------------------------------------------------------#\n");

  while (std::getline(stream, line)) {
    if (line.find(undef_pre) != std::string::npos) {
      line.replace(line.find(undef_pre), undef_pre.length(), "");
      line.replace(line.find(undef_post), undef_post.length(), "");
      PRINT_ENTRY(line.c_str(), "OFF");
      continue;
    }

    size_t def_pos = line.find(define);
    if (def_pos != std::string::npos) {
      std::string rest = line.substr(def_pos + define.length());
      size_t space = rest.find(' ');
      if (space != std::string::npos) {
        // String-valued define: #define NAME "value"
        std::string name  = rest.substr(0, space);
        std::string value = rest.substr(space + 1);
        if (value.size() >= 2 && value.front() == '"' && value.back() == '"')
          value = value.substr(1, value.size() - 2);
        PRINT_ENTRY(name.c_str(), value.c_str());
      } else {
        // Boolean define: #define NAME
        PRINT_ENTRY(rest.c_str(), "ON");
      }
    }
  }
}

void print_arch_info() {
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
      PRINT_ENTRY("Compiled Arch(s)",  compiled_arch.c_str());
    }
    else {
      PRINT_ENTRY(" ",  compiled_arch.c_str());
    }

    if (compiled_arch.find(system_arch) != std::string::npos) {
      supported_arch = true;
    }

    n_compiled_arch++;
  }

  PRINT_ENTRY("System Arch", prop.gcnArchName);

  PRINT_ENTRY("Supported System Arch", supported_arch ? "Yes" : "No");
}

void print_mpi_info() {
#ifdef OMPI_MAJOR_VERSION
  char mpi_version[8];
  snprintf(mpi_version, sizeof(mpi_version), "%d.%d.%d",
           OMPI_MAJOR_VERSION, OMPI_MINOR_VERSION, OMPI_RELEASE_VERSION);
  PRINT_ENTRY("Open MPI", mpi_version);
#else
  PRINT_ENTRY("MPI ", "Unsupported MPI Library");
#endif
}

void print_rocm_info() {
  char rocm_version[32];
  snprintf(rocm_version, sizeof(rocm_version), "%d.%d.%d",
           ROCM_VERSION_MAJOR, ROCM_VERSION_MINOR, ROCM_VERSION_PATCH);
  PRINT_ENTRY("ROCm", rocm_version);
}

void print_usage(const char* progname) {
  std::cout << "Usage: " << progname << " [OPTIONS]\n\n";
  std::cout << "Display rocSHMEM build information and environment variables.\n\n";
  std::cout << "Options:\n";
  std::cout << "  -h, --help       Show this help message\n";
  std::cout << "  --env:all        Print all environment variables (name=value)\n";
  std::cout << "  --env:full       Print all environment variables with full documentation\n";
  std::cout << "\n";
  std::cout << "Default mode: Display build information and modified env vars\n";
  std::cout << "\n";
  std::cout << "Examples:\n";
  std::cout << "  " << progname << " --env:all          # Show build info + all env vars\n";
  std::cout << "  " << progname << " --env:full         # Show build info + env vars with docs\n";
}

int main (int argc, char **argv) {
  rocshmem::envvar::print_mode env_mode = rocshmem::envvar::print_mode::MODIFIED;

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
      print_usage(argv[0]);
      return 0;
    } else if (std::strcmp(argv[i], "--env:all") == 0) {
      env_mode = rocshmem::envvar::print_mode::ALL_VALUES;
    } else if (std::strcmp(argv[i], "--env:full") == 0) {
      env_mode = rocshmem::envvar::print_mode::FULL_DOCUMENTATION;
    } else {
      std::cerr << "Error: Unknown option: " << argv[i] << "\n";
      print_usage(argv[0]);
      return 1;
    }
  }

  printf("################################################################################\n");
  printf("#                                rocSHMEM Info                                 #\n");
  printf("################################################################################\n");

  PRINT_ENTRY("Version", rocshmem::VERSION);
  PRINT_ENTRY("Git Hash", ROCSHMEM_GIT_HASH);
  PRINT_ENTRY("Install Prefix", ROCSHMEM_INSTALL_PREFIX);

  print_arch_info();
  print_rocm_info();
  print_mpi_info();

  parse_config_file();

  printf("################################################################################\n");
  std::cout << "\n";
  rocshmem::envvar::print_envvars(env_mode, std::cout);
  printf("################################################################################\n");

#if defined(USE_GDA)
  printf("\n################################################################################");
  rocshmem::DisplayTopology(false);
  printf("################################################################################\n");
#endif //defined(USE_GDA)

  return 0;
}
