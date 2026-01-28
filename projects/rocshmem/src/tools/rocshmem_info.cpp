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

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

#include <rocm-core/rocm_version.h>
#include <rocshmem/rocshmem.hpp>

#define NAME_COLUMN_WIDTH (28)
#define INFO_COLUMN_WIDTH (47)

#define PRINT_ENTRY(NAME, INFO)   \
  printf("# %-*s: %-*s#\n", NAME_COLUMN_WIDTH, NAME, INFO_COLUMN_WIDTH, INFO)

const std::string config_file_path = std::string(ROCSHMEM_INSTALL_PREFIX)
                                   + std::string("/include/rocshmem/rocshmem_config.h");

bool is_config_file_valid() {
  std::ifstream file;

  if (file.is_open()) {
    fprintf(stderr, "Could not open config file: %s\n", config_file_path.c_str());
    return false;
  }

  file.open(config_file_path);
  if (!file.is_open()) {
    fprintf(stderr, "Error opening config file: %s\n", config_file_path.c_str());
    return false;
  }

  file.close();

  return true;
}

void parse_config_file() {
  std::string line;
  std::ifstream file;

  const std::string define     = "#define ";
  const std::string undef_pre  = "/* #undef ";
  const std::string undef_post = " */";

  printf("#------------------------------------------------------------------------------#\n");
  printf("#                              Build Configuration                             #\n");
  printf("#------------------------------------------------------------------------------#\n");

  file.open(config_file_path);

  while (std::getline(file, line)) {
    if (line.find(undef_pre) != std::string::npos) {
      line.replace(line.find(undef_pre), undef_pre.length(), "");
      line.replace(line.find(undef_post), undef_post.length(), "");
      PRINT_ENTRY(line.c_str(), "OFF");
    }

    if (line.find(define) != std::string::npos) {
      line.replace(line.find(define), define.length(), "");
      PRINT_ENTRY(line.c_str(), "ON");
    }
  }

  file.close();
}

void print_arch_info() {
  hipDeviceProp_t prop;
  std::string compiled_arch;
  std::string system_arch;

  int n_compiled_arch = 1;
  bool supported_arch = false;
  std::istringstream compiled_arch_list(ROCSHMEM_OFFLOAD_TARGETS);

  CHECK_HIP(hipGetDeviceProperties(&prop, 0));

  PRINT_ENTRY("System Arch", prop.gcnArchName);

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

int main (int argc, char **argv) {

  printf("################################################################################\n");
  printf("#                                rocSHMEM Info                                 #\n");
  printf("################################################################################\n");

  PRINT_ENTRY("Version", rocshmem::VERSION);
  PRINT_ENTRY("Git Hash", ROCSHMEM_GIT_HASH);
  PRINT_ENTRY("Install Prefix", ROCSHMEM_INSTALL_PREFIX);

  print_arch_info();
  print_rocm_info();
  print_mpi_info();

  if (is_config_file_valid()) {
    parse_config_file();
  }

  printf("################################################################################\n");
  return 0;
}
