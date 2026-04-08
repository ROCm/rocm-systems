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

#include "build_info.hpp"
#if defined(USE_GDA)
#include "gda/topology.hpp"
#endif // defined(USE_GDA)

#include <cstring>
#include <iostream>

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

int main(int argc, char **argv) {
  rocshmem::envvar::print_mode env_mode = rocshmem::envvar::print_mode::MODIFIED;

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

  rocshmem::print_build_info(std::cout);

  std::cout << "\n";
  rocshmem::envvar::print_envvars(env_mode, std::cout);

#if defined(USE_GDA)
  std::cout << "\n################################################################################\n";
  rocshmem::DisplayTopology(false);
  std::cout << "################################################################################\n";
#endif // defined(USE_GDA)

  return 0;
}
