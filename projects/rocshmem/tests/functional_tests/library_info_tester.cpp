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

#include "library_info_tester.hpp"

#include <rocshmem/rocshmem.hpp>
#include <cstring>

using namespace rocshmem;

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
LibraryInfoTester::LibraryInfoTester(TesterArguments args) : Tester(args) {
  _type = LibraryInfoTestType;
  _print_results = false;
}

LibraryInfoTester::~LibraryInfoTester() {}

void LibraryInfoTester::resetBuffers([[maybe_unused]] size_t size) {}

void LibraryInfoTester::launchKernel([[maybe_unused]] dim3 gridSize,
                                     [[maybe_unused]] dim3 blockSize,
                                     [[maybe_unused]] int loop,
                                     [[maybe_unused]] size_t size) {}

void LibraryInfoTester::verifyResults([[maybe_unused]] size_t size) {
  int major, minor;
  rocshmem_info_get_version(&major, &minor);
  if (major != ROCSHMEM_MAJOR_VERSION || minor != ROCSHMEM_MINOR_VERSION) {
    std::cerr << "FAIL: rocshmem_info_get_version returned "
              << major << "." << minor << ", expected "
              << ROCSHMEM_MAJOR_VERSION << "." << ROCSHMEM_MINOR_VERSION
              << "\n";
    abort();
  }

  char name[ROCSHMEM_MAX_NAME_LEN];
  rocshmem_info_get_name(name);
  if (std::strcmp(name, ROCSHMEM_VENDOR_STRING) != 0) {
    std::cerr << "FAIL: rocshmem_info_get_name returned \"" << name
              << "\", expected \"" << ROCSHMEM_VENDOR_STRING << "\"\n";
    abort();
  }

  int vmajor, vminor, vpatch;
  rocshmem_vendor_get_version_info(&vmajor, &vminor, &vpatch);
  if (vmajor != ROCSHMEM_VENDOR_MAJOR_VERSION ||
      vminor != ROCSHMEM_VENDOR_MINOR_VERSION ||
      vpatch != ROCSHMEM_VENDOR_PATCH_VERSION) {
    std::cerr << "FAIL: rocshmem_vendor_get_version_info returned "
              << vmajor << "." << vminor << "." << vpatch << ", expected "
              << ROCSHMEM_VENDOR_MAJOR_VERSION << "."
              << ROCSHMEM_VENDOR_MINOR_VERSION << "."
              << ROCSHMEM_VENDOR_PATCH_VERSION << "\n";
    abort();
  }

  if (rocshmem_my_pe() == 0) {
    std::cout << "PASS: library info APIs returned expected values\n"
              << "  OpenSHMEM spec version: " << major << "." << minor << "\n"
              << "  Vendor name: " << name << "\n"
              << "  Vendor version: " << vmajor << "." << vminor << "."
              << vpatch << "\n";
  }
}
