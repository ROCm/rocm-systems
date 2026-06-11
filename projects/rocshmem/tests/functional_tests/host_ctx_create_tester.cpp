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

#include "host_ctx_create_tester.hpp"

#include <iostream>

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
HostCtxCreateTester::HostCtxCreateTester(TesterArguments args)
    : Tester(args) {
  _print_results = false;
}

HostCtxCreateTester::~HostCtxCreateTester() {}

void HostCtxCreateTester::resetBuffers([[maybe_unused]] size_t size) {}

void HostCtxCreateTester::preLaunchKernel() {
  int ret = rocshmem_ctx_create(0, &_ctx);
  if (ret != ROCSHMEM_SUCCESS) {
    std::cerr << "FAIL: rocshmem_ctx_create returned " << ret << "\n";
    exit(ret);
  }
}

void HostCtxCreateTester::launchKernel(
    [[maybe_unused]] dim3 gridSize, [[maybe_unused]] dim3 blockSize,
    [[maybe_unused]] int loop, [[maybe_unused]] size_t size) {}

void HostCtxCreateTester::postLaunchKernel() {
  rocshmem_ctx_destroy(_ctx);
}

void HostCtxCreateTester::verifyResults([[maybe_unused]] size_t size) {
  if (rocshmem_my_pe() == 0) {
    std::cout << "PASS: rocshmem_ctx_create, and rocshmem_ctx_destroy succeeded\n";
  }
}
