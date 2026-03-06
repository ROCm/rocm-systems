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

#include "device_bitcode_tester.hpp"
#include <rocshmem/rocshmem.hpp>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace rocshmem;

static std::vector<char> read_binary_file(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary | std::ios::ate);
  if (!ifs) {
    fprintf(stderr, "Cannot open: %s\n", path.c_str());
    rocshmem_global_exit(1);
  }
  auto size = ifs.tellg();
  ifs.seekg(0);
  std::vector<char> buf(static_cast<size_t>(size));
  ifs.read(buf.data(), size);
  return buf;
}

static std::string parent_dir(const std::string& path) {
  auto pos = path.rfind('/');
  return (pos != std::string::npos && pos > 0) ? path.substr(0, pos) : ".";
}

std::string DeviceBitcodeTester::resolve_hsaco_path() {
  hipDeviceProp_t props;
  CHECK_HIP(hipGetDeviceProperties(&props, device_id));
  std::string arch(props.gcnArchName);
  auto colon = arch.find(':');
  if (colon != std::string::npos) arch.resize(colon);

  std::string filename = "device_bitcode_tester_kernel_" + arch + ".hsaco";
  std::string exe_dir = parent_dir(args.executable_name);

  std::string build_path = exe_dir + "/" + filename;
  if (std::ifstream(build_path).good()) return build_path;

  std::string install_path = exe_dir + "/../share/rocshmem/" + filename;
  if (std::ifstream(install_path).good()) return install_path;

  return "";
}

DeviceBitcodeTester::DeviceBitcodeTester(TesterArguments args)
    : Tester(args) {
  my_pe = rocshmem_my_pe();
  n_pes = rocshmem_n_pes();

  std::string hsaco_path = resolve_hsaco_path();
  if (hsaco_path.empty()) {
    if (my_pe == 0)
      printf("device_bitcode: HSACO not found for this GPU; test will skip\n");
    return;
  }
  if (my_pe == 0) printf("device_bitcode: loading %s\n", hsaco_path.c_str());

  auto binary = read_binary_file(hsaco_path);
  CHECK_HIP(hipModuleLoadData(&module, binary.data()));

  int ret = rocshmem_hipmodule_init(module, nullptr);
  if (ret != 0) {
    fprintf(stderr, "[PE %d] rocshmem_hipmodule_init failed: %d\n", my_pe, ret);
    rocshmem_global_exit(1);
  }
}

DeviceBitcodeTester::~DeviceBitcodeTester() {
  if (module) {
    CHECK_HIP(hipModuleUnload(module));
  }
}

void DeviceBitcodeTester::execute() {
  rocshmem_barrier_all();

  if (!module) {
    if (my_pe == 0)
      printf("device_bitcode: SKIPPED (HSACO not built for this platform)\n");
    return;
  }

  int peer = (my_pe + 1) % n_pes;

  if (my_pe == 0) printf("\n=== ROCshmem Device Bitcode Test ===\n");

  // --- test_pe_info: verify rocshmem_my_pe / rocshmem_n_pes on device ---
  {
    int* d_pe;
    int* d_npes;
    CHECK_HIP(hipMalloc(&d_pe, sizeof(int)));
    CHECK_HIP(hipMalloc(&d_npes, sizeof(int)));
    CHECK_HIP(hipMemset(d_pe, 0, sizeof(int)));
    CHECK_HIP(hipMemset(d_npes, 0, sizeof(int)));

    hipFunction_t fn;
    CHECK_HIP(hipModuleGetFunction(&fn, module, "test_pe_info"));
    void* kargs[] = {&d_pe, &d_npes};
    CHECK_HIP(hipModuleLaunchKernel(fn, 1, 1, 1, 64, 1, 1, 0, nullptr,
                                    kargs, nullptr));
    CHECK_HIP(hipDeviceSynchronize());

    int h_pe = -1, h_npes = -1;
    CHECK_HIP(hipMemcpy(&h_pe, d_pe, sizeof(int), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(&h_npes, d_npes, sizeof(int), hipMemcpyDeviceToHost));

    bool pass = (h_pe == my_pe && h_npes == n_pes);
    printf("[PE %d] test_pe_info: my_pe=%d(%d) n_pes=%d(%d) %s\n",
           my_pe, h_pe, my_pe, h_npes, n_pes, pass ? "PASS" : "FAIL");
    if (!pass) all_pass = false;

    CHECK_HIP(hipFree(d_pe));
    CHECK_HIP(hipFree(d_npes));
  }

  rocshmem_barrier_all();

  // --- test_put: each PE puts my_pe*100+42 to peer, verify received value ---
  {
    int* sym = static_cast<int*>(rocshmem_malloc(sizeof(int)));
    *sym = -1;
    rocshmem_barrier_all();

    hipFunction_t fn;
    CHECK_HIP(hipModuleGetFunction(&fn, module, "test_put"));
    void* kargs[] = {&sym, &my_pe, &n_pes};
    CHECK_HIP(hipModuleLaunchKernel(fn, 1, 1, 1, 64, 1, 1, 0, nullptr,
                                    kargs, nullptr));
    CHECK_HIP(hipDeviceSynchronize());
    rocshmem_barrier_all();

    int sender = (my_pe - 1 + n_pes) % n_pes;
    int expected = sender * 100 + 42;
    int actual = *sym;
    bool pass = (actual == expected);
    printf("[PE %d] test_put: got=%d expect=%d (from PE %d) %s\n",
           my_pe, actual, expected, sender, pass ? "PASS" : "FAIL");
    if (!pass) all_pass = false;

    rocshmem_free(sym);
  }

  rocshmem_barrier_all();

  // --- test_putmem_getmem: bulk put to peer, verify received data ---
  {
    constexpr int COUNT = 4;
    int* sym_src = static_cast<int*>(rocshmem_malloc(COUNT * sizeof(int)));
    int* sym_dst = static_cast<int*>(rocshmem_malloc(COUNT * sizeof(int)));
    int* d_result;
    CHECK_HIP(hipMalloc(&d_result, COUNT * sizeof(int)));
    CHECK_HIP(hipMemset(d_result, 0, COUNT * sizeof(int)));

    for (int i = 0; i < COUNT; i++) {
      sym_src[i] = my_pe * 1000 + i;
      sym_dst[i] = -1;
    }
    rocshmem_barrier_all();

    hipFunction_t fn;
    CHECK_HIP(hipModuleGetFunction(&fn, module, "test_putmem_getmem"));
    int count = COUNT;
    void* kargs[] = {&sym_src, &sym_dst, &d_result, &my_pe, &n_pes, &count};
    CHECK_HIP(hipModuleLaunchKernel(fn, 1, 1, 1, 64, 1, 1, 0, nullptr,
                                    kargs, nullptr));
    CHECK_HIP(hipDeviceSynchronize());
    rocshmem_barrier_all();

    int h_result[COUNT];
    CHECK_HIP(hipMemcpy(h_result, d_result, sizeof(h_result),
                         hipMemcpyDeviceToHost));

    int sender = (my_pe - 1 + n_pes) % n_pes;
    bool pass = true;
    for (int i = 0; i < COUNT; i++) {
      int expected = sender * 1000 + i;
      if (h_result[i] != expected) {
        printf("[PE %d] test_putmem_getmem: [%d] got=%d expect=%d FAIL\n",
               my_pe, i, h_result[i], expected);
        pass = false;
      }
    }
    if (pass)
      printf("[PE %d] test_putmem_getmem: %d elements OK PASS\n",
             my_pe, COUNT);
    if (!pass) all_pass = false;

    CHECK_HIP(hipFree(d_result));
    rocshmem_free(sym_src);
    rocshmem_free(sym_dst);
  }

  rocshmem_barrier_all();

  if (my_pe == 0)
    printf("\n=== %s ===\n",
           all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

  if (!all_pass) rocshmem_global_exit(1);
}

void DeviceBitcodeTester::resetBuffers(size_t) {}

void DeviceBitcodeTester::launchKernel(dim3, dim3, int, size_t) {}

void DeviceBitcodeTester::verifyResults(size_t) {}
