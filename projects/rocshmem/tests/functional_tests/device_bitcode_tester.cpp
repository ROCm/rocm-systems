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
#include <cinttypes>
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

void DeviceBitcodeTester::launch(const char* kernel, void** args,
                                 dim3 grid, dim3 block) {
  hipFunction_t fn;
  CHECK_HIP(hipModuleGetFunction(&fn, module, kernel));
  CHECK_HIP(hipModuleLaunchKernel(fn, grid.x, grid.y, grid.z,
                                  block.x, block.y, block.z,
                                  0, nullptr, args, nullptr));
  CHECK_HIP(hipDeviceSynchronize());
}

template <typename T>
void DeviceBitcodeTester::run_rma_test(const char* label, const char* kernel,
                                       int count, T scale, T offset) {
  T* sym_src = static_cast<T*>(rocshmem_malloc(count * sizeof(T)));
  T* sym_dst = static_cast<T*>(rocshmem_malloc(count * sizeof(T)));
  // Use symmetric heap for get result so GDA can write to it (only heap is
  // registered for RDMA); hipMalloc'd dest would never receive the get.
  T* sym_result = static_cast<T*>(rocshmem_malloc(count * sizeof(T)));
  for (int i = 0; i < count; i++) sym_result[i] = static_cast<T>(-1);

  for (int i = 0; i < count; i++) {
    sym_src[i] = static_cast<T>(my_pe) * scale + static_cast<T>(i) + offset;
    sym_dst[i] = static_cast<T>(-1);
  }
  rocshmem_barrier_all();

  void* kargs[] = {&sym_src, &sym_dst, &sym_result, &my_pe, &n_pes, &count};
  launch(kernel, kargs);
  rocshmem_barrier_all();

  int sender = (my_pe - 1 + n_pes) % n_pes;
  bool pass = true;
  for (int i = 0; i < count; i++) {
    T expected = static_cast<T>(sender) * scale + static_cast<T>(i) + offset;
    if (sym_result[i] != expected) {
      printf("[PE %d] %s: [%d] got=%d expect=%d (sender=PE %d) FAIL\n",
             my_pe, label, i, static_cast<int>(sym_result[i]),
             static_cast<int>(expected), sender);
      pass = false;
    }
  }
  if (pass)
    printf("[PE %d] %s: %d elements OK PASS\n", my_pe, label, count);
  if (!pass) all_pass = false;

  rocshmem_free(sym_result);
  rocshmem_free(sym_src);
  rocshmem_free(sym_dst);
}

template <typename T>
void DeviceBitcodeTester::run_scalar_put_test(const char* label,
                                              const char* kernel,
                                              T scale, T offset) {
  T* sym_buf = static_cast<T*>(rocshmem_malloc(sizeof(T)));
  *sym_buf = static_cast<T>(-1);
  rocshmem_barrier_all();

  void* kargs[] = {&sym_buf, &my_pe, &n_pes};
  launch(kernel, kargs);
  rocshmem_barrier_all();

  int sender = (my_pe - 1 + n_pes) % n_pes;
  T expected = static_cast<T>(sender) * scale + offset;
  bool pass = (*sym_buf == expected);
  printf("[PE %d] %s: got=%ld expect=%ld (from PE %d) %s\n",
         my_pe, label, (long)*sym_buf, (long)expected, sender,
         pass ? "PASS" : "FAIL");
  if (!pass) all_pass = false;

  rocshmem_free(sym_buf);
}

void DeviceBitcodeTester::execute() {
  rocshmem_barrier_all();

  if (!module) {
    if (my_pe == 0)
      printf("device_bitcode: SKIPPED (HSACO not built for this platform)\n");
    return;
  }

  if (my_pe == 0) printf("\n=== ROCshmem Device Bitcode Test ===\n");

  { // test_pe_info
    int* d_pe;
    int* d_npes;
    CHECK_HIP(hipMalloc(&d_pe, sizeof(int)));
    CHECK_HIP(hipMalloc(&d_npes, sizeof(int)));
    CHECK_HIP(hipMemset(d_pe, 0, sizeof(int)));
    CHECK_HIP(hipMemset(d_npes, 0, sizeof(int)));

    void* kargs[] = {&d_pe, &d_npes};
    launch("test_pe_info", kargs);

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

  run_scalar_put_test<int>("test_put", "test_put", 100, 42);

  rocshmem_barrier_all();

  run_rma_test<int>("test_putmem_getmem", "test_putmem_getmem", 4, 1000, 0);
  rocshmem_barrier_all();

  run_rma_test<int>("test_typed_int_put_get",
                    "test_typed_int_put_get", 4, 100, 0);
  rocshmem_barrier_all();

  run_rma_test<float>("test_typed_float_put_get",
                      "test_typed_float_put_get", 4, 10.5f, 0.0f);
  rocshmem_barrier_all();

  { // test_typed_atomic (64-bit atomics for GDA compatibility)
    int64_t* sym_counter = static_cast<int64_t*>(rocshmem_malloc(sizeof(int64_t)));
    *sym_counter = 0;
    int64_t* d_old;
    CHECK_HIP(hipMalloc(&d_old, 2 * sizeof(int64_t)));
    CHECK_HIP(hipMemset(d_old, 0, 2 * sizeof(int64_t)));
    rocshmem_barrier_all();

    void* kargs[] = {&sym_counter, &d_old, &my_pe, &n_pes};
    launch("test_typed_atomic", kargs);
    rocshmem_barrier_all();

    int64_t h_old[2];
    CHECK_HIP(hipMemcpy(h_old, d_old, sizeof(h_old), hipMemcpyDeviceToHost));

    bool pass = (h_old[0] == 0 && h_old[1] == 10);
    printf("[PE %d] test_typed_atomic: fetch_add old=%" PRId64 "(expect 0) cas_old=%" PRId64 "(expect 10) %s\n",
           my_pe, h_old[0], h_old[1], pass ? "PASS" : "FAIL");
    if (!pass) all_pass = false;

    CHECK_HIP(hipFree(d_old));
    rocshmem_free(sym_counter);
  }

  rocshmem_barrier_all();

  { // test_typed_put_signal
    constexpr int COUNT = 4;
    int* sym_src = static_cast<int*>(rocshmem_malloc(COUNT * sizeof(int)));
    int* sym_dst = static_cast<int*>(rocshmem_malloc(COUNT * sizeof(int)));
    uint64_t* sym_sig = static_cast<uint64_t*>(rocshmem_malloc(sizeof(uint64_t)));

    for (int i = 0; i < COUNT; i++) {
      sym_src[i] = my_pe * 100 + i;
      sym_dst[i] = -1;
    }
    *sym_sig = 0;
    rocshmem_barrier_all();

    int count = COUNT;
    void* kargs[] = {&sym_src, &sym_dst, &sym_sig, &my_pe, &n_pes, &count};
    launch("test_typed_put_signal", kargs);
    rocshmem_barrier_all();

    int sender = (my_pe - 1 + n_pes) % n_pes;
    bool pass = true;
    for (int i = 0; i < COUNT; i++) {
      int expected = sender * 100 + i;
      if (sym_dst[i] != expected) {
        printf("[PE %d] test_typed_put_signal: [%d] got=%d expect=%d FAIL\n",
               my_pe, i, sym_dst[i], expected);
        pass = false;
      }
    }
    if (pass)
      printf("[PE %d] test_typed_put_signal: %d elements OK PASS\n",
             my_pe, COUNT);
    if (!pass) all_pass = false;

    rocshmem_free(sym_src);
    rocshmem_free(sym_dst);
    rocshmem_free(sym_sig);
  }

  rocshmem_barrier_all();

  { // test_typed_wait_until
    int* sym_flag = static_cast<int*>(rocshmem_malloc(sizeof(int)));
    *sym_flag = 0;
    rocshmem_barrier_all();

    void* kargs[] = {&sym_flag, &my_pe, &n_pes};
    launch("test_typed_wait_until", kargs);
    rocshmem_barrier_all();

    int sender = (my_pe - 1 + n_pes) % n_pes;
    int expected = sender + 1;
    bool pass = (*sym_flag == expected);
    printf("[PE %d] test_typed_wait_until: flag=%d(expect %d from PE %d) %s\n",
           my_pe, *sym_flag, expected, sender, pass ? "PASS" : "FAIL");
    if (!pass) all_pass = false;

    rocshmem_free(sym_flag);
  }

  rocshmem_barrier_all();

  run_rma_test<int>("test_typed_int_put_get_wave",
                    "test_typed_int_put_get_wave", 4, 200, 0);
  rocshmem_barrier_all();

  { // test_typed_amo_extended (64-bit atomics for GDA compatibility)
    int64_t* sym_val = static_cast<int64_t*>(rocshmem_malloc(sizeof(int64_t)));
    *sym_val = 0;
    int64_t* d_results;
    CHECK_HIP(hipMalloc(&d_results, 3 * sizeof(int64_t)));
    CHECK_HIP(hipMemset(d_results, 0, 3 * sizeof(int64_t)));
    rocshmem_barrier_all();

    void* kargs[] = {&sym_val, &d_results, &my_pe};
    launch("test_typed_amo_extended", kargs);
    rocshmem_barrier_all();

    int64_t h[3];
    CHECK_HIP(hipMemcpy(h, d_results, sizeof(h), hipMemcpyDeviceToHost));

    bool pass = (h[0] == 42 && h[1] == 42 && h[2] == 99);
    printf("[PE %d] test_typed_amo_extended: fetch=%" PRId64 "(42) swap_old=%" PRId64 "(42) final=%" PRId64 "(99) %s\n",
           my_pe, h[0], h[1], h[2], pass ? "PASS" : "FAIL");
    if (!pass) all_pass = false;

    CHECK_HIP(hipFree(d_results));
    rocshmem_free(sym_val);
  }

  rocshmem_barrier_all();

  { // test_typed_amo_bitwise (64-bit atomics for GDA compatibility)
    uint64_t* sym_val = static_cast<uint64_t*>(
        rocshmem_malloc(sizeof(uint64_t)));
    *sym_val = 0;
    uint64_t* d_results;
    CHECK_HIP(hipMalloc(&d_results, 3 * sizeof(uint64_t)));
    CHECK_HIP(hipMemset(d_results, 0, 3 * sizeof(uint64_t)));
    rocshmem_barrier_all();

    void* kargs[] = {&sym_val, &d_results, &my_pe};
    launch("test_typed_amo_bitwise", kargs);
    rocshmem_barrier_all();

    uint64_t h[3];
    CHECK_HIP(hipMemcpy(h, d_results, sizeof(h), hipMemcpyDeviceToHost));

    bool pass = (h[0] == 0xFF && h[1] == 0x0F && h[2] == 0x03);
    printf("[PE %d] test_typed_amo_bitwise: fetch_and_old=0x%" PRIx64 "(0xff) after=0x%" PRIx64 "(0xf) final=0x%" PRIx64 "(0x3) %s\n",
           my_pe, h[0], h[1], h[2], pass ? "PASS" : "FAIL");
    if (!pass) all_pass = false;

    CHECK_HIP(hipFree(d_results));
    rocshmem_free(sym_val);
  }

  rocshmem_barrier_all();

  run_scalar_put_test<int64_t>("test_typed_int64_p", "test_typed_int64_p",
                              1000000LL, 12345LL);

  rocshmem_barrier_all();

  { // test_typed_wait_vector
    constexpr int COUNT = 4;
    int* sym_arr = static_cast<int*>(rocshmem_malloc(COUNT * sizeof(int)));
    for (int i = 0; i < COUNT; i++) sym_arr[i] = 0;
    int* d_idx;
    CHECK_HIP(hipMalloc(&d_idx, sizeof(int)));
    CHECK_HIP(hipMemset(d_idx, 0, sizeof(int)));
    rocshmem_barrier_all();

    int count = COUNT;
    void* kargs[] = {&sym_arr, &d_idx, &my_pe, &n_pes, &count};
    launch("test_typed_wait_vector", kargs);
    rocshmem_barrier_all();

    int h_idx = 0;
    CHECK_HIP(hipMemcpy(&h_idx, d_idx, sizeof(int), hipMemcpyDeviceToHost));

    int sender = (my_pe - 1 + n_pes) % n_pes;
    bool pass = (h_idx == 1);
    if (pass) {
      for (int i = 0; i < COUNT; i++) {
        int expected = (sender + 1) * 10 + i;
        if (sym_arr[i] != expected) {
          printf("[PE %d] test_typed_wait_vector: [%d] got=%d expect=%d FAIL\n",
                 my_pe, i, sym_arr[i], expected);
          pass = false;
        }
      }
    }
    printf("[PE %d] test_typed_wait_vector: completed=%d data_ok %s\n",
           my_pe, h_idx, pass ? "PASS" : "FAIL");
    if (!pass) all_pass = false;

    CHECK_HIP(hipFree(d_idx));
    rocshmem_free(sym_arr);
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
