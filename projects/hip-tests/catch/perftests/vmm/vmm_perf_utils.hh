/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// Performance measurement utilities for VMM tests
#pragma once

#include <hip_test_common.hh>
#include <chrono>
#include <vector>
#include <string>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "vmm_test_utils.hh"

namespace vmm_perf {

using namespace vmm_utils;
using Microseconds = std::chrono::microseconds;

// Structure to store timing results
struct TimingResults {
  size_t size_gb;
  Microseconds reserve;
  Microseconds alloc;
  Microseconds map;
  Microseconds unmap;
  Microseconds release;
  Microseconds free;
  Microseconds h2d;
  Microseconds d2h;
  Microseconds hip_malloc;
  Microseconds hip_free;
  
  Microseconds GetTotalAllocTime() const { return reserve + alloc + map; }
  Microseconds GetTotalDeallocTime() const { return unmap + release + free; }
  
  double GetVMMOverheadPercent() const {
    if (hip_malloc.count() == 0) return 0.0;
    return ((GetTotalAllocTime().count() - hip_malloc.count()) * 100.0) / hip_malloc.count();
  }
};

// Simple timer class
class Timer {
public:
  Timer() : start_(std::chrono::high_resolution_clock::now()) {}
  void Start() { start_ = std::chrono::high_resolution_clock::now(); }
  Microseconds Stop() const {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<Microseconds>(end - start_);
  }
private:
  std::chrono::high_resolution_clock::time_point start_;
};

// Print results in tabular format
inline void PrintResultsTable(const std::vector<TimingResults>& results, 
                               const std::string& title = "VMM Performance Results") {
  if (results.empty()) return;

  // Print header
  std::cout << "\n" << std::string(100, '=') << std::endl;
  std::cout << title << " (microseconds)" << std::endl;
  std::cout << std::string(100, '=') << std::endl;
  
  // Print size header
  std::cout << std::left << std::setw(20) << "Operation";
  for (const auto& result : results) {
    std::cout << std::right << std::setw(12) << (std::to_string(result.size_gb) + " GB");
  }
  std::cout << std::endl;
  std::cout << std::string(100, '-') << std::endl;

  // Lambda to print a single row
  auto print_row = [&](const std::string& name, auto TimingResults::*member) {
    std::cout << std::left << std::setw(20) << name;
    for (const auto& result : results) {
      std::cout << std::right << std::setw(12) << (result.*member).count();
    }
    std::cout << std::endl;
  };

  // Print VMM operations
  std::cout << "\n--- VMM Operations ---" << std::endl;
  print_row("Reserve", &TimingResults::reserve);
  print_row("Alloc", &TimingResults::alloc);
  print_row("Map", &TimingResults::map);
  print_row("Unmap", &TimingResults::unmap);
  print_row("Release", &TimingResults::release);
  print_row("Free", &TimingResults::free);
  
  // Print memory copy operations
  std::cout << "\n--- Memory Copy ---" << std::endl;
  print_row("H2D Copy", &TimingResults::h2d);
  print_row("D2H Copy", &TimingResults::d2h);
  
  // Print legacy hipMalloc comparison
  std::cout << "\n--- Legacy hipMalloc ---" << std::endl;
  print_row("hipMalloc", &TimingResults::hip_malloc);
  print_row("hipFree", &TimingResults::hip_free);
  
  // Print total times for comparison
  std::cout << "\n--- Totals ---" << std::endl;
  std::cout << std::left << std::setw(20) << "VMM Total Alloc";
  for (const auto& result : results) {
    std::cout << std::right << std::setw(12) << result.GetTotalAllocTime().count();
  }
  std::cout << std::endl;
  
  std::cout << std::left << std::setw(20) << "VMM Total Dealloc";
  for (const auto& result : results) {
    std::cout << std::right << std::setw(12) << result.GetTotalDeallocTime().count();
  }
  std::cout << std::endl;
  
  std::cout << std::left << std::setw(20) << "VMM Overhead %";
  for (const auto& result : results) {
    std::cout << std::right << std::setw(12) << std::fixed << std::setprecision(1) 
              << result.GetVMMOverheadPercent();
  }
  std::cout << std::endl;
  
  std::cout << std::string(100, '=') << std::endl << std::endl;
}

// Validate memory and measure copy performance
inline bool ValidateUsingCopy(int deviceId, void* dev_ptr, size_t data_size,
                               Microseconds& h2d_elapsed, Microseconds& d2h_elapsed,
                               bool verify_data = true) {
  // Allocate host buffers
  std::vector<int> A_h(GetSizeN<int>(data_size));
  std::vector<int> B_h(GetSizeN<int>(data_size));

  // Initialize data
  for (size_t idx = 0; idx < A_h.size(); ++idx) {
    A_h[idx] = static_cast<int>(idx);
    B_h[idx] = 0;
  }

  HIP_CHECK(hipSetDevice(deviceId));
  
  // Measure H2D copy
  Timer timer;
  HIP_CHECK(hipMemcpy(dev_ptr, A_h.data(), data_size, hipMemcpyHostToDevice));
  h2d_elapsed = timer.Stop();

  // Measure D2H copy
  timer.Start();
  HIP_CHECK(hipMemcpy(B_h.data(), dev_ptr, data_size, hipMemcpyDeviceToHost));
  d2h_elapsed = timer.Stop();

  // Verify data if requested
  if (verify_data) {
    for (size_t idx = 0; idx < A_h.size(); ++idx) {
      if (A_h[idx] != B_h[idx]) {
        std::cerr << "Validation failed at index " << idx 
                  << ": expected " << A_h[idx] 
                  << ", got " << B_h[idx] << std::endl;
        return false;
      }
    }
  }

  return true;
}

// Warm up device for stable performance measurements
inline bool WarmupDevice(int deviceId, size_t granularity, size_t warmup_size = 64 * kMB) {
  std::cout << "Warming up device " << deviceId << "..." << std::endl;
  
  HIP_CHECK(hipSetDevice(deviceId));
  
  // Warmup 1: Complete VMM cycle
  void* dev_ptr = nullptr;
  
  // Reserve
  HIP_CHECK(hipMemAddressReserve(&dev_ptr, warmup_size, granularity, nullptr, 0));
  
  // Create physical memory
  auto prop = CreateDeviceMemProp(deviceId);
  hipMemGenericAllocationHandle_t handle;
  HIP_CHECK(hipMemCreate(&handle, warmup_size, &prop, 0));
  
  // Map
  HIP_CHECK(hipMemMap(dev_ptr, warmup_size, 0, handle, 0));
  
  // Set access
  auto accessDesc = CreateDeviceAccessDesc(deviceId);
  HIP_CHECK(hipMemSetAccess(dev_ptr, warmup_size, &accessDesc, 1));
  
  // Perform copy operations
  std::vector<int> warmup_data(warmup_size / sizeof(int), 42);
  HIP_CHECK(hipMemcpy(dev_ptr, warmup_data.data(), warmup_size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(warmup_data.data(), dev_ptr, warmup_size, hipMemcpyDeviceToHost));
  
  // Cleanup VMM
  HIP_CHECK(hipMemUnmap(dev_ptr, warmup_size));
  HIP_CHECK(hipMemRelease(handle));
  HIP_CHECK(hipMemAddressFree(dev_ptr, warmup_size));
  
  // Warmup 2: Legacy hipMalloc
  void* dev_ptr_legacy = nullptr;
  HIP_CHECK(hipMalloc(&dev_ptr_legacy, warmup_size));
  HIP_CHECK(hipMemcpy(dev_ptr_legacy, warmup_data.data(), warmup_size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(warmup_data.data(), dev_ptr_legacy, warmup_size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(dev_ptr_legacy));
  
  // Ensure all operations complete
  HIP_CHECK(hipDeviceSynchronize());
  
  std::cout << "Warmup complete!" << std::endl << std::endl;
  return true;
}

// Export results to CSV file
inline bool ExportResultsToCSV(const std::vector<TimingResults>& results,
                                const std::string& filename) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to open " << filename << " for writing" << std::endl;
    return false;
  }
  
  // Write header
  file << "Size_GB,Reserve,Alloc,Map,Unmap,Release,Free,H2D,D2H,hipMalloc,hipFree\n";
  
  // Write data
  for (const auto& result : results) {
    file << result.size_gb << ","
         << result.reserve.count() << ","
         << result.alloc.count() << ","
         << result.map.count() << ","
         << result.unmap.count() << ","
         << result.release.count() << ","
         << result.free.count() << ","
         << result.h2d.count() << ","
         << result.d2h.count() << ","
         << result.hip_malloc.count() << ","
         << result.hip_free.count() << "\n";
  }
  
  file.close();
  std::cout << "Results exported to " << filename << std::endl;
  return true;
}

}  // namespace vmm_perf

