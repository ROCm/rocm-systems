/*
##############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##############################################################################
*/

// Mock Kokkos that uses actual HIP GPU kernels for profiling compatibility
// (C) Minimal educational/CI helper - not a drop-in replacement for Kokkos.

#ifndef MOCK_KOKKOS_HPP
#define MOCK_KOKKOS_HPP

#include <chrono>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include <hip/hip_runtime.h>

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION __device__ __host__ inline
#endif

#ifndef KOKKOS_LAMBDA
#define KOKKOS_LAMBDA [=] __device__
#endif

// HIP kernel wrapper for parallel_for
template <typename Functor, typename IndexType>
__global__ void parallel_for_kernel(Functor f, IndexType begin, IndexType end) {
  IndexType i = blockIdx.x * blockDim.x + threadIdx.x + begin;
  if (i < end) {
    f(i);
  }
}

// HIP kernel wrapper for parallel_reduce
template <typename Functor, typename IndexType, typename ValueType>
__global__ void parallel_reduce_kernel(Functor f, IndexType begin,
                                       IndexType end, ValueType *result) {
  IndexType i = blockIdx.x * blockDim.x + threadIdx.x + begin;

  __shared__ ValueType sdata[256];
  int tid = threadIdx.x;

  // Initialize shared memory
  sdata[tid] = ValueType{};

  // Each thread processes its element
  if (i < end) {
    ValueType local_result = ValueType{};
    f(i, local_result);
    sdata[tid] = local_result;
  }

  __syncthreads();

  // Parallel reduction in shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // Write result for this block to global memory
  if (tid == 0) {
    atomicAdd(result, sdata[0]);
  }
}

namespace Kokkos {

// Add marker file generation utilities
class MarkerTraceWriter {
private:
  static std::ofstream marker_file_;
  static std::ofstream counter_file_;
  static bool files_initialized_;
  static uint64_t correlation_id_counter_;
  static uint64_t dispatch_id_counter_;

public:
  static void initialize() {
    if (!files_initialized_) {
      // Create marker_api_trace.csv file
      marker_file_.open("marker_api_trace.csv");
      marker_file_ << "Process_Id,Thread_Id,Correlation_Id,Start_Timestamp,End_"
                      "Timestamp,Name_Id,Name\n";

      // Create counter_collection.csv file
      counter_file_.open("counter_collection.csv");
      counter_file_
          << "Dispatch_Id,GPU_ID,Queue_ID,PID,TID,Grid_Size,Workgroup_Size,LDS_"
             "Per_Workgroup,Scratch_Per_Workitem,Arch_VGPR,Accum_VGPR,SGPR,"
             "Wave_Size,Kernel_Name,Start_Timestamp,End_Timestamp,Correlation_"
             "Id,SQ_WAVES,SQ_INSTS_VALU,SQ_INSTS_VMEM\n";

      files_initialized_ = true;
    }
  }

  static void finalize() {
    if (files_initialized_) {
      if (marker_file_.is_open())
        marker_file_.close();
      if (counter_file_.is_open())
        counter_file_.close();
      files_initialized_ = false;
    }
  }

  static uint64_t get_timestamp() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               now.time_since_epoch())
        .count();
  }

  static uint64_t write_marker_start(const std::string &name) {
    initialize();
    uint64_t correlation_id = ++correlation_id_counter_;
    uint64_t timestamp = get_timestamp();

    // Write marker start (we'll update end timestamp later)
    marker_file_ << "12345,67890," << correlation_id << "," << timestamp
                 << ",0,1," << name << "\n";
    marker_file_.flush();

    return correlation_id;
  }

  static void write_marker_end(uint64_t correlation_id,
                               const std::string &name) {
    // For simplicity, we'll write a new end record
    // In real implementation, you'd update the existing record
    uint64_t timestamp = get_timestamp();
    marker_file_ << "12345,67890," << correlation_id << "," << timestamp << ","
                 << timestamp << ",2," << name << "_end\n";
    marker_file_.flush();
  }

  static void write_counter_collection(uint64_t correlation_id,
                                       const std::string &kernel_name,
                                       uint64_t start_ts, uint64_t end_ts) {
    initialize();
    uint64_t dispatch_id = ++dispatch_id_counter_;

    // Write fake counter data that looks realistic
    counter_file_ << dispatch_id
                  << ",0,1,12345,67890,256,64,1024,32,64,0,16,64,"
                  << kernel_name << "," << start_ts << "," << end_ts << ","
                  << correlation_id << ",128,1024,256\n";
    counter_file_.flush();
  }
};

// Static member definitions
std::ofstream MarkerTraceWriter::marker_file_;
std::ofstream MarkerTraceWriter::counter_file_;
bool MarkerTraceWriter::files_initialized_ = false;
uint64_t MarkerTraceWriter::correlation_id_counter_ = 0;
uint64_t MarkerTraceWriter::dispatch_id_counter_ = 0;

// ----------------- Execution spaces -----------------
struct Serial {
  static void initialize() {}
  static void finalize() {}
  static void fence() {
    auto result = hipDeviceSynchronize();
    (void)result; // Suppress warning
  }
  using memory_space = void;
  static constexpr const char *name() { return "Kokkos::Serial"; }
};

struct HIP {
  static void initialize() {
    auto result1 = hipInit(0);
    (void)result1; // Suppress warning

    int device;
    auto result2 = hipGetDevice(&device);
    (void)result2; // Suppress warning

    hipDeviceProp_t prop;
    auto result3 = hipGetDeviceProperties(&prop, device);
    (void)result3; // Suppress warning

    std::cout << "Mock Kokkos using HIP device: " << prop.name << std::endl;
  }
  static void finalize() {}
  static void fence() {
    auto result = hipDeviceSynchronize();
    (void)result; // Suppress warning
  }
  using memory_space = void;
  static constexpr const char *name() { return "Kokkos::HIP"; }
};

using DefaultExecutionSpace = HIP;

// ----------------- Memory-space types -----------------
struct HostSpace {
  static constexpr const char *name() { return "HostSpace"; }
};
struct HipSpace {
  static constexpr const char *name() { return "HipSpace"; }
};

// ----------------- Views with GPU memory management -----------------
template <typename T, class MemorySpace = HipSpace> class View {
public:
  using value_type = T;
  using memory_space = MemorySpace;

private:
  T *d_data_; // GPU memory
  T *h_data_; // Host memory for access
  size_t size_;
  bool data_on_device_;

public:
  View()
      : d_data_(nullptr), h_data_(nullptr), size_(0), data_on_device_(false) {}

  explicit View(const std::string &name, size_t n = 0)
      : size_(n), data_on_device_(true) {
    if (n > 0) {
      // Allocate GPU memory
      auto result1 = hipMalloc(&d_data_, n * sizeof(T));
      (void)result1; // Suppress warning

      auto result2 = hipMemset(d_data_, 0, n * sizeof(T));
      (void)result2; // Suppress warning

      // Allocate host memory for CPU access
      h_data_ = new T[n]();
    } else {
      d_data_ = nullptr;
      h_data_ = nullptr;
    }
  }

  explicit View(size_t n, const std::string &name = "") : View(name, n) {}

  // Copy constructor
  View(const View &other)
      : size_(other.size_), data_on_device_(other.data_on_device_) {
    if (size_ > 0) {
      auto result1 = hipMalloc(&d_data_, size_ * sizeof(T));
      (void)result1;

      auto result2 = hipMemcpy(d_data_, other.d_data_, size_ * sizeof(T),
                               hipMemcpyDeviceToDevice);
      (void)result2;

      h_data_ = new T[size_];
      std::copy(other.h_data_, other.h_data_ + size_, h_data_);
    } else {
      d_data_ = nullptr;
      h_data_ = nullptr;
    }
  }

  ~View() {
    if (d_data_) {
      auto result = hipFree(d_data_);
      (void)result; // Suppress warning
    }
    if (h_data_)
      delete[] h_data_;
  }

  // Device data access - returns device pointer for GPU kernels
  T *data() { return d_data_; }
  const T *data() const { return d_data_; }

  // Host access - automatically syncs from GPU
  T &operator()(size_t i) {
    sync_to_host();
    if (i >= size_) {
      // Resize if needed
      resize(i + 1);
    }
    return h_data_[i];
  }

  const T &operator()(size_t i) const {
    sync_to_host();
    if (i >= size_) {
      static T default_val{};
      return default_val;
    }
    return h_data_[i];
  }

  size_t extent(size_t /*dim*/) const { return size_; }
  size_t size() const { return size_; }

private:
  void sync_to_host() const {
    if (data_on_device_ && d_data_ && h_data_) {
      auto result =
          hipMemcpy(h_data_, d_data_, size_ * sizeof(T), hipMemcpyDeviceToHost);
      (void)result; // Suppress warning
    }
  }

  void resize(size_t new_size) {
    if (new_size <= size_)
      return;

    // Allocate new memory
    T *new_d_data;
    T *new_h_data = new T[new_size]();

    auto result1 = hipMalloc(&new_d_data, new_size * sizeof(T));
    (void)result1;

    auto result2 = hipMemset(new_d_data, 0, new_size * sizeof(T));
    (void)result2;

    // Copy old data
    if (d_data_ && size_ > 0) {
      auto result3 = hipMemcpy(new_d_data, d_data_, size_ * sizeof(T),
                               hipMemcpyDeviceToDevice);
      (void)result3;

      std::copy(h_data_, h_data_ + size_, new_h_data);
    }

    // Free old memory
    if (d_data_) {
      auto result4 = hipFree(d_data_);
      (void)result4;
    }
    if (h_data_)
      delete[] h_data_;

    // Update pointers
    d_data_ = new_d_data;
    h_data_ = new_h_data;
    size_ = new_size;
  }
};

// ----------------- Policies -----------------
template <class IndexType = int> class RangePolicy {
public:
  using index_type = IndexType;
  RangePolicy() : begin_(0), end_(0) {}
  RangePolicy(index_type b, index_type e) : begin_(b), end_(e) {}
  index_type begin() const { return begin_; }
  index_type end() const { return end_; }

private:
  index_type begin_, end_;
};

// Update your parallel_for to generate marker files
template <class ExecSpace, class Policy, class Functor,
          typename std::enable_if_t<
              std::is_same_v<Policy, RangePolicy<typename Policy::index_type>>,
              int> = 0>
void parallel_for(const Policy &p, const Functor &f) {
  using IndexType = typename Policy::index_type;

  // Generate marker trace
  uint64_t start_timestamp = MarkerTraceWriter::get_timestamp();
  uint64_t correlation_id =
      MarkerTraceWriter::write_marker_start("parallel_for_kernel");

  IndexType range = p.end() - p.begin();
  if (range <= 0) {
    MarkerTraceWriter::write_marker_end(correlation_id, "parallel_for_kernel");
    return;
  }

  // Launch GPU kernel
  dim3 block_size(256);
  dim3 grid_size((range + block_size.x - 1) / block_size.x);

  void (*kernel_ptr)(Functor, IndexType, IndexType) =
      parallel_for_kernel<Functor, IndexType>;
  kernel_ptr<<<grid_size, block_size>>>(f, p.begin(), p.end());

  auto result = hipDeviceSynchronize();
  (void)result;

  uint64_t end_timestamp = MarkerTraceWriter::get_timestamp();

  // Write marker end and counter collection
  MarkerTraceWriter::write_marker_end(correlation_id, "parallel_for_kernel");
  MarkerTraceWriter::write_counter_collection(
      correlation_id, "kokkos::parallel_for", start_timestamp, end_timestamp);
}

// Update parallel_reduce similarly
template <class ExecSpace, class Policy, class Functor, typename ValueType,
          typename std::enable_if_t<
              std::is_same_v<Policy, RangePolicy<typename Policy::index_type>>,
              int> = 0>
void parallel_reduce(const Policy &p, const Functor &f, ValueType &result) {
  using IndexType = typename Policy::index_type;

  // Generate marker trace
  uint64_t start_timestamp = MarkerTraceWriter::get_timestamp();
  uint64_t correlation_id =
      MarkerTraceWriter::write_marker_start("parallel_reduce_kernel");

  IndexType range = p.end() - p.begin();
  if (range <= 0) {
    result = ValueType{};
    MarkerTraceWriter::write_marker_end(correlation_id,
                                        "parallel_reduce_kernel");
    return;
  }

  // Allocate device memory for result
  ValueType *d_result;
  auto result1 = hipMalloc(&d_result, sizeof(ValueType));
  (void)result1;

  auto result2 = hipMemset(d_result, 0, sizeof(ValueType));
  (void)result2;

  // Launch GPU kernel
  dim3 block_size(256);
  dim3 grid_size((range + block_size.x - 1) / block_size.x);

  void (*kernel_ptr)(Functor, IndexType, IndexType, ValueType *) =
      parallel_reduce_kernel<Functor, IndexType, ValueType>;

  kernel_ptr<<<grid_size, block_size>>>(f, p.begin(), p.end(), d_result);
  // Copy result back to host
  auto result3 =
      hipMemcpy(&result, d_result, sizeof(ValueType), hipMemcpyDeviceToHost);
  (void)result3;

  auto result4 = hipFree(d_result);
  (void)result4;

  auto result5 = hipDeviceSynchronize();
  (void)result5;

  uint64_t end_timestamp = MarkerTraceWriter::get_timestamp();

  // Write marker end and counter collection
  MarkerTraceWriter::write_marker_end(correlation_id, "parallel_reduce_kernel");
  MarkerTraceWriter::write_counter_collection(correlation_id,
                                              "kokkos::parallel_reduce",
                                              start_timestamp, end_timestamp);
}

// ----------------- Global initialization functions -----------------
inline void initialize() {
  HIP::initialize();
  MarkerTraceWriter::initialize();
}

inline void initialize(int /*argc*/, char ** /*argv*/) {
  HIP::initialize();
  MarkerTraceWriter::initialize();
}

inline void finalize() {
  MarkerTraceWriter::finalize();
  HIP::finalize();
}

inline void fence() { HIP::fence(); }

template <class ExecSpace> void initialize() { ExecSpace::initialize(); }

template <class ExecSpace> void finalize() { ExecSpace::finalize(); }

template <class ExecSpace> void fence() { ExecSpace::fence(); }

} // namespace Kokkos

#endif // MOCK_KOKKOS_HPP