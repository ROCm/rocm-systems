/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "memcpy_performance_common.hh"

#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

/**
 * @addtogroup memcpy memcpy
 * @{
 * @ingroup PerformanceTestMemory
 */

namespace {

constexpr unsigned char kPattern = 0x5a;
constexpr size_t kDepth = 4;

class Memcpy3DBatchAsync : public Benchmark<Memcpy3DBatchAsync> {
 public:
  void operator()(std::vector<hipMemcpy3DBatchOp>& operations, hipStream_t stream) {
    size_t fail_index;
    TIMED_SECTION_STREAM(kTimerTypeCpu, stream) {
      HIP_CHECK(
          hipMemcpy3DBatchAsync(operations.size(), operations.data(), &fail_index, 0, stream));
    }
  }
};

class Memcpy3DSequentialAsync : public Benchmark<Memcpy3DSequentialAsync> {
 public:
  void operator()(std::vector<hipMemcpy3DParms>& parameters, hipStream_t stream) {
    TIMED_SECTION_STREAM(kTimerTypeCpu, stream) {
      for (hipMemcpy3DParms& parameter : parameters) {
        HIP_CHECK(hipMemcpy3DAsync(&parameter, stream));
      }
    }
  }
};

void RunBenchmark(hipMemcpyKind kind, size_t total_size, size_t copy_count, int src_device = 0,
                  int dst_device = 0) {
  const size_t bytes_per_copy = total_size / copy_count;
  const size_t plane = bytes_per_copy / kDepth;
  size_t height = 1;
  while ((height * 2) * (height * 2) <= plane) {
    height *= 2;
  }
  const size_t width = plane / height;
  const size_t allocation_stride = width * height * kDepth;
  const size_t allocation_size = allocation_stride * copy_count;
  const hipExtent extent = make_hipExtent(width, height, kDepth);
  const bool source_on_host = kind == hipMemcpyHostToDevice;
  const bool destination_on_host = kind == hipMemcpyDeviceToHost;

  HIP_CHECK(hipSetDevice(src_device));
  LinearAllocGuard<unsigned char> src_allocation(
      source_on_host ? LinearAllocs::hipHostMalloc : LinearAllocs::hipMalloc, allocation_size);
  if (source_on_host) {
    std::memset(src_allocation.host_ptr(), kPattern, allocation_size);
  } else {
    HIP_CHECK(hipMemset(src_allocation.ptr(), kPattern, allocation_size));
  }

  HIP_CHECK(hipSetDevice(dst_device));
  LinearAllocGuard<unsigned char> dst_allocation(
      destination_on_host ? LinearAllocs::hipHostMalloc : LinearAllocs::hipMalloc, allocation_size);

  HIP_CHECK(hipSetDevice(src_device));

  std::vector<hipMemcpy3DBatchOp> operations(copy_count);
  std::vector<hipMemcpy3DParms> parameters(copy_count);
  for (size_t i = 0; i < copy_count; ++i) {
    unsigned char* src = src_allocation.ptr() + i * allocation_stride;
    unsigned char* dst = dst_allocation.ptr() + i * allocation_stride;

    hipMemcpy3DBatchOp& operation = operations[i];
    operation = {};
    operation.src.type = hipMemcpyOperandTypePointer;
    operation.src.op.ptr.ptr = src;
    operation.src.op.ptr.rowLength = width;
    operation.src.op.ptr.layerHeight = height;
    operation.src.op.ptr.locHint.type =
        source_on_host ? hipMemLocationTypeHost : hipMemLocationTypeDevice;
    operation.src.op.ptr.locHint.id = source_on_host ? 0 : src_device;
    operation.dst.type = hipMemcpyOperandTypePointer;
    operation.dst.op.ptr.ptr = dst;
    operation.dst.op.ptr.rowLength = width;
    operation.dst.op.ptr.layerHeight = height;
    operation.dst.op.ptr.locHint.type =
        destination_on_host ? hipMemLocationTypeHost : hipMemLocationTypeDevice;
    operation.dst.op.ptr.locHint.id = destination_on_host ? 0 : dst_device;
    operation.extent = extent;
    operation.srcAccessOrder = hipMemcpySrcAccessOrderStream;
    operation.flags = hipMemcpyFlagDefault;

    parameters[i] =
        CreateMemcpy3DParam(make_hipPitchedPtr(dst, width, width, height), make_hipPos(0, 0, 0),
                            make_hipPitchedPtr(src, width, width, height),
                            make_hipPos(0, 0, 0), extent, kind);
  }

  const StreamGuard stream_guard{Streams::created};
  const hipStream_t stream = stream_guard.stream();

  Memcpy3DSequentialAsync sequential_benchmark;
  sequential_benchmark.SetDisplayOutput(false);
  const auto sequential_stats = sequential_benchmark.Run(parameters, stream);
  const float sequential_mean = std::get<0>(sequential_stats);

  const std::string size_label = total_size % (1_MB) == 0
                                     ? std::to_string(total_size / (1_MB)) + " MB"
                                     : std::to_string(total_size / (1_KB)) + " KB";
  Memcpy3DBatchAsync batch_benchmark;
  batch_benchmark.AddSectionName(size_label);
  batch_benchmark.AddSectionName(std::to_string(copy_count) + " copies");
  batch_benchmark.AddSectionName(std::to_string(width) + " B x " + std::to_string(height) + " x " +
                                 std::to_string(kDepth));
  batch_benchmark.RegisterBandwidth(total_size);
  batch_benchmark.RegisterStatsSuffix([sequential_mean](float batch_mean) {
    std::ostringstream ratio;
    ratio << std::fixed << std::setprecision(2) << sequential_mean / batch_mean;
    return " Ratio " + ratio.str() + "x";
  });
  batch_benchmark.Run(operations, stream);
}

}  // namespace

HIP_TEST_CASE(Performance_hipMemcpy3DBatchAsync_H2D_Aligned) {
  const size_t total_size = GENERATE(16_KB, 128_KB, 512_KB, 1_MB, 4_MB);
  const size_t copy_count = GENERATE(16, 128, 1024);
  RunBenchmark(hipMemcpyHostToDevice, total_size, copy_count);
}

HIP_TEST_CASE(Performance_hipMemcpy3DBatchAsync_D2H_Aligned) {
  const size_t total_size = GENERATE(16_KB, 128_KB, 512_KB, 1_MB, 4_MB);
  const size_t copy_count = GENERATE(16, 128, 1024);
  RunBenchmark(hipMemcpyDeviceToHost, total_size, copy_count);
}

HIP_TEST_CASE(Performance_hipMemcpy3DBatchAsync_P2P_Aligned) {
  if (HipTest::getDeviceCount() < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }
  int can_access_peer = 0;
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, 0, 1));
  if (can_access_peer == 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kPeerAccessUnavailable);
  }
  HIP_CHECK(hipSetDevice(0));
  const hipError_t enable_status = hipDeviceEnablePeerAccess(1, 0);
  if (enable_status != hipSuccess && enable_status != hipErrorPeerAccessAlreadyEnabled) {
    HIP_CHECK(enable_status);
  }
  static_cast<void>(hipGetLastError());

  const size_t total_size = GENERATE(16_KB, 128_KB, 512_KB, 1_MB, 4_MB);
  const size_t copy_count = GENERATE(16, 128, 1024);
  RunBenchmark(hipMemcpyDeviceToDevice, total_size, copy_count, 0, 1);
}

/**
 * End doxygen group memcpy.
 * @}
 */
