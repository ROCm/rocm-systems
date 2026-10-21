/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Kernel-submitted copy bandwidth through the HIP copy-engine API: warps of a single kernel push
// copies from a buffer on --srcGpu to buffers on one or more peer GPUs and wait for them.

#include <iomanip>

#include "copy_engine_bench_common.hpp"

using namespace copy_engine_bench;

namespace {

struct Params {
  int src_gpu = 0;
  size_t min_copy_size = 1024;
  size_t max_copy_size = 1024;
  size_t num_copy_commands = 1;
  bool skip_verification = false;
  size_t warmup = 3;
  size_t iterations = 50;
  size_t num_destinations = 1;
  size_t warps_per_wg = 1;
  size_t wgs_per_destination = 1;
  size_t num_queues = 1;
  std::string output_file = "copy_engine_bw.csv";
  std::string signal_memory = "device";
  bool verbose = false;
};

// Lane 0 of every warp copies its own slice of the source to one destination: the warp's
// destination is its work-group's index modulo the number of destinations, and the work-groups of
// a destination take turns over its engines, so |handles| holds each destination's engines in turn.
__global__ void CopyBandwidthKernel(HandleSet handles, const char* src, char* const* dst_bufs,
                                    uint64_t copy_size, uint32_t num_commands,
                                    uint32_t num_destinations, uint32_t num_queues,
                                    uint64_t* signals, uint64_t epoch, int64_t* start_clocks,
                                    int64_t* end_clocks, hipError_t* statuses) {
  if (threadIdx.x % warpSize != 0) return;
  const uint32_t warp = threadIdx.x / warpSize;
  const uint32_t warps_per_block = blockDim.x / warpSize;
  const uint32_t dst_idx = blockIdx.x % num_destinations;
  const uint32_t block_in_dst = blockIdx.x / num_destinations;
  const uint32_t queue = block_in_dst % num_queues;
  const uint32_t slot = blockIdx.x * warps_per_block + warp;
  const uint64_t offset =
      (uint64_t{block_in_dst} * warps_per_block + warp) * copy_size * num_commands;

  start_clocks[slot] = wall_clock64();
  statuses[slot] =
      SubmitBatchAndWait(&handles.handles[dst_idx * num_queues + queue], dst_bufs[dst_idx] + offset,
                         src + offset, copy_size, num_commands, signals + slot, epoch, nullptr);
  end_clocks[slot] = wall_clock64();
}

// Mean over destinations of the span from the first start to the last end among the warps that
// served that destination, in microseconds.
double MeanSpanUs(const int64_t* start, const int64_t* end, size_t num_destinations,
                  size_t warps_per_block, size_t total_warps, double ticks_per_us) {
  std::vector<int64_t> first(num_destinations, INT64_MAX);
  std::vector<int64_t> last(num_destinations, INT64_MIN);
  for (size_t slot = 0; slot < total_warps; ++slot) {
    const size_t dst_idx = (slot / warps_per_block) % num_destinations;
    first[dst_idx] = std::min(first[dst_idx], start[slot]);
    last[dst_idx] = std::max(last[dst_idx], end[slot]);
  }
  double sum = 0.0;
  for (size_t dst_idx = 0; dst_idx < num_destinations; ++dst_idx) {
    sum += static_cast<double>(last[dst_idx] - first[dst_idx]) / ticks_per_us;
  }
  return sum / static_cast<double>(num_destinations);
}

void PrintHeader() {
  std::cout << std::left << std::setw(6) << "Src" << std::setw(6) << "Dests" << std::setw(8)
            << "Grid" << std::setw(8) << "Block" << std::setw(14) << "TransferSize" << std::setw(12)
            << "CopySize" << std::setw(8) << "Copies" << std::setw(14) << "Device(us)"
            << std::setw(12) << "DevStd(us)" << std::setw(14) << "DeviceBW" << std::setw(14)
            << "Host(us)" << std::setw(12) << "HostStd(us)" << std::setw(12) << "HostBW" << "\n";
}

int Run(const Params& params, SignalMemory signal_memory) {
  if (!SelectSourceDevice(params.src_gpu)) return EXIT_FAILURE;
  const std::vector<int> dst_devices = PickDestinations(params.src_gpu, params.num_destinations);
  if (dst_devices.empty()) return EXIT_FAILURE;
  for (int dst : dst_devices) {
    if (!EnablePeerAccess(params.src_gpu, dst)) return EXIT_FAILURE;
  }
  HIP_CHECK(hipSetDevice(params.src_gpu));
  if (dst_devices.size() * params.num_queues > kMaxHandles) {
    std::cerr << "error: --numDestinations x --numOfQueuesPerDestination is at most " << kMaxHandles
              << "\n";
    return EXIT_FAILURE;
  }
  // Each destination's engines in turn, best first.
  std::vector<hipExtCopyEngineDeviceHandle> all_handles;
  for (int dst : dst_devices) {
    if (!GetCopyEngineHandles(params.src_gpu, dst, static_cast<int>(params.num_queues),
                              &all_handles)) {
      return EXIT_FAILURE;
    }
  }
  HandleSet handles{};
  std::copy(all_handles.begin(), all_handles.end(), handles.handles);

  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, params.src_gpu));
  int max_threads = 0;
  HIP_CHECK(
      hipDeviceGetAttribute(&max_threads, hipDeviceAttributeMaxThreadsPerBlock, params.src_gpu));
  const size_t block_size = params.warps_per_wg * static_cast<size_t>(warp_size);
  if (block_size > static_cast<size_t>(max_threads)) {
    std::cerr << "error: --warpsPerWG " << params.warps_per_wg << " exceeds the " << max_threads
              << " threads per block this GPU supports\n";
    return EXIT_FAILURE;
  }
  const double ticks_per_us = WallClockTicksPerUs(params.src_gpu);

  const size_t num_wgs = params.num_destinations * params.wgs_per_destination;
  const size_t total_warps = num_wgs * params.warps_per_wg;
  const size_t max_transfer = params.max_copy_size * params.num_copy_commands *
                              params.warps_per_wg * params.wgs_per_destination;

  std::cout << "Src GPU Device Id: " << params.src_gpu << "\n";
  if (params.verbose) {
    std::cout << "Dest GPU Ids:";
    for (int dst : dst_devices) std::cout << ' ' << dst;
    std::cout << "\nGridDim.x: " << num_wgs << ", BlockDim.x: " << block_size
              << ", wall clock: " << ticks_per_us << " ticks/us\n";
  }

  void* src_buffer = AllocUncached(max_transfer);
  FillMagic(src_buffer, max_transfer);
  std::vector<char*> dst_buffers;
  for (int dst : dst_devices) {
    HIP_CHECK(hipSetDevice(dst));
    dst_buffers.push_back(static_cast<char*>(AllocUncached(max_transfer)));
  }
  HIP_CHECK(hipSetDevice(params.src_gpu));

  char** dst_buffers_d = nullptr;
  HIP_CHECK(hipMalloc(&dst_buffers_d, dst_buffers.size() * sizeof(char*)));
  HIP_CHECK(hipMemcpy(dst_buffers_d, dst_buffers.data(), dst_buffers.size() * sizeof(char*),
                      hipMemcpyHostToDevice));

  uint64_t* signals = AllocSignals(total_warps, signal_memory);
  // A signal sets a value above every one it held before, so the words are never reset.
  uint64_t epoch = 0;
  const size_t timestamp_count = params.iterations * total_warps;
  int64_t* start_d = nullptr;
  int64_t* end_d = nullptr;
  HIP_CHECK(hipMalloc(&start_d, timestamp_count * sizeof(int64_t)));
  HIP_CHECK(hipMalloc(&end_d, timestamp_count * sizeof(int64_t)));
  std::vector<int64_t> start_h(timestamp_count);
  std::vector<int64_t> end_h(timestamp_count);
  hipError_t* statuses = nullptr;
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&statuses), total_warps * sizeof(hipError_t)));

  const dim3 grid(static_cast<unsigned>(num_wgs));
  const dim3 block(static_cast<unsigned>(block_size));

  PrintHeader();
  CsvWriter csv(params.output_file);
  csv.WriteHeader({"Src", "#Destinations", "GridDim", "BlockDim", "Total Transfer Size [B]",
                   "Copy Size [B]", "#Commands", "Device Latency [us] (Mean)",
                   "Device Latency [us] (Std)", "Device Bandwidth [GB/s]",
                   "Host Latency [us] (Mean)", "Host Latency [us] (Std)", "Host Bandwidth [GB/s]"});

  for (size_t copy_size = params.min_copy_size; copy_size <= params.max_copy_size; copy_size *= 2) {
    // Bytes that cross each destination's link per launch.
    const size_t transfer_size =
        copy_size * params.num_copy_commands * params.warps_per_wg * params.wgs_per_destination;
    for (char* dst : dst_buffers) HIP_CHECK(hipMemset(dst, 0, transfer_size));
    HIP_CHECK(hipDeviceSynchronize());

    auto launch = [&](int64_t* starts, int64_t* ends) {
      hipLaunchKernelGGL(
          CopyBandwidthKernel, grid, block, 0, 0, handles, static_cast<const char*>(src_buffer),
          dst_buffers_d, uint64_t{copy_size}, static_cast<uint32_t>(params.num_copy_commands),
          static_cast<uint32_t>(params.num_destinations), static_cast<uint32_t>(params.num_queues),
          signals, ++epoch, starts, ends, statuses);
    };

    if (!params.skip_verification) {
      launch(start_d, end_d);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipDeviceSynchronize());
      CheckStatuses(statuses, total_warps);
      for (size_t d = 0; d < dst_devices.size(); ++d) {
        if (!VerifyBuffer(dst_buffers[d], transfer_size, dst_devices[d])) {
          std::cerr << "Data verification failed\n";
          return EXIT_FAILURE;
        }
      }
      HIP_CHECK(hipSetDevice(params.src_gpu));
    }

    for (size_t i = 0; i < params.warmup; ++i) launch(start_d, end_d);
    HIP_CHECK(hipDeviceSynchronize());
    CheckStatuses(statuses, total_warps);

    std::vector<hipEvent_t> events(params.iterations * 2);
    for (hipEvent_t& event : events) HIP_CHECK(hipEventCreate(&event));
    for (size_t iter = 0; iter < params.iterations; ++iter) {
      hipExtLaunchKernelGGL(
          CopyBandwidthKernel, grid, block, 0, 0, events[iter * 2], events[iter * 2 + 1], 0,
          handles, static_cast<const char*>(src_buffer), dst_buffers_d, uint64_t{copy_size},
          static_cast<uint32_t>(params.num_copy_commands),
          static_cast<uint32_t>(params.num_destinations), static_cast<uint32_t>(params.num_queues),
          signals, ++epoch, start_d + iter * total_warps, end_d + iter * total_warps, statuses);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    CheckStatuses(statuses, total_warps);

    HIP_CHECK(hipMemcpy(start_h.data(), start_d, timestamp_count * sizeof(int64_t),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(end_h.data(), end_d, timestamp_count * sizeof(int64_t), hipMemcpyDeviceToHost));

    std::vector<double> device_us;
    std::vector<double> host_us;
    for (size_t iter = 0; iter < params.iterations; ++iter) {
      device_us.push_back(MeanSpanUs(start_h.data() + iter * total_warps,
                                     end_h.data() + iter * total_warps, params.num_destinations,
                                     params.warps_per_wg, total_warps, ticks_per_us));
      float host_ms = 0.0f;
      HIP_CHECK(hipEventElapsedTime(&host_ms, events[iter * 2], events[iter * 2 + 1]));
      host_us.push_back(host_ms * 1000.0);
    }
    for (hipEvent_t& event : events) HIP_CHECK(hipEventDestroy(event));

    const auto [device_mean, device_std] = MeanStd(device_us);
    const auto [host_mean, host_std] = MeanStd(host_us);
    const double bytes_all_links = static_cast<double>(dst_devices.size()) * transfer_size;
    const double device_gbps = bytes_all_links / 1.0e9 / (device_mean / 1.0e6);
    const double host_gbps = bytes_all_links / 1.0e9 / (host_mean / 1.0e6);

    std::cout << std::left << std::setw(6) << params.src_gpu << std::setw(6) << dst_devices.size()
              << std::setw(8) << num_wgs << std::setw(8) << block_size << std::setw(14)
              << transfer_size << std::setw(12) << copy_size << std::setw(8)
              << params.num_copy_commands << std::setw(14) << device_mean << std::setw(12)
              << device_std << std::setw(14) << device_gbps << std::setw(14) << host_mean
              << std::setw(12) << host_std << std::setw(12) << host_gbps << "\n";
    csv.WriteRow(params.src_gpu, dst_devices.size(), num_wgs, block_size, transfer_size, copy_size,
                 params.num_copy_commands, device_mean, device_std, device_gbps, host_mean,
                 host_std, host_gbps);
  }

  HIP_CHECK(hipHostFree(statuses));
  HIP_CHECK(hipFree(start_d));
  HIP_CHECK(hipFree(end_d));
  FreeSignals(signals, signal_memory);
  HIP_CHECK(hipFree(dst_buffers_d));
  HIP_CHECK(hipFree(src_buffer));
  for (size_t d = 0; d < dst_buffers.size(); ++d) {
    HIP_CHECK(hipSetDevice(dst_devices[d]));
    HIP_CHECK(hipFree(dst_buffers[d]));
  }
  return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
  Params params;
  ArgParser parser("copy-engine-bw",
                   "Kernel-submitted copy bandwidth through the HIP copy-engine API.");
  parser.AddOption("--srcGpu", &params.src_gpu, "Source GPU device ID");
  parser.AddOption("-b,--minCopySize", &params.min_copy_size, "Minimum copy size in bytes");
  parser.AddOption("-e,--maxCopySize", &params.max_copy_size, "Maximum copy size in bytes");
  parser.AddOption("-c,--numCopyCommands", &params.num_copy_commands,
                   "Copies submitted by each warp");
  parser.AddFlag("--skip-verification", &params.skip_verification, "Skip verification");
  parser.AddOption("-w,--warmup", &params.warmup, "Warmup iterations");
  parser.AddOption("-n,--iterations", &params.iterations, "Measured iterations");
  parser.AddOption("-d,--numDestinations", &params.num_destinations, "Number of destination GPUs");
  parser.AddOption("--warpsPerWG", &params.warps_per_wg, "Submitting warps per work-group");
  parser.AddOption("--wgsPerDestination", &params.wgs_per_destination,
                   "Work-groups per destination");
  parser.AddOption("--numOfQueuesPerDestination", &params.num_queues,
                   "Copy engines per destination: the runtime's best N for each pair, which the "
                   "work-groups of a destination take turns over");
  parser.AddOption("-o,--outputFile", &params.output_file, "Filename for CSV results");
  parser.AddOption("--signalMemory", &params.signal_memory,
                   "Where completion words live: device or host");
  parser.AddFlag("-v,--verbose", &params.verbose, "Verbose output");

  int exit_code = 0;
  if (!parser.Parse(argc, argv, &exit_code)) return exit_code;

  SignalMemory signal_memory = SignalMemory::kDevice;
  if (!ParseSignalMemory(params.signal_memory, &signal_memory)) {
    std::cerr << "error: --signalMemory must be 'device' or 'host'\n";
    return 2;
  }
  if (!ValidateSweep(params.min_copy_size, params.max_copy_size)) return 2;
  if (params.num_copy_commands == 0 || params.iterations == 0 || params.num_destinations == 0 ||
      params.warps_per_wg == 0 || params.wgs_per_destination == 0 || params.num_queues == 0) {
    std::cerr << "error: counts must be non-zero\n";
    return 2;
  }
  if (params.min_copy_size % sizeof(uint32_t) != 0) {
    std::cerr << "error: copy sizes must be a multiple of 4 bytes\n";
    return 2;
  }
  return Run(params, signal_memory);
}
