/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Kernel-submitted copy latency through the HIP copy-engine API: one thread submits a copy (or
// a few, back to back) and waits for it, timed on the device and from the host. The time
// splits into the submit calls and the wait that follows the last of them.

#include <iomanip>

#include "copy_engine_bench_common.hpp"

using namespace copy_engine_bench;

namespace {

struct Params {
  int src_gpu = 0;
  int dst_gpu = 1;
  size_t min_size = 256;
  size_t max_size = 1048576;
  size_t commands = 1;
  size_t warmup = 3;
  size_t iterations = 100;
  std::string output_file = "copy_engine_latency.csv";
  bool skip_verification = false;
  std::string signal_memory = "device";
  size_t handle_rank = 0;
};

// One run: |start| to |end| is the whole latency, |submit_end| is when the last submit returned.
// The handle is a kernel argument and the engine it names is read once the first submit starts, so
// nothing is looked up before |start|, as in sdma-ep-latency.
__global__ void CopyLatencyKernel(hipExtCopyEngineDeviceHandle handle, const char* src, char* dst,
                                  uint64_t copy_size, uint32_t num_commands, uint64_t* signal,
                                  uint64_t epoch, int64_t* start, int64_t* submit_end, int64_t* end,
                                  hipError_t* status) {
  if (threadIdx.x != 0) return;
  start[0] = wall_clock64();
  status[0] =
      SubmitBatchAndWait(&handle, dst, src, copy_size, num_commands, signal, epoch, submit_end);
  end[0] = wall_clock64();
}

int Run(const Params& params, SignalMemory signal_memory) {
  int num_devices = 0;
  HIP_CHECK(hipGetDeviceCount(&num_devices));
  if (params.dst_gpu < 0 || params.dst_gpu >= num_devices || params.dst_gpu == params.src_gpu) {
    std::cerr << "error: --dstGpu must be a valid GPU other than --srcGpu\n";
    return EXIT_FAILURE;
  }
  if (!SelectSourceDevice(params.src_gpu)) return EXIT_FAILURE;
  if (!EnablePeerAccess(params.src_gpu, params.dst_gpu)) return EXIT_FAILURE;
  HIP_CHECK(hipSetDevice(params.src_gpu));
  std::vector<hipExtCopyEngineDeviceHandle> handles;
  if (!GetCopyEngineHandles(params.src_gpu, params.dst_gpu,
                            static_cast<int>(params.handle_rank) + 1, &handles)) {
    return EXIT_FAILURE;
  }
  const hipExtCopyEngineDeviceHandle handle = handles[params.handle_rank];
  const double ticks_per_us = WallClockTicksPerUs(params.src_gpu);

  const size_t max_bytes = params.max_size * params.commands;
  void* src_buffer = AllocUncached(max_bytes);
  FillMagic(src_buffer, max_bytes);
  HIP_CHECK(hipSetDevice(params.dst_gpu));
  char* dst_buffer = static_cast<char*>(AllocUncached(max_bytes));
  HIP_CHECK(hipSetDevice(params.src_gpu));

  uint64_t* signal = AllocSignals(1, signal_memory);
  // A signal sets a value above every one it held before, so the word is never reset.
  uint64_t epoch = 0;
  int64_t* clocks_d = nullptr;
  HIP_CHECK(hipMalloc(&clocks_d, 3 * params.iterations * sizeof(int64_t)));
  int64_t* start_d = clocks_d;
  int64_t* submit_end_d = clocks_d + params.iterations;
  int64_t* end_d = clocks_d + 2 * params.iterations;
  std::vector<int64_t> clocks_h(3 * params.iterations);
  hipError_t* status = nullptr;
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&status), sizeof(hipError_t)));

  std::cout << std::left << std::setw(12) << "CopySize" << std::setw(14) << "Device(us)"
            << std::setw(14) << "Host(us)" << std::setw(16) << "DeviceBW(GB/s)" << std::setw(14)
            << "Submit(us)" << std::setw(14) << "Transfer(us)" << "\n";
  CsvWriter csv(params.output_file);
  csv.WriteHeader({"Copy Size [B]", "#Commands", "Device Latency [us] (Mean)",
                   "Device Latency [us] (Std)", "Host Latency [us] (Mean)",
                   "Host Latency [us] (Std)", "Device Bandwidth [GB/s]", "Submit [us]",
                   "Transfer [us]"});

  for (size_t size = params.min_size; size <= params.max_size; size *= 2) {
    const size_t bytes = size * params.commands;
    HIP_CHECK(hipMemset(dst_buffer, 0, bytes));
    HIP_CHECK(hipDeviceSynchronize());

    auto launch = [&](size_t iter) {
      hipLaunchKernelGGL(CopyLatencyKernel, dim3(1), dim3(1), 0, 0, handle,
                         static_cast<const char*>(src_buffer), dst_buffer, uint64_t{size},
                         static_cast<uint32_t>(params.commands), signal, ++epoch, start_d + iter,
                         submit_end_d + iter, end_d + iter, status);
    };

    if (!params.skip_verification) {
      launch(0);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipDeviceSynchronize());
      CheckStatuses(status, 1);
      if (!VerifyBuffer(dst_buffer, bytes, params.dst_gpu)) {
        std::cerr << "Data verification failed\n";
        return EXIT_FAILURE;
      }
      HIP_CHECK(hipSetDevice(params.src_gpu));
    }

    for (size_t i = 0; i < params.warmup; ++i) launch(0);
    HIP_CHECK(hipDeviceSynchronize());
    CheckStatuses(status, 1);

    // Event i opens iteration i and event i + 1 closes it.
    std::vector<hipEvent_t> events(params.iterations + 1);
    for (hipEvent_t& event : events) HIP_CHECK(hipEventCreate(&event));
    for (size_t iter = 0; iter < params.iterations; ++iter) {
      HIP_CHECK(hipEventRecord(events[iter]));
      launch(iter);
    }
    HIP_CHECK(hipEventRecord(events.back()));
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    CheckStatuses(status, 1);
    HIP_CHECK(hipMemcpy(clocks_h.data(), clocks_d, clocks_h.size() * sizeof(int64_t),
                        hipMemcpyDeviceToHost));

    std::vector<double> device_us;
    std::vector<double> host_us;
    std::vector<double> submit_us;
    std::vector<double> transfer_us;
    for (size_t iter = 0; iter < params.iterations; ++iter) {
      const int64_t start = clocks_h[iter];
      const int64_t submit_end = clocks_h[params.iterations + iter];
      const int64_t end = clocks_h[2 * params.iterations + iter];
      device_us.push_back(static_cast<double>(end - start) / ticks_per_us);
      submit_us.push_back(static_cast<double>(submit_end - start) / ticks_per_us /
                          static_cast<double>(params.commands));
      transfer_us.push_back(static_cast<double>(end - submit_end) / ticks_per_us);
      float host_ms = 0.0f;
      HIP_CHECK(hipEventElapsedTime(&host_ms, events[iter], events[iter + 1]));
      host_us.push_back(host_ms * 1000.0);
    }
    for (hipEvent_t& event : events) HIP_CHECK(hipEventDestroy(event));

    const auto [device_mean, device_std] = MeanStd(device_us);
    const auto [host_mean, host_std] = MeanStd(host_us);
    const double submit_mean = MeanStd(submit_us).first;
    const double transfer_mean = MeanStd(transfer_us).first;
    const double bandwidth = static_cast<double>(bytes) / 1.0e9 / (device_mean / 1.0e6);

    std::cout << std::left << std::setw(12) << size << std::setw(14) << device_mean << std::setw(14)
              << host_mean << std::setw(16) << bandwidth << std::setw(14) << submit_mean
              << std::setw(14) << transfer_mean << "\n";
    csv.WriteRow(size, params.commands, device_mean, device_std, host_mean, host_std, bandwidth,
                 submit_mean, transfer_mean);
  }

  HIP_CHECK(hipHostFree(status));
  HIP_CHECK(hipFree(clocks_d));
  FreeSignals(signal, signal_memory);
  HIP_CHECK(hipFree(src_buffer));
  HIP_CHECK(hipSetDevice(params.dst_gpu));
  HIP_CHECK(hipFree(dst_buffer));
  return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
  Params params;
  ArgParser parser("copy-engine-latency",
                   "Kernel-submitted copy latency through the HIP copy-engine API.");
  parser.AddOption("--srcGpu", &params.src_gpu, "Source GPU device ID");
  parser.AddOption("--dstGpu", &params.dst_gpu, "Destination GPU device ID");
  parser.AddOption("-b,--minCopySize", &params.min_size, "Minimum copy size in bytes");
  parser.AddOption("-e,--maxCopySize", &params.max_size, "Maximum copy size in bytes");
  parser.AddOption("-c,--numCopyCommands", &params.commands, "Copies submitted per run");
  parser.AddOption("-w,--warmup", &params.warmup, "Warmup iterations");
  parser.AddOption("-n,--iterations", &params.iterations, "Measured iterations");
  parser.AddOption("-o,--outputFile", &params.output_file, "CSV output path");
  parser.AddFlag("--skip-verification", &params.skip_verification, "Skip data verification");
  parser.AddOption("--handleRank", &params.handle_rank,
                   "Which of the runtime's engines for this pair to submit on, best first (0)");
  parser.AddOption("--signalMemory", &params.signal_memory,
                   "Where completion words live: device or host");

  int exit_code = 0;
  if (!parser.Parse(argc, argv, &exit_code)) return exit_code;

  SignalMemory signal_memory = SignalMemory::kDevice;
  if (!ParseSignalMemory(params.signal_memory, &signal_memory)) {
    std::cerr << "error: --signalMemory must be 'device' or 'host'\n";
    return 2;
  }
  if (!ValidateSweep(params.min_size, params.max_size)) return 2;
  if (params.commands == 0 || params.iterations == 0) {
    std::cerr << "error: counts must be non-zero\n";
    return 2;
  }
  if (params.min_size % sizeof(uint32_t) != 0) {
    std::cerr << "error: copy sizes must be a multiple of 4 bytes\n";
    return 2;
  }
  if (params.max_size > SIZE_MAX / params.commands) {
    std::cerr << "error: --maxCopySize * --numCopyCommands overflows\n";
    return 2;
  }
  return Run(params, signal_memory);
}
