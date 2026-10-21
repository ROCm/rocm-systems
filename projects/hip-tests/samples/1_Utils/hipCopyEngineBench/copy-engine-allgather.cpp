/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// In-place allgather over the HIP copy-engine API, one process driving every GPU. Each GPU's
// buffer holds one chunk per rank; a kernel on each GPU pushes its own chunk into every peer's
// buffer, then waits until every peer's chunk has landed in its own. It mirrors the
// device-initiated mode of rocm-xio's sdma-ep-collectives allgather.

#include <chrono>
#include <iomanip>

#include "copy_engine_bench_common.hpp"

using namespace copy_engine_bench;

namespace {

struct Params {
  size_t num_gpus = 0;
  size_t min_bytes = 4096;
  size_t max_bytes = 256u << 20;
  size_t step_factor = 2;
  size_t warmup = 5;
  size_t iterations = 20;
  bool skip_verification = false;
  std::string output_file = "copy_engine_allgather.csv";
};

constexpr size_t kMaxRanks = kMaxDestinations;

// Everything one GPU's kernel needs. The arrays are indexed by rank.
struct AllgatherArgs {
  HandleSet handles;              // this GPU to each peer
  char* buffers[kMaxRanks];       // every rank's result buffer
  uint64_t* arrivals[kMaxRanks];  // every rank's words, one per sender
  uint64_t* sends_done;           // this GPU's words, one per receiver
  hipError_t* statuses;           // one per submitting warp
  uint64_t chunk_bytes;
  uint32_t rank;
  uint32_t num_ranks;
};

// Warp w of the single block serves peer w (skipping this rank). Its first lane copies this rank's
// chunk to the peer and queues two signals behind it on the same engine: one in the peer's memory
// that says the chunk landed, one in this GPU's memory that says it was sent. Then it waits for the
// peer's chunk to land here. Signals carry a rising epoch so they are never reset, and a peer that
// runs an iteration ahead can only raise them.
__global__ void AllgatherKernel(AllgatherArgs args, uint64_t epoch) {
  const uint32_t warp = threadIdx.x / warpSize;
  if (threadIdx.x % warpSize != 0 || warp + 1 >= args.num_ranks) return;
  const uint32_t peer = warp < args.rank ? warp : warp + 1;
  const uint64_t offset = uint64_t{args.rank} * args.chunk_bytes;

  const hipExtCopyEngineSignal arrived{reinterpret_cast<uint64_t>(args.arrivals[peer] + args.rank),
                                       epoch, hipExtCopyEngineSignalSet, 0};
  const hipExtCopyEngineSignal sent{reinterpret_cast<uint64_t>(args.sends_done + peer), epoch,
                                    hipExtCopyEngineSignalSet, 0};
  const hipExtCopyEngineDeviceHandle* handle = &args.handles.handles[peer];

  hipError_t result = hipExtCopyEngineSubmitLinearOneLane(handle, args.buffers[peer] + offset,
                                                          args.buffers[args.rank] + offset,
                                                          args.chunk_bytes, &arrived);
  if (result == hipSuccess) result = hipExtCopyEngineSubmitSignalOneLane(handle, &sent);
  if (result == hipSuccess) {
    const bool landed = WaitForSignal(args.arrivals[args.rank] + peer, epoch, kMaxSpins);
    const bool sent_done = WaitForSignal(args.sends_done + peer, epoch, kMaxSpins);
    if (!landed || !sent_done) result = hipErrorNotReady;
  }
  args.statuses[warp] = result;
}

// One GPU taking part.
struct Rank {
  int device = 0;
  AllgatherArgs args{};
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  uint64_t* arrivals = nullptr;
};

uint32_t ExpectedWord(size_t rank, size_t index) { return static_cast<uint32_t>(rank + index); }

void LaunchAll(std::vector<Rank>& ranks, size_t chunk_bytes, uint64_t epoch) {
  for (Rank& rank : ranks) {
    HIP_CHECK(hipSetDevice(rank.device));
    rank.args.chunk_bytes = chunk_bytes;
    int warp_size = 0;
    HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, rank.device));
    hipLaunchKernelGGL(AllgatherKernel, dim3(1),
                       dim3(static_cast<unsigned>((ranks.size() - 1) * warp_size)), 0, rank.stream,
                       rank.args, epoch);
  }
}

void SyncAll(const std::vector<Rank>& ranks) {
  for (const Rank& rank : ranks) {
    HIP_CHECK(hipSetDevice(rank.device));
    HIP_CHECK(hipStreamSynchronize(rank.stream));
    HIP_CHECK(hipGetLastError());
    CheckStatuses(rank.args.statuses, ranks.size() - 1);
  }
}

// Gives every GPU a zeroed buffer holding only its own chunk, runs the collective once and checks
// that every GPU ends up with every chunk. Returns the number of wrong words.
size_t Verify(std::vector<Rank>& ranks, size_t chunk_bytes, uint64_t epoch) {
  const size_t words_per_chunk = chunk_bytes / sizeof(uint32_t);
  const size_t total_bytes = chunk_bytes * ranks.size();
  std::vector<uint32_t> expected(words_per_chunk * ranks.size());
  for (size_t r = 0; r < ranks.size(); ++r) {
    for (size_t i = 0; i < words_per_chunk; ++i)
      expected[r * words_per_chunk + i] = ExpectedWord(r, i);
  }
  for (size_t r = 0; r < ranks.size(); ++r) {
    HIP_CHECK(hipSetDevice(ranks[r].device));
    HIP_CHECK(hipMemset(ranks[r].args.buffers[r], 0, total_bytes));
    HIP_CHECK(hipMemcpy(ranks[r].args.buffers[r] + r * chunk_bytes,
                        expected.data() + r * words_per_chunk, chunk_bytes, hipMemcpyHostToDevice));
  }
  for (const Rank& rank : ranks) {
    HIP_CHECK(hipSetDevice(rank.device));
    HIP_CHECK(hipDeviceSynchronize());
  }
  LaunchAll(ranks, chunk_bytes, epoch);
  SyncAll(ranks);

  size_t errors = 0;
  size_t reported = 0;
  std::vector<uint32_t> got(expected.size());
  for (size_t r = 0; r < ranks.size(); ++r) {
    HIP_CHECK(hipSetDevice(ranks[r].device));
    HIP_CHECK(hipMemcpy(got.data(), ranks[r].args.buffers[r], total_bytes, hipMemcpyDeviceToHost));
    for (size_t word = 0; word < got.size(); ++word) {
      if (got[word] == expected[word]) continue;
      ++errors;
      if (reported++ < 10) {
        std::cerr << "mismatch on GPU " << ranks[r].device << " word " << word << ": expected 0x"
                  << std::hex << expected[word] << ", got 0x" << got[word] << std::dec << "\n";
      }
    }
  }
  return errors;
}

int Run(const Params& params) {
  int num_devices = 0;
  HIP_CHECK(hipGetDeviceCount(&num_devices));
  const size_t num_gpus =
      params.num_gpus == 0 ? std::min<size_t>(num_devices, kMaxRanks) : params.num_gpus;
  if (num_gpus < 2 || num_gpus > kMaxRanks || num_gpus > static_cast<size_t>(num_devices)) {
    std::cerr << "error: --numGpus must be between 2 and "
              << std::min<size_t>(num_devices, kMaxRanks) << "\n";
    return EXIT_FAILURE;
  }
  if (params.max_bytes / num_gpus < sizeof(uint32_t)) {
    std::cerr << "error: --maxBytes is too small for " << num_gpus << " GPUs\n";
    return EXIT_FAILURE;
  }

  std::vector<Rank> ranks(num_gpus);
  for (size_t r = 0; r < num_gpus; ++r) ranks[r].device = static_cast<int>(r);
  for (size_t src = 0; src < num_gpus; ++src) {
    for (size_t dst = 0; dst < num_gpus; ++dst) {
      if (src != dst && !EnablePeerAccess(static_cast<int>(src), static_cast<int>(dst))) {
        return EXIT_FAILURE;
      }
    }
  }

  const size_t max_chunk = params.max_bytes / num_gpus / sizeof(uint32_t) * sizeof(uint32_t);
  for (size_t r = 0; r < num_gpus; ++r) {
    Rank& rank = ranks[r];
    if (!SelectSourceDevice(rank.device)) return EXIT_FAILURE;
    rank.args.rank = static_cast<uint32_t>(r);
    rank.args.num_ranks = static_cast<uint32_t>(num_gpus);
    rank.args.buffers[r] = static_cast<char*>(AllocUncached(max_chunk * num_gpus));
    rank.arrivals = AllocSignals(num_gpus, SignalMemory::kDevice);
    rank.args.arrivals[r] = rank.arrivals;
    rank.args.sends_done = AllocSignals(num_gpus, SignalMemory::kDevice);
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&rank.args.statuses),
                            num_gpus * sizeof(hipError_t)));
    HIP_CHECK(hipStreamCreateWithFlags(&rank.stream, hipStreamNonBlocking));
    HIP_CHECK(hipEventCreate(&rank.start));
    HIP_CHECK(hipEventCreate(&rank.stop));
    for (size_t peer = 0; peer < num_gpus; ++peer) {
      if (peer == r) continue;
      std::vector<hipExtCopyEngineDeviceHandle> handle;
      if (!GetCopyEngineHandles(rank.device, static_cast<int>(peer), 1, &handle)) {
        return EXIT_FAILURE;
      }
      rank.args.handles.handles[peer] = handle[0];
    }
  }
  for (Rank& rank : ranks) {
    for (size_t peer = 0; peer < num_gpus; ++peer) {
      rank.args.buffers[peer] = ranks[peer].args.buffers[peer];
      rank.args.arrivals[peer] = ranks[peer].arrivals;
    }
  }

  std::cout << "GPUs: " << num_gpus << ", one block of " << (num_gpus - 1)
            << " submitting warp(s) per GPU\n";
  std::cout << std::left << std::setw(14) << "TotalBytes" << std::setw(12) << "ChunkBytes"
            << std::setw(14) << "Device(us)" << std::setw(14) << "DevMax(us)" << std::setw(14)
            << "Host(us)" << std::setw(14) << "AlgBW(GiB/s)" << std::setw(14) << "BusBW(GiB/s)"
            << std::setw(8) << "Errors" << "\n";
  CsvWriter csv(params.output_file);
  csv.WriteHeader({"#GPUs", "Total Size [B]", "Chunk Size [B]", "Device Latency [us] (Mean)",
                   "Device Latency [us] (Max)", "Host Latency [us]", "Algorithm Bandwidth [GiB/s]",
                   "Bus Bandwidth [GiB/s]", "Errors"});

  // Signals keep rising across sizes and launches, so they are never reset.
  uint64_t epoch = 0;
  constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
  for (size_t size = params.min_bytes; size <= params.max_bytes; size *= params.step_factor) {
    const size_t chunk = size / num_gpus / sizeof(uint32_t) * sizeof(uint32_t);
    if (chunk == 0) continue;
    const size_t total = chunk * num_gpus;

    size_t errors = 0;
    bool verified = false;
    if (!params.skip_verification) {
      errors = Verify(ranks, chunk, ++epoch);
      verified = true;
    }

    for (size_t i = 0; i < params.warmup; ++i) LaunchAll(ranks, chunk, ++epoch);
    SyncAll(ranks);

    for (Rank& rank : ranks) {
      HIP_CHECK(hipSetDevice(rank.device));
      HIP_CHECK(hipEventRecord(rank.start, rank.stream));
    }
    const auto host_begin = std::chrono::steady_clock::now();
    for (size_t i = 0; i < params.iterations; ++i) LaunchAll(ranks, chunk, ++epoch);
    for (Rank& rank : ranks) {
      HIP_CHECK(hipSetDevice(rank.device));
      HIP_CHECK(hipEventRecord(rank.stop, rank.stream));
    }
    SyncAll(ranks);
    const double host_us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - host_begin)
            .count() /
        static_cast<double>(params.iterations);

    double sum_us = 0.0;
    double max_us = 0.0;
    for (const Rank& rank : ranks) {
      float ms = 0.0f;
      HIP_CHECK(hipEventElapsedTime(&ms, rank.start, rank.stop));
      const double us = ms * 1000.0 / static_cast<double>(params.iterations);
      sum_us += us;
      max_us = std::max(max_us, us);
    }
    const double device_us = sum_us / static_cast<double>(num_gpus);
    const double alg_bw = static_cast<double>(total) / kGiB / (device_us / 1.0e6);
    const double bus_bw =
        alg_bw * static_cast<double>(num_gpus - 1) / static_cast<double>(num_gpus);

    std::cout << std::left << std::setw(14) << total << std::setw(12) << chunk << std::setw(14)
              << device_us << std::setw(14) << max_us << std::setw(14) << host_us << std::setw(14)
              << alg_bw << std::setw(14) << bus_bw << std::setw(8)
              << (verified ? std::to_string(errors) : std::string("N/A")) << "\n";
    csv.WriteRow(num_gpus, total, chunk, device_us, max_us, host_us, alg_bw, bus_bw,
                 verified ? std::to_string(errors) : std::string("N/A"));
    if (errors != 0) {
      std::cerr << "Data verification failed\n";
      return EXIT_FAILURE;
    }
  }

  for (Rank& rank : ranks) {
    HIP_CHECK(hipSetDevice(rank.device));
    HIP_CHECK(hipEventDestroy(rank.start));
    HIP_CHECK(hipEventDestroy(rank.stop));
    HIP_CHECK(hipStreamDestroy(rank.stream));
    HIP_CHECK(hipHostFree(rank.args.statuses));
    FreeSignals(rank.arrivals, SignalMemory::kDevice);
    FreeSignals(rank.args.sends_done, SignalMemory::kDevice);
    HIP_CHECK(hipFree(rank.args.buffers[rank.args.rank]));
  }
  return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
  Params params;
  ArgParser parser("copy-engine-allgather",
                   "In-place allgather across GPUs through the HIP copy-engine API.");
  parser.AddOption("-g,--numGpus", &params.num_gpus, "GPUs taking part, 0 for all (at most 16)");
  parser.AddOption("-b,--minBytes", &params.min_bytes, "Smallest total buffer size in bytes");
  parser.AddOption("-e,--maxBytes", &params.max_bytes, "Largest total buffer size in bytes");
  parser.AddOption("-f,--stepFactor", &params.step_factor, "Factor between sizes");
  parser.AddOption("-w,--warmup", &params.warmup, "Warmup iterations");
  parser.AddOption("-n,--iterations", &params.iterations, "Measured iterations");
  parser.AddFlag("--skip-verification", &params.skip_verification, "Skip verification");
  parser.AddOption("-o,--outputFile", &params.output_file, "Filename for CSV results");

  int exit_code = 0;
  if (!parser.Parse(argc, argv, &exit_code)) return exit_code;

  if (!ValidateSweep(params.min_bytes, params.max_bytes)) return 2;
  if (params.step_factor < 2 || params.iterations == 0) {
    std::cerr << "error: --stepFactor must be at least 2 and --iterations non-zero\n";
    return 2;
  }
  return Run(params);
}
