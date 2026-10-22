/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// In-place allgather over the HIP copy-engine API. Each GPU's buffer holds one chunk per rank; a
// kernel on each GPU pushes its own chunk into every peer's buffer, then waits until every peer's
// chunk has landed in its own. It mirrors the device-initiated mode of rocm-xio's
// sdma-ep-collectives allgather.
//
// Built plain, one process drives every GPU. Built with COPY_ENGINE_BENCH_MPI, every MPI rank
// drives the GPU of its own number and reaches its peers' buffers and signals through IPC handles,
// like the reference.

#include <chrono>
#include <iomanip>
#include <memory>

#include "copy_engine_bench_common.hpp"

#ifdef COPY_ENGINE_BENCH_MPI
#include <mpi.h>
#endif

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
  bool skip_barrier = false;
  std::string submit = "fused";
  std::string signal_op = "add";
  std::string output_file = "copy_engine_allgather.csv";
};

constexpr size_t kMaxRanks = kMaxDestinations;

// Everything one GPU's kernel needs. The arrays are indexed by rank.
struct AllgatherArgs {
  HandleSet handles;              // this GPU to each peer
  char* buffers[kMaxRanks];       // every rank's result buffer
  uint64_t* arrivals[kMaxRanks];  // every rank's words, one per sender
  uint64_t* sends_done;           // this GPU's words, one per receiver
  uint64_t* barriers[kMaxRanks];  // every rank's words, one per rank
  bool barrier;                   // start every collective with a barrier over all ranks
  bool add;                       // signals add 1 instead of setting the epoch
  hipError_t* statuses;           // one per submitting warp
  uint64_t chunk_bytes;
  uint32_t rank;
  uint32_t num_ranks;
};

// Warp w of the single block serves peer w (skipping this rank). Optionally every collective
// starts with a barrier: each warp stores the epoch into its peer's barrier word for this rank and
// waits for the peer's, then the block syncs, so no copy starts before every rank has arrived.
// Then the warp's first lane copies this rank's chunk to the peer and queues two signals behind
// it on the same engine: one in the peer's memory that says the chunk landed, one in this GPU's
// memory that says it was sent. With Fused both signals ride in the copy's submit, otherwise the
// second is a submit of its own. Finally it waits for the peer's chunk to land here. Signals carry
// a rising epoch so they are never reset, and a peer that runs an iteration ahead can only raise
// them.
template <bool Fused> __global__ void AllgatherKernel(AllgatherArgs args, uint64_t epoch) {
  const uint32_t warp = threadIdx.x / warpSize;
  const bool submitting_lane = threadIdx.x % warpSize == 0 && warp + 1 < args.num_ranks;
  const uint32_t peer = warp < args.rank ? warp : warp + 1;
  bool barrier_met = true;
  if (args.barrier) {
    if (submitting_lane) {
      __hip_atomic_store(args.barriers[peer] + args.rank, epoch, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_SYSTEM);
      barrier_met = WaitForSignal(args.barriers[args.rank] + peer, epoch, kMaxSpins);
    }
    __syncthreads();
  }
  if (!submitting_lane) return;

  const uint64_t offset = uint64_t{args.rank} * args.chunk_bytes;
  const hipExtCopyEngineDeviceHandle* handle = &args.handles.handles[peer];
  const uint32_t op = args.add ? hipExtCopyEngineSignalAdd : hipExtCopyEngineSignalSet;
  const uint64_t value = args.add ? 1 : epoch;
  const hipExtCopyEngineSignal arrived{reinterpret_cast<uint64_t>(args.arrivals[peer] + args.rank),
                                       value, op, 0};
  const hipExtCopyEngineSignal sent{reinterpret_cast<uint64_t>(args.sends_done + peer), value, op,
                                    0};

  hipError_t result = barrier_met ? hipSuccess : hipErrorNotReady;
  if (result == hipSuccess) {
    if constexpr (Fused) {
      result = hipExtCopyEngineSubmitLinearTwoCompletionsOneLane(
          handle, args.buffers[peer] + offset, args.buffers[args.rank] + offset, args.chunk_bytes,
          &arrived, &sent);
    } else {
      result = hipExtCopyEngineSubmitLinearOneLane(handle, args.buffers[peer] + offset,
                                                   args.buffers[args.rank] + offset,
                                                   args.chunk_bytes, &arrived);
      if (result == hipSuccess) result = hipExtCopyEngineSubmitSignalOneLane(handle, &sent);
    }
  }
  if (result == hipSuccess) {
    const bool landed = WaitForSignal(args.arrivals[args.rank] + peer, epoch, kMaxSpins);
    const bool sent_done = WaitForSignal(args.sends_done + peer, epoch, kMaxSpins);
    if (!landed || !sent_done) result = hipErrorNotReady;
  }
  args.statuses[warp] = result;
}

// The processes taking part. Without MPI that is this one, and every number reduces to itself.
struct World {
  int rank = 0;
  int size = 1;

  bool IsRoot() const { return rank == 0; }

#ifdef COPY_ENGINE_BENCH_MPI
  void Barrier() const { MPI_Barrier(MPI_COMM_WORLD); }
  double Sum(double value) const {
    double total = 0.0;
    MPI_Allreduce(&value, &total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return total;
  }
  double Max(double value) const {
    double result = 0.0;
    MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return result;
  }
  size_t Sum(size_t value) const {
    unsigned long long total = 0;
    unsigned long long mine = value;
    MPI_Allreduce(&mine, &total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    return static_cast<size_t>(total);
  }
#else
  void Barrier() const {}
  double Sum(double value) const { return value; }
  double Max(double value) const { return value; }
  size_t Sum(size_t value) const { return value; }
#endif
};

// One GPU this process drives.
struct Rank {
  int device = 0;
  AllgatherArgs args{};
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  uint64_t* arrivals = nullptr;
  uint64_t* barriers = nullptr;
  // Peer allocations mapped through IPC, closed at the end.
  std::vector<void*> mapped;
};

std::string Fixed2(double value) {
  std::ostringstream text;
  text << std::fixed << std::setprecision(2) << value;
  return text.str();
}

uint32_t ExpectedWord(size_t rank, size_t index) { return static_cast<uint32_t>(rank + index); }

// Launches one kernel per GPU that runs the collective once with signal value |epoch|.
void LaunchAll(std::vector<Rank>& ranks, size_t chunk_bytes, uint64_t epoch, bool fused) {
  for (Rank& rank : ranks) {
    HIP_CHECK(hipSetDevice(rank.device));
    rank.args.chunk_bytes = chunk_bytes;
    int warp_size = 0;
    HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, rank.device));
    const dim3 block(static_cast<unsigned>((rank.args.num_ranks - 1) * warp_size));
    if (fused) {
      hipLaunchKernelGGL(AllgatherKernel<true>, dim3(1), block, 0, rank.stream, rank.args, epoch);
    } else {
      hipLaunchKernelGGL(AllgatherKernel<false>, dim3(1), block, 0, rank.stream, rank.args, epoch);
    }
  }
}

void SyncAll(const std::vector<Rank>& ranks) {
  for (const Rank& rank : ranks) {
    HIP_CHECK(hipSetDevice(rank.device));
    HIP_CHECK(hipStreamSynchronize(rank.stream));
    HIP_CHECK(hipGetLastError());
    CheckStatuses(rank.args.statuses, rank.args.num_ranks - 1);
  }
}

// Gives every GPU a zeroed buffer holding only its own chunk, runs the collective once and checks
// that every GPU ends up with every chunk. Returns the number of wrong words over all processes.
size_t Verify(const World& world, std::vector<Rank>& ranks, size_t chunk_bytes, uint64_t epoch,
              bool fused) {
  const size_t num_ranks = ranks[0].args.num_ranks;
  const size_t words_per_chunk = chunk_bytes / sizeof(uint32_t);
  const size_t total_bytes = chunk_bytes * num_ranks;
  std::vector<uint32_t> expected(words_per_chunk * num_ranks);
  for (size_t r = 0; r < num_ranks; ++r) {
    for (size_t i = 0; i < words_per_chunk; ++i) {
      expected[r * words_per_chunk + i] = ExpectedWord(r, i);
    }
  }
  for (Rank& rank : ranks) {
    const size_t r = rank.args.rank;
    HIP_CHECK(hipSetDevice(rank.device));
    HIP_CHECK(hipMemset(rank.args.buffers[r], 0, total_bytes));
    HIP_CHECK(hipMemcpy(rank.args.buffers[r] + r * chunk_bytes,
                        expected.data() + r * words_per_chunk, chunk_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());
  }
  world.Barrier();
  LaunchAll(ranks, chunk_bytes, epoch, fused);
  SyncAll(ranks);
  world.Barrier();

  size_t errors = 0;
  size_t reported = 0;
  std::vector<uint32_t> got(expected.size());
  for (Rank& rank : ranks) {
    HIP_CHECK(hipSetDevice(rank.device));
    HIP_CHECK(hipMemcpy(got.data(), rank.args.buffers[rank.args.rank], total_bytes,
                        hipMemcpyDeviceToHost));
    for (size_t word = 0; word < got.size(); ++word) {
      if (got[word] == expected[word]) continue;
      ++errors;
      if (reported++ < 10) {
        std::cerr << "mismatch on GPU " << rank.device << " word " << word << ": expected 0x"
                  << std::hex << expected[word] << ", got 0x" << got[word] << std::dec << "\n";
      }
    }
  }
  return world.Sum(errors);
}

#ifdef COPY_ENGINE_BENCH_MPI
// What a process tells the others about its allocations.
struct Exported {
  hipIpcMemHandle_t buffer;
  hipIpcMemHandle_t arrivals;
  hipIpcMemHandle_t barriers;
};

// Fills |rank|'s peer pointers from every process' exports.
void MapPeers(const World& world, Rank& rank) {
  Exported mine{};
  HIP_CHECK(hipIpcGetMemHandle(&mine.buffer, rank.args.buffers[world.rank]));
  HIP_CHECK(hipIpcGetMemHandle(&mine.arrivals, rank.arrivals));
  HIP_CHECK(hipIpcGetMemHandle(&mine.barriers, rank.barriers));
  std::vector<Exported> all(world.size);
  MPI_Allgather(&mine, sizeof(mine), MPI_BYTE, all.data(), sizeof(mine), MPI_BYTE, MPI_COMM_WORLD);
  HIP_CHECK(hipSetDevice(rank.device));
  for (int peer = 0; peer < world.size; ++peer) {
    if (peer == world.rank) continue;
    void* buffer = nullptr;
    void* arrivals = nullptr;
    void* barriers = nullptr;
    HIP_CHECK(hipIpcOpenMemHandle(&buffer, all[peer].buffer, hipIpcMemLazyEnablePeerAccess));
    HIP_CHECK(hipIpcOpenMemHandle(&arrivals, all[peer].arrivals, hipIpcMemLazyEnablePeerAccess));
    HIP_CHECK(hipIpcOpenMemHandle(&barriers, all[peer].barriers, hipIpcMemLazyEnablePeerAccess));
    rank.mapped.push_back(buffer);
    rank.mapped.push_back(arrivals);
    rank.mapped.push_back(barriers);
    rank.args.buffers[peer] = static_cast<char*>(buffer);
    rank.args.arrivals[peer] = static_cast<uint64_t*>(arrivals);
    rank.args.barriers[peer] = static_cast<uint64_t*>(barriers);
  }
}
#endif

int Run(const Params& params, const World& world) {
  int num_devices = 0;
  HIP_CHECK(hipGetDeviceCount(&num_devices));
#ifdef COPY_ENGINE_BENCH_MPI
  const size_t num_gpus = static_cast<size_t>(world.size);
  if (params.num_gpus != 0 && params.num_gpus != num_gpus) {
    if (world.IsRoot()) std::cerr << "error: --numGpus is the number of MPI ranks; use mpirun -n\n";
    return EXIT_FAILURE;
  }
#else
  const size_t num_gpus =
      params.num_gpus == 0 ? std::min<size_t>(num_devices, kMaxRanks) : params.num_gpus;
#endif
  if (num_gpus < 2 || num_gpus > kMaxRanks || num_gpus > static_cast<size_t>(num_devices)) {
    if (world.IsRoot()) {
      std::cerr << "error: need between 2 and " << std::min<size_t>(num_devices, kMaxRanks)
                << " GPUs\n";
    }
    return EXIT_FAILURE;
  }
  if (params.max_bytes / num_gpus < sizeof(uint32_t)) {
    if (world.IsRoot()) std::cerr << "error: --maxBytes is too small for " << num_gpus << " GPUs\n";
    return EXIT_FAILURE;
  }

  // The GPUs this process drives: all of them, or the one of its MPI rank.
  std::vector<Rank> ranks;
#ifdef COPY_ENGINE_BENCH_MPI
  ranks.resize(1);
  ranks[0].args.rank = static_cast<uint32_t>(world.rank);
#else
  ranks.resize(num_gpus);
  for (size_t r = 0; r < num_gpus; ++r) ranks[r].args.rank = static_cast<uint32_t>(r);
#endif
  for (Rank& rank : ranks) {
    rank.device = static_cast<int>(rank.args.rank);
    for (size_t peer = 0; peer < num_gpus; ++peer) {
      if (peer != rank.args.rank && !EnablePeerAccess(rank.device, static_cast<int>(peer))) {
        return EXIT_FAILURE;
      }
    }
  }

  const size_t max_chunk = params.max_bytes / num_gpus / sizeof(uint32_t) * sizeof(uint32_t);
  for (Rank& rank : ranks) {
    const size_t r = rank.args.rank;
    if (!SelectSourceDevice(rank.device)) return EXIT_FAILURE;
    rank.args.num_ranks = static_cast<uint32_t>(num_gpus);
    rank.args.buffers[r] = static_cast<char*>(AllocUncached(max_chunk * num_gpus));
    rank.arrivals = AllocSignals(num_gpus, SignalMemory::kDevice);
    rank.args.arrivals[r] = rank.arrivals;
    rank.barriers = AllocSignals(num_gpus, SignalMemory::kDevice);
    rank.args.barriers[r] = rank.barriers;
    rank.args.barrier = !params.skip_barrier;
    rank.args.add = params.signal_op == "add";
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
#ifdef COPY_ENGINE_BENCH_MPI
  MapPeers(world, ranks[0]);
#else
  for (Rank& rank : ranks) {
    for (size_t peer = 0; peer < num_gpus; ++peer) {
      rank.args.buffers[peer] = ranks[peer].args.buffers[peer];
      rank.args.arrivals[peer] = ranks[peer].arrivals;
      rank.args.barriers[peer] = ranks[peer].barriers;
    }
  }
#endif

  std::unique_ptr<CsvWriter> csv;
  if (world.IsRoot()) {
    std::cout << "GPUs: " << num_gpus << ", one block of " << (num_gpus - 1)
              << " submitting warp(s) per GPU"
              << (world.size > 1 ? ", one MPI rank per GPU" : ", one process") << "\n";
    std::cout << std::left << std::setw(14) << "Size(B)" << std::setw(18) << "SizePerRank (B)"
              << std::setw(14) << "Device(us)" << std::setw(14) << "DevMax(us)" << std::setw(14)
              << "Host(us)" << std::setw(14) << "AlgBW(GiB/s)" << std::setw(14) << "BusBW(GiB/s)"
              << std::setw(8) << "Errors" << "\n";
    csv = std::make_unique<CsvWriter>(params.output_file);
    csv->WriteHeader({"#GPUs", "Size [B]", "Size Per Rank [B]", "Device Latency [us] (Mean)",
                      "Device Latency [us] (Max)", "Host Latency [us]",
                      "Algorithm Bandwidth [GiB/s]", "Bus Bandwidth [GiB/s]", "Errors"});
  }

  // Signals keep rising across sizes and launches, so they are never reset.
  uint64_t epoch = 0;
  const bool fused = params.submit == "fused";
  constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
  for (size_t size = params.min_bytes; size <= params.max_bytes; size *= params.step_factor) {
    const size_t chunk = size / num_gpus / sizeof(uint32_t) * sizeof(uint32_t);
    if (chunk == 0) continue;
    const size_t total = chunk * num_gpus;

    size_t errors = 0;
    const bool verified = !params.skip_verification;
    if (verified) errors = Verify(world, ranks, chunk, ++epoch, fused);

    for (size_t i = 0; i < params.warmup; ++i) LaunchAll(ranks, chunk, ++epoch, fused);
    SyncAll(ranks);
    world.Barrier();

    for (Rank& rank : ranks) {
      HIP_CHECK(hipSetDevice(rank.device));
      HIP_CHECK(hipEventRecord(rank.start, rank.stream));
    }
    const auto host_begin = std::chrono::steady_clock::now();
    for (size_t i = 0; i < params.iterations; ++i) LaunchAll(ranks, chunk, ++epoch, fused);
    for (Rank& rank : ranks) {
      HIP_CHECK(hipSetDevice(rank.device));
      HIP_CHECK(hipEventRecord(rank.stop, rank.stream));
    }
    SyncAll(ranks);
    const double host_us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - host_begin)
            .count() /
        static_cast<double>(params.iterations);
    world.Barrier();

    double sum_us = 0.0;
    double max_us = 0.0;
    for (const Rank& rank : ranks) {
      float ms = 0.0f;
      HIP_CHECK(hipEventElapsedTime(&ms, rank.start, rank.stop));
      const double us = ms * 1000.0 / static_cast<double>(params.iterations);
      sum_us += us;
      max_us = std::max(max_us, us);
    }
    const double device_us = world.Sum(sum_us) / static_cast<double>(num_gpus);
    const double device_max_us = world.Max(max_us);
    const double host_mean_us = world.Sum(host_us) / static_cast<double>(world.size);
    const double alg_bw = static_cast<double>(total) / kGiB / (device_us / 1.0e6);
    const double bus_bw =
        alg_bw * static_cast<double>(num_gpus - 1) / static_cast<double>(num_gpus);

    if (world.IsRoot()) {
      const std::string error_text = verified ? std::to_string(errors) : std::string("N/A");
      std::cout << std::left << std::setw(14) << total << std::setw(18) << chunk << std::setw(14)
                << Fixed2(device_us) << std::setw(14) << Fixed2(device_max_us) << std::setw(14)
                << Fixed2(host_mean_us) << std::setw(14) << Fixed2(alg_bw) << std::setw(14)
                << Fixed2(bus_bw) << std::setw(8) << error_text << "\n";
      csv->WriteRow(num_gpus, total, chunk, Fixed2(device_us), Fixed2(device_max_us),
                    Fixed2(host_mean_us), Fixed2(alg_bw), Fixed2(bus_bw), error_text);
    }
    if (errors != 0) {
      if (world.IsRoot()) std::cerr << "Data verification failed\n";
      return EXIT_FAILURE;
    }
  }

  for (Rank& rank : ranks) {
    HIP_CHECK(hipSetDevice(rank.device));
    for (void* mapped : rank.mapped) HIP_CHECK(hipIpcCloseMemHandle(mapped));
    HIP_CHECK(hipEventDestroy(rank.start));
    HIP_CHECK(hipEventDestroy(rank.stop));
    HIP_CHECK(hipStreamDestroy(rank.stream));
    HIP_CHECK(hipHostFree(rank.args.statuses));
    FreeSignals(rank.arrivals, SignalMemory::kDevice);
    FreeSignals(rank.barriers, SignalMemory::kDevice);
    FreeSignals(rank.args.sends_done, SignalMemory::kDevice);
    HIP_CHECK(hipFree(rank.args.buffers[rank.args.rank]));
  }
  return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
  World world;
#ifdef COPY_ENGINE_BENCH_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &world.rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world.size);
#endif

  Params params;
  ArgParser parser("copy-engine-allgather",
                   "In-place allgather across GPUs through the HIP copy-engine API.");
  parser.AddOption("-g,--numGpus", &params.num_gpus,
                   "GPUs taking part, 0 for all (at most 16); with MPI the number of ranks");
  parser.AddOption("-b,--minBytes", &params.min_bytes, "Smallest total buffer size in bytes");
  parser.AddOption("-e,--maxBytes", &params.max_bytes, "Largest total buffer size in bytes");
  parser.AddOption("-f,--stepFactor", &params.step_factor, "Factor between sizes");
  parser.AddOption("-w,--warmup", &params.warmup, "Warmup iterations");
  parser.AddOption("-n,--iterations", &params.iterations, "Measured iterations");
  parser.AddOption("--submit", &params.submit,
                   "How each copy's two signals are queued: separate (a second submit) or fused "
                   "(hipExtCopyEngineSubmitLinearTwoCompletions)");
  parser.AddOption(
      "--signalOp", &params.signal_op,
      "Signal operation: add (1, one atomic packet) or set (the epoch, two fences on gfx9)");
  parser.AddFlag("--skip-barrier", &params.skip_barrier,
                 "Do not start every collective with a barrier over all ranks");
  parser.AddFlag("--skip-verification", &params.skip_verification, "Skip verification");
  parser.AddOption("-o,--outputFile", &params.output_file, "Filename for CSV results");

  int exit_code = 0;
  int result = 0;
  if (!parser.Parse(argc, argv, &exit_code)) {
    result = exit_code;
  } else if (!ValidateSweep(params.min_bytes, params.max_bytes)) {
    result = 2;
  } else if (params.signal_op != "set" && params.signal_op != "add") {
    std::cerr << "error: --signalOp must be 'set' or 'add'\n";
    result = 2;
  } else if (params.submit != "separate" && params.submit != "fused") {
    std::cerr << "error: --submit must be 'separate' or 'fused'\n";
    result = 2;
  } else if (params.step_factor < 2 || params.iterations == 0) {
    std::cerr << "error: --stepFactor must be at least 2 and --iterations non-zero\n";
    result = 2;
  } else {
    result = Run(params, world);
  }
#ifdef COPY_ENGINE_BENCH_MPI
  if (result != 0) MPI_Abort(MPI_COMM_WORLD, result);
  MPI_Finalize();
#endif
  return result;
}
