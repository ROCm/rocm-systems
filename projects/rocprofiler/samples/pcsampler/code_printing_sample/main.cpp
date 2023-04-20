/* Copyright (c) 2022 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include <algorithm>
#include <chrono>
#include <memory>
#include <numeric>
#include <vector>
#include <cfloat>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>

#include <unistd.h>

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>

#include <rocprofiler/v2/rocprofiler.h>

#include "program.hpp"
#include "program_options.hpp"
#include "disassembly.hpp"

#define XSTR(x) STR(x)
#define STR(x) #x
#define DBL_FMT "." XSTR(DBL_DECIMAL_DIG) "f"

namespace util {

struct hipMalloc_freer {
    void operator()(void * const ptr) { (void)hipFree(ptr); }
};

} // namespace util

namespace prng {

static uint64_t
splitmix64_next(uint64_t * const sm64_state)
{
    uint64_t z = (*sm64_state += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

static inline uint64_t
rotl64(const uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

static uint64_t
xrs_next(uint64_t * const xrs_state)
{
    const uint64_t result =
      rotl64(xrs_state[0] + xrs_state[3], 23) + xrs_state[0];

    const uint64_t t = xrs_state[1] << 17;

    xrs_state[2] ^= xrs_state[0];
    xrs_state[3] ^= xrs_state[1];
    xrs_state[1] ^= xrs_state[2];
    xrs_state[0] ^= xrs_state[3];

    xrs_state[2] ^= t;

    xrs_state[3] = rotl64(xrs_state[3], 45);

    return result;
}

} // namespace prng

namespace kernel {

template <typename T>
__global__ static void
memset_gpu(T * const s, T const c, size_t const n)
{
    size_t i_start = threadIdx.x + blockIdx.x * blockDim.x;
    size_t i_shift = blockDim.x * gridDim.x;
    for (size_t i = i_start; i < n; i += i_shift) {
        s[i] = c;
    }
}

template <typename T>
__global__ static void
count_gpu(
  T const * const xs,
  T * const out,
  size_t const n,
  size_t const nblocks,
  T const gt)
{
    size_t i_start = threadIdx.x + blockIdx.x * blockDim.x;
    size_t i_shift = blockDim.x * gridDim.x;
    for (size_t i = i_start; i < n; i += i_shift) {
        if (xs[i] > gt) {
            atomicAdd(&out[i % nblocks], 1);
        }
    }
}

} // namespace kernel

static char const GETOPT_ARGS[] = "cd:mn:DP";

static void
usage()
{
    fputs("usage: " PROGNAME " [OPTION]... MIN [SEED]\n"
          "  -d DEV\tHIP device number\n"
          "  -n LEN\tLength of random integer array\n"
          "  -D\t\tPrint kernel disassembly\n"
          "  -P\t\tPrint source and disassembly of sampled PC locations\n"
          "where\n"
          "  DEV : i32\n"
          "  MIN : u64\n"
          "  LEN : u64\n"
          "  SEED : u64\n",
          stderr);
}

static int
get_options(int argc, char **argv, program_options * const opts)
{
    int opt;

    while (-1 != (opt = getopt(argc, argv, GETOPT_ARGS))) {
        switch (opt) {
        case 'd':
            // TODO error checking
            opts->device = strtol(optarg, nullptr, 10);
            break;
        case 'n':
            // TODO error checking
            opts->rands_len = strtoul(optarg, nullptr, 10);
            break;
        case 'D':
            opts->disassemble = true;
            break;
        case 'P':
            opts->pc_sampling = true;
            break;
        default:
            usage();
            return EXIT_FAILURE;
        }
    }

    auto const optcount = argc - optind;
    if (!(1 == optcount || 2 == optcount)) {
        usage();
        return EXIT_FAILURE;
    }

    // TODO error checking
    opts->gt = strtoul(argv[optind], nullptr, 10);
    if (2 == argc - optind) {
        opts->seed = strtoull(argv[optind + 1], nullptr, 10);
    }

    return EXIT_SUCCESS;
}

static program_options g_opts;

static void
callback_flush_fn(
  rocprofiler_record_header_t const *record,
  rocprofiler_record_header_t const *end_record,
  rocprofiler_session_id_t session_id,
  rocprofiler_buffer_id_t buffer_id)
{
    while (record < end_record) {
        if (nullptr == record) {
            break;
        }
        if (ROCPROFILER_PC_SAMPLING_RECORD == record->kind) {
            auto const &pcr = (rocprofiler_record_pc_sample_t &)*record;
            printf(
              "dispatch[%" PRIu64 "] timestamp(%" PRIu64
              ") gpu_id(%#" PRIx64 ") pc-sample(%#" PRIx64
              ") se(%" PRIu32 ")\n",
              pcr.pc_sample.dispatch_id.value,
              pcr.pc_sample.timestamp.value,
              pcr.pc_sample.gpu_id.handle,
              pcr.pc_sample.pc,
              pcr.pc_sample.se);
            if (g_opts.pc_sampling) {
                disassembly_print_pc_sample_context(pcr.pc_sample.pc);
            }
        }
        rocprofiler_next_record(record, &record, session_id, buffer_id);
    }
}

static int
run_kernel(program_options const &opts)
{
    rocprofiler_session_id_t sid;
    rocprofiler_filter_id_t fid, fid2;
    rocprofiler_buffer_id_t bid;
    auto rocprofiler_ok = ROCPROFILER_STATUS_SUCCESS;

    if (opts.pc_sampling) {
        ROCPROFILER_CHECK(
          rocprofiler_create_session(ROCPROFILER_NONE_REPLAY_MODE, &sid),
          rocprofiler_ok);
        if (ROCPROFILER_STATUS_SUCCESS != rocprofiler_ok) {
            fputs("error: failed to create rocprofiler session\n", stderr);
            return EXIT_FAILURE;
        }

        rocprofiler_filter_property_t property{};

        ROCPROFILER_CHECK(
          rocprofiler_create_buffer(
            sid, callback_flush_fn, static_cast<size_t>(0x1000), &bid),
          rocprofiler_ok);
        if (ROCPROFILER_STATUS_SUCCESS != rocprofiler_ok) {
            fputs("error: failed to add PC sampling session mode\n", stderr);
            goto out;
        }

        ROCPROFILER_CHECK(
          rocprofiler_create_filter(
            sid, ROCPROFILER_PC_SAMPLING_COLLECTION,
            rocprofiler_filter_data_t{},
            0, &fid, property),
          rocprofiler_ok);
        if (ROCPROFILER_STATUS_SUCCESS != rocprofiler_ok) {
            goto cleanup;
        }

        ROCPROFILER_CHECK(
          rocprofiler_create_filter(
            sid, ROCPROFILER_DISPATCH_TIMESTAMPS_COLLECTION,
            rocprofiler_filter_data_t{},
            0, &fid2, property),
          rocprofiler_ok);
        if (ROCPROFILER_STATUS_SUCCESS != rocprofiler_ok) {
            goto cleanup;
        }

        ROCPROFILER_CHECK(
          rocprofiler_set_filter_buffer(sid, fid, bid),
          rocprofiler_ok);
        if (ROCPROFILER_STATUS_SUCCESS != rocprofiler_ok) {
            goto cleanup;
        }

        ROCPROFILER_CHECK(
          rocprofiler_set_filter_buffer(sid, fid2, bid),
          rocprofiler_ok);
        if (ROCPROFILER_STATUS_SUCCESS != rocprofiler_ok) {
            goto cleanup;
        }

        ROCPROFILER_CHECK(
          rocprofiler_start_session(sid),
          rocprofiler_ok);
        if (ROCPROFILER_STATUS_SUCCESS != rocprofiler_ok) {
            goto cleanup;
        }
    }

    {
    printf("seed = %" PRIu64 "\n", opts.seed);

    std::vector<uint64_t> rands(opts.rands_len);
    using rands_elt_t = decltype(rands)::value_type;

    uint64_t
      sm64_state = opts.seed,
      xrs_state[4];

    {
        using prng::splitmix64_next;
        using prng::xrs_next;

        // Initialize the Xoroshiro PRNG
        xrs_state[0] = splitmix64_next(&sm64_state);
        xrs_state[1] = splitmix64_next(&sm64_state);
        xrs_state[2] = splitmix64_next(&sm64_state);
        xrs_state[3] = splitmix64_next(&sm64_state);

        // Fill rands with random integers
        for (auto &i : rands) {
            i = xrs_next(xrs_state);
        }
    }

    struct tm {
        using monoclk = std::chrono::steady_clock;
        using dur = std::chrono::duration<double>;
    };

    using util::hipMalloc_freer;

    auto const begin_time = tm::monoclk::now();

    auto hip_ok = hipSuccess;
    do {
        HIP_CHECK_BREAK(hipSetDevice(opts.device), hip_ok);

        auto const rands_nbytes = rands.size() * sizeof(rands_elt_t);
        std::unique_ptr<rands_elt_t, hipMalloc_freer> rands_gpu;
        {
            rands_elt_t *rands_gpu_ptr;
            HIP_CHECK_BREAK(hipMalloc(&rands_gpu_ptr, rands_nbytes), hip_ok);
            rands_gpu.reset(rands_gpu_ptr);
        }

        HIP_CHECK_BREAK(
          hipMemcpy(rands_gpu.get(), rands.data(), rands_nbytes,
                    hipMemcpyHostToDevice),
          hip_ok);
        (void)hipDeviceSynchronize();

        uint32_t constexpr nthreads = 256U;
        uint32_t const nblocks = (rands.size() + nthreads - 1) / nthreads;

        using count_elt_t = size_t;

        auto const count_subtotals_nbytes = nblocks * sizeof(count_elt_t);
        std::unique_ptr<count_elt_t, hipMalloc_freer> count_subtotals_gpu;
        {
            count_elt_t *count_subtotals_gpu_ptr;
            HIP_CHECK_BREAK(
              hipMalloc(&count_subtotals_gpu_ptr, count_subtotals_nbytes),
              hip_ok);
            count_subtotals_gpu.reset(count_subtotals_gpu_ptr);
        }

        hipLaunchKernelGGL(
          kernel::memset_gpu, nblocks, nthreads, 0, 0,
          count_subtotals_gpu.get(), 0UL, static_cast<size_t>(nblocks));
        HIP_CHECK_BREAK(hipGetLastError(), hip_ok);
        (void)hipDeviceSynchronize();

        auto const kernel_begin_time = tm::monoclk::now();

        hipLaunchKernelGGL(
          kernel::count_gpu, nblocks, nthreads, 0, 0,
          rands_gpu.get(), count_subtotals_gpu.get(), rands.size(),
          static_cast<size_t>(nblocks), opts.gt);
        HIP_CHECK_BREAK(hipGetLastError(), hip_ok);
        (void)hipDeviceSynchronize();

        auto const kernel_end_time = tm::monoclk::now();

        std::vector<size_t> count_subtotals(nblocks);
        HIP_CHECK_BREAK(
          hipMemcpy(count_subtotals.data(), count_subtotals_gpu.get(),
                    count_subtotals_nbytes, hipMemcpyDeviceToHost),
          hip_ok);
        (void)hipDeviceSynchronize();

        // TODO parallel sum on GPU
        auto const total =
          std::accumulate(
            count_subtotals.cbegin(), count_subtotals.cend(),
            static_cast<size_t>(0));

        auto const all_end_time = tm::monoclk::now();

        tm::dur const kernel_time(kernel_end_time - kernel_begin_time);
        auto total_time(all_end_time - begin_time);
        tm::dur const total_time_without_tool_init(total_time);
        printf("len(rands) = %zu; gt = %zu; count(rands, gt) = %zu\n"
               "main kernel time elapsed: %" DBL_FMT "\n"
               "full time elapsed: %" DBL_FMT "\n",
               rands.size(), opts.gt, total,
               kernel_time.count(),
               total_time_without_tool_init.count());
    } while (false);

    if (opts.disassemble) {
        disassembly_disassemble_kernels(false);
    }
    }

cleanup:
    if (opts.pc_sampling) {
        rocprofiler_terminate_session(sid);
        rocprofiler_flush_data(sid, bid);
        rocprofiler_destroy_session(sid);
    }

out:
    return ROCPROFILER_STATUS_SUCCESS == rocprofiler_ok
      ? EXIT_SUCCESS
      : EXIT_FAILURE;
}

int
main(int argc, char **argv)
{
    if (auto const ret = get_options(argc, argv, &g_opts);
        EXIT_SUCCESS != ret)
    {
        return ret;
    }

    if (hsa_init() != HSA_STATUS_SUCCESS){
        return EXIT_FAILURE;
    }

    int ret = EXIT_FAILURE;
    auto ok = ROCPROFILER_STATUS_SUCCESS;

    ROCPROFILER_CHECK(rocprofiler_initialize(), ok);
    if (ROCPROFILER_STATUS_SUCCESS == ok) {
        ret = run_kernel(g_opts);
    } else {
        goto out;
    }

    rocprofiler_finalize();

out:
    hsa_shut_down();
    return ROCPROFILER_STATUS_SUCCESS == ok && EXIT_FAILURE != ret
      ? EXIT_SUCCESS
      : EXIT_FAILURE;
}
