// hip_matmul_demo.cpp
//
// Streaming callstack demo over a HIP tiled SGEMM, so host stacks pass
// through real HIP runtime calls (hipMemcpy, hipLaunchKernelGGL, ...).
// upload_inputs and launch_matmul each fold an always_inline chain into one
// physical frame; --inline-frames expands it back via addr2line -i.
//
// The raw stream is also written to /tmp/hip_matmul.csnp. Symbolization runs
// in-process against /proc/self/maps; offline symbolization of a .csnp
// captured elsewhere is not implemented.
//
// Usage: ./hip_matmul_demo [/tmp/out.pftrace] [--backend auto|perf|signal]
//                          [--inline-frames] [--blas]
//                          [--freq=N] [--unwind=fp|dwarf] [--no-sampler]
//                          [--stack-bytes=N] [--pages=N] [--filter-stdlib]
//                          [--no-merge]
//   Without --unwind the sampler picks the mode (perf: frame pointers,
//   signal: DWARF). --stack-bytes is the per-sample user-stack snapshot used
//   by DWARF unwinding; --pages is the perf ring size per thread and must
//   hold several snapshots.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>
#include <sys/types.h>
#include <unistd.h>

#if CPU_SNAP_HAVE_HIPBLAS
#include <hipblas/hipblas.h>
#endif

#include "backend_select.hpp"
#include "cpu_snap/core/format_hex.hpp"
#include "cpu_snap/sampling/sampler.hpp"
#include "cpu_snap/processing/symbols/frame_filters.hpp"
#include "cpu_snap/processing/symbols/symbolizer.hpp"
#include "cpu_snap/core/memory_sink.hpp"
#include "cpu_snap/core/sample_reconstructor.hpp"
#include "cpu_snap/sampling/thread_comm.hpp"
#include "perfetto_proto_writer.hpp"
#include "cpu_snap/processing/timeline/slice_emitter.hpp"
#include "symbol_preload.hpp"

namespace
{

constexpr int                   k_matrix_n           = 1024;
constexpr int                   k_tile               = 16;
constexpr int                   k_launches_per_pass  = 8;
constexpr std::chrono::seconds  k_run_duration{ 4 };
constexpr std::uint32_t         k_sample_freq_hz     = 1000;
constexpr std::uint32_t         k_callchain_depth    = 128;
constexpr const char*           k_stream_path        = "/tmp/hip_matmul.csnp";

/** Streams bytes to a file AND a bounded in-memory sink, as required for
 *  the offline reader->symbolizer->perfetto pipeline used after stop(). */
struct dual_sink
{
    std::FILE*                  file = nullptr;
    cpu_snap::core::memory_sink mem;
    std::mutex                  mu;

    dual_sink(const char* path, std::size_t max_bytes)
        : file(std::fopen(path, "wb"))
        , mem(cpu_snap::core::memory_sink::config{ .max_bytes = max_bytes })
    {
        if(file == nullptr)
        {
            throw std::runtime_error(std::string("fopen failed: ") + path);
        }
    }

    ~dual_sink()
    {
        if(file != nullptr) std::fclose(file);
    }

    dual_sink(const dual_sink&)            = delete;
    dual_sink& operator=(const dual_sink&) = delete;

    void operator()(std::span<const std::byte> chunk)
    {
        const std::lock_guard<std::mutex> lk(mu);
        if(file != nullptr)
        {
            std::fwrite(chunk.data(), 1, chunk.size(), file);
        }
        mem(chunk);
    }
};

void
hip_check(hipError_t err, const char* what)
{
    if(err != hipSuccess)
    {
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(err));
    }
}

__global__ void
sgemm_tiled_kernel(const float* a, const float* b, float* c, int n)
{
    __shared__ float tile_a[k_tile][k_tile];
    __shared__ float tile_b[k_tile][k_tile];

    const int row = static_cast<int>(blockIdx.y) * k_tile + static_cast<int>(threadIdx.y);
    const int col = static_cast<int>(blockIdx.x) * k_tile + static_cast<int>(threadIdx.x);

    float acc = 0.0f;
    for(int t = 0; t < n / k_tile; ++t)
    {
        tile_a[threadIdx.y][threadIdx.x] = a[row * n + t * k_tile + threadIdx.x];
        tile_b[threadIdx.y][threadIdx.x] = b[(t * k_tile + threadIdx.y) * n + col];
        __syncthreads();
        for(int k = 0; k < k_tile; ++k)
        {
            acc += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    c[row * n + col] = acc;
}

struct host_matrices
{
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> c;
};

struct device_buffers
{
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
};

[[nodiscard, gnu::noinline]] host_matrices
init_host_matrices(int n)
{
    host_matrices m{ .a = std::vector<float>(static_cast<std::size_t>(n) * n),
                      .b = std::vector<float>(static_cast<std::size_t>(n) * n),
                      .c = std::vector<float>(static_cast<std::size_t>(n) * n, 0.0f) };
    for(int i = 0; i < n * n; ++i)
    {
        m.a[static_cast<std::size_t>(i)] = static_cast<float>((i % 13) - 6) * 0.1f;
        m.b[static_cast<std::size_t>(i)] = static_cast<float>((i % 7) - 3) * 0.1f;
    }
    return m;
}

[[nodiscard, gnu::noinline]] device_buffers
allocate_device_buffers(int n)
{
    device_buffers d;
    const std::size_t bytes = static_cast<std::size_t>(n) * n * sizeof(float);
    hip_check(hipMalloc(&d.a, bytes), "hipMalloc A");
    hip_check(hipMalloc(&d.b, bytes), "hipMalloc B");
    hip_check(hipMalloc(&d.c, bytes), "hipMalloc C");
    return d;
}

[[gnu::noinline]] void
free_device_buffers(device_buffers& d)
{
    hip_check(hipFree(d.a), "hipFree A");
    hip_check(hipFree(d.b), "hipFree B");
    hip_check(hipFree(d.c), "hipFree C");
    d = {};
}

// Deliberately-inlined sub-chain of upload_inputs: stage_matrix ->
// copy_to_device -> hipMemcpy. At -O2 these fold into one physical
// frame inside upload_inputs; --inline-frames asks the symbolizer to
// expand that frame back into this source-level chain via addr2line -i.
[[gnu::always_inline]] inline void
copy_to_device(float* dst, const float* src, std::size_t bytes)
{
    hip_check(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice), "hipMemcpy H2D");
}

[[gnu::always_inline]] inline void
stage_matrix(float* dst, const std::vector<float>& src)
{
    copy_to_device(dst, src.data(), src.size() * sizeof(float));
}

[[gnu::noinline]] void
upload_inputs(const device_buffers& d, const host_matrices& m, [[maybe_unused]] int n)
{
    stage_matrix(d.a, m.a);
    stage_matrix(d.b, m.b);
}

// Deliberately-inlined sub-chain of launch_matmul: compute_grid ->
// enqueue_kernel -> hipLaunchKernelGGL.
[[gnu::always_inline]] inline void
enqueue_kernel(const device_buffers& d, int n, hipStream_t stream, dim3 grid, dim3 block)
{
    hipLaunchKernelGGL(sgemm_tiled_kernel, grid, block, 0, stream, d.a, d.b, d.c, n);
}

[[gnu::always_inline]] inline void
compute_grid(const device_buffers& d, int n, hipStream_t stream)
{
    const dim3 block(k_tile, k_tile);
    const dim3 grid(static_cast<unsigned int>(n / k_tile), static_cast<unsigned int>(n / k_tile));
    for(int i = 0; i < k_launches_per_pass; ++i)
    {
        enqueue_kernel(d, n, stream, grid, block);
    }
}

[[gnu::noinline]] void
launch_matmul(const device_buffers& d, int n, hipStream_t stream)
{
    compute_grid(d, n, stream);
    hip_check(hipGetLastError(), "sgemm_tiled_kernel launch");
}

#if CPU_SNAP_HAVE_HIPBLAS
[[gnu::noinline]] void
launch_matmul_blas(hipblasHandle_t handle, const device_buffers& d, int n)
{
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    for(int i = 0; i < k_launches_per_pass; ++i)
    {
        const hipblasStatus_t st = hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n,
                                                 &alpha, d.a, n, d.b, n, &beta, d.c, n);
        if(st != HIPBLAS_STATUS_SUCCESS)
        {
            throw std::runtime_error("hipblasSgemm failed");
        }
    }
}
#endif

[[gnu::noinline]] void
wait_for_gpu(hipStream_t stream)
{
    hip_check(hipStreamSynchronize(stream), "hipStreamSynchronize");
}

[[gnu::noinline]] void
download_result(const device_buffers& d, host_matrices& m, int n)
{
    const std::size_t bytes = static_cast<std::size_t>(n) * n * sizeof(float);
    hip_check(hipMemcpy(m.c.data(), d.c, bytes, hipMemcpyDeviceToHost), "hipMemcpy C D2H");
}

[[nodiscard, gnu::noinline]] float
verify_result(const host_matrices& m, int n)
{
    constexpr std::array<int, 5> k_probe_rows{ 0, 1, 7, 200, 1023 };
    float max_abs_err = 0.0f;
    for(const int row : k_probe_rows)
    {
        if(row >= n) continue;
        const int col = (row * 31 + 17) % n;
        float     ref = 0.0f;
        for(int k = 0; k < n; ++k)
        {
            ref += m.a[static_cast<std::size_t>(row) * n + static_cast<std::size_t>(k)] *
                   m.b[static_cast<std::size_t>(k) * n + static_cast<std::size_t>(col)];
        }
        const float got = m.c[static_cast<std::size_t>(row) * n + static_cast<std::size_t>(col)];
        max_abs_err      = std::max(max_abs_err, std::fabs(got - ref));
    }
    return max_abs_err;
}

struct matmul_result
{
    std::uint64_t passes    = 0;
    float         max_error = 0.0f;
};

[[nodiscard, gnu::noinline]] matmul_result
run_matmul(int n, std::chrono::seconds duration, [[maybe_unused]] bool use_blas)
{
    matmul_result  result;
    host_matrices  host   = init_host_matrices(n);
    device_buffers device = allocate_device_buffers(n);

    hipStream_t stream = nullptr;
    hip_check(hipStreamCreate(&stream), "hipStreamCreate");

#if CPU_SNAP_HAVE_HIPBLAS
    hipblasHandle_t blas_handle = nullptr;
    if(use_blas)
    {
        if(hipblasCreate(&blas_handle) != HIPBLAS_STATUS_SUCCESS)
        {
            throw std::runtime_error("hipblasCreate failed");
        }
        if(hipblasSetStream(blas_handle, stream) != HIPBLAS_STATUS_SUCCESS)
        {
            throw std::runtime_error("hipblasSetStream failed");
        }
    }
#endif

    const auto deadline = std::chrono::steady_clock::now() + duration;
    while(std::chrono::steady_clock::now() < deadline)
    {
        upload_inputs(device, host, n);
#if CPU_SNAP_HAVE_HIPBLAS
        if(use_blas)
        {
            launch_matmul_blas(blas_handle, device, n);
        }
        else
        {
            launch_matmul(device, n, stream);
        }
#else
        launch_matmul(device, n, stream);
#endif
        wait_for_gpu(stream);
        download_result(device, host, n);
        ++result.passes;
    }

    result.max_error = verify_result(host, n);

#if CPU_SNAP_HAVE_HIPBLAS
    if(use_blas) hip_check(static_cast<hipError_t>(hipblasDestroy(blas_handle)), "hipblasDestroy");
#endif
    hip_check(hipStreamDestroy(stream), "hipStreamDestroy");
    free_device_buffers(device);
    return result;
}

struct cli_options
{
    cs_examples::backend_selection selection;
    bool          inline_frames = false;
    bool          use_blas      = false;
    bool          sampling      = true;
    bool          filter_stdlib = false;
    bool          merge_samples = true;
    std::uint32_t freq_hz       = k_sample_freq_hz;
    cpu_snap::sampling::unwind_mode unwind = cpu_snap::sampling::unwind_mode::auto_best;
    std::uint32_t stack_bytes = cpu_snap::sampling::sampler::perf_attr_options{}.dwarf_stack_bytes;
    std::uint32_t pages       = 0;  // 0 = sized from the DWARF snapshot
};

[[nodiscard]] cli_options
parse_cli(int argc, char** argv)
{
    cli_options opts;
    opts.selection = cs_examples::parse_backend(argc, argv);
    for(int i = 1; i < argc; ++i)
    {
        const std::string_view arg = argv[i];
        if(arg == "--inline-frames") opts.inline_frames = true;
        else if(arg == "--blas") opts.use_blas = true;
        else if(arg == "--no-sampler") opts.sampling = false;
        else if(arg == "--filter-stdlib") opts.filter_stdlib = true;
        else if(arg == "--no-merge") opts.merge_samples = false;
        else if(arg == "--unwind=dwarf") opts.unwind = cpu_snap::sampling::unwind_mode::dwarf;
        else if(arg == "--unwind=fp") opts.unwind = cpu_snap::sampling::unwind_mode::frame_pointer;
        else if(arg.starts_with("--stack-bytes="))
        {
            opts.stack_bytes = static_cast<std::uint32_t>(std::strtoul(argv[i] + 14, nullptr, 10));
        }
        else if(arg.starts_with("--pages="))
        {
            opts.pages = static_cast<std::uint32_t>(std::strtoul(argv[i] + 8, nullptr, 10));
        }
        else if(arg.starts_with("--freq="))
        {
            opts.freq_hz = static_cast<std::uint32_t>(std::strtoul(argv[i] + 7, nullptr, 10));
        }
    }
    return opts;
}

}  // namespace

int
main(int argc, char** argv)
{
    try
    {
        const cli_options opts = parse_cli(argc, argv);
#if CPU_SNAP_HAVE_HIPBLAS
        if(opts.use_blas) std::printf("matmul path: hipblasSgemm\n");
#else
        if(opts.use_blas)
        {
            std::printf("--blas requested but hipBLAS was not found at configure time; "
                        "falling back to the custom kernel\n");
        }
#endif

        if(!opts.sampling)
        {
            const matmul_result result = run_matmul(k_matrix_n, k_run_duration, opts.use_blas);
            std::printf("passes=%lu (no sampler)\n", static_cast<unsigned long>(result.passes));
            return 0;
        }

        const std::string out_path =
            (opts.selection.output_path != nullptr) ? opts.selection.output_path
                                                      : "/tmp/hip_matmul.pftrace";

        constexpr std::size_t k_memory_sink_max_bytes = 256U * 1024U * 1024U;

        dual_sink sink(k_stream_path, k_memory_sink_max_bytes);

        cpu_snap::sampling::sampler::config cfg{
            .perf      = { .sample_freq_hz       = opts.freq_hz,
                           .callchain_depth      = k_callchain_depth,
                           .kernel_pages_per_tid = opts.pages,
                           .dwarf_stack_bytes    = opts.stack_bytes,
                           .unwind               = opts.unwind },
            .threads   = { .watch_new_threads = true },
            .streaming = { .sink = std::ref(sink), .process_name = "hip_matmul_demo" },
        };
        cfg.backend = opts.selection.backend;
        if(opts.selection.backend != cpu_snap::sampling::backend_kind::perf)
        {
            cfg.signal = cs_examples::make_signal_options(opts.freq_hz);
            cfg.signal.stack_snapshot_bytes = opts.stack_bytes;
        }

        cpu_snap::sampling::sampler rec(cfg);
        opts.selection.print(rec);
        rec.start();
        const matmul_result result = run_matmul(k_matrix_n, k_run_duration, opts.use_blas);
        rec.stop();

        const double flops_per_pass =
            2.0 * k_matrix_n * k_matrix_n * k_matrix_n * k_launches_per_pass;
        const double gflops = (flops_per_pass * static_cast<double>(result.passes)) /
                              (std::chrono::duration<double>(k_run_duration).count() * 1e9);
        std::printf("passes=%lu total_samples=%zu lost=%zu stream_bytes=%zu (%s, dropped %zu chunks)\n",
                    static_cast<unsigned long>(result.passes), rec.total_samples(),
                    rec.lost_samples(), sink.mem.bytes().size(), k_stream_path,
                    sink.mem.dropped_chunks());
        std::printf("verify: max_abs_error=%g (%s)\n", static_cast<double>(result.max_error),
                    result.max_error < 1.0f ? "OK" : "FAIL");
        std::printf("throughput: %.2f GFLOP/s\n", gflops);

        // Offline replay: decode the in-memory byte stream, rebuild
        // stacks and the module map, symbolize in this process.
        cpu_snap::core::sample_reconstructor reconstructor;
        reconstructor.feed(sink.mem.bytes());
        const auto& per_tid_samples = reconstructor.samples();

        std::size_t n_sample = 0;
        for(const auto& [tid, vec] : per_tid_samples)
        {
            (void) tid;
            n_sample += vec.size();
        }

        const auto& rc = reconstructor.counters();
        std::printf("reader: bytes_fed=%lu records_emitted=%lu parse_errors=%lu gap_count=%lu "
                    "| decoded_samples=%zu | threads=%zu\n",
                    static_cast<unsigned long>(rc.bytes_fed),
                    static_cast<unsigned long>(rc.records_emitted),
                    static_cast<unsigned long>(rc.parse_errors),
                    static_cast<unsigned long>(rc.gap_count), n_sample, per_tid_samples.size());
        if(n_sample > 0)
        {
            std::printf("bytes/sample: %.2f\n",
                        static_cast<double>(sink.mem.bytes().size()) / static_cast<double>(n_sample));
        }

        cpu_snap::processing::symbols::symbolizer sym(
            cpu_snap::processing::symbols::symbolizer_config{
                .include_binary_name = true,
                .inline_frames       = opts.inline_frames,
            });
        cs_examples::preload_symbols_from(sym, per_tid_samples);

        // Top leaf frames + deepest stack, printed for the acceptance report.
        {
            std::unordered_map<std::uint64_t, std::size_t> leaf_counts;
            for(const auto& [tid, samples] : per_tid_samples)
            {
                for(const auto& s : samples)
                {
                    if(s.frames.size() == 0) continue;
                    ++leaf_counts[s.frames[0]];
                }
            }
            std::vector<std::pair<std::uint64_t, std::size_t>> ranked(leaf_counts.begin(),
                                                                        leaf_counts.end());
            std::ranges::sort(ranked,
                               [](const auto& a, const auto& b) { return a.second > b.second; });
            const std::size_t shown = std::min<std::size_t>(15, ranked.size());
            std::printf("top %zu leaf frames:\n", shown);
            for(std::size_t i = 0; i < shown; ++i)
            {
                std::printf("  %6zu  %s\n", ranked[i].second,
                            std::string(sym.resolve(ranked[i].first)).c_str());
            }

            const cpu_snap::core::raw_sample* deepest = nullptr;
            for(const auto& [tid, samples] : per_tid_samples)
            {
                for(const auto& s : samples)
                {
                    if(deepest == nullptr || s.frames.size() > deepest->frames.size()) deepest = &s;
                }
            }
            if(deepest != nullptr)
            {
                std::printf("deepest stack (%u frames, leaf first):\n", deepest->frames.size());
                for(std::uint32_t i = 0; i < deepest->frames.size(); ++i)
                {
                    std::printf("  #%-2u %s\n", i,
                                std::string(sym.resolve(deepest->frames[i])).c_str());
                }
                if(opts.inline_frames)
                {
                    std::printf("deepest stack, inline-expanded (leaf ip only):\n");
                    for(const auto& frame : sym.resolve_inline(deepest->frames[0]))
                    {
                        std::printf("    inline: %s\n", std::string(frame).c_str());
                    }
                }
            }
        }

        cs_examples::perfetto::perfetto_proto_writer writer(out_path);
        writer.set_process(::getpid(), "hip_matmul_demo");
        std::unordered_set<pid_t> registered;
        for(const auto& [tid, _] : per_tid_samples)
        {
            if(registered.insert(tid).second)
            {
                writer.add_thread(static_cast<int>(tid), cpu_snap::sampling::utils::read_thread_comm(tid));
            }
        }

        cpu_snap::processing::timeline::slice_emitter_config scfg;
        if(opts.inline_frames)
        {
            scfg.inline_resolver = [&sym](std::uint64_t ip) {
                auto chain = sym.resolve_inline(ip);
                if(chain.empty())
                {
                    chain.emplace_back(cpu_snap::core::format_hex(ip));
                }
                return chain;
            };
        }
        else
        {
            scfg.resolver = [&sym](std::uint64_t ip) {
                const auto label = sym.resolve(ip);
                if(!label.empty()) return std::string(label);
                return cpu_snap::core::format_hex(ip);
            };
        }
        scfg.symbol_identity = [&sym](std::uint64_t ip) { return sym.symbol_identity(ip); };
        scfg.merge_samples = opts.merge_samples;
        if(opts.filter_stdlib)
        {
            scfg.library_frame_filter = cpu_snap::processing::symbols::make_stdlib_exclude_filter(
                [&sym](std::uint64_t ip) { return sym.resolve(ip); });
        }
        cpu_snap::processing::timeline::slice_emitter emitter(std::move(scfg));
        emitter.process(per_tid_samples, [&](const cpu_snap::processing::timeline::emitted_slice& s) {
            writer.write_slice(s);
        });
        writer.finalize();

        std::printf("wrote %s\nraw stream: %s\nopen in: https://ui.perfetto.dev/\n",
                    out_path.c_str(), k_stream_path);
        return 0;
    } catch(const std::exception& e)
    {
        std::fprintf(stderr, "%s: %s\n", argc > 0 ? argv[0] : "(unknown)", e.what());
        return EXIT_FAILURE;
    }
}
