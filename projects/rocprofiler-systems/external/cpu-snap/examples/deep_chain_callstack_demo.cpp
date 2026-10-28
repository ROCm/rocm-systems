// deep_chain_callstack_demo.cpp
//
// Streaming callstack demo with a deep (8-frame), non-inlined call chain
// where each frame does a different amount of its own work, so one capture
// shows self time growing towards the leaf and inclusive time towards the root.
//
// The busy-spin lives in one shared noinline busy_spin_for() helper so the
// innermost frame is a single stable symbol; a frame's self time is the
// samples whose caller of busy_spin_for is that frame.
//
// Chain (root -> leaf), own busy-spin time per call:
//   handle_request ~1 ms, parse_request ~2 ms, validate_payload ~4 ms,
//   normalize_fields ~8 ms, compute_pricing ~16 ms, apply_discounts ~32 ms,
//   render_response ~64 ms, serialize_output ~128 ms (leaf)
//
// The raw wire-format stream also lands at /tmp/deep_chain.csnp.
//
// Usage: ./deep_chain_callstack_demo [/tmp/out.pftrace] [--backend auto|perf|signal]

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <sys/types.h>
#include <unistd.h>

#include "backend_select.hpp"
#include "perfetto_proto_writer.hpp"
#include "cpu_snap/core/format_hex.hpp"
#include "cpu_snap/sampling/sampler.hpp"
#include "cpu_snap/processing/timeline/slice_emitter.hpp"
#include "cpu_snap/processing/symbols/symbolizer.hpp"
#include "cpu_snap/core/memory_sink.hpp"
#include "cpu_snap/core/sample_reconstructor.hpp"
#include "symbol_preload.hpp"

namespace
{

constexpr const char* k_stream_path = "/tmp/deep_chain.csnp";
constexpr int         k_iterations  = 3;

constexpr std::array<const char*, 8> k_chain_order = {
    "handle_request",   "parse_request",  "validate_payload", "normalize_fields",
    "compute_pricing",  "apply_discounts", "render_response",  "serialize_output",
};

[[gnu::noinline]] void
busy_spin_for(std::chrono::nanoseconds dur) noexcept
{
    const auto              start = std::chrono::steady_clock::now();
    volatile std::uint64_t  sink  = 0;
    while(std::chrono::steady_clock::now() - start < dur)
    {
        sink = sink + 1;
    }
}

[[gnu::noinline]] void
serialize_output() noexcept
{
    busy_spin_for(std::chrono::milliseconds(128));
}

[[gnu::noinline]] void
render_response() noexcept
{
    busy_spin_for(std::chrono::milliseconds(64));
    serialize_output();
}

[[gnu::noinline]] void
apply_discounts() noexcept
{
    busy_spin_for(std::chrono::milliseconds(32));
    render_response();
}

[[gnu::noinline]] void
compute_pricing() noexcept
{
    busy_spin_for(std::chrono::milliseconds(16));
    apply_discounts();
}

[[gnu::noinline]] void
normalize_fields() noexcept
{
    busy_spin_for(std::chrono::milliseconds(8));
    compute_pricing();
}

[[gnu::noinline]] void
validate_payload() noexcept
{
    busy_spin_for(std::chrono::milliseconds(4));
    normalize_fields();
}

[[gnu::noinline]] void
parse_request() noexcept
{
    busy_spin_for(std::chrono::milliseconds(2));
    validate_payload();
}

[[gnu::noinline]] void
handle_request() noexcept
{
    busy_spin_for(std::chrono::milliseconds(1));
    parse_request();
}

/** Streams bytes to a file AND a bounded in-memory sink (single-threaded
 *  demo, so no locking needed, unlike hip_matmul_demo.cpp's dual_sink). */
struct dual_sink
{
    std::FILE*                  file = nullptr;
    cpu_snap::core::memory_sink mem;

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
        std::fwrite(chunk.data(), 1, chunk.size(), file);
        mem(chunk);
    }
};

}  // namespace

int
main(int argc, char** argv)
{
    try
    {
        const cs_examples::backend_selection sel =
            cs_examples::parse_backend(argc, argv);

        const std::string out_path =
            (sel.output_path != nullptr) ? sel.output_path
                                         : "/tmp/deep_chain_callstack_streaming.pftrace";

        constexpr std::uint32_t k_sample_freq_hz        = 1000;
        constexpr std::size_t   k_memory_sink_max_bytes = 256U * 1024U * 1024U;

        dual_sink sink(k_stream_path, k_memory_sink_max_bytes);

        cpu_snap::sampling::sampler::config cfg{
            .perf      = { .sample_freq_hz       = k_sample_freq_hz,
                           .callchain_depth      = 32,
                           .kernel_pages_per_tid = 8,
                           .unwind               = cpu_snap::sampling::unwind_mode::frame_pointer },
            .threads   = { .watch_new_threads = false },
            .streaming = { .sink         = std::ref(sink),
                           .process_name = "deep_chain_callstack_streaming" },
        };
        cfg.backend = sel.backend;
        if(sel.backend != cpu_snap::sampling::backend_kind::perf)
        {
            cfg.signal = cs_examples::make_signal_options(k_sample_freq_hz);
        }

        cpu_snap::sampling::sampler rec(cfg);
        sel.print(rec);
        rec.start();
        for(int i = 0; i < k_iterations; ++i)
        {
            handle_request();
        }
        rec.stop();
        std::printf("total_samples=%zu, captured %zu bytes -> %s (dropped %zu chunks)\n",
                    rec.total_samples(), sink.mem.bytes().size(), k_stream_path,
                    sink.mem.dropped_chunks());

        cpu_snap::core::sample_reconstructor reconstructor;
        reconstructor.feed(sink.mem.bytes());
        const auto& per_tid = reconstructor.samples();
        std::printf("decoded %zu tids (parse_errors=%lu)\n", per_tid.size(),
                    static_cast<unsigned long>(reconstructor.counters().parse_errors));

        cpu_snap::processing::symbols::symbolizer sym(cpu_snap::processing::symbols::symbolizer_config{
            .include_binary_name = true,
        });

        cs_examples::preload_symbols_from(sym, per_tid);

        cs_examples::perfetto::perfetto_proto_writer writer(out_path);
        writer.set_process(::getpid(), "deep_chain_callstack_streaming");
        for(const auto& [tid, samples] : per_tid)
        {
            (void) samples;
            writer.add_thread(static_cast<int>(tid), "main");
        }

        cpu_snap::processing::timeline::slice_emitter_config scfg;
        scfg.resolver = [&sym](std::uint64_t ip) {
            const auto label = sym.resolve(ip);
            if(!label.empty()) return std::string(label);
            return cpu_snap::core::format_hex(ip);
        };
        scfg.symbol_identity = [&sym](std::uint64_t ip) { return sym.symbol_identity(ip); };
        cpu_snap::processing::timeline::slice_emitter emitter(std::move(scfg));
        emitter.process(per_tid, [&](const cpu_snap::processing::timeline::emitted_slice& s) {
            writer.write_slice(s);
        });
        writer.finalize();
        std::printf("wrote %s\n", out_path.c_str());

        // Per-frame self (exclusive) vs inclusive sample counts, aggregated
        // by chain function name. A sample's leaf frame is almost never
        // one of the 8 chain functions themselves (it lands inside
        // busy_spin_for's steady_clock::now()/clock_gettime/vDSO call
        // chain instead), so "self" attribution walks up from the leaf
        // and credits the FIRST frame that is one of the 8 known chain
        // functions; "inclusive" counts every sample where a chain
        // function appears anywhere in the stack. sample_period_ns
        // matches the sampler's configured 1 kHz rate.
        constexpr std::uint64_t                    k_sample_period_ns = 1'000'000ULL;
        std::unordered_map<const char*, std::uint64_t> self_count;
        std::unordered_map<const char*, std::uint64_t> inclusive_count;
        std::unordered_map<const char*, std::uint64_t> representative_ip;

        auto canonical_chain_name = [](std::string_view resolved) -> const char* {
            for(const char* fn : k_chain_order)
            {
                const std::string_view fn_view(fn);
                if(resolved.size() >= fn_view.size()
                   && resolved.compare(0, fn_view.size(), fn_view) == 0)
                {
                    return fn;
                }
            }
            return nullptr;
        };

        for(const auto& [tid, samples] : per_tid)
        {
            for(const auto& s : samples)
            {
                for(std::uint32_t i = 0; i < s.frames.size(); ++i)
                {
                    if(const char* cf = canonical_chain_name(sym.resolve(s.frames[i])))
                    {
                        ++self_count[cf];
                        representative_ip[cf] = s.frames[i];
                        break;
                    }
                }

                std::unordered_set<const char*> in_stack;
                for(std::uint32_t i = 0; i < s.frames.size(); ++i)
                {
                    const char* cf = canonical_chain_name(sym.resolve(s.frames[i]));
                    if(cf != nullptr && in_stack.insert(cf).second)
                    {
                        ++inclusive_count[cf];
                        representative_ip.try_emplace(cf, s.frames[i]);
                    }
                }
            }
        }

        std::printf("\ncallstack tree (root -> leaf), %d iterations:\n", k_iterations);
        int depth = 0;
        for(const char* fn : k_chain_order)
        {
            const std::uint64_t self_n = self_count.contains(fn) ? self_count.at(fn) : 0;
            const std::uint64_t incl_n = inclusive_count.contains(fn) ? inclusive_count.at(fn) : 0;
            const double self_ms = static_cast<double>(self_n * k_sample_period_ns) / 1e6;
            const double incl_ms = static_cast<double>(incl_n * k_sample_period_ns) / 1e6;
            std::string loc = representative_ip.contains(fn)
                                  ? std::string(sym.file_line(representative_ip.at(fn)))
                                  : std::string{};
            if(const auto slash = loc.find_last_of('/'); slash != std::string::npos)
            {
                loc.erase(0, slash + 1);
            }

            std::string indent(static_cast<std::size_t>(depth) * 2, ' ');
            std::printf("%s%-18s %-35s self=%4zu samples (%7.1f ms)  incl=%4zu samples (%7.1f ms)\n",
                        indent.c_str(), fn, loc.empty() ? "(unresolved)" : loc.c_str(),
                        static_cast<std::size_t>(self_n), self_ms, static_cast<std::size_t>(incl_n),
                        incl_ms);
            ++depth;
        }

        return 0;
    }
    catch(const std::exception& e)
    {
        std::fprintf(stderr, "%s: %s\n", argc > 0 ? argv[0] : "(unknown)", e.what());
        return EXIT_FAILURE;
    }
}
