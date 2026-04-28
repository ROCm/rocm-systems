// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Production-complete definitions for all 10 sampling policy types.
// This header lives in library/ because the six "heavy" policies depend on
// main-library symbols (perf.hpp, tracing.hpp, trace_cache, thread_info).
//
// The four "light" policies (steady_clock, real_signal_dispatcher,
// real_timer_trigger, libunwind_unwinder) are defined in their own headers
// under sampling/src/linux/ and are included here for completeness.
//
// Only include this header from translation units that are compiled as part of
// the main rocprofiler-systems library (not from standalone test binaries).

#if !defined(__linux__)
#    error "sampling_production_policies.hpp is Linux-only"
#endif

// ── Light policies (sampling/ — no main-lib deps) ─────────────────────────
#include "sampling/src/linux/libunwind_unwinder.hpp"
#include "sampling/src/linux/real_signal_dispatcher.hpp"
#include "sampling/src/linux/steady_clock.hpp"

// ── Data / ring-buffer types used by heavy policies ───────────────────────
#include "sampling/data/backtrace_record.hpp"
#include "sampling/data/overflow_sample.hpp"
#include "sampling/data/stack_frame.hpp"
#include "sampling/data/timer_sample.hpp"
#include "sampling/src/sample_ring_buffer.hpp"

// ── Production-only trigger policies (not test-accessible) ────────────────
#include "library/sampling_production_policies/real_overflow_trigger.hpp"
#include "library/sampling_production_policies/real_timer_trigger.hpp"

// ── EmitterPolicy (production) ────────────────────────────────────────────
// Lightweight header — no libunwind / AMD-SMI deps; also included by test TUs.
#include "library/sampling_production_policies/trace_cache_offload_adapter.hpp"

// ── TSV report writer (no main-lib deps; test-accessible) ────────────────
#include "sampling/src/native_report_writer.hpp"

// ── Main-library deps (only valid in main-lib TUs) ───────────────────────
#include "core/components/fwd.hpp"
#include "core/config.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "library/runtime.hpp"
#include "library/thread_info.hpp"
#include "library/tracing.hpp"
#include "library/tracing/annotation.hpp"
#include "logger/debug.hpp"

#include "rocprofiler-systems/categories.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <sys/types.h>
#include <unordered_map>
#include <vector>

namespace rocprofsys::sampling
{

// ── real_fatal_error_policy ───────────────────────────────────────────────
class real_fatal_error_policy
{
public:
    template <class... Args>
    [[noreturn]] void fatal(char const* file, int line, std::string_view fmt,
                            Args const&... /*args*/) noexcept
    {
        static std::mutex           fatal_mtx;
        std::lock_guard<std::mutex> lk{ fatal_mtx };
        LOG_CRITICAL("[{}:{}] fatal sampling error: {}", file, line, fmt);
        ::_Exit(1);
    }
};

// ── tmpfile_offload_store ─────────────────────────────────────────────────
// Serializes full backtrace_record structs to the sampling tmp file.
// Binary format per write() call:
//   [8B] int64_t  tid
//   [8B] uint64_t n_records
//   [n_records * sizeof(backtrace_record)] raw backtrace_record bytes
// backtrace_record is trivially copyable so memcpy serialization is exact.
class tmpfile_offload_store
{
public:
    // FatalErrorPolicy::fatal() is called on I/O failure (NFR-FM-2).
    template <size_t N, class FatalErrorPolicy>
    void write(int64_t tid, sample_ring_buffer<N>& buf, FatalErrorPolicy& fatal) noexcept
    {
        try
        {
            write_impl(tid, buf);
        } catch(std::exception const& e)
        {
            fatal.fatal(__FILE__, __LINE__,
                        "[tmpfile_offload_store] write failed for tid {}: {}", tid,
                        e.what());
        } catch(...)
        {
            fatal.fatal(__FILE__, __LINE__,
                        "[tmpfile_offload_store] write failed for tid {} (unknown)", tid);
        }
    }

private:
    template <size_t N>
    void write_impl(int64_t tid, sample_ring_buffer<N>& buf)
    {
        // Drain all available records from the ring buffer into a local vector
        // before taking the file lock so the ring is free for the signal handler
        // as quickly as possible.
        std::vector<backtrace_record> records;
        records.reserve(buf.count());
        while(auto rec = buf.pop())
            records.push_back(*rec);

        if(records.empty()) return;

        std::lock_guard<std::mutex> lk{ m_mutex };

        if(!m_stream.is_open())
        {
            auto tmp = config::get_tmp_file("sampling");
            if(!tmp)
            {
                // L12 — matches legacy exactly
                LOG_CRITICAL("sampling allocator tries to offload buffer of samples but "
                             "rocprof-sys was configured to not use temporary files");
                return;
            }
            m_path = tmp->filename;
            // Ensure the parent directory exists before opening the fstream.
            auto parent = std::filesystem::path{ m_path }.parent_path();
            if(!parent.empty()) std::filesystem::create_directories(parent);
            m_stream.open(m_path, std::ios::binary | std::ios::in | std::ios::out |
                                      std::ios::app);
            if(!m_stream.is_open())
            {
                // L10 — matches legacy: "Error opening sampling offload temporary file
                // '{}'"
                LOG_CRITICAL("Error opening sampling offload temporary file '{}'",
                             m_path);
                return;
            }
        }

        if(!m_stream.good())
        {
            // L14 — matches legacy: "temporary file for offloading buffer is in an
            // invalid state during offload for thread {}"
            LOG_CRITICAL("temporary file for offloading buffer is in an invalid state "
                         "during offload for thread {}",
                         tid);
            return;
        }

        // L11 — matches legacy: "Offloading {} samples for thread {} to {}"
        LOG_DEBUG("Offloading {} samples for thread {} to {}", records.size(), tid,
                  m_path);

        m_stream.seekp(0, std::ios::end);
        auto offset = m_stream.tellp();
        m_offsets[tid].insert(offset);

        uint64_t n = static_cast<uint64_t>(records.size());
        m_stream.write(reinterpret_cast<char const*>(&tid), sizeof(tid));
        m_stream.write(reinterpret_cast<char const*>(&n), sizeof(n));
        m_stream.write(reinterpret_cast<char const*>(records.data()),
                       static_cast<std::streamsize>(n * sizeof(backtrace_record)));
        m_stream.flush();
    }

public:
    // Read all offloaded backtrace_record entries for tid from the tmp file.
    // Called single-threaded from sampling_service::post_process().
    [[nodiscard]] std::vector<backtrace_record> read(int64_t tid)
    {
        std::lock_guard<std::mutex>   lk{ m_mutex };
        std::vector<backtrace_record> result;

        if(!m_stream.is_open() && m_path.empty())
        {
            // L15 — matches legacy: no tmp file was ever opened (disabled path)
            LOG_WARNING("[sampling] returning no data because using temporary files is "
                        "disabled");
            return result;
        }

        if(!m_stream.is_open() && !m_path.empty())
        {
            // L16 — matches legacy: file was used but stream is no longer open
            LOG_WARNING(
                "[sampling] returning no data because the offload file no longer exists");
            return result;
        }

        auto it = m_offsets.find(tid);
        if(it == m_offsets.end()) return result;

        // L13 — matches legacy: offload entry exists but file has been closed/removed
        if(!m_stream.good())
        {
            LOG_CRITICAL("sampling allocator tried to offload buffer of samples for "
                         "thread {} but the offload file does not exist",
                         tid);
            return result;
        }

        size_t loaded_count = 0;
        for(auto file_offset : it->second)
        {
            m_stream.clear();
            m_stream.seekg(file_offset);
            if(!m_stream.good())
            {
                // L17 — matches legacy: "[sampling] {} failed to open" (seek failed)
                LOG_WARNING("[sampling] {} failed to open", m_path);
                continue;
            }

            int64_t  stored_tid = 0;
            uint64_t n          = 0;
            m_stream.read(reinterpret_cast<char*>(&stored_tid), sizeof(stored_tid));
            m_stream.read(reinterpret_cast<char*>(&n), sizeof(n));
            if(!m_stream.good() || n == 0) continue;

            if(stored_tid != tid)
            {
                // L18 — matches legacy: "[sampling] file position {} returned {} instead
                // of (expected) {}"
                LOG_WARNING("[sampling] file position {} returned {} instead of "
                            "(expected) {}",
                            static_cast<uintptr_t>(file_offset), stored_tid, tid);
                continue;
            }

            size_t base = result.size();
            result.resize(base + n);
            m_stream.read(reinterpret_cast<char*>(result.data() + base),
                          static_cast<std::streamsize>(n * sizeof(backtrace_record)));
            if(!m_stream.good())
                result.resize(base);  // drop partial read
            else
                loaded_count += n;
        }

        // L19 — matches legacy: "[sampling] Loaded {} samples for thread {}"
        LOG_DEBUG("[sampling] Loaded {} samples for thread {}", loaded_count, tid);

        return result;
    }

    // Return the set of tids that have data written to the offload store.
    // Called from sampling_service::post_process() to iterate over known tids
    // without relying on the registry (which is cleared by shutdown()).
    [[nodiscard]] std::vector<int64_t> tids() const
    {
        std::lock_guard<std::mutex> lk{ m_mutex };
        std::vector<int64_t>        result;
        result.reserve(m_offsets.size());
        for(auto const& kv : m_offsets)
            result.push_back(kv.first);
        return result;
    }

    // Close the held file handle and clear all state.
    // Called once at the end of sampling_service::post_process().
    void reset() noexcept
    {
        std::lock_guard<std::mutex> lk{ m_mutex };
        if(m_stream.is_open()) m_stream.close();
        m_offsets.clear();
        m_path.clear();
    }

private:
    mutable std::mutex                                            m_mutex;
    std::fstream                                                  m_stream;
    std::string                                                   m_path;
    std::unordered_map<int64_t, std::set<std::fstream::pos_type>> m_offsets;
};

// ── real_trace_cache_sink ─────────────────────────────────────────────────
// TraceSinkPolicy implementation that pushes post-parse resolved samples to
// trace_cache::buffer_storage so data_processor can write them to rocpd.db.
// store_timer / store_overflow are called from post_process() AFTER DWARF
// resolution — matching the legacy cache_sampling_data() call site.
class real_trace_cache_sink
{
public:
    void store_timer(int64_t tid, std::vector<timer_sample> const& samples)
    {
        // L44 — matches legacy: "[{}] Post-processing metrics for rocpd..."
        LOG_DEBUG("[{}] Post-processing metrics for rocpd...", tid);
        // L01 — matches legacy: "[{}] Storing sampling data to trace cache..."
        LOG_DEBUG("[{}] Storing sampling data to trace cache...", tid);
        LOG_DEBUG("[real_trace_cache_sink] store_timer: tid={} samples={}", tid,
                  samples.size());
        if(samples.empty()) return;

        const auto& info = thread_info::get(tid, SequentTID);
        LOG_DEBUG("[real_trace_cache_sink] store_timer: tid={} thread_info={}", tid,
                  info ? "valid" : "null");
        if(!info) return;

        size_t      sys_id = info->index_data->system_value;
        size_t      seq_id = info->index_data->sequent_value;
        std::string track_name =
            "Thread " + std::to_string(seq_id) + " (S) " + std::to_string(sys_id);
        constexpr uint32_t category_id =
            static_cast<uint32_t>(ROCPROFSYS_CATEGORY_TIMER_SAMPLING);
        constexpr auto category_str = "timer_sampling";

        for(auto const& s : samples)
        {
            if(!info->is_valid_lifetime({ s.beg_ns, s.end_ns })) continue;

            int depth = 0;
            for(auto const& frame : s.stack)
            {
                std::string name       = frame.name.empty()
                                             ? ("0x" + fmt::format("{:X}", frame.address))
                                             : frame.name;
                std::string call_stack = make_call_stack_json(frame);
                std::string line_info  = make_line_info_json(frame);
                // Locked extdata schema: { "depth": <int>, ... future fields ... }
                std::string extdata = make_extdata_json(depth);

                LOG_DEBUG("[real_trace_cache_sink] store_timer: tid={} frame='{}'", tid,
                          name);
                trace_cache::get_buffer_storage().store(
                    trace_cache::backtrace_region_sample{
                        category_id, static_cast<uint64_t>(sys_id), track_name, name,
                        s.beg_ns, s.end_ns, category_str, std::move(call_stack),
                        std::move(line_info), std::move(extdata) });
                ++depth;
            }
        }
    }

    void store_overflow(int64_t tid, std::vector<overflow_sample> const& samples)
    {
        // L01 — matches legacy: "[{}] Storing sampling data to trace cache..."
        LOG_DEBUG("[{}] Storing sampling data to trace cache...", tid);
        LOG_DEBUG("[real_trace_cache_sink] store_overflow: tid={} samples={}", tid,
                  samples.size());
        if(samples.empty()) return;

        const auto& info = thread_info::get(tid, SequentTID);
        LOG_DEBUG("[real_trace_cache_sink] store_overflow: tid={} thread_info={}", tid,
                  info ? "valid" : "null");
        if(!info) return;

        size_t      sys_id     = info->index_data->system_value;
        size_t      seq_id     = info->index_data->sequent_value;
        std::string track_name = "Thread " + std::to_string(seq_id) + " Overflow (S) " +
                                 std::to_string(sys_id);
        constexpr uint32_t category_id =
            static_cast<uint32_t>(ROCPROFSYS_CATEGORY_OVERFLOW_SAMPLING);
        constexpr auto category_str = "overflow_sampling";

        for(auto const& s : samples)
        {
            if(!info->is_valid_lifetime({ s.beg_ns, s.end_ns })) continue;

            int depth = 0;
            for(auto const& frame : s.stack)
            {
                std::string name       = frame.name.empty()
                                             ? ("0x" + fmt::format("{:X}", frame.address))
                                             : frame.name;
                std::string call_stack = make_call_stack_json(frame);
                std::string line_info  = make_line_info_json(frame);
                // Locked extdata schema: { "depth": <int>, ... future fields ... }
                std::string extdata = make_extdata_json(depth);

                LOG_DEBUG("[real_trace_cache_sink] store_overflow: tid={} frame='{}'",
                          tid, name);
                trace_cache::get_buffer_storage().store(
                    trace_cache::backtrace_region_sample{
                        category_id, static_cast<uint64_t>(sys_id), track_name, name,
                        s.beg_ns, s.end_ns, category_str, std::move(call_stack),
                        std::move(line_info), std::move(extdata) });
                ++depth;
            }
        }
    }

private:
    static std::string make_call_stack_json(stack_frame const& frame)
    {
        nlohmann::json j;
        j["name"] = frame.name.empty() ? fmt::format("{:X}", frame.address) : frame.name;
        j["pc"]   = fmt::format("{:X}", frame.address);
        j["file"] = frame.location;
        return j.dump();
    }

    static std::string make_line_info_json(stack_frame const& frame)
    {
        nlohmann::json j;
        j["line_address"] = fmt::format("{:X}", frame.line_address);
        j["name"] = frame.name.empty() ? fmt::format("{:X}", frame.address) : frame.name;
        if(!frame.inlines.empty())
        {
            nlohmann::json inlined;
            auto const&    top  = frame.inlines.front();
            inlined["name"]     = top.name;
            inlined["location"] = top.location;
            inlined["line"]     = std::to_string(top.line);
            j["inlined"]        = inlined;
        }
        return j.dump();
    }

    // Locked extdata schema: { "depth": <int>, ... future fields ... }
    static std::string make_extdata_json(int depth)
    {
        nlohmann::json j;
        j["depth"] = depth;
        return j.dump();
    }
};

// native_report_writer is defined in sampling/src/native_report_writer.hpp,
// included above. No redefinition needed here.

// ── real_perfetto_sink ────────────────────────────────────────────────────
class real_perfetto_sink
{
public:
    void emit_timer(int64_t                          tid, void const* /*info*/,
                    std::vector<timer_sample> const& samples)
    {
        if(!config::get_use_perfetto()) return;
        if(samples.empty()) return;

        // L41 — matches legacy: "[{}] Post-processing metrics for perfetto..."
        LOG_DEBUG("[{}] Post-processing metrics for perfetto...", tid);
        // L42 — matches legacy: "[{}] Post-processing backtraces for perfetto..."
        LOG_DEBUG("[{}] Post-processing backtraces for perfetto...", tid);

        const auto& thread_inf = thread_info::get(tid, SequentTID);
        if(!thread_inf) return;

        auto track = tracing::get_perfetto_track(
            category::timer_sampling{},
            [](auto seq_id, auto sys_id) {
                return std::string("Thread ") + std::to_string(seq_id) + " (S) " +
                       std::to_string(sys_id);
            },
            thread_inf->index_data->sequent_value, thread_inf->index_data->system_value);

        uint64_t beg_ns = std::max(samples.front().beg_ns, thread_inf->get_start());
        uint64_t end_ns = std::min(samples.back().end_ns, thread_inf->get_stop());

        tracing::push_perfetto_track(category::timer_sampling{}, "samples [rocprof-sys]",
                                     track, beg_ns, [&](::perfetto::EventContext ctx) {
                                         if(config::get_perfetto_annotations())
                                             tracing::add_perfetto_annotation(
                                                 ctx, "begin_ns", beg_ns);
                                     });

        for(auto const& s : samples)
        {
            if(!thread_inf->is_valid_lifetime({ s.beg_ns, s.end_ns })) continue;

            for(auto const& frame : s.stack)
            {
                const char* n =
                    intern(frame.name.empty() ? ("0x" + std::to_string(frame.address))
                                              : frame.name);
                tracing::push_perfetto_track(
                    category::timer_sampling{}, n, track, s.beg_ns,
                    [&](::perfetto::EventContext ctx) {
                        if(config::get_perfetto_annotations())
                        {
                            tracing::add_perfetto_annotation(ctx, "begin_ns", s.beg_ns);
                            tracing::add_perfetto_annotation(ctx, "end_ns", s.end_ns);
                            tracing::add_perfetto_annotation(
                                ctx, "pc", "0x" + std::to_string(frame.address));
                        }
                    });
                tracing::pop_perfetto_track(category::timer_sampling{}, n, track,
                                            s.end_ns);
            }
        }

        tracing::pop_perfetto_track(category::timer_sampling{}, "samples [rocprof-sys]",
                                    track, end_ns, [&](::perfetto::EventContext ctx) {
                                        if(config::get_perfetto_annotations())
                                            tracing::add_perfetto_annotation(
                                                ctx, "end_ns", end_ns);
                                    });
    }

    void emit_overflow(int64_t                             tid, void const* /*info*/,
                       std::vector<overflow_sample> const& samples)
    {
        if(!config::get_use_perfetto()) return;
        if(samples.empty()) return;

        // L41 — matches legacy: "[{}] Post-processing metrics for perfetto..."
        LOG_DEBUG("[{}] Post-processing metrics for perfetto...", tid);
        // L42 — matches legacy: "[{}] Post-processing backtraces for perfetto..."
        LOG_DEBUG("[{}] Post-processing backtraces for perfetto...", tid);

        const auto& thread_inf = thread_info::get(tid, SequentTID);
        if(!thread_inf) return;

        auto track = tracing::get_perfetto_track(
            category::overflow_sampling{},
            [](auto seq_id, auto sys_id) {
                return std::string("Thread ") + std::to_string(seq_id) +
                       " Overflow (S) " + std::to_string(sys_id);
            },
            thread_inf->index_data->sequent_value, thread_inf->index_data->system_value);

        uint64_t beg_ns = std::max(samples.front().beg_ns, thread_inf->get_start());
        uint64_t end_ns = std::min(samples.back().end_ns, thread_inf->get_stop());

        tracing::push_perfetto_track(
            category::overflow_sampling{}, "samples [rocprof-sys]", track, beg_ns,
            [&](::perfetto::EventContext ctx) {
                if(config::get_perfetto_annotations())
                    tracing::add_perfetto_annotation(ctx, "begin_ns", beg_ns);
            });

        for(auto const& s : samples)
        {
            if(!thread_inf->is_valid_lifetime({ s.beg_ns, s.end_ns })) continue;

            for(auto const& frame : s.stack)
            {
                const char* n =
                    intern(frame.name.empty() ? ("0x" + std::to_string(frame.address))
                                              : frame.name);
                tracing::push_perfetto_track(
                    category::overflow_sampling{}, n, track, s.beg_ns,
                    [&](::perfetto::EventContext ctx) {
                        if(config::get_perfetto_annotations())
                        {
                            tracing::add_perfetto_annotation(ctx, "begin_ns", s.beg_ns);
                            tracing::add_perfetto_annotation(ctx, "end_ns", s.end_ns);
                            tracing::add_perfetto_annotation(
                                ctx, "pc", "0x" + std::to_string(frame.address));
                        }
                    });
                tracing::pop_perfetto_track(category::overflow_sampling{}, n, track,
                                            s.end_ns);
            }
        }

        tracing::pop_perfetto_track(
            category::overflow_sampling{}, "samples [rocprof-sys]", track, end_ns,
            [&](::perfetto::EventContext ctx) {
                if(config::get_perfetto_annotations())
                    tracing::add_perfetto_annotation(ctx, "end_ns", end_ns);
            });
    }

    char const* intern(std::string const& s)
    {
        std::lock_guard<std::mutex> lk{ m_pool_mtx };
        return m_pool.insert(s).first->c_str();
    }

    std::mutex            m_pool_mtx;
    std::set<std::string> m_pool;
};

}  // namespace rocprofsys::sampling

// ── Signal handler and per-thread state wiring ─────────────────────────────
// Thread-local void* pointers set by setup(), cleared after shutdown() drain.
// Stored as void* so they can be extern-declared in generic impl headers
// without requiring the production type names there.
// Definitions live in services_accessor.cpp (single TU); declared extern here.

#include "sampling/sampling_service.hpp"

namespace rocprofsys::sampling
{

using default_state_t = thread_sampler_state<default_sampling_policies>;

// extern declarations — definitions in services_accessor.cpp.
extern thread_local void*   tl_sampler_state_vp;
extern thread_local void*   tl_offload_vp;
extern thread_local int64_t tl_logical_tid;

// Typed accessors — inline, no linkage cost.
inline default_state_t*
tl_sampler_state()
{
    return static_cast<default_state_t*>(tl_sampler_state_vp);
}
inline trace_cache_offload_adapter*
tl_offload()
{
    return static_cast<trace_cache_offload_adapter*>(tl_offload_vp);
}

}  // namespace rocprofsys::sampling

// Forward declaration — definition in services_accessor.cpp.
extern "C" void
rocprofsys_sampling_signal_handler(int, siginfo_t*, void*);

// Explicit full specializations of the production wiring hooks for
// sampling_service<default_sampling_policies>. Must be included after
// sampling_service.hpp (which pulls in sampling_service_impl.hpp with the
// generic no-op definitions) so the specializations are visible in this TU.
#include "library/sampling_production_policies/sampling_service_production_hooks.hpp"
