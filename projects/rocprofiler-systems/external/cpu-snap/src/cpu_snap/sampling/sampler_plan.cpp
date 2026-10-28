#include "cpu_snap/sampling/sampler_plan.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include <unistd.h>

#include <csignal>

#include "cpu_snap/core/frame_limits.hpp"
#include "cpu_snap/error.hpp"
#include "cpu_snap/sampling/triggers/perf/backend.hpp"
#include "cpu_snap/sampling/triggers/perf/ring_sizing.hpp"

namespace cpu_snap::sampling::internal
{

namespace
{

void
validate_config(const sampler::config& cfg)
{
    if(!cfg.streaming.sink)
    {
        throw config_error(
            "sampler: streaming.sink is mandatory (use core::memory_sink for a "
            "bounded in-memory capture)");
    }
    if(cfg.perf.callchain_depth == 0)
    {
        throw config_error("callchain_depth must be > 0");
    }
    if(cfg.perf.callchain_depth > core::k_max_callchain_depth)
    {
        throw config_error("callchain_depth exceeds core::k_max_callchain_depth");
    }
    if((cfg.perf.kernel_pages_per_tid & (cfg.perf.kernel_pages_per_tid - 1)) != 0)
    {
        throw config_error(
            "kernel_pages_per_tid must be a power of 2 (0 selects a size from the "
            "snapshot)");
    }
}

int
signal_number(const sampler::config& cfg) noexcept
{
    return SIGRTMIN + static_cast<int>(cfg.signal.signal_rt_offset);
}

void
validate_signal_config(const sampler::config& cfg)
{
    if(cfg.signal.sample_freq_hz == 0)
    {
        throw config_error("signal.sample_freq_hz must be > 0");
    }
    if(cfg.signal.stack_snapshot_bytes == 0 ||
       (cfg.signal.stack_snapshot_bytes & 7U) != 0)
    {
        throw config_error(
            "signal.stack_snapshot_bytes must be a non-zero multiple of 8");
    }
    if(cfg.signal.ring_capacity < 2)
    {
        throw config_error("signal.ring_capacity must be >= 2");
    }
    const int signal_no = signal_number(cfg);
    if(signal_no < SIGRTMIN || signal_no > SIGRTMAX)
    {
        throw config_error(
            "signal.signal_rt_offset resolves outside [SIGRTMIN, SIGRTMAX]");
    }
}

// Context-free capability query (see triggers::perf::backend::supports /
// triggers::signal::backend::supports): needed here because kernel_drain's
// perf_event_attr shape must be fixed BEFORE a perf trigger instance exists,
// so unwind_mode resolution cannot wait for a constructed i_trigger to
// virtually answer can_provide(). Mirrors each concrete backend's
// can_provide() override; sampler's constructor separately re-checks the
// real constructed instance below (the official capture_kind handshake).
bool
trigger_supports(backend_kind bk, unwind::capture_kind kind) noexcept
{
    switch(bk)
    {
        case backend_kind::perf: return triggers::perf::backend::supports(kind);
        case backend_kind::signal: return triggers::signal::backend::supports(kind);
        case backend_kind::automatic: break;
    }
    return false;
}

unwind_mode
resolve_unwind_mode(const sampler::config& cfg)
{
    if(cfg.perf.unwind == unwind_mode::auto_best)
    {
        return unwind_mode::dwarf;
    }

    if(!trigger_supports(cfg.backend, capture_kind_for(cfg.perf.unwind, cfg.backend)))
    {
        throw config_error(
            "sampler: selected backend cannot provide the requested unwind_mode");
    }
    return cfg.perf.unwind;
}

void
validate_unwind_constraints(const sampler::config& cfg, unwind_mode resolved)
{
    if(resolved == unwind_mode::dwarf && (cfg.perf.dwarf_stack_bytes & 7U) != 0)
    {
        throw config_error("sampler: dwarf_stack_bytes must be 8-byte aligned");
    }
}

void
resolve_perf_ring_sizing(sampler::config& cfg, unwind_mode resolved)
{
    const bool          dwarf = resolved == unwind_mode::dwarf;
    const std::uint32_t stack = cfg.perf.dwarf_stack_bytes;
    if(dwarf && (stack == 0 || stack > triggers::perf::k_max_dwarf_stack_bytes))
    {
        throw config_error(
            "sampler: dwarf_stack_bytes must be in [8, " +
            std::to_string(triggers::perf::k_max_dwarf_stack_bytes) + "]");
    }

    const auto page_bytes = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    auto&      pages      = cfg.perf.kernel_pages_per_tid;
    if(pages == 0)
    {
        pages = triggers::perf::default_ring_pages(resolved, stack, page_bytes);
        return;
    }
    if(!triggers::perf::ring_holds_a_record(resolved, stack, pages, page_bytes))
    {
        throw config_error(
            "sampler: kernel_pages_per_tid (" + std::to_string(pages) +
            ") is too small for one sample record of " +
            std::to_string(triggers::perf::max_sample_record_bytes(resolved, stack)) +
            " bytes; raise it, set it to 0 to size it automatically, or lower "
            "dwarf_stack_bytes");
    }
}

}  // namespace

triggers::signal::backend::config
make_signal_config(const sampler::config& cfg) noexcept
{
    return triggers::signal::backend::config{
        .freq_hz              = cfg.signal.sample_freq_hz,
        .signal_no            = signal_number(cfg),
        .stack_snapshot_bytes = cfg.signal.stack_snapshot_bytes,
        .callchain_depth      = cfg.perf.callchain_depth,
        .ring_capacity        = static_cast<std::size_t>(cfg.signal.ring_capacity),
        .start_delay_ns       = cfg.signal.start_delay_ns,
        .clock                = cfg.clock,
    };
}

/**
 * @brief capture_kind an explicit (non-auto_best) unwind_mode maps onto for
 *        @p bk.
 *
 * frame_pointer means different capture kinds per trigger: perf's kernel
 * already walks the stack (kernel_callchain), while the signal trigger has
 * no such kernel facility and instead frame-pointer-walks a copied stack
 * itself (fp_copy_unwinder, regs_and_stack).
 */
unwind::capture_kind
capture_kind_for(unwind_mode m, backend_kind bk) noexcept
{
    if(m == unwind_mode::dwarf)
    {
        return unwind::capture_kind::regs_and_stack;
    }
    return (bk == backend_kind::signal) ? unwind::capture_kind::regs_and_stack
                                        : unwind::capture_kind::kernel_callchain;
}

core::stream_clock
stream_clock_for(clock_kind clock) noexcept
{
    switch(clock)
    {
        case clock_kind::boottime: return core::stream_clock::boottime;
        case clock_kind::wallclock: return core::stream_clock::wallclock;
        case clock_kind::monotonic: break;
    }
    return core::stream_clock::monotonic;
}

// file_header.flags bit 1 = frame-pointer stack, bit 2 = DWARF-unwound
// stack, bits 3-4 = stream clock. resolved is never auto_best
// (resolve_unwind_mode already collapses it to a concrete mode before this runs).
std::uint16_t
streaming_flags_for(unwind_mode resolved, clock_kind clock) noexcept
{
    const std::uint32_t unwind_bits = resolved == unwind_mode::dwarf ? 0x4U : 0x2U;
    return static_cast<std::uint16_t>(
        core::with_stream_clock(unwind_bits, stream_clock_for(clock)));
}

triggers::perf::kernel_drain::config
make_drain_config(const sampler::config& cfg) noexcept
{
    return triggers::perf::kernel_drain::config{
        .sample_freq_hz       = cfg.perf.sample_freq_hz,
        .callchain_depth      = cfg.perf.callchain_depth,
        .kernel_pages_per_tid = cfg.perf.kernel_pages_per_tid,
        .exclude_kernel       = cfg.perf.exclude_kernel,
        .exclude_hv           = cfg.perf.exclude_hv,
        .dwarf_stack_bytes    = cfg.perf.dwarf_stack_bytes,
        .unwind               = cfg.perf.unwind,
        .clock                = cfg.clock,
    };
}

resolved_plan
resolve_plan(sampler::config cfg, perf_probe_fn probe, bool trigger_injected)
{
    validate_config(cfg);
    backend_choice choice;
    if(trigger_injected)
    {
        if(cfg.backend == backend_kind::automatic) cfg.backend = backend_kind::perf;
        choice.kind = cfg.backend;
    }
    else
    {
        const backend_kind requested = cfg.backend;
        choice = resolve_backend(requested, cfg.perf.exclude_kernel, probe);
        cfg.backend = choice.kind;
        // signal.sample_freq_hz is a separate field; honor a perf rate set for the default backend
        if(requested == backend_kind::automatic && choice.kind == backend_kind::signal &&
           cfg.signal.sample_freq_hz == k_default_sample_freq_hz)
        {
            cfg.signal.sample_freq_hz = cfg.perf.sample_freq_hz;
        }
    }
    const bool unwind_was_auto = cfg.perf.unwind == unwind_mode::auto_best;
    cfg.perf.unwind            = resolve_unwind_mode(cfg);
    validate_unwind_constraints(cfg, cfg.perf.unwind);
    if(cfg.backend == backend_kind::perf)
    {
        resolve_perf_ring_sizing(cfg, cfg.perf.unwind);
    }
    else if(!trigger_injected)
    {
        validate_signal_config(cfg);
    }
    return resolved_plan{ std::move(cfg), std::move(choice), unwind_was_auto };
}

}  // namespace cpu_snap::sampling::internal
