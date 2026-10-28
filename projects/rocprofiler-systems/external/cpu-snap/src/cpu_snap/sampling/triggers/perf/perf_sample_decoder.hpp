#pragma once

/**
 * @file triggers/perf/perf_sample_decoder.hpp
 * @brief Stateless decoding of PERF_RECORD_SAMPLE records into frame chains.
 *
 * Header-only. C++20.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <sys/types.h>
#include <vector>

#include <linux/perf_event.h>

#include "cpu_snap/sampling/unwind/machine_snapshot.hpp"
#include "cpu_snap/sampling/unwind/perf_regs.hpp"

namespace cpu_snap::sampling::triggers::perf
{

struct perf_sample_decoder
{
    /**
     * @brief Reusable frame buffers for decoding one sample.
     *
     * Both buffers hold `callchain_depth` entries, so the depth is a runtime
     * setting and the hot path never allocates.
     */
    struct decode_scratch
    {
        explicit decode_scratch(std::uint32_t callchain_depth)
        : frames(callchain_depth)
        , raw_ips(callchain_depth)
        {}

        std::vector<std::uint64_t> frames;
        std::vector<std::uint64_t> raw_ips;
    };

    // Record layouts (sample_type bits we set):
    //   frame_pointer  : IP|TID|TIME|CALLCHAIN
    //     [u64 ip] [u32 pid][u32 tid] [u64 time]
    //     [u64 nr_ips] [u64 ips[nr_ips]]
    //   dwarf          : IP|TID|TIME|REGS_USER|STACK_USER (decoded via dwarf_unwinder)

    /**
     * @brief Decode one PERF_RECORD_SAMPLE and forward it to @p on_sample.
     *
     * Every length field in the record is validated against @c hdr->size;
     * a record too short for its fixed prefix is dropped, and a stack dump
     * or callchain that overruns the record yields a sample without frames
     * (or clamped to what the record holds).
     *
     * @param hdr             Contiguous record, @c hdr->size bytes long.
     * @param scratch         Frame buffers; their size is the max frames to produce.
     * @param unwinder        Concrete unwinder for this batch.
     * @param on_sample       Per-sample sink, see drain_all.
     */
    template <unwind::Unwinder Unw, typename SampleSink>
    static void process_sample(const perf_event_header* hdr, decode_scratch& scratch,
                               Unw& unwinder, const SampleSink& on_sample) noexcept
    {
        const auto* record_begin = reinterpret_cast<const std::uint8_t*>(hdr);
        const auto* end          = record_begin + hdr->size;
        const auto* cur          = record_begin + sizeof(perf_event_header);
        if(static_cast<std::size_t>(end - cur) < k_sample_prefix_bytes)
        {
            return;
        }
        const auto pfx = decode_sample_prefix(cur);

        std::uint32_t n_frames = 0;

        // Compile-time branch: Unw is fixed for the whole batch (chosen once
        // by the caller's std::visit), so this resolves to exactly one of
        // the two decode paths with no runtime cost.
        if constexpr(Unw::required_capture() == unwind::capture_kind::kernel_callchain)
        {
            n_frames = decode_frame_pointer_frames(cur, end, scratch, unwinder);
        }
        else
        {
            n_frames = decode_regs_and_stack_frames(cur, end, scratch, unwinder);
        }

        on_sample(static_cast<pid_t>(pfx.tid), pfx.time,
                  std::span<const std::uint64_t>{ scratch.frames.data(), n_frames });
    }

    struct sample_prefix
    {
        std::uint32_t tid;
        std::uint64_t time;
    };

    static constexpr std::size_t k_sample_prefix_bytes = 24;

    static sample_prefix decode_sample_prefix(const std::uint8_t*& cur) noexcept
    {
        sample_prefix prefix{};
        cur += sizeof(std::uint64_t);  // ip (unused)
        cur += sizeof(std::uint32_t);  // pid (unused)
        prefix.tid = *reinterpret_cast<const std::uint32_t*>(cur);
        cur += sizeof(std::uint32_t);
        prefix.time = *reinterpret_cast<const std::uint64_t*>(cur);
        cur += sizeof(std::uint64_t);
        return prefix;
    }

    template <unwind::Unwinder Unw>
    static std::uint32_t decode_frame_pointer_frames(
        const std::uint8_t* cur, const std::uint8_t* end, decode_scratch& scratch,
        Unw& unwinder) noexcept
    {
        std::vector<std::uint64_t>& raw_ips = scratch.raw_ips;
        const std::size_t           depth   = raw_ips.size();
        std::uint32_t               n_raw   = 0;
        if(static_cast<std::size_t>(end - cur) < sizeof(std::uint64_t))
        {
            return 0;
        }
        std::uint64_t nr_ips = *reinterpret_cast<const std::uint64_t*>(cur);
        cur += 8;
        nr_ips = std::min<std::uint64_t>(nr_ips, static_cast<std::size_t>(end - cur) / 8);
        for(std::uint64_t i = 0; i < nr_ips; ++i)
        {
            const std::uint64_t ipi = *reinterpret_cast<const std::uint64_t*>(cur);
            cur += 8;
            // PERF_CONTEXT_* sentinels and kernel-space IPs sit above the
            // user/kernel split on x86_64.
            if(ipi >= k_perf_user_ip_ceiling)
            {
                continue;
            }
            if(n_raw >= depth)
            {
                continue;
            }
            raw_ips[n_raw++] = ipi;
        }

        const unwind::machine_snapshot snap{
            .ready_frames = std::span<const std::uint64_t>{ raw_ips.data(), n_raw },
            .regs         = nullptr,
            .stack_addr   = 0,
            .stack        = {},
        };
        return static_cast<std::uint32_t>(unwinder.unwind(snap, std::span<std::uint64_t>{ scratch.frames }));
    }

    template <unwind::Unwinder Unw>
    static std::uint32_t decode_regs_and_stack_frames(
        const std::uint8_t* cur, const std::uint8_t* end, decode_scratch& scratch,
        Unw& unwinder) noexcept
    {
        // PERF_SAMPLE_REGS_USER layout (when sample_regs_user != 0):
        //   u64 abi
        //   u64 regs[weight(sample_regs_user)]
        // PERF_SAMPLE_STACK_USER layout:
        //   u64 size                      (dump size or 0 if abi==0)
        //   u8  data[size]
        //   u64 dyn_size                  (only present if size > 0)
        const auto remaining = [&] { return static_cast<std::size_t>(end - cur); };

        if(remaining() < sizeof(std::uint64_t))
        {
            return 0;
        }
        const std::uint64_t regs_abi = *reinterpret_cast<const std::uint64_t*>(cur);
        cur += 8;
        const std::uint64_t* regs_ptr = nullptr;
        if(regs_abi != PERF_SAMPLE_REGS_ABI_NONE)
        {
            constexpr std::size_t regs_bytes =
                sizeof(std::uint64_t) * unwind::k_perf_dwarf_n_regs_x86_64;
            if(remaining() < regs_bytes)
            {
                return 0;
            }
            regs_ptr = reinterpret_cast<const std::uint64_t*>(cur);
            cur += regs_bytes;
        }
        if(remaining() < sizeof(std::uint64_t))
        {
            return 0;
        }
        const std::uint64_t stack_dumped_size =
            *reinterpret_cast<const std::uint64_t*>(cur);
        cur += 8;
        const std::uint8_t* stack_data = cur;
        std::uint64_t       stack_size = 0;
        if(stack_dumped_size > 0)
        {
            if(remaining() < sizeof(std::uint64_t) ||
               stack_dumped_size > remaining() - sizeof(std::uint64_t))
            {
                return 0;
            }
            cur += stack_dumped_size;
            stack_size = *reinterpret_cast<const std::uint64_t*>(cur);
            cur += 8;
            if(stack_size > stack_dumped_size)
            {
                stack_size = stack_dumped_size;
            }
        }

        if(regs_ptr == nullptr || stack_size == 0)
        {
            return 0;
        }
        // RSP at sample time = base address of stack snapshot.
        const std::uint64_t rsp = regs_ptr[7 /*PERF_REG_X86_SP slot*/];
        const unwind::machine_snapshot snap{
            .ready_frames = {},
            .regs         = regs_ptr,
            .stack_addr   = rsp,
            .stack        = std::span<const std::byte>{
                reinterpret_cast<const std::byte*>(stack_data), stack_size },
        };
        try
        {
            return static_cast<std::uint32_t>(
                unwinder.unwind(snap, std::span<std::uint64_t>{ scratch.frames }));
        } catch(...)  // per-sample path must not throw; a failed unwind yields an empty chain
        {
            return 0;
        }
    }

    /**
     * x86_64 user-space VA ceiling. PERF_CONTEXT_* sentinels (e.g.
     * PERF_CONTEXT_KERNEL = (__u64)-128) and any kernel-space IP sit at
     * or above this address; user IPs sit strictly below.
     */
    // NOLINTNEXTLINE(readability-identifier-naming): codebase k_*-prefix convention
    static constexpr std::uint64_t k_perf_user_ip_ceiling = 0x0000800000000000ULL;
};

}  // namespace cpu_snap::sampling::triggers::perf
