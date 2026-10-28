// unwind/fp_copy_unwinder.cpp
//
// Out-of-line body for cpu_snap::sampling::unwind::fp_copy_unwinder.

#include "cpu_snap/sampling/unwind/fp_copy_unwinder.hpp"
#include "cpu_snap/sampling/unwind/perf_regs.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>


namespace cpu_snap::sampling::unwind
{

namespace
{

// Bytes needed at [rbp, rbp+16) to read both the saved rbp and the return
// address of one frame.
constexpr std::uint64_t k_frame_link_bytes = 2 * sizeof(std::uint64_t);

}  // namespace

std::size_t
fp_copy_unwinder::unwind(const machine_snapshot& snap, std::span<std::uint64_t> out) const noexcept
{
    if(out.empty() || snap.regs == nullptr)
    {
        return 0;
    }

    std::size_t n = 0;
    out[n++]      = snap.regs[k_perf_slot_rip];  // leaf IP, unadjusted

    const std::uint64_t stack_addr = snap.stack_addr;
    const auto          stack_size = static_cast<std::uint64_t>(snap.stack.size());
    const std::size_t   limit      = out.size();

    std::uint64_t rbp = snap.regs[k_perf_slot_rbp];
    while(n < limit)
    {
        // Bounds-check BEFORE reading: rbp must point inside the copied
        // window with room for both the saved-rbp and return-address
        // slots, or the walk stops rather than risk reading past the
        // (possibly truncated) copied stack.
        if(rbp < stack_addr)
        {
            break;
        }
        const std::uint64_t offset = rbp - stack_addr;
        if(offset > stack_size || stack_size - offset < k_frame_link_bytes)
        {
            break;
        }

        std::uint64_t saved_rbp    = 0;
        std::uint64_t return_addr  = 0;
        const std::byte* frame_ptr = snap.stack.data() + offset;
        std::memcpy(&saved_rbp, frame_ptr, sizeof(saved_rbp));
        std::memcpy(&return_addr, frame_ptr + sizeof(saved_rbp), sizeof(return_addr));

        if(return_addr == 0)
        {
            break;
        }

        out[n++] = return_addr - 1;  // call site, see kernel_callchain_unwinder
        if(saved_rbp <= rbp)         // stack grows down: callers sit at higher addresses
        {
            break;
        }
        rbp = saved_rbp;
    }

    return n;
}

}  // namespace cpu_snap::sampling::unwind
