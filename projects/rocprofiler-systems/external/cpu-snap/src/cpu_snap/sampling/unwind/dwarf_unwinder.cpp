// dwarf_unwinder.cpp
//
// libdwfl-based DWARF stack unwinder; see header for the contract.
// The class body lives in `impl` here so that the public header does not
// pull <elfutils/libdw.h>, <elfutils/libdwfl.h>, or <sys/uio.h> into
// every sampling/sampler.hpp consumer.

#include "cpu_snap/sampling/unwind/dwarf_unwinder.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <elfutils/libdw.h>
#include <elfutils/libdwfl.h>

#include <link.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

namespace cpu_snap::sampling::unwind
{

namespace
{

namespace detail
{
// NOLINTBEGIN(readability-identifier-naming): codebase k_*-prefix convention
/** Number of registers libdwfl needs from us (RAX..R15 + RIP per SysV). */
inline constexpr std::uint32_t k_dwarf_n_regs_x86_64 = 17;

/** DWARF x86_64 register number (System V ABI fig 3.36) and its perf slot. */
struct reg_pair
{
    std::uint32_t dwarf;
    std::uint32_t slot;
};

inline constexpr std::array<reg_pair, k_dwarf_n_regs_x86_64> k_reg_map = {{
    {0, perf_slot_of(PERF_REG_X86_AX)},
    {1, perf_slot_of(PERF_REG_X86_DX)},
    {2, perf_slot_of(PERF_REG_X86_CX)},
    {3, perf_slot_of(PERF_REG_X86_BX)},
    {4, perf_slot_of(PERF_REG_X86_SI)},
    {5, perf_slot_of(PERF_REG_X86_DI)},
    {6, perf_slot_of(PERF_REG_X86_BP)},
    {7, perf_slot_of(PERF_REG_X86_SP)},
    {8, perf_slot_of(PERF_REG_X86_R8)},
    {9, perf_slot_of(PERF_REG_X86_R9)},
    {10, perf_slot_of(PERF_REG_X86_R10)},
    {11, perf_slot_of(PERF_REG_X86_R11)},
    {12, perf_slot_of(PERF_REG_X86_R12)},
    {13, perf_slot_of(PERF_REG_X86_R13)},
    {14, perf_slot_of(PERF_REG_X86_R14)},
    {15, perf_slot_of(PERF_REG_X86_R15)},
    {16, perf_slot_of(PERF_REG_X86_IP)},
}};

// NOLINTEND(readability-identifier-naming)
}  // namespace detail

}  // namespace

class dwarf_unwinder::impl
{
public:
    impl() noexcept
    : m_dwfl_callbacks{ make_callbacks() }
    , m_dwfl{ build_dwfl(m_dwfl_callbacks) }
    {}

    ~impl()
    {
        if(m_dwfl != nullptr)
        {
            dwfl_end(m_dwfl);
        }
    }

    impl(const impl&)            = delete;
    impl& operator=(const impl&) = delete;
    impl(impl&&)                 = delete;
    impl& operator=(impl&&)      = delete;

    std::uint32_t unwind(const dwarf_snapshot& snap, std::uint64_t* out,
                         std::uint32_t max_frames) noexcept
    {
        m_ts.used_live_read = false;
        const std::uint32_t n = unwind_frames(snap, out, max_frames);
        if(m_ts.used_live_read)
        {
            m_live_stack_read_samples.fetch_add(1, std::memory_order_relaxed);
        }
        return n;
    }

    [[nodiscard]] std::uint64_t live_stack_read_samples() const noexcept
    {
        return m_live_stack_read_samples.load(std::memory_order_relaxed);
    }

private:
    std::uint32_t unwind_frames(const dwarf_snapshot& snap, std::uint64_t* out,
                                std::uint32_t max_frames) noexcept
    {
        if(m_dwfl == nullptr || out == nullptr || max_frames == 0 || snap.regs == nullptr)
        {
            return 0;
        }

        // A sample taken on a `ret` has already run the epilogue, but the
        // CFI for that address (clang emits `.cfi_def_cfa %rsp, 8` without
        // cancelling the "rbp saved at CFA-16" rule) still claims rbp is on
        // the stack. libdwfl would reload rbp from the popped slot and derail
        // on the next frame, so unwind from the state after the return.
        if(max_frames > 1 && is_return_instruction(snap.regs[k_perf_slot_rip]))
        {
            if(const std::uint32_t n = unwind_from_return(snap, out, max_frames); n != 0)
            {
                return n;
            }
        }
        // Code without CFI (PLT stub, ld.so, hand-written asm) cannot be unwound
        // by rule. libdwfl then falls back to the rbp chain, which stops after
        // one frame or yields a garbage return address. When the top of the
        // stack is a return address after a call, the leaf is still at its entry
        // and the caller chain is intact, so continue from that return address.
        const std::uint32_t walked = walk(snap, out, max_frames);
        if(max_frames > 1 && !has_cfi(snap.regs[k_perf_slot_rip]))
        {
            if(const std::uint32_t n = unwind_from_return(snap, out, max_frames, true); n > 1)
            {
                return n;
            }
        }
        return walked;
    }

    /**
     * @brief Unwind a snapshot whose rip is on a return instruction.
     *
     * Emits rip as the first frame, then walks from the state the `ret`
     * produces: rip = [rsp], rsp += 8, every other register unchanged.
     * The caller walk starts at return_addr - 1 (inside the call), which is
     * also the call-site address every non-leaf frame is reported as.
     *
     * @param require_call Accept only a return address preceded by a call.
     * @return Number of frames written, or 0 when the return address cannot
     *         be read from the snapshot or is rejected (caller falls back to a
     *         plain walk).
     */
    std::uint32_t unwind_from_return(const dwarf_snapshot& snap, std::uint64_t* out,
                                     std::uint32_t max_frames,
                                     bool          require_call = false) noexcept
    {
        constexpr std::uint64_t k_word = sizeof(std::uint64_t);
        const std::uint64_t     rsp    = snap.regs[k_perf_slot_rsp];
        if(snap.stack_size < k_word || rsp < snap.stack_addr ||
           rsp - snap.stack_addr > snap.stack_size - k_word)
        {
            return 0;
        }
        std::uint64_t return_addr = 0;
        std::memcpy(&return_addr, snap.stack_data + (rsp - snap.stack_addr), k_word);
        if(return_addr == 0 || (require_call && !follows_call(return_addr)))
        {
            return 0;
        }

        std::array<std::uint64_t, k_perf_dwarf_n_regs_x86_64> caller_regs{};
        std::copy_n(snap.regs, caller_regs.size(), caller_regs.begin());
        caller_regs[k_perf_slot_rip] = return_addr - 1;
        caller_regs[k_perf_slot_rsp] = rsp + k_word;
        const dwarf_snapshot caller_snap{ caller_regs.data(), snap.stack_addr, snap.stack_data,
                                          snap.stack_size };

        out[0] = snap.regs[k_perf_slot_rip];
        const std::uint32_t n = walk(caller_snap, out + 1, max_frames - 1);
        if(n == 0)
        {
            return 0;
        }
        return n + 1;
    }

    /** True when the eh_frame or DWARF CFI of the module holding @p pc has a row for it. */
    bool has_cfi(std::uint64_t pc) const noexcept
    {
        Dwfl_Module* const module = dwfl_addrmodule(m_dwfl, pc);
        if(module == nullptr)
        {
            return false;
        }
        const auto covers = [pc](Dwarf_CFI* cfi, Dwarf_Addr bias) {
            if(cfi == nullptr)
            {
                return false;
            }
            Dwarf_Frame* frame = nullptr;
            if(dwarf_cfi_addrframe(cfi, pc - bias, &frame) != 0)
            {
                return false;
            }
            std::free(frame);
            return true;
        };
        Dwarf_Addr bias = 0;
        if(covers(dwfl_module_eh_cfi(module, &bias), bias))
        {
            return true;
        }
        return covers(dwfl_module_dwarf_cfi(module, &bias), bias);
    }

    /** True when the bytes before `return_addr` end a call instruction. */
    static bool follows_call(std::uint64_t return_addr) noexcept
    {
        constexpr std::uint8_t k_call_rel32    = 0xE8;
        constexpr std::uint8_t k_group5        = 0xFF;
        constexpr std::uint8_t k_modrm_rip_rel = 0x15;
        constexpr std::uint8_t k_modrm_reg_mask = 0xF8;
        constexpr std::uint8_t k_modrm_call_reg = 0xD0;

        std::array<std::uint8_t, 6> code{};
        const struct iovec          local{ code.data(), code.size() };
        // NOLINTBEGIN(performance-no-int-to-ptr)
        const struct iovec remote{ reinterpret_cast<void*>(
                                       static_cast<std::uintptr_t>(return_addr - code.size())),
                                   code.size() };
        // NOLINTEND(performance-no-int-to-ptr)
        if(::process_vm_readv(::getpid(), &local, 1, &remote, 1, 0) !=
           static_cast<ssize_t>(code.size()))
        {
            return false;
        }
        const bool rel32     = code[1] == k_call_rel32;
        const bool rip_rel   = code[0] == k_group5 && code[1] == k_modrm_rip_rel;
        const bool reg_call  = code[4] == k_group5 && (code[5] & k_modrm_reg_mask) == k_modrm_call_reg;
        return rel32 || rip_rel || reg_call;
    }

    /** True for ret, ret imm16 and their rep/bnd-prefixed forms. */
    static bool is_return_instruction(std::uint64_t pc) noexcept
    {
        constexpr std::uint8_t k_ret_near     = 0xC3;
        constexpr std::uint8_t k_ret_near_imm = 0xC2;
        constexpr std::uint8_t k_prefix_rep   = 0xF3;
        constexpr std::uint8_t k_prefix_bnd   = 0xF2;

        std::array<std::uint8_t, 2> code{};
        const struct iovec          local{ code.data(), code.size() };
        // NOLINTBEGIN(performance-no-int-to-ptr)
        const struct iovec remote{ reinterpret_cast<void*>(static_cast<std::uintptr_t>(pc)),
                                   code.size() };
        // NOLINTEND(performance-no-int-to-ptr)
        if(::process_vm_readv(::getpid(), &local, 1, &remote, 1, 0) !=
           static_cast<ssize_t>(code.size()))
        {
            return false;
        }
        const std::uint8_t opcode =
            (code[0] == k_prefix_rep || code[0] == k_prefix_bnd) ? code[1] : code[0];
        return opcode == k_ret_near || opcode == k_ret_near_imm;
    }

    /** Changes whenever the dynamic loader maps or unmaps a library. */
    static std::uint64_t loader_generation() noexcept
    {
        std::uint64_t generation = 0;
        ::dl_iterate_phdr(
            [](struct dl_phdr_info* info, std::size_t, void* data) {
                *static_cast<std::uint64_t*>(data) = info->dlpi_adds + info->dlpi_subs;
                return 1;
            },
            &generation);
        return generation;
    }

    /**
     * Rebuild the module list once the loader has mapped or unmapped something
     * since the last build; the cached CFI of unchanged modules is dropped with
     * the old Dwfl, which is cheap next to how rarely libraries come and go.
     * Skipped while the loader is unchanged: code that never resolves (JIT,
     * anonymous mappings) must not make every sample rebuild.
     */
    bool refresh_modules() noexcept
    {
        const std::uint64_t generation = loader_generation();
        if(generation == m_loader_generation)
        {
            return false;
        }
        m_loader_generation = generation;
        Dwfl* fresh         = build_dwfl(m_dwfl_callbacks);
        if(fresh == nullptr)
        {
            return false;
        }
        dwfl_end(m_dwfl);
        m_dwfl     = fresh;
        m_attached = false;
        return true;
    }

    bool attach() noexcept
    {
        if(!m_attached)
        {
            m_attached = dwfl_attach_state(m_dwfl, /*elf=*/nullptr, /*pid=*/k_synthetic_tid,
                                           &k_thread_callbacks, &m_ts);
        }
        return m_attached;
    }

    std::uint32_t walk(const dwarf_snapshot& snap, std::uint64_t* out,
                       std::uint32_t max_frames) noexcept
    {
        if(m_dwfl == nullptr || out == nullptr || max_frames == 0 || snap.regs == nullptr)
        {
            return 0;
        }

        // dwfl_attach_state binds thread state to the Dwfl and may be called
        // only ONCE per session; a second call fails and would yield 0 frames
        // on every sample after the first. Attach once with a STABLE arg
        // (&m_ts) and reuse it: each unwind swaps the per-call snapshot/output
        // into m_ts and reruns dwfl_getthread_frames. This keeps the reported
        // modules / .eh_frame cached across the many samples a single worker
        // drains.
        if(!attach())
        {
            return 0;
        }

        m_ts.snap     = &snap;
        m_ts.out      = out;
        m_ts.cap      = max_frames;
        m_ts.n        = 0;
        m_ts.iterated = false;

        // dwfl_getthread_frames returns 0 on full success, -1 on partial
        // (returned via callback ABORT or libdw error). Either way m_ts.n
        // holds whatever frames libdwfl produced before stopping.
        // 0 means the walk reached the bottom of the stack; anything else is an
        // abort, a libdw error or a frame without CFI, for instance in a library
        // loaded after the module list was reported.
        const int status = dwfl_getthread_frames(m_dwfl, k_synthetic_tid, &frame_cb, &m_ts);
        if(status != 0 && m_ts.n < m_ts.cap && refresh_modules() && attach())
        {
            m_ts.n        = 0;
            m_ts.iterated = false;
            (void) dwfl_getthread_frames(m_dwfl, k_synthetic_tid, &frame_cb, &m_ts);
        }
        return m_ts.n;
    }

    // The kernel never assigns tid 0 to userspace; we pass 1 as a
    // synthetic id so libdwfl's per-thread iteration terminates after
    // the single snapshot we want to unwind.
    // NOLINTBEGIN(readability-identifier-naming): codebase k_*-prefix convention
    static constexpr pid_t k_synthetic_tid = 1;
    // NOLINTEND(readability-identifier-naming)

    struct thread_state
    {
        const dwarf_snapshot* snap     = nullptr;
        std::uint64_t*        out      = nullptr;
        std::uint32_t         cap      = 0;
        std::uint32_t         n        = 0;
        bool                  iterated = false;
        bool                  used_live_read = false;
    };

    static Dwfl_Callbacks make_callbacks() noexcept
    {
        Dwfl_Callbacks cb{};
        cb.find_elf        = &dwfl_linux_proc_find_elf;
        cb.find_debuginfo  = &dwfl_standard_find_debuginfo;
        cb.section_address = &dwfl_offline_section_address;
        cb.debuginfo_path  = nullptr;
        return cb;
    }

    /**
     * Report every file-backed object the dynamic loader knows, each with the
     * range taken from its own program headers. dwfl_linux_proc_report derives
     * ranges from /proc/self/maps and may stretch a module over unrelated
     * mappings, which then shadow the libraries mapped inside that span.
     */
    static void report_loaded_modules(Dwfl* dwfl) noexcept
    {
        constexpr const char* k_main_executable = "/proc/self/exe";
        ::dl_iterate_phdr(
            [](struct dl_phdr_info* info, std::size_t, void* data) {
                const char* name = info->dlpi_name;
                if(name == nullptr || name[0] == '\0')
                {
                    name = k_main_executable;
                }
                else if(name[0] != '/')
                {
                    return 0;
                }
                (void) dwfl_report_elf(static_cast<Dwfl*>(data), name, name, /*fd=*/-1,
                                       static_cast<GElf_Addr>(info->dlpi_addr),
                                       /*add_p_vaddr=*/true);
                return 0;
            },
            dwfl);
    }

    static Dwfl* build_dwfl(Dwfl_Callbacks& callbacks) noexcept
    {
        Dwfl* dwfl = dwfl_begin(&callbacks);
        if(dwfl == nullptr)
        {
            return nullptr;
        }
        report_loaded_modules(dwfl);
        if(dwfl_report_end(dwfl, nullptr, nullptr) != 0)
        {
            dwfl_end(dwfl);
            return nullptr;
        }
        return dwfl;
    }

    static int frame_cb(Dwfl_Frame* state, void* arg) noexcept
    {
        auto* ts = static_cast<thread_state*>(arg);
        if(ts->n >= ts->cap)
        {
            return DWARF_CB_ABORT;
        }

        Dwarf_Addr program_counter = 0;
        bool       is_activation   = false;
        if(!dwfl_frame_pc(state, &program_counter, &is_activation))
        {
            return DWARF_CB_ABORT;
        }

        // A caller frame's pc is a return address; report the call site.
        auto ip = static_cast<std::uint64_t>(program_counter);
        if(!is_activation && ip != 0)
        {
            --ip;
        }
        ts->out[ts->n++] = ip;
        return (ts->n >= ts->cap) ? DWARF_CB_ABORT : DWARF_CB_OK;
    }

    static pid_t cb_next_thread(Dwfl* /*dwfl*/, void* dwfl_arg,
                                void** thread_argp) noexcept
    {
        auto* ts = static_cast<thread_state*>(dwfl_arg);
        if(ts->iterated)
        {
            *thread_argp = nullptr;
            return 0;
        }
        ts->iterated = true;
        *thread_argp = ts;
        return k_synthetic_tid;
    }

    static bool cb_get_thread(Dwfl* /*dwfl*/, pid_t tid, void* dwfl_arg,
                              void** thread_argp) noexcept
    {
        if(tid != k_synthetic_tid)
        {
            return false;
        }
        *thread_argp = dwfl_arg;
        return true;
    }

    static bool cb_memory_read(Dwfl* /*dwfl*/, Dwarf_Addr addr, Dwarf_Word* result,
                               void* dwfl_arg) noexcept
    {
        auto*                 ts = static_cast<thread_state*>(dwfl_arg);
        const dwarf_snapshot* sn = ts->snap;

        if(const auto offset = snapshot_word_offset(*sn, static_cast<std::uint64_t>(addr)))
        {
            std::memcpy(result, sn->stack_data + *offset, sizeof(Dwarf_Word));
            return true;
        }

        ts->used_live_read = true;

        // Outside the snapshot stack: fall back to in-process memory so
        // .eh_frame / .rodata lookups in any loaded binary succeed.
        // process_vm_readv returns EFAULT instead of faulting on a stale or
        // unmapped address, so a corrupt snapshot cannot crash the worker.
        Dwarf_Word         buf{};
        const struct iovec local{ &buf, sizeof(Dwarf_Word) };
        // process_vm_readv ABI requires the remote address as void*; the
        // int-to-ptr cast is unavoidable here.
        // NOLINTBEGIN(performance-no-int-to-ptr)
        const struct iovec remote{
            reinterpret_cast<void*>(static_cast<std::uintptr_t>(addr)),
            sizeof(Dwarf_Word)
        };
        // NOLINTEND(performance-no-int-to-ptr)
        const ssize_t bytes_read =
            ::process_vm_readv(::getpid(), &local, 1, &remote, 1, 0);
        if(bytes_read != static_cast<ssize_t>(sizeof(Dwarf_Word)))
        {
            return false;
        }
        *result = buf;
        return true;
    }

    static bool cb_set_initial_registers(Dwfl_Thread* thread, void* thread_arg) noexcept
    {
        auto*                 ts = static_cast<thread_state*>(thread_arg);
        const dwarf_snapshot* sn = ts->snap;

        std::array<Dwarf_Word, detail::k_dwarf_n_regs_x86_64> dwarf_regs{};
        for(const auto& [dwarf, slot] : detail::k_reg_map)
            dwarf_regs[dwarf] = sn->regs[slot];

        return dwfl_thread_state_registers(thread, /*firstreg=*/0,
                                           /*nregs=*/detail::k_dwarf_n_regs_x86_64,
                                           dwarf_regs.data());
    }

    // NOLINTBEGIN(readability-identifier-naming): codebase k_*-prefix convention
    static inline const Dwfl_Thread_Callbacks k_thread_callbacks = {
        &cb_next_thread,            //
        &cb_get_thread,             //
        &cb_memory_read,            //
        &cb_set_initial_registers,  //
        /*detach=*/nullptr,         //
        /*thread_detach=*/nullptr   //
    };
    // NOLINTEND(readability-identifier-naming)

    // Read before the first module report so a library mapped in between is
    // seen as a change.
    std::uint64_t  m_loader_generation = loader_generation();
    Dwfl_Callbacks m_dwfl_callbacks;
    Dwfl*          m_dwfl;

    // Thread state bound once by dwfl_attach_state and reused for every
    // unwind() (see the comment in unwind). cb_next_thread reads/clears
    // m_ts.iterated; the other callbacks read m_ts.snap/out/cap/n.
    thread_state m_ts{};
    bool         m_attached = false;

    std::atomic<std::uint64_t> m_live_stack_read_samples{ 0 };
};

// ----- public delegating wrapper ------------------------------------------

dwarf_unwinder::dwarf_unwinder() noexcept
{
    // Match the previous noexcept contract: any allocation failure
    // leaves m_impl null; unwind() then returns 0 just as it does when
    // the underlying Dwfl* failed to initialize.
    try
    {
        m_impl = std::make_unique<impl>();
    } catch(...)  // out of memory only; a null impl makes unwind() return 0 (noexcept ctor contract)
    {
        m_impl.reset();
    }
}

dwarf_unwinder::~dwarf_unwinder() = default;

std::uint32_t
dwarf_unwinder::unwind(const dwarf_snapshot& snap, std::uint64_t* out,
                       std::uint32_t max_frames) noexcept
{
    if(m_impl == nullptr)
    {
        return 0;
    }
    return m_impl->unwind(snap, out, max_frames);
}

std::uint64_t
dwarf_unwinder::live_stack_read_samples() const noexcept
{
    return m_impl == nullptr ? 0 : m_impl->live_stack_read_samples();
}

}  // namespace cpu_snap::sampling::unwind
