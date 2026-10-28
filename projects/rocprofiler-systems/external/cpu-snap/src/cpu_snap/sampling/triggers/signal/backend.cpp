// triggers/signal/backend.cpp
//
// Out-of-line bodies for cpu_snap::sampling::triggers::signal::backend,
// including the async-signal-safety-critical signal handler. _GNU_SOURCE is
// set PRIVATE on the sampling target, so REG_RIP / SIGEV_THREAD_ID /
// sigev._sigev_un._tid / ucontext gregs are visible here.
//
// =========================== ASYNC-SIGNAL-SAFETY ===========================
// signal_handler() and everything it calls must use ONLY async-signal-safe
// operations. The audited handler does exactly:
//   * clock_gettime(<configured clock>)       -- AS-safe (vDSO/syscall)
//   * syscall(SYS_gettid) (cached in TLS)    -- AS-safe
//   * atomic loads on the published ring table + relaxed/acquire/release
//     atomics on the per-thread ring                                  -- AS-safe
//   * reads of ucontext_t gregs              -- plain memory reads
//   * process_vm_readv() of the user stack into a PREALLOCATED ring slot
//     (syscall, AS-safe; short-reads on EFAULT instead of faulting)   -- AS-safe
//   * mmap() + sigaltstack() on the first delivery to a thread without an
//     alternate stack (TLS-guarded)                                    -- AS-safe
//   * write()/eventfd write of a u64 (once per ring fill-up) -- AS-safe
//   * errno is saved on entry and restored on every exit path
// It performs NO malloc/new, NO locks, NO std::vector growth, NO std::function,
// NO libdw, NO stdio, and never throws. All such work is deferred to the worker
// (drain_into) or to attach_tid (ring allocation).
//
// Per-thread slot lookup from the handler: a fixed-size, lock-free, open-
// addressed table (tid -> ring*) is published by the worker (attach/detach,
// off-handler) with atomic stores; the handler reads it with atomic loads and
// caches the result in a thread_local pointer, so steady-state lookup is one
// TLS read.
//
// sigaltstack: we use option (b) -- the handler installs its own per-thread
// alternate stack on the FIRST delivery to a thread that has none, so all
// SUBSEQUENT signals on that thread run on a dedicated stack via SA_ONSTACK. A
// thread that already registered an alternate stack keeps it.
// The very first delivery runs on the interrupted thread's own stack; the
// handler's footprint is tiny and uses only fixed buffers, so the small-deep-
// stack risk for that single first signal is acceptable. We deliberately do
// NOT try to set another thread's altstack from the worker (impossible:
// sigaltstack only affects the calling thread).

#include "cpu_snap/sampling/triggers/signal/backend.hpp"

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include <poll.h>
#include <pthread.h>
#include <signal.h>  // NOLINT(modernize-deprecated-headers): POSIX sigaction/SIGRTMIN/SIGEV_THREAD_ID
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/ucontext.h>
#include <sys/uio.h>
#include <time.h>  // NOLINT(modernize-deprecated-headers): POSIX timer_create/timer_settime
#include <ucontext.h>
#include <unistd.h>

#include "cpu_snap/error.hpp"
#include "cpu_snap/sampling/streaming_pipeline.hpp"
#include "cpu_snap/sampling/time.hpp"
#include "cpu_snap/sampling/triggers/signal/ring_table.hpp"
#include "cpu_snap/sampling/triggers/signal/snapshot.hpp"
#include "cpu_snap/sampling/unwind/perf_regs.hpp"

namespace cpu_snap::sampling::triggers::signal
{

namespace
{

constexpr std::uint64_t k_ns_per_sec = 1'000'000'000ULL;

// File-scope handler state. Constant-initialized (static storage), so the table
// is usable before any constructor runs.
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
ring_table                 g_ring_table;
std::atomic<int>           g_wake_fd{ -1 };
std::atomic<std::uint32_t> g_stack_bytes{ 0 };
// Clock the handler stamps snapshots with; written under the g_backend_active
// guard, read lock-free (and async-signal-safely) in the handler.
std::atomic<clockid_t>     g_sample_clock{ CLOCK_MONOTONIC };
static_assert(std::atomic<clockid_t>::is_always_lock_free);
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

// Per-thread cached tid + the mmap'd alternate signal stack for this thread.
//
// ASYNC-SIGNAL-SAFETY: both are declared initial-exec (tls_model). Under the
// default general-dynamic TLS model the FIRST access to a thread_local from a
// dlopened DSO routes through __tls_get_addr, which lazily malloc()s the
// module's TLS block -- and malloc is NOT async-signal-safe, so a signal that
// interrupts the host mid-malloc would self-deadlock. initial-exec resolves the
// slot via a fixed thread-pointer offset with no allocation and no lazy-init
// guard (these objects are trivially constructible), so the handler's first
// touch is AS-safe. The altstack buffer is mmap()'d (an AS-safe syscall), never
// malloc()'d. A pointer + a pid_t fit the static-TLS surplus trivially, so
// initial-exec does not jeopardize dlopen of the autoload .so.
//
// NOTE: the ring pointer is deliberately NOT cached in TLS. A given thread can
// be sampled by one backend, see that backend destroyed, then be sampled again
// by a later backend; a cached ring* would dangle across that boundary. The
// global ring table is the single source of truth and is retired on detach, so
// the handler looks the ring up there every time (hash + ~1 probe).
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
__attribute__((tls_model("initial-exec"))) thread_local pid_t         t_cached_tid = 0;
__attribute__((tls_model("initial-exec"))) thread_local std::uint8_t* t_altstack   = nullptr;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

// glibc greg indices in the PERF_REG_X86_* output-slot order every
// regs_and_stack unwinder (dwarf_unwinder, fp_copy_unwinder) consumes
// (output slot i <- gregs[k_greg_order[i]]). RSP lands at output slot 7, RIP
// at slot 8.
constexpr std::array<int, unwind::k_perf_dwarf_n_regs_x86_64> k_greg_order = {
    REG_RAX, REG_RBX, REG_RCX, REG_RDX, REG_RSI, REG_RDI, REG_RBP, REG_RSP,
    REG_RIP, REG_EFL, REG_R8,  REG_R9,  REG_R10, REG_R11, REG_R12, REG_R13,
    REG_R14, REG_R15,
};

// Output-slot index of RSP within the packed reg array (k_greg_order).
constexpr std::size_t k_rsp_slot = 7;

// Pack glibc ucontext gregs into the output-slot order above. Plain memory
// reads of a fixed table -- async-signal-safe.
void pack_regs(const ucontext_t* uc,
               std::array<std::uint64_t, unwind::k_perf_dwarf_n_regs_x86_64>& out) noexcept
{
    const auto& gregs = uc->uc_mcontext.gregs;
    for(std::size_t i = 0; i < k_greg_order.size(); ++i)
    {
        out[i] = static_cast<std::uint64_t>(gregs[k_greg_order[i]]);
    }
}

constexpr std::size_t k_altstack_bytes = std::size_t{ 64 } * 1024;

// Give a thread that has no alternate signal stack one of ours, so every later
// delivery runs on it via SA_ONSTACK. sigreturn restores the stack recorded in
// the signal frame, so the new stack must be written back into the frame's
// uc_stack or the kernel undoes the sigaltstack() call when the handler
// returns. A thread that already has one (the host's, or ours from an earlier
// delivery) is left alone. The buffer is mmap()'d (AS-safe) once per thread and
// reused if the thread's stack is ever disabled; the worker reclaims it after
// the thread has exited (see adopt_altstack / reap_altstacks).
void install_altstack_if_missing(ucontext_t* uc, snapshot_ring& ring) noexcept
{
    if((uc->uc_stack.ss_flags & SS_DISABLE) == 0)
    {
        return;
    }
    if(t_altstack == nullptr)
    {
        void* buf = ::mmap(nullptr, k_altstack_bytes, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if(buf == MAP_FAILED)
        {
            return;  // run on the interrupted stack; retried on the next delivery
        }
        t_altstack = static_cast<std::uint8_t*>(buf);
        ring.set_altstack(buf);
    }
    stack_t ss{};
    ss.ss_sp    = t_altstack;
    ss.ss_size  = k_altstack_bytes;
    ss.ss_flags = 0;
    if(::sigaltstack(&ss, nullptr) == 0)
    {
        uc->uc_stack = ss;
    }
}

// The handler interrupts arbitrary host code, possibly between a failing
// syscall and its errno check, so errno must be restored on every exit path.
class errno_guard
{
public:
    errno_guard() noexcept : m_saved{ errno } {}
    ~errno_guard() { errno = m_saved; }

    errno_guard(const errno_guard&)            = delete;
    errno_guard& operator=(const errno_guard&) = delete;

private:
    int m_saved;
};

// Fills one ring slot from the interrupted context. Runs inside signal_handler
// with the ring's table slot entered; see the audit at the top of this file.
void capture_sample(snapshot_ring* ring, pid_t tid, void* ucontext) noexcept
{
    install_altstack_if_missing(static_cast<ucontext_t*>(ucontext), *ring);

    snapshot* slot = ring->acquire();
    if(slot == nullptr)
    {
        // Ring full -- acquire() already bumped the lost counter.
        return;
    }

    timespec ts{};
    (void) ::clock_gettime(g_sample_clock.load(std::memory_order_relaxed), &ts);
    slot->ts_ns = static_cast<std::uint64_t>(ts.tv_sec) * k_ns_per_sec +
                  static_cast<std::uint64_t>(ts.tv_nsec);
    slot->tid = tid;

    const auto* uc = static_cast<const ucontext_t*>(ucontext);
    pack_regs(uc, slot->regs);
    const std::uint64_t rsp = slot->regs[k_rsp_slot];
    slot->stack_base_addr   = rsp;

    // Bounded copy of the user stack starting at the interrupted RSP. Clamp to
    // the slot capacity; the slot buffer is preallocated, never grown here.
    std::uint32_t want = g_stack_bytes.load(std::memory_order_acquire);
    if(want > ring->stack_capacity())
    {
        want = ring->stack_capacity();
    }
    // Copy via process_vm_readv rather than a raw memcpy: near the top of the
    // thread's stack the fixed-size slab can extend past the stack mapping, and
    // a memcpy would SIGSEGV inside the handler. process_vm_readv is a plain
    // syscall (async-signal-safe) and returns a SHORT read on EFAULT instead of
    // faulting, so we record exactly the bytes that were actually readable.
    iovec local{};
    local.iov_base = slot->stack.get();
    local.iov_len  = want;
    iovec remote{};
    // NOLINTNEXTLINE(performance-no-int-to-ptr): rsp is a sampled VA by nature
    remote.iov_base = reinterpret_cast<void*>(rsp);
    remote.iov_len  = want;
    const ssize_t got =
        ::process_vm_readv(slot->tid, &local, 1, &remote, 1, 0);
    slot->stack_size = (got > 0) ? static_cast<std::uint32_t>(got) : 0;

    const std::size_t depth = ring->commit();

    // Wake the worker only as the ring crosses its wake depth; below that the
    // worker's poll timeout collects the samples. eventfd write of 1 is
    // async-signal-safe.
    if(depth != ring->wake_depth())
    {
        return;
    }
    const int fd = g_wake_fd.load(std::memory_order_acquire);
    if(fd >= 0)
    {
        const std::uint64_t one = 1;
        // Best-effort; EAGAIN on a saturated counter is fine (worker will drain).
        const ssize_t wr = ::write(fd, &one, sizeof(one));
        (void) wr;
    }
}

// =========================== THE SIGNAL HANDLER ===========================
// SA_SIGINFO | SA_RESTART | SA_ONSTACK. See the async-signal-safety audit at
// the top of this file. NOLINT: this function is the documented AS-safe path.
void signal_handler(int /*signo*/, siginfo_t* /*info*/, void* ucontext) noexcept
{
    const errno_guard preserve_errno;

    // Resolve our tid once per thread, then cache in TLS.
    pid_t tid = t_cached_tid;
    if(tid == 0)
    {
        tid          = static_cast<pid_t>(::syscall(SYS_gettid));
        t_cached_tid = tid;
    }

    // Not (yet) attached, or detached: nothing to publish into, drop. While the
    // slot is entered the worker will not free the ring or close the wake fd.
    std::size_t    slot_idx = 0;
    snapshot_ring* ring     = g_ring_table.enter(tid, slot_idx);
    if(ring == nullptr)
    {
        return;
    }
    capture_sample(ring, tid, ucontext);
    g_ring_table.leave(slot_idx);
}

// Disposition the host had for the capture signal before we claimed it; restored
// by release_signal(). Only meaningful while the capture handler is installed.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
struct sigaction g_prev_action
{};

struct sigaction make_capture_action() noexcept
{
    struct sigaction sa
    {};
    sa.sa_sigaction = &signal_handler;
    sa.sa_flags     = SA_SIGINFO | SA_RESTART | SA_ONSTACK;
    ::sigemptyset(&sa.sa_mask);
    return sa;
}

bool is_capture_action(const struct sigaction& sa) noexcept
{
    return (sa.sa_flags & SA_SIGINFO) != 0 && sa.sa_sigaction == &signal_handler;
}

bool is_default_action(const struct sigaction& sa) noexcept
{
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-cstyle-cast): SIG_DFL is a libc macro
    return (sa.sa_flags & SA_SIGINFO) == 0 && sa.sa_handler == SIG_DFL;
}

// Install the capture handler for @p signal_no and remember the previous
// disposition. Returns EBUSY if the host already owns the signal.
int claim_signal(int signal_no) noexcept
{
    struct sigaction current
    {};
    if(::sigaction(signal_no, nullptr, &current) != 0)
    {
        return errno;
    }
    if(is_capture_action(current))
    {
        return 0;
    }
    if(!is_default_action(current))
    {
        return EBUSY;
    }
    g_prev_action      = current;
    const auto capture = make_capture_action();
    return (::sigaction(signal_no, &capture, nullptr) != 0) ? errno : 0;
}

void release_signal(int signal_no) noexcept
{
    (void) ::sigaction(signal_no, &g_prev_action, nullptr);
}

// Flush every still-pending instance of @p signal_no out of the process.
//
// A timer armed with SIGEV_THREAD_ID queues its expiration into the TARGET
// thread's private pending set. If that thread has the signal blocked (the host
// app may legitimately block this RT signal in some threads) the instance is
// never delivered, and timer_delete does NOT reclaim it (timer_delete(2): "the
// treatment of any pending signal generated by the deleted timer is
// unspecified"). Such instances would otherwise accumulate across sampler
// lifecycles and eventually exhaust RLIMIT_SIGPENDING, making timer_create fail
// with EAGAIN -- the signal backend would become effectively one-shot-per-
// process.
//
// Setting the disposition to SIG_IGN is the one cross-thread lever that clears
// them: signal disposition is a per-process attribute, so the kernel discards
// matching pending signals from the shared queue AND from every thread's
// private pending set (POSIX: setting SIG_IGN on a pending signal discards it,
// blocked or not). We then restore the capture handler. This runs only at
// teardown, after every timer is deleted and the worker is joined, so no
// expiration is in flight and no sample is lost by the brief ignore window.
void flush_pending_signal(int signal_no) noexcept
{
    struct sigaction ign
    {};
    ign.sa_handler = SIG_IGN;  // NOLINT(cppcoreguidelines-pro-type-cstyle-cast): SIG_IGN is a libc macro
    ign.sa_flags   = 0;
    ::sigemptyset(&ign.sa_mask);
    if(::sigaction(signal_no, &ign, nullptr) != 0)
    {
        return;
    }
    const auto capture = make_capture_action();
    (void) ::sigaction(signal_no, &capture, nullptr);
}

// Handler-mapped alternate stacks whose owning thread was detached while still
// alive. They stay mapped (the thread has them registered) until the thread is
// gone. Worker-side only; never touched by the handler.
struct orphan_altstack
{
    pid_t tid;
    void* base;
};

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
std::mutex                    g_orphan_mutex;
std::vector<orphan_altstack>  g_orphan_altstacks;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

// Kernel clock id of the CPU-time clock of thread @p tid in this process
// (MAKE_THREAD_CPUCLOCK); any thread of the process may create a timer on it.
clockid_t thread_cpu_clock(pid_t tid) noexcept
{
    constexpr std::uint32_t k_cpuclock_sched_perthread = 6;
    return static_cast<clockid_t>((~static_cast<std::uint32_t>(tid) << 3) |
                                  k_cpuclock_sched_perthread);
}

bool thread_has_exited(pid_t tid) noexcept
{
    return ::syscall(SYS_tgkill, ::getpid(), tid, 0) == -1 && errno == ESRCH;
}

void adopt_altstack(pid_t tid, void* base) noexcept
{
    if(base == nullptr)
    {
        return;
    }
    try
    {
        const std::lock_guard lock{ g_orphan_mutex };
        g_orphan_altstacks.push_back({ tid, base });
    }
    // NOLINTNEXTLINE(bugprone-empty-catch): out of memory only; the stack is left mapped
    catch(...)
    {
    }
}

void reap_altstacks() noexcept
{
    const std::lock_guard lock{ g_orphan_mutex };
    std::erase_if(g_orphan_altstacks, [](const orphan_altstack& orphan) {
        if(!thread_has_exited(orphan.tid))
        {
            return false;
        }
        ::munmap(orphan.base, k_altstack_bytes);
        return true;
    });
}

// Bumped in every forked child. A backend built before the fork is stale in the
// child: its timers, rings and eventfd registration describe the parent.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::atomic<unsigned> g_fork_epoch{ 0 };

// Single-active-backend guard: the process-wide handler routes into one
// backend's rings at a time (via the global ring table). Concurrent backends
// would alias the table.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::atomic<bool> g_backend_active{ false };

// Runs in the single-threaded forked child. POSIX timers are not inherited, no
// handler can be in flight, and the thread id cached by the forking thread
// belongs to the parent, so the child starts again from an empty registry.
void reset_after_fork() noexcept
{
    t_cached_tid = 0;
    g_ring_table.clear();
    g_wake_fd.store(-1, std::memory_order_relaxed);
    g_sample_clock.store(CLOCK_MONOTONIC, std::memory_order_relaxed);
    g_backend_active.store(false, std::memory_order_relaxed);
    g_fork_epoch.fetch_add(1, std::memory_order_relaxed);
}

// Registered at load time so it runs before the sampler's own atfork handler,
// which calls close_all() on the (by then stale) backend.
[[maybe_unused]] const int g_atfork_registered =
    ::pthread_atfork(nullptr, nullptr, &reset_after_fork);

}  // namespace

backend::backend(config cfg)
: m_cfg{ cfg }
, m_frame_buf(cfg.callchain_depth)
{
    bool expected = false;
    if(!g_backend_active.compare_exchange_strong(expected, true,
                                                 std::memory_order_acq_rel))
    {
        throw usage_error(
            "backend: another signal backend is already active");
    }

    m_wake_fd = ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    if(m_wake_fd < 0)
    {
        const int err = errno;
        g_backend_active.store(false, std::memory_order_release);
        throw environment_error::from_errno(err, "backend: eventfd");
    }

    if(const int err = claim_signal(m_cfg.signal_no); err != 0)
    {
        ::close(m_wake_fd);
        m_wake_fd = -1;
        g_backend_active.store(false, std::memory_order_release);
        throw environment_error::from_errno(
            err, err == EBUSY ? "backend: the realtime signal already has a host handler"
                              : "backend: sigaction");
    }

    m_fork_epoch = g_fork_epoch.load(std::memory_order_relaxed);
    g_stack_bytes.store(m_cfg.stack_snapshot_bytes, std::memory_order_release);
    g_sample_clock.store(utils::clock_id_of(m_cfg.clock), std::memory_order_release);
    g_wake_fd.store(m_wake_fd, std::memory_order_release);
}

backend::~backend()
{
    close_all();
    if(!stale())
    {
        release_signal(m_cfg.signal_no);
        g_wake_fd.store(-1, std::memory_order_release);
        g_sample_clock.store(CLOCK_MONOTONIC, std::memory_order_release);
    }
    // A handler that never left a retired slot may still touch its ring and the
    // wake fd: leak both rather than risk a use-after-free or a write to a
    // recycled descriptor.
    const bool handlers_stuck = !m_retired.empty();
    for(auto& retired : m_retired)
    {
        (void) retired.ring.release();
    }
    if(m_wake_fd >= 0 && !handlers_stuck)
    {
        ::close(m_wake_fd);
        m_wake_fd = -1;
    }
    if(!stale())
    {
        g_backend_active.store(false, std::memory_order_release);
    }
}

bool backend::stale() const noexcept
{
    return m_fork_epoch != g_fork_epoch.load(std::memory_order_relaxed);
}

void backend::arm_timer(timer_t timer, bool first_arm) const noexcept
{
    const std::uint64_t period_ns =
        (m_cfg.freq_hz > 0) ? (k_ns_per_sec / m_cfg.freq_hz) : k_ns_per_sec;

    itimerspec its{};
    // tv_nsec is POSIX `long`; NOLINT matches the codebase convention (see
    // kernel_drain.hpp sysconf usage).
    its.it_interval.tv_sec  = static_cast<time_t>(period_ns / k_ns_per_sec);
    its.it_interval.tv_nsec =
        static_cast<long>(period_ns % k_ns_per_sec);  // NOLINT(google-runtime-int)

    std::uint64_t first_ns = period_ns;
    if(first_arm && m_cfg.start_delay_ns > 0)
    {
        first_ns = m_cfg.start_delay_ns;
    }
    its.it_value.tv_sec  = static_cast<time_t>(first_ns / k_ns_per_sec);
    its.it_value.tv_nsec =
        static_cast<long>(first_ns % k_ns_per_sec);  // NOLINT(google-runtime-int)

    (void) ::timer_settime(timer, 0, &its, nullptr);
}

void backend::attach_tid(pid_t tid)
{
    if(m_timers.contains(tid))
    {
        return;
    }

    reclaim_retired();

    // Reserve the map node first: nothing that can throw may run after the ring
    // is published or the timer is armed, or the handler would outlive its ring.
    const auto it = m_timers.try_emplace(tid).first;
    thread_state& st = it->second;
    try
    {
        // Allocate the per-thread ring OFF the signal path, then publish it before
        // arming the timer so the first tick already has a ring to fill.
        st.ring = std::make_unique<snapshot_ring>(
            m_cfg.ring_capacity, m_cfg.stack_snapshot_bytes);
        snapshot_ring* ring_raw = st.ring.get();
        if(!g_ring_table.publish(tid, ring_raw))
        {
            throw resource_error(std::error_code(ENOSPC, std::generic_category()),
                                 "backend: tid table full");
        }

        sigevent sev{};
        sev.sigev_notify           = SIGEV_THREAD_ID;
        sev.sigev_signo            = m_cfg.signal_no;
        sev._sigev_un._tid         = tid;  // deliver to this specific kernel tid

        const clockid_t clock = m_cfg.cpu_time ? thread_cpu_clock(tid) : CLOCK_MONOTONIC;
        if(::timer_create(clock, &sev, &st.timer) != 0)
        {
            const int err = errno;
            (void) g_ring_table.retire(tid);
            // The thread exited between enumeration and arming -- silent. A CPU
            // clock of a vanished thread is rejected with EINVAL.
            if(err == ESRCH || (m_cfg.cpu_time && err == EINVAL))
            {
                m_timers.erase(it);
                return;
            }
            throw environment_error::from_errno(err, "backend: timer_create");
        }
    } catch(...)
    {
        m_timers.erase(it);
        throw;
    }

    if(m_armed)
    {
        arm_timer(st.timer, /*first_arm=*/true);
    }
    // Publish the live count for tracked_tids() (read off-thread); the worker is
    // the only mutator of m_timers, so reading size() here is race-free.
    m_tracked_count.store(m_timers.size(), std::memory_order_relaxed);
}

void backend::detach_tid(pid_t tid) noexcept
{
    auto it = m_timers.find(tid);
    if(it == m_timers.end())
    {
        return;
    }
    (void) ::timer_delete(it->second.timer);
    retire_ring(tid, std::move(it->second.ring));
    m_timers.erase(it);
    m_tracked_count.store(m_timers.size(), std::memory_order_relaxed);
    reclaim_retired();
}

void backend::retire_ring(pid_t tid, std::unique_ptr<snapshot_ring> ring) noexcept
{
    const std::size_t slot = g_ring_table.retire(tid);
    if(!ring)
    {
        return;
    }
    try
    {
        m_retired.emplace_back(std::move(ring), tid, slot);
    }
    // NOLINTNEXTLINE(bugprone-empty-catch): out of memory only; leaking beats a use-after-free
    catch(...)
    {
        (void) ring.release();
    }
}

void backend::reclaim_retired() noexcept
{
    std::erase_if(m_retired, [](retired_ring& retired) {
        if(retired.slot != ring_table::k_npos && !g_ring_table.idle(retired.slot))
        {
            return false;
        }
        adopt_altstack(retired.tid, retired.ring->altstack());
        return true;
    });
    reap_altstacks();
}

int backend::wait(int timeout_ms) noexcept
{
    if(m_wake_fd < 0)
    {
        return 0;
    }
    pollfd pfd{};
    pfd.fd     = m_wake_fd;
    pfd.events = POLLIN;
    const int ready = ::poll(&pfd, 1, timeout_ms);
    if(ready < 0)
    {
        return (errno == EINTR) ? 0 : -1;
    }
    if(ready == 0)
    {
        return 0;
    }
    // Drain the eventfd counter so the next poll blocks until the next tick.
    std::uint64_t drain_val = 0;
    const ssize_t rd        = ::read(m_wake_fd, &drain_val, sizeof(drain_val));
    (void) rd;
    return 1;
}

void backend::drain_into(internal::streaming_pipeline& pipeline,
                         unwind::unwinder&           unw) noexcept
{
    std::visit(
        [this, &pipeline](auto& concrete_unwinder) {
            for(auto& [tid, st] : m_timers)
            {
                snapshot_ring* ring = st.ring.get();
                if(ring == nullptr)
                {
                    continue;
                }
                for(const snapshot* snap = ring->pop(); snap != nullptr;
                    snap = ring->pop())
                {
                    const std::uint64_t leaf = snap->regs[unwind::k_perf_slot_rip];

                    // The regs are already packed in PERF_REG_X86_* slot
                    // order (RSP at slot 7, RIP at slot 8), and
                    // stack_base_addr is the interrupted RSP that the stack
                    // copy starts at -- so no repacking is needed.
                    const unwind::machine_snapshot ms{
                        .ready_frames = {},
                        .regs         = snap->regs.data(),
                        .stack_addr   = snap->stack_base_addr,
                        .stack        = std::span<const std::byte>{
                            reinterpret_cast<const std::byte*>(snap->stack.get()),
                            snap->stack_size },
                    };
                    const std::size_t n_frames = concrete_unwinder.unwind(
                        ms, std::span<std::uint64_t>{ m_frame_buf });

                    if(n_frames > 0)
                    {
                        pipeline.on_sample(snap->tid, snap->ts_ns,
                                           std::span<const std::uint64_t>{
                                               m_frame_buf.data(), n_frames });
                    }
                    else
                    {
                        // Unwind produced nothing (e.g. truncated / short stack
                        // copy): fall back to the single leaf IP so the sample
                        // is not lost.
                        pipeline.on_sample(snap->tid, snap->ts_ns,
                                           std::span<const std::uint64_t>{ &leaf, 1 });
                    }
                    ring->consume();
                }

                // Surface ring overflow as lost, mirroring perf
                // PERF_RECORD_LOST. Only the delta since the last drain is
                // forwarded.
                const std::uint64_t total_lost = ring->lost();
                if(total_lost > st.forwarded_lost)
                {
                    pipeline.on_lost(total_lost - st.forwarded_lost);
                    st.forwarded_lost = total_lost;
                }
            }
        },
        unw);
}

void backend::disable_all() noexcept
{
    m_armed = false;
    const itimerspec disarm{};  // all-zero it_value disarms the timer
    for(auto& [tid, st] : m_timers)
    {
        (void) ::timer_settime(st.timer, 0, &disarm, nullptr);
    }
}

void backend::enable_all() noexcept
{
    m_armed = true;
    for(auto& [tid, st] : m_timers)
    {
        arm_timer(st.timer, /*first_arm=*/false);
    }
}

void backend::close_all() noexcept
{
    if(stale())
    {
        m_timers.clear();
        m_retired.clear();
        m_tracked_count.store(0, std::memory_order_relaxed);
        return;
    }
    g_wake_fd.store(-1, std::memory_order_release);
    const itimerspec disarm{};  // all-zero it_value disarms the timer
    for(auto& [tid, st] : m_timers)
    {
        // Disarm before delete so no further expirations queue during teardown,
        // then delete the timer and retire the table entry.
        (void) ::timer_settime(st.timer, 0, &disarm, nullptr);
        (void) ::timer_delete(st.timer);
        retire_ring(tid, std::move(st.ring));
    }
    m_timers.clear();
    m_tracked_count.store(0, std::memory_order_relaxed);

    // Reclaim any expirations that were already queued (into a blocked target
    // thread's private pending set) before the timers were deleted; otherwise
    // they leak across lifecycles and exhaust RLIMIT_SIGPENDING. See
    // flush_pending_signal.
    flush_pending_signal(m_cfg.signal_no);

    // Handlers already inside a retired slot finish within microseconds; wait
    // for them so the rings can be freed. A thread stopped mid-handler keeps its
    // ring parked until a later reclaim.
    constexpr auto k_quiesce_timeout = std::chrono::milliseconds{ 200 };
    constexpr auto k_quiesce_poll    = std::chrono::microseconds{ 100 };
    const auto     deadline          = std::chrono::steady_clock::now() + k_quiesce_timeout;
    reclaim_retired();
    while(!m_retired.empty() && std::chrono::steady_clock::now() < deadline)
    {
        std::this_thread::sleep_for(k_quiesce_poll);
        reclaim_retired();
    }

    // Ready for a later attach_tid (the sampler restarts after stop()).
    m_armed = true;
    g_wake_fd.store(m_wake_fd, std::memory_order_release);
}

void backend::collect_tids(std::vector<pid_t>& out) const
{
    out.clear();
    out.reserve(m_timers.size());
    for(const auto& [tid, st] : m_timers)
    {
        out.push_back(tid);
    }
}

}  // namespace cpu_snap::sampling::triggers::signal
