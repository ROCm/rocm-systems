#pragma once

/**
 * @file kernel_drain.hpp
 * @brief Per-tid perf_event_open + mmap-ring lifecycle and decode loop.
 *
 * Owns the kernel-facing state of cpu_snap::sampling::sampler: one fd
 * + one mmap ring per attached tid. PERF_RECORD_SAMPLE decoding is
 * delegated to perf_sample_decoder; PERF_RECORD_LOST is handled here.
 *
 * kernel_drain has NO worker thread, NO fork handling, and NO storage
 * for decoded samples. The owning sampler drives drain_all() from
 * its worker loop and supplies sinks that decide what to do with each
 * decoded sample / lost event.
 *
 * Lives in sampling/ as a sampling-internal header (consumed only
 * by sampler.hpp + tests).
 *
 * Header-only. C++20.
 */

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <span>
#include <string>
#include <sys/types.h>
#include <unordered_map>
#include <vector>

#include <linux/perf_event.h>
#include <sys/epoll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "cpu_snap/error.hpp"
#include "cpu_snap/sampling/clock_kind.hpp"
#include "cpu_snap/sampling/time.hpp"
#include "cpu_snap/sampling/unwind_mode.hpp"
#include "cpu_snap/sampling/perf_ring.hpp"
#include "cpu_snap/sampling/triggers/perf/open_error.hpp"
#include "cpu_snap/sampling/triggers/perf/perf_sample_decoder.hpp"
#include "cpu_snap/sampling/triggers/perf/ring_sizing.hpp"
#include "cpu_snap/sampling/unwind/perf_regs.hpp"

namespace cpu_snap::sampling::triggers::perf
{

/**
 * @brief Per-tid kernel-ring manager.
 *
 * Opens one perf_event_open fd + mmap ring per attached tid, decodes
 * PERF_RECORD_SAMPLE records, and dispatches each decoded sample /
 * lost-record event to caller-supplied sinks.
 *
 * @note The owning sampler is responsible for invoking drain_all
 *       from its worker thread; kernel_drain itself never spawns
 *       threads.
 * @note Frames per decoded sample are capped by config::callchain_depth.
 */
class kernel_drain
{
public:
    /**
     * @brief Configuration for one kernel_drain instance.
     *
     * All fields are copies of the equivalent sampler::config fields;
     * sampler is responsible for validating them before construction.
     */
    struct config
    {
        /** Per-thread sample frequency in Hz (kernel auto-tunes period). */
        std::uint32_t sample_freq_hz;
        /** Max user-space frames per PERF_SAMPLE_CALLCHAIN record. */
        std::uint32_t callchain_depth;
        /** Mmap-ring pages per attached tid; must be a power of two. */
        std::uint32_t kernel_pages_per_tid;
        /** When true, the kernel suppresses kernel-mode stack frames. */
        bool exclude_kernel;
        /** When true, the kernel suppresses hypervisor-mode frames. */
        bool exclude_hv;
        /** PERF_SAMPLE_STACK_USER copy size (DWARF mode only); ignored otherwise. */
        std::uint32_t dwarf_stack_bytes;
        /** Selected unwind strategy (frame pointer or DWARF). */
        unwind_mode unwind;
        /** Clock of PERF_SAMPLE_TIME (perf attr.clockid). */
        clock_kind clock = clock_kind::monotonic;
    };

    /**
     * @brief Construct the drain and create the shared epoll fd used
     *        by wait().
     * @param cfg Configuration (already validated by sampler).
     * @throws cpu_snap::environment_error If epoll_create1 fails.
     */
    explicit kernel_drain(config cfg)
    : m_cfg{ cfg }
    , m_scratch{ cfg.callchain_depth }
    , m_epoll_fd{ create_epoll_fd() }
    {}

    ~kernel_drain() { close_all(); }

    kernel_drain(const kernel_drain&)            = delete;
    kernel_drain& operator=(const kernel_drain&) = delete;
    kernel_drain(kernel_drain&&)                 = delete;
    kernel_drain& operator=(kernel_drain&&)      = delete;

    /**
     * @brief Open one fd + mmap ring for @p tid and arm sampling.
     *
     * No-op if the tid is already attached or if perf_event_open
     * returns ESRCH (thread exited between listing and open).
     * @param tid Linux thread id to attach.
     * @throws cpu_snap::environment_error on perf_event_open / mmap / ioctl
     *         / epoll_ctl failure other than ESRCH (code() is the errno).
     */
    void open_tid(pid_t tid)
    {
        if(m_rings.contains(tid))
        {
            return;
        }
        if(m_epoll_fd < 0)
        {
            m_epoll_fd = create_epoll_fd();
        }
        cpu_snap::sampling::perf_ring_config ring_cfg{};
        try
        {
            ring_cfg = open_ring_for(tid);
        } catch(const environment_error& e)
        {
            if(e.code() == std::error_code(ESRCH, std::generic_category()))
            {
                return;
            }
            throw;
        }

        epoll_event event{};
        event.events  = EPOLLIN;
        event.data.fd = ring_cfg.fd;
        if(::epoll_ctl(m_epoll_fd, EPOLL_CTL_ADD, ring_cfg.fd, &event) < 0)
        {
            const int err = errno;
            // Build a perf_ring solely so its dtor releases mmap + fd.
            cpu_snap::sampling::perf_ring scoped_release{ ring_cfg };
            (void) scoped_release;
            throw environment_error::from_errno(err, "epoll_ctl(ADD)");
        }
        ring_cfg.epoll_fd_for_dereg = m_epoll_fd;
        m_rings.emplace(tid, cpu_snap::sampling::perf_ring{ ring_cfg });
    }

    /**
     * @brief Close the ring for @p tid (if attached).
     * @param tid Linux thread id to detach.
     */
    void close_tid(pid_t tid) noexcept
    {
        auto it = m_rings.find(tid);
        if(it == m_rings.end()) return;
        m_rings.erase(it);
    }

    /**
     * @brief Drain every ring once, invoking @p on_sample per decoded
     *        sample and @p on_lost per PERF_RECORD_LOST event.
     *
     * Never throws.
     *
     * @tparam Unw        Concrete unwinder alternative (see unwind::Unwinder),
     *                    instantiated once per call from the caller's
     *                    std::visit over the sampler-owned unwinder variant.
     * @tparam SampleSink Callable with signature
     *                    `void(pid_t tid, std::uint64_t ts_ns,
     *                          std::span<const std::uint64_t> frames)`.
     * @tparam LostSink   Callable with signature
     *                    `void(std::uint64_t lost_count)`.
     * @param unwinder  Concrete unwinder for this batch.
     * @param on_sample Per-sample sink.
     * @param on_lost   Per-LOST-record sink.
     *
     * @note @p unwinder and the sinks are taken by reference and threaded
     *       into the per-ring decode loop as concrete types. The compiler
     *       inlines through them, eliminating the std::function thunk
     *       that previously dominated the recorder hot path.
     */
    template <unwind::Unwinder Unw, typename SampleSink, typename LostSink>
    void drain_all(Unw& unwinder, const SampleSink& on_sample,
                   const LostSink& on_lost) noexcept
    {
        for(auto& [tid, ring] : m_rings)
        {
            drain_ring(ring, unwinder, on_sample, on_lost);
        }
    }

    /** @brief Disable sample firing on every ring (best-effort). */
    void disable_all() noexcept
    {
        m_enabled = false;
        for(auto& [tid, ring] : m_rings)
        {
            if(ring.fd() >= 0)
            {
                (void) ::ioctl(ring.fd(), PERF_EVENT_IOC_DISABLE, 0);
            }
        }
    }

    /** @brief Re-enable sample firing on every ring (best-effort). */
    void enable_all() noexcept
    {
        m_enabled = true;
        for(auto& [tid, ring] : m_rings)
        {
            if(ring.fd() >= 0)
            {
                (void) ::ioctl(ring.fd(), PERF_EVENT_IOC_ENABLE, 0);
            }
        }
    }

    /**
     * @brief Close every ring without draining.
     *
     * Safe to call from atfork-child.
     */
    void close_all() noexcept
    {
        // After fork the epoll instance is shared with the parent, so a per-ring
        // EPOLL_CTL_DEL would drop the parent's registrations; closing our copy
        // of the epoll fd below is enough.
        for(auto& [tid, ring] : m_rings)
        {
            ring.forget_epoll_registration();
        }
        m_rings.clear();
        m_enabled = true;
        if(m_epoll_fd >= 0)
        {
            ::close(m_epoll_fd);
            m_epoll_fd = -1;
        }
    }

    /**
     * @brief Number of currently-attached tids.
     * @return Attached-ring count.
     */
    [[nodiscard]] std::size_t tracked_tids() const noexcept { return m_rings.size(); }

    /**
     * @brief Whether @p tid is currently attached.
     * @param tid Linux thread id to query.
     * @return True if a ring is open for @p tid.
     */
    [[nodiscard]] bool is_tracked(pid_t tid) const noexcept
    {
        return m_rings.contains(tid);
    }

    /**
     * @brief Invoke @p fn(tid) for every currently-attached tid.
     *
     * The callable runs while iterating an internal container; @p fn
     * MUST NOT call open_tid / close_tid / close_all on this drain.
     * Intended for snapshotting the live set into a caller-owned buffer
     * before mutating.
     *
     * @tparam Fn Callable with signature void(pid_t).
     * @param fn  Callback per tracked tid.
     */
    template <typename Fn>
    void for_each_tid(Fn&& fn) const
    {
        for(const auto& [tid, ring] : m_rings)
        {
            fn(tid);
        }
    }

    /**
     * @brief Block up to @p timeout_ms waiting for at least one attached
     *        ring to have data.
     *
     * Level-triggered: the caller does NOT have to drain to EAGAIN;
     * subsequent waits will fire again if data remains. A ring whose thread
     * has exited (EPOLLHUP) is reported once and then removed from the epoll
     * set; it stays attached until close_tid so its remaining records can
     * still be drained.
     *
     * @param timeout_ms Max time to block, milliseconds. 0 polls, -1 blocks forever.
     * @return Number of ready rings, 0 on timeout / EINTR, -1 on error.
     */
    [[nodiscard]] int wait(int timeout_ms) noexcept
    {
        if(m_epoll_fd < 0)
        {
            return 0;
        }
        const int ready =
            ::epoll_wait(m_epoll_fd, m_epoll_events.data(),
                         static_cast<int>(m_epoll_events.size()), timeout_ms);
        if(ready < 0 && errno == EINTR)
        {
            return 0;
        }
        for(int i = 0; i < ready; ++i)
        {
            if((m_epoll_events[static_cast<std::size_t>(i)].events & (EPOLLHUP | EPOLLERR)) != 0)
            {
                (void) ::epoll_ctl(m_epoll_fd, EPOLL_CTL_DEL,
                                   m_epoll_events[static_cast<std::size_t>(i)].data.fd,
                                   nullptr);
            }
        }
        return ready;
    }

    /** Build perf_event_attr for the requested unwind mode + sampling config. */
    static perf_event_attr make_perf_attr(const config& cfg) noexcept
    {
        struct perf_event_attr attr
        {};
        attr.size        = sizeof(attr);
        attr.type        = PERF_TYPE_SOFTWARE;
        attr.config      = PERF_COUNT_SW_CPU_CLOCK;
        attr.freq        = 1;
        attr.sample_freq = cfg.sample_freq_hz;
        attr.sample_type = PERF_SAMPLE_IP | PERF_SAMPLE_TID | PERF_SAMPLE_TIME;
        if(cfg.unwind == unwind_mode::dwarf)
        {
            attr.sample_type |= PERF_SAMPLE_REGS_USER | PERF_SAMPLE_STACK_USER;
            attr.sample_regs_user  = unwind::k_perf_dwarf_regs_mask_x86_64;
            attr.sample_stack_user = cfg.dwarf_stack_bytes;
        }
        else
        {
            attr.sample_type |= PERF_SAMPLE_CALLCHAIN;
        }
        attr.disabled       = 1;
        attr.use_clockid    = 1;
        attr.clockid        = static_cast<std::int32_t>(utils::clock_id_of(cfg.clock));
        attr.exclude_kernel = cfg.exclude_kernel ? 1 : 0;
        attr.exclude_hv     = cfg.exclude_hv ? 1 : 0;
        attr.wakeup_events  = wakeup_events_for(cfg.unwind, cfg.dwarf_stack_bytes,
                                                cfg.kernel_pages_per_tid, page_bytes());
        return attr;
    }

private:
    static std::size_t page_bytes() noexcept
    {
        return static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    }

    /** mmap (1 + pages) * PAGE_SIZE bytes onto @p fd. Throws environment_error on MAP_FAILED. */
    static std::pair<void*, std::size_t> mmap_ring_pages(int fd, std::uint32_t pages)
    {
        // NOLINTNEXTLINE(google-runtime-int): sysconf() returns long by POSIX
        const long        page_sz = ::sysconf(_SC_PAGESIZE);
        const std::size_t bytes   = static_cast<std::size_t>(1 + pages) * page_sz;
        void* addr = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if(addr == MAP_FAILED)
        {
            const int err = errno;
            ::close(fd);
            throw environment_error::from_errno(err, "mmap perf");
        }
        return { addr, bytes };
    }

    /** PERF_EVENT_IOC_RESET + ENABLE (unless paused). Throws on failure, unwinding mmap + fd. */
    void arm_perf_event(int fd, void* mmap_addr, std::size_t mmap_bytes) const
    {
        if(::ioctl(fd, PERF_EVENT_IOC_RESET, 0) < 0 ||
           (m_enabled && ::ioctl(fd, PERF_EVENT_IOC_ENABLE, 0) < 0))
        {
            const int err = errno;
            ::munmap(mmap_addr, mmap_bytes);
            ::close(fd);
            throw environment_error::from_errno(err, "ioctl enable");
        }
    }

    static const char* clock_name(clock_kind clock) noexcept
    {
        return clock == clock_kind::boottime ? "boottime" : "wallclock";
    }

    static int perf_event_open(perf_event_attr& attr, pid_t tid) noexcept
    {
        return static_cast<int>(::syscall(SYS_perf_event_open, &attr, tid, /*cpu=*/-1,
                                          /*group_fd=*/-1, PERF_FLAG_FD_CLOEXEC));
    }

    cpu_snap::sampling::perf_ring_config open_ring_for(pid_t tid) const
    {
        perf_event_attr attr = make_perf_attr(m_cfg);
        int             fd = perf_event_open(attr, tid);
        if(fd < 0 && errno == EINVAL && m_cfg.clock != clock_kind::monotonic)
        {
            throw environment_error(
                std::error_code(EINVAL, std::generic_category()),
                std::string{ "perf_event_open rejected the requested clock (" } +
                    clock_name(m_cfg.clock) +
                    "); the kernel cannot stamp samples with it, and falling back to "
                    "another clock would mix timestamp domains");
        }
        if(fd < 0 && errno == EINVAL && attr.use_clockid != 0)
        {
            attr.use_clockid = 0;
            attr.clockid     = 0;
            fd               = perf_event_open(attr, tid);
        }
        if(fd < 0)
        {
            const int err = errno;
            throw environment_error(
                std::error_code(err, std::generic_category()),
                describe_perf_open_failure(tid, err, read_perf_event_paranoid()));
        }

        auto [mmap_addr, mmap_bytes] = mmap_ring_pages(fd, m_cfg.kernel_pages_per_tid);
        arm_perf_event(fd, mmap_addr, mmap_bytes);

        return cpu_snap::sampling::perf_ring_config{
            .fd                 = fd,
            .mmap_base          = mmap_addr,
            .mmap_bytes         = mmap_bytes,
            .epoll_fd_for_dereg = -1,
        };
    }

    static int create_epoll_fd()
    {
        const int fd = ::epoll_create1(EPOLL_CLOEXEC);
        if(fd < 0)
        {
            throw environment_error::from_errno(errno, "epoll_create1");
        }
        return fd;
    }

    template <unwind::Unwinder Unw, typename SampleSink, typename LostSink>
    void drain_ring(cpu_snap::sampling::perf_ring& ring, Unw& unwinder,
                    const SampleSink& on_sample, const LostSink& on_lost) noexcept
    {
        ring.drain(
            [&](const perf_event_header* hdr, const std::uint8_t* /*record*/) noexcept {
                if(hdr->type == PERF_RECORD_SAMPLE)
                {
                    perf_sample_decoder::process_sample(hdr, m_scratch, unwinder, on_sample);
                }
                else if(hdr->type == PERF_RECORD_LOST)
                {
                    struct lost_rec
                    {
                        perf_event_header hdr;
                        std::uint64_t     id;
                        std::uint64_t     lost;
                    };
                    on_lost(reinterpret_cast<const lost_rec*>(hdr)->lost);
                }
            });
    }


    // Sized to the design ceiling (256 tracked tids); covers an entire
    // epoll_wait without iterating, so the worker thread sees no heap
    // traffic per drain.
    // NOLINTNEXTLINE(readability-identifier-naming): codebase k_*-prefix convention
    static constexpr std::size_t k_epoll_max_events = 256;

    config                                               m_cfg;
    perf_sample_decoder::decode_scratch                  m_scratch;
    std::unordered_map<pid_t, cpu_snap::sampling::perf_ring> m_rings;
    bool                                                 m_enabled  = true;
    int                                                  m_epoll_fd = -1;
    std::array<epoll_event, k_epoll_max_events>          m_epoll_events{};
};

}  // namespace cpu_snap::sampling::triggers::perf
