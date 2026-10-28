#pragma once

/**
 * @file perf_ring.hpp
 * @brief RAII wrapper over a perf_event_open mmap ring + decoder-templated
 *        drain loop.
 *
 * Owns the kernel-side ring resources (mmap base + bytes, optional epoll
 * de-registration fd) and exposes a single template `drain(Decoder&&)`
 * that walks every record since the last drain, splices ring-wrapped
 * records into an internal scratch buffer, and forwards each
 * @c perf_event_header* + body pointer to the caller's decoder.
 *
 * Used by cpu_snap::sampling::internal::kernel_drain, which supplies a
 * decoder that walks PERF_SAMPLE callchains; the ring management itself
 * is decoder-agnostic.
 *
 * Header-only. C++20.
 */

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include <linux/perf_event.h>
#include <sys/epoll.h>
#include <sys/mman.h>
#include <unistd.h>

namespace cpu_snap::sampling
{

/**
 * Per-ring configuration handed to the perf_ring constructor.
 *
 * Ownership: perf_ring takes over @c fd, @c mmap_base, and the epoll
 * registration. The dtor closes the fd, unmaps the ring, and (when
 * @c epoll_fd_for_dereg is non-negative) calls EPOLL_CTL_DEL.
 */
struct perf_ring_config
{
    /** perf_event_open fd (must outlive caller's reference). */
    int fd = -1;
    /** mmap base address returned by mmap(perf_event_open(fd)). */
    void* mmap_base = nullptr;
    /** Total mmap region bytes; includes the leading metadata page. */
    std::size_t mmap_bytes = 0;
    /** Epoll fd to deregister from on dtor; -1 if not registered. */
    int epoll_fd_for_dereg = -1;
};

/**
 * RAII per-tid perf_event_open ring.
 *
 * Construct with a fully-populated perf_ring_config (caller owns the
 * perf_event_open + mmap + epoll_ctl ADD sequence); the dtor reverses
 * those steps in the correct order. Move-only, never copy.
 *
 * @note Does NOT call PERF_EVENT_IOC_ENABLE / DISABLE; the caller is
 *       responsible for arming the perf fd. perf_ring is a memory +
 *       file-descriptor lifetime owner, not a controller.
 */
class perf_ring
{
public:
    explicit perf_ring(perf_ring_config cfg) noexcept
    : m_cfg{ cfg }
    {
        if(m_cfg.mmap_base != nullptr)
        {
            m_meta = reinterpret_cast<perf_event_mmap_page*>(m_cfg.mmap_base);
            // NOLINTNEXTLINE(misc-include-cleaner): sysconf from <unistd.h>
            const auto page_sz = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
            std::size_t offset = page_sz;
            std::size_t size   = m_cfg.mmap_bytes >= page_sz ? m_cfg.mmap_bytes - page_sz : 0;
            if(m_meta->data_size != 0)
            {
                offset = static_cast<std::size_t>(m_meta->data_offset);
                size   = static_cast<std::size_t>(m_meta->data_size);
            }
            const bool fits = offset <= m_cfg.mmap_bytes && size <= m_cfg.mmap_bytes - offset;
            if(fits && size != 0 && (size & (size - 1)) == 0)
            {
                m_data_ring  = static_cast<char*>(m_cfg.mmap_base) + offset;
                m_data_bytes = size;
            }
        }
    }

    ~perf_ring() { release(); }

    perf_ring(const perf_ring&)            = delete;
    perf_ring& operator=(const perf_ring&) = delete;

    perf_ring(perf_ring&& other) noexcept
    : m_cfg{ other.m_cfg }
    , m_meta{ other.m_meta }
    , m_data_ring{ other.m_data_ring }
    , m_data_bytes{ other.m_data_bytes }
    , m_scratch{ std::move(other.m_scratch) }
    {
        other.m_cfg        = {};
        other.m_meta       = nullptr;
        other.m_data_ring  = nullptr;
        other.m_data_bytes = 0;
    }

    /**
     * Walk every record since the last drain.
     *
     * The decoder is invoked once per record with the record's
     * @c perf_event_header* (already spliced into a contiguous buffer
     * when the record wrapped the ring) and a pointer to the start of
     * the same record (header + body) in linear memory. Total record
     * size is @c hdr->size.
     *
     * @tparam Decoder Callable with signature
     *   `void(const perf_event_header* hdr, const std::uint8_t* record)`
     *   where @c record points to a contiguous @c hdr->size bytes:
     *   the header is at @c record[0..sizeof(perf_event_header)) and
     *   the body follows immediately.
     *
     * @return Number of records drained on this call.
     */
    template <typename Decoder>
    std::size_t drain(Decoder&& decoder)
    {
        if(m_meta == nullptr || m_data_bytes == 0)
        {
            return 0;
        }
        const std::uint64_t head = __atomic_load_n(&m_meta->data_head, __ATOMIC_ACQUIRE);
        std::uint64_t       tail = m_meta->data_tail;
        const std::uint64_t mask = m_data_bytes - 1;
        std::size_t         count = 0;

        // Publishes the consumed position even when the decoder throws, so a
        // failed record is not replayed on the next drain.
        struct tail_publisher
        {
            perf_event_mmap_page*      meta;
            const std::uint64_t&       tail;
            ~tail_publisher() { __atomic_store_n(&meta->data_tail, tail, __ATOMIC_RELEASE); }
        } publisher{ m_meta, tail };

        while(tail < head)
        {
            const std::uint64_t pos = tail & mask;
            perf_event_header   local_hdr{};
            std::memcpy(&local_hdr, m_data_ring + pos, sizeof(local_hdr));

            const std::uint64_t size = local_hdr.size;
            if(size < sizeof(local_hdr) || size > m_data_bytes || size > head - tail)
            {
                tail = head;
                break;
            }

            const auto* record = reinterpret_cast<const std::uint8_t*>(m_data_ring + pos);
            if(pos + size > m_data_bytes)
            {
                if(size > m_scratch.size())
                {
                    m_scratch.resize(size);
                }
                const std::uint64_t first = m_data_bytes - pos;
                std::memcpy(m_scratch.data(), m_data_ring + pos, first);
                std::memcpy(m_scratch.data() + first, m_data_ring, size - first);
                record = m_scratch.data();
            }

            tail += size;
            ++count;
            decoder(reinterpret_cast<const perf_event_header*>(record), record);
        }
        return count;
    }

    [[nodiscard]] int         fd() const noexcept { return m_cfg.fd; }
    [[nodiscard]] void*       mmap_base() const noexcept { return m_cfg.mmap_base; }
    [[nodiscard]] std::size_t data_bytes() const noexcept { return m_data_bytes; }
    [[nodiscard]] bool        valid() const noexcept { return m_meta != nullptr; }

    /**
     * Skip the epoll DEL on destruction. Closing the epoll fd drops the
     * registrations anyway, and after fork() a DEL would remove the
     * parent's entries from the shared epoll instance.
     */
    void forget_epoll_registration() noexcept { m_cfg.epoll_fd_for_dereg = -1; }

private:
    void release() noexcept
    {
        if(m_cfg.epoll_fd_for_dereg >= 0 && m_cfg.fd >= 0)
        {
            (void) ::epoll_ctl(m_cfg.epoll_fd_for_dereg, EPOLL_CTL_DEL, m_cfg.fd,
                               nullptr);
        }
        if(m_cfg.mmap_base != nullptr && m_cfg.mmap_bytes > 0)
        {
            (void) ::munmap(m_cfg.mmap_base, m_cfg.mmap_bytes);
        }
        if(m_cfg.fd >= 0)
        {
            (void) ::close(m_cfg.fd);
        }
        m_cfg        = {};
        m_meta       = nullptr;
        m_data_ring  = nullptr;
        m_data_bytes = 0;
    }

    perf_ring_config          m_cfg{};
    perf_event_mmap_page*     m_meta       = nullptr;
    char*                     m_data_ring  = nullptr;
    std::size_t               m_data_bytes = 0;
    std::vector<std::uint8_t> m_scratch;
};

}  // namespace cpu_snap::sampling
