#pragma once

/**
 * @file memory_sink.hpp
 * @brief Bounded in-memory sink_fn implementation.
 *
 * Policy is stop-when-full: once a chunk would push the accumulated
 * bytes past the configured limit, that chunk and every later chunk are
 * dropped (and counted). This is deliberately NOT a ring buffer: an
 * early chunk can carry NEW_NODE trie records a later SAMPLE depends
 * on, so evicting old bytes would silently corrupt decode of later
 * samples. Dropping is always whole-chunk, so the retained bytes stay
 * a contiguous, validly decodable wire-format prefix.
 *
 * Header-only. C++20.
 */

#include <cstddef>
#include <span>
#include <vector>

namespace cpu_snap::core
{

/**
 * @brief Bounded, sink_fn-compatible in-memory byte accumulator.
 *
 * @note Not thread-safe. A sink_fn is only ever invoked from the
 *       collector's single worker thread, so no
 *       internal locking is needed.
 */
class memory_sink
{
public:
    /** @brief Construction parameters. */
    struct config
    {
        /** Maximum number of bytes to retain before dropping later chunks. */
        std::size_t max_bytes;
    };

    explicit memory_sink(config cfg) noexcept
    : m_cfg(cfg)
    {
    }

    /** @brief sink_fn entry point: append @p chunk, or drop it once full. */
    void operator()(std::span<const std::byte> chunk)
    {
        if(!m_full && m_bytes.size() + chunk.size() <= m_cfg.max_bytes)
        {
            m_bytes.insert(m_bytes.end(), chunk.begin(), chunk.end());
            return;
        }
        m_full = true;
        ++m_dropped_chunks;
        m_dropped_bytes += chunk.size();
    }

    /** @brief Bytes retained so far (a contiguous wire-format prefix). */
    [[nodiscard]] std::span<const std::byte> bytes() const noexcept
    {
        return { m_bytes.data(), m_bytes.size() };
    }

    /** @brief Whether the byte limit has been reached (chunks are being dropped). */
    [[nodiscard]] bool full() const noexcept { return m_full; }

    /** @brief Number of chunks dropped since the limit was reached. */
    [[nodiscard]] std::size_t dropped_chunks() const noexcept { return m_dropped_chunks; }

    /** @brief Total bytes dropped since the limit was reached. */
    [[nodiscard]] std::size_t dropped_bytes() const noexcept { return m_dropped_bytes; }

private:
    config                  m_cfg;
    std::vector<std::byte>  m_bytes;
    bool                    m_full           = false;
    std::size_t             m_dropped_chunks = 0;
    std::size_t             m_dropped_bytes  = 0;
};

}  // namespace cpu_snap::core
