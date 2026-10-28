#pragma once

/**
 * @file reader.hpp
 * @brief Push-based decoder for the cpu_snap v1 wire byte stream.
 *
 * Consumers feed bytes in any granularity via feed(); the reader
 * buffers partial chunks internally and invokes the host event
 * callback for every successfully parsed record. Parse failures are
 * absorbed into counters (never thrown). Rejected chunk headers trigger
 * a forward scan for the next plausible header instead of aborting.
 * See docs/wire-format.md for the full resync/rejection contract.
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "cpu_snap/core/records.hpp"
#include "cpu_snap/core/types.hpp"

namespace cpu_snap::core
{

/** @brief Stream-health counters exposed by reader. */
struct reader_counters
{
    /** Total bytes accepted via @ref reader::feed. */
    std::uint64_t bytes_fed       = 0;
    /** Records that fired the event callback. */
    std::uint64_t records_emitted = 0;
    /** Per-record parse errors absorbed inside an otherwise valid chunk. */
    std::uint64_t parse_errors    = 0;
    /** Number of detected gaps in the chunk_seq sequence. */
    std::uint64_t gap_count       = 0;
    /** Highest chunk_seq observed so far. */
    std::uint64_t last_seen_seq   = 0;
    /** Records skipped because their type tag was unrecognized. */
    std::uint64_t unknown_records = 0;
    /** Chunk headers rejected for an out-of-budget records_byte_count. */
    std::uint64_t chunks_rejected = 0;
    /** Number of times the reader forward-scanned for the next plausible
     *  chunk header after rejecting one. */
    std::uint64_t resyncs         = 0;
    /** Total bytes discarded while resyncing after a rejected chunk header. */
    std::uint64_t bytes_skipped   = 0;
};

/**
 * @brief Symmetric counterpart of chunk_framer.
 *
 * feed(span) is the sole mutating entry point and never throws;
 * partial buffers carry across calls. On unknown record types the
 * reader bumps parse_errors and stops walking the current chunk. A
 * fatal magic mismatch on the very first 64 bytes sets an internal
 * fatal flag and silences future emissions.
 */
class reader
{
public:
    reader();
    ~reader();

    reader(const reader&)            = delete;
    reader& operator=(const reader&) = delete;
    reader(reader&&)                 = delete;
    reader& operator=(reader&&)      = delete;

    /**
     * @brief Install or replace the event callback.
     *
     * The record_variant reference passed to @p fn is valid only for
     * the duration of the call; copy out if the host needs it longer.
     * @param fn Callback to install, or a null function to silence
     *           emissions without affecting counters.
     */
    void on_event(event_fn fn);

    /**
     * @brief Feed the next slice of bytes.
     *
     * Buffers partial records / chunks internally and synchronously
     * invokes the event callback for every full record that becomes
     * available. Never throws.
     * @param data Span of bytes from the wire stream.
     */
    void feed(std::span<const std::byte> data) noexcept;

    /**
     * @brief Resolve a leaf node id back to its IP chain.
     *
     * Walks parent pointers in the trie view assembled from NEW_NODE
     * records seen so far.
     * @param leaf_id Leaf node identifier.
     * @return Innermost-first frames matching raw_sample order, or an
     *         empty vector if @p leaf_id is unknown (for example, it
     *         was declared in a chunk skipped after a resync).
     */
    [[nodiscard]] std::vector<std::uint64_t> resolve_stack(std::uint32_t leaf_id) const;

    /**
     * @brief The file_header captured from the stream prefix.
     * @return Parsed header. Zero-initialised (magic == {0,0,0,0})
     *         until the first 64 bytes have been parsed.
     */
    [[nodiscard]] const file_header& header() const noexcept;

    /**
     * @brief Snapshot of the stream-health counters.
     * @return Reference to the counter block, valid until the next feed() call.
     */
    [[nodiscard]] const reader_counters& counters() const noexcept;

    /**
     * @brief Whether the reader has hit a non-recoverable framing error.
     * @return True once a fatal error has been observed (currently:
     *         bad file-header magic, unsupported major version, or a
     *         chunk_bytes field outside the supported range). No
     *         further events will be emitted.
     */
    [[nodiscard]] bool is_fatal() const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> m_impl;
};

}  // namespace cpu_snap::core
