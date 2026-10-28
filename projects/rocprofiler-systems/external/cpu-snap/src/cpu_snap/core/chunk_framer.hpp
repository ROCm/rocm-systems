#pragma once

/**
 * @file chunk_framer.hpp
 * @brief Serializes record_variant values into 1-MiB-target chunks per
 *        the cpu_snap v1 wire format and hands each completed chunk
 *        to a host-supplied sink_fn.
 *
 * The first thing emitted on start() is the 64-byte file_header;
 * subsequent emissions are chunks of [chunk_header | records body]
 * (no footer, no checksum).
 *
 * Layout invariants:
 *   * Records never span a chunk: if appending the next record would
 *     overflow chunk_bytes, the current chunk is sealed and a new one
 *     started first.
 *   * Inter-record alignment is 4 bytes; records_byte_count in the
 *     chunk header equals the unpadded body length, NOT including the
 *     trailing alignment pad of the last record.
 *
 * Threading: not reentrant. Sink callbacks are invoked synchronously
 * from inside write() / flush() / start(). Exceptions thrown by the
 * sink are caught and counted in sink_errors(); they do NOT propagate
 * to the caller, keeping the collector worker alive.
 *
 * Header-only. C++20. Little-endian native only.
 */

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

#include "cpu_snap/core/records.hpp"
#include "cpu_snap/core/types.hpp"
#include "cpu_snap/error.hpp"

namespace cpu_snap::core
{

static_assert(std::endian::native == std::endian::little,
              "sampling wire format is little-endian; this build assumes a "
              "little-endian host");

// NOLINTBEGIN(readability-identifier-naming): codebase k_*-prefix convention
// k_default_chunk_bytes lives in core/types.hpp so that public headers
// (sampling) can default their configs to it without pulling this
// header's heavy transitive include set.

/** @brief Tightest body the framer can accommodate (must hold one record header). */
inline constexpr std::uint32_t k_min_chunk_bytes = sizeof(chunk_header) + 4;

/** @brief u16 record body_size limit. */
inline constexpr std::uint32_t k_max_record_body_bytes = 0xFFFFU;
// NOLINTEND(readability-identifier-naming)

/**
 * @brief Serializes records into framed chunks and emits each completed
 *        chunk via a host-supplied sink callback.
 *
 * See file header for invariants.
 *
 * @note The chunk buffer is held inline as an alignas(64) std::array of
 *       k_default_chunk_bytes (1 MiB), so a chunk_framer instance is
 *       roughly 1 MiB in size. The class is one-per-streaming_pipeline
 *       (i.e. one-per-sampler), so total footprint at the design
 *       scale of one sampler per process is negligible. Avoid creating
 *       chunk_framer objects on a small thread stack; prefer heap or
 *       data-segment allocation if you cannot rely on a generous main
 *       stack (e.g. inside a test with a thread-stack-size limit).
 */
class chunk_framer
{
public:
    /** @brief Construction parameters captured into the file_header at start(). */
    struct config
    {
        /** Soft flush threshold in bytes; defaults to k_default_chunk_bytes (1 MiB). */
        std::uint32_t chunk_bytes   = k_default_chunk_bytes;
        /** Bitfield copied into file_header.flags (see core::with_stream_clock for the clock bits). */
        std::uint16_t flags         = 0;
        /** Process id stamped into file_header (0 = let framer query getpid()). */
        std::uint64_t pid           = 0;
        /** Unix-epoch nanoseconds for the start-of-capture instant (0 = framer fills it). */
        std::uint64_t start_unix_ns = 0;
        /** Process label written into the file_header (empty = /proc/self/comm). */
        std::string   process_name;
    };

    /**
     * @brief Construct a framer that will emit through @p sink.
     * @param cfg  Header values + chunk-size target. cfg.chunk_bytes
     *             acts as a soft flush threshold and must be in the
     *             closed range [k_min_chunk_bytes, k_default_chunk_bytes];
     *             the underlying buffer is always sized at
     *             k_default_chunk_bytes (1 MiB).
     * @param sink Host callback invoked once per emitted file-header
     *             and once per completed chunk. Must not be null.
     * @throws cpu_snap::config_error if sink is null, chunk_bytes is
     *         below k_min_chunk_bytes, or chunk_bytes exceeds
     *         k_default_chunk_bytes.
     */
    chunk_framer(config cfg, sink_fn sink)
    : m_cfg(std::move(cfg))
    , m_sink(std::move(sink))
    {
        if(!m_sink)
        {
            throw config_error("chunk_framer: sink_fn must not be null");
        }
        if(m_cfg.chunk_bytes < k_min_chunk_bytes)
        {
            throw config_error(
                "chunk_framer: chunk_bytes too small for header + footer");
        }
        if(m_cfg.chunk_bytes > k_default_chunk_bytes)
        {
            throw config_error(
                "chunk_framer: chunk_bytes exceeds inline buffer capacity");
        }
    }

    ~chunk_framer() = default;

    chunk_framer(const chunk_framer&)            = delete;
    chunk_framer& operator=(const chunk_framer&) = delete;
    chunk_framer(chunk_framer&&)                 = delete;
    chunk_framer& operator=(chunk_framer&&)      = delete;

    /**
     * @brief Emit the file_header.
     *
     * Must be called exactly once before write().
     * @throws cpu_snap::usage_error if called more than once.
     */
    void start()
    {
        if(m_started)
        {
            throw usage_error("chunk_framer::start called twice");
        }
        m_started = true;

        file_header hdr{};
        hdr.magic         = { 'C', 'S', 'N', 'P' };
        hdr.version_major = k_record_format_version_major;
        hdr.version_minor = k_record_format_version_minor;
        hdr.start_unix_ns = m_cfg.start_unix_ns;
        hdr.pid           = m_cfg.pid;
        hdr.chunk_bytes   = m_cfg.chunk_bytes;
        hdr.flags         = m_cfg.flags;
        copy_process_name(hdr.process_name, m_cfg.process_name);

        std::array<std::byte, sizeof(file_header)> bytes{};
        std::memcpy(bytes.data(), &hdr, sizeof(hdr));
        emit_to_sink(std::span<const std::byte>(bytes.data(), bytes.size()));

        open_chunk();
    }

    /**
     * @brief Append one record.
     *
     * Flushes the current chunk first if the encoded record would not fit.
     * @param rec Record to append.
     * @throws cpu_snap::usage_error if called before start() or after a
     *         second start().
     * @throws cpu_snap::resource_error if the encoded record body exceeds
     *         k_max_record_body_bytes (u16 limit) or, somehow, exceeds
     *         the configured chunk capacity.
     */
    void write(const record_variant& rec)
    {
        if(!m_started)
        {
            throw usage_error("chunk_framer::write called before start()");
        }

        const std::uint32_t body_size = encoded_body_size(rec);
        if(body_size > k_max_record_body_bytes)
        {
            throw resource_error("chunk_framer: record body exceeds u16 limit");
        }

        const std::uint32_t record_total = align4(sizeof(record_header_wire) + body_size);
        if(static_cast<std::uint32_t>(m_chunk_pos) + record_total > m_cfg.chunk_bytes)
        {
            seal_and_emit_chunk();
            open_chunk();
            if(sizeof(chunk_header) + record_total > m_cfg.chunk_bytes)
            {
                throw resource_error(
                    "chunk_framer: single record larger than chunk capacity");
            }
        }

        write_record(rec, body_size);
    }

    /**
     * @brief Append a checkpoint, split across as many CHECKPOINT records
     *        as needed to respect the u16 body limit.
     *
     * Every record carries the same wall_ns and a disjoint slice of the
     * entries. An empty checkpoint is still written as one record.
     */
    void write_split(const checkpoint_record& rec)
    {
        constexpr std::size_t per_record =
            (k_max_record_body_bytes - sizeof(std::uint64_t) - 2 * sizeof(std::uint32_t)) /
            sizeof(per_tid_checkpoint_entry);

        checkpoint_record part;
        part.wall_ns = rec.wall_ns;
        std::size_t pos = 0;
        do
        {
            const std::size_t count = std::min(per_record, rec.entries.size() - pos);
            part.entries.assign(rec.entries.begin() + static_cast<std::ptrdiff_t>(pos),
                                rec.entries.begin() + static_cast<std::ptrdiff_t>(pos + count));
            write(part);
            pos += count;
        } while(pos < rec.entries.size());
    }

    /**
     * @brief Append a module snapshot, split across as many MAPS_INIT
     *        records as needed to respect the u16 body limit.
     *
     * Readers apply each MAPS_INIT additively, so the split is lossless.
     * An empty snapshot is still written as one record.
     */
    void write_split(const maps_init_record& rec)
    {
        constexpr std::uint32_t fixed = sizeof(std::uint64_t) + 2 * sizeof(std::uint32_t);

        maps_init_record part;
        part.wall_ns       = rec.wall_ns;
        std::uint32_t size = fixed;
        for(const auto& mod : rec.modules)
        {
            const std::uint32_t mod_size = module_size_on_wire(mod);
            if(!part.modules.empty() && size + mod_size > k_max_record_body_bytes)
            {
                write(part);
                part.modules.clear();
                size = fixed;
            }
            part.modules.push_back(mod);
            size += mod_size;
        }
        write(part);
    }

    /** @brief Force-emit any partial chunk. Safe to call multiple times. */
    void flush()
    {
        if(!m_started)
        {
            return;
        }
        if(m_chunk_pos > sizeof(chunk_header))
        {
            seal_and_emit_chunk();
            open_chunk();
        }
    }

    /** @brief Number of exceptions caught and swallowed from sink invocations. */
    [[nodiscard]] std::uint64_t sink_errors() const noexcept { return m_sink_errors; }

private:
    // Wire-form record header struct used purely for sizeof; emission
    // is done field-by-field through append_*.
    struct record_header_wire
    {
        std::uint8_t  type;
        std::uint8_t  reserved;
        std::uint16_t body_size;
    };
    static_assert(sizeof(record_header_wire) == 4,
                  "record header must be 4 bytes on the wire");

    static std::uint32_t align4(std::uint32_t value) noexcept
    {
        return (value + 3U) & ~3U;
    }

    static void copy_process_name(std::array<char, 32>& dst,
                                  const std::string&    src) noexcept
    {
        const std::size_t copy_bytes = std::min(src.size(), dst.size());
        if(copy_bytes > 0)
        {
            std::memcpy(dst.data(), src.data(), copy_bytes);
        }
        if(copy_bytes < dst.size())
        {
            std::memset(dst.data() + copy_bytes, 0, dst.size() - copy_bytes);
        }
    }

    void emit_to_sink(std::span<const std::byte> bytes) noexcept
    {
        try
        {
            m_sink(bytes);
        } catch(...)
        {
            ++m_sink_errors;
        }
    }

    void open_chunk() noexcept
    {
        // Reserve the chunk_header slot at offset 0; the bytes are
        // filled in by seal_and_emit_chunk() so record offsets are
        // body-relative.
        m_chunk_pos = sizeof(chunk_header);
    }

    void seal_and_emit_chunk()
    {
        const auto body_bytes =
            static_cast<std::uint32_t>(m_chunk_pos) - sizeof(chunk_header);

        chunk_header hdr{};
        hdr.magic              = { 'C', 'H' };
        hdr.version            = 0;
        hdr.records_byte_count = body_bytes;
        hdr.chunk_seq          = m_chunk_seq++;
        std::memcpy(m_chunk_buf.data(), &hdr, sizeof(hdr));

        emit_to_sink(std::span<const std::byte>(m_chunk_buf.data(), m_chunk_pos));
    }

    // ---- per-record encoding ----------------------------------------------

    void write_record(const record_variant& rec, std::uint32_t body_size)
    {
        // Caller (write()) guarantees the padded record fits within the
        // soft cap, which is itself <= the inline buffer capacity.
        record_header_wire rh{};
        rh.type      = static_cast<std::uint8_t>(type_of(rec));
        rh.reserved  = 0;
        rh.body_size = static_cast<std::uint16_t>(body_size);
        std::memcpy(m_chunk_buf.data() + m_chunk_pos, &rh, sizeof(rh));
        m_chunk_pos += sizeof(record_header_wire);

        std::byte* body = m_chunk_buf.data() + m_chunk_pos;
        std::visit([&](const auto& body_rec) { encode_body(body, body_rec); }, rec);
        m_chunk_pos += body_size;

        const std::uint32_t padded =
            align4(static_cast<std::uint32_t>(sizeof(record_header_wire) + body_size));
        const std::size_t pad_bytes = padded - (sizeof(record_header_wire) + body_size);
        if(pad_bytes > 0)
        {
            std::memset(m_chunk_buf.data() + m_chunk_pos, 0, pad_bytes);
            m_chunk_pos += pad_bytes;
        }
    }

    // std::visit can theoretically throw bad_variant_access; record_variant is
    // constructed only by the framer and never reaches a valueless_by_exception
    // state on the hot path, so the noexcept declaration is sound.
    // NOLINTNEXTLINE(bugprone-exception-escape)
    static std::uint32_t encoded_body_size(const record_variant& rec) noexcept
    {
        return std::visit(
            [](const auto& body_rec) -> std::uint32_t { return body_size_of(body_rec); },
            rec);
    }

    // ---- body sizes --------------------------------------------------------

    static std::uint32_t body_size_of(const new_node_record& /*unused*/) noexcept
    {
        return 16;
    }
    static std::uint32_t body_size_of(const sample_record& /*unused*/) noexcept
    {
        return 16;
    }
    static std::uint32_t body_size_of(const lost_record& /*unused*/) noexcept
    {
        return 16;
    }
    static std::uint32_t body_size_of(const eof_record& /*unused*/) noexcept
    {
        return 16;
    }

    static std::uint32_t body_size_of(const checkpoint_record& rec) noexcept
    {
        return static_cast<std::uint32_t>(
            sizeof(std::uint64_t) + 2 * sizeof(std::uint32_t) +
            rec.entries.size() * sizeof(per_tid_checkpoint_entry));
    }

    static std::uint32_t module_size_on_wire(const module_info& mod) noexcept
    {
        const std::uint32_t fixed = 3 * sizeof(std::uint64_t) + 2 * sizeof(std::uint32_t);
        const auto          var =
            static_cast<std::uint32_t>(mod.build_id.size() + mod.path.size());
        return align4(fixed + var);
    }

    static std::uint32_t body_size_of(const maps_init_record& rec) noexcept
    {
        std::uint32_t total = sizeof(std::uint64_t) + 2 * sizeof(std::uint32_t);
        for(const auto& mod : rec.modules)
        {
            total += module_size_on_wire(mod);
        }
        return total;
    }

    static std::uint32_t body_size_of(const maps_delta_record& rec) noexcept
    {
        const std::uint32_t prefix =
            sizeof(std::uint64_t) + 4 + sizeof(std::uint32_t);  // wall_ns, op+pad3, _u32
        return prefix + module_size_on_wire(rec.module);
    }

    // ---- body encoders (little-endian; cursor advances per field) ----------

    template <typename T>
    static void put(std::byte*& cur, T value) noexcept
    {
        static_assert(std::is_trivially_copyable_v<T>);
        std::memcpy(cur, &value, sizeof(T));
        cur += sizeof(T);
    }

    static void put_bytes(std::byte*& cur, const void* src, std::size_t n) noexcept
    {
        if(n > 0)
        {
            std::memcpy(cur, src, n);
            cur += n;
        }
    }

    static void encode_body(std::byte* body, const new_node_record& rec) noexcept
    {
        put(body, rec.node_id);
        put(body, rec.parent_id);
        put(body, rec.ip);
    }

    static void encode_body(std::byte* body, const sample_record& rec) noexcept
    {
        put(body, rec.timestamp_ns);
        put(body, rec.tid);
        put(body, rec.leaf_node_id);
    }

    static void encode_body(std::byte* body, const lost_record& rec) noexcept
    {
        put(body, rec.wall_ns);
        put(body, rec.tid);
        put(body, rec.lost_count_delta);
    }

    static void encode_body(std::byte* body, const eof_record& rec) noexcept
    {
        put(body, rec.wall_ns);
        put(body, rec.total_samples_emitted);
    }

    static void encode_body(std::byte* body, const checkpoint_record& rec) noexcept
    {
        put(body, rec.wall_ns);
        put(body, static_cast<std::uint32_t>(rec.entries.size()));
        put(body, static_cast<std::uint32_t>(0));
        for(const auto& entry : rec.entries)
        {
            put(body, entry.tid);
            put(body, entry.leaf_id);
            put(body, entry.samples);
            put(body, entry.lost);
        }
    }

    static void encode_module(std::byte*& body, const module_info& mod) noexcept
    {
        std::byte* start = body;
        put(body, mod.base_va);
        put(body, mod.size);
        put(body, mod.file_offset);
        put(body, static_cast<std::uint32_t>(mod.build_id.size()));
        put(body, static_cast<std::uint32_t>(mod.path.size()));
        put_bytes(body, mod.build_id.data(), mod.build_id.size());
        put_bytes(body, mod.path.data(), mod.path.size());
        const auto          written = static_cast<std::uint32_t>(body - start);
        const std::uint32_t padded  = align4(written);
        for(std::uint32_t i = written; i < padded; ++i)
        {
            put<std::uint8_t>(body, 0);
        }
    }

    static void encode_body(std::byte* body, const maps_init_record& rec) noexcept
    {
        put(body, rec.wall_ns);
        put(body, static_cast<std::uint32_t>(rec.modules.size()));
        put(body, static_cast<std::uint32_t>(0));
        for(const auto& mod : rec.modules)
        {
            encode_module(body, mod);
        }
    }

    static void encode_body(std::byte* body, const maps_delta_record& rec) noexcept
    {
        put(body, rec.wall_ns);
        put(body, static_cast<std::uint8_t>(rec.operation));
        put(body, static_cast<std::uint8_t>(0));
        put(body, static_cast<std::uint8_t>(0));
        put(body, static_cast<std::uint8_t>(0));
        put(body, static_cast<std::uint32_t>(0));
        encode_module(body, rec.module);
    }

    // ---- state -------------------------------------------------------------

    config        m_cfg;
    sink_fn       m_sink;
    std::uint64_t m_chunk_seq       = 0;
    std::uint64_t m_sink_errors     = 0;
    std::size_t   m_chunk_pos       = 0;
    bool          m_started         = false;
    /**
     * Inline 1 MiB chunk buffer, aligned to a cache line. Sized to hold
     * one full chunk (chunk_header + records body). The
     * cursor m_chunk_pos tracks the current write offset; it resets to
     * sizeof(chunk_header) after each sink flush via open_chunk().
     */
    alignas(64) std::array<std::byte, k_default_chunk_bytes> m_chunk_buf{};
};

}  // namespace cpu_snap::core
