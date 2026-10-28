/**
 * @file reader.cpp
 * @brief Implementation of cpu_snap::core::reader.
 *
 * The state machine cycles through EXPECT_FILE_HEADER ->
 * EXPECT_CHUNK_HEADER -> EXPECT_CHUNK_BODY -> (back to header).
 * Each step accumulates exactly the next expected number of bytes
 * into m_pending (header) or m_chunk_buf (body), then dispatches.
 *
 * Internally split into three collaborators in the anonymous
 * namespace below:
 *
 *   * chunk_assembler: state machine + buffer assembly, hands
 *     each complete chunk body to a callback.
 *   * record_walker:   walks one chunk body, decodes each record,
 *     dispatches to the host event_fn (and updates trie via callback).
 *   * trie_resolver:   stores NEW_NODE -> trie entries and walks a
 *     leaf_id back to a frame array on demand.
 */

#include "cpu_snap/core/reader.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

static_assert(std::endian::native == std::endian::little,
              "cpu_snap wire format is little-endian only");

#include "cpu_snap/core/records.hpp"
#include "cpu_snap/core/types.hpp"

namespace cpu_snap::core
{

namespace
{

// NOLINTBEGIN(readability-identifier-naming): k_*-lowercase project convention
constexpr std::size_t k_file_header_bytes   = sizeof(file_header);
constexpr std::size_t k_chunk_header_bytes  = sizeof(chunk_header);
constexpr std::size_t k_record_header_bytes = 4;

// header + one empty record <= chunk_bytes <= the writer's buffer capacity
constexpr std::uint32_t k_reader_min_chunk_bytes =
    static_cast<std::uint32_t>(k_chunk_header_bytes) + 4;
constexpr std::uint32_t k_reader_max_chunk_bytes = k_default_chunk_bytes;

// Upper bound on trie nodes the reader will hold; above the writer's default
// cap, below what a hostile stream could use to exhaust memory.
constexpr std::size_t k_reader_max_trie_nodes = std::size_t{ 1 } << 24;

// Reject a resync candidate whose chunk_seq jumps forward implausibly far.
constexpr std::uint64_t k_max_resync_seq_jump = 1'000'000;
// NOLINTEND(readability-identifier-naming)

/**
 * @brief Round @p value up to the next 4-byte boundary.
 * @param value Unsigned value to align.
 * @return Smallest multiple of 4 not less than @p value.
 */
constexpr std::uint32_t
align4(std::uint32_t value) noexcept
{
    return (value + 3U) & ~3U;
}

/**
 * @brief Load a little-endian POD from a byte buffer.
 *
 * Reads `sizeof(T)` bytes from @p src using memcpy (defined behavior
 * regardless of source alignment) and returns the value in host byte
 * order. Wire format is little-endian on x86_64; no swap needed.
 *
 * @tparam T   Trivially-copyable type (typically std::uint32_t / u64).
 * @param  src Pointer to at least sizeof(T) bytes of input.
 * @return     The loaded value.
 */
template <typename T>
T
load_le(const std::byte* src) noexcept
{
    T value{};
    std::memcpy(&value, src, sizeof(T));
    return value;
}

/**
 * @brief Decode a NEW_NODE record body.
 * @param body      Pointer to body bytes.
 * @param body_size Body byte count. Must equal 16.
 * @param out       Destination record.
 * @return true on success, false on size mismatch.
 */
[[nodiscard]] bool
decode_new_node(const std::byte* body, std::uint16_t body_size,
                new_node_record& out) noexcept
{
    if(body_size != 16) return false;
    out.node_id   = load_le<std::uint32_t>(body);
    out.parent_id = load_le<std::uint32_t>(body + 4);
    out.ip        = load_le<std::uint64_t>(body + 8);
    return true;
}

/**
 * @brief Decode a SAMPLE record body.
 * @param body      Pointer to body bytes.
 * @param body_size Body byte count. Must equal 16.
 * @param out       Destination record.
 * @return true on success, false on size mismatch.
 */
[[nodiscard]] bool
decode_sample(const std::byte* body, std::uint16_t body_size, sample_record& out) noexcept
{
    if(body_size != 16) return false;
    out.timestamp_ns = load_le<std::uint64_t>(body);
    out.tid          = load_le<std::uint32_t>(body + 8);
    out.leaf_node_id = load_le<std::uint32_t>(body + 12);
    return true;
}

/**
 * @brief Decode a LOST record body.
 * @param body      Pointer to body bytes.
 * @param body_size Body byte count. Must equal 16.
 * @param out       Destination record.
 * @return true on success, false on size mismatch.
 */
[[nodiscard]] bool
decode_lost(const std::byte* body, std::uint16_t body_size, lost_record& out) noexcept
{
    if(body_size != 16) return false;
    out.wall_ns          = load_le<std::uint64_t>(body);
    out.tid              = load_le<std::uint32_t>(body + 8);
    out.lost_count_delta = load_le<std::uint32_t>(body + 12);
    return true;
}

/**
 * @brief Decode an EOF record body.
 * @param body      Pointer to body bytes.
 * @param body_size Body byte count. Must equal 16.
 * @param out       Destination record.
 * @return true on success, false on size mismatch.
 */
[[nodiscard]] bool
decode_eof(const std::byte* body, std::uint16_t body_size, eof_record& out) noexcept
{
    if(body_size != 16) return false;
    out.wall_ns               = load_le<std::uint64_t>(body);
    out.total_samples_emitted = load_le<std::uint64_t>(body + 8);
    return true;
}

/**
 * @brief Decode a CHECKPOINT record body (variable length).
 *
 * Body layout: u64 wall_ns; u32 n_entries; u32 reserved; then
 * n_entries * per_tid_checkpoint_entry_bytes of per-tid state.
 *
 * @param body      Pointer to body bytes.
 * @param body_size Body byte count.
 * @param out       Destination record.
 * @return true on success, false on size mismatch.
 */
[[nodiscard]] bool
decode_checkpoint(const std::byte* body, std::uint16_t body_size, checkpoint_record& out)
{
    if(body_size < 16) return false;
    out.wall_ns          = load_le<std::uint64_t>(body);
    const auto n_entries = load_le<std::uint32_t>(body + 8);
    // body[12..16] is the reserved padding word; skip.
    const std::size_t need =
        std::size_t{ 16 } + std::size_t{ n_entries } * k_per_tid_checkpoint_entry_bytes;
    if(need != body_size) return false;

    out.entries.resize(n_entries);
    const std::byte* cur = body + 16;
    for(std::uint32_t i = 0; i < n_entries; ++i)
    {
        per_tid_checkpoint_entry entry{};
        entry.tid          = load_le<std::uint32_t>(cur);
        entry.leaf_id      = load_le<std::uint32_t>(cur + 4);
        entry.samples      = load_le<std::uint64_t>(cur + 8);
        entry.lost         = load_le<std::uint64_t>(cur + 16);
        out.entries[i]     = entry;
        cur += k_per_tid_checkpoint_entry_bytes;
    }
    return true;
}

/**
 * @brief Decode a single module sub-record at the cursor.
 *
 * Fixed prefix: u64 base_va; u64 size; u64 file_offset; u32 build_id_len;
 * u32 path_len. Variable tail: build_id bytes + path bytes, padded to a
 * 4-byte boundary. On success, advances @p cur past the decoded module.
 *
 * @param cur Cursor into the module-list slab; advanced on success.
 * @param end One-past-end of the readable region.
 * @param out Destination module record.
 * @return true on success, false on truncation.
 */
[[nodiscard]] bool
decode_module(const std::byte*& cur, const std::byte* end, module_info& out)
{
    const std::size_t fixed = 3 * sizeof(std::uint64_t) + 2 * sizeof(std::uint32_t);
    if(static_cast<std::size_t>(end - cur) < fixed) return false;

    const std::byte* start = cur;
    out.base_va            = load_le<std::uint64_t>(cur);
    out.size               = load_le<std::uint64_t>(cur + 8);
    out.file_offset        = load_le<std::uint64_t>(cur + 16);
    const auto bid_len     = load_le<std::uint32_t>(cur + 24);
    const auto path_len    = load_le<std::uint32_t>(cur + 28);
    cur += fixed;

    if(static_cast<std::size_t>(end - cur) < std::size_t{ bid_len } + path_len)
    {
        return false;
    }

    out.build_id.assign(reinterpret_cast<const std::uint8_t*>(cur),
                        reinterpret_cast<const std::uint8_t*>(cur) + bid_len);
    cur += bid_len;
    out.path.assign(reinterpret_cast<const char*>(cur),
                    reinterpret_cast<const char*>(cur) + path_len);
    cur += path_len;

    const auto written = static_cast<std::uint32_t>(cur - start);
    const std::uint32_t padded  = align4(written);
    if(static_cast<std::size_t>(end - cur) < padded - written) return false;
    cur += padded - written;
    return true;
}

/**
 * @brief Decode a MAPS_INIT record body (variable length).
 *
 * Body layout: u64 wall_ns; u32 n_modules; u32 reserved; then
 * n_modules module sub-records via decode_module().
 *
 * @param body      Pointer to body bytes.
 * @param body_size Body byte count.
 * @param out       Destination record.
 * @return true on success, false on truncation or trailing bytes.
 */
[[nodiscard]] bool
decode_maps_init(std::span<const std::byte> body, maps_init_record& out)
{
    if(body.size() < 16) return false;
    out.wall_ns       = load_le<std::uint64_t>(body.data());
    const auto n_mods = load_le<std::uint32_t>(body.data() + 8);
    // body[12..16] reserved.

    // Each module entry is at least k_min_module_bytes.
    constexpr std::size_t k_min_module_bytes = 32;
    const std::size_t     need = std::size_t{ 16 } + (std::size_t{ n_mods } * k_min_module_bytes);
    if(need > body.size()) return false;

    const std::byte* cur = body.data() + 16;
    const std::byte* end = body.data() + body.size();
    out.modules.resize(n_mods);
    for(std::uint32_t i = 0; i < n_mods; ++i)
    {
        if(!decode_module(cur, end, out.modules[i])) return false;
    }
    return cur == end;
}

/**
 * @brief Decode a MAPS_DELTA record body (variable length).
 *
 * Body layout: u64 wall_ns; u8 op; u8[3] op-padding; u32 reserved; then
 * exactly one module sub-record via decode_module().
 *
 * @param body      Pointer to body bytes.
 * @param body_size Body byte count.
 * @param out       Destination record.
 * @return true on success, false on truncation.
 */
[[nodiscard]] bool
decode_maps_delta(const std::byte* body, std::uint16_t body_size, maps_delta_record& out)
{
    const std::size_t prefix = sizeof(std::uint64_t) + 4 + sizeof(std::uint32_t);
    if(body_size < prefix) return false;
    out.wall_ns        = load_le<std::uint64_t>(body);
    const auto op_byte = static_cast<std::uint8_t>(body[8]);
    out.operation      = static_cast<maps_delta_record::op>(op_byte);
    // body[9..12] op-padding; body[12..16] reserved word.

    const std::byte* cur = body + prefix;
    const std::byte* end = body + body_size;
    if(!decode_module(cur, end, out.module)) return false;
    return cur == end;
}

/**
 * @brief Per-tid NEW_NODE trie: stores parent_id + ip per node_id and
 *        resolves leaf_id back to an innermost-first frame array.
 */
class trie_resolver
{
public:
    enum class insert_result : std::uint8_t
    {
        inserted,
        identical,
        invalid,
        orphan,
        full,
    };

    /**
     * @brief Declare (node_id, parent_id, ip), upholding the writer
     *        invariant (ids are assigned after their parent) so resolve()
     *        stays acyclic.
     * @return inserted / identical (idempotent redefinition) on success;
     *         invalid if @p parent_id does not precede @p node_id or the
     *         node is redefined differently; orphan if the parent is
     *         unknown; full if the node bound is reached.
     */
    [[nodiscard]] insert_result try_insert(std::uint32_t node_id, std::uint32_t parent_id,
                                           std::uint64_t ip)
    {
        if(parent_id >= node_id) return insert_result::invalid;
        if(const auto it = m_trie.find(node_id); it != m_trie.end())
        {
            return it->second.parent_id == parent_id && it->second.ip == ip
                       ? insert_result::identical
                       : insert_result::invalid;
        }
        if(parent_id != 0 && !m_trie.contains(parent_id)) return insert_result::orphan;
        if(m_trie.size() >= k_reader_max_trie_nodes) return insert_result::full;
        m_trie.emplace(node_id, entry{ parent_id, ip });
        return insert_result::inserted;
    }

    [[nodiscard]] std::vector<std::uint64_t> resolve(std::uint32_t leaf_id) const
    {
        // Defensive; try_insert() already prevents cycles.
        constexpr std::size_t k_max_depth = 4096;

        std::vector<std::uint64_t> out;
        std::uint32_t              cur = leaf_id;
        for(std::size_t depth = 0; cur != 0; ++depth)
        {
            if(depth >= k_max_depth) return {};
            auto it = m_trie.find(cur);
            if(it == m_trie.end()) return {};
            out.push_back(it->second.ip);
            cur = it->second.parent_id;
        }
        return out;
    }

private:
    struct entry
    {
        std::uint32_t parent_id = 0;
        std::uint64_t ip        = 0;
    };
    std::unordered_map<std::uint32_t, entry> m_trie;
};

/**
 * @brief Walks one verified chunk body, decoding + dispatching each
 *        record. NEW_NODE updates flow into the supplied trie; every
 *        emitted record flows into the supplied event sink.
 */
class record_walker
{
public:
    using emit_fn = std::function<void(const record_variant&)>;

    record_walker(trie_resolver& trie, emit_fn emit) noexcept
    : m_trie{ &trie }
    , m_emit{ std::move(emit) }
    {}

    /** Outcome of decoding a single record; distinguishes forward-compatible
     *  unknown tags (skip and keep walking) from genuine decode failures
     *  (known type, but malformed body -> abandon the rest of the chunk) and
     *  well-formed records the trie cannot accept (count and keep walking). */
    enum class dispatch_result : std::uint8_t
    {
        ok,
        unknown_type,
        decode_error,
        rejected,
    };

    /** @return number of parse errors encountered. */
    std::uint64_t walk_chunk(std::span<const std::byte> body) noexcept
    {
        std::uint64_t parse_errors = 0;
        std::size_t   pos          = 0;
        while(pos + k_record_header_bytes <= body.size())
        {
            const auto type_byte = static_cast<std::uint8_t>(body[pos]);
            const auto body_size = load_le<std::uint16_t>(body.data() + pos + 2);
            const std::size_t body_off = pos + k_record_header_bytes;
            if(body_off + body_size > body.size())
            {
                ++parse_errors;
                return parse_errors;
            }

            const auto result = dispatch_record(static_cast<record_type>(type_byte),
                                                body.data() + body_off, body_size);
            if(result == dispatch_result::decode_error)
            {
                ++parse_errors;
                return parse_errors;
            }
            if(result == dispatch_result::unknown_type)
            {
                ++m_unknown_records;
            }
            if(result == dispatch_result::rejected)
            {
                ++parse_errors;
            }

            const std::uint32_t consumed =
                static_cast<std::uint32_t>(k_record_header_bytes) + body_size;
            pos += align4(consumed);
        }
        return parse_errors;
    }

    /** @return count of records skipped for carrying an unrecognized type
     *          tag since this walker's construction. */
    [[nodiscard]] std::uint64_t unknown_records() const noexcept { return m_unknown_records; }

private:
    [[nodiscard]] dispatch_result dispatch_record(record_type type, const std::byte* body,
                                                  std::uint16_t body_size) noexcept
    {
        try
        {
            switch(type)
            {
                case record_type::new_node:
                {
                    new_node_record rec{};
                    if(!decode_new_node(body, body_size, rec)) return dispatch_result::decode_error;
                    switch(m_trie->try_insert(rec.node_id, rec.parent_id, rec.ip))
                    {
                        case trie_resolver::insert_result::invalid:
                            return dispatch_result::decode_error;
                        case trie_resolver::insert_result::orphan:
                        case trie_resolver::insert_result::full:
                            return dispatch_result::rejected;
                        case trie_resolver::insert_result::inserted:
                        case trie_resolver::insert_result::identical:
                            break;
                    }
                    m_emit(rec);
                    return dispatch_result::ok;
                }
                case record_type::sample:
                {
                    sample_record rec{};
                    if(!decode_sample(body, body_size, rec)) return dispatch_result::decode_error;
                    m_emit(rec);
                    return dispatch_result::ok;
                }
                case record_type::lost:
                {
                    lost_record rec{};
                    if(!decode_lost(body, body_size, rec)) return dispatch_result::decode_error;
                    m_emit(rec);
                    return dispatch_result::ok;
                }
                case record_type::eof:
                {
                    eof_record rec{};
                    if(!decode_eof(body, body_size, rec)) return dispatch_result::decode_error;
                    m_emit(rec);
                    return dispatch_result::ok;
                }
                case record_type::checkpoint:
                {
                    checkpoint_record rec{};
                    if(!decode_checkpoint(body, body_size, rec)) return dispatch_result::decode_error;
                    m_emit(rec);
                    return dispatch_result::ok;
                }
                case record_type::maps_init:
                {
                    maps_init_record rec{};
                    if(!decode_maps_init(std::span<const std::byte>(body, body_size), rec))
                    {
                        return dispatch_result::decode_error;
                    }
                    m_emit(rec);
                    return dispatch_result::ok;
                }
                case record_type::maps_delta:
                {
                    maps_delta_record rec{};
                    if(!decode_maps_delta(body, body_size, rec)) return dispatch_result::decode_error;
                    m_emit(rec);
                    return dispatch_result::ok;
                }
                case record_type::invalid:
                default:
                    // Reserved/unknown type: skip by body_size, not fatal.
                    return dispatch_result::unknown_type;
            }
        } catch(...)
        {
            return dispatch_result::decode_error;
        }
    }

    trie_resolver* m_trie;
    emit_fn        m_emit;
    std::uint64_t  m_unknown_records = 0;
};

/**
 * @brief File-header + chunk-header/body state machine.
 *
 * Owns the partial-chunk buffer and the file header. On each complete
 * chunk body, invokes @c on_chunk_ready(body span) so the surrounding
 * reader can hand the bytes to the record_walker.
 */
class chunk_assembler
{
public:
    enum class state : std::uint8_t
    {
        expect_file_header,
        expect_chunk_header,
        expect_chunk_body,
    };

    using on_chunk_fn = std::function<void(std::span<const std::byte>)>;

    chunk_assembler(reader_counters& counters, on_chunk_fn on_chunk) noexcept
    : m_counters{ &counters }
    , m_on_chunk{ std::move(on_chunk) }
    {}

    [[nodiscard]] const file_header& header() const noexcept { return m_hdr; }
    [[nodiscard]] bool               is_fatal() const noexcept { return m_fatal; }

    void feed(std::span<const std::byte> data) noexcept
    {
        m_counters->bytes_fed += data.size();
        if(m_fatal) return;

        std::size_t off = 0;
        while(off < data.size())
        {
            switch(m_state)
            {
                case state::expect_file_header:
                    off += consume_into_pending(data, off, k_file_header_bytes);
                    if(m_pending.size() == k_file_header_bytes)
                    {
                        finish_file_header();
                        if(m_fatal) return;
                    }
                    break;

                case state::expect_chunk_header:
                    off += consume_into_pending(data, off, k_chunk_header_bytes);
                    if(m_pending.size() == k_chunk_header_bytes)
                    {
                        finish_chunk_header();
                    }
                    break;

                case state::expect_chunk_body:
                    off += consume_into_chunk_body(data, off);
                    break;
            }
        }
    }

private:
    std::size_t consume_into_pending(std::span<const std::byte> data, std::size_t off,
                                     std::size_t target) noexcept
    {
        try
        {
            const std::size_t want = target - m_pending.size();
            const std::size_t have = std::min(want, data.size() - off);
            using diff_t           = std::span<const std::byte>::difference_type;
            m_pending.insert(m_pending.end(), data.begin() + static_cast<diff_t>(off),
                             data.begin() + static_cast<diff_t>(off) +
                                 static_cast<diff_t>(have));
            return have;
        } catch(...)
        {
            ++m_counters->parse_errors;
            m_pending.clear();
            return data.size() - off;
        }
    }

    std::size_t consume_into_chunk_body(std::span<const std::byte> data,
                                        std::size_t                off) noexcept
    {
        const std::size_t want = m_records_byte_count - m_chunk_body_filled;
        const std::size_t have = std::min(want, data.size() - off);
        std::memcpy(m_chunk_buf.data() + m_chunk_body_filled, data.data() + off, have);
        m_chunk_body_filled += have;

        if(m_chunk_body_filled == m_records_byte_count)
        {
            emit_chunk();
        }
        return have;
    }

    void finish_file_header() noexcept
    {
        std::memcpy(&m_hdr, m_pending.data(), sizeof(m_hdr));
        m_pending.clear();

        const bool magic_ok = m_hdr.magic[0] == 'C' && m_hdr.magic[1] == 'S' &&
                              m_hdr.magic[2] == 'N' && m_hdr.magic[3] == 'P';
        const bool chunk_bytes_ok = m_hdr.chunk_bytes >= k_reader_min_chunk_bytes &&
                                    m_hdr.chunk_bytes <= k_reader_max_chunk_bytes;
        if(!magic_ok || m_hdr.version_major != k_record_format_version_major || !chunk_bytes_ok)
        {
            ++m_counters->parse_errors;
            m_fatal = true;
            return;
        }
        m_state = state::expect_chunk_header;
    }

    /** @return true if @p ch is structurally plausible: known magic/version,
     *          a records_byte_count that fits inside the advertised
     *          chunk_bytes, and (once a first chunk has been seen) a
     *          chunk_seq that is a small forward step from the last one. */
    [[nodiscard]] bool is_plausible_chunk_header(const chunk_header& ch) const noexcept
    {
        if(ch.magic.at(0) != 'C' || ch.magic.at(1) != 'H') return false;
        if(ch.version != 0) return false;
        const std::uint32_t max_records_bytes =
            m_hdr.chunk_bytes - static_cast<std::uint32_t>(k_chunk_header_bytes);
        if(ch.records_byte_count > max_records_bytes) return false;
        if(m_first_chunk_seen)
        {
            if(ch.chunk_seq <= m_counters->last_seen_seq) return false;
            if(ch.chunk_seq - m_counters->last_seen_seq > k_max_resync_seq_jump) return false;
        }
        return true;
    }

    /** @brief Discard the rejected header and scan m_pending for the next
     *         plausible 'CH' offset; works across feed() boundaries. */
    void resync_from_pending() noexcept
    {
        ++m_counters->resyncs;
        constexpr std::array<std::byte, 2> k_chunk_magic{ std::byte{ 'C' }, std::byte{ 'H' } };

        std::size_t drop = m_pending.size();
        if(m_pending.size() > 1)
        {
            const auto search_from = m_pending.begin() + 1;
            const auto found        = std::search(search_from, m_pending.end(),
                                                   k_chunk_magic.begin(), k_chunk_magic.end());
            if(found != m_pending.end())
            {
                drop = static_cast<std::size_t>(found - m_pending.begin());
            }
            else if(m_pending.back() == std::byte{ 'C' })
            {
                // Keep a lone trailing 'C' in case 'H' arrives on the next feed().
                drop = m_pending.size() - 1;
            }
        }
        m_counters->bytes_skipped += drop;
        m_pending.erase(m_pending.begin(), m_pending.begin() + static_cast<std::ptrdiff_t>(drop));
    }

    void finish_chunk_header() noexcept
    {
        chunk_header ch{};
        std::memcpy(&ch, m_pending.data(), sizeof(ch));

        if(!is_plausible_chunk_header(ch))
        {
            if(ch.magic.at(0) != 'C' || ch.magic.at(1) != 'H' || ch.version != 0)
            {
                ++m_counters->parse_errors;
            }
            else
            {
                ++m_counters->chunks_rejected;
            }
            resync_from_pending();
            return;
        }

        m_records_byte_count = ch.records_byte_count;
        m_chunk_seq          = ch.chunk_seq;
        m_pending.clear();

        if(m_records_byte_count == 0)
        {
            emit_chunk();
            return;
        }

        try
        {
            m_chunk_buf.assign(m_records_byte_count, std::byte{ 0 });
        } catch(...)
        {
            ++m_counters->parse_errors;
            m_state = state::expect_chunk_header;
            return;
        }
        m_chunk_body_filled = 0;
        m_state             = state::expect_chunk_body;
    }

    void emit_chunk() noexcept
    {
        update_gap_tracking();
        try
        {
            if(m_on_chunk)
            {
                m_on_chunk(std::span<const std::byte>{ m_chunk_buf.data(),
                                                       m_chunk_buf.size() });
            }
        } catch(...)  // NOLINT(bugprone-empty-catch): host chunk callback failures are not stream errors
        {
        }
        m_chunk_buf.clear();
        m_chunk_body_filled = 0;
        m_state             = state::expect_chunk_header;
    }

    void update_gap_tracking() noexcept
    {
        if(m_first_chunk_seen && m_chunk_seq != m_counters->last_seen_seq + 1)
        {
            ++m_counters->gap_count;
        }
        m_counters->last_seen_seq = m_chunk_seq;
        m_first_chunk_seen        = true;
    }

    reader_counters* m_counters;
    on_chunk_fn      m_on_chunk;
    state            m_state = state::expect_file_header;
    bool             m_fatal = false;
    file_header      m_hdr{};

    std::vector<std::byte> m_pending;
    std::vector<std::byte> m_chunk_buf;
    std::size_t            m_chunk_body_filled  = 0;
    std::uint32_t          m_records_byte_count = 0;
    std::uint64_t          m_chunk_seq          = 0;
    bool                   m_first_chunk_seen   = false;
};

}  // namespace

struct reader::impl
{
    reader_counters counters_v{};
    trie_resolver   trie;
    event_fn        cb;
    record_walker   walker;
    chunk_assembler assembler;

    impl()
    : walker{ trie, [this](const record_variant& rec) { emit(rec); } }
    , assembler{ counters_v, [this](std::span<const std::byte> body) {
                    counters_v.parse_errors += walker.walk_chunk(body);
                    counters_v.unknown_records = walker.unknown_records();
                } }
    {}

    void on_event(event_fn fn) { cb = std::move(fn); }

    void emit(const record_variant& rec)
    {
        ++counters_v.records_emitted;
        if(cb)
        {
            try
            {
                cb(rec);
            } catch(...)  // NOLINT(bugprone-empty-catch): host callback failures
                          //   are explicitly contractual non-errors; swallowing
                          //   matches the documented reader::feed noexcept guarantee.
            {
                // Host callback failures are not stream errors; swallow.
            }
        }
    }

    void feed(std::span<const std::byte> data) noexcept { assembler.feed(data); }

    [[nodiscard]] std::vector<std::uint64_t> resolve_stack(std::uint32_t leaf_id) const
    {
        return trie.resolve(leaf_id);
    }
};

reader::reader()
: m_impl(std::make_unique<impl>())
{}

reader::~reader() = default;

void
reader::on_event(event_fn fn)
{
    m_impl->on_event(std::move(fn));
}

void
reader::feed(std::span<const std::byte> data) noexcept
{
    m_impl->feed(data);
}

std::vector<std::uint64_t>
reader::resolve_stack(std::uint32_t leaf_id) const
{
    return m_impl->resolve_stack(leaf_id);
}

const file_header&
reader::header() const noexcept
{
    return m_impl->assembler.header();
}

const reader_counters&
reader::counters() const noexcept
{
    return m_impl->counters_v;
}

bool
reader::is_fatal() const noexcept
{
    return m_impl->assembler.is_fatal();
}

}  // namespace cpu_snap::core
