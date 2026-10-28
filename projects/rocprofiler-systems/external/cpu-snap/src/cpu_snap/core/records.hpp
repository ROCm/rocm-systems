#pragma once

/**
 * @file records.hpp
 * @brief On-wire record types for the cpu_snap wire format
 *        (consumed by sampling).
 *
 * chunk_framer writes these; reader reconstructs them. Variable-size
 * records are serialized field-by-field; fixed-size POD records may
 * be emitted by direct memcpy (still subject to little-endian wire
 * order).
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace cpu_snap::core
{

// NOLINTBEGIN(readability-identifier-naming): codebase uses k_*-prefixed
//   lower_case for global constants; the .clang-tidy GlobalConstantCase
//   rule defaults to UPPER_CASE but the entire project ships with k_*.
/** @brief Wire-format major version produced by this build. */
inline constexpr std::uint16_t k_record_format_version_major = 2;
/** @brief Wire-format minor version produced by this build. */
inline constexpr std::uint16_t k_record_format_version_minor = 0;
// NOLINTEND(readability-identifier-naming)

// NOLINTBEGIN(readability-identifier-naming): codebase uses lowercase enum
//   constants project-wide; the .clang-tidy EnumConstantCase rule defaults
//   to UPPER_CASE but record_type::eof would clash with the EOF macro.
/**
 * @brief Type tag for every record body.
 *
 * Codes match the u8 in the per-record header on the wire.
 *
 * @note Codes 0x08..0xFF are reserved for future record kinds; readers
 *       MUST skip unknown records by body_size.
 */
enum class record_type : std::uint8_t
{
    invalid    = 0x00,
    new_node   = 0x01,
    sample     = 0x02,
    checkpoint = 0x03,
    maps_init  = 0x04,
    maps_delta = 0x05,
    lost       = 0x06,
    eof        = 0x07,
};
// NOLINTEND(readability-identifier-naming)

// ----- fixed-size record bodies (16 B each, natural alignment) --------------
//
// These are POD with natural alignment. They are NOT packed: chunk_framer
// emits them field-by-field so on-disk layout is independent of struct
// padding. static_assert on sizeof guards against an accidental field
// reorder that would silently change the in-memory layout assumed by
// downstream component tests.

/**
 * @brief NEW_NODE: declares a trie node before any SAMPLE references it.
 *
 * @note `node_id` 0 is reserved for the trie root; the first emitted
 *       node has id 1. `parent_id == 0` means the node is a child of
 *       the root.
 */
struct new_node_record
{
    /** Id assigned to the new trie node. Strictly positive (0 = root). */
    std::uint32_t node_id   = 0;
    /** Id of the parent trie node. 0 = root. */
    std::uint32_t parent_id = 0;
    /** Instruction pointer represented by this frame. */
    std::uint64_t ip        = 0;
};
static_assert(sizeof(new_node_record) == 16, "new_node_record body must be 16 bytes");

/**
 * @brief SAMPLE: one captured stack collapsed to its leaf trie node.
 */
struct sample_record
{
    /** Sample timestamp in nanoseconds of the stream clock (see stream_clock_of()). */
    std::uint64_t timestamp_ns = 0;
    /** Linux tid of the sampled thread. */
    std::uint32_t tid          = 0;
    /** Trie node id of the innermost (leaf) frame of the captured stack. */
    std::uint32_t leaf_node_id = 0;
};
static_assert(sizeof(sample_record) == 16, "sample_record body must be 16 bytes");

/**
 * @brief LOST: kernel-reported PERF_RECORD_LOST count delta.
 *
 * @note `tid == 0` means the loss is not attributable to a single tid
 *       (aggregate).
 */
struct lost_record
{
    /** Timestamp of the LOST report in nanoseconds of the stream clock (see stream_clock_of()). */
    std::uint64_t wall_ns          = 0;
    /** Thread id the loss is attributed to (0 = aggregate / unknown). */
    std::uint32_t tid              = 0;
    /** Number of samples the kernel dropped since the previous report. */
    std::uint32_t lost_count_delta = 0;
};
static_assert(sizeof(lost_record) == 16, "lost_record body must be 16 bytes");

/**
 * @brief EOF: terminates a stream cleanly.
 *
 * Always the last record in the last chunk of a session.
 */
struct eof_record
{
    /** Timestamp at session end in nanoseconds of the stream clock (see stream_clock_of()). */
    std::uint64_t wall_ns               = 0;
    /** Cumulative SAMPLE-record count emitted across the session. */
    std::uint64_t total_samples_emitted = 0;
};
static_assert(sizeof(eof_record) == 16, "eof_record body must be 16 bytes");

// ----- variable-size record bodies ------------------------------------------
//
// These own heap storage; sizeof is implementation-defined and intentionally
// not asserted. chunk_framer serializes them field-by-field.

/** @brief One row in a CHECKPOINT body. */
struct per_tid_checkpoint_entry
{
    /** Linux tid this entry summarises. */
    std::uint32_t tid     = 0;
    /** Most recently observed leaf trie node id for this tid. */
    std::uint32_t leaf_id = 0;
    /** Cumulative SAMPLE count for this tid up to this checkpoint. */
    std::uint64_t samples = 0;
    /** Cumulative LOST count attributed to this tid up to this checkpoint. */
    std::uint64_t lost    = 0;
};

// NOLINTBEGIN(readability-identifier-naming): k_*-lowercase project convention
/** @brief Wire-format byte size of @ref per_tid_checkpoint_entry. */
inline constexpr std::size_t k_per_tid_checkpoint_entry_bytes = 24;
// NOLINTEND(readability-identifier-naming)
static_assert(sizeof(per_tid_checkpoint_entry) == k_per_tid_checkpoint_entry_bytes,
              "per_tid_checkpoint_entry must be 24 bytes");

/**
 * @brief CHECKPOINT: periodic snapshot of per-tid counters + current leaf id.
 *
 * Allows a reader to resync trie state after a missing or corrupt
 * chunk.
 */
struct checkpoint_record
{
    /** Timestamp of the checkpoint in nanoseconds of the stream clock (see stream_clock_of()). */
    std::uint64_t                         wall_ns = 0;
    /** Per-tid counter rows captured at the checkpoint. */
    std::vector<per_tid_checkpoint_entry> entries;
};

/**
 * @brief One loaded module observed by dl_module_watcher.
 *
 * Wire form is
 *   u64 base_va, u64 size, u64 file_offset,
 *   u32 build_id_len, u32 path_len, u8 build_id[], u8 path[]
 * with 4-byte tail padding.
 *
 * @note `build_id` length is bounded by 64 bytes on the wire; `path`
 *       length is bounded by 4096 bytes on the wire. These bounds are
 *       enforced by the writer, not by the struct.
 */
struct module_info
{
    /** Runtime start address of the first executable mapping (page-aligned). */
    std::uint64_t             base_va     = 0;
    /** Span (in bytes) covered by the module's executable mappings. */
    std::uint64_t             size        = 0;
    /** ELF file offset backing @c base_va. */
    std::uint64_t             file_offset = 0;
    /** GNU build-id of the backing ELF, raw bytes (max 64 on the wire). */
    std::vector<std::uint8_t> build_id;
    /** Absolute filesystem path of the backing ELF (max 4096 on the wire). */
    std::string               path;
};

/** @brief MAPS_INIT: full module list at session start. */
struct maps_init_record
{
    /** Timestamp of the snapshot in nanoseconds of the stream clock (see stream_clock_of()). */
    std::uint64_t            wall_ns = 0;
    /** All executable modules loaded at session start. */
    std::vector<module_info> modules;
};

/** @brief MAPS_DELTA: one module added or removed mid-stream. */
struct maps_delta_record
{
    // NOLINTBEGIN(readability-identifier-naming): lowercase enum constants
    //   match the project-wide convention.
    /** Wire encoding: 0 = add, 1 = remove. */
    enum class op : std::uint8_t
    {
        add    = 0,
        remove = 1,
    };
    // NOLINTEND(readability-identifier-naming)

    /** Timestamp of the delta event in nanoseconds of the stream clock (see stream_clock_of()). */
    std::uint64_t wall_ns   = 0;
    /** Whether the module was added or removed. */
    op            operation = op::add;
    /** Module affected by the delta. */
    module_info   module;
};

/** @brief Sum type holding any record body. */
using record_variant =
    std::variant<new_node_record, sample_record, checkpoint_record, maps_init_record,
                 maps_delta_record, lost_record, eof_record>;

/**
 * @brief Return the type tag for the active alternative of @p variant.
 * @param variant Record union to inspect.
 * @return The record_type tag corresponding to the active alternative.
 */
[[nodiscard]] constexpr record_type
type_of(const record_variant& variant) noexcept
{
    // Indexed by variant index in declaration order of `record_variant`.
    constexpr std::array<record_type, 7> tags = {
        record_type::new_node,  record_type::sample,     record_type::checkpoint,
        record_type::maps_init, record_type::maps_delta, record_type::lost,
        record_type::eof,
    };
    return tags[variant.index()];
}

// ----- on-disk framing (packed: layout matches the wire byte-for-byte) ------

#pragma pack(push, 1)

/**
 * @brief File header. First 64 bytes of every sampling stream.
 *
 * @note `flags` bit 0 is reserved (never set by current writers), bit 1
 *       = frame-pointer stack, bit 2 = DWARF-unwound stack. Bits 1 and 2
 *       are mutually exclusive. Other bits are reserved and must be zero.
 * @note `process_name` is UTF-8, zero-padded; not necessarily
 *       null-terminated if the name fills the buffer.
 */
struct file_header
{
    /** ASCII magic 'CSNP'. */
    std::array<char, 4>  magic;
    /** Wire-format major version (= @ref k_record_format_version_major). */
    std::uint16_t        version_major;
    /** Wire-format minor version (= @ref k_record_format_version_minor). */
    std::uint16_t        version_minor;
    /** Session start in @c CLOCK_REALTIME nanoseconds since epoch. */
    std::uint64_t        start_unix_ns;
    /** Linux pid of the recorded process. */
    std::uint64_t        pid;
    /** Target chunk size used by the writer, in bytes. */
    std::uint32_t        chunk_bytes;
    /** Capability bits: bit 0 reserved, bit 1 = frame-pointer stack,
     *  bit 2 = DWARF-unwound stack, bits 3-4 = stream clock
     *  (see k_flags_clock_shift and stream_clock_of()). */
    std::uint32_t        flags;
    /** Process name, UTF-8, zero-padded; not necessarily null-terminated. */
    std::array<char, 32> process_name;
};
static_assert(sizeof(file_header) == 64, "file_header must be 64 bytes");

/** @brief Chunk header. Precedes every chunk body. */
struct chunk_header
{
    /** ASCII magic 'CH'. */
    std::array<char, 2> magic;
    /** Chunk-header version (currently 0). */
    std::uint16_t       version;
    /** Unpadded length of the records body that follows, in bytes. */
    std::uint32_t       records_byte_count;
    /** Monotonic chunk sequence number; gaps signal dropped chunks. */
    std::uint64_t       chunk_seq;
};
static_assert(sizeof(chunk_header) == 16, "chunk_header must be 16 bytes");

#pragma pack(pop)

/**
 * @brief Clock all timestamps of a stream (samples, module, lost, checkpoint
 *        and eof records) are read from.
 *
 * Encoded in file_header::flags bits 3-4; value 3 is reserved.
 */
enum class stream_clock : std::uint8_t
{
    /** CLOCK_MONOTONIC (flag value 0; also the meaning of every pre-existing file). */
    monotonic = 0,
    /** CLOCK_BOOTTIME (flag value 1). */
    boottime  = 1,
    /** CLOCK_REALTIME, Unix-epoch nanoseconds (flag value 2). */
    wallclock = 2,
};

// NOLINTBEGIN(readability-identifier-naming)
/** @brief Bit position of the stream-clock field inside file_header::flags. */
inline constexpr std::uint32_t k_flags_clock_shift = 3;
/** @brief Mask of the stream-clock field inside file_header::flags. */
inline constexpr std::uint32_t k_flags_clock_mask  = 0x3U << k_flags_clock_shift;
// NOLINTEND(readability-identifier-naming)

/** @return @p flags with the stream-clock field set to @p clock (other bits kept). */
[[nodiscard]] constexpr std::uint32_t
with_stream_clock(std::uint32_t flags, stream_clock clock) noexcept
{
    return (flags & ~k_flags_clock_mask) |
           (static_cast<std::uint32_t>(clock) << k_flags_clock_shift);
}

/**
 * @brief Stream clock encoded in @p hdr.
 * @return The clock; the reserved field value 3 is treated as monotonic, so a
 *         malformed value never breaks parsing.
 */
[[nodiscard]] constexpr stream_clock
stream_clock_of(const file_header& hdr) noexcept
{
    switch((hdr.flags & k_flags_clock_mask) >> k_flags_clock_shift)
    {
        case 1: return stream_clock::boottime;
        case 2: return stream_clock::wallclock;
        default: break;
    }
    return stream_clock::monotonic;
}

}  // namespace cpu_snap::core
