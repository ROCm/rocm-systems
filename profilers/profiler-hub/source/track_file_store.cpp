// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "track_file_store.hpp"

#include "cache_directory.hpp"
#include "debug.hpp"
#include "sql_cancel.hpp"
#include "track_window.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace profiler_hub
{
namespace
{

constexpr uint32_t k_file_header_bytes   = 64;
constexpr uint32_t k_event_record_bytes  = 40;
constexpr uint32_t k_sample_record_bytes = 16;
constexpr uint32_t k_index_record_bytes  = 16;
constexpr char     k_meta_magic[4]       = { 'P', 'H', 'M', '1' };
constexpr char     k_string_magic[4]     = { 'P', 'H', 'N', '1' };
constexpr char     k_event_magic[4]      = { 'P', 'H', 'E', '1' };
constexpr char     k_sample_magic[4]     = { 'P', 'H', 'S', '1' };

static_assert(std::endian::native == std::endian::little,
              "the track cache is little-endian");
static_assert(sizeof(double) == 8, "a sample value is 8 bytes");

struct event_record_t
{
    uint64_t id;
    uint64_t start;
    uint64_t end;
    uint32_t name_id;
    uint32_t type;
    uint32_t nesting_depth;
    uint32_t reserved;
};

struct sample_record_t
{
    uint64_t timestamp;
    double   value;
};

struct depth_index_record_t
{
    uint32_t type;
    uint32_t depth;
    uint64_t id;
};

struct file_header_t
{
    uint64_t record_count{};
    uint64_t checkpoint_count{};
    uint64_t checkpoint_offset{};
    uint32_t nesting_depth{};
    uint64_t start_ts{};
    uint64_t end_ts{};
};

struct loaded_checkpoint_t
{
    uint64_t              record_index{};
    uint64_t              next_start{};
    std::vector<uint64_t> open_indices;
};

struct overlap_bucket_t
{
    uint64_t event_count{};
    double   value{};
};

struct sample_bucket_t
{
    uint64_t count{};
    double   sum{};
};

static_assert(sizeof(event_record_t) == k_event_record_bytes);
static_assert(sizeof(sample_record_t) == k_sample_record_bytes);
static_assert(sizeof(depth_index_record_t) == k_index_record_bytes);

class binary_writer
{
public:
    explicit binary_writer(const std::filesystem::path& path)
    : m_out{ path, std::ios::binary | std::ios::trunc }
    {
        if(!m_out)
        {
            throw std::runtime_error{ "cannot create " + path.string() };
        }
    }

    void bytes(const void* data, size_t size)
    {
        m_out.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
        check();
    }

    void u32(uint32_t value) { bytes(&value, sizeof(value)); }
    void u64(uint64_t value) { bytes(&value, sizeof(value)); }
    void i64(int64_t value) { bytes(&value, sizeof(value)); }
    void f64(double value) { bytes(&value, sizeof(value)); }

    void text(std::string_view value)
    {
        if(value.size() > std::numeric_limits<uint32_t>::max())
        {
            throw std::runtime_error{ "track cache string is too long" };
        }
        u32(static_cast<uint32_t>(value.size()));
        bytes(value.data(), value.size());
    }

    [[nodiscard]] uint64_t offset()
    {
        const auto position = m_out.tellp();
        if(position < 0)
        {
            throw std::runtime_error{ "track cache writer lost its place" };
        }
        return static_cast<uint64_t>(position);
    }

    void seek(uint64_t offset)
    {
        m_out.seekp(static_cast<std::streamoff>(offset));
        check();
    }

private:
    void check()
    {
        if(!m_out)
        {
            throw std::runtime_error{ "track cache write failed" };
        }
    }

    std::ofstream m_out;
};

class binary_reader
{
public:
    explicit binary_reader(const std::filesystem::path& path)
    : m_in{ path, std::ios::binary }
    {
        if(!m_in)
        {
            throw std::runtime_error{ "cannot open " + path.string() };
        }
    }

    void bytes(void* data, size_t size)
    {
        m_in.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
        if(!m_in)
        {
            throw std::runtime_error{ "track cache ended early" };
        }
    }

    uint32_t u32()
    {
        uint32_t value = 0;
        bytes(&value, sizeof(value));
        return value;
    }

    uint64_t u64()
    {
        uint64_t value = 0;
        bytes(&value, sizeof(value));
        return value;
    }

    int64_t i64()
    {
        int64_t value = 0;
        bytes(&value, sizeof(value));
        return value;
    }

    double f64()
    {
        double value = 0;
        bytes(&value, sizeof(value));
        return value;
    }

    std::string text()
    {
        const uint32_t size = u32();
        if(size > (16U << 20U))
        {
            throw std::runtime_error{ "track cache string is too long" };
        }
        std::string value(size, '\0');
        if(size != 0)
        {
            bytes(value.data(), size);
        }
        return value;
    }

    void seek(uint64_t offset)
    {
        m_in.seekg(static_cast<std::streamoff>(offset));
        if(!m_in)
        {
            throw std::runtime_error{ "track cache seek failed" };
        }
    }

    template <typename Record>
    Record record()
    {
        Record value{};
        bytes(&value, sizeof(value));
        return value;
    }

private:
    std::ifstream m_in;
};

uint32_t
to_list_size(size_t size)
{
    if(size > std::numeric_limits<uint32_t>::max())
    {
        throw std::length_error{ "result has more elements than the C API can report" };
    }
    return static_cast<uint32_t>(size);
}

bool
same_magic(const char actual[4], const char expected[4])
{
    return std::memcmp(actual, expected, 4) == 0;
}

void
write_file_header(binary_writer&       out,
                  const char           magic[4],
                  uint32_t             record_size,
                  const file_header_t& header)
{
    out.bytes(magic, 4);
    out.u32(track_file_store::k_format_version);
    out.u32(record_size);
    out.u32(0);
    out.u64(header.record_count);
    out.u64(header.checkpoint_count);
    out.u64(header.checkpoint_offset);
    out.u32(header.nesting_depth);
    out.u32(0);
    out.u64(header.start_ts);
    out.u64(header.end_ts);
}

file_header_t
read_file_header(binary_reader& in, const char magic[4], uint32_t record_size)
{
    char actual[4] = {};
    in.bytes(actual, 4);
    if(!same_magic(actual, magic))
    {
        throw std::runtime_error{ "track cache file is not the expected kind" };
    }
    if(in.u32() != track_file_store::k_format_version)
    {
        throw std::runtime_error{ "track cache file version is not supported" };
    }
    if(in.u32() != record_size)
    {
        throw std::runtime_error{ "track cache record size does not match" };
    }
    std::ignore = in.u32();
    file_header_t header;
    header.record_count      = in.u64();
    header.checkpoint_count  = in.u64();
    header.checkpoint_offset = in.u64();
    header.nesting_depth     = in.u32();
    std::ignore              = in.u32();
    header.start_ts          = in.u64();
    header.end_ts            = in.u64();
    return header;
}

void
add_overlap(std::vector<overlap_bucket_t>& buckets,
            uint64_t                       start,
            uint64_t                       end,
            uint64_t                       origin,
            uint64_t                       bucket_size)
{
    if(end <= start || bucket_size == 0 || buckets.empty()) return;
    const uint64_t end_mark = end - 1;
    if(end_mark < origin) return;

    uint64_t start_bucket = start <= origin ? 0 : (start - origin) / bucket_size;
    uint64_t end_bucket   = (end_mark - origin) / bucket_size;
    if(start_bucket >= buckets.size()) return;
    if(end_bucket >= buckets.size()) end_bucket = buckets.size() - 1;
    if(end_bucket < start_bucket) return;

    for(uint64_t bucket = start_bucket; bucket <= end_bucket; ++bucket)
    {
        const uint64_t bucket_start  = origin + bucket * bucket_size;
        const uint64_t bucket_end    = bucket_start + bucket_size;
        const uint64_t overlap_start = std::max(start, bucket_start);
        const uint64_t overlap_end   = std::min(end, bucket_end);
        if(overlap_end <= overlap_start) continue;
        buckets[static_cast<size_t>(bucket)].event_count += 1;
        buckets[static_cast<size_t>(bucket)].value +=
            static_cast<double>(overlap_end - overlap_start);
    }
}

std::vector<ph_histogram_bucket_t>
sparse_overlaps(const std::vector<overlap_bucket_t>& buckets)
{
    std::vector<ph_histogram_bucket_t> sparse;
    for(size_t index = 0; index < buckets.size(); ++index)
    {
        if(buckets[index].event_count == 0) continue;
        sparse.push_back(
            ph_histogram_bucket_t{ .bucket_index = static_cast<uint32_t>(index),
                                   .event_count  = buckets[index].event_count,
                                   .value        = buckets[index].value });
    }
    return sparse;
}

std::vector<ph_histogram_bucket_t>
overlap_histogram(const std::vector<ph_event_t>& events,
                  uint64_t                       origin,
                  uint64_t                       bucket_size,
                  uint64_t                       bucket_count)
{
    if(bucket_count == 0 || bucket_size == 0) return {};
    std::vector<overlap_bucket_t> buckets(static_cast<size_t>(bucket_count));
    for(const ph_event_t& event : events)
    {
        add_overlap(buckets, event.start, event.end, origin, bucket_size);
    }
    return sparse_overlaps(buckets);
}

std::vector<ph_histogram_bucket_t>
sample_histogram(const std::vector<ph_sample_t>& samples,
                 uint64_t                        origin,
                 uint64_t                        bucket_size,
                 uint64_t                        bucket_count)
{
    if(bucket_count == 0 || bucket_size == 0) return {};
    std::vector<sample_bucket_t> buckets(static_cast<size_t>(bucket_count));
    for(const ph_sample_t& sample : samples)
    {
        if(sample.timestamp < origin) continue;
        const uint64_t index = (sample.timestamp - origin) / bucket_size;
        if(index >= bucket_count) continue;
        buckets[static_cast<size_t>(index)].count += 1;
        buckets[static_cast<size_t>(index)].sum += sample.value;
    }

    std::vector<ph_histogram_bucket_t> populated;
    for(size_t index = 0; index < buckets.size(); ++index)
    {
        if(buckets[index].count == 0) continue;
        populated.push_back(ph_histogram_bucket_t{
            .bucket_index = static_cast<uint32_t>(index),
            .event_count  = buckets[index].count,
            .value = buckets[index].sum / static_cast<double>(buckets[index].count) });
    }
    if(populated.size() < 2) return populated;

    std::vector<ph_histogram_bucket_t> filled;
    for(size_t index = 0; index < populated.size(); ++index)
    {
        filled.push_back(populated[index]);
        if(index + 1 == populated.size()) break;
        for(uint32_t gap = populated[index].bucket_index + 1;
            gap < populated[index + 1].bucket_index;
            ++gap)
        {
            filled.push_back(ph_histogram_bucket_t{ gap, 0, populated[index].value });
        }
    }
    return filled;
}

bool
overlaps_window(uint64_t start, uint64_t end, uint64_t window_start, uint64_t window_end)
{
    return start <= window_end && end >= window_start;
}

}  // namespace

track_file_store::track_file_store(uint32_t checkpoint_interval)
: m_checkpoint_interval{ checkpoint_interval == 0 ? k_checkpoint_interval
                                                  : checkpoint_interval }
{}

bool
track_file_store::open(const std::string&              cache_root,
                       const std::vector<std::string>& source_paths,
                       uint32_t                        schema,
                       uint32_t                        version_major,
                       uint32_t                        version_minor,
                       uint32_t                        version_patch)
{
    m_ready   = false;
    m_writing = false;
    clear_tracks();
    m_sources.clear();
    m_directory.clear();
    if(cache_root.empty() || source_paths.empty()) return false;

    try
    {
        const std::string cache_directory =
            trace_cache_directory(cache_root, source_paths);
        if(cache_directory.empty()) return false;

        m_sources.reserve(source_paths.size());
        for(const std::string& source_path : source_paths)
        {
            const std::filesystem::path file{ source_path };
            source_identity_t           identity;
            identity.path = source_path;
            identity.size = static_cast<uint64_t>(std::filesystem::file_size(file));
            identity.mtime =
                std::filesystem::last_write_time(file).time_since_epoch().count();
            m_sources.push_back(std::move(identity));
        }

        m_schema        = schema;
        m_version_major = version_major;
        m_version_minor = version_minor;
        m_version_patch = version_patch;
        m_directory = (std::filesystem::path{ cache_directory } / "timeline").string();

        const std::filesystem::path metadata =
            std::filesystem::path{ m_directory } / "timeline.meta";
        if(!std::filesystem::is_regular_file(metadata)) return true;

        binary_reader in{ metadata };
        char          magic[4] = {};
        in.bytes(magic, 4);
        if(!same_magic(magic, k_meta_magic) || in.u32() != k_format_version) return true;
        if(in.u32() != m_schema || in.u32() != m_version_major ||
           in.u32() != m_version_minor || in.u32() != m_version_patch)
        {
            return true;
        }

        const uint32_t source_count = in.u32();
        if(source_count != m_sources.size()) return true;
        for(const source_identity_t& identity : m_sources)
        {
            if(in.u64() != identity.size || in.i64() != identity.mtime) return true;
            if(in.text() != identity.path) return true;
        }

        m_hist_origin       = in.u64();
        m_hist_bucket_size  = in.u64();
        m_hist_bucket_count = in.u64();

        const uint32_t track_count = in.u32();
        m_tracks.reserve(track_count);
        for(uint32_t index = 0; index < track_count; ++index)
        {
            track_entry_t entry;
            entry.global_id     = in.u32();
            entry.source_index  = in.u32();
            entry.local_id      = in.u32();
            const uint32_t kind = in.u32();
            if(kind != static_cast<uint32_t>(payload_t::events) &&
               kind != static_cast<uint32_t>(payload_t::samples))
            {
                throw std::runtime_error{ "track cache entry has an unknown kind" };
            }
            entry.kind             = static_cast<payload_t>(kind);
            entry.nesting_depth    = in.u32();
            std::ignore            = in.u32();
            entry.record_count     = in.u64();
            entry.start_ts         = in.u64();
            entry.end_ts           = in.u64();
            const uint32_t buckets = in.u32();
            entry.buckets.reserve(buckets);
            for(uint32_t bucket = 0; bucket < buckets; ++bucket)
            {
                ph_histogram_bucket_t item{};
                item.bucket_index = in.u32();
                item.event_count  = in.u64();
                item.value        = in.f64();
                entry.buckets.push_back(item);
            }
            m_tracks.push_back(std::move(entry));
            index_track(m_tracks.back());
        }

        binary_reader strings{ std::filesystem::path{ m_directory } / "strings.bin" };
        char          string_magic[4] = {};
        strings.bytes(string_magic, 4);
        if(!same_magic(string_magic, k_string_magic) || strings.u32() != k_format_version)
        {
            throw std::runtime_error{ "track cache string table is not supported" };
        }
        const uint32_t string_count = strings.u32();
        m_strings.clear();
        for(uint32_t index = 0; index < string_count; ++index)
        {
            m_strings.push_back(strings.text());
        }
        m_ready = true;
        return true;
    } catch(const std::exception& failure)
    {
        LOG_WARNING(
            "[Profiler-Hub] Ignoring track cache {}: {}", m_directory, failure.what());
        clear_tracks();
        m_strings.clear();
        m_ready = false;
        return true;
    }
}

const char*
track_file_store::intern(std::string_view text)
{
    return m_strings[intern_id(text)].c_str();
}

uint32_t
track_file_store::intern_id(std::string_view text)
{
    const std::lock_guard<std::mutex> lock{ m_string_mutex };
    if(m_ready || !m_writing)
    {
        throw std::logic_error{ "track cache strings are frozen" };
    }
    const auto found = m_string_ids.find(std::string{ text });
    if(found != m_string_ids.end()) return found->second;
    if(m_strings.size() >= std::numeric_limits<uint32_t>::max())
    {
        throw std::runtime_error{ "track cache string table is full" };
    }
    const auto id = static_cast<uint32_t>(m_strings.size());
    m_strings.emplace_back(text);
    m_string_ids.emplace(m_strings.back(), id);
    return id;
}

bool
track_file_store::begin_write()
{
    if(m_directory.empty()) return false;
    try
    {
        const std::filesystem::path directory{ m_directory };
        std::filesystem::remove_all(directory);
        std::filesystem::create_directories(directory);
        clear_tracks();
        {
            const std::lock_guard<std::mutex> lock{ m_string_mutex };
            m_strings.clear();
            m_string_ids.clear();
            m_strings.emplace_back();
            m_string_ids.emplace(m_strings.back(), 0);
        }
        m_hist_origin       = 0;
        m_hist_bucket_size  = 0;
        m_hist_bucket_count = 0;
        m_ready             = false;
        m_writing           = true;
        return true;
    } catch(const std::exception& failure)
    {
        LOG_WARNING("[Profiler-Hub] Cannot create track cache {}: {}",
                    m_directory,
                    failure.what());
        m_writing = false;
        return false;
    }
}

void
track_file_store::set_histogram_grid(uint64_t origin,
                                     uint64_t bucket_size,
                                     uint64_t bucket_count)
{
    m_hist_origin       = origin;
    m_hist_bucket_size  = bucket_size;
    m_hist_bucket_count = bucket_count;
}

void
track_file_store::write_events(uint32_t                       global_id,
                               uint32_t                       source_index,
                               uint32_t                       local_id,
                               const std::vector<ph_event_t>& events)
{
    if(!m_writing) throw std::logic_error{ "track cache is not open for writing" };
    if(events.empty()) return;
    if(m_by_global.contains(global_id))
    {
        throw std::runtime_error{ "track cache already has this track" };
    }

    std::vector<event_record_t> records;
    records.reserve(events.size());
    std::vector<depth_index_record_t> index;
    index.reserve(events.size());
    uint32_t deepest = 0;
    uint64_t end     = 0;
    for(const ph_event_t& event : events)
    {
        event_record_t record{};
        record.id      = event.id;
        record.start   = event.start;
        record.end     = event.end;
        record.name_id = intern_id(
            event.name == nullptr ? std::string_view{} : std::string_view{ event.name });
        record.type          = static_cast<uint32_t>(event.type);
        record.nesting_depth = event.nesting_depth;
        deepest              = std::max(deepest, event.nesting_depth);
        end                  = std::max(end, event.end);
        records.push_back(record);
        index.push_back(
            depth_index_record_t{ record.type, record.nesting_depth, record.id });
    }

    struct live_t
    {
        uint64_t index;
        uint64_t end;
    };
    struct checkpoint_t
    {
        uint64_t              record_index;
        uint64_t              next_start;
        std::vector<uint64_t> open_indices;
    };
    std::vector<live_t>       live;
    std::vector<checkpoint_t> checkpoints;
    for(size_t record_index = 0; record_index < records.size(); ++record_index)
    {
        const uint64_t next_start = records[record_index].start;
        std::erase_if(live,
                      [next_start](const live_t& item) { return item.end < next_start; });
        if(record_index % m_checkpoint_interval == 0)
        {
            checkpoint_t checkpoint;
            checkpoint.record_index = record_index;
            checkpoint.next_start   = next_start;
            checkpoint.open_indices.reserve(live.size());
            for(const live_t& item : live)
            {
                checkpoint.open_indices.push_back(item.index);
            }
            checkpoints.push_back(std::move(checkpoint));
        }
        live.push_back(live_t{ record_index, records[record_index].end });
    }

    const std::filesystem::path path =
        std::filesystem::path{ m_directory } / event_file(global_id);
    binary_writer out{ path };
    file_header_t header;
    header.record_count     = records.size();
    header.checkpoint_count = checkpoints.size();
    header.nesting_depth    = deepest;
    header.start_ts         = records.front().start;
    header.end_ts           = end;
    write_file_header(out, k_event_magic, k_event_record_bytes, header);
    out.bytes(records.data(), records.size() * sizeof(event_record_t));
    header.checkpoint_offset = out.offset();
    for(const checkpoint_t& checkpoint : checkpoints)
    {
        out.u64(checkpoint.record_index);
        out.u64(checkpoint.next_start);
        out.u32(static_cast<uint32_t>(checkpoint.open_indices.size()));
        out.u32(0);
        if(!checkpoint.open_indices.empty())
        {
            out.bytes(checkpoint.open_indices.data(),
                      checkpoint.open_indices.size() * sizeof(uint64_t));
        }
    }
    out.seek(0);
    write_file_header(out, k_event_magic, k_event_record_bytes, header);

    std::sort(index.begin(),
              index.end(),
              [](const depth_index_record_t& lhs, const depth_index_record_t& rhs) {
                  if(lhs.type != rhs.type) return lhs.type < rhs.type;
                  return lhs.id < rhs.id;
              });
    binary_writer index_out{ std::filesystem::path{ m_directory } /
                             index_file(global_id) };
    index_out.bytes(index.data(), index.size() * sizeof(depth_index_record_t));

    track_entry_t entry;
    entry.global_id     = global_id;
    entry.source_index  = source_index;
    entry.local_id      = local_id;
    entry.kind          = payload_t::events;
    entry.nesting_depth = deepest;
    entry.record_count  = records.size();
    entry.start_ts      = header.start_ts;
    entry.end_ts        = header.end_ts;
    entry.buckets =
        overlap_histogram(events, m_hist_origin, m_hist_bucket_size, m_hist_bucket_count);
    m_tracks.push_back(std::move(entry));
    index_track(m_tracks.back());
}

void
track_file_store::write_samples(uint32_t                        global_id,
                                uint32_t                        source_index,
                                uint32_t                        local_id,
                                const std::vector<ph_sample_t>& samples,
                                uint64_t                        track_end)
{
    if(!m_writing) throw std::logic_error{ "track cache is not open for writing" };
    if(samples.empty()) return;
    if(m_by_global.contains(global_id))
    {
        throw std::runtime_error{ "track cache already has this track" };
    }

    std::vector<sample_record_t> records;
    records.reserve(samples.size());
    for(const ph_sample_t& sample : samples)
    {
        records.push_back(sample_record_t{ sample.timestamp, sample.value });
    }

    const std::filesystem::path path =
        std::filesystem::path{ m_directory } / sample_file(global_id);
    binary_writer out{ path };
    file_header_t header;
    header.record_count = records.size();
    header.start_ts     = records.front().timestamp;
    header.end_ts       = track_end;
    write_file_header(out, k_sample_magic, k_sample_record_bytes, header);
    out.bytes(records.data(), records.size() * sizeof(sample_record_t));

    track_entry_t entry;
    entry.global_id    = global_id;
    entry.source_index = source_index;
    entry.local_id     = local_id;
    entry.kind         = payload_t::samples;
    entry.record_count = records.size();
    entry.start_ts     = header.start_ts;
    entry.end_ts       = header.end_ts;
    entry.buckets =
        sample_histogram(samples, m_hist_origin, m_hist_bucket_size, m_hist_bucket_count);
    m_tracks.push_back(std::move(entry));
    index_track(m_tracks.back());
}

bool
track_file_store::finish_write()
{
    if(!m_writing) return false;

    {
        binary_writer strings{ std::filesystem::path{ m_directory } / "strings.bin" };
        strings.bytes(k_string_magic, 4);
        strings.u32(k_format_version);
        strings.u32(static_cast<uint32_t>(m_strings.size()));
        for(const std::string& text : m_strings)
        {
            strings.text(text);
        }
    }

    const std::filesystem::path final_path =
        std::filesystem::path{ m_directory } / "timeline.meta";
    const std::filesystem::path temporary = final_path.string() + ".tmp";
    {
        binary_writer meta{ temporary };
        meta.bytes(k_meta_magic, 4);
        meta.u32(k_format_version);
        meta.u32(m_schema);
        meta.u32(m_version_major);
        meta.u32(m_version_minor);
        meta.u32(m_version_patch);
        meta.u32(static_cast<uint32_t>(m_sources.size()));
        for(const source_identity_t& identity : m_sources)
        {
            meta.u64(identity.size);
            meta.i64(identity.mtime);
            meta.text(identity.path);
        }
        meta.u64(m_hist_origin);
        meta.u64(m_hist_bucket_size);
        meta.u64(m_hist_bucket_count);
        meta.u32(static_cast<uint32_t>(m_tracks.size()));
        for(const track_entry_t& entry : m_tracks)
        {
            meta.u32(entry.global_id);
            meta.u32(entry.source_index);
            meta.u32(entry.local_id);
            meta.u32(static_cast<uint32_t>(entry.kind));
            meta.u32(entry.nesting_depth);
            meta.u32(0);
            meta.u64(entry.record_count);
            meta.u64(entry.start_ts);
            meta.u64(entry.end_ts);
            meta.u32(static_cast<uint32_t>(entry.buckets.size()));
            for(const ph_histogram_bucket_t& bucket : entry.buckets)
            {
                meta.u32(bucket.bucket_index);
                meta.u64(bucket.event_count);
                meta.f64(bucket.value);
            }
        }
    }

    std::error_code error;
    std::filesystem::rename(temporary, final_path, error);
    if(error)
    {
        std::filesystem::remove(final_path, error);
        std::filesystem::rename(temporary, final_path);
    }

    m_writing = false;
    m_ready   = true;
    return true;
}

bool
track_file_store::has_events(uint32_t global_id) const
{
    const track_entry_t* entry = find_global(global_id);
    return entry != nullptr && entry->kind == payload_t::events;
}

bool
track_file_store::has_local_events(uint32_t source_index, uint32_t local_id) const
{
    const track_entry_t* entry = find_local(source_index, local_id);
    return entry != nullptr && entry->kind == payload_t::events;
}

bool
track_file_store::has_samples(uint32_t global_id) const
{
    const track_entry_t* entry = find_global(global_id);
    return entry != nullptr && entry->kind == payload_t::samples;
}

uint32_t
track_file_store::nesting_depth(uint32_t global_id) const
{
    const track_entry_t* entry = find_global(global_id);
    return entry == nullptr ? 0 : entry->nesting_depth;
}

uint64_t
track_file_store::record_count(uint32_t global_id) const
{
    const track_entry_t* entry = find_global(global_id);
    return entry == nullptr ? 0 : entry->record_count;
}

std::optional<uint32_t>
track_file_store::nesting_depth_of(uint32_t        source_index,
                                   uint32_t        local_id,
                                   uint64_t        event_id,
                                   ph_event_type_t type) const
{
    const track_entry_t* entry = find_local(source_index, local_id);
    if(entry == nullptr || entry->kind != payload_t::events || entry->record_count == 0)
    {
        return std::nullopt;
    }

    const std::filesystem::path path =
        std::filesystem::path{ m_directory } / index_file(entry->global_id);
    binary_reader in{ path };
    const auto    wanted = static_cast<uint32_t>(type);
    uint64_t      low    = 0;
    uint64_t      high   = entry->record_count;
    while(low < high)
    {
        const uint64_t middle = low + (high - low) / 2;
        in.seek(middle * sizeof(depth_index_record_t));
        const auto record = in.record<depth_index_record_t>();
        if(record.type < wanted || (record.type == wanted && record.id < event_id))
        {
            low = middle + 1;
        }
        else
        {
            high = middle;
        }
    }
    if(low >= entry->record_count) return std::nullopt;
    in.seek(low * sizeof(depth_index_record_t));
    const auto found = in.record<depth_index_record_t>();
    if(found.type != wanted || found.id != event_id) return std::nullopt;
    return found.depth;
}

ph_event_list_t
track_file_store::events(uint32_t global_id, uint64_t start_ts, uint64_t end_ts)
{
    if(!has_events(global_id))
    {
        return ph_event_list_t{ .list_size = 0, .events = nullptr };
    }

    auto held = std::make_unique<std::vector<ph_event_t>>(
        read_events(global_id, start_ts, end_ts));
    ph_event_list_t list{ .list_size = to_list_size(held->size()),
                          .events    = held->empty() ? nullptr : held->data() };
    if(list.events == nullptr) return list;

    const std::lock_guard<std::mutex> lock{ m_result_mutex };
    m_held_events.push_back(std::move(held));
    return list;
}

ph_sample_list_t
track_file_store::samples(uint32_t global_id, uint64_t start_ts, uint64_t end_ts)
{
    if(!has_samples(global_id))
    {
        return ph_sample_list_t{ .list_size = 0, .samples = nullptr };
    }

    auto held = std::make_unique<std::vector<ph_sample_t>>(
        read_samples(global_id, start_ts, end_ts));
    ph_sample_list_t list{ .list_size = to_list_size(held->size()),
                           .samples   = held->empty() ? nullptr : held->data() };
    if(list.samples == nullptr) return list;

    const std::lock_guard<std::mutex> lock{ m_result_mutex };
    m_held_samples.push_back(std::move(held));
    return list;
}

void
track_file_store::release_events(ph_event_t* events)
{
    if(events == nullptr) return;
    const std::lock_guard<std::mutex> lock{ m_result_mutex };
    const auto                        found = std::find_if(
        m_held_events.begin(), m_held_events.end(), [events](const auto& held) {
            return held != nullptr && held->data() == events;
        });
    if(found != m_held_events.end()) m_held_events.erase(found);
}

void
track_file_store::release_samples(ph_sample_t* samples)
{
    if(samples == nullptr) return;
    const std::lock_guard<std::mutex> lock{ m_result_mutex };
    const auto                        found = std::find_if(
        m_held_samples.begin(), m_held_samples.end(), [samples](const auto& held) {
            return held != nullptr && held->data() == samples;
        });
    if(found != m_held_samples.end()) m_held_samples.erase(found);
}

stored_histogram_t
track_file_store::histogram(uint64_t origin,
                            uint64_t bucket_size,
                            uint64_t bucket_count) const
{
    const bool same_grid = origin == m_hist_origin && bucket_size == m_hist_bucket_size &&
                           bucket_count == m_hist_bucket_count;
    if(!same_grid)
    {
        return rebuild_histogram(origin, bucket_size, bucket_count);
    }

    stored_histogram_t result;
    result.origin_ts    = m_hist_origin;
    result.bucket_size  = m_hist_bucket_size;
    result.bucket_count = m_hist_bucket_count;
    for(const track_entry_t& entry : m_tracks)
    {
        if(entry.buckets.empty()) continue;
        result.tracks.push_back(
            stored_histogram_t::track_t{ entry.global_id, entry.buckets });
    }
    return result;
}

const track_file_store::track_entry_t*
track_file_store::find_global(uint32_t global_id) const
{
    const auto found = m_by_global.find(global_id);
    if(found == m_by_global.end()) return nullptr;
    return &m_tracks[found->second];
}

const track_file_store::track_entry_t*
track_file_store::find_local(uint32_t source_index, uint32_t local_id) const
{
    const uint64_t key =
        (static_cast<uint64_t>(source_index) << 32U) | static_cast<uint64_t>(local_id);
    const auto found = m_by_local.find(key);
    if(found == m_by_local.end()) return nullptr;
    return &m_tracks[found->second];
}

std::string
track_file_store::event_file(uint32_t global_id) const
{
    return "e" + std::to_string(global_id) + ".bin";
}

std::string
track_file_store::sample_file(uint32_t global_id) const
{
    return "s" + std::to_string(global_id) + ".bin";
}

std::string
track_file_store::index_file(uint32_t global_id) const
{
    return "i" + std::to_string(global_id) + ".bin";
}

std::vector<ph_event_t>
track_file_store::read_events(uint32_t global_id,
                              uint64_t start_ts,
                              uint64_t end_ts) const
{
    const track_entry_t* entry = find_global(global_id);
    if(entry == nullptr || entry->kind != payload_t::events) return {};

    const auto name_of = [this](uint32_t name_id) -> const char* {
        if(name_id >= m_strings.size()) return "";
        return m_strings[name_id].c_str();
    };
    const auto to_event = [&](const event_record_t& record) {
        return ph_event_t{ .id            = record.id,
                           .start         = record.start,
                           .end           = record.end,
                           .name          = name_of(record.name_id),
                           .type          = static_cast<ph_event_type_t>(record.type),
                           .nesting_depth = record.nesting_depth,
                           .track_id      = global_id };
    };

    binary_reader in{ std::filesystem::path{ m_directory } / event_file(global_id) };
    const file_header_t header =
        read_file_header(in, k_event_magic, k_event_record_bytes);
    if(header.record_count == 0) return {};

    const uint64_t stride      = m_checkpoint_interval == 0
                                     ? uint64_t{ 1 }
                                     : static_cast<uint64_t>(m_checkpoint_interval);
    uint64_t       seen        = 0;
    const auto     note_record = [&] {
        if(seen % stride == 0)
        {
            throw_if_cancelled();
        }
        ++seen;
    };

    const auto take = [&](uint64_t count) {
        std::vector<ph_event_t> loaded;
        loaded.reserve(static_cast<size_t>(count));
        for(uint64_t index = 0; index < count; ++index)
        {
            note_record();
            loaded.push_back(to_event(in.record<event_record_t>()));
        }
        return loaded;
    };

    if(start_ts == 0 && end_ts == 0)
    {
        in.seek(k_file_header_bytes);
        return take(header.record_count);
    }

    const auto     window       = make_window_filter(start_ts, end_ts).time_window;
    const uint64_t window_start = window.start.value_or(0);
    const uint64_t window_end = window.end.value_or(std::numeric_limits<uint64_t>::max());

    std::vector<loaded_checkpoint_t> checkpoints;
    if(header.checkpoint_count != 0)
    {
        in.seek(header.checkpoint_offset);
        checkpoints.reserve(static_cast<size_t>(header.checkpoint_count));
        for(uint64_t index = 0; index < header.checkpoint_count; ++index)
        {
            loaded_checkpoint_t checkpoint;
            checkpoint.record_index   = in.u64();
            checkpoint.next_start     = in.u64();
            const uint32_t open_count = in.u32();
            std::ignore               = in.u32();
            if(static_cast<uint64_t>(open_count) > header.record_count)
            {
                throw std::runtime_error{ "track cache checkpoint is corrupt" };
            }
            checkpoint.open_indices.resize(open_count);
            if(open_count != 0)
            {
                in.bytes(checkpoint.open_indices.data(),
                         static_cast<size_t>(open_count) * sizeof(uint64_t));
            }
            checkpoints.push_back(std::move(checkpoint));
        }
    }

    size_t chosen = 0;
    for(size_t index = 0; index < checkpoints.size(); ++index)
    {
        if(checkpoints[index].next_start <= window_start)
            chosen = index;
        else
            break;
    }
    uint64_t scan_begin = checkpoints.empty() ? 0 : checkpoints[chosen].record_index;
    uint64_t scan_end   = header.record_count;
    for(size_t index = chosen + 1; index < checkpoints.size(); ++index)
    {
        if(checkpoints[index].next_start > window_end)
        {
            scan_end = checkpoints[index].record_index;
            break;
        }
    }

    std::vector<ph_event_t> loaded;
    if(!checkpoints.empty())
    {
        for(const uint64_t open_index : checkpoints[chosen].open_indices)
        {
            if(open_index >= header.record_count) continue;
            note_record();
            in.seek(k_file_header_bytes + open_index * sizeof(event_record_t));
            const event_record_t record = in.record<event_record_t>();
            if(overlaps_window(record.start, record.end, window_start, window_end))
            {
                loaded.push_back(to_event(record));
            }
        }
    }
    if(scan_end > scan_begin)
    {
        in.seek(k_file_header_bytes + scan_begin * sizeof(event_record_t));
        for(uint64_t index = scan_begin; index < scan_end; ++index)
        {
            note_record();
            const event_record_t record = in.record<event_record_t>();
            if(overlaps_window(record.start, record.end, window_start, window_end))
            {
                loaded.push_back(to_event(record));
            }
        }
    }
    return loaded;
}

std::vector<ph_sample_t>
track_file_store::read_samples(uint32_t global_id,
                               uint64_t start_ts,
                               uint64_t end_ts) const
{
    const track_entry_t* entry = find_global(global_id);
    if(entry == nullptr || entry->kind != payload_t::samples) return {};

    binary_reader in{ std::filesystem::path{ m_directory } / sample_file(global_id) };
    const file_header_t header =
        read_file_header(in, k_sample_magic, k_sample_record_bytes);
    if(header.record_count == 0) return {};

    const auto sample_at = [&](uint64_t index) {
        in.seek(k_file_header_bytes + index * sizeof(sample_record_t));
        return in.record<sample_record_t>();
    };
    const auto append_range =
        [&](std::vector<ph_sample_t>& destination, uint64_t begin, uint64_t end) {
            if(end <= begin) return;
            in.seek(k_file_header_bytes + begin * sizeof(sample_record_t));
            for(uint64_t index = begin; index < end; ++index)
            {
                const sample_record_t record = in.record<sample_record_t>();
                destination.push_back(
                    ph_sample_t{ .timestamp = record.timestamp, .value = record.value });
            }
        };

    if(start_ts == 0 && end_ts == 0)
    {
        std::vector<ph_sample_t> whole;
        append_range(whole, 0, header.record_count);
        return whole;
    }

    const uint64_t upper = end_ts == 0 ? std::numeric_limits<uint64_t>::max() : end_ts;
    uint64_t       low   = 0;
    uint64_t       high  = header.record_count;
    while(low < high)
    {
        const uint64_t middle = low + (high - low) / 2;
        if(sample_at(middle).timestamp < start_ts)
            low = middle + 1;
        else
            high = middle;
    }
    const uint64_t first_inside = low;

    low  = 0;
    high = header.record_count;
    while(low < high)
    {
        const uint64_t middle = low + (high - low) / 2;
        if(upper < sample_at(middle).timestamp)
            high = middle;
        else
            low = middle + 1;
    }
    const uint64_t first_after = low;

    uint64_t begin = first_inside;
    if(begin != 0) --begin;
    uint64_t end = first_after;
    if(end != header.record_count) ++end;

    std::vector<ph_sample_t> selected;
    append_range(selected, begin, end);
    if(first_after == header.record_count && !selected.empty() &&
       header.end_ts > selected.back().timestamp)
    {
        ph_sample_t hold = selected.back();
        hold.timestamp   = header.end_ts;
        selected.push_back(hold);
    }
    return selected;
}

stored_histogram_t
track_file_store::rebuild_histogram(uint64_t origin,
                                    uint64_t bucket_size,
                                    uint64_t bucket_count) const
{
    stored_histogram_t result;
    result.origin_ts    = origin;
    result.bucket_size  = bucket_size;
    result.bucket_count = bucket_count;
    if(bucket_count == 0 || bucket_size == 0) return result;

    for(const track_entry_t& entry : m_tracks)
    {
        if(entry.kind == payload_t::events)
        {
            std::vector<overlap_bucket_t> buckets(static_cast<size_t>(bucket_count));
            binary_reader                 in{ std::filesystem::path{ m_directory } /
                              event_file(entry.global_id) };
            const file_header_t           header =
                read_file_header(in, k_event_magic, k_event_record_bytes);
            for(uint64_t index = 0; index < header.record_count; ++index)
            {
                const event_record_t record = in.record<event_record_t>();
                add_overlap(buckets, record.start, record.end, origin, bucket_size);
            }
            auto sparse = sparse_overlaps(buckets);
            if(!sparse.empty())
            {
                result.tracks.push_back(
                    stored_histogram_t::track_t{ entry.global_id, std::move(sparse) });
            }
        }
        else
        {
            std::vector<ph_sample_t> samples;
            samples.reserve(static_cast<size_t>(entry.record_count));
            binary_reader       in{ std::filesystem::path{ m_directory } /
                              sample_file(entry.global_id) };
            const file_header_t header =
                read_file_header(in, k_sample_magic, k_sample_record_bytes);
            for(uint64_t index = 0; index < header.record_count; ++index)
            {
                const sample_record_t record = in.record<sample_record_t>();
                samples.push_back(
                    ph_sample_t{ .timestamp = record.timestamp, .value = record.value });
            }
            auto sparse = sample_histogram(samples, origin, bucket_size, bucket_count);
            if(!sparse.empty())
            {
                result.tracks.push_back(
                    stored_histogram_t::track_t{ entry.global_id, std::move(sparse) });
            }
        }
    }
    return result;
}

void
track_file_store::index_track(const track_entry_t& entry)
{
    const size_t slot = static_cast<size_t>(&entry - m_tracks.data());
    if(slot >= m_tracks.size())
    {
        throw std::logic_error{ "track cache indexed a track that is not stored" };
    }
    m_by_global.emplace(entry.global_id, slot);
    const uint64_t key = (static_cast<uint64_t>(entry.source_index) << 32U) |
                         static_cast<uint64_t>(entry.local_id);
    m_by_local.emplace(key, slot);
}

void
track_file_store::clear_tracks()
{
    m_tracks.clear();
    m_by_global.clear();
    m_by_local.clear();
}

}  // namespace profiler_hub
