// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/c/profiler_hub_types.h"

#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace profiler_hub
{

/**
 * One histogram, already on the grid the caller asked for.
 * Track ids are the context's global track ids.
 */
struct stored_histogram_t
{
    uint64_t origin_ts{};
    uint64_t bucket_size{};
    uint64_t bucket_count{};

    struct track_t
    {
        uint32_t                           track_id{};
        std::vector<ph_histogram_bucket_t> buckets;
    };

    std::vector<track_t> tracks;
};

/**
 * Timeline cache for one opened trace. The directory is
 * <config>/<hex>/timeline, where the hex name is a stable hash of each
 * source path, size, and last-write time. A metadata file records the
 * histogram and which track lives in which file. Each duration track is a
 * fixed-size record file in start order, with checkpoints of the events
 * still open so a time slice does not have to start at the first byte.
 * Counter tracks are fixed-size sample records. Names live in one string
 * table. The record files are read for a slice; they are not kept in memory.
 *
 * The files are little-endian. A change of k_format_version, or of the
 * source identity, makes the next open rewrite them.
 */
class track_file_store
{
public:
    static constexpr uint32_t k_format_version      = 1;
    static constexpr uint32_t k_checkpoint_interval = 4096;

    explicit track_file_store(uint32_t checkpoint_interval = k_checkpoint_interval);

    /**
     * Points the store at the cache for these sources. A matching metadata
     * file leaves the store ready to read. Otherwise ready() is false until
     * begin_write() and finish_write().
     * @return false when the cache cannot be used. The caller keeps the
     *         in-memory track reader.
     */
    [[nodiscard]] bool open(const std::string&              cache_root,
                            const std::vector<std::string>& source_paths,
                            uint32_t                        schema,
                            uint32_t                        version_major,
                            uint32_t                        version_minor,
                            uint32_t                        version_patch);

    [[nodiscard]] bool               ready() const { return m_ready; }
    [[nodiscard]] bool               writing() const { return m_writing; }
    [[nodiscard]] const std::string& directory() const { return m_directory; }

    /** Copies @p text into the string table. The pointer stays valid for the
     *  life of the store. Only valid while writing. */
    [[nodiscard]] const char* intern(std::string_view text);

    [[nodiscard]] bool begin_write();
    void set_histogram_grid(uint64_t origin, uint64_t bucket_size, uint64_t bucket_count);

    void write_events(uint32_t                       global_id,
                      uint32_t                       source_index,
                      uint32_t                       local_id,
                      const std::vector<ph_event_t>& events);
    void write_samples(uint32_t                        global_id,
                       uint32_t                        source_index,
                       uint32_t                        local_id,
                       const std::vector<ph_sample_t>& samples,
                       uint64_t                        track_end);

    [[nodiscard]] bool finish_write();

    [[nodiscard]] bool has_events(uint32_t global_id) const;
    [[nodiscard]] bool has_local_events(uint32_t source_index, uint32_t local_id) const;
    [[nodiscard]] bool has_samples(uint32_t global_id) const;
    [[nodiscard]] uint32_t nesting_depth(uint32_t global_id) const;
    [[nodiscard]] uint64_t record_count(uint32_t global_id) const;

    [[nodiscard]] std::optional<uint32_t> nesting_depth_of(uint32_t        source_index,
                                                           uint32_t        local_id,
                                                           uint64_t        event_id,
                                                           ph_event_type_t type) const;

    /** Both bounds 0 returns every record. The pointers stay valid until
     *  release_events() or destruction. A window is not kept after release. */
    [[nodiscard]] ph_event_list_t  events(uint32_t global_id,
                                          uint64_t start_ts,
                                          uint64_t end_ts);
    [[nodiscard]] ph_sample_list_t samples(uint32_t global_id,
                                           uint64_t start_ts,
                                           uint64_t end_ts);
    void                           release_events(ph_event_t* events);
    void                           release_samples(ph_sample_t* samples);

    /** Returns the stored histogram when @p origin, @p bucket_size, and
     *  @p bucket_count are the grid it was built on. Any other grid is
     *  rebuilt from the record files. */
    [[nodiscard]] stored_histogram_t histogram(uint64_t origin,
                                               uint64_t bucket_size,
                                               uint64_t bucket_count) const;

private:
    enum class payload_t : uint32_t
    {
        events  = 0,
        samples = 1,
    };

    struct source_identity_t
    {
        std::string path;
        uint64_t    size{};
        int64_t     mtime{};
    };

    struct track_entry_t
    {
        uint32_t                           global_id{};
        uint32_t                           source_index{};
        uint32_t                           local_id{};
        payload_t                          kind{ payload_t::events };
        uint32_t                           nesting_depth{};
        uint64_t                           record_count{};
        uint64_t                           start_ts{};
        uint64_t                           end_ts{};
        std::vector<ph_histogram_bucket_t> buckets;
    };

    [[nodiscard]] uint32_t             intern_id(std::string_view text);
    [[nodiscard]] const track_entry_t* find_global(uint32_t global_id) const;
    [[nodiscard]] const track_entry_t* find_local(uint32_t source_index,
                                                  uint32_t local_id) const;
    [[nodiscard]] std::string          event_file(uint32_t global_id) const;
    [[nodiscard]] std::string          sample_file(uint32_t global_id) const;
    [[nodiscard]] std::string          index_file(uint32_t global_id) const;

    [[nodiscard]] std::vector<ph_event_t>  read_events(uint32_t global_id,
                                                       uint64_t start_ts,
                                                       uint64_t end_ts) const;
    [[nodiscard]] std::vector<ph_sample_t> read_samples(uint32_t global_id,
                                                        uint64_t start_ts,
                                                        uint64_t end_ts) const;
    [[nodiscard]] stored_histogram_t       rebuild_histogram(uint64_t origin,
                                                             uint64_t bucket_size,
                                                             uint64_t bucket_count) const;

    void index_track(const track_entry_t& entry);
    void clear_tracks();

    uint32_t m_checkpoint_interval;
    bool     m_ready{ false };
    bool     m_writing{ false };

    std::string                    m_directory;
    std::vector<source_identity_t> m_sources;
    uint32_t                       m_schema{};
    uint32_t                       m_version_major{};
    uint32_t                       m_version_minor{};
    uint32_t                       m_version_patch{};

    uint64_t m_hist_origin{};
    uint64_t m_hist_bucket_size{};
    uint64_t m_hist_bucket_count{};

    std::deque<std::string>                   m_strings;
    std::unordered_map<std::string, uint32_t> m_string_ids;
    std::mutex                                m_string_mutex;

    std::vector<track_entry_t>           m_tracks;
    std::unordered_map<uint32_t, size_t> m_by_global;
    std::unordered_map<uint64_t, size_t> m_by_local;

    std::mutex                                             m_result_mutex;
    std::vector<std::unique_ptr<std::vector<ph_event_t>>>  m_held_events;
    std::vector<std::unique_ptr<std::vector<ph_sample_t>>> m_held_samples;
};

}  // namespace profiler_hub
