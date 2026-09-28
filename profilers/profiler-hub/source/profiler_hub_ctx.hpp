#pragma once

#include "profiler-hub/c/profiler_hub_types.h"
#include "profiler-hub/cpp/reader.hpp"
#include <cstdint>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <vector>

struct ph_ctx
{
    explicit ph_ctx(std::string_view trace_path);

    ph_schema_version_t get_storage_version();
    ph_track_list_t     get_track_list();
    ph_node_t           get_node();
    bool has_track(uint32_t track_id) const { return m_track_by_id.count(track_id) != 0; }
    ph_event_list_t  get_track_events(uint32_t track_id,
                                      uint64_t start_ts,
                                      uint64_t end_ts);
    ph_sample_list_t get_track_samples(uint32_t track_id,
                                       uint64_t start_ts,
                                       uint64_t end_ts);

private:
    void initialize_track_list();
    void initilaize_node_info();
    void initialize_node_agents();

    std::string                             m_file_path;
    std::shared_ptr<profiler_hub::reader_t> m_reader;
    ph_schema_version_t                     m_schema_version;

    profiler_hub::reader_types::track_info_list_t m_tracks;
    std::vector<ph_track_t>                       m_c_tracks;
    std::unordered_map<uint32_t, profiler_hub::reader_types::track_info_ptr_t>
        m_track_by_id;

    profiler_hub::reader_types::agent_info_list_t m_agents;
    std::vector<ph_agent_t>                       m_c_agents;

    std::vector<std::shared_ptr<profiler_hub::reader_types::node_info_t>> m_nodes;
    std::shared_ptr<ph_node_t>                                            m_c_node;

    profiler_hub::reader_types::timeline_event_list_t m_events;
    std::vector<ph_event_t>                           m_c_events;
    std::vector<ph_sample_t>                          m_c_samples;
};
