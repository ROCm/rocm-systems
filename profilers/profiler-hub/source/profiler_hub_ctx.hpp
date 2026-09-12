#pragma once

#include "profiler-hub/c_interface/profiler_hub_types.h"
#include "profiler-hub/reader.hpp"
#include <memory>
#include <string_view>
#include <vector>

struct ph_ctx
{
    explicit ph_ctx(std::string_view trace_path);

    ph_schema_version_t get_storage_version();
    ph_track_list_t     get_track_list();
    ph_node_t           get_node();

private:
    void initialize_track_list();
    void initilaize_node_info();
    void initialize_node_agents();

    std::string                             m_file_path;
    std::shared_ptr<profiler_hub::reader_t> m_reader;
    ph_schema_version_t                     m_schema_version;

    profiler_hub::reader_types::track_info_list_t m_tracks;
    std::vector<ph_track_t>                       m_c_tracks;

    profiler_hub::reader_types::agent_info_list_t m_agents;
    std::vector<ph_agent_t>                       m_c_agents;

    std::vector<std::shared_ptr<profiler_hub::reader_types::node_info_t>> m_nodes;
    std::shared_ptr<ph_node_t>                                            m_c_node;
};
