#pragma once

#include "fmt/base.h"
#include "profiler-hub/c_interface/profiler_hub.h"
#include "profiler-hub/c_interface/profiler_hub_types.h"
#include "profiler-hub/reader.hpp"
#include "profiler-hub/version.hpp"
#include <cstdint>
#include <memory>

struct ph_ctx
{
    ph_ctx(std::string_view trace_path)
    : m_file_path{ trace_path }
    {
        auto       storage = std::make_unique<profiler_hub::storage_t>(m_file_path, "");
        const auto version = storage->get_storage_version();
        m_schema_version   = { .major = version.major,
                               .minor = version.minor,
                               .patch = version.patch };
        m_reader           = std::make_shared<profiler_hub::reader_t>(std::move(storage));

        initialize_c_track_list();
    }

    ph_schema_version_t get_storage_version() { return m_schema_version; }

    ph_track_list_t get_track_list()
    {
        return ph_track_list_t{ .list_size =
                                    static_cast<std::uint32_t>(m_c_tracks.size()),
                                .tracks = m_c_tracks.data() };
    }

    ph_node_t get_node()
    {
        const auto nodes = m_reader->get_all_nodes();
        for(const auto& node : nodes)
        {
            fmt::println("Node id {}, domanin name {}, hardware name {}, hostname {}",
                         node->node_id,
                         node->domain_name,
                         node->hardware_name,
                         node->hostname);
        }

        return {};
    }

private:
    void initialize_c_track_list()
    {
        m_tracks = m_reader->get_all_tracks();
        m_c_tracks.reserve(m_tracks.size());

        for(const auto& track : m_tracks)
        {
            m_c_tracks.push_back(ph_track_t{
                .id         = static_cast<std::uint32_t>(track->id),
                .track_name = track->name.c_str(),
                .nid        = static_cast<std::uint32_t>(track->node_info->node_id),
                .pid        = static_cast<std::uint32_t>(track->process_info->pid),
                .tid = track->thread_info  // Workaround, some tracks are missing thread
                                           // info (we need an investigation)
                           ? static_cast<std::uint32_t>(track->thread_info->thread_id)
                           : 0,
            });
        }
    }

private:
    std::string                                   m_file_path;
    ph_schema_version_t                           m_schema_version;
    profiler_hub::reader_types::track_info_list_t m_tracks;
    std::vector<ph_track_t>                       m_c_tracks;
    std::shared_ptr<profiler_hub::reader_t>       m_reader;
};
