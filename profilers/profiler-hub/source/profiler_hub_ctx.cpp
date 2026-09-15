#include "profiler_hub_ctx.hpp"
#include "profiler-hub/storage.hpp"

ph_ctx::ph_ctx(std::string_view trace_path)
: m_file_path{ trace_path }
{
    auto       storage = std::make_unique<profiler_hub::storage_t>(m_file_path, "");
    const auto version = storage->get_storage_version();
    m_schema_version   = { .major = version.major,
                           .minor = version.minor,
                           .patch = version.patch };
    m_reader           = std::make_shared<profiler_hub::reader_t>(std::move(storage));

    initialize_track_list();
    initialize_node_agents();
    initilaize_node_info();
}

ph_schema_version_t
ph_ctx::get_storage_version()
{
    return m_schema_version;
}

ph_track_list_t
ph_ctx::get_track_list()
{
    return ph_track_list_t{ .list_size = static_cast<std::uint32_t>(m_c_tracks.size()),
                            .tracks    = m_c_tracks.data() };
}

ph_node_t
ph_ctx::get_node()
{
    return *m_c_node;
}

void
ph_ctx::initialize_track_list()
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

void
ph_ctx::initilaize_node_info()
{
    m_c_node         = std::make_unique<ph_node_t>();
    m_nodes          = m_reader->get_all_nodes();
    const auto& node = m_nodes[0];

    m_c_node->info = {
        .id            = static_cast<uint32_t>(node->node_id),
        .machine_id    = node->machine_id.c_str(),
        .system_name   = node->system_name.c_str(),
        .hostname      = node->hostname.c_str(),
        .release       = node->release.c_str(),
        .version       = node->version.c_str(),
        .hardware_name = node->hardware_name.c_str(),
        .domain_name   = node->domain_name.c_str(),
    };

    m_c_node->agents =
        ph_agent_list_t{ .list_size = static_cast<std::uint32_t>(m_c_agents.size()),
                         .agents    = m_c_agents.data() };

    m_c_node->track_list = get_track_list();
}

void
ph_ctx::initialize_node_agents()
{
    m_agents = m_reader->get_all_agents();
    m_c_agents.reserve(m_agents.size());

    for(const auto& agent : m_agents)
    {
        m_c_agents.push_back(ph_agent_t{
            .id         = static_cast<std::uint32_t>(agent->type_index),
            .agent_type = agent->agent_type.c_str(),
            .absolute_index =
                static_cast<std::uint32_t>(agent->absolute_index.value_or(0)),
            .logical_index = static_cast<std::uint32_t>(agent->logical_index.value_or(0)),
            .uuid          = static_cast<std::uint32_t>(agent->uuid.value_or(0)),
            .name          = agent->name.c_str(),
            .model_name    = agent->model_name.c_str(),
            .vendor_name   = agent->vendor_name.c_str(),
            .product_name  = agent->product_name.c_str(),
            .user_name     = agent->user_name.c_str(),
        });
    }
}
