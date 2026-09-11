#include <profiler-hub/c_interface/profiler_hub.h>
#include <profiler-hub/c_interface/profiler_hub_types.h>

#include <stdio.h>

int
main()
{
    const char* trace_path = "/home/amd/test_dbs/rocpd-3930708-0.db";
    ph_ctx_t    ctx        = {};
    ph_ctx_create(&ctx, trace_path);

    ph_library_version_t library_version;
    ph_get_library_version(ctx, &library_version);
    ph_schema_version_t schema_version;
    ph_get_schema_version(ctx, &schema_version);

    printf("=== Version ===\n");
    printf("%-10s %d.%d.%d\n",
           "Library:",
           library_version.major,
           library_version.minor,
           library_version.patch);
    printf("%-10s %d.%d.%d\n",
           "Schema:",
           schema_version.major,
           schema_version.minor,
           schema_version.patch);

    ph_node_t node;
    ph_get_node(ctx, &node);

    printf("\n=== Node ===\n");
    printf("%-14s %d\n", "id:", node.info.id);
    printf("%-14s %s\n", "machine id:", node.info.machine_id);
    printf("%-14s %s\n", "system name:", node.info.system_name);
    printf("%-14s %s\n", "hostname:", node.info.hostname);
    printf("%-14s %s\n", "release:", node.info.release);
    printf("%-14s %s\n", "version:", node.info.version);
    printf("%-14s %s\n", "hardware name:", node.info.hardware_name);
    printf("%-14s %s\n", "domain name:", node.info.domain_name);

    printf("\n=== Agents (%d) ===\n", node.agents.list_size);
    printf("%-4s %-6s %-6s %-6s %-8s %-40s %-8s %-14s %s\n",
           "id",
           "type",
           "abs",
           "logi",
           "uuid",
           "name",
           "vendor",
           "model",
           "product");
    for(uint32_t i = 0; i < node.agents.list_size; ++i)
    {
        const ph_agent_t* agent = &node.agents.agents[i];
        printf("%-4d %-6s %-6d %-6d %-8d %-40s %-8s %-14s %s\n",
               agent->id,
               agent->agent_type,
               agent->absolute_index,
               agent->logical_index,
               agent->uuid,
               agent->name,
               agent->vendor_name,
               agent->model_name,
               agent->product_name);
    }

    printf("\n=== Tracks (%d, showing first 5) ===\n", node.track_list.list_size);
    printf("%-4s %-12s %-8s %-8s %s\n", "id", "nid", "pid", "tid", "name");
    for(uint32_t i = 0; i < node.track_list.list_size && i < 5; ++i)
    {
        const ph_track_t* track = &node.track_list.tracks[i];
        printf("%-4d %-12d %-8d %-8d %s\n",
               track->id,
               track->nid,
               track->pid,
               track->tid,
               track->track_name);
    }

    ph_ctx_free(ctx);
    return 0;
}
