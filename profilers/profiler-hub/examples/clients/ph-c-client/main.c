

#include <profiler-hub/c_interface/profiler_hub.h>
#include <profiler-hub/c_interface/profiler_hub_types.h>

#include <stdio.h>

int
main()
{
    const char* trace_path = "/home/amd/test_dbs/rocpd-3930708-0.db";
    ph_result_t result     = PH_RESULT_SUCCES;
    ph_ctx_t    ctx        = {};
    result                 = ph_ctx_create(&ctx, trace_path);

    ph_library_version_t libray_version;
    result = ph_get_library_version(ctx, &libray_version);
    ph_schema_version_t schema_version;
    result = ph_get_schema_version(ctx, &schema_version);

    printf("Profiler Hub Version %d.%d.%d. Schema %d.%d.%d\n",
           libray_version.major,
           libray_version.minor,
           libray_version.patch,
           schema_version.major,
           schema_version.minor,
           schema_version.patch);

    ph_track_list_t track_list;
    result = ph_get_track_list(ctx, &track_list);
    printf("Track count %d\n", track_list.list_size);

    ph_node_t node;
    ph_get_node(ctx, &node);

    return 0;
}
