#include <profiler-hub/c_interface/profiler_hub.h>
#include <profiler-hub/c_interface/profiler_hub_types.h>

#include <stdio.h>
#include <time.h>

#define MAX_BENCH_ENTRIES 32

static const char* g_bench_labels[MAX_BENCH_ENTRIES];
static double      g_bench_ms[MAX_BENCH_ENTRIES];
static int         g_bench_count = 0;

static void
record_bench(const char* label, double ms)
{
    g_bench_labels[g_bench_count] = label;
    g_bench_ms[g_bench_count]     = ms;
    g_bench_count++;
}

/* Records elapsed wall-clock time for `call` instead of printing it inline,
 * so timing output doesn't interleave with the data it produces. All
 * recorded timings are printed together at the end, see print_benchmarks(). */
#define TIME_CALL(label, call)                                                           \
    do                                                                                   \
    {                                                                                    \
        struct timespec _bench_t0, _bench_t1;                                            \
        clock_gettime(CLOCK_MONOTONIC, &_bench_t0);                                      \
        call;                                                                            \
        clock_gettime(CLOCK_MONOTONIC, &_bench_t1);                                      \
        record_bench(label,                                                              \
                     (_bench_t1.tv_sec - _bench_t0.tv_sec) * 1000.0 +                    \
                         (_bench_t1.tv_nsec - _bench_t0.tv_nsec) / 1e6);                 \
    } while(0)

static void
print_benchmarks(void)
{
    printf("\n=== Benchmarks (ms) ===\n");
    for(int i = 0; i < g_bench_count; ++i)
    {
        printf("%-28s %10.3f\n", g_bench_labels[i], g_bench_ms[i]);
    }
}

static void
print_version(ph_ctx_t ctx)
{
    ph_library_version_t library_version;
    TIME_CALL("ph_get_library_version", ph_get_library_version(ctx, &library_version));
    ph_schema_version_t schema_version;
    TIME_CALL("ph_get_schema_version", ph_get_schema_version(ctx, &schema_version));

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
}

static void
print_node_info(const ph_node_info_t* info)
{
    printf("\n=== Node ===\n");
    printf("%-14s %d\n", "id:", info->id);
    printf("%-14s %s\n", "machine id:", info->machine_id);
    printf("%-14s %s\n", "system name:", info->system_name);
    printf("%-14s %s\n", "hostname:", info->hostname);
    printf("%-14s %s\n", "release:", info->release);
    printf("%-14s %s\n", "version:", info->version);
    printf("%-14s %s\n", "hardware name:", info->hardware_name);
    printf("%-14s %s\n", "domain name:", info->domain_name);
}

static void
print_agents(const ph_agent_list_t* agents)
{
    printf("\n=== Agents (%d) ===\n", agents->list_size);
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
    for(uint32_t i = 0; i < agents->list_size; ++i)
    {
        const ph_agent_t* agent = &agents->agents[i];
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
}

static void
print_tracks(const ph_track_list_t* tracks)
{
    printf("\n=== Tracks (%d) ===\n", tracks->list_size);
    printf("%-4s %-12s %-8s %-8s %-8s %-8s %s\n",
           "id",
           "nid",
           "pid",
           "tid",
           "agent",
           "events",
           "name");
    for(uint32_t i = 0; i < tracks->list_size; ++i)
    {
        const ph_track_t* track = &tracks->tracks[i];
        printf("%-4d %-12d %-8d %-8d %-8d %-8d %s\n",
               track->id,
               track->nid,
               track->pid,
               track->tid,
               track->agent_id,
               track->event_count,
               track->track_name);
    }
}

/* Finds the first duration track (agent_id == 0) and first counter track
 * (agent_id != 0), leaving 0 in either output if none exist. */
static void
find_sample_tracks(const ph_track_list_t* tracks,
                   uint32_t*              duration_track_id,
                   uint32_t*              counter_track_id)
{
    *duration_track_id = 0;
    *counter_track_id  = 0;
    for(uint32_t i = 0; i < tracks->list_size; ++i)
    {
        const ph_track_t* track = &tracks->tracks[i];
        if(track->agent_id == 0 && *duration_track_id == 0)
        {
            *duration_track_id = track->id;
        }
        if(track->agent_id != 0 && *counter_track_id == 0)
        {
            *counter_track_id = track->id;
        }
    }
}

static void
print_events(const ph_event_list_t* events, uint32_t limit)
{
    for(uint32_t i = 0; i < events->list_size && i < limit; ++i)
    {
        printf("start: %lu, end: %lu, name: %s\n",
               (unsigned long) events->events[i].start,
               (unsigned long) events->events[i].end,
               events->events[i].name);
    }
}

static void
print_samples(const ph_sample_list_t* samples, uint32_t limit)
{
    for(uint32_t i = 0; i < samples->list_size && i < limit; ++i)
    {
        printf("timestamp: %lu, value: %f\n",
               (unsigned long) samples->samples[i].timestamp,
               samples->samples[i].value);
    }
}

/* Demonstrates ph_get_track_events(): full read, then a 25%-75% time-window
 * slice derived from the full result's own timestamp range. */
static void
demo_track_events(ph_ctx_t ctx, uint32_t track_id)
{
    ph_event_list_t events;
    TIME_CALL("ph_get_track_events (all)",
              ph_get_track_events(ctx, track_id, 0, 0, &events));

    printf("\n=== Events for track %d (%d, showing first 5) ===\n",
           track_id,
           events.list_size);
    print_events(&events, 5);

    if(events.list_size == 0) return;

    uint64_t min_start = events.events[0].start;
    uint64_t max_end   = events.events[0].end;
    for(uint32_t i = 1; i < events.list_size; ++i)
    {
        if(events.events[i].start < min_start) min_start = events.events[i].start;
        if(events.events[i].end > max_end) max_end = events.events[i].end;
    }

    const uint64_t range       = max_end - min_start;
    const uint64_t slice_start = min_start + range / 4;
    const uint64_t slice_end   = min_start + (range * 3) / 4;

    ph_event_list_t slice;
    TIME_CALL("ph_get_track_events (slice)",
              ph_get_track_events(ctx, track_id, slice_start, slice_end, &slice));

    printf("\n=== Events for track %d in [%lu, %lu] (%d, showing first 5) ===\n",
           track_id,
           (unsigned long) slice_start,
           (unsigned long) slice_end,
           slice.list_size);
    print_events(&slice, 5);
}

/* Demonstrates ph_get_track_samples(): full read, then a 25%-75% time-window
 * slice derived from the full result's own timestamp range. */
static void
demo_track_samples(ph_ctx_t ctx, uint32_t track_id)
{
    ph_sample_list_t samples;
    TIME_CALL("ph_get_track_samples (all)",
              ph_get_track_samples(ctx, track_id, 0, 0, &samples));

    printf("\n=== Samples for track %d (%d, showing first 5) ===\n",
           track_id,
           samples.list_size);
    print_samples(&samples, 5);

    if(samples.list_size == 0) return;

    uint64_t min_ts = samples.samples[0].timestamp;
    uint64_t max_ts = samples.samples[0].timestamp;
    for(uint32_t i = 1; i < samples.list_size; ++i)
    {
        if(samples.samples[i].timestamp < min_ts) min_ts = samples.samples[i].timestamp;
        if(samples.samples[i].timestamp > max_ts) max_ts = samples.samples[i].timestamp;
    }

    const uint64_t range       = max_ts - min_ts;
    const uint64_t slice_start = min_ts + range / 4;
    const uint64_t slice_end   = min_ts + (range * 3) / 4;

    ph_sample_list_t slice;
    TIME_CALL("ph_get_track_samples (slice)",
              ph_get_track_samples(ctx, track_id, slice_start, slice_end, &slice));

    printf("\n=== Samples for track %d in [%lu, %lu] (%d, showing first 5) ===\n",
           track_id,
           (unsigned long) slice_start,
           (unsigned long) slice_end,
           slice.list_size);
    print_samples(&slice, 5);
}

/* Reads all data (no time filter) for every track, to see aggregate API
 * performance across the whole trace rather than a single track. */
static void
sweep_all_tracks(ph_ctx_t ctx, const ph_track_list_t* tracks)
{
    struct timespec t0, t1;
    unsigned long   total_rows      = 0;
    unsigned long   duration_tracks = 0;
    unsigned long   counter_tracks  = 0;

    printf("\n=== Per-track sweep (all data) ===\n");
    printf("%-4s %-8s %-12s %-10s %s\n", "id", "kind", "rows", "ms", "name");

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for(uint32_t i = 0; i < tracks->list_size; ++i)
    {
        const ph_track_t* track = &tracks->tracks[i];
        struct timespec   track_t0, track_t1;
        uint32_t          rows = 0;

        clock_gettime(CLOCK_MONOTONIC, &track_t0);
        if(track->agent_id == 0)
        {
            ph_event_list_t events;
            ph_get_track_events(ctx, track->id, 0, 0, &events);
            rows = events.list_size;
            duration_tracks++;
        }
        else
        {
            ph_sample_list_t samples;
            ph_get_track_samples(ctx, track->id, 0, 0, &samples);
            rows = samples.list_size;
            counter_tracks++;
        }
        clock_gettime(CLOCK_MONOTONIC, &track_t1);

        const double track_ms = (track_t1.tv_sec - track_t0.tv_sec) * 1000.0 +
                                (track_t1.tv_nsec - track_t0.tv_nsec) / 1e6;
        printf("%-4d %-8s %-12u %-10.3f %s\n",
               track->id,
               track->agent_id == 0 ? "dur" : "cnt",
               rows,
               track_ms,
               track->track_name);

        total_rows += rows;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    const double sweep_ms =
        (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    record_bench("sweep all tracks/data", sweep_ms);

    printf("\n=== Full sweep: all tracks, all data ===\n");
    printf("tracks:          %d (%lu duration, %lu counter)\n",
           tracks->list_size,
           duration_tracks,
           counter_tracks);
    printf("total rows read: %lu\n", total_rows);
    printf("total time:      %.3f ms\n", sweep_ms);
    if(total_rows > 0)
    {
        printf("avg per row:     %.6f ms\n", sweep_ms / (double) total_rows);
    }
}

int
main(int argc, char** argv)
{
    const char* trace_path = argc > 1 ? argv[1] : "/home/amd/test_dbs/rocpd-3930708-0.db";
    ph_ctx_t    ctx        = NULL;
    TIME_CALL("ph_ctx_create", ph_ctx_create(&ctx, trace_path));

    print_version(ctx);

    ph_node_t node;
    TIME_CALL("ph_get_node", ph_get_node(ctx, &node));

    print_node_info(&node.info);
    print_agents(&node.agents);
    print_tracks(&node.track_list);

    uint32_t duration_track_id;
    uint32_t counter_track_id;
    find_sample_tracks(&node.track_list, &duration_track_id, &counter_track_id);

    if(duration_track_id != 0)
    {
        demo_track_events(ctx, duration_track_id);
    }
    if(counter_track_id != 0)
    {
        demo_track_samples(ctx, counter_track_id);
    }

    sweep_all_tracks(ctx, &node.track_list);

    TIME_CALL("ph_ctx_free", ph_ctx_free(ctx));

    print_benchmarks();
    return 0;
}
