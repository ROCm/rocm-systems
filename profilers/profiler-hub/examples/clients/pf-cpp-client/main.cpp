

#include "fmt/base.h"
#include "profiler-hub/reader.hpp"
#include <fmt/format.h>

int
main()
{
    // constexpr auto trace_path = "/home/amd/Downloads/test.db";
    // constexpr auto trace_path = "/home/amd/test_dbs/3106614_results.db";
    constexpr auto trace_path = "/home/amd/test_dbs/rocpd-3930708-0.db";

    auto storage = std::make_unique<profiler_hub::storage_t>(trace_path, "");
    auto reader  = std::make_shared<profiler_hub::reader_t>(std::move(storage));

    const auto tracks = reader->get_all_tracks();
    for(const auto& track : tracks)
    {
        fmt::println("Track -> ID: {}, {}", track->id, track->name);
    }

    const auto track_events = reader->get_events_for_track(tracks[0]);
    for(const auto& event : track_events)
    {
        fmt::println("Event -> Category: {}, Display name: {}, Timestamp: [{} - {}]",
                     event.category,
                     event.display_name,
                     event.start_timestamp,
                     event.end_timestamp);
    }

    return 0;
}
