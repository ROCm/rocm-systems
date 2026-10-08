#include "fmt/base.h"
#include "profiler-hub/cpp/reader.hpp"
#include <fmt/format.h>

#include <memory>

int
main(int argc, char** argv)
{
    if(argc < 2)
    {
        fmt::println(stderr, "usage: {} <trace.db>", argv[0]);
        return 1;
    }

    auto storage = std::make_unique<profiler_hub::storage_t>(argv[1], "");
    auto reader  = std::make_shared<profiler_hub::reader_t>(std::move(storage));

    const auto tracks = reader->get_all_tracks();
    for(const auto& track : tracks)
    {
        fmt::println("Track -> ID: {}, {}", track->id, track->name);
    }

    if(tracks.empty())
    {
        return 0;
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
