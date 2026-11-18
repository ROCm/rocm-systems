// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/perfetto/emitter.hpp"

#include "core/node_info.hpp"
#include "core/track_registry.hpp"

#include <cstdint>
#include <string>
#include <unordered_set>

#include <unistd.h>

namespace rocprofsys::core::perfetto
{
namespace
{
track_registry&
process_default_registry()
{
    static auto value = track_registry{};
    return value;
}

track_registry&
active_or_default_registry()
{
    auto* registry = get_active_track_registry();
    return registry ? *registry : process_default_registry();
}

::perfetto::Track
make_synthetic_process_track(int pid)
{
    const auto seed = std::hash<std::string>{}(std::string{ "rocprofsys_process" });
    const auto uuid = hash_combine_all(seed, static_cast<std::int64_t>(pid));
    return ::perfetto::Track{ uuid };
}
}  // namespace

std::unordered_map<hash_value_t, std::string>&
get_perfetto_track_uuids()
{
    return active_or_default_registry().map();
}

std::mutex&
get_perfetto_track_uuids_mutex()
{
    return active_or_default_registry().mutex();
}

::perfetto::Track
get_active_process_track()
{
    const auto pid = get_emitting_pid();
    if(pid <= 0)
    {
        return ::perfetto::ProcessTrack::Current();
    }
    if(pid == static_cast<int>(::getpid()))
    {
        return ::perfetto::ProcessTrack::Current();
    }
    return make_synthetic_process_track(pid);
}

void
ensure_synthetic_process_track_emitted(int pid)
{
    if(pid <= 0)
    {
        return;
    }
    if(pid == static_cast<int>(::getpid()))
    {
        return;
    }

    static thread_local std::unordered_set<int> emitted{};
    if(!emitted.insert(pid).second)
    {
        return;
    }

    auto const track = make_synthetic_process_track(pid);
    auto       desc  = track.Serialize();
    desc.mutable_process()->set_pid(pid);
    ::perfetto::TrackEvent::SetTrackDescriptor(track, desc);
}

void
emit_system_info_packet(const node_info& info)
{
    ::perfetto::TrackEvent::Trace([&](::perfetto::TrackEvent::TraceContext ctx) {
        auto  packet  = ctx.NewTracePacket();
        auto* utsname = packet->set_system_info()->set_utsname();
        utsname->set_sysname(info.system_name);
        utsname->set_release(info.release);
        utsname->set_version(info.version);
        utsname->set_machine(info.machine);
    });
}
}  // namespace rocprofsys::core::perfetto
