// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "library/perfetto.hpp"
#include "library/config.hpp"
#include "library/tracing.hpp"

namespace omnitrace
{
namespace perfetto
{
auto&
get_config()
{
    static auto _v = ::perfetto::TraceConfig{};
    return _v;
}

auto&
get_session()
{
    static auto _v = std::unique_ptr<::perfetto::TracingSession>{};
    return _v;
}

void
setup()
{
    auto  args            = ::perfetto::TracingInitArgs{};
    auto  track_event_cfg = ::perfetto::protos::gen::TrackEventConfig{};
    auto& cfg             = get_config();

    // environment settings
    auto shmem_size_hint = config::get_perfetto_shmem_size_hint();
    auto buffer_size     = config::get_perfetto_buffer_size();

    auto _policy =
        config::get_perfetto_fill_policy() == "discard"
            ? ::perfetto::protos::gen::TraceConfig_BufferConfig_FillPolicy_DISCARD
            : ::perfetto::protos::gen::TraceConfig_BufferConfig_FillPolicy_RING_BUFFER;
    auto* buffer_config = cfg.add_buffers();
    buffer_config->set_size_kb(buffer_size);
    buffer_config->set_fill_policy(_policy);

    for(const auto& itr : config::get_disabled_categories())
    {
        OMNITRACE_VERBOSE_F(1, "Disabling perfetto track event category: %s\n",
                            itr.c_str());
        track_event_cfg.add_disabled_categories(itr);
    }

    auto* ds_cfg = cfg.add_data_sources()->mutable_config();
    ds_cfg->set_name("track_event");  // this MUST be track_event
    ds_cfg->set_track_event_config_raw(track_event_cfg.SerializeAsString());

    args.shmem_size_hint_kb = shmem_size_hint;

    if(get_backend() != "inprocess") args.backends |= ::perfetto::kSystemBackend;
    if(get_backend() != "system") args.backends |= ::perfetto::kInProcessBackend;

    ::perfetto::Tracing::Initialize(args);
    ::perfetto::TrackEvent::Register();
}

void
start()
{
    auto& cfg             = get_config();
    auto& tracing_session = get_session();
    tracing_session       = ::perfetto::Tracing::NewTrace();
    tracing_session->Setup(cfg);
    tracing_session->StartBlocking();
}
}  // namespace perfetto

std::unique_ptr<::perfetto::TracingSession>&
get_perfetto_session()
{
    return ::omnitrace::perfetto::get_session();
}
}  // namespace omnitrace

PERFETTO_TRACK_EVENT_STATIC_STORAGE();
