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
void
setup()
{
    auto  args            = ::perfetto::TracingInitArgs{};
    auto  track_event_cfg = ::perfetto::protos::gen::TrackEventConfig{};
    auto& cfg             = tracing::get_perfetto_config();

    // environment settings
    auto shmem_size_hint = get_perfetto_shmem_size_hint();
    auto buffer_size     = get_perfetto_buffer_size();

    auto _policy =
        get_perfetto_fill_policy() == "discard"
            ? ::perfetto::protos::gen::TraceConfig_BufferConfig_FillPolicy_DISCARD
            : ::perfetto::protos::gen::TraceConfig_BufferConfig_FillPolicy_RING_BUFFER;
    auto* buffer_config = cfg.add_buffers();
    buffer_config->set_size_kb(buffer_size);
    buffer_config->set_fill_policy(_policy);

    std::set<std::string> _available_categories = {};
    std::set<std::string> _disabled_categories  = {};
    for(auto itr : { OMNITRACE_PERFETTO_CATEGORIES })
        _available_categories.emplace(itr.name);
    auto _enabled_categories = config::get_perfetto_categories();
    for(const auto& itr : _available_categories)
    {
        if(!_enabled_categories.empty() && _enabled_categories.count(itr) == 0)
            _disabled_categories.emplace(itr);
    }

    for(const auto& itr : _disabled_categories)
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
#if defined(CUSTOM_DATA_SOURCE)
    // Add the following:
    ::perfetto::DataSourceDescriptor dsd{};
    dsd.set_name("com.example.custom_data_source");
    CustomDataSource::Register(dsd);
    auto* ds_cfg = cfg.add_data_sources()->mutable_config();
    ds_cfg->set_name("com.example.custom_data_source");
    CustomDataSource::Trace([](CustomDataSource::TraceContext ctx) {
        auto packet = ctx.NewTracePacket();
        packet->set_timestamp(::perfetto::TrackEvent::GetTraceTimeNs());
        packet->set_for_testing()->set_str("Hello world!");
        PRINT_HERE("%s", "Trace");
    });
#endif
    auto& cfg             = tracing::get_perfetto_config();
    auto& tracing_session = tracing::get_perfetto_session();
    tracing_session       = ::perfetto::Tracing::NewTrace();
    tracing_session->Setup(cfg);
    tracing_session->StartBlocking();
}
}  // namespace perfetto
}  // namespace omnitrace

PERFETTO_TRACK_EVENT_STATIC_STORAGE();

#if defined(CUSTOM_DATA_SOURCE)
PERFETTO_DEFINE_DATA_SOURCE_STATIC_MEMBERS(CustomDataSource);
#endif
