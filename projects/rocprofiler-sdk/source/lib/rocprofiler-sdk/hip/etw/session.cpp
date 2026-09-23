// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/hip/etw/session.hpp"

#if defined(_WIN32)

#    include "lib/common/environment.hpp"
#    include "lib/common/logging.hpp"
#    include "lib/common/static_object.hpp"
#    include "lib/common/utility.hpp"
#    include "lib/common/uuid_v7.hpp"
#    include "lib/rocprofiler-sdk/hip/etw/decoder.hpp"

#    include <evntrace.h>

#    include <array>
#    include <cstddef>
#    include <cstdint>
#    include <cstring>
#    include <mutex>
#    include <string>
#    include <vector>

namespace rocprofiler
{
namespace hip
{
namespace etw
{
namespace
{
// rocm_hip_tlg, defined by clr/hipamd/src/trace/hip_trace_etw.cpp.
constexpr auto provider_guid =
    GUID{0x726f636d, 0x6869, 0x7000, {0x74, 0x6c, 0x67, 0x00, 0x00, 0x00, 0x00, 0x02}};

constexpr ULONG session_buffer_size_kb = 64;

constexpr ULONG provider_enable_timeout_ms = 10000;

void WINAPI
event_record_callback(PEVENT_RECORD event_record)
{
    handle_event(event_record);
}

std::wstring
make_session_name()
{
    // Unique per run so concurrent sessions never collide and a crash orphans at most one.
    auto name = std::string{"rocprofiler-sdk-hip-"} + std::to_string(::GetCurrentProcessId()) +
                "-" + common::generate_uuid_v7(common::timestamp_ns());

    return std::wstring{name.begin(), name.end()};
}

std::wstring
make_log_path(const std::wstring& name)
{
    auto directory = std::array<wchar_t, MAX_PATH + 1>{};

    auto length = ::GetTempPathW(static_cast<DWORD>(directory.size()), directory.data());
    if(length == 0 || length > directory.size()) return {};

    return std::wstring{directory.data(), length} + name + L".etl";
}

rocprofiler_status_t
map_control_error(ULONG error, const char* what)
{
    if(error == ERROR_ACCESS_DENIED)
    {
        ROCP_ERROR << what
                   << " was denied. Creating an ETW session requires membership in the "
                      "'Performance Log Users' group (or Administrators); running elevated is "
                      "not sufficient by itself. Add the user with: net localgroup "
                      "\"Performance Log Users\" <user> /add, then sign out and back in.";
        return ROCPROFILER_STATUS_ERROR_PERMISSION_DENIED;
    }

    if(error == ERROR_NO_SYSTEM_RESOURCES)
    {
        ROCP_ERROR << what << " failed: the system-wide limit of 64 ETW sessions is exhausted.";
        return ROCPROFILER_STATUS_ERROR_OUT_OF_RESOURCES;
    }

    ROCP_ERROR << what << " failed with Windows error " << error;
    return ROCPROFILER_STATUS_ERROR;
}

class trace_session
{
public:
    trace_session()                     = default;
    ~trace_session()                    = default;
    trace_session(const trace_session&) = delete;
    trace_session(trace_session&&)      = delete;
    trace_session& operator=(const trace_session&) = delete;
    trace_session& operator=(trace_session&&) = delete;

    rocprofiler_status_t start();
    rocprofiler_status_t stop();
    void                 drain();

private:
    rocprofiler_status_t    open();
    void                    close();
    void                    decode_log();
    EVENT_TRACE_PROPERTIES* properties();

    std::mutex             m_mutex      = {};
    uint32_t               m_ref_count  = 0;
    ULONG                  m_process_id = 0;
    std::wstring           m_name       = {};
    std::wstring           m_log_path   = {};
    std::vector<std::byte> m_properties = {};
    TRACEHANDLE            m_session    = 0;
};

EVENT_TRACE_PROPERTIES*
trace_session::properties()
{
    return reinterpret_cast<EVENT_TRACE_PROPERTIES*>(m_properties.data());
}

rocprofiler_status_t
trace_session::open()
{
    m_name     = make_session_name();
    m_log_path = make_log_path(m_name);

    if(m_log_path.empty())
    {
        ROCP_ERROR << "GetTempPathW failed with Windows error " << ::GetLastError();
        return ROCPROFILER_STATUS_ERROR;
    }

    // StartTraceW copies the session name into this allocation and reads the log file name back
    // out of it, and ControlTraceW needs both at teardown, so it has to outlive every call.
    auto name_bytes = (m_name.size() + 1) * sizeof(wchar_t);
    auto path_bytes = (m_log_path.size() + 1) * sizeof(wchar_t);

    m_properties.assign(sizeof(EVENT_TRACE_PROPERTIES) + name_bytes + path_bytes, std::byte{});

    auto* props = properties();

    props->Wnode.BufferSize = static_cast<ULONG>(m_properties.size());
    props->Wnode.Flags      = WNODE_FLAG_TRACED_GUID;
    // 1 == QPC, which puts EVENT_HEADER::TimeStamp in the same clock domain as
    // common::timestamp_ns() and lets common::qpc_ticks_to_ns() convert it.
    props->Wnode.ClientContext = 1;
    // Not EVENT_TRACE_REAL_TIME_MODE: ETW discards a real-time session's undelivered buffers
    // when the process that wrote them exits, and delivery was measured here to run two flush
    // periods behind. An application that exits promptly after its last HIP call -- which is
    // most of them -- would lose the tail of its trace, or all of it. A file-backed session
    // instead hands everything over at EVENT_TRACE_CONTROL_STOP, so what gets decoded does not
    // depend on how long the application happened to live.
    props->LogFileMode       = EVENT_TRACE_FILE_MODE_SEQUENTIAL;
    props->BufferSize        = session_buffer_size_kb;
    props->LoggerNameOffset  = sizeof(EVENT_TRACE_PROPERTIES);
    props->LogFileNameOffset = static_cast<ULONG>(sizeof(EVENT_TRACE_PROPERTIES) + name_bytes);

    std::memcpy(m_properties.data() + props->LogFileNameOffset, m_log_path.c_str(), path_bytes);

    auto error = ::StartTraceW(&m_session, m_name.c_str(), props);
    if(error != ERROR_SUCCESS)
    {
        m_session = 0;
        m_properties.clear();
        m_log_path.clear();
        return map_control_error(error, "StartTraceW");
    }

    // Set by rocprofv3-launch, which consumes events for a child it created rather than for
    // itself. Internal and unsupported: it is not a user-facing knob. Unset means "trace this
    // process", which is what an in-process tool wants.
    m_process_id = static_cast<ULONG>(
        common::get_env("ROCPROF_ETW_TARGET_PID", static_cast<uint32_t>(::GetCurrentProcessId())));

    // No kernel-side EVENT_FILTER_TYPE_PID scope filter: it caps at eight processes, it was
    // measured here to stall the enable and cost the first event of the trace, and the decode
    // callback has to re-check EVENT_HEADER::ProcessId regardless.
    auto params    = ENABLE_TRACE_PARAMETERS{};
    params.Version = ENABLE_TRACE_PARAMETERS_VERSION_2;

    // A non-zero timeout makes the enable synchronous: it returns only once every already
    // running provider has processed the notification. Without it a short-lived application can
    // finish its HIP calls before the provider observes that anyone is listening.
    //
    // MatchAnyKeyword is all-ones rather than zero. Zero is documented to mean "every event",
    // but that normalization is only applied when the provider is already registered: a provider
    // that registers after the enable replays the stored mask verbatim and matches nothing. The
    // launcher always enables first, so this is the difference between a full trace and none.
    error = ::EnableTraceEx2(m_session,
                             &provider_guid,
                             EVENT_CONTROL_CODE_ENABLE_PROVIDER,
                             0xff,
                             ~0ULL,
                             0,
                             provider_enable_timeout_ms,
                             &params);
    // ERROR_TIMEOUT only means some provider callback was slow to acknowledge; the provider is
    // enabled either way.
    if(error != ERROR_SUCCESS && error != ERROR_TIMEOUT)
    {
        close();
        return map_control_error(error, "EnableTraceEx2");
    }

    set_process_filter(m_process_id);

    ROCP_INFO << "rocm_hip_tlg consumer: tracing process " << m_process_id;

    // The session is collecting as soon as StartTraceW returns and the enable above is
    // synchronous, so a caller that starts a context before launching an application has the
    // guarantee it needs without any further handshake.
    return ROCPROFILER_STATUS_SUCCESS;
}

void
trace_session::decode_log()
{
    auto logfile        = EVENT_TRACE_LOGFILEW{};
    logfile.LogFileName = m_log_path.data();
    // RAW_TIMESTAMP is what keeps Wnode.ClientContext = 1 meaningful. Without it ProcessTrace
    // rewrites EVENT_HEADER::TimeStamp into FILETIME no matter how the session was created, and
    // qpc_ticks_to_ns would then scale a 1601 epoch by the QPC period.
    logfile.ProcessTraceMode = PROCESS_TRACE_MODE_EVENT_RECORD | PROCESS_TRACE_MODE_RAW_TIMESTAMP;
    logfile.EventRecordCallback = &event_record_callback;

    auto trace = ::OpenTraceW(&logfile);
    if(trace == INVALID_PROCESSTRACE_HANDLE)
    {
        ROCP_ERROR << "OpenTraceW failed with Windows error " << ::GetLastError()
                   << "; the trace could not be decoded";
        return;
    }

    // The session has already stopped, so this reaches the end of the file and returns rather
    // than blocking the way it would on a real-time session.
    auto error = ::ProcessTrace(&trace, 1, nullptr, nullptr);
    ROCP_WARNING_IF(error != ERROR_SUCCESS) << "ProcessTrace returned " << error;

    ::CloseTrace(trace);
}

void
trace_session::close()
{
    if(m_session != 0)
    {
        ::EnableTraceEx2(
            m_session, &provider_guid, EVENT_CONTROL_CODE_DISABLE_PROVIDER, 0, 0, 0, 0, nullptr);

        // Writes out every buffer the session still holds, so the file is complete once this
        // returns and decoding it cannot race the provider.
        auto error = ::ControlTraceW(m_session, nullptr, properties(), EVENT_TRACE_CONTROL_STOP);
        ROCP_WARNING_IF(error != ERROR_SUCCESS) << "ControlTraceW(STOP) failed with " << error;

        m_session = 0;

        if(error == ERROR_SUCCESS) decode_log();
    }

    if(!m_log_path.empty())
    {
        ROCP_WARNING_IF(::DeleteFileW(m_log_path.c_str()) == FALSE &&
                        ::GetLastError() != ERROR_FILE_NOT_FOUND)
            << "could not remove the intermediate ETW log";
        m_log_path.clear();
    }

    m_properties.clear();
    m_name.clear();
}

rocprofiler_status_t
trace_session::start()
{
    auto lk = std::unique_lock<std::mutex>{m_mutex};

    if(m_ref_count > 0)
    {
        ++m_ref_count;
        return ROCPROFILER_STATUS_SUCCESS;
    }

    auto status = open();
    if(status != ROCPROFILER_STATUS_SUCCESS) return status;

    m_ref_count = 1;
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
trace_session::stop()
{
    auto lk = std::unique_lock<std::mutex>{m_mutex};

    if(m_ref_count == 0 || --m_ref_count > 0) return ROCPROFILER_STATUS_SUCCESS;

    close();

    auto stats = reset_decoder();

    ROCP_INFO << "rocm_hip_tlg consumer: " << stats.records_emitted << " records emitted, "
              << stats.events_decoded << " events decoded, " << stats.events_dropped
              << " events dropped";
    ROCP_WARNING_IF(stats.events_dropped > 0)
        << stats.events_dropped << " rocm_hip_tlg events could not be decoded";
    // An amdhip64 built without TraceLogging support never registers the provider, so the
    // session stays valid and simply receives nothing. Name the likely cause: an empty trace is
    // otherwise indistinguishable from an application that made no HIP calls.
    ROCP_WARNING_IF(stats.events_decoded == 0)
        << "no rocm_hip_tlg events were captured for process " << m_process_id
        << ". Check that the amdhip64 it loaded supports TraceLogging: tasklist /m amdhip64*";

    return ROCPROFILER_STATUS_SUCCESS;
}

void
trace_session::drain()
{
    auto lk = std::unique_lock<std::mutex>{m_mutex};

    // Leaves the reference count alone: the contexts that took it still have to give it back,
    // and the close() their stop() then performs finds nothing left to do.
    if(m_ref_count > 0) close();
}

trace_session*
get_session()
{
    static auto*& _v = common::static_object<trace_session>::construct();
    return _v;
}
}  // namespace

rocprofiler_status_t
start_session()
{
    auto* session = get_session();
    if(!session) return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;

    return session->start();
}

rocprofiler_status_t
stop_session()
{
    auto* session = get_session();
    if(!session) return ROCPROFILER_STATUS_SUCCESS;

    return session->stop();
}

void
drain_sessions()
{
    if(auto* session = get_session(); session) session->drain();
}
}  // namespace etw
}  // namespace hip
}  // namespace rocprofiler

#else  // !_WIN32

namespace rocprofiler
{
namespace hip
{
namespace etw
{
rocprofiler_status_t
start_session()
{
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
stop_session()
{
    return ROCPROFILER_STATUS_SUCCESS;
}

void
drain_sessions()
{}
}  // namespace etw
}  // namespace hip
}  // namespace rocprofiler

#endif
