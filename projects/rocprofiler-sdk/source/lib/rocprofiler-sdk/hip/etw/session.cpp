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

#    include "lib/common/logging.hpp"
#    include "lib/common/static_object.hpp"
#    include "lib/common/utility.hpp"
#    include "lib/common/uuid_v7.hpp"
#    include "lib/rocprofiler-sdk/hip/etw/decoder.hpp"

#    include <evntrace.h>

#    include <cstddef>
#    include <cstdint>
#    include <mutex>
#    include <string>
#    include <thread>
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

// Bounds both real-time delivery latency and how long EVENT_TRACE_CONTROL_STOP takes to
// drain, which is what keeps teardown from stalling finalization.
constexpr ULONG session_flush_timer_s = 1;

constexpr DWORD process_trace_join_timeout_ms = 10000;

constexpr ULONG provider_enable_timeout_ms = 10000;

constexpr DWORD attach_timeout_ms = 5000;

void WINAPI
event_record_callback(PEVENT_RECORD event_record)
{
    handle_event(event_record);
}

// ETW invokes this as soon as ProcessTrace has attached to the session, which is the earliest
// point at which the provider's events reach this process.
ULONG WINAPI
buffer_callback(PEVENT_TRACE_LOGFILEW logfile)
{
    if(logfile && logfile->Context) ::SetEvent(static_cast<HANDLE>(logfile->Context));
    return TRUE;
}

std::wstring
make_session_name()
{
    // Unique per run so concurrent sessions never collide and a crash orphans at most one.
    auto name = std::string{"rocprofiler-sdk-hip-"} + std::to_string(::GetCurrentProcessId()) +
                "-" + common::generate_uuid_v7(common::timestamp_ns());

    return std::wstring{name.begin(), name.end()};
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

private:
    rocprofiler_status_t    open();
    void                    close();
    void                    close_trace();
    EVENT_TRACE_PROPERTIES* properties();

    std::mutex             m_mutex      = {};
    uint32_t               m_ref_count  = 0;
    std::wstring           m_name       = {};
    std::vector<std::byte> m_properties = {};
    // ProcessTrace reads this back out of the handle, so it has to outlive open().
    EVENT_TRACE_LOGFILEW m_logfile  = {};
    HANDLE               m_attached = nullptr;
    TRACEHANDLE          m_session  = 0;
    TRACEHANDLE          m_trace    = INVALID_PROCESSTRACE_HANDLE;
    std::thread          m_thread   = {};
};

EVENT_TRACE_PROPERTIES*
trace_session::properties()
{
    return reinterpret_cast<EVENT_TRACE_PROPERTIES*>(m_properties.data());
}

rocprofiler_status_t
trace_session::open()
{
    m_name = make_session_name();

    // StartTraceW copies the session name into the tail of this allocation, and
    // ControlTraceW reads it back out at teardown, so it has to outlive both calls.
    m_properties.assign(sizeof(EVENT_TRACE_PROPERTIES) + ((m_name.size() + 1) * sizeof(wchar_t)),
                        std::byte{});

    auto* props = properties();

    props->Wnode.BufferSize = static_cast<ULONG>(m_properties.size());
    props->Wnode.Flags      = WNODE_FLAG_TRACED_GUID;
    // 1 == QPC, which puts EVENT_HEADER::TimeStamp in the same clock domain as
    // common::timestamp_ns() and lets common::qpc_ticks_to_ns() convert it.
    props->Wnode.ClientContext = 1;
    props->LogFileMode         = EVENT_TRACE_REAL_TIME_MODE;
    props->BufferSize          = session_buffer_size_kb;
    props->FlushTimer          = session_flush_timer_s;
    props->LoggerNameOffset    = sizeof(EVENT_TRACE_PROPERTIES);

    auto error = ::StartTraceW(&m_session, m_name.c_str(), props);
    if(error != ERROR_SUCCESS)
    {
        m_session = 0;
        m_properties.clear();
        return map_control_error(error, "StartTraceW");
    }

    auto process_id = static_cast<ULONG>(::GetCurrentProcessId());

    // No kernel-side EVENT_FILTER_TYPE_PID scope filter: it caps at eight processes, it was
    // measured here to stall the enable and cost the first event of the trace, and the decode
    // callback has to re-check EVENT_HEADER::ProcessId regardless.
    auto params    = ENABLE_TRACE_PARAMETERS{};
    params.Version = ENABLE_TRACE_PARAMETERS_VERSION_2;

    // A non-zero timeout makes the enable synchronous: it returns only once every already
    // running provider has processed the notification. Without it a short-lived application can
    // finish its HIP calls before the provider observes that anyone is listening.
    error = ::EnableTraceEx2(m_session,
                             &provider_guid,
                             EVENT_CONTROL_CODE_ENABLE_PROVIDER,
                             TRACE_LEVEL_VERBOSE,
                             0,
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

    set_process_filter(process_id);

    m_attached = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);

    m_logfile                     = EVENT_TRACE_LOGFILEW{};
    m_logfile.LoggerName          = m_name.data();
    m_logfile.ProcessTraceMode    = PROCESS_TRACE_MODE_REAL_TIME | PROCESS_TRACE_MODE_EVENT_RECORD;
    m_logfile.EventRecordCallback = &event_record_callback;
    m_logfile.BufferCallback      = &buffer_callback;
    m_logfile.Context             = m_attached;

    m_trace = ::OpenTraceW(&m_logfile);
    if(m_trace == INVALID_PROCESSTRACE_HANDLE)
    {
        auto open_error = ::GetLastError();
        close();
        return map_control_error(open_error, "OpenTraceW");
    }

    m_thread = std::thread{[handle = m_trace]() mutable {
        auto error_v = ::ProcessTrace(&handle, 1, nullptr, nullptr);
        ROCP_INFO_IF(error_v != ERROR_SUCCESS && error_v != ERROR_CANCELLED)
            << "ProcessTrace returned " << error_v;
    }};

    // Real-time delivery only starts once ProcessTrace has attached, so returning before that
    // loses the events of an application that finishes its work in the interim.
    if(m_attached != nullptr)
    {
        ROCP_CI_LOG_IF(WARNING,
                       ::WaitForSingleObject(m_attached, attach_timeout_ms) != WAIT_OBJECT_0)
            << "the ETW consumer did not attach within " << attach_timeout_ms
            << " ms; the beginning of the trace may be missing";
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

void
trace_session::close_trace()
{
    if(m_trace == INVALID_PROCESSTRACE_HANDLE) return;

    // A real-time session reports ERROR_CTX_CLOSE_PENDING when ProcessTrace has yet to drain;
    // that is the documented success path, not a failure.
    auto error = ::CloseTrace(m_trace);
    ROCP_WARNING_IF(error != ERROR_SUCCESS && error != ERROR_CTX_CLOSE_PENDING)
        << "CloseTrace failed with " << error;

    m_trace = INVALID_PROCESSTRACE_HANDLE;
}

void
trace_session::close()
{
    if(m_session != 0)
    {
        ::EnableTraceEx2(
            m_session, &provider_guid, EVENT_CONTROL_CODE_DISABLE_PROVIDER, 0, 0, 0, 0, nullptr);

        // EVENT_TRACE_CONTROL_STOP flushes the active buffers on its way out, so ProcessTrace
        // still sees everything the provider wrote before the disable landed.
        auto error = ::ControlTraceW(m_session, nullptr, properties(), EVENT_TRACE_CONTROL_STOP);
        ROCP_WARNING_IF(error != ERROR_SUCCESS) << "ControlTraceW(STOP) failed with " << error;

        m_session = 0;
    }

    if(m_thread.joinable())
    {
        auto* handle = static_cast<HANDLE>(m_thread.native_handle());

        // Stopping the session is what hands the flushed buffers to ProcessTrace, and
        // ProcessTrace returns on its own once it has delivered them. Waiting for that before
        // CloseTrace is what keeps the tail of the trace; closing first truncates it.
        auto exited =
            (::WaitForSingleObject(handle, process_trace_join_timeout_ms) == WAIT_OBJECT_0);

        if(!exited)
        {
            close_trace();
            exited =
                (::WaitForSingleObject(handle, process_trace_join_timeout_ms) == WAIT_OBJECT_0);
        }

        if(exited)
        {
            m_thread.join();
        }
        else
        {
            // Leaking the thread is preferable to hanging rocprofiler finalization behind a
            // ProcessTrace that never returned.
            ROCP_CI_LOG(WARNING) << "ProcessTrace did not exit within "
                                 << process_trace_join_timeout_ms << " ms; detaching";
            m_thread.detach();
        }
    }

    close_trace();

    if(m_attached != nullptr)
    {
        ::CloseHandle(m_attached);
        m_attached = nullptr;
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

    return ROCPROFILER_STATUS_SUCCESS;
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
}  // namespace etw
}  // namespace hip
}  // namespace rocprofiler

#endif
