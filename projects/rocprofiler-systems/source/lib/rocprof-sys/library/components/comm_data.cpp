// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "library/components/comm_data.hpp"
#include "core/components/fwd.hpp"
#include "core/config.hpp"
#include "core/node_info.hpp"
#include "core/perfetto.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "library/tracing.hpp"

#include <timemory/units.hpp>

namespace rocprofsys
{
namespace component
{
namespace
{
template <typename Tp, typename... Args>
void
write_perfetto_counter_track(uint64_t _val)
{
    using counter_track = rocprofsys::perfetto_counter_track<Tp>;

    if(rocprofsys::get_use_perfetto() &&
       rocprofsys::get_state() == rocprofsys::State::Active)
    {
        auto _emplace = [](const size_t _idx) {
            if(!counter_track::exists(_idx))
            {
                std::string _label =
                    (_idx > 0) ? fmt::format(" {} [{}]", Tp::label, _idx) : Tp::label;
                counter_track::emplace(_idx, _label, "bytes");
            }
        };

        const size_t          _idx = 0;
        static std::once_flag _once{};
        std::call_once(_once, _emplace, _idx);

        static std::mutex _mutex{};
        static uint64_t   value = 0;
        uint64_t          _now  = 0;
        {
            std::unique_lock<std::mutex> _lk{ _mutex };
            _now = rocprofsys::tracing::now<uint64_t>();
            _val = (value += _val);
        }

        TRACE_COUNTER(Tp::value, counter_track::at(_idx, 0), _now, _val);
    }
}
}  // namespace

namespace
{
void
metadata_initialize_comm_data_categories()
{
    static bool _is_initialized = false;
    if(_is_initialized) return;

    trace_cache::get_metadata_registry().add_string(
        trait::name<category::comm_data>::value);
    trace_cache::get_metadata_registry().add_string(trait::name<category::mpi>::value);
    trace_cache::get_metadata_registry().add_string(trait::name<category::ucx>::value);

    _is_initialized = true;
}

template <typename Track>
void
metadata_initialize_track()
{
    auto _init_track = [&](const char* label) {
        trace_cache::get_metadata_registry().add_track({ label, std::nullopt, "{}" });
    };

    static std::once_flag _once{};
    std::call_once(_once, _init_track, Track::label);
}

void
metadata_initialize_comm_data_pmc()
{
    // find the proper values for a following definitions
    [[maybe_unused]] size_t                EVENT_CODE       = 0;
    [[maybe_unused]] size_t                INSTANCE_ID      = 0;
    [[maybe_unused]] constexpr const char* LONG_DESCRIPTION = "";
    [[maybe_unused]] constexpr const char* COMPONENT        = "";
    [[maybe_unused]] constexpr const char* BLOCK            = "";
    [[maybe_unused]] constexpr const char* EXPRESSION       = "";
    [[maybe_unused]] constexpr const char* MSG              = "bytes";
    [[maybe_unused]] constexpr const auto* TARGET_ARCH      = "CPU";
    auto                                   ni               = node_info::get_instance();
    [[maybe_unused]] constexpr const auto  DEVICE_ID = 0;  // Assuming CPU device ID is 0

#if defined(ROCPROFSYS_USE_MPI)
    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_type::CPU, DEVICE_ID, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          comm_data::mpi_send::label, "Tracks MPI communication data sizes",
          trait::name<category::mpi>::description, LONG_DESCRIPTION, COMPONENT, MSG,
          rocprofsys::trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0, 0 });
    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_type::CPU, DEVICE_ID, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          comm_data::mpi_recv::label, "Tracks MPI communication data sizes",
          trait::name<category::mpi>::description, LONG_DESCRIPTION, COMPONENT, MSG,
          rocprofsys::trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0, 0 });
#endif
    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_type::CPU, DEVICE_ID, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          comm_data::ucx_send::label, "Tracks UCX communication data sizes",
          trait::name<category::ucx>::description, LONG_DESCRIPTION, COMPONENT, MSG,
          rocprofsys::trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0, 0 });
    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_type::CPU, DEVICE_ID, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          comm_data::ucx_recv::label, "Tracks UCX communication data sizes",
          trait::name<category::ucx>::description, LONG_DESCRIPTION, COMPONENT, MSG,
          rocprofsys::trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0, 0 });
}

template <typename Track>
void
cache_comm_data_events(const uint32_t device_id, int bytes)
{
    static std::mutex _mutex{};
    static uint64_t   value = 0;
    uint64_t          _now  = 0;
    {
        std::unique_lock<std::mutex> _lk{ _mutex };
        _now  = rocprofsys::tracing::now<uint64_t>();
        bytes = (value += bytes);
    }
    const std::string track_name      = Track::label;
    const size_t      timestamp_ns    = _now;
    const std::string event_metadata  = "{}";
    const size_t      stack_id        = 0;
    const size_t      parent_stack_id = 0;
    const size_t      correlation_id  = 0;
    const std::string call_stack      = "{}";
    const std::string line_info       = "{}";

    trace_cache::get_buffer_storage().store(trace_cache::pmc_event_with_sample{
        static_cast<size_t>(category_enum_id<category::comm_data>::value),
        track_name.c_str(), timestamp_ns, event_metadata.c_str(), stack_id,
        parent_stack_id, correlation_id, call_stack.c_str(), line_info.c_str(), device_id,
        static_cast<uint8_t>(agent_type::CPU), track_name.c_str(),
        static_cast<double>(value) });
}

}  // namespace

void
comm_data::start()
{
    {
        metadata_initialize_comm_data_categories();
        metadata_initialize_comm_data_pmc();

#if defined(ROCPROFSYS_USE_MPI)
        metadata_initialize_track<mpi_send>();
        metadata_initialize_track<mpi_recv>();
#endif
        metadata_initialize_track<ucx_send>();
        metadata_initialize_track<ucx_recv>();
    }
}

void
comm_data::preinit()
{
    configure();
}

void
comm_data::global_finalize()
{
    configure();
}

void
comm_data::configure()
{
    static bool _once = false;
    if(_once) return;
    _once = true;

    comm_data_tracker_t::label()        = "comm_data";
    comm_data_tracker_t::description()  = "Tracks MPI/RCCL/UCX communication data sizes";
    comm_data_tracker_t::display_unit() = "MB";
    comm_data_tracker_t::unit()         = units::megabyte;

    auto _fmt_flags = comm_data_tracker_t::get_format_flags();
    _fmt_flags &= (std::ios_base::fixed & std::ios_base::scientific);
    _fmt_flags |= (std::ios_base::scientific);
    comm_data_tracker_t::set_precision(3);
    comm_data_tracker_t::set_format_flags(_fmt_flags);
}

#if defined(ROCPROFSYS_USE_MPI)
// MPI_Send
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, int count,
                 MPI_Datatype datatype, int dst, int tag, MPI_Comm)
{
    int _size = mpi_type_size(datatype);
    if(_size == 0) return;

    write_perfetto_counter_track<mpi_send>(count * _size);

    {
        cache_comm_data_events<mpi_send>(0, count * _size);
    }

    if(rocprofsys::get_use_timemory())
    {
        auto      _name = std::string_view{ _data.tool_id };
        tracker_t _a{ _name };
        add(_a, count * _size);
        tracker_t _b{ fmt::format("{}/dst={}", _name, dst) };
        add(_b, count * _size);
        add(fmt::format("{}/dst={}/tag={}", _name, dst, tag), count * _size);
    }
}

// MPI_Recv
void
comm_data::audit(const gotcha_data& _data, audit::incoming, void*, int count,
                 MPI_Datatype datatype, int dst, int tag, MPI_Comm, MPI_Status*)
{
    int _size = mpi_type_size(datatype);
    if(_size == 0) return;

    if(get_use_perfetto()) write_perfetto_counter_track<mpi_recv>(count * _size);

    {
        cache_comm_data_events<mpi_recv>(0, count * _size);
    }

    if(rocprofsys::get_use_timemory())
    {
        auto      _name = std::string_view{ _data.tool_id };
        tracker_t _a{ _name };
        add(_a, count * _size);
        tracker_t _b{ fmt::format("{}/dst={}", _name, dst) };
        add(_b, count * _size);
        add(fmt::format("{}/dst={}/tag={}", _name, dst, tag), count * _size);
    }
}

// MPI_Isend
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, int count,
                 MPI_Datatype datatype, int dst, int tag, MPI_Comm, MPI_Request*)
{
    int _size = mpi_type_size(datatype);
    if(_size == 0) return;

    if(get_use_perfetto()) write_perfetto_counter_track<mpi_send>(count * _size);

    {
        cache_comm_data_events<mpi_send>(0, count * _size);
    }

    if(rocprofsys::get_use_timemory())
    {
        auto      _name = std::string_view{ _data.tool_id };
        tracker_t _a{ _name };
        add(_a, count * _size);
        tracker_t _b{ fmt::format("{}/dst={}", _name, dst) };
        add(_b, count * _size);
        add(fmt::format("{}/dst={}/tag={}", _name, dst, tag), count * _size);
    }
}

// MPI_Irecv
void
comm_data::audit(const gotcha_data& _data, audit::incoming, void*, int count,
                 MPI_Datatype datatype, int dst, int tag, MPI_Comm, MPI_Request*)
{
    int _size = mpi_type_size(datatype);
    if(_size == 0) return;

    if(get_use_perfetto()) write_perfetto_counter_track<mpi_recv>(count * _size);

    {
        cache_comm_data_events<mpi_recv>(0, count * _size);
    }

    if(rocprofsys::get_use_timemory())
    {
        auto      _name = std::string_view{ _data.tool_id };
        tracker_t _a{ _name };
        add(_a, count * _size);
        tracker_t _b{ fmt::format("{}/dst={}", _name, dst) };
        add(_b, count * _size);
        add(fmt::format("{}/dst={}/tag={}", _name, dst, tag), count * _size);
    }
}

// MPI_Bcast
void
comm_data::audit(const gotcha_data& _data, audit::incoming, void*, int count,
                 MPI_Datatype datatype, int root, MPI_Comm)
{
    int _size = mpi_type_size(datatype);
    if(_size == 0) return;

    if(get_use_perfetto()) write_perfetto_counter_track<mpi_send>(count * _size);

    {
        cache_comm_data_events<mpi_send>(0, count * _size);
    }

    if(rocprofsys::get_use_timemory())
    {
        auto      _name = std::string_view{ _data.tool_id };
        tracker_t _t{ _name };
        add(_t, count * _size);
        add(fmt::format("{}/root={}", _name, root), count * _size);
    }
}

// MPI_Allreduce
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, void*, int count,
                 MPI_Datatype datatype, MPI_Op, MPI_Comm)
{
    int _size = mpi_type_size(datatype);
    if(_size == 0) return;

    if(get_use_perfetto())
    {
        write_perfetto_counter_track<mpi_recv>(count * _size);
        write_perfetto_counter_track<mpi_send>(count * _size);
    }

    {
        cache_comm_data_events<mpi_recv>(0, count * _size);
        cache_comm_data_events<mpi_send>(0, count * _size);
    }

    if(rocprofsys::get_use_timemory()) add(_data, count * _size);
}

// MPI_Sendrecv
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, int sendcount,
                 MPI_Datatype sendtype, int dst, int sendtag, void*, int recvcount,
                 MPI_Datatype recvtype, int src, int recvtag, MPI_Comm, MPI_Status*)
{
    int _send_size = mpi_type_size(sendtype);
    int _recv_size = mpi_type_size(recvtype);
    if(_send_size == 0 || _recv_size == 0) return;

    if(get_use_perfetto())
    {
        write_perfetto_counter_track<mpi_send>(sendcount * _send_size);
        write_perfetto_counter_track<mpi_recv>(recvcount * _recv_size);
    }

    {
        cache_comm_data_events<mpi_send>(0, sendcount * _send_size);
        cache_comm_data_events<mpi_recv>(0, recvcount * _recv_size);
    }

    if(rocprofsys::get_use_timemory())
    {
        auto      _name = std::string_view{ _data.tool_id };
        tracker_t _t{ _name };
        add(_t, sendcount * _send_size + recvcount * _recv_size);
        {
            tracker_t _b{ fmt::format("{}/send", _name) };
            add(_b, sendcount * _send_size);
            tracker_t _c{ fmt::format("{}/send={}", _name, dst) };
            add(_b, sendcount * _send_size);
            add(fmt::format("{}/send={}/tag={}", _name, sendtag), sendcount * _send_size);
            add(fmt::format("{}/send={}/tag={}", _name, dst, sendtag),
                sendcount * _send_size);
        }
        {
            tracker_t _b{ fmt::format("{}/recv", _name) };
            add(_b, recvcount * _recv_size);
            tracker_t _c{ fmt::format("{}/recv={}", _name, src) };
            add(_b, recvcount * _recv_size);
            add(fmt::format("{}/recv={}/tag={}", _name, recvtag), recvcount * _recv_size);
            add(fmt::format("{}/recv={}/tag={}", _name, src, recvtag),
                recvcount * _recv_size);
        }
    }
}

// MPI_Gather
// MPI_Scatter
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, int sendcount,
                 MPI_Datatype sendtype, void*, int recvcount, MPI_Datatype recvtype,
                 int root, MPI_Comm)
{
    int _send_size = mpi_type_size(sendtype);
    int _recv_size = mpi_type_size(recvtype);
    if(_send_size == 0 || _recv_size == 0) return;

    if(get_use_perfetto())
    {
        write_perfetto_counter_track<mpi_send>(sendcount * _send_size);
        write_perfetto_counter_track<mpi_recv>(recvcount * _recv_size);
    }

    {
        cache_comm_data_events<mpi_send>(0, sendcount * _send_size);
        cache_comm_data_events<mpi_recv>(0, recvcount * _recv_size);
    }

    if(rocprofsys::get_use_timemory())
    {
        auto      _name = std::string_view{ _data.tool_id };
        tracker_t _t{ _name };
        add(_t, sendcount * _send_size + recvcount * _recv_size);
        tracker_t _r(fmt::format("{}/root={}", _name, root));
        add(_r, sendcount * _send_size + recvcount * _recv_size);
        add(fmt::format("{}/root={}/send", _name, root), sendcount * _send_size);
        add(fmt::format("{}/root={}/recv", _name, root), recvcount * _recv_size);
    }
}

// MPI_Alltoall
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, int sendcount,
                 MPI_Datatype sendtype, void*, int recvcount, MPI_Datatype recvtype,
                 MPI_Comm)
{
    int _send_size = mpi_type_size(sendtype);
    int _recv_size = mpi_type_size(recvtype);
    if(_send_size == 0 || _recv_size == 0) return;

    if(get_use_perfetto())
    {
        write_perfetto_counter_track<mpi_send>(sendcount * _send_size);
        write_perfetto_counter_track<mpi_recv>(recvcount * _recv_size);
    }

    {
        cache_comm_data_events<mpi_send>(0, sendcount * _send_size);
        cache_comm_data_events<mpi_recv>(0, recvcount * _recv_size);
    }

    if(rocprofsys::get_use_timemory())
    {
        auto      _name = std::string_view{ _data.tool_id };
        tracker_t _t{ _name };
        add(_t, sendcount * _send_size + recvcount * _recv_size);
        add(fmt::format("{}/send", _name), sendcount * _send_size);
        add(fmt::format("{}/recv", _name), recvcount * _recv_size);
    }
}
#endif

// UCX communication tracking implementations

// ucp_tag_send_nbx: (void* ep, const void* buffer, size_t count, uint64_t tag, const
// void* param)
void
comm_data::audit(const gotcha_data& _data, audit::incoming, void*, const void*,
                 size_t count, uint64_t tag, const void*)
{
    if(count == 0) return;

    if(get_use_perfetto()) write_perfetto_counter_track<ucx_send>(count);

    {
        cache_comm_data_events<ucx_send>(0, count);
    }

    if(rocprofsys::get_use_timemory())
    {
        auto      _name = std::string_view{ _data.tool_id };
        tracker_t _t{ _name };
        add(_t, count);
        add(fmt::format("{}/tag={}", _name, tag), count);
    }
}

// ucp_tag_recv_nbx: (void* worker, void* buffer, size_t count, uint64_t tag, uint64_t
// tag_mask, const void* param)
void
comm_data::audit(const gotcha_data& _data, audit::incoming, void*, void*, size_t count,
                 uint64_t tag, uint64_t tag_mask, const void*)
{
    if(count == 0) return;

    if(get_use_perfetto()) write_perfetto_counter_track<ucx_recv>(count);

    {
        cache_comm_data_events<ucx_recv>(0, count);
    }

    if(rocprofsys::get_use_timemory())
    {
        auto      _name = std::string_view{ _data.tool_id };
        tracker_t _t{ _name };
        add(_t, count);
        add(fmt::format("{}/tag={}", _name, tag), count);
        add(fmt::format("{}/tag={}/tag_mask={}", _name, tag, tag_mask), count);
    }
}

// ucp_put_nbx: (void* ep, const void* buffer, size_t count, uint64_t remote_addr, void*
// rkey, const void* param)
void
comm_data::audit(const gotcha_data& _data, audit::incoming, void*, const void*,
                 size_t count, uint64_t remote_addr, void*, const void*)
{
    if(count == 0) return;

    if(get_use_perfetto()) write_perfetto_counter_track<ucx_send>(count);

    {
        cache_comm_data_events<ucx_send>(0, count);
    }

    if(rocprofsys::get_use_timemory())
    {
        auto      _name = std::string_view{ _data.tool_id };
        tracker_t _t{ _name };
        add(_t, count);
        add(fmt::format("{}/remote_addr={}", _name, remote_addr), count);
    }
}

// ucp_get_nbx: (void* ep, void* buffer, size_t count, uint64_t remote_addr, void* rkey,
// const void* param)
void
comm_data::audit(const gotcha_data& _data, audit::incoming, void*, void*, size_t count,
                 uint64_t remote_addr, void*, const void*)
{
    if(count == 0) return;

    if(get_use_perfetto()) write_perfetto_counter_track<ucx_recv>(count);

    {
        cache_comm_data_events<ucx_recv>(0, count);
    }

    if(rocprofsys::get_use_timemory())
    {
        auto      _name = std::string_view{ _data.tool_id };
        tracker_t _t{ _name };
        add(_t, count);
        add(fmt::format("{}/remote_addr={}", _name, remote_addr), count);
    }
}

// ucp_am_send_nbx: (void* ep, unsigned id, const void* header, size_t header_length,
// const void* buffer, size_t count, const void* param)
void
comm_data::audit(const gotcha_data& _data, audit::incoming, void*, unsigned id,
                 const void*, size_t header_length, const void*, size_t count,
                 const void*)
{
    if(count == 0 && header_length == 0) return;

    size_t total_size = header_length + count;
    if(get_use_perfetto()) write_perfetto_counter_track<ucx_send>(total_size);

    {
        cache_comm_data_events<ucx_send>(0, total_size);
    }

    if(rocprofsys::get_use_timemory())
    {
        auto      _name = std::string_view{ _data.tool_id };
        tracker_t _t{ _name };
        add(_t, total_size);
        add(fmt::format("{}/am_id={}", _name, id), total_size);
    }
}

// ucp_stream_send_nbx: (void* ep, const void* buffer, size_t count, const void* param)
void
comm_data::audit(const gotcha_data& _data, audit::incoming, void*, const void*,
                 size_t             count, const void*)
{
    if(count == 0) return;

    if(get_use_perfetto()) write_perfetto_counter_track<ucx_send>(count);

    {
        cache_comm_data_events<ucx_send>(0, count);
    }

    if(rocprofsys::get_use_timemory())
    {
        auto      _name = std::string_view{ _data.tool_id };
        tracker_t _t{ _name };
        add(_t, count);
    }
}

// ucp_stream_recv_nbx: (void* ep, void* buffer, size_t count, size_t* length, const void*
// param)
void
comm_data::audit(const gotcha_data& _data, audit::incoming, void*, void*, size_t count,
                 size_t*, const void*)
{
    if(count == 0) return;

    if(get_use_perfetto()) write_perfetto_counter_track<ucx_recv>(count);

    {
        cache_comm_data_events<ucx_recv>(0, count);
    }

    if(rocprofsys::get_use_timemory())
    {
        auto      _name = std::string_view{ _data.tool_id };
        tracker_t _t{ _name };
        add(_t, count);
    }
}

// Legacy: ucp_tag_send_nb/nbx - send with tag matching (for old-style wrappers)
void
comm_data::audit(const gotcha_data& _data, audit::incoming, void*, size_t count, void*,
                 void*, void*)
{
    if(count == 0) return;

    if(get_use_perfetto()) write_perfetto_counter_track<ucx_send>(count);

    {
        cache_comm_data_events<ucx_send>(0, count);
    }

    if(rocprofsys::get_use_timemory())
    {
        auto      _name = std::string_view{ _data.tool_id };
        tracker_t _t{ _name };
        add(_t, count);
    }
}

// Legacy: ucp_tag_recv_nb/nbx - receive with tag matching (for old-style wrappers)
void
comm_data::audit(const gotcha_data& _data, audit::incoming, void*, size_t count, void*,
                 void*, void*, void*, void*)
{
    if(count == 0) return;

    if(get_use_perfetto()) write_perfetto_counter_track<ucx_recv>(count);

    {
        cache_comm_data_events<ucx_recv>(0, count);
    }

    if(rocprofsys::get_use_timemory())
    {
        auto      _name = std::string_view{ _data.tool_id };
        tracker_t _t{ _name };
        add(_t, count);
    }
}

// ucp_put/get operations - RMA
void
comm_data::audit(const gotcha_data& _data, audit::incoming, void*, size_t length,
                 uint64_t, void*, void*)
{
    if(length == 0) return;

    bool is_put = _data.tool_id.find("ucp_put") != std::string::npos;

    if(get_use_perfetto())
    {
        if(is_put)
            write_perfetto_counter_track<ucx_send>(length);
        else
            write_perfetto_counter_track<ucx_recv>(length);
    }

    {
        if(is_put)
            cache_comm_data_events<ucx_send>(0, length);
        else
            cache_comm_data_events<ucx_recv>(0, length);
    }

    if(rocprofsys::get_use_timemory())
    {
        auto      _name = std::string_view{ _data.tool_id };
        tracker_t _t{ _name };
        add(_t, length);
    }
}

// ucp_am_send_nb/nbx - active message send
void
comm_data::audit(const gotcha_data& _data, audit::incoming, void*, unsigned, void*,
                 size_t header_length, void*, size_t length, unsigned, void*)
{
    size_t total_length = header_length + length;
    if(total_length == 0) return;

    if(get_use_perfetto()) write_perfetto_counter_track<ucx_send>(total_length);

    {
        cache_comm_data_events<ucx_send>(0, total_length);
    }

    if(rocprofsys::get_use_timemory())
    {
        auto      _name = std::string_view{ _data.tool_id };
        tracker_t _t{ _name };
        add(_t, total_length);
    }
}

// ucp_stream_send/recv operations
void
comm_data::audit(const gotcha_data& _data, audit::incoming, void*, void*, size_t count,
                 void*, unsigned, void*)
{
    if(count == 0) return;

    bool is_send = _data.tool_id.find("send") != std::string::npos;

    if(get_use_perfetto())
    {
        if(is_send)
            write_perfetto_counter_track<ucx_send>(count);
        else
            write_perfetto_counter_track<ucx_recv>(count);
    }

    {
        if(is_send)
            cache_comm_data_events<ucx_send>(0, count);
        else
            cache_comm_data_events<ucx_recv>(0, count);
    }

    if(rocprofsys::get_use_timemory())
    {
        auto      _name = std::string_view{ _data.tool_id };
        tracker_t _t{ _name };
        add(_t, count);
    }
}

#if defined(ROCPROFSYS_USE_RCCL)
// Kept for reference, but now gathered throught the SDK callbacks.

// ncclReduce
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, const void*,
                 size_t count, ncclDataType_t datatype, ncclRedOp_t, int root, ncclComm_t,
                 hipStream_t)
{
    int _size = rccl_type_size(datatype);
    if(_size <= 0) return;

    if(get_use_perfetto()) write_perfetto_counter_track<rccl_recv>(count * _size);

    if(get_use_rocpd()) rocpd_process_cpu_usage_events<rccl_recv>(0, count * _size);

    if(rocprofsys::get_use_timemory())
    {
        auto      _name = std::string_view{ _data.tool_id };
        tracker_t _t{ _name };
        add(_t, count * _size);
        add(fmt::format("{}/root={}", _name, root), count * _size);
    }
}

// ncclSend
// ncclGather
// ncclBcast
// ncclRecv
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, size_t count,
                 ncclDataType_t datatype, int peer, ncclComm_t, hipStream_t)
{
    int _size = rccl_type_size(datatype);
    if(_size <= 0) return;

    static auto _send_types = std::unordered_set<std::string>{ "ncclSend", "ncclBcast" };
    static auto _recv_types = std::unordered_set<std::string>{ "ncclGather", "ncclRecv" };

    if(_send_types.count(_data.tool_id) > 0)
    {
        if(get_use_perfetto()) write_perfetto_counter_track<rccl_send>(count * _size);
        if(get_use_rocpd()) rocpd_process_cpu_usage_events<rccl_send>(0, count * _size);
    }
    else if(_recv_types.count(_data.tool_id) > 0)
    {
        if(get_use_perfetto()) write_perfetto_counter_track<rccl_recv>(count * _size);
        if(get_use_rocpd()) rocpd_process_cpu_usage_events<rccl_recv>(0, count * _size);
    }
    else
    {
        ROCPROFSYS_CI_THROW(true, "RCCL function not handled: %s", _data.tool_id.c_str());
    }

    if(rocprofsys::get_use_timemory())
    {
        auto        _name  = std::string_view{ _data.tool_id };
        std::string _label = "root";
        if(_name.find("Send") != std::string::npos) _label = "peer";

        tracker_t _t{ _name };
        add(_t, count * _size);
        add(fmt::format("{}/{}={}", _name, _label, peer), count * _size);
    }
}

// ncclBroadcast
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, const void*,
                 size_t count, ncclDataType_t datatype, int root, ncclComm_t, hipStream_t)
{
    int _size = rccl_type_size(datatype);
    if(_size <= 0) return;

    if(get_use_perfetto()) write_perfetto_counter_track<rccl_send>(count * _size);
    if(get_use_rocpd()) rocpd_process_cpu_usage_events<rccl_send>(0, count * _size);

    if(rocprofsys::get_use_timemory())
    {
        auto      _name = std::string_view{ _data.tool_id };
        tracker_t _t{ _name };
        add(_t, count * _size);
        add(fmt::format("{}/root={}", _data.tool_id, root), count * _size);
    }
}

// ncclAllReduce
// ncclReduceScatter
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, const void*,
                 size_t count, ncclDataType_t datatype, ncclRedOp_t, ncclComm_t,
                 hipStream_t)
{
    int _size = rccl_type_size(datatype);
    if(_size <= 0) return;

    static auto _recv_types = std::unordered_set<std::string>{ "ncclAllReduce" };
    static auto _send_types = std::unordered_set<std::string>{ "ncclReduceScatter" };

    if(_send_types.count(_data.tool_id) > 0)
    {
        if(get_use_perfetto()) write_perfetto_counter_track<rccl_send>(count * _size);
        if(get_use_rocpd()) rocpd_process_cpu_usage_events<rccl_send>(0, count * _size);
    }
    else if(_recv_types.count(_data.tool_id) > 0)
    {
        if(get_use_perfetto()) write_perfetto_counter_track<rccl_recv>(count * _size);
        if(get_use_rocpd()) rocpd_process_cpu_usage_events<rccl_recv>(0, count * _size);
    }
    else
    {
        ROCPROFSYS_CI_THROW(true, "RCCL function not handled: %s", _data.tool_id.c_str());
    }

    if(rocprofsys::get_use_timemory()) add(_data, count * _size);
}

// ncclAllGather
// ncclAllToAll
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, const void*,
                 size_t count, ncclDataType_t datatype, ncclComm_t, hipStream_t)
{
    int _size = rccl_type_size(datatype);
    if(_size <= 0) return;

    if(get_use_perfetto()) write_perfetto_counter_track<rccl_recv>(count * _size);
    if(get_use_rocpd()) rocpd_process_cpu_usage_events<rccl_recv>(0, count * _size);
    if(rocprofsys::get_use_timemory()) add(_data, count * _size);
}
#endif
}  // namespace component
}  // namespace rocprofsys

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<float, tim::project::rocprofsys>), true, float)

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(comm_data, false, void)
