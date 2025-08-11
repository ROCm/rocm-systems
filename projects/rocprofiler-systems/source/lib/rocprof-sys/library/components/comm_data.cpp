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
#include "core/perfetto.hpp"
#include "library/tracing.hpp"

#include <timemory/backends/mpi.hpp>
#include <timemory/manager.hpp>
#include <timemory/units.hpp>
#include <timemory/utility/locking.hpp>

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
                std::string _label = (_idx > 0)
                                         ? JOIN(" ", Tp::label, JOIN("", '[', _idx, ']'))
                                         : Tp::label;
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
    comm_data_tracker_t::description()  = "Tracks MPI/RCCL communication data sizes";
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

    if(!rocprofsys::get_use_timemory()) return;
    auto      _name = std::string_view{ _data.tool_id };
    tracker_t _a{ _name };
    add(_a, count * _size);
    tracker_t _b{ JOIN('/', _name, JOIN('=', "dst", dst)) };
    add(_b, count * _size);
    add(JOIN('/', _name, JOIN('=', "dst", dst), JOIN('=', "tag", tag)), count * _size);
}

// MPI_Recv
void
comm_data::audit(const gotcha_data& _data, audit::incoming, void*, int count,
                 MPI_Datatype datatype, int dst, int tag, MPI_Comm, MPI_Status*)
{
    int _size = mpi_type_size(datatype);
    if(_size == 0) return;

    write_perfetto_counter_track<mpi_recv>(count * _size);

    if(!rocprofsys::get_use_timemory()) return;
    auto      _name = std::string_view{ _data.tool_id };
    tracker_t _a{ _name };
    add(_a, count * _size);
    tracker_t _b{ JOIN('/', _name, JOIN('=', "dst", dst)) };
    add(_b, count * _size);
    add(JOIN('/', _name, JOIN('=', "dst", dst), JOIN('=', "tag", tag)), count * _size);
}

// MPI_Isend
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, int count,
                 MPI_Datatype datatype, int dst, int tag, MPI_Comm, MPI_Request*)
{
    int _size = mpi_type_size(datatype);
    if(_size == 0) return;

    write_perfetto_counter_track<mpi_send>(count * _size);

    if(!rocprofsys::get_use_timemory()) return;
    auto      _name = std::string_view{ _data.tool_id };
    tracker_t _a{ _name };
    add(_a, count * _size);
    tracker_t _b{ JOIN('/', _name, JOIN('=', "dst", dst)) };
    add(_b, count * _size);
    add(JOIN('/', _name, JOIN('=', "dst", dst), JOIN('=', "tag", tag)), count * _size);
}

// MPI_Irecv
void
comm_data::audit(const gotcha_data& _data, audit::incoming, void*, int count,
                 MPI_Datatype datatype, int dst, int tag, MPI_Comm, MPI_Request*)
{
    int _size = mpi_type_size(datatype);
    if(_size == 0) return;

    write_perfetto_counter_track<mpi_recv>(count * _size);

    if(!rocprofsys::get_use_timemory()) return;
    auto      _name = std::string_view{ _data.tool_id };
    tracker_t _a{ _name };
    add(_a, count * _size);
    tracker_t _b{ JOIN('/', _name, JOIN('=', "dst", dst)) };
    add(_b, count * _size);
    add(JOIN('/', _name, JOIN('=', "dst", dst), JOIN('=', "tag", tag)), count * _size);
}

// MPI_Bcast
void
comm_data::audit(const gotcha_data& _data, audit::incoming, void*, int count,
                 MPI_Datatype datatype, int root, MPI_Comm)
{
    int _size = mpi_type_size(datatype);
    if(_size == 0) return;

    write_perfetto_counter_track<mpi_send>(count * _size);

    if(!rocprofsys::get_use_timemory()) return;
    auto      _name = std::string_view{ _data.tool_id };
    tracker_t _t{ _name };
    add(_t, count * _size);
    add(JOIN('/', _name, JOIN('=', "root", root)), count * _size);
}

// MPI_Allreduce
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, void*, int count,
                 MPI_Datatype datatype, MPI_Op, MPI_Comm)
{
    int _size = mpi_type_size(datatype);
    if(_size == 0) return;

    write_perfetto_counter_track<mpi_recv>(count * _size);
    write_perfetto_counter_track<mpi_send>(count * _size);

    if(!rocprofsys::get_use_timemory()) return;
    add(_data, count * _size);
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

    write_perfetto_counter_track<mpi_send>(sendcount * _send_size);
    write_perfetto_counter_track<mpi_recv>(recvcount * _recv_size);

    if(!rocprofsys::get_use_timemory()) return;
    auto      _name = std::string_view{ _data.tool_id };
    tracker_t _t{ _name };
    add(_t, sendcount * _send_size + recvcount * _recv_size);
    {
        tracker_t _b{ JOIN('/', _name, "send") };
        add(_b, sendcount * _send_size);
        tracker_t _c{ JOIN('/', _name, JOIN('=', "send", dst)) };
        add(_b, sendcount * _send_size);
        add(JOIN('/', _name, "send", JOIN('=', "tag", sendtag)), sendcount * _send_size);
        add(JOIN('/', _name, JOIN('=', "send", dst), JOIN('=', "tag", sendtag)),
            sendcount * _send_size);
    }
    {
        tracker_t _b{ JOIN('/', _name, "recv") };
        add(_b, recvcount * _recv_size);
        tracker_t _c{ JOIN('/', _name, JOIN('=', "recv", src)) };
        add(_b, recvcount * _recv_size);
        add(JOIN('/', _name, "recv", JOIN('=', "tag", recvtag)), recvcount * _recv_size);
        add(JOIN('/', _name, JOIN('=', "recv", src), JOIN('=', "tag", recvtag)),
            recvcount * _recv_size);
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

    write_perfetto_counter_track<mpi_send>(sendcount * _send_size);
    write_perfetto_counter_track<mpi_recv>(recvcount * _recv_size);

    if(!rocprofsys::get_use_timemory()) return;
    auto      _name = std::string_view{ _data.tool_id };
    tracker_t _t{ _name };
    add(_t, sendcount * _send_size + recvcount * _recv_size);
    tracker_t _r(JOIN('/', _name, JOIN('=', "root", root)));
    add(_r, sendcount * _send_size + recvcount * _recv_size);
    add(JOIN('/', _name, JOIN('=', "root", root), "send"), sendcount * _send_size);
    add(JOIN('/', _name, JOIN('=', "root", root), "recv"), recvcount * _recv_size);
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

    write_perfetto_counter_track<mpi_send>(sendcount * _send_size);
    write_perfetto_counter_track<mpi_recv>(recvcount * _recv_size);

    if(!rocprofsys::get_use_timemory()) return;
    auto      _name = std::string_view{ _data.tool_id };
    tracker_t _t{ _name };
    add(_t, sendcount * _send_size + recvcount * _recv_size);
    add(JOIN('/', _name, "send"), sendcount * _send_size);
    add(JOIN('/', _name, "recv"), recvcount * _recv_size);
}
#endif

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

    write_perfetto_counter_track<rccl_recv>(count * _size);

    if(!rocprofsys::get_use_timemory()) return;
    auto      _name = std::string_view{ _data.tool_id };
    tracker_t _t{ _name };
    add(_t, count * _size);
    add(JOIN('/', _name, JOIN('=', "root", root)), count * _size);
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
        write_perfetto_counter_track<rccl_send>(count * _size);
    }
    else if(_recv_types.count(_data.tool_id) > 0)
    {
        write_perfetto_counter_track<rccl_recv>(count * _size);
    }
    else
    {
        ROCPROFSYS_CI_THROW(true, "RCCL function not handled: %s", _data.tool_id.c_str());
    }

    write_perfetto_counter_track<rccl_recv>(count * _size);

    if(!rocprofsys::get_use_timemory()) return;
    auto        _name  = std::string_view{ _data.tool_id };
    std::string _label = "root";
    if(_name.find("Send") != std::string::npos) _label = "peer";

    tracker_t _t{ _name };
    add(_t, count * _size);
    add(JOIN('/', _name, JOIN('=', _label, peer)), count * _size);
}

// ncclBroadcast
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, const void*,
                 size_t count, ncclDataType_t datatype, int root, ncclComm_t, hipStream_t)
{
    int _size = rccl_type_size(datatype);
    if(_size <= 0) return;

    write_perfetto_counter_track<rccl_send>(count * _size);

    if(!rocprofsys::get_use_timemory()) return;
    auto      _name = std::string_view{ _data.tool_id };
    tracker_t _t{ _name };
    add(_t, count * _size);
    add(JOIN('/', _data.tool_id, JOIN('=', "root", root)), count * _size);
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
        write_perfetto_counter_track<rccl_send>(count * _size);
    }
    else if(_recv_types.count(_data.tool_id) > 0)
    {
        write_perfetto_counter_track<rccl_recv>(count * _size);
    }
    else
    {
        ROCPROFSYS_CI_THROW(true, "RCCL function not handled: %s", _data.tool_id.c_str());
    }

    if(!rocprofsys::get_use_timemory()) return;
    add(_data, count * _size);
}

// ncclAllGather
// ncclAllToAll
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, const void*,
                 size_t count, ncclDataType_t datatype, ncclComm_t, hipStream_t)
{
    int _size = rccl_type_size(datatype);
    if(_size <= 0) return;

    write_perfetto_counter_track<rccl_recv>(count * _size);

    if(!rocprofsys::get_use_timemory()) return;
    add(_data, count * _size);
}
#endif
}  // namespace component
}  // namespace rocprofsys

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<float, tim::project::rocprofsys>), true, float)

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(comm_data, false, void)
