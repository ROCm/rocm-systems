#pragma once

/**
 * @file perfetto_proto_writer.hpp
 * @brief Demo-side Perfetto .pftrace emitter used by the bundled examples.
 *
 * NOT part of the cpu_snap library install surface; consumers who want a
 * different output format (JSON, custom binary, Speedscope, ...) write
 * their own analogous helper.
 *
 * Produces a `Trace { repeated TracePacket }` binary that opens in
 * https://ui.perfetto.dev/ with a per-process track, per-thread tracks
 * and nested slice events (flame-graph view under each thread row).
 *
 * @note Only one sequence (sequence_id = 1).
 *
 * Header-only. The protobuf encoder lives in pb_buf.hpp; field-number
 * constants live in pb_fields.hpp grouped per submessage.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cpu_snap/processing/timeline/slice_emitter.hpp"
#include "pb_buf.hpp"
#include "pb_fields.hpp"

namespace cs_examples::perfetto
{

/**
 * Streaming writer that emits a Perfetto-compatible @c .pftrace file
 * containing per-thread track descriptors and symbolised slice events.
 *
 * @note Not thread-safe. The intended pattern is single-producer:
 *       construct, call set_process / add_thread / write_slice in any
 *       order, then finalize() before the destructor runs.
 */
class perfetto_proto_writer
{
public:
    /**
     * Open @p path for binary writing.
     *
     * @param path Output @c .pftrace path. Existing file is truncated.
     * @throws std::runtime_error When the file cannot be opened for writing.
     */
    explicit perfetto_proto_writer(const std::string& path)
    : m_ofs(path, std::ios::binary)
    {
        if(!m_ofs)
        {
            throw std::runtime_error("perfetto_proto_writer: cannot open " + path);
        }
    }

    ~perfetto_proto_writer()
    {
        if(m_ofs.is_open())
        {
            m_ofs.close();
        }
    }

    perfetto_proto_writer(const perfetto_proto_writer&)                = delete;
    perfetto_proto_writer& operator=(const perfetto_proto_writer&)     = delete;
    perfetto_proto_writer(perfetto_proto_writer&&) noexcept            = delete;
    perfetto_proto_writer& operator=(perfetto_proto_writer&&) noexcept = delete;

    /**
     * Emit a ProcessDescriptor track. Must be called before any
     * add_thread.
     *
     * @param pid  Linux pid of the recorded process.
     * @param name Process name shown in the Perfetto UI.
     */
    void set_process(int pid, const std::string& name)
    {
        namespace td_f = pb_fields::track_descriptor;
        namespace pd_f = pb_fields::process_descriptor;

        m_pid          = pid;
        m_process_uuid = m_next_uuid++;

        pb::buf td;
        td.tag_varint(td_f::k_uuid, m_process_uuid);
        td.tag_string(td_f::k_name, name);
        {
            pb::buf proc;
            proc.tag_int32(pd_f::k_pid, pid);
            proc.tag_string(pd_f::k_name, name);
            td.tag_submsg(td_f::k_process, proc);
        }
        emit_track_descriptor(td);
    }

    /**
     * Emit a ThreadDescriptor track parented under the process.
     *
     * @param tid  Linux tid of the thread.
     * @param name Thread name shown in the Perfetto UI.
     * @return     The internal UUID assigned to the thread track.
     */
    std::uint64_t add_thread(int tid, const std::string& name)
    {
        namespace td_f = pb_fields::track_descriptor;
        namespace th_f = pb_fields::thread_descriptor;

        const auto uuid = m_next_uuid++;
        pb::buf    td;
        td.tag_varint(td_f::k_uuid, uuid);
        td.tag_varint(td_f::k_parent_uuid, m_process_uuid);
        td.tag_string(td_f::k_name, name);
        {
            pb::buf thread;
            thread.tag_int32(th_f::k_pid, m_pid);
            thread.tag_int32(th_f::k_tid, tid);
            thread.tag_string(th_f::k_name, name);
            td.tag_submsg(td_f::k_thread, thread);
        }
        emit_track_descriptor(td);
        m_thread_track_uuid[tid] = uuid;
        return uuid;
    }

    /**
     * Buffer one completed slice span produced by
     * cpu_snap::processing::timeline::slice_emitter. The writer is a pure sink: it
     * records the span here and reconstructs the Perfetto begin/end
     * TrackEvent ordering at finalize().
     *
     * @param slice Completed slice span. @c slice.name is borrowed and
     *              copied into an owned record; the view need not outlive
     *              this call. Spans whose @c slice.tid has no matching
     *              add_thread() are ignored.
     */
    void write_slice(const cpu_snap::processing::timeline::emitted_slice& slice)
    {
        auto track_it = m_thread_track_uuid.find(static_cast<int>(slice.tid));
        if(track_it == m_thread_track_uuid.end())
        {
            return;
        }
        m_slices.push_back({ track_it->second, std::string{ slice.name },
                             slice.start_ts, slice.end_ts, slice.depth });
    }

    /**
     * Reconstruct begin/end TrackEvents from the buffered slice spans
     * and flush all buffered bytes to the underlying file.
     *
     * Each span yields an OPEN(start_ts) and CLOSE(end_ts) boundary.
     * Boundaries are grouped per track and replayed in an order that
     * reproduces the original nesting: ascending timestamp, CLOSE before
     * OPEN at equal timestamps, inner closes (larger start_ts) first,
     * outer opens (larger end_ts) first.
     */
    void finalize()
    {
        flush_buffered_slices();
        m_ofs.flush();
    }

private:
    /** A completed slice span buffered until finalize(). */
    struct buffered_slice
    {
        std::uint64_t track_uuid;
        std::string   name;
        std::uint64_t start_ts;
        std::uint64_t end_ts;
        std::uint32_t depth;
    };

    /** One slice boundary (begin or end) replayed at finalize(). */
    struct boundary_event
    {
        std::uint64_t    ts;
        bool             open;
        std::uint32_t    depth;
        std::string_view name;  // valid only for OPEN; views m_slices entry
    };

    void flush_buffered_slices()
    {
        std::unordered_map<std::uint64_t, std::vector<boundary_event>> per_track;
        for(const auto& slc : m_slices)
        {
            auto& events = per_track[slc.track_uuid];
            events.push_back({ slc.start_ts, /*open=*/true, slc.depth, slc.name });
            events.push_back({ slc.end_ts, /*open=*/false, slc.depth, {} });
        }

        for(auto& [track_uuid, events] : per_track)
        {
            std::sort(events.begin(), events.end(), boundary_less);
            for(const auto& ev : events)
            {
                emit_slice_event_named(track_uuid, ev.ts, ev.open, ev.name);
            }
        }
    }

    /**
     * Order two boundary events so replay reproduces slice nesting:
     * ascending timestamp; at equal ts a CLOSE precedes an OPEN; among
     * CLOSEs the inner (deeper) closes first; among OPENs the outer
     * (shallower) opens first. Depth is the tie-break because timestamps
     * alone do not encode nesting: a whole sub-stack captured in one
     * sample shares identical start and end, so without depth the open
     * order is ambiguous and the flame mis-nests.
     */
    static bool boundary_less(const boundary_event& lhs, const boundary_event& rhs)
    {
        if(lhs.ts != rhs.ts)
        {
            return lhs.ts < rhs.ts;
        }
        if(lhs.open != rhs.open)
        {
            return !lhs.open;  // CLOSE (open=false) before OPEN
        }
        if(!lhs.open)
        {
            return lhs.depth > rhs.depth;  // inner (deeper) closes first
        }
        return lhs.depth < rhs.depth;  // outer (shallower) opens first
    }

    void emit_slice_event_named(std::uint64_t track_uuid, std::uint64_t ts_ns,
                                bool begin, std::string_view name)
    {
        namespace tp_f = pb_fields::trace_packet;
        namespace te_f = pb_fields::track_event;

        pb::buf te;
        te.tag_varint(te_f::k_type,
                      begin ? te_f::k_type_slice_begin : te_f::k_type_slice_end);
        te.tag_varint(te_f::k_track_uuid, track_uuid);
        if(begin)
        {
            te.tag_string(te_f::k_name, std::string{ name });
        }
        pb::buf packet;
        packet.tag_varint(tp_f::k_timestamp, ts_ns);
        packet.tag_varint(tp_f::k_trusted_seq_id, m_seq_id);
        packet.tag_submsg(tp_f::k_track_event, te);
        emit_trace_packet(packet);
    }

    void emit_track_descriptor(const pb::buf& td)
    {
        namespace tp_f = pb_fields::trace_packet;

        pb::buf packet;
        packet.tag_varint(tp_f::k_trusted_seq_id, m_seq_id);
        packet.tag_submsg(tp_f::k_track_descriptor, td);
        emit_trace_packet(packet);
    }

    void emit_trace_packet(const pb::buf& packet)
    {
        namespace tr_f = pb_fields::trace;

        // Wrap each TracePacket as field 1 of the top-level Trace message.
        pb::buf wrapper;
        wrapper.tag_submsg(tr_f::k_packet, packet);
        m_ofs.write(wrapper.bytes().data(), static_cast<std::streamsize>(wrapper.size()));
    }

    std::ofstream                          m_ofs;
    int                                    m_pid          = 0;
    std::uint32_t                          m_seq_id       = 1;
    std::uint64_t                          m_process_uuid = 0;
    std::uint64_t                          m_next_uuid    = 100;
    std::unordered_map<int, std::uint64_t> m_thread_track_uuid;
    std::vector<buffered_slice>            m_slices;
};

}  // namespace cs_examples::perfetto
