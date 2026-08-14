// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "stitch/stitch.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

using rocprof_trace_decoder::codeobj::decode_marker_value;
using rocprof_trace_decoder::codeobj::Funcmap;
using rocprof_trace_decoder::codeobj::FuncmapEntryKind;
using rocprof_trace_decoder::codeobj::MarkerValue;

namespace
{
constexpr size_t MaxPendingShaderdata = 1u << 20;
constexpr auto MarkerRecord = ROCPROFILER_THREAD_TRACE_DECODER_RECORD_MARKER;

uint32_t waveKey(uint8_t cu, uint8_t simd, uint8_t wave_id)
{
    return (uint32_t{cu} << 16) | (uint32_t{simd} << 8) | uint32_t{wave_id};
}

uint32_t publicMarkerKind(FuncmapEntryKind kind)
{
    switch (kind)
    {
        case FuncmapEntryKind::Function: return ROCPROFILER_THREAD_TRACE_DECODER_MARKER_KIND_FUNCTION;
        case FuncmapEntryKind::UserScope: return ROCPROFILER_THREAD_TRACE_DECODER_MARKER_KIND_USER_SCOPE;
        case FuncmapEntryKind::Point: return ROCPROFILER_THREAD_TRACE_DECODER_MARKER_KIND_POINT;
        case FuncmapEntryKind::Kernel: break;
    }
    return ROCPROFILER_THREAD_TRACE_DECODER_MARKER_KIND_UNKNOWN;
}

bool isScope(FuncmapEntryKind kind)
{
    return kind == FuncmapEntryKind::Function || kind == FuncmapEntryKind::UserScope;
}
} // namespace

class MarkerStream
{
public:
    MarkerStream(RecordEmitter& records, ICodeServicer& service) : records(records), service(service) {}

    void waveStart(uint8_t cu, uint8_t simd, uint8_t wave_id, pcinfo_t kernel_entry)
    {
        auto& generation = waves[waveKey(cu, simd, wave_id)];
        if (!generation.pending.empty()) unresolved_generations.push_back(std::move(generation));
        generation = Generation{.kernel_entry = kernel_entry};
    }

    void waveEnd(uint8_t cu, uint8_t simd, uint8_t wave_id, pcinfo_t kernel_entry)
    {
        auto* generation = findGeneration(cu, simd, wave_id);
        if (!generation) return;
        resolve(*generation, kernel_entry);
    }

    void resolveAll(CSRegisterHandler& registers)
    {
        for (auto& [_, generation] : waves)
            if (generation.kernel_entry.code_object_id == 0 && generation.kernel_entry.address != 0)
                resolve(generation, registers.get_wave_start_delayed(generation.kernel_entry.address));
        for (auto& generation : unresolved_generations)
            if (generation.kernel_entry.address != 0)
                resolve(generation, registers.get_wave_start_delayed(generation.kernel_entry.address));
        unresolved_generations.erase(
            std::remove_if(
                unresolved_generations.begin(),
                unresolved_generations.end(),
                [](const auto& generation) { return generation.pending.empty(); }
            ),
            unresolved_generations.end()
        );
    }

    void shaderdata(const att_shader_data_t& record)
    {
        if ((record.flags & (1u << ROCPROFILER_THREAD_TRACE_DECODER_SHADERDATA_FLAGS_PRIV)) != 0) return;

        auto* generation = findGeneration(record.cu, record.simd, record.wave_id);
        if (!generation || !generation->reliable) return;
        if (generation->kernel_entry.code_object_id == 0)
        {
            if (pending_count >= MaxPendingShaderdata)
            {
                pending_count -= generation->pending.size();
                generation->pending.clear();
                generation->reliable = false;
                return;
            }
            generation->pending.push_back(record);
            ++pending_count;
            return;
        }
        decode(*generation, record);
    }

    void packetLoss()
    {
        for (auto& [_, generation] : waves)
        {
            generation.scopes.clear();
            generation.payload.reset();
            generation.pending.clear();
            generation.reliable = false;
        }
        unresolved_generations.clear();
        pending_count = 0;
    }

    void flush() { records.flush(MarkerRecord, marker_records); }

private:
    using EntryPtr = Funcmap::EntryPtr;

    struct Payload
    {
        EntryPtr entry{};
        uint32_t marker_flags = 0;
        uint32_t next_index = 0;
    };

    struct Generation
    {
        pcinfo_t kernel_entry{};
        const Funcmap* funcmap = nullptr;
        bool funcmap_checked = false;
        bool first_marker = true;
        bool reliable = true;
        std::vector<EntryPtr> scopes{};
        std::optional<Payload> payload{};
        std::vector<att_shader_data_t> pending{};
    };

    Generation* findGeneration(uint8_t cu, uint8_t simd, uint8_t wave_id)
    {
        auto it = waves.find(waveKey(cu, simd, wave_id));
        return it == waves.end() ? nullptr : &it->second;
    }

    void resolve(Generation& generation, pcinfo_t kernel_entry)
    {
        if (kernel_entry.code_object_id == 0) return;
        if (generation.kernel_entry.code_object_id != kernel_entry.code_object_id)
        {
            generation.funcmap = nullptr;
            generation.funcmap_checked = false;
        }
        generation.kernel_entry = kernel_entry;
        if (!generation.reliable)
        {
            pending_count -= generation.pending.size();
            generation.pending.clear();
            return;
        }
        if (generation.pending.empty()) return;

        auto pending = std::move(generation.pending);
        pending_count -= pending.size();
        generation.pending.clear();
        for (const auto& record : pending) decode(generation, record);
    }

    void decode(Generation& generation, const att_shader_data_t& source)
    {
        if (!generation.reliable) return;
        if (!generation.funcmap_checked)
        {
            generation.funcmap = service.GetFuncmap(generation.kernel_entry.code_object_id);
            generation.funcmap_checked = true;
        }
        const Funcmap* funcmap = generation.funcmap;
        if (!funcmap) return;

        if (generation.payload)
        {
            emitPayload(generation, source);
            return;
        }

        MarkerValue value = decode_marker_value(static_cast<uint32_t>(source.value), *funcmap);
        EntryPtr entry = funcmap->find(value.id);

        if (!entry && value.id == 0 && value.exit_prev && !value.is_enter && !generation.scopes.empty())
        {
            entry = generation.scopes.back();
            generation.scopes.pop_back();
            emitHeader(generation, source, entry, value);
            return;
        }

        if (!entry || entry->kind == FuncmapEntryKind::Kernel) return;

        if (value.exit_prev && !generation.scopes.empty()) generation.scopes.pop_back();
        emitHeader(generation, source, entry, value);
        if (value.is_enter && isScope(entry->kind)) generation.scopes.push_back(entry);

        if (entry->extra_payload_count != 0)
            generation.payload = Payload{
                .entry = entry,
                .marker_flags = markerFlags(value),
            };
    }

    static uint32_t markerFlags(const MarkerValue& value)
    {
        uint32_t flags = ROCPROFILER_THREAD_TRACE_DECODER_MARKER_FLAGS_NONE;
        if (value.exit_prev) flags |= ROCPROFILER_THREAD_TRACE_DECODER_MARKER_FLAGS_EXIT_PREVIOUS;
        if (value.is_enter) flags |= ROCPROFILER_THREAD_TRACE_DECODER_MARKER_FLAGS_ENTER;
        return flags;
    }

    rocprofiler_thread_trace_decoder_marker_t makeRecord(
        const Generation& generation,
        const att_shader_data_t& source,
        const EntryPtr& entry,
        uint32_t record_kind,
        uint32_t flags
    ) const
    {
        rocprofiler_thread_trace_decoder_marker_t marker{};
        marker.size = sizeof(marker);
        marker.shaderdata = source;
        marker.kernel_entry = generation.kernel_entry;
        marker.code_object_id = generation.kernel_entry.code_object_id;
        marker.name = entry->name.empty() ? nullptr : entry->name.c_str();
        marker.source_location = entry->source_loc.empty() ? nullptr : entry->source_loc.c_str();
        marker.marker_id = entry->id;
        marker.record_kind = record_kind;
        marker.marker_kind = publicMarkerKind(entry->kind);
        marker.marker_flags = flags;
        marker.payload_index = std::numeric_limits<uint32_t>::max();
        marker.payload_count = entry->extra_payload_count;
        marker.delay = 0;
        return marker;
    }

    void emitHeader(
        Generation& generation, const att_shader_data_t& source, const EntryPtr& entry, const MarkerValue& value
    )
    {
        uint32_t flags = markerFlags(value);
        if (generation.first_marker)
        {
            flags |= ROCPROFILER_THREAD_TRACE_DECODER_MARKER_FLAGS_NEW_WAVE;
            generation.first_marker = false;
        }

        auto marker =
            makeRecord(generation, source, entry, ROCPROFILER_THREAD_TRACE_DECODER_MARKER_RECORD_HEADER, flags);
        emit(marker);
    }

    void emitPayload(Generation& generation, const att_shader_data_t& source)
    {
        auto& payload = *generation.payload;
        auto marker = makeRecord(
            generation,
            source,
            payload.entry,
            ROCPROFILER_THREAD_TRACE_DECODER_MARKER_RECORD_PAYLOAD,
            payload.marker_flags
        );
        marker.payload_index = payload.next_index++;
        emit(marker);
        if (payload.next_index == payload.entry->extra_payload_count) generation.payload.reset();
    }

    void emit(const rocprofiler_thread_trace_decoder_marker_t& marker)
    {
        records.append(MarkerRecord, marker_records, marker);
        if (marker_records.size() >= 65536) records.flush(MarkerRecord, marker_records);
    }

    RecordEmitter& records;
    ICodeServicer& service;
    std::unordered_map<uint32_t, Generation> waves{};
    std::vector<Generation> unresolved_generations{};
    std::vector<rocprofiler_thread_trace_decoder_marker_t> marker_records{};
    size_t pending_count = 0;
};

Stitcher::Stitcher(
    std::shared_ptr<ICodeServicer> service,
    rocprof_trace_decoder_trace_callback_t _callback,
    void* _cbdata,
    const RecordFilter& filter
) :
codeobj_service(std::move(service)),
record_emitter(_callback, _cbdata, filter),
marker_stream(
    record_emitter.enabled(MarkerRecord) ? std::make_unique<MarkerStream>(record_emitter, *codeobj_service) : nullptr
)
{}

Stitcher::~Stitcher() = default;

void Stitcher::processShaderdata(const att_shader_data_t& record)
{
    if (marker_stream) marker_stream->shaderdata(record);
}

void Stitcher::markerWaveStart(uint8_t cu, uint8_t simd, uint8_t wave_id, pcinfo_t kernel_entry)
{
    if (marker_stream) marker_stream->waveStart(cu, simd, wave_id, kernel_entry);
}

void Stitcher::markerWaveEnd(uint8_t cu, uint8_t simd, uint8_t wave_id, pcinfo_t kernel_entry)
{
    if (marker_stream) marker_stream->waveEnd(cu, simd, wave_id, kernel_entry);
}

void Stitcher::markerResolveAll(CSRegisterHandler& registers)
{
    if (marker_stream) marker_stream->resolveAll(registers);
}

void Stitcher::markerPacketLoss()
{
    if (marker_stream) marker_stream->packetLoss();
}

void Stitcher::flushMarkers()
{
    if (marker_stream) marker_stream->flush();
}
