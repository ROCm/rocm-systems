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
#include <array>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

using rocprof_trace_decoder::codeobj::Funcmap;
using rocprof_trace_decoder::codeobj::FuncmapEntryKind;
using rocprof_trace_decoder::codeobj::MarkerEncoding;
using Marker = rocprofiler_thread_trace_decoder_marker_t;

namespace
{
constexpr auto MarkerRecord = ROCPROFILER_THREAD_TRACE_DECODER_RECORD_MARKER;
constexpr size_t BatchRecords = 4096;
constexpr size_t BatchPayloadWords = 65536;
constexpr size_t Simds = 4;
constexpr size_t WaveSlots = 16;

uint32_t publicKind(FuncmapEntryKind kind)
{
    switch (kind)
    {
        case FuncmapEntryKind::Function: return ROCPROFILER_THREAD_TRACE_DECODER_MARKER_KIND_FUNCTION;
        case FuncmapEntryKind::UserScope: return ROCPROFILER_THREAD_TRACE_DECODER_MARKER_KIND_USER_SCOPE;
        case FuncmapEntryKind::Point: return ROCPROFILER_THREAD_TRACE_DECODER_MARKER_KIND_POINT;
        default: return ROCPROFILER_THREAD_TRACE_DECODER_MARKER_KIND_UNKNOWN;
    }
}
} // namespace

class MarkerStream
{
    /// What a code object declares for one marker ID. Strings live in the
    /// funcmap, which outlives the parse.
    struct Info
    {
        const char* name = nullptr;
        const char* source_location = nullptr;
        uint32_t id = 0;
        uint32_t kind = ROCPROFILER_THREAD_TRACE_DECODER_MARKER_KIND_UNKNOWN;
        uint32_t payload_count = 0;
    };

    // Compiler-assigned IDs use a bounded dense index; sparse IDs a map.
    struct CodeMap
    {
        const Funcmap* source = nullptr;
        MarkerEncoding encoding{};
        uint32_t id_mask = 0xFFFFFFFCu;
        std::vector<Info> dense;
        std::unordered_map<uint32_t, Info> sparse;

        explicit CodeMap(const Funcmap* map)
        {
            if (!map || (map->entries.empty() && map->wave_size == 0)) return;
            source = map;
            encoding = map->marker_encoding.is_valid() ? map->marker_encoding : MarkerEncoding{};
            id_mask = encoding.marker_id_mask();
            dense.resize(std::min<size_t>(65536, std::max<size_t>(256, map->by_id.size() * 2)));
            for (const auto& [marker_id, entry] : map->by_id)
            {
                if (entry->kind == FuncmapEntryKind::Kernel) continue;
                const Info info{
                    entry->name.empty() ? nullptr : entry->name.c_str(),
                    entry->source_loc.empty() ? nullptr : entry->source_loc.c_str(),
                    marker_id,
                    publicKind(entry->kind),
                    entry->extra_payload_count,
                };
                if (marker_id < dense.size())
                    dense[marker_id] = info;
                else
                    sparse.emplace(marker_id, info);
            }
        }

        /// The declared marker, or nullptr if the code object has none with this ID.
        const Info* find(uint32_t id) const
        {
            if (id < dense.size()) return dense[id].id == id && id != 0 ? &dense[id] : nullptr;
            auto it = sparse.find(id);
            return it == sparse.end() ? nullptr : &it->second;
        }
    };

    struct Generation
    {
        pcinfo_t kernel_entry{};
        CodeMap* map = nullptr;
        uint32_t payload_expected = 0;
        bool started = false;
        bool first_marker = true;
        bool reliable = true;
        bool deferred = false;
        std::vector<uint32_t> payload;
        std::vector<Info> scopes;
        std::vector<att_shader_data_t> pending;
        Marker header{};
        att_shader_data_t first_lost{}; ///< First word that could not be decoded.
        uint32_t lost = 0;
    };

    struct Slot
    {
        Generation current;
        std::vector<Generation> retired;
        std::optional<Generation> saved;
        size_t saved_predecessors = 0;
        size_t current_predecessors = 0;
        bool restored = false;
    };

    struct ClockDomain
    {
        int64_t origin_time;
        uint64_t origin_sample;
        int64_t minimum_relative = 0;
    };

public:
    MarkerStream(RecordEmitter& emitter, ICodeServicer& service) :
    records(emitter), service(service)
    {
        marker_records.reserve(BatchRecords);
        payload_words.reserve(BatchPayloadWords);
    }

    void setRegisters(CSRegisterHandler* handler) { registers = handler; }

    void waveStart(uint8_t cu, uint8_t simd, uint8_t wave, pcinfo_t pc)
    {
        auto* slot = getSlot(cu, simd, wave, true);
        if (!slot) return;
        if (slot->saved && !slot->saved->reliable)
        {
            release(*slot->saved);
            slot->saved.reset();
        }
        retire(*slot);
        slot->current.kernel_entry = pc;
        slot->current.started = true;
        drain(*slot, false);
    }

    void waveEnd(uint8_t cu, uint8_t simd, uint8_t wave, pcinfo_t pc)
    {
        auto* slot = getSlot(cu, simd, wave);
        if (!slot || !slot->current.started) return;
        if (pc.code_object_id != 0) slot->current.kernel_entry = pc;
        // Shaderdata may arrive after WAVE_END. Do not terminate a payload here.
        drain(*slot, false);
    }

    void contextSave(uint8_t cu, uint8_t simd, uint8_t wave)
    {
        auto* slot = getSlot(cu, simd, wave);
        if (!slot || !slot->current.started) return;
        if (slot->saved) release(*slot->saved);
        slot->saved_predecessors = slot->restored ? slot->current_predecessors : slot->retired.size();
        slot->saved = std::move(slot->current);
        slot->current = {};
        slot->restored = false;
    }

    void contextRestore(uint8_t cu, uint8_t simd, uint8_t wave)
    {
        auto* slot = getSlot(cu, simd, wave);
        if (!slot) return;
        if (!slot->saved)
        {
            // An unmatched restore has no reliable marker framing. A repeated
            // restore notification must not end the wave it already resumed.
            if (!slot->restored) retire(*slot);
            return;
        }
        retire(*slot);
        slot->current = std::move(*slot->saved);
        slot->saved.reset();
        slot->restored = true;
        slot->current_predecessors = slot->saved_predecessors;
        drain(*slot, false);
    }

    void shaderdata(const att_shader_data_t& record)
    {
        if (record.flags & (1u << ROCPROFILER_THREAD_TRACE_DECODER_SHADERDATA_FLAGS_PRIV)) return;
        auto* slot = getSlot(record.cu, record.simd, record.wave_id, true);
        if (!slot)
        {
            Generation orphan;
            lose(orphan, record);
            release(orphan);
            return;
        }
        // Late shaderdata still belongs to the saved wave until another wave
        // starts using the slot.
        auto& generation = !slot->current.started && slot->saved ? *slot->saved : slot->current;
        if (!generation.started || !generation.reliable)
            lose(generation, record);
        else if (generation.kernel_entry.code_object_id == 0 || generation.deferred)
            generation.pending.push_back(record);
        else
            decode(generation, record);
    }

    void resolveAll(CSRegisterHandler& handler)
    {
        registers = &handler;
        forEachSlot([&](Slot& slot) { drain(slot, false); });
    }

    void packetLoss()
    {
        forEachSlot(
            [&](Slot& slot)
            {
                // Retired waves ended before the loss; they only await attribution.
                slot.saved_predecessors = 0;
                slot.current_predecessors = 0;
                slot.restored = false;
                invalidate(slot.current);
                slot.current.deferred = false;
                if (slot.saved)
                {
                    release(*slot.saved);
                    slot.saved.reset();
                }
            }
        );
    }

    void flush()
    {
        records.flush(MarkerRecord, marker_records);
        payload_words.clear();
    }

    void finishStream()
    {
        forEachSlot(
            [&](Slot& slot)
            {
                drain(slot, true);
                release(slot.current);
            }
        );
        flush();
        for (auto& unit : units) unit.reset();
        registers = nullptr;
    }

    void beginCalibration()
    {
        calibrating = true;
        finishStream();
        clocks.clear();
    }

    void endCalibration()
    {
        finishStream();
        calibrating = false;
    }
    void copyCalibration(const MarkerStream& other) { clocks = other.clocks; }

private:
    Slot* getSlot(uint8_t cu, uint8_t simd, uint8_t wave, bool create = false)
    {
        if (simd >= Simds || wave >= WaveSlots) return nullptr;
        auto& unit = units[cu];
        if (!unit && create) unit = std::make_unique<Slot[]>(Simds * WaveSlots);
        return unit ? &unit[simd * WaveSlots + wave] : nullptr;
    }

    template <typename F> void forEachSlot(F&& fn)
    {
        for (auto& unit : units)
            if (unit)
                for (size_t i = 0; i < Simds * WaveSlots; ++i) fn(unit[i]);
    }

    void lose(Generation& generation, const att_shader_data_t& first, size_t count = 1)
    {
        if (generation.lost == 0) generation.first_lost = first;
        generation.lost += static_cast<uint32_t>(count);
    }

    /// Words still waiting for attribution can no longer be decoded.
    void losePending(Generation& generation)
    {
        if (generation.pending.empty()) return;
        lose(generation, generation.pending.front(), generation.pending.size());
        std::vector<att_shader_data_t>().swap(generation.pending);
    }

    /// Ends a generation. One INVALID marker, its last record, reports the
    /// words that could not be decoded.
    void release(Generation& generation)
    {
        finish(generation);
        losePending(generation);
        if (generation.lost)
        {
            const bool known = generation.kernel_entry.code_object_id != 0;
            Marker marker{
                .size = sizeof(Marker),
                .shaderdata = generation.first_lost,
                .kernel_entry = known ? generation.kernel_entry : pcinfo_t{},
                .marker_flags = ROCPROFILER_THREAD_TRACE_DECODER_MARKER_FLAGS_INVALID,
                .payload_dropped = generation.lost,
            };
            if (generation.first_marker) marker.marker_flags |= ROCPROFILER_THREAD_TRACE_DECODER_MARKER_FLAGS_NEW_WAVE;
            emit(marker, {});
        }
        generation = {};
    }

    /// Stops decoding a generation; its later words are lost.
    void invalidate(Generation& generation)
    {
        finish(generation);
        losePending(generation);
        std::vector<uint32_t>().swap(generation.payload);
        std::vector<Info>().swap(generation.scopes);
        generation.reliable = false;
    }

    void retire(Slot& slot)
    {
        auto& generation = slot.current;
        // Anything left to emit waits behind older generations of the slot.
        if (generation.pending.empty() && (generation.lost == 0 || slot.retired.empty()))
            release(generation);
        else
        {
            auto position = slot.restored ? slot.retired.begin() + slot.current_predecessors : slot.retired.end();
            slot.retired.insert(position, std::move(generation));
            generation = {};
        }
        slot.restored = false;
        slot.current_predecessors = 0;
    }

    void resolve(Generation& generation)
    {
        auto& pc = generation.kernel_entry;
        if (registers && pc.code_object_id == 0 && pc.address != 0) pc = registers->get_wave_start_delayed(pc.address);
    }

    void replay(Generation& generation)
    {
        if (generation.kernel_entry.code_object_id == 0 || !generation.reliable || generation.pending.empty()) return;
        auto pending = std::move(generation.pending);
        generation.pending.clear();
        for (const auto& record : pending) decode(generation, record);
    }

    size_t drainRetired(Slot& slot, size_t limit, bool final)
    {
        size_t consumed = 0;
        for (; consumed < limit; ++consumed)
        {
            auto& generation = slot.retired[consumed];
            resolve(generation);
            if (generation.kernel_entry.code_object_id == 0 && generation.reliable && !final) break;
            replay(generation);
            release(generation);
        }
        slot.retired.erase(slot.retired.begin(), slot.retired.begin() + consumed);
        return consumed;
    }

    // Delivers whatever no longer waits for an older generation of the slot.
    // Called whenever slot state changes, so a resolved wave is not held back
    // until the end of the buffer.
    void drain(Slot& slot, bool final)
    {
        resolve(slot.current);
        // Keep each logical wave's records together across CWSR. A temporary
        // wave in the same hardware slot must not reset a consumer's stack
        // before the saved wave resumes, even when both use the same code object.
        if (slot.saved)
        {
            resolve(*slot.saved);
            slot.saved_predecessors -= drainRetired(slot, slot.saved_predecessors, final);
            if (slot.saved_predecessors == 0) replay(*slot.saved);
            slot.current.deferred = true;
            if (!final) return;
            release(*slot.saved);
            slot.saved.reset();
        }
        if (slot.restored)
        {
            slot.current_predecessors -= drainRetired(slot, slot.current_predecessors, final);
            slot.current.deferred = slot.current_predecessors != 0;
            if (!slot.current.deferred) replay(slot.current);
            if (!final) return;
            release(slot.current);
            slot.restored = false;
        }
        drainRetired(slot, slot.retired.size(), final);
        slot.current.deferred = !slot.retired.empty();
        if (!slot.current.deferred) replay(slot.current);
    }

    // Payload words outnumber headers. Keep their dispatch separate from the
    // name lookup, scope, and resource-limit handling needed only for headers.
    void decode(Generation& generation, const att_shader_data_t& source)
    {
        if (generation.payload_expected == 0)
        {
            decodeHeader(generation, source);
            return;
        }
        if (!calibrating) generation.payload.push_back(static_cast<uint32_t>(source.value));
        if (--generation.payload_expected == 0)
        {
            emit(generation.header, generation.payload);
            generation.payload.clear();
        }
    }

    void decodeHeader(Generation& generation, const att_shader_data_t& source)
    {
        if (!generation.map)
        {
            const auto id = generation.kernel_entry.code_object_id;
            auto it = maps.find(id);
            if (it == maps.end()) it = maps.try_emplace(id, service.GetFuncmap(id)).first;
            generation.map = &it->second;
        }
        auto& map = *generation.map;
        if (!map.source) return;

        uint32_t raw = static_cast<uint32_t>(source.value);
        uint32_t flags = raw & 3u;
        // s_ttracedata_imm carries a literal, never a sampled shader clock.
        const bool constant_header = source.flags & (1u << ROCPROFILER_THREAD_TRACE_DECODER_SHADERDATA_FLAGS_IMM);
        uint32_t id = constant_header ? raw >> 2 : (raw & map.id_mask) >> 2;
        const Info* declared = map.find(id);
        const bool bare_exit = id == 0 && flags == ROCPROFILER_THREAD_TRACE_DECODER_MARKER_FLAGS_EXIT_PREVIOUS;
        if (!declared && !bare_exit && !constant_header && map.encoding.has_shader_clock())
        {
            id = raw >> 2;
            declared = map.find(id);
        }
        if (id == 0 && !bare_exit) return;
        const bool known = declared != nullptr || bare_exit;
        Info info = declared ? *declared : Info{nullptr, nullptr, id};
        if (flags & ROCPROFILER_THREAD_TRACE_DECODER_MARKER_FLAGS_EXIT_PREVIOUS)
        {
            if (!generation.scopes.empty())
            {
                // A bare exit names the scope it closes.
                if (bare_exit) info = generation.scopes.back();
                generation.scopes.pop_back();
            }
        }

        Marker marker{
            .size = sizeof(Marker),
            .shaderdata = source,
            .kernel_entry = generation.kernel_entry,
            .name = info.name,
            .source_location = info.source_location,
            .marker_id = info.id,
            .marker_kind = info.kind,
            .marker_flags = flags,
        };
        if (generation.first_marker)
        {
            marker.marker_flags |= ROCPROFILER_THREAD_TRACE_DECODER_MARKER_FLAGS_NEW_WAVE;
            generation.first_marker = false;
        }
        // A bare exit carries no payload, whatever its scope declared.
        const uint32_t count = bare_exit ? 0 : info.payload_count;
        if ((calibrating || !clocks.empty()) && known && !constant_header && count == 0)
            correctClock(marker, map.encoding);
        if (flags & ROCPROFILER_THREAD_TRACE_DECODER_MARKER_FLAGS_ENTER) generation.scopes.push_back(info);

        if (count == 0)
        {
            emit(marker, {});
            return;
        }
        generation.header = marker;
        generation.payload_expected = count;
    }

    void correctClock(Marker& marker, MarkerEncoding encoding)
    {
        if (!encoding.has_shader_clock()) return;
        const uint64_t key = (uint64_t(marker.shaderdata.cu) << 24) | (uint64_t(marker.shaderdata.simd) << 16) |
                             (uint64_t(encoding.shader_clock_bits) << 8) | encoding.shader_clock_shift;
        const uint64_t sample = uint64_t(encoding.decode_shader_clock(static_cast<uint32_t>(marker.shaderdata.value)))
                             << encoding.shader_clock_shift;
        auto it = clocks.find(key);
        if (it == clocks.end())
        {
            if (calibrating) clocks.emplace(key, ClockDomain{marker.shaderdata.time, sample});
            return;
        }
        auto& clock = it->second;
        const uint64_t window = uint64_t(1) << (encoding.shader_clock_bits + encoding.shader_clock_shift);
        const uint64_t delta_time = uint64_t(marker.shaderdata.time) - uint64_t(clock.origin_time);
        const uint64_t delta_sample = sample - clock.origin_sample;
        const int64_t relative = int64_t((delta_time - delta_sample + window / 2) & (window - 1)) - int64_t(window / 2);
        if (calibrating)
            clock.minimum_relative = std::min(clock.minimum_relative, relative);
        else
        {
            marker.shaderdata.time -= std::max<int64_t>(
                0, relative - clock.minimum_relative - ((int64_t(1) << encoding.shader_clock_shift) - 1)
            );
        }
    }

    /// Delivers a payload cut short by the end of the wave's data.
    void finish(Generation& generation)
    {
        if (!generation.payload_expected) return;
        generation.header.payload_dropped = generation.payload_expected;
        emit(generation.header, generation.payload);
        generation.payload.clear();
        generation.payload_expected = 0;
    }

    void emit(Marker& marker, const std::vector<uint32_t>& payload)
    {
        if (calibrating || records.status() != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS) return;
        marker.payload_count = static_cast<uint32_t>(payload.size());
        // The pool only grows past its capacity while empty, so earlier payload
        // pointers stay valid until the flush.
        if (marker_records.size() == BatchRecords || payload_words.size() + payload.size() > payload_words.capacity())
            flush();
        marker_records.push_back(marker);
        if (!payload.empty())
        {
            payload_words.insert(payload_words.end(), payload.begin(), payload.end());
            marker_records.back().payload = payload_words.data() + payload_words.size() - payload.size();
        }
    }

    RecordEmitter& records;
    ICodeServicer& service;
    CSRegisterHandler* registers = nullptr;
    std::array<std::unique_ptr<Slot[]>, 256> units{};
    std::unordered_map<uint64_t, CodeMap> maps;
    std::unordered_map<uint64_t, ClockDomain> clocks;
    std::vector<Marker> marker_records;
    std::vector<uint32_t> payload_words;
    bool calibrating = false;
};

Stitcher::Stitcher(
    std::shared_ptr<ICodeServicer> service,
    rocprof_trace_decoder_trace_callback_t callback,
    void* userdata,
    const RecordFilter& filter
) :
codeobj_service(std::move(service)),
record_emitter(callback, userdata, filter),
marker_stream(
    record_emitter.enabled(MarkerRecord) ? std::make_unique<MarkerStream>(record_emitter, *codeobj_service) : nullptr
)
{}

Stitcher::~Stitcher() = default;
void Stitcher::processMarkerShaderdata(const att_shader_data_t& record) { marker_stream->shaderdata(record); }
void Stitcher::markerSetRegisters(CSRegisterHandler& registers)
{
    if (marker_stream) marker_stream->setRegisters(&registers);
}
void Stitcher::markerWaveStart(uint8_t cu, uint8_t simd, uint8_t wave, pcinfo_t pc)
{
    if (marker_stream) marker_stream->waveStart(cu, simd, wave, pc);
}
void Stitcher::markerWaveEnd(uint8_t cu, uint8_t simd, uint8_t wave, pcinfo_t pc)
{
    if (marker_stream) marker_stream->waveEnd(cu, simd, wave, pc);
}
void Stitcher::markerContextSave(uint8_t cu, uint8_t simd, uint8_t wave)
{
    if (marker_stream) marker_stream->contextSave(cu, simd, wave);
}
void Stitcher::markerContextRestore(uint8_t cu, uint8_t simd, uint8_t wave)
{
    if (marker_stream) marker_stream->contextRestore(cu, simd, wave);
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
void Stitcher::finishMarkers()
{
    if (marker_stream) marker_stream->finishStream();
}
void Stitcher::beginMarkerCalibration()
{
    if (marker_stream) marker_stream->beginCalibration();
}
void Stitcher::endMarkerCalibration()
{
    if (marker_stream) marker_stream->endCalibration();
}
void Stitcher::copyMarkerCalibration(const Stitcher& other)
{
    if (marker_stream && other.marker_stream) marker_stream->copyCalibration(*other.marker_stream);
}
