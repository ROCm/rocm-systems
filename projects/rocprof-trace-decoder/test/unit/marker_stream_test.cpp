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

#include <gtest/gtest.h>

#include <algorithm>
#include <deque>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
using rocprof_trace_decoder::codeobj::Funcmap;
using rocprof_trace_decoder::codeobj::parse_funcmap_section;
using Marker = rocprofiler_thread_trace_decoder_marker_t;
using Metadata = rocprofiler_thread_trace_decoder_marker_metadata_t;
using Shaderdata = rocprofiler_thread_trace_decoder_shaderdata_t;

constexpr auto MarkerRecord = ROCPROFILER_THREAD_TRACE_DECODER_RECORD_MARKER;
constexpr auto ShaderdataRecord = ROCPROFILER_THREAD_TRACE_DECODER_RECORD_SHADERDATA;
constexpr uint32_t Enter = ROCPROFILER_THREAD_TRACE_DECODER_MARKER_FLAGS_ENTER;
constexpr uint32_t Exit = ROCPROFILER_THREAD_TRACE_DECODER_MARKER_FLAGS_EXIT_PREVIOUS;
constexpr uint32_t NewWave = ROCPROFILER_THREAD_TRACE_DECODER_MARKER_FLAGS_NEW_WAVE;
constexpr uint32_t Incomplete = ROCPROFILER_THREAD_TRACE_DECODER_MARKER_FLAGS_INCOMPLETE;
constexpr uint8_t Priv = 1u << ROCPROFILER_THREAD_TRACE_DECODER_SHADERDATA_FLAGS_PRIV;
constexpr uint8_t Imm = 1u << ROCPROFILER_THREAD_TRACE_DECODER_SHADERDATA_FLAGS_IMM;

Shaderdata shaderdata(
    int64_t time, uint32_t value, uint8_t cu = 0, uint8_t simd = 0, uint8_t wave = 0, uint8_t flags = 0
)
{ return {.time = time, .value = value, .cu = cu, .simd = simd, .wave_id = wave, .flags = flags}; }

RecordFilter markerFilter(bool immediate = false, bool raw_shaderdata = false)
{
    RecordFilter filter;
    filter.clear();
    filter.set(MarkerRecord, immediate ? ROCPROFILER_THREAD_TRACE_DECODER_RECORD_REQUEST_FLAGS_IMMEDIATE : 0);
    if (raw_shaderdata) filter.set(ShaderdataRecord, 0);
    return filter;
}

class MarkerCodeServicer : public ICodeServicer
{
public:
    assemblyLine GetInstruction(pcinfo_t, int) override
    {
        ADD_FAILURE() << "Marker decoding unexpectedly requested disassembly";
        return {};
    }

    const Funcmap* GetFuncmap(uint64_t code_object_id) override
    {
        ++lookups;
        auto it = funcmaps.find(code_object_id);
        return it == funcmaps.end() ? nullptr : &it->second;
    }

    void add(uint64_t id, const char* text) { funcmaps.emplace(id, parse_funcmap_section(text, true)); }

    std::unordered_map<uint64_t, Funcmap> funcmaps;
    size_t lookups = 0;
};

// The public pointers are borrowed only during the callback. Keep copied
// records in a deque so pointers to their owned strings/metadata stay stable.
struct CapturedMarker : Marker
{
    explicit CapturedMarker(const Marker& marker) :
    Marker(marker),
    owned_metadata(*marker.metadata),
    owned_name(marker.metadata->name ? marker.metadata->name : ""),
    owned_location(marker.metadata->source_location ? marker.metadata->source_location : "")
    {
        if (marker.payload_count) owned_payload.assign(marker.payload, marker.payload + marker.payload_count);
        owned_metadata.name = marker.metadata->name ? owned_name.c_str() : nullptr;
        owned_metadata.source_location = marker.metadata->source_location ? owned_location.c_str() : nullptr;
        metadata = &owned_metadata;
        payload = owned_payload.empty() ? nullptr : owned_payload.data();
    }

    CapturedMarker(const CapturedMarker&) = delete;
    CapturedMarker(CapturedMarker&&) = delete;

    Metadata owned_metadata;
    std::string owned_name;
    std::string owned_location;
    std::vector<uint32_t> owned_payload;
};

struct MarkerCapture
{
    std::deque<CapturedMarker> markers;
    std::vector<Shaderdata> raw;
    std::vector<uint64_t> batches;

    const CapturedMarker* atTime(int64_t time) const
    {
        auto it = std::find_if(
            markers.begin(), markers.end(), [time](const auto& record) { return record.shaderdata.time == time; }
        );
        return it == markers.end() ? nullptr : &*it;
    }
};

rocprofiler_thread_trace_decoder_status_t captureMarkers(
    rocprofiler_thread_trace_decoder_record_type_t type, void* data, uint64_t count, void* userdata
)
{
    auto& capture = *static_cast<MarkerCapture*>(userdata);
    if (type == ShaderdataRecord)
    {
        auto* records = static_cast<const Shaderdata*>(data);
        capture.raw.insert(capture.raw.end(), records, records + count);
    }
    if (type != MarkerRecord) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;

    capture.batches.push_back(count);
    auto* cursor = static_cast<const unsigned char*>(data);
    for (uint64_t i = 0; i < count; ++i)
    {
        const auto& record = *reinterpret_cast<const Marker*>(cursor);
        EXPECT_EQ(record.size, sizeof(Marker));
        EXPECT_NE(record.metadata, nullptr);
        if (!record.metadata) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR;
        EXPECT_EQ(record.metadata->size, sizeof(Metadata));
        EXPECT_TRUE(record.payload || record.payload_count == 0);
        if (!record.payload && record.payload_count) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR;
        capture.markers.emplace_back(record);
        cursor += record.size;
    }
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

class MarkerStreamTest : public testing::Test
{
protected:
    std::shared_ptr<MarkerCodeServicer> service = std::make_shared<MarkerCodeServicer>();
    MarkerCapture capture;
    pcinfo_t kernel{0x100, 7};
};

TEST_F(MarkerStreamTest, MarkersRequireExplicitOptIn)
{
    service->add(7, "P:1:point\n");
    Stitcher stitcher(service, captureMarkers, &capture);
    stitcher.markerWaveStart(0, 0, 0, kernel);
    stitcher.processShaderdata(shaderdata(1, 4));
    stitcher.finishMarkers();

    EXPECT_TRUE(capture.markers.empty());
    EXPECT_EQ(service->lookups, 0u);
}

TEST_F(MarkerStreamTest, CompletePayloadIsAttachedToItsHeaderAndPreservesRawRecords)
{
    service->add(7, "U:1:scope\nP:2:value@test.cpp:42\nR:2:extra_payload_count=2\n");
    Stitcher stitcher(service, captureMarkers, &capture, markerFilter(false, true));
    stitcher.markerWaveStart(0, 1, 2, kernel);
    std::vector<Shaderdata> raw;
    const Shaderdata input[] = {
        shaderdata(10, 6, 0, 1, 2, Imm),
        shaderdata(11, 8, 0, 1, 2),
        shaderdata(12, 1, 0, 1, 2), // Payload looks like a scope exit.
        shaderdata(13, 6, 0, 1, 2, Priv),
        shaderdata(14, 6, 0, 1, 2, Imm), // Payload looks like a scope entry.
        shaderdata(15, 1, 0, 1, 2),
    };
    for (const auto& record : input)
    {
        stitcher.records().append(ShaderdataRecord, raw, record);
        stitcher.processShaderdata(record);
    }
    stitcher.records().flush(ShaderdataRecord, raw);
    stitcher.finishMarkers();

    ASSERT_EQ(capture.raw.size(), 6u);
    for (size_t i = 0; i < capture.raw.size(); ++i)
    {
        EXPECT_EQ(capture.raw[i].time, input[i].time);
        EXPECT_EQ(capture.raw[i].value, input[i].value);
        EXPECT_EQ(capture.raw[i].flags, input[i].flags);
        EXPECT_EQ(capture.raw[i].simd, input[i].simd);
        EXPECT_EQ(capture.raw[i].wave_id, input[i].wave_id);
    }
    ASSERT_EQ(capture.markers.size(), 3u);
    const auto& enter = capture.markers[0];
    const auto& value = capture.markers[1];
    const auto& exit = capture.markers[2];
    EXPECT_EQ(enter.marker_flags, Enter | NewWave);
    EXPECT_EQ(enter.metadata->marker_id, 1u);
    EXPECT_STREQ(enter.metadata->name, "scope");
    EXPECT_EQ(enter.metadata->marker_kind, ROCPROFILER_THREAD_TRACE_DECODER_MARKER_KIND_USER_SCOPE);
    EXPECT_EQ(enter.kernel_entry.address, kernel.address);
    EXPECT_EQ(enter.kernel_entry.code_object_id, 7u);
    EXPECT_EQ(enter.metadata->code_object_id, 7u);
    EXPECT_EQ(value.shaderdata.time, 11);
    EXPECT_EQ(value.shaderdata.value, 8u);
    EXPECT_STREQ(value.metadata->source_location, "test.cpp:42");
    EXPECT_EQ(value.metadata->payload_count, 2u);
    ASSERT_EQ(value.payload_count, 2u);

    EXPECT_EQ(value.payload[0], 1u);

    EXPECT_EQ(value.payload[1], 6u);

    EXPECT_EQ(value.marker_flags, 0u);
    EXPECT_EQ(exit.metadata->marker_id, 1u);
    EXPECT_EQ(exit.shaderdata.time, 15);
    EXPECT_EQ(exit.marker_flags, Exit);
    EXPECT_EQ(enter.shaderdata.time, input[0].time);
}

TEST_F(MarkerStreamTest, ImmediateDeliveryWaitsForTheWholePayloadAndSuppressesRawShaderdata)
{
    service->add(7, "P:1:data\nR:1:extra_payload_count=2\n");
    Stitcher stitcher(service, captureMarkers, &capture, markerFilter(true));
    stitcher.markerWaveStart(0, 0, 0, kernel);
    std::vector<Shaderdata> raw;
    auto emit = [&](int64_t time, uint32_t value)
    {
        auto record = shaderdata(time, value);
        stitcher.records().append(ShaderdataRecord, raw, record);
        stitcher.processShaderdata(record);
    };

    emit(1, 4);
    emit(2, 0xAABBCCDDu);
    stitcher.flushMarkers();
    EXPECT_TRUE(capture.markers.empty());
    emit(3, 0x11223344u);

    EXPECT_TRUE(capture.raw.empty());
    EXPECT_TRUE(raw.empty());
    ASSERT_EQ(capture.markers.size(), 1u);
    EXPECT_EQ(capture.batches, std::vector<uint64_t>{1});
    ASSERT_EQ(capture.markers[0].payload_count, 2u);
    EXPECT_EQ(capture.markers[0].payload[0], 0xAABBCCDDu);
    EXPECT_EQ(capture.markers[0].payload[1], 0x11223344u);
}

TEST_F(MarkerStreamTest, UnknownNumericScopeDoesNotCloseItsKnownParent)
{
    service->add(7, "F:1:function\n");
    Stitcher stitcher(service, captureMarkers, &capture, markerFilter());
    stitcher.markerWaveStart(0, 0, 0, kernel);
    stitcher.processShaderdata(shaderdata(10, 6));
    stitcher.processShaderdata(shaderdata(20, (100u << 2) | 2u));
    stitcher.processShaderdata(shaderdata(30, 1));
    stitcher.processShaderdata(shaderdata(40, 1));
    stitcher.finishMarkers();

    std::vector<int64_t> function_times;
    for (const auto& record : capture.markers)
        if (record.metadata->marker_id == 1) function_times.push_back(record.shaderdata.time);
    EXPECT_EQ(function_times, (std::vector<int64_t>{10, 40}));
    const auto* exit = capture.atTime(40);
    ASSERT_NE(exit, nullptr);
    EXPECT_EQ(exit->marker_flags, Exit);
}

TEST_F(MarkerStreamTest, FusedTransitionPopsAnUnknownScopeBeforeEnteringAKnownScope)
{
    service->add(7, "F:1:parent\nU:2:child\n");
    Stitcher stitcher(service, captureMarkers, &capture, markerFilter());
    stitcher.markerWaveStart(0, 0, 0, kernel);
    stitcher.processShaderdata(shaderdata(10, 6));
    stitcher.processShaderdata(shaderdata(20, (100u << 2) | 2u));
    stitcher.processShaderdata(shaderdata(30, (2u << 2) | 3u));
    stitcher.processShaderdata(shaderdata(40, 1));
    stitcher.processShaderdata(shaderdata(50, 1));
    stitcher.finishMarkers();

    const auto* transition = capture.atTime(30);
    const auto* child_exit = capture.atTime(40);
    const auto* parent_exit = capture.atTime(50);
    ASSERT_NE(transition, nullptr);
    ASSERT_NE(child_exit, nullptr);
    ASSERT_NE(parent_exit, nullptr);
    EXPECT_EQ(transition->metadata->marker_id, 2u);
    EXPECT_EQ(transition->marker_flags, Enter | Exit);
    EXPECT_EQ(child_exit->metadata->marker_id, 2u);
    EXPECT_EQ(parent_exit->metadata->marker_id, 1u);
}

TEST_F(MarkerStreamTest, InterleavedPayloadsUseEachWavesCodeObjectAndFraming)
{
    service->add(7, "P:1:first\nR:1:extra_payload_count=2\n");
    service->add(8, "P:1:second\nR:1:extra_payload_count=1\n");
    Stitcher stitcher(service, captureMarkers, &capture, markerFilter());
    stitcher.markerWaveStart(0, 0, 1, kernel);
    stitcher.markerWaveStart(0, 0, 2, {0x200, 8});
    stitcher.processShaderdata(shaderdata(10, 4, 0, 0, 1));
    stitcher.processShaderdata(shaderdata(11, 4, 0, 0, 2));
    stitcher.processShaderdata(shaderdata(12, 101, 0, 0, 1));
    stitcher.processShaderdata(shaderdata(13, 201, 0, 0, 2));
    stitcher.processShaderdata(shaderdata(14, 102, 0, 0, 1));
    stitcher.finishMarkers();

    ASSERT_EQ(capture.markers.size(), 2u);
    const auto* first = capture.atTime(10);
    const auto* second = capture.atTime(11);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_STREQ(first->metadata->name, "first");
    EXPECT_STREQ(second->metadata->name, "second");
    EXPECT_EQ(first->metadata->code_object_id, 7u);
    EXPECT_EQ(second->metadata->code_object_id, 8u);
    ASSERT_EQ(first->payload_count, 2u);
    ASSERT_EQ(second->payload_count, 1u);
    EXPECT_EQ(first->payload[0], 101u);
    EXPECT_EQ(first->payload[1], 102u);
    EXPECT_EQ(second->payload[0], 201u);
    EXPECT_EQ(first->shaderdata.wave_id, 1u);
    EXPECT_EQ(second->shaderdata.wave_id, 2u);
}

TEST_F(MarkerStreamTest, DelayedReplayKeepsAnOlderGenerationAheadOfSlotReuse)
{
    service->add(7, "F:1:function\nP:2:point\n");
    Stitcher stitcher(service, captureMarkers, &capture, markerFilter());
    stitcher.markerWaveStart(1, 2, 3, {0x200, 0});
    stitcher.processShaderdata(shaderdata(10, 6, 1, 2, 3));
    stitcher.markerWaveEnd(1, 2, 3, {0x200, 0});
    stitcher.processShaderdata(shaderdata(15, 8, 1, 2, 3));
    stitcher.markerWaveStart(1, 2, 3, kernel);
    stitcher.processShaderdata(shaderdata(20, 6, 1, 2, 3));
    stitcher.flushMarkers();

    EXPECT_TRUE(capture.markers.empty());

    CSRegisterHandler registers;
    registers.table_from_start.write().insert(address_range_t{0x100, 0x200, 7});
    stitcher.markerResolveAll(registers);
    stitcher.processShaderdata(shaderdata(40, 1, 1, 2, 3));
    stitcher.finishMarkers();

    ASSERT_EQ(capture.markers.size(), 4u);
    const auto* older_enter = capture.atTime(10);
    const auto* older_point = capture.atTime(15);
    const auto* newer_exit = capture.atTime(40);
    ASSERT_NE(older_enter, nullptr);
    ASSERT_NE(older_point, nullptr);
    ASSERT_NE(newer_exit, nullptr);
    EXPECT_EQ(capture.markers[0].shaderdata.time, 10);
    EXPECT_EQ(capture.markers[1].shaderdata.time, 15);
    EXPECT_EQ(capture.markers[2].shaderdata.time, 20);
    EXPECT_EQ(capture.markers[3].shaderdata.time, 40);
    EXPECT_EQ(capture.markers[2].marker_flags, Enter | NewWave);
    EXPECT_EQ(older_enter->marker_flags, Enter | NewWave);
    EXPECT_EQ(older_point->marker_flags, 0u);
    EXPECT_EQ(newer_exit->marker_flags, Exit);
}

TEST_F(MarkerStreamTest, WaveEndAllowsDelayedPayloadBeforeFinishingTheStream)
{
    service->add(7, "P:1:data\nR:1:extra_payload_count=2\n");
    Stitcher stitcher(service, captureMarkers, &capture, markerFilter(true));
    stitcher.markerWaveStart(0, 0, 0, kernel);
    stitcher.processShaderdata(shaderdata(10, 4));
    stitcher.processShaderdata(shaderdata(11, 123));
    stitcher.markerWaveEnd(0, 0, 0, kernel);
    stitcher.flushMarkers();
    EXPECT_TRUE(capture.markers.empty());

    stitcher.processShaderdata(shaderdata(12, 456));
    stitcher.finishMarkers();
    ASSERT_EQ(capture.markers.size(), 1u);
    EXPECT_EQ(capture.markers[0].marker_flags, NewWave);
    ASSERT_EQ(capture.markers[0].payload_count, 2u);

    EXPECT_EQ(capture.markers[0].payload[1], 456u);
}

TEST_F(MarkerStreamTest, DelayedPartialPayloadFinishesBeforeTheNextGeneration)
{
    service->add(7, "P:1:data\nR:1:extra_payload_count=2\nP:2:point\n");
    Stitcher stitcher(service, captureMarkers, &capture, markerFilter());
    stitcher.markerWaveStart(0, 0, 0, {0x200, 0});
    stitcher.processShaderdata(shaderdata(10, 4));
    stitcher.processShaderdata(shaderdata(11, 123));
    stitcher.markerWaveStart(0, 0, 0, kernel);
    stitcher.processShaderdata(shaderdata(20, 8));

    CSRegisterHandler registers;
    registers.table_from_start.write().insert(address_range_t{0x100, 0x200, 7});
    stitcher.markerResolveAll(registers);
    stitcher.finishMarkers();

    ASSERT_EQ(capture.markers.size(), 2u);
    const auto& partial = capture.markers[0];
    EXPECT_EQ(partial.shaderdata.time, 10);
    EXPECT_EQ(partial.marker_flags, NewWave | Incomplete);
    ASSERT_EQ(partial.payload_count, 1u);
    EXPECT_EQ(partial.payload[0], 123u);
    EXPECT_EQ(capture.markers[1].shaderdata.time, 20);
    EXPECT_EQ(capture.markers[1].marker_flags, NewWave);
}

TEST_F(MarkerStreamTest, AnUnresolvedOlderGenerationDoesNotHideANewerOneAtTraceEnd)
{
    service->add(7, "P:1:point\n");
    Stitcher stitcher(service, captureMarkers, &capture, markerFilter());
    stitcher.markerWaveStart(0, 0, 0, {0x200, 0});
    stitcher.processShaderdata(shaderdata(10, 4));
    stitcher.markerWaveStart(0, 0, 0, kernel);
    stitcher.processShaderdata(shaderdata(20, 4));
    stitcher.finishMarkers();

    ASSERT_EQ(capture.markers.size(), 1u);
    EXPECT_EQ(capture.markers[0].shaderdata.time, 20);
    EXPECT_EQ(capture.markers[0].metadata->code_object_id, 7u);
    EXPECT_EQ(capture.markers[0].marker_flags, NewWave);
}

TEST_F(MarkerStreamTest, PacketLossEmitsPartialPayloadAndRequiresANewGeneration)
{
    service->add(7, "P:1:data\nR:1:extra_payload_count=3\nP:2:point\n");
    Stitcher stitcher(service, captureMarkers, &capture, markerFilter());
    stitcher.markerWaveStart(0, 0, 0, kernel);
    stitcher.processShaderdata(shaderdata(10, 4));
    stitcher.processShaderdata(shaderdata(11, 123));
    stitcher.markerPacketLoss();
    stitcher.processShaderdata(shaderdata(12, 8));
    stitcher.markerWaveStart(0, 0, 0, kernel);
    stitcher.processShaderdata(shaderdata(20, 8));
    stitcher.finishMarkers();

    ASSERT_EQ(capture.markers.size(), 2u);
    const auto& partial = capture.markers[0];
    const auto& recovered = capture.markers[1];
    EXPECT_EQ(partial.marker_flags, NewWave | Incomplete);
    EXPECT_EQ(partial.metadata->payload_count, 3u);
    ASSERT_EQ(partial.payload_count, 1u);
    EXPECT_EQ(partial.payload[0], 123u);
    EXPECT_EQ(recovered.shaderdata.time, 20);
    EXPECT_EQ(recovered.marker_flags, NewWave);
}

TEST_F(MarkerStreamTest, PayloadDeclarationsAreFramedUpToTheBatchLimit)
{
    service->add(7, "P:1:largest\nR:1:extra_payload_count=65536\nP:2:oversized\nR:2:extra_payload_count=65537\n");
    for (bool immediate : {false, true})
    {
        SCOPED_TRACE(testing::Message() << "immediate=" << immediate);
        capture = {};
        Stitcher stitcher(service, captureMarkers, &capture, markerFilter(immediate));
        stitcher.markerWaveStart(0, 0, 0, kernel);
        stitcher.processShaderdata(shaderdata(10, 4));
        for (uint32_t word = 0; word < 65536; ++word) stitcher.processShaderdata(shaderdata(11, word));
        stitcher.markerWaveStart(1, 0, 0, kernel);
        stitcher.processShaderdata(shaderdata(20, 8, 1));
        stitcher.processShaderdata(shaderdata(21, 2u << 2, 1));
        stitcher.finishMarkers();

        ASSERT_EQ(capture.markers.size(), 2u);
        const auto& largest = capture.markers[0];
        EXPECT_EQ(largest.marker_flags, NewWave);
        ASSERT_EQ(largest.payload_count, 65536u);
        EXPECT_EQ(largest.payload[65535], 65535u);
        // Later words cannot be framed, so the wave stops after the header.
        const auto& oversized = capture.markers[1];
        EXPECT_EQ(oversized.shaderdata.time, 20);
        EXPECT_EQ(oversized.marker_flags, NewWave | Incomplete);
        EXPECT_EQ(oversized.payload_count, 0u);
    }
}

TEST_F(MarkerStreamTest, SlotReuseFinishesTheOldPartialPayload)
{
    service->add(7, "P:1:data\nR:1:extra_payload_count=2\nP:2:point\n");
    Stitcher stitcher(service, captureMarkers, &capture, markerFilter());
    stitcher.markerWaveStart(0, 0, 0, kernel);
    stitcher.processShaderdata(shaderdata(10, 4));
    stitcher.processShaderdata(shaderdata(11, 123));
    stitcher.markerWaveStart(0, 0, 0, kernel);
    stitcher.processShaderdata(shaderdata(20, 8));
    stitcher.finishMarkers();

    ASSERT_EQ(capture.markers.size(), 2u);
    EXPECT_EQ(capture.markers[0].marker_flags, NewWave | Incomplete);
    ASSERT_EQ(capture.markers[0].payload_count, 1u);
    EXPECT_EQ(capture.markers[0].payload[0], 123u);
    EXPECT_EQ(capture.markers[1].metadata->marker_id, 2u);
    EXPECT_EQ(capture.markers[1].payload_count, 0u);
    EXPECT_EQ(capture.markers[1].marker_flags, NewWave);
}

TEST_F(MarkerStreamTest, FinishReportsAnEmptyTruncationAndResetsWaveState)
{
    service->add(7, "P:1:data\nR:1:extra_payload_count=2\nP:2:point\n");
    Stitcher stitcher(service, captureMarkers, &capture, markerFilter());
    stitcher.markerWaveStart(0, 0, 0, kernel);
    stitcher.processShaderdata(shaderdata(10, 4));
    stitcher.finishMarkers();
    ASSERT_EQ(capture.markers.size(), 1u);
    EXPECT_EQ(capture.markers[0].marker_flags, NewWave | Incomplete);
    EXPECT_EQ(capture.markers[0].payload_count, 0u);
    EXPECT_EQ(capture.markers[0].payload, nullptr);

    stitcher.processShaderdata(shaderdata(11, 8));
    stitcher.markerWaveStart(0, 0, 0, kernel);
    stitcher.processShaderdata(shaderdata(20, 8));
    stitcher.finishMarkers();
    ASSERT_EQ(capture.markers.size(), 2u);
    EXPECT_EQ(capture.markers[1].shaderdata.time, 20);
    EXPECT_EQ(capture.markers[1].marker_flags, NewWave);
}

TEST_F(MarkerStreamTest, PackedHeadersRetainRawWordsAndDecodeBareExits)
{
    service->add(7, "M:shader_clock_bits=12;shader_clock_shift=4\nW:32\nF:1:function@kernel.cpp:3\n");
    Stitcher stitcher(service, captureMarkers, &capture, markerFilter());
    stitcher.markerWaveStart(0, 0, 0, kernel);
    const uint32_t packed_enter = 0xABCu << 20 | 6u;
    const uint32_t packed_exit = 0xDEFu << 20 | 1u;
    stitcher.processShaderdata(shaderdata(100, packed_enter));
    stitcher.processShaderdata(shaderdata(200, packed_exit));
    stitcher.finishMarkers();

    ASSERT_EQ(capture.markers.size(), 2u);
    EXPECT_EQ(capture.markers[0].shaderdata.value, packed_enter);
    EXPECT_EQ(capture.markers[1].shaderdata.value, packed_exit);
    EXPECT_EQ(capture.markers[0].shaderdata.time, 100);
    EXPECT_EQ(capture.markers[1].shaderdata.time, 200);
    for (const auto& record : capture.markers)
    {
        EXPECT_EQ(record.metadata->marker_id, 1u);
        EXPECT_EQ(record.metadata->wave_size, 32u);
        EXPECT_STREQ(record.metadata->source_location, "kernel.cpp:3");
    }
    EXPECT_EQ(capture.markers[1].marker_flags, Exit);
}

TEST_F(MarkerStreamTest, SparseMaximumMarkerIdDoesNotRequireDenseAllocation)
{
    service->add(7, "P:1073741823:sparse\n");
    Stitcher stitcher(service, captureMarkers, &capture, markerFilter());
    stitcher.markerWaveStart(0, 0, 0, kernel);
    stitcher.processShaderdata(shaderdata(10, 0xFFFFFFFCu));
    stitcher.finishMarkers();

    ASSERT_EQ(capture.markers.size(), 1u);
    EXPECT_EQ(capture.markers[0].metadata->marker_id, 0x3FFFFFFFu);
    EXPECT_STREQ(capture.markers[0].metadata->name, "sparse");
}

TEST_F(MarkerStreamTest, ImmediateHeadersKeepLegacyIdsInPackedCodeObjects)
{
    service->add(7, "M:shader_clock_bits=28;shader_clock_shift=0\nP:1:packed\n");
    Stitcher stitcher(service, captureMarkers, &capture, markerFilter());
    stitcher.markerWaveStart(0, 0, 0, kernel);
    // Numeric ID 17 fits the eight-bit immediate instruction, but its low
    // bits also identify packed marker 1 when this layout's mask is applied.
    stitcher.processShaderdata(shaderdata(10, (17u << 2) | 2u, 0, 0, 0, Imm));
    stitcher.processShaderdata(shaderdata(20, 1, 0, 0, 0, Imm));
    stitcher.finishMarkers();

    ASSERT_EQ(capture.markers.size(), 2u);
    for (const auto& record : capture.markers)
    {
        EXPECT_EQ(record.metadata->marker_id, 17u);
        EXPECT_EQ(record.metadata->marker_kind, ROCPROFILER_THREAD_TRACE_DECODER_MARKER_KIND_UNKNOWN);
        EXPECT_EQ(record.shaderdata.flags, Imm);
    }
    EXPECT_EQ(capture.markers[0].shaderdata.time, 10);
    EXPECT_EQ(capture.markers[1].shaderdata.time, 20);
    EXPECT_EQ(capture.markers[0].marker_flags, NewWave | Enter);
    EXPECT_EQ(capture.markers[1].marker_flags, Exit);
}

TEST_F(MarkerStreamTest, ImmediateHeadersDoNotAffectPackedClockCalibration)
{
    service->add(7, "M:shader_clock_bits=8;shader_clock_shift=4\nP:1:point\n");
    Stitcher stitcher(service, captureMarkers, &capture, markerFilter(true));
    auto decode = [&]()
    {
        stitcher.markerWaveStart(0, 0, 0, kernel);
        stitcher.processShaderdata(shaderdata(1000, (50u << 24) | 4u));
        stitcher.processShaderdata(shaderdata(1200, (61u << 24) | 4u));
        stitcher.processShaderdata(shaderdata(1400, (76u << 24) | 4u));
        // Interpreting this literal's zero high bits as a sample would lower
        // the domain minimum from -16 to -296 and corrupt every packed timestamp.
        stitcher.processShaderdata(shaderdata(4000, 4, 0, 0, 0, Imm));
        stitcher.finishMarkers();
    };

    stitcher.beginMarkerCalibration();
    decode();
    EXPECT_TRUE(capture.markers.empty());
    stitcher.endMarkerCalibration();
    decode();

    ASSERT_EQ(capture.markers.size(), 4u);
    const int64_t arrivals[] = {1000, 1200, 1400};
    const int64_t corrections[] = {1, 25, 0};
    for (size_t i = 0; i < 3; ++i)
    {
        EXPECT_EQ(capture.markers[i].shaderdata.time, arrivals[i] - corrections[i]);
    }
    const auto& constant = capture.markers.back();
    EXPECT_EQ(constant.metadata->marker_id, 1u);
    EXPECT_EQ(constant.shaderdata.flags, Imm);
    EXPECT_EQ(constant.marker_flags, 0u);
    EXPECT_EQ(constant.shaderdata.time, 4000);
}

TEST_F(MarkerStreamTest, ClockCalibrationUsesLaterMinimaAndKeepsPhysicalSimdsIndependent)
{
    service->add(7, "M:shader_clock_bits=8;shader_clock_shift=4\nP:1:point\n");
    Stitcher stitcher(service, captureMarkers, &capture, markerFilter());
    auto decode = [&]()
    {
        stitcher.markerWaveStart(0, 0, 0, kernel);
        stitcher.markerWaveStart(0, 1, 0, kernel);
        stitcher.processShaderdata(shaderdata(1000, (50u << 24) | 4u));
        stitcher.processShaderdata(shaderdata(1100, (20u << 24) | 4u, 0, 1));
        stitcher.processShaderdata(shaderdata(1200, (61u << 24) | 4u));
        stitcher.processShaderdata(shaderdata(1400, (76u << 24) | 4u));
        stitcher.finishMarkers();
    };

    stitcher.beginMarkerCalibration();
    decode();
    EXPECT_TRUE(capture.markers.empty());
    stitcher.endMarkerCalibration();
    decode();

    ASSERT_EQ(capture.markers.size(), 4u);
    const auto& first = capture.markers[0];
    const auto& other_simd = capture.markers[1];
    const auto& second = capture.markers[2];
    const auto& last = capture.markers[3];
    // Relative clock offsets are [0, 24, -16]. Subtract the later minimum
    // and allow the 15-cycle uncertainty introduced by the four-bit shift.
    EXPECT_EQ(first.shaderdata.time, 1000 - 1);
    EXPECT_EQ(second.shaderdata.time, 1200 - 25);
    EXPECT_EQ(last.shaderdata.time, 1400);
    EXPECT_EQ(other_simd.shaderdata.time, 1100);
    EXPECT_EQ(first.shaderdata.value, (50u << 24) | 4u);
}

TEST_F(MarkerStreamTest, InvalidWaveCoordinatesDoNotCorruptValidSlots)
{
    service->add(7, "P:1:point\n");
    Stitcher stitcher(service, captureMarkers, &capture, markerFilter());
    stitcher.markerWaveStart(0, 0, 0, kernel);
    stitcher.markerWaveStart(0, 255, 255, kernel);
    stitcher.processShaderdata(shaderdata(10, 4, 0, 255, 255));
    stitcher.markerWaveEnd(0, 255, 255, kernel);
    stitcher.processShaderdata(shaderdata(20, 4));
    stitcher.finishMarkers();

    ASSERT_EQ(capture.markers.size(), 1u);
    EXPECT_EQ(capture.markers[0].shaderdata.time, 20);
    EXPECT_EQ(capture.markers[0].metadata->marker_id, 1u);
}

TEST_F(MarkerStreamTest, HugeDeclaredPayloadDoesNotPreventDecodingTheNextWave)
{
    service->add(7, "P:1:oversized\nR:1:extra_payload_count=4294967295\nP:2:point\n");
    Stitcher stitcher(service, captureMarkers, &capture, markerFilter());
    stitcher.markerWaveStart(0, 0, 0, kernel);
    stitcher.processShaderdata(shaderdata(10, 4));
    stitcher.processShaderdata(shaderdata(11, 123));
    stitcher.markerWaveStart(0, 0, 0, kernel);
    stitcher.processShaderdata(shaderdata(20, 8));
    stitcher.finishMarkers();

    const auto* next_wave = capture.atTime(20);
    ASSERT_NE(next_wave, nullptr);
    EXPECT_EQ(next_wave->metadata->marker_id, 2u);
    EXPECT_EQ(next_wave->marker_flags, NewWave);
    // Implementations may reject the malformed block or report its observed
    // prefix. Neither policy may treat its enormous declaration as complete.
    for (const auto& record : capture.markers)
        if (record.metadata->marker_id == 1)
        {
            EXPECT_NE(record.marker_flags & Incomplete, 0u);
            EXPECT_EQ(record.metadata->payload_count, std::numeric_limits<uint32_t>::max());
            EXPECT_LE(record.payload_count, 1u);
        }
}

TEST_F(MarkerStreamTest, PooledPayloadPointersRemainValidUntilTheCallback)
{
    service->add(7, "P:1:small\nR:1:extra_payload_count=1\nP:2:large\nR:2:extra_payload_count=130\n");
    {
        Stitcher stitcher(service, captureMarkers, &capture, markerFilter());
        stitcher.markerWaveStart(0, 0, 0, kernel);
        stitcher.processShaderdata(shaderdata(10, 4));
        stitcher.processShaderdata(shaderdata(11, 0x12345678u));
        for (uint32_t batch = 0; batch < 32; ++batch)
        {
            stitcher.processShaderdata(shaderdata(1000 + batch * 200, 8));
            for (uint32_t i = 0; i < 130; ++i)
                stitcher.processShaderdata(shaderdata(1001 + batch * 200 + i, batch * 1000 + i));
        }
        stitcher.finishMarkers();
    }
    service.reset(); // Every field checked below was copied during callbacks.

    ASSERT_EQ(capture.markers.size(), 33u);
    EXPECT_STREQ(capture.markers[0].metadata->name, "small");
    ASSERT_EQ(capture.markers[0].payload_count, 1u);
    EXPECT_EQ(capture.markers[0].payload[0], 0x12345678u);
    for (uint32_t batch = 0; batch < 32; ++batch)
    {
        const auto& record = capture.markers[batch + 1];
        EXPECT_STREQ(record.metadata->name, "large");
        ASSERT_EQ(record.payload_count, 130u);
        for (uint32_t i = 0; i < 130; ++i) EXPECT_EQ(record.payload[i], batch * 1000 + i);
    }
}

TEST_F(MarkerStreamTest, ContextRestoreRecoversTheSavedScope)
{
    service->add(7, "F:1:saved\nF:2:temporary\n");
    Stitcher stitcher(service, captureMarkers, &capture, markerFilter());
    stitcher.markerWaveStart(0, 0, 0, kernel);
    stitcher.processShaderdata(shaderdata(10, 6));
    stitcher.markerContextSave(0, 0, 0);
    stitcher.markerWaveStart(0, 0, 0, kernel);
    stitcher.processShaderdata(shaderdata(20, 10));
    stitcher.processShaderdata(shaderdata(21, 1));
    stitcher.markerWaveEnd(0, 0, 0, kernel);
    stitcher.markerContextRestore(0, 0, 0);
    stitcher.processShaderdata(shaderdata(30, 1));
    stitcher.finishMarkers();

    ASSERT_EQ(capture.markers.size(), 4u);
    EXPECT_EQ(capture.markers[0].shaderdata.time, 10);
    EXPECT_EQ(capture.markers[1].shaderdata.time, 30);
    EXPECT_EQ(capture.markers[2].shaderdata.time, 20);
    EXPECT_EQ(capture.markers[3].shaderdata.time, 21);
    const auto* saved_enter = capture.atTime(10);
    const auto* temporary_enter = capture.atTime(20);
    const auto* saved_exit = capture.atTime(30);
    ASSERT_NE(saved_enter, nullptr);
    ASSERT_NE(temporary_enter, nullptr);
    ASSERT_NE(saved_exit, nullptr);
    EXPECT_STREQ(saved_exit->metadata->name, "saved");
    EXPECT_EQ(saved_exit->metadata->code_object_id, 7u);
    EXPECT_EQ(saved_enter->marker_flags, Enter | NewWave);
    EXPECT_EQ(temporary_enter->marker_flags, Enter | NewWave);
    EXPECT_EQ(temporary_enter->metadata->marker_id, 2u);
    EXPECT_EQ(saved_exit->marker_flags, Exit);
}

TEST_F(MarkerStreamTest, ContextRestoreResumesAPartialPayload)
{
    service->add(7, "P:1:data\nR:1:extra_payload_count=2\nP:2:point\n");
    Stitcher stitcher(service, captureMarkers, &capture, markerFilter());
    stitcher.markerWaveStart(0, 0, 0, kernel);
    stitcher.processShaderdata(shaderdata(10, 4));
    stitcher.processShaderdata(shaderdata(11, 111));
    stitcher.markerContextSave(0, 0, 0);
    stitcher.processShaderdata(shaderdata(12, 999, 0, 0, 0, Priv));
    stitcher.markerWaveStart(0, 0, 0, kernel);
    stitcher.processShaderdata(shaderdata(20, 8));
    stitcher.markerWaveEnd(0, 0, 0, kernel);
    stitcher.markerContextRestore(0, 0, 0);
    stitcher.processShaderdata(shaderdata(30, 222));
    stitcher.finishMarkers();

    ASSERT_EQ(capture.markers.size(), 2u);
    EXPECT_EQ(capture.markers[0].shaderdata.time, 10);
    EXPECT_EQ(capture.markers[1].shaderdata.time, 20);
    const auto* payload = capture.atTime(10);
    const auto* temporary = capture.atTime(20);
    ASSERT_NE(payload, nullptr);
    ASSERT_NE(temporary, nullptr);
    EXPECT_EQ(payload->marker_flags, NewWave);
    ASSERT_EQ(payload->payload_count, 2u);
    EXPECT_EQ(payload->payload[0], 111u);
    EXPECT_EQ(payload->payload[1], 222u);
    EXPECT_EQ(temporary->marker_flags, NewWave);
}

TEST_F(MarkerStreamTest, UnresolvedSavedWaveStaysAheadOfItsTemporaryReplacement)
{
    service->add(7, "F:1:saved\nF:2:temporary\n");
    Stitcher stitcher(service, captureMarkers, &capture, markerFilter());
    stitcher.markerWaveStart(0, 0, 0, {0x200, 0});
    stitcher.processShaderdata(shaderdata(10, 6));
    stitcher.markerContextSave(0, 0, 0);
    stitcher.markerWaveStart(0, 0, 0, kernel);
    stitcher.processShaderdata(shaderdata(20, 10));
    stitcher.processShaderdata(shaderdata(21, 1));
    stitcher.markerWaveEnd(0, 0, 0, kernel);
    stitcher.flushMarkers();
    EXPECT_TRUE(capture.markers.empty());

    stitcher.markerContextRestore(0, 0, 0);
    stitcher.processShaderdata(shaderdata(30, 1));
    CSRegisterHandler registers;
    registers.table_from_start.write().insert(address_range_t{0x100, 0x200, 7});
    stitcher.markerResolveAll(registers);
    stitcher.finishMarkers();

    ASSERT_EQ(capture.markers.size(), 4u);
    const int64_t times[] = {10, 30, 20, 21};
    const uint32_t ids[] = {1, 1, 2, 2};
    const uint32_t flags[] = {Enter | NewWave, Exit, Enter | NewWave, Exit};
    for (size_t i = 0; i < capture.markers.size(); ++i)
    {
        EXPECT_EQ(capture.markers[i].shaderdata.time, times[i]);
        EXPECT_EQ(capture.markers[i].metadata->marker_id, ids[i]);
        EXPECT_EQ(capture.markers[i].marker_flags, flags[i]);
    }
}

TEST_F(MarkerStreamTest, HeldWavesAreDeliveredOnceTheRestoredWaveEnds)
{
    service->add(7, "P:1:saved\nP:2:temporary\nP:3:later\n");
    Stitcher stitcher(service, captureMarkers, &capture, markerFilter(true));
    stitcher.markerWaveStart(0, 0, 0, kernel);
    stitcher.processShaderdata(shaderdata(10, 1u << 2));
    stitcher.markerContextSave(0, 0, 0);
    stitcher.markerWaveStart(0, 0, 0, kernel);
    stitcher.processShaderdata(shaderdata(20, 2u << 2));
    stitcher.markerWaveEnd(0, 0, 0, kernel);
    stitcher.markerContextRestore(0, 0, 0);
    stitcher.processShaderdata(shaderdata(30, 1u << 2));
    stitcher.markerWaveEnd(0, 0, 0, kernel);
    stitcher.markerWaveStart(0, 0, 0, kernel);
    stitcher.processShaderdata(shaderdata(40, 3u << 2));

    // Nothing waits for the end of the buffer once the saved wave is done.
    const int64_t times[] = {10, 30, 20, 40};
    ASSERT_EQ(capture.markers.size(), 4u);
    for (size_t i = 0; i < capture.markers.size(); ++i) EXPECT_EQ(capture.markers[i].shaderdata.time, times[i]);
}

TEST_F(MarkerStreamTest, DelayedAttributionResolvesBeforeTheEndOfTheBuffer)
{
    service->add(7, "P:1:point\n");
    CSRegisterHandler registers;
    Stitcher stitcher(service, captureMarkers, &capture, markerFilter(true));
    stitcher.markerSetRegisters(registers);
    stitcher.markerWaveStart(0, 0, 0, {0x200, 0});
    stitcher.processShaderdata(shaderdata(10, 1u << 2));
    stitcher.markerWaveEnd(0, 0, 0, {0x200, 0});
    stitcher.markerWaveStart(0, 0, 0, kernel);
    stitcher.processShaderdata(shaderdata(20, 1u << 2));
    EXPECT_TRUE(capture.markers.empty());

    registers.table_from_start.write().insert(address_range_t{0x100, 0x200, 7});
    stitcher.markerWaveEnd(0, 0, 0, kernel);
    ASSERT_EQ(capture.markers.size(), 2u);
    EXPECT_EQ(capture.markers[0].shaderdata.time, 10);
    EXPECT_EQ(capture.markers[0].kernel_entry.code_object_id, 7u);
    EXPECT_EQ(capture.markers[1].shaderdata.time, 20);
}

TEST_F(MarkerStreamTest, PacketLossKeepsEndedWavesAwaitingAttribution)
{
    service->add(7, "P:1:point\n");
    CSRegisterHandler registers;
    Stitcher stitcher(service, captureMarkers, &capture, markerFilter(true));
    stitcher.markerSetRegisters(registers);
    stitcher.markerWaveStart(0, 0, 0, {0x200, 0});
    stitcher.processShaderdata(shaderdata(10, 1u << 2));
    stitcher.markerWaveStart(0, 0, 0, kernel);
    stitcher.markerPacketLoss();
    stitcher.processShaderdata(shaderdata(20, 1u << 2)); // The running wave lost its framing.

    registers.table_from_start.write().insert(address_range_t{0x100, 0x200, 7});
    stitcher.markerWaveStart(0, 0, 0, kernel);
    stitcher.processShaderdata(shaderdata(30, 1u << 2));

    ASSERT_EQ(capture.markers.size(), 2u);
    EXPECT_EQ(capture.markers[0].shaderdata.time, 10);
    EXPECT_EQ(capture.markers[0].kernel_entry.code_object_id, 7u);
    EXPECT_EQ(capture.markers[1].shaderdata.time, 30);
}

TEST_F(MarkerStreamTest, RepeatedRestoreKeepsTheResumedWave)
{
    service->add(7, "F:1:saved\n");
    Stitcher stitcher(service, captureMarkers, &capture, markerFilter());
    stitcher.markerWaveStart(0, 0, 0, kernel);
    stitcher.processShaderdata(shaderdata(10, 6));
    stitcher.markerContextSave(0, 0, 0);
    stitcher.markerContextRestore(0, 0, 0);
    stitcher.markerContextRestore(0, 0, 0);
    stitcher.processShaderdata(shaderdata(20, 1));
    stitcher.finishMarkers();

    ASSERT_EQ(capture.markers.size(), 2u);
    EXPECT_EQ(capture.markers[1].shaderdata.time, 20);
    EXPECT_EQ(capture.markers[1].marker_flags, Exit);
    EXPECT_STREQ(capture.markers[1].metadata->name, "saved");
}

TEST_F(MarkerStreamTest, IncompleteMarksEveryMissingDeclaredPayload)
{
    service->add(7, "U:1:data\nR:1:extra_payload_count=1\n");
    Stitcher stitcher(service, captureMarkers, &capture, markerFilter());
    stitcher.markerWaveStart(0, 0, 0, kernel);
    // A bare exit of a payload-bearing scope carries no payload by design.
    stitcher.processShaderdata(shaderdata(10, 6));
    stitcher.processShaderdata(shaderdata(11, 0xAu));
    stitcher.processShaderdata(shaderdata(12, 1));
    // An enter that overflows the scope depth cannot frame its payload.
    for (uint32_t depth = 0; depth <= 4096; ++depth)
    {
        stitcher.processShaderdata(shaderdata(100 + 2 * depth, 6));
        stitcher.processShaderdata(shaderdata(101 + 2 * depth, depth));
    }
    stitcher.finishMarkers();

    const auto* bare_exit = capture.atTime(12);
    ASSERT_NE(bare_exit, nullptr);
    EXPECT_EQ(bare_exit->marker_flags, Exit);
    EXPECT_EQ(bare_exit->metadata->payload_count, 1u);
    EXPECT_EQ(bare_exit->payload_count, 0u);
    const auto* overflow = capture.atTime(100 + 2 * 4096);
    ASSERT_NE(overflow, nullptr);
    EXPECT_EQ(overflow->marker_flags, Enter | Incomplete);
    EXPECT_EQ(overflow->payload_count, 0u);
    EXPECT_EQ(capture.markers.size(), 2u + 4096u + 1u);
}

TEST_F(MarkerStreamTest, PacketLossOfASavedWaveDoesNotDeferAFreshWave)
{
    service->add(7, "P:1:point\n");
    Stitcher stitcher(service, captureMarkers, &capture, markerFilter(true));
    stitcher.markerWaveStart(0, 0, 0, kernel);
    stitcher.processShaderdata(shaderdata(10, 4));
    stitcher.markerContextSave(0, 0, 0);
    stitcher.markerPacketLoss();
    stitcher.markerWaveStart(0, 0, 0, kernel);
    stitcher.processShaderdata(shaderdata(20, 4));

    // The lost saved context can never resume safely; it must not hold later
    // valid generations until a restore or end-of-stream callback.
    ASSERT_EQ(capture.markers.size(), 2u);
    EXPECT_EQ(capture.markers[1].shaderdata.time, 20);
    EXPECT_EQ(capture.markers[1].marker_flags, NewWave);
}

TEST_F(MarkerStreamTest, OversizedLatePayloadInASavedWaveDoesNotDeferAFreshWave)
{
    service->add(7, "U:1:scope\nP:2:oversized\nR:2:extra_payload_count=4294967295\nP:3:point\n");
    Stitcher stitcher(service, captureMarkers, &capture, markerFilter(true));
    stitcher.markerWaveStart(0, 0, 0, kernel);
    stitcher.processShaderdata(shaderdata(10, 6));
    stitcher.markerContextSave(0, 0, 0);
    stitcher.processShaderdata(shaderdata(11, 8));
    stitcher.markerWaveStart(0, 0, 0, kernel);
    stitcher.processShaderdata(shaderdata(20, 12));

    const auto* next_wave = capture.atTime(20);
    ASSERT_NE(next_wave, nullptr);
    EXPECT_EQ(next_wave->marker_flags, NewWave);
    EXPECT_EQ(next_wave->metadata->marker_id, 3u);
    const auto* rejected = capture.atTime(11);
    ASSERT_NE(rejected, nullptr);
    EXPECT_NE(rejected->marker_flags & Incomplete, 0u);
}
} // namespace
