/*
 ***********************************************************************************************************************
 *
 *  Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 *
 **********************************************************************************************************************/

#pragma once

#pragma pack(push)

#include "gpuopen.h"

#define RGP_PROTOCOL_VERSION 11

#define RGP_PROTOCOL_MINIMUM_VERSION 2

/*
***********************************************************************************************************************
*| Version | Change Description                                                                                       |
*| ------- | ---------------------------------------------------------------------------------------------------------|
*|  11.0   | Version bumped to indicate driver support for SE mask only applying to detailed instruction tracing      |
*|  10.0   | Added support for SPM counters and SE masking.                                                           |
*|  9.0    | Decoupled trace parameters from execute trace request.                                                   |
*|  8.0    | Added support for capturing the RGP trace on specific frame or dispatch                                  |
*|         | Added bitfield to control whether driver internal code objects are included in the code object database  |
*|  7.0    | Added support for aborting traces that are still in the pending state on the server.                     |
*|  6.0    | Added support for trace trigger markers.                                                                 |
*|  5.0    | Added support for allow compute presents trace parameter and removed unused clock mode parameter.        |
*|  4.0    | Added support for reporting trace transfer progress.                                                     |
*|  3.0    | Updated TraceParameters struct to allow for specifying profiling clock mode.                             |
*|  2.0    | Add TraceParameters struct and ExecuteTraceRequestPayload so a client can specify trace options.         |
*|  1.0    | Initial version                                                                                          |
***********************************************************************************************************************
*/

#define RGP_DETAILED_SEMASK_VERSION 11
#define RGP_SPM_COUNTERS_VERSION 10
#define RGP_DECOUPLED_TRACE_PARAMETERS 9
#define RGP_FRAME_CAPTURE_VERSION 8
#define RGP_PENDING_ABORT_VERSION 7
#define RGP_TRIGGER_MARKERS_VERSION 6
#define RGP_COMPUTE_PRESENTS_VERSION 5
#define RGP_TRACE_PROGRESS_VERSION 4
#define RGP_PROFILING_CLOCK_MODES_VERSION 3
#define RGP_TRACE_PARAMETERS_VERSION 2
#define RGP_INITIAL_VERSION 1

namespace DevDriver
{
    namespace RGPProtocol
    {
        ///////////////////////
        // RGP Protocol
        enum struct RGPMessage : MessageCode
        {
            Unknown = 0,
            ExecuteTraceRequest,
            TraceDataChunk,
            TraceDataSentinel,
            QueryProfilingStatusRequest,
            QueryProfilingStatusResponse,
            EnableProfilingRequest,
            EnableProfilingResponse,
            TraceDataHeader,
            AbortTrace,
            QueryTraceParametersRequest,
            QueryTraceParametersResponse,
            UpdateTraceParametersRequest,
            UpdateTraceParametersResponse,
            UpdateSpmConfigRequest,
            UpdateSpmConfigData,
            UpdateSpmConfigResponse,
            Count
        };

        // @note: We currently subtract sizeof(uint32) instead of sizeof(RGPMessage) to work around struct packing issues.
        //        The compiler pads out RGPMessage to 4 bytes when it's included in the payload struct. It also pads out
        //        the TraceDataChunk data field to 1000 bytes. This causes the total payload size to be 1004 bytes which is
        //        4 bytes larger than the maximum size allowed.
        DD_STATIC_CONST Size kMaxTraceDataChunkSize = (kMaxPayloadSizeInBytes - sizeof(uint32) - sizeof(uint32));

        ///////////////////////
        // RGP Constants
        const uint32 kMarkerStringLength = 256;
        const uint32 kMaxSpmCountersPerUpdate = 320;

        // Define the number of bits per SPM id value
        constexpr uint32 kSpmBlockIdBits    = 8;
        constexpr uint32 kSpmInstanceIdBits = 12;
        constexpr uint32 kSpmEventIdBits    = 12;

        // Define the max SPM id values based on the number of bits we allocate for them in the network packet
        constexpr uint32 kMaxSpmBlockId    = (1 << kSpmBlockIdBits);
        constexpr uint32 kMaxSpmInstanceId = (1 << kSpmInstanceIdBits);
        constexpr uint32 kMaxSpmEventId    = (1 << kSpmEventIdBits);

        // The application can specify this value for the instance id and it will be expanded into
        // all available instances on the driver side.
        // The counter fields are bit packed when transferred over the network so we need to account for that here
        // rather than simply setting all bits.
        constexpr uint32 kSpmAllInstancesId = (kMaxSpmInstanceId - 1);

        ///////////////////////
        // RGP Types
        DD_NETWORK_STRUCT(TraceDataChunk, 4)
        {
            uint32 dataSize;
            uint8  data[kMaxTraceDataChunkSize];
        };

        DD_CHECK_SIZE(TraceDataChunk, kMaxTraceDataChunkSize + sizeof(int32));

        DD_NETWORK_STRUCT(TraceParameters, 4)
        {
            uint32 gpuMemoryLimitInMb;
            uint32 numPreparationFrames;
            union
            {
                struct
                {
                    uint32 enableInstructionTokens : 1;
                    uint32 reserved : 31;
                };
                uint32 u32All;
            } flags;
        };

        DD_CHECK_SIZE(TraceParameters, 12);

        enum struct ProfilingClockMode : uint32
        {
            Stable = 0,
            Max,
            Normal,
            Count
        };

        DD_NETWORK_STRUCT(TraceParametersV2, 4)
        {
            uint32 gpuMemoryLimitInMb;
            uint32 numPreparationFrames;
            ProfilingClockMode clockMode;
            union
            {
                struct
                {
                    uint32 enableInstructionTokens : 1;
                    uint32 reserved : 31;
                };
                uint32 u32All;
            } flags;
        };

        DD_CHECK_SIZE(TraceParametersV2, 16);

        DD_NETWORK_STRUCT(TraceParametersV3, 4)
        {
            uint32 gpuMemoryLimitInMb;
            uint32 numPreparationFrames;
            union
            {
                struct
                {
                    uint32 enableInstructionTokens : 1;
                    uint32 allowComputePresents    : 1;
                    uint32 reserved : 30;
                };
                uint32 u32All;
            } flags;
        };

        DD_CHECK_SIZE(TraceParametersV3, 12);

        DD_NETWORK_STRUCT(TraceParametersV4, 4)
        {
            uint32 gpuMemoryLimitInMb;
            uint32 numPreparationFrames;
            union
            {
                struct
                {
                    uint32 enableInstructionTokens : 1;
                    uint32 allowComputePresents    : 1;
                    uint32 reserved : 30;
                };
                uint32 u32All;
            } flags;

            // Begin Tag
            uint32 beginTagHigh;
            uint32 beginTagLow;

            // End Tag
            uint32 endTagHigh;
            uint32 endTagLow;

            // Begin/End Marker Strings
            char beginMarker[kMarkerStringLength];
            char endMarker[kMarkerStringLength];
        };

        DD_CHECK_SIZE(TraceParametersV4, 540);

        enum struct CaptureTriggerMode : uint32
        {
            Present = 0,
            Markers,
            Index,
            Count
        };

        DD_NETWORK_STRUCT(TraceParametersV5, 4)
        {
            uint32 gpuMemoryLimitInMb;
            uint32 numPreparationFrames;
            uint32 captureStartIndex;
            uint32 captureStopIndex;
            CaptureTriggerMode captureMode;

            union
            {
                struct
                {
                    uint32 enableInstructionTokens : 1;
                    uint32 allowComputePresents : 1;
                    uint32 captureDriverCodeObjects : 1;
                    uint32 reserved : 29;
                };
                uint32 u32All;
            } flags;

            // Begin Tag
            uint32 beginTagHigh;
            uint32 beginTagLow;

            // End Tag
            uint32 endTagHigh;
            uint32 endTagLow;

            // Begin/End Marker Strings
            char beginMarker[kMarkerStringLength];
            char endMarker[kMarkerStringLength];
        };

        DD_CHECK_SIZE(TraceParametersV5, 552);

        DD_NETWORK_STRUCT(TraceParametersV6, 4)
        {
            uint32 gpuMemoryLimitInMb;
            uint32 numPreparationFrames;
            uint32 captureStartIndex;
            uint32 captureStopIndex;
            CaptureTriggerMode captureMode;

            union
            {
                struct
                {
                    uint32 enableInstructionTokens : 1;
                    uint32 allowComputePresents : 1;
                    uint32 captureDriverCodeObjects : 1;
                    uint32 reserved : 29;
                };
                uint32 u32All;
            } flags;

            // Begin Tag
            uint32 beginTagHigh;
            uint32 beginTagLow;

            // End Tag
            uint32 endTagHigh;
            uint32 endTagLow;

            // Begin/End Marker Strings
            char beginMarker[kMarkerStringLength];
            char endMarker[kMarkerStringLength];

            // Target pipeline hash
            uint32 pipelineHashHi;
            uint32 pipelineHashLo;
        };

        DD_CHECK_SIZE(TraceParametersV6, 560);

        DD_NETWORK_STRUCT(TraceParametersV7, 4)
        {
            uint32             gpuMemoryLimitInMb;
            uint32             numPreparationFrames;
            uint32             captureStartIndex;
            uint32             captureStopIndex;
            CaptureTriggerMode captureMode;

            union
            {
                struct
                {
                    uint32 enableInstructionTokens   : 1;
                    uint32 allowComputePresents      : 1;
                    uint32 captureDriverCodeObjects  : 1;
                    uint32 enableSpm                 : 1;
                    uint32 reserved : 28;
                };
                uint32 u32All;
            } flags;

            // Begin Tag
            uint32 beginTagHigh;
            uint32 beginTagLow;

            // End Tag
            uint32 endTagHigh;
            uint32 endTagLow;

            // Begin/End Marker Strings
            char beginMarker[kMarkerStringLength];
            char endMarker[kMarkerStringLength];

            // Target pipeline hash
            uint32 pipelineHashHi;
            uint32 pipelineHashLo;

            // Shader Engine Mask
            uint32 seMask;
        };

        DD_CHECK_SIZE(TraceParametersV7, 564);

        DD_NETWORK_STRUCT(SpmCounterId, 4)
        {
            uint32 blockId    : kSpmBlockIdBits;
            uint32 instanceId : kSpmInstanceIdBits;
            uint32 eventId    : kSpmEventIdBits;
        };

        DD_CHECK_SIZE(SpmCounterId, 4);

        static_assert(
            kSpmBlockIdBits + kSpmInstanceIdBits + kSpmEventIdBits == 8 * sizeof(SpmCounterId),
            "SpmCounterId is wasting bits");

        enum struct ProfilingStatus : uint32
        {
            NotAvailable = 0,
            Available,
            Enabled,
            Count
        };

        ///////////////////////
        // RGP Payloads

        DD_NETWORK_STRUCT(ExecuteTraceRequestPayload, 4)
        {
            TraceParameters parameters;
        };

        DD_CHECK_SIZE(ExecuteTraceRequestPayload, 12);

        DD_NETWORK_STRUCT(ExecuteTraceRequestPayloadV2, 4)
        {
            TraceParametersV2 parameters;
        };

        DD_CHECK_SIZE(ExecuteTraceRequestPayloadV2, 16);

        DD_NETWORK_STRUCT(ExecuteTraceRequestPayloadV3, 4)
        {
            TraceParametersV3 parameters;
        };

        DD_CHECK_SIZE(ExecuteTraceRequestPayloadV3, 12);

        DD_NETWORK_STRUCT(ExecuteTraceRequestPayloadV4, 4)
        {
            TraceParametersV4 parameters;
        };

        DD_CHECK_SIZE(ExecuteTraceRequestPayloadV4, 540);

        DD_NETWORK_STRUCT(ExecuteTraceRequestPayloadV5, 4)
        {
            TraceParametersV5 parameters;
        };

        DD_CHECK_SIZE(ExecuteTraceRequestPayloadV5, 552);

        DD_NETWORK_STRUCT(TraceDataChunkPayload, 4)
        {
            TraceDataChunk chunk;
        };

        DD_CHECK_SIZE(TraceDataChunkPayload, kMaxTraceDataChunkSize + sizeof(int32));

        DD_NETWORK_STRUCT(TraceDataSentinelPayload, 4)
        {
            Result result;
        };

        DD_CHECK_SIZE(TraceDataSentinelPayload, 4);

        DD_NETWORK_STRUCT(TraceDataHeaderPayload, 4)
        {
            Result result;
            uint32 numChunks;
            uint32 sizeInBytes;
        };

        DD_CHECK_SIZE(TraceDataHeaderPayload, 12);

        DD_NETWORK_STRUCT(QueryProfilingStatusResponsePayload, 4)
        {
            ProfilingStatus status;
        };

        DD_CHECK_SIZE(QueryProfilingStatusResponsePayload, 4);

        DD_NETWORK_STRUCT(EnableProfilingResponsePayload, 4)
        {
            Result result;
        };

        DD_CHECK_SIZE(EnableProfilingResponsePayload, 4);

        DD_NETWORK_STRUCT(QueryTraceParametersResponsePayload, 4)
        {
            Result            result;
            TraceParametersV6 parameters;
        };

        DD_CHECK_SIZE(QueryTraceParametersResponsePayload, 564);

        DD_NETWORK_STRUCT(QueryTraceParametersResponsePayloadV2, 4)
        {
            Result            result;
            TraceParametersV7 parameters;
        };

        DD_CHECK_SIZE(QueryTraceParametersResponsePayloadV2, 568);

        DD_NETWORK_STRUCT(UpdateTraceParametersRequestPayload, 4)
        {
            TraceParametersV6 parameters;
        };

        DD_CHECK_SIZE(UpdateTraceParametersRequestPayload, 560);

        DD_NETWORK_STRUCT(UpdateTraceParametersRequestPayloadV2, 4)
        {
            TraceParametersV7 parameters;
        };

        DD_CHECK_SIZE(UpdateTraceParametersRequestPayloadV2, 564);

        DD_NETWORK_STRUCT(UpdateTraceParametersResponsePayload, 4)
        {
            Result result;
        };

        DD_CHECK_SIZE(UpdateTraceParametersResponsePayload, 4);

        DD_NETWORK_STRUCT(UpdateSpmConfigRequestPayload, 4)
        {
            uint32 sampleFrequency;
            uint32 memoryLimitInMb;
            uint32 numDataPayloads;
        };

        DD_CHECK_SIZE(UpdateSpmConfigRequestPayload, 12);

        DD_NETWORK_STRUCT(UpdateSpmConfigDataPayload, 4)
        {
            uint32       numCounters;
            SpmCounterId counters[kMaxSpmCountersPerUpdate];
        };

        DD_CHECK_SIZE(UpdateSpmConfigDataPayload, 1284);

        DD_NETWORK_STRUCT(UpdateSpmConfigResponsePayload, 4)
        {
            Result result;
        };

        DD_CHECK_SIZE(UpdateSpmConfigResponsePayload, 4);

        DD_NETWORK_STRUCT(RGPPayload, 4)
        {
            RGPMessage  command;
            // pad out to 4 bytes for alignment requirements
            char        padding[3];
            union
            {
                ExecuteTraceRequestPayload            executeTraceRequest;
                ExecuteTraceRequestPayloadV2          executeTraceRequestV2;
                ExecuteTraceRequestPayloadV3          executeTraceRequestV3;
                ExecuteTraceRequestPayloadV4          executeTraceRequestV4;
                ExecuteTraceRequestPayloadV5          executeTraceRequestV5;
                TraceDataChunkPayload                 traceDataChunk;
                TraceDataSentinelPayload              traceDataSentinel;
                TraceDataHeaderPayload                traceDataHeader;
                QueryProfilingStatusResponsePayload   queryProfilingStatusResponse;
                EnableProfilingResponsePayload        enableProfilingStatusResponse;
                QueryTraceParametersResponsePayload   queryTraceParametersResponse;
                QueryTraceParametersResponsePayloadV2 queryTraceParametersResponseV2;
                UpdateTraceParametersRequestPayload   updateTraceParametersRequest;
                UpdateTraceParametersRequestPayloadV2 updateTraceParametersRequestV2;
                UpdateTraceParametersResponsePayload  updateTraceParametersResponse;
                UpdateSpmConfigRequestPayload         updateSpmConfigRequest;
                UpdateSpmConfigDataPayload            updateSpmConfigData;
                UpdateSpmConfigResponsePayload        updateSpmConfigResponse;
            };
        };

        DD_CHECK_SIZE(RGPPayload, kMaxPayloadSizeInBytes);
    }
}

#pragma pack(pop)
