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

#include "gpuopen.h"

#define DRIVERCONTROL_PROTOCOL_VERSION 10

#define DRIVERCONTROL_PROTOCOL_MINIMUM_VERSION 1

/*
***********************************************************************************************************************
*| Version | Change Description                                                                                       |
*| ------- | ---------------------------------------------------------------------------------------------------------|
*|  10.0   | Add ability to set clock mode on all adapters.                                                           |
*|  9.0    | Added a feature that allows tools to indicate when they will be ignoring a specific driver.              |
*|  8.0    | Added a new version of the step driver response that contains the current driver status.                 |
*|  7.0    | Corrected a back-compat issue related to the new device clock query code.                                |
*|  6.0    | Added ability to query device clock frequencies for a given clock mode.                                  |
*|  5.0    | Cleaned up the driver facing interface.                                                                  |
*|  4.0    | Added HaltedOnPostDeviceInit state.                                                                      |
*|  3.0    | Added QueryClientInfoRequest support.                                                                    |
*|  2.1    | Added initialization time step functionality.                                                            |
*|  2.0    | Added initialization time driver status values and a terminate driver command.                           |
*|  1.0    | Initial version                                                                                          |
***********************************************************************************************************************
*/

#define DRIVERCONTROL_SET_CLOCKS_ALL_ADAPTERS_VERSION 10
#define DRIVERCONTROL_IGNORE_DRIVER_VERSION 9
#define DRIVERCONTROL_STEP_RETURN_STATUS_VERSION 8
#define DRIVERCONTROL_QUERY_BY_MODE_BACK_COMPAT_VERSION 7
#define DRIVERCONTROL_QUERY_DEVICE_CLOCKS_BY_MODE_VERSION 6
#define DRIVERCONTROL_DRIVER_INTERFACE_CLEANUP_VERSION 5
#define DRIVERCONTROL_HALTEDPOSTDEVICEINIT_VERSION 4
#define DRIVERCONTROL_QUERYCLIENTINFO_VERSION 3
#define DRIVERCONTROL_INITIALIZATION_STATUS_VERSION 2
#define DRIVERCONTROL_INITIAL_VERSION 1

namespace DevDriver
{
    namespace DriverControlProtocol
    {
        ///////////////////////
        // DriverControl Constants
        DD_STATIC_CONST uint32 kLegacyDriverControlPayloadSize = 16;

        ///////////////////////
        // DriverControl Protocol
        enum struct DriverControlMessage : MessageCode
        {
            Unknown = 0,
            PauseDriverRequest,
            PauseDriverResponse,
            ResumeDriverRequest,
            ResumeDriverResponse,
            QueryNumGpusRequest,
            QueryNumGpusResponse,
            QueryDeviceClockModeRequest,
            QueryDeviceClockModeResponse,
            SetDeviceClockModeRequest,
            SetDeviceClockModeResponse,
            QueryDeviceClockRequest,
            QueryDeviceClockResponse,
            QueryMaxDeviceClockRequest,
            QueryMaxDeviceClockResponse,
            QueryDriverStatusRequest,
            QueryDriverStatusResponse,
            StepDriverRequest,
            StepDriverResponse,
            QueryClientInfoRequest,
            QueryClientInfoResponse,
            QueryDeviceClockByModeRequest,
            QueryDeviceClockByModeResponse,
            StepDriverResponseV2,
            IgnoreDriverRequest,
            IgnoreDriverResponse,
            Count
        };

        ///////////////////////
        // DriverControl Types
        enum struct DeviceClockMode : uint32
        {
            Unknown = 0,
            Default,
            Profiling,
            MinimumMemory,
            MinimumEngine,
            Peak,
            Count
        };

#if GPUOPEN_CLIENT_INTERFACE_MAJOR_VERSION < GPUOPEN_DRIVER_CONTROL_CLEANUP_VERSION
        typedef DevDriver::DriverStatus DriverStatus;
#endif

        ///////////////////////
        // DriverControl Payloads
        DD_NETWORK_STRUCT(DriverControlHeader, 4)
        {
            DriverControlMessage command;
            char _padding[3];

            constexpr DriverControlHeader(DriverControlMessage message)
                : command(message)
                , _padding()
            {
            }
        };

        DD_CHECK_SIZE(DriverControlHeader, 4);

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Pause Driver Request/Response
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        DD_NETWORK_STRUCT(PauseDriverRequestPayload, 4)
        {
            DriverControlHeader header;

            constexpr PauseDriverRequestPayload()
                : header(DriverControlMessage::PauseDriverRequest)
            {
            }
        };

        DD_CHECK_SIZE(PauseDriverRequestPayload, sizeof(DriverControlHeader));

        DD_NETWORK_STRUCT(PauseDriverResponsePayload, 4)
        {
            DriverControlHeader header;
            Result result;

            constexpr PauseDriverResponsePayload(Result result)
                : header(DriverControlMessage::PauseDriverResponse)
                , result(result)
            {
            }
        };

        DD_CHECK_SIZE(PauseDriverResponsePayload, sizeof(DriverControlHeader) + 4);

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Resume Driver Request/Response
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        DD_NETWORK_STRUCT(ResumeDriverRequestPayload, 4)
        {
            DriverControlHeader header;

            constexpr ResumeDriverRequestPayload()
                : header(DriverControlMessage::ResumeDriverRequest)
            {
            }
        };

        DD_CHECK_SIZE(ResumeDriverRequestPayload, sizeof(DriverControlHeader));

        DD_NETWORK_STRUCT(ResumeDriverResponsePayload, 4)
        {
            DriverControlHeader header;
            Result result;

            constexpr ResumeDriverResponsePayload(Result result)
                : header(DriverControlMessage::ResumeDriverResponse)
                , result(result)
            {
            }
        };

        DD_CHECK_SIZE(ResumeDriverResponsePayload, sizeof(DriverControlHeader) + 4);

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Query Num Gpus Request/Response
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        DD_NETWORK_STRUCT(QueryNumGpusRequestPayload, 4)
        {
            DriverControlHeader header;

            constexpr QueryNumGpusRequestPayload()
                : header(DriverControlMessage::QueryNumGpusRequest)
            {
            }
        };

        DD_CHECK_SIZE(QueryNumGpusRequestPayload, sizeof(DriverControlHeader));

        DD_NETWORK_STRUCT(QueryNumGpusResponsePayload, 4)
        {
            DriverControlHeader header;
            Result result;
            uint32 numGpus;

            constexpr QueryNumGpusResponsePayload(Result result, uint32 numGpus)
                : header(DriverControlMessage::QueryNumGpusResponse)
                , result(result)
                , numGpus(numGpus)
            {
            }
        };

        DD_CHECK_SIZE(QueryNumGpusResponsePayload, sizeof(DriverControlHeader) + 8);

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Query Device Clock Mode Request/Response
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        DD_NETWORK_STRUCT(QueryDeviceClockModeRequestPayload, 4)
        {
            DriverControlHeader header;
            uint32 gpuIndex;

            constexpr QueryDeviceClockModeRequestPayload(uint32 gpuIndex)
                : header(DriverControlMessage::QueryDeviceClockModeRequest)
                , gpuIndex(gpuIndex)
            {
            }
        };

        DD_CHECK_SIZE(QueryDeviceClockModeRequestPayload, sizeof(DriverControlHeader) + 4);

        DD_NETWORK_STRUCT(QueryDeviceClockModeResponsePayload, 4)
        {
            DriverControlHeader header;
            Result result;
            DeviceClockMode mode;

            constexpr QueryDeviceClockModeResponsePayload(Result result, DeviceClockMode mode)
                : header(DriverControlMessage::QueryDeviceClockModeResponse)
                , result(result)
                , mode(mode)
            {
            }
        };

        DD_CHECK_SIZE(QueryDeviceClockModeResponsePayload, sizeof(DriverControlHeader) + 8);

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Set Device Clock Mode Request/Response
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        DD_NETWORK_STRUCT(SetDeviceClockModeRequestPayload, 4)
        {
            DriverControlHeader header;
            uint32 gpuIndex;
            DeviceClockMode mode;

            constexpr SetDeviceClockModeRequestPayload(uint32 gpuIndex, DeviceClockMode mode)
                : header(DriverControlMessage::SetDeviceClockModeRequest)
                , gpuIndex(gpuIndex)
                , mode(mode)
            {
            }
        };

        DD_CHECK_SIZE(SetDeviceClockModeRequestPayload, sizeof(DriverControlHeader) + 8);

        DD_NETWORK_STRUCT(SetDeviceClockModeResponsePayload, 4)
        {
            DriverControlHeader header;
            Result result;

            constexpr SetDeviceClockModeResponsePayload(Result result)
                : header(DriverControlMessage::SetDeviceClockModeResponse)
                , result(result)
            {
            }
        };

        DD_CHECK_SIZE(SetDeviceClockModeResponsePayload, sizeof(DriverControlHeader) + 4);

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Query Device Clock Request/Response
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        DD_NETWORK_STRUCT(QueryDeviceClockRequestPayload, 4)
        {
            DriverControlHeader header;
            uint32 gpuIndex;

            constexpr QueryDeviceClockRequestPayload(uint32 gpuIndex)
                : header(DriverControlMessage::QueryDeviceClockRequest)
                , gpuIndex(gpuIndex)
            {
            }
        };

        DD_CHECK_SIZE(QueryDeviceClockRequestPayload, sizeof(DriverControlHeader) + 4);

        DD_NETWORK_STRUCT(QueryDeviceClockResponsePayload, 4)
        {
            DriverControlHeader header;
            Result result;
            float gpuClock;
            float memClock;

            constexpr QueryDeviceClockResponsePayload(Result result, float gpuClock, float memClock)
                : header(DriverControlMessage::QueryDeviceClockResponse)
                , result(result)
                , gpuClock(gpuClock)
                , memClock(memClock)
            {
            }
        };

        DD_CHECK_SIZE(QueryDeviceClockResponsePayload, sizeof(DriverControlHeader) + 12);

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Query Device Clock By Mode Request/Response
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        DD_NETWORK_STRUCT(QueryDeviceClockByModeRequestPayload, 4)
        {
            DriverControlHeader header;
            uint32 gpuIndex;
            DeviceClockMode deviceClockMode;

            constexpr QueryDeviceClockByModeRequestPayload(uint32 gpuIndex, DeviceClockMode clockMode)
                : header(DriverControlMessage::QueryDeviceClockByModeRequest)
                , gpuIndex(gpuIndex)
                , deviceClockMode(clockMode)
            {
            }
        };

        DD_CHECK_SIZE(QueryDeviceClockByModeRequestPayload, sizeof(DriverControlHeader) + 8);

        DD_NETWORK_STRUCT(QueryDeviceClockByModeResponsePayload, 4)
        {
            DriverControlHeader header;
            Result result;
            float gpuClock;
            float memClock;

            constexpr QueryDeviceClockByModeResponsePayload(Result result, float gpuClock, float memClock)
                : header(DriverControlMessage::QueryDeviceClockByModeResponse)
                , result(result)
                , gpuClock(gpuClock)
                , memClock(memClock)
            {
            }
        };

        DD_CHECK_SIZE(QueryDeviceClockByModeResponsePayload, sizeof(DriverControlHeader) + 12);

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Query Max Device Clock Request/Response
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        DD_NETWORK_STRUCT(QueryMaxDeviceClockRequestPayload, 4)
        {
            DriverControlHeader header;
            uint32 gpuIndex;

            constexpr QueryMaxDeviceClockRequestPayload(uint32 gpuIndex)
                : header(DriverControlMessage::QueryMaxDeviceClockRequest)
                , gpuIndex(gpuIndex)
            {
            }
        };

        DD_CHECK_SIZE(QueryMaxDeviceClockRequestPayload, sizeof(DriverControlHeader) + 4);

        DD_NETWORK_STRUCT(QueryMaxDeviceClockResponsePayload, 4)
        {
            DriverControlHeader header;
            Result result;
            float maxGpuClock;
            float maxMemClock;

            constexpr QueryMaxDeviceClockResponsePayload(Result result, float maxGpuClock, float maxMemClock)
                : header(DriverControlMessage::QueryMaxDeviceClockResponse)
                , result(result)
                , maxGpuClock(maxGpuClock)
                , maxMemClock(maxMemClock)
            {
            }
        };

        DD_CHECK_SIZE(QueryMaxDeviceClockResponsePayload, sizeof(DriverControlHeader) + 12);

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Query Driver Status Request/Response
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        DD_NETWORK_STRUCT(QueryDriverStatusRequestPayload, 4)
        {
            DriverControlHeader header;

            constexpr QueryDriverStatusRequestPayload()
                : header(DriverControlMessage::QueryDriverStatusRequest)
            {
            }
        };

        DD_CHECK_SIZE(QueryDriverStatusRequestPayload, sizeof(DriverControlHeader));

        DD_NETWORK_STRUCT(QueryDriverStatusResponsePayload, 4)
        {
            DriverControlHeader header;
            DriverStatus status;

            constexpr QueryDriverStatusResponsePayload(DriverStatus status)
                : header(DriverControlMessage::QueryDriverStatusResponse)
                , status(status)
            {
            }
        };

        DD_CHECK_SIZE(QueryDriverStatusResponsePayload, sizeof(DriverControlHeader) + 4);

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Step Driver Request/Response
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        DD_NETWORK_STRUCT(StepDriverRequestPayload, 4)
        {
            DriverControlHeader header;
            uint32 count;

            constexpr StepDriverRequestPayload(uint32 count)
                : header(DriverControlMessage::StepDriverRequest)
                , count(count)
            {
            }
        };

        DD_CHECK_SIZE(StepDriverRequestPayload, sizeof(DriverControlHeader) + 4);

        DD_NETWORK_STRUCT(StepDriverResponsePayload, 4)
        {
            DriverControlHeader header;
            Result result;

            constexpr StepDriverResponsePayload(Result result)
                : header(DriverControlMessage::StepDriverResponse)
                , result(result)
            {
            }
        };

        DD_CHECK_SIZE(StepDriverResponsePayload, sizeof(DriverControlHeader) + 4);

        DD_NETWORK_STRUCT(StepDriverResponsePayloadV2, 4)
        {
            DriverControlHeader header;
            Result result;
            DriverStatus status;

            constexpr StepDriverResponsePayloadV2(Result result, DriverStatus status)
                : header(DriverControlMessage::StepDriverResponseV2)
                , result(result)
                , status(status)
            {
            }
        };

        DD_CHECK_SIZE(StepDriverResponsePayloadV2, sizeof(DriverControlHeader) + 8);

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Query Client Info Request/Response
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        DD_NETWORK_STRUCT(QueryClientInfoRequestPayload, 4)
        {
            DriverControlHeader header;

            constexpr QueryClientInfoRequestPayload()
                : header(DriverControlMessage::QueryClientInfoRequest)
            {
            }
        };

        DD_CHECK_SIZE(QueryClientInfoRequestPayload, sizeof(DriverControlHeader));

        DD_NETWORK_STRUCT(QueryClientInfoResponsePayload, 4)
        {
            DriverControlHeader header;
            ClientInfoStruct clientInfo;

            constexpr QueryClientInfoResponsePayload(const ClientInfoStruct& clientInfo)
                : header(DriverControlMessage::QueryClientInfoResponse)
                , clientInfo(clientInfo)
            {
            }
        };

        DD_CHECK_SIZE(QueryClientInfoResponsePayload, sizeof(DriverControlHeader) + sizeof(ClientInfoStruct));

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Ignore Driver Request/Response
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        DD_NETWORK_STRUCT(IgnoreDriverRequestPayload, 4)
        {
            DriverControlHeader header;

            constexpr IgnoreDriverRequestPayload()
                : header(DriverControlMessage::IgnoreDriverRequest)
            {
            }
        };

        DD_CHECK_SIZE(IgnoreDriverRequestPayload, sizeof(DriverControlHeader));

        DD_NETWORK_STRUCT(IgnoreDriverResponsePayload, 4)
        {
            DriverControlHeader header;

            constexpr IgnoreDriverResponsePayload()
                : header(DriverControlMessage::IgnoreDriverResponse)
            {
            }
        };

        DD_CHECK_SIZE(IgnoreDriverResponsePayload, sizeof(DriverControlHeader));
    }
}
