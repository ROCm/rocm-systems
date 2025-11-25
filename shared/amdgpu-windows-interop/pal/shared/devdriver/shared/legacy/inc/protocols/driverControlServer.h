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

#include "baseProtocolServer.h"
#include "driverControlProtocol.h"

namespace DevDriver
{
    namespace DriverControlProtocol
    {
        DD_STATIC_CONST uint32 kMaxNumGpus = 16;

        typedef Result(*SetDeviceClockModeCallback)(uint32 gpuIndex, DeviceClockMode clockMode, void* pUserdata);

#if GPUOPEN_CLIENT_INTERFACE_MAJOR_VERSION < GPUOPEN_DRIVER_CONTROL_QUERY_CLOCKS_BY_MODE_VERSION
        typedef Result(*QueryDeviceClockCallback)(uint32 gpuIndex, float* pGpuClock, float* pMemClock, void* pUserdata);
        typedef Result(*QueryMaxDeviceClockCallback)(uint32 gpuIndex, float* pMaxGpuClock, float* pMaxMemClock, void* pUserdata);

        struct DeviceClockCallbackInfo
        {
            QueryDeviceClockCallback    queryClockCallback;
            QueryMaxDeviceClockCallback queryMaxClockCallback;
            SetDeviceClockModeCallback  setCallback;
            void*                       pUserdata;
        };
#else
        typedef Result(*QueryDeviceClockCallback)(uint32 gpuIndex, DevDriver::DriverControlProtocol::DeviceClockMode clockMode, float* pGpuClock, float* pMemClock, void* pUserdata);

        struct DeviceClockCallbackInfo
        {
            QueryDeviceClockCallback    queryClockCallback;
            SetDeviceClockModeCallback  setCallback;
            void*                       pUserdata;
        };
#endif

        enum class SessionState;

        class DriverControlServer : public BaseProtocolServer
        {
        public:
            explicit DriverControlServer(IMsgChannel* pMsgChannel);
            ~DriverControlServer();

            void Finalize() override;

            // Session handling functions
            bool AcceptSession(const SharedPointer<ISession>& pSession) override;
            void SessionEstablished(const SharedPointer<ISession>& pSession) override;
            void UpdateSession(const SharedPointer<ISession>& pSession) override;
            void SessionTerminated(const SharedPointer<ISession>& pSession, Result terminationReason) override;

            // Driver state functions
#if GPUOPEN_CLIENT_INTERFACE_MAJOR_VERSION < GPUOPEN_DRIVER_CONTROL_CLEANUP_VERSION
            // These functions just pass through to the new renamed variants to preserve backward compatibility
            void WaitForDriverResume()        { DriverTick(); }
            void StartDeviceInit()            { StartEarlyDeviceInit(); }
            void FinishDriverInitialization() { FinishDeviceInit(); }
#endif
            void StartEarlyDeviceInit();
            void StartLateDeviceInit();
            void FinishDeviceInit();
            void PauseDriver();
            void ResumeDriver();
            void DriverTick();

            // Other public functions
            bool IsDriverInitialized() const;
            DriverStatus QueryDriverStatus();
            void SetNumGpus(uint32 numGpus);
            void SetDeviceClockCallback(const DeviceClockCallbackInfo& deviceClockCallbackInfo);
            uint32 GetNumGpus();
            DeviceClockMode GetDeviceClockMode(uint32 gpuIndex);

            // Sets the client id that's expected to walk us through the driver initialization process.
            // If this isn't set, the server will attempt to find a suitable client itself via broadcast + discovery.
            void SetDriverInitClientId(ClientId clientId) { m_driverInitClientId = clientId; }

            /// Returns true if this driver will be ignored by tools
            bool IsDriverIgnored() const { return m_isIgnored; }

        private:
            void LockData();
            void UnlockData();

            // Private driver state functions
            void AdvanceDriverInitState();
            void WaitForResume();
            bool DiscoverHaltRequests();
            void HandleDriverHalt();
            bool IsHalted() const
            {
                return ((m_driverStatus == DriverStatus::HaltedOnPlatformInit) ||
                        (m_driverStatus == DriverStatus::HaltedOnDeviceInit)   ||
                        (m_driverStatus == DriverStatus::HaltedPostDeviceInit));
            }

            // Protocol message handlers
            SessionState HandlePauseDriverRequest(SizedPayloadContainer& container);
            SessionState HandleResumeDriverRequest(SizedPayloadContainer& container);
            SessionState HandleQueryDeviceClockModeRequest(SizedPayloadContainer& container);
            SessionState HandleSetDeviceClockModeRequest(SizedPayloadContainer& container);
            SessionState HandleQueryDeviceClockRequest(SizedPayloadContainer& container);
            SessionState HandleQueryDeviceClockByModeRequest(SizedPayloadContainer& container);
            SessionState HandleQueryMaxDeviceClockRequest(SizedPayloadContainer& container);
            SessionState HandleQueryNumGpusRequest(SizedPayloadContainer& container);
            SessionState HandleQueryDriverStatusRequest(SizedPayloadContainer& container, const Version sessionVersion);
            SessionState HandleStepDriverRequest(SizedPayloadContainer& container, const Version sessionVersion);
            SessionState HandleIgnoreDriverRequest(SizedPayloadContainer& container);

            Platform::Mutex m_mutex;
            DriverStatus m_driverStatus;
            Platform::Event m_driverResumedEvent;

            uint32 m_numGpus;
            DeviceClockMode m_deviceClockModes[kMaxNumGpus];
            DeviceClockCallbackInfo m_deviceClockCallbackInfo;
            Platform::Atomic m_numSessions;
            Platform::Atomic m_stepCounter;
            bool m_initStepRequested;

            // The client id of the remote client who's responsible for walking us through the driver initialization
            // process.
            ClientId m_driverInitClientId;

            // This value is set to true if a remote tool has indicated that this driver will be ignored
            bool m_isIgnored;

            DD_STATIC_CONST uint32 kBroadcastIntervalInMs = 100;
            DD_STATIC_CONST uint32 kDefaultDriverStartTimeoutMs = 1000;
        };
    }
} // DevDriver
