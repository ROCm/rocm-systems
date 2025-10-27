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
#include "msgChannel.h"
#include "msgTransport.h"
#include "protocols/systemProtocols.h"
#include "protocols/typemap.h"

namespace DevDriver
{
    class IProtocolServer;

    // Server Creation Info
    // This struct extends the MessageChannelCreateInfo struct and adds information about the destination host
    // the client will connect to. It additionally allows specifying protocol servers to enable during initialization.
    // See msgChannel.h for a full list of members.
    struct ServerCreateInfo : public MessageChannelCreateInfo
    {
        HostInfo                 connectionInfo;    // Connection information describing how the Server should connect
                                                    // to the message bus.
        ProtocolFlags            servers;           // Set of boolean values indicating which servers should be created
                                                    // during initialization.
    };

    DD_STATIC_CONST uint32 kQueryStatusTimeoutInMs = 50;

    class DevDriverServer
    {
    public:
        static bool IsConnectionAvailable(const HostInfo& hostInfo, uint32 timeout = kQueryStatusTimeoutInMs);

        explicit DevDriverServer(const AllocCb& allocCb, const ServerCreateInfo& createInfo);
        ~DevDriverServer();

        Result Initialize();
        void Finalize();
        void Destroy();

        const AllocCb& GetAllocCb() const { return m_allocCb; }

#if GPUOPEN_CLIENT_INTERFACE_MAJOR_VERSION < GPUOPEN_DRIVER_CONTROL_CLEANUP_VERSION
        // Called by the driver to mark the end of Platform and the start of device initialization.
        // Starting with GPUOPEN_DRIVER_CONTROL_CLEANUP_VERSION the driver should call the driver control
        // functions directly.
        void StartDeviceInit();
#endif

        bool IsConnected() const;
        IMsgChannel* GetMessageChannel() const;

        DriverControlProtocol::DriverControlServer* GetDriverControlServer();
        RGPProtocol::RGPServer* GetRGPServer();
        EventProtocol::EventServer* GetEventServer();
        SettingsURIService::SettingsService* GetSettingsService();
        InfoURIService::InfoService* GetInfoService();

        bool ShouldShowOverlay();

    private:
        Result InitializeProtocols();
        void DestroyProtocols();

        Result RegisterProtocol(Protocol protocol);
        void UnregisterProtocol(Protocol protocol);
        void FinalizeProtocol(Protocol protocol);

        IMsgChannel*     m_pMsgChannel;
        AllocCb          m_allocCb;
        ServerCreateInfo m_createInfo;

        template <Protocol protocol, class ...Args>
        inline Result RegisterProtocol(Args... args);

        template <Protocol protocol>
        inline ProtocolServerType<protocol>* GetServer();
    };

} // DevDriver
