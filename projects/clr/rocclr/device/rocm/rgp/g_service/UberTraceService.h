/* Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All rights reserved. */

//# ********************************************************************************************************************
//#
//#  WARNING! WARNING! WARNING! WARNING! WARNING! WARNING! WARNING! WARNING! WARNING! WARNING! WARNING! WARNING!
//#
//#  !!! This code has been generated automatically. Do not hand-modify this code. !!!
//#
//#  When changes are needed, please modify the rpc-gen config or source code in DevDriver/rpc-gen.
//#
//#  rpc-gen v0.2.0
//#
//#  WARNING! WARNING! WARNING! WARNING! WARNING! WARNING! WARNING! WARNING! WARNING! WARNING! WARNING! WARNING!
//#
//# ********************************************************************************************************************

#pragma once

#include <ddRpcServer.h>

namespace UberTrace
{

class IService
{
public:
    virtual ~IService() {}

    static const DDRpcServerRegisterServiceInfo kServiceInfo;

    // Attempts to enable tracing
    virtual DD_RESULT EnableTracing() = 0;

    // Queries the current set of trace parameters
    virtual DD_RESULT QueryTraceParams(
        const DDByteWriter& writer
    ) = 0;

    // Configures the current set of trace parameters
    virtual DD_RESULT ConfigureTraceParams(
        const void* pParamBuffer,
        size_t      paramBufferSize
    ) = 0;

    // Requests execution of a trace
    virtual DD_RESULT RequestTrace() = 0;

    // Cancels a previously requested trace before it starts or after it completes
    virtual DD_RESULT CancelTrace() = 0;

    // Collects the data created by a previously executed trace
    virtual DD_RESULT CollectTrace(
        const DDByteWriter& writer
    ) = 0;

protected:
    IService() {}
};

DD_RESULT RegisterService(DDRpcServer hServer, IService* pService);

} // namespace UberTrace
