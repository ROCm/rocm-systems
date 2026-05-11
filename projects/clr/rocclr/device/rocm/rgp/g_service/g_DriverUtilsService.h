/* Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved. */

//# ********************************************************************************************************************
//#
//#  WARNING! WARNING! WARNING! WARNING! WARNING! WARNING! WARNING! WARNING! WARNING! WARNING! WARNING! WARNING!
//#
//#  !!! This code has been generated automatically. Do not hand-modify this code. !!!
//#
//#  When changes are needed, please modify the rpc-gen config or source code in DevDriver/rpc-gen.
//#
//#  rpc-gen v0.3.0
//#
//#  WARNING! WARNING! WARNING! WARNING! WARNING! WARNING! WARNING! WARNING! WARNING! WARNING! WARNING! WARNING!
//#
//# ********************************************************************************************************************

#pragma once

#include <ddRpcServer.h>

namespace DriverUtils
{

class IDriverUtilsService
{
public:
    virtual ~IDriverUtilsService() {}

    // Informs driver we are collecting trace data
    virtual DD_RESULT EnableTracing() = 0;

    // Informs driver to enable crash analysis mode
    virtual DD_RESULT EnableCrashAnalysisMode() = 0;

    // Queries the driver for extended client info
    virtual DD_RESULT QueryPalDriverInfo(
        const DDByteWriter& writer
    ) = 0;

    // Informs driver to enable different features: Tracing, CrashAnalysis, RT Shader Data Tokens, Debug Vmid
    virtual DD_RESULT EnableDriverFeatures(
        const void* pParamBuffer,
        size_t      paramBufferSize
    ) = 0;

    // Sends a string to PAL to display in the driver overlay
    virtual DD_RESULT SetOverlayString(
        const void* pParamBuffer,
        size_t      paramBufferSize
    ) = 0;

    // Set driver DbgLog's severity level
    virtual DD_RESULT SetDbgLogSeverityLevel(
        const void* pParamBuffer,
        size_t      paramBufferSize
    ) = 0;

    // Set driver DbgLog's origination mask
    virtual DD_RESULT SetDbgLogOriginationMask(
        const void* pParamBuffer,
        size_t      paramBufferSize
    ) = 0;

    // Modify driver DbgLog's origination mask
    virtual DD_RESULT ModifyDbgLogOriginationMask(
        const void* pParamBuffer,
        size_t      paramBufferSize
    ) = 0;



protected:
    IDriverUtilsService() {}
};

DD_RESULT RegisterService(DDRpcServer hServer, IDriverUtilsService* pService);

void UnRegisterService(DDRpcServer hServer);

} // namespace DriverUtils
