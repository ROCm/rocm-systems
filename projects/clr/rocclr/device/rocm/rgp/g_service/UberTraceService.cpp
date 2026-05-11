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


#include <UberTraceService.h>

namespace UberTrace
{

const DDRpcServerRegisterServiceInfo IService::kServiceInfo = []() -> DDRpcServerRegisterServiceInfo {
    DDRpcServerRegisterServiceInfo info = {};
    info.id                             = 0x63727461;
    info.version.major                  = 0;
    info.version.minor                  = 2;
    info.version.patch                  = 0;
    info.pName                          = "UberTrace";
    info.pDescription                   = "A service that provides generic trace functionality";

    return info;
}();

static DD_RESULT RegisterFunctions(
    DDRpcServer hServer,
    IService* pService)
{
    DD_RESULT result = DD_RESULT_SUCCESS;

    // Register "EnableTracing"
    if (result == DD_RESULT_SUCCESS)
    {
        DDRpcServerRegisterFunctionInfo info = {};
        info.serviceId                       = 0x63727461;
        info.id                              = 0x1;
        info.pName                           = "EnableTracing";
        info.pDescription                    = "Attempts to enable tracing";
        info.pFuncUserdata                   = pService;
        info.pfnFuncCb                       = [](
            const DDRpcServerCallInfo* pCall) -> DD_RESULT
        {
            auto* pService = reinterpret_cast<IService*>(pCall->pUserdata);

            // Execute the service implementation
            return pService->EnableTracing();
        };

        result = ddRpcServerRegisterFunction(hServer, &info);
    }

    // Register "QueryTraceParams"
    if (result == DD_RESULT_SUCCESS)
    {
        DDRpcServerRegisterFunctionInfo info = {};
        info.serviceId                       = 0x63727461;
        info.id                              = 0x2;
        info.pName                           = "QueryTraceParams";
        info.pDescription                    = "Queries the current set of trace parameters";
        info.pFuncUserdata                   = pService;
        info.pfnFuncCb                       = [](
            const DDRpcServerCallInfo* pCall) -> DD_RESULT
        {
            auto* pService = reinterpret_cast<IService*>(pCall->pUserdata);

            // Execute the service implementation
            return pService->QueryTraceParams(*pCall->pWriter);
        };

        result = ddRpcServerRegisterFunction(hServer, &info);
    }

    // Register "ConfigureTraceParams"
    if (result == DD_RESULT_SUCCESS)
    {
        DDRpcServerRegisterFunctionInfo info = {};
        info.serviceId                       = 0x63727461;
        info.id                              = 0x3;
        info.pName                           = "ConfigureTraceParams";
        info.pDescription                    = "Configures the current set of trace parameters";
        info.pFuncUserdata                   = pService;
        info.pfnFuncCb                       = [](
            const DDRpcServerCallInfo* pCall) -> DD_RESULT
        {
            auto* pService = reinterpret_cast<IService*>(pCall->pUserdata);

            // Execute the service implementation
            return pService->ConfigureTraceParams(pCall->pParameterData, pCall->parameterDataSize);
        };

        result = ddRpcServerRegisterFunction(hServer, &info);
    }

    // Register "RequestTrace"
    if (result == DD_RESULT_SUCCESS)
    {
        DDRpcServerRegisterFunctionInfo info = {};
        info.serviceId                       = 0x63727461;
        info.id                              = 0x4;
        info.pName                           = "RequestTrace";
        info.pDescription                    = "Requests execution of a trace";
        info.pFuncUserdata                   = pService;
        info.pfnFuncCb                       = [](
            const DDRpcServerCallInfo* pCall) -> DD_RESULT
        {
            auto* pService = reinterpret_cast<IService*>(pCall->pUserdata);

            // Execute the service implementation
            return pService->RequestTrace();
        };

        result = ddRpcServerRegisterFunction(hServer, &info);
    }

    // Register "CancelTrace"
    if (result == DD_RESULT_SUCCESS)
    {
        DDRpcServerRegisterFunctionInfo info = {};
        info.serviceId                       = 0x63727461;
        info.id                              = 0x5;
        info.pName                           = "CancelTrace";
        info.pDescription                    = "Cancels a previously requested trace before it starts or after it completes";
        info.pFuncUserdata                   = pService;
        info.pfnFuncCb                       = [](
            const DDRpcServerCallInfo* pCall) -> DD_RESULT
        {
            auto* pService = reinterpret_cast<IService*>(pCall->pUserdata);

            // Execute the service implementation
            return pService->CancelTrace();
        };

        result = ddRpcServerRegisterFunction(hServer, &info);
    }

    // Register "CollectTrace"
    if (result == DD_RESULT_SUCCESS)
    {
        DDRpcServerRegisterFunctionInfo info = {};
        info.serviceId                       = 0x63727461;
        info.id                              = 0x6;
        info.pName                           = "CollectTrace";
        info.pDescription                    = "Collects the data created by a previously executed trace";
        info.pFuncUserdata                   = pService;
        info.pfnFuncCb                       = [](
            const DDRpcServerCallInfo* pCall) -> DD_RESULT
        {
            auto* pService = reinterpret_cast<IService*>(pCall->pUserdata);

            // Execute the service implementation
            return pService->CollectTrace(*pCall->pWriter);
        };

        result = ddRpcServerRegisterFunction(hServer, &info);
    }

    return result;
}

DD_RESULT RegisterService(
    DDRpcServer hServer,
    IService* pService
)
{
    // Register the service
    DD_RESULT result = ddRpcServerRegisterService(hServer, &IService::kServiceInfo);

    // Register individual functions
    if (result == DD_RESULT_SUCCESS)
    {
        result = RegisterFunctions(hServer, pService);

        if (result != DD_RESULT_SUCCESS)
        {
            // Unregister the service if registering functions fails
            ddRpcServerUnregisterService(hServer, IService::kServiceInfo.id);
        }
    }

    return result;
}

} // namespace UberTrace
