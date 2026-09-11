/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

namespace GinAnvilPluginStubs {

void Reset();

void SetProbeResult(int result);
void SetBootstrapFail(bool fail);
void SetBootstrapNranks(int nranks);
void SetBootstrapIntResult(const int* values, int count);
void SetFactoryCreateFail(bool fail);
void SetFactoryNullHandles(bool nullHandles);
void SetLsaAddrFail(bool fail);
void SetLsaSelfAddr(void* addr);
void SetConnCheckVerifyMissing(bool missing);
void SetConnCheckMissingCalls(int calls);
int GetConnCheckWriteCalls();
int GetConnCheckVerifyCalls();
unsigned long long GetConnCheckWriteStamp(int call);

}  // namespace GinAnvilPluginStubs
