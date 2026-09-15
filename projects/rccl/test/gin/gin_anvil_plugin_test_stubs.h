/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

// size_t below; this header is included first by GinAnvilPlugin_test.cpp and by
// sibling targets built as LANGUAGE CXX, where the HIP compilation prelude that
// would otherwise declare it is absent.
#include <cstddef>

namespace GinAnvilPluginStubs {

void Reset();

void SetProbeResult(int result);
void SetBootstrapFail(bool fail);
void SetBootstrapNranks(int nranks);
void SetFactoryCreateFail(bool fail);
void SetFactoryNullHandles(bool nullHandles);
void SetLsaAddrFail(bool fail);
void SetLsaSelfAddr(void* addr);
void SetUseFabricMem(bool use);
void SetFabricLsaCapable(bool capable);
void SetFabricAddSelfFail(bool fail);
void SetFabricExchangeFail(bool fail);
void SetFabricPeerBase(void* ptr);
void SetFabricPeerStride(size_t stride);
void SetFabricVmmQueryOk(bool ok);
void SetFabricRetainOk(bool ok);

}  // namespace GinAnvilPluginStubs
