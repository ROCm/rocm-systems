/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Compiles src/transport/net_ib_cast/init.cc into the including test TU; include it after fakes/param_redirect.h.
// No test host has /sys/class/infiniband/<hca>/device, so init.cc's realpath() of it goes through g_initRealpath,
// which finds nothing until a test installs an answer. NetIbCastResetInitRefCount() puts init.cc's file-static
// netRefCount back to zero.
#pragma once

#include <functional>

static std::function<char*(const char*, char*)> g_initRealpath = [](const char*, char*) -> char* { return nullptr; };
static char* NetIbCastInitRealpath(const char* path, char* resolved) { return g_initRealpath(path, resolved); }

#define realpath NetIbCastInitRealpath
#include NET_IB_CAST_INIT_CC_PATH
#undef realpath

static void NetIbCastResetInitRefCount() { netRefCount = 0; }
