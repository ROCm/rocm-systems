/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Standalone RCCL net plugin loader for benchmarking tools.
//
// Provides a uniform interface for loading net plugins by shared library
// path or by built-in name (e.g. "rocmNetIb", "ncclNetIb").  Uses the
// getNcclNet_vXX() functions from the RCCL source tree
// (src/plugin/net/net_v11.cc … net_v6.cc) to probe for the plugin
// symbol, cascading from newest to oldest version.
//
// The benchmark must be linked against the RCCL net_vXX.cc object files
// (or the RCCL library) that provide these symbols.
//
// Usage:
//   NetPluginHandle h;
//   ncclNet_t* net = netPluginInit(&h, "rocmNetIb");
//   ...
//   netPluginFinalize(&h);

#ifndef NET_PLUGIN_LOADER_H_
#define NET_PLUGIN_LOADER_H_

#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <map>
#include <string>

// The including translation unit must provide the ncclNet_t definition
// (the benchmarks already typedef this from ncclNet_v10_t).

// ── getNcclNet_vXX from RCCL source tree (src/plugin/net/net_vXX.cc) ─

typedef ncclNet_t* getNcclNet_t(void* netPluginLib);

extern getNcclNet_t getNcclNet_v6;
extern getNcclNet_t getNcclNet_v7;
extern getNcclNet_t getNcclNet_v8;
extern getNcclNet_t getNcclNet_v9;
extern getNcclNet_t getNcclNet_v10;
extern getNcclNet_t getNcclNet_v11;

#define NET_VERSION_COUNT 6
static const int kNetVersions[NET_VERSION_COUNT] = { 11, 10, 9, 8, 7, 6 };
static getNcclNet_t* const kGetNcclNet[NET_VERSION_COUNT] = {
    getNcclNet_v11, getNcclNet_v10, getNcclNet_v9,
    getNcclNet_v8,  getNcclNet_v7,  getNcclNet_v6,
};

// ── Plugin handle ────────────────────────────────────────────────────

struct NetPluginHandle {
    void*       libHandle  = nullptr;
    ncclNet_t*  net        = nullptr;
    int         netVersion = 0;
};

// ── Built-in alias table ─────────────────────────────────────────────

std::map<std::string, std::string> kBuiltinPlugins = {
    { "ROCM-NetIb", "rocmNetIb" },
    { "NetIb", "ncclNetIb" },
    { "IB-CAST", "ibCast" },
};

// ---------------------------------------------------------------------------
// netPluginFinalize  –  close the plugin library
// ---------------------------------------------------------------------------
static void netPluginFinalize(NetPluginHandle* handle) {
    if (!handle) return;
    if (handle->libHandle) {
        dlclose(handle->libHandle);
    }
    handle->libHandle = nullptr;
    handle->net = nullptr;
    handle->netVersion = 0;
}
// ---------------------------------------------------------------------------
// netPluginInit  –  open a net plugin and return its ncclNet_t pointer
//
// pluginName is either:
//   - a built-in alias   ("rocmNetIb", "ncclNetIb")
//   - a path / soname    ("librccl-net.so", "./my_plugin.so", …)
//
// Returns the ncclNet_t* on success, nullptr on failure.
// On failure *handle is left in a safe (zeroed) state.
// ---------------------------------------------------------------------------
static ncclNet_t* netPluginInit(NetPluginHandle* handle, const char* pluginName) {
    if (!handle || !pluginName) return nullptr;
    handle->libHandle = nullptr;
    handle->net = nullptr;
    handle->netVersion = 0;

    const char* libPath    = pluginName;
    const char* symbolName = nullptr;
    int dlMode             = RTLD_NOW | RTLD_LOCAL;

    if (kBuiltinPlugins.find(pluginName) != kBuiltinPlugins.end()) {
        libPath    = "libnccl.so.1";
        symbolName = kBuiltinPlugins.find(pluginName)->second.c_str();
        dlMode     = RTLD_NOW | RTLD_GLOBAL;
    }

    handle->libHandle = dlopen(libPath, dlMode);
    if (!handle->libHandle) {
        fprintf(stderr, "netPluginInit: dlopen(%s) failed: %s\n", libPath, dlerror());
        return nullptr;
    }

    if (symbolName) {
        handle->net = (ncclNet_t*)dlsym(handle->libHandle, symbolName);
        if (handle->net) return handle->net;
        fprintf(stderr, "netPluginInit: dlsym(%s) failed: %s\n", symbolName, dlerror());
    } else {
        for (int i = 0; i < NET_VERSION_COUNT; i++) {
            handle->net = kGetNcclNet[i](handle->libHandle);
            if (handle->net) {
                handle->netVersion = kNetVersions[i];
                return handle->net;
            }
        }
        fprintf(stderr, "netPluginInit: no supported ncclNetPlugin symbol found in %s\n", libPath);
    }

    netPluginFinalize(handle);
    return nullptr;
}
#endif // NET_PLUGIN_LOADER_H_
