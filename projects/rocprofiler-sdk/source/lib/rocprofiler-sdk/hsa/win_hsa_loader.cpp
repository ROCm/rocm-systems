// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Windows-only HSA symbol loader for rocprofiler-sdk.
//
// On Linux, rocprofiler-sdk links directly against libhsa-runtime64.so for the
// handful of HSA functions it needs at start-up (hsa_init, hsa_iterate_agents,
// hsa_agent_get_info, hsa_amd_agent_iterate_memory_pools,
// hsa_amd_memory_pool_get_info). On Windows there is no standalone hsa-runtime
// DLL: hsa-runtime is statically linked into amdhip64_X.dll, which re-exports
// the HSA C ABI via clr/rocclr/hsa_exports.def.
//
// Per user-locked decision R7 (rocprofiler-register-windows-changes.md and the
// approved plan), do NOT add new exports. Resolve the HSA symbols at runtime
// from amdhip64_X.dll using LoadLibraryA + GetProcAddress.
//
// This file is gated in CMake by `if(WIN32)`.

#if defined(_WIN32)

#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <psapi.h>

#  include <hsa/hsa.h>
#  include <hsa/hsa_ext_amd.h>

#  include <atomic>
#  include <cstdio>
#  include <mutex>

namespace rocprofiler
{
namespace hsa
{
namespace win
{
namespace
{
// The amdhip64 module name varies with HIP major version; probe both the
// versioned name (preferred -- matches what amdhip64.lib links against) and
// the unversioned name as a fallback.
constexpr const char* kCandidateDlls[] = {
    "amdhip64_7.dll",
    "amdhip64_6.dll",
    "amdhip64.dll",
};

HMODULE
load_amdhip64()
{
    // First try modules already loaded into the process (the typical case
    // when the application has already opened amdhip64).
    HMODULE  modules[1024];
    DWORD    needed = 0;
    HANDLE   proc   = ::GetCurrentProcess();
    if(::EnumProcessModules(proc, modules, sizeof(modules), &needed))
    {
        const DWORD count = needed / sizeof(HMODULE);
        for(DWORD i = 0; i < count; ++i)
        {
            char name[MAX_PATH] = {0};
            if(::GetModuleBaseNameA(proc, modules[i], name, sizeof(name)) > 0)
            {
                for(const char* cand : kCandidateDlls)
                {
                    // case-insensitive equal
                    if(_stricmp(name, cand) == 0) return modules[i];
                }
            }
        }
    }

    // Fallback: explicit LoadLibraryA. Search PATH and the application
    // directory.
    for(const char* cand : kCandidateDlls)
    {
        HMODULE h = ::LoadLibraryA(cand);
        if(h != nullptr) return h;
    }
    return nullptr;
}

struct hsa_dispatch
{
    decltype(&hsa_init)                            init                          = nullptr;
    decltype(&hsa_iterate_agents)                  iterate_agents                = nullptr;
    decltype(&hsa_agent_get_info)                  agent_get_info                = nullptr;
    decltype(&hsa_amd_agent_iterate_memory_pools)  amd_agent_iterate_memory_pools = nullptr;
    decltype(&hsa_amd_memory_pool_get_info)        amd_memory_pool_get_info      = nullptr;
    HMODULE                                        module                        = nullptr;
    bool                                           ok                            = false;
};

hsa_dispatch&
table()
{
    static hsa_dispatch       _t;
    static std::once_flag     _f;
    std::call_once(_f, []() {
        _t.module = load_amdhip64();
        if(_t.module == nullptr)
        {
            std::fprintf(stderr,
                         "[rocprofiler-sdk][win_hsa_loader] amdhip64 not found in process; "
                         "tried amdhip64_7.dll, amdhip64_6.dll, amdhip64.dll\n");
            return;
        }
#  define _LOAD(field, name)                                                                       \
      _t.field = reinterpret_cast<decltype(_t.field)>(::GetProcAddress(_t.module, name));          \
      if(_t.field == nullptr)                                                                      \
      {                                                                                            \
          std::fprintf(stderr,                                                                     \
                       "[rocprofiler-sdk][win_hsa_loader] amdhip64 missing export '%s'\n",         \
                       name);                                                                      \
          return;                                                                                  \
      }
        _LOAD(init, "hsa_init")
        _LOAD(iterate_agents, "hsa_iterate_agents")
        _LOAD(agent_get_info, "hsa_agent_get_info")
        _LOAD(amd_agent_iterate_memory_pools, "hsa_amd_agent_iterate_memory_pools")
        _LOAD(amd_memory_pool_get_info, "hsa_amd_memory_pool_get_info")
#  undef _LOAD
        _t.ok = true;
    });
    return _t;
}

}  // namespace

bool
ready()
{
    return table().ok;
}

hsa_status_t
init()
{
    auto& t = table();
    return t.ok ? t.init() : HSA_STATUS_ERROR_NOT_INITIALIZED;
}

hsa_status_t
iterate_agents(hsa_status_t (*cb)(hsa_agent_t, void*), void* data)
{
    auto& t = table();
    return t.ok ? t.iterate_agents(cb, data) : HSA_STATUS_ERROR_NOT_INITIALIZED;
}

hsa_status_t
agent_get_info(hsa_agent_t agent, hsa_agent_info_t attr, void* value)
{
    auto& t = table();
    return t.ok ? t.agent_get_info(agent, attr, value) : HSA_STATUS_ERROR_NOT_INITIALIZED;
}

hsa_status_t
amd_agent_iterate_memory_pools(hsa_agent_t agent,
                               hsa_status_t (*cb)(hsa_amd_memory_pool_t, void*),
                               void* data)
{
    auto& t = table();
    return t.ok ? t.amd_agent_iterate_memory_pools(agent, cb, data)
                : HSA_STATUS_ERROR_NOT_INITIALIZED;
}

hsa_status_t
amd_memory_pool_get_info(hsa_amd_memory_pool_t   pool,
                         hsa_amd_memory_pool_info_t attr,
                         void*                     value)
{
    auto& t = table();
    return t.ok ? t.amd_memory_pool_get_info(pool, attr, value)
                : HSA_STATUS_ERROR_NOT_INITIALIZED;
}

}  // namespace win
}  // namespace hsa
}  // namespace rocprofiler

#endif  // _WIN32
