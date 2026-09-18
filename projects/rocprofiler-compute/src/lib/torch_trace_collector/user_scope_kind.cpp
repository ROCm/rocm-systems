// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "user_scope_kind.h"

#include "torch_abi.h"

#include <dlfcn.h>

#include <cstdint>
#include <string_view>

namespace
{

// A static data member in PyTorch 2.13 and later, a plain enumerator with no
// symbol in 2.12.
constexpr const char* kCustomSlotProbeSymbol = "_ZN3c1013DebugInfoKind13PRODUCER_INFOE";

// Its address is the collector's slot identity on 2.13 and later.
constexpr std::string_view kRoctxUserScopeKindName = "rocprofiler-compute.user_scope";

}  // namespace

namespace torch_trace_collector::detail
{

bool c10_has_custom_debug_info_slots()
{
    static const bool has_custom_slots = dlsym(RTLD_DEFAULT, kCustomSlotProbeSymbol) != nullptr;
    return has_custom_slots;
}

c10::DebugInfoKind user_scope_kind()
{
    static const c10::DebugInfoKind kind{c10_has_custom_debug_info_slots()
                                             ? reinterpret_cast<std::uint64_t>(&kRoctxUserScopeKindName)
                                             : torch_abi::kLegacyUserScopeKind};
    return kind;
}

}  // namespace torch_trace_collector::detail
