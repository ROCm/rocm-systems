// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// NFR-PORT-3: Non-Linux static_assert gate.
// Any TU that includes this header on a non-Linux platform and attempts to
// instantiate default_sampling_service will trigger a compile-time error.

#if !defined(__linux__)
namespace rocprofsys::sampling
{

template <class Policies>
class sampling_service;

template <class Policies>
struct sampling_service_platform_guard
{
    static_assert(
        sizeof(Policies) == 0,
        "Thread sampling is Linux-only in this build. "
        "Provide platform-specific policy implementations and instantiate explicitly.");
};

}  // namespace rocprofsys::sampling
#endif
