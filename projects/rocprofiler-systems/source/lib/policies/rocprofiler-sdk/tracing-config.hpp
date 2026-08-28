// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace rocprofsys::policies::rocprofiler_sdk
{

/// @brief Structural requirements for the SdkBackend parameter of
/// rocprofsys::rocprofiler_sdk::tracing_config. Only covers members tracing_config
/// relies on unconditionally; version-gated extension points (OMPT, ROCDECODE,
/// ROCJPEG, ROCSHMEM, HIPFILE, KFD_*, MEMORY_ALLOCATION, PAGE_MIGRATION) are
/// accessed directly at the call site behind an `if constexpr(T::compile_time_version
/// >= ...)` check, not a `requires` probe — this concept does NOT verify that T
/// actually defines those members for the version it declares. Each SdkBackend
/// implementation must keep `compile_time_version` and its member set in lockstep
/// itself; the production backend (backends/rocprofiler_sdk/backend.hpp) does this by
/// deriving both from the same ROCPROFILER_VERSION preprocessor macro. A backend that
/// breaks this contract still satisfies this concept, but fails with a hard compile
/// error the first time tracing_config instantiates the branch using the missing
/// member, rather than a concept-not-satisfied diagnostic at the call site.
template <typename T>
concept tracing_config_backend =
    requires(std::uint32_t* major, std::uint32_t* minor, std::uint32_t* patch) {
        typename T::callback_tracing_kind_t;
        typename T::buffer_tracing_kind_t;

        { T::compile_time_version } -> std::convertible_to<std::uint32_t>;

        { T::get_version(major, minor, patch) };
        { T::get_callback_tracing_names() };
        { T::get_buffer_tracing_names() };

        {
            T::CALLBACK_TRACING_HSA_CORE_API
        } -> std::convertible_to<typename T::callback_tracing_kind_t>;
        {
            T::CALLBACK_TRACING_HSA_AMD_EXT_API
        } -> std::convertible_to<typename T::callback_tracing_kind_t>;
        {
            T::CALLBACK_TRACING_HSA_IMAGE_EXT_API
        } -> std::convertible_to<typename T::callback_tracing_kind_t>;
        {
            T::CALLBACK_TRACING_HSA_FINALIZE_EXT_API
        } -> std::convertible_to<typename T::callback_tracing_kind_t>;
        {
            T::CALLBACK_TRACING_HIP_RUNTIME_API
        } -> std::convertible_to<typename T::callback_tracing_kind_t>;
        {
            T::CALLBACK_TRACING_HIP_COMPILER_API
        } -> std::convertible_to<typename T::callback_tracing_kind_t>;
        {
            T::CALLBACK_TRACING_MARKER_CORE_API
        } -> std::convertible_to<typename T::callback_tracing_kind_t>;
        {
            T::CALLBACK_TRACING_CODE_OBJECT
        } -> std::convertible_to<typename T::callback_tracing_kind_t>;
        {
            T::CALLBACK_TRACING_RCCL_API
        } -> std::convertible_to<typename T::callback_tracing_kind_t>;

        {
            T::BUFFER_TRACING_HSA_CORE_API
        } -> std::convertible_to<typename T::buffer_tracing_kind_t>;
        {
            T::BUFFER_TRACING_HSA_AMD_EXT_API
        } -> std::convertible_to<typename T::buffer_tracing_kind_t>;
        {
            T::BUFFER_TRACING_HSA_IMAGE_EXT_API
        } -> std::convertible_to<typename T::buffer_tracing_kind_t>;
        {
            T::BUFFER_TRACING_HSA_FINALIZE_EXT_API
        } -> std::convertible_to<typename T::buffer_tracing_kind_t>;
        {
            T::BUFFER_TRACING_HIP_RUNTIME_API
        } -> std::convertible_to<typename T::buffer_tracing_kind_t>;
        {
            T::BUFFER_TRACING_HIP_COMPILER_API
        } -> std::convertible_to<typename T::buffer_tracing_kind_t>;
        {
            T::BUFFER_TRACING_MARKER_CORE_API
        } -> std::convertible_to<typename T::buffer_tracing_kind_t>;
        {
            T::BUFFER_TRACING_KERNEL_DISPATCH
        } -> std::convertible_to<typename T::buffer_tracing_kind_t>;
        {
            T::BUFFER_TRACING_MEMORY_COPY
        } -> std::convertible_to<typename T::buffer_tracing_kind_t>;
        {
            T::BUFFER_TRACING_SCRATCH_MEMORY
        } -> std::convertible_to<typename T::buffer_tracing_kind_t>;
    };

/// @brief External dependencies required by rocprofsys::rocprofiler_sdk::tracing_config:
/// process-state transitions, feature toggles, and settings lookup. Production code
/// satisfies this via rocprofsys::rocprofiler_sdk::default_externals; tests substitute
/// a mock.
template <typename Externals>
concept tracing_config_externals = requires(std::string_view setting_name) {
    typename Externals::ProcessState;
    typename Externals::ProcessState::State;
    {
        Externals::ProcessState::Finalized
    } -> std::convertible_to<typename Externals::ProcessState::State>;
    { Externals::ProcessState::set(Externals::ProcessState::Finalized) };
    { Externals::get_use_rcclp() } -> std::convertible_to<bool>;
    { Externals::get_use_ompt() } -> std::convertible_to<bool>;
    { Externals::get_use_unified_memory_profiling() } -> std::convertible_to<bool>;
    { Externals::get_rocm_domains() } -> std::convertible_to<std::string>;
    {
        Externals::get_setting_value(setting_name)
    } -> std::same_as<std::optional<std::string>>;
};

}  // namespace rocprofsys::policies::rocprofiler_sdk
