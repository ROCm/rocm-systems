// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/amd_smi/backend.hpp"
#include "backends/amd_smi/device.hpp"
#include "backends/amd_smi/wrapper.hpp"
#include "core/agent.hpp"
#include "core/collection/scheduler.hpp"
#include "core/collection/source_interface.hpp"
#include "library/pmc/collectors/gpu/device.hpp"
#include "library/pmc/device_providers/amd_smi/provider.hpp"
#include "library/process_sampler.hpp"

#include <rocprofiler-sdk/version.h>

#if defined(ROCPROFSYS_BUILD_HIPFILE)
#    include "backends/hipfile/backend.hpp"
#    include "backends/hipfile/wrapper.hpp"
#    include "library/pmc/device_providers/hipfile/provider.hpp"
#endif

#include <chrono>
#include <memory>
#include <mutex>
#include <vector>

namespace rocprofsys::pmc
{
using scheduler_factory_t = collection::scheduler_factory<>;

using amd_smi_provider_factory_t = device_providers::amd_smi::provider_factory<
    backends::amd_smi::backend_factory<backends::amd_smi::wrapper>>;
using amd_smi_provider_t = amd_smi_provider_factory_t::provider_t;
using amd_smi_device_t   = backends::amd_smi::device<amd_smi_provider_t::backend_t>;
using gpu_device_t       = collectors::gpu::device<amd_smi_device_t>;

#if defined(ROCPROFSYS_BUILD_HIPFILE)
using hipfile_provider_factory_t = device_providers::hipfile::provider_factory<
    backends::hipfile::backend_factory<backends::hipfile::wrapper>>;
using hipfile_provider_t = hipfile_provider_factory_t::provider_t;
#endif

/** The device providers shared by the sources, each created on first use. The sources
 * hold a reference: it has to outlive them. */
class providers
{
public:
    /** Shared by the GPU and NIC sources; null when AMD SMI sampling is off. */
    std::shared_ptr<amd_smi_provider_t> amd_smi();

#if defined(ROCPROFSYS_BUILD_HIPFILE)
    std::shared_ptr<hipfile_provider_t> hipfile();
#endif

    /** The sources must not be polled any more. */
    void shutdown();

    void release();

private:
    std::mutex                          m_mutex;
    std::shared_ptr<amd_smi_provider_t> m_amd_smi;
#if defined(ROCPROFSYS_BUILD_HIPFILE)
    std::shared_ptr<hipfile_provider_t> m_hipfile;
#endif
};

std::unique_ptr<collection::source_interface>
make_gpu_smi_source(std::chrono::nanoseconds interval, providers& shared,
                    scheduler_factory_t make_scheduler);

/** Null when the NIC support is not built. */
std::unique_ptr<collection::source_interface>
make_nic_smi_source(std::chrono::nanoseconds interval, providers& shared,
                    scheduler_factory_t make_scheduler);

/** Null when hipFile support is not built or not enabled. */
std::unique_ptr<collection::source_interface>
make_hipfile_source(std::chrono::nanoseconds interval, providers& shared,
                    scheduler_factory_t make_scheduler);
#if ROCPROFILER_VERSION >= 600
/** Creates the SDK counting contexts, possible only while the SDK is configured. Empty
 * when that failed. */
process_sampler::source_factory
register_gpu_perf_counter_source(const std::vector<std::shared_ptr<agent>>& agent_list);
#endif
}  // namespace rocprofsys::pmc
