// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/amd_smi/sdma_feature.hpp"

#include <concepts>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace rocprofsys::backends::amd_smi
{

/**
 * @brief Concept that an AmdsmiBackend policy type must satisfy.
 *
 * Checked at the point @c backend<T> or @c backend_factory<T> is instantiated,
 * so a mismatch produces a clear error at the template boundary rather than deep
 * inside the template body.
 */
template <typename T>
concept amdsmi_backend_policy = requires {
    typename T::status_t;
    typename T::version_t;
    typename T::socket_handle;
    typename T::processor_handle;
    typename T::gpu_metrics_t;
    typename T::asic_info_t;
    typename T::memory_type_t;
} && requires(T t, typename T::status_t s) {
    { T::STATUS_SUCCESS } -> std::convertible_to<typename T::status_t>;
    { T::MEM_TYPE_VRAM } -> std::convertible_to<typename T::memory_type_t>;
    { T::status_to_string(s) } -> std::convertible_to<std::string>;
    { t.init() } -> std::convertible_to<typename T::status_t>;
    { t.shutdown() } -> std::convertible_to<typename T::status_t>;
};

/**
 * @brief Session-level smart wrapper around an AMD SMI backend policy.
 *
 * Responsibilities:
 *  - Re-export all type aliases from @p AmdsmiBackend so upper layers can
 *    reference @c Backend::processor_handle etc. without naming amdsmi_backend.
 *  - Manage global lifecycle (initialize / shutdown / version).
 *  - Enumerate processor handles.
 *  - Forward per-device raw calls (taking an explicit handle) so
 *    @c backend_proxy<Backend> can call through the shared session.
 *
 * Error checking and type conversion live in backend_proxy, not here.
 *
 * @tparam AmdsmiBackend  Raw AMD SMI C API policy (e.g. amdsmi_backend).
 */
template <amdsmi_backend_policy AmdsmiBackend>
class backend
{
public:
    // ── Type aliases — forwarded from AmdsmiBackend ───────────────────────────
    using status_t         = typename AmdsmiBackend::status_t;
    using version_t        = typename AmdsmiBackend::version_t;
    using socket_handle    = typename AmdsmiBackend::socket_handle;
    using processor_handle = typename AmdsmiBackend::processor_handle;
    using gpu_metrics_t    = typename AmdsmiBackend::gpu_metrics_t;
    using asic_info_t      = typename AmdsmiBackend::asic_info_t;
    using memory_type_t    = typename AmdsmiBackend::memory_type_t;

#if defined(ROCPROFSYS_BUILD_AINIC) && ROCPROFSYS_BUILD_AINIC == 1
    using nic_asic_info_t         = typename AmdsmiBackend::nic_asic_info_t;
    using nic_port_info_t         = typename AmdsmiBackend::nic_port_info_t;
    using nic_rdma_devices_info_t = typename AmdsmiBackend::nic_rdma_devices_info_t;
    using nic_stat_t              = typename AmdsmiBackend::nic_stat_t;
#endif

#if defined(AMD_SMI_SDMA_SUPPORTED) && AMD_SMI_SDMA_SUPPORTED == 1
    using proc_info_t = typename AmdsmiBackend::proc_info_t;
#endif

    // ── Status constants — forwarded ──────────────────────────────────────────
    static constexpr status_t      STATUS_SUCCESS = AmdsmiBackend::STATUS_SUCCESS;
    static constexpr memory_type_t MEM_TYPE_VRAM  = AmdsmiBackend::MEM_TYPE_VRAM;

    [[nodiscard]] static std::string status_to_string(status_t status)
    {
        return AmdsmiBackend::status_to_string(status);
    }

    // ── Constructor ───────────────────────────────────────────────────────────
    backend() noexcept = default;

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    void initialize() { check_status(m_amdsmi.init(), "amdsmi_init"); }

    void shutdown() noexcept { m_amdsmi.shutdown(); }

    [[nodiscard]] version_t get_lib_version()
    {
        version_t ver{};
        check_status(m_amdsmi.get_version(&ver), "amdsmi_get_lib_version");
        return ver;
    }

    // ── Enumeration ───────────────────────────────────────────────────────────

    [[nodiscard]] std::vector<processor_handle> enumerate_gpu_handles()
    {
        return enumerate_handles(
            [this](socket_handle socket, std::uint32_t* count, processor_handle* procs) {
                return m_amdsmi.get_processor_handles(socket, count, procs);
            },
            "amdsmi_get_processor_handles");
    }

#if defined(ROCPROFSYS_BUILD_AINIC) && ROCPROFSYS_BUILD_AINIC == 1
    [[nodiscard]] std::vector<processor_handle> enumerate_nic_handles()
    {
        return enumerate_handles(
            [this](socket_handle socket, std::uint32_t* count, processor_handle* procs) {
                return m_amdsmi.get_processor_handles_by_type(
                    socket, AMDSMI_PROCESSOR_TYPE_AMD_NIC, procs, count);
            },
            "amdsmi_get_processor_handles_by_type");
    }
#endif

    // ── Per-device forwarding ─────────────────────────────────────────────────
    // Each method throws std::runtime_error on AMD SMI failure.

    void get_metrics_info(processor_handle h, gpu_metrics_t* out) const
    {
        check_status(m_amdsmi.get_metrics_info(h, out), "amdsmi_get_gpu_metrics_info");
    }

    void get_gpu_asic_info(processor_handle h, asic_info_t* out) const
    {
        check_status(m_amdsmi.get_gpu_asic_info(h, out), "amdsmi_get_gpu_asic_info");
    }

    void get_memory_usage(processor_handle h, memory_type_t type,
                          std::uint64_t* out) const
    {
        check_status(m_amdsmi.get_memory_usage(h, type, out),
                     "amdsmi_get_gpu_memory_usage");
    }

#if defined(AMD_SMI_SDMA_SUPPORTED) && AMD_SMI_SDMA_SUPPORTED == 1
    // Returns false without throwing when the process list is unavailable (capability
    // probe).
    [[nodiscard]] bool try_get_gpu_process_list(processor_handle h, std::uint32_t* count,
                                                proc_info_t* list) const noexcept
    {
        return m_amdsmi.get_gpu_process_list(h, count, list) == STATUS_SUCCESS;
    }

    void get_gpu_process_list(processor_handle h, std::uint32_t* count,
                              proc_info_t* list) const
    {
        check_status(m_amdsmi.get_gpu_process_list(h, count, list),
                     "amdsmi_get_gpu_process_list");
    }
#endif

#if defined(ROCPROFSYS_BUILD_AINIC) && ROCPROFSYS_BUILD_AINIC == 1
    void get_nic_asic_info(processor_handle h, nic_asic_info_t* out) const
    {
        check_status(m_amdsmi.get_nic_asic_info(h, out), "amdsmi_get_nic_asic_info");
    }

    void get_nic_port_info(processor_handle h, nic_port_info_t* out) const
    {
        check_status(m_amdsmi.get_nic_port_info(h, out), "amdsmi_get_nic_port_info");
    }

    void get_nic_rdma_dev_info(processor_handle h, nic_rdma_devices_info_t* out) const
    {
        check_status(m_amdsmi.get_nic_rdma_dev_info(h, out),
                     "amdsmi_get_nic_rdma_dev_info");
    }

    void get_nic_rdma_port_statistics(processor_handle h, std::uint8_t port_idx,
                                      std::uint32_t* count, nic_stat_t* stats) const
    {
        check_status(m_amdsmi.get_nic_rdma_port_statistics(h, port_idx, count, stats),
                     "amdsmi_get_nic_rdma_port_statistics");
    }
#endif

private:
    AmdsmiBackend m_amdsmi{};

    static void check_status(status_t status, const char* func)
    {
        if(status == STATUS_SUCCESS) return;
        throw std::runtime_error(std::string(func) +
                                 " failed: " + status_to_string(status));
    }

    template <typename QueryFn>
    [[nodiscard]] std::vector<processor_handle> enumerate_handles(QueryFn     query,
                                                                  const char* fn_name)
    {
        std::vector<processor_handle> result;

        std::uint32_t socket_count = 0;
        check_status(m_amdsmi.get_socket_handles(&socket_count, nullptr),
                     "amdsmi_get_socket_handles (count)");

        if(socket_count == 0) return result;

        std::vector<socket_handle> sockets(socket_count);
        check_status(m_amdsmi.get_socket_handles(&socket_count, sockets.data()),
                     "amdsmi_get_socket_handles (data)");

        for(auto socket : sockets)
        {
            std::uint32_t count = 0;
            if(query(socket, &count, nullptr) != STATUS_SUCCESS || count == 0) continue;

            std::vector<processor_handle> procs(count);
            check_status(query(socket, &count, procs.data()),
                         (std::string(fn_name) + " (data)").c_str());

            result.insert(result.end(), procs.begin(), procs.end());
        }

        return result;
    }
};

/**
 * @brief Factory for creating backend<AmdsmiBackend> session instances.
 */
template <amdsmi_backend_policy AmdsmiBackend>
struct backend_factory
{
    using backend_t = backend<AmdsmiBackend>;

    static std::shared_ptr<backend_t> create_backend()
    {
        return std::make_shared<backend_t>();
    }
};

}  // namespace rocprofsys::backends::amd_smi
