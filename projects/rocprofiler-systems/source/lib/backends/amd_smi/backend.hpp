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
 * @brief Concept that a wrapper policy type must satisfy.
 *
 * Checked at the point @c backend<T> or @c backend_factory<T> is instantiated,
 * so a mismatch produces a clear error at the template boundary rather than deep
 * inside the template body.
 */
// SDMA process-list methods — only required when the wrapper declares sdma_supported.
template <typename T>
concept sdma_wrapper_contract = requires { typename T::proc_info_t; } &&
                                requires(T t, typename T::processor_handle ph,
                                         std::uint32_t* cp, typename T::proc_info_t* pp) {
                                    {
                                        t.get_gpu_process_list(ph, cp, pp)
                                    } -> std::convertible_to<typename T::status_t>;
                                };

template <typename T>
concept wrapper_types = requires {
    typename T::status_t;
    typename T::version_t;
    typename T::socket_handle;
    typename T::processor_handle;
    typename T::gpu_metrics_t;
    typename T::asic_info_t;
    typename T::memory_type_t;
    typename T::temperature_type_t;
    typename T::temperature_metric_t;
    { T::sdma_supported } -> std::convertible_to<bool>;
    { T::ainic_feature_gate } -> std::convertible_to<bool>;
};

template <typename T>
concept wrapper_constants = requires(typename T::status_t s) {
    { T::STATUS_SUCCESS } -> std::convertible_to<typename T::status_t>;
    { T::MEM_TYPE_VRAM } -> std::convertible_to<typename T::memory_type_t>;
    { T::TEMP_CURRENT } -> std::convertible_to<typename T::temperature_metric_t>;
    {
        T::TEMPERATURE_TYPE_HOTSPOT
    } -> std::convertible_to<typename T::temperature_type_t>;
    { T::TEMPERATURE_TYPE_EDGE } -> std::convertible_to<typename T::temperature_type_t>;
    { T::status_to_string(s) } -> std::convertible_to<std::string>;
};

template <typename T>
concept wrapper_lifecycle = requires(T t, typename T::version_t* vp) {
    { t.init() } -> std::convertible_to<typename T::status_t>;
    { t.shutdown() } -> std::convertible_to<typename T::status_t>;
    { t.get_version(vp) } -> std::convertible_to<typename T::status_t>;
};

template <typename T>
concept wrapper_enumeration =
    requires(T t, std::uint32_t* cp, typename T::socket_handle sh,
             typename T::socket_handle* shp, typename T::processor_handle* php) {
        { t.get_socket_handles(cp, shp) } -> std::convertible_to<typename T::status_t>;
        {
            t.get_processor_handles(sh, cp, php)
        } -> std::convertible_to<typename T::status_t>;
    };

template <typename T>
concept wrapper_gpu_queries =
    requires(T t, typename T::processor_handle ph, typename T::gpu_metrics_t* gmp,
             typename T::asic_info_t* aip, typename T::memory_type_t mt,
             std::uint64_t* u64p, typename T::temperature_type_t tt,
             typename T::temperature_metric_t tm, std::int64_t* i64p) {
        { t.get_metrics_info(ph, gmp) } -> std::convertible_to<typename T::status_t>;
        { t.get_gpu_asic_info(ph, aip) } -> std::convertible_to<typename T::status_t>;
        { t.get_memory_usage(ph, mt, u64p) } -> std::convertible_to<typename T::status_t>;
        {
            t.get_temp_metric(ph, tt, tm, i64p)
        } -> std::convertible_to<typename T::status_t>;
    };

template <typename T>
concept nic_wrapper_types = requires {
    typename T::processor_type;
    typename T::nic_asic_info_t;
    typename T::nic_port_info_t;
    typename T::nic_rdma_devices_info_t;
    typename T::nic_stat_t;
} && requires(typename T::processor_type pt) {
    { T::NIC_PROCESSOR_TYPE } -> std::convertible_to<typename T::processor_type>;
};

template <typename T>
concept nic_wrapper_queries =
    requires(T t, typename T::processor_handle ph, typename T::nic_asic_info_t* nap,
             typename T::nic_port_info_t* npp, typename T::nic_rdma_devices_info_t* ndp,
             std::uint8_t port_idx, std::uint32_t* cp, typename T::nic_stat_t* nsp) {
        { t.get_nic_asic_info(ph, nap) } -> std::convertible_to<typename T::status_t>;
        { t.get_nic_port_info(ph, npp) } -> std::convertible_to<typename T::status_t>;
        { t.get_nic_rdma_dev_info(ph, ndp) } -> std::convertible_to<typename T::status_t>;
        {
            t.get_nic_rdma_port_statistics(ph, port_idx, cp, nsp)
        } -> std::convertible_to<typename T::status_t>;
    };

template <typename T>
concept wrapper_policy =
    wrapper_types<T> && wrapper_constants<T> && wrapper_lifecycle<T> &&
    wrapper_enumeration<T> && wrapper_gpu_queries<T> &&
    (!T::sdma_supported || sdma_wrapper_contract<T>) &&
    (!T::ainic_feature_gate || (nic_wrapper_types<T> && nic_wrapper_queries<T>) );

/**
 * @brief Session-level smart wrapper around an AMD SMI backend policy.
 *
 * Responsibilities:
 *  - Re-export all type aliases from @p Wrapper so upper layers can
 *    reference @c Backend::processor_handle etc. without naming wrapper.
 *  - Manage global lifecycle (initialize / shutdown / version).
 *  - Enumerate processor handles.
 *  - Forward per-device raw calls (taking an explicit handle) so
 *    @c device<Backend> can call through the shared session.
 *
 * Error checking and type conversion live in device, not here.
 *
 * @tparam Wrapper  Raw AMD SMI C API policy (e.g. @c wrapper).
 */
template <wrapper_policy Wrapper>
class backend
{
public:
    // ── Constexpr feature flags — forwarded from Wrapper ─────────────────────
    static constexpr bool sdma_supported     = Wrapper::sdma_supported;
    static constexpr bool ainic_feature_gate = Wrapper::ainic_feature_gate;

    // ── Type aliases — forwarded from Wrapper ─────────────────────────────────
    using status_t             = typename Wrapper::status_t;
    using version_t            = typename Wrapper::version_t;
    using socket_handle        = typename Wrapper::socket_handle;
    using processor_handle     = typename Wrapper::processor_handle;
    using gpu_metrics_t        = typename Wrapper::gpu_metrics_t;
    using asic_info_t          = typename Wrapper::asic_info_t;
    using memory_type_t        = typename Wrapper::memory_type_t;
    using temperature_type_t   = typename Wrapper::temperature_type_t;
    using temperature_metric_t = typename Wrapper::temperature_metric_t;

    static constexpr temperature_metric_t TEMP_CURRENT = Wrapper::TEMP_CURRENT;
    static constexpr temperature_type_t   TEMPERATURE_TYPE_HOTSPOT =
        Wrapper::TEMPERATURE_TYPE_HOTSPOT;
    static constexpr temperature_type_t TEMPERATURE_TYPE_EDGE =
        Wrapper::TEMPERATURE_TYPE_EDGE;

#if defined(AMD_SMI_SDMA_SUPPORTED) && AMD_SMI_SDMA_SUPPORTED == 1
    using proc_info_t = typename Wrapper::proc_info_t;
#endif

#if defined(ROCPROFSYS_BUILD_AINIC) && ROCPROFSYS_BUILD_AINIC == 1
    using nic_asic_info_t         = typename Wrapper::nic_asic_info_t;
    using nic_port_info_t         = typename Wrapper::nic_port_info_t;
    using nic_rdma_devices_info_t = typename Wrapper::nic_rdma_devices_info_t;
    using nic_stat_t              = typename Wrapper::nic_stat_t;
#endif

    // ── Status constants — forwarded ──────────────────────────────────────────
    static constexpr status_t      STATUS_SUCCESS = Wrapper::STATUS_SUCCESS;
    static constexpr memory_type_t MEM_TYPE_VRAM  = Wrapper::MEM_TYPE_VRAM;

    [[nodiscard]] static std::string status_to_string(status_t status)
    {
        return Wrapper::status_to_string(status);
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
        return enumerate_socket_handles(
            [this](socket_handle socket, std::uint32_t* count, processor_handle* procs) {
                return m_amdsmi.get_processor_handles(socket, count, procs);
            },
            "amdsmi_get_processor_handles");
    }

#if defined(ROCPROFSYS_BUILD_AINIC) && ROCPROFSYS_BUILD_AINIC == 1
    [[nodiscard]] std::vector<processor_handle> enumerate_nic_handles()
    {
        return enumerate_socket_handles(
            [this](socket_handle socket, std::uint32_t* count, processor_handle* procs) {
                return m_amdsmi.get_processor_handles_by_type(
                    socket, Wrapper::NIC_PROCESSOR_TYPE, procs, count);
            },
            "amdsmi_get_processor_handles_by_type");
    }
#endif

    // ── Per-device forwarding ─────────────────────────────────────────────────
    // Each method throws std::runtime_error on AMD SMI failure.
    //
    // NOTE: adding a new per-device method requires coordinated changes in 9 places:
    //   wrapper.hpp            – raw C-API forwarder
    //   wrapper_gpu_queries    – extend concept (backend.hpp)
    //   backend<Wrapper>       – add throwing forwarder here
    //   session_gpu_queries    – extend concept (device.hpp)
    //   device<Backend>        – add domain-converting method (device.hpp)
    //   gpu_backend_contract   – extend collector concept (collectors/gpu/device.hpp)
    //   collector device       – add method (collectors/gpu/device.hpp)
    //   mock_wrapper.hpp       – add stub
    //   mock_gpu_backend.hpp   – add stub

    [[nodiscard]] gpu_metrics_t get_metrics_info(processor_handle h) const
    {
        gpu_metrics_t raw{};
        check_status(m_amdsmi.get_metrics_info(h, &raw), "amdsmi_get_gpu_metrics_info");
        return raw;
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

    std::int64_t get_temp_metric(processor_handle handle, temperature_type_t sensor_type,
                                 temperature_metric_t metric)

    {
        std::int64_t temperature{};
        check_status(m_amdsmi.get_temp_metric(handle, sensor_type, metric, &temperature),
                     "amdsmi_get_temp_metric");
        return temperature;
    }

    [[nodiscard]] bool probe_sdma_support(processor_handle h) const noexcept
    {
        if constexpr(sdma_supported)
        {
            std::uint32_t count = 0;
            return m_amdsmi.get_gpu_process_list(
                       h, &count, static_cast<typename Wrapper::proc_info_t*>(nullptr)) ==
                   STATUS_SUCCESS;
        }
        return false;
    }

#if defined(AMD_SMI_SDMA_SUPPORTED) && AMD_SMI_SDMA_SUPPORTED == 1
    [[nodiscard]] std::vector<proc_info_t> get_gpu_process_list(processor_handle h) const
    {
        std::uint32_t count = 0;
        check_status(m_amdsmi.get_gpu_process_list(h, &count, nullptr),
                     "amdsmi_get_gpu_process_list (count)");
        if(count == 0) return {};
        std::vector<proc_info_t> procs(count);
        check_status(m_amdsmi.get_gpu_process_list(h, &count, procs.data()),
                     "amdsmi_get_gpu_process_list (data)");
        return procs;
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
    Wrapper m_amdsmi{};

    static void check_status(status_t status, const char* func)
    {
        if(status == STATUS_SUCCESS) return;
        throw std::runtime_error(std::string(func) +
                                 " failed: " + status_to_string(status));
    }

    template <typename QueryFn>
    [[nodiscard]] std::vector<processor_handle> enumerate_socket_handles(
        QueryFn query, const char* fn_name)
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
 * @brief Factory for creating backend<Wrapper> session instances.
 */
template <wrapper_policy Wrapper>
struct backend_factory
{
    using backend_t = backend<Wrapper>;

    static std::shared_ptr<backend_t> create_backend()
    {
        return std::make_shared<backend_t>();
    }
};

}  // namespace rocprofsys::backends::amd_smi
