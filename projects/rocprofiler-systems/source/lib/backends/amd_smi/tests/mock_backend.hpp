// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/amd_smi/backend.hpp"
#include "core/sdma_feature.hpp"
#include <cstdint>

#include <gmock/gmock.h>

#include <amd_smi/amdsmi.h>

#include <utility>

namespace rocprofsys::backends::amd_smi::testing
{

/**
 * @brief Mock implementation of AMD SMI backend for unit testing.
 *
 * This is the unified mock backend used across all PMC tests. It provides a complete
 * Google Mock implementation of the backend interface with factory pattern support
 * and default behaviors via set_up_defaults().
 *
 * Used by both provider-level tests and device collector tests (aliased as MockBackend
 * in test_device.cpp for compatibility).
 */
class mock_backend
{
public:
    MOCK_METHOD(amdsmi_status_t, init, ());
    MOCK_METHOD(amdsmi_status_t, init, (std::uint64_t init_flags));
    MOCK_METHOD(amdsmi_status_t, shutdown, ());
    MOCK_METHOD(amdsmi_status_t, get_version, (amdsmi_version_t * version));
    MOCK_METHOD(amdsmi_status_t, get_socket_handles,
                (std::uint32_t * socket_count, amdsmi_socket_handle* socket_handles));
    MOCK_METHOD(amdsmi_status_t, get_processor_handles,
                (amdsmi_socket_handle socket_handle, std::uint32_t* processor_count,
                 amdsmi_processor_handle* processor_handles));
#if defined(ROCPROFSYS_BUILD_AINIC) && ROCPROFSYS_BUILD_AINIC == 1
    MOCK_METHOD(amdsmi_status_t, get_processor_handles_by_type,
                (amdsmi_socket_handle socket_handle, processor_type_t processor_type,
                 amdsmi_processor_handle* processor_handles,
                 std::uint32_t*           processor_count));
#endif
    MOCK_METHOD(amdsmi_status_t, get_processor_type,
                (amdsmi_processor_handle processor_handle,
                 processor_type_t*       processor_type));
    MOCK_METHOD(amdsmi_status_t, get_memory_usage,
                (amdsmi_processor_handle processor_handle, amdsmi_memory_type_t type,
                 std::uint64_t* usage));
    MOCK_METHOD(amdsmi_status_t, get_metrics_info,
                (amdsmi_processor_handle processor_handle,
                 amdsmi_gpu_metrics_t*   metrics));
    MOCK_METHOD(amdsmi_status_t, get_gpu_asic_info,
                (amdsmi_processor_handle processor_handle,
                 amdsmi_asic_info_t*     asic_info));

    // SDMA-specific methods (requires AMD SMI >= 26.3)
#if defined(AMD_SMI_SDMA_SUPPORTED) && AMD_SMI_SDMA_SUPPORTED == 1
    MOCK_METHOD(amdsmi_status_t, get_gpu_process_list,
                (amdsmi_processor_handle processor_handle, std::uint32_t* max_processes,
                 amdsmi_proc_info_t* list));
#endif

    /**
     * @brief Set up default mock behaviors for common operations.
     *
     * Configures the mock to return AMDSMI_STATUS_SUCCESS for init, shutdown,
     * get_memory_usage, and get_metrics_info by default. Tests can override
     * these defaults with specific expectations.
     */
    void set_up_defaults()
    {
        using ::testing::_;
        using ::testing::DoAll;
        using ::testing::Return;
        using ::testing::SetArgPointee;

        ON_CALL(*this, init(_)).WillByDefault(Return(AMDSMI_STATUS_SUCCESS));
        ON_CALL(*this, shutdown()).WillByDefault(Return(AMDSMI_STATUS_SUCCESS));
        ON_CALL(*this, get_memory_usage(_, _, _))
            .WillByDefault(Return(AMDSMI_STATUS_SUCCESS));
        ON_CALL(*this, get_metrics_info(_, _))
            .WillByDefault(Return(AMDSMI_STATUS_SUCCESS));
        ON_CALL(*this, get_gpu_asic_info(_, _))
            .WillByDefault(Return(AMDSMI_STATUS_SUCCESS));
    }
};

/**
 * @brief Factory for creating and injecting mock backend instances in tests.
 *
 * This factory allows tests to inject a mock_backend instance that will be
 * used by the code under test. The mock can be configured with expectations
 * and behaviors before being injected via set_mock_backend().
 */
struct mock_backend_factory
{
    using backend_t = mock_backend;

    static std::shared_ptr<backend_t> s_mock_backend;

    /**
     * @brief Create (retrieve) the mock backend instance.
     * @return Shared pointer to the currently set mock backend.
     */
    static std::shared_ptr<backend_t> create_backend() { return s_mock_backend; }

    /**
     * @brief Set the mock backend instance to be used by tests.
     * @param backend Mock backend instance to inject.
     */
    static void set_mock_backend(std::shared_ptr<backend_t> backend)
    {
        s_mock_backend = std::move(backend);
    }
};

/// Global mock backend instance shared across tests
inline std::shared_ptr<mock_backend> mock_backend_factory::s_mock_backend = nullptr;

static_assert(rocprofsys::backends::amd_smi::management_backend<mock_backend>);

}  // namespace rocprofsys::backends::amd_smi::testing
