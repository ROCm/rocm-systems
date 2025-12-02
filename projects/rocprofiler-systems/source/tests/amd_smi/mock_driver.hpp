// Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// MIT License - See LICENSE file for details.

#pragma once

#include <gmock/gmock.h>

#if ROCPROFSYS_USE_ROCM > 0
#    include <amd_smi/amdsmi.h>
#endif

namespace rocprofsys
{
namespace amd_smi
{
namespace testing
{

#if ROCPROFSYS_USE_ROCM > 0

class mock_driver
{
public:
    MOCK_METHOD(amdsmi_status_t, init, (uint64_t init_flags));
    MOCK_METHOD(amdsmi_status_t, shutdown, ());
    MOCK_METHOD(amdsmi_status_t, get_version, (amdsmi_version_t * version));
    MOCK_METHOD(amdsmi_status_t, get_socket_handles,
                (uint32_t * socket_count, amdsmi_socket_handle* socket_handles));
    MOCK_METHOD(amdsmi_status_t, get_processor_handles,
                (amdsmi_socket_handle socket_handle, uint32_t* processor_count,
                 amdsmi_processor_handle* processor_handles));
    MOCK_METHOD(amdsmi_status_t, get_processor_type,
                (amdsmi_processor_handle processor_handle, processor_type_t* processor_type));
    MOCK_METHOD(amdsmi_status_t, get_activity,
                (amdsmi_processor_handle processor_handle, amdsmi_engine_usage_t* info));
    MOCK_METHOD(amdsmi_status_t, get_temperature_metric,
                (amdsmi_processor_handle processor_handle, amdsmi_temperature_type_t sensor_type,
                 amdsmi_temperature_metric_t metric, int64_t* temperature));
    MOCK_METHOD(amdsmi_status_t, get_power_info,
                (amdsmi_processor_handle processor_handle, amdsmi_power_info_t* info));
    MOCK_METHOD(amdsmi_status_t, get_memory_usage,
                (amdsmi_processor_handle processor_handle, amdsmi_memory_type_t type,
                 uint64_t* usage));
    MOCK_METHOD(amdsmi_status_t, get_metrics_info,
                (amdsmi_processor_handle processor_handle, amdsmi_gpu_metrics_t* metrics));

    void set_up_defaults()
    {
        using ::testing::_;
        using ::testing::Return;

        ON_CALL(*this, init(_)).WillByDefault(Return(AMDSMI_STATUS_SUCCESS));
        ON_CALL(*this, shutdown()).WillByDefault(Return(AMDSMI_STATUS_SUCCESS));
        ON_CALL(*this, get_activity(_, _)).WillByDefault(Return(AMDSMI_STATUS_SUCCESS));
        ON_CALL(*this, get_temperature_metric(_, _, _, _))
            .WillByDefault(Return(AMDSMI_STATUS_SUCCESS));
        ON_CALL(*this, get_power_info(_, _)).WillByDefault(Return(AMDSMI_STATUS_SUCCESS));
        ON_CALL(*this, get_memory_usage(_, _, _)).WillByDefault(Return(AMDSMI_STATUS_SUCCESS));
        ON_CALL(*this, get_metrics_info(_, _)).WillByDefault(Return(AMDSMI_STATUS_SUCCESS));
    }
};

struct mock_driver_factory
{
    using driver_t = mock_driver;

    static std::shared_ptr<driver_t> s_mock_driver;

    static std::shared_ptr<driver_t> create_driver() { return s_mock_driver; }

    static void set_mock_driver(std::shared_ptr<driver_t> driver) { s_mock_driver = driver; }
};

inline std::shared_ptr<mock_driver> mock_driver_factory::s_mock_driver = nullptr;

#endif  // ROCPROFSYS_USE_ROCM > 0

}  // namespace testing
}  // namespace amd_smi
}  // namespace rocprofsys
