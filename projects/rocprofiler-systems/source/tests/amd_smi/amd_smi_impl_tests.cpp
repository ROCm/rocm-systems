// Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// MIT License - See LICENSE file for details.

#include "library/amd_smi/amd_smi_impl.hpp"
#include "library/amd_smi/common.hpp"
#include "library/amd_smi/processor.hpp"
#include "library/amd_smi/service.hpp"
#include "mock_driver.hpp"

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <memory>
#include <vector>

#if ROCPROFSYS_USE_ROCM > 0

namespace rocprofsys
{
namespace amd_smi
{
namespace testing
{

struct mock_settings_policy
{
    static device_filter              s_device_filter;
    static smi_metric_options         s_enabled_metrics;
    static device_filter              get_device_filter() { return s_device_filter; }
    static smi_metric_options         get_enabled_metrics() { return s_enabled_metrics; }
    static void                       reset()
    {
        s_device_filter   = { device_selection_mode::ALL, {} };
        s_enabled_metrics = { .value = 0xFFFF };
    }
};

device_filter      mock_settings_policy::s_device_filter   = { device_selection_mode::ALL, {} };
smi_metric_options mock_settings_policy::s_enabled_metrics = { .value = 0xFFFF };

struct mock_perfetto_policy
{
    static std::vector<size_t>                       s_initialized_devices;
    static std::vector<std::pair<size_t, uint64_t>> s_stored_samples;
    static bool                                      s_post_processed;

    static void init_storage(size_t device_index)
    {
        s_initialized_devices.push_back(device_index);
    }

    static void setup_counter_tracks(size_t /*device_index*/,
                                     const smi_metric_options& /*enabled_metrics*/)
    {}

    static void store_sample(size_t device_index, const smi_metrics& /*metrics*/,
                             unsigned long timestamp)
    {
        s_stored_samples.emplace_back(device_index, timestamp);
    }

    static void post_process(size_t /*device_index*/, smi_metric_options /*enabled_metrics*/,
                             smi_metric_options /*supported_metrics*/)
    {
        s_post_processed = true;
    }

    static void reset()
    {
        s_initialized_devices.clear();
        s_stored_samples.clear();
        s_post_processed = false;
    }
};

std::vector<size_t>                       mock_perfetto_policy::s_initialized_devices;
std::vector<std::pair<size_t, uint64_t>> mock_perfetto_policy::s_stored_samples;
bool                                      mock_perfetto_policy::s_post_processed = false;

struct mock_rocpd_policy
{
    static bool s_category_initialized;
    static bool s_tracks_initialized;
    static bool s_pmc_initialized;

    static void initialize_category_metadata() { s_category_initialized = true; }
    static void initialize_smi_tracks_metadata(size_t /*gpu_id*/) { s_tracks_initialized = true; }
    static void initialize_smi_pmc_metadata(size_t /*gpu_id*/) { s_pmc_initialized = true; }
    static void store_sample(size_t /*device_id*/, const smi_metric_options& /*supported*/,
                             const smi_metric_options& /*enabled*/, const smi_metrics& /*metrics*/,
                             unsigned long /*timestamp*/)
    {}

    static void reset()
    {
        s_category_initialized = false;
        s_tracks_initialized   = false;
        s_pmc_initialized      = false;
    }
};

bool mock_rocpd_policy::s_category_initialized = false;
bool mock_rocpd_policy::s_tracks_initialized   = false;
bool mock_rocpd_policy::s_pmc_initialized      = false;

class mock_service
{
public:
    using processor_t        = processor<mock_driver>;
    using processor_ptr_t    = std::shared_ptr<processor_t>;
    using processor_vector_t = std::vector<processor_ptr_t>;
    using filter_func_t = std::function<processor_vector_t(const processor_vector_t&)>;

    static processor_vector_t s_processors;
    static version            s_version;

    const version& get_version() const { return s_version; }

    processor_vector_t get_processors(const filter_func_t& filter = nullptr)
    {
        if(filter)
        {
            return filter(s_processors);
        }
        return s_processors;
    }

    void shutdown() {}

    static void reset()
    {
        s_processors.clear();
        s_version = {};
    }

    static void add_processor(processor_ptr_t proc) { s_processors.push_back(std::move(proc)); }
};

mock_service::processor_vector_t mock_service::s_processors;
version                          mock_service::s_version = { { 1, 0, 0 }, "test" };

struct mock_service_factory
{
    using smi_service        = mock_service;
    using processor_t        = typename smi_service::processor_t;
    using processor_vector_t = typename smi_service::processor_vector_t;

    static std::shared_ptr<smi_service> create_smi_service()
    {
        return std::make_shared<smi_service>();
    }
};

struct test_config
{
    using SmiServiceFactory = mock_service_factory;
    using SettingsApi       = mock_settings_policy;
    using PerfettoApi       = mock_perfetto_policy;
    using RocpdApi          = mock_rocpd_policy;
};

class AmdSmiImplTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mock_settings_policy::reset();
        mock_perfetto_policy::reset();
        mock_rocpd_policy::reset();
        mock_service::reset();

        m_mock_driver = std::make_shared<::testing::NiceMock<mock_driver>>();
        m_mock_driver->set_up_defaults();
    }

    void TearDown() override
    {
        mock_service::reset();
        m_mock_driver.reset();
    }

    std::shared_ptr<::testing::NiceMock<mock_driver>> m_mock_driver;
};

TEST_F(AmdSmiImplTest, SetupInitializesService)
{
    amd_smi_impl<test_config> impl;

    impl.setup();

    EXPECT_EQ(impl.get_processor_count(), 0u);
}

TEST_F(AmdSmiImplTest, SetupWithProcessorsInitializesPerfetto)
{
    auto proc = std::make_shared<processor<mock_driver>>(
        m_mock_driver, reinterpret_cast<amdsmi_processor_handle>(0x1),
        AMDSMI_PROCESSOR_TYPE_AMD_GPU, 0);
    mock_service::add_processor(proc);

    amd_smi_impl<test_config> impl;
    impl.setup();

    EXPECT_EQ(impl.get_processor_count(), 1u);
    EXPECT_EQ(mock_perfetto_policy::s_initialized_devices.size(), 1u);
    EXPECT_EQ(mock_perfetto_policy::s_initialized_devices[0], 0u);
}

TEST_F(AmdSmiImplTest, ConfigInitializesMetadata)
{
    auto proc = std::make_shared<processor<mock_driver>>(
        m_mock_driver, reinterpret_cast<amdsmi_processor_handle>(0x1),
        AMDSMI_PROCESSOR_TYPE_AMD_GPU, 0);
    mock_service::add_processor(proc);

    amd_smi_impl<test_config> impl;
    impl.setup();
    impl.config();

    EXPECT_TRUE(mock_rocpd_policy::s_category_initialized);
    EXPECT_TRUE(mock_rocpd_policy::s_tracks_initialized);
    EXPECT_TRUE(mock_rocpd_policy::s_pmc_initialized);
}

TEST_F(AmdSmiImplTest, SampleStoresData)
{
    auto proc = std::make_shared<processor<mock_driver>>(
        m_mock_driver, reinterpret_cast<amdsmi_processor_handle>(0x1),
        AMDSMI_PROCESSOR_TYPE_AMD_GPU, 0);
    mock_service::add_processor(proc);

    amd_smi_impl<test_config> impl;
    impl.setup();

    uint64_t test_timestamp = 1000000;
    impl.sample([test_timestamp]() { return test_timestamp; });

    EXPECT_EQ(mock_perfetto_policy::s_stored_samples.size(), 1u);
    EXPECT_EQ(mock_perfetto_policy::s_stored_samples[0].first, 0u);
    EXPECT_EQ(mock_perfetto_policy::s_stored_samples[0].second, test_timestamp);
}

TEST_F(AmdSmiImplTest, PostProcessCallsPerfettoPostProcess)
{
    auto proc = std::make_shared<processor<mock_driver>>(
        m_mock_driver, reinterpret_cast<amdsmi_processor_handle>(0x1),
        AMDSMI_PROCESSOR_TYPE_AMD_GPU, 0);
    mock_service::add_processor(proc);

    amd_smi_impl<test_config> impl;
    impl.setup();
    impl.post_process();

    EXPECT_TRUE(mock_perfetto_policy::s_post_processed);
}

TEST_F(AmdSmiImplTest, DeviceFilterNoneReturnsNoProcessors)
{
    auto proc = std::make_shared<processor<mock_driver>>(
        m_mock_driver, reinterpret_cast<amdsmi_processor_handle>(0x1),
        AMDSMI_PROCESSOR_TYPE_AMD_GPU, 0);
    mock_service::add_processor(proc);

    mock_settings_policy::s_device_filter = { device_selection_mode::NONE, {} };

    amd_smi_impl<test_config> impl;
    impl.setup();

    EXPECT_EQ(impl.get_processor_count(), 0u);
}

TEST_F(AmdSmiImplTest, DeviceFilterSpecificReturnsSelectedProcessors)
{
    auto proc0 = std::make_shared<processor<mock_driver>>(
        m_mock_driver, reinterpret_cast<amdsmi_processor_handle>(0x1),
        AMDSMI_PROCESSOR_TYPE_AMD_GPU, 0);
    auto proc1 = std::make_shared<processor<mock_driver>>(
        m_mock_driver, reinterpret_cast<amdsmi_processor_handle>(0x2),
        AMDSMI_PROCESSOR_TYPE_AMD_GPU, 1);
    auto proc2 = std::make_shared<processor<mock_driver>>(
        m_mock_driver, reinterpret_cast<amdsmi_processor_handle>(0x3),
        AMDSMI_PROCESSOR_TYPE_AMD_GPU, 2);

    mock_service::add_processor(proc0);
    mock_service::add_processor(proc1);
    mock_service::add_processor(proc2);

    mock_settings_policy::s_device_filter = { device_selection_mode::SPECIFIC, { 0, 2 } };

    amd_smi_impl<test_config> impl;
    impl.setup();

    EXPECT_EQ(impl.get_processor_count(), 2u);

    const auto& processors = impl.get_processors();
    EXPECT_EQ(processors[0]->get_index(), 0u);
    EXPECT_EQ(processors[1]->get_index(), 2u);
}

TEST_F(AmdSmiImplTest, ShutdownClearsProcessors)
{
    auto proc = std::make_shared<processor<mock_driver>>(
        m_mock_driver, reinterpret_cast<amdsmi_processor_handle>(0x1),
        AMDSMI_PROCESSOR_TYPE_AMD_GPU, 0);
    mock_service::add_processor(proc);

    amd_smi_impl<test_config> impl;
    impl.setup();

    EXPECT_EQ(impl.get_processor_count(), 1u);

    impl.shutdown();

    EXPECT_EQ(impl.get_processor_count(), 0u);
}

}  // namespace testing
}  // namespace amd_smi
}  // namespace rocprofsys

#endif  // ROCPROFSYS_USE_ROCM > 0
