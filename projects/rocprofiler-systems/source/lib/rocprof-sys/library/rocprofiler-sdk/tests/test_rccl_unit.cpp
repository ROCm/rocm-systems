// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "rocprof-sys/library/rocprofiler-sdk/rccl_internal.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <vector>

using namespace rocprofsys::rocprofiler_sdk;

/**
 * @brief Mock PMC registrar for testing with GMock.
 *
 * This mock uses duck typing - it has the same register_gpu_pmc(uint32_t)
 * method signature as the production registrar, enabling template-based DI.
 */
class mock_pmc_registrar
{
public:
    MOCK_METHOD(void, register_gpu_pmc, (uint32_t device_idx));
};

using rccl_gpu_tracking_state_mock = rccl_gpu_tracking_state_t<mock_pmc_registrar>;

/**
 * @brief Test fixture for RCCL GPU tracking tests.
 *
 * Provides common setup and helper methods for creating tracking state
 * instances with or without mock PMC registrars.
 */
class rccl_test : public ::testing::Test
{
protected:
    std::shared_ptr<mock_pmc_registrar> m_mock_registrar;

    void SetUp() override { m_mock_registrar = std::make_shared<mock_pmc_registrar>(); }

    void TearDown() override { m_mock_registrar.reset(); }

    /**
     * @brief Create tracking state with mock PMC registrar.
     * @return Tracking state configured with the test mock.
     */
    rccl_gpu_tracking_state_mock create_tracking_state_with_mock()
    {
        return rccl_gpu_tracking_state_mock(m_mock_registrar);
    }

    /**
     * @brief Create tracking state without PMC registrar (null).
     * @return Tracking state with no PMC registration.
     */
    rccl_gpu_tracking_state_mock create_tracking_state_null()
    {
        return rccl_gpu_tracking_state_mock(nullptr);
    }

    /**
     * @brief Setup mock to expect registration for N GPUs (0 to num_gpus-1).
     * @param num_gpus Number of GPUs to expect registration for.
     */
    void expect_gpu_registrations(uint32_t num_gpus)
    {
        for(uint32_t i = 0; i < num_gpus; ++i)
        {
            EXPECT_CALL(*m_mock_registrar, register_gpu_pmc(i)).Times(1);
        }
    }

    /**
     * @brief Setup mock to expect registration for specific GPU indices.
     * @param gpu_indices Vector of GPU indices to expect registration for.
     */
    void expect_specific_gpu_registrations(const std::vector<uint32_t>& gpu_indices)
    {
        for(auto idx : gpu_indices)
        {
            EXPECT_CALL(*m_mock_registrar, register_gpu_pmc(idx)).Times(1);
        }
    }
};

struct rccl_type_size_param
{
    ncclDataType_t datatype;
    size_t         expected_size;
    const char*    name;
};

class rccl_type_size_test : public ::testing::TestWithParam<rccl_type_size_param>
{};

TEST_P(rccl_type_size_test, returns_correct_size)
{
    auto param = GetParam();
    EXPECT_EQ(rccl_type_size(param.datatype), param.expected_size);
}

INSTANTIATE_TEST_SUITE_P(
    rccl_datatypes, rccl_type_size_test,
    ::testing::Values(rccl_type_size_param{ ncclInt8, 1, "int8" },
                      rccl_type_size_param{ ncclUint8, 1, "uint8" },
                      rccl_type_size_param{ ncclFloat16, 2, "float16" },
                      rccl_type_size_param{ ncclInt32, 4, "int32" },
                      rccl_type_size_param{ ncclUint32, 4, "uint32" },
                      rccl_type_size_param{ ncclFloat32, 4, "float32" },
                      rccl_type_size_param{ ncclInt64, 8, "int64" },
                      rccl_type_size_param{ ncclUint64, 8, "uint64" },
                      rccl_type_size_param{ ncclFloat64, 8, "float64" },
                      rccl_type_size_param{ ncclBfloat16, 2, "bfloat16" }),
    [](const ::testing::TestParamInfo<rccl_type_size_param>& info) {
        return info.param.name;
    });

TEST_F(rccl_test, rccl_type_size_returns_zero_for_invalid_datatype)
{
    const auto INVALID_DATATYPE = static_cast<ncclDataType_t>(100);
    EXPECT_EQ(rccl_type_size(INVALID_DATATYPE), 0u);
}

TEST_F(rccl_test, rccl_type_size_returns_zero_for_negative_datatype)
{
    const auto NEGATIVE_DATATYPE = static_cast<ncclDataType_t>(-1);
    EXPECT_EQ(rccl_type_size(NEGATIVE_DATATYPE), 0u);
}

TEST_F(rccl_test, rccl_type_size_returns_zero_for_max_int_datatype)
{
    const auto MAX_INT_DATATYPE =
        static_cast<ncclDataType_t>(std::numeric_limits<int>::max());
    EXPECT_EQ(rccl_type_size(MAX_INT_DATATYPE), 0u);
}

TEST_F(rccl_test, tracking_state_default_constructor_creates_empty_state)
{
    rccl_gpu_tracking_state_mock state(nullptr);

    EXPECT_FALSE(state.is_registered(0));
    EXPECT_FALSE(state.is_registered(1));
    EXPECT_FALSE(state.is_registered(99));

    EXPECT_EQ(state.get_bytes(0), 0u);
    EXPECT_EQ(state.get_bytes(1), 0u);
}

TEST_F(rccl_test, tracking_state_constructor_with_null_registrar)
{
    rccl_gpu_tracking_state_mock state(nullptr);

    state.register_gpu(0);
    EXPECT_TRUE(state.is_registered(0));
}

TEST_F(rccl_test, tracking_state_constructor_with_mock_registrar)
{
    auto state = create_tracking_state_with_mock();

    EXPECT_FALSE(state.is_registered(0));
    EXPECT_EQ(state.get_bytes(0), 0u);
}

TEST_F(rccl_test, tracking_state_is_registered_returns_false_initially)
{
    auto state = create_tracking_state_null();
    EXPECT_FALSE(state.is_registered(0));
    EXPECT_FALSE(state.is_registered(1));
}

TEST_F(rccl_test, tracking_state_register_gpu_calls_pmc_registrar_once)
{
    EXPECT_CALL(*m_mock_registrar, register_gpu_pmc(0)).Times(1);
    EXPECT_CALL(*m_mock_registrar, register_gpu_pmc(1)).Times(1);

    auto state = create_tracking_state_with_mock();

    state.register_gpu(0);
    state.register_gpu(0);
    state.register_gpu(1);
}

TEST_F(rccl_test, tracking_state_register_gpu_with_null_registrar_no_crash)
{
    auto state = create_tracking_state_null();
    state.register_gpu(0);
    EXPECT_TRUE(state.is_registered(0));
}

TEST_F(rccl_test, tracking_state_is_registered_returns_true_after_register)
{
    auto state = create_tracking_state_null();
    EXPECT_FALSE(state.is_registered(0));
    state.register_gpu(0);
    EXPECT_TRUE(state.is_registered(0));
    EXPECT_FALSE(state.is_registered(1));
}

TEST_F(rccl_test, tracking_state_register_gpu_is_idempotent)
{
    auto state = create_tracking_state_null();
    state.register_gpu(0);
    state.register_gpu(0);
    EXPECT_TRUE(state.is_registered(0));
}

TEST_F(rccl_test, tracking_state_register_gpu_high_index)
{
    EXPECT_CALL(*m_mock_registrar, register_gpu_pmc(15)).Times(1);

    auto state = create_tracking_state_with_mock();
    state.register_gpu(15);

    EXPECT_TRUE(state.is_registered(15));
    EXPECT_FALSE(state.is_registered(0));
}

TEST_F(rccl_test, tracking_state_register_gpu_very_high_index)
{
    EXPECT_CALL(*m_mock_registrar, register_gpu_pmc(1000)).Times(1);

    auto state = create_tracking_state_with_mock();
    state.register_gpu(1000);

    EXPECT_TRUE(state.is_registered(1000));
}

TEST_F(rccl_test, tracking_state_add_bytes_accumulates)
{
    auto state = create_tracking_state_null();
    EXPECT_EQ(state.add_bytes(0, 100), 100u);
    EXPECT_EQ(state.add_bytes(0, 200), 300u);
    EXPECT_EQ(state.add_bytes(0, 50), 350u);
}

TEST_F(rccl_test, tracking_state_add_bytes_with_zero_bytes)
{
    auto state = create_tracking_state_null();
    EXPECT_EQ(state.add_bytes(0, 0), 0u);
    EXPECT_EQ(state.add_bytes(0, 100), 100u);
    EXPECT_EQ(state.add_bytes(0, 0), 100u);
}

TEST_F(rccl_test, tracking_state_add_bytes_large_value)
{
    auto state = create_tracking_state_null();

    constexpr size_t large_value = std::numeric_limits<size_t>::max() / 2;
    EXPECT_EQ(state.add_bytes(0, large_value), large_value);

    EXPECT_EQ(state.get_bytes(0), large_value);
}

TEST_F(rccl_test, tracking_state_get_bytes_returns_zero_for_unknown_device)
{
    auto state = create_tracking_state_null();
    EXPECT_EQ(state.get_bytes(99), 0u);
}

TEST_F(rccl_test, tracking_state_get_bytes_before_any_add)
{
    auto state = create_tracking_state_null();

    for(uint32_t i = 0; i < 10; ++i)
    {
        EXPECT_EQ(state.get_bytes(i), 0u);
    }
}

TEST_F(rccl_test, tracking_state_get_bytes_matches_cumulative)
{
    auto state = create_tracking_state_null();

    (void) state.add_bytes(0, 100);
    (void) state.add_bytes(0, 200);
    (void) state.add_bytes(0, 300);

    EXPECT_EQ(state.get_bytes(0), 600u);
}

TEST_F(rccl_test, tracking_state_multiple_gpus_independent_registration)
{
    expect_gpu_registrations(4);

    auto state = create_tracking_state_with_mock();

    state.register_gpu(2);
    state.register_gpu(0);
    state.register_gpu(3);
    state.register_gpu(1);

    EXPECT_TRUE(state.is_registered(0));
    EXPECT_TRUE(state.is_registered(1));
    EXPECT_TRUE(state.is_registered(2));
    EXPECT_TRUE(state.is_registered(3));

    EXPECT_FALSE(state.is_registered(4));
}

TEST_F(rccl_test, tracking_state_multiple_gpus_byte_isolation)
{
    auto state = create_tracking_state_null();

    (void) state.add_bytes(0, 100);
    (void) state.add_bytes(1, 200);
    (void) state.add_bytes(2, 300);
    (void) state.add_bytes(0, 50);

    EXPECT_EQ(state.get_bytes(0), 150u);
    EXPECT_EQ(state.get_bytes(1), 200u);
    EXPECT_EQ(state.get_bytes(2), 300u);
    EXPECT_EQ(state.get_bytes(3), 0u);
}

TEST_F(rccl_test, tracking_state_multi_gpu_registration_order)
{
    expect_specific_gpu_registrations({ 3, 2, 1, 0 });

    auto state = create_tracking_state_with_mock();

    state.register_gpu(3);
    state.register_gpu(2);
    state.register_gpu(1);
    state.register_gpu(0);

    for(uint32_t i = 0; i < 4; ++i)
    {
        EXPECT_TRUE(state.is_registered(i));
    }
}

TEST_F(rccl_test, tracking_state_16_gpu_system)
{
    expect_gpu_registrations(16);

    auto state = create_tracking_state_with_mock();

    for(uint32_t i = 0; i < 16; ++i)
    {
        state.register_gpu(i);
    }

    for(uint32_t i = 0; i < 16; ++i)
    {
        EXPECT_TRUE(state.is_registered(i));
    }

    EXPECT_FALSE(state.is_registered(16));
}

TEST_F(rccl_test, tracking_state_reset_clears_state)
{
    auto state = create_tracking_state_null();
    state.register_gpu(0);
    (void) state.add_bytes(0, 100);
    EXPECT_TRUE(state.is_registered(0));
    EXPECT_EQ(state.get_bytes(0), 100u);

    state.reset();

    EXPECT_FALSE(state.is_registered(0));
    EXPECT_EQ(state.get_bytes(0), 0u);
}

TEST_F(rccl_test, tracking_state_reset_clears_multiple_gpus)
{
    auto state = create_tracking_state_null();

    for(uint32_t i = 0; i < 4; ++i)
    {
        state.register_gpu(i);
        (void) state.add_bytes(i, 100 * (i + 1));
    }

    state.reset();

    for(uint32_t i = 0; i < 4; ++i)
    {
        EXPECT_FALSE(state.is_registered(i));
        EXPECT_EQ(state.get_bytes(i), 0u);
    }
}

TEST_F(rccl_test, tracking_state_usable_after_reset)
{
    EXPECT_CALL(*m_mock_registrar, register_gpu_pmc(0)).Times(2);

    auto state = create_tracking_state_with_mock();

    state.register_gpu(0);
    (void) state.add_bytes(0, 100);

    state.reset();

    state.register_gpu(0);
    (void) state.add_bytes(0, 200);

    EXPECT_TRUE(state.is_registered(0));
    EXPECT_EQ(state.get_bytes(0), 200u);
}

TEST_F(rccl_test, rccl_event_info_default_initialized)
{
    rccl_event_info info{};
    EXPECT_EQ(info.size, 0u);
    EXPECT_FALSE(info.is_send);
    EXPECT_EQ(info.comm, nullptr);
}

TEST_F(rccl_test, rccl_event_info_can_be_modified)
{
    rccl_event_info info{};

    info.size    = 1024;
    info.is_send = true;
    info.comm    = reinterpret_cast<ncclComm_t>(0x1234);

    EXPECT_EQ(info.size, 1024u);
    EXPECT_TRUE(info.is_send);
    EXPECT_EQ(info.comm, reinterpret_cast<ncclComm_t>(0x1234));
}

TEST_F(rccl_test, rccl_event_info_large_size)
{
    rccl_event_info info{};

    info.size = std::numeric_limits<size_t>::max();

    EXPECT_EQ(info.size, std::numeric_limits<size_t>::max());
}
