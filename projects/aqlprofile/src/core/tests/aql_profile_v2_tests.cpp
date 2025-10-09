#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstring>
#include <vector>
#include <memory>

#include "aqlprofile-sdk/aql_profile_v2.h"
#include "../logger.h"
#include "../pm4_factory.h"
// Define static members
bool aql_profile::Pm4Factory::concurrent_create_mode_ = false;
bool aql_profile::Pm4Factory::spm_kfd_mode_ = false;
//Pm4Factory::mutex_t Pm4Factory::mutex_;
aql_profile::Pm4Factory::instances_t* aql_profile::Pm4Factory::instances_ = nullptr;
namespace aql_profile {
Logger::mutex_t Logger::mutex_;
Logger* Logger::instance_ = nullptr;
}

namespace aql_profile_v2_tests {

class AqlProfileV2Test : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize test data structures
        memset(&test_agent_info_, 0, sizeof(test_agent_info_));
        memset(&test_agent_info_v1_, 0, sizeof(test_agent_info_v1_));
        memset(&test_handle_, 0, sizeof(test_handle_));
        memset(&test_agent_handle_, 0, sizeof(test_agent_handle_));

        // Set up default agent info
        test_agent_info_.agent_gfxip = "gfx90a";
        test_agent_info_.xcc_num = 1;
        test_agent_info_.se_num = 8;
        test_agent_info_.cu_num = 104;
        test_agent_info_.shader_arrays_per_se = 2;

        // Set up default agent info v1
        test_agent_info_v1_.agent_gfxip = "gfx90a";
        test_agent_info_v1_.xcc_num = 1;
        test_agent_info_v1_.se_num = 8;
        test_agent_info_v1_.cu_num = 104;
        test_agent_info_v1_.shader_arrays_per_se = 2;
        test_agent_info_v1_.domain = 0;
        test_agent_info_v1_.location_id = 0x12345678;

        test_handle_.handle = 0x1234567890ABCDEF;
        test_agent_handle_.handle = 0xFEDCBA0987654321;
    }

    aqlprofile_agent_info_t test_agent_info_;
    aqlprofile_agent_info_v1_t test_agent_info_v1_;
    aqlprofile_handle_t test_handle_;
    aqlprofile_agent_handle_t test_agent_handle_;
};

// Test enum values and ranges
TEST_F(AqlProfileV2Test, EnumValues) {
    // Test memory hint enum
    EXPECT_EQ(AQLPROFILE_MEMORY_HINT_NONE, 0);
    EXPECT_EQ(AQLPROFILE_MEMORY_HINT_HOST, 1);
    EXPECT_EQ(AQLPROFILE_MEMORY_HINT_DEVICE_UNCACHED, 2);
    EXPECT_EQ(AQLPROFILE_MEMORY_HINT_DEVICE_COHERENT, 3);
    EXPECT_EQ(AQLPROFILE_MEMORY_HINT_DEVICE_NONCOHERENT, 4);
    EXPECT_GT(AQLPROFILE_MEMORY_HINT_LAST, AQLPROFILE_MEMORY_HINT_DEVICE_NONCOHERENT);

    // Test agent version enum
    EXPECT_EQ(AQLPROFILE_AGENT_VERSION_NONE, 0);
    EXPECT_EQ(AQLPROFILE_AGENT_VERSION_V0, 1);
    EXPECT_EQ(AQLPROFILE_AGENT_VERSION_V1, 2);
    EXPECT_GT(AQLPROFILE_AGENT_VERSION_LAST, AQLPROFILE_AGENT_VERSION_V1);

    // Test accumulation type enum
    EXPECT_EQ(AQLPROFILE_ACCUMULATION_NONE, 0);
    EXPECT_EQ(AQLPROFILE_ACCUMULATION_LO_RES, 1);
    EXPECT_EQ(AQLPROFILE_ACCUMULATION_HI_RES, 2);
    EXPECT_GT(AQLPROFILE_ACCUMULATION_LAST, AQLPROFILE_ACCUMULATION_HI_RES);
}

// Test block name enum coverage
TEST_F(AqlProfileV2Test, BlockNameEnum) {
    // Test that reserved blocks are in the expected range
    EXPECT_EQ(AQLPROFILE_BLOCK_NAME_RESERVED_0, HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER);
    EXPECT_EQ(AQLPROFILE_BLOCK_NAME_RESERVED_1, HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER + 1);

    // Test that new block names are defined
    EXPECT_GT(AQLPROFILE_BLOCK_NAME_CHA, AQLPROFILE_BLOCK_NAME_RESERVED_5);
    EXPECT_GT(AQLPROFILE_BLOCK_NAME_CHC, AQLPROFILE_BLOCK_NAME_CHA);
    EXPECT_GT(AQLPROFILE_BLOCK_NAME_SQG, AQLPROFILE_BLOCK_NAME_GRBMH);

    // Test final block count
    EXPECT_GT(AQLPROFILE_BLOCKS_NUMBER, HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER);
}

// Test buffer descriptor flags structure
TEST_F(AqlProfileV2Test, BufferDescFlags) {
    aqlprofile_buffer_desc_flags_t flags;

    // Test raw access
    flags.raw = 0;
    EXPECT_EQ(flags.device_access, 0);
    EXPECT_EQ(flags.host_access, 0);
    EXPECT_EQ(flags.memory_hint, 0);

    // Test individual field access
    flags.device_access = 1;
    flags.host_access = 1;
    flags.memory_hint = AQLPROFILE_MEMORY_HINT_HOST;

    EXPECT_EQ(flags.device_access, 1);
    EXPECT_EQ(flags.host_access, 1);
    EXPECT_EQ(flags.memory_hint, AQLPROFILE_MEMORY_HINT_HOST);

    // Test field width constraints
    flags.memory_hint = 0x3F; // 6 bits max
    EXPECT_EQ(flags.memory_hint, 0x3F);

    // Test bit manipulation
    uint32_t expected = (1 << 0) | (1 << 1) | (0x3F << 2);
    EXPECT_EQ(flags.raw, expected);
}

// Test PMC event flags structure
TEST_F(AqlProfileV2Test, PmcEventFlags) {
    aqlprofile_pmc_event_flags_t flags;

    // Test raw access
    flags.raw = 0;
    EXPECT_EQ(flags.sq_flags.accum, 0);

    // Test accumulation field
    flags.sq_flags.accum = AQLPROFILE_ACCUMULATION_LO_RES;
    EXPECT_EQ(flags.sq_flags.accum, AQLPROFILE_ACCUMULATION_LO_RES);

    flags.sq_flags.accum = AQLPROFILE_ACCUMULATION_HI_RES;
    EXPECT_EQ(flags.sq_flags.accum, AQLPROFILE_ACCUMULATION_HI_RES);

    // Test field width (3 bits for accumulation)
    flags.sq_flags.accum = 0x7; // 3 bits max
    EXPECT_EQ(flags.sq_flags.accum, 0x7);
}

// Test PMC event structure
TEST_F(AqlProfileV2Test, PmcEvent) {
    aqlprofile_pmc_event_t event;

    event.block_index = 42;
    event.event_id = 123;
    event.flags.raw = 0;
    event.flags.sq_flags.accum = AQLPROFILE_ACCUMULATION_HI_RES;
    event.block_name = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ;

    EXPECT_EQ(event.block_index, 42);
    EXPECT_EQ(event.event_id, 123);
    EXPECT_EQ(event.flags.sq_flags.accum, AQLPROFILE_ACCUMULATION_HI_RES);
    EXPECT_EQ(event.block_name, HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ);
}

// Test agent info structure
TEST_F(AqlProfileV2Test, AgentInfo) {
    EXPECT_STREQ(test_agent_info_.agent_gfxip, "gfx90a");
    EXPECT_EQ(test_agent_info_.xcc_num, 1);
    EXPECT_EQ(test_agent_info_.se_num, 8);
    EXPECT_EQ(test_agent_info_.cu_num, 104);
    EXPECT_EQ(test_agent_info_.shader_arrays_per_se, 2);

    // Test with different GPU configurations
    aqlprofile_agent_info_t gfx11_info;
    gfx11_info.agent_gfxip = "gfx1100";
    gfx11_info.xcc_num = 2;
    gfx11_info.se_num = 6;
    gfx11_info.cu_num = 96;
    gfx11_info.shader_arrays_per_se = 4;

    EXPECT_STREQ(gfx11_info.agent_gfxip, "gfx1100");
    EXPECT_EQ(gfx11_info.xcc_num, 2);
    EXPECT_EQ(gfx11_info.se_num, 6);
    EXPECT_EQ(gfx11_info.cu_num, 96);
    EXPECT_EQ(gfx11_info.shader_arrays_per_se, 4);
}

// Test agent info v1 structure (extended version)
TEST_F(AqlProfileV2Test, AgentInfoV1) {
    EXPECT_STREQ(test_agent_info_v1_.agent_gfxip, "gfx90a");
    EXPECT_EQ(test_agent_info_v1_.xcc_num, 1);
    EXPECT_EQ(test_agent_info_v1_.se_num, 8);
    EXPECT_EQ(test_agent_info_v1_.cu_num, 104);
    EXPECT_EQ(test_agent_info_v1_.shader_arrays_per_se, 2);
    EXPECT_EQ(test_agent_info_v1_.domain, 0);
    EXPECT_EQ(test_agent_info_v1_.location_id, 0x12345678);

    // Test with different PCI information
    aqlprofile_agent_info_v1_t pci_info;
    pci_info.agent_gfxip = "gfx942";
    pci_info.domain = 0x0001;
    pci_info.location_id = 0x00010203; // Bus=1, Device=2, Function=3

    EXPECT_EQ(pci_info.domain, 0x0001);
    EXPECT_EQ(pci_info.location_id, 0x00010203);
}

// Test handle structures
TEST_F(AqlProfileV2Test, HandleStructures) {
    EXPECT_EQ(test_handle_.handle, 0x1234567890ABCDEF);
    EXPECT_EQ(test_agent_handle_.handle, 0xFEDCBA0987654321);

    // Test handle comparison
    aqlprofile_handle_t handle1 = {0x123};
    aqlprofile_handle_t handle2 = {0x123};
    aqlprofile_handle_t handle3 = {0x456};

    EXPECT_EQ(handle1.handle, handle2.handle);
    EXPECT_NE(handle1.handle, handle3.handle);
}

// Test PMC profile structure
TEST_F(AqlProfileV2Test, PmcProfile) {
    std::vector<aqlprofile_pmc_event_t> events(3);

    // Setup events
    events[0] = {0, 100, {0}, HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ};
    events[1] = {1, 200, {0}, HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA};
    events[2] = {2, 300, {0}, HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCA};

    aqlprofile_pmc_profile_t profile;
    profile.agent = test_agent_handle_;
    profile.events = events.data();
    profile.event_count = events.size();

    EXPECT_EQ(profile.agent.handle, test_agent_handle_.handle);
    EXPECT_EQ(profile.event_count, 3);
    EXPECT_EQ(profile.events[0].block_index, 0);
    EXPECT_EQ(profile.events[0].event_id, 100);
    EXPECT_EQ(profile.events[1].block_index, 1);
    EXPECT_EQ(profile.events[1].event_id, 200);
    EXPECT_EQ(profile.events[2].block_index, 2);
    EXPECT_EQ(profile.events[2].event_id, 300);
}

// Test ATT parameter structure
TEST_F(AqlProfileV2Test, AttParameter) {
    aqlprofile_att_parameter_t param;

    // Test basic parameter
    param.parameter_name = HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_OCCUPANCY_MODE;
    param.value = 1;

    EXPECT_EQ(param.parameter_name, HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_OCCUPANCY_MODE);
    EXPECT_EQ(param.value, 1);

    // Test counter ID and SIMD mask fields
    param.parameter_name = HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SE_MASK;
    param.counter_id = 0x1234567; // 28 bits max
    param.simd_mask = 0xF;        // 4 bits max

    EXPECT_EQ(param.counter_id, 0x1234567);
    EXPECT_EQ(param.simd_mask, 0xF);

}

// Test ATT profile structure
TEST_F(AqlProfileV2Test, AttProfile) {
    hsa_agent_t agent;
    agent.handle = 0xABCDEF1234567890;

    std::vector<aqlprofile_att_parameter_t> params(1);
    params[0].parameter_name = HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_OCCUPANCY_MODE;
    params[0].value = 1;


    aqlprofile_att_profile_t profile;
    profile.agent = agent;
    profile.parameters = params.data();
    profile.parameter_count = params.size();

    EXPECT_EQ(profile.agent.handle, agent.handle);
    EXPECT_EQ(profile.parameter_count, 1);
    EXPECT_EQ(profile.parameters[0].parameter_name, HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_OCCUPANCY_MODE);
    EXPECT_EQ(profile.parameters[0].value, 1);
    // EXPECT_EQ(profile.parameters[1].parameter_name, AQLPROFILE_ATT_PARAMETER_NAME_RT_TIMESTAMP);
    // EXPECT_EQ(profile.parameters[1].value, AQLPROFILE_ATT_PARAMETER_RT_TIMESTAMP_ENABLE);
}

// Test PMC AQL packets structure
TEST_F(AqlProfileV2Test, PmcAqlPackets) {
    aqlprofile_pmc_aql_packets_t packets;

    // Initialize packet headers
    packets.start_packet.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
    packets.stop_packet.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
    packets.read_packet.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;

    // Test packet initialization
    EXPECT_EQ(packets.start_packet.header & HSA_PACKET_HEADER_TYPE,
              HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE);
    EXPECT_EQ(packets.stop_packet.header & HSA_PACKET_HEADER_TYPE,
              HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE);
    EXPECT_EQ(packets.read_packet.header & HSA_PACKET_HEADER_TYPE,
              HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE);
}

// Test ATT control AQL packets structure
TEST_F(AqlProfileV2Test, AttControlAqlPackets) {
    aqlprofile_att_control_aql_packets_t packets;

    // Initialize packet headers
    packets.start_packet.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
    packets.stop_packet.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;

    // Test packet initialization
    EXPECT_EQ(packets.start_packet.header & HSA_PACKET_HEADER_TYPE,
              HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE);
    EXPECT_EQ(packets.stop_packet.header & HSA_PACKET_HEADER_TYPE,
              HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE);
}

// Test ATT code object data structure
TEST_F(AqlProfileV2Test, AttCodeobjData) {
    aqlprofile_att_codeobj_data_t data;

    data.id = 0x123456789ABCDEF0;
    data.addr = 0xDEADBEEFCAFEBABE;
    data.size = 0x10000;
    data.agent.handle = 0x1122334455667788;
    data.isUnload = 1;
    data.fromStart = 0;

    EXPECT_EQ(data.id, 0x123456789ABCDEF0);
    EXPECT_EQ(data.addr, 0xDEADBEEFCAFEBABE);
    EXPECT_EQ(data.size, 0x10000);
    EXPECT_EQ(data.agent.handle, 0x1122334455667788);
    EXPECT_EQ(data.isUnload, 1);
    EXPECT_EQ(data.fromStart, 0);
}

// Test info type enum values
TEST_F(AqlProfileV2Test, InfoTypeEnum) {
    EXPECT_EQ(AQLPROFILE_INFO_COMMAND_BUFFER_SIZE, 0);
    EXPECT_EQ(AQLPROFILE_INFO_PMC_DATA_SIZE, 1);
    EXPECT_EQ(AQLPROFILE_INFO_PMC_DATA, 2);
    EXPECT_EQ(AQLPROFILE_INFO_BLOCK_COUNTERS, 4);
    EXPECT_EQ(AQLPROFILE_INFO_BLOCK_ID, 5);
    EXPECT_EQ(AQLPROFILE_INFO_ENABLE_CMD, 6);
    EXPECT_EQ(AQLPROFILE_INFO_DISABLE_CMD, 7);
}

// Test RT timestamp parameter enum
TEST_F(AqlProfileV2Test, RtTimestampEnum) {
    EXPECT_EQ(AQLPROFILE_ATT_PARAMETER_RT_TIMESTAMP_DEFAULT, 0);
    EXPECT_EQ(AQLPROFILE_ATT_PARAMETER_RT_TIMESTAMP_ENABLE, 1);
    EXPECT_EQ(AQLPROFILE_ATT_PARAMETER_RT_TIMESTAMP_DISABLE, 2);
}

// Test extended parameter names
TEST_F(AqlProfileV2Test, ExtendedParameterNames) {
    EXPECT_EQ(AQLPROFILE_ATT_PARAMETER_NAME_BUFFER_SIZE_HIGH, 11);
    EXPECT_GT(AQLPROFILE_ATT_PARAMETER_NAME_RT_TIMESTAMP, AQLPROFILE_ATT_PARAMETER_NAME_BUFFER_SIZE_HIGH);
}

// Test structure sizes and alignment
TEST_F(AqlProfileV2Test, StructureSizes) {
    // Verify key structure sizes are reasonable
    EXPECT_GT(sizeof(aqlprofile_handle_t), 0);
    EXPECT_GT(sizeof(aqlprofile_agent_handle_t), 0);
    EXPECT_GT(sizeof(aqlprofile_buffer_desc_flags_t), 0);
    EXPECT_GT(sizeof(aqlprofile_pmc_event_flags_t), 0);
    EXPECT_GT(sizeof(aqlprofile_pmc_event_t), 0);
    EXPECT_GT(sizeof(aqlprofile_agent_info_t), 0);
    EXPECT_GT(sizeof(aqlprofile_agent_info_v1_t), 0);
    EXPECT_GT(sizeof(aqlprofile_pmc_profile_t), 0);
    EXPECT_GT(sizeof(aqlprofile_att_parameter_t), 0);
    EXPECT_GT(sizeof(aqlprofile_att_profile_t), 0);
    EXPECT_GT(sizeof(aqlprofile_pmc_aql_packets_t), 0);
    EXPECT_GT(sizeof(aqlprofile_att_control_aql_packets_t), 0);
    EXPECT_GT(sizeof(aqlprofile_att_codeobj_data_t), 0);

    // Verify v1 structure is larger than base version
    EXPECT_GT(sizeof(aqlprofile_agent_info_v1_t), sizeof(aqlprofile_agent_info_t));

    // Verify handle structures are 8 bytes (uint64_t)
    EXPECT_EQ(sizeof(aqlprofile_handle_t), 8);
    EXPECT_EQ(sizeof(aqlprofile_agent_handle_t), 8);
}

// Test union and bitfield functionality
TEST_F(AqlProfileV2Test, UnionBitfieldFunctionality) {
    // Test buffer descriptor flags union
    aqlprofile_buffer_desc_flags_t flags;
    flags.raw = 0xFFFFFFFF;

    // Check that bitfields are properly masked
    EXPECT_EQ(flags.device_access, 1);  // 1 bit
    EXPECT_EQ(flags.host_access, 1);    // 1 bit
    EXPECT_EQ(flags.memory_hint, 0x3F); // 6 bits

    // Test PMC event flags union
    aqlprofile_pmc_event_flags_t pmc_flags;
    pmc_flags.raw = 0xFFFFFFFF;

    EXPECT_EQ(pmc_flags.sq_flags.accum, 0x7); // 3 bits

    // Test ATT parameter union
    aqlprofile_att_parameter_t param;
    param.value = 0xFFFFFFFF;

    EXPECT_EQ(param.counter_id, 0x0FFFFFFF); // 28 bits
    EXPECT_EQ(param.simd_mask, 0xF);         // 4 bits
}

// Test default/invalid values handling
TEST_F(AqlProfileV2Test, DefaultInvalidValues) {
    // Test zero-initialized structures
    aqlprofile_handle_t zero_handle = {0};
    EXPECT_EQ(zero_handle.handle, 0);

    aqlprofile_agent_info_t zero_info = {};
    EXPECT_EQ(zero_info.agent_gfxip, nullptr);
    EXPECT_EQ(zero_info.xcc_num, 0);
    EXPECT_EQ(zero_info.se_num, 0);
    EXPECT_EQ(zero_info.cu_num, 0);
    EXPECT_EQ(zero_info.shader_arrays_per_se, 0);

    // Test with maximum values
    aqlprofile_pmc_event_t max_event = {};
    max_event.block_index = UINT32_MAX;
    max_event.event_id = UINT32_MAX;
    max_event.flags.raw = UINT32_MAX;

    EXPECT_EQ(max_event.block_index, UINT32_MAX);
    EXPECT_EQ(max_event.event_id, UINT32_MAX);
    EXPECT_EQ(max_event.flags.raw, UINT32_MAX);
}

// Mock callback functions for testing
class CallbackMock {
public:
    MOCK_METHOD(hsa_status_t, memory_alloc, (void** ptr, uint64_t size, aqlprofile_buffer_desc_flags_t flags, void* userdata), ());
    MOCK_METHOD(void, memory_dealloc, (void* ptr, void* userdata), ());
    MOCK_METHOD(hsa_status_t, memory_copy, (void* dst, const void* src, size_t size, void* userdata), ());
    MOCK_METHOD(hsa_status_t, pmc_data_callback, (aqlprofile_pmc_event_t event, uint64_t counter_id, uint64_t counter_value, void* userdata), ());
    MOCK_METHOD(hsa_status_t, att_data_callback, (uint32_t shader, void* buffer, uint64_t size, void* callback_data), ());
    MOCK_METHOD(hsa_status_t, eventname_callback, (int id, const char* name, void* data), ());
    MOCK_METHOD(hsa_status_t, coordinate_callback, (int position, int id, int extent, int coordinate, const char* name, void* userdata), ());
};

// Test callback function signatures
TEST_F(AqlProfileV2Test, CallbackSignatures) {
    CallbackMock mock;

    // Test that callback function pointers can be assigned
    aqlprofile_memory_alloc_callback_t alloc_cb =
        [](void** ptr, uint64_t size, aqlprofile_buffer_desc_flags_t flags, void* userdata) -> hsa_status_t {
            return HSA_STATUS_SUCCESS;
        };

    aqlprofile_memory_dealloc_callback_t dealloc_cb =
        [](void* ptr, void* userdata) -> void {};

    aqlprofile_memory_copy_t copy_cb =
        [](void* dst, const void* src, size_t size, void* userdata) -> hsa_status_t {
            return HSA_STATUS_SUCCESS;
        };

    aqlprofile_pmc_data_callback_t pmc_cb =
        [](aqlprofile_pmc_event_t event, uint64_t counter_id, uint64_t counter_value, void* userdata) -> hsa_status_t {
            return HSA_STATUS_SUCCESS;
        };

    aqlprofile_att_data_callback_t att_cb =
        [](uint32_t shader, void* buffer, uint64_t size, void* callback_data) -> hsa_status_t {
            return HSA_STATUS_SUCCESS;
        };

    aqlprofile_eventname_callback_t event_cb =
        [](int id, const char* name, void* data) -> hsa_status_t {
            return HSA_STATUS_SUCCESS;
        };

    aqlprofile_coordinate_callback_t coord_cb =
        [](int position, int id, int extent, int coordinate, const char* name, void* userdata) -> hsa_status_t {
            return HSA_STATUS_SUCCESS;
        };

    // Verify callbacks are assigned
    EXPECT_NE(alloc_cb, nullptr);
    EXPECT_NE(dealloc_cb, nullptr);
    EXPECT_NE(copy_cb, nullptr);
    EXPECT_NE(pmc_cb, nullptr);
    EXPECT_NE(att_cb, nullptr);
    EXPECT_NE(event_cb, nullptr);
    EXPECT_NE(coord_cb, nullptr);
}

// Test actual aqlprofile API functions
class AqlProfileV2ApiTest : public AqlProfileV2Test {
protected:
    void SetUp() override {
        AqlProfileV2Test::SetUp();
        // Initialize callback counters for verification
        callback_call_count_ = 0;
        last_callback_id_ = -1;
        last_callback_name_ = "";
    }

    static int callback_call_count_;
    static int last_callback_id_;
    static std::string last_callback_name_;

    // Mock callback functions for testing
    static hsa_status_t eventname_callback_mock(int id, const char* name, void* data) {
        callback_call_count_++;
        last_callback_id_ = id;
        if (name) last_callback_name_ = name;
        return HSA_STATUS_SUCCESS;
    }

    static hsa_status_t coordinate_callback_mock(int position, int id, int extent,
                                                int coordinate, const char* name, void* userdata) {
        callback_call_count_++;
        return HSA_STATUS_SUCCESS;
    }

    static hsa_status_t pmc_data_callback_mock(aqlprofile_pmc_event_t event, uint64_t counter_id,
                                             uint64_t counter_value, void* userdata) {
        callback_call_count_++;
        return HSA_STATUS_SUCCESS;
    }

    static hsa_status_t att_data_callback_mock(uint32_t shader, void* buffer,
                                             uint64_t size, void* callback_data) {
        callback_call_count_++;
        return HSA_STATUS_SUCCESS;
    }

    static hsa_status_t memory_alloc_mock(void** ptr, uint64_t size,
                                        aqlprofile_buffer_desc_flags_t flags, void* userdata) {
        if (ptr && size > 0) {
            *ptr = malloc(size);
            return *ptr ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR_OUT_OF_RESOURCES;
        }
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    static void memory_dealloc_mock(void* ptr, void* userdata) {
        if (ptr) free(ptr);
    }

    static hsa_status_t memory_copy_mock(void* dst, const void* src, size_t size, void* userdata) {
        if (dst && src && size > 0) {
            memcpy(dst, src, size);
            return HSA_STATUS_SUCCESS;
        }
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
};

// Initialize static members
int AqlProfileV2ApiTest::callback_call_count_ = 0;
int AqlProfileV2ApiTest::last_callback_id_ = -1;
std::string AqlProfileV2ApiTest::last_callback_name_ = "";

} // namespace aql_profile_v2_tests
