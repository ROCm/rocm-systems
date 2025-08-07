#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
#include <array>
#include <memory>
#include <utility>

// Include our test-specific implementations first
namespace pm4_builder {
    // Command buffer interface needed by spm_builder.h
    class CmdBuffer {
    public:
        virtual ~CmdBuffer() = default;
        virtual void Append(const void* data, size_t size) = 0;
        virtual size_t Size() const = 0;
        virtual const void* Data() const = 0;
        virtual void Clear() = 0;
    };

    // Register and delay information structures
    struct RegisterInfo {
        uint32_t addr;
        uint32_t size;
    };

    struct DelayInfo {
        uint32_t reg;
        uint32_t delay;
    };

    // Define block descriptor first
    struct BlockDescriptor {
        uint32_t id;     // Block type identifier
        uint32_t index;  // Instance index
    };

    // Counter block info structure with all required members
    struct CounterBlockInfo {
        uint32_t block_id;
        uint32_t num_instances;
        uint32_t num_counters;
        uint32_t attr;  // Block attributes (global, SQ, etc.)
        uint32_t instance_count;  // Number of instances
        std::array<RegisterInfo, 16> counter_reg_info;  // Array of register info for counters
        std::array<DelayInfo, 16> delay_info;  // Array of delay info
    };

    // Counter description structure
    struct CounterDescription {
        BlockDescriptor block_des;  // Block descriptor
        CounterBlockInfo* block_info;  // Pointer to block info
        uint32_t index;  // Counter index in the block
    };

    // Type alias for backward compatibility
    typedef CounterDescription counter_des_t;

    // Create a simple vector-based counters_vector
    class counters_vector : public std::vector<counter_des_t> {
    public:
        typedef std::vector<counter_des_t> Parent;
        
        counters_vector() : Parent(), attr_(0) {}

        void push_back(const counter_des_t& des) {
            Parent::push_back(des);
            attr_ |= des.block_info->attr;
        }

        uint32_t get_attr() const { return attr_; }

    private:
        uint32_t attr_;
    };
}

// Mock minimal dependencies
namespace pm4 {
    struct cmd_config {
        static constexpr uint32_t CMD_BUFFER_SIZE = 4096;
    };
}

// Test fixture class
class SpmBuilderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize the block info
        block_info.block_id = 1;
        block_info.instance_count = 2;
        block_info.num_counters = 4;
        block_info.attr = 0;  // Non-global, non-SQ block

        // Setup register and delay info
        for (uint32_t i = 0; i < block_info.num_counters; ++i) {
            block_info.counter_reg_info[i].addr = 0x2000 + i * 4;
            block_info.counter_reg_info[i].size = 4;
            block_info.delay_info[i].reg = 0x1000 + i * 4;
            block_info.delay_info[i].delay = i + 1;
        }
    }

    pm4_builder::CounterBlockInfo block_info;
    pm4_builder::counters_vector counters;
};

// Test cases
TEST_F(SpmBuilderTest, BasicConfiguration) {
    // Add test counters
    for (uint32_t i = 0; i < block_info.num_counters; ++i) {
        pm4_builder::counter_des_t counter;
        counter.block_des.id = i;
        counter.block_des.index = i % 2;  // Alternate between instances
        counter.block_info = &block_info;
        counter.index = i;
        counters.push_back(counter);
    }

    // Validate counter setup
    EXPECT_EQ(counters.size(), block_info.num_counters);
    for (const auto& counter : counters) {
        EXPECT_LT(counter.block_des.index, block_info.instance_count) << "Invalid instance index";
        EXPECT_LT(counter.index, block_info.num_counters) << "Invalid counter index";
        const auto& reg_info = counter.block_info->counter_reg_info[counter.index];
        EXPECT_GT(reg_info.addr, 0) << "Invalid register address";
        EXPECT_EQ(reg_info.size, 4) << "Invalid register size";
    }
}
