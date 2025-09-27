// Simple HSA application that creates a signal and sets its value using hsa_signal_store_relaxed
// Based on patterns from standalone_sq_waves_test.cpp

#define __HIP_PLATFORM_AMD__

#include <hsa/hsa.h>
#include <cstdio>
#include <cstdlib>
#include <atomic>

// Simple logging macros
#define LOG_INFO(...) do { printf("[INFO] "); printf(__VA_ARGS__); printf("\n"); fflush(stdout); } while(0)
#define LOG_ERROR(...) do { fprintf(stderr, "[ERROR] "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fflush(stderr); } while(0)
#define CHECK_HSA(status) do { if(status != HSA_STATUS_SUCCESS) { LOG_ERROR("HSA call failed at %s:%d with status %d", __FILE__, __LINE__, status); exit(1); } } while(0)

// Debugger blocking functions
namespace {
std::atomic<bool>& debugger_block_flag() {
    static std::atomic<bool> block = {true};
    return block;
}
}

extern "C" {
void app_debugger_block() {
    debugger_block_flag().exchange(true);
    fprintf(stderr, "AT BLOCK\n");
    fflush(stderr);
    while(debugger_block_flag().load() == true) {
        // Spin wait until app_debugger_continue is called
    }
}

void app_debugger_continue() {
    debugger_block_flag().exchange(false);
}
}

int main(int argc, char** argv) {
    LOG_INFO("Starting HSA signal store test");

    // Initialize HSA
    LOG_INFO("Initializing HSA runtime");
    CHECK_HSA(hsa_init());

    // Find GPU agent
    hsa_agent_t gpu_agent = {0};
    auto iterate_agents = [](hsa_agent_t agent, void* data) -> hsa_status_t {
        hsa_device_type_t device_type;
        hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
        if(device_type == HSA_DEVICE_TYPE_GPU) {
            *((hsa_agent_t*)data) = agent;
            return HSA_STATUS_INFO_BREAK;
        }
        return HSA_STATUS_SUCCESS;
    };

    hsa_iterate_agents(iterate_agents, &gpu_agent);
    if(gpu_agent.handle == 0) {
        LOG_ERROR("No GPU agent found");
        return 1;
    }

    // Get agent name
    char agent_name[64] = {0};
    CHECK_HSA(hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_NAME, agent_name));
    LOG_INFO("Found GPU agent: %s (handle: 0x%lx)", agent_name, gpu_agent.handle);

    // Create signal with initial value 1
    LOG_INFO("Creating HSA signal with initial value 1");
    hsa_signal_t signal;
    CHECK_HSA(hsa_signal_create(1, 0, nullptr, &signal));

    // Verify initial value
    hsa_signal_value_t initial_value = hsa_signal_load_relaxed(signal);
    LOG_INFO("Signal created with initial value: %ld", initial_value);

    // Use hsa_signal_store_relaxed to set signal value to 0
    LOG_INFO("Setting signal value to 0 using hsa_signal_store_relaxed");

    app_debugger_block();  // Block before hsa_signal_store_relaxed
    hsa_signal_store_relaxed(signal, 0);
    app_debugger_block();  // Block after hsa_signal_store_relaxed

    // Verify the value was changed
    hsa_signal_value_t new_value = hsa_signal_load_relaxed(signal);
    LOG_INFO("Signal value after store operation: %ld", new_value);

    // Verify the store worked correctly
    if(new_value == 0) {
        LOG_INFO("Signal store operation completed successfully");
    } else {
        LOG_ERROR("Signal store operation failed - value is %ld instead of 0", new_value);
    }

    // Clean up the signal
    LOG_INFO("Destroying signal");
    hsa_signal_destroy(signal);

    // Shutdown HSA
    LOG_INFO("Shutting down HSA runtime");
    hsa_shut_down();

    LOG_INFO("Signal store test completed successfully");
    return 0;
}