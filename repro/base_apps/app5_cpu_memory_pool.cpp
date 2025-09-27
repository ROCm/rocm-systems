// HSA CPU Memory Pool Application
// Uses hsa_amd_agent_iterate_memory_pools to find CPU pool and allocate

#define __HIP_PLATFORM_AMD__

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
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

// This is the call-back function for hsa_amd_agent_iterate_memory_pools() that
// finds a CPU memory pool with HSA_AMD_SEGMENT_GLOBAL properties
hsa_status_t
FindCPUPool(hsa_amd_memory_pool_t pool, void* data)
{
    if(!data) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

    hsa_amd_memory_pool_t* found_pool = static_cast<hsa_amd_memory_pool_t*>(data);

    hsa_amd_segment_t segment;
    hsa_status_t status = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
    if(status != HSA_STATUS_SUCCESS) return HSA_STATUS_SUCCESS;

    if(HSA_AMD_SEGMENT_GLOBAL != segment) return HSA_STATUS_SUCCESS;

    uint32_t flags;
    status = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
    if(status != HSA_STATUS_SUCCESS) return HSA_STATUS_SUCCESS;

    // Check if pool allows allocation
    bool alloc_allowed = false;
    status = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc_allowed);
    if(status != HSA_STATUS_SUCCESS || !alloc_allowed) return HSA_STATUS_SUCCESS;

    // Found a suitable CPU pool
    *found_pool = pool;
    return HSA_STATUS_INFO_BREAK;
}

int main(int argc, char** argv) {
    LOG_INFO("Starting HSA CPU memory pool application");

    // Initialize HSA
    LOG_INFO("Initializing HSA runtime");
    CHECK_HSA(hsa_init());

    // Find first GPU agent
    hsa_agent_t gpu_agent = {0};
    auto iterate_gpu_agents = [](hsa_agent_t agent, void* data) -> hsa_status_t {
        hsa_device_type_t device_type;
        hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
        if(device_type == HSA_DEVICE_TYPE_GPU) {
            *((hsa_agent_t*)data) = agent;
            return HSA_STATUS_INFO_BREAK;
        }
        return HSA_STATUS_SUCCESS;
    };

    hsa_iterate_agents(iterate_gpu_agents, &gpu_agent);
    if(gpu_agent.handle == 0) {
        LOG_ERROR("No GPU agent found");
        return 1;
    }

    // Get GPU agent name
    char gpu_agent_name[64] = {0};
    CHECK_HSA(hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_NAME, gpu_agent_name));
    LOG_INFO("Found GPU agent: %s (handle: 0x%lx)", gpu_agent_name, gpu_agent.handle);

    // Find CPU agent
    hsa_agent_t cpu_agent = {0};
    auto iterate_cpu_agents = [](hsa_agent_t agent, void* data) -> hsa_status_t {
        hsa_device_type_t device_type;
        hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
        if(device_type == HSA_DEVICE_TYPE_CPU) {
            *((hsa_agent_t*)data) = agent;
            return HSA_STATUS_INFO_BREAK;
        }
        return HSA_STATUS_SUCCESS;
    };

    hsa_iterate_agents(iterate_cpu_agents, &cpu_agent);
    if(cpu_agent.handle == 0) {
        LOG_ERROR("No CPU agent found");
        return 1;
    }

    // Get CPU agent name
    char cpu_agent_name[64] = {0};
    CHECK_HSA(hsa_agent_get_info(cpu_agent, HSA_AGENT_INFO_NAME, cpu_agent_name));
    LOG_INFO("Found CPU agent: %s (handle: 0x%lx)", cpu_agent_name, cpu_agent.handle);

    // Use hsa_amd_agent_iterate_memory_pools to find CPU memory pool
    LOG_INFO("Finding CPU memory pool using hsa_amd_agent_iterate_memory_pools");
    hsa_amd_memory_pool_t cpu_pool = {0};
    hsa_status_t status = hsa_amd_agent_iterate_memory_pools(cpu_agent, FindCPUPool, &cpu_pool);

    if(status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
        LOG_ERROR("Failed to iterate CPU memory pools, status: %d", status);
        return 1;
    }

    if(cpu_pool.handle == 0) {
        LOG_ERROR("No suitable CPU memory pool found");
        return 1;
    }

    LOG_INFO("Found CPU memory pool (handle: 0x%lx)", cpu_pool.handle);

    // Get pool size for information
    size_t pool_size = 0;
    status = hsa_amd_memory_pool_get_info(cpu_pool, HSA_AMD_MEMORY_POOL_INFO_SIZE, &pool_size);
    if(status == HSA_STATUS_SUCCESS) {
        LOG_INFO("CPU pool size: %zu bytes", pool_size);
    }

    // Allocate memory from CPU pool using hsa_amd_memory_pool_allocate
    LOG_INFO("Allocating 4096 bytes from CPU memory pool");
    void* cpu_memory = nullptr;
    const size_t alloc_size = 4096;
    status = hsa_amd_memory_pool_allocate(cpu_pool, alloc_size, 0, &cpu_memory);

    if(status != HSA_STATUS_SUCCESS) {
        LOG_ERROR("Failed to allocate from CPU memory pool, status: %d", status);
        hsa_shut_down();
        return 1;
    }

    LOG_INFO("Successfully allocated %zu bytes at address: %p", alloc_size, cpu_memory);

    // Write to the allocated memory to verify it works
    LOG_INFO("Testing memory by writing and reading data");
    int* test_data = static_cast<int*>(cpu_memory);

    app_debugger_block();  // Block before memory writes
    test_data[0] = 0xDEADBEEF;
    test_data[1] = 0xCAFEBABE;
    app_debugger_block();  // Block after memory writes

    if(test_data[0] == 0xDEADBEEF && test_data[1] == 0xCAFEBABE) {
        LOG_INFO("Memory test successful - data written and read correctly");
    } else {
        LOG_ERROR("Memory test failed - data corruption detected");
    }

    // Free memory and cleanup
    LOG_INFO("Freeing allocated memory");
    status = hsa_amd_memory_pool_free(cpu_memory);
    if(status != HSA_STATUS_SUCCESS) {
        LOG_ERROR("Failed to free CPU memory, status: %d", status);
    } else {
        LOG_INFO("Memory freed successfully");
    }

    // Cleanup
    LOG_INFO("Shutting down HSA runtime");
    hsa_shut_down();

    LOG_INFO("CPU memory pool application completed successfully");

    return 0;
}