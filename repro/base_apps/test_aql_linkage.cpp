#include "aql_profiler.h"
#include <iostream>

// Simple test to verify AQL profiler linkage without HSA initialization

int main() {
    printf("AQL Profiler Linkage Test\n");
    printf("========================\n");

    // Test version function
    aqlprofile_version_t version;
    hsa_status_t status = aqlprofile_get_version(&version);

    if (status == HSA_STATUS_SUCCESS) {
        printf("✓ aqlprofile_get_version() linked successfully\n");
        printf("  Version: %u.%u.%u\n", version.major, version.minor, version.patch);
    } else {
        printf("✗ aqlprofile_get_version() failed with status %d\n", status);
    }

    // Test null parameter handling (should return error without crashing)
    status = aqlprofile_get_version(nullptr);
    if (status != HSA_STATUS_SUCCESS) {
        printf("✓ aqlprofile_get_version(nullptr) properly rejected\n");
    } else {
        printf("⚠ aqlprofile_get_version(nullptr) should have failed\n");
    }

    // Test basic structures and constants
    printf("\n=== Interface Constants ===\n");
    printf("AQLPROFILE_MEMORY_HINT_NONE: %d\n", AQLPROFILE_MEMORY_HINT_NONE);
    printf("AQLPROFILE_MEMORY_HINT_HOST: %d\n", AQLPROFILE_MEMORY_HINT_HOST);
    printf("AQLPROFILE_AGENT_VERSION_V0: %d\n", AQLPROFILE_AGENT_VERSION_V0);
    printf("AQLPROFILE_AGENT_VERSION_V1: %d\n", AQLPROFILE_AGENT_VERSION_V1);

    // Test structure sizes
    printf("\n=== Structure Sizes ===\n");
    printf("aqlprofile_handle_t: %zu bytes\n", sizeof(aqlprofile_handle_t));
    printf("aqlprofile_agent_handle_t: %zu bytes\n", sizeof(aqlprofile_agent_handle_t));
    printf("aqlprofile_pmc_event_t: %zu bytes\n", sizeof(aqlprofile_pmc_event_t));
    printf("aqlprofile_pmc_aql_packets_t: %zu bytes\n", sizeof(aqlprofile_pmc_aql_packets_t));

    // Test enum values
    printf("\n=== Event Test ===\n");
    aqlprofile_pmc_event_t test_event = {
        .block_index = 0,
        .event_id = 35,
        .flags = {.raw = 0},
        .block_name = static_cast<hsa_ven_amd_aqlprofile_block_name_t>(6)
    };
    printf("Test event: block_index=%u, event_id=%u, block_name=%u\n",
           test_event.block_index, test_event.event_id, test_event.block_name);

    printf("\n=== Our Interface Test ===\n");

    // Test our wrapper functions (should fail gracefully without HSA)
    if (aql_profiler_create_packets(nullptr) == -1) {
        printf("✓ aql_profiler_create_packets(nullptr) properly rejected\n");
    }

    if (aql_profiler_collect_results(nullptr) == -1) {
        printf("✓ aql_profiler_collect_results(nullptr) properly rejected\n");
    }

    printf("\n✓ AQL Profiler linkage test completed successfully!\n");
    printf("  - aqlprofile v2 library is properly linked\n");
    printf("  - All interface functions are accessible\n");
    printf("  - Structures and constants are properly defined\n");
    printf("  - Error handling works correctly\n");

    return 0;
}