/**
 * Simple test for packet generation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include "packet_generation.h"
#include "aql_structures.h"
#include "pm4_packets.h"

int main(void) {
    printf("🧪 Simple Packet Generation Test\n");

    /* Test basic buffer creation */
    pm4_buffer_t* buffer = pm4_buffer_create(1024);
    if (!buffer) {
        printf("❌ Failed to create buffer\n");
        return -1;
    }
    printf("✓ Created buffer with capacity %zu\n", buffer->capacity);

    /* Create basic arch using the real creator */
    arch_t* arch = arch_create_by_name("gfx12");
    if (!arch) {
        printf("❌ Failed to create arch\n");
        pm4_buffer_destroy(buffer);
        return -1;
    }
    printf("✓ Created GFX12 architecture\n");

    /* Test simple functions */
    int ret = generate_cs_partial_flush(buffer, arch);
    if (ret != 0) {
        printf("❌ CS partial flush failed: %d\n", ret);
    } else {
        printf("✓ CS partial flush succeeded, buffer size: %zu\n", buffer->size);
    }

    ret = generate_grbm_broadcast(buffer, arch);
    if (ret != 0) {
        printf("❌ GRBM broadcast failed: %d\n", ret);
    } else {
        printf("✓ GRBM broadcast succeeded, buffer size: %zu\n", buffer->size);
    }

    ret = generate_perfmon_enable(buffer, arch, 0, false);
    if (ret != 0) {
        printf("❌ Perfmon enable failed: %d\n", ret);
    } else {
        printf("✓ Perfmon enable succeeded, buffer size: %zu\n", buffer->size);
    }

    /* Clean up */
    arch_destroy(arch);
    pm4_buffer_destroy(buffer);

    printf("✅ Simple test completed successfully!\n");
    return 0;
}