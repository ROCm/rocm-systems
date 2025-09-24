/**
 * @file arch_creator.c
 * @brief Main architecture creation dispatcher for GPU performance monitoring
 */

#include "aql_structures.h"
#include "arch_creator_common.h"

#ifdef __KERNEL__
#include <linux/string.h>
#else
#include <string.h>
#endif

/* Main architecture creation function - dispatcher */
arch_t* arch_create_by_name(const char* arch_name) {
    if (!arch_name) return NULL;

    /* GFX12 architecture */
    if (strcmp(arch_name, "gfx12") == 0 || strcmp(arch_name, "GFX12") == 0) {
        return create_gfx12_arch();
    }

    /* Future architectures can be added here with their respective creator functions:
     *
     * if (strcmp(arch_name, "gfx11") == 0 || strcmp(arch_name, "GFX11") == 0) {
     *     return create_gfx11_arch();
     * }
     *
     * if (strcmp(arch_name, "gfx10") == 0 || strcmp(arch_name, "GFX10") == 0) {
     *     return create_gfx10_arch();
     * }
     *
     * if (strcmp(arch_name, "gfx9") == 0 || strcmp(arch_name, "GFX9") == 0) {
     *     return create_gfx9_arch();
     * }
     */

    /* Unknown architecture */
    return NULL;
}