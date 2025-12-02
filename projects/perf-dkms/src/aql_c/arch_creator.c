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

/**
 * @brief Create an architecture structure by name string
 *
 * Factory function that instantiates the appropriate architecture-specific
 * structure based on the GPU architecture name. Currently supports GFX12
 * with placeholders for future architectures (GFX11, GFX10, GFX9).
 *
 * This function is analogous to the Pm4Factory::Create() factory pattern in
 * projects/aqlprofile/src/core/pm4_factory.h:113-377, but implemented in C
 * with explicit architecture creators instead of C++ polymorphism.
 *
 * @param arch_name String identifying the architecture (e.g., "gfx12", "GFX12")
 * @return Pointer to initialized architecture structure, or NULL if unknown/unsupported
 *
 * @note Comparison is case-insensitive (both "gfx12" and "GFX12" work)
 * @note Caller is responsible for freeing with arch_destroy()
 * @see arch_destroy(), Pm4Factory::Create in projects/aqlprofile/src/core/pm4_factory.h:113
 */
arch_t *arch_create_by_name(const char *arch_name)
{
	if (!arch_name)
		return NULL;

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

/**
 * @brief Destroy an architecture structure and free all associated memory
 *
 * Recursively frees all dynamically allocated components of an architecture
 * structure, including:
 * - Block information structures
 * - Counter register info arrays
 * - Dimension arrays
 * - Command buffer
 * - The arch_t structure itself
 *
 * This function is analogous to Pm4Factory::~Pm4Factory() destructor in
 * projects/aqlprofile/src/core/pm4_factory.h:233-238, but implemented
 * explicitly in C without destructors.
 *
 * @param arch Pointer to architecture structure to destroy, or NULL (no-op)
 *
 * @note Safe to call with NULL pointer
 * @note After calling, the arch pointer is invalid and should not be used
 * @note Iterates through all HW_IP_BLOCK_LAST blocks to find and free allocated ones
 * @see arch_create_by_name(), Pm4Factory::~Pm4Factory in
 *      projects/aqlprofile/src/core/pm4_factory.h:233
 */
void arch_destroy(arch_t *arch)
{
	if (!arch)
		return;

	/* Free block information */
	for (uint32_t i = 0; i < HW_IP_BLOCK_LAST; i++) {
		if (arch->block_map.blocks[i]) {
			/* Free counter register info arrays */
			if (arch->block_map.blocks[i]->counter_reg_info) {
				FREE(arch->block_map.blocks[i]->counter_reg_info);
			}

			/* Free dimensions arrays */
			if (arch->block_map.blocks[i]->dimensions) {
				FREE(arch->block_map.blocks[i]->dimensions);
			}

			FREE(arch->block_map.blocks[i]);
		}
	}

	/* Free command buffer */
	if (arch->command) {
		FREE(arch->command);
	}

	/* Free the architecture structure itself */
	FREE(arch);
}