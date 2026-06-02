/**
 * @file arch_creator.h
 * @brief Architecture creator interface for GPU architectures
 */

#ifndef ARCH_CREATOR_H
#define ARCH_CREATOR_H

#include "aql_structures.h"

/**
 * @brief Create an architecture structure by name string
 *
 * Factory function that instantiates the appropriate GPU architecture based
 * on the architecture name. Currently supports GFX12 with placeholders for
 * future architectures.
 *
 * @param arch_name String identifying the architecture (e.g., "gfx12", "GFX12")
 * @return Pointer to initialized architecture structure, or NULL if unknown/unsupported
 *
 * @note Caller is responsible for freeing with arch_destroy()
 * @see arch_destroy(), create_gfx12_arch(), create_gfx9_arch()
 */
arch_t *arch_create_by_name(const char *arch_name);

/**
 * @brief Create a GFX12 architecture structure
 *
 * @return Pointer to initialized GFX12 architecture, or NULL on allocation failure
 * @note Caller must call arch_destroy() to free the returned structure
 */
arch_t *arch_create_gfx12(void);

/**
 * @brief Create a GFX9 architecture structure
 *
 * @return Pointer to initialized GFX9 (Vega/MI100/MI200) architecture, or NULL on allocation failure
 * @note Caller must call arch_destroy() to free the returned structure
 */
arch_t *arch_create_gfx9(void);

/**
 * @brief Destroy an architecture structure and free all memory
 *
 * Recursively frees all dynamically allocated components including blocks,
 * counters, dimensions, and command buffers.
 *
 * @param arch Pointer to architecture structure to destroy, or NULL (no-op)
 *
 * @note Safe to call with NULL pointer
 * @see arch_create_by_name(), arch_create_gfx12()
 */
void arch_destroy(arch_t *arch);

#endif /* ARCH_CREATOR_H */