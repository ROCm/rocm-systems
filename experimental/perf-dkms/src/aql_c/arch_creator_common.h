/**
 * @file arch_creator_common.h
 * @brief Common definitions and functions for architecture creators
 */

#ifndef ARCH_CREATOR_COMMON_H
#define ARCH_CREATOR_COMMON_H

#include "aql_structures.h"

#ifdef __KERNEL__
#include <linux/string.h>
#include <linux/slab.h>
#include <linux/errno.h>
#define ALLOC(size) kmalloc(size, GFP_KERNEL)
#define FREE(ptr) kfree(ptr)
#define ALLOC_ARRAY(type, count) kmalloc_array(count, sizeof(type), GFP_KERNEL)
#else
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#define ALLOC(size) malloc(size)
#define FREE(ptr) free(ptr)
#define ALLOC_ARRAY(type, count) malloc((count) * sizeof(type))
#endif

/* Function declarations for architecture-specific creators */
arch_t *create_gfx12_arch(void);
arch_t *create_gfx9_arch(void);

/**
 * @brief Structure to hold decoded GPU topology dimension coordinates
 *
 * Represents a specific location in the GPU's hardware hierarchy by
 * SE (Shader Engine), SA (Shader Array), and WGP (Work Group Processor)
 * coordinates, along with the equivalent flat array index.
 *
 * This structure is used to convert between hierarchical (SE/SA/WGP) and
 * linear (flat array) addressing when reading performance counters from
 * different parts of the GPU.
 */
typedef struct {
	uint32_t se; /* Shader Engine index (0 to num_se-1) */
	uint32_t sa; /* Shader Array index within SE (0 to num_sa-1) */
	uint32_t wgp; /* Work Group Processor index within SA (0 to wgp_per_sa-1) */
	uint32_t flat_index; /* Equivalent flat array index */
} dimension_coords_t;

/**
 * @brief Calculate flat array index from SE/SA/WGP coordinates
 *
 * Converts hierarchical GPU topology coordinates (Shader Engine, Shader Array,
 * Work Group Processor) into a linear flat array index. Used when allocating
 * counter result buffers or iterating over GPU topology.
 *
 * Formula: index = (se * sa_count * wgp_per_sa) + (sa * wgp_per_sa) + wgp
 *
 * @param se Shader Engine index (0-based)
 * @param sa Shader Array index within SE (0-based)
 * @param wgp Work Group Processor index within SA (0-based)
 * @param sa_count Total number of Shader Arrays per SE
 * @param wgp_per_sa Total number of WGPs per SA
 * @return Flat array index representing this location
 *
 * @note For GFX12: 4 SEs × 2 SAs × 4 WGPs = 32 total locations
 * @see decode_dimension_index()
 */
static inline uint32_t encode_dimension_index(uint32_t se, uint32_t sa, uint32_t wgp,
					      uint32_t sa_count, uint32_t wgp_per_sa)
{
	return (se * sa_count * wgp_per_sa) + (sa * wgp_per_sa) + wgp;
}

/**
 * @brief Decode flat array index back to SE/SA/WGP coordinates
 *
 * Converts a linear flat array index back into hierarchical GPU topology
 * coordinates (Shader Engine, Shader Array, Work Group Processor). Used when
 * interpreting counter results stored in flat arrays.
 *
 * Inverse operation of encode_dimension_index().
 *
 * @param flat_index Linear array index to decode
 * @param sa_count Total number of Shader Arrays per SE
 * @param wgp_per_sa Total number of WGPs per SA
 * @return dimension_coords_t structure with decoded SE/SA/WGP coordinates
 *
 * @note Coordinates are calculated using integer division and modulo
 * @see encode_dimension_index(), validate_dimension_coords()
 */
static inline dimension_coords_t decode_dimension_index(uint32_t flat_index, uint32_t sa_count,
							uint32_t wgp_per_sa)
{
	dimension_coords_t coords;
	coords.flat_index = flat_index;

	/* Calculate coordinates using integer division */
	uint32_t sa_wgp_total = sa_count * wgp_per_sa;
	coords.se = flat_index / sa_wgp_total;
	uint32_t remainder = flat_index % sa_wgp_total;
	coords.sa = remainder / wgp_per_sa;
	coords.wgp = remainder % wgp_per_sa;

	return coords;
}

/**
 * @brief Validate that dimension coordinates are within valid bounds
 *
 * Checks that SE/SA/WGP coordinates are within the valid range for the
 * specified GPU topology. Used to catch out-of-bounds accesses before
 * they cause memory errors or invalid hardware register accesses.
 *
 * @param coords Pointer to dimension coordinates to validate
 * @param se_count Total number of Shader Engines on GPU
 * @param sa_count Total number of Shader Arrays per SE
 * @param wgp_per_sa Total number of WGPs per SA
 * @return 1 if coordinates are valid, 0 if invalid or coords is NULL
 *
 * @note Returns 0 (invalid) if coords pointer is NULL
 * @see decode_dimension_index(), encode_dimension_index()
 */
static inline int validate_dimension_coords(const dimension_coords_t *coords, uint32_t se_count,
					    uint32_t sa_count, uint32_t wgp_per_sa)
{
	if (!coords)
		return 0;
	return (coords->se < se_count && coords->sa < sa_count && coords->wgp < wgp_per_sa);
}

/**
 * @brief Initialize a counter register information structure
 *
 * Sets up a counter_reg_info_t structure with hardware register addresses
 * and initializes the allocation state to FREE. Used during architecture
 * block creation to define each performance counter's register layout.
 *
 * This function is analogous to the counter register setup in
 * projects/aqlprofile/gfxip/gfx12/gfx12_block_info.h where register
 * addresses are defined for each counter.
 *
 * @param reg_info Pointer to counter_reg_info_t to initialize
 * @param select_addr Register address for event selection (SELECT register)
 * @param control_addr Register address for counter control (0 if not used)
 * @param register_addr_lo Register address for counter value low 32 bits
 * @param register_addr_hi Register address for counter value high 32 bits (0 if 32-bit)
 *
 * @note control_addr and register_addr_hi may be 0 for counters without control
 *       registers or 32-bit counters
 * @note Initializes allocation state to COUNTER_STATE_FREE
 * @see counter_reg_info_t, projects/aqlprofile/gfxip/gfx12/gfx12_block_info.h
 */
static inline void create_counter_reg_info(counter_reg_info_t *reg_info, uint32_t select_addr,
					   uint32_t control_addr, uint32_t register_addr_lo,
					   uint32_t register_addr_hi)
{
	if (!reg_info)
		return;

	reg_info->select_addr = select_addr;
	reg_info->control_addr = control_addr;
	reg_info->register_addr_lo = register_addr_lo;
	reg_info->register_addr_hi = register_addr_hi;

	/* Initialize allocation info to FREE state */
	atomic_set(&reg_info->allocation.state, COUNTER_STATE_FREE);
	reg_info->allocation.event_id = 0;
	reg_info->allocation.instance_id = 0;
	reg_info->allocation.user_id = 0;
	reg_info->allocation.description = NULL;
	reg_info->allocation.allocation_time = 0;
	reg_info->allocation.data_cpu_addr = NULL;
	reg_info->allocation.data_gpu_addr = 0;
	reg_info->allocation.data_size = 0;
}

#endif /* ARCH_CREATOR_COMMON_H */