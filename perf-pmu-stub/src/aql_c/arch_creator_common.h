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
arch_t* create_gfx12_arch(void);
/* Future architecture creators can be declared here:
 * arch_t* create_gfx11_arch(void);
 * arch_t* create_gfx10_arch(void);
 * arch_t* create_gfx9_arch(void);
 */

/* Structure to hold decoded dimension information */
typedef struct {
    uint32_t se;    /* Shader Engine index */
    uint32_t sa;    /* Shader Array index */
    uint32_t wgp;   /* Work Group Processor index */
    uint32_t flat_index; /* Original flat array index */
} dimension_coords_t;

/* Helper function to calculate flat array index from SE/SA/WGP coordinates */
static inline uint32_t encode_dimension_index(uint32_t se, uint32_t sa, uint32_t wgp,
                                              uint32_t sa_count, uint32_t wgp_per_sa) {
    return (se * sa_count * wgp_per_sa) + (sa * wgp_per_sa) + wgp;
}

/* Helper function to decode flat array index back to SE/SA/WGP coordinates */
static inline dimension_coords_t decode_dimension_index(uint32_t flat_index,
                                                        uint32_t sa_count, uint32_t wgp_per_sa) {
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

/* Helper function to validate dimension coordinates */
static inline int validate_dimension_coords(const dimension_coords_t* coords,
                                            uint32_t se_count, uint32_t sa_count, uint32_t wgp_per_sa) {
    if (!coords) return 0;
    return (coords->se < se_count &&
            coords->sa < sa_count &&
            coords->wgp < wgp_per_sa);
}

/* Common helper function to initialize counter register info */
static inline void create_counter_reg_info(counter_reg_info_t* reg_info, uint32_t select_addr, uint32_t control_addr,
                                           uint32_t register_addr_lo, uint32_t register_addr_hi) {
    if (!reg_info) return;

    reg_info->select_addr = select_addr;
    reg_info->control_addr = control_addr;
    reg_info->register_addr_lo = register_addr_lo;
    reg_info->register_addr_hi = register_addr_hi;

    /* Initialize allocation info to FREE state */
    reg_info->allocation.state = COUNTER_STATE_FREE;
    reg_info->allocation.event_id = 0;
    reg_info->allocation.instance_id = 0;
    reg_info->allocation.user_id = 0;
    reg_info->allocation.description = NULL;
    reg_info->allocation.allocation_time = 0;
}

#endif /* ARCH_CREATOR_COMMON_H */