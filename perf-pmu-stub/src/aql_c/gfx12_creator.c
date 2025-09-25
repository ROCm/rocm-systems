/**
 * @file gfx12_creator.c
 * @brief GFX12-specific architecture creation implementation
 */

#include "aql_structures.h"
#include "arch_creator_common.h"

#ifdef __KERNEL__
#include <linux/string.h>
#include <linux/slab.h>
#include <linux/errno.h>
#else
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#endif

/* GFX12 Register Offsets - from Rust offset.rs */
#define mmGRBM_GFX_INDEX                    49664
#define mmCP_PERFMON_CNTL                   55304
#define mmCOMPUTE_PERFCOUNT_ENABLE          11787
#define mmSQ_PERFCOUNTER_CTRL               55776
#define mmSQ_PERFCOUNTER_CTRL2              55778

/* CPC registers */
#define mmCPC_PERFCOUNTER0_SELECT           55305
#define mmCPC_PERFCOUNTER0_LO               53254
#define mmCPC_PERFCOUNTER0_HI               53255
#define mmCPC_PERFCOUNTER1_SELECT           55299
#define mmCPC_PERFCOUNTER1_LO               53252
#define mmCPC_PERFCOUNTER1_HI               53253

/* SQ registers */
#define mmSQ_PERFCOUNTER0_SELECT            55744
#define mmSQ_PERFCOUNTER0_LO                53696
#define mmSQ_PERFCOUNTER1_LO                53698
#define mmSQ_PERFCOUNTER2_SELECT            55746
#define mmSQ_PERFCOUNTER2_LO                53700
#define mmSQ_PERFCOUNTER4_SELECT            55748
#define mmSQ_PERFCOUNTER6_SELECT            55750
#define mmSQ_PERFCOUNTER8_SELECT            55752
#define mmSQ_PERFCOUNTER10_SELECT           55754
#define mmSQ_PERFCOUNTER12_SELECT           55756
#define mmSQ_PERFCOUNTER14_SELECT           55758
#define mmSQ_PERFCOUNTER3_LO                53702
#define mmSQ_PERFCOUNTER4_LO                53704
#define mmSQ_PERFCOUNTER5_LO                53706
#define mmSQ_PERFCOUNTER6_LO                53708
#define mmSQ_PERFCOUNTER7_LO                53710

/* GRBM registers */
#define mmGRBM_PERFCOUNTER0_SELECT          14400
#define mmGRBM_PERFCOUNTER0_LO              12352
#define mmGRBM_PERFCOUNTER0_HI              12353
#define mmGRBM_PERFCOUNTER1_SELECT          14401
#define mmGRBM_PERFCOUNTER1_LO              12355
#define mmGRBM_PERFCOUNTER1_HI              12356

/* GL2C registers */
#define mmGL2C_PERFCOUNTER0_SELECT          15232
#define mmGL2C_PERFCOUNTER0_LO              13184
#define mmGL2C_PERFCOUNTER0_HI              13185
#define mmGL2C_PERFCOUNTER1_SELECT          15234
#define mmGL2C_PERFCOUNTER1_LO              13186
#define mmGL2C_PERFCOUNTER1_HI              13187
#define mmGL2C_PERFCOUNTER2_SELECT          15236
#define mmGL2C_PERFCOUNTER2_LO              13188
#define mmGL2C_PERFCOUNTER2_HI              13189
#define mmGL2C_PERFCOUNTER3_SELECT          15238
#define mmGL2C_PERFCOUNTER3_LO              13190
#define mmGL2C_PERFCOUNTER3_HI              13191

/* SPI registers */
#define mmSPI_PERFCOUNTER0_SELECT           14720
#define mmSPI_PERFCOUNTER0_LO               12673
#define mmSPI_PERFCOUNTER0_HI               12672
#define mmSPI_PERFCOUNTER1_SELECT           14721
#define mmSPI_PERFCOUNTER1_LO               12675
#define mmSPI_PERFCOUNTER1_HI               12674
#define mmSPI_PERFCOUNTER2_SELECT           14722
#define mmSPI_PERFCOUNTER2_LO               12677
#define mmSPI_PERFCOUNTER2_HI               12676
#define mmSPI_PERFCOUNTER3_SELECT           14723
#define mmSPI_PERFCOUNTER3_LO               12679
#define mmSPI_PERFCOUNTER3_HI               12678
#define mmSPI_PERFCOUNTER4_SELECT           14724
#define mmSPI_PERFCOUNTER4_LO               12681
#define mmSPI_PERFCOUNTER4_HI               12680
#define mmSPI_PERFCOUNTER5_SELECT           14725
#define mmSPI_PERFCOUNTER5_LO               12683
#define mmSPI_PERFCOUNTER5_HI               12682

/* Block info constants - from Rust block_info.rs */
#define GFX12_CPC_COUNTER_BLOCK_NUM_COUNTERS      2
#define GFX12_CPC_COUNTER_BLOCK_MAX_EVENT         0x1F  /* Placeholder - actual from enums */
#define GFX12_SQ_COUNTER_BLOCK_NUM_COUNTERS       8
#define GFX12_SQ_COUNTER_BLOCK_MAX_EVENT          0xFF  /* Placeholder - actual from enums */
#define GFX12_GRBM_COUNTER_BLOCK_NUM_COUNTERS     2
#define GFX12_GRBM_COUNTER_BLOCK_MAX_EVENT        51
#define GFX12_GL2C_COUNTER_BLOCK_NUM_COUNTERS     4
#define GFX12_GL2C_COUNTER_BLOCK_MAX_EVENT        249
#define GFX12_GL2C_COUNTER_BLOCK_NUM_INSTANCES    16
#define GFX12_SPI_COUNTER_BLOCK_NUM_COUNTERS      6
#define GFX12_SPI_COUNTER_BLOCK_MAX_EVENT         318
#define GFX12_SPI_COUNTER_BLOCK_NUM_INSTANCES     1

/* Counter block attributes - from Rust enums */
#define GFX12_COUNTER_BLOCK_DFLT_ATTR             1
#define GFX12_COUNTER_BLOCK_SPM_GLOBAL_ATTR       0x1000

/* GFX12 Architecture parameters - from Rust demo.rs */
#define GFX12_NUM_XCC           1
#define GFX12_NUM_SE            4
#define GFX12_NUM_SA            2
#define GFX12_NUM_CU            64
#define GFX12_NUM_WGP_PER_SA    4

/* Create CPC block info for GFX12 */
static block_info_t* create_gfx12_cpc_block(void) {
    block_info_t* block = ALLOC(sizeof(block_info_t));
    if (!block) return NULL;

    /* Allocate counter register info array */
    counter_reg_info_t* counter_regs = ALLOC_ARRAY(counter_reg_info_t, GFX12_CPC_COUNTER_BLOCK_NUM_COUNTERS);
    if (!counter_regs) {
        FREE(block);
        return NULL;
    }

    /* Initialize CPC counter registers - based on Rust implementation */
    create_counter_reg_info(&counter_regs[0], mmCPC_PERFCOUNTER0_SELECT, 0,
                           mmCPC_PERFCOUNTER0_LO, mmCPC_PERFCOUNTER0_HI);
    create_counter_reg_info(&counter_regs[1], mmCPC_PERFCOUNTER1_SELECT, 0,
                           mmCPC_PERFCOUNTER1_LO, mmCPC_PERFCOUNTER1_HI);

    /* Create dimensions for CPC block - global block with no SE/SA dependencies */
    block->dimension_count = 1;
    block->dimensions = ALLOC_ARRAY(dimension_t, block->dimension_count);
    if (!block->dimensions) {
        FREE(counter_regs);
        FREE(block);
        return NULL;
    }
    block->dimensions[0] = (dimension_t){.size = GFX12_NUM_XCC, .dim = HARDWARE_DIM_XCC};

    block->name = "CPC";
    block->id = HW_IP_BLOCK_CPC;
    block->instance_count = 1;
    block->event_id_max = GFX12_CPC_COUNTER_BLOCK_MAX_EVENT;
    block->counter_count = GFX12_CPC_COUNTER_BLOCK_NUM_COUNTERS;
    block->counter_reg_info = counter_regs;
    block->attr = GFX12_COUNTER_BLOCK_DFLT_ATTR | GFX12_COUNTER_BLOCK_SPM_GLOBAL_ATTR;
    block->delay_info = NULL;
    block->spm_block_id = 0;

    return block;
}

/* Create SQ block info for GFX12 */
static block_info_t* create_gfx12_sq_block(void) {
    block_info_t* block = ALLOC(sizeof(block_info_t));
    if (!block) return NULL;

    /* Allocate counter register info array */
    counter_reg_info_t* counter_regs = ALLOC_ARRAY(counter_reg_info_t, GFX12_SQ_COUNTER_BLOCK_NUM_COUNTERS);
    if (!counter_regs) {
        FREE(block);
        return NULL;
    }

    /* Initialize SQ counter registers - based on Rust implementation */
    create_counter_reg_info(&counter_regs[0], mmSQ_PERFCOUNTER0_SELECT, mmSQ_PERFCOUNTER_CTRL,
                           mmSQ_PERFCOUNTER0_LO, 0);
    create_counter_reg_info(&counter_regs[1], mmSQ_PERFCOUNTER2_SELECT, mmSQ_PERFCOUNTER_CTRL,
                           mmSQ_PERFCOUNTER1_LO, 0);
    create_counter_reg_info(&counter_regs[2], mmSQ_PERFCOUNTER4_SELECT, mmSQ_PERFCOUNTER_CTRL,
                           mmSQ_PERFCOUNTER2_LO, 0);
    create_counter_reg_info(&counter_regs[3], mmSQ_PERFCOUNTER6_SELECT, mmSQ_PERFCOUNTER_CTRL,
                           mmSQ_PERFCOUNTER3_LO, 0);
    create_counter_reg_info(&counter_regs[4], mmSQ_PERFCOUNTER8_SELECT, mmSQ_PERFCOUNTER_CTRL,
                           mmSQ_PERFCOUNTER4_LO, 0);
    create_counter_reg_info(&counter_regs[5], mmSQ_PERFCOUNTER10_SELECT, mmSQ_PERFCOUNTER_CTRL,
                           mmSQ_PERFCOUNTER5_LO, 0);
    create_counter_reg_info(&counter_regs[6], mmSQ_PERFCOUNTER12_SELECT, mmSQ_PERFCOUNTER_CTRL,
                           mmSQ_PERFCOUNTER6_LO, 0);
    create_counter_reg_info(&counter_regs[7], mmSQ_PERFCOUNTER14_SELECT, mmSQ_PERFCOUNTER_CTRL,
                           mmSQ_PERFCOUNTER7_LO, 0);

    /* Create dimensions for SQ block - SE dependent block */
    /* Based on experiments: total dimensions multiply to 32 (4 SE × 2 SA × 4 WGP = 32) */
    block->dimension_count = 3;
    block->dimensions = ALLOC_ARRAY(dimension_t, block->dimension_count);
    if (!block->dimensions) {
        FREE(counter_regs);
        FREE(block);
        return NULL;
    }
    block->dimensions[0] = (dimension_t){.size = GFX12_NUM_SE, .dim = HARDWARE_DIM_SE};
    block->dimensions[1] = (dimension_t){.size = GFX12_NUM_SA, .dim = HARDWARE_DIM_SA};
    block->dimensions[2] = (dimension_t){.size = GFX12_NUM_WGP_PER_SA, .dim = HARDWARE_DIM_WGP};

    block->name = "SQ";
    block->id = HW_IP_BLOCK_SQ;
    block->instance_count = 1;
    block->event_id_max = GFX12_SQ_COUNTER_BLOCK_MAX_EVENT;
    block->counter_count = GFX12_SQ_COUNTER_BLOCK_NUM_COUNTERS;
    block->counter_reg_info = counter_regs;
    block->attr = GFX12_COUNTER_BLOCK_DFLT_ATTR;
    block->delay_info = NULL;
    block->spm_block_id = 9; /* SPM_SE_BLOCK_NAME_SQG from Rust */

    return block;
}

/* Create GRBM block info for GFX12 */
static block_info_t* create_gfx12_grbm_block(void) {
    block_info_t* block = ALLOC(sizeof(block_info_t));
    if (!block) return NULL;

    /* Allocate counter register info array */
    counter_reg_info_t* counter_regs = ALLOC_ARRAY(counter_reg_info_t, GFX12_GRBM_COUNTER_BLOCK_NUM_COUNTERS);
    if (!counter_regs) {
        FREE(block);
        return NULL;
    }

    /* Initialize GRBM counter registers - based on aqlprofile reference */
    create_counter_reg_info(&counter_regs[0], mmGRBM_PERFCOUNTER0_SELECT, 0,
                           mmGRBM_PERFCOUNTER0_LO, mmGRBM_PERFCOUNTER0_HI);
    create_counter_reg_info(&counter_regs[1], mmGRBM_PERFCOUNTER1_SELECT, 0,
                           mmGRBM_PERFCOUNTER1_LO, mmGRBM_PERFCOUNTER1_HI);

    /* Create dimensions for GRBM block - global block with no SE/SA dependencies */
    block->dimension_count = 1;
    block->dimensions = ALLOC_ARRAY(dimension_t, block->dimension_count);
    if (!block->dimensions) {
        FREE(counter_regs);
        FREE(block);
        return NULL;
    }
    block->dimensions[0] = (dimension_t){.size = GFX12_NUM_XCC, .dim = HARDWARE_DIM_XCC};

    block->name = "GRBM";
    block->id = HW_IP_BLOCK_GRBM;
    block->instance_count = 1;
    block->event_id_max = GFX12_GRBM_COUNTER_BLOCK_MAX_EVENT;
    block->counter_count = GFX12_GRBM_COUNTER_BLOCK_NUM_COUNTERS;
    block->counter_reg_info = counter_regs;
    block->attr = GFX12_COUNTER_BLOCK_DFLT_ATTR;
    block->delay_info = NULL;
    block->spm_block_id = 0;

    return block;
}

/* Create GL2C block info for GFX12 */
static block_info_t* create_gfx12_gl2c_block(void) {
    block_info_t* block = ALLOC(sizeof(block_info_t));
    if (!block) return NULL;

    /* Allocate counter register info array */
    counter_reg_info_t* counter_regs = ALLOC_ARRAY(counter_reg_info_t, GFX12_GL2C_COUNTER_BLOCK_NUM_COUNTERS);
    if (!counter_regs) {
        FREE(block);
        return NULL;
    }

    /* Initialize GL2C counter registers - based on aqlprofile reference */
    create_counter_reg_info(&counter_regs[0], mmGL2C_PERFCOUNTER0_SELECT, 0,
                           mmGL2C_PERFCOUNTER0_LO, mmGL2C_PERFCOUNTER0_HI);
    create_counter_reg_info(&counter_regs[1], mmGL2C_PERFCOUNTER1_SELECT, 0,
                           mmGL2C_PERFCOUNTER1_LO, mmGL2C_PERFCOUNTER1_HI);
    create_counter_reg_info(&counter_regs[2], mmGL2C_PERFCOUNTER2_SELECT, 0,
                           mmGL2C_PERFCOUNTER2_LO, mmGL2C_PERFCOUNTER2_HI);
    create_counter_reg_info(&counter_regs[3], mmGL2C_PERFCOUNTER3_SELECT, 0,
                           mmGL2C_PERFCOUNTER3_LO, mmGL2C_PERFCOUNTER3_HI);

    /* Create dimensions for GL2C block - global block with no SE/SA dependencies */
    block->dimension_count = 1;
    block->dimensions = ALLOC_ARRAY(dimension_t, block->dimension_count);
    if (!block->dimensions) {
        FREE(counter_regs);
        FREE(block);
        return NULL;
    }
    block->dimensions[0] = (dimension_t){.size = GFX12_NUM_XCC, .dim = HARDWARE_DIM_XCC};

    block->name = "GL2C";
    block->id = HW_IP_BLOCK_GL2C;
    block->instance_count = GFX12_GL2C_COUNTER_BLOCK_NUM_INSTANCES;
    block->event_id_max = GFX12_GL2C_COUNTER_BLOCK_MAX_EVENT;
    block->counter_count = GFX12_GL2C_COUNTER_BLOCK_NUM_COUNTERS;
    block->counter_reg_info = counter_regs;
    block->attr = GFX12_COUNTER_BLOCK_DFLT_ATTR;
    block->delay_info = NULL;
    block->spm_block_id = 0;

    return block;
}

/* Create SPI block info for GFX12 */
static block_info_t* create_gfx12_spi_block(void) {
    block_info_t* block = ALLOC(sizeof(block_info_t));
    if (!block) return NULL;

    /* Allocate counter register info array */
    counter_reg_info_t* counter_regs = ALLOC_ARRAY(counter_reg_info_t, GFX12_SPI_COUNTER_BLOCK_NUM_COUNTERS);
    if (!counter_regs) {
        FREE(block);
        return NULL;
    }

    /* Initialize SPI counter registers - based on aqlprofile reference */
    create_counter_reg_info(&counter_regs[0], mmSPI_PERFCOUNTER0_SELECT, 0,
                           mmSPI_PERFCOUNTER0_LO, mmSPI_PERFCOUNTER0_HI);
    create_counter_reg_info(&counter_regs[1], mmSPI_PERFCOUNTER1_SELECT, 0,
                           mmSPI_PERFCOUNTER1_LO, mmSPI_PERFCOUNTER1_HI);
    create_counter_reg_info(&counter_regs[2], mmSPI_PERFCOUNTER2_SELECT, 0,
                           mmSPI_PERFCOUNTER2_LO, mmSPI_PERFCOUNTER2_HI);
    create_counter_reg_info(&counter_regs[3], mmSPI_PERFCOUNTER3_SELECT, 0,
                           mmSPI_PERFCOUNTER3_LO, mmSPI_PERFCOUNTER3_HI);
    create_counter_reg_info(&counter_regs[4], mmSPI_PERFCOUNTER4_SELECT, 0,
                           mmSPI_PERFCOUNTER4_LO, mmSPI_PERFCOUNTER4_HI);
    create_counter_reg_info(&counter_regs[5], mmSPI_PERFCOUNTER5_SELECT, 0,
                           mmSPI_PERFCOUNTER5_LO, mmSPI_PERFCOUNTER5_HI);

    /* Create dimensions for SPI block - SE block with SE dimensions */
    block->dimension_count = 2;
    block->dimensions = ALLOC_ARRAY(dimension_t, block->dimension_count);
    if (!block->dimensions) {
        FREE(counter_regs);
        FREE(block);
        return NULL;
    }
    block->dimensions[0] = (dimension_t){.size = GFX12_NUM_XCC, .dim = HARDWARE_DIM_XCC};
    block->dimensions[1] = (dimension_t){.size = GFX12_NUM_SE, .dim = HARDWARE_DIM_SE};

    block->name = "SPI";
    block->id = HW_IP_BLOCK_SPI;
    block->instance_count = GFX12_SPI_COUNTER_BLOCK_NUM_INSTANCES;
    block->event_id_max = GFX12_SPI_COUNTER_BLOCK_MAX_EVENT;
    block->counter_count = GFX12_SPI_COUNTER_BLOCK_NUM_COUNTERS;
    block->counter_reg_info = counter_regs;
    block->attr = GFX12_COUNTER_BLOCK_DFLT_ATTR;
    block->delay_info = NULL;
    block->spm_block_id = 0;

    return block;
}

/* Create GFX12 architecture - based on Rust demo.rs values */
arch_t* create_gfx12_arch(void) {
    arch_t* arch = ALLOC(sizeof(arch_t));
    if (!arch) return NULL;

    /* Initialize architecture fields - from Rust demo.rs */
    arch->type = ARCH_TYPE_GFX12;
    arch->num_xcc = GFX12_NUM_XCC;
    arch->num_se = GFX12_NUM_SE;
    arch->num_sa = GFX12_NUM_SA;
    arch->num_cu = GFX12_NUM_CU;
    arch->num_wgp_per_sa = GFX12_NUM_WGP_PER_SA;
    arch->command = NULL; /* Will be allocated as needed */

    /* Initialize block map */
    memset(&arch->block_map, 0, sizeof(block_info_map_t));

    /* Create and add block info entries */
    block_info_t* cpc_block = create_gfx12_cpc_block();
    block_info_t* sq_block = create_gfx12_sq_block();
    block_info_t* grbm_block = create_gfx12_grbm_block();
    block_info_t* gl2c_block = create_gfx12_gl2c_block();
    block_info_t* spi_block = create_gfx12_spi_block();

    if (!cpc_block || !sq_block || !grbm_block || !gl2c_block || !spi_block) {
        FREE(arch);
        if (cpc_block) {
            FREE(cpc_block->dimensions);
            FREE(cpc_block->counter_reg_info);
            FREE(cpc_block);
        }
        if (sq_block) {
            FREE(sq_block->dimensions);
            FREE(sq_block->counter_reg_info);
            FREE(sq_block);
        }
        if (grbm_block) {
            FREE(grbm_block->dimensions);
            FREE(grbm_block->counter_reg_info);
            FREE(grbm_block);
        }
        if (gl2c_block) {
            FREE(gl2c_block->dimensions);
            FREE(gl2c_block->counter_reg_info);
            FREE(gl2c_block);
        }
        if (spi_block) {
            FREE(spi_block->dimensions);
            FREE(spi_block->counter_reg_info);
            FREE(spi_block);
        }
        return NULL;
    }

    /* Add blocks to the map */
    arch->block_map.blocks[HW_IP_BLOCK_CPC] = cpc_block;
    arch->block_map.blocks[HW_IP_BLOCK_SQ] = sq_block;
    arch->block_map.blocks[HW_IP_BLOCK_GRBM] = grbm_block;
    arch->block_map.blocks[HW_IP_BLOCK_GL2C] = gl2c_block;
    arch->block_map.blocks[HW_IP_BLOCK_SPI] = spi_block;
    arch->block_map.block_count = 5;

    return arch;
}