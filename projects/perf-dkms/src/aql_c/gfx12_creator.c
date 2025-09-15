/**
 * @file gfx12_creator.c
 * @brief GFX12-specific architecture creation implementation
 */

#include "aql_structures.h"
#include "arch_creator_common.h"
#include "gfx12_events.h"

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

/* Register space bases - from pm4_packets.h */
#define UCONFIG_SPACE_START                 0x0000C000
#define PERSISTENT_SPACE_START              0x00002C00

/* Event types - from pm4_packets.h */
#define VGT_EVENT_TYPE_CS_PARTIAL_FLUSH     0x07

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

/* TA registers */
#define mmTA_PERFCOUNTER0_SELECT            15040  /* 0x3ac0 */
#define mmTA_PERFCOUNTER0_LO                12992  /* 0x32c0 */
#define mmTA_PERFCOUNTER0_HI                12993  /* 0x32c1 */
#define mmTA_PERFCOUNTER1_SELECT            15042  /* 0x3ac2 */
#define mmTA_PERFCOUNTER1_LO                12994  /* 0x32c2 */
#define mmTA_PERFCOUNTER1_HI                12995  /* 0x32c3 */

/* TCP registers */
#define mmTCP_PERFCOUNTER0_SELECT           15168  /* 0x3b40 */
#define mmTCP_PERFCOUNTER0_LO               13120  /* 0x3340 */
#define mmTCP_PERFCOUNTER0_HI               13121  /* 0x3341 */
#define mmTCP_PERFCOUNTER1_SELECT           15170  /* 0x3b42 */
#define mmTCP_PERFCOUNTER1_LO               13122  /* 0x3342 */
#define mmTCP_PERFCOUNTER1_HI               13123  /* 0x3343 */
#define mmTCP_PERFCOUNTER2_SELECT           15172  /* 0x3b44 */
#define mmTCP_PERFCOUNTER2_LO               13124  /* 0x3344 */
#define mmTCP_PERFCOUNTER2_HI               13125  /* 0x3345 */
#define mmTCP_PERFCOUNTER3_SELECT           15174  /* 0x3b46 */
#define mmTCP_PERFCOUNTER3_LO               13126  /* 0x3346 */
#define mmTCP_PERFCOUNTER3_HI               13127  /* 0x3347 */

/* TD registers */
#define mmTD_PERFCOUNTER0_SELECT            15296  /* 0x3bc0 */
#define mmTD_PERFCOUNTER0_LO                13248  /* 0x33c0 */
#define mmTD_PERFCOUNTER0_HI                13249  /* 0x33c1 */
#define mmTD_PERFCOUNTER1_SELECT            15298  /* 0x3bc2 */
#define mmTD_PERFCOUNTER1_LO                13250  /* 0x33c2 */
#define mmTD_PERFCOUNTER1_HI                13251  /* 0x33c3 */

/* TCC registers */
#define mmTCC_PERFCOUNTER0_SELECT           15424  /* 0x3c40 */
#define mmTCC_PERFCOUNTER0_LO               13376  /* 0x3440 */
#define mmTCC_PERFCOUNTER0_HI               13377  /* 0x3441 */
#define mmTCC_PERFCOUNTER1_SELECT           15426  /* 0x3c42 */
#define mmTCC_PERFCOUNTER1_LO               13378  /* 0x3442 */
#define mmTCC_PERFCOUNTER1_HI               13379  /* 0x3443 */
#define mmTCC_PERFCOUNTER2_SELECT           15428  /* 0x3c44 */
#define mmTCC_PERFCOUNTER2_LO               13380  /* 0x3444 */
#define mmTCC_PERFCOUNTER2_HI               13381  /* 0x3445 */
#define mmTCC_PERFCOUNTER3_SELECT           15430  /* 0x3c46 */
#define mmTCC_PERFCOUNTER3_LO               13382  /* 0x3446 */
#define mmTCC_PERFCOUNTER3_HI               13383  /* 0x3447 */

/* SX registers */
#define mmSX_PERFCOUNTER0_SELECT            14976  /* 0x3a80 */
#define mmSX_PERFCOUNTER0_LO                12928  /* 0x3280 */
#define mmSX_PERFCOUNTER0_HI                12929  /* 0x3281 */
#define mmSX_PERFCOUNTER1_SELECT            14978  /* 0x3a82 */
#define mmSX_PERFCOUNTER1_LO                12930  /* 0x3282 */
#define mmSX_PERFCOUNTER1_HI                12931  /* 0x3283 */
#define mmSX_PERFCOUNTER2_SELECT            14980  /* 0x3a84 */
#define mmSX_PERFCOUNTER2_LO                12932  /* 0x3284 */
#define mmSX_PERFCOUNTER2_HI                12933  /* 0x3285 */
#define mmSX_PERFCOUNTER3_SELECT            14982  /* 0x3a86 */
#define mmSX_PERFCOUNTER3_LO                12934  /* 0x3286 */
#define mmSX_PERFCOUNTER3_HI                12935  /* 0x3287 */

/* DB registers */
#define mmDB_PERFCOUNTER0_SELECT            14592  /* 0x3900 */
#define mmDB_PERFCOUNTER0_LO                12800  /* 0x3200 */
#define mmDB_PERFCOUNTER0_HI                12801  /* 0x3201 */
#define mmDB_PERFCOUNTER1_SELECT            14594  /* 0x3902 */
#define mmDB_PERFCOUNTER1_LO                12802  /* 0x3202 */
#define mmDB_PERFCOUNTER1_HI                12803  /* 0x3203 */
#define mmDB_PERFCOUNTER2_SELECT            14596  /* 0x3904 */
#define mmDB_PERFCOUNTER2_LO                12804  /* 0x3204 */
#define mmDB_PERFCOUNTER2_HI                12805  /* 0x3205 */
#define mmDB_PERFCOUNTER3_SELECT            14598  /* 0x3906 */
#define mmDB_PERFCOUNTER3_LO                12806  /* 0x3206 */
#define mmDB_PERFCOUNTER3_HI                12807  /* 0x3207 */

/* PA_SC registers */
#define mmPA_SC_PERFCOUNTER0_SELECT         14208  /* 0x3780 */
#define mmPA_SC_PERFCOUNTER0_LO             12672  /* 0x3180 */
#define mmPA_SC_PERFCOUNTER0_HI             12673  /* 0x3181 */
#define mmPA_SC_PERFCOUNTER1_SELECT         14210  /* 0x3782 */
#define mmPA_SC_PERFCOUNTER1_LO             12674  /* 0x3182 */
#define mmPA_SC_PERFCOUNTER1_HI             12675  /* 0x3183 */
#define mmPA_SC_PERFCOUNTER2_SELECT         14212  /* 0x3784 */
#define mmPA_SC_PERFCOUNTER2_LO             12676  /* 0x3184 */
#define mmPA_SC_PERFCOUNTER2_HI             12677  /* 0x3185 */
#define mmPA_SC_PERFCOUNTER3_SELECT         14214  /* 0x3786 */
#define mmPA_SC_PERFCOUNTER3_LO             12678  /* 0x3186 */
#define mmPA_SC_PERFCOUNTER3_HI             12679  /* 0x3187 */

/* PA_SU registers */
#define mmPA_SU_PERFCOUNTER0_SELECT         14216  /* 0x3788 */
#define mmPA_SU_PERFCOUNTER0_LO             12680  /* 0x3188 */
#define mmPA_SU_PERFCOUNTER0_HI             12681  /* 0x3189 */
#define mmPA_SU_PERFCOUNTER1_SELECT         14218  /* 0x378a */
#define mmPA_SU_PERFCOUNTER1_LO             12682  /* 0x318a */
#define mmPA_SU_PERFCOUNTER1_HI             12683  /* 0x318b */
#define mmPA_SU_PERFCOUNTER2_SELECT         14220  /* 0x378c */
#define mmPA_SU_PERFCOUNTER2_LO             12684  /* 0x318c */
#define mmPA_SU_PERFCOUNTER2_HI             12685  /* 0x318d */
#define mmPA_SU_PERFCOUNTER3_SELECT         14222  /* 0x378e */
#define mmPA_SU_PERFCOUNTER3_LO             12686  /* 0x318e */
#define mmPA_SU_PERFCOUNTER3_HI             12687  /* 0x318f */

/* GDS registers */
#define mmGDS_PERFCOUNTER0_SELECT           14352  /* 0x3810 */
#define mmGDS_PERFCOUNTER0_LO               12816  /* 0x3210 */
#define mmGDS_PERFCOUNTER0_HI               12817  /* 0x3211 */
#define mmGDS_PERFCOUNTER1_SELECT           14354  /* 0x3812 */
#define mmGDS_PERFCOUNTER1_LO               12818  /* 0x3212 */
#define mmGDS_PERFCOUNTER1_HI               12819  /* 0x3213 */
#define mmGDS_PERFCOUNTER2_SELECT           14356  /* 0x3814 */
#define mmGDS_PERFCOUNTER2_LO               12820  /* 0x3214 */
#define mmGDS_PERFCOUNTER2_HI               12821  /* 0x3215 */
#define mmGDS_PERFCOUNTER3_SELECT           14358  /* 0x3816 */
#define mmGDS_PERFCOUNTER3_LO               12822  /* 0x3216 */
#define mmGDS_PERFCOUNTER3_HI               12823  /* 0x3217 */

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
#define GFX12_TA_COUNTER_BLOCK_NUM_COUNTERS       2
#define GFX12_TA_COUNTER_BLOCK_MAX_EVENT          254
#define GFX12_TA_COUNTER_BLOCK_NUM_INSTANCES      2
#define GFX12_TCP_COUNTER_BLOCK_NUM_COUNTERS      4
#define GFX12_TCP_COUNTER_BLOCK_MAX_EVENT         99
#define GFX12_TCP_COUNTER_BLOCK_NUM_INSTANCES     2
#define GFX12_TD_COUNTER_BLOCK_NUM_COUNTERS       2
#define GFX12_TD_COUNTER_BLOCK_MAX_EVENT          127
#define GFX12_TD_COUNTER_BLOCK_NUM_INSTANCES      2
#define GFX12_TCC_COUNTER_BLOCK_NUM_COUNTERS      4
#define GFX12_TCC_COUNTER_BLOCK_MAX_EVENT         255
#define GFX12_TCC_COUNTER_BLOCK_NUM_INSTANCES     16
#define GFX12_SX_COUNTER_BLOCK_NUM_COUNTERS       4
#define GFX12_SX_COUNTER_BLOCK_MAX_EVENT          189
#define GFX12_SX_COUNTER_BLOCK_NUM_INSTANCES      1
#define GFX12_DB_COUNTER_BLOCK_NUM_COUNTERS       4
#define GFX12_DB_COUNTER_BLOCK_MAX_EVENT          218
#define GFX12_DB_COUNTER_BLOCK_NUM_INSTANCES      1
#define GFX12_PA_SC_COUNTER_BLOCK_NUM_COUNTERS    4
#define GFX12_PA_SC_COUNTER_BLOCK_MAX_EVENT       171
#define GFX12_PA_SC_COUNTER_BLOCK_NUM_INSTANCES   1

#define GFX12_PA_SU_COUNTER_BLOCK_NUM_COUNTERS    4
#define GFX12_PA_SU_COUNTER_BLOCK_MAX_EVENT       171
#define GFX12_PA_SU_COUNTER_BLOCK_NUM_INSTANCES   1

#define GFX12_GDS_COUNTER_BLOCK_NUM_COUNTERS      4
#define GFX12_GDS_COUNTER_BLOCK_MAX_EVENT         121
#define GFX12_GDS_COUNTER_BLOCK_NUM_INSTANCES     1

/* Counter block attributes - from Rust enums */
#define GFX12_COUNTER_BLOCK_DFLT_ATTR             1
#define GFX12_COUNTER_BLOCK_SPM_GLOBAL_ATTR       0x1000

/* GFX12 Architecture parameters - from Rust demo.rs */
#define GFX12_NUM_XCC           1
#define GFX12_NUM_SE            4
#define GFX12_NUM_SA            2
#define GFX12_NUM_CU            64
#define GFX12_NUM_WGP_PER_SA    4

/**
 * @brief Create CPC (Command Processor Compute) block information for GFX12
 *
 * Initializes the block_info_t structure for the CPC hardware block, which
 * contains performance counters for command processor operations including
 * command buffer processing, packet dispatch, and compute engine activity.
 * CPC is a global block with no SE/SA/WGP dependencies.
 *
 * This function is analogous to the GFX12 CPC block definition in
 * projects/aqlprofile/gfxip/gfx12/gfx12_block_info.h and the BlockInfo
 * construction in projects/aqlprofile/src/core/gfx12_factory.cpp
 *
 * Configuration:
 * - 2 performance counters
 * - 1 XCC dimension (global scope)
 * - SELECT registers for event configuration
 * - 64-bit counter result registers (LO + HI)
 *
 * @return Pointer to initialized CPC block_info_t, or NULL on allocation failure
 *
 * @note CPC counters are global and do not require iteration over topology
 * @see create_gfx12_sq_block(), GFX12 block info in
 *      projects/aqlprofile/gfxip/gfx12/gfx12_block_info.h
 */
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

/**
 * @brief Create SQ (Shader Sequencer) block information for GFX12
 *
 * Initializes the block_info_t structure for the SQ hardware block, which
 * contains performance counters for shader execution including wave counts,
 * instruction counts, and wait states. SQ counters are per-WGP and require
 * iteration over the GPU's SE x SA x WGP topology when reading.
 *
 * This function is analogous to the GFX12 SQ block definition in
 * projects/aqlprofile/gfxip/gfx12/gfx12_block_info.h and the BlockInfo
 * construction in projects/aqlprofile/src/core/gfx12_factory.cpp
 *
 * Configuration:
 * - 8 performance counters per WGP
 * - 3D topology: 4 SE x 2 SA x 2 WGP = 16 instances
 * - SELECT registers for event configuration
 * - CTRL registers for shader stage enable (PS/GS/HS/CS)
 * - Counter result registers (LO only, 32-bit)
 *
 * @return Pointer to initialized SQ block_info_t, or NULL on allocation failure
 *
 * @note SQ dimension is 2 WGPs per SA (not the full 4 WGP_PER_SA)
 * @note Total counter instances: 8 counters × 16 locations = 128 readings
 * @see create_gfx12_cpc_block(), GFX12 block info in
 *      projects/aqlprofile/gfxip/gfx12/gfx12_block_info.h
 */
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
    /* Based on experiments: total dimensions multiply to 16 (4 SE × 2 SA × 2 WGP = 16) */
    block->dimension_count = 3;
    block->dimensions = ALLOC_ARRAY(dimension_t, block->dimension_count);
    if (!block->dimensions) {
        FREE(counter_regs);
        FREE(block);
        return NULL;
    }
    block->dimensions[0] = (dimension_t){.size = GFX12_NUM_SE, .dim = HARDWARE_DIM_SE};
    block->dimensions[1] = (dimension_t){.size = GFX12_NUM_SA, .dim = HARDWARE_DIM_SA};
    /* SQ block has 2 WGP per SA (not GFX12_NUM_WGP_PER_SA which is 4) */
    block->dimensions[2] = (dimension_t){.size = 2, .dim = HARDWARE_DIM_WGP};

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

/**
 * @brief Create GRBM (Graphics Register Bus Manager) block information for GFX12
 *
 * Initializes the block_info_t structure for the GRBM hardware block, which
 * provides performance counters for GPU-wide activity including GUI pipeline
 * utilization, command buffer processing, and overall GPU busy state.
 * GRBM is a global block monitoring system-level GPU activities.
 *
 * This function is analogous to the GFX12 GRBM block definition in
 * projects/aqlprofile/gfxip/gfx12/gfx12_block_info.h
 *
 * Configuration:
 * - 2 performance counters
 * - 1 XCC dimension (global scope)
 * - SELECT registers for event configuration
 * - 64-bit counter result registers (LO + HI)
 *
 * @return Pointer to initialized GRBM block_info_t, or NULL on allocation failure
 *
 * @note GRBM counters monitor system-level GPU activity
 * @see create_gfx12_cpc_block(), GFX12 block info in
 *      projects/aqlprofile/gfxip/gfx12/gfx12_block_info.h
 */
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

/**
 * @brief Create GL2C (Graphics L2 Cache Channel) block information for GFX12
 *
 * Initializes the block_info_t structure for the GL2C hardware block, which
 * provides performance counters for L2 cache operations including hits, misses,
 * read/write requests, and memory traffic. GL2C has multiple instances (16 on GFX12)
 * that can be monitored independently or aggregated.
 *
 * This function is analogous to the GFX12 GL2C block definition in
 * projects/aqlprofile/gfxip/gfx12/gfx12_block_info.h
 *
 * Configuration:
 * - 4 performance counters per instance
 * - 16 instances across the GPU
 * - 1 XCC dimension (global scope)
 * - SELECT registers for event configuration
 * - 64-bit counter result registers (LO + HI)
 *
 * @return Pointer to initialized GL2C block_info_t, or NULL on allocation failure
 *
 * @note GL2C monitors L2 cache behavior critical for memory performance analysis
 * @see create_gfx12_tcc_block(), GFX12 block info in
 *      projects/aqlprofile/gfxip/gfx12/gfx12_block_info.h
 */
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

/**
 * @brief Create SPI (Shader Processor Input) block information for GFX12
 *
 * Initializes the block_info_t structure for the SPI hardware block, which
 * provides performance counters for shader input operations including wave
 * allocation, resource allocation, and shader launch activity. SPI is
 * per-SE (Shader Engine) and monitors shader dispatch behavior.
 *
 * This function is analogous to the GFX12 SPI block definition in
 * projects/aqlprofile/gfxip/gfx12/gfx12_block_info.h
 *
 * Configuration:
 * - 6 performance counters
 * - XCC + SE dimensions (4 SEs on GFX12)
 * - SELECT registers for event configuration
 * - 64-bit counter result registers (LO + HI)
 *
 * @return Pointer to initialized SPI block_info_t, or NULL on allocation failure
 *
 * @note SPI counters must be read separately for each SE
 * @see create_gfx12_sq_block(), GFX12 block info in
 *      projects/aqlprofile/gfxip/gfx12/gfx12_block_info.h
 */
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

/**
 * @brief Create TA (Texture Addresser) block information for GFX12
 *
 * Initializes the block_info_t structure for the TA hardware block, which
 * provides performance counters for texture address calculation including
 * texture load/store operations, address generation, and filtering operations.
 * TA is a WGP-level block with full topology dimensions.
 *
 * This function is analogous to the GFX12 TA block definition in
 * projects/aqlprofile/gfxip/gfx12/gfx12_block_info.h
 *
 * Configuration:
 * - 2 performance counters per instance
 * - 2 instances per WGP
 * - Full 4D topology: XCC x SE x SA x WGP
 * - SELECT registers for event configuration
 * - 64-bit counter result registers (LO + HI)
 *
 * @return Pointer to initialized TA block_info_t, or NULL on allocation failure
 *
 * @note TA counters require iteration over full GPU topology (SE/SA/WGP)
 * @see create_gfx12_tcp_block(), create_gfx12_td_block(), GFX12 block info in
 *      projects/aqlprofile/gfxip/gfx12/gfx12_block_info.h
 */
static block_info_t* create_gfx12_ta_block(void) {
    block_info_t* block = ALLOC(sizeof(block_info_t));
    if (!block) return NULL;

    /* Allocate counter register info array */
    counter_reg_info_t* counter_regs = ALLOC_ARRAY(counter_reg_info_t, GFX12_TA_COUNTER_BLOCK_NUM_COUNTERS);
    if (!counter_regs) {
        FREE(block);
        return NULL;
    }

    /* Initialize TA counter registers - based on aqlprofile reference */
    create_counter_reg_info(&counter_regs[0], mmTA_PERFCOUNTER0_SELECT, 0,
                           mmTA_PERFCOUNTER0_LO, mmTA_PERFCOUNTER0_HI);
    create_counter_reg_info(&counter_regs[1], mmTA_PERFCOUNTER1_SELECT, 0,
                           mmTA_PERFCOUNTER1_LO, mmTA_PERFCOUNTER1_HI);

    /* Allocate dimensions - TA is a WGP block with XCC+SE+SA+WGP structure (4 dimensions) */
    dimension_t* dimensions = ALLOC_ARRAY(dimension_t, 4);
    if (!dimensions) {
        FREE(counter_regs);
        FREE(block);
        return NULL;
    }

    /* Initialize WGP dimensions - based on aqlprofile CounterBlockWgpAttr */
    dimensions[0].dim = HARDWARE_DIM_XCC;
    dimensions[0].size = GFX12_NUM_XCC;
    dimensions[1].dim = HARDWARE_DIM_SE;
    dimensions[1].size = GFX12_NUM_SE;
    dimensions[2].dim = HARDWARE_DIM_SA;
    dimensions[2].size = GFX12_NUM_SA;
    dimensions[3].dim = HARDWARE_DIM_WGP;
    dimensions[3].size = GFX12_NUM_WGP_PER_SA;

    /* Initialize TA block */
    block->name = "TA";
    block->id = HW_IP_BLOCK_TA;
    block->instance_count = GFX12_TA_COUNTER_BLOCK_NUM_INSTANCES; /* 2 instances */
    block->event_id_max = GFX12_TA_COUNTER_BLOCK_MAX_EVENT; /* 254 */
    block->counter_count = GFX12_TA_COUNTER_BLOCK_NUM_COUNTERS; /* 2 counters */
    block->dimensions = dimensions;
    block->dimension_count = 4; /* XCC, SE, SA, WGP */
    block->counter_reg_info = counter_regs;
    block->attr = GFX12_COUNTER_BLOCK_DFLT_ATTR;
    block->delay_info = NULL;
    block->spm_block_id = 0;

    return block;
}

/**
 * @brief Create TCP (Texture Cache Processor) block information for GFX12
 *
 * Initializes the block_info_t structure for the TCP hardware block, which
 * provides performance counters for texture cache operations including
 * texture fetch requests, cache hits/misses, and data transfer rates.
 * TCP is a WGP-level block responsible for L1 texture cache.
 *
 * This function is analogous to the GFX12 TCP block definition in
 * projects/aqlprofile/gfxip/gfx12/gfx12_block_info.h
 *
 * Configuration:
 * - 4 performance counters per instance
 * - 2 instances per WGP
 * - Full 4D topology: XCC x SE x SA x WGP
 * - SELECT registers for event configuration
 * - 64-bit counter result registers (LO + HI)
 *
 * @return Pointer to initialized TCP block_info_t, or NULL on allocation failure
 *
 * @note TCP L1 cache is critical for texture memory performance
 * @see create_gfx12_ta_block(), create_gfx12_tcc_block(), GFX12 block info in
 *      projects/aqlprofile/gfxip/gfx12/gfx12_block_info.h
 */
static block_info_t* create_gfx12_tcp_block(void) {
    block_info_t* block = ALLOC(sizeof(block_info_t));
    if (!block) return NULL;

    /* Allocate counter register info array */
    counter_reg_info_t* counter_regs = ALLOC_ARRAY(counter_reg_info_t, GFX12_TCP_COUNTER_BLOCK_NUM_COUNTERS);
    if (!counter_regs) {
        FREE(block);
        return NULL;
    }

    /* Initialize TCP counter registers - based on aqlprofile reference */
    create_counter_reg_info(&counter_regs[0], mmTCP_PERFCOUNTER0_SELECT, 0,
                           mmTCP_PERFCOUNTER0_LO, mmTCP_PERFCOUNTER0_HI);
    create_counter_reg_info(&counter_regs[1], mmTCP_PERFCOUNTER1_SELECT, 0,
                           mmTCP_PERFCOUNTER1_LO, mmTCP_PERFCOUNTER1_HI);
    create_counter_reg_info(&counter_regs[2], mmTCP_PERFCOUNTER2_SELECT, 0,
                           mmTCP_PERFCOUNTER2_LO, mmTCP_PERFCOUNTER2_HI);
    create_counter_reg_info(&counter_regs[3], mmTCP_PERFCOUNTER3_SELECT, 0,
                           mmTCP_PERFCOUNTER3_LO, mmTCP_PERFCOUNTER3_HI);

    /* Allocate dimensions - TCP is a WGP block with XCC+SE+SA+WGP structure (4 dimensions) */
    dimension_t* dimensions = ALLOC_ARRAY(dimension_t, 4);
    if (!dimensions) {
        FREE(counter_regs);
        FREE(block);
        return NULL;
    }

    /* Initialize WGP dimensions - based on aqlprofile CounterBlockWgpAttr */
    dimensions[0].dim = HARDWARE_DIM_XCC;
    dimensions[0].size = GFX12_NUM_XCC;
    dimensions[1].dim = HARDWARE_DIM_SE;
    dimensions[1].size = GFX12_NUM_SE;
    dimensions[2].dim = HARDWARE_DIM_SA;
    dimensions[2].size = GFX12_NUM_SA;
    dimensions[3].dim = HARDWARE_DIM_WGP;
    dimensions[3].size = GFX12_NUM_WGP_PER_SA;

    /* Initialize TCP block */
    block->name = "TCP";
    block->id = HW_IP_BLOCK_TCP;
    block->instance_count = GFX12_TCP_COUNTER_BLOCK_NUM_INSTANCES; /* 2 instances */
    block->event_id_max = GFX12_TCP_COUNTER_BLOCK_MAX_EVENT; /* 99 */
    block->counter_count = GFX12_TCP_COUNTER_BLOCK_NUM_COUNTERS; /* 4 counters */
    block->dimensions = dimensions;
    block->dimension_count = 4; /* XCC, SE, SA, WGP */
    block->counter_reg_info = counter_regs;
    block->attr = GFX12_COUNTER_BLOCK_DFLT_ATTR;
    block->delay_info = NULL;
    block->spm_block_id = 0;

    return block;
}

/**
 * @brief Create TD (Texture Data) block information for GFX12
 *
 * Initializes the block_info_t structure for the TD hardware block, which
 * provides performance counters for texture data fetch operations including
 * texture load operations, texture filtering, and memory read requests.
 * TD is a WGP-level block handling texture data retrieval.
 *
 * This function is analogous to the GFX12 TD block definition in
 * projects/aqlprofile/gfxip/gfx12/gfx12_block_info.h
 *
 * Configuration:
 * - 2 performance counters per instance
 * - 2 instances per WGP
 * - Full 4D topology: XCC x SE x SA x WGP
 * - SELECT registers for event configuration
 * - 64-bit counter result registers (LO + HI)
 *
 * @return Pointer to initialized TD block_info_t, or NULL on allocation failure
 *
 * @note TD works with TA and TCP to complete texture operations
 * @see create_gfx12_ta_block(), create_gfx12_tcp_block(), GFX12 block info in
 *      projects/aqlprofile/gfxip/gfx12/gfx12_block_info.h
 */
static block_info_t* create_gfx12_td_block(void) {
    block_info_t* block = ALLOC(sizeof(block_info_t));
    if (!block) return NULL;

    /* Allocate counter register info array */
    counter_reg_info_t* counter_regs = ALLOC_ARRAY(counter_reg_info_t, GFX12_TD_COUNTER_BLOCK_NUM_COUNTERS);
    if (!counter_regs) {
        FREE(block);
        return NULL;
    }

    /* Initialize TD counter registers - based on aqlprofile reference */
    create_counter_reg_info(&counter_regs[0], mmTD_PERFCOUNTER0_SELECT, 0,
                           mmTD_PERFCOUNTER0_LO, mmTD_PERFCOUNTER0_HI);
    create_counter_reg_info(&counter_regs[1], mmTD_PERFCOUNTER1_SELECT, 0,
                           mmTD_PERFCOUNTER1_LO, mmTD_PERFCOUNTER1_HI);

    /* Allocate dimensions - TD is a WGP block with XCC+SE+SA+WGP structure (4 dimensions) */
    dimension_t* dimensions = ALLOC_ARRAY(dimension_t, 4);
    if (!dimensions) {
        FREE(counter_regs);
        FREE(block);
        return NULL;
    }

    /* Initialize WGP dimensions - based on aqlprofile CounterBlockWgpAttr */
    dimensions[0].dim = HARDWARE_DIM_XCC;
    dimensions[0].size = GFX12_NUM_XCC;
    dimensions[1].dim = HARDWARE_DIM_SE;
    dimensions[1].size = GFX12_NUM_SE;
    dimensions[2].dim = HARDWARE_DIM_SA;
    dimensions[2].size = GFX12_NUM_SA;
    dimensions[3].dim = HARDWARE_DIM_WGP;
    dimensions[3].size = GFX12_NUM_WGP_PER_SA;

    /* Initialize TD block */
    block->name = "TD";
    block->id = HW_IP_BLOCK_TD;
    block->instance_count = GFX12_TD_COUNTER_BLOCK_NUM_INSTANCES; /* 2 instances */
    block->event_id_max = GFX12_TD_COUNTER_BLOCK_MAX_EVENT; /* 127 */
    block->counter_count = GFX12_TD_COUNTER_BLOCK_NUM_COUNTERS; /* 2 counters */
    block->dimensions = dimensions;
    block->dimension_count = 4; /* XCC, SE, SA, WGP */
    block->counter_reg_info = counter_regs;
    block->attr = GFX12_COUNTER_BLOCK_DFLT_ATTR;
    block->delay_info = NULL;
    block->spm_block_id = 0;

    return block;
}

/**
 * @brief Create TCC (Texture Cache Controller) block information for GFX12
 *
 * Initializes the block_info_t structure for the TCC hardware block, which
 * provides performance counters for L2 texture cache operations including
 * cache hits/misses, read/write requests, and coherency operations.
 * TCC has multiple instances (16 on GFX12) managing L2 cache slices.
 *
 * This function is analogous to the GFX12 TCC block definition in
 * projects/aqlprofile/gfxip/gfx12/gfx12_block_info.h
 *
 * Configuration:
 * - 4 performance counters per instance
 * - 16 instances across the GPU
 * - 1 XCC dimension (global scope)
 * - SELECT registers for event configuration
 * - 64-bit counter result registers (LO + HI)
 *
 * @return Pointer to initialized TCC block_info_t, or NULL on allocation failure
 *
 * @note TCC L2 cache is critical for overall memory hierarchy performance
 * @see create_gfx12_tcp_block(), create_gfx12_gl2c_block(), GFX12 block info in
 *      projects/aqlprofile/gfxip/gfx12/gfx12_block_info.h
 */
static block_info_t* create_gfx12_tcc_block(void) {
    block_info_t* block = ALLOC(sizeof(block_info_t));
    if (!block) return NULL;

    /* Allocate counter register info array */
    counter_reg_info_t* counter_regs = ALLOC_ARRAY(counter_reg_info_t, GFX12_TCC_COUNTER_BLOCK_NUM_COUNTERS);
    if (!counter_regs) {
        FREE(block);
        return NULL;
    }

    /* Initialize TCC counter registers - based on aqlprofile reference */
    create_counter_reg_info(&counter_regs[0], mmTCC_PERFCOUNTER0_SELECT, 0,
                           mmTCC_PERFCOUNTER0_LO, mmTCC_PERFCOUNTER0_HI);
    create_counter_reg_info(&counter_regs[1], mmTCC_PERFCOUNTER1_SELECT, 0,
                           mmTCC_PERFCOUNTER1_LO, mmTCC_PERFCOUNTER1_HI);
    create_counter_reg_info(&counter_regs[2], mmTCC_PERFCOUNTER2_SELECT, 0,
                           mmTCC_PERFCOUNTER2_LO, mmTCC_PERFCOUNTER2_HI);
    create_counter_reg_info(&counter_regs[3], mmTCC_PERFCOUNTER3_SELECT, 0,
                           mmTCC_PERFCOUNTER3_LO, mmTCC_PERFCOUNTER3_HI);

    /* Create dimensions for TCC block - global block with no SE/SA dependencies */
    block->dimension_count = 1;
    block->dimensions = ALLOC_ARRAY(dimension_t, block->dimension_count);
    if (!block->dimensions) {
        FREE(counter_regs);
        FREE(block);
        return NULL;
    }
    block->dimensions[0] = (dimension_t){.size = GFX12_NUM_XCC, .dim = HARDWARE_DIM_XCC};

    block->name = "TCC";
    block->id = HW_IP_BLOCK_TCC;
    block->instance_count = GFX12_TCC_COUNTER_BLOCK_NUM_INSTANCES;
    block->event_id_max = GFX12_TCC_COUNTER_BLOCK_MAX_EVENT;
    block->counter_count = GFX12_TCC_COUNTER_BLOCK_NUM_COUNTERS;
    block->counter_reg_info = counter_regs;
    block->attr = GFX12_COUNTER_BLOCK_DFLT_ATTR;
    block->delay_info = NULL;
    block->spm_block_id = 0;

    return block;
}

/**
 * @brief Create SX (Shader Export) block information for GFX12
 *
 * Initializes the block_info_t structure for the SX hardware block, which
 * provides performance counters for pixel and vertex shader export operations
 * including color/depth exports, parameter cache activity, and memory writes.
 * SX is per-SE and handles shader output data.
 *
 * This function is analogous to the GFX12 SX block definition in
 * projects/aqlprofile/gfxip/gfx12/gfx12_block_info.h
 *
 * Configuration:
 * - 4 performance counters
 * - XCC + SE dimensions (4 SEs on GFX12)
 * - SELECT registers for event configuration
 * - 64-bit counter result registers (LO + HI)
 *
 * @return Pointer to initialized SX block_info_t, or NULL on allocation failure
 *
 * @note SX counters track shader output and export efficiency
 * @see create_gfx12_db_block(), GFX12 block info in
 *      projects/aqlprofile/gfxip/gfx12/gfx12_block_info.h
 */
static block_info_t* create_gfx12_sx_block(void) {
    block_info_t* block = ALLOC(sizeof(block_info_t));
    if (!block) return NULL;

    /* Allocate counter register info array */
    counter_reg_info_t* counter_regs = ALLOC_ARRAY(counter_reg_info_t, GFX12_SX_COUNTER_BLOCK_NUM_COUNTERS);
    if (!counter_regs) {
        FREE(block);
        return NULL;
    }

    /* Initialize SX counter registers - based on aqlprofile reference */
    create_counter_reg_info(&counter_regs[0], mmSX_PERFCOUNTER0_SELECT, 0,
                           mmSX_PERFCOUNTER0_LO, mmSX_PERFCOUNTER0_HI);
    create_counter_reg_info(&counter_regs[1], mmSX_PERFCOUNTER1_SELECT, 0,
                           mmSX_PERFCOUNTER1_LO, mmSX_PERFCOUNTER1_HI);
    create_counter_reg_info(&counter_regs[2], mmSX_PERFCOUNTER2_SELECT, 0,
                           mmSX_PERFCOUNTER2_LO, mmSX_PERFCOUNTER2_HI);
    create_counter_reg_info(&counter_regs[3], mmSX_PERFCOUNTER3_SELECT, 0,
                           mmSX_PERFCOUNTER3_LO, mmSX_PERFCOUNTER3_HI);

    /* Create dimensions for SX block - SE block with SE dimensions */
    block->dimension_count = 2;
    block->dimensions = ALLOC_ARRAY(dimension_t, block->dimension_count);
    if (!block->dimensions) {
        FREE(counter_regs);
        FREE(block);
        return NULL;
    }
    block->dimensions[0] = (dimension_t){.size = GFX12_NUM_XCC, .dim = HARDWARE_DIM_XCC};
    block->dimensions[1] = (dimension_t){.size = GFX12_NUM_SE, .dim = HARDWARE_DIM_SE};

    block->name = "SX";
    block->id = HW_IP_BLOCK_SX;
    block->instance_count = GFX12_SX_COUNTER_BLOCK_NUM_INSTANCES;
    block->event_id_max = GFX12_SX_COUNTER_BLOCK_MAX_EVENT;
    block->counter_count = GFX12_SX_COUNTER_BLOCK_NUM_COUNTERS;
    block->counter_reg_info = counter_regs;
    block->attr = GFX12_COUNTER_BLOCK_DFLT_ATTR;
    block->delay_info = NULL;
    block->spm_block_id = 0;

    return block;
}

/**
 * @brief Create DB (Depth Buffer) block information for GFX12
 *
 * Initializes the block_info_t structure for the DB hardware block, which
 * provides performance counters for depth/stencil operations including
 * Z-test operations, hierarchical-Z activity, and depth buffer writes.
 * DB is per-SE and manages depth/stencil rendering.
 *
 * This function is analogous to the GFX12 DB block definition in
 * projects/aqlprofile/gfxip/gfx12/gfx12_block_info.h
 *
 * Configuration:
 * - 4 performance counters
 * - XCC + SE dimensions (4 SEs on GFX12)
 * - SELECT registers for event configuration
 * - 64-bit counter result registers (LO + HI)
 *
 * @return Pointer to initialized DB block_info_t, or NULL on allocation failure
 *
 * @note DB counters are essential for depth/stencil performance analysis
 * @see create_gfx12_sx_block(), create_gfx12_pa_sc_block(), GFX12 block info in
 *      projects/aqlprofile/gfxip/gfx12/gfx12_block_info.h
 */
static block_info_t* create_gfx12_db_block(void) {
    block_info_t* block = ALLOC(sizeof(block_info_t));
    if (!block) return NULL;

    /* Allocate counter register info array */
    counter_reg_info_t* counter_regs = ALLOC_ARRAY(counter_reg_info_t, GFX12_DB_COUNTER_BLOCK_NUM_COUNTERS);
    if (!counter_regs) {
        FREE(block);
        return NULL;
    }

    /* Initialize DB counter registers - based on aqlprofile reference */
    create_counter_reg_info(&counter_regs[0], mmDB_PERFCOUNTER0_SELECT, 0,
                           mmDB_PERFCOUNTER0_LO, mmDB_PERFCOUNTER0_HI);
    create_counter_reg_info(&counter_regs[1], mmDB_PERFCOUNTER1_SELECT, 0,
                           mmDB_PERFCOUNTER1_LO, mmDB_PERFCOUNTER1_HI);
    create_counter_reg_info(&counter_regs[2], mmDB_PERFCOUNTER2_SELECT, 0,
                           mmDB_PERFCOUNTER2_LO, mmDB_PERFCOUNTER2_HI);
    create_counter_reg_info(&counter_regs[3], mmDB_PERFCOUNTER3_SELECT, 0,
                           mmDB_PERFCOUNTER3_LO, mmDB_PERFCOUNTER3_HI);

    /* Create dimensions for DB block - SE block with SE dimensions */
    block->dimension_count = 2;
    block->dimensions = ALLOC_ARRAY(dimension_t, block->dimension_count);
    if (!block->dimensions) {
        FREE(counter_regs);
        FREE(block);
        return NULL;
    }
    block->dimensions[0] = (dimension_t){.size = GFX12_NUM_XCC, .dim = HARDWARE_DIM_XCC};
    block->dimensions[1] = (dimension_t){.size = GFX12_NUM_SE, .dim = HARDWARE_DIM_SE};

    block->name = "DB";
    block->id = HW_IP_BLOCK_DB;
    block->instance_count = GFX12_DB_COUNTER_BLOCK_NUM_INSTANCES;
    block->event_id_max = GFX12_DB_COUNTER_BLOCK_MAX_EVENT;
    block->counter_count = GFX12_DB_COUNTER_BLOCK_NUM_COUNTERS;
    block->counter_reg_info = counter_regs;
    block->attr = GFX12_COUNTER_BLOCK_DFLT_ATTR;
    block->delay_info = NULL;
    block->spm_block_id = 0;

    return block;
}

/**
 * @brief Create PA_SC (Primitive Assembly - Scan Converter) block information for GFX12
 *
 * Initializes the block_info_t structure for the PA_SC hardware block, which
 * provides performance counters for scan conversion including primitive processing,
 * quad generation, small primitive culling, and rasterization activity.
 * PA_SC is per-SE and handles geometry-to-pixel conversion.
 *
 * This function is analogous to the GFX12 PA_SC block definition in
 * projects/aqlprofile/gfxip/gfx12/gfx12_block_info.h
 *
 * Configuration:
 * - 4 performance counters
 * - XCC + SE dimensions (4 SEs on GFX12)
 * - SELECT registers for event configuration
 * - 64-bit counter result registers (LO + HI)
 *
 * @return Pointer to initialized PA_SC block_info_t, or NULL on allocation failure
 *
 * @note PA_SC is critical for understanding rasterization performance
 * @see create_gfx12_pa_su_block(), GFX12 block info in
 *      projects/aqlprofile/gfxip/gfx12/gfx12_block_info.h
 */
static block_info_t* create_gfx12_pa_sc_block(void) {
    block_info_t* block = ALLOC(sizeof(block_info_t));
    if (!block) return NULL;

    /* Allocate counter register info array */
    counter_reg_info_t* counter_regs = ALLOC_ARRAY(counter_reg_info_t, GFX12_PA_SC_COUNTER_BLOCK_NUM_COUNTERS);
    if (!counter_regs) {
        FREE(block);
        return NULL;
    }

    /* Initialize PA_SC counter registers - based on aqlprofile reference */
    create_counter_reg_info(&counter_regs[0], mmPA_SC_PERFCOUNTER0_SELECT, 0,
                           mmPA_SC_PERFCOUNTER0_LO, mmPA_SC_PERFCOUNTER0_HI);
    create_counter_reg_info(&counter_regs[1], mmPA_SC_PERFCOUNTER1_SELECT, 0,
                           mmPA_SC_PERFCOUNTER1_LO, mmPA_SC_PERFCOUNTER1_HI);
    create_counter_reg_info(&counter_regs[2], mmPA_SC_PERFCOUNTER2_SELECT, 0,
                           mmPA_SC_PERFCOUNTER2_LO, mmPA_SC_PERFCOUNTER2_HI);
    create_counter_reg_info(&counter_regs[3], mmPA_SC_PERFCOUNTER3_SELECT, 0,
                           mmPA_SC_PERFCOUNTER3_LO, mmPA_SC_PERFCOUNTER3_HI);

    /* Create dimensions for PA_SC block - SE block with SE dimensions */
    block->dimension_count = 2;
    block->dimensions = ALLOC_ARRAY(dimension_t, block->dimension_count);
    if (!block->dimensions) {
        FREE(counter_regs);
        FREE(block);
        return NULL;
    }
    block->dimensions[0] = (dimension_t){.size = GFX12_NUM_XCC, .dim = HARDWARE_DIM_XCC};
    block->dimensions[1] = (dimension_t){.size = GFX12_NUM_SE, .dim = HARDWARE_DIM_SE};

    block->name = "PA_SC";
    block->id = HW_IP_BLOCK_PA_SC;
    block->instance_count = GFX12_PA_SC_COUNTER_BLOCK_NUM_INSTANCES;
    block->event_id_max = GFX12_PA_SC_COUNTER_BLOCK_MAX_EVENT;
    block->counter_count = GFX12_PA_SC_COUNTER_BLOCK_NUM_COUNTERS;
    block->counter_reg_info = counter_regs;
    block->attr = GFX12_COUNTER_BLOCK_DFLT_ATTR;
    block->delay_info = NULL;
    block->spm_block_id = 0;

    return block;
}

/**
 * @brief Create PA_SU (Primitive Assembly - Setup Unit) block information for GFX12
 *
 * Initializes the block_info_t structure for the PA_SU hardware block, which
 * provides performance counters for primitive setup operations including
 * triangle setup, clipping, culling, and viewport transformation.
 * PA_SU is per-SE and prepares primitives for rasterization.
 *
 * This function is analogous to the GFX12 PA_SU block definition in
 * projects/aqlprofile/gfxip/gfx12/gfx12_block_info.h
 *
 * Configuration:
 * - 4 performance counters
 * - XCC + SE dimensions (4 SEs on GFX12)
 * - SELECT registers for event configuration
 * - 64-bit counter result registers (LO + HI)
 *
 * @return Pointer to initialized PA_SU block_info_t, or NULL on allocation failure
 *
 * @note PA_SU tracks geometry setup efficiency
 * @see create_gfx12_pa_sc_block(), GFX12 block info in
 *      projects/aqlprofile/gfxip/gfx12/gfx12_block_info.h
 */
static block_info_t* create_gfx12_pa_su_block(void) {
    block_info_t* block = ALLOC(sizeof(block_info_t));
    if (!block) return NULL;

    /* Allocate counter register info array */
    counter_reg_info_t* counter_regs = ALLOC_ARRAY(counter_reg_info_t, GFX12_PA_SU_COUNTER_BLOCK_NUM_COUNTERS);
    if (!counter_regs) {
        FREE(block);
        return NULL;
    }

    /* Initialize PA_SU counter registers - based on aqlprofile reference */
    create_counter_reg_info(&counter_regs[0], mmPA_SU_PERFCOUNTER0_SELECT, 0,
                           mmPA_SU_PERFCOUNTER0_LO, mmPA_SU_PERFCOUNTER0_HI);
    create_counter_reg_info(&counter_regs[1], mmPA_SU_PERFCOUNTER1_SELECT, 0,
                           mmPA_SU_PERFCOUNTER1_LO, mmPA_SU_PERFCOUNTER1_HI);
    create_counter_reg_info(&counter_regs[2], mmPA_SU_PERFCOUNTER2_SELECT, 0,
                           mmPA_SU_PERFCOUNTER2_LO, mmPA_SU_PERFCOUNTER2_HI);
    create_counter_reg_info(&counter_regs[3], mmPA_SU_PERFCOUNTER3_SELECT, 0,
                           mmPA_SU_PERFCOUNTER3_LO, mmPA_SU_PERFCOUNTER3_HI);

    /* Create dimensions for PA_SU block - SE block with SE dimensions */
    block->dimension_count = 2;
    block->dimensions = ALLOC_ARRAY(dimension_t, block->dimension_count);
    if (!block->dimensions) {
        FREE(counter_regs);
        FREE(block);
        return NULL;
    }

    block->dimensions[0] = (dimension_t){.size = GFX12_NUM_XCC, .dim = HARDWARE_DIM_XCC};
    block->dimensions[1] = (dimension_t){.size = GFX12_NUM_SE, .dim = HARDWARE_DIM_SE};

    block->name = "PA_SU";
    block->id = HW_IP_BLOCK_PA_SU;
    block->instance_count = GFX12_PA_SU_COUNTER_BLOCK_NUM_INSTANCES;
    block->event_id_max = GFX12_PA_SU_COUNTER_BLOCK_MAX_EVENT;
    block->counter_count = GFX12_PA_SU_COUNTER_BLOCK_NUM_COUNTERS;
    block->counter_reg_info = counter_regs;
    block->attr = GFX12_COUNTER_BLOCK_DFLT_ATTR;
    block->delay_info = NULL;
    block->spm_block_id = 0;

    return block;
}

/**
 * @brief Create GDS (Global Data Store) block information for GFX12
 *
 * Initializes the block_info_t structure for the GDS hardware block, which
 * provides performance counters for global data share operations including
 * atomic operations, ordered append/consume, and inter-wave communication.
 * GDS is a global block providing fast shared memory for compute shaders.
 *
 * This function is analogous to the GFX12 GDS block definition in
 * projects/aqlprofile/gfxip/gfx12/gfx12_block_info.h
 *
 * Configuration:
 * - 4 performance counters
 * - 1 XCC dimension (global scope)
 * - SELECT registers for event configuration
 * - 64-bit counter result registers (LO + HI)
 *
 * @return Pointer to initialized GDS block_info_t, or NULL on allocation failure
 *
 * @note GDS is used for compute shader synchronization and data sharing
 * @see create_gfx12_cpc_block(), GFX12 block info in
 *      projects/aqlprofile/gfxip/gfx12/gfx12_block_info.h
 */
static block_info_t* create_gfx12_gds_block(void) {
    block_info_t* block = ALLOC(sizeof(block_info_t));
    if (!block) return NULL;

    /* Allocate counter register info array */
    counter_reg_info_t* counter_regs = ALLOC_ARRAY(counter_reg_info_t, GFX12_GDS_COUNTER_BLOCK_NUM_COUNTERS);
    if (!counter_regs) {
        FREE(block);
        return NULL;
    }

    /* Initialize GDS counter registers - based on aqlprofile reference */
    create_counter_reg_info(&counter_regs[0], mmGDS_PERFCOUNTER0_SELECT, 0,
                           mmGDS_PERFCOUNTER0_LO, mmGDS_PERFCOUNTER0_HI);
    create_counter_reg_info(&counter_regs[1], mmGDS_PERFCOUNTER1_SELECT, 0,
                           mmGDS_PERFCOUNTER1_LO, mmGDS_PERFCOUNTER1_HI);
    create_counter_reg_info(&counter_regs[2], mmGDS_PERFCOUNTER2_SELECT, 0,
                           mmGDS_PERFCOUNTER2_LO, mmGDS_PERFCOUNTER2_HI);
    create_counter_reg_info(&counter_regs[3], mmGDS_PERFCOUNTER3_SELECT, 0,
                           mmGDS_PERFCOUNTER3_LO, mmGDS_PERFCOUNTER3_HI);

    /* Create dimensions for GDS block - XCC block with XCC dimensions only */
    block->dimension_count = 1;
    block->dimensions = ALLOC_ARRAY(dimension_t, block->dimension_count);
    if (!block->dimensions) {
        FREE(counter_regs);
        FREE(block);
        return NULL;
    }

    block->dimensions[0] = (dimension_t){.size = GFX12_NUM_XCC, .dim = HARDWARE_DIM_XCC};

    block->name = "GDS";
    block->id = HW_IP_BLOCK_GDS;
    block->instance_count = GFX12_GDS_COUNTER_BLOCK_NUM_INSTANCES;
    block->event_id_max = GFX12_GDS_COUNTER_BLOCK_MAX_EVENT;
    block->counter_count = GFX12_GDS_COUNTER_BLOCK_NUM_COUNTERS;
    block->counter_reg_info = counter_regs;
    block->attr = GFX12_COUNTER_BLOCK_DFLT_ATTR;
    block->delay_info = NULL;
    block->spm_block_id = 0;

    return block;
}

/**
 * @brief Create and initialize the complete GFX12 architecture structure
 *
 * Constructs the full architecture descriptor for AMD RDNA 3 (GFX12) GPUs,
 * including all hardware block definitions, register mappings, control registers,
 * and event mappings. This is the top-level factory function for GFX12.
 *
 * This function is analogous to Pm4Factory::Gfx12Create() in
 * projects/aqlprofile/src/core/gfx12_factory.cpp which creates the GFX12-specific
 * factory with CmdBuilder, PmcBuilder, and BlockInfo structures.
 *
 * Architecture configuration includes:
 * - Hardware topology: 4 SEs x 2 SAs x 4 WGPs = 64 CUs
 * - 14 hardware blocks (SQ, CPC, GL2C, GRBM, SPI, TA, TCP, TD, TCC, SX, DB, PA_SC, PA_SU, GDS)
 * - Per-block counter registers and dimensions
 * - Control register offsets for PM4 packet generation
 * - Event ID mappings from counter names to hardware event codes
 *
 * @return Pointer to initialized GFX12 architecture, or NULL on allocation failure
 *
 * @note Caller must call arch_destroy() to free the returned structure
 * @note All 14 block creators must succeed or the entire arch creation fails
 * @see arch_destroy(), Pm4Factory::Gfx12Create in
 *      projects/aqlprofile/src/core/gfx12_factory.cpp
 */
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
    block_info_t* ta_block = create_gfx12_ta_block();
    block_info_t* tcp_block = create_gfx12_tcp_block();
    block_info_t* td_block = create_gfx12_td_block();
    block_info_t* tcc_block = create_gfx12_tcc_block();
    block_info_t* sx_block = create_gfx12_sx_block();
    block_info_t* db_block = create_gfx12_db_block();
    block_info_t* pa_sc_block = create_gfx12_pa_sc_block();
    block_info_t* pa_su_block = create_gfx12_pa_su_block();
    block_info_t* gds_block = create_gfx12_gds_block();

    if (!cpc_block || !sq_block || !grbm_block || !gl2c_block || !spi_block || !ta_block || !tcp_block || !td_block || !tcc_block || !sx_block || !db_block || !pa_sc_block || !pa_su_block || !gds_block) {
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
        if (ta_block) {
            FREE(ta_block->dimensions);
            FREE(ta_block->counter_reg_info);
            FREE(ta_block);
        }
        if (tcp_block) {
            FREE(tcp_block->dimensions);
            FREE(tcp_block->counter_reg_info);
            FREE(tcp_block);
        }
        if (td_block) {
            FREE(td_block->dimensions);
            FREE(td_block->counter_reg_info);
            FREE(td_block);
        }
        if (tcc_block) {
            FREE(tcc_block->dimensions);
            FREE(tcc_block->counter_reg_info);
            FREE(tcc_block);
        }
        if (sx_block) {
            FREE(sx_block->dimensions);
            FREE(sx_block->counter_reg_info);
            FREE(sx_block);
        }
        if (db_block) {
            FREE(db_block->dimensions);
            FREE(db_block->counter_reg_info);
            FREE(db_block);
        }
        if (pa_sc_block) {
            FREE(pa_sc_block->dimensions);
            FREE(pa_sc_block->counter_reg_info);
            FREE(pa_sc_block);
        }
        if (pa_su_block) {
            FREE(pa_su_block->dimensions);
            FREE(pa_su_block->counter_reg_info);
            FREE(pa_su_block);
        }
        if (gds_block) {
            FREE(gds_block->dimensions);
            FREE(gds_block->counter_reg_info);
            FREE(gds_block);
        }
        return NULL;
    }

    /* Add blocks to the map */
    arch->block_map.blocks[HW_IP_BLOCK_CPC] = cpc_block;
    arch->block_map.blocks[HW_IP_BLOCK_SQ] = sq_block;
    arch->block_map.blocks[HW_IP_BLOCK_GRBM] = grbm_block;
    arch->block_map.blocks[HW_IP_BLOCK_GL2C] = gl2c_block;
    arch->block_map.blocks[HW_IP_BLOCK_SPI] = spi_block;
    arch->block_map.blocks[HW_IP_BLOCK_TA] = ta_block;
    arch->block_map.blocks[HW_IP_BLOCK_TCP] = tcp_block;
    arch->block_map.blocks[HW_IP_BLOCK_TD] = td_block;
    arch->block_map.blocks[HW_IP_BLOCK_TCC] = tcc_block;
    arch->block_map.blocks[HW_IP_BLOCK_SX] = sx_block;
    arch->block_map.blocks[HW_IP_BLOCK_DB] = db_block;
    arch->block_map.blocks[HW_IP_BLOCK_PA_SC] = pa_sc_block;
    arch->block_map.blocks[HW_IP_BLOCK_PA_SU] = pa_su_block;
    arch->block_map.blocks[HW_IP_BLOCK_GDS] = gds_block;
    arch->block_map.block_count = 14;

    /* Initialize control registers for GFX12 */
    arch->control_regs = (arch_control_regs_t) {
        .grbm_gfx_index = mmGRBM_GFX_INDEX,                    /* 49664 */
        .cp_perfmon_cntl = mmCP_PERFMON_CNTL,                  /* 55304 */
        .compute_perfcount_enable = mmCOMPUTE_PERFCOUNT_ENABLE, /* 11787 */
        .sq_perfcounter_ctrl2 = mmSQ_PERFCOUNTER_CTRL2,        /* 55778 */
        .uconfig_space_start = UCONFIG_SPACE_START,            /* 0x0000C000 */
        .persistent_space_start = PERSISTENT_SPACE_START,       /* 0x00002C00 */
        .cs_partial_flush_event = VGT_EVENT_TYPE_CS_PARTIAL_FLUSH, /* 0x07 */
        .event_index_flush = 4,                                /* Event flush index */
        .gcr_cntl_default = 0x00018000,                       /* From Rust impl */
        .poll_interval_default = 0x10,                        /* Poll every 4K */
        .counter_control_bits = {
            .sq_ps_en_bit = 0,                                /* PS enable bit */
            .sq_gs_en_bit = 2,                                /* GS enable bit */
            .sq_hs_en_bit = 4,                                /* HS enable bit */
            .sq_cs_en_bit = 6                                 /* CS enable bit */
        },
        .perfmon_states = {
            .perfmon_state_disable = 0,                       /* Disable state */
            .perfmon_state_enable = 1,                        /* Enable state */
            .perfmon_state_stop = 2,                          /* Stop state */
            .perfmon_sample_bit = 10                          /* Sample enable bit */
        }
    };

    /* Set up the event map for this architecture */
    arch->event_map = get_gfx12_events();
    arch->event_count = get_gfx12_event_count();

    return arch;
}