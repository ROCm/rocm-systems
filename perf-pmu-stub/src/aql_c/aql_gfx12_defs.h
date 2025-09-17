/**
 * @file aql_gfx12_defs.h
 * @brief GFX12 (RDNA3) architecture definitions and constants
 *
 * This file contains GFX12-specific definitions extracted from the SOC24
 * enumeration files and register definitions. These constants are used
 * by the GFX12 architecture implementation.
 */

#ifndef AQL_GFX12_DEFS_H
#define AQL_GFX12_DEFS_H

#ifdef __KERNEL__
#include <linux/types.h>
#else
#include <stdint.h>
#endif
#include "aql_types.h"

/*
 * Register Space Definitions (from soc24_enum.h)
 */

/* CONFIG_SPACE - Privileged configuration registers */
#define AQL_GFX12_CONFIG_SPACE_START           0x00002000
#define AQL_GFX12_CONFIG_SPACE_END             0x00009fff

/* UCONFIG_SPACE - User configuration registers */
#define AQL_GFX12_UCONFIG_SPACE_START          0x0000c000
#define AQL_GFX12_UCONFIG_SPACE_END            0x0000ffff

/*
 * PM4 Packet Opcodes (common across RDNA architectures)
 */
#define AQL_GFX12_PACKET3_EVENT_WRITE          0x46
#define AQL_GFX12_PACKET3_ACQUIRE_MEM          0x58
#define AQL_GFX12_PACKET3_WAIT_REG_MEM         0x3C
#define AQL_GFX12_PACKET3_WRITE_DATA           0x37
#define AQL_GFX12_PACKET3_SET_CONFIG_REG       0x68
#define AQL_GFX12_PACKET3_SET_UCONFIG_REG      0x79
#define AQL_GFX12_PACKET3_SET_SH_REG           0x76
#define AQL_GFX12_PACKET3_INDIRECT_BUFFER      0x3F
#define AQL_GFX12_PACKET3_NOP                  0x10

/*
 * Event Types for EVENT_WRITE packets
 */
#define AQL_GFX12_CS_PARTIAL_FLUSH             0x07
#define AQL_GFX12_THREAD_TRACE_FINISH          0x24

/*
 * PM4 Packet Header Construction
 */
#define AQL_GFX12_PACKET3(opcode, count) \
    (0xC0000000 | (((count) & 0x3FFF) << 16) | ((opcode) & 0xFF))

/*
 * PACKET3_EVENT_WRITE field definitions
 */
#define AQL_GFX12_EVENT_WRITE__EVENT_TYPE_MASK     0x3F
#define AQL_GFX12_EVENT_WRITE__EVENT_TYPE(x)       ((x) & 0x3F)
#define AQL_GFX12_EVENT_WRITE__EVENT_INDEX_MASK    0xF00
#define AQL_GFX12_EVENT_WRITE__EVENT_INDEX(x)      (((x) & 0xF) << 8)

/* Event index values */
#define AQL_GFX12_EVENT_INDEX_CS_PARTIAL_FLUSH     0x4
#define AQL_GFX12_EVENT_INDEX_OTHER                0x0

/*
 * PACKET3_ACQUIRE_MEM field definitions
 */
#define AQL_GFX12_ACQUIRE_MEM__COHER_SIZE(x)       ((x) & 0xFFFFFFFF)
#define AQL_GFX12_ACQUIRE_MEM__COHER_SIZE_HI(x)    ((x) & 0xFFFFFFFF)
#define AQL_GFX12_ACQUIRE_MEM__COHER_BASE_LO(x)    ((x) & 0xFFFFFFFF)
#define AQL_GFX12_ACQUIRE_MEM__COHER_BASE_HI(x)    ((x) & 0xFF)
#define AQL_GFX12_ACQUIRE_MEM__POLL_INTERVAL(x)    ((x) & 0xFFFF)
#define AQL_GFX12_ACQUIRE_MEM__GCR_CNTL(x)         ((x) & 0xFFFFFFFF)

/* GCR_CNTL values for cache control */
#define AQL_GFX12_GCR_CNTL_SEQ_FORWARD            0x00010000L
#define AQL_GFX12_GCR_CNTL_SEQ_MASK               0x00030000L
#define AQL_GFX12_GCR_CNTL_GL2_WB_MASK            0x00008000L

/*
 * PACKET3_WAIT_REG_MEM field definitions
 */
#define AQL_GFX12_WAIT_REG_MEM__OPERATION_MASK     0x7
#define AQL_GFX12_WAIT_REG_MEM__OPERATION(x)       ((x) & 0x7)
#define AQL_GFX12_WAIT_REG_MEM__MEM_SPACE_MASK     0x10
#define AQL_GFX12_WAIT_REG_MEM__MEM_SPACE(x)       (((x) & 0x1) << 4)
#define AQL_GFX12_WAIT_REG_MEM__FUNCTION_MASK      0x700
#define AQL_GFX12_WAIT_REG_MEM__FUNCTION(x)        (((x) & 0x7) << 8)
#define AQL_GFX12_WAIT_REG_MEM__POLL_INTERVAL(x)   ((x) & 0xFFFF)

/* Operation values */
#define AQL_GFX12_WAIT_REG_MEM_OP_WAIT_REG_MEM     0x0

/* Memory space values */
#define AQL_GFX12_WAIT_REG_MEM_SPACE_REGISTER      0x0
#define AQL_GFX12_WAIT_REG_MEM_SPACE_MEMORY        0x1

/* Function values */
#define AQL_GFX12_WAIT_REG_MEM_FUNC_EQUAL          0x3
#define AQL_GFX12_WAIT_REG_MEM_FUNC_NOT_EQUAL      0x4

/*
 * PACKET3_SET_*_REG field definitions
 */
#define AQL_GFX12_SET_REG__REG_OFFSET(x)           ((x) & 0xFFFF)

/*
 * PACKET3_WRITE_DATA field definitions
 */
#define AQL_GFX12_WRITE_DATA__DST_SEL_MASK         0x7
#define AQL_GFX12_WRITE_DATA__DST_SEL(x)           ((x) & 0x7)
#define AQL_GFX12_WRITE_DATA__WR_CONFIRM_MASK      0x100000
#define AQL_GFX12_WRITE_DATA__WR_CONFIRM(x)        (((x) & 0x1) << 20)

/* Destination selection values */
#define AQL_GFX12_WRITE_DATA_DST_MMREG             0x0
#define AQL_GFX12_WRITE_DATA_DST_TC_L2             0x2
#define AQL_GFX12_WRITE_DATA_DST_GDS               0x3

/*
 * PACKET3_INDIRECT_BUFFER field definitions
 */
#define AQL_GFX12_IB__IB_BASE_LO(x)                ((x) & 0xFFFFFFFC)
#define AQL_GFX12_IB__IB_BASE_HI(x)                ((x) & 0xFFFF)
#define AQL_GFX12_IB__IB_SIZE(x)                   ((x) & 0xFFFFF)

/*
 * GFX12 Architecture Capabilities
 */
#define AQL_GFX12_HAS_PRED_EXEC                    false  /* No PRED_EXEC on RDNA3 */
#define AQL_GFX12_HAS_UCONFIG_SPACE               true   /* UCONFIG space available */
#define AQL_GFX12_HAS_DUAL_SDMA                   true   /* Dual SDMA engines */
#define AQL_GFX12_HAS_GL_CACHE_HIERARCHY          true   /* GL1/GL2 cache blocks */
#define AQL_GFX12_HAS_SMN_ADDRESSING              false  /* No SMN addressing */
#define AQL_GFX12_HAS_WGP_SUPPORT                 true   /* Workgroup Processor support */

/*
 * GFX12 Real Hardware Performance Counter Select Registers
 * These are actual hardware register addresses extracted from GFX12 register definitions
 * Source: /linux/registers/gc/gc_12_0_0_offset.h
 */
#define AQL_GFX12_PERF_SEL_BASE_CB                   0x3C01  /* Real hardware address */
#define AQL_GFX12_PERF_SEL_BASE_CPC                  0x3809  /* Real hardware address */
#define AQL_GFX12_PERF_SEL_BASE_CPF                  0x3807  /* Real hardware address */
#define AQL_GFX12_PERF_SEL_BASE_CPG                  0x3802  /* Real hardware address */
#define AQL_GFX12_PERF_SEL_BASE_DB                   0x3C40  /* Real hardware address */
#define AQL_GFX12_PERF_SEL_BASE_GDS                  0x3A80  /* Placeholder - not found in GFX12 */
#define AQL_GFX12_PERF_SEL_BASE_GRBM                 0x3840  /* Real hardware address */
#define AQL_GFX12_PERF_SEL_BASE_GRBMSE               0x8040  /* Placeholder - not found in GFX12 */
#define AQL_GFX12_PERF_SEL_BASE_PA_SC                0x3940  /* Real hardware address */
#define AQL_GFX12_PERF_SEL_BASE_PA_SU                0x3900  /* Real hardware address */
#define AQL_GFX12_PERF_SEL_BASE_SPI                  0x2400  /* Placeholder - not found in GFX12 */
#define AQL_GFX12_PERF_SEL_BASE_SQ                   0x39C0  /* Real hardware address */
#define AQL_GFX12_PERF_SEL_BASE_SX                   0x2100  /* Placeholder - not found in GFX12 */
#define AQL_GFX12_PERF_SEL_BASE_TA                   0x3AC0  /* Real hardware address */
#define AQL_GFX12_PERF_SEL_BASE_TCP                  0x3B40  /* Real hardware address */
#define AQL_GFX12_PERF_SEL_BASE_TD                   0x3B00  /* Real hardware address */
#define AQL_GFX12_PERF_SEL_BASE_TCA                  0x3400  /* Placeholder - not found in GFX12 */
#define AQL_GFX12_PERF_SEL_BASE_GL2C                 0x3B80  /* Real hardware address (replaces TCC) */

/*
 * GFX12 Performance Counter Register Offsets
 */
#define AQL_GFX12_PERF_COUNTER_SELECT_OFFSET         0x00
#define AQL_GFX12_PERF_COUNTER_SELECT1_OFFSET        0x01
#define AQL_GFX12_PERF_COUNTER_LO_OFFSET             0x02
#define AQL_GFX12_PERF_COUNTER_HI_OFFSET             0x03

/*
 * GFX12 Counter Block Information
 * These definitions map to the unified block IDs in aql_types.h
 */

/* Core graphics blocks available in GFX12 */
#define AQL_GFX12_SUPPORTED_BLOCKS { \
    AQL_BLOCK_CB,        /* Color Buffer */ \
    AQL_BLOCK_CPC,       /* Command Processor Compute */ \
    AQL_BLOCK_CPF,       /* Command Processor Fetch */ \
    AQL_BLOCK_CPG,       /* Command Processor Graphics */ \
    AQL_BLOCK_DB,        /* Depth Buffer */ \
    AQL_BLOCK_GDS,       /* Global Data Share */ \
    AQL_BLOCK_GRBM,      /* Graphics Register Bus Manager */ \
    AQL_BLOCK_GRBM_SE,   /* GRBM Shader Engine */ \
    AQL_BLOCK_SPI,       /* Shader Processor Input */ \
    AQL_BLOCK_SQ,        /* Shader Quad */ \
    AQL_BLOCK_SQ_GS,     /* SQ Geometry Shader */ \
    AQL_BLOCK_SQ_PS,     /* SQ Pixel Shader */ \
    AQL_BLOCK_SQ_HS,     /* SQ Hull Shader */ \
    AQL_BLOCK_SQ_CS,     /* SQ Compute Shader */ \
    AQL_BLOCK_SX,        /* Shader Export */ \
    AQL_BLOCK_TA,        /* Texture Addresser */ \
    AQL_BLOCK_TCP,       /* Texture Cache Per-pipe */ \
    AQL_BLOCK_TD,        /* Texture Data */ \
    /* RDNA3 specific cache blocks */ \
    AQL_BLOCK_GL1A,      /* GL1 Texture Cache Arbiter */ \
    AQL_BLOCK_GL1C,      /* GL1 Texture Cache Controller */ \
    AQL_BLOCK_GL2A,      /* GL2 Cache Arbiter */ \
    AQL_BLOCK_GL2C,      /* GL2 Cache Controller */ \
    AQL_BLOCK_GCR,       /* Graphics Command Ring */ \
    AQL_BLOCK_GUS,       /* Graphics Unified Scheduler */ \
    /* Enhanced system blocks */ \
    AQL_BLOCK_SDMA0,     /* SDMA Engine 0 */ \
    AQL_BLOCK_SDMA1,     /* SDMA Engine 1 */ \
    AQL_BLOCK_UMC,       /* Unified Memory Controller */ \
    AQL_BLOCK_GCEA,      /* Graphics Command Engine Arbiter */ \
    AQL_BLOCK_RPB,       /* Render Backend */ \
    AQL_BLOCK_RMI,       /* Read/Write Memory Interface */ \
}

/* Number of supported counter blocks */
#define AQL_GFX12_NUM_SUPPORTED_BLOCKS            31

/*
 * GFX12 Counter Block Register Information
 */
typedef struct {
    aql_block_id_t block_id;
    uint32_t perf_sel_base;
    uint32_t counter_count;
    const char* block_name;
} aql_gfx12_counter_block_info_t;

static const aql_gfx12_counter_block_info_t aql_gfx12_counter_blocks[] = {
    { AQL_BLOCK_CB,     AQL_GFX12_PERF_SEL_BASE_CB,     4, "CB" },     /* Real: 0x3C01 */
    { AQL_BLOCK_CPC,    AQL_GFX12_PERF_SEL_BASE_CPC,    2, "CPC" },    /* Real: 0x3809 */
    { AQL_BLOCK_CPF,    AQL_GFX12_PERF_SEL_BASE_CPF,    2, "CPF" },    /* Real: 0x3807 */
    { AQL_BLOCK_CPG,    AQL_GFX12_PERF_SEL_BASE_CPG,    2, "CPG" },    /* Real: 0x3802 */
    { AQL_BLOCK_DB,     AQL_GFX12_PERF_SEL_BASE_DB,     4, "DB" },     /* Real: 0x3C40 */
    { AQL_BLOCK_GDS,    AQL_GFX12_PERF_SEL_BASE_GDS,    4, "GDS" },    /* Placeholder */
    { AQL_BLOCK_GRBM,   AQL_GFX12_PERF_SEL_BASE_GRBM,   2, "GRBM" },   /* Real: 0x3840 */
    { AQL_BLOCK_GRBM_SE, AQL_GFX12_PERF_SEL_BASE_GRBMSE, 4, "GRBMSE" }, /* Placeholder */
    { AQL_BLOCK_PA_SC,  AQL_GFX12_PERF_SEL_BASE_PA_SC,  8, "PA_SC" },  /* Real: 0x3940 */
    { AQL_BLOCK_PA_SU,  AQL_GFX12_PERF_SEL_BASE_PA_SU,  4, "PA_SU" },  /* Real: 0x3900 */
    { AQL_BLOCK_SPI,    AQL_GFX12_PERF_SEL_BASE_SPI,    6, "SPI" },    /* Placeholder */
    { AQL_BLOCK_SQ,     AQL_GFX12_PERF_SEL_BASE_SQ,     16, "SQ" },    /* Real: 0x39C0 */
    { AQL_BLOCK_SX,     AQL_GFX12_PERF_SEL_BASE_SX,     4, "SX" },     /* Placeholder */
    { AQL_BLOCK_TA,     AQL_GFX12_PERF_SEL_BASE_TA,     2, "TA" },     /* Real: 0x3AC0 */
    { AQL_BLOCK_TCP,    AQL_GFX12_PERF_SEL_BASE_TCP,    4, "TCP" },    /* Real: 0x3B40 */
    { AQL_BLOCK_TD,     AQL_GFX12_PERF_SEL_BASE_TD,     2, "TD" },     /* Real: 0x3B00 */
    { AQL_BLOCK_TCA,    AQL_GFX12_PERF_SEL_BASE_TCA,    4, "TCA" },    /* Placeholder */
    { AQL_BLOCK_GL2C,   AQL_GFX12_PERF_SEL_BASE_GL2C,   4, "GL2C" },   /* Real: 0x3B80 (replaces TCC) */
    { AQL_BLOCK_UNKNOWN, 0, 0, NULL } /* Terminator */
};

/*
 * GRBM Index Construction for GFX12
 * GFX12 supports Workgroup Processors (WGP) in addition to traditional addressing
 */
static inline uint32_t aql_gfx12_make_grbm_index(uint32_t se, uint32_t sa,
                                                 uint32_t wgp, uint32_t instance) {
    return ((se & 0x3) << 16) |        /* Shader Engine [17:16] */
           ((sa & 0x1) << 12) |        /* Shader Array [12] */
           ((wgp & 0x3) << 8) |        /* Workgroup Processor [9:8] */
           (instance & 0xFF);          /* Instance [7:0] */
}

/*
 * Cache Policy for GFX12 Memory Operations
 */
#define AQL_GFX12_CACHE_POLICY                    0x0  /* Default cache policy */

/*
 * Packet Size Constants
 */
#define AQL_GFX12_EVENT_WRITE_SIZE_DWORDS         2
#define AQL_GFX12_ACQUIRE_MEM_SIZE_DWORDS         8  /* RDNA3 uses 8-dword ACQUIRE_MEM */
#define AQL_GFX12_WAIT_REG_MEM_SIZE_DWORDS        7
#define AQL_GFX12_SET_REG_SIZE_DWORDS             3
#define AQL_GFX12_WRITE_DATA_MIN_SIZE_DWORDS      4
#define AQL_GFX12_INDIRECT_BUFFER_SIZE_DWORDS     4
#define AQL_GFX12_NOP_MIN_SIZE_DWORDS             1

#endif /* AQL_GFX12_DEFS_H */