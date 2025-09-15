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

#include <stdint.h>

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