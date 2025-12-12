/**
 * @file gfx12_creator.c
 * @brief GFX12-specific architecture creation implementation
 *
 * =============================================================================
 * REGISTER ADDRESS CONVENTION
 * =============================================================================
 *
 * All register addresses in this file are stored as ABSOLUTE UCONFIG ADDRESSES
 * suitable for direct use in PM4 SET_UCONFIG_REG and COPY_DATA packets.
 *
 * PM4 Packet Register Encoding:
 * - SET_UCONFIG_REG: Takes register offset from UCONFIG_SPACE_START (0xC000)
 *   The packet hardware subtracts 0xC000 from the address we provide.
 *
 * - COPY_DATA: Takes the FULL absolute UCONFIG address (no subtraction).
 *   Example: GL2C_PERFCOUNTER0_LO uses 0xd380 directly in COPY_DATA.
 *
 * BASE_IDX Handling (from AMD gc_12_0_0_offset.h):
 * -------------------------------------------------
 * AMD hardware registers have a BASE_IDX field that indicates which register
 * aperture they belong to:
 *
 *   BASE_IDX=0: Standard UCONFIG registers
 *     Formula: absolute_addr = 0xC000 + register_offset
 *     Example: SQ_PERFCOUNTER0_SELECT = 0x3860 (BASE_IDX=0)
 *              absolute_addr = 0xC000 + 0x3860 = 0xD860
 *
 *   BASE_IDX=1: Offset-adjusted UCONFIG registers (require -0x2000 correction)
 *     Formula: absolute_addr = 0xC000 + (register_offset - 0x2000)
 *     Example: GL2C_PERFCOUNTER0_SELECT = 0x3b80 (BASE_IDX=1)
 *              absolute_addr = 0xC000 + (0x3b80 - 0x2000) = 0xDB80
 *
 * Why the 0x2000 offset?
 * ----------------------
 * BASE_IDX=1 registers have their offsets defined in a separate address space
 * starting at 0x2000. To map them into the UCONFIG aperture (0xC000), we must
 * subtract 0x2000 from the hardware-defined offset before adding 0xC000.
 *
 * This is NOT a runtime adjustment - all register defines below already have
 * this conversion applied and store the final absolute UCONFIG address.
 *
 * Register Value Format:
 * ----------------------
 * All register addresses use HEXADECIMAL format for consistency and clarity.
 * Comments show the source register name from AMD headers.
 *
 * Source Attribution:
 * -------------------
 * Register definitions are derived from AMD AMDGPU driver headers:
 *   - gc_12_0_0_offset.h: Register base offsets and BASE_IDX values
 *   - gc_12_0_0_sh_mask.h: Register field bit masks (not used here)
 *
 * =============================================================================
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

/* =============================================================================
 * REGISTER SPACE CONSTANTS
 * =============================================================================
 * Source: pm4_packets.h and AMD hardware documentation
 */
#define UCONFIG_SPACE_START 0x0000C000
#define PERSISTENT_SPACE_START 0x00002C00
#define BASE_IDX1_OFFSET 0x00002000

/* =============================================================================
 * GLOBAL CONTROL REGISTERS
 * =============================================================================
 * Source: gc_12_0_0_offset.h (via Rust offset.rs)
 * All BASE_IDX=0 (standard UCONFIG addressing)
 */
#define mmGRBM_GFX_INDEX 0xC200 /* regGRBM_GFX_INDEX */
#define mmCP_PERFMON_CNTL 0xD808 /* regCP_PERFMON_CNTL */
#define mmCOMPUTE_PERFCOUNT_ENABLE 0x2E0B /* regCOMPUTE_PERFCOUNT_ENABLE */
#define mmSQ_PERFCOUNTER_CTRL 0xD9E0 /* regSQ_PERFCOUNTER_CTRL */
#define mmSQ_PERFCOUNTER_CTRL2 0xD9E2 /* regSQ_PERFCOUNTER_CTRL2 */

/* Event types - from pm4_packets.h */
#define VGT_EVENT_TYPE_CS_PARTIAL_FLUSH 0x07

/* =============================================================================
 * CPC (Command Processor Compute) REGISTERS
 * =============================================================================
 * Source: gc_12_0_0_offset.h
 * BASE_IDX: 1
 * Address calculation: 0xA000 + offset (absolute UCONFIG addresses)
 */
#define mmCPC_PERFCOUNTER0_SELECT 0xD809 /* 0xA000 + 0x3809 */
#define mmCPC_PERFCOUNTER0_LO 0xD006 /* 0xA000 + 0x3006 */
#define mmCPC_PERFCOUNTER0_HI 0xD007 /* 0xA000 + 0x3007 */
#define mmCPC_PERFCOUNTER1_SELECT 0xD803 /* 0xA000 + 0x3803 */
#define mmCPC_PERFCOUNTER1_LO 0xD004 /* 0xA000 + 0x3004 */
#define mmCPC_PERFCOUNTER1_HI 0xD005 /* 0xA000 + 0x3005 */

/* =============================================================================
 * SQ (Shader Sequencer) REGISTERS
 * =============================================================================
 * Source: gc_12_0_0_offset.h
 * BASE_IDX: 1
 * Address calculation: 0xA000 + offset (absolute UCONFIG addresses)
 */
#define mmSQ_PERFCOUNTER0_SELECT 0xD9C0 /* regSQ_PERFCOUNTER0_SELECT */
#define mmSQ_PERFCOUNTER0_LO 0xD1C0 /* regSQ_PERFCOUNTER0_LO */
#define mmSQ_PERFCOUNTER1_LO 0xD1C2 /* regSQ_PERFCOUNTER1_LO */
#define mmSQ_PERFCOUNTER2_SELECT 0xD9C2 /* regSQ_PERFCOUNTER2_SELECT */
#define mmSQ_PERFCOUNTER2_LO 0xD1C4 /* regSQ_PERFCOUNTER2_LO */
#define mmSQ_PERFCOUNTER3_LO 0xD1C6 /* regSQ_PERFCOUNTER3_LO */
#define mmSQ_PERFCOUNTER4_SELECT 0xD9C4 /* regSQ_PERFCOUNTER4_SELECT */
#define mmSQ_PERFCOUNTER4_LO 0xD1C8 /* regSQ_PERFCOUNTER4_LO */
#define mmSQ_PERFCOUNTER5_LO 0xD1CA /* regSQ_PERFCOUNTER5_LO */
#define mmSQ_PERFCOUNTER6_SELECT 0xD9C6 /* regSQ_PERFCOUNTER6_SELECT */
#define mmSQ_PERFCOUNTER6_LO 0xD1CC /* regSQ_PERFCOUNTER6_LO */
#define mmSQ_PERFCOUNTER7_LO 0xD1CE /* regSQ_PERFCOUNTER7_LO */
#define mmSQ_PERFCOUNTER8_SELECT 0xD9C8 /* regSQ_PERFCOUNTER8_SELECT */
#define mmSQ_PERFCOUNTER10_SELECT 0xD9CA /* regSQ_PERFCOUNTER10_SELECT */
#define mmSQ_PERFCOUNTER12_SELECT 0xD9CC /* regSQ_PERFCOUNTER12_SELECT */
#define mmSQ_PERFCOUNTER14_SELECT 0xD9CE /* regSQ_PERFCOUNTER14_SELECT */

/* =============================================================================
 * GRBM (Graphics Register Bus Manager) REGISTERS
 * =============================================================================
 * Source: gc_12_0_0_offset.h
 * BASE_IDX: 1
 * Address calculation: 0xA000 + offset (absolute UCONFIG addresses)
 */
#define mmGRBM_PERFCOUNTER0_SELECT 0xD840 /* 0xA000 + 0x3840 */
#define mmGRBM_PERFCOUNTER0_LO 0xD040 /* 0xA000 + 0x3040 */
#define mmGRBM_PERFCOUNTER0_HI 0xD041 /* 0xA000 + 0x3041 */
#define mmGRBM_PERFCOUNTER1_SELECT 0xD841 /* 0xA000 + 0x3841 */
#define mmGRBM_PERFCOUNTER1_LO 0xD043 /* 0xA000 + 0x3043 */
#define mmGRBM_PERFCOUNTER1_HI 0xD044 /* 0xA000 + 0x3044 */

/* =============================================================================
 * GL2C (Graphics L2 Cache Channel) REGISTERS
 * =============================================================================
 * Source: gc_12_0_0_offset.h
 * BASE_IDX: 1
 * Address calculation: 0xA000 + offset (absolute UCONFIG addresses)
 *
 * IMPORTANT: GL2C counters require:
 *   1. Per-instance configuration in START packet (16 instances)
 *   2. Split 32-bit LO+HI reads in READ packet (not combined 64-bit)
 *   3. Broadcast->instance toggle before each instance read
 *   4. Absolute UCONFIG addresses in COPY_DATA (use values below directly)
 */
#define mmGL2C_PERFCOUNTER0_SELECT 0xDB80 /* 0xA000 + 0x3B80 */
#define mmGL2C_PERFCOUNTER0_LO 0xD380 /* 0xA000 + 0x3380 */
#define mmGL2C_PERFCOUNTER0_HI 0xD381 /* 0xA000 + 0x3381 */
#define mmGL2C_PERFCOUNTER1_SELECT 0xDB82 /* 0xA000 + 0x3B82 */
#define mmGL2C_PERFCOUNTER1_LO 0xD382 /* 0xA000 + 0x3382 */
#define mmGL2C_PERFCOUNTER1_HI 0xD383 /* 0xA000 + 0x3383 */
#define mmGL2C_PERFCOUNTER2_SELECT 0xDB84 /* 0xA000 + 0x3B84 */
#define mmGL2C_PERFCOUNTER2_LO 0xD384 /* 0xA000 + 0x3384 */
#define mmGL2C_PERFCOUNTER2_HI 0xD385 /* 0xA000 + 0x3385 */
#define mmGL2C_PERFCOUNTER3_SELECT 0xDB86 /* 0xA000 + 0x3B86 */
#define mmGL2C_PERFCOUNTER3_LO 0xD386 /* 0xA000 + 0x3386 */
#define mmGL2C_PERFCOUNTER3_HI 0xD387 /* 0xA000 + 0x3387 */

/* =============================================================================
 * SPI (Shader Processor Input) REGISTERS
 * =============================================================================
 * Source: gc_12_0_0_offset.h
 * BASE_IDX: 1
 * Address calculation: 0xA000 + offset (absolute UCONFIG addresses)
 */
#define mmSPI_PERFCOUNTER0_SELECT 0xD980 /* 0xA000 + 0x3980 */
#define mmSPI_PERFCOUNTER0_LO 0xD181 /* 0xA000 + 0x3181 */
#define mmSPI_PERFCOUNTER0_HI 0xD180 /* 0xA000 + 0x3180 */
#define mmSPI_PERFCOUNTER1_SELECT 0xD981 /* 0xA000 + 0x3981 */
#define mmSPI_PERFCOUNTER1_LO 0xD183 /* 0xA000 + 0x3183 */
#define mmSPI_PERFCOUNTER1_HI 0xD182 /* 0xA000 + 0x3182 */
#define mmSPI_PERFCOUNTER2_SELECT 0xD982 /* 0xA000 + 0x3982 */
#define mmSPI_PERFCOUNTER2_LO 0xD185 /* 0xA000 + 0x3185 */
#define mmSPI_PERFCOUNTER2_HI 0xD184 /* 0xA000 + 0x3184 */
#define mmSPI_PERFCOUNTER3_SELECT 0xD983 /* 0xA000 + 0x3983 */
#define mmSPI_PERFCOUNTER3_LO 0xD187 /* 0xA000 + 0x3187 */
#define mmSPI_PERFCOUNTER3_HI 0xD186 /* 0xA000 + 0x3186 */
#define mmSPI_PERFCOUNTER4_SELECT 0xD984 /* 0xA000 + 0x3984 */
#define mmSPI_PERFCOUNTER4_LO 0xD189 /* 0xA000 + 0x3189 */
#define mmSPI_PERFCOUNTER4_HI 0xD188 /* 0xA000 + 0x3188 */
#define mmSPI_PERFCOUNTER5_SELECT 0xD985 /* 0xA000 + 0x3985 */
#define mmSPI_PERFCOUNTER5_LO 0xD18B /* 0xA000 + 0x318B */
#define mmSPI_PERFCOUNTER5_HI 0xD18A /* 0xA000 + 0x318A */

/* =============================================================================
 * TA (Texture Addresser) REGISTERS
 * =============================================================================
 * Source: gc_12_0_0_offset.h
 * BASE_IDX: 1
 * Address calculation: 0xA000 + offset (absolute UCONFIG addresses)
 */
#define mmTA_PERFCOUNTER0_SELECT 0xDAC0 /* 0xA000 + 0x3AC0 */
#define mmTA_PERFCOUNTER0_LO 0xD2C0 /* 0xA000 + 0x32C0 */
#define mmTA_PERFCOUNTER0_HI 0xD2C1 /* 0xA000 + 0x32C1 */
#define mmTA_PERFCOUNTER1_SELECT 0xDAC2 /* 0xA000 + 0x3AC2 */
#define mmTA_PERFCOUNTER1_LO 0xD2C2 /* 0xA000 + 0x32C2 */
#define mmTA_PERFCOUNTER1_HI 0xD2C3 /* 0xA000 + 0x32C3 */

/* =============================================================================
 * TCP (Texture Cache Processor) REGISTERS
 * =============================================================================
 * Source: gc_12_0_0_offset.h
 * BASE_IDX: 1
 * Address calculation: 0xA000 + offset (absolute UCONFIG addresses)
 */
#define mmTCP_PERFCOUNTER0_SELECT 0xDB40 /* 0xA000 + 0x3B40 */
#define mmTCP_PERFCOUNTER0_LO 0xD340 /* 0xA000 + 0x3340 */
#define mmTCP_PERFCOUNTER0_HI 0xD341 /* 0xA000 + 0x3341 */
#define mmTCP_PERFCOUNTER1_SELECT 0xDB42 /* 0xA000 + 0x3B42 */
#define mmTCP_PERFCOUNTER1_LO 0xD342 /* 0xA000 + 0x3342 */
#define mmTCP_PERFCOUNTER1_HI 0xD343 /* 0xA000 + 0x3343 */
#define mmTCP_PERFCOUNTER2_SELECT 0xDB44 /* 0xA000 + 0x3B44 */
#define mmTCP_PERFCOUNTER2_LO 0xD344 /* 0xA000 + 0x3344 */
#define mmTCP_PERFCOUNTER2_HI 0xD345 /* 0xA000 + 0x3345 */
#define mmTCP_PERFCOUNTER3_SELECT 0xDB45 /* 0xA000 + 0x3B45 */
#define mmTCP_PERFCOUNTER3_LO 0xD346 /* 0xA000 + 0x3346 */
#define mmTCP_PERFCOUNTER3_HI 0xD347 /* 0xA000 + 0x3347 */

/* =============================================================================
 * TD (Texture Data) REGISTERS
 * =============================================================================
 * Source: gc_12_0_0_offset.h
 * BASE_IDX: 1
 * Address calculation: 0xA000 + offset (absolute UCONFIG addresses)
 */
#define mmTD_PERFCOUNTER0_SELECT 0xDB00 /* 0xA000 + 0x3B00 */
#define mmTD_PERFCOUNTER0_LO 0xD300 /* 0xA000 + 0x3300 */
#define mmTD_PERFCOUNTER0_HI 0xD301 /* 0xA000 + 0x3301 */
#define mmTD_PERFCOUNTER1_SELECT 0xDB02 /* 0xA000 + 0x3B02 */
#define mmTD_PERFCOUNTER1_LO 0xD302 /* 0xA000 + 0x3302 */
#define mmTD_PERFCOUNTER1_HI 0xD303 /* 0xA000 + 0x3303 */

/* =============================================================================
 * TCC (Texture Cache Controller) REGISTERS
 * =============================================================================
 * WARNING: TCC does not exist in GFX12 — it was renamed to GL2C.
 * These registers are from GFX9 (gc_9_2_1_offset.h, BASE_IDX: 1).
 * Kept for potential multi-arch support; not used in GFX12 counter_registry.
 * Address calculation: 0xA000 + offset (absolute UCONFIG addresses)
 */
#define mmTCC_PERFCOUNTER0_SELECT 0xDB80 /* 0xA000 + 0x3B80 (same as GL2C in GFX12) */
#define mmTCC_PERFCOUNTER0_LO 0xD380 /* 0xA000 + 0x3380 */
#define mmTCC_PERFCOUNTER0_HI 0xD381 /* 0xA000 + 0x3381 */
#define mmTCC_PERFCOUNTER1_SELECT 0xDB82 /* 0xA000 + 0x3B82 */
#define mmTCC_PERFCOUNTER1_LO 0xD382 /* 0xA000 + 0x3382 */
#define mmTCC_PERFCOUNTER1_HI 0xD383 /* 0xA000 + 0x3383 */
#define mmTCC_PERFCOUNTER2_SELECT 0xDB84 /* 0xA000 + 0x3B84 */
#define mmTCC_PERFCOUNTER2_LO 0xD384 /* 0xA000 + 0x3384 */
#define mmTCC_PERFCOUNTER2_HI 0xD385 /* 0xA000 + 0x3385 */
#define mmTCC_PERFCOUNTER3_SELECT 0xDB86 /* 0xA000 + 0x3B86 */
#define mmTCC_PERFCOUNTER3_LO 0xD386 /* 0xA000 + 0x3386 */
#define mmTCC_PERFCOUNTER3_HI 0xD387 /* 0xA000 + 0x3387 */

/* =============================================================================
 * SX (Shader Export) REGISTERS
 * =============================================================================
 * Source: gc_12_0_0_offset.h
 * BASE_IDX: 1
 * Address calculation: 0xA000 + offset (absolute UCONFIG addresses)
 */
#define mmSX_PERFCOUNTER0_SELECT 0xDA80 /* 0xA000 + 0x3A80 */
#define mmSX_PERFCOUNTER0_LO 0xD280 /* 0xA000 + 0x3280 */
#define mmSX_PERFCOUNTER0_HI 0xD281 /* 0xA000 + 0x3281 */
#define mmSX_PERFCOUNTER1_SELECT 0xDA82 /* 0xA000 + 0x3A82 */
#define mmSX_PERFCOUNTER1_LO 0xD282 /* 0xA000 + 0x3282 */
#define mmSX_PERFCOUNTER1_HI 0xD283 /* 0xA000 + 0x3283 */
#define mmSX_PERFCOUNTER2_SELECT 0xDA84 /* 0xA000 + 0x3A84 */
#define mmSX_PERFCOUNTER2_LO 0xD284 /* 0xA000 + 0x3284 */
#define mmSX_PERFCOUNTER2_HI 0xD285 /* 0xA000 + 0x3285 */
#define mmSX_PERFCOUNTER3_SELECT 0xDA86 /* 0xA000 + 0x3A86 */
#define mmSX_PERFCOUNTER3_LO 0xD286 /* 0xA000 + 0x3286 */
#define mmSX_PERFCOUNTER3_HI 0xD287 /* 0xA000 + 0x3287 */

/* =============================================================================
 * DB (Depth Buffer) REGISTERS
 * =============================================================================
 * Source: gc_12_0_0_offset.h
 * BASE_IDX: 1
 * Address calculation: 0xA000 + offset (absolute UCONFIG addresses)
 */
#define mmDB_PERFCOUNTER0_SELECT 0xDC40 /* 0xA000 + 0x3C40 */
#define mmDB_PERFCOUNTER0_LO 0xD440 /* 0xA000 + 0x3440 */
#define mmDB_PERFCOUNTER0_HI 0xD441 /* 0xA000 + 0x3441 */
#define mmDB_PERFCOUNTER1_SELECT 0xDC42 /* 0xA000 + 0x3C42 */
#define mmDB_PERFCOUNTER1_LO 0xD442 /* 0xA000 + 0x3442 */
#define mmDB_PERFCOUNTER1_HI 0xD443 /* 0xA000 + 0x3443 */
#define mmDB_PERFCOUNTER2_SELECT 0xDC44 /* 0xA000 + 0x3C44 */
#define mmDB_PERFCOUNTER2_LO 0xD444 /* 0xA000 + 0x3444 */
#define mmDB_PERFCOUNTER2_HI 0xD445 /* 0xA000 + 0x3445 */
#define mmDB_PERFCOUNTER3_SELECT 0xDC46 /* 0xA000 + 0x3C46 */
#define mmDB_PERFCOUNTER3_LO 0xD446 /* 0xA000 + 0x3446 */
#define mmDB_PERFCOUNTER3_HI 0xD447 /* 0xA000 + 0x3447 */

/* =============================================================================
 * PA_SC (Primitive Assembly - Scan Converter) REGISTERS
 * =============================================================================
 * Source: gc_12_0_0_offset.h
 * BASE_IDX: 1
 * Address calculation: 0xA000 + offset (absolute UCONFIG addresses)
 */
#define mmPA_SC_PERFCOUNTER0_SELECT 0xD940 /* 0xA000 + 0x3940 */
#define mmPA_SC_PERFCOUNTER0_LO 0xD140 /* 0xA000 + 0x3140 */
#define mmPA_SC_PERFCOUNTER0_HI 0xD141 /* 0xA000 + 0x3141 */
#define mmPA_SC_PERFCOUNTER1_SELECT 0xD942 /* 0xA000 + 0x3942 */
#define mmPA_SC_PERFCOUNTER1_LO 0xD142 /* 0xA000 + 0x3142 */
#define mmPA_SC_PERFCOUNTER1_HI 0xD143 /* 0xA000 + 0x3143 */
#define mmPA_SC_PERFCOUNTER2_SELECT 0xD943 /* 0xA000 + 0x3943 */
#define mmPA_SC_PERFCOUNTER2_LO 0xD144 /* 0xA000 + 0x3144 */
#define mmPA_SC_PERFCOUNTER2_HI 0xD145 /* 0xA000 + 0x3145 */
#define mmPA_SC_PERFCOUNTER3_SELECT 0xD944 /* 0xA000 + 0x3944 */
#define mmPA_SC_PERFCOUNTER3_LO 0xD146 /* 0xA000 + 0x3146 */
#define mmPA_SC_PERFCOUNTER3_HI 0xD147 /* 0xA000 + 0x3147 */

/* =============================================================================
 * PA_SU (Primitive Assembly - Setup Unit) REGISTERS
 * =============================================================================
 * Source: gc_12_0_0_offset.h
 * BASE_IDX: 1
 * Address calculation: 0xA000 + offset (absolute UCONFIG addresses)
 */
#define mmPA_SU_PERFCOUNTER0_SELECT 0xD900 /* 0xA000 + 0x3900 */
#define mmPA_SU_PERFCOUNTER0_LO 0xD100 /* 0xA000 + 0x3100 */
#define mmPA_SU_PERFCOUNTER0_HI 0xD101 /* 0xA000 + 0x3101 */
#define mmPA_SU_PERFCOUNTER1_SELECT 0xD902 /* 0xA000 + 0x3902 */
#define mmPA_SU_PERFCOUNTER1_LO 0xD102 /* 0xA000 + 0x3102 */
#define mmPA_SU_PERFCOUNTER1_HI 0xD103 /* 0xA000 + 0x3103 */
#define mmPA_SU_PERFCOUNTER2_SELECT 0xD904 /* 0xA000 + 0x3904 */
#define mmPA_SU_PERFCOUNTER2_LO 0xD104 /* 0xA000 + 0x3104 */
#define mmPA_SU_PERFCOUNTER2_HI 0xD105 /* 0xA000 + 0x3105 */
#define mmPA_SU_PERFCOUNTER3_SELECT 0xD906 /* 0xA000 + 0x3906 */
#define mmPA_SU_PERFCOUNTER3_LO 0xD106 /* 0xA000 + 0x3106 */
#define mmPA_SU_PERFCOUNTER3_HI 0xD107 /* 0xA000 + 0x3107 */

/* =============================================================================
 * GDS (Global Data Store) REGISTERS
 * =============================================================================
 * WARNING: GDS does not exist in GFX12. These registers are from GFX9
 * (gc_9_2_1_offset.h, BASE_IDX: 1). Kept for potential multi-arch support.
 * Not used in GFX12 counter_registry.
 * Address calculation: 0xA000 + offset (absolute UCONFIG addresses)
 */
#define mmGDS_PERFCOUNTER0_SELECT 0xDA80 /* 0xA000 + 0x3A80 (GFX9) */
#define mmGDS_PERFCOUNTER0_LO 0xD210 /* 0xA000 + 0x3210 (GFX9) */
#define mmGDS_PERFCOUNTER0_HI 0xD211 /* 0xA000 + 0x3211 (GFX9) */
#define mmGDS_PERFCOUNTER1_SELECT 0xDA82 /* 0xA000 + 0x3A82 (GFX9) */
#define mmGDS_PERFCOUNTER1_LO 0xD212 /* 0xA000 + 0x3212 (GFX9) */
#define mmGDS_PERFCOUNTER1_HI 0xD213 /* 0xA000 + 0x3213 (GFX9) */
#define mmGDS_PERFCOUNTER2_SELECT 0xDA84 /* 0xA000 + 0x3A84 (GFX9) */
#define mmGDS_PERFCOUNTER2_LO 0xD214 /* 0xA000 + 0x3214 (GFX9) */
#define mmGDS_PERFCOUNTER2_HI 0xD215 /* 0xA000 + 0x3215 (GFX9) */
#define mmGDS_PERFCOUNTER3_SELECT 0xDA86 /* 0xA000 + 0x3A86 (GFX9) */
#define mmGDS_PERFCOUNTER3_LO 0xD216 /* 0xA000 + 0x3216 (GFX9) */
#define mmGDS_PERFCOUNTER3_HI 0xD217 /* 0xA000 + 0x3217 (GFX9) */

/* Block info constants - from Rust block_info.rs */
#define GFX12_CPC_COUNTER_BLOCK_NUM_COUNTERS 2
#define GFX12_CPC_COUNTER_BLOCK_MAX_EVENT 0x1F /* Placeholder - actual from enums */
#define GFX12_SQ_COUNTER_BLOCK_NUM_COUNTERS 8
#define GFX12_SQ_COUNTER_BLOCK_MAX_EVENT 0xFF /* Placeholder - actual from enums */
#define GFX12_GRBM_COUNTER_BLOCK_NUM_COUNTERS 2
#define GFX12_GRBM_COUNTER_BLOCK_MAX_EVENT 51
#define GFX12_GL2C_COUNTER_BLOCK_NUM_COUNTERS 4
#define GFX12_GL2C_COUNTER_BLOCK_MAX_EVENT 249
#define GFX12_GL2C_COUNTER_BLOCK_NUM_INSTANCES 16
#define GFX12_SPI_COUNTER_BLOCK_NUM_COUNTERS 6
#define GFX12_SPI_COUNTER_BLOCK_MAX_EVENT 318
#define GFX12_SPI_COUNTER_BLOCK_NUM_INSTANCES 1
#define GFX12_TA_COUNTER_BLOCK_NUM_COUNTERS 2
#define GFX12_TA_COUNTER_BLOCK_MAX_EVENT 254
#define GFX12_TA_COUNTER_BLOCK_NUM_INSTANCES 2
#define GFX12_TCP_COUNTER_BLOCK_NUM_COUNTERS 4
#define GFX12_TCP_COUNTER_BLOCK_MAX_EVENT 99
#define GFX12_TCP_COUNTER_BLOCK_NUM_INSTANCES 2
#define GFX12_TD_COUNTER_BLOCK_NUM_COUNTERS 2
#define GFX12_TD_COUNTER_BLOCK_MAX_EVENT 127
#define GFX12_TD_COUNTER_BLOCK_NUM_INSTANCES 2
#define GFX12_TCC_COUNTER_BLOCK_NUM_COUNTERS 4
#define GFX12_TCC_COUNTER_BLOCK_MAX_EVENT 255
#define GFX12_TCC_COUNTER_BLOCK_NUM_INSTANCES 16
#define GFX12_SX_COUNTER_BLOCK_NUM_COUNTERS 4
#define GFX12_SX_COUNTER_BLOCK_MAX_EVENT 189
#define GFX12_SX_COUNTER_BLOCK_NUM_INSTANCES 1
#define GFX12_DB_COUNTER_BLOCK_NUM_COUNTERS 4
#define GFX12_DB_COUNTER_BLOCK_MAX_EVENT 218
#define GFX12_DB_COUNTER_BLOCK_NUM_INSTANCES 1
#define GFX12_PA_SC_COUNTER_BLOCK_NUM_COUNTERS 4
#define GFX12_PA_SC_COUNTER_BLOCK_MAX_EVENT 171
#define GFX12_PA_SC_COUNTER_BLOCK_NUM_INSTANCES 1

#define GFX12_PA_SU_COUNTER_BLOCK_NUM_COUNTERS 4
#define GFX12_PA_SU_COUNTER_BLOCK_MAX_EVENT 171
#define GFX12_PA_SU_COUNTER_BLOCK_NUM_INSTANCES 1

#define GFX12_GDS_COUNTER_BLOCK_NUM_COUNTERS 4
#define GFX12_GDS_COUNTER_BLOCK_MAX_EVENT 121
#define GFX12_GDS_COUNTER_BLOCK_NUM_INSTANCES 1

/* Counter block attributes - from Rust enums */
#define GFX12_COUNTER_BLOCK_DFLT_ATTR 1
#define GFX12_COUNTER_BLOCK_SPM_GLOBAL_ATTR 0x1000

/* GFX12 Architecture parameters - from actual gfx1200 GPU topology
 * These values should match the physical GPU configuration.
 * For reference gfx1200 (AMD Radeon RX 9060 XT):
 *   - 32 Compute Units (CUs)
 *   - 2 Shader Engines (SEs)
 *   - 2 Shader Arrays per SE (SAs)
 *   - 16 Workgroup Processors (WGPs = CUs / 2)
 *   - 4 WGPs per SA (16 WGPs / 4 SAs)
 */
#define GFX12_NUM_XCC 1
#define GFX12_NUM_SE 2
#define GFX12_NUM_SA 2
#define GFX12_NUM_CU 32
#define GFX12_NUM_WGP_PER_SA 4

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
static block_info_t *create_gfx12_cpc_block(void)
{
	block_info_t *block = ALLOC(sizeof(block_info_t));
	if (!block)
		return NULL;

	/* Allocate counter register info array */
	counter_reg_info_t *counter_regs =
		ALLOC_ARRAY(counter_reg_info_t, GFX12_CPC_COUNTER_BLOCK_NUM_COUNTERS);
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
	block->dimensions[0] = (dimension_t){ .size = GFX12_NUM_XCC, .dim = HARDWARE_DIM_XCC };

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
static block_info_t *create_gfx12_sq_block(void)
{
	block_info_t *block = ALLOC(sizeof(block_info_t));
	if (!block)
		return NULL;

	/* Allocate counter register info array */
	counter_reg_info_t *counter_regs =
		ALLOC_ARRAY(counter_reg_info_t, GFX12_SQ_COUNTER_BLOCK_NUM_COUNTERS);
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

	/* Create dimensions for SQ block - SE/SA/WGP dependent block
     * SQ counters require 4 sub-instances per SA to match hardware layout.
     * With instance_index = (wgp << 2), iterating wgp=0,1,2,3 gives
     * instance_index = 0x00, 0x04, 0x08, 0x0c (matching aqlprofile).
     * Total instances: 4 SE × 2 SA × 4 sub-instances = 32 */
	block->dimension_count = 3;
	block->dimensions = ALLOC_ARRAY(dimension_t, block->dimension_count);
	if (!block->dimensions) {
		FREE(counter_regs);
		FREE(block);
		return NULL;
	}
	block->dimensions[0] = (dimension_t){ .size = GFX12_NUM_SE, .dim = HARDWARE_DIM_SE };
	block->dimensions[1] = (dimension_t){ .size = GFX12_NUM_SA, .dim = HARDWARE_DIM_SA };
	block->dimensions[2] =
		(dimension_t){ .size = 4, .dim = HARDWARE_DIM_WGP }; /* 4 sub-instances per SA */

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
static block_info_t *create_gfx12_grbm_block(void)
{
	block_info_t *block = ALLOC(sizeof(block_info_t));
	if (!block)
		return NULL;

	/* Allocate counter register info array */
	counter_reg_info_t *counter_regs =
		ALLOC_ARRAY(counter_reg_info_t, GFX12_GRBM_COUNTER_BLOCK_NUM_COUNTERS);
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
	block->dimensions[0] = (dimension_t){ .size = GFX12_NUM_XCC, .dim = HARDWARE_DIM_XCC };

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
static block_info_t *create_gfx12_gl2c_block(void)
{
	block_info_t *block = ALLOC(sizeof(block_info_t));
	if (!block)
		return NULL;

	/* Allocate counter register info array */
	counter_reg_info_t *counter_regs =
		ALLOC_ARRAY(counter_reg_info_t, GFX12_GL2C_COUNTER_BLOCK_NUM_COUNTERS);
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
	block->dimensions[0] = (dimension_t){ .size = GFX12_NUM_XCC, .dim = HARDWARE_DIM_XCC };

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
static block_info_t *create_gfx12_spi_block(void)
{
	block_info_t *block = ALLOC(sizeof(block_info_t));
	if (!block)
		return NULL;

	/* Allocate counter register info array */
	counter_reg_info_t *counter_regs =
		ALLOC_ARRAY(counter_reg_info_t, GFX12_SPI_COUNTER_BLOCK_NUM_COUNTERS);
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
	block->dimensions[0] = (dimension_t){ .size = GFX12_NUM_XCC, .dim = HARDWARE_DIM_XCC };
	block->dimensions[1] = (dimension_t){ .size = GFX12_NUM_SE, .dim = HARDWARE_DIM_SE };

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
static block_info_t *create_gfx12_ta_block(void)
{
	block_info_t *block = ALLOC(sizeof(block_info_t));
	if (!block)
		return NULL;

	/* Allocate counter register info array */
	counter_reg_info_t *counter_regs =
		ALLOC_ARRAY(counter_reg_info_t, GFX12_TA_COUNTER_BLOCK_NUM_COUNTERS);
	if (!counter_regs) {
		FREE(block);
		return NULL;
	}

	/* Initialize TA counter registers - based on aqlprofile reference */
	create_counter_reg_info(&counter_regs[0], mmTA_PERFCOUNTER0_SELECT, 0, mmTA_PERFCOUNTER0_LO,
				mmTA_PERFCOUNTER0_HI);
	create_counter_reg_info(&counter_regs[1], mmTA_PERFCOUNTER1_SELECT, 0, mmTA_PERFCOUNTER1_LO,
				mmTA_PERFCOUNTER1_HI);

	/* Allocate dimensions - TA is a WGP block with XCC+SE+SA+WGP structure (4 dimensions) */
	dimension_t *dimensions = ALLOC_ARRAY(dimension_t, 4);
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
static block_info_t *create_gfx12_tcp_block(void)
{
	block_info_t *block = ALLOC(sizeof(block_info_t));
	if (!block)
		return NULL;

	/* Allocate counter register info array */
	counter_reg_info_t *counter_regs =
		ALLOC_ARRAY(counter_reg_info_t, GFX12_TCP_COUNTER_BLOCK_NUM_COUNTERS);
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
	dimension_t *dimensions = ALLOC_ARRAY(dimension_t, 4);
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
static block_info_t *create_gfx12_td_block(void)
{
	block_info_t *block = ALLOC(sizeof(block_info_t));
	if (!block)
		return NULL;

	/* Allocate counter register info array */
	counter_reg_info_t *counter_regs =
		ALLOC_ARRAY(counter_reg_info_t, GFX12_TD_COUNTER_BLOCK_NUM_COUNTERS);
	if (!counter_regs) {
		FREE(block);
		return NULL;
	}

	/* Initialize TD counter registers - based on aqlprofile reference */
	create_counter_reg_info(&counter_regs[0], mmTD_PERFCOUNTER0_SELECT, 0, mmTD_PERFCOUNTER0_LO,
				mmTD_PERFCOUNTER0_HI);
	create_counter_reg_info(&counter_regs[1], mmTD_PERFCOUNTER1_SELECT, 0, mmTD_PERFCOUNTER1_LO,
				mmTD_PERFCOUNTER1_HI);

	/* Allocate dimensions - TD is a WGP block with XCC+SE+SA+WGP structure (4 dimensions) */
	dimension_t *dimensions = ALLOC_ARRAY(dimension_t, 4);
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
static block_info_t *create_gfx12_tcc_block(void)
{
	block_info_t *block = ALLOC(sizeof(block_info_t));
	if (!block)
		return NULL;

	/* Allocate counter register info array */
	counter_reg_info_t *counter_regs =
		ALLOC_ARRAY(counter_reg_info_t, GFX12_TCC_COUNTER_BLOCK_NUM_COUNTERS);
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
	block->dimensions[0] = (dimension_t){ .size = GFX12_NUM_XCC, .dim = HARDWARE_DIM_XCC };

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
static block_info_t *create_gfx12_sx_block(void)
{
	block_info_t *block = ALLOC(sizeof(block_info_t));
	if (!block)
		return NULL;

	/* Allocate counter register info array */
	counter_reg_info_t *counter_regs =
		ALLOC_ARRAY(counter_reg_info_t, GFX12_SX_COUNTER_BLOCK_NUM_COUNTERS);
	if (!counter_regs) {
		FREE(block);
		return NULL;
	}

	/* Initialize SX counter registers - based on aqlprofile reference */
	create_counter_reg_info(&counter_regs[0], mmSX_PERFCOUNTER0_SELECT, 0, mmSX_PERFCOUNTER0_LO,
				mmSX_PERFCOUNTER0_HI);
	create_counter_reg_info(&counter_regs[1], mmSX_PERFCOUNTER1_SELECT, 0, mmSX_PERFCOUNTER1_LO,
				mmSX_PERFCOUNTER1_HI);
	create_counter_reg_info(&counter_regs[2], mmSX_PERFCOUNTER2_SELECT, 0, mmSX_PERFCOUNTER2_LO,
				mmSX_PERFCOUNTER2_HI);
	create_counter_reg_info(&counter_regs[3], mmSX_PERFCOUNTER3_SELECT, 0, mmSX_PERFCOUNTER3_LO,
				mmSX_PERFCOUNTER3_HI);

	/* Create dimensions for SX block - SE block with SE dimensions */
	block->dimension_count = 2;
	block->dimensions = ALLOC_ARRAY(dimension_t, block->dimension_count);
	if (!block->dimensions) {
		FREE(counter_regs);
		FREE(block);
		return NULL;
	}
	block->dimensions[0] = (dimension_t){ .size = GFX12_NUM_XCC, .dim = HARDWARE_DIM_XCC };
	block->dimensions[1] = (dimension_t){ .size = GFX12_NUM_SE, .dim = HARDWARE_DIM_SE };

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
static block_info_t *create_gfx12_db_block(void)
{
	block_info_t *block = ALLOC(sizeof(block_info_t));
	if (!block)
		return NULL;

	/* Allocate counter register info array */
	counter_reg_info_t *counter_regs =
		ALLOC_ARRAY(counter_reg_info_t, GFX12_DB_COUNTER_BLOCK_NUM_COUNTERS);
	if (!counter_regs) {
		FREE(block);
		return NULL;
	}

	/* Initialize DB counter registers - based on aqlprofile reference */
	create_counter_reg_info(&counter_regs[0], mmDB_PERFCOUNTER0_SELECT, 0, mmDB_PERFCOUNTER0_LO,
				mmDB_PERFCOUNTER0_HI);
	create_counter_reg_info(&counter_regs[1], mmDB_PERFCOUNTER1_SELECT, 0, mmDB_PERFCOUNTER1_LO,
				mmDB_PERFCOUNTER1_HI);
	create_counter_reg_info(&counter_regs[2], mmDB_PERFCOUNTER2_SELECT, 0, mmDB_PERFCOUNTER2_LO,
				mmDB_PERFCOUNTER2_HI);
	create_counter_reg_info(&counter_regs[3], mmDB_PERFCOUNTER3_SELECT, 0, mmDB_PERFCOUNTER3_LO,
				mmDB_PERFCOUNTER3_HI);

	/* Create dimensions for DB block - SE block with SE dimensions */
	block->dimension_count = 2;
	block->dimensions = ALLOC_ARRAY(dimension_t, block->dimension_count);
	if (!block->dimensions) {
		FREE(counter_regs);
		FREE(block);
		return NULL;
	}
	block->dimensions[0] = (dimension_t){ .size = GFX12_NUM_XCC, .dim = HARDWARE_DIM_XCC };
	block->dimensions[1] = (dimension_t){ .size = GFX12_NUM_SE, .dim = HARDWARE_DIM_SE };

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
static block_info_t *create_gfx12_pa_sc_block(void)
{
	block_info_t *block = ALLOC(sizeof(block_info_t));
	if (!block)
		return NULL;

	/* Allocate counter register info array */
	counter_reg_info_t *counter_regs =
		ALLOC_ARRAY(counter_reg_info_t, GFX12_PA_SC_COUNTER_BLOCK_NUM_COUNTERS);
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
	block->dimensions[0] = (dimension_t){ .size = GFX12_NUM_XCC, .dim = HARDWARE_DIM_XCC };
	block->dimensions[1] = (dimension_t){ .size = GFX12_NUM_SE, .dim = HARDWARE_DIM_SE };

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
static block_info_t *create_gfx12_pa_su_block(void)
{
	block_info_t *block = ALLOC(sizeof(block_info_t));
	if (!block)
		return NULL;

	/* Allocate counter register info array */
	counter_reg_info_t *counter_regs =
		ALLOC_ARRAY(counter_reg_info_t, GFX12_PA_SU_COUNTER_BLOCK_NUM_COUNTERS);
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

	block->dimensions[0] = (dimension_t){ .size = GFX12_NUM_XCC, .dim = HARDWARE_DIM_XCC };
	block->dimensions[1] = (dimension_t){ .size = GFX12_NUM_SE, .dim = HARDWARE_DIM_SE };

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
static block_info_t *create_gfx12_gds_block(void)
{
	block_info_t *block = ALLOC(sizeof(block_info_t));
	if (!block)
		return NULL;

	/* Allocate counter register info array */
	counter_reg_info_t *counter_regs =
		ALLOC_ARRAY(counter_reg_info_t, GFX12_GDS_COUNTER_BLOCK_NUM_COUNTERS);
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

	block->dimensions[0] = (dimension_t){ .size = GFX12_NUM_XCC, .dim = HARDWARE_DIM_XCC };

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
arch_t *create_gfx12_arch(void)
{
	arch_t *arch = ALLOC(sizeof(arch_t));
	if (!arch)
		return NULL;

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
	block_info_t *cpc_block = create_gfx12_cpc_block();
	block_info_t *sq_block = create_gfx12_sq_block();
	block_info_t *grbm_block = create_gfx12_grbm_block();
	block_info_t *gl2c_block = create_gfx12_gl2c_block();
	block_info_t *spi_block = create_gfx12_spi_block();
	block_info_t *ta_block = create_gfx12_ta_block();
	block_info_t *tcp_block = create_gfx12_tcp_block();
	block_info_t *td_block = create_gfx12_td_block();
	block_info_t *tcc_block = create_gfx12_tcc_block();
	block_info_t *sx_block = create_gfx12_sx_block();
	block_info_t *db_block = create_gfx12_db_block();
	block_info_t *pa_sc_block = create_gfx12_pa_sc_block();
	block_info_t *pa_su_block = create_gfx12_pa_su_block();
	block_info_t *gds_block = create_gfx12_gds_block();

	if (!cpc_block || !sq_block || !grbm_block || !gl2c_block || !spi_block || !ta_block ||
	    !tcp_block || !td_block || !tcc_block || !sx_block || !db_block || !pa_sc_block ||
	    !pa_su_block || !gds_block) {
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
	/* Block count must be >= highest block ID + 1
     * Highest registered block is HW_IP_BLOCK_TCC=19, so we need at least 20
     * Using HW_IP_BLOCK_LAST allows for all possible blocks
     */
	arch->block_map.block_count = HW_IP_BLOCK_LAST;

	/* Initialize control registers for GFX12 */
	arch->control_regs = (arch_control_regs_t) {
        .grbm_gfx_index = mmGRBM_GFX_INDEX,                    /* 49664 */
        .cp_perfmon_cntl = mmCP_PERFMON_CNTL,                  /* 55304 */
        .compute_perfcount_enable = mmCOMPUTE_PERFCOUNT_ENABLE, /* 11787 */
        .sq_perfcounter_ctrl2 = mmSQ_PERFCOUNTER_CTRL2,        /* 55778 */
        .sq_perfcounter_ctrl = 0,
        .sq_perfcounter_mask = 0,
        .rlc_perfmon_clk_cntl = 0,
        .uconfig_space_start = UCONFIG_SPACE_START,            /* 0x0000C000 */
        .persistent_space_start = PERSISTENT_SPACE_START,       /* 0x00002C00 */
        .cs_partial_flush_event = VGT_EVENT_TYPE_CS_PARTIAL_FLUSH, /* 0x07 */
        .event_index_flush = 4,                                /* Event flush index */
        .gcr_cntl_default = 0x00018000,                       /* From Rust impl */
        .cp_coher_cntl_default = 0,
        .poll_interval_default = 0x10,                        /* Poll every 4K */
        .counter_control_bits = {
            .sq_ps_en_bit = 0,                                /* PS enable bit */
            .sq_gs_en_bit = 2,                                /* GS enable bit */
            .sq_hs_en_bit = 4,                                /* HS enable bit */
            .sq_cs_en_bit = 6,                                /* CS enable bit */
            .sq_vs_en_bit = 0,
            .sq_es_en_bit = 0,
            .sq_ls_en_bit = 0
        },
        .sq_select_masks = { 0 },
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