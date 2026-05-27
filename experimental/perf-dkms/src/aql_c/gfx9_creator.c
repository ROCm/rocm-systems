/**
 * @file gfx9_creator.c
 * @brief GFX9-specific architecture creation implementation
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
 *   Example: TCC_PERFCOUNTER0_LO uses 0xD380 directly in COPY_DATA.
 *
 * BASE_IDX Handling (from AMD gc_9_2_1_offset.h):
 * -------------------------------------------------
 * AMD hardware registers have a BASE_IDX field that indicates which register
 * aperture they belong to:
 *
 *   BASE_IDX=0: Standard UCONFIG registers
 *     Formula: absolute_addr = 0xC000 + register_offset
 *     Example: COMPUTE_PERFCOUNT_ENABLE = 0x0E0B (BASE_IDX=0)
 *              absolute_addr = 0xC000 + 0x0E0B = 0xCE0B
 *
 *   BASE_IDX=1: Offset-adjusted UCONFIG registers (require -0x2000 correction)
 *     Formula: absolute_addr = 0xC000 + (register_offset - 0x2000)
 *     Example: SQ_PERFCOUNTER0_SELECT = 0x39C0 (BASE_IDX=1)
 *              absolute_addr = 0xC000 + (0x39C0 - 0x2000) = 0xD9C0
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
 *   - gc_9_2_1_offset.h: Register base offsets and BASE_IDX values
 *   - gc_9_2_1_sh_mask.h: Register field bit masks (not used here)
 *
 * =============================================================================
 */

#include "aql_structures.h"
#include "arch_creator_common.h"
#include "gfx9_events.h"

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
 * Source: gc_9_2_1_offset.h
 * GRBM_GFX_INDEX: offset 0x2200, BASE_IDX=1 -> 0xC200
 * CP_PERFMON_CNTL: offset 0x3808, BASE_IDX=1 -> 0xD808
 * COMPUTE_PERFCOUNT_ENABLE: offset 0x0E0B, BASE_IDX=0 -> 0xCE0B
 * SQ_PERFCOUNTER_CTRL: offset 0x39E0, BASE_IDX=1 -> 0xD9E0
 * SQ_PERFCOUNTER_CTRL2: offset 0x39E2, BASE_IDX=1 -> 0xD9E2
 * SQ_PERFCOUNTER_MASK: offset 0x39E1, BASE_IDX=1 -> 0xD9E1
 * RLC_PERFMON_CLK_CNTL: offset 0x3CBF, BASE_IDX=1 -> 0xDCBF
 */
#define mmGRBM_GFX_INDEX 0xC200 /* regGRBM_GFX_INDEX */
#define mmCP_PERFMON_CNTL 0xD808 /* regCP_PERFMON_CNTL */
#define mmCOMPUTE_PERFCOUNT_ENABLE 0xCE0B /* regCOMPUTE_PERFCOUNT_ENABLE */
#define mmSQ_PERFCOUNTER_CTRL 0xD9E0 /* regSQ_PERFCOUNTER_CTRL */
#define mmSQ_PERFCOUNTER_CTRL2 0xD9E2 /* regSQ_PERFCOUNTER_CTRL2 */
#define mmSQ_PERFCOUNTER_MASK 0xD9E1 /* regSQ_PERFCOUNTER_MASK */
#define mmRLC_PERFMON_CLK_CNTL 0xDCBF /* regRLC_PERFMON_CLK_CNTL */

/* Event types - from pm4_packets.h */
#define VGT_EVENT_TYPE_CS_PARTIAL_FLUSH 0x07

/* =============================================================================
 * CPC (Command Processor Compute) REGISTERS
 * =============================================================================
 * Source: gc_9_2_1_offset.h
 * BASE_IDX: 1
 * Address calculation: 0xC000 + (offset - 0x2000)
 */
#define mmCPC_PERFCOUNTER0_SELECT 0xD809 /* offset 0x3809 */
#define mmCPC_PERFCOUNTER0_LO 0xD006 /* offset 0x3006 */
#define mmCPC_PERFCOUNTER0_HI 0xD007 /* offset 0x3007 */
#define mmCPC_PERFCOUNTER1_SELECT 0xD803 /* offset 0x3803 */
#define mmCPC_PERFCOUNTER1_LO 0xD004 /* offset 0x3004 */
#define mmCPC_PERFCOUNTER1_HI 0xD005 /* offset 0x3005 */

/* =============================================================================
 * CPF (Command Processor Fetcher) REGISTERS
 * =============================================================================
 * Source: gc_9_2_1_offset.h
 * BASE_IDX: 1
 * Address calculation: 0xC000 + (offset - 0x2000)
 */
#define mmCPF_PERFCOUNTER0_SELECT 0xD807 /* offset 0x3807 */
#define mmCPF_PERFCOUNTER0_LO 0xD00A /* offset 0x300A */
#define mmCPF_PERFCOUNTER0_HI 0xD00B /* offset 0x300B */
#define mmCPF_PERFCOUNTER1_SELECT 0xD805 /* offset 0x3805 */
#define mmCPF_PERFCOUNTER1_LO 0xD008 /* offset 0x3008 */
#define mmCPF_PERFCOUNTER1_HI 0xD009 /* offset 0x3009 */

/* =============================================================================
 * SQ (Shader Sequencer) REGISTERS
 * =============================================================================
 * Source: gc_9_2_1_offset.h
 * BASE_IDX: 1
 * Address calculation: 0xC000 + (offset - 0x2000)
 *
 * GFX9 SQ has 16 counters but we expose 8 usable ones.
 * SELECT registers: 0xD9C0 + n
 * LO registers: 0xF1C0 + (n * 2)
 * HI registers: 0xF1C1 + (n * 2)
 */
#define mmSQ_PERFCOUNTER0_SELECT 0xD9C0 /* offset 0x39C0 */
#define mmSQ_PERFCOUNTER0_LO 0xF1C0 /* offset 0x31C0 -> 0xC000 + 0x31C0 = 0xF1C0 */
#define mmSQ_PERFCOUNTER0_HI 0xF1C1 /* offset 0x31C1 */
#define mmSQ_PERFCOUNTER1_SELECT 0xD9C1 /* offset 0x39C1 */
#define mmSQ_PERFCOUNTER1_LO 0xF1C2 /* offset 0x31C2 */
#define mmSQ_PERFCOUNTER1_HI 0xF1C3 /* offset 0x31C3 */
#define mmSQ_PERFCOUNTER2_SELECT 0xD9C2 /* offset 0x39C2 */
#define mmSQ_PERFCOUNTER2_LO 0xF1C4 /* offset 0x31C4 */
#define mmSQ_PERFCOUNTER2_HI 0xF1C5 /* offset 0x31C5 */
#define mmSQ_PERFCOUNTER3_SELECT 0xD9C3 /* offset 0x39C3 */
#define mmSQ_PERFCOUNTER3_LO 0xF1C6 /* offset 0x31C6 */
#define mmSQ_PERFCOUNTER3_HI 0xF1C7 /* offset 0x31C7 */
#define mmSQ_PERFCOUNTER4_SELECT 0xD9C4 /* offset 0x39C4 */
#define mmSQ_PERFCOUNTER4_LO 0xF1C8 /* offset 0x31C8 */
#define mmSQ_PERFCOUNTER4_HI 0xF1C9 /* offset 0x31C9 */
#define mmSQ_PERFCOUNTER5_SELECT 0xD9C5 /* offset 0x39C5 */
#define mmSQ_PERFCOUNTER5_LO 0xF1CA /* offset 0x31CA */
#define mmSQ_PERFCOUNTER5_HI 0xF1CB /* offset 0x31CB */
#define mmSQ_PERFCOUNTER6_SELECT 0xD9C6 /* offset 0x39C6 */
#define mmSQ_PERFCOUNTER6_LO 0xF1CC /* offset 0x31CC */
#define mmSQ_PERFCOUNTER6_HI 0xF1CD /* offset 0x31CD */
#define mmSQ_PERFCOUNTER7_SELECT 0xD9C7 /* offset 0x39C7 */
#define mmSQ_PERFCOUNTER7_LO 0xF1CE /* offset 0x31CE */
#define mmSQ_PERFCOUNTER7_HI 0xF1CF /* offset 0x31CF */

/* =============================================================================
 * GRBM (Graphics Register Bus Manager) REGISTERS
 * =============================================================================
 * Source: gc_9_2_1_offset.h
 * BASE_IDX: 1
 * Address calculation: 0xC000 + (offset - 0x2000)
 */
#define mmGRBM_PERFCOUNTER0_SELECT 0xD840 /* offset 0x3840 */
#define mmGRBM_PERFCOUNTER0_LO 0xD040 /* offset 0x3040 */
#define mmGRBM_PERFCOUNTER0_HI 0xD041 /* offset 0x3041 */
#define mmGRBM_PERFCOUNTER1_SELECT 0xD841 /* offset 0x3841 */
#define mmGRBM_PERFCOUNTER1_LO 0xD043 /* offset 0x3043 */
#define mmGRBM_PERFCOUNTER1_HI 0xD044 /* offset 0x3044 */

/* =============================================================================
 * TCC (Texture Cache Controller / GL2C equivalent) REGISTERS
 * =============================================================================
 * Source: gc_9_2_1_offset.h
 * BASE_IDX: 1
 * Address calculation: 0xC000 + (offset - 0x2000)
 *
 * IMPORTANT: On GFX9, TCC is the L2 cache block (renamed to GL2C on GFX10+).
 * We register this block under BOTH HW_IP_BLOCK_TCC and HW_IP_BLOCK_GL2C
 * so that counter_collection_t lookups with block_id=HW_IP_BLOCK_GL2C work
 * transparently on GFX9.
 */
#define mmTCC_PERFCOUNTER0_SELECT 0xDB80 /* offset 0x3B80 */
#define mmTCC_PERFCOUNTER0_LO 0xD380 /* offset 0x3380 */
#define mmTCC_PERFCOUNTER0_HI 0xD381 /* offset 0x3381 */
#define mmTCC_PERFCOUNTER1_SELECT 0xDB82 /* offset 0x3B82 */
#define mmTCC_PERFCOUNTER1_LO 0xD382 /* offset 0x3382 */
#define mmTCC_PERFCOUNTER1_HI 0xD383 /* offset 0x3383 */
#define mmTCC_PERFCOUNTER2_SELECT 0xDB84 /* offset 0x3B84 */
#define mmTCC_PERFCOUNTER2_LO 0xD384 /* offset 0x3384 */
#define mmTCC_PERFCOUNTER2_HI 0xD385 /* offset 0x3385 */
#define mmTCC_PERFCOUNTER3_SELECT 0xDB85 /* offset 0x3B85 */
#define mmTCC_PERFCOUNTER3_LO 0xD386 /* offset 0x3386 */
#define mmTCC_PERFCOUNTER3_HI 0xD387 /* offset 0x3387 */

/* =============================================================================
 * SPI (Shader Processor Input) REGISTERS
 * =============================================================================
 * Source: gc_9_2_1_offset.h
 * BASE_IDX: 1
 * Address calculation: 0xC000 + (offset - 0x2000)
 *
 * NOTE: SPI has swapped LO/HI order (HI has lower address than LO).
 * This is correct per the hardware register definitions.
 */
#define mmSPI_PERFCOUNTER0_SELECT 0xD980 /* offset 0x3980 */
#define mmSPI_PERFCOUNTER0_LO 0xD181 /* offset 0x3181 */
#define mmSPI_PERFCOUNTER0_HI 0xD180 /* offset 0x3180 */
#define mmSPI_PERFCOUNTER1_SELECT 0xD981 /* offset 0x3981 */
#define mmSPI_PERFCOUNTER1_LO 0xD183 /* offset 0x3183 */
#define mmSPI_PERFCOUNTER1_HI 0xD182 /* offset 0x3182 */
#define mmSPI_PERFCOUNTER2_SELECT 0xD982 /* offset 0x3982 */
#define mmSPI_PERFCOUNTER2_LO 0xD185 /* offset 0x3185 */
#define mmSPI_PERFCOUNTER2_HI 0xD184 /* offset 0x3184 */
#define mmSPI_PERFCOUNTER3_SELECT 0xD983 /* offset 0x3983 */
#define mmSPI_PERFCOUNTER3_LO 0xD187 /* offset 0x3187 */
#define mmSPI_PERFCOUNTER3_HI 0xD186 /* offset 0x3186 */
#define mmSPI_PERFCOUNTER4_SELECT 0xD988 /* offset 0x3988 */
#define mmSPI_PERFCOUNTER4_LO 0xD189 /* offset 0x3189 */
#define mmSPI_PERFCOUNTER4_HI 0xD188 /* offset 0x3188 */
#define mmSPI_PERFCOUNTER5_SELECT 0xD989 /* offset 0x3989 */
#define mmSPI_PERFCOUNTER5_LO 0xD18B /* offset 0x318B */
#define mmSPI_PERFCOUNTER5_HI 0xD18A /* offset 0x318A */

/* =============================================================================
 * TA (Texture Addresser) REGISTERS
 * =============================================================================
 * Source: gc_9_2_1_offset.h
 * BASE_IDX: 1
 * Address calculation: 0xC000 + (offset - 0x2000)
 */
#define mmTA_PERFCOUNTER0_SELECT 0xDAC0 /* offset 0x3AC0 */
#define mmTA_PERFCOUNTER0_LO 0xD2C0 /* offset 0x32C0 */
#define mmTA_PERFCOUNTER0_HI 0xD2C1 /* offset 0x32C1 */
#define mmTA_PERFCOUNTER1_SELECT 0xDAC2 /* offset 0x3AC2 */
#define mmTA_PERFCOUNTER1_LO 0xD2C2 /* offset 0x32C2 */
#define mmTA_PERFCOUNTER1_HI 0xD2C3 /* offset 0x32C3 */

/* =============================================================================
 * TCP (Texture Cache Processor) REGISTERS
 * =============================================================================
 * Source: gc_9_2_1_offset.h
 * BASE_IDX: 1
 * Address calculation: 0xC000 + (offset - 0x2000)
 */
#define mmTCP_PERFCOUNTER0_SELECT 0xDB40 /* offset 0x3B40 */
#define mmTCP_PERFCOUNTER0_LO 0xD340 /* offset 0x3340 */
#define mmTCP_PERFCOUNTER0_HI 0xD341 /* offset 0x3341 */
#define mmTCP_PERFCOUNTER1_SELECT 0xDB42 /* offset 0x3B42 */
#define mmTCP_PERFCOUNTER1_LO 0xD342 /* offset 0x3342 */
#define mmTCP_PERFCOUNTER1_HI 0xD343 /* offset 0x3343 */
#define mmTCP_PERFCOUNTER2_SELECT 0xDB44 /* offset 0x3B44 */
#define mmTCP_PERFCOUNTER2_LO 0xD344 /* offset 0x3344 */
#define mmTCP_PERFCOUNTER2_HI 0xD345 /* offset 0x3345 */
#define mmTCP_PERFCOUNTER3_SELECT 0xDB45 /* offset 0x3B45 */
#define mmTCP_PERFCOUNTER3_LO 0xD346 /* offset 0x3346 */
#define mmTCP_PERFCOUNTER3_HI 0xD347 /* offset 0x3347 */

/* Block info constants - GFX9 (gfx906) specific values */
#define GFX9_CPC_COUNTER_BLOCK_NUM_COUNTERS 2
#define GFX9_CPC_COUNTER_BLOCK_MAX_EVENT 0x1F

#define GFX9_CPF_COUNTER_BLOCK_NUM_COUNTERS 2
#define GFX9_CPF_COUNTER_BLOCK_MAX_EVENT 0x1F

#define GFX9_SQ_COUNTER_BLOCK_NUM_COUNTERS 8
#define GFX9_SQ_COUNTER_BLOCK_MAX_EVENT 0xFF

#define GFX9_GRBM_COUNTER_BLOCK_NUM_COUNTERS 2
#define GFX9_GRBM_COUNTER_BLOCK_MAX_EVENT 51

#define GFX9_TCC_COUNTER_BLOCK_NUM_COUNTERS 4
#define GFX9_TCC_COUNTER_BLOCK_MAX_EVENT 255
#define GFX9_TCC_COUNTER_BLOCK_NUM_INSTANCES 16

#define GFX9_SPI_COUNTER_BLOCK_NUM_COUNTERS 6
#define GFX9_SPI_COUNTER_BLOCK_MAX_EVENT 318
#define GFX9_SPI_COUNTER_BLOCK_NUM_INSTANCES 1

#define GFX9_TA_COUNTER_BLOCK_NUM_COUNTERS 2
#define GFX9_TA_COUNTER_BLOCK_MAX_EVENT 254
#define GFX9_TA_COUNTER_BLOCK_NUM_INSTANCES 16

#define GFX9_TCP_COUNTER_BLOCK_NUM_COUNTERS 4
#define GFX9_TCP_COUNTER_BLOCK_MAX_EVENT 99
#define GFX9_TCP_COUNTER_BLOCK_NUM_INSTANCES 16

/* Counter block attributes */
#define GFX9_COUNTER_BLOCK_DFLT_ATTR 1
#define GFX9_COUNTER_BLOCK_SPM_GLOBAL_ATTR 0x1000

/* GFX9 Architecture parameters - gfx906 (AMD Radeon Instinct MI50/MI60)
 *   - 60 Compute Units (CUs)
 *   - 4 Shader Engines (SEs)
 *   - 1 Shader Array (SH) per SE
 *   - No WGP concept on GFX9 (CU-based architecture)
 */
#define GFX9_NUM_XCC 1
#define GFX9_NUM_SE 4
#define GFX9_NUM_SA 1
#define GFX9_NUM_CU 60
#define GFX9_NUM_WGP_PER_SA 0

/**
 * @brief Create CPC (Command Processor Compute) block information for GFX9
 *
 * Initializes the block_info_t structure for the CPC hardware block, which
 * contains performance counters for command processor operations including
 * command buffer processing, packet dispatch, and compute engine activity.
 * CPC is a global block with no SE/SA dependencies.
 *
 * Configuration:
 * - 2 performance counters
 * - Global scope (no dimensions)
 * - SELECT registers for event configuration
 * - 64-bit counter result registers (LO + HI)
 *
 * @return Pointer to initialized CPC block_info_t, or NULL on allocation failure
 *
 * @note CPC counters are global and do not require iteration over topology
 */
static block_info_t *create_gfx9_cpc_block(void)
{
	block_info_t *block = ALLOC(sizeof(block_info_t));
	if (!block)
		return NULL;

	/* Allocate counter register info array */
	counter_reg_info_t *counter_regs =
		ALLOC_ARRAY(counter_reg_info_t, GFX9_CPC_COUNTER_BLOCK_NUM_COUNTERS);
	if (!counter_regs) {
		FREE(block);
		return NULL;
	}

	/* Initialize CPC counter registers */
	create_counter_reg_info(&counter_regs[0], mmCPC_PERFCOUNTER0_SELECT, 0,
				mmCPC_PERFCOUNTER0_LO, mmCPC_PERFCOUNTER0_HI);
	create_counter_reg_info(&counter_regs[1], mmCPC_PERFCOUNTER1_SELECT, 0,
				mmCPC_PERFCOUNTER1_LO, mmCPC_PERFCOUNTER1_HI);

	/* CPC is a global block with no dimensions */
	block->dimension_count = 0;
	block->dimensions = NULL;

	block->name = "CPC";
	block->id = HW_IP_BLOCK_CPC;
	block->instance_count = 1;
	block->event_id_max = GFX9_CPC_COUNTER_BLOCK_MAX_EVENT;
	block->counter_count = GFX9_CPC_COUNTER_BLOCK_NUM_COUNTERS;
	block->counter_reg_info = counter_regs;
	block->attr = GFX9_COUNTER_BLOCK_DFLT_ATTR | GFX9_COUNTER_BLOCK_SPM_GLOBAL_ATTR;
	block->delay_info = NULL;
	block->spm_block_id = 0;

	return block;
}

/**
 * @brief Create CPF (Command Processor Fetcher) block information for GFX9
 *
 * Initializes the block_info_t structure for the CPF hardware block, which
 * contains performance counters for command processor fetch operations
 * including instruction fetching and queue management.
 * CPF is a global block with no SE/SA dependencies.
 *
 * Configuration:
 * - 2 performance counters
 * - Global scope (no dimensions)
 * - SELECT registers for event configuration
 * - 64-bit counter result registers (LO + HI)
 *
 * @return Pointer to initialized CPF block_info_t, or NULL on allocation failure
 *
 * @note CPF counters are global and do not require iteration over topology
 */
static block_info_t *create_gfx9_cpf_block(void)
{
	block_info_t *block = ALLOC(sizeof(block_info_t));
	if (!block)
		return NULL;

	/* Allocate counter register info array */
	counter_reg_info_t *counter_regs =
		ALLOC_ARRAY(counter_reg_info_t, GFX9_CPF_COUNTER_BLOCK_NUM_COUNTERS);
	if (!counter_regs) {
		FREE(block);
		return NULL;
	}

	/* Initialize CPF counter registers */
	create_counter_reg_info(&counter_regs[0], mmCPF_PERFCOUNTER0_SELECT, 0,
				mmCPF_PERFCOUNTER0_LO, mmCPF_PERFCOUNTER0_HI);
	create_counter_reg_info(&counter_regs[1], mmCPF_PERFCOUNTER1_SELECT, 0,
				mmCPF_PERFCOUNTER1_LO, mmCPF_PERFCOUNTER1_HI);

	/* CPF is a global block with no dimensions */
	block->dimension_count = 0;
	block->dimensions = NULL;

	block->name = "CPF";
	block->id = HW_IP_BLOCK_CPF;
	block->instance_count = 1;
	block->event_id_max = GFX9_CPF_COUNTER_BLOCK_MAX_EVENT;
	block->counter_count = GFX9_CPF_COUNTER_BLOCK_NUM_COUNTERS;
	block->counter_reg_info = counter_regs;
	block->attr = GFX9_COUNTER_BLOCK_DFLT_ATTR | GFX9_COUNTER_BLOCK_SPM_GLOBAL_ATTR;
	block->delay_info = NULL;
	block->spm_block_id = 0;

	return block;
}

/**
 * @brief Create SQ (Shader Sequencer) block information for GFX9
 *
 * Initializes the block_info_t structure for the SQ hardware block, which
 * contains performance counters for shader execution including wave counts,
 * instruction counts, and wait states. SQ counters are per-SE/SA and require
 * iteration over the GPU's SE x SA topology when reading.
 *
 * Configuration:
 * - 8 performance counters
 * - SE x SA dimensions (4 SE x 1 SA on gfx906)
 * - SELECT registers for event configuration
 * - CTRL register shared across all SQ counters (SQ_PERFCOUNTER_CTRL)
 * - 64-bit counter result registers (LO + HI)
 *
 * @return Pointer to initialized SQ block_info_t, or NULL on allocation failure
 *
 * @note GFX9 SQ has 16 hardware counters but we expose 8 usable ones
 */
static block_info_t *create_gfx9_sq_block(void)
{
	block_info_t *block = ALLOC(sizeof(block_info_t));
	if (!block)
		return NULL;

	/* Allocate counter register info array */
	counter_reg_info_t *counter_regs =
		ALLOC_ARRAY(counter_reg_info_t, GFX9_SQ_COUNTER_BLOCK_NUM_COUNTERS);
	if (!counter_regs) {
		FREE(block);
		return NULL;
	}

	/* Initialize SQ counter registers
	 * Each counter: SELECT=0xD9C0+n, CTRL=0xD9E0 (shared),
	 *               LO=0xF1C0+(n*2), HI=0xF1C1+(n*2)
	 */
	create_counter_reg_info(&counter_regs[0], mmSQ_PERFCOUNTER0_SELECT, mmSQ_PERFCOUNTER_CTRL,
				mmSQ_PERFCOUNTER0_LO, mmSQ_PERFCOUNTER0_HI);
	create_counter_reg_info(&counter_regs[1], mmSQ_PERFCOUNTER1_SELECT, mmSQ_PERFCOUNTER_CTRL,
				mmSQ_PERFCOUNTER1_LO, mmSQ_PERFCOUNTER1_HI);
	create_counter_reg_info(&counter_regs[2], mmSQ_PERFCOUNTER2_SELECT, mmSQ_PERFCOUNTER_CTRL,
				mmSQ_PERFCOUNTER2_LO, mmSQ_PERFCOUNTER2_HI);
	create_counter_reg_info(&counter_regs[3], mmSQ_PERFCOUNTER3_SELECT, mmSQ_PERFCOUNTER_CTRL,
				mmSQ_PERFCOUNTER3_LO, mmSQ_PERFCOUNTER3_HI);
	create_counter_reg_info(&counter_regs[4], mmSQ_PERFCOUNTER4_SELECT, mmSQ_PERFCOUNTER_CTRL,
				mmSQ_PERFCOUNTER4_LO, mmSQ_PERFCOUNTER4_HI);
	create_counter_reg_info(&counter_regs[5], mmSQ_PERFCOUNTER5_SELECT, mmSQ_PERFCOUNTER_CTRL,
				mmSQ_PERFCOUNTER5_LO, mmSQ_PERFCOUNTER5_HI);
	create_counter_reg_info(&counter_regs[6], mmSQ_PERFCOUNTER6_SELECT, mmSQ_PERFCOUNTER_CTRL,
				mmSQ_PERFCOUNTER6_LO, mmSQ_PERFCOUNTER6_HI);
	create_counter_reg_info(&counter_regs[7], mmSQ_PERFCOUNTER7_SELECT, mmSQ_PERFCOUNTER_CTRL,
				mmSQ_PERFCOUNTER7_LO, mmSQ_PERFCOUNTER7_HI);

	/* Create dimensions for SQ block - SE/SA dependent block (no WGP on GFX9) */
	block->dimension_count = 2;
	block->dimensions = ALLOC_ARRAY(dimension_t, block->dimension_count);
	if (!block->dimensions) {
		FREE(counter_regs);
		FREE(block);
		return NULL;
	}
	block->dimensions[0] = (dimension_t){ .size = GFX9_NUM_SE, .dim = HARDWARE_DIM_SE };
	block->dimensions[1] = (dimension_t){ .size = GFX9_NUM_SA, .dim = HARDWARE_DIM_SA };

	block->name = "SQ";
	block->id = HW_IP_BLOCK_SQ;
	block->instance_count = 1;
	block->event_id_max = GFX9_SQ_COUNTER_BLOCK_MAX_EVENT;
	block->counter_count = GFX9_SQ_COUNTER_BLOCK_NUM_COUNTERS;
	block->counter_reg_info = counter_regs;
	block->attr = GFX9_COUNTER_BLOCK_DFLT_ATTR;
	block->delay_info = NULL;
	block->spm_block_id = 9; /* SPM_SE_BLOCK_NAME_SQG */

	return block;
}

/**
 * @brief Create GRBM (Graphics Register Bus Manager) block information for GFX9
 *
 * Initializes the block_info_t structure for the GRBM hardware block, which
 * provides performance counters for GPU-wide activity including GUI pipeline
 * utilization, command buffer processing, and overall GPU busy state.
 * GRBM is a global block monitoring system-level GPU activities.
 *
 * Configuration:
 * - 2 performance counters
 * - Global scope (no dimensions)
 * - SELECT registers for event configuration
 * - 64-bit counter result registers (LO + HI)
 *
 * @return Pointer to initialized GRBM block_info_t, or NULL on allocation failure
 *
 * @note GRBM counters monitor system-level GPU activity
 */
static block_info_t *create_gfx9_grbm_block(void)
{
	block_info_t *block = ALLOC(sizeof(block_info_t));
	if (!block)
		return NULL;

	/* Allocate counter register info array */
	counter_reg_info_t *counter_regs =
		ALLOC_ARRAY(counter_reg_info_t, GFX9_GRBM_COUNTER_BLOCK_NUM_COUNTERS);
	if (!counter_regs) {
		FREE(block);
		return NULL;
	}

	/* Initialize GRBM counter registers */
	create_counter_reg_info(&counter_regs[0], mmGRBM_PERFCOUNTER0_SELECT, 0,
				mmGRBM_PERFCOUNTER0_LO, mmGRBM_PERFCOUNTER0_HI);
	create_counter_reg_info(&counter_regs[1], mmGRBM_PERFCOUNTER1_SELECT, 0,
				mmGRBM_PERFCOUNTER1_LO, mmGRBM_PERFCOUNTER1_HI);

	/* GRBM is a global block with no dimensions */
	block->dimension_count = 0;
	block->dimensions = NULL;

	block->name = "GRBM";
	block->id = HW_IP_BLOCK_GRBM;
	block->instance_count = 1;
	block->event_id_max = GFX9_GRBM_COUNTER_BLOCK_MAX_EVENT;
	block->counter_count = GFX9_GRBM_COUNTER_BLOCK_NUM_COUNTERS;
	block->counter_reg_info = counter_regs;
	block->attr = GFX9_COUNTER_BLOCK_DFLT_ATTR;
	block->delay_info = NULL;
	block->spm_block_id = 0;

	return block;
}

/**
 * @brief Create TCC block information for GFX9 with specified block ID
 *
 * Helper function that creates a TCC block with either HW_IP_BLOCK_TCC or
 * HW_IP_BLOCK_GL2C as the block ID. This allows registering the same physical
 * TCC hardware under both names so that GL2C counter lookups from userspace
 * work transparently on GFX9.
 *
 * Configuration:
 * - 4 performance counters per instance
 * - 16 instances across the GPU
 * - Global scope (no SE/SA dimensions, just instances)
 * - SELECT registers for event configuration
 * - 64-bit counter result registers (LO + HI)
 *
 * @param block_id Either HW_IP_BLOCK_TCC or HW_IP_BLOCK_GL2C
 * @param name Block name string ("TCC" or "GL2C")
 * @return Pointer to initialized block_info_t, or NULL on allocation failure
 */
static block_info_t *create_gfx9_tcc_block_impl(hardware_ip_block_t block_id, const char *name)
{
	block_info_t *block = ALLOC(sizeof(block_info_t));
	if (!block)
		return NULL;

	/* Allocate counter register info array */
	counter_reg_info_t *counter_regs =
		ALLOC_ARRAY(counter_reg_info_t, GFX9_TCC_COUNTER_BLOCK_NUM_COUNTERS);
	if (!counter_regs) {
		FREE(block);
		return NULL;
	}

	/* Initialize TCC counter registers - same registers for both TCC and GL2C aliases */
	create_counter_reg_info(&counter_regs[0], mmTCC_PERFCOUNTER0_SELECT, 0,
				mmTCC_PERFCOUNTER0_LO, mmTCC_PERFCOUNTER0_HI);
	create_counter_reg_info(&counter_regs[1], mmTCC_PERFCOUNTER1_SELECT, 0,
				mmTCC_PERFCOUNTER1_LO, mmTCC_PERFCOUNTER1_HI);
	create_counter_reg_info(&counter_regs[2], mmTCC_PERFCOUNTER2_SELECT, 0,
				mmTCC_PERFCOUNTER2_LO, mmTCC_PERFCOUNTER2_HI);
	create_counter_reg_info(&counter_regs[3], mmTCC_PERFCOUNTER3_SELECT, 0,
				mmTCC_PERFCOUNTER3_LO, mmTCC_PERFCOUNTER3_HI);

	/* TCC is a global block with no SE/SA dimensions (just instances) */
	block->dimension_count = 0;
	block->dimensions = NULL;

	block->name = name;
	block->id = block_id;
	block->instance_count = GFX9_TCC_COUNTER_BLOCK_NUM_INSTANCES;
	block->event_id_max = GFX9_TCC_COUNTER_BLOCK_MAX_EVENT;
	block->counter_count = GFX9_TCC_COUNTER_BLOCK_NUM_COUNTERS;
	block->counter_reg_info = counter_regs;
	block->attr = GFX9_COUNTER_BLOCK_DFLT_ATTR;
	block->delay_info = NULL;
	block->spm_block_id = 0;

	return block;
}

/**
 * @brief Create TCC (Texture Cache Controller) block information for GFX9
 *
 * Creates TCC block registered as HW_IP_BLOCK_TCC.
 *
 * @return Pointer to initialized TCC block_info_t, or NULL on allocation failure
 * @see create_gfx9_gl2c_block()
 */
static block_info_t *create_gfx9_tcc_block(void)
{
	return create_gfx9_tcc_block_impl(HW_IP_BLOCK_TCC, "TCC");
}

/**
 * @brief Create GL2C alias block for GFX9 (points to TCC hardware)
 *
 * Creates a second block entry for the TCC hardware, registered as
 * HW_IP_BLOCK_GL2C. This allows counter_collection_t with
 * block_id=HW_IP_BLOCK_GL2C to work transparently on GFX9.
 *
 * @return Pointer to initialized GL2C block_info_t, or NULL on allocation failure
 * @see create_gfx9_tcc_block()
 */
static block_info_t *create_gfx9_gl2c_block(void)
{
	return create_gfx9_tcc_block_impl(HW_IP_BLOCK_GL2C, "GL2C");
}

/**
 * @brief Create SPI (Shader Processor Input) block information for GFX9
 *
 * Initializes the block_info_t structure for the SPI hardware block, which
 * provides performance counters for shader input operations including wave
 * allocation, resource allocation, and shader launch activity. SPI is
 * per-SE (Shader Engine) and monitors shader dispatch behavior.
 *
 * Configuration:
 * - 6 performance counters
 * - SE x SA dimensions (4 SEs x 1 SA on gfx906)
 * - SELECT registers for event configuration
 * - 64-bit counter result registers (LO + HI)
 * - NOTE: SPI has swapped LO/HI order (HI has lower address than LO)
 *
 * @return Pointer to initialized SPI block_info_t, or NULL on allocation failure
 *
 * @note SPI counters must be read separately for each SE
 */
static block_info_t *create_gfx9_spi_block(void)
{
	block_info_t *block = ALLOC(sizeof(block_info_t));
	if (!block)
		return NULL;

	/* Allocate counter register info array */
	counter_reg_info_t *counter_regs =
		ALLOC_ARRAY(counter_reg_info_t, GFX9_SPI_COUNTER_BLOCK_NUM_COUNTERS);
	if (!counter_regs) {
		FREE(block);
		return NULL;
	}

	/* Initialize SPI counter registers
	 * NOTE: SPI has swapped LO/HI order - HI address is lower than LO.
	 * This is correct per GFX9 hardware register definitions.
	 */
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

	/* Create dimensions for SPI block - SE/SA dependent */
	block->dimension_count = 2;
	block->dimensions = ALLOC_ARRAY(dimension_t, block->dimension_count);
	if (!block->dimensions) {
		FREE(counter_regs);
		FREE(block);
		return NULL;
	}
	block->dimensions[0] = (dimension_t){ .size = GFX9_NUM_SE, .dim = HARDWARE_DIM_SE };
	block->dimensions[1] = (dimension_t){ .size = GFX9_NUM_SA, .dim = HARDWARE_DIM_SA };

	block->name = "SPI";
	block->id = HW_IP_BLOCK_SPI;
	block->instance_count = GFX9_SPI_COUNTER_BLOCK_NUM_INSTANCES;
	block->event_id_max = GFX9_SPI_COUNTER_BLOCK_MAX_EVENT;
	block->counter_count = GFX9_SPI_COUNTER_BLOCK_NUM_COUNTERS;
	block->counter_reg_info = counter_regs;
	block->attr = GFX9_COUNTER_BLOCK_DFLT_ATTR;
	block->delay_info = NULL;
	block->spm_block_id = 0;

	return block;
}

/**
 * @brief Create TA (Texture Addresser) block information for GFX9
 *
 * Initializes the block_info_t structure for the TA hardware block, which
 * provides performance counters for texture address calculation including
 * texture load/store operations, address generation, and filtering operations.
 * TA is an SE-level block with multiple instances (1 per CU).
 *
 * Configuration:
 * - 2 performance counters per instance
 * - 16 instances per SA (1 per CU)
 * - SE x SA dimensions (4 SE x 1 SA on gfx906)
 * - SELECT registers for event configuration
 * - 64-bit counter result registers (LO + HI)
 *
 * @return Pointer to initialized TA block_info_t, or NULL on allocation failure
 *
 * @note TA counters require iteration over SE/SA topology
 */
static block_info_t *create_gfx9_ta_block(void)
{
	block_info_t *block = ALLOC(sizeof(block_info_t));
	if (!block)
		return NULL;

	/* Allocate counter register info array */
	counter_reg_info_t *counter_regs =
		ALLOC_ARRAY(counter_reg_info_t, GFX9_TA_COUNTER_BLOCK_NUM_COUNTERS);
	if (!counter_regs) {
		FREE(block);
		return NULL;
	}

	/* Initialize TA counter registers */
	create_counter_reg_info(&counter_regs[0], mmTA_PERFCOUNTER0_SELECT, 0, mmTA_PERFCOUNTER0_LO,
				mmTA_PERFCOUNTER0_HI);
	create_counter_reg_info(&counter_regs[1], mmTA_PERFCOUNTER1_SELECT, 0, mmTA_PERFCOUNTER1_LO,
				mmTA_PERFCOUNTER1_HI);

	/* Allocate dimensions - TA is an SE-level block with SE+SA structure (2 dimensions) */
	dimension_t *dimensions = ALLOC_ARRAY(dimension_t, 2);
	if (!dimensions) {
		FREE(counter_regs);
		FREE(block);
		return NULL;
	}

	/* Initialize SE/SA dimensions - no WGP on GFX9 */
	dimensions[0].dim = HARDWARE_DIM_SE;
	dimensions[0].size = GFX9_NUM_SE;
	dimensions[1].dim = HARDWARE_DIM_SA;
	dimensions[1].size = GFX9_NUM_SA;

	/* Initialize TA block */
	block->name = "TA";
	block->id = HW_IP_BLOCK_TA;
	block->instance_count = GFX9_TA_COUNTER_BLOCK_NUM_INSTANCES; /* 16 instances */
	block->event_id_max = GFX9_TA_COUNTER_BLOCK_MAX_EVENT; /* 254 */
	block->counter_count = GFX9_TA_COUNTER_BLOCK_NUM_COUNTERS; /* 2 counters */
	block->dimensions = dimensions;
	block->dimension_count = 2; /* SE, SA */
	block->counter_reg_info = counter_regs;
	block->attr = GFX9_COUNTER_BLOCK_DFLT_ATTR;
	block->delay_info = NULL;
	block->spm_block_id = 0;

	return block;
}

/**
 * @brief Create TCP (Texture Cache Processor) block information for GFX9
 *
 * Initializes the block_info_t structure for the TCP hardware block, which
 * provides performance counters for texture cache operations including
 * texture fetch requests, cache hits/misses, and data transfer rates.
 * TCP is an SE-level block responsible for L1 texture cache with multiple
 * instances (1 per CU).
 *
 * Configuration:
 * - 4 performance counters per instance
 * - 16 instances per SA (1 per CU)
 * - SE x SA dimensions (4 SE x 1 SA on gfx906)
 * - SELECT registers for event configuration
 * - 64-bit counter result registers (LO + HI)
 *
 * @return Pointer to initialized TCP block_info_t, or NULL on allocation failure
 *
 * @note TCP L1 cache is critical for texture memory performance
 */
static block_info_t *create_gfx9_tcp_block(void)
{
	block_info_t *block = ALLOC(sizeof(block_info_t));
	if (!block)
		return NULL;

	/* Allocate counter register info array */
	counter_reg_info_t *counter_regs =
		ALLOC_ARRAY(counter_reg_info_t, GFX9_TCP_COUNTER_BLOCK_NUM_COUNTERS);
	if (!counter_regs) {
		FREE(block);
		return NULL;
	}

	/* Initialize TCP counter registers */
	create_counter_reg_info(&counter_regs[0], mmTCP_PERFCOUNTER0_SELECT, 0,
				mmTCP_PERFCOUNTER0_LO, mmTCP_PERFCOUNTER0_HI);
	create_counter_reg_info(&counter_regs[1], mmTCP_PERFCOUNTER1_SELECT, 0,
				mmTCP_PERFCOUNTER1_LO, mmTCP_PERFCOUNTER1_HI);
	create_counter_reg_info(&counter_regs[2], mmTCP_PERFCOUNTER2_SELECT, 0,
				mmTCP_PERFCOUNTER2_LO, mmTCP_PERFCOUNTER2_HI);
	create_counter_reg_info(&counter_regs[3], mmTCP_PERFCOUNTER3_SELECT, 0,
				mmTCP_PERFCOUNTER3_LO, mmTCP_PERFCOUNTER3_HI);

	/* Allocate dimensions - TCP is an SE-level block with SE+SA structure (2 dimensions) */
	dimension_t *dimensions = ALLOC_ARRAY(dimension_t, 2);
	if (!dimensions) {
		FREE(counter_regs);
		FREE(block);
		return NULL;
	}

	/* Initialize SE/SA dimensions - no WGP on GFX9 */
	dimensions[0].dim = HARDWARE_DIM_SE;
	dimensions[0].size = GFX9_NUM_SE;
	dimensions[1].dim = HARDWARE_DIM_SA;
	dimensions[1].size = GFX9_NUM_SA;

	/* Initialize TCP block */
	block->name = "TCP";
	block->id = HW_IP_BLOCK_TCP;
	block->instance_count = GFX9_TCP_COUNTER_BLOCK_NUM_INSTANCES; /* 16 instances */
	block->event_id_max = GFX9_TCP_COUNTER_BLOCK_MAX_EVENT; /* 99 */
	block->counter_count = GFX9_TCP_COUNTER_BLOCK_NUM_COUNTERS; /* 4 counters */
	block->dimensions = dimensions;
	block->dimension_count = 2; /* SE, SA */
	block->counter_reg_info = counter_regs;
	block->attr = GFX9_COUNTER_BLOCK_DFLT_ATTR;
	block->delay_info = NULL;
	block->spm_block_id = 0;

	return block;
}

/**
 * @brief Create and initialize the complete GFX9 architecture structure
 *
 * Constructs the full architecture descriptor for AMD GCN 5th gen (GFX9) GPUs
 * (e.g., Vega, MI50/MI60), including all hardware block definitions, register
 * mappings, control registers, and event mappings. This is the top-level
 * factory function for GFX9.
 *
 * Architecture configuration includes:
 * - Hardware topology: 4 SEs x 1 SA x 60 CUs (no WGP concept)
 * - 10 hardware blocks (SQ, CPC, CPF, TCC, GL2C, GRBM, SPI, TA, TCP)
 *   Note: GL2C is an alias for TCC registers on GFX9
 * - Per-block counter registers and dimensions
 * - Control register offsets for PM4 packet generation
 * - GFX9-specific SQ select masks and shader stage enable bits
 * - Event ID mappings from counter names to hardware event codes
 *
 * @return Pointer to initialized GFX9 architecture, or NULL on allocation failure
 *
 * @note Caller must call arch_destroy() to free the returned structure
 * @note All block creators must succeed or the entire arch creation fails
 * @see arch_destroy()
 */
arch_t *create_gfx9_arch(void)
{
	arch_t *arch = ALLOC(sizeof(arch_t));
	if (!arch)
		return NULL;

	/* Initialize architecture fields - gfx906 defaults */
	arch->type = ARCH_TYPE_GFX9;
	arch->num_xcc = GFX9_NUM_XCC;
	arch->num_se = GFX9_NUM_SE;
	arch->num_sa = GFX9_NUM_SA;
	arch->num_cu = GFX9_NUM_CU;
	arch->num_wgp_per_sa = GFX9_NUM_WGP_PER_SA;
	arch->command = NULL; /* Will be allocated as needed */

	/* Initialize block map */
	memset(&arch->block_map, 0, sizeof(block_info_map_t));

	/* Create and add block info entries */
	block_info_t *cpc_block = create_gfx9_cpc_block();
	block_info_t *cpf_block = create_gfx9_cpf_block();
	block_info_t *sq_block = create_gfx9_sq_block();
	block_info_t *grbm_block = create_gfx9_grbm_block();
	block_info_t *tcc_block = create_gfx9_tcc_block();
	block_info_t *gl2c_block = create_gfx9_gl2c_block();
	block_info_t *spi_block = create_gfx9_spi_block();
	block_info_t *ta_block = create_gfx9_ta_block();
	block_info_t *tcp_block = create_gfx9_tcp_block();

	if (!cpc_block || !cpf_block || !sq_block || !grbm_block || !tcc_block ||
	    !gl2c_block || !spi_block || !ta_block || !tcp_block) {
		FREE(arch);
		if (cpc_block) {
			FREE(cpc_block->dimensions);
			FREE(cpc_block->counter_reg_info);
			FREE(cpc_block);
		}
		if (cpf_block) {
			FREE(cpf_block->dimensions);
			FREE(cpf_block->counter_reg_info);
			FREE(cpf_block);
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
		if (tcc_block) {
			FREE(tcc_block->dimensions);
			FREE(tcc_block->counter_reg_info);
			FREE(tcc_block);
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
		return NULL;
	}

	/* Add blocks to the map */
	arch->block_map.blocks[HW_IP_BLOCK_CPC] = cpc_block;
	arch->block_map.blocks[HW_IP_BLOCK_CPF] = cpf_block;
	arch->block_map.blocks[HW_IP_BLOCK_SQ] = sq_block;
	arch->block_map.blocks[HW_IP_BLOCK_GRBM] = grbm_block;
	arch->block_map.blocks[HW_IP_BLOCK_TCC] = tcc_block;
	arch->block_map.blocks[HW_IP_BLOCK_GL2C] = gl2c_block;
	arch->block_map.blocks[HW_IP_BLOCK_SPI] = spi_block;
	arch->block_map.blocks[HW_IP_BLOCK_TA] = ta_block;
	arch->block_map.blocks[HW_IP_BLOCK_TCP] = tcp_block;
	/* Block count must be >= highest block ID + 1
	 * Using HW_IP_BLOCK_LAST allows for all possible blocks
	 */
	arch->block_map.block_count = HW_IP_BLOCK_LAST;

	/* Initialize control registers for GFX9 */
	arch->control_regs = (arch_control_regs_t) {
		.grbm_gfx_index = 0xC200,
		.cp_perfmon_cntl = 0xD808,
		.compute_perfcount_enable = 0xCE0B,
		.sq_perfcounter_ctrl2 = 0xD9E2,
		.sq_perfcounter_ctrl = 0xD9E0,
		.sq_perfcounter_mask = 0xD9E1,
		.rlc_perfmon_clk_cntl = 0xDCBF,
		.uconfig_space_start = UCONFIG_SPACE_START,            /* 0x0000C000 */
		.persistent_space_start = PERSISTENT_SPACE_START,      /* 0x00002C00 */
		.cs_partial_flush_event = VGT_EVENT_TYPE_CS_PARTIAL_FLUSH, /* 0x07 */
		.event_index_flush = 4,                                /* Event flush index */
		.gcr_cntl_default = 0,                                 /* GFX9 doesn't use GCR_CNTL */
		.cp_coher_cntl_default = 0x28C40000,                   /* TC_ACTION_ENA|TCL1_ACTION_ENA|TC_WB_ACTION_ENA|SH_ICACHE|SH_KCACHE */
		.poll_interval_default = 0x10,                         /* Poll every 4K */
		.counter_control_bits = {
			.sq_ps_en_bit = 0,                             /* PS enable bit */
			.sq_gs_en_bit = 2,                             /* GS enable bit */
			.sq_hs_en_bit = 4,                             /* HS enable bit */
			.sq_cs_en_bit = 6,                             /* CS enable bit */
			.sq_vs_en_bit = 1,                             /* VS enable bit (GFX9) */
			.sq_es_en_bit = 3,                             /* ES enable bit (GFX9) */
			.sq_ls_en_bit = 5                              /* LS enable bit (GFX9) */
		},
		.sq_select_masks = {
			.simd_mask = 0xF,
			.sqc_bank_mask = 0xF,
			.sqc_client_mask = 0xF,
			.simd_mask_shift = 24,
			.sqc_bank_mask_shift = 12,
			.sqc_client_mask_shift = 16
		},
		.perfmon_states = {
			.perfmon_state_disable = 0,                    /* Disable state */
			.perfmon_state_enable = 1,                     /* Enable state */
			.perfmon_state_stop = 2,                       /* Stop state */
			.perfmon_sample_bit = 10                       /* Sample enable bit */
		}
	};

	/* Set up the event map for this architecture */
	arch->event_map = get_gfx9_events();
	arch->event_count = get_gfx9_event_count();

	return arch;
}
