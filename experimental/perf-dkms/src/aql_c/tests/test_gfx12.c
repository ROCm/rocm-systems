/**
 * @file test_gfx12.c
 * @brief Userspace tests for GFX12 architecture creation and validation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* Include the AQL structures and creators */
#include "../aql_structures.h"
#include "../arch_creator_common.h"

/* Test result tracking */
static int tests_run = 0;
static int tests_passed = 0;

#define TEST_ASSERT(condition, message)                \
	do {                                           \
		tests_run++;                           \
		if (condition) {                       \
			tests_passed++;                \
			printf("PASS: %s\n", message); \
		} else {                               \
			printf("FAIL: %s\n", message); \
		}                                      \
	} while (0)

/* Forward declaration of external function */
extern arch_t *arch_create_by_name(const char *arch_name);

/* Helper function to free architecture structure */
static void free_arch(arch_t *arch)
{
	if (!arch)
		return;

	for (int i = 0; i < HW_IP_BLOCK_LAST; i++) {
		block_info_t *block = arch->block_map.blocks[i];
		if (block) {
			if (block->counter_reg_info) {
				free(block->counter_reg_info);
			}
			if (block->dimensions) {
				free(block->dimensions);
			}
			free(block);
		}
	}
	free(arch);
}

/* Test basic GFX12 architecture creation */
static void test_gfx12_arch_creation(void)
{
	printf("\n=== Testing GFX12 Architecture Creation ===\n");

	arch_t *arch = arch_create_by_name("gfx12");
	TEST_ASSERT(arch != NULL, "GFX12 architecture creation");

	if (arch) {
		TEST_ASSERT(arch->type == ARCH_TYPE_GFX12, "Architecture type is GFX12");
		TEST_ASSERT(arch->num_xcc == 1, "GFX12 XCC count is 1");
		TEST_ASSERT(arch->num_se == 4, "GFX12 SE count is 4");
		TEST_ASSERT(arch->num_sa == 2, "GFX12 SA count is 2");
		TEST_ASSERT(arch->num_cu == 64, "GFX12 CU count is 64");
		TEST_ASSERT(arch->num_wgp_per_sa == 4, "GFX12 WGP per SA count is 4");
		TEST_ASSERT(arch->block_map.block_count == 14, "Block map has 14 blocks");

		free_arch(arch);
	}
}

/* Test CPC block configuration */
static void test_gfx12_cpc_block(void)
{
	printf("\n=== Testing GFX12 CPC Block ===\n");

	arch_t *arch = arch_create_by_name("gfx12");
	TEST_ASSERT(arch != NULL, "Architecture created for CPC block test");

	if (arch) {
		block_info_t *cpc_block = arch->block_map.blocks[HW_IP_BLOCK_CPC];
		TEST_ASSERT(cpc_block != NULL, "CPC block exists");

		if (cpc_block) {
			TEST_ASSERT(strcmp(cpc_block->name, "CPC") == 0,
				    "CPC block name is correct");
			TEST_ASSERT(cpc_block->id == HW_IP_BLOCK_CPC, "CPC block ID is correct");
			TEST_ASSERT(cpc_block->counter_count == 2, "CPC block has 2 counters");
			TEST_ASSERT(cpc_block->counter_reg_info != NULL,
				    "CPC counter register info exists");
			TEST_ASSERT(cpc_block->dimension_count == 1, "CPC block has 1 dimension");
			TEST_ASSERT(cpc_block->dimensions != NULL, "CPC dimensions array exists");

			if (cpc_block->dimensions) {
				TEST_ASSERT(cpc_block->dimensions[0].dim == HARDWARE_DIM_XCC,
					    "CPC dimension is XCC");
				TEST_ASSERT(cpc_block->dimensions[0].size == 1,
					    "CPC XCC dimension size is 1");
			}

			/* Test counter register info */
			if (cpc_block->counter_reg_info) {
				counter_reg_info_t *reg0 = &cpc_block->counter_reg_info[0];
				TEST_ASSERT(reg0->select_addr == 55305,
					    "CPC counter 0 select address");
				TEST_ASSERT(reg0->register_addr_lo == 53254,
					    "CPC counter 0 LO address");
				TEST_ASSERT(reg0->register_addr_hi == 53255,
					    "CPC counter 0 HI address");
				TEST_ASSERT(reg0->allocation.state == COUNTER_STATE_FREE,
					    "CPC counter 0 initial state is FREE");

				counter_reg_info_t *reg1 = &cpc_block->counter_reg_info[1];
				TEST_ASSERT(reg1->select_addr == 55299,
					    "CPC counter 1 select address");
				TEST_ASSERT(reg1->register_addr_lo == 53252,
					    "CPC counter 1 LO address");
				TEST_ASSERT(reg1->register_addr_hi == 53253,
					    "CPC counter 1 HI address");
				TEST_ASSERT(reg1->allocation.state == COUNTER_STATE_FREE,
					    "CPC counter 1 initial state is FREE");
			}
		}

		free_arch(arch);
	}
}

/* Test SQ block configuration */
static void test_gfx12_sq_block(void)
{
	printf("\n=== Testing GFX12 SQ Block ===\n");

	arch_t *arch = arch_create_by_name("gfx12");
	TEST_ASSERT(arch != NULL, "Architecture created for SQ block test");

	if (arch) {
		block_info_t *sq_block = arch->block_map.blocks[HW_IP_BLOCK_SQ];
		TEST_ASSERT(sq_block != NULL, "SQ block exists");

		if (sq_block) {
			TEST_ASSERT(strcmp(sq_block->name, "SQ") == 0, "SQ block name is correct");
			TEST_ASSERT(sq_block->id == HW_IP_BLOCK_SQ, "SQ block ID is correct");
			TEST_ASSERT(sq_block->counter_count == 8, "SQ block has 8 counters");
			TEST_ASSERT(sq_block->counter_reg_info != NULL,
				    "SQ counter register info exists");
			TEST_ASSERT(sq_block->dimension_count == 3, "SQ block has 3 dimensions");
			TEST_ASSERT(sq_block->dimensions != NULL, "SQ dimensions array exists");
			TEST_ASSERT(sq_block->spm_block_id == 9, "SQ SPM block ID is correct");

			/* Test dimensions */
			if (sq_block->dimensions) {
				TEST_ASSERT(sq_block->dimensions[0].dim == HARDWARE_DIM_SE,
					    "SQ dimension 0 is SE");
				TEST_ASSERT(sq_block->dimensions[0].size == 4,
					    "SQ SE dimension size is 4");

				TEST_ASSERT(sq_block->dimensions[1].dim == HARDWARE_DIM_SA,
					    "SQ dimension 1 is SA");
				TEST_ASSERT(sq_block->dimensions[1].size == 2,
					    "SQ SA dimension size is 2");

				TEST_ASSERT(sq_block->dimensions[2].dim == HARDWARE_DIM_WGP,
					    "SQ dimension 2 is WGP");
				TEST_ASSERT(sq_block->dimensions[2].size == 4,
					    "SQ WGP dimension size is 4");
			}

			/* Test first few counter register configurations */
			if (sq_block->counter_reg_info) {
				counter_reg_info_t *reg0 = &sq_block->counter_reg_info[0];
				TEST_ASSERT(reg0->select_addr == 55744,
					    "SQ counter 0 select address");
				TEST_ASSERT(reg0->control_addr == 55776,
					    "SQ counter 0 control address");
				TEST_ASSERT(reg0->register_addr_lo == 53696,
					    "SQ counter 0 LO address");
				TEST_ASSERT(reg0->allocation.state == COUNTER_STATE_FREE,
					    "SQ counter 0 initial state is FREE");

				counter_reg_info_t *reg1 = &sq_block->counter_reg_info[1];
				TEST_ASSERT(reg1->select_addr == 55746,
					    "SQ counter 1 select address");
				TEST_ASSERT(reg1->control_addr == 55776,
					    "SQ counter 1 control address");
				TEST_ASSERT(reg1->register_addr_lo == 53698,
					    "SQ counter 1 LO address");
				TEST_ASSERT(reg1->allocation.state == COUNTER_STATE_FREE,
					    "SQ counter 1 initial state is FREE");
			}
		}

		free_arch(arch);
	}
}

/* Test GRBM block configuration */
static void test_gfx12_grbm_block(void)
{
	printf("\n=== Testing GFX12 GRBM Block ===\n");

	arch_t *arch = arch_create_by_name("gfx12");
	TEST_ASSERT(arch != NULL, "Architecture created for GRBM block test");

	if (arch) {
		block_info_t *grbm_block = arch->block_map.blocks[HW_IP_BLOCK_GRBM];
		TEST_ASSERT(grbm_block != NULL, "GRBM block exists");

		if (grbm_block) {
			TEST_ASSERT(strcmp(grbm_block->name, "GRBM") == 0,
				    "GRBM block name is correct");
			TEST_ASSERT(grbm_block->id == HW_IP_BLOCK_GRBM, "GRBM block ID is correct");
			TEST_ASSERT(grbm_block->counter_count == 2, "GRBM block has 2 counters");
			TEST_ASSERT(grbm_block->event_id_max == 51, "GRBM block max event is 51");
			TEST_ASSERT(grbm_block->counter_reg_info != NULL,
				    "GRBM counter register info exists");
			TEST_ASSERT(grbm_block->dimension_count == 1, "GRBM block has 1 dimension");
			TEST_ASSERT(grbm_block->dimensions != NULL, "GRBM dimensions array exists");

			if (grbm_block->dimensions) {
				TEST_ASSERT(grbm_block->dimensions[0].dim == HARDWARE_DIM_XCC,
					    "GRBM dimension is XCC");
				TEST_ASSERT(grbm_block->dimensions[0].size == 1,
					    "GRBM XCC dimension size is 1");
			}

			/* Test counter register info */
			if (grbm_block->counter_reg_info) {
				counter_reg_info_t *reg0 = &grbm_block->counter_reg_info[0];
				TEST_ASSERT(reg0->select_addr == 14400,
					    "GRBM counter 0 select address");
				TEST_ASSERT(reg0->register_addr_lo == 12352,
					    "GRBM counter 0 LO address");
				TEST_ASSERT(reg0->register_addr_hi == 12353,
					    "GRBM counter 0 HI address");
				TEST_ASSERT(reg0->allocation.state == COUNTER_STATE_FREE,
					    "GRBM counter 0 initial state is FREE");

				counter_reg_info_t *reg1 = &grbm_block->counter_reg_info[1];
				TEST_ASSERT(reg1->select_addr == 14401,
					    "GRBM counter 1 select address");
				TEST_ASSERT(reg1->register_addr_lo == 12355,
					    "GRBM counter 1 LO address");
				TEST_ASSERT(reg1->register_addr_hi == 12356,
					    "GRBM counter 1 HI address");
				TEST_ASSERT(reg1->allocation.state == COUNTER_STATE_FREE,
					    "GRBM counter 1 initial state is FREE");
			}
		}

		free_arch(arch);
	}
}

/* Test GL2C block configuration */
static void test_gfx12_gl2c_block(void)
{
	printf("\n=== Testing GFX12 GL2C Block ===\n");

	arch_t *arch = arch_create_by_name("gfx12");
	TEST_ASSERT(arch != NULL, "Architecture created for GL2C block test");

	if (arch) {
		block_info_t *gl2c_block = arch->block_map.blocks[HW_IP_BLOCK_GL2C];
		TEST_ASSERT(gl2c_block != NULL, "GL2C block exists");

		if (gl2c_block) {
			TEST_ASSERT(strcmp(gl2c_block->name, "GL2C") == 0,
				    "GL2C block name is correct");
			TEST_ASSERT(gl2c_block->id == HW_IP_BLOCK_GL2C, "GL2C block ID is correct");
			TEST_ASSERT(gl2c_block->counter_count == 4, "GL2C block has 4 counters");
			TEST_ASSERT(gl2c_block->event_id_max == 249, "GL2C block max event is 249");
			TEST_ASSERT(gl2c_block->instance_count == 16,
				    "GL2C block has 16 instances");
			TEST_ASSERT(gl2c_block->counter_reg_info != NULL,
				    "GL2C counter register info exists");
			TEST_ASSERT(gl2c_block->dimension_count == 1, "GL2C block has 1 dimension");
			TEST_ASSERT(gl2c_block->dimensions != NULL, "GL2C dimensions array exists");

			if (gl2c_block->dimensions) {
				TEST_ASSERT(gl2c_block->dimensions[0].dim == HARDWARE_DIM_XCC,
					    "GL2C dimension is XCC");
				TEST_ASSERT(gl2c_block->dimensions[0].size == 1,
					    "GL2C XCC dimension size is 1");
			}

			/* Test counter register info */
			if (gl2c_block->counter_reg_info) {
				counter_reg_info_t *reg0 = &gl2c_block->counter_reg_info[0];
				TEST_ASSERT(reg0->select_addr == 15232,
					    "GL2C counter 0 select address");
				TEST_ASSERT(reg0->register_addr_lo == 13184,
					    "GL2C counter 0 LO address");
				TEST_ASSERT(reg0->register_addr_hi == 13185,
					    "GL2C counter 0 HI address");
				TEST_ASSERT(reg0->control_addr == 0,
					    "GL2C counter 0 has no control address");
				TEST_ASSERT(reg0->allocation.state == COUNTER_STATE_FREE,
					    "GL2C counter 0 initial state is FREE");

				counter_reg_info_t *reg1 = &gl2c_block->counter_reg_info[1];
				TEST_ASSERT(reg1->select_addr == 15234,
					    "GL2C counter 1 select address");
				TEST_ASSERT(reg1->register_addr_lo == 13186,
					    "GL2C counter 1 LO address");
				TEST_ASSERT(reg1->register_addr_hi == 13187,
					    "GL2C counter 1 HI address");
				TEST_ASSERT(reg1->control_addr == 0,
					    "GL2C counter 1 has no control address");

				counter_reg_info_t *reg2 = &gl2c_block->counter_reg_info[2];
				TEST_ASSERT(reg2->select_addr == 15236,
					    "GL2C counter 2 select address");
				TEST_ASSERT(reg2->register_addr_lo == 13188,
					    "GL2C counter 2 LO address");
				TEST_ASSERT(reg2->register_addr_hi == 13189,
					    "GL2C counter 2 HI address");
				TEST_ASSERT(reg2->control_addr == 0,
					    "GL2C counter 2 has no control address");

				counter_reg_info_t *reg3 = &gl2c_block->counter_reg_info[3];
				TEST_ASSERT(reg3->select_addr == 15238,
					    "GL2C counter 3 select address");
				TEST_ASSERT(reg3->register_addr_lo == 13190,
					    "GL2C counter 3 LO address");
				TEST_ASSERT(reg3->register_addr_hi == 13191,
					    "GL2C counter 3 HI address");
				TEST_ASSERT(reg3->control_addr == 0,
					    "GL2C counter 3 has no control address");
			}
		}

		free_arch(arch);
	}
}

/* Test SPI block configuration */
static void test_gfx12_spi_block(void)
{
	printf("\n=== Testing GFX12 SPI Block ===\n");

	arch_t *arch = arch_create_by_name("gfx12");
	TEST_ASSERT(arch != NULL, "Architecture created for SPI block test");

	if (arch) {
		block_info_t *spi_block = arch->block_map.blocks[HW_IP_BLOCK_SPI];
		TEST_ASSERT(spi_block != NULL, "SPI block exists");

		if (spi_block) {
			TEST_ASSERT(strcmp(spi_block->name, "SPI") == 0,
				    "SPI block name is correct");
			TEST_ASSERT(spi_block->id == HW_IP_BLOCK_SPI, "SPI block ID is correct");
			TEST_ASSERT(spi_block->counter_count == 6, "SPI block has 6 counters");
			TEST_ASSERT(spi_block->event_id_max == 318, "SPI block max event is 318");
			TEST_ASSERT(spi_block->instance_count == 1, "SPI block has 1 instance");
			TEST_ASSERT(spi_block->counter_reg_info != NULL,
				    "SPI counter register info exists");
			TEST_ASSERT(spi_block->dimension_count == 2, "SPI block has 2 dimensions");
			TEST_ASSERT(spi_block->dimensions != NULL, "SPI dimensions array exists");

			if (spi_block->dimensions) {
				TEST_ASSERT(spi_block->dimensions[0].dim == HARDWARE_DIM_XCC,
					    "SPI dimension 0 is XCC");
				TEST_ASSERT(spi_block->dimensions[0].size == 1,
					    "SPI XCC dimension size is 1");
				TEST_ASSERT(spi_block->dimensions[1].dim == HARDWARE_DIM_SE,
					    "SPI dimension 1 is SE");
				TEST_ASSERT(spi_block->dimensions[1].size == 4,
					    "SPI SE dimension size is 4");
			}

			/* Test counter register info */
			if (spi_block->counter_reg_info) {
				counter_reg_info_t *reg0 = &spi_block->counter_reg_info[0];
				TEST_ASSERT(reg0->select_addr == 14720,
					    "SPI counter 0 select address");
				TEST_ASSERT(reg0->register_addr_lo == 12673,
					    "SPI counter 0 LO address");
				TEST_ASSERT(reg0->register_addr_hi == 12672,
					    "SPI counter 0 HI address");
				TEST_ASSERT(reg0->control_addr == 0,
					    "SPI counter 0 has no control address");
				TEST_ASSERT(reg0->allocation.state == COUNTER_STATE_FREE,
					    "SPI counter 0 initial state is FREE");

				counter_reg_info_t *reg1 = &spi_block->counter_reg_info[1];
				TEST_ASSERT(reg1->select_addr == 14721,
					    "SPI counter 1 select address");
				TEST_ASSERT(reg1->register_addr_lo == 12675,
					    "SPI counter 1 LO address");
				TEST_ASSERT(reg1->register_addr_hi == 12674,
					    "SPI counter 1 HI address");
				TEST_ASSERT(reg1->control_addr == 0,
					    "SPI counter 1 has no control address");

				counter_reg_info_t *reg5 = &spi_block->counter_reg_info[5];
				TEST_ASSERT(reg5->select_addr == 14725,
					    "SPI counter 5 select address");
				TEST_ASSERT(reg5->register_addr_lo == 12683,
					    "SPI counter 5 LO address");
				TEST_ASSERT(reg5->register_addr_hi == 12682,
					    "SPI counter 5 HI address");
				TEST_ASSERT(reg5->control_addr == 0,
					    "SPI counter 5 has no control address");
			}
		}

		free_arch(arch);
	}
}

/* Test GFX12 TD block configuration */
static void test_gfx12_td_block(void)
{
	printf("\n=== Testing GFX12 TD Block ===\n");
	arch_t *arch = arch_create_by_name("gfx12");
	TEST_ASSERT(arch != NULL, "Architecture created for TD block test");

	if (arch) {
		block_info_t *td_block = arch->block_map.blocks[HW_IP_BLOCK_TD];
		TEST_ASSERT(td_block != NULL, "TD block exists");

		if (td_block) {
			/* Validate basic block properties */
			TEST_ASSERT(strcmp(td_block->name, "TD") == 0, "TD block name is correct");
			TEST_ASSERT(td_block->id == HW_IP_BLOCK_TD, "TD block type correct");
			TEST_ASSERT(td_block->counter_count == 2, "TD block has 2 counters");
			TEST_ASSERT(td_block->event_id_max == 127, "TD block max event ID is 127");
			TEST_ASSERT(td_block->instance_count == 2, "TD block has 2 instances");

			/* Validate WGP dimensions - TD is a WGP-level block with 4 dimensions */
			TEST_ASSERT(td_block->dimensions != NULL, "TD block dimensions exist");
			TEST_ASSERT(td_block->dimension_count == 4, "TD block has 4 dimensions");

			if (td_block->dimensions && td_block->dimension_count >= 4) {
				TEST_ASSERT(td_block->dimensions[0].dim == HARDWARE_DIM_XCC,
					    "TD dimension 0 is XCC");
				TEST_ASSERT(td_block->dimensions[0].size == 1,
					    "TD XCC dimension size is 1");
				TEST_ASSERT(td_block->dimensions[1].dim == HARDWARE_DIM_SE,
					    "TD dimension 1 is SE");
				TEST_ASSERT(td_block->dimensions[1].size == 4,
					    "TD SE dimension size is 4");
				TEST_ASSERT(td_block->dimensions[2].dim == HARDWARE_DIM_SA,
					    "TD dimension 2 is SA");
				TEST_ASSERT(td_block->dimensions[2].size == 2,
					    "TD SA dimension size is 2");
				TEST_ASSERT(td_block->dimensions[3].dim == HARDWARE_DIM_WGP,
					    "TD dimension 3 is WGP");
				TEST_ASSERT(td_block->dimensions[3].size == 4,
					    "TD WGP dimension size is 4");
			}

			/* Validate counter register info */
			TEST_ASSERT(td_block->counter_reg_info != NULL,
				    "TD block counter registers exist");
			if (td_block->counter_reg_info) {
				counter_reg_info_t *reg0 = &td_block->counter_reg_info[0];
				counter_reg_info_t *reg1 = &td_block->counter_reg_info[1];

				/* Counter 0 registers - based on estimated addresses */
				TEST_ASSERT(reg0->select_addr == 15296,
					    "TD counter 0 select register (0x3bc0)");
				TEST_ASSERT(reg0->control_addr == 0,
					    "TD counter 0 no control register");
				TEST_ASSERT(reg0->register_addr_lo == 13248,
					    "TD counter 0 lo register (0x33c0)");
				TEST_ASSERT(reg0->register_addr_hi == 13249,
					    "TD counter 0 hi register (0x33c1)");

				/* Counter 1 registers */
				TEST_ASSERT(reg1->select_addr == 15298,
					    "TD counter 1 select register (0x3bc2)");
				TEST_ASSERT(reg1->control_addr == 0,
					    "TD counter 1 no control register");
				TEST_ASSERT(reg1->register_addr_lo == 13250,
					    "TD counter 1 lo register (0x33c2)");
				TEST_ASSERT(reg1->register_addr_hi == 13251,
					    "TD counter 1 hi register (0x33c3)");
			}
		}

		free_arch(arch);
	}
}

/* Test GFX12 TCC block configuration */
static void test_gfx12_tcc_block(void)
{
	printf("\n=== Testing GFX12 TCC Block ===\n");
	arch_t *arch = arch_create_by_name("gfx12");
	TEST_ASSERT(arch != NULL, "Architecture created for TCC block test");

	if (arch) {
		block_info_t *tcc_block = arch->block_map.blocks[HW_IP_BLOCK_TCC];
		TEST_ASSERT(tcc_block != NULL, "TCC block exists");

		if (tcc_block) {
			/* Validate basic block properties */
			TEST_ASSERT(strcmp(tcc_block->name, "TCC") == 0,
				    "TCC block name is correct");
			TEST_ASSERT(tcc_block->id == HW_IP_BLOCK_TCC, "TCC block type correct");
			TEST_ASSERT(tcc_block->counter_count == 4, "TCC block has 4 counters");
			TEST_ASSERT(tcc_block->event_id_max == 255,
				    "TCC block max event ID is 255");
			TEST_ASSERT(tcc_block->instance_count == 16, "TCC block has 16 instances");

			/* Validate global dimensions - TCC is a global block with 1 dimension */
			TEST_ASSERT(tcc_block->dimensions != NULL, "TCC block dimensions exist");
			TEST_ASSERT(tcc_block->dimension_count == 1, "TCC block has 1 dimension");

			if (tcc_block->dimensions && tcc_block->dimension_count >= 1) {
				TEST_ASSERT(tcc_block->dimensions[0].dim == HARDWARE_DIM_XCC,
					    "TCC dimension 0 is XCC");
				TEST_ASSERT(tcc_block->dimensions[0].size == 1,
					    "TCC XCC dimension size is 1");
			}

			/* Validate counter register info */
			TEST_ASSERT(tcc_block->counter_reg_info != NULL,
				    "TCC block counter registers exist");
			if (tcc_block->counter_reg_info) {
				counter_reg_info_t *reg0 = &tcc_block->counter_reg_info[0];
				counter_reg_info_t *reg1 = &tcc_block->counter_reg_info[1];
				counter_reg_info_t *reg2 = &tcc_block->counter_reg_info[2];
				counter_reg_info_t *reg3 = &tcc_block->counter_reg_info[3];

				/* Counter 0 registers - based on estimated addresses */
				TEST_ASSERT(reg0->select_addr == 15424,
					    "TCC counter 0 select register (0x3c40)");
				TEST_ASSERT(reg0->control_addr == 0,
					    "TCC counter 0 no control register");
				TEST_ASSERT(reg0->register_addr_lo == 13376,
					    "TCC counter 0 lo register (0x3440)");
				TEST_ASSERT(reg0->register_addr_hi == 13377,
					    "TCC counter 0 hi register (0x3441)");

				/* Counter 1 registers */
				TEST_ASSERT(reg1->select_addr == 15426,
					    "TCC counter 1 select register (0x3c42)");
				TEST_ASSERT(reg1->control_addr == 0,
					    "TCC counter 1 no control register");
				TEST_ASSERT(reg1->register_addr_lo == 13378,
					    "TCC counter 1 lo register (0x3442)");
				TEST_ASSERT(reg1->register_addr_hi == 13379,
					    "TCC counter 1 hi register (0x3443)");

				/* Counter 2 registers */
				TEST_ASSERT(reg2->select_addr == 15428,
					    "TCC counter 2 select register (0x3c44)");
				TEST_ASSERT(reg2->control_addr == 0,
					    "TCC counter 2 no control register");
				TEST_ASSERT(reg2->register_addr_lo == 13380,
					    "TCC counter 2 lo register (0x3444)");
				TEST_ASSERT(reg2->register_addr_hi == 13381,
					    "TCC counter 2 hi register (0x3445)");

				/* Counter 3 registers */
				TEST_ASSERT(reg3->select_addr == 15430,
					    "TCC counter 3 select register (0x3c46)");
				TEST_ASSERT(reg3->control_addr == 0,
					    "TCC counter 3 no control register");
				TEST_ASSERT(reg3->register_addr_lo == 13382,
					    "TCC counter 3 lo register (0x3446)");
				TEST_ASSERT(reg3->register_addr_hi == 13383,
					    "TCC counter 3 hi register (0x3447)");
			}
		}

		free_arch(arch);
	}
}

/* Test GFX12 SX block configuration */
static void test_gfx12_sx_block(void)
{
	printf("\n=== Testing GFX12 SX Block ===\n");
	arch_t *arch = arch_create_by_name("gfx12");
	TEST_ASSERT(arch != NULL, "Architecture created for SX block test");

	if (arch) {
		block_info_t *sx_block = arch->block_map.blocks[HW_IP_BLOCK_SX];
		TEST_ASSERT(sx_block != NULL, "SX block exists");

		if (sx_block) {
			/* Validate basic block properties */
			TEST_ASSERT(strcmp(sx_block->name, "SX") == 0, "SX block name is correct");
			TEST_ASSERT(sx_block->id == HW_IP_BLOCK_SX, "SX block type correct");
			TEST_ASSERT(sx_block->counter_count == 4, "SX block has 4 counters");
			TEST_ASSERT(sx_block->event_id_max == 189, "SX block max event ID is 189");
			TEST_ASSERT(sx_block->instance_count == 1, "SX block has 1 instance");

			/* Validate SE dimensions - SX is a SE block with 2 dimensions */
			TEST_ASSERT(sx_block->dimensions != NULL, "SX block dimensions exist");
			TEST_ASSERT(sx_block->dimension_count == 2, "SX block has 2 dimensions");

			if (sx_block->dimensions && sx_block->dimension_count >= 2) {
				TEST_ASSERT(sx_block->dimensions[0].dim == HARDWARE_DIM_XCC,
					    "SX dimension 0 is XCC");
				TEST_ASSERT(sx_block->dimensions[0].size == 1,
					    "SX XCC dimension size is 1");
				TEST_ASSERT(sx_block->dimensions[1].dim == HARDWARE_DIM_SE,
					    "SX dimension 1 is SE");
				TEST_ASSERT(sx_block->dimensions[1].size == 4,
					    "SX SE dimension size is 4");
			}

			/* Validate counter register info */
			TEST_ASSERT(sx_block->counter_reg_info != NULL,
				    "SX block counter registers exist");
			if (sx_block->counter_reg_info) {
				counter_reg_info_t *reg0 = &sx_block->counter_reg_info[0];
				counter_reg_info_t *reg1 = &sx_block->counter_reg_info[1];
				counter_reg_info_t *reg2 = &sx_block->counter_reg_info[2];
				counter_reg_info_t *reg3 = &sx_block->counter_reg_info[3];

				/* Counter 0 registers - based on estimated addresses */
				TEST_ASSERT(reg0->select_addr == 14976,
					    "SX counter 0 select register (0x3a80)");
				TEST_ASSERT(reg0->control_addr == 0,
					    "SX counter 0 no control register");
				TEST_ASSERT(reg0->register_addr_lo == 12928,
					    "SX counter 0 lo register (0x3280)");
				TEST_ASSERT(reg0->register_addr_hi == 12929,
					    "SX counter 0 hi register (0x3281)");

				/* Counter 1 registers */
				TEST_ASSERT(reg1->select_addr == 14978,
					    "SX counter 1 select register (0x3a82)");
				TEST_ASSERT(reg1->control_addr == 0,
					    "SX counter 1 no control register");
				TEST_ASSERT(reg1->register_addr_lo == 12930,
					    "SX counter 1 lo register (0x3282)");
				TEST_ASSERT(reg1->register_addr_hi == 12931,
					    "SX counter 1 hi register (0x3283)");

				/* Counter 2 registers */
				TEST_ASSERT(reg2->select_addr == 14980,
					    "SX counter 2 select register (0x3a84)");
				TEST_ASSERT(reg2->control_addr == 0,
					    "SX counter 2 no control register");
				TEST_ASSERT(reg2->register_addr_lo == 12932,
					    "SX counter 2 lo register (0x3284)");
				TEST_ASSERT(reg2->register_addr_hi == 12933,
					    "SX counter 2 hi register (0x3285)");

				/* Counter 3 registers */
				TEST_ASSERT(reg3->select_addr == 14982,
					    "SX counter 3 select register (0x3a86)");
				TEST_ASSERT(reg3->control_addr == 0,
					    "SX counter 3 no control register");
				TEST_ASSERT(reg3->register_addr_lo == 12934,
					    "SX counter 3 lo register (0x3286)");
				TEST_ASSERT(reg3->register_addr_hi == 12935,
					    "SX counter 3 hi register (0x3287)");
			}
		}

		free_arch(arch);
	}
}

/* Test GFX12 DB block configuration */
static void test_gfx12_db_block(void)
{
	printf("\n=== Testing GFX12 DB Block ===\n");
	arch_t *arch = arch_create_by_name("gfx12");
	TEST_ASSERT(arch != NULL, "Architecture created for DB block test");

	if (arch) {
		block_info_t *db_block = arch->block_map.blocks[HW_IP_BLOCK_DB];
		TEST_ASSERT(db_block != NULL, "DB block exists");

		if (db_block) {
			/* Validate basic block properties */
			TEST_ASSERT(strcmp(db_block->name, "DB") == 0, "DB block name is correct");
			TEST_ASSERT(db_block->id == HW_IP_BLOCK_DB, "DB block type correct");
			TEST_ASSERT(db_block->counter_count == 4, "DB block has 4 counters");
			TEST_ASSERT(db_block->event_id_max == 218, "DB block max event ID is 218");
			TEST_ASSERT(db_block->instance_count == 1, "DB block has 1 instance");

			/* Validate SE dimensions - DB is a SE block with 2 dimensions */
			TEST_ASSERT(db_block->dimensions != NULL, "DB block dimensions exist");
			TEST_ASSERT(db_block->dimension_count == 2, "DB block has 2 dimensions");

			if (db_block->dimensions && db_block->dimension_count >= 2) {
				TEST_ASSERT(db_block->dimensions[0].dim == HARDWARE_DIM_XCC,
					    "DB dimension 0 is XCC");
				TEST_ASSERT(db_block->dimensions[0].size == 1,
					    "DB XCC dimension size is 1");
				TEST_ASSERT(db_block->dimensions[1].dim == HARDWARE_DIM_SE,
					    "DB dimension 1 is SE");
				TEST_ASSERT(db_block->dimensions[1].size == 4,
					    "DB SE dimension size is 4");
			}

			/* Validate counter register info */
			TEST_ASSERT(db_block->counter_reg_info != NULL,
				    "DB block counter registers exist");
			if (db_block->counter_reg_info) {
				counter_reg_info_t *reg0 = &db_block->counter_reg_info[0];
				counter_reg_info_t *reg1 = &db_block->counter_reg_info[1];
				counter_reg_info_t *reg2 = &db_block->counter_reg_info[2];
				counter_reg_info_t *reg3 = &db_block->counter_reg_info[3];

				/* Counter 0 registers - based on estimated addresses */
				TEST_ASSERT(reg0->select_addr == 14592,
					    "DB counter 0 select register (0x3900)");
				TEST_ASSERT(reg0->control_addr == 0,
					    "DB counter 0 no control register");
				TEST_ASSERT(reg0->register_addr_lo == 12800,
					    "DB counter 0 lo register (0x3200)");
				TEST_ASSERT(reg0->register_addr_hi == 12801,
					    "DB counter 0 hi register (0x3201)");

				/* Counter 1 registers */
				TEST_ASSERT(reg1->select_addr == 14594,
					    "DB counter 1 select register (0x3902)");
				TEST_ASSERT(reg1->control_addr == 0,
					    "DB counter 1 no control register");
				TEST_ASSERT(reg1->register_addr_lo == 12802,
					    "DB counter 1 lo register (0x3202)");
				TEST_ASSERT(reg1->register_addr_hi == 12803,
					    "DB counter 1 hi register (0x3203)");

				/* Counter 2 registers */
				TEST_ASSERT(reg2->select_addr == 14596,
					    "DB counter 2 select register (0x3904)");
				TEST_ASSERT(reg2->control_addr == 0,
					    "DB counter 2 no control register");
				TEST_ASSERT(reg2->register_addr_lo == 12804,
					    "DB counter 2 lo register (0x3204)");
				TEST_ASSERT(reg2->register_addr_hi == 12805,
					    "DB counter 2 hi register (0x3205)");

				/* Counter 3 registers */
				TEST_ASSERT(reg3->select_addr == 14598,
					    "DB counter 3 select register (0x3906)");
				TEST_ASSERT(reg3->control_addr == 0,
					    "DB counter 3 no control register");
				TEST_ASSERT(reg3->register_addr_lo == 12806,
					    "DB counter 3 lo register (0x3206)");
				TEST_ASSERT(reg3->register_addr_hi == 12807,
					    "DB counter 3 hi register (0x3207)");
			}
		}

		free_arch(arch);
	}
}

/* Test GFX12 PA_SC block configuration */
static void test_gfx12_pa_sc_block(void)
{
	printf("\n=== Testing GFX12 PA_SC Block ===\n");
	arch_t *arch = arch_create_by_name("gfx12");
	TEST_ASSERT(arch != NULL, "Architecture created for PA_SC block test");

	if (arch) {
		block_info_t *pa_sc_block = arch->block_map.blocks[HW_IP_BLOCK_PA_SC];
		TEST_ASSERT(pa_sc_block != NULL, "PA_SC block exists");

		if (pa_sc_block) {
			/* Validate basic block properties */
			TEST_ASSERT(strcmp(pa_sc_block->name, "PA_SC") == 0,
				    "PA_SC block name is correct");
			TEST_ASSERT(pa_sc_block->id == HW_IP_BLOCK_PA_SC,
				    "PA_SC block type correct");
			TEST_ASSERT(pa_sc_block->counter_count == 4, "PA_SC block has 4 counters");
			TEST_ASSERT(pa_sc_block->event_id_max == 171,
				    "PA_SC block max event ID is 171");
			TEST_ASSERT(pa_sc_block->instance_count == 1, "PA_SC block has 1 instance");

			/* Validate SE dimensions - PA_SC is a SE block with 2 dimensions */
			TEST_ASSERT(pa_sc_block->dimensions != NULL,
				    "PA_SC block dimensions exist");
			TEST_ASSERT(pa_sc_block->dimension_count == 2,
				    "PA_SC block has 2 dimensions");

			if (pa_sc_block->dimensions && pa_sc_block->dimension_count >= 2) {
				TEST_ASSERT(pa_sc_block->dimensions[0].dim == HARDWARE_DIM_XCC,
					    "PA_SC dimension 0 is XCC");
				TEST_ASSERT(pa_sc_block->dimensions[0].size == 1,
					    "PA_SC XCC dimension size is 1");
				TEST_ASSERT(pa_sc_block->dimensions[1].dim == HARDWARE_DIM_SE,
					    "PA_SC dimension 1 is SE");
				TEST_ASSERT(pa_sc_block->dimensions[1].size == 4,
					    "PA_SC SE dimension size is 4");
			}

			/* Validate counter register info */
			TEST_ASSERT(pa_sc_block->counter_reg_info != NULL,
				    "PA_SC block counter registers exist");
			if (pa_sc_block->counter_reg_info) {
				counter_reg_info_t *reg0 = &pa_sc_block->counter_reg_info[0];
				counter_reg_info_t *reg1 = &pa_sc_block->counter_reg_info[1];
				counter_reg_info_t *reg2 = &pa_sc_block->counter_reg_info[2];
				counter_reg_info_t *reg3 = &pa_sc_block->counter_reg_info[3];

				/* Counter 0 registers - based on estimated addresses */
				TEST_ASSERT(reg0->select_addr == 14208,
					    "PA_SC counter 0 select register (0x3780)");
				TEST_ASSERT(reg0->control_addr == 0,
					    "PA_SC counter 0 no control register");
				TEST_ASSERT(reg0->register_addr_lo == 12672,
					    "PA_SC counter 0 lo register (0x3180)");
				TEST_ASSERT(reg0->register_addr_hi == 12673,
					    "PA_SC counter 0 hi register (0x3181)");

				/* Counter 1 registers */
				TEST_ASSERT(reg1->select_addr == 14210,
					    "PA_SC counter 1 select register (0x3782)");
				TEST_ASSERT(reg1->control_addr == 0,
					    "PA_SC counter 1 no control register");
				TEST_ASSERT(reg1->register_addr_lo == 12674,
					    "PA_SC counter 1 lo register (0x3182)");
				TEST_ASSERT(reg1->register_addr_hi == 12675,
					    "PA_SC counter 1 hi register (0x3183)");

				/* Counter 2 registers */
				TEST_ASSERT(reg2->select_addr == 14212,
					    "PA_SC counter 2 select register (0x3784)");
				TEST_ASSERT(reg2->control_addr == 0,
					    "PA_SC counter 2 no control register");
				TEST_ASSERT(reg2->register_addr_lo == 12676,
					    "PA_SC counter 2 lo register (0x3184)");
				TEST_ASSERT(reg2->register_addr_hi == 12677,
					    "PA_SC counter 2 hi register (0x3185)");

				/* Counter 3 registers */
				TEST_ASSERT(reg3->select_addr == 14214,
					    "PA_SC counter 3 select register (0x3786)");
				TEST_ASSERT(reg3->control_addr == 0,
					    "PA_SC counter 3 no control register");
				TEST_ASSERT(reg3->register_addr_lo == 12678,
					    "PA_SC counter 3 lo register (0x3186)");
				TEST_ASSERT(reg3->register_addr_hi == 12679,
					    "PA_SC counter 3 hi register (0x3187)");
			}
		}

		free_arch(arch);
	}
}

static void test_gfx12_pa_su_block(void)
{
	printf("\n=== Testing GFX12 PA_SU Block ===\n");
	arch_t *arch = arch_create_by_name("gfx12");
	TEST_ASSERT(arch != NULL, "Architecture created for PA_SU block test");

	if (arch) {
		block_info_t *pa_su_block = arch->block_map.blocks[HW_IP_BLOCK_PA_SU];
		TEST_ASSERT(pa_su_block != NULL, "PA_SU block exists");

		if (pa_su_block) {
			/* Validate basic block properties */
			TEST_ASSERT(strcmp(pa_su_block->name, "PA_SU") == 0,
				    "PA_SU block name is correct");
			TEST_ASSERT(pa_su_block->id == HW_IP_BLOCK_PA_SU,
				    "PA_SU block type correct");
			TEST_ASSERT(pa_su_block->counter_count == 4, "PA_SU block has 4 counters");
			TEST_ASSERT(pa_su_block->event_id_max == 171,
				    "PA_SU block max event ID is 171");
			TEST_ASSERT(pa_su_block->instance_count == 1, "PA_SU block has 1 instance");

			/* Validate SE dimensions - PA_SU is a SE block with 2 dimensions */
			TEST_ASSERT(pa_su_block->dimensions != NULL,
				    "PA_SU block dimensions exist");
			TEST_ASSERT(pa_su_block->dimension_count == 2,
				    "PA_SU block has 2 dimensions");

			if (pa_su_block->dimensions && pa_su_block->dimension_count >= 2) {
				TEST_ASSERT(pa_su_block->dimensions[0].dim == HARDWARE_DIM_XCC,
					    "PA_SU dimension 0 is XCC");
				TEST_ASSERT(pa_su_block->dimensions[0].size == 1,
					    "PA_SU XCC dimension size is 1");
				TEST_ASSERT(pa_su_block->dimensions[1].dim == HARDWARE_DIM_SE,
					    "PA_SU dimension 1 is SE");
				TEST_ASSERT(pa_su_block->dimensions[1].size == 4,
					    "PA_SU SE dimension size is 4");
			}

			/* Validate counter register info */
			TEST_ASSERT(pa_su_block->counter_reg_info != NULL,
				    "PA_SU block counter registers exist");
			if (pa_su_block->counter_reg_info) {
				counter_reg_info_t *reg0 = &pa_su_block->counter_reg_info[0];
				counter_reg_info_t *reg1 = &pa_su_block->counter_reg_info[1];
				counter_reg_info_t *reg2 = &pa_su_block->counter_reg_info[2];
				counter_reg_info_t *reg3 = &pa_su_block->counter_reg_info[3];

				/* Counter 0 registers */
				TEST_ASSERT(reg0->select_addr == 14216,
					    "PA_SU counter 0 select register (0x3788)");
				TEST_ASSERT(reg0->control_addr == 0,
					    "PA_SU counter 0 no control register");
				TEST_ASSERT(reg0->register_addr_lo == 12680,
					    "PA_SU counter 0 lo register (0x3188)");
				TEST_ASSERT(reg0->register_addr_hi == 12681,
					    "PA_SU counter 0 hi register (0x3189)");

				/* Counter 1 registers */
				TEST_ASSERT(reg1->select_addr == 14218,
					    "PA_SU counter 1 select register (0x378a)");
				TEST_ASSERT(reg1->control_addr == 0,
					    "PA_SU counter 1 no control register");
				TEST_ASSERT(reg1->register_addr_lo == 12682,
					    "PA_SU counter 1 lo register (0x318a)");
				TEST_ASSERT(reg1->register_addr_hi == 12683,
					    "PA_SU counter 1 hi register (0x318b)");

				/* Counter 2 registers */
				TEST_ASSERT(reg2->select_addr == 14220,
					    "PA_SU counter 2 select register (0x378c)");
				TEST_ASSERT(reg2->control_addr == 0,
					    "PA_SU counter 2 no control register");
				TEST_ASSERT(reg2->register_addr_lo == 12684,
					    "PA_SU counter 2 lo register (0x318c)");
				TEST_ASSERT(reg2->register_addr_hi == 12685,
					    "PA_SU counter 2 hi register (0x318d)");

				/* Counter 3 registers */
				TEST_ASSERT(reg3->select_addr == 14222,
					    "PA_SU counter 3 select register (0x378e)");
				TEST_ASSERT(reg3->control_addr == 0,
					    "PA_SU counter 3 no control register");
				TEST_ASSERT(reg3->register_addr_lo == 12686,
					    "PA_SU counter 3 lo register (0x318e)");
				TEST_ASSERT(reg3->register_addr_hi == 12687,
					    "PA_SU counter 3 hi register (0x318f)");
			}
		}

		free_arch(arch);
	}
}

static void test_gfx12_gds_block(void)
{
	printf("\n=== Testing GFX12 GDS Block ===\n");
	arch_t *arch = arch_create_by_name("gfx12");
	TEST_ASSERT(arch != NULL, "Architecture created for GDS block test");

	if (arch) {
		block_info_t *gds_block = arch->block_map.blocks[HW_IP_BLOCK_GDS];
		TEST_ASSERT(gds_block != NULL, "GDS block exists");

		if (gds_block) {
			/* Validate basic block properties */
			TEST_ASSERT(strcmp(gds_block->name, "GDS") == 0,
				    "GDS block name is correct");
			TEST_ASSERT(gds_block->id == HW_IP_BLOCK_GDS, "GDS block type correct");
			TEST_ASSERT(gds_block->counter_count == 4, "GDS block has 4 counters");
			TEST_ASSERT(gds_block->event_id_max == 121,
				    "GDS block max event ID is 121");
			TEST_ASSERT(gds_block->instance_count == 1, "GDS block has 1 instance");

			/* Validate XCC dimensions - GDS is a XCC block with 1 dimension */
			TEST_ASSERT(gds_block->dimensions != NULL, "GDS block dimensions exist");
			TEST_ASSERT(gds_block->dimension_count == 1, "GDS block has 1 dimension");

			if (gds_block->dimensions && gds_block->dimension_count >= 1) {
				TEST_ASSERT(gds_block->dimensions[0].dim == HARDWARE_DIM_XCC,
					    "GDS dimension 0 is XCC");
				TEST_ASSERT(gds_block->dimensions[0].size == 1,
					    "GDS XCC dimension size is 1");
			}

			/* Validate counter register info */
			TEST_ASSERT(gds_block->counter_reg_info != NULL,
				    "GDS block counter registers exist");
			if (gds_block->counter_reg_info) {
				counter_reg_info_t *reg0 = &gds_block->counter_reg_info[0];
				counter_reg_info_t *reg1 = &gds_block->counter_reg_info[1];
				counter_reg_info_t *reg2 = &gds_block->counter_reg_info[2];
				counter_reg_info_t *reg3 = &gds_block->counter_reg_info[3];

				/* Counter 0 registers */
				TEST_ASSERT(reg0->select_addr == 14352,
					    "GDS counter 0 select register (0x3810)");
				TEST_ASSERT(reg0->control_addr == 0,
					    "GDS counter 0 no control register");
				TEST_ASSERT(reg0->register_addr_lo == 12816,
					    "GDS counter 0 lo register (0x3210)");
				TEST_ASSERT(reg0->register_addr_hi == 12817,
					    "GDS counter 0 hi register (0x3211)");

				/* Counter 1 registers */
				TEST_ASSERT(reg1->select_addr == 14354,
					    "GDS counter 1 select register (0x3812)");
				TEST_ASSERT(reg1->control_addr == 0,
					    "GDS counter 1 no control register");
				TEST_ASSERT(reg1->register_addr_lo == 12818,
					    "GDS counter 1 lo register (0x3212)");
				TEST_ASSERT(reg1->register_addr_hi == 12819,
					    "GDS counter 1 hi register (0x3213)");

				/* Counter 2 registers */
				TEST_ASSERT(reg2->select_addr == 14356,
					    "GDS counter 2 select register (0x3814)");
				TEST_ASSERT(reg2->control_addr == 0,
					    "GDS counter 2 no control register");
				TEST_ASSERT(reg2->register_addr_lo == 12820,
					    "GDS counter 2 lo register (0x3214)");
				TEST_ASSERT(reg2->register_addr_hi == 12821,
					    "GDS counter 2 hi register (0x3215)");

				/* Counter 3 registers */
				TEST_ASSERT(reg3->select_addr == 14358,
					    "GDS counter 3 select register (0x3816)");
				TEST_ASSERT(reg3->control_addr == 0,
					    "GDS counter 3 no control register");
				TEST_ASSERT(reg3->register_addr_lo == 12822,
					    "GDS counter 3 lo register (0x3216)");
				TEST_ASSERT(reg3->register_addr_hi == 12823,
					    "GDS counter 3 hi register (0x3217)");
			}
		}

		free_arch(arch);
	}
}

/* Test counter allocation simulation */
static void test_counter_allocation(void)
{
	printf("\n=== Testing Counter Allocation ===\n");

	arch_t *arch = arch_create_by_name("gfx12");
	TEST_ASSERT(arch != NULL, "Architecture created for allocation test");

	if (arch) {
		block_info_t *cpc_block = arch->block_map.blocks[HW_IP_BLOCK_CPC];
		if (cpc_block && cpc_block->counter_reg_info) {
			counter_reg_info_t *reg = &cpc_block->counter_reg_info[0];

			/* Simulate counter allocation */
			reg->allocation.state = COUNTER_STATE_ALLOCATED;
			reg->allocation.event_id = 0x10;
			reg->allocation.instance_id = 1;
			reg->allocation.user_id = 1234;
			reg->allocation.allocation_time = 1000;

			TEST_ASSERT(reg->allocation.state == COUNTER_STATE_ALLOCATED,
				    "Counter allocated state");
			TEST_ASSERT(reg->allocation.event_id == 0x10, "Counter event ID set");
			TEST_ASSERT(reg->allocation.instance_id == 1, "Counter instance ID set");
			TEST_ASSERT(reg->allocation.user_id == 1234, "Counter user ID set");
			TEST_ASSERT(reg->allocation.allocation_time == 1000,
				    "Counter allocation time set");

			/* Simulate counter release */
			reg->allocation.state = COUNTER_STATE_FREE;
			reg->allocation.event_id = 0;
			reg->allocation.instance_id = 0;
			reg->allocation.user_id = 0;
			reg->allocation.allocation_time = 0;

			TEST_ASSERT(reg->allocation.state == COUNTER_STATE_FREE,
				    "Counter freed state");
		}

		free_arch(arch);
	}
}

/* Test TA block configuration */
static void test_gfx12_ta_block(void)
{
	printf("\n=== Testing GFX12 TA Block ===\n");

	arch_t *arch = arch_create_by_name("gfx12");
	TEST_ASSERT(arch != NULL, "Architecture created for TA block test");

	if (arch) {
		block_info_t *ta_block = arch->block_map.blocks[HW_IP_BLOCK_TA];
		TEST_ASSERT(ta_block != NULL, "TA block exists");

		if (ta_block) {
			/* Validate basic block properties */
			TEST_ASSERT(strcmp(ta_block->name, "TA") == 0, "TA block name is correct");
			TEST_ASSERT(ta_block->id == HW_IP_BLOCK_TA, "TA block type correct");
			TEST_ASSERT(ta_block->counter_count == 2, "TA block has 2 counters");
			TEST_ASSERT(ta_block->event_id_max == 254, "TA block max event ID is 254");
			TEST_ASSERT(ta_block->instance_count == 2, "TA block has 2 instances");

			/* Validate WGP dimensions - TA is a WGP-level block with 4 dimensions */
			TEST_ASSERT(ta_block->dimensions != NULL, "TA block dimensions exist");
			TEST_ASSERT(ta_block->dimension_count == 4, "TA block has 4 dimensions");
			if (ta_block->dimensions && ta_block->dimension_count >= 4) {
				TEST_ASSERT(ta_block->dimensions[0].dim == HARDWARE_DIM_XCC,
					    "TA dimension 0 is XCC");
				TEST_ASSERT(ta_block->dimensions[0].size == 1,
					    "TA XCC dimension size is 1");
				TEST_ASSERT(ta_block->dimensions[1].dim == HARDWARE_DIM_SE,
					    "TA dimension 1 is SE");
				TEST_ASSERT(ta_block->dimensions[1].size == 4,
					    "TA SE dimension size is 4");
				TEST_ASSERT(ta_block->dimensions[2].dim == HARDWARE_DIM_SA,
					    "TA dimension 2 is SA");
				TEST_ASSERT(ta_block->dimensions[2].size == 2,
					    "TA SA dimension size is 2");
				TEST_ASSERT(ta_block->dimensions[3].dim == HARDWARE_DIM_WGP,
					    "TA dimension 3 is WGP");
				TEST_ASSERT(ta_block->dimensions[3].size == 4,
					    "TA WGP dimension size is 4");
			}

			/* Validate counter register info */
			TEST_ASSERT(ta_block->counter_reg_info != NULL,
				    "TA block counter registers exist");
			if (ta_block->counter_reg_info) {
				counter_reg_info_t *reg0 = &ta_block->counter_reg_info[0];
				counter_reg_info_t *reg1 = &ta_block->counter_reg_info[1];

				/* Counter 0 registers - from aqlprofile reference */
				TEST_ASSERT(reg0->select_addr == 15040,
					    "TA counter 0 select register (0x3ac0)");
				TEST_ASSERT(reg0->control_addr == 0,
					    "TA counter 0 no control register");
				TEST_ASSERT(reg0->register_addr_lo == 12992,
					    "TA counter 0 lo register (0x32c0)");
				TEST_ASSERT(reg0->register_addr_hi == 12993,
					    "TA counter 0 hi register (0x32c1)");

				/* Counter 1 registers */
				TEST_ASSERT(reg1->select_addr == 15042,
					    "TA counter 1 select register (0x3ac2)");
				TEST_ASSERT(reg1->control_addr == 0,
					    "TA counter 1 no control register");
				TEST_ASSERT(reg1->register_addr_lo == 12994,
					    "TA counter 1 lo register (0x32c2)");
				TEST_ASSERT(reg1->register_addr_hi == 12995,
					    "TA counter 1 hi register (0x32c3)");

				/* Test counter allocation simulation */
				reg0->allocation.state = COUNTER_STATE_ALLOCATED;
				reg0->allocation.event_id = 0x42; /* Valid TA event */
				reg0->allocation.instance_id = 0; /* First instance */
				reg0->allocation.user_id = 5678;
				reg0->allocation.allocation_time = 2000;

				TEST_ASSERT(reg0->allocation.state == COUNTER_STATE_ALLOCATED,
					    "TA counter allocated state");
				TEST_ASSERT(reg0->allocation.event_id == 0x42,
					    "TA counter event ID set");
				TEST_ASSERT(reg0->allocation.instance_id == 0,
					    "TA counter instance ID set");
				TEST_ASSERT(reg0->allocation.user_id == 5678,
					    "TA counter user ID set");
				TEST_ASSERT(reg0->allocation.allocation_time == 2000,
					    "TA counter allocation time set");

				/* Simulate counter release */
				reg0->allocation.state = COUNTER_STATE_FREE;
				reg0->allocation.event_id = 0;
				reg0->allocation.instance_id = 0;
				reg0->allocation.user_id = 0;
				reg0->allocation.allocation_time = 0;

				TEST_ASSERT(reg0->allocation.state == COUNTER_STATE_FREE,
					    "TA counter freed state");
			}
		}

		free_arch(arch);
	}
}

/* Test TCP block configuration */
static void test_gfx12_tcp_block(void)
{
	printf("\n=== Testing GFX12 TCP Block ===\n");

	arch_t *arch = arch_create_by_name("gfx12");
	TEST_ASSERT(arch != NULL, "Architecture created for TCP block test");

	if (arch) {
		block_info_t *tcp_block = arch->block_map.blocks[HW_IP_BLOCK_TCP];
		TEST_ASSERT(tcp_block != NULL, "TCP block exists");

		if (tcp_block) {
			/* Validate basic block properties */
			TEST_ASSERT(strcmp(tcp_block->name, "TCP") == 0,
				    "TCP block name is correct");
			TEST_ASSERT(tcp_block->id == HW_IP_BLOCK_TCP, "TCP block type correct");
			TEST_ASSERT(tcp_block->counter_count == 4, "TCP block has 4 counters");
			TEST_ASSERT(tcp_block->event_id_max == 99, "TCP block max event ID is 99");
			TEST_ASSERT(tcp_block->instance_count == 2, "TCP block has 2 instances");

			/* Validate WGP dimensions - TCP is a WGP-level block with 4 dimensions */
			TEST_ASSERT(tcp_block->dimensions != NULL, "TCP block dimensions exist");
			TEST_ASSERT(tcp_block->dimension_count == 4, "TCP block has 4 dimensions");
			if (tcp_block->dimensions && tcp_block->dimension_count >= 4) {
				TEST_ASSERT(tcp_block->dimensions[0].dim == HARDWARE_DIM_XCC,
					    "TCP dimension 0 is XCC");
				TEST_ASSERT(tcp_block->dimensions[0].size == 1,
					    "TCP XCC dimension size is 1");
				TEST_ASSERT(tcp_block->dimensions[1].dim == HARDWARE_DIM_SE,
					    "TCP dimension 1 is SE");
				TEST_ASSERT(tcp_block->dimensions[1].size == 4,
					    "TCP SE dimension size is 4");
				TEST_ASSERT(tcp_block->dimensions[2].dim == HARDWARE_DIM_SA,
					    "TCP dimension 2 is SA");
				TEST_ASSERT(tcp_block->dimensions[2].size == 2,
					    "TCP SA dimension size is 2");
				TEST_ASSERT(tcp_block->dimensions[3].dim == HARDWARE_DIM_WGP,
					    "TCP dimension 3 is WGP");
				TEST_ASSERT(tcp_block->dimensions[3].size == 4,
					    "TCP WGP dimension size is 4");
			}

			/* Validate counter register info */
			TEST_ASSERT(tcp_block->counter_reg_info != NULL,
				    "TCP block counter registers exist");
			if (tcp_block->counter_reg_info) {
				counter_reg_info_t *reg0 = &tcp_block->counter_reg_info[0];
				counter_reg_info_t *reg1 = &tcp_block->counter_reg_info[1];
				counter_reg_info_t *reg2 = &tcp_block->counter_reg_info[2];
				counter_reg_info_t *reg3 = &tcp_block->counter_reg_info[3];

				/* Counter 0 registers - from aqlprofile reference */
				TEST_ASSERT(reg0->select_addr == 15168,
					    "TCP counter 0 select register (0x3b40)");
				TEST_ASSERT(reg0->control_addr == 0,
					    "TCP counter 0 no control register");
				TEST_ASSERT(reg0->register_addr_lo == 13120,
					    "TCP counter 0 lo register (0x3340)");
				TEST_ASSERT(reg0->register_addr_hi == 13121,
					    "TCP counter 0 hi register (0x3341)");

				/* Counter 1 registers */
				TEST_ASSERT(reg1->select_addr == 15170,
					    "TCP counter 1 select register (0x3b42)");
				TEST_ASSERT(reg1->control_addr == 0,
					    "TCP counter 1 no control register");
				TEST_ASSERT(reg1->register_addr_lo == 13122,
					    "TCP counter 1 lo register (0x3342)");
				TEST_ASSERT(reg1->register_addr_hi == 13123,
					    "TCP counter 1 hi register (0x3343)");

				/* Counter 2 registers */
				TEST_ASSERT(reg2->select_addr == 15172,
					    "TCP counter 2 select register (0x3b44)");
				TEST_ASSERT(reg2->control_addr == 0,
					    "TCP counter 2 no control register");
				TEST_ASSERT(reg2->register_addr_lo == 13124,
					    "TCP counter 2 lo register (0x3344)");
				TEST_ASSERT(reg2->register_addr_hi == 13125,
					    "TCP counter 2 hi register (0x3345)");

				/* Counter 3 registers */
				TEST_ASSERT(reg3->select_addr == 15174,
					    "TCP counter 3 select register (0x3b46)");
				TEST_ASSERT(reg3->control_addr == 0,
					    "TCP counter 3 no control register");
				TEST_ASSERT(reg3->register_addr_lo == 13126,
					    "TCP counter 3 lo register (0x3346)");
				TEST_ASSERT(reg3->register_addr_hi == 13127,
					    "TCP counter 3 hi register (0x3347)");

				/* Test counter allocation simulation */
				reg0->allocation.state = COUNTER_STATE_ALLOCATED;
				reg0->allocation.event_id = 0x30; /* Valid TCP event */
				reg0->allocation.instance_id = 1; /* Second instance */
				reg0->allocation.user_id = 7890;
				reg0->allocation.allocation_time = 3000;

				TEST_ASSERT(reg0->allocation.state == COUNTER_STATE_ALLOCATED,
					    "TCP counter allocated state");
				TEST_ASSERT(reg0->allocation.event_id == 0x30,
					    "TCP counter event ID set");
				TEST_ASSERT(reg0->allocation.instance_id == 1,
					    "TCP counter instance ID set");
				TEST_ASSERT(reg0->allocation.user_id == 7890,
					    "TCP counter user ID set");
				TEST_ASSERT(reg0->allocation.allocation_time == 3000,
					    "TCP counter allocation time set");

				/* Simulate counter release */
				reg0->allocation.state = COUNTER_STATE_FREE;
				reg0->allocation.event_id = 0;
				reg0->allocation.instance_id = 0;
				reg0->allocation.user_id = 0;
				reg0->allocation.allocation_time = 0;

				TEST_ASSERT(reg0->allocation.state == COUNTER_STATE_FREE,
					    "TCP counter freed state");
			}
		}

		free_arch(arch);
	}
}

/* Test invalid architecture name */
static void test_invalid_arch(void)
{
	printf("\n=== Testing Invalid Architecture ===\n");

	arch_t *arch = arch_create_by_name("invalid_arch");
	TEST_ASSERT(arch == NULL, "Invalid architecture name returns NULL");

	arch = arch_create_by_name(NULL);
	TEST_ASSERT(arch == NULL, "NULL architecture name returns NULL");
}

/* Test case sensitivity */
static void test_case_sensitivity(void)
{
	printf("\n=== Testing Case Sensitivity ===\n");

	arch_t *arch1 = arch_create_by_name("gfx12");
	arch_t *arch2 = arch_create_by_name("GFX12");

	TEST_ASSERT(arch1 != NULL, "Lowercase gfx12 works");
	TEST_ASSERT(arch2 != NULL, "Uppercase GFX12 works");

	if (arch1) {
		TEST_ASSERT(arch1->type == ARCH_TYPE_GFX12, "Lowercase creates GFX12 type");
		free_arch(arch1);
	}

	if (arch2) {
		TEST_ASSERT(arch2->type == ARCH_TYPE_GFX12, "Uppercase creates GFX12 type");
		free_arch(arch2);
	}
}

/* Main test runner */
int main(void)
{
	printf("=== GFX12 Architecture Test Suite ===\n");

	test_gfx12_arch_creation();
	test_gfx12_cpc_block();
	test_gfx12_sq_block();
	test_gfx12_grbm_block();
	test_gfx12_gl2c_block();
	test_gfx12_spi_block();
	test_gfx12_ta_block();
	test_gfx12_tcp_block();
	test_gfx12_td_block();
	test_gfx12_tcc_block();
	test_gfx12_sx_block();
	test_gfx12_db_block();
	test_gfx12_pa_sc_block();
	test_gfx12_pa_su_block();
	test_gfx12_gds_block();
	test_counter_allocation();
	test_invalid_arch();
	test_case_sensitivity();

	printf("\n=== Test Results ===\n");
	printf("Tests run: %d\n", tests_run);
	printf("Tests passed: %d\n", tests_passed);
	printf("Tests failed: %d\n", tests_run - tests_passed);

	if (tests_passed == tests_run) {
		printf("All tests PASSED!\n");
		return 0;
	} else {
		printf("Some tests FAILED!\n");
		return 1;
	}
}