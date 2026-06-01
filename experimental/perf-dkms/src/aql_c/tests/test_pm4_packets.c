/**
 * @file test_pm4_packets.c
 * @brief Unit tests for PM4 packet builder implementation
 *
 * Tests verify binary compatibility with Rust implementation and proper
 * packet construction for all supported PM4 packet types.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "../pm4_packets.h"
#include "../aql_structures.h"

/* Test register addresses - from GFX12 specification */
#define TEST_GRBM_GFX_INDEX_REG (UCONFIG_SPACE_START + 0x30800)
#define TEST_CP_PERFMON_CNTL_REG (UCONFIG_SPACE_START + 0x36020)
#define TEST_SQ_PERFCOUNTER_CTRL_REG (UCONFIG_SPACE_START + 0x30800)
#define TEST_COMPUTE_PERFCOUNT_ENABLE_REG (PERSISTENT_SPACE_START + 0x1C84)

/* Test utilities */
static void print_buffer_hex(const char *name, const uint32_t *buffer, size_t size_dwords)
{
	printf("%s (%zu DWORDs):\n", name, size_dwords);
	for (size_t i = 0; i < size_dwords; i++) {
		printf("  [%zu]: 0x%08X\n", i, buffer[i]);
	}
	printf("\n");
}

static int compare_buffers(const uint32_t *buf1, const uint32_t *buf2, size_t size_dwords)
{
	for (size_t i = 0; i < size_dwords; i++) {
		if (buf1[i] != buf2[i]) {
			printf("Buffer mismatch at [%zu]: 0x%08X != 0x%08X\n", i, buf1[i], buf2[i]);
			return -1;
		}
	}
	return 0;
}

/* Test basic buffer operations */
static int test_buffer_create_destroy(void)
{
	printf("Testing PM4 buffer create/destroy...\n");

	pm4_buffer_t *buffer = pm4_buffer_create(256);
	assert(buffer != NULL);
	assert(pm4_buffer_get_size(buffer) == 0);
	assert(pm4_buffer_get_data(buffer) != NULL);

	pm4_buffer_destroy(buffer);
	printf("  ✓ Buffer create/destroy passed\n\n");
	return 0;
}

/* Test SetUConfigReg packet generation */
static int test_set_uconfig_reg_packet(void)
{
	printf("Testing SetUConfigReg packet...\n");

	pm4_buffer_t *buffer = pm4_buffer_create(256);

	/* Build SetUConfigReg packet */
	uint32_t reg_offset = TEST_GRBM_GFX_INDEX_REG;
	uint32_t reg_value = 0x80000100; /* Test GRBM broadcast value */

	int result = pm4_append_set_uconfig_reg(buffer, reg_offset, reg_value);
	assert(result == 0);

	/* Verify packet structure */
	uint32_t *data = pm4_buffer_get_data(buffer);
	size_t size = pm4_buffer_get_size(buffer);
	assert(size == 3);

	/* Expected packet structure matching Rust SetUConfigReg */
	uint32_t expected[3];
	expected[0] = PM4_TYPE_3_HEADER(PM4_SET_UCONFIG_REG_OPCODE, 12); /* Header with count=1 */
	expected[1] = (reg_offset - UCONFIG_SPACE_START) & 0xFFFF; /* Register offset */
	expected[2] = reg_value; /* Register value */

	print_buffer_hex("SetUConfigReg actual", data, size);
	print_buffer_hex("SetUConfigReg expected", expected, 3);

	assert(compare_buffers(data, expected, 3) == 0);

	pm4_buffer_destroy(buffer);
	printf("  ✓ SetUConfigReg packet passed\n\n");
	return 0;
}

/* Test EventWrite packet generation */
static int test_event_write_packet(void)
{
	printf("Testing EventWrite packet...\n");

	pm4_buffer_t *buffer = pm4_buffer_create(256);

	/* Build EventWrite packet for CS partial flush */
	int result = pm4_append_event_write(buffer, VGT_EVENT_TYPE_CS_PARTIAL_FLUSH, 4);
	assert(result == 0);

	/* Verify packet structure */
	uint32_t *data = pm4_buffer_get_data(buffer);
	size_t size = pm4_buffer_get_size(buffer);
	assert(size == 2);

	/* Expected packet structure matching Rust EventWrite */
	uint32_t expected[2];
	expected[0] = PM4_TYPE_3_HEADER(PM4_EVENT_WRITE_OPCODE, 8); /* Header with count=2 */

	/* Build barrier event matching Rust BarrierEvent */
	pm4_barrier_event_t event = { 0 };
	event.bits.event_type = VGT_EVENT_TYPE_CS_PARTIAL_FLUSH;
	event.bits.event_index = 4;
	expected[1] = event.raw & 0xFFFF; /* Only lower 16 bits used */

	print_buffer_hex("EventWrite actual", data, size);
	print_buffer_hex("EventWrite expected", expected, 2);

	assert(compare_buffers(data, expected, 2) == 0);

	pm4_buffer_destroy(buffer);
	printf("  ✓ EventWrite packet passed\n\n");
	return 0;
}

/* Test CopyData packet generation */
static int test_copy_data_packet(void)
{
	printf("Testing CopyData packet...\n");

	pm4_buffer_t *buffer = pm4_buffer_create(256);

	/* Build CopyData packet for counter read */
	pm4_copy_data_flags_t flags = { 0 };
	flags.bits.src_sel = 0; /* Non-priv counters */
	flags.bits.dst_sel = 2; /* TC_L2 */
	flags.bits.src_temporal = 3; /* LU */
	flags.bits.dst_temporal = 3; /* LU */
	flags.bits.count_sel = 0; /* 32-bit data */
	flags.bits.wr_confirm = 0; /* No confirmation */

	uint32_t src_reg_lo = 0x8000;
	uint32_t src_reg_hi = 0x0000;
	uint64_t dst_addr = 0x123456789ABCDEF0ULL;

	int result = pm4_append_copy_data(buffer, flags, src_reg_lo, src_reg_hi, dst_addr);
	assert(result == 0);

	/* Verify packet structure */
	uint32_t *data = pm4_buffer_get_data(buffer);
	size_t size = pm4_buffer_get_size(buffer);
	assert(size == 6);

	/* Expected packet structure matching Rust CopyDataPacket */
	uint32_t expected[6];
	expected[0] = PM4_TYPE_3_HEADER(PM4_COPY_DATA_OPCODE, 24); /* Header with count=4 */
	expected[1] = flags.raw;
	expected[2] = src_reg_lo;
	expected[3] = src_reg_hi;
	expected[4] = (uint32_t)(dst_addr & 0xFFFFFFFF);
	expected[5] = (uint32_t)(dst_addr >> 32);

	print_buffer_hex("CopyData actual", data, size);
	print_buffer_hex("CopyData expected", expected, 6);

	assert(compare_buffers(data, expected, 6) == 0);

	pm4_buffer_destroy(buffer);
	printf("  ✓ CopyData packet passed\n\n");
	return 0;
}

/* Test WriteSHRegister packet generation */
static int test_write_sh_reg_packet(void)
{
	printf("Testing WriteSHRegister packet...\n");

	pm4_buffer_t *buffer = pm4_buffer_create(256);

	/* Build WriteSHRegister packet */
	uint32_t reg_offset = TEST_COMPUTE_PERFCOUNT_ENABLE_REG;
	uint32_t reg_value = 0x00000001; /* Enable compute performance counting */
	uint8_t vmid_shift = 0;
	uint8_t index = 0;

	int result = pm4_append_write_sh_reg(buffer, reg_offset, reg_value, vmid_shift, index);
	assert(result == 0);

	/* Verify packet structure */
	uint32_t *data = pm4_buffer_get_data(buffer);
	size_t size = pm4_buffer_get_size(buffer);
	assert(size == 3);

	/* Expected packet structure matching Rust WriteSHRegister */
	uint32_t expected[3];
	expected[0] = PM4_TYPE_3_HEADER(PM4_WRITE_SH_REG_OPCODE, 12); /* Header with count=1 */

	/* Build word1 with bitfield layout */
	uint16_t offset = (uint16_t)((reg_offset - PERSISTENT_SPACE_START) & 0xFFFF);
	expected[1] = (offset & 0xFFFF) | /* bits 0-15: reg_offset */
		      ((vmid_shift & 0x1F) << 23) | /* bits 23-27: vmid_shift */
		      ((index & 0xF) << 28); /* bits 28-31: index */
	expected[2] = reg_value;

	print_buffer_hex("WriteSHRegister actual", data, size);
	print_buffer_hex("WriteSHRegister expected", expected, 3);

	assert(compare_buffers(data, expected, 3) == 0);

	pm4_buffer_destroy(buffer);
	printf("  ✓ WriteSHRegister packet passed\n\n");
	return 0;
}

/* Test AcquireMem (FlushCache) packet generation */
static int test_acquire_mem_packet(void)
{
	printf("Testing AcquireMem packet...\n");

	pm4_buffer_t *buffer = pm4_buffer_create(256);

	/* Build AcquireMem packet for cache coherency */
	uint64_t base_addr = 0x123456000ULL; /* 256-byte aligned */
	uint64_t size = 1024; /* 1KB */
	uint32_t gcr_cntl = 0x18000; /* GCR control value */

	int result = pm4_append_acquire_mem(buffer, base_addr, size, gcr_cntl);
	assert(result == 0);

	/* Verify packet structure */
	uint32_t *data = pm4_buffer_get_data(buffer);
	size_t packet_size = pm4_buffer_get_size(buffer);
	assert(packet_size == 8);

	/* Calculate expected coherency parameters */
	uint32_t coher_size_lo, coher_size_hi, coher_base_lo, coher_base_hi;
	pm4_calculate_cache_coher_params(base_addr, size, &coher_size_lo, &coher_size_hi,
					 &coher_base_lo, &coher_base_hi);

	/* Expected packet structure matching Rust FlushCachePacket */
	uint32_t expected[8];
	expected[0] = PM4_TYPE_3_HEADER(PM4_ACQUIRE_MEM_OPCODE, 32); /* Header with count=6 */
	expected[1] = 0; /* Reserved */
	expected[2] = coher_size_lo;
	expected[3] = coher_size_hi & 0xFF; /* Only 8 bits */
	expected[4] = coher_base_lo;
	expected[5] = coher_base_hi & 0xFFFFFF; /* Only 24 bits */
	expected[6] = 0x10 & 0xFFFF; /* poll_interval */
	expected[7] = gcr_cntl & 0x7FFFF; /* 19 bits */

	print_buffer_hex("AcquireMem actual", data, packet_size);
	print_buffer_hex("AcquireMem expected", expected, 8);

	assert(compare_buffers(data, expected, 8) == 0);

	pm4_buffer_destroy(buffer);
	printf("  ✓ AcquireMem packet passed\n\n");
	return 0;
}

/* Test GRBM broadcast helper */
static int test_grbm_broadcast(void)
{
	printf("Testing GRBM broadcast helper...\n");

	pm4_buffer_t *buffer = pm4_buffer_create(256);

	int result = pm4_grbm_broadcast(buffer, TEST_GRBM_GFX_INDEX_REG);
	assert(result == 0);

	/* Verify it created a SetUConfigReg packet with broadcast flags */
	uint32_t *data = pm4_buffer_get_data(buffer);
	size_t size = pm4_buffer_get_size(buffer);
	assert(size == 3);

	/* Expected GRBM broadcast value */
	pm4_grbm_gfx_index_t expected_index = { 0 };
	expected_index.bits.instance_index = 0;
	expected_index.bits.sa_index = 0;
	expected_index.bits.se_index = 0;
	expected_index.bits.sa_broadcast_writes = 1;
	expected_index.bits.instance_broadcast_writes = 1;
	expected_index.bits.se_broadcast_writes = 1;

	assert(data[2] == expected_index.raw);

	pm4_buffer_destroy(buffer);
	printf("  ✓ GRBM broadcast helper passed\n\n");
	return 0;
}

/* Test pm4_op_t conversion functions */
static int test_pm4_op_conversion(void)
{
	printf("Testing pm4_op_t conversions...\n");

	/* Test SetUConfigReg conversion */
	uint32_t test_packet[3] = { PM4_TYPE_3_HEADER(PM4_SET_UCONFIG_REG_OPCODE, 1), 0x1234,
				    0x56789ABC };

	pm4_op_t op = pm4_op_from_buffer(test_packet, 3);
	assert(op.type == PM4_OP_SET_UCONFIG_REG);
	assert(op.size_dwords == 3);
	assert(op.packet.set_uconfig_reg.header == test_packet[0]);
	assert(op.packet.set_uconfig_reg.reg_offset == 0x1234);
	assert(op.packet.set_uconfig_reg.reg_value == 0x56789ABC);

	/* Test round-trip conversion */
	uint32_t output_buffer[3];
	int written = pm4_op_to_buffer(&op, output_buffer, 3);
	assert(written == 3);
	assert(compare_buffers(test_packet, output_buffer, 3) == 0);

	printf("  ✓ pm4_op_t conversions passed\n\n");
	return 0;
}

/* Test complex command sequence matching Rust counter_start implementation */
static int test_counter_start_sequence(void)
{
	printf("Testing counter start sequence (matching Rust counter_start)...\n");

	pm4_buffer_t *buffer = pm4_buffer_create(256);

	/* 1. GRBM broadcast */
	int result = pm4_grbm_broadcast(buffer, TEST_GRBM_GFX_INDEX_REG);
	assert(result == 0);

	/* 2. Select performance counter (SQ_WAVES event_id=4) */
	uint32_t select_reg = 0x8000; /* SQ_PERFCOUNTER0_SELECT */
	uint32_t select_value = 4; /* perf_sel = 4 for SQ_WAVES */
	result = pm4_append_set_uconfig_reg(buffer, UCONFIG_SPACE_START + select_reg, select_value);
	assert(result == 0);

	/* 3. Enable SQ counter control */
	uint32_t control_reg = 0x8010; /* SQ_PERFCOUNTER_CTRL */
	uint32_t control_value = 0x0000005F; /* Enable PS, GS, HS, CS */
	result = pm4_append_set_uconfig_reg(buffer, UCONFIG_SPACE_START + control_reg,
					    control_value);
	assert(result == 0);

	/* 4. Enable compute performance counting */
	result = pm4_append_write_sh_reg(buffer, TEST_COMPUTE_PERFCOUNT_ENABLE_REG, 1, 0, 0);
	assert(result == 0);

	/* 5. Enable CP perfmon control */
	result = pm4_append_set_uconfig_reg(buffer, TEST_CP_PERFMON_CNTL_REG, 0x00000000);
	assert(result == 0);

	/* 6. Start perfmon */
	result = pm4_append_set_uconfig_reg(buffer, TEST_CP_PERFMON_CNTL_REG, 0x00000001);
	assert(result == 0);

	/* Verify we have expected number of packets */
	size_t total_size = pm4_buffer_get_size(buffer);
	size_t expected_size = 3 + 3 + 3 + 3 + 3 + 3; /* 6 packets, each 3 DWORDs */
	assert(total_size == expected_size);

	print_buffer_hex("Counter start sequence", pm4_buffer_get_data(buffer), total_size);

	pm4_buffer_destroy(buffer);
	printf("  ✓ Counter start sequence passed\n\n");
	return 0;
}

/* Test GFX9 ACQUIRE_MEM packet (7-DWORD format with CP_COHER_CNTL) */
static int test_acquire_mem_gfx9_packet(void)
{
	printf("Testing AcquireMem GFX9 packet...\n");

	pm4_buffer_t *buffer = pm4_buffer_create(256);

	uint64_t base_addr = 0x123456000ULL;
	uint64_t size = 1024;
	uint32_t cp_coher_cntl = 0x28C40000;

	int result = pm4_append_acquire_mem_gfx9(buffer, base_addr, size, cp_coher_cntl);
	assert(result == 0);

	uint32_t *data = pm4_buffer_get_data(buffer);
	size_t packet_size = pm4_buffer_get_size(buffer);

	/* GFX9 ACQUIRE_MEM is 7 DWORDs, not 8 */
	assert(packet_size == 7);

	/* Verify header */
	uint32_t expected_header = PM4_TYPE_3_HEADER(PM4_ACQUIRE_MEM_OPCODE, 28);
	assert(data[0] == expected_header);

	/* DWord[1] is CP_COHER_CNTL (not reserved like GFX12) */
	assert(data[1] == 0x28C40000);

	/* Verify coherency params */
	uint32_t coher_size_lo, coher_size_hi, coher_base_lo, coher_base_hi;
	pm4_calculate_cache_coher_params(base_addr, size, &coher_size_lo, &coher_size_hi,
					 &coher_base_lo, &coher_base_hi);
	assert(data[2] == coher_size_lo);
	assert(data[3] == (coher_size_hi & 0xFF));
	assert(data[4] == coher_base_lo);
	assert(data[5] == (coher_base_hi & 0xFFFFFF));

	/* No DWord[7] — packet ends at index 6 */
	assert(data[6] == (0x10 & 0xFFFF));

	/* Verify validation accepts 7-DWORD ACQUIRE_MEM */
	assert(pm4_validate_packet(data, 7) == 0);

	pm4_buffer_destroy(buffer);
	printf("  ✓ AcquireMem GFX9 packet passed\n\n");
	return 0;
}

/* Test GFX9 GRBM index (direct instance, no WGP shift) */
static int test_grbm_index_gfx9(void)
{
	printf("Testing GRBM index GFX9...\n");

	pm4_buffer_t *buffer_gfx9 = pm4_buffer_create(256);
	pm4_buffer_t *buffer_gfx12 = pm4_buffer_create(256);

	/* GFX9: instance=5 stays 5 (direct) */
	int result = pm4_set_grbm_index_gfx9(buffer_gfx9, TEST_GRBM_GFX_INDEX_REG, 5, 1, 2);
	assert(result == 0);

	/* GFX12: wg_index=5 becomes 5<<2=20 */
	result = pm4_set_grbm_index(buffer_gfx12, TEST_GRBM_GFX_INDEX_REG, 5, 1, 2);
	assert(result == 0);

	uint32_t *data_gfx9 = pm4_buffer_get_data(buffer_gfx9);
	uint32_t *data_gfx12 = pm4_buffer_get_data(buffer_gfx12);

	/* Extract GRBM values (DWord[2] of SetUConfigReg) */
	uint32_t grbm_gfx9 = data_gfx9[2];
	uint32_t grbm_gfx12 = data_gfx12[2];

	/* GFX9: instance_index = 5 (bits 0-7) */
	assert((grbm_gfx9 & 0xFF) == 5);

	/* GFX12: instance_index = 20 (5<<2, bits 0-6) */
	assert((grbm_gfx12 & 0x7F) == 20);

	/* They should differ */
	assert(grbm_gfx9 != grbm_gfx12);

	/* GFX9: sh_index=1 in bits 8-15 */
	assert(((grbm_gfx9 >> 8) & 0xFF) == 1);

	/* GFX9: se_index=2 in bits 16-23 */
	assert(((grbm_gfx9 >> 16) & 0xFF) == 2);

	pm4_buffer_destroy(buffer_gfx9);
	pm4_buffer_destroy(buffer_gfx12);
	printf("  ✓ GRBM index GFX9 passed\n\n");
	return 0;
}

/* Test GFX9 GRBM SE+SH index with instance broadcast */
static int test_grbm_se_sh_index_gfx9(void)
{
	printf("Testing GRBM SE+SH index GFX9...\n");

	pm4_buffer_t *buffer = pm4_buffer_create(256);

	int result = pm4_set_grbm_se_sh_index_gfx9(buffer, TEST_GRBM_GFX_INDEX_REG, 1, 3);
	assert(result == 0);

	uint32_t *data = pm4_buffer_get_data(buffer);
	uint32_t grbm_val = data[2];

	/* instance_index = 0 */
	assert((grbm_val & 0xFF) == 0);

	/* sh_index = 1 */
	assert(((grbm_val >> 8) & 0xFF) == 1);

	/* se_index = 3 */
	assert(((grbm_val >> 16) & 0xFF) == 3);

	/* instance_broadcast_writes = 1 (bit 30) */
	assert((grbm_val & (1u << 30)) != 0);

	/* sh_broadcast_writes = 0 (bit 29) */
	assert((grbm_val & (1u << 29)) == 0);

	/* se_broadcast_writes = 0 (bit 31) */
	assert((grbm_val & (1u << 31)) == 0);

	pm4_buffer_destroy(buffer);
	printf("  ✓ GRBM SE+SH index GFX9 passed\n\n");
	return 0;
}

/* Test that existing pm4_grbm_broadcast() produces 0xE0000000 (works for both GFX9 and GFX12) */
static int test_grbm_broadcast_works_for_gfx9(void)
{
	printf("Testing GRBM broadcast works for GFX9...\n");

	pm4_buffer_t *buffer = pm4_buffer_create(256);

	int result = pm4_grbm_broadcast(buffer, TEST_GRBM_GFX_INDEX_REG);
	assert(result == 0);

	uint32_t *data = pm4_buffer_get_data(buffer);
	uint32_t grbm_val = data[2];

	/* Broadcast bits 29-31 are at identical positions on GFX9 and GFX12 */
	/* Expected: sa/sh_broadcast(bit29) | instance_broadcast(bit30) | se_broadcast(bit31) */
	assert(grbm_val == 0xE0000000);

	pm4_buffer_destroy(buffer);
	printf("  ✓ GRBM broadcast works for GFX9 passed\n\n");
	return 0;
}

/* Main test runner */
int main(void)
{
	printf("=== PM4 Packet Builder Tests ===\n\n");

	/* Run all tests */
	assert(test_buffer_create_destroy() == 0);
	assert(test_set_uconfig_reg_packet() == 0);
	assert(test_event_write_packet() == 0);
	assert(test_copy_data_packet() == 0);
	assert(test_write_sh_reg_packet() == 0);
	assert(test_acquire_mem_packet() == 0);
	assert(test_grbm_broadcast() == 0);
	assert(test_pm4_op_conversion() == 0);
	assert(test_counter_start_sequence() == 0);

	/* GFX9-specific tests */
	assert(test_acquire_mem_gfx9_packet() == 0);
	assert(test_grbm_index_gfx9() == 0);
	assert(test_grbm_se_sh_index_gfx9() == 0);
	assert(test_grbm_broadcast_works_for_gfx9() == 0);

	printf("=== All tests passed! ===\n");
	return 0;
}