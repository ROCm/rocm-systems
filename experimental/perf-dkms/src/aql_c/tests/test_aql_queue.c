/**
 * @file test_aql_queue.c
 * @brief Unit tests for AQL queue ring buffer, IB pool, and PM4 IB packets
 *
 * Tests verify:
 * - Ring buffer initialization, write, wrapping, and slot tracking
 * - IB pool allocation, alignment, and wrapping
 * - AQL PM4 IB packet format (byte-by-byte verification)
 * - Packet size static assertion (sizeof == 64)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "../aql_queue.h"

/* ========================================================================
 * Test helpers
 * ======================================================================== */

static void print_hex(const char *label, const void *data, size_t len)
{
	const uint8_t *p = (const uint8_t *)data;
	printf("  %s (%zu bytes):", label, len);
	for (size_t i = 0; i < len; i++) {
		if (i % 16 == 0)
			printf("\n    ");
		printf("%02x ", p[i]);
	}
	printf("\n");
}

/* ========================================================================
 * Ring buffer tests
 * ======================================================================== */

static int test_ring_init(void)
{
	printf("Testing ring init...\n");

	uint8_t buf[4096];
	aql_ring_t ring;

	/* Valid init */
	assert(aql_ring_init(&ring, buf, 0x1000, 4096) == 0);
	assert(ring.base == buf);
	assert(ring.gpu_addr == 0x1000);
	assert(ring.size == 4096);
	assert(ring.wptr == 0);
	assert(ring.rptr == 0);

	/* Invalid: not power of 2 */
	assert(aql_ring_init(&ring, buf, 0x1000, 1000) == -1);

	/* Invalid: too small */
	assert(aql_ring_init(&ring, buf, 0x1000, 32) == -1);

	/* Invalid: NULL base */
	assert(aql_ring_init(&ring, NULL, 0x1000, 4096) == -1);

	/* Invalid: zero size */
	assert(aql_ring_init(&ring, buf, 0x1000, 0) == -1);

	/* Valid: minimum size (one packet) */
	assert(aql_ring_init(&ring, buf, 0x1000, 64) == 0);

	printf("  PASS\n");
	return 0;
}

static int test_ring_write_single(void)
{
	printf("Testing ring write single packet...\n");

	uint8_t buf[4096];
	memset(buf, 0, sizeof(buf));

	aql_ring_t ring;
	assert(aql_ring_init(&ring, buf, 0x1000, 4096) == 0);

	/* Build a test packet */
	aql_pm4_ib_packet_t pkt;
	assert(aql_build_pm4_ib_packet(&pkt, 0x2000, 16, 0x3000) == 0);

	/* Write it */
	assert(aql_ring_write_packet(&ring, &pkt) == 0);

	/* Verify wptr advanced by 64 */
	assert(ring.wptr == 64);

	/* Verify packet is in the buffer at offset 0 */
	aql_pm4_ib_packet_t *written = (aql_pm4_ib_packet_t *)buf;
	assert(written->header == pkt.header);
	assert(written->pm4_ib_format == 1);
	assert(written->pm4_ib_command[0] == pkt.pm4_ib_command[0]);
	assert(written->pm4_ib_command[1] == pkt.pm4_ib_command[1]);
	assert(written->pm4_ib_command[2] == pkt.pm4_ib_command[2]);
	assert(written->pm4_ib_command[3] == pkt.pm4_ib_command[3]);
	assert(written->completion_signal == 0x3000);

	/* Slot index should now point to slot 1 */
	assert(aql_ring_slot_index(&ring) == 1);

	printf("  PASS\n");
	return 0;
}

static int test_ring_write_wrap(void)
{
	printf("Testing ring write with wrapping...\n");

	/* Ring holds exactly 4 packets (256 bytes) */
	uint8_t buf[256];
	memset(buf, 0, sizeof(buf));

	aql_ring_t ring;
	assert(aql_ring_init(&ring, buf, 0x1000, 256) == 0);

	aql_pm4_ib_packet_t pkt;
	assert(aql_build_pm4_ib_packet(&pkt, 0x2000, 8, 0x3000) == 0);

	/* Fill all 4 slots */
	for (int i = 0; i < 4; i++) {
		assert(aql_ring_write_packet(&ring, &pkt) == 0);
	}
	assert(ring.wptr == 256);
	assert(aql_ring_available_slots(&ring) == 0);

	/* 5th write should fail (ring full) */
	assert(aql_ring_write_packet(&ring, &pkt) == -1);

	/* Simulate GPU consuming 2 packets */
	ring.rptr = 128;

	/* Now 2 slots available */
	assert(aql_ring_available_slots(&ring) == 2);

	/* Write should succeed, wrapping around */
	assert(aql_ring_write_packet(&ring, &pkt) == 0);
	assert(ring.wptr == 320);

	/* Verify wrapping: wptr=320, ring_size=256, slot_offset = 320 & 255 = 64 */
	/* Slot index = 64/64 = 1 */
	assert(aql_ring_slot_index(&ring) == 1);

	/* Verify the packet was written at offset 0 (wrapped) */
	aql_pm4_ib_packet_t *wrapped = (aql_pm4_ib_packet_t *)(buf + 0);
	assert(wrapped->pm4_ib_format == 1);

	printf("  PASS\n");
	return 0;
}

static int test_ring_available_slots(void)
{
	printf("Testing ring available slots...\n");

	uint8_t buf[1024];
	aql_ring_t ring;
	assert(aql_ring_init(&ring, buf, 0x1000, 1024) == 0);

	/* Initially all slots available: 1024/64 = 16 */
	assert(aql_ring_available_slots(&ring) == 16);

	/* Write 5 packets */
	aql_pm4_ib_packet_t pkt;
	aql_build_pm4_ib_packet(&pkt, 0x2000, 4, 0x3000);
	for (int i = 0; i < 5; i++)
		aql_ring_write_packet(&ring, &pkt);

	assert(aql_ring_available_slots(&ring) == 11);

	/* GPU consumes 3 */
	ring.rptr = 3 * 64;
	assert(aql_ring_available_slots(&ring) == 14);

	printf("  PASS\n");
	return 0;
}

/* ========================================================================
 * IB pool tests
 * ======================================================================== */

static int test_ib_pool_init(void)
{
	printf("Testing IB pool init...\n");

	uint8_t buf[4096];
	aql_ib_pool_t pool;

	assert(aql_ib_pool_init(&pool, buf, 0x5000, 4096) == 0);
	assert(pool.base == buf);
	assert(pool.gpu_addr == 0x5000);
	assert(pool.size == 4096);
	assert(pool.offset == 0);

	/* Invalid params */
	assert(aql_ib_pool_init(&pool, NULL, 0x5000, 4096) == -1);
	assert(aql_ib_pool_init(&pool, buf, 0x5000, 0) == -1);

	printf("  PASS\n");
	return 0;
}

static int test_ib_pool_alloc(void)
{
	printf("Testing IB pool allocation...\n");

	uint8_t buf[4096];
	aql_ib_pool_t pool;
	void *cpu;
	uint64_t gpu;

	assert(aql_ib_pool_init(&pool, buf, 0x5000, 4096) == 0);

	/* First allocation: 100 bytes -> aligned to 256 */
	assert(aql_ib_pool_alloc(&pool, 100, &cpu, &gpu) == 0);
	assert(cpu == buf);
	assert(gpu == 0x5000);
	assert(pool.offset == 256); /* aligned up */

	/* Second allocation: 200 bytes -> aligned to 256 */
	assert(aql_ib_pool_alloc(&pool, 200, &cpu, &gpu) == 0);
	assert(cpu == buf + 256);
	assert(gpu == 0x5000 + 256);
	assert(pool.offset == 512);

	/* Exact alignment: 256 bytes -> stays 256 */
	assert(aql_ib_pool_alloc(&pool, 256, &cpu, &gpu) == 0);
	assert(cpu == buf + 512);
	assert(gpu == 0x5000 + 512);
	assert(pool.offset == 768);

	printf("  PASS\n");
	return 0;
}

static int test_ib_pool_wrap(void)
{
	printf("Testing IB pool wrap...\n");

	uint8_t buf[1024];
	aql_ib_pool_t pool;
	void *cpu;
	uint64_t gpu;

	assert(aql_ib_pool_init(&pool, buf, 0x5000, 1024) == 0);

	/* Allocate 3 x 256 = 768 bytes */
	for (int i = 0; i < 3; i++) {
		assert(aql_ib_pool_alloc(&pool, 256, &cpu, &gpu) == 0);
	}
	assert(pool.offset == 768);

	/* Next 256-byte allocation would need 1024, but only 256 remain -> wrap */
	assert(aql_ib_pool_alloc(&pool, 300, &cpu, &gpu) == 0);
	assert(cpu == buf);          /* wrapped to beginning */
	assert(gpu == 0x5000);
	assert(pool.offset == 512);  /* 300 aligned to 512 */

	/* Allocation too large for pool (> 1024) should fail */
	assert(aql_ib_pool_alloc(&pool, 2000, &cpu, &gpu) == -1);

	printf("  PASS\n");
	return 0;
}

static int test_ib_pool_reset(void)
{
	printf("Testing IB pool reset...\n");

	uint8_t buf[1024];
	aql_ib_pool_t pool;
	void *cpu;
	uint64_t gpu;

	assert(aql_ib_pool_init(&pool, buf, 0x5000, 1024) == 0);

	/* Allocate some space */
	aql_ib_pool_alloc(&pool, 256, &cpu, &gpu);
	aql_ib_pool_alloc(&pool, 256, &cpu, &gpu);
	assert(pool.offset == 512);

	/* Reset */
	aql_ib_pool_reset(&pool);
	assert(pool.offset == 0);

	/* Allocate again from beginning */
	assert(aql_ib_pool_alloc(&pool, 100, &cpu, &gpu) == 0);
	assert(cpu == buf);
	assert(gpu == 0x5000);

	printf("  PASS\n");
	return 0;
}

/* ========================================================================
 * AQL PM4 IB packet tests
 * ======================================================================== */

static int test_build_pm4_ib_packet(void)
{
	printf("Testing build PM4 IB packet...\n");

	aql_pm4_ib_packet_t pkt;
	uint64_t ib_addr = 0x0000000100002000ULL;
	uint32_t ib_size_dw = 32;
	uint64_t signal_addr = 0x0000000100003000ULL;

	assert(aql_build_pm4_ib_packet(&pkt, ib_addr, ib_size_dw, signal_addr) == 0);

	print_hex("AQL PM4 IB packet", &pkt, 64);

	/* Verify header: VENDOR_SPECIFIC (0) | BARRIER (bit 8) = 0x0100 */
	uint16_t expected_header = (HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE_SHIFT)
				 | (1 << HSA_PACKET_HEADER_BARRIER_SHIFT);
	assert(pkt.header == expected_header);
	printf("  header = 0x%04x (expected 0x%04x) OK\n", pkt.header, expected_header);

	/* Verify pm4_ib_format = 1 */
	assert(pkt.pm4_ib_format == 1);
	printf("  pm4_ib_format = %u OK\n", pkt.pm4_ib_format);

	/* Verify INDIRECT_BUFFER PM4 header */
	assert(pkt.pm4_ib_command[0] == PM4_IB_HEADER);
	printf("  pm4_ib_command[0] = 0x%08x (IB header) OK\n", pkt.pm4_ib_command[0]);

	/* Verify IB address (DWORD-aligned) */
	uint32_t expected_lo = (uint32_t)(ib_addr & 0xFFFFFFFC);
	uint32_t expected_hi = (uint32_t)(ib_addr >> 32);
	assert(pkt.pm4_ib_command[1] == expected_lo);
	assert(pkt.pm4_ib_command[2] == expected_hi);
	printf("  pm4_ib_command[1] = 0x%08x (addr lo) OK\n", pkt.pm4_ib_command[1]);
	printf("  pm4_ib_command[2] = 0x%08x (addr hi) OK\n", pkt.pm4_ib_command[2]);

	/* Verify IB size with VALID bit */
	uint32_t expected_size = ib_size_dw | IB_VALID_BIT;
	assert(pkt.pm4_ib_command[3] == expected_size);
	printf("  pm4_ib_command[3] = 0x%08x (size|valid) OK\n", pkt.pm4_ib_command[3]);

	/* Verify dw_count_remain = 10 */
	assert(pkt.dw_count_remain == 10);
	printf("  dw_count_remain = %u OK\n", pkt.dw_count_remain);

	/* Verify reserved fields are all zero */
	for (int i = 0; i < 8; i++) {
		assert(pkt.reserved[i] == 0);
	}
	printf("  reserved[0..7] = 0 OK\n");

	/* Verify completion signal */
	assert(pkt.completion_signal == signal_addr);
	printf("  completion_signal = 0x%016llx OK\n",
	       (unsigned long long)pkt.completion_signal);

	printf("  PASS\n");
	return 0;
}

static int test_packet_size_assert(void)
{
	printf("Testing packet size...\n");

	assert(sizeof(aql_pm4_ib_packet_t) == 64);
	printf("  sizeof(aql_pm4_ib_packet_t) = %zu == 64 OK\n",
	       sizeof(aql_pm4_ib_packet_t));

	printf("  PASS\n");
	return 0;
}

static int test_build_packet_invalid_params(void)
{
	printf("Testing build packet with invalid params...\n");

	aql_pm4_ib_packet_t pkt;

	/* NULL packet */
	assert(aql_build_pm4_ib_packet(NULL, 0x2000, 16, 0x3000) == -1);

	/* Zero size */
	assert(aql_build_pm4_ib_packet(&pkt, 0x2000, 0, 0x3000) == -1);

	/* Valid with zero signal (allowed) */
	assert(aql_build_pm4_ib_packet(&pkt, 0x2000, 16, 0) == 0);
	assert(pkt.completion_signal == 0);

	/* Valid with zero IB addr (allowed, though unusual) */
	assert(aql_build_pm4_ib_packet(&pkt, 0, 16, 0x3000) == 0);
	assert(pkt.pm4_ib_command[1] == 0);
	assert(pkt.pm4_ib_command[2] == 0);

	printf("  PASS\n");
	return 0;
}

static int test_ib_addr_alignment(void)
{
	printf("Testing IB address DWORD alignment in packet...\n");

	aql_pm4_ib_packet_t pkt;

	/* Unaligned address: 0x2001 -> should mask to 0x2000 */
	assert(aql_build_pm4_ib_packet(&pkt, 0x2001, 16, 0x3000) == 0);
	assert(pkt.pm4_ib_command[1] == 0x2000); /* masked to DWORD boundary */

	/* Already aligned */
	assert(aql_build_pm4_ib_packet(&pkt, 0x2004, 16, 0x3000) == 0);
	assert(pkt.pm4_ib_command[1] == 0x2004);

	printf("  PASS\n");
	return 0;
}

/* ========================================================================
 * Integration test: full write path
 * ======================================================================== */

static int test_full_write_path(void)
{
	printf("Testing full write path (pool alloc + packet build + ring write)...\n");

	uint8_t ring_buf[1024];
	uint8_t ib_buf[4096];
	uint8_t signal_buf[8];
	memset(ring_buf, 0, sizeof(ring_buf));
	memset(ib_buf, 0, sizeof(ib_buf));
	memset(signal_buf, 0, sizeof(signal_buf));

	aql_ring_t ring;
	aql_ib_pool_t pool;

	assert(aql_ring_init(&ring, ring_buf, 0x10000, 1024) == 0);
	assert(aql_ib_pool_init(&pool, ib_buf, 0x20000, 4096) == 0);

	uint64_t signal_gpu_addr = 0x30000;

	/* Simulate PM4 command data (16 DWORDs) */
	uint32_t pm4_data[16];
	for (int i = 0; i < 16; i++)
		pm4_data[i] = 0xDEAD0000 | i;

	/* Step 1: Allocate IB space */
	void *ib_cpu;
	uint64_t ib_gpu;
	assert(aql_ib_pool_alloc(&pool, 16 * 4, &ib_cpu, &ib_gpu) == 0);

	/* Step 2: Copy PM4 commands to IB */
	memcpy(ib_cpu, pm4_data, 16 * 4);

	/* Step 3: Build AQL packet */
	aql_pm4_ib_packet_t pkt;
	assert(aql_build_pm4_ib_packet(&pkt, ib_gpu, 16, signal_gpu_addr) == 0);

	/* Step 4: Write to ring */
	assert(aql_ring_write_packet(&ring, &pkt) == 0);

	/* Verify: ring wptr advanced */
	assert(ring.wptr == 64);

	/* Verify: packet in ring points to IB */
	aql_pm4_ib_packet_t *ring_pkt = (aql_pm4_ib_packet_t *)ring_buf;
	assert(ring_pkt->pm4_ib_command[1] == (uint32_t)(ib_gpu & 0xFFFFFFFC));
	assert(ring_pkt->pm4_ib_command[2] == (uint32_t)(ib_gpu >> 32));

	/* Verify: IB contains our PM4 data */
	uint32_t *ib_data = (uint32_t *)ib_cpu;
	for (int i = 0; i < 16; i++) {
		assert(ib_data[i] == pm4_data[i]);
	}

	/* Verify: signal address in packet */
	assert(ring_pkt->completion_signal == signal_gpu_addr);

	printf("  PASS\n");
	return 0;
}

/* ========================================================================
 * Main test runner
 * ======================================================================== */

int main(void)
{
	printf("=== AQL Queue Tests ===\n\n");

	/* Ring buffer tests */
	assert(test_ring_init() == 0);
	assert(test_ring_write_single() == 0);
	assert(test_ring_write_wrap() == 0);
	assert(test_ring_available_slots() == 0);

	/* IB pool tests */
	assert(test_ib_pool_init() == 0);
	assert(test_ib_pool_alloc() == 0);
	assert(test_ib_pool_wrap() == 0);
	assert(test_ib_pool_reset() == 0);

	/* AQL PM4 IB packet tests */
	assert(test_build_pm4_ib_packet() == 0);
	assert(test_packet_size_assert() == 0);
	assert(test_build_packet_invalid_params() == 0);
	assert(test_ib_addr_alignment() == 0);

	/* Integration */
	assert(test_full_write_path() == 0);

	printf("\n=== All AQL queue tests passed! ===\n");
	return 0;
}
