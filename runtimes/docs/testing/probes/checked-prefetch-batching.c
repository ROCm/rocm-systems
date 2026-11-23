// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/* The shell driver includes the production helpers from the patched source.
 * These mocks check coverage, submission bounds, and fence ownership; they do
 * not establish SDMA execution or migration correctness on hardware.
 */
#include <assert.h>
#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef uint64_t u64;
typedef uint32_t u32;
#define PAGE_SIZE 4096ULL
#define PAGE_MASK (~(PAGE_SIZE - 1))
#define U64_MAX UINT64_MAX
#define MIGRATE_PFN_VALID 1UL
#define MIGRATE_PFN_MIGRATE 2UL
#define TTM_PL_VRAM 2
#define AMDGPU_KERNEL_JOB_ID_TTM_CLEAR_BUFFER 1
#define min_t(type, a, b) ((type)(a) < (type)(b) ? (type)(a) : (type)(b))

struct page { int unused; };
struct dma_fence { bool complete; };
struct amdgpu_ttm_buffer_entity { bool lock; };
struct amdgpu_device {
	struct {
		bool buffer_funcs_enabled;
		unsigned int num_clear_entities;
		struct amdgpu_ttm_buffer_entity clear_entities[1];
	} mman;
};
struct migrate_vma {
	unsigned long npages;
	unsigned long *src;
	unsigned long *dst;
};

static struct {
	u64 address[32];
	u32 bytes[32];
	unsigned int submits, waits, puts, live;
	unsigned int fail_submit, fail_status;
	int status;
	u64 base;
	struct dma_fence fence;
} mock;
static struct amdgpu_device device;

static struct page *migrate_pfn_to_page(unsigned long pfn)
{
	static struct page page;
	return pfn & MIGRATE_PFN_VALID ? &page : NULL;
}

static void mutex_lock(bool *lock)
{
	assert(!*lock);
	*lock = true;
}

static void mutex_unlock(bool *lock)
{
	assert(*lock);
	*lock = false;
}

static u64 amdgpu_ttm_domain_start(struct amdgpu_device *adev, int domain)
{
	assert(adev == &device && domain == TTM_PL_VRAM);
	return mock.base;
}

static int amdgpu_ttm_fill_mem(struct amdgpu_device *adev,
	struct amdgpu_ttm_buffer_entity *entity, u32 value, u64 address,
	u32 bytes, void *resv, struct dma_fence **fence, bool tmz, u64 job_id)
{
	assert(adev == &device && entity->lock && !mock.live);
	assert(!value && !resv && !tmz);
	assert(job_id == AMDGPU_KERNEL_JOB_ID_TTM_CLEAR_BUFFER);
	assert(bytes && bytes <= (256U << 20));
	assert(!(address & ~PAGE_MASK) && !(bytes & ~PAGE_MASK));
	assert(mock.submits < 32);
	mock.address[mock.submits] = address;
	mock.bytes[mock.submits++] = bytes;
	if (mock.submits == mock.fail_submit)
		return -ENOMEM;
	mock.fence.complete = false;
	mock.live++;
	*fence = &mock.fence;
	return 0;
}

static int dma_fence_wait(struct dma_fence *fence, bool interruptible)
{
	assert(fence == &mock.fence && mock.live == 1);
	assert(!device.mman.clear_entities[0].lock && !interruptible);
	fence->complete = true;
	mock.waits++;
	return 0;
}

static int dma_fence_get_status(struct dma_fence *fence)
{
	assert(fence == &mock.fence && fence->complete);
	return mock.submits == mock.fail_status ? mock.status : 1;
}

static void dma_fence_put(struct dma_fence *fence)
{
	assert(fence == &mock.fence && fence->complete && mock.live == 1);
	mock.live--;
	mock.puts++;
}

#include "checked-prefetch-batching.inc"

static void reset(void)
{
	assert(!mock.live && !device.mman.clear_entities[0].lock);
	memset(&mock, 0, sizeof(mock));
	device.mman.buffer_funcs_enabled = true;
	device.mman.num_clear_entities = 1;
	mock.base = 1ULL << 40;
}

static void retired(unsigned int submitted, unsigned int completed)
{
	assert(mock.submits == submitted && mock.waits == completed);
	assert(mock.puts == completed && !mock.live);
	assert(!device.mman.clear_entities[0].lock);
}

static void contiguous(void)
{
	const unsigned long npages = (1UL << 30) / PAGE_SIZE;
	struct migrate_vma migrate = { .npages = npages };
	u64 *offset = calloc(npages, sizeof(*offset));
	migrate.src = calloc(npages, sizeof(*migrate.src));
	migrate.dst = calloc(npages, sizeof(*migrate.dst));
	assert(offset && migrate.src && migrate.dst);
	for (unsigned long i = 0; i < npages; i++) {
		migrate.src[i] = MIGRATE_PFN_MIGRATE;
		migrate.dst[i] = MIGRATE_PFN_VALID;
		offset[i] = i * PAGE_SIZE;
	}
	reset();
	assert(!svm_migrate_zero_to_vram(&device, &migrate, offset));
	retired(4, 4);
	for (unsigned int i = 0; i < 4; i++) {
		assert(mock.address[i] == mock.base + ((u64)i << 28));
		assert(mock.bytes[i] == (256U << 20));
	}
	free(offset);
	free(migrate.src);
	free(migrate.dst);
}

static void mixed(void)
{
	/* Fresh pages at offset zero, initialized source, absent destination,
	 * nonmigrating source, and physical discontinuities must stay distinct.
	 */
	unsigned long src[] = {2, 2, 3, 2, 2, 0, 2, 2, 2, 2};
	unsigned long dst[] = {1, 1, 1, 0, 1, 1, 1, 1, 1, 1};
	u64 offset[] = {0, 4096, 8192, 12288, 16384, 20480,
		       24576, 40960, 45056, 32768};
	struct migrate_vma migrate = {10, src, dst};
	u64 expected_offset[] = {0, 16384, 24576, 40960, 32768};
	u32 expected_size[] = {8192, 4096, 4096, 8192, 4096};
	reset();
	assert(!svm_migrate_zero_to_vram(&device, &migrate, offset));
	retired(5, 5);
	for (unsigned int i = 0; i < 5; i++) {
		assert(mock.address[i] == mock.base + expected_offset[i]);
		assert(mock.bytes[i] == expected_size[i]);
	}
	reset();
	migrate.npages = 0;
	assert(!svm_migrate_zero_to_vram(&device, &migrate, offset));
	retired(0, 0);
	/* Stop before later runs if a fill fails, and drain previous runs. */
	migrate.npages = 10;
	mock.fail_submit = 2;
	assert(svm_migrate_zero_to_vram(&device, &migrate, offset) == -ENOMEM);
	retired(2, 1);
}

static void bounds_and_failures(void)
{
	reset();
	assert(!amdgpu_ttm_zero_vram_checked(&device, PAGE_SIZE,
					   (256ULL << 20) + PAGE_SIZE));
	retired(2, 2);
	assert(mock.address[1] == mock.base + PAGE_SIZE + (256ULL << 20));
	assert(mock.bytes[1] == PAGE_SIZE);
	for (unsigned int fail = 1; fail <= 3; fail++) {
		reset();
		mock.fail_submit = fail;
		assert(amdgpu_ttm_zero_vram_checked(&device, 0, 1ULL << 30) == -ENOMEM);
		retired(fail, fail - 1);
		reset();
		mock.fail_status = fail;
		mock.status = -EIO;
		assert(amdgpu_ttm_zero_vram_checked(&device, 0, 1ULL << 30) == -EIO);
		retired(fail, fail);
	}
	reset();
	mock.fail_status = 1;
	mock.status = 0;
	assert(amdgpu_ttm_zero_vram_checked(&device, 0, PAGE_SIZE) == -EIO);
	retired(1, 1);
	reset();
	assert(amdgpu_ttm_zero_vram_checked(&device, 1, PAGE_SIZE) == -EINVAL);
	assert(amdgpu_ttm_zero_vram_checked(&device, 0, PAGE_SIZE - 1) == -EINVAL);
	assert(amdgpu_ttm_zero_vram_checked(&device, 0, 0) == -EINVAL);
	assert(amdgpu_ttm_zero_vram_checked(&device, U64_MAX & PAGE_MASK,
					  PAGE_SIZE) == -EINVAL);
	mock.base = U64_MAX & PAGE_MASK;
	assert(amdgpu_ttm_zero_vram_checked(&device, 0, PAGE_SIZE) == -EINVAL);
	mock.base = 0;
	device.mman.num_clear_entities = 0;
	assert(amdgpu_ttm_zero_vram_checked(&device, 0, PAGE_SIZE) == -EINVAL);
	device.mman.num_clear_entities = 1;
	device.mman.buffer_funcs_enabled = false;
	assert(amdgpu_ttm_zero_vram_checked(&device, 0, PAGE_SIZE) == -EINVAL);
	retired(0, 0);
}

int main(void)
{
	contiguous();
	mixed();
	bounds_and_failures();
	puts("checked prefetch batching: coverage, bounds, and retirement PASS");
	return 0;
}
