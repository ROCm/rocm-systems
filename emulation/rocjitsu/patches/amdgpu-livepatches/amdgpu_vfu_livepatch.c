// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// amdgpu_vfu_livepatch.c — kernel live patch for amdgpu in vfio-user emulation
//
// This livepatch stubs functions that require real GPU hardware execution in order
// to make amdgpu load and the KFD stack initialize when backed by a rocjitsu
// vfio-user device instead of real hardware.
//
// Build:
//   make -C /lib/modules/$(uname -r)/build M=$(pwd) amdgpu_vfu_livepatch.ko
//
// Load order (before amdgpu):
//   modprobe gpu_sched
//   insmod amdgpu_vfu_livepatch.ko
//   modprobe amdgpu discovery=2 fw_load_type=0 ip_block_mask=0x1F \
//            vm_update_mode=3 gpu_recovery=0

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/livepatch.h>

// --- Ring test stubs ---
// SDMA and GFX ring tests require the GPU to execute DMA/PM4 packets.
// Without real hardware, these always timeout (-110 ETIMEDOUT).
// Return 0 immediately so the rings are marked ready.

static int patched_sdma_v4_4_2_ring_test_ring(void *ring)
{
	return 0;
}

static int patched_sdma_v4_4_2_ring_test_ib(void *ring, long timeout)
{
	return 0;
}

static int patched_gfx_v9_4_3_ring_test_ib(void *ring, long timeout)
{
	return 0;
}

// --- VM invalidation engine bypass ---
// amdgpu_gmc_allocate_vm_inv_eng assigns a VM invalidation engine to each ring.
// With 5 SDMA instances (for MI350P), the MMHUB engine bitmap is exhausted
// before all rings get assigned, returning -EINVAL and aborting gmc_late_init.
// In emulation, TLB invalidation is a no-op, so return success unconditionally.

static int patched_amdgpu_gmc_allocate_vm_inv_eng(void *adev)
{
	return 0;
}

// --- Fence wait stubs ---
// amdgpu_fence_wait_polling: polled by KIQ TLB invalidation (gmc_flush_gpu_tlb).
// Returns remaining timeout (positive = success). Return 1 to indicate immediate
// completion without actually waiting for GPU fence signal.

static long patched_amdgpu_fence_wait_polling(void *ring, unsigned int wait_seq,
					       long timeout)
{
	return 1;
}

// amdkfd_fence_wait_timeout: polled by DQM (device queue manager) after
// issuing SET_RESOURCES/MAP_PROCESS packets via the HWS MEC ring.
// Without GPU firmware, these packets never execute and the fence never completes.
// Return 0 (success) immediately.

static int patched_amdkfd_fence_wait_timeout(void *dqm,
					      unsigned long long fence_value,
					      unsigned int timeout_ms)
{
	return 0;
}

// --- Livepatch registration ---

static struct klp_func funcs[] = {
	{ .old_name = "sdma_v4_4_2_ring_test_ring",
	  .new_func  = patched_sdma_v4_4_2_ring_test_ring, },
	{ .old_name = "sdma_v4_4_2_ring_test_ib",
	  .new_func  = patched_sdma_v4_4_2_ring_test_ib, },
	{ .old_name = "gfx_v9_4_3_ring_test_ib",
	  .new_func  = patched_gfx_v9_4_3_ring_test_ib, },
	{ .old_name = "amdgpu_gmc_allocate_vm_inv_eng",
	  .new_func  = patched_amdgpu_gmc_allocate_vm_inv_eng, },
	{ .old_name = "amdgpu_fence_wait_polling",
	  .new_func  = patched_amdgpu_fence_wait_polling, },
	{ .old_name = "amdkfd_fence_wait_timeout",
	  .new_func  = patched_amdkfd_fence_wait_timeout, },
	{ }
};

static struct klp_object objs[] = {
	{ .name = "amdgpu", .funcs = funcs, },
	{ }
};

static struct klp_patch patch = {
	.mod     = THIS_MODULE,
	.objs    = objs,
	.replace = true,
};

static int __init amdgpu_vfu_livepatch_init(void)
{
	return klp_enable_patch(&patch);
}

module_init(amdgpu_vfu_livepatch_init);
MODULE_LICENSE("GPL");
MODULE_INFO(livepatch, "Y");
MODULE_DESCRIPTION("amdgpu live patch for vfio-user GPU emulation (rocjitsu)");
