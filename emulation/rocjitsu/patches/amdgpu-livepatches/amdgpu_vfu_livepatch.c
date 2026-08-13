// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// amdgpu_vfu_livepatch.c — kernel live patch for amdgpu in vfio-user emulation
//
// This livepatch stubs functions that require real GPU hardware execution in order
// to make amdgpu load and the KFD/HIP stack initialize when backed by a rocjitsu
// vfio-user device instead of real hardware.
//
// Build:
//   echo 'obj-m := amdgpu_vfu_livepatch.o' > Kbuild
//   touch Makefile
//   make -C /lib/modules/$(uname -r)/build M=$(pwd) modules
//
// Load order (before amdgpu):
//   modprobe gpu_sched
//   insmod amdgpu_vfu_livepatch.ko
//   modprobe amdgpu discovery=2 fw_load_type=0 ip_block_mask=0x1F \
//            vm_update_mode=3 gpu_recovery=0

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/livepatch.h>
#include <linux/dma-fence.h>
#include <linux/delay.h>
#include <linux/atomic.h>

// ---- Ring test stubs ----
// SDMA and GFX IB ring tests require the GPU to execute DMA/PM4 packets.

static int patched_sdma_v4_4_2_ring_test_ring(void *ring) { return 0; }
static int patched_sdma_v4_4_2_ring_test_ib(void *ring, long timeout) { return 0; }
static int patched_gfx_v9_4_3_ring_test_ib(void *ring, long timeout) { return 0; }

// ---- VM invalidation engine bypass ----
// With 5 SDMA instances, the MMHUB engine bitmap is exhausted.

static int patched_amdgpu_gmc_allocate_vm_inv_eng(void *adev) { return 0; }

// ---- KIQ fence wait stubs ----
// Returns success immediately without waiting for GPU to write fence value.

static long patched_amdgpu_fence_wait_polling(void *ring, unsigned int wait_seq,
                                               long timeout) { return 1; }

// ---- DQM fence wait stub ----
// amdkfd_fence_wait_timeout: polled after HWS SET_RESOURCES/MAP_PROCESS packets.

static int patched_amdkfd_fence_wait_timeout(void *dqm,
                                              unsigned long long fence_value,
                                              unsigned int timeout_ms) { return 0; }

// ---- DRM scheduler job stub ----
// Returns a pre-signaled stub fence so the DRM scheduler immediately
// considers the job complete without GPU execution.

static void *patched_amdgpu_job_run(void *sched_job)
{
    return dma_fence_get_stub();
}

// ---- KFD event wait stub ----
// kfd_ioctl_wait_events_args:
//   +0:  uint64_t events_ptr
//   +8:  uint32_t num_events
//   +12: uint32_t wait_for_all
//   +16: uint32_t timeout   (ms; 0=poll, ~0=infinite)
//   +20: uint32_t wait_result  (0=COMPLETE, 1=TIMEOUT, 2=FAIL)
//
// Background: ROCr has background worker threads that block in WAIT_EVENTS
// waiting for GPU completion events (signaled by kfd_signal_event_interrupt
// via IH ring). Without real GPU interrupts, these never arrive.
//
// Strategy: sleep 10ms then return COMPLETE every 5th call, TIMEOUT on others.
// This gives worker threads a periodic "tick" to advance their state machine
// without crashing on unhandled TIMEOUT.
//
// Limitation: spurious COMPLETE causes ROCs to access uninitialized queue
// state (null deref) in some code paths. Full solution requires IH ring
// emulation in rocjitsu-vfu to inject proper GPU completion events.

static atomic_t wait_events_count = ATOMIC_INIT(0);

static int patched_kfd_ioctl_wait_events(void *filp, void *p, void *data)
{
    uint32_t timeout  = *(uint32_t *)((uint8_t *)data + 16);
    uint32_t *wait_result = (uint32_t *)((uint8_t *)data + 20);
    if (timeout == 0) {
        *wait_result = 0; /* KFD_IOC_WAIT_RESULT_COMPLETE */
    } else {
        msleep(10);
        uint32_t n = (uint32_t)atomic_fetch_add(1, &wait_events_count);
        *wait_result = ((n % 5) == 0) ? 0 : 1; /* COMPLETE every 5th, else TIMEOUT */
    }
    return 0;
}

// ---- Queue buffer acquisition stub ----
// kfd_queue_acquire_buffers validates that queue ring, pointers, and CWSR
// area are properly mapped in the GPU VM. In this kernel build,
// kfd_queue_buffer_svm_get always returns -EINVAL (CONFIG_HSA_AMD_SVM_AMDKCL
// not set), so the CWSR buffer can't be acquired via the SVM fallback path.
//
// By returning 0 unconditionally, CREATE_QUEUE proceeds to pqm_create_queue
// which sets up the DQM queue state. The queue BO pointers (ring_bo, cwsr_bo)
// remain NULL, so actual GPU queue execution will fail, but the kernel-side
// queue structures are created correctly.
//
// Signature: int kfd_queue_acquire_buffers(struct kfd_process_device *pdd,
//                                          struct queue_properties *properties)

static int patched_kfd_queue_acquire_buffers(void *pdd, void *properties)
{
    return 0;
}

// ---- Livepatch registration ----

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
    { .old_name = "amdgpu_job_run",
      .new_func  = patched_amdgpu_job_run, },
    { .old_name = "kfd_ioctl_wait_events",
      .new_func  = patched_kfd_ioctl_wait_events, },
    { .old_name = "kfd_queue_acquire_buffers",
      .new_func  = patched_kfd_queue_acquire_buffers, },
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
