// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// amdgpu_vfu_livepatch.c — kernel live patch for amdgpu in vfio-user emulation
//
// Stubs functions that require real GPU hardware so amdgpu loads and the
// KFD/HIP stack initializes when backed by rocjitsu vfio-user.
//
// Build:
//   echo 'obj-m := amdgpu_vfu_livepatch.o' > Kbuild
//   touch Makefile
//   make -C /lib/modules/$(uname -r)/build M=$(pwd) modules
//
// Load order (livepatch BEFORE amdgpu):
//   modprobe gpu_sched
//   insmod amdgpu_vfu_livepatch.ko
//   modprobe amdgpu discovery=2 fw_load_type=0 ip_block_mask=0x1F \
//            vm_update_mode=3 gpu_recovery=0

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/livepatch.h>
#include <linux/dma-fence.h>
#include <linux/delay.h>
#include <linux/kprobes.h>
#include <linux/atomic.h>

// ---- Ring test stubs ----

static int patched_sdma_v4_4_2_ring_test_ring(void *ring) { return 0; }
static int patched_sdma_v4_4_2_ring_test_ib(void *ring, long timeout) { return 0; }
static int patched_gfx_v9_4_3_ring_test_ib(void *ring, long timeout) { return 0; }

// ---- VM invalidation engine bypass ----
// 5 SDMA instances exhaust the MMHUB engine bitmap.

static int patched_amdgpu_gmc_allocate_vm_inv_eng(void *adev) { return 0; }

// ---- KIQ fence wait stub ----

static long patched_amdgpu_fence_wait_polling(void *ring, unsigned int wait_seq,
                                               long timeout) { return 1; }

// ---- DQM fence wait stub ----

static int patched_amdkfd_fence_wait_timeout(void *dqm,
                                              unsigned long long fence_value,
                                              unsigned int timeout_ms) { return 0; }

// ---- DRM scheduler job stub ----

static void *patched_amdgpu_job_run(void *sched_job)
{
    return dma_fence_get_stub();
}

// ---- Queue buffer acquisition stub ----
// kfd_queue_buffer_svm_get always returns -EINVAL in this kernel
// (CONFIG_HSA_AMD_SVM_AMDKCL not set). Stubbing lets CREATE_QUEUE
// proceed to set up DQM queue structures.

static int patched_kfd_queue_acquire_buffers(void *pdd, void *properties)
{
    return 0;
}

// ---- KFD event signaling via kfd_signal_event_interrupt ----
// Resolved at runtime via kprobe after amdgpu loads.
//
// kfd_signal_event_interrupt(u32 pasid, u32 partial_id, u32 valid_id_bits):
//   With partial_id=0, valid_id_bits=0: scans ALL signal_page slots for
//   the process with the given PASID and signals any that are marked.
//
// kfd_lookup_process_by_pasid(u32 pasid, void*extra): returns kfd_process*
//   or NULL if not found.

static void (*fn_kfd_signal)(unsigned int pasid, unsigned int partial_id,
                             unsigned int valid_id_bits);
static void *(*fn_kfd_lookup)(unsigned int pasid, void *extra);
static atomic_t kfd_fns_resolved = ATOMIC_INIT(0);

static void resolve_kfd_fns(void)
{
    if (atomic_read(&kfd_fns_resolved))
        return;

    struct kprobe kp1 = { .symbol_name = "kfd_signal_event_interrupt" };
    struct kprobe kp2 = { .symbol_name = "kfd_lookup_process_by_pasid" };

    if (register_kprobe(&kp1) == 0) {
        fn_kfd_signal = (typeof(fn_kfd_signal))kp1.addr;
        unregister_kprobe(&kp1);
        pr_info("[vfu_lp] kfd_signal_event_interrupt at %px\n", fn_kfd_signal);
    }
    if (register_kprobe(&kp2) == 0) {
        fn_kfd_lookup = (typeof(fn_kfd_lookup))kp2.addr;
        unregister_kprobe(&kp2);
        pr_info("[vfu_lp] kfd_lookup_process_by_pasid at %px\n", fn_kfd_lookup);
    }
    if (fn_kfd_signal && fn_kfd_lookup)
        atomic_set(&kfd_fns_resolved, 1);
}

// ---- KFD event wait stub ----
//
// Without real GPU interrupts, WAIT_EVENTS blocks forever waiting for
// GPU completion events. We:
//   1. Sleep 5ms to pace the ROCr polling loop
//   2. Call kfd_signal_event_interrupt for this process's PASID to fire
//      any KFD events that are already marked in the signal_page
//   3. Return TIMEOUT so ROCr retries, now with events signaled
//
// The PASID is found by scanning 1..1023 and checking which matches
// this kfd_process pointer via kfd_lookup_process_by_pasid.
//
// Current limitation: hipStreamCreate still blocks because the futex
// it waits on is driven by the queue's ACTIVE flag, which is set by
// MEC firmware after processing MAP_QUEUES — not by KFD events.
// Next step: KIQ PM4 MAP_QUEUES packet execution in rocjitsu-vfu.
//
// kfd_ioctl_wait_events_args:
//   +0:  uint64_t events_ptr
//   +8:  uint32_t num_events
//   +12: uint32_t wait_for_all
//   +16: uint32_t timeout
//   +20: uint32_t wait_result  (0=COMPLETE, 1=TIMEOUT, 2=FAIL)

static int patched_kfd_ioctl_wait_events(void *filp, void *p, void *data)
{
    uint32_t timeout  = *(uint32_t *)((uint8_t *)data + 16);
    uint32_t *wait_result = (uint32_t *)((uint8_t *)data + 20);

    if (timeout == 0) {
        *wait_result = 0; /* COMPLETE */
        return 0;
    }

    if (!atomic_read(&kfd_fns_resolved))
        resolve_kfd_fns();

    msleep(5);

    /* Signal events for this process by finding its PASID */
    if (fn_kfd_lookup && fn_kfd_signal) {
        unsigned int pasid;
        for (pasid = 1; pasid < 1024; ++pasid) {
            void *found = fn_kfd_lookup(pasid, NULL);
            if (found == p) {
                fn_kfd_signal(pasid, 0, 0);
                break;
            }
        }
    }

    *wait_result = 1; /* TIMEOUT — ROCr retries, now with events signaled */
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
    { .old_name = "kfd_queue_acquire_buffers",
      .new_func  = patched_kfd_queue_acquire_buffers, },
    { .old_name = "kfd_ioctl_wait_events",
      .new_func  = patched_kfd_ioctl_wait_events, },
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
    resolve_kfd_fns();
    return klp_enable_patch(&patch);
}

module_init(amdgpu_vfu_livepatch_init);
MODULE_LICENSE("GPL");
MODULE_INFO(livepatch, "Y");
MODULE_DESCRIPTION("amdgpu live patch for vfio-user GPU emulation (rocjitsu)");
