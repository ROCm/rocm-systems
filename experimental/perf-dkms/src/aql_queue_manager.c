/*
 * aql_queue_manager.c - AQL GPU Queue Lifecycle Management
 *
 * Implements queue create/destroy/submit/wait using KFD ioctls via
 * the vm_mmap bridge. No kernel patch needed.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/fs.h>
#include <linux/file.h>
#include <linux/fdtable.h>
#include <linux/mm.h>
#include <linux/mman.h>
#include <linux/vmalloc.h>
#include <linux/io.h>
#include <asm/cacheflush.h>
#include <linux/delay.h>
#include <linux/jiffies.h>
#include <linux/highmem.h>
#include <linux/sched/mm.h>
#include <uapi/linux/kfd_ioctl.h>

#include "aql_queue_manager.h"
#include "kfd_ioctl_bridge.h"
#include "aql_perf.h"

/* ========================================================================
 * AMDKFD_IOC_PROFILER ioctl definitions (DKMS-only, not in upstream UAPI)
 *
 * The profiler ioctl enables compute_perfcount_enable in the MQD for all
 * queues on a GPU, allowing performance counter access from compute queues.
 * On upstream kernels this ioctl doesn't exist, so we fall back to the
 * kernel patch that sets compute_perfcount_enable in init_mqd().
 * ======================================================================== */

#ifndef AMDKFD_IOC_PROFILER

enum kfd_profiler_ops {
	KFD_IOC_PROFILER_PMC = 0,
};

struct kfd_ioctl_pmc_settings {
	__u32 gpu_id;
	__u32 lock;
	__u32 perfcount_enable;
};

struct kfd_ioctl_profiler_args {
	__u32 op;
	union {
		struct kfd_ioctl_pmc_settings pmc;
		__u32 version;
		/* pc_sample and ptl_control omitted — not needed */
		__u8 _pad[68]; /* ensure struct is large enough */
	};
};

#define AMDKFD_IOC_PROFILER	\
	_IOWR('K', 0x86, struct kfd_ioctl_profiler_args)

#endif /* AMDKFD_IOC_PROFILER */

/**
 * profiler_lock_and_enable - Lock GPU for profiling and enable perfcount
 * @kfd_filp: Open /dev/kfd file pointer
 * @gpu_id: KFD GPU ID
 *
 * Calls AMDKFD_IOC_PROFILER with KFD_IOC_PROFILER_PMC to:
 *   1. Lock the GPU for profiling (prevents concurrent profiling)
 *   2. Enable compute_perfcount_enable on all existing queues
 *
 * This replaces the kernel patch that sets compute_perfcount_enable
 * in init_mqd() for all MQD manager versions.
 *
 * Returns: 0 on success, -ENOTTY if ioctl not supported (upstream kernel),
 *          other negative error codes on failure
 */
static int profiler_lock_and_enable(struct file *kfd_filp, uint32_t gpu_id)
{
	struct kfd_ioctl_profiler_args args = {};
	int ret;

	args.op = KFD_IOC_PROFILER_PMC;
	args.pmc.gpu_id = gpu_id;
	args.pmc.lock = 1;
	args.pmc.perfcount_enable = 1;

	ret = kfd_bridge_ioctl(kfd_filp, AMDKFD_IOC_PROFILER,
			       &args, sizeof(args));
	if (ret == -ENOTTY || ret == -EINVAL) {
		aql_info("profiler_lock: AMDKFD_IOC_PROFILER not supported "
			 "(upstream kernel?), relying on kernel patch");
		return ret;
	}
	if (ret) {
		aql_err("profiler_lock: AMDKFD_IOC_PROFILER failed: %d", ret);
		return ret;
	}

	aql_info("profiler_lock: GPU %u locked for profiling, perfcount enabled",
		 gpu_id);
	return 0;
}

/**
 * profiler_unlock_and_disable - Unlock GPU profiling and disable perfcount
 * @kfd_filp: Open /dev/kfd file pointer
 * @gpu_id: KFD GPU ID
 */
static void profiler_unlock_and_disable(struct file *kfd_filp, uint32_t gpu_id)
{
	struct kfd_ioctl_profiler_args args = {};
	int ret;

	args.op = KFD_IOC_PROFILER_PMC;
	args.pmc.gpu_id = gpu_id;
	args.pmc.lock = 0;
	args.pmc.perfcount_enable = 0;

	ret = kfd_bridge_ioctl(kfd_filp, AMDKFD_IOC_PROFILER,
			       &args, sizeof(args));
	if (ret && ret != -ENOTTY && ret != -EINVAL)
		aql_debug("profiler_unlock: AMDKFD_IOC_PROFILER failed: %d", ret);
	else if (!ret)
		aql_info("profiler_unlock: GPU %u unlocked, perfcount disabled",
			 gpu_id);
}

/* Parse a u32 value from "key value\n" format in a buffer.
 * Returns 0 on success, <0 on failure. */
static int parse_sysfs_u32(const char *buf, const char *key, u32 *out)
{
	const char *p = strstr(buf, key);
	const char *val;
	char tmp[16];
	int i;

	if (!p)
		return -ENOENT;
	val = p + strlen(key);
	/* skip whitespace */
	while (*val == ' ' || *val == '\t')
		val++;
	/* copy digits to temp buffer */
	for (i = 0; i < 15 && val[i] && val[i] != '\n' && val[i] != ' '; i++)
		tmp[i] = val[i];
	tmp[i] = '\0';
	return kstrtou32(tmp, 10, out);
}

/* KFD mmap type constants (from kfd_priv.h, not exported in UAPI) */
#ifndef KFD_MMAP_TYPE_DOORBELL
#define KFD_GPU_ID_HASH_WIDTH		16
#define KFD_MMAP_TYPE_SHIFT		62
#define KFD_MMAP_TYPE_DOORBELL		(0x3ULL << KFD_MMAP_TYPE_SHIFT)
#define KFD_MMAP_GPU_ID_SHIFT		46
#define KFD_MMAP_GPU_ID_MASK		(((1ULL << KFD_GPU_ID_HASH_WIDTH) - 1) \
					 << KFD_MMAP_GPU_ID_SHIFT)
#define KFD_MMAP_GPU_ID(gpu_id)		((((uint64_t)(gpu_id)) \
					  << KFD_MMAP_GPU_ID_SHIFT) \
					 & KFD_MMAP_GPU_ID_MASK)
#endif

/* ========================================================================
 * CWSR (Context Save/Restore) size computation — mirrors KFD logic
 * ======================================================================== */

#define SGPR_SIZE_PER_CU	0x4000
#define LDS_SIZE_PER_CU		0x10000
#define HWREG_SIZE_PER_CU	0x1000
#define DEBUGGER_BYTES_ALIGN	64
#define DEBUGGER_BYTES_PER_WAVE	32
#define SIZEOF_HSA_USER_CONTEXT_SAVE_AREA_HEADER 40

static u32 get_vgpr_size_per_cu(u32 gfxv)
{
	if (gfxv == 90402 || gfxv == 90010 || gfxv == 90008 || gfxv == 90500)
		return 0x80000;
	else if (gfxv == 110000 || gfxv == 110001 ||
		 gfxv == 120000 || gfxv == 120001)
		return 0x60000;
	return 0x40000;
}

static void compute_cwsr_sizes(u32 gfxv, u32 simd_count, u32 simd_per_cu,
			       u32 array_count, u32 simd_arrays_per_engine,
			       u32 num_xcc,
			       u32 *out_ctl_stack_size, u32 *out_cwsr_size)
{
	u32 cu_num = simd_count / simd_per_cu / num_xcc;
	u32 wave_num = (gfxv < 100100) ?
		min(cu_num * 40, array_count / simd_arrays_per_engine * 512)
		: cu_num * 32;
	u32 cntl_stack_per_wave = (gfxv >= 100100) ? 12 : 8;
	u32 wg_data_per_cu = get_vgpr_size_per_cu(gfxv) + SGPR_SIZE_PER_CU
			   + LDS_SIZE_PER_CU + HWREG_SIZE_PER_CU;
	u32 wg_data_size = ALIGN(cu_num * wg_data_per_cu, PAGE_SIZE);
	u32 ctl_stack = wave_num * cntl_stack_per_wave + 8;

	ctl_stack = ALIGN(SIZEOF_HSA_USER_CONTEXT_SAVE_AREA_HEADER + ctl_stack,
			  PAGE_SIZE);

	/* GFX10 limit */
	if ((gfxv / 10000 * 10000) == 100000)
		ctl_stack = min(ctl_stack, (u32)0x7000);

	*out_ctl_stack_size = ctl_stack;
	*out_cwsr_size = ctl_stack + wg_data_size;
}

/* ========================================================================
 * GPU Memory Allocation via KFD Bridge
 * ======================================================================== */

/**
 * alloc_gpu_buffer - Allocate a GPU-accessible buffer via KFD USERPTR
 *
 * Allocates user-space pages, registers them with KFD as USERPTR so the
 * GPU can access them. We pin the pages and vmap them here so the kernel
 * always has a valid VA. This avoids the DRM render node mmap path.
 */
static int alloc_gpu_buffer(struct file *kfd_filp, uint32_t gpu_id,
			    struct aql_gpu_buffer *buf, uint32_t size,
			    uint32_t extra_flags, pgprot_t cache_mode)
{
	struct kfd_ioctl_alloc_memory_of_gpu_args args = {};
	unsigned long user_addr;
	int nr_pages;
	int ret;

	nr_pages = (size + PAGE_SIZE - 1) / PAGE_SIZE;

	/* Step 1: Allocate user pages (anonymous mapping) */
	user_addr = vm_mmap(NULL, 0, size,
			    PROT_READ | PROT_WRITE,
			    MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, 0);
	if (IS_ERR_VALUE(user_addr)) {
		aql_err("alloc_gpu_buffer: user page alloc failed: %ld",
			(long)user_addr);
		return (int)user_addr;
	}

	/* Step 2: Pin user pages + vmap for kernel VA */
	buf->pages = kcalloc(nr_pages, sizeof(struct page *), GFP_KERNEL);
	if (!buf->pages) {
		vm_munmap(user_addr, size);
		return -ENOMEM;
	}

	mmap_read_lock(current->mm);
	ret = pin_user_pages(user_addr, nr_pages,
			     FOLL_WRITE | FOLL_LONGTERM,
			     buf->pages);
	mmap_read_unlock(current->mm);
	if (ret < 0) {
		aql_err("alloc_gpu_buffer: pin_user_pages failed: %d", ret);
		kfree(buf->pages);
		buf->pages = NULL;
		vm_munmap(user_addr, size);
		return ret;
	}
	if (ret != nr_pages) {
		aql_err("alloc_gpu_buffer: pinned %d/%d pages", ret, nr_pages);
		unpin_user_pages(buf->pages, ret);
		kfree(buf->pages);
		buf->pages = NULL;
		vm_munmap(user_addr, size);
		return -EFAULT;
	}
	buf->nr_pages = nr_pages;

	buf->cpu_addr = vmap(buf->pages, nr_pages, VM_MAP, cache_mode);
	if (!buf->cpu_addr) {
		aql_err("alloc_gpu_buffer: vmap failed");
		unpin_user_pages(buf->pages, nr_pages);
		kfree(buf->pages);
		buf->pages = NULL;
		buf->nr_pages = 0;
		vm_munmap(user_addr, size);
		return -ENOMEM;
	}

	/* Step 3: Register with KFD as USERPTR for GPU access */
	args.gpu_id = gpu_id;
	args.va_addr = user_addr;  /* GPU VA = user VA for USERPTR */
	args.size = size;
	args.mmap_offset = user_addr;  /* For USERPTR: mmap_offset = CPU address */
	args.flags = KFD_IOC_ALLOC_MEM_FLAGS_USERPTR
		   | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
		   | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE
		   | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT
		   | extra_flags;

	ret = kfd_bridge_ioctl(kfd_filp, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU,
			       &args, sizeof(args));
	if (ret) {
		aql_err("alloc_gpu_buffer: ALLOC_MEMORY_OF_GPU (USERPTR) failed: %d (size=%u)",
			ret, size);
		vunmap(buf->cpu_addr);
		buf->cpu_addr = NULL;
		unpin_user_pages(buf->pages, nr_pages);
		kfree(buf->pages);
		buf->pages = NULL;
		buf->nr_pages = 0;
		vm_munmap(user_addr, size);
		return ret;
	}

	buf->handle = args.handle;
	buf->gpu_addr = args.va_addr;
	buf->mmap_offset = args.mmap_offset;
	buf->size = size;
	buf->user_addr = user_addr;

	aql_debug("alloc_gpu_buffer: handle=0x%llx, gpu_addr=0x%llx, cpu=%p, user=0x%lx, size=%u, nocache=%d",
		  buf->handle, buf->gpu_addr, buf->cpu_addr, buf->user_addr, size,
		  pgprot_val(cache_mode) != pgprot_val(PAGE_KERNEL));

	return 0;
}

/**
 * alloc_gpu_buffer_gtt - Allocate a GTT GPU buffer via KFD
 *
 * Allocates GPU-accessible GTT memory (system RAM in GPU page tables).
 * GTT BOs can be pinned to GART, which is required for wptr/rptr buffers.
 *
 * Flow:
 *   1. Reserve user VA via vm_mmap(NULL) — this becomes the GPU VA
 *   2. Allocate GTT BO via KFD at that VA — KFD returns mmap_offset
 *   3. Remap the same VA to the BO's pages via DRM render node mmap
 *   4. Pin + vmap for kernel VA
 */
static int alloc_gpu_buffer_gtt(struct file *kfd_filp, uint32_t gpu_id,
				struct aql_gpu_buffer *buf, uint32_t size,
				struct file *drm_filp, int drm_fd)
{
	struct kfd_ioctl_alloc_memory_of_gpu_args args = {};
	unsigned long user_addr, drm_addr;
	int nr_pages;
	int ret;

	nr_pages = (size + PAGE_SIZE - 1) / PAGE_SIZE;

	/* Step 1: Reserve a user VA range (provides valid user address for
	 * access_ok checks and becomes the GPU VA) */
	user_addr = vm_mmap(NULL, 0, size,
			    PROT_READ | PROT_WRITE,
			    MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, 0);
	if (IS_ERR_VALUE(user_addr)) {
		aql_err("alloc_gpu_buffer_gtt: VA reservation failed: %ld",
			(long)user_addr);
		return (int)user_addr;
	}

	/* Step 2: Allocate GTT BO via KFD at the reserved VA */
	args.gpu_id = gpu_id;
	args.va_addr = user_addr;
	args.size = size;
	args.flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT
		   | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
		   | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE
		   | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;

	ret = kfd_bridge_ioctl(kfd_filp, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU,
			       &args, sizeof(args));
	if (ret) {
		aql_err("alloc_gpu_buffer_gtt: ALLOC_MEMORY_OF_GPU (GTT) failed: %d", ret);
		vm_munmap(user_addr, size);
		return ret;
	}

	buf->handle = args.handle;
	buf->gpu_addr = args.va_addr;
	buf->mmap_offset = args.mmap_offset;
	buf->size = size;

	/* Step 3: Unmap the anonymous pages and remap with DRM to get
	 * access to the actual GTT BO pages */
	vm_munmap(user_addr, size);

	drm_addr = vm_mmap(drm_filp, user_addr, size,
			   PROT_READ | PROT_WRITE,
			   MAP_SHARED | MAP_FIXED, args.mmap_offset);
	if (IS_ERR_VALUE(drm_addr)) {
		aql_err("alloc_gpu_buffer_gtt: DRM remap failed: %ld (mmap_offset=0x%llx)",
			(long)drm_addr, args.mmap_offset);
		struct kfd_ioctl_free_memory_of_gpu_args fa = {};
		fa.handle = buf->handle;
		kfd_bridge_ioctl(kfd_filp, AMDKFD_IOC_FREE_MEMORY_OF_GPU,
				 &fa, sizeof(fa));
		buf->handle = 0;
		return (int)drm_addr;
	}
	buf->user_addr = drm_addr;

	/* Step 4: For GTT BOs, we cannot pin/vmap the DRM-mapped pages
	 * (VM_PFNMAP). cpu_addr is left NULL — the caller must use the
	 * user_addr with kthread_use_mm for access from kernel context. */
	buf->cpu_addr = NULL;

	buf->is_gtt = true;

	aql_debug("alloc_gpu_buffer_gtt: handle=0x%llx, gpu_addr=0x%llx, cpu=%p, user=0x%lx, size=%u",
		  buf->handle, buf->gpu_addr, buf->cpu_addr, buf->user_addr, size);

	return 0;

err_munmap:
	vm_munmap(drm_addr, size);
	buf->user_addr = 0;
	{
		struct kfd_ioctl_free_memory_of_gpu_args fa = {};
		fa.handle = buf->handle;
		kfd_bridge_ioctl(kfd_filp, AMDKFD_IOC_FREE_MEMORY_OF_GPU,
				 &fa, sizeof(fa));
	}
	buf->handle = 0;
	return ret;
}

/**
 * free_gpu_buffer - Free a GPU buffer via KFD ioctl and clean up
 */
static void free_gpu_buffer(struct file *kfd_filp, uint32_t gpu_id,
			    struct aql_gpu_buffer *buf)
{
	struct kfd_ioctl_free_memory_of_gpu_args args = {};

	if (!buf->handle)
		return;

	args.handle = buf->handle;

	kfd_bridge_ioctl(kfd_filp, AMDKFD_IOC_FREE_MEMORY_OF_GPU,
			 &args, sizeof(args));

	/* Clean up kernel mapping */
	if (buf->cpu_addr) {
		vunmap(buf->cpu_addr);
		buf->cpu_addr = NULL;
	}

	/* Release pages */
	if (buf->pages && buf->nr_pages > 0) {
		if (buf->is_gtt) {
			/* GTT: pages from get_user_pages */
			int i;
			for (i = 0; i < buf->nr_pages; i++)
				put_page(buf->pages[i]);
		} else {
			/* USERPTR: pages from pin_user_pages */
			unpin_user_pages(buf->pages, buf->nr_pages);
		}
		kfree(buf->pages);
		buf->pages = NULL;
		buf->nr_pages = 0;
	}

	/* Release user VA mapping */
	if (buf->user_addr && buf->size)
		vm_munmap(buf->user_addr, buf->size);

	buf->handle = 0;
	buf->gpu_addr = 0;
	buf->user_addr = 0;
}

/**
 * map_gpu_buffers - Map all GPU buffers to the GPU via KFD ioctl
 *
 * Uses the nested bridge variant since MAP_MEMORY_TO_GPU has
 * an embedded device_ids_array_ptr.
 */
static int map_gpu_buffers(struct file *kfd_filp, uint32_t gpu_id,
			   struct aql_gpu_buffer *bufs, int num_bufs)
{
	int i, ret;

	for (i = 0; i < num_bufs; i++) {
		struct kfd_ioctl_map_memory_to_gpu_args args = {};

		if (!bufs[i].handle)
			continue;

		args.handle = bufs[i].handle;
		args.n_devices = 1;

		/* Use nested bridge: device_ids_array_ptr is at offset
		 * of that field within the args struct */
		ret = kfd_bridge_ioctl_nested(
			kfd_filp, AMDKFD_IOC_MAP_MEMORY_TO_GPU,
			&args, sizeof(args),
			&gpu_id, sizeof(gpu_id),
			offsetof(struct kfd_ioctl_map_memory_to_gpu_args,
				 device_ids_array_ptr));
		if (ret) {
			aql_err("map_gpu_buffers: MAP_MEMORY_TO_GPU failed for buf %d: %d",
				i, ret);
			return ret;
		}
	}

	return 0;
}

/**
 * unmap_gpu_buffers - Unmap all GPU buffers from the GPU
 */
static void unmap_gpu_buffers(struct file *kfd_filp, uint32_t gpu_id,
			      struct aql_gpu_buffer *bufs, int num_bufs)
{
	int i;

	for (i = 0; i < num_bufs; i++) {
		struct kfd_ioctl_unmap_memory_from_gpu_args args = {};

		if (!bufs[i].handle)
			continue;

		args.handle = bufs[i].handle;
		args.n_devices = 1;

		kfd_bridge_ioctl_nested(
			kfd_filp, AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU,
			&args, sizeof(args),
			&gpu_id, sizeof(gpu_id),
			offsetof(struct kfd_ioctl_unmap_memory_from_gpu_args,
				 device_ids_array_ptr));
	}
}

/* ========================================================================
 * ACQUIRE_VM: bind process to GPU via DRM render node
 * ======================================================================== */

/**
 * find_render_node_for_gpu - Find /dev/dri/renderDN for a KFD gpu_id
 *
 * Walks /sys/class/kfd/kfd/topology/nodes/ to find the node matching
 * gpu_id, then finds the corresponding DRM render minor.
 */
static int find_render_minor_for_gpu(uint32_t gpu_id)
{
	char path[128];
	char buf[32];
	struct file *f;
	loff_t pos;
	ssize_t len;
	int node_id;
	uint32_t node_gpu_id;
	int render_minor = -ENODEV;

	/*
	 * Walk KFD topology nodes to find the one matching gpu_id,
	 * then read its drm_render_minor property.
	 */
	for (node_id = 0; node_id < 16; node_id++) {
		snprintf(path, sizeof(path),
			 "/sys/class/kfd/kfd/topology/nodes/%d/gpu_id", node_id);
		f = filp_open(path, O_RDONLY, 0);
		if (IS_ERR(f))
			continue;

		pos = 0;
		memset(buf, 0, sizeof(buf));
		len = kernel_read(f, buf, sizeof(buf) - 1, &pos);
		filp_close(f, NULL);

		if (len <= 0)
			continue;

		node_gpu_id = 0;
		if (kstrtou32(strim(buf), 10, &node_gpu_id) || node_gpu_id != gpu_id)
			continue;

		/* Found the node — now read drm_render_minor from properties */
		snprintf(path, sizeof(path),
			 "/sys/class/kfd/kfd/topology/nodes/%d/properties", node_id);
		f = filp_open(path, O_RDONLY, 0);
		if (IS_ERR(f))
			break;

		{
			char *propbuf;
			char *line;

			propbuf = kmalloc(4096, GFP_KERNEL);
			if (!propbuf) {
				filp_close(f, NULL);
				break;
			}

			pos = 0;
			len = kernel_read(f, propbuf, 4095, &pos);
			filp_close(f, NULL);

			aql_debug("find_render_minor: properties read len=%zd", len);

			if (len > 0) {
				propbuf[len] = '\0';
				line = strstr(propbuf, "drm_render_minor");
				if (line) {
					uint32_t minor;
					char *val = line + strlen("drm_render_minor");
					char *nl;
					while (*val == ' ' || *val == '\t')
						val++;
					/* Terminate at newline for kstrtou32 */
					nl = strchr(val, '\n');
					if (nl)
						*nl = '\0';
					if (!kstrtou32(val, 10, &minor) && minor > 0) {
						render_minor = (int)minor;
						aql_debug("find_render_minor: GPU %u -> "
							 "renderD%d (node %d)",
							 gpu_id, render_minor, node_id);
					} else {
						aql_err("find_render_minor: kstrtou32 "
							"failed for '%.20s'", val);
					}
				} else {
					aql_err("find_render_minor: 'drm_render_minor'"
						" not found in %zd bytes", len);
				}
			} else {
				aql_err("find_render_minor: kernel_read returned %zd",
					len);
			}
			kfree(propbuf);
		}
		break;
	}

	if (render_minor < 0)
		aql_err("find_render_minor: no render node found for GPU %u",
			gpu_id);

	return render_minor;
}

/**
 * acquire_vm_for_gpu - Call ACQUIRE_VM ioctl to bind process VM to GPU
 *
 * Opens the DRM render node, installs it as an fd, calls ACQUIRE_VM
 * via the bridge. On success, the DRM fd is kept open (stored in
 * out_drm_fd) because we need it for vm_mmap of GPU buffers. The
 * caller must call close_fd() when done with mmap operations.
 */
static int acquire_vm_for_gpu(struct file *kfd_filp, uint32_t gpu_id,
			      int render_minor,
			      struct file **out_drm_filp, int *out_drm_fd)
{
	struct kfd_ioctl_acquire_vm_args args = {};
	struct file *drm_filp;
	char drm_path[64];
	int drm_fd;
	int ret;

	*out_drm_filp = NULL;
	*out_drm_fd = -1;

	snprintf(drm_path, sizeof(drm_path), "/dev/dri/renderD%d", render_minor);
	drm_filp = filp_open(drm_path, O_RDWR, 0);
	if (IS_ERR(drm_filp)) {
		aql_err("acquire_vm: failed to open %s: %ld",
			drm_path, PTR_ERR(drm_filp));
		return PTR_ERR(drm_filp);
	}

	/* Install the DRM file into the current process's fd table
	 * so that fget() in the ACQUIRE_VM handler can find it.
	 * We keep this fd open so vm_mmap can work with the same
	 * drm_file that was used for ACQUIRE_VM. */
	drm_fd = get_unused_fd_flags(O_CLOEXEC);
	if (drm_fd < 0) {
		aql_err("acquire_vm: get_unused_fd_flags failed: %d", drm_fd);
		filp_close(drm_filp, NULL);
		return drm_fd;
	}
	fd_install(drm_fd, drm_filp);

	/* Call ACQUIRE_VM — the handler does fget(drm_fd) and keeps
	 * a reference to the DRM file on success */
	args.drm_fd = drm_fd;
	args.gpu_id = gpu_id;

	aql_debug("acquire_vm: calling ACQUIRE_VM gpu_id=%u drm_fd=%d "
		 "render=%s kfd_filp=%px", gpu_id, drm_fd, drm_path, kfd_filp);

	/* Debug: verify fget works for drm_fd before calling ioctl */
	{
		struct file *test_filp = fget(drm_fd);
		if (test_filp) {
			aql_debug("acquire_vm: fget(%d) OK, filp=%px f_op=%px",
				 drm_fd, test_filp, test_filp->f_op);
			fput(test_filp);
		} else {
			aql_err("acquire_vm: fget(%d) returned NULL!", drm_fd);
		}
	}

	/* Debug: check KFD process from filep->private_data */
	{
		void *priv = kfd_filp->private_data;
		aql_debug("acquire_vm: kfd_filp->private_data=%px "
			 "current->pid=%d current->group_leader->pid=%d",
			 priv, current->pid, current->group_leader->pid);
	}

	ret = kfd_bridge_ioctl(kfd_filp, AMDKFD_IOC_ACQUIRE_VM,
			       &args, sizeof(args));
	if (ret) {
		aql_err("acquire_vm: ACQUIRE_VM failed for GPU %u drm_fd=%d: %d",
			gpu_id, drm_fd, ret);
		close_fd(drm_fd);
	} else {
		aql_debug("acquire_vm: VM acquired for GPU %u via %s (fd=%d)",
			  gpu_id, drm_path, drm_fd);
		*out_drm_filp = drm_filp;
		*out_drm_fd = drm_fd;
	}

	return ret;
}

/* Signal kind: AMD_SIGNAL_KIND_USER (1) = MES decrements value on completion */
#define AQL_SIGNAL_KIND_USER 1

/* ========================================================================
 * Queue create / destroy
 * ======================================================================== */

int aql_queue_create(struct aql_gpu_queue *queue,
		     struct file *kfd_filp, uint32_t gpu_id)
{
	struct kfd_ioctl_create_queue_args cq_args = {};
	int ret;
	int i;

	if (!queue || !kfd_filp)
		return -EINVAL;

	memset(queue, 0, sizeof(*queue));
	mutex_init(&queue->lock);
	queue->gpu_id = gpu_id;

	/* Find the render node for this GPU (needed for acquire_vm and mmap) */
	queue->render_minor = find_render_minor_for_gpu(gpu_id);
	if (queue->render_minor < 0) {
		aql_err("queue_create: no render node found for GPU %u", gpu_id);
		return queue->render_minor;
	}

	queue->drm_fd = -1;

	/* Step 0: Acquire VM — binds the KFD process to the GPU's DRM device.
	 * This initializes pdd->drm_priv which is required for memory allocs.
	 * We keep the DRM fd open for later mmap of GPU buffers. */
	ret = acquire_vm_for_gpu(kfd_filp, gpu_id, queue->render_minor,
				 &queue->drm_filp, &queue->drm_fd);
	if (ret) {
		aql_err("queue_create: acquire_vm failed for GPU %u: %d",
			gpu_id, ret);
		return ret;
	}

	/* Compute CWSR sizes from GPU properties (read from sysfs in
	 * aql_perf_discover_gpus). For now use the known values for
	 * gfx12 gfx1201 with 128 SIMDs, 2 SIMDs/CU, 8 arrays, 2 SA/engine.
	 * TODO: pass these from aql_perf_discover_gpus */
	{
		/* Read props from sysfs topology (node index found earlier) */
		u32 simd_count = 128, simd_per_cu = 2, array_count = 8;
		u32 simd_arrays_per_engine = 2, num_xcc = 1;
		u32 gfxv = 120001;

		/* Read actual values from sysfs */
		{
			char buf_str[64];
			struct file *f;
			loff_t pos = 0;
			ssize_t len;
			int node_id = -1;

			/* Find the topology node for this GPU */
			for (node_id = 0; node_id < 16; node_id++) {
				char path[128];
				snprintf(path, sizeof(path),
					 "/sys/class/kfd/kfd/topology/nodes/%d/gpu_id",
					 node_id);
				f = filp_open(path, O_RDONLY, 0);
				if (IS_ERR(f))
					continue;
				memset(buf_str, 0, sizeof(buf_str));
				len = kernel_read(f, buf_str, sizeof(buf_str)-1, &pos);
				filp_close(f, NULL);
				if (len > 0) {
					u32 id = 0;
					if (kstrtou32(strim(buf_str), 10, &id) == 0 &&
					    id == gpu_id)
						break;
				}
				pos = 0;
			}

			if (node_id < 16) {
				char path[128];
				/* Read properties file and parse key values */
				snprintf(path, sizeof(path),
					 "/sys/class/kfd/kfd/topology/nodes/%d/properties",
					 node_id);
				f = filp_open(path, O_RDONLY, 0);
				if (!IS_ERR(f)) {
					char *props_buf = kzalloc(4096, GFP_KERNEL);
					if (props_buf) {
						pos = 0;
						len = kernel_read(f, props_buf, 4095, &pos);
						if (len > 0) {
							props_buf[len] = '\0';
							parse_sysfs_u32(props_buf, "simd_count ", &simd_count);
							parse_sysfs_u32(props_buf, "simd_per_cu ", &simd_per_cu);
							parse_sysfs_u32(props_buf, "array_count ", &array_count);
							parse_sysfs_u32(props_buf, "simd_arrays_per_engine ", &simd_arrays_per_engine);
							parse_sysfs_u32(props_buf, "gfx_target_version ", &gfxv);
							parse_sysfs_u32(props_buf, "num_xcc ", &num_xcc);
						}
						kfree(props_buf);
					}
					filp_close(f, NULL);
				}
			}
		}

		compute_cwsr_sizes(gfxv, simd_count, simd_per_cu,
				   array_count, simd_arrays_per_engine,
				   num_xcc,
				   &queue->ctl_stack_size, &queue->cwsr_size);

		/* Compute total buffer allocation size: KFD's acquire_buffers
		 * computes total = (ctx_save_restore_size + debug_memory_size) * num_xcc
		 * and checks the buffer mapping is exactly that size.
		 * We report cwsr_size as ctx_save_restore_size, allocate total. */
		{
			u32 cu_num = simd_count / simd_per_cu / num_xcc;
			u32 wave_num = (gfxv < 100100) ?
				min(cu_num * 40, array_count / simd_arrays_per_engine * 512)
				: cu_num * 32;
			u32 debug_mem = ALIGN(wave_num * DEBUGGER_BYTES_PER_WAVE,
					      DEBUGGER_BYTES_ALIGN);
			/* cwsr_alloc_size = what KFD expects the buffer to be */
			queue->cwsr_alloc_size = ALIGN(
				(queue->cwsr_size + debug_mem) * num_xcc,
				PAGE_SIZE);
		}

		aql_debug("queue_create: GPU %u: ctl_stack_size=0x%x, cwsr_size=0x%x, cwsr_alloc=0x%x",
			 gpu_id, queue->ctl_stack_size, queue->cwsr_size,
			 queue->cwsr_alloc_size);
	}

	/* Step 1: Allocate GPU memory buffers.
	 * Most buffers use USERPTR (anonymous pages registered with KFD).
	 * Signal buffer uses GTT because KFD needs to pin wptr/rptr BOs
	 * to GART, which only works for GTT-type BOs, not USERPTR. */
	{
		struct {
			uint32_t size;
		} buf_config[AQL_QUEUE_NUM_BUFFERS] = {
			[AQL_BUF_RING]   = { AQL_QUEUE_RING_SIZE },
			[AQL_BUF_IB]     = { AQL_QUEUE_IB_SIZE },
			[AQL_BUF_DATA]   = { AQL_QUEUE_DATA_SIZE },
			[AQL_BUF_EOP]    = { AQL_QUEUE_EOP_SIZE },
			[AQL_BUF_SIGNAL] = { AQL_QUEUE_SIGNAL_SIZE },
			[AQL_BUF_CWSR]   = { queue->cwsr_alloc_size },
			[AQL_BUF_WPTR]   = { PAGE_SIZE },
		};

		/* Allocate GPU buffers.
		 * Ring, IB, wptr, and signal use GTT — GPU firmware/MES
		 * accesses ring buffer and IB via GART. USERPTR buffers
		 * may not be accessible to MES via GPUVM on VFIO passthrough.
		 * GTT buffers have cpu_addr=NULL (VM_PFNMAP), accessed via
		 * user_addr with kthread_use_mm + copy_to/from_user.
		 * Data, EOP, CWSR remain USERPTR (only CPU accesses these). */
		for (i = 0; i < AQL_QUEUE_NUM_BUFFERS; i++) {
			if (i == AQL_BUF_RING || i == AQL_BUF_IB ||
			    i == AQL_BUF_WPTR || i == AQL_BUF_SIGNAL) {
				/* GTT buffers: allocated by GPU driver, GART-pinnable */
				ret = alloc_gpu_buffer_gtt(kfd_filp, gpu_id,
							   &queue->bufs[i],
							   buf_config[i].size,
							   queue->drm_filp,
							   queue->drm_fd);
				if (ret)
					goto err_free_bufs;
			} else {
				/* USERPTR buffers: data, EOP, CWSR */
				ret = alloc_gpu_buffer(kfd_filp, gpu_id,
						       &queue->bufs[i],
						       buf_config[i].size, 0,
						       PAGE_KERNEL);
				if (ret)
					goto err_free_bufs;
			}
		}
	}

	aql_debug("queue_create: all buffers allocated for GPU %u", gpu_id);

	/* Step 2: Map all buffers to GPU */
	ret = map_gpu_buffers(kfd_filp, gpu_id, queue->bufs,
			      AQL_QUEUE_NUM_BUFFERS);
	if (ret)
		goto err_free_bufs;

	aql_debug("queue_create: all buffers mapped for GPU %u", gpu_id);

	/* Step 3: Initialize ring and IB pool structures.
	 * Ring and IB are GTT, so cpu_addr is NULL. Use user_addr instead.
	 * ring.base and ib_pool.base will hold USER VAs, accessed via
	 * kthread_use_mm + copy_to/from_user in the hot path. */
	ret = aql_ring_init(&queue->ring,
			    (void *)queue->bufs[AQL_BUF_RING].user_addr,
			    queue->bufs[AQL_BUF_RING].gpu_addr,
			    AQL_QUEUE_RING_SIZE);
	if (ret)
		goto err_unpin;

	/* Fill ring with HSA_PACKET_TYPE_INVALID (1) headers like ROCr does.
	 * MES uses the header to detect valid packets. Type 0 (VENDOR_SPECIFIC)
	 * in a zero-initialized ring slot could confuse the scheduler.
	 * Ring is GTT (user VA) — write via put_user from process context. */
	{
		unsigned long ring_user = queue->bufs[AQL_BUF_RING].user_addr;
		uint32_t num_slots = AQL_QUEUE_RING_SIZE / AQL_PACKET_SIZE;
		uint32_t s;
		/* First zero the entire ring buffer */
		uint64_t zero = 0;
		uint32_t off;
		for (off = 0; off < AQL_QUEUE_RING_SIZE; off += 8)
			put_user(zero, (uint64_t __user *)(ring_user + off));
		/* Then set header of each slot to INVALID (1) */
		for (s = 0; s < num_slots; s++) {
			uint16_t inv = 1;
			put_user(inv, (uint16_t __user *)(ring_user + s * AQL_PACKET_SIZE));
		}
		aql_debug("queue_create: filled %u ring slots with INVALID headers (GTT user VA)",
			 num_slots);
	}

	ret = aql_ib_pool_init(&queue->ib_pool,
			       (void *)queue->bufs[AQL_BUF_IB].user_addr,
			       queue->bufs[AQL_BUF_IB].gpu_addr,
			       AQL_QUEUE_IB_SIZE);
	if (ret)
		goto err_unpin;

	/* Set up data buffer and signal pointers */
	queue->data_buf = queue->bufs[AQL_BUF_DATA].cpu_addr;
	queue->data_gpu_addr = queue->bufs[AQL_BUF_DATA].gpu_addr;
	queue->data_size = AQL_QUEUE_DATA_SIZE;

	/* Signal buffer: use separate GTT buffer (AQL_BUF_SIGNAL).
	 * Must be a proper amd_signal_t structure (64 bytes, 64-byte aligned):
	 *   offset  0: kind (int64)  = AMD_SIGNAL_KIND_USER (1)
	 *   offset  8: value (int64) = 1 (GPU decrements to 0 on completion)
	 *   offset 16: event_mailbox_ptr (uint64) = 0 (no event)
	 *   offset 24: event_id (uint32) = 0
	 *   offset 28: reserved1 (uint32) = 0
	 *   offset 32: start_ts (uint64) = 0
	 *   offset 40: end_ts (uint64) = 0
	 *   offset 48: queue_ptr (uint64) = 0
	 *   offset 56: reserved3[2] (uint32[2]) = 0
	 * completion_signal in AQL packet = GPU VA of this amd_signal_t.
	 * MES reads kind to determine signal type, decrements value. */
	{
		unsigned long sig_user_base = queue->bufs[AQL_BUF_SIGNAL].user_addr;
		uint64_t zero = 0;
		int off;

		queue->signal = (volatile uint64_t *)(sig_user_base + 8);
		queue->signal_gpu_addr = queue->bufs[AQL_BUF_SIGNAL].gpu_addr;

		/* Zero the entire 64-byte amd_signal_t structure first */
		for (off = 0; off < 64; off += 8)
			put_user(zero, (uint64_t __user *)(sig_user_base + off));
		/* Set kind at offset 0: AMD_SIGNAL_KIND_USER = 1 */
		put_user((uint64_t)AQL_SIGNAL_KIND_USER, (uint64_t __user *)sig_user_base);
		/* Set value = 1 at offset 8 */
		put_user((uint64_t)1, (uint64_t __user *)(sig_user_base + 8));

		aql_debug("queue_create: signal buf gpu=0x%llx user=0x%lx (amd_signal_t: kind=%d, value=1)",
			 queue->signal_gpu_addr, sig_user_base, AQL_SIGNAL_KIND_USER);
	}

	/* Save mm reference for kthread_use_mm in hot path
	 * (needed to write wptr in GTT buffer from work queue context) */
	queue->mm = current->mm;
	mmget(queue->mm);

	/* wptr/rptr live in the GTT buffer (AQL_BUF_WPTR).
	 * GTT BOs can be pinned to GART, which KFD requires.
	 * cpu_addr is NULL for GTT — we access via user_addr + kthread_use_mm.
	 * Store the user VA for hot path access. */
	queue->wptr_addr = (volatile uint64_t *)(queue->bufs[AQL_BUF_WPTR].user_addr);
	queue->rptr_addr = (volatile uint64_t *)(queue->bufs[AQL_BUF_WPTR].user_addr + 64);
	/* Initialize wptr/rptr via access_process_vm since we can't directly
	 * access GTT pages from kernel. We're in process context here. */
	{
		uint64_t zero = 0;
		put_user(zero, (uint64_t __user *)queue->bufs[AQL_BUF_WPTR].user_addr);
		put_user(zero, (uint64_t __user *)(queue->bufs[AQL_BUF_WPTR].user_addr + 64));
	}

	/* Step 4b: Enable compute_perfcount_enable via profiler ioctl.
	 * This must happen BEFORE CREATE_QUEUE so the MQD is initialized
	 * with compute_perfcount_enable=1 (via profiler_process check in
	 * init_mqd). On upstream kernels without the profiler ioctl, we
	 * fall back to the kernel patch. Failure here is non-fatal. */
	{
		int pret = profiler_lock_and_enable(kfd_filp, gpu_id);
		if (!pret)
			aql_debug("queue_create: perfcount enabled via profiler ioctl");
		else
			aql_debug("queue_create: profiler ioctl returned %d, "
				 "using kernel patch fallback", pret);
	}

	/* Step 5: Create AQL compute queue via KFD */
	cq_args.gpu_id = gpu_id;
	cq_args.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
	cq_args.ring_base_address = queue->bufs[AQL_BUF_RING].gpu_addr;
	cq_args.ring_size = AQL_QUEUE_RING_SIZE;
	cq_args.queue_percentage = 100;
	cq_args.queue_priority = 7; /* Normal priority (0-15 range) */
	cq_args.eop_buffer_address = queue->bufs[AQL_BUF_EOP].gpu_addr;
	cq_args.eop_buffer_size = AQL_QUEUE_EOP_SIZE;

	/* wptr/rptr addresses — use GTT buffer's gpu_addr (== user_addr for
	 * KFD's kfd_queue_buffer_get VM lookup) */
	cq_args.write_pointer_address = queue->bufs[AQL_BUF_WPTR].gpu_addr;
	cq_args.read_pointer_address = queue->bufs[AQL_BUF_WPTR].gpu_addr + 64;

	/* CWSR (Context Save/Restore) — required by KFD for compute queues */
	cq_args.ctx_save_restore_address = queue->bufs[AQL_BUF_CWSR].user_addr;
	cq_args.ctx_save_restore_size = queue->cwsr_size;
	cq_args.ctl_stack_size = queue->ctl_stack_size;

	aql_debug("queue_create: CREATE_QUEUE args: ring=0x%llx size=0x%x "
		 "wptr=0x%llx rptr=0x%llx eop=0x%llx eop_size=0x%x "
		 "cwsr=0x%llx cwsr_size=0x%x ctl_stack=0x%x",
		 cq_args.ring_base_address, cq_args.ring_size,
		 cq_args.write_pointer_address, cq_args.read_pointer_address,
		 cq_args.eop_buffer_address, cq_args.eop_buffer_size,
		 cq_args.ctx_save_restore_address, cq_args.ctx_save_restore_size,
		 cq_args.ctl_stack_size);

	ret = kfd_bridge_ioctl(kfd_filp, AMDKFD_IOC_CREATE_QUEUE,
			       &cq_args, sizeof(cq_args));
	if (ret) {
		aql_err("queue_create: CREATE_QUEUE failed: %d", ret);
		goto err_unpin;
	}

	queue->queue_id = cq_args.queue_id;
	queue->doorbell_offset = cq_args.doorbell_offset;

	aql_info("queue_create: queue %u created for GPU %u, doorbell_offset=0x%llx",
		 queue->queue_id, gpu_id, queue->doorbell_offset);

	/* Step 6: Map doorbell for kernel access
	 * KFD doorbell mmap uses io_remap_pfn_range (VM_PFNMAP, noncached).
	 * Can't pin_user_pages — no struct pages.  Keep the user VA mapped
	 * and write via kthread_use_mm + put_user in the hot path.
	 *
	 * Mmap size must equal kfd_doorbell_process_slice():
	 *   MES GPUs: roundup(8 * 1024, PAGE_SIZE) = 8192
	 *   Non-MES:  roundup(doorbell_size * 1024, PAGE_SIZE)
	 * We use the MES formula since gfx12 is MES-enabled.
	 */
	{
		/* KFD doorbell mmap: offset = KFD_MMAP_TYPE_DOORBELL | gpu_id */
		uint64_t db_mmap_offset = KFD_MMAP_TYPE_DOORBELL
					| KFD_MMAP_GPU_ID(gpu_id);

		/* Doorbell process slice: roundup(8 * 1024, PAGE_SIZE) */
		uint32_t db_slice_size = roundup(8 * 1024, PAGE_SIZE);

		/* doorbell_offset_in_process is in the low bits of doorbell_offset
		 * (bits below KFD_MMAP_GPU_ID_SHIFT) */
		queue->doorbell_byte_offset = (uint32_t)(queue->doorbell_offset
					& ((1ULL << KFD_MMAP_GPU_ID_SHIFT) - 1));
		queue->doorbell_mmap_size = db_slice_size;

		aql_debug("queue_create: doorbell mmap offset=0x%llx size=%u byte_offset=%u",
			 db_mmap_offset, db_slice_size, queue->doorbell_byte_offset);

		queue->doorbell_user_addr = vm_mmap(kfd_filp, 0, db_slice_size,
						    PROT_READ | PROT_WRITE,
						    MAP_SHARED, db_mmap_offset);
		if (IS_ERR_VALUE(queue->doorbell_user_addr)) {
			aql_err("queue_create: doorbell vm_mmap failed: %ld",
				(long)queue->doorbell_user_addr);
			ret = (int)queue->doorbell_user_addr;
			queue->doorbell_user_addr = 0;
			goto err_destroy_queue;
		}

		aql_debug("queue_create: doorbell mapped at user VA 0x%lx",
			 queue->doorbell_user_addr);

		/* Get doorbell physical address via follow_pfnmap and ioremap_wc.
		 * put_user to VM_PFNMAP pages fails from kthread_use_mm context,
		 * so we need a kernel MMIO mapping for the hot path.
		 * Force PTE population first by reading from the VA. */
		{
			unsigned long db_uva = queue->doorbell_user_addr
						+ queue->doorbell_byte_offset;
			struct follow_pfnmap_args pfnmap_args = {};
			struct vm_area_struct *vma;
			phys_addr_t db_phys;
			uint64_t db_val;
			int gu_ret;

			/* Fault-in the doorbell PTE */
			gu_ret = get_user(db_val, (uint64_t __user *)db_uva);
			if (gu_ret) {
				aql_err("queue_create: doorbell fault-in failed: %d", gu_ret);
				ret = gu_ret;
				goto err_unmap_doorbell;
			}
			aql_debug("queue_create: doorbell fault-in OK, val=0x%llx", db_val);

			/* Get physical address */
			mmap_read_lock(current->mm);
			vma = find_vma(current->mm, db_uva);
			if (!vma || vma->vm_start > db_uva) {
				mmap_read_unlock(current->mm);
				aql_err("queue_create: doorbell VMA not found");
				ret = -EFAULT;
				goto err_unmap_doorbell;
			}
			pfnmap_args.vma = vma;
			pfnmap_args.address = db_uva;
			ret = follow_pfnmap_start(&pfnmap_args);
			if (ret) {
				mmap_read_unlock(current->mm);
				aql_err("queue_create: follow_pfnmap_start failed: %d", ret);
				goto err_unmap_doorbell;
			}
			db_phys = (pfnmap_args.pfn << PAGE_SHIFT)
				  | (db_uva & ~PAGE_MASK);
			follow_pfnmap_end(&pfnmap_args);
			mmap_read_unlock(current->mm);

			/* ioremap_wc for kernel MMIO access (works from any context) */
			queue->doorbell_kaddr = ioremap_wc(db_phys & PAGE_MASK,
							    PAGE_SIZE);
			if (!queue->doorbell_kaddr) {
				aql_err("queue_create: doorbell ioremap_wc failed for phys 0x%llx",
					(unsigned long long)db_phys);
				ret = -ENOMEM;
				goto err_unmap_doorbell;
			}
			queue->doorbell_kaddr = (void __iomem *)
				((unsigned long)queue->doorbell_kaddr
				 + (db_phys & ~PAGE_MASK));
			aql_debug("queue_create: doorbell phys=0x%llx kaddr=%px",
				 (unsigned long long)db_phys, queue->doorbell_kaddr);
		}
	}

	queue->initialized = true;

	/* Initial doorbell write: tell MES wptr=0 (no packets) to establish
	 * known doorbell state. Without this, MES may have stale doorbell
	 * value from a previous queue at the same doorbell slot. */
	writeq(0, queue->doorbell_kaddr);
	aql_info("queue_create: GPU %u queue %u fully initialized",
		 gpu_id, queue->queue_id);

	return 0;

err_unmap_doorbell:
	if (queue->doorbell_kaddr) {
		iounmap((void __iomem *)((unsigned long)queue->doorbell_kaddr
					 & PAGE_MASK));
		queue->doorbell_kaddr = NULL;
	}
	if (queue->doorbell_user_addr) {
		vm_munmap(queue->doorbell_user_addr, queue->doorbell_mmap_size);
		queue->doorbell_user_addr = 0;
	}
err_destroy_queue:
	{
		struct kfd_ioctl_destroy_queue_args dq_args = {};
		dq_args.queue_id = cq_args.queue_id;
		kfd_bridge_ioctl(kfd_filp, AMDKFD_IOC_DESTROY_QUEUE,
				 &dq_args, sizeof(dq_args));
	}
	profiler_unlock_and_disable(kfd_filp, gpu_id);
err_unpin:
	unmap_gpu_buffers(kfd_filp, gpu_id, queue->bufs, AQL_QUEUE_NUM_BUFFERS);
err_free_bufs:
	for (i = AQL_QUEUE_NUM_BUFFERS - 1; i >= 0; i--)
		free_gpu_buffer(kfd_filp, gpu_id, &queue->bufs[i]);

	if (queue->mm) {
		mmput(queue->mm);
		queue->mm = NULL;
	}

	if (queue->drm_fd >= 0) {
		close_fd(queue->drm_fd);
		queue->drm_fd = -1;
	}
	queue->drm_filp = NULL;

	return ret;
}

void aql_queue_destroy(struct aql_gpu_queue *queue,
		       struct file *kfd_filp)
{
	int i;

	if (!queue || !queue->initialized)
		return;

	aql_info("queue_destroy: destroying queue %u for GPU %u",
		 queue->queue_id, queue->gpu_id);

	/* Wait for GPU to finish any in-flight work before destroying queue.
	 * MES needs time to fully process the last packet and go idle.
	 * Without this delay, DESTROY_QUEUE right after a submission can
	 * crash MES on GFX12.
	 *
	 * NOTE: We skip rptr diagnostic read here because aql_queue_destroy
	 * may be called from user process context (rmmod), where
	 * kthread_use_mm is unsafe. The rptr info is just diagnostic. */
	aql_debug("queue_destroy: wptr_dispatch=%llu before destroy, waiting 100ms",
		 (unsigned long long)(queue->ring.wptr / AQL_PACKET_SIZE));
	msleep(100);

	/* Borrow queue->mm for cleanup operations.
	 * vm_munmap and free_gpu_buffer need the correct mm context.
	 * aql_queue_destroy may be called from user process context (rmmod)
	 * where current->mm != queue->mm. We need to use kthread_use_mm,
	 * but that's only safe from kernel threads. From user processes,
	 * we must save/restore current->mm manually.
	 *
	 * IMPORTANT: Only use kthread_use_mm from kernel threads (PF_KTHREAD).
	 * From user processes, skip mm-dependent operations. */
	{
		bool is_kthread = !!(current->flags & PF_KTHREAD);
		bool need_mm_switch = queue->mm && (current->mm != queue->mm);
		bool have_mm = false;

		if (need_mm_switch && is_kthread) {
			kthread_use_mm(queue->mm);
			have_mm = true;
		} else if (need_mm_switch && !is_kthread) {
			/* From user process (e.g., rmmod): skip vm_munmap calls.
			 * The mm will be cleaned up when mmput drops the ref. */
			aql_debug("queue_destroy: running from user process, "
				 "skipping vm_munmap (mm cleanup via mmput)");
		}

		/* Step 1: Destroy queue */
		{
			struct kfd_ioctl_destroy_queue_args args = {};
			args.queue_id = queue->queue_id;
			kfd_bridge_ioctl(kfd_filp, AMDKFD_IOC_DESTROY_QUEUE,
					 &args, sizeof(args));
		}

		/* Step 1b: Unlock profiler (release perfcount_enable) */
		profiler_unlock_and_disable(kfd_filp, queue->gpu_id);

		/* Step 2: Unmap doorbell */
		if (queue->doorbell_kaddr) {
			iounmap((void __iomem *)((unsigned long)queue->doorbell_kaddr
						 & PAGE_MASK));
			queue->doorbell_kaddr = NULL;
		}
		if (queue->doorbell_user_addr && (have_mm || current->mm == queue->mm)) {
			vm_munmap(queue->doorbell_user_addr, queue->doorbell_mmap_size);
			queue->doorbell_user_addr = 0;
		}

		/* Step 3: Unmap from GPU */
		unmap_gpu_buffers(kfd_filp, queue->gpu_id, queue->bufs,
				  AQL_QUEUE_NUM_BUFFERS);

		/* Step 4: Free GPU memory (includes vunmap + unpin) */
		for (i = 0; i < AQL_QUEUE_NUM_BUFFERS; i++)
			free_gpu_buffer(kfd_filp, queue->gpu_id, &queue->bufs[i]);

		if (have_mm)
			kthread_unuse_mm(queue->mm);
	}

	/* Step 5: Release mm reference */
	if (queue->mm) {
		mmput(queue->mm);
		queue->mm = NULL;
	}

	/* Step 6: Release DRM render node */
	if (queue->drm_fd >= 0) {
		close_fd(queue->drm_fd);
		queue->drm_fd = -1;
	}
	queue->drm_filp = NULL;

	queue->initialized = false;
	aql_info("queue_destroy: queue %u destroyed", queue->queue_id);
}

/* ========================================================================
 * Hot path: submit + wait (no ioctls, no mm needed)
 * ======================================================================== */

int aql_queue_submit(struct aql_gpu_queue *queue,
		     const uint32_t *pm4_data, uint32_t pm4_size_dw,
		     uint64_t *out_dispatch_idx)
{
	aql_pm4_ib_packet_t pkt;
	void *ib_user;           /* IB user VA (GTT buffer) */
	uint64_t ib_gpu;
	uint64_t old_dispatch_idx; /* saved BEFORE ring write, like ROCr */
	uint32_t ib_local[256];  /* kernel-side IB staging buffer (max 1KB) */
	bool need_mm;
	int ret;

	/*
	 * Queue submit is the low-level transport boundary:
	 * - allocates IB space in queue-owned GTT pool
	 * - writes PM4 stream plus trailing completion WRITE_DATA
	 * - enqueues one vendor AQL packet into ring
	 * - updates wptr + doorbell using ROCr-compatible ordering
	 *
	 * Caller receives expected dispatch index for wait().
	 */
	if (!queue || !queue->initialized || !pm4_data || !pm4_size_dw)
		return -EINVAL;

	mutex_lock(&queue->lock);

	/* Step 1: Allocate IB space from pool and build PM4 in kernel buffer.
	 * IB pool is GTT (user VA), so we build the PM4 data in ib_local[]
	 * first, then copy_to_user to the GTT buffer.
	 * Reserve 6 extra DWORDs for WRITE_DATA at end of IB. */
	{
		uint32_t alloc_dw = pm4_size_dw + 6; /* +6 for WRITE_DATA */
		if (alloc_dw > ARRAY_SIZE(ib_local)) {
			aql_err("queue_submit: IB too large (%u > %lu DW)",
				alloc_dw, ARRAY_SIZE(ib_local));
			mutex_unlock(&queue->lock);
			return -E2BIG;
		}
		ret = aql_ib_pool_alloc(&queue->ib_pool, alloc_dw * 4,
					&ib_user, &ib_gpu);
	}
	if (ret) {
		aql_err("queue_submit: IB pool alloc failed");
		mutex_unlock(&queue->lock);
		return ret;
	}

	/* Build IB data in kernel staging buffer */
	memcpy(ib_local, pm4_data, pm4_size_dw * 4);

	/* Append WRITE_DATA PM4 to signal completion via signal buffer.
	 * Writes 0 to signal value (offset +8 in amd_signal_t) so the
	 * wait path can detect completion even if MES signal handling fails. */
	{
		uint64_t sig_value_gpu = queue->signal_gpu_addr + 8;
		ib_local[pm4_size_dw + 0] = (3u << 30) | (0x37 << 8) | (4 << 16);
		ib_local[pm4_size_dw + 1] = (5u << 8) | (1u << 20);
		ib_local[pm4_size_dw + 2] = (uint32_t)(sig_value_gpu & 0xFFFFFFFC);
		ib_local[pm4_size_dw + 3] = (uint32_t)(sig_value_gpu >> 32);
		ib_local[pm4_size_dw + 4] = 0;
		ib_local[pm4_size_dw + 5] = 0;
		pm4_size_dw += 6;
	}

	/* Copy IB data from kernel staging buffer to GTT user VA.
	 * GTT memory is coherent with GPU (no cache flush needed). */
	need_mm = !current->mm || (current->mm != queue->mm);
	if (need_mm)
		kthread_use_mm(queue->mm);
	ret = copy_to_user((void __user *)ib_user, ib_local, pm4_size_dw * 4);
	if (need_mm)
		kthread_unuse_mm(queue->mm);
	if (ret) {
		aql_err("queue_submit: IB copy_to_user failed: %d", ret);
		mutex_unlock(&queue->lock);
		return -EFAULT;
	}

	/* Step 2: Reset signal value to 1 before each submission.
	 * MES decrements signal value to 0 on completion. */
	{
		bool need_mm_sig = !current->mm || (current->mm != queue->mm);
		if (need_mm_sig)
			kthread_use_mm(queue->mm);
		put_user((uint64_t)1, (uint64_t __user *)queue->signal);
		if (need_mm_sig)
			kthread_unuse_mm(queue->mm);
	}

	/* Step 3: Build AQL PM4 IB packet with real signal.
	 * completion_signal = GPU VA of amd_signal_t struct.
	 * MES reads kind at +0, decrements value at +8. */
	ret = aql_build_pm4_ib_packet(&pkt, ib_gpu, pm4_size_dw,
				      queue->signal_gpu_addr);
	if (ret) {
		mutex_unlock(&queue->lock);
		return ret;
	}

	/* Step 4: Save slot offset and OLD dispatch index for header-polling
	 * and doorbell write. CRITICAL: ROCr writes the OLD write_idx to both
	 * wptr memory and doorbell (the index of the slot being filled, not
	 * the next free slot). AddWriteIndexAcqRel(1) returns the OLD value.
	 * For first packet: dispatch_idx=0, meaning "I filled slot 0".
	 * We must capture this BEFORE advancing wptr. */
	queue->last_submit_slot_offset = queue->ring.wptr & (queue->ring.size - 1);
	old_dispatch_idx = queue->ring.wptr / AQL_PACKET_SIZE;

	/* Check if ring is full */
	if (queue->ring.wptr - queue->ring.rptr >= queue->ring.size) {
		aql_err("queue_submit: ring full");
		mutex_unlock(&queue->lock);
		return -ENOSPC;
	}

	/* Write AQL packet to ring via copy_to_user (ring is GTT user VA).
	 * ROCr convention: write packet body (bytes 4..63) FIRST, then
	 * write header+ven_hdr (first 4 bytes) as atomic uint32_t store.
	 * This ensures MES sees a complete packet when it checks the header.
	 * Memory barrier between body and header for ordering. */
	{
		unsigned long slot_user = (unsigned long)queue->ring.base +
					  queue->last_submit_slot_offset;
		uint32_t hdr_dw = *(uint32_t *)&pkt; /* header(16b) + ven_hdr(16b) */

		if (need_mm)
			kthread_use_mm(queue->mm);

		/* Copy body first (bytes 4..63) */
		ret = copy_to_user((void __user *)(slot_user + 4),
				   ((uint8_t *)&pkt) + 4,
				   AQL_PACKET_SIZE - 4);
		smp_wmb(); /* ensure body visible before header */

		/* Write header+ven_hdr as 32-bit store (matches ROCr atomic store) */
		if (!ret)
			ret = put_user(hdr_dw, (uint32_t __user *)slot_user);

		if (need_mm)
			kthread_unuse_mm(queue->mm);

		if (ret) {
			aql_err("queue_submit: ring packet copy_to_user failed: %d", ret);
			mutex_unlock(&queue->lock);
			return -EFAULT;
		}
	}

	/* Advance ring write pointer */
	queue->ring.wptr += AQL_PACKET_SIZE;
	/* GTT ring is coherent with GPU — no cache flush needed. */

	/* Step 5: Update write pointer + ring doorbell.
	 *
	 * CRITICAL: AQL queues use dispatch indices (packet count), not byte
	 * offsets. wptr in GART = NEW value (total packets submitted).
	 * doorbell = OLD value (index of slot just filled).
	 * MES reads wptr from GART to determine work; doorbell wakes MES. */
	{
		bool need_mm = !current->mm || (current->mm != queue->mm);
		uint64_t new_dispatch_idx = queue->ring.wptr / AQL_PACKET_SIZE;
		uint64_t doorbell_val = old_dispatch_idx;

		/* Write wptr via put_user (needs mm for GTT user VA) */
		if (need_mm)
			kthread_use_mm(queue->mm);
		put_user(new_dispatch_idx, (uint64_t __user *)queue->wptr_addr);
		if (need_mm)
			kthread_unuse_mm(queue->mm);

		/* Memory barrier before doorbell */
		smp_wmb();

		/* Ring doorbell via ioremap'd MMIO.
		 * Doorbell value = OLD dispatch index (slot just filled). */
		writeq(doorbell_val, queue->doorbell_kaddr);

		aql_debug("queue_submit: gpu=%u pm4=%u DW dispatch=%llu",
			  queue->gpu_id, pm4_size_dw, new_dispatch_idx);
	}

	/* Return the dispatch index this caller should wait for.
	 * This MUST be captured under the queue lock, before another
	 * concurrent submitter advances wptr further. */
	if (out_dispatch_idx)
		*out_dispatch_idx = queue->ring.wptr / AQL_PACKET_SIZE;

	mutex_unlock(&queue->lock);
	return 0;
}

int aql_queue_wait(struct aql_gpu_queue *queue,
		   uint64_t expected_dispatch_idx,
		   unsigned long timeout_ms)
{
	unsigned long deadline;
	int spins = 0;
	uint64_t rptr_val = 0;
	bool need_mm;

	/*
	 * Completion detection is rptr-based:
	 * - MES increments queue rptr after consuming packets
	 * - wait succeeds once rptr >= expected dispatch index
	 * - on timeout, log slot/rptr diagnostics for ring triage
	 */
	if (!queue || !queue->initialized)
		return -EINVAL;

	/* Rptr-based completion detection.
	 *
	 * MES advances rptr (dispatch index) after processing each AQL packet.
	 * We wait until rptr >= the expected dispatch index returned by
	 * aql_queue_submit(). rptr is in GTT user VA memory, accessed via
	 * get_user + kthread_use_mm. GTT is coherent -- no cache flush needed. */

	need_mm = !current->mm || (current->mm != queue->mm);

	deadline = jiffies + msecs_to_jiffies(timeout_ms);

	while (time_before(jiffies, deadline)) {
		if (need_mm)
			kthread_use_mm(queue->mm);
		get_user(rptr_val, (uint64_t __user *)queue->rptr_addr);
		if (need_mm)
			kthread_unuse_mm(queue->mm);

		if (rptr_val >= expected_dispatch_idx) {
			aql_debug("queue_wait: completion detected! rptr=%llu >= expected=%llu (after %d polls)",
				 (unsigned long long)rptr_val,
				 (unsigned long long)expected_dispatch_idx, spins);
			/* Update local rptr tracking from GPU rptr value.
			 * Use rptr_val (not wptr) to avoid overwriting state
			 * from concurrent submissions. */
			queue->ring.rptr = rptr_val * AQL_PACKET_SIZE;
			return 0;
		}

		/* Progressive backoff */
		if (++spins < 100) {
			cpu_relax();
		} else if (spins < 1000) {
			udelay(1);
		} else {
			usleep_range(10, 50);
		}
	}

	/* Timeout — collect diagnostics */
	{
		uint16_t hdr_val = 0;
		unsigned long slot_user = (unsigned long)queue->ring.base +
					  queue->last_submit_slot_offset;

		if (need_mm)
			kthread_use_mm(queue->mm);
		get_user(rptr_val, (uint64_t __user *)queue->rptr_addr);
		get_user(hdr_val, (uint16_t __user *)slot_user);
		if (need_mm)
			kthread_unuse_mm(queue->mm);

		aql_err("queue_wait: timeout after %lu ms, rptr=%llu expected=%llu slot_header=0x%04x wptr_bytes=%llu",
			timeout_ms,
			(unsigned long long)rptr_val,
			(unsigned long long)expected_dispatch_idx,
			hdr_val,
			(unsigned long long)queue->ring.wptr);
	}
	return -ETIMEDOUT;
}

int aql_queue_get_data_buffer(struct aql_gpu_queue *queue,
			      void **out_cpu, uint64_t *out_gpu)
{
	if (!queue || !queue->initialized || !out_cpu || !out_gpu)
		return -EINVAL;

	*out_cpu = queue->data_buf;
	*out_gpu = queue->data_gpu_addr;
	return 0;
}
