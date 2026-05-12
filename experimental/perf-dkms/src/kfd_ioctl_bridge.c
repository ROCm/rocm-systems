/*
 * kfd_ioctl_bridge.c - Bridge for calling KFD ioctls from kernel space
 *
 * Uses vm_mmap to create a temporary anonymous userspace page, copies
 * kernel arguments to it, calls the ioctl handler (which does
 * copy_from_user internally), then copies results back.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/mman.h>
#include <linux/uaccess.h>
#include <linux/sched/mm.h>

#include "kfd_ioctl_bridge.h"
#include "aql_perf.h"

int kfd_bridge_ioctl(struct file *kfd_filp, unsigned int cmd,
		     void *kernel_args, size_t args_size)
{
	unsigned long upage;
	int ret;

	/*
	 * KFD ioctl bridge pattern:
	 * KFD uAPI handlers expect user pointers and internally perform
	 * copy_from_user/copy_to_user. Kernel callers therefore marshal args
	 * through a temporary user mapping and invoke unlocked_ioctl directly.
	 *
	 * Preconditions:
	 * - current task must have an mm (process context)
	 * - argument blob must fit in one temporary page
	 */
	if (!kfd_filp || !kernel_args || !args_size)
		return -EINVAL;

	if (args_size > PAGE_SIZE)
		return -EINVAL;

	/* Must have a valid mm (process context) */
	if (!current->mm)
		return -ENOSYS;

	/* Allocate an anonymous userspace page */
	upage = vm_mmap(NULL, 0, PAGE_SIZE,
			PROT_READ | PROT_WRITE,
			MAP_ANONYMOUS | MAP_PRIVATE, 0);
	if (IS_ERR_VALUE(upage)) {
		aql_err("kfd_bridge: vm_mmap failed: %ld", (long)upage);
		return (int)upage;
	}

	/* Copy kernel args to userspace page */
	if (copy_to_user((void __user *)upage, kernel_args, args_size)) {
		aql_err("kfd_bridge: copy_to_user failed");
		ret = -EFAULT;
		goto out;
	}

	/* Call the ioctl handler — it will do copy_from_user internally */
	ret = kfd_filp->f_op->unlocked_ioctl(kfd_filp, cmd, upage);
	if (ret) {
		aql_debug("kfd_bridge: ioctl 0x%x returned %d", cmd, ret);
		goto out;
	}

	/* Copy results back to kernel space (for ioctls that write output) */
	if (_IOC_DIR(cmd) & _IOC_READ) {
		if (copy_from_user(kernel_args, (void __user *)upage, args_size)) {
			aql_err("kfd_bridge: copy_from_user failed");
			ret = -EFAULT;
			goto out;
		}
	}

out:
	vm_munmap(upage, PAGE_SIZE);
	return ret;
}

int kfd_bridge_ioctl_nested(struct file *kfd_filp, unsigned int cmd,
			    void *kernel_args, size_t args_size,
			    void *nested_data, size_t nested_size,
			    size_t nested_ptr_offset)
{
	unsigned long upage;
	uint64_t nested_uaddr;
	int ret;

	if (!kfd_filp || !kernel_args || !args_size)
		return -EINVAL;

	if (!nested_data || !nested_size)
		return -EINVAL;

	if (args_size + nested_size > PAGE_SIZE)
		return -EINVAL;

	/* The pointer field must be within the args struct */
	if (nested_ptr_offset + sizeof(uint64_t) > args_size)
		return -EINVAL;

	if (!current->mm)
		return -ENOSYS;

	/* Allocate userspace page */
	upage = vm_mmap(NULL, 0, PAGE_SIZE,
			PROT_READ | PROT_WRITE,
			MAP_ANONYMOUS | MAP_PRIVATE, 0);
	if (IS_ERR_VALUE(upage)) {
		aql_err("kfd_bridge_nested: vm_mmap failed: %ld", (long)upage);
		return (int)upage;
	}

	/* Patch the nested pointer field in the args to point to the
	 * nested data location in the user page (right after the args) */
	nested_uaddr = (uint64_t)(upage + args_size);
	*(uint64_t *)((char *)kernel_args + nested_ptr_offset) = nested_uaddr;

	/* Copy args struct to beginning of user page */
	if (copy_to_user((void __user *)upage, kernel_args, args_size)) {
		ret = -EFAULT;
		goto out;
	}

	/* Copy nested data right after the args */
	if (copy_to_user((void __user *)(upage + args_size),
			 nested_data, nested_size)) {
		ret = -EFAULT;
		goto out;
	}

	/* Call ioctl */
	ret = kfd_filp->f_op->unlocked_ioctl(kfd_filp, cmd, upage);
	if (ret) {
		aql_debug("kfd_bridge_nested: ioctl 0x%x returned %d", cmd, ret);
		goto out;
	}

	/* Copy results back */
	if (_IOC_DIR(cmd) & _IOC_READ) {
		if (copy_from_user(kernel_args, (void __user *)upage, args_size)) {
			ret = -EFAULT;
			goto out;
		}
	}

out:
	vm_munmap(upage, PAGE_SIZE);
	return ret;
}
