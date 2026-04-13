/*
 * kfd_ioctl_bridge.h - Bridge for calling KFD ioctls from kernel space
 *
 * Provides a mechanism to call KFD ioctl handlers from kernel space by
 * allocating a temporary userspace page via vm_mmap, copying kernel args
 * to it, invoking the ioctl handler, and copying results back.
 *
 * This eliminates the need for exported KFD wrapper functions entirely.
 * Must be called from process context with valid current->mm.
 */

#ifndef _KFD_IOCTL_BRIDGE_H
#define _KFD_IOCTL_BRIDGE_H

#include <linux/types.h>
#include <linux/fs.h>

/**
 * kfd_bridge_ioctl - Call a KFD ioctl from kernel space
 * @kfd_filp: Open file pointer to /dev/kfd
 * @cmd: Ioctl command number (e.g., AMDKFD_IOC_ALLOC_MEMORY_OF_GPU)
 * @kernel_args: Pointer to kernel-space ioctl arguments structure
 * @args_size: Size of the arguments structure in bytes
 *
 * Allocates a temporary userspace page via vm_mmap, copies the kernel
 * args to it (so copy_from_user in the ioctl handler succeeds), calls
 * the ioctl handler, then copies results back to kernel_args.
 *
 * Must be called from process context with valid current->mm.
 *
 * Returns: 0 on success, negative error code on failure
 */
int kfd_bridge_ioctl(struct file *kfd_filp, unsigned int cmd,
		     void *kernel_args, size_t args_size);

/**
 * kfd_bridge_ioctl_nested - Call a KFD ioctl with nested userspace pointers
 * @kfd_filp: Open file pointer to /dev/kfd
 * @cmd: Ioctl command number
 * @kernel_args: Pointer to kernel-space ioctl arguments structure
 * @args_size: Size of the arguments structure in bytes
 * @nested_data: Data to place after the args struct in the user page
 * @nested_size: Size of the nested data in bytes
 * @nested_ptr_offset: Byte offset within kernel_args of the pointer field
 *                     that should point to the nested data
 *
 * For ioctls like MAP_MEMORY_TO_GPU that have embedded userspace pointers
 * (e.g., device_ids_array_ptr), this function places the nested data after
 * the args struct in the same user page and patches the pointer field to
 * point to it.
 *
 * The total of args_size + nested_size must fit in PAGE_SIZE.
 *
 * Returns: 0 on success, negative error code on failure
 */
int kfd_bridge_ioctl_nested(struct file *kfd_filp, unsigned int cmd,
			    void *kernel_args, size_t args_size,
			    void *nested_data, size_t nested_size,
			    size_t nested_ptr_offset);

#endif /* _KFD_IOCTL_BRIDGE_H */
