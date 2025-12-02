/*
 * kfd_test.c - KFD ioctl test implementation for perf-dkms
 *
 * This module demonstrates calling kfd_ioctl_get_version directly from kernel space
 * using the exported function interface.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/fs.h>
#include <linux/file.h>
#include <uapi/linux/kfd_ioctl.h>

#include "amdgpu_pmu.h"
#include "kfd_test.h"

/* Forward declarations for KFD structures */
struct kfd_process;

/* External declarations for the exported KFD functions */
extern long kfd_ioctl_get_version(struct file *filep, struct kfd_process *p, void *data);

/* Global test result */
static struct kfd_test_result kfd_result = { 0 };

/**
 * kfd_test_get_version - Test KFD ioctl get version call
 *
 * This function demonstrates how to call kfd_ioctl_get_version properly from kernel space
 * by opening /dev/kfd to get a valid file pointer and kfd_process.
 *
 * Returns: 0 on success, negative error code on failure
 */
int kfd_test_get_version(void)
{
	struct file *kfd_file = NULL;
	struct kfd_ioctl_get_version_args version_args;
	struct kfd_process *process;
	long ret = 0;

	/* Clear previous results */
	memset(&kfd_result, 0, sizeof(kfd_result));
	memset(&version_args, 0, sizeof(version_args));

	pmu_info("KFD Test: Opening /dev/kfd to get valid file and process\n");

	/* Open /dev/kfd which calls kfd_open() and sets up filep->private_data */
	kfd_file = filp_open("/dev/kfd", O_RDWR, 0);
	if (IS_ERR(kfd_file)) {
		ret = PTR_ERR(kfd_file);
		snprintf(kfd_result.error_msg, sizeof(kfd_result.error_msg),
			 "Failed to open /dev/kfd: error %ld", ret);
		kfd_result.error_code = (int)ret;
		pmu_err("KFD Test: %s\n", kfd_result.error_msg);
		return ret;
	}

	pmu_info("KFD Test: Successfully opened /dev/kfd\n");

	/* Extract the kfd_process from private_data (set by kfd_open) */
	process = kfd_file->private_data;
	if (!process) {
		ret = -EINVAL;
		snprintf(kfd_result.error_msg, sizeof(kfd_result.error_msg),
			 "No kfd_process found in file private_data");
		kfd_result.error_code = (int)ret;
		pmu_err("KFD Test: %s\n", kfd_result.error_msg);
		filp_close(kfd_file, NULL);
		return ret;
	}

	pmu_info("KFD Test: Found valid kfd_process, calling kfd_ioctl_get_version\n");

	/* Now call kfd_ioctl_get_version with proper parameters */
	ret = kfd_ioctl_get_version(kfd_file, process, &version_args);

	if (ret != 0) {
		snprintf(kfd_result.error_msg, sizeof(kfd_result.error_msg),
			 "kfd_ioctl_get_version failed: error %ld", ret);
		kfd_result.error_code = (int)ret;
		pmu_err("KFD Test: %s\n", kfd_result.error_msg);
		filp_close(kfd_file, NULL);
		return ret;
	}

	/* Success - store the version information */
	kfd_result.success = true;
	kfd_result.major_version = version_args.major_version;
	kfd_result.minor_version = version_args.minor_version;
	kfd_result.error_code = 0;

	pmu_info("KFD Test: SUCCESS! KFD Version: %u.%u\n", kfd_result.major_version,
		 kfd_result.minor_version);

	/* Clean up - this calls kfd_release which unrefs the process */
	filp_close(kfd_file, NULL);

	pmu_info("KFD Test: Test completed successfully\n");
	return 0;
}

/**
 * kfd_test_get_result - Get the result of the last KFD test
 * @result: Pointer to store the test result
 *
 * Returns: 0 on success, -EINVAL if result pointer is NULL
 */
int kfd_test_get_result(struct kfd_test_result *result)
{
	if (!result)
		return -EINVAL;

	memcpy(result, &kfd_result, sizeof(struct kfd_test_result));
	return 0;
}

/**
 * kfd_test_print_result - Print the last KFD test result to kernel log
 */
void kfd_test_print_result(void)
{
	if (kfd_result.success) {
		pmu_info("KFD Test Result: SUCCESS - Version %u.%u\n", kfd_result.major_version,
			 kfd_result.minor_version);
	} else {
		pmu_info("KFD Test Result: FAILED - Error %d: %s\n", kfd_result.error_code,
			 kfd_result.error_msg);
	}
}