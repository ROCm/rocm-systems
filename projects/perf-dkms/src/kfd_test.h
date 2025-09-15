/*
 * kfd_test.h - Header for KFD ioctl test functionality
 *
 * This header defines the interface for testing KFD ioctl calls from kernel space.
 */

#ifndef _KFD_TEST_H
#define _KFD_TEST_H

#include <linux/types.h>

/* KFD test result structure */
struct kfd_test_result {
    bool success;            /* True if test passed */
    u32 major_version;       /* KFD major version (if successful) */
    u32 minor_version;       /* KFD minor version (if successful) */
    int error_code;          /* Error code (if failed) */
    char error_msg[256];     /* Error message (if failed) */
};

/* Function prototypes */

/**
 * kfd_test_get_version - Test KFD ioctl get version call
 *
 * This function demonstrates how to call kfd_ioctl_get_version from kernel space
 * by opening /dev/kfd and making a direct ioctl call.
 *
 * Returns: 0 on success, negative error code on failure
 */
int kfd_test_get_version(void);

/**
 * kfd_test_get_result - Get the result of the last KFD test
 * @result: Pointer to store the test result
 *
 * Returns: 0 on success, -EINVAL if result pointer is NULL
 */
int kfd_test_get_result(struct kfd_test_result *result);

/**
 * kfd_test_print_result - Print the last KFD test result to kernel log
 */
void kfd_test_print_result(void);

#endif /* _KFD_TEST_H */