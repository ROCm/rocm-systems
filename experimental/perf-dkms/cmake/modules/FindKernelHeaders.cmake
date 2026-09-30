# FindKernelHeaders.cmake
# Finds Linux kernel headers for building kernel modules
#
# This module defines:
#  KERNEL_BUILD_DIR - Directory containing kernel build files
#  KERNEL_VERSION - Kernel version string
#  KernelHeaders_FOUND - Whether kernel headers were found

# Get kernel version if not specified
if(NOT KERNEL_VERSION)
    execute_process(
        COMMAND uname -r
        OUTPUT_VARIABLE KERNEL_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

# Find kernel build directory
if(NOT KERNEL_BUILD_DIR)
    set(KERNEL_BUILD_DIR "/lib/modules/${KERNEL_VERSION}/build")
endif()

# Validate that the kernel build directory exists
if(EXISTS "${KERNEL_BUILD_DIR}")
    set(KernelHeaders_FOUND TRUE)
    message(STATUS "Found kernel headers: ${KERNEL_BUILD_DIR}")
    message(STATUS "Kernel version: ${KERNEL_VERSION}")

    # Verify essential kernel build files exist
    if(NOT EXISTS "${KERNEL_BUILD_DIR}/Makefile")
        message(WARNING "Kernel Makefile not found in ${KERNEL_BUILD_DIR}")
        set(KernelHeaders_FOUND FALSE)
    endif()
else()
    set(KernelHeaders_FOUND FALSE)
    message(WARNING "Kernel build directory not found: ${KERNEL_BUILD_DIR}")
endif()

# Standard find_package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(KernelHeaders
    REQUIRED_VARS KERNEL_BUILD_DIR
    VERSION_VAR KERNEL_VERSION
)

mark_as_advanced(KERNEL_BUILD_DIR KERNEL_VERSION)
