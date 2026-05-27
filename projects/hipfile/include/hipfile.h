/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <hip/hip_runtime_api.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/types.h>

/* Needed for struct sockaddr */
#ifdef _WIN32
#include <winsock2.h>
#else
#include <sys/socket.h>
#endif

#if defined(__GNUC__)
#define HIPFILE_API __attribute__((visibility("default")))
#else
#define HIPFILE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ***********************************************************************
//  DOXYGEN SETUP
// ***********************************************************************

// Without the `file`/`defgroup` directives, Doxgyen will not emit
// documentation for anything aside from structs in a C header.
//
// *ALL* Doxygen comment blocks should have an `ingroup` directive so they
// group properly in the output.

/**
 * @file
 *
 * @mainpage hipFile API Reference
 *
 * @section contents Contents
 * - @ref core
 * - @ref error
 * - @ref driver
 * - @ref rdma
 * - @ref file
 * - @ref batch
 * - @ref async
 */

/*!
 * @defgroup core Core Functionality
 *
 * @defgroup error Errors and Error Handling
 *
 * @defgroup driver GPU IO Driver API
 *
 * @defgroup rdma Userspace RDMA API
 *
 * @defgroup file File Handle API
 *
 * @defgroup batch Batch API
 *
 * @defgroup async Async API
 */

// ***********************************************************************
//  LIBRARY VERSION NUMBERS
// ***********************************************************************

/*!
 * @brief hipFile major version number
 * @ingroup core
 */
#define HIPFILE_VERSION_MAJOR 0
/*!
 * @brief hipFile minor version number
 * @ingroup core
 */
#define HIPFILE_VERSION_MINOR 2
/*!
 * @brief hipFile patch version number
 * @ingroup core
 */
#define HIPFILE_VERSION_PATCH 0

// ***********************************************************************
//  PLATFORM-INDEPENDENT TYPES
// ***********************************************************************

/*!
 * @brief Platform-independent offset type
 * @ingroup core
 */
#ifdef _WIN32
typedef __int64 hoff_t;
#else
typedef off_t hoff_t;
#endif

/* Handle ssize_t on Windows (does not need Doxygen) */
#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

// ***********************************************************************
//  ERROR HANDLING
// ***********************************************************************

/*!
 * @brief The base value for hipFile error codes
 * @ingroup error
 */
#define HIPFILE_BASE_ERR 5000

/* clang-format off */
/*!
 * @brief hipFile function return codes
 * @ingroup error
 *
 * An error code of -1 indicates a that a C or POSIX error has occurred and
 * errno is likely to have been set.
 *
 * @note HIPFILE_BASE_ERR + 21 and 32 are intentionally omitted.
 */
typedef enum hipFileOpError {
    hipFileSuccess                 = 0,                     //!< hipFile operation completed successfully
    hipFileDriverNotInitialized    = HIPFILE_BASE_ERR + 1,  //!< GPU IO driver is not loaded
    hipFileDriverInvalidProps      = HIPFILE_BASE_ERR + 2,  //!< Invalid GPU IO driver property provided
    hipFileDriverUnsupportedLimit  = HIPFILE_BASE_ERR + 3,  //!< GPU IO Driver property value is unsupported
    hipFileDriverVersionMismatch   = HIPFILE_BASE_ERR + 4,  //!< hipFile version does not match GPU IO driver version
    hipFileDriverVersionReadError  = HIPFILE_BASE_ERR + 5,  //!< Unable to read the GPU IO driver version
    hipFileDriverClosing           = HIPFILE_BASE_ERR + 6,  //!< GPU IO driver is closing and not accepting new requests
    hipFilePlatformNotSupported    = HIPFILE_BASE_ERR + 7,  //!< hipFile is not supported on the current platform
    hipFileIONotSupported          = HIPFILE_BASE_ERR + 8,  //!< hipFile is not supported on the selected file
    hipFileDeviceNotSupported      = HIPFILE_BASE_ERR + 9,  //!< The selected GPU does not support hipFile
    hipFileDriverError             = HIPFILE_BASE_ERR + 10, //!< GPU IO driver error
    hipFileHipDriverError          = HIPFILE_BASE_ERR + 11, //!< GPU driver error: Inspect the hipError_t value for additional information
    hipFileHipPointerInvalid       = HIPFILE_BASE_ERR + 12, //!< Invalid GPU pointer
    hipFileHipMemoryTypeInvalid    = HIPFILE_BASE_ERR + 13, //!< Memory type backing pointer is incompatible with hipFile
    hipFileHipPointerRangeError    = HIPFILE_BASE_ERR + 14, //!< Pointer range exceeds allocated memory region
    hipFileHipContextMismatch      = HIPFILE_BASE_ERR + 15, //!< GPU driver context mismatch
    hipFileInvalidMappingSize      = HIPFILE_BASE_ERR + 16, //!< Accessing memory beyond pinned memory buffer
    hipFileInvalidMappingRange     = HIPFILE_BASE_ERR + 17, //!< Accessing memory beyond mapped memory region
    hipFileInvalidFileType         = HIPFILE_BASE_ERR + 18, //!< Unsupported file type
    hipFileInvalidFileOpenFlag     = HIPFILE_BASE_ERR + 19, //!< Unsupported file open flags
    hipFileDIONotSet               = HIPFILE_BASE_ERR + 20, //!< O_DIRECT flag not set
    /* Value 5021 intentionally unused */
    hipFileInvalidValue            = HIPFILE_BASE_ERR + 22, //!< One or more arguments have an invalid value
    hipFileMemoryAlreadyRegistered = HIPFILE_BASE_ERR + 23, //!< Device pointer is already registered
    hipFileMemoryNotRegistered     = HIPFILE_BASE_ERR + 24, //!< Device pointer is not registered
    hipFilePermissionDenied        = HIPFILE_BASE_ERR + 25, //!< Permission error on device or file access
    hipFileDriverAlreadyOpen       = HIPFILE_BASE_ERR + 26, //!< GPU IO driver is already open
    hipFileHandleNotRegistered     = HIPFILE_BASE_ERR + 27, //!< File handle for GPU IO is not registered
    hipFileHandleAlreadyRegistered = HIPFILE_BASE_ERR + 28, //!< File handle for GPU IO is already registered
    hipFileDeviceNotFound          = HIPFILE_BASE_ERR + 29, //!< Selected device not found
    hipFileInternalError           = HIPFILE_BASE_ERR + 30, //!< Internal GPU IO library error
    hipFileGetNewFDFailed          = HIPFILE_BASE_ERR + 31, //!< Unable to obtain a new file descriptor
    /* Value 5032 intentionally unused */
    hipFileDriverSetupError        = HIPFILE_BASE_ERR + 33, //!< GPU IO Driver initialization error
    hipFileIODisabled              = HIPFILE_BASE_ERR + 34, //!< GPU IO config file prohibits GPU IO on specified file
    hipFileBatchSubmitFailed       = HIPFILE_BASE_ERR + 35, //!< Failed to submit request for batch operation
    hipFileGPUMemoryPinningFailed  = HIPFILE_BASE_ERR + 36, //!< Failed to allocated pinned device memory
    hipFileBatchFull               = HIPFILE_BASE_ERR + 37, //!< Batch operation queue is full
    hipFileAsyncNotSupported       = HIPFILE_BASE_ERR + 38, //!< hipFile async IO is not supported
    hipFileIOMaxError              = HIPFILE_BASE_ERR + 39, //!< Internal flag to mark largest hipFile error code
} hipFileOpError_t;
/* clang-format on */

/*!
 * @brief Return a descriptive string for a hipFile error
 * @ingroup error
 *
 * @param [in] status Return code provided by hipFile
 *
 * @return Description of the error encountered
 */
HIPFILE_API
const char *hipFileGetOpErrorString(hipFileOpError_t status);

// Ignoring return values from hipFile APIs is discouraged.
// In C++17 and C23 and up, we can make that emit a warning.
#if (defined(__cplusplus) && __cplusplus >= 201703L) || __STDC_VERSION__ >= 202311L
#define __HIPFILE_NODISCARD [[nodiscard]]
#else
#define __HIPFILE_NODISCARD
#endif

/*!
 * @brief Error status returned from hipFile API calls
 * @ingroup error
 *
 * @note This struct has the `[[nodiscard]]` attribute in C++ >= 17 and
 *       C >= 23 so unhandled return values will generate warnings
 */
typedef struct __HIPFILE_NODISCARD hipFileError {
    hipFileOpError_t err;         //!< Errors related to hipFile or the GPU IO driver
    hipError_t       hip_drv_err; //!< Errors related to the GPU driver
} hipFileError_t;

/*!
 * @brief Determine if an error code is a hipFile error
 * @ingroup error
 * @hideinitializer
 *
 * Signature
 * @code
 * bool IS_HIPFILE_ERR(hipFileOpError err)
 * @endcode
 *
 * @return true/false
 */
#define IS_HIPFILE_ERR(hip_op_err) (abs(hip_op_err) > HIPFILE_BASE_ERR)

/*!
 * @brief Get an error string for a hipFile error
 * @ingroup error
 * @hideinitializer
 *
 * Signature
 * @code
 * const char *HIPFILE_ERRSTR(hipFileOpError err)
 * @endcode
 *
 * @return A string description of a hipFile error
 */
#define HIPFILE_ERRSTR(hip_op_err) hipFileGetOpErrorString((hipFileOpError_t)abs(hip_op_err))

/*!
 * @brief Determine if an error is a hipFile driver error
 * @ingroup error
 * @hideinitializer
 *
 * Signature
 * @code
 * bool IS_HIP_DRV_ERR(hipFileError_t err)
 * @endcode
 *
 * @return true/false
 */
#ifndef IS_HIP_DRV_ERR
#define IS_HIP_DRV_ERR(hip_err) (hip_err.err == hipFileHipDriverError)
#endif

/*!
 * @brief Get the driver error from a hipFileError_t
 * @ingroup error
 * @hideinitializer
 *
 * Signature
 * @code
 * hipError_t HIP_DRV_ERR(hipFileError_t err)
 * @endcode
 *
 * @return The hipError_t component of err
 */
#ifndef HIP_DRV_ERR
#define HIP_DRV_ERR(hip_err) (hip_err.hip_drv_err)
#endif

// ***********************************************************************
//  GPU IO DRIVER API
// ***********************************************************************

/*
 * Most of the enums in this defgroup indicate that their intended
 * use is as flags, which is impossible if the consecutive values in
 * cufile.h are used. Note the flag fields in hipFileDriverProps_t.nvfs.
 * The documentation in cufile.h is also incomplete and many of the
 * structs/enums are poorly documented, if at all. It may require some
 * experimentation to figure out how the fields are supposed to work.
 *
 * The ideal solution is probably going to involve replacing the
 * enums with named collections of #defined bitwise flags.
 */

/*!
 * @brief Filesystems/storage protocols supported by GPU IO on this system
 * @ingroup driver
 *
 * @note Value 10 is reserved for YRCloudFile
 */
typedef enum hipFileDriverStatusFlags {
    hipFileLustreSupported       = 0, //!< Lustre is supported
    hipFileWekaFSSupported       = 1, //!< Weka is supported
    hipFileNFSSupported          = 2, //!< NFS is supported
    hipFileGPFSSupported         = 3, //!< GPFS/IBM Storage Scale is supported
    hipFileNVMeSupported         = 4, //!< Local NVMe is supported
    hipFileNVMeoFSupported       = 5, //!< NVMeoF is supported
    hipFileSCSISupported         = 6, //!< SCSI is supported
    hipFileScaleFluxCSDSupported = 7, //!< ScaleFlux CSD is supported
    hipFileNVMeshSupported       = 8, //!< NVMesh is supported
    hipFileBeeGFSSupported       = 9, //!< BeeGFS is supported
    /* 10 is reserved for YRCloudFile */
    hipFileNVMeP2PSupported = 11, //!< NVMeP2P is supported
    hipFileScatefsSupported = 12, //!< ScateFS is supported
} hipFileDriverStatusFlags_t;

/*!
 * @brief Control flags for passing to the GPU IO driver
 * @ingroup driver
 */
typedef enum hipFileDriverControlFlags {
    hipFileUsePollMode     = 0, //!< Enable polling for IO completion
    hipFileAllowCompatMode = 1, //!< Allow GPU IO to fall back to POSIX IO
} hipFileDriverControlFlags_t;

/*!
 * @brief GPU IO Transport & Features supported by the system
 * @ingroup driver
 */
typedef enum hipFileFeatureFlags {
    hipFileDynRoutingSupported = 0, //!< RDMA dynamic routing is supported
    hipFileBatchIOSupported    = 1, //!< Batch operations are supported
    hipFileStreamsSupported    = 2, //!< Streams are supported
    hipFileParallelIOSupported = 3, //!< Parallel IO is supported
} hipFileFeatureFlags_t;

/*!
 * @brief GPU IO configuration
 * @ingroup driver
 */
typedef struct hipFileDriverProps {
    /*!
     * @brief GPU IO Driver Configuration
     */
    struct {
        unsigned major_version; //!< Major version of the GPU IO driver
        unsigned minor_version; //!< Minor version of the GPU IO driver

        uint64_t poll_thresh_size;   //!< Maximum IO size (in KiB) for which polling is used when poll mode is
                                     //!< enabled. Must be a multiple of 4K on NVIDIA.
        uint64_t max_direct_io_size; //!< Maximum IO size (in KiB) used by the GPU IO driver. Must be a
                                     //!< multiple of 64K on NVIDIA.

        unsigned driver_status_flags;  //!< Bitfield that maps to hipFileDriverStatusFlags_t
        unsigned driver_control_flags; //!< Bitfield that maps to hipFileDriverControlFlags_t
    } nvfs;

    unsigned feature_flags; //!< Bitfield that maps to hipFileFeatureFlags_t

    uint64_t max_device_cache_size; //!< Maximum amount of GPU memory (in KiB) that can be used for bounce
                                    //!< buffers. Must be a multiple of 4K on NVIDIA.
    uint64_t per_buffer_cache_size; //!< Amount of GPU memory (in KiB) allocated for each bounce buffer.
                                    //!< Must be a multiple of 4K on NVIDIA.
    uint64_t max_device_pinned_mem_size; //!< Maximum amount of GPU memory (in KiB) that can be pinned.
                                         //!< Must be a multiple of 4K on NVIDIA.

    unsigned max_batch_io_count;         //!< Maximum number of batch operations that can be submitted at once
    unsigned max_batch_io_timeout_msecs; //!< Timeout (in msec) for a batch operation to complete
} hipFileDriverProps_t;

// ***********************************************************************
//  RDMA API
// ***********************************************************************

/*!
 * @brief Userspace RDMA configuration
 * @ingroup rdma
 */
typedef struct hipFileRDMAInfo {
    int         version;  //!< Version of the RDMA API to use
    int         desc_len; //!< Length of the description string
    const char *desc_str; //!< Describes the configuration of the RDMA operation
} hipFileRDMAInfo_t;

/*!
 * @defgroup rdma_flags RDMA Characteristic Flags
 * @ingroup rdma
 *
 * @brief Bitwise flags for RDMA characteristics
 * @{
 */

/// Flag for if the RDMA operation is registered
#define HIPFILE_RDMA_REGISTER 1

/// Flag for if the RDMA operation has relaxed ordering
#define HIPFILE_RDMA_RELAXED_ORDERING (1 << 1)
/*! @} */

// ***********************************************************************
//  FILE HANDLE API
// ***********************************************************************

/*!
 * @brief IO operations for RDMA filesystems
 * @ingroup file
 */
typedef struct hipFileFSOps {
    /*!
     * Type of remote FS used. If NULL, use fstat to discover.
     */
    const char *(*fs_type)(void *handle);
    /*!
     * Get a list of host RDMA addresses. If NULL, use any address.
     */
    int (*getRDMADeviceList)(void *handle, struct sockaddr **hostaddrs);
    /*!
     * Get the assigned priority of a RDMA device. If -1, there is no preference.
     */
    int (*getRDMADevicePriority)(void *handle, char *, size_t, hoff_t, struct sockaddr *hostaddr);
    /*!
     * Read from the remote filesystem. If NULL, use the Linux VFS.
     */
    ssize_t (*read)(void *handle, char *, size_t, hoff_t, hipFileRDMAInfo_t *);
    /*!
     * Write to the remote filesystem. If NULL, use the Linux VFS.
     */
    ssize_t (*write)(void *handle, const char *, size_t, hoff_t, hipFileRDMAInfo_t *);
} hipFileFSOps_t;

/*!
 * @brief Type of file handle being used
 * @ingroup file
 */
typedef enum hipFileFileHandleType {
    hipFileHandleTypeOpaqueFD    = 1, //!< POSIX file descriptor
    hipFileHandleTypeOpaqueWin32 = 2, //!< Win32 HANDLE file handle
    hipFileHandleTypeUserspaceFS = 3, //!< Userspace RDMA filesystem
} hipFileFileHandleType_t;

/*!
 * @brief Top-level structure for performing GPU IO
 * @ingroup file
 *
 *  hipFileHandleTypeOpaqueFD    -> handle.fd non-negative, fs_ops ignored
 *  hipFileHandleTypeOpaqueWin32 -> handle.hFile non-NULL, fs_ops ignored
 *  hipFileHandleTypeUserspaceFS -> handle.fd non-negative, fs_ops non-NULL
 */
typedef struct hipFileDescr {
    hipFileFileHandleType_t type; //!< Type of file handle being used
    union {
        int   fd;                 //!< POSIX file descriptor
        void *hFile;              //!< Win32 HANDLE file handle
    } handle;                     //!< Union containing the OS-specific file handle
    const hipFileFSOps_t *fs_ops; //!< Userspace RDMA filesystem operations
} hipFileDescr_t;

/*!
 * @brief Opaque file handle used by the GPU IO driver/library
 * @ingroup file
 */
typedef void *hipFileHandle_t;

/*!
 * @brief Registers an open file for GPU IO
 * @ingroup file
 *
 * @note If the library has not already been initialized, the first call to
 *       `hipFileHandleRegister()` will initialize the library and increment
 *       the reference count.
 *
 * @param [out] fh    \hipfile_handle_param
 * @param [in]  descr Parameters for opening the file
 *
 * @return \hipfile_error_return
 */
HIPFILE_API
hipFileError_t hipFileHandleRegister(hipFileHandle_t *fh, hipFileDescr_t *descr);

/*!
 * @brief Deregisters a file from GPU IO
 * @ingroup file
 *
 * @param [in] fh \hipfile_handle_param
 */
HIPFILE_API
void hipFileHandleDeregister(hipFileHandle_t fh);

/*!
 * @brief Registers a GPU memory region to be used with GPU IO
 * @ingroup file
 *
 * The memory region should be allocated by the user before being passed to the API call
 *
 * @note If the library has not already been initialized, the first call to
 *       `hipFileBufRegister()` will initialize the library and increment
 *       the reference count.
 *
 * @param [in] buffer_base \buffer_base_param
 * @param [in] length      Size of the allocated buffer in bytes
 * @param [in] flags       Control flags for this buffer
 *
 * @return \hipfile_error_return
 */
HIPFILE_API
hipFileError_t hipFileBufRegister(const void *buffer_base, size_t length, int flags);

/*!
 * @brief Deregisters a GPU memory region from being used with GPU IO
 * @ingroup file
 *
 * @param [in] buffer_base \buffer_base_param
 *
 * @return \hipfile_error_return
 */
HIPFILE_API
hipFileError_t hipFileBufDeregister(const void *buffer_base);

/*!
 * @brief Synchronously read data from a file into a GPU buffer
 * @ingroup file
 *
 * @param [in] fh            \hipfile_handle_param
 * @param [in] buffer_base   \buffer_base_param
 * @param [in] size          Number of bytes that should be read
 * @param [in] file_offset   Offset into the file that should be read from
 * @param [in] buffer_offset Offset of the GPU buffer that that the data should be written to
 *
 * \max_io_size_note
 *
 * @return if >= 0: Number of bytes read
 * @return if -1:   System error (check `errno` for the specific error)
 * @return else:    Negative value of the related hipFileOpError_t
 */
HIPFILE_API
ssize_t hipFileRead(hipFileHandle_t fh, void *buffer_base, size_t size, hoff_t file_offset,
                    hoff_t buffer_offset);

/*!
 * @brief Synchronously write data from a GPU buffer to a file
 * @ingroup file
 *
 * @param [in] fh            \hipfile_handle_param
 * @param [in] buffer_base   \buffer_base_param
 * @param [in] size          Number of bytes that should be written
 * @param [in] file_offset   Offset into the file that should be written to
 * @param [in] buffer_offset Offset of the GPU buffer that the data should be read from
 *
 * \max_io_size_note
 *
 * @return if >= 0: Number of bytes written
 * @return if -1:   System error (check `errno` for the specific error)
 * @return else:    Negative value of the related hipFileOpError_t
 */
HIPFILE_API
ssize_t hipFileWrite(hipFileHandle_t fh, const void *buffer_base, size_t size, hoff_t file_offset,
                     hoff_t buffer_offset);

// ***********************************************************************
//  GPU IO DRIVER API
// ***********************************************************************

/*!
 * @brief Initialize the GPU IO driver for this process
 * @ingroup driver
 *
 * Each call to `hipFileDriverOpen()` increments the library's reference
 * count. If a call to `hipFileDriverOpen()` results in the reference count
 * transitioning from zero to one, the library's state will be initialized.
 *
 * Calling `hipFileDriverOpen()` is optional. The first call to
 * `hipFileBufRegister()` or `hipFileHandleRegister()` will trigger
 * library initialization and increment the library's reference count.
 *
 * @return \hipfile_error_return
 */
HIPFILE_API
hipFileError_t hipFileDriverOpen(void);

/*!
 * @brief Close the GPU IO driver for this process
 * @ingroup driver
 *
 * Each call to `hipFileDriverClose()` decrements the library's reference
 * count. If a call to `hipFileDriverClose()` results in the reference count
 * transitioning from one to zero, the library's state will be destroyed.
 *
 * Explicitly closing the library is not required; the library's state will be
 * destroyed automatically at program exit.
 *
 * @return \hipfile_error_return
 */
HIPFILE_API
hipFileError_t hipFileDriverClose(void);

/*!
 * @brief Obtain the current reference count for the library
 * @ingroup driver
 *
 * @see hipFileDriverOpen()
 * @see hipFileDriverClose()
 *
 * @return The library's reference count
 */
HIPFILE_API
int64_t hipFileUseCount(void);

/*!
 * @brief Get a list of GPU IO driver properties
 * @ingroup driver
 *
 * \warn_not_implemented
 *
 * @param [out] props See `hipFileDriverProps_t` for what properties are reported
 *
 * @return \hipfile_error_return
 */
HIPFILE_API
hipFileError_t hipFileDriverGetProperties(hipFileDriverProps_t *props);

/*!
 * @brief Enable/disable polling mode for GPU IO
 * @ingroup driver
 *
 * @note On NVIDIA, `poll_threshold_size` must be an increment of 4K
 *
 * \warn_not_implemented
 *
 * @param [in] poll `true` to enable polling, `false` to disable
 * @param [in] poll_threshold_size Maximum IO size (in KiB) for which polling is
 *                                 used when enabled
 *
 * @return \hipfile_error_return
 */
HIPFILE_API
hipFileError_t hipFileDriverSetPollMode(bool poll, size_t poll_threshold_size);

/*!
 * @brief Set the maximum IO chunk size
 * @ingroup driver
 *
 * @note Must be in 64k increments on NVIDIA
 *
 * \warn_not_implemented
 *
 * @param [in] max_direct_io_size Maximum IO chunk size (in KiB) for each IO request
 *
 * @return \hipfile_error_return
 */
HIPFILE_API
hipFileError_t hipFileDriverSetMaxDirectIOSize(size_t max_direct_io_size);

/*!
 * @brief Set the maximum amount of GPU memory that can be used for bounce buffers
 * @ingroup driver
 *
 * @note Must be in 4k increments on NVIDIA
 *
 * \warn_not_implemented
 *
 * @param [in] max_cache_size Maximum GPU memory (in KiB) that can be reserved for bounce buffers
 *
 * @return \hipfile_error_return
 */
HIPFILE_API
hipFileError_t hipFileDriverSetMaxCacheSize(size_t max_cache_size);

/*!
 * @brief Set the maximum amount of GPU memory that can be pinned
 * @ingroup driver
 *
 * @note Must be in 4K increments on NVIDIA
 *
 * \warn_not_implemented
 *
 * @param [in] max_pinned_size Maximum GPU memory (in KiB) that can be pinned
 *
 * @return \hipfile_error_return
 */
HIPFILE_API
hipFileError_t hipFileDriverSetMaxPinnedMemSize(size_t max_pinned_size);

// ***********************************************************************
//  BATCH API
// ***********************************************************************

/*!
 * @brief The direction of data movement in a batch IO request
 * @ingroup batch
 */
typedef enum hipFileOpcode {
    hipFileBatchRead  = 0, //!< Read batch IO operation
    hipFileBatchWrite = 1, //!< Write batch IO operation
} hipFileOpcode_t;

/*!
 * @brief The status of a batch IO operation
 * @ingroup batch
 */
typedef enum hipFileStatus {
    hipFileWaiting  = 1 << 0, //!< Batch IO operation pending acceptance
    hipFilePending  = 1 << 1, //!< Batch IO operation accepted and queued
    hipFileInvalid  = 1 << 2, //!< Batch IO operation was rejected for being invalid or could not be queued
    hipFileCanceled = 1 << 3, //!< Batch IO operation was canceled
    hipFileComplete = 1 << 4, //!< Batch IO operation completed
    hipFileTimeout  = 1 << 5, //!< Batch IO operation timed out
    hipFileFailed   = 1 << 6, //!< Batch IO operation failed
} hipFileStatus_t;

/*!
 * @brief Mode of a batch IO operation
 * @ingroup batch
 */
typedef enum hipFileBatchMode {
    hipFileBatch = 1, //!< Normal batch IO operation
} hipFileBatchMode_t;

/*!
 * @brief Input parameters for a batch IO request
 * @ingroup batch
 */
typedef struct hipFileIOParams {
    hipFileBatchMode_t mode; //!< Mode of the batch IO request
    union {
        struct {
            void   *devPtr_base;   //!< Base address of the GPU buffer where data should be read/written
            int64_t file_offset;   //!< Offset into the file that should be read/written
            int64_t devPtr_offset; //!< Offset of the GPU buffer that should be read/written
            size_t  size;          //!< Number of bytes to read or write
        } batch;                   //!< Parameters for the read/write batch request
    } u;                           //!< Wrapping union for batch IO parameters
    hipFileHandle_t fh;            //!< Registered hipFile handle for the target file
    hipFileOpcode_t opcode;        //!< Direction data is moving for the batch request
    void           *cookie;        //!< Optionally used to track IO operations (e.g. self-reference pointer)
} hipFileIOParams_t;

/*!
 * @brief Status of a batch IO operation
 * @ingroup batch
 */
typedef struct hipFileIOEvents {
    void *cookie; //!< Optionally used to track IO operations (e.g. pointer to corresponding hipFileIOParams)
    hipFileStatus_t status; //!< Status of the batch IO operation
    size_t          ret;    //!< Number of bytes transferred or POSIX error code if negative
} hipFileIOEvents_t;

/*!
 * @brief Opaque batch operations handle
 * @ingroup batch
 */
typedef void *hipFileBatchHandle_t;

/*!
 * @brief Prepare the system to perform a batch IO operation
 * @ingroup batch
 *
 * @param [out] batch_idp \batch_handle_param
 * @param [in] max_nr     Maximum number of requests that can be submitted to this batch handle
 *
 * @return \hipfile_error_return
 */
HIPFILE_API
hipFileError_t hipFileBatchIOSetUp(hipFileBatchHandle_t *batch_idp, unsigned max_nr);

/*!
 * @brief Enqueue a batch of IO requests for the GPU to complete asynchronously
 * @ingroup batch
 *
 * @param [in]     batch_idp     \batch_handle_param
 * @param [in]     nr            Number of batch IO requests to submit
 * @param [in]     iocbp         An array of `nr` batch IO requests to submit to the GPU.
 *                               Data will be read into or written from the buffer specified in
 *                               each request.
 * @param [in]     flags Control Flags for the batch IO. Currently unused.
 *
 * @return \hipfile_error_return
 */
HIPFILE_API
hipFileError_t hipFileBatchIOSubmit(hipFileBatchHandle_t batch_idp, unsigned nr, hipFileIOParams_t *iocbp,
                                    unsigned flags);

/*!
 * @brief Poll for the status of completed batch IO operations
 * @ingroup batch
 *
 * \warn_not_implemented
 *
 * @param [in] batch_idp \batch_handle_param
 * @param [in] min_nr Minimum number of batch operation statuses that should be returned.
 *                    If `timeout` is exceeded, fewer statuses may be returned.
 * @param [in,out] nr Maximum number of batch operation statuses that can be returned.
 *                    This is parameter is modified to return the number of statuses returned in `iocbp`.
 *
 * @param [out] iocbp An array containing up to `nr` statuses from the overall batch operation
 * @param [in] timeout Maximum amount of time this function should poll for status updates
 *
 * @return \hipfile_error_return
 */
HIPFILE_API
hipFileError_t hipFileBatchIOGetStatus(hipFileBatchHandle_t batch_idp, unsigned min_nr, unsigned *nr,
                                       hipFileIOEvents_t *iocbp, struct timespec *timeout);

/*!
 * @brief Cancels all pending batch IO operations
 * @ingroup batch
 *
 * \warn_not_implemented
 *
 * @param [in] batch_idp \batch_handle_param
 *
 * @return \hipfile_error_return
 */
HIPFILE_API
hipFileError_t hipFileBatchIOCancel(hipFileBatchHandle_t batch_idp);

/*!
 * @brief Destroys the batch IO handle and frees the associated resources
 * @ingroup batch
 *
 * \warn_not_implemented_void
 *
 * @param [in] batch_idp \batch_handle_param
 */
HIPFILE_API
void hipFileBatchIODestroy(hipFileBatchHandle_t batch_idp);

// ***********************************************************************
//  ASYNC API
// ***********************************************************************

/*!
 * @defgroup stream_flags Stream registration flags
 * @ingroup async
 *
 * @brief Flags to configure async GPU IO behaviour at stream registration
 * @{
 */

/// Buffer offset is fixed at time of submission
#define HIPFILE_STREAM_FIXED_BUF_OFFSET 1

/// File offset is fixed at time of submission
#define HIPFILE_STREAM_FIXED_FILE_OFFSET (1 << 1)

/// File size is fixed at time of submission
#define HIPFILE_STREAM_FIXED_FILE_SIZE (1 << 2)

/// Offsets and size are 4k aligned
#define HIPFILE_STREAM_PAGE_ALIGNED_INPUTS (1 << 3)

/// Mask for selecting flag bits
#define HIPFILE_STREAM_FLAGS_MASK 0xf
/*! @} */

/*!
 * @brief Perform an asynchronous read from a stream
 * @ingroup async
 *
 * @param [in]  fh              \hipfile_handle_param
 * @param [in]  buffer_base     \buffer_base_param
 * @param [in]  size_p          Number of bytes that should be read
 * @param [in]  file_offset_p   Offset into the file that should be read from
 * @param [in]  buffer_offset_p Offset of the GPU buffer that that the data should be written to
 * @param [out] bytes_read_p    Number of bytes read
 * @param [in]  stream          \hipstream_param. \hipstream_if_null.
 *
 * \max_io_size_note
 *
 * @return \hipfile_error_return
 */
HIPFILE_API
hipFileError_t hipFileReadAsync(hipFileHandle_t fh, void *buffer_base, size_t *size_p, hoff_t *file_offset_p,
                                hoff_t *buffer_offset_p, ssize_t *bytes_read_p, hipStream_t stream);

/*!
 * @brief Perform an asynchronous write to a stream
 * @ingroup async
 *
 * @param [in]  fh              \hipfile_handle_param
 * @param [in]  buffer_base     \buffer_base_param
 * @param [in]  size_p          Number of bytes that should be written
 * @param [in]  file_offset_p   Offset into the file that should be written to
 * @param [in]  buffer_offset_p Offset of the GPU buffer that that the data should be read from
 * @param [out] bytes_written_p Number of bytes written
 * @param [in]  stream          \hipstream_param. \hipstream_if_null.
 *
 * \max_io_size_note
 *
 * @return \hipfile_error_return
 */
HIPFILE_API
hipFileError_t hipFileWriteAsync(hipFileHandle_t fh, void *buffer_base, size_t *size_p, hoff_t *file_offset_p,
                                 hoff_t *buffer_offset_p, ssize_t *bytes_written_p, hipStream_t stream);

/*!
 * @brief Register a stream to be used by for asynchronous GPU IO
 * @ingroup async
 *
 * @param [in] stream \hipstream_param
 * @param [in] flags Flags that can optimize stream processing if parameters
 *                   are known/are aligned at time of request submission
 *
 * @return \hipfile_error_return
 */
HIPFILE_API
hipFileError_t hipFileStreamRegister(hipStream_t stream, unsigned flags);

/*!
 * @brief Deregister a stream and free the associated resources
 * @ingroup async
 *
 * @param [in] stream \hipstream_param
 *
 * @return \hipfile_error_return
 */
HIPFILE_API
hipFileError_t hipFileStreamDeregister(hipStream_t stream);

// ***********************************************************************
//  CORE API
// ***********************************************************************

/*!
 * @brief Get the version of the hipFile library
 * @ingroup core
 *
 * @param [out] major The major version
 * @param [out] minor The minor version
 * @param [out] patch The patch version
 *
 * @note Parameters can be set to NULL to ignore that part of the version
 *
 * @return \hipfile_error_return
 */
HIPFILE_API
hipFileError_t hipFileGetVersion(unsigned *major, unsigned *minor, unsigned *patch);

// ***********************************************************************
//  PROPERTIES API
// ***********************************************************************

/*!
 * @brief size_t configuration parameters
 * @ingroup core
 */
typedef enum hipFileSizeTConfigParameter_t {
    hipFileParamProfileStats,                       //!< Statistics
    hipFileParamExecutionMaxIOQueueDepth,           //!< Max IO queue depth
    hipFileParamExecutionMaxIOThreads,              //!< Max number of IO threads
    hipFileParamExecutionMinIOThresholdSizeKB,      //!< Min IO threshold (KiB)
    hipFileParamExecutionMaxRequestParallelism,     //!< Max request parallelism
    hipFileParamPropertiesMaxDirectIOSizeKB,        //!< Max direct IO size (KiB)
    hipFileParamPropertiesMaxDeviceCacheSizeKB,     //!< Max device cache size (KiB)
    hipFileParamPropertiesPerBufferCacheSizeKB,     //!< Max buffer cache size (KiB)
    hipFileParamPropertiesMaxDevicePinnedMemSizeKB, //!< Max device pinned memory size (KiB)
    hipFileParamPropertiesIOBatchsize,              //!< IO batch size
    hipFileParamPollthresholdSizeKB,                //!< Poll threshold size (KiB)
    hipFileParamPropertiesBatchIOTimeoutMs,         //!< Batch IO timeout (ms)
} hipFileSizeTConfigParameter_t;

/*!
 * @brief Boolean configuration parameters
 * @ingroup core
 */
typedef enum hipFileBoolConfigParameter_t {
    hipFileParamPropertiesUsePollMode,       //!< Use poll mode
    hipFileParamPropertiesAllowCompatMode,   //!< Allow compatibility mode
    hipFileParamForceCompatMode,             //!< Force compatibility mode
    hipFileParamFsMiscApiCheckAggressive,    //!< Use aggressive checks in API calls
    hipFileParamExecutionParallelIO,         //!< Allow parallel IO execution
    hipFileParamProfileNvtx,                 //!< Use NVTX profiler
    hipFileParamPropertiesAllowSystemMemory, //!< Allow system memory use
    hipFileParamUsePcip2pdma,                //!< Use P2P DMA
    hipFileParamPreferIOUring,               //!< Prefer io_uring
    hipFileParamForceOdirectMode,            //!< Force O_DIRECT mode
    hipFileParamSkipTopologyDetection,       //!< Skip topology detection
    hipFileParamStreamMemopsBypass,          //!< Bypass stream memory operations
} hipFileBoolConfigParameter_t;

/*!
 * @brief String configuration parameters
 * @ingroup core
 */
typedef enum hipFileStringConfigParameter_t {
    hipFileParamLoggingLevel,   //!< Logging level
    hipFileParamEnvLogfilePath, //!< Logfile path
    hipFileParamLogDir,         //!< Logfile directory
} hipFileStringConfigParameter_t;

/*!
 * @brief Get the value of a size_t configuration parameter
 * @ingroup core
 *
 * \warn_not_implemented
 *
 * @param param The configuration parameter
 * @param value The location to store the value of the configuration parameter
 *
 * @return \hipfile_error_return
 *
 * @note If the driver is open, the value returned is the value currently in use by the driver.
 * @note If the driver is closed, the value returned is the value that was last set by hipFileSetParameter*.
 */
HIPFILE_API
hipFileError_t hipFileGetParameterSizeT(hipFileSizeTConfigParameter_t param, size_t *value);

/*!
 * @brief Get the value of a Boolean configuration parameter
 * @ingroup core
 *
 * \warn_not_implemented
 *
 * @param param The configuration parameter
 * @param value The location to store the value of the configuration parameter
 *
 * @return \hipfile_error_return
 *
 * @note If the driver is open, the value returned is the value currently in use by the driver.
 * @note If the driver is closed, the value returned is the value that was last set by hipFileSetParameter*.
 */
HIPFILE_API
hipFileError_t hipFileGetParameterBool(hipFileBoolConfigParameter_t param, bool *value);

/*!
 * @brief Get the value of a string configuration parameter
 * @ingroup core
 *
 * \warn_not_implemented
 *
 * @param param    The configuration parameter
 * @param desc_str The location to store the value of the configuration parameter
 * @param len      The length of the desc_str parameter
 *
 * @return \hipfile_error_return
 *
 * @note If the driver is open, the value returned is the value currently in use by the driver.
 * @note If the driver is closed, the value returned is the value that was last set by hipFileSetParameter*.
 */
HIPFILE_API
hipFileError_t hipFileGetParameterString(hipFileStringConfigParameter_t param, char *desc_str, int len);

/*!
 * @brief Set the value of a size_t configuration parameter
 * @ingroup core
 *
 * \warn_not_implemented
 *
 * @param param The configuration parameter
 * @param value The value of the configuration parameter
 *
 * @return \hipfile_error_return
 *
 * @note Configuration parameter values may only be set when the driver is closed. Values are applied when the
 * driver is opened.
 */
HIPFILE_API
hipFileError_t hipFileSetParameterSizeT(hipFileSizeTConfigParameter_t param, size_t value);

/*!
 * @brief Set the value of a Boolean configuration parameter
 * @ingroup core
 *
 * \warn_not_implemented
 *
 * @param param The configuration parameter
 * @param value The value of the configuration parameter
 *
 * @return \hipfile_error_return
 *
 * @note Configuration parameter values may only be set when the driver is closed. Values are applied when the
 * driver is opened.
 */
HIPFILE_API
hipFileError_t hipFileSetParameterBool(hipFileBoolConfigParameter_t param, bool value);

/*!
 * @brief Set the value of a string configuration parameter
 * @ingroup core
 *
 * \warn_not_implemented
 *
 * @param param    The configuration parameter
 * @param desc_str The value of the configuration parameter
 *
 * @return \hipfile_error_return
 *
 * @note Configuration parameter values may only be set when the driver is closed. Values are applied when the
 * driver is opened.
 */
HIPFILE_API
hipFileError_t hipFileSetParameterString(hipFileStringConfigParameter_t param, const char *desc_str);

// Not a part of the public API
#undef __HIPFILE_NODISCARD

#ifdef __cplusplus
}
#endif
