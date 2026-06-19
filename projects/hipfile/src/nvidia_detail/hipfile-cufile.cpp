/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hipfile-cufile.h"

#include <climits>
#include <cuda.h>
#include <hip/hip_runtime_api.h>
#include <stdexcept>

hipFileOpError_t
toHipFileOpError(CUfileOpError cu_status)
{
    switch (cu_status) {
        case CU_FILE_SUCCESS:
            return hipFileSuccess;
        case CU_FILE_DRIVER_NOT_INITIALIZED:
            return hipFileDriverNotInitialized;
        case CU_FILE_DRIVER_INVALID_PROPS:
            return hipFileDriverInvalidProps;
        case CU_FILE_DRIVER_UNSUPPORTED_LIMIT:
            return hipFileDriverUnsupportedLimit;
        case CU_FILE_DRIVER_VERSION_MISMATCH:
            return hipFileDriverVersionMismatch;
        case CU_FILE_DRIVER_VERSION_READ_ERROR:
            return hipFileDriverVersionReadError;
        case CU_FILE_DRIVER_CLOSING:
            return hipFileDriverClosing;
        case CU_FILE_PLATFORM_NOT_SUPPORTED:
            return hipFilePlatformNotSupported;
        case CU_FILE_IO_NOT_SUPPORTED:
            return hipFileIONotSupported;
        case CU_FILE_DEVICE_NOT_SUPPORTED:
            return hipFileDeviceNotSupported;
        case CU_FILE_NVFS_DRIVER_ERROR:
            return hipFileDriverError;
        case CU_FILE_CUDA_DRIVER_ERROR:
            return hipFileHipDriverError;
        case CU_FILE_CUDA_POINTER_INVALID:
            return hipFileHipPointerInvalid;
        case CU_FILE_CUDA_MEMORY_TYPE_INVALID:
            return hipFileHipMemoryTypeInvalid;
        case CU_FILE_CUDA_POINTER_RANGE_ERROR:
            return hipFileHipPointerRangeError;
        case CU_FILE_CUDA_CONTEXT_MISMATCH:
            return hipFileHipContextMismatch;
        case CU_FILE_INVALID_MAPPING_SIZE:
            return hipFileInvalidMappingSize;
        case CU_FILE_INVALID_MAPPING_RANGE:
            return hipFileInvalidMappingRange;
        case CU_FILE_INVALID_FILE_TYPE:
            return hipFileInvalidFileType;
        case CU_FILE_INVALID_FILE_OPEN_FLAG:
            return hipFileInvalidFileOpenFlag;
        case CU_FILE_DIO_NOT_SET:
            return hipFileDIONotSet;
        case CU_FILE_INVALID_VALUE:
            return hipFileInvalidValue;
        case CU_FILE_MEMORY_ALREADY_REGISTERED:
            return hipFileMemoryAlreadyRegistered;
        case CU_FILE_MEMORY_NOT_REGISTERED:
            return hipFileMemoryNotRegistered;
        case CU_FILE_PERMISSION_DENIED:
            return hipFilePermissionDenied;
        case CU_FILE_DRIVER_ALREADY_OPEN:
            return hipFileDriverAlreadyOpen;
        case CU_FILE_HANDLE_NOT_REGISTERED:
            return hipFileHandleNotRegistered;
        case CU_FILE_HANDLE_ALREADY_REGISTERED:
            return hipFileHandleAlreadyRegistered;
        case CU_FILE_DEVICE_NOT_FOUND:
            return hipFileDeviceNotFound;
        case CU_FILE_INTERNAL_ERROR:
            return hipFileInternalError;
        case CU_FILE_GETNEWFD_FAILED:
            return hipFileGetNewFDFailed;
        case CU_FILE_NVFS_SETUP_ERROR:
            return hipFileDriverSetupError;
        case CU_FILE_IO_DISABLED:
            return hipFileIODisabled;
        case CU_FILE_BATCH_SUBMIT_FAILED:
            return hipFileBatchSubmitFailed;
        case CU_FILE_GPU_MEMORY_PINNING_FAILED:
            return hipFileGPUMemoryPinningFailed;
        case CU_FILE_BATCH_FULL:
            return hipFileBatchFull;
        case CU_FILE_ASYNC_NOT_SUPPORTED:
            return hipFileAsyncNotSupported;
        case CU_FILE_IO_MAX_ERROR:
            return hipFileIOMaxError;
        default:
            throw std::invalid_argument("Invalid CUfileOpError value");
    }
}

CUfileOpError
toCUfileOpError(hipFileOpError_t hip_status)
{
    switch (hip_status) {
        case hipFileSuccess:
            return CU_FILE_SUCCESS;
        case hipFileDriverNotInitialized:
            return CU_FILE_DRIVER_NOT_INITIALIZED;
        case hipFileDriverInvalidProps:
            return CU_FILE_DRIVER_INVALID_PROPS;
        case hipFileDriverUnsupportedLimit:
            return CU_FILE_DRIVER_UNSUPPORTED_LIMIT;
        case hipFileDriverVersionMismatch:
            return CU_FILE_DRIVER_VERSION_MISMATCH;
        case hipFileDriverVersionReadError:
            return CU_FILE_DRIVER_VERSION_READ_ERROR;
        case hipFileDriverClosing:
            return CU_FILE_DRIVER_CLOSING;
        case hipFilePlatformNotSupported:
            return CU_FILE_PLATFORM_NOT_SUPPORTED;
        case hipFileIONotSupported:
            return CU_FILE_IO_NOT_SUPPORTED;
        case hipFileDeviceNotSupported:
            return CU_FILE_DEVICE_NOT_SUPPORTED;
        case hipFileDriverError:
            return CU_FILE_NVFS_DRIVER_ERROR;
        case hipFileHipDriverError:
            return CU_FILE_CUDA_DRIVER_ERROR;
        case hipFileHipPointerInvalid:
            return CU_FILE_CUDA_POINTER_INVALID;
        case hipFileHipMemoryTypeInvalid:
            return CU_FILE_CUDA_MEMORY_TYPE_INVALID;
        case hipFileHipPointerRangeError:
            return CU_FILE_CUDA_POINTER_RANGE_ERROR;
        case hipFileHipContextMismatch:
            return CU_FILE_CUDA_CONTEXT_MISMATCH;
        case hipFileInvalidMappingSize:
            return CU_FILE_INVALID_MAPPING_SIZE;
        case hipFileInvalidMappingRange:
            return CU_FILE_INVALID_MAPPING_RANGE;
        case hipFileInvalidFileType:
            return CU_FILE_INVALID_FILE_TYPE;
        case hipFileInvalidFileOpenFlag:
            return CU_FILE_INVALID_FILE_OPEN_FLAG;
        case hipFileDIONotSet:
            return CU_FILE_DIO_NOT_SET;
        case hipFileInvalidValue:
            return CU_FILE_INVALID_VALUE;
        case hipFileMemoryAlreadyRegistered:
            return CU_FILE_MEMORY_ALREADY_REGISTERED;
        case hipFileMemoryNotRegistered:
            return CU_FILE_MEMORY_NOT_REGISTERED;
        case hipFilePermissionDenied:
            return CU_FILE_PERMISSION_DENIED;
        case hipFileDriverAlreadyOpen:
            return CU_FILE_DRIVER_ALREADY_OPEN;
        case hipFileHandleNotRegistered:
            return CU_FILE_HANDLE_NOT_REGISTERED;
        case hipFileHandleAlreadyRegistered:
            return CU_FILE_HANDLE_ALREADY_REGISTERED;
        case hipFileDeviceNotFound:
            return CU_FILE_DEVICE_NOT_FOUND;
        case hipFileInternalError:
            return CU_FILE_INTERNAL_ERROR;
        case hipFileGetNewFDFailed:
            return CU_FILE_GETNEWFD_FAILED;
        case hipFileDriverSetupError:
            return CU_FILE_NVFS_SETUP_ERROR;
        case hipFileIODisabled:
            return CU_FILE_IO_DISABLED;
        case hipFileBatchSubmitFailed:
            return CU_FILE_BATCH_SUBMIT_FAILED;
        case hipFileGPUMemoryPinningFailed:
            return CU_FILE_GPU_MEMORY_PINNING_FAILED;
        case hipFileBatchFull:
            return CU_FILE_BATCH_FULL;
        case hipFileAsyncNotSupported:
            return CU_FILE_ASYNC_NOT_SUPPORTED;
        case hipFileIOMaxError:
            return CU_FILE_IO_MAX_ERROR;
        default:
            return CU_FILE_INVALID_VALUE;
    }
}

hipFileError_t
toHipFileError(const CUfileError_t &cu_status)
{
    hipFileError_t hipFileStatus;
    hipFileStatus.err = toHipFileOpError(cu_status.err);
    if IS_CUDA_ERR (cu_status) {
        HIP_DRV_ERR(hipFileStatus) = hipCUResultTohipError(CU_FILE_CUDA_ERR(cu_status));
    }
    else {
        HIP_DRV_ERR(hipFileStatus) = hipSuccess;
    }
    return hipFileStatus;
}

CUfileError_t
toCUfileError(hipFileError_t hip_status)
{
    CUfileError_t cufileStatus;
    cufileStatus.err = toCUfileOpError(hip_status.err);
    if IS_HIP_DRV_ERR (hip_status) {
        CU_FILE_CUDA_ERR(cufileStatus) = hipErrorToCUResult(HIP_DRV_ERR(hip_status));
    }
    else {
        CU_FILE_CUDA_ERR(cufileStatus) = CUDA_SUCCESS;
    }
    return cufileStatus;
}

hipFileOpError_t
toHipFileDriverStatusFlags(CUfileDriverStatusFlags_t cu_flags, hipFileDriverStatusFlags_t *hip_flags)
{
    switch (cu_flags) {
        case CU_FILE_LUSTRE_SUPPORTED:
            *hip_flags = hipFileLustreSupported;
            break;
        case CU_FILE_WEKAFS_SUPPORTED:
            *hip_flags = hipFileWekaFSSupported;
            break;
        case CU_FILE_NFS_SUPPORTED:
            *hip_flags = hipFileNFSSupported;
            break;
        case CU_FILE_GPFS_SUPPORTED:
            *hip_flags = hipFileGPFSSupported;
            break;
        case CU_FILE_NVME_SUPPORTED:
            *hip_flags = hipFileNVMeSupported;
            break;
        case CU_FILE_NVMEOF_SUPPORTED:
            *hip_flags = hipFileNVMeoFSupported;
            break;
        case CU_FILE_SCSI_SUPPORTED:
            *hip_flags = hipFileSCSISupported;
            break;
        case CU_FILE_SCALEFLUX_CSD_SUPPORTED:
            *hip_flags = hipFileScaleFluxCSDSupported;
            break;
        case CU_FILE_NVMESH_SUPPORTED:
            *hip_flags = hipFileNVMeshSupported;
            break;
        case CU_FILE_BEEGFS_SUPPORTED:
            *hip_flags = hipFileBeeGFSSupported;
            break;
        case CU_FILE_NVME_P2P_SUPPORTED:
            *hip_flags = hipFileNVMeP2PSupported;
            break;
        case CU_FILE_SCATEFS_SUPPORTED:
            *hip_flags = hipFileScatefsSupported;
            break;
        default:
            return hipFileInternalError;
    }

    return hipFileSuccess;
}

hipFileOpError_t
toCUfileDriverStatusFlags(hipFileDriverStatusFlags_t hip_flags, CUfileDriverStatusFlags_t *cu_flags)
{
    switch (hip_flags) {
        case hipFileLustreSupported:
            *cu_flags = CU_FILE_LUSTRE_SUPPORTED;
            break;
        case hipFileWekaFSSupported:
            *cu_flags = CU_FILE_WEKAFS_SUPPORTED;
            break;
        case hipFileNFSSupported:
            *cu_flags = CU_FILE_NFS_SUPPORTED;
            break;
        case hipFileGPFSSupported:
            *cu_flags = CU_FILE_GPFS_SUPPORTED;
            break;
        case hipFileNVMeSupported:
            *cu_flags = CU_FILE_NVME_SUPPORTED;
            break;
        case hipFileNVMeoFSupported:
            *cu_flags = CU_FILE_NVMEOF_SUPPORTED;
            break;
        case hipFileSCSISupported:
            *cu_flags = CU_FILE_SCSI_SUPPORTED;
            break;
        case hipFileScaleFluxCSDSupported:
            *cu_flags = CU_FILE_SCALEFLUX_CSD_SUPPORTED;
            break;
        case hipFileNVMeshSupported:
            *cu_flags = CU_FILE_NVMESH_SUPPORTED;
            break;
        case hipFileBeeGFSSupported:
            *cu_flags = CU_FILE_BEEGFS_SUPPORTED;
            break;
        case hipFileNVMeP2PSupported:
            *cu_flags = CU_FILE_NVME_P2P_SUPPORTED;
            break;
        case hipFileScatefsSupported:
            *cu_flags = CU_FILE_SCATEFS_SUPPORTED;
            break;
        default:
            return hipFileInternalError;
    }

    return hipFileSuccess;
}

hipFileOpError_t
toHipFileDriverControlFlags(CUfileDriverControlFlags_t cu_flags, hipFileDriverControlFlags_t *hip_flags)
{
    switch (cu_flags) {
        case CU_FILE_USE_POLL_MODE:
            *hip_flags = hipFileUsePollMode;
            break;
        case CU_FILE_ALLOW_COMPAT_MODE:
            *hip_flags = hipFileAllowCompatMode;
            break;
        default:
            return hipFileInternalError;
    }

    return hipFileSuccess;
}

hipFileOpError_t
toCUfileDriverControlFlags(hipFileDriverControlFlags_t hip_flags, CUfileDriverControlFlags_t *cu_flags)
{
    switch (hip_flags) {
        case hipFileUsePollMode:
            *cu_flags = CU_FILE_USE_POLL_MODE;
            break;
        case hipFileAllowCompatMode:
            *cu_flags = CU_FILE_ALLOW_COMPAT_MODE;
            break;
        default:
            return hipFileInternalError;
    }

    return hipFileSuccess;
}

hipFileOpError_t
toHipFileFeatureFlags(CUfileFeatureFlags_t cu_flags, hipFileFeatureFlags_t *hip_flags)
{
    switch (cu_flags) {
        case CU_FILE_DYN_ROUTING_SUPPORTED:
            *hip_flags = hipFileDynRoutingSupported;
            break;
        case CU_FILE_BATCH_IO_SUPPORTED:
            *hip_flags = hipFileBatchIOSupported;
            break;
        case CU_FILE_STREAMS_SUPPORTED:
            *hip_flags = hipFileStreamsSupported;
            break;
        case CU_FILE_PARALLEL_IO_SUPPORTED:
            *hip_flags = hipFileParallelIOSupported;
            break;
        default:
            return hipFileInternalError;
    }

    return hipFileSuccess;
}

hipFileOpError_t
toCUfileFeatureFlags(hipFileFeatureFlags_t hip_flags, CUfileFeatureFlags_t *cu_flags)
{
    switch (hip_flags) {
        case hipFileDynRoutingSupported:
            *cu_flags = CU_FILE_DYN_ROUTING_SUPPORTED;
            break;
        case hipFileBatchIOSupported:
            *cu_flags = CU_FILE_BATCH_IO_SUPPORTED;
            break;
        case hipFileStreamsSupported:
            *cu_flags = CU_FILE_STREAMS_SUPPORTED;
            break;
        case hipFileParallelIOSupported:
            *cu_flags = CU_FILE_PARALLEL_IO_SUPPORTED;
            break;
        default:
            return hipFileInternalError;
    }

    return hipFileSuccess;
}

hipFileOpError_t
toHipFileFileHandleType(enum CUfileFileHandleType cu_type, enum hipFileFileHandleType *hip_type)
{
    switch (cu_type) {
        case CU_FILE_HANDLE_TYPE_OPAQUE_FD:
            *hip_type = hipFileHandleTypeOpaqueFD;
            break;
        case CU_FILE_HANDLE_TYPE_OPAQUE_WIN32:
            *hip_type = hipFileHandleTypeOpaqueWin32;
            break;
        case CU_FILE_HANDLE_TYPE_USERSPACE_FS:
            *hip_type = hipFileHandleTypeUserspaceFS;
            break;
        default:
            return hipFileInternalError;
    }

    return hipFileSuccess;
}

CUfileFileHandleType
toCUfileFileHandleType(hipFileFileHandleType_t hip_type)
{
    switch (hip_type) {
        case hipFileHandleTypeOpaqueFD:
            return CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        case hipFileHandleTypeOpaqueWin32:
            return CU_FILE_HANDLE_TYPE_OPAQUE_WIN32;
        case hipFileHandleTypeUserspaceFS:
            return CU_FILE_HANDLE_TYPE_USERSPACE_FS;
        default:
            throw std::invalid_argument("Invalid hipFileFileHandleType_t value");
    }
}

/*
 * AIS Team Note:
 *
 * HIP seems to do 1 of 3 things when it comes to converting
 * HIP data types to CUDA data types (or vice-versa).
 *      1) C-Style (typecast) / C++ <reinterpret_cast>
 *      2) typedef cudaDataType hipDataType
 *         a) For enums, #define are used to "rename" the enum members.
 *      3) An explicit conversion function.
 *
 * There does not seem to be any documentation to suggest a preferred
 * method.
 *
 * Several HIP <=> CUDA data type conversions are typically done on either
 * primitive data types or on opaque file handles.
 *
 * Conversions using typedefs are typically done on structs/enums where
 * at compile time you are targeting the actual CUDA symbol.
 *
 * Otherwise, for other data types where there isn't a perfect mapping,
 * a conversion function is typically used.
 *
 * For this first attempt, while it is more verbose, we'll use a conversion
 * function for now for safety and flexibility. We can try changing this later
 * once we have adequate testing.
 */

hipFileDriverProps_t
toHipFileDriverProps(const CUfileDrvProps_t &cu_props)
{
    hipFileDriverProps_t hip_props;

    hip_props.nvfs.major_version         = cu_props.nvfs.major_version;
    hip_props.nvfs.minor_version         = cu_props.nvfs.minor_version;
    hip_props.nvfs.poll_thresh_size      = cu_props.nvfs.poll_thresh_size;
    hip_props.nvfs.max_direct_io_size    = cu_props.nvfs.max_direct_io_size;
    hip_props.nvfs.driver_status_flags   = cu_props.nvfs.dstatusflags;
    hip_props.nvfs.driver_control_flags  = cu_props.nvfs.dcontrolflags;
    hip_props.feature_flags              = cu_props.fflags;
    hip_props.max_device_cache_size      = cu_props.max_device_cache_size;
    hip_props.per_buffer_cache_size      = cu_props.per_buffer_cache_size;
    hip_props.max_device_pinned_mem_size = cu_props.max_device_pinned_mem_size;
    hip_props.max_batch_io_count         = cu_props.max_batch_io_size;
    hip_props.max_batch_io_timeout_msecs = cu_props.max_batch_io_timeout_msecs;

    return hip_props;
}

hipFileOpError_t
toCUfileDrvProps(const hipFileDriverProps_t *hip_props, CUfileDrvProps_t *cu_props)
{
    // Be careful of potential invalid casts
    if (hip_props->max_device_cache_size > UINT_MAX) {
        return hipFileDriverInvalidProps;
    }
    if (hip_props->per_buffer_cache_size > UINT_MAX) {
        return hipFileDriverInvalidProps;
    }
    if (hip_props->max_device_pinned_mem_size > UINT_MAX) {
        return hipFileDriverInvalidProps;
    }

    cu_props->nvfs.major_version      = hip_props->nvfs.major_version;
    cu_props->nvfs.minor_version      = hip_props->nvfs.minor_version;
    cu_props->nvfs.poll_thresh_size   = hip_props->nvfs.poll_thresh_size;
    cu_props->nvfs.max_direct_io_size = hip_props->nvfs.max_direct_io_size;
    cu_props->nvfs.dstatusflags       = hip_props->nvfs.driver_status_flags;
    cu_props->nvfs.dcontrolflags      = hip_props->nvfs.driver_control_flags;

    cu_props->fflags                     = hip_props->feature_flags;
    cu_props->max_device_cache_size      = static_cast<unsigned>(hip_props->max_device_cache_size);
    cu_props->per_buffer_cache_size      = static_cast<unsigned>(hip_props->per_buffer_cache_size);
    cu_props->max_device_pinned_mem_size = static_cast<unsigned>(hip_props->max_device_pinned_mem_size);
    cu_props->max_batch_io_size          = hip_props->max_batch_io_count;
    cu_props->max_batch_io_timeout_msecs = hip_props->max_batch_io_timeout_msecs;

    return hipFileSuccess;
}

hipFileOpError_t
toHipFileRDMAInfo(const cufileRDMAInfo_t *cu_info, hipFileRDMAInfo_t *hip_info)
{
    hip_info->version  = cu_info->version;
    hip_info->desc_len = cu_info->desc_len;
    hip_info->desc_str = cu_info->desc_str;

    return hipFileSuccess;
}

hipFileOpError_t
toCufileRDMAInfo(const hipFileRDMAInfo_t *hip_info, cufileRDMAInfo_t *cu_info)
{
    cu_info->version  = hip_info->version;
    cu_info->desc_len = hip_info->desc_len;
    cu_info->desc_str = hip_info->desc_str;

    return hipFileSuccess;
}

/*
 * FSOps behaves differently from other cuFile data types that we have
 * seen. We have not found any examples that use it directly, and it is
 * stored in hipFileDescr_t as a pointer.
 *
 * We think cuFile may internally be responsible for allocating the memory
 * required for FSOps when needed, rather than the user initializing this
 * variable.
 *
 * Because this exists as a pointer, we will typecast it for now rather than
 * providing explicit conversion functions.
 */

hipFileOpError_t
toHipFileDescr(const CUfileDescr_t *cu_fd, hipFileDescr_t *hip_fd)
{
    hipFileFileHandleType_t type;

    hipFileOpError_t error = toHipFileFileHandleType(cu_fd->type, &type);
    if (error == hipFileInternalError) {
        return hipFileInternalError;
    }

    hip_fd->type = type;
    switch (cu_fd->type) {
        case CU_FILE_HANDLE_TYPE_OPAQUE_FD:
            hip_fd->handle.fd = cu_fd->handle.fd;
            break;
        case CU_FILE_HANDLE_TYPE_OPAQUE_WIN32:
            hip_fd->handle.hFile = cu_fd->handle.handle;
            break;
        case CU_FILE_HANDLE_TYPE_USERSPACE_FS:
            // Assuming this is POSIX-only...
            hip_fd->handle.fd = cu_fd->handle.fd;
            break;
        default:
            return hipFileInternalError;
    }
    hip_fd->fs_ops = reinterpret_cast<const hipFileFSOps_t *>(cu_fd->fs_ops);

    return hipFileSuccess;
}

CUfileDescr_t
toCUfileDescr(const hipFileDescr_t &hip_fd)
{
    CUfileDescr_t cu_fd;

    cu_fd.fs_ops = reinterpret_cast<const CUfileFSOps_t *>(hip_fd.fs_ops);

    switch (hip_fd.type) {
        case hipFileHandleTypeOpaqueFD:
            cu_fd.type      = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
            cu_fd.handle.fd = hip_fd.handle.fd;
            break;
        case hipFileHandleTypeOpaqueWin32:
            cu_fd.type          = CU_FILE_HANDLE_TYPE_OPAQUE_WIN32;
            cu_fd.handle.handle = hip_fd.handle.hFile;
            break;
        case hipFileHandleTypeUserspaceFS:
            // Assuming this is POSIX-only...
            cu_fd.type      = CU_FILE_HANDLE_TYPE_USERSPACE_FS;
            cu_fd.handle.fd = hip_fd.handle.fd;
            break;
        default:
            throw std::invalid_argument("Invalid hipFileFileHandleType_t value");
    }

    return cu_fd;
}

hipFileOpError_t
toHipFileOpcode(CUfileOpcode_t cu_opcode, hipFileOpcode_t *hip_opcode)
{
    switch (cu_opcode) {
        case CUFILE_READ:
            *hip_opcode = hipFileBatchRead;
            break;
        case CUFILE_WRITE:
            *hip_opcode = hipFileBatchWrite;
            break;
        default:
            return hipFileInternalError;
    }

    return hipFileSuccess;
}

CUfileOpcode_t
toCUfileOpcode(hipFileOpcode_t hip_opcode)
{
    switch (hip_opcode) {
        case hipFileBatchRead:
            return CUFILE_READ;
        case hipFileBatchWrite:
            return CUFILE_WRITE;
        default:
            throw std::invalid_argument("Invalid hipFileOpcode_t value");
    }
}

hipFileStatus_t
toHipFileStatus(CUfileStatus_t cu_status)
{
    switch (cu_status) {
        case CUFILE_WAITING:
            return hipFileWaiting;
        case CUFILE_PENDING:
            return hipFilePending;
        case CUFILE_INVALID:
            return hipFileInvalid;
        case CUFILE_CANCELED:
            return hipFileCanceled;
        case CUFILE_COMPLETE:
            return hipFileComplete;
        case CUFILE_TIMEOUT:
            return hipFileTimeout;
        case CUFILE_FAILED:
            return hipFileFailed;
        default:
            throw std::invalid_argument("Invalid CUfileStatus_t value");
    }
}

hipFileOpError_t
toCUfileStatus(hipFileStatus_t hip_status, CUfileStatus_t *cu_status)
{
    switch (hip_status) {
        case hipFileWaiting:
            *cu_status = CUFILE_WAITING;
            break;
        case hipFilePending:
            *cu_status = CUFILE_PENDING;
            break;
        case hipFileInvalid:
            *cu_status = CUFILE_INVALID;
            break;
        case hipFileCanceled:
            *cu_status = CUFILE_CANCELED;
            break;
        case hipFileComplete:
            *cu_status = CUFILE_COMPLETE;
            break;
        case hipFileTimeout:
            *cu_status = CUFILE_TIMEOUT;
            break;
        case hipFileFailed:
            *cu_status = CUFILE_FAILED;
            break;
        default:
            return hipFileInternalError;
    }

    return hipFileSuccess;
}

hipFileOpError_t
toHipFileBatchMode(CUfileBatchMode_t cu_mode, hipFileBatchMode_t *hip_mode)
{
    switch (cu_mode) {
        case CUFILE_BATCH:
            *hip_mode = hipFileBatch;
            break;
        default:
            return hipFileInternalError;
    }

    return hipFileSuccess;
}

CUfileBatchMode_t
toCUfileBatchMode(hipFileBatchMode_t hip_mode)
{
    switch (hip_mode) {
        case hipFileBatch:
            return CUFILE_BATCH;
        default:
            throw std::invalid_argument("Invalid hipFileBatchMode_t value");
    }
}

hipFileOpError_t
toHipFileIOParams(const CUfileIOParams_t *cu_params, hipFileIOParams_t *hip_params)
{
    hipFileBatchMode_t batch_mode;
    hipFileOpcode_t    opcode;

    // Don't modify hip_params if these two conversions fail.
    hipFileOpError_t error = toHipFileBatchMode(cu_params->mode, &batch_mode);
    if (error == hipFileInternalError) {
        return hipFileInternalError;
    }
    error = toHipFileOpcode(cu_params->opcode, &opcode);
    if (error == hipFileInternalError) {
        return hipFileInternalError;
    }

    hip_params->mode = batch_mode;

    hip_params->u.batch.devPtr_base   = cu_params->u.batch.devPtr_base;
    hip_params->u.batch.file_offset   = cu_params->u.batch.file_offset;
    hip_params->u.batch.devPtr_offset = cu_params->u.batch.devPtr_offset;
    hip_params->u.batch.size          = cu_params->u.batch.size;

    hip_params->fh     = cu_params->fh;
    hip_params->opcode = opcode;
    hip_params->cookie = cu_params->cookie;

    return hipFileSuccess;
}

CUfileIOParams_t
toCUfileIOParams(const hipFileIOParams_t &hip_params)
{
    CUfileIOParams_t cu_params;

    cu_params.mode                  = toCUfileBatchMode(hip_params.mode);
    cu_params.u.batch.devPtr_base   = hip_params.u.batch.devPtr_base;
    cu_params.u.batch.file_offset   = hip_params.u.batch.file_offset;
    cu_params.u.batch.devPtr_offset = hip_params.u.batch.devPtr_offset;
    cu_params.u.batch.size          = hip_params.u.batch.size;
    cu_params.fh                    = hip_params.fh;
    cu_params.opcode                = toCUfileOpcode(hip_params.opcode);
    cu_params.cookie                = hip_params.cookie;

    return cu_params;
}

hipFileIOEvents_t
toHipFileIOEvents(const CUfileIOEvents_t &cu_events)
{
    hipFileIOEvents_t hip_events;

    hip_events.cookie = cu_events.cookie;
    hip_events.status = toHipFileStatus(cu_events.status);
    hip_events.ret    = cu_events.ret;

    return hip_events;
}

hipFileOpError_t
toCUfileIOEvents(const hipFileIOEvents_t *hip_events, CUfileIOEvents_t *cu_events)
{
    CUfileStatus_t status;

    hipFileOpError_t error = toCUfileStatus(hip_events->status, &status);
    if (error == hipFileInternalError) {
        return hipFileInternalError;
    }

    cu_events->cookie = hip_events->cookie;
    cu_events->status = status;
    cu_events->ret    = hip_events->ret;

    return hipFileSuccess;
}

CUFileSizeTConfigParameter_t
toCUFileSizeTConfigParameter(hipFileSizeTConfigParameter_t param)
{
    switch (param) {
        case hipFileParamProfileStats:
            return CUFILE_PARAM_PROFILE_STATS;
        case hipFileParamExecutionMaxIOQueueDepth:
            return CUFILE_PARAM_EXECUTION_MAX_IO_QUEUE_DEPTH;
        case hipFileParamExecutionMaxIOThreads:
            return CUFILE_PARAM_EXECUTION_MAX_IO_THREADS;
        case hipFileParamExecutionMinIOThresholdSizeKB:
            return CUFILE_PARAM_EXECUTION_MIN_IO_THRESHOLD_SIZE_KB;
        case hipFileParamExecutionMaxRequestParallelism:
            return CUFILE_PARAM_EXECUTION_MAX_REQUEST_PARALLELISM;
        case hipFileParamPropertiesMaxDirectIOSizeKB:
            return CUFILE_PARAM_PROPERTIES_MAX_DIRECT_IO_SIZE_KB;
        case hipFileParamPropertiesMaxDeviceCacheSizeKB:
            return CUFILE_PARAM_PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB;
        case hipFileParamPropertiesPerBufferCacheSizeKB:
            return CUFILE_PARAM_PROPERTIES_PER_BUFFER_CACHE_SIZE_KB;
        case hipFileParamPropertiesMaxDevicePinnedMemSizeKB:
            return CUFILE_PARAM_PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB;
        case hipFileParamPropertiesIOBatchsize:
            return CUFILE_PARAM_PROPERTIES_IO_BATCHSIZE;
        case hipFileParamPollthresholdSizeKB:
            return CUFILE_PARAM_POLLTHRESHOLD_SIZE_KB;
        case hipFileParamPropertiesBatchIOTimeoutMs:
            return CUFILE_PARAM_PROPERTIES_BATCH_IO_TIMEOUT_MS;
        default:
            throw std::invalid_argument("Invalid hipFileSizeTConfigParameter_t value");
    }
}

CUFileBoolConfigParameter_t
toCUFileBoolConfigParameter(hipFileBoolConfigParameter_t param)
{
    switch (param) {
        case hipFileParamPropertiesUsePollMode:
            return CUFILE_PARAM_PROPERTIES_USE_POLL_MODE;
        case hipFileParamPropertiesAllowCompatMode:
            return CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE;
        case hipFileParamForceCompatMode:
            return CUFILE_PARAM_FORCE_COMPAT_MODE;
        case hipFileParamFsMiscApiCheckAggressive:
            return CUFILE_PARAM_FS_MISC_API_CHECK_AGGRESSIVE;
        case hipFileParamExecutionParallelIO:
            return CUFILE_PARAM_EXECUTION_PARALLEL_IO;
        case hipFileParamProfileNvtx:
            return CUFILE_PARAM_PROFILE_NVTX;
        case hipFileParamPropertiesAllowSystemMemory:
            return CUFILE_PARAM_PROPERTIES_ALLOW_SYSTEM_MEMORY;
        case hipFileParamUsePcip2pdma:
            return CUFILE_PARAM_USE_PCIP2PDMA;
        case hipFileParamPreferIOUring:
            return CUFILE_PARAM_PREFER_IO_URING;
        case hipFileParamForceOdirectMode:
            return CUFILE_PARAM_FORCE_ODIRECT_MODE;
        case hipFileParamSkipTopologyDetection:
            return CUFILE_PARAM_SKIP_TOPOLOGY_DETECTION;
        case hipFileParamStreamMemopsBypass:
            return CUFILE_PARAM_STREAM_MEMOPS_BYPASS;
        default:
            throw std::invalid_argument("Invalid hipFileBoolConfigParameter_t value");
    }
}

CUFileStringConfigParameter_t
toCUFileStringConfigParameter(hipFileStringConfigParameter_t param)
{
    switch (param) {
        case hipFileParamLoggingLevel:
            return CUFILE_PARAM_LOGGING_LEVEL;
        case hipFileParamEnvLogfilePath:
            return CUFILE_PARAM_ENV_LOGFILE_PATH;
        case hipFileParamLogDir:
            return CUFILE_PARAM_LOG_DIR;
        default:
            throw std::invalid_argument("Invalid hipFileStringConfigParameter_t value");
    }
}
