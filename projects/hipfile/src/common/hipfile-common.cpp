/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hipfile.h"

#include <hip/hip_runtime_api.h>

const char *
hipFileGetOpErrorString(hipFileOpError_t status)
{
    switch (status) {
        case hipFileSuccess:
            return "hipFile success";
        case hipFileDriverNotInitialized:
            return "GPU IO driver is not loaded";
        case hipFileDriverInvalidProps:
            return "Invalid GPU IO driver property provided";
        case hipFileDriverUnsupportedLimit:
            return "GPU IO Driver property value is unsupported";
        case hipFileDriverVersionMismatch:
            return "hipFile version does not match GPU IO driver version";
        case hipFileDriverVersionReadError:
            return "Unable to read the GPU IO driver version";
        case hipFileDriverClosing:
            return "GPU IO driver is closing and not accepting new requests";
        case hipFilePlatformNotSupported:
            return "hipFile is not supported on the current platform";
        case hipFileIONotSupported:
            return "hipFile is not supported on the selected file";
        case hipFileDeviceNotSupported:
            return "The selected GPU does not support hipFile";
        case hipFileDriverError:
            return "GPU IO driver error";
        case hipFileHipDriverError:
            return "GPU driver error: Inspect hip_drv_err for additional information";
        case hipFileHipPointerInvalid:
            return "Invalid GPU pointer";
        case hipFileHipMemoryTypeInvalid:
            return "Memory type backing pointer is incompatible with hipFile";
        case hipFileHipPointerRangeError:
            return "Pointer range exceeds allocated memory region";
        case hipFileHipContextMismatch:
            return "GPU driver context mismatch";
        case hipFileInvalidMappingSize:
            return "Accessing memory beyond pinned memory buffer";
        case hipFileInvalidMappingRange:
            return "Accessing memory beyond mapped memory region";
        case hipFileInvalidFileType:
            return "Unsupported file type";
        case hipFileInvalidFileOpenFlag:
            return "Unsupported file open flags";
        case hipFileDIONotSet:
            return "O_DIRECT flag not set";
        case hipFileInvalidValue:
            return "One or more arguments have an invalid value";
        case hipFileMemoryAlreadyRegistered:
            return "Device pointer is already registered";
        case hipFileMemoryNotRegistered:
            return "Device pointer is not registered";
        case hipFilePermissionDenied:
            return "Permission error on device or file access";
        case hipFileDriverAlreadyOpen:
            return "GPU IO driver is already open";
        case hipFileHandleNotRegistered:
            return "File handle for GPU IO is not registered";
        case hipFileHandleAlreadyRegistered:
            return "File handle for GPU IO is already registered";
        case hipFileDeviceNotFound:
            return "Selected device not found";
        case hipFileInternalError:
            return "Internal GPU IO library error";
        case hipFileGetNewFDFailed:
            return "Unable to obtain a new file descriptor";
        case hipFileDriverSetupError:
            return "GPU IO Driver initialization error";
        case hipFileIODisabled:
            return "GPU IO config file prohibits GPU IO on specified file";
        case hipFileBatchSubmitFailed:
            return "Failed to submit request for batch operation";
        case hipFileGPUMemoryPinningFailed:
            return "Failed to allocated pinned device memory";
        case hipFileBatchFull:
            return "Batch operation queue is full";
        case hipFileAsyncNotSupported:
            return "hipFile async IO is not supported";
        case hipFileIOMaxError:
            return "Internal flag to mark largest hipFile error code";
        default:
            return "Unknown hipFile error";
    }
}

hipFileError_t
hipFileGetVersion(unsigned *major, unsigned *minor, unsigned *patch)
try {
    // NULL parameters are ignored
    if (major != nullptr) {
        *major = HIPFILE_VERSION_MAJOR;
    }
    if (minor != nullptr) {
        *minor = HIPFILE_VERSION_MINOR;
    }
    if (patch != nullptr) {
        *patch = HIPFILE_VERSION_PATCH;
    }

    return {hipFileSuccess, hipSuccess};
}
catch (...) {
    return {hipFileInternalError, hipSuccess};
}
