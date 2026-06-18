/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "backend.h"
#include "backend/fallback.h"
#include "batch/batch.h"
#include "buffer.h"
#include "context.h"
#include "file.h"
#include "hip.h"
#include "hipfile.h"
#include "hipfile-private.h"
#include "hipfile-warnings.h"
#include "api_trace/api-trace-internal.h"
#include "io.h"
#include "state.h"
#include "stats.h"

#include <cerrno>
#include <cstdint>
#include <hip/hip_runtime_api.h>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include <sys/types.h>
#include <system_error>

using namespace std;

namespace hipFile {

/// Catch C++ exceptions from the hipFile code and convert
/// them into error values that can be returned from public
/// C API calls.
static inline hipFileError_t
handle_exception() noexcept
try {
    throw;
}
catch (hipFileError_t e) {
    return e;
}
catch (const Hip::RuntimeError &e) {
    return {hipFileHipDriverError, e.error};
}
catch (...) {
    return {hipFileInternalError, hipSuccess};
}

static void
checkNull(std::initializer_list<void *> ptrs)
{
    for (auto ptr : ptrs) {
        if (!ptr) {
            throw hipFileError_t{hipFileInvalidValue, hipSuccess};
        }
    }
}

// Ensures that the driver is initialized without incrementing the reference
// count if the driver is already initialized. Used by hipFile to ensure its
// behaviour is consistent on AMD and NVIDIA
static inline void
ensureDriverInit()
{
    hipFile::Context<hipFile::DriverState>::get()->ensureInitialized();
}

hipFileError_t
hipFileHandleRegister(hipFileHandle_t *fh, hipFileDescr_t *descr)
try {
    hipFileInit();
    if (fh == nullptr || descr == nullptr) {
        return {hipFileInvalidValue, hipSuccess};
    }

    switch (descr->type) {
        case hipFileHandleTypeOpaqueFD: {
            UnregisteredFile uf{descr->handle.fd};
            *fh = Context<DriverState>::get()->registerFile(std::move(uf));
            Context<StatsCollection>::get()->fileRegistration();
            return {hipFileSuccess, hipSuccess};
        }
        case hipFileHandleTypeOpaqueWin32:
        case hipFileHandleTypeUserspaceFS:
        default:
            return {hipFileIONotSupported, hipSuccess};
    }
}
catch (const FileAlreadyRegistered &) {
    return {hipFileHandleAlreadyRegistered, hipSuccess};
}
catch (...) {
    return handle_exception();
}

void
hipFileHandleDeregister(hipFileHandle_t fh)
try {
    hipFileInit();
    if (fh == nullptr) {
        return;
    }

    Context<DriverState>::get()->deregisterFile(fh);
    return;
}
catch (...) {
    return;
}

hipFileError_t
hipFileBufRegister(const void *buffer_base, size_t length, int flags)
try {
    hipFileInit();
    Context<DriverState>::get()->registerBuffer(buffer_base, length, flags);
    Context<StatsCollection>::get()->bufferRegistration();
    return {hipFileSuccess, hipSuccess};
}
catch (const BufferAlreadyRegistered &) {
    return {hipFileMemoryAlreadyRegistered, hipSuccess};
}
catch (const InvalidMemoryType &) {
    return {hipFileHipMemoryTypeInvalid, hipSuccess};
}
catch (const InvalidPointerRange &) {
    return {hipFileHipPointerRangeError, hipSuccess};
}
catch (const Hip::RuntimeError &e) {
    if (e.error == hipErrorInvalidValue) {
        return {hipFileInvalidValue, hipSuccess};
    }
    return handle_exception();
}
catch (const std::invalid_argument &) {
    return {hipFileInvalidValue, hipSuccess};
}
catch (...) {
    return handle_exception();
}

hipFileError_t
hipFileBufDeregister(const void *buffer_base)
try {
    hipFileInit();
    Context<DriverState>::get()->deregisterBuffer(buffer_base);
    return {hipFileSuccess, hipSuccess};
}
catch (const DriverNotInitialized &) {
    // Mimic cuFile and return hipFileDriverClosing instead
    // of hipFileDrivernotInitialized
    return {hipFileDriverClosing, hipSuccess};
}
catch (const BufferNotRegistered &) {
    return {hipFileMemoryNotRegistered, hipSuccess};
}
catch (const BufferOperationsOutstanding &) {
    return {hipFileInternalError, hipSuccess};
}
catch (...) {
    return handle_exception();
}

/// @brief Get the cached list of backends obtained from DriverState
static const vector<shared_ptr<Backend>> &
getCachedBackends()
{
    HIPFILE_WARN_NO_EXIT_DTOR_OFF
    static const auto backends{Context<DriverState>::get()->getBackends()};
    HIPFILE_WARN_NO_EXIT_DTOR_ON
    return backends;
}

ssize_t
hipFileIo(IoType type, hipFileHandle_t fh, const void *buffer_base, size_t size, hoff_t file_offset,
          hoff_t buffer_offset, const vector<shared_ptr<Backend>> &backends)
try {
    auto [file, buffer] = Context<DriverState>::get()->getFileAndBuffer(fh, buffer_base);
    int                      score{-1};
    std::shared_ptr<Backend> backend{};

    for (const auto &_backend : backends) {
        auto _score = _backend->score(file, buffer, size, file_offset, buffer_offset);
        if (score < _score) {
            score   = _score;
            backend = _backend;
        }
    }

    if (!backend) {
        if (backends.size() == 0) {
            throw std::runtime_error("No available backends for IO request");
        }
        else {
            throw std::runtime_error("All backends refused IO request");
        }
    }

    return backend->io(type, file, buffer, size, file_offset, buffer_offset);
}
catch (const DriverNotInitialized &) {
    return -hipFileDriverNotInitialized;
}
catch (hipFileError_t e) {
    return -e.err;
}
catch (const InvalidMemoryType &) {
    return -hipFileHipMemoryTypeInvalid;
}
catch (const std::invalid_argument &) {
    return -hipFileInvalidValue;
}
catch (const FileNotRegistered &) {
    return -hipFileHandleNotRegistered;
}
catch (const Hip::RuntimeError &e) {
    return -e.error;
}
catch (const std::system_error &e) {
    errno = e.code().value();
    return -1;
}
catch (...) {
    return -hipFileInternalError;
}

ssize_t
hipFileRead(hipFileHandle_t fh, void *buffer_base, size_t size, hoff_t file_offset, hoff_t buffer_offset)
try {
    hipFileInit();
    auto result =
        hipFileIo(IoType::Read, fh, buffer_base, size, file_offset, buffer_offset, getCachedBackends());

    if (result == -hipFileDriverNotInitialized) {
        // Match cuFile behaviour
        errno  = EINVAL;
        result = -1;
    }
    return result;
}
catch (...) {
    hipFileError_t err = handle_exception();
    return -err.err;
}

ssize_t
hipFileWrite(hipFileHandle_t fh, const void *buffer_base, size_t size, hoff_t file_offset,
             hoff_t buffer_offset)
try {
    hipFileInit();
    auto result =
        hipFileIo(IoType::Write, fh, buffer_base, size, file_offset, buffer_offset, getCachedBackends());

    if (result == -hipFileDriverNotInitialized) {
        // Match cuFile behaviour
        errno  = EINVAL;
        result = -1;
    }
    return result;
}
catch (...) {
    hipFileError_t err = handle_exception();
    return -err.err;
}

hipFileError_t
hipFileDriverOpen()
try {
    hipFileInit();
    Context<DriverState>::get()->incrRefCount();

    return {hipFileSuccess, hipSuccess};
}
catch (...) {
    return handle_exception();
}

hipFileError_t
hipFileDriverClose()
try {
    hipFileInit();
    if (Context<DriverState>::get()->getRefCount() > 0) {
        Context<DriverState>::get()->decrRefCount();
        return {hipFileSuccess, hipSuccess};
    }
    else {
        return {hipFileDriverNotInitialized, hipSuccess};
    }
}
catch (...) {
    return handle_exception();
}

int64_t
hipFileUseCount()
try {
    hipFileInit();
    return Context<DriverState>::get()->getRefCount();
}
catch (...) {
    return -1;
}

hipFileError_t
hipFileDriverGetProperties(hipFileDriverProps_t *props)
try {
    hipFileInit();
    (void)props;

    throw std::runtime_error("Not Implemented");
}
catch (...) {
    return handle_exception();
}

hipFileError_t
hipFileDriverSetPollMode(bool poll, size_t poll_threshold_size)
try {
    hipFileInit();
    (void)poll;
    (void)poll_threshold_size;

    throw std::runtime_error("Not Implemented");
}
catch (...) {
    return handle_exception();
}

hipFileError_t
hipFileDriverSetMaxDirectIOSize(size_t max_direct_io_size)
try {
    hipFileInit();
    (void)max_direct_io_size;

    throw std::runtime_error("Not Implemented");
}
catch (...) {
    return handle_exception();
}

hipFileError_t
hipFileDriverSetMaxCacheSize(size_t max_cache_size)
try {
    hipFileInit();
    (void)max_cache_size;

    throw std::runtime_error("Not Implemented");
}
catch (...) {
    return handle_exception();
}

hipFileError_t
hipFileDriverSetMaxPinnedMemSize(size_t max_pinned_size)
try {
    hipFileInit();
    (void)max_pinned_size;

    throw std::runtime_error("Not Implemented");
}
catch (...) {
    return handle_exception();
}

hipFileError_t
hipFileBatchIOSetUp(hipFileBatchHandle_t *batch_idp, unsigned max_nr)
try {
    hipFileInit();
    if (batch_idp == nullptr) {
        return {hipFileInvalidValue, hipSuccess};
    }

    *batch_idp = Context<DriverState>::get()->createBatchContext(max_nr);

    return {hipFileSuccess, hipSuccess};
}
catch (const std::invalid_argument &) {
    return {hipFileInvalidValue, hipSuccess};
}
catch (...) {
    return handle_exception();
}

hipFileError_t
hipFileBatchIOSubmit(hipFileBatchHandle_t batch_idp, unsigned nr, hipFileIOParams_t *iocbp, unsigned flags)
try {
    hipFileInit();
    (void)flags; // Unused at this time.

    std::shared_ptr<IBatchContext> batch_context = Context<DriverState>::get()->getBatchContext(batch_idp);
    batch_context->submit_operations(iocbp, nr);

    return {hipFileSuccess, hipSuccess};
}
catch (const std::invalid_argument &) {
    return {hipFileInvalidValue, hipSuccess};
}
catch (...) {
    return handle_exception();
}

hipFileError_t
hipFileBatchIOGetStatus(hipFileBatchHandle_t batch_idp, unsigned min_nr, unsigned *nr,
                        hipFileIOEvents_t *iocbp, struct timespec *timeout)
try {
    hipFileInit();
    (void)batch_idp;
    (void)min_nr;
    (void)nr;
    (void)iocbp;
    (void)timeout;

    throw std::runtime_error("Not Implemented");
}
catch (...) {
    return handle_exception();
}

hipFileError_t
hipFileBatchIOCancel(hipFileBatchHandle_t batch_idp)
try {
    hipFileInit();
    (void)batch_idp;

    throw std::runtime_error("Not Implemented");
}
catch (...) {
    return handle_exception();
}

void
hipFileBatchIODestroy(hipFileBatchHandle_t batch_idp)
try {
    hipFileInit();
    (void)batch_idp;

    throw std::runtime_error("Not Implemented");
}
catch (...) {
    return;
}

static hipFileError_t
hipFileIOAsync(IoType io_type, hipFileHandle_t fh, void *buffer_base, size_t *size_p, hoff_t *file_offset_p,
               hoff_t *buffer_offset_p, ssize_t *bytes_transferred_p, hipStream_t hipStream)
try {
    if (Context<DriverState>::get()->getRefCount() == 0) {
        ensureDriverInit();
        return {hipFileInvalidValue, hipSuccess};
    }

    checkNull({buffer_base, size_p, file_offset_p, buffer_offset_p, bytes_transferred_p});

    auto [file, buffer, stream] =
        Context<DriverState>::get()->getFileBufferAndStream(fh, buffer_base, hipStream);
    Fallback().async_io(io_type, file, buffer, size_p, file_offset_p, buffer_offset_p, bytes_transferred_p,
                        stream);

    return {hipFileSuccess, hipSuccess};
}
catch (hipFileError_t e) {
    return e;
}
catch (const std::invalid_argument &) {
    return {hipFileInvalidValue, hipSuccess};
}
catch (const std::bad_alloc &) {
    return {hipFileHipDriverError, hipErrorOutOfMemory};
}
catch (const FileNotRegistered &) {
    return {hipFileHandleNotRegistered, hipSuccess};
}
catch (...) {
    return handle_exception();
}

hipFileError_t
hipFileReadAsync(hipFileHandle_t fh, void *buffer_base, size_t *size_p, hoff_t *file_offset_p,
                 hoff_t *buffer_offset_p, ssize_t *bytes_read_p, hipStream_t stream)
try {
    hipFileInit();
    return hipFileIOAsync(IoType::Read, fh, buffer_base, size_p, file_offset_p, buffer_offset_p, bytes_read_p,
                          stream);
}
catch (...) {
    return handle_exception();
}

hipFileError_t
hipFileWriteAsync(hipFileHandle_t fh, void *buffer_base, size_t *size_p, hoff_t *file_offset_p,
                  hoff_t *buffer_offset_p, ssize_t *bytes_written_p, hipStream_t stream)
try {
    hipFileInit();
    return hipFileIOAsync(IoType::Write, fh, buffer_base, size_p, file_offset_p, buffer_offset_p,
                          bytes_written_p, stream);
}
catch (...) {
    return handle_exception();
}

hipFileError_t
hipFileStreamRegister(hipStream_t stream, unsigned flags)
try {
    hipFileInit();
    Context<DriverState>::get()->registerStream(stream, flags);
    return {hipFileSuccess, hipSuccess};
}
catch (std::invalid_argument &) {
    return {hipFileInvalidValue, hipSuccess};
}
catch (...) {
    return handle_exception();
}

hipFileError_t
hipFileStreamDeregister(hipStream_t stream)
try {
    hipFileInit();
    Context<DriverState>::get()->deregisterStream(stream);
    return {hipFileSuccess, hipSuccess};
}
catch (std::invalid_argument &) {
    return {hipFileInvalidValue, hipSuccess};
}
catch (...) {
    return handle_exception();
}

// ***********************************************************************
//  PROPERTIES API
// ***********************************************************************

hipFileError_t
hipFileGetParameterSizeT(hipFileSizeTConfigParameter_t param, size_t *value)
try {
    hipFileInit();
    (void)param;
    (void)value;

    throw std::runtime_error("Not Implemented");
}
catch (...) {
    return handle_exception();
}

hipFileError_t
hipFileGetParameterBool(hipFileBoolConfigParameter_t param, bool *value)
try {
    hipFileInit();
    (void)param;
    (void)value;

    throw std::runtime_error("Not Implemented");
}
catch (...) {
    return handle_exception();
}

hipFileError_t
hipFileGetParameterString(hipFileStringConfigParameter_t param, char *desc_str, int len)
try {
    hipFileInit();
    (void)param;
    (void)desc_str;
    (void)len;

    throw std::runtime_error("Not Implemented");
}
catch (...) {
    return handle_exception();
}

hipFileError_t
hipFileSetParameterSizeT(hipFileSizeTConfigParameter_t param, size_t value)
try {
    hipFileInit();
    (void)param;
    (void)value;

    throw std::runtime_error("Not Implemented");
}
catch (...) {
    return handle_exception();
}

hipFileError_t
hipFileSetParameterBool(hipFileBoolConfigParameter_t param, bool value)
try {
    hipFileInit();
    (void)param;
    (void)value;

    throw std::runtime_error("Not Implemented");
}
catch (...) {
    return handle_exception();
}

hipFileError_t
hipFileSetParameterString(hipFileStringConfigParameter_t param, const char *desc_str)
try {
    hipFileInit();
    (void)param;
    (void)desc_str;

    throw std::runtime_error("Not Implemented");
}
catch (...) {
    return handle_exception();
}

} // namespace
