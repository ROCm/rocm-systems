/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "backend/fallback.h"
#include "backend/fastpath.h"
#include "batch/batch.h"
#include "buffer.h"
#include "configuration.h"
#include "context.h"
#include "file.h"
#include "state.h"
#include "stream.h"

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <utility>

using namespace std;

namespace hipFile {
struct Backend;
}

namespace hipFile {

DriverState::DriverState() : ref_count{0}
{
    this->file_map   = std::make_unique<FileMap>();
    this->batch_map  = std::make_unique<BatchContextMap>();
    this->buffer_map = std::make_unique<BufferMap>();
    this->stream_map = std::make_unique<StreamMap>();
}

DriverState::~DriverState()
{
}

//
// Batch interface
//

hipFileBatchHandle_t
DriverState::createBatchContext(unsigned capacity)
{
    return batch_map->createContext(capacity);
}

void
DriverState::destroyBatchContext(hipFileBatchHandle_t handle)
{
    batch_map->destroyContext(handle);
}

std::shared_ptr<IBatchContext>
DriverState::getBatchContext(hipFileBatchHandle_t handle)
{
    return batch_map->get(handle);
}

//
// Buffer interface
//

void
DriverState::registerBuffer(const void *buf, size_t length, int flags)
{
    unique_lock<shared_mutex> ulock{state_mutex};

    // For NVIDIA cuFile compatibility, implicitly "initialize"
    // the driver if it's currently uninitialized
    if (ref_count == 0) {
        ref_count++;
    }

    buffer_map->registerBuffer(buf, length, flags);
}

void
DriverState::deregisterBuffer(const void *buf)
{
    unique_lock<shared_mutex> ulock{state_mutex};

    if (ref_count == 0) {
        throw DriverNotInitialized();
    }

    buffer_map->deregisterBuffer(buf);
}

shared_ptr<IBuffer>
DriverState::getRegisteredBuffer(const void *buf)
{
    shared_lock<shared_mutex> slock{state_mutex};

    if (ref_count == 0) {
        throw DriverNotInitialized();
    }

    return buffer_map->getRegisteredBuffer(buf);
}

shared_ptr<IBuffer>
DriverState::getBuffer(const void *buf)
{
    // NOTE: This mutex only protects the map, so we'll
    //       also need to protect the data
    shared_lock<shared_mutex> slock{state_mutex};

    return buffer_map->getBuffer(buf);
}

//
// File interface
//

hipFileHandle_t
DriverState::registerFile(UnregisteredFile &&uf)
{
    unique_lock<shared_mutex> ulock{state_mutex};

    // For NVIDIA cuFile compatibility, implicitly "initialize"
    // the driver if it's currently uninitialized
    if (ref_count == 0) {
        ref_count++;
    }

    return file_map->registerFile(std::move(uf));
}

void
DriverState::deregisterFile(hipFileHandle_t fh)
{
    unique_lock<shared_mutex> ulock{state_mutex};

    if (ref_count == 0) {
        throw DriverNotInitialized();
    }

    file_map->deregisterFile(fh);
}

shared_ptr<IFile>
DriverState::getFile(hipFileHandle_t fh)
{
    // NOTE: This mutex only protects the map, so we'll
    //       also need to protect the data
    shared_lock<shared_mutex> slock{state_mutex};

    if (ref_count == 0) {
        throw DriverNotInitialized();
    }

    return file_map->getFile(fh);
}

//
// Stream interface
//

void
DriverState::registerStream(const hipStream_t hip_stream, uint32_t flags)
{
    unique_lock<shared_mutex> ulock{state_mutex};
    stream_map->registerStream(hip_stream, flags);
}

void
DriverState::deregisterStream(const hipStream_t hip_stream)
{
    shared_ptr<IStream> stream;
    {
        unique_lock<shared_mutex> ulock{state_mutex};
        stream = stream_map->deregisterStream(hip_stream);
    }
    // Drain outside state_mutex: it waits on in-flight IO, which must not hold the global lock.
    if (stream) {
        stream->drainAndRelease();
    }
}

shared_ptr<IStream>
DriverState::getStream(hipStream_t hip_stream)
{
    shared_lock<shared_mutex> slock{state_mutex};
    return stream_map->getStream(hip_stream);
}

//
// Buffer and file calls
//
// These are for reducing the number of lock calls and are implemented
// as needed for the hipFile code
//

file_buffer_pair
DriverState::getFileAndBuffer(hipFileHandle_t fh, const void *buf)
{
    // NOTE: This mutex only protects the map, so we'll
    //       also need to protect the data
    shared_lock<shared_mutex> slock{state_mutex};

    if (ref_count == 0) {
        throw DriverNotInitialized();
    }

    return {file_map->getFile(fh), buffer_map->getBuffer(buf)};
}

//
// Buffer and file and stream calls
//

file_buffer_stream_tuple
DriverState::getFileBufferAndStream(hipFileHandle_t fh, const void *buf, hipStream_t hipStream)
{
    shared_ptr<IFile>   file;
    shared_ptr<IBuffer> buffer;
    shared_ptr<IStream> stream;
    {
        shared_lock<shared_mutex> slock{state_mutex};

        if (ref_count == 0) {
            throw DriverNotInitialized();
        }

        file   = file_map->getFile(fh);
        buffer = buffer_map->getBuffer(buf);
        stream =
            isDefaultStream(hipStream) ? stream_map->getStream(hipStream) : stream_map->findStream(hipStream);
    }

    if (!stream) {
        // Implicit registration on first async use, as cuFile does. Built outside state_mutex: it makes HIP
        // calls. A candidate that loses the race is dropped after the lock is released.
        auto candidate = StreamMap::makeImplicitStream(hipStream);

        unique_lock<shared_mutex> ulock{state_mutex};
        if (ref_count == 0) {
            throw DriverNotInitialized();
        }
        stream = stream_map->adoptImplicitStream(hipStream, candidate);
    }

    return {std::move(file), std::move(buffer), std::move(stream)};
}

//
// Reference counts
//

void
DriverState::incrRefCount()
{
    unique_lock<shared_mutex> ulock{state_mutex};

    ref_count++;
}

void
DriverState::decrRefCount()
{
    unique_lock<shared_mutex> ulock{state_mutex};

    // Decrement the reference count
    if (ref_count > 0) {
        ref_count--;
    }

    // Clear the maps if the reference count just hit zero
    if (ref_count == 0) {
        batch_map->clear();
        buffer_map->clear();
        file_map->clear();
        for (auto &backend : getBackends()) {
            try {
                backend->driver_state_teardown();
            }
            catch (...) {
                // Ignore exceptions during teardown
            }
        }
    }

    // Negative reference count, should never get here
    if (ref_count < 0) {
        throw std::runtime_error("negative state reference count");
    }
}

int64_t
DriverState::getRefCount() const
{
    shared_lock<shared_mutex> slock{state_mutex};

    return ref_count;
}

void
DriverState::ensureInitialized()
{
    unique_lock<shared_mutex> ulock{state_mutex};

    if (ref_count == 0) {
        ref_count++;
    }
}

std::vector<std::shared_ptr<Backend>>
DriverState::getBackends() const
{
    static bool once = [&]() {
        std::shared_ptr<Fallback> fallback_backend = std::make_shared<Fallback>();
        std::shared_ptr<Fastpath> fastpath_backend = std::make_shared<Fastpath>();
        fastpath_backend->register_fallback_backend(fallback_backend);
        backends.push_back(fallback_backend);
        backends.push_back(fastpath_backend);
        return true;
    }();
    (void)once;

    return backends;
}
}
