/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "backend.h"
#include "hipfile.h"

#include <hip/hip_runtime_api.h>
#include <map>
#include <memory>
#include <mutex>
#include <sys/types.h>
#include <vector>

namespace hipFile {
class IBuffer;
}
namespace hipFile {
class IFile;
}
namespace hipFile {
enum class IoType;
}

namespace hipFile {

struct BounceStream {
    BounceStream(int device_id, hipStream_t stream);
    ~BounceStream();
    BounceStream(const BounceStream &)            = delete;
    BounceStream &operator=(const BounceStream &) = delete;

    int         device_id;
    hipStream_t stream;
};

class BounceStreamPool {
public:
    ~BounceStreamPool();
    void                          clear();
    std::shared_ptr<BounceStream> getStream(int device_id);

private:
    struct StreamRing {
        std::vector<std::shared_ptr<BounceStream>> streams;
        size_t                                     current{0};
    };
    std::map<int, StreamRing> stream_rings;
    std::mutex                mtx;
};

struct Fallback : public Backend {
    using Backend::io;
    virtual ~Fallback() override = default;

    static constexpr size_t DefaultChunkSize = 16 * 1024 * 1024;

    int score(const std::shared_ptr<IFile> &file, const std::shared_ptr<IBuffer> &buffer, size_t size,
              hoff_t file_offset, hoff_t buffer_offset) const override;

    // Once we can import gtest.h and make test suites or test friends everything
    // below here should be made protected.
    ssize_t io(IoType type, std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer, size_t size,
               hoff_t file_offset, hoff_t buffer_offset, size_t chunk_size);

    void driver_state_teardown() override;

protected:
    ssize_t _io_impl(IoType type, std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer, size_t size,
                     hoff_t file_offset, hoff_t buffer_offset) override;
    ssize_t _io_impl(IoType type, std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer, size_t size,
                     hoff_t file_offset, hoff_t buffer_offset, size_t chunk_size);

private:
    BounceStreamPool bounce_stream_pool;
};

}
