/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "backend.h"
#include "buffer.h"
#include "file.h"
#include "hipfile.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <sys/types.h>

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

struct Fastpath : public BackendWithFallback {
    Fastpath();
    virtual ~Fastpath() override = default;

    int score(const std::shared_ptr<IFile> &file, const std::shared_ptr<IBuffer> &buffer, size_t size,
              hoff_t file_offset, hoff_t buffer_offset) const override;

protected:
    ssize_t _io_impl(IoType type, std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer, size_t size,
                     hoff_t file_offset, hoff_t buffer_offset) override;
    bool    is_fallback_eligible(std::exception_ptr e_ptr, ssize_t nbytes) const override;

private:
    // Unique id assigned at construction. _io_impl compares this against a
    // thread_local "last inited for" id to decide whether a thread needs to
    // call hipInit(). A new id on each fresh Fastpath() object forces every
    // thread to re-init, giving per-test isolation without losing the
    // per-thread production invariant.
    const std::uint64_t instance_id_;
};

}
