/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "backend.h"
#include "buffer.h"
#include "file.h"

#include <cstddef>
#include <exception>
#include <gmock/gmock.h>

namespace hipFile {

struct MBackend : Backend {
    MOCK_METHOD(int, score,
                (const std::shared_ptr<IFile> &, const std::shared_ptr<IBuffer> &, size_t, hoff_t, hoff_t),
                (const, override));
    MOCK_METHOD(ssize_t, io,
                (hipFile::IoType type, std::shared_ptr<IFile>, std::shared_ptr<IBuffer>, size_t, hoff_t,
                 hoff_t),
                (override));
    MOCK_METHOD(ssize_t, _io_impl,
                (hipFile::IoType type, std::shared_ptr<IFile>, std::shared_ptr<IBuffer>, size_t, hoff_t,
                 hoff_t),
                (override));
};

struct MBackendWithFallback : BackendWithFallback {
    MOCK_METHOD(int, score,
                (const std::shared_ptr<IFile> &, const std::shared_ptr<IBuffer> &, size_t, hoff_t, hoff_t),
                (const, override));
    MOCK_METHOD(ssize_t, _io_impl,
                (IoType type, std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer, size_t size,
                 hoff_t file_offset, hoff_t buffer_offset),
                (override));
    MOCK_METHOD(bool, is_fallback_eligible, (std::exception_ptr e_ptr, ssize_t nbytes), (const, override));
};
}
