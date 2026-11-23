/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#include <memory>

#include "roc_encoder.h"
#include "../../api/rocencode/rocencode.h"

/**
 * @brief The opaque `void* encoder` handed to the application.
 *
 * Owns the RocEncoder and carries a magic word so that a stale or foreign
 * pointer is rejected rather than dereferenced.
 */
class EncHandle {
public:
    static constexpr uint32_t kMagic = 0x524F4345u;  // 'ROCE'

    EncHandle(ROCENC_DEVICE_TYPE device_type, void* device)
        : roc_encoder_(std::make_shared<RocEncoder>(device_type, device)) {}

    ~EncHandle() { magic_ = 0; }

    bool IsValid() const { return magic_ == kMagic; }

    std::shared_ptr<RocEncoder> roc_encoder_;

private:
    uint32_t magic_ = kMagic;
};
