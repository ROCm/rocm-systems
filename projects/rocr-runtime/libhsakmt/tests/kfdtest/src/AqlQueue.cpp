/*
 * Copyright (C) 2014-2018 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#include "AqlQueue.hpp"
#include "GoogleTestExtension.hpp"


AqlQueue::AqlQueue(void) {
}


AqlQueue::~AqlQueue(void) {
}

unsigned int AqlQueue::Wptr() {
    return *m_Resources.Queue_write_ptr;
}

unsigned int AqlQueue::Rptr() {
    return *m_Resources.Queue_read_ptr;
}

unsigned int AqlQueue::RptrWhenConsumed() {
    return Wptr();
}

void AqlQueue::SubmitPacket() {
    // m_pending Wptr is in dwords
    *m_Resources.Queue_write_ptr = m_pendingWptr;
    *(m_Resources.Queue_DoorBell) = Wptr();
}

