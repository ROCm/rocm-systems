/*
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#ifndef __KFD_AIS_TEST__H__
#define __KFD_AIS_TEST__H__

#include <gtest/gtest.h>
#include "KFDBaseComponentTest.hpp"

class KFDAISTest :  public KFDBaseComponentTest {
 public:
    static const int BUFFER_SIZE    = 32 * 4096;        // 32 KB
    static const int FILE_SIZE      = 1024 * 1024;      // 1 MB
    static const int N_BUFFERS      = FILE_SIZE / BUFFER_SIZE;
    static constexpr const char *FILENAME = "/tmp/kfdais_test_file.bin";

    KFDAISTest(void) : m_pBufs(N_BUFFERS, NULL), m_pPatterns(N_BUFFERS, 0), m_fd(-1) {}

    ~KFDAISTest(void) {}
 protected:
	virtual void SetUp();
	virtual void TearDown();

    int createFileInNVME(void);
    void deleteFileInNVME(void) {
        if (m_fd >= 0)
            close(m_fd);
        unlink(FILENAME);
    }
    int fillFileWithPatterns(void);
    int checkBuffersWithPatterns(int gpuNode);
    int directReadWritefromFile(void *buf, int size,
                                HSAint64 fileOffset, HsaAisFlags rwFlags);
    int allocBuffers(HSAuint32 gpuNode);
    void freeBuffers();

    int m_fd;
    std::vector<void *> m_pBufs;  // Vector to hold allocated buffers
    std::vector<int> m_pPatterns; // Patterns to validate read / write

    int checkIfFileIsInNVME(int fd);
};

#endif // __KFD_AIS_TEST__H__
