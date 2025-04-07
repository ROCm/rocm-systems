/*
 * Copyright (C) 2017-2018 Advanced Micro Devices, Inc.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#include "KFDBaseComponentTest.hpp"
#include "BaseQueue.hpp"

#ifndef __KFD_MEMORY_TEST__H__
#define __KFD_MEMORY_TEST__H__

#define CMA_MEMORY_TEST_ARRAY_SIZE 4
#define CMA_TEST_COUNT 3

enum CMA_MEM_TYPE {
    CMA_MEM_TYPE_SYSTEM = 0,
    CMA_MEM_TYPE_USERPTR,
    CMA_MEM_TYPE_LOCAL_MEM,
};

enum CMA_TEST_TYPE {
    CMA_READ_TEST = 0,
    CMA_WRITE_TEST
};

enum CMA_TEST_STATUS {
    CMA_TEST_SUCCESS = 0,
    CMA_IPC_PIPE_ERROR = 1,
    CMA_CHECK_PATTERN_ERROR,
    CMA_TEST_ABORT,
    CMA_TEST_NOMEM,
    CMA_PARENT_FAIL,
    CMA_TEST_HSA_READ_FAIL,
    CMA_TEST_HSA_WRITE_FAIL
};

/* @struct testMemoryDescriptor
 * @brief Describes test buffers for Cross Memory Attach Test.
 */
struct testMemoryDescriptor {
    CMA_MEM_TYPE m_MemType;
    HSAuint64 m_MemSize;
    /* The buffer will be initialized with this pattern */
    HSAuint32 m_FillPattern;
    /* After CMA test, this pattern is expected in the first word */
    HSAuint32 m_CheckFirstWordPattern;
    /* After CMA test, this pattern is expected in the last word */
    HSAuint32 m_CheckLastWordPattern;

    testMemoryDescriptor(CMA_MEM_TYPE memType, HSAuint64 memSize,
        HSAuint32 fillPattern, HSAuint32 firstCheckPattern,
        HSAuint32 lastCheckPattern) :
        m_MemType(memType),
        m_MemSize(memSize),
        m_FillPattern(fillPattern),
        m_CheckFirstWordPattern(firstCheckPattern),
        m_CheckLastWordPattern(lastCheckPattern) {}
    ~testMemoryDescriptor(){}
};

/* @class KFDCMAArray
 * @brief Array of buffers that will be passed between the parent and child
 *        process for Cross memory read and write tests
 */
class KFDCMAArray {
    /* Used to store the actual buffer array */
    HsaMemoryBuffer* m_MemArray[CMA_MEMORY_TEST_ARRAY_SIZE];
    /* Used for passing to thunk CMA functions */
    HsaMemoryRange m_HsaMemoryRange[CMA_MEMORY_TEST_ARRAY_SIZE];
    /* Though previous arrays are fixed sizes only m_ValidCount
     * ones are valid
     */
    HSAuint64 m_ValidCount;
    QueueArray m_QueueArray;

 public:
    KFDCMAArray();
    ~KFDCMAArray() {
        Destroy();
    }

    CMA_TEST_STATUS Init(testMemoryDescriptor(*memDescriptor)[CMA_MEMORY_TEST_ARRAY_SIZE], int node);
    CMA_TEST_STATUS Destroy();

    HsaMemoryRange*  getMemoryRange() { return m_HsaMemoryRange; }
    HSAuint64 getValidRangeCount() { return m_ValidCount; }
    void FillPattern(testMemoryDescriptor(*memDescriptor)[CMA_MEMORY_TEST_ARRAY_SIZE]);
    CMA_TEST_STATUS checkPattern(testMemoryDescriptor(*memDescriptor)[CMA_MEMORY_TEST_ARRAY_SIZE]);
    CMA_TEST_STATUS sendCMAArray(int writePipe);
    CMA_TEST_STATUS recvCMAArray(int readPipe);
};


// @class KFDIPCTest
class KFDIPCTest :  public KFDBaseComponentTest {
 public:
    KFDIPCTest(void) : m_ChildPid(-1) {}
    ~KFDIPCTest(void);
 protected:
    virtual void SetUp();
    virtual void TearDown();

    /* For IPC testing */
    void BasicTestChildProcess(int defaultGPUNode, int *pipefd, HsaMemFlags mflags);
    void BasicTestParentProcess(int defaultGPUNode, pid_t childPid, int *pipefd, HsaMemFlags mflags);

    /* For CMA testing */
    CMA_TEST_STATUS CrossMemoryAttachChildProcess(int defaultGPUNode, int writePipe,
                                                  int readPipe, CMA_TEST_TYPE testType);
    CMA_TEST_STATUS CrossMemoryAttachParentProcess(int defaultGPUNode, pid_t cid,
                                                   int writePipe, int readPipe, CMA_TEST_TYPE testType);
 protected:
    pid_t m_ChildPid;
};

#endif  // __KFD_MEMORY_TEST__H__
