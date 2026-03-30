/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "OCLTestListImp.h"

//
// Includes for tests
//
#include "OCLGLBuffer.h"
#include "OCLGLBufferMultipleQueues.h"
#include "OCLGLDepthBuffer.h"
#include "OCLGLDepthTex.h"
#include "OCLGLFenceSync.h"
#include "OCLGLMsaaTexture.h"
#include "OCLGLMultiContext.h"
#include "OCLGLTexture.h"
#include "OCLGLPerfSepia.h"

//
//  Helper macro for adding tests
//
template <typename T> static void* dictionary_CreateTestFunc(void) { return new T(); }

#define TEST(name) {#name, &dictionary_CreateTestFunc<name>}

TestEntry TestList[] = {
    TEST(OCLGLBuffer),    TEST(OCLGLBufferMultipleQueues),
    TEST(OCLGLTexture),   TEST(OCLGLMultiContext),
    TEST(OCLGLFenceSync), TEST(OCLGLDepthTex),
    TEST(OCLGLPerfSepia),
};

unsigned int TestListCount = sizeof(TestList) / sizeof(TestList[0]);
unsigned int TestLibVersion = 0;
const char* TestLibName = "oclgl";
