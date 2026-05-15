/*
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "GoogleTestExtension.hpp"
#include "KFDASMTest.hpp"
#include "ShaderStore.hpp"
#include "Assemble.hpp"

void KFDASMTest::SetUp() {}
void KFDASMTest::TearDown() {}

static const std::vector<uint32_t> TargetList = {
    0x080001,
    0x080002,
    0x080003,
    0x080005,
    0x080100,
    0x090000,
    0x090002,
    0x090004,
    0x090006,
    0x090008,
    0x090009,
    0x09000a,
    0x09000c,
    0x090402,
    0x0a0100,
    0x0a0101,
    0x0a0102,
    0x0a0103,
    0x0a0300,
    0x0a0301,
    0x0a0302,
    0x0a0303,
    0x0a0304,
    0x0a0305,
    0x0a0306,
    0x0c0000,
};

TEST_F(KFDASMTest, AssembleShaders) {
    TEST_START(TESTPROFILE_RUNALL)

    for (auto &t : TargetList) {
        Assembler asmblr(t);

        LOG() << "Running ASM test for target " << asmblr.GetTargetAsic() << std::endl;

        for (auto &s : ShaderList) {
            EXPECT_SUCCESS(asmblr.RunAssemble(s));
        }
    }

    TEST_END
}
