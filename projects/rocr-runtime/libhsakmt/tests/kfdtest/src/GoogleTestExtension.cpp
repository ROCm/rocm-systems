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

#include "GoogleTestExtension.hpp"
#include "OSWrapper.hpp"

bool Ok2Run(unsigned int testProfile) {
    bool testMatchProfile = true;
    if ((testProfile & g_TestRunProfile) == 0) {
        WARN() << "Test is skipped beacuse profile does not match current run mode" << std::endl;
        testMatchProfile = false;
    }

    return testMatchProfile;
}

// This predication is used when specific HW capabilities must exist for the test to succeed.
bool TestReqEnvCaps(unsigned int envCaps) {
    bool testMatchEnv = true;
    if ((envCaps & g_TestENVCaps) != envCaps) {
        WARN() << "Test is skipped due to HW capability issues" << std::endl;
        testMatchEnv = false;
    }

    return testMatchEnv;
}

// This predication is used when specific HW capabilities must be absent for the test to succeed.
// e.g Testing capabilities not supported by HW scheduling
bool TestReqNoEnvCaps(unsigned int envCaps) {
    bool testMatchEnv = true;
    if ((envCaps & g_TestENVCaps) != 0) {
        WARN() << "Test is skipped due to HW capability issues" << std::endl;
        testMatchEnv = false;
    }

    return testMatchEnv;
}

std::ostream& operator<< (KFDLog log, LOGTYPE level) {
    const char *heading;

    if (level == LOGTYPE_WARNING) {
        SetConsoleTextColor(TEXTCOLOR_YELLOW);
        heading = "[----------] ";
    } else {
        SetConsoleTextColor(TEXTCOLOR_GREEN);
        heading = "[          ] ";
    }

    std::clog << heading;
    SetConsoleTextColor(TEXTCOLOR_WHITE);

    return std::clog;
}

