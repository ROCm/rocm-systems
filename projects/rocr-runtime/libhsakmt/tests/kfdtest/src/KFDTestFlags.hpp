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

#ifndef __KFD_TEST_FLAGS__H__
#define __KFD_TEST_FLAGS__H__

extern unsigned int g_TestRunProfile;
extern unsigned int g_TestENVCaps;
extern unsigned int g_TestTimeOut;
extern int g_TestNodeId;
extern int g_TestDstNodeId;
extern bool g_IsChildProcess;
extern bool g_IsEmuMode;

// Each test should call TEST_START with the test custom profile and HW scheduling
enum TESTPROFILE{
    TESTPROFILE_DEV =          0x1,
    TESTPROFILE_PROMO =    0x2,
    // 0x4 - 0x8000 - unused flags
    // Can add any flag that will mark only part of the tests to run
    TESTPROFILE_RUNALL = 0xFFFF
};

enum ENVCAPS{
    ENVCAPS_NOADDEDCAPS    =  0x0,
    ENVCAPS_HWSCHEDULING   =  0x1,
    ENVCAPS_16BITPASID             =  0x2,
    ENVCAPS_32BITLINUX              =  0x4,
    ENVCAPS_64BITLINUX              =  0x8
    // 0x8 - 0x8000 - unused flags
    // Can add any flag that will mark specific hw limitation or capability
};

enum KfdFamilyId {
    FAMILY_UNKNOWN = 0,
    FAMILY_CI,    // Sea Islands: Hawaii (P), Maui (P), Bonaire (M)
    FAMILY_KV,    // Fusion Kaveri: Spectre, Spooky; Fusion Kabini: Kalindi
    FAMILY_VI,    // Volcanic Islands: Iceland (V), Tonga (M)
    FAMILY_CZ,    // Carrizo, Nolan, Amur
    FAMILY_AI,    // Arctic Islands
    FAMILY_RV,    // Raven
    FAMILY_AR,    // Arcturus
    FAMILY_AL,    // Aldebaran
    FAMILY_AV,    // Aqua Vanjaram
    FAMILY_NV,    // Navi10
    FAMILY_GFX11, // GFX11
    FAMILY_GFX12, // GFX12
};

#endif  //  __KFD_TEST_FLAGS__H__
