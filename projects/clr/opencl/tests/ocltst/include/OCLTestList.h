/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCLMODULE_H_
#define _OCLMODULE_H_

#ifdef _WIN32
#define OCLLCONV __cdecl
#endif
#ifdef __linux__
#define OCLLCONV
#endif

class OCLTest;

//
//  exported function pointer typedefs
//
typedef unsigned int(OCLLCONV* TestCountFuncPtr)(void);
typedef const char*(OCLLCONV* TestNameFuncPtr)(unsigned int);
typedef OCLTest*(OCLLCONV* CreateTestFuncPtr)(unsigned int);
typedef void(OCLLCONV* DestroyTestFuncPtr)(OCLTest*);
typedef unsigned int(OCLLCONV* TestVersionFuncPtr)(void);
typedef const char*(OCLLCONV* TestLibNameFuncPtr)(void);

#endif  // _OCLMODULE_H_
