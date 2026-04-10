/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

// When testing it is sometimes inconvenient if a variable is statically initialized.
// Declaring a variable as STATIC will result in the variable being static only
// when tests are not being built.
#ifdef AIS_TESTING
#define HIPFILE_STATIC
#else
#define HIPFILE_STATIC static
#endif
