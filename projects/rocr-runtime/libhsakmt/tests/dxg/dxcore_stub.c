/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/* A libdxcore.so that loads but exports none of the D3DKMT entry points, so
 * DxcoreLoader::Initialize() fails after hsaKmtOpenKFD() has opened /dev/dxg.
 * The library needs at least one symbol of its own to be a valid shared object.
 */
int dxg_lifecycle_test_dxcore_stub(void) { return 0; }
