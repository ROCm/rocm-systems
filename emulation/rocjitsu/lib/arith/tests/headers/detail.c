// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Standalone C/C++ compilation of a selected detail header.

// Absolute includes and no include path check self-containment for consumers.
#include "@header_path@"

// Provide a complete translation unit for standalone header compilation.
int main(void) { return 0; }
