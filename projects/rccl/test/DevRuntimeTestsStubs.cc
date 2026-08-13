/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * No-op host stubs for the DevRuntimeTests micro-test binary.
 *
 * dev_runtime.cc is #included whole into DevRuntimeTests.cpp, which leaves
 * undefined references to everything the translation unit calls but does not
 * define. Provide inert host-side definitions here so the binary links without
 * librccl.so or a GPU. Populate incrementally from the linker's
 * "undefined reference" list.
 *************************************************************************/

// (stubs added as the link step reveals what is missing)
