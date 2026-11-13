// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "kernel_name.h"

#include <gtest/gtest.h>

using namespace rocprofiler_compute_tool;

TEST(TestKernelName, MangledName_IsDemangled)
{
    EXPECT_EQ(format_kernel_name("_Z7vecCopyPdS_S_i"), "vecCopy(double*, double*, double*, int)");
}

TEST(TestKernelName, KernelDescriptorSuffix_IsDropped)
{
    // ".kd" does not demangle, so it has to go first.
    EXPECT_EQ(format_kernel_name("_Z7vecCopyPdS_S_i.kd"), "vecCopy(double*, double*, double*, int)");
}

TEST(TestKernelName, UnmangledName_IsLeftAlone)
{
    EXPECT_EQ(format_kernel_name("plain_kernel"), "plain_kernel");
}

TEST(TestKernelName, NullName_IsEmpty)
{
    EXPECT_EQ(format_kernel_name(nullptr), "");
}

TEST(TestKernelName, DemangledName_IsTruncatedToTheKernel)
{
    EXPECT_EQ(truncate_name("vecCopy(double*, double*, double*, int)"), "vecCopy");
    EXPECT_EQ(truncate_name("Foo<int, float>::foo(a[], int (int))"), "foo");
}

TEST(TestKernelName, KernelDescriptorSuffix_IsStrippedFromNonemptyName)
{
    EXPECT_EQ(strip_kernel_descriptor_suffix("_Z3fooi.kd"), "_Z3fooi");
    EXPECT_EQ(strip_kernel_descriptor_suffix("_Z3fooi"), "_Z3fooi");
    EXPECT_EQ(strip_kernel_descriptor_suffix(".kd"), ".kd");
    EXPECT_EQ(strip_kernel_descriptor_suffix(""), "");
    EXPECT_EQ(strip_kernel_descriptor_suffix("foo.kd.kd"), "foo.kd");
    EXPECT_EQ(format_kernel_name(".kd"), ".kd");
}
