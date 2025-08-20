/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "channel_descriptor_common.hh"

/**
 * @addtogroup hipCreateChannelDesc hipCreateChannelDesc
 * @{
 * @ingroup DeviceLanguageTest
 * `hipCreateChannelDesc<T>()` -
 * Creates a dedicated channel descriptor based on passed built-in or vector type T.
 */

/**
 * Test Description
 * ------------------------
 *  - Validates that 1D channel descriptor is created as expected.
 *  - Compares channel descriptor with the manually created one.
 *  - Takes into consideration following 1D built-in and vector types:
 *    -# char (signed and unsigned)
 *    -# short (signed and unsigned)
 *    -# int (signed and unsigned)
 *    -# float
 *    -# long (signed and unsigned)
 * Test source
 * ------------------------
 *  - unit/channelDescriptor/channel_descriptor.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_ChannelDescriptor_Positive_Basic_1D", "", char, unsigned char, uchar1,
                   signed char, char1, unsigned short, ushort1, short, signed short, short1, int,
                   unsigned int, uint1, signed int, int1, float, float1, long, unsigned long,
                   ulong1, signed long, long1) {
  ChannelDescriptorTest1D<TestType> channel_desc_test;
  channel_desc_test.Run();
}

/**
 * Test Description
 * ------------------------
 *  - Validates that 2D channel descriptor is created as expected.
 *  - Compares channel descriptor with the manually created one.
 *  - Takes into consideration following 2D built-in and vector types:
 *    -# char (signed and unsigned)
 *    -# short (signed and unsigned)
 *    -# int (signed and unsigned)
 *    -# float
 *    -# long (signed and unsigned)
 * Test source
 * ------------------------
 *  - unit/channelDescriptor/channel_descriptor.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_ChannelDescriptor_Positive_Basic_2D", "", uchar2, char2, ushort2, short2,
                   uint2, int2, float2, ulong2, long2) {
  ChannelDescriptorTest2D<TestType> channel_desc_test;
  channel_desc_test.Run();
}

#ifndef __GNUC__
/**
 * Test Description
 * ------------------------
 *  - Validates that 3D channel descriptor is created as expected.
 *  - Compares channel descriptor with the manually created one.
 *  - Takes into consideration following 3D built-in and vector types:
 *    -# char (signed and unsigned)
 *    -# short (signed and unsigned)
 *    -# int (signed and unsigned)
 *    -# float
 *    -# long (signed and unsigned)
 * Test source
 * ------------------------
 *  - unit/channelDescriptor/channel_descriptor.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 *  - Available for non-GNUC compilers.
 */
TEMPLATE_TEST_CASE("Unit_ChannelDescriptor_Positive_Basic_3D", "", uchar3, char3, ushort3, short3,
                   uint3, int3, float3, ulong3, long3) {
  ChannelDescriptorTest3D<TestType> channel_desc_test;
  channel_desc_test.Run();
}
#endif

/**
 * Test Description
 * ------------------------
 *  - Validates that 4D channel descriptor is created as expected.
 *  - Compares channel descriptor with the manually created one.
 *  - Takes into consideration following 4D built-in and vector types:
 *    -# char (signed and unsigned)
 *    -# short (signed and unsigned)
 *    -# int (signed and unsigned)
 *    -# float
 *    -# long (signed and unsigned)
 * Test source
 * ------------------------
 *  - unit/channelDescriptor/channel_descriptor.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_ChannelDescriptor_Positive_Basic_4D", "", uchar4, char4, ushort4, short4,
                   uint4, int4, float4, ulong4, long4) {
  ChannelDescriptorTest4D<TestType> channel_desc_test;
  channel_desc_test.Run();
}

/**
 * Test Description
 * ------------------------
 *  - Validates that an empty channel descriptor is created as expected.
 *  - Compares channel descriptor with the manually created one.
 *  - Takes into consideration all dimensions of the following built-in and vector types:
 *    -# long long (signed and unsigned)
 *    -# double
 * Test source
 * ------------------------
 *  - unit/channelDescriptor/channel_descriptor.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_ChannelDescriptor_Positive_FormatNone", "", long long, signed long long,
                   unsigned long long, longlong1, longlong2, longlong3, longlong4, ulonglong1,
                   ulonglong2, ulonglong3, ulonglong4, double1, double2, double3, double4) {
  ChannelDescriptorTestNone<TestType> channel_desc_test;
  channel_desc_test.Run();
}

#if HT_AMD
/**
 * Test Description
 * ------------------------
 *  - Validates that the channel descriptor is created as expected.
 *  - Compares channel descriptor with the manually created one.
 *  - Takes into consideration 16-bit floating-point type.
 *    -# Creates 1D channel descriptor.
 *    -# Creates 2D channel descriptor.
 * Test source
 * ------------------------
 *  - unit/channelDescriptor/channel_descriptor.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_ChannelDescriptor_Positive_16BitFloatingPoint") {
  int size = static_cast<int>(sizeof(unsigned short) * 8);
  hipChannelFormatKind kind = hipChannelFormatKindFloat;
  hipChannelFormatDesc channel_desc{};
  hipChannelFormatDesc referent_channel_desc{};

  SECTION("hipCreateChannelDescHalf") {
    referent_channel_desc = {size, 0, 0, 0, kind};
    channel_desc = hipCreateChannelDescHalf();
  }
  SECTION("hipCreateChannelDescHalf1") {
    referent_channel_desc = {size, 0, 0, 0, kind};
    channel_desc = hipCreateChannelDescHalf1();
  }
  SECTION("hipCreateChannelDescHalf2") {
    referent_channel_desc = {size, size, 0, 0, kind};
    channel_desc = hipCreateChannelDescHalf2();
  }

  REQUIRE(channel_desc.x == referent_channel_desc.x);
  REQUIRE(channel_desc.y == referent_channel_desc.y);
  REQUIRE(channel_desc.z == referent_channel_desc.z);
  REQUIRE(channel_desc.w == referent_channel_desc.w);
  REQUIRE(channel_desc.f == referent_channel_desc.f);
}
#endif

/**
 * End doxygen group DeviceLanguageTest.
 * @}
 */
