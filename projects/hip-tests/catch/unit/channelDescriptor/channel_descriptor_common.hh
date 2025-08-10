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

#include <hip_test_common.hh>

enum class ChannelDimension { OneDim, TwoDim, ThreeDim, FourDim };

template <typename T> class ChannelDescriptorTestShell {
 protected:
  int size;
  hipChannelFormatKind kind;
  ChannelDimension dimension;
  virtual void SetSizeAndKind() = 0;

 public:
  void Run() {
    hipChannelFormatDesc channel_desc{};
    SetSizeAndKind();
    hipChannelFormatDesc referent_channel_desc{0, 0, 0, 0, kind};
    switch (dimension) {
      case ChannelDimension::FourDim:
        referent_channel_desc.w = size;
      case ChannelDimension::ThreeDim:
        referent_channel_desc.z = size;
      case ChannelDimension::TwoDim:
        referent_channel_desc.y = size;
      default:
        referent_channel_desc.x = size;
    }
    channel_desc = hipCreateChannelDesc<T>();
    REQUIRE(channel_desc.x == referent_channel_desc.x);
    REQUIRE(channel_desc.y == referent_channel_desc.y);
    REQUIRE(channel_desc.z == referent_channel_desc.z);
    REQUIRE(channel_desc.w == referent_channel_desc.w);
    REQUIRE(channel_desc.f == referent_channel_desc.f);
  }

  ChannelDescriptorTestShell(const ChannelDimension dimension)
      : size(0), kind(hipChannelFormatKindNone), dimension(dimension) {}
  ChannelDescriptorTestShell(const ChannelDescriptorTestShell&) = delete;
  ChannelDescriptorTestShell(ChannelDescriptorTestShell&&) = delete;
};

template <typename T> class ChannelDescriptorTest1D : public ChannelDescriptorTestShell<T> {
 public:
  ChannelDescriptorTest1D() : ChannelDescriptorTestShell<T>(ChannelDimension::OneDim) {}

 protected:
  void SetSizeAndKind() {
    if (std::is_same_v<char, T>) {
      this->size = static_cast<int>(sizeof(char) * 8);
      this->kind = hipChannelFormatKindSigned;
    } else if (std::is_same_v<unsigned char, T> || std::is_same_v<uchar1, T>) {
      this->size = static_cast<int>(sizeof(unsigned char) * 8);
      this->kind = hipChannelFormatKindUnsigned;
    } else if (std::is_same_v<signed char, T> || std::is_same_v<char1, T>) {
      this->size = static_cast<int>(sizeof(signed char) * 8);
      this->kind = hipChannelFormatKindSigned;
    } else if (std::is_same_v<unsigned short, T> || std::is_same_v<ushort1, T>) {
      this->size = static_cast<int>(sizeof(unsigned short) * 8);
      this->kind = hipChannelFormatKindUnsigned;
    } else if (std::is_same_v<short, T> || std::is_same_v<signed short, T> ||
               std::is_same_v<short1, T>) {
      this->size = static_cast<int>(sizeof(signed short) * 8);
      this->kind = hipChannelFormatKindSigned;
    } else if (std::is_same_v<unsigned int, T> || std::is_same_v<uint1, T>) {
      this->size = static_cast<int>(sizeof(unsigned int) * 8);
      this->kind = hipChannelFormatKindUnsigned;
    } else if (std::is_same_v<int, T> || std::is_same_v<signed int, T> || std::is_same_v<int1, T>) {
      this->size = static_cast<int>(sizeof(signed int) * 8);
      this->kind = hipChannelFormatKindSigned;
    } else if (std::is_same_v<float, T> || std::is_same_v<float1, T>) {
      this->size = static_cast<int>(sizeof(float) * 8);
      this->kind = hipChannelFormatKindFloat;
    }
    #if !defined(__LP64__)
    else if (std::is_same_v<unsigned long, T> || std::is_same_v<ulong1, T>) {
      this->size = static_cast<int>(sizeof(unsigned long) * 8);
      this->kind = hipChannelFormatKindUnsigned;
    } else if (std::is_same_v<long, T> || std::is_same_v<signed long, T> ||
               std::is_same_v<long1, T>) {
      this->size = static_cast<int>(sizeof(signed long) * 8);
      this->kind = hipChannelFormatKindSigned;
    }
    #endif
  }
};

template <typename T> class ChannelDescriptorTest2D : public ChannelDescriptorTestShell<T> {
 public:
  ChannelDescriptorTest2D() : ChannelDescriptorTestShell<T>(ChannelDimension::TwoDim) {}

 protected:
  void SetSizeAndKind() {
    if (std::is_same_v<uchar2, T>) {
      this->size = static_cast<int>(sizeof(unsigned char) * 8);
      this->kind = hipChannelFormatKindUnsigned;
    } else if (std::is_same_v<char2, T>) {
      this->size = static_cast<int>(sizeof(signed char) * 8);
      this->kind = hipChannelFormatKindSigned;
    } else if (std::is_same_v<ushort2, T>) {
      this->size = static_cast<int>(sizeof(unsigned short) * 8);
      this->kind = hipChannelFormatKindUnsigned;
    } else if (std::is_same_v<short2, T>) {
      this->size = static_cast<int>(sizeof(signed short) * 8);
      this->kind = hipChannelFormatKindSigned;
    } else if (std::is_same_v<uint2, T>) {
      this->size = static_cast<int>(sizeof(unsigned int) * 8);
      this->kind = hipChannelFormatKindUnsigned;
    } else if (std::is_same_v<int2, T>) {
      this->size = static_cast<int>(sizeof(signed int) * 8);
      this->kind = hipChannelFormatKindSigned;
    } else if (std::is_same_v<float2, T>) {
      this->size = static_cast<int>(sizeof(float) * 8);
      this->kind = hipChannelFormatKindFloat;
    }
    #if !defined(__LP64__)
    else if (std::is_same_v<ulong2, T>) {
      this->size = static_cast<int>(sizeof(unsigned long) * 8);
      this->kind = hipChannelFormatKindUnsigned;
    } else if (std::is_same_v<long2, T>) {
      this->size = static_cast<int>(sizeof(signed long) * 8);
      this->kind = hipChannelFormatKindSigned;
    }
    #endif
  }
};

#ifndef __GNUC__
template <typename T> class ChannelDescriptorTest3D : public ChannelDescriptorTestShell<T> {
 public:
  ChannelDescriptorTest3D() : ChannelDescriptorTestShell<T>(ChannelDimension::ThreeDim) {}

 protected:
  void SetSizeAndKind() {
    if (std::is_same_v<uchar3, T>) {
      this->size = static_cast<int>(sizeof(unsigned char) * 8);
      this->kind = hipChannelFormatKindUnsigned;
    } else if (std::is_same_v<char3, T>) {
      this->size = static_cast<int>(sizeof(signed char) * 8);
      this->kind = hipChannelFormatKindSigned;
    } else if (std::is_same_v<ushort3, T>) {
      this->size = static_cast<int>(sizeof(unsigned short) * 8);
      this->kind = hipChannelFormatKindUnsigned;
    } else if (std::is_same_v<short3, T>) {
      this->size = static_cast<int>(sizeof(signed short) * 8);
      this->kind = hipChannelFormatKindSigned;
    } else if (std::is_same_v<uint3, T>) {
      this->size = static_cast<int>(sizeof(unsigned int) * 8);
      this->kind = hipChannelFormatKindUnsigned;
    } else if (std::is_same_v<int3, T>) {
      this->size = static_cast<int>(sizeof(signed int) * 8);
      this->kind = hipChannelFormatKindSigned;
    } else if (std::is_same_v<float3, T>) {
      this->size = static_cast<int>(sizeof(float) * 8);
      this->kind = hipChannelFormatKindFloat;
    }
    #if !defined(__LP64__)
    else if (std::is_same_v<ulong3, T>) {
      this->size = static_cast<int>(sizeof(unsigned long) * 8);
      this->kind = hipChannelFormatKindUnsigned;
    } else if (std::is_same_v<long3, T>) {
      this->size = static_cast<int>(sizeof(signed long) * 8);
      this->kind = hipChannelFormatKindSigned;
    }
    #endif
  }
};
#endif

template <typename T> class ChannelDescriptorTest4D : public ChannelDescriptorTestShell<T> {
 public:
  ChannelDescriptorTest4D() : ChannelDescriptorTestShell<T>(ChannelDimension::FourDim) {}

 protected:
  void SetSizeAndKind() {
    if (std::is_same_v<uchar4, T>) {
      this->size = static_cast<int>(sizeof(unsigned char) * 8);
      this->kind = hipChannelFormatKindUnsigned;
    } else if (std::is_same_v<char4, T>) {
      this->size = static_cast<int>(sizeof(signed char) * 8);
      this->kind = hipChannelFormatKindSigned;
    } else if (std::is_same_v<ushort4, T>) {
      this->size = static_cast<int>(sizeof(unsigned short) * 8);
      this->kind = hipChannelFormatKindUnsigned;
    } else if (std::is_same_v<short4, T>) {
      this->size = static_cast<int>(sizeof(signed short) * 8);
      this->kind = hipChannelFormatKindSigned;
    } else if (std::is_same_v<uint4, T>) {
      this->size = static_cast<int>(sizeof(unsigned int) * 8);
      this->kind = hipChannelFormatKindUnsigned;
    } else if (std::is_same_v<int4, T>) {
      this->size = static_cast<int>(sizeof(signed int) * 8);
      this->kind = hipChannelFormatKindSigned;
    } else if (std::is_same_v<float4, T>) {
      this->size = static_cast<int>(sizeof(float) * 8);
      this->kind = hipChannelFormatKindFloat;
    }
    #if !defined(__LP64__)
    else if (std::is_same_v<ulong4, T>) {
      this->size = static_cast<int>(sizeof(unsigned long) * 8);
      this->kind = hipChannelFormatKindUnsigned;
    } else if (std::is_same_v<long4, T>) {
      this->size = static_cast<int>(sizeof(signed long) * 8);
      this->kind = hipChannelFormatKindSigned;
    }
    #endif
  }
};

template <typename T> class ChannelDescriptorTestNone : public ChannelDescriptorTestShell<T> {
 public:
  ChannelDescriptorTestNone() : ChannelDescriptorTestShell<T>(ChannelDimension::OneDim) {}

 protected:
  void SetSizeAndKind() {}
};
