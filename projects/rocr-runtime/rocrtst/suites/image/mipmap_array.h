/*

Copyright © 2025 Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/
#pragma once

#include "common/base_rocr_utils.h"
#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_image.h"
#include "hsa/hsa_amd_mipmap.h"
#include "hsa/hsa_ext_image.h"

using MipmappedArray_t = hsa_ext_image_t;

using hsa_ext_image_extension_table_t = hsa_ext_images_1_pfn_t;
#include <vector>
#include <typeinfo>
#include <type_traits>
#include <string>
#include <limits>
#include <cmath>
#include <type_traits>

struct char1 {
    char x;
    char1() = default;
    char1(char x_) : x(x_) {}
};
struct uchar1 {
    unsigned char x;
    uchar1() = default;
    uchar1(unsigned char x_) : x(x_) {}
};
struct short1 {
    short x;
    short1() = default;
    short1(short x_) : x(x_) {}
};
struct ushort1 {
    unsigned short x;
    ushort1() = default;
    ushort1(unsigned short x_) : x(x_) {}
};
struct int1 {
    int x;
    int1() = default;
    int1(int x_) : x(x_) {}
};
struct uint1 {
    unsigned int x;
    uint1() = default;
    uint1(unsigned int x_) : x(x_) {}
};
struct float1 {
    float x;
    float1() = default;
    float1(float x_) : x(x_) {}
};

struct char2 {
    char x, y;
    char2() = default;
    char2(char x_, char y_) : x(x_), y(y_) {}
};
struct uchar2 {
    unsigned char x, y;
    uchar2() = default;
    uchar2(unsigned char x_, unsigned char y_) : x(x_), y(y_) {}
};
struct short2 {
    short x, y;
    short2() = default;
    short2(short x_, short y_) : x(x_), y(y_) {}
};
struct ushort2 {
    unsigned short x, y;
    ushort2() = default;
    ushort2(unsigned short x_, unsigned short y_) : x(x_), y(y_) {}
};
struct int2 {
    int x, y;
    int2() = default;
    int2(int x_, int y_) : x(x_), y(y_) {}
};
struct uint2 {
    unsigned int x, y;
    uint2() = default;
    uint2(unsigned int x_, unsigned int y_) : x(x_), y(y_) {}
};
struct float2 {
    float x, y;
    float2() = default;
    float2(float x_, float y_) : x(x_), y(y_) {}
};

struct char4 {
    char x, y, z, w;
    char4() = default;
    char4(char x_, char y_, char z_, char w_) : x(x_), y(y_), z(z_), w(w_) {}
};
struct uchar4 {
    unsigned char x, y, z, w;
    uchar4() = default;
    uchar4(unsigned char x_, unsigned char y_, unsigned char z_, unsigned char w_) : x(x_), y(y_), z(z_), w(w_) {}
};
struct short4 {
    short x, y, z, w;
    short4() = default;
    short4(short x_, short y_, short z_, short w_) : x(x_), y(y_), z(z_), w(w_) {}
};
struct ushort4 {
    unsigned short x, y, z, w;
    ushort4() = default;
    ushort4(unsigned short x_, unsigned short y_, unsigned short z_, unsigned short w_) : x(x_), y(y_), z(z_), w(w_) {}
};
struct int4 {
    int x, y, z, w;
    int4() = default;
    int4(int x_, int y_, int z_, int w_) : x(x_), y(y_), z(z_), w(w_) {}
};
struct uint4 {
    unsigned int x, y, z, w;
    uint4() = default;
    uint4(unsigned int x_, unsigned int y_, unsigned int z_, unsigned int w_) : x(x_), y(y_), z(z_), w(w_) {}
};
struct float4 {
    float x, y, z, w;
    float4() = default;
    float4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
};

enum RocRTextureReadMode {
    ROCR_READ_MODE_ELEMENT_TYPE = 0,
    ROCR_READ_MODE_NORMALIZED_FLOAT = 1
};

enum RocRTextureFilterMode {
    ROCR_FILTER_MODE_POINT = 0,
    ROCR_FILTER_MODE_LINEAR = 1
};

enum RocRTextureAddressMode {
    ROCR_ADDRESS_MODE_CLAMP = 0,
    ROCR_ADDRESS_MODE_BORDER = 1,
    ROCR_ADDRESS_MODE_WRAP = 2
};

namespace rocrtst {

template <typename T>
struct MipmapLevelData {
    T* data;           // level array data
    uint32_t width;    // level array width
    uint32_t height;   // level array height
    uint32_t depth;    // level array depth
};

template <typename T>
constexpr bool is_scalar_type() {
    return std::is_scalar<T>::value;
}

template <typename T>
constexpr int type_rank() {
    if constexpr (std::is_scalar<T>::value) return 0;
    else if constexpr (std::is_same_v<T, char1> || std::is_same_v<T, uchar1> || 
                       std::is_same_v<T, short1> || std::is_same_v<T, ushort1> ||
                       std::is_same_v<T, int1> || std::is_same_v<T, uint1> || std::is_same_v<T, float1>) return 1;
    else if constexpr (std::is_same_v<T, char2> || std::is_same_v<T, uchar2> || 
                       std::is_same_v<T, short2> || std::is_same_v<T, ushort2> ||
                       std::is_same_v<T, int2> || std::is_same_v<T, uint2> || std::is_same_v<T, float2>) return 2;
    else if constexpr (std::is_same_v<T, char4> || std::is_same_v<T, uchar4> || 
                       std::is_same_v<T, short4> || std::is_same_v<T, ushort4> ||
                       std::is_same_v<T, int4> || std::is_same_v<T, uint4> || std::is_same_v<T, float4>) return 4;
    else return 0;
}

template <typename T, typename F>
inline T getTypeFromNormalizedFloat(const F &f) {
    T t;
    if constexpr (std::is_scalar<T>::value) {
        t = static_cast<T>(f * std::numeric_limits<T>::max());
    } else {
        if constexpr (type_rank<T>() > 0)
            t.x = static_cast<decltype(T::x)>(f * std::numeric_limits<decltype(T::x)>::max());
        if constexpr (type_rank<T>() > 1)
            t.y = static_cast<decltype(T::y)>(f * std::numeric_limits<decltype(T::y)>::max());
        if constexpr (type_rank<T>() > 2)
            t.z = static_cast<decltype(T::z)>(f * std::numeric_limits<decltype(T::z)>::max());
        if constexpr (type_rank<T>() > 3)
            t.w = static_cast<decltype(T::w)>(f * std::numeric_limits<decltype(T::w)>::max());
    }
    return t;
}

template <class T>
inline auto getNormalizedFloatType(const T &t) {
    if constexpr (std::is_scalar<T>::value) {
        return static_cast<float>(t) / std::numeric_limits<T>::max();
    } else {
        if constexpr (type_rank<T>() == 1) {
            float1 f{static_cast<float>(t.x) / std::numeric_limits<decltype(T::x)>::max()};
            return f;
        }
        if constexpr (type_rank<T>() == 2) {
            float2 f{static_cast<float>(t.x) / std::numeric_limits<decltype(T::x)>::max(),
                     static_cast<float>(t.y) / std::numeric_limits<decltype(T::y)>::max()};
            return f;
        }
        if constexpr (type_rank<T>() == 4) {
            float4 f{static_cast<float>(t.x) / std::numeric_limits<decltype(T::x)>::max(),
                     static_cast<float>(t.y) / std::numeric_limits<decltype(T::y)>::max(),
                     static_cast<float>(t.z) / std::numeric_limits<decltype(T::z)>::max(),
                     static_cast<float>(t.w) / std::numeric_limits<decltype(T::w)>::max()};
            return f;
        }
    }
}

template <typename T>
inline bool constexpr isFloat() {
    if constexpr (std::is_scalar<T>::value) {
        return std::is_floating_point<T>::value;
    } else {
        return std::is_floating_point<decltype(T::x)>::value;
    }
    return false;
}

template<typename T>
inline T getRandom() {
    float r = static_cast<float>(rand()) / RAND_MAX;
    if constexpr (std::is_floating_point<T>::value) {
        // Restrict any float within (-1000, 1000) to prevent calculation issues
        return static_cast<T>(r * 1000.0);
    } else {
        return static_cast<T>(std::numeric_limits<T>::max() * r);
    }
}

template<typename T>
inline void initVal(T& val) {
    if constexpr (std::is_scalar<T>::value) {
        val = getRandom<T>();
    } else {
        if constexpr (type_rank<T>() > 0) val.x = getRandom<decltype(T::x)>();
        if constexpr (type_rank<T>() > 1) val.y = getRandom<decltype(T::y)>();
        if constexpr (type_rank<T>() > 2) val.z = getRandom<decltype(T::z)>();
        if constexpr (type_rank<T>() > 3) val.w = getRandom<decltype(T::w)>();
    }
}

// Helper to check if a type has x member
template<typename T, typename = void>
struct has_x_member : std::false_type {};

template<typename T>
struct has_x_member<T, std::void_t<decltype(T::x)>> : std::true_type {};

template<typename T>
inline std::string getString(const T& t) {
    if constexpr (std::is_scalar<T>::value) {
        if constexpr (std::is_same<T, char>::value || std::is_same<T, unsigned char>::value) {
            return std::to_string(static_cast<int>(t));
        } else {
            return std::to_string(t);
        }
    }
    else if constexpr (has_x_member<T>::value) {
        if constexpr (std::is_same<decltype(T::x), char>::value ||
                      std::is_same<decltype(T::x), unsigned char>::value) {
            if constexpr (type_rank<T>() == 1) {
                return "(" + std::to_string(static_cast<int>(t.x)) + ")";
            } else if constexpr (type_rank<T>() == 2) {
                return "(" + std::to_string(static_cast<int>(t.x)) + ", " +
                       std::to_string(static_cast<int>(t.y)) + ")";
            } else if constexpr (type_rank<T>() == 4) {
                return "(" + std::to_string(static_cast<int>(t.x)) + ", " +
                       std::to_string(static_cast<int>(t.y)) + ", " +
                       std::to_string(static_cast<int>(t.z)) + ", " +
                       std::to_string(static_cast<int>(t.w)) + ")";
            }
        } else {
            if constexpr (type_rank<T>() == 1) {
                return "(" + std::to_string(t.x) + ")";
            } else if constexpr (type_rank<T>() == 2) {
                return "(" + std::to_string(t.x) + ", " + std::to_string(t.y) + ")";
            } else if constexpr (type_rank<T>() == 4) {
                return "(" + std::to_string(t.x) + ", " + std::to_string(t.y) + ", " +
                       std::to_string(t.z) + ", " + std::to_string(t.w) + ")";
            }
        }
    }
    else {
        return "complex_type";
    }
    return "unknown";
}

class MipmapArrayTest : public BaseRocR {
public:
    MipmapArrayTest(void);
    virtual ~MipmapArrayTest(void);

    // Test framework methods
    virtual void SetUp(void);
    virtual void Run(void);
    virtual void DisplayTestInfo(void);
    virtual void DisplayResults(void) const;
    virtual void Close(void);

    void MipmapCreate1DArrayTest(void);
    void MipmapDestroy1DArrayTest(void);
    void MipmapGetLevel1DArrayTest(void);

    void MipmapCreate2DArrayTest(void);
    void MipmapDestroy2DArrayTest(void);
    void MipmapGetLevel2DArrayTest(void);

    void MipmapCreate3DArrayTest(void);
    void MipmapDestroy3DArrayTest(void);
    void MipmapGetLevel3DArrayTest(void);

private:
    hsa_ext_image_t test_image_;
    uint32_t num_mipmap_levels_;
    MipmappedArray_t mipmapped_array_;
    hsa_ext_image_extension_table_t image_extension_table_;
    bool image_extension_supported_;
    hsa_ext_image_descriptor_t mipmap_desc_;
    hsa_ext_image_format_t image_format_;
};

class Mipmap1DArrayTest : public BaseRocR {
public:
    Mipmap1DArrayTest();
    virtual ~Mipmap1DArrayTest();

    void SetUp();
    void TearDown();
    void Run();
    void DisplayTestInfo();
    void DisplayResults() const;
    void Close();

    template<typename T, RocRTextureReadMode readMode = ROCR_READ_MODE_ELEMENT_TYPE,
             RocRTextureFilterMode filterMode = ROCR_FILTER_MODE_POINT,
             RocRTextureAddressMode addressMode = ROCR_ADDRESS_MODE_CLAMP>
    void testMipmapTextureObj1D(size_t width, float offsetX = 0.0f);

    template<typename T>
    void populateMipmaps1D(MipmappedArray_t mipmapArray, uint32_t width,
                          std::vector<MipmapLevelData<T>>& mipmapData);

    template<typename T, RocRTextureFilterMode filterMode = ROCR_FILTER_MODE_POINT,
             RocRTextureAddressMode addressMode = ROCR_ADDRESS_MODE_CLAMP>
    void verifyMipmapLevel1D(hsa_agent_t agent, MipmappedArray_t texMipmap, 
                            T* data, size_t width, float level, float offsetX);

    void TestElementTypeReadMode1D();
    void TestNormalizedFloatReadMode1D();
    void TestLinearFiltering1D();
    void TestAddressModes1D();
    void TestVariousDimensions1D();
    void TestErrorConditions1D();
    void TestMemoryIntegrity1D();

    template<typename T>
    hsa_ext_image_format_t GetImageFormat();

    template<typename T>
    hsa_ext_image_channel_type_t GetChannelType();

    template<typename T>
    size_t GetTypeSize();

    template<typename T>
    std::string GetTypeName();
    uint32_t CalculateMipmapLevels(uint32_t width);
    void LogTestProgress(const std::string& test_name, const std::string& details = "");

    template<typename T>
    bool ValidateMipmapLevel(hsa_agent_t agent, MipmappedArray_t mipmap_array, uint32_t level, uint32_t expected_width);

    template<typename T>
    bool CreateAndVerifyMipmap1DArray(uint32_t width);

private:
    hsa_ext_image_extension_table_t image_extension_table_;
    bool image_extension_supported_;
    std::vector<hsa_agent_t> gpu_agents_;

    // Test statistics
    uint32_t tests_passed_;
    uint32_t tests_failed_;
    uint32_t total_tests_;

    static const std::vector<uint32_t> kTest1DDimensions;
};

class Mipmap2DArrayTest : public BaseRocR {
public:
    Mipmap2DArrayTest();
    virtual ~Mipmap2DArrayTest();

    void SetUp();
    void TearDown();
    void Run();
    void DisplayTestInfo();
    void DisplayResults() const;
    void Close();

    template<typename T, RocRTextureReadMode readMode = ROCR_READ_MODE_ELEMENT_TYPE,
             RocRTextureFilterMode filterMode = ROCR_FILTER_MODE_POINT,
             RocRTextureAddressMode addressMode = ROCR_ADDRESS_MODE_CLAMP>
    void testMipmapTextureObj2D(size_t width, size_t height, float offsetX = 0.0f, float offsetY = 0.0f);

    template<typename T>
    void populateMipmaps2D(MipmappedArray_t mipmapArray, uint32_t width, uint32_t height,
                          std::vector<MipmapLevelData<T>>& mipmapData);

    template<typename T, RocRTextureFilterMode filterMode = ROCR_FILTER_MODE_POINT,
             RocRTextureAddressMode addressMode = ROCR_ADDRESS_MODE_CLAMP>
    void verifyMipmapLevel2D(hsa_agent_t agent, MipmappedArray_t texMipmap, 
                            T* data, size_t width, size_t height, float level, 
                            float offsetX, float offsetY);

    void TestElementTypeReadMode2D();
    void TestNormalizedFloatReadMode2D();
    void TestLinearFiltering2D();
    void TestAddressModes2D();
    void TestSquareAndRectangularImages();
    void TestVariousDimensions2D();
    void TestErrorConditions2D();
    void TestMemoryIntegrity2D();

    template<typename T>
    hsa_ext_image_format_t GetImageFormat();

    template<typename T>
    hsa_ext_image_channel_type_t GetChannelType();

    template<typename T>
    size_t GetTypeSize();

    template<typename T>
    std::string GetTypeName();
    uint32_t CalculateMipmapLevels(uint32_t width, uint32_t height);
    void LogTestProgress(const std::string& test_name, const std::string& details = "");

private:
    hsa_ext_image_extension_table_t image_extension_table_;
    bool image_extension_supported_;
    std::vector<hsa_agent_t> gpu_agents_;

    // Test statistics
    uint32_t tests_passed_;
    uint32_t tests_failed_;
    uint32_t total_tests_;

    static const std::vector<std::pair<uint32_t, uint32_t>> kTest2DDimensions;
};

class Mipmap3DArrayTest : public BaseRocR {
public:
    Mipmap3DArrayTest();
    virtual ~Mipmap3DArrayTest();

    void SetUp();
    void TearDown();
    void Run();
    void DisplayTestInfo();
    void DisplayResults() const;
    void Close();

    template<typename T, RocRTextureReadMode readMode = ROCR_READ_MODE_ELEMENT_TYPE,
             RocRTextureFilterMode filterMode = ROCR_FILTER_MODE_POINT,
             RocRTextureAddressMode addressMode = ROCR_ADDRESS_MODE_CLAMP>
    void testMipmapTextureObj3D(size_t width, size_t height, size_t depth, 
                               float offsetX = 0.0f, float offsetY = 0.0f, float offsetZ = 0.0f);

    template<typename T>
    void populateMipmaps3D(MipmappedArray_t mipmapArray, uint32_t width, uint32_t height, uint32_t depth,
                          std::vector<MipmapLevelData<T>>& mipmapData);

    template<typename T, RocRTextureFilterMode filterMode = ROCR_FILTER_MODE_POINT,
             RocRTextureAddressMode addressMode = ROCR_ADDRESS_MODE_CLAMP>
    void verifyMipmapLevel3D(hsa_agent_t agent, MipmappedArray_t texMipmap,
                            T* data, size_t width, size_t height, size_t depth, float level,
                            float offsetX, float offsetY, float offsetZ);

    void TestElementTypeReadMode3D();
    void TestNormalizedFloatReadMode3D();
    void TestLinearFiltering3D();
    void TestAddressModes3D();
    void TestVariousDimensions3D();
    void TestErrorConditions3D();
    void TestMemoryIntegrity3D();

    template<typename T>
    hsa_ext_image_format_t GetImageFormat();

    template<typename T>
    hsa_ext_image_channel_type_t GetChannelType();

    template<typename T>
    size_t GetTypeSize();

    template<typename T>
    std::string GetTypeName();
    uint32_t CalculateMipmapLevels(uint32_t width, uint32_t height, uint32_t depth);
    void LogTestProgress(const std::string& test_name, const std::string& details = "");

private:
    hsa_ext_image_extension_table_t image_extension_table_;
    bool image_extension_supported_;
    std::vector<hsa_agent_t> gpu_agents_;

    // Test statistics
    uint32_t tests_passed_;
    uint32_t tests_failed_;
    uint32_t total_tests_;

    static const std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> kTest3DDimensions;
};

} // namespace rocrtst
