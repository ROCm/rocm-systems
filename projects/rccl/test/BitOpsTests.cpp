/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "bitops.h"
#include "gtest/gtest.h"

namespace RcclUnitTesting
{

TEST(DIVUP, DIVUPSuccess) {
    EXPECT_EQ(DIVUP(0, 8), 0);
    EXPECT_EQ(DIVUP(1, 8), 1);
    EXPECT_EQ(DIVUP(7, 8), 1);
    EXPECT_EQ(DIVUP(8, 8), 1);
    EXPECT_EQ(DIVUP(9, 8), 2);
}

TEST(ROUNDUP, ROUNDUPSuccess) {
    EXPECT_EQ(ROUNDUP(0, 8), 0);
    EXPECT_EQ(ROUNDUP(1, 8), 8);
    EXPECT_EQ(ROUNDUP(7, 8), 8);
    EXPECT_EQ(ROUNDUP(8, 8), 8);
    EXPECT_EQ(ROUNDUP(9, 8), 16);
    EXPECT_EQ(ROUNDUP(15, 8), 16);
    EXPECT_EQ(ROUNDUP(16, 8), 16);
    EXPECT_EQ(ROUNDUP(17, 8), 24);
}

TEST(ALIGN_POWER, ALIGN_POWERSuccess) {
    EXPECT_EQ(ALIGN_POWER(7, 8), 8);
    EXPECT_EQ(ALIGN_POWER(8, 8), 8);
    EXPECT_EQ(ALIGN_POWER(9, 8), 16);
}

TEST(ALIGN_SIZE, ALIGN_SIZESuccess) {
    int alignSize = 1;
    ALIGN_SIZE(alignSize, 8);
    EXPECT_EQ(alignSize, 8);
    alignSize = 7;
    ALIGN_SIZE(alignSize, 8);
    EXPECT_EQ(alignSize, 8);
}

TEST(u32fp8MaxValue, u32fp8MaxValueSuccess) {
    EXPECT_EQ(u32fp8MaxValue(), 0xf0000000);
}

TEST(u32fp8Decode, u32fp8DecodeSuccess) {
    uint32_t u32Val{2};
    EXPECT_EQ(u32fp8Decode(u32Val), static_cast<uint8_t>(2));
}

TEST(u32fp8Encode, u32fp8EncodeSuccess) {
    uint8_t u8Val{2};
    EXPECT_EQ(u32fp8Encode(u8Val), static_cast<uint32_t>(2));
}

TEST(u32fpEncode, u32fpEncodeSuccess) {
    // log2x is 1, use bitsPerPow2 = 1
    uint32_t u32Val{0xFFFFFFFF}; // 32 bits set to 1
    uint32_t u32ExpectVal = 63;
    int bitsPerPow2 = 1;
    EXPECT_EQ(u32fpEncode(u32Val, bitsPerPow2), u32ExpectVal);
}

TEST(u32fpDecode, u32fpDecodeSuccess) {
    // log2x is 1, use bitsPerPow2 = 1
    uint32_t u32Val{0xFFFFFFFF}; // 32 bits set to 1
    uint32_t u32ExpectVal = 63;
    int bitsPerPow2 = 1;
    EXPECT_EQ(u32fpDecode(u32Val, bitsPerPow2), u32ExpectVal);
}

TEST(getHash, getHashSuccess) {
    std::vector<uint32_t> u32Vec{2, 4, 8, 16, 32, 64, 128, 256};
    uint64_t expectedHash = 0xa4495d05731e3337;
    auto ret = getHash(u32Vec.data(), u32Vec.size() * sizeof(uint32_t));
    EXPECT_EQ(ret, expectedHash);

    std::vector<float> floatVec{2.0f};
    expectedHash = 0xedbaa57e84d6dbaa;
    ret = getHash(floatVec.data());
    EXPECT_EQ(ret, expectedHash);
}

TEST(eatHash, eatHashSuccess){
    uint64_t acc[2] = {0, 0};
    uint64_t expectedAcc[2] = {8617830242246783886ull, 2410367826245614052ull};
    std::vector<uint32_t> u32Vec{2, 4, 8, 16, 32, 64, 128, 256};
    eatHash(acc, u32Vec.data());
    EXPECT_EQ(acc[0], expectedAcc[0]);
    EXPECT_EQ(acc[1], expectedAcc[1]);
}

template <typename T>
class BitOpsTemplateAllIntTestsFixture : public testing::Test {
public:
    ~BitOpsTemplateAllIntTestsFixture() override = default;
protected:
    T valX_{};
    T valY_{};
    T valZ_{};
};

using BitOpsAllIntTypes = ::testing::Types<short, unsigned short, int, unsigned int, long, unsigned long, long long, unsigned long long>;

TYPED_TEST_SUITE(BitOpsTemplateAllIntTestsFixture, BitOpsAllIntTypes);

TYPED_TEST(BitOpsTemplateAllIntTestsFixture, divUpSuccess) {
    this->valX_ = 0;
    this->valY_= 8;
    this->valZ_= 0;
    EXPECT_EQ(divUp(this->valX_, this->valY_), this->valZ_);
    this->valX_ = 1;
    this->valY_= 8;
    this->valZ_= 1;
    EXPECT_EQ(divUp(this->valX_, this->valY_), this->valZ_);
    this->valX_ = 7;
    this->valY_= 8;
    this->valZ_= 1;
    EXPECT_EQ(divUp(this->valX_, this->valY_), this->valZ_);
}

TYPED_TEST(BitOpsTemplateAllIntTestsFixture, roundUpSuccess) {
    this->valX_ = 0;
    this->valY_= 8;
    this->valZ_= 0;
    EXPECT_EQ(roundUp(this->valX_, this->valY_), this->valZ_);
    this->valX_ = 1;
    this->valY_= 8;
    this->valZ_= 8;
    EXPECT_EQ(roundUp(this->valX_, this->valY_), this->valZ_);
    this->valX_ = 7;
    this->valY_= 8;
    this->valZ_= 8;
    EXPECT_EQ(roundUp(this->valX_, this->valY_), this->valZ_);
}

TYPED_TEST(BitOpsTemplateAllIntTestsFixture, roundDownSuccess) {
    this->valX_ = 0;
    this->valY_= 8;
    this->valZ_= 0;
    EXPECT_EQ(roundDown(this->valX_, this->valY_), this->valZ_);
    this->valX_ = 1;
    this->valY_= 8;
    this->valZ_= 0;
    EXPECT_EQ(roundDown(this->valX_, this->valY_), this->valZ_);
    this->valX_ = 7;
    this->valY_= 8;
    this->valZ_= 0;
    EXPECT_EQ(roundDown(this->valX_, this->valY_), this->valZ_);
}

TYPED_TEST(BitOpsTemplateAllIntTestsFixture, alignUpSuccess) {
    this->valX_ = 0;
    this->valY_= 8;
    this->valZ_= 0;
    EXPECT_EQ(alignUp(this->valX_, this->valY_), this->valZ_);
    this->valX_ = 1;
    this->valY_= 8;
    this->valZ_= 8;
    EXPECT_EQ(alignUp(this->valX_, this->valY_), this->valZ_);
    this->valX_ = 7;
    this->valY_= 8;
    this->valZ_= 8;
    EXPECT_EQ(alignUp(this->valX_, this->valY_), this->valZ_);
}

TYPED_TEST(BitOpsTemplateAllIntTestsFixture, alignDownSuccess) {
    this->valX_ = 0;
    this->valY_= 8;
    this->valZ_= 0;
    EXPECT_EQ(alignDown(this->valX_, this->valY_), this->valZ_);
    this->valX_ = 1;
    this->valY_= 8;
    this->valZ_= 0;
    EXPECT_EQ(alignDown(this->valX_, this->valY_), this->valZ_);
    this->valX_ = 9;
    this->valY_= 8;
    this->valZ_= 8;
    EXPECT_EQ(alignDown(this->valX_, this->valY_), this->valZ_);
}

TYPED_TEST(BitOpsTemplateAllIntTestsFixture, BitOpsCountOneBitsSuccess) {
    this->valX_ = 3; // 0b11
    int expectedBitCount = 2; 
    EXPECT_EQ(countOneBits(this->valX_), expectedBitCount);
}

TYPED_TEST(BitOpsTemplateAllIntTestsFixture, BitOpsFirstOneBitsSuccess) {
    this->valX_ = 3; // 0b11
    int expectedFirstOneBitIndex = 0; 
    EXPECT_EQ(firstOneBit(this->valX_), expectedFirstOneBitIndex);
}

TYPED_TEST(BitOpsTemplateAllIntTestsFixture, BitOpsPopFirstOneBitsSuccess) {
    this->valX_ = 3; // 0b11
    int expectedPopFirstOneBitIndex = 0; // 0b01
    EXPECT_EQ(popFirstOneBit(&(this->valX_)), expectedPopFirstOneBitIndex);
}

TYPED_TEST(BitOpsTemplateAllIntTestsFixture, log2DownSuccess) {
    this->valX_ = 0;
    EXPECT_EQ(log2Down(this->valX_), -1);
    this->valX_ = 1;
    EXPECT_EQ(log2Down(this->valX_), 0);
    this->valX_ = 2;
    EXPECT_EQ(log2Down(this->valX_), 1);
}


TYPED_TEST(BitOpsTemplateAllIntTestsFixture, log2UpSuccess) {
    this->valX_ = 0;
    EXPECT_EQ(log2Up(this->valX_), 0);
    this->valX_ = 1;
    EXPECT_EQ(log2Up(this->valX_), 0);
    this->valX_ = 2;
    EXPECT_EQ(log2Up(this->valX_), 1);
    this->valX_ = 3;
    EXPECT_EQ(log2Up(this->valX_), 2);
}

TYPED_TEST(BitOpsTemplateAllIntTestsFixture, pow2UpSuccess){
    this->valX_ = 0;
    EXPECT_EQ(pow2Up(this->valX_), 1);
    this->valX_ = 1;
    EXPECT_EQ(pow2Up(this->valX_), 1);
    this->valX_ = 2;
    EXPECT_EQ(pow2Up(this->valX_), 2);
    this->valX_ = 3;
    EXPECT_EQ(pow2Up(this->valX_), 4);
}

TYPED_TEST(BitOpsTemplateAllIntTestsFixture, pow2DownSuccess){
    this->valX_ = 1;
    EXPECT_EQ(pow2Down(this->valX_), 1);
    this->valX_ = 2;
    EXPECT_EQ(pow2Down(this->valX_), 2);
    this->valX_ = 3;
    EXPECT_EQ(pow2Down(this->valX_), 2);
}

template <typename T>
class BitOpsTemplateUnsignedTestsFixture : public testing::Test {
public:
    ~BitOpsTemplateUnsignedTestsFixture() override = default;
protected:
    T reverseSubBits_{2};
    T reverseBits_{2};
};

using BitOpsUnsignedTypes = ::testing::Types<unsigned short, unsigned int, unsigned long, unsigned long long>;

TYPED_TEST_SUITE(BitOpsTemplateUnsignedTestsFixture, BitOpsUnsignedTypes);

TYPED_TEST(BitOpsTemplateUnsignedTestsFixture, reverseSubBitsSuccess) {
    auto ret = reverseSubBits<TypeParam, 1>(this->reverseSubBits_);
    EXPECT_EQ(ret, this->reverseSubBits_);
    ret = reverseSubBits<TypeParam, 2>(this->reverseSubBits_);
    EXPECT_EQ(ret, 1);
    ret = reverseSubBits<TypeParam, 4>(this->reverseSubBits_);
    EXPECT_EQ(ret, 4); 
    ret = reverseSubBits<TypeParam, 8>(this->reverseSubBits_);
    EXPECT_EQ(ret, 64); 
    ret = reverseSubBits<TypeParam, 16>(this->reverseSubBits_);
    EXPECT_EQ(ret, 16384);
    if(!std::is_same<TypeParam, unsigned short>::value) {
        ret = reverseSubBits<TypeParam, 32>(this->reverseSubBits_);
        EXPECT_EQ(ret, 1073741824);
    }
}

TYPED_TEST(BitOpsTemplateUnsignedTestsFixture, reverseBitsSuccess) {
    auto ret = reverseBits(this->reverseBits_, 2);
    EXPECT_EQ(ret, 1);
    ret = reverseBits(this->reverseBits_, 16);
    EXPECT_EQ(ret, 16384);
}

}