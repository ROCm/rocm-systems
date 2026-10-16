// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/cdna5/isa.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>

namespace rocjitsu {

class DecoderPoolTestAccess {
public:
  static const Decoder::Pool *pool(const Decoder &decoder) { return decoder.pool_.get(); }
};

} // namespace rocjitsu

namespace {

// Leave room for ABI differences while rejecting embedded allocation buffers.
static_assert(sizeof(rocjitsu::Decoder) <= 64, "base decoders must retain a compact layout");
static_assert(sizeof(rocjitsu::IsaDecoder<rocjitsu::cdna5::Isa>) <= 64,
              "ISA decoders must retain a compact layout");

// s_endpgm in the GFX9/CDNA SOPP encoding, which gfx1250 rejects.
constexpr rj_code_binary_inst_t kCdnaSEndpgm = 0xBF810000u;

static_assert(std::is_same_v<decltype(&rj_code_inst_destroy), void (*)(rj_code_inst_t *)>,
              "standalone destruction must not accept borrowed const instructions");
static_assert(
    std::is_same_v<decltype(rj_code_basic_block_first_inst(nullptr)), const rj_code_inst_t *>,
    "borrowed instructions must remain const-qualified");
static_assert(std::is_same_v<decltype(rj_code_inst_next(nullptr)), const rj_code_inst_t *>,
              "instruction traversal must preserve the borrowed const qualification");

TEST(DecoderCApiTest, InvalidInstructionReturnsErrorAndClearsOutput) {
  rj_code_decoder_t *decoder = nullptr;
  ASSERT_EQ(rj_code_decoder_create(ROCJITSU_CODE_ARCH_CDNA5, &decoder), ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(decoder, nullptr);

  auto *instruction = reinterpret_cast<rj_code_inst_t *>(static_cast<uintptr_t>(1));
  EXPECT_EQ(rj_code_decoder_decode(decoder, &kCdnaSEndpgm, &instruction), ROCJITSU_STATUS_ERROR);
  EXPECT_EQ(instruction, nullptr);

  rj_code_decoder_destroy(decoder);
}

TEST(DecoderCApiTest, InvalidArgumentsClearWritableOutputs) {
  auto *instruction = reinterpret_cast<rj_code_inst_t *>(static_cast<uintptr_t>(1));
  EXPECT_EQ(rj_code_decoder_decode(nullptr, &kCdnaSEndpgm, &instruction),
            ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(instruction, nullptr);

  auto *decoder = reinterpret_cast<rj_code_decoder_t *>(static_cast<uintptr_t>(1));
  EXPECT_EQ(rj_code_decoder_create(ROCJITSU_CODE_ARCH_INVALID, &decoder),
            ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(decoder, nullptr);

  decoder = reinterpret_cast<rj_code_decoder_t *>(static_cast<uintptr_t>(1));
  EXPECT_EQ(rj_code_decoder_create_for_target(nullptr, &decoder), ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(decoder, nullptr);

  decoder = reinterpret_cast<rj_code_decoder_t *>(static_cast<uintptr_t>(1));
  EXPECT_EQ(rj_code_decoder_create_for_target("", &decoder), ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(decoder, nullptr);
}

TEST(DecoderCApiTest, StandaloneInstructionsAreCallerOwned) {
  rj_code_decoder_t *decoder = nullptr;
  ASSERT_EQ(rj_code_decoder_create(ROCJITSU_CODE_ARCH_CDNA3, &decoder), ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(decoder, nullptr);

  rj_code_inst_t *instruction = nullptr;
  ASSERT_EQ(rj_code_decoder_decode(decoder, &kCdnaSEndpgm, &instruction), ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(instruction, nullptr);
  EXPECT_STREQ(rj_code_inst_mnemonic(instruction), "s_endpgm");
  rj_code_inst_destroy(instruction);

  rj_code_inst_t *survivor = nullptr;
  ASSERT_EQ(rj_code_decoder_decode(decoder, &kCdnaSEndpgm, &survivor), ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(survivor, nullptr);
  rj_code_decoder_destroy(decoder);

  EXPECT_STREQ(rj_code_inst_mnemonic(survivor), "s_endpgm");
  EXPECT_GT(rj_code_inst_size(survivor), 0u);
  char disassembly[64]{};
  EXPECT_EQ(rj_code_inst_disassemble(survivor, disassembly, sizeof(disassembly)),
            ROCJITSU_STATUS_SUCCESS);
  EXPECT_NE(std::strstr(disassembly, "s_endpgm"), nullptr);
  EXPECT_EQ(rj_code_inst_next(survivor), nullptr);

  rj_code_inst_destroy(survivor);
  rj_code_inst_destroy(nullptr);
}

TEST(DecoderCApiTest, StandaloneInstructionIgnoresAmbientDecoderPool) {
  auto pooled_decoder = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(pooled_decoder, nullptr);
  pooled_decoder->enable_pool();
  const auto *active_pool =
      static_cast<const rocjitsu::Decoder::Pool *>(rocjitsu::Instruction::alloc_pool_);
  ASSERT_NE(active_pool, nullptr);

  rj_code_decoder_t *decoder = nullptr;
  ASSERT_EQ(rj_code_decoder_create(ROCJITSU_CODE_ARCH_CDNA3, &decoder), ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(decoder, nullptr);

  rj_code_inst_t *instruction = nullptr;
  ASSERT_EQ(rj_code_decoder_decode(decoder, &kCdnaSEndpgm, &instruction), ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(instruction, nullptr);
  EXPECT_FALSE(active_pool->owns(instruction));
  EXPECT_EQ(rocjitsu::Instruction::alloc_pool_, active_pool);
  rj_code_inst_destroy(instruction);
  EXPECT_EQ(rocjitsu::Instruction::alloc_pool_, active_pool);

  instruction = nullptr;
  ASSERT_EQ(rj_code_decoder_decode(decoder, &kCdnaSEndpgm, &instruction), ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(instruction, nullptr);
  EXPECT_FALSE(active_pool->owns(instruction));

  rj_code_decoder_destroy(decoder);
  pooled_decoder.reset();
  EXPECT_EQ(rocjitsu::Instruction::alloc_pool_, nullptr);

  EXPECT_STREQ(rj_code_inst_mnemonic(instruction), "s_endpgm");
  rj_code_inst_destroy(instruction);
}

TEST(DecoderCApiTest, HeapAllocationScopeForgetsDestroyedPool) {
  auto pooled_decoder = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(pooled_decoder, nullptr);
  pooled_decoder->enable_pool();
  ASSERT_NE(rocjitsu::Instruction::alloc_pool_, nullptr);

  {
    rocjitsu::Instruction::ScopedHeapAllocation heap_allocation;
    pooled_decoder.reset();
  }
  EXPECT_EQ(rocjitsu::Instruction::alloc_pool_, nullptr);

  auto decoder = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<rocjitsu::Instruction> instruction(decode_valid(*decoder, &kCdnaSEndpgm));
  ASSERT_NE(instruction, nullptr);
  EXPECT_EQ(instruction->mnemonic(), "s_endpgm");
}

TEST(DecoderPoolTest, AllocatesOnlyOnFirstEnablement) {
  auto decoder = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(decoder, nullptr);
  EXPECT_EQ(rocjitsu::DecoderPoolTestAccess::pool(*decoder), nullptr);
  {
    std::unique_ptr<rocjitsu::Instruction> instruction(decode_valid(*decoder, &kCdnaSEndpgm));
    ASSERT_NE(instruction, nullptr);
    EXPECT_EQ(instruction->mnemonic(), "s_endpgm");
    EXPECT_EQ(rocjitsu::DecoderPoolTestAccess::pool(*decoder), nullptr);
  }

  decoder->enable_pool();
  const auto *pool = rocjitsu::DecoderPoolTestAccess::pool(*decoder);
  ASSERT_NE(pool, nullptr);
  EXPECT_EQ(rocjitsu::Instruction::alloc_pool_, pool);

  decoder->disable_pool();
  EXPECT_EQ(rocjitsu::Instruction::alloc_pool_, nullptr);
  EXPECT_EQ(rocjitsu::DecoderPoolTestAccess::pool(*decoder), pool);

  decoder->enable_pool();
  EXPECT_EQ(rocjitsu::Instruction::alloc_pool_, pool);
  EXPECT_EQ(rocjitsu::DecoderPoolTestAccess::pool(*decoder), pool);
}

TEST(DecoderPoolTest, ReenablePreservesPoolAndOutstandingInstruction) {
  auto decoder = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(decoder, nullptr);
  decoder->enable_pool();
  auto *pool = static_cast<rocjitsu::Decoder::Pool *>(rocjitsu::Instruction::alloc_pool_);
  ASSERT_NE(pool, nullptr);
  std::unique_ptr<rocjitsu::Instruction> first(decode_valid(*decoder, &kCdnaSEndpgm));
  ASSERT_NE(first, nullptr);
  EXPECT_TRUE(pool->owns(first.get()));
  decoder->enable_pool();
  EXPECT_EQ(rocjitsu::Instruction::alloc_pool_, pool);
  decoder->disable_pool();
  EXPECT_EQ(rocjitsu::Instruction::alloc_pool_, nullptr);
  {
    std::unique_ptr<rocjitsu::Instruction> heap_instruction(decode_valid(*decoder, &kCdnaSEndpgm));
    EXPECT_NE(heap_instruction, nullptr);
    EXPECT_FALSE(pool->owns(heap_instruction.get()));
  }
  decoder->enable_pool();
  EXPECT_EQ(rocjitsu::Instruction::alloc_pool_, pool);
  std::unique_ptr<rocjitsu::Instruction> second(decode_valid(*decoder, &kCdnaSEndpgm));
  ASSERT_NE(second, nullptr);
  EXPECT_TRUE(pool->owns(second.get()));
  EXPECT_NE(first.get(), second.get());
  EXPECT_EQ(first->mnemonic(), "s_endpgm");
}

TEST(DecoderPoolTest, UnusedDecoderDoesNotInvalidateAnotherPool) {
  auto pooled_decoder = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(pooled_decoder, nullptr);
  pooled_decoder->enable_pool();
  auto *pool = static_cast<rocjitsu::Decoder::Pool *>(rocjitsu::Instruction::alloc_pool_);
  ASSERT_NE(pool, nullptr);
  auto alloc_fn = rocjitsu::Instruction::alloc_fn_;
  auto dealloc_fn = rocjitsu::Instruction::dealloc_fn_;
  {
    auto unused = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
    ASSERT_NE(unused, nullptr);
    unused->disable_pool();
    EXPECT_EQ(rocjitsu::Instruction::alloc_pool_, pool);
    EXPECT_EQ(rocjitsu::Instruction::alloc_fn_, alloc_fn);
    EXPECT_EQ(rocjitsu::Instruction::dealloc_fn_, dealloc_fn);
  }
  EXPECT_EQ(rocjitsu::Instruction::alloc_pool_, pool);
  EXPECT_EQ(rocjitsu::Instruction::alloc_fn_, alloc_fn);
  EXPECT_EQ(rocjitsu::Instruction::dealloc_fn_, dealloc_fn);
  {
    rocjitsu::Instruction::ScopedHeapAllocation heap_allocation;
    auto unused = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
    ASSERT_NE(unused, nullptr);
    unused->disable_pool();
  }
  EXPECT_EQ(rocjitsu::Instruction::alloc_pool_, pool);
  EXPECT_EQ(rocjitsu::Instruction::alloc_fn_, alloc_fn);
  EXPECT_EQ(rocjitsu::Instruction::dealloc_fn_, dealloc_fn);
  std::unique_ptr<rocjitsu::Instruction> instruction(decode_valid(*pooled_decoder, &kCdnaSEndpgm));
  ASSERT_NE(instruction, nullptr);
  EXPECT_TRUE(pool->owns(instruction.get()));
}

TEST(DecoderPoolTest, UnusedDecoderPreservesNullContextAllocatorHooks) {
  rocjitsu::Instruction::ScopedHeapAllocation restore_allocator;
  const rocjitsu::Instruction::AllocFn alloc_fn = [](void *, size_t size) {
    return ::operator new(size);
  };
  const rocjitsu::Instruction::DeallocFn dealloc_fn = [](void *, void *ptr) {
    ::operator delete(ptr);
  };
  rocjitsu::Instruction::alloc_fn_ = alloc_fn;
  rocjitsu::Instruction::dealloc_fn_ = dealloc_fn;
  ASSERT_EQ(rocjitsu::Instruction::alloc_pool_, nullptr);
  {
    auto unused = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
    ASSERT_NE(unused, nullptr);
    unused->disable_pool();
    EXPECT_EQ(rocjitsu::Instruction::alloc_pool_, nullptr);
    EXPECT_EQ(rocjitsu::Instruction::alloc_fn_, alloc_fn);
    EXPECT_EQ(rocjitsu::Instruction::dealloc_fn_, dealloc_fn);
  }
  EXPECT_EQ(rocjitsu::Instruction::alloc_pool_, nullptr);
  EXPECT_EQ(rocjitsu::Instruction::alloc_fn_, alloc_fn);
  EXPECT_EQ(rocjitsu::Instruction::dealloc_fn_, dealloc_fn);
  {
    rocjitsu::Instruction::ScopedHeapAllocation heap_allocation;
    auto unused = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
    ASSERT_NE(unused, nullptr);
    unused->disable_pool();
  }
  EXPECT_EQ(rocjitsu::Instruction::alloc_pool_, nullptr);
  EXPECT_EQ(rocjitsu::Instruction::alloc_fn_, alloc_fn);
  EXPECT_EQ(rocjitsu::Instruction::dealloc_fn_, dealloc_fn);
}

} // namespace
