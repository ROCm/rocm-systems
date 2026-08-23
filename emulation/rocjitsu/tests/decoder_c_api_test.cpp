// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "simdojo/sim/component.h"
#include "simdojo/sim/simulation.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

// s_endpgm in the GFX9/CDNA SOPP encoding, which gfx1250 rejects.
constexpr rj_code_binary_inst_t kCdnaSEndpgm = 0xBF810000u;

class DecoderPoolScope {
public:
  DecoderPoolScope() { rocjitsu::Decoder::enable_thread_pool(); }
  ~DecoderPoolScope() { rocjitsu::Decoder::disable_thread_pool(); }

  DecoderPoolScope(const DecoderPoolScope &) = delete;
  DecoderPoolScope &operator=(const DecoderPoolScope &) = delete;
};

class WorkerDecodeComponent final : public simdojo::Component {
public:
  WorkerDecodeComponent(rocjitsu::Decoder &decoder, bool &pool_was_active,
                        bool &instruction_used_pool)
      : Component("worker_decode"), decoder_(decoder), pool_was_active_(pool_was_active),
        instruction_used_pool_(instruction_used_pool) {
    decode_event_.set_handler([this](simdojo::Tick, simdojo::Message *) {
      auto *pool = static_cast<rocjitsu::Decoder::Pool *>(rocjitsu::Instruction::alloc_pool_);
      pool_was_active_ = pool != nullptr;
      rocjitsu::DecodeResult decoded = decoder_.decode(&kCdnaSEndpgm);
      instruction_used_pool_ = decoded.succeeded() && pool && pool->owns(decoded.value().get());
    });
  }

  void startup() override { schedule_event(&decode_event_, 1); }

private:
  rocjitsu::Decoder &decoder_;
  bool &pool_was_active_;
  bool &instruction_used_pool_;
  simdojo::Event decode_event_{this, simdojo::EventType::TIMER_CALLBACK};
};

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
  DecoderPoolScope pool_scope;
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
  EXPECT_EQ(rocjitsu::Instruction::alloc_pool_, active_pool);

  EXPECT_STREQ(rj_code_inst_mnemonic(instruction), "s_endpgm");
  rj_code_inst_destroy(instruction);
}

TEST(DecoderCApiTest, HeapAllocationScopeForgetsDisabledPool) {
  auto pooled_decoder = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(pooled_decoder, nullptr);
  pooled_decoder->enable_pool();
  ASSERT_NE(rocjitsu::Instruction::alloc_pool_, nullptr);

  {
    rocjitsu::Instruction::ScopedHeapAllocation heap_allocation;
    pooled_decoder->disable_pool();
    pooled_decoder.reset();
  }
  EXPECT_EQ(rocjitsu::Instruction::alloc_pool_, nullptr);

  auto decoder = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<rocjitsu::Instruction> instruction(decode_valid(*decoder, &kCdnaSEndpgm));
  ASSERT_NE(instruction, nullptr);
  EXPECT_EQ(instruction->mnemonic(), "s_endpgm");
}

TEST(DecoderPoolTest, HoldsLargeCdna5CarryInstructions) {
  DecoderPoolScope pool_scope;
  auto decoder = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  decoder->enable_pool();
  const auto *active_pool =
      static_cast<const rocjitsu::Decoder::Pool *>(rocjitsu::Instruction::alloc_pool_);
  ASSERT_NE(active_pool, nullptr);

  constexpr rj_code_binary_inst_t carry_words[] = {
      0xD7006A06u, 0x02020D08u, // v_add_co_u32 v6, vcc_lo, v8, v6
  };
  rocjitsu::DecodeResult carry = decoder->decode(carry_words);
  ASSERT_TRUE(carry.succeeded());
  EXPECT_EQ(carry.value()->mnemonic(), "v_add_co_u32");
  EXPECT_TRUE(active_pool->owns(carry.value().get()));

  constexpr rj_code_binary_inst_t carry_in_words[] = {
      0xD5207C09u, 0x00060EC1u, // v_add_co_ci_u32_e64 v9, null, -1, v7, s1
  };
  rocjitsu::DecodeResult carry_in = decoder->decode(carry_in_words);
  ASSERT_TRUE(carry_in.succeeded());
  EXPECT_EQ(carry_in.value()->mnemonic(), "v_add_co_ci_u32");
  EXPECT_TRUE(active_pool->owns(carry_in.value().get()));
}

TEST(DecoderPoolTest, GrowsPastFirstSlab) {
  DecoderPoolScope pool_scope;
  auto decoder = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(decoder, nullptr);
  decoder->enable_pool();
  const auto *active_pool =
      static_cast<const rocjitsu::Decoder::Pool *>(rocjitsu::Instruction::alloc_pool_);
  ASSERT_NE(active_pool, nullptr);

  std::vector<std::unique_ptr<rocjitsu::Instruction>> instructions;
  instructions.reserve(rocjitsu::Decoder::Pool::BLOCKS_PER_SLAB + 1);
  for (size_t i = 0; i <= rocjitsu::Decoder::Pool::BLOCKS_PER_SLAB; ++i) {
    rocjitsu::DecodeResult decoded = decoder->decode(&kCdnaSEndpgm);
    ASSERT_TRUE(decoded.succeeded());
    EXPECT_TRUE(active_pool->owns(decoded.value().get()));
    instructions.push_back(std::move(decoded).value());
  }
}

TEST(DecoderPoolTest, ThreadPoolIsSharedAcrossDecoders) {
  DecoderPoolScope pool_scope;
  auto first_decoder = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  auto second_decoder = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(first_decoder, nullptr);
  ASSERT_NE(second_decoder, nullptr);

  rocjitsu::Decoder::enable_thread_pool();
  auto *thread_pool = static_cast<rocjitsu::Decoder::Pool *>(rocjitsu::Instruction::alloc_pool_);
  ASSERT_NE(thread_pool, nullptr);

  rocjitsu::DecodeResult decoded = first_decoder->decode(&kCdnaSEndpgm);
  ASSERT_TRUE(decoded.succeeded());
  ASSERT_TRUE(thread_pool->owns(decoded.value().get()));

  rocjitsu::Decoder::enable_thread_pool();
  EXPECT_EQ(rocjitsu::Instruction::alloc_pool_, thread_pool);
  constexpr rj_code_binary_inst_t carry_words[] = {
      0xD7006A06u, 0x02020D08u, // v_add_co_u32 v6, vcc_lo, v8, v6
  };
  rocjitsu::DecodeResult carry = second_decoder->decode(carry_words);
  ASSERT_TRUE(carry.succeeded());
  EXPECT_TRUE(thread_pool->owns(carry.value().get()));

  decoded.value().reset();
  carry.value().reset();
}

TEST(DecoderPoolTest, DistinguishesSlabBlocksFromOtherPointers) {
  rocjitsu::Decoder::Pool pool;
  void *block = pool.allocate(1);
  ASSERT_NE(block, nullptr);
  EXPECT_TRUE(pool.owns(block));
  EXPECT_FALSE(pool.owns(static_cast<std::byte *>(block) + 1));
  pool.deallocate(block);

  void *oversized = pool.allocate(rocjitsu::Decoder::Pool::BLOCK_SIZE + 1);
  ASSERT_NE(oversized, nullptr);
  EXPECT_FALSE(pool.owns(oversized));
  pool.deallocate(oversized);
}

TEST(DecoderPoolTest, WorkerLifecycleCrossesDecoderConstructionThread) {
  ASSERT_EQ(rocjitsu::Instruction::alloc_pool_, nullptr);
  auto decoder = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(decoder, nullptr);

  bool pool_was_active = false;
  bool instruction_used_pool = false;
  bool pool_was_released = false;
  simdojo::SimulationEngine::Config config{};
  config.worker_start = [] { rocjitsu::Decoder::enable_thread_pool(); };
  config.worker_stop = [] { rocjitsu::Decoder::disable_thread_pool(); };
  simdojo::SimulationEngine engine(config);
  auto root = std::make_unique<simdojo::CompositeComponent>("root");
  root->add_child(
      std::make_unique<WorkerDecodeComponent>(*decoder, pool_was_active, instruction_used_pool));
  engine.topology().set_root(std::move(root));
  engine.create();

  std::thread worker([&] {
    engine.run();
    pool_was_released = rocjitsu::Instruction::alloc_pool_ == nullptr;
  });
  worker.join();

  EXPECT_TRUE(pool_was_active);
  EXPECT_TRUE(instruction_used_pool);
  EXPECT_TRUE(pool_was_released);
  EXPECT_EQ(rocjitsu::Instruction::alloc_pool_, nullptr);
}

TEST(DecoderPoolTest, WorkerBindingComposesExistingHooksInLifoOrder) {
  ASSERT_EQ(rocjitsu::Instruction::alloc_pool_, nullptr);
  bool prior_start_saw_heap = false;
  bool prior_stop_saw_heap = false;
  simdojo::SimulationEngine::Config config{};
  config.worker_start = [&] {
    prior_start_saw_heap = rocjitsu::Instruction::alloc_pool_ == nullptr;
  };
  config.worker_stop = [&] { prior_stop_saw_heap = rocjitsu::Instruction::alloc_pool_ == nullptr; };

  rocjitsu::config::bind_decoder_worker_pool(config);
  config.worker_start();
  EXPECT_TRUE(prior_start_saw_heap);
  EXPECT_NE(rocjitsu::Instruction::alloc_pool_, nullptr);

  config.worker_stop();
  EXPECT_TRUE(prior_stop_saw_heap);
  EXPECT_EQ(rocjitsu::Instruction::alloc_pool_, nullptr);
}

} // namespace
