// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/rj_code.h"

#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/executable.h"
#include "rocjitsu/code/instruction_list.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/refcount.h"

#include <cstring>
#include <memory>
#include <vector>

using namespace rocjitsu;

namespace {

Decoder *create_decoder_for_target(rj_code_target_id_t target) {
  static thread_local std::unique_ptr<Decoder> cdna3_decoder;
  static thread_local std::unique_ptr<Decoder> cdna4_decoder;

  switch (target) {
  case ROCJITSU_CODE_TARGET_GFX942:
    if (!cdna3_decoder)
      cdna3_decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
    return cdna3_decoder.get();
  case ROCJITSU_CODE_TARGET_GFX950:
    if (!cdna4_decoder)
      cdna4_decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    return cdna4_decoder.get();
  default:
    return nullptr;
  }
}

} // namespace

struct rj_code_executable_t : RefCounted {
  std::unique_ptr<Executable> exec;
};

struct rj_code_object_t : RefCounted {
  AmdGpuCodeObject *co = nullptr;
};

struct rj_code_inst_list_t : RefCounted {
  InstructionList list;
  std::vector<std::unique_ptr<Instruction>> storage;
};

struct rj_code_basic_block_list_t : RefCounted {
  std::vector<std::unique_ptr<BasicBlock>> blocks;
};

struct rj_code_basic_block_t : RefCounted {
  BasicBlock *block = nullptr;
};

rj_status_t rj_code_executable_create(const char *path, rj_code_executable_t **exec) {
  if (!path || !exec)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;

  auto executable = std::make_unique<Executable>(path);
  if (!executable->is_valid())
    return ROCJITSU_STATUS_INVALID_CODE_OBJECT;

  *exec = new rj_code_executable_t{};
  (*exec)->exec = std::move(executable);
  return ROCJITSU_STATUS_SUCCESS;
}

void rj_code_executable_retain(rj_code_executable_t *exec) {
  if (exec)
    exec->retain();
}

void rj_code_executable_release(rj_code_executable_t *exec) {
  if (!exec)
    return;
  if (exec->release())
    delete exec;
}

void rj_code_executable_destroy(rj_code_executable_t *exec) {
  if (!exec)
    return;
  if (exec->destroy())
    delete exec;
}

uint32_t rj_code_executable_num_code_objects(const rj_code_executable_t *exec,
                                             rj_code_target_id_t target) {
  if (!exec || !exec->exec)
    return 0;
  return exec->exec->num_code_objects(target);
}

rj_status_t rj_code_executable_get_code_object(const rj_code_executable_t *exec,
                                               rj_code_target_id_t target, uint32_t index,
                                               rj_code_object_t **obj) {
  if (!exec || !exec->exec || !obj)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;

  auto *co = const_cast<Executable *>(exec->exec.get())->code_object(target, index);
  if (!co)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;

  *obj = new rj_code_object_t{};
  (*obj)->co = co;
  (*obj)->retain();
  return ROCJITSU_STATUS_SUCCESS;
}

void rj_code_object_retain(rj_code_object_t *obj) {
  if (obj)
    obj->retain();
}

void rj_code_object_release(rj_code_object_t *obj) {
  if (!obj)
    return;
  if (obj->release())
    delete obj;
}

void rj_code_object_destroy(rj_code_object_t *obj) {
  if (!obj)
    return;
  if (obj->destroy())
    delete obj;
}

rj_status_t rj_code_inst_list_create(rj_code_object_t *obj, rj_code_target_id_t target_id,
                                     rj_code_inst_list_t **inst_list) {
  if (!obj || !obj->co || !inst_list)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;

  auto *decoder = create_decoder_for_target(target_id);
  if (!decoder)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;

  auto owned = std::make_unique<rj_code_inst_list_t>();

  for (const auto *sec : obj->co->text_sections()) {
    const auto *inst_data = reinterpret_cast<const uint32_t *>(sec->data());
    std::size_t inst_data_size = sec->size() / sizeof(uint32_t);
    uint64_t pc = 0;
    while (pc < inst_data_size) {
      auto inst = decoder->decode(&inst_data[pc]);
      owned->list.push_back(*inst);
      ++pc;
      if (inst->size() == 8)
        ++pc;
      owned->storage.push_back(std::move(inst));
    }
  }

  *inst_list = owned.release();
  return ROCJITSU_STATUS_SUCCESS;
}

void rj_code_inst_list_retain(rj_code_inst_list_t *inst_list) {
  if (inst_list)
    inst_list->retain();
}

void rj_code_inst_list_release(rj_code_inst_list_t *inst_list) {
  if (!inst_list)
    return;
  if (inst_list->release())
    delete inst_list;
}

void rj_code_inst_list_destroy(rj_code_inst_list_t *inst_list) {
  if (!inst_list)
    return;
  if (inst_list->destroy())
    delete inst_list;
}

rj_status_t rj_code_basic_block_list_create(rj_code_object_t *obj, rj_code_target_id_t target_id,
                                            rj_code_basic_block_list_t **list) {
  if (!obj || !obj->co || !list)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;

  auto *decoder = create_decoder_for_target(target_id);
  if (!decoder)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;

  auto owned = std::make_unique<rj_code_basic_block_list_t>();
  owned->blocks = BasicBlock::build(*obj->co, *decoder);

  *list = owned.release();
  return ROCJITSU_STATUS_SUCCESS;
}

void rj_code_basic_block_list_retain(rj_code_basic_block_list_t *list) {
  if (list)
    list->retain();
}

void rj_code_basic_block_list_release(rj_code_basic_block_list_t *list) {
  if (!list)
    return;
  if (list->release())
    delete list;
}

void rj_code_basic_block_list_destroy(rj_code_basic_block_list_t *list) {
  if (!list)
    return;
  if (list->destroy())
    delete list;
}

uint32_t rj_code_basic_block_list_size(const rj_code_basic_block_list_t *list) {
  if (!list)
    return 0;
  return static_cast<uint32_t>(list->blocks.size());
}

rj_status_t rj_code_basic_block_list_get(const rj_code_basic_block_list_t *list, uint32_t index,
                                         rj_code_basic_block_t **block) {
  if (!list || !block || index >= list->blocks.size())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;

  *block = new rj_code_basic_block_t{};
  (*block)->block = list->blocks[index].get();
  (*block)->retain();
  return ROCJITSU_STATUS_SUCCESS;
}

void rj_code_basic_block_retain(rj_code_basic_block_t *block) {
  if (block)
    block->retain();
}

void rj_code_basic_block_release(rj_code_basic_block_t *block) {
  if (!block)
    return;
  if (block->release())
    delete block;
}

void rj_code_basic_block_destroy(rj_code_basic_block_t *block) {
  if (!block)
    return;
  if (block->destroy())
    delete block;
}

uint64_t rj_code_basic_block_start_offset(const rj_code_basic_block_t *block) {
  if (!block || !block->block)
    return 0;
  return block->block->start_offset();
}

uint32_t rj_code_basic_block_size(const rj_code_basic_block_t *block) {
  if (!block || !block->block)
    return 0;
  return block->block->size();
}

uint32_t rj_code_basic_block_num_instructions(const rj_code_basic_block_t *block) {
  if (!block || !block->block)
    return 0;
  return block->block->num_instructions();
}

const char *rj_code_inst_mnemonic(const rj_code_inst_t *inst) {
  if (!inst)
    return nullptr;
  return reinterpret_cast<const Instruction *>(inst)->mnemonic().c_str();
}

uint32_t rj_code_inst_size(const rj_code_inst_t *inst) {
  if (!inst)
    return 0;
  return static_cast<uint32_t>(reinterpret_cast<const Instruction *>(inst)->size());
}

uint32_t rj_code_inst_flags(const rj_code_inst_t *inst) {
  if (!inst)
    return 0;
  const auto *i = reinterpret_cast<const Instruction *>(inst);
  uint32_t flags = 0;
  if (i->is_branch())
    flags |= RJ_CODE_INST_BRANCH;
  if (i->is_memory_op())
    flags |= RJ_CODE_INST_MEMORY_OP;
  return flags;
}

rj_status_t rj_code_inst_disassemble(const rj_code_inst_t *inst, char *buf, uint32_t buf_size) {
  if (!inst || !buf || buf_size == 0)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;

  const auto &dis = reinterpret_cast<const Instruction *>(inst)->disassemble();
  std::strncpy(buf, dis.c_str(), buf_size - 1);
  buf[buf_size - 1] = '\0';
  return ROCJITSU_STATUS_SUCCESS;
}

const rj_code_inst_t *rj_code_basic_block_first_inst(const rj_code_basic_block_t *block) {
  if (!block || !block->block)
    return nullptr;
  auto &list = block->block->instructions();
  if (list.empty())
    return nullptr;
  auto it = list.begin();
  return reinterpret_cast<const rj_code_inst_t *>(it.node_pointer());
}

const rj_code_inst_t *rj_code_inst_next(const rj_code_inst_t *inst) {
  if (!inst)
    return nullptr;
  const auto *node =
      reinterpret_cast<const util::IListNode<Instruction, util::IListParent<BasicBlock>> *>(inst);
  const auto *next_node = node->next_;
  // The sentinel node is not a valid Instruction. Detect it by checking if
  // the next node's next pointer loops back (sentinel.next == first or sentinel).
  // A simpler heuristic: the sentinel is the only node where next->prev == node
  // AND the node is NOT an Instruction. Since we can't easily distinguish,
  // we use the parent pointer: real instructions in the list have parent_ set
  // to their BasicBlock, while the sentinel does not.
  if (!next_node)
    return nullptr;
  // Sentinel nodes have parent_ == nullptr (they're default-constructed).
  const auto *next_as_inst =
      reinterpret_cast<const util::IListNode<Instruction, util::IListParent<BasicBlock>> *>(
          next_node);
  if (!next_as_inst->parent_)
    return nullptr;
  return reinterpret_cast<const rj_code_inst_t *>(next_node);
}
