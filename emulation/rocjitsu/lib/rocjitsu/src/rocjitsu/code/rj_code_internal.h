// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/rj_code.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/executable.h"
#include "rocjitsu/code/instruction_list.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/refcount.h"

#include <memory>
#include <vector>

struct rj_code_executable_t : rocjitsu::RefCounted {
  std::shared_ptr<rocjitsu::Executable> exec;
};

struct rj_code_object_t : rocjitsu::RefCounted {
  std::shared_ptr<rocjitsu::AmdGpuCodeObject> co;
};

struct rj_code_inst_list_t : rocjitsu::RefCounted {
  rocjitsu::InstructionList list;
  std::vector<std::unique_ptr<rocjitsu::Instruction>> storage;
};

struct rj_code_basic_block_list_t : rocjitsu::RefCounted {
  using Storage = std::vector<std::unique_ptr<rocjitsu::BasicBlock>>;
  std::shared_ptr<Storage> blocks;
};

struct rj_code_basic_block_t : rocjitsu::RefCounted {
  std::shared_ptr<rocjitsu::BasicBlock> block;
};
