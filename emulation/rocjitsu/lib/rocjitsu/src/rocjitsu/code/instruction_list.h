// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file instruction_list.h
/// @brief Type alias for an intrusive list of instructions within a basic block.

#ifndef ROCJITSU_CODE_INSTRUCTION_LIST_H_
#define ROCJITSU_CODE_INSTRUCTION_LIST_H_

#include "rocjitsu/isa/instruction.h"
#include "util/intrusive_list.h"

namespace rocjitsu {

class BasicBlock;

/// @brief Intrusive list of Instruction nodes owned by a BasicBlock.
using InstructionList = util::IntrusiveList<Instruction, util::IListParent<BasicBlock>>;

} // namespace rocjitsu

#endif // ROCJITSU_CODE_INSTRUCTION_LIST_H_
