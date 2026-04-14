// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"

using namespace mlir;
using namespace waveasm;

// typesCompatible() is defined in WaveASMTypes.h as a shared utility.

//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//

void LoopOp::build(OpBuilder &builder, OperationState &result,
                   ValueRange initArgs) {
  result.addTypes(initArgs.getTypes());
  result.addOperands(initArgs);

  // Create body region with block arguments matching initArgs types
  Region *bodyRegion = result.addRegion();
  Block *bodyBlock = new Block();
  bodyRegion->push_back(bodyBlock);

  for (Value arg : initArgs) {
    bodyBlock->addArgument(arg.getType(), result.location);
  }
}

void LoopOp::getSuccessorRegions(RegionBranchPoint point,
                                 SmallVectorImpl<RegionSuccessor> &regions) {

  // Entry: branch to body region
  if (point.isParent()) {
    regions.emplace_back(&getBody());
    return;
  }

  // From body: can loop back to body or exit to parent
  regions.emplace_back(&getBody());
  regions.emplace_back(getResults());
}

OperandRange LoopOp::getEntrySuccessorOperands(RegionBranchPoint point) {
  return getInitArgs();
}

bool LoopOp::areTypesCompatible(Type lhs, Type rhs) {
  return typesCompatible(lhs, rhs);
}

// Old API compatibility wrapper (required by this MLIR version)

Block &LoopOp::getBodyBlock() { return getBody().front(); }

Block::BlockArgListType LoopOp::getRegionIterArgs() {
  return getBodyBlock().getArguments();
}

unsigned LoopOp::getNumRegionIterArgs() {
  return getBodyBlock().getNumArguments();
}

SmallVector<Region *> LoopOp::getLoopRegions() { return {&getBody()}; }

// LoopLikeOpInterface methods
MutableArrayRef<OpOperand> LoopOp::getInitsMutable() {
  return getInitArgsMutable();
}

std::optional<MutableArrayRef<OpOperand>> LoopOp::getYieldedValuesMutable() {
  auto *term = getBodyBlock().getTerminator();
  if (auto condOp = dyn_cast<ConditionOp>(term))
    return condOp.getIterArgsMutable();
  return std::nullopt;
}

LogicalResult LoopOp::verify() {
  // SingleBlockImplicitTerminator<ConditionOp> already ensures the body
  // has exactly one block terminated by ConditionOp.
  auto condOp = cast<ConditionOp>(getBodyBlock().getTerminator());

  // Verify initArgs match block arguments
  if (getInitArgs().size() != getNumRegionIterArgs())
    return emitOpError("different number of inits (")
           << getInitArgs().size() << ") and region iter_args ("
           << getNumRegionIterArgs() << ")";

  // Verify condition iter_args are compatible with block arguments
  if (!typesCompatible(getBodyBlock().getArgumentTypes(),
                       condOp.getIterArgs().getTypes()))
    return emitOpError(
        "condition iter_arg types must be compatible with loop arg types");

  // Verify result types are compatible with iter arg types
  if (!typesCompatible(getResultTypes(), getBodyBlock().getArgumentTypes()))
    return emitOpError("result types must be compatible with iter arg types");

  return success();
}

void LoopOp::print(OpAsmPrinter &p) {
  // Print (%i = %init, %acc = %init_acc) or () if empty
  p << " (";
  if (getNumRegionIterArgs() > 0) {
    llvm::interleaveComma(
        llvm::zip(getRegionIterArgs(), getInitArgs()), p,
        [&](auto it) { p << std::get<0>(it) << " = " << std::get<1>(it); });
  }
  p << ")";

  // Print : (input_types) -> (result_types)
  p << " : ";
  auto inputTypes = getInitArgs().getTypes();
  auto resultTypes = getResults().getTypes();
  p.printFunctionalType(inputTypes, resultTypes);

  // Print body region
  p << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);

  p.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult LoopOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse: loop (%i = %init, %acc = %init_acc) : (types) -> (types) { ... }

  SmallVector<OpAsmParser::Argument> regionArgs;
  SmallVector<OpAsmParser::UnresolvedOperand> initOperands;

  // Parse optional (%arg = %init, ...)
  OptionalParseResult listResult =
      parser.parseOptionalAssignmentList(regionArgs, initOperands);
  if (listResult.has_value() && failed(listResult.value()))
    return failure();

  // Parse : (arg_types) -> (result_types) as function type
  FunctionType functionType;
  SMLoc typeLoc = parser.getCurrentLocation();
  if (failed(parser.parseColonType(functionType)))
    return failure();

  // Add result types
  result.addTypes(functionType.getResults());

  // Check operand count matches input types
  if (functionType.getNumInputs() != initOperands.size()) {
    return parser.emitError(typeLoc)
           << "expected as many input types as operands (expected "
           << initOperands.size() << " got " << functionType.getNumInputs()
           << ")";
  }

  // Resolve init operands
  if (failed(parser.resolveOperands(initOperands, functionType.getInputs(),
                                    parser.getCurrentLocation(),
                                    result.operands)))
    return failure();

  // Set argument types for region
  for (size_t i = 0, e = regionArgs.size(); i != e; ++i)
    regionArgs[i].type = functionType.getInput(i);

  // Parse body region
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs))
    return failure();

  // Parse optional attributes
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

Block &IfOp::getThenBlock() { return getThenRegion().front(); }

Block *IfOp::getElseBlock() {
  Region &r = getElseRegion();
  return r.empty() ? nullptr : &r.front();
}

bool IfOp::hasElse() { return !getElseRegion().empty(); }

void IfOp::build(OpBuilder &builder, OperationState &result, Value condition,
                 bool withElseRegion) {
  result.addOperands(condition);

  Region *thenRegion = result.addRegion();
  thenRegion->push_back(new Block());

  Region *elseRegion = result.addRegion();
  if (withElseRegion) {
    elseRegion->push_back(new Block());
  }
}

void IfOp::build(OpBuilder &builder, OperationState &result,
                 TypeRange resultTypes, Value condition, bool withElseRegion) {
  result.addTypes(resultTypes);
  result.addOperands(condition);

  Region *thenRegion = result.addRegion();
  thenRegion->push_back(new Block());

  Region *elseRegion = result.addRegion();
  if (withElseRegion) {
    elseRegion->push_back(new Block());
  }
}

void IfOp::getSuccessorRegions(RegionBranchPoint point,
                               SmallVectorImpl<RegionSuccessor> &regions) {

  // Entry: can go to then or else
  if (point.isParent()) {
    regions.push_back(RegionSuccessor(&getThenRegion()));
    if (hasElse())
      regions.push_back(RegionSuccessor(&getElseRegion()));
    else
      regions.emplace_back(getResults());
    return;
  }

  // Both then and else exit to parent
  regions.emplace_back(getResults());
}

bool IfOp::areTypesCompatible(Type lhs, Type rhs) {
  return typesCompatible(lhs, rhs);
}

// Old API compatibility wrapper (required by this MLIR version)

LogicalResult IfOp::verify() {
  // SingleBlockImplicitTerminator<YieldOp> already ensures the then region
  // has exactly one block terminated by YieldOp.
  auto thenYield = cast<YieldOp>(getThenBlock().getTerminator());

  if (hasElse()) {
    // SingleBlockImplicitTerminator also covers the else region.
    auto elseYield = cast<YieldOp>(getElseBlock()->getTerminator());

    if (!typesCompatible(thenYield.getResults().getTypes(),
                         elseYield.getResults().getTypes()))
      return emitOpError("then and else must yield compatible types");
  }

  // If the op produces results, an else region MUST exist to provide values
  // when the condition is false. Without an else, result values are undefined.
  if (!getResultTypes().empty() && !hasElse()) {
    return emitOpError("if operation with results must have an else region");
  }

  // Verify result types are compatible with yields
  if (!typesCompatible(getResultTypes(), thenYield.getResults().getTypes()))
    return emitOpError("result types must be compatible with yield types");

  return success();
}

void IfOp::print(OpAsmPrinter &p) {
  p << " " << getCondition();

  // Print result types
  if (!getResultTypes().empty()) {
    p << " : " << getCondition().getType() << " -> ";
    llvm::interleaveComma(getResultTypes(), p);
  }

  // Print then region
  p << " ";
  p.printRegion(getThenRegion(), /*printEntryBlockArgs=*/false);

  // Print else region if present
  if (hasElse()) {
    p << " else ";
    p.printRegion(getElseRegion(), /*printEntryBlockArgs=*/false);
  }

  p.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult IfOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse: if %cond : type -> types { then } else { else }

  OpAsmParser::UnresolvedOperand condition;
  Type condType;

  if (parser.parseOperand(condition))
    return failure();

  // Parse optional types
  SmallVector<Type> resultTypes;
  if (succeeded(parser.parseOptionalColon())) {
    if (parser.parseType(condType))
      return failure();

    if (succeeded(parser.parseOptionalArrow())) {
      if (parser.parseTypeList(resultTypes))
        return failure();
    }
  } else {
    // Default condition type
    condType = SCCType::get(parser.getContext());
  }

  // Resolve condition
  if (parser.resolveOperand(condition, condType, result.operands))
    return failure();

  result.addTypes(resultTypes);

  // Parse then region
  Region *thenRegion = result.addRegion();
  if (parser.parseRegion(*thenRegion))
    return failure();

  // Ensure terminator exists
  if (thenRegion->empty())
    thenRegion->push_back(new Block());

  // Parse optional else region
  Region *elseRegion = result.addRegion();
  if (succeeded(parser.parseOptionalKeyword("else"))) {
    if (parser.parseRegion(*elseRegion))
      return failure();

    if (elseRegion->empty())
      elseRegion->push_back(new Block());
  }

  // Parse optional attributes
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// ConditionOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ConditionOp - RegionBranchTerminatorOpInterface
//===----------------------------------------------------------------------===//

MutableOperandRange
ConditionOp::getMutableSuccessorOperands(RegionBranchPoint point) {
  return getIterArgsMutable();
}

void ConditionOp::getSuccessorRegions(
    ArrayRef<Attribute> operands, SmallVectorImpl<RegionSuccessor> &regions) {
  auto loopOp = cast<LoopOp>((*this)->getParentOp());
  regions.emplace_back(&loopOp.getBody());
  regions.emplace_back(loopOp.getResults());
}

LogicalResult ConditionOp::verify() {
  auto loopOp = dyn_cast<LoopOp>((*this)->getParentOp());
  if (!loopOp) {
    return emitOpError("must be nested in a waveasm.loop operation");
  }

  // Verify iter_args count matches loop args
  if (getIterArgs().size() != loopOp.getNumRegionIterArgs()) {
    return emitOpError("iter_args count must match loop args");
  }

  // Verify types match (structurally, allowing physical register differences)
  if (!typesCompatible(loopOp.getBodyBlock().getArgumentTypes(),
                       getIterArgs().getTypes()))
    return emitOpError("iter_arg types must be compatible with loop arg types");

  return success();
}
