//===- GeneratableOpInterface.cpp - Generatable operations interface ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "GeneratableOpInterface"
#include "mlir/Interfaces/GeneratableOpInterface.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Signals.h"
#include <random>
#include <type_traits>

using namespace mlir;

//===----------------------------------------------------------------------===//
// Debug Utils
//===----------------------------------------------------------------------===//

std::string getImmediateCaller() {
  std::string str;
  llvm::raw_string_ostream os(str);
  llvm::sys::PrintStackTrace(os, 4);
  os.flush();

  std::istringstream iss(str);
  std::string line;
  std::string column;
  int lineNumber = 0;
  int columnNumber = 0;

  while (std::getline(iss, line)) {
    lineNumber++;
    if (lineNumber == 4) {
      std::istringstream iss2(line);
      while (std::getline(iss2, column, ' ')) {
        columnNumber++;
        if (columnNumber == 3) {
          return column;
        }
      }
    }
  }

  return "";
}

//===----------------------------------------------------------------------===//
// GeneratorOpBuilder
//===----------------------------------------------------------------------===//

GeneratorOpBuilder::GeneratorOpBuilder(MLIRContext *ctx,
                                       GeneratorOpBuilderConfig generatorConfig)
    : OpBuilder(ctx) {
  // Collect all operations usable for generation.
  for (RegisteredOperationName ron : ctx->getRegisteredOperations())
    if (ron.hasInterface<GeneratableOpInterface>() &&
        generatorConfig.getProb(ron.getStringRef()) > 0)
      availableOps.push_back(ron);

  // Setup random number generator.
  rngGen = std::mt19937(generatorConfig.seed());
}

//===----------------------------------------------------------------------===//
// State Management
//===----------------------------------------------------------------------===//

GeneratorOpBuilder::Snapshot GeneratorOpBuilder::takeSnapshot() {
  // TODO: Take a snapshot of the IR.
  return GeneratorOpBuilder::Snapshot(saveInsertionPoint());
}

void GeneratorOpBuilder::rollback(GeneratorOpBuilder::Snapshot snapshot) {
  // TODO: Restore the snapshot.
  restoreInsertionPoint(snapshot.getInsertionPoint());
}

//===----------------------------------------------------------------------===//
// Operation Creation
//===----------------------------------------------------------------------===//

bool GeneratorOpBuilder::canCreate(const OperationState &state) {
  // Enforce return type if needed.
  if (requiredReturnType.has_value())
    if (llvm::find(state.types, requiredReturnType.value()) ==
        state.types.end())
      return false;

  // Enforce nested region limit.
  unsigned depth = 0;
  Block *block = getBlock();

  while (block != nullptr) {
    depth++;
    // Move up the hierarchy.
    Operation *parent = block->getParentOp();
    block = nullptr;

    if (parent != nullptr)
      block = parent->getBlock();
  }

  if (depth > generatorConfig.regionDepthLimit()) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "GeneratorOpBuilderImpl::canCreate reached region depth limit\n");
    return false;
  }

  // Enforce block length limit.
  block = getBlock();
  if (!bypassBlockLengthLimit && block != nullptr)
    if (block->getOperations().size() >= generatorConfig.blockLengthLimit())
      return false;

  return true;
}

Operation *GeneratorOpBuilder::create(const OperationState &state) {
  if (canCreate(state))
    return OpBuilder::create(state);
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Samplers
//===----------------------------------------------------------------------===//

unsigned GeneratorOpBuilder::sampleUniform(int32_t max) {
  if (max < 0)
    llvm::errs()
        << "upper bound of 'sampleUniform(int32_t max)' must be positive\n";
  std::uniform_int_distribution<unsigned> dist(0, max);
  return dist(rngGen);
}

bool GeneratorOpBuilder::sampleBool() {
  // 0.5 is the probability for generating true.
  std::bernoulli_distribution dist(0.5);
  return dist(rngGen);
}

llvm::SmallVector<Type> GeneratorOpBuilder::sampleTypes(int32_t min) {
  if (min < 0)
    llvm::errs()
        << "lower bound of 'sampleTypes(int32_t min)' must be positive\n";

  llvm::SmallVector<Type> availableTypes;
  Block *block = getBlock();

  while (block != nullptr) {
    // Add all types of operations in the block.
    for (Operation &op : block->getOperations())
      for (OpResult res : op.getResults())
        if (llvm::find(availableTypes, res.getType()) == availableTypes.end())
          availableTypes.push_back(res.getType());

    // Add all types of block arguments.
    for (BlockArgument bArg : block->getArguments())
      if (llvm::find(availableTypes, bArg.getType()) == availableTypes.end())
        availableTypes.push_back(bArg.getType());

    // Move up the hierarchy.
    Operation *parent = block->getParentOp();
    block = nullptr;

    if (parent != nullptr && !parent->hasTrait<OpTrait::IsIsolatedFromAbove>())
      block = parent->getBlock();
  }

  // Nothing to sample from.
  if (availableTypes.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "GeneratorOpBuilderImpl::sampleTypes has no "
                               "types to sample from");
    return {};
  }

  // Geometric distribution, p=0.2
  std::geometric_distribution<> dist(0.2);
  int length = dist(rngGen) + min;

  llvm::SmallVector<Type> types;
  for (int i = 0; i < length; ++i) {
    llvm::Optional<Type> type = sample(availableTypes);

    // Failed to sample.
    if (!type.has_value()) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "GeneratorOpBuilderImpl::sampleTypes failed to sample a type");
      return {};
    }

    types.push_back(type.value());
  }

  LLVM_DEBUG(llvm::dbgs() << "GeneratorOpBuilderImpl::sampleTypes generated {"
                          << types << "}\n");
  return types;
}

llvm::Optional<Value> GeneratorOpBuilder::sampleValueOfType(Type t) {
  llvm::SmallVector<Value> possibleValues;
  llvm::SmallVector<Value> excludedValues;

  Block *block = getBlock();

  while (block != nullptr) {
    // Add all operations in the block of type t.
    for (Operation &op : block->getOperations())
      for (OpResult res : op.getResults())
        if (res.getType() == t)
          possibleValues.push_back(res);

    // Add all arguments of the block of type t.
    for (BlockArgument bArg : block->getArguments())
      if (bArg.getType() == t)
        possibleValues.push_back(bArg);

    // Move up the hierarchy.
    Operation *parent = block->getParentOp();
    block = nullptr;

    if (parent != nullptr &&
        !parent->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
      for (OpResult res : parent->getResults())
        excludedValues.push_back(res);

      block = parent->getBlock();
    }
  }

  for (Value val : excludedValues) {
    Value *it = llvm::find(possibleValues, val);
    if (it != possibleValues.end())
      possibleValues.erase(it);
  }

  if (possibleValues.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "GeneratorOpBuilderImpl::sampleValueOfType "
                               "failed to find possible values of type "
                            << t << "\n");
    return std::nullopt;
  }

  llvm::Optional<Value> value = sample(possibleValues);
  if (!value.has_value()) {
    LLVM_DEBUG(llvm::dbgs() << "GeneratorOpBuilderImpl::sampleValueOfType "
                               "failed to sample a value of type "
                            << t << "\n");
    return std::nullopt;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "GeneratorOpBuilderImpl::sampleValueOfType sampled value ";
    if (Operation *op = value.value().getDefiningOp())
      llvm::dbgs() << op->getName();
    else
      llvm::dbgs() << value.value();
    llvm::dbgs() << " of type " << t << "\n";
  });

  return value;
}

//===----------------------------------------------------------------------===//
// Generators
//===----------------------------------------------------------------------===//

LogicalResult GeneratorOpBuilder::generateOperation(
    llvm::SmallVector<RegisteredOperationName> ops) {
  LLVM_DEBUG(
      llvm::dbgs()
      << "GeneratorOpBuilderImpl::generateOperation started generating\n");

  Block *block = getBlock();
  if (block == nullptr) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "GeneratorOpBuilderImpl::generateOperation requires a block\n");
    return failure();
  }

  if (ops.empty())
    ops = availableOps;

  LogicalResult logicalResult = failure();
  llvm::Optional<RegisteredOperationName> op;
  GeneratorOpBuilder::Snapshot snapshot = takeSnapshot();

  while (logicalResult.failed()) {
    // Rollback first.
    rollback(snapshot);

    // Lookup probabilities.
    llvm::SmallVector<unsigned> probs;
    for (RegisteredOperationName op : ops)
      probs.push_back(generatorConfig.getProb(op.getStringRef()));

    op = sample(ops, probs);
    if (!op.has_value()) {
      LLVM_DEBUG(llvm::dbgs() << "GeneratorOpBuilderImpl::generateOperation "
                                 "failed to sample an operation\n");
      return failure();
    }

    LLVM_DEBUG(
        llvm::dbgs()
        << "GeneratorOpBuilderImpl::generateOperation trying to generate "
        << op.value() << "\n");

    logicalResult = generate(op.value());

    RegisteredOperationName *it = llvm::find(ops, op);
    if (it != ops.end())
      ops.erase(it);
  }

  LLVM_DEBUG(
      llvm::dbgs()
      << "GeneratorOpBuilderImpl::generateOperation successfully generated "
      << op.value() << "\n");

  return success();
}

LogicalResult GeneratorOpBuilder::generateTerminator() {
  LLVM_DEBUG(
      llvm::dbgs()
      << "GeneratorOpBuilderImpl::generateTerminator started generating\n");

  Block *block = getBlock();
  if (block == nullptr) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "GeneratorOpBuilderImpl::generateTerminator requires a block\n");
    return failure();
  }

  // Filter available ops.
  llvm::SmallVector<mlir::RegisteredOperationName> terminatorOps;
  for (RegisteredOperationName op : availableOps)
    if (op.hasTrait<OpTrait::IsTerminator>())
      terminatorOps.push_back(op);

  if (terminatorOps.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "GeneratorOpBuilderImpl::generateTerminator requires "
                  "at least one terminator\n");
    return failure();
  }

  if (generateOperation(terminatorOps).failed()) {
    LLVM_DEBUG(llvm::dbgs() << "GeneratorOpBuilderImpl::generateTerminator "
                               "failed to generate terminator\n");
    return failure();
  }

  return success();
}

LogicalResult GeneratorOpBuilder::generateBlock(bool ensureTerminator) {
  LLVM_DEBUG(llvm::dbgs()
             << "GeneratorOpBuilderImpl::generateBlock started generating\n");

  Block *block = getBlock();
  if (block == nullptr) {
    LLVM_DEBUG(llvm::dbgs()
               << "GeneratorOpBuilderImpl::generateBlock requires a block\n");
    return failure();
  }

  // Randomly generate operations.
  // Geometric distribution, p=0.2
  std::geometric_distribution<> dist(0.2);
  int length = dist(rngGen);

  // Filter available ops.
  llvm::SmallVector<mlir::RegisteredOperationName> possibleOps;
  for (RegisteredOperationName op : availableOps)
    if (!op.hasTrait<OpTrait::IsTerminator>())
      possibleOps.push_back(op);

  for (int i = 0; i < length && generateOperation(possibleOps).succeeded();
       ++i) {
  }

  LLVM_DEBUG(llvm::dbgs() << "GeneratorOpBuilderImpl::generateBlock "
                             "successfully generated additional operations\n");

  bypassBlockLengthLimit = true;

  // If required, continue until a terminator has been generated.
  if (!ensureTerminator) {
    bypassBlockLengthLimit = false;
    Operation *parent = block->getParentOp();
    if (parent != nullptr)
      setInsertionPointAfter(parent);
    else
      LLVM_DEBUG(llvm::dbgs()
                 << "GeneratorOpBuilderImpl::generateBlock has no parent\n");
    LLVM_DEBUG(llvm::dbgs()
               << "GeneratorOpBuilderImpl::generateBlock "
                  "successfully generated block without terminator\n");
    return success();
  }

  // Try to generate terminator.
  if (generateTerminator().failed()) {
    LLVM_DEBUG(llvm::dbgs() << "GeneratorOpBuilderImpl::generateBlock failed "
                               "to generate terminator\n");
    return failure();
  }

  bypassBlockLengthLimit = false;
  Operation *parent = block->getParentOp();
  if (parent != nullptr)
    setInsertionPointAfter(parent);
  else
    LLVM_DEBUG(llvm::dbgs()
               << "GeneratorOpBuilderImpl::generateBlock has no parent\n");
  LLVM_DEBUG(llvm::dbgs() << "GeneratorOpBuilderImpl::generateBlock "
                             "successfully generated block with terminator\n");
  return success();
}

LogicalResult GeneratorOpBuilder::generate(RegisteredOperationName ron) {
  if (!ron.hasInterface<GeneratableOpInterface>()) {
    llvm::errs() << ron << "does not implement GeneratableOpInterface";
    return failure();
  }
  return ron.getInterface<GeneratableOpInterface>()->generate(*this);
}

//===----------------------------------------------------------------------===//
// Table-generated class definitions
//===----------------------------------------------------------------------===//

/// Include the definitions of the copy operation interface.
#include "mlir/Interfaces/GeneratableOpInterface.cpp.inc"
