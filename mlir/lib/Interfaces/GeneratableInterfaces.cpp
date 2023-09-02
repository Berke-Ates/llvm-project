//===- GeneratableInterfaces.cpp - Generatable interfaces -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "GeneratableInterfaces"
#include "mlir/Interfaces/GeneratableInterfaces.h"
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
  rng = std::mt19937(generatorConfig.seed());
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

bool GeneratorOpBuilder::canCreate(Operation *op) {
  // Enforce nested region limit.
  unsigned depth = 0;
  Block *block = getBlock();

  while (block) {
    depth++;
    // Move up the hierarchy.
    Operation *parent = block->getParentOp();
    block = nullptr;

    if (parent)
      block = parent->getBlock();
  }

  if (depth > generatorConfig.regionDepthLimit()) {
    LLVM_DEBUG(llvm::dbgs()
               << "GeneratorOpBuilder::canCreate reached region depth limit\n");
    return false;
  }

  // Enforce block length limit.
  block = getBlock();
  if (!op->hasTrait<OpTrait::IsTerminator>() && block)
    if (block->getOperations().size() >= generatorConfig.blockLengthLimit())
      return false;

  return true;
}

Operation *GeneratorOpBuilder::create(const OperationState &state) {
  Operation *op = OpBuilder::create(state);

  if (op && canCreate(op))
    return op;

  if (op)
    op->erase();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Collectors
//===----------------------------------------------------------------------===//

llvm::SmallVector<Value>
GeneratorOpBuilder::collectValues(std::function<bool(const Value &)> filter) {
  llvm::SmallVector<Value> possibleValues;
  llvm::SmallVector<Value> excludedValues;
  Block *block = getBlock();

  // FIXME: Only collect values that are defined before the insertion point.
  // (visible)
  while (block) {
    // Add all values in the block.
    for (Operation &op : block->getOperations())
      for (OpResult res : op.getResults())
        if (!filter || filter(res))
          possibleValues.push_back(res);

    // Add all arguments of the block.
    for (BlockArgument bArg : block->getArguments())
      if (!filter || filter(bArg))
        possibleValues.push_back(bArg);

    // Move up the hierarchy.
    Operation *parent = block->getParentOp();
    block = nullptr;

    if (parent && !parent->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
      llvm::append_range(excludedValues, parent->getResults());
      block = parent->getBlock();
    }
  }

  // Remove results of the parents.
  for (Value val : excludedValues)
    llvm::erase_value(possibleValues, val);

  return possibleValues;
}

llvm::SmallVector<Type>
GeneratorOpBuilder::collectTypes(std::function<bool(const Type &)> filter) {
  llvm::DenseSet<Type> possibleTypes;

  for (Value v : collectValues())
    if (!filter || filter(v.getType()))
      possibleTypes.insert(v.getType());

  return llvm::SmallVector<Type>(possibleTypes.begin(), possibleTypes.end());
}

//===----------------------------------------------------------------------===//
// Samplers
//===----------------------------------------------------------------------===//

llvm::Optional<llvm::SmallVector<Value>> GeneratorOpBuilder::sampleValues(
    llvm::SmallVector<std::function<bool(const Value &)>> filters) {
  llvm::SmallVector<Value> possibleValues = {};

  for (auto filter : filters) {
    llvm::SmallVector<Value> values = collectValues(filter);
    if (values.empty())
      return std::nullopt;

    possibleValues.push_back(sample(values).value());
  }

  return possibleValues;
}

llvm::Optional<Value>
GeneratorOpBuilder::sampleValue(std::function<bool(const Value &)> filter) {
  llvm::Optional<llvm::SmallVector<Value>> values = sampleValues({filter});
  if (!values.has_value())
    return std::nullopt;
  return values.value()[0];
}

llvm::Optional<llvm::SmallVector<Value>>
GeneratorOpBuilder::sampleValuesOfTypes(llvm::SmallVector<Type> types) {
  llvm::SmallVector<Value> possibleValues = {};

  for (Type t : types) {
    llvm::SmallVector<Value> values =
        collectValues([t](const Value &v) { return v.getType() == t; });

    if (values.empty())
      return std::nullopt;
    possibleValues.push_back(sample(values).value());
  }

  return possibleValues;
}

llvm::Optional<Value> GeneratorOpBuilder::sampleValueOfType(Type type) {
  llvm::Optional<llvm::SmallVector<Value>> values = sampleValuesOfTypes({type});
  if (!values.has_value())
    return std::nullopt;
  return values.value()[0];
}

llvm::SmallVector<Type>
GeneratorOpBuilder::sampleTypes(unsigned min,
                                std::function<bool(const Type &)> filter) {
  llvm::SmallVector<Type> availableTypes = collectTypes(filter);

  int length = sampleNumber<int>(/*useGeometric=*/true) + min;
  llvm::Optional<llvm::SmallVector<Type>> types =
      sample(availableTypes, length);

  // Failed to sample.
  if (!types.has_value())
    return {};

  return types.value();
}

//===----------------------------------------------------------------------===//
// Generators
//===----------------------------------------------------------------------===//

Operation *GeneratorOpBuilder::generateOperation(
    llvm::SmallVector<RegisteredOperationName> ops) {
  if (ops.empty())
    llvm::errs() << "GeneratorOpBuilder::generateOperation requires a "
                    "non-empty list of operations\n";

  GeneratorOpBuilder::Snapshot snapshot = takeSnapshot();

  while (!ops.empty()) {
    // Rollback first.
    rollback(snapshot);

    // Lookup probabilities.
    llvm::SmallVector<unsigned> probs;
    for (RegisteredOperationName ron : ops)
      probs.push_back(generatorConfig.getProb(ron.getStringRef()));

    // Sample an operation.
    llvm::Optional<RegisteredOperationName> sampledOp = sample(ops, probs);
    if (!sampledOp.has_value()) {
      LLVM_DEBUG(llvm::dbgs() << "GeneratorOpBuilder::generateOperation "
                                 "failed to sample an operation\n");
      return nullptr;
    }

    Operation *op = generate(sampledOp.value());
    if (op) {
      // Set the insertion point after the inserted operation.
      setInsertionPointAfter(op);
      return op;
    }

    llvm::erase_value(ops, sampledOp.value());
  }

  return nullptr;
}

Operation *GeneratorOpBuilder::generateTerminator() {
  // Filter available ops.
  llvm::SmallVector<mlir::RegisteredOperationName> terminatorOps;
  for (RegisteredOperationName op : availableOps)
    if (op.hasTrait<OpTrait::IsTerminator>())
      terminatorOps.push_back(op);

  Operation *op = generateOperation(terminatorOps);
  if (!op) {
    LLVM_DEBUG(llvm::dbgs() << "GeneratorOpBuilder::generateTerminator "
                               "failed to generate terminator\n");
  }

  return op;
}

LogicalResult GeneratorOpBuilder::generateBlock(Block *block,
                                                bool ensureTerminator) {
  if (!block) {
    llvm::errs() << "GeneratorOpBuilder::generateBlock requires a block\n";
    return failure();
  }

  // Guard the insertion point.
  OpBuilder::InsertionGuard guard(*this);

  // Move the insertion point.
  setInsertionPointToStart(block);

  // Filter available ops.
  llvm::SmallVector<mlir::RegisteredOperationName> possibleOps;
  for (RegisteredOperationName op : availableOps)
    if (!op.hasTrait<OpTrait::IsTerminator>())
      possibleOps.push_back(op);

  // Generate operations.
  int length = sampleNumber<int>(/*useGeometric=*/true);
  for (int i = 0; i < length && generateOperation(possibleOps); ++i)
    ;

  // Try to generate terminator.
  setInsertionPointToEnd(block);
  if (ensureTerminator && !generateTerminator()) {
    LLVM_DEBUG(llvm::errs() << "GeneratorOpBuilder::generateBlock failed "
                               "to generate terminator\n");
    return failure();
  }

  return success();
}

Operation *GeneratorOpBuilder::generate(RegisteredOperationName ron) {
  if (!ron.hasInterface<GeneratableOpInterface>()) {
    llvm::errs() << ron << "does not implement GeneratableOpInterface";
    return nullptr;
  }
  return ron.getInterface<GeneratableOpInterface>()->generate(*this);
}

//===----------------------------------------------------------------------===//
// Table-generated class definitions
//===----------------------------------------------------------------------===//

/// Include the definitions of the generatable interfaces.
#include "mlir/Interfaces/GeneratableInterfaces.cpp.inc"
