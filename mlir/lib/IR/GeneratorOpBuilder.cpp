//===- GeneratorOpBuilder.cpp - MLIR Program Generator --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "GeneratorOpBuilder"
#include "mlir/IR/GeneratorOpBuilder.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
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
// GeneratorOpBuilder::Config
//===----------------------------------------------------------------------===//

void GeneratorOpBuilder::Config::registerOpConfigs() {
  for (RegisteredOperationName ron : context->getRegisteredOperations())
    if (ron.hasInterface<GeneratableOpInterface>())
      ron.getInterface<GeneratableOpInterface>()->registerConfigs(*this);
}

//===----------------------------------------------------------------------===//
// GeneratorOpBuilder
//===----------------------------------------------------------------------===//

GeneratorOpBuilder::GeneratorOpBuilder(Config config)
    : OpBuilder(config.getContext()), config(config) {
  // Freeze configuration.
  config.freeze();

  // Collect all operations usable for generation.
  for (RegisteredOperationName ron :
       config.getContext()->getRegisteredOperations())
    if (ron.hasInterface<GeneratableOpInterface>() && getProb(ron) > 0)
      availableOps.push_back(ron);

  // Setup random number generator.
  rng = std::mt19937(config.get<unsigned>("_gen.seed").value());
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

  if (depth > config.get<unsigned>("_gen.regionDepthLimit").value()) {
    LLVM_DEBUG(llvm::dbgs()
               << "GeneratorOpBuilder::canCreate reached region depth limit\n");
    return false;
  }

  // Enforce block length limit.
  block = getBlock();
  if (!op->hasTrait<OpTrait::IsTerminator>() && block)
    if (block->getOperations().size() >=
        config.get<unsigned>("_gen.blockLengthLimit").value())
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

llvm::SmallVector<Location> GeneratorOpBuilder::getUnknownLocs(unsigned num) {
  llvm::SmallVector<Location> locs;
  for (unsigned i = 0; i < num; ++i)
    locs.push_back(getUnknownLoc());
  return locs;
}

Operation *
GeneratorOpBuilder::addResultTypes(Operation *op,
                                   llvm::ArrayRef<Type> resultTypes) {
  if (!op || op->getNumResults() > 0)
    return nullptr;

  // Guard the insertion point.
  OpBuilder::InsertionGuard guard(*this);
  OpBuilder::setInsertionPoint(op);

  OperationState state(op->getLoc(), op->getName());
  state.addTypes(resultTypes);
  state.addAttributes(op->getAttrs());
  state.addOperands(op->getOperands());
  for (unsigned i = 0; i < op->getNumRegions(); ++i)
    state.addRegion()->takeBody(op->getRegion(i));

  return OpBuilder::create(state);
}

//===----------------------------------------------------------------------===//
// Collectors
//===----------------------------------------------------------------------===//

llvm::SmallVector<Value>
GeneratorOpBuilder::collectValues(std::function<bool(const Value &)> filter) {
  llvm::SmallVector<Value> possibleValues;
  llvm::SmallVector<Value> excludedValues;
  Block *block = getBlock();

  // FIXME: Only collect values that are defined above the insertion point.
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

llvm::SmallVector<llvm::StringRef> GeneratorOpBuilder::collectSymbols(
    std::function<bool(const Operation &, const llvm::StringRef &)> filter) {
  // FIXME: Extend this simplified symbol search with parents/children.
  llvm::SmallVector<llvm::StringRef> possibleSymbols;
  llvm::SmallVector<llvm::StringRef> excludedSymbols;
  Block *block = getBlock();

  while (block) {
    // Add all symbols in the block.
    for (Operation &op : block->getOperations())
      if (op.hasAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
        if (!filter || filter(op, SymbolTable::getSymbolName(&op)))
          possibleSymbols.push_back(SymbolTable::getSymbolName(&op));

    // Move up the hierarchy.
    Operation *parent = block->getParentOp();
    block = nullptr;

    if (parent && !parent->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
      if (parent->hasAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
        excludedSymbols.push_back(SymbolTable::getSymbolName(parent));
      block = parent->getBlock();
    }
  }

  // Remove results of the parents.
  for (llvm::StringRef sym : excludedSymbols)
    llvm::erase_value(possibleSymbols, sym);

  return possibleSymbols;
}

//===----------------------------------------------------------------------===//
// Samplers
//===----------------------------------------------------------------------===//

std::string GeneratorOpBuilder::sampleString() {
  char alphas[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
  char nums[] = "0123456789";
  llvm::SmallVector<char> charset;
  for (char c : alphas)
    charset.push_back(c);

  unsigned length = sampleGeometric<unsigned>() + 1;
  std::string result;
  result.reserve(length);
  result += sample(charset).value();

  for (char c : nums)
    charset.push_back(c);

  for (unsigned i = 1; i < length; ++i)
    result += sample(charset).value();

  return result;
}

llvm::Optional<llvm::SmallVector<Value>> GeneratorOpBuilder::sampleValues(
    llvm::SmallVector<std::function<bool(const Value &)>> filters,
    bool unusedFirst) {
  llvm::SmallVector<Value> possibleValues = {};

  for (auto filter : filters) {
    llvm::SmallVector<Value> valuesUnused = collectValues(
        [&](const Value &v) { return filter(v) && v.getUses().empty(); });
    llvm::SmallVector<Value> valuesAll = collectValues(filter);

    if (unusedFirst)
      if (!valuesUnused.empty()) {
        possibleValues.push_back(sample(valuesUnused).value());
        continue;
      }

    if (valuesAll.empty())
      return std::nullopt;

    possibleValues.push_back(sample(valuesAll).value());
  }

  return possibleValues;
}

llvm::Optional<Value>
GeneratorOpBuilder::sampleValue(std::function<bool(const Value &)> filter,
                                bool unusedFirst) {
  llvm::Optional<llvm::SmallVector<Value>> values =
      sampleValues({filter}, unusedFirst);
  if (!values.has_value())
    return std::nullopt;
  return values.value()[0];
}

llvm::Optional<llvm::SmallVector<Value>>
GeneratorOpBuilder::sampleValuesOfTypes(llvm::SmallVector<Type> types,
                                        bool unusedFirst) {
  llvm::SmallVector<Value> possibleValues = {};

  for (Type t : types) {
    llvm::SmallVector<Value> valuesUnused = collectValues([t](const Value &v) {
      return v.getType() == t && v.getUses().empty();
    });

    llvm::SmallVector<Value> valuesAll =
        collectValues([t](const Value &v) { return v.getType() == t; });

    if (unusedFirst)
      if (!valuesUnused.empty()) {
        possibleValues.push_back(sample(valuesUnused).value());
        continue;
      }

    if (valuesAll.empty())
      return std::nullopt;
    possibleValues.push_back(sample(valuesAll).value());
  }

  return possibleValues;
}

llvm::Optional<Value> GeneratorOpBuilder::sampleValueOfType(Type type,
                                                            bool unusedFirst) {
  llvm::Optional<llvm::SmallVector<Value>> values =
      sampleValuesOfTypes({type}, unusedFirst);
  if (!values.has_value())
    return std::nullopt;
  return values.value()[0];
}

llvm::SmallVector<Value>
GeneratorOpBuilder::sampleValues(unsigned min,
                                 std::function<bool(const Value &)> filter) {
  llvm::SmallVector<Value> availableValues = collectValues(filter);

  unsigned length = sampleGeometric<unsigned>() + min;
  llvm::Optional<llvm::SmallVector<Value>> values =
      sample(availableValues, length);

  // Failed to sample.
  if (!values.has_value())
    return {};

  return values.value();
}

llvm::SmallVector<Type>
GeneratorOpBuilder::sampleTypes(unsigned min,
                                std::function<bool(const Type &)> filter) {
  llvm::SmallVector<Type> availableTypes = collectTypes(filter);

  unsigned length = sampleGeometric<unsigned>() + min;
  llvm::Optional<llvm::SmallVector<Type>> types =
      sample(availableTypes, length);

  // Failed to sample.
  if (!types.has_value())
    return {};

  return types.value();
}

llvm::Optional<llvm::StringRef> GeneratorOpBuilder::sampleSymbol(
    std::function<bool(const Operation &, const llvm::StringRef &)> filter) {
  llvm::SmallVector<llvm::StringRef> availableSymbols = collectSymbols(filter);
  if (availableSymbols.empty())
    return std::nullopt;

  return sample(availableSymbols);
}

//===----------------------------------------------------------------------===//
// Generators
//===----------------------------------------------------------------------===//

Operation *GeneratorOpBuilder::generate(RegisteredOperationName ron) {
  if (!ron.hasInterface<GeneratableOpInterface>()) {
    llvm::errs() << ron << "does not implement GeneratableOpInterface";
    return nullptr;
  }
  return ron.getInterface<GeneratableOpInterface>()->generate(*this);
}

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
      probs.push_back(getProb(ron));

    // Sample an operation.
    llvm::Optional<RegisteredOperationName> sampledOp = sample(ops, probs);
    if (!sampledOp.has_value()) {
      LLVM_DEBUG(llvm::dbgs() << "GeneratorOpBuilder::generateOperation "
                                 "failed to sample an operation\n");
      return nullptr;
    }

    Operation *op = generate(sampledOp.value());
    if (!op) {
      llvm::erase_value(ops, sampledOp.value());
      continue;
    }

    // Silence diagnostics for verification.
    DiagnosticEngine::HandlerID handlerID =
        context->getDiagEngine().registerHandler([](Diagnostic &diag) {});
    LogicalResult verified = verify(op);
    context->getDiagEngine().eraseHandler(handlerID);

    if (verified.succeeded()) {
      OpBuilder::setInsertionPointAfter(op);
      return op;
    }

    // Abort further generation attemps as verification failure indicates bugs
    // in the Op generation functions.
    return nullptr;
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
  if (!op)
    LLVM_DEBUG(llvm::dbgs() << "GeneratorOpBuilder::generateTerminator "
                               "failed to generate terminator\n");

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
  OpBuilder::setInsertionPointToStart(block);

  // Filter available ops.
  llvm::SmallVector<mlir::RegisteredOperationName> possibleOps;
  for (RegisteredOperationName op : availableOps)
    if (!op.hasTrait<OpTrait::IsTerminator>())
      possibleOps.push_back(op);

  // Generate operations.
  unsigned length = sampleGeometric<unsigned>();
  for (unsigned i = 0; i < length; ++i)
    if (!generateOperation(possibleOps))
      break;

  // Try to generate terminator.
  OpBuilder::setInsertionPointToEnd(block);
  if (ensureTerminator && !generateTerminator()) {
    LLVM_DEBUG(llvm::errs() << "GeneratorOpBuilder::generateBlock failed "
                               "to generate terminator\n");
    return failure();
  }

  return success();
}

unsigned GeneratorOpBuilder::getProb(RegisteredOperationName ron) {
  llvm::Optional<int> prob = config.get<int>(ron.getStringRef());
  if (!prob.has_value() || prob.value() < 0)
    return config.get<unsigned>("_gen.defaultProb").value();

  return prob.value();
}

//===----------------------------------------------------------------------===//
// Table-generated class definitions
//===----------------------------------------------------------------------===//

/// Include the definitions of the generatable interfaces.
#include "mlir/Interfaces/GeneratableOpInterface.cpp.inc"
#include "mlir/Interfaces/GeneratableTypeInterface.cpp.inc"
