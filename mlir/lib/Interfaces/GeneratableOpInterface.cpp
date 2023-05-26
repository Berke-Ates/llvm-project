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
using namespace mlir::detail;

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
// GeneratorOpBuilderImpl
//===----------------------------------------------------------------------===//

namespace mlir {
namespace detail {
struct GeneratorOpBuilderImpl {
  // TODO: GeneratorOpBuilder should track generated op stack for termination.

  explicit GeneratorOpBuilderImpl(MLIRContext *ctx,
                                  GeneratorOpBuilderConfig generatorConfig,
                                  GeneratorOpBuilder &builder)
      : generatorConfig(generatorConfig), builder(builder) {
    // Collect all operations usable for generation.
    for (RegisteredOperationName ron : ctx->getRegisteredOperations())
      if (ron.hasInterface<GeneratableOpInterface>() &&
          generatorConfig.getProb(ron.getStringRef()) > 0)
        availableOps.push_back(ron);

    // Setup random number generator.
    rngGen = std::mt19937(generatorConfig.seed());
  }

  /// Checks if the given OperationState adheres to the generator constraints.
  bool canCreate(const OperationState &state);

  /// Returns a random number between 0 and max (inclusive) using uniform
  /// distribution.
  unsigned sampleUniform(int32_t max);

  /// Samples from a vector of choices. If no probability vector is provided,
  /// it samples using a uniform distribution.
  template <typename T>
  llvm::Optional<T> sample(llvm::SmallVector<T> choices,
                           llvm::SmallVector<unsigned> probs = {});

  /// Returns a random boolean.
  bool sampleBool();

  /// Returns a random number using a normal distribution around zero. This
  /// function combines all specialized functions.
  template <typename T>
  T sampleNumber();

  /// Randomly generates an operation in the current position. Returns the
  /// generated Values if successful.
  llvm::Optional<llvm::SmallVector<Value>>
  generateOperation(llvm::SmallVector<RegisteredOperationName> ops = {});

  /// Randomly generates a terminator operation in the current position.
  /// Returns the generated Values if successful.
  llvm::Optional<llvm::SmallVector<Value>> generateTerminator();

  /// Samples from a geometric distribution of available types in the current
  /// position.
  llvm::SmallVector<Type> sampleTypes(int32_t min = 0);

  /// Checks if a value of the given type is available in the current
  /// position.
  bool hasValueOfType(Type t);

  /// Randomly chooses a generated value of the given type, if one exists.
  llvm::Optional<Value> sampleValueOfType(Type t);

  /// Randomly generates an operation with the given return type, if possible.
  llvm::Optional<Value> generateValueOfType(Type t);

  /// Randomly tries to chooses a generated value of the given type, if one
  /// exists. If this fails, randomly generates an operation with the given
  /// return type, if possible.
  llvm::Optional<Value> sampleOrGenerateValueOfType(Type t);

  /// Fills the current block with operations until the required types are
  /// generated. Additionally generates operations using a geometric
  /// distribution.
  LogicalResult generateBlock(bool ensureTerminator = false,
                              llvm::SmallVector<Type> requiredTypes = {});

  /// Utility function to generate an operation.
  mlir::LogicalResult generate(RegisteredOperationName ron) {
    if (!ron.hasInterface<GeneratableOpInterface>()) {
      llvm::errs() << ron << "does not implement GeneratableOpInterface";
      return failure();
    }
    return ron.getInterface<GeneratableOpInterface>()->generate(builder);
  }

  /// Utility function to get generatable types of an operation.
  llvm::SmallVector<Type> getGeneratableTypes(RegisteredOperationName ron) {
    if (!ron.hasInterface<GeneratableOpInterface>()) {
      llvm::errs() << ron << "does not implement GeneratableOpInterface";
      return {};
    }
    return ron.getInterface<GeneratableOpInterface>()->getGeneratableTypes(
        builder);
  }

  /// Random number generator.
  std::mt19937 rngGen;

  /// The configuration used for IR generation.
  GeneratorOpBuilderConfig generatorConfig;

  /// A reference to the builder to pass to the generation functions.
  GeneratorOpBuilder &builder;

  /// All operations that can be generated.
  llvm::SmallVector<RegisteredOperationName> availableOps;

  /// If set, forces operations to have the specified return type.
  llvm::Optional<Type> requiredReturnType;

  /// Allows bypassing the block length limit to ensure that terminators can be
  /// generated.
  bool bypassBlockLengthLimit;
};
} // namespace detail
} // namespace mlir

bool GeneratorOpBuilderImpl::canCreate(const OperationState &state) {
  // Enforce return type if needed.
  if (requiredReturnType.has_value())
    if (llvm::find(state.types, requiredReturnType.value()) ==
        state.types.end())
      return false;

  // Enforce nested region limit.
  unsigned depth = 0;
  Block *block = builder.getBlock();

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
  block = builder.getBlock();
  if (!bypassBlockLengthLimit && block != nullptr)
    if (block->getOperations().size() >= generatorConfig.blockLengthLimit())
      return false;

  return true;
}

unsigned GeneratorOpBuilderImpl::sampleUniform(int32_t max) {
  if (max < 0)
    llvm::errs()
        << "upper bound of 'sampleUniform(int32_t max)' must be positive\n";
  std::uniform_int_distribution<unsigned> dist(0, max);
  return dist(rngGen);
}

template <typename T>
llvm::Optional<T>
GeneratorOpBuilderImpl::sample(llvm::SmallVector<T> choices,
                               llvm::SmallVector<unsigned> probs) {
  if (choices.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "GeneratorOpBuilderImpl::sample requires a "
                               "nonempty list of choices. Caller: "
                            << getImmediateCaller() << "\n");
    return std::nullopt;
  }

  if (!probs.empty() && probs.size() != choices.size()) {
    LLVM_DEBUG(llvm::dbgs() << "GeneratorOpBuilderImpl::sample requires probs "
                               "to be empty or match choices. Caller: "
                            << getImmediateCaller() << "\n");
    return std::nullopt;
  }

  // Fill up with ones if probs is empty.
  while (probs.size() < choices.size())
    probs.push_back(1);

  // Probability based sampling.
  for (unsigned i = 1; i < probs.size(); ++i)
    probs[i] += probs[i - 1];

  if (probs[probs.size() - 1] < 1) {
    LLVM_DEBUG(llvm::dbgs()
               << "GeneratorOpBuilderImpl::sample requires at "
                  "least one choice with probability of one. Caller: "
               << getImmediateCaller() << "\n");
    return std::nullopt;
  }

  std::uniform_int_distribution<unsigned> dist(1, probs[probs.size() - 1]);

  unsigned rngNum = dist(rngGen);
  unsigned idx = 0;
  while (rngNum > probs[idx])
    idx++;

  return choices[idx];
}

bool GeneratorOpBuilderImpl::sampleBool() {
  // 0.5 is the probability for generating true.
  std::bernoulli_distribution dist(0.5);
  return dist(rngGen);
}

template <typename T>
T GeneratorOpBuilderImpl::sampleNumber() {
  static_assert(std::is_arithmetic<T>::value, "Numeric type required");
  // Normal distribution, mean=0, stddev=1
  std::normal_distribution<> dist(0, 1);

  if constexpr (std::is_integral<T>::value) {
    return static_cast<T>(std::round(dist(rngGen)));
  } else if constexpr (std::is_floating_point<T>::value) {
    return static_cast<T>(dist(rngGen));
  }
}

llvm::Optional<llvm::SmallVector<Value>>
GeneratorOpBuilderImpl::generateOperation(
    llvm::SmallVector<RegisteredOperationName> ops) {
  LLVM_DEBUG(
      llvm::dbgs()
      << "GeneratorOpBuilderImpl::generateOperation started generating\n");

  Block *block = builder.getBlock();
  if (block == nullptr) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "GeneratorOpBuilderImpl::generateOperation requires a block\n");
    return std::nullopt;
  }

  if (ops.empty())
    ops = availableOps;

  LogicalResult logicalResult = failure();
  llvm::Optional<RegisteredOperationName> op;
  OpBuilder::InsertPoint ip = builder.saveInsertionPoint();

  // TODO: GeneratorOpBuilder should track changes and rollback on failure
  while (logicalResult.failed()) {
    builder.restoreInsertionPoint(ip);

    // Lookup probabilities.
    llvm::SmallVector<unsigned> probs;
    for (RegisteredOperationName op : ops)
      probs.push_back(generatorConfig.getProb(op.getStringRef()));

    op = sample(ops, probs);
    if (!op.has_value()) {
      LLVM_DEBUG(llvm::dbgs() << "GeneratorOpBuilderImpl::generateOperation "
                                 "failed to sample an operation\n");
      return std::nullopt;
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

  // Results of the last inserted operation.
  // FIXME: GeneratorOpBuilder should track the last inserted op
  return block->back().getResults();
}

llvm::Optional<llvm::SmallVector<Value>>
GeneratorOpBuilderImpl::generateTerminator() {
  LLVM_DEBUG(
      llvm::dbgs()
      << "GeneratorOpBuilderImpl::generateTerminator started generating\n");

  Block *block = builder.getBlock();
  if (block == nullptr) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "GeneratorOpBuilderImpl::generateTerminator requires a block\n");
    return std::nullopt;
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
    return std::nullopt;
  }

  if (!generateOperation(terminatorOps).has_value()) {
    LLVM_DEBUG(llvm::dbgs() << "GeneratorOpBuilderImpl::generateTerminator "
                               "failed to generate terminator\n");
    return std::nullopt;
  }

  // Results of the last inserted operation.
  // FIXME: GeneratorOpBuilder should track the last inserted op
  return block->back().getResults();
}

llvm::SmallVector<Type> GeneratorOpBuilderImpl::sampleTypes(int32_t min) {
  if (min < 0)
    llvm::errs()
        << "lower bound of 'sampleTypes(int32_t min)' must be positive\n";

  llvm::SmallVector<Type> availableTypes;
  for (RegisteredOperationName op : availableOps)
    for (Type t : getGeneratableTypes(op))
      if (llvm::find(availableTypes, t) == availableTypes.end())
        availableTypes.push_back(t);

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

bool GeneratorOpBuilderImpl::hasValueOfType(Type t) {
  llvm::SmallVector<Value> possibleValues;
  llvm::SmallVector<Value> excludedValues;

  Block *block = builder.getBlock();

  while (block != nullptr) {
    // Add all operations in the block of type t.
    for (Operation &op : block->getOperations())
      for (OpResult res : op.getResults())
        if (res.getType() == t)
          possibleValues.push_back(res);

    // Add all arguments of the block of type t
    for (BlockArgument bArg : block->getArguments())
      if (bArg.getType() == t)
        possibleValues.push_back(bArg);

    // Move up the hierarchy.
    Operation *parent = block->getParentOp();
    block = nullptr;

    if (parent != nullptr) {
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

  return !possibleValues.empty();
}

llvm::Optional<Value> GeneratorOpBuilderImpl::sampleValueOfType(Type t) {
  llvm::SmallVector<Value> possibleValues;
  llvm::SmallVector<Value> excludedValues;

  Block *block = builder.getBlock();

  while (block != nullptr) {
    // Add all operations in the block of type t.
    for (Operation &op : block->getOperations())
      for (OpResult res : op.getResults())
        if (res.getType() == t)
          possibleValues.push_back(res);

    // Add all arguments of the block of type t
    for (BlockArgument bArg : block->getArguments())
      if (bArg.getType() == t)
        possibleValues.push_back(bArg);

    // Move up the hierarchy.
    Operation *parent = block->getParentOp();
    block = nullptr;

    if (parent != nullptr) {
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

llvm::Optional<Value> GeneratorOpBuilderImpl::generateValueOfType(Type t) {
  LLVM_DEBUG(
      llvm::dbgs()
      << "GeneratorOpBuilderImpl::generateValueOfType started generating\n");

  llvm::SmallVector<RegisteredOperationName> possibleOps;

  for (RegisteredOperationName op : availableOps) {
    llvm::SmallVector<Type> opTypes = getGeneratableTypes(op);
    if (llvm::find(opTypes, t) != opTypes.end())
      possibleOps.push_back(op);
  }

  // Nothing to sample from.
  if (possibleOps.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "GeneratorOpBuilderImpl::generateValueOfType "
                               "has no operations generating the type "
                            << t << "\n");
    return {};
  }

  Block *block = builder.getBlock();
  if (block == nullptr) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "GeneratorOpBuilderImpl::generateValueOfType requires a block\n");
    return std::nullopt;
  }

  requiredReturnType = t;
  llvm::Optional<llvm::SmallVector<Value>> values =
      generateOperation(possibleOps);
  requiredReturnType = std::nullopt;

  // Failed to generate.
  if (!values.has_value()) {
    LLVM_DEBUG(llvm::dbgs() << "GeneratorOpBuilderImpl::generateValueOfType "
                               "failed to generate operation for type "
                            << t << "\n");
    return std::nullopt;
  }

  llvm::SmallVector<Value> possible_values;
  for (Value value : values.value())
    if (value.getType() == t)
      possible_values.push_back(value);

  if (!possible_values.empty())
    return sample(possible_values);

  LLVM_DEBUG(llvm::dbgs() << "GeneratorOpBuilderImpl::generateValueOfType "
                             "generated operation without the requested type "
                          << t << "\n");
  return std::nullopt;
}

llvm::Optional<Value>
GeneratorOpBuilderImpl::sampleOrGenerateValueOfType(Type t) {
  // First try to sample.
  llvm::Optional<Value> outputValue = sampleValueOfType(t);
  if (outputValue.has_value())
    return outputValue;
  // If sampling fails, try to generate.
  return generateValueOfType(t);
}

LogicalResult
GeneratorOpBuilderImpl::generateBlock(bool ensureTerminator,
                                      llvm::SmallVector<Type> requiredTypes) {
  LLVM_DEBUG(llvm::dbgs()
             << "GeneratorOpBuilderImpl::generateBlock started generating\n");

  Block *block = builder.getBlock();
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

  for (int i = 0; i < length; ++i)
    generateOperation(possibleOps);

  LLVM_DEBUG(llvm::dbgs() << "GeneratorOpBuilderImpl::generateBlock "
                             "successfully generated additional operations\n");

  bypassBlockLengthLimit = true;
  // Generate until all required types are generated.
  for (Type t : requiredTypes)
    if (!generateValueOfType(t).has_value()) {
      LLVM_DEBUG(llvm::dbgs() << "GeneratorOpBuilderImpl::generateBlock failed "
                                 "to generate value of type "
                              << t << "\n");
      return failure();
    }

  LLVM_DEBUG(llvm::dbgs() << "GeneratorOpBuilderImpl::generateBlock "
                             "successfully generated required types\n");

  // If required, continue until a terminator has been generated.
  if (!ensureTerminator) {
    bypassBlockLengthLimit = false;
    builder.setInsertionPointAfter(block->getParentOp());
    LLVM_DEBUG(llvm::dbgs()
               << "GeneratorOpBuilderImpl::generateBlock "
                  "successfully generated block without terminator\n");
    return success();
  }

  // Try to generate terminator.
  if (!generateTerminator().has_value()) {
    LLVM_DEBUG(llvm::dbgs() << "GeneratorOpBuilderImpl::generateBlock failed "
                               "to generate terminator\n");
    return failure();
  }

  bypassBlockLengthLimit = false;
  builder.setInsertionPointAfter(block->getParentOp());
  LLVM_DEBUG(llvm::dbgs() << "GeneratorOpBuilderImpl::generateBlock "
                             "successfully generated block with terminator\n");
  return success();
}

//===----------------------------------------------------------------------===//
// GeneratorOpBuilder
//===----------------------------------------------------------------------===//

GeneratorOpBuilder::GeneratorOpBuilder(MLIRContext *ctx,
                                       GeneratorOpBuilderConfig generatorConfig)
    : OpBuilder(ctx),
      impl(new detail::GeneratorOpBuilderImpl(ctx, generatorConfig, *this)) {}

// XXX: Might need to destruct impl
GeneratorOpBuilder::~GeneratorOpBuilder() = default;

Operation *GeneratorOpBuilder::create(const OperationState &state) {
  if (impl->canCreate(state))
    return OpBuilder::create(state);
  return nullptr;
}

Operation *
GeneratorOpBuilder::create(Location loc, StringAttr opName, ValueRange operands,
                           TypeRange types, ArrayRef<NamedAttribute> attributes,
                           BlockRange successors,
                           MutableArrayRef<std::unique_ptr<Region>> regions) {
  OperationState state(loc, opName, operands, types, attributes, successors,
                       regions);
  return create(state);
}

unsigned GeneratorOpBuilder::sampleUniform(int32_t max) {
  return impl->sampleUniform(max);
}

bool GeneratorOpBuilder::sampleBool() { return impl->sampleBool(); }

int8_t GeneratorOpBuilder::sampleNumberInt8() {
  return impl->sampleNumber<int8_t>();
}

int16_t GeneratorOpBuilder::sampleNumberInt16() {
  return impl->sampleNumber<int16_t>();
}

int32_t GeneratorOpBuilder::sampleNumberInt32() {
  return impl->sampleNumber<int32_t>();
}

int64_t GeneratorOpBuilder::sampleNumberInt64() {
  return impl->sampleNumber<int64_t>();
}

float_t GeneratorOpBuilder::sampleNumberFloat() {
  return impl->sampleNumber<float_t>();
}

double_t GeneratorOpBuilder::sampleNumberDouble() {
  return impl->sampleNumber<double_t>();
}

llvm::Optional<llvm::SmallVector<Value>>
GeneratorOpBuilder::generateOperation() {
  return impl->generateOperation();
}

llvm::Optional<llvm::SmallVector<Value>>
GeneratorOpBuilder::generateTerminator() {
  return impl->generateTerminator();
}

llvm::SmallVector<Type> GeneratorOpBuilder::sampleTypes(int32_t min) {
  return impl->sampleTypes(min);
}

bool GeneratorOpBuilder::hasValueOfType(Type t) {
  return impl->hasValueOfType(t);
}

llvm::Optional<Value> GeneratorOpBuilder::sampleValueOfType(Type t) {
  return impl->sampleValueOfType(t);
}

llvm::Optional<Value> GeneratorOpBuilder::generateValueOfType(Type t) {
  return impl->generateValueOfType(t);
}

llvm::Optional<Value> GeneratorOpBuilder::sampleOrGenerateValueOfType(Type t) {
  return impl->sampleOrGenerateValueOfType(t);
}

LogicalResult
GeneratorOpBuilder::generateBlock(bool ensureTerminator,
                                  llvm::SmallVector<Type> requiredTypes) {
  return impl->generateBlock(ensureTerminator, requiredTypes);
}

//===----------------------------------------------------------------------===//
// Table-generated class definitions
//===----------------------------------------------------------------------===//

/// Include the definitions of the copy operation interface.
#include "mlir/Interfaces/GeneratableOpInterface.cpp.inc"
