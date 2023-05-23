//===- GeneratableOpInterface.cpp - Generatable operations interface ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/GeneratableOpInterface.h"
#include <random>
#include <type_traits>

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// GeneratorOpBuilderImpl
//===----------------------------------------------------------------------===//

namespace mlir {
namespace detail {
struct GeneratorOpBuilderImpl {
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

  /// Returns a random number between 0 and max (inclusive) using uniform
  /// distribution.
  unsigned sampleUniform(int32_t max);

  /// Samples from a vector of choices. If no probability vector is provided,
  /// it samples using a uniform distribution.
  template <typename T>
  llvm::Optional<T> sample(SmallVector<T> choices,
                           SmallVector<unsigned> probs = {});

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
  TypeRange sampleTypeRange();

  /// Checks if a value of the given type is available in the current position.
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
    if (!ron.hasInterface<GeneratableOpInterface>())
      return failure();
    return ron.getInterface<GeneratableOpInterface>()->generate(builder);
  }

  /// Utility function to get generatable types of an operation.
  llvm::SmallVector<Type> getGeneratableTypes(RegisteredOperationName ron) {
    if (!ron.hasInterface<GeneratableOpInterface>())
      return {};
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
};
} // namespace detail
} // namespace mlir

unsigned GeneratorOpBuilderImpl::sampleUniform(int32_t max) {
  if (max < 0)
    emitError(builder.getUnknownLoc())
        << "upper bound of 'sampleUniform(int32_t max)' must be positive";
  std::uniform_int_distribution<unsigned> dist(0, max);
  return dist(rngGen);
}

template <typename T>
llvm::Optional<T> GeneratorOpBuilderImpl::sample(SmallVector<T> choices,
                                                 SmallVector<unsigned> probs) {
  if (choices.empty())
    return std::nullopt;

  if (!probs.empty() && probs.size() != choices.size())
    return std::nullopt;

  // Fill up with ones if probs is empty.
  while (probs.size() < choices.size())
    probs.push_back(1);

  // Probability based sampling.
  for (unsigned i = 1; i < probs.size(); ++i)
    probs[i] += probs[i - 1];

  if (probs[probs.size() - 1] < 1)
    return std::nullopt;

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
  if (ops.empty())
    ops = availableOps;

  // Lookup probabilities.
  SmallVector<unsigned> probs;
  for (RegisteredOperationName op : ops)
    probs.push_back(generatorConfig.getProb(op.getStringRef()));

  LogicalResult logicalResult = failure();

  while (logicalResult.failed()) {
    Optional<RegisteredOperationName> op = sample(ops, probs);
    if (!op.has_value())
      return std::nullopt;

    logicalResult = generate(op.value());
  }

  // Results of the last inserted operation.
  Block *block = builder.getBlock();
  if (block != nullptr)
    return block->back().getResults();

  return {};
}

llvm::Optional<llvm::SmallVector<Value>>
GeneratorOpBuilderImpl::generateTerminator() {
  if (availableOps.empty())
    return std::nullopt;

  Block *block = builder.getBlock();
  if (block != nullptr)
    return std::nullopt;

  // Filter available ops.
  SmallVector<mlir::RegisteredOperationName> terminatorOps;
  for (RegisteredOperationName op : availableOps)
    if (op.hasTrait<OpTrait::IsTerminator>())
      terminatorOps.push_back(op);

  if (terminatorOps.empty() || !generateOperation(terminatorOps).has_value())
    return std::nullopt;

  // Results of the last inserted operation.
  return block->back().getResults();
}

TypeRange GeneratorOpBuilderImpl::sampleTypeRange() {
  SmallVector<Type> availableTypes;

  for (auto op : availableOps)
    availableTypes.append(getGeneratableTypes(op));

  // Nothing to sample from.
  if (availableTypes.empty())
    return {};

  // Geometric distribution, p=0.5
  std::geometric_distribution<> dist(0.5);
  int length = dist(rngGen);

  SmallVector<Type> types;
  for (int i = 0; i < length; ++i) {
    Optional<Type> type = sample(availableTypes);

    // Failed to sample.
    if (!type.has_value())
      return {};

    types.push_back(type.value());
  }

  return types;
}

bool GeneratorOpBuilderImpl::hasValueOfType(Type t) {
  return sampleValueOfType(t).has_value();
}

llvm::Optional<Value> GeneratorOpBuilderImpl::sampleValueOfType(Type t) {
  SmallVector<Value> possibleValues;
  SmallVector<Value> excludedValues;

  Block *block = builder.getBlock();

  while (block != nullptr) {
    // Add all operations in the block of type t.
    for (Operation &op : block->getOperations())
      for (OpResult res : op.getResults())
        if (res.getType() == t)
          possibleValues.push_back(res);

    // Move up the hierarchy.
    Operation *parent = block->getParentOp();
    block = nullptr;

    if (parent != nullptr) {
      for (OpResult res : parent->getResults())
        excludedValues.push_back(res);

      block = parent->getBlock();
    }
  }

  // XXX: This might not work as intended.
  for (Value val : excludedValues) {
    emitError(builder.getUnknownLoc()) << "ENTERING VALUE SAMPLING ERASURE";
    if (llvm::find(possibleValues, val) != possibleValues.end())
      possibleValues.erase(&val);
  }

  return sample(possibleValues);
}

llvm::Optional<Value> GeneratorOpBuilderImpl::generateValueOfType(Type t) {
  SmallVector<RegisteredOperationName> possibleOps;

  for (auto op : availableOps) {
    SmallVector<Type> opTypes = getGeneratableTypes(op);
    if (llvm::find(opTypes, t) != opTypes.end())
      possibleOps.push_back(op);
  }

  // Nothing to sample from.
  if (possibleOps.empty())
    return {};

  Block *block = builder.getBlock();
  if (block != nullptr)
    return std::nullopt;

  while (!possibleOps.empty()) {
    auto values = generateOperation(possibleOps);

    // Failed to generate.
    if (!values.has_value())
      return std::nullopt;

    SmallVector<Value> possible_values;

    for (Value value : values.value())
      if (value.getType() == t)
        possible_values.push_back(value);

    if (!possible_values.empty())
      return sample(possible_values);

    block->back().erase();
  }

  return std::nullopt;
}

llvm::Optional<Value>
GeneratorOpBuilderImpl::sampleOrGenerateValueOfType(Type t) {
  // First try to sample.
  Optional<Value> outputValue = sampleValueOfType(t);
  if (outputValue.has_value())
    return outputValue;
  // If sampling fails, try to generate.
  return generateValueOfType(t);
}

LogicalResult
GeneratorOpBuilderImpl::generateBlock(bool ensureTerminator,
                                      llvm::SmallVector<Type> requiredTypes) {

  Block *block = builder.getBlock();
  if (block == nullptr)
    return failure();

  // Generate until all required types are generated.
  for (Type t : requiredTypes)
    if (!generateValueOfType(t).has_value())
      return failure();

  // Randomly continue generating.

  // Geometric distribution, p=0.5
  std::geometric_distribution<> dist(0.5);
  int length = dist(rngGen);

  // Filter available ops.
  SmallVector<mlir::RegisteredOperationName> possibleOps;
  for (RegisteredOperationName op : availableOps)
    if (!op.hasTrait<OpTrait::IsTerminator>())
      possibleOps.push_back(op);

  for (int i = 0; i < length; ++i)
    generateOperation(possibleOps);

  // If required, continue until a terminator has been generated.
  if (!ensureTerminator)
    return success();

  Operation *lastOp = &block->back();
  while (!lastOp->hasTrait<OpTrait::IsTerminator>()) {
    auto values = generateOperation(availableOps);

    // Failed to generate.
    if (!values.has_value())
      return failure();

    lastOp = &block->back();
  }

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

TypeRange GeneratorOpBuilder::sampleTypeRange() {
  return impl->sampleTypeRange();
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
