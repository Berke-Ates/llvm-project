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
// GeneratorOpBuilderImpl
//===----------------------------------------------------------------------===//

namespace mlir {
namespace detail {
struct GeneratorOpBuilderImpl {
  explicit GeneratorOpBuilderImpl(MLIRContext *ctx,
                                  GeneratorOpBuilderConfig generatorConfig,
                                  GeneratorOpBuilder &builder)
      : generatorConfig(generatorConfig), builder(builder) {
    // Collect all operations usable for generation
    for (RegisteredOperationName ron : ctx->getRegisteredOperations())
      if (ron.hasInterface<GeneratableOpInterface>())
        availableOps.push_back(ron.getInterface<GeneratableOpInterface>());

    // Collect all generatable types
    for (auto op : availableOps)
      availableTypes.append(op->getGeneratableTypes(ctx));

    // Setup random number generator
    std::random_device random_device;
    rngGen = std::mt19937(random_device());
  }

  /// Returns a random number between 0 and max (inclusive) using uniform
  /// distribution.
  unsigned sampleUniform(unsigned max);

  /// Samples from a vector of choices.
  template <typename T>
  llvm::Optional<T> sample(SmallVector<T> choices);

  /// Returns a random boolean.
  bool sampleBool();

  /// Returns a random number using a normal distribution around zero. This
  /// function combines all specialized functions.
  template <typename T>
  T sampleNumber();

  /// Randomly generates an operation.
  llvm::Optional<llvm::SmallVector<Value>> generateOperation();

  /// Samples from a geometric distribution of types.
  TypeRange sampleTypeRange();

  /// Randomly chooses a generated value of the given type, if one exists.
  llvm::Optional<Value> sampleValueOfType(Type t);

  /// Randomly generates an operation with the given return type, if possible.
  llvm::Optional<Value> generateValueOfType(Type t);

  /// Generates a region until a terminator is generated (if required).
  LogicalResult generateRegion(bool requiresTerminator);

  /// Random number generator.
  std::mt19937 rngGen;

  /// The configuration used for IR generation.
  GeneratorOpBuilderConfig generatorConfig;

  /// A reference to the builder to pass to the generation functions.
  GeneratorOpBuilder &builder;

  /// All operations that can be generated.
  llvm::SmallVector<detail::GeneratableOpInterfaceInterfaceTraits::Concept *>
      availableOps;

  /// All types that can be produced by a generatable operation.
  llvm::SmallVector<Type> availableTypes;
};
} // namespace detail
} // namespace mlir

unsigned GeneratorOpBuilderImpl::sampleUniform(unsigned max) {
  std::uniform_int_distribution<unsigned> dist(0, max);
  return dist(rngGen);
}

template <typename T>
llvm::Optional<T> GeneratorOpBuilderImpl::sample(SmallVector<T> choices) {
  if (choices.size() == 0)
    return std::nullopt;

  std::uniform_int_distribution<int> dist(0, choices.size() - 1);
  return choices[dist(rngGen)];
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
GeneratorOpBuilderImpl::generateOperation() {
  if (availableOps.empty())
    return std::nullopt;

  LogicalResult logicalResult = failure();

  while (logicalResult.failed())
    logicalResult = sample(availableOps).value()->generate(builder);

  // Results of the last inserted operation
  return builder.getBlock()->back().getResults();
}

TypeRange GeneratorOpBuilderImpl::sampleTypeRange() {
  if (availableTypes.empty())
    return {};

  // Geometric distribution, p=0.5
  std::geometric_distribution<> dist(0.5);
  int length = dist(rngGen);

  SmallVector<Type> types;
  for (int i = 0; i < length; ++i)
    types.push_back(sample(availableTypes).value());

  return types;
}

llvm::Optional<Value> GeneratorOpBuilderImpl::sampleValueOfType(Type t) {
  SmallVector<Value> possibleValues;
  SmallVector<Value> excludedValues;

  Block *block = builder.getBlock();

  while (block != nullptr) {
    // Add all operations in the block of type t
    for (Operation &op : block->getOperations())
      for (OpResult res : op.getResults())
        if (res.getType() == t)
          possibleValues.push_back(res);

    // Move up the hierarchy
    Operation *parent = block->getParentOp();
    block = nullptr;

    if (parent != nullptr) {
      for (OpResult res : parent->getResults())
        excludedValues.push_back(res);

      block = parent->getBlock();
    }
  }

  for (Value val : excludedValues)
    if (std::find(possibleValues.begin(), possibleValues.end(), val) !=
        possibleValues.end())
      possibleValues.erase(&val);

  return sample(possibleValues);
}

llvm::Optional<Value> GeneratorOpBuilderImpl::generateValueOfType(Type t) {
  if (std::find(availableTypes.begin(), availableTypes.end(), t) !=
      availableTypes.end())
    return std::nullopt;

  while (true) {
    llvm::Optional<SmallVector<Value>> values = generateOperation();
    if (!values.has_value())
      return std::nullopt;

    SmallVector<Value> possible_values;

    for (Value value : values.value())
      if (value.getType() == t)
        possible_values.push_back(value);

    if (!possible_values.empty())
      return sample(possible_values);
  }

  return std::nullopt;
}

LogicalResult GeneratorOpBuilderImpl::generateRegion(bool requiresTerminator) {
  // 0.5 is the probability for generating true.
  std::bernoulli_distribution dist(0.5);

  Operation *lastOp = nullptr;
  while (lastOp == nullptr ||
         (requiresTerminator && !lastOp->hasTrait<OpTrait::IsTerminator>()) ||
         (!requiresTerminator && dist(rngGen))) {
    llvm::Optional<SmallVector<Value>> values = generateOperation();
    if (!values.has_value())
      return failure();

    lastOp = &builder.getBlock()->back();
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

GeneratorOpBuilder::~GeneratorOpBuilder() = default;

unsigned GeneratorOpBuilder::sampleUniform(unsigned max) {
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

TypeRange GeneratorOpBuilder::sampleTypeRange() {
  return impl->sampleTypeRange();
}

llvm::Optional<Value> GeneratorOpBuilder::sampleValueOfType(Type t) {
  return impl->sampleValueOfType(t);
}

llvm::Optional<Value> GeneratorOpBuilder::generateValueOfType(Type t) {
  return impl->generateValueOfType(t);
}

LogicalResult GeneratorOpBuilder::generateRegion(bool requiresTerminator) {
  return impl->generateRegion(requiresTerminator);
}

//===----------------------------------------------------------------------===//
// Table-generated class definitions
//===----------------------------------------------------------------------===//

/// Include the definitions of the copy operation interface.
#include "mlir/Interfaces/GeneratableOpInterface.cpp.inc"
