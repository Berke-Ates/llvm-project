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
  explicit GeneratorOpBuilderImpl(MLIRContext *ctx, GeneratorOpBuilder &builder)
      : builder(builder) {
    // Collect all operations usable for generation
    for (RegisteredOperationName ron : ctx->getRegisteredOperations())
      if (ron.hasInterface<GeneratableOpInterface>())
        available_ops.push_back(ron.getInterface<GeneratableOpInterface>());

    // Collect all generatable types
    for (auto op : available_ops)
      available_types.append(op->getGeneratableTypes(ctx));
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

  /// A reference to the builder to pass to the generation functions.
  GeneratorOpBuilder &builder;

  /// All operations that can be generated.
  llvm::SmallVector<detail::GeneratableOpInterfaceInterfaceTraits::Concept *>
      available_ops;

  /// All types that can be produced by a generatable operation.
  llvm::SmallVector<Type> available_types;
};
} // namespace detail
} // namespace mlir

unsigned GeneratorOpBuilderImpl::sampleUniform(unsigned max) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<unsigned> dist(0, max);
  return dist(gen);
}

template <typename T>
llvm::Optional<T> GeneratorOpBuilderImpl::sample(SmallVector<T> choices) {
  if (choices.size() == 0)
    return std::nullopt;

  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<int> dist(0, choices.size() - 1);
  return choices[dist(gen)];
}

bool GeneratorOpBuilderImpl::sampleBool() {
  std::random_device rd;
  std::mt19937 gen(rd());
  // 0.5 is the probability for generating true.
  std::bernoulli_distribution dist(0.5);
  return dist(gen);
}

template <typename T>
T GeneratorOpBuilderImpl::sampleNumber() {
  static_assert(std::is_arithmetic<T>::value, "Numeric type required");

  std::random_device rd;
  std::mt19937 gen(rd());

  // Normal distribution, mean=0, stddev=1
  std::normal_distribution<> dist(0, 1);

  if constexpr (std::is_integral<T>::value) {
    return static_cast<T>(std::round(dist(gen)));
  } else if constexpr (std::is_floating_point<T>::value) {
    return static_cast<T>(dist(gen));
  }
}

llvm::Optional<llvm::SmallVector<Value>>
GeneratorOpBuilderImpl::generateOperation() {
  if (available_ops.empty())
    return std::nullopt;

  LogicalResult logicalResult = failure();

  while (logicalResult.failed())
    logicalResult = sample(available_ops).value()->generate(builder);

  // Results of the last inserted operation
  return builder.getBlock()->back().getResults();
}

TypeRange GeneratorOpBuilderImpl::sampleTypeRange() {
  if (available_types.empty())
    return {};

  std::random_device rd;
  std::mt19937 gen(rd());

  // Geometric distribution, p=0.5
  std::geometric_distribution<> dist(0.5);
  int length = dist(gen);

  SmallVector<Type> types;
  for (int i = 0; i < length; ++i)
    types.push_back(sample(available_types).value());

  return types;
}

llvm::Optional<Value> GeneratorOpBuilderImpl::sampleValueOfType(Type t) {
  SmallVector<Value> possible_values;
  SmallVector<Value> excluded_values;

  Block *block = builder.getBlock();

  while (block != nullptr) {
    // Add all operations in the block of type t
    for (Operation &op : block->getOperations())
      for (OpResult res : op.getResults())
        if (res.getType() == t)
          possible_values.push_back(res);

    // Move up the hierarchy
    Operation *parent = block->getParentOp();
    block = nullptr;

    if (parent != nullptr) {
      for (OpResult res : parent->getResults())
        excluded_values.push_back(res);

      block = parent->getBlock();
    }
  }

  for (Value val : excluded_values)
    if (std::find(possible_values.begin(), possible_values.end(), val) !=
        possible_values.end())
      possible_values.erase(&val);

  return sample(possible_values);
}

llvm::Optional<Value> GeneratorOpBuilderImpl::generateValueOfType(Type t) {
  if (std::find(available_types.begin(), available_types.end(), t) !=
      available_types.end())
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
  std::random_device rd;
  std::mt19937 gen(rd());
  // 0.5 is the probability for generating true.
  std::bernoulli_distribution dist(0.5);

  Operation *lastOp = nullptr;
  while (lastOp == nullptr ||
         (requiresTerminator && !lastOp->hasTrait<OpTrait::IsTerminator>()) ||
         (!requiresTerminator && dist(gen))) {
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

GeneratorOpBuilder::GeneratorOpBuilder(MLIRContext *ctx)
    : OpBuilder(ctx), impl(new detail::GeneratorOpBuilderImpl(ctx, *this)) {}

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
