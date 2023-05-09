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
  }

  /// Samples from a vector of choices.
  template <typename T>
  T sample(SmallVector<T> choices);

  /// Returns a random boolean.
  bool sampleBool();

  /// Returns a random number using a normal distribution around zero. This
  /// function combines all specialized functions.
  template <typename T>
  T sampleNumber();

  /// Samples from a geometric distribution of types.
  TypeRange sampleTypeRange();

  /// Randomly chooses a generated value of the given type.
  Value sampleValueOfType(Type t);

  /// Generates a region until a terminator is generated (if required).
  LogicalResult generateRegion();

  /// A reference to the builder to pass to the generation functions.
  GeneratorOpBuilder &builder;

  /// All operations that can be generated.
  llvm::SmallVector<detail::GeneratableOpInterfaceInterfaceTraits::Concept *>
      available_ops;
};
} // namespace detail
} // namespace mlir

bool sampleBool() {
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

LogicalResult GeneratorOpBuilderImpl::generateRegion() {
  available_ops[0]->generate(builder);
  return success();
}

//===----------------------------------------------------------------------===//
// GeneratorOpBuilder
//===----------------------------------------------------------------------===//

GeneratorOpBuilder::GeneratorOpBuilder(MLIRContext *ctx)
    : OpBuilder(ctx), impl(new detail::GeneratorOpBuilderImpl(ctx, *this)) {}

GeneratorOpBuilder::~GeneratorOpBuilder() = default;

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

float_t GeneratorOpBuilder::sampleNumberFloat32() {
  return impl->sampleNumber<float_t>();
}

double_t GeneratorOpBuilder::sampleNumberFloat64() {
  return impl->sampleNumber<double_t>();
}

TypeRange GeneratorOpBuilder::sampleTypeRange() {
  return impl->sampleTypeRange();
}

Value GeneratorOpBuilder::sampleValueOfType(Type t) {
  return impl->sampleValueOfType(t);
}

LogicalResult GeneratorOpBuilder::generateRegion() {
  return impl->generateRegion();
}

//===----------------------------------------------------------------------===//
// Table-generated class definitions
//===----------------------------------------------------------------------===//

/// Include the definitions of the copy operation interface.
#include "mlir/Interfaces/GeneratableOpInterface.cpp.inc"
