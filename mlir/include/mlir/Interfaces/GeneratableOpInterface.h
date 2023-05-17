//===- GeneratableOpInterface.h - Generatable interface ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the operation interface for generatable operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_GENERATABLEOPINTERFACE_H
#define MLIR_INTERFACES_GENERATABLEOPINTERFACE_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// Configuration options for the generator
//===----------------------------------------------------------------------===//

struct GeneratorOpBuilderConfig {
  // TODO: Implement config struct

  void loadFromFile() {}
  void dumpConfig(raw_ostream &os) {}
};

//===----------------------------------------------------------------------===//
// OpBuilder with utilities for IR generation
//===----------------------------------------------------------------------===//

namespace detail {
struct GeneratorOpBuilderImpl;
} // namespace detail

/// This class implements a builder for use in generation functions. It
/// extends the base OpBuilder and provides utility functions to generate common
/// structures and to sample from a configured distribution.
class GeneratorOpBuilder final : public OpBuilder {
public:
  explicit GeneratorOpBuilder(MLIRContext *ctxt,
                              GeneratorOpBuilderConfig generatorConfig);
  ~GeneratorOpBuilder();

  /// Returns a random number between 0 and max (inclusive) using uniform
  /// distribution.
  unsigned sampleUniform(unsigned max);

  /// Returns a random boolean.
  bool sampleBool();

  /// Returns a random number using a normal distribution around zero.
  int8_t sampleNumberInt8();

  /// Returns a random number using a normal distribution around zero.
  int16_t sampleNumberInt16();

  /// Returns a random number using a normal distribution around zero.
  int32_t sampleNumberInt32();

  /// Returns a random number using a normal distribution around zero.
  int64_t sampleNumberInt64();

  /// Returns a random number using a normal distribution around zero.
  float_t sampleNumberFloat();

  /// Returns a random number using a normal distribution around zero.
  double_t sampleNumberDouble();

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

private:
  std::unique_ptr<detail::GeneratorOpBuilderImpl> impl;
};

} // namespace mlir

//===----------------------------------------------------------------------===//
// Generated Interface Declarations
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/GeneratableOpInterface.h.inc"

#endif // MLIR_INTERFACES_GENERATABLEOPINTERFACE_H
