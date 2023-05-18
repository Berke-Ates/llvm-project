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
class GeneratableOpInterface;

//===----------------------------------------------------------------------===//
// Configuration options for the generator
//===----------------------------------------------------------------------===//

struct GeneratorOpBuilderConfig {
private:
  unsigned Seed;
  unsigned RegionDepth;
  unsigned DefaultProb;
  llvm::StringMap<unsigned> OpProbs;

public:
  /// Initial seed of the random number generator.
  unsigned seed() { return Seed; }
  /// Maximal depth of nested regions.
  unsigned regionDepth() { return RegionDepth; }
  /// Default probability of generating an operation.
  unsigned defaultProb() { return DefaultProb; }
  /// Probabilities of generating operations.
  llvm::StringMap<unsigned> opProbs() { return OpProbs; }

  LogicalResult
  loadFromFileContent(StringRef configFileContent,
                      std::string *errorMessage = (std::string *)nullptr) {
    // Clear previous default values
    OpProbs = {};

    std::istringstream input(configFileContent.str());
    std::string line;
    unsigned lineNr = 0;

    while (std::getline(input, line)) {
      std::istringstream lineStream(line);
      std::string key;
      char equalsSign;
      unsigned value;
      lineNr++;

      if (lineStream >> key >> equalsSign >> value && equalsSign == '=') {
        if (key == "seed")
          Seed = value;
        else if (key == "regionDepth")
          RegionDepth = value;
        else if (key == "defaultProb")
          DefaultProb = value;
        else
          OpProbs[key] = value;
      } else {
        if (errorMessage)
          *errorMessage =
              "failed to parse config file at line " + std::to_string(lineNr);
        return failure();
      }
    }

    return success();
  }

  void loadDefaultValues(MLIRContext *ctx) {
    Seed = time(0);
    RegionDepth = 5;
    DefaultProb = 1;
    OpProbs = {};

    for (RegisteredOperationName ron : ctx->getRegisteredOperations())
      if (ron.hasInterface<GeneratableOpInterface>())
        OpProbs[ron.getStringRef()] = DefaultProb;
  }

  void dumpConfig(raw_ostream &os) {
    os << "seed = " << Seed << "\n";
    os << "regionDepth = " << RegionDepth << "\n";
    os << "defaultProb = " << DefaultProb << "\n";

    for (StringRef key : OpProbs.keys())
      os << key << " = " << OpProbs[key] << "\n";
  }
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

  /// Randomly tries to choose generated value of the given type, if one exists.
  /// If this fails, randomly generates an operation with the given return type,
  /// if possible.
  llvm::Optional<Value> sampleOrGenerateValueOfType(Type t);

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
