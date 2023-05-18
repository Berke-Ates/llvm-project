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
  int seed = time(0);
  int regionDepth = 5;
  int defaultProb = 1;
  llvm::StringMap<int> opSettings = {};

  LogicalResult
  loadFromFileContent(StringRef configFileContent,
                      std::string *errorMessage = (std::string *)nullptr) {
    std::istringstream input(configFileContent.str());
    std::string line;
    unsigned lineNr = 0;

    while (std::getline(input, line)) {
      std::istringstream lineStream(line);
      std::string key;
      char equalsSign;
      int value;
      lineNr++;

      if (lineStream >> key >> equalsSign >> value && equalsSign == '=') {
        if (key == "seed")
          seed = value;
        else if (key == "regionDepth")
          regionDepth = value;
        else if (key == "defaultProb")
          defaultProb = value;
        else
          opSettings[key] = value;
      } else {
        if (errorMessage)
          *errorMessage =
              "failed to parse config file at line " + std::to_string(lineNr);
        return failure();
      }
    }

    return success();
  }

  void dumpConfig(raw_ostream &os) {
    os << "seed = " << seed << "\n";
    os << "regionDepth = " << regionDepth << "\n";
    os << "defaultProb = " << defaultProb << "\n";

    for (StringRef key : opSettings.keys())
      os << key << " = " << opSettings[key] << "\n";
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
