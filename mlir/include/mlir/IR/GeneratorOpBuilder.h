//===- GeneratorOpBuilder.h - MLIR Program Generator ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Operation builder that contains various utility functions for generating
// random IR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_GENERATOROPBUILDER_H
#define MLIR_IR_GENERATOROPBUILDER_H

#include "mlir/IR/Builders.h"
#include "llvm/ADT/StringMap.h"
#include <functional>
#include <random>

namespace mlir {
class GeneratableOpInterface;

//===----------------------------------------------------------------------===//
// Configuration options for the generator
//===----------------------------------------------------------------------===//

struct GeneratorOpBuilderConfig {
  /// Returns the initial seed for the random number generator.
  unsigned seed() { return Seed; }

  /// Sets the initial seed for the random number generator.
  void seed(unsigned seed) { Seed = seed; }

  /// Returns the maximum depth for nested regions.
  unsigned regionDepthLimit() { return RegionDepthLimit; }

  /// Returns the maximum number of operations that can be generated in a block.
  unsigned blockLengthLimit() { return BlockLengthLimit; }

  /// Returns the default probability for generating an operation.
  unsigned defaultProb() { return DefaultProb; }

  /// Returns a map of operation names to their respective probabilities.
  llvm::StringMap<unsigned> opProbs() { return OpProbs; }

  /// Returns the probability of generating an operation with a given name.
  /// If the operation name is not in the map, returns the default probability.
  unsigned getProb(StringRef name) {
    return OpProbs.contains(name) ? OpProbs[name] : DefaultProb;
  }

  /// Loads configuration from a string containing the content of a config file.
  /// Optionally returns an error message if the parsing fails.
  LogicalResult
  loadFromFileContent(StringRef configFileContent,
                      std::string *errorMessage = (std::string *)nullptr) {
    // Clear previous default values
    OpProbs = {};

    std::istringstream input(configFileContent.str());
    std::string line;
    unsigned lineNr = 0;

    while (std::getline(input, line)) {
      if (line.empty())
        continue;
      std::istringstream lineStream(line);
      std::string key;
      char equalsSign;
      unsigned value;
      lineNr++;

      if (lineStream >> key >> equalsSign >> value && equalsSign == '=') {
        if (key == "seed")
          Seed = value;
        else if (key == "regionDepthLimit")
          RegionDepthLimit = value;
        else if (key == "blockLengthLimit")
          BlockLengthLimit = value;
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

  /// Initializes the configuration with default values.
  void loadDefaultValues(MLIRContext *ctx) {
    Seed = time(0);
    RegionDepthLimit = 3;
    BlockLengthLimit = 20;
    DefaultProb = 1;
    OpProbs = {};

    for (RegisteredOperationName ron : ctx->getRegisteredOperations())
      if (ron.hasInterface<GeneratableOpInterface>())
        OpProbs[ron.getStringRef()] = DefaultProb;
  }

  /// Dumps the current configuration to an output stream.
  void dumpConfig(llvm::raw_ostream &os) {
    os << "seed = " << Seed << "\n";
    os << "regionDepthLimit = " << RegionDepthLimit << "\n";
    os << "blockLengthLimit = " << BlockLengthLimit << "\n";
    os << "defaultProb = " << DefaultProb << "\n";

    // Sort OpProbs for nicer print.
    std::vector<llvm::StringMapEntry<unsigned> *> vec;
    vec.reserve(OpProbs.size());
    for (auto &pair : OpProbs)
      vec.push_back(&pair);

    std::sort(vec.begin(), vec.end(),
              [](const llvm::StringMapEntry<unsigned> *a,
                 const llvm::StringMapEntry<unsigned> *b) {
                return a->getKey().compare(b->getKey()) < 0;
              });

    for (auto *entry : vec)
      os << entry->getKey() << " = " << entry->getValue() << "\n";
  }

private:
  unsigned Seed;
  unsigned RegionDepthLimit;
  unsigned BlockLengthLimit;
  unsigned DefaultProb;
  llvm::StringMap<unsigned> OpProbs;
};

//===----------------------------------------------------------------------===//
// OpBuilder with utilities for IR generation
//===----------------------------------------------------------------------===//

/// This class implements a builder for use in generation functions. It
/// extends the base OpBuilder and provides utility functions to generate
/// common structures and to sample from a configured distribution.
class GeneratorOpBuilder final : public OpBuilder {
public:
  explicit GeneratorOpBuilder(MLIRContext *ctxt,
                              GeneratorOpBuilderConfig generatorConfig);

  //===--------------------------------------------------------------------===//
  // State Management
  //===--------------------------------------------------------------------===//

  /// This class represents a saved snapshot of an operation.
  class Snapshot {
  public:
    Snapshot() = default;
    Snapshot(OpBuilder::InsertPoint ip) : ip(ip) {}

    OpBuilder::InsertPoint getInsertionPoint() { return ip; }

  private:
    OpBuilder::InsertPoint ip;
  };

  /// Takes a snapshot of the current state.
  Snapshot takeSnapshot();

  /// Rolls back to the provided snapshot.
  void rollback(Snapshot snapshot);

  //===--------------------------------------------------------------------===//
  // Operation Creation
  //===--------------------------------------------------------------------===//

  /// Checks if the given OperationState adheres to the generator constraints.
  bool canCreate(Operation *op);

  /// Creates an operation given the fields represented as an OperationState and
  /// enforces generator constraints.
  Operation *create(const OperationState &state);

  //===--------------------------------------------------------------------===//
  // Collectors
  //===--------------------------------------------------------------------===//

  /// Returns a list of all available values in the current location.
  llvm::SmallVector<Value>
  collectValues(std::function<bool(const Value &)> filter = nullptr);

  /// Returns a list of all available types in the current location.
  llvm::SmallVector<Type>
  collectTypes(std::function<bool(const Type &)> filter = nullptr);

  //===--------------------------------------------------------------------===//
  // Samplers
  //===--------------------------------------------------------------------===//

  /// Samples from a vector of choices. If no probability vector is provided,
  /// it samples using a uniform distribution.
  template <typename T>
  llvm::Optional<T> sample(llvm::SmallVector<T> choices,
                           llvm::SmallVector<unsigned> probs = {}) {
    if (choices.empty() || (!probs.empty() && probs.size() != choices.size()))
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
    unsigned rngNum = dist(rng);
    unsigned idx = 0;
    while (rngNum > probs[idx])
      idx++;

    return choices[idx];
  }

  /// Samples multiple times from a vector of choices. If no probability vector
  /// is provided, it samples using a uniform distribution.
  template <typename T>
  llvm::Optional<llvm::SmallVector<T>>
  sample(llvm::SmallVector<T> choices, unsigned num,
         bool allowDuplicates = true, llvm::SmallVector<unsigned> probs = {}) {
    if (choices.empty() || (!probs.empty() && probs.size() != choices.size()))
      return std::nullopt;

    // Fill up with ones if probs is empty.
    while (probs.size() < choices.size())
      probs.push_back(1);

    llvm::SmallVector<T> selected;

    for (unsigned i = 0; i < num; ++i) {
      llvm::Optional<T> choice = sample(choices, probs);
      if (!choice.has_value())
        return std::nullopt;

      selected.push_back(choice.value());

      // If duplicates are not allowed, remove the choice from choices and
      // probs.
      if (!allowDuplicates) {
        auto it = llvm::find(choices, choice);
        auto index = std::distance(choices.begin(), it);
        choices.erase(it);
        probs.erase(probs.begin() + index);
      }
    }

    return selected;
  }

  /// Returns a random boolean.
  bool sampleBool() { return sample<bool>({true, false}).value(); }

  /// Returns a random number between min and max (inclusive) using uniform
  /// distribution.
  template <typename T>
  T sampleUniform(T min, T max) {
    return std::uniform_int_distribution<T>(min, max)(rng);
  }

  /// Returns a random number using normal or geometric distribution.
  template <typename T>
  T sampleNumber(bool useGeometric = false) {
    static_assert(std::is_arithmetic<T>::value, "Numeric type required");

    double num;
    if (useGeometric) {
      std::geometric_distribution<> dist(0.2);
      num = dist(rng);
    } else {
      std::normal_distribution<> dist(0, 1);
      num = dist(rng);
    }

    if constexpr (std::is_integral<T>::value) {
      return static_cast<T>(std::round(num));
    } else if constexpr (std::is_floating_point<T>::value) {
      return static_cast<T>(num);
    }

    // Default return value, should never be reached.
    return T();
  }

  /// Returns a list of values using the filters.
  llvm::Optional<llvm::SmallVector<Value>>
  sampleValues(llvm::SmallVector<std::function<bool(const Value &)>> filters);

  /// Returns a value using the filter.
  llvm::Optional<Value> sampleValue(std::function<bool(const Value &)> filter);

  /// Returns a list of values of the provided types.
  llvm::Optional<llvm::SmallVector<Value>>
  sampleValuesOfTypes(llvm::SmallVector<Type> types);

  /// Returns a value of the provided type.
  llvm::Optional<Value> sampleValueOfType(Type type);

  /// Samples from a geometric distribution of available types in the current
  /// position.
  llvm::SmallVector<Type>
  sampleTypes(unsigned min = 0,
              std::function<bool(const Type &)> filter = nullptr);

  //===--------------------------------------------------------------------===//
  // Generators
  //===--------------------------------------------------------------------===//

  /// Randomly generates an operation in the current position and returns it.
  /// Returns nullptr if it fails.
  Operation *generateOperation(llvm::SmallVector<RegisteredOperationName> ops);

  /// Randomly generates a terminator operation in the current position and
  /// returns it. Returns nullptr if it fails.
  Operation *generateTerminator();

  /// Fills the current block with operations.
  LogicalResult generateBlock(Block *block, bool ensureTerminator = false);

private:
  /// Utility function to generate an operation.
  Operation *generate(RegisteredOperationName ron);

  /// Random number generator.
  std::mt19937 rng;

  /// The configuration used for IR generation.
  GeneratorOpBuilderConfig generatorConfig;

  /// All operations that can be generated.
  llvm::SmallVector<RegisteredOperationName> availableOps = {};
}; // namespace mlir

} // namespace mlir

#endif // MLIR_IR_GENERATOROPBUILDER_H
