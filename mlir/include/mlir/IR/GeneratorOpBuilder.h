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
#include <chrono>
#include <functional>
#include <random>
#include <variant>

namespace mlir {
class GeneratableOpInterface;

//===----------------------------------------------------------------------===//
// OpBuilder with utilities for IR generation
//===----------------------------------------------------------------------===//

/// This class implements a builder for use in generation functions. It
/// extends the base OpBuilder and provides utility functions to generate
/// common structures and to sample from a configured distribution.
class GeneratorOpBuilder final : public OpBuilder {
public:
  //===--------------------------------------------------------------------===//
  // Configuration Options
  //===--------------------------------------------------------------------===//

  /// This class manages configurations used during the generation.
  class Config {
  public:
    /// Represents one particular configuration entry.
    /// FIXME: Rewrite to allow user-defined values.
    struct Entry {
      std::variant<unsigned, int, double> value;

      LogicalResult parseString(llvm::StringRef s) {
        unsigned unsignedV;
        if (std::holds_alternative<unsigned>(value) &&
            !s.getAsInteger(10, unsignedV)) {
          value = unsignedV;
          return success();
        }

        int intV;
        if (std::holds_alternative<int>(value) && !s.getAsInteger(10, intV)) {
          value = intV;
          return success();
        }

        double doubleV;
        if (std::holds_alternative<double>(value) && !s.getAsDouble(doubleV)) {
          value = doubleV;
          return success();
        }

        return failure();
      }

      friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                           const Entry &entry) {
        std::visit([&os](auto &&arg) { os << arg; }, entry.value);
        return os;
      }
    };

    Config(MLIRContext *ctx) : context(ctx) {
      unsigned default_seed =
          std::chrono::high_resolution_clock::now().time_since_epoch().count();

      (void)registerConfig<unsigned>("_gen.seed", default_seed);
      (void)registerConfig<unsigned>("_gen.regionDepthLimit", 5);
      (void)registerConfig<unsigned>("_gen.blockLengthLimit", 100);
      (void)registerConfig<unsigned>("_gen.defaultProb", 1);
      (void)registerConfig<double>("_gen.geometricProb", 0.2);
      (void)registerConfig<double>("_gen.normalMean", 0);
      (void)registerConfig<double>("_gen.normalStddev", 10);

      for (RegisteredOperationName ron : ctx->getRegisteredOperations())
        if (ron.hasInterface<GeneratableOpInterface>())
          (void)registerConfig<int>(ron.getStringRef().str(), -1);

      registerOpConfigs();
    }

    MLIRContext *getContext() const { return context; }

    /// Registers a new configuration. Entry name must be unique.
    template <typename T>
    LogicalResult registerConfig(llvm::StringRef name, T value) {
      if (entries.contains(name)) {
        llvm::errs() << "Config " << name << " already registered\n";
        return failure();
      }

      entries[name] = {value};
      return success();
    }

    /// Returns the value of a config if it exists in the requested type.
    template <typename T>
    llvm::Optional<T> get(llvm::StringRef name) {
      if (!entries.contains(name) ||
          !std::holds_alternative<T>(entries[name].value)) {
        // llvm::errs() << "Config " << name << " unknown or wrong type\n";
        return std::nullopt;
      }

      return std::get<T>(entries[name].value);
    }

    /// Sets the value of a config if it exists in the requested type.
    template <typename T>
    LogicalResult set(llvm::StringRef name, T value) {
      if (frozen) {
        llvm::errs() << "Config is frozen\n";
        return failure();
      }

      if (!entries.contains(name) ||
          !std::holds_alternative<T>(entries[name].value)) {
        llvm::errs() << "Config " << name << " unknown or wrong type\n";
        return failure();
      }

      entries[name].value = value;
      return success();
    }

    /// Loads configuration from a string containing the content of a config
    /// file. Writes the error message to `errorMessage` if errors occur and
    /// `errorMessage` is not nullptr.
    LogicalResult
    loadFromFileContent(llvm::StringRef fileContent,
                        std::string *errorMessage = (std::string *)nullptr) {
      if (frozen) {
        llvm::errs() << "Config is frozen\n";
        return failure();
      }

      for (unsigned lineNr = 1; !fileContent.empty(); ++lineNr) {
        StringRef line, rest;
        std::tie(line, rest) = fileContent.split("\n");
        fileContent = rest;
        line = line.trim();
        if (line.empty())
          continue;

        StringRef key, value;
        std::tie(key, value) = line.split("=");
        key = key.trim();
        value = value.trim();
        if (!entries.contains(key)) {
          if (errorMessage)
            *errorMessage = "unknown key at line " + std::to_string(lineNr);
          return failure();
        }

        if (entries[key].parseString(value).failed()) {
          if (errorMessage)
            *errorMessage =
                "failed to parse value at line " + std::to_string(lineNr);
          return failure();
        }
      }

      return success();
    }

    /// Dumps the current configuration to an output stream.
    void dump(llvm::raw_ostream &os) {
      // Sort configs for nicer print.
      std::vector<llvm::StringMapEntry<Entry> *> entryVec;
      entryVec.reserve(entries.size());
      for (llvm::StringMapEntry<Entry> &entry : entries)
        entryVec.push_back(&entry);

      llvm::sort(entryVec, [](const llvm::StringMapEntry<Entry> *a,
                              const llvm::StringMapEntry<Entry> *b) {
        return a->getKey().compare(b->getKey()) < 0;
      });

      for (llvm::StringMapEntry<Entry> *e : entryVec)
        os << e->getKey() << " = " << e->getValue() << "\n";
    }

    /// Freezes this configuration to prevent any changes and replaces negative
    /// operation probabilities with default probability.
    void freeze() {
      // Replace unspecified probabilites by default probability.
      unsigned defaultProb = get<unsigned>("_gen.defaultProb").value();

      for (RegisteredOperationName ron : context->getRegisteredOperations())
        if (ron.hasInterface<GeneratableOpInterface>() &&
            get<int>(ron.getStringRef()).value() < 0)
          (void)set<int>(ron.getStringRef(), defaultProb);

      frozen = true;
    }

    /// Registers configs from registered operations.
    void registerOpConfigs();

  private:
    MLIRContext *context;
    llvm::StringMap<Entry> entries = {};
    bool frozen = false;
  };

  /// The configuration used for IR generation.
  Config config;

  //===--------------------------------------------------------------------===//
  // GeneratorOpBuilder
  //===--------------------------------------------------------------------===//

  explicit GeneratorOpBuilder(Config config);

  // Disallow generators to move the insertion point up.
  void setInsertionPoint(Block *block, Block::iterator insertPoint) = delete;
  void setInsertionPoint(Operation *op) = delete;
  void setInsertionPointAfter(Operation *op) = delete;
  void setInsertionPointAfterValue(Value val) = delete;
  void setInsertionPointToStart(Block *block) = delete;

  // FIXME: Constrain this as well.
  // void setInsertionPointToEnd(Block *block) = delete;

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

  /// Creates an operation given the fields represented as an OperationState
  /// and enforces generator constraints.
  Operation *create(const OperationState &state);

  /// Returns a list with `num` unknown locations.
  llvm::SmallVector<Location> getUnknownLocs(unsigned num);

  /// Adds result types to an exisitng operation by creating a clone above the
  /// operation. Make sure to erase the original operation.
  Operation *addResultTypes(Operation *op, llvm::ArrayRef<Type> resultTypes);

  //===--------------------------------------------------------------------===//
  // Collectors
  //===--------------------------------------------------------------------===//

  /// Returns a list of all available values in the current location.
  llvm::SmallVector<Value>
  collectValues(std::function<bool(const Value &)> filter = nullptr);

  /// Returns a list of all available types in the current location.
  llvm::SmallVector<Type>
  collectTypes(std::function<bool(const Type &)> filter = nullptr);

  /// Returns a list of all available symbols in the current location.
  llvm::SmallVector<llvm::StringRef> collectSymbols(
      std::function<bool(const Operation &, const llvm::StringRef &)> filter =
          nullptr);

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

  /// Samples multiple times from a vector of choices. If no probability
  /// vector is provided, it samples using a uniform distribution.
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

  /// Returns a random number using normal distribution.
  template <typename T>
  T sampleNormal() {
    static_assert(std::is_arithmetic<T>::value, "Numeric type required");

    std::normal_distribution<> dist(
        config.get<double>("_gen.normalMean").value(),
        config.get<double>("_gen.normalStddev").value());
    if constexpr (std::is_integral<T>::value) {
      return static_cast<T>(std::round(dist(rng)));
    } else if constexpr (std::is_floating_point<T>::value) {
      return static_cast<T>(dist(rng));
    }

    // Default return value, should never be reached.
    return T();
  }

  /// Returns a random number using geometric distribution.
  template <typename T>
  T sampleGeometric() {
    static_assert(std::is_arithmetic<T>::value, "Numeric type required");

    std::geometric_distribution<> dist(
        config.get<double>("_gen.geometricProb").value());
    if constexpr (std::is_integral<T>::value) {
      return static_cast<T>(std::round(dist(rng)));
    } else if constexpr (std::is_floating_point<T>::value) {
      return static_cast<T>(dist(rng));
    }

    // Default return value, should never be reached.
    return T();
  }

  /// Returns a random string of at least length one using alphanumeric
  /// characters. The string will start with an alphabetical character and use
  /// a geometric distribution for the length.
  std::string sampleString();

  /// Returns a list of values using the filters.
  llvm::Optional<llvm::SmallVector<Value>>
  sampleValues(llvm::SmallVector<std::function<bool(const Value &)>> filters);

  /// Returns a value using the filter.
  llvm::Optional<Value>
  sampleValue(std::function<bool(const Value &)> filter = nullptr);

  /// Returns a list of values of the provided types.
  llvm::Optional<llvm::SmallVector<Value>>
  sampleValuesOfTypes(llvm::SmallVector<Type> types, bool unusedFirst = false);

  /// Returns a value of the provided type.
  llvm::Optional<Value> sampleValueOfType(Type type, bool unusedFirst = false);

  /// Samples from a geometric distribution of available values in the current
  /// position.
  llvm::SmallVector<Value>
  sampleValues(unsigned min = 0,
               std::function<bool(const Value &)> filter = nullptr);

  /// Samples from a geometric distribution of available types in the current
  /// position.
  llvm::SmallVector<Type>
  sampleTypes(unsigned min = 0,
              std::function<bool(const Type &)> filter = nullptr);

  /// Returns a symbol using the filter.
  llvm::Optional<llvm::StringRef> sampleSymbol(
      std::function<bool(const Operation &, const llvm::StringRef &)> filter =
          nullptr);

  //===--------------------------------------------------------------------===//
  // Generators
  //===--------------------------------------------------------------------===//

  /// Utility function to generate an operation.
  Operation *generate(RegisteredOperationName ron);

  /// Randomly generates an operation in the current position and returns it.
  /// Returns nullptr if it fails.
  Operation *generateOperation(llvm::SmallVector<RegisteredOperationName> ops);

  /// Randomly generates a terminator operation in the current position and
  /// returns it. Returns nullptr if it fails.
  Operation *generateTerminator();

  /// Fills the current block with operations.
  LogicalResult generateBlock(Block *block, bool ensureTerminator = false);

private:
  /// Utility function to get operation probabilities;
  unsigned getProb(RegisteredOperationName ron);

  /// Random number generator.
  std::mt19937 rng;

  /// All operations that can be generated.
  llvm::SmallVector<RegisteredOperationName> availableOps = {};
}; // namespace mlir

} // namespace mlir

#endif // MLIR_IR_GENERATOROPBUILDER_H
