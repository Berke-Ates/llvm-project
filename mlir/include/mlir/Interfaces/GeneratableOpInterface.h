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
// Generation OpBuilder
//===----------------------------------------------------------------------===//

namespace detail {
struct GeneratorOpBuilderImpl;
} // namespace detail

/// This class implements a builder for use in generation functions. It
/// extends the base OpBuilder and provides utility functions to generate common
/// structures and to sample from a configured distribution.
class GeneratorOpBuilder final : public OpBuilder {
public:
  explicit GeneratorOpBuilder(MLIRContext *ctxt);
  ~GeneratorOpBuilder();

  /// Generates a region until a terminator is generated (if required)
  LogicalResult generateRegion();

private:
  std::unique_ptr<detail::GeneratorOpBuilderImpl> impl;
};

} // namespace mlir

//===----------------------------------------------------------------------===//
// Generated Interface Declarations
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/GeneratableOpInterface.h.inc"

#endif // MLIR_INTERFACES_GENERATABLEOPINTERFACE_H
