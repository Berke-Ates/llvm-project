//===- GeneratableOpInterface.cpp - Generatable operations interface ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/GeneratableOpInterface.h"

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

  /// Generates a region until a terminator is generated (if required)
  LogicalResult generateRegion();

  GeneratorOpBuilder &builder;

  llvm::SmallVector<detail::GeneratableOpInterfaceInterfaceTraits::Concept *>
      available_ops;
};
} // namespace detail
} // namespace mlir

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

LogicalResult GeneratorOpBuilder::generateRegion() {
  return impl->generateRegion();
}

//===----------------------------------------------------------------------===//
// Table-generated class definitions
//===----------------------------------------------------------------------===//

/// Include the definitions of the copy operation interface.
#include "mlir/Interfaces/GeneratableOpInterface.cpp.inc"
