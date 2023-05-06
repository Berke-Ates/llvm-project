//===- MlirSmithMain.cpp - MLIR Program Generator -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a utility that randomly generates MLIR programs and prints the result
// back out. It is designed to support fuzz testing.
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-smith/MlirSmithMain.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/GeneratorInterfaces.h"

using namespace mlir;

/// A simple pattern rewriter that can be constructed from a context.
class TrivialPatternRewriter : public PatternRewriter {
public:
  explicit TrivialPatternRewriter(MLIRContext *context)
      : PatternRewriter(context) {}
};

LogicalResult mlir::mlirSmithMain(int argc, char **argv,
                                  DialectRegistry &registry) {
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  // Collect all operations usable for generation
  llvm::SmallVector<detail::GeneratorInterfaceInterfaceTraits::Concept *>
      available_ops;

  for (RegisteredOperationName ron : context.getRegisteredOperations())
    if (ron.hasInterface<GeneratorInterface>())
      available_ops.push_back(ron.getInterface<GeneratorInterface>());

  TrivialPatternRewriter rewriter(&context);

  for (auto a : available_ops) {
    a->generate(rewriter);
  }

  return success();
}
