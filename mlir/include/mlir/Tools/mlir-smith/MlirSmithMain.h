//===- MlirSmithMain.h - MLIR Program Generator main ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-smith for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIR_SMITH_MLIRSMITHMAIN_H
#define MLIR_TOOLS_MLIR_SMITH_MLIRSMITHMAIN_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include <functional>

namespace mlir {
class DialectRegistry;
class Operation;
class GeneratorOpBuilder;

LogicalResult mlirSmithMain(int argc, char **argv, DialectRegistry &registry,
                            std::function<Operation *(GeneratorOpBuilder &)>
                                entry = func::FuncOp::generate);

} // namespace mlir

#endif // MLIR_TOOLS_MLIR_SMITH_MLIRSMITHMAIN_H
