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

#include "mlir/Support/LogicalResult.h"

namespace mlir {
class DialectRegistry;

LogicalResult mlirSmithMain(int argc, char **argv, DialectRegistry &registry);

} // namespace mlir

#endif // MLIR_TOOLS_MLIR_SMITH_MLIRSMITHMAIN_H
