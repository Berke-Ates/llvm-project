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

LogicalResult mlir::mlirSmithMain(int argc, char **argv,
                                  DialectRegistry &registry) {
  printf("entered main\n");

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  for (RegisteredOperationName registeredOperationName :
       context.getRegisteredOperations()) {
    if (registeredOperationName.hasInterface<GeneratorInterface>()) {
      registeredOperationName.dump();
      printf("\n");
    }
  }

  return success();
}
