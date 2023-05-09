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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/GeneratableOpInterface.h"

using namespace mlir;

LogicalResult mlir::mlirSmithMain(int argc, char **argv,
                                  DialectRegistry &registry) {
  // TODO: Implement CLI arguments
  // TODO: Load config file
  MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  GeneratorOpBuilder builder(&ctx);

  // Create top-level module & func
  OperationState moduleState(builder.getUnknownLoc(),
                             ModuleOp::getOperationName());
  ModuleOp::build(builder, moduleState);
  ModuleOp module = cast<ModuleOp>(builder.create(moduleState));
  builder.setInsertionPointToStart(module.getBody());

  // FIXME: Generate return types
  OperationState funcState(builder.getUnknownLoc(),
                           func::FuncOp::getOperationName());
  func::FuncOp::build(builder, funcState, "main",
                      builder.getFunctionType({}, {}));
  func::FuncOp funcOp = cast<func::FuncOp>(builder.create(funcState));
  builder.setInsertionPointToStart(funcOp.addEntryBlock());

  if (builder.generateRegion().failed())
    return failure();

  // Insert return operation
  // FIXME: Should choose types to fit function signature
  OperationState returnState(builder.getUnknownLoc(),
                             func::ReturnOp::getOperationName());
  func::ReturnOp::build(builder, returnState, {});
  builder.create(returnState);

  // Print result
  module.print(llvm::outs());
  return success();
}
