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
  Location loc = builder.getUnknownLoc();

  // Create top-level module & func
  OperationState moduleState(loc, ModuleOp::getOperationName());
  ModuleOp::build(builder, moduleState);
  ModuleOp module = cast<ModuleOp>(builder.create(moduleState));
  builder.setInsertionPointToStart(module.getBody());

  OperationState funcState(loc, func::FuncOp::getOperationName());
  FunctionType retType = builder.getFunctionType({}, builder.sampleTypeRange());
  func::FuncOp::build(builder, funcState, "main", retType);
  func::FuncOp funcOp = cast<func::FuncOp>(builder.create(funcState));
  builder.setInsertionPointToStart(funcOp.addEntryBlock());

  if (builder.generateRegion(/*requiresTerminator=*/true).failed())
    return failure();

  // Print result
  module.print(llvm::outs());
  return success();
}
