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
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/GeneratableOpInterface.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using namespace llvm;

LogicalResult mlir::mlirSmithMain(int argc, char **argv,
                                  DialectRegistry &registry) {
  static cl::OptionCategory mlirSmithCategory("mlir-smith options");

  static cl::opt<std::string> outputFilename(
      "o", cl::desc("Output filename"), cl::value_desc("filename"),
      cl::init("-"), cl::cat(mlirSmithCategory));

  static cl::opt<std::string> configFilename(
      "c", cl::desc("Config filename"), cl::value_desc("filename"),
      cl::Optional, cl::cat(mlirSmithCategory));

  cl::HideUnrelatedOptions(mlirSmithCategory);
  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "MLIR generation tool");

  // Set up the output file.
  std::string errorMessage;
  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  // Load configuration
  if (!configFilename.empty()) {
    std::string configFile = configFilename.getValue();
    auto config = openInputFile(configFile, &errorMessage);
    if (!config) {
      llvm::errs() << errorMessage << "\n";
      return failure();
    }

    // TODO: Load config file and setup configuration
  }

  // Load dialects
  MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  GeneratorOpBuilder builder(&ctx);
  Location loc = builder.getUnknownLoc();

  // Create top-level module and main function.
  OperationState moduleState(loc, ModuleOp::getOperationName());
  ModuleOp::build(builder, moduleState);
  ModuleOp module = cast<ModuleOp>(builder.create(moduleState));
  builder.setInsertionPointToStart(module.getBody());

  OperationState funcState(loc, func::FuncOp::getOperationName());
  FunctionType retType = builder.getFunctionType({}, builder.sampleTypeRange());
  func::FuncOp::build(builder, funcState, "main", retType);
  func::FuncOp funcOp = cast<func::FuncOp>(builder.create(funcState));
  builder.setInsertionPointToStart(funcOp.addEntryBlock());

  // Generate main function body.
  if (builder.generateRegion(/*requiresTerminator=*/true).failed())
    return failure();

  module.print(output->os());

  // Keep the output file if the generation was successful.
  output->keep();
  return success();
}
