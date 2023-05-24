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

  static cl::opt<bool> shouldDumpConfig("dump", cl::desc("Dump config"),
                                        cl::init(false),
                                        cl::cat(mlirSmithCategory));

  static cl::opt<unsigned> seedOpt(
      "seed", cl::desc("Random number generator seed"), cl::value_desc("seed"),
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

  // Load configuration.
  MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  GeneratorOpBuilderConfig config;
  config.loadDefaultValues(&ctx);

  if (!configFilename.empty()) {
    auto configFile = openInputFile(configFilename.getValue(), &errorMessage);
    if (!configFile) {
      llvm::errs() << errorMessage << "\n";
      return failure();
    }

    if (config.loadFromFileContent(configFile->getBuffer(), &errorMessage)
            .failed()) {
      llvm::errs() << errorMessage << "\n";
      return failure();
    }
  }

  // Override configuration values with CLI values
  if (seedOpt.getNumOccurrences() > 0)
    config.Seed = seedOpt.getValue();

  // Dump config.
  if (shouldDumpConfig) {
    config.dumpConfig(output->os());
    output->keep();
    return success();
  }

  // Start generation.
  GeneratorOpBuilder builder(&ctx, config);
  Location loc = builder.getUnknownLoc();

  // IDEA: Could move this to the operations themselves
  // Create top-level module.
  OperationState moduleState(loc, ModuleOp::getOperationName());
  ModuleOp::build(builder, moduleState);
  Operation *moduleOp = builder.create(moduleState);
  if (moduleOp == nullptr) {
    llvm::errs() << "failed to generate top-level module operation\n";
    return failure();
  }

  ModuleOp module = cast<ModuleOp>(moduleOp);
  builder.setInsertionPointToStart(module.getBody());

  // Create main function.
  OperationState funcState(loc, func::FuncOp::getOperationName());
  SmallVector<Type> retTypes = builder.sampleTypes();
  FunctionType funcType = builder.getFunctionType({}, retTypes);
  func::FuncOp::build(builder, funcState, "main", funcType);
  Operation *funcOp = builder.create(funcState);

  if (funcOp != nullptr) {
    func::FuncOp func = cast<func::FuncOp>(funcOp);
    builder.setInsertionPointToStart(func.addEntryBlock());

    // Generate main function body.
    LogicalResult funcBodyRes =
        builder.generateBlock(/*ensureTerminator=*/true,
                              /*requiredTypes=*/retTypes);
    if (funcBodyRes.failed())
      llvm::errs() << "failed to generate main function body\n";
  }

  // Output the result.
  module.print(output->os());

  // Keep the output file if the generation was successful.
  output->keep();
  return success();
}
