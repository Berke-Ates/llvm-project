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
#include "mlir/IR/GeneratorOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using namespace llvm;

LogicalResult
mlir::mlirSmithMain(int argc, char **argv, DialectRegistry &registry,
                    std::function<Operation *(GeneratorOpBuilder &)> entry) {
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

  static cl::opt<bool> shouldCollectMetrics(
      "metrics", cl::desc("Collect Metrics: Code (B), Time (µs)"),
      cl::init(false), cl::cat(mlirSmithCategory));

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

  // Load dialects.
  MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  // Load configuration.
  GeneratorOpBuilder::Config config(&ctx);

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
    (void)config.set<unsigned>("_gen.seed", seedOpt.getValue());

  // Dump config.
  if (shouldDumpConfig) {
    config.freeze();
    config.dump(output->os());
    output->keep();
    return success();
  }

  // Collect metrics.
  auto startTime = std::chrono::high_resolution_clock::now();

  // Start generation.
  GeneratorOpBuilder builder(config);
  Location loc = builder.getUnknownLoc();

  // Create top-level module.
  OperationState moduleState(loc, ModuleOp::getOperationName());
  ModuleOp::build(builder, moduleState);
  Operation *moduleOp = builder.create(moduleState);
  if (moduleOp == nullptr) {
    llvm::errs() << "failed to generate top-level module operation\n";
    return failure();
  }

  ModuleOp module = cast<ModuleOp>(moduleOp);
  builder.setInsertionPointToEnd(module.getBody());
  if (!entry(builder)) {
    llvm::errs() << "failed to generate module body\n";
    module.erase();
    return failure();
  }

  // Verify generated MLIR.
  LogicalResult result = verify(module);

  // Collect metrics.
  auto stopTime = std::chrono::high_resolution_clock::now();

  // Print code if not collecting metrics.
  unsigned codeBytes = 0;
  if (shouldCollectMetrics) {
    std::string genCode;
    llvm::raw_string_ostream genCodeStream(genCode);
    module.print(genCodeStream);
    codeBytes = genCode.size();
  } else {
    module.print(output->os());
  }

  // Cleanup.
  module.erase();

  // Print metrics.
  if (shouldCollectMetrics) {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        stopTime - startTime);
    output->os() << codeBytes << ", " << duration.count() << "\n";
  }

  output->keep();
  return result;
}
