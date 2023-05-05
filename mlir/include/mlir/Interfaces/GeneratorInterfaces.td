//===- GeneratorInterfaces.td - Generator Interfaces -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a set of interfaces that can be used to define information
// related to operations usable for random MLIR program generation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_GENERATORINTERFACES
#define MLIR_INTERFACES_GENERATORINTERFACES

include "mlir/IR/OpBase.td"

def GeneratorInterface : OpInterface<"GeneratorInterface"> {
  let description = [{
    This interface is designed for operations that support the generation of 
    random instances with random attribute values and operand types. The purpose
    of this interface is to enable the creation of random MLIR programs, akin to
    the functionality provided by CSmith for C programs. Operations implementing
    this interface can be used in tools like MLIR-Smith to generate diverse and 
    complex MLIR programs for various purposes, including testing, benchmarking,
    and compiler optimization evaluations.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    StaticInterfaceMethod<
      /*desc=*/"Attempts to generate this operation at the current location",
      /*retTy=*/"::mlir::LogicalResult",
      /*methodName=*/"generate",
      /*args=*/(ins "::mlir::PatternRewriter &":$rewriter)
    >,
  ];
}

#endif // MLIR_INTERFACES_GENERATORINTERFACES
