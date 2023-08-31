//===- GeneratableOpInterface.td - Generatable interface ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface for generatable operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_GENERATABLEOPINTERFACE
#define MLIR_INTERFACES_GENERATABLEOPINTERFACE

include "mlir/IR/OpBase.td"

def GeneratableOpInterface : OpInterface<"GeneratableOpInterface"> {
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
      /*desc=*/"Attempts to generate this operation at the current location and returns it. On failure returns nullptr.",
      /*retTy=*/"::mlir::Operation *",
      /*methodName=*/"generate",
      /*args=*/(ins "::mlir::GeneratorOpBuilder &":$builder)
    >,
  ];
}

#endif // MLIR_INTERFACES_GENERATABLEOPINTERFACE
