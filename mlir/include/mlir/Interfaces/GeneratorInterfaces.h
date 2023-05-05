//===- GeneratorInterfaces.h - Generator Interfaces for MLIR ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the generator interfaces defined in
// `GeneratorInterfaces.td`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_GENERATORINTERFACES_H
#define MLIR_INTERFACES_GENERATORINTERFACES_H

#include "mlir/IR/PatternMatch.h"

/// Include the generated interface declarations.
#include "mlir/Interfaces/GeneratorInterfaces.h.inc"

#endif // MLIR_INTERFACES_GENERATORINTERFACES_H
