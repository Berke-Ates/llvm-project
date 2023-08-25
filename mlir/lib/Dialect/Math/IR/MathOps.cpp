//===- MathOps.cpp - MLIR operations for math implementation --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include <optional>

using namespace mlir;
using namespace mlir::math;

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Math/IR/MathOps.cpp.inc"

//===----------------------------------------------------------------------===//
// AbsFOp
//===----------------------------------------------------------------------===//

OpFoldResult math::AbsFOp::fold(FoldAdaptor adaptor) {
  return constFoldUnaryOp<FloatAttr>(adaptor.getOperands(),
                                     [](const APFloat &a) { return abs(a); });
}

LogicalResult math::AbsFOp::generate(GeneratorOpBuilder &builder) {
  llvm::SmallVector<Type> possibleTypes = {
      builder.getF16Type(),
      builder.getF32Type(),
      builder.getF64Type(),
  };

  while (!possibleTypes.empty()) {
    unsigned idx = builder.sampleUniform(possibleTypes.size() - 1);
    Type resultType = possibleTypes[idx];

    llvm::Optional<Value> lhs = builder.sampleValueOfType(resultType);

    if (!lhs.has_value()) {
      Type *it = llvm::find(possibleTypes, resultType);
      if (it != possibleTypes.end())
        possibleTypes.erase(it);
      continue;
    }

    OperationState state(builder.getUnknownLoc(),
                         math::AbsFOp::getOperationName());
    math::AbsFOp::build(builder, state, lhs.value());
    if (builder.create(state) != nullptr)
      return success();

    Type *it = llvm::find(possibleTypes, resultType);
    if (it != possibleTypes.end())
      possibleTypes.erase(it);
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// AbsIOp
//===----------------------------------------------------------------------===//

OpFoldResult math::AbsIOp::fold(FoldAdaptor adaptor) {
  return constFoldUnaryOp<IntegerAttr>(adaptor.getOperands(),
                                       [](const APInt &a) { return a.abs(); });
}

LogicalResult math::AbsIOp::generate(GeneratorOpBuilder &builder) {
  llvm::SmallVector<Type> possibleTypes = {
      builder.getI1Type(),  builder.getIndexType(), builder.getI8Type(),
      builder.getI16Type(), builder.getI32Type(),   builder.getI64Type(),
  };

  while (!possibleTypes.empty()) {
    unsigned idx = builder.sampleUniform(possibleTypes.size() - 1);
    Type resultType = possibleTypes[idx];

    llvm::Optional<Value> lhs = builder.sampleValueOfType(resultType);

    if (!lhs.has_value()) {
      Type *it = llvm::find(possibleTypes, resultType);
      if (it != possibleTypes.end())
        possibleTypes.erase(it);
      continue;
    }

    OperationState state(builder.getUnknownLoc(),
                         math::AbsIOp::getOperationName());
    math::AbsIOp::build(builder, state, lhs.value());
    if (builder.create(state) != nullptr)
      return success();

    Type *it = llvm::find(possibleTypes, resultType);
    if (it != possibleTypes.end())
      possibleTypes.erase(it);
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// AtanOp
//===----------------------------------------------------------------------===//

OpFoldResult math::AtanOp::fold(FoldAdaptor adaptor) {
  return constFoldUnaryOpConditional<FloatAttr>(
      adaptor.getOperands(), [](const APFloat &a) -> std::optional<APFloat> {
        switch (a.getSizeInBits(a.getSemantics())) {
        case 64:
          return APFloat(atan(a.convertToDouble()));
        case 32:
          return APFloat(atanf(a.convertToFloat()));
        default:
          return {};
        }
      });
}

LogicalResult math::AtanOp::generate(GeneratorOpBuilder &builder) {
  llvm::SmallVector<Type> possibleTypes = {
      builder.getF16Type(),
      builder.getF32Type(),
      builder.getF64Type(),
  };

  while (!possibleTypes.empty()) {
    unsigned idx = builder.sampleUniform(possibleTypes.size() - 1);
    Type resultType = possibleTypes[idx];

    llvm::Optional<Value> lhs = builder.sampleValueOfType(resultType);

    if (!lhs.has_value()) {
      Type *it = llvm::find(possibleTypes, resultType);
      if (it != possibleTypes.end())
        possibleTypes.erase(it);
      continue;
    }

    OperationState state(builder.getUnknownLoc(),
                         math::AtanOp::getOperationName());
    math::AtanOp::build(builder, state, lhs.value());
    if (builder.create(state) != nullptr)
      return success();

    Type *it = llvm::find(possibleTypes, resultType);
    if (it != possibleTypes.end())
      possibleTypes.erase(it);
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// Atan2Op
//===----------------------------------------------------------------------===//

OpFoldResult math::Atan2Op::fold(FoldAdaptor adaptor) {
  return constFoldBinaryOpConditional<FloatAttr>(
      adaptor.getOperands(),
      [](const APFloat &a, const APFloat &b) -> std::optional<APFloat> {
        if (a.isZero() && b.isZero())
          return llvm::APFloat::getNaN(a.getSemantics());

        if (a.getSizeInBits(a.getSemantics()) == 64 &&
            b.getSizeInBits(b.getSemantics()) == 64)
          return APFloat(atan2(a.convertToDouble(), b.convertToDouble()));

        if (a.getSizeInBits(a.getSemantics()) == 32 &&
            b.getSizeInBits(b.getSemantics()) == 32)
          return APFloat(atan2f(a.convertToFloat(), b.convertToFloat()));

        return {};
      });
}

LogicalResult math::Atan2Op::generate(GeneratorOpBuilder &builder) {
  llvm::SmallVector<Type> possibleTypes = {
      builder.getF16Type(),
      builder.getF32Type(),
      builder.getF64Type(),
  };

  while (!possibleTypes.empty()) {
    unsigned idx = builder.sampleUniform(possibleTypes.size() - 1);
    Type resultType = possibleTypes[idx];

    llvm::Optional<Value> lhs = builder.sampleValueOfType(resultType);
    llvm::Optional<Value> rhs = builder.sampleValueOfType(resultType);

    if (!lhs.has_value() || !rhs.has_value()) {
      Type *it = llvm::find(possibleTypes, resultType);
      if (it != possibleTypes.end())
        possibleTypes.erase(it);
      continue;
    }

    OperationState state(builder.getUnknownLoc(),
                         math::Atan2Op::getOperationName());
    math::Atan2Op::build(builder, state, lhs.value(), rhs.value());
    if (builder.create(state) != nullptr)
      return success();

    Type *it = llvm::find(possibleTypes, resultType);
    if (it != possibleTypes.end())
      possibleTypes.erase(it);
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// CeilOp
//===----------------------------------------------------------------------===//

OpFoldResult math::CeilOp::fold(FoldAdaptor adaptor) {
  return constFoldUnaryOp<FloatAttr>(
      adaptor.getOperands(), [](const APFloat &a) {
        APFloat result(a);
        result.roundToIntegral(llvm::RoundingMode::TowardPositive);
        return result;
      });
}

LogicalResult math::CeilOp::generate(GeneratorOpBuilder &builder) {
  llvm::SmallVector<Type> possibleTypes = {
      builder.getF16Type(),
      builder.getF32Type(),
      builder.getF64Type(),
  };

  while (!possibleTypes.empty()) {
    unsigned idx = builder.sampleUniform(possibleTypes.size() - 1);
    Type resultType = possibleTypes[idx];

    llvm::Optional<Value> lhs = builder.sampleValueOfType(resultType);

    if (!lhs.has_value()) {
      Type *it = llvm::find(possibleTypes, resultType);
      if (it != possibleTypes.end())
        possibleTypes.erase(it);
      continue;
    }

    OperationState state(builder.getUnknownLoc(),
                         math::CeilOp::getOperationName());
    math::CeilOp::build(builder, state, lhs.value());
    if (builder.create(state) != nullptr)
      return success();

    Type *it = llvm::find(possibleTypes, resultType);
    if (it != possibleTypes.end())
      possibleTypes.erase(it);
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// CopySignOp
//===----------------------------------------------------------------------===//

OpFoldResult math::CopySignOp::fold(FoldAdaptor adaptor) {
  return constFoldBinaryOp<FloatAttr>(adaptor.getOperands(),
                                      [](const APFloat &a, const APFloat &b) {
                                        APFloat result(a);
                                        result.copySign(b);
                                        return result;
                                      });
}

LogicalResult math::CopySignOp::generate(GeneratorOpBuilder &builder) {
  llvm::SmallVector<Type> possibleTypes = {
      builder.getF16Type(),
      builder.getF32Type(),
      builder.getF64Type(),
  };

  while (!possibleTypes.empty()) {
    unsigned idx = builder.sampleUniform(possibleTypes.size() - 1);
    Type resultType = possibleTypes[idx];

    llvm::Optional<Value> lhs = builder.sampleValueOfType(resultType);
    llvm::Optional<Value> rhs = builder.sampleValueOfType(resultType);

    if (!lhs.has_value() || !rhs.has_value()) {
      Type *it = llvm::find(possibleTypes, resultType);
      if (it != possibleTypes.end())
        possibleTypes.erase(it);
      continue;
    }

    OperationState state(builder.getUnknownLoc(),
                         math::CopySignOp::getOperationName());
    math::CopySignOp::build(builder, state, lhs.value(), rhs.value());
    if (builder.create(state) != nullptr)
      return success();

    Type *it = llvm::find(possibleTypes, resultType);
    if (it != possibleTypes.end())
      possibleTypes.erase(it);
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// CosOp
//===----------------------------------------------------------------------===//

OpFoldResult math::CosOp::fold(FoldAdaptor adaptor) {
  return constFoldUnaryOpConditional<FloatAttr>(
      adaptor.getOperands(), [](const APFloat &a) -> std::optional<APFloat> {
        switch (a.getSizeInBits(a.getSemantics())) {
        case 64:
          return APFloat(cos(a.convertToDouble()));
        case 32:
          return APFloat(cosf(a.convertToFloat()));
        default:
          return {};
        }
      });
}

LogicalResult math::CosOp::generate(GeneratorOpBuilder &builder) {
  llvm::SmallVector<Type> possibleTypes = {
      builder.getF16Type(),
      builder.getF32Type(),
      builder.getF64Type(),
  };

  while (!possibleTypes.empty()) {
    unsigned idx = builder.sampleUniform(possibleTypes.size() - 1);
    Type resultType = possibleTypes[idx];

    llvm::Optional<Value> lhs = builder.sampleValueOfType(resultType);

    if (!lhs.has_value()) {
      Type *it = llvm::find(possibleTypes, resultType);
      if (it != possibleTypes.end())
        possibleTypes.erase(it);
      continue;
    }

    OperationState state(builder.getUnknownLoc(),
                         math::CosOp::getOperationName());
    math::CosOp::build(builder, state, lhs.value());
    if (builder.create(state) != nullptr)
      return success();

    Type *it = llvm::find(possibleTypes, resultType);
    if (it != possibleTypes.end())
      possibleTypes.erase(it);
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// SinOp
//===----------------------------------------------------------------------===//

OpFoldResult math::SinOp::fold(FoldAdaptor adaptor) {
  return constFoldUnaryOpConditional<FloatAttr>(
      adaptor.getOperands(), [](const APFloat &a) -> std::optional<APFloat> {
        switch (a.getSizeInBits(a.getSemantics())) {
        case 64:
          return APFloat(sin(a.convertToDouble()));
        case 32:
          return APFloat(sinf(a.convertToFloat()));
        default:
          return {};
        }
      });
}

LogicalResult math::SinOp::generate(GeneratorOpBuilder &builder) {
  llvm::SmallVector<Type> possibleTypes = {
      builder.getF16Type(),
      builder.getF32Type(),
      builder.getF64Type(),
  };

  while (!possibleTypes.empty()) {
    unsigned idx = builder.sampleUniform(possibleTypes.size() - 1);
    Type resultType = possibleTypes[idx];

    llvm::Optional<Value> lhs = builder.sampleValueOfType(resultType);

    if (!lhs.has_value()) {
      Type *it = llvm::find(possibleTypes, resultType);
      if (it != possibleTypes.end())
        possibleTypes.erase(it);
      continue;
    }

    OperationState state(builder.getUnknownLoc(),
                         math::SinOp::getOperationName());
    math::SinOp::build(builder, state, lhs.value());
    if (builder.create(state) != nullptr)
      return success();

    Type *it = llvm::find(possibleTypes, resultType);
    if (it != possibleTypes.end())
      possibleTypes.erase(it);
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// CountLeadingZerosOp
//===----------------------------------------------------------------------===//

OpFoldResult math::CountLeadingZerosOp::fold(FoldAdaptor adaptor) {
  return constFoldUnaryOp<IntegerAttr>(
      adaptor.getOperands(),
      [](const APInt &a) { return APInt(a.getBitWidth(), a.countl_zero()); });
}

LogicalResult math::CountLeadingZerosOp::generate(GeneratorOpBuilder &builder) {
  llvm::SmallVector<Type> possibleTypes = {
      builder.getI1Type(),  builder.getIndexType(), builder.getI8Type(),
      builder.getI16Type(), builder.getI32Type(),   builder.getI64Type(),
  };

  while (!possibleTypes.empty()) {
    unsigned idx = builder.sampleUniform(possibleTypes.size() - 1);
    Type resultType = possibleTypes[idx];

    llvm::Optional<Value> lhs = builder.sampleValueOfType(resultType);

    if (!lhs.has_value()) {
      Type *it = llvm::find(possibleTypes, resultType);
      if (it != possibleTypes.end())
        possibleTypes.erase(it);
      continue;
    }

    OperationState state(builder.getUnknownLoc(),
                         math::CountLeadingZerosOp::getOperationName());
    math::CountLeadingZerosOp::build(builder, state, lhs.value());
    if (builder.create(state) != nullptr)
      return success();

    Type *it = llvm::find(possibleTypes, resultType);
    if (it != possibleTypes.end())
      possibleTypes.erase(it);
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// CountTrailingZerosOp
//===----------------------------------------------------------------------===//

OpFoldResult math::CountTrailingZerosOp::fold(FoldAdaptor adaptor) {
  return constFoldUnaryOp<IntegerAttr>(
      adaptor.getOperands(),
      [](const APInt &a) { return APInt(a.getBitWidth(), a.countr_zero()); });
}

LogicalResult
math::CountTrailingZerosOp::generate(GeneratorOpBuilder &builder) {
  llvm::SmallVector<Type> possibleTypes = {
      builder.getI1Type(),  builder.getIndexType(), builder.getI8Type(),
      builder.getI16Type(), builder.getI32Type(),   builder.getI64Type(),
  };

  while (!possibleTypes.empty()) {
    unsigned idx = builder.sampleUniform(possibleTypes.size() - 1);
    Type resultType = possibleTypes[idx];

    llvm::Optional<Value> lhs = builder.sampleValueOfType(resultType);

    if (!lhs.has_value()) {
      Type *it = llvm::find(possibleTypes, resultType);
      if (it != possibleTypes.end())
        possibleTypes.erase(it);
      continue;
    }

    OperationState state(builder.getUnknownLoc(),
                         math::CountTrailingZerosOp::getOperationName());
    math::CountTrailingZerosOp::build(builder, state, lhs.value());
    if (builder.create(state) != nullptr)
      return success();

    Type *it = llvm::find(possibleTypes, resultType);
    if (it != possibleTypes.end())
      possibleTypes.erase(it);
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// CtPopOp
//===----------------------------------------------------------------------===//

OpFoldResult math::CtPopOp::fold(FoldAdaptor adaptor) {
  return constFoldUnaryOp<IntegerAttr>(
      adaptor.getOperands(),
      [](const APInt &a) { return APInt(a.getBitWidth(), a.popcount()); });
}

LogicalResult math::CtPopOp::generate(GeneratorOpBuilder &builder) {
  llvm::SmallVector<Type> possibleTypes = {
      builder.getI1Type(),  builder.getIndexType(), builder.getI8Type(),
      builder.getI16Type(), builder.getI32Type(),   builder.getI64Type(),
  };

  while (!possibleTypes.empty()) {
    unsigned idx = builder.sampleUniform(possibleTypes.size() - 1);
    Type resultType = possibleTypes[idx];

    llvm::Optional<Value> lhs = builder.sampleValueOfType(resultType);

    if (!lhs.has_value()) {
      Type *it = llvm::find(possibleTypes, resultType);
      if (it != possibleTypes.end())
        possibleTypes.erase(it);
      continue;
    }

    OperationState state(builder.getUnknownLoc(),
                         math::CtPopOp::getOperationName());
    math::CtPopOp::build(builder, state, lhs.value());
    if (builder.create(state) != nullptr)
      return success();

    Type *it = llvm::find(possibleTypes, resultType);
    if (it != possibleTypes.end())
      possibleTypes.erase(it);
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// ErfOp
//===----------------------------------------------------------------------===//

OpFoldResult math::ErfOp::fold(FoldAdaptor adaptor) {
  return constFoldUnaryOpConditional<FloatAttr>(
      adaptor.getOperands(), [](const APFloat &a) -> std::optional<APFloat> {
        switch (a.getSizeInBits(a.getSemantics())) {
        case 64:
          return APFloat(erf(a.convertToDouble()));
        case 32:
          return APFloat(erff(a.convertToFloat()));
        default:
          return {};
        }
      });
}

LogicalResult math::ErfOp::generate(GeneratorOpBuilder &builder) {
  llvm::SmallVector<Type> possibleTypes = {
      builder.getF16Type(),
      builder.getF32Type(),
      builder.getF64Type(),
  };

  while (!possibleTypes.empty()) {
    unsigned idx = builder.sampleUniform(possibleTypes.size() - 1);
    Type resultType = possibleTypes[idx];

    llvm::Optional<Value> lhs = builder.sampleValueOfType(resultType);

    if (!lhs.has_value()) {
      Type *it = llvm::find(possibleTypes, resultType);
      if (it != possibleTypes.end())
        possibleTypes.erase(it);
      continue;
    }

    OperationState state(builder.getUnknownLoc(),
                         math::ErfOp::getOperationName());
    math::ErfOp::build(builder, state, lhs.value());
    if (builder.create(state) != nullptr)
      return success();

    Type *it = llvm::find(possibleTypes, resultType);
    if (it != possibleTypes.end())
      possibleTypes.erase(it);
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// IPowIOp
//===----------------------------------------------------------------------===//

OpFoldResult math::IPowIOp::fold(FoldAdaptor adaptor) {
  return constFoldBinaryOpConditional<IntegerAttr>(
      adaptor.getOperands(),
      [](const APInt &base, const APInt &power) -> std::optional<APInt> {
        unsigned width = base.getBitWidth();
        auto zeroValue = APInt::getZero(width);
        APInt oneValue{width, 1ULL, /*isSigned=*/true};
        APInt minusOneValue{width, -1ULL, /*isSigned=*/true};

        if (power.isZero())
          return oneValue;

        if (power.isNegative()) {
          // Leave 0 raised to negative power not folded.
          if (base.isZero())
            return {};
          if (base.eq(oneValue))
            return oneValue;
          // If abs(base) > 1, then the result is zero.
          if (base.ne(minusOneValue))
            return zeroValue;
          // base == -1:
          //   -1: power is odd
          //    1: power is even
          if (power[0] == 1)
            return minusOneValue;

          return oneValue;
        }

        // power is positive.
        APInt result = oneValue;
        APInt curBase = base;
        APInt curPower = power;
        while (true) {
          if (curPower[0] == 1)
            result *= curBase;
          curPower.lshrInPlace(1);
          if (curPower.isZero())
            return result;
          curBase *= curBase;
        }
      });

  return Attribute();
}

LogicalResult math::IPowIOp::generate(GeneratorOpBuilder &builder) {
  llvm::SmallVector<Type> possibleTypes = {
      builder.getI1Type(),  builder.getIndexType(), builder.getI8Type(),
      builder.getI16Type(), builder.getI32Type(),   builder.getI64Type(),
  };

  while (!possibleTypes.empty()) {
    unsigned idx = builder.sampleUniform(possibleTypes.size() - 1);
    Type resultType = possibleTypes[idx];

    llvm::Optional<Value> lhs = builder.sampleValueOfType(resultType);
    llvm::Optional<Value> rhs = builder.sampleValueOfType(resultType);

    if (!lhs.has_value() || !rhs.has_value()) {
      Type *it = llvm::find(possibleTypes, resultType);
      if (it != possibleTypes.end())
        possibleTypes.erase(it);
      continue;
    }

    OperationState state(builder.getUnknownLoc(),
                         math::IPowIOp::getOperationName());
    math::IPowIOp::build(builder, state, lhs.value(), rhs.value());
    if (builder.create(state) != nullptr)
      return success();

    Type *it = llvm::find(possibleTypes, resultType);
    if (it != possibleTypes.end())
      possibleTypes.erase(it);
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// LogOp
//===----------------------------------------------------------------------===//

OpFoldResult math::LogOp::fold(FoldAdaptor adaptor) {
  return constFoldUnaryOpConditional<FloatAttr>(
      adaptor.getOperands(), [](const APFloat &a) -> std::optional<APFloat> {
        if (a.isNegative())
          return {};

        if (a.getSizeInBits(a.getSemantics()) == 64)
          return APFloat(log(a.convertToDouble()));

        if (a.getSizeInBits(a.getSemantics()) == 32)
          return APFloat(logf(a.convertToFloat()));

        return {};
      });
}

LogicalResult math::LogOp::generate(GeneratorOpBuilder &builder) {
  llvm::SmallVector<Type> possibleTypes = {
      builder.getF16Type(),
      builder.getF32Type(),
      builder.getF64Type(),
  };

  while (!possibleTypes.empty()) {
    unsigned idx = builder.sampleUniform(possibleTypes.size() - 1);
    Type resultType = possibleTypes[idx];

    llvm::Optional<Value> lhs = builder.sampleValueOfType(resultType);

    if (!lhs.has_value()) {
      Type *it = llvm::find(possibleTypes, resultType);
      if (it != possibleTypes.end())
        possibleTypes.erase(it);
      continue;
    }

    OperationState state(builder.getUnknownLoc(),
                         math::LogOp::getOperationName());
    math::LogOp::build(builder, state, lhs.value());
    if (builder.create(state) != nullptr)
      return success();

    Type *it = llvm::find(possibleTypes, resultType);
    if (it != possibleTypes.end())
      possibleTypes.erase(it);
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// Log2Op
//===----------------------------------------------------------------------===//

OpFoldResult math::Log2Op::fold(FoldAdaptor adaptor) {
  return constFoldUnaryOpConditional<FloatAttr>(
      adaptor.getOperands(), [](const APFloat &a) -> std::optional<APFloat> {
        if (a.isNegative())
          return {};

        if (a.getSizeInBits(a.getSemantics()) == 64)
          return APFloat(log2(a.convertToDouble()));

        if (a.getSizeInBits(a.getSemantics()) == 32)
          return APFloat(log2f(a.convertToFloat()));

        return {};
      });
}

LogicalResult math::Log2Op::generate(GeneratorOpBuilder &builder) {
  llvm::SmallVector<Type> possibleTypes = {
      builder.getF16Type(),
      builder.getF32Type(),
      builder.getF64Type(),
  };

  while (!possibleTypes.empty()) {
    unsigned idx = builder.sampleUniform(possibleTypes.size() - 1);
    Type resultType = possibleTypes[idx];

    llvm::Optional<Value> lhs = builder.sampleValueOfType(resultType);

    if (!lhs.has_value()) {
      Type *it = llvm::find(possibleTypes, resultType);
      if (it != possibleTypes.end())
        possibleTypes.erase(it);
      continue;
    }

    OperationState state(builder.getUnknownLoc(),
                         math::Log2Op::getOperationName());
    math::Log2Op::build(builder, state, lhs.value());
    if (builder.create(state) != nullptr)
      return success();

    Type *it = llvm::find(possibleTypes, resultType);
    if (it != possibleTypes.end())
      possibleTypes.erase(it);
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// Log10Op
//===----------------------------------------------------------------------===//

OpFoldResult math::Log10Op::fold(FoldAdaptor adaptor) {
  return constFoldUnaryOpConditional<FloatAttr>(
      adaptor.getOperands(), [](const APFloat &a) -> std::optional<APFloat> {
        if (a.isNegative())
          return {};

        switch (a.getSizeInBits(a.getSemantics())) {
        case 64:
          return APFloat(log10(a.convertToDouble()));
        case 32:
          return APFloat(log10f(a.convertToFloat()));
        default:
          return {};
        }
      });
}

LogicalResult math::Log10Op::generate(GeneratorOpBuilder &builder) {
  llvm::SmallVector<Type> possibleTypes = {
      builder.getF16Type(),
      builder.getF32Type(),
      builder.getF64Type(),
  };

  while (!possibleTypes.empty()) {
    unsigned idx = builder.sampleUniform(possibleTypes.size() - 1);
    Type resultType = possibleTypes[idx];

    llvm::Optional<Value> lhs = builder.sampleValueOfType(resultType);

    if (!lhs.has_value()) {
      Type *it = llvm::find(possibleTypes, resultType);
      if (it != possibleTypes.end())
        possibleTypes.erase(it);
      continue;
    }

    OperationState state(builder.getUnknownLoc(),
                         math::Log10Op::getOperationName());
    math::Log10Op::build(builder, state, lhs.value());
    if (builder.create(state) != nullptr)
      return success();

    Type *it = llvm::find(possibleTypes, resultType);
    if (it != possibleTypes.end())
      possibleTypes.erase(it);
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// Log1pOp
//===----------------------------------------------------------------------===//

OpFoldResult math::Log1pOp::fold(FoldAdaptor adaptor) {
  return constFoldUnaryOpConditional<FloatAttr>(
      adaptor.getOperands(), [](const APFloat &a) -> std::optional<APFloat> {
        switch (a.getSizeInBits(a.getSemantics())) {
        case 64:
          if ((a + APFloat(1.0)).isNegative())
            return {};
          return APFloat(log1p(a.convertToDouble()));
        case 32:
          if ((a + APFloat(1.0f)).isNegative())
            return {};
          return APFloat(log1pf(a.convertToFloat()));
        default:
          return {};
        }
      });
}

LogicalResult math::Log1pOp::generate(GeneratorOpBuilder &builder) {
  llvm::SmallVector<Type> possibleTypes = {
      builder.getF16Type(),
      builder.getF32Type(),
      builder.getF64Type(),
  };

  while (!possibleTypes.empty()) {
    unsigned idx = builder.sampleUniform(possibleTypes.size() - 1);
    Type resultType = possibleTypes[idx];

    llvm::Optional<Value> lhs = builder.sampleValueOfType(resultType);

    if (!lhs.has_value()) {
      Type *it = llvm::find(possibleTypes, resultType);
      if (it != possibleTypes.end())
        possibleTypes.erase(it);
      continue;
    }

    OperationState state(builder.getUnknownLoc(),
                         math::Log1pOp::getOperationName());
    math::Log1pOp::build(builder, state, lhs.value());
    if (builder.create(state) != nullptr)
      return success();

    Type *it = llvm::find(possibleTypes, resultType);
    if (it != possibleTypes.end())
      possibleTypes.erase(it);
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// PowFOp
//===----------------------------------------------------------------------===//

OpFoldResult math::PowFOp::fold(FoldAdaptor adaptor) {
  return constFoldBinaryOpConditional<FloatAttr>(
      adaptor.getOperands(),
      [](const APFloat &a, const APFloat &b) -> std::optional<APFloat> {
        if (a.getSizeInBits(a.getSemantics()) == 64 &&
            b.getSizeInBits(b.getSemantics()) == 64)
          return APFloat(pow(a.convertToDouble(), b.convertToDouble()));

        if (a.getSizeInBits(a.getSemantics()) == 32 &&
            b.getSizeInBits(b.getSemantics()) == 32)
          return APFloat(powf(a.convertToFloat(), b.convertToFloat()));

        return {};
      });
}

LogicalResult math::PowFOp::generate(GeneratorOpBuilder &builder) {
  llvm::SmallVector<Type> possibleTypes = {
      builder.getF16Type(),
      builder.getF32Type(),
      builder.getF64Type(),
  };

  while (!possibleTypes.empty()) {
    unsigned idx = builder.sampleUniform(possibleTypes.size() - 1);
    Type resultType = possibleTypes[idx];

    llvm::Optional<Value> lhs = builder.sampleValueOfType(resultType);
    llvm::Optional<Value> rhs = builder.sampleValueOfType(resultType);

    if (!lhs.has_value() || !rhs.has_value()) {
      Type *it = llvm::find(possibleTypes, resultType);
      if (it != possibleTypes.end())
        possibleTypes.erase(it);
      continue;
    }

    OperationState state(builder.getUnknownLoc(),
                         math::PowFOp::getOperationName());
    math::PowFOp::build(builder, state, lhs.value(), rhs.value());
    if (builder.create(state) != nullptr)
      return success();

    Type *it = llvm::find(possibleTypes, resultType);
    if (it != possibleTypes.end())
      possibleTypes.erase(it);
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// SqrtOp
//===----------------------------------------------------------------------===//

OpFoldResult math::SqrtOp::fold(FoldAdaptor adaptor) {
  return constFoldUnaryOpConditional<FloatAttr>(
      adaptor.getOperands(), [](const APFloat &a) -> std::optional<APFloat> {
        if (a.isNegative())
          return {};

        switch (a.getSizeInBits(a.getSemantics())) {
        case 64:
          return APFloat(sqrt(a.convertToDouble()));
        case 32:
          return APFloat(sqrtf(a.convertToFloat()));
        default:
          return {};
        }
      });
}

LogicalResult math::SqrtOp::generate(GeneratorOpBuilder &builder) {
  llvm::SmallVector<Type> possibleTypes = {
      builder.getF16Type(),
      builder.getF32Type(),
      builder.getF64Type(),
  };

  while (!possibleTypes.empty()) {
    unsigned idx = builder.sampleUniform(possibleTypes.size() - 1);
    Type resultType = possibleTypes[idx];

    llvm::Optional<Value> lhs = builder.sampleValueOfType(resultType);

    if (!lhs.has_value()) {
      Type *it = llvm::find(possibleTypes, resultType);
      if (it != possibleTypes.end())
        possibleTypes.erase(it);
      continue;
    }

    OperationState state(builder.getUnknownLoc(),
                         math::SqrtOp::getOperationName());
    math::SqrtOp::build(builder, state, lhs.value());
    if (builder.create(state) != nullptr)
      return success();

    Type *it = llvm::find(possibleTypes, resultType);
    if (it != possibleTypes.end())
      possibleTypes.erase(it);
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// ExpOp
//===----------------------------------------------------------------------===//

OpFoldResult math::ExpOp::fold(FoldAdaptor adaptor) {
  return constFoldUnaryOpConditional<FloatAttr>(
      adaptor.getOperands(), [](const APFloat &a) -> std::optional<APFloat> {
        switch (a.getSizeInBits(a.getSemantics())) {
        case 64:
          return APFloat(exp(a.convertToDouble()));
        case 32:
          return APFloat(expf(a.convertToFloat()));
        default:
          return {};
        }
      });
}

LogicalResult math::ExpOp::generate(GeneratorOpBuilder &builder) {
  llvm::SmallVector<Type> possibleTypes = {
      builder.getF16Type(),
      builder.getF32Type(),
      builder.getF64Type(),
  };

  while (!possibleTypes.empty()) {
    unsigned idx = builder.sampleUniform(possibleTypes.size() - 1);
    Type resultType = possibleTypes[idx];

    llvm::Optional<Value> lhs = builder.sampleValueOfType(resultType);

    if (!lhs.has_value()) {
      Type *it = llvm::find(possibleTypes, resultType);
      if (it != possibleTypes.end())
        possibleTypes.erase(it);
      continue;
    }

    OperationState state(builder.getUnknownLoc(),
                         math::ExpOp::getOperationName());
    math::ExpOp::build(builder, state, lhs.value());
    if (builder.create(state) != nullptr)
      return success();

    Type *it = llvm::find(possibleTypes, resultType);
    if (it != possibleTypes.end())
      possibleTypes.erase(it);
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// Exp2Op
//===----------------------------------------------------------------------===//

OpFoldResult math::Exp2Op::fold(FoldAdaptor adaptor) {
  return constFoldUnaryOpConditional<FloatAttr>(
      adaptor.getOperands(), [](const APFloat &a) -> std::optional<APFloat> {
        switch (a.getSizeInBits(a.getSemantics())) {
        case 64:
          return APFloat(exp2(a.convertToDouble()));
        case 32:
          return APFloat(exp2f(a.convertToFloat()));
        default:
          return {};
        }
      });
}

LogicalResult math::Exp2Op::generate(GeneratorOpBuilder &builder) {
  llvm::SmallVector<Type> possibleTypes = {
      builder.getF16Type(),
      builder.getF32Type(),
      builder.getF64Type(),
  };

  while (!possibleTypes.empty()) {
    unsigned idx = builder.sampleUniform(possibleTypes.size() - 1);
    Type resultType = possibleTypes[idx];

    llvm::Optional<Value> lhs = builder.sampleValueOfType(resultType);

    if (!lhs.has_value()) {
      Type *it = llvm::find(possibleTypes, resultType);
      if (it != possibleTypes.end())
        possibleTypes.erase(it);
      continue;
    }

    OperationState state(builder.getUnknownLoc(),
                         math::Exp2Op::getOperationName());
    math::Exp2Op::build(builder, state, lhs.value());
    if (builder.create(state) != nullptr)
      return success();

    Type *it = llvm::find(possibleTypes, resultType);
    if (it != possibleTypes.end())
      possibleTypes.erase(it);
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// ExpM1Op
//===----------------------------------------------------------------------===//

OpFoldResult math::ExpM1Op::fold(FoldAdaptor adaptor) {
  return constFoldUnaryOpConditional<FloatAttr>(
      adaptor.getOperands(), [](const APFloat &a) -> std::optional<APFloat> {
        switch (a.getSizeInBits(a.getSemantics())) {
        case 64:
          return APFloat(expm1(a.convertToDouble()));
        case 32:
          return APFloat(expm1f(a.convertToFloat()));
        default:
          return {};
        }
      });
}

LogicalResult math::ExpM1Op::generate(GeneratorOpBuilder &builder) {
  llvm::SmallVector<Type> possibleTypes = {
      builder.getF16Type(),
      builder.getF32Type(),
      builder.getF64Type(),
  };

  while (!possibleTypes.empty()) {
    unsigned idx = builder.sampleUniform(possibleTypes.size() - 1);
    Type resultType = possibleTypes[idx];

    llvm::Optional<Value> lhs = builder.sampleValueOfType(resultType);

    if (!lhs.has_value()) {
      Type *it = llvm::find(possibleTypes, resultType);
      if (it != possibleTypes.end())
        possibleTypes.erase(it);
      continue;
    }

    OperationState state(builder.getUnknownLoc(),
                         math::ExpM1Op::getOperationName());
    math::ExpM1Op::build(builder, state, lhs.value());
    if (builder.create(state) != nullptr)
      return success();

    Type *it = llvm::find(possibleTypes, resultType);
    if (it != possibleTypes.end())
      possibleTypes.erase(it);
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// TanOp
//===----------------------------------------------------------------------===//

OpFoldResult math::TanOp::fold(FoldAdaptor adaptor) {
  return constFoldUnaryOpConditional<FloatAttr>(
      adaptor.getOperands(), [](const APFloat &a) -> std::optional<APFloat> {
        switch (a.getSizeInBits(a.getSemantics())) {
        case 64:
          return APFloat(tan(a.convertToDouble()));
        case 32:
          return APFloat(tanf(a.convertToFloat()));
        default:
          return {};
        }
      });
}

LogicalResult math::TanOp::generate(GeneratorOpBuilder &builder) {
  llvm::SmallVector<Type> possibleTypes = {
      builder.getF16Type(),
      builder.getF32Type(),
      builder.getF64Type(),
  };

  while (!possibleTypes.empty()) {
    unsigned idx = builder.sampleUniform(possibleTypes.size() - 1);
    Type resultType = possibleTypes[idx];

    llvm::Optional<Value> lhs = builder.sampleValueOfType(resultType);

    if (!lhs.has_value()) {
      Type *it = llvm::find(possibleTypes, resultType);
      if (it != possibleTypes.end())
        possibleTypes.erase(it);
      continue;
    }

    OperationState state(builder.getUnknownLoc(),
                         math::TanOp::getOperationName());
    math::TanOp::build(builder, state, lhs.value());
    if (builder.create(state) != nullptr)
      return success();

    Type *it = llvm::find(possibleTypes, resultType);
    if (it != possibleTypes.end())
      possibleTypes.erase(it);
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// TanhOp
//===----------------------------------------------------------------------===//

OpFoldResult math::TanhOp::fold(FoldAdaptor adaptor) {
  return constFoldUnaryOpConditional<FloatAttr>(
      adaptor.getOperands(), [](const APFloat &a) -> std::optional<APFloat> {
        switch (a.getSizeInBits(a.getSemantics())) {
        case 64:
          return APFloat(tanh(a.convertToDouble()));
        case 32:
          return APFloat(tanhf(a.convertToFloat()));
        default:
          return {};
        }
      });
}

LogicalResult math::TanhOp::generate(GeneratorOpBuilder &builder) {
  llvm::SmallVector<Type> possibleTypes = {
      builder.getF16Type(),
      builder.getF32Type(),
      builder.getF64Type(),
  };

  while (!possibleTypes.empty()) {
    unsigned idx = builder.sampleUniform(possibleTypes.size() - 1);
    Type resultType = possibleTypes[idx];

    llvm::Optional<Value> lhs = builder.sampleValueOfType(resultType);

    if (!lhs.has_value()) {
      Type *it = llvm::find(possibleTypes, resultType);
      if (it != possibleTypes.end())
        possibleTypes.erase(it);
      continue;
    }

    OperationState state(builder.getUnknownLoc(),
                         math::TanhOp::getOperationName());
    math::TanhOp::build(builder, state, lhs.value());
    if (builder.create(state) != nullptr)
      return success();

    Type *it = llvm::find(possibleTypes, resultType);
    if (it != possibleTypes.end())
      possibleTypes.erase(it);
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// RoundEvenOp
//===----------------------------------------------------------------------===//

OpFoldResult math::RoundEvenOp::fold(FoldAdaptor adaptor) {
  return constFoldUnaryOp<FloatAttr>(
      adaptor.getOperands(), [](const APFloat &a) {
        APFloat result(a);
        result.roundToIntegral(llvm::RoundingMode::NearestTiesToEven);
        return result;
      });
}

LogicalResult math::RoundEvenOp::generate(GeneratorOpBuilder &builder) {
  llvm::SmallVector<Type> possibleTypes = {
      builder.getF16Type(),
      builder.getF32Type(),
      builder.getF64Type(),
  };

  while (!possibleTypes.empty()) {
    unsigned idx = builder.sampleUniform(possibleTypes.size() - 1);
    Type resultType = possibleTypes[idx];

    llvm::Optional<Value> lhs = builder.sampleValueOfType(resultType);

    if (!lhs.has_value()) {
      Type *it = llvm::find(possibleTypes, resultType);
      if (it != possibleTypes.end())
        possibleTypes.erase(it);
      continue;
    }

    OperationState state(builder.getUnknownLoc(),
                         math::RoundEvenOp::getOperationName());
    math::RoundEvenOp::build(builder, state, lhs.value());
    if (builder.create(state) != nullptr)
      return success();

    Type *it = llvm::find(possibleTypes, resultType);
    if (it != possibleTypes.end())
      possibleTypes.erase(it);
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// FloorOp
//===----------------------------------------------------------------------===//

OpFoldResult math::FloorOp::fold(FoldAdaptor adaptor) {
  return constFoldUnaryOp<FloatAttr>(
      adaptor.getOperands(), [](const APFloat &a) {
        APFloat result(a);
        result.roundToIntegral(llvm::RoundingMode::TowardNegative);
        return result;
      });
}

LogicalResult math::FloorOp::generate(GeneratorOpBuilder &builder) {
  llvm::SmallVector<Type> possibleTypes = {
      builder.getF16Type(),
      builder.getF32Type(),
      builder.getF64Type(),
  };

  while (!possibleTypes.empty()) {
    unsigned idx = builder.sampleUniform(possibleTypes.size() - 1);
    Type resultType = possibleTypes[idx];

    llvm::Optional<Value> lhs = builder.sampleValueOfType(resultType);

    if (!lhs.has_value()) {
      Type *it = llvm::find(possibleTypes, resultType);
      if (it != possibleTypes.end())
        possibleTypes.erase(it);
      continue;
    }

    OperationState state(builder.getUnknownLoc(),
                         math::FloorOp::getOperationName());
    math::FloorOp::build(builder, state, lhs.value());
    if (builder.create(state) != nullptr)
      return success();

    Type *it = llvm::find(possibleTypes, resultType);
    if (it != possibleTypes.end())
      possibleTypes.erase(it);
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// RoundOp
//===----------------------------------------------------------------------===//

OpFoldResult math::RoundOp::fold(FoldAdaptor adaptor) {
  return constFoldUnaryOpConditional<FloatAttr>(
      adaptor.getOperands(), [](const APFloat &a) -> std::optional<APFloat> {
        switch (a.getSizeInBits(a.getSemantics())) {
        case 64:
          return APFloat(round(a.convertToDouble()));
        case 32:
          return APFloat(roundf(a.convertToFloat()));
        default:
          return {};
        }
      });
}

LogicalResult math::RoundOp::generate(GeneratorOpBuilder &builder) {
  llvm::SmallVector<Type> possibleTypes = {
      builder.getF16Type(),
      builder.getF32Type(),
      builder.getF64Type(),
  };

  while (!possibleTypes.empty()) {
    unsigned idx = builder.sampleUniform(possibleTypes.size() - 1);
    Type resultType = possibleTypes[idx];

    llvm::Optional<Value> lhs = builder.sampleValueOfType(resultType);

    if (!lhs.has_value()) {
      Type *it = llvm::find(possibleTypes, resultType);
      if (it != possibleTypes.end())
        possibleTypes.erase(it);
      continue;
    }

    OperationState state(builder.getUnknownLoc(),
                         math::RoundOp::getOperationName());
    math::RoundOp::build(builder, state, lhs.value());
    if (builder.create(state) != nullptr)
      return success();

    Type *it = llvm::find(possibleTypes, resultType);
    if (it != possibleTypes.end())
      possibleTypes.erase(it);
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// TruncOp
//===----------------------------------------------------------------------===//

OpFoldResult math::TruncOp::fold(FoldAdaptor adaptor) {
  return constFoldUnaryOpConditional<FloatAttr>(
      adaptor.getOperands(), [](const APFloat &a) -> std::optional<APFloat> {
        switch (a.getSizeInBits(a.getSemantics())) {
        case 64:
          return APFloat(trunc(a.convertToDouble()));
        case 32:
          return APFloat(truncf(a.convertToFloat()));
        default:
          return {};
        }
      });
}

LogicalResult math::TruncOp::generate(GeneratorOpBuilder &builder) {
  llvm::SmallVector<Type> possibleTypes = {
      builder.getF16Type(),
      builder.getF32Type(),
      builder.getF64Type(),
  };

  while (!possibleTypes.empty()) {
    unsigned idx = builder.sampleUniform(possibleTypes.size() - 1);
    Type resultType = possibleTypes[idx];

    llvm::Optional<Value> lhs = builder.sampleValueOfType(resultType);

    if (!lhs.has_value()) {
      Type *it = llvm::find(possibleTypes, resultType);
      if (it != possibleTypes.end())
        possibleTypes.erase(it);
      continue;
    }

    OperationState state(builder.getUnknownLoc(),
                         math::TruncOp::getOperationName());
    math::TruncOp::build(builder, state, lhs.value());
    if (builder.create(state) != nullptr)
      return success();

    Type *it = llvm::find(possibleTypes, resultType);
    if (it != possibleTypes.end())
      possibleTypes.erase(it);
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// CbrtOp
//===----------------------------------------------------------------------===//

LogicalResult math::CbrtOp::generate(GeneratorOpBuilder &builder) {
  llvm::SmallVector<Type> possibleTypes = {
      builder.getF16Type(),
      builder.getF32Type(),
      builder.getF64Type(),
  };

  while (!possibleTypes.empty()) {
    unsigned idx = builder.sampleUniform(possibleTypes.size() - 1);
    Type resultType = possibleTypes[idx];

    llvm::Optional<Value> lhs = builder.sampleValueOfType(resultType);

    if (!lhs.has_value()) {
      Type *it = llvm::find(possibleTypes, resultType);
      if (it != possibleTypes.end())
        possibleTypes.erase(it);
      continue;
    }

    OperationState state(builder.getUnknownLoc(),
                         math::CbrtOp::getOperationName());
    math::CbrtOp::build(builder, state, lhs.value());
    if (builder.create(state) != nullptr)
      return success();

    Type *it = llvm::find(possibleTypes, resultType);
    if (it != possibleTypes.end())
      possibleTypes.erase(it);
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// FPowIOp
//===----------------------------------------------------------------------===//

LogicalResult math::FPowIOp::generate(GeneratorOpBuilder &builder) {
  llvm::SmallVector<Type> floatTypes = {
      builder.getF16Type(),
      builder.getF32Type(),
      builder.getF64Type(),
  };

  llvm::SmallVector<Type> intTypes = {
      builder.getI1Type(),  builder.getIndexType(), builder.getI8Type(),
      builder.getI16Type(), builder.getI32Type(),   builder.getI64Type(),
  };

  llvm::SmallVector<std::tuple<Type, Type>> typeTuples;
  for (Type floatType : floatTypes)
    for (Type intType : intTypes)
      typeTuples.push_back(std::make_tuple(floatType, intType));

  while (!typeTuples.empty()) {
    unsigned idx = builder.sampleUniform(typeTuples.size() - 1);
    std::tuple<Type, Type> types = typeTuples[idx];
    llvm::Optional<Value> lhs = builder.sampleValueOfType(std::get<0>(types));
    llvm::Optional<Value> rhs = builder.sampleValueOfType(std::get<1>(types));

    if (!lhs.has_value() || !rhs.has_value()) {
      std::tuple<Type, Type> *it = llvm::find(typeTuples, types);
      if (it != typeTuples.end())
        typeTuples.erase(it);
      continue;
    }

    OperationState state(builder.getUnknownLoc(),
                         math::FPowIOp::getOperationName());
    math::FPowIOp::build(builder, state, lhs.value(), rhs.value());
    if (builder.create(state) != nullptr)
      return success();

    std::tuple<Type, Type> *it = llvm::find(typeTuples, types);
    if (it != typeTuples.end())
      typeTuples.erase(it);
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// FmaOp
//===----------------------------------------------------------------------===//

LogicalResult math::FmaOp::generate(GeneratorOpBuilder &builder) {
  llvm::SmallVector<Type> possibleTypes = {
      builder.getF16Type(),
      builder.getF32Type(),
      builder.getF64Type(),
  };

  while (!possibleTypes.empty()) {
    unsigned idx = builder.sampleUniform(possibleTypes.size() - 1);
    Type resultType = possibleTypes[idx];

    llvm::Optional<Value> v1 = builder.sampleValueOfType(resultType);
    llvm::Optional<Value> v2 = builder.sampleValueOfType(resultType);
    llvm::Optional<Value> v3 = builder.sampleValueOfType(resultType);

    if (!v1.has_value() || !v2.has_value() || !v3.has_value()) {
      Type *it = llvm::find(possibleTypes, resultType);
      if (it != possibleTypes.end())
        possibleTypes.erase(it);
      continue;
    }

    OperationState state(builder.getUnknownLoc(),
                         math::FmaOp::getOperationName());
    math::FmaOp::build(builder, state, v1.value(), v2.value(), v3.value());
    if (builder.create(state) != nullptr)
      return success();

    Type *it = llvm::find(possibleTypes, resultType);
    if (it != possibleTypes.end())
      possibleTypes.erase(it);
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// RsqrtOp
//===----------------------------------------------------------------------===//

LogicalResult math::RsqrtOp::generate(GeneratorOpBuilder &builder) {
  llvm::SmallVector<Type> possibleTypes = {
      builder.getF16Type(),
      builder.getF32Type(),
      builder.getF64Type(),
  };

  while (!possibleTypes.empty()) {
    unsigned idx = builder.sampleUniform(possibleTypes.size() - 1);
    Type resultType = possibleTypes[idx];

    llvm::Optional<Value> lhs = builder.sampleValueOfType(resultType);

    if (!lhs.has_value()) {
      Type *it = llvm::find(possibleTypes, resultType);
      if (it != possibleTypes.end())
        possibleTypes.erase(it);
      continue;
    }

    OperationState state(builder.getUnknownLoc(),
                         math::RsqrtOp::getOperationName());
    math::RsqrtOp::build(builder, state, lhs.value());
    if (builder.create(state) != nullptr)
      return success();

    Type *it = llvm::find(possibleTypes, resultType);
    if (it != possibleTypes.end())
      possibleTypes.erase(it);
  }

  return failure();
}

/// Materialize an integer or floating point constant.
Operation *math::MathDialect::materializeConstant(OpBuilder &builder,
                                                  Attribute value, Type type,
                                                  Location loc) {
  return arith::ConstantOp::materialize(builder, value, type, loc);
}
