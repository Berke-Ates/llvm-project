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
// AbsFOp folder
//===----------------------------------------------------------------------===//

OpFoldResult math::AbsFOp::fold(FoldAdaptor adaptor) {
  return constFoldUnaryOp<FloatAttr>(adaptor.getOperands(),
                                     [](const APFloat &a) { return abs(a); });
}

LogicalResult math::AbsFOp::generate(GeneratorOpBuilder &builder) {
  // TODO: ADD THIS OP
  return failure();
}

llvm::SmallVector<Type>
math::AbsFOp::getGeneratableTypes(GeneratorOpBuilder &builder) {
  return {};
}

//===----------------------------------------------------------------------===//
// AbsIOp folder
//===----------------------------------------------------------------------===//

OpFoldResult math::AbsIOp::fold(FoldAdaptor adaptor) {
  return constFoldUnaryOp<IntegerAttr>(adaptor.getOperands(),
                                       [](const APInt &a) { return a.abs(); });
}

LogicalResult math::AbsIOp::generate(GeneratorOpBuilder &builder) {
  // TODO: ADD THIS OP
  return failure();
}

llvm::SmallVector<Type>
math::AbsIOp::getGeneratableTypes(GeneratorOpBuilder &builder) {
  return {};
}

//===----------------------------------------------------------------------===//
// AtanOp folder
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
  // TODO: ADD THIS OP
  return failure();
}

llvm::SmallVector<Type>
math::AtanOp::getGeneratableTypes(GeneratorOpBuilder &builder) {
  return {};
}

//===----------------------------------------------------------------------===//
// Atan2Op folder
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
  // TODO: ADD THIS OP
  return failure();
}

llvm::SmallVector<Type>
math::Atan2Op::getGeneratableTypes(GeneratorOpBuilder &builder) {
  return {};
}

//===----------------------------------------------------------------------===//
// CeilOp folder
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
  // TODO: ADD THIS OP
  return failure();
}

llvm::SmallVector<Type>
math::CeilOp::getGeneratableTypes(GeneratorOpBuilder &builder) {
  return {};
}

//===----------------------------------------------------------------------===//
// CopySignOp folder
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
  // TODO: ADD THIS OP
  return failure();
}

llvm::SmallVector<Type>
math::CopySignOp::getGeneratableTypes(GeneratorOpBuilder &builder) {
  return {};
}

//===----------------------------------------------------------------------===//
// CosOp folder
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
  // TODO: ADD THIS OP
  return failure();
}

llvm::SmallVector<Type>
math::CosOp::getGeneratableTypes(GeneratorOpBuilder &builder) {
  return {};
}

//===----------------------------------------------------------------------===//
// SinOp folder
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
  // TODO: ADD THIS OP
  return failure();
}

llvm::SmallVector<Type>
math::SinOp::getGeneratableTypes(GeneratorOpBuilder &builder) {
  return {};
}

//===----------------------------------------------------------------------===//
// CountLeadingZerosOp folder
//===----------------------------------------------------------------------===//

OpFoldResult math::CountLeadingZerosOp::fold(FoldAdaptor adaptor) {
  return constFoldUnaryOp<IntegerAttr>(
      adaptor.getOperands(),
      [](const APInt &a) { return APInt(a.getBitWidth(), a.countl_zero()); });
}

LogicalResult math::CountLeadingZerosOp::generate(GeneratorOpBuilder &builder) {
  // TODO: ADD THIS OP
  return failure();
}

llvm::SmallVector<Type>
math::CountLeadingZerosOp::getGeneratableTypes(GeneratorOpBuilder &builder) {
  return {};
}

//===----------------------------------------------------------------------===//
// CountTrailingZerosOp folder
//===----------------------------------------------------------------------===//

OpFoldResult math::CountTrailingZerosOp::fold(FoldAdaptor adaptor) {
  return constFoldUnaryOp<IntegerAttr>(
      adaptor.getOperands(),
      [](const APInt &a) { return APInt(a.getBitWidth(), a.countr_zero()); });
}

LogicalResult
math::CountTrailingZerosOp::generate(GeneratorOpBuilder &builder) {
  // TODO: ADD THIS OP
  return failure();
}

llvm::SmallVector<Type>
math::CountTrailingZerosOp::getGeneratableTypes(GeneratorOpBuilder &builder) {
  return {};
}

//===----------------------------------------------------------------------===//
// CtPopOp folder
//===----------------------------------------------------------------------===//

OpFoldResult math::CtPopOp::fold(FoldAdaptor adaptor) {
  return constFoldUnaryOp<IntegerAttr>(
      adaptor.getOperands(),
      [](const APInt &a) { return APInt(a.getBitWidth(), a.popcount()); });
}

LogicalResult math::CtPopOp::generate(GeneratorOpBuilder &builder) {
  // TODO: ADD THIS OP
  return failure();
}

llvm::SmallVector<Type>
math::CtPopOp::getGeneratableTypes(GeneratorOpBuilder &builder) {
  return {};
}

//===----------------------------------------------------------------------===//
// ErfOp folder
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
  // TODO: ADD THIS OP
  return failure();
}

llvm::SmallVector<Type>
math::ErfOp::getGeneratableTypes(GeneratorOpBuilder &builder) {
  return {};
}

//===----------------------------------------------------------------------===//
// IPowIOp folder
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
  // TODO: ADD THIS OP
  return failure();
}

llvm::SmallVector<Type>
math::IPowIOp::getGeneratableTypes(GeneratorOpBuilder &builder) {
  return {};
}

//===----------------------------------------------------------------------===//
// LogOp folder
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
  // TODO: ADD THIS OP
  return failure();
}

llvm::SmallVector<Type>
math::LogOp::getGeneratableTypes(GeneratorOpBuilder &builder) {
  return {};
}

//===----------------------------------------------------------------------===//
// Log2Op folder
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
  // TODO: ADD THIS OP
  return failure();
}

llvm::SmallVector<Type>
math::Log2Op::getGeneratableTypes(GeneratorOpBuilder &builder) {
  return {};
}

//===----------------------------------------------------------------------===//
// Log10Op folder
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
  // TODO: ADD THIS OP
  return failure();
}

llvm::SmallVector<Type>
math::Log10Op::getGeneratableTypes(GeneratorOpBuilder &builder) {
  return {};
}

//===----------------------------------------------------------------------===//
// Log1pOp folder
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
  // TODO: ADD THIS OP
  return failure();
}

llvm::SmallVector<Type>
math::Log1pOp::getGeneratableTypes(GeneratorOpBuilder &builder) {
  return {};
}

//===----------------------------------------------------------------------===//
// PowFOp folder
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
  // TODO: ADD THIS OP
  return failure();
}

llvm::SmallVector<Type>
math::PowFOp::getGeneratableTypes(GeneratorOpBuilder &builder) {
  return {};
}

//===----------------------------------------------------------------------===//
// SqrtOp folder
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
  // TODO: ADD THIS OP
  return failure();
}

llvm::SmallVector<Type>
math::SqrtOp::getGeneratableTypes(GeneratorOpBuilder &builder) {
  return {};
}

//===----------------------------------------------------------------------===//
// ExpOp folder
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
  // TODO: ADD THIS OP
  return failure();
}

llvm::SmallVector<Type>
math::ExpOp::getGeneratableTypes(GeneratorOpBuilder &builder) {
  return {};
}

//===----------------------------------------------------------------------===//
// Exp2Op folder
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
  // TODO: ADD THIS OP
  return failure();
}

llvm::SmallVector<Type>
math::Exp2Op::getGeneratableTypes(GeneratorOpBuilder &builder) {
  return {};
}

//===----------------------------------------------------------------------===//
// ExpM1Op folder
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
  // TODO: ADD THIS OP
  return failure();
}

llvm::SmallVector<Type>
math::ExpM1Op::getGeneratableTypes(GeneratorOpBuilder &builder) {
  return {};
}

//===----------------------------------------------------------------------===//
// TanOp folder
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
  // TODO: ADD THIS OP
  return failure();
}

llvm::SmallVector<Type>
math::TanOp::getGeneratableTypes(GeneratorOpBuilder &builder) {
  return {};
}

//===----------------------------------------------------------------------===//
// TanhOp folder
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
  // TODO: ADD THIS OP
  return failure();
}

llvm::SmallVector<Type>
math::TanhOp::getGeneratableTypes(GeneratorOpBuilder &builder) {
  return {};
}

//===----------------------------------------------------------------------===//
// RoundEvenOp folder
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
  // TODO: ADD THIS OP
  return failure();
}

llvm::SmallVector<Type>
math::RoundEvenOp::getGeneratableTypes(GeneratorOpBuilder &builder) {
  return {};
}

//===----------------------------------------------------------------------===//
// FloorOp folder
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
  // TODO: ADD THIS OP
  return failure();
}

llvm::SmallVector<Type>
math::FloorOp::getGeneratableTypes(GeneratorOpBuilder &builder) {
  return {};
}

//===----------------------------------------------------------------------===//
// RoundOp folder
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
  // TODO: ADD THIS OP
  return failure();
}

llvm::SmallVector<Type>
math::RoundOp::getGeneratableTypes(GeneratorOpBuilder &builder) {
  return {};
}

//===----------------------------------------------------------------------===//
// TruncOp folder
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
  // TODO: ADD THIS OP
  return failure();
}

llvm::SmallVector<Type>
math::TruncOp::getGeneratableTypes(GeneratorOpBuilder &builder) {
  return {};
}

//===----------------------------------------------------------------------===//
// CbrtOp
//===----------------------------------------------------------------------===//

LogicalResult math::CbrtOp::generate(GeneratorOpBuilder &builder) {
  // TODO: ADD THIS OP
  return failure();
}

llvm::SmallVector<Type>
math::CbrtOp::getGeneratableTypes(GeneratorOpBuilder &builder) {
  return {};
}

//===----------------------------------------------------------------------===//
// FPowIOp
//===----------------------------------------------------------------------===//

LogicalResult math::FPowIOp::generate(GeneratorOpBuilder &builder) {
  // TODO: ADD THIS OP
  return failure();
}

llvm::SmallVector<Type>
math::FPowIOp::getGeneratableTypes(GeneratorOpBuilder &builder) {
  return {};
}

//===----------------------------------------------------------------------===//
// FmaOp
//===----------------------------------------------------------------------===//

LogicalResult math::FmaOp::generate(GeneratorOpBuilder &builder) {
  // TODO: ADD THIS OP
  return failure();
}

llvm::SmallVector<Type>
math::FmaOp::getGeneratableTypes(GeneratorOpBuilder &builder) {
  return {};
}

//===----------------------------------------------------------------------===//
// RsqrtOp
//===----------------------------------------------------------------------===//

LogicalResult math::RsqrtOp::generate(GeneratorOpBuilder &builder) {
  // TODO: ADD THIS OP
  return failure();
}

llvm::SmallVector<Type>
math::RsqrtOp::getGeneratableTypes(GeneratorOpBuilder &builder) {
  return {};
}

/// Materialize an integer or floating point constant.
Operation *math::MathDialect::materializeConstant(OpBuilder &builder,
                                                  Attribute value, Type type,
                                                  Location loc) {
  return arith::ConstantOp::materialize(builder, value, type, loc);
}
