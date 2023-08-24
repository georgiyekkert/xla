/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This file provides constant folding patterns for MHLO operations.

#include "mhlo/transforms/mhlo_folders/mhlo_folders.h"

#include <cstdint>

#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/rewriters.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "utils/convert_op_folder.h"

namespace mlir {
namespace mhlo {

//===----------------------------------------------------------------------===//
// ConvertOp
//===----------------------------------------------------------------------===//

LogicalResult ConvertOpFolder::matchAndRewrite(
    ConvertOp convert_op, PatternRewriter& rewriter) const {
  Value operand = convert_op.getOperand();
  auto operandTy = operand.getType().cast<TensorType>();
  auto resultTy = convert_op.getResult().getType().cast<TensorType>();
  if (operandTy == resultTy) {
    rewriter.replaceOp(convert_op, operand);
    return success();
  }

  // If the result has non-static shape, a convert op is necessary to go from
  // static shape to non-static shape.
  if (!resultTy.hasStaticShape()) {
    return rewriter.notifyMatchFailure(
        convert_op,
        "The result has non-static shape, a convert op is necessary to go from "
        "static shape to non-static shape");
  }

  // If the operand is constant, we can do the conversion now.
  ElementsAttr elementsAttr;
  if (!matchPattern(operand, m_Constant(&elementsAttr))) {
    return rewriter.notifyMatchFailure(convert_op,
                                       "The operand is not a constant.");
  }

  // Prevent folding if the result is too large.
  if (elementsAttr.getNumElements() > op_fold_limit) {
    return rewriter.notifyMatchFailure(
        convert_op, "The result is too large, over the folding limit.");
  }
  ElementsAttr newElementsAttr = hlo::convertElementsAttr(
      elementsAttr, getElementTypeOrSelf(convert_op.getResult()));
  if (!newElementsAttr) {
    return rewriter.notifyMatchFailure(convert_op,
                                       "Failed to convert the result.");
  }
  rewriter.replaceOpWithNewOp<ConstantOp>(convert_op, newElementsAttr);
  return success();
}

void populateMhloFolderPatterns(MLIRContext* context,
                                RewritePatternSet* patterns,
                                int64_t opFoldLimit) {
  patterns->add<ConvertOpFolder>(context, opFoldLimit);
}
}  // end namespace mhlo
}  // end namespace mlir
