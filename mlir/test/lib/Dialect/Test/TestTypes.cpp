//===- TestTypes.cpp - MLIR Test Dialect Types ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains types defined by the TestDialect for testing various
// features of MLIR.
//
//===----------------------------------------------------------------------===//

#include "TestTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/TableGen/TypeDefGenHelpers.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace tblgen {
namespace parser_helpers {

// Custom parser for SignednessSemantics
template <>
struct Parse<TestIntegerType::SignednessSemantics> {
  static ParseResult go(MLIRContext *ctxt, DialectAsmParser &parser,
                        StringRef parameterName,
                        TestIntegerType::SignednessSemantics &result) {
    StringRef signStr;
    auto loc = parser.getCurrentLocation();
    if (parser.parseKeyword(&signStr))
      return mlir::failure();
    if (signStr.compare_lower("u") || signStr.compare_lower("unsigned"))
      result = TestIntegerType::SignednessSemantics::Unsigned;
    else if (signStr.compare_lower("s") || signStr.compare_lower("signed"))
      result = TestIntegerType::SignednessSemantics::Signed;
    else if (signStr.compare_lower("n") || signStr.compare_lower("none"))
      result = TestIntegerType::SignednessSemantics::Signless;
    else {
      parser.emitError(loc, "expected signed, unsigned, or none");
      return mlir::failure();
    }
    return mlir::success();
  }
};

// Custom printer for SignednessSemantics
template <>
struct Print<TestIntegerType::SignednessSemantics> {
  static void go(DialectAsmPrinter &printer,
                 const TestIntegerType::SignednessSemantics &ss) {
    switch (ss) {
    case TestIntegerType::SignednessSemantics::Unsigned:
      printer << "unsigned";
      break;
    case TestIntegerType::SignednessSemantics::Signed:
      printer << "signed";
      break;
    case TestIntegerType::SignednessSemantics::Signless:
      printer << "none";
      break;
    }
  }
};

} // namespace parser_helpers
} // namespace tblgen

bool operator==(const FieldInfo &a, const FieldInfo &b) {
  return a.name == b.name && a.type == b.type;
}

llvm::hash_code hash_value(const FieldInfo &fi) {
  return llvm::hash_combine(fi.name, fi.type);
}

// Example type validity checker
LogicalResult TestIntegerType::verifyConstructionInvariants(
    mlir::Location loc, mlir::TestIntegerType::SignednessSemantics ss,
    unsigned int width) {

  if (width > 8)
    return mlir::failure();
  return mlir::success();
}

struct TestType;
} // end namespace mlir

#define GET_TYPEDEF_CLASSES
#include "TestTypeDefs.cpp.inc"
