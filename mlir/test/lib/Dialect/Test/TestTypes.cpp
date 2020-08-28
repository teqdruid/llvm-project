//===- TestTypes.cpp - MLIR Test Dialect Types --------------------*- C++ -*-===//
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

namespace llvm {
    class hash_code;
    hash_code hash_value(float f);
}

#include "TestTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/TableGen/TypeDefGenHelpers.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {

namespace tblgen {
namespace parser_helpers {

template<>
struct parse<TestIntegerType::SignednessSemantics> {
  static ParseResult go(
      MLIRContext* ctxt,
      DialectAsmParser& parser,
      StringRef typeStr,
      TestIntegerType::SignednessSemantics& result)
    {
      StringRef signStr;
      auto loc = parser.getCurrentLocation();
      if (parser.parseKeyword(&signStr)) return mlir::failure();
      if (signStr.compare_lower("u") || signStr.compare_lower("unsigned")) result = TestIntegerType::SignednessSemantics::Unsigned;
      else if (signStr.compare_lower("s") || signStr.compare_lower("signed")) result = TestIntegerType::SignednessSemantics::Signed;
      else if (signStr.compare_lower("n") || signStr.compare_lower("none")) result = TestIntegerType::SignednessSemantics::Signless;
      else {
        parser.emitError(loc, "expected signed, unsigned, or none");
        return mlir::failure();
      }
      return mlir::success();
    }
};

template<>
struct print<TestIntegerType::SignednessSemantics> {
  static void go(DialectAsmPrinter& printer, const TestIntegerType::SignednessSemantics& ss) {
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

}
}

LogicalResult TestIntegerType::verifyConstructionInvariants(
    mlir::Location loc,
    mlir::TestIntegerType::SignednessSemantics ss,
    unsigned int width) {

  if (width > 8) return mlir::failure();
  return mlir::success();
}

struct TestType;
#define GET_TYPEDEF_CLASSES
#include "TestTypeDefs.cpp.inc"



} // end namespace mlir
