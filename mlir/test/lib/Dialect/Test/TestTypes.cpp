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

struct TestType;
#define GET_TYPEDEF_CLASSES
#include "TestTypeDefs.cpp.inc"

} // end namespace mlir
