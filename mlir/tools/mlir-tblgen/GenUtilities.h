//===- DocGenUtilities.h - MLIR doc gen utilities ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines common utilities for generating documents from tablegen
// structures.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRTBLGEN_GENUTILITIES_H_
#define MLIR_TOOLS_MLIRTBLGEN_GENUTILITIES_H_

#include "llvm/ADT/StringExtras.h"

namespace mlir {
namespace tblgen {

// Simple RAII helper for defining ifdef-undef-endif scopes.
class IfDefScope {
public:
  IfDefScope(llvm::StringRef name, llvm::raw_ostream &os);
  ~IfDefScope();

private:
  llvm::StringRef name;
  llvm::raw_ostream &os;
};

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TOOLS_MLIRTBLGEN_GENUTILITIES_H_
