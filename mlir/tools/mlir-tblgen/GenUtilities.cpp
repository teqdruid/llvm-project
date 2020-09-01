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

#include "GenUtilities.h"
#include "llvm/ADT/StringExtras.h"

namespace mlir {
namespace tblgen {

IfDefScope::IfDefScope(llvm::StringRef name, llvm::raw_ostream &os)
    : name(name), os(os) {
  os << "#ifdef " << name << "\n"
     << "#undef " << name << "\n\n";
}

IfDefScope::~IfDefScope() { os << "\n#endif  // " << name << "\n\n"; }

} // namespace tblgen
} // namespace mlir
