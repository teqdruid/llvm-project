//===- TypeDef.cpp - TypeDef wrapper class --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TypeDef wrapper to simplify using TableGen Record defining a MLIR dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/TypeDef.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;

Dialect TypeDef::getDialect() const {
  return Dialect(
    dyn_cast<llvm::DefInit>(def->getValue("owningDialect")->getValue())->getDef()
  );
}

StringRef TypeDef::getName() const { return def->getName(); }

StringRef TypeDef::getStorageClassName() const { return def->getValueAsString("storageClass"); }
StringRef TypeDef::getStorageNamespace() const { return def->getValueAsString("storageNamespace"); }

bool TypeDef::genStorageClass() const{
  return def->getValueAsBit("genStorageClass");
}
bool TypeDef::hasStorageCustomConstructor() const{
  return def->getValueAsBit("hasStorageCustomConstructor");
}
llvm::Optional<ArrayRef<TypeMember>> TypeDef::getMembers() const{
  // auto membersDag = def->getValueAsDag("members");
  return ArrayRef<TypeMember>();
}
StringRef TypeDef::getMnemonic() const{
  return def->getValueAsString("mnemonic");
}
StringRef TypeDef::getPrinterCode() const{
  return def->getValueAsString("printer");
}
StringRef TypeDef::getParserCode() const{
  return def->getValueAsString("parser");
}
bool TypeDef::genAccessors() const{
  return def->getValueAsBit("genAccessors");
}
bool TypeDef::genVerifyInvariantsDecl() const{
  return def->getValueAsBit("genVerifyInvariantsDecl");
}

llvm::Optional<StringRef> TypeDef::getExtraDecls() const {
  auto value = def->getValueAsString("extraDecls");
  return value.empty() ? llvm::Optional<StringRef>() : value;
}

bool TypeDef::operator==(const TypeDef &other) const {
  return def == other.def;
}

bool TypeDef::operator<(const TypeDef &other) const {
  return getName() < other.getName();
}
