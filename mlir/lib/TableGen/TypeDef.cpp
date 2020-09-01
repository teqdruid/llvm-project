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
#include "llvm/ADT/StringExtras.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;

Dialect TypeDef::getDialect() const {
  return Dialect(
      dyn_cast<llvm::DefInit>(def->getValue("owningDialect")->getValue())
          ->getDef());
}

StringRef TypeDef::getName() const { return def->getName(); }
StringRef TypeDef::getCppClassName() const {
  return def->getValueAsString("cppClassName");
}

StringRef TypeDef::getStorageClassName() const {
  return def->getValueAsString("storageClass");
}
StringRef TypeDef::getStorageNamespace() const {
  return def->getValueAsString("storageNamespace");
}

bool TypeDef::genStorageClass() const {
  return def->getValueAsBit("genStorageClass");
}
bool TypeDef::hasStorageCustomConstructor() const {
  return def->getValueAsBit("hasStorageCustomConstructor");
}
void TypeDef::getMembers(SmallVectorImpl<TypeMember> &members) const {
  auto membersDag = def->getValueAsDag("members");
  if (membersDag != nullptr)
    for (unsigned i = 0; i < membersDag->getNumArgs(); i++)
      members.push_back(TypeMember(membersDag, i));
}
unsigned TypeDef::getNumMembers() const {
  auto membersDag = def->getValueAsDag("members");
  if (membersDag == nullptr)
    return 0;
  return membersDag->getNumArgs();
}
llvm::Optional<StringRef> TypeDef::getMnemonic() const {
  auto code = def->getValue("mnemonic");
  if (llvm::StringInit *CI = dyn_cast<llvm::StringInit>(code->getValue()))
    return CI->getValue();
  if (isa<llvm::UnsetInit>(code->getValue()))
    return llvm::Optional<StringRef>();

  llvm::PrintFatalError(
      def->getLoc(),
      "Record `" + def->getName() +
          "', field `printer' does not have a code initializer!");
}
llvm::Optional<StringRef> TypeDef::getPrinterCode() const {
  auto code = def->getValue("printer");
  if (llvm::CodeInit *CI = dyn_cast<llvm::CodeInit>(code->getValue()))
    return CI->getValue();
  if (isa<llvm::UnsetInit>(code->getValue()))
    return llvm::Optional<StringRef>();

  llvm::PrintFatalError(
      def->getLoc(),
      "Record `" + def->getName() +
          "', field `printer' does not have a code initializer!");
}
llvm::Optional<StringRef> TypeDef::getParserCode() const {
  return def->getValueAsString("parser");
}
bool TypeDef::genAccessors() const {
  return def->getValueAsBit("genAccessors");
}
bool TypeDef::genVerifyInvariantsDecl() const {
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

StringRef TypeMember::getName() const {
  return def->getArgName(num)->getValue();
}
llvm::Optional<StringRef> TypeMember::getAllocator() const {
  auto memberType = def->getArg(num);
  if (auto stringType = dyn_cast<llvm::StringInit>(memberType)) {
    return llvm::Optional<StringRef>();
  } else if (auto typeMember = dyn_cast<llvm::DefInit>(memberType)) {
    auto code = typeMember->getDef()->getValue("allocator");
    if (llvm::CodeInit *CI = dyn_cast<llvm::CodeInit>(code->getValue()))
      return CI->getValue();
    if (isa<llvm::UnsetInit>(code->getValue()))
      return llvm::Optional<StringRef>();

    llvm::PrintFatalError(
        typeMember->getDef()->getLoc(),
        "Record `" + def->getArgName(num)->getValue() +
            "', field `printer' does not have a code initializer!");
  } else {
    llvm::errs() << "Members DAG arguments must be either strings or defs "
                    "which inherit from TypeMember\n";
    return StringRef();
  }
}
StringRef TypeMember::getCppType() const {
  auto memberType = def->getArg(num);
  if (auto stringType = dyn_cast<llvm::StringInit>(memberType)) {
    return stringType->getValue();
  } else if (auto typeMember = dyn_cast<llvm::DefInit>(memberType)) {
    return typeMember->getDef()->getValueAsString("cppType");
  } else {
    llvm::errs() << "Members DAG arguments must be either strings or defs "
                    "which inherit from TypeMember\n";
    return StringRef();
  }
}
