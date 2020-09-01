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

#ifndef MLIR_TABLEGEN_TYPEDEF_H
#define MLIR_TABLEGEN_TYPEDEF_H

#include "mlir/Support/LLVM.h"
#include "mlir/TableGen/Dialect.h"
#include "llvm/TableGen/Record.h"
#include <functional>
#include <string>

namespace llvm {
class Record;
class DagInit;
} // end namespace llvm

namespace mlir {
namespace tblgen {

class TypeMember;

// Wrapper class that contains a MLIR dialect's information defined in TableGen
// and provides helper methods for accessing them.
class TypeDef {
public:
  explicit TypeDef(const llvm::Record *def) : def(def) {}

  // Get the dialect for which this type belongs
  Dialect getDialect() const;

  // Returns the name of this TypeDef record
  StringRef getName() const;

  // Returns the name of the C++ class to generate
  StringRef getCppClassName() const;

  // Returns the name of the storage class for this type
  StringRef getStorageClassName() const;

  // Returns the C++ namespace for this types storage class
  StringRef getStorageNamespace() const;

  // Returns true if we should generate the storage class
  bool genStorageClass() const;

  // I don't remember what this is for or how it'd work...
  bool hasStorageCustomConstructor() const;

  // Return the list of fields for the storage class and constructors
  void getMembers(SmallVectorImpl<TypeMember> &) const;
  unsigned getNumMembers() const;

  // Iterate though members, applying a map function before adding to list
  template <typename T>
  void getMembersAs(SmallVectorImpl<T> &members,
                    std::function<T(TypeMember)> map) const;

  // Return the keyword/mnemonic to use in the printer/parser methods if we are
  // supposed to auto-generate them
  llvm::Optional<StringRef> getMnemonic() const;

  // Returns the code to use as the types printer method. If empty, generate
  // just the declaration. If null and mnemonic is non-null, generate the
  // declaration and definition.
  llvm::Optional<StringRef> getPrinterCode() const;

  // Returns the code to use as the types parser method. If empty, generate
  // just the declaration. If null and mnemonic is non-null, generate the
  // declaration and definition.
  llvm::Optional<StringRef> getParserCode() const;

  // Should we generate accessors based on the types members?
  bool genAccessors() const;

  // Return true if we need to generate the verifyConstructionInvariants
  // declaration and getChecked method
  bool genVerifyInvariantsDecl() const;

  // Returns the dialects extra class declaration code.
  llvm::Optional<StringRef> getExtraDecls() const;

  // Returns whether two dialects are equal by checking the equality of the
  // underlying record.
  bool operator==(const TypeDef &other) const;

  // Compares two dialects by comparing the names of the dialects.
  bool operator<(const TypeDef &other) const;

  // Returns whether the dialect is defined.
  operator bool() const { return def != nullptr; }

private:
  const llvm::Record *def;
};

class TypeMember {
public:
  explicit TypeMember(const llvm::DagInit *def, unsigned num)
      : def(def), num(num) {}

  StringRef getName() const;
  llvm::Optional<StringRef> getAllocator() const;
  StringRef getCppType() const;

private:
  const llvm::DagInit *def;
  const unsigned num;
};

template <typename T>
void TypeDef::getMembersAs(SmallVectorImpl<T> &members,
                           std::function<T(TypeMember)> map) const {
  auto membersDag = def->getValueAsDag("members");
  if (membersDag != nullptr)
    for (unsigned i = 0; i < membersDag->getNumArgs(); i++)
      members.push_back(map(TypeMember(membersDag, i)));
}

} // end namespace tblgen
} // end namespace mlir

#endif // MLIR_TABLEGEN_TYPEDEF_H
