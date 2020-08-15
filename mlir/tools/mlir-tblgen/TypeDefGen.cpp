//===- TypeDefGen.cpp - MLIR typeDef definitions generator ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TypeDefGen uses the description of typeDefs to generate C++ definitions.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Interfaces.h"
#include "mlir/TableGen/OpClass.h"
#include "mlir/TableGen/OpTrait.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/TableGen/TypeDef.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#define DEBUG_TYPE "mlir-tblgen-opdefgen"

using namespace mlir;
using namespace mlir::tblgen;

static llvm::cl::OptionCategory typedefGenCat("Options for -gen-typedef-*");
static llvm::cl::opt<std::string>
    selectedDialect("typedefs-dialect", llvm::cl::desc("Gen types for this dialect"),
                    llvm::cl::cat(typedefGenCat), llvm::cl::CommaSeparated);

/// Utility iterator used for filtering records for a specific typeDef.
namespace {
using TypeDefFilterIterator =
    llvm::filter_iterator<ArrayRef<llvm::Record *>::iterator,
                          std::function<bool(const llvm::Record *)>>;
} // end anonymous namespace


//===----------------------------------------------------------------------===//
// GEN: TypeDef declarations
//===----------------------------------------------------------------------===//

/// The code block for the start of a typeDef class declaration.
///
/// {0}: The name of the typeDef class.
/// {1}: The typeDef storage class namespace.
/// {2}: The storage class name
/// {3}: The list of members as a list of arguments
static const char *const typeDefDeclBeginStr = R"(
  namespace {1} {
    class {1};
  }
  class Type : public {0}::TypeBase<{0}, Type,
                                        {1}::{2}> {
public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    static {0} get(::mlir::MLIRContext* ctxt{3});
)";

/// The code block for the verifyConstructionInvariants and getChecked
///
/// {0}: List of members, arguments style
/// {1}: C++ type class name
static const char *const typeDefDeclVerifyStr = R"(
    static LogicalResult verifyConstructionInvariants(Location loc{0});
    static {1} getChecked(Location loc{0});
)";


    // static StringRef getMnemonic() { return "{3}"; }
    // static Type parse(mlir::MLIRContext* ctxt, mlir::DialectAsmParser& parser);
    // void print(mlir::DialectAsmPrinter& printer) const;

static std::string constructMembersArguments(TypeDef& typeDef) {
  SmallVector<TypeMember, 4> members;
  typeDef.getMembers(members);

  std::string membersArguments;
  llvm::raw_string_ostream args(membersArguments);
  for (auto member : members) {
    args << llvm::formatv(", {0} {1}", member.getCppType(), member.getName());
  }
  return membersArguments;
}

static std::string constructMembersNames(TypeDef& typeDef) {
  SmallVector<TypeMember, 4> members;
  typeDef.getMembers(members);

  std::string membersArguments;
  llvm::raw_string_ostream args(membersArguments);
  for (auto member : members) {
    args << llvm::formatv(", {1}", member.getName());
  }
  return membersArguments;
}

/// Generate the declaration for the given typeDef class.
static void emitTypeDefDecl(TypeDef &typeDef, raw_ostream &os) {
  std::string membersArguments = constructMembersArguments(typeDef);
  // std::string membersNames;
  os << llvm::formatv(typeDefDeclBeginStr,
            typeDef.getCppClassName(),
            typeDef.getStorageNamespace(),
            typeDef.getStorageClassName(),
            membersArguments);
  
  if (typeDef.genVerifyInvariantsDecl())
    os << llvm::formatv(typeDefDeclVerifyStr,
            membersArguments,
            typeDef.getCppClassName());

  // // Check for any attributes/types registered to this typeDef.  If there are,
  // // add the hooks for parsing/printing.
  // if (!typeDefAttrs.empty())
  //   os << attrParserDecl;
  // if (!typeDefTypes.empty())
  //   os << typeParserDecl;

  // // Add the decls for the various features of the typeDef.
  // if (typeDef.hasConstantMaterializer())
  //   os << constantMaterializerDecl;
  // if (typeDef.hasOperationAttrVerify())
  //   os << opAttrVerifierDecl;
  // if (typeDef.hasRegionArgAttrVerify())
  //   os << regionArgAttrVerifierDecl;
  // if (typeDef.hasRegionResultAttrVerify())
  //   os << regionResultAttrVerifierDecl;
  // if (llvm::Optional<StringRef> extraDecl = typeDef.getExtraClassDeclaration())
  //   os << *extraDecl;

  // // End the typeDef decl.
  // os << "};\n";
}

/// Find all the TypeDefs for the specified dialect. If no dialect specified and
/// can only find one dialect's types, use that.
static bool findAllTypeDefs(const llvm::RecordKeeper &recordKeeper,
                            SmallVectorImpl<TypeDef>& typeDefs) {
  auto recDefs = recordKeeper.getAllDerivedDefinitions("TypeDef");
  auto defs = llvm::map_range(recDefs,
    [&](const llvm::Record* rec) { return TypeDef(rec); } );
  if (defs.empty())
    return false;
  
  StringRef dialectName;
  if (selectedDialect.getNumOccurrences() == 0) {
    if (defs.empty())
      return false;
    
    llvm::SmallSet<Dialect, 4> dialects;
    for (auto typeDef: defs) {
      dialects.insert(typeDef.getDialect());
    }
    if (dialects.size() != 1) {
      llvm::errs() << "TypeDefs belonging to more than one dialect. Must select one via '--dialect'\n";
      return true;
    }

    dialectName = (*dialects.begin()).getName();
  } else if (selectedDialect.getNumOccurrences() == 1) {
    dialectName = selectedDialect.getValue();
  } else {
    llvm::errs() << "cannot select multiple dialects for which to generate types"
                    "via '--dialect'\n";
    return true;
  }

  for (auto typeDef: defs) {
    if (typeDef.getDialect().getName().equals(dialectName))
      typeDefs.push_back(typeDef);
  }
  return false;
}

static bool emitTypeDefDecls(const llvm::RecordKeeper &recordKeeper,
                             raw_ostream &os) {
  emitSourceFileHeader("TypeDef Declarations", os);

  SmallVector<TypeDef, 16> typeDefs;
  if (findAllTypeDefs(recordKeeper, typeDefs))
    return true;

  for (auto typeDef : typeDefs) {
    emitTypeDefDecl(typeDef, os);
  }

  return false;
}

//===----------------------------------------------------------------------===//
// GEN: TypeDef registration hooks
//===----------------------------------------------------------------------===//

static mlir::GenRegistration
    genTypeDefDecls("gen-typedef-decls", "Generate TypeDef declarations",
                    [](const llvm::RecordKeeper &records, raw_ostream &os) {
                      return emitTypeDefDecls(records, os);
                    });
