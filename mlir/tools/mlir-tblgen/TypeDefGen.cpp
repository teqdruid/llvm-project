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

#include "GenUtilities.h"
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
static const char *const typeDefDeclBeginStr = R"(
  namespace {1} {
    class {2};
  }
  class {0}: public Type::TypeBase<{0}, Type,
                                        {1}::{2}> {{
public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

)";

/// {0}: The name of the typeDef class.
/// {1}: The list of members as a list of arguments
static const char *const typeDefAfterExtra = R"(

    static {0} get(::mlir::MLIRContext* ctxt{1});

    static Type parse(mlir::MLIRContext* ctxt, mlir::DialectAsmParser& parser);
    void print(mlir::DialectAsmPrinter& printer) const;

)";

/// The code block for the verifyConstructionInvariants and getChecked
///
/// {0}: List of members, parameters style
/// {1}: C++ type class name
static const char *const typeDefDeclVerifyStr = R"(
    static LogicalResult verifyConstructionInvariants(Location loc{0});
    static {1} getChecked(Location loc{0});
)";

/// Create a list of members and types for function decls
static std::string constructMemberParameters(TypeDef& typeDef) {
  SmallVector<TypeMember, 4> members;
  typeDef.getMembers(members);

  std::string memberParameters;
  llvm::raw_string_ostream args(memberParameters);
  for (auto member : members) {
    args << llvm::formatv(", {0} {1}", member.getCppType(), member.getName());
  }
  return memberParameters;
}

/// Create a list of member names for function calls
static std::string constructMembersNames(TypeDef& typeDef) {
  SmallVector<TypeMember, 4> members;
  typeDef.getMembers(members);

  std::string memberArgs;
  llvm::raw_string_ostream args(memberArgs);
  for (auto member : members) {
    args << llvm::formatv(", {1}", member.getName());
  }
  return memberArgs;
}

/// Generate the declaration for the given typeDef class.
static void emitTypeDefDecl(TypeDef &typeDef, raw_ostream &os) {
  std::string memberParameters = constructMemberParameters(typeDef);
  // std::string membersNames;
  os << llvm::formatv(typeDefDeclBeginStr,
            typeDef.getCppClassName(),
            typeDef.getStorageNamespace(),
            typeDef.getStorageClassName());

  // Emit the extra declarations first in case there's a type definition in there
  if (llvm::Optional<StringRef> extraDecl = typeDef.getExtraDecls())
    os << *extraDecl;

  // Then output everything which could have c++ type names
  os << llvm::formatv(typeDefAfterExtra,
            typeDef.getCppClassName(),
            memberParameters);

  if (typeDef.genVerifyInvariantsDecl())
    os << llvm::formatv(typeDefDeclVerifyStr,
            memberParameters,
            typeDef.getCppClassName());
            
  if (auto mnenomic = typeDef.getMnemonic()) {
    os << "    static StringRef getMnemonic() { return \"" << mnenomic << "\"; }\n";
  }

  if (typeDef.genAccessors()) {
    SmallVector<TypeMember, 4> members;
    typeDef.getMembers(members);

    for (auto member : members) {
      SmallString<16> name = member.getName();
      name[0] = llvm::toUpper(name[0]);
      os << llvm::formatv("    {0} get{1}();\n", member.getCppType(), name);
    }
  }

  // End the typeDef decl.
  os << "  };\n";
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
      llvm::errs() << "TypeDefs belonging to more than one dialect. Must select one via '--typedefs-dialect'\n";
      return true;
    }

    dialectName = (*dialects.begin()).getName();
  } else if (selectedDialect.getNumOccurrences() == 1) {
    dialectName = selectedDialect.getValue();
  } else {
    llvm::errs() << "cannot select multiple dialects for which to generate types"
                    "via '--typedefs-dialect'\n";
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

  IfDefScope scope("GET_TYPEDEF_CLASSES", os);
  for (auto typeDef : typeDefs) {
    os << "  class " << typeDef.getCppClassName() << ";\n";
  }
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
