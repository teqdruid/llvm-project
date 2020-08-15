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

extern llvm::cl::opt<std::string> selectedDialect;

/// Utility iterator used for filtering records for a specific typeDef.
namespace {
using TypeDefFilterIterator =
    llvm::filter_iterator<ArrayRef<llvm::Record *>::iterator,
                          std::function<bool(const llvm::Record *)>>;
} // end anonymous namespace

/// Given a set of records for a T, filter the ones that correspond to
/// the given dialect name.
static iterator_range<TypeDefFilterIterator>
filterForDialect(ArrayRef<llvm::Record *> records, StringRef dialectName) {
  auto filterFn = [&](const llvm::Record *record) {
    return TypeDef(record).getDialect().getName().equals(dialectName);
  };
  return {TypeDefFilterIterator(records.begin(), records.end(), filterFn),
          TypeDefFilterIterator(records.end(), records.end(), filterFn)};
}

//===----------------------------------------------------------------------===//
// GEN: TypeDef declarations
//===----------------------------------------------------------------------===//

/// The code block for the start of a typeDef class declaration.
///
/// {0}: The name of the typeDef class.
/// {1}: The typeDef namespace.
static const char *const typeDefDeclBeginStr = R"(
class {0} : public ::mlir::TypeDef {
  explicit {0}(::mlir::MLIRContext *context)
    : ::mlir::TypeDef(getDialectNamespace(), context,
      ::mlir::TypeID::get<{0}>()) {{
    initialize();
  }
  void initialize();
  friend class ::mlir::MLIRContext;
public:
  static ::llvm::StringRef getTypeDefNamespace() { return "{1}"; }
)";

/// Generate the declaration for the given typeDef class.
static void emitTypeDefDecl(TypeDef &typeDef, raw_ostream &os) {
  // Emit the start of the decl.
  // std::string cppName = typeDef.getCppClassName();
  // os << llvm::formatv(typeDefDeclBeginStr, cppName, typeDef.getName());

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


static bool findAllTypeDefs(const llvm::RecordKeeper &recordKeeper,
                                          SmallVectorImpl<TypeDef>& typeDefs) {
  auto recDefs = recordKeeper.getAllDerivedDefinitions("TypeDef");
  auto defs = llvm::map_range(recDefs,
    [&](const llvm::Record* rec) { return TypeDef(rec); } );
  if (defs.empty())
    return false;
  
  // std::vector<TypeDef> tds(defs.begin(), defs.end());

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
  // llvm::outs() << "selected dialect: " << dialectName << "\n";
  return false;
}

static bool emitTypeDefDecls(const llvm::RecordKeeper &recordKeeper,
                             raw_ostream &os) {
  emitSourceFileHeader("TypeDef Declarations", os);

  SmallVector<TypeDef, 16> typeDefs;
  if (findAllTypeDefs(recordKeeper, typeDefs))
    return true;

  for (auto typeDef : typeDefs) {
    os << typeDef.getName() << "\n";
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
