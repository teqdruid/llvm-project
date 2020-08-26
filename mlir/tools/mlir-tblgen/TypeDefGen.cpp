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
#include "mlir/TableGen/ParserPrinterHelpers.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#define DEBUG_TYPE "mlir-tblgen-typedefgen"

using namespace mlir;
using namespace mlir::tblgen;

static llvm::cl::OptionCategory typedefGenCat("Options for -gen-typedef-*");
static llvm::cl::opt<std::string>
    selectedDialect("typedefs-dialect", llvm::cl::desc("Gen types for this dialect"),
                    llvm::cl::cat(typedefGenCat), llvm::cl::CommaSeparated);


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
    
    llvm::SmallSet<mlir::tblgen::Dialect, 4> dialects;
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

/// Create a list of members and types for function decls
static std::string constructMemberParameters(TypeDef& typeDef, bool prependComma) {
  SmallVector<std::string, 4> members;
  if (prependComma)
    members.push_back("");
  typeDef.getMembersAs<std::string>(members, [](auto member) {
    return (member.getCppType() + " " + member.getName()).str(); });
  if (members.size() > 0)
    return llvm::join(members, ", ");
  return "";
}

//===----------------------------------------------------------------------===//
// GEN: TypeDef declarations
//===----------------------------------------------------------------------===//

/// The code block for the start of a typeDef class declaration -- singleton case
///
/// {0}: The name of the typeDef class.
static const char *const typeDefDeclSingletonBeginStr = R"(
  class {0}: public Type::TypeBase<{0}, Type, TypeStorage> {{
public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

)";

/// The code block for the start of a typeDef class declaration -- parametric case
///
/// {0}: The name of the typeDef class.
/// {1}: The typeDef storage class namespace.
/// {2}: The storage class name
static const char *const typeDefDeclParametricBeginStr = R"(
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



/// Generate the declaration for the given typeDef class.
static void emitTypeDefDecl(TypeDef &typeDef, raw_ostream &os) {
  std::string memberParameters = constructMemberParameters(typeDef, true);
  // std::string membersNames;
  if (typeDef.getNumMembers() == 0)
    os << llvm::formatv(typeDefDeclSingletonBeginStr,
              typeDef.getCppClassName(),
              typeDef.getStorageNamespace(),
              typeDef.getStorageClassName());
  else
    os << llvm::formatv(typeDefDeclParametricBeginStr,
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

static bool emitTypeDefDecls(const llvm::RecordKeeper &recordKeeper,
                             raw_ostream &os) {
  emitSourceFileHeader("TypeDef Declarations", os);

  SmallVector<TypeDef, 16> typeDefs;
  if (findAllTypeDefs(recordKeeper, typeDefs))
    return true;

  IfDefScope scope("GET_TYPEDEF_CLASSES", os);
  os << "  Type generatedTypeParser(mlir::MLIRContext* ctxt, mlir::DialectAsmParser& parser, llvm::StringRef mnenomic);\n";
  os << "  bool generatedTypePrinter(Type type, mlir::DialectAsmPrinter& printer);\n";
  os << "\n";

  for (auto typeDef : typeDefs) {
    os << "  class " << typeDef.getCppClassName() << ";\n";
  }
  for (auto typeDef : typeDefs) {
    emitTypeDefDecl(typeDef, os);
  }

  return false;
}

//===----------------------------------------------------------------------===//
// GEN: TypeDef definitions
//===----------------------------------------------------------------------===//

static bool emitTypeDefList(SmallVectorImpl<TypeDef>& typeDefs,
                            raw_ostream& os) {
  IfDefScope scope("GET_TYPEDEF_LIST", os);
  for (auto i = typeDefs.begin(); i != typeDefs.end(); i++) {
    os << "  " << i->getCppClassName();
    if (i < typeDefs.end() - 1)
      os << ",\n";
    else
      os << "\n";
  }
  return false;
}


/// Beginning of storage class
/// {0}: Storage class namespace
/// {1}: Storage class c++ name
/// {2}: Members parameters
/// {3}: Member initialzer string
/// {4}: Member name list;
/// {5}: Member types
static const char* const typeDefStorageClassBegin = R"(
namespace {0} {{
  struct {1} : public TypeStorage {{
      {1} ({2})
          : {3} {{ }

      /// The hash key for this storage is a pair of the integer and type params.
      using KeyTy = std::tuple<{5}>;

      /// Define the comparison function for the key type.
      bool operator==(const KeyTy &key) const {{
          return key == KeyTy({4});
      }

      static llvm::hash_code hashKey(const KeyTy &key) {
        auto [{4}] = key;
)";

/// The storage class' constructor template
/// {0}: storage class name
/// {1}: list of members
static const char *const typeDefStorageClassConstructorBegin = R"(
      /// Define a construction method for creating a new instance of this storage.
      static {0} *construct(TypeStorageAllocator &allocator,
                                          const KeyTy &key) {
          auto [{1}] = key;
)";

/// The storage class' constructor return template
/// {0}: storage class name
/// {1}: list of members
static const char *const typeDefStorageClassConstructorReturn = R"(
          return new (allocator.allocate<{0}>())
              {0}({1});
      }
)";

static std::string constructMembersInitializers(TypeDef& typeDef) {
  SmallVector<std::string, 4> members;
  typeDef.getMembersAs<std::string>(members, [](auto member) {
     return (member.getName() + "(" + member.getName() + ")").str(); } );
  return llvm::join(members, ", ");
}

static bool emitCustomAllocationCode(TypeDef& typeDef, raw_ostream& os) {
  SmallVector<TypeMember, 4> members;
  typeDef.getMembers(members);
  auto fmtCtxt = FmtContext()
    .addSubst("_allocator", "allocator");
  for (auto member : members) {
    auto allocCode = member.getAllocator();
    if (allocCode) {
      fmtCtxt.withSelf(member.getName());
      fmtCtxt.addSubst("_dst", member.getName());
      auto fmtObj = tgfmt(*allocCode, &fmtCtxt);
      os << "          ";
      fmtObj.format(os);
      os << "\n";
    }
  }
  return false;
}

static bool emitStorageClass(TypeDef typeDef,
                            raw_ostream& os) {
  SmallVector<TypeMember, 4> members;
  typeDef.getMembers(members);

  auto memberNames = llvm::map_range(members, [](TypeMember member) { return member.getName(); });
  auto memberTypes = llvm::map_range(members, [](TypeMember member) { return member.getCppType(); });

  auto memberList = llvm::join(memberNames, ", ");
  auto memberTypeList = llvm::join(memberTypes, ", ");
  auto memberParameters = constructMemberParameters(typeDef, false);
  auto memberInits = constructMembersInitializers(typeDef);
  os << llvm::formatv(typeDefStorageClassBegin,
            typeDef.getStorageNamespace(),
            typeDef.getStorageClassName(),
            memberParameters,
            memberInits,
            memberList,
            memberTypeList);
  os << "        return llvm::hash_combine(\n";
  for (auto memberIter = members.begin(); memberIter < members.end(); memberIter++) {
    // os << llvm::formatv("          llvm::hash_value({0})", memberIter->getName());
    os << "          " << memberIter->getName();
    if (memberIter < members.end() - 1) {
      os << ",\n";
    }
  }
  os << ");\n";
  os << "      }\n";

  if (typeDef.hasStorageCustomConstructor())
    os << "static " << typeDef.getStorageClassName() << " *construct(TypeStorageAllocator &allocator, const KeyTy &key);\n";
  else {
    os << llvm::formatv(typeDefStorageClassConstructorBegin,
            typeDef.getStorageClassName(),
            memberList);
    if (emitCustomAllocationCode(typeDef, os))
      return false;
    os << llvm::formatv(typeDefStorageClassConstructorReturn,
            typeDef.getStorageClassName(),
            memberList);
  }

  for (auto member : members) {
    os << "      " << member.getCppType() << " " << member.getName() << ";\n";
  }
  os << "  };\n";
  os << "};\n";

  return false;
}

// Emit code which prints the type to printer
static bool emitPrinterAutogen(TypeDef typeDef, raw_ostream& os) {
  if (auto mnemonic = typeDef.getMnemonic()) {
    os << "  printer << \"" << *mnemonic << "\";\n";
    SmallVector<TypeMember, 4> members;
    typeDef.getMembers(members);
    if (members.size() > 0) {
      os << "  printer << \"<\";\n";
      for (auto memberIter = members.begin(); memberIter < members.end(); memberIter++) {
        os << "  ::mlir::tblgen::parser_helpers::print<" << memberIter->getCppType()
           << ">::go(printer, getImpl()->" << memberIter->getName() << ");\n";
        if (memberIter < members.end() - 1) {
          os << "  printer << \", \";\n";
        }
      }
      os << "  printer << \">\";\n";
    }
  }
  return false;
}

static bool emitParserAutogen(TypeDef typeDef, raw_ostream& os) {
  SmallVector<TypeMember, 4> members;
  typeDef.getMembers(members);
  if (members.size() > 0) {
    os << "  llvm::BumpPtrAllocator allocator;\n";
    os << "  if (parser.parseLess()) return Type();\n";
    for (auto memberIter = members.begin(); memberIter < members.end(); memberIter++) {
      os << "  " << memberIter->getCppType() << " " << memberIter->getName() << ";\n";
      os << "  if (::mlir::tblgen::parser_helpers::parse<" << memberIter->getCppType() << ">::go(ctxt, parser, allocator, " << memberIter->getName() << "))\n";
      os << "    return Type();\n";
      if (memberIter < members.end() - 1) {
        os << "  if (parser.parseComma()) return Type();\n";
      }
    }
    os << "  if (parser.parseGreater()) return Type();\n";
    auto memberNames = llvm::map_range(members, [](TypeMember member) { return member.getName(); });
    os << "  return get(ctxt, " << llvm::join(memberNames, ", ") << ");\n";
  }
  return false;
}

// Print all the typedef-specific definition code
static bool emitTypeDefDef(TypeDef typeDef,
                           raw_ostream& os) {
  SmallVector<TypeMember, 4> members;
  typeDef.getMembers(members);

  if (typeDef.genStorageClass() && typeDef.getNumMembers() > 0)
    if (emitStorageClass(typeDef, os))
      return true;

  // auto memberNames = llvm::map_range(members, [](TypeMember member) { return member.getName(); });
  // os << typeDef.getCppClassName() << " " << typeDef.getCppClassName() << "::get(::mlir::MLIRContext* ctxt"
  //    << llvm::join(memberNames, ", ") << ") {\n";
  // os << "  return Base::get"
  if (typeDef.genAccessors()) {


    for (auto member : members) {
      SmallString<16> name = member.getName();
      name[0] = llvm::toUpper(name[0]);
      os << llvm::formatv("{0} {3}::get{1}() { return getImpl()->{2}; }\n",
        member.getCppType(), name, member.getName(), typeDef.getCppClassName());
    }
  }


  auto printerCode = typeDef.getPrinterCode();
  if (printerCode && typeDef.getMnemonic()) {
    os << "void " << typeDef.getCppClassName() << "::print(mlir::DialectAsmPrinter& printer) const {\n";
    if (*printerCode == "") emitPrinterAutogen(typeDef, os);
    else os << *printerCode << "\n";
    os << "}\n";
  }

  auto parserCode = typeDef.getParserCode();
  if (parserCode && typeDef.getMnemonic()) {
    os << "Type " << typeDef.getCppClassName() << "::parse(mlir::MLIRContext* ctxt, mlir::DialectAsmParser& parser) {\n";
    if (*parserCode == "") emitParserAutogen(typeDef, os);
    else os << *parserCode << "\n";
    os << "}\n";
  }

  return false;
}

// Emit the dialect printer/parser dispatch. Client code should call these
// functions from their dialect's print/parse methods.
static bool emitParsePrintDispatch(SmallVectorImpl<TypeDef>& typeDefs,
                            raw_ostream& os) {
  os << "Type generatedTypeParser(mlir::MLIRContext* ctxt, mlir::DialectAsmParser& parser, llvm::StringRef mnemonic) {\n";
  for (auto typeDef : typeDefs) {
    if (typeDef.getMnemonic())
      os << llvm::formatv("  if (mnemonic == {0}::getMnemonic()) return {0}::parse(ctxt, parser);\n", typeDef.getCppClassName());
  }
  os << "  return Type();\n";
  os << "}\n\n";

  os << "bool generatedTypePrinter(Type type, mlir::DialectAsmPrinter& printer) {\n"
     << "  bool notfound = false;\n"
     << "  TypeSwitch<Type>(type)\n";
  for (auto typeDef : typeDefs) {
    if (typeDef.getMnemonic())
      os << llvm::formatv("    .Case<{0}>([&](Type t) {{ t.dyn_cast<{0}>().print(printer); })\n", typeDef.getCppClassName());
  }
  os << "    .Default([&notfound](Type) { notfound = true; });\n"
     << "  return notfound;\n"
     << "}\n\n";
  return false;
}

static bool emitTypeDefDefs(const llvm::RecordKeeper &recordKeeper,
                             raw_ostream &os) {
  emitSourceFileHeader("TypeDef Definitions", os);

  SmallVector<TypeDef, 16> typeDefs;
  if (findAllTypeDefs(recordKeeper, typeDefs))
    return true;
  
  if (emitTypeDefList(typeDefs, os))
    return true;

  IfDefScope scope("GET_TYPEDEF_CLASSES", os);
  if (emitParsePrintDispatch(typeDefs, os))
    return true;
  for (auto typeDef : typeDefs) {
    emitTypeDefDef(typeDef, os);
  }

  return false;
}

//===----------------------------------------------------------------------===//
// GEN: TypeDef registration hooks
//===----------------------------------------------------------------------===//

static mlir::GenRegistration
    genTypeDefDefs("gen-typedef-defs", "Generate TypeDef definitions",
                    [](const llvm::RecordKeeper &records, raw_ostream &os) {
                      return emitTypeDefDefs(records, os);
                    });

static mlir::GenRegistration
    genTypeDefDecls("gen-typedef-decls", "Generate TypeDef declarations",
                    [](const llvm::RecordKeeper &records, raw_ostream &os) {
                      return emitTypeDefDecls(records, os);
                    });
