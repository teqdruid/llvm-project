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

#include "mlir/Support/LogicalResult.h"
#include "mlir/TableGen/CodeGenHelpers.h"
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

#define DEBUG_TYPE "mlir-tblgen-typedefgen"

using namespace mlir;
using namespace mlir::tblgen;

static llvm::cl::OptionCategory typedefGenCat("Options for -gen-typedef-*");
static llvm::cl::opt<std::string>
    selectedDialect("typedefs-dialect",
                    llvm::cl::desc("Gen types for this dialect"),
                    llvm::cl::cat(typedefGenCat), llvm::cl::CommaSeparated);

/// Find all the TypeDefs for the specified dialect. If no dialect specified and
/// can only find one dialect's types, use that.
static mlir::LogicalResult
findAllTypeDefs(const llvm::RecordKeeper &recordKeeper,
                SmallVectorImpl<TypeDef> &typeDefs) {
  auto recDefs = recordKeeper.getAllDerivedDefinitions("TypeDef");
  auto defs = llvm::map_range(
      recDefs, [&](const llvm::Record *rec) { return TypeDef(rec); });
  if (defs.empty())
    return mlir::success();

  StringRef dialectName;
  if (selectedDialect.getNumOccurrences() == 0) {
    if (defs.empty())
      return mlir::success();

    llvm::SmallSet<mlir::tblgen::Dialect, 4> dialects;
    for (auto typeDef : defs) {
      dialects.insert(typeDef.getDialect());
    }
    if (dialects.size() != 1) {
      llvm::errs() << "TypeDefs belonging to more than one dialect. Must "
                      "select one via '--typedefs-dialect'\n";
      return mlir::failure();
    }

    dialectName = (*dialects.begin()).getName();
  } else if (selectedDialect.getNumOccurrences() == 1) {
    dialectName = selectedDialect.getValue();
  } else {
    llvm::errs()
        << "cannot select multiple dialects for which to generate types"
           "via '--typedefs-dialect'\n";
    return mlir::failure();
  }

  for (auto typeDef : defs) {
    if (typeDef.getDialect().getName().equals(dialectName))
      typeDefs.push_back(typeDef);
  }
  return mlir::success();
}

/// Create a string list of parameters and types for function decls
/// String construction helper function: parameter1Type parameter1Name,
/// parameter2Type parameter2Name
static std::string constructParameterParameters(TypeDef &typeDef,
                                                bool prependComma) {
  SmallVector<std::string, 4> parameters;
  if (prependComma)
    parameters.push_back("");
  typeDef.getParametersAs<std::string>(parameters, [](auto parameter) {
    return (parameter.getCppType() + " " + parameter.getName()).str();
  });
  if (parameters.size() > 0)
    return llvm::join(parameters, ", ");
  return "";
}

/// Create an initializer for the storage class
/// String construction helper function: parameter1(parameter1),
/// parameter2(parameter2),
/// [...]
static std::string constructParametersInitializers(TypeDef &typeDef) {
  SmallVector<std::string, 4> parameters;
  typeDef.getParametersAs<std::string>(parameters, [](auto parameter) {
    return (parameter.getName() + "(" + parameter.getName() + ")").str();
  });
  return llvm::join(parameters, ", ");
}

//===----------------------------------------------------------------------===//
// GEN: TypeDef declarations
//===----------------------------------------------------------------------===//

/// The code block for the start of a typeDef class declaration -- singleton
/// case
///
/// {0}: The name of the typeDef class.
static const char *const typeDefDeclSingletonBeginStr = R"(
  class {0}: public ::mlir::Type::TypeBase<{0}, ::mlir::Type, ::mlir::TypeStorage> {{
public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

)";

/// The code block for the start of a typeDef class declaration -- parametric
/// case
///
/// {0}: The name of the typeDef class.
/// {1}: The typeDef storage class namespace.
/// {2}: The storage class name
/// {3}: The list of parameters with types
static const char *const typeDefDeclParametricBeginStr = R"(
  namespace {1} {
    struct {2};
  }
  class {0}: public ::mlir::Type::TypeBase<{0}, ::mlir::Type,
                                        {1}::{2}> {{
public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

)";

// snippet for print/parse
static const char *const typeDefParsePrint = R"(
    static ::mlir::Type parse(::mlir::MLIRContext* ctxt, ::mlir::DialectAsmParser& parser);
    void print(::mlir::DialectAsmPrinter& printer) const;
)";

/// The code block for the verifyConstructionInvariants and getChecked
///
/// {0}: List of parameters, parameters style
/// {1}: C++ type class name
static const char *const typeDefDeclVerifyStr = R"(
    static ::mlir::LogicalResult verifyConstructionInvariants(Location loc{0});
    static {1} getChecked(Location loc{0});
)";

/// Generate the declaration for the given typeDef class.
static void emitTypeDefDecl(TypeDef &typeDef, raw_ostream &os) {
  // Emit the beginning string template: either the singleton or parametric
  // template
  if (typeDef.getNumParameters() == 0)
    os << llvm::formatv(typeDefDeclSingletonBeginStr, typeDef.getCppClassName(),
                        typeDef.getStorageNamespace(),
                        typeDef.getStorageClassName());
  else
    os << llvm::formatv(
        typeDefDeclParametricBeginStr, typeDef.getCppClassName(),
        typeDef.getStorageNamespace(), typeDef.getStorageClassName());

  // Emit the extra declarations first in case there's a type definition in
  // there
  if (llvm::Optional<StringRef> extraDecl = typeDef.getExtraDecls())
    os << *extraDecl;

  // Get the CppType1 param1, CppType2 param2 argument list
  std::string parameterParameters = constructParameterParameters(typeDef, true);

  os << llvm::formatv("  static {0} get(::mlir::MLIRContext* ctxt{1});\n",
                      typeDef.getCppClassName(), parameterParameters);

  // parse/print
  os << typeDefParsePrint;

  // verify invariants
  if (typeDef.genVerifyInvariantsDecl())
    os << llvm::formatv(typeDefDeclVerifyStr, parameterParameters,
                        typeDef.getCppClassName());

  // mnenomic, if specified
  if (auto mnenomic = typeDef.getMnemonic()) {
    os << "    static ::llvm::StringRef getMnemonic() { return \"" << mnenomic
       << "\"; }\n";
  }

  if (typeDef.genAccessors()) {
    SmallVector<TypeParameter, 4> parameters;
    typeDef.getParameters(parameters);

    for (auto parameter : parameters) {
      SmallString<16> name = parameter.getName();
      name[0] = llvm::toUpper(name[0]);
      os << llvm::formatv("    {0} get{1}() const;\n", parameter.getCppType(),
                          name);
    }
  }

  // End the typeDef decl.
  os << "  };\n";
}

/// Main entry point for decls
static bool emitTypeDefDecls(const llvm::RecordKeeper &recordKeeper,
                             raw_ostream &os) {
  emitSourceFileHeader("TypeDef Declarations", os);

  SmallVector<TypeDef, 16> typeDefs;
  if (mlir::failed(findAllTypeDefs(recordKeeper, typeDefs)))
    return true;

  IfDefScope scope("GET_TYPEDEF_CLASSES", os);
  if (typeDefs.size() > 0) {
    NamespaceEmitter nsEmitter(os, typeDefs.begin()->getDialect());
    // well known print/parse dispatch function declarations
    os << "  ::mlir::Type generatedTypeParser(::mlir::MLIRContext* ctxt, "
          "::mlir::DialectAsmParser& parser, ::llvm::StringRef mnenomic);\n";
    os << "  bool generatedTypePrinter(::mlir::Type type, "
          "::mlir::DialectAsmPrinter& "
          "printer);\n";
    os << "\n";

    // declare all the type classes first (in case they reference each other)
    for (auto typeDef : typeDefs) {
      os << "  class " << typeDef.getCppClassName() << ";\n";
    }

    // declare all the typedefs
    for (auto typeDef : typeDefs) {
      emitTypeDefDecl(typeDef, os);
    }
  }

  return false;
}

//===----------------------------------------------------------------------===//
// GEN: TypeDef list
//===----------------------------------------------------------------------===//

static mlir::LogicalResult emitTypeDefList(SmallVectorImpl<TypeDef> &typeDefs,
                                           raw_ostream &os) {
  IfDefScope scope("GET_TYPEDEF_LIST", os);
  for (auto *i = typeDefs.begin(); i != typeDefs.end(); i++) {
    os << i->getDialect().getCppNamespace() << "::" << i->getCppClassName();
    if (i < typeDefs.end() - 1)
      os << ",\n";
    else
      os << "\n";
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// GEN: TypeDef definitions
//===----------------------------------------------------------------------===//

/// Beginning of storage class
/// {0}: Storage class namespace
/// {1}: Storage class c++ name
/// {2}: Parameters parameters
/// {3}: Parameter initialzer string
/// {4}: Parameter name list;
/// {5}: Parameter types
static const char *const typeDefStorageClassBegin = R"(
namespace {0} {{
  struct {1} : public ::mlir::TypeStorage {{
      {1} ({2})
          : {3} {{ }

      /// The hash key for this storage is a pair of the integer and type params.
      using KeyTy = std::tuple<{5}>;

      /// Define the comparison function for the key type.
      bool operator==(const KeyTy &key) const {{
          return key == KeyTy({4});
      }

      static ::llvm::hash_code hashKey(const KeyTy &key) {{
)";

/// The storage class' constructor template
/// {0}: storage class name
static const char *const typeDefStorageClassConstructorBegin = R"(
      /// Define a construction method for creating a new instance of this storage.
      static {0} *construct(::mlir::TypeStorageAllocator &allocator,
                                          const KeyTy &key) {{
)";

/// The storage class' constructor return template
/// {0}: storage class name
/// {1}: list of parameters
static const char *const typeDefStorageClassConstructorReturn = R"(
          return new (allocator.allocate<{0}>())
              {0}({1});
      }
)";

/// use tgfmt to emit custom allocation code for each parameter, if necessary
static mlir::LogicalResult emitCustomAllocationCode(TypeDef &typeDef,
                                                    raw_ostream &os) {
  SmallVector<TypeParameter, 4> parameters;
  typeDef.getParameters(parameters);
  auto fmtCtxt = FmtContext().addSubst("_allocator", "allocator");
  for (auto parameter : parameters) {
    auto allocCode = parameter.getAllocator();
    if (allocCode) {
      fmtCtxt.withSelf(parameter.getName());
      fmtCtxt.addSubst("_dst", parameter.getName());
      auto fmtObj = tgfmt(*allocCode, &fmtCtxt);
      os << "          ";
      fmtObj.format(os);
      os << "\n";
    }
  }
  return mlir::success();
}

static mlir::LogicalResult emitStorageClass(TypeDef typeDef, raw_ostream &os) {
  SmallVector<TypeParameter, 4> parameters;
  typeDef.getParameters(parameters);

  // Initialize a bunch of variables to be used later on
  auto parameterNames = llvm::map_range(
      parameters, [](TypeParameter parameter) { return parameter.getName(); });
  auto parameterTypes =
      llvm::map_range(parameters, [](TypeParameter parameter) {
        return parameter.getCppType();
      });
  auto parameterList = llvm::join(parameterNames, ", ");
  auto parameterTypeList = llvm::join(parameterTypes, ", ");
  auto parameterParameters = constructParameterParameters(typeDef, false);
  auto parameterInits = constructParametersInitializers(typeDef);

  // emit most of the storage class up until the hashKey body
  os << llvm::formatv(typeDefStorageClassBegin, typeDef.getStorageNamespace(),
                      typeDef.getStorageClassName(), parameterParameters,
                      parameterInits, parameterList, parameterTypeList);

  // extract each parameter from the key (auto unboxing is a c++17 feature)
  for (size_t i = 0; i < parameters.size(); i++) {
    os << llvm::formatv("      auto {0} = std::get<{1}>(key);\n",
                        parameters[i].getName(), i);
  }
  // then combine them all. this requires all the parameters types to have a
  // hash_value defined
  os << "        return ::llvm::hash_combine(\n";
  for (auto *parameterIter = parameters.begin();
       parameterIter < parameters.end(); parameterIter++) {
    os << "          " << parameterIter->getName();
    if (parameterIter < parameters.end() - 1) {
      os << ",\n";
    }
  }
  os << ");\n";
  os << "      }\n";

  // if user wants to build the storage constructor themselves, declare it here
  // and then they can write the definition elsewhere
  if (typeDef.hasStorageCustomConstructor())
    os << "  static " << typeDef.getStorageClassName()
       << " *construct(::mlir::TypeStorageAllocator &allocator, const KeyTy "
          "&key);\n";
  else {
    os << llvm::formatv(typeDefStorageClassConstructorBegin,
                        typeDef.getStorageClassName());
    // I want C++17's unboxing!!!
    for (size_t i = 0; i < parameters.size(); i++) {
      os << llvm::formatv("      auto {0} = std::get<{1}>(key);\n",
                          parameters[i].getName(), i);
    }
    // Reassign the parameter variables with allocation code, if it's specified
    if (mlir::failed(emitCustomAllocationCode(typeDef, os)))
      return mlir::failure();
    // return an allocated copy
    os << llvm::formatv(typeDefStorageClassConstructorReturn,
                        typeDef.getStorageClassName(), parameterList);
  }

  // Emit the parameters' class parameters
  for (auto parameter : parameters) {
    os << "      " << parameter.getCppType() << " " << parameter.getName()
       << ";\n";
  }
  os << "  };\n";
  os << "} // namespace " << typeDef.getStorageNamespace() << "\n";

  return mlir::success();
}

/// Emit the body of an autogenerated printer
static mlir::LogicalResult emitPrinterAutogen(TypeDef typeDef,
                                              raw_ostream &os) {
  if (auto mnemonic = typeDef.getMnemonic()) {
    SmallVector<TypeParameter, 4> parameters;
    typeDef.getParameters(parameters);

    os << "  printer << \"" << *mnemonic << "\";\n";

    // if non-parametric, we're done
    if (parameters.size() > 0) {
      os << "  printer << \"<\";\n";

      // emit a printer for each parameter separated by ','.
      // printer structs for common C++ types are defined in
      // TypeDefGenHelpers.h, which must be #included by the consuming code.
      for (auto *parameterIter = parameters.begin();
           parameterIter < parameters.end(); parameterIter++) {
        // Each printer struct must be put on the stack then 'go' called
        os << "  ::mlir::tblgen::parser_helpers::Print<"
           << parameterIter->getCppType() << ">::go(printer, getImpl()->"
           << parameterIter->getName() << ");\n";

        // emit the comma unless we're the last parameter
        if (parameterIter < parameters.end() - 1) {
          os << "  printer << \", \";\n";
        }
      }
      os << "  printer << \">\";\n";
    }
  }
  return mlir::success();
}

/// Emit the body of an autogenerated parser
static mlir::LogicalResult emitParserAutogen(TypeDef typeDef, raw_ostream &os) {
  SmallVector<TypeParameter, 4> parameters;
  typeDef.getParameters(parameters);

  // by the time we get to this function, the mnenomic has already been parsed
  if (parameters.size() > 0) {
    os << "  if (parser.parseLess()) return ::mlir::Type();\n";

    // emit a parser for each parameter separated by ','.
    // parse structs for common C++ types are defined in
    // TypeDefGenHelpers.h, which must be #included by the consuming code.
    for (auto *parameterIter = parameters.begin();
         parameterIter < parameters.end(); parameterIter++) {
      os << "  " << parameterIter->getCppType() << " "
         << parameterIter->getName() << ";\n";
      os << llvm::formatv(
          "  ::mlir::tblgen::parser_helpers::Parse<{0}> {1}Parser;\n",
          parameterIter->getCppType(), parameterIter->getName());
      os << llvm::formatv("  if ({0}Parser.go(ctxt, parser, \"{1}\", {0})) "
                          "return ::mlir::Type();\n",
                          parameterIter->getName(),
                          parameterIter->getCppType());

      // parse a comma unless we're the last parameter
      if (parameterIter < parameters.end() - 1) {
        os << "  if (parser.parseComma()) return ::mlir::Type();\n";
      }
    }
    os << "  if (parser.parseGreater()) return ::mlir::Type();\n";
    // done with the parsing

    // all the parameters are now in variables named the same as the parameters
    auto parameterNames =
        llvm::map_range(parameters, [](TypeParameter parameter) {
          return parameter.getName();
        });
    os << "  return get(ctxt, " << llvm::join(parameterNames, ", ") << ");\n";
  } else {
    os << "  return get(ctxt);\n";
  }
  return mlir::success();
}

/// Print all the typedef-specific definition code
static mlir::LogicalResult emitTypeDefDef(TypeDef typeDef, raw_ostream &os) {
  NamespaceEmitter ns(os, typeDef.getDialect());
  SmallVector<TypeParameter, 4> parameters;
  typeDef.getParameters(parameters);

  // emit the storage class, if requested and necessary
  if (typeDef.genStorageClass() && typeDef.getNumParameters() > 0)
    if (mlir::failed(emitStorageClass(typeDef, os)))
      return mlir::failure();

  std::string paramFuncParams = constructParameterParameters(typeDef, true);
  SmallVector<StringRef, 5> paramNames;
  paramNames.push_back("");
  typeDef.getParametersAs<StringRef>(
      paramNames, [](TypeParameter param) { return param.getName(); });
  os << llvm::formatv("{0} {0}::get(::mlir::MLIRContext* ctxt{1}) {{\n"
                      "  return Base::get(ctxt{2});\n"
                      "}\n",
                      typeDef.getCppClassName(), paramFuncParams,
                      llvm::join(paramNames, ","));
  // emit the accessors
  if (typeDef.genAccessors()) {
    for (auto parameter : parameters) {
      SmallString<16> name = parameter.getName();
      name[0] = llvm::toUpper(name[0]);
      os << llvm::formatv(
          "{0} {3}::get{1}() const { return getImpl()->{2}; }\n",
          parameter.getCppType(), name, parameter.getName(),
          typeDef.getCppClassName());
    }
  }

  // emit the printer code, if appropriate
  auto printerCode = typeDef.getPrinterCode();
  if (printerCode && typeDef.getMnemonic()) {
    // Both the mnenomic and printerCode must be defined (for parity with
    // parserCode)
    os << "void " << typeDef.getCppClassName()
       << "::print(mlir::DialectAsmPrinter& printer) const {\n";
    if (*printerCode == "") {
      // if no code specified, autogenerate a parser
      if (mlir::failed(emitPrinterAutogen(typeDef, os)))
        return mlir::failure();
    } else {
      os << *printerCode << "\n";
    }
    os << "}\n";
  }

  // emit a parser, if appropriate
  auto parserCode = typeDef.getParserCode();
  if (parserCode && typeDef.getMnemonic()) {
    // The mnenomic must be defined so the dispatcher knows how to dispatch
    os << "::mlir::Type " << typeDef.getCppClassName()
       << "::parse(::mlir::MLIRContext* ctxt, ::mlir::DialectAsmParser& "
          "parser) "
          "{\n";
    if (*parserCode == "") {
      if (mlir::failed(emitParserAutogen(typeDef, os)))
        return mlir::failure();
    } else
      os << *parserCode << "\n";
    os << "}\n";
  }

  return mlir::success();
}

/// Emit the dialect printer/parser dispatch. Client code should call these
/// functions from their dialect's print/parse methods.
static mlir::LogicalResult
emitParsePrintDispatch(SmallVectorImpl<TypeDef> &typeDefs, raw_ostream &os) {
  if (typeDefs.size() == 0)
    return mlir::success();
  const Dialect &dialect = typeDefs.begin()->getDialect();
  NamespaceEmitter ns(os, dialect);
  os << "::mlir::Type generatedTypeParser(::mlir::MLIRContext* ctxt, "
        "::mlir::DialectAsmParser& parser, ::llvm::StringRef mnemonic) {\n";
  for (auto typeDef : typeDefs) {
    if (typeDef.getMnemonic())
      os << llvm::formatv("  if (mnemonic == {0}::{1}::getMnemonic()) return "
                          "{0}::{1}::parse(ctxt, parser);\n",
                          typeDef.getDialect().getCppNamespace(),
                          typeDef.getCppClassName());
  }
  os << "  return ::mlir::Type();\n";
  os << "}\n\n";

  os << "bool generatedTypePrinter(::mlir::Type type, "
        "::mlir::DialectAsmPrinter& "
        "printer) {\n"
     << "  bool notfound = false;\n"
     << "  ::llvm::TypeSwitch<::mlir::Type>(type)\n";
  for (auto typeDef : typeDefs) {
    if (typeDef.getMnemonic())
      os << llvm::formatv("    .Case<{0}::{1}>([&](::mlir::Type t) {{ "
                          "t.dyn_cast<{0}::{1}>().print(printer); })\n",
                          typeDef.getDialect().getCppNamespace(),
                          typeDef.getCppClassName());
  }
  os << "    .Default([&notfound](::mlir::Type) { notfound = true; });\n"
     << "  return notfound;\n"
     << "}\n\n";
  return mlir::success();
}

/// Entry point for typedef definitions
static bool emitTypeDefDefs(const llvm::RecordKeeper &recordKeeper,
                            raw_ostream &os) {
  emitSourceFileHeader("TypeDef Definitions", os);

  SmallVector<TypeDef, 16> typeDefs;
  if (mlir::failed(findAllTypeDefs(recordKeeper, typeDefs)))
    return true;

  if (mlir::failed(emitTypeDefList(typeDefs, os)))
    return true;

  IfDefScope scope("GET_TYPEDEF_CLASSES", os);
  if (mlir::failed(emitParsePrintDispatch(typeDefs, os)))
    return true;
  for (auto typeDef : typeDefs) {
    if (mlir::failed(emitTypeDefDef(typeDef, os)))
      return true;
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
