// RUN: mlir-tblgen -gen-typedef-decls -I %S/../../include %s | FileCheck %s --check-prefix=DECL
// RUN: mlir-tblgen -gen-typedef-defs -I %S/../../include %s | FileCheck %s --check-prefix=DEF

include "mlir/IR/OpBase.td"

// DECL: #ifdef GET_TYPEDEF_CLASSES
// DECL: #undef GET_TYPEDEF_CLASSES

// DECL: Type generatedTypeParser(mlir::MLIRContext* ctxt, mlir::DialectAsmParser& parser, llvm::StringRef mnenomic);
// DECL: bool generatedTypePrinter(Type type, mlir::DialectAsmPrinter& printer);

// DEF: #ifdef GET_TYPEDEF_LIST
// DEF: #undef GET_TYPEDEF_LIST
// DEF: SimpleAType,
// DEF: CompoundAType,
// DEF: IndexType,
// DEF: SingleMemberType,
// DEF: IntegerType

// DEF-LABEL: Type generatedTypeParser(mlir::MLIRContext* ctxt, mlir::DialectAsmParser& parser, llvm::StringRef mnemonic)
// DEF: if (mnemonic == CompoundAType::getMnemonic()) return CompoundAType::parse(ctxt, parser);
// DEF return Type();

def Test_Dialect: Dialect {
// DECL-NOT: TestDialect
// DEF-NOT: TestDialect
    let name = "TestDialect";
}

class TestType<string name> : TypeDef<Test_Dialect, name> { }

def A_SimpleTypeA : TestType<"SimpleA"> {
// DECL: class SimpleAType: public Type
// DECL: static Type parse(mlir::MLIRContext* ctxt, mlir::DialectAsmParser& parser);
// DECL: void print(mlir::DialectAsmPrinter& printer) const;
}

// A more complex parameterized type
def B_CompoundTypeA : TestType<"CompoundA"> {
  let summary = "A more complex parameterized type";
  let description = "This type is to test a reasonably complex type";
  let mnemonic = "cmpnd_a";
  let members = (
      ins
      "int":$widthOfSomething,
      "SimpleTypeA": $exampleTdType,
      "SomeCppStruct": $exampleCppType,
      ArrayRefMember<"int", "Matrix dimensions">:$dims
  );

  let genVerifyInvariantsDecl = 1;

// DECL-LABEL: class CompoundAType: public Type
// DECL: static LogicalResult verifyConstructionInvariants(Location loc, int widthOfSomething, SimpleTypeA exampleTdType, SomeCppStruct exampleCppType);
// DECL: static CompoundAType getChecked(Location loc, int widthOfSomething, SimpleTypeA exampleTdType, SomeCppStruct exampleCppType);
// DECL: static StringRef getMnemonic() { return "cmpnd_a"; }
// DECL: int getWidthOfSomething() const;
// DECL: SimpleTypeA getExampleTdType() const;
// DECL: SomeCppStruct getExampleCppType() const;
}

def C_IndexType : TestType<"Index"> {
    let mnemonic = "index";

    let members = (
      ins
      StringRefMember<"Label for index">:$label
    );

// DECL-LABEL: class IndexType: public Type
// DECL: static Type parse(mlir::MLIRContext* ctxt, mlir::DialectAsmParser& parser);
// DECL: void print(mlir::DialectAsmPrinter& printer) const;
// DECL: static StringRef getMnemonic() { return "index"; }
}

def D_SingleMemberType : TestType<"SingleMember"> {
  let members = (
    ins
    "int": $num
  );
// DECL-LABEL: struct SingleMemberTypeStorage;
// DECL-LABEL: class SingleMemberType
// DECL-NEXT:                   detail::SingleMemberTypeStorage
// DECL: static Type parse(mlir::MLIRContext* ctxt, mlir::DialectAsmParser& parser);
// DECL: void print(mlir::DialectAsmPrinter& printer) const;
}

def E_IntegerType : TestType<"Integer"> {
    let parser = [{}];
    let printer = [{}];
    let mnemonic = "int";
    let genVerifyInvariantsDecl = 1;
    let members = (
        ins
        "SignednessSemantics":$signedness, 
        TypeMember<"unsigned", "Bitwdith of integer">:$width
    );

// DECL-LABEL: IntegerType: public Type

    let extraDecls = [{
  /// Signedness semantics.
  enum SignednessSemantics {
    Signless, /// No signedness semantics
    Signed,   /// Signed integer
    Unsigned, /// Unsigned integer
  };

  /// This extra function is necessary since it doesn't include signedness
  static IntegerType getChecked(unsigned width, Location location);

  /// Return true if this is a signless integer type.
  bool isSignless() const { return getSignedness() == Signless; }
  /// Return true if this is a signed integer type.
  bool isSigned() const { return getSignedness() == Signed; }
  /// Return true if this is an unsigned integer type.
  bool isUnsigned() const { return getSignedness() == Unsigned; }
    }];

// DECL: /// Signedness semantics.
// DECL-NEXT: enum SignednessSemantics {
// DECL-NEXT:   Signless, /// No signedness semantics
// DECL-NEXT:   Signed,   /// Signed integer
// DECL-NEXT:   Unsigned, /// Unsigned integer
// DECL-NEXT: };
// DECL: /// This extra function is necessary since it doesn't include signedness
// DECL-NEXT: static IntegerType getChecked(unsigned width, Location location);

// DECL: /// Return true if this is a signless integer type.
// DECL-NEXT: bool isSignless() const { return getSignedness() == Signless; }
// DECL-NEXT: /// Return true if this is a signed integer type.
// DECL-NEXT: bool isSigned() const { return getSignedness() == Signed; }
// DECL-NEXT: /// Return true if this is an unsigned integer type.
// DECL-NEXT: bool isUnsigned() const { return getSignedness() == Unsigned; }
}
