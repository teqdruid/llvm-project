//===-- TestTypeDefs.td - Test dialect type definitions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_TYPEDEFS
#define TEST_TYPEDEFS

// To get the test dialect def
include "TestOps.td"

class Test_Type<string name> : TypeDef<Test_Dialect, name> { }

def SimpleTypeA : Test_Type<"SimpleA"> {
    let mnemonic = "smpla";
}

// A more complex parameterized type
def CompoundTypeA : Test_Type<"CompoundA"> {
    // Override the default mnemonic
    let mnemonic = "cmpnd_a";

    // What types do we contain?
    let members = (
        ins
        "int":$widthOfSomething,
        "SimpleAType": $exampleTdType,
        "float": $f,
        "double": $d,
        ArrayRefMember<"int", "">: $arrayOfInts,
        ArrayRefMember<"Type", "An example of an array of types as a type parameter">: $arrayOfTypes,
        "StringRef": $simpleString,
        ArrayRefMember<"StringRef", "">: $arrayOfStrings
    );

    let extraDecls = [{
        struct SomeCppStruct {};
    }];
}

def IntegerType : Test_Type<"TestInteger"> {
    let mnemonic = "int";
    let genVerifyInvariantsDecl = 1;
    let members = (
        ins
        "TestIntegerType::SignednessSemantics":$signedness,
        "unsigned":$width
    );

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
}

class FieldInfo_Type<string name> : Test_Type<name> {
let members = (
    ins
    ArrayRefOfSelfAllocationMember<"FieldInfo", "Models struct fields">: $fields
);

let printer = [{
    printer << "struct" << "<";
    for (size_t i=0; i<getImpl()->fields.size(); i++) {
    const auto& field = getImpl()->fields[i];
    printer << "{" << field.name << "," << field.type << "}";
    if (i < getImpl()->fields.size() - 1)
        printer << ",";
    }
    printer << ">";
}];

let parser = [{
    SmallVector<FieldInfo, 4> members;
    if (parser.parseLess()) return Type();
    while (mlir::succeeded(parser.parseOptionalLBrace())) {
        StringRef name;
        if (parser.parseKeyword(&name)) return Type();
        if (parser.parseComma()) return Type();
        Type type;
        if (parser.parseType(type)) return Type();
        if (parser.parseRBrace()) return Type();
        members.push_back(FieldInfo {name, type});
        if (parser.parseOptionalComma()) break;
    }
    if (parser.parseGreater()) return Type();
    return get(ctxt, members);
}];
}

def StructType : FieldInfo_Type<"Struct"> {
    let mnemonic = "struct";
}


#endif // TEST_TYPEDEFS
