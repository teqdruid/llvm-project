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
        // A standard c++ int
        "int":$widthOfSomething,
        // The simple type defined above
        "SimpleAType": $exampleTdType,
        // Some C++ type
        // "CompoundAType::SomeCppStruct": $exampleCppType
        "float": $f,
        "double": $d,
        ArrayRefMember<"int">: $arrayOfInts,
        ArrayRefMember<"Type">: $arrayOfTypes,
        "StringRef": $simpleString,
        ArrayRefMember<"StringRef">: $arrayOfStrings
    );

    let extraDecls = [{
        struct SomeCppStruct {};
    }];
}

#endif // TEST_TYPEDEFS
