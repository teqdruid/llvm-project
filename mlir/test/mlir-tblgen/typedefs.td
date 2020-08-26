// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"
include "mlir/Dialect/StandardOps/IR/Ops.td"

// ************
// Test defs
def Test_Dialect: Dialect {
    let name = "TestDialect";
}

// Base class for other typedefs. Provides dialact-specific defaults
class TestType<string name> : TypeDef<Test_Dialect, name> { }

// SimpleTypeA is a simple type. A storage class will not be generated.
def SimpleTypeA : TestType<"SimpleA"> { }

// A more complex parameterized type
def CompoundTypeA : TestType<"CompoundA"> {
    // Override the default mnemonic
    let mnemonic = "cmpnd_a";

    // What types do we contain?
    let members = (
        ins
        // A standard c++ int
        "int":$widthOfSomething,
        // The simple type defined above
        "SimpleTypeA": $exampleTdType,
        // Some C++ type
        "SomeCppStruct": $exampleCppType
    );

    let genVerifyInvariantsDecl = 1;
}

// Base class for standard types
class StdType<string name> : TypeDef<StandardOps_Dialect, name> { }

def IndexType : StdType<"Index"> {
    // let mnemonic = "index";
}

def SingleMemberType : StdType<"SingleMember"> {
  let members = (
    ins
    "int": $num
  );
}

def IntegerType : StdType<"Integer"> {
    let parser = [{}];
    let printer = [{}];
    let mnemonic = "int";
    let genVerifyInvariantsDecl = 1;
    let members = (
        ins
        "SignednessSemantics":$signedness, 
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

def TupleType : StdType<"Tuple"> {
    let mnemonic = "tuple";
    let members = (
        ins
        "int":$widthOfSomething,
        ArrayRefMember<"Type">:$types,
        SelfAllocationMember<"ComplexCppClass">:$bizarreField
    );
    let printer = ?;

    let extraDecls = [{
  /// Accumulate the types contained in this tuple and tuples nested within it.
  /// Note that this only flattens nested tuples, not any other container type,
  /// e.g. a tuple<i32, tensor<i32>, tuple<f32, tuple<i64>>> is flattened to
  /// (i32, tensor<i32>, f32, i64)
  void getFlattenedTypes(SmallVectorImpl<Type> &types);

  /// Return the number of held types.
  size_t size() const;

  /// Iterate over the held elements.
  using iterator = ArrayRef<Type>::iterator;
  iterator begin() const { return getTypes().begin(); }
  iterator end() const { return getTypes().end(); }

  /// Return the element type at index 'index'.
  Type getType(size_t index) const {
    assert(index < size() && "invalid index for tuple type");
    return getTypes()[index];
  }
    }];
}
