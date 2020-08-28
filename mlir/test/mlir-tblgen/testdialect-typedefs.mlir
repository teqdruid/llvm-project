// RUN: mlir-opt %s | mlir-opt -verify-diagnostics | FileCheck %s

//////////////
// Tests the types in the 'Test' dialect, not the ones in 'typedefs.mlir'

// CHECK: @simpleA(%arg0: !test.smpla)
func @simpleA(%A : !test.smpla) -> () {
  return
}

// CHECK: @compoundA(%arg0: !test.cmpnd_a<1, !test.smpla, 1.000000e+00, 2.200000e+00, [5, 6], [i1, i2], "example str", ["array", "of", "strings"]>)
func @compoundA(%A : !test.cmpnd_a<1, !test.smpla, 1.0, 2.2, [5, 6], [i1, i2], "example str", ["array","of","strings"]>) -> () {
  return
}

// CHECK: @testInt(%arg0: !test.int<unsigned, 8>, %arg1: !test.int<unsigned, 2>, %arg2: !test.int<unsigned, 1>)
func @testInt(%A : !test.int<s, 8>, %B : !test.int<unsigned, 2>, %C : !test.int<n, 1>) {
  return
}

// CHECK: @structTest(%arg0: !test.struct<{field1,!test.smpla},{field2,!test.int<unsigned, 3>}>)
func @structTest (%A : !test.struct< {field1, !test.smpla}, {field2, !test.int<none, 3>} > ) {
  return 
}
