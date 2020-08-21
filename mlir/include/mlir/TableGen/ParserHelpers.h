//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TypeDef wrapper to simplify using TableGen Record defining a MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_PARSER_HELPERS_H
#define MLIR_TABLEGEN_PARSER_HELPERS_H

#include "mlir/IR/DialectImplementation.h"
#include <type_traits>

namespace mlir {
namespace tblgen {
namespace parser_helpers {

  // parse works some C++ template magic to map to the correct type
  template<typename T, typename Enable = void>
  struct parse {
    static ParseResult go(MLIRContext* ctxt, DialectAsmParser& parser, T& result);
  };

  // Int specialization
  template <typename T>
  using enable_if_integral_type = typename std::enable_if<std::is_integral<T>::value>::type;
  template<typename T>
  struct parse<T, enable_if_integral_type<T>> {
    static ParseResult go(MLIRContext* ctxt, DialectAsmParser& parser, T& result) {
      return parser.parseInteger(result);
    }
  };

  // Float specialization
  template <typename T>
  using enable_if_float_type = typename std::enable_if<std::is_floating_point<T>::value>::type;
  template<typename T>
  struct parse<T, enable_if_float_type<T>> {
    static ParseResult go(MLIRContext* ctxt, DialectAsmParser& parser, T& result) {
      double d;
      if (parser.parseFloat(d))
        return mlir::failure();
      result = d;
      return mlir::success();
    }
  };

  // mlir::Type specialization
  template <typename T>
  using enable_if_mlir_type = typename std::enable_if<std::is_convertible<T, mlir::Type>::value>::type;
  template<typename T>
  struct parse<T, enable_if_mlir_type<T>> {
    static ParseResult go(MLIRContext* ctxt, DialectAsmParser& parser, Type& result) {
      return parser.parseType(result);
    }
  };

} // end namespace parser_helpers
} // end namespace tblgen
} // end namespace mlir

#endif // MLIR_TABLEGEN_PARSER_HELPERS_H
