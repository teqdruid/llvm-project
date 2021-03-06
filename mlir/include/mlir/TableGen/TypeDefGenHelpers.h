//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Accessory functions / templates to assist autogenerated code. The print/parse
// struct templates define standard serializations which can be overridden with
// custom printers/parsers. These structs can be used for temporary stack
// storage also.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_PARSER_HELPERS_H
#define MLIR_TABLEGEN_PARSER_HELPERS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include <type_traits>

namespace mlir {
namespace tblgen {
namespace parser_helpers {

//===----------------------------------------------------------------------===//
//
// Template enables identify various types for which we have specializations
//
//===----------------------------------------------------------------------===//

template <typename...>
using void_t = void;

template <typename T>
using remove_constref =
    typename std::remove_const<typename std::remove_reference<T>::type>::type;

template <typename T, typename TestType>
using enable_if_type = typename std::enable_if<
    std::is_same<remove_constref<T>, TestType>::value>::type;

template <typename T, typename TestType>
using is_not_type =
    std::is_same<typename std::is_same<remove_constref<T>, TestType>::type,
                 typename std::false_type::type>;

template <typename T>
using get_indexable_type = remove_constref<decltype(std::declval<T>()[0])>;

template <typename T>
using enable_if_arrayref =
    enable_if_type<T, typename mlir::ArrayRef<get_indexable_type<T>>>;

//===----------------------------------------------------------------------===//
//
// These structs handle Type members' parsing for common types
//
//===----------------------------------------------------------------------===//

template <typename T, typename Enable = void>
struct Parse {
  ParseResult go(MLIRContext *ctxt,        // The context, should it be needed
                 DialectAsmParser &parser, // The parser
                 StringRef memberName, // Type member name, for error printing
                                       // (if necessary)
                 T &result);           // Put the parsed value here
};

// Int specialization
template <typename T>
using enable_if_integral_type =
    typename std::enable_if<std::is_integral<T>::value &&
                            is_not_type<T, bool>::value>::type;
template <typename T>
struct Parse<T, enable_if_integral_type<T>> {
  ParseResult go(MLIRContext *ctxt, DialectAsmParser &parser,
                 StringRef memberName, T &result) {
    return parser.parseInteger(result);
  }
};

// Bool specialization -- 'true' / 'false' instead of 0/1
template <typename T>
struct Parse<T, enable_if_type<T, bool>> {
  ParseResult go(MLIRContext *ctxt, DialectAsmParser &parser,
                 StringRef memberName, bool &result) {
    StringRef boolStr;
    if (parser.parseKeyword(&boolStr))
      return mlir::failure();
    if (!boolStr.compare_lower("false")) {
      result = false;
      return mlir::success();
    }
    if (!boolStr.compare_lower("true")) {
      result = true;
      return mlir::success();
    }
    llvm::errs() << "Parser expected true/false, not '" << boolStr << "'\n";
    return mlir::failure();
  }
};

// Float specialization
template <typename T>
using enable_if_float_type =
    typename std::enable_if<std::is_floating_point<T>::value>::type;
template <typename T>
struct Parse<T, enable_if_float_type<T>> {
  ParseResult go(MLIRContext *ctxt, DialectAsmParser &parser,
                 StringRef memberName, T &result) {
    double d;
    if (parser.parseFloat(d))
      return mlir::failure();
    result = d;
    return mlir::success();
  }
};

// mlir::Type specialization
template <typename T>
using enable_if_mlir_type =
    typename std::enable_if<std::is_convertible<T, mlir::Type>::value>::type;
template <typename T>
struct Parse<T, enable_if_mlir_type<T>> {
  ParseResult go(MLIRContext *ctxt, DialectAsmParser &parser,
                 StringRef memberName, T &result) {
    Type type;
    auto loc = parser.getCurrentLocation();
    if (parser.parseType(type))
      return mlir::failure();
    if ((result = type.dyn_cast_or_null<T>()) == nullptr) {
      parser.emitError(loc, "expected type '" + memberName + "'");
      return mlir::failure();
    }
    return mlir::success();
  }
};

// StringRef specialization
template <typename T>
struct Parse<T, enable_if_type<T, StringRef>> {
  ParseResult go(MLIRContext *ctxt, DialectAsmParser &parser,
                 StringRef memberName, StringRef &result) {
    StringAttr a;
    if (parser.parseAttribute<StringAttr>(a))
      return mlir::failure();
    result = a.getValue();
    return mlir::success();
  }
};

// ArrayRef specialization
template <typename T>
struct Parse<T, enable_if_arrayref<T>> {
  using inner_t = get_indexable_type<T>;
  Parse<inner_t> innerParser;
  llvm::SmallVector<inner_t, 4> members;

  ParseResult go(MLIRContext *ctxt, DialectAsmParser &parser,
                 StringRef memberName, ArrayRef<inner_t> &result) {
    if (parser.parseLSquare())
      return mlir::failure();
    if (failed(parser.parseOptionalRSquare())) {
      do {
        inner_t member; // = std::declval<inner_t>();
        innerParser.go(ctxt, parser, memberName, member);
        members.push_back(member);
      } while (succeeded(parser.parseOptionalComma()));
      if (parser.parseRSquare())
        return mlir::failure();
    }
    result = ArrayRef<inner_t>(members);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
//
// These structs handle Type members' printing for common types
//
//===----------------------------------------------------------------------===//

template <typename T, typename Enable = void>
struct Print {
  static void go(DialectAsmPrinter &printer, const T &obj);
};

// Several C++ types can just be piped into the printer
template <typename T>
using enable_if_trivial_print =
    typename std::enable_if<std::is_convertible<T, mlir::Type>::value ||
                            (std::is_integral<T>::value &&
                             is_not_type<T, bool>::value) ||
                            std::is_floating_point<T>::value>::type;
template <typename T>
struct Print<T, enable_if_trivial_print<remove_constref<T>>> {
  static void go(DialectAsmPrinter &printer, const T &obj) { printer << obj; }
};

// StringRef has to be quoted to match the parse specialization above
template <typename T>
struct Print<T, enable_if_type<T, StringRef>> {
  static void go(DialectAsmPrinter &printer, const T &obj) {
    printer << "\"" << obj << "\"";
  }
};

// bool specialization
template <typename T>
struct Print<T, enable_if_type<T, bool>> {
  static void go(DialectAsmPrinter &printer, const bool &obj) {
    if (obj)
      printer << "true";
    else
      printer << "false";
  }
};

// ArrayRef specialization
template <typename T>
struct Print<T, enable_if_arrayref<T>> {
  static void go(DialectAsmPrinter &printer,
                 const ArrayRef<get_indexable_type<T>> &obj) {
    printer << "[";
    for (size_t i = 0; i < obj.size(); i++) {
      Print<remove_constref<decltype(obj[i])>>::go(printer, obj[i]);
      if (i < obj.size() - 1)
        printer << ", ";
    }
    printer << "]";
  }
};

} // end namespace parser_helpers
} // end namespace tblgen
} // end namespace mlir

#endif // MLIR_TABLEGEN_PARSER_HELPERS_H
