//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// parse / print works some C++ template magic to map to the correct type
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_PARSER_HELPERS_H
#define MLIR_TABLEGEN_PARSER_HELPERS_H

#include <type_traits>
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"

namespace llvm {
    hash_code hash_value(float f) {
        return *(uint32_t*)&f;
    }
}

namespace mlir {
namespace tblgen {
namespace parser_helpers {

  template <typename...>
  using void_t = void;

  template<typename T>
  using remove_constref = typename std::remove_const< typename std::remove_reference<T>::type >::type;

  template <typename T, typename TestType>
  using enable_if_type = typename std::enable_if<std::is_same<remove_constref<T>, TestType>::value>::type;

  template <typename T, typename TestType>
  using is_not_type = std::is_same<
                        typename std::is_same<remove_constref<T>, TestType>::type,
                        typename std::false_type::type>;

  template<typename T>
  using get_indexable_type = remove_constref< decltype(std::declval<T>()[0]) >;

  template<typename T>
  using enable_if_arrayref = enable_if_type<T, typename mlir::ArrayRef<get_indexable_type<T>> >;

  //////////
  /// Parse helpers
  template<typename T, typename Enable = void>
  struct parse {
    static ParseResult go(MLIRContext* ctxt, DialectAsmParser& parser, llvm::BumpPtrAllocator& alloc, StringRef typeStr, T& result);
  };

  // Int specialization
  template <typename T>
  using enable_if_integral_type = typename std::enable_if<
                                    std::is_integral<T>::value &&
                                    is_not_type<T, bool>::value >::type;
  template<typename T>
  struct parse<T, enable_if_integral_type<T>> {
    static ParseResult go(MLIRContext* ctxt, DialectAsmParser& parser, llvm::BumpPtrAllocator& alloc, StringRef typeStr, T& result) {
      return parser.parseInteger(result);
    }
  };

  template<typename T>
  struct parse<T, enable_if_type<T, bool>> {
    static ParseResult go(MLIRContext* ctxt, DialectAsmParser& parser, llvm::BumpPtrAllocator& alloc, StringRef typeStr, bool& result) {
      StringRef boolStr;
      if (parser.parseKeyword(&boolStr)) return mlir::failure();
      if (!boolStr.compare_lower("false")) { result = false; return mlir::success(); }
      if (!boolStr.compare_lower("true")) { result = true; return mlir::success(); }
      llvm::errs() << "Parser expected true/false, not '" << boolStr << "'\n";
      return mlir::failure();
    }
  };

  // Float specialization
  template <typename T>
  using enable_if_float_type = typename std::enable_if<std::is_floating_point<T>::value>::type;
  template<typename T>
  struct parse<T, enable_if_float_type<T>> {
    static ParseResult go(MLIRContext* ctxt, DialectAsmParser& parser, llvm::BumpPtrAllocator& alloc, StringRef typeStr, T& result) {
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
    static ParseResult go(MLIRContext* ctxt, DialectAsmParser& parser, llvm::BumpPtrAllocator& alloc, StringRef typeStr, T& result) {
      Type type;
      auto loc = parser.getCurrentLocation();
      if (parser.parseType(type)) return mlir::failure();
      if ((result = type.dyn_cast_or_null<T>()) == nullptr) {
        parser.emitError(loc, "expected type '" + typeStr + "'");
        return mlir::failure();
      }
      return mlir::success();
    }
  };

  template<typename T>
  struct parse<T, enable_if_type<T, StringRef>> {
    static ParseResult go(MLIRContext* ctxt, DialectAsmParser& parser, llvm::BumpPtrAllocator& alloc, StringRef typeStr, StringRef& result) {
      StringAttr a;
      if (parser.parseAttribute<StringAttr>(a)) return mlir::failure();
      result = a.getValue();
      return mlir::success();
    }
  };

  template<typename T>
  struct parse<T, enable_if_arrayref<T>> {
    using inner_t = get_indexable_type<T>;

    static ParseResult go(MLIRContext* ctxt, DialectAsmParser& parser, llvm::BumpPtrAllocator& alloc, StringRef typeStr, ArrayRef<inner_t>& result) {
      std::vector<inner_t>* members = new std::vector<inner_t>();
      if (parser.parseLSquare()) return mlir::failure();
        if (failed(parser.parseOptionalRSquare())) {
        do {
          inner_t member;// = std::declval<inner_t>();
          parse<inner_t>::go(ctxt, parser, alloc, typeStr, member);
          members->push_back(member);
        } while (succeeded(parser.parseOptionalComma()));
        if (parser.parseRSquare()) return mlir::failure();
      }
      result = ArrayRef<inner_t>(*members);
      return mlir::success();
    }
  };

  //////////
  /// Print helpers
  template<typename T, typename Enable = void>
  struct print {
    static void go(DialectAsmPrinter& printer, const T& obj);
  };

  template <typename T>
  using enable_if_trivial = typename std::enable_if<
                          std::is_convertible<T, mlir::Type>::value ||
                          ( std::is_integral<T>::value && is_not_type<T, bool>::value ) ||
                          std::is_floating_point<T>::value>::type;
  template<typename T>
  struct print<T, enable_if_trivial< remove_constref<T> >> {
    static void go(DialectAsmPrinter& printer, const T& obj) {
      printer << obj;
    }
  };

  template<typename T>
  struct print<T, enable_if_type<T, StringRef> > {
    static void go(DialectAsmPrinter& printer, const T& obj) {
      printer << "\"" << obj << "\"";
    }
  };

  template<typename T>
  struct print<T, enable_if_type<T, bool>> {
    static void go(DialectAsmPrinter& printer, const bool& obj) {
      if (obj) printer << "true";
      else     printer << "false";
    }
  };

  template<typename T>
  struct print<T, enable_if_arrayref<T>> {
    static void go(DialectAsmPrinter& printer, const ArrayRef<get_indexable_type<T>>& obj) {
      printer << "[";
      for (size_t i=0; i < obj.size(); i++) {
        print<remove_constref<decltype(obj[i])>>::go(printer, obj[i]);
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
