/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef HighOrderOperatorsQuadInternal_h
#define HighOrderOperatorsQuadInternal_h

#include <Teuchos_BLAS.hpp>
#include <element_promotion/CVFEMTypeDefs.h>

#include <Tpetra_Details_gemm.hpp>

namespace sierra {
namespace nalu {
namespace internal {

inline Teuchos::ETransp char_to_teuchos_enum(char x) { return (x == 'N') ? Teuchos::NO_TRANS : Teuchos::TRANS; }

template <int poly_order, typename ViewTypeA, typename ViewTypeB, typename ViewTypeC>
void gemm_nxn(
  char transA, char transB,
  double alpha,
  const ViewTypeA& A,
  const ViewTypeB& B,
  double beta,
  ViewTypeC& C)
{
    int n = A.dimension_0();
    Teuchos::BLAS<int, double>().GEMM(
      char_to_teuchos_enum(transA), char_to_teuchos_enum(transB),
      n, n, n,
      alpha, (double*)A.ptr_on_device(), n, (double*)B.ptr_on_device(), n,
      beta, (double*)C.ptr_on_device(), n);






//  // Kokkos-Kernels gemm currently has a message, "do not use"
//  static_assert(std::is_same<typename ViewTypeA::value_type, typename ViewTypeB::value_type>::value,
//    "Types don't match");
//
//  static_assert(std::is_same<typename ViewTypeA::value_type, typename ViewTypeC::value_type>::value,
//    "Types don't match");
//
//  constexpr int n = poly_order+1;
//
//
//
//
//  using c_value_type = typename ViewTypeC::value_type;
//
////  // Start the operations
//  if (transB == 'N') {
//   if (transA == 'N') {
//      // Form C = alpha*A*B + beta*C
//      for (int j = 0; j < n; ++j) {
//        if (beta == 0.0) {
//          for (int i = 0; i < n; ++i) {
//            C(i,j) = 0.0;
//          }
//        }
//        else if (beta != 1.0) {
//          for (int i = 0; i < n; ++i) {
//            C(i,j) = beta*C(i,j);
//          }
//        }
//        for (int l = 0; l < n; ++l) {
//          // Don't use c_value_type here, since it unnecessarily
//          // forces type conversion before we assign to C(i,j).
//          auto temp = alpha*B(l,j);
//          for (int i = 0; i < n; ++i) {
//            C(i,j) = C(i,j) + temp*A(i,l);
//          }
//        }
//      }
//    }
//    else {
//      // Form C = alpha*A**T*B + beta*C
//      for (int j = 0; j < n; ++j) {
//        for (int i = 0; i < n; ++i) {
//          c_value_type temp = 0.0;
//          for (int l = 0; l < n; ++l) {
//            temp = temp + A(l,i)*B(l,j);
//          }
//          if (beta == 0.0) {
//            C(i,j) = alpha*temp;
//          }
//          else {
//            C(i,j) = alpha*temp + beta*C(i,j);
//          }
//        }
//      }
//    }
//  }
//  else {
//    if (transA == 'N') {
//      // Form C = alpha*A*B**T + beta*C
//      for (int j = 0; j < n; ++j) {
//        if (beta == 0.0) {
//          for (int i = 0; i < n; ++i) {
//            C(i,j) = 0.0;
//          }
//        }
//        else if (beta != 1.0) {
//          for (int i = 0; i < n; ++i) {
//            C(i,j) = beta*C(i,j);
//          }
//        }
//        for (int l = 0; l < n; ++l) {
//          // Don't use c_value_type here, since it unnecessarily
//          // forces type conversion before we assign to C(i,j).
//          auto temp = alpha*B(j,l);
//          for (int i = 0; i < n; ++i) {
//            C(i,j) = C(i,j) + temp*A(i,l);
//          }
//        }
//      }
//    }
//    else {
//      // Form C = alpha*A**T*B**T + beta*C
//      for (int j = 0; j < n; ++j) {
//        for (int i = 0; i < n; ++i) {
//          c_value_type temp = 0.0;
//          for (int l = 0; l < n; ++l) {
//            temp = temp + A(l,i)*B(j,l);
//          }
//          if (beta == 0.0) {
//            C(i,j) = alpha*temp;
//          }
//          else {
//            C(i,j) = alpha*temp + beta*C(i,j);
//          }
//        }
//      }
//    }
//  }
}
//--------------------------------------------------------------------------
template <int poly_order, typename MatViewType, typename ViewTypeIn, typename ViewTypeOut>
void apply_x(const MatViewType coeffMatrix, const ViewTypeIn in, ViewTypeOut out)
{
  constexpr int n = poly_order+1;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      out(j,i) = 0.0;
    }

    for (int l = 0; l < n; ++l) {
      auto temp = in(j,l);
      for (int i = 0; i < n; ++i) {
        out(j,i) += temp*coeffMatrix(i,l);
      }
    }
  }
//  gemm_nxn<poly_order>('T', 'N', 1.0, coeffMatrix, in, 0.0, out);
}
//--------------------------------------------------------------------------
template <int poly_order, typename MatViewType, typename ViewTypeIn, typename ViewTypeOut>
void apply_y(MatViewType coeffMatrix, ViewTypeIn in, ViewTypeOut out)
{
  constexpr int n = poly_order+1;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      out(j,i) = 0.0;
    }

    for (int l = 0; l < n; ++l) {
      auto temp = coeffMatrix(j,l);
      for (int i = 0; i < n; ++i) {
        out(j,i) += temp*in(l,i);
      }
    }
  }

//  gemm_nxn<poly_order>('N', 'N', 1.0,  in, coeffMatrix, 0.0, out);
}
//--------------------------------------------------------------------------
template <int poly_order, typename MatViewType, typename ViewTypeIn, typename ViewTypeOut>
void apply_yx(MatViewType coeffMatrix1, MatViewType coeffMatrix2, ViewTypeIn in,  ViewTypeOut out)
{
  nodal_matrix_array<poly_order, typename MatViewType::value_type> temp_array;

  constexpr int n = poly_order+1;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      temp_array[j][i] = 0.0;
    }

    for (int l = 0; l < n; ++l) {
      auto temp = coeffMatrix1(j,l);
      for (int i = 0; i < n; ++i) {
        temp_array[j][i] += temp*in(l,i);
      }
    }
  }


  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      out(j,i) = 0.0;
    }

    for (int l = 0; l < n; ++l) {
      auto temp = temp_array[j][l];
      for (int i = 0; i < n; ++i) {
        out(j,i) += temp*coeffMatrix2(i,l);
      }
    }
  }
//  MatViewType temp("");
//  gemm_nxn<poly_order>('N', 'N', 1.0,  in, coeffMatrix1, 0.0, temp);
//  gemm_nxn<poly_order>('T', 'N', 1.0, coeffMatrix2, temp, 1.0, out);
}
//--------------------------------------------------------------------------
template <int poly_order, typename MatViewType>
void apply_xy(const MatViewType coeffMatrix1, const MatViewType coeffMatrix2, const MatViewType in,  MatViewType out)
{
  nodal_matrix_array<poly_order, typename MatViewType::value_type> temp_array;
  constexpr int n = poly_order+1;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      temp_array[j][i] = 0.0;
    }

    for (int l = 0; l < n; ++l) {
      auto temp = in(j,l);
      for (int i = 0; i < n; ++i) {
        temp_array[j][i] += temp*coeffMatrix1(i,l);
      }
    }
  }

  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      out(j,i) = 0.0;
    }

    for (int l = 0; l < n; ++l) {
      auto temp = coeffMatrix2(j,l);
      for (int i = 0; i < n; ++i) {
        out(j,i) += temp*temp_array[l][i];
      }
    }
  }

//  MatViewType temp("");
//  gemm_nxn<poly_order>('N', 'N', 1.0,  in, coeffMatrix1, 0.0, temp);
//  gemm_nxn<poly_order>('T', 'N', 1.0, coeffMatrix2, temp, 1.0, out);
}
//--------------------------------------------------------------------------
template <int poly_order, typename MatViewType, typename ViewTypeIn, typename ViewTypeOut>
void Dx(
  const  MatViewType nodalDeriv,
  const ViewTypeIn in,
  ViewTypeOut out)
{
  apply_x<poly_order>(nodalDeriv, in, out);
}
//--------------------------------------------------------------------------
template <int poly_order, typename MatViewType, typename ViewTypeIn, typename ViewTypeOut>
void Dy(
  const  MatViewType nodalDeriv,
  const ViewTypeIn in,
  ViewTypeOut out)
{
  apply_y<poly_order>(nodalDeriv, in, out);
}
//--------------------------------------------------------------------------
template <int poly_order, typename MatViewType, typename ViewTypeIn, typename ViewTypeOut>
void I_xhat(
  const  MatViewType scsInterp,
  const ViewTypeIn in,
  ViewTypeOut out)
{
   apply_x<poly_order>(scsInterp, in, out);
}
//--------------------------------------------------------------------------
template <int poly_order, typename MatViewType, typename ViewTypeIn, typename ViewTypeOut>
void I_yhat(
  const  MatViewType scsInterp,
  const ViewTypeIn in,
  ViewTypeOut out)
{
   apply_y<poly_order>(scsInterp, in, out);
}
//--------------------------------------------------------------------------
template <int poly_order, typename MatViewType, typename ViewTypeIn, typename ViewTypeOut>
void Dx_xhat(
  const  MatViewType scsDeriv,
  const ViewTypeIn in,
  ViewTypeOut out)
{
  apply_x<poly_order>(scsDeriv, in, out);
}
//--------------------------------------------------------------------------
template <int poly_order, typename MatViewType,  typename ViewTypeIn, typename ViewTypeOut>
void Dy_xhat(
  const MatViewType scsInterp,
  const MatViewType nodalDeriv,
  const ViewTypeIn in,
  ViewTypeOut out)
{
  MatViewType temp{""};
  apply_x<poly_order>(scsInterp, in, temp);
  apply_y<poly_order>(nodalDeriv, temp, out);
}
//--------------------------------------------------------------------------
template <int poly_order, typename MatViewType, typename ViewTypeIn, typename ViewTypeOut>
void Dx_yhat(
  const MatViewType scsInterp,
  const MatViewType nodalDeriv,
  const ViewTypeIn in,
  ViewTypeOut out)
{
  MatViewType temp{""};
  apply_y<poly_order>(scsInterp, in, temp);
  apply_x<poly_order>(nodalDeriv, temp, out);
}
//--------------------------------------------------------------------------
template <int poly_order, typename MatViewType, typename ViewTypeIn, typename ViewTypeOut>
void Dy_yhat(
  const MatViewType scsDeriv,
  const ViewTypeIn in,
  ViewTypeOut out)
{
  apply_y<poly_order>(scsDeriv,in, out);
}

}
} // namespace nalu
} // namespace Sierra

#endif

