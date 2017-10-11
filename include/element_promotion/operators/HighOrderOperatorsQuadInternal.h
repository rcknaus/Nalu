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

inline Teuchos::ETransp char_to_teuchos_enum(char x) {
  return (x == 'N') ? Teuchos::NO_TRANS : Teuchos::TRANS;
}

template <int poly_order, typename ViewTypeA, typename ViewTypeB, typename ViewTypeC>
void gemm_nxn(
  char transA,
  char transB,
  double alpha,
  const ViewTypeA& A,
  const ViewTypeB& B,
  double beta,
  ViewTypeC& C)
{
  // Kokkos-Kernels gemm currently has a message, "do not use"
  static_assert(
    std::is_same<typename ViewTypeA::value_type,
                 typename ViewTypeB::value_type>::value,
      "Types don't match");

  Tpetra::Details::Blas::gemm(transA, transB, alpha, A, B, beta, C);
}
//--------------------------------------------------------------------------
template <int poly_order, typename ViewTypeIn, typename ViewTypeOut>
void apply_x(
  const nodal_matrix_view<poly_order> coeffMatrix,
  const ViewTypeIn in,
  ViewTypeOut out)
{
  gemm_nxn<poly_order>('T', 'N', 1.0, coeffMatrix, in, 0.0, out);
}
//--------------------------------------------------------------------------
template <int poly_order, typename ViewTypeIn, typename ViewTypeOut>
void apply_y(
  const nodal_matrix_view<poly_order> coeffMatrix,
  const ViewTypeIn in,
  ViewTypeOut out)
{
  gemm_nxn('N', 'N', 1.0,  in, coeffMatrix, 0.0, out);
}
//--------------------------------------------------------------------------
template <int poly_order, typename ViewType>
void apply_yx(
  const nodal_matrix_view<poly_order> coeffMatrix1,
  const nodal_matrix_view<poly_order> coeffMatrix2,
  const nodal_matrix_view<poly_order> in,
  nodal_matrix_view<poly_order> out)
{
  nodal_matrix_view<poly_order> temp("");
  gemm_nxn('N', 'N', 1.0,  in, coeffMatrix1, 0.0, temp);
  gemm_nxn('T', 'N', 1.0, coeffMatrix2, temp, 1.0, out);
}
//--------------------------------------------------------------------------
template <int poly_order>
void apply_xy(
  const nodal_matrix_view<poly_order> coeffMatrix1,
  const nodal_matrix_view<poly_order> coeffMatrix2,
  const nodal_matrix_view<poly_order> in,
  nodal_matrix_view<poly_order> out)
{
  nodal_matrix_view<poly_order> temp("");
  gemm_nxn('N', 'N', 1.0,  in, coeffMatrix1, 0.0, temp);
  gemm_nxn('T', 'N', 1.0, coeffMatrix2, temp, 1.0, out);
}
//--------------------------------------------------------------------------
template <int poly_order, typename ViewTypeIn, typename ViewTypeOut>
void Dx(
  const nodal_matrix_view<poly_order> nodalDeriv,
  const ViewTypeIn in,
  ViewTypeOut out)
{
  apply_x<poly_order>(nodalDeriv, in, out);
}
//--------------------------------------------------------------------------
template <int poly_order, typename ViewTypeIn, typename ViewTypeOut>
void Dy(
  const nodal_matrix_view<poly_order> nodalDeriv,
  const ViewTypeIn in,
  ViewTypeOut out)
{
  apply_y<poly_order>(nodalDeriv, in, out);
}
//--------------------------------------------------------------------------
template <int poly_order, typename ViewTypeIn, typename ViewTypeOut>
void I_xhat(
  const nodal_matrix_view<poly_order> scsInterp,
  const ViewTypeIn in,
  ViewTypeOut out)
{
   apply_x<poly_order>(scsInterp, in, out);
}
//--------------------------------------------------------------------------
template <int poly_order, typename ViewTypeIn, typename ViewTypeOut>
void I_yhat(
  const nodal_matrix_view<poly_order> scsInterp,
  const ViewTypeIn in,
  ViewTypeOut out)
{
   apply_y<poly_order>(scsInterp, in, out);
}
//--------------------------------------------------------------------------
template <int poly_order, typename ViewTypeIn, typename ViewTypeOut>
void Dx_xhat(
  const nodal_matrix_view<poly_order> scsDeriv,
  const ViewTypeIn in,
  ViewTypeOut out)
{
  apply_x<poly_order>(scsDeriv, in, out);
}
//--------------------------------------------------------------------------
template <int poly_order, typename ViewTypeIn, typename ViewTypeOut>
void Dy_xhat(
  const nodal_matrix_view<poly_order> scsInterp,
  const nodal_matrix_view<poly_order> nodalDeriv,
  const ViewTypeIn in,
  ViewTypeOut out)
{
  nodal_matrix_view<poly_order> temp{""};
  apply_x<poly_order>(scsInterp, in, temp);
  apply_y<poly_order>(nodalDeriv, temp, out);
}
//--------------------------------------------------------------------------
template <int poly_order, typename ViewTypeIn, typename ViewTypeOut>
void Dx_yhat(
  const nodal_matrix_view<poly_order> scsInterp,
  const nodal_matrix_view<poly_order> nodalDeriv,
  const ViewTypeIn in,
  ViewTypeOut out)
{
  nodal_matrix_view<poly_order> temp{""};
  apply_y<poly_order>(scsInterp, in, temp);
  apply_x<poly_order>(nodalDeriv, temp, out);
}
//--------------------------------------------------------------------------
template <int poly_order, typename ViewTypeIn, typename ViewTypeOut>
void Dy_yhat(
  const nodal_matrix_view<poly_order> scsDeriv,
  const ViewTypeIn in,
  ViewTypeOut out)
{
  apply_y<poly_order>(scsDeriv,in, out);
}

}
} // namespace nalu
} // namespace Sierra

#endif

