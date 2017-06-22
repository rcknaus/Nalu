/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef HighOrderOperatorsQuad_h
#define HighOrderOperatorsQuad_h

#include <element_promotion/operators/HighOrderOperatorsQuadInternal.h>
#include <element_promotion/operators/CoefficientMatrices.h>
#include <element_promotion/operators/DirectionEnums.h>
#include <element_promotion/CVFEMTypeDefs.h>

namespace sierra {
namespace nalu {

template <int poly_order, stk::topology::topology_t base_topology> class CVFEMOperators {};

template <int poly_order>
class CVFEMOperators<poly_order, stk::topology::QUADRILATERAL_4_2D>
{
public:
  using ElemTraits = AlgTraitsQuad<poly_order>;

  const CoefficientMatrices<poly_order> mat_{};

  void nodal_grad(
    const nodal_scalar_view<ElemTraits> f,
    const nodal_vector_view<ElemTraits> grad) const
  {
    // computes reference-element gradient at nodes
    internal::apply_x<poly_order>(mat_.nodalDeriv, f, Kokkos::subview(grad, 0, Kokkos::ALL, Kokkos::ALL));
    internal::apply_y<poly_order>(mat_.nodalDeriv, f, Kokkos::subview(grad, 1, Kokkos::ALL, Kokkos::ALL));
  }
  //--------------------------------------------------------------------------
  void scs_yhat_grad(
    const nodal_scalar_view<ElemTraits> f,
    nodal_vector_view<ElemTraits> grad)  const
  {
    // computes reference-element at scs of constant yhat coordinate
    internal::Dx_yhat<poly_order>(mat_.scsInterp, mat_.nodalDeriv, f, Kokkos::subview(grad, 0, Kokkos::ALL, Kokkos::ALL));
    internal::Dy_yhat<poly_order>(mat_.scsDeriv, f, Kokkos::subview(grad, 1, Kokkos::ALL, Kokkos::ALL));
  }
  //--------------------------------------------------------------------------
  void scs_xhat_grad(
    const nodal_scalar_view<ElemTraits> f,
    nodal_vector_view<ElemTraits> grad)  const
  {
    // computes reference-element gradient at scs of constant xhat coordinate
    internal::Dx_xhat<poly_order>(mat_.scsDeriv, f, Kokkos::subview(grad, 0, Kokkos::ALL, Kokkos::ALL));
    internal::Dy_xhat<poly_order>(mat_.scsInterp, mat_.nodalDeriv, f, Kokkos::subview(grad, 1, Kokkos::ALL, Kokkos::ALL));
  }
  //--------------------------------------------------------------------------
  void scs_xhat_interp(
    const nodal_scalar_view<ElemTraits> f,
    nodal_scalar_view<ElemTraits> fIp) const
  {
    // interpolates f to the scs of constant xhat coordinate
    internal::apply_x<poly_order>(mat_.scsInterp, f, fIp);
  }
  //--------------------------------------------------------------------------
  void scs_yhat_interp(
    const nodal_scalar_view<ElemTraits> f,
    nodal_scalar_view<ElemTraits> fIp) const
  {
    // interpolates f to the scs of constant yhat coordinate
    internal::apply_y<poly_order>(mat_.scsInterp, f, fIp);
  }
  //--------------------------------------------------------------------------
  void scs_xhat_interp(
    const nodal_vector_view<ElemTraits> f,
    nodal_vector_view<ElemTraits> fIp) const
  {
    // interpolates f to the scs of constant xhat coordinate
    internal::apply_x<poly_order>(mat_.scsInterp, f, Kokkos::subview(fIp, 0, Kokkos::ALL, Kokkos::ALL));
    internal::apply_x<poly_order>(mat_.scsInterp, f, Kokkos::subview(fIp, 1, Kokkos::ALL, Kokkos::ALL));
  }
  //--------------------------------------------------------------------------
  void scs_yhat_interp(
    const nodal_vector_view<ElemTraits> f,
    nodal_vector_view<ElemTraits> fIp) const
  {
    // interpolates f to the scs of constant yhat coordinate
    internal::apply_y<poly_order>(mat_.scsInterp, f, Kokkos::subview(fIp, 0, Kokkos::ALL, Kokkos::ALL));
    internal::apply_y<poly_order>(mat_.scsInterp, f, Kokkos::subview(fIp, 1, Kokkos::ALL, Kokkos::ALL));
  }
  //--------------------------------------------------------------------------
  void volume_xhat(
    const nodal_scalar_view<ElemTraits> f,
    nodal_scalar_view<ElemTraits> f_bar) const
  {
    // integrates along the xhat direction
    internal::apply_x<poly_order>(mat_.nodalWeights, f, f_bar);
  }
  //--------------------------------------------------------------------------
  void volume_yhat(
    const nodal_scalar_view<ElemTraits> f,
    nodal_scalar_view<ElemTraits> f_bar) const
  {
    // integrates along the yhat direction
    internal::apply_y<poly_order>(mat_.nodalWeights, f, f_bar);
  }
  //--------------------------------------------------------------------------
  void volume_2D(
    const nodal_scalar_view<ElemTraits> f,
    nodal_scalar_view<ElemTraits> f_bar) const
  {
    // computes volume integral along 2D volumes (e.g. "scv" in 2D)
    nodal_scalar_view<ElemTraits> temp{""};
    internal::gemm_nxn('N', 'N', 1.0, f, mat_.nodalWeights, 0.0, temp);
    internal::gemm_nxn('T', 'N', 1.0, mat_.nodalWeights, temp, 1.0, f_bar);
  }
  //--------------------------------------------------------------------------
  void scatter_flux_xhat(
    const nodal_scalar_view<ElemTraits> flux,
    nodal_scalar_view<ElemTraits> residual) const
  {
    // +/- fluxes to adjacent left/right nodes

    for (unsigned n = 0; n < poly_order+1; ++n) {
      residual(0,n) -= flux(0,n);
      for (unsigned p = 1; p < poly_order; ++p) {
        residual(p,n) -= flux(p,n) - flux(p-1,n);
      }
      residual(poly_order,n) += flux(poly_order-1,n);
    }
  }
  //--------------------------------------------------------------------------
  void scatter_flux_yhat(
    const nodal_scalar_view<ElemTraits> flux,
    nodal_scalar_view<ElemTraits> residual) const
  {
    // +/- fluxes to adjacent left/right nodes

    for (unsigned m = 0; m < poly_order+1; ++m) {
      residual(m,0) -= flux(m, 0);
      for (unsigned p = 1; p < poly_order; ++p) {
        residual(m,p) -= flux(m, p) - flux(m, p - 1);
      }
      residual(m, poly_order) += flux(m, poly_order-1);
    }
  }
};

template <int p>
using CVFEMOperatorsQuad = CVFEMOperators<p, stk::topology::QUADRILATERAL_4_2D>;

} // namespace naluUnit
} // namespace Sierra

#endif
