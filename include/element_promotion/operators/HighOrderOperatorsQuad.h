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

template <int poly_order, typename Scalar, stk::topology::topology_t base_topology> class CVFEMOperators {};

template <int poly_order, typename Scalar>
class CVFEMOperators<poly_order, Scalar, stk::topology::QUADRILATERAL_4_2D>
{
public:
  using AlgTraits = AlgTraitsQuad<poly_order>;

  const CoefficientMatrices<poly_order, Scalar> mat_{};

  void nodal_grad(
    const nodal_scalar_view<AlgTraits, Scalar>& f,
    nodal_vector_view<AlgTraits, Scalar>& grad) const
  {
    // computes reference-element gradient at nodes
    internal::apply_x<poly_order>(mat_.nodalDeriv, f, grad, XH);
    internal::apply_y<poly_order>(mat_.nodalDeriv, f, grad, YH);
  }
  //--------------------------------------------------------------------------
  void scs_yhat_grad(
    const nodal_scalar_view<AlgTraits, Scalar>& f,
    nodal_vector_view<AlgTraits, Scalar>& grad) const
  {
    // computes reference-element at scs of constant yhat coordinate
    internal::apply_yx<poly_order>(mat_.scsInterp, mat_.nodalDeriv, f, grad, XH);
    internal::apply_y<poly_order>(mat_.scsDeriv, f, grad, YH);
  }
  //--------------------------------------------------------------------------
  void scs_xhat_grad(
    const nodal_scalar_view<AlgTraits, Scalar>& f,
    nodal_vector_view<AlgTraits, Scalar>& grad)  const
  {
    // computes reference-element gradient at scs of constant xhat coordinate
    internal::apply_x<poly_order>(mat_.scsDeriv, f, grad, XH);
    internal::apply_xy<poly_order>(mat_.scsInterp, mat_.nodalDeriv, f, grad, YH);
  }
  //--------------------------------------------------------------------------
  void scs_xhat_interp(
    const nodal_scalar_view<AlgTraits, Scalar>& f,
    nodal_scalar_view<AlgTraits, Scalar>& fIp) const
  {
    // interpolates f to the scs of constant xhat coordinate
    internal::apply_x<poly_order>(mat_.scsInterp, f, fIp);
  }
  //--------------------------------------------------------------------------
  void scs_yhat_interp(
    const nodal_scalar_view<AlgTraits, Scalar>& f,
    nodal_scalar_view<AlgTraits, Scalar>& fIp) const
  {
    // interpolates f to the scs of constant yhat coordinate
    internal::apply_y<poly_order>(mat_.scsInterp, f, fIp);
  }
  //--------------------------------------------------------------------------
  void scs_xhat_interp(
    const nodal_vector_view<AlgTraits, Scalar>& f,
    nodal_vector_view<AlgTraits, Scalar>& fIp) const
  {
    // interpolates f to the scs of constant xhat coordinate
    internal::apply_x<poly_order>(mat_.scsInterp, f, XH, fIp, XH);
    internal::apply_x<poly_order>(mat_.scsInterp, f, YH, fIp, YH);
  }
  //--------------------------------------------------------------------------
  void scs_yhat_interp(
    const nodal_vector_view<AlgTraits, Scalar>& f,
    nodal_vector_view<AlgTraits, Scalar>& fIp) const
  {
    // interpolates f to the scs of constant yhat coordinate
    internal::apply_y<poly_order>(mat_.scsInterp, f, XH, fIp, XH);
    internal::apply_y<poly_order>(mat_.scsInterp, f, YH, fIp, YH);
  }
  //--------------------------------------------------------------------------
  void volume_xhat(
    const nodal_scalar_view<AlgTraits, Scalar>& f,
    nodal_scalar_view<AlgTraits, Scalar>& f_bar) const
  {
    // integrates along the xhat direction
    internal::apply_x<poly_order>(mat_.nodalWeights, f, f_bar);
  }
  //--------------------------------------------------------------------------
  void volume_yhat(
    const nodal_scalar_view<AlgTraits, Scalar>& f,
    nodal_scalar_view<AlgTraits, Scalar>& f_bar) const
  {
    // integrates along the yhat direction
    internal::apply_y<poly_order>(mat_.nodalWeights, f, f_bar);
  }
  //--------------------------------------------------------------------------
  void volume_2D(
    const nodal_scalar_view<AlgTraits, Scalar>& f,
    nodal_scalar_view<AlgTraits, Scalar>& f_bar) const
  {
    // integrates volumetrically
    internal::apply_xy<poly_order>(mat_.nodalWeights, mat_.nodalWeights, f, f_bar);
  }
  //--------------------------------------------------------------------------
  void integrate_and_scatter_xhat(
    const nodal_scalar_view<AlgTraits, Scalar>& f,
    nodal_scalar_view<AlgTraits, Scalar>& f_bar) const
  {
    // integrates along the xhat direction and differences the resulting flux
    internal::apply_x_and_scatter<poly_order>(mat_.nodalWeights, f, f_bar);
  }
  //--------------------------------------------------------------------------
  void integrate_and_scatter_yhat(
    const nodal_scalar_view<AlgTraits, Scalar>& f,
    nodal_scalar_view<AlgTraits, Scalar>& f_bar) const
  {
    // integrates along the yhat direction and differences the resulting flux
    internal::apply_y_and_scatter<poly_order>(mat_.nodalWeights, f, f_bar);
  }
};

template <int p, typename  Scalar = DoubleType>
using CVFEMOperatorsQuad = CVFEMOperators<p, Scalar, stk::topology::QUADRILATERAL_4_2D>;

} // namespace naluUnit
} // namespace Sierra

#endif

