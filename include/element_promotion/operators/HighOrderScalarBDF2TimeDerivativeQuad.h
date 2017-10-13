/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef HighOrderScalarBDF2TimeDerivativeQuad_h
#define HighOrderScalarBDF2TimeDerivativeQuad_h

#include <element_promotion/operators/HighOrderOperatorsQuad.h>
#include <element_promotion/operators/CoefficientMatrices.h>
#include <element_promotion/operators/DirectionEnums.h>
#include <element_promotion/CVFEMTypeDefs.h>

namespace sierra {
namespace nalu {
namespace tensor_assembly {

template <int nodes1D> int idx(int i, int j) { return i*nodes1D+j; };

template <int poly_order, typename Scalar>
void elemental_time_derivative_jacobian(
  const CVFEMOperatorsQuad<poly_order, Scalar>& ops,
  const nodal_scalar_view<AlgTraitsQuad<poly_order>, Scalar>& metric,
  double gamma1,
  const nodal_scalar_view<AlgTraitsQuad<poly_order>, Scalar>& rho_p1,
  matrix_view<AlgTraitsQuad<poly_order>, Scalar>& lhs)
{
  constexpr int n1D = AlgTraitsQuad<poly_order>::nodes1D_;
  const auto& weight = ops.mat_.nodalWeights;

  for (int n = 0; n < n1D; ++n) {
    for (int m = 0; m < n1D; ++m) {
      for (int j = 0; j < n1D; ++j) {
        auto wnj = gamma1 * weight(n,j);
        for (int i = 0; i < n1D; ++i) {
          lhs(idx<n1D>(n,m), idx<n1D>(j,i)) += wnj * weight(m,i) * metric(j,i) * rho_p1(j,i);
        }
      }
    }
  }
}
//--------------------------------------------------------------------------
template <int poly_order, typename Scalar>
void elemental_time_derivative_action(
  const CVFEMOperatorsQuad<poly_order, Scalar>& ops,
  const nodal_scalar_view<AlgTraitsQuad<poly_order>, Scalar>& metric,
  double gamma1,
  double gamma2,
  double gamma3,
  const nodal_scalar_view<AlgTraitsQuad<poly_order>, Scalar>& rho_m1,
  const nodal_scalar_view<AlgTraitsQuad<poly_order>, Scalar>& rho_p0,
  const nodal_scalar_view<AlgTraitsQuad<poly_order>, Scalar>& rho_p1,
  const nodal_scalar_view<AlgTraitsQuad<poly_order>, Scalar>& q_m1,
  const nodal_scalar_view<AlgTraitsQuad<poly_order>, Scalar>& q_p0,
  const nodal_scalar_view<AlgTraitsQuad<poly_order>, Scalar>& q_p1,
  nodal_scalar_view<AlgTraitsQuad<poly_order>, Scalar>& dqdt,
  nodal_scalar_view<AlgTraitsQuad<poly_order>, Scalar>& rhs)
{
  constexpr int n1D = AlgTraitsQuad<poly_order>::nodes1D_;
  for (int j = 0; j < n1D; ++j) {
    for (int i = 0; i < n1D; ++i) {
      dqdt(j,i) = -(
          gamma1 * rho_p1(j,i) * metric(j,i) * q_p1(j,i)
        + gamma2 * rho_p0(j,i) * metric(j,i) * q_p0(j,i)
        + gamma3 * rho_m1(j,i) * metric(j,i) * q_m1(j,i)
      );
    }
  }
  ops.volume_2D(dqdt, rhs);
}

} // namespace HighOrderLaplacianQuad
} // namespace naluUnit
} // namespace Sierra

#endif
