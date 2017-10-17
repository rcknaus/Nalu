/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef HighOrderLaplacianQuad_h
#define HighOrderLaplacianQuad_h

#include <element_promotion/operators/HighOrderOperatorsQuad.h>
#include <element_promotion/operators/CoefficientMatrices.h>
#include <element_promotion/operators/DirectionEnums.h>
#include <element_promotion/CVFEMTypeDefs.h>

namespace sierra {
namespace nalu {
namespace tensor_assembly {

template <int nodes1D> int idx(int i, int j) { return i*nodes1D+j; };

template <int poly_order, typename Scalar>
void elemental_diffusion_jacobian(
  const CVFEMOperators<poly_order, Scalar, stk::topology::QUAD_4_2D>& ops,
  const scs_tensor_view<AlgTraitsQuad<poly_order>, Scalar>& metric,
  matrix_view<AlgTraitsQuad<poly_order>, Scalar>& lhs)
{
  constexpr int n1D = AlgTraitsQuad<poly_order>::nodes1D_;
  const auto& mat = ops.mat_;

  // flux past constant yhat lines
  for (int n = 0; n < n1D; ++n) {
    // x- element boundary
    constexpr int m_minus = 0;
    for (int j = 0; j < n1D; ++j) {

      Scalar non_orth = 0.0;
      for (int k = 0; k < n1D; ++k) {
        non_orth += mat.nodalWeights(n, k) * mat.nodalDeriv(k, j) * metric(XH, YH, m_minus, k);
      }

      Scalar orth = mat.nodalWeights(n, j) * metric(XH, XH, m_minus, j);
      for (int i = 0; i < n1D; ++i) {
        lhs(idx<n1D>(n, m_minus), idx<n1D>(j, i)) +=
            orth * mat.scsDeriv(m_minus, i) + non_orth * mat.scsInterp(m_minus, i);
      }
    }

    // interior flux
    for (int m = 1; m < n1D - 1; ++m) {
      for (int j = 0; j < n1D; ++j) {
        const Scalar w = mat.nodalWeights(n, j);
        const Scalar orthm1 = w * metric(XH, XH, m - 1, j);
        const Scalar orthp0 = w * metric(XH, XH, m + 0, j);

        Scalar non_orthp0 = 0.0;
        Scalar non_orthm1 = 0.0;
        for (int k = 0; k < n1D; ++k) {
          const Scalar wd = mat.nodalWeights(n, k) * mat.nodalDeriv(k, j);
          non_orthm1 += wd * metric(XH, YH, m - 1, k);
          non_orthp0 += wd * metric(XH, YH, m + 0, k);
        }

        for (int i = 0; i < n1D; ++i) {
          const Scalar fm = orthm1 * mat.scsDeriv(m - 1, i) + non_orthm1 * mat.scsInterp(m - 1, i);
          const Scalar fp = orthp0 * mat.scsDeriv(m + 0, i) + non_orthp0 * mat.scsInterp(m + 0, i);
          lhs(idx<n1D>(n, m), idx<n1D>(j, i)) += (fp - fm);
        }
      }
    }

    // x+ element boundary
    constexpr int m_plus = n1D - 1;
    for (int j = 0; j < n1D; ++j) {
      Scalar non_orth = 0.0;
      for (int k = 0; k < n1D; ++k) {
        non_orth += mat.nodalWeights(n, k) * mat.nodalDeriv(k, j) * metric(XH, YH, m_plus - 1, k);
      }

      const Scalar orth = mat.nodalWeights(n, j) * metric(XH, XH, m_plus - 1, j);
      for (int i = 0; i < n1D; ++i) {
        lhs(idx<n1D>(n, m_plus), idx<n1D>(j, i)) -=
            orth * mat.scsDeriv(m_plus - 1, i) + non_orth * mat.scsInterp(m_plus - 1, i);
      }
    }
  }

  // flux past constant xhat lines
  for (int m = 0; m < n1D; ++m) {
    // y- boundary
    constexpr int n_minus = 0;
    for (int i = 0; i < n1D; ++i) {
      Scalar non_orth = 0.0;
      for (int k = 0; k < n1D; ++k) {
        non_orth += mat.nodalWeights(m, k) * mat.nodalDeriv(k, i) * metric(YH, XH, n_minus, k);
      }

      const Scalar orth = mat.nodalWeights(m, i) * metric(YH, YH, n_minus, i);
      for (int j = 0; j < n1D; ++j) {
        lhs(idx<n1D>(n_minus, m), idx<n1D>(j, i)) +=
            orth * mat.scsDeriv(n_minus, j) + non_orth * mat.scsInterp(n_minus, j);
      }
    }

    // interior flux
    for (int n = 1; n < n1D - 1; ++n) {
      for (int i = 0; i < n1D; ++i) {
        const Scalar w = mat.nodalWeights(m, i);
        const Scalar orthm1 = w * metric(YH, YH, n - 1, i);
        const Scalar orthp0 = w * metric(YH, YH, n + 0, i);

        Scalar non_orthp0 = 0.0;
        Scalar non_orthm1 = 0.0;
        for (int k = 0; k < n1D; ++k) {
          const Scalar wd = mat.nodalWeights(m, k) * mat.nodalDeriv(k, i);
          non_orthm1 += wd * metric(YH, XH, n - 1, k);
          non_orthp0 += wd * metric(YH, XH, n + 0, k);
        }

        for (int j = 0; j < n1D; ++j) {
          const Scalar fm = orthm1 * mat.scsDeriv(n - 1, j) + non_orthm1 * mat.scsInterp(n - 1, j);
          const Scalar fp = orthp0 * mat.scsDeriv(n + 0, j) + non_orthp0 * mat.scsInterp(n + 0, j);
          lhs(idx<n1D>(n, m), idx<n1D>(j, i)) += (fp - fm);
        }
      }
    }

    // y+ boundary
    constexpr int n_plus = n1D - 1;
    for (int i = 0; i < n1D; ++i) {
      Scalar non_orth = 0.0;
      for (int k = 0; k < n1D; ++k) {
        non_orth += mat.nodalWeights(m, k) * mat.nodalDeriv(k, i) * metric(YH, XH, n_plus - 1, k);
      }

      const Scalar orth = mat.nodalWeights(m, i) * metric(YH, YH, n_plus - 1, i);
      for (int j = 0; j < n1D; ++j) {
        lhs(idx<n1D>(n_plus, m), idx<n1D>(j, i)) -=
            orth * mat.scsDeriv(n_plus - 1, j) + non_orth * mat.scsInterp(n_plus - 1, j);
      }
    }
  }
}
//--------------------------------------------------------------------------
template <int poly_order, typename Scalar>
void elemental_diffusion_action(
  CVFEMOperators<poly_order, Scalar, stk::topology::QUAD_4_2D>& ops,
  const scs_tensor_view<AlgTraitsQuad<poly_order>, Scalar>& metric,
  const nodal_scalar_view<AlgTraitsQuad<poly_order>, Scalar>& scalar,
  nodal_scalar_view<AlgTraitsQuad<poly_order>, Scalar>& rhs)
{
  using AlgTraits = AlgTraitsQuad<poly_order>;
  
  constexpr int n1D = AlgTraitsQuad<poly_order>::nodes1D_;
  constexpr int nscs = AlgTraitsQuad<poly_order>::nscs_;

  Scalar integrand[AlgTraits::nodes1D_*AlgTraits::nodes1D_];
  nodal_scalar_view<AlgTraits,Scalar> v_integrand(integrand);
  Kokkos::deep_copy(v_integrand, 0.0);

  Scalar grad_phi[AlgTraits::nDim_ * AlgTraits::nodes1D_*AlgTraits::nodes1D_];
  nodal_vector_view<AlgTraits,Scalar> v_grad_phi(grad_phi);
  Kokkos::deep_copy(v_grad_phi, 0.0);

  ops.scs_xhat_grad(scalar, v_grad_phi);
  for (int k = 0; k < nscs; ++k) {
    for (int n = 0; n < n1D; ++n) {
      v_integrand(n,k) = metric(XH,XH, k, n) * v_grad_phi(XH, n, k) + metric(XH,YH, k, n) * v_grad_phi(YH, n, k);
    }
  }
  ops.integrate_and_scatter_yhat(v_integrand, rhs);

  ops.scs_yhat_grad(scalar, v_grad_phi);
  for (int k = 0; k < nscs; ++k) {
    for (int n = 0; n < n1D; ++n) {
      v_integrand(k, n) = metric(YH,XH, k, n) * v_grad_phi(XH, k, n) + metric(YH,YH, k, n) * v_grad_phi(YH, k, n);
    }
  }
  ops.integrate_and_scatter_xhat(v_integrand, rhs);
}

} // namespace HighOrderLaplacianQuad
} // namespace naluUnit
} // namespace Sierra

#endif
