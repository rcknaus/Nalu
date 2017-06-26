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

template <int nodes1D> int idx(int i, int j) { return (i * nodes1D + j); };

template <int poly_order>
void elemental_advection_diffusion_jacobian(
  const CVFEMOperatorsQuad<poly_order> ops,
  const scs_vector_view<AlgTraitsQuad<poly_order>> adv_metric,
  const scs_tensor_view<AlgTraitsQuad<poly_order>> diff_metric,
  matrix_view<AlgTraitsQuad<poly_order>> lhs)
{
  using ftype = typename matrix_view<AlgTraitsQuad<poly_order>>::value_type;

  constexpr int n1D = AlgTraitsQuad<poly_order>::nodes1D_;
  const auto& mat = ops.mat_;

  // flux past constant yhat lines
  for (int n = 0; n < n1D; ++n) {
    // x- element boundary
    constexpr int m_minus = 0;
    for (int j = 0; j < n1D; ++j) {
      ftype difforth = mat.nodalWeights(n, j) * diff_metric(XH, XH, m_minus, j);
      ftype interp_term = mat.nodalWeights(n, j) * adv_metric(XH, m_minus, j);
      for (int k = 0; k < n1D; ++k) {
        interp_term += mat.nodalWeights(n, k) * mat.nodalDeriv(k, j) * diff_metric(XH, YH, m_minus, k);
      }

      for (int i = 0; i < n1D; ++i) {
        lhs(idx<n1D>(n, m_minus), idx<n1D>(j, i)) +=
            difforth * mat.scsDeriv(m_minus, i) + interp_term * mat.scsInterp(m_minus, i);
      }
    }

    // interior flux
    for (int m = 1; m < n1D - 1; ++m) {
      for (int j = 0; j < n1D; ++j) {
        const ftype w = mat.nodalWeights(n, j);
        const ftype difforthm1 = w * diff_metric(XH, XH, m - 1, j);
        const ftype difforthp0 = w * diff_metric(XH, XH, m + 0, j);

        ftype interp_term_m1 = w * adv_metric(XH, m - 1, j);
        ftype interp_term_p0 = w * adv_metric(XH, m + 0, j);
        for (int k = 0; k < n1D; ++k) {
          const ftype wd = mat.nodalWeights(n, k) * mat.nodalDeriv(k, j);
          interp_term_m1 += wd * diff_metric(XH, YH, m - 1, k);
          interp_term_p0 += wd * diff_metric(XH, YH, m + 0, k);
        }

        for (int i = 0; i < n1D; ++i) {
          const ftype fm = difforthm1 * mat.scsDeriv(m - 1, i) + interp_term_m1 * mat.scsInterp(m - 1, i);
          const ftype fp = difforthp0 * mat.scsDeriv(m + 0, i) + interp_term_p0 * mat.scsInterp(m + 0, i);
          lhs(idx<n1D>(n, m), idx<n1D>(j, i)) += (fp - fm);
        }
      }
    }

    // x+ element boundary
    constexpr int m_plus = n1D - 1;
    for (int j = 0; j < n1D; ++j) {
      const ftype difforth = mat.nodalWeights(n, j) * diff_metric(XH, XH, m_plus - 1, j);
      ftype interp_term = mat.nodalWeights(n,j) * adv_metric(XH, m_plus-1, j);
      for (int k = 0; k < n1D; ++k) {
        interp_term += mat.nodalWeights(n, k) * mat.nodalDeriv(k, j) * diff_metric(XH, YH, m_plus - 1, k);
      }
      for (int i = 0; i < n1D; ++i) {
        lhs(idx<n1D>(n, m_plus), idx<n1D>(j, i)) -=
            difforth * mat.scsDeriv(m_plus - 1, i) + interp_term * mat.scsInterp(m_plus - 1, i);
      }
    }
  }

  // flux past constant xhat lines
  for (int m = 0; m < n1D; ++m) {
    // y- boundary
    constexpr int n_minus = 0;
    for (int i = 0; i < n1D; ++i) {
      const ftype difforth = mat.nodalWeights(m, i) * diff_metric(YH, YH, n_minus, i);
      ftype interp_term = mat.nodalWeights(m, i) * adv_metric(YH, n_minus, i);
      for (int k = 0; k < n1D; ++k) {
        interp_term += mat.nodalWeights(m, k) * mat.nodalDeriv(k, i) * diff_metric(YH, XH, n_minus, k);
      }
      for (int j = 0; j < n1D; ++j) {
        lhs(idx<n1D>(n_minus, m), idx<n1D>(j, i)) +=
            difforth * mat.scsDeriv(n_minus, j) + interp_term * mat.scsInterp(n_minus, j);
      }
    }

    // interior flux
    for (int n = 1; n < n1D - 1; ++n) {
      for (int i = 0; i < n1D; ++i) {
        const ftype w = mat.nodalWeights(m, i);
        const ftype difforthm1 = w * diff_metric(YH, YH, n - 1, i);
        const ftype difforthp0 = w * diff_metric(YH, YH, n + 0, i);

        ftype interp_term_m1 = w * adv_metric(YH, n - 1, i);
        ftype interp_term_p0 = w * adv_metric(YH, n + 0, i);
        for (int k = 0; k < n1D; ++k) {
          const ftype wd = mat.nodalWeights(m, k) * mat.nodalDeriv(k, i);
          interp_term_m1 += wd * diff_metric(YH, XH, n - 1, k);
          interp_term_p0 += wd * diff_metric(YH, XH, n + 0, k);
        }

        for (int j = 0; j < n1D; ++j) {
          const ftype fm = difforthm1 * mat.scsDeriv(n - 1, j) + interp_term_m1 * mat.scsInterp(n - 1, j);
          const ftype fp = difforthp0 * mat.scsDeriv(n + 0, j) + interp_term_p0 * mat.scsInterp(n + 0, j);
          lhs(idx<n1D>(n, m), idx<n1D>(j, i)) += (fp - fm);
        }
      }
    }

    // y+ boundary
    constexpr int n_plus = n1D - 1;
    for (int i = 0; i < n1D; ++i) {
      const ftype difforth = mat.nodalWeights(m, i) * diff_metric(YH, YH, n_plus - 1, i);
      ftype interp_term = mat.nodalWeights(m, i) * adv_metric(YH, n_plus - 1, i);
      for (int k = 0; k < n1D; ++k) {
        interp_term += mat.nodalWeights(m, k) * mat.nodalDeriv(k, i) * diff_metric(YH, XH, n_plus - 1, k);
      }

      for (int j = 0; j < n1D; ++j) {
        lhs(idx<n1D>(n_plus, m), idx<n1D>(j, i)) -=
            difforth * mat.scsDeriv(n_plus - 1, j) + interp_term * mat.scsInterp(n_plus - 1, j);
      }
    }
  }
}
//--------------------------------------------------------------------------
template <int poly_order>
void elemental_advection_diffusion_action(
  const CVFEMOperatorsQuad<poly_order> ops,
  const scs_vector_view<AlgTraitsQuad<poly_order>> adv_metric,
  const scs_tensor_view<AlgTraitsQuad<poly_order>> diff_metric,
  const nodal_scalar_view<AlgTraitsQuad<poly_order>> scalar,
  nodal_scalar_view<AlgTraitsQuad<poly_order>> rhs
)
{
  using ElemTraits = AlgTraitsQuad<poly_order>;

  constexpr int n1D = ElemTraits::nodes1D_;
  constexpr int nscs = ElemTraits::nscs_;

  nodal_scalar_view<ElemTraits> integrand{""};
  nodal_vector_view<ElemTraits> grad_phi{""};
  nodal_scalar_view<ElemTraits> flux{""};

  ops.scs_xhat_grad(scalar, grad_phi);
  for (int k = 0; k < nscs; ++k) {
    for (int n = 0; n < n1D; ++n) {
      integrand(n,k) = diff_metric(XH,XH, k, n) * grad_phi(XH, n, k) + diff_metric(XH,YH, k, n) * grad_phi(YH, n, k);
    }
  }
  ops.volume_yhat(integrand, flux);
  ops.scatter_flux_yhat(flux, rhs);

  ops.scs_yhat_grad(scalar, grad_phi);
  for (int q = 0; q < nscs; ++q) {
    for (int n = 0; n < n1D; ++n) {
      integrand(q, n) = diff_metric(YH,XH, q, n) * grad_phi(XH, q, n) + diff_metric(YH,YH, q, n) * grad_phi(YH, q, n);
    }
  }
  ops.volume_xhat(integrand, flux);
  ops.scatter_flux_xhat(flux, rhs);
}

} // namespace HighOrderLaplacianQuad
} // namespace naluUnit
} // namespace Sierra

#endif
