/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef HighOrderGeometryQuadDiffusion_h
#define HighOrderGeometryQuadDiffusion_h

#include <element_promotion/operators/CoefficientMatrices.h>
#include <element_promotion/operators/HighOrderOperatorsQuad.h>
#include <element_promotion/operators/DirectionEnums.h>
#include <element_promotion/CVFEMTypeDefs.h>
#include <AlgTraits.h>

#include <stk_util/environment/ReportHandler.hpp>

namespace sierra {
namespace nalu {
namespace high_order_metrics
{
  template <int p, typename Scalar>
  void compute_laplacian_metric_linear(
    const CoefficientMatrices<p, Scalar>& mat,
    const nodal_vector_view<AlgTraitsQuad<p>, Scalar>& coord,
    scs_tensor_view<AlgTraitsQuad<p>, Scalar>& metric)
  {
    enum { E_0 = 0, E_1 = 1, E_2 = 2, E_3 = 3 };

    constexpr int numEdges = 4;
    constexpr int nodesPerEdge = 2;

    // mapped coords
    const Scalar edgev[numEdges][nodesPerEdge] = {
         { coord(XH, p, 0) - coord(XH, 0, 0), coord(YH, p, 0) - coord(YH, 0, 0) },
         { coord(XH, p, p) - coord(XH, p, 0), coord(YH, p, p) - coord(YH, p, 0) },
         { coord(XH, 0, p) - coord(XH, p, p), coord(YH, 0, p) - coord(YH, p, p) },
         { coord(XH, 0, 0) - coord(XH, 0, p), coord(YH, 0, 0) - coord(YH, 0, p) },
    };

    const auto scs_interp = mat.linear_scs_interp;
    const auto nodal_interp = mat.linear_nodal_interp;

    for (int j = 0; j < p; ++j) {
      const Scalar dx_dyh = scs_interp(0,j) * edgev[E_0][XH] - scs_interp(1,j) * edgev[E_2][XH];
      const Scalar dy_dyh = scs_interp(0,j) * edgev[E_0][YH] - scs_interp(1,j) * edgev[E_2][YH];

      const Scalar orth = dx_dyh * dx_dyh + dy_dyh * dy_dyh;
      for (int i = 0; i < p + 1; ++i) {
        const Scalar dx_dxh = -nodal_interp(0,i) * edgev[E_3][XH] + nodal_interp(1,i) * edgev[E_1][XH];
        const Scalar dy_dxh = -nodal_interp(0,i) * edgev[E_3][YH] + nodal_interp(1,i) * edgev[E_1][YH];

        const Scalar fac = Scalar(1.0) / (dx_dyh * dy_dxh - dx_dxh * dy_dyh);
        metric(XH,XH,j,i) =  fac * orth;
        metric(XH,YH,j,i) = -fac * (dx_dxh * dx_dyh + dy_dxh * dy_dyh);
      }
    }

    for (int j = 0; j < p; ++j) {
      const Scalar dx_dxh = -scs_interp(0,j) * edgev[E_3][XH] + scs_interp(1,j) * edgev[E_1][XH];
      const Scalar dy_dxh = -scs_interp(0,j) * edgev[E_3][YH] + scs_interp(1,j) * edgev[E_1][YH];

      const Scalar orth = dx_dxh * dx_dxh + dy_dxh * dy_dxh;
      for (int i = 0; i < p + 1; ++i) {
        const Scalar dx_dyh = nodal_interp(0,i) * edgev[E_0][XH] - nodal_interp(1,i) * edgev[E_2][XH];
        const Scalar dy_dyh = nodal_interp(0,i) * edgev[E_0][YH] - nodal_interp(1,i) * edgev[E_2][YH];

        const Scalar fac = Scalar(1.0) / (dx_dyh * dy_dxh - dx_dxh * dy_dyh);
        metric(YH,XH,j,i) = -fac * (dx_dxh * dx_dyh + dy_dxh * dy_dyh);
        metric(YH,YH,j,i) =  fac * orth;
      }
    }
  }

  template <int p, typename Scalar>
  void compute_diffusion_metric_linear(
    const CVFEMOperatorsQuad<p, Scalar>& ops,
    const nodal_vector_view<AlgTraitsQuad<p>, Scalar>& coord,
    const nodal_scalar_view<AlgTraitsQuad<p>, Scalar>& diffusivity,
    scs_tensor_view<AlgTraitsQuad<p>, Scalar>& metric)
  {
    enum { E_0 = 0, E_1 = 1, E_2 = 2, E_3 = 3 };

    constexpr int numEdges = 4;
    constexpr int nodesPerEdge = 2;

    // mapped coords
    const Scalar edgev[numEdges][nodesPerEdge] = {
         { coord(XH, p, 0) - coord(XH, 0, 0), coord(YH, p, 0) - coord(YH, 0, 0) },
         { coord(XH, p, p) - coord(XH, p, 0), coord(YH, p, p) - coord(YH, p, 0) },
         { coord(XH, 0, p) - coord(XH, p, p), coord(YH, 0, p) - coord(YH, p, p) },
         { coord(XH, 0, 0) - coord(XH, 0, p), coord(YH, 0, 0) - coord(YH, 0, p) }
    };

    const auto scs_interp = ops.mat_.linear_scs_interp;
    const auto nodal_interp = ops.mat_.linear_nodal_interp;

    nodal_scalar_view<AlgTraitsQuad<p>, Scalar> diffIp{""};
    ops.scs_xhat_interp(diffusivity, diffIp);

    for (int j = 0; j < p; ++j) {
      const Scalar dx_dyh = scs_interp(0,j) * edgev[E_0][XH] - scs_interp(1,j) * edgev[E_2][XH];
      const Scalar dy_dyh = scs_interp(0,j) * edgev[E_0][YH] - scs_interp(1,j) * edgev[E_2][YH];

      const Scalar orth = dx_dyh * dx_dyh + dy_dyh * dy_dyh;
      for (int i = 0; i < p + 1; ++i) {
        const Scalar dx_dxh = -nodal_interp(0,i) * edgev[E_3][XH] + nodal_interp(1,i) * edgev[E_1][XH];
        const Scalar dy_dxh = -nodal_interp(0,i) * edgev[E_3][YH] + nodal_interp(1,i) * edgev[E_1][YH];

        const Scalar fac = diffIp(i,j) / (dx_dyh * dy_dxh - dx_dxh * dy_dyh);
        metric(XH,XH,j,i) =  fac * orth;
        metric(XH,YH,j,i) = -fac * (dx_dxh * dx_dyh + dy_dxh * dy_dyh);
      }
    }

    ops.scs_yhat_interp(diffusivity, diffIp);

    for (int j = 0; j < p; ++j) {
      const Scalar dx_dxh = -scs_interp(0,j) * edgev[E_3][XH] + scs_interp(1,j) * edgev[E_1][XH];
      const Scalar dy_dxh = -scs_interp(0,j) * edgev[E_3][YH] + scs_interp(1,j) * edgev[E_1][YH];

      const Scalar orth = dx_dxh * dx_dxh + dy_dxh * dy_dxh;
      for (int i = 0; i < p + 1; ++i) {
        const Scalar dx_dyh = nodal_interp(0,i) * edgev[E_0][XH] - nodal_interp(1,i) * edgev[E_2][XH];
        const Scalar dy_dyh = nodal_interp(0,i) * edgev[E_0][YH] - nodal_interp(1,i) * edgev[E_2][YH];

        const Scalar fac = diffIp(j,i) / (dx_dyh * dy_dxh - dx_dxh * dy_dyh);
        metric(YH,XH,j,i) = -fac * (dx_dxh * dx_dyh + dy_dxh * dy_dyh);
        metric(YH,YH,j,i) =  fac * orth;
      }
    }
  }

  // need a symmetric version tensor too, e.g. for the NSO

} // namespace HighOrderGeometryQuad
} // namespace naluUnit
} // namespace Sierra

#endif
